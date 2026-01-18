import json
import time
import random
import difflib
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import playergamelog, teamgamelog


# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

# Streamlit Cloud stability
API_TIMEOUT_SECONDS = 28
GAMELOG_RETRIES = 3
BUILD_THROTTLE_SECONDS = 0.05

# Old heuristic bump engine (fallback, always fast)
MIN_BENCH_FLOOR = 6.0
MAX_MINUTES_PLAYER = 40.0
MAX_MIN_INCREASE = 22.0
OFF_BUMP_STRENGTH = 0.22
OFF_BUMP_CAP = 1.25
OFF_BUMP_TOP_N = 6

# Absence profile (season-only, game-level with/without)
MIN_ABSENCE_GAMES = 3
MAX_OUT_PLAYERS_PER_TEAM = 3

ABS_MIN_DELTA_CAP = 10.0
ABS_MIN_DELTA_FLOOR = -4.0

ABS_STAT_MULT_MIN = 0.85
ABS_STAT_MULT_MAX = 1.25
ABS_STACK_MULT_MAX = 1.35

# Which OUT candidates to precompute profiles for (limits runtime)
OUT_CANDIDATE_MIN_MINUTES = 24.0
OUT_CANDIDATE_MIN_SALARY = 5500

# Gist storage
GIST_FILE_SLATE = "latest_dk_slate.csv"
GIST_FILE_OUT = "out_flags.json"
GIST_FILE_BASE = "base_projections_full.csv"
GIST_FILE_ABS = "absence_profiles.json"     # <- NEW: precomputed teammate bumps
GIST_FILE_FINAL = "latest_projections.csv"
GIST_FILE_META = "meta.json"


# ==========================
# PAGE
# ==========================
st.set_page_config(page_title="DK NBA Optimizer", layout="wide", initial_sidebar_state="expanded")
st.title("DraftKings NBA Optimizer — Fast Late Injuries (Precomputed Absence Profiles)")


# ==========================
# Helpers
# ==========================
def clean_name(s: str) -> str:
    return " ".join(str(s).lower().replace(".", "").split())

def parse_minutes(s) -> float:
    s = str(s)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60.0

def dk_positions_list(p: str) -> list:
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def slot_eligible(pos_list: list, slot: str) -> bool:
    pos = set(pos_list or [])
    if slot in ["PG", "SG", "SF", "PF", "C"]:
        return slot in pos
    if slot == "G":
        return bool(pos & {"PG", "SG"})
    if slot == "F":
        return bool(pos & {"SF", "PF"})
    if slot == "UTIL":
        return len(pos) > 0
    return False

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.number)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default

def dk_fp(stats: dict) -> float:
    pts = float(stats.get("PTS", 0))
    reb = float(stats.get("REB", 0))
    ast = float(stats.get("AST", 0))
    stl = float(stats.get("STL", 0))
    blk = float(stats.get("BLK", 0))
    tov = float(stats.get("TOV", 0))
    tpm = float(stats.get("FG3M", 0))

    fp = pts + 1.25 * reb + 1.5 * ast + 2 * stl + 2 * blk - 0.5 * tov + 0.5 * tpm

    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2:
        fp += 1.5
    if cats_10 >= 3:
        fp += 3.0
    return round(fp, 2)

def parse_positions_cell(x):
    if isinstance(x, list):
        return x
    s = str(x)
    if s.startswith("[") and s.endswith("]"):
        s2 = s.strip("[]").replace("'", "").replace('"', "")
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        return parts
    if "/" in s:
        return dk_positions_list(s)
    return [s.strip()] if s.strip() else []


# ==========================
# Gist persistence
# ==========================
def _get_secret(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default

GITHUB_TOKEN = _get_secret("GITHUB_TOKEN", None)
GIST_ID = _get_secret("GIST_ID", None)

def gist_headers():
    if not GITHUB_TOKEN:
        return {}
    return {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def gist_get(gist_id: str) -> dict:
    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=gist_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def gist_update_files(gist_id: str, files: dict):
    payload = {"files": {fn: {"content": content} for fn, content in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=gist_headers(), json=payload, timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read_file(gist_json: dict, filename: str):
    files = gist_json.get("files", {})
    if filename not in files:
        return None
    if "content" in files[filename] and files[filename]["content"] is not None and not files[filename].get("truncated", False):
        return files[filename]["content"]
    raw_url = files[filename].get("raw_url")
    if not raw_url:
        return None
    r = requests.get(raw_url, timeout=20)
    r.raise_for_status()
    return r.text


# ==========================
# NBA helpers (IDs, logs, caching)
# ==========================
@st.cache_data(ttl=3600)
def player_id_map():
    plist = nba_players.get_players()
    return {clean_name(p["full_name"]): int(p["id"]) for p in plist}

def get_player_id(name: str) -> int:
    key = clean_name(name)
    m = player_id_map()
    if key in m:
        return m[key]

    matches = nba_players.find_players_by_full_name(name)
    if matches:
        return int(matches[0]["id"])

    parts = str(name).strip().split()
    if not parts:
        raise ValueError(f"Player not found: {name}")
    last = parts[-1]
    last_matches = nba_players.find_players_by_last_name(last) or []
    if last_matches:
        target = clean_name(name)
        best = None
        best_score = -1.0
        for cand in last_matches:
            cand_name = cand.get("full_name", "")
            score = difflib.SequenceMatcher(None, target, clean_name(cand_name)).ratio()
            if score > best_score:
                best_score = score
                best = cand
        if best is not None and best_score >= 0.55:
            return int(best["id"])

    raise ValueError(f"Player not found: {name}")

@st.cache_data(ttl=86400)
def team_abbrev_to_id():
    tlist = nba_teams.get_teams()
    return {str(t["abbreviation"]).upper(): int(t["id"]) for t in tlist}

@st.cache_data(ttl=3600)
def gamelog(pid: int):
    last_err = None
    for attempt in range(1, GAMELOG_RETRIES + 1):
        try:
            df = playergamelog.PlayerGameLog(
                player_id=pid,
                season=SEASON,
                timeout=API_TIMEOUT_SECONDS
            ).get_data_frames()[0]
            if df is None or df.empty:
                return pd.DataFrame(), "EMPTY_RESPONSE"
            return df, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep((1.7 ** attempt) + random.uniform(0.2, 0.7))
    return pd.DataFrame(), f"FAILED_AFTER_RETRIES ({last_err})"

@st.cache_data(ttl=3600)
def team_games(team_id: int):
    df = teamgamelog.TeamGameLog(team_id=team_id, season=SEASON, timeout=API_TIMEOUT_SECONDS).get_data_frames()[0]
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def per_min_rates_recent(gl: pd.DataFrame, n: int):
    work = gl.head(int(n)).copy()
    work["MIN_float"] = work["MIN"].apply(parse_minutes)
    work = work.dropna(subset=["MIN_float"])
    if work.empty:
        return None, None
    for c in STAT_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    tot_min = float(work["MIN_float"].sum())
    if tot_min <= 0:
        return None, None
    rates = {c: float(work[c].sum()) / tot_min for c in STAT_COLS}
    avg_min = float(work["MIN_float"].mean())
    return rates, avg_min

def per_min_rates_from_games(gl: pd.DataFrame, game_ids: set):
    if gl is None or gl.empty:
        return None, None, 0
    work = gl[gl["Game_ID"].astype(str).isin({str(x) for x in game_ids})].copy()
    if work.empty:
        return None, None, 0
    work["MIN_float"] = work["MIN"].apply(parse_minutes)
    work = work.dropna(subset=["MIN_float"])
    if work.empty:
        return None, None, 0
    for c in STAT_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    tot_min = float(work["MIN_float"].sum())
    if tot_min <= 0:
        return None, None, 0
    rates = {c: float(work[c].sum()) / tot_min for c in STAT_COLS}
    avg_min = float(work["MIN_float"].mean())
    n_games = int(work.shape[0])
    return rates, avg_min, n_games

def project_player_base(name: str, last_n: int):
    pid = get_player_id(name)
    gl, err = gamelog(pid)
    if gl is None or gl.empty:
        raise RuntimeError(f"NO_GAMELOG ({err})")
    rates, mins = per_min_rates_recent(gl, last_n)
    if rates is None or mins is None or mins <= 0:
        raise RuntimeError("NO_MINUTES_IN_SAMPLE")
    stats = {c: round(float(rates[c] * mins), 2) for c in STAT_COLS}
    fp = dk_fp(stats)
    return round(float(mins), 2), float(fp), stats


# ==========================
# DK CSV loader
# ==========================
def read_dk_csv(uploaded_or_text) -> pd.DataFrame:
    if hasattr(uploaded_or_text, "read"):
        df = pd.read_csv(uploaded_or_text)
    else:
        df = pd.read_csv(StringIO(uploaded_or_text))

    required = ["Name", "Position", "Salary", "TeamAbbrev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DK CSV missing required columns: {', '.join(missing)}")

    out = pd.DataFrame({
        "Name": df["Name"].astype(str),
        "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
        "Positions": df["Position"].astype(str).apply(dk_positions_list),
        "Team": df["TeamAbbrev"].astype(str).str.upper(),
    })
    out["Name_clean"] = out["Name"].apply(clean_name)
    out["GameInfo"] = df["Game Info"].astype(str) if "Game Info" in df.columns else ""
    return out


# ==========================
# Old heuristic bumps (fallback, no API calls)
# ==========================
def apply_old_out_bumps(df_in: pd.DataFrame, out_clean_set: set) -> pd.DataFrame:
    df = df_in.copy()
    if "Notes" not in df.columns:
        df["Notes"] = ""
    if "Name_clean" not in df.columns:
        df["Name_clean"] = df["Name"].apply(clean_name)

    df["Status"] = df["Status"].astype(str)
    df.loc[df["Name_clean"].isin(out_clean_set), "Status"] = "OUT"

    ok_mask = df["Status"] == "OK"
    out_mask = df["Status"] == "OUT"

    df["BaseMinutes"] = df["Minutes"]
    for c in STAT_COLS:
        df[f"PM_{c}"] = np.where(
            df["BaseMinutes"].fillna(0) > 0,
            df[c].fillna(0) / df["BaseMinutes"].replace(0, np.nan),
            0.0
        )
        df[f"PM_{c}"] = df[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for team in df.loc[out_mask, "Team"].dropna().unique():
        team_out = df[(df["Team"] == team) & out_mask].copy()
        team_ok = df[(df["Team"] == team) & ok_mask].copy()
        if team_out.empty or team_ok.empty:
            continue

        missing_min = float(team_out["BaseMinutes"].fillna(0).sum())
        if missing_min <= 0:
            continue

        weights = team_ok["BaseMinutes"].fillna(0).clip(lower=MIN_BENCH_FLOOR)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue
        weights = weights / wsum

        for idx in team_ok.index:
            inc = min(float(weights.loc[idx]) * missing_min, MAX_MIN_INCREASE)
            old_m = safe_float(df.at[idx, "Minutes"], 0.0)
            new_m = min(old_m + inc, MAX_MINUTES_PLAYER)
            df.at[idx, "Minutes"] = new_m
            for c in STAT_COLS:
                df.at[idx, c] = round(safe_float(df.at[idx, f"PM_{c}"], 0.0) * new_m, 2)
            if inc > 0:
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" HEUR_MIN+{inc:.1f}").strip()

        team_ok2 = df[(df["Team"] == team) & ok_mask].copy()
        team_ok2["rank"] = team_ok2["Minutes"].fillna(0) + 40.0 * team_ok2["PM_AST"].fillna(0)
        recipients = team_ok2.sort_values("rank", ascending=False).head(OFF_BUMP_TOP_N)

        missing_off = float((team_out["PTS"].fillna(0) + team_out["AST"].fillna(0) + team_out["FG3M"].fillna(0)).sum())
        recipient_off = float((recipients["PTS"].fillna(0) + recipients["AST"].fillna(0) + recipients["FG3M"].fillna(0)).sum())

        if missing_off > 0 and recipient_off > 0:
            raw_mult = 1.0 + OFF_BUMP_STRENGTH * (missing_off / recipient_off)
            mult = min(raw_mult, OFF_BUMP_CAP)
            for idx in recipients.index:
                df.at[idx, "PTS"] = round(safe_float(df.at[idx, "PTS"], 0.0) * mult, 2)
                df.at[idx, "AST"] = round(safe_float(df.at[idx, "AST"], 0.0) * mult, 2)
                df.at[idx, "FG3M"] = round(safe_float(df.at[idx, "FG3M"], 0.0) * mult, 2)
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" HEUR_OFFx{mult:.2f}").strip()

        for idx in df.index[(df["Team"] == team) & ok_mask]:
            stats = {c: safe_float(df.at[idx, c], 0.0) for c in STAT_COLS}
            df.at[idx, "DK_FP"] = dk_fp(stats)
            sal = safe_float(df.at[idx, "Salary"], 0.0)
            df.at[idx, "Value"] = (df.at[idx, "DK_FP"] / (sal / 1000.0)) if sal > 0 else np.nan

    return df


# ==========================
# Absence Profiles: precompute once (slow), apply later (fast)
# ==========================
def choose_out_candidates(base_df_ok: pd.DataFrame) -> pd.DataFrame:
    # Only consider higher-minute / higher-salary slate players as potential OUT candidates
    cand = base_df_ok.copy()
    cand["Minutes"] = pd.to_numeric(cand["Minutes"], errors="coerce")
    cand["Salary"] = pd.to_numeric(cand["Salary"], errors="coerce")
    return cand[
        (cand["Minutes"].fillna(0) >= OUT_CANDIDATE_MIN_MINUTES) |
        (cand["Salary"].fillna(0) >= OUT_CANDIDATE_MIN_SALARY)
    ].copy()

def build_absence_profiles(base_df: pd.DataFrame) -> dict:
    """
    Returns nested dict:
    profiles[team][out_clean] = {
        "n_games": int,
        "teammates": {
            teammate_clean: {"min_delta": float, "mult": float}
        }
    }
    'mult' is a single capped multiplier applied to all stat categories (simple but effective).
    """
    profiles = {}
    if base_df.empty:
        return profiles

    base_df = base_df.copy()
    base_df["Name_clean"] = base_df.get("Name_clean", base_df["Name"].apply(clean_name))
    base_df["Status"] = base_df["Status"].astype(str)

    ok = base_df[base_df["Status"] == "OK"].copy()
    if ok.empty:
        return profiles

    out_candidates = choose_out_candidates(ok)

    teams = sorted(ok["Team"].dropna().unique())
    team_map = team_abbrev_to_id()

    # Precompute gamelogs are already cached from base build, so this uses cache heavily.
    for team in teams:
        team_id = team_map.get(team.upper())
        if not team_id:
            continue

        tg = team_games(team_id)
        if tg is None or tg.empty or "Game_ID" not in tg.columns:
            continue

        team_game_ids = set(tg["Game_ID"].astype(str).tolist())
        team_ok = ok[ok["Team"] == team].copy()
        if team_ok.empty:
            continue

        team_cands = out_candidates[out_candidates["Team"] == team].copy()
        if team_cands.empty:
            continue

        profiles.setdefault(team, {})

        for out_row in team_cands.itertuples(index=False):
            out_name = str(out_row.Name)
            out_clean = str(out_row.Name_clean)

            try:
                pid_out = get_player_id(out_name)
                gl_out, _ = gamelog(pid_out)
                if gl_out is None or gl_out.empty:
                    continue

                played_ids = set(gl_out["Game_ID"].astype(str).tolist())
                abs_ids = team_game_ids - played_ids
                if len(abs_ids) < MIN_ABSENCE_GAMES:
                    continue

                # Baseline per-minute from BASE projections
                # Baseline rates = projected stat / projected minutes (using base model)
                base_rates = {}
                for tm in team_ok.itertuples(index=False):
                    m = safe_float(tm.Minutes, np.nan)
                    if pd.isna(m) or m <= 0:
                        continue
                    base_rates[str(tm.Name_clean)] = {
                        "min": float(m),
                        "pm": {
                            "PTS": safe_float(tm.PTS, 0.0) / float(m),
                            "REB": safe_float(tm.REB, 0.0) / float(m),
                            "AST": safe_float(tm.AST, 0.0) / float(m),
                            "STL": safe_float(tm.STL, 0.0) / float(m),
                            "BLK": safe_float(tm.BLK, 0.0) / float(m),
                            "TOV": safe_float(tm.TOV, 0.0) / float(m),
                            "FG3M": safe_float(tm.FG3M, 0.0) / float(m),
                        }
                    }

                teammates_dict = {}
                used_any = False

                for tm in team_ok.itertuples(index=False):
                    tm_name = str(tm.Name)
                    tm_clean = str(tm.Name_clean)
                    if tm_clean not in base_rates:
                        continue

                    pid_tm = get_player_id(tm_name)
                    gl_tm, _ = gamelog(pid_tm)
                    rates_off, min_off, n_games = per_min_rates_from_games(gl_tm, abs_ids)

                    if rates_off is None or min_off is None or n_games < MIN_ABSENCE_GAMES:
                        continue

                    base_min = base_rates[tm_clean]["min"]
                    dmin = float(min_off) - float(base_min)
                    dmin = max(ABS_MIN_DELTA_FLOOR, min(ABS_MIN_DELTA_CAP, dmin))

                    # Create a single multiplier based on 3 main categories (PTS/AST/REB),
                    # so it behaves well even with small sample noise.
                    mults = []
                    for c in ["PTS", "AST", "REB", "FG3M"]:
                        base_pm = base_rates[tm_clean]["pm"].get(c, 0.0)
                        off_pm = float(rates_off.get(c, 0.0))
                        if base_pm > 0 and off_pm > 0:
                            mults.append(off_pm / base_pm)

                    if len(mults) == 0:
                        continue

                    raw_mult = float(np.median(mults))
                    mult = max(ABS_STAT_MULT_MIN, min(ABS_STAT_MULT_MAX, raw_mult))

                    teammates_dict[tm_clean] = {
                        "min_delta": round(float(dmin), 3),
                        "mult": round(float(mult), 4),
                    }
                    used_any = True

                if used_any:
                    profiles[team][out_clean] = {
                        "n_games": int(len(abs_ids)),
                        "teammates": teammates_dict
                    }

            except Exception:
                continue

    return profiles

def apply_absence_profiles_fast(base_df_in: pd.DataFrame, out_clean_set: set, profiles: dict) -> tuple[pd.DataFrame, set]:
    """
    Apply precomputed absence profiles with NO NBA calls.
    Returns (df_after, teams_with_profile_used).
    """
    df = base_df_in.copy()
    if "Notes" not in df.columns:
        df["Notes"] = ""
    df["Name_clean"] = df.get("Name_clean", df["Name"].apply(clean_name))
    df["Status"] = df["Status"].astype(str)

    # mark OUT based on checkboxes
    df.loc[df["Name_clean"].isin(out_clean_set), "Status"] = "OUT"

    teams_used = set()

    # baseline per-minute from base df (before bumps)
    df["BaseMinutes"] = pd.to_numeric(df["Minutes"], errors="coerce")
    for c in STAT_COLS:
        df[f"PM_{c}"] = np.where(
            df["BaseMinutes"].fillna(0) > 0,
            pd.to_numeric(df[c], errors="coerce").fillna(0) / df["BaseMinutes"].replace(0, np.nan),
            0.0
        )
        df[f"PM_{c}"] = df[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # for each team, apply up to MAX_OUT_PLAYERS_PER_TEAM profiles
    out_rows = df[df["Status"] == "OUT"][["Team", "Name_clean", "Name"]].copy()
    for team in out_rows["Team"].dropna().unique():
        if team not in profiles:
            continue

        outs_team = out_rows[out_rows["Team"] == team].head(MAX_OUT_PLAYERS_PER_TEAM)
        if outs_team.empty:
            continue

        ok_idx = df.index[(df["Team"] == team) & (df["Status"] == "OK")].tolist()
        if not ok_idx:
            continue

        # initialize stacking
        cum_min_delta = {idx: 0.0 for idx in ok_idx}
        cum_mult = {idx: 1.0 for idx in ok_idx}
        used_any_profile = False

        for outp in outs_team.itertuples(index=False):
            out_clean = str(outp.Name_clean)
            prof = profiles.get(team, {}).get(out_clean)
            if not prof:
                continue

            teammates = prof.get("teammates", {})
            for idx in ok_idx:
                tm_clean = str(df.at[idx, "Name_clean"])
                if tm_clean not in teammates:
                    continue

                dmin = float(teammates[tm_clean].get("min_delta", 0.0))
                mult = float(teammates[tm_clean].get("mult", 1.0))

                cum_min_delta[idx] += dmin
                cum_mult[idx] = min(ABS_STACK_MULT_MAX, cum_mult[idx] * mult)
                used_any_profile = True

        if not used_any_profile:
            continue

        teams_used.add(team)

        # apply
        for idx in ok_idx:
            base_min = safe_float(df.at[idx, "BaseMinutes"], 0.0)
            new_min = base_min + cum_min_delta[idx]
            new_min = max(0.0, min(MAX_MINUTES_PLAYER, new_min))
            df.at[idx, "Minutes"] = round(float(new_min), 2)

            m = float(cum_mult[idx])
            for c in STAT_COLS:
                pm = safe_float(df.at[idx, f"PM_{c}"], 0.0)
                df.at[idx, c] = round(float(pm) * m * float(new_min), 2)

            df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" ABSx{m:.2f}").strip()

            stats = {c: safe_float(df.at[idx, c], 0.0) for c in STAT_COLS}
            df.at[idx, "DK_FP"] = dk_fp(stats)
            sal = safe_float(df.at[idx, "Salary"], 0.0)
            df.at[idx, "Value"] = (df.at[idx, "DK_FP"] / (sal / 1000.0)) if sal > 0 else np.nan

    return df, teams_used


# ==========================
# Optimizer
# ==========================
def optimize(pool_df: pd.DataFrame) -> pd.DataFrame:
    prob = LpProblem("DK_NBA_CLASSIC", LpMaximize)
    rows = pool_df.to_dict("records")
    n = len(rows)
    if n == 0:
        raise ValueError("Optimizer pool is empty (no projections).")

    x = {}
    for i in range(n):
        for slot in DK_SLOTS:
            if slot_eligible(rows[i]["Positions"], slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    prob += lpSum(rows[i]["DK_FP"] * x[(i, slot)] for (i, slot) in x)

    for slot in DK_SLOTS:
        prob += lpSum(x[(i, slot)] for i in range(n) if (i, slot) in x) == 1

    for i in range(n):
        prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1

    prob += lpSum(rows[i]["Salary"] * x[(i, slot)] for (i, slot) in x) <= DK_SALARY_CAP
    prob.solve(PULP_CBC_CMD(msg=False))

    lineup_rows = []
    for slot in DK_SLOTS:
        chosen_i = None
        for i in range(n):
            if (i, slot) in x and x[(i, slot)].value() == 1:
                chosen_i = i
                break
        if chosen_i is None:
            raise ValueError("No feasible lineup found (pool too small / slot constraints).")

        r = rows[chosen_i]
        lineup_rows.append({
            "Slot": slot,
            "Name": r["Name"],
            "Team": r["Team"],
            "PosEligible": "/".join(r["Positions"]),
            "Salary": int(r["Salary"]),
            "Minutes": r["Minutes"],
            "PTS": r["PTS"],
            "REB": r["REB"],
            "AST": r["AST"],
            "3PM": r["FG3M"],
            "STL": r["STL"],
            "BLK": r["BLK"],
            "TOV": r["TOV"],
            "DK_FP": round(float(r["DK_FP"]), 2),
            "Value": r["Value"],
            "Notes": r.get("Notes", ""),
        })
    return pd.DataFrame(lineup_rows)


# ==========================
# Secrets check
# ==========================
if not (GITHUB_TOKEN and GIST_ID):
    st.error("Missing secrets. Add GITHUB_TOKEN and GIST_ID in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# ==========================
# Sidebar
# ==========================
uploaded = st.sidebar.file_uploader("Upload DraftKings CSV (saved for phone too)", type="csv")
last_n = st.sidebar.number_input("Recent games window (N)", 1, 30, int(st.session_state.get("last_n", DEFAULT_LAST_N)))
st.session_state["last_n"] = int(last_n)

use_absence_profiles = st.sidebar.checkbox("Use Absence Profiles (season-only, precomputed)", value=True)
use_fallback_bumps = st.sidebar.checkbox("Fallback Heuristic bumps if no profile", value=True)
show_debug = st.sidebar.checkbox("Show debug tables", value=True)

reload_btn = st.sidebar.button("Reload saved slate/base/profiles/final")


# ==========================
# Load gist
# ==========================
gist_json = gist_get(GIST_ID) if (reload_btn or "gist_json" not in st.session_state) else st.session_state["gist_json"]
st.session_state["gist_json"] = gist_json

if uploaded is not None:
    csv_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    gist_update_files(GIST_ID, {
        GIST_FILE_SLATE: csv_text,
        GIST_FILE_META: json.dumps({"season": SEASON, "saved_at": time.time()}, indent=2),
    })
    st.sidebar.success("Saved slate.")
    slate_df = read_dk_csv(csv_text)
else:
    slate_text = gist_read_file(gist_json, GIST_FILE_SLATE)
    if not slate_text:
        st.info("Upload a DK CSV to begin.")
        st.stop()
    slate_df = read_dk_csv(slate_text)

base_text = gist_read_file(gist_json, GIST_FILE_BASE)
abs_text = gist_read_file(gist_json, GIST_FILE_ABS)
final_text = gist_read_file(gist_json, GIST_FILE_FINAL)

out_flags_text = gist_read_file(gist_json, GIST_FILE_OUT)
saved_out_flags = {}
if out_flags_text:
    try:
        saved_out_flags = json.loads(out_flags_text)
    except Exception:
        saved_out_flags = {}


# ==========================
# OUT checkboxes
# ==========================
slate_view = slate_df.copy()
slate_view["OUT"] = slate_view["Name_clean"].map(lambda k: bool(saved_out_flags.get(k, False)))

st.write(f"**Season:** {SEASON}  |  **Cap:** ${DK_SALARY_CAP:,}  |  **Slots:** {', '.join(DK_SLOTS)}")
st.subheader("Step 1 — Mark OUT players (checkbox).")

edited = st.data_editor(
    slate_view[["OUT", "Name", "Team", "Positions", "Salary", "GameInfo"]],
    use_container_width=True,
    hide_index=True,
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name", "Team", "Positions", "Salary", "GameInfo"],
)

name_to_clean = dict(zip(slate_view["Name"], slate_view["Name_clean"]))
live_flags = {}
for _, row in edited.iterrows():
    nm = str(row["Name"])
    cl = name_to_clean.get(nm, clean_name(nm))
    live_flags[cl] = bool(row["OUT"])
out_clean_set = {k for k, v in live_flags.items() if v}

colA, colB = st.columns(2)
with colA:
    if st.button("Save OUT selections"):
        gist_update_files(GIST_ID, {GIST_FILE_OUT: json.dumps(live_flags, indent=2)})
        st.success("Saved OUT selections.")
with colB:
    st.caption("Tip: after late injuries, you’ll click Step B (fast).")


# ==========================
# Step A (slow): Build Base + Profiles
# ==========================
st.divider()
st.subheader("Step A (slow) — Build BASE projections + Absence Profiles (run once per slate)")

rebuild_base = st.button("Build BASE + Absence Profiles (slow)")
retry_err = st.button("Retry ERR players only (slow)")

if rebuild_base:
    rows = []
    total = len(slate_df)
    prog = st.progress(0, text="Starting BASE projections...")

    for i, r in enumerate(slate_df.itertuples(index=False), start=1):
        prog.progress(int(i / max(total, 1) * 100), text=f"Running gamelog({r.Name}) ({i}/{total})")
        try:
            mins, fp, stats = project_player_base(r.Name, st.session_state["last_n"])
            sal = safe_float(r.Salary, np.nan)
            value = (fp / (sal / 1000.0)) if (pd.notna(sal) and sal > 0) else np.nan
            rows.append({
                "Name": r.Name,
                "Name_clean": r.Name_clean,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": sal,
                "Minutes": mins,
                "PTS": stats["PTS"],
                "REB": stats["REB"],
                "AST": stats["AST"],
                "FG3M": stats["FG3M"],
                "STL": stats["STL"],
                "BLK": stats["BLK"],
                "TOV": stats["TOV"],
                "DK_FP": fp,
                "Value": value,
                "Status": "OK",
                "Notes": "",
            })
        except Exception as e:
            rows.append({
                "Name": r.Name,
                "Name_clean": r.Name_clean,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": safe_float(r.Salary, np.nan),
                "Minutes": np.nan,
                "PTS": np.nan, "REB": np.nan, "AST": np.nan, "FG3M": np.nan, "STL": np.nan, "BLK": np.nan, "TOV": np.nan,
                "DK_FP": np.nan,
                "Value": np.nan,
                "Status": f"ERR: {str(e)[:120]}",
                "Notes": "",
            })
        time.sleep(BUILD_THROTTLE_SECONDS)

    prog.empty()
    base_df = pd.DataFrame(rows)

    st.info("Building Absence Profiles (season-only) — this is the part that makes late injuries fast.")
    prof_prog = st.progress(0, text="Building absence profiles...")
    try:
        ok_df = base_df[base_df["Status"] == "OK"].copy()
        # small progress estimate: number of teams
        teams = sorted(ok_df["Team"].dropna().unique())
        profiles = build_absence_profiles(base_df)
        prof_prog.progress(100, text="Absence profiles complete.")
    except Exception:
        profiles = {}
        prof_prog.progress(100, text="Absence profiles failed; will rely on fallback bumps.")
    time.sleep(0.2)
    prof_prog.empty()

    gist_update_files(GIST_ID, {
        GIST_FILE_BASE: base_df.to_csv(index=False),
        GIST_FILE_ABS: json.dumps(profiles),
        GIST_FILE_META: json.dumps(
            {"season": SEASON, "saved_at": time.time(), "last_n": int(st.session_state["last_n"]), "built": "base+profiles"},
            indent=2
        ),
    })

    st.success("Saved BASE projections + Absence Profiles. Late injuries should be fast now.")
    st.write(base_df["Status"].value_counts(dropna=False))
    if show_debug:
        st.dataframe(base_df, use_container_width=True, hide_index=True)

if retry_err:
    if not base_text:
        st.error("No BASE saved yet. Run Step A first.")
    else:
        base_df = pd.read_csv(StringIO(base_text))
        base_df["Positions"] = base_df["Positions"].apply(parse_positions_cell)

        err_mask = base_df["Status"].astype(str).str.startswith("ERR:")
        err_rows = base_df[err_mask].copy()

        st.write(f"Retrying {len(err_rows)} ERR players...")
        prog = st.progress(0, text="Retrying ERR players...")

        for j, row in enumerate(err_rows.itertuples(index=True), start=1):
            prog.progress(int(j / max(len(err_rows), 1) * 100), text=f"Retry gamelog({row.Name}) ({j}/{len(err_rows)})")
            try:
                mins, fp, stats = project_player_base(row.Name, st.session_state["last_n"])
                sal = safe_float(row.Salary, np.nan)
                value = (fp / (sal / 1000.0)) if (pd.notna(sal) and sal > 0) else np.nan

                base_df.at[row.Index, "Minutes"] = mins
                base_df.at[row.Index, "PTS"] = stats["PTS"]
                base_df.at[row.Index, "REB"] = stats["REB"]
                base_df.at[row.Index, "AST"] = stats["AST"]
                base_df.at[row.Index, "FG3M"] = stats["FG3M"]
                base_df.at[row.Index, "STL"] = stats["STL"]
                base_df.at[row.Index, "BLK"] = stats["BLK"]
                base_df.at[row.Index, "TOV"] = stats["TOV"]
                base_df.at[row.Index, "DK_FP"] = fp
                base_df.at[row.Index, "Value"] = value
                base_df.at[row.Index, "Status"] = "OK"
                base_df.at[row.Index, "Notes"] = ""
            except Exception as e:
                base_df.at[row.Index, "Status"] = f"ERR: {str(e)[:120]}"

            time.sleep(BUILD_THROTTLE_SECONDS)

        prog.empty()
        gist_update_files(GIST_ID, {GIST_FILE_BASE: base_df.to_csv(index=False)})
        st.success("Updated BASE projections (ERR retries).")
        st.write(base_df["Status"].value_counts(dropna=False))


# ==========================
# Step B (fast): Apply OUT using profiles + fallback
# ==========================
st.divider()
st.subheader("Step B (fast) — Apply OUT bumps + Save FINAL (no NBA calls)")

apply_out_fast = st.button("Apply OUT bumps + Save FINAL (fast)")

if apply_out_fast:
    if not base_text:
        st.error("No BASE saved yet. Run Step A first.")
        st.stop()

    gist_update_files(GIST_ID, {GIST_FILE_OUT: json.dumps(live_flags, indent=2)})

    base_df = pd.read_csv(StringIO(base_text))
    base_df["Positions"] = base_df["Positions"].apply(parse_positions_cell)
    base_df["Name_clean"] = base_df.get("Name_clean", base_df["Name"].apply(clean_name))
    base_df["Status"] = base_df["Status"].astype(str)

    profiles = {}
    if abs_text:
        try:
            profiles = json.loads(abs_text)
        except Exception:
            profiles = {}

    work = base_df.copy()

    teams_used = set()
    if use_absence_profiles and out_clean_set and profiles:
        try:
            work, teams_used = apply_absence_profiles_fast(work, out_clean_set, profiles)
        except Exception:
            teams_used = set()

    # Fallback heuristic bumps for teams where absence profiles didn't apply
    if use_fallback_bumps and out_clean_set:
        # Determine teams with OUT but not used by profiles
        work["Name_clean"] = work.get("Name_clean", work["Name"].apply(clean_name))
        work["Status"] = work["Status"].astype(str)
        out_teams = work[work["Name_clean"].isin(out_clean_set)]["Team"].dropna().unique()
        for t in out_teams:
            if t not in teams_used:
                work = apply_old_out_bumps(work, out_clean_set)

    # FINAL display: OK and not OUT
    work["Name_clean"] = work.get("Name_clean", work["Name"].apply(clean_name))
    final_df = work[(work["Status"] == "OK") & (~work["Name_clean"].isin(out_clean_set))].copy()
    final_df = final_df.sort_values("DK_FP", ascending=False)

    gist_update_files(GIST_ID, {
        GIST_FILE_FINAL: final_df.to_csv(index=False),
        GIST_FILE_META: json.dumps(
            {"season": SEASON, "saved_at": time.time(), "last_n": int(st.session_state["last_n"]), "built": "final_fast"},
            indent=2
        ),
    })

    st.success("Saved FINAL projections (fast).")
    st.dataframe(final_df.rename(columns={"FG3M": "3PM"}), use_container_width=True, hide_index=True)

    if show_debug:
        st.subheader("Debug: Who was OUT / ERR?")
        st.write(work["Status"].value_counts(dropna=False))
        with st.expander("Non-OK rows (debug)"):
            st.dataframe(work[work["Status"] != "OK"], use_container_width=True, hide_index=True)


# ==========================
# Optimizer
# ==========================
st.divider()
st.subheader("Optimizer (uses FINAL projections)")

final_text = gist_read_file(gist_json, GIST_FILE_FINAL)
if not final_text:
    st.info("No FINAL projections yet. Run Step A then Step B.")
    st.stop()

pool_df = pd.read_csv(StringIO(final_text))
pool_df["Positions"] = pool_df["Positions"].apply(parse_positions_cell)

if "FG3M" not in pool_df.columns and "3PM" in pool_df.columns:
    pool_df["FG3M"] = pool_df["3PM"]

if st.button("Optimize Lineup (Classic)"):
    p = pool_df.dropna(subset=["Salary", "DK_FP"]).copy()
    p = p[p["Salary"] > 0].copy()

    try:
        lineup = optimize(p)
        st.subheader("Optimized Lineup")
        st.dataframe(lineup, use_container_width=True, hide_index=True)
        st.metric("Total Salary", int(lineup["Salary"].sum()))
        st.metric("Total DK FP", round(float(lineup["DK_FP"].sum()), 2))

        out_csv = lineup.to_csv(index=False).encode("utf-8")
        st.download_button("Download Lineup CSV", data=out_csv, file_name="dk_optimized_lineup.csv", mime="text/csv")

    except Exception as e:
        st.error(str(e))
