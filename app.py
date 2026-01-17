import json
import time
import random
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

# Cloud stability
API_TIMEOUT_SECONDS = 35
GAMELOG_RETRIES = 4
BUILD_THROTTLE_SECONDS = 0.10

# Old bump engine (fallback / also used when sample too small)
MIN_BENCH_FLOOR = 6.0
MAX_MINUTES_PLAYER = 40.0
MAX_MIN_INCREASE = 22.0

OFF_BUMP_STRENGTH = 0.22
OFF_BUMP_CAP = 1.25
OFF_BUMP_TOP_N = 6

# Season-only Historical Absence Bumps (recommended)
MIN_ABSENCE_GAMES = 3              # must have at least this many missed games to trust absences
MAX_OUT_PLAYERS_PER_TEAM = 3       # safety
ABS_MIN_DELTA_CAP = 10.0           # max minutes added from absence profile per OUT player
ABS_MIN_DELTA_FLOOR = -4.0         # allow small decreases
ABS_STAT_MULT_MIN = 0.85           # cap multipliers so it doesn't go crazy
ABS_STAT_MULT_MAX = 1.25
ABS_STACK_MULT_MAX = 1.35          # cap stacking from multiple OUTs

# Gist file names
GIST_FILE_SLATE = "latest_dk_slate.csv"
GIST_FILE_OUT = "out_flags.json"
GIST_FILE_PROJ = "latest_projections.csv"
GIST_FILE_META = "meta.json"


# ==========================
# PAGE
# ==========================
st.set_page_config(page_title="DK NBA Optimizer", layout="wide", initial_sidebar_state="expanded")
st.title("DraftKings NBA Optimizer — Persistent Slate + OUT Checkboxes + On-demand Projections")


# ==========================
# Helpers / DK scoring
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

def dk_fp(stats: dict) -> float:
    pts = float(stats.get("PTS", 0))
    reb = float(stats.get("REB", 0))
    ast = float(stats.get("AST", 0))
    stl = float(stats.get("STL", 0))
    blk = float(stats.get("BLK", 0))
    tov = float(stats.get("TOV", 0))
    tpm = float(stats.get("FG3M", 0))

    fp = pts + 1.25 * reb + 1.5 * ast + 2 * stl + 2 * blk - 0.5 * tov + 0.5 * tpm

    # DK bonuses
    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2:
        fp += 1.5
    if cats_10 >= 3:
        fp += 3.0

    return round(fp, 2)

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


# ==========================
# GitHub Gist persistence
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
@st.cache_data(ttl=86400)
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
            time.sleep((1.6 ** attempt) + random.uniform(0.3, 0.9))
    return pd.DataFrame(), f"FAILED_AFTER_RETRIES ({last_err})"

@st.cache_data(ttl=3600)
def team_games(team_id: int):
    df = teamgamelog.TeamGameLog(team_id=team_id, season=SEASON, timeout=API_TIMEOUT_SECONDS).get_data_frames()[0]
    if df is None or df.empty:
        return pd.DataFrame()
    return df

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

def project_player(name: str, last_n: int):
    pid = get_player_id(name)
    gl, err = gamelog(pid)
    if gl is None or gl.empty:
        raise RuntimeError(f"NO_GAMELOG ({err})")

    rates, mins = per_min_rates_recent(gl, last_n)
    if rates is None or mins is None or mins <= 0:
        raise RuntimeError("NO_MINUTES_IN_SAMPLE")

    proj_stats = {c: round(float(rates[c] * mins), 2) for c in STAT_COLS}
    fp = dk_fp(proj_stats)
    return round(float(mins), 2), float(fp), proj_stats


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
# Old bump engine (fallback)
# ==========================
def apply_old_out_bumps(full_proj_df: pd.DataFrame, only_team: str | None = None) -> pd.DataFrame:
    df = full_proj_df.copy()
    ok_mask = df["Status"].astype(str) == "OK"
    out_mask = df["Status"].astype(str) == "OUT"

    df["BaseMinutes"] = df["Minutes"]
    for c in STAT_COLS:
        df[f"PM_{c}"] = np.where(
            df["BaseMinutes"].fillna(0) > 0,
            df[c].fillna(0) / df["BaseMinutes"].replace(0, np.nan),
            0.0
        )
        df[f"PM_{c}"] = df[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "Notes" not in df.columns:
        df["Notes"] = ""

    teams_to_process = df.loc[out_mask, "Team"].dropna().unique()
    if only_team is not None:
        teams_to_process = [only_team] if only_team in set(teams_to_process) else []

    for team in teams_to_process:
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
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" MIN+{inc:.1f}").strip()

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
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" OFFx{mult:.2f}").strip()

        for idx in df.index[(df["Team"] == team) & ok_mask]:
            stats = {c: safe_float(df.at[idx, c], 0.0) for c in STAT_COLS}
            df.at[idx, "DK_FP"] = dk_fp(stats)
            sal = safe_float(df.at[idx, "Salary"], 0.0)
            df.at[idx, "Value"] = (df.at[idx, "DK_FP"] / (sal / 1000.0)) if sal > 0 else np.nan

    return df


# ==========================
# Season-only Historical Absence Bumps
# ==========================
@st.cache_data(ttl=21600)  # 6 hours
def absence_game_ids_for_player(team_abbrev: str, out_player_name: str):
    team_id = team_abbrev_to_id().get(team_abbrev.upper())
    if not team_id:
        return set(), 0

    tg = team_games(team_id)
    if tg is None or tg.empty or "Game_ID" not in tg.columns:
        return set(), 0

    team_game_ids = set(tg["Game_ID"].astype(str).tolist())

    pid = get_player_id(out_player_name)
    gl, _ = gamelog(pid)
    if gl is None or gl.empty:
        return set(), 0

    played_ids = set(gl["Game_ID"].astype(str).tolist())
    abs_ids = team_game_ids - played_ids
    return abs_ids, len(abs_ids)

def apply_absence_bumps(df_in: pd.DataFrame, out_clean_set: set) -> tuple[pd.DataFrame, set]:
    """
    Returns (df_after, teams_used_absence).
    For teams where no absence profile is usable, it does NOT change them.
    """
    df = df_in.copy()
    if "Notes" not in df.columns:
        df["Notes"] = ""
    if "Name_clean" not in df.columns:
        df["Name_clean"] = df["Name"].apply(clean_name)

    teams_used_absence = set()

    teams_with_out = df[df["Name_clean"].isin(out_clean_set)]["Team"].dropna().unique()
    for team in teams_with_out:
        out_rows = df[(df["Team"] == team) & (df["Name_clean"].isin(out_clean_set))].copy()
        ok_rows = df[(df["Team"] == team) & (df["Status"] == "OK")].copy()
        if out_rows.empty or ok_rows.empty:
            continue

        out_rows = out_rows.head(MAX_OUT_PLAYERS_PER_TEAM)

        baseline_pm = {}
        for idx in ok_rows.index:
            m = safe_float(df.at[idx, "Minutes"], np.nan)
            if pd.notna(m) and m > 0:
                baseline_pm[idx] = {c: safe_float(df.at[idx, c], 0.0) / m for c in STAT_COLS}

        used_any_absence = False
        cum_mult = {idx: 1.0 for idx in ok_rows.index}
        cum_min_delta = {idx: 0.0 for idx in ok_rows.index}
        notes_add = {idx: "" for idx in ok_rows.index}

        for _, outp in out_rows.iterrows():
            out_name = str(outp["Name"])
            abs_ids, abs_count = absence_game_ids_for_player(team, out_name)
            if abs_count < MIN_ABSENCE_GAMES or len(abs_ids) < MIN_ABSENCE_GAMES:
                continue

            for idx in ok_rows.index:
                teammate_name = str(df.at[idx, "Name"])
                try:
                    pid_tm = get_player_id(teammate_name)
                    gl_tm, _ = gamelog(pid_tm)
                    rates_off, min_off, n_games = per_min_rates_from_games(gl_tm, abs_ids)
                    if rates_off is None or min_off is None or n_games < MIN_ABSENCE_GAMES:
                        continue

                    base_min = safe_float(df.at[idx, "Minutes"], 0.0)
                    dmin = float(min_off) - float(base_min)
                    dmin = max(ABS_MIN_DELTA_FLOOR, min(ABS_MIN_DELTA_CAP, dmin))
                    cum_min_delta[idx] += dmin

                    for c in STAT_COLS:
                        base_rate = baseline_pm.get(idx, {}).get(c, None)
                        if base_rate is None or base_rate <= 0:
                            continue
                        off_rate = float(rates_off.get(c, 0.0))
                        if off_rate <= 0:
                            continue
                        mult = off_rate / base_rate
                        mult = max(ABS_STAT_MULT_MIN, min(ABS_STAT_MULT_MAX, mult))
                        cum_mult[idx] = min(ABS_STACK_MULT_MAX, cum_mult[idx] * mult)

                    notes_add[idx] += f" {out_name.split(' ')[-1]}OFF({abs_count})"
                    used_any_absence = True
                except Exception:
                    continue

        if not used_any_absence:
            continue

        teams_used_absence.add(team)

        for idx in ok_rows.index:
            base_min = safe_float(df.at[idx, "Minutes"], 0.0)
            new_min = base_min + cum_min_delta.get(idx, 0.0)
            new_min = max(0.0, min(MAX_MINUTES_PLAYER, new_min))
            df.at[idx, "Minutes"] = round(new_min, 2)

            m = cum_mult.get(idx, 1.0)
            if base_min > 0:
                for c in STAT_COLS:
                    base_rate = safe_float(df.at[idx, c], 0.0) / base_min
                    df.at[idx, c] = round(base_rate * m * new_min, 2)

            extra = notes_add.get(idx, "").strip()
            if extra:
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" ABS:{extra}").strip()

            stats = {c: safe_float(df.at[idx, c], 0.0) for c in STAT_COLS}
            df.at[idx, "DK_FP"] = dk_fp(stats)
            sal = safe_float(df.at[idx, "Salary"], 0.0)
            df.at[idx, "Value"] = (df.at[idx, "DK_FP"] / (sal / 1000.0)) if sal > 0 else np.nan

    return df, teams_used_absence


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
            raise ValueError("No feasible lineup found. (Pool too small / position constraints.)")

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
# Required secrets
# ==========================
if not (GITHUB_TOKEN and GIST_ID):
    st.error("Missing Streamlit Secrets. Add GITHUB_TOKEN and GIST_ID in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# ==========================
# Sidebar controls
# ==========================
uploaded = st.sidebar.file_uploader("Upload DraftKings CSV (saves for phone too)", type="csv")
last_n = st.sidebar.number_input("Recent games window (N)", 1, 30, int(st.session_state.get("last_n", DEFAULT_LAST_N)))
st.session_state["last_n"] = int(last_n)

st.sidebar.caption("Bumps:")
use_absence_bumps = st.sidebar.checkbox("Use season-only historical absence bumps (recommended)", value=True)
use_fallback_bumps = st.sidebar.checkbox("Fallback bumps (minutes redistribution)", value=True)

show_debug = st.sidebar.checkbox("Show debug tables (missing players, errors)", value=True)

reload_btn = st.sidebar.button("Reload saved slate/projections")


# ==========================
# Load from gist (or upload overrides)
# ==========================
gist_json = gist_get(GIST_ID) if (reload_btn or "gist_json" not in st.session_state) else st.session_state["gist_json"]
st.session_state["gist_json"] = gist_json

if uploaded is not None:
    csv_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    gist_update_files(GIST_ID, {
        GIST_FILE_SLATE: csv_text,
        GIST_FILE_META: json.dumps({"season": SEASON, "saved_at": time.time()}, indent=2),
    })
    st.sidebar.success("Saved slate. Open on phone and it will load the same slate.")
    slate_df = read_dk_csv(csv_text)
else:
    slate_text = gist_read_file(gist_json, GIST_FILE_SLATE)
    if not slate_text:
        st.info("Upload a DK CSV in the sidebar to begin.")
        st.stop()
    slate_df = read_dk_csv(slate_text)

proj_text = gist_read_file(gist_json, GIST_FILE_PROJ)

out_flags_text = gist_read_file(gist_json, GIST_FILE_OUT)
saved_out_flags = {}
if out_flags_text:
    try:
        saved_out_flags = json.loads(out_flags_text)
    except Exception:
        saved_out_flags = {}

# Build checkbox table
slate_view = slate_df.copy()
slate_view["OUT"] = slate_view["Name_clean"].map(lambda k: bool(saved_out_flags.get(k, False)))

st.write(f"**Season:** {SEASON}  |  **Cap:** ${DK_SALARY_CAP:,}  |  **Slots:** {', '.join(DK_SLOTS)}")
st.subheader("Step 1 — Mark OUT players (checkbox). Projections only run when you click Run Projections.")

edited = st.data_editor(
    slate_view[["OUT", "Name", "Team", "Positions", "Salary", "GameInfo"]],
    use_container_width=True,
    hide_index=True,
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name", "Team", "Positions", "Salary", "GameInfo"],
)

# Live OUT set from checkboxes
name_to_clean = dict(zip(slate_view["Name"], slate_view["Name_clean"]))
live_flags = {}
for _, row in edited.iterrows():
    nm = str(row["Name"])
    cl = name_to_clean.get(nm, clean_name(nm))
    live_flags[cl] = bool(row["OUT"])
out_clean_set = {k for k, v in live_flags.items() if v}

# Show last saved projections (no recompute)
if proj_text:
    try:
        last_proj = pd.read_csv(StringIO(proj_text))
        st.subheader("Last Saved Projections (no recompute)")
        st.dataframe(last_proj.rename(columns={"FG3M": "3PM"}), use_container_width=True, hide_index=True)
    except Exception:
        pass

st.divider()
st.subheader("Step 2 — Run projections (only when you click)")

run_proj = st.button("Run Projections (apply OUT bumps, remove OUT, save results)")
latest_display_df = None

if run_proj:
    # Persist OUT selections automatically on projection run (phone + desktop sync)
    gist_update_files(GIST_ID, {GIST_FILE_OUT: json.dumps(live_flags, indent=2)})

    rows = []
    total = len(slate_df)
    progress = st.progress(0, text="Starting projections...")

    for i, r in enumerate(slate_df.itertuples(index=False), start=1):
        progress.progress(int(i / max(total, 1) * 100), text=f"Running gamelog({r.Name}) ({i}/{total})")

        status = "OUT" if r.Name_clean in out_clean_set else "OK"

        try:
            mins, fp, stats = project_player(r.Name, st.session_state["last_n"])
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
                "Status": status,
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
                "Status": f"ERR: {str(e)[:90]}",
                "Notes": "",
            })

        time.sleep(BUILD_THROTTLE_SECONDS)

    progress.empty()
    full_proj_df = pd.DataFrame(rows)

    # Apply bumps using OUT players in background
    teams_used_absence = set()
    if out_clean_set:
        if use_absence_bumps:
            try:
                full_proj_df, teams_used_absence = apply_absence_bumps(full_proj_df, out_clean_set)
            except Exception:
                teams_used_absence = set()

        # fallback bumps for teams where absence bumps weren't used
        if use_fallback_bumps:
            for t in full_proj_df[(full_proj_df["Status"] == "OUT")]["Team"].dropna().unique():
                if t not in teams_used_absence:
                    full_proj_df = apply_old_out_bumps(full_proj_df, only_team=t)

    # DEBUG: show why players "disappear"
    if show_debug:
        st.subheader("Status counts (debug)")
        st.write(full_proj_df["Status"].value_counts(dropna=False))

        with st.expander("Show non-OK rows (ERR / OUT)"):
            st.dataframe(
                full_proj_df[full_proj_df["Status"] != "OK"].sort_values(["Team", "Status"]),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Search a player in FULL results (debug)"):
            search_name = st.text_input("Type a player name to find (debug)", "")
            if search_name.strip():
                key = clean_name(search_name)
                hits = full_proj_df[full_proj_df["Name_clean"].str.contains(key, na=False)]
                st.dataframe(hits, use_container_width=True, hide_index=True)

    # Bulletproof OUT removal (remove OUT players even if status weird)
    if "Name_clean" not in full_proj_df.columns:
        full_proj_df["Name_clean"] = full_proj_df["Name"].apply(clean_name)

    proj_df = full_proj_df[
        (full_proj_df["Status"] == "OK") &
        (~full_proj_df["Name_clean"].isin(out_clean_set))
    ].copy()

    display_df = proj_df.copy().sort_values("DK_FP", ascending=False)

    gist_update_files(GIST_ID, {
        GIST_FILE_PROJ: display_df.to_csv(index=False),
        GIST_FILE_META: json.dumps(
            {
                "season": SEASON,
                "saved_at": time.time(),
                "last_n": int(st.session_state["last_n"]),
                "absence_bumps": bool(use_absence_bumps),
            },
            indent=2
        ),
    })

    st.success("Projections complete and saved. Phone/desktop will show these without re-running.")
    st.subheader("Projections (OUT removed, bumps applied)")
    st.dataframe(display_df.rename(columns={"FG3M": "3PM"}), use_container_width=True, hide_index=True)

    latest_display_df = display_df

st.divider()
st.subheader("Optimizer (uses saved projections)")

# Load projections for optimizer
pool_df = None
if latest_display_df is not None:
    pool_df = latest_display_df.copy()
else:
    if proj_text:
        try:
            pool_df = pd.read_csv(StringIO(proj_text))
        except Exception:
            pool_df = None

if pool_df is None or pool_df.empty:
    st.info("No projections saved yet. Run projections first.")
    st.stop()

# Fix: older saved files might have 3PM
if "FG3M" not in pool_df.columns and "3PM" in pool_df.columns:
    pool_df["FG3M"] = pool_df["3PM"]

# Restore Positions to list after CSV reload
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

pool_df["Positions"] = pool_df["Positions"].apply(parse_positions_cell)

run_opt = st.button("Optimize Lineup (Classic)")
if run_opt:
    p = pool_df.copy()
    p = p.dropna(subset=["Salary", "DK_FP"]).copy()
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
