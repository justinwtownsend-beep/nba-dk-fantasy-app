import json
import time
import difflib
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from nba_api.stats.endpoints import leaguedashplayerstats

# Optimizer (ILP)
try:
    import pulp
except Exception:
    pulp = None


# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
TIMEOUT = 20

# Gist persistence (mobile/desktop share)
GIST_OUT = "out.json"
GIST_FINAL = "final.csv"
GIST_LOCKS = "locks.json"  # stores locked teams + locked players (clean names)

# Minutes fix (ONLY requested change)
MAX_MINUTES = 34.0
TEAM_TOTAL_MINUTES = 240.0  # 48*5

SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
PRIMARY_POS_ORDER = ["PG", "SG", "SF", "PF", "C"]


# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DK Projections + DvP + DK Classic Optimizer (Late Swap Locks + Minutes Cap)")

st.caption(
    f"Season fixed to {SEASON}. Minutes are capped at {MAX_MINUTES} even with injuries, "
    "and excess minutes are redistributed to teammates."
)


# ==========================
# HELPERS
# ==========================
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def deaccent(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
    )

def clean_name(s: str) -> str:
    s = deaccent(s).lower()
    s = s.replace(".", "").replace(",", "").replace("’", "'").replace("`", "'")
    return " ".join(s.split())

def strip_suffix(name: str) -> str:
    parts = clean_name(name).split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)

def safe_num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def normalize_team_abbrev(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip().upper()
    m = {
        "NY": "NYK",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        "PHO": "PHX",
        "WAS": "WAS",
    }
    return m.get(t, t)

def parse_opponent_from_game_info(team: str, game_info: str) -> str:
    if not isinstance(game_info, str) or "@" not in game_info:
        return ""
    left = game_info.split(" ")[0]  # 'BOS@NYK'
    if "@" not in left:
        return ""
    a, b = left.split("@", 1)
    a = normalize_team_abbrev(a)
    b = normalize_team_abbrev(b)
    team = normalize_team_abbrev(team)
    if team == a:
        return b
    if team == b:
        return a
    return ""

def primary_pos(pos_str: str) -> str:
    if not isinstance(pos_str, str) or not pos_str.strip():
        return ""
    parts = [p.strip().upper() for p in pos_str.split("/") if p.strip()]
    for p in PRIMARY_POS_ORDER:
        if p in parts:
            return p
    return parts[0] if parts else ""

def eligible_slots(pos_str: str):
    if not isinstance(pos_str, str) or not pos_str.strip():
        return set(["UTIL"])
    parts = {p.strip().upper() for p in pos_str.split("/") if p.strip()}
    slots = set(["UTIL"])
    for p in parts:
        if p in {"PG","SG","SF","PF","C"}:
            slots.add(p)
    if "PG" in parts or "SG" in parts:
        slots.add("G")
    if "SF" in parts or "PF" in parts:
        slots.add("F")
    return slots

def dk_fp(pts, reb, ast, stl, blk, tov, fg3m):
    fp = pts + 1.25*reb + 1.5*ast + 2.0*stl + 2.0*blk - 0.5*tov + 0.5*fg3m
    cats = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats >= 2:
        fp += 1.5
    if cats >= 3:
        fp += 3.0
    return fp


# ==========================
# GIST IO
# ==========================
def gh_headers(token: str):
    return {"Authorization": f"token {token}"}

def gist_get(gist_id: str, token: str):
    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=gh_headers(token), timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read(gist_id: str, token: str, filename: str):
    g = gist_get(gist_id, token)
    if filename not in g.get("files", {}):
        return None
    f = g["files"][filename]
    if not f.get("truncated"):
        return f.get("content")
    rr = requests.get(f["raw_url"], timeout=25)
    rr.raise_for_status()
    return rr.text

def gist_write(gist_id: str, token: str, files: dict):
    payload = {"files": {k: {"content": v} for k, v in files.items()}}
    r = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=gh_headers(token),
        json=payload,
        timeout=25
    )
    r.raise_for_status()


# ==========================
# NBA season per-game
# ==========================
@st.cache_data(ttl=1800)
def nba_league_dash():
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        timeout=TIMEOUT
    ).get_data_frames()[0]

    keep = ["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    df = df[keep].copy()
    df.columns = ["PLAYER_ID", "NBA_Name", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_nba_row(name: str, nba_df: pd.DataFrame):
    cn = clean_name(name)
    sn = strip_suffix(name)

    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        return exact.iloc[0]

    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        return exact2.iloc[0]

    parts = sn.split()
    if parts:
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last]
        if not cand.empty:
            best_row, best_score = None, 0.0
            for _, row in cand.iterrows():
                score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_row is not None and best_score >= 0.75:
                return best_row

    hit = difflib.get_close_matches(cn, nba_df["NBA_Name_clean"].tolist(), n=1, cutoff=0.90)
    if hit:
        return nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]

    return None


# ==========================
# DVP LOADING + APPLY
# ==========================
def load_dvp_csv(file_bytes) -> pd.DataFrame:
    """
    Expect your manual export from HashtagBasketball.
    Normalizes to: DEF_TEAM, POS, PTS, REB, AST, FG3M as multipliers.
    """
    txt = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(txt))

    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    # team column
    team_col = None
    for c in df.columns:
        cc = c.strip().lower()
        if cc in {"team", "def", "defense", "opp", "opponent"}:
            team_col = c
            break
        if "team" in cc and team_col is None:
            team_col = c
    if team_col is None:
        team_col = df.columns[0]

    # position column
    pos_col = None
    for c in df.columns:
        if c.strip().lower() in {"pos", "position"}:
            pos_col = c
            break
    if pos_col is None:
        raise ValueError("Could not find POS/Position column in DvP CSV.")

    # stat columns
    stat_map = {}
    for c in df.columns:
        cc = c.strip().lower().replace(" ", "")
        if cc in {"pts", "points"}:
            stat_map["PTS"] = c
        elif cc in {"reb", "rebounds"}:
            stat_map["REB"] = c
        elif cc in {"ast", "assists"}:
            stat_map["AST"] = c
        elif cc in {"3pm", "fg3m", "3p", "3ptm"}:
            stat_map["FG3M"] = c

    out = df[[team_col, pos_col] + list(stat_map.values())].copy()
    out.columns = ["DEF_TEAM", "POS"] + list(stat_map.keys())

    out["DEF_TEAM"] = out["DEF_TEAM"].apply(normalize_team_abbrev)
    out["POS"] = out["POS"].astype(str).str.upper().str.strip()

    # to multipliers
    for s in ["PTS","REB","AST","FG3M"]:
        if s in out.columns:
            out[s] = pd.to_numeric(out[s], errors="coerce")
            def to_mult(v):
                if v is None or not np.isfinite(v):
                    return 1.0
                if 0.2 <= v <= 2.0:
                    return float(v)
                return 1.0 + float(v)/100.0
            out[s] = out[s].apply(to_mult)
        else:
            out[s] = 1.0

    return out

def dvp_multiplier(dvp_df: pd.DataFrame, opp_team: str, pos: str, stat: str) -> float:
    if dvp_df is None or dvp_df.empty:
        return 1.0
    opp_team = normalize_team_abbrev(opp_team)
    pos = str(pos).upper().strip()
    sub = dvp_df[(dvp_df["DEF_TEAM"] == opp_team) & (dvp_df["POS"] == pos)]
    if sub.empty or stat not in sub.columns:
        return 1.0
    v = safe_num(sub.iloc[0][stat], 1.0)
    return 1.0 if v <= 0 else float(v)


# ==========================
# MINUTES FIX (ONLY change requested)
# ==========================
def cap_and_redistribute_minutes(
    df: pd.DataFrame,
    team_col="Team",
    minutes_col="Minutes",
    out_bool_col="IsOut",
    max_minutes=34.0,
    team_total_minutes=240.0
) -> pd.DataFrame:
    """
    Cap minutes at max_minutes for active players, then redistribute removed minutes
    to active teammates under the cap. Also gently nudges team totals to 240 if far off.
    """
    df = df.copy()
    df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0.0)
    df[out_bool_col] = df[out_bool_col].fillna(False).astype(bool)

    for team, idx in df.groupby(team_col).groups.items():
        idx = list(idx)
        active = [i for i in idx if not bool(df.loc[i, out_bool_col])]
        if not active:
            continue

        # cap
        before = df.loc[active, minutes_col].copy()
        df.loc[active, minutes_col] = df.loc[active, minutes_col].clip(lower=0.0, upper=max_minutes)
        removed = float((before - df.loc[active, minutes_col]).clip(lower=0.0).sum())

        # redistribute removed
        if removed > 1e-6:
            remaining = removed
            for _ in range(12):
                if remaining <= 1e-6:
                    break
                receivers = [i for i in active if float(df.loc[i, minutes_col]) < max_minutes - 1e-6]
                if not receivers:
                    break
                weights = df.loc[receivers, minutes_col].clip(lower=1.0)
                weights = weights / float(weights.sum())
                add = weights * remaining
                new_vals = (df.loc[receivers, minutes_col] + add).clip(upper=max_minutes)
                actual = float((new_vals - df.loc[receivers, minutes_col]).sum())
                df.loc[receivers, minutes_col] = new_vals
                remaining -= actual

        # optional normalize team total minutes
        team_min = float(df.loc[active, minutes_col].sum())
        diff = team_total_minutes - team_min
        if abs(diff) > 3.0:
            if diff > 0:
                remaining = diff
                for _ in range(12):
                    if remaining <= 1e-6:
                        break
                    receivers = [i for i in active if float(df.loc[i, minutes_col]) < max_minutes - 1e-6]
                    if not receivers:
                        break
                    weights = df.loc[receivers, minutes_col].clip(lower=1.0)
                    weights = weights / float(weights.sum())
                    add = weights * remaining
                    new_vals = (df.loc[receivers, minutes_col] + add).clip(upper=max_minutes)
                    actual = float((new_vals - df.loc[receivers, minutes_col]).sum())
                    df.loc[receivers, minutes_col] = new_vals
                    remaining -= actual
            else:
                take = -diff
                donors = sorted(active, key=lambda i: float(df.loc[i, minutes_col]), reverse=True)
                for i2 in donors:
                    if take <= 1e-6:
                        break
                    can_take = max(0.0, float(df.loc[i2, minutes_col]))
                    d = min(can_take, take)
                    df.loc[i2, minutes_col] = float(df.loc[i2, minutes_col]) - d
                    take -= d

    return df


# ==========================
# PROJECTIONS
# ==========================
def build_projections(dk: pd.DataFrame, out_flags: dict, nba_df: pd.DataFrame, dvp_df: pd.DataFrame | None, progress_cb=None):
    df = dk.copy()
    df["Name_clean"] = df["Name"].apply(clean_name)
    df["IsOut"] = df["Name_clean"].apply(lambda x: bool(out_flags.get(x, False)))

    # Match NBA rows
    hits = []
    for j, r in df.iterrows():
        if progress_cb and j % 15 == 0:
            progress_cb(j, len(df), r["Name"])
        hits.append(match_nba_row(r["Name"], nba_df))

    df["NBA_Matched"] = [h is not None for h in hits]
    df["NBA_MIN"] = [safe_num(h["MIN"], np.nan) if h is not None else np.nan for h in hits]
    df["Minutes"] = df["NBA_MIN"].fillna(24.0)

    # Injury bumps: move OUT minutes to active teammates proportional to baseline minutes
    for team, idx in df.groupby("Team").groups.items():
        idx = list(idx)
        out_idx = [i for i in idx if bool(df.loc[i, "IsOut"])]
        act_idx = [i for i in idx if not bool(df.loc[i, "IsOut"])]

        if not out_idx or not act_idx:
            continue

        removed_min = float(df.loc[out_idx, "Minutes"].sum())
        if removed_min <= 1e-6:
            continue

        weights = df.loc[act_idx, "Minutes"].clip(lower=1.0)
        weights = weights / float(weights.sum())
        df.loc[act_idx, "Minutes"] = df.loc[act_idx, "Minutes"] + weights * removed_min
        df.loc[out_idx, "Minutes"] = 0.0

    # ✅ ONLY requested change: cap minutes at 34 and redistribute
    df = cap_and_redistribute_minutes(
        df,
        team_col="Team",
        minutes_col="Minutes",
        out_bool_col="IsOut",
        max_minutes=MAX_MINUTES,
        team_total_minutes=TEAM_TOTAL_MINUTES
    )

    # Project stats from season per-minute rates * minutes + DvP for PTS/REB/AST/FG3M
    df["PrimaryPos"] = df["Position"].apply(primary_pos)
    for stat in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
        df[stat] = 0.0

    df["Status"] = np.where(df["IsOut"], "OUT", np.where(df["NBA_Matched"], "OK", "ERR"))

    for i, r in df.iterrows():
        if r["IsOut"]:
            continue
        h = match_nba_row(r["Name"], nba_df)
        if h is None:
            continue
        season_min = safe_num(h["MIN"], 0.0)
        if season_min <= 0:
            continue

        mins = safe_num(r["Minutes"], 0.0)
        # base
        base = {}
        for stat in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
            per_min = safe_num(h[stat], 0.0) / season_min
            base[stat] = per_min * mins

        # DvP multipliers (only for these four)
        opp = r.get("Opp","")
        pos = r.get("PrimaryPos","")
        base["PTS"]  *= dvp_multiplier(dvp_df, opp, pos, "PTS")
        base["REB"]  *= dvp_multiplier(dvp_df, opp, pos, "REB")
        base["AST"]  *= dvp_multiplier(dvp_df, opp, pos, "AST")
        base["FG3M"] *= dvp_multiplier(dvp_df, opp, pos, "FG3M")

        for stat in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]:
            df.loc[i, stat] = base[stat]

    df["DK_FP"] = df.apply(lambda x: dk_fp(x["PTS"], x["REB"], x["AST"], x["STL"], x["BLK"], x["TOV"], x["FG3M"]), axis=1)
    df["Value"] = df.apply(lambda x: (x["DK_FP"] / (x["Salary"]/1000.0)) if safe_num(x["Salary"], 0) > 0 else 0.0, axis=1)

    for c in ["Minutes","PTS","REB","AST","STL","BLK","TOV","FG3M","DK_FP","Value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).round(2)

    return df


# ==========================
# OPTIMIZER with late swap locks
# ==========================
def optimize_dk_classic(df_pool: pd.DataFrame, locked_player_names_clean: set):
    if pulp is None:
        raise RuntimeError("PuLP not installed. Add 'pulp' to requirements.txt.")

    players = df_pool.reset_index(drop=True).copy()
    n = len(players)

    # decision vars: x[i, slot]
    x = {}
    for i in range(n):
        elig = eligible_slots(players.loc[i, "Position"])
        for slot in DK_SLOTS:
            if slot in elig:
                x[(i, slot)] = pulp.LpVariable(f"x_{i}_{slot}", lowBound=0, upBound=1, cat="Binary")

    prob = pulp.LpProblem("dk_classic", pulp.LpMaximize)
    prob += pulp.lpSum(x[(i, s)] * float(players.loc[i, "DK_FP"]) for (i, s) in x.keys())

    # salary cap
    prob += pulp.lpSum(x[(i, s)] * float(players.loc[i, "Salary"]) for (i, s) in x.keys()) <= SALARY_CAP

    # each slot exactly 1
    for slot in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for (i, s) in x.keys() if s == slot) == 1

    # each player at most once
    for i in range(n):
        prob += pulp.lpSum(x[(ii, s)] for (ii, s) in x.keys() if ii == i) <= 1

    # lock players (must be selected in some slot)
    if locked_player_names_clean:
        for i in range(n):
            if players.loc[i, "Name_clean"] in locked_player_names_clean:
                prob += pulp.lpSum(x[(ii, s)] for (ii, s) in x.keys() if ii == i) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Optimizer status: {pulp.LpStatus[prob.status]}")

    chosen = []
    for (i, slot), var in x.items():
        if var.value() == 1:
            row = players.loc[i].to_dict()
            row["Slot"] = slot
            chosen.append(row)

    out = pd.DataFrame(chosen)
    slot_order = {s: k for k, s in enumerate(DK_SLOTS)}
    out["SlotOrder"] = out["Slot"].map(slot_order)
    out = out.sort_values("SlotOrder").drop(columns=["SlotOrder"])
    return out, int(out["Salary"].sum()), float(out["DK_FP"].sum())


# ==========================
# SIDEBAR: Uploads + Gist
# ==========================
st.sidebar.header("Uploads")

dk_file = st.sidebar.file_uploader("DraftKings Salary CSV", type="csv")
dvp_file = st.sidebar.file_uploader("DvP CSV (HashtagBasketball export)", type="csv")

use_gist = st.sidebar.checkbox("Persist OUT + locks + projections (mobile/desktop)", value=True)

gist_id = None
gh_token = None
if use_gist:
    if "GIST_ID" in st.secrets and "GITHUB_TOKEN" in st.secrets:
        gist_id = st.secrets["GIST_ID"]
        gh_token = st.secrets["GITHUB_TOKEN"]
    else:
        st.sidebar.warning("Missing GIST_ID / GITHUB_TOKEN in secrets. Turning persistence off.")
        use_gist = False

if dk_file is None:
    st.info("Upload your DraftKings slate CSV to begin.")
    st.stop()


# ==========================
# LOAD DK SLATE
# ==========================
dk_text = dk_file.getvalue().decode("utf-8", errors="ignore")
dk_raw = pd.read_csv(StringIO(dk_text))

# standardize DK columns
if "Name" not in dk_raw.columns and "Name + ID" in dk_raw.columns:
    dk_raw["Name"] = dk_raw["Name + ID"].astype(str).str.replace(r"\s+$begin:math:text$\\d\+$end:math:text$$", "", regex=True)

if "Team" not in dk_raw.columns and "TeamAbbrev" in dk_raw.columns:
    dk_raw["Team"] = dk_raw["TeamAbbrev"]

if "Positions" in dk_raw.columns and "Position" not in dk_raw.columns:
    dk_raw["Position"] = dk_raw["Positions"]

# game info
game_col = None
for c in ["Game Info", "GameInfo", "Game"]:
    if c in dk_raw.columns:
        game_col = c
        break

required = ["Name", "Salary", "Team"]
missing = [c for c in required if c not in dk_raw.columns]
if missing:
    st.error(f"DK CSV missing required columns: {missing}")
    st.stop()

dk = pd.DataFrame({
    "Name": dk_raw["Name"].astype(str),
    "Salary": pd.to_numeric(dk_raw["Salary"], errors="coerce").fillna(0).astype(int),
    "Team": dk_raw["Team"].astype(str).apply(normalize_team_abbrev),
    "Position": dk_raw["Position"].astype(str) if "Position" in dk_raw.columns else "",
})

if game_col is not None:
    dk["GameInfo"] = dk_raw[game_col].astype(str)
    dk["Opp"] = dk.apply(lambda r: parse_opponent_from_game_info(r["Team"], r["GameInfo"]), axis=1)
else:
    dk["GameInfo"] = ""
    dk["Opp"] = ""

dk["Name_clean"] = dk["Name"].apply(clean_name)

teams_on_slate = sorted(dk["Team"].unique().tolist())


# ==========================
# LOAD DVP
# ==========================
dvp_df = None
if dvp_file is not None:
    try:
        dvp_df = load_dvp_csv(dvp_file.getvalue())
        st.sidebar.success("DvP loaded.")
    except Exception as e:
        st.sidebar.error(f"DvP load error: {e}")
        dvp_df = None
else:
    st.sidebar.info("DvP optional.")


# ==========================
# LOAD OUT + LOCKS (Gist)
# ==========================
out_flags = {}
locks_state = {"locked_teams": [], "locked_players": []}

if use_gist:
    txt_out = gist_read(gist_id, gh_token, GIST_OUT)
    if txt_out:
        try:
            out_flags = json.loads(txt_out)
        except Exception:
            out_flags = {}

    txt_locks = gist_read(gist_id, gh_token, GIST_LOCKS)
    if txt_locks:
        try:
            locks_state = json.loads(txt_locks)
        except Exception:
            locks_state = {"locked_teams": [], "locked_players": []}

if "out_flags" not in st.session_state:
    st.session_state["out_flags"] = dict(out_flags)

if "locked_teams" not in st.session_state:
    st.session_state["locked_teams"] = list(locks_state.get("locked_teams", []))

if "locked_players" not in st.session_state:
    st.session_state["locked_players"] = list(locks_state.get("locked_players", []))

if "last_lineup" not in st.session_state:
    st.session_state["last_lineup"] = None  # stored as list of clean names


# ==========================
# STEP A: OUT checkboxes
# ==========================
st.subheader("Step A — Slate (Mark OUT players)")

with st.expander("OUT checkboxes", expanded=True):
    cols = st.columns(3)
    slate_sorted = dk.sort_values(["Team","Salary"], ascending=[True, False]).reset_index(drop=True)
    for i, row in slate_sorted.iterrows():
        col = cols[i % 3]
        key = f"out_{row['Name_clean']}"
        default = bool(st.session_state["out_flags"].get(row["Name_clean"], False))
        val = col.checkbox(f"{row['Team']} — {row['Name']} (${row['Salary']})", value=default, key=key)
        st.session_state["out_flags"][row["Name_clean"]] = bool(val)

if use_gist:
    try:
        gist_write(gist_id, gh_token, {GIST_OUT: json.dumps(st.session_state["out_flags"])})
    except Exception as e:
        st.warning(f"Could not save OUT flags: {e}")


# ==========================
# STEP B: Run projections
# ==========================
st.subheader("Step B — Run Projections")

status_line = st.empty()
pbar = st.progress(0)

def progress_cb(i, n, name):
    pct = int((i / max(1, n)) * 100)
    status_line.write(f"Running season rates match… **{name}** ({i+1}/{n})")
    pbar.progress(min(100, pct))

run_proj = st.button("Run Projections")

proj_df = None

if run_proj:
    nba_df = nba_league_dash()
    with st.spinner("Building projections…"):
        proj_df = build_projections(dk, st.session_state["out_flags"], nba_df, dvp_df, progress_cb=progress_cb)

    pbar.progress(100)
    status_line.write("✅ Projections complete.")

    if use_gist:
        try:
            gist_write(gist_id, gh_token, {GIST_FINAL: proj_df.to_csv(index=False)})
        except Exception as e:
            st.warning(f"Could not save final.csv: {e}")

else:
    if use_gist:
        final_txt = gist_read(gist_id, gh_token, GIST_FINAL)
        if final_txt:
            try:
                proj_df = pd.read_csv(StringIO(final_txt))
                status_line.write("Loaded last saved projections from Gist.")
                pbar.progress(0)
            except Exception:
                proj_df = None

if proj_df is None:
    st.stop()

# Hide OUT from display but keep them for bumps
proj_df["Name_clean"] = proj_df["Name"].apply(clean_name)
display_df = proj_df[proj_df["Status"].astype(str).str.upper() != "OUT"].copy()

st.dataframe(
    display_df.sort_values(["Value","Salary"], ascending=[False, False])[
        ["Name","Team","Opp","Position","PrimaryPos","Salary","Minutes",
         "PTS","REB","AST","FG3M","STL","BLK","TOV","DK_FP","Value","Status"]
    ],
    use_container_width=True
)

# sanity check team minutes
with st.expander("Minutes sanity check (active only)", expanded=False):
    mins_check = display_df.groupby("Team")["Minutes"].sum().reset_index().sort_values("Minutes", ascending=False)
    st.dataframe(mins_check, use_container_width=True)

# DvP coverage check
if dvp_df is not None and not dvp_df.empty:
    with st.expander("DvP coverage check", expanded=False):
        opps = sorted(set(display_df["Opp"].astype(str).apply(normalize_team_abbrev)) - {""})
        dvp_teams = sorted(set(dvp_df["DEF_TEAM"].astype(str)))
        missing = sorted([t for t in opps if t not in dvp_teams])
        if missing:
            st.warning(f"DvP missing DEF teams: {missing} → multiplier defaults to 1.0")
        else:
            st.success("DvP covers all opponents in this slate.")


# ==========================
# STEP C: Late swap locks (restore)
# ==========================
st.subheader("Step C — Late Swap Locks")

lock_cols = st.columns([2, 3])
with lock_cols[0]:
    st.session_state["locked_teams"] = st.multiselect(
        "Lock started teams (won’t be swapped out)",
        options=teams_on_slate,
        default=[t for t in st.session_state["locked_teams"] if t in teams_on_slate],
    )

# player lock grid
lock_df = display_df[["Name","Team","Salary","DK_FP","Value","Status","Name_clean"]].copy()
lock_df["Locked"] = lock_df["Name_clean"].isin(set(st.session_state["locked_players"]))

edited = st.data_editor(
    lock_df.drop(columns=["Name_clean"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Locked": st.column_config.CheckboxColumn("Lock Player", help="Locked players are forced into the optimized lineup."),
    },
)

# rebuild locked players list from editor
locked_players_clean = set()
name_to_clean = dict(zip(lock_df["Name"], lock_df["Name_clean"]))
for _, r in edited.iterrows():
    if bool(r.get("Locked", False)):
        locked_players_clean.add(name_to_clean.get(r["Name"], clean_name(r["Name"])))

st.session_state["locked_players"] = sorted(list(locked_players_clean))

# persist locks
if use_gist:
    try:
        gist_write(
            gist_id,
            gh_token,
            {GIST_LOCKS: json.dumps({"locked_teams": st.session_state["locked_teams"],
                                    "locked_players": st.session_state["locked_players"]})}
        )
    except Exception as e:
        st.warning(f"Could not save locks: {e}")

# Team locks imply all previously-selected lineup players from those teams remain locked:
# We'll enforce team locks by converting them into locked players IF user has an existing lineup.
# If no existing lineup, team locks still help by locking any player you manually checked.
locked_teams = set(st.session_state["locked_teams"])


# ==========================
# STEP D: Optimizer (DK Classic)
# ==========================
st.subheader("Step D — Optimize Lineup (DK Classic)")

if pulp is None:
    st.error("Optimizer requires PuLP. Add `pulp` to requirements.txt and redeploy.")
    st.stop()

# Build candidate pool: OK only, not OUT/ERR
pool = proj_df[(proj_df["Status"].astype(str).str.upper() == "OK") & (proj_df["Salary"] > 0)].copy()
pool["Name_clean"] = pool["Name"].apply(clean_name)

# De-dupe players to avoid 3x LeBron issues
pool = pool.sort_values("DK_FP", ascending=False).drop_duplicates(subset=["Name_clean"]).copy()

# Apply team locks by forcing already-locked players from those teams (if present)
# (Team locks work best after you already have a lineup once.)
locked_from_team = set(pool.loc[pool["Team"].isin(locked_teams), "Name_clean"].tolist())
# We do NOT auto-lock entire team (that can force too many players). We keep team locks for “current lineup lock” behavior.
# So, only lock-team affects the "keep lineup stable" feature below.

keep_existing = st.checkbox("Late swap mode: keep previously-optimized lineup players from locked teams", value=True)

# If we already optimized once, and keep_existing is on:
# lock the subset of that lineup whose team is locked
if keep_existing and st.session_state["last_lineup"] is not None:
    last_lineup_clean = set(st.session_state["last_lineup"])
    last_lineup_rows = pool[pool["Name_clean"].isin(last_lineup_clean)]
    lock_these = set(last_lineup_rows[last_lineup_rows["Team"].isin(locked_teams)]["Name_clean"].tolist())
else:
    lock_these = set()

# final locked players = manual locks + late-swap locks
final_locked = set(st.session_state["locked_players"]) | lock_these

opt_btn = st.button("Optimize Lineup")

if opt_btn:
    try:
        lineup, tot_sal, tot_fp = optimize_dk_classic(pool, final_locked)
        st.success(f"Optimized — Salary ${tot_sal:,} | Proj DK FP {tot_fp:.2f}")

        st.dataframe(
            lineup[["Slot","Name","Team","Opp","Position","Salary","Minutes","PTS","REB","AST","FG3M","DK_FP","Value"]],
            use_container_width=True
        )

        # store last lineup for late swap mode
        st.session_state["last_lineup"] = lineup["Name"].apply(clean_name).tolist()

    except Exception as e:
        st.error(f"Optimizer error: {e}")
        st.write("Common causes:")
        st.write("- Too many locked players (forces impossible lineup)")
        st.write("- Locked players don't fit DK slot constraints")
        st.write("- Salary cap impossible with locks")
        st.write(f"Details: {e}")