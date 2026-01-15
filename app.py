import re
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

# Streamlit Cloud stability
API_TIMEOUT_SECONDS = 35
GAMELOG_RETRIES = 4
BUILD_THROTTLE_SECONDS = 0.15

# OUT teammate bumps
ENABLE_OUT_BUMPS = True
MIN_REDIS_TOP_N = 8
MIN_BENCH_FLOOR = 6.0
MAX_MINUTES_PLAYER = 40.0
MAX_MIN_INCREASE = 22.0

# Display
HIDE_OUT_PLAYERS_FROM_TABLE = True

# Persistence (upload on desktop, view on phone)
SAVED_CSV_PATH = Path("latest_dk_slate.csv")

st.set_page_config(
    page_title="DK NBA Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("DraftKings NBA Optimizer (Cloud-safe + OUT bumps)")

# ==========================
# DK scoring
# ==========================
def dk_fp(s):
    return round(
        float(s.get("PTS", 0))
        + 1.25 * float(s.get("REB", 0))
        + 1.5 * float(s.get("AST", 0))
        + 2.0 * float(s.get("STL", 0))
        + 2.0 * float(s.get("BLK", 0))
        - 0.5 * float(s.get("TOV", 0))
        + 0.5 * float(s.get("FG3M", 0)),
        2,
    )

# ==========================
# Helpers
# ==========================
def clean_name(s): 
    return " ".join(str(s).lower().replace(".", "").split())

def parse_minutes(s):
    s = str(s)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60.0

def dk_positions_list(p): 
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def slot_eligible(pos, slot):
    pos = set(pos)
    if slot in pos:
        return True
    if slot == "G":
        return bool(pos & {"PG", "SG"})
    if slot == "F":
        return bool(pos & {"SF", "PF"})
    if slot == "UTIL":
        return True
    return False

def parse_out_list(txt):
    if not txt:
        return set()
    return {clean_name(x) for x in txt.replace(",", "\n").split("\n") if x.strip()}

# ==========================
# NBA lookups
# ==========================
@st.cache_data(ttl=86400)
def player_id_map():
    return {clean_name(p["full_name"]): p["id"] for p in nba_players.get_players()}

def get_player_id(name):
    m = player_id_map()
    key = clean_name(name)
    if key in m:
        return m[key]
    raise KeyError(f"Player not found in nba_api static list: {name}")

# ==========================
# NBA gamelog (retry safe)
# ==========================
@st.cache_data(ttl=3600)
def gamelog(pid):
    last_err = None
    for i in range(GAMELOG_RETRIES):
        try:
            df = playergamelog.PlayerGameLog(
                player_id=pid,
                season=SEASON,
                timeout=API_TIMEOUT_SECONDS
            ).get_data_frames()[0]
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e
            time.sleep((1.6 ** i) + random.random())
    raise RuntimeError(f"NO_GAMELOG ({last_err})")

def project_player(name, n):
    gl = gamelog(get_player_id(name))
    gl = gl.copy()
    gl["MIN_float"] = gl["MIN"].apply(parse_minutes)
    gl = gl.head(int(n))

    # Avoid divide-by-zero
    tot_min = float(gl["MIN_float"].sum())
    if tot_min <= 0:
        raise RuntimeError("NO_MINUTES")

    mins = float(gl["MIN_float"].mean())
    stats = {}
    for c in STAT_COLS:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0)
        stats[c] = float(gl[c].sum()) / tot_min * mins

    return mins, stats

# ==========================
# Load DK CSV
# ==========================
def load_dk(file_or_path):
    df = pd.read_csv(file_or_path)

    required = ["Name", "Position", "Salary", "TeamAbbrev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DK CSV missing columns: {missing}")

    df["Positions"] = df["Position"].apply(dk_positions_list)
    df["Team"] = df["TeamAbbrev"].astype(str).str.upper()
    df["Name_clean"] = df["Name"].apply(clean_name)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

    return df[["Name", "Name_clean", "Team", "Positions", "Salary"]].copy()

# ==========================
# OUT bumps (minutes redistribution)
# ==========================
def apply_out_bumps(df, out_set):
    if not out_set or df is None or df.empty:
        return df

    df = df.copy()
    df["BaseMinutes"] = df["Minutes"]

    # per-minute baseline rates
    for c in STAT_COLS:
        df[f"PM_{c}"] = df[c] / df["Minutes"].replace(0, np.nan)
        df[f"PM_{c}"] = df[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # bump team by team
    for team in df[df["Name_clean"].isin(out_set)]["Team"].dropna().unique():
        out_players = df[(df["Team"] == team) & (df["Name_clean"].isin(out_set))]
        team_ok = df[(df["Team"] == team) & (~df["Name_clean"].isin(out_set))]

        missing_min = float(out_players["BaseMinutes"].sum())
        if missing_min <= 0 or team_ok.empty:
            continue

        # baseline weights by minutes (bench floor so backups can jump)
        weights = team_ok["BaseMinutes"].clip(lower=MIN_BENCH_FLOOR)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue
        weights = weights / wsum

        for i in team_ok.index:
            inc = min(float(weights.loc[i]) * missing_min, MAX_MIN_INCREASE)
            df.loc[i, "Minutes"] = min(float(df.loc[i, "Minutes"]) + inc, MAX_MINUTES_PLAYER)
            for c in STAT_COLS:
                df.loc[i, c] = float(df.loc[i, f"PM_{c}"]) * float(df.loc[i, "Minutes"])

    # recompute DK FP and value
    df["DK_FP"] = df.apply(dk_fp, axis=1)
    df["Value"] = df["DK_FP"] / (df["Salary"] / 1000.0)

    return df

# ==========================
# Sidebar (persistence enabled)
# ==========================
with st.sidebar:
    st.subheader("Upload DK CSV (desktop upload → phone view)")
    dk_file = st.file_uploader("DraftKings NBA CSV", type="csv")

    last_n = st.number_input("Recent games window (N)", 1, 30, int(st.session_state.get("last_n", DEFAULT_LAST_N)))
    st.session_state["last_n"] = int(last_n)

    out_text = st.text_area("Players OUT (one per line or comma-separated)", value=st.session_state.get("out_text", ""))
    st.session_state["out_text"] = out_text
    out_set = parse_out_list(out_text)

    if dk_file is not None:
        # Save uploaded CSV so it can be read from other devices/sessions
        SAVED_CSV_PATH.write_bytes(dk_file.getvalue())
        st.success("Saved slate! You can open this app on your phone and it will load the same CSV.")
    else:
        if SAVED_CSV_PATH.exists():
            st.info("No upload in this session. Loading the last saved slate (uploaded from another device).")
        else:
            st.info("Upload a DK CSV to begin.")

st.write(f"**Season:** {SEASON}  |  **Cap:** ${DK_SALARY_CAP:,}  |  **Slots:** {', '.join(DK_SLOTS)}")

# ==========================
# Load slate (file if present, else saved file)
# ==========================
if dk_file is not None:
    slate = load_dk(dk_file)
else:
    if SAVED_CSV_PATH.exists():
        slate = load_dk(SAVED_CSV_PATH)
    else:
        st.info("Open the sidebar (☰) and upload your DraftKings CSV.")
        st.stop()

# ==========================
# Build projections (with per-player progress)
# ==========================
st.subheader("Projections")

rows = []
total = len(slate)
progress = st.progress(0, text="Starting projections...")

for i, r in enumerate(slate.itertuples(index=False), start=1):
    progress.progress(
        int(i / max(total, 1) * 100),
        text=f"Running gamelog({r.Name}) ({i}/{total})"
    )
    try:
        mins, stats = project_player(r.Name, st.session_state["last_n"])
        row = {
            "Name": r.Name,
            "Name_clean": r.Name_clean,
            "Team": r.Team,
            "Positions": r.Positions,
            "Salary": float(r.Salary),
            "Minutes": float(mins),
        }
        for c in STAT_COLS:
            row[c] = float(stats.get(c, 0.0))
        row["DK_FP"] = dk_fp(row)
        row["Value"] = row["DK_FP"] / (row["Salary"] / 1000.0)
        rows.append(row)
    except Exception:
        # Skip failures in this simplified build; if you want ERR rows back, tell me and I’ll re-add them.
        pass

    time.sleep(BUILD_THROTTLE_SECONDS)

progress.empty()

proj_df = pd.DataFrame(rows)

# Apply teammate bumps based on OUT list (OUT players hidden from display, but used in bump math)
if ENABLE_OUT_BUMPS and out_set:
    proj_df = apply_out_bumps(proj_df, out_set)

# Hide OUT players from the table (but keep them in proj_df in case you later add exposure logic)
display_df = proj_df.copy()
if HIDE_OUT_PLAYERS_FROM_TABLE and out_set:
    display_df = display_df[~display_df["Name_clean"].isin(out_set)].copy()

display_df = display_df.rename(columns={"FG3M": "3PM"})

st.dataframe(
    display_df[
        ["Name", "Team", "Positions", "Salary", "Minutes",
         "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV",
         "DK_FP", "Value"]
    ].sort_values("DK_FP", ascending=False),
    use_container_width=True,
    hide_index=True
)

# ==========================
# Optimizer
# ==========================
st.subheader("Optimizer")

def optimize(pool):
    prob = LpProblem("DK", LpMaximize)
    x = {}

    for i, r in pool.iterrows():
        for s in DK_SLOTS:
            if slot_eligible(r["Positions"], s):
                x[(i, s)] = LpVariable(f"x_{i}_{s}", 0, 1, LpBinary)

    prob += lpSum(pool.loc[i, "DK_FP"] * x[(i, s)] for (i, s) in x)

    for s in DK_SLOTS:
        prob += lpSum(x[(i, s)] for i in pool.index if (i, s) in x) == 1

    prob += lpSum(pool.loc[i, "Salary"] * x[(i, s)] for (i, s) in x) <= DK_SALARY_CAP

    for i in pool.index:
        prob += lpSum(x[(i, s)] for s in DK_SLOTS if (i, s) in x) <= 1

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = []
    for s in DK_SLOTS:
        chosen = None
        for (i, sl) in x:
            if sl == s and x[(i, sl)].value() == 1:
                chosen = i
                break
        if chosen is not None:
            lineup.append(pool.loc[chosen])

    return pd.DataFrame(lineup)

if st.button("Optimize Lineup"):
    pool = display_df.dropna(subset=["Salary", "DK_FP"]).copy()
    pool = pool[pool["Salary"] > 0].copy()

    lineup = optimize(pool)

    st.subheader("Optimized Lineup")
    st.dataframe(
        lineup[["Name", "Team", "Positions", "Salary", "Minutes", "DK_FP", "Value"]].sort_values("Slot", ascending=True)
        if "Slot" in lineup.columns else lineup[["Name", "Team", "Positions", "Salary", "Minutes", "DK_FP", "Value"]],
        hide_index=True,
        use_container_width=True
    )

    st.metric("Total Salary", int(lineup["Salary"].sum()) if not lineup.empty else 0)
    st.metric("Total DK FP", float(lineup["DK_FP"].sum()) if not lineup.empty else 0.0)
