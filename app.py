import re
import time
import random
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

API_TIMEOUT_SECONDS = 35
GAMELOG_RETRIES = 4
BUILD_THROTTLE_SECONDS = 0.15
RETRY_THROTTLE_SECONDS = 0.20

ENABLE_OUT_BUMPS = True

MIN_REDIS_TOP_N = 8
MIN_BENCH_FLOOR = 6.0
MAX_MINUTES_PLAYER = 40.0
MAX_MIN_INCREASE = 22.0

OFF_BUMP_STRENGTH = 0.22
OFF_BUMP_CAP = 1.25
OFF_BUMP_TOP_N = 6

HIDE_OUT_PLAYERS_FROM_TABLE = True

st.set_page_config(page_title="DK NBA Optimizer", layout="wide")
st.title("DraftKings NBA Optimizer (Cloud-safe + OUT bumps)")

# ==========================
# DK scoring
# ==========================
def dk_fp(s):
    return round(
        s["PTS"]
        + 1.25 * s["REB"]
        + 1.5 * s["AST"]
        + 2 * s["STL"]
        + 2 * s["BLK"]
        - 0.5 * s["TOV"]
        + 0.5 * s["FG3M"],
        2,
    )

# ==========================
# Helpers
# ==========================
def clean_name(s): return " ".join(str(s).lower().replace(".", "").split())

def parse_minutes(s):
    if ":" not in str(s): return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60

def dk_positions_list(p): return [x.strip().upper() for x in str(p).split("/")]

def slot_eligible(pos, slot):
    pos = set(pos)
    if slot in pos: return True
    if slot == "G": return bool(pos & {"PG","SG"})
    if slot == "F": return bool(pos & {"SF","PF"})
    if slot == "UTIL": return True
    return False

def parse_out_list(txt):
    if not txt: return set()
    return {clean_name(x) for x in txt.replace(",", "\n").split("\n") if x.strip()}

# ==========================
# NBA lookups
# ==========================
@st.cache_data(ttl=86400)
def player_id_map():
    return {clean_name(p["full_name"]): p["id"] for p in nba_players.get_players()}

def get_player_id(name):
    return player_id_map()[clean_name(name)]

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
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            time.sleep((1.6 ** i) + random.random())
    raise RuntimeError(f"NO_GAMELOG ({last_err})")

def project_player(name, n):
    gl = gamelog(get_player_id(name))
    gl["MIN_float"] = gl["MIN"].apply(parse_minutes)
    gl = gl.head(n)
    mins = gl["MIN_float"].mean()
    stats = {c: gl[c].sum() / gl["MIN_float"].sum() * mins for c in STAT_COLS}
    return mins, stats

# ==========================
# Load DK CSV
# ==========================
def load_dk(file):
    df = pd.read_csv(file)
    df["Positions"] = df["Position"].apply(dk_positions_list)
    df["Team"] = df["TeamAbbrev"]
    df["Name_clean"] = df["Name"].apply(clean_name)
    return df[["Name","Name_clean","Team","Positions","Salary"]]

# ==========================
# OUT bumps
# ==========================
def apply_out_bumps(df, out_set):
    if not out_set:
        return df

    df = df.copy()
    df["BaseMinutes"] = df["Minutes"]

    for c in STAT_COLS:
        df[f"PM_{c}"] = df[c] / df["Minutes"].replace(0, np.nan)

    for team in df[df["Name_clean"].isin(out_set)]["Team"].unique():
        out_players = df[(df["Team"] == team) & (df["Name_clean"].isin(out_set))]
        team_ok = df[(df["Team"] == team) & (~df["Name_clean"].isin(out_set))]

        missing_min = out_players["BaseMinutes"].sum()
        if missing_min <= 0:
            continue

        weights = team_ok["BaseMinutes"].clip(lower=MIN_BENCH_FLOOR)
        weights /= weights.sum()

        for i in team_ok.index:
            inc = min(weights.loc[i] * missing_min, MAX_MIN_INCREASE)
            df.loc[i, "Minutes"] = min(df.loc[i, "Minutes"] + inc, MAX_MINUTES_PLAYER)
            for c in STAT_COLS:
                df.loc[i, c] = df.loc[i, f"PM_{c}"] * df.loc[i, "Minutes"]

    df["DK_FP"] = df.apply(dk_fp, axis=1)
    df["Value"] = df["DK_FP"] / (df["Salary"] / 1000)
    return df

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    dk_file = st.file_uploader("Upload DK CSV", type="csv")
    last_n = st.number_input("Recent games", 1, 30, DEFAULT_LAST_N)
    out_text = st.text_area("Players OUT")
    out_set = parse_out_list(out_text)

if not dk_file:
    st.stop()

slate = load_dk(dk_file)

# ==========================
# Build projections (PROGRESS RESTORED)
# ==========================
rows = []
total = len(slate)
progress = st.progress(0, text="Starting projections...")

for i, r in enumerate(slate.itertuples(index=False), start=1):
    progress.progress(
        int(i / total * 100),
        text=f"Running gamelog({r.Name}) ({i}/{total})"
    )

    try:
        mins, stats = project_player(r.Name, last_n)
        row = {
            **r._asdict(),
            "Minutes": mins,
            **stats,
        }
        row["DK_FP"] = dk_fp(row)
        row["Value"] = row["DK_FP"] / (r.Salary / 1000)
        rows.append(row)
    except Exception:
        pass

    time.sleep(BUILD_THROTTLE_SECONDS)

progress.empty()

proj_df = pd.DataFrame(rows)

# Apply bumps
proj_df = apply_out_bumps(proj_df, out_set)

# ==========================
# DISPLAY (OUT hidden)
# ==========================
display_df = proj_df[~proj_df["Name_clean"].isin(out_set)].copy()
display_df = display_df.rename(columns={"FG3M": "3PM"})

st.subheader("Projections")
st.dataframe(
    display_df[
        ["Name","Team","Positions","Salary","Minutes",
         "PTS","REB","AST","3PM","STL","BLK","TOV",
         "DK_FP","Value"]
    ],
    use_container_width=True,
    hide_index=True
)

# ==========================
# OPTIMIZER
# ==========================
def optimize(pool):
    prob = LpProblem("DK", LpMaximize)
    x = {}

    for i, r in pool.iterrows():
        for s in DK_SLOTS:
            if slot_eligible(r.Positions, s):
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
        for (i, sl) in x:
            if sl == s and x[(i, sl)].value() == 1:
                lineup.append(pool.loc[i])

    return pd.DataFrame(lineup)

if st.button("Optimize Lineup"):
    lineup = optimize(display_df)
    st.subheader("Optimized Lineup")
    st.dataframe(
        lineup[["Name","Team","Salary","DK_FP"]],
        hide_index=True,
        use_container_width=True
    )
