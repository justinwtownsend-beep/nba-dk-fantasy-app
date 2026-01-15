import re
import time
import numpy as np
import pandas as pd
import streamlit as st

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
)

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

OFF_WINDOW_GAMES = 30
BUMP_CAP_LOW = 0.70
BUMP_CAP_HIGH = 1.40

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

st.set_page_config(page_title="DK NBA Optimizer", layout="wide")
st.title("DraftKings NBA Classic – Projections + Optimizer")

# ==========================
# DK scoring
# ==========================
def dk_fp(stats):
    pts, reb, ast, stl, blk, tov, tpm = (
        stats["PTS"], stats["REB"], stats["AST"],
        stats["STL"], stats["BLK"], stats["TOV"], stats["FG3M"]
    )
    fp = pts + 1.25*reb + 1.5*ast + 2*stl + 2*blk - 0.5*tov + 0.5*tpm
    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2: fp += 1.5
    if cats_10 >= 3: fp += 3.0
    return round(fp, 2)

# ==========================
# Helpers
# ==========================
def clean_name(s):
    return " ".join(s.lower().replace(".", "").split())

def parse_minutes(s):
    if ":" not in str(s): return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec)/60

def dk_positions_list(pos):
    return [p.strip().upper() for p in pos.split("/")]

def slot_eligible(pos_list, slot):
    if slot in ["PG","SG","SF","PF","C"]:
        return slot in pos_list
    if slot == "G": return any(p in pos_list for p in ["PG","SG"])
    if slot == "F": return any(p in pos_list for p in ["SF","PF"])
    return True  # UTIL

def parse_out_list(text):
    return {clean_name(x) for x in text.replace(",", "\n").split() if x.strip()}

def parse_game_info(s):
    m = re.search(r"([A-Z]{2,4})@([A-Z]{2,4})", s)
    return (m.group(1), m.group(2)) if m else (None, None)

# ==========================
# NBA lookups
# ==========================
@st.cache_data(ttl=86400)
def player_id_map():
    return {clean_name(p["full_name"]): p["id"] for p in nba_players.get_players()}

def get_player_id(name):
    return player_id_map()[clean_name(name)]

@st.cache_data(ttl=86400)
def team_map():
    return {t["abbreviation"]: t for t in nba_teams.get_teams()}

# ==========================
# NBA data
# ==========================
@st.cache_data(ttl=3600)
def gamelog(pid):
    return playergamelog.PlayerGameLog(player_id=pid, season=SEASON).get_data_frames()[0]

def per_min_rates(gl, n):
    gl = gl.head(n).copy()
    gl["MIN"] = gl["MIN"].apply(parse_minutes)
    gl = gl.dropna()
    tot = gl["MIN"].sum()
    rates = {c: gl[c].sum()/tot for c in STAT_COLS}
    return rates, gl["MIN"].mean()

@st.cache_data(ttl=3600)
def team_context():
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON, measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON, measure_type_detailed_defense="Opponent"
    ).get_data_frames()[0]
    return adv, opp

# ==========================
# Projection
# ==========================
def project_player(name, opp_abbrev, last_n):
    pid = get_player_id(name)
    gl = gamelog(pid)
    rates, mins = per_min_rates(gl, last_n)

    proj = {c: rates[c]*mins for c in STAT_COLS}
    fp = dk_fp(proj)
    return round(mins,2), fp

# ==========================
# DK CSV loader (YOUR HEADERS)
# ==========================
def read_dk_csv(file):
    df = pd.read_csv(file)

    out = pd.DataFrame({
        "Name": df["Name"],
        "Salary": df["Salary"],
        "Positions": df["Position"].apply(dk_positions_list),
        "Team": df["TeamAbbrev"],
        "GameInfo": df["Game Info"]
    })

    out["Name_clean"] = out["Name"].apply(clean_name)
    out["Opp"] = out["GameInfo"].apply(lambda x: parse_game_info(x)[1])
    return out

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    dk_file = st.file_uploader("Upload DK CSV", type="csv")
    last_n = st.number_input("Recent games window", 1, 30, DEFAULT_LAST_N)
    out_text = st.text_area("Players OUT")
    max_team = st.number_input("Max players per team", 0, 8, 4)

if not dk_file:
    st.stop()

slate = read_dk_csv(dk_file)
out_set = parse_out_list(out_text)

# ==========================
# Build projections
# ==========================
rows = []
for r in slate.itertuples():
    if r.Name_clean in out_set:
        continue
    try:
        mins, fp = project_player(r.Name, r.Opp, last_n)
        rows.append({
            "Name": r.Name,
            "Team": r.Team,
            "Positions": r.Positions,
            "Salary": r.Salary,
            "Minutes": mins,
            "DK_FP": fp,
            "Value": fp/(r.Salary/1000)
        })
    except:
        pass

proj_df = pd.DataFrame(rows).sort_values("DK_FP", ascending=False)

st.subheader("All Projections")
st.dataframe(proj_df, use_container_width=True)

# ==========================
# Optimizer
# ==========================
def optimize(pool):
    prob = LpProblem("DK", LpMaximize)
    x = {}

    for i, r in pool.iterrows():
        for s in DK_SLOTS:
            if slot_eligible(r.Positions, s):
                x[(i,s)] = LpVariable(f"x_{i}_{s}", 0, 1, LpBinary)

    prob += lpSum(r.DK_FP * x[(i,s)] for (i,s), r in [(k, pool.loc[k[0]]) for k in x])

    for s in DK_SLOTS:
        prob += lpSum(x[(i,s)] for (i,s) in x if s == s) == 1

    for i in pool.index:
        prob += lpSum(x[(i,s)] for s in DK_SLOTS if (i,s) in x) <= 1

    prob += lpSum(pool.loc[i].Salary * x[(i,s)] for (i,s) in x) <= DK_SALARY_CAP

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = []
    for (i,s), v in x.items():
        if v.value() == 1:
            r = pool.loc[i]
            lineup.append([s, r.Name, r.Team, r.Salary, r.DK_FP])

    return pd.DataFrame(lineup, columns=["Slot","Name","Team","Salary","DK_FP"])

if st.button("Optimize Lineup"):
    lineup = optimize(proj_df)
    st.subheader("Optimized Lineup")
    st.dataframe(lineup.sort_values("Slot"), use_container_width=True)
    st.metric("Total Salary", lineup.Salary.sum())
    st.metric("Total DK FP", lineup.DK_FP.sum())
