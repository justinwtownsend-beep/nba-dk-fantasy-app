import time
import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

SEASON = "2025-26"
DEFAULT_LAST_N = 10

st.set_page_config(page_title="NBA DK Fantasy Projector", layout="centered")

st.title("NBA DraftKings Fantasy Projector")

# -----------------------
# DK scoring
# -----------------------
def dk_fantasy_points(stats):
    pts = stats["PTS"]
    reb = stats["REB"]
    ast = stats["AST"]
    stl = stats["STL"]
    blk = stats["BLK"]
    tov = stats["TOV"]
    tpm = stats["FG3M"]

    fp = pts + 1.25*reb + 1.5*ast + 2*stl + 2*blk - 0.5*tov + 0.5*tpm
    cats = sum([pts>=10, reb>=10, ast>=10, stl>=10, blk>=10])
    if cats>=2: fp+=1.5
    if cats>=3: fp+=3
    return round(fp,2)

# -----------------------
# Helpers
# -----------------------
def get_player_id(name):
    return players.find_players_by_full_name(name)[0]["id"]

def get_team_id(name):
    name=name.lower()
    for t in teams.get_teams():
        if name in t["full_name"].lower() or name==t["nickname"].lower():
            return t["id"]

def parse_min(m):
    if ":" in str(m):
        x,y=m.split(":")
        return float(x)+float(y)/60
    return float(m)

def injury_mult(x):
    if x=="questionable": return 0.92,0.97
    if x=="limited": return 0.85,0.95
    if x=="out": return 0,0
    return 1,1

# -----------------------
# NBA data
# -----------------------
def get_player(player, n):
    gl=playergamelog.PlayerGameLog(player_id=get_player_id(player),season=SEASON).get_data_frames()[0].head(n)
    gl["MIN"]=gl["MIN"].apply(parse_min)
    stats=["PTS","REB","AST","STL","BLK","TOV","FG3M"]
    for s in stats: gl[s]=pd.to_numeric(gl[s])
    return gl

@st.cache_data(ttl=3600)
def get_teams():
    adv=leaguedashteamstats.LeagueDashTeamStats(season=SEASON,measure_type_detailed_defense="Advanced",per_mode_detailed="PerGame").get_data_frames()[0]
    opp=leaguedashteamstats.LeagueDashTeamStats(season=SEASON,measure_type_detailed_defense="Opponent",per_mode_detailed="PerGame").get_data_frames()[0]
    return adv,opp

# -----------------------
# UI
# -----------------------
away=st.text_input("Away Team")
home=st.text_input("Home Team")

player=st.text_input("Player Name")
side=st.selectbox("Player Team",["AWAY","HOME"])
inj=st.selectbox("Injury",["healthy","questionable","limited","out"])
games=st.number_input("Recent games",1,30,DEFAULT_LAST_N)

if st.button("Project"):
    adv,opp=get_teams()
    opp_team=home if side=="AWAY" else away
    opp_id=get_team_id(opp_team)

    gl=get_player(player,games)
    mins=gl["MIN"].mean()

    rates={c:gl[c].sum()/gl["MIN"].sum() for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M"]}
    min_mult,use_mult=injury_mult(inj)
    mins*=min_mult

    proj={k:rates[k]*mins for k in rates}

    opp_row=opp[opp["TEAM_ID"]==opp_id]
    for col,mapk in [("OPP_PTS","PTS"),("OPP_REB","REB"),("OPP_AST","AST"),("OPP_FG3M","FG3M"),("OPP_STL","STL"),("OPP_BLK","BLK"),("OPP_TOV","TOV")]:
        proj[mapk]*=float(opp_row[col])/opp[col].mean()

    proj["PTS"]*=use_mult
    proj["AST"]*=use_mult
    proj["FG3M"]*=use_mult

    for k in proj: proj[k]=round(float(proj[k]),2)

    st.write("Projected stat line",proj)
    st.metric("DK Fantasy Points",dk_fantasy_points(proj))
