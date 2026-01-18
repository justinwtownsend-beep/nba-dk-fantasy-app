# ==========================
# DraftKings NBA Optimizer
# SPEED-FIRST STABLE MODE
# ==========================

import json
import time
import difflib
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

# ==========================
# CONFIG (SPEED FIRST)
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

API_TIMEOUT = 15      # ↓ was 28
GAMELOG_RETRIES = 3   # ↓ was 5
BUILD_SLEEP = 0.03

MAX_MINUTES = 40
BENCH_FLOOR = 6

GIST_SLATE = "slate.csv"
GIST_OUT   = "out.json"
GIST_BASE  = "base.csv"
GIST_FINAL = "final.csv"

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DraftKings NBA Optimizer — Speed Stable")

# ==========================
# HELPERS
# ==========================
def clean_name(s):
    return " ".join(str(s).lower().replace(".", "").replace(",", "").split())

def parse_minutes(x):
    s = str(x)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60

def dk_fp(r):
    fp = (
        r["PTS"]
        + 1.25 * r["REB"]
        + 1.5 * r["AST"]
        + 2 * r["STL"]
        + 2 * r["BLK"]
        - 0.5 * r["TOV"]
        + 0.5 * r["FG3M"]
    )
    cats = sum(r[c] >= 10 for c in ["PTS", "REB", "AST", "STL", "BLK"])
    if cats >= 2: fp += 1.5
    if cats >= 3: fp += 3
    return round(fp, 2)

def parse_positions(p):
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def eligible_for_slot(pos, slot):
    pos = set(pos)
    if slot in ["PG","SG","SF","PF","C"]: return slot in pos
    if slot == "G": return bool(pos & {"PG","SG"})
    if slot == "F": return bool(pos & {"SF","PF"})
    if slot == "UTIL": return True
    return False

# ==========================
# GIST
# ==========================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GIST_ID = st.secrets["GIST_ID"]

def gh(): return {"Authorization": f"token {GITHUB_TOKEN}"}

def gist_read(name):
    r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=gh())
    r.raise_for_status()
    g = r.json()
    if name not in g["files"]: return None
    f = g["files"][name]
    if not f["truncated"]: return f["content"]
    rr = requests.get(f["raw_url"])
    rr.raise_for_status()
    return rr.text

def gist_write(files):
    payload = {"files": {k: {"content": v} for k,v in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=gh(), json=payload)
    r.raise_for_status()

# ==========================
# NBA
# ==========================
@st.cache_data(ttl=3600)
def player_map():
    return {clean_name(p["full_name"]): p["id"] for p in nba_players.get_players()}

def player_id(name):
    key = clean_name(name)
    m = player_map()
    if key in m: return m[key]
    hits = nba_players.find_players_by_full_name(name)
    if hits: return hits[0]["id"]
    raise ValueError(f"Player not found: {name}")

@st.cache_data(ttl=3600)
def gamelog(pid):
    for _ in range(GAMELOG_RETRIES):
        try:
            return playergamelog.PlayerGameLog(
                player_id=pid,
                season=SEASON,
                timeout=API_TIMEOUT
            ).get_data_frames()[0]
        except:
            time.sleep(0.6)
    raise RuntimeError("Gamelog failed")

def project_base(name, n):
    pid = player_id(name)
    gl = gamelog(pid).head(n)
    gl["MIN_f"] = gl["MIN"].apply(parse_minutes)
    mins = gl["MIN_f"].mean()
    rates = {c: gl[c].sum()/gl["MIN_f"].sum() for c in STAT_COLS}
    stats = {c: round(rates[c]*mins,2) for c in STAT_COLS}
    return mins, stats

# ==========================
# LOAD SLATE
# ==========================
upload = st.sidebar.file_uploader("Upload DK CSV", type="csv")
if upload:
    text = upload.getvalue().decode()
    gist_write({GIST_SLATE: text})
else:
    text = gist_read(GIST_SLATE)
    if not text:
        st.info("Upload DraftKings CSV")
        st.stop()

df = pd.read_csv(StringIO(text))
slate = pd.DataFrame({
    "Name": df["Name"],
    "Salary": df["Salary"],
    "Team": df["TeamAbbrev"],
    "Positions": df["Position"].apply(parse_positions),
})
slate["Name_clean"] = slate["Name"].apply(clean_name)

# ==========================
# OUT SELECTION
# ==========================
saved_out = json.loads(gist_read(GIST_OUT) or "{}")
slate["OUT"] = slate["Name_clean"].map(lambda x: saved_out.get(x, False))

st.subheader("Step 1 — Mark OUT players")
edited = st.data_editor(
    slate[["OUT","Name","Team","Salary","Positions"]],
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name","Team","Salary","Positions"],
    hide_index=True,
    use_container_width=True
)

out_flags = {clean_name(r["Name"]): bool(r["OUT"]) for _,r in edited.iterrows()}
out_set = {k for k,v in out_flags.items() if v}

if st.button("Save OUT"):
    gist_write({GIST_OUT: json.dumps(out_flags)})

# ==========================
# STEP A — BASE
# ==========================
st.divider()
st.subheader("Step A — Build BASE (slow)")

if st.button("Build BASE"):
    rows=[]
    prog = st.progress(0)
    for i,r in slate.iterrows():
        prog.progress((i+1)/len(slate), f"Running gamelog({r.Name})")
        try:
            mins,stats = project_base(r.Name, DEFAULT_LAST_N)
            rows.append({**r,"Minutes":mins,**stats,"Status":"OK","Notes":""})
        except Exception as e:
            rows.append({**r,"Minutes":np.nan,**{c:np.nan for c in STAT_COLS},"Status":"ERR","Notes":str(e)})
        time.sleep(BUILD_SLEEP)
    base=pd.DataFrame(rows)
    base["DK_FP"]=base.apply(lambda r: dk_fp(r) if r.Status=="OK" else np.nan,axis=1)
    gist_write({GIST_BASE: base.to_csv(index=False)})
    st.success("Saved BASE")

# ==========================
# STEP B — APPLY OUT
# ==========================
st.divider()
st.subheader("Step B — Apply OUT (fast)")

if st.button("Apply OUT"):
    base=pd.read_csv(StringIO(gist_read(GIST_BASE)))
    base["Positions"]=base["Positions"].apply(eval)
    base["Minutes"]=pd.to_numeric(base["Minutes"],errors="coerce")

    base.loc[base["Name_clean"].isin(out_set),"Status"]="OUT"

    for c in STAT_COLS:
        base[f"PM_{c}"]=base[c]/base["Minutes"]

    for team in base.Team.unique():
        out_t=base[(base.Team==team)&(base.Status=="OUT")]
        ok_t=base[(base.Team==team)&(base.Status=="OK")]
        if out_t.empty or ok_t.empty: continue
        missing=out_t["Minutes"].sum()
        w=ok_t["Minutes"].clip(lower=BENCH_FLOOR)
        for idx in ok_t.index:
            base.loc[idx,"Minutes"]=min(base.loc[idx,"Minutes"]+missing*w.loc[idx]/w.sum(),MAX_MINUTES)

    for idx in base.index[base.Status=="OK"]:
        for c in STAT_COLS:
            base.loc[idx,c]=round(base.loc[idx,f"PM_{c}"]*base.loc[idx,"Minutes"],2)
        base.loc[idx,"DK_FP"]=dk_fp(base.loc[idx])

    final=base[(base.Status=="OK") & (~base.Name_clean.isin(out_set))]
    gist_write({GIST_FINAL: final.to_csv(index=False)})
    st.dataframe(final,use_container_width=True)

# ==========================
# OPTIMIZER (FIXED)
# ==========================
st.divider()
st.subheader("Optimizer")

if st.button("Optimize"):
    pool=pd.read_csv(StringIO(gist_read(GIST_FINAL)))
    pool["Positions"]=pool["Positions"].apply(eval)

    prob=LpProblem("DK",LpMaximize)
    x={}

    for i,r in pool.iterrows():
        for s in DK_SLOTS:
            if eligible_for_slot(r.Positions,s):
                x[(i,s)]=LpVariable(f"x_{i}_{s}",0,1,LpBinary)

    prob+=lpSum(pool.loc[i,"DK_FP"]*x[(i,s)] for (i,s) in x)

    for s in DK_SLOTS:
        prob+=lpSum(x[(i,s)] for i in pool.index if (i,s) in x)==1

    for i in pool.index:
        prob+=lpSum(x[(i,s)] for s in DK_SLOTS if (i,s) in x)<=1

    prob+=lpSum(pool.loc[i,"Salary"]*x[(i,s)] for (i,s) in x)<=DK_SALARY_CAP
    prob.solve(PULP_CBC_CMD(msg=False))

    lineup=[]
    for s in DK_SLOTS:
        for i in pool.index:
            if (i,s) in x and x[(i,s)].value()==1:
                lineup.append({"Slot":s,**pool.loc[i].to_dict()})
    st.dataframe(pd.DataFrame(lineup),use_container_width=True)
