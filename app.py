# ==========================
# DraftKings NBA Optimizer
# STABLE MODE (Minutes + Stats FIXED + Optimizer FIXED + Missing Players Debug)
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
# CONFIG
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

API_TIMEOUT = 25
GAMELOG_RETRIES = 3
BUILD_SLEEP = 0.05

MAX_MINUTES = 40
BENCH_FLOOR = 6

# Gist files
GIST_SLATE = "slate.csv"
GIST_OUT = "out.json"
GIST_BASE = "base.csv"
GIST_FINAL = "final.csv"

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DraftKings NBA Optimizer — Stable Mode")

# ==========================
# HELPERS
# ==========================
def clean_name(s: str) -> str:
    return " ".join(str(s).lower().replace(".", "").split())

def parse_minutes(x) -> float:
    s = str(x)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60

def dk_fp(r) -> float:
    fp = (
        float(r["PTS"]) +
        1.25 * float(r["REB"]) +
        1.5 * float(r["AST"]) +
        2.0 * float(r["STL"]) +
        2.0 * float(r["BLK"]) -
        0.5 * float(r["TOV"]) +
        0.5 * float(r["FG3M"])
    )
    cats = sum([float(r[c]) >= 10 for c in ["PTS", "REB", "AST", "STL", "BLK"]])
    if cats >= 2:
        fp += 1.5
    if cats >= 3:
        fp += 3.0
    return round(fp, 2)

def parse_positions(p):
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def eligible_for_slot(pos_list, slot: str) -> bool:
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

# ==========================
# GIST
# ==========================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GIST_ID = st.secrets["GIST_ID"]

def gh():
    return {"Authorization": f"token {GITHUB_TOKEN}"}

def gist():
    r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=gh(), timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read(name: str):
    g = gist()
    if name not in g.get("files", {}):
        return None
    f = g["files"][name]
    if not f.get("truncated"):
        return f.get("content")
    r = requests.get(f["raw_url"], timeout=25)
    r.raise_for_status()
    return r.text

def gist_write(files: dict):
    payload = {"files": {k: {"content": v} for k, v in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=gh(), json=payload, timeout=25)
    r.raise_for_status()

# ==========================
# NBA
# ==========================
@st.cache_data(ttl=3600)
def player_map():
    return {clean_name(p["full_name"]): int(p["id"]) for p in nba_players.get_players()}

def player_id(name: str) -> int:
    key = clean_name(name)
    m = player_map()
    if key in m:
        return m[key]

    hits = nba_players.find_players_by_full_name(name)
    if hits:
        return int(hits[0]["id"])

    last = str(name).split()[-1]
    hits = nba_players.find_players_by_last_name(last) or []
    best, best_score = None, 0.0
    for h in hits:
        score = difflib.SequenceMatcher(None, key, clean_name(h.get("full_name", ""))).ratio()
        if score > best_score:
            best, best_score = h, score
    if best and best_score > 0.55:
        return int(best["id"])

    raise ValueError(f"Player not found: {name}")

@st.cache_data(ttl=3600)
def gamelog(pid: int) -> pd.DataFrame:
    last_err = None
    for _ in range(GAMELOG_RETRIES):
        try:
            return playergamelog.PlayerGameLog(
                player_id=pid,
                season=SEASON,
                timeout=API_TIMEOUT
            ).get_data_frames()[0]
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
    raise RuntimeError(f"Gamelog failed: {last_err}")

def project_base(name: str, n: int):
    pid = player_id(name)
    gl = gamelog(pid).head(int(n)).copy()
    gl["MIN_f"] = gl["MIN"].apply(parse_minutes)

    total_min = float(gl["MIN_f"].sum())
    if total_min <= 0:
        raise RuntimeError("NO_MINUTES_IN_SAMPLE")

    mins = float(gl["MIN_f"].mean())
    rates = {c: float(gl[c].sum()) / total_min for c in STAT_COLS}
    stats = {c: round(rates[c] * mins, 2) for c in STAT_COLS}
    return round(mins, 2), stats

# ==========================
# LOAD SLATE
# ==========================
upload = st.sidebar.file_uploader("Upload DK CSV", type="csv")
if upload:
    text = upload.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_SLATE: text})
else:
    text = gist_read(GIST_SLATE)
    if not text:
        st.info("Upload a DraftKings CSV to begin.")
        st.stop()

df = pd.read_csv(StringIO(text))
required_cols = ["Name", "Salary", "TeamAbbrev", "Position"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"DK CSV missing columns: {missing_cols}")
    st.stop()

slate = pd.DataFrame({
    "Name": df["Name"].astype(str),
    "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
    "Team": df["TeamAbbrev"].astype(str).str.upper(),
    "Positions": df["Position"].astype(str).apply(parse_positions),
})
slate["Name_clean"] = slate["Name"].apply(clean_name)

# ==========================
# OUT CHECKBOXES
# ==========================
saved_out = {}
try:
    saved_out = json.loads(gist_read(GIST_OUT) or "{}")
except Exception:
    saved_out = {}

slate["OUT"] = slate["Name_clean"].map(lambda x: bool(saved_out.get(x, False)))

st.subheader("Step 1 — Mark OUT players")
edited = st.data_editor(
    slate[["OUT", "Name", "Team", "Salary", "Positions"]],
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name", "Team", "Salary", "Positions"],
    use_container_width=True,
    hide_index=True,
)

out_flags = {clean_name(r["Name"]): bool(r["OUT"]) for _, r in edited.iterrows()}
out_set = {k for k, v in out_flags.items() if v}

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Save OUT"):
        gist_write({GIST_OUT: json.dumps(out_flags, indent=2)})
        st.success("Saved OUT players")
with c2:
    st.caption(f"OUT selected: **{len(out_set)}**")

# ==========================
# STEP A — BASE
# ==========================
st.divider()
st.subheader("Step A — Build BASE (slow)")
last_n = st.number_input("Recent games (N)", 1, 20, DEFAULT_LAST_N)

if st.button("Build BASE"):
    rows = []
    prog = st.progress(0, text="Starting BASE build...")
    for i, r in slate.iterrows():
        prog.progress((i + 1) / len(slate), text=f"Running gamelog({r.Name}) ({i+1}/{len(slate)})")
        try:
            mins, stats = project_base(r.Name, int(last_n))
            row = {**r.to_dict(), "Minutes": mins, **stats, "Status": "OK", "Notes": ""}
        except Exception as e:
            row = {**r.to_dict(), "Minutes": np.nan, **{c: np.nan for c in STAT_COLS}, "Status": "ERR", "Notes": str(e)[:180]}
        rows.append(row)
        time.sleep(BUILD_SLEEP)

    base = pd.DataFrame(rows)
    base["DK_FP"] = base.apply(lambda rr: dk_fp(rr) if rr["Status"] == "OK" else np.nan, axis=1)
    gist_write({GIST_BASE: base.to_csv(index=False)})
    st.success("Saved BASE")

# ==========================
# STEP B — APPLY OUT (FAST) + DEBUG MISSING PLAYERS
# ==========================
st.divider()
st.subheader("Step B — Apply OUT + Save FINAL (fast)")

if st.button("Apply OUT (fast)"):
    base_text = gist_read(GIST_BASE)
    if not base_text:
        st.error("No BASE found. Run Step A first.")
        st.stop()

    # always persist OUT selections with Step B
    gist_write({GIST_OUT: json.dumps(out_flags, indent=2)})

    base = pd.read_csv(StringIO(base_text))

    # restore list from string
    base["Positions"] = base["Positions"].apply(eval)
    base["Minutes"] = pd.to_numeric(base["Minutes"], errors="coerce")
    base["Salary"] = pd.to_numeric(base["Salary"], errors="coerce")
    base["Notes"] = ""
    base["Status"] = base["Status"].astype(str)
    base["Name_clean"] = base.get("Name_clean", base["Name"].apply(clean_name))

    # mark OUT
    base.loc[base["Name_clean"].isin(out_set), "Status"] = "OUT"

    # lock per-minute rates BEFORE changing minutes
    for c in STAT_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        base[f"PM_{c}"] = np.where(
            (base["Status"] == "OK") & (base["Minutes"].fillna(0) > 0),
            base[c].fillna(0) / base["Minutes"].replace(0, np.nan),
            0.0
        )
        base[f"PM_{c}"] = base[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # redistribute minutes by team (multi-out safe)
    for team in base["Team"].dropna().unique():
        out_t = base[(base["Team"] == team) & (base["Status"] == "OUT")]
        ok_t = base[(base["Team"] == team) & (base["Status"] == "OK")]
        if out_t.empty or ok_t.empty:
            continue

        missing = float(out_t["Minutes"].fillna(0).sum())
        if missing <= 0:
            continue

        weights = ok_t["Minutes"].fillna(0).clip(lower=BENCH_FLOOR)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue

        for idx in ok_t.index:
            inc = missing * float(weights.loc[idx]) / wsum
            base.loc[idx, "Minutes"] = min(float(base.loc[idx, "Minutes"]) + inc, MAX_MINUTES)
            base.loc[idx, "Notes"] = (base.loc[idx, "Notes"] + f" MIN+{inc:.1f}").strip()

    # recompute stats from locked per-minute rates
    for idx in base.index[base["Status"] == "OK"]:
        m = base.loc[idx, "Minutes"]
        if pd.isna(m) or float(m) <= 0:
            continue
        for c in STAT_COLS:
            base.loc[idx, c] = round(float(base.loc[idx, f"PM_{c}"]) * float(m), 2)
        base.loc[idx, "DK_FP"] = dk_fp(base.loc[idx])

    # FINAL = OK and not OUT
    final = base[(base["Status"] == "OK") & (~base["Name_clean"].isin(out_set))].copy()
    gist_write({GIST_FINAL: final.to_csv(index=False)})

    st.success("Saved FINAL")
    st.dataframe(final, use_container_width=True)

    # -------- Missing players debug (players on slate but not in final) --------
    final_names = set(final["Name_clean"].astype(str).tolist())
    slate_names = set(slate["Name_clean"].astype(str).tolist())
    missing_from_final = sorted(list(slate_names - final_names))

    with st.expander(f"Debug: Players missing from projections ({len(missing_from_final)})"):
        if len(missing_from_final) == 0:
            st.write("None 🎉")
        else:
            miss_df = base[base["Name_clean"].isin(missing_from_final)][
                ["Name", "Team", "Salary", "Status", "Notes"]
            ].copy()
            miss_df = miss_df.sort_values(["Team", "Status", "Salary"], ascending=[True, True, False])
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

# ==========================
# OPTIMIZER (FIXED: no duplicates)
# ==========================
st.divider()
st.subheader("Optimizer (Fixed)")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.info("No FINAL projections saved yet. Run Step A then Step B.")
else:
    if st.button("Optimize"):
        pool = pd.read_csv(StringIO(final_text))
        pool["Positions"] = pool["Positions"].apply(eval)
        pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
        pool["DK_FP"] = pd.to_numeric(pool["DK_FP"], errors="coerce")
        pool["Name_clean"] = pool.get("Name_clean", pool["Name"].apply(clean_name))

        pool = pool.dropna(subset=["Salary", "DK_FP"]).copy()
        pool = pool[pool["Salary"] > 0].copy()

        if pool.empty:
            st.error("No valid players in pool to optimize.")
            st.stop()

        prob = LpProblem("DK", LpMaximize)

        x = {}
        for i, r in pool.iterrows():
            for slot in DK_SLOTS:
                if eligible_for_slot(r["Positions"], slot):
                    x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

        # objective: maximize projected DK points
        prob += lpSum(pool.loc[i, "DK_FP"] * x[(i, slot)] for (i, slot) in x)

        # exactly one player per slot
        for slot in DK_SLOTS:
            prob += lpSum(x[(i, slot)] for i in pool.index if (i, slot) in x) == 1

        # each player can be used at most once across all slots (prevents dupes)
        for i in pool.index:
            prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1

        # salary cap
        prob += lpSum(pool.loc[i, "Salary"] * x[(i, slot)] for (i, slot) in x) <= DK_SALARY_CAP

        prob.solve(PULP_CBC_CMD(msg=False))

        lineup = []
        for slot in DK_SLOTS:
            chosen = None
            for i in pool.index:
                if (i, slot) in x and x[(i, slot)].value() == 1:
                    chosen = i
                    break
            if chosen is None:
                st.error("No feasible lineup found (pool too small / slot constraints).")
                st.stop()
            row = pool.loc[chosen].to_dict()
            lineup.append({"Slot": slot, **row})

        lineup_df = pd.DataFrame(lineup)
        st.dataframe(lineup_df, use_container_width=True)
        st.metric("Total Salary", int(lineup_df["Salary"].sum()))
        st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))
