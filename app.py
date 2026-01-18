# ==========================
# DraftKings NBA Optimizer
# FAST MODE (Streamlit Cloud Friendly)
# - Step A uses ONE league endpoint (fast)
# - Name matching fixed (diacritics like Dončić -> Doncic)
# - OUT bumps redistribute minutes and scale stats correctly
# - Missing players debug
# - Optimizer fixed (no duplicates)
# ==========================

import json
import difflib
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from nba_api.stats.endpoints import leaguedashplayerstats

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

API_TIMEOUT = 20
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
st.title("DraftKings NBA Optimizer — FAST Mode")

# ==========================
# HELPERS
# ==========================
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def deaccent(s: str) -> str:
    """Turn Dončić -> Doncic, Šarić -> Saric, Jokić -> Jokic, etc."""
    s = str(s)
    return (
        unicodedata.normalize("NFKD", s)
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

def parse_positions(p):
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def eligible_for_slot(pos_list, slot):
    pos = set(pos_list or [])
    if slot in ["PG", "SG", "SF", "PF", "C"]:
        return slot in pos
    if slot == "G":
        return bool(pos & {"PG", "SG"})
    if slot == "F":
        return bool(pos & {"SF", "PF"})
    if slot == "UTIL":
        return True
    return False

def dk_fp(r):
    fp = (
        float(r["PTS"])
        + 1.25 * float(r["REB"])
        + 1.5 * float(r["AST"])
        + 2.0 * float(r["STL"])
        + 2.0 * float(r["BLK"])
        - 0.5 * float(r["TOV"])
        + 0.5 * float(r["FG3M"])
    )
    cats = sum([float(r[c]) >= 10 for c in ["PTS", "REB", "AST", "STL", "BLK"]])
    if cats >= 2:
        fp += 1.5
    if cats >= 3:
        fp += 3.0
    return round(fp, 2)

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

def gist_read(name):
    g = gist()
    if name not in g.get("files", {}):
        return None
    f = g["files"][name]
    if not f.get("truncated"):
        return f.get("content")
    r = requests.get(f["raw_url"], timeout=25)
    r.raise_for_status()
    return r.text

def gist_write(files):
    payload = {"files": {k: {"content": v} for k, v in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=gh(), json=payload, timeout=25)
    r.raise_for_status()

# ==========================
# NBA (FAST: one endpoint)
# ==========================
@st.cache_data(ttl=900)
def league_stats_df():
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        timeout=API_TIMEOUT
    ).get_data_frames()[0]

    keep = ["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    df = df[keep].copy()
    df.columns = ["NBA_Name", "NBA_Team", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")

    return df

def match_player_to_nba(slate_name, nba_df):
    cn = clean_name(slate_name)
    sn = strip_suffix(slate_name)

    # exact clean
    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        return exact.iloc[0]

    # exact stripped (removes Jr/III etc.)
    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        return exact2.iloc[0]

    # last-name filtered fuzzy (much safer than fuzzy vs whole league)
    parts = sn.split()
    if parts:
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last]
        if not cand.empty:
            # choose best similarity
            best_row = None
            best_score = 0.0
            for _, row in cand.iterrows():
                score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                if score > best_score:
                    best_row = row
                    best_score = score
            if best_row is not None and best_score >= 0.75:
                return best_row

    # fallback: tight fuzzy on whole list (higher cutoff to avoid wrong matches)
    candidates = nba_df["NBA_Name_clean"].tolist()
    hit = difflib.get_close_matches(cn, candidates, n=1, cutoff=0.90)
    if hit:
        return nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]

    return None

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

if st.button("Save OUT"):
    gist_write({GIST_OUT: json.dumps(out_flags, indent=2)})
    st.success("Saved OUT players")

# ==========================
# STEP A — BUILD BASE (FAST)
# ==========================
st.divider()
st.subheader("Step A — Build BASE (FAST)")

if st.button("Build BASE (FAST)"):
    nba_df = league_stats_df()
    rows = []
    prog = st.progress(0, text="Loading league stats + mapping DK slate...")

    for i, r in slate.iterrows():
        prog.progress((i + 1) / len(slate), text=f"Mapping {r['Name']} ({i+1}/{len(slate)})")
        hit = match_player_to_nba(r["Name"], nba_df)
        if hit is None:
            row = {
                **r.to_dict(),
                "Minutes": np.nan,
                **{c: np.nan for c in STAT_COLS},
                "Status": "ERR",
                "Notes": "No match in league stats (name mismatch / not in NBA stats)"
            }
        else:
            mins = float(hit["MIN"])
            stats = {c: float(hit[c]) for c in STAT_COLS}
            row = {**r.to_dict(), "Minutes": mins, **stats, "Status": "OK", "Notes": ""}

        rows.append(row)

    base = pd.DataFrame(rows)
    base["DK_FP"] = base.apply(lambda rr: dk_fp(rr) if rr["Status"] == "OK" else np.nan, axis=1)
    gist_write({GIST_BASE: base.to_csv(index=False)})

    st.success("Saved BASE (FAST)")
    st.dataframe(base[["Name", "Team", "Salary", "Minutes", "DK_FP", "Status"]], use_container_width=True)

# ==========================
# STEP B — APPLY OUT + DEBUG
# ==========================
st.divider()
st.subheader("Step B — Apply OUT + Save FINAL")

if st.button("Apply OUT (fast)"):
    base_text = gist_read(GIST_BASE)
    if not base_text:
        st.error("No BASE found. Run Step A first.")
        st.stop()

    gist_write({GIST_OUT: json.dumps(out_flags, indent=2)})

    base = pd.read_csv(StringIO(base_text))
    base["Positions"] = base["Positions"].apply(eval)
    base["Minutes"] = pd.to_numeric(base["Minutes"], errors="coerce")
    base["Salary"] = pd.to_numeric(base["Salary"], errors="coerce")
    base["Status"] = base["Status"].astype(str)
    base["Notes"] = base.get("Notes", "").fillna("").astype(str)

    base["BumpNotes"] = ""

    # mark OUT
    base.loc[base["Name_clean"].isin(out_set), "Status"] = "OUT"

    # lock per-minute rates BEFORE changing minutes (OK only)
    for c in STAT_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        base[f"PM_{c}"] = np.where(
            (base["Status"] == "OK") & (base["Minutes"].fillna(0) > 0),
            base[c].fillna(0) / base["Minutes"].replace(0, np.nan),
            0.0
        )
        base[f"PM_{c}"] = base[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # redistribute minutes by team
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
            base.loc[idx, "BumpNotes"] = (base.loc[idx, "BumpNotes"] + f" MIN+{inc:.1f}").strip()

    # recompute stats
    for idx in base.index[base["Status"] == "OK"]:
        m = base.loc[idx, "Minutes"]
        if pd.isna(m) or float(m) <= 0:
            continue
        for c in STAT_COLS:
            base.loc[idx, c] = round(float(base.loc[idx, f"PM_{c}"]) * float(m), 2)
        base.loc[idx, "DK_FP"] = dk_fp(base.loc[idx])

    final = base[(base["Status"] == "OK") & (~base["Name_clean"].isin(out_set))].copy()
    gist_write({GIST_FINAL: final.to_csv(index=False)})

    st.success("Saved FINAL")
    show_cols = ["Name","Team","Positions","Salary","Minutes","PTS","REB","AST","FG3M","STL","BLK","TOV","DK_FP","BumpNotes"]
    st.dataframe(final[show_cols], use_container_width=True)

    # Debug missing
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
            miss_df = miss_df.sort_values(["Team","Status","Salary"], ascending=[True, True, False])
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

# ==========================
# OPTIMIZER (FIXED: no duplicates)
# ==========================
st.divider()
st.subheader("Optimizer (Fixed)")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.info("No FINAL saved yet. Run Step A then Step B.")
else:
    if st.button("Optimize"):
        pool = pd.read_csv(StringIO(final_text))
        pool["Positions"] = pool["Positions"].apply(eval)
        pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
        pool["DK_FP"] = pd.to_numeric(pool["DK_FP"], errors="coerce")
        pool = pool.dropna(subset=["Salary","DK_FP"]).copy()
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

        prob += lpSum(pool.loc[i, "DK_FP"] * x[(i, slot)] for (i, slot) in x)

        for slot in DK_SLOTS:
            prob += lpSum(x[(i, slot)] for i in pool.index if (i, slot) in x) == 1

        for i in pool.index:
            prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1

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
            lineup.append({"Slot": slot, **pool.loc[chosen].to_dict()})

        lineup_df = pd.DataFrame(lineup)
        st.dataframe(lineup_df, use_container_width=True)
        st.metric("Total Salary", int(lineup_df["Salary"].sum()))
        st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))
