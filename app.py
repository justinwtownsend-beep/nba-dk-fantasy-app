# ==========================
# DraftKings NBA Optimizer
# FAST + Recency (Top Salaries) + Manual Hashtag DvP Upload
# ==========================

import json
import difflib
import unicodedata
import time
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog


# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

# league endpoint (fast)
LEAGUE_TIMEOUT = 20

# gamelog endpoint (slow-ish, but only for top N)
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

MAX_MINUTES = 40
BENCH_FLOOR = 6

# Recency blend weights
MIN_REC_W = 0.70      # minutes: favor recent
PM_REC_W = 0.30       # per-minute: mostly season

# Opponent DvP caps (keep realistic)
DVP_CAP_LOW = 0.92
DVP_CAP_HIGH = 1.08

# Gist files
GIST_SLATE = "slate.csv"
GIST_DVP = "dvp.csv"
GIST_OUT = "out.json"
GIST_BASE = "base.csv"
GIST_FINAL = "final.csv"

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DraftKings NBA Optimizer — Fast + Recency + Manual DvP")

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

def parse_positions(p):
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def primary_pos(pos_list):
    # pick a consistent "primary" for DvP lookup
    # DK uses multi-position; we just take the first listed
    if not pos_list:
        return None
    return str(pos_list[0]).upper()

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

def parse_minutes_min(x):
    s = str(x)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# DK "Game Info": "LAL@BOS 07:30PM ET"
def parse_opponent_from_gameinfo(team_abbrev: str, game_info: str):
    if not isinstance(game_info, str):
        return None
    gi = game_info.strip().upper()
    if "@" not in gi:
        return None
    head = gi.split()[0]  # LAL@BOS
    if "@" not in head:
        return None
    away, home = head.split("@", 1)
    if team_abbrev == away:
        return home
    if team_abbrev == home:
        return away
    return None

def find_col(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    # fuzzy
    best = None
    best_score = 0
    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            score = difflib.SequenceMatcher(None, cl, cand.lower().strip()).ratio()
            if score > best_score:
                best_score = score
                best = c
    return best

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
# NBA
# ==========================
@st.cache_data(ttl=900)
def league_player_df():
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        timeout=LEAGUE_TIMEOUT
    ).get_data_frames()[0]

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    df = df[keep].copy()
    df.columns = ["PLAYER_ID", "NBA_Name", "NBA_Team", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_player_to_nba(slate_name, nba_df):
    cn = clean_name(slate_name)
    sn = strip_suffix(slate_name)

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
                    best_row, best_score = row, score
            if best_row is not None and best_score >= 0.75:
                return best_row

    candidates = nba_df["NBA_Name_clean"].tolist()
    hit = difflib.get_close_matches(cn, candidates, n=1, cutoff=0.90)
    if hit:
        return nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]
    return None

def gamelog_recent(pid: int, last_n: int):
    last_err = None
    for attempt in range(1, GAMELOG_RETRIES + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=int(pid),
                season=SEASON,
                timeout=GAMELOG_TIMEOUT
            ).get_data_frames()[0]
            gl = gl.head(int(last_n)).copy()
            if gl.empty:
                raise RuntimeError("EMPTY_GAMELOG")
            gl["MIN_f"] = gl["MIN"].apply(parse_minutes_min)
            total_min = float(gl["MIN_f"].sum())
            if total_min <= 0:
                raise RuntimeError("NO_MINUTES_IN_SAMPLE")
            rec_min = float(gl["MIN_f"].mean())
            rec_rates = {c: float(gl[c].sum()) / total_min for c in STAT_COLS}
            return rec_min, rec_rates
        except Exception as e:
            last_err = str(e)
            time.sleep(0.4 * attempt)
    raise RuntimeError(f"RECENT_GAMELOG_FAIL: {last_err}")

# ==========================
# UPLOADS
# ==========================
st.sidebar.subheader("Uploads")

# DK slate upload (saved)
upload_slate = st.sidebar.file_uploader("Upload DK Slate CSV", type="csv")
if upload_slate:
    slate_text = upload_slate.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_SLATE: slate_text})
else:
    slate_text = gist_read(GIST_SLATE)
    if not slate_text:
        st.info("Upload DK Slate CSV in sidebar to begin.")
        st.stop()

# DVP upload (saved)
upload_dvp = st.sidebar.file_uploader("Upload Hashtag DvP CSV", type="csv")
if upload_dvp:
    dvp_text = upload_dvp.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_DVP: dvp_text})
else:
    dvp_text = gist_read(GIST_DVP)
    if not dvp_text:
        st.warning("Upload Hashtag DvP CSV (bottom table) to apply opponent adjustments.")
        dvp_text = None

# ==========================
# LOAD SLATE
# ==========================
df = pd.read_csv(StringIO(slate_text))
required_cols = ["Name", "Salary", "TeamAbbrev", "Position"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"DK CSV missing columns: {missing_cols}")
    st.stop()

game_info_col = None
for cand in ["Game Info", "GameInfo", "Game_Info", "Game"]:
    if cand in df.columns:
        game_info_col = cand
        break

slate = pd.DataFrame({
    "Name": df["Name"].astype(str),
    "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
    "Team": df["TeamAbbrev"].astype(str).str.upper(),
    "Positions": df["Position"].astype(str).apply(parse_positions),
})
slate["Name_clean"] = slate["Name"].apply(clean_name)
slate["PrimaryPos"] = slate["Positions"].apply(primary_pos)

if game_info_col:
    slate["GameInfo"] = df[game_info_col].astype(str)
else:
    slate["GameInfo"] = ""

slate["Opp"] = slate.apply(lambda r: parse_opponent_from_gameinfo(r["Team"], r["GameInfo"]), axis=1)

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
    slate[["OUT", "Name", "Team", "Opp", "PrimaryPos", "Salary", "Positions"]],
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name", "Team", "Opp", "PrimaryPos", "Salary", "Positions"],
    use_container_width=True,
    hide_index=True,
)

out_flags = {clean_name(r["Name"]): bool(r["OUT"]) for _, r in edited.iterrows()}
out_set = {k for k, v in out_flags.items() if v}

if st.button("Save OUT"):
    gist_write({GIST_OUT: json.dumps(out_flags, indent=2)})
    st.success("Saved OUT players")

# ==========================
# RECENCY SETTINGS
# ==========================
st.sidebar.markdown("---")
st.sidebar.subheader("Recency Settings")
use_recency = st.sidebar.checkbox("Use recency blend (top salaries)", value=True)
top_n = st.sidebar.slider("Top N salaries to recency-blend", 0, 60, 25, 5)
last_n_games = st.sidebar.slider("Recent games (N)", 3, 15, 10, 1)

# ==========================
# LOAD DVP
# ==========================
def load_dvp(dvp_text: str):
    if not dvp_text:
        return None, None

    dvp = pd.read_csv(StringIO(dvp_text))
    # try to find key columns
    team_col = find_col(dvp, ["team", "tm", "opp"])
    pos_col = find_col(dvp, ["pos", "position"])
    if team_col is None or pos_col is None:
        return None, f"DvP CSV missing Team/Position columns. Found columns: {list(dvp.columns)}"

    # normalize
    dvp["TEAM"] = dvp[team_col].astype(str).str.upper().str.strip()
    dvp["POS"] = dvp[pos_col].astype(str).str.upper().str.strip()

    # stat columns (common in Hashtag table exports)
    col_pts = find_col(dvp, ["pts", "points"])
    col_reb = find_col(dvp, ["reb", "trb", "rebounds"])
    col_ast = find_col(dvp, ["ast", "assists"])
    col_3pm = find_col(dvp, ["3pm", "3p", "fg3m", "3pm/g"])
    col_stl = find_col(dvp, ["stl", "steals"])
    col_blk = find_col(dvp, ["blk", "blocks"])
    col_tov = find_col(dvp, ["to", "tov", "turnovers"])

    needed = [col_pts, col_reb, col_ast, col_3pm, col_stl, col_blk, col_tov]
    if any(c is None for c in needed):
        return None, f"DvP CSV missing some stat columns. Columns found: {list(dvp.columns)}"

    # keep & numeric
    out = dvp[["TEAM", "POS", col_pts, col_reb, col_ast, col_3pm, col_stl, col_blk, col_tov]].copy()
    out.columns = ["TEAM", "POS", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
    for c in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["PTS","REB","AST","FG3M","STL","BLK","TOV"]).copy()

    # league avg by POS for multipliers
    league_avg = out.groupby("POS")[["PTS","REB","AST","FG3M","STL","BLK","TOV"]].mean().reset_index()
    return (out, league_avg), None

dvp_pack = None
dvp_err = None
if dvp_text:
    dvp_pack, dvp_err = load_dvp(dvp_text)
    if dvp_err:
        st.warning(dvp_err)

# ==========================
# STEP A — BUILD BASE
# ==========================
st.divider()
st.subheader("Step A — Build BASE")

if st.button("Build BASE"):
    nba_df = league_player_df()
    rows = []

    top_salary_names = set()
    if use_recency and top_n > 0:
        tmp = slate.dropna(subset=["Salary"]).sort_values("Salary", ascending=False).head(int(top_n))
        top_salary_names = set(tmp["Name_clean"].tolist())

    prog = st.progress(0, text="Mapping DK slate to league stats...")
    for i, r in slate.iterrows():
        prog.progress((i + 1) / len(slate), text=f"Mapping {r['Name']} ({i+1}/{len(slate)})")
        hit = match_player_to_nba(r["Name"], nba_df)

        if hit is None:
            row = {**r.to_dict(),
                   "Minutes": np.nan, **{c: np.nan for c in STAT_COLS},
                   "Status": "ERR",
                   "Notes": "No match in league stats (name mismatch)"}
            rows.append(row)
            continue

        season_min = float(hit["MIN"])
        season_stats = {c: float(hit[c]) for c in STAT_COLS}

        notes = ""
        mins = season_min
        stats = season_stats

        if use_recency and (r["Name_clean"] in top_salary_names):
            try:
                rec_min, rec_rates = gamelog_recent(int(hit["PLAYER_ID"]), int(last_n_games))
                season_rates = {c: (season_stats[c] / season_min if season_min > 0 else 0.0) for c in STAT_COLS}
                mins = MIN_REC_W * rec_min + (1 - MIN_REC_W) * season_min
                blended_rates = {c: PM_REC_W * rec_rates[c] + (1 - PM_REC_W) * season_rates[c] for c in STAT_COLS}
                stats = {c: round(blended_rates[c] * mins, 2) for c in STAT_COLS}
                notes = f"RECENCY({last_n_games})"
            except Exception as e:
                notes = f"RECENCY_FAIL: {str(e)[:80]}"

        row = {**r.to_dict(), "Minutes": round(float(mins), 2), **stats, "Status": "OK", "Notes": notes}
        rows.append(row)

    base = pd.DataFrame(rows)
    base["DK_FP"] = base.apply(lambda rr: dk_fp(rr) if rr["Status"] == "OK" else np.nan, axis=1)
    gist_write({GIST_BASE: base.to_csv(index=False)})

    st.success("Saved BASE")
    st.dataframe(base[["Name","Team","Opp","PrimaryPos","Salary","Minutes","DK_FP","Status","Notes"]], use_container_width=True)

# ==========================
# STEP B — APPLY OUT + DVP + SAVE FINAL
# ==========================
st.divider()
st.subheader("Step B — Apply OUT, then DvP, then Save FINAL")

if st.button("Run Projections (OUT + DvP)"):
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
    base["DvPNotes"] = ""
    base["DvPMult"] = 1.0

    # mark OUT
    base.loc[base["Name_clean"].isin(out_set), "Status"] = "OUT"

    # lock per-minute rates
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

    # recompute stats after OUT bump
    for idx in base.index[base["Status"] == "OK"]:
        m = base.loc[idx, "Minutes"]
        if pd.isna(m) or float(m) <= 0:
            continue
        for c in STAT_COLS:
            base.loc[idx, c] = round(float(base.loc[idx, f"PM_{c}"]) * float(m), 2)

    # apply DvP opponent-by-position if provided
    if dvp_pack is not None:
        dvp_df, league_avg = dvp_pack

        # quick dicts keyed by (TEAM, POS)
        dvp_key = {}
        for _, rr in dvp_df.iterrows():
            dvp_key[(rr["TEAM"], rr["POS"])] = rr

        avg_key = {}
        for _, rr in league_avg.iterrows():
            avg_key[rr["POS"]] = rr

        for idx, r in base[base["Status"] == "OK"].iterrows():
            opp = str(r.get("Opp", "")).upper().strip()
            pos = str(r.get("PrimaryPos", "")).upper().strip()
            if not opp or opp == "NAN" or not pos or pos == "NAN":
                base.loc[idx, "DvPNotes"] = "DVP:NA"
                continue

            if (opp, pos) not in dvp_key or pos not in avg_key:
                base.loc[idx, "DvPNotes"] = f"DVP:MISS({opp},{pos})"
                continue

            allowed = dvp_key[(opp, pos)]
            avg = avg_key[pos]

            # multipliers: allowed vs league avg allowed for that POS
            mults = {}
            for c in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
                av = float(avg[c])
                al = float(allowed[c])
                m = (al / av) if av > 0 else 1.0
                mults[c] = clamp(m, DVP_CAP_LOW, DVP_CAP_HIGH)

            # apply to stats (TOV also scaled but small cap keeps sane)
            base.loc[idx, "PTS"] = round(float(base.loc[idx, "PTS"]) * mults["PTS"], 2)
            base.loc[idx, "REB"] = round(float(base.loc[idx, "REB"]) * mults["REB"], 2)
            base.loc[idx, "AST"] = round(float(base.loc[idx, "AST"]) * mults["AST"], 2)
            base.loc[idx, "FG3M"] = round(float(base.loc[idx, "FG3M"]) * mults["FG3M"], 2)
            base.loc[idx, "STL"] = round(float(base.loc[idx, "STL"]) * mults["STL"], 2)
            base.loc[idx, "BLK"] = round(float(base.loc[idx, "BLK"]) * mults["BLK"], 2)
            base.loc[idx, "TOV"] = round(float(base.loc[idx, "TOV"]) * mults["TOV"], 2)

            base.loc[idx, "DvPMult"] = round(np.mean(list(mults.values())), 4)
            base.loc[idx, "DvPNotes"] = f"DVP {opp} {pos} PTS{mults['PTS']:.2f} REB{mults['REB']:.2f} AST{mults['AST']:.2f}"

    # DK FP final
    base.loc[base["Status"] == "OK", "DK_FP"] = base[base["Status"] == "OK"].apply(dk_fp, axis=1)

    final = base[(base["Status"] == "OK") & (~base["Name_clean"].isin(out_set))].copy()
    gist_write({GIST_FINAL: final.to_csv(index=False)})

    st.success("Saved FINAL")
    show_cols = ["Name","Team","Opp","PrimaryPos","Salary","Minutes","PTS","REB","AST","FG3M","STL","BLK","TOV","DK_FP","Notes","BumpNotes","DvPNotes"]
    st.dataframe(final[show_cols], use_container_width=True)

# ==========================
# OPTIMIZER
# ==========================
st.divider()
st.subheader("Optimizer")

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
