# ==========================
# DraftKings NBA Optimizer — Stable + Fast + DvP (Manual CSV) + Late Swap Locks
# + Player EXCLUDE (optimizer-only)
# + Team Started (late swap: exclude from NEW selections)
# + Team Exclude (optimizer-only; does NOT lock players)
# + Vegas (manual CSV: Team/Opponent/Spread/Total) applied in Step B
# + Props module (Top 15 projected by PTS/REB/AST/FG3M) with 70/80/90% bands + optional P(Over/Under)
# ==========================

import json
import difflib
import unicodedata
import time
import re
import hashlib
from io import StringIO
from math import erf, sqrt

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

LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

# Minutes caps
STARTER_MIN_CUTOFF = 28.0
STARTER_CAP = 36.0
BENCH_CAP = 32.0
MAX_MINUTES_ABS = 34.0  # HARD CAP YOU REQUESTED
BENCH_FLOOR = 6.0

# Recency blend weights
MIN_REC_W = 0.70
PM_REC_W = 0.30

# DvP caps
DVP_CAP_LOW = 0.95
DVP_CAP_HIGH = 1.05

# Injury "opportunity" bump caps (conservative)
BUMP_CAPS = {
    "PTS": 0.15,   # +15%
    "AST": 0.20,   # +20%
    "FG3M": 0.20,  # +20%
    "REB": 0.12,   # +12%
}

# Vegas adjustment (conservative)
VEGAS_TOTAL_BASELINE = 225.0
VEGAS_PACE_CAP = 0.05     # max ±5%
VEGAS_SPREAD_T1 = 8.0     # moderate blowout risk
VEGAS_SPREAD_T2 = 12.0    # high blowout risk

# --- Props config ---
PROPS_TOPK_PER_STAT = 15
VOL_LAST_N = 15
VOL_TIMEOUT = 12
VOL_RETRIES = 2

# Gist files
GIST_SLATE = "slate.csv"
GIST_DVP = "dvp.csv"
GIST_VEGAS = "vegas.csv"
GIST_OUT = "out.json"
GIST_LOCKS = "locks.json"
GIST_EXCLUDE = "exclude.json"
GIST_BASE = "base.csv"
GIST_FINAL = "final.csv"

# Team abbreviation normalization
TEAM_ALIASES = {
    "NY": "NYK",
    "SA": "SAS",
    "GS": "GSW",
    "NO": "NOP",
    "PHO": "PHX",
    "UTAH": "UTA",
    "WSH": "WAS",
}


# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DK NBA Optimizer — Stable + DvP (Manual CSV) + Late Swap Locks")


# ==========================
# HELPERS
# ==========================
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def md5_text(s: str | None) -> str:
    if not s:
        return "none"
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def norm_team(t: str) -> str:
    if t is None:
        return ""
    t = str(t).replace("\xa0", " ").strip().upper()
    if not t or t == "NAN":
        return ""
    t = t.split()[0]
    return TEAM_ALIASES.get(t, t)

def parse_minutes_min(x):
    s = str(x)
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan
    m, sec = s.split(":")
    try:
        return float(m) + float(sec) / 60
    except Exception:
        return np.nan

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

def parse_opponent_from_gameinfo(team_abbrev: str, game_info: str):
    if not isinstance(game_info, str):
        return None
    gi = game_info.strip().upper()
    if "@" not in gi:
        return None
    head = gi.split()[0]
    if "@" not in head:
        return None
    away, home = head.split("@", 1)
    team_abbrev = norm_team(team_abbrev)
    away = norm_team(away)
    home = norm_team(home)
    if team_abbrev == away:
        return home
    if team_abbrev == home:
        return away
    return None

def _to_float_first_token(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("\xa0", " ").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return np.nan
    return float(m.group(0))

def _team_first_token(val):
    if pd.isna(val):
        return ""
    s = str(val).replace("\xa0", " ").strip().upper()
    return s.split()[0] if s else ""

# --- Normal distribution helpers for props ---
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def z_for_two_sided(conf: float) -> float:
    if conf >= 0.90: return 1.645
    if conf >= 0.80: return 1.282
    return 1.036


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
    try:
        r = requests.patch(
            f"https://api.github.com/gists/{GIST_ID}",
            headers=gh(),
            json=payload,
            timeout=25
        )
        if r.status_code == 403:
            st.warning(
                "GitHub blocked a Gist write (403). Likely rate limit/abuse protection. "
                "Continuing without saving this change to the Gist."
            )
            return False
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        st.warning("Gist write failed. Continuing without saving.")
        return False


# ==========================
# NBA (FAST)
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
    df["NBA_First"] = df["NBA_Name_clean"].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else "")
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_player_to_nba(slate_name, dk_team, nba_df):
    cn = clean_name(slate_name)
    sn = strip_suffix(slate_name)
    dk_team = norm_team(dk_team)

    def team_ok(dfsub):
        if not dk_team:
            return dfsub
        return dfsub[dfsub["NBA_Team"] == dk_team]

    sub = nba_df[nba_df["NBA_Name_clean"] == cn].copy()
    sub = team_ok(sub)
    if len(sub) >= 1:
        return sub.iloc[0]

    sub = nba_df[nba_df["NBA_Name_stripped"] == sn].copy()
    sub = team_ok(sub)
    if len(sub) == 1:
        return sub.iloc[0]

    parts = sn.split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]

        cand = nba_df[nba_df["NBA_Last"] == last].copy()
        cand = team_ok(cand)
        if not cand.empty:
            cand["FIRST_SIM"] = cand["NBA_First"].apply(
                lambda x: difflib.SequenceMatcher(None, first, str(x)).ratio()
            )
            cand = cand[cand["FIRST_SIM"] >= 0.80].copy()
            if len(cand) == 1:
                return cand.iloc[0]

            if len(cand) > 1:
                cand["FULL_SIM"] = cand["NBA_Name_stripped"].apply(
                    lambda x: difflib.SequenceMatcher(None, sn, str(x)).ratio()
                )
                cand = cand.sort_values(["FULL_SIM", "FIRST_SIM"], ascending=False)
                if float(cand.iloc[0]["FULL_SIM"]) >= 0.92:
                    return cand.iloc[0]

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

@st.cache_data(ttl=900)
def gamelog_volatility(pid: int, last_n: int):
    last_err = None
    for attempt in range(1, VOL_RETRIES + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=int(pid),
                season=SEASON,
                timeout=VOL_TIMEOUT
            ).get_data_frames()[0]
            gl = gl.head(int(last_n)).copy()
            if gl.empty:
                raise RuntimeError("EMPTY_GAMELOG")

            gl["MIN_f"] = gl["MIN"].apply(parse_minutes_min)
            gl = gl[gl["MIN_f"].fillna(0) > 0].copy()
            if gl.empty:
                raise RuntimeError("NO_VALID_MINUTES")

            stds = {
                "PTS": float(gl["PTS"].std(ddof=1)) if len(gl) > 1 else 0.0,
                "REB": float(gl["REB"].std(ddof=1)) if len(gl) > 1 else 0.0,
                "AST": float(gl["AST"].std(ddof=1)) if len(gl) > 1 else 0.0,
                "FG3M": float(gl["FG3M"].std(ddof=1)) if len(gl) > 1 else 0.0,
            }
            mean_min = float(gl["MIN_f"].mean())
            return stds, mean_min
        except Exception as e:
            last_err = str(e)
            time.sleep(0.35 * attempt)

    raise RuntimeError(f"VOL_FAIL: {last_err}")


# ==========================
# SIDEBAR
# ==========================
st.sidebar.subheader("Reliability")
deterministic_mode = st.sidebar.checkbox("Deterministic mode (freeze BASE)", value=True)
st.sidebar.caption("When ON: Step B uses saved BASE only. Projections won't change unless you rebuild BASE or change OUT/locks/DvP/Vegas.")

st.sidebar.markdown("---")
st.sidebar.subheader("Recency Settings")
use_recency = st.sidebar.checkbox("Use recency blend (top salaries)", value=True)
top_n = st.sidebar.slider("Top N salaries to recency-blend", 0, 60, 25, 5)
last_n_games = st.sidebar.slider("Recent games (N)", 3, 15, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Uploads")

upload_slate = st.sidebar.file_uploader("Upload DK Slate CSV", type="csv")
if upload_slate:
    slate_text = upload_slate.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_SLATE: slate_text})
else:
    slate_text = gist_read(GIST_SLATE)
    if not slate_text:
        st.info("Upload DK Slate CSV to begin.")
        st.stop()

upload_dvp = st.sidebar.file_uploader("Upload Hashtag DvP CSV (Book1.csv format)", type="csv")
if upload_dvp:
    dvp_text = upload_dvp.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_DVP: dvp_text})
else:
    dvp_text = gist_read(GIST_DVP)
    if not dvp_text:
        st.warning("Upload Hashtag DvP CSV to apply opponent adjustments (app still works without it).")
        dvp_text = None

upload_vegas = st.sidebar.file_uploader("Upload Vegas CSV (Team,Opponent,Spread,Total) — optional", type="csv")
if upload_vegas:
    vegas_text = upload_vegas.getvalue().decode("utf-8", errors="ignore")
    gist_write({GIST_VEGAS: vegas_text})
else:
    vegas_text = gist_read(GIST_VEGAS)
    if not vegas_text:
        vegas_text = None


# ==========================
# LOAD SLATE
# ==========================
df = pd.read_csv(StringIO(slate_text))
required_cols = ["Name", "Salary", "TeamAbbrev", "Position"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"DK slate missing columns: {missing_cols}")
    st.stop()

game_info_col = None
for cand in ["Game Info", "GameInfo", "Game_Info", "Game"]:
    if cand in df.columns:
        game_info_col = cand
        break

slate = pd.DataFrame({
    "Name": df["Name"].astype(str),
    "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
    "Team": df["TeamAbbrev"].astype(str).apply(norm_team),
    "Positions": df["Position"].astype(str).apply(parse_positions),
})
slate["Name_clean"] = slate["Name"].apply(clean_name)
slate["PrimaryPos"] = slate["Positions"].apply(primary_pos)
slate["GameInfo"] = df[game_info_col].astype(str) if game_info_col else ""
slate["Opp"] = slate.apply(lambda r: parse_opponent_from_gameinfo(r["Team"], r["GameInfo"]), axis=1)
slate["Opp"] = slate["Opp"].apply(norm_team)

teams_on_slate = sorted([t for t in slate["Team"].dropna().unique().tolist() if t and t != "NAN"])


# ==========================
# LOAD SAVED OUT + LOCKS + EXCLUDES
# ==========================
try:
    saved_out = json.loads(gist_read(GIST_OUT) or "{}")
except Exception:
    saved_out = {}

try:
    saved_locks = json.loads(gist_read(GIST_LOCKS) or "{}")
except Exception:
    saved_locks = {}

try:
    saved_exclude = json.loads(gist_read(GIST_EXCLUDE) or "{}")
except Exception:
    saved_exclude = {}

saved_locked_teams = set(saved_locks.get("locked_teams", []))
saved_locked_players = set(saved_locks.get("locked_players", []))


# ==========================
# RUN SIGNATURE
# ==========================
base_text_for_sig = gist_read(GIST_BASE) or ""
sig = {
    "slate": md5_text(slate_text),
    "base": md5_text(base_text_for_sig) if base_text_for_sig else "none",
    "out": md5_text(json.dumps(saved_out, sort_keys=True)) if saved_out else "none",
    "exclude": md5_text(json.dumps(saved_exclude, sort_keys=True)) if saved_exclude else "none",
    "dvp": md5_text(dvp_text) if dvp_text else "none",
    "vegas": md5_text(vegas_text) if vegas_text else "none",
    "recency": f"use={use_recency},topN={top_n},lastN={last_n_games}",
    "caps": f"starter={STARTER_CAP},bench={BENCH_CAP},abs={MAX_MINUTES_ABS}",
}
with st.expander("Run Signature (helps explain why projections changed)", expanded=True):
    st.code(json.dumps(sig, indent=2))


# ==========================
# LATE SWAP CONTROLS + TOP TABLE
# ==========================
st.subheader("Late Swap Controls")

locked_teams = st.multiselect(
    "Teams started (exclude from NEW optimizer selections)",
    teams_on_slate,
    default=[t for t in teams_on_slate if t in saved_locked_teams]
)

exclude_teams = st.multiselect(
    "Exclude teams from optimizer pool (do NOT lock them)",
    teams_on_slate,
    default=[]
)

slate["OUT"] = slate["Name_clean"].map(lambda x: bool(saved_out.get(x, False)))
slate["LOCK"] = slate["Name_clean"].map(lambda x: bool(x in saved_locked_players))
slate["EXCLUDE"] = slate["Name_clean"].map(lambda x: bool(saved_exclude.get(x, False)))

edited = st.data_editor(
    slate[["OUT", "LOCK", "EXCLUDE", "Name", "Salary", "Team", "Opp", "PrimaryPos", "Positions"]],
    column_config={
        "OUT": st.column_config.CheckboxColumn("OUT"),
        "LOCK": st.column_config.CheckboxColumn("LOCK"),
        "EXCLUDE": st.column_config.CheckboxColumn("EXCLUDE"),
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
    },
    disabled=["Name", "Salary", "Team", "Opp", "PrimaryPos", "Positions"],
    use_container_width=True,
    hide_index=True,
)

out_flags = {clean_name(r["Name"]): bool(r["OUT"]) for _, r in edited.iterrows()}
out_set = {k for k, v in out_flags.items() if v}

lock_flags = {clean_name(r["Name"]): bool(r["LOCK"]) for _, r in edited.iterrows()}
locked_players_set = {k for k, v in lock_flags.items() if v}

exclude_flags = {clean_name(r["Name"]): bool(r["EXCLUDE"]) for _, r in edited.iterrows()}
excluded_players_set = {k for k, v in exclude_flags.items() if v}

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Save OUT + LOCKS + EXCLUDES"):
        gist_write({
            GIST_OUT: json.dumps(out_flags, indent=2),
            GIST_LOCKS: json.dumps({
                "locked_teams": sorted(list(set(locked_teams))),
                "locked_players": sorted(list(set(locked_players_set))),
            }, indent=2),
            GIST_EXCLUDE: json.dumps(exclude_flags, indent=2),
        })
        st.success("Saved OUT + LOCKS + EXCLUDES")
with c2:
    if st.button("Clear Locks"):
        gist_write({GIST_LOCKS: json.dumps({"locked_teams": [], "locked_players": []}, indent=2)})
        st.success("Cleared locks (refresh page)")
with c3:
    if st.button("Clear Excludes"):
        gist_write({GIST_EXCLUDE: json.dumps({}, indent=2)})
        st.success("Cleared excludes (refresh page)")


# ==========================
# LOAD DVP (Book1.csv FORMAT)
# ==========================
def load_dvp_book1(text: str):
    if not text:
        return None, None

    dvp = pd.read_csv(StringIO(text))
    required = ["Sort: Position", "Sort: Team", "Sort: PTS", "Sort: 3PM", "Sort: REB", "Sort: AST", "Sort: STL", "Sort: BLK", "Sort: TO"]
    missing = [c for c in required if c not in dvp.columns]
    if missing:
        return None, f"DvP CSV missing columns: {missing}"

    out = pd.DataFrame()
    out["POS"] = dvp["Sort: Position"].astype(str).str.upper().str.strip()
    out["TEAM"] = dvp["Sort: Team"].apply(lambda x: norm_team(_team_first_token(x)))

    out["PTS"] = dvp["Sort: PTS"].apply(_to_float_first_token)
    out["FG3M"] = dvp["Sort: 3PM"].apply(_to_float_first_token)
    out["REB"] = dvp["Sort: REB"].apply(_to_float_first_token)
    out["AST"] = dvp["Sort: AST"].apply(_to_float_first_token)
    out["STL"] = dvp["Sort: STL"].apply(_to_float_first_token)
    out["BLK"] = dvp["Sort: BLK"].apply(_to_float_first_token)
    out["TOV"] = dvp["Sort: TO"].apply(_to_float_first_token)

    out = out.dropna(subset=["TEAM", "POS", "PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]).copy()
    out = out[(out["TEAM"] != "") & (out["POS"] != "")].copy()

    league_avg = out.groupby("POS")[["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]].mean().reset_index()
    return (out, league_avg), None

dvp_pack = None
if dvp_text:
    dvp_pack, dvp_err = load_dvp_book1(dvp_text)
    if dvp_err:
        st.warning(dvp_err)
    else:
        st.sidebar.success("DvP loaded ✓")


# ==========================
# LOAD VEGAS (manual CSV)
# ==========================
def load_vegas(text: str):
    if not text:
        return None, None
    try:
        v = pd.read_csv(StringIO(text))
    except Exception:
        return None, "Vegas CSV could not be read."

    v.columns = [str(c).strip() for c in v.columns]
    req = ["Team", "Opponent", "Spread", "Total"]
    missing = [c for c in req if c not in v.columns]
    if missing:
        return None, f"Vegas CSV missing columns: {missing}"

    out = pd.DataFrame()
    out["TEAM"] = v["Team"].astype(str).apply(norm_team)
    out["OPP"] = v["Opponent"].astype(str).apply(norm_team)
    out["SPREAD"] = pd.to_numeric(v["Spread"], errors="coerce")
    out["TOTAL"] = pd.to_numeric(v["Total"], errors="coerce")

    out = out.dropna(subset=["TEAM", "OPP", "SPREAD", "TOTAL"]).copy()
    out = out[(out["TEAM"] != "") & (out["OPP"] != "")].copy()

    # map by TEAM
    key = {rr["TEAM"]: {"OPP": rr["OPP"], "SPREAD": float(rr["SPREAD"]), "TOTAL": float(rr["TOTAL"])} for _, rr in out.iterrows()}
    return key, None

vegas_map = None
if vegas_text:
    vegas_map, vegas_err = load_vegas(vegas_text)
    if vegas_err:
        st.warning(vegas_err)
    else:
        st.sidebar.success("Vegas loaded ✓")


# ==========================
# STEP A — BUILD BASE
# ==========================
st.divider()
st.subheader("Step A — Build BASE")

if deterministic_mode:
    confirm_rebuild = st.checkbox("I want to rebuild BASE (this can change projections).", value=False)
else:
    confirm_rebuild = True

if st.button("Build BASE", disabled=(deterministic_mode and not confirm_rebuild)):
    nba_df = league_player_df()
    rows = []

    top_salary_names = set()
    if use_recency and top_n > 0:
        tmp = slate.dropna(subset=["Salary"]).sort_values("Salary", ascending=False).head(int(top_n))
        top_salary_names = set(tmp["Name_clean"].tolist())

    prog = st.progress(0, text="Mapping DK slate to league stats...")
    for i, r in slate.iterrows():
        prog.progress((i + 1) / len(slate), text=f"Mapping {r['Name']} ({i+1}/{len(slate)})")

        hit = match_player_to_nba(r["Name"], r["Team"], nba_df)
        if hit is None:
            rows.append({
                **r.to_dict(),
                "Minutes": np.nan,
                **{c: np.nan for c in STAT_COLS},
                "Status": "ERR",
                "Notes": "No safe match in league stats",
                "Matched_NBA_Name": "",
                "Matched_NBA_Team": "",
            })
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

        row = {
            **r.to_dict(),
            "Minutes": round(float(mins), 2),
            **stats,
            "Status": "OK",
            "Notes": notes,
            "Matched_NBA_Name": str(hit["NBA_Name"]),
            "Matched_NBA_Team": str(hit["NBA_Team"]),
        }

        if r["Team"] and str(hit["NBA_Team"]) != str(r["Team"]):
            row["Status"] = "ERR"
            row["Notes"] = f"TEAM_MISMATCH (DK {r['Team']} vs NBA {hit['NBA_Team']})"

        rows.append(row)

    base = pd.DataFrame(rows)
    base["DK_FP"] = base.apply(lambda rr: dk_fp(rr) if rr["Status"] == "OK" else np.nan, axis=1)
    base["BASE_Minutes"] = base["Minutes"]
    gist_write({GIST_BASE: base.to_csv(index=False)})

    st.success("Saved BASE")
    st.dataframe(
        base[["Name", "Salary", "Team", "Opp", "PrimaryPos", "Minutes", "DK_FP", "Status", "Notes", "Matched_NBA_Name", "Matched_NBA_Team"]],
        use_container_width=True
    )


# ==========================
# STEP B — RUN PROJECTIONS (Injury minutes + opportunity + Vegas + DvP)
# ==========================
st.divider()
st.subheader("Step B — Run Projections (Injury Bumps + Vegas + DvP)")

if st.button("Run Projections"):
    base_text = gist_read(GIST_BASE)
    if not base_text:
        st.error("No BASE found. Run Step A first.")
        st.stop()

    base = pd.read_csv(StringIO(base_text))
    base["Positions"] = base["Positions"].apply(eval)
    base["Minutes"] = pd.to_numeric(base["Minutes"], errors="coerce")
    base["BASE_Minutes"] = pd.to_numeric(base.get("BASE_Minutes", base["Minutes"]), errors="coerce")
    base["Salary"] = pd.to_numeric(base["Salary"], errors="coerce")
    base["Status"] = base["Status"].astype(str)
    base["Notes"] = base.get("Notes", "").fillna("").astype(str)

    base["BumpNotes"] = ""
    base["UsageNotes"] = ""
    base["VegasNotes"] = ""
    base["DvPNotes"] = ""
    base["DvPMult"] = 1.0

    # Apply OUT flags
    base.loc[base["Name_clean"].isin(out_set), "Status"] = "OUT"

    # Numeric
    for c in STAT_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # Per-minute rates from BASE
    for c in STAT_COLS:
        base[f"PM_{c}"] = np.where(
            (base["Status"] == "OK") & (base["Minutes"].fillna(0) > 0),
            base[c].fillna(0) / base["Minutes"].replace(0, np.nan),
            0.0
        )
        base[f"PM_{c}"] = base[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Role-based cap + absolute cap
    def minute_cap(row):
        bm = float(row["BASE_Minutes"]) if pd.notna(row["BASE_Minutes"]) else 0.0
        cap_role = STARTER_CAP if bm >= STARTER_MIN_CUTOFF else BENCH_CAP
        return min(cap_role, MAX_MINUTES_ABS)

    base["MIN_CAP"] = base.apply(minute_cap, axis=1)

    # 1) Minutes redistribution from OUT
    for team in base["Team"].dropna().unique():
        out_t = base[(base["Team"] == team) & (base["Status"] == "OUT")]
        ok_t = base[(base["Team"] == team) & (base["Status"] == "OK")]
        if out_t.empty or ok_t.empty:
            continue

        missing = float(out_t["Minutes"].fillna(0).sum())
        if missing <= 0:
            continue

        weights = ok_t["BASE_Minutes"].fillna(0).clip(lower=BENCH_FLOOR)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue

        for idx in ok_t.index:
            inc = missing * float(weights.loc[idx]) / wsum
            new_m = float(base.loc[idx, "Minutes"]) + inc
            cap = float(base.loc[idx, "MIN_CAP"])

            if new_m > cap:
                inc = max(0.0, cap - float(base.loc[idx, "Minutes"]))
                new_m = cap

            base.loc[idx, "Minutes"] = round(new_m, 2)
            if inc > 0:
                base.loc[idx, "BumpNotes"] = (base.loc[idx, "BumpNotes"] + f" MIN+{inc:.1f}").strip()

    # 1b) Vegas spread -> small minutes risk adjustment (AFTER injuries, BEFORE stat recompute)
    if vegas_map is not None:
        for idx, r in base[base["Status"] == "OK"].iterrows():
            team = norm_team(r.get("Team", ""))
            if not team or team not in vegas_map:
                continue
            spread = float(vegas_map[team]["SPREAD"])
            abs_sp = abs(spread)

            bm = float(r.get("BASE_Minutes", 0.0)) if pd.notna(r.get("BASE_Minutes", np.nan)) else 0.0
            is_starter = bm >= STARTER_MIN_CUTOFF

            if abs_sp >= VEGAS_SPREAD_T2:
                mult = 0.95 if is_starter else 1.02
                tag = "SPREAD12"
            elif abs_sp >= VEGAS_SPREAD_T1:
                mult = 0.97 if is_starter else 1.01
                tag = "SPREAD8"
            else:
                continue

            new_m = float(r["Minutes"]) * mult
            # enforce cap
            new_m = min(new_m, float(r["MIN_CAP"]))
            base.loc[idx, "Minutes"] = round(new_m, 2)
            base.loc[idx, "VegasNotes"] = (base.loc[idx, "VegasNotes"] + f" {tag}x{mult:.2f}").strip()

    # Recompute stats from per-minute using updated minutes
    for idx in base.index[base["Status"] == "OK"]:
        m = base.loc[idx, "Minutes"]
        if pd.isna(m) or float(m) <= 0:
            continue
        for c in STAT_COLS:
            base.loc[idx, c] = round(float(base.loc[idx, f"PM_{c}"]) * float(m), 2)

    # 2) Opportunity redistribution (PTS/AST/FG3M/REB)
    for team in base["Team"].dropna().unique():
        out_t = base[(base["Team"] == team) & (base["Status"] == "OUT")]
        ok_t = base[(base["Team"] == team) & (base["Status"] == "OK")]
        if out_t.empty or ok_t.empty:
            continue

        removed = {
            "PTS": float(out_t["PTS"].fillna(0).sum()),
            "AST": float(out_t["AST"].fillna(0).sum()),
            "FG3M": float(out_t["FG3M"].fillna(0).sum()),
            "REB": float(out_t["REB"].fillna(0).sum()),
        }
        if sum(removed.values()) <= 0:
            continue

        w_pts = ok_t["PTS"].fillna(0).clip(lower=0.01)
        w_ast = ok_t["AST"].fillna(0).clip(lower=0.01)
        w_3 = ok_t["FG3M"].fillna(0).clip(lower=0.01)
        w_reb = ok_t["REB"].fillna(0).clip(lower=0.01)

        sums = {
            "PTS": float(w_pts.sum()),
            "AST": float(w_ast.sum()),
            "FG3M": float(w_3.sum()),
            "REB": float(w_reb.sum()),
        }

        for idx in ok_t.index:
            if sums["PTS"] > 0 and removed["PTS"] > 0:
                add = removed["PTS"] * (float(w_pts.loc[idx]) / sums["PTS"])
                cap = float(base.loc[idx, "PTS"]) * BUMP_CAPS["PTS"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "PTS"] = round(float(base.loc[idx, "PTS"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" PTS+{add:.1f}"

            if sums["AST"] > 0 and removed["AST"] > 0:
                add = removed["AST"] * (float(w_ast.loc[idx]) / sums["AST"])
                cap = float(base.loc[idx, "AST"]) * BUMP_CAPS["AST"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "AST"] = round(float(base.loc[idx, "AST"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" AST+{add:.1f}"

            if sums["FG3M"] > 0 and removed["FG3M"] > 0:
                add = removed["FG3M"] * (float(w_3.loc[idx]) / sums["FG3M"])
                cap = float(base.loc[idx, "FG3M"]) * BUMP_CAPS["FG3M"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "FG3M"] = round(float(base.loc[idx, "FG3M"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" 3PM+{add:.1f}"

            if sums["REB"] > 0 and removed["REB"] > 0:
                add = removed["REB"] * (float(w_reb.loc[idx]) / sums["REB"])
                cap = float(base.loc[idx, "REB"]) * BUMP_CAPS["REB"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "REB"] = round(float(base.loc[idx, "REB"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" REB+{add:.1f}"

    base["UsageNotes"] = base["UsageNotes"].fillna("").astype(str).str.strip()

    # 2b) Vegas TOTAL -> pace multiplier (after injury + opportunity, before DvP)
    if vegas_map is not None:
        for idx, r in base[base["Status"] == "OK"].iterrows():
            team = norm_team(r.get("Team", ""))
            if not team or team not in vegas_map:
                continue
            total = float(vegas_map[team]["TOTAL"])
            raw = total / VEGAS_TOTAL_BASELINE
            pace_mult = clamp(raw, 1.0 - VEGAS_PACE_CAP, 1.0 + VEGAS_PACE_CAP)

            # apply only to the "pace" stats
            for c in ["PTS", "AST", "FG3M"]:
                base.loc[idx, c] = round(float(base.loc[idx, c]) * pace_mult, 2)

            base.loc[idx, "VegasNotes"] = (base.loc[idx, "VegasNotes"] + f" TOTAL{total:.1f}x{pace_mult:.3f}").strip()

    # 3) Apply DvP vs opponent by position
    if dvp_pack is not None:
        dvp_df, league_avg = dvp_pack
        dvp_key = {(rr["TEAM"], rr["POS"]): rr for _, rr in dvp_df.iterrows()}
        avg_key = {rr["POS"]: rr for _, rr in league_avg.iterrows()}

        for idx, r in base[base["Status"] == "OK"].iterrows():
            opp = norm_team(r.get("Opp", ""))
            pos = str(r.get("PrimaryPos", "")).upper().strip()
            if not opp or not pos or opp == "NAN" or pos == "NAN":
                base.loc[idx, "DvPNotes"] = "DVP:NA"
                continue

            if (opp, pos) not in dvp_key or pos not in avg_key:
                base.loc[idx, "DvPNotes"] = f"DVP:MISS({opp},{pos})"
                continue

            allowed = dvp_key[(opp, pos)]
            avg = avg_key[pos]

            mults = {}
            for c in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
                av = float(avg[c])
                al = float(allowed[c])
                mlt = (al / av) if av > 0 else 1.0
                mults[c] = clamp(mlt, DVP_CAP_LOW, DVP_CAP_HIGH)

            for c in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
                base.loc[idx, c] = round(float(base.loc[idx, c]) * mults[c], 2)

            base.loc[idx, "DvPMult"] = round(float(np.mean(list(mults.values()))), 4)
            base.loc[idx, "DvPNotes"] = (
                f"DVP {opp} {pos} "
                f"PTS{mults['PTS']:.2f} REB{mults['REB']:.2f} AST{mults['AST']:.2f}"
            )

    # DK FP
    base.loc[base["Status"] == "OK", "DK_FP"] = base[base["Status"] == "OK"].apply(dk_fp, axis=1)

    # FINAL = OK and not OUT
    final = base[(base["Status"] == "OK") & (~base["Name_clean"].isin(out_set))].copy()

    gist_write({GIST_FINAL: final.to_csv(index=False)})
    st.success("Saved FINAL")

    show_cols = [
        "Name", "Salary", "Team", "Opp", "PrimaryPos",
        "Minutes", "MIN_CAP",
        "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
        "DK_FP",
        "Notes", "BumpNotes", "UsageNotes", "VegasNotes", "DvPNotes"
    ]
    st.dataframe(final[show_cols], use_container_width=True)


# ==========================
# OPTIMIZER (LATE SWAP)
# ==========================
st.divider()
st.subheader("Optimizer (Late Swap — respects Team Started + Player LOCK)")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.info("No FINAL saved yet. Run Step A then Step B.")
    st.stop()

pool = pd.read_csv(StringIO(final_text))
pool["Positions"] = pool["Positions"].apply(eval)
pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
pool["DK_FP"] = pd.to_numeric(pool["DK_FP"], errors="coerce")
pool = pool.dropna(subset=["Salary", "DK_FP"]).copy()
pool = pool[pool["Salary"] > 0].copy()

if "Name_clean" not in pool.columns:
    pool["Name_clean"] = pool["Name"].apply(clean_name)

started_teams = set(locked_teams)
excluded_teams_set = set(exclude_teams)

def assign_locked_to_slots(locked_df):
    players = list(locked_df.index)
    cand = {i: [s for s in DK_SLOTS if eligible_for_slot(locked_df.loc[i, "Positions"], s)] for i in players}
    players_sorted = sorted(players, key=lambda i: len(cand[i]))

    used_slots = set()
    assignment = {}

    def backtrack(k):
        if k == len(players_sorted):
            return True
        i = players_sorted[k]
        for s in cand[i]:
            if s in used_slots:
                continue
            used_slots.add(s)
            assignment[s] = i
            if backtrack(k + 1):
                return True
            used_slots.remove(s)
            assignment.pop(s, None)
        return False

    ok = backtrack(0)
    return assignment if ok else None

if st.button("Optimize (respect locks)"):
    locked_df = pool[pool["Name_clean"].isin(locked_players_set)].copy()

    excluded_effective = set(excluded_players_set) - set(locked_players_set)

    candidate_df = pool.copy()

    # Exclude teams from pool (does NOT lock)
    if excluded_teams_set:
        candidate_df = candidate_df[~candidate_df["Team"].isin(excluded_teams_set)].copy()

    # Exclude players from pool
    if excluded_effective:
        candidate_df = candidate_df[~candidate_df["Name_clean"].isin(excluded_effective)].copy()

    # Late swap: exclude started teams from NEW selections
    if started_teams:
        candidate_df = candidate_df[~candidate_df["Team"].isin(started_teams)].copy()

    # Always re-add locked players (even if team excluded/started)
    if not locked_df.empty:
        candidate_df = pd.concat([candidate_df, locked_df], axis=0).drop_duplicates(subset=["Name_clean"])

    locked_assignment = {}
    remaining_slots = DK_SLOTS.copy()
    salary_locked = 0.0

    if not locked_df.empty:
        locked_assignment = assign_locked_to_slots(locked_df)
        if locked_assignment is None:
            st.error("Locked players cannot fit into DK slots. Un-lock one player and try again.")
            st.stop()
        used_slots = set(locked_assignment.keys())
        remaining_slots = [s for s in DK_SLOTS if s not in used_slots]
        salary_locked = float(locked_df.loc[list(locked_assignment.values()), "Salary"].sum())

    remaining_cap = DK_SALARY_CAP - salary_locked
    if remaining_cap < 0:
        st.error(f"Locked salary exceeds cap. Locked salary = {int(salary_locked)}")
        st.stop()

    if len(remaining_slots) == 0:
        lineup = []
        for slot, idx in locked_assignment.items():
            row = locked_df.loc[idx].to_dict()
            row["Slot"] = slot
            row["Locked"] = True
            lineup.append(row)
        lineup_df = pd.DataFrame(lineup).sort_values("Slot")
        st.dataframe(lineup_df[["Slot", "Locked", "Name", "Team", "Salary", "DK_FP", "Minutes"]], use_container_width=True)
        st.metric("Total Salary", int(salary_locked))
        st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))
        st.stop()

    opt_pool = candidate_df[~candidate_df["Name_clean"].isin(locked_players_set)].copy()

    prob = LpProblem("DK_LATE_SWAP", LpMaximize)

    x = {}
    for i, r in opt_pool.iterrows():
        for slot in remaining_slots:
            if eligible_for_slot(r["Positions"], slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    if not x:
        st.error("No feasible candidates (too many teams started/excluded/players excluded).")
        st.stop()

    prob += lpSum(opt_pool.loc[i, "DK_FP"] * x[(i, slot)] for (i, slot) in x)

    for slot in remaining_slots:
        prob += lpSum(x[(i, slot)] for i in opt_pool.index if (i, slot) in x) == 1

    for i in opt_pool.index:
        prob += lpSum(x[(i, slot)] for slot in remaining_slots if (i, slot) in x) <= 1

    prob += lpSum(opt_pool.loc[i, "Salary"] * x[(i, slot)] for (i, slot) in x) <= remaining_cap

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = []

    for slot, idx in (locked_assignment or {}).items():
        row = locked_df.loc[idx].to_dict()
        row["Slot"] = slot
        row["Locked"] = True
        lineup.append(row)

    for slot in remaining_slots:
        chosen = None
        for i in opt_pool.index:
            if (i, slot) in x and x[(i, slot)].value() == 1:
                chosen = i
                break
        if chosen is None:
            st.error("No feasible lineup found.")
            st.stop()
        row = opt_pool.loc[chosen].to_dict()
        row["Slot"] = slot
        row["Locked"] = False
        lineup.append(row)

    lineup_df = pd.DataFrame(lineup).sort_values("Slot")
    st.dataframe(lineup_df[["Slot", "Locked", "Name", "Team", "Salary", "Minutes", "DK_FP"]], use_container_width=True)
    st.metric("Total Salary", int(lineup_df["Salary"].sum()))
    st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))

    if started_teams:
        st.caption(f"Started teams excluded from NEW selections: {', '.join(sorted(list(started_teams)))}")
    if excluded_teams_set:
        st.caption(f"Teams excluded from optimizer pool: {', '.join(sorted(list(excluded_teams_set)))}")
    if excluded_effective:
        st.caption(f"Excluded players from optimizer pool: {len(excluded_effective)} player(s)")


# ==========================
# PROPS MODULE (Top 15 projected by PTS/REB/AST/FG3M)
# ==========================
st.divider()
st.subheader("Props (Top 15 by Stat) — 70/80/90% confidence + optional P(Over/Under)")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.info("Run Step B first to create FINAL projections.")
    st.stop()

final_df = pd.read_csv(StringIO(final_text))
final_df["Name_clean"] = final_df["Name"].astype(str).apply(clean_name)

for c in ["Minutes", "PTS", "REB", "AST", "FG3M", "Salary"]:
    if c in final_df.columns:
        final_df[c] = pd.to_numeric(final_df[c], errors="coerce")

stat_list = ["PTS", "REB", "AST", "FG3M"]
top_sets = []
for stat in stat_list:
    topk = final_df.dropna(subset=[stat]).sort_values(stat, ascending=False).head(PROPS_TOPK_PER_STAT)
    top_sets.append(topk[["Name", "Name_clean", "Team", "Minutes", stat]].copy())

props_pool = pd.concat(top_sets, axis=0).drop_duplicates(subset=["Name_clean"]).copy()
props_pool = props_pool.dropna(subset=["Minutes"]).copy()

st.caption(
    f"Volatility computed for union of top {PROPS_TOPK_PER_STAT} projected players in each stat "
    f"(PTS/REB/AST/3PM). Total players: {len(props_pool)}"
)

props_file = st.file_uploader(
    "Upload Props CSV (optional) with columns: Name, Market (PTS/REB/AST/3PM), Line",
    type="csv",
    key="props_upload"
)

props_lines = None
if props_file:
    props_lines = pd.read_csv(props_file)
    props_lines.columns = [str(c).strip() for c in props_lines.columns]
    needed = {"Name", "Market", "Line"}
    if not needed.issubset(set(props_lines.columns)):
        st.error(f"Props CSV missing columns. Need: {sorted(list(needed))}")
        props_lines = None
    else:
        props_lines["Name_clean"] = props_lines["Name"].astype(str).apply(clean_name)
        props_lines["Market"] = props_lines["Market"].astype(str).str.upper().str.strip()
        props_lines["Line"] = pd.to_numeric(props_lines["Line"], errors="coerce")

conf_levels = st.multiselect("Confidence levels", [0.70, 0.80, 0.90], default=[0.70, 0.80, 0.90])
use_lines = st.checkbox("If props CSV uploaded, compute P(Over/Under) vs the line", value=True)

if st.button("Build Prop Table"):
    nba_df = league_player_df()
    prog = st.progress(0, text="Computing volatility (recent game logs)...")

    out_rows = []
    props_pool_reset = props_pool.reset_index(drop=True)

    for i, r in props_pool_reset.iterrows():
        prog.progress((i + 1) / len(props_pool_reset), text=f"Volatility: {r['Name']} ({i+1}/{len(props_pool_reset)})")

        hit = match_player_to_nba(r["Name"], r["Team"], nba_df)
        if hit is None:
            continue

        pid = int(hit["PLAYER_ID"])
        try:
            stds_raw, mean_min_hist = gamelog_volatility(pid, VOL_LAST_N)
        except Exception:
            continue

        proj_min = float(r["Minutes"]) if pd.notna(r["Minutes"]) else 0.0
        if proj_min <= 0:
            continue

        scale = sqrt(max(0.25, proj_min / max(1.0, mean_min_hist)))

        row_base = final_df[final_df["Name_clean"] == r["Name_clean"]].head(1)
        if row_base.empty:
            continue
        row_base = row_base.iloc[0]

        for stat in ["PTS", "REB", "AST", "FG3M"]:
            mu = float(row_base.get(stat, np.nan))
            if not np.isfinite(mu):
                continue

            sigma = float(stds_raw.get(stat, 0.0)) * float(scale)
            sigma = max(0.5, sigma)

            row = {
                "Name": row_base["Name"],
                "Team": row_base.get("Team", ""),
                "Minutes": round(float(row_base.get("Minutes", proj_min)), 2),
                "Stat": stat if stat != "FG3M" else "3PM",
                "Mean": round(mu, 2),
                "Sigma": round(sigma, 2),
            }

            for conf in conf_levels:
                z = z_for_two_sided(float(conf))
                lo = mu - z * sigma
                hi = mu + z * sigma
                row[f"{int(conf*100)}%_Low"] = round(lo, 2)
                row[f"{int(conf*100)}%_High"] = round(hi, 2)

            out_rows.append(row)

    props_out = pd.DataFrame(out_rows)
    if props_out.empty:
        st.warning("No prop rows generated (name matching/volatility may have failed).")
        st.stop()

    if (props_lines is not None) and use_lines:
        m = props_lines.dropna(subset=["Line"]).copy()
        market_map = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "3PM", "FG3M": "3PM"}
        m["Stat"] = m["Market"].map(market_map)
        m = m.dropna(subset=["Stat"]).copy()

        tmp = props_out.copy()
        tmp["Name_clean"] = tmp["Name"].astype(str).apply(clean_name)

        merged = tmp.merge(
            m[["Name_clean", "Stat", "Line"]],
            on=["Name_clean", "Stat"],
            how="left"
        )

        def prob_over(mu, sigma, line):
            if not np.isfinite(line):
                return np.nan
            z = (float(mu) - float(line)) / max(1e-9, float(sigma))
            return float(norm_cdf(z))

        merged["Line"] = pd.to_numeric(merged["Line"], errors="coerce")
        merged["P(Over)"] = merged.apply(lambda rr: prob_over(rr["Mean"], rr["Sigma"], rr["Line"]), axis=1)
        merged["P(Under)"] = 1.0 - merged["P(Over)"]
        merged["P(Over)"] = merged["P(Over)"].round(3)
        merged["P(Under)"] = merged["P(Under)"].round(3)

        props_out = merged.drop(columns=["Name_clean"], errors="ignore")

    sort_stat = {"PTS": 0, "3PM": 1, "REB": 2, "AST": 3}
    props_out["_stat_rank"] = props_out["Stat"].map(sort_stat).fillna(99)
    props_out = props_out.sort_values(["_stat_rank", "Mean"], ascending=[True, False]).drop(columns=["_stat_rank"])

    st.success("Props table ready.")
    st.dataframe(props_out, use_container_width=True)