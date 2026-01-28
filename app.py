# ==========================
# DraftKings NBA Optimizer (Stable + Fast + More Accurate Injury Bumps)
# - Fast base from NBA API season stats + optional recency blend for top salaries
# - Deterministic mode (freeze BASE so projections don't "mysteriously" change)
# - Injury handling upgraded: Minutes + Opportunity (PTS/AST/3PM/REB) redistribution (no on/off splits)
# - Realistic minute caps: starters vs bench + absolute cap (MAX_MINUTES_ABS)
# - DvP (manual upload) with tighter caps
# - Late swap locks (team locks exclude from NEW picks + player locks)
# - Gist write 403-safe; avoids writing OUT during Step B
# ==========================

import json
import difflib
import unicodedata
import time
import re
import hashlib
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

LEAGUE_TIMEOUT = 20
GAMELOG_TIMEOUT = 12
GAMELOG_RETRIES = 2

# --- Minute caps (realistic)
STARTER_MIN_CUTOFF = 28.0
STARTER_CAP = 36.0
BENCH_CAP = 32.0
MAX_MINUTES_ABS = 34.0  # <--- your hard rule

BENCH_FLOOR = 6.0

# Recency blend weights
MIN_REC_W = 0.70
PM_REC_W = 0.30

# DvP caps (tighter = more stable)
DVP_CAP_LOW = 0.95
DVP_CAP_HIGH = 1.05

# Injury "opportunity" bump caps (conservative)
BUMP_CAPS = {
    "PTS": 0.15,   # +15%
    "AST": 0.20,   # +20%
    "FG3M": 0.20,  # +20%
    "REB": 0.12,   # +12%
}

# Gist files
GIST_SLATE = "slate.csv"
GIST_DVP = "dvp.csv"
GIST_OUT = "out.json"
GIST_LOCKS = "locks.json"
GIST_BASE = "base.csv"
GIST_FINAL = "final.csv"


# Team abbreviation normalization (Hashtag vs DK vs NBA)
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
st.title("DK NBA Optimizer — Stable + Fast + DvP + Late Swap Locks")


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

def first_token(name: str) -> str:
    parts = clean_name(name).split()
    return parts[0] if parts else ""

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

# DK "Game Info": "LAL@BOS 07:30PM ET"
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
    """
    DvP cells look like: "21.0   21" or "3.5  10"
    Return first float found in the string.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).replace("\xa0", " ").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return np.nan
    return float(m.group(0))

def _team_first_token(val):
    """
    Team cells look like: 'OKC   1' (team + rank).
    """
    if pd.isna(val):
        return ""
    s = str(val).replace("\xa0", " ").strip().upper()
    return s.split()[0] if s else ""


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

# SAFE gist write (403 won't crash app)
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
    except requests.exceptions.RequestException as e:
        st.warning(f"Gist write failed ({type(e).__name__}). Continuing without saving.")
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
    """
    Safer matching:
    - Try exact matches first
    - If last-name collision, require strong first-name agreement
    - Prefer team matches when possible
    """
    cn = clean_name(slate_name)
    sn = strip_suffix(slate_name)
    dk_team = norm_team(dk_team)

    # 1) exact clean
    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        if dk_team and (exact.iloc[0]["NBA_Team"] == dk_team):
            return exact.iloc[0]
        return exact.iloc[0]

    # 2) exact stripped
    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        if dk_team and (exact2.iloc[0]["NBA_Team"] == dk_team):
            return exact2.iloc[0]
        return exact2.iloc[0]

    # 3) last name candidates
    parts = sn.split()
    if parts:
        first = parts[0]
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last].copy()
        if not cand.empty:
            # Prefer team match if available
            if dk_team:
                cand_team = cand[cand["NBA_Team"] == dk_team]
                if not cand_team.empty:
                    cand = cand_team

            # If multiple candidates share last name, require first-name similarity
            best_row, best_score = None, 0.0
            for _, row in cand.iterrows():
                full_score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                first_score = difflib.SequenceMatcher(None, first, row["NBA_First"]).ratio() if row["NBA_First"] else 0.0

                # Guardrail: if first name doesn't match well, demand very high full score
                if first_score < 0.70 and full_score < 0.93:
                    continue

                score = 0.65 * full_score + 0.35 * first_score
                if score > best_score:
                    best_row, best_score = row, score

            if best_row is not None and best_score >= 0.78:
                return best_row

    # 4) close match on full clean list, with team preference
    candidates = nba_df["NBA_Name_clean"].tolist()
    hit = difflib.get_close_matches(cn, candidates, n=5, cutoff=0.90)
    if hit:
        sub = nba_df[nba_df["NBA_Name_clean"].isin(hit)].copy()
        if dk_team and not sub[sub["NBA_Team"] == dk_team].empty:
            sub = sub[sub["NBA_Team"] == dk_team]
        # choose best similarity among remaining
        sub["SIM"] = sub["NBA_Name_clean"].apply(lambda x: difflib.SequenceMatcher(None, cn, x).ratio())
        sub = sub.sort_values("SIM", ascending=False)
        return sub.iloc[0]

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
# SIDEBAR: RELIABILITY MODE
# ==========================
st.sidebar.subheader("Reliability")
deterministic_mode = st.sidebar.checkbox("Deterministic mode (freeze BASE)", value=True)
st.sidebar.caption(
    "When ON: Step B uses saved BASE only. Projections won't change unless you rebuild BASE or change OUT/locks/DvP."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Recency Settings")
use_recency = st.sidebar.checkbox("Use recency blend (top salaries)", value=True)
top_n = st.sidebar.slider("Top N salaries to recency-blend", 0, 60, 25, 5)
last_n_games = st.sidebar.slider("Recent games (N)", 3, 15, 10, 1)


# ==========================
# UPLOADS
# ==========================
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
# LOAD SAVED OUT + LOCKS
# ==========================
try:
    saved_out = json.loads(gist_read(GIST_OUT) or "{}")
except Exception:
    saved_out = {}

try:
    saved_locks = json.loads(gist_read(GIST_LOCKS) or "{}")
except Exception:
    saved_locks = {}

saved_locked_teams = set(saved_locks.get("locked_teams", []))
saved_locked_players = set(saved_locks.get("locked_players", []))


# ==========================
# TOP: RUN SIGNATURE (so you can trust stability)
# ==========================
base_text_for_sig = gist_read(GIST_BASE) or ""
sig = {
    "slate": md5_text(slate_text),
    "base": md5_text(base_text_for_sig) if base_text_for_sig else "none",
    "out": md5_text(json.dumps(saved_out, sort_keys=True)) if saved_out else "none",
    "dvp": md5_text(dvp_text) if dvp_text else "none",
    "recency": f"use={use_recency},topN={top_n},lastN={last_n_games}",
    "caps": f"starter={STARTER_CAP},bench={BENCH_CAP},abs={MAX_MINUTES_ABS}",
}
with st.expander("Run Signature (helps explain why projections changed)", expanded=True):
    st.code(json.dumps(sig, indent=2))


# ==========================
# TEAM LOCK UI
# ==========================
st.subheader("Late Swap Controls")

locked_teams = st.multiselect(
    "Teams started / lock all players",
    teams_on_slate,
    default=[t for t in teams_on_slate if t in saved_locked_teams]
)

slate["LOCK"] = slate["Team"].isin(set(locked_teams))
slate["LOCK"] = slate.apply(lambda r: True if r["Name_clean"] in saved_locked_players else bool(r["LOCK"]), axis=1)
slate["OUT"] = slate["Name_clean"].map(lambda x: bool(saved_out.get(x, False)))

edited = st.data_editor(
    slate[["OUT","LOCK","Name","Team","Opp","PrimaryPos","Salary","Positions"]],
    column_config={
        "OUT": st.column_config.CheckboxColumn("OUT"),
        "LOCK": st.column_config.CheckboxColumn("LOCK"),
    },
    disabled=["Name","Team","Opp","PrimaryPos","Salary","Positions"],
    use_container_width=True,
    hide_index=True,
)

out_flags = {clean_name(r["Name"]): bool(r["OUT"]) for _, r in edited.iterrows()}
out_set = {k for k, v in out_flags.items() if v}
lock_flags = {clean_name(r["Name"]): bool(r["LOCK"]) for _, r in edited.iterrows()}
locked_players_set = {k for k, v in lock_flags.items() if v}

c1, c2 = st.columns(2)
with c1:
    if st.button("Save OUT + LOCKS"):
        gist_write({
            GIST_OUT: json.dumps(out_flags, indent=2),
            GIST_LOCKS: json.dumps({
                "locked_teams": sorted(list(set(locked_teams))),
                "locked_players": sorted(list(locked_players_set)),
            }, indent=2),
        })
        st.success("Saved OUT + LOCKS")
with c2:
    if st.button("Clear Locks"):
        gist_write({GIST_LOCKS: json.dumps({"locked_teams": [], "locked_players": []}, indent=2)})
        st.success("Cleared locks (refresh page)")


# ==========================
# LOAD DVP (Book1.csv FORMAT)
# ==========================
def load_dvp_book1(text: str):
    if not text:
        return None, None

    dvp = pd.read_csv(StringIO(text))
    required = ["Sort: Position","Sort: Team","Sort: PTS","Sort: 3PM","Sort: REB","Sort: AST","Sort: STL","Sort: BLK","Sort: TO"]
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

    out = out.dropna(subset=["TEAM","POS","PTS","FG3M","REB","AST","STL","BLK","TOV"]).copy()
    out = out[(out["TEAM"] != "") & (out["POS"] != "")].copy()

    league_avg = out.groupby("POS")[["PTS","REB","AST","FG3M","STL","BLK","TOV"]].mean().reset_index()
    return (out, league_avg), None

dvp_pack = None
if dvp_text:
    dvp_pack, dvp_err = load_dvp_book1(dvp_text)
    if dvp_err:
        st.warning(dvp_err)
    else:
        st.sidebar.success("DvP loaded ✓")


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
            row = {
                **r.to_dict(),
                "Minutes": np.nan,
                **{c: np.nan for c in STAT_COLS},
                "Status": "ERR",
                "Notes": "No match in league stats (name mismatch)",
                "Matched_NBA_Name": "",
                "Matched_NBA_Team": "",
            }
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

        # Add debug so you can see if names are being matched incorrectly
        row = {
            **r.to_dict(),
            "Minutes": round(float(mins), 2),
            **stats,
            "Status": "OK",
            "Notes": notes,
            "Matched_NBA_Name": str(hit["NBA_Name"]),
            "Matched_NBA_Team": str(hit["NBA_Team"]),
        }

        # Guardrail: if team mismatches, mark ERR (prevents wrong stats on cheap slate guys)
        if r["Team"] and str(hit["NBA_Team"]) != str(r["Team"]):
            row["Status"] = "ERR"
            row["Notes"] = f"TEAM_MISMATCH (DK {r['Team']} vs NBA {hit['NBA_Team']})"

        rows.append(row)

    base = pd.DataFrame(rows)
    base["DK_FP"] = base.apply(lambda rr: dk_fp(rr) if rr["Status"] == "OK" else np.nan, axis=1)
    base["BASE_Minutes"] = base["Minutes"]  # keep original minutes for caps/weights
    gist_write({GIST_BASE: base.to_csv(index=False)})
    st.success("Saved BASE")
    st.dataframe(
        base[["Name","Team","Opp","PrimaryPos","Salary","Minutes","DK_FP","Status","Notes","Matched_NBA_Name","Matched_NBA_Team"]],
        use_container_width=True
    )


# ==========================
# STEP B — RUN PROJECTIONS (Injury minutes + opportunity + DvP)
# ==========================
st.divider()
st.subheader("Step B — Run Projections (Injury Bumps + DvP)")

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
    base["OppNotes"] = ""
    base["DvPNotes"] = ""
    base["DvPMult"] = 1.0

    # Apply OUT flags (from editor state)
    base.loc[base["Name_clean"].isin(out_set), "Status"] = "OUT"

    # Make sure numeric
    for c in STAT_COLS:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # Per-minute rates from BASE (before any bumps)
    for c in STAT_COLS:
        base[f"PM_{c}"] = np.where(
            (base["Status"] == "OK") & (base["Minutes"].fillna(0) > 0),
            base[c].fillna(0) / base["Minutes"].replace(0, np.nan),
            0.0
        )
        base[f"PM_{c}"] = base[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Role-based cap per player (starter vs bench) + absolute cap
    def minute_cap(row):
        bm = float(row["BASE_Minutes"]) if pd.notna(row["BASE_Minutes"]) else 0.0
        cap_role = STARTER_CAP if bm >= STARTER_MIN_CUTOFF else BENCH_CAP
        return min(cap_role, MAX_MINUTES_ABS)

    base["MIN_CAP"] = base.apply(minute_cap, axis=1)

    # --------------------------
    # 1) Minutes redistribution (fast)
    # --------------------------
    for team in base["Team"].dropna().unique():
        out_t = base[(base["Team"] == team) & (base["Status"] == "OUT")]
        ok_t = base[(base["Team"] == team) & (base["Status"] == "OK")]
        if out_t.empty or ok_t.empty:
            continue

        missing = float(out_t["Minutes"].fillna(0).sum())
        if missing <= 0:
            continue

        # Weight by baseline minutes (bench gets floor)
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

    # Recompute stats from per-minute using updated minutes (still fast)
    for idx in base.index[base["Status"] == "OK"]:
        m = base.loc[idx, "Minutes"]
        if pd.isna(m) or float(m) <= 0:
            continue
        for c in STAT_COLS:
            base.loc[idx, c] = round(float(base.loc[idx, f"PM_{c}"]) * float(m), 2)

    # --------------------------
    # 2) Opportunity redistribution (PTS/AST/FG3M/REB) (still fast, improves accuracy)
    # --------------------------
    base["UsageNotes"] = ""

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

        # Only redistribute meaningful removed opportunity
        if sum(removed.values()) <= 0:
            continue

        # Weights per category (proxy for role)
        w_pts = ok_t["PTS"].fillna(0).clip(lower=0.01)
        w_ast = ok_t["AST"].fillna(0).clip(lower=0.01)
        w_3   = ok_t["FG3M"].fillna(0).clip(lower=0.01)
        w_reb = ok_t["REB"].fillna(0).clip(lower=0.01)

        sums = {
            "PTS": float(w_pts.sum()),
            "AST": float(w_ast.sum()),
            "FG3M": float(w_3.sum()),
            "REB": float(w_reb.sum()),
        }

        # apply redistribution with caps (percent caps vs current)
        for idx in ok_t.index:
            # PTS
            if sums["PTS"] > 0 and removed["PTS"] > 0:
                add = removed["PTS"] * (float(w_pts.loc[idx]) / sums["PTS"])
                cap = float(base.loc[idx, "PTS"]) * BUMP_CAPS["PTS"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "PTS"] = round(float(base.loc[idx, "PTS"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" PTS+{add:.1f}"

            # AST
            if sums["AST"] > 0 and removed["AST"] > 0:
                add = removed["AST"] * (float(w_ast.loc[idx]) / sums["AST"])
                cap = float(base.loc[idx, "AST"]) * BUMP_CAPS["AST"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "AST"] = round(float(base.loc[idx, "AST"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" AST+{add:.1f}"

            # 3PM
            if sums["FG3M"] > 0 and removed["FG3M"] > 0:
                add = removed["FG3M"] * (float(w_3.loc[idx]) / sums["FG3M"])
                cap = float(base.loc[idx, "FG3M"]) * BUMP_CAPS["FG3M"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "FG3M"] = round(float(base.loc[idx, "FG3M"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" 3PM+{add:.1f}"

            # REB
            if sums["REB"] > 0 and removed["REB"] > 0:
                add = removed["REB"] * (float(w_reb.loc[idx]) / sums["REB"])
                cap = float(base.loc[idx, "REB"]) * BUMP_CAPS["REB"]
                add = min(add, cap)
                if add > 0:
                    base.loc[idx, "REB"] = round(float(base.loc[idx, "REB"]) + add, 2)
                    base.loc[idx, "UsageNotes"] += f" REB+{add:.1f}"

        # clean notes spacing
        base["UsageNotes"] = base["UsageNotes"].fillna("").astype(str).str.strip()

    # --------------------------
    # 3) Apply DvP vs opponent by position (light-touch)
    # --------------------------
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
            for c in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
                av = float(avg[c])
                al = float(allowed[c])
                mlt = (al / av) if av > 0 else 1.0
                mults[c] = clamp(mlt, DVP_CAP_LOW, DVP_CAP_HIGH)

            for c in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
                base.loc[idx, c] = round(float(base.loc[idx, c]) * mults[c], 2)

            base.loc[idx, "DvPMult"] = round(float(np.mean(list(mults.values()))), 4)
            base.loc[idx, "DvPNotes"] = (
                f"DVP {opp} {pos} "
                f"PTS{mults['PTS']:.2f} REB{mults['REB']:.2f} AST{mults['AST']:.2f}"
            )

    # DK FP
    base.loc[base["Status"] == "OK", "DK_FP"] = base[base["Status"] == "OK"].apply(dk_fp, axis=1)

    # Remove OUT from displayed pool
    final = base[(base["Status"] == "OK") & (~base["Name_clean"].isin(out_set))].copy()

    gist_write({GIST_FINAL: final.to_csv(index=False)})
    st.success("Saved FINAL")

    show_cols = [
        "Name","Team","Opp","PrimaryPos","Salary","Minutes","MIN_CAP",
        "PTS","REB","AST","FG3M","STL","BLK","TOV","DK_FP",
        "Notes","BumpNotes","UsageNotes","DvPNotes"
    ]
    st.dataframe(final[show_cols], use_container_width=True)


# ==========================
# OPTIMIZER (LATE SWAP)
# ==========================
st.divider()
st.subheader("Optimizer (Late Swap — respects Team Locks + Player LOCK)")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.info("No FINAL saved yet. Run Step A then Step B.")
    st.stop()

pool = pd.read_csv(StringIO(final_text))
pool["Positions"] = pool["Positions"].apply(eval)
pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
pool["DK_FP"] = pd.to_numeric(pool["DK_FP"], errors="coerce")
pool = pool.dropna(subset=["Salary","DK_FP"]).copy()
pool = pool[pool["Salary"] > 0].copy()

if "Name_clean" not in pool.columns:
    pool["Name_clean"] = pool["Name"].apply(clean_name)

started_teams = set(locked_teams)

def assign_locked_to_slots(locked_df):
    players = list(locked_df.index)
    cand = {i: [s for s in DK_SLOTS if eligible_for_slot(locked_df.loc[i,"Positions"], s)] for i in players}
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
            if backtrack(k+1):
                return True
            used_slots.remove(s)
            assignment.pop(s, None)
        return False

    ok = backtrack(0)
    return assignment if ok else None

if st.button("Optimize (respect locks)"):
    locked_df = pool[pool["Name_clean"].isin(locked_players_set)].copy()

    # Exclude started teams from NEW selections (but keep locked players even if started)
    candidate_df = pool.copy()
    if started_teams:
        candidate_df = candidate_df[~candidate_df["Team"].isin(started_teams)].copy()
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
        st.dataframe(lineup_df[["Slot","Locked","Name","Team","Salary","DK_FP","Minutes"]], use_container_width=True)
        st.metric("Total Salary", int(salary_locked))
        st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))
        st.stop()

    # only optimize over non-locked players
    opt_pool = candidate_df[~candidate_df["Name_clean"].isin(locked_players_set)].copy()

    prob = LpProblem("DK_LATE_SWAP", LpMaximize)

    x = {}
    for i, r in opt_pool.iterrows():
        for slot in remaining_slots:
            if eligible_for_slot(r["Positions"], slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    if not x:
        st.error("No feasible candidates for remaining slots (too many teams locked / too many players out).")
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
    st.dataframe(lineup_df[["Slot","Locked","Name","Team","Salary","Minutes","DK_FP"]], use_container_width=True)
    st.metric("Total Salary", int(lineup_df["Salary"].sum()))
    st.metric("Total DK FP", round(float(lineup_df["DK_FP"].sum()), 2))

    if started_teams:
        st.caption(f"Started/locked teams excluded from NEW selections: {', '.join(sorted(list(started_teams)))}")