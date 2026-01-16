import json
import time
import random
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

# Cloud stability
API_TIMEOUT_SECONDS = 35
GAMELOG_RETRIES = 4
BUILD_THROTTLE_SECONDS = 0.12

# Bumps
ENABLE_OUT_BUMPS = True
MIN_BENCH_FLOOR = 6.0
MAX_MINUTES_PLAYER = 40.0
MAX_MIN_INCREASE = 22.0

OFF_BUMP_STRENGTH = 0.22
OFF_BUMP_CAP = 1.25
OFF_BUMP_TOP_N = 6

# Gist file names
GIST_FILE_SLATE = "latest_dk_slate.csv"
GIST_FILE_OUT = "out_flags.json"
GIST_FILE_PROJ = "latest_projections.csv"
GIST_FILE_META = "meta.json"


# ==========================
# PAGE
# ==========================
st.set_page_config(page_title="DK NBA Optimizer", layout="wide", initial_sidebar_state="expanded")
st.title("DraftKings NBA Optimizer — Persistent Slate + OUT Checkboxes + On-demand Projections")


# ==========================
# Helpers / DK scoring
# ==========================
def clean_name(s: str) -> str:
    return " ".join(str(s).lower().replace(".", "").split())

def parse_minutes(s) -> float:
    s = str(s)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60.0

def dk_positions_list(p: str) -> list:
    return [x.strip().upper() for x in str(p).split("/") if x.strip()]

def slot_eligible(pos_list: list, slot: str) -> bool:
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

def dk_fp(stats: dict) -> float:
    pts = float(stats.get("PTS", 0))
    reb = float(stats.get("REB", 0))
    ast = float(stats.get("AST", 0))
    stl = float(stats.get("STL", 0))
    blk = float(stats.get("BLK", 0))
    tov = float(stats.get("TOV", 0))
    tpm = float(stats.get("FG3M", 0))

    fp = pts + 1.25 * reb + 1.5 * ast + 2 * stl + 2 * blk - 0.5 * tov + 0.5 * tpm

    # DK bonuses
    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2:
        fp += 1.5
    if cats_10 >= 3:
        fp += 3.0

    return round(fp, 2)

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.number)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


# ==========================
# GitHub Gist persistence
# ==========================
def _get_secret(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default

GITHUB_TOKEN = _get_secret("GITHUB_TOKEN", None)
GIST_ID = _get_secret("GIST_ID", None)

def gist_headers():
    if not GITHUB_TOKEN:
        return {}
    return {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def gist_get(gist_id: str) -> dict:
    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=gist_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def gist_update_files(gist_id: str, files: dict):
    payload = {"files": {fn: {"content": content} for fn, content in files.items()}}
    r = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=gist_headers(), json=payload, timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read_file(gist_json: dict, filename: str):
    files = gist_json.get("files", {})
    if filename not in files:
        return None

    if "content" in files[filename] and files[filename]["content"] is not None and not files[filename].get("truncated", False):
        return files[filename]["content"]

    raw_url = files[filename].get("raw_url")
    if not raw_url:
        return None
    r = requests.get(raw_url, timeout=20)
    r.raise_for_status()
    return r.text


# ==========================
# NBA player mapping + gamelog (retries)
# ==========================
@st.cache_data(ttl=86400)
def player_id_map():
    plist = nba_players.get_players()
    return {clean_name(p["full_name"]): int(p["id"]) for p in plist}

def get_player_id(name: str) -> int:
    key = clean_name(name)
    m = player_id_map()
    if key in m:
        return m[key]
    matches = nba_players.find_players_by_full_name(name)
    if matches:
        return int(matches[0]["id"])
    raise ValueError(f"Player not found: {name}")

@st.cache_data(ttl=3600)
def gamelog(pid: int):
    last_err = None
    for attempt in range(1, GAMELOG_RETRIES + 1):
        try:
            df = playergamelog.PlayerGameLog(
                player_id=pid,
                season=SEASON,
                timeout=API_TIMEOUT_SECONDS
            ).get_data_frames()[0]
            if df is None or df.empty:
                return pd.DataFrame(), "EMPTY_RESPONSE"
            return df, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep((1.6 ** attempt) + random.uniform(0.3, 0.9))
    return pd.DataFrame(), f"FAILED_AFTER_RETRIES ({last_err})"

def per_min_rates(gl: pd.DataFrame, n: int):
    work = gl.head(int(n)).copy()
    work["MIN_float"] = work["MIN"].apply(parse_minutes)
    work = work.dropna(subset=["MIN_float"])
    if work.empty:
        return None, None

    for c in STAT_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    tot_min = float(work["MIN_float"].sum())
    if tot_min <= 0:
        return None, None

    rates = {c: float(work[c].sum()) / tot_min for c in STAT_COLS}
    avg_min = float(work["MIN_float"].mean())
    return rates, avg_min

def project_player(name: str, last_n: int):
    pid = get_player_id(name)
    gl, err = gamelog(pid)
    if gl is None or gl.empty:
        raise RuntimeError(f"NO_GAMELOG ({err})")

    rates, mins = per_min_rates(gl, last_n)
    if rates is None or mins is None or mins <= 0:
        raise RuntimeError("NO_MINUTES_IN_SAMPLE")

    proj_stats = {c: round(float(rates[c] * mins), 2) for c in STAT_COLS}
    fp = dk_fp(proj_stats)
    return round(float(mins), 2), float(fp), proj_stats


# ==========================
# DK CSV loader
# ==========================
def read_dk_csv(uploaded_or_text) -> pd.DataFrame:
    if hasattr(uploaded_or_text, "read"):
        df = pd.read_csv(uploaded_or_text)
    else:
        df = pd.read_csv(StringIO(uploaded_or_text))

    required = ["Name", "Position", "Salary", "TeamAbbrev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DK CSV missing required columns: {', '.join(missing)}")

    out = pd.DataFrame({
        "Name": df["Name"].astype(str),
        "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
        "Positions": df["Position"].astype(str).apply(dk_positions_list),
        "Team": df["TeamAbbrev"].astype(str).str.upper(),
    })
    out["Name_clean"] = out["Name"].apply(clean_name)

    if "Game Info" in df.columns:
        out["GameInfo"] = df["Game Info"].astype(str)
    else:
        out["GameInfo"] = ""

    return out


# ==========================
# OUT bumps
# ==========================
def apply_out_bumps(full_proj_df: pd.DataFrame) -> pd.DataFrame:
    df = full_proj_df.copy()

    ok_mask = df["Status"].astype(str) == "OK"
    out_mask = df["Status"].astype(str) == "OUT"

    df["BaseMinutes"] = df["Minutes"]
    for c in STAT_COLS:
        df[f"PM_{c}"] = np.where(
            df["BaseMinutes"].fillna(0) > 0,
            df[c].fillna(0) / df["BaseMinutes"].replace(0, np.nan),
            0.0
        )
        df[f"PM_{c}"] = df[f"PM_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "Notes" not in df.columns:
        df["Notes"] = ""

    for team in df.loc[out_mask, "Team"].dropna().unique():
        team_out = df[(df["Team"] == team) & out_mask].copy()
        team_ok = df[(df["Team"] == team) & ok_mask].copy()
        if team_out.empty or team_ok.empty:
            continue

        missing_min = float(team_out["BaseMinutes"].fillna(0).sum())
        if missing_min <= 0:
            continue

        weights = team_ok["BaseMinutes"].fillna(0).clip(lower=MIN_BENCH_FLOOR)
        wsum = float(weights.sum())
        if wsum <= 0:
            continue
        weights = weights / wsum

        # Minutes redistribution
        for idx in team_ok.index:
            inc = min(float(weights.loc[idx]) * missing_min, MAX_MIN_INCREASE)
            old_m = safe_float(df.at[idx, "Minutes"], 0.0)
            new_m = min(old_m + inc, MAX_MINUTES_PLAYER)
            df.at[idx, "Minutes"] = new_m

            for c in STAT_COLS:
                df.at[idx, c] = round(safe_float(df.at[idx, f"PM_{c}"], 0.0) * new_m, 2)

            if inc > 0:
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" MIN+{inc:.1f}").strip()

        # Small offense bump
        team_ok2 = df[(df["Team"] == team) & ok_mask].copy()
        team_ok2["rank"] = team_ok2["Minutes"].fillna(0) + 40.0 * team_ok2["PM_AST"].fillna(0)
        recipients = team_ok2.sort_values("rank", ascending=False).head(OFF_BUMP_TOP_N)

        missing_off = float((team_out["PTS"].fillna(0) + team_out["AST"].fillna(0) + team_out["FG3M"].fillna(0)).sum())
        recipient_off = float((recipients["PTS"].fillna(0) + recipients["AST"].fillna(0) + recipients["FG3M"].fillna(0)).sum())

        if missing_off > 0 and recipient_off > 0:
            raw_mult = 1.0 + OFF_BUMP_STRENGTH * (missing_off / recipient_off)
            mult = min(raw_mult, OFF_BUMP_CAP)
            for idx in recipients.index:
                df.at[idx, "PTS"] = round(safe_float(df.at[idx, "PTS"], 0.0) * mult, 2)
                df.at[idx, "AST"] = round(safe_float(df.at[idx, "AST"], 0.0) * mult, 2)
                df.at[idx, "FG3M"] = round(safe_float(df.at[idx, "FG3M"], 0.0) * mult, 2)
                df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" OFFx{mult:.2f}").strip()

        # Recompute DK_FP / Value
        for idx in df.index[(df["Team"] == team) & ok_mask]:
            stats = {c: safe_float(df.at[idx, c], 0.0) for c in STAT_COLS}
            df.at[idx, "DK_FP"] = dk_fp(stats)
            sal = safe_float(df.at[idx, "Salary"], 0.0)
            df.at[idx, "Value"] = (df.at[idx, "DK_FP"] / (sal / 1000.0)) if sal > 0 else np.nan

    return df


# ==========================
# Optimizer
# ==========================
def optimize(pool_df: pd.DataFrame) -> pd.DataFrame:
    prob = LpProblem("DK_NBA_CLASSIC", LpMaximize)

    rows = pool_df.to_dict("records")
    n = len(rows)
    if n == 0:
        raise ValueError("Optimizer pool is empty (no projections).")

    x = {}
    for i in range(n):
        for slot in DK_SLOTS:
            if slot_eligible(rows[i]["Positions"], slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    prob += lpSum(rows[i]["DK_FP"] * x[(i, slot)] for (i, slot) in x)

    for slot in DK_SLOTS:
        prob += lpSum(x[(i, slot)] for i in range(n) if (i, slot) in x) == 1

    for i in range(n):
        prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1

    prob += lpSum(rows[i]["Salary"] * x[(i, slot)] for (i, slot) in x) <= DK_SALARY_CAP

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup_rows = []
    for slot in DK_SLOTS:
        chosen_i = None
        for i in range(n):
            if (i, slot) in x and x[(i, slot)].value() == 1:
                chosen_i = i
                break
        if chosen_i is None:
            raise ValueError("No feasible lineup found. (Pool too small / position constraints.)")

        r = rows[chosen_i]
        lineup_rows.append({
            "Slot": slot,
            "Name": r["Name"],
            "Team": r["Team"],
            "PosEligible": "/".join(r["Positions"]),
            "Salary": int(r["Salary"]),
            "Minutes": r["Minutes"],
            "PTS": r["PTS"],
            "REB": r["REB"],
            "AST": r["AST"],
            "3PM": r["FG3M"],
            "STL": r["STL"],
            "BLK": r["BLK"],
            "TOV": r["TOV"],
            "DK_FP": round(float(r["DK_FP"]), 2),
            "Value": r["Value"],
            "Notes": r.get("Notes", ""),
        })

    return pd.DataFrame(lineup_rows)


# ==========================
# Required secrets
# ==========================
if not (GITHUB_TOKEN and GIST_ID):
    st.error("Missing Streamlit Secrets. Add GITHUB_TOKEN and GIST_ID in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# ==========================
# Sidebar controls
# ==========================
uploaded = st.sidebar.file_uploader("Upload DraftKings CSV (saves for phone too)", type="csv")
last_n = st.sidebar.number_input("Recent games window (N)", 1, 30, int(st.session_state.get("last_n", DEFAULT_LAST_N)))
st.session_state["last_n"] = int(last_n)

reload_btn = st.sidebar.button("Reload saved slate/projections")


# ==========================
# Load from gist (or upload overrides)
# ==========================
gist_json = gist_get(GIST_ID) if (reload_btn or "gist_json" not in st.session_state) else st.session_state["gist_json"]
st.session_state["gist_json"] = gist_json

slate_df = None
if uploaded is not None:
    csv_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    gist_update_files(GIST_ID, {
        GIST_FILE_SLATE: csv_text,
        GIST_FILE_META: json.dumps({"season": SEASON, "saved_at": time.time()}, indent=2),
    })
    st.sidebar.success("Saved slate. Open on phone and it will load the same slate.")
    slate_df = read_dk_csv(csv_text)
else:
    slate_text = gist_read_file(gist_json, GIST_FILE_SLATE)
    if not slate_text:
        st.info("Upload a DK CSV in the sidebar to begin.")
        st.stop()
    slate_df = read_dk_csv(slate_text)

# Load saved projections (for display without recompute)
proj_text = gist_read_file(gist_json, GIST_FILE_PROJ)

# Load saved OUT flags (we still read them, but projections will use LIVE checkboxes)
out_flags_text = gist_read_file(gist_json, GIST_FILE_OUT)
saved_out_flags = {}
if out_flags_text:
    try:
        saved_out_flags = json.loads(out_flags_text)
    except Exception:
        saved_out_flags = {}

# Add OUT checkbox column (defaults from saved)
slate_view = slate_df.copy()
slate_view["OUT"] = slate_view["Name_clean"].map(lambda k: bool(saved_out_flags.get(k, False)))

st.write(f"**Season:** {SEASON}  |  **Cap:** ${DK_SALARY_CAP:,}  |  **Slots:** {', '.join(DK_SLOTS)}")
st.subheader("Step 1 — Mark OUT players (checkbox). Projections only run when you click Run Projections.")

edited = st.data_editor(
    slate_view[["OUT", "Name", "Team", "Positions", "Salary", "GameInfo"]],
    use_container_width=True,
    hide_index=True,
    column_config={"OUT": st.column_config.CheckboxColumn("OUT")},
    disabled=["Name", "Team", "Positions", "Salary", "GameInfo"],
)

# ---- LIVE OUT SET FROM CHECKBOXES ----
name_to_clean = dict(zip(slate_view["Name"], slate_view["Name_clean"]))
live_flags = {}
for _, row in edited.iterrows():
    nm = str(row["Name"])
    cl = name_to_clean.get(nm, clean_name(nm))
    live_flags[cl] = bool(row["OUT"])
out_clean_set = {k for k, v in live_flags.items() if v}

# Show last saved projections (no recompute)
if proj_text:
    try:
        last_proj = pd.read_csv(StringIO(proj_text))
        st.subheader("Last Saved Projections (no recompute)")
        st.dataframe(last_proj, use_container_width=True, hide_index=True)
    except Exception:
        pass

st.divider()
st.subheader("Step 2 — Run projections (only when you click)")

run_proj = st.button("Run Projections (apply OUT bumps, remove OUT from projections, save results)")
latest_display_df = None

if run_proj:
    # Persist OUT selections automatically on projection run (phone + desktop sync)
    gist_update_files(GIST_ID, {GIST_FILE_OUT: json.dumps(live_flags, indent=2)})

    rows = []
    total = len(slate_df)
    progress = st.progress(0, text="Starting projections...")

    for i, r in enumerate(slate_df.itertuples(index=False), start=1):
        progress.progress(int(i / max(total, 1) * 100), text=f"Running gamelog({r.Name}) ({i}/{total})")

        status = "OUT" if r.Name_clean in out_clean_set else "OK"

        try:
            mins, fp, stats = project_player(r.Name, st.session_state["last_n"])
            sal = safe_float(r.Salary, np.nan)
            value = (fp / (sal / 1000.0)) if (pd.notna(sal) and sal > 0) else np.nan

            rows.append({
                "Name": r.Name,
                "Name_clean": r.Name_clean,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": sal,
                "Minutes": mins,
                "PTS": stats["PTS"],
                "REB": stats["REB"],
                "AST": stats["AST"],
                "FG3M": stats["FG3M"],
                "STL": stats["STL"],
                "BLK": stats["BLK"],
                "TOV": stats["TOV"],
                "DK_FP": fp,
                "Value": value,
                "Status": status,
            })
        except Exception as e:
            rows.append({
                "Name": r.Name,
                "Name_clean": r.Name_clean,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": safe_float(r.Salary, np.nan),
                "Minutes": np.nan,
                "PTS": np.nan, "REB": np.nan, "AST": np.nan, "FG3M": np.nan, "STL": np.nan, "BLK": np.nan, "TOV": np.nan,
                "DK_FP": np.nan,
                "Value": np.nan,
                "Status": f"ERR: {str(e)[:90]}",
            })

        time.sleep(BUILD_THROTTLE_SECONDS)

    progress.empty()

    full_proj_df = pd.DataFrame(rows)

    # Apply bumps using OUT players in background
    if ENABLE_OUT_BUMPS and out_clean_set:
        full_proj_df = apply_out_bumps(full_proj_df)

    # BULLETPROOF OUT REMOVAL
    if "Name_clean" not in full_proj_df.columns:
        full_proj_df["Name_clean"] = full_proj_df["Name"].apply(clean_name)

    proj_df = full_proj_df[
        (full_proj_df["Status"] == "OK") &
        (~full_proj_df["Name_clean"].isin(out_clean_set))
    ].copy()

    # Save projections with FG3M (do NOT rename to 3PM before saving)
    display_df = proj_df.copy().sort_values("DK_FP", ascending=False)

    gist_update_files(GIST_ID, {
        GIST_FILE_PROJ: display_df.to_csv(index=False),
        GIST_FILE_META: json.dumps({"season": SEASON, "saved_at": time.time(), "last_n": int(st.session_state["last_n"])}, indent=2),
    })

    st.success("Projections complete and saved. Phone/desktop will show these without re-running.")
    st.subheader("Projections (OUT removed, bumps applied)")
    # Show 3PM label for humans (but keep FG3M in data for optimizer)
    st.dataframe(display_df.rename(columns={"FG3M": "3PM"}), use_container_width=True, hide_index=True)

    latest_display_df = display_df

st.divider()
st.subheader("Optimizer (uses saved projections)")

# Load projections for optimizer
pool_df = None
if latest_display_df is not None:
    pool_df = latest_display_df.copy()
else:
    if proj_text:
        try:
            pool_df = pd.read_csv(StringIO(proj_text))
        except Exception:
            pool_df = None

if pool_df is None or pool_df.empty:
    st.info("No projections saved yet. Run projections first.")
    st.stop()

# ---- FIX: FG3M vs 3PM (handles older saved projection files) ----
if "FG3M" not in pool_df.columns and "3PM" in pool_df.columns:
    pool_df["FG3M"] = pool_df["3PM"]

# Restore Positions to list after CSV reload
def parse_positions_cell(x):
    if isinstance(x, list):
        return x
    s = str(x)
    if s.startswith("[") and s.endswith("]"):
        s2 = s.strip("[]").replace("'", "").replace('"', "")
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        return parts
    if "/" in s:
        return dk_positions_list(s)
    return [s.strip()] if s.strip() else []

pool_df["Positions"] = pool_df["Positions"].apply(parse_positions_cell)

run_opt = st.button("Optimize Lineup (Classic)")
if run_opt:
    p = pool_df.copy()
    p = p.dropna(subset=["Salary", "DK_FP"]).copy()
    p = p[p["Salary"] > 0].copy()

    try:
        lineup = optimize(p)

        st.subheader("Optimized Lineup")
        st.dataframe(lineup, use_container_width=True, hide_index=True)
        st.metric("Total Salary", int(lineup["Salary"].sum()))
        st.metric("Total DK FP", round(float(lineup["DK_FP"].sum()), 2))

        out_csv = lineup.to_csv(index=False).encode("utf-8")
        st.download_button("Download Lineup CSV", data=out_csv, file_name="dk_optimized_lineup.csv", mime="text/csv")

    except Exception as e:
        st.error(str(e))
