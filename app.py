import re
import time
import numpy as np
import pandas as pd
import streamlit as st

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import playergamelog

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
DEFAULT_LAST_N = 10

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

st.set_page_config(page_title="DK NBA Optimizer", layout="wide")
st.title("DraftKings NBA Classic – Projections + Optimizer")

# ==========================
# DK scoring
# ==========================
def dk_fp(stats: dict) -> float:
    pts = float(stats.get("PTS", 0))
    reb = float(stats.get("REB", 0))
    ast = float(stats.get("AST", 0))
    stl = float(stats.get("STL", 0))
    blk = float(stats.get("BLK", 0))
    tov = float(stats.get("TOV", 0))
    tpm = float(stats.get("FG3M", 0))

    fp = pts + 1.25 * reb + 1.5 * ast + 2 * stl + 2 * blk - 0.5 * tov + 0.5 * tpm

    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2:
        fp += 1.5
    if cats_10 >= 3:
        fp += 3.0

    return round(fp, 2)

# ==========================
# Helpers
# ==========================
def clean_name(s: str) -> str:
    return " ".join((s or "").lower().replace(".", "").split())

def parse_minutes(s) -> float:
    s = str(s)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60.0

def dk_positions_list(pos: str) -> list:
    if not isinstance(pos, str) or not pos.strip():
        return []
    return [p.strip().upper() for p in pos.split("/") if p.strip()]

def slot_eligible(pos_list: list, slot: str) -> bool:
    pos = set(pos_list)
    if slot in ["PG", "SG", "SF", "PF", "C"]:
        return slot in pos
    if slot == "G":
        return ("PG" in pos) or ("SG" in pos)
    if slot == "F":
        return ("SF" in pos) or ("PF" in pos)
    if slot == "UTIL":
        return len(pos) > 0
    return False

def parse_out_list(text: str) -> set:
    if not text:
        return set()
    text = text.replace(",", "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return {clean_name(p) for p in parts}

def parse_game_info(s: str):
    # DK: "LAL@BOS 07:30PM ET"
    if not isinstance(s, str):
        return None, None
    m = re.search(r"\b([A-Z]{2,4})@([A-Z]{2,4})\b", s.upper())
    if not m:
        return None, None
    return m.group(1), m.group(2)

# ==========================
# NBA lookup caches
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
    # fallback: search
    matches = nba_players.find_players_by_full_name(name)
    if not matches:
        raise ValueError(f"Player not found: {name}")
    return int(matches[0]["id"])

@st.cache_data(ttl=86400)
def team_map():
    return {t["abbreviation"].upper(): t for t in nba_teams.get_teams()}

# ==========================
# NBA data
# ==========================
@st.cache_data(ttl=3600)
def gamelog(pid: int) -> pd.DataFrame:
    """
    Timeout + fallback so one bad player can't hang the app.
    """
    try:
        df = playergamelog.PlayerGameLog(player_id=pid, season=SEASON, timeout=10).get_data_frames()[0]
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def per_min_rates(gl: pd.DataFrame, n: int):
    if gl is None or gl.empty:
        return {c: 0.0 for c in STAT_COLS}, 0.0

    work = gl.head(n).copy()
    work["MIN_float"] = work["MIN"].apply(parse_minutes)
    work = work.dropna(subset=["MIN_float"])
    if work.empty:
        return {c: 0.0 for c in STAT_COLS}, 0.0

    for c in STAT_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    tot_min = float(work["MIN_float"].sum())
    if tot_min <= 0:
        return {c: 0.0 for c in STAT_COLS}, 0.0

    rates = {c: float(work[c].sum()) / tot_min for c in STAT_COLS}
    avg_min = float(work["MIN_float"].mean())
    return rates, avg_min

# ==========================
# Projection (returns stats now)
# ==========================
def project_player(name: str, last_n: int):
    pid = get_player_id(name)
    gl = gamelog(pid)

    rates, mins = per_min_rates(gl, last_n)

    proj_stats = {c: round(float(rates[c] * mins), 2) for c in STAT_COLS}
    fp = dk_fp(proj_stats)

    return round(float(mins), 2), float(fp), proj_stats

# ==========================
# DK CSV loader (YOUR HEADERS)
# ==========================
def read_dk_csv(file):
    df = pd.read_csv(file)

    required = ["Name", "Position", "Salary", "TeamAbbrev", "Game Info"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DK CSV missing required columns: {', '.join(missing)}")

    out = pd.DataFrame({
        "Name": df["Name"].astype(str),
        "Salary": pd.to_numeric(df["Salary"], errors="coerce"),
        "Positions": df["Position"].astype(str).apply(dk_positions_list),
        "Team": df["TeamAbbrev"].astype(str).str.upper(),
        "GameInfo": df["Game Info"].astype(str),
    })
    out["Name_clean"] = out["Name"].apply(clean_name)

    # opponent abbrev from Game Info
    away, home, opp = [], [], []
    for gi, tm in zip(out["GameInfo"], out["Team"]):
        a, h = parse_game_info(gi)
        away.append(a); home.append(h)
        if a and h:
            opp.append(h if tm == a else a)
        else:
            opp.append(None)

    out["Away"] = away
    out["Home"] = home
    out["Opp"] = opp

    return out

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.subheader("Upload DK CSV")
    dk_file = st.file_uploader("DraftKings NBA CSV", type="csv")

    st.divider()
    st.subheader("Projection settings")
    last_n = st.number_input("Recent games window (N)", 1, 30, st.session_state.get("last_n", DEFAULT_LAST_N))
    st.session_state["last_n"] = int(last_n)

    st.divider()
    st.subheader("Late scratches")
    out_text = st.text_area("Players OUT (one per line or comma-separated)", value=st.session_state.get("out_text", ""), height=130)
    st.session_state["out_text"] = out_text

    exclude_out = st.checkbox("Exclude OUT players from pool", value=st.session_state.get("exclude_out", True))
    st.session_state["exclude_out"] = bool(exclude_out)

    st.divider()
    st.subheader("Optimizer constraints")
    max_team = st.number_input("Max players per team (0 = no limit)", 0, 8, st.session_state.get("max_team", 4))
    st.session_state["max_team"] = int(max_team)

st.write(f"**Season:** {SEASON}  |  **Salary cap:** ${DK_SALARY_CAP:,}  |  **Slots:** {', '.join(DK_SLOTS)}")

if not dk_file:
    st.info("Upload your DraftKings NBA CSV to begin.")
    st.stop()

# ==========================
# Load slate
# ==========================
try:
    slate = read_dk_csv(dk_file)
except Exception as e:
    st.error(str(e))
    st.stop()

manual_out = parse_out_list(st.session_state.get("out_text", ""))
slate["IsOutManual"] = slate["Name_clean"].isin(manual_out)

st.subheader("Slate (from DK CSV)")
st.dataframe(
    slate[["Name", "Team", "Positions", "Salary", "GameInfo", "Opp", "IsOutManual"]],
    use_container_width=True,
    hide_index=True
)

# ==========================
# Build projections
# ==========================
st.subheader("Projections (Full Slate)")

build_btn = st.button("Build/Refresh Projections")

if "proj_df" not in st.session_state:
    st.session_state["proj_df"] = None

def build_projections(slate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(slate_df)
    prog = st.progress(0, text="Projecting players...")

    for i, r in enumerate(slate_df.itertuples(index=False), start=1):
        prog.progress(int(i / max(total, 1) * 100), text=f"Running gamelog({r.Name}) ({i}/{total})...")

        # if OUT and excluding, keep row with zeros so you can still see them
        if r.IsOutManual and st.session_state["exclude_out"]:
            rows.append({
                "Name": r.Name,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": float(r.Salary) if pd.notna(r.Salary) else np.nan,
                "Minutes": 0.0,
                "PTS": 0.0, "REB": 0.0, "AST": 0.0, "3PM": 0.0, "STL": 0.0, "BLK": 0.0, "TOV": 0.0,
                "DK_FP": 0.0,
                "Value": 0.0,
                "Status": "OUT",
            })
            continue

        try:
            mins, fp, stats = project_player(r.Name, int(st.session_state["last_n"]))
            value = (fp / (float(r.Salary) / 1000.0)) if (pd.notna(r.Salary) and float(r.Salary) > 0) else np.nan

            rows.append({
                "Name": r.Name,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": float(r.Salary) if pd.notna(r.Salary) else np.nan,
                "Minutes": mins,
                "PTS": stats["PTS"],
                "REB": stats["REB"],
                "AST": stats["AST"],
                "3PM": stats["FG3M"],
                "STL": stats["STL"],
                "BLK": stats["BLK"],
                "TOV": stats["TOV"],
                "DK_FP": fp,
                "Value": value,
                "Status": "OK" if not r.IsOutManual else "OUT (included)",
            })
        except Exception:
            rows.append({
                "Name": r.Name,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": float(r.Salary) if pd.notna(r.Salary) else np.nan,
                "Minutes": np.nan,
                "PTS": np.nan, "REB": np.nan, "AST": np.nan, "3PM": np.nan, "STL": np.nan, "BLK": np.nan, "TOV": np.nan,
                "DK_FP": np.nan,
                "Value": np.nan,
                "Status": "ERR",
            })

    prog.empty()
    return pd.DataFrame(rows)

if build_btn or st.session_state["proj_df"] is None:
    st.session_state["proj_df"] = build_projections(slate)

proj_df = st.session_state["proj_df"].copy()
proj_df = proj_df.sort_values("DK_FP", ascending=False)

st.dataframe(
    proj_df[["Name","Team","Positions","Salary","Minutes","PTS","REB","AST","3PM","STL","BLK","TOV","DK_FP","Value","Status"]],
    use_container_width=True,
    hide_index=True
)

# ==========================
# Build optimizer pool
# ==========================
pool = proj_df.copy()
pool = pool.dropna(subset=["Salary", "DK_FP"])
pool = pool[pool["Salary"] > 0]
pool = pool[pool["Status"].isin(["OK", "OUT (included)"])]

if st.session_state["exclude_out"]:
    pool = pool[pool["Status"] != "OUT"]

# Locks / Excludes
st.subheader("Optimizer Controls")
pool_names = pool["Name"].tolist()
lock_players = st.multiselect("Lock players (force into lineup)", options=pool_names, default=[])
exclude_players = st.multiselect("Exclude players", options=pool_names, default=[])

pool = pool[~pool["Name"].isin(exclude_players)].copy()

# ==========================
# Optimizer (FIXED)
# ==========================
def optimize(pool_df: pd.DataFrame) -> pd.DataFrame:
    prob = LpProblem("DK_NBA_CLASSIC", LpMaximize)

    rows = list(pool_df.itertuples(index=False))
    n = len(rows)
    if n == 0:
        raise ValueError("Optimizer pool is empty. (Check OUT/exclude settings.)")

    # decision vars: x[i,slot]
    x = {}
    for i, r in enumerate(rows):
        for slot in DK_SLOTS:
            if slot_eligible(r.Positions, slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    # objective
    prob += lpSum(rows[i].DK_FP * x[(i, slot)] for (i, slot) in x)

    # each slot exactly once
    for slot in DK_SLOTS:
        prob += lpSum(x[(i, slot)] for i in range(n) if (i, slot) in x) == 1, f"fill_{slot}"

    # each player at most once
    for i in range(n):
        prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1, f"use_once_{i}"

    # salary cap
    prob += lpSum(rows[i].Salary * x[(i, slot)] for (i, slot) in x) <= DK_SALARY_CAP, "salary_cap"

    # max players per team
    max_team = int(st.session_state.get("max_team", 0))
    if max_team and max_team > 0:
        teams_unique = sorted(set(r.Team for r in rows))
        for t in teams_unique:
            idxs = [i for i, r in enumerate(rows) if r.Team == t]
            prob += lpSum(x[(i, slot)] for i in idxs for slot in DK_SLOTS if (i, slot) in x) <= max_team, f"max_{t}"

    # locks
    for lp in lock_players:
        idxs = [i for i, r in enumerate(rows) if r.Name == lp]
        if idxs:
            i = idxs[0]
            prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) == 1, f"lock_{i}"

    # solve
    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = []
    for slot in DK_SLOTS:
        chosen_i = None
        for i in range(n):
            if (i, slot) in x and x[(i, slot)].value() == 1:
                chosen_i = i
                break
        if chosen_i is None:
            raise ValueError("No feasible lineup found. Try loosening constraints / locks.")

        r = rows[chosen_i]
        lineup.append({
            "Slot": slot,
            "Name": r.Name,
            "Team": r.Team,
            "Salary": int(r.Salary),
            "Minutes": r.Minutes,
            "PTS": r.PTS,
            "REB": r.REB,
            "AST": r.AST,
            "3PM": r._8 if hasattr(r, "_8") else r.__getattribute__("3PM") if hasattr(r, "3PM") else r.__getattribute__("3PM") if False else r.PTS,  # safe fallback (not used)
            "DK_FP": round(float(r.DK_FP), 2),
        })

    lineup_df = pd.DataFrame(lineup)

    # The "3PM" in itertuples can become an invalid attribute; replace from pool_df merge
    lineup_df = lineup_df.merge(
        pool_df[["Name", "3PM"]], on="Name", how="left", suffixes=("", "_fix")
    )
    lineup_df["3PM"] = lineup_df["3PM_fix"]
    lineup_df = lineup_df.drop(columns=["3PM_fix"])

    return lineup_df

st.subheader("Optimizer")
run_opt = st.button("Optimize Lineup")

if run_opt:
    try:
        lineup = optimize(pool)

        st.subheader("Optimized Lineup")
        st.dataframe(lineup.sort_values("Slot"), use_container_width=True, hide_index=True)

        st.metric("Total Salary", int(lineup["Salary"].sum()))
        st.metric("Total DK FP", round(float(lineup["DK_FP"].sum()), 2))

        out_csv = lineup.sort_values("Slot").to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Lineup CSV",
            data=out_csv,
            file_name="dk_optimized_lineup.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))
