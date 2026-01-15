import re
import time
import random
import numpy as np
import pandas as pd
import streamlit as st

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

# Use FG3M internally (safe column), display as 3PM
STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

# Streamlit Cloud stability
API_TIMEOUT_SECONDS = 35
GAMELOG_RETRIES = 4
BUILD_THROTTLE_SECONDS = 0.15
RETRY_THROTTLE_SECONDS = 0.20

# OUT teammate bumps (minutes + offense)
ENABLE_OUT_BUMPS = True

# Minutes redistribution behavior
MIN_REDIS_TOP_N = 8          # distribute minutes across top N weighted teammates
MIN_BENCH_FLOOR = 6.0        # bench guys still get some weight
MAX_MINUTES_PLAYER = 40.0    # cap projected minutes after bumps
MAX_MIN_INCREASE = 22.0      # cap how many minutes a single player can gain from bumps

# Offense bump behavior (PTS/AST/3PM per-minute boost)
OFF_BUMP_STRENGTH = 0.22     # how aggressive the per-minute offense bump is
OFF_BUMP_CAP = 1.25          # max multiplier to PTS/AST/FG3M per-minute
OFF_BUMP_TOP_N = 6           # apply per-minute offense bump to top N recipients

st.set_page_config(page_title="DK NBA Optimizer", layout="wide")
st.title("DraftKings NBA Classic – Projections + Optimizer (Cloud-safe + OUT bumps)")

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

    matches = nba_players.find_players_by_full_name(name)
    if matches:
        return int(matches[0]["id"])

    raise ValueError(f"Player not found: {name}")

# ==========================
# NBA data (Cloud-safe: retries + backoff)
# ==========================
@st.cache_data(ttl=3600)
def gamelog(pid: int):
    """
    Returns (df, err_msg).
    Retries + backoff + jitter to survive Streamlit Cloud throttling/timeouts.
    """
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
            sleep_s = (1.6 ** attempt) + random.uniform(0.3, 0.9)
            time.sleep(sleep_s)

    return pd.DataFrame(), f"FAILED_AFTER_RETRIES ({last_err})"

def per_min_rates(gl: pd.DataFrame, n: int):
    work = gl.head(n).copy()
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

# ==========================
# Projection (empty/fail => ERR, not OK zeros)
# ==========================
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

    opp = []
    for gi, tm in zip(out["GameInfo"], out["Team"]):
        a, h = parse_game_info(gi)
        if a and h:
            opp.append(h if tm == a else a)
        else:
            opp.append(None)
    out["Opp"] = opp

    return out

# ==========================
# OUT bumps (minutes + offense)
# ==========================
def _pos_overlap(a: list, b: list) -> bool:
    return len(set(a or []) & set(b or [])) > 0

def apply_out_bumps(proj_df: pd.DataFrame, slate_df: pd.DataFrame, out_clean_set: set) -> pd.DataFrame:
    """
    Applies two adjustments when players are manually OUT:
    1) Minutes redistribution: missing minutes get redistributed (with extra weight to the "backup" position group)
    2) Offense bump: a per-minute multiplier to PTS/AST/FG3M for key recipients (cap applied)

    IMPORTANT: This uses only data we already have (no extra endpoints), so it stays Cloud-friendly.
    """
    if proj_df is None or proj_df.empty or not out_clean_set:
        return proj_df

    df = proj_df.copy()

    # Ensure baseline columns exist (for transparency)
    if "BaseMinutes" not in df.columns:
        df["BaseMinutes"] = df["Minutes"]
    if "BaseDK_FP" not in df.columns:
        df["BaseDK_FP"] = df["DK_FP"]
    if "Notes" not in df.columns:
        df["Notes"] = ""

    # Map name_clean -> positions from slate (more reliable than tuple attr)
    pos_map = dict(zip(slate_df["Name_clean"], slate_df["Positions"]))

    # Identify OUT players in projection df
    df["Name_clean"] = df["Name"].apply(clean_name)
    out_mask = df["Name_clean"].isin(out_clean_set)

    # Only bump teammates that have valid projections (Status OK)
    ok_mask = df["Status"].astype(str).str.startswith("OK")
    # OUT players used only as "removed minutes/usage" sources if we have a projection
    out_with_proj = out_mask & df["Status"].astype(str).isin(["OUT", "OUT (included)"])

    # If user marked players OUT but they didn't project, skip bumps (no minutes to redistribute)
    if out_with_proj.sum() == 0:
        return df

    # Pre-compute per-minute baseline rates from projections
    for c in STAT_COLS:
        pm_col = f"PM_{c}"
        df[pm_col] = np.where(
            df["BaseMinutes"].fillna(0) > 0,
            df[c].fillna(0) / df["BaseMinutes"].replace(0, np.nan),
            0.0
        )
        df[pm_col] = df[pm_col].fillna(0.0)

    # Team-by-team adjustments
    teams = sorted(df.loc[out_with_proj, "Team"].dropna().unique())
    for team in teams:
        team_out = df[(df["Team"] == team) & out_with_proj].copy()
        team_ok = df[(df["Team"] == team) & ok_mask].copy()

        if team_out.empty or team_ok.empty:
            continue

        # Total missing minutes and missing "offense involvement" proxy
        missing_minutes = float(team_out["BaseMinutes"].fillna(0).sum())

        # Offense proxy: PTS + AST + FG3M (from baseline projections)
        missing_off = float(
            (team_out["PTS"].fillna(0) + team_out["AST"].fillna(0) + team_out["FG3M"].fillna(0)).sum()
        )

        if missing_minutes <= 0:
            continue

        # Recipients: consider everyone OK on team, but weight top candidates
        cand = team_ok.copy()

        # Base weight uses minutes (starters more likely to soak minutes),
        # but keep a floor so bench players can jump when their position matches.
        cand["w_base"] = cand["BaseMinutes"].fillna(0).clip(lower=MIN_BENCH_FLOOR)

        # Add position-match bonus:
        # If a starting PG is OUT, the backup PG/SG on that team should gain the most minutes.
        # We'll do this by adding weight to candidates whose DK eligibility overlaps with OUT player eligibility.
        cand["w_pos"] = 0.0
        for out_row in team_out.itertuples(index=False):
            out_key = getattr(out_row, "Name_clean")
            out_pos = pos_map.get(out_key, []) or []
            # Bench boost: give extra bonus to lower-minute guys who overlap positions
            for idx, r in cand.iterrows():
                if _pos_overlap(r["Positions"], out_pos):
                    # base bonus
                    bonus = 10.0
                    # extra if bench-ish (likely direct backup)
                    if safe_float(r["BaseMinutes"], 0) < 22:
                        bonus += 10.0
                    cand.at[idx, "w_pos"] += bonus

        cand["w_total"] = cand["w_base"] + cand["w_pos"]

        # Pick top N recipients by weight (this is where backups can jump into top N)
        cand = cand.sort_values("w_total", ascending=False).head(MIN_REDIS_TOP_N).copy()

        total_w = float(cand["w_total"].sum())
        if total_w <= 0:
            continue

        # Allocate minute increases
        minute_increases = {}
        for _, r in cand.iterrows():
            share = float(r["w_total"]) / total_w
            inc = missing_minutes * share
            inc = min(inc, MAX_MIN_INCREASE)

            base_m = safe_float(r["Minutes"], 0.0)
            new_m = min(base_m + inc, MAX_MINUTES_PLAYER)

            minute_increases[r["Name_clean"]] = new_m - base_m

        # Apply minute changes and recompute stats from per-minute baselines
        for name_clean_key, inc in minute_increases.items():
            if inc <= 0:
                continue
            idxs = df.index[(df["Team"] == team) & (df["Name_clean"] == name_clean_key) & ok_mask]
            if len(idxs) == 0:
                continue
            i = idxs[0]

            old_m = safe_float(df.at[i, "Minutes"], 0.0)
            new_m = min(old_m + inc, MAX_MINUTES_PLAYER)
            df.at[i, "Minutes"] = new_m

            # Recompute each stat from baseline per-minute rate * new minutes
            for c in STAT_COLS:
                pm = safe_float(df.at[i, f"PM_{c}"], 0.0)
                df.at[i, c] = round(pm * new_m, 2)

            df.at[i, "Notes"] = (str(df.at[i, "Notes"]) + f" MIN+{inc:.1f}").strip()

        # Apply offense per-minute multiplier to top recipients (optional)
        if missing_off > 0:
            # Build recipient offense baseline after minute change but using baseline per-minute rates
            team_ok_after = df[(df["Team"] == team) & ok_mask].copy()

            # Choose recipients: prioritize high w_total guys (same cand) and ball-handlers (AST pm)
            # We'll merge w_total back in:
            w_map = dict(zip(cand["Name_clean"], cand["w_total"]))
            team_ok_after["w_total"] = team_ok_after["Name_clean"].map(w_map).fillna(0.0)
            team_ok_after["ast_pm"] = team_ok_after["PM_AST"].fillna(0.0)
            team_ok_after["pts_pm"] = team_ok_after["PM_PTS"].fillna(0.0)

            team_ok_after["off_rank"] = team_ok_after["w_total"] + 40.0 * team_ok_after["ast_pm"] + 10.0 * team_ok_after["pts_pm"]
            recipients = team_ok_after.sort_values("off_rank", ascending=False).head(OFF_BUMP_TOP_N)

            recipient_off = float((recipients["PTS"].fillna(0) + recipients["AST"].fillna(0) + recipients["FG3M"].fillna(0)).sum())
            if recipient_off > 0:
                raw_mult = 1.0 + OFF_BUMP_STRENGTH * (missing_off / recipient_off)
                mult = min(raw_mult, OFF_BUMP_CAP)

                for idx in recipients.index:
                    # Multiply offense stats only; keep defense/reb/tov as-is
                    df.at[idx, "PTS"] = round(safe_float(df.at[idx, "PTS"], 0.0) * mult, 2)
                    df.at[idx, "AST"] = round(safe_float(df.at[idx, "AST"], 0.0) * mult, 2)
                    df.at[idx, "FG3M"] = round(safe_float(df.at[idx, "FG3M"], 0.0) * mult, 2)
                    df.at[idx, "Notes"] = (str(df.at[idx, "Notes"]) + f" OFFx{mult:.2f}").strip()

        # Recompute DK_FP and Value for the whole team OK players (only those affected is fine, but this is cheap)
        team_ok_final = df.index[(df["Team"] == team) & ok_mask]
        for i in team_ok_final:
            stats = {
                "PTS": safe_float(df.at[i, "PTS"], 0.0),
                "REB": safe_float(df.at[i, "REB"], 0.0),
                "AST": safe_float(df.at[i, "AST"], 0.0),
                "STL": safe_float(df.at[i, "STL"], 0.0),
                "BLK": safe_float(df.at[i, "BLK"], 0.0),
                "TOV": safe_float(df.at[i, "TOV"], 0.0),
                "FG3M": safe_float(df.at[i, "FG3M"], 0.0),
            }
            fp = dk_fp(stats)
            df.at[i, "DK_FP"] = fp
            sal = safe_float(df.at[i, "Salary"], 0.0)
            df.at[i, "Value"] = (fp / (sal / 1000.0)) if sal > 0 else np.nan

    return df

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.subheader("Upload DK CSV")
    dk_file = st.file_uploader("DraftKings NBA CSV", type="csv")

    st.divider()
    st.subheader("Projection settings")
    last_n = st.number_input(
        "Recent games window (N)",
        1, 30,
        st.session_state.get("last_n", DEFAULT_LAST_N)
    )
    st.session_state["last_n"] = int(last_n)

    st.divider()
    st.subheader("Late scratches")
    out_text = st.text_area(
        "Players OUT (one per line or comma-separated)",
        value=st.session_state.get("out_text", ""),
        height=130
    )
    st.session_state["out_text"] = out_text

    exclude_out = st.checkbox("Exclude OUT players from pool", value=st.session_state.get("exclude_out", True))
    st.session_state["exclude_out"] = bool(exclude_out)

    st.divider()
    st.subheader("Optimizer constraints")
    max_team = st.number_input("Max players per team (0 = no limit)", 0, 8, st.session_state.get("max_team", 4))
    st.session_state["max_team"] = int(max_team)

    st.divider()
    st.subheader("Bumps")
    enable_bumps_ui = st.checkbox("Enable teammate bumps when players are OUT", value=st.session_state.get("enable_bumps_ui", True))
    st.session_state["enable_bumps_ui"] = bool(enable_bumps_ui)

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

col1, col2 = st.columns([1, 2])
with col1:
    build_btn = st.button("Build/Refresh Projections")
with col2:
    st.caption("Cloud note: NBA Stats can time out. Use retries + ‘Retry ERR Players Only’ for stragglers.")

if "proj_df" not in st.session_state:
    st.session_state["proj_df"] = None

def build_projections(slate_df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT change:
    - We still TRY to project OUT players so we know their minutes/usage for teammate bumps.
    - They are excluded from the optimizer pool later if exclude_out is enabled.
    """
    rows = []
    total = len(slate_df)
    prog = st.progress(0, text="Projecting players...")

    for i, r in enumerate(slate_df.itertuples(index=False), start=1):
        prog.progress(int(i / max(total, 1) * 100), text=f"Running gamelog({r.Name}) ({i}/{total})...")

        try:
            mins, fp, stats = project_player(r.Name, int(st.session_state["last_n"]))
            sal = float(r.Salary) if pd.notna(r.Salary) else np.nan
            value = (fp / (sal / 1000.0)) if (pd.notna(sal) and sal > 0) else np.nan

            status = "OUT" if r.IsOutManual else "OK"

            rows.append({
                "Name": r.Name,
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
            # ERR rows are not faked as zeros
            rows.append({
                "Name": r.Name,
                "Team": r.Team,
                "Positions": r.Positions,
                "Salary": float(r.Salary) if pd.notna(r.Salary) else np.nan,
                "Minutes": np.nan,
                "PTS": np.nan, "REB": np.nan, "AST": np.nan, "FG3M": np.nan, "STL": np.nan, "BLK": np.nan, "TOV": np.nan,
                "DK_FP": np.nan,
                "Value": np.nan,
                "Status": f"ERR: {str(e)[:90]}",
            })

        time.sleep(BUILD_THROTTLE_SECONDS)

    prog.empty()
    return pd.DataFrame(rows)

def retry_err_players(proj_df: pd.DataFrame, slate_df: pd.DataFrame) -> pd.DataFrame:
    err_mask = proj_df["Status"].astype(str).str.startswith("ERR")
    err_names = proj_df.loc[err_mask, "Name"].tolist()
    if not err_names:
        return proj_df

    prog = st.progress(0, text="Retrying ERR players only...")
    proj_df = proj_df.copy()

    # For correct OUT/OK status after retry
    out_clean = parse_out_list(st.session_state.get("out_text", ""))
    out_name_clean_map = {clean_name(n): True for n in out_clean}

    for i, nm in enumerate(err_names, start=1):
        prog.progress(int(i / max(len(err_names), 1) * 100), text=f"Retrying {nm} ({i}/{len(err_names)})...")

        try:
            mins, fp, stats = project_player(nm, int(st.session_state["last_n"]))
            sal = float(proj_df.loc[proj_df["Name"] == nm, "Salary"].iloc[0])
            value = (fp / (sal / 1000.0)) if sal and sal > 0 else np.nan

            status = "OUT" if out_name_clean_map.get(clean_name(nm), False) else "OK"

            proj_df.loc[proj_df["Name"] == nm, ["Minutes","PTS","REB","AST","FG3M","STL","BLK","TOV","DK_FP","Value","Status"]] = [
                mins, stats["PTS"], stats["REB"], stats["AST"], stats["FG3M"], stats["STL"], stats["BLK"], stats["TOV"],
                fp, value, status
            ]
        except Exception as e:
            proj_df.loc[proj_df["Name"] == nm, "Status"] = f"ERR: {str(e)[:90]}"

        time.sleep(RETRY_THROTTLE_SECONDS)

    prog.empty()
    return proj_df

# Build on first load or when clicked
if build_btn or st.session_state["proj_df"] is None:
    st.session_state["proj_df"] = build_projections(slate)

# Apply bumps after projections build
if ENABLE_OUT_BUMPS and st.session_state.get("enable_bumps_ui", True):
    st.session_state["proj_df"] = apply_out_bumps(st.session_state["proj_df"], slate, manual_out)

proj_df = st.session_state["proj_df"].copy()
proj_df = proj_df.sort_values("DK_FP", ascending=False)

# Display-friendly 3PM column
display_df = proj_df.rename(columns={"FG3M": "3PM"})

st.dataframe(
    display_df[[
        "Name","Team","Positions","Salary",
        "Minutes","PTS","REB","AST","3PM","STL","BLK","TOV",
        "DK_FP","Value","Status",
        "Notes" if "Notes" in display_df.columns else "Status"
    ]],
    use_container_width=True,
    hide_index=True
)

# Retry ERR players only
retry_btn = st.button("Retry ERR Players Only")
if retry_btn:
    st.session_state["proj_df"] = retry_err_players(st.session_state["proj_df"], slate)
    # re-apply bumps after retry
    if ENABLE_OUT_BUMPS and st.session_state.get("enable_bumps_ui", True):
        manual_out = parse_out_list(st.session_state.get("out_text", ""))
        st.session_state["proj_df"] = apply_out_bumps(st.session_state["proj_df"], slate, manual_out)

    st.success("Retried ERR players (and re-applied bumps).")
    proj_df = st.session_state["proj_df"].copy().sort_values("DK_FP", ascending=False)

# ==========================
# Build optimizer pool
# ==========================
pool = st.session_state["proj_df"].copy()

# Only rows with real projections
pool = pool.dropna(subset=["Salary", "DK_FP"])
pool = pool[pool["Salary"] > 0]

# Keep only OK (and optionally OUT included)
pool = pool[pool["Status"].isin(["OK"])]

# Exclude OUT if option enabled (OUT never in pool)
if st.session_state["exclude_out"]:
    pool = pool[pool["Status"] != "OUT"]

# Locks / Excludes
st.subheader("Optimizer Controls")
pool_names = pool["Name"].tolist()
lock_players = st.multiselect("Lock players (force into lineup)", options=pool_names, default=[])
exclude_players = st.multiselect("Exclude players", options=pool_names, default=[])

pool = pool[~pool["Name"].isin(exclude_players)].copy()

# ==========================
# Optimizer (fills all 8 slots)
# ==========================
def optimize(pool_df: pd.DataFrame) -> pd.DataFrame:
    prob = LpProblem("DK_NBA_CLASSIC", LpMaximize)

    rows = pool_df.to_dict("records")
    n = len(rows)
    if n == 0:
        raise ValueError("Optimizer pool is empty. (Check OUT / ERR / exclude settings.)")

    x = {}
    for i in range(n):
        for slot in DK_SLOTS:
            if slot_eligible(rows[i]["Positions"], slot):
                x[(i, slot)] = LpVariable(f"x_{i}_{slot}", 0, 1, LpBinary)

    prob += lpSum(rows[i]["DK_FP"] * x[(i, slot)] for (i, slot) in x)

    for slot in DK_SLOTS:
        prob += lpSum(x[(i, slot)] for i in range(n) if (i, slot) in x) == 1, f"fill_{slot}"

    for i in range(n):
        prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) <= 1, f"use_once_{i}"

    prob += lpSum(rows[i]["Salary"] * x[(i, slot)] for (i, slot) in x) <= DK_SALARY_CAP, "salary_cap"

    max_team = int(st.session_state.get("max_team", 0))
    if max_team and max_team > 0:
        teams_unique = sorted(set(r["Team"] for r in rows))
        for t in teams_unique:
            idxs = [i for i in range(n) if rows[i]["Team"] == t]
            prob += lpSum(x[(i, slot)] for i in idxs for slot in DK_SLOTS if (i, slot) in x) <= max_team, f"max_{t}"

    for lp in lock_players:
        idxs = [i for i in range(n) if rows[i]["Name"] == lp]
        if idxs:
            i = idxs[0]
            prob += lpSum(x[(i, slot)] for slot in DK_SLOTS if (i, slot) in x) == 1, f"lock_{i}"

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup_rows = []
    for slot in DK_SLOTS:
        chosen_i = None
        for i in range(n):
            if (i, slot) in x and x[(i, slot)].value() == 1:
                chosen_i = i
                break
        if chosen_i is None:
            raise ValueError("No feasible lineup found. Loosen locks/max-team/exclusions.")

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
            "Notes": r.get("Notes", "")
        })

    return pd.DataFrame(lineup_rows)

st.subheader("Optimizer")
run_opt = st.button("Optimize Lineup (Classic)")

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
