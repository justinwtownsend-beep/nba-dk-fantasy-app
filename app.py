import time
import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    commonteamroster
)

# ==========================
# CONSTANTS / DEFAULTS
# ==========================
SEASON = "2025-26"      # fixed
SITE = "DK"            # fixed
DEFAULT_LAST_N = 10

# How many recent teammate games to consider when computing "OUT-player OFF" rates
OFF_WINDOW_GAMES = 30

# Caps to prevent tiny-sample weirdness from blowing up projections
BUMP_CAP_LOW = 0.70
BUMP_CAP_HIGH = 1.40

st.set_page_config(page_title="NBA DK Fantasy Projector", layout="wide")
st.title("NBA DraftKings Fantasy Projector (Whole Team + Manual OUT + On/Off Bumps)")

# ==========================
# DK SCORING
# ==========================
def dk_fantasy_points(stats: dict) -> float:
    pts = float(stats.get("PTS", 0))
    reb = float(stats.get("REB", 0))
    ast = float(stats.get("AST", 0))
    stl = float(stats.get("STL", 0))
    blk = float(stats.get("BLK", 0))
    tov = float(stats.get("TOV", 0))
    tpm = float(stats.get("FG3M", 0))

    fp = (
        pts * 1.0
        + reb * 1.25
        + ast * 1.5
        + stl * 2.0
        + blk * 2.0
        - tov * 0.5
        + tpm * 0.5
    )

    # DK bonus: double-double / triple-double
    cats_10 = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats_10 >= 2:
        fp += 1.5
    if cats_10 >= 3:
        fp += 3.0

    return round(fp, 2)

# ==========================
# HELPERS
# ==========================
def clean_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def parse_out_list(text: str):
    """
    Accepts one name per line OR comma-separated.
    Returns a set of normalized names.
    """
    if not text:
        return set()
    text = text.replace(",", "\n")
    parts = [p.strip() for p in text.split("\n")]
    parts = [p for p in parts if p]
    return {clean_name(p) for p in parts}

def parse_minutes(min_str):
    if pd.isna(min_str):
        return np.nan
    s = str(min_str)
    if ":" not in s:
        return float(s)
    m, sec = s.split(":")
    return float(m) + float(sec) / 60.0

def get_player_id(player_name: str) -> int:
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"No player found for '{player_name}'. Use full name (e.g., 'Luka Doncic').")
    exact = [m for m in matches if m["full_name"].lower() == player_name.lower()]
    chosen = exact[0] if exact else matches[0]
    return int(chosen["id"])

def get_team_obj_by_name(team_name: str) -> dict:
    tlist = teams.get_teams()
    name = team_name.strip().lower()

    alias_map = {
        "new orleans": "pelicans", "nop": "pelicans",
        "la clippers": "clippers", "los angeles clippers": "clippers",
        "la lakers": "lakers", "los angeles lakers": "lakers",
        "gsw": "warriors", "golden state": "warriors",
        "okc": "thunder", "oklahoma city": "thunder",
        "ny": "knicks", "new york": "knicks",
        "bk": "nets", "brooklyn": "nets",
        "phx": "suns", "phoenix": "suns",
        "sa": "spurs", "san antonio": "spurs",
    }
    name = alias_map.get(name, name)

    for t in tlist:
        if t["full_name"].lower() == name:
            return t
    for t in tlist:
        if t["nickname"].lower() == name:
            return t
    for t in tlist:
        if name in t["full_name"].lower():
            return t

    raise ValueError(f"Team not found for '{team_name}'. Try 'Mavericks' or 'Dallas Mavericks'.")

# ==========================
# INJURY MULTIPLIERS (manual override)
# ==========================
def injury_multipliers(status: str):
    s = status.strip().lower()
    if s in ["healthy", "h", ""]:
        return 1.0, 1.0
    if s in ["questionable", "q"]:
        return 0.92, 0.97
    if s in ["doubtful", "d"]:
        return 0.80, 0.92
    if s in ["limited", "minutes limit", "ml"]:
        return 0.85, 0.95
    if s in ["out", "o"]:
        return 0.0, 0.0
    return 1.0, 1.0

# ==========================
# NBA: GAME LOGS (cached)
# ==========================
@st.cache_data(ttl=3600)
def get_player_gamelog_df(player_id: int, season: str) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    # Defensive: ensure key columns exist
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def compute_per_minute_rates(gl: pd.DataFrame, stat_cols: list) -> dict:
    if gl is None or gl.empty:
        return {c: 0.0 for c in stat_cols}

    work = gl.copy()
    work["MIN_float"] = work["MIN"].apply(parse_minutes)
    work = work.dropna(subset=["MIN_float"])
    if work.empty or work["MIN_float"].sum() <= 0:
        return {c: 0.0 for c in stat_cols}

    for c in stat_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

    total_min = float(work["MIN_float"].sum())
    return {c: float(work[c].sum()) / total_min for c in stat_cols}

@st.cache_data(ttl=3600)
def get_recent_player_baseline(player_id: int, season: str, last_n_games: int):
    gl = get_player_gamelog_df(player_id, season)
    if gl.empty:
        raise ValueError("No game logs returned. If SEASON not supported yet, switch to 2024-25.")

    gl = gl.head(last_n_games).copy()
    gl["MIN_float"] = gl["MIN"].apply(parse_minutes)
    gl = gl.dropna(subset=["MIN_float"])
    if gl.empty:
        raise ValueError("No valid minutes in recent game logs.")

    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    for c in stat_cols:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0)

    total_min = gl["MIN_float"].sum()
    rates = {c: (gl[c].sum() / total_min) for c in stat_cols}
    proj_min = float(gl["MIN_float"].mean())
    return rates, proj_min

# ==========================
# NBA: TEAM CONTEXT (cached)
# ==========================
@st.cache_data(ttl=3600)
def get_team_context(season: str):
    adv_df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]

    pace_col = next((c for c in ["PACE", "POSS", "POSS_PG"] if c in adv_df.columns), None)
    if pace_col:
        adv_df[pace_col] = pd.to_numeric(adv_df[pace_col], errors="coerce")

    opp_df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent"
    ).get_data_frames()[0]

    for c in ["OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M", "OPP_STL", "OPP_BLK", "OPP_TOV"]:
        if c in opp_df.columns:
            opp_df[c] = pd.to_numeric(opp_df[c], errors="coerce")

    return adv_df, opp_df, pace_col

def get_opponent_stat_multipliers(opp_df: pd.DataFrame, opp_team_id: int):
    cols = ["OPP_PTS", "OPP_REB", "OPP_AST", "OPP_FG3M", "OPP_STL", "OPP_BLK", "OPP_TOV"]
    league_avgs = {c: float(opp_df[c].mean()) if c in opp_df.columns else None for c in cols}
    row = opp_df.loc[opp_df["TEAM_ID"] == opp_team_id]
    if row.empty:
        return {c: 1.0 for c in cols}

    mults = {}
    for c in cols:
        if c in opp_df.columns and league_avgs[c] and league_avgs[c] != 0:
            mults[c] = float(row[c].iloc[0]) / league_avgs[c]
        else:
            mults[c] = 1.0
    return mults

# ==========================
# NBA: ROSTER (cached)
# ==========================
@st.cache_data(ttl=3600)
def nba_get_roster_names(nba_team_id: int, season: str):
    df = commonteamroster.CommonTeamRoster(team_id=nba_team_id, season=season).get_data_frames()[0]
    return df["PLAYER"].dropna().tolist()

# ==========================
# ON/OFF BUMPS (manual OUT list drives this)
# ==========================
def capped_ratio(num: float, den: float) -> float:
    if den is None or den == 0:
        return 1.0
    r = num / den
    return float(np.clip(r, BUMP_CAP_LOW, BUMP_CAP_HIGH))

@st.cache_data(ttl=1800)
def out_player_game_ids(player_id: int, season: str) -> set:
    """
    Returns set of GAME_ID where the OUT player played this season.
    """
    gl = get_player_gamelog_df(player_id, season)
    if gl.empty or "Game_ID" not in gl.columns:
        # nba_api uses "Game_ID" or "GAME_ID" depending on endpoint version
        pass

    col = "Game_ID" if "Game_ID" in gl.columns else ("GAME_ID" if "GAME_ID" in gl.columns else None)
    if col is None:
        return set()

    return set(gl[col].astype(str).tolist())

@st.cache_data(ttl=1800)
def teammate_off_multipliers(teammate_id: int, season: str, out_games_set: set) -> dict:
    """
    Compute per-stat multipliers for a teammate when OUT-player is OFF the floor,
    approximated as: games where OUT-player did not play.
    Returns dict multipliers for all stat cols.
    """
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    gl = get_player_gamelog_df(teammate_id, season)
    if gl.empty:
        return {c: 1.0 for c in stat_cols}

    col = "Game_ID" if "Game_ID" in gl.columns else ("GAME_ID" if "GAME_ID" in gl.columns else None)
    if col is None:
        return {c: 1.0 for c in stat_cols}

    # Focus on a recent window so this reflects current rotations
    recent = gl.head(OFF_WINDOW_GAMES).copy()
    recent[col] = recent[col].astype(str)

    off = recent[~recent[col].isin(out_games_set)].copy()

    # If almost no OFF games, don’t bump (avoid noise)
    if off.shape[0] < 3:
        return {c: 1.0 for c in stat_cols}

    base_rates = compute_per_minute_rates(recent, stat_cols)
    off_rates = compute_per_minute_rates(off, stat_cols)

    mults = {}
    for c in stat_cols:
        mults[c] = capped_ratio(off_rates.get(c, 0.0), base_rates.get(c, 0.0))
    return mults

def combine_multipliers(mults_list: list) -> dict:
    """
    Combine multiple OUT-player effects by multiplying, then clipping again.
    """
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    out = {c: 1.0 for c in stat_cols}

    for m in mults_list:
        for c in stat_cols:
            out[c] *= float(m.get(c, 1.0))

    # Clip combined effect too
    for c in stat_cols:
        out[c] = float(np.clip(out[c], BUMP_CAP_LOW, BUMP_CAP_HIGH))

    return out

# ==========================
# PROJECTION: SINGLE PLAYER (core)
# ==========================
def project_player(
    player_name: str,
    player_side: str,
    injury_status: str,
    last_n_games: int,
    away_team_obj: dict,
    home_team_obj: dict,
    bump_mults: dict | None = None
):
    adv_df, opp_df, pace_col = get_team_context(SEASON)

    player_side = player_side.strip().upper()
    if player_side not in ["AWAY", "HOME"]:
        player_side = "AWAY"

    opp_team_obj = home_team_obj if player_side == "AWAY" else away_team_obj
    opp_team_id = int(opp_team_obj["id"])

    # pace multiplier
    pace_mult = 1.0
    if pace_col and pace_col in adv_df.columns:
        league_pace = float(adv_df[pace_col].mean())
        row = adv_df.loc[adv_df["TEAM_ID"] == opp_team_id]
        if not row.empty and league_pace:
            opp_pace = float(row[pace_col].iloc[0])
            pace_mult = opp_pace / league_pace

    opp_mults = get_opponent_stat_multipliers(opp_df, opp_team_id)

    pid = get_player_id(player_name)
    time.sleep(0.12)  # gentle throttle

    rates, base_min = get_recent_player_baseline(pid, SEASON, last_n_games)

    # Apply ON/OFF bump multipliers (FULL stat bump)
    if bump_mults:
        for k in rates.keys():
            rates[k] = float(rates[k]) * float(bump_mults.get(k, 1.0))

    # Injury multipliers (still apply if you manually override in single-player section)
    min_mult, usage_mult = injury_multipliers(injury_status)
    proj_min = base_min * min_mult

    proj = {k: rates[k] * proj_min for k in rates.keys()}

    # pace
    for k in proj.keys():
        proj[k] *= pace_mult

    # opponent multipliers by category
    proj["PTS"]  *= opp_mults["OPP_PTS"] * usage_mult
    proj["REB"]  *= opp_mults["OPP_REB"]
    proj["AST"]  *= opp_mults["OPP_AST"] * usage_mult
    proj["FG3M"] *= opp_mults["OPP_FG3M"] * usage_mult
    proj["STL"]  *= opp_mults["OPP_STL"]
    proj["BLK"]  *= opp_mults["OPP_BLK"]
    proj["TOV"]  *= opp_mults["OPP_TOV"]

    proj_stats = {k: round(float(max(0, v)), 2) for k, v in proj.items()}
    fp = dk_fantasy_points(proj_stats)

    return {
        "player": player_name,
        "player_side": player_side,
        "opponent": opp_team_obj["full_name"],
        "minutes_base": round(float(base_min), 2),
        "minutes_proj": round(float(proj_min), 2),
        "pace_mult": round(float(pace_mult), 3),
        "proj_stats": proj_stats,
        "dk_fp": fp,
    }

# ==========================
# SIDEBAR: MATCHUP + SETTINGS + MANUAL OUT LIST
# ==========================
with st.sidebar:
    st.subheader("Matchup (locks the game)")
    away_team_in = st.text_input("Away Team", value=st.session_state.get("away_team", ""))
    home_team_in = st.text_input("Home Team", value=st.session_state.get("home_team", ""))

    if st.button("Save matchup"):
        st.session_state["away_team"] = away_team_in
        st.session_state["home_team"] = home_team_in
        st.success(f"Saved: {away_team_in} @ {home_team_in}")

    st.divider()
    st.subheader("Projection settings")
    last_n = st.number_input("Recent games window (N)", min_value=1, max_value=30, value=st.session_state.get("last_n", DEFAULT_LAST_N))
    st.session_state["last_n"] = int(last_n)

    exclude_out = st.checkbox("Exclude OUT players", value=st.session_state.get("exclude_out", True))
    st.session_state["exclude_out"] = exclude_out

    out_as_zero = st.checkbox("If not excluded, set OUT FP=0", value=st.session_state.get("out_as_zero", True))
    st.session_state["out_as_zero"] = out_as_zero

    st.divider()
    st.subheader("Manual OUT list")
    st.caption("One per line or comma-separated (exact-ish names). Example:\n\nLuka Doncic\nDereck Lively II")
    out_text = st.text_area("Players OUT", value=st.session_state.get("out_text", ""), height=160)
    st.session_state["out_text"] = out_text

    st.divider()
    use_onoff = st.checkbox("Apply On/Off bumps for OUT players (FULL stats)", value=st.session_state.get("use_onoff", True))
    st.session_state["use_onoff"] = use_onoff
    st.caption(f"Bump caps: {BUMP_CAP_LOW:.2f} to {BUMP_CAP_HIGH:.2f} | OFF window: {OFF_WINDOW_GAMES} games")

    st.caption(f"Season fixed: {SEASON} | Site fixed: {SITE}")

away_team = st.session_state.get("away_team", "").strip()
home_team = st.session_state.get("home_team", "").strip()

if not away_team or not home_team:
    st.info("Set Away/Home in the sidebar, click **Save matchup**, then project.")
    st.stop()

away_team_obj = get_team_obj_by_name(away_team)
home_team_obj = get_team_obj_by_name(home_team)

manual_out_set = parse_out_list(st.session_state.get("out_text", ""))

st.write(f"**Game:** {away_team_obj['full_name']} @ {home_team_obj['full_name']}  |  **Season:** {SEASON}  |  **Scoring:** DK")

# ==========================
# WHOLE GAME PROJECTION
# ==========================
st.subheader("Whole Game Projection (both rosters)")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    do_whole = st.button("Project both teams")
with c2:
    max_players = st.number_input("Max players per team (0 = all)", min_value=0, max_value=20, value=0)
with c3:
    st.caption("Tip: If slow, set Max players per team to 12–14 or lower N.")

def is_manual_out(name: str) -> bool:
    return clean_name(name) in manual_out_set

if do_whole:
    away_roster = nba_get_roster_names(int(away_team_obj["id"]), SEASON)
    home_roster = nba_get_roster_names(int(home_team_obj["id"]), SEASON)

    if max_players and max_players > 0:
        away_roster = away_roster[:max_players]
        home_roster = home_roster[:max_players]

    # Identify which OUT players belong to which team roster (so bumps only hit teammates)
    away_out_names = [p for p in away_roster if is_manual_out(p)]
    home_out_names = [p for p in home_roster if is_manual_out(p)]

    # Precompute game-id sets for OUT players (cached)
    away_out_gamesets = []
    home_out_gamesets = []

    if st.session_state["use_onoff"]:
        for nm in away_out_names:
            try:
                away_out_gamesets.append(out_player_game_ids(get_player_id(nm), SEASON))
            except Exception:
                pass
        for nm in home_out_names:
            try:
                home_out_gamesets.append(out_player_game_ids(get_player_id(nm), SEASON))
            except Exception:
                pass

    rows = []
    total = len(away_roster) + len(home_roster)
    prog = st.progress(0, text="Projecting players...")

    def build_bump_for_teammate(teammate_name: str, team_out_gamesets: list) -> dict | None:
        """
        If on/off is enabled and the team has OUT players, compute combined multipliers for this teammate.
        """
        if not st.session_state["use_onoff"]:
            return None
        if not team_out_gamesets:
            return None

        try:
            tid = get_player_id(teammate_name)
        except Exception:
            return None

        mults_list = []
        for gs in team_out_gamesets:
            try:
                m = teammate_off_multipliers(tid, SEASON, gs)
                mults_list.append(m)
            except Exception:
                continue

        if not mults_list:
            return None

        return combine_multipliers(mults_list)

    i = 0
    for side, roster, team_out_gamesets in [
        ("AWAY", away_roster, away_out_gamesets),
        ("HOME", home_roster, home_out_gamesets),
    ]:
        for p in roster:
            i += 1
            prog.progress(int((i / max(total, 1)) * 100), text=f"Projecting {p} ({i}/{total})...")

            # Manual OUT handling
            if is_manual_out(p):
                if st.session_state["exclude_out"]:
                    continue
                if st.session_state["out_as_zero"]:
                    rows.append({
                        "Player": p,
                        "Side": side,
                        "Status": "OUT (manual)",
                        "BumpApplied": "No",
                        "Min": 0.0,
                        "DK_FP": 0.0,
                        "PTS": 0.0, "REB": 0.0, "AST": 0.0, "3PM": 0.0, "STL": 0.0, "BLK": 0.0, "TOV": 0.0,
                        "OpponentUsed": home_team_obj["full_name"] if side == "AWAY" else away_team_obj["full_name"],
                    })
                    continue

            # Compute bump multipliers for this player (based on teammates being OUT)
            bump = build_bump_for_teammate(p, team_out_gamesets)
            bump_flag = "Yes" if bump else "No"

            try:
                res = project_player(
                    player_name=p,
                    player_side=side,
                    injury_status="healthy",
                    last_n_games=int(st.session_state["last_n"]),
                    away_team_obj=away_team_obj,
                    home_team_obj=home_team_obj,
                    bump_mults=bump
                )
                ps = res["proj_stats"]

                rows.append({
                    "Player": p,
                    "Side": side,
                    "Status": "Healthy",
                    "BumpApplied": bump_flag,
                    "Min": res["minutes_proj"],
                    "DK_FP": res["dk_fp"],
                    "PTS": ps["PTS"], "REB": ps["REB"], "AST": ps["AST"], "3PM": ps["FG3M"],
                    "STL": ps["STL"], "BLK": ps["BLK"], "TOV": ps["TOV"],
                    "OpponentUsed": res["opponent"],
                })

            except Exception:
                rows.append({
                    "Player": p,
                    "Side": side,
                    "Status": "ERROR",
                    "BumpApplied": bump_flag,
                    "Min": np.nan,
                    "DK_FP": np.nan,
                    "PTS": np.nan, "REB": np.nan, "AST": np.nan, "3PM": np.nan, "STL": np.nan, "BLK": np.nan, "TOV": np.nan,
                    "OpponentUsed": home_team_obj["full_name"] if side == "AWAY" else away_team_obj["full_name"],
                })

    prog.empty()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Side", "DK_FP"], ascending=[True, False])

        st.subheader("Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Team Totals (sum of projected DK FP)")
        totals = df.groupby("Side")["DK_FP"].sum().reset_index().rename(columns={"DK_FP": "Team_DK_FP_Sum"})
        st.dataframe(totals, use_container_width=True, hide_index=True)

        if st.session_state["use_onoff"]:
            st.caption("Bumps are computed from teammate per-minute rates in games where OUT-player did not play (recent window), with caps to avoid tiny-sample noise.")
    else:
        st.warning("No players returned.")

# ==========================
# SINGLE PLAYER QUICK CHECK
# ==========================
st.divider()
st.subheader("Single Player Quick Projection")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    single_player = st.text_input("Player full name", value="")
with c2:
    single_side = st.selectbox("Side", ["AWAY", "HOME"], index=0)
with c3:
    single_inj = st.selectbox("Injury (manual override)", ["healthy", "questionable", "doubtful", "out", "limited"], index=0)
with c4:
    run_single = st.button("Project player")

if run_single and single_player.strip():
    res = project_player(
        player_name=single_player.strip(),
        player_side=single_side,
        injury_status=single_inj,
        last_n_games=int(st.session_state["last_n"]),
        away_team_obj=away_team_obj,
        home_team_obj=home_team_obj,
        bump_mults=None
    )
    ps = res["proj_stats"]

    st.write(f"**Opponent used:** {res['opponent']}  |  **Minutes:** {res['minutes_base']} → {res['minutes_proj']}  |  **Pace mult:** {res['pace_mult']}")
    st.code(
        f"PTS {ps['PTS']} | REB {ps['REB']} | AST {ps['AST']} | 3PM {ps['FG3M']} | "
        f"STL {ps['STL']} | BLK {ps['BLK']} | TOV {ps['TOV']}"
    )
    st.metric("Projected DK Fantasy Points", res["dk_fp"])
