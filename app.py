# app.py
import json
import time
import difflib
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests

from nba_api.stats.endpoints import leaguedashplayerstats

# ==========================
# CONFIG
# ==========================
SEASON = "2025-26"
TIMEOUT = 20

# Gist file names (shared across apps)
GIST_OUT = "out.json"
GIST_FINAL = "final.csv"

# Minutes cap
MAX_MINUTES = 34.0
TEAM_TOTAL_MINUTES = 240.0

# ==========================
# PAGE
# ==========================
st.set_page_config(layout="wide")
st.title("DK Projections + Optimizer (Stable Minutes Cap)")

st.markdown(
    f"""
**Season:** {SEASON}  
**Minutes rule:** cap everyone at **{MAX_MINUTES}** and **redistribute** extra minutes to teammates (so no more 40+ min projections).
"""
)

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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_num(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

# DK FP calc
def dk_fp(pts, reb, ast, stl, blk, tov, fg3m):
    fp = pts + 1.25*reb + 1.5*ast + 2.0*stl + 2.0*blk - 0.5*tov + 0.5*fg3m
    cats = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
    if cats >= 2:
        fp += 1.5
    if cats >= 3:
        fp += 3.0
    return fp

# ==========================
# GIST IO
# ==========================
def gh_headers(token: str):
    return {"Authorization": f"token {token}"}

def gist_get(gist_id: str, token: str):
    r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=gh_headers(token), timeout=25)
    r.raise_for_status()
    return r.json()

def gist_read(gist_id: str, token: str, filename: str):
    g = gist_get(gist_id, token)
    if filename not in g.get("files", {}):
        return None
    f = g["files"][filename]
    if not f.get("truncated"):
        return f.get("content")
    rr = requests.get(f["raw_url"], timeout=25)
    rr.raise_for_status()
    return rr.text

def gist_write(gist_id: str, token: str, files: dict):
    payload = {"files": {k: {"content": v} for k, v in files.items()}}
    r = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=gh_headers(token),
        json=payload,
        timeout=25
    )
    r.raise_for_status()

# ==========================
# NBA: Season-to-date per-game
# ==========================
@st.cache_data(ttl=1800)
def nba_league_dash():
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed="PerGame",
        timeout=TIMEOUT
    ).get_data_frames()[0]

    # keep what we need
    keep = ["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    df = df[keep].copy()
    df.columns = ["PLAYER_ID", "NBA_Name", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    df["NBA_Name_clean"] = df["NBA_Name"].apply(clean_name)
    df["NBA_Name_stripped"] = df["NBA_Name"].apply(strip_suffix)
    df["NBA_Last"] = df["NBA_Name_clean"].apply(lambda x: x.split()[-1] if isinstance(x, str) and x.split() else "")
    return df

def match_nba_row(name: str, nba_df: pd.DataFrame):
    cn = clean_name(name)
    sn = strip_suffix(name)

    exact = nba_df[nba_df["NBA_Name_clean"] == cn]
    if not exact.empty:
        return exact.iloc[0]

    exact2 = nba_df[nba_df["NBA_Name_stripped"] == sn]
    if not exact2.empty:
        return exact2.iloc[0]

    # last-name candidates
    parts = sn.split()
    if parts:
        last = parts[-1]
        cand = nba_df[nba_df["NBA_Last"] == last]
        if not cand.empty:
            best_row, best_score = None, 0.0
            for _, row in cand.iterrows():
                score = difflib.SequenceMatcher(None, sn, row["NBA_Name_stripped"]).ratio()
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_row is not None and best_score >= 0.75:
                return best_row

    # close match fallback
    hit = difflib.get_close_matches(cn, nba_df["NBA_Name_clean"].tolist(), n=1, cutoff=0.90)
    if hit:
        return nba_df[nba_df["NBA_Name_clean"] == hit[0]].iloc[0]

    return None

# ==========================
# MINUTES: Cap + Redistribute
# ==========================
def cap_and_redistribute_minutes(
    df: pd.DataFrame,
    max_min: float = 34.0,
    team_total_minutes: float = 240.0,
    team_col: str = "Team",
    min_col: str = "Minutes",
    out_col: str = "IsOut"
) -> pd.DataFrame:
    """
    Caps minutes at max_min for active players, then redistributes removed minutes
    to other active teammates under the cap.

    Also optionally nudges team totals toward team_total_minutes.
    """
    df = df.copy()
    df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0.0)
    df[out_col] = df[out_col].fillna(False).astype(bool)

    # process each team
    for team, sub_idx in df.groupby(team_col).groups.items():
        idx = list(sub_idx)
        active_idx = [i for i in idx if not bool(df.loc[i, out_col])]

        if not active_idx:
            continue

        # cap
        before = df.loc[active_idx, min_col].copy()
        df.loc[active_idx, min_col] = df.loc[active_idx, min_col].clip(lower=0.0, upper=max_min)
        after = df.loc[active_idx, min_col]
        removed = float((before - after).clip(lower=0.0).sum())

        # redistribute removed to under-cap actives
        if removed > 1e-6:
            remaining = removed
            for _ in range(12):
                if remaining <= 1e-6:
                    break
                receivers = [i for i in active_idx if float(df.loc[i, min_col]) < max_min - 1e-6]
                if not receivers:
                    break

                weights = df.loc[receivers, min_col].clip(lower=0.0)
                if float(weights.sum()) <= 1e-9:
                    weights = pd.Series(1.0, index=receivers)
                weights = weights / float(weights.sum())

                add = weights * remaining
                new_vals = (df.loc[receivers, min_col] + add).clip(upper=max_min)
                actual_add = float((new_vals - df.loc[receivers, min_col]).sum())
                df.loc[receivers, min_col] = new_vals
                remaining -= actual_add

        # normalize toward team_total_minutes (optional guardrail)
        team_active_min = float(df.loc[active_idx, min_col].sum())
        diff = team_total_minutes - team_active_min

        # if way off, fix gently
        if abs(diff) > 3.0:
            if diff > 0:
                # need to add minutes to under-cap actives
                remaining = diff
                for _ in range(12):
                    if remaining <= 1e-6:
                        break
                    receivers = [i for i in active_idx if float(df.loc[i, min_col]) < max_min - 1e-6]
                    if not receivers:
                        break
                    weights = df.loc[receivers, min_col].clip(lower=0.0)
                    if float(weights.sum()) <= 1e-9:
                        weights = pd.Series(1.0, index=receivers)
                    weights = weights / float(weights.sum())
                    add = weights * remaining
                    new_vals = (df.loc[receivers, min_col] + add).clip(upper=max_min)
                    actual_add = float((new_vals - df.loc[receivers, min_col]).sum())
                    df.loc[receivers, min_col] = new_vals
                    remaining -= actual_add
            else:
                # need to remove minutes from highest-minute actives
                take = -diff
                donors = sorted(active_idx, key=lambda i: float(df.loc[i, min_col]), reverse=True)
                for i2 in donors:
                    if take <= 1e-6:
                        break
                    can_take = max(0.0, float(df.loc[i2, min_col]))
                    d = min(can_take, take)
                    df.loc[i2, min_col] = float(df.loc[i2, min_col]) - d
                    take -= d

    return df

# ==========================
# BUILD PROJECTIONS
# ==========================
def build_projections(dk: pd.DataFrame, out_flags: dict, nba_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes DK slate rows and OUT flags, then:
      - baseline minutes from NBA season avg minutes
      - injury bumps: redistribute minutes of OUT players to teammates
      - cap minutes at MAX_MINUTES and redistribute again
      - project PTS/REB/AST/STL/BLK/TOV/3PM using per-minute season rates * minutes
    """
    df = dk.copy()

    # OUT flags based on clean name
    df["Name_clean"] = df["Name"].apply(clean_name)
    df["IsOut"] = df["Name_clean"].apply(lambda x: bool(out_flags.get(x, False)))

    # match NBA rows
    nba_rows = []
    for _, r in df.iterrows():
        hit = match_nba_row(r["Name"], nba_df)
        nba_rows.append(hit)

    df["NBA_Matched"] = [h is not None for h in nba_rows]
    df["NBA_MIN"] = [safe_num(h["MIN"], np.nan) if h is not None else np.nan for h in nba_rows]

    # baseline minutes: use NBA season avg, with reasonable fallback
    df["Minutes"] = df["NBA_MIN"].fillna(24.0)

    # -------- Injury minute bump model (simple + stable) --------
    # For each team:
    #   remove OUT minutes
    #   distribute those minutes to active teammates proportional to their baseline minutes
    for team, idx in df.groupby("Team").groups.items():
        idx = list(idx)
        out_idx = [i for i in idx if bool(df.loc[i, "IsOut"])]
        act_idx = [i for i in idx if not bool(df.loc[i, "IsOut"])]

        if not out_idx or not act_idx:
            continue

        removed_min = float(df.loc[out_idx, "Minutes"].sum())
        if removed_min <= 1e-6:
            continue

        weights = df.loc[act_idx, "Minutes"].clip(lower=0.0)
        if float(weights.sum()) <= 1e-9:
            weights = pd.Series(1.0, index=act_idx)
        weights = weights / float(weights.sum())

        df.loc[act_idx, "Minutes"] = df.loc[act_idx, "Minutes"] + weights * removed_min

        # OUT players minutes -> 0 (keep them in background, but they should not project)
        df.loc[out_idx, "Minutes"] = 0.0

    # -------- Cap + redistribute minutes (your requested fix) --------
    df = cap_and_redistribute_minutes(
        df,
        max_min=MAX_MINUTES,
        team_total_minutes=TEAM_TOTAL_MINUTES,
        team_col="Team",
        min_col="Minutes",
        out_col="IsOut"
    )

    # -------- Convert season per-game -> per-minute rates -> project stats --------
    # Build per-minute rate lookup by matched NBA player
    for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]:
        df[stat] = 0.0

    df["Status"] = np.where(df["IsOut"], "OUT", np.where(df["NBA_Matched"], "OK", "ERR"))

    for i, r in df.iterrows():
        if r["IsOut"]:
            continue
        hit = match_nba_row(r["Name"], nba_df)
        if hit is None:
            continue

        season_min = safe_num(hit["MIN"], 0.0)
        if season_min <= 0:
            continue

        mins = safe_num(r["Minutes"], 0.0)
        for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]:
            per_min = safe_num(hit[stat], 0.0) / season_min
            df.loc[i, stat] = per_min * mins

    # DK FP + value
    df["DK_FP"] = df.apply(lambda x: dk_fp(x["PTS"], x["REB"], x["AST"], x["STL"], x["BLK"], x["TOV"], x["FG3M"]), axis=1)
    df["Value"] = df.apply(lambda x: (x["DK_FP"] / (x["Salary"]/1000.0)) if safe_num(x["Salary"], 0) > 0 else 0.0, axis=1)

    # nice rounding
    for c in ["Minutes", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "DK_FP", "Value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].round(2)

    return df

# ==========================
# INPUT: DK CSV
# ==========================
st.sidebar.header("Step A — Upload DraftKings CSV")
dk_file = st.sidebar.file_uploader("DK Salary CSV", type="csv")

use_gist = st.sidebar.checkbox("Use GitHub Gist persistence (recommended)", value=True)

gist_id = None
gh_token = None
if use_gist:
    if "GIST_ID" in st.secrets and "GITHUB_TOKEN" in st.secrets:
        gist_id = st.secrets["GIST_ID"]
        gh_token = st.secrets["GITHUB_TOKEN"]
    else:
        st.sidebar.warning("Missing GIST_ID / GITHUB_TOKEN in secrets. Gist persistence disabled.")
        use_gist = False

if dk_file is None:
    st.info("Upload your DraftKings slate CSV to begin.")
    st.stop()

dk_text = dk_file.getvalue().decode("utf-8", errors="ignore")
dk_raw = pd.read_csv(StringIO(dk_text))

# Try to standardize DK columns
# Expect at least: Name, Salary, TeamAbbrev (or Team), Position/Positions
col_map = {}
if "Name" not in dk_raw.columns and "Name + ID" in dk_raw.columns:
    dk_raw["Name"] = dk_raw["Name + ID"].astype(str).str.replace(r"\s+$begin:math:text$\\d\+$end:math:text$$", "", regex=True)

if "Team" not in dk_raw.columns and "TeamAbbrev" in dk_raw.columns:
    dk_raw["Team"] = dk_raw["TeamAbbrev"]

if "Positions" in dk_raw.columns and "Position" not in dk_raw.columns:
    dk_raw["Position"] = dk_raw["Positions"]

required = ["Name", "Salary", "Team"]
missing = [c for c in required if c not in dk_raw.columns]
if missing:
    st.error(f"DK CSV missing required columns: {missing}")
    st.stop()

dk = pd.DataFrame({
    "Name": dk_raw["Name"].astype(str),
    "Salary": pd.to_numeric(dk_raw["Salary"], errors="coerce").fillna(0).astype(int),
    "Team": dk_raw["Team"].astype(str).str.upper(),
    "Position": dk_raw["Position"].astype(str) if "Position" in dk_raw.columns else "",
})

dk["Name_clean"] = dk["Name"].apply(clean_name)

# ==========================
# OUT FLAGS (persisted)
# ==========================
out_flags = {}
if use_gist:
    txt = gist_read(gist_id, gh_token, GIST_OUT)
    if txt:
        try:
            out_flags = json.loads(txt)
        except Exception:
            out_flags = {}

# initialize session out_flags
if "out_flags" not in st.session_state:
    st.session_state["out_flags"] = out_flags.copy()

# ==========================
# STEP A UI: Mark OUT checkboxes
# ==========================
st.subheader("Step A — Slate (mark OUT players)")

# quick team view
teams = sorted(dk["Team"].unique().tolist())
with st.expander("Quick: Mark OUT by player (checkbox list)", expanded=True):
    st.write("Tip: Use search in your browser (Ctrl+F) to find a name fast.")
    cols = st.columns(3)
    for i, row in dk.sort_values(["Team", "Salary"], ascending=[True, False]).reset_index(drop=True).iterrows():
        col = cols[i % 3]
        key = f"out_{row['Name_clean']}"
        default = bool(st.session_state["out_flags"].get(row["Name_clean"], False))
        val = col.checkbox(f"{row['Team']} — {row['Name']} (${row['Salary']})", value=default, key=key)
        st.session_state["out_flags"][row["Name_clean"]] = bool(val)

# persist out.json
if use_gist:
    try:
        gist_write(gist_id, gh_token, {GIST_OUT: json.dumps(st.session_state["out_flags"])})
    except Exception as e:
        st.warning(f"Could not save OUT flags to Gist: {e}")

# ==========================
# STEP B: RUN PROJECTIONS
# ==========================
st.subheader("Step B — Run Projections")

run_proj = st.button("Run Projections (with injury bumps + minute cap)")

if run_proj:
    nba_df = nba_league_dash()
    with st.spinner("Building projections..."):
        proj_df = build_projections(dk, st.session_state["out_flags"], nba_df)

    # Hide OUT players from the displayed projection table (per your request)
    display_df = proj_df[proj_df["Status"] != "OUT"].copy()

    # Save final.csv to gist
    if use_gist:
        try:
            gist_write(gist_id, gh_token, {GIST_FINAL: proj_df.to_csv(index=False)})
        except Exception as e:
            st.warning(f"Could not save final.csv to Gist: {e}")

    st.success("Projections built.")
    st.caption("OUT players are hidden from this table but remain in the background for bumps.")

    st.dataframe(
        display_df.sort_values(["Value", "Salary"], ascending=[False, False])[
            ["Name", "Team", "Position", "Salary", "Minutes", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "DK_FP", "Value", "Status"]
        ],
        use_container_width=True
    )

    # Also show a quick minutes sanity table
    st.subheader("Minutes sanity check")
    mins_check = display_df.groupby("Team")["Minutes"].sum().reset_index().sort_values("Minutes", ascending=False)
    st.dataframe(mins_check, use_container_width=True)

else:
    # If final exists in gist, load and show latest
    if use_gist:
        final_txt = gist_read(gist_id, gh_token, GIST_FINAL)
        if final_txt:
            try:
                proj_df = pd.read_csv(StringIO(final_txt))
                display_df = proj_df[proj_df["Status"] != "OUT"].copy()
                st.caption("Loaded last saved projections from Gist.")
                st.dataframe(
                    display_df.sort_values(["Value", "Salary"], ascending=[False, False])[
                        ["Name", "Team", "Position", "Salary", "Minutes", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "DK_FP", "Value", "Status"]
                    ],
                    use_container_width=True
                )
            except Exception:
                pass

st.markdown("---")
st.caption(
    "If you want the optimizer re-added into this exact file (with locks / late swap constraints), tell me what DK roster rules you want enforced "
    "(classic: PG, SG, SF, PF, C, G, F, UTIL + $50k) and whether you want team-start locks or player locks."
)