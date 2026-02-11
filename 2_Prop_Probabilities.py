import math
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import requests


# --------------------------
# GIST (same secrets as main app)
# --------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GIST_ID = st.secrets["GIST_ID"]
GIST_FINAL = "final.csv"

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


# --------------------------
# Normal CDF (no scipy)
# --------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def p_over(mu: float, sigma: float, line: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return 1.0 - norm_cdf(z)


# --------------------------
# Volatility model (minutes-adjusted)
# Conservative defaults; you can tune later.
# sigma_base is at ~32 minutes then scaled by sqrt(MIN/32)
# --------------------------
def sigma_base_for_stat(stat: str, mu: float) -> float:
    stat = stat.upper()
    if stat == "PTS":
        return 0.18 * mu + 2.0
    if stat == "REB":
        return 0.22 * mu + 1.2
    if stat == "AST":
        return 0.25 * mu + 1.0
    if stat in ["FG3M", "3PM", "3P"]:
        return 0.35 * mu + 0.8
    # fallback
    return 0.20 * mu + 1.5

def sigma_minutes_adjusted(stat: str, mu: float, minutes: float) -> float:
    base = sigma_base_for_stat(stat, mu)
    minutes = float(minutes) if minutes is not None and not pd.isna(minutes) else 32.0
    minutes = max(10.0, min(40.0, minutes))
    return base * math.sqrt(minutes / 32.0)


# --------------------------
# UI
# --------------------------
st.set_page_config(layout="wide")
st.title("Prop Probabilities — Top Projections → P(Over) Ladder")

final_text = gist_read(GIST_FINAL)
if not final_text:
    st.error("No final.csv found. Run projections (Step B) in the main app first.")
    st.stop()

df = pd.read_csv(StringIO(final_text))

required = ["Name", "Salary", "Minutes", "PTS", "REB", "AST", "FG3M", "DK_FP"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"final.csv missing columns: {missing}")
    st.stop()

# Ensure numeric
for c in ["Salary", "Minutes", "PTS", "REB", "AST", "FG3M", "DK_FP"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Minutes"]).copy()

st.sidebar.subheader("Settings")
rank_metric = st.sidebar.selectbox("Rank Top 10 by", ["DK_FP", "PTS", "REB", "AST", "FG3M"], index=0)
top_n = st.sidebar.slider("Top N players", 5, 25, 10, 1)
stat = st.sidebar.selectbox("Prop stat to evaluate", ["PTS", "REB", "AST", "FG3M"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Line ladder is generated around each player's projection.")
ladder_mode = st.sidebar.selectbox("Line ladder mode", ["Offsets from projection", "Custom list"], index=0)

if ladder_mode == "Offsets from projection":
    # Example: [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
    offsets_str = st.sidebar.text_input("Offsets (comma-separated)", value="-3,-2,-1,-0.5,0.5,1,2,3")
    offsets = []
    for tok in offsets_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            offsets.append(float(tok))
        except:
            pass
    if not offsets:
        offsets = [-2, -1, -0.5, 0.5, 1, 2]
    ladder_label = [f"{o:+g}" for o in offsets]
else:
    lines_str = st.sidebar.text_input("Lines (comma-separated)", value="15.5, 17.5, 19.5, 21.5, 23.5")
    lines = []
    for tok in lines_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            lines.append(float(tok))
        except:
            pass
    lines = sorted(list(set(lines)))
    if not lines:
        lines = [10.5, 12.5, 14.5]
    ladder_label = [f"{x:g}" for x in lines]

# Top N selection
top_df = df.sort_values(rank_metric, ascending=False).head(int(top_n)).copy()

# Build results table
rows = []
for _, r in top_df.iterrows():
    name = r["Name"]
    minutes = float(r["Minutes"]) if not pd.isna(r["Minutes"]) else 32.0
    mu = float(r[stat]) if not pd.isna(r[stat]) else np.nan
    if pd.isna(mu):
        continue
    sigma = sigma_minutes_adjusted(stat, mu, minutes)

    row = {
        "Name": name,
        "Salary": int(r["Salary"]) if not pd.isna(r["Salary"]) else None,
        "Minutes": round(minutes, 1),
        f"Proj_{stat}": round(mu, 2),
        "Sigma": round(sigma, 2),
    }

    if ladder_mode == "Offsets from projection":
        for o, lab in zip(offsets, ladder_label):
            line = mu + float(o)
            row[f"P(Over) @ (Proj{lab})"] = round(p_over(mu, sigma, line), 3)
    else:
        for line, lab in zip(lines, ladder_label):
            row[f"P(Over) @ {lab}"] = round(p_over(mu, sigma, float(line)), 3)

    rows.append(row)

out = pd.DataFrame(rows)

st.subheader(f"Top {top_n} by {rank_metric} — {stat} P(Over) Ladder")
st.dataframe(out, use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download prop probabilities CSV", data=csv_bytes, file_name="prop_probabilities.csv", mime="text/csv")