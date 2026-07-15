"""
GradeSense — Streamlit app
==========================
Real-time G3 prediction with SHAP explanation.

The sidebar exposes only the top-8 contributing features (by RF
feature importance): studytime, failures, absences, goout, health,
Walc, Fedu, Medu. Every other input the model needs is filled with
a fixed "typical student" value (dataset median for numerics, mode
for categoricals) inside this file — so the model still gets a
complete 30-column row, but the UI stays uncluttered.

Visible sliders are grouped under two section headers:
  - "Study & Attendance"  : studytime, failures, absences
  - "Lifestyle & Family"  : goout, health, Walc, Fedu, Medu

Run from the project root:
    streamlit run app/main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make app/ importable under any launch mode (bare streamlit run, AppTest,
# python -m, etc.). Harmless when the directory is already on sys.path.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import init_db, log_prediction, read_history

# %% ---------------------------------------------------------------------
# Paths and artifact loading (cached so we don't re-read on every input)
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

# These categorical columns and the load order MUST match model/train.py.
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
TARGET = "G3"

# %% ---------------------------------------------------------------------
# Human-readable label dictionaries
# ------------------------------------------------------------------------
# Three label forms for every feature:
#   - LONG:   used in sidebar slider labels and the suggestion text
#             (full, plain English, with scale hints in parentheses)
#   - SHORT:  used in the SHAP chart axis (1-2 words, no scale hints)
#
# Anything not listed here falls back to a title-cased version of its
# raw column name, so future columns just work.
LABELS: dict[str, dict[str, str]] = {
    "school":     {"long": "School",                          "short": "School"},
    "sex":        {"long": "Sex",                             "short": "Sex"},
    "age":        {"long": "Age",                             "short": "Age",
                   "suffix": "(years)"},
    "address":    {"long": "Home Address Type",               "short": "Address"},
    "famsize":    {"long": "Family Size",                     "short": "Family Size"},
    "Pstatus":    {"long": "Parents Live Together",           "short": "Parents Together"},
    "Medu":       {"long": "Mother's Education Level",        "short": "Mother's Education",
                   "suffix": "(0 = none, 4 = higher education)"},
    "Fedu":       {"long": "Father's Education Level",        "short": "Father's Education",
                   "suffix": "(0 = none, 4 = higher education)"},
    "Mjob":       {"long": "Mother's Job",                    "short": "Mother's Job"},
    "Fjob":       {"long": "Father's Job",                    "short": "Father's Job"},
    "reason":     {"long": "Reason for Choosing This School", "short": "School Choice"},
    "guardian":   {"long": "Guardian",                        "short": "Guardian"},
    "traveltime": {"long": "Travel Time to School",           "short": "Travel Time",
                   "suffix": "(1 = <15min, 4 = >1h)"},
    "studytime":  {"long": "Weekly Study Time",               "short": "Study Time",
                   "suffix": "(1 = <2h, 4 = >10h)"},
    "failures":   {"long": "Past Class Failures",             "short": "Past Failures"},
    "schoolsup":  {"long": "Extra School Support",            "short": "School Support"},
    "famsup":     {"long": "Family Educational Support",      "short": "Family Support"},
    "paid":       {"long": "Extra Paid Math Classes",         "short": "Paid Classes"},
    "activities": {"long": "In Extracurricular Activities",   "short": "Activities"},
    "nursery":    {"long": "Attended Nursery School",         "short": "Nursery"},
    "higher":     {"long": "Wants Higher Education",          "short": "Higher Ed"},
    "internet":   {"long": "Internet Access at Home",         "short": "Internet"},
    "romantic":   {"long": "In a Romantic Relationship",      "short": "Romantic"},
    "famrel":     {"long": "Family Relationship Quality",     "short": "Family Relationship",
                   "suffix": "(1 = very bad, 5 = excellent)"},
    "freetime":   {"long": "Free Time After School",          "short": "Free Time",
                   "suffix": "(1 = very low, 5 = very high)"},
    "goout":      {"long": "Going Out with Friends",          "short": "Going Out",
                   "suffix": "(1 = very low, 5 = very high)"},
    "Dalc":       {"long": "Workday Alcohol Consumption",     "short": "Workday Alcohol",
                   "suffix": "(1 = very low, 5 = very high)"},
    "Walc":       {"long": "Weekend Alcohol Consumption",     "short": "Weekend Alcohol",
                   "suffix": "(1 = very low, 5 = very high)"},
    "health":     {"long": "Self-Rated Health",               "short": "Health",
                   "suffix": "(1 = very bad, 5 = very good)"},
    "absences":   {"long": "Absences (Days Missed)",          "short": "Absences"},
}


def long_label(col: str) -> str:
    """Sidebar / suggestion text label. Falls back to title-cased col name."""
    entry = LABELS.get(col)
    if not entry:
        return col.replace("_", " ").title()
    suffix = entry.get("suffix", "")
    return f"{entry['long']} {suffix}".strip()


def short_label(col: str) -> str:
    """SHAP chart axis label. Falls back to title-cased col name."""
    entry = LABELS.get(col)
    if not entry:
        return col.replace("_", " ").title()
    return entry["short"]


# %% ---------------------------------------------------------------------
# "Typical student" defaults for the features we DON'T expose in the UI
# ------------------------------------------------------------------------
# Computed from the UCI Math dataset (student-mat.csv). Numeric features
# use the median; categorical features use the mode. This is the row the
# model sees when the user hasn't moved a slider yet.
#
# Keep this in sync with model/train.py's column order (the model
# pipeline expects exactly this set of columns, in this order).
DEFAULT_INPUT: dict[str, object] = {
    # Categorical (mode from the dataset)
    "school":     "GP",
    "sex":        "F",
    "address":    "U",
    "famsize":    "GT3",
    "Pstatus":    "T",
    "Mjob":       "other",
    "Fjob":       "other",
    "reason":     "course",
    "guardian":   "mother",
    "schoolsup":  "no",
    "famsup":     "yes",
    "paid":       "no",
    "activities": "no",
    "nursery":    "yes",
    "higher":     "yes",
    "internet":   "yes",
    "romantic":   "no",
    # Numeric (median from the dataset) — every column the model
    # expects, even the ones the user can adjust. The slider values
    # overlay these at prediction time, but the dict still needs
    # the keys so DEFAULT_INPUT can act as a complete row template.
    "age":        17,
    "Medu":       3,
    "Fedu":       2,
    "traveltime": 1,
    "studytime":  2,
    "failures":   0,
    "famrel":     4,
    "freetime":   3,
    "goout":      3,
    "Dalc":       1,
    "Walc":       2,
    "health":     4,
    "absences":   4,
}

# %% ---------------------------------------------------------------------
# Sidebar slider groupings
# ------------------------------------------------------------------------
# The 8 dashboard-exposed sliders, in display order. Each entry has the
# raw column name (matches DEFAULT_INPUT / model columns) and the
# slider's min/max. The label rendered in the sidebar comes from
# long_label() above so we never show raw column names.
SLIDER_GROUPS: list[dict] = [
    {
        "key":     "study",
        "title":   "Study & Attendance",
        "sliders": [
            {"col": "studytime", "min": 1, "max": 4},
            {"col": "failures",  "min": 0, "max": 4},
            {"col": "absences",  "min": 0, "max": 75,
             "help": "School days missed. Original data range 0-93; capped at 75 (99th percentile)."},
        ],
    },
    {
        "key":     "lifestyle",
        "title":   "Lifestyle & Family Background",
        "sliders": [
            {"col": "goout",  "min": 1, "max": 5},
            {"col": "health", "min": 1, "max": 5},
            {"col": "Walc",   "min": 1, "max": 5},
            {"col": "Fedu",   "min": 0, "max": 4},
            {"col": "Medu",   "min": 0, "max": 4},
        ],
    },
]

# %% ---------------------------------------------------------------------
# Map of plain-English advice per SHAP base-feature
# ------------------------------------------------------------------------
# Used by the auto-suggestion sentence. Keys are the post-encoding base
# feature names (the bit before the first underscore in a SHAP column).
# The advice text uses the LONG human-readable name.
SUGGESTION_TIPS: dict[str, str] = {
    "absences":  f"{long_label('absences')} are the biggest drag on this prediction — even modest improvements in attendance should lift the grade.",
    "failures":  f"{long_label('failures')} are pulling the prediction down — extra tutoring or repeat coursework is the most impactful next step.",
    "studytime": f"{long_label('studytime')} is below the level that the model expects — adding a few hours of focused study a week should help.",
    "goout":     f"{long_label('goout')} is pushing the prediction down — rebalancing social time with study time would help.",
    "Dalc":      f"{long_label('Dalc')} is hurting the prediction — even small reductions typically show up in grades.",
    "Walc":      f"{long_label('Walc')} is hurting the prediction — even small reductions typically show up in grades.",
    "health":    f"{long_label('health')} is pulling the prediction down — getting more sleep and addressing health issues would help.",
    "freetime":  f"{long_label('freetime')} is high relative to study time — reallocating some of it to focused study should lift the grade.",
    "age":       f"{long_label('age')} is acting as a drag on the prediction — usually a proxy for being held back; targeted support is the lever.",
    "famrel":    f"{long_label('famrel')} is dragging on the prediction — family support at home would help.",
    "Medu":      f"{long_label('Medu')} is acting as a drag — usually a proxy for home academic support; tutoring can substitute.",
    "Fedu":      f"{long_label('Fedu')} is acting as a drag — usually a proxy for home academic support; tutoring can substitute.",
    "traveltime":f"{long_label('traveltime')} is pulling the prediction down — using that time for study (notes, reading) could offset it.",
    "school":    f"The {long_label('school')} indicator is pulling the prediction down — usually a baseline effect rather than something the student can change.",
    "Mjob":      f"{long_label('Mjob')} is acting as a drag — usually a proxy for socioeconomic context, not something the student can change.",
    "Fjob":      f"{long_label('Fjob')} is acting as a drag — usually a proxy for socioeconomic context, not something the student can change.",
    "reason":    f"The {long_label('reason')} is dragging on the prediction — usually a baseline effect rather than something actionable.",
    "guardian":  f"{long_label('guardian')} is acting as a drag — usually a baseline effect rather than something the student can change.",
    "schoolsup": f"The student is not receiving extra school support — enrolling in any available support programs should help.",
    "famsup":    "Family educational support is low — asking for more structure and accountability at home should help.",
    "paid":      f"{long_label('paid')} classes are not in place — adding one (if affordable) should help.",
    "activities":f"{long_label('activities')} is acting as a small drag — joining one can help indirectly via engagement.",
    "nursery":   f"{long_label('nursery')} is acting as a small drag — usually a baseline effect, not actionable now.",
    "higher":    f"{long_label('higher')} is pulling the prediction down — committing to a plan for after secondary school should help.",
    "internet":  f"{long_label('internet')} is a drag — arranging regular access (library, school, etc.) should help.",
    "romantic":  f"{long_label('romantic')} is pulling the prediction down — usually a small effect, but worth being aware of.",
    "sex":       f"{long_label('sex')} is acting as a drag — a baseline effect, not something the student can change.",
    "address":   f"{long_label('address')} is acting as a small drag — usually a baseline effect.",
    "famsize":   f"{long_label('famsize')} is acting as a small drag — a baseline effect, not something the student can change.",
    "Pstatus":   f"{long_label('Pstatus')} is a small drag — a baseline effect, not something the student can change.",
}


# %% ---------------------------------------------------------------------
# Artifact + DB loading
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model + pipeline + SHAP explainer + cached background values."""
    pipeline   = joblib.load(MODEL_DIR / "pipeline.joblib")
    explainer  = joblib.load(MODEL_DIR / "explainer.joblib")
    bg_values  = joblib.load(MODEL_DIR / "shap_values.joblib")
    base_value = float((MODEL_DIR / "shap_base_value.txt").read_text().strip())
    with (MODEL_DIR / "shap_feature_names.json").open() as f:
        feature_names = json.load(f)
    return pipeline, explainer, bg_values, base_value, feature_names


def predict_and_explain(student_df: pd.DataFrame, artifacts):
    """Run the full pipeline: raw row -> predicted G3 + per-feature SHAP."""
    pipeline, explainer, _bg, base_value, feature_names = artifacts
    preprocess = pipeline.named_steps["preprocess"]
    model      = pipeline.named_steps["model"]

    x_enc = preprocess.transform(student_df)                  # (1, n_features)
    prediction = float(model.predict(x_enc)[0])
    shap_row   = np.asarray(explainer.shap_values(x_enc))[0]  # (n_features,)

    contrib = (
        pd.DataFrame({"feature": feature_names, "shap": shap_row})
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return prediction, base_value, contrib


def risk_tier(g3: float) -> tuple[str, str]:
    """Return (label, color-hex) for the risk pill.

    Colors track the app palette (see .streamlit/config.toml + the CSS
    block in main.py): green for low risk, desaturated amber for medium,
    red for high. Slightly more muted than Streamlit's defaults so the
    pill sits comfortably inside the white card.
    """
    if g3 >= 14:
        return "Low risk",    "#059669"   # green (matches --positive)
    if g3 >= 10:
        return "Medium risk", "#D97706"   # desaturated amber
    return "High risk", "#DC2626"         # red (matches --negative)


def render_suggestion(contrib: pd.DataFrame) -> str:
    """
    Pick the most-negative SHAP feature and look up plain-English advice.
    Uses the LONG human-readable name in the message. For one-hot columns
    (e.g. "Mjob_at_home") we strip the value suffix and look up the base
    name; the lookup table already contains a fully-phrased sentence.

    Falls back to a generic line when the contrib DataFrame is empty.
    """
    if contrib.empty:
        return "No specific suggestion — the model is close to its baseline."
    most_negative = contrib.sort_values("shap").iloc[0]
    base_name = most_negative["feature"].split("_", 1)[0]
    return SUGGESTION_TIPS.get(
        base_name,
        f"{long_label(base_name)} is the largest drag on the prediction — addressing it should help.",
    )


def render_shap_plotly(contrib: pd.DataFrame) -> go.Figure:
    """
    Bar chart of feature contributions for the current student, rendered
    in Plotly so it can pick up the app's palette (positive=green,
    negative=red) and feel like one product. Y-axis uses SHORT labels
    (1-2 words), so we never show raw column names on the chart.

    One-hot dummies (e.g. Mjob_at_home, Mjob_teacher) are collapsed into
    their base feature ("Mother's Job") by SUMMING the SHAP values. This
    matches the EDA-time feature-grouping table and avoids two bars
    stacking at the same y-position with the same label.
    """
    # Palette — kept in sync with .streamlit/config.toml and the CSS
    # block at the top of the file. See plan: --ink, --positive,
    # --negative, --accent.
    COLOR_POSITIVE = "#059669"
    COLOR_NEGATIVE = "#DC2626"
    COLOR_INK      = "#0F172A"
    COLOR_MUTED    = "#475569"
    COLOR_GRID     = "#E2E8F0"

    top = contrib.head(56).copy()  # enough rows that grouping still gives 10
    top["base"] = top["feature"].map(lambda f: f.split("_", 1)[0])
    # Sum SHAP values within each base feature, then sort by importance.
    grouped = (
        top.groupby("base", as_index=False)["shap"]
           .sum()
           .assign(abs_shap=lambda d: d["shap"].abs())
           .sort_values("abs_shap", ascending=False)
           .head(10)
           .iloc[::-1]
    )
    grouped["display"] = grouped["base"].map(short_label)
    colors = [COLOR_NEGATIVE if v < 0 else COLOR_POSITIVE for v in grouped["shap"]]

    fig = go.Figure(go.Bar(
        x=grouped["shap"],
        y=grouped["display"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLOR_INK, line_width=1)

    fig.update_layout(
        template="simple_white",
        title=None,                        # subtitle is rendered via st.caption instead
        xaxis=dict(
            title=dict(text="SHAP contribution to predicted G3",
                       font=dict(color=COLOR_MUTED, size=12)),
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickfont=dict(color=COLOR_MUTED, size=11),
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            tickfont=dict(color=COLOR_INK, size=12),
        ),
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# %% ---------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="GradeSense — Student Performance Predictor",
    page_icon="\U0001F393",
    layout="wide",
)

# %% ---------------------------------------------------------------------
# Custom CSS — typography + cards + sidebar dividers
# ------------------------------------------------------------------------
# .streamlit/config.toml handles the global theme (accent color, base
# background, body text). This block fills in the things config.toml
# doesn't reach: the Inter web font, the card chrome for the prediction
# summary, a quieter treatment for the sidebar section dividers, and
# a bolder treatment for the sidebar "Student profile" header.
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif;
        }

        /* Tighter, calmer headings so the page doesn't feel like a default
           Streamlit hero section. */
        h1 { font-weight: 700; font-size: 1.75rem; letter-spacing: -0.01em; }
        h2 { font-weight: 600; font-size: 1.15rem; letter-spacing: -0.005em; }
        h3 { font-weight: 600; font-size: 1.00rem; }

        /* Prediction summary card — sits on the off-white surface defined
           in .streamlit/config.toml, with a soft shadow to read as a card. */
        .gradesense-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 22px 26px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04),
                        0 1px 3px rgba(15,23,42,0.04);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            margin: 4px 0 8px 0;
        }
        .gradesense-card .gs-label {
            color: #475569;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .gradesense-card .gs-value {
            color: #0F172A;
            font-size: 44px;
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }
        .gradesense-card .gs-delta {
            color: #475569;
            font-size: 14px;
            margin-top: 4px;
        }
        .gradesense-card .gs-delta strong { color: #0F172A; font-weight: 600; }
        .gradesense-card .gs-right { text-align: right; }
        .gs-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            color: #FFFFFF;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        .gs-pill-caption {
            color: #94A3B8;
            font-size: 11px;
            margin-top: 6px;
            letter-spacing: 0.02em;
        }

        /* Sidebar section dividers — make the two groups ("Study &
           Attendance", "Lifestyle & Family Background") feel like
           card-like subsections rather than default subheaders. */
        section[data-testid="stSidebar"] h2 {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
            border-top: 1px solid #E2E8F0;
            padding-top: 14px;
            margin-top: 18px;
        }
        section[data-testid="stSidebar"] h2:first-of-type {
            border-top: none;
            padding-top: 0;
            margin-top: 8px;
        }

        /* Sidebar "Student profile" header — bump it up a step so it
           reads as the section title, not as a peer of the small
           uppercase dividers below it. The default st.header is ~1.5rem;
           we nudge to 1.6rem and lock in weight 700. */
        section[data-testid="stSidebar"] h1 {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: #0F172A;
            margin: 4px 0 6px 0;
            padding: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# %% ---------------------------------------------------------------------
# Header — centered logo
# ------------------------------------------------------------------------
# Replace the emoji + text title with the brand logo. The canonical home
# for the file is `app/assets/GradeSense_Logo.svg` (alongside the other
# app modules) — we check there first. The project root is a fallback
# for convenience: drop the file there if you don't want to move it
# inside the package. The 1/4 / 1/2 / 1/4 column split is the cleanest
# way to center a single st.image on a wide page.
_LOGO_CANDIDATES = [
    PROJECT_ROOT / "app" / "assets" / "GradeSense_Logo.svg",
    PROJECT_ROOT / "GradeSense_Logo.svg",
]
LOGO_PATH = next((p for p in _LOGO_CANDIDATES if p.exists()), _LOGO_CANDIDATES[0])

_logo_left, _logo_center, _logo_right = st.columns([1, 2, 1])
with _logo_center:
    st.image(str(LOGO_PATH), width=200)
st.caption(
    "Real-time G3 prediction with explainable AI. G1 and G2 are "
    "intentionally excluded — the model uses behavioral and lifestyle "
    "features only. Adjust the eight sliders in the sidebar; the rest of the "
    "student profile uses typical values."
)

artifacts = load_artifacts()
init_db()  # idempotent; safe to call every rerun

# %% ---------------------------------------------------------------------
# Sidebar — only the top 8 features by importance, grouped under 2 headers
# ------------------------------------------------------------------------
# The model still receives a full 30-column row. Missing fields come
# from DEFAULT_INPUT above.
with st.sidebar:
    st.header("Student profile")
    st.caption("Adjust the sliders — the prediction updates in real time. "
               "Other features the model uses are held at dataset averages.")

    # Render each group as one subheader + its sliders, in SLIDER_GROUPS order.
    slider_values: dict[str, int] = {}
    for grp in SLIDER_GROUPS:
        st.subheader(grp["title"])
        for s in grp["sliders"]:
            kwargs = dict(
                label=long_label(s["col"]),
                min_value=s["min"],
                max_value=s["max"],
                value=int(DEFAULT_INPUT[s["col"]]),
                key=f"slider_{s['col']}",
            )
            if "help" in s:
                kwargs["help"] = s["help"]
            slider_values[s["col"]] = int(st.slider(**kwargs))

# %% ---------------------------------------------------------------------
# Build the full 30-column row: visible sliders + DEFAULT_INPUT
# ------------------------------------------------------------------------
student_row = pd.DataFrame([{**DEFAULT_INPUT, **slider_values}])

prediction, base_value, contrib = predict_and_explain(student_row, artifacts)
tier_label, tier_color = risk_tier(prediction)

# Persist this prediction. Each Streamlit rerun (triggered by any input
# change) writes a row, giving a complete audit trail of what the user saw.
log_prediction(
    predicted_g3=prediction,
    risk_tier=tier_label.replace(" risk", ""),   # store "Low" / "Medium" / "High"
    payload=student_row.iloc[0].to_dict(),
)

# %% ---------------------------------------------------------------------
# Main area — prediction card + SHAP + suggestion
# ------------------------------------------------------------------------
# One card: predicted G3 (big) + vs-class-average delta + risk pill on
# the right. Replaces the previous two-column st.metric / risk-badge
# split so all three pieces of information read as one unit.
delta = prediction - base_value
delta_sign = "+" if delta >= 0 else ""
st.markdown(
    f"""
    <div class="gradesense-card">
        <div>
            <div class="gs-label">Predicted final grade (G3, 0–20 scale)</div>
            <div class="gs-value">{prediction:.2f}</div>
            <div class="gs-delta">
                <strong>{delta_sign}{delta:.2f}</strong> vs class average
                &nbsp;·&nbsp; baseline {base_value:.2f}
                &nbsp;·&nbsp; G1 and G2 not used as inputs
            </div>
        </div>
        <div class="gs-right">
            <span class="gs-pill" style="background:{tier_color};">{tier_label}</span>
            <div class="gs-pill-caption">Low ≥ 14 &nbsp;·&nbsp; Medium 10–13 &nbsp;·&nbsp; High &lt; 10</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Why this prediction?")
st.caption(
    "SHAP bar chart: each feature's contribution to the predicted G3. "
    "Green pushes the grade up, red pushes it down."
)
st.plotly_chart(render_shap_plotly(contrib), width="stretch", theme=None)

st.divider()

st.subheader("\U0001F4A1 Suggested next step")
st.info(render_suggestion(contrib))

# %% ---------------------------------------------------------------------
# Recent predictions — collapsible log pulled from /app/history.db
# ------------------------------------------------------------------------
st.divider()
with st.expander("Recent Predictions", expanded=False):
    history = read_history(limit=20)
    if not history:
        st.caption("No predictions logged yet — adjust the inputs above and one will appear here.")
    else:
        # Two side-by-side tables: a compact summary on the left, the
        # most recent full input row on the right.
        summary_rows = [
            {
                "When (UTC)":     r["created_at"],
                "Predicted G3":   f"{r['predicted_g3']:.2f}",
                "Risk tier":      r["risk_tier"],
                "Top driver":     (short_label(sorted(r["payload"].items(),
                                                       key=lambda kv: -abs(kv[1]) if isinstance(kv[1], (int, float)) else 0)[0][0])
                                    if r["payload"] else ""),
            }
            for r in history
        ]
        st.caption(f"Showing the {len(history)} most recent predictions, newest first.")
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        with st.expander("Show full input payload of the most recent prediction", expanded=False):
            st.json(history[0]["payload"])

# %% ---------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------
st.caption(
    "Model: RandomForestRegressor (n=500, min_samples_leaf=3). "
    "Trained on UCI Student Performance (Math) with G1/G2 dropped to prevent leakage. "
    "Test RMSE 3.83, R² 0.28 — see model/metrics.json."
)
