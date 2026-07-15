"""
GradeSense — Exploratory Data Analysis
======================================
Loads /data/student-mat.csv, inspects it, and produces charts
showing which features most affect the final grade (G3).

Run from the project root:
    python notebooks/eda.py

Outputs PNGs to /notebooks/figures/.

Tip: open in VS Code and use the "# %%" cell markers to step through
interactively with the Jupyter extension.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% ---------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent chart styling — readable in light and dark README themes.
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# %% ---------------------------------------------------------------------
# Load the data
# ------------------------------------------------------------------------
# The UCI file is semicolon-delimited with quoted strings.
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# %% ---------------------------------------------------------------------
# Column info
# ------------------------------------------------------------------------
print("\n--- Column dtypes ---")
print(df.dtypes)

print("\n--- First 3 rows ---")
print(df.head(3).to_string())

# %% ---------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------------
print("\n--- Numeric summary ---")
print(df.describe().T.round(2))

print("\n--- Categorical summary (object cols) ---")
cat_cols = df.select_dtypes(include="object").columns.tolist()
if cat_cols:
    print(df[cat_cols].describe().T)

# %% ---------------------------------------------------------------------
# Missing values
# ------------------------------------------------------------------------
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing": missing, "pct": missing_pct})
print("\n--- Missing values per column ---")
print(missing_report)
total_missing = int(missing.sum())
print(f"\nTotal missing cells: {total_missing}")
if total_missing == 0:
    print("(Dataset has no missing values — UCI Student Performance is already cleaned.)")

# %% ---------------------------------------------------------------------
# Correlations with G3
# ------------------------------------------------------------------------
# Use only numeric columns; one-hot any categoricals for a fuller picture.
numeric_corr = df.corr(numeric_only=True)["G3"].drop("G3").sort_values(ascending=False)
print("\n--- Pearson correlation with G3 (numeric features only) ---")
print(numeric_corr.round(3))

# Add one-hot categoricals to get a complete feature ranking.
df_encoded = pd.get_dummies(df, drop_first=True)
full_corr = df_encoded.corr()["G3"].drop("G3").sort_values(key=lambda s: s.abs(), ascending=False)
print("\n--- |Pearson correlation| with G3 (all features, one-hot encoded, top 15) ---")
print(full_corr.head(15).round(3))

# %% ---------------------------------------------------------------------
# Chart 1 — G3 distribution (sanity check)
# ------------------------------------------------------------------------
fig, ax = plt.subplots()
counts, bins, _ = ax.hist(df["G3"], bins=range(0, 22), edgecolor="white", color="#4C78A8")
for c, b in zip(counts, bins):
    if c > 0:
        ax.text(b + 0.5, c, int(c), ha="center", va="bottom", fontsize=9)
ax.set_title("Distribution of final grade (G3)")
ax.set_xlabel("G3 (0-20)")
ax.set_ylabel("Number of students")
ax.set_xticks(range(0, 21, 2))
fig.tight_layout()
fig.savefig(FIG_DIR / "01_g3_distribution.png")
plt.show()

# %% ---------------------------------------------------------------------
# Chart 2 — Top correlations with G3 (bar chart)
# ------------------------------------------------------------------------
top_n = 12
top = numeric_corr.head(top_n)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#54A24B" if v > 0 else "#E45756" for v in top.values]
ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Top {top_n} numeric features correlated with G3")
ax.set_xlabel("Pearson r with G3")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_top_correlations.png")
plt.show()

# %% ---------------------------------------------------------------------
# Chart 3 — Study time, failures, absences, goout vs G3 (the request)
# ------------------------------------------------------------------------
# Build a 2x2 panel of the four features the user called out.
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

def box(ax, col, title, order=None):
    data = [df.loc[df[col] == v, "G3"].values for v in (order or sorted(df[col].unique()))]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    palette = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#B279A2"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(order or sorted(df[col].unique()))
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("G3")
    ax.set_ylim(-1, 21)

box(axes[0, 0], "studytime", "Study time vs G3")          # 1..4
box(axes[0, 1], "failures",  "Past failures vs G3")        # 0..4
box(axes[1, 0], "goout",     "Going out vs G3")           # 1..5

# Absences is too wide (0..93) for a per-integer boxplot, so bin into quartiles.
df["_absences_bin"] = pd.qcut(df["absences"], q=4, duplicates="drop")
order = list(df["_absences_bin"].cat.categories)
data = [df.loc[df["_absences_bin"] == v, "G3"].values for v in order]
bp = axes[1, 1].boxplot(data, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], ["#4C78A8", "#54A24B", "#E45756", "#F58518"][:len(order)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[1, 1].set_xticks(range(1, len(order) + 1))
axes[1, 1].set_xticklabels([str(o) for o in order], rotation=15, ha="right")
axes[1, 1].set_title("Absences (quartiles) vs G3")
axes[1, 1].set_xlabel("absences quartile")
axes[1, 1].set_ylabel("G3")
axes[1, 1].set_ylim(-1, 21)

fig.suptitle("How lifestyle and study habits relate to final grade", fontsize=14, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_features_vs_g3.png", bbox_inches="tight")
plt.show()

# %% ---------------------------------------------------------------------
# Chart 4 — G1 and G2 (prior period grades) vs G3, the strongest signal
# ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col in zip(axes, ["G1", "G2"]):
    ax.scatter(df[col], df["G3"], alpha=0.55, s=28, color="#4C78A8", edgecolor="white")
    # Fit line
    coef = np.polyfit(df[col], df["G3"], 1)
    xs = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="#E45756", linewidth=2, label=f"y = {coef[0]:.2f}x + {coef[1]:.2f}")
    r = df[[col, "G3"]].corr().iloc[0, 1]
    ax.set_title(f"{col} vs G3   (r = {r:.3f})")
    ax.set_xlabel(col)
    ax.set_ylabel("G3")
    ax.legend(loc="upper left", frameon=False)

fig.suptitle("Prior period grades are the strongest predictors of G3", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_prior_grades_vs_g3.png", bbox_inches="tight")
plt.show()

# %% ---------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------
print("\n--- Key takeaways ---")
print(f"1. Strongest numeric predictors of G3: {numeric_corr.head(3).index.tolist()}")
print(f"2. Strongest negative numeric predictor: {numeric_corr.idxmin()} (r = {numeric_corr.min():.3f})")
print(f"3. No missing values: {total_missing == 0}")
print(f"4. Charts saved to: {FIG_DIR}")
print("\nDone.")
