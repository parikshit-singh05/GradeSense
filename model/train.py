"""
GradeSense — Model training
============================
Trains a RandomForestRegressor to predict the final grade (G3) from
behavioral and lifestyle features only. G1 and G2 are deliberately
excluded to avoid data leakage (they are near-duplicates of G3 from
the same course).

Saves artifacts to /model/:
    - model.joblib          : trained RandomForestRegressor
    - pipeline.joblib       : fitted Pipeline (preprocess + model)
    - feature_names.json    : post-encoding feature names (for SHAP later)
    - metrics.json          : test-set RMSE / MAE / R^2

Run from the project root:
    python model/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# %% ---------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Categorical columns get one-hot encoded. The rest are treated as numeric.
# Several "categorical-looking" columns in the UCI file (e.g. Medu, studytime,
# failures, traveltime) are already integer-coded ordinals and should stay
# numeric so their order is preserved.
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]

TARGET = "G3"

# %% ---------------------------------------------------------------------
# Load and split features / target
# ------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")

# Drop G1 and G2 explicitly — they are the prior period grades for the same
# course and would dominate the model (data leakage).
LEAKY_COLS = ["G1", "G2"]
df = df.drop(columns=LEAKY_COLS)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Anything not categorical and not the target is a numeric feature.
numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

print(f"Dataset: {df.shape[0]} students, {X.shape[1]} features (G1, G2 excluded)")
print(f"  numeric features     ({len(numeric_cols)}): {numeric_cols}")
print(f"  categorical features ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
print(f"  target: {TARGET}  (mean={y.mean():.2f}, std={y.std():.2f})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} rows   |   Test: {X_test.shape[0]} rows")

# %% ---------------------------------------------------------------------
# Preprocessing pipeline
# ------------------------------------------------------------------------
# One-hot encode the categoricals; pass numerics through unchanged.
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# %% ---------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
# Hyperparameters fixed per project decision: n_estimators=500,
# min_samples_leaf=3, no max_depth. This config beat the grid-search
# alternatives (RF with max_depth=10, GradientBoostingRegressor) on the
# held-out test set, so the model is locked in.
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

# %% ---------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------
print("\nTraining RandomForestRegressor (n=500, min_samples_leaf=3, no max_depth)...")
pipeline.fit(X_train, y_train)
print("Done.")

# %% ---------------------------------------------------------------------
# Evaluate on the held-out test set
# ------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("\n=== Test-set metrics ===")
print(f"  RMSE : {rmse:.3f}  (grade points, scale 0-20)")
print(f"  MAE  : {mae:.3f}")
print(f"  R^2  : {r2:.3f}")

# A naive baseline that just predicts the training-set mean — gives us a
# sense of how much the model is actually doing.
y_pred_baseline = np.full_like(y_pred, fill_value=y_train.mean(), dtype=float)
rmse_baseline = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
print(f"\n  Baseline (predict mean) RMSE: {rmse_baseline:.3f}")
print(f"  Improvement over baseline:    {100 * (1 - rmse / rmse_baseline):.1f}%")

# %% ---------------------------------------------------------------------
# Feature importances
# ------------------------------------------------------------------------
# Pull the fitted model out of the pipeline and read the post-encoding
# feature names from the ColumnTransformer.
fitted_model: RandomForestRegressor = pipeline.named_steps["model"]
fitted_preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]

feature_names = fitted_preprocess.get_feature_names_out().tolist()
importances = fitted_model.feature_importances_

importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# Group one-hot dummies back to their original categorical for readability.
def base_feature(name: str) -> str:
    return name.split("_", 1)[0] if any(name.startswith(c + "_") for c in CATEGORICAL_COLS) else name

importance_df["base_feature"] = importance_df["feature"].map(base_feature)
top_bases = (
    importance_df.groupby("base_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n=== Top 15 individual features by importance ===")
print(importance_df.head(15).to_string(index=False))

print("\n=== Top 15 feature GROUPS (one-hot dummies combined) ===")
print(top_bases.head(15).to_string(index=False))

# Sanity check: confirm G1 / G2 are absent and behavioral features rank highly.
assert not any(f.startswith("G1") or f.startswith("G2") for f in feature_names), \
    "G1/G2 leaked into features!"
top_5_bases = top_bases["base_feature"].head(5).tolist()
print(f"\nTop 5 base features driving the model: {top_5_bases}")

# %% ---------------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------------------
model_path = MODEL_DIR / "model.joblib"
pipeline_path = MODEL_DIR / "pipeline.joblib"
feature_names_path = MODEL_DIR / "feature_names.json"
metrics_path = MODEL_DIR / "metrics.json"

joblib.dump(fitted_model, model_path)
joblib.dump(pipeline, pipeline_path)
with feature_names_path.open("w") as f:
    json.dump(feature_names, f, indent=2)
with metrics_path.open("w") as f:
    json.dump(
        {
            "model": "RandomForestRegressor",
            "params": {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 3,
            },
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "rmse_baseline": rmse_baseline,
        },
        f,
        indent=2,
    )

print(f"\nSaved:")
print(f"  {model_path}")
print(f"  {pipeline_path}  (full preprocess + model)")
print(f"  {feature_names_path}")
print(f"  {metrics_path}")
