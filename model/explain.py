"""
GradeSense — SHAP explanations
==============================
Loads the trained pipeline (model + preprocess) from /model/, computes
SHAP values for the training set using shap.TreeExplainer, and saves
artifacts the Streamlit app can load instantly.

Saves to /model/:
    - explainer.joblib     : shap.TreeExplainer wrapping the fitted RF
    - shap_values.joblib   : SHAP values matrix (n_train, n_features)
                             for the training set — the background the
                             app will subtract to score new students
    - shap_base_value.txt  : the explainer's expected_value (E[f(X)])
    - shap_feature_names.json : feature names aligned to the matrix

The app should:
    1. Load explainer.joblib + shap_values.joblib
    2. Transform a single new student with the saved pipeline's
       preprocess step
    3. Compute shap_values for that single row against the background

We expose a small CLI to sanity-check the wiring end-to-end.

Run from the project root:
    python model/explain.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# %% ---------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
MODEL_DIR = PROJECT_ROOT / "model"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "G3"
LEAKY_COLS = ["G1", "G2"]   # must match train.py

# %% ---------------------------------------------------------------------
# Load the trained pipeline
# ------------------------------------------------------------------------
pipeline_path = MODEL_DIR / "pipeline.joblib"
pipeline = joblib.load(pipeline_path)
model = pipeline.named_steps["model"]
preprocess = pipeline.named_steps["preprocess"]

print(f"Loaded pipeline from {pipeline_path}")
print(f"  model: {type(model).__name__}  (n_estimators={model.n_estimators})")

# %% ---------------------------------------------------------------------
# Reload + split the data the same way train.py did, so the background
# set matches what the model was fit on.
# ------------------------------------------------------------------------
from sklearn.model_selection import train_test_split  # local import keeps the
                                                     # dependency surface small

df = pd.read_csv(DATA_PATH, sep=";").drop(columns=LEAKY_COLS)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set: {X_train.shape[0]} rows x {X_train.shape[1]} cols (G1, G2 dropped)")

# %% ---------------------------------------------------------------------
# Transform the training set through the fitted preprocessor
# ------------------------------------------------------------------------
# This is the numeric/encoded matrix the model actually sees. SHAP has to
# operate on this — not on the raw categorical strings.
X_train_enc = preprocess.transform(X_train)
feature_names = preprocess.get_feature_names_out().tolist()
print(f"Encoded feature matrix: {X_train_enc.shape}")
print(f"  features: {len(feature_names)} (e.g. {feature_names[:3]} ...)")

# %% ---------------------------------------------------------------------
# Build the TreeExplainer
# ------------------------------------------------------------------------
# shap.TreeExplainer on a random forest is exact (no sampling) and is
# the recommended path for tree models — it's also cheap to re-construct
# from the fitted model, so saving it is mostly a convenience for the
# app (avoids re-wrapping the model on every load).
explainer = shap.TreeExplainer(model)
# shap >= ~0.42 returns expected_value as a 0-d or length-1 ndarray for
# tree regressors. Coerce to a Python float either way.
ev = np.asarray(explainer.expected_value).item()
base_value = float(ev)
print(f"\nSHAP base value (E[f(X)]): {base_value:.3f}  "
      f"(approx training-set mean G3 = {y_train.mean():.3f})")

# %% ---------------------------------------------------------------------
# Compute SHAP values over the training set (the "background")
# ------------------------------------------------------------------------
# This is the part that actually takes time (~seconds for an RF on 316
# rows). We cache it so the app never has to recompute.
print("Computing SHAP values over the training set...")
shap_values = explainer.shap_values(X_train_enc)
print(f"  shape: {shap_values.shape}  (n_samples x n_features)")

# %% ---------------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------------------
explainer_path = MODEL_DIR / "explainer.joblib"
shap_values_path = MODEL_DIR / "shap_values.joblib"
base_value_path = MODEL_DIR / "shap_base_value.txt"
shap_features_path = MODEL_DIR / "shap_feature_names.json"

joblib.dump(explainer, explainer_path)
joblib.dump(shap_values, shap_values_path)
base_value_path.write_text(f"{base_value}\n")
with shap_features_path.open("w") as f:
    json.dump(feature_names, f, indent=2)

print(f"\nSaved:")
print(f"  {explainer_path}")
print(f"  {shap_values_path}  ({shap_values.nbytes / 1024:.1f} KB)")
print(f"  {base_value_path}  ({base_value:.3f})")
print(f"  {shap_features_path}")

# %% ---------------------------------------------------------------------
# Sanity check: pick one test student and show their top-3 drivers
# ------------------------------------------------------------------------
# Use the first test student so the explanation is reproducible.
sample_idx = 0
sample_raw = X_test.iloc[[sample_idx]]
sample_actual = float(y_test.iloc[sample_idx])

sample_enc = preprocess.transform(sample_raw)
sample_pred = float(model.predict(sample_enc)[0])
sample_shap = explainer.shap_values(sample_enc)[0]  # shape (n_features,)

# Rank features by absolute SHAP contribution.
contrib = (
    pd.DataFrame({"feature": feature_names, "shap": sample_shap})
    .assign(abs_shap=lambda d: d["shap"].abs())
    .sort_values("abs_shap", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Sanity check: one test student ===")
print(f"  student index in test set: {sample_idx}")
print(f"  actual G3                 : {sample_actual}")
print(f"  predicted G3               : {sample_pred:.3f}")
print(f"  base value (E[f(X)])       : {base_value:.3f}")
print(f"  sum of SHAP contributions  : {sample_shap.sum():+.3f}  "
      f"(should approx equal predicted - base)")

print("\nTop 3 features driving this student's predicted G3:")
for i, row in contrib.head(3).iterrows():
    direction = "pushes grade UP" if row["shap"] > 0 else "pushes grade DOWN"
    print(f"  {i+1}. {row['feature']:<22}  SHAP = {row['shap']:+.3f}   {direction}")
