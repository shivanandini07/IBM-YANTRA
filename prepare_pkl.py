"""
prepare_pkl.py
==============
Prepares best_model.joblib as a watsonx.ai Runtime-compatible .pkl file.

Steps
-----
1. Load data/processed/best_model.joblib
2. Verify model metadata (name, threshold, steps)
3. Run a sanity-check prediction (no changes to the model)
4. Save as data/processed/yantraguard_model.pkl (pickle protocol 4)
5. Round-trip verify: reload the .pkl and confirm prediction is identical

Usage
-----
    python prepare_pkl.py
"""

import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOBLIB_PATH = Path("data/processed/best_model.joblib")
PKL_PATH    = Path("data/processed/yantraguard_model.pkl")

# ---------------------------------------------------------------------------
# Step 1 — Load existing pipeline
# ---------------------------------------------------------------------------
print("=" * 60)
print("  YantraGuard AI — PKL Preparation for watsonx.ai Runtime")
print("=" * 60)

if not JOBLIB_PATH.exists():
    print(f"ERROR: {JOBLIB_PATH} not found. Run `python -m src.train` first.")
    sys.exit(1)

print(f"\n[1/5] Loading {JOBLIB_PATH} ...")
pipeline = joblib.load(JOBLIB_PATH)

model_name     = getattr(pipeline, "model_name",     "N/A")
threshold      = getattr(pipeline, "threshold",      "N/A")
input_features = getattr(pipeline, "input_features", "N/A")
steps          = [s[0] for s in pipeline.steps]
clf_type       = type(pipeline.named_steps["classifier"]).__name__

print(f"    Model name     : {model_name}")
print(f"    Threshold      : {threshold}")
print(f"    Input features : {input_features}")
print(f"    Pipeline steps : {steps}")
print(f"    Classifier     : {clf_type}")
print("    [OK]")

# ---------------------------------------------------------------------------
# Step 2 — Sanity-check predictions BEFORE saving
# ---------------------------------------------------------------------------
print("\n[2/5] Running sanity-check predictions (model unchanged) ...")

LOW_STRESS = pd.DataFrame([{
    "Type":                      "M",
    "Air temperature [K]":       298.1,
    "Process temperature [K]":   308.6,
    "Rotational speed [rpm]":    1551.0,
    "Torque [Nm]":               42.8,
    "Tool wear [min]":           0.0,
}])

HIGH_STRESS = pd.DataFrame([{
    "Type":                      "L",
    "Air temperature [K]":       302.0,
    "Process temperature [K]":   311.0,
    "Rotational speed [rpm]":    1380.0,
    "Torque [Nm]":               60.0,
    "Tool wear [min]":           210.0,
}])

prob_low  = float(pipeline.predict_proba(LOW_STRESS)[0, 1])
prob_high = float(pipeline.predict_proba(HIGH_STRESS)[0, 1])

print(f"    Low-stress  failure prob : {prob_low:.4f}  (expected < 0.20)")
print(f"    High-stress failure prob : {prob_high:.4f}  (expected > 0.70)")

assert prob_low  < 0.20, f"FAIL: Low-stress prob {prob_low:.4f} should be < 0.20"
assert prob_high > 0.70, f"FAIL: High-stress prob {prob_high:.4f} should be > 0.70"
print("    [OK] Both predictions are within expected ranges")

# ---------------------------------------------------------------------------
# Step 3 — Save as .pkl (pickle protocol 4)
# ---------------------------------------------------------------------------
print(f"\n[3/5] Saving as {PKL_PATH} (pickle protocol 4) ...")
PKL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(PKL_PATH, "wb") as f:
    pickle.dump(pipeline, f, protocol=4)

size_kb = PKL_PATH.stat().st_size / 1024
print(f"    File size : {size_kb:.1f} KB")
print(f"    Path      : {PKL_PATH.resolve()}")
print("    [OK]")

# ---------------------------------------------------------------------------
# Step 4 — Round-trip verify: reload and predict
# ---------------------------------------------------------------------------
print(f"\n[4/5] Round-trip verification: reload {PKL_PATH} and predict ...")

with open(PKL_PATH, "rb") as f:
    reloaded = pickle.load(f)

prob_low_rt  = float(reloaded.predict_proba(LOW_STRESS)[0, 1])
prob_high_rt = float(reloaded.predict_proba(HIGH_STRESS)[0, 1])

print(f"    Low-stress  (reloaded) : {prob_low_rt:.4f}")
print(f"    High-stress (reloaded) : {prob_high_rt:.4f}")

assert abs(prob_low_rt  - prob_low)  < 1e-8, "FAIL: Low-stress prediction mismatch after reload!"
assert abs(prob_high_rt - prob_high) < 1e-8, "FAIL: High-stress prediction mismatch after reload!"
print("    [OK] Predictions are bit-for-bit identical after reload")

# ---------------------------------------------------------------------------
# Step 5 — Metadata check on reloaded model
# ---------------------------------------------------------------------------
print("\n[5/5] Metadata check on reloaded model ...")
assert getattr(reloaded, "model_name", None) == model_name,     "model_name mismatch"
assert abs(getattr(reloaded, "threshold", 0) - threshold) < 1e-9, "threshold mismatch"
assert getattr(reloaded, "input_features", None) == input_features, "input_features mismatch"
print(f"    model_name     : {getattr(reloaded, 'model_name', 'N/A')}")
print(f"    threshold      : {getattr(reloaded, 'threshold', 'N/A')}")
print(f"    input_features : {getattr(reloaded, 'input_features', 'N/A')}")
print("    [OK]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  PREPARATION COMPLETE")
print("=" * 60)
print(f"  Source  : {JOBLIB_PATH}")
print(f"  Output  : {PKL_PATH}")
print(f"  Size    : {size_kb:.1f} KB")
print(f"  Protocol: pickle 4 (Python 3.8+ / watsonx.ai Runtime compatible)")
print(f"  Model   : {model_name}")
print(f"  Threshold: {threshold}")
print()
print("  Next step: upload yantraguard_model.pkl to watsonx.ai Runtime")
print("  using the training notebook (notebooks/01_yantraguard_training.ipynb)")
print("  or via src/ibm_client.py  WatsonxClient().deploy_model()")
print("=" * 60)
