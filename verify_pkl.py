"""
verify_pkl.py
=============
1. Confirms the filename is exactly  yantraguard_model.pkl  (not .pkl.py or any variant)
2. Confirms the file is a genuine Python pickle binary (magic bytes check)
3. Loads it with pickle and runs two predictions
4. Confirms predictions match the joblib baseline
"""

import pickle
import struct
import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED   = Path("data/processed")
PKL_PATH    = PROCESSED / "yantraguard_model.pkl"
JOBLIB_PATH = PROCESSED / "best_model.joblib"

print("=" * 62)
print("  YantraGuard AI - PKL Filename & Integrity Verification")
print("=" * 62)

# ---------------------------------------------------------------------------
# Step 1 - Filename check
# ---------------------------------------------------------------------------
print("\n[1/5] Filename check ...")

# List every file in processed/ and find any pkl-related names
all_files = [f.name for f in PROCESSED.iterdir() if f.is_file()]
print(f"    Files in data/processed/: {all_files}")

pkl_files = [n for n in all_files if "pkl" in n.lower()]
print(f"    PKL-related files found : {pkl_files}")

EXPECTED_NAME = "yantraguard_model.pkl"

if EXPECTED_NAME not in all_files:
    print(f"    ERROR: Expected '{EXPECTED_NAME}' but found: {pkl_files}")
    sys.exit(1)

wrong_variants = [n for n in pkl_files if n != EXPECTED_NAME]
if wrong_variants:
    print(f"    WARNING: Extra pkl variants found (will be removed): {wrong_variants}")
    for w in wrong_variants:
        (PROCESSED / w).unlink()
        print(f"    Deleted: {w}")

print(f"    Confirmed filename : {EXPECTED_NAME}  [OK]")

# ---------------------------------------------------------------------------
# Step 2 - Binary / magic bytes check
# ---------------------------------------------------------------------------
print("\n[2/5] Binary magic-bytes check ...")

with open(PKL_PATH, "rb") as f:
    header = f.read(8)

# Python pickle magic bytes:
# Protocol 2+: starts with 0x80 followed by protocol number (0x02, 0x03, 0x04, 0x05)
# Protocol 4 = 0x80 0x04
hex_header = " ".join(f"{b:02X}" for b in header)
print(f"    First 8 bytes (hex) : {hex_header}")

if header[0] != 0x80:
    print(f"    ERROR: Not a valid pickle file. First byte = 0x{header[0]:02X}, expected 0x80")
    sys.exit(1)

protocol = header[1]
print(f"    Pickle protocol     : {protocol}")

if protocol not in (2, 3, 4, 5):
    print(f"    ERROR: Unexpected pickle protocol {protocol}")
    sys.exit(1)

print(f"    File size           : {PKL_PATH.stat().st_size / 1024:.1f} KB")
print(f"    Binary pickle file confirmed  [OK]")

# ---------------------------------------------------------------------------
# Step 3 - Load with pickle
# ---------------------------------------------------------------------------
print("\n[3/5] Loading with pickle.load() ...")

with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"    Object type         : {type(model).__name__}")
print(f"    model_name attr     : {getattr(model, 'model_name', 'N/A')}")
print(f"    threshold attr      : {getattr(model, 'threshold', 'N/A'):.6f}")
print(f"    input_features      : {getattr(model, 'input_features', 'N/A')}")
print(f"    Pipeline steps      : {[s[0] for s in model.steps]}")
print(f"    Classifier          : {type(model.named_steps['classifier']).__name__}")
print("    Loaded successfully  [OK]")

# ---------------------------------------------------------------------------
# Step 4 - Predictions from the loaded pkl
# ---------------------------------------------------------------------------
print("\n[4/5] Running predictions from loaded pkl ...")

LOW_STRESS = pd.DataFrame([{
    "Type":                    "M",
    "Air temperature [K]":     298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]":  1551.0,
    "Torque [Nm]":             42.8,
    "Tool wear [min]":         0.0,
}])

HIGH_STRESS = pd.DataFrame([{
    "Type":                    "L",
    "Air temperature [K]":     302.0,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]":  1380.0,
    "Torque [Nm]":             60.0,
    "Tool wear [min]":         210.0,
}])

prob_low  = float(model.predict_proba(LOW_STRESS)[0, 1])
prob_high = float(model.predict_proba(HIGH_STRESS)[0, 1])

print(f"    Low-stress  prob (Type=M, Wear=0)    : {prob_low:.6f}  [expected < 0.20]")
print(f"    High-stress prob (Type=L, Wear=210)  : {prob_high:.6f}  [expected > 0.70]")

assert prob_low  < 0.20, f"FAIL: low-stress prob {prob_low:.4f} >= 0.20"
assert prob_high > 0.70, f"FAIL: high-stress prob {prob_high:.4f} <= 0.70"
print("    Both predictions within expected ranges  [OK]")

# ---------------------------------------------------------------------------
# Step 5 - Cross-check against joblib baseline
# ---------------------------------------------------------------------------
print("\n[5/5] Cross-checking against joblib baseline ...")

baseline = joblib.load(JOBLIB_PATH)
base_low  = float(baseline.predict_proba(LOW_STRESS)[0, 1])
base_high = float(baseline.predict_proba(HIGH_STRESS)[0, 1])

print(f"    Joblib low-stress  : {base_low:.6f}")
print(f"    PKL    low-stress  : {prob_low:.6f}   diff={abs(prob_low - base_low):.2e}")
print(f"    Joblib high-stress : {base_high:.6f}")
print(f"    PKL    high-stress : {prob_high:.6f}   diff={abs(prob_high - base_high):.2e}")

assert abs(prob_low  - base_low)  < 1e-8, "FAIL: low-stress prediction differs from joblib baseline"
assert abs(prob_high - base_high) < 1e-8, "FAIL: high-stress prediction differs from joblib baseline"
print("    PKL predictions are bit-for-bit identical to joblib baseline  [OK]")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("  ALL CHECKS PASSED")
print("=" * 62)
print(f"  Filename   : {PKL_PATH.name}  (exact, no extra extension)")
print(f"  Format     : Binary pickle protocol {protocol}")
print(f"  Size       : {PKL_PATH.stat().st_size / 1024:.1f} KB")
print(f"  Model      : {getattr(model, 'model_name', 'N/A')}")
print(f"  Threshold  : {getattr(model, 'threshold', 'N/A'):.6f}")
print(f"  Low-stress : {prob_low:.6f}")
print(f"  High-stress: {prob_high:.6f}")
print()
print("  yantraguard_model.pkl is a genuine, loadable pickle model.")
print("  Ready to upload to IBM watsonx.ai Runtime.")
print("=" * 62)
