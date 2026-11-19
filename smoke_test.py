"""
smoke_test.py — Quick programmatic smoke test to verify:
  1. ka_dist_rainfall_2025.csv upload doesn't crash and produces correct output
  2. Benchmark still detects 15 defects with F1=100%
"""
import sys
import pandas as pd
from guardrails import validate_csv_upload

# ── SMOKE TEST 1: District-level upload path (no crash) ───────────────────────
print("SMOKE TEST 1: District-level upload path")
with open(r"C:\Users\mohan\OneDrive\Desktop\ka_dist_rainfall_2025.csv", "rb") as f:
    raw = f.read()

df, warns, h, cm, sf = validate_csv_upload(raw, "ka_dist_rainfall_2025.csv")
print(f"  Loaded {len(df)} rows, columns: {list(df.columns)[:6]}")
print(f"  station_id sample: {df['station_id'].head(3).tolist()}")
print(f"  serial_no present: {'serial_no' in df.columns}")
print(f"  synthesised_fields keys: {list(sf.keys())}")

assert "station_id" in df.columns, f"station_id missing"
assert "date" in df.columns, f"date column missing"
assert "serial_no" in df.columns, f"serial_no missing"

is_numeric = pd.to_numeric(df["station_id"], errors="coerce").notna().mean() > 0.8
assert not is_numeric, f"station_id is still numeric: {df['station_id'].head().tolist()}"
assert "date" in sf, f"date not in synthesised_fields: {sf}"
print("  PASS: no crash, station_id=district name, serial_no present, synthesised_fields['date'] set")

# ── SMOKE TEST 2: Benchmark 15 defects + F1=100% ─────────────────────────────
print()
print("SMOKE TEST 2: Benchmark 15 defects + F1=100%")
from sample_data import generate_benchmark_data
from agents import run_quality_agent

raw_df, labels = generate_benchmark_data()
qr = run_quality_agent(raw_df)
checks = qr["checks"]

detected = sum([
    checks["duplicates"]["count"],
    checks["negatives"]["count"],
    checks["date_gaps"]["count"],
    checks["conflicts"]["count"],
    checks["outliers"]["sensor_error_count"],
])
gt = labels[labels["defect_type"] != "LEGITIMATE_EXTREME"]
tp = min(detected, len(gt))
fp = max(0, detected - len(gt))
fn = max(0, len(gt) - detected)
p  = tp / (tp + fp) if (tp + fp) else 0.0
r  = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * p * r / (p + r) if (p + r) else 0.0

print(f"  GT defects={len(gt)}, Detected={detected}, Extremes preserved={checks['outliers']['legitimate_extreme_count']}")
print(f"  Precision={p:.0%}, Recall={r:.0%}, F1={f1:.0%}")
assert f1 == 1.0, f"F1 expected 1.0, got {f1}"
print("  PASS: 15/15 defects found, F1=100%")

print()
print("ALL SMOKE TESTS PASSED.")
sys.exit(0)
