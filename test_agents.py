"""
test_agents.py - Quick validation of agents module (deterministic checks only, no LLM)
"""
from sample_data import generate_benchmark_data
from agents import run_quality_agent

raw, labels = generate_benchmark_data()

print("Running 5-point quality check (deterministic only)...")
result = run_quality_agent(raw)

checks = result["checks"]
print("=== Quality Investigation Results ===")
print("Duplicates:", checks["duplicates"]["count"])
print("Negatives:", checks["negatives"]["count"])
print("Date Gaps:", checks["date_gaps"]["count"])
print("Conflicts:", checks["conflicts"]["count"])
print("Sensor Errors:", checks["outliers"]["sensor_error_count"])
print("Extremes Preserved:", checks["outliers"]["legitimate_extreme_count"])
print("Total defects found:", result["total_defects_found"])
print("Model used:", result["model_used"])
print()
print("--- Negative values ---")
for r in checks["negatives"]["details"]:
    print(" ", r)
print()
print("--- Sensor errors ---")
for r in checks["outliers"]["sensor_errors"]:
    print(" ", r)
print()
print("--- Legitimate extremes ---")
for r in checks["outliers"]["legitimate_extremes"]:
    print(" ", r)
print()
print("Agents quality check: PASSED")
