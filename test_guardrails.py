"""
test_guardrails.py - Quick validation of guardrails and sample_data modules
"""
from sample_data import generate_benchmark_data
from guardrails import (
    validate_csv_upload, ImmutabilityGuardrail,
    ExtremePreservationGuardrail, generate_audit_trail, compute_sha256
)

raw, labels = generate_benchmark_data()

# Test immutability guardrail
guard = ImmutabilityGuardrail()
checksum = guard.register_raw(raw)
print("Raw checksum (first 16):", checksum[:16], "...")
unchanged, current = guard.verify_raw_unchanged(raw)
print("Immutability check (should be True):", unchanged)

# Test extreme preservation
extremes = ExtremePreservationGuardrail.find_valid_extremes(raw)
print("Valid extremes found:", len(extremes))
for _, row in extremes.iterrows():
    sid = row["station_id"]
    dt  = row["date"]
    val = row["rainfall_mm"]
    print(f"  {sid} | {dt} | {val} mm")

# Test CSV upload validation
csv_bytes = raw.to_csv(index=False).encode()
validated_df, _warns, _hash, _cm, _sf = validate_csv_upload(csv_bytes, "test.csv")
print("Upload validation passed:", len(validated_df), "rows")

# Test SHA256
h = compute_sha256(raw)
print("SHA256 (first 16):", h[:16], "...")
print("All guardrails OK!")
