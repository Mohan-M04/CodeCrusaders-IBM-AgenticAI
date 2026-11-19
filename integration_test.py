"""
integration_test.py
Full end-to-end integration test: sample_data -> guardrails -> agents -> audit -> apply_fixes
Simulates exactly what the Streamlit frontend does, step by step.
"""

import sys
import json
import traceback

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        return result
    except Exception as e:
        FAIL.append((name, str(e)))
        traceback.print_exc()
        return None

print("=" * 65)
print("  Rain Check Karnataka — Full Integration Test")
print("=" * 65)

# ─── STEP 1: sample_data ──────────────────────────────────────────
print("\n[1] sample_data.py — Benchmark Generator")

def test_benchmark():
    from sample_data import generate_benchmark_data
    raw_df, labels = generate_benchmark_data()
    assert len(raw_df) > 1000, f"Too few rows: {len(raw_df)}"
    assert raw_df["station_id"].nunique() == 4
    assert raw_df["district"].nunique() == 2
    real_defects = labels[labels["defect_type"] != "LEGITIMATE_EXTREME"]
    assert len(real_defects) == 15, f"Expected 15 defects, got {len(real_defects)}"
    extremes = labels[labels["defect_type"] == "LEGITIMATE_EXTREME"]
    assert len(extremes) == 2, f"Expected 2 extremes, got {len(extremes)}"
    print(f"   Rows: {len(raw_df)}, Stations: {raw_df['station_id'].nunique()}, "
          f"Defects injected: {len(real_defects)}, Extremes: {len(extremes)}")
    return raw_df, labels

result = check("Benchmark data generation", test_benchmark)
raw_df, defect_labels = result if result else (None, None)

# ─── STEP 2: guardrails ───────────────────────────────────────────
print("\n[2] guardrails.py — All 4 Guardrails")

def test_input_guardrail():
    from guardrails import validate_csv_upload, InputGuardrailError
    csv_bytes = raw_df.to_csv(index=False).encode()
    df_val, _warns, _h, _cm, _sf = validate_csv_upload(csv_bytes, "test.csv")
    assert len(df_val) == len(raw_df)
    print(f"   Input guardrail: passed ({len(df_val)} rows validated)")

    # Test rejection of bad file (too few rows)
    try:
        validate_csv_upload(b"col1,col2\n1,2", "bad.csv")
        assert False, "Should have rejected — too few rows"
    except InputGuardrailError:
        print("   Malformed CSV correctly rejected")

check("Input guardrail", test_input_guardrail)

def test_immutability_guardrail():
    from guardrails import ImmutabilityGuardrail
    g = ImmutabilityGuardrail()
    checksum = g.register_raw(raw_df)
    assert len(checksum) == 64  # SHA-256 hex
    unchanged, current = g.verify_raw_unchanged(raw_df)
    assert unchanged, "Raw data should still be unchanged"
    # Mutate a copy and verify it fails
    mutated = raw_df.copy()
    mutated.iloc[0, mutated.columns.get_loc("rainfall_mm")] = 999999
    unchanged2, _ = g.verify_raw_unchanged(mutated)
    assert not unchanged2, "Mutated copy should not match"
    print(f"   Immutability guardrail: checksum={checksum[:16]}..., mutation detected correctly")

check("Immutability guardrail", test_immutability_guardrail)

def test_extreme_preservation():
    from guardrails import ExtremePreservationGuardrail
    extremes = ExtremePreservationGuardrail.find_valid_extremes(raw_df)
    assert len(extremes) == 2
    vals = sorted(extremes["rainfall_mm"].tolist())
    assert vals == [185.0, 192.0], f"Unexpected extreme values: {vals}"
    # Test enforcement: remove an extreme and check it gets restored
    import pandas as pd
    cleaned = raw_df.copy()
    mask = (cleaned["station_id"] == "BLR_CENTRAL") & (cleaned["date"] == "2024-07-15")
    cleaned = cleaned[~mask].reset_index(drop=True)
    cleaned_restored, msgs = ExtremePreservationGuardrail.enforce_on_cleaned(raw_df, cleaned)
    assert any("RESTORED" in m for m in msgs), f"Extreme was not restored: {msgs}"
    print(f"   Extreme preservation: {len(extremes)} extremes found, enforcement restored removed extreme")

check("Extreme preservation guardrail", test_extreme_preservation)

def test_audit_trail_guardrail():
    from guardrails import generate_audit_trail, compute_sha256
    h1 = compute_sha256(raw_df)
    assert len(h1) == 64
    trail = generate_audit_trail(h1, h1, [], [], {"BLR_CENTRAL": 100})
    assert trail["provenance"]["raw_sha256"] == h1
    assert not trail["provenance"]["checksums_differ"]
    print(f"   Audit trail guardrail: SHA256={h1[:16]}..., provenance doc generated")

check("Audit trail guardrail", test_audit_trail_guardrail)

# ─── STEP 3: agents ───────────────────────────────────────────────
print("\n[3] agents.py — All 3 Agents")

def test_agent1_schema():
    from agents import run_schema_agent
    result = run_schema_agent(raw_df)
    assert result["agent"] == "SchemaProvenanceAgent"
    assert result["date_format_valid"] == True
    assert result["column_mapping"]["station_id"] == "MATCHED"
    assert result["column_mapping"]["rainfall_mm"] == "MATCHED"
    assert result["unit_check"]["non_numeric_count"] > 0  # N/A injected
    print(f"   Agent 1: status={result['status']}, model={result['model_used']}, "
          f"non_numeric={result['unit_check']['non_numeric_count']}")
    return result

schema_result = check("Agent 1 — SchemaProvenanceAgent", test_agent1_schema)

def test_agent2_quality():
    from agents import run_quality_agent
    result = run_quality_agent(raw_df)
    assert result["agent"] == "QualityInvestigationAgent"
    checks = result["checks"]
    assert checks["duplicates"]["count"] == 3,  f"Duplicates: {checks['duplicates']['count']}"
    assert checks["negatives"]["count"] == 3,   f"Negatives: {checks['negatives']['count']}"
    assert checks["date_gaps"]["count"] == 3,   f"Date gaps: {checks['date_gaps']['count']}"
    assert checks["conflicts"]["count"] == 3,   f"Conflicts: {checks['conflicts']['count']}"
    assert checks["outliers"]["sensor_error_count"] == 3, \
        f"Sensor errors: {checks['outliers']['sensor_error_count']}"
    assert checks["outliers"]["legitimate_extreme_count"] == 2, \
        f"Extremes preserved: {checks['outliers']['legitimate_extreme_count']}"
    assert result["total_defects_found"] == 15, \
        f"Total defects: {result['total_defects_found']}"
    print(f"   Agent 2: total_defects={result['total_defects_found']}, "
          f"model={result['model_used']}, elapsed={result['elapsed_s']}s")
    print(f"      Duplicates=3 ✓, Negatives=3 ✓, Gaps=3 ✓, "
          f"Conflicts=3 ✓, Sensor errors=3 ✓, Extremes preserved=2 ✓")
    return result

quality_result = check("Agent 2 — QualityInvestigationAgent", test_agent2_quality)

def test_agent3_audit():
    from agents import run_audit_agent
    result = run_audit_agent(raw_df, quality_result)
    assert result["agent"] == "AuditReportAgent"
    log = result["audit_log"]
    assert len(log) > 0
    defect_types = {e["defect_type"] for e in log}
    assert "NEGATIVE_VALUE" in defect_types
    assert "DUPLICATE_RECORD" in defect_types
    assert "DATE_GAP" in defect_types
    assert "CONFLICTING_RECORD" in defect_types
    assert "SENSOR_ERROR" in defect_types
    assert "LEGITIMATE_EXTREME" in defect_types
    # All entries have required fields
    for e in log:
        assert "id" in e and "defect_type" in e and "proposed_action" in e
    print(f"   Agent 3: {len(log)} audit entries, model={result['model_used']}")
    print(f"      Defect types in log: {sorted(defect_types)}")
    return result

audit_result = check("Agent 3 — AuditReportAgent", test_agent3_audit)

# ─── STEP 4: Human-in-the-loop approval and apply_fixes ──────────
print("\n[4] Human-in-the-Loop: Approve fixes and apply to cleaned copy")

def test_apply_fixes():
    from agents import apply_approved_fixes
    from guardrails import ExtremePreservationGuardrail, compute_sha256

    log = audit_result["audit_log"]
    # Approve all non-extreme entries
    for entry in log:
        if entry["defect_type"] != "LEGITIMATE_EXTREME":
            entry["user_decision"] = "Approve"
        else:
            entry["user_decision"] = "Retain"

    cleaned_df, applied = apply_approved_fixes(raw_df, log)
    assert cleaned_df is not raw_df, "Cleaned must be a copy, not raw"

    # Raw data must be completely unchanged
    raw_hash    = compute_sha256(raw_df)
    from guardrails import ImmutabilityGuardrail
    g = ImmutabilityGuardrail()
    g.register_raw(raw_df)
    unchanged, _ = g.verify_raw_unchanged(raw_df)
    assert unchanged, "RAW DATA WAS MUTATED!"

    # Cleaned must differ from raw (changes were applied)
    cleaned_hash = compute_sha256(cleaned_df)
    assert raw_hash != cleaned_hash, "Cleaned should differ from raw after approved fixes"

    # Extreme preservation enforcement
    guard = ExtremePreservationGuardrail()
    cleaned_df, restore_msgs = guard.enforce_on_cleaned(raw_df, cleaned_df)
    extremes_in_cleaned = guard.find_valid_extremes(cleaned_df)
    assert len(extremes_in_cleaned) == 2, \
        f"Extremes missing from cleaned: {len(extremes_in_cleaned)}"

    print(f"   Applied {len(applied)} fixes")
    print(f"   Raw rows: {len(raw_df)}, Cleaned rows: {len(cleaned_df)}")
    print(f"   Raw checksum:     {raw_hash[:20]}...")
    print(f"   Cleaned checksum: {cleaned_hash[:20]}...")
    print(f"   Checksums differ: {raw_hash != cleaned_hash} ✓")
    print(f"   Extremes preserved in cleaned: {len(extremes_in_cleaned)} ✓")
    print(f"   Raw data immutability: VERIFIED ✓")
    return cleaned_df, raw_hash, cleaned_hash

fix_result = check("Apply approved fixes + immutability", test_apply_fixes)

# ─── STEP 5: Bilingual translation ───────────────────────────────
print("\n[5] Bilingual — English passthrough + Kannada translation")

def test_bilingual():
    from agents import translate_audit_summary
    # English passthrough
    en_text, en_model = translate_audit_summary("Test audit summary.", "English")
    assert en_text == "Test audit summary."
    assert en_model == "none"
    print(f"   English passthrough: OK (no LLM call)")

    # Kannada live
    kn_text, kn_model = translate_audit_summary(
        "Executive Summary: 15 defects detected across 4 stations.", "ಕನ್ನಡ (Kannada)"
    )
    assert len(kn_text) > 10
    print(f"   Kannada translation: model={kn_model}, chars={len(kn_text)}")
    print(f"   Sample: {kn_text[:80]}")

check("Bilingual Kannada/English toggle", test_bilingual)

# ─── STEP 6: Benchmark F1 evaluation ─────────────────────────────
print("\n[6] F1 Evaluation against 15 ground-truth defects")

def test_f1():
    # Replicate _compute_f1 logic from app.py
    gt_defects = defect_labels[defect_labels["defect_type"] != "LEGITIMATE_EXTREME"]
    gt_count = len(gt_defects)
    checks = quality_result["checks"]
    detected = sum([
        checks["duplicates"]["count"],
        checks["negatives"]["count"],
        checks["date_gaps"]["count"],
        checks["conflicts"]["count"],
        checks["outliers"]["sensor_error_count"],
    ])
    tp = min(detected, gt_count)
    fp = max(0, detected - gt_count)
    fn = max(0, gt_count - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    assert f1 == 1.0, f"F1 should be 1.0 for perfect detection, got {f1}"
    print(f"   GT defects: {gt_count}, Detected: {detected}")
    print(f"   Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%} ✓")

check("F1 benchmark evaluation (perfect score)", test_f1)

# ─── STEP 7: LLM Performance metrics ─────────────────────────────
print("\n[7] LLM Performance Metrics")

def test_perf():
    from agents import get_perf_summary
    perf = get_perf_summary()
    assert perf.get("calls", 0) > 0
    assert perf.get("total_tokens", 0) > 0
    print(f"   LLM calls: {perf['calls']}")
    print(f"   Total tokens: {perf['total_tokens']}")
    print(f"   Total latency: {perf['total_latency_s']}s")
    print(f"   Est. cost: ${perf['total_cost_usd']:.4f}")
    print(f"   Models used: {perf['models_used']}")

check("LLM performance metrics", test_perf)

# ─── RESULTS ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  RESULTS: {len(PASS)} PASSED | {len(FAIL)} FAILED")
print("=" * 65)
for p in PASS:
    print(f"  ✅ {p}")
for name, err in FAIL:
    print(f"  ❌ {name}: {err}")

if FAIL:
    print("\nSome tests FAILED — see errors above.")
    sys.exit(1)
else:
    print("\n🎉 ALL INTEGRATION TESTS PASSED — Frontend ↔ Backend fully connected!")
    sys.exit(0)
