"""
test_full_suite.py
Runs:
  A) New uploader tests (encoding, delimiter, alias mapping, district-level, Excel)
  B) 12 integration tests (all existing backend logic)
"""
import sys, io, json, traceback
import pandas as pd
from guardrails import (
    validate_csv_upload, InputGuardrailError,
    _apply_alias_mapping, _detect_district_level, _expand_district_level,
    ImmutabilityGuardrail, ExtremePreservationGuardrail,
    generate_audit_trail, compute_sha256, _sha256_bytes,
)

PASS, FAIL = [], []

def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        return result
    except Exception as e:
        FAIL.append((name, str(e)))
        traceback.print_exc()
        return None

def _make_csv(columns, rows, sep=",", encoding="utf-8"):
    lines = [sep.join(columns)] + [sep.join(str(v) for v in r) for r in rows]
    return "\n".join(lines).encode(encoding)

STD_COLS = ["station_id","station_name","district","taluk","date","rainfall_mm"]
ROWS15   = [["BLR_CENTRAL","Bengaluru Central","Bengaluru Urban","BN North",
              f"2024-01-{i+1:02d}","5.2"] for i in range(15)]

# ════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION A — Uploader & Guardrail Tests")
print("═"*60)

# A1: Standard UTF-8 CSV (alias map fires for exact-match cols too, so warns may be non-empty)
def a1():
    df, warns, h, cm, sf = validate_csv_upload(_make_csv(STD_COLS, ROWS15), "std.csv")
    assert len(df) == 15
    assert len(h) == 64
    # Alias notes are OK; just ensure no hard-error warnings
    hard_warns = [w for w in warns if w.startswith("⚠️")]
    assert len(hard_warns) == 0, f"Unexpected hard warnings: {hard_warns}"
    print(f"  Standard UTF-8: OK (hash={h[:12]}..., info_warns={len(warns)})")
check("A1: Standard UTF-8 comma CSV", a1)

# A2: UTF-8-sig BOM (the ka_dist file uses this)
def a2():
    raw = b"\xef\xbb\xbf" + _make_csv(STD_COLS, ROWS15)[0:]  # prepend BOM
    df, warns, h, cm, sf = validate_csv_upload(raw, "bom.csv")
    assert len(df) == 15
    assert any("BOM" in w or "utf-8-sig" in w for w in warns), f"No BOM warning: {warns}"
    assert len(h) == 64
    print(f"  UTF-8-sig BOM: OK, warning='{warns[0][:50]}'")
check("A2: UTF-8 BOM (utf-8-sig) detection", a2)

# A3: CP1252 with accent character
def a3():
    rows_accent = [["BLR_CENTRAL","Bengaluru Céntre","Bengaluru Urban","BN",
                    f"2024-01-{i+1:02d}","5.2"] for i in range(15)]
    b = _make_csv(STD_COLS, rows_accent, encoding="cp1252")
    df, warns, h, cm, sf = validate_csv_upload(b, "cp1252.csv")
    assert len(df) == 15
    print(f"  CP1252: OK, warns={[w[:40] for w in warns]}")
check("A3: CP1252 encoding with accent", a3)

# A4: Semicolon-delimited
def a4():
    b = _make_csv(STD_COLS, ROWS15, sep=";")
    df, warns, h, cm, sf = validate_csv_upload(b, "semi.csv")
    assert len(df) == 15
    assert any("semicolon" in w.lower() for w in warns)
    print(f"  Semicolon: OK")
check("A4: Semicolon-delimited CSV", a4)

# A5: Alias column names — should map WITHOUT rejection
# Note: "Sl No" maps to serial_no (never station_id); alias mapping fires for the others.
def a5():
    alt = ["Sl No","Station","District_Name","Taluka","Obs_Date","Rainfall(mm)"]
    rows = [["1","BLR Central","Bengaluru Urban","BN North",
             f"2024-01-{i+1:02d}","5.2"] for i in range(15)]
    b = _make_csv(alt, rows)
    df, warns, h, cm, sf = validate_csv_upload(b, "alias.csv")
    assert len(df) == 15, f"Should load: {len(df)}"
    # "Sl No" → serial_no (correct — it's a row number, not a station identifier)
    assert "serial_no" in df.columns, f"Sl No should map to serial_no, cols={list(df.columns)}"
    # Other aliases should have fired (station_name, district, taluk, date, rainfall_mm)
    assert len(cm) > 0, f"Expected alias mapping, got: {cm}"
    print(f"  Alias mapping: OK, Sl No→serial_no ✓, mapped={list(cm.values())[:4]}")
check("A5: Alias column names (auto-mapped, not rejected)", a5)

# A6: District-level data (the real ka_dist_rainfall_2025.csv format)
def a6():
    with open(r"C:\Users\mohan\OneDrive\Desktop\ka_dist_rainfall_2025.csv","rb") as f:
        raw = f.read()
    orig_hash = _sha256_bytes(raw)
    df, warns, h, cm, sf = validate_csv_upload(raw, "ka_dist_rainfall_2025.csv")
    assert h == orig_hash, "SHA-256 must be computed on original bytes"
    assert len(df) >= 10
    # After expansion, should have standard columns
    assert "station_id"  in df.columns, f"Missing station_id: {list(df.columns)}"
    assert "date"        in df.columns, f"Missing date: {list(df.columns)}"
    assert "rainfall_mm" in df.columns, f"Missing rainfall_mm: {list(df.columns)}"
    district_rows = df[df["station_id"].notna()]
    assert len(district_rows) >= 10
    assert any("district" in w.lower() or "District-level" in w for w in warns)
    print(f"  ka_dist file: OK — {len(df)} rows, cols={list(df.columns)[:6]}")
    print(f"  Original SHA-256: {h[:16]}...")
    print(f"  Warnings ({len(warns)}): {[w[:60] for w in warns[:3]]}")
check("A6: Real ka_dist_rainfall_2025.csv (utf-8-sig + district-level)", a6)

# A7: SHA-256 on original bytes (not post-encoding bytes)
def a7():
    raw = _make_csv(STD_COLS, ROWS15)
    orig_hash = _sha256_bytes(raw)
    df, warns, h, cm, sf = validate_csv_upload(raw, "hash_test.csv")
    assert h == orig_hash, f"Hash mismatch: {h} vs {orig_hash}"
    print(f"  SHA-256 on original bytes: OK (hash={h[:16]}...)")
check("A7: SHA-256 computed on original bytes", a7)

# A8: Hard reject — oversized
def a8():
    big = _make_csv(STD_COLS, ROWS15) + b"\x00" * (51 * 1024 * 1024)
    try:
        validate_csv_upload(big, "big.csv")
        assert False
    except InputGuardrailError as e:
        assert "50 MB" in str(e)
        print(f"  Oversized: correctly rejected")
check("A8: Oversized file → hard reject", a8)

# A9: Hard reject — wrong extension (now must be in {csv,tsv,txt,xls,xlsx})
def a9():
    b = _make_csv(STD_COLS, ROWS15)
    try:
        validate_csv_upload(b, "data.pdf")
        assert False
    except InputGuardrailError as e:
        print(f"  Wrong extension: correctly rejected ({str(e)[:60]})")
check("A9: Wrong extension (.pdf) → hard reject", a9)

# A10: Hard reject — injection pattern
def a10():
    evil = b"station_id,date,rainfall_mm\n<script>x</script>,2024-01-01,5.0\n"*15
    try:
        validate_csv_upload(evil, "xss.csv")
        assert False
    except InputGuardrailError as e:
        assert "suspicious" in str(e).lower()
        print(f"  Injection: correctly rejected")
check("A10: Injection pattern → hard reject", a10)

# A11: TSV extension accepted
def a11():
    b = _make_csv(STD_COLS, ROWS15, sep="\t")
    df, warns, h, cm, sf = validate_csv_upload(b, "data.tsv")
    assert len(df) == 15
    print(f"  TSV extension: OK")
check("A11: TSV extension accepted", a11)

# ─────────────────────────────────────────────────────────────────────────────
# NEW TESTS (Fix verification)
# ─────────────────────────────────────────────────────────────────────────────

# A12: Sl No → serial_no, station_id comes from district name (not serial number)
def a12():
    """District-level file: 'Sl No' must become serial_no; station_id = district name."""
    dist_cols = ["Sl No", "District", "Annual Normal", "Annual Actual"]
    dist_rows = [
        [str(i+1), f"District_{i+1:02d}", f"{100+i}", f"{90+i}"]
        for i in range(15)
    ]
    b = _make_csv(dist_cols, dist_rows)
    df, warns, h, cm, sf = validate_csv_upload(b, "dist.csv")
    assert "station_id" in df.columns,  f"station_id missing: {list(df.columns)}"
    assert "serial_no" in df.columns,   f"serial_no missing: {list(df.columns)}"
    # station_id must NOT be numeric (should be district name)
    import pandas as pd
    is_numeric = pd.to_numeric(df["station_id"], errors="coerce").notna().mean() > 0.8
    assert not is_numeric, f"station_id is still numeric: {df['station_id'].head().tolist()}"
    # serial_no must be numeric
    is_serial_numeric = pd.to_numeric(df["serial_no"], errors="coerce").notna().mean() > 0.8
    assert is_serial_numeric, f"serial_no should be numeric: {df['serial_no'].head().tolist()}"
    # synthesised_fields must note that station_id was derived
    assert "station_id" in sf, f"synthesised_fields missing station_id key: {sf}"
    assert "district" in sf["station_id"].lower(), f"synthesised_fields note wrong: {sf}"
    print(f"  Sl No→serial_no, station_id=district: OK — {df['station_id'].iloc[0]}")
check("A12: Sl No→serial_no; station_id is district name, not serial", a12)


# A13: synthesised_fields appears in audit trail JSON for district-level upload
def a13():
    """synthesised_fields (from district-level file) must appear in audit JSON."""
    from guardrails import generate_audit_trail
    dist_cols = ["Sl No", "District", "Annual Normal", "Annual Actual"]
    dist_rows = [
        [str(i+1), f"Dist_{i+1}", f"{100+i}", f"{90+i}"]
        for i in range(15)
    ]
    b = _make_csv(dist_cols, dist_rows)
    df, warns, h, cm, sf = validate_csv_upload(b, "dist_audit.csv")
    assert sf, f"synthesised_fields is empty for district-level file: {sf}"
    # Build an audit trail with synthesised_fields
    trail = generate_audit_trail(h, h, [], [], {}, synthesised_fields=sf)
    prov = trail["provenance"]
    assert "synthesised_fields" in prov, f"synthesised_fields missing from provenance: {prov}"
    assert prov["synthesised_fields"] == sf, (
        f"synthesised_fields mismatch: {prov['synthesised_fields']} vs {sf}"
    )
    print(f"  synthesised_fields in audit JSON: OK — fields={list(sf.keys())}")
check("A13: synthesised_fields in audit trail JSON", a13)


# A14: N/A labelling flag — _is_aggregate_data() returns True iff 'date' in synthesised_fields
def a14():
    """
    _is_aggregate_data() must return True for district-level files and False for
    daily station files.  We test the underlying condition directly (without
    importing app.py which requires a Streamlit runtime).
    """
    # Simulate what app.py does: check synthesised_fields.get("date")
    # District-level upload → has 'date' in synthesised_fields
    dist_cols = ["Sl No", "District", "Annual Normal", "Annual Actual"]
    dist_rows = [
        [str(i+1), f"Dist_{i+1}", f"{100+i}", f"{90+i}"]
        for i in range(15)
    ]
    b_dist = _make_csv(dist_cols, dist_rows)
    _df_d, _w_d, _h_d, _cm_d, sf_dist = validate_csv_upload(b_dist, "dist_na.csv")
    assert "date" in sf_dist, (
        f"district-level upload must set synthesised_fields['date']: {sf_dist}"
    )
    # Simulated _is_aggregate_data() — True when 'date' in sf
    assert bool(sf_dist.get("date")) is True, "N/A flag should be True for aggregate data"

    # Daily station upload → no 'date' in synthesised_fields
    _df_s, _w_s, _h_s, _cm_s, sf_station = validate_csv_upload(
        _make_csv(STD_COLS, ROWS15), "station.csv"
    )
    assert "date" not in sf_station, (
        f"daily file must NOT set synthesised_fields['date']: {sf_station}"
    )
    assert bool(sf_station.get("date")) is False, "N/A flag should be False for daily data"

    print(f"  N/A labelling flag: aggregate=True ✓, daily=False ✓")
check("A14: N/A flag logic — aggregate data vs daily station data", a14)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A2 — District-Level Consistency Check Tests
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  SECTION A2 — District-Level Check Tests")
print("═"*60)

from agents import (
    run_district_checks,
    _check_departure_mismatch,
    _check_seasonal_sum_mismatch,
    _check_negative_or_missing,
    _check_extreme_departure,
)


def _make_dist_df(rows):
    """Build a minimal district DataFrame for testing."""
    return pd.DataFrame(rows)


# A15: Departure mismatch — detects where filed dep differs from (A-N)/N*100 by >1 pp
def a15():
    df = _make_dist_df([
        # Normal=100, Actual=125 → recomputed=25.0, filed=20 → diff=5 → flag
        {"district": "TestDist1", "annual normal": 100, "annual actual": 125,
         "annual departure (%)": 20, "rainfall_mm": 125},
        # Normal=100, Actual=90 → recomputed=-10.0, filed=-10 → diff=0 → no flag
        {"district": "TestDist2", "annual normal": 100, "annual actual": 90,
         "annual departure (%)": -10, "rainfall_mm": 90},
        # Normal=200, Actual=260 → recomputed=30.0, filed=30 → diff=0 → no flag
        {"district": "TestDist3", "annual normal": 200, "annual actual": 260,
         "annual departure (%)": 30, "rainfall_mm": 260},
    ])
    result = _check_departure_mismatch(df)
    assert result["count"] == 1, f"Expected 1 mismatch, got {result['count']}: {result['details']}"
    assert result["details"][0]["district"] == "TestDist1"
    assert result["details"][0]["recomputed_value"] == 25.0
    print(f"  Departure mismatch: 1 flagged ✓, {result['details'][0]['reason'][:60]}")
check("A15: Departure mismatch check (1 flagged, 2 clean)", a15)


# A16: Seasonal sum mismatch — flags when Pre+SWM+NEM differ from Annual by >2%
def a16():
    df = _make_dist_df([
        # 100 + 400 + 50 = 550 vs annual=600 → diff=8.3% → flag
        {"district": "DistA", "annual actual": 600,
         "pre monsoon actual": 100, "swm actual": 400, "nem actual": 50,
         "rainfall_mm": 600},
        # 100 + 400 + 100 = 600 vs annual=600 → diff=0% → clean
        {"district": "DistB", "annual actual": 600,
         "pre monsoon actual": 100, "swm actual": 400, "nem actual": 100,
         "rainfall_mm": 600},
    ])
    result = _check_seasonal_sum_mismatch(df)
    assert not result.get("skipped"), f"Should not skip: {result}"
    assert result["count"] == 1, f"Expected 1, got {result['count']}: {result['details']}"
    assert result["details"][0]["district"] == "DistA"
    print(f"  Seasonal sum mismatch: 1 flagged ✓")
check("A16: Seasonal sum mismatch check (1 flagged, 1 clean)", a16)


# A17: Negative or missing values in annual actual
def a17():
    df = _make_dist_df([
        {"district": "DistNeg",   "rainfall_mm": -5.0},
        {"district": "DistBlank", "rainfall_mm": None},
        {"district": "DistOk",    "rainfall_mm": 850.0},
    ])
    result = _check_negative_or_missing(df)
    assert result["count"] == 2, f"Expected 2, got {result['count']}: {result['details']}"
    flagged = {r["district"] for r in result["details"]}
    assert flagged == {"DistNeg", "DistBlank"}, f"Wrong districts flagged: {flagged}"
    print(f"  Negative/missing: 2 flagged ({flagged}) ✓")
check("A17: Negative/missing rainfall check (2 flagged, 1 clean)", a17)


# A18: Extreme departure — flags |dep| > 60%
def a18():
    df = _make_dist_df([
        {"district": "DistHigh",  "annual departure (%)": 85.0},   # +85 → flag
        {"district": "DistLow",   "annual departure (%)": -70.0},  # -70 → flag
        {"district": "DistOk",    "annual departure (%)": 25.0},   # +25 → clean
        {"district": "DistEdge",  "annual departure (%)": -60.0},  # exactly -60 → clean (not >60)
    ])
    result = _check_extreme_departure(df)
    assert result["count"] == 2, f"Expected 2, got {result['count']}: {result['details']}"
    flagged = {r["district"] for r in result["details"]}
    assert flagged == {"DistHigh", "DistLow"}, f"Wrong districts: {flagged}"
    print(f"  Extreme departure: 2 flagged ({flagged}) ✓")
check("A18: Extreme departure check (2 flagged, 2 clean)", a18)


# A19: Benchmark path gives exactly 15 defects (district checks do NOT affect benchmark)
def a19():
    from sample_data import generate_benchmark_data
    from agents import run_quality_agent
    raw_df, labels = generate_benchmark_data()
    # Run WITHOUT is_aggregate — simulates benchmark path
    qr = run_quality_agent(raw_df, is_aggregate=False)
    assert qr["district_checks"] is None, "district_checks must be None for non-aggregate"
    assert qr["total_defects_found"] == 15, f"Expected 15, got {qr['total_defects_found']}"
    checks = qr["checks"]
    assert checks["duplicates"]["count"] == 3
    assert checks["negatives"]["count"] == 3
    assert checks["date_gaps"]["count"] == 3
    assert checks["conflicts"]["count"] == 3
    assert checks["outliers"]["sensor_error_count"] == 3
    assert checks["outliers"]["legitimate_extreme_count"] == 2
    print(f"  Benchmark: 15 defects, district_checks=None ✓")
check("A19: Benchmark path unchanged — 15 defects, district_checks=None", a19)


# ════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SECTION B — Full Integration Tests (12 existing checks)")
print("═"*60)

from sample_data import generate_benchmark_data
from agents import (
    run_schema_agent, run_quality_agent, run_audit_agent,
    apply_approved_fixes, get_perf_summary, translate_audit_summary,
)

def b1():
    raw_df, labels = generate_benchmark_data()
    assert len(raw_df) > 1000
    assert raw_df["station_id"].nunique() == 4
    real_defects = labels[labels["defect_type"] != "LEGITIMATE_EXTREME"]
    assert len(real_defects) == 15
    extremes = labels[labels["defect_type"] == "LEGITIMATE_EXTREME"]
    assert len(extremes) == 2
    print(f"  Rows: {len(raw_df)}, Defects: {len(real_defects)}, Extremes: {len(extremes)}")
    return raw_df, labels
result = check("B1: Benchmark data generation", b1)
raw_df, defect_labels = result if result else (None, None)

def b2():
    csv_bytes = raw_df.to_csv(index=False).encode()
    df_val, warns, h, cm, sf = validate_csv_upload(csv_bytes, "test.csv")
    assert len(df_val) == len(raw_df)
    print(f"  Input guardrail: {len(df_val)} rows validated")
check("B2: Input guardrail on benchmark CSV", b2)

def b3():
    g = ImmutabilityGuardrail()
    cs = g.register_raw(raw_df)
    assert len(cs) == 64
    unchanged, _ = g.verify_raw_unchanged(raw_df)
    assert unchanged
    mutated = raw_df.copy()
    mutated.iloc[0, mutated.columns.get_loc("rainfall_mm")] = 999999
    unchanged2, _ = g.verify_raw_unchanged(mutated)
    assert not unchanged2
    print(f"  Immutability: checksum={cs[:16]}..., mutation detected ✓")
check("B3: Immutability guardrail", b3)

def b4():
    extremes = ExtremePreservationGuardrail.find_valid_extremes(raw_df)
    assert len(extremes) == 2
    cleaned = raw_df.copy()
    mask = (cleaned["station_id"] == "BLR_CENTRAL") & (cleaned["date"] == "2024-07-15")
    cleaned = cleaned[~mask].reset_index(drop=True)
    cleaned_restored, msgs = ExtremePreservationGuardrail.enforce_on_cleaned(raw_df, cleaned)
    assert any("RESTORED" in m for m in msgs)
    print(f"  Extreme preservation: {len(extremes)} extremes, enforcement works ✓")
check("B4: Extreme preservation guardrail", b4)

def b5():
    h = compute_sha256(raw_df)
    assert len(h) == 64
    trail = generate_audit_trail(h, h, [], [], {})
    assert trail["provenance"]["raw_sha256"] == h
    print(f"  Audit trail: SHA256={h[:16]}... ✓")
check("B5: Audit trail guardrail", b5)

def b6():
    result = run_schema_agent(raw_df)
    assert result["date_format_valid"] == True
    assert result["column_mapping"]["station_id"] == "MATCHED"
    print(f"  Agent 1: status={result['status']}, model={result['model_used']}")
    return result
schema_result = check("B6: Agent 1 — SchemaProvenanceAgent", b6)

def b7():
    result = run_quality_agent(raw_df)
    checks = result["checks"]
    assert checks["duplicates"]["count"] == 3
    assert checks["negatives"]["count"] == 3
    assert checks["date_gaps"]["count"] == 3
    assert checks["conflicts"]["count"] == 3
    assert checks["outliers"]["sensor_error_count"] == 3
    assert checks["outliers"]["legitimate_extreme_count"] == 2
    assert result["total_defects_found"] == 15
    print(f"  Agent 2: 15/15 defects, model={result['model_used']} ✓")
    return result
quality_result = check("B7: Agent 2 — QualityInvestigationAgent", b7)

def b8():
    result = run_audit_agent(raw_df, quality_result)
    dtypes = {e["defect_type"] for e in result["audit_log"]}
    for t in ("NEGATIVE_VALUE","DUPLICATE_RECORD","DATE_GAP","CONFLICTING_RECORD",
              "SENSOR_ERROR","LEGITIMATE_EXTREME"):
        assert t in dtypes, f"Missing: {t}"
    print(f"  Agent 3: {len(result['audit_log'])} entries, model={result['model_used']} ✓")
    return result
audit_result = check("B8: Agent 3 — AuditReportAgent", b8)

def b9():
    log = audit_result["audit_log"]
    for e in log:
        if e["defect_type"] != "LEGITIMATE_EXTREME":
            e["user_decision"] = "Approve"
        else:
            e["user_decision"] = "Retain"
    cleaned_df, applied = apply_approved_fixes(raw_df, log)
    assert cleaned_df is not raw_df
    g = ImmutabilityGuardrail(); g.register_raw(raw_df)
    unchanged, _ = g.verify_raw_unchanged(raw_df)
    assert unchanged, "RAW DATA WAS MUTATED"
    raw_h = compute_sha256(raw_df)
    cln_h = compute_sha256(cleaned_df)
    assert raw_h != cln_h
    guard = ExtremePreservationGuardrail()
    cleaned_df, _ = guard.enforce_on_cleaned(raw_df, cleaned_df)
    assert len(guard.find_valid_extremes(cleaned_df)) == 2
    print(f"  Apply fixes: {len(applied)} applied, raw immutable ✓, extremes preserved ✓")
    return cleaned_df
check("B9: Apply approved fixes + immutability", b9)

def b10():
    en, m1 = translate_audit_summary("Test.", "English")
    assert en == "Test." and m1 == "none"
    kn, m2 = translate_audit_summary("15 defects detected.", "ಕನ್ನಡ (Kannada)")
    assert len(kn) > 5
    print(f"  Bilingual: English passthrough ✓, Kannada model={m2}")
check("B10: Bilingual Kannada/English toggle", b10)

def b11():
    gt = defect_labels[defect_labels["defect_type"] != "LEGITIMATE_EXTREME"]
    checks = quality_result["checks"]
    detected = sum([
        checks["duplicates"]["count"], checks["negatives"]["count"],
        checks["date_gaps"]["count"], checks["conflicts"]["count"],
        checks["outliers"]["sensor_error_count"],
    ])
    tp = min(detected, len(gt))
    fp = max(0, detected - len(gt))
    fn = max(0, len(gt) - detected)
    p  = tp/(tp+fp) if (tp+fp) else 0.0
    r  = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    assert f1 == 1.0, f"F1={f1}"
    print(f"  F1: Precision={p:.0%}, Recall={r:.0%}, F1={f1:.0%} ✓")
check("B11: F1 benchmark (perfect score)", b11)

def b12():
    perf = get_perf_summary()
    assert perf.get("calls", 0) > 0
    print(f"  Perf: {perf['calls']} calls, {perf['total_tokens']} tokens, "
          f"${perf['total_cost_usd']:.4f}, models={perf['models_used']}")
check("B12: LLM performance metrics", b12)

# ════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print(f"  TOTAL: {len(PASS)} PASSED | {len(FAIL)} FAILED")
print("═"*60)
for p in PASS:
    print(f"  ✅ {p}")
for name, err in FAIL:
    print(f"  ❌ {name}: {err[:100]}")

if FAIL:
    print("\nSome tests FAILED.")
    sys.exit(1)
else:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
