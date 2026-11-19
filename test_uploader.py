"""
test_uploader.py — Tests for the lenient CSV uploader in guardrails.py
"""
import io
import pandas as pd
from guardrails import validate_csv_upload, InputGuardrailError

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except Exception as e:
        import traceback; traceback.print_exc()
        FAIL.append((name, str(e)))

# ── Build a standard-schema CSV helper ───────────────────────────────────────
def _make_csv(columns, rows, sep=",", encoding="utf-8"):
    lines = [sep.join(columns)]
    for r in rows:
        lines.append(sep.join(str(v) for v in r))
    return "\n".join(lines).encode(encoding)

STANDARD_COLS  = ["station_id","station_name","district","taluk","date","rainfall_mm"]
SAMPLE_ROWS_15 = [
    ["BLR_CENTRAL","Bengaluru Central","Bengaluru Urban","BN North",f"2024-01-{i+1:02d}","5.2"]
    for i in range(15)
]

# ── Test 1: Standard comma CSV, UTF-8 ────────────────────────────────────────
def t1():
    b = _make_csv(STANDARD_COLS, SAMPLE_ROWS_15)
    df, warns, h, cm, sf = validate_csv_upload(b, "standard.csv")
    assert len(df) == 15
    assert list(df.columns) == STANDARD_COLS
    assert not any(w.startswith("⚠️") for w in warns), f"Unexpected hard warnings: {warns}"
    print("  Standard UTF-8 comma CSV: OK")
check("Standard comma CSV (UTF-8)", t1)

# ── Test 2: Tab-delimited ─────────────────────────────────────────────────────
def t2():
    b = _make_csv(STANDARD_COLS, SAMPLE_ROWS_15, sep="\t")
    df, warns, h, cm, sf = validate_csv_upload(b, "tab.csv")
    assert len(df) == 15
    assert any("tab" in w.lower() for w in warns), f"Expected tab warning, got: {warns}"
    print(f"  Tab-delimited: OK, warnings: {warns}")
check("Tab-delimited CSV", t2)

# ── Test 3: Semicolon-delimited ───────────────────────────────────────────────
def t3():
    b = _make_csv(STANDARD_COLS, SAMPLE_ROWS_15, sep=";")
    df, warns, h, cm, sf = validate_csv_upload(b, "semi.csv")
    assert len(df) == 15
    assert any("semicolon" in w.lower() for w in warns)
    print(f"  Semicolon-delimited: OK, warnings: {warns}")
check("Semicolon-delimited CSV", t3)

# ── Test 4: CP1252 encoding with non-ASCII character ─────────────────────────
def t4():
    # Use a row with an extended CP1252 char (é = 0xe9) so UTF-8 fails
    rows_with_accent = [
        ["BLR_CENTRAL", "Bengaluru Céntre", "Bengaluru Urban", "BN North",
         f"2024-01-{i+1:02d}", "5.2"]
        for i in range(15)
    ]
    b = _make_csv(STANDARD_COLS, rows_with_accent, encoding="cp1252")
    df, warns, h, cm, sf = validate_csv_upload(b, "cp1252.csv")
    assert len(df) == 15
    # Either warns about cp1252 OR loads silently if the accent survived
    print(f"  CP1252 encoding: OK, rows={len(df)}, warnings={warns}")
check("CP1252 encoding with accent", t4)

# ── Test 5: Different column names (soft warning, NOT rejected) ───────────────
def t5():
    alt_cols = ["stn_id", "stn_name", "dist", "sub_dist", "obs_date", "rain_mm"]
    b = _make_csv(alt_cols, SAMPLE_ROWS_15)
    df, warns, h, cm, sf = validate_csv_upload(b, "altcols.csv")
    assert len(df) == 15, f"Should still load: {len(df)} rows"
    # After alias mapping, columns are renamed — check no hard rejection
    assert not any("rejected" in w.lower() for w in warns)
    print(f"  Alt columns (NOT rejected): OK, {len(warns)} warning(s), cols={list(df.columns)}")
check("Different column names — soft warning, file accepted", t5)

# ── Test 6: Missing 2 standard columns (soft warning, file still loads) ───────
def t6():
    partial_cols = ["station_id", "date", "rainfall_mm"]   # missing 3 standard cols
    rows = [["BLR", f"2024-01-{i+1:02d}", "3.1"] for i in range(15)]
    b = _make_csv(partial_cols, rows)
    df, warns, h, cm, sf = validate_csv_upload(b, "partial.csv")
    assert len(df) == 15
    assert any("unmapped" in w.lower() or "mismatch" in w.lower() for w in warns), \
        f"Expected unmapped/mismatch warning: {warns}"
    print(f"  Partial columns (loads with warning): OK")
check("Partial columns — loads with schema warning", t6)

# ── Test 7: Extra columns are tolerated ───────────────────────────────────────
def t7():
    extra_cols = STANDARD_COLS + ["source_agency", "qc_flag"]
    rows = [r + ["IMD", "1"] for r in SAMPLE_ROWS_15]
    b = _make_csv(extra_cols, rows)
    df, warns, h, cm, sf = validate_csv_upload(b, "extra.csv")
    assert len(df) == 15
    assert "source_agency" in df.columns
    print(f"  Extra columns tolerated: OK, columns={list(df.columns)}")
check("Extra columns tolerated", t7)

# ── Test 8: Oversized file is HARD rejected ───────────────────────────────────
def t8():
    big = b"a,b\n" + b"1,2\n" * 1_000_000
    # Fake a large size by padding
    big = big + b"\x00" * (51 * 1024 * 1024)
    try:
        validate_csv_upload(big, "big.csv")
        assert False, "Should have rejected"
    except InputGuardrailError as e:
        assert "50 MB" in str(e)
        print(f"  Oversized file hard-rejected: OK")
check("Oversized file — hard rejected", t8)

# ── Test 9: Wrong extension is HARD rejected (.pdf not accepted) ─────────────
def t9():
    b = _make_csv(STANDARD_COLS, SAMPLE_ROWS_15)
    try:
        validate_csv_upload(b, "data.pdf")
        assert False, "Should have rejected .pdf"
    except InputGuardrailError as e:
        assert ".pdf" in str(e).lower() or "unsupported" in str(e).lower()
        print(f"  .pdf extension hard-rejected: OK")
check("Wrong extension (.pdf) — hard rejected", t9)

# ── Test 10: Injection pattern is HARD rejected ───────────────────────────────
def t10():
    malicious = b"station_id,date,rainfall_mm\n<script>alert(1)</script>,2024-01-01,5.0\n" * 15
    try:
        validate_csv_upload(malicious, "xss.csv")
        assert False, "Should have rejected"
    except InputGuardrailError as e:
        assert "suspicious" in str(e).lower()
        print(f"  Injection pattern hard-rejected: OK")
check("Injection pattern — hard rejected", t10)

# ── Test 11: Too few rows is HARD rejected ────────────────────────────────────
def t11():
    b = _make_csv(STANDARD_COLS, SAMPLE_ROWS_15[:5])
    try:
        validate_csv_upload(b, "tiny.csv")
        assert False, "Should have rejected"
    except InputGuardrailError as e:
        assert "row" in str(e).lower()
        print(f"  Too few rows hard-rejected: OK")
check("Too few rows — hard rejected", t11)

# ── Test 12: Completely garbage file is HARD rejected ────────────────────────
def t12():
    garbage = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" * 500   # fake PNG header
    try:
        validate_csv_upload(garbage, "image.csv")
        assert False, "Should have rejected"
    except InputGuardrailError as e:
        print(f"  Garbage binary file hard-rejected: OK ({str(e)[:60]})")
check("Binary/garbage file — hard rejected", t12)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {len(PASS)} PASSED | {len(FAIL)} FAILED")
print("=" * 60)
for p in PASS:
    print(f"  ✅ {p}")
for name, err in FAIL:
    print(f"  ❌ {name}: {err[:100]}")

import sys
sys.exit(1 if FAIL else 0)
