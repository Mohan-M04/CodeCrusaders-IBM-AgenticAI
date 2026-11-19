"""
print_district_findings.py
Run district-level consistency checks on the real ka_dist_rainfall_2025.csv
and print all findings.
"""
from guardrails import validate_csv_upload
from agents import run_district_checks

with open(r"C:\Users\mohan\OneDrive\Desktop\ka_dist_rainfall_2025.csv", "rb") as f:
    raw = f.read()

df, warns, h, cm, sf = validate_csv_upload(raw, "ka_dist_rainfall_2025.csv")
print(f"Loaded {len(df)} district rows (STATE row removed)")
print()

result = run_district_checks(df)
checks = result["checks"]

print("=" * 68)
print("  DISTRICT-LEVEL CONSISTENCY CHECK RESULTS  —  ka_dist_rainfall_2025")
print("=" * 68)

# (a) Departure mismatch
dm = checks["departure_mismatch"]
print(f"\n(a) DEPARTURE MISMATCH  —  {dm['count']} inconsistencies")
print(f"    Rule: filed 'Annual Departure (%)' must equal (Actual-Normal)/Normal*100 within 1 pp")
if dm.get("skipped"):
    print(f"    SKIPPED: {dm['skip_reason']}")
elif dm["count"] == 0:
    print("    ✅ All departure values match recomputed formula.")
else:
    print(f"    {'District':<24}  {'Filed (%)':>9}  {'Recomputed (%)':>14}  {'Diff (pp)':>9}")
    print("    " + "-" * 62)
    for r in dm["details"]:
        print(
            f"    {r['district']:<24}  {r['filed_value']:>9.1f}  "
            f"{r['recomputed_value']:>14.1f}  {r['diff_pct_points']:>9.1f}"
        )

# (b) Seasonal sum mismatch
ss = checks["seasonal_sum_mismatch"]
print(f"\n(b) SEASONAL SUM MISMATCH  —  {ss['count']} inconsistencies")
print(f"    Rule: Pre Monsoon Actual + SWM Actual + NEM Actual must equal Annual Actual within 2%")
if ss.get("skipped"):
    print(f"    INFO: {ss['skip_reason']}")
elif ss["count"] == 0:
    print("    ✅ All seasonal sums match Annual Actual within 2% tolerance.")
else:
    print(f"    {'District':<24}  {'Annual Actual':>13}  {'Season Sum':>10}  {'Diff (%)':>8}")
    print("    " + "-" * 62)
    for r in ss["details"]:
        print(
            f"    {r['district']:<24}  {r['filed_value']:>13.1f}  "
            f"{r['recomputed_value']:>10.1f}  {r['diff_pct']:>8.1f}"
        )

# (c) Negative / missing
nm = checks["negative_or_missing"]
print(f"\n(c) NEGATIVE / MISSING VALUES  —  {nm['count']} found")
if nm.get("skipped"):
    print(f"    SKIPPED: {nm['skip_reason']}")
elif nm["count"] == 0:
    print("    ✅ No negative or missing Annual Actual values.")
else:
    for r in nm["details"]:
        print(f"    {r['district']:<24}  value={r['filed_value']}  —  {r['reason']}")

# (d) Extreme departure
ed = checks["extreme_departure"]
print(f"\n(d) EXTREME DEPARTURE  —  {ed['count']} to verify  (DO NOT AUTO-DELETE)")
print(f"    Rule: |Annual Departure (%)| > 60% — verify against KSNDMC source")
if ed.get("skipped"):
    print(f"    SKIPPED: {ed['skip_reason']}")
elif ed["count"] == 0:
    print("    ✅ No extreme departure values (all within ±60%).")
else:
    print(f"    {'District':<24}  {'Departure (%)':>13}  Note")
    print("    " + "-" * 62)
    for r in ed["details"]:
        note = "drought risk" if r["filed_value"] < 0 else "flood/excess risk"
        print(f"    {r['district']:<24}  {r['filed_value']:>13.1f}  {note}")

print()
print(f"  Total actionable inconsistencies (a + b + c): {result['total_inconsistencies']}")
print(f"  Extreme departures to verify (d):             {result['extreme_departure_count']}")
print("=" * 68)
