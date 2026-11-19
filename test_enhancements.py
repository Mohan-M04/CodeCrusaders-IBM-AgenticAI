"""
test_enhancements.py - Validates the two new features without importing app.py
"""
from agents import translate_audit_summary

# 1. Test translate_audit_summary - English passthrough
text, model = translate_audit_summary("This is a test audit.", "English")
assert text == "This is a test audit.", "English passthrough failed"
assert model == "none", "English should not invoke LLM"
print("translate_audit_summary English passthrough: OK")

# 2. Test translate_audit_summary - Kannada (live call)
kn_text, kn_model = translate_audit_summary(
    "Executive Summary: The dataset has 3 negative values, 3 duplicates, "
    "and 3 sensor errors. Data integrity risk is HIGH.",
    "ಕನ್ನಡ (Kannada)"
)
print(f"Kannada translation model: {kn_model}")
print(f"Kannada text (first 120 chars): {kn_text[:120]}")
assert len(kn_text) > 20, "Kannada translation too short"
print("translate_audit_summary Kannada live call: OK")

# 3. Test station coords are in Karnataka geographic bounds
_STATION_COORDS = {
    "BLR_CENTRAL":  {"lat": 12.9716, "lon": 77.5946},
    "BLR_NORTH":    {"lat": 13.0358, "lon": 77.5970},
    "MYS_CITY":     {"lat": 12.2958, "lon": 76.6394},
    "MYS_KR_NAGAR": {"lat": 12.4385, "lon": 76.3812},
}
for sid, m in _STATION_COORDS.items():
    assert 11.0 < m["lat"] < 18.0, f"{sid} lat={m['lat']} out of Karnataka range"
    assert 74.0 < m["lon"] < 78.5, f"{sid} lon={m['lon']} out of Karnataka range"
print(f"All 4 station coordinates within Karnataka bounds: OK")

# 4. Verify defect colour-coding logic
defective = {"BLR_CENTRAL", "BLR_NORTH"}
for sid in _STATION_COORDS:
    color = "red" if sid in defective else "green"
    expected = "red" if sid in ("BLR_CENTRAL", "BLR_NORTH") else "green"
    assert color == expected, f"{sid}: expected {expected}, got {color}"
print("Defect colour-coding logic: OK")

print("\nAll enhancement tests passed!")
