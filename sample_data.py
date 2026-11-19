"""
sample_data.py
Karnataka Rainfall Benchmark Generator for Rain Check Karnataka.
Generates realistic 2024 daily rainfall data for 4 stations across 2 districts.
Injects exactly 15 ground-truth labelled defects into a benchmark copy.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ─── Station Metadata ────────────────────────────────────────────────────────
STATIONS = [
    {"station_id": "BLR_CENTRAL", "station_name": "Bengaluru Central AWS",
     "district": "Bengaluru Urban", "taluk": "Bengaluru North"},
    {"station_id": "BLR_NORTH",   "station_name": "Bengaluru North Observatory",
     "district": "Bengaluru Urban", "taluk": "Yelahanka"},
    {"station_id": "MYS_CITY",    "station_name": "Mysuru City AWS",
     "district": "Mysuru",          "taluk": "Mysuru"},
    {"station_id": "MYS_KR_NAGAR","station_name": "Mysuru K.R. Nagar Station",
     "district": "Mysuru",          "taluk": "Krishnarajasagara"},
]

# Monthly rainfall multipliers (Karnataka seasonal pattern)
# Jan-Feb: dry; Mar-May: pre-monsoon; Jun-Sep: SW monsoon; Oct-Nov: NE monsoon; Dec: dry
MONTHLY_WEIGHTS = {
    1: 0.05, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.25,
    6: 0.80, 7: 0.90, 8: 0.85, 9: 0.75, 10: 0.50,
    11: 0.20, 12: 0.05
}

ANNUAL_RAINFALL_MM = {
    "BLR_CENTRAL":  970,
    "BLR_NORTH":    920,
    "MYS_CITY":    780,
    "MYS_KR_NAGAR": 710,
}

np.random.seed(42)


def _generate_daily_series(station_id: str, year: int = 2024) -> pd.DataFrame:
    """Generate a plausible daily rainfall series for one station."""
    start = date(year, 1, 1)
    end   = date(year, 12, 31)
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    annual = ANNUAL_RAINFALL_MM[station_id]
    rows = []
    for d in dates:
        p_rain = MONTHLY_WEIGHTS[d.month]
        # Rain occurs with probability proportional to monthly weight
        if np.random.random() < p_rain:
            # Gamma distribution gives realistic right-skewed rain amounts
            amount = round(float(np.random.gamma(shape=2.0, scale=annual * p_rain / 60)), 1)
            amount = min(amount, 150.0)
        else:
            amount = 0.0
        rows.append(amount)

    meta = next(s for s in STATIONS if s["station_id"] == station_id)
    df = pd.DataFrame({
        "station_id":   station_id,
        "station_name": meta["station_name"],
        "district":     meta["district"],
        "taluk":        meta["taluk"],
        "date":         [str(d) for d in dates],
        "rainfall_mm":  rows,
    })
    return df


def _inject_defects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject exactly 15 ground-truth defects into a copy of df.
    Returns (defective_df, defect_labels_df).
    Preserves 1–2 legitimate extreme monsoon events (≥185 mm).
    """
    dfc = df.copy().reset_index(drop=True)
    # Convert rainfall_mm to object dtype so we can inject mixed-type corrupt values
    dfc["rainfall_mm"] = dfc["rainfall_mm"].astype(object)
    defects = []

    # ── Inject 1 legitimate extreme rain event BEFORE defect injection ──────
    # Heavy rain day on BLR_CENTRAL during peak monsoon – must be PRESERVED
    extreme_mask = (dfc["station_id"] == "BLR_CENTRAL") & (dfc["date"] == "2024-07-15")
    if extreme_mask.sum() > 0:
        idx_extreme = dfc.index[extreme_mask][0]
        dfc.at[idx_extreme, "rainfall_mm"] = 185.0
        defects.append({"index": idx_extreme, "defect_type": "LEGITIMATE_EXTREME",
                        "description": "185.0 mm – valid SW monsoon heavy rain; MUST BE PRESERVED"})

    # Second legitimate extreme – MYS_CITY heavy day
    extreme_mask2 = (dfc["station_id"] == "MYS_CITY") & (dfc["date"] == "2024-08-05")
    if extreme_mask2.sum() > 0:
        idx_extreme2 = dfc.index[extreme_mask2][0]
        dfc.at[idx_extreme2, "rainfall_mm"] = 192.0
        defects.append({"index": idx_extreme2, "defect_type": "LEGITIMATE_EXTREME",
                        "description": "192.0 mm – valid SW monsoon heavy rain; MUST BE PRESERVED"})

    # ── DEFECT 1-3: Negative rainfall values ─────────────────────────────────
    neg_targets = [
        ("BLR_NORTH",    "2024-03-10", -4.5),
        ("MYS_CITY",     "2024-05-22", -1.2),
        ("MYS_KR_NAGAR", "2024-11-03", -7.8),
    ]
    for sid, dt, val in neg_targets:
        mask = (dfc["station_id"] == sid) & (dfc["date"] == dt)
        if mask.sum() > 0:
            idx = dfc.index[mask][0]
            dfc.at[idx, "rainfall_mm"] = val
            defects.append({"index": idx, "defect_type": "NEGATIVE_VALUE",
                            "description": f"Negative rainfall {val} mm at {sid} on {dt}"})

    # ── DEFECT 4-6: Duplicate records (exact copy rows) ──────────────────────
    dup_targets = [
        ("BLR_CENTRAL", "2024-04-18"),
        ("BLR_NORTH",   "2024-06-30"),
        ("MYS_CITY",    "2024-09-14"),
    ]
    dup_rows = []
    for sid, dt in dup_targets:
        mask = (dfc["station_id"] == sid) & (dfc["date"] == dt)
        if mask.sum() > 0:
            orig_idx = dfc.index[mask][0]
            dup_row = dfc.loc[orig_idx].copy()
            dup_rows.append(dup_row)
            defects.append({"index": -1, "defect_type": "DUPLICATE_RECORD",
                            "description": f"Duplicate row for {sid} on {dt}"})

    if dup_rows:
        dup_df = pd.DataFrame(dup_rows)
        dfc = pd.concat([dfc, dup_df], ignore_index=True)

    # ── DEFECT 7-9: Date sequence gaps (remove rows to create gaps) ──────────
    gap_targets = [
        ("BLR_CENTRAL", "2024-02-14"),
        ("BLR_NORTH",   "2024-07-04"),
        ("MYS_KR_NAGAR","2024-10-12"),
    ]
    for sid, dt in gap_targets:
        mask = (dfc["station_id"] == sid) & (dfc["date"] == dt)
        if mask.sum() > 0:
            idx = dfc.index[mask][0]
            dfc = dfc.drop(index=idx).reset_index(drop=True)
            defects.append({"index": -1, "defect_type": "DATE_GAP",
                            "description": f"Missing observation on {dt} for {sid}"})

    # ── DEFECT 10-12: Conflicting records (same station+date, different value) ─
    conflict_targets = [
        ("BLR_CENTRAL",  "2024-06-20", 55.5),
        ("MYS_CITY",     "2024-08-18", 33.3),
        ("MYS_KR_NAGAR", "2024-07-22", 77.7),
    ]
    conflict_rows = []
    for sid, dt, alt_val in conflict_targets:
        mask = (dfc["station_id"] == sid) & (dfc["date"] == dt)
        if mask.sum() > 0:
            orig_idx = dfc.index[mask][0]
            conflict_row = dfc.loc[orig_idx].copy()
            conflict_row["rainfall_mm"] = alt_val
            conflict_rows.append(conflict_row)
            defects.append({"index": -1, "defect_type": "CONFLICTING_RECORD",
                            "description": f"Conflicting value {alt_val} mm vs original for {sid} on {dt}"})

    if conflict_rows:
        c_df = pd.DataFrame(conflict_rows)
        dfc = pd.concat([dfc, c_df], ignore_index=True)

    # ── DEFECT 13-15: Corrupted / extreme invalid placeholders ───────────────
    corrupt_targets = [
        ("BLR_NORTH",    "2024-01-25", 9999.0),
        ("BLR_CENTRAL",  "2024-12-05", "N/A"),
        ("MYS_KR_NAGAR", "2024-05-30", -9999.0),
    ]
    for sid, dt, val in corrupt_targets:
        mask = (dfc["station_id"] == sid) & (dfc["date"] == dt)
        if mask.sum() > 0:
            idx = dfc.index[mask][0]
            dfc.at[idx, "rainfall_mm"] = val
            defects.append({"index": idx, "defect_type": "CORRUPTED_VALUE",
                            "description": f"Invalid placeholder '{val}' at {sid} on {dt}"})

    defect_labels_df = pd.DataFrame(defects)
    # Sort dfc for readability
    dfc = dfc.sort_values(["station_id", "date"]).reset_index(drop=True)
    return dfc, defect_labels_df


def generate_benchmark_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Karnataka 2024 benchmark dataset with 15 injected defects.

    Returns
    -------
    raw_df : pd.DataFrame
        Defective benchmark dataset (simulates 'uploaded' raw data).
    defect_labels_df : pd.DataFrame
        Ground-truth defect labels for evaluation (Precision/Recall/F1).
    """
    frames = [_generate_daily_series(s["station_id"]) for s in STATIONS]
    clean_df = pd.concat(frames, ignore_index=True)
    clean_df = clean_df.sort_values(["station_id", "date"]).reset_index(drop=True)

    raw_df, defect_labels_df = _inject_defects(clean_df)
    return raw_df, defect_labels_df


if __name__ == "__main__":
    raw, labels = generate_benchmark_data()
    print(f"Generated {len(raw)} rows across {raw['station_id'].nunique()} stations")
    print(f"Defects injected: {len(labels)}")
    print(labels.to_string(index=False))
