"""
guardrails.py
Four Safety Guardrails for Rain Check Karnataka.
Ensures data integrity, immutability, and audit provenance.
"""

import hashlib
import json
import io
import re
from typing import Any

import pandas as pd
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

STANDARD_COLUMNS  = {"station_id", "station_name", "district", "taluk", "date", "rainfall_mm"}
MINIMUM_ROWS      = 10
MAXIMUM_FILE_SIZE = 50 * 1024 * 1024    # 50 MB
EXTREME_RAIN_PRESERVE_THRESHOLD_MM = 150.0
SENSOR_ERROR_THRESHOLD_MM          = 400.0

# Keep REQUIRED_COLUMNS as an alias so existing code that imports it still works
REQUIRED_COLUMNS = STANDARD_COLUMNS

# Accepted file extensions
_ACCEPTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}

# Encoding and delimiter candidates tried in order
_ENCODINGS  = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
_DELIMITERS = [",", "\t", ";", "|"]

# ── Deterministic column-alias dictionary ─────────────────────────────────────
# Maps common real-world column name variants → standard name.
# Keys are lowercased & stripped before lookup.
COLUMN_ALIASES: dict[str, str] = {
    # station_id  (genuine station identifiers only — NOT serial/row numbers)
    "station_id": "station_id", "station id": "station_id",
    "station_code": "station_id", "stn_id": "station_id",
    "stn id": "station_id", "stn code": "station_id",
    # serial / row-number columns → serial_no  (must NEVER become station_id)
    "sl no": "serial_no", "sl_no": "serial_no",
    "serial no": "serial_no", "s.no": "serial_no",
    "sr no": "serial_no", "sr_no": "serial_no",
    "sno": "serial_no", "s no": "serial_no",
    # station_name
    "station_name": "station_name", "station name": "station_name",
    "station": "station_name", "stn_name": "station_name",
    "observatory": "station_name", "aws name": "station_name",
    # district
    "district": "district", "district_name": "district",
    "dist": "district", "dist_name": "district",
    "district name": "district", "mandal": "district",
    # taluk
    "taluk": "taluk", "taluka": "taluk", "sub_district": "taluk",
    "sub district": "taluk", "tehsil": "taluk",
    # date
    "date": "date", "obs_date": "date", "observation_date": "date",
    "report_date": "date", "reported date": "date", "day": "date",
    # rainfall_mm
    "rainfall_mm": "rainfall_mm", "rainfall(mm)": "rainfall_mm",
    "rainfall (mm)": "rainfall_mm", "rain_mm": "rainfall_mm",
    "rain(mm)": "rainfall_mm", "rain (mm)": "rainfall_mm",
    "rainfall": "rainfall_mm", "rain": "rainfall_mm",
    "precip": "rainfall_mm", "precipitation": "rainfall_mm",
    "precipitation(mm)": "rainfall_mm", "precip_mm": "rainfall_mm",
    "annual actual": "rainfall_mm",   # district-level aggregated
    "actual": "rainfall_mm",
    "total rainfall": "rainfall_mm",
}


# ─── Guardrail 1: Input Validation ───────────────────────────────────────────

class InputGuardrailError(ValueError):
    """Raised when an uploaded file fails input validation (hard rejection only)."""


def _sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _is_html_content(data: bytes) -> bool:
    """Return True if the content looks like an HTML page (xls-disguised-as-html)."""
    preview = data[:2048].decode("utf-8", errors="replace").lower().lstrip()
    return preview.startswith("<!doctype html") or preview.startswith("<html")


def _read_excel(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Read an Excel file (.xlsx or .xls) or an HTML table disguised as .xls.
    Returns (df, format_description).
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    # HTML disguised as .xls → parse with pandas HTML reader
    if _is_html_content(file_bytes):
        try:
            tables = pd.read_html(io.BytesIO(file_bytes), encoding="utf-8")
            if tables:
                return tables[0].astype(str), "HTML table (disguised as .xls)"
        except Exception:
            pass
        try:
            tables = pd.read_html(io.BytesIO(file_bytes), encoding="latin-1")
            if tables:
                return tables[0].astype(str), "HTML table (disguised as .xls, latin-1)"
        except Exception as exc:
            raise InputGuardrailError(
                f"File appears to be an HTML page disguised as '{filename}' "
                f"and could not be parsed as a table: {exc}"
            ) from exc

    # True Excel
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
        return df, f".{ext} Excel workbook"
    except Exception as exc:
        raise InputGuardrailError(
            f"Could not read Excel file '{filename}': {exc}"
        ) from exc


def _try_parse_csv(file_bytes: bytes) -> tuple[pd.DataFrame, str, str]:
    """
    Try every combination of encoding × delimiter until one produces a
    DataFrame with ≥ 2 columns.  Returns (df, encoding_used, delimiter_used).
    Raises InputGuardrailError if nothing works.
    """
    last_exc: Exception | None = None
    for enc in _ENCODINGS:
        for delim in _DELIMITERS:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    dtype=str,
                    encoding=enc,
                    sep=delim,
                    engine="python",
                    on_bad_lines="skip",
                )
                if df.shape[1] >= 2:
                    return df, enc, delim
            except Exception as exc:
                last_exc = exc
    raise InputGuardrailError(
        f"File could not be parsed as a tabular CSV with any supported "
        f"encoding ({', '.join(_ENCODINGS)}) or delimiter "
        f"(comma, tab, semicolon, pipe). "
        f"Last error: {last_exc}"
    )


def _apply_alias_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Attempt to rename columns using the deterministic COLUMN_ALIASES dict.
    Returns (df_renamed, mapping_applied) where mapping_applied maps
    original_col → standard_col for every alias hit found.
    Columns with no alias match are left unchanged.
    """
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()

    for col in df.columns:
        # Normalise: lowercase, strip whitespace, collapse multiple spaces
        normalised = re.sub(r"\s+", " ", col.lower().strip())
        # Also try removing parentheses & special chars for fuzzy match
        fuzzy = re.sub(r"[^a-z0-9 ]", "", normalised)

        target = COLUMN_ALIASES.get(normalised) or COLUMN_ALIASES.get(fuzzy)
        if target and target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, rename_map


def _detect_district_level(df: pd.DataFrame) -> bool:
    """
    Return True if the data appears to be district-level aggregated
    (has a 'district' column but no 'date' column).
    """
    cols = set(df.columns)
    return "district" in cols and "date" not in cols


def _expand_district_level(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Convert a district-level aggregated DataFrame into a station-level-compatible
    format that the agents can process.

    Strategy:
    - station_id   = district NAME (not any serial/row number column)
    - station_name = district name
    - date         = "2025-01-01"  (synthetic placeholder — clearly labelled)
    - rainfall_mm  = annual actual (or best numeric column found)
    - taluk        = "district-level"

    Returns (expanded_df, info_messages, synthesised_fields).
    synthesised_fields records which columns were synthetically added, for
    inclusion in the provenance / audit JSON.
    """
    msgs: list[str] = []
    synthesised: dict[str, str] = {}   # {col: reason}
    out = df.copy()

    if "district" not in out.columns:
        return out, msgs, synthesised

    msgs.append(
        "ℹ️ **District-level data detected** — this file contains one row per district "
        "with seasonal/annual totals, not daily station observations. "
        "District names are used as station IDs, and a synthetic placeholder date is added. "
        "Checks that require daily time-series (date gaps, duplicate dates) are not "
        "applicable to this data and will be labelled N/A."
    )

    # ── station_id MUST come from district name, never from a serial number ──
    # If the alias mapper placed a serial number into station_id, fix it now.
    if "station_id" in out.columns:
        # Detect if station_id looks like a numeric serial (1, 2, 3 …)
        is_serial = pd.to_numeric(out["station_id"], errors="coerce").notna().mean() > 0.8
        if is_serial:
            # Move it to serial_no (preserve for reference) then overwrite station_id
            out["serial_no"] = out["station_id"]
            out["station_id"] = out["district"].str.lower().str.replace(r"\s+", "_", regex=True)
            msgs.append(
                "ℹ️ Column **station_id** contained row serial numbers — "
                "overwritten with district names. Serial numbers preserved in **serial_no**."
            )
            synthesised["station_id"] = "derived from district name (serial numbers moved to serial_no)"
    else:
        out["station_id"] = out["district"].str.lower().str.replace(r"\s+", "_", regex=True)
        synthesised["station_id"] = "derived from district name"

    # ── station_name from district ────────────────────────────────────────────
    if "station_name" not in out.columns:
        out["station_name"] = out["district"]
        synthesised["station_name"] = "copied from district column"

    # ── taluk placeholder ─────────────────────────────────────────────────────
    if "taluk" not in out.columns:
        out["taluk"] = "district-level"
        synthesised["taluk"] = "placeholder (district-level data)"

    # ── Synthetic date ────────────────────────────────────────────────────────
    DATE_PLACEHOLDER = "2025-01-01"
    if "date" not in out.columns:
        out["date"] = DATE_PLACEHOLDER
        synthesised["date"] = (
            f"synthetic placeholder '{DATE_PLACEHOLDER}' — "
            "no date column in source file; daily checks (gaps, duplicate dates) are N/A"
        )
        msgs.append(
            f"⚠️ **No date column in your file — a placeholder date ({DATE_PLACEHOLDER}) "
            f"was added.** Daily checks such as date gaps and duplicate dates are not "
            f"applicable to this file and will be labelled **N/A — aggregate data**."
        )

    # ── rainfall_mm ───────────────────────────────────────────────────────────
    if "rainfall_mm" not in out.columns:
        for cand in ["annual actual", "actual"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "rainfall_mm"})
                msgs.append(f"ℹ️ Column **'{cand}'** mapped to **rainfall_mm** (annual actual).")
                break
        else:
            for col in out.columns:
                skip = {"station_id", "station_name", "district", "taluk", "date", "serial_no"}
                if col not in skip:
                    if pd.to_numeric(out[col], errors="coerce").notna().mean() > 0.5:
                        out = out.rename(columns={col: "rainfall_mm"})
                        msgs.append(f"ℹ️ Column **'{col}'** used as rainfall_mm (best numeric match).")
                        break

    # ── Drop STATE summary row ────────────────────────────────────────────────
    state_mask = out["district"].str.upper().str.strip().isin(["STATE", "KARNATAKA", "TOTAL"])
    dropped = int(state_mask.sum())
    if dropped > 0:
        out = out[~state_mask].reset_index(drop=True)
        msgs.append(f"ℹ️ Removed {dropped} state/total summary row(s) from the dataset.")

    return out, msgs, synthesised


def validate_csv_upload(
    file_bytes: bytes,
    filename: str,
) -> tuple[pd.DataFrame, list[str], str, dict[str, str], dict[str, str]]:
    """
    INPUT GUARDRAIL — lenient, multi-format version.

    Accepts: .csv, .tsv, .txt, .xls, .xlsx
    Handles: utf-8-sig BOM, cp1252, latin-1, comma/tab/semicolon delimiters,
             HTML-disguised .xls, real Excel, district-level vs station-level data.

    Hard rejects (raises InputGuardrailError):
    ─ File > 50 MB
    ─ Extension not in {.csv, .tsv, .txt, .xls, .xlsx}
    ─ Injection patterns (XSS, path traversal, null bytes)
    ─ Completely unparseable (not text or Excel)
    ─ < MINIMUM_ROWS rows after parsing

    Soft feedback (returned, never an exception):
    ─ Encoding / delimiter detection notes
    ─ Column mismatch — alias dict applied first, IBM Granite fallback noted
    ─ District-level data adaptation messages

    SHA-256 is computed on the ORIGINAL uploaded bytes BEFORE any re-encoding.

    Returns
    ───────
    df                : pd.DataFrame   — parsed & schema-adapted dataframe
    warnings          : list[str]      — soft messages for the UI
    raw_checksum      : str            — SHA-256 of the original uploaded bytes
    col_mapping       : dict[str,str]  — {original_col: standard_col} alias hits
    synthesised_fields: dict[str,str]  — {col: reason} for synthetically added columns
    """
    warnings: list[str] = []
    synthesised_fields: dict[str, str] = {}

    # ── SHA-256 on original bytes immediately (before any processing) ─────────
    raw_checksum = _sha256_bytes(file_bytes)

    # ── Hard: size ────────────────────────────────────────────────────────────
    if len(file_bytes) > MAXIMUM_FILE_SIZE:
        raise InputGuardrailError(
            f"File '{filename}' is {len(file_bytes) / 1024 / 1024:.1f} MB — "
            f"exceeds the 50 MB limit."
        )

    # ── Hard: extension ───────────────────────────────────────────────────────
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in _ACCEPTED_EXTENSIONS:
        raise InputGuardrailError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(_ACCEPTED_EXTENSIONS))}"
        )

    # ── Hard: injection / path traversal (text preview only) ─────────────────
    _preview = file_bytes[:4096].decode("utf-8", errors="replace").lower()
    for pat in ["../", "..\\", "<script", "javascript:", "\x00"]:
        if pat in _preview:
            raise InputGuardrailError(
                f"File '{filename}' contains a suspicious pattern ('{pat}') "
                f"and was rejected for security reasons."
            )

    # ── Parse: Excel path vs text path ───────────────────────────────────────
    enc_used = ""
    delim_used = ""
    format_desc = ""

    if ext in {".xls", ".xlsx"}:
        df, format_desc = _read_excel(file_bytes, filename)
        warnings.append(f"ℹ️ Loaded as **{format_desc}**.")
    elif _is_html_content(file_bytes):
        # CSV extension but actually HTML (e.g. .csv saved from Excel web export)
        df, format_desc = _read_excel(file_bytes, filename)
        warnings.append(
            f"ℹ️ File appears to be an **HTML table** (not plain CSV). "
            f"Parsed successfully as: {format_desc}."
        )
    else:
        df, enc_used, delim_used = _try_parse_csv(file_bytes)
        if enc_used not in ("utf-8", "utf-8-sig"):
            warnings.append(
                f"ℹ️ Encoding detected as **{enc_used}** (not UTF-8) — loaded successfully."
            )
        elif enc_used == "utf-8-sig":
            warnings.append(
                "ℹ️ UTF-8 BOM (byte-order mark) detected and stripped automatically."
            )
        if delim_used != ",":
            dname = {"\t": "tab", ";": "semicolon", "|": "pipe"}.get(delim_used, repr(delim_used))
            warnings.append(f"ℹ️ Delimiter detected as **{dname}** — loaded successfully.")

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = df.columns.astype(str).str.lower().str.strip()
    df = df.dropna(axis=1, how="all")   # remove entirely-empty columns

    # ── Hard: minimum rows ────────────────────────────────────────────────────
    if len(df) < MINIMUM_ROWS:
        raise InputGuardrailError(
            f"File has only {len(df)} data rows (minimum required: {MINIMUM_ROWS}). "
            f"Please upload a file with more observations."
        )

    # ── Deterministic alias mapping ───────────────────────────────────────────
    df, col_mapping = _apply_alias_mapping(df)
    if col_mapping:
        mapped_str = ", ".join(f"**{k}** → `{v}`" for k, v in col_mapping.items())
        warnings.append(f"ℹ️ Column aliases auto-applied: {mapped_str}.")

    # ── District-level detection & adaptation ─────────────────────────────────
    if _detect_district_level(df):
        df, dist_msgs, synthesised_fields = _expand_district_level(df)
        warnings.extend(dist_msgs)

    # ── Soft: remaining column mismatch ───────────────────────────────────────
    actual_cols = set(df.columns)
    still_missing = STANDARD_COLUMNS - actual_cols
    if still_missing:
        warnings.append(
            f"⚠️ Columns still unmapped after alias detection: **{sorted(still_missing)}**. "
            f"Agent 1 (SchemaProvenanceAgent) will use IBM Granite 4 H Small "
            f"to attempt automatic mapping."
        )
    extra = actual_cols - STANDARD_COLUMNS
    if extra and not still_missing:
        warnings.append(
            f"ℹ️ Extra columns present: **{sorted(extra)[:5]}** — passed through as-is."
        )

    # ── Soft: rainfall column check ───────────────────────────────────────────
    if "rainfall_mm" not in df.columns:
        # Scan for a numeric-majority column not already mapped
        for col in df.columns:
            if col in {"station_id", "station_name", "district", "taluk", "date"}:
                continue
            if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.5:
                warnings.append(
                    f"⚠️ No rainfall_mm column found. "
                    f"Column **'{col}'** appears numeric — Agent 1 will confirm mapping."
                )
                break
        else:
            warnings.append(
                "⚠️ No numeric column found for rainfall values. "
                "Agent 1 will attempt to identify the correct column."
            )

    return df, warnings, raw_checksum, col_mapping, synthesised_fields


# ─── Guardrail 2: Immutability Guardrail ─────────────────────────────────────

class ImmutabilityGuardrail:
    """
    IMMUTABILITY GUARDRAIL:
    Tracks the SHA-256 checksum of the raw dataframe at ingestion time.
    Can verify at any point that the raw data has not been mutated.
    All cleaning operations must work on a copy.
    """

    def __init__(self):
        self._raw_checksum: str | None = None
        self._raw_snapshot: bytes | None = None

    def register_raw(self, df: pd.DataFrame) -> str:
        """
        Register a raw dataframe. Computes and stores its SHA-256 checksum.
        Returns the checksum string.
        """
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        checksum = hashlib.sha256(csv_bytes).hexdigest()
        self._raw_checksum = checksum
        self._raw_snapshot = csv_bytes
        return checksum

    def verify_raw_unchanged(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Verify that a dataframe matches the registered raw snapshot.
        Returns (is_unchanged: bool, current_checksum: str).
        """
        if self._raw_checksum is None:
            return False, "No raw dataframe registered."
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        current   = hashlib.sha256(csv_bytes).hexdigest()
        return current == self._raw_checksum, current

    @property
    def raw_checksum(self) -> str | None:
        return self._raw_checksum

    def compute_cleaned_checksum(self, cleaned_df: pd.DataFrame) -> str:
        """Compute and return SHA-256 checksum of the cleaned dataframe."""
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()


# ─── Guardrail 3: Extreme Preservation Guardrail ─────────────────────────────

class ExtremePreservationGuardrail:
    """
    EXTREME PRESERVATION GUARDRAIL:
    Prevents accidental auto-deletion or nullification of valid high-rainfall events.
    Valid extremes are those in [150, 400] mm range that are NOT placeholder values.
    """

    @staticmethod
    def find_valid_extremes(df: pd.DataFrame) -> pd.DataFrame:
        """Return rows that are valid extreme rainfall events (must be preserved)."""
        num = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        mask = (
            (num >= EXTREME_RAIN_PRESERVE_THRESHOLD_MM) &
            (num <= SENSOR_ERROR_THRESHOLD_MM) &
            (num != 9999.0)
        )
        return df[mask].copy()

    @staticmethod
    def check_proposed_deletion(df: pd.DataFrame,
                                 indices_to_modify: list[int]) -> list[dict]:
        """
        Before applying any fix, check whether proposed modifications
        would affect any valid extreme event.
        Returns list of violations (empty = safe to proceed).
        """
        extremes = ExtremePreservationGuardrail.find_valid_extremes(df)
        extreme_indices = set(extremes.index.tolist())
        violations = []
        for idx in indices_to_modify:
            if idx in extreme_indices:
                row = df.loc[idx]
                violations.append({
                    "index": idx,
                    "station_id": row.get("station_id"),
                    "date": row.get("date"),
                    "rainfall_mm": row.get("rainfall_mm"),
                    "message": (
                        f"⚠️ Index {idx} ({row.get('station_id')} on {row.get('date')}, "
                        f"{row.get('rainfall_mm')} mm) is a valid extreme event "
                        f"and must NOT be modified or deleted."
                    ),
                })
        return violations

    @staticmethod
    def enforce_on_cleaned(raw_df: pd.DataFrame,
                            cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Post-fix enforcement: ensure all valid extremes from raw_df are
        still present and unmodified in cleaned_df.
        Restores any that were accidentally removed.
        Returns (enforced_df, list_of_restoration_messages).
        """
        raw_extremes = ExtremePreservationGuardrail.find_valid_extremes(raw_df)
        messages = []

        for _, row in raw_extremes.iterrows():
            sid = row["station_id"]
            dt  = row["date"]
            val = row["rainfall_mm"]
            mask = (cleaned_df["station_id"] == sid) & (cleaned_df["date"] == dt)
            if mask.sum() == 0:
                # Restore the row
                cleaned_df = pd.concat(
                    [cleaned_df, pd.DataFrame([row])], ignore_index=True
                )
                messages.append(
                    f"RESTORED: {sid} on {dt} = {val} mm (valid extreme was removed)"
                )
            else:
                # Ensure value is unchanged
                current_val = cleaned_df.loc[mask, "rainfall_mm"].iloc[0]
                if str(current_val) != str(val):
                    cleaned_df.loc[mask, "rainfall_mm"] = val
                    messages.append(
                        f"CORRECTED: {sid} on {dt} restored to {val} mm "
                        f"(had been changed to {current_val})"
                    )

        return cleaned_df, messages


# ─── Guardrail 4: Audit Trail Guardrail ──────────────────────────────────────

def generate_audit_trail(
    raw_checksum: str,
    cleaned_checksum: str,
    audit_log: list[dict],
    applied_changes: list[dict],
    station_summary: dict[str, Any],
    synthesised_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    AUDIT TRAIL GUARDRAIL:
    Generates a full provenance document with:
    - SHA-256 checksums of raw and cleaned datasets
    - Complete audit log
    - Applied changes log
    - Integrity verification status
    - synthesised_fields: columns that were synthetically added during ingestion
    """
    trail = {
        "schema_version": "1.1",
        "tool": "Rain Check Karnataka: Agentic Rainfall Data Quality Investigator",
        "problem_statement": "PS-18",
        "provenance": {
            "raw_sha256": raw_checksum,
            "cleaned_sha256": cleaned_checksum,
            "checksums_differ": raw_checksum != cleaned_checksum,
            "integrity_note": (
                "Raw and cleaned datasets are distinct." if raw_checksum != cleaned_checksum
                else "No approved changes were applied — datasets are identical."
            ),
            "synthesised_fields": synthesised_fields or {},
        },
        "station_summary": station_summary,
        "total_audit_entries": len(audit_log),
        "total_applied_changes": len(applied_changes),
        "audit_log": audit_log,
        "applied_changes": applied_changes,
    }
    return trail


def compute_sha256(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of a dataframe's CSV representation."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ─── Singleton immutability guardrail instance ───────────────────────────────
# (Used by app.py to track the session's raw data)
_immutability_guard = ImmutabilityGuardrail()


def get_immutability_guard() -> ImmutabilityGuardrail:
    return _immutability_guard


def reset_immutability_guard() -> ImmutabilityGuardrail:
    """Reset the singleton guard (call when a new dataset is loaded)."""
    global _immutability_guard
    _immutability_guard = ImmutabilityGuardrail()
    return _immutability_guard
