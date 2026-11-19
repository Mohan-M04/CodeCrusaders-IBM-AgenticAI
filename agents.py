"""
agents.py
Three Specialized AI Agents for Rain Check Karnataka.
Uses IBM watsonx.ai (ibm/granite-4-h-small) with Groq/Gemini fallback.
"""

import os
import json
import time
import hashlib
import logging
import textwrap
from datetime import date, timedelta
from typing import Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── LLM Client Factory ──────────────────────────────────────────────────────

WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")

# Shared performance tracker
_perf_log: list[dict] = []


def _get_perf_log() -> list[dict]:
    return _perf_log


def _record_perf(agent: str, model: str, latency: float,
                 prompt_tokens: int, completion_tokens: int) -> None:
    _perf_log.append({
        "agent": agent,
        "model": model,
        "latency_s": round(latency, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        # Granite-4-h-small pricing estimate (per 1k tokens)
        "cost_usd": round((prompt_tokens + completion_tokens) / 1000 * 0.0006, 6),
    })


def _call_watsonx(prompt: str, agent_name: str) -> tuple[str, bool]:
    """Call IBM watsonx.ai Granite-4-h-small via chat API. Returns (response_text, success)."""
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return "", False
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        t0 = time.time()
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        api_client = APIClient(credentials)
        model = ModelInference(
            model_id="ibm/granite-4-h-small",
            api_client=api_client,
            project_id=WATSONX_PROJECT_ID,
        )
        # Use chat API (non-deprecated)
        messages = [{"role": "user", "content": prompt}]
        resp = model.chat(
            messages=messages,
            params={"max_tokens": 800, "temperature": 0.1},
        )
        latency = time.time() - t0

        # Extract response text
        text = ""
        try:
            text = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = str(resp)

        # Extract token usage
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        pt = usage.get("prompt_tokens", len(prompt) // 4)
        ct = usage.get("completion_tokens", len(text) // 4)

        _record_perf(agent_name, "ibm/granite-4-h-small", latency, pt, ct)
        return text, True

    except Exception as exc:
        logger.warning("watsonx call failed: %s", exc)
        return "", False


def _call_groq(prompt: str, agent_name: str) -> tuple[str, bool]:
    """Fallback: Groq — tries qwen/qwen3.8-27b then openai/gpt-oss-20b."""
    if not GROQ_API_KEY:
        return "", False
    try:
        from groq import Groq
        t0 = time.time()
        client = Groq(api_key=GROQ_API_KEY)
        # Try models in preference order (gracefully handles unavailable models)
        resp = None
        used_model = "groq"
        for model_id in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                         "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.1,
                )
                used_model = f"groq/{model_id}"
                break
            except Exception:
                continue
        if resp is None:
            return "", False
        latency = time.time() - t0
        text = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else len(prompt) // 4
        ct = resp.usage.completion_tokens if resp.usage else len(text) // 4
        _record_perf(agent_name, used_model, latency, pt, ct)
        return text, True
    except Exception as exc:
        logger.warning("Groq call failed: %s", exc)
        return "", False


def _call_gemini(prompt: str, agent_name: str) -> tuple[str, bool]:
    """Fallback: Google Gemini Flash (uses google-genai SDK)."""
    if not GEMINI_API_KEY:
        return "", False
    try:
        from google import genai
        t0 = time.time()
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        latency = time.time() - t0
        text = response.text or ""
        pt = len(prompt) // 4
        ct = len(text) // 4
        _record_perf(agent_name, "gemini-2.0-flash", latency, pt, ct)
        return text, True
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return "", False


def call_llm(prompt: str, agent_name: str = "unknown") -> tuple[str, str]:
    """
    Call LLM with waterfall: watsonx → Groq → Gemini.
    Returns (response_text, model_used).
    """
    text, ok = _call_watsonx(prompt, agent_name)
    if ok and text.strip():
        return text, "ibm/granite-4-h-small"

    text, ok = _call_groq(prompt, agent_name)
    if ok and text.strip():
        return text, "groq/llama-3.3-70b"

    text, ok = _call_gemini(prompt, agent_name)
    if ok and text.strip():
        return text, "gemini-2.0-flash"

    return "⚠️ All LLM backends unavailable. Running deterministic checks only.", "none"


# ─── AGENT 1: SchemaProvenanceAgent ──────────────────────────────────────────

STANDARD_SCHEMA = {
    "station_id":   {"dtype": "object",  "required": True,  "description": "Unique station identifier"},
    "station_name": {"dtype": "object",  "required": True,  "description": "Human-readable station name"},
    "district":     {"dtype": "object",  "required": True,  "description": "Karnataka district"},
    "taluk":        {"dtype": "object",  "required": False, "description": "Sub-district administrative unit"},
    "date":         {"dtype": "object",  "required": True,  "description": "Observation date YYYY-MM-DD"},
    "rainfall_mm":  {"dtype": "numeric", "required": True,  "description": "Rainfall in millimetres"},
}


def run_schema_agent(df: pd.DataFrame) -> dict[str, Any]:
    """
    AGENT 1 – SchemaProvenanceAgent.
    Validates schema, column mapping, date format, units, and district metadata.
    Uses IBM Granite for structured JSON analysis report.
    """
    result: dict[str, Any] = {
        "agent": "SchemaProvenanceAgent",
        "model_used": "none",
        "status": "PASS",
        "issues": [],
        "column_mapping": {},
        "date_format_valid": False,
        "unit_check": {},
        "district_summary": {},
        "llm_analysis": "",
        "llm_raw_response": "",
    }

    # ── Deterministic checks ─────────────────────────────────────────────────
    actual_cols = set(df.columns.str.lower().str.strip())
    required = {k for k, v in STANDARD_SCHEMA.items() if v["required"]}
    missing_cols = required - actual_cols
    extra_cols   = actual_cols - set(STANDARD_SCHEMA.keys())

    result["column_mapping"] = {c: ("MATCHED" if c in actual_cols else "MISSING")
                                 for c in STANDARD_SCHEMA}

    if missing_cols:
        result["issues"].append(f"Missing required columns: {sorted(missing_cols)}")
        result["status"] = "FAIL"

    if extra_cols:
        result["issues"].append(f"Unrecognised extra columns: {sorted(extra_cols)}")

    # Date format validation
    if "date" in df.columns:
        try:
            parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
            bad_dates = parsed.isna().sum()
            result["date_format_valid"] = (bad_dates == 0)
            if bad_dates > 0:
                result["issues"].append(f"{bad_dates} rows have invalid date format (expected YYYY-MM-DD)")
                result["status"] = "WARN"
        except Exception as exc:
            result["issues"].append(f"Date parsing error: {exc}")

    # Rainfall unit check
    if "rainfall_mm" in df.columns:
        num_series = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        non_numeric = num_series.isna().sum()
        result["unit_check"] = {
            "column": "rainfall_mm",
            "non_numeric_count": int(non_numeric),
            "max_value": float(num_series.max()) if not num_series.isna().all() else None,
            "min_value": float(num_series.min()) if not num_series.isna().all() else None,
            "unit_declared": "mm",
        }
        if non_numeric > 0:
            result["issues"].append(f"{non_numeric} non-numeric values in rainfall_mm column")
            result["status"] = "WARN"

    # District metadata
    if "district" in df.columns:
        result["district_summary"] = df.groupby("district")["station_id"].nunique().to_dict()

    # ── LLM Schema Analysis ───────────────────────────────────────────────────
    prompt = textwrap.dedent(f"""
        You are a Karnataka rainfall data quality expert.
        Analyse the schema of this dataset and return ONLY a JSON object.

        Dataset columns: {list(df.columns)}
        Row count: {len(df)}
        Detected issues: {result["issues"]}
        Date format valid: {result["date_format_valid"]}
        Rainfall_mm stats: {result.get("unit_check", {})}
        Districts found: {list(result["district_summary"].keys())}

        Return JSON with keys:
        - schema_verdict: "PASS" | "WARN" | "FAIL"
        - provenance_notes: list of strings (observations about data origin/quality)
        - drift_detected: boolean
        - drift_description: string (empty if no drift)
        - recommendations: list of strings

        JSON only, no markdown fences:
    """).strip()

    llm_text, model_used = call_llm(prompt, "SchemaProvenanceAgent")
    result["model_used"]    = model_used
    result["llm_raw_response"] = llm_text

    # Parse JSON from LLM response
    try:
        clean = llm_text.strip()
        # Strip markdown fences if present
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        llm_json = json.loads(clean)
        result["llm_analysis"] = llm_json
        if llm_json.get("schema_verdict") in ("WARN", "FAIL"):
            result["status"] = llm_json["schema_verdict"]
    except Exception:
        result["llm_analysis"] = {"raw": llm_text}

    return result


# ─── AGENT 2: QualityInvestigationAgent ──────────────────────────────────────

# Thresholds for outlier detection (Karnataka domain knowledge)
EXTREME_RAIN_THRESHOLD_MM = 150.0   # ≥ this is extreme but possibly valid
SENSOR_ERROR_THRESHOLD_MM = 400.0   # ≥ this is almost certainly a sensor error
INVALID_PLACEHOLDER       = 9999.0  # Known IMD invalid placeholder


def _check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Exact duplicate rows: same station_id + date + rainfall_mm value.
    Rows with same station+date but different rainfall values are conflicts, not duplicates.
    """
    num = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(-99999)
    df2 = df.copy()
    df2["_num_rain"] = num

    # Only flag as duplicate when station_id + date + rainfall_mm are all identical
    dupes_all   = df2[df2.duplicated(subset=["station_id", "date", "_num_rain"], keep=False)].copy()
    dupes_extra = df2[df2.duplicated(subset=["station_id", "date", "_num_rain"], keep="first")].copy()

    return {
        "check": "Duplicates",
        "count": len(dupes_extra),
        "affected_indices": dupes_extra.index.tolist(),
        "details": dupes_all[["station_id", "date", "rainfall_mm"]].to_dict(orient="records"),
    }


def _check_negatives(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect physically-impossible negative rainfall values.
    Excludes the extreme invalid placeholder -9999.0, which is caught by the
    outlier/sensor-error check instead (to avoid double-counting).
    """
    num = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    # -9999.0 and similar extreme negatives are sensor-error placeholders, handled in outliers
    neg_mask = (num < 0) & (num > -100)
    negatives = df[neg_mask].copy()
    return {
        "check": "Negative Values",
        "count": int(neg_mask.sum()),
        "affected_indices": negatives.index.tolist(),
        "details": negatives[["station_id", "date", "rainfall_mm"]].to_dict(orient="records"),
    }


def _check_date_gaps(df: pd.DataFrame) -> dict[str, Any]:
    """Find missing observation dates per station within the data's date window."""
    gaps = []
    df2 = df.copy()
    df2["_date_parsed"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["_date_parsed"])

    for station_id, grp in df2.groupby("station_id"):
        all_dates = set(pd.date_range(grp["_date_parsed"].min(),
                                      grp["_date_parsed"].max(), freq="D"))
        present   = set(grp["_date_parsed"])
        missing   = sorted(all_dates - present)
        for d in missing:
            gaps.append({"station_id": station_id, "missing_date": str(d.date()),
                         "index": -1})

    return {
        "check": "Date Gaps",
        "count": len(gaps),
        "affected_indices": [-1] * len(gaps),
        "details": gaps,
    }


def _check_conflicts(df: pd.DataFrame) -> dict[str, Any]:
    """Same station+date with different rainfall_mm values."""
    conflicts = []
    num = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    df2 = df.copy()
    df2["_num_rain"] = num
    grouped = df2.groupby(["station_id", "date"])

    for (sid, dt), grp in grouped:
        if len(grp) > 1:
            vals = grp["_num_rain"].dropna().unique()
            if len(vals) > 1:
                conflicts.append({
                    "station_id": sid,
                    "date": dt,
                    "conflicting_values": grp["rainfall_mm"].tolist(),
                    "indices": grp.index.tolist(),
                })

    return {
        "check": "Conflicting Records",
        "count": len(conflicts),
        "affected_indices": [i for c in conflicts for i in c["indices"]],
        "details": conflicts,
    }


def _check_outliers(df: pd.DataFrame) -> dict[str, Any]:
    """
    Triage extreme rainfall values:
    - Valid extreme: 150–400 mm (legitimate heavy monsoon)
    - Sensor error: > 400 mm OR == 9999.0 OR non-numeric
    """
    num = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    non_numeric_mask  = df["rainfall_mm"].apply(
        lambda x: not str(x).replace(".", "", 1).replace("-", "", 1).lstrip().rstrip().isnumeric()
        if not isinstance(x, (int, float)) else False
    )
    invalid_mask  = (num == INVALID_PLACEHOLDER) | (num <= -100) | (num > SENSOR_ERROR_THRESHOLD_MM)
    extreme_mask  = (num >= EXTREME_RAIN_THRESHOLD_MM) & (num <= SENSOR_ERROR_THRESHOLD_MM) & ~invalid_mask

    errors   = df[invalid_mask | non_numeric_mask].copy()
    extremes = df[extreme_mask].copy()

    return {
        "check": "Outliers & Extremes",
        "sensor_error_count":  int((invalid_mask | non_numeric_mask).sum()),
        "legitimate_extreme_count": int(extreme_mask.sum()),
        "sensor_error_indices": errors.index.tolist(),
        "extreme_preserved_indices": extremes.index.tolist(),
        "sensor_errors": errors[["station_id", "date", "rainfall_mm"]].to_dict(orient="records"),
        "legitimate_extremes": extremes[["station_id", "date", "rainfall_mm"]].to_dict(orient="records"),
        "count": int((invalid_mask | non_numeric_mask).sum()),  # Only actual errors counted
    }


def run_quality_agent(df: pd.DataFrame) -> dict[str, Any]:
    """
    AGENT 2 – QualityInvestigationAgent.
    Runs 5 deterministic quality checks and uses IBM Granite for triage narrative.
    """
    t0 = time.time()
    checks = {
        "duplicates":  _check_duplicates(df),
        "negatives":   _check_negatives(df),
        "date_gaps":   _check_date_gaps(df),
        "conflicts":   _check_conflicts(df),
        "outliers":    _check_outliers(df),
    }

    total_defects = sum(
        c.get("count", 0) for c in checks.values()
    )

    # ── LLM Triage Narrative ─────────────────────────────────────────────────
    summary = {k: {"count": v.get("count", 0),
                   "sensor_errors": v.get("sensor_error_count"),
                   "extremes_preserved": v.get("legitimate_extreme_count")}
               for k, v in checks.items()}

    prompt = textwrap.dedent(f"""
        You are a Karnataka rainfall data quality investigator.
        A 5-point investigation has completed. Provide a triage narrative as JSON.

        Investigation summary:
        {json.dumps(summary, indent=2)}

        Return JSON with keys:
        - overall_quality: "GOOD" | "MODERATE" | "POOR"
        - quality_score_pct: integer 0-100
        - critical_issues: list of strings
        - recommended_actions: list of strings
        - extreme_rain_note: string (how to handle the extreme rainfall events found)

        JSON only, no markdown fences:
    """).strip()

    llm_text, model_used = call_llm(prompt, "QualityInvestigationAgent")

    llm_analysis: dict = {}
    try:
        clean = llm_text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        llm_analysis = json.loads(clean)
    except Exception:
        llm_analysis = {"raw": llm_text}

    return {
        "agent": "QualityInvestigationAgent",
        "model_used": model_used,
        "total_defects_found": total_defects,
        "checks": checks,
        "llm_triage": llm_analysis,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ─── AGENT 3: AuditReportAgent ───────────────────────────────────────────────

ACTION_FLAG         = "Flag for Review"
ACTION_NULL         = "Impute Null (NaN)"
ACTION_RETAIN       = "Retain as Valid Extreme"
ACTION_REMOVE_DUP   = "Remove Duplicate Row"
ACTION_REMOVE_CONF  = "Remove Conflicting Row (keep lower value)"


def _propose_action(defect_type: str) -> str:
    mapping = {
        "NEGATIVE_VALUE":      ACTION_NULL,
        "DUPLICATE_RECORD":    ACTION_REMOVE_DUP,
        "DATE_GAP":            ACTION_FLAG,
        "CONFLICTING_RECORD":  ACTION_REMOVE_CONF,
        "CORRUPTED_VALUE":     ACTION_NULL,
        "SENSOR_ERROR":        ACTION_NULL,
        "LEGITIMATE_EXTREME":  ACTION_RETAIN,
    }
    return mapping.get(defect_type, ACTION_FLAG)


def run_audit_agent(df: pd.DataFrame,
                    quality_results: dict[str, Any]) -> dict[str, Any]:
    """
    AGENT 3 – AuditReportAgent.
    Generates reversible audit log and proposes Human-in-the-Loop actions.
    Does NOT apply any changes — changes are only applied after user approval.
    """
    checks = quality_results.get("checks", {})
    audit_entries: list[dict] = []
    entry_id = 1

    # ── Build audit log from check results ───────────────────────────────────
    # Duplicates
    for rec in checks.get("duplicates", {}).get("details", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "DUPLICATE_RECORD",
            "station_id": rec.get("station_id"), "date": rec.get("date"),
            "original_value": rec.get("rainfall_mm"),
            "proposed_action": ACTION_REMOVE_DUP,
            "user_decision": None, "reversible": True,
        })
        entry_id += 1

    # Negatives
    for rec in checks.get("negatives", {}).get("details", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "NEGATIVE_VALUE",
            "station_id": rec.get("station_id"), "date": rec.get("date"),
            "original_value": rec.get("rainfall_mm"),
            "proposed_action": ACTION_NULL,
            "user_decision": None, "reversible": True,
        })
        entry_id += 1

    # Date gaps (flag only – cannot add data)
    for rec in checks.get("date_gaps", {}).get("details", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "DATE_GAP",
            "station_id": rec.get("station_id"), "date": rec.get("missing_date"),
            "original_value": "MISSING",
            "proposed_action": ACTION_FLAG,
            "user_decision": None, "reversible": True,
        })
        entry_id += 1

    # Conflicts
    for rec in checks.get("conflicts", {}).get("details", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "CONFLICTING_RECORD",
            "station_id": rec.get("station_id"), "date": rec.get("date"),
            "original_value": str(rec.get("conflicting_values")),
            "proposed_action": ACTION_REMOVE_CONF,
            "user_decision": None, "reversible": True,
        })
        entry_id += 1

    # Sensor errors (from outlier check)
    for rec in checks.get("outliers", {}).get("sensor_errors", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "SENSOR_ERROR",
            "station_id": rec.get("station_id"), "date": rec.get("date"),
            "original_value": rec.get("rainfall_mm"),
            "proposed_action": ACTION_NULL,
            "user_decision": None, "reversible": True,
        })
        entry_id += 1

    # Legitimate extremes – always RETAIN
    for rec in checks.get("outliers", {}).get("legitimate_extremes", []):
        audit_entries.append({
            "id": entry_id, "defect_type": "LEGITIMATE_EXTREME",
            "station_id": rec.get("station_id"), "date": rec.get("date"),
            "original_value": rec.get("rainfall_mm"),
            "proposed_action": ACTION_RETAIN,
            "user_decision": None, "reversible": False,  # Cannot delete valid extremes
        })
        entry_id += 1

    # ── LLM Explanation ───────────────────────────────────────────────────────
    summary_for_llm = [
        {"defect_type": e["defect_type"], "station_id": e["station_id"],
         "date": e["date"], "value": e["original_value"],
         "proposed_action": e["proposed_action"]}
        for e in audit_entries[:12]  # Limit to avoid token overflow
    ]

    prompt = textwrap.dedent(f"""
        You are an expert Karnataka rainfall data auditor.
        Review these {len(audit_entries)} data quality findings and write a professional audit report summary.

        Findings (first 12 shown):
        {json.dumps(summary_for_llm, indent=2)}

        Write a concise audit report with:
        1. Executive Summary (2-3 sentences)
        2. Key Findings by Category
        3. Data Integrity Risk Assessment (Low/Medium/High)
        4. Recommended Next Steps

        Keep it under 400 words. Plain text only, no markdown headers:
    """).strip()

    llm_text, model_used = call_llm(prompt, "AuditReportAgent")

    return {
        "agent": "AuditReportAgent",
        "model_used": model_used,
        "total_entries": len(audit_entries),
        "audit_log": audit_entries,
        "llm_explanation": llm_text,
    }


def apply_approved_fixes(raw_df: pd.DataFrame,
                         audit_log: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply only user-APPROVED fixes from the audit log to a COPY of raw_df.
    Raw data is NEVER modified.
    Returns (cleaned_df, applied_changes).
    """
    cleaned = raw_df.copy()
    applied = []

    approved = [e for e in audit_log if e.get("user_decision") == "Approve"]

    for entry in approved:
        defect_type = entry["defect_type"]
        sid         = entry["station_id"]
        dt          = entry["date"]
        action      = entry["proposed_action"]

        if defect_type == "LEGITIMATE_EXTREME":
            # Hard guardrail: never delete valid extremes
            continue

        mask = (cleaned["station_id"] == sid) & (cleaned["date"] == dt)

        if action == ACTION_NULL and defect_type in ("NEGATIVE_VALUE", "SENSOR_ERROR", "CORRUPTED_VALUE"):
            if mask.sum() > 0:
                cleaned.loc[mask, "rainfall_mm"] = np.nan
                applied.append({**entry, "applied": True, "new_value": None})

        elif action == ACTION_REMOVE_DUP and defect_type == "DUPLICATE_RECORD":
            dup_mask = cleaned.duplicated(subset=["station_id", "date"], keep="first") & mask
            cleaned = cleaned[~dup_mask].reset_index(drop=True)
            applied.append({**entry, "applied": True, "new_value": "row_removed"})

        elif action == ACTION_REMOVE_CONF and defect_type == "CONFLICTING_RECORD":
            # Keep the row with the lower rainfall value
            grp = cleaned[mask]
            if len(grp) > 1:
                min_idx = pd.to_numeric(grp["rainfall_mm"], errors="coerce").idxmin()
                drop_idx = [i for i in grp.index if i != min_idx]
                cleaned = cleaned.drop(index=drop_idx).reset_index(drop=True)
                applied.append({**entry, "applied": True, "new_value": "kept_lower_value"})

        elif action == ACTION_FLAG:
            # FLAG adds a suffix note — no data mutation
            applied.append({**entry, "applied": True, "new_value": "flagged_only"})

    return cleaned, applied


def translate_audit_summary(english_text: str, language: str) -> tuple[str, str]:
    """
    Translate an English audit summary into the requested language using IBM Granite.
    language: "English" | "ಕನ್ನಡ (Kannada)"
    Returns (translated_text, model_used).
    If language is English, returns the original text without an LLM call.
    """
    if language == "English" or not english_text.strip():
        return english_text, "none"

    prompt = textwrap.dedent(f"""
        You are a bilingual Karnataka government data auditor fluent in both English and Kannada.

        Translate the following rainfall data quality audit report into natural, formal Kannada
        (ಕನ್ನಡ). Preserve all technical terms (station names, numerical values, percentages) as-is.
        Return ONLY the Kannada translation — no English, no preamble.

        Original English audit report:
        ---
        {english_text}
        ---

        Kannada translation:
    """).strip()

    text, model = call_llm(prompt, "AuditTranslationAgent")
    return text, model


def get_perf_summary() -> dict[str, Any]:
    """Return aggregated LLM performance metrics."""
    if not _perf_log:
        return {}
    total_tokens  = sum(p["total_tokens"] for p in _perf_log)
    total_latency = sum(p["latency_s"] for p in _perf_log)
    total_cost    = sum(p["cost_usd"] for p in _perf_log)
    models_used   = list({p["model"] for p in _perf_log})
    return {
        "calls": len(_perf_log),
        "total_tokens": total_tokens,
        "total_latency_s": round(total_latency, 2),
        "total_cost_usd": round(total_cost, 6),
        "models_used": models_used,
        "per_call": _perf_log,
    }
