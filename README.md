# 🌧️ Rain Check Karnataka — Agentic Rainfall Data Quality Investigator

<p align="center">
  <img src="https://img.shields.io/badge/Model-IBM%20Granite%204--H--Small-0062ff?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM Granite"/>
  <img src="https://img.shields.io/badge/Platform-IBM%20Cloud%20watsonx.ai-be95ff?style=for-the-badge&logo=ibm&logoColor=white" alt="watsonx.ai"/>
  <img src="https://img.shields.io/badge/Agents-3%20Specialized-22c55e?style=for-the-badge" alt="Agents"/>
  <img src="https://img.shields.io/badge/Tests-26%2F26%20Passing-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/F1%20Score-100%25-success?style=for-the-badge" alt="F1 Score"/>
  <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

<p align="center">
  <strong>Inter-Institute IBM Bob Hackathon 2026 · Problem Statement 18</strong><br/>
  Climate Data Engineering · Data Reliability · Auditable Agentic Workflows<br/><br/>
  <strong>Team:</strong> Code Crusaders &nbsp;|&nbsp;
  <strong>Members:</strong> Mohan M &amp; Kusuma &nbsp;|&nbsp;
  <strong>Institution:</strong> Sambhram Institute of Technology, Bengaluru
</p>

---

## 📌 Executive Summary

Rainfall summaries across Karnataka frequently become unreliable when telemetry stations encounter sensor freezes, transmission packet drops, and transmission glitches. A **missing observation is not zero rainfall**, and an **intense monsoon cloudburst is not a defect**.

**Rain Check Karnataka** is an enterprise-grade, fully auditable Agentic AI quality investigator that:
- Validates schemas and detects schema drift using **IBM Granite 4-H-Small**
- Executes a **5-point deterministic verification matrix** (duplicates, negatives, date gaps, conflicts, outlier triage)
- Enforces **mathematical raw data immutability** via SHA-256 cryptographic provenance
- Generates **reversible audit reports** with Human-in-the-Loop governance
- Supports **bilingual reporting** in English and **ಕನ್ನಡ (Kannada)**
- Handles both **daily station files** and **district-level aggregated data** (e.g. KSNDMC annual reports)

---

## 🤖 Multi-Agent Architecture

```
 [Raw Rainfall Dataset  ·  CSV / TSV / XLS / XLSX]
                          │
                          ▼
 ┌────────────────────────────────────────────────────────────┐
 │          Cryptographic Immutability Guardrail              │
 │    SHA-256 hash registered on raw bytes — read-only locked │
 └────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────┐
 │  AGENT 1 — SchemaProvenanceAgent                           │
 │  Model : IBM Granite 4-H-Small (watsonx.ai)                │
 │  Tasks : Schema drift detection · Date format validation   │
 │          Unit verification (mm) · District metadata mapping│
 └────────────────────────────┬───────────────────────────────┘
                              │  Validated Schema Handoff
                              ▼
 ┌────────────────────────────────────────────────────────────┐
 │  AGENT 2 — QualityInvestigationAgent                       │
 │  Engine : 5-Point Deterministic Matrix + IBM Granite       │
 │  Tasks : Duplicates · Negatives · Date gaps · Conflicts    │
 │          Outlier triage (preserves real extreme monsoons)  │
 └────────────────────────────┬───────────────────────────────┘
                              │  Detected Anomalies
                              ▼
 ┌────────────────────────────────────────────────────────────┐
 │  AGENT 3 — AuditReportAgent                                │
 │  Model : IBM Granite 4-H-Small + Groq / Gemini fallback    │
 │  Tasks : Reversible change log · Kannada translation       │
 │          Voice briefing (Web Speech API)                   │
 └────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
 ┌────────────────────────────────────────────────────────────┐
 │          Human-in-the-Loop Approval Barrier                │
 │  Approved fixes written ONLY to a separate cleaned copy    │
 │  Raw data remains 100% immutable at all times              │
 └────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 5-Point Quality Verification Matrix

| # | Check | Description |
|---|-------|-------------|
| 1 | **Duplicates** | Detects identical records with same station + date |
| 2 | **Negative Values** | Flags physically impossible negative rainfall (e.g. `-4.5 mm`) |
| 3 | **Date Gaps** | Finds missing dates in the continuous observation window |
| 4 | **Conflicting Records** | Isolates same station + date with different values |
| 5 | **Outlier & Extreme Triage** | Preserves real cloudburst events (e.g. `185 mm`) while flagging sensor error placeholders (`9999.0 mm`) |

> ✅ **Domain rule enforced:** Legitimate extreme monsoon observations (150–400 mm) are **never** auto-deleted. The `ExtremePreservationGuardrail` will restore any extreme that is accidentally removed.

---

## 📊 Benchmark Results

Tested against the **Karnataka 2024 benchmark dataset** — 4 stations, 2 districts, 1,467 rows, **15 ground-truth injected defects + 2 preserved extreme events**.

| Metric | Result |
|--------|--------|
| Integration Tests | **26 / 26 passing** |
| Detection Precision | **100%** |
| Detection Recall | **100%** |
| F1 Score | **100%** |
| Extreme Events Preserved | **2 / 2** (185 mm BLR_CENTRAL · 192 mm MYS_CITY) |
| IBM Granite Cost | **$0.0014** per full multi-agent audit run |
| Avg LLM Latency | < 3 s per agent call |

---

## 💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| Foundation Model | IBM Granite 4-H-Small (`ibm/granite-4-h-small`) via IBM Cloud watsonx.ai |
| Fallback LLMs | Groq (`llama-3.3-70b-versatile` → `llama-3.1-8b-instant`) · Google Gemini (`gemini-2.0-flash`) |
| Frontend | Python · Streamlit (native components only) |
| Visualisation | Plotly (station geospatial map, completeness heatmap, defect bar chart) |
| Data | Pandas · NumPy |
| Security | SHA-256 provenance · Input guardrails (injection detection, size limit, extension allowlist) |
| Dev Environment | IBM Bob (Agent Mode) |
| Voice | Web Speech API (`window.speechSynthesis`) — browser-native, no server call |

---

## 🗂️ Project Structure

```
CodeCrusaders-IBM-AgenticAI/
│
├── app.py                  # Streamlit 4-tab dashboard (main entry point)
├── agents.py               # 3 AI agents + Kannada translation + LLM perf metrics
├── guardrails.py           # 4 safety guardrails (input, immutability, extreme, audit)
├── sample_data.py          # Karnataka 2024 benchmark generator (15 injected defects)
├── requirements.txt        # Python dependencies
│
├── test_full_suite.py      # 26-test master suite (Section A: uploader · Section B: integration)
├── test_uploader.py        # 12 guardrail & upload tests
├── integration_test.py     # End-to-end pipeline integration test
├── test_guardrails.py      # Quick guardrail smoke test
├── test_agents.py          # Agent 2 deterministic quality check test
├── test_enhancements.py    # Bilingual, map coords, and defect colour-coding tests
└── smoke_test.py           # District-level upload + benchmark F1 smoke test
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mohan-M04/CodeCrusaders-IBM-AgenticAI.git
cd CodeCrusaders-IBM-AgenticAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# IBM Cloud watsonx.ai (required for IBM Granite)
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Groq (Fallback 1 — optional but recommended)
GROQ_API_KEY=your_groq_api_key

# Google Gemini (Fallback 2 — optional)
GEMINI_API_KEY=your_gemini_api_key
```

> 💡 The app works with **any single API key**. IBM Granite is tried first; Groq and Gemini are automatic fallbacks.

### 4. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📋 Usage Walkthrough

### Option A — Karnataka 2024 Benchmark (recommended first run)

1. Click **"🗂️ Load Karnataka Benchmark Dataset (2024 · 15 Injected Defects)"** in the sidebar
2. **Tab 1 → Run Schema Provenance Analysis** — Agent 1 validates schema via IBM Granite
3. **Tab 2 → Run 5-Point Quality Investigation** — Agent 2 detects all 15 defects
4. **Tab 3 → Generate Audit Report** — Agent 3 builds the reversible change log
5. Approve/Reject individual fixes, then click **Apply Approved Fixes**
6. **Tab 4** — View completeness heatmap and F1 benchmark scorecard

### Option B — Upload Your Own Rainfall File

Accepted formats: `.csv`, `.tsv`, `.txt`, `.xls`, `.xlsx`

| File type | Handled automatically |
|-----------|----------------------|
| UTF-8 BOM (utf-8-sig) | ✅ Stripped |
| CP1252 / Latin-1 encoding | ✅ Auto-detected |
| Tab / semicolon / pipe delimiters | ✅ Auto-detected |
| District-level aggregated data (one row per district) | ✅ Expanded with synthetic placeholder date; date-dependent checks labelled **N/A** |
| Column name variations (`Sl No`, `Rain(mm)`, `Obs_Date`, etc.) | ✅ Auto-mapped via alias dictionary |

---

## 🔒 Data Integrity Guarantees

| Guardrail | Mechanism |
|-----------|-----------|
| **Raw Immutability** | SHA-256 checksum of raw dataset registered at load time; verified before every write |
| **Input Validation** | File size ≤ 50 MB · extension allowlist · injection pattern detection (XSS, path traversal, null bytes) |
| **Extreme Preservation** | Rainfall events in [150, 400] mm are **never auto-deleted**; enforced post-fix by `ExtremePreservationGuardrail` |
| **Audit Provenance** | Full JSON audit trail with SHA-256 checksums, applied changes, synthesised fields, and station summary |
| **Human Gate** | Zero changes applied without explicit per-entry Approve/Reject decision |

---

## 🧪 Running the Tests

```bash
# Full 26-test master suite (fastest end-to-end check)
python test_full_suite.py

# End-to-end pipeline integration test
python integration_test.py

# Individual module tests
python test_uploader.py
python test_guardrails.py
python test_agents.py
python test_enhancements.py

# Smoke test: district upload + F1 benchmark
python smoke_test.py
```

Expected output for the master suite:
```
  TOTAL: 26 PASSED | 0 FAILED
  🎉 ALL TESTS PASSED!
```

---

## 🌐 Bilingual Support (English / ಕನ್ನಡ)

The Audit Report in Tab 3 can be switched between English and Kannada. IBM Granite 4-H-Small performs the translation on demand. The result is cached in session state so the LLM is only called once per audit cycle.

---

## 📄 License

This project was built for the **Inter-Institute IBM Bob Hackathon 2026** and is provided for educational and demonstration purposes.

---

<p align="center">
  Built with ❤️ using <strong>IBM Bob</strong> · <strong>IBM Granite 4-H-Small</strong> · <strong>Streamlit</strong><br/>
  Team Code Crusaders · Sambhram Institute of Technology, Bengaluru
</p>
