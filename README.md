# 🌧️ Rain Check Karnataka — Agentic Rainfall Data Quality Investigator

[![IBM Granite](https://img.shields.io/badge/Model-IBM%20Granite%204--H--Small-blue.svg)](https://www.ibm.com/granite)
[![Multi-Agent](https://img.shields.io/badge/Architecture-3%20Specialized%20Agents-green.svg)]()
[![Tests](https://img.shields.io/badge/Integration%20Tests-26%2F26%20Passing-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/Benchmark%20F1-100%25-success.svg)]()
[![Platform](https://img.shields.io/badge/Platform-IBM%20Cloud%20watsonx.ai-orange.svg)]()

> **Inter-Institute IBM Bob Hackathon 2026**  
> **Problem Statement 18:** Climate Data Engineering, Data Reliability & Auditable Agentic Workflows  
> **Team:** Code Crusaders  
> **Members:** Mohan M & Kusuma  
> **Institution:** Sambhram Institute of Technology, Bengaluru  

---

## 📌 Executive Summary
Rainfall summaries across Karnataka frequently become unreliable when telemetry stations encounter sensor freezes, transmission packet drops, and transmission glitches. A missing observation is **not** zero rainfall, and an intense monsoon cloudburst is **not** a defect. 

**Rain Check Karnataka** is an enterprise-grade, auditable Agentic AI quality investigator that validates schemas, executes a 5-point deterministic verification matrix, enforces mathematical raw data immutability via SHA-256 cryptographic provenance, and generates reversible audit reports with Human-in-the-Loop governance.

---

## 🤖 Multi-Agent Architecture
            [Raw Rainfall Dataset (CSV / XLS)]
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │         Cryptographic Immutability Guardrail           │
  │        SHA-256 Hash Registered & Read-Only Locked      │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ AGENT 1: Schema & Provenance Agent                     │
  │ Model: IBM Granite 4-H-Small (watsonx.ai)              │
  │ Function: Proposes schema mappings, verifies station   │
  │ IDs, date formats (YYYY-MM-DD), and millimetre units.  │
  └───────────────────────────┬────────────────────────────┘
                              │ (Validated Schema Handoff)
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ AGENT 2: Quality Investigation Agent                   │
  │ Engine: 5-Point Deterministic Matrix + IBM Granite     │
  │ Function: Scans duplicates, negatives, date gaps,      │
  │ conflicts, and separates outliers from real extremes.  │
  └───────────────────────────┬────────────────────────────┘
                              │ (Detected Anomalies)
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ AGENT 3: Audit Report & Governance Agent               │
  │ Model: IBM Granite 4-H-Small + Gemini / Groq           │
  │ Function: Produces reversible change log, generates    │
  │ Kannada (ಕನ್ನಡ) & English reports, and powers voice.   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │         Human-in-the-Loop Approval Barrier             │
  │  Approved fixes written ONLY to separate cleaned copy  │
  └────────────────────────────────────────────────────────┘
  
---

## 🛡️ The 5 Core Quality Checks
1. **Duplicates Check:** Detects identical records (same station, date, and rainfall value).
2. **Negative Values Check:** Identifies physically impossible negative rainfall readings (e.g. `-4.5 mm`).
3. **Expected-Date Gaps:** Flags gaps in the continuous observation calendar window.
4. **Conflicting Records:** Isolates instances where the same station records conflicting values for the same date.
5. **Outlier & Cloudburst Preservation:** Preserves real extreme events (e.g., `185.0 mm` Bengaluru cloudburst) while isolating sensor error placeholders (`9999.0 mm`).

---

## 📊 Verification & Benchmark Results
* **Integration Tests:** **26 / 26 passing** (`test_full_suite.py`, `test_uploader.py`, `test_guardrails.py`, `integration_test.py`).
* **Detection Precision:** **100%** (15 / 15 injected ground-truth defects detected).
* **Detection Recall:** **100%**.
* **F1 Score:** **100%**.
* **Legitimate Extremes Preserved:** **2 / 2** (185 mm BLR_CENTRAL, 192 mm MYS_CITY).
* **Cost Efficiency:** **$0.0014** per complete multi-agent audit run.

---

## 💻 Tech Stack
* **Development Environment:** IBM Bob (Agent Mode)
* **Foundation Models:** IBM Granite 4 H Small (`ibm/granite-4-h-small`) via IBM Cloud watsonx.ai
* **Fallback Gateways:** Groq (Qwen/Llama) and Google Gemini (2.5 Flash)
* **Frontend:** Pure Python + Streamlit with modern CSS3 glassmorphic design
* **Interactive Features:** HTML5 semantic containers, Web Speech API Voice Briefing (`window.speechSynthesis`), and Geospatial Karnataka Station Map.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Mohan-M04/CodeCrusaders-IBM-AgenticAI.git
cd CodeCrusaders-IBM-AgenticAI