# 💸 ReconAI: AI payment-failure & reconciliation copilot
---
title: ReconAI
emoji: 💸
sdk: streamlit
app_port: 8501
---

Small online merchants lose hours every week working out **why payments failed** and **why their
settlement doesn't match their bank credit**. ReconAI explains both in plain language.

| | |
|---|---|
| **Failure analysis** | Finds the segments failing above baseline (method × time window × issuer), ranks them by *excess failures* and ₹ lost, names the top error reason. |
| **3-way reconciliation** | Payments → settlement report → settlements → bank statement. Catches 6 issue types: missing settlements, fee mismatches, settlement ≠ its own items, short bank credit, missing bank credit, duplicate captures. |
| **Ask AI** | Qwen **tool-use** agent (local, no API key): calls KPI / hotspot / mismatch / settlement-breakdown tools, or a guarded read-only SQL tool. Every tool call is shown for auditability. |
| **Weekly digest** | Qwen writes a **structured-output** digest (forced JSON tool schema); falls back to rules if the model is down. Runs by cron + email. |
| **Impact tab** | Logs sessions, questions, 👍/👎, digests → the adoption metrics a hiring manager asks about. |

## Architecture
```
CSV upload / synthetic generator ─► Postgres (SQLite fallback) ─► deterministic engines
   (Razorpay-style recon report)         │                      (recon.py, failures.py, metrics.py)
                                          │                                  │
                                          ▼                                  ▼
                                 Streamlit UI  ◄──── Qwen (tool use + structured output) ──── weekly cron
```
**Key design choice:** all numbers are computed by tested Python code; Qwen only chooses tools and
explains results. Financial answers stay grounded and auditable, and the AI can't silently mis-add.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env          # DATABASE_URL optional (for Postgres)
streamlit run app.py          # auto-seeds demo data on first run
# Postgres: docker compose up db   (then set DATABASE_URL from .env.example)
python -m reconai.digest --send   # weekly digest (HTML in ./out)
pytest -q                     # 14 tests; injected anomalies must be found exactly, 0 false positives
```
Demo data has known anomalies injected (e.g. a ₹4,200 short bank credit) so the product always has real findings. Try:
*“Why is my settlement short by ₹4,200?”* · *“Which payments were never settled?”* · *“Why are my card payments failing?”*

## Deploy (live link)
Streamlit needs a long-running server, so **Vercel (serverless) is not suitable**. Use:
- **Railway** (best): New Project → Deploy from GitHub (uses the Dockerfile) → add the Postgres plugin → `DATABASE_URL` is auto-injected. Generate a domain.
- **Hugging Face Spaces**: create a Docker Space, push this repo, and put this at the top of the Space's README: `---\ntitle: ReconAI\nsdk: docker\napp_port: 7860\n---`. Add a free Neon/Supabase `DATABASE_URL` as a secret (Spaces disk is not persistent). For file uploads, run Streamlit with `--server.enableXsrfProtection=false`.
- **Weekly digest**: `.github/workflows/weekly_digest.yml` (Mondays 9:00 IST) or a Railway cron service running `python -m reconai.digest --send`.

## Data & honesty notes
- Synthetic data **modelled on** Razorpay's public settlement-recon report and payment-entity fields (`entity_id, type, credit, debit, fee, tax, settlement_id, settlement_utr`, `error_code/source/reason`). It is representative, not an exact copy. For real merchants, export the report from the dashboard and upload.
- Simplifications: daily T+2 calendar-day settlements, one flat fee plan per method (`config.py`), one currency.
- Never present demo numbers as real results. Put measured adoption numbers in `docs/CASE_STUDY_TEMPLATE.md`.

## Layout
`app.py` UI · `reconai/{recon,failures,metrics}.py` engines · `ai_agent.py` Qwen tool-use + structured output · `sqlguard.py` read-only SQL guard · `digest.py` weekly job · `generate_data.py` synthetic data · `db.py` Postgres/SQLite · `tests/`
