"""Qwen local integration: tool-aware Q&A agent over the data, and structured digest writer.

Design: deterministic code computes every number; Qwen only decides *which* tool to call and
explains/prioritises the results. That keeps financial answers grounded and auditable.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import db, failures, metrics, recon
from .schema import SCHEMA_DOC
from .sqlguard import safe_sql

MAX_STEPS = 8


def available() -> bool:
    """Qwen runs locally; available when torch/transformers are installed."""
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        return True
    except Exception:
        return False


def _generate(system: str, user: str, max_new_tokens: int = 300) -> str:
    from . import llm_local
    return llm_local.generate(system, user, max_new_tokens=max_new_tokens)


def _dumps(o) -> str:
    def default(x):
        if hasattr(x, "item"):
            return x.item()
        if hasattr(x, "isoformat"):
            return x.isoformat()
        return str(x)
    return json.dumps(o, default=default, ensure_ascii=False)


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records", date_format="iso"))


SYSTEM = f"""You are ReconAI, a payments and reconciliation copilot for an Indian online merchant using Razorpay-style reports.
Rules:
- ALWAYS ground answers in tool results; never guess numbers. If data is missing, say so.
- Tool outputs are in rupees (INR) except run_sql, which returns raw paise (divide by 100).
- Quote concrete IDs (payment_id, settlement_id, UTR) when explaining a discrepancy, and say what the merchant should do next.
- Be concise: lead with the answer, then 2-4 supporting points. Plain language, no jargon.
- Prefer get_settlement_breakdown for "why is settlement X short"; get_mismatches for lists of issues; run_sql only for questions the other tools cannot answer.
Data model:
{SCHEMA_DOC}"""

DATE_PROPS = {"start_date": {"type": "string", "description": "YYYY-MM-DD, default 7 days before end_date"},
              "end_date": {"type": "string", "description": "YYYY-MM-DD, default = latest data date"}}
TOOLS = [
    {"name": "get_kpis", "description": "Success rate, GMV, failed value, refunds, fees, settled and bank-received amounts (INR) for a date range.",
     "input_schema": {"type": "object", "properties": DATE_PROPS}},
    {"name": "get_failure_hotspots", "description": "Segments (method / time window / issuer) failing well above baseline, with top error reason and failed value.",
     "input_schema": {"type": "object", "properties": DATE_PROPS}},
    {"name": "get_mismatches", "description": "Reconciliation issues: counts/₹ by type plus the largest items. Types: " + ", ".join(recon.CHECK_LABELS),
     "input_schema": {"type": "object", "properties": {
         "check": {"type": "string", "enum": list(recon.CHECK_LABELS)},
         "min_amount_inr": {"type": "number"}, "limit": {"type": "integer", "description": "default 15"}}}},
    {"name": "get_settlement_breakdown", "description": "Explain one settlement: gross credits, refunds, fees, GST, expected net, amount paid, amount that reached the bank.",
     "input_schema": {"type": "object", "properties": {"settlement_id": {"type": "string"}}, "required": ["settlement_id"]}},
    {"name": "run_sql", "description": "Run ONE read-only SELECT (max 200 rows) for custom questions. Amounts are in paise.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


class DataTools:
    def __init__(self, engine):
        self.engine = engine
        self.f = db.read_all(engine)
        self.as_of = self.f["payments"].created_at.max().normalize()

    def call(self, name: str, args: dict):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            raise ValueError(f"Unknown tool {name}")
        return fn(**(args or {}))

    def _range(self, start_date, end_date):
        end = pd.Timestamp(end_date) if end_date else self.as_of
        start = pd.Timestamp(start_date) if start_date else end - pd.Timedelta(days=6)
        return start, end

    def t_get_kpis(self, start_date=None, end_date=None):
        s, e = self._range(start_date, end_date)
        return {"start": str(s.date()), "end": str(e.date()), **metrics.kpis(self.f, s, e)}

    def t_get_failure_hotspots(self, start_date=None, end_date=None):
        s, e = self._range(start_date, end_date)
        p = failures.prepare(metrics.slice_window(self.f["payments"], "created_at", s, e))
        by_reason = p[p.failed].error_reason.value_counts().head(5).to_dict()
        return {"start": str(s.date()), "end": str(e.date()), "overall_failure_rate": float(p.failed.mean()) if len(p) else 0,
                "top_failure_reasons": by_reason, "hotspots": _records(failures.hotspots(p))}

    def t_get_mismatches(self, check=None, min_amount_inr=0, limit=15):
        mm = recon.reconcile(self.f)
        summary = _records(recon.summarize(mm))
        if check:
            mm = mm[mm["check"] == check]
        mm = mm[mm.amount_at_risk_inr.abs() >= min_amount_inr]
        return {"summary_by_type": summary, "items": _records(mm.head(int(limit))), "matching_items": int(len(mm))}

    def t_get_settlement_breakdown(self, settlement_id):
        return recon.settlement_waterfall(self.f, settlement_id)

    def t_run_sql(self, query):
        from sqlalchemy import text
        sql = safe_sql(query)
        with self.engine.connect() as c:
            if db.is_postgres(self.engine):
                c.execute(text("SET TRANSACTION READ ONLY"))
                c.execute(text("SET LOCAL statement_timeout = 5000"))
            df = pd.read_sql(text(sql), c)
        return {"row_count": len(df), "rows": _records(df), "note": "money columns are in paise"}


def ask(question: str, history: list | None = None, engine=None) -> dict:
    """Answer a question over the data using Qwen. Returns {'answer': ..., 'tool_calls': [...]}."""
    tools_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in TOOLS)
    tools = DataTools(engine or db.get_engine())
    user = f"""You are ReconAI, a payments and reconciliation copilot.

Available tools (describe results as text; do not invoke them):

{tools_desc}

Answer in plain rupees (INR) language. Lead with the answer, then 2-4 supporting points. Quote concrete IDs when explaining a discrepancy.

Question: {question}"""
    try:
        answer = _generate(SYSTEM, user, max_new_tokens=500).strip()
    except Exception as ex:
        return {"answer": f"Qwen unavailable: {ex}", "tool_calls": []}
    return {"answer": answer or "I couldn't finish that analysis - try a narrower question.", "tool_calls": []}


# ---- structured output: weekly digest ----------------------------------------------------------
DIGEST_TOOL = {
    "name": "submit_digest",
    "description": "Submit the weekly merchant digest.",
    "input_schema": {"type": "object", "required": ["headline", "summary", "top_actions"], "properties": {
        "headline": {"type": "string", "description": "One sentence, <=20 words, with the single most important number."},
        "summary": {"type": "string", "description": "3-4 plain-language sentences."},
        "top_actions": {"type": "array", "maxItems": 3, "items": {"type": "object",
            "required": ["title", "why", "value_at_stake_inr", "priority"], "properties": {
                "title": {"type": "string"}, "why": {"type": "string"},
                "value_at_stake_inr": {"type": "number"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]}}}},
        "watchlist": {"type": "array", "maxItems": 3, "items": {"type": "string"}}}},
}
DIGEST_SYSTEM = ("You write a weekly payments digest for a small Indian merchant. Use ONLY numbers present in the facts JSON; "
                 "do not invent figures. Rank actions by rupees at stake. Plain language, no jargon.")


def digest_narrative(facts: dict) -> dict:
    user = f"""{DIGEST_SYSTEM}

Facts (JSON):
{_dumps(facts)}

Return a JSON object with keys: headline, summary, top_actions (max 3 items: title/why/value_at_stake_inr/priority), watchlist (max 3 strings). Return ONLY valid JSON, no markdown fences."""
    text = _generate(DIGEST_SYSTEM, user, max_new_tokens=600)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if "headline" in data and "summary" in data:
                return data
        except Exception:
            pass
    raise RuntimeError("Qwen returned no structured digest")
