"""Claude integration: (1) a tool-using Q&A agent over the data, (2) a structured-output digest writer.

Design: deterministic code computes every number; Claude only decides *which* tool to call and
explains/prioritises the results. That keeps financial answers grounded and auditable.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import config, db, failures, metrics, recon
from .schema import SCHEMA_DOC
from .sqlguard import safe_sql

MAX_STEPS = 8


class NoApiKey(RuntimeError):
    pass


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _client():
    if not available():
        raise NoApiKey("Set ANTHROPIC_API_KEY to enable AI features.")
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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
    """Run the tool-use loop. history = prior [{'role','content'}] text turns."""
    client = _client()
    tools = DataTools(engine or db.get_engine())
    messages = list(history or []) + [{"role": "user", "content": question}]
    calls, resp = [], None
    for _ in range(MAX_STEPS):
        resp = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=1500, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                out, err = tools.call(b.name, b.input), False
            except Exception as ex:  # let the model recover from bad SQL / args
                out, err = {"error": str(ex)}, True
            calls.append({"tool": b.name, "input": b.input, "error": err})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": _dumps(out)[:15000], "is_error": err})
        messages.append({"role": "user", "content": results})
    text = "".join(b.text for b in resp.content if b.type == "text").strip() if resp else ""
    return {"answer": text or "I couldn't finish that analysis - try a narrower question.", "tool_calls": calls}


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
    resp = _client().messages.create(
        model=config.CLAUDE_MODEL, max_tokens=1200, system=DIGEST_SYSTEM, tools=[DIGEST_TOOL],
        tool_choice={"type": "tool", "name": "submit_digest"},
        messages=[{"role": "user", "content": "Facts (JSON):\n" + _dumps(facts)}])
    for b in resp.content:
        if b.type == "tool_use":
            return dict(b.input)
    raise RuntimeError("Model returned no structured digest")
