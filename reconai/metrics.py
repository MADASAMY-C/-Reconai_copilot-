"""KPIs and trends over a date window. All money in the output is rupees (float)."""
from __future__ import annotations

import pandas as pd


def _bounds(start, end):
    s = pd.Timestamp(start).normalize()
    return s, pd.Timestamp(end).normalize() + pd.Timedelta(days=1)


def slice_window(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    s, e = _bounds(start, end)
    return df[(df[col] >= s) & (df[col] < e)]


def kpis(f: dict, start, end) -> dict:
    p = slice_window(f["payments"], "created_at", start, end)
    ok = p.captured_at.notna()
    r = slice_window(f["refunds"], "created_at", start, end)
    s = slice_window(f["settlements"], "settled_at", start, end)
    b = slice_window(f["bank_credits"], "credited_at", start, end)
    attempts = int(len(p))
    return {
        "attempts": attempts,
        "successful": int(ok.sum()),
        "success_rate": float(ok.mean()) if attempts else 0.0,
        "gmv_inr": float(p.amount[ok].sum() / 100),
        "failed_value_inr": float(p.amount[~ok].sum() / 100),
        "refunds_inr": float(r.amount.sum() / 100),
        "fees_and_tax_inr": float((s.fees.sum() + s.tax.sum()) / 100),
        "settled_inr": float(s.amount.sum() / 100),
        "bank_received_inr": float(b.amount.sum() / 100),
    }


def daily(f: dict, start, end) -> pd.DataFrame:
    p = slice_window(f["payments"], "created_at", start, end)
    q = p.assign(date=p.created_at.dt.normalize(), ok=p.captured_at.notna())
    q["gmv"] = q.amount.where(q.ok, 0) / 100
    g = q.groupby("date", as_index=False).agg(attempts=("ok", "size"), successes=("ok", "sum"), gmv_inr=("gmv", "sum"))
    g["success_rate"] = g.successes / g.attempts
    return g
