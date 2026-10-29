"""Three-way reconciliation: payments -> settlement report -> settlements -> bank statement."""
from __future__ import annotations

import pandas as pd

from . import config

CHECK_LABELS = {
    "missing_settlement": "Captured but never settled",
    "fee_mismatch": "Fee higher/lower than plan",
    "settlement_total_mismatch": "Settlement ≠ sum of its items",
    "short_bank_credit": "Bank credited less than settled",
    "missing_bank_credit": "Settlement never reached bank",
    "duplicate_capture": "Duplicate capture (double charge)",
}
CHECK_ACTIONS = {
    "missing_settlement": "Raise a ticket with Razorpay support with these payment IDs and capture dates.",
    "fee_mismatch": "Compare the charged MDR with your pricing agreement and request a fee correction.",
    "settlement_total_mismatch": "Ask support for the settlement breakdown; the payout total does not add up.",
    "short_bank_credit": "Check the bank statement for deductions (chargebacks, adjustments) against the UTR.",
    "missing_bank_credit": "Trace the UTR with your bank; confirm whether the payout was returned or held.",
    "duplicate_capture": "Refund the duplicate payment to the customer before it becomes a dispute.",
}
COLS = ["check", "severity", "entity_id", "settlement_id", "date", "amount_at_risk_inr", "detail"]


def _severity(paise: float) -> str:
    inr = abs(paise) / 100
    return "high" if inr >= 5000 else "medium" if inr >= 500 else "low"


def _frame(check, ids, sids, dates, risk_paise, details) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({"check": [check] * n, "entity_id": list(ids), "settlement_id": list(sids),
                         "date": pd.to_datetime(list(dates)), "amount_at_risk_paise": list(risk_paise),
                         "detail": list(details)})


def reconcile(f: dict) -> pd.DataFrame:
    pay, items, sett, bank = f["payments"], f["settlement_items"], f["settlements"], f["bank_credits"]
    if pay.empty:
        return pd.DataFrame(columns=COLS)
    tol = config.TOLERANCE_PAISE
    as_of = pay.created_at.max().normalize()
    sla = pd.Timedelta(days=config.SETTLEMENT_SLA_DAYS)
    cap = pay[pay.captured_at.notna()]
    parts = []

    # 1. captured payments that never appear in any settlement after the SLA
    due = cap[cap.captured_at.dt.normalize() + sla < as_of]
    miss = due[~due.payment_id.isin(items.entity_id)]
    parts.append(_frame("missing_settlement", miss.payment_id, [None] * len(miss),
                        miss.captured_at.dt.normalize(), miss.amount,
                        [f"₹{a/100:,.2f} captured on {d:%d %b} via {m}; not in any settlement after "
                         f"{config.SETTLEMENT_SLA_DAYS} days" for a, d, m in zip(miss.amount, miss.captured_at, miss.method)]))

    # 2. fee charged vs plan
    pi = items[items.type == "payment"].copy()
    rate = pi.method.map(config.FEE_RATES).fillna(0.0)
    pi["expected_fee"] = (pi.credit * rate).round()
    pi["diff"] = pi.fee - pi.expected_fee
    bad = pi[pi["diff"].abs() > tol]
    parts.append(_frame("fee_mismatch", bad.entity_id, bad.settlement_id, bad.settled_at.dt.normalize(),
                        (bad["diff"] * (1 + config.GST_RATE)).round(),
                        [f"Fee ₹{fe/100:,.2f} ({fe/c:.2%}) vs expected ₹{ex/100:,.2f} ({ex/c:.2%}) on {m} payment"
                         for fe, ex, c, m in zip(bad.fee, bad.expected_fee, bad.credit, bad.method)]))

    # 3. settlement total vs the sum of its own items
    net = (items.assign(net=items.credit - items.debit - items.fee - items.tax)
           .groupby("settlement_id").net.sum().rename("items_net").reset_index())
    t = sett.merge(net, on="settlement_id", how="left")
    t["diff"] = t.items_net - t.amount
    t = t[t["diff"].abs() > tol]
    parts.append(_frame("settlement_total_mismatch", t.settlement_id, t.settlement_id, t.settled_at.dt.normalize(),
                        t["diff"], [f"Items net to ₹{a/100:,.2f} but settlement paid ₹{b/100:,.2f}"
                                    for a, b in zip(t.items_net, t.amount)]))

    # 4/5. settlement vs bank statement
    m = sett.merge(bank[["utr", "amount", "credited_at"]].rename(columns={"amount": "bank_amount"}),
                   on="utr", how="left")
    short = m[m.bank_amount.notna() & ((m.amount - m.bank_amount).abs() > tol)]
    parts.append(_frame("short_bank_credit", short.settlement_id, short.settlement_id, short.settled_at.dt.normalize(),
                        short.amount - short.bank_amount,
                        [f"UTR {u}: settled ₹{a/100:,.2f}, bank received ₹{b/100:,.2f}"
                         for u, a, b in zip(short.utr, short.amount, short.bank_amount)]))
    nob = m[m.bank_amount.isna() & (m.settled_at.dt.normalize() + pd.Timedelta(days=1) < as_of)]
    parts.append(_frame("missing_bank_credit", nob.settlement_id, nob.settlement_id, nob.settled_at.dt.normalize(),
                        nob.amount, [f"UTR {u}: ₹{a/100:,.2f} settled on {d:%d %b}, no matching bank credit"
                                     for u, a, d in zip(nob.utr, nob.amount, nob.settled_at)]))

    # 6. same order captured more than once
    d = cap.sort_values("captured_at")
    dup = d[d.duplicated("order_id", keep="first")]
    parts.append(_frame("duplicate_capture", dup.payment_id, [None] * len(dup), dup.captured_at.dt.normalize(),
                        dup.amount, [f"Order {o} captured again (₹{a/100:,.2f}) - likely double charge"
                                     for o, a in zip(dup.order_id, dup.amount)]))

    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else pd.DataFrame(columns=COLS + ["amount_at_risk_paise"])
    if out.empty:
        return pd.DataFrame(columns=COLS)
    out["severity"] = out.amount_at_risk_paise.map(_severity)
    out["amount_at_risk_inr"] = (out.amount_at_risk_paise / 100).round(2)
    out = out.sort_values("amount_at_risk_inr", key=lambda s: s.abs(), ascending=False)
    return out[COLS].reset_index(drop=True)


def summarize(mm: pd.DataFrame) -> pd.DataFrame:
    if mm.empty:
        return pd.DataFrame(columns=["check", "issue", "count", "amount_at_risk_inr"])
    g = mm.groupby("check", as_index=False).agg(count=("entity_id", "size"), amount_at_risk_inr=("amount_at_risk_inr", "sum"))
    g["issue"] = g["check"].map(CHECK_LABELS)
    return g.sort_values("amount_at_risk_inr", key=lambda s: s.abs(), ascending=False)[["check", "issue", "count", "amount_at_risk_inr"]]


def settlement_waterfall(f: dict, settlement_id: str) -> dict:
    s = f["settlements"][f["settlements"].settlement_id == settlement_id]
    if s.empty:
        return {"error": f"Settlement {settlement_id} not found"}
    s = s.iloc[0]
    it = f["settlement_items"][f["settlement_items"].settlement_id == settlement_id]
    b = f["bank_credits"][f["bank_credits"].utr == s.utr]
    gross, refunds = int(it.credit.sum()), int(it.debit.sum())
    fees, tax = int(it.fee.sum()), int(it.tax.sum())
    expected = gross - refunds - fees - tax
    bank_amt = int(b.amount.iloc[0]) if len(b) else None
    r = lambda x: None if x is None else round(x / 100, 2)  # noqa: E731
    return {"settlement_id": settlement_id, "utr": s.utr, "settled_at": str(s.settled_at),
            "payment_items": int((it.type == "payment").sum()), "refund_items": int((it.type == "refund").sum()),
            "gross_credits_inr": r(gross), "refunds_inr": r(refunds), "fees_inr": r(fees), "gst_on_fees_inr": r(tax),
            "expected_net_inr": r(expected), "settlement_amount_inr": r(int(s.amount)),
            "bank_amount_inr": r(bank_amt),
            "settlement_vs_expected_inr": r(int(s.amount) - expected),
            "bank_vs_settlement_inr": None if bank_amt is None else r(bank_amt - int(s.amount))}
