"""Synthetic merchant data modelled on Razorpay's public API / settlement-recon report shapes.

* Amounts are integer paise, IDs look like Razorpay's (pay_*, order_*, rfnd_*, setl_*).
* Settlements are daily, T+2 (calendar days) - a deliberate simplification.
* Realistic failure patterns AND reconciliation anomalies are injected on purpose so the
  product has real things to find. `truth` records exactly what was injected (used by tests).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from . import config
from .schema import TABLES

ALNUM = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
HOUR_W = np.array([1, .5, .4, .3, .3, .5, 1, 2, 3, 4, 4, 4.5, 5, 4.5, 4, 4, 4.5, 5, 6, 7, 8, 8, 6, 3], float)
HOUR_W /= HOUR_W.sum()
METHODS, METHOD_P = ["upi", "card", "netbanking", "wallet"], [.55, .30, .10, .05]
BASE_FAIL = {"upi": .04, "card": .07, "netbanking": .09, "wallet": .05}
ISSUERS, ISSUER_P = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"], [.30, .25, .20, .15, .10]
NETWORKS = ["Visa", "Mastercard", "RuPay"]

BANK_TIMEOUT = ("bank_technical_error", "bank", "Bank / UPI app timed out while authorising the payment")
ISSUER_DECLINE = ("card_declined_by_issuer", "bank", "Card declined by the issuing bank")
REASONS = {  # (reason, source, description, weight)
    "upi": [("insufficient_funds", "customer", "Account has insufficient balance", .35),
            ("incorrect_pin", "customer", "Customer entered the wrong UPI PIN", .35),
            ("customer_cancelled", "customer", "Customer cancelled the payment", .30)],
    "card": [("authentication_failed", "customer", "3DS / OTP authentication failed", .40),
             ("insufficient_funds", "customer", "Insufficient credit limit or balance", .35),
             ("incorrect_otp", "customer", "Customer entered an incorrect OTP", .25)],
    "netbanking": [("bank_technical_error", "bank", "Bank page failed to respond", .50),
                   ("customer_cancelled", "customer", "Customer closed the bank page", .50)],
    "wallet": [("insufficient_funds", "customer", "Wallet balance too low", .60),
               ("customer_cancelled", "customer", "Customer cancelled the payment", .40)],
}


def _rid(rng, prefix, n=14):
    return prefix + "".join(rng.choice(ALNUM, n))


def _reason(rng, method, upi_peak, hdfc):
    if upi_peak and rng.random() < .8:
        return BANK_TIMEOUT
    if hdfc and rng.random() < .8:
        return ISSUER_DECLINE
    opts = REASONS[method]
    w = np.array([o[3] for o in opts])
    return opts[int(rng.choice(len(opts), p=w / w.sum()))][:3]


def _payments(rng, start, end, per_day):
    days = pd.date_range(start, end, freq="D")
    parts = []
    for i, d in enumerate(days):
        n = int(rng.poisson(per_day * (1.15 if d.dayofweek >= 5 else 1.0)))
        hours = rng.choice(24, n, p=HOUR_W)
        created = d + pd.to_timedelta(hours * 3600 + rng.integers(0, 3600, n), unit="s")
        method = rng.choice(METHODS, n, p=METHOD_P)
        amount = (np.clip(np.exp(rng.normal(6.4, 0.9, n)), 20, 40000).round() * 100).astype("int64")
        has_issuer = np.isin(method, ["card", "netbanking"])
        issuer = np.where(has_issuer, rng.choice(ISSUERS, n, p=ISSUER_P), None)
        network = np.where(method == "card", rng.choice(NETWORKS, n, p=[.5, .3, .2]), None)
        p = pd.Series(method).map(BASE_FAIL).to_numpy()
        upi_peak = (method == "upi") & np.isin(hours, [20, 21])             # bank timeouts 8-10 PM
        hdfc = (method == "card") & (issuer == "HDFC") & (i >= len(days) - 21)  # recent HDFC issue
        p = np.where(upi_peak, .16, np.where(hdfc, .22, p))
        failed = rng.random(n) < p
        reasons = [_reason(rng, m, u, h) if f else (None, None, None)
                   for m, u, h, f in zip(method, upi_peak, hdfc, failed)]
        captured = pd.Series(created + pd.to_timedelta(rng.integers(2, 30, n), unit="s")).where(~failed)
        parts.append(pd.DataFrame({
            "payment_id": [_rid(rng, "pay_") for _ in range(n)],
            "order_id": [_rid(rng, "order_") for _ in range(n)],
            "amount": amount, "currency": "INR",
            "status": np.where(failed, "failed", "captured"),
            "method": method, "card_network": network, "issuer": issuer,
            "error_code": [("GATEWAY_ERROR" if r[1] == "bank" else "BAD_REQUEST_ERROR") if r[0] else None
                           for r in reasons],
            "error_source": [r[1] for r in reasons],
            "error_reason": [r[0] for r in reasons],
            "error_description": [r[2] for r in reasons],
            "created_at": created, "captured_at": captured,
        }))
    return pd.concat(parts, ignore_index=True)


def generate(days: int = 60, per_day: int = 110, seed: int = 42, end=None):
    """Return (frames, truth). frames maps table name -> DataFrame."""
    assert days >= 30, "need at least 30 days so anomalies can be injected"
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end if end is not None else pd.Timestamp.today()).normalize()
    start = end - pd.Timedelta(days=days - 1)
    pay = _payments(rng, start, end, per_day)
    truth: dict = {}

    # ---- choose payments to corrupt -------------------------------------------------------
    cap = pay[pay.captured_at.notna()]
    win = cap[(cap.captured_at >= end - pd.Timedelta(days=25)) & (cap.captured_at < end - pd.Timedelta(days=5))]
    elig = win[(win.method == "card") & (win.amount >= 50000)]
    fee_ids = rng.choice(elig.payment_id.to_numpy(), 12, replace=False)
    rest = win[~win.payment_id.isin(fee_ids)]
    picks = rng.choice(rest.payment_id.to_numpy(), 10, replace=False)
    missing_ids, dup_src = list(picks[:7]), list(picks[7:])
    truth.update(fee_mismatch=list(fee_ids), missing_settlement=missing_ids)

    dups = pay[pay.payment_id.isin(dup_src)].copy()
    dups["payment_id"] = [_rid(rng, "pay_") for _ in range(len(dups))]
    dups["created_at"] = dups.created_at + pd.Timedelta(minutes=3)
    dups["captured_at"] = dups.captured_at + pd.Timedelta(minutes=3)
    pay = pd.concat([pay, dups], ignore_index=True)
    truth["duplicate_capture"] = list(dups.payment_id)

    # ---- refunds --------------------------------------------------------------------------
    excluded = set(missing_ids) | set(dup_src) | set(dups.payment_id)
    el = pay[pay.captured_at.notna() & ~pay.payment_id.isin(excluded)]
    rf = el[rng.random(len(el)) < .025].copy()
    off = (pd.to_timedelta(rng.integers(1, 8, len(rf)), unit="D")
           + pd.to_timedelta(rng.integers(0, 86400, len(rf)), unit="s"))
    rf["r_created"] = rf.captured_at + off
    rf = rf[rf.r_created < end + pd.Timedelta(days=1)].copy()
    full = rng.random(len(rf)) < .7
    rf["r_amount"] = np.where(full, rf.amount, (rf.amount // 200) * 100).astype("int64")
    rf["refund_id"] = [_rid(rng, "rfnd_") for _ in range(len(rf))]
    pay.loc[pay.payment_id.isin(rf.payment_id[full]), "status"] = "refunded"
    refunds = pd.DataFrame({"refund_id": rf.refund_id.to_numpy(), "payment_id": rf.payment_id.to_numpy(),
                            "amount": rf.r_amount.to_numpy(), "created_at": rf.r_created.to_numpy(),
                            "status": "processed"})

    # ---- settlement recon report rows ----------------------------------------------------
    c2 = pay[pay.captured_at.notna()]
    amt = c2.amount.to_numpy()
    fee = (c2.amount * c2.method.map(config.FEE_RATES)).round().to_numpy()
    bad = c2.payment_id.isin(fee_ids).to_numpy()
    fee = np.where(bad, np.round(amt * 0.03), fee).astype("int64")   # charged 3% instead of 2%
    ip = pd.DataFrame({"entity_id": c2.payment_id.to_numpy(), "type": "payment",
                       "order_id": c2.order_id.to_numpy(), "method": c2.method.to_numpy(),
                       "credit": amt, "debit": 0, "fee": fee,
                       "event_date": c2.captured_at.dt.normalize().to_numpy()})
    ip["tax"] = (ip.fee * config.GST_RATE).round().astype("int64")
    ir = pd.DataFrame({"entity_id": rf.refund_id.to_numpy(), "type": "refund",
                       "order_id": rf.order_id.to_numpy(), "method": rf.method.to_numpy(),
                       "credit": 0, "debit": rf.r_amount.to_numpy(), "fee": 0, "tax": 0,
                       "event_date": rf.r_created.dt.normalize().to_numpy()})
    items = pd.concat([ip, ir], ignore_index=True)
    items = items[~items.entity_id.isin(missing_ids)]                       # never settled
    items = items[items.event_date <= end - pd.Timedelta(days=2)].copy()   # T+2 not reached yet

    dates = sorted(items.event_date.drop_duplicates())
    sid = {d: _rid(rng, "setl_") for d in dates}
    utr = {d: f"RZRP{int(rng.integers(10**11, 10**12))}" for d in dates}
    items["settlement_id"] = items.event_date.map(sid)
    items["settlement_utr"] = items.event_date.map(utr)
    items["settled_at"] = items.event_date + pd.Timedelta(days=2, hours=11)
    g = items.groupby("settlement_id", as_index=False).agg(
        credit=("credit", "sum"), debit=("debit", "sum"), fees=("fee", "sum"), tax=("tax", "sum"),
        utr=("settlement_utr", "first"), settled_at=("settled_at", "first"))
    g["amount"] = g.credit - g.debit - g.fees - g.tax
    settlements = g[["settlement_id", "amount", "fees", "tax", "utr", "settled_at"]].copy()
    items = items.drop(columns="event_date")

    # ---- settlement / bank anomalies ------------------------------------------------------
    def idx(days_ago):
        m = settlements.settled_at.dt.normalize() == end - pd.Timedelta(days=days_ago)
        return settlements.index[m][0]

    i_short, i_miss, i_tot, i_short2 = idx(3), idx(7), idx(13), idx(18)
    settlements.loc[i_tot, "amount"] -= 85_000        # settlement paid Rs 850 less than its items
    bank = settlements[["utr", "amount", "settled_at"]].copy()
    bank["credited_at"] = bank.settled_at + pd.Timedelta(hours=1, minutes=17)
    bank.loc[i_short, "amount"] -= 420_000            # bank credited Rs 4,200 short
    bank.loc[i_short2, "amount"] -= 138_000           # and Rs 1,380 short
    truth.update(short_bank_credit=[settlements.loc[i_short, "settlement_id"], settlements.loc[i_short2, "settlement_id"]],
                 missing_bank_credit=[settlements.loc[i_miss, "settlement_id"]],
                 settlement_total_mismatch=[settlements.loc[i_tot, "settlement_id"]])
    bank = bank.drop(index=i_miss)                    # settlement never reached the bank
    bank["narration"] = "NEFT-" + bank.utr + "-RAZORPAY SOFTWARE PVT LTD"
    bank = bank.drop(columns="settled_at")

    payments = pay[["payment_id", "order_id", "amount", "currency", "status", "method", "card_network",
                    "issuer", "error_code", "error_source", "error_reason", "error_description",
                    "created_at", "captured_at"]].sort_values("created_at").reset_index(drop=True)
    frames = {"payments": payments, "refunds": refunds,
              "settlement_items": items[["entity_id", "type", "order_id", "method", "credit", "debit", "fee",
                                         "tax", "settlement_id", "settlement_utr", "settled_at"]].reset_index(drop=True),
              "settlements": settlements.sort_values("settled_at").reset_index(drop=True),
              "bank_credits": bank.sort_values("credited_at").reset_index(drop=True)}
    return frames, truth


def to_zip_bytes(frames: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for t in TABLES:
            z.writestr(f"{t}.csv", frames[t].to_csv(index=False))
    return buf.getvalue()
