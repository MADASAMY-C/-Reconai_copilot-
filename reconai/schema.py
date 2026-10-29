"""Table definitions shared by the DB layer, ingestion and the AI agent (no heavy deps)."""

TABLES = ["payments", "refunds", "settlement_items", "settlements", "bank_credits"]

DATE_COLS = {
    "payments": ["created_at", "captured_at"],
    "refunds": ["created_at"],
    "settlement_items": ["settled_at"],
    "settlements": ["settled_at"],
    "bank_credits": ["credited_at"],
    "usage_events": ["ts"],
}

# Minimum columns an uploaded CSV must have.
REQUIRED = {
    "payments": ["payment_id", "order_id", "amount", "method", "status", "created_at", "captured_at"],
    "refunds": ["refund_id", "payment_id", "amount", "created_at"],
    "settlement_items": ["entity_id", "type", "order_id", "method", "credit", "debit", "fee", "tax",
                         "settlement_id", "settlement_utr", "settled_at"],
    "settlements": ["settlement_id", "amount", "fees", "tax", "utr", "settled_at"],
    "bank_credits": ["utr", "amount", "credited_at"],
}

# Optional columns filled with a default when absent.
OPTIONAL = {
    "payments": {"currency": "INR", "card_network": None, "issuer": None, "error_code": None,
                 "error_source": None, "error_reason": None, "error_description": None},
    "refunds": {"status": "processed"},
    "bank_credits": {"narration": ""},
}

# Integer paise columns.
NUM_COLS = {
    "payments": ["amount"],
    "refunds": ["amount"],
    "settlement_items": ["credit", "debit", "fee", "tax"],
    "settlements": ["amount", "fees", "tax"],
    "bank_credits": ["amount"],
}

SCHEMA_DOC = """\
All money columns are INTEGER PAISE (100 paise = Rs 1). Timestamps are naive IST.
payments(payment_id, order_id, amount, currency, status['captured'|'failed'|'refunded'], method['upi'|'card'|'netbanking'|'wallet'], card_network, issuer, error_code, error_source, error_reason, error_description, created_at, captured_at)
refunds(refund_id, payment_id, amount, created_at, status)
settlement_items(entity_id, type['payment'|'refund'], order_id, method, credit, debit, fee, tax, settlement_id, settlement_utr, settled_at)  -- Razorpay-style settlement recon report rows
settlements(settlement_id, amount[net paid out], fees, tax, utr, settled_at)
bank_credits(utr, amount, credited_at, narration)  -- what actually hit the bank account
Expected net of a settlement = SUM(credit) - SUM(debit) - SUM(fee) - SUM(tax) over its settlement_items.
"""
