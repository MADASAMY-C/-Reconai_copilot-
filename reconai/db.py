"""Database layer: Postgres in production, SQLite fallback for zero-config local runs."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from . import config
from .schema import DATE_COLS, NUM_COLS, OPTIONAL, REQUIRED, TABLES

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_pay_created ON payments(created_at)",
    "CREATE INDEX IF NOT EXISTS ix_pay_order ON payments(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_pay_id ON payments(payment_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_setl ON settlement_items(settlement_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_entity ON settlement_items(entity_id)",
    "CREATE INDEX IF NOT EXISTS ix_setl_utr ON settlements(utr)",
    "CREATE INDEX IF NOT EXISTS ix_bank_utr ON bank_credits(utr)",
]


@lru_cache(maxsize=1)
def get_engine():
    url = config.DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, pool_pre_ping=True)


def is_postgres(engine) -> bool:
    return engine.dialect.name == "postgresql"


def read_table(name: str, engine=None) -> pd.DataFrame:
    e = engine or get_engine()
    return pd.read_sql(text(f"SELECT * FROM {name}"), e, parse_dates=DATE_COLS.get(name))


def read_all(engine=None) -> dict:
    return {t: read_table(t, engine) for t in TABLES}


def has_data(engine=None) -> bool:
    e = engine or get_engine()
    if "payments" not in inspect(e).get_table_names():
        return False
    with e.connect() as c:
        return (c.execute(text("SELECT COUNT(*) FROM payments")).scalar() or 0) > 0


def load_frames(frames: dict, engine=None) -> None:
    """Replace all business tables atomically (usage_events is left untouched)."""
    e = engine or get_engine()
    with e.begin() as conn:
        for t in TABLES:
            frames[t].to_sql(t, conn, if_exists="replace", index=False, chunksize=1000)
        for ddl in INDEXES:
            conn.execute(text(ddl))


def ingest(raw: dict, engine=None) -> None:
    """Validate user-uploaded CSV DataFrames, coerce types, then load them."""
    clean = {}
    for t in TABLES:
        df = raw.get(t)
        if df is None:
            if t == "refunds":
                df = pd.DataFrame(columns=REQUIRED["refunds"])
            else:
                raise ValueError(f"Missing file: {t}.csv")
        missing = [c for c in REQUIRED[t] if c not in df.columns]
        if missing:
            raise ValueError(f"{t}.csv is missing columns: {missing}")
        df = df.copy()
        for c, default in OPTIONAL.get(t, {}).items():
            if c not in df.columns:
                df[c] = default
        for c in DATE_COLS[t]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in NUM_COLS[t]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().astype("int64")
        clean[t] = df
    load_frames(clean, engine)


# ---- adoption tracking ---------------------------------------------------------------------
def log_event(session_id: str, event: str, detail: str = "", engine=None) -> None:
    try:
        pd.DataFrame([{"ts": pd.Timestamp.now(), "session_id": session_id, "event": event,
                       "detail": str(detail)[:500]}]).to_sql("usage_events", engine or get_engine(),
                                                            if_exists="append", index=False)
    except Exception:  # analytics must never break the product
        pass


def read_events(engine=None) -> pd.DataFrame:
    try:
        return read_table("usage_events", engine)
    except Exception:
        return pd.DataFrame(columns=["ts", "session_id", "event", "detail"])
