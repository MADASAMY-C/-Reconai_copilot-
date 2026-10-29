"""Guard for LLM-written SQL: single read-only SELECT over whitelisted tables, row-capped."""
import re

from .schema import TABLES

ALLOWED_TABLES = set(TABLES)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|pragma|copy|"
    r"vacuum|merge|set|execute|information_schema|sqlite_master)\b|\bpg_\w+", re.I)


def safe_sql(query: str, limit: int = 200) -> str:
    q = (query or "").strip().rstrip(";").strip()
    if not q:
        raise ValueError("Empty query.")
    if ";" in q:
        raise ValueError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if "--" in q or "/*" in q:
        raise ValueError("Comments are not allowed.")
    m = FORBIDDEN.search(q)
    if m:
        raise ValueError(f"Forbidden keyword: {m.group(0)}")
    ctes = {c.lower() for c in re.findall(r"\b(\w+)\s+as\s*\(", q, re.I)}
    for ref in re.findall(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", q, re.I):
        name = ref.lower()
        if name not in ALLOWED_TABLES and name not in ctes:
            raise ValueError(f"Table not allowed: {ref}. Allowed: {sorted(ALLOWED_TABLES)}")
    return f"SELECT * FROM ({q}) AS _q LIMIT {int(limit)}"
