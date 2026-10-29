"""Payment-failure analysis: where, when and why payments fail, ranked by money lost."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare(pay: pd.DataFrame) -> pd.DataFrame:
    p = pay.copy()
    p["failed"] = (p.status == "failed") | p.captured_at.isna()
    p["hour"] = p.created_at.dt.hour
    p["window"] = (p.hour // 2 * 2).map(lambda h: f"{h:02d}-{h + 2:02d}h")
    p["issuer"] = p.issuer.fillna("n/a").astype(str)
    return p


def failure_by(p: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    q = p.assign(fv=np.where(p.failed, p.amount, 0))
    g = q.groupby(cols, as_index=False).agg(attempts=("failed", "size"), failures=("failed", "sum"),
                                            failed_value=("fv", "sum"))
    g["failure_rate"] = g.failures / g.attempts
    g["failed_value_inr"] = g.failed_value / 100
    return g.drop(columns="failed_value")


def hotspots(p: pd.DataFrame, min_attempts: int = 40, min_lift: float = 1.4, top: int = 6) -> pd.DataFrame:
    """Segments failing well above the baseline, ranked by *excess* failures (not just rate)."""
    cols_out = ["segment", "attempts", "failures", "failure_rate", "baseline_rate", "lift",
                "excess_failures", "failed_value_inr", "top_error"]
    if p.empty or not p.failed.any():
        return pd.DataFrame(columns=cols_out)
    base = float(p.failed.mean())
    cands = []
    for cols in (["method"], ["window"], ["method", "window"], ["method", "issuer"]):
        g = failure_by(p, cols)
        if "issuer" in cols:
            g = g[g.issuer != "n/a"]
        fr = (p[p.failed].groupby(cols + ["error_reason"]).size().rename("n").reset_index()
              .sort_values("n", ascending=False).drop_duplicates(cols))
        g = g.merge(fr[cols + ["error_reason"]].rename(columns={"error_reason": "top_error"}), on=cols, how="left")
        for r in g.to_dict("records"):
            r["keys"] = {c: r[c] for c in cols}
            r["segment"] = " · ".join(str(r[c]) for c in cols)
            r["lift"] = r["failure_rate"] / base if base else 0
            r["excess_failures"] = r["failures"] - base * r["attempts"]
            r["baseline_rate"] = base
            if r["attempts"] >= min_attempts and r["lift"] >= min_lift and r["excess_failures"] >= 5:
                cands.append(r)
    cands.sort(key=lambda r: (-len(r["keys"]), -r["excess_failures"]))
    chosen = []
    for r in cands:  # drop broad segments already explained by a more specific one
        k = set(r["keys"].items())
        if any(k < set(s["keys"].items()) and s["excess_failures"] >= .6 * r["excess_failures"] for s in chosen):
            continue
        chosen.append(r)
    out = pd.DataFrame(chosen)
    if out.empty:
        return pd.DataFrame(columns=cols_out)
    return out.sort_values("excess_failures", ascending=False).head(top)[cols_out].reset_index(drop=True)
