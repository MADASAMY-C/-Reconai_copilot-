import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from reconai import failures, metrics, recon
from reconai.generate_data import generate
from reconai.sqlguard import safe_sql


@pytest.fixture(scope="module")
def data():
    return generate(days=45, per_day=60, seed=7)


def test_reconciliation_finds_every_injected_issue(data):
    f, truth = data
    mm = recon.reconcile(f)
    for check, ids in truth.items():
        got = set(mm[mm["check"] == check].entity_id)
        assert got == set(ids), f"{check}: expected {set(ids)}, got {got}"


def test_clean_data_has_no_false_positives(data):
    f, truth = data
    mm = recon.reconcile(f)
    assert len(mm) == sum(len(v) for v in truth.values())


def test_short_credit_amounts(data):
    f, _ = data
    mm = recon.reconcile(f)
    risks = sorted(mm[mm["check"] == "short_bank_credit"].amount_at_risk_inr)
    assert risks == [1380.0, 4200.0]


def test_waterfall_explains_short_credit(data):
    f, truth = data
    w = recon.settlement_waterfall(f, truth["short_bank_credit"][0])
    assert w["bank_vs_settlement_inr"] == -4200.0 and w["settlement_vs_expected_inr"] == 0


def test_hotspots_find_upi_peak_and_hdfc(data):
    f, _ = data
    h = failures.hotspots(failures.prepare(f["payments"]))
    segs = " | ".join(h.segment)
    assert "upi" in segs and "HDFC" in segs


def test_kpis(data):
    f, _ = data
    end = f["payments"].created_at.max()
    k = metrics.kpis(f, end - pd.Timedelta(days=6), end)
    assert 0.85 < k["success_rate"] < 0.99 and k["gmv_inr"] > 0


@pytest.mark.parametrize("q", ["DELETE FROM payments", "select 1; drop table payments", "select * from users",
                               "select * from payments -- x", "with x as (delete from payments returning *) select * from x",
                               "select pg_read_file('/etc/passwd')", "select * from information_schema.tables"])
def test_sql_guard_blocks(q):
    with pytest.raises(ValueError):
        safe_sql(q)


def test_sql_guard_allows_and_caps():
    sql = safe_sql("WITH t AS (SELECT method, COUNT(*) c FROM payments GROUP BY method) SELECT * FROM t;", limit=50)
    assert sql.endswith("LIMIT 50")
