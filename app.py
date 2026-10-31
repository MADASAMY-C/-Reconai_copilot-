"""ReconAI - AI payment-failure & reconciliation copilot (Streamlit)."""
from __future__ import annotations

import sys

# Python 3.14's typing.TypedDict rejects closed=True, which Altair's generated
# schema wrappers use. Fall back to typing_extensions.TypedDict (supports it)
# before Altair imports.
if sys.version_info >= (3, 14):
    import typing
    import typing_extensions

    typing.TypedDict = typing_extensions.TypedDict  # noqa: B010

import uuid

import altair as alt
import pandas as pd
import streamlit as st

from reconai import ai_agent, db, digest, failures, generate_data, metrics, recon, seed

st.set_page_config(page_title="ReconAI · Payments copilot", page_icon="💸", layout="wide")


@st.cache_resource(show_spinner="Preparing database…")
def get_engine():
    e = db.get_engine()
    seed.ensure_demo_data(e)  # first boot on a fresh DB -> live demo works instantly
    return e


ENGINE = get_engine()
if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:12]
    db.log_event(st.session_state.sid, "session_start", engine=ENGINE)
SID = st.session_state.sid


def log(event: str, detail: str = ""):
    db.log_event(SID, event, detail, ENGINE)


@st.cache_data(ttl=600, show_spinner=False)
def frames() -> dict:
    return db.read_all(ENGINE)


@st.cache_data(ttl=600, show_spinner=False)
def mismatches() -> pd.DataFrame:
    return recon.reconcile(frames())


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# ------------------------------------------------------------------ sidebar
F = frames()
as_of = F["payments"].created_at.max().normalize()
with st.sidebar:
    st.title("💸 ReconAI")
    st.caption("Payment failures & reconciliation, explained.")
    days = st.radio("Period", [7, 14, 30], format_func=lambda d: f"Last {d} days")
    st.caption(f"Data through **{as_of:%d %b %Y}** · DB: `{ENGINE.dialect.name}`")
    if ai_agent.available():
        st.success("Qwen connected")
    else:
        st.warning("Qwen not available: AI chat off, rule-based digest on.")
    with st.expander("📂 Use your own data"):
        st.caption("Upload Razorpay-style CSVs (amounts in paise). Download the demo zip for the exact column layout.")
        st.download_button("Demo CSVs (zip)", generate_data.to_zip_bytes(F), "reconai_demo_csvs.zip")
        ups = {t: st.file_uploader(f"{t}.csv", type="csv", key=f"up_{t}") for t in db.TABLES}
        if st.button("Load uploaded CSVs"):
            try:
                db.ingest({t: pd.read_csv(u) for t, u in ups.items() if u is not None}, ENGINE)
                log("data_uploaded")
                st.cache_data.clear()
                st.rerun()
            except Exception as ex:
                st.error(str(ex))
        if st.button("Reset to demo data"):
            seed.ensure_demo_data(ENGINE, force=True)
            st.cache_data.clear()
            st.rerun()

start = as_of - pd.Timedelta(days=days - 1)
pstart, pend = start - pd.Timedelta(days=days), start - pd.Timedelta(days=1)
MM = mismatches()

tabs = st.tabs(["📊 Overview", "❌ Payment failures", "🧾 Reconciliation", "💬 Ask AI", "📬 Weekly digest", "📈 Impact"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    cur, prev = metrics.kpis(F, start, as_of), metrics.kpis(F, pstart, pend)
    c = st.columns(5)
    c[0].metric("GMV", inr(cur["gmv_inr"]), f"{(cur['gmv_inr'] / prev['gmv_inr'] - 1) * 100:+.1f}%" if prev["gmv_inr"] else None)
    c[1].metric("Success rate", f"{cur['success_rate']:.1%}", f"{(cur['success_rate'] - prev['success_rate']) * 100:+.1f} pp")
    c[2].metric("Failed value", inr(cur["failed_value_inr"]), delta_color="inverse")
    c[3].metric("Net settled", inr(cur["settled_inr"]))
    c[4].metric("Flagged in recon", inr(MM.amount_at_risk_inr.abs().sum()), f"{len(MM)} issues", delta_color="off")
    d = metrics.daily(F, start, as_of).set_index("date")
    a, b = st.columns(2)
    a.subheader("Daily GMV (₹)")
    b.subheader("Daily success rate")
    if d.empty or d["gmv_inr"].isna().all():
        a.info("No payment data in this period.")
        b.info("No payment data in this period.")
    else:
        a.bar_chart(d["gmv_inr"])
        b.line_chart(d["success_rate"])
    st.info("Tip: open **Ask AI** and try *“Why is my settlement short by ₹4,200?”*")

# ------------------------------------------------------------------ failures
with tabs[1]:
    P = failures.prepare(metrics.slice_window(F["payments"], "created_at", start, as_of))
    st.subheader("Where are payments failing?")
    hs = failures.hotspots(P)
    if hs.empty:
        st.success("No segment is failing meaningfully above baseline.")
    else:
        show = hs.assign(failure_rate=(hs.failure_rate * 100).round(1), baseline_rate=(hs.baseline_rate * 100).round(1),
                         lift=hs.lift.round(1), excess_failures=hs.excess_failures.round(0))
        st.dataframe(show.rename(columns={"failure_rate": "fail %", "baseline_rate": "baseline %", "failed_value_inr": "failed ₹"}),
                     hide_index=True, use_container_width=True)
    l, r = st.columns(2)
    hm = failures.failure_by(P, ["method", "hour"])
    l.subheader("Failure rate by method × hour")
    if hm.empty or hm["failure_rate"].isna().all():
        l.info("No failure data to chart.")
    else:
        l.altair_chart(alt.Chart(hm).mark_rect().encode(
            x=alt.X("hour:O"), y="method:N", color=alt.Color("failure_rate:Q", scale=alt.Scale(scheme="reds"), title="fail rate"),
            tooltip=["method", "hour", "attempts", alt.Tooltip("failure_rate:Q", format=".1%")]), use_container_width=True)
    r.subheader("Why they fail")
    failed_reasons = P[P.failed].error_reason.value_counts()
    if failed_reasons.empty:
        r.info("No failures in this period.")
    else:
        r.bar_chart(failed_reasons)
    if st.button("🤖 Explain these failures & suggest fixes", disabled=not ai_agent.available()):
        log("explain_failures")
        with st.spinner("Analysing…"):
            out = ai_agent.ask(f"Explain the payment failure hotspots between {start:%Y-%m-%d} and {as_of:%Y-%m-%d}, "
                               "what is likely causing them, and the top 3 fixes I can make as a merchant.", engine=ENGINE)
        st.markdown(out["answer"])

# ------------------------------------------------------------------ reconciliation
with tabs[2]:
    st.subheader("Reconciliation: payments → settlements → bank")
    summ = recon.summarize(MM)
    if MM.empty:
        st.success("Everything reconciles.")
    else:
        st.dataframe(summ.rename(columns={"amount_at_risk_inr": "₹ at risk"}), hide_index=True, use_container_width=True)
        f1, f2 = st.columns(2)
        checks = f1.multiselect("Issue type", list(recon.CHECK_LABELS), default=list(recon.CHECK_LABELS),
                                format_func=recon.CHECK_LABELS.get)
        sev = f2.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
        view = MM[MM["check"].isin(checks) & MM.severity.isin(sev)]
        st.dataframe(view, hide_index=True, use_container_width=True)
        if st.download_button("⬇️ Export issues (CSV)", view.to_csv(index=False), "reconai_issues.csv"):
            log("csv_exported")
        with st.expander("What to do about each type"):
            for k, v in recon.CHECK_ACTIONS.items():
                st.markdown(f"**{recon.CHECK_LABELS[k]}**: {v}")
    st.subheader("Settlement explainer")
    sids = F["settlements"].sort_values("settled_at", ascending=False).settlement_id.tolist()
    sid = st.selectbox("Settlement", sids)
    w = recon.settlement_waterfall(F, sid)
    rows = [("Gross payments", w["gross_credits_inr"]), ("− Refunds", -w["refunds_inr"]), ("− Fees", -w["fees_inr"]),
            ("− GST on fees", -w["gst_on_fees_inr"]), ("= Expected net", w["expected_net_inr"]),
            ("Settlement paid", w["settlement_amount_inr"]), ("Reached bank", w["bank_amount_inr"])]
    st.table(pd.DataFrame(rows, columns=["Step", "₹"]).fillna("— not received —"))
    if w["bank_vs_settlement_inr"]:
        st.error(f"Bank received ₹{abs(w['bank_vs_settlement_inr']):,.2f} {'less' if w['bank_vs_settlement_inr'] < 0 else 'more'} than settled.")
    elif w["bank_amount_inr"] is None:
        st.error("No bank credit found for this UTR.")
    if w["settlement_vs_expected_inr"]:
        st.error(f"Settlement differs from its own line items by ₹{w['settlement_vs_expected_inr']:,.2f}.")

# ------------------------------------------------------------------ ask AI
with tabs[3]:
    st.subheader("Ask your data anything")
    if not ai_agent.available():
        st.warning("Qwen not available: AI chat off, rule-based digest on.")
    chat = st.session_state.setdefault("chat", [])
    ideas = ["Why is my settlement short by ₹4,200?", "Which payments were captured but never settled?",
             "Why did my success rate drop recently?", "Am I being overcharged on fees?"]
    cols = st.columns(len(ideas))
    for col, q in zip(cols, ideas):
        if col.button(q, use_container_width=True, disabled=not ai_agent.available()):
            st.session_state.pending = q
    for i, m in enumerate(chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                with st.expander(f"🔧 {len(m['calls'])} tool call(s)"):
                    st.json(m["calls"])
                if m.get("fb"):
                    st.caption("Thanks for the feedback.")
                else:
                    u, dn, _ = st.columns([1, 1, 8])
                    if u.button("👍", key=f"up{i}") or dn.button("👎", key=f"dn{i}"):
                        m["fb"] = True
                        log("feedback_up" if st.session_state.get(f"up{i}") else "feedback_down", chat[i - 1]["content"])
                        st.rerun()
    q = st.chat_input("e.g. Which UPI payments failed most last night?", disabled=not ai_agent.available())
    q = q or st.session_state.pop("pending", None)
    if q:
        hist = [{"role": m["role"], "content": m["content"]} for m in chat[-6:]]
        chat.append({"role": "user", "content": q})
        log("question_asked", q)
        with st.spinner("Querying your data…"):
            try:
                out = ai_agent.ask(q, hist, ENGINE)
                chat.append({"role": "assistant", "content": out["answer"], "calls": out["tool_calls"]})
            except Exception as ex:
                chat.append({"role": "assistant", "content": f"Something went wrong: {ex}", "calls": []})
        st.rerun()

# ------------------------------------------------------------------ digest
with tabs[4]:
    st.subheader("Weekly digest")
    st.caption("Also runs on a schedule: `python -m reconai.digest --send` (GitHub Actions cron included).")
    if st.button("Generate this week's digest"):
        facts = digest.build_facts(F, 7)
        nar, source = digest.make_narrative(facts)
        st.session_state.digest = (digest.render_html(facts, nar), nar["headline"], source)
        log("digest_generated", source)
    if "digest" in st.session_state:
        body, headline, source = st.session_state.digest
        st.caption(f"Narrative written by: **{'Qwen (structured output)' if source == 'claude' else 'rule-based fallback'}**")
        st.components.v1.html(body, height=520, scrolling=True)
        st.download_button("Download HTML", body, "weekly_digest.html", "text/html")
        if st.button("Email it to me"):
            ok = digest.send_email(headline, body)
        if ok:
            st.success("Sent.")
        else:
            st.warning("Set SMTP_HOST and DIGEST_TO to enable email.")

# ------------------------------------------------------------------ impact
with tabs[5]:
    st.subheader("Adoption & impact")
    ev = db.read_events(ENGINE)
    if ev.empty:
        st.info("No usage yet. Share the link and come back.")
    else:
        sessions = ev[ev.event == "session_start"].session_id.nunique()
        qs = int((ev.event == "question_asked").sum())
        dg = int((ev.event == "digest_generated").sum())
        up, dn = int((ev.event == "feedback_up").sum()), int((ev.event == "feedback_down").sum())
        k = st.columns(4)
        k[0].metric("Sessions", sessions)
        k[1].metric("Questions asked", qs)
        k[2].metric("Digests generated", dg)
        k[3].metric("Helpful rate", f"{up / (up + dn):.0%}" if up + dn else "n/a")
        chart_data = ev.assign(date=ev.ts.dt.normalize()).groupby(["date", "event"]).size().unstack(fill_value=0)
        if not chart_data.empty:
            st.bar_chart(chart_data)
        st.markdown("**Estimated time saved**: replace these assumptions with numbers you measured in user interviews.")
        m1, m2 = st.columns(2)
        mq = m1.number_input("Minutes saved per answered question", 0, 120, 10)
        md = m2.number_input("Minutes saved per digest", 0, 240, 30)
        st.metric("Estimated hours saved", f"{(qs * mq + dg * md) / 60:.1f} h")
