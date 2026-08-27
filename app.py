"""Streamlit UI for the triage and account brief tools.

    streamlit run app.py

Written for the people who actually use these tools: a support engineer who
wants a ticket triaged, and a TAM who wants an account brief before a call.
Neither should need to know what a JSON schema is.

Runs from the committed response cache, so it works without an API key. When a
request is not cached and no key is configured, the UI says so plainly instead
of showing a traceback.
"""

from __future__ import annotations

import streamlit as st

from src.data_loader import load_accounts, load_tickets
from src.llm_client import LLMError

st.set_page_config(page_title="Support & Account Tooling", page_icon="ST", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading. Streamlit reruns the whole script on every interaction,
# so without this the dataset would be parsed again on every click.
# ---------------------------------------------------------------------------


@st.cache_resource
def get_tickets():
    return load_tickets()


@st.cache_resource
def get_accounts():
    return sorted(load_accounts(), key=lambda a: a.company or a.account_id)


@st.cache_resource
def get_kb_index():
    from src.retrieval import get_index

    return get_index()


URGENCY_HELP = {
    "P1": "Business stopped. Production down, data being lost, or total loss of access.",
    "P2": "Major impact with a painful workaround. A core workflow broken for many users.",
    "P3": "Moderate impact and a workaround exists.",
    "P4": "Low impact. Cosmetic, informational, or a request with no time pressure.",
}


# ---------------------------------------------------------------------------
# Triage tab
# ---------------------------------------------------------------------------


def render_triage() -> None:
    st.subheader("Triage a support ticket")
    st.caption(
        "Paste a ticket, or pick one from the dataset. The tool classifies it, "
        "finds the documentation that supports the call, and drafts a first reply."
    )

    tickets = get_tickets()
    mode = st.radio(
        "Ticket source",
        ["Pick from the dataset", "Paste my own"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Pick from the dataset":
        options = {f"{t.ticket_id} | {t.subject[:70]}": t for t in tickets[:80]}
        chosen = st.selectbox("Ticket", list(options))
        selected = options[chosen]
        subject, body = selected.subject, selected.body
        st.text_area("Body", body, height=170, disabled=True)
    else:
        subject = st.text_input("Subject", placeholder="Pipeline failing since this morning")
        body = st.text_area(
            "Body",
            height=200,
            placeholder="Describe the problem as the customer wrote it...",
        )

    if not st.button("Run triage", type="primary", use_container_width=True):
        return
    if not (body or "").strip():
        st.warning("Add a ticket body first.")
        return

    from src.schemas import TicketInput
    from src.triage import triage_ticket

    try:
        with st.spinner("Retrieving documentation and classifying..."):
            result = triage_ticket(TicketInput(subject=subject or "", body=body), index=get_kb_index())
    except LLMError as exc:
        st.error(str(exc))
        st.info(
            "This ticket has not been triaged before, so it needs a live model call. "
            "Add an API key to .env, or pick a ticket that is already in the cache."
        )
        return

    if result.needs_human_review:
        st.warning("Flagged for human review before sending: " + "; ".join(result.review_reasons))

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Classification")
        for label, field in [
            ("Product", result.product),
            ("Area", result.product_area),
            ("Category", result.category),
            ("Urgency", result.urgency),
        ]:
            help_text = URGENCY_HELP.get(field.value) if label == "Urgency" else None
            st.metric(label, field.value, help=help_text)
            st.caption(f"{field.reasoning}  \nConfidence {field.confidence:.0%}")

    with right:
        st.markdown("#### Routing")
        st.success(f"Send to **{result.responder_team}**")
        if result.routing_reasoning:
            st.caption(result.routing_reasoning)

        st.markdown("#### Matching documentation")
        if result.kb_references:
            for reference in result.kb_references:
                with st.expander(reference.breadcrumb):
                    if reference.matched_codes:
                        st.caption("Matched error codes: " + ", ".join(reference.matched_codes))
                    st.caption(f"Source: {reference.source}")
                    st.write(reference.excerpt)
        else:
            st.info("Nothing in the knowledge base matched this ticket.")

        if result.known_issue:
            st.caption(f"Known issue pattern: {result.known_issue}")

    st.markdown("#### Draft first response")
    st.text_area(
        "Edit before sending",
        result.draft_response,
        height=190,
        label_visibility="collapsed",
    )
    st.caption(
        f"Overall confidence {result.overall_confidence:.0%} | "
        f"prompt {result.prompt_version} | model {result.model}"
        + (" | served from cache" if result.cached else "")
    )


# ---------------------------------------------------------------------------
# Account brief tab
# ---------------------------------------------------------------------------


def render_brief() -> None:
    st.subheader("Build an account brief")
    st.caption(
        "Everything a TAM needs before a call: where the account stands, what is "
        "at risk, and what to raise. Churn signals quote the customer directly."
    )

    accounts = get_accounts()
    options = {
        f"{a.company} | {a.plan_tier} | {a.health_status} | ${a.arr_usd:,.0f}": a for a in accounts
    }
    chosen = st.selectbox("Account", list(options))
    account = options[chosen]

    window = st.select_slider(
        "Ticket window",
        options=[30, 60, 90, 180, 0],
        value=90,
        format_func=lambda d: "All history" if d == 0 else f"Last {d} days",
    )

    top = st.columns(4)
    top[0].metric("Seats active", f"{account.seats_active:,}", f"of {account.seats_licensed:,} licensed")
    top[1].metric("Health", account.health_status, account.usage_trend)
    top[2].metric("Renews", str(account.renewal_date))
    top[3].metric("NPS", account.nps_score if account.nps_score is not None else "not recorded")

    if not st.button("Build brief", type="primary", use_container_width=True):
        return

    from src.account_brief import build_account_brief

    try:
        with st.spinner("Reading ticket history and extracting risks..."):
            brief = build_account_brief(account.account_id, days=None if window == 0 else window)
    except LLMError as exc:
        st.error(str(exc))
        st.info(
            "This brief has not been generated before, so it needs live model calls. "
            "Add an API key to .env, or pick an account that is already in the cache."
        )
        return

    st.markdown("### Executive summary")
    st.write(brief.executive_summary)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Open risks")
        if brief.open_risks:
            for risk in brief.open_risks:
                icon = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(risk.severity, "")
                st.markdown(f"{icon} **{risk.risk}**")
                st.caption(f"{risk.evidence}" + (f"  \nSource: {risk.source}" if risk.source else ""))
        else:
            st.info("No open risks identified from the available data.")

    with right:
        st.markdown("### Churn and escalation signals")
        if brief.churn_signals:
            for signal in brief.churn_signals:
                st.markdown(f"**{signal.signal}** | {signal.ticket_id}")
                st.info(f'"{signal.quote}"')
                st.caption(signal.rationale)
        else:
            st.success("No churn language found in this window.")

    st.markdown("### Talking points")
    for index, point in enumerate(brief.talking_points, 1):
        st.markdown(f"{index}. {point}")

    stats = brief.stats
    st.markdown("### Ticket picture")
    cols = st.columns(4)
    cols[0].metric("Tickets in window", stats.total)
    cols[1].metric("Still open", stats.unresolved)
    cols[2].metric("High urgency", stats.by_urgency.get("P1", 0) + stats.by_urgency.get("P2", 0))
    cols[3].metric(
        "Mean satisfaction",
        f"{stats.mean_satisfaction:.1f}" if stats.mean_satisfaction is not None else "none given",
    )

    if brief.data_gaps:
        with st.expander(f"Data gaps and caveats ({len(brief.data_gaps)})"):
            for gap in brief.data_gaps:
                st.caption(gap)

    st.download_button(
        "Download brief as markdown",
        brief.to_markdown(),
        file_name=f"brief_{brief.account_id}.md",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Support and Account Tooling")

with st.sidebar:
    st.markdown("### About")
    st.caption(
        "Two internal tools built on the mock dataset: ticket triage for support "
        "engineers, and account briefs for TAMs."
    )
    st.markdown("### Data")
    st.caption(
        f"{len(get_tickets())} tickets | {len(get_accounts())} accounts | "
        f"{len(get_kb_index())} knowledge base sections"
    )
    st.markdown("### Note")
    st.caption(
        "Responses are cached, so repeat requests are instant and identical. "
        "A request that has never been made needs an API key in .env."
    )

triage_tab, brief_tab = st.tabs(["Ticket triage", "Account brief"])
with triage_tab:
    render_triage()
with brief_tab:
    render_brief()
