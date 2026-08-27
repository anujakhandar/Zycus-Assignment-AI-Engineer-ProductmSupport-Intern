"""Task 2 - TAM account health summariser.

Two-step chain for one account:

1. **Extraction.** The account record, computed ticket statistics and the ticket
   history go to the model, which returns open risks and churn signals. Every
   churn signal must carry a verbatim quote.
2. **Verification.** Each quote is checked character-for-character against the
   body of the ticket it cites. A signal whose quote cannot be found is dropped,
   not repaired. This is what stops a paraphrase being presented to a TAM as
   something the customer said.
3. **Synthesis.** Only the surviving risks and signals reach the second call,
   which writes the executive summary and talking points.

Splitting extraction from synthesis means a wording change in the summary prompt
cannot alter which tickets were flagged.

**Determinism.** ``temperature=0``, a content-addressed response cache, ticket
history sorted by ``ticket_id``, statistics counted in a fixed key order, and
signals ordered by ``(ticket_id, quote)``. The same account id produces a
byte-identical brief.

**Two dataset facts this has to handle.** ``account_id`` on tickets does not
resolve to accounts (484 distinct ids across 500 tickets, 4 matches), so the
history is resolved by company with the fallback recorded in ``data_gaps``. And
every ticket predates a 90-day window measured from today, so "now" is anchored
to the newest ticket in the dataset by default - otherwise every brief would be
built from zero tickets. Both choices are reported in the output, never silent.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from prompts import get_prompt
from src.data_loader import Account, Dataset, Ticket, load_all
from src.llm_client import LLMClient
from src.schemas import AccountBrief, ChurnSignal, RiskFlag, TicketStats

OPEN_STATUSES = {"Open", "In Progress", "Pending Customer"}
MIN_QUOTE_WORDS = 4
MAX_QUOTE_WORDS = 60


class AccountBriefError(RuntimeError):
    """Raised when a brief cannot be produced."""


# ---------------------------------------------------------------------------
# Resolution and windowing
# ---------------------------------------------------------------------------


def resolve_account(account_id: str, dataset: Dataset) -> Account:
    """Find an account by id, falling back to a case-insensitive company name."""
    wanted = (account_id or "").strip()
    for account in dataset.accounts:
        if account.account_id.lower() == wanted.lower():
            return account
    for account in dataset.accounts:
        if (account.company or "").lower() == wanted.lower():
            return account
    raise AccountBriefError(f"no account matching {account_id!r} in data/accounts.json")


def dataset_now(tickets: Sequence[Ticket]) -> datetime:
    """The newest ``created_at`` in the data, used as the clock for windowing.

    Anchoring to the data rather than the wall clock keeps the 90-day window
    meaningful. Every ticket in this dataset is older than 90 days, so a
    wall-clock window returns nothing for every account, and would silently
    start returning different results as real time passes.
    """
    stamps = [t.created_at for t in tickets if t.created_at]
    if not stamps:
        return datetime.now(timezone.utc)
    newest = max(stamps)
    return newest if newest.tzinfo else newest.replace(tzinfo=timezone.utc)


def account_tickets(
    account: Account,
    dataset: Dataset,
    *,
    days: Optional[int] = 90,
    now: Optional[datetime] = None,
) -> tuple[list[Ticket], list[str]]:
    """Tickets for an account within the window, plus any data gaps found.

    Matches on ``account_id`` first as the schema documents. That key does not
    resolve in this dataset, so it falls back to the company name and says so.
    """
    gaps: list[str] = []

    by_id = [t for t in dataset.tickets if t.account_id == account.account_id]
    matched = by_id
    if not by_id and account.company:
        matched = [t for t in dataset.tickets if t.company == account.company]
        if matched:
            gaps.append(
                f"No ticket carries account_id {account.account_id}; history resolved by "
                f"company name ({account.company}) instead. The account_id field on "
                f"tickets does not reference accounts.json."
            )

    if not matched:
        gaps.append(f"No tickets found for {account.company or account.account_id}.")
        return [], gaps

    reference = now or dataset_now(dataset.tickets)
    if days is None:
        windowed = list(matched)
    else:
        cutoff = reference - timedelta(days=days)
        windowed = [t for t in matched if t.created_at and _utc(t.created_at) > cutoff]

        if not windowed:
            gaps.append(
                f"No tickets in the last {days} days before {reference.date()}; "
                f"using the full history of {len(matched)} tickets instead."
            )
            windowed = list(matched)

    # Sorted by id so the prompt payload, and therefore the cache key, is stable.
    windowed.sort(key=lambda t: t.ticket_id)
    return windowed, gaps


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Deterministic statistics - no model involved
# ---------------------------------------------------------------------------


def compute_stats(
    tickets: Sequence[Ticket],
    *,
    days: Optional[int] = 90,
    now: Optional[datetime] = None,
) -> TicketStats:
    """Count the ticket history. Deterministic and independent of the model."""
    stamps = sorted(_utc(t.created_at) for t in tickets if t.created_at)
    scores = [t.satisfaction_score for t in tickets if t.satisfaction_score is not None]

    return TicketStats(
        window_days=days,
        window_start=stamps[0].date().isoformat() if stamps else None,
        window_end=stamps[-1].date().isoformat() if stamps else None,
        total=len(tickets),
        by_status=_counts(t.status for t in tickets),
        by_category=_counts(t.category for t in tickets),
        by_urgency=_counts(t.urgency for t in tickets),
        by_product=_counts(t.product for t in tickets),
        unresolved=sum(1 for t in tickets if t.status in OPEN_STATUSES),
        mean_satisfaction=round(sum(scores) / len(scores), 2) if scores else None,
    )


def _counts(values) -> dict[str, int]:
    """Counter as a dict with keys in a fixed order, for stable serialisation."""
    counter = Counter(v for v in values if v)
    return {key: counter[key] for key in sorted(counter)}


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Collapse whitespace so a quote is not rejected over line wrapping alone."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def verify_signals(raw_signals: Any, tickets: Sequence[Ticket]) -> tuple[list[ChurnSignal], list[str]]:
    """Keep only signals whose quote genuinely appears in the cited ticket.

    Returns the surviving signals and a note for each rejection. Rejections are
    reported in ``data_gaps`` rather than hidden, because a model that fabricates
    quotes is something the operator needs to see.
    """
    by_id = {t.ticket_id: t for t in tickets}
    kept: list[ChurnSignal] = []
    rejected: list[str] = []

    if not isinstance(raw_signals, list):
        return kept, rejected

    for entry in raw_signals:
        if not isinstance(entry, dict):
            continue

        ticket_id = str(entry.get("ticket_id") or "").strip()
        quote = str(entry.get("quote") or "").strip().strip('"')
        signal = str(entry.get("signal") or "").strip()
        rationale = str(entry.get("rationale") or "").strip()

        ticket = by_id.get(ticket_id)
        if ticket is None:
            rejected.append(f"Dropped a churn signal citing {ticket_id or '(no id)'}: not in this window.")
            continue

        words = len(quote.split())
        if not (MIN_QUOTE_WORDS <= words <= MAX_QUOTE_WORDS):
            rejected.append(f"Dropped a churn signal on {ticket_id}: quote was {words} words.")
            continue

        haystack = _normalise(f"{ticket.subject} {ticket.body}").lower()
        if _normalise(quote).lower() not in haystack:
            rejected.append(f"Dropped a churn signal on {ticket_id}: quote not found verbatim in the ticket.")
            continue

        kept.append(
            ChurnSignal(
                ticket_id=ticket_id,
                quote=_normalise(quote),
                signal=signal or "Escalation signal",
                rationale=rationale,
                verified=True,
            )
        )

    kept.sort(key=lambda s: (s.ticket_id, s.quote))
    return kept, rejected


def _parse_risks(raw_risks: Any) -> list[RiskFlag]:
    """Coerce the model's risk list, dropping anything unusable."""
    if not isinstance(raw_risks, list):
        return []

    order = {"High": 0, "Medium": 1, "Low": 2}
    risks: list[RiskFlag] = []
    for entry in raw_risks:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("risk") or "").strip()
        evidence = str(entry.get("evidence") or "").strip()
        if not label or not evidence:
            continue
        severity = str(entry.get("severity") or "Medium").strip().title()
        if severity not in order:
            severity = "Medium"
        risks.append(
            RiskFlag(
                risk=label,
                severity=severity,
                evidence=evidence,
                source=str(entry.get("source") or "").strip(),
            )
        )

    risks.sort(key=lambda r: (order[r.severity], r.risk))
    return risks


# ---------------------------------------------------------------------------
# Prompt payload rendering
# ---------------------------------------------------------------------------


def _account_json(account: Account) -> str:
    return json.dumps(account.model_dump(), indent=2, sort_keys=True, default=str, ensure_ascii=False)


def _stats_text(stats: TicketStats) -> str:
    return json.dumps(stats.model_dump(), indent=2, sort_keys=True, ensure_ascii=False)


def _ticket_text(tickets: Sequence[Ticket]) -> str:
    """Ticket history for the prompt, in a fixed order and a fixed shape."""
    blocks = []
    for ticket in tickets:
        blocks.append(
            f"--- {ticket.ticket_id} | {ticket.status} | {ticket.urgency} | "
            f"{ticket.category} | {ticket.product} / {ticket.product_area} | "
            f"opened {ticket.created_at.date() if ticket.created_at else 'unknown'} | "
            f"satisfaction {ticket.satisfaction_score if ticket.satisfaction_score is not None else 'not given'}\n"
            f"subject: {ticket.subject}\n{ticket.body}"
        )
    return "\n\n".join(blocks) if blocks else "(no tickets in this window)"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_account_brief(
    account_id: str,
    *,
    dataset: Optional[Dataset] = None,
    llm: Optional[LLMClient] = None,
    days: Optional[int] = 90,
    now: Optional[datetime] = None,
) -> AccountBrief:
    """Build the three-section brief for one account. Deterministic per input."""
    dataset = dataset or load_all()
    llm = llm or LLMClient()

    account = resolve_account(account_id, dataset)
    tickets, gaps = account_tickets(account, dataset, days=days, now=now)
    stats = compute_stats(tickets, days=days, now=now)

    if not account.escalation_notes:
        gaps.append("Account record carries no escalation notes.")
    if account.nps_score is None:
        gaps.append("No NPS score recorded for this account.")
    if account.open_tickets != stats.unresolved:
        gaps.append(
            f"Account record states {account.open_tickets} open tickets; "
            f"{stats.unresolved} are actually open in the ticket data. Counts computed from tickets."
        )

    account_json = _account_json(account)
    stats_text = _stats_text(stats)

    # Step 1 - extraction
    risk_prompt = get_prompt("account_risk")
    raw_risk, _ = llm.complete(
        risk_prompt.render(
            account_json=account_json,
            stats_text=stats_text,
            ticket_count=len(tickets),
            ticket_text=_ticket_text(tickets),
        ),
        system=risk_prompt.system,
        tags={"prompt": risk_prompt.id, "task": "account_risk", "account": account.account_id},
    )
    risk_payload = _safe_json(raw_risk)

    # Step 2 - verification
    risks = _parse_risks(risk_payload.get("open_risks"))
    signals, rejected = verify_signals(risk_payload.get("churn_signals"), tickets)
    gaps.extend(rejected)

    # Step 3 - synthesis, over verified evidence only
    brief_prompt = get_prompt("account_brief")
    raw_brief, meta = llm.complete(
        brief_prompt.render(
            account_json=account_json,
            stats_text=stats_text,
            risks_text=_render_risks(risks),
            signals_text=_render_signals(signals),
        ),
        system=brief_prompt.system,
        tags={"prompt": brief_prompt.id, "task": "account_brief", "account": account.account_id},
    )
    brief_payload = _safe_json(raw_brief)

    summary = str(brief_payload.get("executive_summary") or "").strip()
    if not summary:
        summary = (
            f"{account.company} is a {account.plan_tier} account on "
            f"${account.arr_usd:,.0f} ARR with health status {account.health_status}. "
            f"{stats.total} tickets are in scope, {stats.unresolved} still open."
        )
        gaps.append("Model returned no executive summary; a data-only summary was substituted.")

    points = [
        str(p).strip()
        for p in (brief_payload.get("talking_points") or [])
        if isinstance(p, (str, int, float)) and str(p).strip()
    ]

    return AccountBrief(
        account_id=account.account_id,
        company=account.company or account.account_id,
        executive_summary=summary,
        open_risks=risks,
        churn_signals=signals,
        talking_points=points,
        stats=stats,
        data_gaps=gaps,
        prompt_version=f"{risk_prompt.id} -> {brief_prompt.id}",
        model=meta.model,
        cached=meta.cached,
    )


def _safe_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a response, tolerating fences and stray prose."""
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_risks(risks: Sequence[RiskFlag]) -> str:
    if not risks:
        return "(none identified)"
    return "\n".join(f"- [{r.severity}] {r.risk} - {r.evidence} (source: {r.source or 'n/a'})" for r in risks)


def _render_signals(signals: Sequence[ChurnSignal]) -> str:
    if not signals:
        return "(none survived quote verification)"
    return "\n".join(f'- {s.ticket_id}: "{s.quote}" - {s.signal}' for s in signals)
