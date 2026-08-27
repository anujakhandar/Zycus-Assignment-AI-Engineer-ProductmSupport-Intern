"""Test cases for both pipelines.

Every case is built from the provided mock dataset. Adversarial cases are
*mutations* of real records - a truncated body, a stripped product name, an
account with its ticket history withheld - rather than invented tickets, so
nothing here introduces data from outside the starter repo.

Each case carries acceptance criteria rather than a single expected string,
because these are generative outputs. A criterion is a named predicate over the
result; :mod:`evals.scoring` turns the set of them into a 0-1 quality score and
a pass/fail verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.data_loader import Dataset, Ticket, load_all
from src.schemas import AccountBrief, TriageResult

# One shared load, so every case sees the same data and the run stays fast.
DATASET: Dataset = load_all()
_BY_ID = {t.ticket_id: t for t in DATASET.tickets}


def ticket(ticket_id: str) -> Ticket:
    return _BY_ID[ticket_id]


def _find(**criteria: Any) -> Ticket:
    """First ticket matching every field, sorted by id so it never varies."""
    for candidate in sorted(DATASET.tickets, key=lambda t: t.ticket_id):
        if all(getattr(candidate, key, None) == value for key, value in criteria.items()):
            return candidate
    raise LookupError(f"no ticket matching {criteria}")


def _find_containing(needle: str, **criteria: Any) -> Ticket:
    """First ticket whose body contains ``needle`` and matches ``criteria``."""
    for candidate in sorted(DATASET.tickets, key=lambda t: t.ticket_id):
        if needle in candidate.body and all(
            getattr(candidate, key, None) == value for key, value in criteria.items()
        ):
            return candidate
    raise LookupError(f"no ticket containing {needle!r} with {criteria}")


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    """One named, weighted check over a pipeline result."""

    name: str
    check: Callable[[Any], bool]
    weight: float = 1.0
    critical: bool = False  # a failed critical criterion fails the whole case


@dataclass
class TriageCase:
    """A triage test case."""

    case_id: str
    description: str
    subject: str
    body: str
    criteria: list[Criterion]
    adversarial: bool = False
    judge_rubric: Optional[str] = None
    source_ticket: Optional[str] = None


@dataclass
class BriefCase:
    """An account-brief test case."""

    case_id: str
    description: str
    account: str
    criteria: list[Criterion]
    adversarial: bool = False
    judge_rubric: Optional[str] = None
    days: Optional[int] = 90
    withhold_tickets: bool = False  # adversarial: simulate incomplete data
    rerun_for_determinism: bool = False


# -- reusable predicates ----------------------------------------------------


def cites_any(*fragments: str) -> Callable[[TriageResult], bool]:
    def check(result: TriageResult) -> bool:
        sources = " ".join(reference.source for reference in result.kb_references)
        return any(fragment in sources for fragment in fragments)

    return check


def urgency_in(*allowed: str) -> Callable[[TriageResult], bool]:
    return lambda result: result.urgency.value in allowed


def field_is(field_name: str, *allowed: str) -> Callable[[TriageResult], bool]:
    return lambda result: getattr(result, field_name).value in allowed


def has_reasoning(field_name: str, min_words: int = 4) -> Callable[[TriageResult], bool]:
    return lambda result: len(getattr(result, field_name).reasoning.split()) >= min_words


def draft_length(low: int, high: int) -> Callable[[TriageResult], bool]:
    return lambda result: low <= len(result.draft_response.split()) <= high


def draft_avoids(*banned: str) -> Callable[[TriageResult], bool]:
    """The draft must not promise a fix, a root cause, a refund or a deadline."""

    def check(result: TriageResult) -> bool:
        lowered = result.draft_response.lower()
        return not any(phrase in lowered for phrase in banned)

    return check


PROMISE_WORDS = ("we will fix", "guarantee", "refund", "root cause is", "by tomorrow", "within 24 hours")


# ---------------------------------------------------------------------------
# Task 1 cases
# ---------------------------------------------------------------------------

_DATA_LOSS = _find_containing("URGENT", category="Data Loss")
_SCHEMA = _find_containing("SCHEMA_MISMATCH")
_BILLING = _find_containing("outgrown our current")
_FEATURE = _find_containing("only allows individual")
_SSO = _find_containing("new joiners")

TRIAGE_CASES: list[TriageCase] = [
    TriageCase(
        case_id="T1",
        description="Explicit data-loss report should be high urgency and route to Tier-2",
        subject=_DATA_LOSS.subject,
        body=_DATA_LOSS.body,
        source_ticket=_DATA_LOSS.ticket_id,
        criteria=[
            Criterion("urgency is P1 or P2", urgency_in("P1", "P2"), weight=2.0, critical=True),
            Criterion("category is Data Loss", field_is("category", "Data Loss"), weight=2.0),
            Criterion("product identified", field_is("product", _DATA_LOSS.product), weight=1.0),
            Criterion("urgency reasoning given", has_reasoning("urgency")),
            Criterion("draft is 30-160 words", draft_length(30, 160)),
            Criterion("draft promises nothing", draft_avoids(*PROMISE_WORDS), weight=1.5),
        ],
        judge_rubric=(
            "Does the draft response acknowledge potential data loss with appropriate "
            "seriousness, give a concrete next step, and avoid promising recovery?"
        ),
    ),
    TriageCase(
        case_id="T2",
        description="Ticket quoting SCHEMA_MISMATCH must surface the documenting KB section",
        subject=_SCHEMA.subject,
        body=_SCHEMA.body,
        source_ticket=_SCHEMA.ticket_id,
        criteria=[
            Criterion(
                "cites a doc covering the error code",
                cites_any("databridge-pro", "performance-and-integrations", "troubleshooting"),
                weight=2.0,
                critical=True,
            ),
            Criterion("at least one KB reference", lambda r: len(r.kb_references) >= 1, weight=1.5),
            Criterion("category is Bug or Integration or Data Loss",
                      field_is("category", "Bug", "Integration", "Data Loss")),
            Criterion("draft promises nothing", draft_avoids(*PROMISE_WORDS)),
        ],
        judge_rubric=(
            "Does the response reference the error code the customer reported and "
            "point to a concrete diagnostic step drawn from the cited documentation?"
        ),
    ),
    TriageCase(
        case_id="T3",
        description="Plan upgrade request should route to Billing Operations",
        subject=_BILLING.subject,
        body=_BILLING.body,
        source_ticket=_BILLING.ticket_id,
        criteria=[
            Criterion("category is Billing", field_is("category", "Billing"), weight=2.0),
            Criterion("routed to Billing Operations",
                      lambda r: r.responder_team == "Billing Operations", weight=2.0, critical=True),
            Criterion("not treated as urgent", urgency_in("P3", "P4"), weight=1.5),
            Criterion("cites the billing doc", cites_any("billing"), weight=1.0),
        ],
        judge_rubric="Does the draft address the upgrade request concretely rather than generically?",
    ),
    TriageCase(
        case_id="T4",
        description="Bulk-operation request is a feature ask, not a defect",
        subject=_FEATURE.subject,
        body=_FEATURE.body,
        source_ticket=_FEATURE.ticket_id,
        criteria=[
            Criterion("category is Feature Request",
                      field_is("category", "Feature Request"), weight=2.0, critical=True),
            Criterion("routed to Product Management",
                      lambda r: r.responder_team == "Product Management", weight=1.5),
            Criterion("low urgency", urgency_in("P3", "P4"), weight=1.0),
            Criterion("draft does not promise the feature", draft_avoids(*PROMISE_WORDS), weight=1.5),
        ],
        judge_rubric=(
            "Does the draft acknowledge the feature request honestly without committing "
            "to building it or giving a delivery date?"
        ),
    ),
    TriageCase(
        case_id="T5",
        description="SSO access failure for new joiners routes to Security Engineering",
        subject=_SSO.subject,
        body=_SSO.body,
        source_ticket=_SSO.ticket_id,
        criteria=[
            Criterion("product area is an access area",
                      field_is("product_area", "SSO", "Authentication", "Permissions"), weight=2.0),
            Criterion("cites the SSO troubleshooting doc",
                      cites_any("authentication-sso", "securevault"), weight=2.0, critical=True),
            Criterion("routing explained", lambda r: len(r.routing_reasoning.split()) >= 4),
            Criterion("draft is 30-160 words", draft_length(30, 160)),
        ],
        judge_rubric=(
            "Does the response identify group-to-role mapping or a comparable concrete "
            "cause from the cited SSO documentation, rather than generic advice?"
        ),
    ),
    # -- adversarial ---------------------------------------------------------
    TriageCase(
        case_id="T6-adversarial-tone",
        description=(
            "ADVERSARIAL: urgent-sounding language over a low-impact request. "
            "Urgency must follow business impact, not customer tone."
        ),
        subject="URGENT!!! NEED THIS FIXED IMMEDIATELY",
        body=(
            "URGENT!!! This is completely unacceptable and needs fixing RIGHT NOW.\n\n"
            "The column headers in the AnalyticsHub Reports export are displayed in "
            "title case but our style guide requires sentence case. One person on our "
            "team noticed. There is no impact on the data itself and we can rename the "
            "columns in Excel afterwards.\n\nPlease escalate this to your management."
        ),
        adversarial=True,
        criteria=[
            Criterion("not escalated to P1", lambda r: r.urgency.value != "P1", weight=3.0, critical=True),
            Criterion("rated P3 or P4", urgency_in("P3", "P4"), weight=2.0),
            Criterion("urgency reasoning cites impact", has_reasoning("urgency", 5), weight=1.0),
            Criterion("draft stays professional", draft_avoids(*PROMISE_WORDS)),
        ],
        judge_rubric=(
            "The customer used urgent language for a cosmetic issue with a workaround. "
            "Does the reasoning justify the urgency by actual business impact rather "
            "than by the customer's tone, and does the draft remain courteous?"
        ),
    ),
    TriageCase(
        case_id="T7-adversarial-sparse",
        description=(
            "ADVERSARIAL: near-empty ticket (a real body truncated to its first few "
            "words). Must not fabricate specifics; must ask for human review."
        ),
        subject="",
        body=" ".join(_DATA_LOSS.body.split()[:5]),
        adversarial=True,
        source_ticket=_DATA_LOSS.ticket_id,
        criteria=[
            Criterion("flagged for human review",
                      lambda r: r.needs_human_review is True, weight=3.0, critical=True),
            Criterion("confidence is low", lambda r: r.overall_confidence < 0.75, weight=2.0),
            Criterion("review reason recorded", lambda r: len(r.review_reasons) >= 1, weight=1.0),
            Criterion("still returns a valid urgency", urgency_in("P1", "P2", "P3", "P4")),
        ],
        judge_rubric=(
            "The ticket is almost empty. Does the output avoid inventing specifics "
            "(versions, error codes, user counts) that the text does not contain?"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Task 2 cases
# ---------------------------------------------------------------------------


def _account_by(**criteria: Any) -> str:
    for account in sorted(DATASET.accounts, key=lambda a: a.account_id):
        if all(getattr(account, key, None) == value for key, value in criteria.items()):
            return account.account_id
    raise LookupError(f"no account matching {criteria}")


_AT_RISK = _account_by(health_status="At Risk")
_HEALTHY = _account_by(health_status="Healthy")
_CHURNING = _account_by(health_status="Churning")


def summary_sentences(low: int, high: int) -> Callable[[AccountBrief], bool]:
    def check(brief: AccountBrief) -> bool:
        count = len([s for s in brief.executive_summary.replace("!", ".").split(".") if s.strip()])
        return low <= count <= high

    return check


def all_quotes_verified(brief: AccountBrief) -> bool:
    """Every surfaced churn signal must have passed verbatim verification."""
    return all(signal.verified for signal in brief.churn_signals)


def quotes_really_in_tickets(brief: AccountBrief) -> bool:
    """Independent re-check: find each quote in the dataset ticket it cites."""
    import re

    for signal in brief.churn_signals:
        source = _BY_ID.get(signal.ticket_id)
        if source is None:
            return False
        haystack = re.sub(r"\s+", " ", f"{source.subject} {source.body}").lower()
        if re.sub(r"\s+", " ", signal.quote).lower() not in haystack:
            return False
    return True


BRIEF_CASES: list[BriefCase] = [
    BriefCase(
        case_id="B1",
        description="At-risk account produces a summary, risks and talking points",
        account=_AT_RISK,
        criteria=[
            Criterion("summary is 3-5 sentences", summary_sentences(3, 5), weight=2.0),
            Criterion("at least one open risk", lambda b: len(b.open_risks) >= 1, weight=2.0, critical=True),
            Criterion("3-5 talking points", lambda b: 3 <= len(b.talking_points) <= 5, weight=2.0),
            Criterion("every risk carries evidence",
                      lambda b: all(r.evidence.strip() for r in b.open_risks), weight=1.5),
            Criterion("stats computed", lambda b: b.stats.total > 0, weight=1.0),
        ],
        judge_rubric=(
            "Is the executive summary specific to this account - naming real numbers "
            "such as ARR, seats, ticket counts or renewal date - rather than generic "
            "account-management filler?"
        ),
    ),
    BriefCase(
        case_id="B2",
        description="Churning account: churn signals must be quote-backed and verified",
        account=_CHURNING,
        criteria=[
            Criterion("all signals verified", all_quotes_verified, weight=3.0, critical=True),
            Criterion("quotes exist verbatim in the cited tickets",
                      quotes_really_in_tickets, weight=3.0, critical=True),
            Criterion("each signal has a rationale",
                      lambda b: all(s.rationale.strip() for s in b.churn_signals), weight=1.0),
            Criterion("summary is 3-5 sentences", summary_sentences(3, 5), weight=1.5),
        ],
        judge_rubric=(
            "Does each churn signal's quote genuinely support the risk claimed for it, "
            "or is the quote unrelated to the stated signal?"
        ),
    ),
    BriefCase(
        case_id="B3",
        description="Healthy account should not be dressed up as a crisis",
        account=_HEALTHY,
        criteria=[
            Criterion("summary is 3-5 sentences", summary_sentences(3, 5), weight=2.0),
            Criterion("no fabricated churn signals", all_quotes_verified, weight=2.0, critical=True),
            Criterion("talking points present", lambda b: len(b.talking_points) >= 3, weight=1.5),
            Criterion("high-severity risks are not invented",
                      lambda b: sum(1 for r in b.open_risks if r.severity == "High") <= 3, weight=1.0),
        ],
        judge_rubric=(
            "The account is marked Healthy. Does the brief reflect that proportionately "
            "rather than manufacturing alarm?"
        ),
    ),
    BriefCase(
        case_id="B4",
        description="Determinism: the same account twice must be byte-identical",
        account=_AT_RISK,
        rerun_for_determinism=True,
        criteria=[
            Criterion("second run is identical", lambda b: True, weight=3.0, critical=True),
        ],
    ),
    BriefCase(
        case_id="B5",
        description="Data gaps are disclosed rather than papered over",
        account=_AT_RISK,
        criteria=[
            Criterion("data gaps reported", lambda b: len(b.data_gaps) >= 1, weight=2.0, critical=True),
            Criterion("account_id resolution gap disclosed",
                      lambda b: any("account_id" in gap for gap in b.data_gaps), weight=2.0),
            Criterion("window recorded in stats",
                      lambda b: b.stats.window_start is not None, weight=1.0),
        ],
    ),
    # -- adversarial ---------------------------------------------------------
    BriefCase(
        case_id="B6-adversarial-no-tickets",
        description=(
            "ADVERSARIAL: incomplete account data - ticket history withheld. "
            "Must report the gap and must not invent a ticket history."
        ),
        account=_AT_RISK,
        withhold_tickets=True,
        adversarial=True,
        criteria=[
            Criterion("no churn signals invented",
                      lambda b: len(b.churn_signals) == 0, weight=3.0, critical=True),
            Criterion("missing history disclosed",
                      lambda b: any("no tickets" in gap.lower() for gap in b.data_gaps),
                      weight=3.0, critical=True),
            Criterion("ticket total is zero", lambda b: b.stats.total == 0, weight=1.0),
            Criterion("still produces a summary",
                      lambda b: len(b.executive_summary.split()) >= 10, weight=1.0),
        ],
        judge_rubric=(
            "No ticket history was available. Does the brief state that limitation "
            "plainly instead of implying it reviewed tickets it never saw?"
        ),
    ),
]


ALL_CASE_IDS = [case.case_id for case in TRIAGE_CASES] + [case.case_id for case in BRIEF_CASES]
