"""Loading and normalisation for the support datasets.

Three sources are covered:

* ``data/tickets.json``    - support tickets (one JSON array)
* ``data/accounts.json``   - customer accounts (one JSON array)
* ``data/knowledge-base/`` - markdown reference docs, chunked for retrieval

Records are validated into pydantic models so the rest of the codebase can rely
on field names and types instead of poking at raw dicts. Unknown fields are kept
(``extra="allow"``) so an upstream schema addition does not break loading.

Run ``python -m src.data_loader`` for a summary of what is on disk.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TICKETS_PATH = DATA_DIR / "tickets.json"
ACCOUNTS_PATH = DATA_DIR / "accounts.json"
KB_DIR = DATA_DIR / "knowledge-base"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Ticket(BaseModel):
    """A single support ticket as stored in tickets.json."""

    model_config = ConfigDict(extra="allow")

    ticket_id: str
    account_id: Optional[str] = None
    company: Optional[str] = None
    subject: str = ""
    body: str = ""
    product: Optional[str] = None
    product_area: Optional[str] = None
    category: Optional[str] = None
    urgency: Optional[str] = None
    status: Optional[str] = None
    plan_tier: Optional[str] = None
    assigned_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    channel: Optional[str] = None
    satisfaction_score: Optional[float] = None

    @property
    def text(self) -> str:
        """Subject and body joined - the usual input for classification."""
        return f"{self.subject}\n\n{self.body}".strip()


class PrimaryContact(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    title: Optional[str] = None


class Account(BaseModel):
    """A customer account as stored in accounts.json."""

    model_config = ConfigDict(extra="allow")

    account_id: str
    company: Optional[str] = None
    tam: Optional[str] = None
    plan_tier: Optional[str] = None
    arr_usd: Optional[float] = None
    seats_licensed: Optional[int] = None
    seats_active: Optional[int] = None
    products: list[str] = Field(default_factory=list)
    health_status: Optional[str] = None
    usage_trend: Optional[str] = None
    open_tickets: Optional[int] = None
    p1_tickets_last_30d: Optional[int] = None
    customer_since: Optional[str] = None
    renewal_date: Optional[str] = None
    last_qbr_date: Optional[str] = None
    primary_contact: Optional[PrimaryContact] = None
    escalation_notes: list[str] = Field(default_factory=list)
    nps_score: Optional[float] = None
    last_login_days_ago: Optional[int] = None
    integrations_active: list[str] = Field(default_factory=list)
    region: Optional[str] = None
    industry: Optional[str] = None

    @property
    def seat_utilisation(self) -> Optional[float]:
        """Active seats as a fraction of licensed seats, or None if unknown."""
        if not self.seats_licensed:
            return None
        return (self.seats_active or 0) / self.seats_licensed


MatchKind = Literal["account_id", "company", "unmatched"]


class EnrichedTicket(BaseModel):
    """A ticket paired with its account, where one could be found."""

    model_config = ConfigDict(extra="allow")

    ticket: Ticket
    account: Optional[Account] = None
    matched_on: MatchKind = "unmatched"

    @property
    def has_account(self) -> bool:
        return self.account is not None

    @property
    def ticket_id(self) -> str:
        return self.ticket.ticket_id


class JoinReport(BaseModel):
    """How well tickets lined up with accounts - worth logging on every load."""

    total_tickets: int = 0
    matched_by_account_id: int = 0
    matched_by_company: int = 0
    unmatched: int = 0
    tickets_missing_account_id: int = 0
    unmatched_account_ids: list[str] = Field(default_factory=list)

    @property
    def match_rate(self) -> float:
        if not self.total_tickets:
            return 0.0
        return (self.matched_by_account_id + self.matched_by_company) / self.total_tickets

    def summary(self) -> str:
        return (
            f"{self.total_tickets} tickets: "
            f"{self.matched_by_account_id} matched by account_id, "
            f"{self.matched_by_company} by company, "
            f"{self.unmatched} unmatched "
            f"({self.match_rate:.0%} overall)"
        )


class KBChunk(BaseModel):
    """One retrievable slice of a knowledge-base document."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    source: str
    category: str
    doc_title: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    text: str = ""
    char_count: int = 0

    @property
    def breadcrumb(self) -> str:
        """Heading trail, handy as a citation label in a prompt."""
        if self.heading_path:
            return " > ".join(self.heading_path)
        return self.doc_title or self.source


class Dataset(BaseModel):
    """Everything loaded from disk, ready to hand to the task modules."""

    model_config = ConfigDict(extra="allow")

    tickets: list[Ticket] = Field(default_factory=list)
    accounts: list[Account] = Field(default_factory=list)
    enriched_tickets: list[EnrichedTicket] = Field(default_factory=list)
    kb_chunks: list[KBChunk] = Field(default_factory=list)
    join_report: JoinReport = Field(default_factory=JoinReport)

    @property
    def accounts_by_id(self) -> dict[str, Account]:
        return build_account_index(self.accounts)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _read_json_array(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        # Tolerate a wrapper object such as {"tickets": [...]}.
        for value in payload.values():
            if isinstance(value, list):
                return value
        raise ValueError(f"{path} contains an object with no list to load")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array, got {type(payload).__name__}")
    return payload


def load_tickets(path: Path | str = TICKETS_PATH) -> list[Ticket]:
    """Load and validate every ticket."""
    return [Ticket.model_validate(row) for row in _read_json_array(Path(path))]


def load_accounts(path: Path | str = ACCOUNTS_PATH) -> list[Account]:
    """Load and validate every account."""
    return [Account.model_validate(row) for row in _read_json_array(Path(path))]


def build_account_index(accounts: Iterable[Account]) -> dict[str, Account]:
    """Map account_id -> Account. Later duplicates win, as with a plain dict."""
    return {account.account_id: account for account in accounts}


def build_company_index(accounts: Iterable[Account]) -> dict[str, Account]:
    """Map a normalised company name -> Account, for the fallback join."""
    index: dict[str, Account] = {}
    for account in accounts:
        if account.company:
            index[_normalise_company(account.company)] = account
    return index


def _normalise_company(name: str) -> str:
    return " ".join(name.split()).casefold()


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def join_tickets_to_accounts(
    tickets: Iterable[Ticket],
    accounts: Iterable[Account],
    *,
    fallback_to_company: bool = False,
) -> list[EnrichedTicket]:
    """Pair each ticket with its account.

    ``account_id`` is the documented key, and plenty of tickets point at an id
    that is not in accounts.json - that gap is expected, so an unmatched ticket
    comes back with ``account=None`` rather than raising.

    In this dataset the id-based join only lands for a handful of tickets while
    every ticket company does have an account record, so ``fallback_to_company``
    is offered as an opt-in second pass. Leave it off when the id is meant to be
    authoritative; turn it on when broader coverage matters more than a strict
    key match. ``EnrichedTicket.matched_on`` records which pass produced the hit.
    """
    account_list = list(accounts)
    by_id = build_account_index(account_list)
    by_company = build_company_index(account_list) if fallback_to_company else {}

    joined: list[EnrichedTicket] = []
    for ticket in tickets:
        account = by_id.get(ticket.account_id) if ticket.account_id else None
        matched_on: MatchKind = "account_id" if account else "unmatched"

        if account is None and fallback_to_company and ticket.company:
            account = by_company.get(_normalise_company(ticket.company))
            if account is not None:
                matched_on = "company"

        joined.append(EnrichedTicket(ticket=ticket, account=account, matched_on=matched_on))
    return joined


def summarise_join(enriched: Iterable[EnrichedTicket], *, max_examples: int = 10) -> JoinReport:
    """Count how the join went, with a sample of the ids that missed."""
    report = JoinReport()
    misses: Counter[str] = Counter()

    for item in enriched:
        report.total_tickets += 1
        if not item.ticket.account_id:
            report.tickets_missing_account_id += 1
        if item.matched_on == "account_id":
            report.matched_by_account_id += 1
        elif item.matched_on == "company":
            report.matched_by_company += 1
        else:
            report.unmatched += 1
            if item.ticket.account_id:
                misses[item.ticket.account_id] += 1

    report.unmatched_account_ids = [account_id for account_id, _ in misses.most_common(max_examples)]
    return report


def get_account_tickets(
    account_id: str,
    tickets: Iterable[Ticket],
    days: Optional[int] = 90,
    *,
    now: Optional[datetime] = None,
) -> list[Ticket]:
    """Tickets for one account, newest first.

    ``days=None`` returns the full history; otherwise only tickets created
    within that window are returned (the schema doc suggests 90 days for
    account health work).
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(days=days) if days is not None else None

    selected = [
        ticket
        for ticket in tickets
        if ticket.account_id == account_id
        and (cutoff is None or (_as_utc(ticket.created_at) or reference) > cutoff)
    ]
    return sorted(selected, key=lambda t: _as_utc(t.created_at) or reference, reverse=True)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalise to timezone-aware UTC so comparisons never mix naive and aware."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(?:```|~~~)")


def chunk_markdown(text: str, *, source: str, category: str) -> list[KBChunk]:
    """Split one markdown document into retrieval-sized chunks.

    Chunks break on horizontal rules (``---``), which these documents use as
    major section boundaries. Heading state is tracked across the whole file, so
    every chunk carries the heading trail it sits under even when the headings
    were declared in an earlier chunk:

    * ``heading_path`` - the trail in effect where the chunk starts
    * ``headings``     - headings that appear inside the chunk itself

    Rules inside fenced code blocks are ignored, and a ``---`` directly beneath
    a line of text is treated as a setext heading underline, not a break.
    """
    lines = text.splitlines()
    chunks: list[KBChunk] = []

    stack: list[tuple[int, str]] = []  # (level, title)
    # Tracks the H1 in force at flush time rather than the first one in the
    # file: troubleshooting/performance-and-integrations.md holds two separate
    # documents under two H1s, and citing the second one's chunks under the
    # first one's title would be wrong.
    doc_title: Optional[str] = None
    buffer: list[str] = []
    buffer_path: list[str] = []
    buffer_headings: list[str] = []
    in_fence = False
    previous_blank = True

    def current_path() -> list[str]:
        return [title for _, title in stack]

    def flush() -> None:
        nonlocal buffer, buffer_headings, buffer_path
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                KBChunk(
                    chunk_id=f"{source}#{len(chunks):02d}",
                    source=source,
                    category=category,
                    doc_title=doc_title,
                    heading_path=list(buffer_path),
                    headings=list(buffer_headings),
                    text=body,
                    char_count=len(body),
                )
            )
        buffer = []
        buffer_headings = []
        buffer_path = current_path()

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            previous_blank = False
            continue

        if not in_fence and previous_blank and _HR_RE.match(line):
            flush()
            previous_blank = True
            continue

        if not in_fence:
            heading = _HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                if level == 1:
                    doc_title = title
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                if not any(item.strip() for item in buffer):
                    # The heading opens this chunk, so it belongs to the trail.
                    buffer_path = current_path()
                else:
                    buffer_headings.append(title)
                buffer.append(line)
                previous_blank = False
                continue

        buffer.append(line)
        previous_blank = not line.strip()

    flush()
    return chunks


def load_knowledge_base(root: Path | str = KB_DIR) -> list[KBChunk]:
    """Read every markdown file under ``root`` and return all chunks.

    ``category`` comes from the first directory below the root (``products``,
    ``troubleshooting``, ``billing``, ``onboarding``), which is what retrieval
    filters on.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"knowledge base not found at {root_path}")

    chunks: list[KBChunk] = []
    for path in sorted(root_path.rglob("*.md")):
        relative = path.relative_to(root_path)
        category = relative.parts[0] if len(relative.parts) > 1 else "general"
        chunks.extend(
            chunk_markdown(
                path.read_text(encoding="utf-8"),
                source=relative.as_posix(),
                category=category,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def load_all(
    *,
    tickets_path: Path | str = TICKETS_PATH,
    accounts_path: Path | str = ACCOUNTS_PATH,
    kb_root: Path | str = KB_DIR,
    fallback_to_company: bool = False,
) -> Dataset:
    """Load tickets, accounts and the knowledge base in one call."""
    tickets = load_tickets(tickets_path)
    accounts = load_accounts(accounts_path)
    enriched = join_tickets_to_accounts(tickets, accounts, fallback_to_company=fallback_to_company)

    return Dataset(
        tickets=tickets,
        accounts=accounts,
        enriched_tickets=enriched,
        kb_chunks=load_knowledge_base(kb_root),
        join_report=summarise_join(enriched),
    )


if __name__ == "__main__":
    data = load_all()
    print(f"tickets  : {len(data.tickets)}")
    print(f"accounts : {len(data.accounts)}")
    print(f"join     : {data.join_report.summary()}")
    if data.join_report.unmatched_account_ids:
        sample = ", ".join(data.join_report.unmatched_account_ids[:5])
        print(f"           unmatched ids (sample): {sample}")

    with_company = summarise_join(
        join_tickets_to_accounts(data.tickets, data.accounts, fallback_to_company=True)
    )
    print(f"           with company fallback: {with_company.summary()}")

    print(f"kb chunks: {len(data.kb_chunks)}")
    for category, count in sorted(Counter(c.category for c in data.kb_chunks).items()):
        print(f"           {category:<16} {count:>3}")
    longest = max(data.kb_chunks, key=lambda c: c.char_count)
    print(f"           largest chunk: {longest.char_count} chars - {longest.breadcrumb}")
