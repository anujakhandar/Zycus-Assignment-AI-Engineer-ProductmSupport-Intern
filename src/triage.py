"""Task 1 - intelligent ticket triage.

Pipeline for one ticket:

1. **Deterministic pre-pass.** Pull error codes out of the text and guess the
   product and area by exact name match. Cheap, and it gives retrieval a hint
   before any model has seen the ticket.
2. **Retrieval.** BM25 over the knowledge base, boosted by the hinted product
   and by exact error-code matches (:mod:`src.retrieval`).
3. **One model call.** Classification with reasoning, known-issue match,
   responder team, and the draft first response, all in a single request
   against the retrieved sections.
4. **Validation and repair.** Every predicted value is checked against the
   dataset's real vocabularies; anything invalid falls back to a deterministic
   rule rather than being passed through. Citations are intersected with what
   was actually retrieved, so the model cannot cite a document it was not shown.
5. **Review triage.** Low confidence, no supporting document, a very short
   ticket, or a P1 all raise ``needs_human_review``.

One call rather than a classify-then-draft chain is a deliberate latency choice;
see the design note.

The classifier does not read ``category``/``urgency`` from the dataset, and the
brief asks for triage "without any human labelling". That is just as well: in
this dataset those two fields are independent of the ticket text, so they are
not usable as labels. :func:`compare_to_recorded` reports the disagreement
rather than treating either side as truth.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from prompts import get_prompt
from src.data_loader import Ticket
from src.llm_client import LLMClient
from src.retrieval import (
    KnowledgeBaseIndex,
    RetrievalHit,
    extract_error_codes,
    format_hits_for_prompt,
    get_index,
)
from src.schemas import (
    ALL_PRODUCT_AREAS,
    PRODUCT_AREAS,
    Classification,
    KBReference,
    TicketInput,
    TriageResult,
)

CATEGORIES = [
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
]
URGENCIES = ["P1", "P2", "P3", "P4"]
TEAMS = [
    "Tier-1 Support",
    "Tier-2 Support",
    "Platform Engineering",
    "Security Engineering",
    "Integrations Engineering",
    "Billing Operations",
    "Onboarding & Enablement",
    "Product Management",
]

# Deterministic routing, used when the model returns a team outside the
# catalogue. Order matters: the first matching rule wins.
_TEAM_BY_CATEGORY = {
    "Billing": "Billing Operations",
    "Feature Request": "Product Management",
    "How-To": "Tier-1 Support",
    "Onboarding": "Onboarding & Enablement",
    "Integration": "Integrations Engineering",
    "Performance": "Platform Engineering",
    "Data Loss": "Tier-2 Support",
    "Bug": "Tier-2 Support",
}
_SECURITY_AREAS = {"Authentication", "SSO", "Encryption", "Key Management", "Audit Logs"}

MIN_BODY_CHARS = 80
LOW_CONFIDENCE = 0.6


class TriageError(RuntimeError):
    """Raised when a response cannot be turned into a TriageResult."""


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def guess_product(text: str) -> Optional[str]:
    """Exact product-name match, longest name first so 'DataBridge Pro' wins."""
    lowered = (text or "").lower()
    for product in sorted(PRODUCT_AREAS, key=len, reverse=True):
        if product.lower() in lowered:
            return product
    return None


def guess_product_area(text: str, product: Optional[str] = None) -> Optional[str]:
    """Exact area-name match, restricted to the product's areas when known."""
    lowered = (text or "").lower()
    candidates = PRODUCT_AREAS.get(product or "", ALL_PRODUCT_AREAS)
    for area in sorted(candidates, key=len, reverse=True):
        if area.lower() in lowered:
            return area
    return None


def fallback_team(category: Optional[str], product_area: Optional[str]) -> str:
    """Deterministic routing used when the model's team is not in the catalogue."""
    if product_area in _SECURITY_AREAS:
        return "Security Engineering"
    return _TEAM_BY_CATEGORY.get(category or "", "Tier-1 Support")


def _coerce(value: Any, allowed: list[str], fallback: str) -> tuple[str, bool]:
    """Match ``value`` against ``allowed`` case-insensitively.

    Returns ``(value, was_valid)``. An unrecognised value is replaced by
    ``fallback`` and reported, never passed through.
    """
    if isinstance(value, str):
        stripped = value.strip()
        for option in allowed:
            if stripped.lower() == option.lower():
                return option, True
    return fallback, False


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    Tolerates a markdown fence or stray prose around the object, because a
    single malformed response should not lose a whole run.
    """
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise TriageError(f"no JSON object found in response: {cleaned[:200]!r}")

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise TriageError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise TriageError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _classification(payload: Any, allowed: list[str], fallback: str) -> tuple[Classification, bool]:
    """Build a Classification from one field of the model's JSON."""
    if not isinstance(payload, dict):
        payload = {"value": payload}

    value, valid = _coerce(payload.get("value"), allowed, fallback)
    reasoning = str(payload.get("reasoning") or "").strip()
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    if not valid:
        confidence = min(confidence, 0.3)
        reasoning = (reasoning + " [value not in catalogue; replaced by rule-based fallback]").strip()

    return Classification(value=value, reasoning=reasoning, confidence=confidence), valid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def triage_ticket(
    ticket: TicketInput | Ticket | dict | str,
    *,
    index: Optional[KnowledgeBaseIndex] = None,
    llm: Optional[LLMClient] = None,
    top_k: int = 4,
) -> TriageResult:
    """Triage one ticket. This is the callable the brief asks for.

    Accepts raw text, a ``{"subject": ..., "body": ...}`` dict, a
    :class:`~src.schemas.TicketInput`, or a dataset :class:`~src.data_loader.Ticket`.
    """
    ticket_input = _as_ticket_input(ticket)
    index = index or get_index()
    llm = llm or LLMClient()
    prompt = get_prompt("triage")

    # 1. deterministic pre-pass
    hinted_product = guess_product(ticket_input.text)
    hinted_area = guess_product_area(ticket_input.text, hinted_product)
    codes = extract_error_codes(ticket_input.text)

    # 2. retrieval
    hits = index.retrieve_for_ticket(
        ticket_input.subject,
        ticket_input.body,
        top_k=top_k,
        product=hinted_product,
        product_area=hinted_area,
    )

    # 3. one model call
    user_prompt = prompt.render(
        subject=ticket_input.subject or "(no subject)",
        body=ticket_input.body,
        kb_context=format_hits_for_prompt(hits),
    )
    raw, meta = llm.complete(
        user_prompt,
        system=prompt.system,
        tags={"prompt": prompt.id, "task": "triage"},
    )
    payload = parse_json_response(raw)

    # 4. validation and repair
    review_reasons: list[str] = []

    product, product_valid = _classification(
        payload.get("product"), list(PRODUCT_AREAS), hinted_product or "DataBridge Pro"
    )
    if not product_valid:
        review_reasons.append("product was not a catalogue value")

    allowed_areas = PRODUCT_AREAS.get(product.value, ALL_PRODUCT_AREAS)
    area_fallback = (
        hinted_area if hinted_area in allowed_areas else allowed_areas[0]
    )
    product_area, area_valid = _classification(payload.get("product_area"), allowed_areas, area_fallback)
    if not area_valid:
        review_reasons.append(f"product_area was not one of {product.value}'s areas")

    category, category_valid = _classification(payload.get("category"), CATEGORIES, "How-To")
    if not category_valid:
        review_reasons.append("category was not a catalogue value")

    urgency, urgency_valid = _classification(payload.get("urgency"), URGENCIES, "P3")
    if not urgency_valid:
        review_reasons.append("urgency was not P1-P4")

    team, team_valid = _coerce(
        payload.get("responder_team"), TEAMS, fallback_team(category.value, product_area.value)
    )
    if not team_valid:
        review_reasons.append("responder team was not a catalogue value; routed by rule")

    # Citations are intersected with what was retrieved: the model cannot cite
    # a document it was never shown.
    references = _build_references(payload.get("cited_chunk_ids"), hits)

    draft = str(payload.get("draft_response") or "").strip()
    if not draft:
        draft = _fallback_draft(ticket_input, team)
        review_reasons.append("model returned no draft response; template used")

    confidences = [product.confidence, product_area.confidence, category.confidence, urgency.confidence]
    overall = round(sum(confidences) / len(confidences), 3)

    if overall < LOW_CONFIDENCE:
        review_reasons.append(f"mean confidence {overall:.2f} below {LOW_CONFIDENCE}")
    if not references:
        review_reasons.append("no knowledge-base section supports this triage")
    if len(ticket_input.body.strip()) < MIN_BODY_CHARS:
        review_reasons.append(f"ticket body under {MIN_BODY_CHARS} characters")
    if urgency.value == "P1":
        review_reasons.append("P1 always gets human confirmation")

    known_issue = payload.get("known_issue")
    if isinstance(known_issue, str) and known_issue.strip().lower() in {"", "null", "none"}:
        known_issue = None

    return TriageResult(
        ticket_id=ticket_input.ticket_id,
        product=product,
        product_area=product_area,
        category=category,
        urgency=urgency,
        known_issue=known_issue if isinstance(known_issue, str) else None,
        kb_references=references,
        responder_team=team,
        routing_reasoning=str(payload.get("routing_reasoning") or "").strip(),
        draft_response=draft,
        overall_confidence=overall,
        needs_human_review=bool(review_reasons),
        review_reasons=review_reasons,
        prompt_version=prompt.id,
        model=meta.model,
        cached=meta.cached,
    )


def _as_ticket_input(ticket: TicketInput | Ticket | dict | str) -> TicketInput:
    """Normalise every accepted input shape into a TicketInput."""
    if isinstance(ticket, TicketInput):
        return ticket
    if isinstance(ticket, Ticket):
        return TicketInput(
            subject=ticket.subject,
            body=ticket.body,
            ticket_id=ticket.ticket_id,
            account_id=ticket.account_id,
            company=ticket.company,
            channel=ticket.channel,
        )
    if isinstance(ticket, dict):
        return TicketInput.model_validate(ticket)
    if isinstance(ticket, str):
        return TicketInput(body=ticket)
    raise TypeError(f"cannot triage a {type(ticket).__name__}")


def _build_references(cited: Any, hits: list[RetrievalHit]) -> list[KBReference]:
    """Turn cited chunk ids into references, keeping only retrieved chunks."""
    by_id = {hit.chunk.chunk_id: hit for hit in hits}
    ids = [c for c in cited if isinstance(c, str)] if isinstance(cited, list) else []

    references: list[KBReference] = []
    for chunk_id in ids:
        hit = by_id.get(chunk_id)
        if hit is None:
            continue
        references.append(
            KBReference(
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.source,
                breadcrumb=hit.chunk.breadcrumb,
                score=hit.score,
                matched_codes=hit.matched_codes,
                excerpt=hit.excerpt(300),
            )
        )
    return references


def _fallback_draft(ticket: TicketInput, team: str) -> str:
    """Neutral holding reply, used only when the model returns no draft."""
    subject = ticket.subject or "your request"
    return (
        f"Thanks for getting in touch about {subject}. "
        f"We have logged this and it is being routed to our {team} team for review. "
        "So that we can move quickly, could you confirm the environment affected and "
        "roughly when the behaviour started? We will follow up as soon as we have an update."
    )


def compare_to_recorded(result: TriageResult, ticket: Ticket) -> dict[str, Any]:
    """Compare a prediction against the fields recorded on a dataset ticket.

    Reported as agreement, not accuracy. In this dataset ``category`` and
    ``urgency`` are statistically independent of the ticket text, so a
    disagreement is evidence about the data, not necessarily about the model.
    ``product`` and ``product_area`` are genuinely consistent in the data and
    are the fields where agreement is meaningful.
    """
    return {
        "ticket_id": ticket.ticket_id,
        "product": {"predicted": result.product.value, "recorded": ticket.product,
                    "agrees": result.product.value == ticket.product},
        "product_area": {"predicted": result.product_area.value, "recorded": ticket.product_area,
                         "agrees": result.product_area.value == ticket.product_area},
        "category": {"predicted": result.category.value, "recorded": ticket.category,
                     "agrees": result.category.value == ticket.category},
        "urgency": {"predicted": result.urgency.value, "recorded": ticket.urgency,
                    "agrees": result.urgency.value == ticket.urgency},
    }
