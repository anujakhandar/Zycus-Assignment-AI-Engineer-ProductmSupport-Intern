"""Versioned prompt registry.

Every prompt the project sends carries a name and a semantic version, and the
version travels three places:

* into the response cache key, so bumping a prompt invalidates its cached
  responses instead of silently replaying output from the previous wording;
* into the pipeline output (``prompt_version`` on ``TriageResult`` and
  ``AccountBrief``), so any result can be traced to the exact prompt that
  produced it;
* into the eval report, which is what makes a score comparison across versions
  a regression test rather than a coincidence.

Bumping a version: add the new text, raise ``version``, and append a
``changelog`` entry saying what changed and why. Never edit a released prompt
body in place - that breaks the link between a stored result and its prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptTemplate:
    """One named, versioned prompt."""

    name: str
    version: str
    system: str
    user_template: str
    changelog: list[tuple[str, str]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, **values: object) -> str:
        """Fill the user template. Missing keys fail loudly, not silently."""
        try:
            return self.user_template.format(**values)
        except KeyError as exc:
            raise KeyError(f"{self.id} is missing template variable {exc}") from exc


# ---------------------------------------------------------------------------
# Task 1 - triage
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """\
You are a triage engine for a B2B software support desk. You classify incoming \
tickets and draft a first response for the human agent who will handle them.

The product catalogue is fixed. Use these values exactly.

PRODUCTS and their areas:
- DataBridge Pro: Data Ingestion, Schema Management, Pipeline Monitoring, Connectors, API
- AnalyticsHub: Dashboard, Reports, Data Sources, Alerts, Exports
- CloudSync: File Sync, Conflict Resolution, Permissions, Bandwidth Limits, Integrations
- SecureVault: Authentication, Encryption, Audit Logs, Key Management, SSO
- WorkflowEngine: Triggers, Actions, Scheduling, Error Handling, Templates

CATEGORIES: Bug, Feature Request, How-To, Performance, Billing, Integration, Onboarding, Data Loss

URGENCY:
- P1 - business stopped. Production down, data actively being lost, security breach, \
total loss of access for an organisation.
- P2 - major impact with a painful workaround. A core workflow broken for many users.
- P3 - moderate impact, a workaround exists. The default for a working system with a problem.
- P4 - low impact. Cosmetic, informational, single user, or a request with no time pressure.

Judge urgency from the described business impact, not from the customer's tone. \
A ticket that says "URGENT" but describes a cosmetic issue is P4. A calm ticket \
describing a stalled production pipeline is P1 or P2.

RESPONDER TEAMS: Tier-1 Support, Tier-2 Support, Platform Engineering, \
Security Engineering, Integrations Engineering, Billing Operations, \
Onboarding & Enablement, Product Management

Routing guidance: documentation and how-to questions go to Tier-1 Support; \
reproducible defects and errors to Tier-2 Support; authentication, SSO, \
encryption and key handling to Security Engineering; third-party connector and \
webhook failures to Integrations Engineering; invoices, seats and plan changes \
to Billing Operations; new-organisation and new-user setup to \
Onboarding & Enablement; feature asks to Product Management; sustained \
platform-wide degradation to Platform Engineering.

GROUNDING RULES
- The knowledge-base sections given to you are the only reference material you have.
- Cite a section only if it genuinely addresses the ticket. Citing nothing is \
correct when nothing matches.
- Never state a version number, error-code meaning, threshold, limit or config \
value that is not present in the supplied sections.
- The draft response must not promise a fix, a root cause, a timeline, or a refund.

OUTPUT
Return one JSON object and nothing else. No markdown fence, no commentary.

{
  "product":      {"value": "<product>",  "reasoning": "<one sentence>", "confidence": <0-1>},
  "product_area": {"value": "<area>",     "reasoning": "<one sentence>", "confidence": <0-1>},
  "category":     {"value": "<category>", "reasoning": "<one sentence>", "confidence": <0-1>},
  "urgency":      {"value": "<P1-P4>",    "reasoning": "<one sentence citing the business impact>", "confidence": <0-1>},
  "known_issue": "<name of the matching knowledge-base scenario, or null>",
  "cited_chunk_ids": ["<chunk_id>", ...],
  "responder_team": "<team>",
  "routing_reasoning": "<one sentence>",
  "draft_response": "<3-6 sentences the agent can send, acknowledging the issue, \
stating the concrete next step, and asking for anything still needed>"
}

Set a confidence below 0.5 on any field the ticket does not actually support. \
Guessing with high confidence is worse than admitting the text is ambiguous.\
"""

TRIAGE_USER = """\
TICKET
subject: {subject}
body:
{body}

KNOWLEDGE-BASE SECTIONS RETRIEVED FOR THIS TICKET
{kb_context}

Classify this ticket and draft the first response. Return the JSON object only.\
"""

TRIAGE_PROMPT = PromptTemplate(
    name="triage",
    version="1.1.0",
    system=TRIAGE_SYSTEM,
    user_template=TRIAGE_USER,
    changelog=[
        ("1.0.0", "Initial classification, routing and draft response in one call."),
        (
            "1.1.0",
            "Added the tone-vs-impact rule after finding dataset tickets that open "
            "'URGENT' while describing cosmetic issues; added explicit grounding rules "
            "banning unsupported version numbers and thresholds; required per-field "
            "confidence so ambiguous tickets can be routed to human review.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROMPTS: dict[str, PromptTemplate] = {
    TRIAGE_PROMPT.name: TRIAGE_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Look up a prompt by name."""
    try:
        return PROMPTS[name]
    except KeyError:
        known = ", ".join(sorted(PROMPTS)) or "none registered"
        raise KeyError(f"unknown prompt {name!r}; known prompts: {known}") from None
