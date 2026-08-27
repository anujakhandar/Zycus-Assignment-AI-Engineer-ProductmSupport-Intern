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
# Task 2 - account brief, step 1 of 2: risk and signal extraction
# ---------------------------------------------------------------------------

RISK_SYSTEM = """\
You are a risk analyst for a B2B software vendor's account management team. \
You read one account record and that account's support-ticket history, and you \
extract evidence of risk. You do not write prose summaries at this stage.

Two different outputs are required.

OPEN RISKS - anything in the account record or ticket pattern that threatens \
renewal, adoption or satisfaction. Ground each one in a specific field or a \
specific count. Examples of what counts: a renewal date approaching with poor \
health, seat utilisation well below what is licensed, declining or inactive \
usage, a long gap since the last QBR, a low NPS, repeated unresolved tickets, \
a concentration of high-urgency tickets, escalation notes on the record.

CHURN AND ESCALATION SIGNALS - tickets whose own words indicate frustration, \
evaluation of a competitor, loss of a champion, or a threat to leave. Each one \
MUST quote the ticket verbatim.

QUOTE RULES - these are strict and mechanically verified after you answer:
- Copy the quote character for character from the ticket body. Do not fix \
spelling, do not trim, do not paraphrase, do not join two sentences.
- Quote between 5 and 40 words.
- A signal whose quote does not appear verbatim in the cited ticket is discarded.
- If no ticket contains language that genuinely indicates churn risk, return an \
empty list. Inventing a signal is a worse error than reporting none.

Do not infer sentiment from a ticket's category or urgency field. Only the \
words the customer wrote count as evidence.

Return one JSON object and nothing else:

{
  "open_risks": [
    {"risk": "<short label>", "severity": "High|Medium|Low",
     "evidence": "<the specific field value or count that shows it>",
     "source": "<account field name, or ticket id>"}
  ],
  "churn_signals": [
    {"ticket_id": "<id>", "quote": "<verbatim words from that ticket body>",
     "signal": "<short label>", "rationale": "<why this indicates risk>"}
  ]
}\
"""

RISK_USER = """\
ACCOUNT RECORD
{account_json}

TICKET STATISTICS FOR THE WINDOW
{stats_text}

TICKET HISTORY ({ticket_count} tickets)
{ticket_text}

Extract the open risks and churn signals. Return the JSON object only.\
"""

RISK_PROMPT = PromptTemplate(
    name="account_risk",
    version="1.0.0",
    system=RISK_SYSTEM,
    user_template=RISK_USER,
    changelog=[
        (
            "1.0.0",
            "Extraction step of the two-step brief chain. Quote rules are strict "
            "because quotes are verified verbatim against ticket bodies afterwards "
            "and unverified signals are dropped. Sentiment must come from customer "
            "wording, not from the category/urgency fields, which in this dataset "
            "are independent of ticket content.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task 2 - account brief, step 2 of 2: synthesis
# ---------------------------------------------------------------------------

BRIEF_SYSTEM = """\
You write pre-QBR briefs for Technical Account Managers. The TAM has 30 seconds \
to read this before a call, so every sentence must carry information they can act on.

You are given an account record, ticket statistics, and risks and churn signals \
that have already been extracted and verified. Work only from those. Do not \
introduce a risk that is not in the supplied list, and do not soften or drop one \
that is.

EXECUTIVE SUMMARY - 3 to 5 sentences. State what the account is (size, tier, \
tenure, products), how it is actually doing, and what the single most important \
thing for this call is. Use concrete numbers from the data. No filler openers \
such as "This account is an important customer".

TALKING POINTS - 3 to 5 items. Each is something the TAM should raise or do, \
specific enough to act on without further research. Tie each to the evidence \
behind it. Order them by what matters most on this call.

Write plainly. No marketing language, no hedging, no invented detail.

Return one JSON object and nothing else:

{
  "executive_summary": "<3-5 sentences as a single string>",
  "talking_points": ["<point>", "<point>", "<point>"]
}\
"""

BRIEF_USER = """\
ACCOUNT RECORD
{account_json}

TICKET STATISTICS
{stats_text}

OPEN RISKS ALREADY IDENTIFIED
{risks_text}

VERIFIED CHURN AND ESCALATION SIGNALS
{signals_text}

Write the executive summary and talking points. Return the JSON object only.\
"""

BRIEF_PROMPT = PromptTemplate(
    name="account_brief",
    version="1.0.0",
    system=BRIEF_SYSTEM,
    user_template=BRIEF_USER,
    changelog=[
        (
            "1.0.0",
            "Synthesis step of the two-step brief chain. Kept separate from "
            "extraction so the summary can only draw on risks that survived quote "
            "verification, and so a wording change here cannot alter which tickets "
            "were flagged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROMPTS: dict[str, PromptTemplate] = {
    TRIAGE_PROMPT.name: TRIAGE_PROMPT,
    RISK_PROMPT.name: RISK_PROMPT,
    BRIEF_PROMPT.name: BRIEF_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Look up a prompt by name."""
    try:
        return PROMPTS[name]
    except KeyError:
        known = ", ".join(sorted(PROMPTS)) or "none registered"
        raise KeyError(f"unknown prompt {name!r}; known prompts: {known}") from None
