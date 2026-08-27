"""Structured output contracts for the triage and account-brief pipelines.

Every model response is parsed into one of these before it leaves the pipeline,
so an endpoint, the eval harness and the CLI all see the same shape. Enum
values are pinned to the vocabularies that actually occur in the dataset (see
DATA_SCHEMA.md) rather than left as free strings - a model that invents a
sixth product or a "P0" is a validation failure, not a silent bad record.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Vocabularies. These mirror the enum values documented in DATA_SCHEMA.md and
# confirmed present in data/tickets.json.
Product = Literal[
    "DataBridge Pro",
    "AnalyticsHub",
    "CloudSync",
    "SecureVault",
    "WorkflowEngine",
]

Category = Literal[
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
]

Urgency = Literal["P1", "P2", "P3", "P4"]

ResponderTeam = Literal[
    "Tier-1 Support",
    "Tier-2 Support",
    "Platform Engineering",
    "Security Engineering",
    "Integrations Engineering",
    "Billing Operations",
    "Onboarding & Enablement",
    "Product Management",
]

# product_area values that exist in the dataset, grouped by product. Also used
# to sanity-check that a predicted area belongs to the predicted product.
PRODUCT_AREAS: dict[str, list[str]] = {
    "DataBridge Pro": [
        "Data Ingestion",
        "Schema Management",
        "Pipeline Monitoring",
        "Connectors",
        "API",
    ],
    "AnalyticsHub": ["Dashboard", "Reports", "Data Sources", "Alerts", "Exports"],
    "CloudSync": [
        "File Sync",
        "Conflict Resolution",
        "Permissions",
        "Bandwidth Limits",
        "Integrations",
    ],
    "SecureVault": [
        "Authentication",
        "Encryption",
        "Audit Logs",
        "Key Management",
        "SSO",
    ],
    "WorkflowEngine": [
        "Triggers",
        "Actions",
        "Scheduling",
        "Error Handling",
        "Templates",
    ],
}

ALL_PRODUCT_AREAS = sorted({area for areas in PRODUCT_AREAS.values() for area in areas})


# ---------------------------------------------------------------------------
# Task 1 - triage
# ---------------------------------------------------------------------------


class TicketInput(BaseModel):
    """A ticket arriving at the triage endpoint.

    Only ``body`` is required: the brief asks for raw free text, so a bare
    string with no metadata has to work. Anything else the caller happens to
    know is optional context, never a requirement.
    """

    model_config = ConfigDict(extra="ignore")

    subject: str = ""
    body: str
    ticket_id: Optional[str] = None
    account_id: Optional[str] = None
    company: Optional[str] = None
    channel: Optional[str] = None

    @field_validator("body")
    @classmethod
    def _body_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ticket body is empty - nothing to triage")
        return value

    @property
    def text(self) -> str:
        return f"{self.subject}\n\n{self.body}".strip()


class KBReference(BaseModel):
    """A knowledge-base chunk cited by the triage result."""

    chunk_id: str
    source: str
    breadcrumb: str
    score: float = 0.0
    matched_codes: list[str] = Field(default_factory=list)
    excerpt: str = ""


class Classification(BaseModel):
    """One predicted field plus the reasoning the brief asks to be surfaced."""

    value: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class TriageResult(BaseModel):
    """The structured output of Task 1."""

    model_config = ConfigDict(extra="ignore")

    ticket_id: Optional[str] = None

    product: Classification
    product_area: Classification
    category: Classification
    urgency: Classification

    known_issue: Optional[str] = Field(
        default=None,
        description="Name of the matching KB scenario, or null when nothing matches.",
    )
    kb_references: list[KBReference] = Field(default_factory=list)

    responder_team: str
    routing_reasoning: str = ""
    draft_response: str

    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    prompt_version: str = ""
    model: str = ""
    cached: bool = False

    @property
    def summary_line(self) -> str:
        return (
            f"{self.urgency.value} | {self.category.value} | "
            f"{self.product.value} / {self.product_area.value} -> {self.responder_team}"
        )


# ---------------------------------------------------------------------------
# Task 2 - account brief
# ---------------------------------------------------------------------------


class RiskFlag(BaseModel):
    """One open risk, tied back to the field or ticket that evidences it."""

    risk: str
    severity: Literal["High", "Medium", "Low"] = "Medium"
    evidence: str
    source: str = Field(
        default="",
        description="Where the evidence came from, e.g. an account field or a ticket_id.",
    )


class ChurnSignal(BaseModel):
    """A churn or escalation signal justified by a verbatim ticket quote.

    ``quote`` must appear character-for-character in the cited ticket; the
    pipeline verifies this after generation and drops any signal that fails,
    which is what stops the model paraphrasing a quote into existence.
    """

    ticket_id: str
    quote: str
    signal: str
    rationale: str
    verified: bool = False


class TicketStats(BaseModel):
    """Counts over the ticket window the brief was built from."""

    window_days: Optional[int] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    total: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_urgency: dict[str, int] = Field(default_factory=dict)
    by_product: dict[str, int] = Field(default_factory=dict)
    unresolved: int = 0
    mean_satisfaction: Optional[float] = None


class AccountBrief(BaseModel):
    """The structured output of Task 2, rendered to markdown for the TAM."""

    model_config = ConfigDict(extra="ignore")

    account_id: str
    company: str

    executive_summary: str
    open_risks: list[RiskFlag] = Field(default_factory=list)
    churn_signals: list[ChurnSignal] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)

    stats: TicketStats = Field(default_factory=TicketStats)
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Anything the brief could not establish, stated rather than guessed.",
    )

    prompt_version: str = ""
    model: str = ""
    cached: bool = False

    def to_markdown(self) -> str:
        """Render the three sections the brief asks for."""
        lines = [f"# Account Brief - {self.company} ({self.account_id})", ""]

        lines += ["## 1. Executive summary", "", self.executive_summary, ""]

        lines += ["## 2. Open risks & flagged issues", ""]
        if self.open_risks:
            for risk in self.open_risks:
                source = f" _(source: {risk.source})_" if risk.source else ""
                lines.append(f"- **[{risk.severity}] {risk.risk}**{source}")
                lines.append(f"  - Evidence: {risk.evidence}")
        else:
            lines.append("- No open risks identified from the available data.")
        lines.append("")

        if self.churn_signals:
            lines += ["### Churn / escalation signals", ""]
            for signal in self.churn_signals:
                lines.append(f"- **{signal.signal}** ({signal.ticket_id})")
                lines.append(f'  - Quote: "{signal.quote}"')
                lines.append(f"  - Why it matters: {signal.rationale}")
            lines.append("")

        lines += ["## 3. Recommended talking points", ""]
        if self.talking_points:
            lines += [f"{i}. {point}" for i, point in enumerate(self.talking_points, 1)]
        else:
            lines.append("_No talking points generated._")
        lines.append("")

        if self.data_gaps:
            lines += ["---", "", "**Data gaps**", ""]
            lines += [f"- {gap}" for gap in self.data_gaps]
            lines.append("")

        return "\n".join(lines)
