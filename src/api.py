"""REST interface over both pipelines.

    python main.py serve
    then open http://127.0.0.1:8000/docs

The brief allows triage to be exposed either as a callable Python function or as
a REST endpoint. Both exist here: :func:`src.triage.triage_ticket` is importable
and used directly by the CLI, the Streamlit app and the eval harness, and this
module wraps it for callers that want HTTP.

The knowledge base index is built once at startup and shared by every request,
rather than rebuilt per call.

Model errors map to 503 rather than 500. A missing key, an exhausted call budget
or an uncached request in offline mode are all service configuration problems the
caller can act on, not server faults.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.account_brief import AccountBriefError, build_account_brief
from src.data_loader import load_accounts, load_all
from src.llm_client import LLMError
from src.retrieval import get_index
from src.schemas import AccountBrief, TicketInput, TriageResult
from src.triage import triage_ticket

_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the index and load the dataset once at startup, not per request."""
    _state["index"] = get_index()
    _state["dataset"] = load_all()
    yield
    _state.clear()


app = FastAPI(
    title="Support triage and account health tooling",
    description="Ticket triage for support engineers, account briefs for TAMs.",
    version="1.0.0",
    lifespan=lifespan,
)


class BriefRequest(BaseModel):
    account_id: str = Field(description="Account id or company name, e.g. ACC-1785")
    days: Optional[int] = Field(default=90, description="Ticket window in days. Null means all history.")


@app.get("/health")
def health() -> dict:
    """Liveness plus what the process has loaded."""
    dataset = _state.get("dataset")
    return {
        "status": "ok",
        "tickets": len(dataset.tickets) if dataset else 0,
        "accounts": len(dataset.accounts) if dataset else 0,
        "kb_sections": len(_state["index"]) if _state.get("index") else 0,
    }


@app.post("/triage", response_model=TriageResult)
def triage(ticket: TicketInput) -> TriageResult:
    """Triage one ticket from a subject and body."""
    try:
        return triage_ticket(ticket, index=_state.get("index"))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/accounts")
def accounts() -> list[dict]:
    """Every account id with the fields needed to choose one."""
    return [
        {
            "account_id": a.account_id,
            "company": a.company,
            "plan_tier": a.plan_tier,
            "health_status": a.health_status,
            "arr_usd": a.arr_usd,
            "renewal_date": a.renewal_date,
        }
        for a in sorted(load_accounts(), key=lambda a: a.company or a.account_id)
    ]


@app.post("/brief", response_model=AccountBrief)
def brief(request: BriefRequest) -> AccountBrief:
    """Build the account brief for one account."""
    try:
        return build_account_brief(
            request.account_id,
            dataset=_state.get("dataset"),
            days=request.days,
        )
    except AccountBriefError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
