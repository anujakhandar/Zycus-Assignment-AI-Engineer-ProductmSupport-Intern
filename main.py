"""Single entry point for every task in this project.

    python main.py triage --ticket TKT-10042
    python main.py triage --text "Our pipeline has been failing since this morning"
    python main.py triage --file ticket.json --json

Run ``python main.py --help`` for the full list of commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Console encoding on Windows defaults to cp1252, which cannot print the
# em dashes and arrows in the knowledge base or the model's output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_ticket(args: argparse.Namespace):
    """Resolve --ticket / --text / --file into something triage_ticket accepts."""
    from src.data_loader import load_tickets
    from src.schemas import TicketInput

    if args.text:
        return TicketInput(subject=args.subject or "", body=args.text)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return TicketInput.model_validate(json.loads(raw))
        return TicketInput(subject=args.subject or "", body=raw)

    tickets = load_tickets()
    if args.ticket:
        wanted = args.ticket.strip().upper()
        for ticket in tickets:
            if ticket.ticket_id.upper() == wanted:
                return ticket
        raise SystemExit(f"no ticket with id {args.ticket!r} in data/tickets.json")

    return tickets[0]  # default sample so a bare `main.py triage` still runs


def cmd_triage(args: argparse.Namespace) -> int:
    from src.triage import triage_ticket

    ticket = _load_ticket(args)
    result = triage_ticket(ticket, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return 0

    print("=" * 72)
    print(f"TRIAGE  {result.ticket_id or '(ad-hoc ticket)'}")
    print("=" * 72)
    for label, field in [
        ("Product", result.product),
        ("Product area", result.product_area),
        ("Category", result.category),
        ("Urgency", result.urgency),
    ]:
        print(f"\n{label:<14} {field.value}   (confidence {field.confidence:.2f})")
        print(f"{'':<14} {field.reasoning}")

    print(f"\n{'Responder':<14} {result.responder_team}")
    if result.routing_reasoning:
        print(f"{'':<14} {result.routing_reasoning}")

    print(f"\n{'Known issue':<14} {result.known_issue or '(none matched)'}")

    print(f"\n{'KB sections':<14} ", end="")
    if result.kb_references:
        print()
        for reference in result.kb_references:
            codes = f"  [{', '.join(reference.matched_codes)}]" if reference.matched_codes else ""
            print(f"{'':<14} - {reference.breadcrumb}{codes}")
            print(f"{'':<16} {reference.source}  (score {reference.score})")
    else:
        print("(nothing in the knowledge base supports this ticket)")

    print("\nDraft first response")
    print("-" * 72)
    print(result.draft_response)
    print("-" * 72)

    flag = "YES" if result.needs_human_review else "no"
    print(f"\nOverall confidence {result.overall_confidence:.2f}   Human review: {flag}")
    for reason in result.review_reasons:
        print(f"   - {reason}")

    print(f"\nprompt {result.prompt_version}   model {result.model}   cached {result.cached}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Support triage and TAM tooling built on the mock dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    triage = subparsers.add_parser("triage", help="Task 1 - triage a support ticket")
    source = triage.add_mutually_exclusive_group()
    source.add_argument("--ticket", help="ticket_id from data/tickets.json, e.g. TKT-10042")
    source.add_argument("--text", help="raw ticket body as a string")
    source.add_argument("--file", help="path to a .json ticket or a .txt body")
    triage.add_argument("--subject", default="", help="subject line, with --text or a .txt --file")
    triage.add_argument("--top-k", type=int, default=4, help="knowledge-base sections to retrieve")
    triage.add_argument("--json", action="store_true", help="emit the raw structured output")
    triage.set_defaults(func=cmd_triage)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from src.llm_client import LLMError

    try:
        return args.func(args)
    except LLMError as exc:
        # A missing key, an exhausted budget or an offline miss are all
        # operator problems, not bugs. Report them plainly rather than
        # dumping a traceback on someone running this for the first time.
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
