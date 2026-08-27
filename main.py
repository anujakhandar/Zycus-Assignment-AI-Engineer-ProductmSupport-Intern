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

# Checked before anything else so an unsupported interpreter reports itself
# clearly instead of failing later inside a dependency. 3.10 is the floor set by
# google-genai, python-dotenv and streamlit alike.
MINIMUM_PYTHON = (3, 10)
if sys.version_info < MINIMUM_PYTHON:
    sys.exit(
        f"This project needs Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer. "
        f"You are running {sys.version.split()[0]}."
    )

# Console encoding on Windows defaults to cp1252, which cannot print every
# character that appears in the knowledge base or in model output.
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


def cmd_brief(args: argparse.Namespace) -> int:
    from src.account_brief import build_account_brief

    brief = build_account_brief(args.account, days=None if args.all_history else args.days)

    if args.json:
        print(json.dumps(brief.model_dump(), indent=2, ensure_ascii=False))
        return 0

    print(brief.to_markdown())
    print(f"\nprompt chain {brief.prompt_version}   model {brief.model}   cached {brief.cached}")
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    """List account ids, so `brief` can be run without opening the JSON."""
    from src.data_loader import load_accounts

    accounts = sorted(load_accounts(), key=lambda a: a.company or a.account_id)
    print(f"{'account_id':<12} {'company':<28} {'tier':<14} {'health':<10} {'ARR':>10}")
    print("-" * 78)
    for account in accounts:
        print(
            f"{account.account_id:<12} {(account.company or ''):<28} "
            f"{(account.plan_tier or ''):<14} {(account.health_status or ''):<10} "
            f"{account.arr_usd or 0:>10,.0f}"
        )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evals.runner import run_all, write_reports

    print(f"Running evaluation harness (judge {'on' if not args.no_judge else 'off'})...")
    report = run_all(task=args.task, use_judge=not args.no_judge)
    json_path, md_path = write_reports(report)

    summary = report["summary"]
    print("\n" + "=" * 72)
    print(
        f"{summary['passed']}/{summary['total_cases']} cases passed "
        f"({summary['pass_rate']:.0%})   mean quality {summary['mean_quality']:.3f}"
    )
    print(
        f"adversarial {summary['adversarial_passed']}/{summary['adversarial_cases']}   "
        f"live calls {summary['live_api_calls']}   cache hits {summary['cache_hits']}"
    )
    print("=" * 72)
    for result in report["results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"  {mark}  {result['case_id']:<26} {result['quality_score']:.3f}")
    print(f"\nwrote {json_path.name} and {md_path.name}")

    # Non-zero exit on failure so CI can gate on this.
    return 0 if summary["failed"] == 0 else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving on http://{args.host}:{args.port}   docs at /docs")
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
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

    brief = subparsers.add_parser("brief", help="Task 2 - build a TAM account brief")
    brief.add_argument("--account", required=True, help="account_id or company name, e.g. ACC-3847")
    brief.add_argument("--days", type=int, default=90, help="ticket window in days (default 90)")
    brief.add_argument("--all-history", action="store_true", help="ignore the window entirely")
    brief.add_argument("--json", action="store_true", help="emit the raw structured output")
    brief.set_defaults(func=cmd_brief)

    accounts = subparsers.add_parser("accounts", help="list the accounts available to brief")
    accounts.set_defaults(func=cmd_accounts)

    evaluate = subparsers.add_parser("eval", help="Task 3 - run the evaluation harness")
    evaluate.add_argument("--task", choices=["triage", "brief"], help="run only one task's cases")
    evaluate.add_argument("--no-judge", action="store_true", help="rule-based scoring only")
    evaluate.set_defaults(func=cmd_eval)

    serve = subparsers.add_parser("serve", help="run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="reload on code changes")
    serve.set_defaults(func=cmd_serve)

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
