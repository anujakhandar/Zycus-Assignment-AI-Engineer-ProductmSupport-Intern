"""Eval runner: executes every case and writes the report.

    python main.py eval                 # all cases
    python main.py eval --task triage   # one task
    python main.py eval --no-judge      # rule-based only, no judge calls

Writes ``eval_report.json`` (full detail) and ``eval_report.md`` (the summary
table). Both are committed so the results can be read without running anything.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evals.cases import BRIEF_CASES, TRIAGE_CASES, DATASET, BriefCase, TriageCase
from evals.scoring import (
    CaseResult,
    PASS_THRESHOLD,
    apply_criteria,
    combine,
    judge_output,
    verdict,
)
from src.account_brief import build_account_brief
from src.data_loader import Dataset
from src.llm_client import LLMClient
from src.schemas import TicketInput
from src.triage import triage_ticket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = PROJECT_ROOT / "eval_report.json"
MD_REPORT = PROJECT_ROOT / "eval_report.md"


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


def run_triage_case(case: TriageCase, *, llm: LLMClient, use_judge: bool) -> CaseResult:
    try:
        result = triage_ticket(
            TicketInput(subject=case.subject, body=case.body),
            llm=llm,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            task="triage",
            description=case.description,
            adversarial=case.adversarial,
            passed=False,
            quality_score=0.0,
            rule_score=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    rule_score, outcomes = apply_criteria(result, case.criteria)

    judge_score, judge_reasoning = None, ""
    if use_judge and case.judge_rubric:
        judge_score, judge_reasoning = judge_output(
            case.judge_rubric, _render_triage(result), llm=llm, case_id=case.case_id
        )

    quality = combine(rule_score, judge_score)
    passed, critical = verdict(quality, outcomes)

    return CaseResult(
        case_id=case.case_id,
        task="triage",
        description=case.description,
        adversarial=case.adversarial,
        passed=passed,
        quality_score=quality,
        rule_score=round(rule_score, 3),
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
        criteria=outcomes,
        failed_criteria=[o.name for o in outcomes if not o.passed],
        critical_failures=critical,
        prompt_version=result.prompt_version,
        model=result.model,
        cached=result.cached,
    )


def run_brief_case(case: BriefCase, *, llm: LLMClient, use_judge: bool) -> CaseResult:
    dataset = DATASET
    if case.withhold_tickets:
        # Adversarial: same account, ticket history removed, to check the
        # pipeline reports the gap rather than inventing a history.
        account = next(a for a in DATASET.accounts if a.account_id == case.account)
        dataset = Dataset(
            tickets=[t for t in DATASET.tickets if t.company != account.company],
            accounts=DATASET.accounts,
            kb_chunks=DATASET.kb_chunks,
        )

    try:
        brief = build_account_brief(case.account, dataset=dataset, llm=llm, days=case.days)
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            task="account_brief",
            description=case.description,
            adversarial=case.adversarial,
            passed=False,
            quality_score=0.0,
            rule_score=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    rule_score, outcomes = apply_criteria(brief, case.criteria)

    # Determinism case: build it a second time and compare rendered output.
    if case.rerun_for_determinism:
        second = build_account_brief(case.account, dataset=dataset, llm=llm, days=case.days)
        identical = brief.to_markdown() == second.to_markdown()
        for outcome in outcomes:
            if outcome.name == "second run is identical":
                outcome.passed = identical
        rule_score = 1.0 if identical else 0.0

    judge_score, judge_reasoning = None, ""
    if use_judge and case.judge_rubric:
        judge_score, judge_reasoning = judge_output(
            case.judge_rubric, brief.to_markdown(), llm=llm, case_id=case.case_id
        )

    quality = combine(rule_score, judge_score)
    passed, critical = verdict(quality, outcomes)

    return CaseResult(
        case_id=case.case_id,
        task="account_brief",
        description=case.description,
        adversarial=case.adversarial,
        passed=passed,
        quality_score=quality,
        rule_score=round(rule_score, 3),
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
        criteria=outcomes,
        failed_criteria=[o.name for o in outcomes if not o.passed],
        critical_failures=critical,
        prompt_version=brief.prompt_version,
        model=brief.model,
        cached=brief.cached,
    )


def _render_triage(result) -> str:
    """Flatten a triage result into the text the judge scores."""
    references = "\n".join(f"- {r.breadcrumb} ({r.source})" for r in result.kb_references) or "(none)"
    return (
        f"Product: {result.product.value} - {result.product.reasoning}\n"
        f"Area: {result.product_area.value} - {result.product_area.reasoning}\n"
        f"Category: {result.category.value} - {result.category.reasoning}\n"
        f"Urgency: {result.urgency.value} - {result.urgency.reasoning}\n"
        f"Responder team: {result.responder_team} - {result.routing_reasoning}\n"
        f"Known issue: {result.known_issue or '(none)'}\n"
        f"Cited sections:\n{references}\n\n"
        f"Draft response:\n{result.draft_response}\n\n"
        f"Needs human review: {result.needs_human_review} {result.review_reasons}"
    )


# ---------------------------------------------------------------------------
# Orchestration and reporting
# ---------------------------------------------------------------------------


def run_all(
    *,
    task: Optional[str] = None,
    use_judge: bool = True,
    llm: Optional[LLMClient] = None,
) -> dict:
    """Run the selected cases and return the report payload."""
    llm = llm or LLMClient()
    started = time.time()
    results: list[CaseResult] = []

    if task in (None, "triage"):
        for case in TRIAGE_CASES:
            print(f"  running {case.case_id} ...", flush=True)
            results.append(run_triage_case(case, llm=llm, use_judge=use_judge))

    if task in (None, "brief", "account_brief"):
        for case in BRIEF_CASES:
            print(f"  running {case.case_id} ...", flush=True)
            results.append(run_brief_case(case, llm=llm, use_judge=use_judge))

    return _summarise(results, elapsed=time.time() - started, llm=llm, use_judge=use_judge)


def _summarise(results: list[CaseResult], *, elapsed: float, llm: LLMClient, use_judge: bool) -> dict:
    passed = sum(1 for r in results if r.passed)
    scores = [r.quality_score for r in results]
    adversarial = [r for r in results if r.adversarial]

    by_task: dict[str, dict] = {}
    for result in results:
        bucket = by_task.setdefault(result.task, {"total": 0, "passed": 0, "scores": []})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
        bucket["scores"].append(result.quality_score)
    for bucket in by_task.values():
        bucket["mean_quality"] = round(sum(bucket["scores"]) / len(bucket["scores"]), 3)
        bucket.pop("scores")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass_threshold": PASS_THRESHOLD,
        "judge_enabled": use_judge,
        "model": llm.model,
        "summary": {
            "total_cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 3) if results else 0.0,
            "mean_quality": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "adversarial_cases": len(adversarial),
            "adversarial_passed": sum(1 for r in adversarial if r.passed),
            "elapsed_seconds": round(elapsed, 1),
            "live_api_calls": llm.live_calls,
            "cache_hits": llm.cache.hits,
        },
        "by_task": by_task,
        "results": [r.to_dict() for r in results],
    }


def write_reports(report: dict) -> tuple[Path, Path]:
    """Write eval_report.json and eval_report.md."""
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return JSON_REPORT, MD_REPORT


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Evaluation report",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Model: `{report['model']}`  ",
        f"Pass threshold: `{report['pass_threshold']}`  ",
        f"LLM judge: `{'enabled' if report['judge_enabled'] else 'disabled'}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test cases | {summary['total_cases']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| Pass rate | {summary['pass_rate']:.0%} |",
        f"| Mean quality score | {summary['mean_quality']:.3f} |",
        f"| Adversarial cases | {summary['adversarial_passed']}/{summary['adversarial_cases']} passed |",
        f"| Live API calls | {summary['live_api_calls']} |",
        f"| Cache hits | {summary['cache_hits']} |",
        f"| Runtime | {summary['elapsed_seconds']}s |",
        "",
        "## By task",
        "",
        "| Task | Cases | Passed | Mean quality |",
        "|---|---|---|---|",
    ]
    for task, bucket in sorted(report["by_task"].items()):
        lines.append(f"| {task} | {bucket['total']} | {bucket['passed']} | {bucket['mean_quality']:.3f} |")

    lines += [
        "",
        "## Cases",
        "",
        "| Case | Task | Result | Quality | Rule | Judge | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        if result["adversarial"]:
            status += " (adv)"
        judge = f"{result['judge_score']:.2f}" if result["judge_score"] is not None else "-"
        notes = result["error"] or ", ".join(result["failed_criteria"]) or "all criteria met"
        lines.append(
            f"| `{result['case_id']}` | {result['task']} | {status} | "
            f"{result['quality_score']:.3f} | {result['rule_score']:.3f} | {judge} | {notes} |"
        )

    failures = [r for r in report["results"] if not r["passed"]]
    if failures:
        lines += ["", "## Failures in detail", ""]
        for result in failures:
            lines.append(f"### `{result['case_id']}` - {result['description']}")
            lines.append("")
            if result["error"]:
                lines.append(f"- Error: `{result['error']}`")
            for name in result["critical_failures"]:
                lines.append(f"- **Critical criterion failed:** {name}")
            for name in result["failed_criteria"]:
                if name not in result["critical_failures"]:
                    lines.append(f"- Criterion failed: {name}")
            if result["judge_reasoning"]:
                lines.append(f"- Judge: {result['judge_reasoning']}")
            lines.append("")

    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python main.py eval",
        "```",
        "",
        "Cached model responses are committed under `fixtures/llm_cache/`, so this "
        "runs without an API key and returns the same results.",
        "",
    ]
    return "\n".join(lines)
