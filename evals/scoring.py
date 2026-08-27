"""Scoring: rule-based criteria plus an LLM judge.

Each case produces a quality score in [0, 1] and a pass/fail verdict.

* **Rule score** - the weighted fraction of criteria that passed. Deterministic,
  free, and the part that catches structural regressions.
* **Judge score** - an LLM scoring the output against the case's rubric, for the
  qualities a predicate cannot express ("is this summary specific to this
  account or generic filler?").

Combined as ``0.7 * rule + 0.3 * judge`` when a rubric exists, rule-only
otherwise. Rules dominate deliberately: a judge is useful for nuance but is
itself a model, and a harness that leans on it inherits its variance.

A case fails if any **critical** criterion fails, or if the combined score falls
below :data:`PASS_THRESHOLD`. Critical criteria are the ones encoding a safety
property - an unverified quote, an invented ticket history, a cosmetic issue
escalated to P1.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from evals.cases import Criterion
from src.llm_client import LLMClient

PASS_THRESHOLD = 0.70
RULE_WEIGHT = 0.70
JUDGE_WEIGHT = 0.30

JUDGE_SYSTEM = """\
You are evaluating the output of an automated support-tooling pipeline. You are \
given a rubric question and the output. Score how well the output satisfies the \
rubric.

Be strict and concrete. Reward specificity, grounding in the supplied data, and \
appropriate caution. Penalise invented detail, generic filler, and confident \
claims the input does not support.

Scoring guide:
  1.0  fully satisfies the rubric
  0.75 satisfies it with a minor weakness
  0.5  partially satisfies it
  0.25 largely fails but shows some awareness
  0.0  fails the rubric entirely

Return one JSON object and nothing else:
{"score": <0-1>, "reasoning": "<one or two sentences>"}\
"""

JUDGE_USER = """\
RUBRIC
{rubric}

OUTPUT UNDER EVALUATION
{output}

Score the output against the rubric. Return the JSON object only.\
"""


@dataclass
class CriterionResult:
    name: str
    passed: bool
    weight: float
    critical: bool
    error: Optional[str] = None


@dataclass
class CaseResult:
    """The scored outcome of one test case."""

    case_id: str
    task: str
    description: str
    adversarial: bool
    passed: bool
    quality_score: float
    rule_score: float
    judge_score: Optional[float] = None
    judge_reasoning: str = ""
    criteria: list[CriterionResult] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    error: Optional[str] = None
    prompt_version: str = ""
    model: str = ""
    cached: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["criteria"] = [asdict(c) if not isinstance(c, dict) else c for c in self.criteria]
        return payload


def apply_criteria(result: Any, criteria: Sequence[Criterion]) -> tuple[float, list[CriterionResult]]:
    """Run every criterion. An exception counts as a failure, never a crash."""
    outcomes: list[CriterionResult] = []
    earned = 0.0
    total = 0.0

    for criterion in criteria:
        total += criterion.weight
        try:
            passed = bool(criterion.check(result))
            error = None
        except Exception as exc:  # a broken predicate must not abort the run
            passed = False
            error = f"{type(exc).__name__}: {exc}"

        if passed:
            earned += criterion.weight
        outcomes.append(
            CriterionResult(
                name=criterion.name,
                passed=passed,
                weight=criterion.weight,
                critical=criterion.critical,
                error=error,
            )
        )

    return (earned / total if total else 0.0), outcomes


def judge_output(
    rubric: str,
    output: str,
    *,
    llm: LLMClient,
    case_id: str,
) -> tuple[Optional[float], str]:
    """Score an output against a rubric with an LLM judge.

    A judge failure degrades to "no judge score" rather than failing the case -
    the rule score still stands, so the harness stays usable without one.
    """
    try:
        raw, _ = llm.complete(
            JUDGE_USER.format(rubric=rubric, output=output),
            system=JUDGE_SYSTEM,
            max_tokens=512,
            tags={"task": "judge", "case": case_id},
        )
    except Exception as exc:
        return None, f"judge unavailable: {type(exc).__name__}: {exc}"

    payload = _safe_json(raw)
    if "score" not in payload:
        return None, "judge returned no score"

    try:
        score = min(max(float(payload["score"]), 0.0), 1.0)
    except (TypeError, ValueError):
        return None, "judge score was not a number"

    return score, str(payload.get("reasoning") or "").strip()


def combine(rule_score: float, judge_score: Optional[float]) -> float:
    """Blend rule and judge scores."""
    if judge_score is None:
        return round(rule_score, 3)
    return round(RULE_WEIGHT * rule_score + JUDGE_WEIGHT * judge_score, 3)


def verdict(quality: float, outcomes: Sequence[CriterionResult]) -> tuple[bool, list[str]]:
    """Pass/fail. Any critical failure fails the case regardless of score."""
    critical_failures = [o.name for o in outcomes if o.critical and not o.passed]
    return (not critical_failures and quality >= PASS_THRESHOLD), critical_failures


def _safe_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
