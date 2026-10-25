"""Inspect scorer wrappers around the pure deterministic logic in scoring.py.

These are deliberately *thin*: all decision logic lives in scoring.py (which has
no Inspect or LLM dependency and is unit-tested directly). A scorer here only
adapts Inspect's (TaskState, Target) -> Score contract onto those pure functions,
so the consequential checks can be tested without spinning up the eval graph.

Each scorer reads the per-sample ground truth from state.metadata["truth"] and the
agent's structured output from state.output.completion. An output that cannot be
parsed as a JSON object is itself a failure: it scores 0 on every anchor, is
tagged `format_break`, and is routed to `requires_human_review` — never silently
skipped. Faithfulness additionally routes narrative-sweep suspects to
`requires_human_review` with a conservative provisional 0 (see scoring.py for the
error-direction analysis).

Value semantics: each anchor returns an ordinal 0..3, so the registered metrics are
mean() and stderr() (not accuracy(), which assumes a binary correct/incorrect).
"""

from __future__ import annotations

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from .scoring import (
    check_completeness,
    check_correctness,
    check_faithfulness,
    parse_worker_output,
)

FORMAT_BREAK_FM = "FM-000"  # unparseable structured output


def _truth(state: TaskState) -> dict:
    truth = (state.metadata or {}).get("truth")
    if not isinstance(truth, dict):
        raise ValueError(
            "sample metadata is missing 'truth'; dataset and scorer are out of sync"
        )
    return truth


def _format_break_score() -> Score:
    return Score(
        value=0,
        answer="<unparseable>",
        explanation="Worker output could not be parsed as a JSON object (format break); routed to human review.",
        metadata={
            "failure_modes": [FORMAT_BREAK_FM],
            "format_break": True,
            "requires_human_review": True,
        },
    )


@scorer(metrics=[mean(), stderr()])
def correctness_anchor():
    """Deterministic accuracy of the analytical facts: magnitude, direction, band."""

    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_worker_output(state.output.completion)
        if parsed is None:
            return _format_break_score()
        points, fms = check_correctness(parsed, _truth(state))
        return Score(
            value=points,
            answer=str(parsed.get("variance_amount_usd")),
            explanation=f"correctness {points}/3; failure_modes={fms or 'none'}",
            metadata={"failure_modes": fms, "dimension": "correctness"},
        )

    return score


@scorer(metrics=[mean(), stderr()])
def faithfulness_anchor():
    """Fabrication cliff: any cited entity absent from the input fixture -> 0.
    Narrative-sweep suspects -> provisional 0 + requires_human_review."""

    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_worker_output(state.output.completion)
        if parsed is None:
            return _format_break_score()
        points, fms, review = check_faithfulness(parsed, _truth(state))
        if review:
            explanation = (
                f"faithfulness provisional {points}/3; narrative tokens not in the "
                f"structured citations: {review}; routed to human review"
            )
        else:
            explanation = f"faithfulness {points}/3; failure_modes={fms or 'none'}"
        return Score(
            value=points,
            answer=str(parsed.get("entities_cited")),
            explanation=explanation,
            metadata={
                "failure_modes": fms,
                "dimension": "faithfulness",
                "requires_human_review": bool(review),
                "review_tokens": review,
            },
        )

    return score


@scorer(metrics=[mean(), stderr()])
def completeness_anchor():
    """Required-element presence: asset, period, magnitude each disclosed."""

    async def score(state: TaskState, target: Target) -> Score:
        parsed = parse_worker_output(state.output.completion)
        if parsed is None:
            return _format_break_score()
        points, fms = check_completeness(parsed, _truth(state))
        return Score(
            value=points,
            answer=f"asset={parsed.get('asset')} period={parsed.get('period')}",
            explanation=f"completeness {points}/3; failure_modes={fms or 'none'}",
            metadata={"failure_modes": fms, "dimension": "completeness"},
        )

    return score
