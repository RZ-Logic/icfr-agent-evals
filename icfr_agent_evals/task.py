"""The Phase 1 evaluation task: variance-commentary content eval.

Wires the seeded synthetic dataset, the (pre-elicitation) worker system prompt, a
single generate() step, and the three deterministic anchor scorers. No LLM-graded
sub-dimensions and no action/disposition scoring yet — those are Phase 1 Week 2 and
Phase 2 respectively.

Reproducibility: for the Claude 4.6 generation and later, Anthropic model IDs are
DATELESS and immutable — `claude-sonnet-5` is the canonical pinned ID mapping to a
single fixed snapshot (weights behind an existing ID are never updated; updated
versions ship under new IDs). Date-suffixed IDs apply only to pre-4.6-generation
models (e.g. Haiku 4.5). Floating convenience aliases remain prohibited. Verified
against Anthropic's model docs, July 2026 (ADR-0003 records the worker re-pin from
Sonnet 4.6 to Sonnet 5 before any production run; the predecessor moves into the
cross-model baseline set so the generational comparison is a reported result).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

# Absolute imports (not relative): the `inspect eval icfr_agent_evals/task.py@...`
# CLI loads THIS file by path, execing it without package context, so relative
# imports would fail there. The package is installed, so absolute imports resolve
# both under the CLI and under normal `import icfr_agent_evals.task`.
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.scorers import (
    completeness_anchor,
    correctness_anchor,
    faithfulness_anchor,
)
from icfr_agent_evals.solver import SYSTEM_PROMPT, WORKER_PROMPT_VERSION

# Canonical dateless ID (Claude 4.6+ generation naming). ADR-0003.
DEFAULT_WORKER = "anthropic/claude-sonnet-5"

# Cross-model baselines (Week 2). Pinned, not latest-aliased. Sonnet 4.6 is
# retained deliberately: worker-vs-predecessor is the generational comparison.
BASELINE_MODELS = [
    "anthropic/claude-haiku-4-5-20251001",  # pre-4.6 generation: date-suffixed ID
    "anthropic/claude-sonnet-4-6",          # predecessor of the worker
    "anthropic/claude-opus-4-7",
]


@task
def variance_commentary(
    split: str = "eval",
    seed: int = 20260607,
    epochs: int = 4,
) -> Task:
    """Variance-commentary content eval over the seeded synthetic set.

    Args:
        split: 'eval' (default) or 'dev'. Prompt elicitation must use 'dev' only.
        seed: dataset seed; pinned for reproducibility.
        epochs: repeats per sample. Default 4 = the unified run design: the same
            four generations serve Sample-Consistency calibration (>= 3 floor) AND
            the k = 4 trials for pass^k. Epochs and pass^k trials are one set of
            generations, not multiplicative separate runs. The deterministic
            anchors themselves are epoch-invariant on a fixed completion.
    """
    return Task(
        dataset=build_dataset(split=split, seed=seed),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=[
            correctness_anchor(),
            faithfulness_anchor(),
            completeness_anchor(),
        ],
        epochs=epochs,
        metadata={"worker_prompt_version": WORKER_PROMPT_VERSION},
    )
