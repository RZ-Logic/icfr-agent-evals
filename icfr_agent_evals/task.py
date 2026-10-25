"""The Phase 1 evaluation task: variance-commentary content eval.

Wires the seeded synthetic dataset, the roster setup check, the (pre-elicitation)
worker system prompt, a single generate() step, and the deterministic anchor
scorers. No LLM-graded sub-dimensions and no action/disposition scoring yet; those
are Phase 1 Week 2 and Phase 2 respectively.

Run configuration (pre-registration section 5):
  - the worker is chosen by --model and checked against the frozen roster by the
    setup solver; no model is bound on the Task, because a Task-level model
    silently overrides --model;
  - judge roles are bound with --model-role and are required;
  - reasoning_effort is pinned high; no sampling parameter is set anywhere;
  - epochs are an explicit Epochs(n, reducer=["mean"]), because the pinned
    inspect-ai records epochs_reducer as None when the default is used;
  - the dataset generator seed is the pinned task argument (recorded in
    task_args); Inspect's --seed is left unset;
  - Task version and metadata carry everything needed to reproduce the run
    config from the EvalLog header alone.

Reliability (content-pass^k, the review queue, FM counts) is computed after the
run by reliability.py from every (sample, epoch) score, never from the reduced
scores the "mean" reducer produces.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

# Absolute imports (not relative): the `inspect eval icfr_agent_evals/task.py@...`
# CLI loads THIS file by path, execing it without package context, so relative
# imports would fail there. The package is installed, so absolute imports resolve
# both under the CLI and under normal `import icfr_agent_evals.task`.
from icfr_agent_evals.casefile import DATASET_VERSION, EVAL_SEED, SCHEMA_VERSION, dataset_sha256, split_seed
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.roster import REASONING_EFFORT, retirement_metadata, roster_check, roster_metadata
from icfr_agent_evals.scorers import (
    completeness_anchor,
    correctness_anchor,
    fabrication_anchor,
    faithfulness_anchor,
    injection_compliance,
    reconciling_items_anchor,
)
from icfr_agent_evals.solver import SYSTEM_PROMPT, WORKER_PROMPT_SHA256, WORKER_PROMPT_VERSION

TASK_REDUCER = "mean"  # gated metrics never use pass_k or pass_at (section 6)


@task
def variance_commentary(
    split: str = "eval",
    seed: int = EVAL_SEED,
    epochs: int = 4,
    grader_step: int = 1,
) -> Task:
    """Variance-commentary content eval over the seeded synthetic set.

    Args:
        split: 'eval' (default), 'dev', 'ci' or 'calibration'. Prompt elicitation
            must use 'dev' only.
        seed: the pinned dataset generator seed; every split derives its own seed
            from it (casefile.split_seed).
        epochs: repeats per sample. Default 4 = the unified run design: the same
            four generations serve Sample-Consistency calibration and the k = 4
            trials for content-pass^4.
        grader_step: 1, or 2 once step 2 of the judge remedy order (section 7)
            moves the grader to its escalation pin.
    """
    dataset = build_dataset(split=split, seed=seed)
    return Task(
        dataset=dataset,
        setup=roster_check(grader_step=grader_step),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=[
            correctness_anchor(),
            fabrication_anchor(),
            reconciling_items_anchor(),
            faithfulness_anchor(),
            completeness_anchor(),
            injection_compliance(),
        ],
        epochs=Epochs(epochs, reducer=[TASK_REDUCER]),
        config=GenerateConfig(reasoning_effort=REASONING_EFFORT),
        version=DATASET_VERSION,
        metadata={
            "worker_prompt_version": WORKER_PROMPT_VERSION,
            "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "dataset_version": DATASET_VERSION,
            "dataset_sha256": dataset_sha256(list(dataset)),
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "split_seed": split_seed(split, seed),
            "roster": roster_metadata(grader_step),
            "retirement_floors": retirement_metadata(),
        },
    )
