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
  - epochs are declared as Epochs(n, reducer=["mean"]): with the default, the
    pinned inspect-ai writes no reducer into the log header;
  - the generator seed is a task argument, so the log header carries it in
    task_args; the --seed option is not used;
  - the evaluation split is locked until the pre-registration tag (rule C3):
    split 'eval' raises unless allow_evaluation_split is true, so an omitted or
    mistyped -T split never reaches an evaluation case (Inspect drops an unknown
    -T key silently, and split defaults to 'eval'); the argument lands in
    task_args;
  - subset is a task argument too ("pilot": the 20 flagged dev cases of the
    section 13 pilot row), so task_args records it; dataset_sha256 hashes the
    samples the task selects (the split, or the subset), and split_dataset_sha256
    the whole split they were selected from, so a pilot log never carries the
    full split's hash. Both are computed when the task is built, before Inspect
    applies --limit or --sample-id, so on such a run dataset_sha256 also covers
    samples that never ran; the section 10 gates refuse those runs, and any
    other reader that trusts the hash must check the samples that ran;
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
from icfr_agent_evals.dataset import build_dataset, check_subset, dataset_name, select_subset
from icfr_agent_evals.roster import REASONING_EFFORT, retirement_metadata, roster_check, roster_metadata
from icfr_agent_evals.scorers import (
    completeness_anchor,
    correctness_anchor,
    fabrication_anchor,
    faithfulness_anchor,
    injection_compliance,
    reconciling_items_anchor,
)
from icfr_agent_evals.scoring import check_wallet_labels_in_table
from icfr_agent_evals.solver import SYSTEM_PROMPT, WORKER_PROMPT_SHA256, WORKER_PROMPT_VERSION

TASK_REDUCER = "mean"  # gated metrics never use pass_k or pass_at (section 6)
EVALUATION_SPLIT = "eval"


def check_evaluation_split(split: str, allow_evaluation_split: bool = False) -> None:
    """Refuse the evaluation split unless the unlock is exactly True (rule C3).

    Called before any split is built, so a refused call never generates, hashes or
    scores an evaluation case. Only the literal True unlocks: Inspect parses -T
    values as YAML, so true, yes and on unlock, while 1 or a quoted string stays
    locked."""
    if split == EVALUATION_SPLIT and allow_evaluation_split is not True:
        raise ValueError(
            "refusing split 'eval' before the pre-registration tag (rule C3); pass "
            "allow_evaluation_split=True (-T allow_evaluation_split=true) only for the "
            "tagged run, and -T split=dev for pilot, smoke and elicitation runs"
        )


@task
def variance_commentary(
    split: str = "eval",
    seed: int = EVAL_SEED,
    epochs: int = 4,
    grader_step: int = 1,
    subset: str | None = None,
    allow_evaluation_split: bool = False,
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
        grader_step: 1, 2 or 3: the step of the grader ladder (section 5). 2 is
            GPT-6.1 Terra and 3 is GPT-6.1 Sol, reached only through the judge
            remedy order (section 7); steps 2 and 3 run on the OpenAI Batch API.
        subset: None (the whole split) or 'pilot' (the 20 dev cases flagged for
            the section 13 pilot; split must be 'dev'). Checked before the split
            is built.
        allow_evaluation_split: False (default) refuses split 'eval' before the
            pre-registration tag; True is for the tagged run only. Checked after
            the subset check and before the split is built.
    """
    check_subset(split, subset)
    check_evaluation_split(split, allow_evaluation_split)
    full = build_dataset(split=split, seed=seed)
    # D-R9 m7: every wallet label the scorer derives is a row of its case's wallet table
    check_wallet_labels_in_table(full)
    dataset = full
    if subset is not None:
        chosen = {s.id for s in select_subset(full, split, subset)}
        dataset = full.filter(lambda s: s.id in chosen, name=dataset_name(split, subset))
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
            "split_dataset_sha256": dataset_sha256(list(full)),
            "subset": subset,
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "split_seed": split_seed(split, seed),
            "roster": roster_metadata(grader_step),
            "retirement_floors": retirement_metadata(),
        },
    )
