"""Template baseline for the Phase 1 content eval (pre-registration section 8,
primary outcomes; section 13, pilot and difficulty check).

No model. template_output computes a schema-valid worker output by code from
casefile.visible_structured(truth) alone: the two balances, the wallet scope and
the rows of the open-items schedule. It never reads hidden truth (item roles,
channels, memo-only items, eligibility verdicts, the explained split), so a case
whose explanation is stated only in memo text is one it cannot get right. Each
model's deterministic anchor scores are read as the margin over this row.

  - variance = on-chain minus ledger; direction from its sign; band from
    policy.materiality_band;
  - reconciling_items = the schedule rows casefile.item_eligibility accepts on
    the visible fields, each cited once with its amount and type;
  - entities_cited = exactly those items' references;
  - narrative = a fixed template stating only those values: the variance amount,
    direction and band, each cited reference with its amount, and the explained
    and unexplained portions computed from the cited rows.

The template wording is fitted to the narrative extractor (narrative.py) and the
sweep (scoring.py), never the reverse: on a standard case without a memo-only link
it must fire no failure mode and route nothing to review. The offline difficulty
check in tests/test_template_baseline.py holds it to that.

Run it with:
    inspect eval icfr_agent_evals/baseline.py@variance_commentary_template_baseline

The task binds Inspect's "none" provider, which raises if anything asks it to
generate, so a stray model call errors the sample instead of sampling a model.
"""

from __future__ import annotations

import json

from inspect_ai import Epochs, Task, task
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, TaskState, solver

# Absolute imports, as in task.py: the inspect CLI execs this file by path.
from icfr_agent_evals.casefile import (
    EVAL_SEED,
    ReconcilingItem,
    VisibleStructured,
    item_effect_usd,
    item_eligibility,
    truth_of,
    visible_structured,
)
from icfr_agent_evals.policy import materiality_band
from icfr_agent_evals.roster import RosterError
from icfr_agent_evals.task import TASK_REDUCER, variance_commentary

BASELINE_VERSION = "template-v1"  # a template wording change is a code change
BASELINE_MODEL = "none/none"


def _usd(value: float, signed: bool = False) -> str:
    sign = ("+" if value > 0 else "-" if value < 0 else "") if signed else ""
    return f"{sign}${abs(value):,.2f}"


def eligible_rows(view: VisibleStructured) -> list[ReconcilingItem]:
    """Schedule rows the section 6 eligibility rule accepts on visible fields, in
    schedule order, each reference once."""
    rows: list[ReconcilingItem] = []
    seen: set[str] = set()
    for row in view.open_items:
        elig = item_eligibility(row, asset=view.asset, scope_wallets=view.scope_wallets, cutoff_date=view.cutoff_date)
        if elig.eligible and row.reference not in seen:
            seen.add(row.reference)
            rows.append(row)
    return rows


def template_narrative(
    variance: float, band: str, cited: list[ReconcilingItem], explained: float, unexplained: float
) -> str:
    """The fixed narrative. Amounts are magnitudes, except when the cited rows move
    the variance the other way from its direction (or explain more than all of
    it): then the explained and unexplained portions carry explicit signs on the
    on-chain minus ledger convention, which the extractor checks (narrative D8)."""
    direction = "surplus" if variance > 0 else "shortfall"
    sentences = [f"The variance is a {direction} of {_usd(variance)}, in materiality band {band}."]
    if not cited:
        sentences.append(f"No open item explains the {direction}, so the entire {_usd(variance)} {direction} is unexplained.")
        return " ".join(sentences)
    for row in cited:
        sentences.append(f"Open item {row.reference} is {_usd(row.amount_usd)}.")
    if explained * variance > 0 and unexplained * variance >= 0:
        sentences.append(f"The cited items explain {_usd(explained)} of the {direction}.")
        sentences.append(f"The remaining {_usd(unexplained)} is unexplained.")
    else:
        sentences.append(f"The cited items explain {_usd(explained, signed=True)} of the variance.")
        sentences.append(f"The remaining {_usd(unexplained, signed=True)} is unexplained.")
    return " ".join(sentences)


def template_output(view: VisibleStructured) -> dict:
    """A schema-valid worker output computed by code from the visible structured
    fields alone (pre-registration section 8 template baseline)."""
    variance = round(view.onchain_reserves_usd - view.ledger_liabilities_usd, 2)
    band = materiality_band(variance)
    cited = eligible_rows(view)
    explained = round(sum(item_effect_usd(row) for row in cited), 2)
    unexplained = round(variance - explained, 2)
    return {
        "asset": view.asset,
        "period": view.period,
        "variance_amount_usd": abs(variance),
        "variance_direction": "surplus" if variance > 0 else "shortfall",
        "materiality_band": band,
        "entities_cited": [row.reference for row in cited],
        "reconciling_items": [
            {"reference": row.reference, "amount_usd": row.amount_usd, "type": row.type} for row in cited
        ],
        "narrative": template_narrative(variance, band, cited, explained, unexplained),
    }


# --- Inspect wiring --------------------------------------------------------------------

@solver
def template_baseline():
    """Set state.output to the template output for the sample. No model is called:
    the answer key is rehydrated only to take its visible structured projection."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        view = visible_structured(truth_of(state.metadata))
        output = ModelOutput.from_content(model=BASELINE_MODEL, content=json.dumps(template_output(view)))
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        state.output = output
        state.messages.append(output.message)
        return state

    return solve


def baseline_check():
    """Setup solver in place of roster.roster_check: the evaluated model must be the
    none provider and no model role may be bound. No scorer in this task uses a
    judge role, so a bound role would be configuration the run never exercises."""
    from inspect_ai.model import get_model, model_roles

    @solver(name="template_baseline_check")
    def _baseline_check():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            evaluated = str(get_model())
            if evaluated != BASELINE_MODEL:
                raise RosterError(f"the template baseline runs on {BASELINE_MODEL}, not {evaluated!r}")
            bound = sorted(model_roles())
            if bound:
                raise RosterError(f"the template baseline takes no model roles; bound: {bound}")
            return state

        return solve

    return _baseline_check()


@task
def variance_commentary_template_baseline(split: str = "eval", seed: int = EVAL_SEED, epochs: int = 4) -> Task:
    """The template baseline over a split, scored exactly as the main task scores a
    model: the same dataset, scorers, Epochs and metadata, taken from
    task.variance_commentary so the two can never drift apart.

    The model is bound on the Task, the reverse of the main task's rule: here a
    Task-level model overriding --model is the safe direction, because whatever
    --model says, nothing can be sampled. Epochs stay at the main task's value so
    the reference row has the same trial structure; every epoch is identical.
    """
    main = variance_commentary(split=split, seed=seed, epochs=epochs)
    return Task(
        dataset=main.dataset,
        setup=baseline_check(),
        solver=template_baseline(),
        scorer=main.scorer,
        epochs=Epochs(epochs, reducer=[TASK_REDUCER]),
        model=BASELINE_MODEL,
        version=main.version,
        metadata={
            **main.metadata,
            "baseline": {
                "kind": "template",
                "version": BASELINE_VERSION,
                "model": BASELINE_MODEL,
                "worker_prompt_used": False,
            },
        },
    )

