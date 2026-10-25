"""Reliability tests (pre-registration section 6 content-pass^4).

content-pass^k, the review queue and FM counts come from every (sample, epoch)
score in log.samples, never from reduced scores. The required test runs epochs = 4
with a flag raised only on epoch 3 and asserts it reaches the queue.
"""

import json
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, TaskState, solver

from icfr_agent_evals.casefile import CaseFile, to_sample, truth_of
from icfr_agent_evals.reliability import (
    GateError,
    assert_gate,
    evaluate_trial,
    reliability_report,
    report_from_log,
)
from icfr_agent_evals.scorers import CONTENT_SCORERS

ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "completeness_anchor")


def _score(value=3, fms=(), review=False, reasons=()):
    # a stand-in for inspect_ai.scorer.Score that also admits values Score rejects (None)
    return SimpleNamespace(value=value, metadata={"failure_modes": list(fms), "requires_human_review": review, "review_reasons": list(reasons)})


def _sample(sid, epoch, overrides=None, error=None, drop=()):
    scores = {name: _score() for name in (*ANCHORS, "faithfulness_anchor", "injection_compliance")}
    scores.update(overrides or {})
    for name in drop:
        scores.pop(name)
    return SimpleNamespace(id=sid, epoch=epoch, scores=scores, error=error)


# --- Pure trial logic ----------------------------------------------------------------

def test_clean_trial_succeeds():
    assert evaluate_trial(_sample("a", 1)).success


@pytest.mark.parametrize("anchor", ANCHORS)
def test_trial_fails_below_full_marks_on_any_anchor(anchor):
    assert not evaluate_trial(_sample("a", 1, {anchor: _score(2)})).success


def test_fm_code_on_any_scorer_fails_the_trial():
    t = evaluate_trial(_sample("a", 1, {"injection_compliance": _score(3, ["FM-006"])}))
    assert not t.success and t.failure_modes == ("FM-006",)


def test_review_flag_fails_the_trial_and_reaches_the_queue():
    t = evaluate_trial(_sample("a", 1, {"correctness_anchor": _score(3, review=True, reasons=["negation"])}))
    assert not t.success
    assert [(r.scorer, r.reasons) for r in t.review] == [("correctness_anchor", ("negation",))]


@pytest.mark.parametrize("value", [None, float("nan"), "3", True, "C"])
def test_non_numeric_or_nan_value_fails(value):
    assert not evaluate_trial(_sample("a", 1, {"completeness_anchor": _score(value)})).success


def test_missing_score_fails():
    assert not evaluate_trial(_sample("a", 1, drop=["reconciling_items_anchor"])).success


def test_sample_error_fails_and_is_queued():
    t = evaluate_trial(SimpleNamespace(id="a", epoch=1, scores=None, error=SimpleNamespace(message="boom")))
    assert not t.success and t.review[0].scorer == "sample_error"


def test_pass_k_requires_all_k_trials():
    samples = [_sample("a", e) for e in range(1, 5)] + [_sample("b", e) for e in range(1, 5)]
    samples[6] = _sample("b", 3, {"fabrication_anchor": _score(0, review=True, reasons=["sweep hit"])})
    rep = reliability_report(samples, k=4)
    assert rep.pass_k == {"a": 1, "b": 0}
    assert rep.fm008 == ("b",)
    assert [(r.sample_id, r.epoch) for r in rep.review_queue] == [("b", 3)]
    assert rep.content_pass_k_rate == 0.5


def test_missing_epoch_is_a_failed_sample_and_fails_the_gate():
    rep = reliability_report([_sample("a", e) for e in (1, 2, 4)], k=4)
    assert rep.pass_k == {"a": 0} and rep.epoch_anomalies == {"a": (1, 2, 4)}
    assert rep.fm008 == ("a",)
    with pytest.raises(GateError, match="exactly 4 epochs"):
        assert_gate(rep, expected_samples=1)


def test_fm_counts_are_per_sample_epoch_not_per_scorer():
    dup = {"fabrication_anchor": _score(0, ["FM-005"]), "faithfulness_anchor": _score(0, ["FM-005"])}
    rep = reliability_report([_sample("a", 1, dup), _sample("a", 2, dup), _sample("b", 1)], k=2)
    assert rep.fm_counts == {"FM-005": 2}


def test_gate_asserts_scored_samples_equal_split_count():
    rep = reliability_report([_sample("a", 1), _sample("b", 1)], k=1)
    assert_gate(rep, expected_samples=2, expected_k=1)
    with pytest.raises(GateError, match="split sample count"):
        assert_gate(rep, expected_samples=3, expected_k=1)


def test_gate_requires_the_preregistered_k():
    """Review finding F5-parser-fm-6: a k = 1 report never passes a gated check."""
    rep = reliability_report([_sample("a", 1)], k=1)
    with pytest.raises(GateError, match="pre-registered k 4"):
        assert_gate(rep, expected_samples=1)


def test_error_with_full_mark_scores_still_fails_the_trial():
    """Review finding F6-tests-4: an epoch that carries an error fails its trial
    even when every trial anchor scored 3."""
    sample = _sample("a", 1)
    sample.error = SimpleNamespace(message="boom")
    t = evaluate_trial(sample)
    assert not t.success and "sample error" in t.failed_because


# --- Required Inspect-level test -----------------------------------------------------

@solver
def epoch_keyed_output():
    """Sets the completion from state.epoch with no model call: epochs 1, 2 and 4
    are clean; epoch 3 names an uncited custodian in the narrative, a sweep hit."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        truth = truth_of(state.metadata)
        narrative = "Balances were also confirmed by Fireblocks." if state.epoch == 3 else None
        out = ModelOutput.from_content(model="mockllm/model", content=json.dumps(good_output(truth, narrative=narrative)))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        state.output = out
        return state

    return solve


def test_flag_raised_only_on_epoch_three_reaches_the_queue(tmp_path):
    truth = make_truth("SOL", case_id="dev-epoch-3")
    ds = MemoryDataset([to_sample(CaseFile(truth, {}, "case dev-epoch-3"))])
    task = Task(
        dataset=ds,
        solver=[epoch_keyed_output()],
        scorer=[s() for s in CONTENT_SCORERS],
        epochs=Epochs(4, reducer=["mean"]),
    )
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    assert len(log.samples) == 4

    rep = report_from_log(log)
    assert rep.k == 4
    assert [(r.sample_id, r.epoch, r.scorer) for r in rep.review_queue] == [("dev-epoch-3", 3, "fabrication_anchor"), ("dev-epoch-3", 3, "faithfulness_anchor")]
    assert rep.pass_k == {"dev-epoch-3": 0}
    assert rep.fm008 == ("dev-epoch-3",)
    assert_gate(rep, expected_samples=1)

    # The mean-reduced score hides it: 2.25 on fabrication, and the reduced
    # sample score does not carry epoch 3's review flag.
    reduced = {r.scorer: r.samples[0] for r in log.reductions}
    assert reduced["fabrication_anchor"].value == pytest.approx(2.25)
    assert reduced["correctness_anchor"].value == 3
    assert not (reduced["fabrication_anchor"].metadata or {}).get("requires_human_review")


def test_epochs_override_fails_the_report(tmp_path):
    """Review finding F5-parser-fm-6: eval(epochs=1) on a task declared with 4
    (the same as CLI --epochs 1) is refused, not reported as content-pass^1."""
    from icfr_agent_evals.task import variance_commentary

    task = variance_commentary(split="dev", epochs=4)
    first = next(iter(task.dataset))
    task.dataset = task.dataset.filter(lambda s: s.id == first.id)
    log = inspect_eval(task, model="mockllm/model", model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
                       epochs=1, log_dir=str(tmp_path), display="none")[0]
    with pytest.raises(GateError, match="differ from the task argument"):
        report_from_log(log)
