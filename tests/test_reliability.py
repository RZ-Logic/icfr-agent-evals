"""Reliability tests (pre-registration section 6 content-pass^4).

content-pass^k, the review queue and FM counts come from every (sample, epoch)
score in log.samples, never from reduced scores. The required test runs epochs = 4
with a flag raised only on epoch 3 and asserts it reaches the queue.
"""

import json
from types import SimpleNamespace

import pytest
from casefactory import ALL_ASSETS, good_output, make_truth
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, TaskState, solver

from icfr_agent_evals.casefile import CaseFile, to_sample, truth_of
from icfr_agent_evals.regression import CODE_ROUTE, SWEEP_ROUTE, route_key
from icfr_agent_evals.reliability import (
    QUEUE_CLASSES,
    GateError,
    assert_gate,
    evaluate_trial,
    format_queue_volume,
    is_template_baseline_log,
    queue_items,
    queue_volume,
    reliability_report,
    report_from_log,
    require_run_config,
)
from icfr_agent_evals.scoring import FABRICATION_REVIEW
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

    with pytest.raises(GateError, match="run config"):  # contract C2: no ModelEvent, no pinned effort
        report_from_log(log)
    rep = report_from_log(log, run_config=False)  # a hand-built Task that makes no model call
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


# --- Queue volume: distinct items by class, never scorer rows (F2-07) ------------------

SWEEP_TEXT = "Balances were also confirmed by Fireblocks."


@solver
def queue_mix_output():
    """Epoch 1 clean, epoch 2 a format break, epoch 3 a narrative sweep hit,
    epoch 4 a sample error; keyed by state.epoch, no model call."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.epoch == 4:
            raise RuntimeError("queue-mix sample error")
        truth = truth_of(state.metadata)
        if state.epoch == 2:
            content = "not one JSON object"
        else:
            content = json.dumps(good_output(truth, narrative=SWEEP_TEXT if state.epoch == 3 else None))
        out = ModelOutput.from_content(model="mockllm/model", content=content)
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        state.output = out
        return state

    return solve


@pytest.fixture(scope="module")
def queue_mix(tmp_path_factory):
    truths = [make_truth(asset, case_id=f"dev-queue-{asset.lower()}") for asset in ALL_ASSETS]
    ds = MemoryDataset([to_sample(CaseFile(t, {}, f"case {t.case_id}")) for t in truths])
    task = Task(dataset=ds, solver=[queue_mix_output()], scorer=[s() for s in CONTENT_SCORERS],
                epochs=Epochs(4, reducer=["mean"]))
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path_factory.mktemp("queue-mix")),
                       display="none", fail_on_error=False)[0]
    assert log.status == "success", log.error
    assert len(log.samples) == 4 * len(ALL_ASSETS)
    return report_from_log(log, run_config=False)  # a hand-built Task that makes no model call


def test_queue_volume_counts_distinct_items_by_class_on_a_mock_log(queue_mix):
    vol = queue_volume(queue_mix)
    n = len(ALL_ASSETS)
    assert vol.items == {"format_break": n, "sample_error": n, "narrative_sweep": n, "code_review": 0, "fm_listed": 0}
    assert vol.sample_epochs == vol.items
    assert vol.total_items == vol.total_sample_epochs == 3 * n
    # the audit trail is larger: a format break is one row per content scorer, and
    # faithfulness_anchor repeats the sweep reason of fabrication_anchor
    assert vol.scorer_rows == len(queue_mix.review_queue) == n * (len(CONTENT_SCORERS) + 2 + 1)
    assert vol.scorer_rows > vol.total_items
    line = format_queue_volume(vol)
    assert f"{3 * n} distinct (sample, epoch, reason) items on {3 * n} distinct (sample, epoch) pairs" in line


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_queue_items_per_asset_keep_reason_class_and_scorers(queue_mix, asset):
    sid = f"dev-queue-{asset.lower()}"
    items = {i.epoch: i for i in queue_items(queue_mix) if i.sample_id == sid}
    assert sorted(items) == [2, 3, 4]
    fb, sweep, err = items[2], items[3], items[4]
    assert fb.queue_class == "format_break" and fb.reason.startswith("format break:") and fb.route_key is None
    assert len(fb.scorers) == len(CONTENT_SCORERS)
    assert sweep.queue_class == SWEEP_ROUTE and sweep.route_key == route_key(sweep.reason)
    assert sweep.route_key.startswith(f"{SWEEP_ROUTE}: ") and "Fireblocks" in sweep.reason
    assert set(sweep.scorers) == {"fabrication_anchor", "faithfulness_anchor"}
    assert err.queue_class == "sample_error" and "queue-mix sample error" in err.reason
    assert err.scorers == ("sample_error",) and err.route_key is None
    # the clean epoch reaches neither the queue nor the volume
    assert all(t.success for t in queue_mix.trials if t.sample_id == sid and t.epoch == 1)


def test_queue_volume_code_review_fm_listing_and_unstated_reason():
    """A code-check reason is code_review by its route key, the fabrication listing
    that rides with FM-005 is fm_listed (not a 10(c) route), a flag with no reason is
    one code_review item, and two scorers repeating one reason are one item."""
    overrides = {
        "reconciling_items_anchor": _score(3, review=True, reasons=["negation: not reconciled"]),
        "faithfulness_anchor": _score(3, review=True, reasons=["negation: not reconciled"]),
        "fabrication_anchor": _score(0, ["FM-005"], review=True, reasons=[FABRICATION_REVIEW]),
        "completeness_anchor": _score(3, review=True),
    }
    rep = reliability_report([_sample("a", 1, overrides), _sample("a", 2)], k=2)
    items = {i.reason: i for i in queue_items(rep)}
    assert items["negation: not reconciled"].queue_class == CODE_ROUTE
    assert items["negation: not reconciled"].route_key == f"{CODE_ROUTE}: negation"
    assert items["negation: not reconciled"].scorers == ("reconciling_items_anchor", "faithfulness_anchor")
    assert items[FABRICATION_REVIEW].queue_class == "fm_listed" and items[FABRICATION_REVIEW].route_key is None
    assert items["review flagged with no stated reason"].queue_class == CODE_ROUTE
    vol = queue_volume(rep)
    assert vol.items == {"format_break": 0, "sample_error": 0, "narrative_sweep": 0, "code_review": 2, "fm_listed": 1}
    assert vol.sample_epochs == {"format_break": 0, "sample_error": 0, "narrative_sweep": 0, "code_review": 1, "fm_listed": 1}
    assert (vol.total_items, vol.total_sample_epochs, vol.scorer_rows) == (3, 1, 4)


def test_queue_volume_of_a_clean_report_is_zero_in_every_class():
    vol = queue_volume(reliability_report([_sample("a", e) for e in (1, 2)], k=2))
    assert set(vol.items) == set(vol.sample_epochs) == set(QUEUE_CLASSES)
    assert not any(vol.items.values()) and vol.total_items == vol.total_sample_epochs == vol.scorer_rows == 0


def test_review_rows_stay_one_per_scorer():
    """ReviewRow and review_queue keep their audit-trail shape (F2-07 adds a view)."""
    dup = {"fabrication_anchor": _score(0, review=True, reasons=["sweep hit"]),
           "faithfulness_anchor": _score(0, review=True, reasons=["sweep hit"])}
    rep = reliability_report([_sample("a", 1, dup)], k=1)
    assert [(r.scorer, r.reasons) for r in rep.review_queue] == [("fabrication_anchor", ("sweep hit",)),
                                                                 ("faithfulness_anchor", ("sweep hit",))]
    assert queue_volume(rep).total_items == 1


# --- Run configuration (contract C2): roster.check_run_config in the reader ----------

C2_ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
C2_OVERRIDES = {"clean": {}, "effort_low": {"reasoning_effort": "low"}, "temperature": {"temperature": 0.7}}


def _two_cases(task):
    keep = sorted(str(s.id) for s in task.dataset)[:2]
    task.dataset = task.dataset.filter(lambda s: str(s.id) in keep)
    return task


@pytest.fixture(scope="module")
def c2_logs(tmp_path_factory):
    """Two dev pilot cases at epochs 4 on mockllm: a clean run, the same run with an
    eval-level override (what a CLI --reasoning-effort low or --temperature 0.7 does),
    and the template baseline (none/none). Every run ends in status success."""
    from icfr_agent_evals.baseline import variance_commentary_template_baseline
    from icfr_agent_evals.task import variance_commentary

    out = {}
    for label, kw in C2_OVERRIDES.items():
        out[label] = inspect_eval(_two_cases(variance_commentary(split="dev", subset="pilot", epochs=4)),
                                  model="mockllm/model", model_roles=C2_ROLES, display="none",
                                  log_dir=str(tmp_path_factory.mktemp(f"c2-{label}")), **kw)[0]
    out["template"] = inspect_eval(_two_cases(variance_commentary_template_baseline(split="dev", subset="pilot")),
                                   log_dir=str(tmp_path_factory.mktemp("c2-template")), display="none")[0]
    for label, log in out.items():
        assert log.status == "success", (label, log.error)
    return out


def test_report_from_log_reads_a_clean_mock_pilot_log(c2_logs):
    rep = report_from_log(c2_logs["clean"])
    assert rep.k == 4 and rep.scored_samples == 2
    summary = require_run_config(c2_logs["clean"])
    assert summary["reasoning_effort"] == "high" and summary["scored_sample_epochs"] == 8


@pytest.mark.parametrize("label, field", [("effort_low", "reasoning_effort"), ("temperature", "temperature")])
def test_report_from_log_refuses_an_overridden_log_naming_the_field(c2_logs, label, field):
    from icfr_agent_evals import roster

    with pytest.raises(roster.RunConfigError) as raw:
        roster.check_run_config(c2_logs[label])
    with pytest.raises(GateError) as err:
        report_from_log(c2_logs[label])
    assert field in str(err.value) and str(err.value) == str(raw.value)  # the roster message, kept


def test_report_from_log_reads_the_template_log_without_the_check(c2_logs):
    """The template baseline has no evaluated model: check_run_config always refuses
    it, so the reader skips it (tests/test_run_config.py pins the refusal)."""
    from icfr_agent_evals import roster

    log = c2_logs["template"]
    with pytest.raises(roster.RunConfigError):
        roster.check_run_config(log)
    assert is_template_baseline_log(log) and not is_template_baseline_log(c2_logs["clean"])
    assert require_run_config(log) is None
    assert report_from_log(log).scored_samples == 2


def test_run_config_false_reads_an_overridden_log(c2_logs):
    """The opt-out exists for hand-built logs that make no model call; it is never the default."""
    assert report_from_log(c2_logs["effort_low"], run_config=False).scored_samples == 2


def test_require_run_config_passes_the_expected_counts(c2_logs):
    assert require_run_config(c2_logs["clean"], expected_samples=2)["samples"] == 2
    with pytest.raises(GateError, match="scored sample-epochs|expected"):
        require_run_config(c2_logs["clean"], expected_samples=20)
    with pytest.raises(GateError, match="epochs"):
        report_from_log(c2_logs["clean"], k=1)  # k = 1 asks for a one-epoch run


# --- FM-008 on the review-settled trial (section 6, section 13 item) --------------------------------

def _settled_fixture():
    from test_trials import AMOUNT, SWEEP, clean, record_for
    from test_trials import _score as tscore
    from test_trials import sample as tsample

    routed = tsample("a", 1, fabrication=tscore(0, reasons=[SWEEP]))
    b2 = tsample("b", 2, correctness=tscore(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=routed) + clean("b", e2=b2)
    rec = record_for(samples, overrides={("b", 2, route_key(AMOUNT)): "wrong"})
    return samples, rec


def test_fm008_without_a_record_is_the_labeled_code_only_reading():
    samples, _ = _settled_fixture()
    rep = reliability_report(samples, k=4)
    assert rep.fm008_basis == "code_only" and rep.settled is None
    assert rep.fm008 == rep.fm008_code_only == ("a", "b")


def test_fm008_with_a_record_reads_the_review_settled_trial():
    samples, rec = _settled_fixture()
    rep = reliability_report(samples, k=4, record=rec)
    assert rep.fm008_basis == "review_settled"
    assert rep.fm008 == ("b",) and rep.fm008_code_only == ("a", "b")
    assert rep.settled.pass_k == {"a": 1, "b": 0}
    assert rep.pass_k == {"a": 0, "b": 0}  # pass_k stays the code-only figure for the old call sites


def test_fm008_with_an_incomplete_record_is_refused():
    from icfr_agent_evals.trials import RecordError

    samples, rec = _settled_fixture()
    rec.rows = rec.rows[1:]
    with pytest.raises(RecordError):
        reliability_report(samples, k=4, record=rec)


def test_report_from_log_finds_the_run_by_eval_id_and_reads_review_settled_fm008():
    from icfr_agent_evals.reliability import content_pass_figures

    samples, rec = _settled_fixture()
    rec.runs = {"R1": {"eval_id": "OTHER"}, "R2": {"eval_id": "EVAL-B3"}}
    for m in rec.members:
        m["run_id"] = "R2"
    log = SimpleNamespace(samples=samples, eval=SimpleNamespace(
        eval_id="EVAL-B3", config=SimpleNamespace(epochs=4), task_args={"epochs": 4}))
    rep = report_from_log(log, run_config=False, record=rec)
    assert rep.fm008_basis == "review_settled" and rep.fm008 == ("b",)
    res = content_pass_figures(log, rec)
    assert {k: v["passed"] for k, v in res["figures"].items()} == {
        "review_settled": 1, "audit_adjusted": 1, "route_blind": 2, "code_only": 0}
