"""Sample Consistency tests (pre-registration section 6 Calibration, section 2 H4).

Unit tests on constructed (sample, epoch) records, then one end-to-end run of
variance_commentary on a handful of dev-split cases with epochs = 4 through
mockllm, where each case receives a planned multiset of four outputs.
"""

import dataclasses
from decimal import Decimal
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth, tx_hash

from icfr_agent_evals.consistency import (
    CALIBRATION_SCORE_NOTE,
    DIFFICULTY_BEARING,
    FM007_NOTE,
    SYNTHETIC,
    CanonicalAnswer,
    answer_correct,
    assert_gate,
    calibration_table,
    canonical_answer,
    consistency_report,
    format_report,
    h4_verdict,
    modal_class,
    read_epoch,
    report_from_log,
    roc_auc,
    sample_consistency,
    whole_dollars,
)
from icfr_agent_evals.reliability import GateError
from icfr_agent_evals.reliability import report_from_log as reliability_report_from_log
from test_reliability import c2_logs  # noqa: F401  (contract C2 mock logs: clean, overrides, template)
from icfr_agent_evals.rubric import ANCHOR_STATUS, WIRED_SCORERS, Dimension

PROSE = "Here is my commentary: the reserves look fine."
TRUTH = make_truth()  # BTC 2025-12, $120,000.00 M2 shortfall, one eligible item WD-2512-04417


def _out(truth=TRUTH, **changes) -> str:
    out = good_output(truth)
    out.update(changes)
    return json.dumps(out)


def _wrong_band(truth=TRUTH) -> str:
    return _out(truth, materiality_band="M3")


def _wrong_direction(truth=TRUTH) -> str:
    return _out(truth, variance_direction="surplus")


def _record(epoch, completion, truth=TRUTH, sid=None, error=None):
    return SimpleNamespace(
        id=sid or truth.case_id,
        epoch=epoch,
        metadata={"truth": truth.to_dict()},
        output=SimpleNamespace(completion=completion),
        error=error,
    )


def _sample(completions, truth=TRUTH, sid=None):
    """sample_consistency over completions given for epochs 1..k in order."""
    answers = [read_epoch(_record(e, c, truth, sid)) for e, c in enumerate(completions, start=1)]
    return sample_consistency(answers, truth, len(completions))


def _parsed(text: str) -> dict:
    return json.loads(text)


# --- Canonical answer ----------------------------------------------------------------

def test_canonical_answer_uses_the_scoring_normalizers():
    a = canonical_answer(good_output(TRUTH))
    b = canonical_answer({
        **good_output(TRUTH),
        "asset": " btc ",
        "period": "2025/12",
        "variance_direction": "Shortfall ",
        "materiality_band": "m2",
        "reconciling_items": [{"reference": " WD-2512-04417 ", "amount_usd": 80_000.0}],
    })
    assert a == b
    assert a == CanonicalAnswer("shortfall", "m2", 120_000, "BTC", "2025-12", (("WD-2512-04417", 80_000),))


def test_amount_rounds_half_up_to_whole_dollars_with_sign_kept():
    assert whole_dollars(120_000.40) == whole_dollars(120_000.49) == 120_000
    assert whole_dollars(120_000.50) == 120_001
    assert whole_dollars(120_000.49) != whole_dollars(120_000.51)
    assert whole_dollars(-120_000.0) != whole_dollars(120_000.0)
    assert whole_dollars(7) == 7


def test_reference_multiset_ignores_order_and_keeps_repeats():
    base = good_output(TRUTH)
    one = [{"reference": "WD-2512-04417", "amount_usd": 80_000.0}]
    two = [{"reference": "JE-25-12-0193", "amount_usd": 40_000.0}, *one]
    assert canonical_answer({**base, "reconciling_items": two}) == canonical_answer({**base, "reconciling_items": two[::-1]})
    assert canonical_answer({**base, "reconciling_items": one * 2}) != canonical_answer({**base, "reconciling_items": one})


def test_an_alias_citation_is_a_different_answer_but_still_correct():
    # exact canonical reference comparison (section 6); aliases are not resolved for agreement
    alias = _out(reconciling_items=[{"reference": tx_hash("BTC"), "amount_usd": 80_000.0}])
    s = _sample([_out(), _out(), alias, alias])
    assert s.agreement == 0.5 and s.tie and s.modal_correct
    assert all(a.correct for a in s.epochs)


def test_unparsed_period_text_still_compares():
    a = canonical_answer({**good_output(TRUTH), "period": "Q4 2025"})
    assert a.period == "unparsed:Q4 2025"


# --- Correctness from the existing anchors -------------------------------------------

def test_good_output_is_correct():
    assert answer_correct(good_output(TRUTH), TRUTH) == (True, ())


@pytest.mark.parametrize("change, fragment", [
    ({"materiality_band": "M3"}, "structured points 2/3"),
    ({"variance_direction": "surplus"}, "structured points 2/3"),
    ({"variance_amount_usd": 120_002.0}, "structured points 2/3"),
    ({"period": "2025-11"}, "FM-016"),
    ({"asset": "ETH"}, "FM-016"),
    ({"reconciling_items": []}, "reconciling items 1/3"),
    ({"reconciling_items": [{"reference": "WD-9999-00000", "amount_usd": 80_000.0}]}, "FM-015"),
])
def test_wrong_canonical_fields_are_incorrect(change, fragment):
    ok, reasons = answer_correct({**good_output(TRUTH), **change}, TRUTH)
    assert not ok and any(fragment in r for r in reasons), reasons


def test_amount_within_one_dollar_is_correct():
    assert answer_correct({**good_output(TRUTH), "variance_amount_usd": 120_000.99}, TRUTH)[0]


def test_read_epoch_format_break_and_sample_error_are_parse_failures():
    fb = read_epoch(_record(1, PROSE))
    assert fb.parse_failure and not fb.correct and fb.parse_error.startswith("format break")
    err = read_epoch(_record(2, _out(), error=SimpleNamespace(message="boom")))
    assert err.parse_failure and not err.correct and "boom" in err.parse_error
    missing = read_epoch(SimpleNamespace(id="x", epoch=3, metadata={"truth": TRUTH.to_dict()}, output=None, error=None))
    assert missing.parse_failure


# --- Mode and agreement --------------------------------------------------------------

def test_four_identical_correct():
    s = _sample([_out()] * 4)
    assert (s.agreement, s.modal_correct, s.unanimous, s.tie) == (1.0, True, True, False)


def test_three_correct_one_wrong_band():
    s = _sample([_out(), _wrong_band(), _out(), _out()])
    assert (s.agreement, s.modal_correct, s.tie) == (0.75, True, False)
    assert s.modal_epochs == (1, 3, 4)


@pytest.mark.parametrize("order", [("c", "c", "w", "w"), ("w", "w", "c", "c"), ("c", "w", "c", "w")])
def test_two_two_tie_goes_to_the_incorrect_class(order):
    s = _sample([_out() if x == "c" else _wrong_direction() for x in order])
    assert (s.agreement, s.modal_correct, s.tie) == (0.5, False, True)
    assert s.modal_answer.variance_direction == "surplus"


def test_tie_between_two_wrong_classes_goes_to_the_earliest_epoch():
    s = _sample([_wrong_direction(), _wrong_band(), _wrong_band(), _wrong_direction()])
    assert s.modal_epochs == (1, 4) and s.modal_answer.variance_direction == "surplus"


def test_parse_failures_never_agree_even_with_each_other():
    s = _sample([PROSE] * 4)
    assert (s.agreement, s.modal_correct, s.unanimous, s.tie) == (0.25, False, False, True)
    assert s.modal_answer is None


def test_a_parse_failure_does_not_join_a_class():
    s = _sample([_out(), _out(), _out(), PROSE])
    assert (s.agreement, s.modal_correct) == (0.75, True)


def test_four_different_answers_with_one_correct_is_wrong_at_one_quarter():
    s = _sample([_out(), _wrong_band(), _wrong_direction(), PROSE])
    assert (s.agreement, s.modal_correct, s.tie) == (0.25, False, True)


def test_all_agree_wrong_is_unanimous_and_fires_fm007():
    report = consistency_report([_record(e, _wrong_band()) for e in range(1, 5)], k=4)
    s = report.samples[0]
    assert (s.agreement, s.modal_correct, s.unanimous) == (1.0, False, True)
    assert report.fm007 == (TRUTH.case_id,)


def test_a_class_with_mixed_correctness_is_wrong():
    # 120,000.60 is within $1 of the truth and 120,001.40 is not; both round to 120,001
    near, far = _out(variance_amount_usd=120_000.60), _out(variance_amount_usd=120_001.40)
    s = _sample([near, near, far, far])
    assert (s.agreement, s.unanimous, s.modal_correct) == (1.0, True, False)


def test_modal_class_needs_epochs():
    with pytest.raises(ValueError):
        modal_class([])


# --- Discrimination ------------------------------------------------------------------

def test_auc_perfect_inverse_and_all_tied():
    assert roc_auc([1.0, 0.75, 0.25], [True, True, False]).auc == 1.0
    assert roc_auc([0.25, 1.0], [True, False]).auc == 0.0
    assert roc_auc([0.5, 0.5, 0.5], [True, False, False]).auc == 0.5


def test_auc_counts_ties_one_half():
    # correct {1.0, 0.5}, wrong {1.0, 0.5, 0.25}: pairs 0.5+1+1 + 0+0.5+1 = 4 of 6
    d = roc_auc([1.0, 0.5, 1.0, 0.5, 0.25], [True, True, False, False, False])
    assert d.auc == pytest.approx(4 / 6) and (d.n_correct, d.n_wrong) == (2, 3) and d.note is None


@pytest.mark.parametrize("labels, empty", [([True, True], "wrong"), ([False, False], "correct")])
def test_auc_undefined_when_a_class_is_empty(labels, empty):
    d = roc_auc([1.0, 0.5], labels)
    assert d.auc is None and empty in d.note


def test_auc_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        roc_auc([1.0], [True, False])


# --- Calibration table and report ----------------------------------------------------

def test_calibration_table_has_every_level_and_signed_gap():
    rows = calibration_table([_sample([_out()] * 4), _sample([_wrong_band()] * 4), _sample([PROSE] * 4)], 4)
    assert [r.agreement for r in rows] == [0.25, 0.5, 0.75, 1.0]
    assert [(r.n, r.correct) for r in rows] == [(1, 0), (0, 0), (0, 0), (2, 1)]
    assert rows[1].accuracy is None and rows[1].gap is None
    assert rows[3].accuracy == 0.5 and rows[3].gap == 0.5  # over-confident by half
    assert rows[0].gap == 0.25


def _kind(truth, kind, sid):
    return dataclasses.replace(truth, case_kind=kind, category=None if kind == "standard" else "AC-1", case_id=sid)


def test_report_splits_synthetic_and_difficulty_bearing():
    plans = {
        ("std-a", "standard"): [_out()] * 4,
        ("std-b", "standard"): [_out()] * 3 + [_wrong_band()],
        ("hc-a", "adversarial"): [_wrong_band()] * 4,
        ("hc-b", "edge"): [_out(), _out(), PROSE, _wrong_band()],
    }
    records = []
    for (sid, kind), outs in plans.items():
        truth = _kind(TRUTH, kind, sid)
        records += [_record(e, c, truth, sid) for e, c in enumerate(outs, start=1)][::-1]  # order must not matter
    report = consistency_report(records, k=4)
    assert [s.sample_id for s in report.samples] == ["hc-a", "hc-b", "std-a", "std-b"]
    assert {s.sample_id: s.subset for s in report.samples} == {
        "std-a": SYNTHETIC, "std-b": SYNTHETIC, "hc-a": DIFFICULTY_BEARING, "hc-b": DIFFICULTY_BEARING,
    }
    syn, hard = report.by_subset[SYNTHETIC], report.by_subset[DIFFICULTY_BEARING]
    assert syn.n == 2 and syn.accuracy == 1.0 and syn.discrimination.auc is None  # saturated: no wrong samples
    assert hard.n == 2 and hard.discrimination.auc == 0.0  # the unanimous answer is the wrong one
    assert report.overall.n == 4 and report.overall.discrimination.auc == pytest.approx(0.5 / 3)
    assert report.fm007 == ("hc-a",)
    text = format_report(report)
    assert FM007_NOTE in text and CALIBRATION_SCORE_NOTE in text and "AUC=n/a" in text


@pytest.mark.parametrize("epochs", [(1, 2, 3), (1, 2, 3, 3), (1, 2, 3, 4, 5), (0, 1, 2, 3)])
def test_a_sample_without_exactly_k_epochs_fails_the_gate(epochs):
    with pytest.raises(GateError):
        consistency_report([_record(e, _out()) for e in epochs], k=4)


def test_assert_gate_checks_k_and_sample_count():
    report = consistency_report([_record(e, _out()) for e in range(1, 5)], k=4)
    assert_gate(report, expected_samples=1)
    with pytest.raises(GateError):
        assert_gate(report, expected_samples=2)
    with pytest.raises(GateError):
        assert_gate(consistency_report([_record(1, _out())], k=1), expected_samples=1)


def _log(samples, epochs=4, declared=4):
    return SimpleNamespace(
        samples=samples,
        eval=SimpleNamespace(config=SimpleNamespace(epochs=epochs), task_args={"epochs": declared}),
    )


def test_report_from_log_guards():
    with pytest.raises(GateError):
        report_from_log(_log(None))
    with pytest.raises(GateError):
        report_from_log(_log([_record(1, _out())], epochs=1, declared=4))
    report = report_from_log(_log([_record(e, _out()) for e in range(1, 5)]), run_config=False)  # hand-built log, no ModelEvent
    assert report.k == 4 and report.samples[0].agreement == 1.0


def test_calibration_stays_unscored():
    # Calibration is reported, not scored, in Phase 1 (no 0 to 3 anchor), so no scorer reports it
    assert Dimension.CALIBRATION.value not in WIRED_SCORERS.values()
    assert "consistency.py" in ANCHOR_STATUS[Dimension.CALIBRATION]


# --- End to end: Inspect, mockllm, dev split, epochs = 4 ----------------------------

def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "consistency_mock.py"
    spec = importlib.util.spec_from_file_location("consistency_mock", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_end_to_end_planned_multisets(tmp_path):
    mock = _script()
    log, report, picked = mock.run(str(tmp_path))
    assert log.eval.task_args.get("split") == "dev" and log.eval.config.epochs == 4
    assert len(log.samples) == 4 * len(picked) == 32
    assert {p.subset for p in picked.values()} == {SYNTHETIC, DIFFICULTY_BEARING}
    for sid, plan in picked.items():
        s = report.sample(sid)
        assert s.agreement == pytest.approx(plan.agreement), plan.name
        assert s.modal_correct == plan.modal_correct, plan.name
        assert s.subset == plan.subset, plan.name
        assert sorted("p" if a.parse_failure else "c" if a.correct else "w" for a in s.epochs) == sorted(
            "p" if x == "p" else "c" if x == "c" else "w" for x in plan.outputs
        ), plan.name
    assert mock.check(report, picked) == []
    assert report.overall.discrimination.auc == pytest.approx(11.5 / 15)
    # FM-007 is sample-level: S4 (four identical wrong amounts) fires it, and it
    # never enters a per-epoch trial, so content-pass^4 is already 0 there
    assert report.fm007_count == 1 and report.fm007_by_subset == {SYNTHETIC: 1, DIFFICULTY_BEARING: 0}
    rel = reliability_report_from_log(log)
    assert all(rel.pass_k[sid] == 0 for sid in report.fm007)
    assert "FM-007" not in rel.fm_counts
    # H4 on the dev split exercises the code only; one correct difficulty-bearing sample
    verdict = h4_verdict(report)
    assert (verdict.model, verdict.split, verdict.is_h4_test) == ("mockllm/model", "dev", False)
    assert report.by_subset[SYNTHETIC].discrimination.auc == pytest.approx(0.75)
    assert not verdict.supported and "fewer than 2 correct" in verdict.reason


# --- Review fixes C1 to C3 -----------------------------------------------------------

CONTRADICTING = "On-chain reserves exceeded ledger liabilities, a surplus for the period."


def test_narrative_contradiction_is_not_correct_and_unanimous_wrong_is_fm007():
    # C1: correct structured fields with a narrative that contradicts the direction
    # set the Correctness anchor to 0 (section 6), so the answer is wrong
    bad = good_output(TRUTH, narrative=CONTRADICTING)
    ok, reasons = answer_correct(bad, TRUTH)
    assert not ok and any("FM-003" in r for r in reasons), reasons
    report = consistency_report([_record(e, json.dumps(bad)) for e in range(1, 5)], k=4)
    s = report.samples[0]
    assert s.unanimous and not s.modal_correct
    assert report.fm007 == (TRUTH.case_id,)


def test_epochs_differing_only_in_narrative_share_a_class_that_is_wrong():
    bad = json.dumps(good_output(TRUTH, narrative=CONTRADICTING))
    s = _sample([_out(), bad, _out(), _out()])
    assert s.agreement == 1.0 and not s.modal_correct


def _items(amount):
    return [{"reference": "WD-2512-04417", "amount_usd": amount}]


def test_reconciling_item_amount_is_part_of_the_answer():
    # C2: a different item amount is a different answer (FM-015 on a mismatch)
    base = good_output(TRUTH)
    right = canonical_answer({**base, "reconciling_items": _items(80_000.0)})
    assert right.reconciling_references == (("WD-2512-04417", 80_000),)
    assert canonical_answer({**base, "reconciling_items": _items(50_000.0)}) != right
    # scoring compares the item amount unsigned, so the sign is not a different answer
    assert canonical_answer({**base, "reconciling_items": _items(-80_000.0)}) == right


def test_epochs_with_different_item_amounts_disagree():
    w50, w60 = _out(reconciling_items=_items(50_000.0)), _out(reconciling_items=_items(60_000.0))
    s = _sample([_out(), w50, w50, _out()])
    assert s.agreement == 0.5 and s.tie and not s.modal_correct
    report = consistency_report([_record(e, c) for e, c in enumerate([w50, w60, w50, w60], start=1)], k=4)
    assert report.samples[0].agreement == 0.5 and not report.samples[0].unanimous
    assert report.fm007 == ()


@pytest.mark.parametrize("amount", ["1e300", "123456789012345678901234567890", "-1e300"])
def test_a_huge_finite_amount_is_a_wrong_answer_not_a_crash(amount):
    # C3: any finite JSON number the parser accepts must be measured, never raise
    huge = _out().replace('"variance_amount_usd": 120000.0', '"variance_amount_usd": ' + amount)
    assert amount in huge
    s = consistency_report([_record(e, huge if e == 1 else _out()) for e in range(1, 5)], k=4).samples[0]
    assert not s.epochs[0].parse_failure and not s.epochs[0].correct
    assert s.agreement == 0.75 and s.modal_correct
    assert whole_dollars(amount) == int(Decimal(amount))


def test_huge_item_amount_does_not_crash():
    huge = _out().replace('"amount_usd": 80000.0', '"amount_usd": 1e300')
    assert "1e300" in huge
    s = consistency_report([_record(e, huge) for e in range(1, 5)], k=4).samples[0]
    assert s.unanimous and not s.modal_correct


# --- Run configuration (contract C2) in the consistency report ------------------------

def test_consistency_report_reads_a_clean_mock_pilot_log(c2_logs):
    report = report_from_log(c2_logs["clean"])
    assert report.k == 4 and report.scored_samples == 2


@pytest.mark.parametrize("label, field", [("effort_low", "reasoning_effort"), ("temperature", "temperature")])
def test_consistency_report_refuses_an_overridden_log_naming_the_field(c2_logs, label, field):
    with pytest.raises(GateError, match=field):
        report_from_log(c2_logs[label])


def test_consistency_report_reads_the_template_log_without_the_check(c2_logs):
    """The reader skips C2 on the template; pilot_report never builds Sample Consistency for it (F2-09)."""
    assert report_from_log(c2_logs["template"]).scored_samples == 2
