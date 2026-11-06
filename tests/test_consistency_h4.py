"""DeLong interval, H4 verdict and FM-007 (pre-registration section 2 H4 and section 6
Calibration, ratified 2026-11-03), and the rubric record of ratified score rules.

The DeLong interval is checked three ways: against a brute-force computation of
the structural components written here from the pairwise kernel (no ranks), against
a published worked example (Hanley and McNeil 1982, Radiology 143:29-36, Table I
ratings, as rerun under DeLong by the Stata [R] roctab manual: area 0.8932, std.
err. 0.0307, 95% CI 0.83295 to 0.95339), and against values computed once with
scipy.stats.rankdata, numpy and sklearn.metrics.roc_auc_score in a throwaway venv
outside the project (scipy 1.18.1, numpy 2.5.4, scikit-learn 1.9.1), pasted below.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth

from icfr_agent_evals import rubric
from icfr_agent_evals.consistency import (
    DELONG_Z,
    DIFFICULTY_BEARING,
    EVALUATION_SPLIT,
    FM007_NOTE,
    MODEL_DISPLAY_NAMES,
    NO_RUN_HEADER,
    SYNTHETIC,
    Discrimination,
    consistency_report,
    delong_ci,
    delong_components,
    fires_fm007,
    format_h4,
    format_report,
    h4_arm,
    h4_verdict,
    midranks,
    model_display_name,
    report_from_log,
    roc_auc,
)
from icfr_agent_evals.regression import SIGNOFF_CODES, SIGNOFF_TRIGGERS
from icfr_agent_evals.roster import EVALUATED_MODELS

# --- Independent reference: structural components from the pairwise kernel ----------


def _psi(x: float, y: float) -> float:
    return 1.0 if x > y else 0.5 if x == y else 0.0


def brute_delong(pos, neg, z=DELONG_Z):
    """DeLong 1988 directly: V10(X_i) = mean_j psi(X_i, Y_j), V01(Y_j) = mean_i psi(X_i, Y_j)."""
    m, n = len(pos), len(neg)
    v10 = [sum(_psi(x, y) for y in neg) / n for x in pos]
    v01 = [sum(_psi(x, y) for x in pos) / m for y in neg]
    auc = sum(_psi(x, y) for x in pos for y in neg) / (m * n)
    s10 = sum((v - auc) ** 2 for v in v10) / (m - 1)
    s01 = sum((v - auc) ** 2 for v in v01) / (n - 1)
    se = math.sqrt(s10 / m + s01 / n)
    return auc, se, max(0.0, auc - z * se), min(1.0, auc + z * se)


def _disc(pos, neg) -> Discrimination:
    return roc_auc([*pos, *neg], [True] * len(pos) + [False] * len(neg))


HANLEY_ABNORMAL = [1] * 3 + [2] * 2 + [3] * 2 + [4] * 11 + [5] * 33   # 51 abnormal (positive)
HANLEY_NORMAL = [1] * 33 + [2] * 6 + [3] * 6 + [4] * 11 + [5] * 2     # 58 normal

# scipy/numpy/sklearn reference values (z = scipy.stats.norm.ppf(0.975) there)
REFERENCE = {
    "hanley_mcneil_1982": (HANLEY_ABNORMAL, HANLEY_NORMAL, 0.8931710615280596, 0.030724408379381122),
    "agreement_levels_a": ([1.0, 1.0, 0.75, 0.75, 0.5, 1.0, 0.25], [0.5, 0.25, 1.0, 0.25, 0.5],
                           0.7285714285714286, 0.16413036132965797),
    "agreement_levels_b": ([1.0] * 9 + [0.75] * 3 + [0.5], [1.0, 0.75, 0.5, 0.5, 0.25, 0.25],
                           0.8461538461538461, 0.10991300127873235),
    "continuous": ([0.9, 0.8, 0.35, 0.7, 0.55, 0.62], [0.1, 0.4, 0.38, 0.6, 0.2],
                   0.8666666666666667, 0.1169045194450012),
}


def test_published_worked_example_hanley_mcneil_under_delong():
    d = _disc(HANLEY_ABNORMAL, HANLEY_NORMAL)
    assert round(d.auc, 4) == 0.8932
    assert round(d.se, 4) == 0.0307
    assert (round(d.ci_low, 5), round(d.ci_high, 5)) == (0.83295, 0.95339)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_matches_scipy_numpy_sklearn_reference(name):
    pos, neg, auc, se = REFERENCE[name]
    d = _disc(pos, neg)
    assert d.auc == pytest.approx(auc, abs=1e-12)
    assert d.se == pytest.approx(se, abs=1e-12)
    assert d.ci_low == pytest.approx(max(0.0, auc - DELONG_Z * se), abs=1e-12)
    assert d.ci_high == pytest.approx(min(1.0, auc + DELONG_Z * se), abs=1e-12)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_matches_brute_force_structural_components(name):
    pos, neg, _, _ = REFERENCE[name]
    d = _disc(pos, neg)
    auc, se, lo, hi = brute_delong(pos, neg)
    assert (d.auc, d.se, d.ci_low, d.ci_high) == pytest.approx((auc, se, lo, hi), abs=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_random_agreement_data_matches_brute_force(seed):
    rng = random.Random(seed)
    levels = [0.25, 0.5, 0.75, 1.0]
    m, n = rng.randint(2, 30), rng.randint(2, 30)
    pos = [rng.choice(levels[rng.randint(0, 2):]) for _ in range(m)]
    neg = [rng.choice(levels[: rng.randint(2, 4)]) for _ in range(n)]
    d = _disc(pos, neg)
    v10, v01 = delong_components(pos, neg)
    assert v10 == pytest.approx([sum(_psi(x, y) for y in neg) / n for x in pos], abs=1e-12)
    assert v01 == pytest.approx([sum(_psi(x, y) for x in pos) / m for y in neg], abs=1e-12)
    assert (d.auc, d.se, d.ci_low, d.ci_high) == pytest.approx(brute_delong(pos, neg), abs=1e-12)
    assert 0.0 <= d.ci_low <= d.auc <= d.ci_high <= 1.0


def test_midranks_share_the_mean_rank():
    assert midranks([0.5, 0.25, 0.5, 1.0, 0.5]) == [3.0, 1.0, 3.0, 5.0, 3.0]
    assert midranks([]) == []


# --- Degenerate cases -------------------------------------------------------------------

@pytest.mark.parametrize("pos, neg, empty", [([1.0, 0.5], [], "wrong"), ([], [1.0, 0.5], "correct")])
def test_empty_class_leaves_auc_and_ci_undefined(pos, neg, empty):
    d = _disc(pos, neg)
    assert d.auc is None and d.se is None and not d.ci_defined and not d.informative
    assert empty in d.note and empty in d.ci_note


@pytest.mark.parametrize("pos, neg, short", [([1.0], [0.5, 0.25], "correct"), ([1.0, 0.75], [0.25], "wrong")])
def test_one_sample_in_a_class_defines_the_auc_but_not_the_ci(pos, neg, short):
    d = _disc(pos, neg)
    assert d.auc == 1.0 and d.note is None
    assert not d.ci_defined and d.se is None and not d.informative
    assert f"fewer than 2 {short}" in d.ci_note
    with pytest.raises(ValueError):
        delong_ci(pos, neg)


def test_all_ties_give_a_zero_width_interval_at_one_half():
    d = _disc([0.5] * 3, [0.5] * 4)
    assert d.auc == 0.5 and d.se == 0.0 and (d.ci_low, d.ci_high) == (0.5, 0.5)
    assert not d.informative  # a lower bound of 0.5 does not exceed 0.5


def test_perfect_separation_gives_a_zero_width_interval_at_one():
    d = _disc([1.0, 1.0, 0.75], [0.5, 0.25])
    assert d.auc == 1.0 and d.se == 0.0 and (d.ci_low, d.ci_high) == (1.0, 1.0) and d.informative
    inverse = _disc([0.25, 0.5], [1.0, 0.75, 0.75])
    assert inverse.auc == 0.0 and (inverse.ci_low, inverse.ci_high) == (0.0, 0.0) and not inverse.informative


def test_interval_is_clipped_to_the_unit_range():
    pos, neg, auc, se = REFERENCE["agreement_levels_a"]
    d = _disc(pos, neg)
    assert auc + DELONG_Z * se > 1.0 and d.ci_high == 1.0


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_class_swap_mirrors_auc_and_keeps_the_standard_error(name):
    pos, neg, _, _ = REFERENCE[name]
    d, swapped = _disc(pos, neg), _disc(neg, pos)
    assert swapped.auc == pytest.approx(1.0 - d.auc, abs=1e-12)
    assert swapped.se == pytest.approx(d.se, abs=1e-12)
    assert swapped.ci_low == pytest.approx(max(0.0, 1.0 - d.auc - DELONG_Z * d.se), abs=1e-12)
    assert swapped.ci_high == pytest.approx(min(1.0, 1.0 - d.auc + DELONG_Z * d.se), abs=1e-12)


def test_z_is_the_two_sided_95_percent_quantile():
    assert DELONG_Z == 1.959964
    assert 0.5 * math.erfc(-DELONG_Z / math.sqrt(2)) == pytest.approx(0.975, abs=1e-7)


# --- H4 arm and verdict ------------------------------------------------------------------

INFORMATIVE = REFERENCE["agreement_levels_b"][:2]     # CI lower bound 0.6307
WIDE = REFERENCE["agreement_levels_a"][:2]            # CI lower bound 0.4069


def test_arm_reasons():
    assert h4_arm(SYNTHETIC, _disc([1.0], [])).reason.startswith("uninformative: AUC undefined")
    assert h4_arm(SYNTHETIC, _disc([1.0], [0.5, 0.25])).reason.startswith("uninformative: CI undefined")
    wide = h4_arm(SYNTHETIC, _disc(*WIDE))
    assert not wide.informative and "is at most 0.5" in wide.reason
    good = h4_arm(DIFFICULTY_BEARING, _disc(*INFORMATIVE))
    assert good.informative and "exceeds 0.5" in good.reason


def test_lower_bound_exactly_one_half_is_not_informative():
    d = Discrimination(0.6, 5, 5, se=0.1 / DELONG_Z, ci_low=0.5, ci_high=0.7)
    assert not d.informative and not h4_arm(DIFFICULTY_BEARING, d).informative


def _summary(d: Discrimination):
    return SimpleNamespace(discrimination=d)


def _verdict(syn: Discrimination, hard: Discrimination, split=EVALUATION_SPLIT, model="anthropic/claude-sonnet-5-5",
             k=4, run_scope_issues=()):
    report = SimpleNamespace(by_subset={SYNTHETIC: _summary(syn), DIFFICULTY_BEARING: _summary(hard)},
                             model=model, split=split, k=k, run_scope_issues=run_scope_issues)
    return h4_verdict(report)


@pytest.mark.parametrize("syn", [
    _disc([1.0] * 5, []),               # saturated: AUC undefined, no wrong samples
    _disc([1.0] * 5, [1.0]),            # one wrong sample: CI undefined
    _disc(*WIDE),                       # lower bound at most 0.5
])
def test_supported_when_synthetic_uninformative_and_difficulty_bearing_informative(syn):
    v = _verdict(syn, _disc(*INFORMATIVE))
    assert v.supported and v.reason.startswith("supported:") and v.is_h4_test
    assert v.model == "anthropic/claude-sonnet-5-5"


def test_not_supported_when_synthetic_is_informative():
    v = _verdict(_disc(*INFORMATIVE), _disc(*INFORMATIVE))
    assert not v.supported and "synthetic arm informative" in v.reason and "difficulty-bearing" not in v.reason


@pytest.mark.parametrize("hard", [_disc([1.0, 0.75], []), _disc([1.0], [0.5, 0.25]), _disc(*WIDE)])
def test_not_supported_when_difficulty_bearing_is_uninformative(hard):
    v = _verdict(_disc([1.0] * 5, []), hard)
    assert not v.supported and "difficulty-bearing arm uninformative" in v.reason


def test_both_failures_are_reported():
    v = _verdict(_disc(*INFORMATIVE), _disc(*WIDE))
    assert not v.supported and "synthetic arm informative" in v.reason and "difficulty-bearing arm uninformative" in v.reason


def test_a_verdict_off_the_evaluation_split_is_labeled_as_not_the_h4_test():
    v = _verdict(_disc([1.0] * 5, []), _disc(*INFORMATIVE), split="dev")
    assert v.supported and not v.is_h4_test
    assert "not the H4 test" in format_h4(v) and "anthropic/claude-sonnet-5-5" in format_h4(v)
    assert "split dev, not the evaluation split" in format_h4(v)
    assert "the pre-registered H4 test" in format_h4(_verdict(_disc([1.0] * 5, []), _disc(*INFORMATIVE)))


@pytest.mark.parametrize("kw,issue", [
    ({"k": 1}, "k 1, not the pre-registered k 4"),
    ({"model": "mockllm/model"}, "model mockllm/model is not one of the four roster workers"),
    ({"model": None}, "model unrecorded is not one of the four roster workers"),
    ({"run_scope_issues": ("run used limit 3; the gates score the whole split only",)}, "run used limit 3"),
    ({"run_scope_issues": ("run used subset 'pilot'; H4 scores the whole split",)}, "run used subset 'pilot'"),
    ({"run_scope_issues": None}, NO_RUN_HEADER),
])
def test_an_evaluation_split_verdict_off_the_whole_split_or_roster_is_a_rehearsal(kw, issue):
    v = _verdict(_disc([1.0] * 5, []), _disc(*INFORMATIVE), **kw)
    assert v.supported and not v.is_h4_test
    assert any(issue in i for i in v.scope_issues)
    text = format_h4(v)
    assert "not the H4 test, a rehearsal" in text and issue in text and "the pre-registered H4 test" not in text


def test_h4_names_the_model_as_h1_to_h3_do():
    assert set(MODEL_DISPLAY_NAMES) == set(EVALUATED_MODELS)
    assert MODEL_DISPLAY_NAMES == {
        "anthropic/claude-haiku-5-5": "Haiku 5.5",
        "anthropic/claude-sonnet-5": "Sonnet 5",
        "anthropic/claude-sonnet-5-5": "Sonnet 5.5",
        "anthropic/claude-opus-5-5": "Opus 5.5",
    }
    for model in EVALUATED_MODELS:
        v = _verdict(_disc([1.0] * 5, []), _disc(*INFORMATIVE), model=model)
        assert v.is_h4_test
        assert format_h4(v).startswith(f"H4, model {MODEL_DISPLAY_NAMES[model]} ({model}) (the pre-registered H4 test)")
    assert model_display_name("mockllm/model") == "mockllm/model" and model_display_name(None) == "unrecorded"


# --- FM-007 on constructed records ---------------------------------------------------------

TRUTH = make_truth()


def _out(**changes) -> str:
    out = good_output(TRUTH)
    out.update(changes)
    return json.dumps(out)


def _record(epoch, completion, sid, kind):
    truth = dataclasses.replace(TRUTH, case_kind=kind, category=None if kind == "standard" else "AC-1", case_id=sid)
    return SimpleNamespace(id=sid, epoch=epoch, metadata={"truth": truth.to_dict()},
                           output=SimpleNamespace(completion=completion), error=None)


PLANS = {
    ("std-a", "standard"): [_out(materiality_band="M3")] * 4,          # unanimous wrong: FM-007
    ("std-b", "standard"): [_out()] * 4,                               # unanimous right
    ("std-c", "standard"): [_out(materiality_band="M3")] * 3 + [_out()],  # wrong, not unanimous
    ("hc-a", "adversarial"): [_out(variance_direction="surplus")] * 4,  # FM-007
    ("hc-b", "edge"): [_out(materiality_band="M3")] * 4,               # FM-007
    ("hc-c", "edge"): ["prose"] * 4,                                   # parse failures never agree
}


def _report(**kw):
    records = [_record(e, c, sid, kind) for (sid, kind), outs in PLANS.items() for e, c in enumerate(outs, start=1)]
    return consistency_report(records, k=4, **kw)


def test_fm007_counted_overall_and_by_subset():
    report = _report(model="mockllm/model", split="dev")
    assert report.fm007 == ("hc-a", "hc-b", "std-a")
    assert report.fm007_count == 3
    assert report.fm007_by_subset == {SYNTHETIC: 1, DIFFICULTY_BEARING: 2}
    assert report.by_subset[SYNTHETIC].fm007 == ("std-a",)
    assert report.overall.fm007 == report.fm007
    assert not fires_fm007(report.sample("std-c")) and not fires_fm007(report.sample("hc-c"))
    text = format_report(report)
    assert FM007_NOTE in text and "3 sample(s) (synthetic 1, difficulty_bearing 2)" in text
    assert "model mockllm/model, split dev" in text and "H4, model mockllm/model" in text


def test_fm007_is_not_a_signoff_code():
    assert "FM-007" not in SIGNOFF_CODES and "FM-007" not in SIGNOFF_TRIGGERS


def test_report_from_log_carries_model_and_split():
    records = [_record(e, _out(), "std-b", "standard") for e in range(1, 5)]
    log = SimpleNamespace(samples=records, eval=SimpleNamespace(
        config=SimpleNamespace(epochs=4), task_args={"epochs": 4, "split": "dev"}, model="mockllm/model", metadata={}))
    report = report_from_log(log, run_config=False)  # hand-built log, no ModelEvent
    assert (report.model, report.split) == ("mockllm/model", "dev")
    log.eval.task_args = {"epochs": 4}
    log.eval.metadata = {"split": "eval"}
    assert report_from_log(log, run_config=False).split == "eval"


def _eval_log(n_ids=1, limit=None, sample_id=None, subset=None, header_samples=1, model="anthropic/claude-sonnet-5-5"):
    ids = ["std-b", "std-c"][:n_ids]
    records = [_record(e, _out(), sid, "standard") for sid in ids for e in range(1, 5)]
    return SimpleNamespace(samples=records, eval=SimpleNamespace(
        config=SimpleNamespace(epochs=4, limit=limit, sample_id=sample_id),
        task_args={"epochs": 4, "split": "eval", "subset": subset}, model=model,
        metadata={"split": "eval", "subset": subset},
        dataset=SimpleNamespace(samples=header_samples, sample_ids=ids)))


def test_report_from_log_records_whole_split_scope():
    whole = report_from_log(_eval_log(), run_config=False)  # hand-built logs: the scope issues, not C2
    assert whole.run_scope_issues == ()
    assert h4_verdict(whole).is_h4_test
    cases = {
        "limit": (_eval_log(limit=1), "run used limit 1"),
        "sample_id": (_eval_log(sample_id="std-b"), "run used sample_id 'std-b'"),
        "subset": (_eval_log(subset="pilot"), "run used subset 'pilot'"),
        "short": (_eval_log(header_samples=2), "scored samples 1 differ from the dataset's sample count 2"),
        "off roster": (_eval_log(model="mockllm/model"), "model mockllm/model is not one of the four roster workers"),
    }
    for name, (log, issue) in cases.items():
        v = h4_verdict(report_from_log(log, run_config=False))
        assert not v.is_h4_test, name
        assert any(issue in i for i in v.scope_issues), (name, v.scope_issues)


def test_report_built_without_a_log_is_never_the_h4_test():
    records = [_record(e, _out(), "std-b", "standard") for e in range(1, 5)]
    report = consistency_report(records, k=4, model="anthropic/claude-sonnet-5-5", split="eval")
    v = h4_verdict(report)
    assert not v.is_h4_test and v.scope_issues == (NO_RUN_HEADER,)


# --- Rubric record ----------------------------------------------------------------------

RATIFIED_KEYS = {"faithfulness.combination", "reconciling_items.fm015", "reconciling_items.omission", "correctness.fm016"}
D5_KEY = "correctness.band_overstatement"


def test_rubric_ratified_and_provisional_records():
    assert set(rubric.RATIFIED) == RATIFIED_KEYS | {D5_KEY}
    assert rubric.RATIFIED_ON == {**dict.fromkeys(RATIFIED_KEYS, "2026-11-03"), D5_KEY: "2026-11-04"}
    assert rubric.PROVISIONAL == {}  # D5 ratified 2026-11-04
    assert not set(rubric.RATIFIED) & set(rubric.PROVISIONAL)
    assert rubric.RATIFIED["reconciling_items.omission"].startswith("an omitted eligible item scores 2")


def test_rubric_calibration_line_states_the_ratified_rules():
    line = rubric.ANCHOR_STATUS[rubric.Dimension.CALIBRATION]
    for phrase in ("reported, not scored in Phase 1", "DeLong", "FM-007", "H4", "p = 1", "m = 6",
                   "new tagged pre-registration"):
        assert phrase in line, phrase
    assert "candidate" not in line and "reported only" not in line and "pending" not in line
    assert "calibration.semantic_agreement" not in rubric.LLM_GRADED_SUBDIMENSIONS
    assert not any(s.startswith("calibration.") for s in rubric.LLM_GRADED_SUBDIMENSIONS)
