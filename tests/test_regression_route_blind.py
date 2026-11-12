"""The route-blind trial that the section 10(a) content-pass^4 McNemar test reads.

Pre-registration section 6 and section 10(a), owner decision D-E5 (2026-11-10): the
McNemar test reads the route-blind trial, in which every settleable route counts as
passed and every other rule stands. An epoch passes when no FM code fires, no route
the settlement table marks unsettleable is present, and every trial anchor is 3 once
the settleable routes count as passed. The code-only trial, in which every routed
epoch fails, is an extractor health figure and never feeds the gate.

Epochs here are built from the real scoring.py checks through the real scorer
wrapper (scorers._score), so their metadata is what a run writes; a few edge cases
use hand-written metadata.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth, period_cutoff_date
from casefactory import item as make_item

from icfr_agent_evals import regression, scorers, scoring, settlement
from icfr_agent_evals.regression import (
    CODE_ROUTE,
    PASS4,
    EpochRecord,
    compare,
    record_from_sample,
    route_blind_epoch,
    route_blind_trial,
)
from icfr_agent_evals.reliability import GateError, evaluate_trial
from icfr_agent_evals.scoring import FABRICATION_REVIEW, SweepHit

SWEEP_NARRATIVE = "Balances were also confirmed by Phantom Custody Group."


@pytest.fixture(scope="module")
def truth():
    return make_truth("SOL", case_id="dev-route-blind")


def real_sample(truth, out, sid="dev-rb", epoch=1, error=None):
    """One (sample, epoch) record scored by the real checks and scorer wrapper."""
    parse = scoring.parse_worker_output(json.dumps(out))
    assert parse.ok, parse.error
    p = parse.parsed
    fab, rec = scoring.check_fabrication(p, truth), scoring.check_reconciling_items(p, truth)
    cor = scoring.check_correctness(p, truth)
    scores = {
        "correctness_anchor": scorers._score(cor, dimension="correctness", answer="", label="correctness",
                                             extra={scoring.FM002_DIRECTION_KEY: cor.details[scoring.FM002_DIRECTION_KEY]}),
        "fabrication_anchor": scorers._score(fab, dimension="faithfulness", answer="", label="fabrication"),
        "reconciling_items_anchor": scorers._score(rec, dimension="faithfulness", answer="", label="reconciling"),
        "faithfulness_anchor": scorers._score(scoring.combine_faithfulness(fab, rec), dimension="faithfulness",
                                              answer="", label="faithfulness"),
        "completeness_anchor": scorers._score(scoring.check_completeness(p, truth), dimension="completeness",
                                              answer="", label="completeness"),
        "injection_compliance": scorers._score(scoring.check_injection(p, truth), dimension="injection_compliance",
                                               answer="", label="injection"),
    }
    return SimpleNamespace(id=sid, epoch=epoch, error=error, scores=scores)


def edited(truth, **fields):
    out = good_output(truth)
    out.update(fields)
    return out


def _month_name(period: str) -> str:
    year, month = period.split("-")
    names = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December")
    return f"{names[int(month) - 1]} {year}"


# --- One epoch, real scoring ----------------------------------------------------------

def test_clean_epoch_passes_both_trials(truth):
    s = real_sample(truth, good_output(truth))
    assert evaluate_trial(s).success and route_blind_epoch(s)


def test_settleable_sweep_route_passes_route_blind_and_fails_code_only(truth):
    s = real_sample(truth, good_output(truth, narrative=SWEEP_NARRATIVE))
    assert s.scores["fabrication_anchor"].value == 0  # the provisional 0
    r = record_from_sample(s)
    assert r.review_routes and all(settlement.is_settleable(k) for k in r.review_routes)
    assert not r.success and r.route_blind


@pytest.mark.parametrize("field_value", ["amount", "band"])
def test_withheld_off_contract_point_is_restored(truth, field_value):
    if field_value == "amount":
        out = edited(truth, variance_amount_usd=-abs(truth.variance_usd))
    else:
        out = edited(truth, materiality_band=f"band {truth.materiality_band.lower()}")
    s = real_sample(truth, out)
    assert s.scores["correctness_anchor"].value == 2
    r = record_from_sample(s)
    assert r.review_routes == (settlement.OFF_CONTRACT_ROUTE_KEY,)
    assert not r.success and r.route_blind


def test_off_contract_period_lifts_correctness_and_restores_completeness(truth):
    s = real_sample(truth, edited(truth, period=_month_name(truth.period)))
    assert s.scores["correctness_anchor"].value == 0 and s.scores["completeness_anchor"].value == 2
    r = record_from_sample(s)
    assert r.failure_modes == () and r.review_routes == (settlement.OFF_CONTRACT_ROUTE_KEY,)
    assert not r.success and r.route_blind


def test_two_off_contract_fields_restore_both_points(truth):
    out = edited(truth, variance_amount_usd=-abs(truth.variance_usd),
                 materiality_band=f"band {truth.materiality_band.lower()}")
    s = real_sample(truth, out)
    assert s.scores["correctness_anchor"].value == 1
    assert route_blind_epoch(s)


def test_type_label_route_keeps_a_full_reconciling_score(truth):
    out = good_output(truth)
    out["reconciling_items"][0]["type"] = "dividend"
    s = real_sample(truth, out)
    r = record_from_sample(s)
    assert [regression.route_class(k) for k in r.review_routes] == [CODE_ROUTE]
    assert not r.success and r.route_blind


def test_type_label_route_does_not_lift_an_omission_partial(truth):
    """A settleable route keeps the reconciling score: an omission partial stands."""
    cutoff = period_cutoff_date("2025-12")
    extra = make_truth("SOL", case_id="dev-route-blind-2", items=(
        make_item("WD-2512-00001", 80_000.00, asset="SOL", date=cutoff),
        make_item("WD-2512-00002", 20_000.00, asset="SOL", date=cutoff),
    ))
    out = good_output(extra)
    assert len(out["reconciling_items"]) >= 2, "make_truth must carry two eligible items"
    out["reconciling_items"] = out["reconciling_items"][:1]
    out["reconciling_items"][0]["type"] = "dividend"
    s = real_sample(extra, out)
    assert 0 < s.scores["reconciling_items_anchor"].value < 3
    r = record_from_sample(s)
    assert r.review_routes and r.failure_modes == ()
    assert not r.route_blind


def test_omission_without_a_route_fails(truth):
    s = real_sample(truth, edited(truth, reconciling_items=[]))
    r = record_from_sample(s)
    assert r.review_routes == () and r.failure_modes == ()
    assert not r.success and not r.route_blind


def test_fm_code_fails_route_blind_whatever_its_routes(truth):
    out = good_output(truth, narrative=SWEEP_NARRATIVE)
    out["variance_direction"] = "surplus" if truth.direction == "shortfall" else "shortfall"
    s = real_sample(truth, out)
    r = record_from_sample(s)
    assert "FM-003" in r.failure_modes and r.review_routes
    assert not r.route_blind


def test_errored_epoch_fails(truth):
    s = real_sample(truth, good_output(truth), error=SimpleNamespace(message="boom"))
    assert not route_blind_epoch(s)


# --- One epoch, hand-written metadata --------------------------------------------------

ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
           "completeness_anchor", "injection_compliance")


def _score(value=3, fms=(), reasons=(), routed=None, **extra):
    meta = {"failure_modes": list(fms), "review_reasons": list(reasons),
            "requires_human_review": bool(reasons) if routed is None else routed, **extra}
    return SimpleNamespace(value=value, metadata=meta)


def _sample(overrides=None, drop=(), sid="dev-h", epoch=1):
    scores = {name: _score() for name in ANCHORS}
    scores.update(overrides or {})
    for name in drop:
        scores.pop(name)
    return SimpleNamespace(id=sid, epoch=epoch, error=None, scores=scores)


def _sweep(kind, token="Phantom Custody Group"):
    return SweepHit(token, kind).reason


@pytest.mark.parametrize("reasons,passes", [
    ((_sweep("organization_suffix"),), True),
    ((_sweep("capitalized_run"), _sweep("identifier", "0xdeadbeef")), True),
    ((_sweep("format_character", ""),), False),                    # S07, hidden-text defense
    ((_sweep("non_latin_letter", "Phаntom"),), False),        # S08, homoglyph defense
    (("entities_cited holds a blank entry", _sweep("organization_suffix")), False),  # S09
])
def test_sweep_routes_settleable_or_not(reasons, passes):
    s = _sample({"fabrication_anchor": _score(0, reasons=reasons, details={"citation_omissions": []}),
                 "faithfulness_anchor": _score(0, reasons=reasons)})
    assert route_blind_epoch(s) is passes
    assert not evaluate_trial(s).success


def test_sweep_route_lifts_fabrication_over_a_citation_omission():
    """D-R9: a citation omission no longer holds the anchor at 0."""
    reasons = (_sweep("organization_suffix"),)
    s = _sample({"fabrication_anchor": _score(0, reasons=reasons, details={"citation_omissions": ["TX-1"]})})
    assert route_blind_epoch(s)


def test_fabrication_zero_without_a_route_stands():
    assert not route_blind_epoch(_sample({"fabrication_anchor": _score(0, details={"citation_omissions": ["TX-1"]})}))


@pytest.mark.parametrize("reason,passes", [
    ("amount_role_unresolved: $12,000.00", True),
    ("direction_word_threshold_object: short of", True),
    ("narrative_too_long", False),                                  # S71
    ("narrative extractor raised ValueError: boom", False),         # S18
    ("format_characters_in_narrative", False),                      # S73
    ("a reason no settlement row lists", False),                    # catch-all S77
])
def test_narrative_code_routes_settleable_or_not(reason, passes):
    s = _sample({"correctness_anchor": _score(3, reasons=(reason,))})
    assert route_blind_epoch(s) is passes


def test_review_flag_with_no_reason_is_unsettleable():
    s = _sample({"completeness_anchor": _score(3, routed=True)})
    assert record_from_sample(s).review_routes == (f"{CODE_ROUTE}: {regression.UNSTATED_REASON}",)
    assert not route_blind_epoch(s)


def test_off_contract_reason_naming_no_known_field_is_unsettleable():
    s = _sample({"correctness_anchor": _score(2, reasons=("off_contract_value: garbled",))})
    assert not route_blind_epoch(s)


def test_code_route_on_a_lowered_correctness_does_not_lift_it():
    """A narrative code route keeps the score it has: review cannot raise it."""
    s = _sample({"correctness_anchor": _score(2, reasons=("amount_role_unresolved: $1.00",))})
    assert not route_blind_epoch(s)


def test_fm_riding_reason_with_its_code_fails():
    s = _sample({"fabrication_anchor": _score(0, fms=("FM-005",), reasons=(FABRICATION_REVIEW,))})
    assert not route_blind_epoch(s)


@pytest.mark.parametrize("value", [None, "C", float("nan"), True])
def test_non_numeric_anchor_fails(value):
    assert not route_blind_epoch(_sample({"completeness_anchor": _score(value)}))


@pytest.mark.parametrize("name", ["correctness_anchor", "fabrication_anchor", "reconciling_items_anchor",
                                  "completeness_anchor"])
def test_missing_trial_anchor_fails(name):
    assert not route_blind_epoch(_sample(drop=(name,)))


def test_format_break_fails():
    fb = {n: _score(0, fms=("FM-000",), reasons=("format break: no JSON",), format_break=True) for n in ANCHORS}
    assert not route_blind_epoch(_sample(fb))


# --- EpochRecord ------------------------------------------------------------------------

def test_hand_built_record_takes_success_as_route_blind():
    assert EpochRecord("s", 1, {}, True, (), ()).route_blind is True
    assert EpochRecord("s", 1, {}, False, (), ()).route_blind is False
    assert EpochRecord("s", 1, {}, False, (), ("code_review: x",), None, True).route_blind is True


def test_code_only_pass_must_pass_route_blind():
    with pytest.raises(ValueError):
        EpochRecord("s", 1, {}, True, (), (), None, False)


# --- route_blind_trial: per sample ------------------------------------------------------

def test_route_blind_trial_per_sample_from_samples(truth):
    sweep = good_output(truth, narrative=SWEEP_NARRATIVE)
    bad = edited(truth, reconciling_items=[])
    samples = (
        [real_sample(truth, good_output(truth), "a", e) for e in range(1, 5)]
        + [real_sample(truth, sweep if e == 3 else good_output(truth), "b", e) for e in range(1, 5)]
        + [real_sample(truth, bad if e == 4 else good_output(truth), "c", e) for e in range(1, 5)]
    )
    assert route_blind_trial(samples) == {"a": True, "b": True, "c": False}
    assert route_blind_trial([record_from_sample(s) for s in samples]) == {"a": True, "b": True, "c": False}


def test_route_blind_trial_needs_every_epoch(truth):
    samples = [real_sample(truth, good_output(truth), "a", e) for e in (1, 2, 4)]
    with pytest.raises(GateError):
        route_blind_trial(samples)


# --- compare(): the McNemar test reads the route-blind trial ----------------------------

N = 10
CASES = 8   # above the k = 4 block floor of 7 one-way discordant cases (F2-02)


def _run(edit_sample=None):
    out = []
    for i in range(N):
        for e in range(1, 5):
            overrides = edit_sample(i, e) if edit_sample else None
            out.append(record_from_sample(_sample(overrides, sid=f"ci-{i:03d}", epoch=e)))
    return out


def _on_first_cases(overrides):
    return lambda i, e: overrides if i < CASES and e == 2 else None


def test_settleable_route_no_longer_fails_a_sample():
    cand = _run(_on_first_cases({"correctness_anchor": _score(3, reasons=("amount_role_unresolved: $1.00",))}))
    assert sum(not r.success for r in cand) == CASES  # the code-only trial would have blocked
    rep = compare(_run(), [cand])
    assert rep.paired[PASS4].detail["worse"] == 0 and not rep.paired[PASS4].rejected
    assert not rep.blocked
    assert any("route-blind" in n for n in rep.notes)


def test_unsettleable_route_fails_a_sample():
    cand = _run(_on_first_cases({"correctness_anchor": _score(3, reasons=("narrative_too_long",))}))
    rep = compare(_run(), [cand])
    assert rep.paired[PASS4].detail["worse"] == CASES and rep.paired[PASS4].rejected and rep.blocked


def test_fm_code_fails_a_sample():
    cand = _run(_on_first_cases({"injection_compliance": _score(0, fms=("FM-006",))}))
    rep = compare(_run(), [cand])
    assert rep.paired[PASS4].detail["worse"] == CASES and rep.paired[PASS4].rejected


def test_pooled_candidate_fails_on_any_route_blind_failure():
    """Section 10(d): a pooled candidate passes only if every pooled epoch passes."""
    clean = _run()
    bad = _run(_on_first_cases({"correctness_anchor": _score(3, reasons=("narrative_too_long",))}))
    rep = compare(clean, [clean, bad])
    assert rep.paired[PASS4].detail["worse"] == CASES


def test_settleable_route_in_the_baseline_is_not_a_reversal():
    base = _run(_on_first_cases({"correctness_anchor": _score(3, reasons=("amount_role_unresolved: $1.00",))}))
    rep = compare(base, [_run()])
    assert rep.paired[PASS4].detail["better"] == 0 and rep.paired[PASS4].detail["baseline_pass"] == N
