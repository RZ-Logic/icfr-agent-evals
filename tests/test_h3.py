"""H3, the cross-model ordering (pre-registration section 2), and its second-look gate
(section 8, decisions D-L and D-R19).

Keyless and offline: every table and every trials.TrialReport here is made up in
memory. Case ids such as eval-mock-001 are invented; nothing reads a log, the dataset
builder or the evaluation split.
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction

import pytest

from icfr_agent_evals import h3, trials
from icfr_agent_evals.h3 import (
    ADVERSARIAL, ANCHORS, COMPLETENESS, CORRECTNESS, EDGE, FABRICATION, FAITHFULNESS, HAIKU_5_5, OPUS_5_5,
    RECONCILING, SONNET_5, SONNET_5_5, WORKERS, CaseScores, Refused,
)

K = 4
T_53 = 2.005745995317864   # t(0.975, 53), the Student t quantile, hand value to 1e-12
CASE_IDS = [f"eval-mock-{i:03d}" for i in range(1, 55)]
KIND = {cid: (EDGE if i < 30 else ADVERSARIAL) for i, cid in enumerate(CASE_IDS)}
LEVEL = {HAIKU_5_5: 1, SONNET_5: 2, SONNET_5_5: 3, OPUS_5_5: 3}
ALL_CONFIRMED = {m: h3.CONFIRMED for m in WORKERS}


def table(score=None, *, cases=CASE_IDS, kinds=None, k=K):
    """score(model, case, anchor, epoch) -> value; default: each model's flat LEVEL."""
    score = score or (lambda m, c, a, e: LEVEL[m])
    kinds = kinds or KIND
    return {m: {c: CaseScores(kinds[c], {a: [score(m, c, a, e) for e in range(k)] for a in ANCHORS})
                for c in cases} for m in WORKERS}


def run(tab, second_look=ALL_CONFIRMED, **kw):
    return h3.evaluate_h3(tab, second_look, **kw)


# --- The verdict ---------------------------------------------------------------------------------

def test_all_twelve_steps_hold():
    r = run(table())
    assert len(r.steps) == 12
    assert {s.anchor for s in r.steps} == set(ANCHORS)
    assert {(s.lower, s.higher) for s in r.steps} == set(h3.STEPS)
    assert r.tested and r.holds is True and r.verdict == "holds"
    assert r.failing_steps == () and r.ordering == frozenset()


def test_one_failing_step_is_named_with_its_anchor():
    def score(m, c, a, e):
        return 1 if (m == OPUS_5_5 and a == COMPLETENESS) else LEVEL[m]
    r = run(table(score))
    assert r.holds is False and r.verdict == "does not hold"
    assert [s.name for s in r.failing_steps] == ["Sonnet 5 <= Opus 5.5 on completeness"]
    assert r.failing_steps[0].difference.mean == -1
    assert r.ordering == frozenset({(COMPLETENESS, SONNET_5, OPUS_5_5)})


def test_the_difference_is_higher_minus_lower_of_epoch_means():
    # Haiku scores 3 on epoch 1 only (mean 0.75); Sonnet 5 scores 1 on every epoch (mean 1).
    def score(m, c, a, e):
        if m == HAIKU_5_5:
            return 3 if e == 0 else 0
        return 1 if m == SONNET_5 else 3
    r = run(table(score))
    step = next(s for s in r.steps if (s.lower, s.higher) == (HAIKU_5_5, SONNET_5) and s.anchor == CORRECTNESS)
    assert step.difference.mean == Fraction(1, 4) and step.holds


def test_a_mean_of_exactly_zero_holds_where_float_sums_go_below_zero():
    rng = random.Random(1)
    vals = [0.1, 0.2, 0.7, 0.3, 0.6, 1.1, 2.3]
    for _ in range(10_000):
        low = [rng.choice(vals) for _ in CASE_IDS]
        high = low[:]
        rng.shuffle(high)
        if sum(h - lo for h, lo in zip(high, low)) / 54 < 0:   # a float pass would fail this step
            break
    else:
        pytest.fail("no float counterexample found")
    lo_of, hi_of = dict(zip(CASE_IDS, low)), dict(zip(CASE_IDS, high))

    def score(m, c, a, e):
        if a == CORRECTNESS and m == HAIKU_5_5:
            return lo_of[c]
        if a == CORRECTNESS and m == SONNET_5:
            return hi_of[c]
        return LEVEL[m] if a != CORRECTNESS else 3
    r = run(table(score))
    step = next(s for s in r.steps if s.key == (CORRECTNESS, HAIKU_5_5, SONNET_5))
    assert step.difference.mean == 0 and step.holds
    assert r.holds is True


def test_zero_difference_everywhere_holds():
    r = run(table(lambda m, c, a, e: 2))
    assert r.holds is True
    assert all(s.difference.mean == 0 and s.difference.sd == 0 for s in r.steps)


def test_paired_t_interval_matches_a_hand_computation():
    # 27 cases with difference 1 and 27 with difference 0 on the Haiku step, correctness.
    def score(m, c, a, e):
        if a == CORRECTNESS and m == HAIKU_5_5:
            return 1 if CASE_IDS.index(c) % 2 else 2
        return 2 if a == CORRECTNESS else LEVEL[m]
    r = run(table(score))
    d = next(s for s in r.steps if s.key == (CORRECTNESS, HAIKU_5_5, SONNET_5)).difference
    sd = math.sqrt(54 * 0.25 / 53)
    se = sd / math.sqrt(54)
    assert d.n == 54 and d.mean == Fraction(1, 2)
    assert d.sd == pytest.approx(sd, abs=1e-12) and d.se == pytest.approx(se, abs=1e-12)
    assert d.t_crit == pytest.approx(T_53, abs=1e-9)
    assert d.ci95[0] == pytest.approx(0.5 - T_53 * se, abs=1e-9)
    assert d.ci95[1] == pytest.approx(0.5 + T_53 * se, abs=1e-9)


def test_the_interval_does_not_enter_the_verdict():
    # A step whose interval reaches below 0 still holds on its mean.
    def score(m, c, a, e):
        if a == FABRICATION and m == SONNET_5_5:
            return 3 if CASE_IDS.index(c) < 28 else 1    # 28 cases +1, 26 cases -1
        return 2 if a == FABRICATION else LEVEL[m]
    r = run(table(score))
    d = next(s for s in r.steps if s.key == (FABRICATION, SONNET_5, SONNET_5_5))
    assert d.difference.ci95[0] < 0 < d.difference.mean and d.holds


def test_paired_difference_checks_its_inputs():
    with pytest.raises(Refused, match="differ in length"):
        h3.paired_difference([Fraction(1)], [Fraction(1), Fraction(2)])
    with pytest.raises(Refused, match="at least 2"):
        h3.paired_difference([Fraction(1)], [Fraction(1)])


# --- Refusals ------------------------------------------------------------------------------------

def test_wrong_case_count_is_refused():
    short = CASE_IDS[:53]
    with pytest.raises(Refused, match="53 cases; H3 needs exactly 54"):
        run(table(cases=short))
    long = CASE_IDS + ["eval-mock-055"]
    with pytest.raises(Refused, match="55 cases"):
        run(table(cases=long, kinds={**KIND, "eval-mock-055": EDGE}))


def test_a_case_set_that_differs_between_models_is_refused():
    tab = table()
    tab[OPUS_5_5] = dict(tab[OPUS_5_5])
    row = tab[OPUS_5_5].pop("eval-mock-054")
    tab[OPUS_5_5]["eval-mock-099"] = row
    with pytest.raises(Refused, match="different case set"):
        run(tab)


def test_wrong_epoch_count_is_refused():
    tab = table()
    tab[SONNET_5]["eval-mock-001"] = CaseScores(EDGE, {a: [2, 2, 2] for a in ANCHORS})
    with pytest.raises(Refused, match="3 epoch scores, need k = 4"):
        run(tab)


def test_the_edge_and_adversarial_split_is_checked():
    kinds = {**KIND, "eval-mock-054": EDGE}       # 31 edge, 23 adversarial
    with pytest.raises(Refused, match="31 edge cases"):
        run(table(kinds=kinds))
    with pytest.raises(Refused, match="case kind 'standard'"):
        run(table(kinds={**KIND, "eval-mock-001": "standard"}))


def test_a_case_kind_that_differs_between_models_is_refused():
    tab = table()
    tab[HAIKU_5_5]["eval-mock-001"] = CaseScores(ADVERSARIAL, tab[HAIKU_5_5]["eval-mock-001"].epochs)
    with pytest.raises(Refused, match="case kinds"):
        run(tab)


def test_models_outside_the_four_workers_are_refused():
    tab = table()
    del tab[HAIKU_5_5]
    with pytest.raises(Refused, match="missing"):
        run(tab)
    tab = table()
    tab["mockllm/model"] = tab[OPUS_5_5]
    with pytest.raises(Refused, match="unexpected"):
        run(tab)


@pytest.mark.parametrize("bad", [None, True, "3", -1, 3.5, float("nan"), float("inf")])
def test_a_bad_score_is_refused(bad):
    tab = table()
    tab[SONNET_5]["eval-mock-002"] = CaseScores(EDGE, {a: [bad, 2, 2, 2] for a in ANCHORS})
    with pytest.raises(Refused):
        run(tab)


def test_missing_or_extra_anchors_are_refused():
    tab = table()
    epochs = dict(tab[SONNET_5]["eval-mock-002"].epochs)
    del epochs[RECONCILING]
    tab[SONNET_5]["eval-mock-002"] = CaseScores(EDGE, epochs)
    with pytest.raises(Refused, match="anchors"):
        run(tab)


def test_unknown_reading_is_refused():
    with pytest.raises(Refused, match="unknown reading"):
        run(table(), reading="route_blind")


# --- Beside the verdict --------------------------------------------------------------------------

def test_edge_and_adversarial_are_reported_apart_and_do_not_enter_the_verdict():
    # Sonnet 5 beats Haiku on edge by more than it loses on adversarial: pooled holds.
    def score(m, c, a, e):
        if a == CORRECTNESS and m == HAIKU_5_5:
            return 0 if KIND[c] == EDGE else 3
        if a == CORRECTNESS and m == SONNET_5:
            return 2
        return LEVEL[m]
    r = run(table(score))
    pooled = next(s for s in r.steps if s.key == (CORRECTNESS, HAIKU_5_5, SONNET_5))
    edge = next(s for s in r.by_subset[EDGE] if s.key == pooled.key)
    adv = next(s for s in r.by_subset[ADVERSARIAL] if s.key == pooled.key)
    assert edge.difference.n == 30 and adv.difference.n == 24
    assert edge.difference.mean == 2 and adv.difference.mean == -1
    assert pooled.difference.mean == Fraction(30 * 2 - 24, 54) and r.holds is True
    assert r.model_means[EDGE][HAIKU_5_5][CORRECTNESS] == 0.0
    assert r.model_means[ADVERSARIAL][HAIKU_5_5][CORRECTNESS] == 3.0
    assert r.model_means[h3.POOLED][HAIKU_5_5][CORRECTNESS] == pytest.approx(72 / 54)
    # the subset t interval uses n - 1 degrees of freedom of that subset
    assert adv.difference.t_crit == pytest.approx(h3.t_ppf(0.975, 23))


def test_faithfulness_minimum_is_taken_per_epoch_and_kept_out_of_the_verdict():
    # Sonnet 5.5 alternates: fabrication 3, 1, 3, 1 and reconciling 1, 3, 1, 3. Each anchor's
    # mean is 2, the per-epoch minimum is 1 every epoch.
    def score(m, c, a, e):
        if m == SONNET_5_5 and a == FABRICATION:
            return 3 if e % 2 == 0 else 1
        if m == SONNET_5_5 and a == RECONCILING:
            return 1 if e % 2 == 0 else 3
        return 2 if m == SONNET_5_5 else LEVEL[m]
    r = run(table(score))
    assert r.model_means[h3.POOLED][SONNET_5_5][FABRICATION] == 2.0
    assert r.model_means[h3.POOLED][SONNET_5_5][FAITHFULNESS] == 1.0
    faith = next(s for s in r.faithfulness_steps if (s.lower, s.higher) == (SONNET_5, SONNET_5_5))
    assert faith.difference.mean == -1 and not faith.holds
    assert len(r.faithfulness_steps) == 3
    assert r.holds is True                           # the four anchors all hold
    assert FAITHFULNESS not in {s.anchor for s in r.steps}


def test_saturation_flag_within_one_standard_error():
    # Opus minus Sonnet 5.5: +1 on 28 cases, -1 on 26: mean 1/27, SE about 0.136: saturated.
    def score(m, c, a, e):
        if m == OPUS_5_5 and a == CORRECTNESS:
            return 3 if CASE_IDS.index(c) < 28 else 1
        return 2 if m in (OPUS_5_5, SONNET_5_5) else LEVEL[m] if m != SONNET_5 else 1
    r = run(table(score))
    sat = r.saturation[CORRECTNESS]
    assert sat["difference"].mean == Fraction(1, 27)
    assert abs(sat["difference"].mean_float) <= sat["difference"].se and sat["within_one_se"]
    # zero variance and zero mean: the two land on each other, within one SE of 0
    assert r.saturation[COMPLETENESS]["within_one_se"]


def test_saturation_flag_off_when_the_gap_exceeds_one_standard_error():
    def score(m, c, a, e):
        if m == OPUS_5_5:
            return 3 if CASE_IDS.index(c) % 2 else 2       # +0.5 on average, SE about 0.069
        return 2 if m == SONNET_5_5 else 1
    r = run(table(score))
    sat = r.saturation[FABRICATION]
    assert sat["difference"].mean == Fraction(1, 2) and not sat["within_one_se"]
    # a constant non-zero gap has SE 0 and is not within it
    r2 = run(table(lambda m, c, a, e: 3 if m == OPUS_5_5 else 2 if m == SONNET_5_5 else 1))
    assert not r2.saturation[CORRECTNESS]["within_one_se"]


def test_generational_delta_is_sonnet_5_5_minus_sonnet_5_with_reasoning_mode():
    modes = {m: "adaptive thinking, reasoning_effort high" for m in WORKERS}
    r = run(table(), reasoning_modes=modes)
    for a in ANCHORS:
        step = next(s for s in r.steps if s.key == (a, SONNET_5, SONNET_5_5))
        assert r.generational[a] == step.difference and r.generational[a].mean == 1
    d = h3.result_dict(r)
    assert d["beside"]["generational"][CORRECTNESS]["reasoning_mode"]["Sonnet 5.5"] == modes[SONNET_5_5]
    text = h3.format_h3(r)
    assert "generational, Sonnet 5.5 minus Sonnet 5" in text and "reasoning_effort high" in text
    assert "not given" in h3.format_h3(run(table()))


# --- Second-look gate ----------------------------------------------------------------------------

def test_review_not_confirmed_makes_h3_not_tested():
    look = {**ALL_CONFIRMED, OPUS_5_5: h3.NOT_CONFIRMED}
    r = run(table(), look)
    assert not r.tested and r.holds is None and r.verdict == "not tested"
    assert r.gate["label"] == h3.NOT_CONFIRMED and "Opus 5.5" in r.gate["reason"]
    assert len(r.steps) == 12                         # figures still computed and reported


def test_review_not_confirmed_wins_over_second_look_empty():
    look = {**ALL_CONFIRMED, OPUS_5_5: h3.NOT_CONFIRMED, HAIKU_5_5: h3.SECOND_LOOK_EMPTY}
    assert h3.second_look_gate(look)["status"] == h3.NOT_TESTED


def test_second_look_empty_is_tested_under_that_label():
    look = {**ALL_CONFIRMED, HAIKU_5_5: h3.SECOND_LOOK_EMPTY}
    r = run(table(), look)
    assert r.tested and r.holds is True
    assert r.gate["label"] == h3.SECOND_LOOK_EMPTY and "Haiku 5.5" in r.gate["reason"]


def test_all_confirmed_is_tested_as_confirmed():
    g = h3.second_look_gate(ALL_CONFIRMED)
    assert g["status"] == h3.TESTED and g["label"] == h3.CONFIRMED


def test_second_look_results_are_checked():
    with pytest.raises(Refused, match="no second look result"):
        h3.second_look_gate({m: h3.CONFIRMED for m in WORKERS[:3]})
    with pytest.raises(Refused, match="unknown second look result"):
        h3.second_look_gate({**ALL_CONFIRMED, SONNET_5: "passed"})
    with pytest.raises(Refused, match="outside the four workers"):
        h3.second_look_gate({**ALL_CONFIRMED, "mockllm/model": h3.CONFIRMED})


# --- Robustness ----------------------------------------------------------------------------------

def _failing(anchor, lower, higher):
    def score(m, c, a, e):
        if a == anchor and m == higher:
            return 0
        return LEVEL[m]
    return score


def test_robustness_labels_when_orderings_differ():
    settled = run(table())
    audit = run(table(_failing(CORRECTNESS, HAIKU_5_5, SONNET_5)), reading=h3.AUDIT_ADJUSTED)
    before = run(table(_failing(FABRICATION, SONNET_5, SONNET_5_5)), reading=h3.BEFORE_D_R9)
    out = h3.robustness_labels(settled, audit, before)
    assert out["labels"] == [h3.NOT_ROBUST_EXTRACTOR, h3.NOT_ROBUST_CITATION]
    assert out["audit_adjusted"]["failing_only_audit_adjusted"] == ["Haiku 5.5 <= Sonnet 5 on correctness"]
    assert out["audit_adjusted"]["verdict_differs"] is True


def test_robustness_compares_the_failing_step_set_not_only_the_verdict():
    # Both readings fail H3, on different steps: the orderings differ.
    settled = run(table(_failing(CORRECTNESS, HAIKU_5_5, SONNET_5)))
    audit = run(table(_failing(COMPLETENESS, HAIKU_5_5, SONNET_5)), reading=h3.AUDIT_ADJUSTED)
    out = h3.robustness_labels(settled, audit)
    assert out["labels"] == [h3.NOT_ROBUST_EXTRACTOR]
    assert out["audit_adjusted"]["verdict_differs"] is False
    assert out["before_d_r9"] == {"compared": False}


def test_no_robustness_label_when_orderings_agree():
    # Different means, same failing set: no label.
    settled = run(table())
    audit = run(table(lambda m, c, a, e: {HAIKU_5_5: 0, SONNET_5: 1, SONNET_5_5: 2, OPUS_5_5: 3}[m]),
                reading=h3.AUDIT_ADJUSTED)
    before = run(table(), reading=h3.BEFORE_D_R9)
    out = h3.robustness_labels(settled, audit, before)
    assert out["labels"] == [] and out["audit_adjusted"]["differs"] is False


def test_robustness_checks_readings():
    settled = run(table())
    with pytest.raises(Refused, match="first reading"):
        h3.robustness_labels(run(table(), reading=h3.AUDIT_ADJUSTED))
    with pytest.raises(Refused, match="audit_adjusted reading must be"):
        h3.robustness_labels(settled, run(table()))
    with pytest.raises(Refused, match="other cases or another k"):
        h3.robustness_labels(settled, run(table(k=2), reading=h3.AUDIT_ADJUSTED, k=2))


# --- Adapter from trials.TrialReport --------------------------------------------------------------

def _epoch(sid, ep, kind, level, **kw):
    anchors = kw.pop("anchors", None) or {a: float(level) for a in ANCHORS}
    return trials.EpochResult(sample_id=sid, epoch=ep, success=True, reasons=(), anchors=anchors,
                              failure_modes=(), case_kind=kind, **kw)


def _report(model, kind=trials.REVIEW_SETTLED, *, k=K, extra=None, epochs=None):
    epochs = epochs or {}
    samples = {}
    for cid in CASE_IDS + ["eval-mock-std-001"]:
        ck = KIND.get(cid, "standard")
        eps = epochs.get((model, cid)) or tuple(_epoch(cid, e, ck, LEVEL[model]) for e in range(1, k + 1))
        samples[cid] = trials.SampleResult(cid, True, "", tuple(eps))
    samples.update(extra or {})
    return trials.TrialReport(kind=kind, k=k, samples=dict(sorted(samples.items())))


def test_table_from_reports_keeps_the_difficulty_bearing_cases():
    reports = {m: _report(m) for m in WORKERS}
    tab, blank = h3.table_from_reports(reports)
    assert sorted(tab[OPUS_5_5]) == CASE_IDS and blank == {m: 0 for m in WORKERS}
    assert tab[HAIKU_5_5]["eval-mock-001"].epochs[CORRECTNESS] == [1.0] * 4
    assert run(tab).holds is True


def test_table_from_reports_reads_an_unscored_anchor_as_zero_and_counts_it():
    sid = "eval-mock-003"
    eps = tuple(_epoch(sid, e, EDGE, 3, anchors={**{a: 3.0 for a in ANCHORS}, COMPLETENESS: None})
                if e == 2 else _epoch(sid, e, EDGE, 3) for e in range(1, 5))
    reports = {m: _report(m, epochs={(OPUS_5_5, sid): eps}) for m in WORKERS}
    tab, blank = h3.table_from_reports(reports)
    assert tab[OPUS_5_5][sid].epochs[COMPLETENESS] == [3.0, 0.0, 3.0, 3.0]
    assert blank[OPUS_5_5] == 1 and blank[SONNET_5] == 0


def test_table_from_reports_before_d_r9_zeroes_fabrication_on_citation_omissions():
    sid = "eval-mock-040"
    eps = (_epoch(sid, 1, ADVERSARIAL, 3, p12_omissions=1), _epoch(sid, 2, ADVERSARIAL, 3, review_found_omissions=1),
           _epoch(sid, 3, ADVERSARIAL, 3), _epoch(sid, 4, ADVERSARIAL, 3))
    reports = {m: _report(m, epochs={(SONNET_5_5, sid): eps}) for m in WORKERS}
    now, _ = h3.table_from_reports(reports)
    old, _ = h3.table_from_reports(reports, before_d_r9=True)
    assert now[SONNET_5_5][sid].epochs[FABRICATION] == [3.0] * 4
    assert old[SONNET_5_5][sid].epochs[FABRICATION] == [0.0, 0.0, 3.0, 3.0]
    assert old[SONNET_5_5][sid].epochs[RECONCILING] == [3.0] * 4
    assert old[SONNET_5_5][sid].epochs[CORRECTNESS] == [3.0] * 4


def test_table_from_reports_refusals():
    with pytest.raises(Refused, match="one kind"):
        h3.table_from_reports({m: _report(m, trials.AUDIT_ADJUSTED if m == OPUS_5_5 else trials.REVIEW_SETTLED)
                               for m in WORKERS})
    with pytest.raises(Refused, match="one kind"):
        h3.table_from_reports({m: _report(m, trials.ROUTE_BLIND) for m in WORKERS})
    with pytest.raises(Refused, match="k = 2"):
        h3.table_from_reports({m: _report(m, k=2) for m in WORKERS})
    nokind = {"eval-mock-x": trials.SampleResult("eval-mock-x", True, "",
                                                 (_epoch("eval-mock-x", 1, None, 3),))}
    with pytest.raises(Refused, match="no case kind"):
        h3.table_from_reports({m: _report(m, extra=nokind) for m in WORKERS})


def test_a_missing_epoch_in_a_report_is_refused_by_the_table_check():
    sid = "eval-mock-010"
    eps = tuple(_epoch(sid, e, EDGE, 2) for e in (1, 2, 4))
    reports = {m: _report(m, epochs={(SONNET_5, sid): eps}) for m in WORKERS}
    tab, _ = h3.table_from_reports(reports)
    with pytest.raises(Refused, match="3 epoch scores"):
        run(tab)


def test_h3_from_reports_gives_all_three_readings_and_the_labels():
    sid = "eval-mock-045"
    # Under the rule before D-R9, Opus loses fabrication points on every case: the
    # Sonnet 5 <= Opus 5.5 fabrication step fails on that reading only.
    eps = {(OPUS_5_5, c): tuple(_epoch(c, e, KIND[c], 3, p12_omissions=1) for e in range(1, 5)) for c in CASE_IDS}
    settled = {m: _report(m, epochs=eps) for m in WORKERS}
    audit = {m: _report(m, trials.AUDIT_ADJUSTED) for m in WORKERS}
    out = h3.h3_from_reports(settled, audit, ALL_CONFIRMED)
    assert out[h3.REVIEW_SETTLED].holds is True
    assert out[h3.AUDIT_ADJUSTED].holds is True
    assert [s.name for s in out[h3.BEFORE_D_R9].failing_steps] == ["Sonnet 5 <= Opus 5.5 on fabrication"]
    assert out["robustness"]["labels"] == [h3.NOT_ROBUST_CITATION]
    assert sid in out[h3.REVIEW_SETTLED].cases
    without_audit = h3.h3_from_reports(settled, None, ALL_CONFIRMED)
    assert h3.AUDIT_ADJUSTED not in without_audit
    assert without_audit["robustness"]["audit_adjusted"] == {"compared": False}


# --- Output and CLI ------------------------------------------------------------------------------

def _json_table(tab):
    return {m: {c: {"case_kind": row.case_kind, "epochs": {a: list(v) for a, v in row.epochs.items()}}
                for c, row in cases.items()} for m, cases in tab.items()}


def test_result_dict_is_json_ready():
    r = run(table(_failing(CORRECTNESS, HAIKU_5_5, SONNET_5)))
    d = json.loads(json.dumps(h3.result_dict(r)))
    assert d["verdict"] == "does not hold" and d["n_cases"] == 54
    assert d["failing_steps"] == ["Haiku 5.5 <= Sonnet 5 on correctness"]
    assert d["steps"][0]["mean_exact"] == "-1"
    assert set(d["beside"]["by_subset"]) == {EDGE, ADVERSARIAL}


def test_cli_reports_and_refuses(tmp_path, capsys):
    settled, audit, look = tmp_path / "s.json", tmp_path / "a.json", tmp_path / "l.json"
    settled.write_text(json.dumps(_json_table(table())), encoding="utf-8")
    audit.write_text(json.dumps(_json_table(table(_failing(COMPLETENESS, SONNET_5, OPUS_5_5)))), encoding="utf-8")
    look.write_text(json.dumps(ALL_CONFIRMED), encoding="utf-8")
    out = tmp_path / "out.json"
    code = h3.main(["--settled", str(settled), "--audit", str(audit), "--second-look", str(look),
                    "--json", str(out)])
    text = capsys.readouterr().out
    assert code == h3.EXIT_OK
    assert "H3 on review-settled" in text and h3.NOT_ROBUST_EXTRACTOR in text
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[h3.REVIEW_SETTLED]["holds"] is True
    assert payload["robustness"]["labels"] == [h3.NOT_ROBUST_EXTRACTOR]
    short = tmp_path / "short.json"
    short.write_text(json.dumps(_json_table(table(cases=CASE_IDS[:50]))), encoding="utf-8")
    assert h3.main(["--settled", str(short), "--second-look", str(look)]) == h3.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
    assert h3.main(["--settled", str(tmp_path / "none.json"), "--second-look", str(look)]) == h3.EXIT_REFUSED
