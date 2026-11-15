"""Judge-trust statistics tests (pre-registration section 7, decision D-R19).

Published checks: Shrout and Fleiss (1979) Table 2 (6 targets, 4 judges) gives
ICC(2,1) = 0.29 and ICC(3,1) = 0.71, and Krippendorff's reliability-data example
(4 observers, 12 units, missing values) gives nominal 0.743, ordinal 0.815 and
interval 0.849. The McGraw-Wong interval is checked against its source formula on
a small matrix worked by hand. Every score here is made up; no log is read.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from icfr_agent_evals import judge_trust as jt
from icfr_agent_evals.rubric import LLM_GRADED_SUBDIMENSIONS

G, C = "faithfulness.narrative_grounding", "completeness.narrative_coverage"
assert set(LLM_GRADED_SUBDIMENSIONS) == {G, C}

SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
KRIPPENDORFF = list(zip(
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
))
CFG = {"model": "mock/grader", "reasoning_effort": "high", "response_caching": False}


def _ids(n, prefix="dev-mock"):
    return [f"{prefix}-{i:03d}" for i in range(n)]


def _pass(grounding, coverage=None, prefix="dev-mock"):
    coverage = grounding if coverage is None else coverage
    return {G: dict(zip(_ids(len(grounding), prefix), grounding)),
            C: dict(zip(_ids(len(coverage), prefix), coverage))}


# --- F distribution ---------------------------------------------------------------------------

@pytest.mark.parametrize("d1, d2, table", [(5, 15, 3.576), (15, 5, 6.428), (1, 1, 647.79), (4, 4, 9.605),
                                           (39, 39, 1.895)])
def test_f_ppf_matches_table_values(d1, d2, table):
    assert jt.f_ppf(0.975, d1, d2) == pytest.approx(table, abs=5e-3)


@pytest.mark.parametrize("q, d1, d2", [(0.975, 4, 2.7752), (0.025, 3.3, 39), (0.5, 0.7, 12.5)])
def test_f_ppf_inverts_f_cdf(q, d1, d2):
    assert jt.f_cdf(jt.f_ppf(q, d1, d2), d1, d2) == pytest.approx(q, abs=1e-9)


def test_f_rejects_bad_arguments():
    assert jt.f_cdf(0.0, 2, 3) == 0.0 and jt.f_cdf(math.inf, 2, 3) == 1.0
    for bad in [(0.0, 2, 3), (1.0, 2, 3), (0.5, 0, 3), (0.5, 2, math.inf)]:
        with pytest.raises(ValueError):
            jt.f_ppf(*bad)


# --- ICC ---------------------------------------------------------------------------------------

def test_shrout_fleiss_table_2():
    an = jt.two_way_anova(SHROUT_FLEISS)
    # Shrout and Fleiss (1979) Table 2: BMS 11.24, JMS 32.49, EMS 1.02
    assert float(an["msr"]) == pytest.approx(11.24, abs=5e-3)
    assert float(an["msc"]) == pytest.approx(32.49, abs=5e-3)
    assert float(an["mse"]) == pytest.approx(1.02, abs=5e-3)
    a1, c1 = jt.icc_a1(SHROUT_FLEISS), jt.icc_c1(SHROUT_FLEISS)
    assert round(a1["estimate"], 2) == 0.29 and a1["estimate_exact"] == "184/635"
    assert round(c1["estimate"], 2) == 0.71 and c1["estimate_exact"] == "920/1287"
    # 95% limits on this data as the psych R package reports them: ICC2 0.019 to 0.76, ICC3 0.342 to 0.95
    assert a1["ci_lower"] == pytest.approx(0.019, abs=1e-3) and round(a1["ci_upper"], 2) == 0.76
    assert c1["ci_lower"] == pytest.approx(0.342, abs=1e-3) and round(c1["ci_upper"], 2) == 0.95
    assert a1["defined"] and c1["defined"] and a1["n"] == 6 and a1["k"] == 4


def test_mcgraw_wong_interval_against_its_source_formula():
    m = [[0, 1], [1, 1], [2, 3], [3, 3], [1, 2]]
    # By hand: grand mean 1.7; row means .5, 1, 2.5, 3, 1.5 give SSR = 2 * 4.3 = 8.6, MSR = 2.15;
    # column means 1.4, 2.0 give SSC = 5 * 0.18 = 0.9, MSC = 0.9; SST = 10.1, SSE = 0.6, MSE = 0.15.
    an = jt.two_way_anova(m)
    assert (an["msr"], an["msc"], an["mse"]) == (Fraction(43, 20), Fraction(9, 10), Fraction(3, 20))
    # ICC(A,1) = (2.15 - .15) / (2.15 + .15 + 2 (.9 - .15) / 5) = 2 / 2.6 = 10/13
    r = jt.icc_a1(m)
    assert r["estimate_exact"] == "10/13"
    # McGraw and Wong (1996): a = k p / (n (1 - p)) = 4/3, b = 1 + k p (n - 1) / (n (1 - p)) = 19/3,
    # so a MSC = 1.2, b MSE = 0.95 and v = 2.15^2 / (1.2^2 / 1 + 0.95^2 / 4) = 4.6225 / 1.665625
    v = 4.6225 / 1.665625
    assert r["df_v"] == pytest.approx(v, rel=1e-12)
    # Shrout and Fleiss (1979) give v in another form; it must agree.
    n, k, p, fj = 5, 2, 10 / 13, 0.9 / 0.15
    sf = ((n - 1) * (k - 1) * (k * p * fj + n * (1 + (k - 1) * p) - k * p) ** 2
          / ((n - 1) * k * k * p * p * fj * fj + (n * (1 + (k - 1) * p) - k * p) ** 2))
    assert sf == pytest.approx(v, rel=1e-12)
    f_l, f_u = jt.f_ppf(0.975, 4, v), jt.f_ppf(0.975, v, 4)
    c = 2 * 0.9 + (10 - 2 - 5) * 0.15   # k MSC + (k n - k - n) MSE = 2.25
    lower = 5 * (2.15 - f_l * 0.15) / (f_l * c + 5 * 2.15)
    upper = 5 * (f_u * 2.15 - 0.15) / (c + 5 * f_u * 2.15)
    assert r["ci_lower"] == pytest.approx(lower, rel=1e-12) and r["ci_upper"] == pytest.approx(upper, rel=1e-12)
    assert r["ci_lower"] == pytest.approx(-0.0481, abs=1e-4) and r["ci_upper"] == pytest.approx(0.9729, abs=1e-4)


def test_icc_c1_exact_f_interval_by_hand():
    m = [[0, 1], [1, 1], [2, 3], [3, 3], [1, 2]]
    r = jt.icc_c1(m)
    assert r["estimate_exact"] == "20/23"   # (2.15 - .15) / (2.15 + .15)
    f0 = 2.15 / 0.15
    f_l, f_u = f0 / jt.f_ppf(0.975, 4, 4), f0 * jt.f_ppf(0.975, 4, 4)
    assert r["ci_lower"] == pytest.approx((f_l - 1) / (f_l + 1), rel=1e-12)
    assert r["ci_upper"] == pytest.approx((f_u - 1) / (f_u + 1), rel=1e-12)


def test_icc_undefined_and_degenerate_cases():
    flat = [[2, 2], [2, 2], [2, 2]]
    for fn in (jt.icc_a1, jt.icc_c1):
        r = fn(flat)
        assert r["estimate"] is None and r["ci_lower"] is None and not r["defined"]
    perfect = [[0, 0], [1, 1], [3, 3]]
    a1 = jt.icc_a1(perfect)
    assert (a1["estimate"], a1["ci_lower"], a1["ci_upper"]) == (1.0, 1.0, 1.0)
    assert jt.icc_c1(perfect)["ci_lower"] == 1.0
    offset = [[1, 3], [1, 3], [1, 3]]       # each rater constant, the raters differ
    a1 = jt.icc_a1(offset)
    assert (a1["estimate"], a1["ci_lower"], a1["ci_upper"]) == (0.0, 0.0, 0.0)
    assert not jt.icc_c1(offset)["defined"]
    shifted = [[0, 1], [1, 2], [2, 3]]      # consistency perfect, absolute agreement not
    assert jt.icc_c1(shifted)["estimate"] == 1.0 and jt.icc_a1(shifted)["estimate"] < 1.0


@pytest.mark.parametrize("bad", [[[1, 2]], [[1], [2]], [[1, 2], [1]], [[True, 1], [0, 1]], [[1.0, math.nan], [0, 1]]])
def test_icc_refuses_bad_matrices(bad):
    with pytest.raises(jt.Refused):
        jt.icc_a1(bad)


def test_icc_confidence_checked():
    with pytest.raises(ValueError):
        jt.icc_a1(SHROUT_FLEISS, confidence=1.0)


# --- Paired mean difference ------------------------------------------------------------------

def test_paired_mean_difference():
    r = jt.paired_mean_difference([(3, 2), (2, 2), (1, 0), (0, 0)])
    assert r["mean_exact"] == "1/2" and r["df"] == 3
    sd = math.sqrt(1 / 3)
    half = jt.t_ppf(0.975, 3) * sd / 2
    assert r["ci_lower"] == pytest.approx(0.5 - half) and r["ci_upper"] == pytest.approx(0.5 + half)
    one = jt.paired_mean_difference([(3, 1)])
    assert one["mean"] == 2.0 and one["ci_lower"] is None
    flat = jt.paired_mean_difference([(3, 2), (2, 1)])
    assert flat["ci_lower"] == flat["ci_upper"] == 1.0
    with pytest.raises(jt.Refused):
        jt.paired_mean_difference([])


# --- Krippendorff's alpha ---------------------------------------------------------------------

def test_krippendorff_published_example():
    nominal = jt.krippendorff_alpha(KRIPPENDORFF, metric="nominal")
    ordinal = jt.krippendorff_alpha(KRIPPENDORFF, metric="ordinal")
    interval = jt.krippendorff_alpha(KRIPPENDORFF, metric="interval")
    assert round(nominal["alpha"], 3) == 0.743
    assert round(ordinal["alpha"], 3) == 0.815
    assert round(interval["alpha"], 3) == 0.849
    # unit 12 holds one value, so 11 units and 40 values are pairable
    assert ordinal["pairable_units"] == 11 and ordinal["pairable_values"] == 40 and ordinal["units"] == 12
    assert ordinal["raw_agreement_exact"] == "4/5" and ordinal["unit_agreement_exact"] == "8/11"
    with pytest.raises(ValueError):
        jt.krippendorff_alpha(KRIPPENDORFF, metric="ratio")


def test_alpha_undefined_never_one():
    single = jt.krippendorff_alpha([[2, 2, 2]] * 5)
    assert single["alpha"] is None and single["alpha_exact"] is None and not single["defined"]
    assert single["raw_agreement"] == 1.0 and "zero expected disagreement" in single["note"]
    empty = jt.krippendorff_alpha([[1, None, None], [None, 2, None]])
    assert not empty["defined"] and empty["raw_agreement"] is None and empty["pairable_values"] == 0


def test_alpha_refuses_non_integer_values():
    with pytest.raises(jt.Refused):
        jt.krippendorff_alpha([[1, 2.0]])
    with pytest.raises(jt.Refused):
        jt.krippendorff_alpha([[True, 1]])
    with pytest.raises(jt.Refused):
        jt.krippendorff_alpha([])


def test_bootstrap_alpha_is_seeded_and_recorded():
    a = jt.bootstrap_alpha(KRIPPENDORFF, seed=11, resamples=500)
    assert a == jt.bootstrap_alpha(KRIPPENDORFF, seed=11, resamples=500)
    assert a != jt.bootstrap_alpha(KRIPPENDORFF, seed=12, resamples=500)
    assert a["seed"] == 11 and a["resamples"] == 500
    assert a["defined_resamples"] + a["undefined_resamples"] == 500
    assert a["ci_lower"] <= 0.815 <= a["ci_upper"] and 0.0 <= a["q"] <= 1.0
    with pytest.raises(jt.Refused):
        jt.bootstrap_alpha(KRIPPENDORFF, seed=True)
    with pytest.raises(jt.Refused):
        jt.bootstrap_alpha(KRIPPENDORFF, seed=1, resamples=0)


def test_bootstrap_drops_and_counts_undefined_resamples():
    units = [[2, 2, 2]] * 9 + [[1, 2, 2]]
    b = jt.bootstrap_alpha(units, seed=3, resamples=400)
    assert b["undefined_resamples"] > 0
    assert b["defined_resamples"] == 400 - b["undefined_resamples"]
    none = jt.bootstrap_alpha([[2, 2, 2]] * 4, seed=3, resamples=50)
    assert none["defined_resamples"] == 0 and none["ci_lower"] is None and none["q"] is None


def test_bootstrap_q_compares_exactly():
    one = [[0, 1, 1]]                      # every resample is this unit, so every alpha is the same
    exact = Fraction(jt.krippendorff_alpha(one)["alpha_exact"])
    at = jt.bootstrap_alpha(one, seed=1, resamples=20, threshold=exact)
    assert at["q"] == 0.0                  # equal to the threshold is not below it
    above = jt.bootstrap_alpha(one, seed=1, resamples=20, threshold=exact + Fraction(1, 10**12))
    assert above["q"] == 1.0


# --- Gate helper ------------------------------------------------------------------------------

def test_lower_limit_gate_at_and_below_the_limit():
    at = jt.lower_limit_gate({G: {"ci_lower": 0.60, "estimate": 0.8}, C: {"ci_lower": 0.75, "estimate": 0.9}}, 0.60)
    assert at["passed"] and at["gating_subdimension"] == G and at["gating_estimate"] == 0.8
    below = jt.lower_limit_gate({G: {"ci_lower": 0.70, "estimate": 0.8}, C: {"ci_lower": 0.5999, "estimate": 0.9}},
                                0.60)
    assert not below["passed"] and below["gating_subdimension"] == C and below["gating_lower_limit"] == 0.5999
    undefined = jt.lower_limit_gate({G: {"ci_lower": None, "estimate": None}, C: {"ci_lower": 0.9}}, 0.60)
    assert not undefined["passed"] and undefined["undefined"] == [G] and undefined["gating_lower_limit"] is None
    tie = jt.lower_limit_gate({G: {"ci_lower": 0.7}, C: {"ci_lower": 0.7}}, 0.60)
    assert tie["gating_subdimension"] == G
    with pytest.raises(jt.Refused):
        jt.lower_limit_gate({}, 0.60)


# --- Validity and reliability -----------------------------------------------------------------

HUMAN = [0, 1, 2, 3] * 10
CLOSE = [0, 1, 2, 3] * 9 + [1, 1, 2, 2]
NOISY = [0, 1, 2, 3] * 5 + [3, 0, 1, 1] * 5


def test_validity_gate_passes_and_fails_on_the_smaller_lower_limit():
    ok = jt.validity_gate(_pass(CLOSE), _pass(HUMAN))
    assert ok["passed"] and ok["kappa_gate"] == jt.KAPPA_NOTE
    assert ok["gating_lower_limit"] >= 0.60 and ok["gating_estimate"] is not None
    per = ok["per_subdimension"][G]
    assert per["icc_a1"]["n"] == 40 and per["icc_c1"]["form"] == "ICC(C,1)"
    assert per["distribution"]["human"] == {"0": 10, "1": 10, "2": 10, "3": 10}
    bad = jt.validity_gate(_pass(CLOSE, NOISY), _pass(HUMAN))
    assert not bad["passed"] and bad["gating_subdimension"] == C
    assert bad["gating_lower_limit"] == bad["per_subdimension"][C]["icc_a1"]["ci_lower"] < 0.60


def test_validity_gate_undefined_fails():
    flat = _pass([2] * 6)
    r = jt.validity_gate(flat, flat)
    assert not r["passed"] and set(r["undefined"]) == {G, C}


def test_validity_gate_refuses_binary_checks(monkeypatch):
    monkeypatch.setattr(jt, "BINARY_CHECKS", {"check.mock": G})
    with pytest.raises(jt.Refused, match="binary checks"):
        jt.validity_gate(_pass(CLOSE), _pass(HUMAN))


@pytest.mark.parametrize("mutate", [
    lambda p: p[G].pop("dev-mock-000"),
    lambda p: p[G].__setitem__("dev-mock-000", 4),
    lambda p: p[G].__setitem__("dev-mock-000", True),
    lambda p: p[G].__setitem__("dev-mock-000", 2.0),
    lambda p: p.pop(C),
    lambda p: p.__setitem__("calibration.semantic_agreement", {"dev-mock-000": 1}),
    lambda p: p[G].__setitem__(7, 1),
])
def test_validity_gate_refuses_bad_passes(mutate):
    grader = _pass(CLOSE)
    mutate(grader)
    with pytest.raises(jt.Refused):
        jt.validity_gate(grader, _pass(HUMAN))


def test_subdimension_list_checked():
    with pytest.raises(jt.Refused):
        jt.validity_gate(_pass(CLOSE), _pass(HUMAN), subdimensions=["faithfulness.unknown"])
    with pytest.raises(jt.Refused):
        jt.validity_gate(_pass(CLOSE), _pass(HUMAN), subdimensions=[])
    with pytest.raises(jt.Refused):
        jt.validity_gate(_pass(CLOSE), _pass(HUMAN), subdimensions=[G, G])
    kept = jt.validity_gate({G: _pass(CLOSE)[G]}, {G: _pass(HUMAN)[G]}, subdimensions=[G])
    assert kept["subdimensions"] == [G] and kept["passed"]


def test_reliability_gate_reports_mean_difference():
    grader = [min(3, s + 1) if i % 4 == 0 else s for i, s in enumerate(HUMAN * 3)]
    r = jt.reliability_gate(_pass(grader, prefix="eval-mock"), _pass(HUMAN * 3, prefix="eval-mock"))
    md = r["per_subdimension"][G]["mean_difference"]
    assert md["mean_exact"] == "1/4" and md["ci_lower"] < 0.25 < md["ci_upper"] and md["n"] == 120
    assert r["remedy_order"] == jt.REMEDY_NOTE and r["statistic"].endswith("calibrator")


def test_reliability_exclusion_excludes_below_the_limit():
    r = jt.reliability_exclusion(_pass(CLOSE, NOISY, prefix="eval-mock"), _pass(HUMAN, prefix="eval-mock"))
    assert not r["gate_passed"] and r["kept"] == [G] and r["triggers_grader_change"] is False
    (ex,) = r["excluded"]
    assert ex["subdimension"] == C and ex["ci_lower"] < 0.60 and ex["n"] == 40
    assert "deterministic anchors stand alone" in ex["disclosure"] and "outside the judge remedy order" in r["remedy_order"]
    clean = jt.reliability_exclusion(_pass(CLOSE), _pass(HUMAN))
    assert clean["gate_passed"] and clean["excluded"] == [] and clean["kept"] == [G, C]


def test_reliability_exclusion_refuses_an_undefined_lower_limit():
    flat = _pass([1] * 5)
    with pytest.raises(jt.Refused, match="undefined"):
        jt.reliability_exclusion(flat, flat)


# --- Test-retest ------------------------------------------------------------------------------

AT_080 = [[2, 2, 2], [2, 2, 2], [3, 3, 3], [2, 0, 2], [2, 2, 2]]   # ordinal alpha exactly 4/5


def _passes_from_units(units_g, units_c=None):
    units_c = units_g if units_c is None else units_c
    return [_pass([u[i] for u in units_g], [u[i] for u in units_c]) for i in range(3)]


def test_retest_alpha_at_exactly_080_passes():
    assert jt.krippendorff_alpha(AT_080)["alpha_exact"] == "4/5"
    r = jt.retest_alpha(_passes_from_units(AT_080), bootstrap_seed=7, resamples=200)
    assert r["passed"] and r["gating_alpha"] == 0.8 and r["bootstrap_seed"] == 7
    stat = r["per_subdimension"][G]
    assert stat["bootstrap"]["seed"] == 7 and stat["raw_agreement"] is not None
    assert 0.0 <= stat["bootstrap"]["q"] <= 1.0


def test_retest_alpha_below_and_smaller_value():
    low = [[0, 1, 0], [1, 1, 2], [2, 3, 2], [3, 3, 1], [1, 0, 1]]
    r = jt.retest_alpha(_passes_from_units(AT_080, low), bootstrap_seed=7, resamples=200)
    assert not r["passed"] and r["gating_subdimension"] == C
    assert r["gating_alpha"] == r["per_subdimension"][C]["alpha"] < 0.8
    assert r["per_subdimension"][G]["passed"]


def test_retest_alpha_undefined_fails_beside_raw_agreement():
    single = [[2, 2, 2]] * 5
    r = jt.retest_alpha(_passes_from_units(AT_080, single), bootstrap_seed=7, resamples=50)
    stat = r["per_subdimension"][C]
    assert not r["passed"] and r["undefined"] == [C] and r["gating_alpha"] is None
    assert stat["alpha"] is None and stat["raw_agreement"] == 1.0 and not stat["passed"]


@pytest.mark.parametrize("count", [2, 4])
def test_retest_alpha_refuses_other_pass_counts(count):
    passes = (_passes_from_units(AT_080) * 2)[:count]
    with pytest.raises(jt.Refused, match="exactly 3"):
        jt.retest_alpha(passes, bootstrap_seed=1)


def test_run_test_retest_calls_passes_2_and_3_once_each():
    passes = _passes_from_units(AT_080)
    calls = []

    def score(number):
        calls.append(number)
        return passes[number - 1]

    rec = jt.run_test_retest(passes[0], score, grader_configs=[CFG] * 3, bootstrap_seed=5, resamples=100)
    assert calls == [2, 3] and rec["passes"] == 3 and len(rec["scores"]) == 3
    assert rec["response_caching_disabled"] == [True, True, True] and rec["grader_configs"] == [CFG] * 3
    assert rec["bootstrap_seed"] == 5 and rec["alpha"]["passed"] and rec["pass_1"] == "acceptance-look scoring"


@pytest.mark.parametrize("passes", [2, 4, True])
def test_run_test_retest_refuses_other_pass_counts(passes):
    p = _passes_from_units(AT_080)
    with pytest.raises(jt.Refused, match="exactly 3"):
        jt.run_test_retest(p[0], lambda n: p[n - 1], grader_configs=[CFG] * 3, bootstrap_seed=1, passes=passes)


@pytest.mark.parametrize("configs", [
    [CFG] * 2,
    [CFG] * 4,
    [CFG, CFG, {**CFG, "response_caching": True}],
    [CFG, CFG, {k: v for k, v in CFG.items() if k != "response_caching"}],
    [CFG, CFG, {**CFG, "reasoning_effort": "medium"}],
    [CFG, CFG, "pinned"],
])
def test_run_test_retest_refuses_bad_configs(configs):
    p = _passes_from_units(AT_080)
    with pytest.raises(jt.Refused):
        jt.run_test_retest(p[0], lambda n: p[n - 1], grader_configs=configs, bootstrap_seed=1)


def test_run_test_retest_refuses_an_added_or_dropped_output():
    p = _passes_from_units(AT_080)
    dropped = {G: dict(list(p[1][G].items())[:-1]), C: p[1][C]}
    added = {G: {**p[1][G], "dev-mock-999": 2}, C: p[1][C]}
    for later in (dropped, added):
        with pytest.raises(jt.Refused, match="output ids differ"):
            jt.run_test_retest(p[0], lambda n, later=later: later if n == 2 else p[2],
                               grader_configs=[CFG] * 3, bootstrap_seed=1)


# --- Bias audit -------------------------------------------------------------------------------

def test_perturbation_builders():
    assert jt.reversed_options([0, 1, 2, 3]) == [3, 2, 1, 0]
    a = jt.permuted_options(["L0", "L1", "L2", "L3"], seed=20260607)
    assert a == jt.permuted_options(["L0", "L1", "L2", "L3"], seed=20260607)
    assert sorted(a["order"]) == [0, 1, 2, 3] and a["seed"] == 20260607
    assert a["options"] == [["L0", "L1", "L2", "L3"][i] for i in a["order"]]
    assert isinstance(a["is_identity"], bool) and isinstance(a["is_reversal"], bool)
    m = jt.relabel_map()
    assert m["label_of"] == {0: "D", 1: "C", 2: "B", 3: "A"}
    assert all(m["score_of"][m["label_of"][s]] == s for s in range(4))
    assert jt.unrelabel_pass({G: {"dev-mock-000": "A", "dev-mock-001": "D"}}) == {G: {"dev-mock-000": 3, "dev-mock-001": 0}}
    with pytest.raises(jt.Refused):
        jt.unrelabel_pass({G: {"dev-mock-000": "E"}})
    with pytest.raises(jt.Refused):
        jt.permuted_dimensions(["grounding"], seed=1)
    d = jt.permuted_dimensions([G, C], seed=1)
    assert sorted(d["dimensions"]) == sorted([G, C]) and d["seed"] == 1
    with pytest.raises(jt.Refused):
        jt.reversed_options([0])
    with pytest.raises(jt.Refused):
        jt.permuted_options([0, 1], seed="1")


BASE = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]


def _bumped(indices, delta=1, base=BASE):
    return [s + delta if i in indices else s for i, s in enumerate(base)]


def _audit(perturbed_scores, human=None, retest=None, **kw):
    retest = retest or [_pass(BASE), _pass(_bumped({0, 1, 4})), _pass(BASE)]   # baseline 3/10
    pert = {name: _pass(list(perturbed_scores.get(name, list(retest[0][G].values())))) for name in jt.BIAS_PASSES}
    if kw.get("dimension_order_applies"):
        pert[jt.DIMENSION_ORDER_PERMUTED] = _pass(list(retest[0][G].values()))
    kw.setdefault("dimension_order_applies", False)
    return jt.bias_audit(retest, pert, human or retest[0], permutation_seed=99, **kw)


def test_bias_audit_shift_exactly_010_over_baseline_does_not_fail():
    # retest mean per output is BASE + 1/30 on three outputs; mean shift = mean(pert) - mean(BASE) - 1/10
    pert = _bumped({0, 1, 2, 4, 5})          # mean(pert) - mean(BASE) = 1/2, so shift = 2/5
    r = _audit({jt.OPTION_ORDER_REVERSED: pert})
    s = r["per_subdimension"][G]
    assert s["baseline_exact"] == "3/10" and s["passes"][jt.OPTION_ORDER_REVERSED]["shift_exact"] == "2/5"
    assert s["passes"][jt.OPTION_ORDER_REVERSED]["excess_exact"] == "1/10"
    assert (0.4 - 0.3) > 0.1                 # float arithmetic would flip this value
    assert r["passed"] and not r["reliability_claim_blocked"]


def test_bias_audit_shift_above_margin_fails():
    r = _audit({jt.LABELS_RELABELED: _bumped({0, 1, 2, 4, 5, 6})})   # shift 1/2, excess 1/5
    s = r["per_subdimension"][G]
    assert s["passes"][jt.LABELS_RELABELED]["failed"] and s["failed"]
    assert not r["passed"] and r["reliability_claim_blocked"]


def test_bias_audit_negative_shift_uses_its_magnitude():
    at = _audit({jt.OPTION_ORDER_PERMUTED: _bumped({1, 2, 3}, delta=-1)})      # shift -2/5, excess 1/10
    assert at["per_subdimension"][G]["passes"][jt.OPTION_ORDER_PERMUTED]["excess_exact"] == "1/10"
    assert at["passed"]
    over = _audit({jt.OPTION_ORDER_PERMUTED: _bumped({1, 2, 3, 5}, delta=-1)})  # shift -1/2
    assert not over["passed"]


def test_bias_audit_baseline_is_the_largest_pairwise_difference():
    retest = [_pass(BASE), _pass(_bumped({0, 1})), _pass(_bumped({1, 2, 3}, delta=-1))]
    r = _audit({}, retest=retest)
    s = r["per_subdimension"][G]
    assert s["retest_pass_means"] == ["3/2", "17/10", "6/5"] and s["baseline_exact"] == "1/2"


def test_bias_audit_leniency_at_and_over_025():
    retest = [_pass([0, 1, 2, 3])] * 3
    for human, failed in [([0, 1, 2, 2], False), ([1, 1, 2, 3], False), ([0, 0, 2, 2], True)]:
        r = _audit({name: [0, 1, 2, 3] for name in jt.BIAS_PASSES}, human=_pass(human), retest=retest)
        len_ = r["per_subdimension"][G]["leniency"]
        assert abs(Fraction(len_["mean_exact"])) in (Fraction(1, 4), Fraction(1, 2))
        assert len_["failed"] is failed and r["passed"] is (not failed)
        assert r["per_subdimension"][G]["icc_c1_minus_icc_a1"] is not None


def test_bias_audit_reports_icc_gap_beside_leniency():
    retest = [_pass([0, 1, 2, 3, 1, 2])] * 3
    r = _audit({name: [0, 1, 2, 3, 1, 2] for name in jt.BIAS_PASSES}, human=_pass([0, 0, 1, 2, 1, 1]), retest=retest)
    s = r["per_subdimension"][G]
    gap = Fraction(s["icc_c1_minus_icc_a1_exact"])
    a1 = jt.icc_a1(list(zip([0, 1, 2, 3, 1, 2], [0, 0, 1, 2, 1, 1])))
    c1 = jt.icc_c1(list(zip([0, 1, 2, 3, 1, 2], [0, 0, 1, 2, 1, 1])))
    assert gap == Fraction(c1["estimate_exact"]) - Fraction(a1["estimate_exact"]) and gap > 0
    assert s["leniency"]["mean_exact"] == "2/3" and s["leniency"]["failed"]


def test_bias_audit_dimension_pass_only_where_it_applies():
    applies = _audit({}, dimension_order_applies=True, dimension_seed=4)
    assert applies["passes"][-1] == jt.DIMENSION_ORDER_PERMUTED and applies["dimension_seed"] == 4
    assert _audit({})["passes"] == list(jt.BIAS_PASSES) and _audit({})["permutation_seed"] == 99
    retest = [_pass(BASE)] * 3
    pert = {name: _pass(BASE) for name in jt.BIAS_PASSES}
    with pytest.raises(jt.Refused, match="differ from the required"):
        jt.bias_audit(retest, {**pert, jt.DIMENSION_ORDER_PERMUTED: _pass(BASE)}, retest[0],
                      dimension_order_applies=False, permutation_seed=1)
    with pytest.raises(jt.Refused, match="differ from the required"):
        jt.bias_audit(retest, pert, retest[0], dimension_order_applies=True, permutation_seed=1, dimension_seed=2)
    with pytest.raises(jt.Refused, match="differ from the required"):
        jt.bias_audit(retest, {k: v for k, v in pert.items() if k != jt.LABELS_RELABELED}, retest[0],
                      dimension_order_applies=False, permutation_seed=1)
    with pytest.raises(jt.Refused, match="dimension permutation seed"):
        jt.bias_audit(retest, {**pert, jt.DIMENSION_ORDER_PERMUTED: _pass(BASE)}, retest[0],
                      dimension_order_applies=True, permutation_seed=1)
    with pytest.raises(jt.Refused, match="permutation seed"):
        jt.bias_audit(retest, pert, retest[0], dimension_order_applies=False, permutation_seed=None)


def test_bias_audit_refuses_mismatched_outputs_and_pass_counts():
    retest = [_pass(BASE)] * 3
    pert = {name: _pass(BASE) for name in jt.BIAS_PASSES}
    short = {name: _pass(BASE[:-1]) for name in jt.BIAS_PASSES}
    with pytest.raises(jt.Refused, match="output ids differ"):
        jt.bias_audit(retest, short, retest[0], dimension_order_applies=False, permutation_seed=1)
    with pytest.raises(jt.Refused, match="output ids differ"):
        jt.bias_audit(retest, pert, _pass(BASE[:-1]), dimension_order_applies=False, permutation_seed=1)
    with pytest.raises(jt.Refused, match="exactly 3"):
        jt.bias_audit(retest[:2], pert, retest[0], dimension_order_applies=False, permutation_seed=1)


def test_bias_audit_unperturbed_scores_pass():
    # each perturbed pass repeats pass 1, so its shift is -1/10 against the retest means and the
    # baseline of 3/10 absorbs it
    r = _audit({})
    s = r["per_subdimension"][C]
    assert all(p["shift_exact"] == "-1/10" and not p["failed"] for p in s["passes"].values())
    assert r["passed"] and r["shift_margin"] == "1/10" and r["leniency_limit"] == "1/4"
