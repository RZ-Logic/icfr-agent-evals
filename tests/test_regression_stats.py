"""CI regression gate tests (pre-registration section 10 (a) to (d)).

Written from the regression.py contract and the pre-registration alone, as an
independent check on the implementation. Reference values were computed once
with scipy 1.18.1 in a throwaway venv and hardcoded here; the package itself
imports no scipy. Each table names the scipy call that produced it. Holm
families are worked by hand in the comments.

Covers: the Student t CDF, the one-sided paired t-test of (a), the exact
one-sided McNemar test of (a) and (b) with the (b) floor (fewer than 6
discordant cases cannot reject at 0.025), Holm step-down at family-wise alpha
0.05 with m = 6, and compare(): pairing by sample id, epoch-count checks,
unscored dimensions, re-run pooling under (d) and the regression sign-off
check of (c) with the R1/R2 review-volume ladder.
"""

from __future__ import annotations

import pytest

from icfr_agent_evals.regression import (
    CODE_ROUTE,
    DIMENSIONS,
    FABRICATION_ALPHA,
    FABRICATION_CODES,
    FAMILY_ALPHA,
    PASS4,
    SIGNOFF_CODES,
    SWEEP_ROUTE,
    EpochRecord,
    SignoffCase,
    compare,
    holm,
    mcnemar_one_sided,
    paired_t_one_sided,
    t_cdf,
)
from icfr_agent_evals.reliability import GateError

REL = 1e-9


# --- Contract constants ---------------------------------------------------------------

def test_contract_constants():
    assert DIMENSIONS == ("correctness", "faithfulness", "calibration", "completeness", "robustness")
    assert PASS4 == "content_pass4"
    assert FAMILY_ALPHA == 0.05
    assert FABRICATION_ALPHA == 0.025
    assert FABRICATION_CODES == ("FM-005", "FM-015")
    assert SIGNOFF_CODES == ("FM-000", "FM-001", "FM-002", "FM-003", "FM-004", "FM-005", "FM-006", "FM-015", "FM-016")
    assert SWEEP_ROUTE == "narrative_sweep"
    assert CODE_ROUTE == "code_review"


# --- t_cdf ----------------------------------------------------------------------------

# (t, df, scipy.stats.t.cdf(t, df))
T_CDF_GRID = [
    (-50.0, 1, 0.006365349100972797),
    (-10.0, 1, 0.03172551743055357),
    (-3.0, 1, 0.10241638234956672),
    (-1.5, 1, 0.18716704181099886),
    (-0.5, 1, 0.35241638234956674),
    (0.0, 1, 0.5),
    (0.5, 1, 0.6475836176504333),
    (1.5, 1, 0.8128329581890011),
    (3.0, 1, 0.8975836176504333),
    (10.0, 1, 0.9682744825694465),
    (50.0, 1, 0.9936346508990272),
    (-1000.0, 1, 0.0003183097800805589),
    (1000.0, 1, 0.9996816902199195),
    (-50.0, 3, 8.808576020635984e-06),
    (-10.0, 3, 0.0010641995292070751),
    (-3.0, 3, 0.02883444281121865),
    (-1.5, 3, 0.11529193262241147),
    (-0.5, 3, 0.3257239824240755),
    (0.0, 3, 0.5),
    (0.5, 3, 0.6742760175759245),
    (1.5, 3, 0.8847080673775886),
    (3.0, 3, 0.9711655571887814),
    (10.0, 3, 0.9989358004707929),
    (50.0, 3, 0.9999911914239794),
    (-1000.0, 3, 1.102653821288296e-09),
    (1000.0, 3, 0.9999999988973461),
    (-50.0, 199, 6.289145464669021e-115),
    (-10.0, 199, 1.233593630259139e-19),
    (-3.0, 199, 0.001522413362553087),
    (-1.5, 199, 0.06759961872077885),
    (-0.5, 199, 0.30881375616766604),
    (0.0, 199, 0.5),
    (0.5, 199, 0.691186243832334),
    (1.5, 199, 0.9324003812792212),
    (3.0, 199, 0.9984775866374469),
    (10.0, 199, 1.0),
    (50.0, 199, 1.0),
    (-1000.0, 199, 0.0),
    (1000.0, 199, 1.0),
    (-10.0, 2.5, 0.0022207478836537117),
    (-1.5, 2.5, 0.12391822654314809),
    (0.5, 2.5, 0.6711510400651426),
    (3.0, 2.5, 0.9637119522254841),
    (-10.0, 30, 2.287625704114809e-11),
    (-3.0, 30, 0.0026949820328259727),
    (-1.5, 30, 0.072032964564323),
    (0.5, 30, 0.6896384975574363),
    (10.0, 30, 0.9999999999771237),
]


@pytest.mark.parametrize(("t", "df", "expected"), T_CDF_GRID)
def test_t_cdf_matches_scipy(t, df, expected):
    assert t_cdf(t, df) == pytest.approx(expected, rel=REL, abs=1e-12)


# lower-tail p-values smaller than 1e-12, compared relatively (scipy.stats.t.cdf)
T_CDF_TAIL = [
    (-1000.0, 3, 1.102653821288296e-09),
    (-1000.0, 2.5, 2.274746394830745e-08),
    (-50.0, 3, 8.808576020635984e-06),
    (-10.0, 30, 2.287625704114809e-11),
    (-10.0, 199, 1.233593630259139e-19),
    (-50.0, 30, 9.357708829611355e-31),
]


@pytest.mark.parametrize(("t", "df", "expected"), T_CDF_TAIL)
def test_t_cdf_lower_tail_keeps_relative_precision(t, df, expected):
    assert t_cdf(t, df) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("df", [1, 3, 2.5, 30, 199])
@pytest.mark.parametrize("t", [0.1, 0.7, 2.0, 4.5, 12.0])
def test_t_cdf_symmetric_and_bounded(t, df):
    lo, hi = t_cdf(-t, df), t_cdf(t, df)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert lo + hi == pytest.approx(1.0, rel=REL)


def test_t_cdf_monotone():
    grid = [-40.0, -8.0, -2.0, -0.25, 0.0, 0.25, 2.0, 8.0, 40.0]
    for df in (1, 3, 199):
        values = [t_cdf(t, df) for t in grid]
        assert values == sorted(values)


# --- paired_t_one_sided ---------------------------------------------------------------

# (baseline, candidate, p, t, mean_diff, sd_diff): scipy.stats.ttest_rel(candidate, baseline, alternative='less')
PAIRED_CASES = {
    "mixed_drop": (
        [3, 3, 3, 2.5, 2, 3, 1.75, 3],
        [2.5, 3, 2.25, 2.5, 1, 2.75, 1.75, 2],
        0.012807133590546103, -2.8243130467507513, -0.4375, 0.43813729094233206,
    ),
    "improvement": (
        [1, 2, 1.5, 2, 0.5, 1],
        [2, 2.5, 1.75, 3, 1.5, 1.25],
        0.9962806719068242, 4.338609156373123, 0.6666666666666666, 0.3763863263545405,
    ),
    "small_n2": (
        [3, 2],
        [2.5, 2.25],
        0.39758361765043326, -0.3333333333333333, -0.125, 0.5303300858899106,
    ),
    "noisy_null": (
        [2.75, 3, 2.5, 3, 2.25, 3, 3, 2.75, 2.5, 3],
        [3, 2.75, 2.5, 3, 2.5, 2.75, 3, 3, 2.25, 3],
        0.5, 0.0, 0.0, 0.2041241452319315,
    ),
}


@pytest.mark.parametrize("name", sorted(PAIRED_CASES))
def test_paired_t_matches_scipy(name):
    baseline, candidate, p_ref, t_ref, mean_ref, sd_ref = PAIRED_CASES[name]
    p, detail = paired_t_one_sided(baseline, candidate)
    assert p == pytest.approx(p_ref, rel=REL, abs=1e-15)
    # detail keys are named by example in the contract; check the ones present
    if "t" in detail:
        assert detail["t"] == pytest.approx(t_ref, rel=REL, abs=1e-12)
    if "df" in detail:
        assert detail["df"] == len(baseline) - 1
    if "mean_diff" in detail:
        assert detail["mean_diff"] == pytest.approx(mean_ref, rel=REL, abs=1e-15)
    if "sd_diff" in detail:
        assert detail["sd_diff"] == pytest.approx(sd_ref, rel=REL)


def test_paired_t_direction_improvement_gives_p_near_one():
    p, _ = paired_t_one_sided(*PAIRED_CASES["improvement"][:2])
    assert p > 0.99
    p_rev, _ = paired_t_one_sided(PAIRED_CASES["improvement"][1], PAIRED_CASES["improvement"][0])
    assert p_rev < 0.01
    assert p + p_rev == pytest.approx(1.0, rel=REL)


@pytest.mark.parametrize(("baseline", "candidate"), [([], []), ([3.0], [2.0])])
def test_paired_t_needs_two_pairs(baseline, candidate):
    with pytest.raises(ValueError):
        paired_t_one_sided(baseline, candidate)


def test_paired_t_constant_drop_blocks():
    # sd == 0 and mean(d) < 0: p = 0.0, a constant drop blocks
    p, _ = paired_t_one_sided([3.0, 2.0, 3.0, 1.0], [2.75, 1.75, 2.75, 0.75])
    assert p == 0.0


@pytest.mark.parametrize("delta", [0.0, 0.25])
def test_paired_t_constant_no_drop_passes(delta):
    # sd == 0 and mean(d) >= 0: p = 1.0
    base = [3.0, 2.0, 2.5, 1.0]
    p, _ = paired_t_one_sided(base, [b + delta for b in base])
    assert p == 1.0


# --- mcnemar_one_sided ----------------------------------------------------------------

# (worse, better, scipy.stats.binomtest(worse, worse + better, 0.5, alternative='greater').pvalue)
MCNEMAR_CASES = [
    (5, 0, 0.03125),
    (6, 0, 0.015625),
    (7, 0, 0.0078125),
    (8, 0, 0.00390625),
    (3, 1, 0.3125),
    (10, 4, 0.08978271484375),
    (1, 1, 0.75),
    (0, 3, 1.0),
    (12, 3, 0.017578125),
    (6, 1, 0.0625),
    (0, 0, 1.0),
]


@pytest.mark.parametrize(("worse", "better", "expected"), MCNEMAR_CASES)
def test_mcnemar_matches_scipy(worse, better, expected):
    assert mcnemar_one_sided(worse, better) == pytest.approx(expected, rel=REL)


def test_mcnemar_fabrication_floor():
    # section 10(b): fewer than 6 discordant cases cannot reject at 0.025
    assert mcnemar_one_sided(5, 0) == pytest.approx(0.03125, rel=REL)
    assert mcnemar_one_sided(5, 0) > FABRICATION_ALPHA
    assert mcnemar_one_sided(6, 0) == pytest.approx(0.015625, rel=REL)
    assert mcnemar_one_sided(6, 0) <= FABRICATION_ALPHA


# --- holm -----------------------------------------------------------------------------

def _check_holm(result, expected):
    assert set(result) == set(expected)
    for name, (adj, rejected) in expected.items():
        got_adj, got_rej = result[name]
        assert got_adj == pytest.approx(adj, rel=REL), name
        assert got_rej is rejected, name


def test_holm_family_of_six():
    # sorted: d .005, a .009, f .02, c .03, b .04, e .5; thresholds .05/6, .05/5, .05/4 ...
    # d .005 <= .00833 reject; a .009 <= .01 reject; f .02 > .0125 stop.
    # adjusted: d 6*.005=.03; a max(.03, 5*.009=.045)=.045; f max(.045, 4*.02=.08)=.08;
    # c max(.08, 3*.03=.09)=.09; b max(.09, 2*.04=.08)=.09; e max(.09, .5)=.5
    pvals = {"a": 0.009, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.5, "f": 0.02}
    _check_holm(holm(pvals), {
        "d": (0.03, True), "a": (0.045, True), "f": (0.08, False),
        "c": (0.09, False), "b": (0.09, False), "e": (0.5, False),
    })


def test_holm_alpha_argument():
    # same family at alpha 0.10: thresholds .0167, .02, .025, .0333, .05, .1; e .5 fails
    pvals = {"a": 0.009, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.5, "f": 0.02}
    _check_holm(holm(pvals, alpha=0.10), {
        "d": (0.03, True), "a": (0.045, True), "f": (0.08, True),
        "c": (0.09, True), "b": (0.09, True), "e": (0.5, False),
    })


def test_holm_stops_at_first_non_rejection():
    # m = 3, thresholds .0167, .025, .05. p .02 fails first, so q .021 (<= .025) and
    # r .03 (<= .05) are not rejected although each clears its own threshold.
    # adjusted: .06, max(.06, .042)=.06, max(.06, .03)=.06
    _check_holm(holm({"p": 0.02, "q": 0.021, "r": 0.03}), {
        "p": (0.06, False), "q": (0.06, False), "r": (0.06, False),
    })


def test_holm_ties():
    # x, y tie at .01: 3*.01=.03 and max(.03, 2*.01)=.03, both rejected; z .04 <= .05
    _check_holm(holm({"y": 0.01, "x": 0.01, "z": 0.04}), {
        "x": (0.03, True), "y": (0.03, True), "z": (0.04, True),
    })
    # tie that fails the first threshold: .03 > .025, both kept, adjusted .06
    _check_holm(holm({"y": 0.03, "x": 0.03}), {"x": (0.06, False), "y": (0.06, False)})


def test_holm_adjusted_p_capped_at_one():
    _check_holm(holm({"a": 0.4, "b": 0.6, "c": 0.9}), {
        "a": (1.0, False), "b": (1.0, False), "c": (1.0, False),
    })
    _check_holm(holm({"a": 0.001, "b": 0.5, "c": 0.7}), {
        "a": (0.003, True), "b": (1.0, False), "c": (1.0, False),
    })
    for adj, _ in holm({"a": 0.4, "b": 0.6, "c": 0.9}).values():
        assert adj <= 1.0


def test_holm_single_test():
    _check_holm(holm({"a": 0.04}), {"a": (0.04, True)})
    _check_holm(holm({"a": 0.06}), {"a": (0.06, False)})


# --- compare(): fixtures --------------------------------------------------------------

FULL = {d: 3.0 for d in DIMENSIONS}
ANCHORS_ONLY = {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}


def sid(i: int) -> str:
    return f"s{i:03d}"


def make_run(n=20, k=4, edit=None, dims=None):
    """n samples x epochs 1..k, all clean; edit(i, epoch, fields) mutates one record's fields."""
    out = []
    for i in range(n):
        for e in range(1, k + 1):
            fields = {
                "dims": dict(FULL if dims is None else dims),
                "success": True,
                "failure_modes": (),
                "review_routes": (),
            }
            if edit is not None:
                edit(i, e, fields)
            out.append(EpochRecord(sample_id=sid(i), epoch=e, dims=fields["dims"], success=fields["success"],
                                   failure_modes=tuple(fields["failure_modes"]),
                                   review_routes=tuple(fields["review_routes"])))
    return out


def correctness_drop(drops):
    """Sample i scores 2 instead of 3 on its first drops[i] epochs."""
    def edit(i, e, f):
        if i < len(drops) and e <= drops[i]:
            f["dims"]["correctness"] = 2.0
    return edit


def fire(cases, epochs, code):
    """Fire an FM code on the given cases and epochs; the trial fails."""
    def edit(i, e, f):
        if i in cases and e in epochs:
            f["failure_modes"] = (*f["failure_modes"], code)
            f["success"] = False
    return edit


def route(cases, epochs, name):
    """Route the given epochs to review without an FM code; the trial fails."""
    def edit(i, e, f):
        if i in cases and e in epochs:
            f["review_routes"] = (*f["review_routes"], name)
            f["success"] = False
    return edit


def chain(*edits):
    def edit(i, e, f):
        for one in edits:
            one(i, e, f)
    return edit


def _noisy(i, e, f):
    for j, d in enumerate(DIMENSIONS):
        f["dims"][d] = float((i + e + j) % 4)
    f["success"] = (i + e) % 3 != 0
    if i % 5 == 0 and e == 2:
        f["failure_modes"] = ("FM-001",)
    if i % 7 == 0 and e == 4:
        f["review_routes"] = (SWEEP_ROUTE,)


def _cases(report, trigger=None):
    return {c for c in report.signoff if trigger is None or c.trigger == trigger}


def _assert_six_paired(report):
    assert set(report.paired) == {*DIMENSIONS, PASS4}
    assert set(report.fabrication) == set(FABRICATION_CODES)


# --- compare(): section 10(a) ---------------------------------------------------------

def test_identical_runs_never_block():
    base = make_run(edit=_noisy)
    report = compare(base, [list(base)])
    _assert_six_paired(report)
    assert not report.blocked
    assert report.signoff == ()
    for name, test in report.paired.items():
        assert test.p == 1.0, name
        assert not test.rejected
        assert test.alpha == FAMILY_ALPHA
        assert test.p_holm is not None
    for code, test in report.fabrication.items():
        assert test.p == 1.0, code
        assert not test.rejected
        assert test.p_holm is None
        assert test.alpha == FABRICATION_ALPHA


def test_large_correctness_drop_blocks_via_holm():
    # ttest_rel(c, b, alternative='less') with d_i = -x/4, x cycling 1, 2, 3 over 30 samples
    drops = [1, 2, 3] * 10
    report = compare(make_run(n=30), [make_run(n=30, edit=correctness_drop(drops))])
    test = report.paired["correctness"]
    assert test.p == pytest.approx(4.397574795532201e-14, rel=1e-6)
    assert test.rejected and test.p_holm <= FAMILY_ALPHA
    assert test.n == 30
    assert report.blocked
    for name in ("faithfulness", "calibration", "completeness", "robustness", PASS4):
        assert not report.paired[name].rejected


def test_pass4_drop_blocks_via_mcnemar():
    # 8 samples go from all-4-pass to one failed epoch: worse 8, better 0, p 2^-8;
    # Holm with five p = 1 tests: 6 * 0.00390625 = 0.0234375
    report = compare(make_run(n=12), [make_run(n=12, edit=fire(set(range(8)), {3}, "FM-007"))])
    test = report.paired[PASS4]
    assert test.p == pytest.approx(0.00390625, rel=REL)
    assert test.p_holm == pytest.approx(0.0234375, rel=REL)
    assert test.rejected and report.blocked
    assert test.n == 12
    if "worse" in test.detail:
        assert test.detail["worse"] == 8 and test.detail["better"] == 0


def test_pass4_counts_better_cases():
    # baseline fails on samples 0-1, candidate fixes them and breaks 2-7: worse 6, better 2
    # binomtest(6, 8, 0.5, alternative='greater').pvalue = 0.14453125
    report = compare(make_run(edit=fire({0, 1}, {1}, "FM-007")), [make_run(edit=fire({2, 3, 4, 5, 6, 7}, {1}, "FM-007"))])
    assert report.paired[PASS4].p == pytest.approx(0.14453125, rel=REL)
    assert not report.paired[PASS4].rejected


def test_single_dimension_drop_significant_unadjusted_but_not_under_holm():
    # 4 of 20 samples drop one correctness epoch: d = -0.25 x4, 0 x16
    # ttest_rel(c, b, alternative='less').pvalue = 0.021043143355250874 (< 0.05),
    # Holm m = 6 with five p = 1 tests: 6p = 0.12625885... > 0.05
    report = compare(make_run(), [make_run(edit=correctness_drop([1, 1, 1, 1]))])
    test = report.paired["correctness"]
    assert test.p == pytest.approx(0.021043143355250874, rel=REL)
    assert test.p < FAMILY_ALPHA
    assert test.p_holm == pytest.approx(6 * 0.021043143355250874, rel=REL)
    assert not test.rejected
    assert not report.blocked


def test_unscored_dimensions_count_in_m_six():
    # calibration and robustness absent from both runs: p = 1, not rejected, noted, still in m = 6.
    # correctness: drops 1 x8, 2 x2, 0 x10 -> ttest_rel p = 0.00043659696248341234
    # pass^4: 5 samples break, binomtest(5, 5, .5, 'greater') = 0.03125
    # Holm m = 6: correctness 6p = 0.0026195818; pass4 max(prev, 5 * 0.03125) = 0.15625; rest 1.0
    # (with m = 4 these would be 4p and 3 * 0.03125)
    p_corr = 0.00043659696248341234
    drops = [1] * 8 + [2] * 2
    cand = make_run(dims=ANCHORS_ONLY, edit=chain(correctness_drop(drops), fire(set(range(10, 15)), {4}, "FM-007")))
    report = compare(make_run(dims=ANCHORS_ONLY), [cand])
    _assert_six_paired(report)
    assert report.paired["correctness"].p == pytest.approx(p_corr, rel=REL)
    assert report.paired["correctness"].p_holm == pytest.approx(6 * p_corr, rel=REL)
    assert report.paired["correctness"].rejected
    assert report.paired[PASS4].p == pytest.approx(0.03125, rel=REL)
    assert report.paired[PASS4].p_holm == pytest.approx(0.15625, rel=REL)
    assert not report.paired[PASS4].rejected
    for name in ("calibration", "robustness"):
        assert report.paired[name].p == 1.0
        assert not report.paired[name].rejected
        assert report.paired[name].p_holm == pytest.approx(1.0)
        assert any(name in note and "not scored" in note.lower() for note in report.notes), report.notes
    assert report.blocked


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_dimension_in_one_run_only_is_gate_error(side):
    with_cal = make_run(dims={**ANCHORS_ONLY, "calibration": 3.0})
    without = make_run(dims=ANCHORS_ONLY)
    base, cand = (with_cal, without) if side == "baseline" else (without, with_cal)
    with pytest.raises(GateError):
        compare(base, [cand])


def test_sample_id_mismatch_is_gate_error():
    base = make_run(n=10)
    with pytest.raises(GateError):
        compare(base, [make_run(n=9)])  # candidate misses s009
    with pytest.raises(GateError):
        compare(base, [make_run(n=11)])  # candidate adds s010
    renamed = [EpochRecord(sample_id="x" + r.sample_id, epoch=r.epoch, dims=r.dims, success=r.success,
                           failure_modes=r.failure_modes, review_routes=r.review_routes) for r in base]
    with pytest.raises(GateError):
        compare(base, [renamed])


def _drop_record(run, sample, epoch):
    return [r for r in run if not (r.sample_id == sample and r.epoch == epoch)]


def _relabel(run, sample, old, new):
    out = []
    for r in run:
        if r.sample_id == sample and r.epoch == old:
            r = EpochRecord(sample_id=r.sample_id, epoch=new, dims=r.dims, success=r.success,
                            failure_modes=r.failure_modes, review_routes=r.review_routes)
        out.append(r)
    return out


@pytest.mark.parametrize("which", ["baseline", "candidate"])
@pytest.mark.parametrize("damage", ["missing_epoch", "fifth_epoch", "epoch_5_for_4", "duplicate_epoch", "epoch_zero"])
def test_wrong_epoch_counts_are_gate_errors(which, damage):
    good = make_run(n=6)
    if damage == "missing_epoch":
        bad = _drop_record(good, sid(2), 4)
    elif damage == "fifth_epoch":
        bad = good + [EpochRecord(sample_id=sid(2), epoch=5, dims=dict(FULL), success=True, failure_modes=(), review_routes=())]
    elif damage == "epoch_5_for_4":
        bad = _relabel(good, sid(2), 4, 5)
    elif damage == "duplicate_epoch":
        bad = _relabel(good, sid(2), 4, 3)
    else:
        bad = _relabel(good, sid(2), 4, 0)
    base, cand = (bad, make_run(n=6)) if which == "baseline" else (make_run(n=6), bad)
    with pytest.raises(GateError):
        compare(base, [cand])


def test_second_candidate_run_with_wrong_epochs_is_gate_error():
    with pytest.raises(GateError):
        compare(make_run(n=6), [make_run(n=6), make_run(n=6, k=3)])


# --- compare(): section 10(d) pooling ---------------------------------------------------

def test_pooled_epoch_means_use_all_eight_epochs():
    # baseline correctness 3, 2, 3, 2 (mean 2.5); candidate run 1 all 3, run 2 all 2 (pooled mean 2.5)
    def base_edit(i, e, f):
        f["dims"]["correctness"] = 3.0 if e % 2 else 2.0

    def run2_edit(i, e, f):
        f["dims"]["correctness"] = 2.0

    report = compare(make_run(edit=base_edit), [make_run(), make_run(edit=run2_edit)])
    assert report.paired["correctness"].p == 1.0
    assert not report.blocked


def test_pooled_pass4_needs_all_eight_epochs():
    # run 1 clean, run 2 fails epoch 3 on 7 of 10 samples: worse 7, p 2^-7 = 0.0078125,
    # Holm 6 * 0.0078125 = 0.046875 <= 0.05
    report = compare(make_run(n=10), [make_run(n=10), make_run(n=10, edit=fire(set(range(7)), {3}, "FM-007"))])
    assert report.paired[PASS4].p == pytest.approx(0.0078125, rel=REL)
    assert report.paired[PASS4].p_holm == pytest.approx(0.046875, rel=REL)
    assert report.paired[PASS4].rejected and report.blocked


def test_pooled_fabrication_indicator_is_any_of_eight():
    # FM-005 only on run 2 epoch 4, 6 new cases: binomtest(6, 6, .5, 'greater') = 0.015625 <= 0.025
    report = compare(make_run(n=10), [make_run(n=10), make_run(n=10, edit=fire(set(range(6)), {4}, "FM-005"))])
    test = report.fabrication["FM-005"]
    assert test.p == pytest.approx(0.015625, rel=REL)
    assert test.rejected and report.blocked
    assert report.fabrication["FM-015"].p == 1.0


def test_pooled_signoff_compares_raw_counts():
    # s000: baseline FM-001 on 2 epochs, pooled 1 + 2 = 3 of 8 -> R1 triggers (2, 3)
    # s001: baseline 2, pooled 1 + 1 = 2 -> equal, no trigger
    # s002: baseline 1, pooled 0 -> fall, no trigger
    # s003: baseline 1, pooled 2 + 1 = 3 -> rise of 2, triggers under R1 and R2
    base = make_run(n=6, edit=chain(fire({0, 1}, {1, 2}, "FM-001"), fire({2, 3}, {1}, "FM-001")))
    run1 = make_run(n=6, edit=chain(fire({0, 1}, {1}, "FM-001"), fire({3}, {1, 2}, "FM-001")))
    run2 = make_run(n=6, edit=chain(fire({0}, {1, 2}, "FM-001"), fire({1, 3}, {4}, "FM-001")))
    r1 = compare(base, [run1, run2], ladder="R1")
    assert _cases(r1) == {
        SignoffCase(sid(0), "FM-001", 2, 3, "R1"),
        SignoffCase(sid(3), "FM-001", 1, 3, "R1"),
    }
    r2 = compare(base, [run1, run2], ladder="R2")
    assert _cases(r2) == {SignoffCase(sid(3), "FM-001", 1, 3, "R2")}


# --- compare(): section 10(b) fabrication ----------------------------------------------

@pytest.mark.parametrize("code", FABRICATION_CODES)
def test_fabrication_on_six_new_cases_blocks(code):
    report = compare(make_run(), [make_run(edit=fire(set(range(6)), {2}, code))])
    test = report.fabrication[code]
    assert test.p == pytest.approx(0.015625, rel=REL)
    assert test.rejected and test.p_holm is None and test.alpha == FABRICATION_ALPHA
    assert test.n == 20
    # pass^4 worse 6: 6 * 0.015625 = 0.09375 under Holm, so the block comes from (b)
    assert not any(t.rejected for t in report.paired.values())
    assert report.blocked


def test_fm005_on_five_new_cases_does_not_block_but_all_reach_signoff():
    report = compare(make_run(), [make_run(edit=fire(set(range(5)), {2}, "FM-005"))])
    assert report.fabrication["FM-005"].p == pytest.approx(0.03125, rel=REL)
    assert not report.fabrication["FM-005"].rejected
    assert not report.blocked
    assert _cases(report, "FM-005") == {SignoffCase(sid(i), "FM-005", 0, 1, "R1") for i in range(5)}


def test_fabrication_discordance_counts_better_cases():
    # baseline fabricates on s010, candidate stops; candidate fabricates on s000-s005: worse 6, better 1
    # binomtest(6, 7, .5, 'greater') = 0.0625 > 0.025
    report = compare(make_run(edit=fire({10}, {1}, "FM-015")), [make_run(edit=fire(set(range(6)), {1}, "FM-015"))])
    assert report.fabrication["FM-015"].p == pytest.approx(0.0625, rel=REL)
    assert not report.fabrication["FM-015"].rejected


def test_fabrication_rise_within_case_leaves_indicator_but_reaches_signoff():
    # 1 -> 4 fabricating epochs on 6 cases: indicator unchanged (worse 0), sign-off lists all six
    report = compare(make_run(edit=fire(set(range(6)), {1}, "FM-005")),
                     [make_run(edit=fire(set(range(6)), {1, 2, 3, 4}, "FM-005"))])
    assert report.fabrication["FM-005"].p == 1.0
    assert not report.fabrication["FM-005"].rejected
    assert _cases(report, "FM-005") == {SignoffCase(sid(i), "FM-005", 1, 4, "R1") for i in range(6)}


def test_baseline_only_fabrication_never_blocks():
    report = compare(make_run(edit=fire(set(range(10)), {1}, "FM-005")), [make_run()])
    assert report.fabrication["FM-005"].p == 1.0
    assert not report.blocked
    assert report.signoff == ()


# --- compare(): section 10(c) sign-off ladder --------------------------------------------

@pytest.mark.parametrize("code", SIGNOFF_CODES)
def test_every_signoff_code_triggers_on_a_one_epoch_rise_under_r1(code):
    report = compare(make_run(n=4), [make_run(n=4, edit=fire({1}, {3}, code))])
    assert _cases(report) == {SignoffCase(sid(1), code, 0, 1, "R1")}


@pytest.mark.parametrize("name", [SWEEP_ROUTE, CODE_ROUTE])
def test_review_routes_trigger_under_r1(name):
    report = compare(make_run(n=4), [make_run(n=4, edit=route({2}, {1}, name))])
    assert _cases(report) == {SignoffCase(sid(2), name, 0, 1, "R1")}


@pytest.mark.parametrize("code", ["FM-007", "FM-008"])
def test_codes_outside_signoff_list_do_not_trigger(code):
    report = compare(make_run(n=4), [make_run(n=4, edit=fire({0, 1, 2, 3}, {1, 2, 3, 4}, code))])
    assert report.signoff == ()


def test_signoff_never_sets_blocked():
    report = compare(make_run(), [make_run(edit=fire({0}, {1}, "FM-001"))])
    assert _cases(report) == {SignoffCase(sid(0), "FM-001", 0, 1, "R1")}
    assert not report.blocked


def _ladder_runs():
    base = make_run(n=10, edit=fire({6}, {1}, "FM-002"))
    cand = make_run(n=10, edit=chain(
        fire({0}, {1}, "FM-001"),          # +1 FM-001
        fire({1}, {2}, "FM-005"),          # +1 FM-005
        fire({2}, {3}, "FM-015"),          # +1 FM-015
        route({3}, {4}, SWEEP_ROUTE),      # +1 narrative sweep
        route({4}, {1}, CODE_ROUTE),       # +1 other code route
        fire({5}, {1, 2}, "FM-001"),       # +2 FM-001
        fire({6}, {1, 2, 3}, "FM-002"),    # 1 -> 3 FM-002
        route({7}, {2, 3}, CODE_ROUTE),    # +2 other code route
        fire({8}, {1}, "FM-001"),          # two codes on one case
        fire({8}, {1}, "FM-004"),
    ))
    return base, cand


def test_ladder_r1_triggers_every_rise():
    base, cand = _ladder_runs()
    report = compare(base, [cand], ladder="R1")
    assert _cases(report) == {
        SignoffCase(sid(0), "FM-001", 0, 1, "R1"),
        SignoffCase(sid(1), "FM-005", 0, 1, "R1"),
        SignoffCase(sid(2), "FM-015", 0, 1, "R1"),
        SignoffCase(sid(3), SWEEP_ROUTE, 0, 1, "R1"),
        SignoffCase(sid(4), CODE_ROUTE, 0, 1, "R1"),
        SignoffCase(sid(5), "FM-001", 0, 2, "R1"),
        SignoffCase(sid(6), "FM-002", 1, 3, "R1"),
        SignoffCase(sid(7), CODE_ROUTE, 0, 2, "R1"),
        SignoffCase(sid(8), "FM-001", 0, 1, "R1"),
        SignoffCase(sid(8), "FM-004", 0, 1, "R1"),
    }


def test_ladder_r2_skips_single_epoch_rises_except_fabrication_and_sweep():
    base, cand = _ladder_runs()
    report = compare(base, [cand], ladder="R2")
    assert _cases(report) == {
        SignoffCase(sid(1), "FM-005", 0, 1, "R1"),
        SignoffCase(sid(2), "FM-015", 0, 1, "R1"),
        SignoffCase(sid(3), SWEEP_ROUTE, 0, 1, "R1"),
        SignoffCase(sid(5), "FM-001", 0, 2, "R2"),
        SignoffCase(sid(6), "FM-002", 1, 3, "R2"),
        SignoffCase(sid(7), CODE_ROUTE, 0, 2, "R2"),
    }


@pytest.mark.parametrize("ladder", ["R3", "r1", "", "R12"])
def test_unknown_ladder_is_value_error(ladder):
    with pytest.raises(ValueError):
        compare(make_run(n=4), [make_run(n=4)], ladder=ladder)
