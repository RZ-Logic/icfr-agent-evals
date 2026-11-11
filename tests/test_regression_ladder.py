"""The section 10(c) review-volume ladder with rule R3, and the A/A estimate.

Pre-registration section 10(c) and owner decision D-E5 (2026-11-10): R1 stands
when the A/A estimate's mean review volume under R1 is at most 10 cases per clean
PR; above it R2 applies; if R2's mean is also above 10, R3 applies and stands
whatever its volume. Under R3, FM-005, FM-015 and every narrative-sweep route key
stay on R1, the other FM codes keep R2's rise of two, and each code-check route key
triggers on a rise of three or more epochs. The A/A estimate splits one 8-epoch run
into all 35 4-and-4 halves, the half holding epoch 1 as the baseline, and reports
the review volume under R1, R2 and R3, counted by route key.
"""

from __future__ import annotations

import itertools
import math

import pytest

from icfr_agent_evals.regression import (
    AA_EPOCHS,
    LADDERS,
    REVIEW_VOLUME_CAP,
    RISE,
    SIGNOFF_CODES,
    EpochRecord,
    SignoffCase,
    aa_estimate,
    aa_splits,
    compare,
    format_aa_estimate,
    select_ladder,
    trigger_rule,
)
from icfr_agent_evals.reliability import GateError

FULL = {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}
CODE_KEY = "code_review: amount_role_unresolved"
CODE_KEY_2 = "code_review: band_word_ambiguous"
SWEEP_KEY = "narrative_sweep: narrative organization_suffix not in the structured citations: 'Zeta Trust'"


def sid(i: int) -> str:
    return f"s{i:03d}"


def run(n: int, epochs: int = 4, fires: dict | None = None) -> list[EpochRecord]:
    """n clean samples x epochs; fires maps (sample index, trigger) -> the epochs it fires on."""
    out = []
    for i in range(n):
        for e in range(1, epochs + 1):
            codes, routes = [], []
            for (j, trigger), on in (fires or {}).items():
                if j == i and e in on:
                    (codes if trigger.startswith("FM-") else routes).append(trigger)
            out.append(EpochRecord(sid(i), e, dict(FULL), not codes and not routes, tuple(codes), tuple(routes)))
    return out


def cases(report) -> set[SignoffCase]:
    return set(report.signoff)


# --- The rules ----------------------------------------------------------------------------

def test_ladder_constants():
    assert LADDERS == ("R1", "R2", "R3")
    assert RISE == {"R1": 1, "R2": 2, "R3": 3}
    assert REVIEW_VOLUME_CAP == 10 and AA_EPOCHS == 8


@pytest.mark.parametrize("trigger,expected", [
    ("FM-005", ("R1", "R1", "R1")),
    ("FM-015", ("R1", "R1", "R1")),
    (SWEEP_KEY, ("R1", "R1", "R1")),
    ("narrative_sweep", ("R1", "R1", "R1")),
    ("FM-000", ("R1", "R2", "R2")),
    ("FM-001", ("R1", "R2", "R2")),
    ("FM-016", ("R1", "R2", "R2")),
    (CODE_KEY, ("R1", "R2", "R3")),
    ("code_review", ("R1", "R2", "R3")),
])
def test_trigger_rule_by_rung(trigger, expected):
    assert tuple(trigger_rule(trigger, rung) for rung in LADDERS) == expected


def test_every_other_signoff_code_keeps_r2_under_r3():
    for code in SIGNOFF_CODES:
        assert trigger_rule(code, "R3") == ("R1" if code in ("FM-005", "FM-015") else "R2")


@pytest.mark.parametrize("rung", ["R4", "r3", ""])
def test_trigger_rule_rejects_an_unknown_rung(rung):
    with pytest.raises(ValueError):
        trigger_rule("FM-001", rung)


# --- R3 thresholds on a toy table ---------------------------------------------------------

def _toy():
    """Baseline: one code-check key on s006 epoch 1. Candidate rises listed beside."""
    base = run(10, fires={(6, CODE_KEY): {1}})
    cand = run(10, fires={
        (0, CODE_KEY): {1, 2},            # code-check key +2
        (1, CODE_KEY): {1, 2, 3},         # code-check key +3
        (2, "FM-005"): {1},               # +1 FM-005
        (3, "FM-015"): {2},               # +1 FM-015
        (4, SWEEP_KEY): {3},              # +1 sweep key
        (5, "FM-001"): {1},               # +1 other FM code
        (6, CODE_KEY): {1, 2, 3, 4},      # code-check key 1 -> 4, a rise of 3
        (7, "FM-002"): {1, 2},            # +2 other FM code
        (8, CODE_KEY_2): {1, 2, 3, 4},    # +4 code-check key
        (9, CODE_KEY): {1},               # +1 code-check key
    })
    return base, cand


def test_code_check_key_rising_by_two_triggers_under_r2_not_r3():
    base, cand = _toy()
    r2, r3 = compare(base, [cand], ladder="R2"), compare(base, [cand], ladder="R3")
    assert SignoffCase(sid(0), CODE_KEY, 0, 2, "R2") in cases(r2)
    assert not any(c.sample_id == sid(0) for c in r3.signoff)


def test_toy_table_under_each_rung():
    base, cand = _toy()
    assert cases(compare(base, [cand], ladder="R1")) == {
        SignoffCase(sid(0), CODE_KEY, 0, 2, "R1"),
        SignoffCase(sid(1), CODE_KEY, 0, 3, "R1"),
        SignoffCase(sid(2), "FM-005", 0, 1, "R1"),
        SignoffCase(sid(3), "FM-015", 0, 1, "R1"),
        SignoffCase(sid(4), SWEEP_KEY, 0, 1, "R1"),
        SignoffCase(sid(5), "FM-001", 0, 1, "R1"),
        SignoffCase(sid(6), CODE_KEY, 1, 4, "R1"),
        SignoffCase(sid(7), "FM-002", 0, 2, "R1"),
        SignoffCase(sid(8), CODE_KEY_2, 0, 4, "R1"),
        SignoffCase(sid(9), CODE_KEY, 0, 1, "R1"),
    }
    assert cases(compare(base, [cand], ladder="R2")) == {
        SignoffCase(sid(0), CODE_KEY, 0, 2, "R2"),
        SignoffCase(sid(1), CODE_KEY, 0, 3, "R2"),
        SignoffCase(sid(2), "FM-005", 0, 1, "R1"),
        SignoffCase(sid(3), "FM-015", 0, 1, "R1"),
        SignoffCase(sid(4), SWEEP_KEY, 0, 1, "R1"),
        SignoffCase(sid(6), CODE_KEY, 1, 4, "R2"),
        SignoffCase(sid(7), "FM-002", 0, 2, "R2"),
        SignoffCase(sid(8), CODE_KEY_2, 0, 4, "R2"),
    }
    r3 = compare(base, [cand], ladder="R3")
    assert cases(r3) == {
        SignoffCase(sid(1), CODE_KEY, 0, 3, "R3"),
        SignoffCase(sid(2), "FM-005", 0, 1, "R1"),
        SignoffCase(sid(3), "FM-015", 0, 1, "R1"),
        SignoffCase(sid(4), SWEEP_KEY, 0, 1, "R1"),
        SignoffCase(sid(6), CODE_KEY, 1, 4, "R3"),
        SignoffCase(sid(7), "FM-002", 0, 2, "R2"),
        SignoffCase(sid(8), CODE_KEY_2, 0, 4, "R3"),
    }
    assert any(n.startswith("ladder R3") for n in r3.notes)


def test_r3_counts_each_route_key_on_its_own():
    """Two code-check keys on one case each rise by 3: two sign-off cases."""
    base = run(4)
    cand = run(4, fires={(0, CODE_KEY): {1, 2, 3}, (0, CODE_KEY_2): {2, 3, 4}})
    assert cases(compare(base, [cand], ladder="R3")) == {
        SignoffCase(sid(0), CODE_KEY, 0, 3, "R3"),
        SignoffCase(sid(0), CODE_KEY_2, 0, 3, "R3"),
    }


def test_r3_pooled_candidate_counts_raw_pooled_epochs():
    """Section 10(d): two pooled runs with the key on two epochs each make 4 against 0."""
    base = run(4)
    one = run(4, fires={(0, CODE_KEY): {1, 2}})
    assert cases(compare(base, [one], ladder="R3")) == set()
    assert cases(compare(base, [one, one], ladder="R3")) == {SignoffCase(sid(0), CODE_KEY, 0, 4, "R3")}


# --- Ladder selection from mean volumes -------------------------------------------------

@pytest.mark.parametrize("volume,rung", [
    ({"R1": 0.0, "R2": 0.0, "R3": 0.0}, "R1"),
    ({"R1": 10.0, "R2": 3.0, "R3": 1.0}, "R1"),
    ({"R1": 10.01, "R2": 10.0, "R3": 2.0}, "R2"),
    ({"R1": 25.0, "R2": 4.0, "R3": 1.0}, "R2"),
    ({"R1": 25.0, "R2": 10.5, "R3": 4.0}, "R3"),
    ({"R1": 40.0, "R2": 30.0, "R3": 25.0}, "R3"),   # R3 stands whatever its volume
])
def test_select_ladder(volume, rung):
    assert select_ladder(volume) == rung


@pytest.mark.parametrize("volume", [{}, {"R1": 1.0}, {"R1": -1.0, "R2": 0.0}, {"R1": math.nan, "R2": 0.0},
                                    {"R1": 1.0, "R2": math.inf}, {"R1": True, "R2": 0.0}])
def test_select_ladder_rejects_bad_volumes(volume):
    with pytest.raises(ValueError):
        select_ladder(volume)


# --- The A/A estimate ---------------------------------------------------------------------

def test_aa_splits_are_the_35_halves_holding_epoch_1_as_baseline():
    splits = aa_splits()
    assert len(splits) == 35 == math.comb(7, 3)
    assert len(set(splits)) == 35
    for base, cand in splits:
        assert 1 in base and len(base) == len(cand) == 4
        assert sorted(base + cand) == list(range(1, 9))
        assert list(base) == sorted(base) and list(cand) == sorted(cand)


@pytest.mark.parametrize("epochs,k", [(6, 4), (8, 3), (0, 0)])
def test_aa_splits_need_two_halves_of_k(epochs, k):
    with pytest.raises(ValueError):
        aa_splits(epochs, k)


AA_FIRES = {
    (0, CODE_KEY): {2, 3, 4, 5},         # a code-check key on four of eight epochs: rises of 4, 2, 0
    (0, CODE_KEY_2): {2, 3, 4},          # a second key on the same case, counted on its own: rises of 3, 1
    (1, "FM-001"): {5, 6},               # another FM code on two epochs
    (2, SWEEP_KEY): {8},                 # a sweep key on one epoch
    (3, "FM-005"): {1, 7},               # a fabrication code on two epochs
}


def _expected_volume(fires):
    """Brute force over every 4-of-8 half with epoch 1 in the baseline, written apart
    from the code under test: rung -> (mean cases by key, mean distinct samples)."""
    halves = [(1, *rest) for rest in itertools.combinations(range(2, 9), 3)]
    out = {}
    for rung, rise in (("R1", 1), ("R2", 2), ("R3", 3)):
        by_key, by_sample = [], []
        for base in halves:
            hits = []
            for (i, trigger), on in fires.items():
                b = len(on & set(base))
                c = len(on) - b
                if trigger in ("FM-005", "FM-015") or trigger.startswith("narrative_sweep"):
                    need = 1
                elif trigger.startswith("FM-"):
                    need = min(rise, 2)
                else:
                    need = rise
                if c - b >= need:
                    hits.append(i)
            by_key.append(len(hits))
            by_sample.append(len(set(hits)))
        out[rung] = (sum(by_key) / len(halves), sum(by_sample) / len(halves))
    return out


def test_aa_estimate_reports_volume_under_each_rung_by_route_key():
    records = run(6, epochs=8, fires=AA_FIRES)
    est = aa_estimate(records)
    assert len(est.splits) == 35
    expected = _expected_volume(AA_FIRES)
    for rung in LADDERS:
        assert est.mean_volume[rung] == pytest.approx(expected[rung][0], abs=1e-12)
        assert est.mean_volume_samples[rung] == pytest.approx(expected[rung][1], abs=1e-12)
    assert est.mean_volume["R1"] >= est.mean_volume["R2"] >= est.mean_volume["R3"]
    assert est.mean_volume["R3"] < est.mean_volume["R2"]  # the code-check keys need a rise of 3
    assert est.ladder == "R1"  # well under 10 cases per clean PR
    assert est.false_block_rate == 0.0 and not est.false_block_shortfall
    text = format_aa_estimate(est)
    assert "R1:" in text and "R2:" in text and "R3:" in text and "counted by route key" in text


def test_aa_estimate_moves_down_the_ladder_on_volume():
    """Many single-epoch FM-001 and code-key firings: R1 volume above 10, R2 also
    above 10 through the code-check keys, so the ladder takes R3."""
    fires = {}
    for i in range(40):
        fires[(i, "FM-001")] = {5}
        fires[(i, CODE_KEY)] = {5, 6}
    est = aa_estimate(run(40, epochs=8, fires=fires))
    assert est.mean_volume["R1"] > REVIEW_VOLUME_CAP
    assert est.mean_volume["R2"] > REVIEW_VOLUME_CAP
    assert est.ladder == "R3"
    assert est.mean_volume["R3"] == 0.0  # two of eight epochs can never rise by three


def test_aa_estimate_needs_eight_epochs():
    with pytest.raises(GateError):
        aa_estimate(run(3, epochs=4))
