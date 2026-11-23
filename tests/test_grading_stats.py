"""scripts/grading_stats.py: the reliability gate with its worst-case figure and break-even
counts, the case-cluster bootstrap, the circuit breaker, the queue and the LLM sub-dimension
means of the full-run grading job (D-R23: reliability gate, failed judgments, circuit
breaker, queue, means).

Pure and offline: every input is a hand-built record; no file is read, no model is called.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction

import pytest

import gradingfactory as gf
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import judge_trust as jt

G, C = gp.DIMENSIONS
DIMS = gp.DIMENSIONS
gs = gf.load_script("grading_stats")
P, F = gf.parsed, gf.failed
M1, M2, M3 = "m/one", "m/two", "m/three"


def FJ():
    """A failed judgment: asked three times with no completed call, no reply."""
    return {**gf.failed(), "status": "failed", "reply": None}


def uid(i: int) -> str:
    return f"u{i:04d}"


def rows_from(pairs, *, start=1, cluster=None):
    """Complete rows from (grader (g, c), calibrator (g, c)) pairs."""
    out = []
    for i, (a, b) in enumerate(pairs, start=start):
        sid = cluster(i) if cluster else f"s{i:04d}"
        out.append(gf.row(uid(i), P(*a), P(*b), sample_id=sid))
    return out


def passes(rows):
    g = {d: {r["uid"]: r["grader"]["scores"][d] for r in rows} for d in DIMS}
    c = {d: {r["uid"]: r["calibrator"]["scores"][d] for r in rows} for d in DIMS}
    return g, c


def varied(n, *, offset=0):
    """n complete pairs in exact agreement over every level, both sub-dimensions varied."""
    return [((i % 4, (i // 4 + offset) % 4), (i % 4, (i // 4 + offset) % 4)) for i in range(n)]


# --- Constants --------------------------------------------------------------------------------

def test_constants():
    assert gs.GATE_THRESHOLD == jt.ICC_LOWER_LIMIT == 0.60
    assert gs.BREAKER_LIMIT == Fraction(1, 4) and gs.DISAGREEMENT_POINTS == 1
    assert gs.BOOTSTRAP_RESAMPLES == 10_000 and gs.Z95 == 1.96
    assert gs.SUBSETS == ("all", "synthetic", "edge", "adversarial", "difficulty_bearing")
    assert gs.TAG_ORDER == (gs.TAG_DISAGREEMENT, gs.TAG_GRADER_LOW, gs.TAG_CALIBRATOR_LOW, gs.TAG_GRADER_FAILED,
                            gs.TAG_CALIBRATOR_FAILED)
    assert gs.DISPOSITIONS_WITH_CALIBRATOR == ("grader-correct", "calibrator-correct", "both-wrong",
                                               "ambiguous-edge-case-add-to-Verified-subset")
    assert gs.DISPOSITIONS_NO_CALIBRATOR == ("grader-correct", "grader-wrong",
                                             "ambiguous-edge-case-add-to-Verified-subset")
    assert gs.DISPOSITIONS_GRADER_FAILED == ("calibrator-correct", "both-wrong",
                                             "ambiguous-edge-case-add-to-Verified-subset")
    assert gs.DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR == ("grader-wrong", "ambiguous-edge-case-add-to-Verified-subset")
    assert gs.EXCLUSION_BREAKER == "circuit breaker tripped"
    assert gs.LABEL_BOOTSTRAP == "reported, not gated" == gs.LABEL_WORST_CASE == gs.LABEL_BREAK_EVEN
    assert gs.WORST_CASE_NOTE.startswith("only icc_a1 is meaningful")
    assert "complete pairs reaches 0.60" in gs.GATE_RULE and "reported, not gated" in gs.GATE_RULE
    assert gs.GATE_RULE.endswith("(D-R23, reliability gate)")
    assert gs.BREAK_EVEN_RULE.startswith("the fewest added dropped pairs")
    assert gs.CI_NOTE == "normal 95 percent interval, clipped to the 0 to 3 scale"
    assert set(gs.LABEL_PAIRED) == {"primary", "sensitivity"}
    # the two-figure gate is withdrawn: no gate function or reason of it is left
    assert not hasattr(gs, "two_figure_gate") and not hasattr(gs, "EXCLUSION_NOT_ROBUST")


def test_reply_helpers():
    assert gs.is_parsed(P(1, 1)) and not gs.is_parsed(F()) and not gs.is_parsed(None)
    assert gs.any_low(P(1, 1, low_c=True)) and not gs.any_low(P(1, 1)) and not gs.any_low(F())
    assert gs.disagrees(P(0, 1), P(2, 1), DIMS) and not gs.disagrees(P(0, 1), P(1, 2), DIMS)
    assert not gs.disagrees(P(0, 1), F(), DIMS)
    assert gs.disagrees(P(3, 3), P(3, 0), DIMS)


# --- Worst-case fill and the reliability gate ---------------------------------------------------

def test_worst_case_fill_one_parsed():
    rows = [gf.row(uid(s + 1), P(s, s), F("refused")) for s in range(4)]
    rows += [gf.row(uid(s + 11), F(), P(s, 3 - s)) for s in range(4)]
    rows.append(gf.row(uid(30), P(2, 1), P(2, 2)))
    gpass, cpass, fills = gs.fill_worst_case(rows)
    assert [cpass[G][uid(s + 1)] for s in range(4)] == [3, 3, 0, 0]
    assert [cpass[C][uid(s + 1)] for s in range(4)] == [3, 3, 0, 0]
    assert [gpass[G][uid(s + 1)] for s in range(4)] == [0, 1, 2, 3]       # the parsed reply is kept
    assert [gpass[G][uid(s + 11)] for s in range(4)] == [3, 3, 0, 0]
    assert [gpass[C][uid(s + 11)] for s in range(4)] == [0, 0, 3, 3]       # coverage scores are 3, 2, 1, 0
    assert gpass[G][uid(30)] == 2 and cpass[C][uid(30)] == 2            # a complete row keeps its scores
    assert {(f["uid"], f["dim"]) for f in fills} == {(uid(i), d) for i in (1, 2, 3, 4, 11, 12, 13, 14) for d in DIMS}
    assert all(f["rule"] == "one parsed" for f in fills)
    for f in fills:
        assert f["grader"] == gpass[f["dim"]][f["uid"]] and f["calibrator"] == cpass[f["dim"]][f["uid"]]


def test_worst_case_fill_neither_parsed_alternates_in_uid_order():
    rows = [gf.row(uid(9), F(), F("refused")), gf.row(uid(2), F("refused"), F()), gf.row(uid(5), F(), F()),
            gf.row(uid(1), P(1, 1), P(1, 2))]
    gpass, cpass, fills = gs.fill_worst_case(rows)
    for d in DIMS:
        assert (gpass[d][uid(2)], cpass[d][uid(2)]) == (0, 3)      # k = 0
        assert (gpass[d][uid(5)], cpass[d][uid(5)]) == (3, 0)      # k = 1
        assert (gpass[d][uid(9)], cpass[d][uid(9)]) == (0, 3)      # k = 2
    assert cpass[C][uid(1)] == 2 and cpass[G][uid(1)] == 1
    assert {f["rule"] for f in fills} == {"neither parsed"}
    assert {(f["uid"], f["dim"]) for f in fills} == {(uid(i), d) for i in (2, 5, 9) for d in DIMS}


def test_complete_and_dropped_rows():
    rows = [gf.row(uid(1), P(1, 1), P(1, 1)), gf.row(uid(2), F(), P(1, 1)), gf.row(uid(3), P(1, 1), F("refused"))]
    assert [r["uid"] for r in gs.complete_rows(rows)] == [uid(1)]
    assert [r["uid"] for r in gs.dropped_rows(rows)] == [uid(2), uid(3)]


def test_the_gate_and_the_worst_case_figure_agree_with_no_dropped_pair():
    pairs = [((i % 4, (i * 3) % 4), (i % 4 if i % 7 else (i + 1) % 4, (i * 3) % 4)) for i in range(40)]
    rows = rows_from(pairs)
    gate = gs.reliability_gate(rows)
    want = jt.reliability_exclusion(*passes(rows), subdimensions=DIMS)
    assert gate["complete"]["computed"] and gate["worst_case"]["computed"]
    assert gate["complete"]["exclusion"] == want == gate["worst_case"]["exclusion"]
    assert gate["rows"] == 40 and gate["pairs_complete"] == 40 and gate["pairs_dropped"] == 0
    assert gate["neither_parsed"] == 0 and gate["fills"] == []
    assert gate["kept"] == want["kept"] and gate["threshold"] == 0.60
    assert gate["rule"] == gs.GATE_RULE and "0.60" in gate["rule"]
    assert gate["break_even_rule"] == gs.BREAK_EVEN_RULE
    assert set(gate) == {"rows", "pairs_complete", "pairs_dropped", "neither_parsed", "complete", "worst_case",
                         "fills", "break_even", "break_even_rule", "kept", "excluded", "threshold", "rule"}


def test_the_worst_case_figure_is_marked_reported_not_gated():
    rows = rows_from(varied(40)) + [gf.row(uid(100), P(3, 3), F("refused"))]
    gate = gs.reliability_gate(rows)
    worst = gate["worst_case"]
    assert worst["label"] == "reported, not gated" == gs.LABEL_WORST_CASE
    assert worst["note"] == gs.WORST_CASE_NOTE and "only icc_a1 is meaningful" in worst["note"]
    assert worst["computed"] is True and set(worst["exclusion"]["kept"]) <= set(DIMS)
    # a figure that cannot be computed still carries the marker
    lone = gs.reliability_gate([gf.row(uid(1), F(), P(1, 1))])
    assert lone["worst_case"]["computed"] is False
    assert lone["worst_case"]["label"] == gs.LABEL_WORST_CASE and lone["worst_case"]["note"] == gs.WORST_CASE_NOTE


def test_the_break_even_block_is_marked_reported_not_gated():
    rows = rows_from(varied(40)) + [gf.row(uid(100), P(3, 3), F("refused"))]
    gate = gs.reliability_gate(rows)
    assert gate["break_even"]["label"] == "reported, not gated" == gs.LABEL_WORST_CASE
    assert gate["break_even"]["label"] == gate["worst_case"]["label"]
    assert set(gate["break_even"]) == {"label", *DIMS}
    for d in DIMS:
        assert set(gate["break_even"][d]) == {"count", "searched_up_to", "reason"}
    # an undefined figure still carries the marker
    lone = gs.reliability_gate([gf.row(uid(1), F(), P(1, 1))])
    assert lone["break_even"]["label"] == gs.LABEL_BREAK_EVEN


def test_passing_on_complete_pairs_and_failing_worst_case_is_kept():
    """The tagged gate decides alone: the worst-case figure is reported, never gated."""
    complete = rows_from(varied(40))
    dropped = [gf.row(uid(100 + i), P(3, 3), F("refused")) for i in range(8)]
    dropped += [gf.row(uid(200 + i), F(), P(0, 0)) for i in range(4)]
    rows = complete + dropped
    # preconditions, on the package statistic
    g, c = gs.fill_worst_case(rows)[:2]
    for d in DIMS:
        assert jt.icc_a1([[g[d][u], c[d][u]] for u in sorted(g[d])])["ci_lower"] < 0.60
        assert jt.icc_a1([[r["grader"]["scores"][d], r["calibrator"]["scores"][d]] for r in complete])["ci_lower"] >= 0.60
    gate = gs.reliability_gate(rows)
    assert gate["pairs_complete"] == 40 and gate["pairs_dropped"] == 12 and gate["rows"] == 52
    assert gate["kept"] == list(DIMS) and gate["excluded"] == []
    assert gate["complete"]["exclusion"]["kept"] == list(DIMS)
    assert gate["worst_case"]["exclusion"]["kept"] == []                 # published beside the gate
    assert gate["worst_case"]["label"] == "reported, not gated"
    assert gate["worst_case"]["note"] == gs.WORST_CASE_NOTE
    for d in DIMS:
        be = gate["break_even"][d]
        assert isinstance(be["count"], int) and be["count"] >= 1 and be["reason"] is None
    out = gs.claims(gate, gs.breaker([], rows))
    assert all(out[d]["kept"] for d in DIMS)


def test_one_dropped_pair_can_leave_the_gate_passing():
    rows = rows_from(varied(80)) + [gf.row(uid(500), P(2, 1), F())]
    gate = gs.reliability_gate(rows)
    assert gate["kept"] == list(DIMS) and gate["excluded"] == [] and gate["pairs_dropped"] == 1


def test_an_undefined_figure_excludes_as_undefined():
    same = rows_from([((2, 2), (2, 2))] * 10)
    gate = gs.reliability_gate(same)
    assert gate["kept"] == []
    assert {e["reason"] for e in gate["excluded"]} == {jt.EXCLUSION_UNDEFINED}
    assert all(gate["break_even"][d] == {"count": 0, "searched_up_to": 0, "reason": gs.BREAK_EVEN_EXCLUDED}
               for d in DIMS)
    one = rows_from([((2, 1), (2, 1))])
    gate = gs.reliability_gate(one)
    assert gate["complete"]["computed"] is False and gate["complete"]["exclusion"] is None
    assert gate["complete"]["reason"]
    assert gate["kept"] == [] and {e["reason"] for e in gate["excluded"]} == {jt.EXCLUSION_UNDEFINED}
    assert all(e["complete"] is None for e in gate["excluded"])
    none_complete = [gf.row(uid(1), F(), P(1, 1)), gf.row(uid(2), P(2, 2), F("refused"))]
    gate = gs.reliability_gate(none_complete)
    assert gate["complete"]["computed"] is False and gate["pairs_complete"] == 0
    assert {e["reason"] for e in gate["excluded"]} == {jt.EXCLUSION_UNDEFINED}
    assert gs.one_figure({d: {} for d in DIMS}, {d: {} for d in DIMS}, DIMS)["computed"] is False


def _max_disagreement(k):
    return [[0, 3] if i % 2 == 0 else [3, 0] for i in range(k)]


def test_break_even_count_is_the_first_added_pair_count_that_fails():
    complete = rows_from(varied(240))
    rows = complete + [gf.row(uid(900), P(3, 3), F("refused"))]
    gate = gs.reliability_gate(rows)
    assert gate["kept"] == list(DIMS) and gate["pairs_complete"] == 240
    for d in DIMS:
        be = gate["break_even"][d]
        count = be["count"]
        assert isinstance(count, int) and count >= 1
        assert be["searched_up_to"] == count and be["reason"] is None
        base = [[r["grader"]["scores"][d], r["calibrator"]["scores"][d]] for r in complete]
        before = jt.icc_a1(base + _max_disagreement(count - 1))["ci_lower"]
        after = jt.icc_a1(base + _max_disagreement(count))["ci_lower"]
        assert before is not None and before >= 0.60
        assert after is None or after < 0.60


def test_break_even_is_zero_when_excluded_on_complete_pairs():
    matrix = [[3, 3]] * 228 + [[2, 2]] * 6 + [[3, 2]] * 3 + [[2, 3]] * 3
    cov = varied(240)
    pairs = [((a, cov[i][0][1]), (b, cov[i][1][1])) for i, (a, b) in enumerate(matrix)]
    gate = gs.reliability_gate(rows_from(pairs))
    assert gate["kept"] == [C]
    assert gate["break_even"][G] == {"count": 0, "searched_up_to": 0, "reason": "excluded on complete pairs"}
    assert isinstance(gate["break_even"][C]["count"], int) and gate["break_even"][C]["count"] >= 1


def test_break_even_none_and_undefined(monkeypatch):
    complete = rows_from(varied(6))
    monkeypatch.setattr(gs, "_lower_limit_or_none", lambda m: 0.95)
    assert gs.break_even(complete, G, kept_on_complete=True) == {
        "count": None, "searched_up_to": 6, "reason": gs.BREAK_EVEN_NONE}
    monkeypatch.undo()

    def refuse(matrix, **kw):
        raise jt.Refused("no")

    monkeypatch.setattr(gs.jt, "icc_a1", refuse)
    assert gs.break_even(complete, G, kept_on_complete=True) == {"count": 1, "searched_up_to": 1, "reason": None}


def test_the_ceiling_example_of_the_disclosure():
    matrix = [[3, 3]] * 228 + [[2, 2]] * 6 + [[3, 2]] * 3 + [[2, 3]] * 3
    exact = sum(a == b for a, b in matrix) / len(matrix)
    assert len(matrix) == 240 and exact == 0.975
    r = jt.icc_a1(matrix)
    assert round(r["estimate"], 3) == 0.655 and round(r["ci_lower"], 3) == 0.576
    one_way = jt.icc_a1([[3, 3]] * 228 + [[2, 2]] * 6 + [[3, 2]] * 6)
    assert round(one_way["estimate"], 3) == 0.656 and round(one_way["ci_lower"], 3) == 0.577
    # through the gate: grounding at the ceiling fails below 0.60 on complete pairs; coverage is kept
    cov = varied(240)
    pairs = [((a, cov[i][0][1]), (b, cov[i][1][1])) for i, (a, b) in enumerate(matrix)]
    gate = gs.reliability_gate(rows_from(pairs))
    assert gate["kept"] == [C]
    (ex,) = gate["excluded"]
    assert ex["subdimension"] == G and ex["reason"] == jt.EXCLUSION_BELOW
    assert round(ex["complete"]["estimate"], 3) == 0.655 and round(ex["complete"]["ci_lower"], 3) == 0.576


def test_icc_estimate_matches_judge_trust_exactly():
    rng = random.Random(20261122)
    checked = 0
    for _ in range(60):
        n = rng.randint(2, 30)
        m = [[rng.randint(0, 3), rng.randint(0, 3)] for _ in range(n)]
        want = jt.icc_a1(m)["estimate_exact"]
        got = gs.icc_a1_estimate(m)
        if want is None:
            assert got is None
        else:
            assert Fraction(got) == Fraction(want)
            checked += 1
    assert checked > 40
    assert gs.icc_a1_estimate([[2, 2]] * 5) is None


# --- Case-cluster bootstrap ---------------------------------------------------------------------

def _replica_bootstrap(rows, seed, resamples, d):
    clusters = sorted({r["sample_id"] for r in rows})
    members = {c: [r for r in rows if r["sample_id"] == c] for c in clusters}
    rng = random.Random(seed)
    values, undefined = [], 0
    for _ in range(resamples):
        picks = rng.choices(range(len(clusters)), k=len(clusters))
        m = [[r["grader"]["scores"][d], r["calibrator"]["scores"][d]] for p in picks for r in members[clusters[p]]]
        est = jt.icc_a1(m)["estimate_exact"] if len(m) >= 2 else None
        if est is None:
            undefined += 1
            continue
        values.append(float(Fraction(est)))
    values.sort()
    lower = values[math.floor(0.025 * len(values))] if values else None
    return lower, len(values), undefined


def test_case_cluster_bootstrap_resamples_cases_with_all_their_rows_and_counts_undefined():
    flat = [((2, 2), (2, 2))] * 6                                      # clusters a and b: all equal
    varied_pairs = [((0, 1), (0, 1)), ((3, 2), (3, 3)), ((1, 0), (2, 0)), ((2, 3), (2, 3))]
    rows = rows_from(flat, cluster=lambda i: "a" if i <= 3 else "b")
    rows += rows_from(varied_pairs, start=10, cluster=lambda i: "c")
    rows += rows_from([((1, 2), (1, 2)), ((0, 0), (1, 0))], start=20, cluster=lambda i: "d")
    rows.append(gf.row(uid(99), P(3, 3), F("refused"), sample_id="e"))   # dropped: not in the bootstrap
    out = gs.case_cluster_bootstrap(rows, seed=17, resamples=400)
    again = gs.case_cluster_bootstrap(rows, seed=17, resamples=400)
    assert out == again
    complete = gs.complete_rows(rows)
    for d in DIMS:
        b = out[d]
        lower, defined, undefined = _replica_bootstrap(complete, 17, 400, d)
        assert b["clusters"] == 4 and b["rows"] == 12 and b["seed"] == 17 and b["resamples"] == 400
        assert b["defined_resamples"] == defined and b["undefined_resamples"] == undefined
        assert b["defined_resamples"] + b["undefined_resamples"] == 400 and b["undefined_resamples"] > 0
        assert b["ci_lower"] == pytest.approx(lower)
        assert b["label"] == gs.LABEL_BOOTSTRAP
    other = gs.case_cluster_bootstrap(rows, seed=18, resamples=400)
    assert other[G]["seed"] == 18


# --- Circuit breaker ----------------------------------------------------------------------------

def unit(i, grader, model=M1, sid=None, epoch=1):
    return {"uid": uid(i), "model": model, "sample_id": sid or f"s{i:04d}", "epoch": epoch, "case_kind": "standard",
            "defect_epoch": False, "messages_sha256": "0" * 64, "grader": grader}


def test_breaker_pair_and_generation_rates_trip_strictly_above_one_quarter():
    units = [unit(i, P(1, 1, low_g=i <= 2)) for i in range(1, 9)]         # 2 of 8 low: exactly 1/4
    b = gs.breaker(units, [])
    assert set(b) == {"unit", "pair_rate", "generation_rate", "limit", "trip", "per_worker", "note"}
    gr, pr = b["generation_rate"], b["pair_rate"]
    assert gr == {**gr, "tagged": 2, "denominator": 8, "rate": 0.25, "rate_exact": "1/4"}
    assert pr["denominator"] == 0 and pr["rate"] is None and pr["rate_exact"] is None
    assert b["trip"] is False and b["unit"] == "full run" and b["limit"] == "1/4"
    units[2]["grader"] = P(1, 1, low_g=True)                               # 3 of 8
    assert gs.breaker(units, [])["trip"] is True
    # pair rate: 1 of 4 complete pairs disagree, no trip; 2 of 4, trip, while the generation rate stays under
    units = [unit(i, P(1, 1)) for i in range(1, 21)]
    rows = [gf.row(uid(i), units[i - 1]["grader"], P(1, 1)) for i in range(1, 5)]
    rows[0]["calibrator"] = P(3, 1)
    b = gs.breaker(units, rows)
    assert (b["pair_rate"]["tagged"], b["pair_rate"]["denominator"]) == (1, 4) and b["trip"] is False
    rows[1]["calibrator"] = P(1, 1, low_c=True)                            # calibrator low counts in the pair rate
    b = gs.breaker(units, rows)
    assert (b["pair_rate"]["tagged"], b["pair_rate"]["denominator"], b["pair_rate"]["rate_exact"]) == (2, 4, "1/2")
    assert b["generation_rate"]["tagged"] == 2 and b["generation_rate"]["denominator"] == 20 and b["trip"] is True


def test_failed_judgments_are_tagged_but_left_out_of_the_breaker():
    units = [unit(1, F()), unit(2, P(1, 1)), unit(3, P(1, 1)), unit(4, P(2, 2))]
    rows = [gf.row(uid(2), units[1]["grader"], F("refused")), gf.row(uid(3), units[2]["grader"], P(1, 1)),
            gf.row(uid(1), units[0]["grader"], P(3, 3))]
    b = gs.breaker(units, rows)
    assert (b["pair_rate"]["tagged"], b["pair_rate"]["denominator"]) == (0, 1)
    assert (b["generation_rate"]["tagged"], b["generation_rate"]["denominator"]) == (0, 3)
    assert not gs.breaker_tagged(units[0]["grader"], rows[2], DIMS)
    assert not gs.breaker_tagged(units[1]["grader"], rows[0], DIMS)
    assert gs.queue_tags(units[0]["grader"], rows[2], DIMS) == [gs.TAG_GRADER_FAILED]
    assert gs.queue_tags(units[1]["grader"], rows[0], DIMS) == [gs.TAG_CALIBRATOR_FAILED]
    items = gs.queue_items(units, rows)
    assert [i["unit_id"] for i in items] == [uid(1), uid(2)]


def test_a_failed_judgment_is_a_reply_that_did_not_parse():
    """A failed judgment (no reply after three asks) is a dropped pair in the gate, ungraded
    in the means, never a breaker tag, and tagged failed in the queue."""
    # gate: a failed calibrator reply and a failed grader reply are dropped pairs, filled one-parsed
    rows = rows_from(varied(40)) + [gf.row(uid(100), P(3, 0), FJ()), gf.row(uid(101), FJ(), P(1, 2))]
    gate = gs.reliability_gate(rows)
    assert gate["pairs_complete"] == 40 and gate["pairs_dropped"] == 2 and gate["neither_parsed"] == 0
    assert {(f["uid"], f["rule"]) for f in gate["fills"]} == {(uid(100), "one parsed"), (uid(101), "one parsed")}
    assert [r["uid"] for r in gs.dropped_rows(rows)] == [uid(100), uid(101)]
    both = gs.reliability_gate(rows_from(varied(40)) + [gf.row(uid(102), FJ(), FJ())])
    assert both["neither_parsed"] == 1 and both["pairs_dropped"] == 1
    # breaker: never a tag; the dropped pair leaves the pair rate, its generation stays on its grader reply
    units = [unit(1, FJ()), unit(2, P(1, 1)), unit(3, P(1, 1, low_g=True))]
    brows = [gf.row(uid(2), units[1]["grader"], FJ()), gf.row(uid(3), units[2]["grader"], FJ())]
    b = gs.breaker(units, brows)
    assert b["pair_rate"]["denominator"] == 0
    assert (b["generation_rate"]["tagged"], b["generation_rate"]["denominator"]) == (1, 2)
    assert not gs.breaker_tagged(units[0]["grader"], None, DIMS)
    # queue: tagged failed, the item keeps the error and has no reply
    assert gs.queue_tags(units[0]["grader"], None, DIMS) == [gs.TAG_GRADER_FAILED]
    assert gs.queue_tags(units[1]["grader"], brows[0], DIMS) == [gs.TAG_CALIBRATOR_FAILED]
    items = {i["unit_id"]: i for i in gs.queue_items(units, brows)}
    assert items[uid(1)]["grader"]["status"] == "failed" and items[uid(1)]["grader"]["reply"] is None
    assert items[uid(1)]["grader"]["error"] == FJ()["error"]
    assert items[uid(1)]["allowed_dispositions"] == list(gs.DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR)
    assert items[uid(2)]["calibrator"]["status"] == "failed"
    assert gs.TAG_CALIBRATOR_FAILED in items[uid(3)]["tags"] and gs.TAG_GRADER_LOW in items[uid(3)]["tags"]
    # a failed grader reply with a calibrator score offers the grader-failed dispositions
    calibrated = gs.queue_items([unit(4, FJ())], [gf.row(uid(4), FJ(), P(2, 2))])
    assert calibrated[0]["allowed_dispositions"] == list(gs.DISPOSITIONS_GRADER_FAILED)
    # means: ungraded and missing in both readings
    gens = [gen(M1, "s1", 1, u=uid(1)), gen(M1, "s1", 2, u=uid(2))]
    out = gs.llm_means(gens, {uid(1): FJ(), uid(2): P(2, 1)})[M1]["all"][G]
    for reading in ("primary", "sensitivity"):
        assert out[reading]["values"] == 1 and out[reading]["mean"] == pytest.approx(2.0)
        assert out[reading]["ungraded_generations"] == 1


def test_breaker_per_worker_rates_never_trip():
    units = [unit(1, P(1, 1, low_g=True), model=M1)] + [unit(i, P(1, 1), model=M2) for i in range(2, 12)]
    b = gs.breaker(units, [])
    assert b["trip"] is False and b["generation_rate"]["rate_exact"] == "1/11"
    assert b["per_worker"][M1]["generation_rate"]["rate"] == 1.0
    assert b["per_worker"][M2]["generation_rate"]["rate"] == 0.0
    assert set(b["per_worker"][M1]) == {"pair_rate", "generation_rate", "descriptive"}
    assert "never a trip" in json.dumps(b["per_worker"])
    assert "undefined rate does not trip" in b["note"]


def test_a_trip_excludes_both_sub_dimensions():
    rows = rows_from(varied(40))
    gate = gs.reliability_gate(rows)
    assert gate["kept"] == list(DIMS)
    units = [unit(i, P(1, 1, low_g=True)) for i in range(1, 41)]
    tripped = gs.breaker(units, [])
    assert tripped["trip"] is True
    out = gs.claims(gate, tripped)
    for d in DIMS:
        assert out[d]["kept"] is False and gs.EXCLUSION_BREAKER in out[d]["reasons"]
    assert out["anchors_stand_alone"] is True
    calm = gs.breaker([unit(i, P(1, 1)) for i in range(1, 41)], [])
    out = gs.claims(gate, calm)
    assert all(out[d]["kept"] is True and out[d]["reasons"] == [] for d in DIMS)
    assert out["anchors_stand_alone"] is False
    pairs = [((a, cv[0][1]), (b, cv[1][1])) for (a, b), cv in
             zip([[3, 3]] * 228 + [[2, 2]] * 6 + [[3, 2]] * 3 + [[2, 3]] * 3, varied(240))]
    half = gs.reliability_gate(rows_from(pairs))
    out = gs.claims(half, calm)
    assert out[G]["kept"] is False and jt.EXCLUSION_BELOW in out[G]["reasons"] and out[C]["kept"] is True
    assert out["anchors_stand_alone"] is False


# --- Queue --------------------------------------------------------------------------------------

def test_no_calibrator_score_dispositions_follow_the_grader_reply():
    """With no calibrator score (no row, or a calibrator reply that did not parse), a parsed
    grader reply keeps grader-correct, grader-wrong and ambiguous; a grader reply that did
    not parse, was refused or is a failed judgment offers only grader-wrong and ambiguous."""
    for calibrator in (None, F(), F("refused"), FJ()):
        assert gs.allowed_dispositions(P(1, 2), calibrator) == gs.DISPOSITIONS_NO_CALIBRATOR
        assert gs.allowed_dispositions(P(1, 2, low_g=True), calibrator) == gs.DISPOSITIONS_NO_CALIBRATOR
        for grader in (F(), F("refused"), FJ()):
            got = gs.allowed_dispositions(grader, calibrator)
            assert got == gs.DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR == ("grader-wrong",
                                                                         "ambiguous-edge-case-add-to-Verified-subset")
            assert "grader-correct" not in got
    # with a calibrator score the sets are unchanged
    assert gs.allowed_dispositions(P(1, 2), P(1, 2)) == gs.DISPOSITIONS_WITH_CALIBRATOR
    for grader in (F(), F("refused"), FJ()):
        assert gs.allowed_dispositions(grader, P(1, 2)) == gs.DISPOSITIONS_GRADER_FAILED
    # the same through the queue items
    units = [unit(1, F()), unit(2, F("refused")), unit(3, FJ()), unit(4, P(1, 1, low_g=True)), unit(5, F()),
             unit(6, P(2, 2))]
    rows = [gf.row(uid(5), units[4]["grader"], F()), gf.row(uid(6), units[5]["grader"], FJ())]
    it = {i["unit_id"]: i for i in gs.queue_items(units, rows)}
    for u in (uid(1), uid(2), uid(3), uid(5)):
        assert it[u]["allowed_dispositions"] == ["grader-wrong", "ambiguous-edge-case-add-to-Verified-subset"]
    for u in (uid(4), uid(6)):
        assert it[u]["allowed_dispositions"] == list(gs.DISPOSITIONS_NO_CALIBRATOR)
        assert "grader-correct" in it[u]["allowed_dispositions"]


def test_queue_tags_and_allowed_dispositions():
    units = [unit(1, P(0, 1)), unit(2, P(1, 1, low_g=True)), unit(3, F()), unit(4, P(2, 2)), unit(5, F("refused")),
             unit(6, P(2, 2)), unit(7, P(2, 2, low_c=True))]
    rows = [gf.row(uid(1), units[0]["grader"], P(2, 1, low_c=True), model=gf.ALPHA),
            gf.row(uid(4), units[3]["grader"], F("refused"), model=gf.BRAVO),
            gf.row(uid(5), units[4]["grader"], P(2, 2), model=gf.CHARLIE),
            gf.row(uid(6), units[5]["grader"], P(2, 3), model=gf.ALPHA)]
    for u, model in zip(units, (gf.ALPHA, gf.DELTA, gf.DELTA, gf.BRAVO, gf.CHARLIE, gf.ALPHA, gf.DELTA)):
        u["model"] = model
    by = {r["uid"]: r for r in rows}
    assert gs.queue_tags(units[0]["grader"], by[uid(1)], DIMS) == [gs.TAG_DISAGREEMENT, gs.TAG_CALIBRATOR_LOW]
    assert gs.queue_tags(units[1]["grader"], None, DIMS) == [gs.TAG_GRADER_LOW]
    assert gs.queue_tags(units[2]["grader"], None, DIMS) == [gs.TAG_GRADER_FAILED]
    assert gs.queue_tags(units[3]["grader"], by[uid(4)], DIMS) == [gs.TAG_CALIBRATOR_FAILED]
    assert gs.queue_tags(units[4]["grader"], by[uid(5)], DIMS) == [gs.TAG_GRADER_FAILED]
    assert gs.queue_tags(units[5]["grader"], by[uid(6)], DIMS) == []
    assert gs.queue_tags(units[6]["grader"], None, DIMS) == [gs.TAG_GRADER_LOW]
    items = gs.queue_items(units, rows)
    assert [i["unit_id"] for i in items] == [uid(1), uid(2), uid(3), uid(4), uid(5), uid(7)]
    it = {i["unit_id"]: i for i in items}
    assert it[uid(1)]["allowed_dispositions"] == list(gs.DISPOSITIONS_WITH_CALIBRATOR)
    # the grader refused and the calibrator parsed: no grader score to be correct about
    assert it[uid(5)]["allowed_dispositions"] == list(gs.DISPOSITIONS_GRADER_FAILED)
    for u in (uid(2), uid(4), uid(7)):
        assert it[u]["allowed_dispositions"] == list(gs.DISPOSITIONS_NO_CALIBRATOR)
    # the grader reply did not parse and no calibrator score exists: never grader-correct
    assert it[uid(3)]["allowed_dispositions"] == list(gs.DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR)
    assert it[uid(2)]["calibrator"] is None and it[uid(4)]["calibrator"]["status"] == "refused"
    assert it[uid(1)]["grader"] == {"status": "parsed", "scores": {G: 0, C: 1}, "low_confidence": {G: False, C: False},
                                    "reasons": {G: "r", C: "r"}, "reply": None, "error": None}
    # a reply that did not parse keeps its raw reply and its error
    assert it[uid(3)]["grader"]["reply"] == "raw text" and it[uid(3)]["grader"]["error"] == "not one JSON object"
    assert it[uid(5)]["grader"]["error"] == F("refused")["error"]
    assert it[uid(4)]["calibrator"]["reply"] == "raw text"
    assert it[uid(5)]["calibrator"]["reply"] is None and it[uid(5)]["calibrator"]["error"] is None
    for i in items:
        for view in (i["grader"], i["calibrator"]):
            assert view is None or set(view) == {"status", "scores", "low_confidence", "reasons", "reply", "error"}
    for i in items:
        assert i["disposition"] is None and i["disposition_note"] is None
        assert i["disposed_by"] is None and i["disposed_on"] is None
        assert i["tags"] == [t for t in gs.TAG_ORDER if t in i["tags"]]
    assert not any(s in json.dumps(items) for s in gf.LEAK_STRINGS)
    counts = gs.queue_counts(items)
    assert counts["items"] == 6
    want = {gs.TAG_DISAGREEMENT: 1, gs.TAG_GRADER_LOW: 2, gs.TAG_CALIBRATOR_LOW: 1, gs.TAG_GRADER_FAILED: 2,
            gs.TAG_CALIBRATOR_FAILED: 1}
    assert {t: counts["by_tag"].get(t, 0) for t in gs.TAG_ORDER} == want


# --- Means --------------------------------------------------------------------------------------

def gen(model, sid, epoch, status="gradeable", kind="standard", defect=False, u=None):
    return {"model": model, "sample_id": sid, "epoch": epoch, "case_kind": kind, "status": status,
            "defect_epoch": defect, "uid": u}


def test_subsets_of():
    assert gs.subsets_of("standard") == ("all", "synthetic")
    assert gs.subsets_of("edge") == ("all", "edge", "difficulty_bearing")
    assert gs.subsets_of("adversarial") == ("all", "adversarial", "difficulty_bearing")
    assert gs.subsets_of(None) == ("all",)


def _means_fixture():
    gens = [gen(M1, "s1", 1, u=uid(1)), gen(M1, "s1", 2, u=uid(2)), gen(M1, "s1", 3, status="errored"),
            gen(M1, "s1", 4, status="unparsed"),
            gen(M1, "s2", 1, kind="edge", u=uid(5)), gen(M1, "s2", 2, kind="edge", u=uid(6)),
            gen(M1, "s2", 3, kind="edge", defect=True, u=uid(7)), gen(M1, "s2", 4, kind="edge", u=uid(8)),
            *[gen(M1, "s3", e, kind="adversarial", u=uid(8 + e)) for e in range(1, 5)]]
    replies = {uid(1): P(3, 0), uid(2): P(1, 2), uid(5): P(2, 1), uid(6): F(), uid(7): P(2, 1), uid(8): P(0, 3),
               **{uid(8 + e): F() for e in range(1, 5)}}
    return gens, replies


def test_means_primary_sensitivity_and_defect_rows():
    gens, replies = _means_fixture()
    out = gs.llm_means(gens, replies)
    allg = out[M1]["all"][G]
    p, s = allg["primary"], allg["sensitivity"]
    assert p["label"] == gs.LABEL_PRIMARY and s["label"] == gs.LABEL_SENSITIVITY
    # the mean of per-sample epoch means: s1 (3, 1) gives 2, s2 (2, 2, 0) gives 4/3
    assert p["mean"] == pytest.approx((2 + 4 / 3) / 2) and p["values"] == 5 and p["samples"] == 2
    assert p["samples_fewer_than_4_epochs"] == 2 and p["ungraded_generations"] == 5
    assert p["samples_with_no_value"] == 1
    # sensitivity: s1 (3, 1, 0, 0) gives 1, s2 (2, 2, 0) gives 4/3
    assert s["mean"] == pytest.approx((1 + 4 / 3) / 2) and s["values"] == 7 and s["samples_fewer_than_4_epochs"] == 1
    assert s["samples_with_no_value"] == 1
    assert allg["primary_defect_dropped"]["mean"] == pytest.approx((2 + 1) / 2)
    assert allg["sensitivity_defect_dropped"]["mean"] == pytest.approx((1 + 1) / 2)
    assert out[M1]["all"][C]["primary"]["mean"] == pytest.approx(((0 + 2) / 2 + (1 + 1 + 3) / 3) / 2)
    syn = out[M1]["synthetic"][G]["primary"]
    assert syn["mean"] == pytest.approx(2.0) and syn["samples"] == 1 and syn["se"] is None
    edge = out[M1]["edge"][G]["primary"]
    assert edge["mean"] == pytest.approx(4 / 3) and edge["samples"] == 1
    assert out[M1]["adversarial"][G]["primary"]["samples_with_no_value"] == 1
    assert out[M1]["difficulty_bearing"][G]["primary"]["samples"] == 1
    # se of the per-sample means: s1 2.0 and s2 4/3
    sd = math.sqrt(((2 - 5 / 3) ** 2 + (4 / 3 - 5 / 3) ** 2) / 1)
    assert p["se"] == pytest.approx(sd / math.sqrt(2))
    assert p["ci95_low"] == pytest.approx(5 / 3 - 1.96 * p["se"]) and p["ci95_high"] == pytest.approx(5 / 3 + 1.96 * p["se"])
    assert p["ci95_clipped"] is False and p["ci_note"] == gs.CI_NOTE


def _anchor_means_method(values_by_sample):
    """The mean of per-sample epoch means with the standard error of those means and a
    normal 95 percent interval clipped to the 0 to 3 scale."""
    per_sample = [sum(vals) / len(vals) for vals in values_by_sample.values() if vals]
    mean = sum(per_sample) / len(per_sample)
    sd = math.sqrt(sum((v - mean) ** 2 for v in per_sample) / (len(per_sample) - 1))
    se = sd / math.sqrt(len(per_sample))
    return mean, se, max(0.0, mean - 1.96 * se), min(3.0, mean + 1.96 * se)


def test_mean_ci_uses_the_standard_error_of_per_sample_epoch_means():
    data = {"a": [3, 2, 3, 3], "b": [1, 1, 2], "c": [0], "d": []}
    got = gs.mean_block(data)
    mean, se, lo, hi = _anchor_means_method(data)
    assert got["mean"] == pytest.approx(mean) and got["se"] == pytest.approx(se)
    assert got["ci95_low"] == pytest.approx(lo) and got["ci95_high"] == pytest.approx(hi)
    assert got["ci95_low"] == 0.0 and got["ci95_clipped"] is True              # the raw lower limit is below 0
    assert got["values"] == 8 and got["samples"] == 3 and got["samples_fewer_than_4_epochs"] == 2
    single = gs.mean_block({"a": [2, 3]})
    assert single["mean"] == pytest.approx(2.5) and single["se"] is None
    assert single["ci95_low"] is None and single["ci95_high"] is None and single["ci95_clipped"] is False
    assert single["ci_note"] == gs.CI_NOTE
    empty = gs.mean_block({"a": []})
    assert empty["mean"] is None and empty["ci95_low"] is None and empty["ci95_clipped"] is False


def test_the_point_estimate_is_the_mean_of_per_sample_epoch_means_with_unequal_epochs():
    got = gs.mean_block({"a": [3, 3, 3, 3], "b": [0]})
    assert got["mean"] == pytest.approx(1.5)                                    # pooled would give 2.4
    assert got["values"] == 5 and got["samples"] == 2 and got["samples_fewer_than_4_epochs"] == 1
    # the point estimate and the paired difference now agree: model B scores 2 and 1 on all epochs
    gens, replies = [], {}
    n = 0
    for model, scores in ((M1, {"s1": [3, 3, 3, 3], "s2": [0]}), (M2, {"s1": [2, 2, 2, 2], "s2": [1, 1, 1, 1]})):
        for sid, vals in scores.items():
            for e, v in enumerate(vals, start=1):
                n += 1
                gens.append(gen(model, sid, e, u=uid(n)))
                replies[uid(n)] = P(v, v)
    means = gs.llm_means(gens, replies)
    assert means[M1]["all"][G]["primary"]["mean"] == pytest.approx(1.5)
    assert means[M2]["all"][G]["primary"]["mean"] == pytest.approx(1.5)
    (entry,) = [x for x in gs.paired_differences(gens, replies)
                if x["subset"] == "all" and x["dim"] == G and x["reading"] == "primary"]
    assert entry["result"]["mean"] == pytest.approx(0.0)


def test_the_normal_interval_is_clipped_to_the_scale():
    wide = gs.mean_block({"a": [3, 3, 3, 3], "b": [0, 0, 0, 0]})
    assert wide["mean"] == pytest.approx(1.5) and wide["se"] == pytest.approx(1.5)
    assert wide["ci95_low"] == 0.0 and wide["ci95_high"] == 3.0 and wide["ci95_clipped"] is True
    assert wide["ci_note"] == "normal 95 percent interval, clipped to the 0 to 3 scale"
    top = gs.mean_block({"a": [3, 3, 3, 3], "b": [3, 3, 3, 2]})
    assert top["ci95_high"] == 3.0 and top["ci95_low"] > 0.0 and top["ci95_clipped"] is True
    narrow = gs.mean_block({"a": [2, 2, 2, 1], "b": [2, 1, 1, 1], "c": [2, 2, 2, 2]})
    assert narrow["ci95_clipped"] is False
    assert 0.0 < narrow["ci95_low"] < narrow["mean"] < narrow["ci95_high"] < 3.0


def test_mean_block_matches_trials_anchor_means():
    from types import SimpleNamespace

    from icfr_agent_evals import reliability as rel
    from icfr_agent_evals import trials

    data = {"a": [2, 2, 3, 2], "b": [1, 1, 2, 1], "c": [2, 1, 2, 2]}       # four values each, inside the scale
    anchor = rel.TRIAL_ANCHORS[0]
    report = SimpleNamespace(samples={sid: SimpleNamespace(epochs=[SimpleNamespace(anchors={anchor: v}) for v in vals])
                                      for sid, vals in data.items()})
    want = trials.anchor_means(report)[anchor]
    got = gs.mean_block(data)
    assert got["ci95_clipped"] is False
    for k in ("mean", "se", "ci95_low", "ci95_high"):
        assert got[k] == pytest.approx(want[k])


def test_paired_differences_use_samples_graded_for_both_models():
    gens, replies = [], {}
    n = 0
    plan = {(M1, "s1"): 3, (M1, "s2"): 2, (M1, "s3"): None, (M2, "s1"): None, (M2, "s2"): 1, (M2, "s3"): 0}
    for (m, sid), score in plan.items():
        for e in range(1, 5):
            n += 1
            gens.append(gen(m, sid, e, u=uid(n)))
            replies[uid(n)] = F() if score is None else P(score, score)
    out = gs.paired_differences(gens, replies)
    hit = [x for x in out if x["subset"] == "all" and x["dim"] == G and x["reading"] == "primary"]
    (entry,) = hit
    assert entry["first"] == M2 and entry["second"] == M1 and entry["samples"] == 1
    assert entry["result"] == jt.paired_mean_difference([(1.0, 2.0)])
    assert entry["label"] == gs.LABEL_PAIRED["primary"]
    assert entry["label"] == "first minus second, samples with a primary value for both models"
    assert {x["reading"] for x in out} == {"primary", "sensitivity"}
    for x in out:
        assert x["label"] == gs.LABEL_PAIRED[x["reading"]]
    assert "format breaks and errored epochs counted as 0" in gs.LABEL_PAIRED["sensitivity"]
    none_shared = [x for x in out if x["subset"] == "edge"]
    assert all(x["samples"] == 0 and x["result"] is None for x in none_shared)


def test_paired_differences_cover_every_model_pair():
    gens, replies = [], {}
    n = 0
    for m in (M3, M1, M2):
        for e in range(1, 5):
            n += 1
            gens.append(gen(m, "s1", e, u=uid(n)))
            replies[uid(n)] = P(1, 1)
    out = gs.paired_differences(gens, replies)
    pairs = {(x["first"], x["second"]) for x in out}
    assert pairs == {(b, a) for a, b in itertools.combinations(sorted((M1, M2, M3)), 2)}


# --- Aggregates and dispositions ----------------------------------------------------------------

def _world():
    gens, units, rows, replies = [], [], [], {}
    n = 0
    for mi, m in enumerate((M1, M2)):
        for si in range(8):
            sid = f"s{si}"
            for e in range(1, 5):
                n += 1
                u = uid(n)
                gens.append(gen(m, sid, e, kind=("standard", "edge")[si % 2], u=u))
                rep = P((si + e) % 4, (si * 2 + e) % 4, low_g=(si == 3 and e == 1))
                replies[u] = rep
                units.append({**unit(n, rep, model=m, sid=sid, epoch=e), "case_kind": ("standard", "edge")[si % 2]})
                if m == M1 and e == 1 + si % 4:
                    cal = P((si + e) % 4, (si * 2 + e + (si == 5) * 2) % 4)
                    rows.append({"uid": u, "model": m, "sample_id": sid, "epoch": e, "position": si + 1,
                                 "grader": rep, "calibrator": cal})
    return units, gens, rows, replies


def test_aggregates_holds_every_sealed_figure():
    units, gens, rows, replies = _world()
    agg = gs.aggregates(units, gens, rows, replies, bootstrap_seed=5, resamples=50)
    assert set(agg) == {"gate", "bootstrap", "breaker", "claims", "means", "paired_differences", "queue_counts",
                        "lost_rows_are_not_dropped_pairs"}
    assert agg["lost_rows_are_not_dropped_pairs"] is True
    assert agg["gate"] == gs.reliability_gate(rows)
    assert agg["bootstrap"] == gs.case_cluster_bootstrap(rows, seed=5, resamples=50)
    assert agg["breaker"] == gs.breaker(units, rows)
    assert agg["queue_counts"] == gs.queue_counts(gs.queue_items(units, rows))
    json.dumps(agg)


def test_no_function_reads_a_disposition():
    units, gens, rows, replies = _world()
    before = gs.aggregates(units, gens, rows, replies, bootstrap_seed=5, resamples=50)
    items_before = gs.queue_items(units, rows)
    filled = {"disposition": "grader-wrong", "disposition_note": "x", "disposed_by": "XX", "disposed_on": "2026-12-01"}
    units2 = [{**u, **filled, "grader": {**u["grader"], **filled}} for u in units]
    rows2 = [{**r, **filled, "grader": {**r["grader"], **filled}, "calibrator": {**r["calibrator"], **filled}}
             for r in rows]
    replies2 = {k: {**v, **filled} for k, v in replies.items()}
    after = gs.aggregates(units2, gens, rows2, replies2, bootstrap_seed=5, resamples=50)
    assert after == before
    items_after = gs.queue_items(units2, rows2)
    assert [i["unit_id"] for i in items_after] == [i["unit_id"] for i in items_before]
    assert all(i["disposition"] is None for i in items_after)


def test_stats_outputs_are_json_safe_with_exact_strings():
    units, gens, rows, replies = _world()
    b = gs.breaker(units, rows)
    for key in ("pair_rate", "generation_rate"):
        if b[key]["rate"] is not None:
            assert Fraction(b[key]["rate_exact"]) == Fraction(b[key]["tagged"], b[key]["denominator"])
    json.dumps(gs.llm_means(gens, replies))
    json.dumps(gs.paired_differences(gens, replies))
