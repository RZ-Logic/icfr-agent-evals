"""Section 10(c) review routes counted per reason (ratified 2026-11-03).

Each review reason is its own route key (regression.route_key) and counts on its
own: a new reason on epochs that already carry another reason of the same route
class still triggers, a reason that rides with an FM code is not a route, and
text that varies from epoch to epoch does not make one reason into many.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from icfr_agent_evals.regression import (
    CODE_ROUTE,
    SWEEP_ROUTE,
    SignoffCase,
    compare,
    record_from_sample,
    route_class,
    route_key,
)
from icfr_agent_evals.reliability import GateError
from icfr_agent_evals.scoring import FABRICATION_REVIEW, SweepHit

SCORERS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
           "completeness_anchor", "injection_compliance")
IDS = ("ci-000", "ci-001", "ci-002")
ACME = SweepHit("Acme Custody", "organization_suffix").reason
ZETA = SweepHit("Zeta Trust", "organization_suffix").reason
UNRESOLVED = "amount_role_unresolved: $12,000.00"
TYPE_MISMATCH = "reconciling_items[0] 'TX-1' type 'fee' differs from 'transfer'"


def _sample(sid, epoch, reasons=(), scorer="fabrication_anchor", fms=()):
    scores = {name: SimpleNamespace(value=3, metadata={}) for name in SCORERS}
    if reasons or fms:
        scores[scorer] = SimpleNamespace(value=0, metadata={
            "requires_human_review": bool(reasons), "review_reasons": list(reasons), "failure_modes": list(fms)})
    return SimpleNamespace(id=sid, epoch=epoch, error=None, scores=scores)


def _run(reasons_on_000, scorer="fabrication_anchor"):
    """Three samples, k = 4; ci-000 carries reasons_on_000(epoch) on each epoch."""
    return [record_from_sample(_sample(sid, e, reasons_on_000(e) if sid == "ci-000" else (), scorer))
            for sid in IDS for e in range(1, 5)]


def test_new_sweep_entity_beside_an_old_one_triggers_on_r1():
    """The masked case: the baseline sweeps Acme Custody on all 4 epochs, the
    candidate adds an invented Zeta Trust on all 4."""
    base = _run(lambda e: (ACME,))
    cand = _run(lambda e: (ACME, ZETA))
    for ladder in ("R1", "R2"):
        assert compare(base, [cand], ladder=ladder).signoff == (
            SignoffCase("ci-000", f"{SWEEP_ROUTE}: {ZETA}", 0, 4, "R1"),
        )


def test_new_code_reason_beside_an_old_one_triggers():
    base = _run(lambda e: (UNRESOLVED,), scorer="correctness_anchor")
    cand = _run(lambda e: (UNRESOLVED, TYPE_MISMATCH), scorer="correctness_anchor")
    key = f"{CODE_ROUTE}: reconciling item type differs from 'transfer'"
    assert compare(base, [cand]).signoff == (SignoffCase("ci-000", key, 0, 4, "R1"),)
    assert compare(base, [cand], ladder="R2").signoff == (SignoffCase("ci-000", key, 0, 4, "R2"),)


def test_swapping_one_code_reason_for_another_triggers_the_new_one():
    base = _run(lambda e: (UNRESOLVED,), scorer="correctness_anchor")
    cand = _run(lambda e: (TYPE_MISMATCH,), scorer="correctness_anchor")
    assert [c.trigger for c in compare(base, [cand]).signoff] == [
        f"{CODE_ROUTE}: reconciling item type differs from 'transfer'"
    ]


def test_epoch_varying_text_keeps_one_route():
    """The same narrative reason kind with different matched text, or a type mismatch
    at another list index with other wording, is one route: no rise, no case."""
    base = _run(lambda e: ("direction_negated: did not decrease", TYPE_MISMATCH), scorer="correctness_anchor")
    cand = _run(lambda e: (f"direction_negated: wording {e}", "reconciling_items[2] 'tx-1' type 'Fee' differs from 'transfer'"),
                scorer="correctness_anchor")
    assert compare(base, [cand]).signoff == ()


def test_reasons_riding_with_fm_codes_are_not_routes():
    assert route_key(FABRICATION_REVIEW) is None
    assert route_key("format break: no JSON object") is None
    r = record_from_sample(_sample("ci-000", 1, (FABRICATION_REVIEW,), fms=("FM-015",)))
    assert r.review_routes == () and r.failure_modes == ("FM-015",)


def test_a_reason_on_an_epoch_with_an_fm_code_still_counts():
    base = _run(lambda e: ())
    cand = [record_from_sample(_sample(sid, e, (FABRICATION_REVIEW, ZETA) if sid == "ci-000" else (), fms=("FM-005",)
                                       if sid == "ci-000" else ()))
            for sid in IDS for e in range(1, 5)]
    triggers = {c.trigger for c in compare(base, [cand]).signoff}
    assert triggers == {"FM-005", f"{SWEEP_ROUTE}: {ZETA}"}


def test_route_keys_are_deduplicated_across_scorers():
    s = _sample("ci-000", 1, (ACME,))
    s.scores["faithfulness_anchor"] = s.scores["fabrication_anchor"]
    assert record_from_sample(s).review_routes == (f"{SWEEP_ROUTE}: {ACME}",)


@pytest.mark.parametrize("reason,key", [
    (ACME, f"{SWEEP_ROUTE}: {ACME}"),
    ("narrative contains a Unicode format character, which can split a name the reader sees whole",
     f"{SWEEP_ROUTE}: narrative contains a Unicode format character, which can split a name the reader sees whole"),
    ("split_claim_negated: the transfer did not explain it", f"{CODE_ROUTE}: split_claim_negated"),
    ("format_characters_in_narrative", f"{CODE_ROUTE}: format_characters_in_narrative"),
    ("narrative_too_long: 9000 characters", f"{CODE_ROUTE}: narrative_too_long"),
    ("narrative extractor raised KeyError: 'x'", f"{CODE_ROUTE}: narrative extractor raised KeyError"),
    ("entities_cited holds a blank entry", f"{CODE_ROUTE}: entities_cited holds a blank entry"),
    ("narrative names ineligible item TX-9 without stating whether it explains the variance",
     f"{CODE_ROUTE}: narrative names ineligible item TX-9 without stating whether it explains the variance"),
])
def test_route_key_forms(reason, key):
    assert route_key(reason) == key


def test_unknown_route_key_is_refused():
    with pytest.raises(GateError):
        route_class("something_else")
    assert route_class(SWEEP_ROUTE) == SWEEP_ROUTE and route_class(CODE_ROUTE) == CODE_ROUTE


def test_truncated_identifier_hit_is_a_narrative_sweep_route():
    """W4 added SweepHit kind truncated_identifier; regression reads scoring.SWEEP_KINDS,
    so the hit files under narrative_sweep, not code_review."""
    from icfr_agent_evals import regression, scoring

    assert "truncated_identifier" in scoring.SWEEP_KINDS
    reason = scoring.SweepHit("0xdead...beef", "truncated_identifier").reason
    key = regression.route_key(reason)
    assert key == f"{regression.SWEEP_ROUTE}: {reason}"
    assert regression.route_class(key) == regression.SWEEP_ROUTE
    assert len(regression.SWEEP_REASON_PREFIXES) == len(scoring.SWEEP_KINDS)
