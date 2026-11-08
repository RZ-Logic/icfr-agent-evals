"""Fix round 3, lane N2 (2026-11-07): negation, split and exclusion.

Patterns of pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md owned by lane N2:
P1 (split side), P6, P7, P8 (contract C1, narrative.exclusion_spans) and P9. RV
(2026-11-07) reverted P6, P7, P8 (the C1 widening) and P9 after the R1 red team,
with their tests; P1 (split side) and its tests stay.

Every rule carries the real Sonnet sentence (quoted from sonnet-routes-labeled.json,
with its row id), one paraphrase in another register, and one or more planted-error
twins that must still fail with the right FM or stay routed. Truths are each pilot
case's own answer key from the dev split; no model is called.

A cleared route here is a pass-proof: the code binds the band, asset, item or figure
to one role and checks it against the truth record. The amount checks of a sentence
are lane N1's, so tests of the split side look only at split, band and asset verdicts.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from casefactory import good_output

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset, select_subset
from icfr_agent_evals.narrative import check_narrative_consistency, exclusion_spans


@lru_cache(maxsize=1)
def _truths() -> dict:
    samples = select_subset(list(build_dataset(split="dev")), "dev", "pilot")
    out = {}
    for s in samples:
        t = truth_of(s.metadata)
        assert t.split == "dev"
        out[str(s.id)] = t
    return out


def truth(case_id: str):
    return _truths()[case_id]


def run(case_id: str, text: str, extra_items: list[dict] | None = None):
    t = truth(case_id)
    out = good_output(t, narrative=text)
    if extra_items:
        out["reconciling_items"] = list(out["reconciling_items"]) + extra_items
    return check_narrative_consistency(out, t)


def kinds(result, kind: str) -> list[dict]:
    return [e for e in result.extractions if e["kind"] == kind]


def reasons_with(result, *prefixes: str) -> list[str]:
    return [r for r in result.review_reasons if r.startswith(prefixes)]


def covered(text: str, needle: str) -> bool:
    """True when every occurrence of needle lies inside one exclusion span."""
    spans = exclusion_spans(text)
    start = text.find(needle)
    assert start >= 0, needle
    while start >= 0:
        end = start + len(needle)
        if not any(a <= start and end <= b for a, b in spans):
            return False
        start = text.find(needle, end)
    return True


def touched(text: str, needle: str) -> bool:
    """True when any character of the first occurrence of needle is inside a span."""
    start = text.find(needle)
    assert start >= 0, needle
    end = start + len(needle)
    return any(a < end and start < b for a, b in exclusion_spans(text))


HC1 = "dev-hc-001-BTC-M2"
HC2 = "dev-hc-002-USDT-M2"
HC3 = "dev-hc-003-ETH-M1"
HC8 = "dev-hc-008-USDC-M3"
HC9 = "dev-hc-009-XRP-M3"
HC14 = "dev-hc-014-USDC-M1"
STD4 = "dev-std-004-SOL-M1"
STD8 = "dev-std-008-USDC-M1"
STD10 = "dev-std-010-BTC-M2"
STD12 = "dev-std-012-SOL-M3"
STD13 = "dev-std-013-USDC-M2"


# --- P1 split side: "Explained variance is $0.00, so the full $X is unexplained" -----------
# Pass-proof: the label binds the figure after its copula, that figure equals the
# explained portion of the truth and of the cited items, and a paired figure bound to an
# unexplained cue equals the unexplained portion of both.

P1_CLEARED = [
    ("S016-S017", STD4, "Explained variance is $0.00, so the full $4,316.25 is unexplained and remains with the "
                        "open ops ticket; the workpaper has not been reviewed."),
    ("S046-S047", HC8, "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained."),
    ("S059-S060", STD4, "Explained variance is therefore $0.00 and the full $4,316.25 remains unexplained."),
    ("S096-S097", HC8, "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained, with the root "
                       "cause still open in the preparer's notes and the workpaper not yet reviewed."),
    ("para", HC8, "The explained variance comes to $0.00 and $3,691,252.88 is still unexplained."),
]


@pytest.mark.parametrize("row,case_id,sentence", P1_CLEARED)
def test_p1_portion_label_with_matching_pair_clears_the_split_route(row, case_id, sentence):
    r = run(case_id, sentence)
    assert "FM-001" not in r.failure_modes
    assert reasons_with(r, "split_") == [], r.review_reasons
    splits = kinds(r, "split_claim")
    assert splits and all(c["verdict"] == "ok" and c.get("resolution") == "portion_label" for c in splits)


@pytest.mark.parametrize("sentence", [
    # the two figures swapped
    "Explained variance is $4,316.25, so the full $0.00 is unexplained.",
    # the paired unexplained figure wrong
    "Explained variance is $0.00, so the full $4,000.00 is unexplained.",
    # no paired figure at all
    "Explained variance is $0.00.",
])
def test_p1_planted_split_twins_stay_routed_or_fail(sentence):
    r = run(STD4, sentence)
    assert "FM-001" in r.failure_modes or reasons_with(r, "split_")
    assert not any(c.get("resolution") == "portion_label" for c in kinds(r, "split_claim"))


def test_p1_label_needs_the_cited_items_to_agree_with_the_truth():
    # dev-std-010-BTC-M2 is fully explained by two eligible items ($66,311.24)
    text = "Explained variance is $66,311.24 and $0.00 remains unexplained."
    t = truth(STD10)
    full = good_output(t, narrative=text)
    r = check_narrative_consistency(full, t)
    assert any(c.get("resolution") == "portion_label" and c["verdict"] == "ok" for c in kinds(r, "split_claim"))
    # the same sentence when the answer cites only one of the two items: the model's own
    # explained portion is not $66,311.24, so there is no proof
    short = dict(full, reconciling_items=list(full["reconciling_items"])[:1])
    r = check_narrative_consistency(short, t)
    assert not any(c.get("resolution") == "portion_label" for c in kinds(r, "split_claim"))
