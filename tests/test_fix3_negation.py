"""Fix round 3, lane N2 (2026-11-07): negation, split and exclusion.

Patterns of pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md owned by lane N2:
P1 (split side), P6, P7, P8 (contract C1, narrative.exclusion_spans) and P9. RV
(2026-11-07) reverted P6, P7 and P9 after the R1 red team, with their tests.

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


# --- P8: contract C1, exclusion_spans over whole coordinated lists ---------------------------

C1_COVERED = [
    # S002 to S004 (dev-hc-003-ETH-M1 e1): a colon list after an exclusion head
    ("S002-S004", "None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, "
                  "DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on "
                  "both sides.",
     ["JE-25-11-7405", "DP-2511-680653", "DP-2511-340361"]),
    # S007 to S010 (dev-hc-009-XRP-M3 e1)
    ("S007-S010", "No open item qualifies as a reconciling item: DP-2511-428189 is cleared, DP-2511-679793 relates "
                  "to TREAS01 (out of scope, own funds), DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 "
                  "is a suspense entry.",
     ["DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801"]),
    # S125 to S127: the head names "explain", so it is never covered itself
    ("S125-S127", "No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits "
                  "in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides.",
     ["JE-25-11-7405", "DP-2511-680653", "DP-2511-340361"]),
    # S115 to S116: a non-universal head
    ("S115-S116", "I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
                  "WD-2511-006377 is an ETH item, not USDC.",
     ["JE-25-11-1225", "WD-2511-006377"]),
    # S080 to S082: one negated reliance over a coordinated list
    ("S080-S082", "I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the "
                  "post-cutoff item WD-2511-221933.",
     ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]),
    # S140 to S141: "so neither is relied on"
    ("S140-S141", "JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied "
                  "on; this leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077.",
     ["JE-25-11-1225", "WD-2511-006377"]),
    # S098: "cannot be quantified or relied on"
    ("S098", "DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is an eligible type, but its amount is blank in "
             "both the open-items schedule and the ledger extract, and no matching on-chain transaction appears, so "
             "it cannot be quantified or relied on.",
     ["DP-2511-582345"]),
    # S020 and S021 (dev-std-004-SOL-M1 e1): a run of item sentences after a universal head
    ("S020-S021", "None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense "
                  "clearing entry. WD-2511-781009 sits on the out-of-scope staking-rewards wallet. WD-2512-786753 was "
                  "booked and broadcast after the cutoff. DP-2511-983545 is a USDC movement, not SOL.",
     ["JE-25-11-2733", "WD-2511-781009", "WD-2512-786753", "DP-2511-983545"]),
    # S048 (dev-hc-008-USDC-M3 e2)
    ("S048", "No open item can be relied on. DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) has no amount in "
             "the schedule or the ledger extract and no matching chain transaction, so it cannot be quantified.",
     ["DP-2511-582345"]),
    # paraphrases
    ("para-terse", "None of the schedule items qualify: JE-25-11-7405, a manual entry; DP-2511-340361, cleared.",
     ["JE-25-11-7405"]),
    ("para-were-also", "The cleared deposit DP-2511-340361 and the manual adjustment JE-25-11-7405 were also "
                       "excluded.",
     ["DP-2511-340361", "JE-25-11-7405"]),
]


@pytest.mark.parametrize("row,text,refs", C1_COVERED)
def test_c1_widened_spans_cover_every_reference_in_the_list(row, text, refs):
    for ref in refs:
        assert covered(text, ref), (row, ref, [text[a:b] for a, b in exclusion_spans(text)])


@pytest.mark.parametrize("text,needle", [
    # S049 (dev-hc-009-XRP-M3 e2): the consequence clause with the totals stays outside
    ("No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is unexplained: "
     "DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 is dated after the "
     "cutoff, and JE-25-11-1801 is a suspense entry.", "so $0.00 is explained and the full $3,397,826.64 is unexplained"),
    ("No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment.", "No open item is eligible to "
                                                                                     "explain it"),
    ("None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense clearing "
     "entry. The variance stays unexplained.", "The variance stays unexplained"),
])
def test_c1_core_clauses_are_never_covered(text, needle):
    assert not touched(text, needle), [text[a:b] for a, b in exclusion_spans(text)]


def test_c1_s049_list_is_covered_after_the_consequence_clause():
    text = ("No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is "
            "unexplained: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 "
            "is dated after the cutoff, and JE-25-11-1801 is a suspense entry.")
    for ref in ("DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801"):
        assert covered(text, ref), ref


C1_PLANTED = [
    # a list member that states reliance: the list is not widened
    ("None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, and DP-2511-340361 "
     "explains the shortfall.", "DP-2511-340361"),
    ("None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, and DP-2511-340361 "
     "accounts for part of it.", "DP-2511-340361"),
    ("I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
     "WD-2511-006377 is relied on.", "JE-25-11-1225"),
    # a head whose exclusion is negated by its quantifier
    ("No item is excluded: JE-25-11-7405 is a manual adjustment and DP-2511-340361 was cleared.", "DP-2511-340361"),
    # a head that is a core clause and not a universal item head
    ("The variance is excluded from M2: JE-25-11-7405 is a manual adjustment.", "JE-25-11-7405"),
    # a clause with its own subject between the head and the colon
    ("No open item qualifies as a reconciling item, and three items explain the shortfall: DP-2511-340361 is a "
     "deposit.", "DP-2511-340361"),
    ("No open item qualifies, so they explain the shortfall: DP-2511-340361 is a deposit.", "DP-2511-340361"),
    # a run sentence that relies on the item, and a head that is not universal
    ("None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 explains $1,095.35 of it.",
     "JE-25-11-2733"),
    ("None of the listed items is excluded. JE-25-11-2733 is a suspense clearing entry.", "JE-25-11-2733"),
    # the run ends at the first sentence that does not open with a reference
    ("None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense clearing entry. "
     "The shortfall is explained by WD-2511-781009.", "WD-2511-781009"),
]


@pytest.mark.parametrize("text,ref", C1_PLANTED)
def test_c1_planted_lists_are_not_widened(text, ref):
    assert not covered(text, ref), [text[a:b] for a, b in exclusion_spans(text)]


@pytest.mark.parametrize("text", [
    "No item is excluded.",
    "None of the items were disregarded.",
    "DP-2511-340361 sits in no out-of-scope wallet.",
])
def test_c1_quantifier_negates_a_positive_exclusion_predicate(text):
    assert exclusion_spans(text) == []


def test_c1_quantifier_does_not_cross_a_conjunction():
    text = "DP-2511-582345 has no amount and is excluded."
    assert covered(text, "DP-2511-582345")


def test_c1_masked_text_gives_the_same_widened_spans():
    text = C1_COVERED[0][1]
    masked = text
    for ref in C1_COVERED[0][2]:
        masked = masked.replace(ref, "\ue001" * len(ref))
    assert exclusion_spans(masked) == exclusion_spans(text)
    run_text = C1_COVERED[7][1]
    masked = run_text
    for ref in C1_COVERED[7][2]:
        masked = masked.replace(ref, "\ue001" * len(ref))
    assert exclusion_spans(masked) == exclusion_spans(run_text)


def test_c1_spans_stay_sorted_disjoint_and_in_range():
    for _row, text, _refs in C1_COVERED:
        spans = exclusion_spans(text)
        assert spans == sorted(spans)
        assert all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))
        assert all(0 <= a < b <= len(text) for a, b in spans)
