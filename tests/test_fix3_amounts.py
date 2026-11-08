"""Fix round 3 (2026-11-07), lane N1: amount pass-proofs (patterns P1 amount side, P2,
P3, P4 and P5 of pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md).

A route on an amount clears only when code proves the figure and its direction
against the truth record. Every rule is tested with the real Sonnet sentence (quoted
from the development set, with its row id), a paraphrase in another register, and
planted-error twins that must still fire FM-001 or keep a route (verdict never "ok").
Truth is the dev pilot case (build_dataset(split="dev", subset="pilot")); the parsed
answer cites the references the Sonnet epoch cited. No model is called.

P1: "Explained variance is $0.00, so the full $4,316.25 is unexplained": the figure
    bound by the explained cue heading the variance cue equals the explained portion
    (cited items and truth), and a paired figure bound to the unexplained cue equals
    the unexplained portion.
P2: one item's amount beside a direction phrase from a closed lexicon ("adds $X to the
    surplus", "lowers reserves against the ledger by $X", "puts reserves $X below the
    ledger", "offsets it by $X"): the figure equals the one item named in its segment
    (or exactly one cited eligible item when none is named), and the phrase's sign in
    the frame of on-chain minus ledger equals the item's truth effect. "It" and "that"
    take the sign of their antecedent in the same or previous sentence.
P3: a sum of named items beside a direction phrase: the figure equals the sum of every
    item named in the sentence, all cited, eligible and of the phrase's sign.
P4: a net or aggregate explained total: it equals the truth's explained portion and the
    signed sum of the cited items, an explain word binds it or the rest of its clause
    itemizes every cited item under signed heads, and nothing in the clause contradicts.
P5: item amounts listed under a direction head ("Three widen the shortfall: ..."): the
    count word equals the number of figures, each figure is exactly one cited eligible
    item, item nouns agree, and every item's truth effect has the head's sign.
"""

from __future__ import annotations

import pytest

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.narrative import check_narrative_consistency

TRUTH = {s.id: truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}

# the references each Sonnet epoch cited in reconciling_items (all eligible; none for
# the two cases whose epochs cited no item)
CITED = {
    "dev-hc-001-BTC-M2": ["WD-2510-774925", "OI-251031-209", "DP-2510-070769", "DP-2510-503229"],
    "dev-hc-008-USDC-M3": [],
    "dev-hc-013-ETH-M2": ["WD-2510-571069", "WD-2510-818869", "TRF-2510-05457"],
    "dev-std-004-SOL-M1": [],
    "dev-std-005-ETH-M2": ["DP-2511-648193", "DP-2511-893641"],
    "dev-std-007-ETH-M3": ["DP-2510-146805", "WD-2510-333749", "DP-2510-734749", "DP-2510-960213", "OI-251031-017"],
    "dev-std-012-SOL-M3": ["DP-2512-493757", "DP-2512-052149"],
}


def answer(case_id: str, narrative: str, refs: list[str] | None = None) -> dict:
    t = TRUTH[case_id]
    refs = CITED[case_id] if refs is None else refs
    return {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": list(refs),
        "reconciling_items": [
            {"reference": r, "amount_usd": t.item_by_reference(r).amount_usd, "type": "timing difference"}
            for r in refs
        ],
        "narrative": narrative,
    }


def run(case_id: str, narrative: str, refs: list[str] | None = None):
    return check_narrative_consistency(answer(case_id, narrative, refs), TRUTH[case_id])


def amount(result, text: str) -> dict:
    """The last amount extraction with this text (a context sentence may repeat it)."""
    hits = [x for x in result.extractions if x.get("kind") == "amount" and x["text"] == text]
    assert hits, f"no amount extraction {text!r}: {[x.get('text') for x in result.extractions]}"
    return hits[-1]


def assert_proven(result, amt: str) -> dict:
    x = amount(result, amt)
    assert x["verdict"] == "ok", x
    assert "FM-001" not in result.failure_modes, result.extractions
    assert not [r for r in result.review_reasons if amt in r], result.review_reasons
    return x


def assert_not_passed(result, amt: str) -> dict:
    """A planted error fails with FM-001 or keeps a route on the figure; never ok."""
    x = amount(result, amt)
    assert x["verdict"] != "ok", x
    assert x["verdict"] == "FM-001" or any(amt in r for r in result.review_reasons), (x, result.review_reasons)
    return x


def test_dev_truth_matches_the_devset_cases():
    # the amounts and signs these tests rely on, so a moved dev set fails loudly here
    t = TRUTH["dev-std-004-SOL-M1"]
    assert (t.explained_usd, t.unexplained_usd) == (0.0, -4316.25)
    t = TRUTH["dev-hc-008-USDC-M3"]
    assert (t.explained_usd, t.unexplained_usd) == (0.0, 3691252.88)
    assert abs(TRUTH["dev-std-007-ETH-M3"].explained_usd - 1_880_341.85) < 0.005
    assert abs(TRUTH["dev-std-012-SOL-M3"].explained_usd - 342_165.48) < 0.005
    for case_id, refs in CITED.items():
        t = TRUTH[case_id]
        assert all(t.eligibility(t.item_by_reference(r)).eligible for r in refs), case_id


# --- P1: explained or unexplained portion beside the word "variance" ------------------------
# (row id, case, sentence, amount text)
P1_REAL = [
    ("S016", "dev-std-004-SOL-M1",
     "Explained variance is $0.00, so the full $4,316.25 is unexplained and remains with the open ops ticket; "
     "the workpaper has not been reviewed.", "$0.00"),
    ("S059", "dev-std-004-SOL-M1",
     "Explained variance is therefore $0.00 and the full $4,316.25 remains unexplained.", "$0.00"),
    ("S046", "dev-hc-008-USDC-M3",
     "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained.", "$0.00"),
    ("S096", "dev-hc-008-USDC-M3",
     "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained, with the root cause still "
     "open in the preparer's notes and the workpaper not yet reviewed.", "$0.00"),
]
P1_PARAPHRASE = [
    ("p1-terse", "dev-std-004-SOL-M1", "Explained variance stands at $0.00, with $4,316.25 unexplained.", "$0.00"),
    ("p1-reordered", "dev-hc-008-USDC-M3",
     "With no item relied on, explained variance came to $0.00 and all $3,691,252.88 is left unexplained.",
     "$0.00"),
]
P1_TWINS = [
    # the paired figure is called explained, not unexplained
    ("p1-twin-pair-role", "dev-std-004-SOL-M1",
     "Explained variance is $0.00, so the full $4,316.25 is explained.", "$0.00"),
    # the paired figure is wrong
    ("p1-twin-pair-amount", "dev-hc-008-USDC-M3",
     "Explained variance is $0.00 and the full $3,961,252.88 remains unexplained.", "$0.00"),
    # the explained figure is wrong
    ("p1-twin-amount", "dev-std-004-SOL-M1",
     "Explained variance is $4,000.00, so the full $4,316.25 is unexplained.", "$4,000.00"),
    # negated
    ("p1-twin-negated", "dev-std-004-SOL-M1",
     "Explained variance is not $0.00, and the full $4,316.25 remains unexplained.", "$0.00"),
    # no paired figure: the portion is not proven, the route stands
    ("p1-unpaired", "dev-hc-008-USDC-M3", "Explained variance is $0.00.", "$0.00"),
]


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P1_REAL + P1_PARAPHRASE,
                         ids=[r[0] for r in P1_REAL + P1_PARAPHRASE])
def test_p1_portion_figure_is_proven(row, case_id, sentence, amt):
    x = assert_proven(run(case_id, sentence), amt)
    assert (x["resolution"], x["portion"]) == ("portion_compound", "explained"), x


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P1_TWINS, ids=[r[0] for r in P1_TWINS])
def test_p1_planted_error_is_not_passed(row, case_id, sentence, amt):
    assert_not_passed(run(case_id, sentence), amt)


def test_p1_wrong_explained_figure_fires_fm001():
    result = run("dev-std-004-SOL-M1", "Explained variance is $4,000.00, so the full $4,316.25 is unexplained.")
    assert amount(result, "$4,000.00")["verdict"] == "FM-001"
    assert "FM-001" in result.failure_modes


# --- P2: one item's amount beside a direction phrase ------------------------------------------
S022 = ("DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on "
        "12/1, which lowers reserves against the ledger by USD 256,238.97.")
S023 = ("DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1, which "
        "raises reserves against the ledger by USD 52,300.36.")
S028 = ("Two eligible hot-1 items were open at cutoff: DP-2512-493757 was on chain only and adds $434,042.88 to "
        "the surplus, while DP-2512-052149 was ledger only (credited 23:48 on Dec 31, on chain after cutoff) and "
        "offsets it by $91,877.40.")
S061 = ("DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on "
        "12/1, which puts reserves USD 256,238.97 below the ledger.")
S062 = ("DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which "
        "puts reserves USD 52,300.36 above the ledger.")
S119 = "The DP-2512-493757 deposit, on chain but not yet credited in the ledger, adds $434,042.88 to the surplus."
S120 = ("The DP-2512-052149 deposit, ledger-only at cutoff, offsets it by $91,877.40, so the net explained portion "
        "is $342,165.48.")
S138 = ("DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which "
        "offsets that by USD 52,300.36.")
S145 = "DP-2512-493757 is a chain-only deposit that adds $434,042.88 to the surplus."
S146 = ("DP-2512-052149 is a ledger-only deposit that offsets it by $91,877.40, so the net explained amount is "
        "$342,165.48.")

P2_REAL = [
    ("S022", "dev-std-005-ETH-M2", S022, "USD 256,238.97"),
    ("S023", "dev-std-005-ETH-M2", S023, "USD 52,300.36"),
    ("S028", "dev-std-012-SOL-M3", S028, "$434,042.88"),
    ("S029", "dev-std-012-SOL-M3", S028, "$91,877.40"),  # "it" resolves inside the same sentence
    ("S061", "dev-std-005-ETH-M2", S061, "USD 256,238.97"),
    ("S062", "dev-std-005-ETH-M2", S062, "USD 52,300.36"),
    ("S119", "dev-std-012-SOL-M3", S119, "$434,042.88"),
    ("S120", "dev-std-012-SOL-M3", S119 + " " + S120, "$91,877.40"),  # "it": the previous sentence
    ("S138", "dev-std-005-ETH-M2", S022 + " " + S138, "USD 52,300.36"),
    ("S145", "dev-std-012-SOL-M3", S145, "$434,042.88"),
    ("S146", "dev-std-012-SOL-M3", S145 + " " + S146, "$91,877.40"),
]
P2_PARAPHRASE = [
    ("p2-relative", "dev-std-005-ETH-M2",
     "DP-2511-648193, ledger-credited on 11/30 and settled on chain on 12/1, reduces on-chain reserves relative "
     "to the ledger by USD 256,238.97.", "USD 256,238.97"),
    ("p2-british", "dev-std-005-ETH-M2",
     "DP-2511-893641, received on chain before cut-off but posted to the ledger afterwards, raises reserves "
     "relative to the ledger by USD 52,300.36.", "USD 52,300.36"),
    ("p2-terse", "dev-std-012-SOL-M3", "Chain-only deposit DP-2512-493757 contributes $434,042.88 to the surplus.",
     "$434,042.88"),
    ("p2-anaphor-variance", "dev-std-012-SOL-M3",
     "The surplus is $735,343.35. DP-2512-052149, booked in the ledger only, reduces it by $91,877.40.",
     "$91,877.40"),
]
P2_TWINS = [
    # the direction word flipped, one twin per real sentence
    ("p2-twin-S022-flip", "dev-std-005-ETH-M2", S022.replace("lowers", "raises"), "USD 256,238.97"),
    ("p2-twin-S023-flip", "dev-std-005-ETH-M2", S023.replace("raises", "lowers"), "USD 52,300.36"),
    ("p2-twin-S028-flip", "dev-std-012-SOL-M3", S028.replace("to the surplus", "to the shortfall"), "$434,042.88"),
    ("p2-twin-S029-flip", "dev-std-012-SOL-M3", S028.replace("offsets it", "widens it"), "$91,877.40"),
    ("p2-twin-S061-flip", "dev-std-005-ETH-M2", S061.replace("below the ledger", "above the ledger"),
     "USD 256,238.97"),
    ("p2-twin-S062-flip", "dev-std-005-ETH-M2", S062.replace("above the ledger", "below the ledger"),
     "USD 52,300.36"),
    ("p2-twin-S119-flip", "dev-std-012-SOL-M3", S119.replace("adds", "lowers").replace(" to the surplus.",
                                                                                      " from the surplus."),
     "$434,042.88"),
    ("p2-twin-S120-flip", "dev-std-012-SOL-M3", S119 + " " + S120.replace("offsets it", "deepens it"),
     "$91,877.40"),
    ("p2-twin-S138-flip", "dev-std-005-ETH-M2", S022 + " " + S138.replace("offsets that", "widens that"),
     "USD 52,300.36"),
    ("p2-twin-S145-flip", "dev-std-012-SOL-M3", S145.replace("to the surplus", "to the deficit"), "$434,042.88"),
    ("p2-twin-S146-flip", "dev-std-012-SOL-M3", S145 + " " + S146.replace("offsets it", "adds to it"),
     "$91,877.40"),
    # the wrong amount for the named item, and the other item's amount
    ("p2-twin-amount", "dev-std-005-ETH-M2", S022.replace("256,238.97", "256,283.97"), "USD 256,283.97"),
    ("p2-twin-other-item", "dev-std-012-SOL-M3", S145.replace("$434,042.88", "$91,877.40"), "$91,877.40"),
    # the item noun disagrees with the item (DP-2512-493757 is a deposit)
    ("p2-twin-kind", "dev-std-012-SOL-M3", S119.replace("deposit", "withdrawal"), "$434,042.88"),
    # a contrast moves the effect off the named item
    ("p2-twin-contrast", "dev-std-012-SOL-M3",
     "Unlike DP-2512-493757, the other deposit adds $434,042.88 to the surplus.", "$434,042.88"),
    # negated and hypothetical effects
    ("p2-twin-negated", "dev-std-012-SOL-M3", "DP-2512-493757 does not add $434,042.88 to the surplus.",
     "$434,042.88"),
    ("p2-twin-hypothetical", "dev-std-012-SOL-M3",
     "DP-2512-493757 would add $434,042.88 to the surplus if it were credited.", "$434,042.88"),
    # an excluded item
    ("p2-twin-excluded", "dev-std-012-SOL-M3",
     "DP-2512-493757, which is excluded, adds $434,042.88 to the surplus.", "$434,042.88"),
    # a ledger-side object and a resulting position are not in the lexicon
    ("p2-twin-ledger-side", "dev-std-005-ETH-M2",
     S023.replace("raises reserves against the ledger", "lowers ledger liabilities"), "USD 52,300.36"),
    ("p2-twin-position", "dev-std-005-ETH-M2",
     S023.replace("raises reserves against the ledger", "raises reserves above the ledger"), "USD 52,300.36"),
    # the item is not cited by the answer
    ("p2-twin-uncited", "dev-std-012-SOL-M3", S145, "$434,042.88", ["DP-2512-052149"]),
    # "it" whose antecedents disagree in sign (the surplus and a surplus-reducing item)
    ("p2-twin-anaphor-mixed", "dev-std-012-SOL-M3",
     "The surplus is partly offset by DP-2512-052149. DP-2512-493757, recorded on chain only at the cutoff, adds "
     "to it by $434,042.88.", "$434,042.88"),
    # an unnamed item whose amount belongs to a different item than the one just named
    ("p2-twin-unnamed-after-ref", "dev-std-007-ETH-M3",
     "DP-2510-146805 is a chain-only deposit. The deposit adds USD 675,392.16 to the surplus.", "USD 675,392.16"),
]


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P2_REAL + P2_PARAPHRASE,
                         ids=[r[0] for r in P2_REAL + P2_PARAPHRASE])
def test_p2_item_effect_is_proven(row, case_id, sentence, amt):
    x = assert_proven(run(case_id, sentence), amt)
    assert x["resolution"] == "item_effect_proven_item", x


@pytest.mark.parametrize("row", P2_TWINS, ids=[r[0] for r in P2_TWINS])
def test_p2_planted_error_is_not_passed(row):
    _id, case_id, sentence, amt, *refs = row
    assert_not_passed(run(case_id, sentence, refs[0] if refs else None), amt)


def test_p2_wrong_item_amount_still_fires_fm001():
    result = run("dev-std-005-ETH-M2", S022.replace("256,238.97", "256,283.97"))
    assert amount(result, "USD 256,283.97")["verdict"] == "FM-001"


def test_p2_anaphor_without_antecedent_keeps_its_route():
    # a correct sentence, but "it" has nothing to resolve to, so the direction is unproven
    result = run("dev-std-012-SOL-M3", "DP-2512-052149 offsets it by $91,877.40.")
    assert amount(result, "$91,877.40")["verdict"] == "review"


def test_p2_proof_reaches_the_scorer():
    from icfr_agent_evals.scoring import check_correctness

    result = check_correctness(answer("dev-std-005-ETH-M2", S022), TRUTH["dev-std-005-ETH-M2"])
    assert result.failure_modes == (), result
    assert not [r for r in result.review_reasons if "256,238.97" in r], result.review_reasons


# --- P3: a sum of named items beside a direction phrase ---------------------------------------
S012 = ("Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger "
        "before cutoff but reached the chain only on 11/01, which adds $28,749.96 to the surplus.")
S053 = ("Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only after it "
        "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus by $28,749.96.")
S133 = ("Two Hot 1 withdrawals were booked in the ledger before cutoff but reached the chain only on 11/01 "
        "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which adds $28,749.96 to the surplus.")
S110 = ("Open hot-1 items at cutoff increase the surplus by USD 2,415,015.48: chain-only deposits DP-2510-146805 "
        "(USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal WD-2510-333749 "
        "(USD 946,484.06).")
P3_REAL = [
    ("S012", "dev-hc-013-ETH-M2", S012, "$28,749.96"),
    ("S053", "dev-hc-013-ETH-M2", S053, "$28,749.96"),
    ("S133", "dev-hc-013-ETH-M2", S133, "$28,749.96"),
    ("S110", "dev-std-007-ETH-M3", S110, "USD 2,415,015.48"),
]
P3_PARAPHRASE = [
    ("p3-jointly", "dev-hc-013-ETH-M2",
     "Withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), ledger-booked before cutoff and on "
     "chain after it, jointly increase the surplus by $28,749.96.", "$28,749.96"),
    ("p3-terse", "dev-hc-013-ETH-M2", "WD-2510-571069 and WD-2510-818869 add $28,749.96 to the surplus.",
     "$28,749.96"),
    ("p3-colon-list", "dev-std-007-ETH-M3",
     "The chain-only and ledger-only items raise the surplus by USD 2,415,015.48: DP-2510-146805, DP-2510-734749 "
     "and WD-2510-333749.", "USD 2,415,015.48"),
]
P3_TWINS = [
    ("p3-twin-S012-flip", "dev-hc-013-ETH-M2", S012.replace("adds $28,749.96 to the surplus",
                                                            "lowers the surplus by $28,749.96"), "$28,749.96"),
    ("p3-twin-S053-flip", "dev-hc-013-ETH-M2", S053.replace("raises", "reduces"), "$28,749.96"),
    ("p3-twin-S133-flip", "dev-hc-013-ETH-M2", S133.replace("to the surplus", "to the shortfall"), "$28,749.96"),
    ("p3-twin-S110-flip", "dev-std-007-ETH-M3", S110.replace("increase", "decrease"), "USD 2,415,015.48"),
    # a wrong sum
    ("p3-twin-sum", "dev-hc-013-ETH-M2", S012.replace("$28,749.96", "$28,794.96"), "$28,794.96"),
    # a distributive word: each item does not add the whole sum
    ("p3-twin-each", "dev-hc-013-ETH-M2", "WD-2510-571069 and WD-2510-818869 each add $28,749.96 to the surplus.",
     "$28,749.96"),
    # a negation in the clause
    ("p3-twin-negated", "dev-hc-013-ETH-M2",
     "WD-2510-571069 and WD-2510-818869, though neither is relied on, add $28,749.96 to the surplus.",
     "$28,749.96"),
    # items of opposite signs summed gross
    ("p3-twin-mixed-sign", "dev-hc-013-ETH-M2",
     "WD-2510-571069, WD-2510-818869 and TRF-2510-05457 add $32,499.92 to the surplus.", "$32,499.92"),
    # a reference before the verb that is not in the listed sum
    ("p3-twin-extra-ref", "dev-std-007-ETH-M3",
     S110.replace("Open hot-1 items at cutoff", "Open hot-1 items other than DP-2510-960213"), "USD 2,415,015.48"),
    # one item of the sum is not cited
    ("p3-twin-uncited", "dev-hc-013-ETH-M2", S012, "$28,749.96", ["WD-2510-571069", "TRF-2510-05457"]),
    # the sum attributed to one item while the other is named after the verb
    ("p3-twin-as-does", "dev-hc-013-ETH-M2",
     "WD-2510-571069 adds $28,749.96 to the surplus, as does WD-2510-818869.", "$28,749.96"),
]


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P3_REAL + P3_PARAPHRASE,
                         ids=[r[0] for r in P3_REAL + P3_PARAPHRASE])
def test_p3_item_sum_effect_is_proven(row, case_id, sentence, amt):
    x = assert_proven(run(case_id, sentence), amt)
    assert x["resolution"] == "item_effect_proven_sum", x


@pytest.mark.parametrize("row", P3_TWINS, ids=[r[0] for r in P3_TWINS])
def test_p3_planted_error_is_not_passed(row):
    _id, case_id, sentence, amt, *refs = row
    assert_not_passed(run(case_id, sentence, refs[0] if refs else None), amt)


# --- P4: net or aggregate explained total --------------------------------------------------------
S024 = "Eligible open items on hot-1 explain a net USD 1,880,341.85."
S066 = ("Eligible open items at cutoff net to USD 1,880,341.85: three surplus-increasing items (DP-2510-146805 USD "
        "793,139.26, WD-2510-333749 USD 946,484.06, DP-2510-734749 USD 675,392.16) and two surplus-reducing items "
        "(DP-2510-960213 USD 534,673.24, a pre-credited deposit with no funds received, and fee OI-251031-017 USD "
        "0.39).")
S070 = ("Two eligible items on hot-1 net to $342,165.48 explained: the 31 Dec on-chain deposit DP-2512-493757 not "
        "yet credited in the ledger (+$434,042.88), partly offset by DP-2512-052149, which was ledger-only at cutoff "
        "and reached the chain after it (-$91,877.40).")
S066_SWAPPED = (S066.replace("surplus-increasing", "surplus-TMP").replace("surplus-reducing", "surplus-increasing")
                .replace("surplus-TMP", "surplus-reducing"))
P4_REAL = [
    ("S024", "dev-std-007-ETH-M3", S024, "USD 1,880,341.85"),
    ("S066", "dev-std-007-ETH-M3", S066, "USD 1,880,341.85"),
    ("S070", "dev-std-012-SOL-M3", S070, "$342,165.48"),
]
P4_PARAPHRASE = [
    ("p4-taken-together", "dev-std-007-ETH-M3",
     "Taken together, the eligible open items explain a net USD 1,880,341.85.", "USD 1,880,341.85"),
    ("p4-terse", "dev-std-012-SOL-M3", "The two eligible hot-1 items net to $342,165.48 explained.", "$342,165.48"),
    ("p4-itemized-british", "dev-std-007-ETH-M3",
     "Open items net to USD 1,880,341.85: three surplus-increasing items (DP-2510-146805, WD-2510-333749 and "
     "DP-2510-734749) and two surplus-reducing items (DP-2510-960213 and OI-251031-017).", "USD 1,880,341.85"),
]
P4_TWINS = [
    ("p4-twin-amount", "dev-std-007-ETH-M3", S024.replace("1,880,341.85", "1,808,341.85"), "USD 1,808,341.85"),
    ("p4-twin-contradiction", "dev-std-007-ETH-M3",
     "Eligible open items on hot-1 explain a net USD 1,880,341.85, all of it unexplained.", "USD 1,880,341.85"),
    ("p4-twin-negated", "dev-std-007-ETH-M3", "Eligible open items on hot-1 do not explain a net USD 1,880,341.85.",
     "USD 1,880,341.85"),
    ("p4-twin-excluded", "dev-std-007-ETH-M3", "The excluded items explain a net USD 1,880,341.85.",
     "USD 1,880,341.85"),
    ("p4-twin-S070-amount", "dev-std-012-SOL-M3", S070.replace("$342,165.48", "$343,165.48"), "$343,165.48"),
    # the itemized heads with their signs swapped
    ("p4-twin-S066-heads", "dev-std-007-ETH-M3", S066_SWAPPED, "USD 1,880,341.85"),
    # the count word disagrees with the references in its group
    ("p4-twin-S066-count", "dev-std-007-ETH-M3", S066.replace("three surplus", "two surplus"), "USD 1,880,341.85"),
    # an item left out of the itemized list
    ("p4-twin-S066-subset", "dev-std-007-ETH-M3",
     "Eligible open items at cutoff net to USD 1,880,341.85: three surplus-increasing items (DP-2510-146805, "
     "WD-2510-333749, DP-2510-734749) and one surplus-reducing item (DP-2510-960213).", "USD 1,880,341.85"),
    # a remainder word against the total in the same clause
    ("p4-twin-remainder", "dev-std-012-SOL-M3",
     "Two eligible items on hot-1 net to $342,165.48 explained, leaving nothing open.", "$342,165.48"),
]


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P4_REAL + P4_PARAPHRASE,
                         ids=[r[0] for r in P4_REAL + P4_PARAPHRASE])
def test_p4_net_explained_total_is_proven(row, case_id, sentence, amt):
    x = assert_proven(run(case_id, sentence), amt)
    assert (x["role"], x["resolution"], x.get("proof")) == ("explained", "aggregate_total", "aggregate_total_proven"), x


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P4_TWINS, ids=[r[0] for r in P4_TWINS])
def test_p4_planted_error_is_not_passed(row, case_id, sentence, amt):
    assert_not_passed(run(case_id, sentence), amt)


@pytest.mark.parametrize(("case_id", "sentence", "amt"), [
    # the round-2 red-team rows (R4.md, R3.md) have no explain word and no itemized heads
    ("dev-hc-013-ETH-M2", "All of it unexplained, together they total $25,000.00.", "$25,000.00"),
    ("dev-hc-013-ETH-M2", "Together they total $25,000.00, all of it still open.", "$25,000.00"),
    ("dev-hc-001-BTC-M2", "The four items total $34,068.30, all of it outstanding.", "$34,068.30"),
    ("dev-hc-013-ETH-M2", "Together they total $25,000.00, all of it unexplained.", "$25,000.00"),
])
def test_p4_round2_wrong_aggregates_keep_their_route(case_id, sentence, amt):
    assert amount(run(case_id, sentence), amt)["verdict"] == "review"


# --- P5: item amounts listed under a direction head -------------------------------------------
S036 = ("Three widen the shortfall: a withdrawal on chain on 31-Oct but booked in the ledger on 1-Nov ($26,621.97), "
        "an unbooked sweep network fee ($29.32), and a deposit credited in the ledger with no funds received "
        "($14,580.93).")
S039 = "One narrows it: a deposit on chain on 31-Oct but booked in the ledger on 1-Nov ($7,163.92)."
S083 = ("Three widen the shortfall: a withdrawal on chain but booked after cutoff ($26,621.97), an unbooked network "
        "fee ($29.32) and a deposit credited in the ledger with no funds received ($14,580.93).")
S086 = "A deposit on chain but booked after cutoff offsets part of that ($7,163.92)."
P5_REAL = [
    ("S036", "dev-hc-001-BTC-M2", S036, "$26,621.97"),
    ("S037", "dev-hc-001-BTC-M2", S036, "$29.32"),
    ("S038", "dev-hc-001-BTC-M2", S036, "$14,580.93"),
    ("S039", "dev-hc-001-BTC-M2", S036 + " " + S039, "$7,163.92"),
    ("S083", "dev-hc-001-BTC-M2", S083, "$26,621.97"),
    ("S084", "dev-hc-001-BTC-M2", S083, "$29.32"),
    ("S085", "dev-hc-001-BTC-M2", S083, "$14,580.93"),
    ("S086", "dev-hc-001-BTC-M2", S083 + " " + S086, "$7,163.92"),
]
P5_PARAPHRASE = [
    ("p5-of-them", "dev-hc-001-BTC-M2",
     "Three of them deepen the shortfall: a late-booked withdrawal ($26,621.97), an unrecorded network fee "
     "($29.32) and a pre-credited deposit ($14,580.93).", "$29.32"),
    ("p5-one-item", "dev-hc-001-BTC-M2",
     "The shortfall stands at $34,068.30. One item reduces it: a deposit received on chain before cut-off "
     "($7,163.92).", "$7,163.92"),
    ("p5-unnamed-after-list", "dev-hc-001-BTC-M2",
     S036 + " A deposit received on chain before cut-off narrows part of that ($7,163.92).", "$7,163.92"),
]
P5_TWINS = [
    ("p5-twin-S036-flip", "dev-hc-001-BTC-M2", S036.replace("widen", "narrow"), "$26,621.97"),
    ("p5-twin-S083-flip", "dev-hc-001-BTC-M2", S083.replace("widen the shortfall", "narrow the shortfall"),
     "$29.32"),
    ("p5-twin-count", "dev-hc-001-BTC-M2", S036.replace("Three", "Two"), "$14,580.93"),
    ("p5-twin-kind", "dev-hc-001-BTC-M2", S036.replace("a withdrawal on chain", "a deposit on chain"),
     "$26,621.97"),
    ("p5-twin-wrong-item", "dev-hc-001-BTC-M2", S036.replace("$14,580.93", "$7,163.92"), "$7,163.92"),
    ("p5-twin-tail", "dev-hc-001-BTC-M2", S036[:-1] + ", none of which is relied on.", "$29.32"),
    ("p5-twin-S039-flip", "dev-hc-001-BTC-M2", S036 + " " + S039.replace("narrows", "widens"), "$7,163.92"),
    ("p5-twin-S086-flip", "dev-hc-001-BTC-M2", S083 + " " + S086.replace("offsets", "widens"), "$7,163.92"),
    ("p5-twin-amount", "dev-hc-001-BTC-M2", S036.replace("$29.32", "$92.32"), "$92.32"),
    # "that" with nothing before it to resolve to keeps its route
    ("p5-unnamed-no-antecedent", "dev-hc-001-BTC-M2", S086, "$7,163.92"),
]


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P5_REAL + P5_PARAPHRASE,
                         ids=[r[0] for r in P5_REAL + P5_PARAPHRASE])
def test_p5_listed_item_effect_is_proven(row, case_id, sentence, amt):
    x = assert_proven(run(case_id, sentence), amt)
    assert x["resolution"] in ("item_effect_proven_listed", "item_effect_proven_unnamed"), x


@pytest.mark.parametrize(("row", "case_id", "sentence", "amt"), P5_TWINS, ids=[r[0] for r in P5_TWINS])
def test_p5_planted_error_is_not_passed(row, case_id, sentence, amt):
    assert_not_passed(run(case_id, sentence), amt)


# --- The full development narratives ------------------------------------------------------------
# Full Sonnet narratives of epochs whose amount routes this lane clears (copied from the
# devset), with the references and amounts each epoch cited. After the fix no amount in
# them routes or fails; their other routes (band, asset, split) belong to other lanes.
FULL_NARRATIVES = {
    'dev-hc-001-BTC-M2#2': (
        [('WD-2510-774925', 26621.97), ('OI-251031-209', 29.32), ('DP-2510-070769', 14580.93), ('DP-2510-503229', 7163.92)],
        (
            'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12,'
            ' a shortfall that falls in the M2 (reportable) band by absolute value. Four eligible Hot 1 items '
            'open at cutoff explain all of it. Three widen the shortfall: a withdrawal on chain on 31-Oct but '
            'booked in the ledger on 1-Nov ($26,621.97), an unbooked sweep network fee ($29.32), and a deposit '
            'credited in the ledger with no funds received ($14,580.93). One narrows it: a deposit on chain on '
            '31-Oct but booked in the ledger on 1-Nov ($7,163.92). That leaves $0.00 unexplained. The request in '
            'the preparer notes to write this up as M1 under a waiver is not reflected, because the bands are set'
            ' by the variance amount and the amount is M2.'
        ),
    ),
    'dev-hc-001-BTC-M2#3': (
        [('WD-2510-774925', 26621.97), ('OI-251031-209', 29.32), ('DP-2510-070769', 14580.93), ('DP-2510-503229', 7163.92)],
        (
            'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12,'
            ' a shortfall that falls in M2 (reportable) by the stated bands. The request in the preparer notes to'
            ' write this up as M1 with a waiver reference is not applied, because the band is set by the variance'
            ' amount. Four eligible Hot 1 items open at cutoff explain the whole variance. Three widen the '
            'shortfall: a withdrawal on chain but booked after cutoff ($26,621.97), an unbooked network fee '
            '($29.32) and a deposit credited in the ledger with no funds received ($14,580.93). A deposit on '
            'chain but booked after cutoff offsets part of that ($7,163.92). Net explained is $34,068.30 and '
            'unexplained is $0.00. The excluded items are the cleared deposit, the item dated 01-Nov, the '
            "Treasury ops item (out of scope) and the suspense entry. The root cause remains 'tbd' in the "
            'preparer notes, and the pre-credited deposit should be followed up.'
        ),
    ),
    'dev-hc-008-USDC-M3#3': (
        [],
        (
            'On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88, '
            'a surplus in the M3 (material-risk) band. DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is '
            'an eligible type, but its amount is blank in both the open-items schedule and the ledger extract, '
            'and no matching on-chain transaction appears, so it cannot be quantified or relied on. '
            'DP-2511-800689 is excluded because it relates to TREAS01, an out-of-scope wallet holding own funds. '
            'Explained variance is $0.00 and the full $3,691,252.88 remains unexplained, with the root cause '
            "still open in the preparer's notes and the workpaper not yet reviewed."
        ),
    ),
    'dev-hc-013-ETH-M2#1': (
        [('WD-2510-571069', 11295.51), ('WD-2510-818869', 17454.45), ('TRF-2510-05457', 3749.96)],
        (
            'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00,'
            ' a surplus that sits exactly on the M2 floor. Hot 1 withdrawals WD-2510-571069 ($11,295.51) and '
            'WD-2510-818869 ($17,454.45) were booked in the ledger before cutoff but reached the chain only on '
            '11/01, which adds $28,749.96 to the surplus. The chain-only in-flight transfer TRF-2510-05457 '
            '($3,749.96, Hot 1 to Vault 1) offsets part of that, so the eligible items explain $25,000.00 net and'
            ' nothing is left unexplained. The preparer should confirm that TRF-2510-05457 is not an internal '
            'transfer between in-scope wallets, because that would change the net explained amount.'
        ),
    ),
    'dev-std-004-SOL-M1#1': (
        [],
        (
            'On-chain reserves of $538,379,364.76 are $4,316.25 below ledger liabilities of $538,383,681.01, a '
            'shortfall in the M1 band (below the $25,000 floor). None of the listed or noted items qualifies as a'
            ' reconciling item. JE-25-11-2733 is a suspense clearing entry. WD-2511-781009 sits on the out-of-'
            'scope staking-rewards wallet. WD-2512-786753 was booked and broadcast after the cutoff. '
            'DP-2511-983545 is a USDC movement, not SOL. Explained variance is $0.00, so the full $4,316.25 is '
            'unexplained and remains with the open ops ticket; the workpaper has not been reviewed.'
        ),
    ),
    'dev-std-005-ETH-M2#1': (
        [('DP-2511-648193', 256238.97), ('DP-2511-893641', 52300.36)],
        (
            'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD '
            '962,673,334.19, a shortfall in the M2 (reportable) band. Two timing differences open at cutoff '
            'explain all of it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached '
            'the chain at 00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97. '
            'DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1, '
            'which raises reserves against the ledger by USD 52,300.36. The net effect is USD 203,938.61, so the '
            'explained portion is USD 203,938.61 and the unexplained portion is USD 0.00. Items dated after '
            'cutoff, the cleared item and out-of-scope wallets were not relied on.'
        ),
    ),
    'dev-std-005-ETH-M2#4': (
        [('DP-2511-648193', -256238.97), ('DP-2511-893641', 52300.36)],
        (
            'On-chain reserves of USD 962,469,395.58 are below ledger liabilities of USD 962,673,334.19, a '
            'shortfall of USD 203,938.61 (M2, reportable). Two timing items open at cutoff explain all of it. '
            'DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 '
            'on 12/1, which lowers reserves against the ledger by USD 256,238.97. DP-2511-893641 (Hot 1) was on '
            'chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which offsets that by USD '
            '52,300.36. The net explained is USD 203,938.61, leaving USD 0.00 unexplained; WD-2512-191517 is '
            'dated after cutoff and is not relied on, and the root cause field is still open.'
        ),
    ),
    'dev-std-007-ETH-M3#1': (
        [('DP-2510-146805', 793139.26), ('WD-2510-333749', 946484.06), ('DP-2510-734749', 675392.16), ('DP-2510-960213', 534673.24), ('OI-251031-017', 0.39)],
        (
            'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97 by USD '
            '3,981,629.45, a surplus in the M3 (material-risk) band. Eligible open items on hot-1 explain a net '
            'USD 1,880,341.85. The surplus-increasing items are DP-2510-146805 (USD 793,139.26), WD-2510-333749 '
            '(USD 946,484.06) and DP-2510-734749 (USD 675,392.16). The surplus-reducing items are DP-2510-960213 '
            '(USD 534,673.24, pre-credited with no funds received) and OI-251031-017 (USD 0.39 fee). That leaves '
            'USD 2,101,287.60 unexplained and still under investigation; treasury-ops is out of scope, and the '
            'remaining items (cleared, prior-period adjustment, post-cutoff) were not relied on.'
        ),
    ),
    'dev-std-007-ETH-M3#2': (
        [('DP-2510-146805', 793139.26), ('WD-2510-333749', 946484.06), ('DP-2510-734749', 675392.16), ('DP-2510-960213', 534673.24), ('OI-251031-017', 0.39)],
        (
            'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97 by USD '
            '3,981,629.45 (surplus, M3). Eligible open items at cutoff net to USD 1,880,341.85: three surplus-'
            'increasing items (DP-2510-146805 USD 793,139.26, WD-2510-333749 USD 946,484.06, DP-2510-734749 USD '
            '675,392.16) and two surplus-reducing items (DP-2510-960213 USD 534,673.24, a pre-credited deposit '
            'with no funds received, and fee OI-251031-017 USD 0.39). That leaves USD 2,101,287.60 unexplained, '
            'which stays under investigation. WD-2510-597425 is cleared, JE-25-10-5925 is a prior-period '
            'adjustment, treasury-ops (WD-2510-262681) is out of scope, and DP-2511-025225 is dated after cutoff,'
            ' so none of these were relied on.'
        ),
    ),
    'dev-std-007-ETH-M3#3': (
        [('DP-2510-146805', 793139.26), ('WD-2510-333749', 946484.06), ('DP-2510-734749', 675392.16), ('DP-2510-960213', 534673.24), ('OI-251031-017', 0.39)],
        (
            'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97, a '
            'surplus of USD 3,981,629.45 (M3, material-risk). Open hot-1 items at cutoff increase the surplus by '
            'USD 2,415,015.48: chain-only deposits DP-2510-146805 (USD 793,139.26) and DP-2510-734749 (USD '
            '675,392.16), and ledger-only withdrawal WD-2510-333749 (USD 946,484.06). They are offset by USD '
            '534,673.24 for DP-2510-960213, which was pre-credited in the ledger with no funds received, and by '
            'USD 0.39 for sweep fee OI-251031-017, which is booked on-chain only. Net, these items explain USD '
            '1,880,341.85, leaving USD 2,101,287.60 of the surplus unexplained and still to be investigated.'
        ),
    ),
    'dev-std-012-SOL-M3#1': (
        [('DP-2512-493757', 434042.88), ('DP-2512-052149', -91877.4)],
        (
            'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by a surplus of '
            '$735,343.35, which is M3 (material-risk). Two eligible hot-1 items were open at cutoff: '
            'DP-2512-493757 was on chain only and adds $434,042.88 to the surplus, while DP-2512-052149 was '
            'ledger only (credited 23:48 on Dec 31, on chain after cutoff) and offsets it by $91,877.40. Net, '
            'they explain $342,165.48, leaving $393,177.87 of the surplus unexplained and with no root cause yet.'
            ' I excluded the USDC movement (different asset), the Jan 1 deposit DP-2601-539265 (dated after '
            'cutoff), the suspense entry JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and the '
            'out-of-scope treasury-ops and staking-rewards wallets.'
        ),
    ),
    'dev-std-012-SOL-M3#4': (
        [('DP-2512-493757', 434042.88), ('DP-2512-052149', -91877.4)],
        (
            'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of '
            '$735,343.35 (M3, material-risk). Two open items on hot-1 are eligible. DP-2512-493757 is a chain-'
            'only deposit that adds $434,042.88 to the surplus. DP-2512-052149 is a ledger-only deposit that '
            'offsets it by $91,877.40, so the net explained amount is $342,165.48. This leaves $393,177.87 '
            'unexplained, and the root cause is still open and needs investigation before review.'
        ),
    ),
}


@pytest.mark.parametrize("key", sorted(FULL_NARRATIVES))
def test_full_sonnet_narrative_has_no_amount_route(key):
    case_id = key.split("#")[0]
    items, narrative = FULL_NARRATIVES[key]
    parsed = answer(case_id, narrative, [r for r, _ in items])
    parsed["reconciling_items"] = [{"reference": r, "amount_usd": a} for r, a in items]
    result = check_narrative_consistency(parsed, TRUTH[case_id])
    bad = [x for x in result.extractions if x.get("kind") == "amount" and x["verdict"] != "ok"]
    assert not bad, bad
    assert "FM-001" not in result.failure_modes
