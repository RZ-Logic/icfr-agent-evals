"""Fix round 3 (2026-11-07), lane N1: amount pass-proofs (patterns P1 amount side, P2,
P3, P4 and P5 of pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md). RV (2026-11-07)
reverted P2, P3, P4 and P5 after the R1 red team, with their tests; P1 and its tests
stay. The P2 to P5 descriptions below are kept for the record.

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
    ledger", "offsets it by $X"): the figure equals the one item named in its segment,
    and the phrase's sign in
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


# --- The full development narratives ------------------------------------------------------------
# Full Sonnet narratives of epochs whose amount routes this lane clears (copied from the
# devset), with the references and amounts each epoch cited. RV (2026-11-07) removed the
# epochs that needed the reverted P2 to P5 rules; the two P1 epochs stay. After the fix no amount in
# them routes or fails; their other routes (band, asset, split) belong to other lanes.
FULL_NARRATIVES = {
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
