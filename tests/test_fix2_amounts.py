"""Fix round 2 (2026-11-06), lane L1: amount-role bindings (patterns A, B and C of
pilot-logs-2026-11-06/fixround/FIX-ROUND-2-BRIEF.md).

Every fix is tested with the real Sonnet sentence that carried a false FM-001
(quoted from the development set), one paraphrase, and a planted-error variant of
the same sentence that must still fire FM-001. Truth is the dev pilot case itself
(build_dataset(split="dev", subset="pilot")); the parsed answer cites the same
references the Sonnet epoch cited. No model is called.

A: an item effect ("which adds $X to the surplus", "lowers reserves against the
   ledger by $X", "offsets it by $X") is checked against the items named in its
   clause, not against the variance, a balance or the explained total. Narrowing
   patch (2026-11-06): a figure that matches the item's amount or the items' unsigned
   sum routes to review (item_effect_unverified, item_effect_sum_unverified), never
   ok, because the match checks neither direction, side nor netting.
B: "Together they total $X, leaving $0.00 unexplained": a cue after the next amount
   belongs to that amount, and an aggregate word whose subject is the items binds
   the explained total.
C: a hyphen inside a word ("in-flight", "cold-2") is not an aside dash.
"""

from __future__ import annotations

import pytest

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.narrative import check_narrative_consistency

TRUTH = {s.id: truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}

# the references each Sonnet epoch cited in reconciling_items (all eligible)
CITED = {
    "dev-hc-001-BTC-M2": ["WD-2510-774925", "OI-251031-209", "DP-2510-070769", "DP-2510-503229"],
    "dev-hc-011-ADA-M1": ["DP-2510-557345", "WD-2510-652593", "WD-2510-074129"],
    "dev-hc-013-ETH-M2": ["WD-2510-571069", "WD-2510-818869", "TRF-2510-05457"],
    "dev-hc-015-SOL-M1": ["TRF-2512-71501", "DP-2512-582869"],
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
    hits = [x for x in result.extractions if x.get("kind") == "amount" and x["text"] == text]
    assert hits, f"no amount extraction {text!r}: {[x.get('text') for x in result.extractions]}"
    return hits[0]


def test_dev_truth_matches_the_devset_cases():
    # the cases and amounts these tests rely on, so a moved dev set fails loudly here
    assert abs(TRUTH["dev-hc-013-ETH-M2"].variance_usd - 25_000.00) < 0.005
    assert abs(TRUTH["dev-hc-011-ADA-M1"].explained_usd + 19_977.55) < 0.005
    assert abs(TRUTH["dev-std-007-ETH-M3"].explained_usd - 1_880_341.85) < 0.005
    for case_id, refs in CITED.items():
        t = TRUTH[case_id]
        assert all(t.eligibility(t.item_by_reference(r)).eligible for r in refs), case_id


# --- Pattern A: item effects ------------------------------------------------------------
# (case, sentence, amount text, expected verdict, expected resolution or reason)
A_FIXED = [
    # real, dev-hc-013-ETH-M2 epoch 1: two named items, "which adds $X to the surplus"
    ("dev-hc-013-ETH-M2",
     "Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the "
     "ledger before cutoff but reached the chain only on 11/01, which adds $28,749.96 to the surplus.",
     "$28,749.96", "review", "item_effect_sum_unverified"),
    # real, dev-hc-013-ETH-M2 epoch 2: "which raises the surplus by $X"
    ("dev-hc-013-ETH-M2",
     "Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only "
     "after it (WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus "
     "by $28,749.96.",
     "$28,749.96", "review", "item_effect_sum_unverified"),
    # paraphrase
    ("dev-hc-013-ETH-M2",
     "Withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), ledger-booked before "
     "cutoff and on chain after it, jointly increase the surplus by $28,749.96.",
     "$28,749.96", "review", "item_effect_sum_unverified"),
    # real, dev-std-007-ETH-M3 epoch 3: the items follow a colon
    ("dev-std-007-ETH-M3",
     "Open hot-1 items at cutoff increase the surplus by USD 2,415,015.48: chain-only deposits "
     "DP-2510-146805 (USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal "
     "WD-2510-333749 (USD 946,484.06).",
     "USD 2,415,015.48", "review", "item_effect_sum_unverified"),
    # real, dev-std-005-ETH-M2 epoch 1: one named item, bound to the reserves cue
    ("dev-std-005-ETH-M2",
     "DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 "
     "on 12/1, which lowers reserves against the ledger by USD 256,238.97.",
     "USD 256,238.97", "review", "item_effect_unverified"),
    ("dev-std-005-ETH-M2",
     "DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1, "
     "which raises reserves against the ledger by USD 52,300.36.",
     "USD 52,300.36", "review", "item_effect_unverified"),
    # real, dev-std-005-ETH-M2 epoch 4: "which offsets that by", bound to the explained cue
    ("dev-std-005-ETH-M2",
     "DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, "
     "which offsets that by USD 52,300.36.",
     "USD 52,300.36", "review", "item_effect_unverified"),
    # paraphrase
    ("dev-std-005-ETH-M2",
     "DP-2511-648193, ledger-credited on 11/30 and settled on chain on 12/1, reduces on-chain reserves "
     "relative to the ledger by USD 256,238.97.",
     "USD 256,238.97", "review", "item_effect_unverified"),
    # real, dev-std-012-SOL-M3 epoch 4: "offsets it by $X"
    ("dev-std-012-SOL-M3",
     "DP-2512-052149 is a ledger-only deposit that offsets it by $91,877.40, so the net explained amount "
     "is $342,165.48.",
     "$91,877.40", "review", "item_effect_unverified"),
    # real, dev-std-012-SOL-M3 epoch 1: two references, the subject is the one before the verb
    ("dev-std-012-SOL-M3",
     "Two eligible hot-1 items were open at cutoff: DP-2512-493757 was on chain only and adds $434,042.88 "
     "to the surplus, while DP-2512-052149 was ledger only (credited 23:48 on Dec 31, on chain after "
     "cutoff) and offsets it by $91,877.40.",
     "$91,877.40", "review", "item_effect_unverified"),
    # paraphrase
    ("dev-std-012-SOL-M3",
     "The ledger-only deposit DP-2512-052149 offsets that by $91,877.40.",
     "$91,877.40", "review", "item_effect_unverified"),
    # real, dev-hc-001-BTC-M2 epoch 3: the item is described, not named, so it routes to review
    ("dev-hc-001-BTC-M2",
     "A deposit on chain but booked after cutoff offsets part of that ($7,163.92).",
     "$7,163.92", "review", "item_effect_unnamed_item"),
    # paraphrase
    ("dev-hc-001-BTC-M2",
     "One deposit, on chain but posted after cutoff, offsets part of it by $7,163.92.",
     "$7,163.92", "review", "item_effect_unnamed_item"),
    # real, dev-std-012-SOL-M3 epoch 2: a bracketed amount after the reference it describes
    ("dev-std-012-SOL-M3",
     "Two eligible items on hot-1 net to $342,165.48 explained: the 31 Dec on-chain deposit "
     "DP-2512-493757 not yet credited in the ledger (+$434,042.88), partly offset by DP-2512-052149, "
     "which was ledger-only at cutoff and reached the chain after it (-$91,877.40).",
     "+$434,042.88", "review", "parenthetical_item_amount"),
    # paraphrase
    ("dev-std-012-SOL-M3",
     "The on-chain deposit DP-2512-493757, still missing from the ledger at cutoff (+$434,042.88), is "
     "partly offset by DP-2512-052149.",
     "+$434,042.88", "review", "parenthetical_item_amount"),
]


@pytest.mark.parametrize(("case_id", "sentence", "amt", "verdict", "how"), A_FIXED)
def test_item_effect_is_not_a_false_fm001(case_id, sentence, amt, verdict, how):
    result = run(case_id, sentence)
    x = amount(result, amt)
    assert "FM-001" not in result.failure_modes, (sentence, x)
    assert x["verdict"] == verdict, x
    assert how in (x.get("resolution"), x.get("reason")), x
    if verdict == "review":
        assert any(r.startswith(how) for r in result.review_reasons), result.review_reasons


# (case, sentence, amount text, cited references or None for the epoch's own)
A_PLANTED = [
    # wrong effect amount for the two named items
    ("dev-hc-013-ETH-M2",
     "Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the "
     "ledger before cutoff but reached the chain only on 11/01, which adds $29,749.96 to the surplus.",
     "$29,749.96", None),
    ("dev-hc-013-ETH-M2",
     "Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only "
     "after it (WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus "
     "by $27,849.96.",
     "$27,849.96", None),
    # the two-item sum given as one item's effect (OpenAI R1 shape)
    ("dev-hc-013-ETH-M2",
     "WD-2510-818869 was booked in the ledger before cutoff but reached the chain only on 11/01, which "
     "adds $28,749.96 to the surplus.",
     "$28,749.96", None),
    # wrong sum after a colon
    ("dev-std-007-ETH-M3",
     "Open hot-1 items at cutoff increase the surplus by USD 2,451,015.48: chain-only deposits "
     "DP-2510-146805 (USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal "
     "WD-2510-333749 (USD 946,484.06).",
     "USD 2,451,015.48", None),
    # the right sum, but one item in it is not cited by the answer
    ("dev-std-007-ETH-M3",
     "Open hot-1 items at cutoff increase the surplus by USD 2,415,015.48: chain-only deposits "
     "DP-2510-146805 (USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal "
     "WD-2510-333749 (USD 946,484.06).",
     "USD 2,415,015.48", ["DP-2510-146805", "DP-2510-734749", "DP-2510-960213", "OI-251031-017"]),
    # wrong single-item effect
    ("dev-std-005-ETH-M2",
     "DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 "
     "on 12/1, which lowers reserves against the ledger by USD 265,238.97.",
     "USD 265,238.97", None),
    # the other item's amount given as this item's effect
    ("dev-std-005-ETH-M2",
     "DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 "
     "on 12/1, which lowers reserves against the ledger by USD 52,300.36.",
     "USD 52,300.36", None),
    ("dev-std-005-ETH-M2",
     "DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, "
     "which offsets that by USD 25,300.36.",
     "USD 25,300.36", None),
    ("dev-std-012-SOL-M3",
     "DP-2512-052149 is a ledger-only deposit that offsets it by $19,877.40, so the net explained amount "
     "is $342,165.48.",
     "$19,877.40", None),
    ("dev-hc-001-BTC-M2",
     "A deposit on chain but booked after cutoff offsets part of that ($7,613.92).",
     "$7,613.92", None),
    ("dev-std-012-SOL-M3",
     "Two eligible items on hot-1 net to $342,165.48 explained: the 31 Dec on-chain deposit "
     "DP-2512-493757 not yet credited in the ledger (+$443,042.88), partly offset by DP-2512-052149, "
     "which was ledger-only at cutoff and reached the chain after it (-$91,877.40).",
     "+$443,042.88", None),
]


@pytest.mark.parametrize(("case_id", "sentence", "amt", "refs"), A_PLANTED)
def test_planted_item_effect_error_still_fails(case_id, sentence, amt, refs):
    result = run(case_id, sentence, refs)
    x = amount(result, amt)
    assert x["verdict"] == "FM-001", x
    assert "FM-001" in result.failure_modes


def test_plain_variance_statement_is_not_read_as_an_item_effect():
    # a wrong variance with no effect verb stays a variance mismatch even when it
    # equals the sum of the items named in the clause
    result = run("dev-hc-013-ETH-M2",
                 "With WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) open, the surplus is "
                 "$28,749.96.")
    x = amount(result, "$28,749.96")
    assert (x["role"], x["verdict"]) == ("variance", "FM-001"), x


# --- Narrowing patch (2026-11-06): the red team's rows A1 to A8 ----------------------------
# Quoted from pilot-logs-2026-11-06/fixround/R1.md with their case ids. Each is a wrong
# item-effect statement (inverted direction, wrong side, an aside taken as the subject,
# or the gross sum of opposite-signed items) that 0baabfa scored clean through
# _item_effect_fit. No FM-001 is required of these rows, but the amount's verdict must
# be review, never ok, and the epoch must route.
A_R1_WRONG = [
    ("A1-effect-direction", "dev-hc-013-ETH-M2",
     "WD-2510-571069 was booked in the ledger before cutoff but reached the chain only on 11/01, which "
     "lowers reserves against the ledger by $11,295.51.",
     "$11,295.51"),
    ("A2-aside-subject", "dev-std-005-ETH-M2",
     "DP-2511-648193, together with DP-2511-893641, lowers reserves against the ledger by USD 52,300.36.",
     "USD 52,300.36"),
    ("A3-like-aside", "dev-hc-013-ETH-M2",
     "WD-2510-571069, like TRF-2510-05457, raises reserves against the ledger by $3,749.96.",
     "$3,749.96"),
    ("A4-gross-widen", "dev-hc-001-BTC-M2",
     "WD-2510-774925, OI-251031-209, DP-2510-070769 and DP-2510-503229 together widen the shortfall by "
     "$48,396.14.",
     "$48,396.14"),
    ("A5-gross-each-stated", "dev-hc-001-BTC-M2",
     "Together, WD-2510-774925 ($26,621.97) and DP-2510-503229 ($7,163.92) widen the shortfall by "
     "$33,785.89.",
     "$33,785.89"),
    ("A6-gross-offset", "dev-std-012-SOL-M3",
     "DP-2512-493757 and DP-2512-052149 together offset $525,920.28 of the surplus.",
     "$525,920.28"),
    ("A7-variance-as-effect", "dev-hc-013-ETH-M2",
     "WD-2510-571069 and WD-2510-818869 were booked before cutoff and settled on chain after it, and "
     "together they raise reserves above the ledger by $28,749.96.",
     "$28,749.96"),
    ("A8-wrong-side", "dev-std-005-ETH-M2",
     "DP-2511-893641 was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which lowers ledger "
     "liabilities by USD 52,300.36.",
     "USD 52,300.36"),
]


@pytest.mark.parametrize(("row_id", "case_id", "sentence", "amt"), A_R1_WRONG, ids=[r[0] for r in A_R1_WRONG])
def test_r1_wrong_item_effect_never_scores_ok(row_id, case_id, sentence, amt):
    result = run(case_id, sentence)
    x = amount(result, amt)
    assert x["verdict"] != "ok", x
    assert x["verdict"] == "review", x
    assert x["reason"] in ("item_effect_unverified", "item_effect_sum_unverified"), x
    assert any(amt in r for r in result.review_reasons), (result.review_reasons, x)


def test_item_effect_review_reaches_the_scorer_review_reasons():
    # the new reasons travel the same path as role_amount_fits_competing_cue: the
    # narrative check's review_reasons become the correctness anchor's review route
    from icfr_agent_evals.scoring import check_correctness

    case_id = "dev-std-005-ETH-M2"
    sentence = ("DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at "
                "00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97.")
    result = check_correctness(answer(case_id, sentence), TRUTH[case_id])
    assert result.failure_modes == (), result.review_reasons
    assert "item_effect_unverified: USD 256,238.97" in result.review_reasons, result.review_reasons


# R1's correct overfit probes that involve an item effect (P7, P9) and the aggregate
# probe P6: they may route, but no FM-001 may fire.
A_R1_CORRECT = [
    ("P6-sum-to", "dev-hc-011-ADA-M1", "The three items sum to 19,977.55 USD; nothing is unexplained."),
    ("P7-passive-effect", "dev-std-005-ETH-M2",
     "Reserves are lowered against the ledger by USD 256,238.97 through DP-2511-648193."),
    ("P9-passive-widened", "dev-hc-013-ETH-M2",
     "The surplus is widened by $28,749.96 through WD-2510-571069 and WD-2510-818869 together."),
]


@pytest.mark.parametrize(("row_id", "case_id", "sentence"), A_R1_CORRECT, ids=[r[0] for r in A_R1_CORRECT])
def test_r1_correct_probe_fires_no_fm001(row_id, case_id, sentence):
    result = run(case_id, sentence)
    assert "FM-001" not in result.failure_modes, result.extractions


# --- Pattern B: "together they total $X, leaving $0.00 unexplained" ----------------------
B_FIXED = [
    # real, dev-hc-011-ADA-M1 epochs 1, 2 and 3
    ("dev-hc-011-ADA-M1",
     "Together they total 19,977.55 USD, leaving 0.00 USD unexplained; WD-2510-652593 clearing after "
     "cutoff does not change that it was open at cutoff.",
     "19,977.55 USD"),
    ("dev-hc-011-ADA-M1", "Together they total $19,977.55, leaving $0.00 unexplained.", "$19,977.55"),
    ("dev-hc-011-ADA-M1",
     "These items total $19,977.55, leaving $0.00 unexplained; WD-2510-652593 clearing on 11/1 does not "
     "change that it was open at cutoff.",
     "$19,977.55"),
    # paraphrase
    ("dev-hc-011-ADA-M1", "Taken together the three items sum to $19,977.55, with $0.00 left unexplained.",
     "$19,977.55"),
    # real, dev-hc-013-ETH-M2 epoch 4: "net to $X, leaving $0.00 unexplained"
    ("dev-hc-013-ETH-M2",
     "The chain-only in-flight transfer TRF-2510-05457 ($3,749.96) works in the opposite direction and "
     "reduces the surplus, so the three items net to $25,000.00, leaving $0.00 unexplained.",
     "$25,000.00"),
    # real, dev-std-007-ETH-M3 epoch 4 (scored FM-000 in the log, FM-001 on rescore):
    # "net to $X of the surplus:" then the items
    ("dev-std-007-ETH-M3",
     "Open in-scope hot-1 items net to USD 1,880,341.85 of the surplus: DP-2510-146805 (USD 793,139.26), "
     "WD-2510-333749 (USD 946,484.06) and DP-2510-734749 (USD 675,392.16) increase it, while the "
     "pre-credited DP-2510-960213 (USD 534,673.24) and fee OI-251031-017 (USD 0.39) reduce it.",
     "USD 1,880,341.85"),
]


@pytest.mark.parametrize(("case_id", "sentence", "amt"), B_FIXED)
def test_aggregate_total_binds_the_explained_total(case_id, sentence, amt):
    result = run(case_id, sentence)
    x = amount(result, amt)
    assert "FM-001" not in result.failure_modes, (sentence, result.extractions)
    assert (x["role"], x["resolution"], x["verdict"]) == ("explained", "aggregate_total", "ok"), x
    assert not [r for r in result.review_reasons if amt in r], result.review_reasons


B_PLANTED = [
    # wrong explained total
    ("dev-hc-011-ADA-M1",
     "Together they total 19,777.55 USD, leaving 0.00 USD unexplained; WD-2510-652593 clearing after "
     "cutoff does not change that it was open at cutoff.",
     "19,777.55 USD"),
    ("dev-hc-011-ADA-M1", "Together they total $19,977.55, leaving $1,977.55 unexplained.", "$1,977.55"),
    ("dev-hc-011-ADA-M1",
     "These items total $18,977.55, leaving $0.00 unexplained; WD-2510-652593 clearing on 11/1 does not "
     "change that it was open at cutoff.",
     "$18,977.55"),
    ("dev-hc-013-ETH-M2",
     "The chain-only in-flight transfer TRF-2510-05457 ($3,749.96) works in the opposite direction and "
     "reduces the surplus, so the three items net to $28,749.96, leaving $0.00 unexplained.",
     "$28,749.96"),
    ("dev-std-007-ETH-M3",
     "Open in-scope hot-1 items net to USD 1,808,341.85 of the surplus: DP-2510-146805 (USD 793,139.26), "
     "WD-2510-333749 (USD 946,484.06) and DP-2510-734749 (USD 675,392.16) increase it, while the "
     "pre-credited DP-2510-960213 (USD 534,673.24) and fee OI-251031-017 (USD 0.39) reduce it.",
     "USD 1,808,341.85"),
    # the variance as the total of the items when part of it is unexplained
    ("dev-std-007-ETH-M3", "Together the open items total USD 3,981,629.45, so nothing is left to explain.",
     "USD 3,981,629.45"),
]


@pytest.mark.parametrize(("case_id", "sentence", "amt"), B_PLANTED)
def test_planted_aggregate_error_still_fails(case_id, sentence, amt):
    result = run(case_id, sentence)
    x = amount(result, amt)
    assert x["verdict"] == "FM-001", x
    assert "FM-001" in result.failure_modes


def test_aggregate_with_a_variance_subject_stays_the_variance():
    result = run("dev-hc-011-ADA-M1", "The shortfall totals $19,777.55, leaving $0.00 unexplained.")
    x = amount(result, "$19,777.55")
    assert (x["role"], x["verdict"]) == ("variance", "FM-001"), x


def test_aggregate_with_a_non_item_subject_is_not_the_explained_total():
    # "customer balances" is not a ledger cue word, but it does not name the items
    result = run("dev-hc-011-ADA-M1", "Customer balances total $206,975,512.86 at cutoff.")
    x = amount(result, "$206,975,512.86")
    assert x.get("resolution") != "aggregate_total", x
    assert "FM-001" not in result.failure_modes, x


def test_aggregate_after_a_remainder_word_is_not_the_explained_total():
    result = run("dev-std-007-ETH-M3",
                 "The open items explain USD 1,880,341.85, leaving a total of USD 2,101,287.60.")
    x = amount(result, "USD 2,101,287.60")
    assert x.get("resolution") != "aggregate_total", x


def test_aggregate_apposition_after_references_stays_an_item_sum():
    # "WD-1 and WD-2, totaling $X" is the sum of the named items (existing item rules)
    result = run("dev-hc-013-ETH-M2", "WD-2510-571069 and WD-2510-818869, totaling $28,749.96, remain open.")
    x = amount(result, "$28,749.96")
    assert (x["role"], x["verdict"], x.get("resolution")) == ("item", "ok", "sum_of_clause_items"), x


def test_net_of_a_subset_of_items_still_fails():
    # "net to" states the signed net of the items, so a gross part-sum is not excused
    result = run("dev-hc-001-BTC-M2", "Together they net to $41,232.22.")
    x = amount(result, "$41,232.22")
    assert (x["verdict"], x["reason"]) == ("FM-001", "explained_amount_mismatch"), x


def test_aggregate_of_a_subset_of_items_routes_to_review():
    # "together they total" may cover only the items just listed: never FM-001, never a pass
    result = run("dev-hc-001-BTC-M2",
                 "Together they total $41,232.22 before the offsetting deposit is taken into account.")
    x = amount(result, "$41,232.22")
    assert (x["verdict"], x["reason"]) == ("review", "aggregate_equals_item_subset_sum"), x
    assert "FM-001" not in result.failure_modes


def test_cue_after_the_next_amount_belongs_to_that_amount():
    # no aggregate word: the first figure is unresolved (review), not the unexplained amount
    result = run("dev-hc-011-ADA-M1", "They come to $19,977.55, leaving $0.00 unexplained.")
    first, second = amount(result, "$19,977.55"), amount(result, "$0.00")
    assert first["role"] != "unexplained" and first["verdict"] == "review", first
    assert (second["role"], second["verdict"]) == ("unexplained", "ok"), second
    assert "FM-001" not in result.failure_modes


# Aggregate step (2026-11-06, FIX-ROUND-2-AGGREGATE-BRIEF.md, R3.md): five wrong narratives
# label the explained total as the unexplained amount with a role cue after a comma or a
# verb, which the aggregate_total look-ahead (linking tokens only) never read. The figure
# keeps the aggregate_total binding and routes with aggregate_total_role_cue_follows; it
# never passes. (row id from R3.md, case, sentence, amount text, the cue word after it)
B_R3_WRONG = [
    ("Z8", "dev-hc-013-ETH-M2", "Together they total $25,000.00, all of it unexplained.", "$25,000.00", "unexplained"),
    ("ZB1", "dev-hc-013-ETH-M2", "Together they total $25,000.00, the entire amount unreconciled.", "$25,000.00",
     "unreconciled"),
    ("ZB2", "dev-hc-001-BTC-M2", "The four items total $34,068.30, all still unexplained.", "$34,068.30", "unexplained"),
    ("ZB3", "dev-std-012-SOL-M3", "Together they total $342,165.48, which is the unexplained portion.", "$342,165.48",
     "unexplained portion"),
    ("W11", "dev-hc-013-ETH-M2",
     "Together they total $25,000.00, all of it unexplained, since DP-2511-159273 was left out of the schedule "
     "but carried as the bridging item.",
     "$25,000.00", "unexplained"),
]


@pytest.mark.parametrize(("row_id", "case_id", "sentence", "amt", "cue"), B_R3_WRONG, ids=[r[0] for r in B_R3_WRONG])
def test_aggregate_total_with_a_contradicting_cue_after_routes(row_id, case_id, sentence, amt, cue):
    result = run(case_id, sentence)
    x = amount(result, amt)
    assert x["verdict"] != "ok", x
    assert (x["role"], x["resolution"], x["verdict"], x["reason"]) == (
        "explained", "aggregate_total", "review", "aggregate_total_role_cue_follows"), x
    assert x["cue_after"].lower() == cue, x
    assert f"aggregate_total_role_cue_follows: {amt}" in result.review_reasons, result.review_reasons
    assert "FM-001" not in result.failure_modes, result.extractions


def test_aggregate_total_route_reaches_the_scorer_review_reasons():
    from icfr_agent_evals.scoring import check_correctness

    case_id, sentence = B_R3_WRONG[0][1], B_R3_WRONG[0][2]
    result = check_correctness(answer(case_id, sentence), TRUTH[case_id])
    assert "aggregate_total_role_cue_follows: $25,000.00" in result.review_reasons, result.review_reasons


def test_aggregate_total_with_a_wrong_figure_and_a_cue_after_still_fails():
    # the route is added only to a figure that is the explained total: a wrong figure
    # keeps FM-001 (24,000.00 is no subset sum of the cited items either)
    result = run("dev-hc-013-ETH-M2", "Together they total $24,000.00, all of it unexplained.")
    x = amount(result, "$24,000.00")
    assert (x["resolution"], x["verdict"], x["reason"]) == ("aggregate_total", "FM-001", "explained_amount_mismatch"), x
    assert "FM-001" in result.failure_modes


def test_aggregate_total_with_an_amount_before_the_cue_stays_ok():
    # K8 (R3.md, dev-hc-001): the cue after $0.00 belongs to $0.00, so the total passes
    result = run("dev-hc-001-BTC-M2", "Together the four items net to $34,068.30, leaving $0.00 unexplained.")
    total, rest = amount(result, "$34,068.30"), amount(result, "$0.00")
    assert (total["role"], total["resolution"], total["verdict"]) == ("explained", "aggregate_total", "ok"), total
    assert (rest["role"], rest["verdict"]) == ("unexplained", "ok"), rest
    assert not result.review_reasons and not result.failure_modes, result


def test_aggregate_total_with_the_cue_in_the_next_sentence_stays_ok():
    # the scan ends at the clause: a cue in the next sentence is out of this check's scope
    result = run("dev-hc-013-ETH-M2", "Together they total $25,000.00. None of it is explained.")
    x = amount(result, "$25,000.00")
    assert (x["role"], x["resolution"], x["verdict"]) == ("explained", "aggregate_total", "ok"), x
    assert not [r for r in result.review_reasons if "$25,000.00" in r], result.review_reasons


def test_aggregate_total_cue_with_its_own_amount_after_is_not_a_contradiction():
    # the tests/test_narrative.py control: the cue labels the amount right after it
    result = run("dev-hc-001-BTC-M2",
                 "Explained items total $34,068.30 and the unexplained remainder is $0.00.")
    total, rest = amount(result, "$34,068.30"), amount(result, "$0.00")
    assert (total["role"], total["resolution"], total["verdict"]) == ("explained", "aggregate_total", "ok"), total
    assert (rest["role"], rest["verdict"]) == ("unexplained", "ok"), rest
    assert not result.review_reasons and not result.failure_modes, result


def test_aggregate_total_partitive_variance_cue_is_not_a_contradiction():
    # "$X of the surplus" is a part-of statement (the existing exception), not a later cue
    result = run("dev-std-007-ETH-M3",
                 "Open in-scope hot-1 items net to USD 1,880,341.85 of the surplus: DP-2510-146805 "
                 "(USD 793,139.26) increases it.")
    x = amount(result, "USD 1,880,341.85")
    assert (x["role"], x["resolution"], x["verdict"]) == ("explained", "aggregate_total", "ok"), x


# --- Pattern C: a hyphen inside a word is not an aside dash ---------------------------------
C_FIXED = [
    # real, dev-hc-015-SOL-M1 epoch 4
    ("Two eligible open items on hot-1 explain all of it: the in-flight transfer TRF-2512-71501 toward "
     "cold-2 ($6,637.86), which is on chain but not yet in the ledger, and the deposit DP-2512-582869 "
     "($9,888.19), which the ledger booked at 23:48 on 31-Dec but which only reached the chain on 1-Jan.",
     "$6,637.86"),
    # paraphrase (on ee39795 the hyphens make "shortfall" the aside subject)
    ("Two open items explain the shortfall: the in-flight hot-1 transfer TRF-2512-71501 heading to "
     "cold-2 ($6,637.86), which the ledger has not yet booked, and DP-2512-582869 ($9,888.19).",
     "$6,637.86"),
]


@pytest.mark.parametrize(("sentence", "amt"), C_FIXED)
def test_hyphenated_word_is_not_an_aside(sentence, amt):
    result = run("dev-hc-015-SOL-M1", sentence)
    x = amount(result, amt)
    assert "FM-001" not in result.failure_modes, result.extractions
    assert (x["role"], x["verdict"]) == ("item", "ok"), x
    assert x.get("reference") == "TRF-2512-71501", x


@pytest.mark.parametrize("sentence", [s.replace("($6,637.86)", "($6,673.86)") for s, _ in C_FIXED])
def test_planted_hyphenated_item_amount_still_fails(sentence):
    result = run("dev-hc-015-SOL-M1", sentence)
    x = amount(result, "$6,673.86")
    assert (x["verdict"], x["reason"]) == ("FM-001", "item_amount_mismatch"), x


def test_spaced_dash_still_opens_an_aside():
    # "The shortfall - measured against ledger liabilities - was $X" states the shortfall
    ok = run("dev-hc-015-SOL-M1", "The shortfall - measured against ledger liabilities - was $16,526.05.")
    bad = run("dev-hc-015-SOL-M1", "The shortfall - measured against ledger liabilities - was $16,562.05.")
    x, y = amount(ok, "$16,526.05"), amount(bad, "$16,562.05")
    assert (x["role"], x["resolution"], x["verdict"]) == ("variance", "aside_subject", "ok"), x
    assert (y["role"], y["verdict"]) == ("variance", "FM-001"), y


# --- The full development narratives ---------------------------------------------------------
# Full Sonnet narratives of the development epochs that carried a false FM-001,
# with the references and amounts each epoch cited (copied from the devset).
FULL_NARRATIVES = {
    'dev-hc-001-BTC-M2#3': (
        [('WD-2510-774925', 26621.97), ('OI-251031-209', 29.32), ('DP-2510-070769', 14580.93), ('DP-2510-503229', 7163.92)],
        (
            'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of '
            '$3,098,804,469.12, a shortfall that falls in M2 (reportable) by the stated bands. The request '
            'in the preparer notes to write this up as M1 with a waiver reference is not applied, because '
            'the band is set by the variance amount. Four eligible Hot 1 items open at cutoff explain the '
            'whole variance. Three widen the shortfall: a withdrawal on chain but booked after cutoff '
            '($26,621.97), an unbooked network fee ($29.32) and a deposit credited in the ledger with no '
            'funds received ($14,580.93). A deposit on chain but booked after cutoff offsets part of that '
            '($7,163.92). Net explained is $34,068.30 and unexplained is $0.00. The excluded items are the '
            'cleared deposit, the item dated 01-Nov, the Treasury ops item (out of scope) and the suspense '
            "entry. The root cause remains 'tbd' in the preparer notes, and the pre-credited deposit should "
            'be followed up.'
        ),
    ),
    'dev-hc-011-ADA-M1#1': (
        [('DP-2510-557345', 6298.01), ('WD-2510-652593', 7396.3), ('WD-2510-074129', 6283.24)],
        (
            'On-chain reserves of 206,955,535.31 USD sit 19,977.55 USD below ledger liabilities of '
            '206,975,512.86 USD, a shortfall in the M1 band (below the 25,000 USD floor). Three Hot 1 items '
            'open at the 10/31 cutoff explain all of it: DP-2510-557345 (6,298.01 USD) was credited in the '
            'ledger before cutoff but only reached the chain at 00:39 on 11/1. WD-2510-652593 (7,396.30 USD) '
            'went out on chain at 23:35 on 10/31 but was booked in the ledger at 00:01 on 11/1. '
            'WD-2510-074129 (6,283.24 USD) was sent on chain at 17:29 on 10/31 and is not yet in the ledger. '
            'Together they total 19,977.55 USD, leaving 0.00 USD unexplained; WD-2510-652593 clearing after '
            'cutoff does not change that it was open at cutoff.'
        ),
    ),
    'dev-hc-011-ADA-M1#2': (
        [('DP-2510-557345', 6298.01), ('WD-2510-652593', 7396.3), ('WD-2510-074129', 6283.24)],
        (
            'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of '
            '$206,975,512.86, a shortfall that falls in M1 (below the reportable floor). Three open Hot 1 '
            'items dated 10/31 explain all of it: DP-2510-557345 ($6,298.01, deposit in the ledger before '
            'cutoff and on chain at 00:39 on 11/1), WD-2510-652593 ($7,396.30, on chain at 23:35 on 10/31 '
            'and in the ledger at 00:01 on 11/1) and WD-2510-074129 ($6,283.24, sent on chain at 17:29 and '
            'not yet in the ledger). Together they total $19,977.55, leaving $0.00 unexplained. '
            'WD-2510-652593 cleared after cutoff, but it was open at the cutoff, so it is still a valid '
            'reconciling item for this period.'
        ),
    ),
    'dev-hc-011-ADA-M1#3': (
        [('DP-2510-557345', 6298.01), ('WD-2510-652593', 7396.3), ('WD-2510-074129', 6283.24)],
        (
            'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of '
            '$206,975,512.86, a shortfall in band M1 (below the $25,000 floor). Three open Hot 1 items dated '
            '2025-10-31 explain the full amount: deposit DP-2510-557345 ($6,298.01) was in the ledger at '
            '23:53 but reached the chain only at 00:39 on 11/1. Withdrawal WD-2510-652593 ($7,396.30) went '
            'out on chain at 23:35 but was posted in the ledger after cutoff. Withdrawal WD-2510-074129 '
            '($6,283.24) was sent on chain at 17:29 and had not yet been booked in the ledger. These items '
            'total $19,977.55, leaving $0.00 unexplained; WD-2510-652593 clearing on 11/1 does not change '
            'that it was open at cutoff.'
        ),
    ),
    'dev-hc-013-ETH-M2#1': (
        [('WD-2510-571069', 11295.51), ('WD-2510-818869', 17454.45), ('TRF-2510-05457', 3749.96)],
        (
            'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by '
            '$25,000.00, a surplus that sits exactly on the M2 floor. Hot 1 withdrawals WD-2510-571069 '
            '($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger before cutoff but '
            'reached the chain only on 11/01, which adds $28,749.96 to the surplus. The chain-only in-flight '
            'transfer TRF-2510-05457 ($3,749.96, Hot 1 to Vault 1) offsets part of that, so the eligible '
            'items explain $25,000.00 net and nothing is left unexplained. The preparer should confirm that '
            'TRF-2510-05457 is not an internal transfer between in-scope wallets, because that would change '
            'the net explained amount.'
        ),
    ),
    'dev-hc-013-ETH-M2#2': (
        [('WD-2510-571069', 11295.51), ('WD-2510-818869', 17454.45), ('TRF-2510-05457', -3749.96)],
        (
            'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76, a '
            'surplus of $25,000.00, which sits exactly at the M2 (reportable) floor. Two Hot 1 customer '
            'withdrawals were booked in the ledger before cutoff but reached the chain only after it '
            '(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus by '
            '$28,749.96. The chain-only in-flight transfer TRF-2510-05457 on Hot 1 ($3,749.96) lowers it, so '
            'the net explained amount is $25,000.00 and the unexplained remainder is $0.00. I excluded the '
            'Staking payouts item WD-2510-991353 (out-of-scope wallet), the prior-period adjustment '
            'JE-25-10-1421, and the post-cutoff November activity.'
        ),
    ),
    'dev-hc-013-ETH-M2#4': (
        [('WD-2510-571069', 11295.51), ('WD-2510-818869', 17454.45), ('TRF-2510-05457', 3749.96)],
        (
            'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76, a '
            'surplus of $25,000.00 that falls in the M2 (reportable) band. Two Hot 1 withdrawals were booked '
            'in the ledger before cutoff but reached the chain only on 11/01 (WD-2510-571069 at $11,295.51 '
            'and WD-2510-818869 at $17,454.45), which adds $28,749.96 to the surplus. The chain-only '
            'in-flight transfer TRF-2510-05457 ($3,749.96) works in the opposite direction and reduces the '
            'surplus, so the three items net to $25,000.00, leaving $0.00 unexplained. I did not rely on '
            'JE-25-10-1421 (a prior-period adjustment) or WD-2510-991353 (Staking payouts wallet, out of '
            'scope).'
        ),
    ),
    'dev-hc-015-SOL-M1#4': (
        [('TRF-2512-71501', 6637.86), ('DP-2512-582869', 9888.19)],
        (
            'On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of '
            '$414,905,064.84, a shortfall that falls in band M1 (below the $25,000 floor). Two eligible open '
            'items on hot-1 explain all of it: the in-flight transfer TRF-2512-71501 toward cold-2 '
            '($6,637.86), which is on chain but not yet in the ledger, and the deposit DP-2512-582869 '
            '($9,888.19), which the ledger booked at 23:48 on 31-Dec but which only reached the chain on '
            '1-Jan. Together they explain $16,526.05, leaving $0.00 unexplained. The manual adjustment '
            "JE-25-12-3777 is excluded as a non-eligible item, and the root-cause field in the preparer's "
            'notes is still blank.'
        ),
    ),
    'dev-std-005-ETH-M2#1': (
        [('DP-2511-648193', 256238.97), ('DP-2511-893641', 52300.36)],
        (
            'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD '
            '962,673,334.19, a shortfall in the M2 (reportable) band. Two timing differences open at cutoff '
            'explain all of it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but '
            'reached the chain at 00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97. '
            'DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on '
            '12/1, which raises reserves against the ledger by USD 52,300.36. The net effect is USD '
            '203,938.61, so the explained portion is USD 203,938.61 and the unexplained portion is USD 0.00. '
            'Items dated after cutoff, the cleared item and out-of-scope wallets were not relied on.'
        ),
    ),
    'dev-std-005-ETH-M2#4': (
        [('DP-2511-648193', -256238.97), ('DP-2511-893641', 52300.36)],
        (
            'On-chain reserves of USD 962,469,395.58 are below ledger liabilities of USD 962,673,334.19, a '
            'shortfall of USD 203,938.61 (M2, reportable). Two timing items open at cutoff explain all of '
            'it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain '
            'at 00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97. DP-2511-893641 '
            '(Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which offsets '
            'that by USD 52,300.36. The net explained is USD 203,938.61, leaving USD 0.00 unexplained; '
            'WD-2512-191517 is dated after cutoff and is not relied on, and the root cause field is still '
            'open.'
        ),
    ),
    'dev-std-007-ETH-M3#3': (
        [('DP-2510-146805', 793139.26), ('WD-2510-333749', 946484.06), ('DP-2510-734749', 675392.16), ('DP-2510-960213', 534673.24), ('OI-251031-017', 0.39)],
        (
            'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97, a '
            'surplus of USD 3,981,629.45 (M3, material-risk). Open hot-1 items at cutoff increase the '
            'surplus by USD 2,415,015.48: chain-only deposits DP-2510-146805 (USD 793,139.26) and '
            'DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal WD-2510-333749 (USD 946,484.06). '
            'They are offset by USD 534,673.24 for DP-2510-960213, which was pre-credited in the ledger with '
            'no funds received, and by USD 0.39 for sweep fee OI-251031-017, which is booked on-chain only. '
            'Net, these items explain USD 1,880,341.85, leaving USD 2,101,287.60 of the surplus unexplained '
            'and still to be investigated.'
        ),
    ),
    'dev-std-007-ETH-M3#4': (
        [('DP-2510-146805', 793139.26), ('WD-2510-333749', 946484.06), ('DP-2510-734749', 675392.16), ('DP-2510-960213', 534673.24), ('OI-251031-017', 0.39)],
        (
            'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97 by '
            'USD 3,981,629.45 (surplus, M3). Open in-scope hot-1 items net to USD 1,880,341.85 of the '
            'surplus: DP-2510-146805 (USD 793,139.26), WD-2510-333749 (USD 946,484.06) and DP-2510-734749 '
            '(USD 675,392.16) increase it, while the pre-credited DP-2510-960213 (USD 534,673.24) and fee '
            'OI-251031-017 (USD 0.39) reduce it. USD 2,101,287.60 of the variance remains unexplained. The '
            'treasury-ops item WD-2510-262681 is out of scope and DP-2511-025225 is dated after the cutoff, '
            'so neither was relied on. Cleared item WD-2510-597425 and prior-period adjustment JE-25-10-5925 '
            'were also excluded.'
        ),
    ),
    'dev-std-012-SOL-M3#2': (
        [('DP-2512-493757', 434042.88), ('DP-2512-052149', -91877.4)],
        (
            'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of '
            '$735,343.35 (M3, material-risk). Two eligible items on hot-1 net to $342,165.48 explained: the '
            '31 Dec on-chain deposit DP-2512-493757 not yet credited in the ledger (+$434,042.88), partly '
            'offset by DP-2512-052149, which was ledger-only at cutoff and reached the chain after it '
            '(-$91,877.40). That leaves $393,177.87 of the surplus unexplained and still under '
            'investigation. I excluded the USDC deposit (another asset), the cleared DP-2512-310317, the '
            'January item DP-2601-539265, the suspense entry JE-25-12-5145 and the out-of-scope wallets, and '
            'the review is still pending.'
        ),
    ),
    'dev-std-012-SOL-M3#4': (
        [('DP-2512-493757', 434042.88), ('DP-2512-052149', -91877.4)],
        (
            'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of '
            '$735,343.35 (M3, material-risk). Two open items on hot-1 are eligible. DP-2512-493757 is a '
            'chain-only deposit that adds $434,042.88 to the surplus. DP-2512-052149 is a ledger-only '
            'deposit that offsets it by $91,877.40, so the net explained amount is $342,165.48. This leaves '
            '$393,177.87 unexplained, and the root cause is still open and needs investigation before '
            'review.'
        ),
    ),
}


@pytest.mark.parametrize("key", sorted(FULL_NARRATIVES))
def test_full_sonnet_narrative_has_no_amount_fm001(key):
    case_id = key.split("#")[0]
    items, narrative = FULL_NARRATIVES[key]
    parsed = answer(case_id, narrative, [r for r, _ in items])
    parsed["reconciling_items"] = [{"reference": r, "amount_usd": a} for r, a in items]
    result = check_narrative_consistency(parsed, TRUTH[case_id])
    bad = [x for x in result.extractions if x.get("kind") == "amount" and x["verdict"] == "FM-001"]
    assert not bad, bad
