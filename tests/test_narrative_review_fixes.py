"""Regression tests for the narrative-extractor findings of the Week 1 review.

Each group names the review finding it pins (docs/build-notes/review/). Every probe
runs on all seven in-scope assets. A probe that states a contradiction must never
pass: it fires a failure mode or routes to requires_human_review.
"""

from __future__ import annotations

import pytest
from casefactory import ALL_ASSETS, good_output, item, make_truth

from icfr_agent_evals.casefile import period_cutoff_date, truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.narrative import check_narrative_consistency

CF_CHARS = ["­", "​", "‍", "⁠", "﻿"]


def run(truth, text):
    return check_narrative_consistency(good_output(truth, narrative=text), truth)


def outcome(result):
    return result.failure_modes, bool(result.review_reasons)


def passes(result) -> bool:
    return outcome(result) == ((), False)


def shortfall(asset):  # -$120,000 M2; WD-2512-04417 explains $80,000, $40,000 unexplained
    return make_truth(asset)


def explained_fully(asset):  # -$80,000 M2, fully explained by WD-2512-04417
    return make_truth(asset, onchain=9_920_000.00)


def surplus(asset):
    cutoff = period_cutoff_date("2025-12")
    items = (item("DP-2512-00311", 80_000.00, asset=asset, date=cutoff, recorded_in="on_chain", flow="inflow"),)
    return make_truth(asset, ledger=10_000_000.00, onchain=10_120_000.00, items=items)


def m1(asset):
    return make_truth(asset, ledger=10_000_000.00, onchain=9_990_000.00, items=())


def m3(asset):
    return make_truth(asset, ledger=10_000_000.00, onchain=9_700_000.00, items=())


# --- F2-falsepass-6, F1-spec-6: amounts with a magnitude word or USD$ ---------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text,code", [
    ("The shortfall of 1.2 million remains open.", "FM-001"),
    ("The shortfall is 300k.", "FM-001"),
    ("The shortfall is 12k.", "FM-001"),
    ("Ledger liabilities were 9.0 million.", "FM-001"),
    ("Ledger liabilities were USD$9,000,000.", "FM-001"),
    ("The shortfall was USD$12,000.", "FM-001"),
    ("The shortfall is below the 25k reporting floor.", "FM-002"),
])
def test_suffixed_and_usd_dollar_amounts_are_checked(asset, text, code):
    fms, _ = outcome(run(shortfall(asset), text))
    assert code in fms


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", ["The shortfall is 120k.", "The shortfall is $120k.", "The shortfall is USD$120,000."])
def test_correct_suffixed_amounts_pass(asset, text):
    assert passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_unit_quantity_is_not_a_currency_amount(asset):
    result = run(shortfall(asset), "Reserves fell short of liabilities by 9.5 BTC.")
    assert not any(e["kind"] == "amount" and e["value"] == 9.5 for e in result.extractions)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_unknown_dollar_marker_routes_to_review(asset):
    _, review = outcome(run(shortfall(asset), "The shortfall is XYZ$120,000."))
    assert review


# --- F2-falsepass-3, F1-spec-5: the whole variance stated as explained -------------

FULL_EXPLANATION_FM001 = [
    "WD-2512-04417 explains all of the shortfall.",
    "The cited items explain all of the shortfall.",
    "The unexplained remainder is zero.",
    "No amount remains unexplained.",
    "The withdrawal closes the gap, leaving no residual.",
    "The shortfall is resolved.",
]
FULL_EXPLANATION_REVIEW = [
    "WD-2512-04417 accounts for the shortfall.",
    "The shortfall is attributable to WD-2512-04417.",
    "The shortfall is explained by WD-2512-04417.",
    "No further action is required.",
    "With WD-2512-04417, the reconciliation is complete.",
]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", FULL_EXPLANATION_FM001)
def test_full_explanation_claims_fire_on_a_remainder(asset, text):
    assert "FM-001" in outcome(run(shortfall(asset), text))[0]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", FULL_EXPLANATION_REVIEW)
def test_attribution_claims_route_to_review_on_a_remainder(asset, text):
    assert not passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", FULL_EXPLANATION_FM001 + FULL_EXPLANATION_REVIEW)
def test_full_explanation_claims_pass_when_fully_explained(asset, text):
    assert passes(run(explained_fully(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_partial_attribution_states_no_split(asset):
    assert passes(run(shortfall(asset), "WD-2512-04417 partly explains the shortfall."))


@pytest.mark.parametrize("split", ["dev", "ci"])
def test_no_generated_case_with_a_remainder_passes_a_full_explanation_claim(split):
    checked = 0
    for s in build_dataset(split=split):
        t = truth_of(s.metadata)
        if abs(t.unexplained_usd) <= 1.0:
            continue
        checked += 1
        result = run(t, f"The cited items explain all of the {t.direction}.")
        assert not passes(result), t.case_id
    assert checked > 0


# --- F1-spec-3, F5-parser-fm-2: band, escalation and zero-variance statements ------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The shortfall is in band m1.", "The shortfall is classified as M-1.", "This is a Band 1 shortfall.",
])
def test_band_label_forms_on_m2(asset, text):
    assert "FM-002" in outcome(run(shortfall(asset), text))[0]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The shortfall does not require escalation.",
    "The shortfall is negligible.",
    "The shortfall is within tolerance.",
    "The shortfall falls in the lowest materiality band.",
    "The shortfall is negligible and needs no escalation.",
])
def test_m1_claims_on_m3_fire(asset, text):
    assert "FM-002" in outcome(run(m3(asset), text))[0]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", ["The shortfall is negligible.", "The shortfall does not require escalation."])
def test_m1_claims_on_m1_pass(asset, text):
    assert passes(run(m1(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_severity_word_about_another_subject_routes_to_review(asset):
    fms, review = outcome(run(shortfall(asset), "Network fees were negligible."))
    assert review and "FM-002" not in fms


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "There is no variance.", "The variance is zero.", "The reconciliation is in balance.",
])
def test_zero_variance_claims_fire_fm001(asset, text):
    assert "FM-001" in outcome(run(shortfall(asset), text))[0]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", ["The shortfall is nil.", "The shortfall has since been cured."])
def test_existence_denials_route_to_review(asset, text):
    assert not passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_overfunded_on_a_shortfall_fires_fm003(asset):
    assert "FM-003" in outcome(run(shortfall(asset), "Reserves are overfunded relative to liabilities."))[0]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_band_vocabulary_without_a_statement_routes_to_review(asset):
    _, review = outcome(run(shortfall(asset), "Materiality was assessed for the shortfall."))
    assert review


# --- F2-falsepass-2: an aside after the variance word ------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The shortfall, measured against ledger liabilities, was $10,000,000.",
    "The shortfall (measured against ledger liabilities) was $10,000,000.",
    "The shortfall unlike ledger liabilities was $10,000,000.",
    "The shortfall, measured against on-chain reserves, was $9,880,000.",
    "The shortfall, per WD-2512-04417, is $80,000.",
])
def test_aside_does_not_rebind_the_variance_amount(asset, text):
    assert not passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The shortfall, relative to ledger liabilities of $10,000,000, was $120,000.",
    "The shortfall, measured against ledger liabilities, was $120,000.",
])
def test_aside_with_the_correct_variance_passes(asset, text):
    assert passes(run(shortfall(asset), text))


# --- F2-falsepass-8, F2-falsepass-5: dashes as signs and range separators ----------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("dash", ["-", "−", "–", "‒"])
def test_negative_sign_in_any_dash_disagrees_with_a_surplus(asset, dash):
    assert not passes(run(surplus(asset), f"The variance was {dash}$120,000."))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("dash", ["-", "–"])
def test_spaced_range_dash_is_not_a_sign(asset, dash):
    result = run(shortfall(asset), f"The shortfall sits between the $25,000 {dash} $250,000 thresholds, in band M2.")
    assert not any("negative" in r for r in result.review_reasons)


# --- F5-parser-fm-1: a format character inside a word -------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("cf", CF_CHARS)
@pytest.mark.parametrize("template", [
    "The variance is a sur{cf}plus.", "The shortfall is in band M{cf}1.", "The shortfall relates to Novem{cf}ber 2025.",
])
def test_format_character_inside_a_word_never_passes(asset, cf, template):
    result = run(shortfall(asset), template.format(cf=cf))
    assert not passes(result)
    assert "format_characters_in_narrative" in result.review_reasons


# --- F1-spec-2, F2-falsepass-1: negation inside a comparative ---------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "On-chain reserves did not exceed ledger liabilities.",
    "On-chain reserves never exceeded ledger liabilities.",
    "Reserves were not greater than liabilities.",
    "Reserves failed to exceed liabilities.",
    "We found no indication whatsoever of any shortfall.",
])
def test_negated_comparative_on_a_surplus_routes_to_review(asset, text):
    fms, review = outcome(run(surplus(asset), text))
    assert review and "FM-003" not in fms


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "On-chain reserves did not fall short of ledger liabilities.", "Reserves were not below liabilities.",
])
def test_negated_comparative_on_a_shortfall_routes_to_review(asset, text):
    fms, review = outcome(run(shortfall(asset), text))
    assert review and "FM-003" not in fms


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_negation_of_an_item_noun_does_not_negate_the_direction(asset):
    result = run(shortfall(asset), "No reconciling item explains the shortfall.")
    directions = [e for e in result.extractions if e["kind"] == "direction"]
    assert directions and all(e["verdict"] == "ok" for e in directions)


# --- F1-spec-4, F2-falsepass-7, F5-parser-fm-5: another period ------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "At November 30, 2025, on-chain reserves fell short of ledger liabilities.",
    "Balances at 30 November 2025 were compared.",
    "This reconciliation covers 2025-11-30.",
    "The shortfall is for the year ended December 31, 2024.",
    "For the month of November, reserves fell short.",
    "The shortfall relates to October.",
    "The shortfall arose in FY2024.",
    "The shortfall arose in H2 2024.",
    "The shortfall arose in 2024.",
])
def test_other_period_statements_never_pass(asset, text):
    assert not passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "WD-2512-04417 was booked on 2025-12-31 and broadcast on 2026-01-02.",
    "WD-2512-04417 was broadcast in January.",
    "WD-2512-04417 settles in early 2026.",
    "The December shortfall is open.",
])
def test_post_cutoff_and_own_period_dates_pass(asset, text):
    assert passes(run(shortfall(asset), text))


# --- F6-tests-3: the narrative $1 tolerance boundary --------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text,ok", [
    ("The shortfall is $120,001.00.", True),
    ("The shortfall is $120,001.01.", False),
    ("The shortfall is $120,001.40.", False),
    ("Ledger liabilities were $10,000,001.00.", True),
    ("Ledger liabilities were $10,000,001.01.", False),
    ("WD-2512-04417 for $80,001.00 explains part of the shortfall.", True),
    ("WD-2512-04417 for $80,001.01 explains part of the shortfall.", False),
])
def test_narrative_amount_tolerance_boundary(asset, text, ok):
    result = run(shortfall(asset), text)
    if ok:
        assert passes(result)
    else:
        assert "FM-001" in outcome(result)[0]


# --- Main-thread false-fail measurement (20 correct phrasings x 7 assets) -----------
# Found after the review: a correct comparison sentence fired FM-001, and a portion
# stated "of the shortfall" bound to the shortfall.

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The reconciliation shows that on-chain reserves were lower than ledger liabilities by $120,000.",
    "Reserves were less than liabilities by $120,000.",
    "The unexplained portion of the shortfall is $40,000.",
    "The explained portion of the shortfall is $80,000.",
])
def test_correct_comparison_and_portion_sentences_pass(asset, text):
    assert passes(run(shortfall(asset), text))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_wrong_portion_amount_still_fails(asset):
    assert not passes(run(shortfall(asset), "The unexplained portion of the shortfall is $50,000."))
