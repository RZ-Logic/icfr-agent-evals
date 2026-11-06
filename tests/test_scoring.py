"""Scoring-logic tests: the heart of the defensibility claim.

These prove the deterministic anchors *discriminate*: a correct commentary scores
full marks, and each specific defect drops the right dimension with the right
failure-mode code. If these pass with no API key, the claim "bad grading cannot
hide a consequential failure" has teeth, because the consequential checks are this
code, not a model.

Every check takes a schema-valid parsed output and a casefile.CaseTruth and returns
an AnchorResult. Fabrication (FM-005) is a cliff on the structured citations;
narrative-sweep hits score a provisional 0 and route to human review instead (see
test_sweep.py). The reconciling-items anchor has its own file (test_reconciling.py).
"""

import pytest
from casefactory import ALL_ASSETS, good_output, make_truth, scope_wallet

from icfr_agent_evals import narrative
from icfr_agent_evals.narrative import NarrativeResult
from icfr_agent_evals.scoring import (
    PLACEHOLDERS,
    check_completeness,
    check_correctness,
    check_fabrication,
    check_faithfulness,
    normalize_period,
)

TRUTH = make_truth("BTC")          # $120,000.00 M2 shortfall, BTC 2025-12
GOOD = good_output(TRUTH)


# --- Correctness ------------------------------------------------------------------

def test_correctness_all_correct():
    r = check_correctness(GOOD, TRUTH)
    assert r.points == 3 and r.failure_modes == () and not r.requires_human_review


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_correctness_all_correct_every_asset(asset):
    t = make_truth(asset)
    assert check_correctness(good_output(t), t).points == 3


def test_correctness_wrong_magnitude():
    r = check_correctness({**GOOD, "variance_amount_usd": 999.0}, TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-001",)


def test_correctness_signed_magnitude_routes_to_review():
    """The field is the absolute value. A signed figure equal to the true shortfall is
    off contract: review, not FM-001, and the point stays withheld (F3-07, D17).
    Before the fix round this test asserted FM-001, the defect."""
    r = check_correctness({**GOOD, "variance_amount_usd": -120_000.0}, TRUTH)
    assert "FM-001" not in r.failure_modes and r.points == 2
    assert any(x.startswith("off_contract_value: variance_amount_usd") for x in r.review_reasons)


def test_correctness_wrong_direction():
    r = check_correctness({**GOOD, "variance_direction": "surplus"}, TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-003",)


def test_correctness_wrong_band():
    r = check_correctness({**GOOD, "materiality_band": "M1"}, TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-002",)


def test_correctness_all_wrong():
    bad = {**GOOD, "variance_amount_usd": 1.0, "variance_direction": "surplus", "materiality_band": "M3"}
    r = check_correctness(bad, TRUTH)
    assert r.points == 0 and set(r.failure_modes) == {"FM-001", "FM-002", "FM-003"}


def test_correctness_magnitude_within_tolerance():
    r = check_correctness({**GOOD, "variance_amount_usd": 120_000.0 + 0.5}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("band,direction", [("m2", "SHORTFALL"), (" M2 ", " Shortfall "), ("ｍ２", "shortfall")])
def test_k1_band_and_direction_normalize_the_same_way(band, direction):
    r = check_correctness({**GOOD, "materiality_band": band, "variance_direction": direction}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("band", ["M4", "material", "", "2"])
def test_k1_band_outside_enum_is_wrong_value_not_format_break(band):
    r = check_correctness({**GOOD, "materiality_band": band}, TRUTH)
    assert r.failure_modes == ("FM-002",) and r.points == 2


@pytest.mark.parametrize("direction", ["deficit", "short", "", "neutral"])
def test_k1_direction_outside_enum_is_wrong_value(direction):
    r = check_correctness({**GOOD, "variance_direction": direction}, TRUTH)
    assert r.failure_modes == ("FM-003",) and r.points == 2


# --- Structured FM-016 ---------------------------------------------------------------

def test_k3_other_asset_and_period_fires_fm016_not_full_marks():
    """K3: a BTC 2025-12 case answered with another asset and period must fire
    FM-016 and set Correctness to 0, not score 3/3."""
    r = check_correctness({**GOOD, "asset": "ETH", "period": "2025-11"}, TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-016",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_fm016_on_asset_mismatch_every_asset(asset):
    t = make_truth(asset)
    other = next(a for a in ALL_ASSETS if a != asset)
    r = check_correctness({**good_output(t), "asset": other}, t)
    assert r.points == 0 and "FM-016" in r.failure_modes


@pytest.mark.parametrize("period", ["2025-10", "2024-12", "N/A", "November 2025", "2025-13"])
def test_fm016_on_period_mismatch(period):
    r = check_correctness({**GOOD, "period": period}, TRUTH)
    assert r.points == 0 and "FM-016" in r.failure_modes


@pytest.mark.parametrize("asset,period", [(" btc ", "2025-12"), ("BTC", "2025/12"), ("BTC", "2025-12-31"), ("BTC", " 2025-12 ")])
def test_identity_normalization_accepts_equivalent_forms(asset, period):
    r = check_correctness({**GOOD, "asset": asset, "period": period}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


def test_normalize_period_rejects_invalid_dates():
    assert normalize_period("2025-12-31") == "2025-12"
    assert normalize_period("2025-11-31") is None
    assert normalize_period("2025-12-31T00:00") is None


# --- Narrative consistency composition (extractor is B3's; composition is tested here) --

def test_narrative_failure_mode_sets_correctness_to_zero(monkeypatch):
    monkeypatch.setattr(narrative, "check_narrative_consistency", lambda p, t: NarrativeResult(failure_modes=("FM-003",)))
    r = check_correctness(GOOD, TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-003",)


def test_narrative_review_reason_keeps_structured_score(monkeypatch):
    monkeypatch.setattr(
        narrative, "check_narrative_consistency", lambda p, t: NarrativeResult(review_reasons=("negated direction",))
    )
    r = check_correctness(GOOD, TRUTH)
    assert r.points == 3 and r.failure_modes == () and r.review_reasons == ("negated direction",)
    assert r.requires_human_review


def test_narrative_codes_merge_with_structured_codes(monkeypatch):
    monkeypatch.setattr(narrative, "check_narrative_consistency", lambda p, t: NarrativeResult(failure_modes=("FM-001", "FM-016")))
    r = check_correctness({**GOOD, "asset": "ETH"}, TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-016", "FM-001")


def test_narrative_extractor_exception_routes_to_review(monkeypatch):
    def boom(p, t):
        raise KeyError("x")

    monkeypatch.setattr(narrative, "check_narrative_consistency", boom)
    r = check_correctness(GOOD, TRUTH)
    assert r.requires_human_review and "KeyError" in r.review_reasons[0]


def test_narrative_extractor_receives_parsed_and_truth(monkeypatch):
    seen = {}

    def spy(p, t):
        seen["args"] = (p, t)
        return NarrativeResult()

    monkeypatch.setattr(narrative, "check_narrative_consistency", spy)
    check_correctness(GOOD, TRUTH)
    assert seen["args"] == (GOOD, TRUTH)


# --- Faithfulness, fabrication -------------------------------------------------------

def test_fabrication_clean():
    r = check_fabrication(GOOD, TRUTH)
    assert r.points == 3 and r.failure_modes == () and not r.requires_human_review


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_fabricated_entity_is_a_cliff(asset):
    t = make_truth(asset)
    bad = {**good_output(t), "entities_cited": ["BitGo Trust", "Ghost Custodian LLC"]}
    r = check_fabrication(bad, t)
    assert r.points == 0 and r.failure_modes == ("FM-005",)


def test_fabrication_instance_is_listed_for_review():
    """Section 10(b): every fabrication instance routes to requires_human_review."""
    r = check_fabrication({**GOOD, "entities_cited": ["Ghost Custodian LLC"]}, TRUTH)
    assert r.requires_human_review


def test_fabrication_no_citations_is_not_fabrication():
    """Citing nothing is a coverage question, not fabrication."""
    r = check_fabrication({**GOOD, "entities_cited": []}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


def test_blank_citation_routes_to_review_not_fm005():
    r = check_fabrication({**GOOD, "entities_cited": ["BitGo Trust", "  "]}, TRUTH)
    assert r.points == 0 and r.failure_modes == () and r.requires_human_review


def test_citation_survives_organization_case_fold():
    r = check_fabrication({**GOOD, "entities_cited": ["bitgo trust"]}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("asset", ["SOL", "XRP"])
def test_case_variant_of_base58_wallet_is_fabrication(asset):
    t = make_truth(asset)
    r = check_fabrication({**good_output(t), "entities_cited": [scope_wallet(asset).swapcase()]}, t)
    assert r.failure_modes == ("FM-005",)


@pytest.mark.parametrize("asset", ["ETH", "USDC", "USDT", "BTC", "ADA"])
def test_case_variant_of_case_insensitive_wallet_is_grounded(asset):
    t = make_truth(asset)
    r = check_fabrication({**good_output(t), "entities_cited": [scope_wallet(asset).upper()]}, t)
    assert r.failure_modes == () and r.points == 3


def test_reference_case_variant_is_fabrication():
    r = check_fabrication({**GOOD, "entities_cited": ["wd-2512-04417"]}, TRUTH)
    assert r.failure_modes == ("FM-005",)


def test_ticker_is_not_a_valid_citation():
    r = check_fabrication({**GOOD, "entities_cited": ["BTC"]}, TRUTH)
    assert r.failure_modes == ("FM-005",)


# Sweep probes run on all seven assets (section 6; review finding F6-tests-5).

@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_narrative_org_routes_to_review_not_fm005(asset):
    truth = make_truth(asset)
    out = {**good_output(truth), "narrative": "Confirmed balances with Ghost Custodian LLC today."}
    r = check_fabrication(out, truth)
    assert r.points == 0 and r.failure_modes == () and r.requires_human_review


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_narrative_cited_wallet_is_grounded(asset):
    truth = make_truth(asset)
    w = scope_wallet(asset)
    out = {**good_output(truth), "entities_cited": ["BitGo Trust", w],
           "narrative": f"Wallet {w} reconciled against ledger liabilities."}
    r = check_fabrication(out, truth)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_generic_titlecase_finance_terms_not_flagged(asset):
    truth = make_truth(asset)
    out = {**good_output(truth), "narrative": "Reserve Reconciliation shows Ledger Liabilities exceeded On-Chain Reserves."}
    r = check_fabrication(out, truth)
    assert r.points == 3 and not r.requires_human_review


# --- Faithfulness composition ----------------------------------------------------------

def test_faithfulness_is_min_of_fabrication_and_reconciling_items():
    assert check_faithfulness(GOOD, TRUTH).points == 3
    omitted = {**GOOD, "reconciling_items": []}
    assert check_faithfulness(omitted, TRUTH).points == 1
    fabricated = {**GOOD, "entities_cited": ["Ghost LLC"]}
    r = check_faithfulness(fabricated, TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-005",)


def test_faithfulness_merges_codes_from_both_anchors():
    bad = {
        **GOOD,
        "entities_cited": ["Ghost LLC"],
        "reconciling_items": [{"reference": "NOPE-9999-01", "amount_usd": 1.0}],
    }
    r = check_faithfulness(bad, TRUTH)
    assert r.points == 0 and set(r.failure_modes) == {"FM-005", "FM-015"}


# --- Completeness (presence only) ------------------------------------------------------

def test_completeness_all_present():
    r = check_completeness(GOOD, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("value", ["N/A", "", "TBD", "unknown", "none", "  n/a ", "YYYY-MM", "pending"])
def test_k4_period_placeholder_is_absent(value):
    r = check_completeness({**GOOD, "period": value}, TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-004",)


@pytest.mark.parametrize("value", ["N/A", "", "TBD", "none", "NULL", "unknown", "the asset", "$", "123"])
def test_k4_asset_placeholder_or_non_ticker_is_absent(value):
    r = check_completeness({**GOOD, "asset": value}, TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-004",)


def test_placeholder_list_covers_the_brief_examples():
    assert {"n/a", "tbd", "unknown", "none"} <= PLACEHOLDERS


def test_completeness_is_presence_not_accuracy():
    """A wrong but present asset counts as present; its accuracy is FM-016 in
    Correctness. This keeps the dimensions orthogonal."""
    r = check_completeness({**GOOD, "asset": "ETH", "period": "2025-10"}, TRUTH)
    assert r.points == 3 and r.failure_modes == ()


def test_completeness_magnitude_present_as_finite_number():
    assert check_completeness({**GOOD, "variance_amount_usd": 0}, TRUTH).points == 3
