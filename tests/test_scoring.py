"""Scoring-logic tests — the heart of the defensibility claim.

These prove the deterministic anchors *discriminate*: a correct commentary scores
full marks, and each specific defect drops the right dimension with the right
failure-mode code. If these pass with no API key, the claim "bad grading cannot
hide a consequential failure" has teeth, because the consequential checks are this
code, not a model.

Faithfulness returns (points, failure_modes, review_tokens): FM-005 is the
deterministic fabrication cliff on the structured citations; review_tokens carries
narrative-sweep suspects, which score a conservative provisional 0 and route to
human review rather than silently passing (see scoring.py error-direction notes).
"""

import pytest

from icfr_agent_evals.scoring import (
    canonicalize_entity,
    check_completeness,
    check_correctness,
    check_faithfulness,
    parse_worker_output,
)

# A canonical ground-truth fixture and a matching correct answer.
TRUTH = {
    "asset": "BTC",
    "period": "2025-12",
    "variance_usd": -120_000.0,        # shortfall, M2
    "direction": "shortfall",
    "materiality_band": "M2",
    "valid_entities": ["BitGo Trust", "bc1qexamplewallet"],
}

GOOD = {
    "asset": "BTC",
    "period": "2025-12",
    "variance_amount_usd": 120_000.0,  # absolute value, matches |truth|
    "variance_direction": "shortfall",
    "materiality_band": "M2",
    "entities_cited": ["BitGo Trust"],
    "narrative": "On-chain reserves fell short of ledger liabilities for the period.",
}


# --- Correctness ------------------------------------------------------------------

def test_correctness_all_correct():
    points, fms = check_correctness(GOOD, TRUTH)
    assert points == 3 and fms == []


def test_correctness_wrong_magnitude():
    bad = {**GOOD, "variance_amount_usd": 999.0}
    points, fms = check_correctness(bad, TRUTH)
    assert points == 2 and fms == ["FM-001"]


def test_correctness_wrong_direction():
    bad = {**GOOD, "variance_direction": "surplus"}
    points, fms = check_correctness(bad, TRUTH)
    assert points == 2 and fms == ["FM-003"]


def test_correctness_wrong_band():
    bad = {**GOOD, "materiality_band": "M1"}
    points, fms = check_correctness(bad, TRUTH)
    assert points == 2 and fms == ["FM-002"]


def test_correctness_all_wrong():
    bad = {
        **GOOD,
        "variance_amount_usd": 1.0,
        "variance_direction": "surplus",
        "materiality_band": "M3",
    }
    points, fms = check_correctness(bad, TRUTH)
    assert points == 0
    assert set(fms) == {"FM-001", "FM-002", "FM-003"}


def test_correctness_magnitude_within_tolerance():
    near = {**GOOD, "variance_amount_usd": 120_000.0 + 0.5}  # within USD_TOLERANCE
    points, fms = check_correctness(near, TRUTH)
    assert points == 3 and fms == []


def test_correctness_bool_is_not_a_valid_number():
    """A JSON true would coerce to 1.0 if not guarded; it must not earn the point."""
    bad = {**GOOD, "variance_amount_usd": True}
    points, fms = check_correctness(bad, TRUTH)
    assert "FM-001" in fms


# --- Canonicalization -------------------------------------------------------------

def test_canonicalize_case_and_whitespace():
    assert canonicalize_entity("  BitGo   Trust ") == canonicalize_entity("bitgo trust")


def test_canonicalize_0x_prefix():
    assert canonicalize_entity("0xAbC123def456") == canonicalize_entity("abc123def456")


def test_canonicalize_0x_prefix_only_when_hex():
    # "0xGhostCorp" is not hex after the prefix; the prefix must NOT be stripped
    assert canonicalize_entity("0xGhostCorp") == "0xghostcorp"


# --- Faithfulness (fabrication cliff + narrative sweep) ---------------------------

def test_faithfulness_clean():
    points, fms, review = check_faithfulness(GOOD, TRUTH)
    assert points == 3 and fms == [] and review == []


def test_faithfulness_fabricated_entity_is_a_cliff():
    bad = {**GOOD, "entities_cited": ["BitGo Trust", "Ghost Custodian LLC"]}
    points, fms, review = check_faithfulness(bad, TRUTH)
    assert points == 0 and fms == ["FM-005"] and review == []


def test_faithfulness_no_citations_is_not_fabrication():
    """Citing nothing is a coverage question (LLM-graded later), not fabrication."""
    bad = {**GOOD, "entities_cited": []}
    points, fms, review = check_faithfulness(bad, TRUTH)
    assert points == 3 and fms == [] and review == []


def test_faithfulness_citation_survives_case_normalization():
    """A worker that lowercases the custodian must not be falsely failed."""
    normalized = {**GOOD, "entities_cited": ["bitgo trust"]}
    points, fms, review = check_faithfulness(normalized, TRUTH)
    assert points == 3 and fms == [] and review == []


def test_faithfulness_narrative_org_routes_to_review_not_fm005():
    """An uncited org-like name in the narrative is a review candidate with a
    conservative provisional 0 — not an automatic FM-005 verdict."""
    sneaky = {**GOOD, "narrative": "Confirmed balances with Ghost Custodian LLC today."}
    points, fms, review = check_faithfulness(sneaky, TRUTH)
    assert points == 0 and fms == []
    assert any("Ghost Custodian" in token for token in review)


def test_faithfulness_narrative_address_routes_to_review():
    sneaky = {**GOOD, "narrative": "Funds observed at 0xdeadbeefdeadbeef during review."}
    points, fms, review = check_faithfulness(sneaky, TRUTH)
    assert points == 0 and fms == [] and len(review) == 1


def test_faithfulness_narrative_cited_wallet_is_grounded():
    """Mentioning the fixture wallet in the narrative is grounded, not suspect."""
    grounded = {
        **GOOD,
        "entities_cited": ["BitGo Trust", "bc1qexamplewallet"],
        "narrative": "Wallet bc1qexamplewallet reconciled against ledger liabilities.",
    }
    points, fms, review = check_faithfulness(grounded, TRUTH)
    assert points == 3 and fms == [] and review == []


def test_faithfulness_generic_titlecase_finance_terms_not_flagged():
    stylistic = {**GOOD, "narrative": "Reserve Reconciliation shows Ledger Liabilities exceeded On-Chain Reserves."}
    points, fms, review = check_faithfulness(stylistic, TRUTH)
    assert points == 3 and fms == [] and review == []


# --- Completeness (presence only) -------------------------------------------------

def test_completeness_all_present():
    points, fms = check_completeness(GOOD, TRUTH)
    assert points == 3 and fms == []


@pytest.mark.parametrize("missing_key", ["asset", "period"])
def test_completeness_missing_identifier(missing_key):
    bad = {k: v for k, v in GOOD.items() if k != missing_key}
    points, fms = check_completeness(bad, TRUTH)
    assert points == 2 and fms == ["FM-004"]


def test_completeness_missing_magnitude():
    bad = {k: v for k, v in GOOD.items() if k != "variance_amount_usd"}
    points, fms = check_completeness(bad, TRUTH)
    assert points == 2 and fms == ["FM-004"]


def test_completeness_is_presence_not_accuracy():
    """A wrong-but-present asset still counts as present here; its accuracy is
    Correctness's concern, not Completeness's. This is what keeps the dimensions
    orthogonal."""
    wrong_but_present = {**GOOD, "asset": "ETH"}
    points, fms = check_completeness(wrong_but_present, TRUTH)
    assert points == 3 and fms == []


# --- Parser tolerance -------------------------------------------------------------

def test_parse_plain_json():
    assert parse_worker_output('{"asset": "BTC"}') == {"asset": "BTC"}


def test_parse_fenced_json():
    text = 'Here you go:\n```json\n{"asset": "BTC"}\n```'
    assert parse_worker_output(text) == {"asset": "BTC"}


def test_parse_json_with_preamble():
    text = 'Sure! {"asset": "BTC", "materiality_band": "M2"} done.'
    assert parse_worker_output(text) == {"asset": "BTC", "materiality_band": "M2"}


def test_parse_recovers_after_invalid_first_object():
    """The old greedy fallback swallowed everything between the first '{' and the
    last '}' and failed; balanced scanning tries each candidate in turn."""
    text = '{not valid} but then {"asset": "BTC"}'
    assert parse_worker_output(text) == {"asset": "BTC"}


def test_parse_nested_object():
    text = 'Result: {"asset": "BTC", "detail": {"band": "M2"}} end.'
    assert parse_worker_output(text) == {"asset": "BTC", "detail": {"band": "M2"}}


@pytest.mark.parametrize("bad", [None, "", "no json here", "[1, 2, 3]", "{not valid}"])
def test_parse_unparseable_returns_none(bad):
    assert parse_worker_output(bad) is None
