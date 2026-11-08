"""Fix round 3 (2026-11-07), lane S1: scoring.py patterns P8 (wiring), P10, P11, P12.

Brief: pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md. RV (2026-11-07)
reverted P8 (wiring), P10 and P11 after the R1 red team, with their tests; P12 and
its tests stay. The P8, P10 and P11 descriptions below are kept for the record.
Each rule carries the real Sonnet sentence (dev-set row id in the test id), a
paraphrase in another register, and a planted-error twin that must still fail or
route.

  P8   an ineligible item named inside an exclusion statement (contract C1,
       narrative.exclusion_spans) and governed by its exclusion phrase is excluded,
       not a route. Some real sentences need lane N2's wider C1 spans: those tests
       are marked xfail (strict) while the checked-out exclusion_spans does not
       cover the mention, and run as plain tests once it does (after integration).
  P10  a type label is read as the one type it names (brackets dropped, case,
       hyphen, underscore and space folded, closed synonyms); a memo-only item's
       label of another allowed type is not routed (owner ruling, S015).
  P11  a count word before a wallet label of the case file ("Two Hot 1 items") is
       not a capitalized-run hit; before anything else it still is. RV (2026-11-07)
       reverted P11 after the R1 red team, with its tests; the wallet-label test
       stays, because P12 uses fixture_wallet_labels.
  P12  a sweep hit equal to a fixture reference number or wallet label that
       entities_cited leaves out is a citation omission: details
       citation_omissions, anchor 0, no route, never FM-005 (owner ruling).

Keyless: pure functions on dev-split truths, no model, no key, no network.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest
from casefactory import good_output

from icfr_agent_evals import narrative as N
from icfr_agent_evals import scoring as S
from icfr_agent_evals.casefile import fold_dashes, nfkc_spaced, truth_of
from icfr_agent_evals.dataset import build_dataset


@lru_cache(maxsize=1)
def _pilot_truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


def truth(case_id: str):
    return _pilot_truths()[case_id]


def output(case_id: str, narrative_text: str, *, entities: list[str] | None = None,
           items: list[dict] | None = None) -> dict:
    out = good_output(truth(case_id), narrative=narrative_text)
    if entities is not None:
        out["entities_cited"] = list(entities)
    if items is not None:
        out["reconciling_items"] = list(items)
    return out


HC003 = "dev-hc-003-ETH-M1"
HC005 = "dev-hc-005-USDT-M3"
HC007 = "dev-hc-007-SOL-M2"
HC008 = "dev-hc-008-USDC-M3"
HC009 = "dev-hc-009-XRP-M3"
HC011 = "dev-hc-011-ADA-M1"
HC013 = "dev-hc-013-ETH-M2"
HC015 = "dev-hc-015-SOL-M1"
STD002 = "dev-std-002-BTC-M1"
STD004 = "dev-std-004-SOL-M1"
STD005 = "dev-std-005-ETH-M2"
STD008 = "dev-std-008-USDC-M1"
STD010 = "dev-std-010-BTC-M2"
STD012 = "dev-std-012-SOL-M3"
STD013 = "dev-std-013-USDC-M2"


# =====================================================================================
# P12: citation omission
# =====================================================================================

STD013_ENTITIES = ["WD-2510-799057", "HOT01"]  # Sonnet's list on std-013 e2
STD013_ITEMS = [{"reference": "WD-2510-799057", "amount_usd": 34349.43, "type": "timing difference"}]
S077 = ("I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, "
        "or the post-cutoff item WD-2511-221933.")
STD010_ENTITIES = ["DP-2512-574449", "OI-251231-441", "HOT01", "HOT02"]  # Sonnet's list on std-010 e3
S027 = "The cleared item, the TREAS01 balance and the post-cutoff postings were not relied on."
S118 = "OI-251231-441 ($15.24) is an unbooked chain-only network fee on the HOT02 sweep to COLD02."


@pytest.mark.parametrize("row, case_id, text, entities, omitted", [
    ("S077-S079", STD013, S077, STD013_ENTITIES, ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]),
    ("S027", STD010, S027, STD010_ENTITIES, ["TREAS01"]),
    ("S118", STD010, S118, STD010_ENTITIES, ["COLD02"]),
], ids=["S077-S079-references", "S027-wallet-label", "S118-wallet-label"])
def test_p12_real_sentence_is_a_citation_omission(row, case_id, text, entities, omitted):
    r = S.check_fabrication(output(case_id, text, entities=entities), truth(case_id))
    assert r.points == 0
    assert r.failure_modes == ()
    assert r.review_reasons == ()  # no route on that ground
    assert sorted(r.details["citation_omissions"]) == sorted(omitted)


def test_p12_faithfulness_anchor_scores_0_and_carries_the_list():
    t = truth(STD013)
    out = output(STD013, S077, entities=STD013_ENTITIES, items=STD013_ITEMS)
    r = S.check_faithfulness(out, t)
    assert r.points == 0 and "FM-005" not in r.failure_modes
    assert sorted(r.details["citation_omissions"]) == ["DP-2510-831633", "JE-25-10-9025", "WD-2511-221933"]


def test_p12_paraphrase_terse_register():
    text = "Neither JE-25-10-9025 nor WD-2511-221933 was used; COLD01 and COLD03 are untouched."
    t = truth(STD013)
    labels = S.fixture_wallet_labels(t)
    r = S.check_fabrication(output(STD013, text, entities=STD013_ENTITIES), t)
    assert r.points == 0 and r.failure_modes == ()
    # COLD01 is a wallet of std-013: an omission; COLD03 is not, so it keeps its route
    assert "COLD01" in labels and "COLD03" not in labels
    assert {"JE-25-10-9025", "WD-2511-221933", "COLD01"} <= set(r.details["citation_omissions"])
    assert "COLD03" not in r.details["citation_omissions"]
    assert any("COLD03" in reason for reason in r.review_reasons)


def test_p12_citing_the_entities_clears_the_omission():
    entities = STD013_ENTITIES + ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]
    r = S.check_fabrication(output(STD013, S077, entities=entities), truth(STD013))
    assert r.points == S.SCORE_MAX and r.details["citation_omissions"] == []


@pytest.mark.parametrize("text, token", [
    # a reference one digit off the fixture's: not grounded in the case file, a route
    ("I did not rely on the manual adjustment JE-25-10-9026.", "JE-25-10-9026"),
    # a wallet label the case file does not show
    ("OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep to COLD07.", "COLD07"),
], ids=["twin-invented-reference", "twin-invented-wallet"])
def test_p12_twin_ungrounded_hit_keeps_its_route(text, token):
    case_id = STD013 if token.startswith("JE") else STD010
    entities = STD013_ENTITIES if token.startswith("JE") else STD010_ENTITIES
    r = S.check_fabrication(output(case_id, text, entities=entities), truth(case_id))
    assert r.points == 0 and r.failure_modes == ()
    assert token not in r.details["citation_omissions"]
    assert any(token in reason for reason in r.review_reasons)


def test_p12_twin_fixture_organization_keeps_its_route():
    """The ruling covers reference numbers and wallet labels; an organization the case
    file shows but entities_cited leaves out keeps its sweep route."""
    text = "The OTC desk Halsworth Prime Ltd confirmed the December settlement."
    r = S.check_fabrication(output(HC007, text, entities=["HOT01"]), truth(HC007))
    assert "Halsworth Prime Ltd" not in r.details["citation_omissions"]
    assert r.requires_human_review


def test_p12_twin_fabricated_citation_is_still_fm005():
    r = S.check_fabrication(output(STD013, S077, entities=STD013_ENTITIES + ["Ghost Custody LLC"]), truth(STD013))
    assert r.failure_modes == ("FM-005",)
    assert sorted(r.details["citation_omissions"]) == ["DP-2510-831633", "JE-25-10-9025", "WD-2511-221933"]


# =====================================================================================
# Fixture wallet labels (P11 reverted by RV; the labels serve P12)
# =====================================================================================

def test_p11_fixture_wallet_labels_follow_the_generator_styles():
    """CaseTruth has no wallet table: the labels are derived from valid_entities.
    Every derived label on the pilot dev cases is one of the generator's three styles."""
    style = re.compile(
        r"(?:hot|warm|cold)-\d|treasury-ops|staking-rewards|(?:Hot|Warm|Vault) \d|Treasury ops|Staking payouts"
        r"|(?:HOT|WARM|COLD)\d\d|TREAS01|STAKE01"
    )
    for case_id, t in _pilot_truths().items():
        labels = S.fixture_wallet_labels(t)
        assert labels, case_id
        assert all(style.fullmatch(label) for label in labels), (case_id, sorted(labels))
