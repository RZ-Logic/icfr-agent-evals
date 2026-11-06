"""Fix round 2 (2026-11-06), lane L3: fabrication check and narrative sweep.

Pattern F of pilot-logs-2026-11-06/fixround/FIX-ROUND-2-BRIEF.md, with the real
Sonnet citation lists that failed (quoted from the devset), paraphrases, and
planted-error variants that must still fail.

  F  the case's asset ticker or chain name, or a wallet label from the wallet-scope
     table, cited in entities_cited is not a fabricated counterparty (FM-005); a
     name the case file does not show is still FM-005.

Pattern G (the count-word strip in the capitalized-run sweep and the exclusion
readings behind "names ineligible item X without stating whether it explains the
variance") and the lane P2 narrowing patch were reverted on 2026-11-06 (R2.md);
their tests left with them, see the note at the end of this file.

Keyless: pure functions on dev-split truths, no model, no key, no network.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from casefactory import ALL_ASSETS, good_output, make_truth

from icfr_agent_evals import scoring as S
from icfr_agent_evals.casefile import ASSET_CHAIN, truth_of
from icfr_agent_evals.dataset import build_dataset


@lru_cache(maxsize=1)
def _pilot_truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


def truth(case_id: str):
    return _pilot_truths()[case_id]


def output(case_id: str, *, narrative_text: str | None = None, entities: list[str] | None = None) -> dict:
    t = truth(case_id)
    out = good_output(t, narrative=narrative_text)
    if entities is not None:
        out["entities_cited"] = list(entities)
    return out


# =====================================================================================
# Pattern F: ticker, chain and wallet labels in entities_cited
# =====================================================================================

HC003 = "dev-hc-003-ETH-M1"
HC003_EP2_ENTITIES = [
    "ETH", "JE-25-11-7405", "DP-2511-680653", "DP-2511-340361", "Hot 1", "Hot 2", "Staking payouts",
]
STD007 = "dev-std-007-ETH-M3"
STD007_EP2_ENTITIES = [
    "ETH", "hot-1", "treasury-ops", "OI-251031-017", "DP-2510-146805", "WD-2510-333749", "DP-2510-734749",
    "DP-2510-960213", "WD-2510-597425", "JE-25-10-5925", "WD-2510-262681", "DP-2511-025225",
]
STD012 = "dev-std-012-SOL-M3"
STD012_EP4_ENTITIES = ["SOL", "hot-1", "DP-2512-493757", "DP-2512-052149"]


@pytest.mark.parametrize(
    "case_id, entities",
    [(HC003, HC003_EP2_ENTITIES), (STD007, STD007_EP2_ENTITIES), (STD012, STD012_EP4_ENTITIES)],
    ids=["hc-003-ep2-ETH", "std-007-ep2-ETH", "std-012-ep4-SOL"],
)
def test_f_real_case_ticker_citation_is_not_fm005(case_id, entities):
    """The three devset epochs that fired FM-005 on the case's own ticker."""
    r = S.check_fabrication(output(case_id, entities=entities), truth(case_id))
    assert "FM-005" not in r.failure_modes
    assert r.details["fabricated"] == []
    assert r.details["case_header_terms_cited"] == [entities[0]]


@pytest.mark.parametrize("cited", ["ethereum", "Ethereum", "eth", "ETH "])
def test_f_paraphrase_chain_name_and_ticker_casing_pass(cited):
    r = S.check_fabrication(output(HC003, entities=[cited, "Hot 1"]), truth(HC003))
    assert r.failure_modes == () and r.points == 3


@pytest.mark.parametrize(
    "case_id, label",
    [(HC003, "Staking payouts"), (HC003, "Treasury ops"), (STD007, "treasury-ops"),
     ("dev-std-004-SOL-M1", "staking-rewards"), ("dev-hc-008-USDC-M3", "TREAS01")],
)
def test_f_out_of_scope_wallet_label_is_in_the_case_file(case_id, label):
    """A wallet label from the wallet-scope table, in scope or not, is in the fixture."""
    r = S.check_fabrication(output(case_id, entities=[label]), truth(case_id))
    assert "FM-005" not in r.failure_modes


@pytest.mark.parametrize(
    "fake",
    [
        "DOGE",                          # a ticker the case file never shows
        "BTC",                           # another asset, no BTC item on this ETH case
        "solana",                        # another chain
        "Staking Payouts Ltd",           # a counterparty dressed as the label
        "Staking rewards",               # a label another case uses, not this one
        "Hot 9",                         # a wallet label the scope table does not show
        "ETH Custody Partners",          # the ticker inside an invented name
        "Ethereum Foundation",           # the chain inside an invented name
        "ETH (ethereum)",                # the header string is not one entity
    ],
)
def test_f_planted_fabricated_counterparty_still_fm005(fake):
    r = S.check_fabrication(output(HC003, entities=HC003_EP2_ENTITIES + [fake]), truth(HC003))
    assert r.failure_modes == ("FM-005",) and r.points == 0
    assert r.details["fabricated"] == [fake]
    assert S.FABRICATION_REVIEW in r.review_reasons


def test_f_schedule_item_ticker_of_another_asset_routes_to_review():
    """std-008 (USDC) shows an ETH item, WD-2511-006377: citing "ETH" is in the case
    file but is not the case's asset, so it routes to review, never FM-005 and never
    a clean pass."""
    case_id = "dev-std-008-USDC-M1"
    t = truth(case_id)
    assert any(i.asset == "ETH" for i in t.items)
    r = S.check_fabrication(output(case_id, entities=["HOT01", "ETH"]), t)
    assert "FM-005" not in r.failure_modes and r.points == 0 and r.requires_human_review
    assert r.details["schedule_tickers_cited"] == ["ETH"]


def test_f_planted_ticker_absent_from_the_schedule_still_fm005():
    case_id = "dev-std-008-USDC-M1"
    t = truth(case_id)
    assert not any(i.asset == "XRP" for i in t.items)
    r = S.check_fabrication(output(case_id, entities=["HOT01", "XRP"]), t)
    assert r.failure_modes == ("FM-005",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f_every_asset_own_ticker_and_chain_pass(asset):
    t = make_truth(asset)
    for cited in (asset, ASSET_CHAIN[asset], asset.lower()):
        r = S.check_fabrication({**good_output(t), "entities_cited": [t.custodian, cited]}, t)
        assert r.failure_modes == (), cited


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f_every_asset_foreign_ticker_still_fm005(asset):
    t = make_truth(asset)
    shown = {i.asset for i in t.items} | {asset}
    foreign = next(a for a in ALL_ASSETS if a not in shown)
    r = S.check_fabrication({**good_output(t), "entities_cited": [t.custodian, foreign]}, t)
    assert r.failure_modes == ("FM-005",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f_cited_ticker_never_grounds_a_name_containing_it(asset):
    """Citing the ticker grounds the ticker alone; "<ticker> Custody" in the
    narrative is still a sweep hit (whole-entity equality, section 6)."""
    t = make_truth(asset)
    out = {**good_output(t), "entities_cited": [t.custodian, asset],
           "narrative": f"Part of the balance sits with {asset} Custody."}
    r = S.check_fabrication(out, t)
    assert r.failure_modes == () and r.requires_human_review
    assert any(f"{asset} Custody" in h["token"] for h in r.details["sweep_hits"])


# =====================================================================================
# The sections "Pattern G, part 1", "Pattern G, part 2", "Contract C1 wiring" and
# "Narrowing patch (lane P2)" that followed here were removed with the 2026-11-06
# revert of the G1 and G2 rules in scoring.py (FIX-ROUND-2-REVERT-BRIEF.md, R2.md).
# The structural reliance fix is pending. narrative.exclusion_spans (contract C1)
# stays in narrative.py, unwired, with its own tests in tests/test_narrative*.py and
# tests/test_fix2_band_asset.py.
# =====================================================================================
