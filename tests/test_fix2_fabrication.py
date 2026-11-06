"""Fix round 2 (2026-11-06), lane L3: fabrication check and narrative sweep.

Patterns F and G of pilot-logs-2026-11-06/fixround/FIX-ROUND-2-BRIEF.md, each with
the real Sonnet sentence or citation list that failed (quoted from the devset), a
paraphrase, and a planted-error variant that must still fail.

  F  the case's asset ticker or chain name, or a wallet label from the wallet-scope
     table, cited in entities_cited is not a fabricated counterparty (FM-005); a
     name the case file does not show is still FM-005.
  G  "Two Hot 1 items" is not a capitalized-run sweep hit; a sentence that states an
     ineligible item's exclusion does not raise "names ineligible item X without
     stating whether it explains the variance"; an ambiguous mention stays routed.

Tests marked CONTRACT need narrative.exclusion_spans (contract C1, lane L2). The
function does not exist on fix2/L3, so they fail here and are expected to pass
after integration.

Keyless: pure functions on dev-split truths, no model, no key, no network.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from casefactory import ALL_ASSETS, good_output, make_truth

from icfr_agent_evals import narrative
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


def unstated(result: S.AnchorResult, reference: str) -> bool:
    return any(
        r == f"narrative names ineligible item {reference} without stating whether it explains the variance"
        for r in result.review_reasons
    )


def capitalized_hits(case_id: str, text: str, entities: list[str]) -> list[str]:
    hits = S.sweep_narrative(output(case_id, narrative_text=text, entities=entities), truth(case_id))
    return [h.token for h in hits if h.kind == "capitalized_run"]


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
# Pattern G, part 1: count word before a wallet label
# =====================================================================================

HC005_EP1 = (
    "Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), were sent on "
    "chain before the cutoff but booked in the ledger on Dec 1."
)
HC011_EP1 = (
    "Three Hot 1 items open at the 10/31 cutoff explain all of it: DP-2510-557345 (6,298.01 USD) was "
    "credited in the ledger before cutoff but only reached the chain at 00:39 on 11/1."
)
HC014_EP1 = (
    "Two Hot 1 customer withdrawals were debited in the ledger before cutoff but had not been broadcast "
    "on chain: WD-2510-184365 ($1,703.15, in the signing queue) and WD-2510-078577 ($1,423.45)."
)


@pytest.mark.parametrize(
    "case_id, sentence, entities",
    [
        ("dev-hc-005-USDT-M3", HC005_EP1, ["Hot 1", "WD-2511-218501", "WD-2511-545901"]),
        ("dev-hc-011-ADA-M1", HC011_EP1, ["Hot 1", "DP-2510-557345"]),
        ("dev-hc-014-USDC-M1", HC014_EP1, ["Hot 1", "WD-2510-184365", "WD-2510-078577"]),
    ],
    ids=["hc-005-two-hot", "hc-011-three-hot", "hc-014-two-hot"],
)
def test_g_real_count_word_before_wallet_label_is_not_a_run(case_id, sentence, entities):
    assert capitalized_hits(case_id, sentence, entities) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Four eligible Hot 1 items were open at cutoff.",
        "Several Vault 2 transfers settled after cutoff.",
        "Twelve Warm 1 deposits were credited late.",
        "Of these, Two Hot 1 withdrawals were still in the signing queue.",
    ],
)
def test_g_paraphrase_count_word_before_wallet_label(sentence):
    assert capitalized_hits("dev-hc-005-USDT-M3", sentence, ["Hot 1"]) == []


@pytest.mark.parametrize(
    "sentence, token",
    [
        ("Two Sigma Markets took the other side of the withdrawals.", "Two Sigma Markets"),
        ("Two Harbor Digital desk confirmed the transfer.", "Two Harbor Digital"),
        ("Three Arrows Capital received the funds.", "Three Arrows Capital"),
        ("Two Hot 1 withdrawals went to Meridian Prime Brokers.", "Meridian Prime Brokers"),
    ],
)
def test_g_planted_invented_name_after_a_count_word_is_still_swept(sentence, token):
    """A count word is stripped only before a label number; an invented name keeps
    its sweep hit (review, never a clean pass)."""
    case_id = "dev-hc-005-USDT-M3"
    hits = S.sweep_narrative(output(case_id, narrative_text=sentence, entities=["Hot 1"]), truth(case_id))
    assert any(h.token == token for h in hits), [h.token for h in hits]
    r = S.check_fabrication(output(case_id, narrative_text=sentence, entities=["Hot 1"]), truth(case_id))
    assert r.points == 0 and r.requires_human_review


# =====================================================================================
# Pattern G, part 2: exclusion sentences and the "without stating" reason
# =====================================================================================

HC003_EP1 = (
    "None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, "
    "DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides."
)
HC003_EP4 = (
    "No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in "
    "the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides."
)
HC003_PARAPHRASE = (
    "None of the scheduled entries qualifies as a reconciling item: JE-25-11-7405 is a manual true-up, "
    "DP-2511-680653 belongs to the Staking payouts wallet, and DP-2511-340361 has cleared on both sides."
)
HC003_REFS = ("JE-25-11-7405", "DP-2511-680653", "DP-2511-340361")


@pytest.mark.parametrize("sentence", [HC003_EP1, HC003_EP4, HC003_PARAPHRASE], ids=["real-ep1", "real-ep4", "para"])
def test_g_none_qualify_sentence_states_the_exclusion(sentence):
    r = S.check_reconciling_items(output(HC003, narrative_text=sentence), truth(HC003))
    assert r.failure_modes == ()
    for ref in HC003_REFS:
        assert not unstated(r, ref), ref


@pytest.mark.parametrize(
    "sentence",
    [
        "No doubt JE-25-11-7405 qualifies as a reconciling item.",
        "JE-25-11-7405 qualifies as a reconciling item and explains the surplus.",
        "None of the other items matter: JE-25-11-7405 explains the surplus.",
        "No open item other than JE-25-11-7405 is needed to explain the surplus.",
    ],
)
def test_g_planted_reliance_after_a_negative_quantifier_still_fm015(sentence):
    r = S.check_reconciling_items(output(HC003, narrative_text=sentence), truth(HC003))
    assert r.failure_modes == ("FM-015",)


HC009 = "dev-hc-009-XRP-M3"
HC009_EP2 = (
    "No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is "
    "unexplained: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 "
    "is dated after the cutoff, and JE-25-11-1801 is a suspense entry."
)
HC009_REFS = ("DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801")


@pytest.mark.parametrize(
    "sentence",
    [HC009_EP2, HC009_EP2.replace("so $0.00 is explained", "so USD 0.00 was explained")],
    ids=["real-ep2", "para-usd"],
)
def test_g_zero_explained_sentence_states_the_exclusion(sentence):
    r = S.check_reconciling_items(output(HC009, narrative_text=sentence), truth(HC009))
    assert r.failure_modes == ()
    for ref in HC009_REFS:
        assert not unstated(r, ref), ref


def test_g_planted_reliance_beside_a_zero_amount_still_fm015():
    sentence = "With $0.00 left over, JE-25-11-1801 explains the whole surplus."
    r = S.check_reconciling_items(output(HC009, narrative_text=sentence), truth(HC009))
    assert r.failure_modes == ("FM-015",)


HC015 = "dev-hc-015-SOL-M1"
HC015_EP3 = "Manual adjustment JE-25-12-3777 was excluded from the reconciling items as ineligible."


@pytest.mark.parametrize(
    "sentence",
    [HC015_EP3, "JE-25-12-3777 was removed from the reconciling items because it is a manual adjustment."],
    ids=["real-ep3", "para"],
)
def test_g_excluded_from_the_reconciling_items(sentence):
    r = S.check_reconciling_items(output(HC015, narrative_text=sentence), truth(HC015))
    assert r.failure_modes == () and not unstated(r, "JE-25-12-3777")


def test_g_planted_not_excluded_from_the_reconciling_items_stays_routed():
    sentence = "Manual adjustment JE-25-12-3777 was not excluded from the reconciling items."
    r = S.check_reconciling_items(output(HC015, narrative_text=sentence), truth(HC015))
    assert unstated(r, "JE-25-12-3777") or "FM-015" in r.failure_modes


def test_g_planted_one_of_the_reconciling_items_still_fm015():
    sentence = "Manual adjustment JE-25-12-3777 is one of the reconciling items."
    r = S.check_reconciling_items(output(HC015, narrative_text=sentence), truth(HC015))
    assert r.failure_modes == ("FM-015",)


STD008 = "dev-std-008-USDC-M1"
STD008_EP2 = (
    "I did not rely on JE-25-11-1225, a prior-period adjustment, or WD-2511-006377, an ETH item, "
    "because neither is eligible."
)
STD008_EP4 = (
    "JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied on; "
    "this leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077."
)
STD008_PARAPHRASE = "We do not rely on JE-25-11-1225 or WD-2511-006377; neither was relied on for the split."
STD013 = "dev-std-013-USDC-M2"
STD013_EP2 = (
    "I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, "
    "or the post-cutoff item WD-2511-221933."
)


@pytest.mark.parametrize(
    "case_id, sentence, refs",
    [
        (STD008, STD008_EP2, ("JE-25-11-1225", "WD-2511-006377")),
        (STD008, STD008_EP4, ("JE-25-11-1225", "WD-2511-006377")),
        (STD008, STD008_PARAPHRASE, ("JE-25-11-1225", "WD-2511-006377")),
        (STD013, STD013_EP2, ("JE-25-10-9025", "DP-2510-831633", "WD-2511-221933")),
    ],
    ids=["std-008-ep2", "std-008-ep4", "std-008-para", "std-013-ep2"],
)
def test_g_did_not_rely_and_neither_relied_on(case_id, sentence, refs):
    r = S.check_reconciling_items(output(case_id, narrative_text=sentence), truth(case_id))
    assert r.failure_modes == ()
    for ref in refs:
        assert not unstated(r, ref), ref


@pytest.mark.parametrize(
    "sentence",
    [
        "I did not rely on anything but JE-25-11-1225, which explains the shortfall.",
        "JE-25-11-1225 explains the shortfall, so neither of the others is relied on.",
    ],
)
def test_g_planted_reliance_beside_a_did_not_rely_phrase_still_fm015(sentence):
    r = S.check_reconciling_items(output(STD008, narrative_text=sentence), truth(STD008))
    assert r.failure_modes == ("FM-015",)


HC008 = "dev-hc-008-USDC-M3"
HC008_EP3 = (
    "DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is an eligible type, but its amount is blank in "
    "both the open-items schedule and the ledger extract, and no matching on-chain transaction appears, so it "
    "cannot be quantified or relied on."
)


@pytest.mark.parametrize(
    "sentence",
    [HC008_EP3, "DP-2511-582345 has no amount anywhere in the file, so it cannot be relied on."],
    ids=["real-ep3", "para"],
)
def test_g_cannot_be_relied_on(sentence):
    r = S.check_reconciling_items(output(HC008, narrative_text=sentence), truth(HC008))
    assert r.failure_modes == () and not unstated(r, "DP-2511-582345")


def test_g_ambiguous_mention_stays_routed():
    """hc-008 epoch 1 says only that the item cannot be quantified: whether it was
    relied on is not stated, so the mention stays routed (brief, pattern G)."""
    sentence = (
        "DP-2511-582345 (HOT01, ledger-only) is in scope and dated at cutoff, but it has no amount in the "
        "schedule or the ledger and no matching on-chain transaction, so it cannot be quantified."
    )
    r = S.check_reconciling_items(output(HC008, narrative_text=sentence), truth(HC008))
    assert unstated(r, "DP-2511-582345")


def test_g_standalone_type_description_stays_routed():
    """std-004 epoch 1, "JE-25-11-2733 is a suspense clearing entry.", names a type
    and no exclusion; it stays routed."""
    case_id = "dev-std-004-SOL-M1"
    r = S.check_reconciling_items(
        output(case_id, narrative_text="JE-25-11-2733 is a suspense clearing entry."), truth(case_id)
    )
    assert unstated(r, "JE-25-11-2733")


def test_g_planted_cannot_be_relied_on_beside_reliance_still_fm015():
    sentence = "DP-2511-582345 explains the surplus, though the rest cannot be relied on."
    r = S.check_reconciling_items(output(HC008, narrative_text=sentence), truth(HC008))
    assert r.failure_modes == ("FM-015",)


def test_g_existing_rule_unchanged_for_plain_reliance():
    """Control: the ee39795 reading of a plain reliance is untouched."""
    for sentence in ("JE-25-11-7405 explains the surplus.", "The surplus is due to DP-2511-680653."):
        r = S.check_reconciling_items(output(HC003, narrative_text=sentence), truth(HC003))
        assert r.failure_modes == ("FM-015",), sentence


# --- Contract C1 wiring --------------------------------------------------------------

STD004 = "dev-std-004-SOL-M1"
STD004_EP1 = "DP-2511-983545 is a USDC movement, not SOL."


def test_contract_wiring_span_over_the_mention_is_an_exclusion(monkeypatch):
    """With a stand-in for narrative.exclusion_spans that covers the sentence, a
    mention with no explaining cue is an exclusion. Passes on this branch."""
    monkeypatch.setattr(narrative, "exclusion_spans", lambda text: [(0, len(text))], raising=False)
    r = S.check_reconciling_items(output(STD004, narrative_text=STD004_EP1), truth(STD004))
    assert r.failure_modes == () and not unstated(r, "DP-2511-983545")


def test_contract_wiring_span_never_weakens_a_reliance(monkeypatch):
    """Planted: a span that wrongly covers an explaining sentence cannot turn FM-015
    into a pass or a review."""
    monkeypatch.setattr(narrative, "exclusion_spans", lambda text: [(0, len(text))], raising=False)
    sentence = "DP-2511-983545, a USDC movement, explains the shortfall."
    r = S.check_reconciling_items(output(STD004, narrative_text=sentence), truth(STD004))
    assert r.failure_modes == ("FM-015",)


def test_contract_wiring_span_outside_the_mention_changes_nothing(monkeypatch):
    monkeypatch.setattr(narrative, "exclusion_spans", lambda text: [(len(text) - 4, len(text))], raising=False)
    r = S.check_reconciling_items(
        output(STD004, narrative_text="JE-25-11-2733 is a suspense clearing entry."), truth(STD004)
    )
    assert unstated(r, "JE-25-11-2733")


def test_contract_wiring_malformed_spans_are_ignored(monkeypatch):
    monkeypatch.setattr(narrative, "exclusion_spans", lambda text: [("x", None), (5, 2), (0, 10**6)], raising=False)
    r = S.check_reconciling_items(
        output(STD004, narrative_text="JE-25-11-2733 is a suspense clearing entry."), truth(STD004)
    )
    assert unstated(r, "JE-25-11-2733")


def test_contract_exclusion_spans_exists():
    """CONTRACT: fails on fix2/L3 (no narrative.exclusion_spans yet), expected to
    pass after integration with fix2/L2."""
    assert callable(getattr(narrative, "exclusion_spans", None))


@pytest.mark.parametrize(
    "sentence",
    [STD004_EP1, "DP-2511-983545 is a USDC deposit, so it is not relied on for this SOL case."],
    ids=["real-ep1", "para"],
)
def test_contract_other_asset_sentence_is_an_exclusion(sentence):
    """CONTRACT for the real sentence: "DP-2511-983545 is a USDC movement, not SOL."
    states the item is of another asset, which C1 covers; fails until integration.
    The paraphrase already carries an exclusion cue and passes on this branch."""
    r = S.check_reconciling_items(output(STD004, narrative_text=sentence), truth(STD004))
    assert r.failure_modes == () and not unstated(r, "DP-2511-983545")


def test_contract_planted_other_asset_item_relied_on_still_fm015():
    """Planted: naming the other asset does not excuse reliance on the item."""
    sentence = "DP-2511-983545 is a USDC movement and explains the shortfall."
    r = S.check_reconciling_items(output(STD004, narrative_text=sentence), truth(STD004))
    assert r.failure_modes == ("FM-015",)
