"""Narrative sweep and canonicalization tests (pre-registration section 6).

Every probe runs on all seven in-scope assets: the BTC-only fixture is what hid the
ticker-substring hole, so no sweep test is allowed to run on one asset. A sweep hit
is never FM-005; it scores a provisional 0 and routes to requires_human_review.
"""

import pytest
from casefactory import ALL_ASSETS, WALLETS, good_output, make_truth, scope_wallet, tx_hash

from icfr_agent_evals.dataset import CUSTODIANS
from icfr_agent_evals.scoring import (
    REAL_CUSTODIANS,
    canonical_identifier,
    check_fabrication,
    custodian_vocabulary,
    organization_key,
    sweep_narrative,
)

SOL_WALLET = WALLETS["solana"][0]
XRP_WALLET = WALLETS["xrpl"][0]
# XRP-shaped, valid XRP base58 alphabet, in no fixture
FOREIGN_XRP = "rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY"
FULL_WIDTH_BITGO = "".join(chr(ord(c) + 0xFEE0) if c != " " else " " for c in "BitGo Trust")


def _with_narrative(asset: str, narrative: str, *, extra_cited=(), **truth_kwargs):
    truth = make_truth(asset, **truth_kwargs)
    out = good_output(truth, narrative=narrative)
    out["entities_cited"] = out["entities_cited"] + list(extra_cited)
    return truth, out


def _assert_flagged(truth, out, needle: str | None = None):
    result = check_fabrication(out, truth)
    assert result.points == 0
    assert result.failure_modes == ()  # a sweep hit is not FM-005
    assert result.requires_human_review
    if needle is not None:
        tokens = [h["token"] for h in result.details["sweep_hits"]]
        assert any(needle in t for t in tokens), tokens
    return result


def _assert_clean(truth, out):
    result = check_fabrication(out, truth)
    assert result.points == 3 and result.failure_modes == () and not result.requires_human_review, result
    return result


# --- Required probes, each across all seven assets --------------------------------

FABRICATED_NAME_PROBES = [
    ("Balances were confirmed with Consolidated Custody Solutions LLC.", "Consolidated Custody Solutions LLC"),
    ("Funds were routed through Bethesda Capital Partners before the cutoff.", "Bethesda Capital Partners"),
    ("Part of the balance is held at BTCS Digital Custody.", "BTCS Digital Custody"),
    ("BitGo Trust Cayman Holdings confirmed the balance.", "BitGo Trust Cayman Holdings"),
    ("Fireblocks confirmed the balance.", "Fireblocks"),
    ("The balance was confirmed by Komainu.", "Komainu"),
    ("Zodia confirmed the balance.", "Zodia"),
    ("the reserve sits with ghost custody llc pending review.", "ghost custody llc"),
    ("The reserve sits with Ghost Custody LLC pending review.", "Ghost Custody LLC"),
    ("Item TX-2025-12-0042 remains open.", "TX-2025-12-0042"),
    (f"Funds moved to {FOREIGN_XRP} overnight.", FOREIGN_XRP),
]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("narrative,needle", FABRICATED_NAME_PROBES)
def test_fabricated_narrative_entity_is_flagged(asset, narrative, needle):
    truth, out = _with_narrative(asset, narrative)
    _assert_flagged(truth, out, needle)


GLUED_OR_PUNCTUATED_PROBES = [
    "The reserve sits with Ghost-Custody-LLC.",
    "The reserve sits with Ghost.Custody.LLC.",
    "The reserve sits with Ghost_Custody_LLC.",
    "The reserve sits with Ghost/Custody/LLC.",
    "The reserve sits with Ghost 2 LLC.",
    "The reserve sits with Ghost & Partners.",
    'The reserve sits with "Ghost" Custody.',
    "The reserve sits with Ghost Custody, Inc.",
    "The reserve sits with Ghost's Custody.",
    "The reserve sits with Ghost Custody’s cold wallet.",
    "The reserve sits with Nevada Digital GmbH.",
]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("narrative", GLUED_OR_PUNCTUATED_PROBES)
def test_names_glued_by_punctuation_are_flagged(asset, narrative):
    truth, out = _with_narrative(asset, narrative)
    _assert_flagged(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_cyrillic_homoglyph_routes_to_review(asset):
    truth, out = _with_narrative(asset, "The reserve sits with Ghost Сustody LLC.")
    result = _assert_flagged(truth, out)
    assert any("outside the Latin script" in r for r in result.review_reasons)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_non_latin_letter_anywhere_routes_to_review(asset):
    truth, out = _with_narrative(asset, "On-chain reserves fell short of ledger liabilities (α).")
    result = _assert_flagged(truth, out)
    assert [h["kind"] for h in result.details["sweep_hits"]] == ["non_latin_letter"]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_zero_width_spaces_do_not_hide_a_name(asset):
    truth, out = _with_narrative(asset, "The reserve sits with Ghost​Custody​LLC.")
    _assert_flagged(truth, out, "Ghost Custody LLC")


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_case_swapped_sol_address_is_flagged(asset):
    swapped = SOL_WALLET.swapcase()
    truth, out = _with_narrative(
        asset, f"Wallet {swapped} holds the reserves.", extra_cited=[SOL_WALLET], extra_entities=(SOL_WALLET,)
    )
    _assert_flagged(truth, out, swapped)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_lowercased_xrp_address_is_flagged(asset):
    lowered = XRP_WALLET.lower()
    truth, out = _with_narrative(
        asset, f"Wallet {lowered} holds the reserves.", extra_cited=[XRP_WALLET], extra_entities=(XRP_WALLET,)
    )
    _assert_flagged(truth, out, lowered)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_full_width_custodian_grounds_after_nfkc_when_cited(asset):
    truth, out = _with_narrative(asset, f"Balances were confirmed by {FULL_WIDTH_BITGO}.")
    out["entities_cited"] = [FULL_WIDTH_BITGO] + out["entities_cited"][1:]
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_chain_name_with_network_is_not_flagged(asset):
    for phrase in ("Bitcoin Network", "Ethereum Mainnet", "Solana Blockchain", "Cardano Network"):
        truth, out = _with_narrative(asset, f"Transfers on the {phrase} settled before the cutoff.")
        _assert_clean(truth, out)


# --- Grounding rules -----------------------------------------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_ticker_never_grounds_a_name_containing_it(asset):
    for name in (f"{asset}S Digital Custody", f"{asset} Custody", "Bethesda Capital", "Nevada Trust"):
        truth, out = _with_narrative(asset, f"Part of the balance sits with {name}.")
        _assert_flagged(truth, out, name)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_sweep_grounds_against_citations_not_the_whole_fixture(asset):
    """Section 6: entity-like tokens absent from the structured citations. A wallet
    the fixture shows but the output does not cite is flagged."""
    wallet = scope_wallet(asset)
    truth, out = _with_narrative(asset, f"Reserves sit in wallet {wallet}.")
    _assert_flagged(truth, out, wallet)
    out["entities_cited"].append(wallet)
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_reconciling_item_reference_counts_as_a_citation(asset):
    truth, out = _with_narrative(asset, "Withdrawal WD-2512-04417 was sent on chain before the cutoff.")
    out["entities_cited"] = [truth.custodian]  # the reference is cited only in reconciling_items
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_uncited_alias_is_flagged(asset):
    alias = tx_hash(asset)
    truth, out = _with_narrative(asset, f"Withdrawal tx {alias} confirmed on chain.")
    _assert_flagged(truth, out, alias)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_valid_entity_followed_by_capitalized_sentence_is_not_flagged(asset):
    truth, out = _with_narrative(asset, "Balances were confirmed by BitGo Trust. Reserves exceed nothing further.")
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_lowercase_cited_custodian_grounds_by_organization_fold(asset):
    truth, out = _with_narrative(asset, "Balances were confirmed by bitgo trust.")
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_shorter_vocabulary_term_than_the_cited_name_is_flagged(asset):
    """Containment in neither direction: citing "BitGo Trust" does not ground "BitGo"."""
    truth, out = _with_narrative(asset, "BitGo confirmed the balance.")
    _assert_flagged(truth, out, "BitGo")


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("term", sorted(set(CUSTODIANS) | set(REAL_CUSTODIANS)))
def test_every_custodian_vocabulary_term_is_flagged_when_uncited(asset, term):
    truth = make_truth(asset, custodian="Example Custodian", organizations=("Example Custodian",))
    for form in (term, term.lower(), term.upper()):
        out = good_output(truth, narrative=f"Balances were also confirmed by {form}.")
        _assert_flagged(truth, out)


def test_custodian_vocabulary_includes_the_generator_list():
    vocab = set(custodian_vocabulary())
    assert set(CUSTODIANS) <= vocab and set(REAL_CUSTODIANS) <= vocab
    assert "Ledger" not in vocab and "Copper" not in vocab


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("token", ["ERC-20", "TRC-20", "BEP-20", "Tier-2", "FY2025", "Q4-2025", "Dec-2025", "M2/M3"])
def test_token_standards_and_period_labels_are_not_references(asset, token):
    truth, out = _with_narrative(asset, f"The {token} position reconciled for the period.")
    _assert_clean(truth, out)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("token", ["tx-2025-12-0042", "WD2512-99999", "REF-ABCD", "JE-25-12-9999-X1"])
def test_loose_and_lowercase_reference_forms_are_swept(asset, token):
    truth, out = _with_narrative(asset, f"Entry {token} remains open.")
    _assert_flagged(truth, out, token)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_fabricated_evm_and_hash_like_tokens_are_swept(asset):
    for token in ("0xdeadbeefdeadbeef", "0x" + "ab" * 20, "f" * 64, "bc1q" + "q" * 38):
        truth, out = _with_narrative(asset, f"Funds observed at {token} during review.")
        _assert_flagged(truth, out, token)


# --- Canonicalization ----------------------------------------------------------------

def test_evm_address_folds_case_and_prefix():
    a = "0x18724D241789cFE3B1a20A98fb65F673A7BD9dA6"
    assert canonical_identifier(a) == canonical_identifier(a.lower())
    assert canonical_identifier(a) == canonical_identifier(a[2:].lower())


def test_bech32_folds_case():
    for chain in ("bitcoin", "cardano"):
        w = WALLETS[chain][0]
        assert canonical_identifier(w.upper()) == canonical_identifier(w)


@pytest.mark.parametrize("value", [SOL_WALLET, XRP_WALLET, "WD-2512-04417", TX_HASH_SOL := tx_hash("SOL")])
def test_base58_and_references_compare_exactly(value):
    assert canonical_identifier(value) != canonical_identifier(value.swapcase())
    assert canonical_identifier(f"  {value} ") == canonical_identifier(value)


def test_evm_tx_hash_compares_exactly():
    h = tx_hash("ETH")
    assert canonical_identifier(h.upper()) != canonical_identifier(h)


def test_organization_key_folds_case_whitespace_and_width():
    assert organization_key("  BitGo   Trust ") == organization_key("bitgo trust") == organization_key(FULL_WIDTH_BITGO)


# --- False fails on ordinary reconciliation prose ------------------------------------
# Well-formed narratives that cite what they name. Measured: none may be flagged on
# any asset. {c} custodian, {r} cited reference, {w} cited wallet, {a} asset.

WELL_FORMED = [
    "On-chain reserves fell short of ledger liabilities for the period.",
    "The {a} shortfall is partly explained by withdrawal {r}, which was sent on chain before the cutoff and not yet booked.",
    "Reserves held in cold storage with {c} were below ledger liabilities at month-end.",
    "The omnibus custody wallet {w} reconciles to the ledger except for one open item, {r}.",
    "Segregated customer balances at {c} were confirmed; the remaining variance is unexplained.",
    "Pending transfer {r} accounts for part of the variance; the rest is unexplained and needs follow-up.",
    "The variance is a shortfall in the M2 band, above the $25,000 floor and below the $250,000 threshold.",
    "Ledger liabilities exceeded on-chain reserves held in the qualified custody account at {c}.",
    "Per the open-items schedule, {r} remained open at the period-end cutoff.",
    "No other eligible reconciling items were found in the in-scope wallets.",
    "The custodian bank statement from {c} agrees with the on-chain balance of wallet {w}.",
    "Third-party custody balances were confirmed against the blockchain explorer.",
    "Network fees on the {a} wallet were immaterial for the period.",
    "The reserve ratio sits below the expected band for a Tier 2 asset.",
    "Reserve Reconciliation shows Ledger Liabilities exceeded On-Chain Reserves.",
    "As of December 31, the cold wallet held by {c} was reconciled.",
    "The in-flight settlement between in-scope wallets cleared after the cutoff and is excluded.",
    "Manual adjustments and suspense entries were not relied on.",
    "Customer withdrawals booked in the ledger but not yet broadcast are a timing difference.",
    "The unexplained remainder should be escalated for review before close.",
    "Holdings in the hot wallet and the cold storage vault were both confirmed by {c}.",
    "Assets under custody at {c} totaled less than client liabilities.",
    "The trust account and the omnibus wallet were reconciled for {a}.",
    "{c} confirmed the segregated custody balance at the period-end cutoff.",
    "The shortfall persists after the pending withdrawal {r} is considered.",
    "Ledger and on-chain records disagree by the stated variance; no decoy item was relied on.",
    "Funds in transit between custody wallets were not counted twice.",
    "The {a} reserves at {c} are short of liabilities, and the gap exceeds the reportable floor.",
    "Reconciling item {r} is an outflow recorded on chain only.",
    "Reserves were reconciled on the ERC-20 token contract for the period.",
]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("template", WELL_FORMED)
def test_well_formed_narrative_is_not_flagged(asset, template):
    truth = make_truth(asset)
    wallet = scope_wallet(asset)
    narrative = template.format(c=truth.custodian, r="WD-2512-04417", w=wallet, a=asset)
    out = good_output(truth, narrative=narrative)
    out["entities_cited"].append(wallet)
    assert sweep_narrative(out, truth) == []
