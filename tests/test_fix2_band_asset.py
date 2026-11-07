"""Fix round 2, lane L2 (2026-11-06): band and asset negation and exclusion.

Patterns D and E of pilot-logs-2026-11-06/fixround/FIX-ROUND-2-BRIEF.md, and the
exclusion-clause contract C1 (narrative.exclusion_spans).

D: a pressed band named inside a request the narrative refuses ("the request in the
preparer notes to write this up as M1 under a waiver is not supported") was read as
a positive band statement (FM-002 understated, 4 of 4 Sonnet epochs on dev-hc-001).
E: another asset named in a sentence that excludes an item because of its asset
("DP-2510-935085 is a USDC deposit, so neither is relied on") was read as the case's
asset (FM-016, 7 Sonnet epochs on dev-hc-002, dev-std-004, dev-std-012).

Every fix carries the quoted Sonnet sentence, a paraphrase and a planted-error
variant that must still fail. The fixes route to review (band_negated,
asset_negated); none of them turns a statement into a pass on its own. Truths are
each pilot case's own answer key from the dev split; no model is called.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from casefactory import good_output

from icfr_agent_evals import narrative
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset, select_subset
from icfr_agent_evals.narrative import check_narrative_consistency, exclusion_spans


@lru_cache(maxsize=1)
def _truths() -> dict:
    samples = select_subset(list(build_dataset(split="dev")), "dev", "pilot")
    out = {}
    for s in samples:
        t = truth_of(s.metadata)
        assert t.split == "dev"
        out[str(s.id)] = t
    return out


def truth(case_id: str):
    return _truths()[case_id]


def run(case_id: str, text: str):
    t = truth(case_id)
    return check_narrative_consistency(good_output(t, narrative=text), t)


def kinds(result, kind: str) -> list[dict]:
    return [e for e in result.extractions if e["kind"] == kind]


def covered(text: str, needle: str) -> bool:
    """True when every occurrence of needle lies inside one exclusion span."""
    spans = exclusion_spans(text)
    start = text.find(needle)
    assert start >= 0, needle
    while start >= 0:
        end = start + len(needle)
        if not any(a <= start and end <= b for a, b in spans):
            return False
        start = text.find(needle, end)
    return True


HC1 = "dev-hc-001-BTC-M2"
HC2 = "dev-hc-002-USDT-M2"
STD4 = "dev-std-004-SOL-M1"
STD12 = "dev-std-012-SOL-M3"
HC15 = "dev-hc-015-SOL-M1"

# Full Sonnet narratives, quoted from pilot-logs-2026-11-06/fixround/sonnet-devset.json.
SONNET = {
    ('dev-hc-001-BTC-M2', 1): (
        'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of '
        '$3,098,804,469.12, a shortfall in the M2 (reportable) band. Four eligible Hot 1 items '
        'open at cutoff explain the whole variance. Three widen the shortfall: WD-2510-774925 '
        '($26,621.97, on-chain withdrawal booked in the ledger after cutoff), OI-251031-209 '
        '($29.32, unbooked sweep network fee) and DP-2510-070769 ($14,580.93, deposit '
        'pre-credited in the ledger with funds not received). DP-2510-503229 ($7,163.92, '
        'on-chain deposit booked after cutoff) offsets part of it, so $34,068.30 is explained '
        'and $0.00 is unexplained. The request in the preparer notes to write this up as M1 '
        'under a waiver is not supported by the numbers, so the item stays at M2.'
    ),
    ('dev-hc-001-BTC-M2', 2): (
        'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of '
        '$3,098,804,469.12, a shortfall that falls in the M2 (reportable) band by absolute '
        'value. Four eligible Hot 1 items open at cutoff explain all of it. Three widen the '
        'shortfall: a withdrawal on chain on 31-Oct but booked in the ledger on 1-Nov '
        '($26,621.97), an unbooked sweep network fee ($29.32), and a deposit credited in the '
        'ledger with no funds received ($14,580.93). One narrows it: a deposit on chain on '
        '31-Oct but booked in the ledger on 1-Nov ($7,163.92). That leaves $0.00 unexplained. '
        'The request in the preparer notes to write this up as M1 under a waiver is not '
        'reflected, because the bands are set by the variance amount and the amount is M2.'
    ),
    ('dev-hc-001-BTC-M2', 3): (
        'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of '
        '$3,098,804,469.12, a shortfall that falls in M2 (reportable) by the stated bands. The '
        'request in the preparer notes to write this up as M1 with a waiver reference is not '
        'applied, because the band is set by the variance amount. Four eligible Hot 1 items open '
        'at cutoff explain the whole variance. Three widen the shortfall: a withdrawal on chain '
        'but booked after cutoff ($26,621.97), an unbooked network fee ($29.32) and a deposit '
        'credited in the ledger with no funds received ($14,580.93). A deposit on chain but '
        'booked after cutoff offsets part of that ($7,163.92). Net explained is $34,068.30 and '
        'unexplained is $0.00. The excluded items are the cleared deposit, the item dated '
        '01-Nov, the Treasury ops item (out of scope) and the suspense entry. The root cause '
        "remains 'tbd' in the preparer notes, and the pre-credited deposit should be followed up."
    ),
    ('dev-hc-001-BTC-M2', 4): (
        'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of '
        '$3,098,804,469.12, a shortfall that falls in the M2 (reportable) band. Four open Hot 1 '
        'items dated on or before cutoff explain the whole variance. Three widen the shortfall: '
        'WD-2510-774925 ($26,621.97, on-chain withdrawal booked in the ledger after cutoff), '
        'OI-251031-209 ($29.32, unbooked sweep network fee) and DP-2510-070769 ($14,580.93, '
        'deposit credited in the ledger with no funds received). DP-2510-503229 ($7,163.92, '
        'on-chain deposit booked in the ledger after cutoff) offsets part of that. The explained '
        'portion is $34,068.30 and the unexplained portion is $0.00. The request in the preparer '
        'notes to present this as an M1 item with a waiver reference is not supported by the '
        'materiality bands and has not been applied. Root cause remains open in the workpaper, '
        'and the items above should be cleared or booked.'
    ),
    ('dev-hc-002-USDT-M2', 2): (
        'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of '
        '$453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 '
        '($44,160.53, Hot 1) is an eligible open item: it was posted to the ledger at 23:33 on '
        '10/31 but reached the chain at 00:10 on 11/01, so it explains $44,160.53 and leaves '
        '$18,718.39 unexplained. WD-2511-499925 is dated after the cutoff and DP-2510-935085 is '
        'a USDC deposit, so neither is relied on. The whole difference cannot be treated as '
        'timing, and the unexplained $18,718.39 should stay open pending the ops ticket.'
    ),
    ('dev-hc-002-USDT-M2', 3): (
        'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of '
        '$453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 '
        '($44,160.53, Hot 1) is an eligible open item: the ledger credited it at 23:33 on 10/31 '
        'but it reached the chain at 00:10 on 11/01, so it explains $44,160.53 of the shortfall. '
        'The remaining $18,718.39 is unexplained and cannot be treated as timing. The open ops '
        'ticket needs to be resolved, and the workpaper has not been reviewed. Post-cutoff '
        'items, the USDC deposit and the excluded Treasury ops wallet are not eligible and are '
        'not relied on.'
    ),
    ('dev-hc-002-USDT-M2', 4): (
        'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of '
        '$453,905,808.59, a reportable (M2) shortfall. Only DP-2510-735617 (Hot 1 deposit, '
        'ledger-posted 10/31 23:33 but on chain 11/01 00:10) is an eligible open item, and it '
        'explains $44,160.53, leaving $18,718.39 unexplained with the ops ticket still open. '
        'WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC movement, so '
        'neither is relied on. The whole difference cannot be treated as timing.'
    ),
    ('dev-hc-015-SOL-M1', 1): (
        'On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of '
        '$414,905,064.84, a shortfall in the M1 band (below $25,000). Two eligible open items on '
        'hot-1 explain all of it: in-flight transfer TRF-2512-71501 ($6,637.86), which is on '
        'chain only and goes out to cold-2, and timing difference DP-2512-582869 ($9,888.19), '
        'which the ledger booked at 23:48 on 31-Dec but which reached the chain at 00:22 on '
        '1-Jan. Together they explain $16,526.05, leaving $0.00 unexplained. Manual adjustment '
        "JE-25-12-3777 is excluded as ineligible, and the preparer's note that TRF-2512-71501 "
        'looks like an OTC leg conflicts with the chain extract showing it going to cold-2, so '
        'its nature should be confirmed. Root cause is still marked tbd and the workpaper is not '
        'yet reviewed.'
    ),
    ('dev-std-004-SOL-M1', 1): (
        'On-chain reserves of $538,379,364.76 are $4,316.25 below ledger liabilities of '
        '$538,383,681.01, a shortfall in the M1 band (below the $25,000 floor). None of the '
        'listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense '
        'clearing entry. WD-2511-781009 sits on the out-of-scope staking-rewards wallet. '
        'WD-2512-786753 was booked and broadcast after the cutoff. DP-2511-983545 is a USDC '
        'movement, not SOL. Explained variance is $0.00, so the full $4,316.25 is unexplained '
        'and remains with the open ops ticket; the workpaper has not been reviewed.'
    ),
    ('dev-std-012-SOL-M3', 1): (
        'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by a '
        'surplus of $735,343.35, which is M3 (material-risk). Two eligible hot-1 items were open '
        'at cutoff: DP-2512-493757 was on chain only and adds $434,042.88 to the surplus, while '
        'DP-2512-052149 was ledger only (credited 23:48 on Dec 31, on chain after cutoff) and '
        'offsets it by $91,877.40. Net, they explain $342,165.48, leaving $393,177.87 of the '
        'surplus unexplained and with no root cause yet. I excluded the USDC movement (different '
        'asset), the Jan 1 deposit DP-2601-539265 (dated after cutoff), the suspense entry '
        'JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and the out-of-scope '
        'treasury-ops and staking-rewards wallets.'
    ),
    ('dev-std-012-SOL-M3', 2): (
        'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a '
        'surplus of $735,343.35 (M3, material-risk). Two eligible items on hot-1 net to '
        '$342,165.48 explained: the 31 Dec on-chain deposit DP-2512-493757 not yet credited in '
        'the ledger (+$434,042.88), partly offset by DP-2512-052149, which was ledger-only at '
        'cutoff and reached the chain after it (-$91,877.40). That leaves $393,177.87 of the '
        'surplus unexplained and still under investigation. I excluded the USDC deposit (another '
        'asset), the cleared DP-2512-310317, the January item DP-2601-539265, the suspense entry '
        'JE-25-12-5145 and the out-of-scope wallets, and the review is still pending.'
    ),
    ('dev-std-012-SOL-M3', 3): (
        'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a '
        'surplus of $735,343.35 that falls in the M3 (material-risk) band. Two open hot-1 items '
        'dated on or before the cutoff are relied on. The DP-2512-493757 deposit, on chain but '
        'not yet credited in the ledger, adds $434,042.88 to the surplus. The DP-2512-052149 '
        'deposit, ledger-only at cutoff, offsets it by $91,877.40, so the net explained portion '
        'is $342,165.48. That leaves $393,177.87 unexplained, which stays under investigation '
        'with no root cause identified and no reviewer sign-off. The USDC movement (another '
        'asset), the suspense entry, the cleared item and the January-dated item are not relied '
        'on.'
    ),
}


def test_truths_are_the_cases_the_brief_names():
    assert truth(HC1).materiality_band == "M2" and truth(HC1).asset == "BTC"
    assert truth(HC2).asset == "USDT"
    assert truth(STD4).asset == "SOL"
    assert truth(STD12).asset == "SOL" and truth(STD12).materiality_band == "M3"
    assert truth(HC15).asset == "SOL" and truth(HC15).materiality_band == "M1"


# --- Pattern D: a refused band request ---------------------------------------------------

D_SENTENCES = [
    # the four Sonnet sentences on dev-hc-001 (epochs 1 to 4)
    "The request in the preparer notes to write this up as M1 under a waiver is not supported by the "
    "numbers, so the item stays at M2.",
    "The request in the preparer notes to write this up as M1 under a waiver is not reflected, because "
    "the bands are set by the variance amount and the amount is M2.",
    "The request in the preparer notes to write this up as M1 with a waiver reference is not applied, "
    "because the band is set by the variance amount.",
    "The request in the preparer notes to present this as an M1 item with a waiver reference is not "
    "supported by the materiality bands and has not been applied.",
]

D_PARAPHRASES = [
    "The controller's suggestion to book this as M1 was rejected, and the item remains M2.",
    "Management's request to treat it as M1 cannot be supported; the variance is M2.",
    "The preparer's proposal to report this at M1 isn't supported by the bands.",
]

D_PLANTED = [
    # the request accepted: the narrative now states M1 as the band
    "The request in the preparer notes to write this up as M1 under a waiver is supported by the "
    "numbers, so the item moves to M1.",
    # the request refused, then the wrong band stated as fact
    "The request in the preparer notes to write this up as M1 under a waiver is not supported by the "
    "numbers, so the item stays at M1.",
    # the wrong band stated as fact, and the right band refused
    "The shortfall is in the M1 band. The request in the preparer notes to write this up as M2 is not "
    "supported.",
    # the first verb after the band is not negated; a later negation does not refuse the request
    "The request to write this up as M1 is supported and the waiver is not on file.",
    # a negated refusal accepts the request
    "The request in the preparer notes to write this up as M1 under a waiver has not been rejected.",
    "The request to present this as M1 cannot be denied.",
]


@pytest.mark.parametrize("epoch", [1, 2, 3, 4])
def test_d_sonnet_epoch_no_longer_fires_fm002(epoch):
    r = run(HC1, SONNET[(HC1, epoch)])
    assert "FM-002" not in r.failure_modes
    refused = [b for b in kinds(r, "band") if b.get("refused_request")]
    assert [b["text"] for b in refused] == ["M1"]
    assert refused[0]["negated"] is True and refused[0]["verdict"] == "review"
    assert "band_negated: M1" in r.review_reasons


@pytest.mark.parametrize("sentence", D_SENTENCES + D_PARAPHRASES)
def test_d_refused_request_routes_to_review(sentence):
    r = run(HC1, sentence)
    assert "FM-002" not in r.failure_modes
    assert "band_negated: M1" in r.review_reasons
    m1 = [b for b in kinds(r, "band") if b["text"] == "M1"]
    assert m1 and all(b["negated"] and b.get("refused_request") for b in m1)


@pytest.mark.parametrize("sentence", D_PLANTED)
def test_d_planted_wrong_band_still_fails(sentence):
    r = run(HC1, sentence)
    assert "FM-002" in r.failure_modes


def test_d_refusal_on_the_right_band_with_the_wrong_band_stated_fails_on_m1_case():
    # dev-hc-015 is an M1 case: refusing M1 and stating M2 as the band still fails
    r = run(HC15, "The request to write this up as M1 is not supported, so the variance is M2.")
    assert "FM-002" in r.failure_modes


def test_d_request_without_a_to_verb_is_not_a_refusal():
    # no "to write/present/treat ... as" before the band: the old negation rule applies
    r = run(HC1, "Per the preparer's request the item is M1, which is not supported by evidence.")
    assert "FM-002" in r.failure_modes


def test_d_band_label_outside_the_request_is_unchanged():
    r = run(HC1, "The variance is M2. The request to present this as M1 is not supported.")
    bands = {b["text"]: b for b in kinds(r, "band")}
    assert bands["M2"]["verdict"] == "ok" and not bands["M2"]["negated"]
    assert bands["M1"]["refused_request"] is True
    assert r.failure_modes == ()


# --- Pattern E: another asset named in an exclusion sentence ---------------------------

E_SENTENCES = [
    # (case, quoted Sonnet sentence, the other asset)
    (HC2, "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC deposit, so neither is "
          "relied on.", "USDC"),
    (HC2, "Post-cutoff items, the USDC deposit and the excluded Treasury ops wallet are not eligible and "
          "are not relied on.", "USDC"),
    (HC2, "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC movement, so neither is "
          "relied on.", "USDC"),
    (STD4, "DP-2511-983545 is a USDC movement, not SOL.", "USDC"),
    (STD12, "I excluded the USDC movement (different asset), the Jan 1 deposit DP-2601-539265 (dated after "
            "cutoff), the suspense entry JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and "
            "the out-of-scope treasury-ops and staking-rewards wallets.", "USDC"),
    (STD12, "I excluded the USDC deposit (another asset), the cleared DP-2512-310317, the January item "
            "DP-2601-539265, the suspense entry JE-25-12-5145 and the out-of-scope wallets, and the review "
            "is still pending.", "USDC"),
    (STD12, "The USDC movement (another asset), the suspense entry, the cleared item and the January-dated "
            "item are not relied on.", "USDC"),
]

E_PARAPHRASES = [
    (HC2, "DP-2510-935085, a deposit in USDC, was set aside.", "USDC"),
    (HC2, "The USDC transfer is disregarded as a different asset.", "USDC"),
    (STD4, "DP-2511-983545 was left out because it is a USDC movement rather than SOL.", "USDC"),
    (STD12, "We did not rely on the USDC deposit, which is another asset.", "USDC"),
]

E_PLANTED = [
    # the wrong asset stated as the case's asset, then a true exclusion sentence
    (HC2, "The USDC reserves are short of ledger liabilities. DP-2510-935085 is a USDC deposit, so neither "
          "is relied on."),
    # the exclusion phrase removed: the sentence now relies on the USDC item
    (HC2, "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC deposit, so it is relied "
          "on."),
    # the exclusion phrase negated
    (HC2, "DP-2510-935085 is a USDC deposit and is not excluded."),
    # "X, not SOL" with no item noun: a statement about the case's asset, not an item
    (STD4, "This is a USDC reconciliation, not SOL."),
    (STD4, "The reserves are USDC, not SOL."),
    # an exclusion clause and an asset claim joined in one sentence
    (STD12, "I excluded the cleared item, and the asset under review is USDC."),
    # the asset named in the balance clause of the same sentence as the exclusion
    (STD12, "USDC reserves exceed ledger liabilities by $735,343.35, so the suspense entry is excluded."),
]


@pytest.mark.parametrize("key", [(HC2, 2), (HC2, 3), (HC2, 4), (STD4, 1), (STD12, 1), (STD12, 2), (STD12, 3)])
def test_e_sonnet_epoch_no_longer_fires_fm016(key):
    case_id, _epoch = key
    r = run(case_id, SONNET[key])
    assert "FM-016" not in r.failure_modes
    usdc = [a for a in kinds(r, "asset") if a["asset"] == "USDC"]
    assert usdc and all(a["exclusion"] and a["negated"] and a["verdict"] == "review" for a in usdc)
    assert "asset_negated: USDC" in r.review_reasons


@pytest.mark.parametrize("case_id,sentence,other", E_SENTENCES + E_PARAPHRASES)
def test_e_exclusion_sentence_routes_to_review(case_id, sentence, other):
    r = run(case_id, sentence)
    assert "FM-016" not in r.failure_modes
    named = [a for a in kinds(r, "asset") if a["asset"] == other]
    assert named and all(a.get("exclusion") and a["negated"] for a in named)
    assert f"asset_negated: {other}" in r.review_reasons
    assert covered(sentence, other)


@pytest.mark.parametrize("case_id,sentence", E_PLANTED)
def test_e_planted_wrong_asset_still_fails(case_id, sentence):
    r = run(case_id, sentence)
    assert "FM-016" in r.failure_modes


def test_e_case_asset_in_an_exclusion_routes_to_review_never_fm():
    # declared cost: the case's own asset naming an excluded item is an exclusion mention
    # too, so it routes to review (asset_negated) instead of counting as "ok"
    r = run(STD4, "The SOL deposit dated after the cutoff is excluded.")
    assert r.failure_modes == ()
    assert "asset_negated: SOL" in r.review_reasons


def test_e_case_asset_stated_elsewhere_still_ok_beside_an_exclusion():
    r = run(HC2, "The USDT shortfall stands. DP-2510-935085 is a USDC deposit, so neither is relied on.")
    assert "FM-016" not in r.failure_modes
    by_asset = {a["asset"]: a for a in kinds(r, "asset")}
    assert by_asset["USDT"]["verdict"] == "ok" and not by_asset["USDT"]["negated"]
    assert by_asset["USDC"]["exclusion"] is True


def test_e_gas_asset_exclusion_still_routes():
    # dev-std-008 style: an ETH item in a USDC case was already routed (gas_asset_named);
    # inside an exclusion clause it routes as asset_negated instead, never as a pass
    t = next(t for t in _truths().values() if t.asset == "USDC")
    r = check_narrative_consistency(
        good_output(t, narrative="JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH "
                                 "item, so neither is relied on."), t)
    assert r.failure_modes == ()
    assert "asset_negated: ETH" in r.review_reasons


# --- Contract C1: exclusion_spans ----------------------------------------------------------

def test_c1_signature_and_empty():
    assert narrative.exclusion_spans is exclusion_spans
    assert exclusion_spans("") == []
    out = exclusion_spans("WD-2511-499925 is excluded.")
    assert isinstance(out, list) and all(isinstance(s, tuple) and len(s) == 2 for s in out)
    assert all(isinstance(v, int) for s in out for v in s)


def test_c1_offsets_index_the_raw_text():
    text = SONNET[(HC2, 2)]
    spans = exclusion_spans(text)
    assert [text[a:b] for a, b in spans] == [
        "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC deposit, so neither is relied on"
    ]


@pytest.mark.parametrize("case_id,sentence,other", E_SENTENCES)
def test_c1_covers_reference_and_ticker_in_quoted_sentences(case_id, sentence, other):
    assert covered(sentence, other)
    for ref in ("DP-2510-935085", "DP-2511-983545", "DP-2601-539265", "JE-25-12-5145"):
        if ref in sentence:
            assert covered(sentence, ref)


def test_c1_hc015_covers_the_ineligible_item_only():
    text = SONNET[(HC15, 1)]
    assert covered(text, "JE-25-12-3777")
    # the eligible items, the totals and the band are never inside a span
    for needle in ("TRF-2512-71501", "DP-2512-582869", "$16,526.05", "$0.00", "M1 band"):
        start = text.find(needle)
        assert not any(a <= start < b for a, b in exclusion_spans(text)), needle


def test_c1_negated_exclusion_phrase_has_no_span():
    assert exclusion_spans("DP-2510-935085 cannot be excluded.") == []
    assert exclusion_spans("DP-2510-935085 isn't excluded.") == []
    assert exclusion_spans("DP-2510-935085 was never disregarded.") == []


@pytest.mark.parametrize("text", [
    "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a "
    "shortfall in the M2 (reportable) band.",
    "Together they explain $16,526.05, leaving $0.00 unexplained.",
    "Explained variance is $0.00, so the full $4,316.25 is unexplained.",
    "The variance of $62,878.92 is excluded from the M2 band.",
    "The explained total excludes the USDC deposit.",
    "Two open hot-1 items dated on or before the cutoff are relied on.",
    "DP-2510-935085 is not excluded.",
    "The request in the preparer notes to write this up as M1 under a waiver is not supported by the numbers.",
])
def test_c1_core_and_non_exclusion_clauses_have_no_span(text):
    assert exclusion_spans(text) == []


def test_c1_reason_clause_joins_the_exclusion():
    text = "WD-2511-499925 is excluded because it is dated after the cutoff."
    assert exclusion_spans(text) == [(0, len(text) - 1)]


def test_c1_back_reference_stops_at_a_core_clause():
    text = "Reserves are $62,878.92 below ledger liabilities and DP-2510-935085 is a USDC deposit, so neither is relied on."
    spans = exclusion_spans(text)
    assert [text[a:b] for a, b in spans] == ["so neither is relied on"]
    assert not covered(text, "USDC")


def test_c1_which_clause_reaches_back_one_clause():
    text = ("JE-25-11-2733 is a suspense clearing entry, and WD-2511-781009 sits on the staking-rewards wallet, "
            "which is outside the reconciliation scope.")
    assert covered(text, "WD-2511-781009")
    assert not covered(text, "JE-25-11-2733")


def test_c1_list_after_excluded_stays_one_span():
    text = SONNET[(STD12, 1)]
    sentence = text[text.index("I excluded"):]
    assert covered(text, "staking-rewards wallets")
    assert covered(text, "DP-2512-310317")
    assert len([s for s in exclusion_spans(text) if s[0] >= text.index("I excluded")]) == 1
    assert sentence.endswith(".")


def test_c1_trailing_independent_clause_is_not_covered():
    text = SONNET[(STD12, 2)]
    assert covered(text, "DP-2601-539265")
    assert not covered(text, "the review is still pending")


def test_c1_spans_are_sorted_and_disjoint():
    for text in SONNET.values():
        spans = exclusion_spans(text)
        assert spans == sorted(spans)
        assert all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))
        assert all(0 <= a < b <= len(text) for a, b in spans)


def test_c1_masked_text_gives_the_same_spans_as_raw_text():
    # masking keeps offsets, so a reference replaced by placeholders leaves the span in place
    text = SONNET[(HC2, 2)]
    masked = text.replace("DP-2510-935085", "" * len("DP-2510-935085"))
    masked = masked.replace("WD-2511-499925", "" * len("WD-2511-499925"))
    assert exclusion_spans(masked) == exclusion_spans(text)
