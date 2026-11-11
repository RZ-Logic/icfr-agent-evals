"""Fix round 5 (2026-11-10): an item's own amount, or a subtotal of relied items, read as
the explained total or the variance when the narrative also states the correct net.

The owner ruled eight dev audit rows of pilot-logs-2026-11-10 extractor errors (Opus 5.5
rows O3 to O6, Sonnet 5 rows S6 to S9 of AUDIT-ROWS-dev-2026-11-10.csv). Each such
FM-001 becomes a route on an existing settlement key, never a pass:

- P1: the figure is one cited eligible item's own amount and that item is its subject
  ("DP-1 ... accounts for $X of shortfall", "a deposit (DP-1) of $X", "offset by a $X
  timing difference (DP-1)"): item_effect_unverified.
- P2: the figure is a leg of a netting bracket ("Netting these items ($A minus $B)
  accounts for the entire $N shortfall"), the legs are cited eligible items' amounts,
  their difference is the explained total and the figure after the bracket states it:
  item_effect_unverified.
- P3: the figure is the sum of two or more, not all, cited eligible items whose amounts
  the previous clause states ("Together they add $S to the surplus"):
  item_effect_sum_unverified.
P1 and P3 also need another figure bound to the explained role that equals the explained
total from the cited items and in the truth record.

- test_real_row_routes: each audit row's own answer, as the worker wrote it: no FM-001,
  the expected key on the misread figure, and correctness has no failure mode.
- test_real_row_wrong_twin_keeps_code: the same answers with one fact wrong (the stated
  net, or the misread figure no longer an item's amount): the misread figure keeps FM-001.
- test_paraphrase_routes / test_paraphrase_wrong_twin_keeps_code: paraphrases of each
  pattern in a minimal answer for the case (structured fields equal the answer key,
  reconciling_items are the eligible items), and their wrong twins (the stated net wrong,
  the amount not an item's own, the subtotal not of relied items, the total genuinely
  misstated): the twin's misread figure keeps FM-001 and the epoch stays FM.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache

import pytest

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.narrative import check_narrative_consistency
from icfr_agent_evals.scoring import check_correctness, check_fabrication, check_reconciling_items

_ADDRESS_LIKE = re.compile(r"^(?:0x[0-9a-fA-F]+|[0-9A-Za-z]{30,})$")


@lru_cache(maxsize=1)
def _truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


def _parsed(case: str, text: str) -> dict:
    t = _truths()[case]
    cited = [i.reference for i in t.eligible_items()]
    for entity in t.valid_entities:
        if _ADDRESS_LIKE.match(entity) or entity in cited:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(entity)}(?![A-Za-z0-9])", text):
            cited.append(entity)
    return {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": cited,
        "reconciling_items": [
            {"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in t.eligible_items()
        ],
        "narrative": text,
    }


def _status(case: str, parsed: dict) -> tuple[list[str], list[str], str]:
    t = _truths()[case]
    results = [check_correctness(parsed, t), check_reconciling_items(parsed, t), check_fabrication(parsed, t)]
    fms = [c for r in results for c in r.failure_modes]
    reasons = [x for r in results for x in r.review_reasons]
    if fms:
        status = "FM"
    elif reasons:
        status = "ROUTE"
    elif min(r.points for r in results) < 3:
        status = "ZERO"
    else:
        status = "CLEAN"
    return fms, reasons, status


def _figure(case: str, parsed: dict, figure: str) -> list[dict]:
    """The narrative extractions of the amount written as `figure`."""
    res = check_narrative_consistency(parsed, _truths()[case])
    return [e for e in res.extractions if e["kind"] == "amount" and e["text"] == figure]


# The audit rows' own answers, copied from the open dev logs of pilot-logs-2026-11-10.
REAL = {'O4': {'case': 'dev-std-005-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-11',
                   'variance_amount_usd': 203938.61,
                   'variance_direction': 'shortfall',
                   'materiality_band': 'M2',
                   'entities_cited': ['DP-2511-648193',
                                      'DP-2511-893641',
                                      'DP-2511-398173',
                                      'WD-2512-191517',
                                      'Hot 1',
                                      'Hot 2'],
                   'reconciling_items': [{'reference': 'DP-2511-648193',
                                          'amount_usd': -256238.97,
                                          'type': 'timing difference (ledger-only deposit, in flight at cutoff)'},
                                         {'reference': 'DP-2511-893641',
                                          'amount_usd': 52300.36,
                                          'type': 'timing difference (chain-only deposit, posted to ledger after '
                                                  'cutoff)'}],
                   'narrative': 'On-chain reserves of $962,469,395.58 against ledger liabilities of '
                                '$962,673,334.19 give a shortfall of $203,938.61, which falls in the M2 '
                                '(reportable) band. Two deposits were open at the cutoff and offset each other. '
                                'DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but did not '
                                'reach the chain until 00:10 on 12/1, which accounts for $256,238.97 of '
                                'shortfall. DP-2511-893641 (Hot 1) arrived on chain at 23:47 on 11/30 but was '
                                'not posted to the ledger until 00:00 on 12/1, which offsets $52,300.36. The net '
                                "$203,938.61 is fully explained and $0.00 remains unexplained, so the preparer's "
                                'note that DP-2511-648193 alone accounts for that much of the shortfall should '
                                'be corrected to net it against DP-2511-893641. DP-2511-398173 was excluded '
                                'because it cleared on both sides on 11/29, and WD-2512-191517 was excluded '
                                'because it is dated after the cutoff (12/1).'}},
 'O5': {'case': 'dev-std-005-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-11',
                   'variance_amount_usd': 203938.61,
                   'variance_direction': 'shortfall',
                   'materiality_band': 'M2',
                   'entities_cited': ['DP-2511-648193',
                                      'DP-2511-893641',
                                      'Hot 1',
                                      'Hot 2',
                                      'WD-2512-191517',
                                      'DP-2511-398173'],
                   'reconciling_items': [{'reference': 'DP-2511-648193',
                                          'amount_usd': 256238.97,
                                          'type': 'in-flight settlement: ledger-only customer deposit at cutoff, '
                                                  'Hot 2; increases the shortfall'},
                                         {'reference': 'DP-2511-893641',
                                          'amount_usd': 52300.36,
                                          'type': 'timing: chain-only customer deposit at cutoff, Hot 1; offsets '
                                                  'the shortfall'}],
                   'narrative': 'On-chain reserves of USD 962,469,395.58 against ledger liabilities of USD '
                                '962,673,334.19 leave a shortfall of USD 203,938.61 (M2, reportable). Two open '
                                'cutoff items explain it. DP-2511-648193 (Hot 2) was credited in the ledger at '
                                '23:52 on 11/30 but did not arrive on chain until 00:10 on 12/1, adding USD '
                                '256,238.97 to the shortfall. DP-2511-893641 (Hot 1) arrived on chain at 23:47 '
                                'on 11/30 but was not posted to the ledger until 00:00 on 12/1, offsetting USD '
                                '52,300.36; together they explain USD 203,938.61, leaving USD 0.00 unexplained, '
                                "so the preparer's note that DP-2511-648193 alone accounts for USD 256,238.97 of "
                                'the shortfall overstates it unless netted against DP-2511-893641. '
                                'WD-2512-191517 is excluded because it is dated after cutoff, and DP-2511-398173 '
                                'is excluded because it cleared on both sides on 11/29.'}},
 'O3': {'case': 'dev-hc-013-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-10',
                   'variance_amount_usd': 25000.0,
                   'variance_direction': 'surplus',
                   'materiality_band': 'M2',
                   'entities_cited': ['Hot 1',
                                      'Vault 1',
                                      'Staking payouts',
                                      'WD-2510-571069',
                                      'WD-2510-818869',
                                      'TRF-2510-05457',
                                      'JE-25-10-1421',
                                      'WD-2510-991353'],
                   'reconciling_items': [{'reference': 'WD-2510-571069',
                                          'amount_usd': 11295.51,
                                          'type': 'timing difference'},
                                         {'reference': 'WD-2510-818869',
                                          'amount_usd': 17454.45,
                                          'type': 'timing difference'},
                                         {'reference': 'TRF-2510-05457',
                                          'amount_usd': -3749.96,
                                          'type': 'in-flight settlement'}],
                   'narrative': 'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of '
                                '$4,581,447,893.76 by $25,000.00, a surplus at the lower boundary of the M2 '
                                '(reportable) band. Two Hot 1 customer withdrawals were posted in the ledger '
                                'before cutoff but broadcast on chain only on 2025-11-01: WD-2510-571069 '
                                '($11,295.51) and WD-2510-818869 ($17,454.45). Together they add $28,749.96 to '
                                'the surplus, partly offset by the chain-only Hot 1 to Vault 1 in-flight '
                                'transfer TRF-2510-05457 (-$3,749.96). This explains the full $25,000.00 and '
                                'leaves $0.00 unexplained, subject to the reviewer confirming that '
                                'TRF-2510-05457 is a genuine unbooked outflow rather than an internal move '
                                'between two in-scope wallets, which would net to zero. The prior-period '
                                'adjustment JE-25-10-1421 and the Staking payouts item WD-2510-991353, which '
                                'sits in an out-of-scope wallet, were not relied on.'}},
 'O6': {'case': 'dev-std-005-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-11',
                   'variance_amount_usd': 203938.61,
                   'variance_direction': 'shortfall',
                   'materiality_band': 'M2',
                   'entities_cited': ['Hot 1',
                                      'Hot 2',
                                      'DP-2511-648193',
                                      'DP-2511-893641',
                                      'WD-2512-191517',
                                      'DP-2511-398173'],
                   'reconciling_items': [{'reference': 'DP-2511-648193',
                                          'amount_usd': -256238.97,
                                          'type': 'timing (deposit credited in ledger 11/30, on chain 12/01; '
                                                  'ledger only at cutoff)'},
                                         {'reference': 'DP-2511-893641',
                                          'amount_usd': 52300.36,
                                          'type': 'timing (deposit on chain 11/30, ledger posting 12/01; chain '
                                                  'only at cutoff)'}],
                   'narrative': 'On-chain reserves of USD 962,469,395.58 against ledger liabilities of USD '
                                '962,673,334.19 give a shortfall of USD 203,938.61 (M2, reportable). Two open '
                                'timing items in scope explain it. DP-2511-648193 (Hot 2) was credited in the '
                                'ledger on 11/30 but not received on chain until 12/01, which accounts for USD '
                                '256,238.97 of shortfall. That is partly offset by DP-2511-893641 (Hot 1), which '
                                'arrived on chain on 11/30 but was not posted in the ledger until 12/01, worth '
                                'USD 52,300.36 of surplus; together they explain USD 203,938.61 net, leaving USD '
                                "0.00 unexplained, so the preparer's note that DP-2511-648193 alone accounts for "
                                'its full amount of the shortfall should be corrected to net it against '
                                'DP-2511-893641. WD-2512-191517 (dated after cutoff) and DP-2511-398173 (cleared '
                                'on both sides) were not relied on.'}},
 'S8': {'case': 'dev-std-012-SOL-M3',
        'parsed': {'asset': 'SOL',
                   'period': '2025-12',
                   'variance_amount_usd': 735343.35,
                   'variance_direction': 'surplus',
                   'materiality_band': 'M3',
                   'entities_cited': ['hot-1', 'DP-2512-493757', 'DP-2512-052149'],
                   'reconciling_items': [{'reference': 'DP-2512-493757',
                                          'amount_usd': 434042.88,
                                          'type': 'pending transfer'},
                                         {'reference': 'DP-2512-052149',
                                          'amount_usd': 91877.4,
                                          'type': 'timing difference'}],
                   'narrative': 'For SOL in the 2025-12 period, on-chain reserves of $707,486,534.45 exceeded '
                                'ledger liabilities of $706,751,191.10, producing a $735,343.35 surplus '
                                'classified as M3 (material-risk). Two open items in scope for wallet hot-1 '
                                'partially explain the gap: a $434,042.88 on-chain-only pending transfer '
                                '(DP-2512-493757) that put reserves ahead of the ledger, offset by a $91,877.40 '
                                'ledger-only timing difference (DP-2512-052149) whose on-chain confirmation was '
                                'still pending at cutoff, for a net explained amount of $342,165.48. After '
                                'applying these eligible reconciling items, $393,177.87 of the surplus remains '
                                'unexplained and warrants further investigation.'}},
 'S6': {'case': 'dev-std-005-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-11',
                   'variance_amount_usd': 203938.61,
                   'variance_direction': 'shortfall',
                   'materiality_band': 'M2',
                   'entities_cited': ['Hot 1',
                                      'Hot 2',
                                      'DP-2511-893641',
                                      'DP-2511-648193',
                                      'DP-2511-398173',
                                      'RR-ETH-2511-5169'],
                   'reconciling_items': [{'reference': 'DP-2511-893641',
                                          'amount_usd': 52300.36,
                                          'type': 'timing difference'},
                                         {'reference': 'DP-2511-648193',
                                          'amount_usd': 256238.97,
                                          'type': 'timing difference'}],
                   'narrative': 'On-chain reserves of $962,469,395.58 were $203,938.61 below ledger liabilities '
                                'of $962,673,334.19 at the 11/30/2025 cutoff, an M2 shortfall. Two open items, '
                                'both dated on or before cutoff and in-scope, explain the gap: DP-2511-893641, a '
                                '$52,300.36 timing difference in which the deposit hit Hot 1 on-chain before '
                                'cutoff but was not booked in the ledger until 12/1, which offsets part of the '
                                'shortfall; and DP-2511-648193, a $256,238.97 customer deposit booked to the '
                                'ledger against Hot 2 before cutoff that had not yet confirmed on-chain, which '
                                'drives the shortfall. Netting these two timing differences ($256,238.97 minus '
                                '$52,300.36) accounts for the entire $203,938.61 shortfall, leaving no '
                                'unexplained variance. DP-2511-398173 was excluded as it cleared on both sides '
                                'on 11/29, prior to cutoff.'}},
 'S7': {'case': 'dev-std-005-ETH-M2',
        'parsed': {'asset': 'ETH',
                   'period': '2025-11',
                   'variance_amount_usd': 203938.61,
                   'variance_direction': 'shortfall',
                   'materiality_band': 'M2',
                   'entities_cited': ['ETH',
                                      'Hot 1',
                                      'Hot 2',
                                      'DP-2511-893641',
                                      'DP-2511-648193',
                                      'DP-2511-398173'],
                   'reconciling_items': [{'reference': 'DP-2511-893641',
                                          'amount_usd': 52300.36,
                                          'type': 'timing difference (chain only, Hot 1)'},
                                         {'reference': 'DP-2511-648193',
                                          'amount_usd': 256238.97,
                                          'type': 'timing difference (ledger only, Hot 2)'}],
                   'narrative': 'For ETH at the 11/30/2025 23:59:59 UTC cutoff, ledger liabilities totaled '
                                '$962,673,334.19 against in-scope on-chain reserves of $962,469,395.58, a '
                                '$203,938.61 shortfall (M2). Two open timing differences dated on or before '
                                'cutoff fully explain this gap: DP-2511-893641 ($52,300.36), a Hot 1 deposit '
                                'confirmed on-chain before cutoff but not booked to the ledger until 12/1, and '
                                'DP-2511-648193 ($256,238.97), a Hot 2 deposit credited in the ledger before '
                                'cutoff whose on-chain receipt did not post until 12/1. Netting these items '
                                '($256,238.97 ledger-only less $52,300.36 chain-only) accounts for the full '
                                '$203,938.61 shortfall, leaving $0 unexplained. DP-2511-398173 was excluded '
                                'because it cleared on both sides on 11/29, before cutoff.'}},
 'S9': {'case': 'dev-std-012-SOL-M3',
        'parsed': {'asset': 'SOL',
                   'period': '2025-12',
                   'variance_amount_usd': 735343.35,
                   'variance_direction': 'surplus',
                   'materiality_band': 'M3',
                   'entities_cited': ['SOL',
                                      'hot-1',
                                      'DP-2512-493757',
                                      'DP-2512-052149',
                                      'DP-2512-310317',
                                      'DP-2601-539265',
                                      'JE-25-12-5145'],
                   'reconciling_items': [{'reference': 'DP-2512-493757',
                                          'amount_usd': 434042.88,
                                          'type': 'pending transfer (on-chain only)'},
                                         {'reference': 'DP-2512-052149',
                                          'amount_usd': 91877.4,
                                          'type': 'timing difference (ledger only)'}],
                   'narrative': 'For SOL in the 2025-12 period (cutoff 2025-12-31 23:59:59 UTC), on-chain '
                                'reserves of $707,486,534.45 exceeded ledger liabilities of $706,751,191.10, '
                                'producing a surplus of $735,343.35, classified M3 (material-risk). Two eligible '
                                'open items at the hot-1 wallet partly explain this: a pending on-chain deposit '
                                '(DP-2512-493757) of $434,042.88 not yet posted to the ledger, offset by a '
                                'ledger-only timing difference (DP-2512-052149) of $91,877.40 not yet reflected '
                                'on-chain, together explaining a net $342,165.48 of the surplus. The remaining '
                                '$393,177.87 of the surplus is unexplained and requires further investigation; '
                                'DP-2512-310317 (cleared), DP-2601-539265 (dated after cutoff), and '
                                'JE-25-12-5145 (suspense clearing) were excluded as ineligible reconciling '
                                'items.'}}}

# row id -> (misread figure, expected key, pattern)
REAL_EXPECT = {
    "O3": ("$28,749.96", "item_effect_sum_unverified", "P3"),
    "O4": ("$256,238.97", "item_effect_unverified", "P1"),
    "O5": ("USD 256,238.97", "item_effect_unverified", "P1"),
    "O6": ("USD 256,238.97", "item_effect_unverified", "P1"),
    "S6": ("$52,300.36", "item_effect_unverified", "P2"),
    "S7": ("$52,300.36", "item_effect_unverified", "P2"),
    "S8": ("$91,877.40", "item_effect_unverified", "P1"),
    "S9": ("$434,042.88", "item_effect_unverified", "P1"),
}

# (twin id, row id, old text, new text, figure that must keep FM-001)
REAL_TWINS = [
    ("O3-net", "O3", "explains the full $25,000.00", "explains the full $26,000.00", "$28,749.96"),
    ("O3-sum", "O3", "add $28,749.96 to", "add $28,794.96 to", "$28,794.96"),
    ("O4-net", "O4", "The net $203,938.61", "The net $213,938.61", "$256,238.97"),
    ("O4-item", "O4", "accounts for $256,238.97 of", "accounts for $265,238.97 of", "$265,238.97"),
    ("O5-net", "O5", "explain USD 203,938.61", "explain USD 213,938.61", "USD 256,238.97"),
    ("O5-item", "O5", "alone accounts for USD 256,238.97", "alone accounts for USD 265,238.97", "USD 265,238.97"),
    ("O6-net", "O6", "explain USD 203,938.61 net", "explain USD 213,938.61 net", "USD 256,238.97"),
    ("O6-item", "O6", "accounts for USD 256,238.97 of", "accounts for USD 265,238.97 of", "USD 265,238.97"),
    ("S6-net", "S6", "the entire $203,938.61 shortfall", "the entire $213,938.61 shortfall", "$52,300.36"),
    ("S6-leg", "S6", "minus $52,300.36)", "minus $53,300.36)", "$53,300.36"),
    ("S7-net", "S7", "the full $203,938.61 shortfall", "the full $213,938.61 shortfall", "$52,300.36"),
    ("S7-leg", "S7", "less $52,300.36 chain-only", "less $53,200.36 chain-only", "$53,200.36"),
    ("S8-net", "S8", "amount of $342,165.48", "amount of $352,165.48", "$91,877.40"),
    ("S8-item", "S8", "a $91,877.40 ledger-only", "a $97,877.40 ledger-only", "$97,877.40"),
    ("S9-net", "S9", "a net $342,165.48", "a net $352,165.48", "$434,042.88"),
    ("S9-item", "S9", "of $434,042.88 not yet", "of $443,042.88 not yet", "$443,042.88"),
]

_S005 = ("On-chain reserves of $962,469,395.58 are $203,938.61 below ledger liabilities of $962,673,334.19, "
         "a shortfall in the M2 (reportable) band. ")
_S012 = ("On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by $735,343.35, "
         "a surplus in the M3 (material-risk) band. ")
_H013 = ("On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, "
         "a surplus in the M2 (reportable) band. ")

# (id, case, text, misread figure, expected key)
PARA = [
    ("P1-a1", "dev-std-005-ETH-M2", _S005 +
     "DP-2511-648193 was credited in the ledger before cutoff but reached the chain on 12/1, which accounts for "
     "$256,238.97 of the shortfall. DP-2511-893641 reached the chain before cutoff and was posted in the ledger on "
     "12/1. The two items explain a net $203,938.61, and $0.00 is unexplained.",
     "$256,238.97", "item_effect_unverified"),
    ("P1-a2", "dev-std-005-ETH-M2", _S005 +
     "Two open deposits explain the gap, and the net they explain is $203,938.61, with $0.00 unexplained. "
     "The preparer's claim that DP-2511-648193 by itself accounts for $256,238.97 of the shortfall ignores "
     "DP-2511-893641.",
     "$256,238.97", "item_effect_unverified"),
    ("P1-a3", "dev-std-012-SOL-M3", _S012 +
     "Two eligible items partly explain this: an on-chain deposit (DP-2512-493757) of $434,042.88 not yet in "
     "the ledger, offset by DP-2512-052149, a ledger-only timing difference, together explaining a net "
     "$342,165.48. The other $393,177.87 is unexplained.",
     "$434,042.88", "item_effect_unverified"),
    ("P1-b1", "dev-std-012-SOL-M3", _S012 +
     "Two open items partly explain the gap: an on-chain deposit (DP-2512-493757) raised reserves, offset by a "
     "$91,877.40 ledger-only timing difference (DP-2512-052149), for a net explained amount of $342,165.48. "
     "$393,177.87 remains unexplained.",
     "$91,877.40", "item_effect_unverified"),
    ("P2-1", "dev-std-005-ETH-M2", _S005 +
     "DP-2511-648193 and DP-2511-893641 were open at cutoff. Netting the two deposits ($256,238.97 less "
     "$52,300.36) accounts for the full $203,938.61 shortfall, leaving $0.00 unexplained.",
     "$52,300.36", "item_effect_unverified"),
    ("P2-2", "dev-std-005-ETH-M2", _S005 +
     "DP-2511-648193 and DP-2511-893641 were open at cutoff. Net of each other ($256,238.97 minus $52,300.36), "
     "they account for the entire $203,938.61 shortfall, so nothing is unexplained.",
     "$52,300.36", "item_effect_unverified"),
    ("P3-1", "dev-hc-013-ETH-M2", _H013 +
     "Two withdrawals were booked before cutoff and broadcast after it: WD-2510-571069 ($11,295.51) and "
     "WD-2510-818869 ($17,454.45). Combined, they add $28,749.96 to the surplus, partly offset by the "
     "in-flight transfer TRF-2510-05457 ($3,749.96). The items explain the full $25,000.00, leaving $0.00 "
     "unexplained.",
     "$28,749.96", "item_effect_sum_unverified"),
    ("P3-2", "dev-hc-013-ETH-M2", _H013 +
     "WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were posted in the ledger before cutoff and "
     "broadcast after it. Together they add $28,749.96 to the surplus. The chain-only transfer TRF-2510-05457 "
     "($3,749.96) offsets part of that, and the three items explain $25,000.00 with $0.00 unexplained.",
     "$28,749.96", "item_effect_sum_unverified"),
]

# (twin id, paraphrase id, old text, new text, figure that must keep FM-001)
PARA_TWINS = [
    # the stated net wrong
    ("P1-a1-net", "P1-a1", "a net $203,938.61", "a net $213,938.61", "$256,238.97"),
    ("P1-a2-net", "P1-a2", "explain is $203,938.61", "explain is $213,938.61", "$256,238.97"),
    ("P1-a3-net", "P1-a3", "a net\n$342,165.48", "a net\n$352,165.48", "$434,042.88"),
    ("P1-b1-net", "P1-b1", "amount of $342,165.48", "amount of $352,165.48", "$91,877.40"),
    ("P2-1-net", "P2-1", "the full $203,938.61", "the full $213,938.61", "$52,300.36"),
    ("P2-2-net", "P2-2", "the entire $203,938.61", "the entire $213,938.61", "$52,300.36"),
    ("P3-1-net", "P3-1", "the full $25,000.00", "the full $26,000.00", "$28,749.96"),
    ("P3-2-net", "P3-2", "explain $25,000.00", "explain $26,000.00", "$28,749.96"),
    # the amount not an item's own amount
    ("P1-a1-item", "P1-a1", "accounts for\n$256,238.97", "accounts for\n$265,238.97", "$265,238.97"),
    ("P1-a2-item", "P1-a2", "accounts for $256,238.97", "accounts for $265,238.97", "$265,238.97"),
    ("P1-a3-item", "P1-a3", "of $434,042.88", "of $443,042.88", "$443,042.88"),
    ("P1-b1-item", "P1-b1", "a\n$91,877.40", "a\n$97,877.40", "$97,877.40"),
    ("P2-1-leg", "P2-1", "less\n$52,300.36)", "less\n$13,935.88)", "$13,935.88"),
    ("P2-2-leg", "P2-2", "minus $52,300.36)", "minus $53,300.36)", "$53,300.36"),
    # the item named is not the item whose amount the figure is
    ("P1-a1-other", "P1-a1", "DP-2511-648193 was credited", "DP-2511-893641 was credited", "$256,238.97"),
    ("P1-b1-other", "P1-b1", "difference (DP-2512-052149)", "difference (DP-2512-493757)", "$91,877.40"),
    # the subtotal not of relied items
    ("P3-1-sub", "P3-1", "WD-2510-818869 ($17,454.45). Combined, they add $28,749.96",
     "JE-25-10-1421 ($2,914.70). Combined, they add $14,210.21", "$14,210.21"),
    ("P3-2-sub", "P3-2", "WD-2510-818869 ($17,454.45) were", "WD-2510-991353 ($17,454.45) were", "$28,749.96"),
    ("P3-1-sum", "P3-1", "add $28,749.96", "add $28,794.96", "$28,794.96"),
    # the total genuinely misstated: a group or total is the figure's subject
    ("P1-a1-total", "P1-a1", "which accounts for\n$256,238.97 of the shortfall.",
     "which together with the other accounts for\n$256,238.97 of the shortfall.", "$256,238.97"),
    ("P1-a2-total", "P1-a2", "DP-2511-648193 by itself accounts for $256,238.97",
     "DP-2511-648193 and the other item in total account for $256,238.97", "$256,238.97"),
    ("P1-a3-total", "P1-a3", "of $434,042.88 not yet in", "of, in total, $434,042.88 not yet in", "$434,042.88"),
]


def _real(row: str) -> tuple[str, dict]:
    return REAL[row]["case"], copy.deepcopy(REAL[row]["parsed"])


@pytest.mark.parametrize("row", sorted(REAL_EXPECT))
def test_real_row_routes(row):
    case, parsed = _real(row)
    figure, key, rule = REAL_EXPECT[row]
    res = check_narrative_consistency(parsed, _truths()[case])
    assert "FM-001" not in res.failure_modes, res
    assert f"{key}: {figure}" in res.review_reasons, res.review_reasons
    ex = [e for e in res.extractions if e["kind"] == "amount" and e["text"] == figure and e.get("fix5_rule")]
    assert [(e["verdict"], e["reason"], e["downgraded_from"], e["fix5_rule"]) for e in ex] == [
        ("review", key, "FM-001", rule)], ex
    fms, reasons, status = _status(case, parsed)
    assert fms == [] and status == "ROUTE", (fms, reasons)


@pytest.mark.parametrize("twin,row,old,new,figure", REAL_TWINS, ids=[t[0] for t in REAL_TWINS])
def test_real_row_wrong_twin_keeps_code(twin, row, old, new, figure):
    case, parsed = _real(row)
    assert parsed["narrative"].count(old) == 1, old
    misread = REAL_EXPECT[row][0]
    # the figure may also appear as an item's own amount: check the occurrence the rule routes
    k = [bool(e.get("fix5_rule")) for e in _figure(case, parsed, misread)].index(True)
    parsed["narrative"] = parsed["narrative"].replace(old, new)
    ex = _figure(case, parsed, figure)
    if figure == misread:
        ex = ex[k:k + 1]
    assert ex and all(e["verdict"] == "FM-001" and not e.get("fix5_rule") for e in ex), ex
    fms, reasons, status = _status(case, parsed)
    assert "FM-001" in fms and status == "FM", (fms, reasons)


@pytest.mark.parametrize("name,case,text,figure,key", PARA, ids=[p[0] for p in PARA])
def test_paraphrase_routes(name, case, text, figure, key):
    parsed = _parsed(case, text)
    res = check_narrative_consistency(parsed, _truths()[case])
    assert "FM-001" not in res.failure_modes, res
    assert f"{key}: {figure}" in res.review_reasons, res.review_reasons
    fms, reasons, status = _status(case, parsed)
    assert fms == [] and status == "ROUTE", (fms, reasons)


_PARA = {p[0]: p for p in PARA}


@pytest.mark.parametrize("twin,para,old,new,figure", PARA_TWINS, ids=[t[0] for t in PARA_TWINS])
def test_paraphrase_wrong_twin_keeps_code(twin, para, old, new, figure):
    _, case, text, _, _ = _PARA[para]
    old, new = old.replace("\n", " "), new.replace("\n", " ")
    assert text.count(old) == 1, old
    parsed = _parsed(case, text.replace(old, new))
    ex = _figure(case, parsed, figure)
    assert ex and all(e["verdict"] == "FM-001" and not e.get("fix5_rule") for e in ex), ex
    fms, reasons, status = _status(case, parsed)
    assert "FM-001" in fms and status == "FM", (fms, reasons)


def test_every_pattern_has_paraphrases_and_twins():
    rules = {REAL_EXPECT[r][2] for r in REAL_EXPECT}
    assert rules == {"P1", "P2", "P3"}
    for rule in rules:
        assert any(p[0].startswith(rule) for p in PARA), rule
        assert any(t[1].startswith(rule) for t in PARA_TWINS), rule
    assert {t[1] for t in REAL_TWINS} == set(REAL_EXPECT)
