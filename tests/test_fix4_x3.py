"""Fix round 4, fixer X3 (2026-11-09): the FM-015 reliance family routes instead of failing.

The Fable 5.2 review (review-fable52-2026-11-09/run/REVIEW.md section 2.1) found five
constructions where a correct narrative names an ineligible item in a sentence whose
explain cue is affirmed only by the reading of scoring._cue_polarity, so FM-015 fired:

  FC20  'matched and resolved on both sides', said of an item cleared on both sides
  FC21  the cue inside a preparer's or controller's claim the sentence declines
  FC19  negation right after the cue ('explains none of the surplus')
  FC18  'did not rely on' or 'leaving aside' after the cue, naming the ineligible item
  FK02  negation more than three words before the cue (KC16): 'No open item is
        eligible to explain it:', 'No open item qualifies as a reconciling item:',
        'cannot be used to explain', 'cannot be relied on as a reconciling item',
        'None of ..., A, B and C, explains any of'

Design rule: each guard turns a 'relied' verdict into 'unclear' (the existing reliance
route, 'narrative names ineligible item REF without stating whether it explains the
variance'), never into 'excluded'. So a guard can only move FM to ROUTE. An exception
word anywhere in the sentence ('except', 'other than', 'but', 'only', ...) keeps every
guard off.

Rows: the five review leads, two or more right-text paraphrases per construction (these
route), and wrong twins (same wording, one fact wrong) that keep FM or ROUTE, never
CLEAN or ZERO. Each row records its status and codes at a783322 (before) and on this
branch (after). Built the way tests/test_fix3_oracle.py builds a minimal answer.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from icfr_agent_evals import scoring
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.scoring import check_correctness, check_fabrication, check_reconciling_items

RELIANCE_ROUTE = "without stating whether it explains the variance"


@lru_cache(maxsize=1)
def _truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


_ADDRESS_LIKE = re.compile(r"^(?:0x[0-9a-fA-F]+|[0-9A-Za-z]{30,})$")


def _answer(case_id: str, text: str) -> dict:
    t = _truths()[case_id]
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
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type}
                              for i in t.eligible_items()],
        "narrative": text,
    }


def _status(case_id: str, text: str):
    t = _truths()[case_id]
    parsed = _answer(case_id, text)
    results = [check_correctness(parsed, t), check_reconciling_items(parsed, t), check_fabrication(parsed, t)]
    fms = sorted({c for r in results for c in r.failure_modes})
    reasons = [x for r in results for x in r.review_reasons]
    if fms:
        status = "FM"
    elif reasons:
        status = "ROUTE"
    elif min(r.points for r in results) < 3:
        status = "ZERO"
    else:
        status = "CLEAN"
    return status, fms, reasons, results[1]


# (id, case, text, status before, codes before, status after, codes after)
ROWS = [
    ('FC20:lead:L3-06a', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) is the only eligible item, leaving $0.00 unexplained. DP-2510-831633 was matched and resolved on both sides before the cutoff, so it has no effect on the variance.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC21:lead:L3-07a', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, and the extracts do not support the preparer's view that withdrawal batch timing explains the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.",
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC19:lead:L3-05a', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and explains none of the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC18:lead:L3-02a', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the surplus in full, and I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633 or the post-cutoff item WD-2511-221933.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:lead:L3-01a', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the Staking payouts wallet, and DP-2511-340361 cleared on both sides. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC20:right:P1', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the full surplus, leaving $0.00 unexplained. DP-2510-831633 was reconciled on both sides on 30-Oct-2025, before the cutoff, so it does not affect the variance.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC20:right:P2', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items explain the full amount: DP-2512-574449 ($66,296.00), a deposit credited in the ledger before the funds arrived on chain, and OI-251231-441 ($15.24), a sweep network fee on chain but not booked in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained. DP-2512-039821 was matched and resolved on both sides before the cutoff and has no effect on the shortfall.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC20:right:P3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving $70,350.50 unexplained. WD-2511-928269 resolved on both sides before the cutoff, so it plays no part in the variance.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC20:wrong:W1', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the full surplus, leaving $0.00 unexplained. JE-25-10-9025 was matched and resolved on both sides before the cutoff, so it has no effect on the variance.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC20:wrong:W2', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the full surplus, leaving $0.00 unexplained. DP-2510-831633 was matched and resolved on both sides before the cutoff, and it explains part of the surplus.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC20:wrong:W3', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) is the only eligible item, leaving $10,000.00 unexplained. DP-2510-831633 was matched and resolved on both sides before the cutoff, so it has no effect on the variance.',
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001']),
    ('FC21:right:P1', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, and the extracts do not support the controller's claim that a deposit timing difference explains the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.",
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC21:right:P2', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,718.39 unexplained. WD-2511-499925 and DP-2510-935085 are not usable, and the records do not support the controller's view that timing explains the whole shortfall.",
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001']),
    ('FC21:right:P3', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and DP-2511-680653 sits in the Staking payouts wallet, and the schedule does not support the preparer's assertion that withdrawal timing explains the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.",
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC21:wrong:W1', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, and the extracts support the preparer's view that withdrawal batch timing explains the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.",
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC21:wrong:W2', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, and the extracts do not support the preparer's view that withdrawal batch timing explains the surplus. Explained variance is USD 18,738.48 and USD 0.00 remains unexplained.",
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC21:wrong:W3', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 explains the surplus, and the extracts do not support the preparer's view that withdrawal batch timing explains it. Explained variance is USD 8,555.30 and USD 10,183.18 remains unexplained.",
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001', 'FM-015']),
    ('FC19:right:P1', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. The manual adjustment JE-25-11-7405 accounts for none of the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC19:right:P2', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the full surplus, leaving $0.00 unexplained. The manual adjustment JE-25-10-9025 explains none of it.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC19:right:P3', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items explain the full amount: DP-2512-574449 ($66,296.00), a deposit credited in the ledger before the funds arrived on chain, and OI-251231-441 ($15.24), a sweep network fee on chain but not booked in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained. WD-2601-039329 accounts for none of the shortfall.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC19:wrong:W1', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and explains all of the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001', 'FM-015']),
    ('FC19:wrong:W2', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and explains none of the surplus. Explained variance is USD 8,555.30 and USD 10,183.18 remains unexplained.',
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001']),
    ('FC19:wrong:W3', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and explains nothing less than the full surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC18:right:P1', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) accounts for the full surplus, leaving aside the manual adjustment JE-25-10-9025, the cleared deposit DP-2510-831633 and the post-cutoff withdrawal WD-2511-221933.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC18:right:P2', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. DP-2512-574449 ($66,296.00) and OI-251231-441 ($15.24) explain the full shortfall, leaving $0.00 unexplained, and I did not rely on DP-2512-039821 or WD-2601-039329.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC18:right:P3', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) explains part of the shortfall and $18,718.39 stays unexplained, and I did not rely on WD-2511-499925 or DP-2510-935085.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FC18:wrong:W1', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. The manual adjustment JE-25-10-9025 explains the surplus in full, and I did not rely on timing difference WD-2510-799057, the cleared item DP-2510-831633 or the post-cutoff item WD-2511-221933.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC18:wrong:W2', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) and JE-25-10-9025 explain the surplus in full, and I did not rely on the cleared item DP-2510-831633.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC18:wrong:W3', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the surplus in full, and I relied on the manual adjustment JE-25-10-9025 as well.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FK02:right:P1', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No item is eligible to explain the surplus: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the Staking payouts wallet, and DP-2511-340361 cleared on both sides. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:right:P2', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and cannot be relied on as a reconciling item. DP-2511-680653 is in the Staking payouts wallet and cannot be used to explain the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:right:P3', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of the three open items, JE-25-11-7405, DP-2511-680653 and DP-2511-340361, explains any of the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:wrong:W1', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. Every open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the Staking payouts wallet, and DP-2511-340361 cleared on both sides. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FK02:wrong:W2', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and can be used to explain the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FK02:wrong:W3', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the Staking payouts wallet, and DP-2511-340361 cleared on both sides. Explained variance is USD 18,738.48 and USD 0.00 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:right:P4', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the Staking payouts wallet, and DP-2511-340361 cleared on both sides. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'ROUTE', []),
    ('FK02:wrong:W5', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item is eligible to explain it except JE-25-11-7405, a manual adjustment. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FK02:wrong:W6', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item qualifies as a reconciling item: JE-25-11-7405 is the only one that does. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC18:wrong:W4', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the surplus in full, and I did not rely on anything but the manual adjustment JE-25-10-9025.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC19:wrong:W4', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. JE-25-11-7405 is a manual adjustment and explains none of the surplus except its own $8,555.30. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-001', 'FM-015'], 'FM', ['FM-001', 'FM-015']),
    ('FC20:wrong:W4', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. Timing difference WD-2510-799057 ($34,349.43) explains the full surplus, leaving $0.00 unexplained. DP-2510-831633 was matched and resolved on both sides, but it still accounts for part of the surplus.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FC21:wrong:W4', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, and the extracts do not support the preparer's view that anything other than JE-25-11-7405 explains the surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.",
     'FM', ['FM-015'], 'FM', ['FM-015']),
    ('FK02:wrong:W4', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of the other items matter; JE-25-11-7405 explains any remaining surplus. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained.',
     'FM', ['FM-015'], 'FM', ['FM-015']),
]


@pytest.mark.parametrize("rid,case,text,sb,cb,sx,cx", ROWS, ids=[r[0] for r in ROWS])
def test_x3_row_status(rid, case, text, sb, cb, sx, cx):
    status, fms, reasons, rec = _status(case, text)
    assert (status, fms) == (sx, cx)


@pytest.mark.parametrize("rid,case,text,sb,cb,sx,cx", ROWS, ids=[r[0] for r in ROWS])
def test_x3_change_is_fm_to_route_only(rid, case, text, sb, cb, sx, cx):
    """The design rule on every row: no change, FM to ROUTE, or FM keeping FM."""
    assert (sb, sx) in {("FM", "FM"), ("FM", "ROUTE"), ("ROUTE", "ROUTE")} or sb == sx
    assert sx not in ("CLEAN", "ZERO")
    assert set(cx) <= set(cb)


@pytest.mark.parametrize("rid,case,text,sb,cb,sx,cx",
                         [r for r in ROWS if ":right:" in r[0] or ":lead:" in r[0]],
                         ids=[r[0] for r in ROWS if ":right:" in r[0] or ":lead:" in r[0]])
def test_x3_right_text_has_no_fm015_and_routes_on_reliance(rid, case, text, sb, cb, sx, cx):
    status, fms, reasons, rec = _status(case, text)
    assert "FM-015" not in fms
    assert any(RELIANCE_ROUTE in r for r in rec.review_reasons)


@pytest.mark.parametrize("rid,case,text,sb,cb,sx,cx",
                         [r for r in ROWS if ":wrong:" in r[0]],
                         ids=[r[0] for r in ROWS if ":wrong:" in r[0]])
def test_x3_wrong_twin_never_passes(rid, case, text, sb, cb, sx, cx):
    status, fms, reasons, rec = _status(case, text)
    assert status in ("FM", "ROUTE")


_GUARD_SENTENCES = [
    "DP-2510-831633 was matched and resolved on both sides before the cutoff, so it has no effect on the variance.",
    "None of JE-25-11-7405 and DP-2511-680653 is eligible, and the extracts do not support the preparer's view "
    "that withdrawal batch timing explains the surplus.",
    "JE-25-11-7405 is a manual adjustment and explains none of the surplus.",
    "Timing difference WD-2510-799057 explains the surplus in full, and I did not rely on JE-25-10-9025.",
    "No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment.",
    "No open item qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment.",
    "JE-25-11-7405 cannot be used to explain the surplus.",
    "JE-25-11-7405 cannot be relied on as a reconciling item.",
    "None of the three open items, JE-25-11-7405 and DP-2511-680653, explains any of the surplus.",
]


@pytest.mark.parametrize("sentence", _GUARD_SENTENCES)
def test_x3_guard_never_excludes(sentence):
    """A guard yields 'unclear' (review), never 'excluded' (pass)."""
    t = _truths()["dev-std-013-USDC-M2"]
    for item in t.items:
        assert scoring._reliance_verdict(sentence, item, ("not_open", "recorded_both")) in ("relied", "unclear")


@pytest.mark.parametrize("sentence", [
    "No open item qualifies as a reconciling item except JE-25-11-7405.",
    "No open item is eligible to explain it except JE-25-11-7405.",
    "Timing difference WD-2510-799057 explains the surplus, and I did not rely on anything but JE-25-10-9025.",
    "JE-25-11-7405 explains none of the surplus except its own amount.",
    "No open item qualifies as a reconciling item: JE-25-11-7405 is the only one that does.",
    "JE-25-11-7405 explains nothing less than the full surplus.",
    "JE-25-10-9025 was matched and resolved on both sides, but it still accounts for part of the surplus.",
])
def test_x3_exceptions_keep_relied(sentence):
    assert scoring._reliance_verdict(sentence) == "relied"


def test_x3_both_sides_needs_an_item_cleared_on_both_sides():
    s = "JE-25-10-9025 was matched and resolved on both sides before the cutoff."
    assert scoring._reliance_verdict(s, None, ("disallowed_type",)) == "relied"
    assert scoring._reliance_verdict(s, None, ("not_open", "recorded_both")) == "unclear"


def test_x3_not_relied_phrase_needs_every_mention_after_it():
    t = _truths()["dev-std-013-USDC-M2"]
    je = t.item_by_reference("JE-25-10-9025")
    before = "JE-25-10-9025 explains the surplus in full, and I did not rely on DP-2510-831633."
    after = "WD-2510-799057 explains the surplus in full, and I did not rely on JE-25-10-9025."
    assert scoring._reliance_verdict(before, je, ("disallowed_type",)) == "relied"
    assert scoring._reliance_verdict(after, je, ("disallowed_type",)) == "unclear"
