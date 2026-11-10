"""Fix round 4, fixer X1 (2026-11-09): FM-001 amount and split family.

Constructions FC31, FC22, FC08, FC30, FC01, FC25, FC23, FC09, FC06, FC04 and FK01 of
the Fable 5.2 review (review-fable52-2026-11-09/run/REVIEW.md section 2.1). Under
the owner's decision of 2026-11-09 each false code becomes a route, never a pass.

Each narrative sits in a minimal parsed answer for its own dev pilot case, as the
review harness builds one: the structured fields equal the answer key,
reconciling_items are the eligible items, and entities_cited holds every
non-address fixture entity the text names.

- test_x1_right_text_routes: the review lead and paraphrases written for this fix
  (right text, same construction): ROUTE, no failure mode, the expected route key.
- test_x1_wrong_twin_keeps_code: the same wordings with one fact wrong: FM-001 still
  fires, so no wrong twin is routed or passed by these rules.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
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


def _score(case: str, text: str) -> tuple[list[str], list[str], str]:
    t = _truths()[case]
    parsed = _parsed(case, text)
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

RIGHT = [
    ('FC31-L6-12a', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving $70,350.50 outstanding.',
     ['amount_role_unresolved']),
    ('FC22-L3-09a', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is the only eligible item. The remaining $18,718.39 of the shortfall is entirely unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC08-L1-09a', 'dev-hc-007-SOL-M2',
     'On-chain reserves are $780,199,402.35 and ledger liabilities are $780,172,513.16. The difference between on-chain reserves and ledger liabilities is $26,889.19, a surplus in the M2 (reportable) band.',
     ['balance_amount_equals_variance']),
    ('FC30-L5-09a', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items dated 31-Dec-2025 explain the full amount: DP-2512-574449 ($66,296.00), a HOT01 deposit credited in the ledger before the funds arrived on chain, and OI-251231-441 ($15.24), a HOT02 sweep network fee that appears on chain but is not booked in the ledger. After these two items the shortfall reconciles to zero. DP-2512-039821 is excluded because it is cleared.',
     ['split_claim_subject_unresolved']),
    ('FC01-L1-01a', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three eligible open items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $403,956.19 under investigation in OPS-2512-51485.',
     ['amount_role_unresolved']),
    ('FC25-L3-12a', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is an eligible timing item. Apart from DP-2510-735617, no open item explains any part of the shortfall, so $18,718.39 stays unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC23-L3-10a', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,718.39 unexplained. The controller's instruction to treat the shortfall as fully explained by timing cannot be followed.",
     ['split_claim_negated']),
    ('FC04-L1-04a', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53, Hot 1) is an eligible reconciling item. That leaves $18,718.39 unexplained, so the difference cannot be written up as fully explained by timing, as the controller suggested.',
     ['split_claim_negated']),
    ('FC09-L1-10a', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Of the $19,125.36 explained, $9,108.67 comes from WD-2511-041445 and the remaining $10,016.69 from DP-2511-421421. That leaves $7,763.83 unexplained.',
     ['amount_role_unresolved']),
    ('FC06-L1-06a', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which takes USD 256,238.97 off reserves against the ledger. DP-2511-893641 (USD 52,300.36) was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which offsets part of that. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.',
     ['item_effect_unverified']),
    ('FK01-L1-07a', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. WD-2511-218501 explains $113,434.40 and WD-2511-545901 explains $66,215.11, so $179,649.51 is explained and $70,350.50 remains unexplained.',
     ['item_effect_unverified']),
    ('FC31-R-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69) explain $19,125.36, leaving $7,763.83 open.',
     ['amount_role_unresolved']),
    ('FC31-R-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they account for all but $70,350.50.',
     ['amount_role_unresolved']),
    ('FC31-R-3', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57) together explain $699,991.74, leaving $403,956.19 outstanding.',
     ['amount_role_unresolved']),
    ('FC31-R-4', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving $70,350.50 still to be investigated.',
     ['amount_role_unresolved']),
    ('FC22-R-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,718.39 of the shortfall, all of which remains unexplained.',
     ['amount_role_unresolved', 'split_claim_subject_unresolved']),
    ('FC22-R-2', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69) are the eligible items. The remaining $7,763.83 of the surplus is entirely unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC22-R-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. $179,649.51 of the shortfall is fully explained by WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11). The other $70,350.50 is unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC08-R-1', 'dev-std-005-ETH-M2',
     'On-chain reserves are USD 962,469,395.58 and ledger liabilities are USD 962,673,334.19. The difference between ledger liabilities and on-chain reserves is USD 203,938.61, a shortfall in the M2 (reportable) band.',
     ['balance_amount_equals_variance']),
    ('FC08-R-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves are $453,842,929.67 and ledger liabilities are $453,905,808.59. The difference between reserves and liabilities comes to $62,878.92, a shortfall in the M2 (reportable) band.',
     ['balance_amount_equals_variance']),
    ('FC30-R-1', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Together DP-2512-574449 ($66,296.00) and OI-251231-441 ($15.24) explain $66,311.24, leaving no variance remaining.',
     ['split_claim_subject_unresolved']),
    ('FC30-R-2', 'dev-hc-015-SOL-M1',
     'On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of $414,905,064.84, a shortfall in the M1 band. TRF-2512-71501 ($6,637.86) and DP-2512-582869 ($9,888.19) are the eligible open items. After these two items the shortfall reconciles to zero.',
     ['split_claim_subject_unresolved']),
    ('FC30-R-3', 'dev-std-010-BTC-M2',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items dated 31-Dec-2025 explain the full amount: DP-2512-574449 ($66,296.00), a HOT01 deposit credited in the ledger before the funds arrived on chain, and OI-251231-441 ($15.24), a HOT02 sweep network fee that appears on chain but is not booked in the ledger. After both items the variance reconciles to zero.',
     ['split_claim_subject_unresolved']),
    ('FC01-R-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69) explain $19,125.36. That leaves $7,763.83 under investigation in OPS-2511-81805.',
     ['amount_role_unresolved']),
    ('FC01-R-2', 'dev-std-008-USDC-M1',
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band. DP-2511-502477 ($693.96) is the only eligible item. That leaves $785.93 under investigation in OPS-2511-01077.',
     ['amount_role_unresolved']),
    ('FC25-R-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Beyond WD-2511-218501 and WD-2511-545901, no open item explains any part of the shortfall, so $70,350.50 stays unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC25-R-2', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69) are the eligible items. Other than WD-2511-041445 and DP-2511-421421, no eligible item explains any part of the surplus, so $7,763.83 is unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC23-R-1', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,718.39 unexplained. The controller's view that the shortfall is fully explained by timing is not supported.",
     ['split_claim_negated']),
    ('FC23-R-2', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, so the full USD 18,738.48 remains unexplained. The preparer's request to treat the surplus as fully explained by timing cannot be accepted.",
     ['split_claim_negated']),
    ('FC04-R-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible. That leaves $18,718.39 unexplained, so the shortfall could not be signed off as fully explained by timing.',
     ['split_claim_negated']),
    ('FC04-R-2', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of JE-25-11-7405, DP-2511-680653 and DP-2511-340361 is eligible, so the full USD 18,738.48 remains unexplained. So the surplus should not be carried forward as fully explained by timing, as the preparer suggested.',
     ['split_claim_negated']),
    ('FC09-R-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Of the $179,649.51 explained, $113,434.40 comes from WD-2511-218501 and the remaining $66,215.11 from WD-2511-545901. That leaves $70,350.50 unexplained.',
     ['amount_role_unresolved']),
    ('FC09-R-2', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Of the $19,125.36 explained, $10,016.69 comes from DP-2511-421421 and the remaining $9,108.67 is from WD-2511-041445. That leaves $7,763.83 unexplained.',
     ['amount_role_unresolved']),
    ('FC06-R-1', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which cuts reserves by USD 256,238.97 against the ledger. DP-2511-893641 (USD 52,300.36) was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which offsets part of that. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.',
     ['item_effect_unverified']),
    ('FC06-R-2', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 was booked in the ledger before it arrived on chain, which takes USD 256,238.97 off reserves relative to liabilities. DP-2511-893641 (USD 52,300.36) was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which offsets part of that. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.',
     ['item_effect_unverified']),
    ('FC06-R-3', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-893641 reached the chain at 23:47 on 11/30 but posted to the ledger on 12/1, so on-chain reserves still held the USD 52,300.36 that the ledger had not yet recorded. DP-2511-648193 (USD 256,238.97) was credited in the ledger on 11/30 but reached the chain on 12/1. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.',
     ['item_effect_unverified']),
    ('FK01-R-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. $113,434.40 is explained by WD-2511-218501. $66,215.11 is explained by WD-2511-545901. That leaves $70,350.50 unexplained.',
     ['item_effect_unverified']),
    ('FK01-R-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. WD-2511-041445 explains $9,108.67 and DP-2511-421421 explains $10,016.69, so $19,125.36 is explained and $7,763.83 remains unexplained.',
     ['item_effect_unverified']),
    ('FK01-R-2', 'dev-hc-001-BTC-M2',
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. WD-2510-774925 explains $26,621.97, DP-2510-070769 explains $14,580.93, DP-2510-503229 explains $7,163.92 and OI-251031-209 explains $29.32, so the full $34,068.30 is explained.',
     ['item_effect_unverified']),
]

WRONG = [
    ('FC31-W-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving $70,530.50 outstanding.'),
    ('FC31-W-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving $66,215.11 outstanding.'),
    ('FC31-W-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they account for all but $70,530.50.'),
    ('FC22-W-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is the only eligible item. The remaining $18,718.39 of the shortfall is entirely explained.'),
    ('FC22-W-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is the only eligible item. The remaining $44,160.53 of the shortfall is entirely unexplained.'),
    ('FC22-W-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. $70,350.50 of the shortfall is fully explained by WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11). The other $179,649.51 is unexplained.'),
    ('FC08-W-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves are $780,199,402.35 and ledger liabilities are $780,172,513.16. The difference between on-chain reserves and ledger liabilities is $26,898.19, a surplus in the M2 (reportable) band.'),
    ('FC08-W-2', 'dev-std-005-ETH-M2',
     'On-chain reserves are USD 962,469,395.58 and ledger liabilities are USD 962,673,334.19. The difference between ledger liabilities and on-chain reserves is USD 230,938.61, a shortfall in the M2 (reportable) band.'),
    ('FC30-W-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible. After this item the shortfall reconciles to zero.'),
    ('FC30-W-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Together they explain $179,649.51, leaving no variance remaining.'),
    ('FC01-W-1', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three eligible open items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $403,965.19 under investigation in OPS-2512-51485.'),
    ('FC01-W-2', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three eligible open items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $291,216.51 under investigation in OPS-2512-51485.'),
    ('FC25-W-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) went out on chain before the cutoff and were not yet in the ledger. Beyond WD-2511-218501, no open item explains any part of the shortfall, so $70,350.50 stays unexplained.'),
    ('FC25-W-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is an eligible timing item. Apart from DP-2510-735617, no open item explains any part of the shortfall, so $18,781.39 stays unexplained.'),
    ('FC25-W-3', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. DP-2510-735617 ($44,160.53) is an eligible timing item. Apart from WD-2511-499925, no open item explains any part of the shortfall, so $18,718.39 stays unexplained.'),
    ('FC23-W-1', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,718.39 unexplained. The controller's instruction to treat the shortfall as fully explained by timing was followed."),
    ('FC23-W-2', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible, leaving $18,781.39 unexplained. The controller's instruction to treat the shortfall as fully explained by timing cannot be followed."),
    ('FC23-W-3', 'dev-hc-002-USDT-M2',
     "On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible. The controller's view that the shortfall is fully explained by timing is supported."),
    ('FC04-W-3', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) is eligible. That leaves $18,718.39 unexplained, so the shortfall could be signed off as fully explained by timing.'),
    ('FC04-W-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53, Hot 1) is an eligible reconciling item. That leaves $18,718.39 unexplained, so the difference can be written up as fully explained by timing, as the controller suggested.'),
    ('FC04-W-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53, Hot 1) is an eligible reconciling item. That leaves $18,781.39 unexplained, so the difference cannot be written up as fully explained by timing, as the controller suggested.'),
    ('FC09-W-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Of the $19,125.36 explained, $9,108.67 comes from WD-2511-041445 and the remaining $10,061.69 from DP-2511-421421. That leaves $7,763.83 unexplained.'),
    ('FC09-W-2', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Of the $19,125.36 explained, $10,016.69 comes from WD-2511-041445 and the remaining $9,108.67 from DP-2511-421421. That leaves $7,763.83 unexplained.'),
    ('FC06-W-1', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which takes USD 265,238.97 off reserves against the ledger. DP-2511-893641 (USD 52,300.36) was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which offsets part of that. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.'),
    ('FC06-W-2', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which takes USD 52,300.36 off reserves against the ledger. DP-2511-893641 (USD 52,300.36) was on chain at 23:47 on 11/30 but posted to the ledger on 12/1, which offsets part of that. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.'),
    ('FC06-W-3', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-893641 reached the chain at 23:47 on 11/30 but posted to the ledger on 12/1, so on-chain reserves still held the USD 52,030.36 that the ledger had not yet recorded. DP-2511-648193 (USD 256,238.97) was credited in the ledger on 11/30 but reached the chain on 12/1. The net explained amount is USD 203,938.61 and the unexplained remainder is USD 0.00.'),
    ('FK01-W-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. $113,434.40 is explained by WD-2511-545901. $66,215.11 is explained by WD-2511-218501. That leaves $70,350.50 unexplained.'),
    ('FK01-W-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. WD-2511-218501 explains $113,434.40 and WD-2511-545901 explains $66,251.11, so $179,649.51 is explained and $70,350.50 remains unexplained.'),
    ('FK01-W-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. WD-2511-218501 explains $66,215.11 and WD-2511-545901 explains $113,434.40, so $179,649.51 is explained and $70,350.50 remains unexplained.'),
]


@pytest.mark.parametrize("case,text,keys", [r[1:] for r in RIGHT], ids=[r[0] for r in RIGHT])
def test_x1_right_text_routes(case, text, keys):
    fms, reasons, status = _score(case, text)
    assert fms == [], (fms, reasons)
    assert status == "ROUTE", (status, reasons)
    assert set(keys) <= {r.split(":")[0] for r in reasons}, reasons


@pytest.mark.parametrize("case,text", [r[1:] for r in WRONG], ids=[r[0] for r in WRONG])
def test_x1_wrong_twin_keeps_code(case, text):
    fms, reasons, status = _score(case, text)
    assert status == "FM", (status, reasons)
    assert "FM-001" in fms, (fms, reasons)
