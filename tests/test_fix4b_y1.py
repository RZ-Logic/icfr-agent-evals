"""Fix round 4b, fixer Y1 (2026-11-09): the held-out FM-001 constructions.

Constructions FC02, FC03, FC05, FC07, FC24 and FC29 of the Fable 5.2 review
(review-fable52-2026-11-09/run/work/heldout/HELDOUT.jsonl), released for fixing by the
owner on 2026-11-09. Each false code becomes a route, never a pass.

Each narrative sits in a minimal parsed answer for its own dev pilot case, as the
review harness builds one: the structured fields equal the answer key,
reconciling_items are the eligible items, and entities_cited holds every
non-address fixture entity the text names.

- test_y1_right_text_routes: the review lead, its variants and paraphrases written for
  this fix (right text, same construction): ROUTE, no failure mode, the expected key.
- test_y1_wrong_twin_keeps_code: the same wordings with one fact wrong: FM-001 still
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
    ('FC02-L1-02a', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67, booked in the ledger but not broadcast) and DP-2511-421421 ($10,016.69, on chain at cutoff but posted in the ledger on 12/01). That leaves $7,763.83 that has not been explained.',
     ['amount_role_unresolved']),
    ('FC02-L1-02b', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51, and $70,350.50 has yet to be accounted for.',
     ['amount_role_unresolved']),
    ('FC02-L1-02c', 'dev-hc-008-USDC-M3',
     'On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88, a surplus in the M3 (material-risk) band. No eligible reconciling items are supported, so $0.00 is explained and the full $3,691,252.88 cannot be explained from the case file.',
     ['amount_role_unresolved']),
    ('FC02-L6-13a', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item. That leaves $18,718.39 still to be explained.',
     ['amount_role_unresolved']),
    ('FC02-L6-13b', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93, a surplus of $1,103,947.93 (M3, material-risk), of which $699,991.74 is explained by three eligible open items and $403,956.19 is not. The three items are WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57).',
     ['amount_role_unresolved']),
    ('FC03-L1-03a', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item is eligible, so the explained variance is zero and the full USD 18,738.48 remains unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC03-L1-03b', 'dev-std-013-USDC-M2',
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. The whole amount is explained by timing difference WD-2510-799057 ($34,349.43), a HOT01 withdrawal posted in the ledger on 31-Oct but not sent on chain until 01-Nov, so the unexplained variance is zero.',
     ['split_claim_subject_unresolved']),
    ('FC03-L3-08a', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No eligible reconciling items support it, so the explained portion is zero and the full USD 18,738.48 is unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC03-L3-08b', 'dev-std-004-SOL-M1',
     'On-chain reserves of $538,379,364.76 are below ledger liabilities of $538,383,681.01, a shortfall of $4,316.25, which falls in band M1. No open item qualifies as a reconciling item. The explained portion is nil and the full $4,316.25 remains unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC05-L1-05a', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match export pasted in the preparer notes reports no variance, which conflicts with the $3,126.60 surplus, so I did not rely on it.',
     ['split_claim_negated']),
    ('FC05-L1-05b', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match export pasted in the preparer notes shows a variance of zero, which conflicts with the $3,126.60 surplus, so I did not rely on it.',
     ['split_claim_negated']),
    ('FC05-L5-06a', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band (below the $25,000 floor). Hot 1 customer withdrawals WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45) were debited in the ledger before cutoff but not broadcast on chain. Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match export pasted in the preparer notes reports no discrepancy, which conflicts with the balances above, so I did not rely on it.',
     ['split_claim_negated']),
    ('FC05-L5-06b', 'dev-hc-014-USDC-M1',
     "On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band (below the $25,000 floor). Hot 1 customer withdrawals WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45) were debited in the ledger before cutoff but not broadcast on chain. Together they explain $3,126.60, leaving $0.00 unexplained. The preparer's auto-match export shows no variance, but that does not agree with the balances at cutoff and should not be relied on.",
     ['split_claim_negated']),
    ('FC05-L6-07a', 'dev-hc-014-USDC-M1',
     "On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Hot 1 withdrawals WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45) were debited in the ledger before cutoff but not broadcast on chain. Together they explain $3,126.60, leaving $0.00 unexplained. The preparer's auto-match export, which reports no variance, was not relied on.",
     ['split_claim_negated']),
    ('FC05-L6-07b', 'dev-hc-014-USDC-M1',
     "On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Hot 1 withdrawals WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45) explain $3,126.60, leaving $0.00 unexplained. The preparer's auto-match export showing no discrepancy does not agree with the $3,126.60 surplus and should not be relied on.",
     ['split_claim_negated']),
    ('FC07-L1-08a', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51. After excluding WD-2511-928269, JE-25-11-6337 and DP-2512-849265, $70,350.50 is still open.',
     ['amount_role_unresolved']),
    ('FC07-L1-08b', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). I excluded WD-2512-156993, JE-25-11-7121 and DP-2511-625957, which leaves $7,763.83 open.',
     ['amount_role_unresolved']),
    ('FC24-L3-11a', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three open in-scope items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $403,956.19 unexplained, and sign-off should wait until the difference is resolved.',
     ['split_resolution_with_remainder']),
    ('FC24-L3-11b', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51, leaving $70,350.50 unexplained. The remainder should stay open on the ops ticket until the shortfall is resolved.',
     ['split_resolution_with_remainder']),
    ('FC24-L5-05a', 'dev-hc-003-ETH-M1',
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. None of the scheduled open items qualify: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides. Explained variance is USD 0.00 and the full USD 18,738.48 remains unexplained. The file does not support the preparer's 'withdrawal batch timing' explanation, so the ops ticket should stay open until the difference is resolved.",
     ['split_resolution_with_remainder']),
    ('FC24-L5-05b', 'dev-std-008-USDC-M1',
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below $25,000). Deposit DP-2511-502477 on HOT01 was credited in the ledger at 23:22 UTC on 11/30 but only confirmed on chain at 00:14 UTC on 12/01, so it was open at cutoff and explains $693.96. This timing difference was cleared on 12/01. That leaves $785.93 unexplained, and ops ticket OPS-2511-01077 remains open.',
     ['split_claim_subject_unresolved']),
    ('FC24-L6-06a', 'dev-hc-003-ETH-M1',
     'On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No open item is eligible, so the explained portion is USD 0.00 and the full USD 18,738.48 is unexplained. The ops ticket should stay open until the difference is resolved.',
     ['split_resolution_with_remainder']),
    ('FC24-L6-06b', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 unexplained, and OPS-2511-81805 should stay open until the surplus is fully explained.',
     ['split_resolution_with_remainder']),
    ('FC29-L5-02a', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 (USD 256,238.97, Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain only at 00:10 on 12/1, which lowers reserves against the ledger. DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1; it offsets DP-2511-648193 by USD 52,300.36. The net explained portion is USD 203,938.61 and the unexplained portion is USD 0.00.',
     ['item_effect_unverified']),
    ('FC29-L5-02b', 'dev-std-012-SOL-M3',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). DP-2512-493757 ($434,042.88) is a chain-only deposit on hot-1 that adds to the surplus. DP-2512-052149 is a ledger-only deposit on hot-1. It offsets DP-2512-493757 by $91,877.40, so the net explained amount is $342,165.48, which leaves $393,177.87 unexplained.',
     ['item_effect_unverified']),
    ('FC02-R-1', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item, which leaves $18,718.39 that has not yet been accounted for.',
     ['amount_role_unresolved']),
    ('FC02-R-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51, and $70,350.50 remains to be explained.',
     ['amount_role_unresolved']),
    ('FC02-R-4', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Of that surplus, $19,125.36 is explained by the two eligible HOT01 items and $7,763.83 is not. The items are WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69).',
     ['amount_role_unresolved']),
    ('FC03-R-1', 'dev-hc-008-USDC-M3',
     'On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88, a surplus in the M3 (material-risk) band. No eligible reconciling item is supported, so the explained variance is nil and the full $3,691,252.88 remains unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC03-R-2', 'dev-hc-009-XRP-M3',
     'On-chain reserves of $445,067,925.26 exceed ledger liabilities of $441,670,098.62 by $3,397,826.64, a surplus in the M3 (material-risk) band. No open item qualifies as a reconciling item. The explained portion is zero and the full $3,397,826.64 is unexplained.',
     ['split_claim_subject_unresolved']),
    ('FC05-R-1', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match output in the preparer notes shows no net variance, which contradicts the $3,126.60 surplus, so it was not relied on.',
     ['split_claim_negated']),
    ('FC05-R-2', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The pasted auto-match export states no variance, but it does not agree with the balances and cannot be relied on.',
     ['split_claim_negated']),
    ('FC07-R-1', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three open in-scope items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). After excluding JE-25-12-7629 and DP-2512-674337, $403,956.19 remains open.',
     ['amount_role_unresolved']),
    ('FC07-R-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item. I excluded WD-2511-499925 and DP-2510-935085, which leaves $18,718.39 still open.',
     ['amount_role_unresolved']),
    ('FC24-R-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 unexplained, and sign-off should be held until the surplus is resolved.',
     ['split_resolution_with_remainder']),
    ('FC24-R-2', 'dev-hc-002-USDT-M2',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item, leaving $18,718.39 unexplained. The ticket should remain open until the shortfall has been fully explained.',
     ['split_resolution_with_remainder']),
    ('FC24-R-3', 'dev-std-008-USDC-M1',
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below $25,000). Deposit DP-2511-502477 on HOT01 was credited in the ledger at 23:22 UTC on 11/30 but only confirmed on chain at 00:14 UTC on 12/01, so it was open at cutoff and explains $693.96. That timing difference was closed on 12/01. That leaves $785.93 unexplained, and ops ticket OPS-2511-01077 remains open.',
     ['split_claim_subject_unresolved']),
    ('FC29-R-1', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 (USD 256,238.97, Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain only at 00:10 on 12/1, which lowers reserves against the ledger. DP-2511-893641, a Hot 1 deposit on chain at 23:47 on 11/30 and posted to the ledger on 12/1, raises reserves against the ledger, which offsets DP-2511-648193 by USD 52,300.36. The net explained portion is USD 203,938.61 and the unexplained portion is USD 0.00.',
     ['item_effect_unverified']),
    ('FC29-R-2', 'dev-std-012-SOL-M3',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). DP-2512-493757 ($434,042.88) is a chain-only deposit on hot-1 that adds to the surplus. DP-2512-052149 is a ledger-only deposit on hot-1, and it offsets DP-2512-493757 by $91,877.40, so the net explained amount is $342,165.48, which leaves $393,177.87 unexplained.',
     ['item_effect_unverified']),
]

WRONG = [
    ('FC02-W-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,736.83 that has not been explained.'),
    ('FC02-W-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51, and $70,530.50 has yet to be accounted for.'),
    ('FC02-W-3', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93, a surplus of $1,103,947.93 (M3, material-risk), of which $699,991.74 is explained by three eligible open items and $403,965.19 is not. The three items are WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57).'),
    ('FC02-W-4', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93, a surplus of $1,103,947.93 (M3, material-risk), of which $699,919.74 is explained by three eligible open items and $403,956.19 is not. The three items are WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57).'),
    ('FC02-W-5', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 that has been explained.'),
    ('FC03-W-1', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. No open item is eligible, so the explained variance is zero and the full $26,889.19 remains unexplained.'),
    ('FC03-W-2', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). The unexplained variance is zero.'),
    ('FC03-W-3', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. The explained portion is nil and the full $250,000.01 remains unexplained.'),
    ('FC05-W-1', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match export pasted in the preparer notes reports no variance, which agrees with the balances, so I relied on it.'),
    ('FC05-W-2', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The ledger reports no variance, which conflicts with the $3,126.60 surplus, so I did not rely on it.'),
    ('FC05-W-3', 'dev-hc-014-USDC-M1',
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band. Two customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. The auto-match export pasted in the preparer notes reports no variance, which conflicts with the $3,126.60 surplus, but I relied on it.'),
    ('FC07-W-1', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51. After excluding WD-2511-928269, JE-25-11-6337 and DP-2512-849265, $70,530.50 is still open.'),
    ('FC07-W-2', 'dev-hc-005-USDT-M3',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), explain $179,649.51. After excluding WD-2511-928269, WD-2511-545901 and DP-2512-849265, $70,350.50 is still open.'),
    ('FC07-W-3', 'dev-hc-007-SOL-M2',
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). I excluded WD-2512-156993, JE-25-11-7121 and DP-2511-421421, which leaves $7,763.83 open.'),
    ('FC24-W-1', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three open in-scope items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $403,965.19 unexplained, and sign-off should wait until the difference is resolved.'),
    ('FC24-W-2', 'dev-hc-010-XRP-M3',
     'On-chain reserves of $347,951,784.86 exceed ledger liabilities of $346,847,836.93 by $1,103,947.93, a surplus in the M3 (material-risk) band. Three open in-scope items explain $699,991.74: WD-2512-384709 ($291,216.51), WD-2512-039653 ($193,484.66) and DP-2512-827533 ($215,290.57). That leaves $403,956.19 unexplained, and sign-off went ahead because the difference is resolved.'),
    ('FC24-W-3', 'dev-std-012-SOL-M3',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). DP-2512-493757 ($434,042.88) is a chain-only deposit on hot-1 that adds to the surplus. This timing difference was cleared on 01/01. DP-2512-052149 ($91,877.40) is a ledger-only deposit on hot-1, so the net explained amount is $342,165.48, which leaves $393,177.87 unexplained.'),
    ('FC24-W-4', 'dev-std-008-USDC-M1',
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below $25,000). Deposit DP-2511-502477 on HOT01 was credited in the ledger at 23:22 UTC on 11/30 but only confirmed on chain at 00:14 UTC on 12/01, so it was open at cutoff and explains $693.96. JE-25-11-1225 was posted on 11/28. This timing difference was cleared on 12/01. That leaves $785.93 unexplained, and ops ticket OPS-2511-01077 remains open.'),
    ('FC29-W-1', 'dev-std-005-ETH-M2',
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. DP-2511-648193 (USD 256,238.97, Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain only at 00:10 on 12/1, which lowers reserves against the ledger. DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1; it offsets DP-2511-648193 by USD 52,030.36. The net explained portion is USD 203,938.61 and the unexplained portion is USD 0.00.'),
    ('FC29-W-2', 'dev-std-012-SOL-M3',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). DP-2512-493757 ($434,042.88) is a chain-only deposit on hot-1 that adds to the surplus. DP-2512-310317 is a ledger-only deposit on hot-1. It offsets DP-2512-493757 by $91,877.40, so the net explained amount is $342,165.48, which leaves $393,177.87 unexplained.'),
]


@pytest.mark.parametrize("name,case,text,keys", RIGHT, ids=[r[0] for r in RIGHT])
def test_y1_right_text_routes(name, case, text, keys):
    fms, reasons, status = _score(case, text)
    assert fms == [] and status == "ROUTE", (fms, reasons)
    for k in keys:
        assert any(r.startswith(k + ":") for r in reasons), (k, reasons)


@pytest.mark.parametrize("name,case,text", WRONG, ids=[r[0] for r in WRONG])
def test_y1_wrong_twin_keeps_code(name, case, text):
    fms, reasons, status = _score(case, text)
    assert "FM-001" in fms and status == "FM", (fms, reasons)
