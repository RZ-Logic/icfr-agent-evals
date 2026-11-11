"""Fix round 4, lane X2 (2026-11-09): band, direction, asset and period false codes route.

The Fable 5.2 review of a783322 (review-fable52-2026-11-09/run/REVIEW.md section 2.1)
found false codes FC10, FC11, FC13, FC16, FC17, FC26, FC27, FC28, FK03, FK04 and FK05:
right narratives that fire FM-002, FM-003 or FM-016. Owner decision 2026-11-09: each
turns into a route, never a pass. The guards live in icfr_agent_evals/narrative.py
(_x2_band_guard, _x2_direction_guard, _x2_asset_guard, _x2_period_guard) and act only
where a statement's verdict is already an FM.

Part A: each construction's lead narrative (FINDINGS-VISIBLE.jsonl) routes with its key.
Part B: paraphrases written for this lane (right text) route.
Part C: wrong twins (same wording, one fact wrong) keep a code or a route, never CLEAN;
codes as measured on fix4/X2.
Part D: right controls that are CLEAN or ROUTE on a783322 keep the same status and reasons.

Each narrative sits in a minimal parsed answer for its dev pilot case, built the way
tests/test_fix3_oracle.py and the review harness r1h.py build one. Status words: FM = at
least one failure code; ROUTE = no code, at least one review reason; ZERO = no code, no
reason, an anchor below 3; CLEAN = otherwise.
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


def _answer(case_id: str, text: str) -> dict:
    t = _truths()[case_id]
    cited = [i.reference for i in t.eligible_items()]
    for entity in t.valid_entities:
        if _ADDRESS_LIKE.match(entity) or entity in cited:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(entity)}(?![A-Za-z0-9])", text):
            cited.append(entity)
    return {
        "asset": t.asset, "period": t.period, "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction, "materiality_band": t.materiality_band, "entities_cited": cited,
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type}
                              for i in t.eligible_items()],
        "narrative": text,
    }


def score(case_id: str, text: str) -> dict:
    t = _truths()[case_id]
    parsed = _answer(case_id, text)
    results = [check_correctness(parsed, t), check_reconciling_items(parsed, t), check_fabrication(parsed, t)]
    fms = [f for r in results for f in r.failure_modes]
    reasons = [x for r in results for x in r.review_reasons]
    if fms:
        status = "FM"
    elif reasons:
        status = "ROUTE"
    elif min(r.points for r in results) < 3:
        status = "ZERO"
    else:
        status = "CLEAN"
    return {"status": status, "fms": fms, "reasons": reasons}


def _keys(reasons) -> set[str]:
    return {str(r).split(":", 1)[0] for r in reasons}


# --- Part A: the lead narratives ---------------------------------------------------------
# (construction, lead id, case, codes on a783322, route key on fix4/X2, text)
LEADS = [
    ('FC16', 'L2-12a', 'dev-std-010-BTC-M2', ('FM-002',), 'band_word_ambiguous',
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items dated 31-Dec-2025 explain the full amount: DP-2512-574449 ($66,296.00), a deposit pre-credited in the ledger with the funds not received, and OI-251231-441 ($15.24), an immaterial sweep network fee that is on chain but not booked in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC11', 'L2-03a', 'dev-hc-011-ADA-M1', ('FM-002',), 'band_threshold_unresolved',
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall that falls in M1 (below the reportable band). Three open Hot 1 items at cutoff explain the full amount: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained.'),
    ('FC13', 'L2-06a', 'dev-hc-001-BTC-M2', ('FM-002',), 'lane_f_refused_request_object',
     'On-chain reserves of $3,098,770,400.82 are below ledger liabilities of $3,098,804,469.12, a shortfall of $34,068.30, which falls in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I did not apply the preparer-note request to record this as M1 (below floor), because the band is set by the variance amount.'),
    ('FC28', 'L4-11a', 'dev-hc-005-USDT-M3', ('FM-016',), 'asset_word_ambiguous',
     'On-chain reserves of $2,383,761,360.60 are below ledger liabilities of $2,384,011,360.61, and the true USD shortfall is $250,000.01, just over the $250,000 threshold, so the band is M3. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained.'),
    ('FC26', 'L4-04a', 'dev-hc-001-BTC-M2', ('FM-002',), 'band_negated',
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall. The request from E. Marsh to classify this as M1 was not applied, because the band is set by the variance amount.'),
    ('FC17', 'L2-14a', 'dev-hc-007-SOL-M2', ('FM-003',), 'direction_word_threshold_object',
     'On-chain reserves of $780,199,402.35, taken from the in-scope wallets with the node fully synced at cutoff, exceed ledger liabilities of $780,172,513.16 by $26,889.19, which is M2 (reportable) and well short of the M3 threshold. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67, booked in the ledger but not broadcast) and DP-2511-421421 ($10,016.69, on chain at cutoff but posted in the ledger on 12/01). That leaves $7,763.83 unexplained, and OPS-2511-81805 remains open.'),
    ('FC27', 'L4-10a', 'dev-hc-002-USDT-M2', ('FM-016',), 'period_date_ambiguous',
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item: it was ledger-posted on Oct-31 but reached the chain on Nov-01, so it was open at cutoff. That leaves $18,718.39 unexplained.'),
    ('FC10', 'L2-01a', 'dev-hc-005-USDT-M3', ('FM-002',), 'band_statement_subject_unresolved',
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained, which on its own is below the M3 threshold, and the ops ticket remains open.'),
    ('FK03', 'L4-03b', 'dev-std-012-SOL-M3', ('FM-016',), 'asset_negated',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2601-539265 (dated after the cutoff) and JE-25-12-5145 (a suspense entry). DP-2512-788225 is a USDC movement on hot-1 and is not on this sub-ledger.'),
    ('FK05', 'L4-06a', 'dev-hc-001-BTC-M2', ('FM-016',), 'period_date_ambiguous',
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall. I excluded DP-2510-829241 (cleared), WD-2511-968445 (dated 01-Nov-2025), DP-2510-784653 (Treasury ops wallet, out of scope) and JE-25-10-0933 (suspense entry).'),
    ('FK04', 'L4-01a', 'dev-std-012-SOL-M3', ('FM-016',), 'asset_negated',
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance: DP-2512-493757 adds $434,042.88 to the surplus, while DP-2512-052149 reduces it by $91,877.40. I excluded DP-2512-788225 (USDC, a different asset), DP-2601-539265 (dated after the cutoff) and JE-25-12-5145 (a suspense entry), leaving $393,177.87 unexplained.'),
]


@pytest.mark.parametrize("cons,lid,case_id,base_codes,key,text", [pytest.param(*r, id=f"{r[0]}-{r[1]}") for r in LEADS])
def test_lead_routes(cons, lid, case_id, base_codes, key, text):
    r = score(case_id, text)
    assert r["status"] == "ROUTE" and r["fms"] == [], r
    assert key in _keys(r["reasons"]), r


def test_lead_count():
    assert sorted(r[0] for r in LEADS) == sorted(
        ["FC10", "FC11", "FC13", "FC16", "FC17", "FC26", "FC27", "FC28", "FK03", "FK04", "FK05"])


# --- Parts B and C: paraphrases (right) and wrong twins -----------------------------------
# (construction, kind, id, case, status on a783322, codes on a783322, status on fix4/X2,
#  codes on fix4/X2, text)
PROBES = [
    ('FC16', 'right', 'p1', 'dev-std-010-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. DP-2512-574449 ($66,296.00) is a deposit pre-credited in the ledger before the funds arrived, and OI-251231-441 ($15.24) is an immaterial network fee recorded on chain but not yet in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC16', 'right', 'p2', 'dev-hc-013-ETH-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves exceed ledger liabilities by $25,000.00, a surplus in the M2 (reportable) band. WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger but not yet broadcast, and TRF-2510-05457 ($3,749.96) is an immaterial in-flight settlement transfer that left the hot wallet before it reached the ledger. Together they explain $25,000.00, leaving $0.00 unexplained.'),
    ('FC16', 'right', 'p3', 'dev-hc-011-ADA-M1', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall in M1 (below floor). The supporting material shows three open Hot 1 items at cutoff: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained.'),
    ('FC16', 'wrong', 'w1', 'dev-std-010-BTC-M2', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M1 band. DP-2512-574449 ($66,296.00) is a deposit pre-credited in the ledger before the funds arrived, and OI-251231-441 ($15.24) is an immaterial network fee recorded on chain but not yet in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC16', 'wrong', 'w2', 'dev-std-010-BTC-M2', 'FM', ('FM-001', 'FM-002'), 'FM', ('FM-001',),
     'On-chain reserves of $4,710,661,082.13 are $76,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. DP-2512-574449 ($66,296.00) is a deposit pre-credited in the ledger before the funds arrived, and OI-251231-441 ($15.24) is an immaterial network fee recorded on chain but not yet in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC16', 'wrong', 'w3', 'dev-std-010-BTC-M2', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, an immaterial shortfall. DP-2512-574449 ($66,296.00) and OI-251231-441 ($15.24) together explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC11', 'right', 'p1', 'dev-hc-011-ADA-M1', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall that sits below the reportable range, in M1. Three open Hot 1 items at cutoff explain the full amount: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained.'),
    ('FC11', 'right', 'p2', 'dev-std-002-BTC-M1', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves exceed ledger liabilities by $11,643.52, a surplus below M2, so the band is M1. WD-2510-164297 ($11,643.52), booked in the ledger but not yet broadcast, explains the full surplus, leaving $0.00 unexplained.'),
    ('FC11', 'right', 'p3', 'dev-hc-005-USDT-M3', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall above the reportable band, so M3 applies. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained.'),
    ('FC11', 'wrong', 'w1', 'dev-hc-011-ADA-M1', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall that falls in M2 (below the reportable band). Three open Hot 1 items at cutoff explain the full amount: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained.'),
    ('FC11', 'wrong', 'w2', 'dev-hc-005-USDT-M3', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall below the reportable band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained.'),
    ('FC13', 'right', 'p1', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I declined the treasury request to record this as M1 (immaterial), because the band is set by the variance amount.'),
    ('FC13', 'right', 'p2', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. The preparer-note request to write this up as M1 (below the floor) is not supported, because the band is set by the variance amount.'),
    ('FC13', 'right', 'p3', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I did not apply the request to classify this as M1, because the variance is above the M1 band.'),
    ('FC13', 'wrong', 'w1', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I applied the treasury request to record this as M1 (immaterial), because audit agreed the waiver.'),
    ('FC13', 'wrong', 'w2', 'dev-hc-001-BTC-M2', 'FM', ('FM-001', 'FM-002'), 'FM', ('FM-001',),
     'On-chain reserves of $3,098,770,400.82 are $43,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I declined the treasury request to record this as M1 (immaterial), because the band is set by the variance amount.'),
    ('FC28', 'right', 'p1', 'dev-hc-005-USDT-M3', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $2,383,761,360.60 sit below ledger liabilities of $2,384,011,360.61; the true USD shortfall of $250,000.01 puts the variance in M3 (material-risk). WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) left Hot 1 on chain before cutoff and reached the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained.'),
    ('FC28', 'right', 'p2', 'dev-std-013-USDC-M2', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves exceed ledger liabilities by $34,349.43, a true USD surplus in the M2 (reportable) band. WD-2510-799057 ($34,349.43) was booked in the ledger before cutoff but not yet broadcast, which explains the full surplus, leaving $0.00 unexplained.'),
    ('FC28', 'wrong', 'w1', 'dev-hc-005-USDT-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $2,383,761,360.60 sit below ledger liabilities of $2,384,011,360.61; the TrueUSD shortfall of $250,000.01 puts the variance in M3 (material-risk). WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) left Hot 1 on chain before cutoff and reached the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained.'),
    ('FC28', 'wrong', 'w2', 'dev-hc-005-USDT-M3', 'FM', ('FM-002', 'FM-016'), 'FM', ('FM-002',),
     'On-chain reserves of $2,383,761,360.60 sit below ledger liabilities of $2,384,011,360.61; the true USD shortfall of $250,000.01 puts the variance in M2 (reportable). WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) left Hot 1 on chain before cutoff and reached the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained.'),
    ('FC26', 'right', 'p1', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. The instruction from E. Marsh to record this as M1 was rejected, because the band follows the variance amount.'),
    ('FC26', 'right', 'p2', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. The push from E. Marsh to treat this as M1 was declined, since the band is set by the variance amount.'),
    ('FC26', 'right', 'p3', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'ROUTE', (),
     "On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2512-788225, flagged in D. Kowalczyk's note as a USDC deposit."),
    ('FC26', 'wrong', 'w1', 'dev-hc-001-BTC-M2', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. The request from E. Marsh to classify this as M1 was applied, because audit agreed the waiver.'),
    ('FC26', 'wrong', 'w2', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     "On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I relied on DP-2512-788225, flagged in D. Kowalczyk's note as a USDC deposit."),
    ('FC17', 'right', 'p1', 'dev-hc-007-SOL-M2', 'FM', ('FM-003',), 'ROUTE', (),
     'On-chain reserves of $780,199,402.35, measured on the in-scope wallets after the node had fully synced at the cutoff, exceed ledger liabilities of $780,172,513.16 by $26,889.19, which sits in M2 (reportable), well short of the M3 threshold. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 unexplained.'),
    ('FC17', 'right', 'p2', 'dev-std-013-USDC-M2', 'FM', ('FM-003',), 'ROUTE', (),
     'On-chain reserves, as read from every in-scope wallet listed in the workpaper at the cutoff, exceed ledger liabilities by $34,349.43, which is reportable (M2) yet falls short of the $250,000 threshold. WD-2510-799057 ($34,349.43) was booked before cutoff but not yet broadcast and explains all of it, leaving $0.00 unexplained.'),
    ('FC17', 'wrong', 'w1', 'dev-hc-007-SOL-M2', 'FM', ('FM-003',), 'FM', ('FM-003',),
     'On-chain reserves of $780,199,402.35, measured on the in-scope wallets after the node had fully synced at the cutoff, fall short of ledger liabilities of $780,172,513.16 by $26,889.19, which sits in M2 (reportable), well short of the M3 threshold. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 unexplained.'),
    ('FC17', 'wrong', 'w2', 'dev-hc-007-SOL-M2', 'FM', ('FM-002', 'FM-003'), 'FM', ('FM-002',),
     'On-chain reserves of $780,199,402.35, measured on the in-scope wallets after the node had fully synced at the cutoff, exceed ledger liabilities of $780,172,513.16 by $26,889.19, which sits in M3 (material-risk), well short of the M3 threshold. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). That leaves $7,763.83 unexplained.'),
    ('FC27', 'right', 'p1', 'dev-std-002-BTC-M1', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves exceed ledger liabilities by $11,643.52, a surplus in M1 (below floor). WD-2510-164297 ($11,643.52) was booked in the ledger on Oct-31 and broadcast on Nov-01, so it was open at cutoff and explains the full surplus, leaving $0.00 unexplained.'),
    ('FC27', 'right', 'p2', 'dev-std-010-BTC-M2', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. DP-2512-574449 ($66,296.00) was credited in the ledger on Dec-31 before the funds arrived on chain, and OI-251231-441 ($15.24) is a network fee on chain but not booked. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('FC27', 'wrong', 'w1', 'dev-hc-002-USDT-M2', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item: it was ledger-posted on Sep-30 but reached the chain on Nov-01, so it was open at cutoff. That leaves $18,718.39 unexplained.'),
    ('FC27', 'wrong', 'w2', 'dev-std-002-BTC-M1', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves exceed ledger liabilities by $11,643.52, a surplus in M1 (below floor). WD-2510-164297 ($11,643.52) was booked in the ledger on Aug-31 and broadcast on Nov-01, so it was open at cutoff and explains the full surplus, leaving $0.00 unexplained.'),
    ('FK05', 'right', 'p1', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2601-539265, dated 01-Jan-2026, after the cutoff, and JE-25-12-5145, a suspense entry.'),
    ('FK05', 'right', 'p2', 'dev-std-010-BTC-M2', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. DP-2512-574449 ($66,296.00) and OI-251231-441 ($15.24) together explain $66,311.24, leaving $0.00 unexplained. WD-2601-039329, the January 2026 withdrawal, falls after the cutoff and is not relied on.'),
    ('FK05', 'right', 'p3', 'dev-hc-005-USDT-M3', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained. DP-2512-849265 is dated December 2025, after the cutoff, so it is not relied on.'),
    ('FK05', 'wrong', 'w1', 'dev-hc-001-BTC-M2', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I excluded DP-2510-829241 (cleared), WD-2511-968445 (dated 01-Dec-2025), DP-2510-784653 (Treasury ops wallet, out of scope) and JE-25-10-0933 (suspense entry).'),
    ('FK05', 'wrong', 'w2', 'dev-hc-001-BTC-M2', 'FM', ('FM-002', 'FM-016'), 'FM', ('FM-002',),
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M1 band. Four open Hot 1 items dated on or before the cutoff explain the whole shortfall, leaving $0.00 unexplained. I excluded DP-2510-829241 (cleared), WD-2511-968445 (dated 01-Nov-2025), DP-2510-784653 (Treasury ops wallet, out of scope) and JE-25-10-0933 (suspense entry).'),
    ('FK03', 'right', 'p1', 'dev-hc-002-USDT-M2', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies, leaving $18,718.39 unexplained. DP-2510-935085 is a USDC deposit on Hot 1 and sits outside this sub-ledger.'),
    ('FK03', 'right', 'p2', 'dev-std-004-SOL-M1', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $538,379,364.76 are $4,316.25 below ledger liabilities of $538,383,681.01, a shortfall in the M1 band (below the $25,000 floor). None of the open items is eligible, so the whole $4,316.25 remains unexplained. DP-2511-983545 (USDC) cannot be relied on for this sub-ledger.'),
    ('FK03', 'right', 'p3', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained; DP-2512-788225 is a USDC deposit on hot-1 and belongs to another sub-ledger.'),
    ('FK03', 'wrong', 'w1', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2601-539265 (dated after the cutoff) and JE-25-12-5145 (a suspense entry). DP-2512-788225 is a USDT movement on hot-1 and is not on this sub-ledger.'),
    ('FK03', 'wrong', 'w2', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2601-539265 (dated after the cutoff) and JE-25-12-5145 (a suspense entry). DP-2512-493757 is a USDC movement on hot-1 and is not on this sub-ledger.'),
    ('FK04', 'right', 'p1', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible items explain $342,165.48 after excluding DP-2512-788225 (USDC) and DP-2601-539265 (after the cutoff), leaving $393,177.87 unexplained.'),
    ('FK04', 'right', 'p2', 'dev-hc-002-USDT-M2', 'FM', ('FM-016',), 'ROUTE', (),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies after excluding DP-2510-935085 (USDC), leaving $18,718.39 unexplained.'),
    ('FK04', 'wrong', 'w1', 'dev-std-012-SOL-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible items explain $342,165.48 after excluding DP-2512-788225 (USDT) and DP-2601-539265 (after the cutoff), leaving $393,177.87 unexplained.'),
    ('FK04', 'wrong', 'w2', 'dev-std-012-SOL-M3', 'FM', ('FM-001', 'FM-016'), 'FM', ('FM-001',),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible items explain $342,165.48 after excluding DP-2512-788225 (USDC) and DP-2601-539265 (after the cutoff), leaving $493,177.87 unexplained.'),
    ('FC10', 'right', 'p1', 'dev-hc-005-USDT-M3', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. The unexplained remainder of $70,350.50 is below the M3 threshold by itself, and the ops ticket remains open.'),
    ('FC10', 'right', 'p2', 'dev-hc-002-USDT-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item, leaving $18,718.39 unexplained, which on its own is below the reportable floor.'),
    ('FC10', 'right', 'p3', 'dev-hc-007-SOL-M2', 'FM', ('FM-002',), 'ROUTE', (),
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67) and DP-2511-421421 ($10,016.69). $7,763.83 remains unexplained, which is itself below the M2 threshold.'),
    ('FC10', 'wrong', 'w1', 'dev-hc-005-USDT-M3', 'FM', ('FM-001', 'FM-002'), 'FM', ('FM-001',),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $80,350.50 unexplained, which on its own is below the M3 threshold, and the ops ticket remains open.'),
    ('FC10', 'wrong', 'w2', 'dev-hc-005-USDT-M3', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $2,383,761,360.60 are below ledger liabilities of $2,384,011,360.61, a shortfall of $250,000.01, which on its own is below the M3 threshold. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained.'),
    ('FC10', 'wrong', 'w3', 'dev-std-012-SOL-M3', 'FM', ('FM-002',), 'FM', ('FM-002',),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained, which on its own is below the M3 threshold. '),
    ('FK03', 'wrong', 'w3', 'dev-hc-002-USDT-M2', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies, leaving $18,718.39 unexplained. DP-2510-935085 is a USDC deposit on Hot 1, so it is relied on.'),
    ('FK03', 'wrong', 'w4', 'dev-std-004-SOL-M1', 'FM', ('FM-001', 'FM-016', 'FM-015'), 'FM', ('FM-001', 'FM-016', 'FM-015'),
     'On-chain reserves of $538,379,364.76 are $4,316.25 below ledger liabilities of $538,383,681.01, a shortfall in the M1 band (below the $25,000 floor). DP-2511-983545 is a USDC movement, not SOL, and it accounts for $859.01 of the shortfall.'),
    ('FC27', 'wrong', 'w3', 'dev-hc-005-USDT-M3', 'FM', ('FM-016',), 'FM', ('FM-016',),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained, and the Dec-25 shortfall was reviewed.'),
]


@pytest.mark.parametrize("cons,kind,pid,case_id,base_status,base_codes,status,codes,text", [
    pytest.param(*r, id=f"{r[0]}-{r[2]}") for r in PROBES if r[1] == "right"])
def test_right_text_routes(cons, kind, pid, case_id, base_status, base_codes, status, codes, text):
    r = score(case_id, text)
    assert r["status"] == "ROUTE" and r["fms"] == [], r


@pytest.mark.parametrize("cons,kind,pid,case_id,base_status,base_codes,status,codes,text", [
    pytest.param(*r, id=f"{r[0]}-{r[2]}") for r in PROBES if r[1] == "wrong"])
def test_wrong_twin_is_never_clean(cons, kind, pid, case_id, base_status, base_codes, status, codes, text):
    r = score(case_id, text)
    assert r["status"] in ("FM", "ROUTE"), r
    assert r["status"] == status and tuple(r["fms"]) == codes, r


def test_every_construction_has_two_paraphrases_and_a_wrong_twin():
    for cons in {r[0] for r in LEADS}:
        assert sum(1 for r in PROBES if r[0] == cons and r[1] == "right") >= 2, cons
        assert sum(1 for r in PROBES if r[0] == cons and r[1] == "wrong") >= 1, cons


def test_every_change_is_fm_to_route():
    for r in PROBES:
        if (r[4], r[5]) != (r[6], r[7]):
            assert r[4] == "FM" and (r[6] == "ROUTE" or set(r[7]) < set(r[5])), r[:8]


# --- Part D: controls keep their a783322 status and reasons -------------------------------
# (id, case, status, codes, reasons, text)
CONTROLS = [
    ('CTL:right:c1', 'dev-hc-011-ADA-M1', 'CLEAN', (), (),
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall in M1 (below floor). Three open Hot 1 items at cutoff explain the full amount: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained.'),
    ('CTL:right:c2', 'dev-hc-002-USDT-M2', 'CLEAN', (), (),
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53) qualifies as a reconciling item: it was ledger-posted on 31 October 2025 but reached the chain on 1 November 2025, so it was open at cutoff. That leaves $18,718.39 unexplained.'),
    ('CTL:right:c3', 'dev-std-012-SOL-M3', 'ROUTE', (), ('aggregate_total_role_cue_follows: $342,165.48', 'lane_f_exclusion_context: USDC'),
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained. I excluded DP-2512-788225 (USDC, a different asset), DP-2601-539265 (dated after the cutoff) and JE-25-12-5145 (a suspense entry).'),
    ('CTL:right:c4', 'dev-hc-005-USDT-M3', 'CLEAN', (), (),
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall above the $250,000 threshold, so the band is M3 (material-risk). Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51. That leaves $70,350.50 unexplained.'),
    ('CTL:right:c5', 'dev-std-002-BTC-M1', 'CLEAN', (), (),
     'On-chain reserves exceed ledger liabilities by $11,643.52, a surplus in M1 (below floor). WD-2510-164297 ($11,643.52) was booked in the ledger before cutoff and broadcast after it, so it explains the full surplus, leaving $0.00 unexplained.'),
]


@pytest.mark.parametrize("cid,case_id,status,codes,reasons,text", [pytest.param(*r, id=r[0]) for r in CONTROLS])
def test_control_unchanged(cid, case_id, status, codes, reasons, text):
    r = score(case_id, text)
    assert (r["status"], tuple(r["fms"]), tuple(r["reasons"])) == (status, codes, reasons), r
