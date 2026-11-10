"""Fix round 4b, lane Y2 (2026-11-09): held-out FM-002 false codes FC12, FC14, FC15 route.

The Fable 5.2 review of a783322 sealed nine false-code constructions in
review-fable52-2026-11-09/run/work/heldout/HELDOUT.jsonl. Owner decision 2026-11-09:
they are released for fixing, and each turns into a route, never a pass. This lane takes
the three FM-002 ones:

- FC12: a neighbouring band named on a boundary case ("sits exactly on the M1/M2
  boundary", "the rounded figure on the summary tab would read as M2").
- FC14: "too small to be reportable" or "not large enough to be reportable" on an M1 case.
- FC15: a below-type comparator right before the variance figure on a boundary case
  ("well under the $250,000.01 shortfall").

The guard is icfr_agent_evals/narrative.py _y2_band_guard. It runs only where a band
statement's verdict is already FM-002 and no X2 guard took it, and each trigger is tied
to a case fact (variance on the boundary, rounded variance in the named band, case band
below the word's bands, compared figure equal to the variance).

Part A: the lead and every variant of the three constructions route with their key.
Part B: paraphrases written for this lane (right text) route.
Part C: wrong twins (same wording, one fact wrong) keep FM-002; codes as measured.
Each narrative sits in a minimal parsed answer for its dev pilot case, built the way
tests/test_fix4_x2.py and the review harness r1h.py build one. Status words: FM = at
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
    fms = sorted({f for r in results for f in r.failure_modes})
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


# --- Part A: lead and variants ---------------------------------------------------------
# (construction, row id, lead or variant, case, codes at c52703b, route key on fix4b/Y2, text)
HELDOUT = [
    ("FC12", "L2-04a", "lead", "dev-hc-005-USDT-M3", ('FM-002',), "band_statement_subject_unresolved",
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that is just over the M3 threshold and so M3; the rounded figure on the summary tab would read as M2, but the band follows the exact amount. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "L2-04b", "variant", "dev-hc-013-ETH-M2", ('FM-002',), "band_threshold_unresolved",
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M1/M2 boundary and is M2 because the M2 floor is inclusive. The eligible open Hot 1 items explain $25,000.00 net, leaving $0.00 unexplained."),
    ("FC12", "L2-04c", "variant", "dev-hc-005-USDT-M3", ('FM-002',), "band_statement_subject_unresolved",
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band; the preparer's rounding on the summary tab does not bring the shortfall down to M2. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "L6-04a", "variant", "dev-hc-013-ETH-M2", ('FM-002',), "band_threshold_unresolved",
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M1/M2 boundary and so falls in M2 (reportable). The eligible Hot 1 items explain $25,000.00, leaving $0.00 unexplained."),
    ("FC12", "L6-04b", "variant", "dev-hc-013-ETH-M2", ('FM-002',), "band_threshold_unresolved",
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the boundary between M1 and M2, which the bands place in M2 (reportable). The eligible Hot 1 items explain $25,000.00, so nothing is left unexplained."),
    ("FC12", "L6-05a", "variant", "dev-hc-005-USDT-M3", ('FM-002',), "band_statement_subject_unresolved",
     "On-chain reserves of $2,383,761,360.60 are below ledger liabilities of $2,384,011,360.61, a shortfall of $250,000.01, which is M3 (material-risk). The rounded figure on the summary tab would fall in M2, but the exact shortfall does not."),
    ("FC12", "L6-05b", "variant", "dev-hc-005-USDT-M3", ('FM-002',), "band_statement_subject_unresolved",
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band; rounding it as the summary tab does would place it in M2, so the exact figure is used. Hot 1 withdrawals WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11) explain $179,649.51, leaving $70,350.50 unexplained."),
    ("FC14", "L2-09a", "lead", "dev-hc-003-ETH-M1", ('FM-002',), "band_negated",
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No eligible reconciling items support it, so the explained portion is USD 0.00 and the full USD 18,738.48 is unexplained. The difference should stay open pending the ops ticket and the full wallet export, even though it is too small to be reportable."),
    ("FC14", "L2-09b", "variant", "dev-hc-015-SOL-M1", ('FM-002',), "band_negated",
     "On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of $414,905,064.84, a shortfall in band M1 that is not large enough to be reportable. Two open items explain $16,526.05 in total: TRF-2512-71501 ($6,637.86) and DP-2512-582869 ($9,888.19), so the unexplained remainder is $0.00."),
    ("FC15", "L2-11a", "lead", "dev-hc-005-USDT-M3", ('FM-002',), "band_threshold_unresolved",
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, well under the $250,000.01 shortfall, and leave $70,350.50 unexplained."),
    ("FC15", "L2-11b", "variant", "dev-hc-013-ETH-M2", ('FM-002',), "band_threshold_unresolved",
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M2 floor. The eligible open Hot 1 items explain $25,000.00 net, which is within the $25,000.00 surplus, leaving $0.00 unexplained."),
]


@pytest.mark.parametrize("cons,rid,role,case_id,base_codes,key,text",
                         [pytest.param(*r, id=f"{r[0]}-{r[1]}") for r in HELDOUT])
def test_heldout_row_routes(cons, rid, role, case_id, base_codes, key, text):
    assert base_codes == ("FM-002",)
    r = score(case_id, text)
    assert r["status"] == "ROUTE" and r["fms"] == [], r
    assert key in _keys(r["reasons"]), r


def test_heldout_rows_cover_the_three_constructions():
    by = {}
    for r in HELDOUT:
        by.setdefault(r[0], []).append(r[1])
    assert by == {"FC12": ["L2-04a", "L2-04b", "L2-04c", "L6-04a", "L6-04b", "L6-05a", "L6-05b"],
                  "FC14": ["L2-09a", "L2-09b"], "FC15": ["L2-11a", "L2-11b"]}


# --- Parts B and C: paraphrases (right) and wrong twins ---------------------------------
# (construction, kind, id, case, status at c52703b, codes at c52703b, status on fix4b/Y2,
#  codes on fix4b/Y2, text)
PROBES = [
    ("FC12", "right", "p1", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that sits right on the M2/M3 boundary, just over it and so in M3 (material-risk). Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "right", "p2", "dev-hc-013-ETH-M2", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus at the line between M1 and M2 that the inclusive M2 floor places in M2 (reportable). The eligible open Hot 1 items explain $25,000.00 net, leaving $0.00 unexplained."),
    ("FC12", "right", "p3", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in M3. The summary tab's rounded difference of $250,000.00 might suggest M2, but the band rests on the exact figure. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "right", "p4", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Read from the rounded summary tab the shortfall could look like M2, yet the exact amount governs. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "wrong", "w1", "dev-hc-013-ETH-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M1/M2 boundary and is M1 because the M2 floor is exclusive. The eligible open Hot 1 items explain $25,000.00 net, leaving $0.00 unexplained."),
    ("FC12", "wrong", "w2", "dev-hc-007-SOL-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus that sits exactly on the M1/M2 boundary and so falls in M2 (reportable). Two eligible HOT01 items explain $19,125.36, leaving $7,763.83 unexplained."),
    ("FC12", "wrong", "w3", "dev-hc-013-ETH-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus just below the M1/M2 boundary. The eligible open Hot 1 items explain $25,000.00 net, leaving $0.00 unexplained."),
    ("FC12", "wrong", "w4", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that sits exactly on the M1/M2 boundary. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "wrong", "w5", "dev-hc-013-ETH-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus in M2; the rounded figure on the summary tab would read as M1, but the band follows the exact amount. The eligible open Hot 1 items explain $25,000.00 net, leaving $0.00 unexplained."),
    ("FC12", "wrong", "w6", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that, after rounding, is M2. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC12", "wrong", "w7", "dev-std-012-SOL-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 in M3; the rounded figure on the summary tab would read as M2, but the band follows the exact amount. Two eligible open items on hot-1 net to $342,165.48 of explained variance, leaving $393,177.87 unexplained."),
    ("FC12", "wrong", "w8", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in M2 because the rounded figure on the summary tab would read as M2. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC14", "right", "p1", "dev-std-002-BTC-M1", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves exceed ledger liabilities by $11,643.52, a surplus too small to count as reportable, so it sits in M1. WD-2510-164297 ($11,643.52), booked in the ledger but not yet broadcast, explains the full surplus, leaving $0.00 unexplained."),
    ("FC14", "right", "p2", "dev-hc-011-ADA-M1", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall in M1 that is too small to be material. Three open Hot 1 items at cutoff explain the full amount: DP-2510-557345 ($6,298.01), WD-2510-652593 ($7,396.30) and WD-2510-074129 ($6,283.24). Together they explain $19,977.55, leaving $0.00 unexplained."),
    ("FC14", "right", "p3", "dev-hc-007-SOL-M2", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, which is M2 (reportable) but too small to be material-risk. Two eligible HOT01 items explain $19,125.36, leaving $7,763.83 unexplained."),
    ("FC14", "wrong", "w1", "dev-hc-015-SOL-M1", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of $414,905,064.84, a shortfall in band M1 that is large enough to be reportable. Two open items explain $16,526.05 in total: TRF-2512-71501 ($6,637.86) and DP-2512-582869 ($9,888.19), so the unexplained remainder is $0.00."),
    ("FC14", "wrong", "w2", "dev-hc-003-ETH-M1", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M1. No eligible reconciling items support it, so the explained portion is USD 0.00 and the full USD 18,738.48 is unexplained. The difference is not too small to be reportable."),
    ("FC14", "wrong", "w3", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall too small to be in the reportable band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
    ("FC14", "wrong", "w4", "dev-hc-003-ETH-M1", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48 that falls in band M2. No eligible reconciling items support it, so the explained portion is USD 0.00 and the full USD 18,738.48 is unexplained. The difference should stay open pending the ops ticket and the full wallet export, even though it is too small to be reportable."),
    ("FC14", "wrong", "w5", "dev-hc-007-SOL-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus that is not large enough to be immaterial. Two eligible HOT01 items explain $19,125.36, leaving $7,763.83 unexplained."),
    ("FC15", "right", "p1", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, less than the $250,000.01 shortfall, and leave $70,350.50 unexplained."),
    ("FC15", "right", "p2", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "ROUTE", (),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in M3. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. The two withdrawals explain $179,649.51, which is below the $250,000.01 variance, so $70,350.50 remains unexplained."),
    ("FC15", "wrong", "w1", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, well under the $250,000.01 threshold, and leave $70,350.50 unexplained."),
    ("FC15", "wrong", "w2", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, well under the $250,000.00 shortfall, and leave $70,350.50 unexplained."),
    ("FC15", "wrong", "w3", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, well under the $250,000.01 surplus, and leave $70,350.50 unexplained."),
    ("FC15", "wrong", "w4", "dev-hc-013-ETH-M2", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M2 floor. The eligible open Hot 1 items explain $25,000.00 net, which is within the $25,000.01 surplus, leaving $0.00 unexplained."),
    ("FC15", "wrong", "w5", "dev-hc-005-USDT-M3", "FM", ('FM-002',), "FM", ('FM-002',),
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall well under the $250,000.01 shortfall threshold. Two open Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it, which explains $179,649.51, leaving $70,350.50 unexplained."),
]


@pytest.mark.parametrize("cons,kind,pid,case_id,base_status,base_codes,status,codes,text", [
    pytest.param(*r, id=f"{r[0]}-{r[2]}") for r in PROBES if r[1] == "right"])
def test_right_paraphrase_routes(cons, kind, pid, case_id, base_status, base_codes, status, codes, text):
    r = score(case_id, text)
    assert base_status == "FM" and status == "ROUTE"
    assert r["status"] == "ROUTE" and r["fms"] == [], r


@pytest.mark.parametrize("cons,kind,pid,case_id,base_status,base_codes,status,codes,text", [
    pytest.param(*r, id=f"{r[0]}-{r[2]}") for r in PROBES if r[1] == "wrong"])
def test_wrong_twin_keeps_the_code(cons, kind, pid, case_id, base_status, base_codes, status, codes, text):
    r = score(case_id, text)
    assert r["status"] == "FM" and tuple(r["fms"]) == codes == base_codes, r
    assert "FM-002" in r["fms"], r


def test_every_construction_has_paraphrases_and_twins():
    for cons in ("FC12", "FC14", "FC15"):
        kinds = [r[1] for r in PROBES if r[0] == cons]
        assert kinds.count("right") >= 2 and kinds.count("wrong") >= 2, cons


def test_every_change_is_fm_to_route():
    for r in PROBES:
        before, after = (r[4], r[5]), (r[6], r[7])
        if before == after:
            continue
        assert before[0] == "FM" and after == ("ROUTE", ()), r
