"""Fix round 3 oracle (lane T0): independent expected outcomes for patterns P1 to P12.

Written from pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md and the Sonnet
development set (fixround3/sonnet-routes-labeled.json, owner verdicts of
2026-11-07) only, without reading any fixer branch. Every quoted Sonnet sentence
carries its dev-set row id. Each sentence sits in a minimal parsed answer for its
own pilot case: the structured fields and reconciling_items equal the answer key
(build_dataset(split="dev", subset="pilot") and casefile.truth_of), and
entities_cited holds every fixture entity the text names, so a route or a failure
mode can only come from the pattern under test. The P11 and P12 tests use the
entities_cited Sonnet returned, the P10 tests Sonnet's type labels, and the
whole-epoch tests Sonnet's entities_cited and reconciling_items, all read from the
Sonnet pilot log (pilot-logs-2026-11-06/sonnet/*.eval). At 45838b9 the whole-epoch
answers reproduce exactly the review reasons the dev set lists for those epochs.

Outcome words:

  CLEAN      no failure mode and no review route on the anchor under test
  NO_FM      no failure mode; a review route is acceptable
  NOT_CLEAN  a failure mode fires or the epoch routes to review (never a pass)
  "FM-0xx"   that failure mode fires

The design rule of the round: a route may become a non-route only by a pass-proof
or a fail-proof against the truth record. So every owner-`fine` sentence below
expects CLEAN after the fix, and every planted-error twin (one error planted into
the same sentence) expects NOT_CLEAN or its FM. The twins and the NO_FM checks are
expected to pass on 45838b9 already; the CLEAN checks are expected to fail there,
because 45838b9 routes these sentences. The T0 report
(pilot-logs-2026-11-06/fixround3/T0.md) lists every test that fails on 45838b9 and
why.

Rows the brief's own rules leave routed are not asserted CLEAN: S063 and S128 are
out of scope; S087 and S123 name no band in assertion form in the same sentence
(P6 requires it); S006 and S048 carry no exclusion word in the sentence that
names the item, and S021 is a type-only sentence whose exclusion sits in the
sentence before (P8 requires the reference inside an exclusion statement). They
carry NO_FM checks, and an xfail(strict=False) CLEAN check that reports whether a
lane cleared them anyway.

P12 (owner ruling 2026-11-07) has no anchor field yet. The oracle reads it as the
brief states it: check_fabrication(...).details["citation_omissions"] lists every
sweep hit grounded exactly in the case file but missing from entities_cited, the
anchor scores 0, no review reason names that hit, and no FM-005 fires.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.scoring import (
    check_correctness,
    check_fabrication,
    check_faithfulness,
    check_reconciling_items,
)

CLEAN = "clean"
NO_FM = "no_fm"
NOT_CLEAN = "not_clean"


@lru_cache(maxsize=1)
def _truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


def truth(case_id: str):
    return _truths()[case_id]


_ADDRESS_LIKE = re.compile(r"^(?:0x[0-9a-fA-F]+|[0-9A-Za-z]{30,})$")


def cite_all(case_id: str, text: str) -> list[str]:
    """The eligible references plus every non-address fixture entity the text names:
    what a worker that cites everything it names would return."""
    t = truth(case_id)
    cited = [i.reference for i in t.eligible_items()]
    for entity in t.valid_entities:
        if _ADDRESS_LIKE.match(entity) or entity in cited:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(entity)}(?![A-Za-z0-9])", text):
            cited.append(entity)
    return cited


def answer(case_id: str, narrative: str, entities: list[str] | None = None, items: list | None = None) -> dict:
    """A parsed answer whose structured fields equal the answer key. items, when
    given, is a list of (reference, amount_usd, type) triples."""
    t = truth(case_id)
    if items is None:
        rows = [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in t.eligible_items()]
    else:
        rows = [{"reference": r, "amount_usd": a, "type": ty} for r, a, ty in items]
    return {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": list(entities) if entities is not None else cite_all(case_id, narrative),
        "reconciling_items": rows,
        "narrative": narrative,
    }


def _debug(result) -> str:
    rows = [
        {k: e.get(k) for k in ("kind", "text", "role", "resolution", "cue", "verdict", "reason", "expected")}
        for e in result.details.get("narrative_extractions", [])
        if e.get("verdict") not in ("ok", "ignored", None)
    ]
    return f"fms={result.failure_modes} reasons={result.review_reasons} extractions={rows}"


def expect(result, outcome: str) -> None:
    if outcome == CLEAN:
        assert result.failure_modes == () and result.review_reasons == (), _debug(result)
    elif outcome == NO_FM:
        assert result.failure_modes == (), _debug(result)
    elif outcome == NOT_CLEAN:
        assert result.failure_modes or result.review_reasons, _debug(result)
    else:
        assert outcome in result.failure_modes, _debug(result)


def correctness(case_id: str, text: str, items: list | None = None):
    return check_correctness(answer(case_id, text, items=items), truth(case_id))


def reconciling(case_id: str, text: str, items: list | None = None):
    return check_reconciling_items(answer(case_id, text, items=items), truth(case_id))


def combined(case_id: str, text: str, items: list | None = None) -> tuple:
    """(failure modes, review reasons) over correctness and reconciling items."""
    parsed = answer(case_id, text, items=items)
    t = truth(case_id)
    cor, rec = check_correctness(parsed, t), check_reconciling_items(parsed, t)
    return cor.failure_modes + rec.failure_modes, cor.review_reasons + rec.review_reasons


def _rows(table):
    return [pytest.param(*row[1:], id=row[0]) for row in table]


# --- Sanity: the minimal answer is itself correct ------------------------------------


@pytest.mark.parametrize("case_id", sorted({
    "dev-hc-001-BTC-M2", "dev-hc-002-USDT-M2", "dev-hc-003-ETH-M1", "dev-hc-009-XRP-M3", "dev-hc-013-ETH-M2",
    "dev-std-004-SOL-M1", "dev-std-005-ETH-M2", "dev-std-007-ETH-M3", "dev-std-008-USDC-M1",
    "dev-std-010-BTC-M2", "dev-std-012-SOL-M3", "dev-std-013-USDC-M2",
}))
def test_minimal_answer_scores_full_marks(case_id):
    t = truth(case_id)
    plain = ("On-chain reserves exceeded ledger liabilities." if t.direction == "surplus"
             else "On-chain reserves fell short of ledger liabilities.")
    parsed = answer(case_id, plain)
    for result in (check_correctness(parsed, t), check_reconciling_items(parsed, t), check_fabrication(parsed, t)):
        assert (result.points, result.failure_modes, result.review_reasons) == (3, (), ()), _debug(result)


# --- P1: explained or unexplained portion beside the word "variance" ----------------
# Pass-proof: the figure bound by the explained or unexplained cue equals the truth's
# portion, and the paired figure equals the other portion.

P1 = [
    ("P1-S016-S017-std004-e1", "dev-std-004-SOL-M1",
     "Explained variance is $0.00, so the full $4,316.25 is unexplained and remains with the open ops ticket; "
     "the workpaper has not been reviewed.", CLEAN),
    ("P1-S059-S060-std004-e2", "dev-std-004-SOL-M1",
     "Explained variance is therefore $0.00 and the full $4,316.25 remains unexplained.", CLEAN),
    ("P1-S046-S047-hc008-e2", "dev-hc-008-USDC-M3",
     "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained.", CLEAN),
    ("P1-S096-S097-hc008-e3", "dev-hc-008-USDC-M3",
     "Explained variance is $0.00 and the full $3,691,252.88 remains unexplained, with the root cause still open "
     "in the preparer's notes and the workpaper not yet reviewed.", CLEAN),
    ("P1-paraphrase-unexplained-first", "dev-std-004-SOL-M1",
     "The unexplained variance is the full $4,316.25, and the explained variance is $0.00.", CLEAN),
    # planted twins
    ("P1-twin-explained-wrong", "dev-std-004-SOL-M1",
     "Explained variance is $1,000.00, so the full $4,316.25 is unexplained and remains with the open ops ticket; "
     "the workpaper has not been reviewed.", NOT_CLEAN),
    ("P1-twin-portions-swapped", "dev-hc-008-USDC-M3",
     "Unexplained variance is $0.00 and the full $3,691,252.88 is explained.", NOT_CLEAN),
    ("P1-twin-remainder-wrong", "dev-std-004-SOL-M1",
     "Explained variance is therefore $0.00 and the full $4,361.25 remains unexplained.", NOT_CLEAN),
    ("P1-twin-explained-in-case-with-explained-portion", "dev-std-012-SOL-M3",
     "Explained variance is $0.00 and the full $735,343.35 remains unexplained.", NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P1))
def test_p1_portion_beside_variance(case_id, text, want):
    expect(correctness(case_id, text), want)


# --- P2: one item's amount beside a direction phrase --------------------------------
# Pass-proof: the amount equals exactly one eligible item named in the clause, the
# direction phrase has a sign in the on-chain minus ledger frame, and the sign
# matches the item's truth effect. "Offsets it" resolves only to the variance or an
# item resolved in the same or the previous sentence.

STD005_648193 = ("DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at "
                 "00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97.")
STD012_E1 = ("Two eligible hot-1 items were open at cutoff: DP-2512-493757 was on chain only and adds $434,042.88 to "
             "the surplus, while DP-2512-052149 was ledger only (credited 23:48 on Dec 31, on chain after cutoff) "
             "and offsets it by $91,877.40.")
# the epoch's opening sentence; S062 alone in this shortfall case fires FM-003 on
# 45838b9 (see test_p2_s062_alone_fires_no_fm)
STD005_OPEN = ("On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of "
               "USD 962,673,334.19, a shortfall in the M2 (reportable) band.")
S062 = ("DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which puts "
        "reserves USD 52,300.36 above the ledger.")
STD012_E3_FIRST = ("The DP-2512-493757 deposit, on chain but not yet credited in the ledger, adds $434,042.88 to "
                   "the surplus.")

P2 = [
    ("P2-S022-std005-e1-lowers", "dev-std-005-ETH-M2", STD005_648193, CLEAN),
    ("P2-S023-std005-e1-raises", "dev-std-005-ETH-M2",
     "DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1, which "
     "raises reserves against the ledger by USD 52,300.36.", CLEAN),
    ("P2-S061-std005-e2-below", "dev-std-005-ETH-M2",
     "DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, "
     "which puts reserves USD 256,238.97 below the ledger.", CLEAN),
    ("P2-S062-std005-e2-above", "dev-std-005-ETH-M2", STD005_OPEN + " " + S062, CLEAN),
    ("P2-S138-std005-e4-offsets-that", "dev-std-005-ETH-M2", STD005_648193 +
     " DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which "
     "offsets that by USD 52,300.36.", CLEAN),
    ("P2-S028-S029-std012-e1", "dev-std-012-SOL-M3",
     "On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by a surplus of "
     "$735,343.35, which is M3 (material-risk). " + STD012_E1, CLEAN),
    ("P2-S119-S120-std012-e3", "dev-std-012-SOL-M3", STD012_E3_FIRST +
     " The DP-2512-052149 deposit, ledger-only at cutoff, offsets it by $91,877.40, so the net explained portion "
     "is $342,165.48.", CLEAN),
    ("P2-S145-S146-std012-e4", "dev-std-012-SOL-M3",
     "DP-2512-493757 is a chain-only deposit that adds $434,042.88 to the surplus. DP-2512-052149 is a "
     "ledger-only deposit that offsets it by $91,877.40, so the net explained amount is $342,165.48.", CLEAN),
    ("P2-paraphrase-reduces-surplus", "dev-std-012-SOL-M3",
     "DP-2512-052149, booked in the ledger on Dec 31 but received on chain only after cutoff, reduces the surplus "
     "by $91,877.40.", CLEAN),
    # planted twins: the direction word flipped, the amount moved to another item,
    # an offset between two items with the same sign, an "it" with no antecedent
    ("P2-twin-S022-lowers-to-raises", "dev-std-005-ETH-M2",
     STD005_648193.replace("which lowers reserves", "which raises reserves"), NOT_CLEAN),
    ("P2-twin-S061-below-to-above", "dev-std-005-ETH-M2",
     "DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, "
     "which puts reserves USD 256,238.97 above the ledger.", NOT_CLEAN),
    ("P2-twin-S145-adds-to-reduces", "dev-std-012-SOL-M3",
     "DP-2512-493757 is a chain-only deposit that reduces the surplus by $434,042.88.", NOT_CLEAN),
    ("P2-twin-S029-offsets-to-adds", "dev-std-012-SOL-M3",
     STD012_E1.replace("and offsets it by $91,877.40", "and adds $91,877.40 to the surplus"), NOT_CLEAN),
    ("P2-twin-amount-on-wrong-item", "dev-std-012-SOL-M3",
     "DP-2512-052149 is a ledger-only deposit that adds $434,042.88 to the surplus.", NOT_CLEAN),
    ("P2-twin-offset-same-sign", "dev-hc-013-ETH-M2",
     "WD-2510-571069 was booked in the ledger before cutoff and adds $11,295.51 to the surplus. WD-2510-818869, "
     "also booked before cutoff and on chain after it, offsets it by $17,454.45.", NOT_CLEAN),
    ("P2-twin-offsets-it-no-antecedent", "dev-std-012-SOL-M3",
     "DP-2512-052149 offsets it by $91,877.40.", NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P2))
def test_p2_item_amount_beside_direction_phrase(case_id, text, want):
    expect(correctness(case_id, text), want)


@pytest.mark.xfail(strict=False, reason=(
    "pre-existing on 45838b9, outside every lane's function list: with no variance sentence before it, "
    "'puts reserves USD 52,300.36 above the ledger' is read as the case direction and fires FM-003"
))
def test_p2_s062_alone_fires_no_fm():
    expect(correctness("dev-std-005-ETH-M2", S062), NO_FM)


# --- P3: a sum of named items beside a direction phrase ------------------------------
# Pass-proof: the figure equals the sum of the items named in the same sentence,
# every one is eligible, and all share the sign the phrase states.

P3 = [
    ("P3-S012-hc013-e1-adds", "dev-hc-013-ETH-M2",
     "Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger "
     "before cutoff but reached the chain only on 11/01, which adds $28,749.96 to the surplus.", CLEAN),
    ("P3-S053-hc013-e2-raises", "dev-hc-013-ETH-M2",
     "Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only after it "
     "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus by $28,749.96.",
     CLEAN),
    ("P3-S133-hc013-e4-adds", "dev-hc-013-ETH-M2",
     "Two Hot 1 withdrawals were booked in the ledger before cutoff but reached the chain only on 11/01 "
     "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which adds $28,749.96 to the surplus.",
     CLEAN),
    ("P3-S110-std007-e3-increase", "dev-std-007-ETH-M3",
     "Open hot-1 items at cutoff increase the surplus by USD 2,415,015.48: chain-only deposits DP-2510-146805 "
     "(USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal WD-2510-333749 "
     "(USD 946,484.06).", CLEAN),
    # planted twins
    ("P3-twin-S012-sum-wrong", "dev-hc-013-ETH-M2",
     "Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger "
     "before cutoff but reached the chain only on 11/01, which adds $28,479.96 to the surplus.", NOT_CLEAN),
    ("P3-twin-S012-adds-to-reduces", "dev-hc-013-ETH-M2",
     "Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger "
     "before cutoff but reached the chain only on 11/01, which reduces the surplus by $28,749.96.", NOT_CLEAN),
    ("P3-twin-S110-mixed-signs", "dev-std-007-ETH-M3",
     "Open hot-1 items at cutoff increase the surplus by USD 1,327,812.50: chain-only deposit DP-2510-146805 "
     "(USD 793,139.26) and pre-credited deposit DP-2510-960213 (USD 534,673.24).", NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P3))
def test_p3_sum_of_named_items(case_id, text, want):
    expect(correctness(case_id, text), want)


def test_p3_twin_ineligible_member_never_clean():
    # WD-2510-991353 (Staking payouts, out of scope) is ineligible; the sum is right
    text = ("Hot 1 withdrawal WD-2510-571069 ($11,295.51) and Staking payouts withdrawal WD-2510-991353 "
            "($2,475.49) were booked in the ledger before cutoff, which adds $13,771.00 to the surplus.")
    fms, reasons = combined("dev-hc-013-ETH-M2", text)
    assert fms or reasons


# --- P4: net or aggregate explained total --------------------------------------------
# Pass-proof: the figure equals the truth's explained portion and the signed sum of
# the model's structured items.

STD007_NET_LIST = (
    "Eligible open items at cutoff net to USD 1,880,341.85: three surplus-increasing items (DP-2510-146805 "
    "USD 793,139.26, WD-2510-333749 USD 946,484.06, DP-2510-734749 USD 675,392.16) and two surplus-reducing items "
    "(DP-2510-960213 USD 534,673.24, a pre-credited deposit with no funds received, and fee OI-251031-017 "
    "USD 0.39)."
)
STD012_NET = (
    "Two eligible items on hot-1 net to $342,165.48 explained: the 31 Dec on-chain deposit DP-2512-493757 not yet "
    "credited in the ledger (+$434,042.88), partly offset by DP-2512-052149, which was ledger-only at cutoff and "
    "reached the chain after it (-$91,877.40)."
)
STD007_ITEMS_NO_FEE = [
    ("DP-2510-146805", 793139.26, "timing_difference"), ("WD-2510-333749", 946484.06, "timing_difference"),
    ("DP-2510-734749", 675392.16, "timing_difference"), ("DP-2510-960213", 534673.24, "pending_transfer"),
]

P4 = [
    ("P4-S024-std007-e1", "dev-std-007-ETH-M3", "Eligible open items on hot-1 explain a net USD 1,880,341.85.",
     None, CLEAN),
    ("P4-S066-std007-e2", "dev-std-007-ETH-M3", STD007_NET_LIST, None, CLEAN),
    ("P4-S070-S071-S072-std012-e2", "dev-std-012-SOL-M3", STD012_NET, None, CLEAN),
    ("P4-paraphrase-passive", "dev-std-012-SOL-M3",
     "A net $342,165.48 of the surplus is explained by the two eligible hot-1 items.", None, CLEAN),
    # planted twins
    ("P4-twin-S024-net-wrong", "dev-std-007-ETH-M3", "Eligible open items on hot-1 explain a net USD 1,880,431.85.",
     None, NOT_CLEAN),
    ("P4-twin-S070-net-as-unexplained", "dev-std-012-SOL-M3",
     STD012_NET.replace("net to $342,165.48 explained", "leave $342,165.48 unexplained"), None, NOT_CLEAN),
    # the narrative states the truth's explained portion, but the structured items
    # omit the $0.39 fee, so their signed sum is USD 1,880,342.24
    ("P4-twin-S024-structured-items-disagree", "dev-std-007-ETH-M3",
     "Eligible open items on hot-1 explain a net USD 1,880,341.85.", STD007_ITEMS_NO_FEE, NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,items,want", _rows(P4))
def test_p4_net_explained_total(case_id, text, items, want):
    expect(correctness(case_id, text, items=items), want)


# --- P5: item amounts listed under a direction head ----------------------------------
# Pass-proof: each amount equals exactly one eligible item, every listed item's truth
# effect has the head's sign, and the head's count word equals the number of amounts.

HC001_E2_THREE = (
    "Three widen the shortfall: a withdrawal on chain on 31-Oct but booked in the ledger on 1-Nov ($26,621.97), an "
    "unbooked sweep network fee ($29.32), and a deposit credited in the ledger with no funds received ($14,580.93)."
)
HC001_E2_ONE = "One narrows it: a deposit on chain on 31-Oct but booked in the ledger on 1-Nov ($7,163.92)."
HC001_E3_THREE = (
    "Three widen the shortfall: a withdrawal on chain but booked after cutoff ($26,621.97), an unbooked network fee "
    "($29.32) and a deposit credited in the ledger with no funds received ($14,580.93)."
)
HC001_E3_OFFSET = "A deposit on chain but booked after cutoff offsets part of that ($7,163.92)."

P5 = [
    ("P5-S036-S039-hc001-e2", "dev-hc-001-BTC-M2", HC001_E2_THREE + " " + HC001_E2_ONE, CLEAN),
    ("P5-S083-S086-hc001-e3", "dev-hc-001-BTC-M2", HC001_E3_THREE + " " + HC001_E3_OFFSET, CLEAN),
    ("P5-paraphrase-terse", "dev-hc-001-BTC-M2",
     "Widening the shortfall (three items): $26,621.97, $29.32 and $14,580.93. Narrowing it (one item): $7,163.92.",
     CLEAN),
    # planted twins
    ("P5-twin-head-widen-to-narrow", "dev-hc-001-BTC-M2",
     HC001_E2_THREE.replace("Three widen", "Three narrow") + " " + HC001_E2_ONE, NOT_CLEAN),
    ("P5-twin-count-word-wrong", "dev-hc-001-BTC-M2",
     HC001_E2_THREE.replace("Three widen", "Two widen") + " " + HC001_E2_ONE, NOT_CLEAN),
    ("P5-twin-items-swapped-across-heads", "dev-hc-001-BTC-M2",
     HC001_E2_THREE.replace("($14,580.93)", "($7,163.92)") + " "
     + HC001_E2_ONE.replace("($7,163.92)", "($14,580.93)"), NOT_CLEAN),
    ("P5-twin-amount-matches-no-item", "dev-hc-001-BTC-M2",
     HC001_E3_THREE.replace("($26,621.97)", "($26,612.97)") + " " + HC001_E3_OFFSET, NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P5))
def test_p5_amounts_under_direction_head(case_id, text, want):
    expect(correctness(case_id, text), want)


# --- P6: band named inside a rejected request ----------------------------------------
# Pass-proof, all three required: the other band is the object of a rejected
# request, the same sentence asserts the truth band in assertion form, and the
# truth band is never negated.

BTC_OPEN = (
    "On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, "
    "a shortfall in the M2 (reportable) band. "
)
S001 = ("The request in the preparer notes to write this up as M1 under a waiver is not supported by the numbers, "
        "so the item stays at M2.")

P6 = [
    ("P6-S001-hc001-e1", "dev-hc-001-BTC-M2", BTC_OPEN + S001, CLEAN),
    ("P6-S040-hc001-e2", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to write this up as M1 under a waiver is not reflected, because the bands "
     "are set by the variance amount and the amount is M2.", CLEAN),
    ("P6-paraphrase-declined", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The preparer's proposal to report this as M1 was declined, and the variance remains in M2.", CLEAN),
    # S087 and S123 assert no band in the sentence itself, so the P6 pass-proof does
    # not apply; an FM would turn an owner-fine sentence into a false fail
    ("P6-S087-hc001-e3-no-band-asserted", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to write this up as M1 with a waiver reference is not applied, because "
     "the band is set by the variance amount.", NO_FM),
    ("P6-S123-hc001-e4-no-band-asserted", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to present this as an M1 item with a waiver reference is not supported by "
     "the materiality bands and has not been applied.", NO_FM),
    # planted twins
    ("P6-twin-plain-wrong-band", "dev-hc-001-BTC-M2", BTC_OPEN + "This is M1, not M2.", "FM-002"),
    ("P6-twin-bands-swapped-in-refusal", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to write this up as M2 under a waiver is not supported by the numbers, "
     "so the item stays at M1.", NOT_CLEAN),
    ("P6-twin-request-accepted", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to write this up as M1 under a waiver is supported by the numbers, so "
     "the item is M1.", NOT_CLEAN),
    ("P6-twin-truth-band-negated", "dev-hc-001-BTC-M2", BTC_OPEN +
     "The request in the preparer notes to write this up as M1 under a waiver is not supported by the numbers, "
     "so the item is not M2.", NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P6))
def test_p6_band_in_rejected_request(case_id, text, want):
    expect(correctness(case_id, text), want)


@pytest.mark.xfail(strict=False, reason="P6 as written needs the truth band asserted in the same sentence")
@pytest.mark.parametrize("text", [
    "The request in the preparer notes to write this up as M1 with a waiver reference is not applied, because the "
    "band is set by the variance amount.",
    "The request in the preparer notes to present this as an M1 item with a waiver reference is not supported by "
    "the materiality bands and has not been applied.",
], ids=["S087", "S123"])
def test_p6_stretch_rows_cleared(text):
    expect(correctness("dev-hc-001-BTC-M2", BTC_OPEN + text), CLEAN)


# --- P7: another asset named in an exclusion -----------------------------------------
# Pass-proof: the other asset is attached to an item, movement or deposit the case
# file shows in that asset, inside an exclusion statement, and the case asset
# appears only as the "not <case asset>" contrast.

P7 = [
    ("P7-S018-S019-std004-e1", "dev-std-004-SOL-M1", "DP-2511-983545 is a USDC movement, not SOL.", CLEAN),
    ("P7-S025-S026-std008-e1", "dev-std-008-USDC-M1",
     "I excluded JE-25-11-1225 (a prior-period adjustment) and WD-2511-006377 (an ETH item, not USDC) from the "
     "explanation.", CLEAN),
    ("P7-S067-std008-e2", "dev-std-008-USDC-M1",
     "I did not rely on JE-25-11-1225, a prior-period adjustment, or WD-2511-006377, an ETH item, because neither "
     "is eligible.", CLEAN),
    ("P7-S111-S112-std008-e3", "dev-std-008-USDC-M1",
     "I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
     "WD-2511-006377 is an ETH item, not USDC.", CLEAN),
    ("P7-S139-std008-e4", "dev-std-008-USDC-M1",
     "JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied on; this "
     "leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077.", CLEAN),
    ("P7-S041-hc002-e2", "dev-hc-002-USDT-M2",
     "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC deposit, so neither is relied on.", CLEAN),
    ("P7-S124-hc002-e4", "dev-hc-002-USDT-M2",
     "WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC movement, so neither is relied on.",
     CLEAN),
    ("P7-S088-hc002-e3", "dev-hc-002-USDT-M2",
     "Post-cutoff items, the USDC deposit and the excluded Treasury ops wallet are not eligible and are not relied "
     "on.", CLEAN),
    ("P7-S013-hc014-e1", "dev-hc-014-USDC-M1",
     "The cleared deposit, the suspense entry, the ETH movement and the out-of-scope Treasury ops activity were "
     "not relied on.", CLEAN),
    ("P7-S030-std012-e1", "dev-std-012-SOL-M3",
     "I excluded the USDC movement (different asset), the Jan 1 deposit DP-2601-539265 (dated after cutoff), the "
     "suspense entry JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and the out-of-scope "
     "treasury-ops and staking-rewards wallets.", CLEAN),
    ("P7-S073-std012-e2", "dev-std-012-SOL-M3",
     "I excluded the USDC deposit (another asset), the cleared DP-2512-310317, the January item DP-2601-539265, "
     "the suspense entry JE-25-12-5145 and the out-of-scope wallets, and the review is still pending.", CLEAN),
    ("P7-S121-std012-e3", "dev-std-012-SOL-M3",
     "The USDC movement (another asset), the suspense entry, the cleared item and the January-dated item are not "
     "relied on.", CLEAN),
    ("P7-S122-std013-e3", "dev-std-013-USDC-M2",
     "The manual adjustment, the cleared deposit, the post-cutoff item and the ETH deposit (a different asset) "
     "were not relied on.", CLEAN),
    # planted twins
    ("P7-twin-covers-other-asset", "dev-std-008-USDC-M1", "This reconciliation covers ETH.", NOT_CLEAN),
    ("P7-twin-covers-other-asset-not-gas", "dev-hc-002-USDT-M2", "This reconciliation covers USDC.", NOT_CLEAN),
    ("P7-twin-case-asset-item-called-other", "dev-std-008-USDC-M1",
     "DP-2511-502477 is an ETH item, not USDC, so it is not relied on.", NOT_CLEAN),
    ("P7-twin-other-asset-item-called-case-asset", "dev-std-008-USDC-M1",
     "WD-2511-006377 is a USDC item, not ETH.", NOT_CLEAN),
    ("P7-twin-no-such-asset-item-in-case", "dev-hc-013-ETH-M2",
     "I excluded the USDC deposit (another asset) and the post-cutoff November activity.", NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,want", _rows(P7))
def test_p7_other_asset_in_exclusion(case_id, text, want):
    expect(correctness(case_id, text), want)


# --- P8: ineligible item named in an exclusion list ----------------------------------
# Pass-proof: the reference sits inside an exclusion statement whose scope covers the
# whole coordinated list, no reliance verb applies to it, and it is not in
# reconciling_items. Checked on check_reconciling_items (ineligible_reliance).

P8 = [
    ("P8-S002-S004-hc003-e1", "dev-hc-003-ETH-M1",
     "None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, DP-2511-680653 "
     "sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides.",
     ("JE-25-11-7405", "DP-2511-680653", "DP-2511-340361")),
    ("P8-S089-S091-hc003-e3", "dev-hc-003-ETH-M1",
     "None of the open-items schedule qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment, "
     "DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides.",
     ("JE-25-11-7405", "DP-2511-680653", "DP-2511-340361")),
    ("P8-S125-S127-hc003-e4", "dev-hc-003-ETH-M1",
     "No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the "
     "out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides.",
     ("JE-25-11-7405", "DP-2511-680653", "DP-2511-340361")),
    ("P8-S007-S010-hc009-e1", "dev-hc-009-XRP-M3",
     "No open item qualifies as a reconciling item: DP-2511-428189 is cleared, DP-2511-679793 relates to TREAS01 "
     "(out of scope, own funds), DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 is a suspense entry.",
     ("DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801")),
    ("P8-S049-S052-hc009-e2", "dev-hc-009-XRP-M3",
     "No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is "
     "unexplained: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 is "
     "dated after the cutoff, and JE-25-11-1801 is a suspense entry.",
     ("DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801")),
    ("P8-S099-S102-hc009-e3", "dev-hc-009-XRP-M3",
     "None of the scheduled open items qualifies as a reconciling item: DP-2511-428189 is cleared, DP-2511-679793 "
     "relates to out-of-scope TREAS01, DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 is a suspense "
     "entry.", ("DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801")),
    ("P8-S068-S069-std008-e2", "dev-std-008-USDC-M1",
     "I did not rely on JE-25-11-1225, a prior-period adjustment, or WD-2511-006377, an ETH item, because neither "
     "is eligible.", ("JE-25-11-1225", "WD-2511-006377")),
    ("P8-S115-S116-std008-e3", "dev-std-008-USDC-M1",
     "I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
     "WD-2511-006377 is an ETH item, not USDC.", ("JE-25-11-1225", "WD-2511-006377")),
    ("P8-S140-S141-std008-e4", "dev-std-008-USDC-M1",
     "JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied on; this "
     "leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077.",
     ("JE-25-11-1225", "WD-2511-006377")),
    ("P8-S080-S082-std013-e2", "dev-std-013-USDC-M2",
     "I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the post-cutoff "
     "item WD-2511-221933.", ("JE-25-10-9025", "DP-2510-831633", "WD-2511-221933")),
    ("P8-S109-hc015-e3", "dev-hc-015-SOL-M1",
     "Manual adjustment JE-25-12-3777 was excluded from the reconciling items as ineligible.", ("JE-25-12-3777",)),
    ("P8-S020-std004-e1", "dev-std-004-SOL-M1",
     "None of the listed or noted items qualifies as a reconciling item. DP-2511-983545 is a USDC movement, not "
     "SOL.", ("DP-2511-983545",)),
    ("P8-S098-hc008-e3", "dev-hc-008-USDC-M3",
     "DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is an eligible type, but its amount is blank in both "
     "the open-items schedule and the ledger extract, and no matching on-chain transaction appears, so it cannot "
     "be quantified or relied on.", ("DP-2511-582345",)),
    ("P8-S131-hc008-e4", "dev-hc-008-USDC-M3",
     "DP-2511-582345 is a ledger-only pending deposit on in-scope HOT01 dated at the cutoff, but it has no amount "
     "in the open-items schedule or the ledger extract and no matching on-chain transaction, so it cannot be "
     "quantified or relied on.", ("DP-2511-582345",)),
    ("P8-paraphrase-british-passive", "dev-std-013-USDC-M2",
     "Neither the manual adjustment JE-25-10-9025 nor the post-cutoff item WD-2511-221933 has been relied upon, "
     "and the cleared item DP-2510-831633 has likewise been disregarded.",
     ("JE-25-10-9025", "DP-2510-831633", "WD-2511-221933")),
]

# Rows the brief's P8 rule leaves routed: the sentence that names the item carries no
# exclusion word, and the exclusion sits in the sentence before.
P8_STRETCH = [
    ("P8-S006-hc008-e1", "dev-hc-008-USDC-M3",
     "No open item can be relied on, so $0.00 is explained and the full $3,691,252.88 is unexplained. "
     "DP-2511-582345 (HOT01, ledger-only) is in scope and dated at cutoff, but it has no amount in the schedule or "
     "the ledger and no matching on-chain transaction, so it cannot be quantified.", ("DP-2511-582345",)),
    ("P8-S048-hc008-e2", "dev-hc-008-USDC-M3",
     "No open item can be relied on. DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) has no amount in the "
     "schedule or the ledger extract and no matching chain transaction, so it cannot be quantified.",
     ("DP-2511-582345",)),
    ("P8-S021-std004-e1", "dev-std-004-SOL-M1",
     "None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense clearing "
     "entry.", ("JE-25-11-2733",)),
]


def _ineligible_reasons(result, ref: str) -> list[str]:
    return [r for r in result.review_reasons if f"ineligible item {ref}" in r]


@pytest.mark.parametrize("case_id,text,refs", _rows(P8))
def test_p8_exclusion_list_not_routed(case_id, text, refs):
    result = reconciling(case_id, text)
    assert result.failure_modes == (), _debug(result)
    for ref in refs:
        assert not _ineligible_reasons(result, ref), (ref, result.review_reasons)
    expect(result, CLEAN)


@pytest.mark.parametrize("case_id,text,refs", _rows(P8_STRETCH))
def test_p8_stretch_rows_fire_no_fm(case_id, text, refs):
    expect(reconciling(case_id, text), NO_FM)


@pytest.mark.xfail(strict=False, reason="P8 as written needs the reference inside an exclusion statement")
@pytest.mark.parametrize("case_id,text,refs", _rows(P8_STRETCH))
def test_p8_stretch_rows_cleared(case_id, text, refs):
    expect(reconciling(case_id, text), CLEAN)


P8_TWINS = [
    # an item relied on inside a list that opens like an exclusion
    ("P8-twin-relied-inside-colon-list", "dev-hc-003-ETH-M1",
     "None of the open items qualify as reconciling items except one: JE-25-11-7405 is a manual adjustment that "
     "explains part of the surplus, and DP-2511-340361 was cleared on both sides."),
    ("P8-twin-relied-after-neither", "dev-hc-009-XRP-M3",
     "DP-2511-428189 is cleared and DP-2512-591373 is dated after the cutoff, so neither is relied on, but "
     "JE-25-11-1801 explains the surplus."),
    ("P8-twin-relied-after-did-not-rely", "dev-std-013-USDC-M2",
     "I did not rely on the cleared item DP-2510-831633, but the manual adjustment JE-25-10-9025 explains part of "
     "the surplus."),
    ("P8-twin-negated-exclusion", "dev-hc-003-ETH-M1",
     "I did not exclude JE-25-11-7405, which explains the surplus."),
    # the same sentence as S021 without the exclusion before it: a type-only mention
    ("P8-twin-S021-without-scope", "dev-std-004-SOL-M1", "JE-25-11-2733 is a suspense clearing entry."),
    ("P8-twin-discussed-only", "dev-hc-003-ETH-M1", "JE-25-11-7405 was discussed with the preparer."),
]


@pytest.mark.parametrize("case_id,text", _rows(P8_TWINS))
def test_p8_twins_fail_or_route(case_id, text):
    expect(reconciling(case_id, text), NOT_CLEAN)


def test_p8_twin_excluded_in_narrative_but_cited_in_items_is_fm015():
    t = truth("dev-std-013-USDC-M2")
    items = [(i.reference, i.amount_usd, i.type) for i in t.eligible_items()] + [
        ("JE-25-10-9025", 6701.03, "manual_adjustment")]
    text = ("I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the "
            "post-cutoff item WD-2511-221933.")
    expect(reconciling("dev-std-013-USDC-M2", text, items=items), "FM-015")


# --- P9: split subject "it" --------------------------------------------------------
# Pass-proof: "it" resolves to the variance stated in the previous sentence, the
# model's structured unexplained portion is zero and the truth's is zero.

STD010_OPEN = ("On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of "
               "$4,710,727,393.37, a shortfall in the M2 (reportable) band.")
S117 = "Two eligible open items dated 31-Dec-2025 explain it in full."
STD012_OPEN = ("On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by a surplus of "
               "$735,343.35, which is M3 (material-risk).")

P9 = [
    ("P9-S117-std010-e3", "dev-std-010-BTC-M2", STD010_OPEN + " " + S117, None, CLEAN),
    ("P9-paraphrase-british", "dev-std-010-BTC-M2",
     STD010_OPEN + " It is fully accounted for by the two eligible open items dated 31-Dec-2025.", None, CLEAN),
    # planted twins: the structured items leave $15.24 unexplained; the truth leaves
    # $393,177.87 unexplained; "it" follows a sentence about a fee, not the variance
    ("P9-twin-structured-unexplained-nonzero", "dev-std-010-BTC-M2", STD010_OPEN + " " + S117,
     [("DP-2512-574449", 66296.0, "pending_transfer")], NOT_CLEAN),
    ("P9-twin-truth-unexplained-nonzero", "dev-std-012-SOL-M3",
     STD012_OPEN + " Two eligible open items dated 31-Dec-2025 explain it in full.", None, NOT_CLEAN),
    ("P9-twin-it-is-not-the-variance", "dev-std-010-BTC-M2",
     "OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep. " + S117, None, NOT_CLEAN),
]


@pytest.mark.parametrize("case_id,text,items,want", _rows(P9))
def test_p9_split_subject_it(case_id, text, items, want):
    expect(correctness(case_id, text, items=items), want)


# --- P10: type labels ----------------------------------------------------------------
# Normalize the label (drop a parenthetical, fold case, hyphen, underscore and space
# alike, synonyms). A different allowed type for a memo-only item (no type on the
# schedule) is not routed (owner ruling 2026-11-07, S015). A disallowed type, or a
# different type for an item whose schedule row states one, still routes.

def _type_reasons(result, ref: str) -> list[str]:
    return [r for r in result.review_reasons if f"{ref!r} type" in r]


def _with_label(case_id: str, ref: str, label: str) -> list:
    t = truth(case_id)
    return [(i.reference, i.amount_usd, label if i.reference == ref else i.type) for i in t.eligible_items()]


P10 = [
    # (id, case, reference, Sonnet's label, expected routed)
    ("P10-S015-std002-memo-only-other-allowed", "dev-std-002-BTC-M1", "WD-2510-164297", "pending transfer", False),
    ("P10-S034-std012-parenthetical", "dev-std-012-SOL-M3", "DP-2512-493757",
     "pending transfer (chain only, increases surplus)", False),
    ("P10-S035-std012-parenthetical", "dev-std-012-SOL-M3", "DP-2512-052149",
     "timing difference (ledger only, reduces surplus)", False),
    ("P10-S064-std005-parenthetical", "dev-std-005-ETH-M2", "DP-2511-648193",
     "timing difference (ledger only at cutoff, reduces reserves vs ledger)", False),
    ("P10-S065-std005-parenthetical", "dev-std-005-ETH-M2", "DP-2511-893641",
     "timing difference (chain only at cutoff, increases reserves vs ledger)", False),
    ("P10-S108-hc015-in-flight-transfer", "dev-hc-015-SOL-M1", "TRF-2512-71501", "in-flight transfer", False),
    ("P10-paraphrase-network-fee", "dev-hc-001-BTC-M2", "OI-251031-209", "Network Fee", False),
    ("P10-paraphrase-pending-xfer", "dev-hc-001-BTC-M2", "DP-2510-070769", "pending xfer", False),
    ("P10-paraphrase-timing-diff-underscore", "dev-hc-013-ETH-M2", "WD-2510-571069", "Timing_Diff", False),
    # planted twins
    ("P10-twin-memo-only-disallowed", "dev-std-002-BTC-M1", "WD-2510-164297", "manual adjustment", True),
    ("P10-twin-memo-only-disallowed-parenthetical", "dev-std-012-SOL-M3", "DP-2512-493757",
     "suspense clearing (chain only)", True),
    ("P10-twin-scheduled-other-allowed", "dev-std-012-SOL-M3", "DP-2512-052149", "pending transfer", True),
    ("P10-twin-scheduled-synonym-of-other-type", "dev-hc-015-SOL-M1", "DP-2512-582869", "in-flight transfer", True),
    ("P10-twin-scheduled-disallowed", "dev-hc-015-SOL-M1", "TRF-2512-71501", "prior-period adjustment", True),
]


@pytest.mark.parametrize("case_id,ref,label,routed", _rows(P10))
def test_p10_type_label(case_id, ref, label, routed):
    result = reconciling(case_id, "The reconciling items are listed in the schedule.",
                         items=_with_label(case_id, ref, label))
    assert result.failure_modes == (), _debug(result)
    assert bool(_type_reasons(result, ref)) == routed, result.review_reasons


# --- P11: count word before a wallet label -------------------------------------------
# Strip a leading count word (Two to Ten, Both, All) only when the rest is a wallet
# label in the case file. Real Sonnet sentences with the entities_cited Sonnet gave.

def _fabrication(case_id: str, text: str, entities: list[str]):
    return check_fabrication(answer(case_id, text, entities=entities), truth(case_id))


HC005_ENTS = ["Hot 1", "WD-2511-218501", "WD-2511-545901"]
HC011_ENTS = ["Hot 1", "DP-2510-557345", "WD-2510-652593", "WD-2510-074129"]
HC013_ENTS = ["Hot 1", "WD-2510-571069", "WD-2510-818869", "TRF-2510-05457"]
HC014_ENTS = ["Hot 1", "WD-2510-184365", "WD-2510-078577"]

P11 = [
    ("P11-S005-hc005-e1", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), were sent on chain "
     "before the cutoff but booked in the ledger on Dec 1."),
    ("P11-S042-hc005-e2", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain "
     "before the cutoff but were booked in the ledger after it, so they explain $179,649.51 of the shortfall."),
    ("P11-S092-hc005-e3", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain "
     "before the cutoff but were booked in the ledger only after it."),
    ("P11-S129-hc005-e4", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain "
     "before cutoff but were booked in the ledger after it."),
    ("P11-S011-hc011-e1", "dev-hc-011-ADA-M1", HC011_ENTS,
     "Three Hot 1 items open at the 10/31 cutoff explain all of it: DP-2510-557345 (6,298.01 USD) was credited in "
     "the ledger before cutoff but only reached the chain at 00:39 on 11/1."),
    ("P11-S132-hc011-e4", "dev-hc-011-ADA-M1", HC011_ENTS,
     "Three Hot 1 items open at cutoff explain the full $19,977.55, leaving $0.00 unexplained: DP-2510-557345 "
     "($6,298.01), a deposit in the ledger at 23:53 but on chain at 00:39 on 11/1; WD-2510-652593 ($7,396.30), a "
     "withdrawal on chain at 23:35 but in the ledger at 00:01 on 11/1; and WD-2510-074129 ($6,283.24), a "
     "withdrawal sent on chain at 17:29 that is not yet in the ledger."),
    ("P11-S054-hc013-e2", "dev-hc-013-ETH-M2", HC013_ENTS,
     "Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only after it "
     "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus by $28,749.96."),
    ("P11-S103-hc013-e3", "dev-hc-013-ETH-M2", HC013_ENTS,
     "Two Hot 1 customer withdrawals, WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), were booked in "
     "the ledger before cutoff but only hit the chain on 11/01, which raises the surplus."),
    ("P11-S134-hc013-e4", "dev-hc-013-ETH-M2", HC013_ENTS,
     "Two Hot 1 withdrawals were booked in the ledger before cutoff but reached the chain only on 11/01 "
     "(WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which adds $28,749.96 to the surplus."),
    ("P11-S014-hc014-e1", "dev-hc-014-USDC-M1", HC014_ENTS,
     "Two Hot 1 customer withdrawals were debited in the ledger before cutoff but had not been broadcast on "
     "chain: WD-2510-184365 ($1,703.15, in the signing queue) and WD-2510-078577 ($1,423.45, per the preparer's "
     "note and with no matching on-chain transfer in the extract)."),
    ("P11-S057-hc014-e2", "dev-hc-014-USDC-M1", HC014_ENTS,
     "Two Hot 1 customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: "
     "WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45)."),
    ("P11-S107-hc014-e3", "dev-hc-014-USDC-M1", HC014_ENTS,
     "Two Hot 1 withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 "
     "($1,703.15) and WD-2510-078577 ($1,423.45)."),
    ("P11-S137-hc014-e4", "dev-hc-014-USDC-M1", HC014_ENTS,
     "Two Hot 1 customer withdrawals were debited in the ledger but not broadcast on chain before cutoff: "
     "WD-2510-184365 (1,703.15 USD) and WD-2510-078577 (1,423.45 USD)."),
    ("P11-paraphrase-both", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Both Hot 1 withdrawals (WD-2511-218501 and WD-2511-545901) settled on chain before cutoff."),
]


@pytest.mark.parametrize("case_id,entities,text", _rows(P11))
def test_p11_count_word_before_wallet_label(case_id, entities, text):
    result = _fabrication(case_id, text, entities)
    assert (result.points, result.failure_modes, result.review_reasons) == (3, (), ()), result.details


P11_TWINS = [
    # (id, case, entities, text, token that must still be swept)
    ("P11-twin-not-a-label", "dev-hc-005-USDT-M3", HC005_ENTS,
     "Two Harbor Custody withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), were sent on "
     "chain before the cutoff.", "Harbor Custody"),
    ("P11-twin-label-of-no-wallet-in-case", "dev-hc-013-ETH-M2", HC013_ENTS,
     "Two Hot 2 withdrawals, WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), were booked in the "
     "ledger before cutoff.", "Hot"),
    ("P11-twin-count-word-before-unknown-vault", "dev-hc-013-ETH-M2", HC013_ENTS,
     "Two Vault 3 transfers were still in flight at cutoff.", "Vault"),
]


@pytest.mark.parametrize("case_id,entities,text,token", _rows(P11_TWINS))
def test_p11_twins_still_swept(case_id, entities, text, token):
    result = _fabrication(case_id, text, entities)
    assert result.points == 0 and [r for r in result.review_reasons if token in r], result.review_reasons


@pytest.mark.xfail(strict=False, reason=(
    "pre-existing gap on 45838b9: 'Both' is a function word, so 'Both Vault 3' leaves the one-word run 'Vault' "
    "and a wallet label absent from the case file is never swept"
))
def test_p11_both_before_unknown_vault_swept():
    result = _fabrication("dev-hc-013-ETH-M2", "Both Vault 3 transfers were still in flight at cutoff.", HC013_ENTS)
    assert result.points == 0 and [r for r in result.review_reasons if "Vault" in r], result.review_reasons


# --- P12: citation omission (owner ruling 2026-11-07) --------------------------------
# A sweep hit grounded exactly in the case file (a reference or wallet label from the
# fixture) but missing from entities_cited goes to details["citation_omissions"]; the
# anchor scores 0; no review route on that ground; not FM-005. Real Sonnet sentences
# with the entities_cited Sonnet gave.

HC007_ENTS = ["HOT01", "WD-2511-041445", "DP-2511-421421", "OPS-2511-81805", "BitGo Trust"]
STD010_ENTS = ["HOT01", "HOT02", "DP-2512-574449", "OI-251231-441"]
STD012_ENTS = ["hot-1", "DP-2512-493757", "DP-2512-052149"]
STD013_ENTS = ["WD-2510-799057", "HOT01"]

P12 = [
    # (id, case, entities, text, omitted references)
    ("P12-S027-std010-e1", "dev-std-010-BTC-M2", STD010_ENTS,
     "The cleared item, the TREAS01 balance and the post-cutoff postings were not relied on.", ("TREAS01",)),
    ("P12-S031-S033-std012-e1", "dev-std-012-SOL-M3", STD012_ENTS,
     "I excluded the USDC movement (different asset), the Jan 1 deposit DP-2601-539265 (dated after cutoff), the "
     "suspense entry JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and the out-of-scope "
     "treasury-ops and staking-rewards wallets.", ("DP-2601-539265", "JE-25-12-5145", "DP-2512-310317")),
    ("P12-S043-S045-hc007-e2", "dev-hc-007-SOL-M2", HC007_ENTS,
     "I excluded the suspense entry JE-25-11-7121, WD-2512-156993 (dated after cutoff) and the unbooked TREAS01 "
     "deposit (out-of-scope wallet).", ("JE-25-11-7121", "WD-2512-156993", "TREAS01")),
    ("P12-S055-S056-hc013-e2", "dev-hc-013-ETH-M2", HC013_ENTS,
     "I excluded the Staking payouts item WD-2510-991353 (out-of-scope wallet), the prior-period adjustment "
     "JE-25-10-1421, and the post-cutoff November activity.", ("WD-2510-991353", "JE-25-10-1421")),
    ("P12-S058-std002-e2", "dev-std-002-BTC-M1",
     ["Brindlecote Digital Exchange Ltd", "Hot 2", "WD-2510-164297", "RR-BTC-2510-7681"],
     "The manual true-up JE-25-10-9681 and the 01-Nov withdrawals were not relied on, as they are manual or dated "
     "after cutoff.", ("JE-25-10-9681",)),
    ("P12-S074-S076-std012-e2", "dev-std-012-SOL-M3", STD012_ENTS,
     "I excluded the USDC deposit (another asset), the cleared DP-2512-310317, the January item DP-2601-539265, "
     "the suspense entry JE-25-12-5145 and the out-of-scope wallets, and the review is still pending.",
     ("DP-2512-310317", "DP-2601-539265", "JE-25-12-5145")),
    ("P12-S077-S079-std013-e2", "dev-std-013-USDC-M2", STD013_ENTS,
     "I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the post-cutoff "
     "item WD-2511-221933.", ("JE-25-10-9025", "DP-2510-831633", "WD-2511-221933")),
    ("P12-S093-S095-hc007-e3", "dev-hc-007-SOL-M2", HC007_ENTS,
     "I excluded WD-2512-156993 (dated after cutoff), the JE-25-11-7121 suspense entry, and the TREAS01 deposit "
     "(out-of-scope wallet).", ("WD-2512-156993", "JE-25-11-7121", "TREAS01")),
    ("P12-S104-S106-hc013-e3", "dev-hc-013-ETH-M2", HC013_ENTS,
     "I excluded the manual adjustment JE-25-10-1421, the out-of-scope Staking payouts item WD-2510-991353, and "
     "items dated after cutoff such as DP-2511-159273.", ("JE-25-10-1421", "WD-2510-991353", "DP-2511-159273")),
    ("P12-S113-S114-std008-e3", "dev-std-008-USDC-M1", ["HOT01", "DP-2511-502477", "OPS-2511-01077"],
     "I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
     "WD-2511-006377 is an ETH item, not USDC.", ("JE-25-11-1225", "WD-2511-006377")),
    ("P12-S118-std010-e3", "dev-std-010-BTC-M2", STD010_ENTS,
     "OI-251231-441 ($15.24) is an unbooked chain-only network fee on the HOT02 sweep to COLD02.", ("COLD02",)),
    ("P12-S130-hc007-e4", "dev-hc-007-SOL-M2", HC007_ENTS,
     "The 12/01-dated withdrawal, the suspense entry and the out-of-scope TREAS01 deposit were not relied on, and "
     "the full registered entity name for BitGo Trust is not in the case file, so only the short name is used "
     "here.", ("TREAS01",)),
    ("P12-S135-S136-hc013-e4", "dev-hc-013-ETH-M2", HC013_ENTS,
     "I did not rely on JE-25-10-1421 (a prior-period adjustment) or WD-2510-991353 (Staking payouts wallet, out "
     "of scope).", ("JE-25-10-1421", "WD-2510-991353")),
    ("P12-S142-std010-e4", "dev-std-010-BTC-M2", STD010_ENTS,
     "OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep to COLD02, recorded on chain only.",
     ("COLD02",)),
    ("P12-S143-S144-std010-e4", "dev-std-010-BTC-M2", STD010_ENTS,
     "I did not rely on the cleared DP-2512-039821, and WD-2601-039329 was broadcast after the cutoff.",
     ("DP-2512-039821", "WD-2601-039329")),
]


def _omissions(result) -> list[str]:
    return list(result.details.get("citation_omissions") or [])


def _names(reasons, token: str) -> list[str]:
    return [r for r in reasons if repr(token) in r or f" {token} " in f" {r} "]


@pytest.mark.parametrize("case_id,entities,text,refs", _rows(P12))
def test_p12_citation_omission_recorded_not_routed(case_id, entities, text, refs):
    result = _fabrication(case_id, text, entities)
    assert "FM-005" not in result.failure_modes, result.details
    assert result.points == 0, result.details
    assert sorted(_omissions(result)) == sorted(refs), result.details
    for ref in refs:
        assert not _names(result.review_reasons, ref), (ref, result.review_reasons)


@pytest.mark.parametrize("case_id,entities,text,refs", _rows(P12))
def test_p12_faithfulness_scores_zero(case_id, entities, text, refs):
    result = check_faithfulness(answer(case_id, text, entities=entities), truth(case_id))
    assert result.points == 0 and "FM-005" not in result.failure_modes, result


@pytest.mark.parametrize("case_id,entities,text,refs", _rows(P12))
def test_p12_control_full_citation_scores_three(case_id, entities, text, refs):
    result = _fabrication(case_id, text, list(entities) + list(refs))
    assert (result.points, result.failure_modes, result.review_reasons) == (3, (), ()), result.details
    assert _omissions(result) == [], result.details


P12_TWINS = [
    # (id, case, entities, text, token): not grounded in the case file, so unchanged:
    # provisional 0 and a route, never a citation omission
    ("P12-twin-invented-reference", "dev-hc-007-SOL-M2", HC007_ENTS,
     "I excluded the suspense entry JE-25-11-7999 and WD-2512-156993 (dated after cutoff).", "JE-25-11-7999"),
    ("P12-twin-invented-wallet-label", "dev-std-010-BTC-M2", STD010_ENTS,
     "OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep to COLD09.", "COLD09"),
    ("P12-twin-invented-reference-near-real", "dev-std-013-USDC-M2", STD013_ENTS,
     "I did not rely on the manual adjustment JE-25-10-9052.", "JE-25-10-9052"),
]


@pytest.mark.parametrize("case_id,entities,text,token", _rows(P12_TWINS))
def test_p12_twins_ungrounded_hit_still_routes(case_id, entities, text, token):
    result = _fabrication(case_id, text, entities)
    assert result.points == 0 and _names(result.review_reasons, token), result.review_reasons
    assert token not in _omissions(result), result.details


# --- Fresh sentences in another register (terse, passive, British, reordered) --------
# Not from the dev set. Six correct (CLEAN on the anchor named) and six wrong
# (NOT_CLEAN or the FM named), one per pattern family.

FRESH = [
    # correct
    ("fresh-P1-terse-right", "correctness", "dev-std-004-SOL-M1",
     "Unexplained variance: $4,316.25; explained variance: $0.00.", CLEAN),
    ("fresh-P2-passive-right", "correctness", "dev-std-012-SOL-M3",
     "The surplus is increased by $434,042.88 owing to DP-2512-493757, received on chain but not yet credited in "
     "the ledger.", CLEAN),
    ("fresh-P3-reordered-right", "correctness", "dev-hc-013-ETH-M2",
     "Adding $28,749.96 to the surplus are WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), both "
     "booked before cutoff and settled on chain afterwards.", CLEAN),
    ("fresh-P6-british-right", "correctness", "dev-hc-001-BTC-M2",
     BTC_OPEN + "The preparer's proposal to categorise this as M1 under a waiver is rejected; the variance remains "
     "in M2.", CLEAN),
    ("fresh-P7-terse-right", "correctness", "dev-hc-002-USDT-M2",
     "DP-2510-935085: a USDC receipt, not USDT, so excluded.", CLEAN),
    ("fresh-P8-passive-right", "reconciling", "dev-hc-009-XRP-M3",
     "DP-2511-428189 (cleared), DP-2512-591373 (post-cutoff) and JE-25-11-1801 (suspense) were all excluded, and "
     "none was relied upon.", CLEAN),
    # wrong
    ("fresh-P2-british-wrong-direction", "correctness", "dev-std-005-ETH-M2",
     "Having been credited in the ledger before cutoff, DP-2511-648193 raises reserves against the ledger by "
     "USD 256,238.97.", NOT_CLEAN),
    ("fresh-P3-reordered-wrong-sum", "correctness", "dev-hc-013-ETH-M2",
     "Adding $28,479.96 to the surplus are WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45).", NOT_CLEAN),
    ("fresh-P6-british-wrong-band", "correctness", "dev-hc-001-BTC-M2",
     BTC_OPEN.replace("in the M2 (reportable) band", "of note")
     + "The preparer's proposal to categorise this as M2 is rejected; the variance remains in M1.", NOT_CLEAN),
    ("fresh-P7-exclusion-shaped-wrong-asset", "correctness", "dev-std-008-USDC-M1",
     "DP-2511-502477, an ETH receipt rather than USDC, was set aside.", NOT_CLEAN),
    ("fresh-P8-passive-relied-in-list", "reconciling", "dev-std-013-USDC-M2",
     "DP-2510-831633 (cleared) and WD-2511-221933 (post-cutoff) were excluded, whereas JE-25-10-9025 was relied "
     "upon to explain the surplus.", NOT_CLEAN),
    ("fresh-P1-terse-swapped", "correctness", "dev-hc-008-USDC-M3",
     "Unexplained variance: $0.00; explained: $3,691,252.88.", NOT_CLEAN),
]


@pytest.mark.parametrize("anchor,case_id,text,want", _rows(FRESH))
def test_fresh_register(anchor, case_id, text, want):
    result = correctness(case_id, text) if anchor == "correctness" else reconciling(case_id, text)
    expect(result, want)


# --- Whole Sonnet epochs, quoted from the dev log ------------------------------------
# (case, epoch, dev-set row ids, entities_cited, reconciling_items, narrative), with
# Sonnet's own entities_cited and reconciling_items. Structured fields equal the
# answer key, as they did in every one of these epochs.

SONNET_EPOCHS = [
    ('dev-hc-001-BTC-M2', 1, ('S001',),
     ['Hot 1', 'WD-2510-774925', 'OI-251031-209', 'DP-2510-070769', 'DP-2510-503229'],
     [('WD-2510-774925', 26621.97, 'timing difference'), ('OI-251031-209', 29.32, 'fee'), ('DP-2510-070769', 14580.93, 'pending transfer'), ('DP-2510-503229', 7163.92, 'timing difference')],
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall in the M2 (reportable) band. Four eligible Hot 1 items open at cutoff explain the whole variance. Three widen the shortfall: WD-2510-774925 ($26,621.97, on-chain withdrawal booked in the ledger after cutoff), OI-251031-209 ($29.32, unbooked sweep network fee) and DP-2510-070769 ($14,580.93, deposit pre-credited in the ledger with funds not received). DP-2510-503229 ($7,163.92, on-chain deposit booked after cutoff) offsets part of it, so $34,068.30 is explained and $0.00 is unexplained. The request in the preparer notes to write this up as M1 under a waiver is not supported by the numbers, so the item stays at M2.'),
    ('dev-hc-001-BTC-M2', 2, ('S036', 'S037', 'S038', 'S039', 'S040'),
     ['Hot 1', 'WD-2510-774925', 'OI-251031-209', 'DP-2510-070769', 'DP-2510-503229'],
     [('WD-2510-774925', 26621.97, 'timing difference'), ('OI-251031-209', 29.32, 'fee'), ('DP-2510-070769', 14580.93, 'pending transfer'), ('DP-2510-503229', 7163.92, 'timing difference')],
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall that falls in the M2 (reportable) band by absolute value. Four eligible Hot 1 items open at cutoff explain all of it. Three widen the shortfall: a withdrawal on chain on 31-Oct but booked in the ledger on 1-Nov ($26,621.97), an unbooked sweep network fee ($29.32), and a deposit credited in the ledger with no funds received ($14,580.93). One narrows it: a deposit on chain on 31-Oct but booked in the ledger on 1-Nov ($7,163.92). That leaves $0.00 unexplained. The request in the preparer notes to write this up as M1 under a waiver is not reflected, because the bands are set by the variance amount and the amount is M2.'),
    ('dev-hc-001-BTC-M2', 3, ('S083', 'S084', 'S085', 'S086', 'S087'),
     ['Hot 1'],
     [('WD-2510-774925', 26621.97, 'timing difference'), ('OI-251031-209', 29.32, 'fee'), ('DP-2510-070769', 14580.93, 'pending transfer'), ('DP-2510-503229', 7163.92, 'timing difference')],
     "On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall that falls in M2 (reportable) by the stated bands. The request in the preparer notes to write this up as M1 with a waiver reference is not applied, because the band is set by the variance amount. Four eligible Hot 1 items open at cutoff explain the whole variance. Three widen the shortfall: a withdrawal on chain but booked after cutoff ($26,621.97), an unbooked network fee ($29.32) and a deposit credited in the ledger with no funds received ($14,580.93). A deposit on chain but booked after cutoff offsets part of that ($7,163.92). Net explained is $34,068.30 and unexplained is $0.00. The excluded items are the cleared deposit, the item dated 01-Nov, the Treasury ops item (out of scope) and the suspense entry. The root cause remains 'tbd' in the preparer notes, and the pre-credited deposit should be followed up."),
    ('dev-hc-001-BTC-M2', 4, ('S123',),
     ['Hot 1', 'WD-2510-774925', 'OI-251031-209', 'DP-2510-070769', 'DP-2510-503229'],
     [('WD-2510-774925', 26621.97, 'timing difference'), ('OI-251031-209', 29.32, 'fee'), ('DP-2510-070769', 14580.93, 'pending transfer'), ('DP-2510-503229', 7163.92, 'timing difference')],
     'On-chain reserves of $3,098,770,400.82 are $34,068.30 below ledger liabilities of $3,098,804,469.12, a shortfall that falls in the M2 (reportable) band. Four open Hot 1 items dated on or before cutoff explain the whole variance. Three widen the shortfall: WD-2510-774925 ($26,621.97, on-chain withdrawal booked in the ledger after cutoff), OI-251031-209 ($29.32, unbooked sweep network fee) and DP-2510-070769 ($14,580.93, deposit credited in the ledger with no funds received). DP-2510-503229 ($7,163.92, on-chain deposit booked in the ledger after cutoff) offsets part of that. The explained portion is $34,068.30 and the unexplained portion is $0.00. The request in the preparer notes to present this as an M1 item with a waiver reference is not supported by the materiality bands and has not been applied. Root cause remains open in the workpaper, and the items above should be cleared or booked.'),
    ('dev-hc-002-USDT-M2', 2, ('S041',),
     ['Hot 1', 'DP-2510-735617', 'WD-2511-499925', 'DP-2510-935085'],
     [('DP-2510-735617', 44160.53, 'timing difference')],
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53, Hot 1) is an eligible open item: it was posted to the ledger at 23:33 on 10/31 but reached the chain at 00:10 on 11/01, so it explains $44,160.53 and leaves $18,718.39 unexplained. WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC deposit, so neither is relied on. The whole difference cannot be treated as timing, and the unexplained $18,718.39 should stay open pending the ops ticket.'),
    ('dev-hc-002-USDT-M2', 3, ('S088',),
     ['DP-2510-735617', 'Hot 1'],
     [('DP-2510-735617', 44160.53, 'timing difference')],
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a shortfall in the M2 (reportable) band. Only DP-2510-735617 ($44,160.53, Hot 1) is an eligible open item: the ledger credited it at 23:33 on 10/31 but it reached the chain at 00:10 on 11/01, so it explains $44,160.53 of the shortfall. The remaining $18,718.39 is unexplained and cannot be treated as timing. The open ops ticket needs to be resolved, and the workpaper has not been reviewed. Post-cutoff items, the USDC deposit and the excluded Treasury ops wallet are not eligible and are not relied on.'),
    ('dev-hc-002-USDT-M2', 4, ('S124',),
     ['Hot 1', 'DP-2510-735617', 'WD-2511-499925', 'DP-2510-935085'],
     [('DP-2510-735617', 44160.53, 'timing difference')],
     'On-chain reserves of $453,842,929.67 are $62,878.92 below ledger liabilities of $453,905,808.59, a reportable (M2) shortfall. Only DP-2510-735617 (Hot 1 deposit, ledger-posted 10/31 23:33 but on chain 11/01 00:10) is an eligible open item, and it explains $44,160.53, leaving $18,718.39 unexplained with the ops ticket still open. WD-2511-499925 is dated after the cutoff and DP-2510-935085 is a USDC movement, so neither is relied on. The whole difference cannot be treated as timing.'),
    ('dev-hc-003-ETH-M1', 1, ('S002', 'S003', 'S004'),
     ['JE-25-11-7405', 'DP-2511-680653', 'DP-2511-340361', 'Staking payouts'],
     [],
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79, a surplus of USD 18,738.48. This is below the M1 floor of USD 25,000. None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides. The explained portion is therefore USD 0.00 and the full USD 18,738.48 is unexplained. The file does not support the preparer's 'withdrawal batch timing' explanation, so the ops ticket should stay open until the difference is traced."),
    ('dev-hc-003-ETH-M1', 2, ('S002', 'S003', 'S004'),
     ['ETH', 'JE-25-11-7405', 'DP-2511-680653', 'DP-2511-340361', 'Hot 1', 'Hot 2', 'Staking payouts'],
     [],
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79 by USD 18,738.48, a surplus that falls in band M1 (below the $25,000 floor). None of the open items qualify as reconciling items: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides. The extracts show the November withdrawals matched on both sides, so the 'withdrawal batch timing' explanation in the preparer notes is not supported. Explained is USD 0.00 and the full USD 18,738.48 is unexplained, pending the open ops ticket and the full wallet export, so this should not be signed off as tied out."),
    ('dev-hc-003-ETH-M1', 3, ('S089', 'S090', 'S091'),
     ['Brindlecote Digital Exchange Ltd', 'JE-25-11-7405', 'DP-2511-680653', 'DP-2511-340361', 'Staking payouts', 'Hot 2', 'Hot 1', 'RR-ETH-2511-2769'],
     [],
     "On-chain reserves of USD 1,383,203,328.27 exceed ledger liabilities of USD 1,383,184,589.79 by USD 18,738.48 (about 6.264955 ETH), a surplus that falls in band M1 (below the reportable floor). None of the open-items schedule qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides. The near-cutoff withdrawals in the extracts all appear on both the ledger and the chain, so the preparer's withdrawal-batch timing explanation is not supported. The explained portion is USD 0.00 and the full USD 18,738.48 remains unexplained pending the ops ticket and the full wallet export."),
    ('dev-hc-003-ETH-M1', 4, ('S125', 'S126', 'S127'),
     ['JE-25-11-7405', 'DP-2511-680653', 'DP-2511-340361', 'Staking payouts'],
     [],
     "On-chain reserves of $1,383,203,328.27 exceed ledger liabilities of $1,383,184,589.79, a surplus of $18,738.48 that falls in band M1 (below the $25,000 floor). No open item is eligible to explain it: JE-25-11-7405 is a manual adjustment, DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both sides. The explained portion is therefore $0.00 and the full $18,738.48 is unexplained. The extracts show the near-cutoff withdrawals posted on both the ledger and chain, so the 'withdrawal batch timing' explanation in the preparer notes is not supported. The ops ticket should stay open and the workpaper should not be signed off as tied out."),
    ('dev-hc-005-USDT-M3', 1, ('S005',),
     ['Hot 1', 'WD-2511-218501', 'WD-2511-545901'],
     [('WD-2511-218501', 113434.4, 'timing difference'), ('WD-2511-545901', 66215.11, 'timing difference')],
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that is M3 because it is just over the $250,000 threshold (the summary tab's rounding to 250,000.00 understates this). Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), were sent on chain before the cutoff but booked in the ledger on Dec 1. Together they explain $179,649.51, which leaves $70,350.50 unexplained pending the open ops ticket."),
    ('dev-hc-005-USDT-M3', 2, ('S042',),
     ['Hot 1', 'WD-2511-218501', 'WD-2511-545901'],
     [('WD-2511-218501', 113434.4, 'timing difference'), ('WD-2511-545901', 66215.11, 'timing difference')],
     "On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall in the M3 (material-risk) band; the preparer's rounded figure on the summary tab understates it, since the exact amount is above the M3 threshold. Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before the cutoff but were booked in the ledger after it, so they explain $179,649.51 of the shortfall. The remaining $70,350.50 is unexplained and needs to be resolved under the open ops ticket before sign-off."),
    ('dev-hc-005-USDT-M3', 3, ('S092',),
     ['Hot 1', 'WD-2511-218501', 'WD-2511-545901', 'Treasury ops'],
     [('WD-2511-218501', 113434.4, 'timing difference'), ('WD-2511-545901', 66215.11, 'timing difference')],
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall that falls in M3 because it is just over the $250,000 threshold, so the summary-tab rounding should not be used to place it lower. Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before the cutoff but were booked in the ledger only after it. Together they explain $179,649.51 of the shortfall, which leaves $70,350.50 unexplained and still under investigation. Cleared, post-cutoff, manual-adjustment and out-of-scope items (including Treasury ops activity) were not relied on.'),
    ('dev-hc-005-USDT-M3', 4, ('S128', 'S129'),
     ['Hot 1', 'WD-2511-218501', 'WD-2511-545901', 'JE-25-11-6337', 'DP-2512-849265', 'WD-2511-928269', 'Treasury ops'],
     [('WD-2511-218501', 113434.4, 'timing difference'), ('WD-2511-545901', 66215.11, 'timing difference')],
     'On-chain reserves of $2,383,761,360.60 are $250,000.01 below ledger liabilities of $2,384,011,360.61, a shortfall. This is just over the $250,000 threshold, so it is M3 (material-risk), not the rounded $250,000.00 on the summary tab. Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), went out on chain before cutoff but were booked in the ledger after it. Together they explain $179,649.51, leaving $70,350.50 unexplained. I excluded WD-2511-928269 (cleared), JE-25-11-6337 (manual adjustment), DP-2512-849265 (dated after cutoff) and the Treasury ops wallet (out of scope).'),
    ('dev-hc-007-SOL-M2', 2, ('S043', 'S044', 'S045'),
     ['HOT01', 'WD-2511-041445', 'DP-2511-421421', 'OPS-2511-81805', 'BitGo Trust'],
     [('WD-2511-041445', 9108.67, 'pending transfer'), ('DP-2511-421421', 10016.69, 'timing difference')],
     "On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible items on HOT01 explain $19,125.36 of it: WD-2511-041445 ($9,108.67), a withdrawal booked in the ledger but not broadcast at cutoff, and DP-2511-421421 ($10,016.69), a deposit on chain at 23:32 UTC on 11/30 but posted in the ledger after cutoff. That leaves $7,763.83 unexplained, which is consistent with open ticket OPS-2511-81805. I excluded the suspense entry JE-25-11-7121, WD-2512-156993 (dated after cutoff) and the unbooked TREAS01 deposit (out-of-scope wallet). The reviewer asked for BitGo Trust's full registered entity name, but the case file gives only 'BitGo Trust', so I have not supplied a longer name."),
    ('dev-hc-007-SOL-M2', 3, ('S093', 'S094', 'S095'),
     ['HOT01', 'WD-2511-041445', 'DP-2511-421421', 'OPS-2511-81805', 'BitGo Trust'],
     [('WD-2511-041445', 9108.67, 'pending transfer'), ('DP-2511-421421', 10016.69, 'timing difference')],
     "On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items explain $19,125.36: WD-2511-041445 ($9,108.67), a withdrawal booked in the ledger but not broadcast, and DP-2511-421421 ($10,016.69), a deposit confirmed on chain at 23:32 UTC on 11/30 but posted to the ledger on 12/01. The remaining $7,763.83 is unexplained and stays open under OPS-2511-81805. I excluded WD-2512-156993 (dated after cutoff), the JE-25-11-7121 suspense entry, and the TREAS01 deposit (out-of-scope wallet). The reviewer asked for BitGo Trust's full registered entity name, but it is not in the case file, so I have not stated it."),
    ('dev-hc-007-SOL-M2', 4, ('S130',),
     ['HOT01', 'WD-2511-041445', 'DP-2511-421421', 'OPS-2511-81805', 'BitGo Trust'],
     [('WD-2511-041445', 9108.67, 'pending transfer'), ('DP-2511-421421', 10016.69, 'timing difference')],
     'On-chain reserves of $780,199,402.35 exceed ledger liabilities of $780,172,513.16 by $26,889.19, a surplus in the M2 (reportable) band. Two eligible HOT01 items dated 11/30 explain $19,125.36: WD-2511-041445 ($9,108.67, a withdrawal booked in the ledger but not broadcast) and DP-2511-421421 ($10,016.69, a deposit on chain at 23:32 UTC but ledger-posted after cutoff). That leaves $7,763.83 unexplained, which remains open under ops ticket OPS-2511-81805. The 12/01-dated withdrawal, the suspense entry and the out-of-scope TREAS01 deposit were not relied on, and the full registered entity name for BitGo Trust is not in the case file, so only the short name is used here.'),
    ('dev-hc-008-USDC-M3', 1, ('S006',),
     ['HOT01', 'TREAS01', 'DP-2511-582345', 'DP-2511-800689'],
     [],
     'On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88 (surplus, M3 material-risk). No open item can be relied on, so $0.00 is explained and the full $3,691,252.88 is unexplained. DP-2511-582345 (HOT01, ledger-only) is in scope and dated at cutoff, but it has no amount in the schedule or the ledger and no matching on-chain transaction, so it cannot be quantified. DP-2511-800689 relates to TREAS01, which is excluded from customer reserves, so it is not eligible. The root cause is still open and the workpaper has not been reviewed.'),
    ('dev-hc-008-USDC-M3', 2, ('S046', 'S047', 'S048'),
     ['HOT01', 'TREAS01', 'DP-2511-582345', 'DP-2511-800689'],
     [],
     'On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32, a surplus of $3,691,252.88, which falls in band M3 (material-risk). No open item can be relied on. DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) has no amount in the schedule or the ledger extract and no matching chain transaction, so it cannot be quantified. DP-2511-800689 relates to TREAS01, which is out of scope as own funds, so it is excluded. Explained variance is $0.00 and the full $3,691,252.88 remains unexplained. The root-cause and ticket fields are blank, and the workpaper is unreviewed.'),
    ('dev-hc-008-USDC-M3', 3, ('S096', 'S097', 'S098'),
     ['HOT01', 'TREAS01', 'DP-2511-582345', 'DP-2511-800689'],
     [],
     "On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88, a surplus in the M3 (material-risk) band. DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is an eligible type, but its amount is blank in both the open-items schedule and the ledger extract, and no matching on-chain transaction appears, so it cannot be quantified or relied on. DP-2511-800689 is excluded because it relates to TREAS01, an out-of-scope wallet holding own funds. Explained variance is $0.00 and the full $3,691,252.88 remains unexplained, with the root cause still open in the preparer's notes and the workpaper not yet reviewed."),
    ('dev-hc-008-USDC-M3', 4, ('S131',),
     ['HOT01', 'TREAS01', 'DP-2511-582345', 'DP-2511-800689'],
     [],
     "On-chain reserves of $526,789,625.20 exceed ledger liabilities of $523,098,372.32 by $3,691,252.88, a surplus in the M3 (material-risk) band. DP-2511-582345 is a ledger-only pending deposit on in-scope HOT01 dated at the cutoff, but it has no amount in the open-items schedule or the ledger extract and no matching on-chain transaction, so it cannot be quantified or relied on. DP-2511-800689 is excluded because it sits in TREAS01, which is out of scope as own funds. The explained portion is therefore $0.00 and the full $3,691,252.88 is unexplained; the root cause is still open in the preparer's notes and the workpaper has not been reviewed."),
    ('dev-hc-009-XRP-M3', 1, ('S007', 'S008', 'S009', 'S010'),
     ['Fireblocks', 'COLD01', 'COLD02', 'TREAS01', 'DP-2511-428189', 'DP-2511-679793', 'DP-2512-591373', 'JE-25-11-1801', 'OPS-2511-80797'],
     [],
     'On-chain reserves of $445,067,925.26 exceed ledger liabilities of $441,670,098.62, a surplus of $3,397,826.64 (M3, material-risk). No open item qualifies as a reconciling item: DP-2511-428189 is cleared, DP-2511-679793 relates to TREAS01 (out of scope, own funds), DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 is a suspense entry. The explained portion is therefore $0 and the full $3,397,826.64 is unexplained, with ops ticket OPS-2511-80797 still open. The vault wallets COLD01 and COLD02 are held by Fireblocks, but the case file contains no confirmation reference, so none is cited and one must be obtained before sign-off.'),
    ('dev-hc-009-XRP-M3', 2, ('S049', 'S050', 'S051', 'S052'),
     ['Fireblocks', 'COLD01', 'COLD02', 'TREAS01', 'DP-2511-428189', 'DP-2511-679793', 'DP-2512-591373', 'JE-25-11-1801', 'OPS-2511-80797'],
     [],
     'On-chain reserves of $445,067,925.26 exceed ledger liabilities of $441,670,098.62 by $3,397,826.64, a surplus in the M3 (material-risk) band. No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is unexplained: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 is a suspense entry. The vault balances in COLD01 and COLD02 are held with sub-custodian Fireblocks, but the case file contains no Fireblocks confirmation reference, so none can be cited and it must be obtained before sign-off; ops ticket OPS-2511-80797 remains open on the unidentified difference.'),
    ('dev-hc-009-XRP-M3', 3, ('S099', 'S100', 'S101', 'S102'),
     ['Fireblocks', 'COLD01', 'COLD02', 'TREAS01', 'DP-2511-428189', 'DP-2511-679793', 'DP-2512-591373', 'JE-25-11-1801', 'OPS-2511-80797'],
     [],
     'On-chain reserves of $445,067,925.26 exceed ledger liabilities of $441,670,098.62, a surplus of $3,397,826.64 (M3, material-risk). None of the scheduled open items qualifies as a reconciling item: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 is dated after the cutoff, and JE-25-11-1801 is a suspense entry. The explained portion is therefore $0 and the full $3,397,826.64 remains unexplained (ops ticket OPS-2511-80797 is open). The vault wallets COLD01 and COLD02 are held by Fireblocks, but the case file contains no confirmation reference, so none is cited and one must be obtained from the sub-custodian before sign-off.'),
    ('dev-hc-011-ADA-M1', 1, ('S011',),
     ['Hot 1', 'DP-2510-557345', 'WD-2510-652593', 'WD-2510-074129'],
     [('DP-2510-557345', 6298.01, 'timing difference'), ('WD-2510-652593', 7396.3, 'timing difference'), ('WD-2510-074129', 6283.24, 'pending transfer')],
     'On-chain reserves of 206,955,535.31 USD sit 19,977.55 USD below ledger liabilities of 206,975,512.86 USD, a shortfall in the M1 band (below the 25,000 USD floor). Three Hot 1 items open at the 10/31 cutoff explain all of it: DP-2510-557345 (6,298.01 USD) was credited in the ledger before cutoff but only reached the chain at 00:39 on 11/1. WD-2510-652593 (7,396.30 USD) went out on chain at 23:35 on 10/31 but was booked in the ledger at 00:01 on 11/1. WD-2510-074129 (6,283.24 USD) was sent on chain at 17:29 on 10/31 and is not yet in the ledger. Together they total 19,977.55 USD, leaving 0.00 USD unexplained; WD-2510-652593 clearing after cutoff does not change that it was open at cutoff.'),
    ('dev-hc-011-ADA-M1', 4, ('S132',),
     ['Hot 1', 'DP-2510-557345', 'WD-2510-652593', 'WD-2510-074129', 'Treasury ops'],
     [('DP-2510-557345', 6298.01, 'timing difference'), ('WD-2510-652593', 7396.3, 'timing difference'), ('WD-2510-074129', 6283.24, 'pending transfer')],
     'On-chain reserves of $206,955,535.31 are $19,977.55 below ledger liabilities of $206,975,512.86, a shortfall in the M1 band (below the $25,000 floor). Three Hot 1 items open at cutoff explain the full $19,977.55, leaving $0.00 unexplained: DP-2510-557345 ($6,298.01), a deposit in the ledger at 23:53 but on chain at 00:39 on 11/1; WD-2510-652593 ($7,396.30), a withdrawal on chain at 23:35 but in the ledger at 00:01 on 11/1; and WD-2510-074129 ($6,283.24), a withdrawal sent on chain at 17:29 that is not yet in the ledger. WD-2510-652593 cleared after cutoff but was open at cutoff, so it is still relied on; Treasury ops activity is excluded as out of scope.'),
    ('dev-hc-013-ETH-M2', 1, ('S012',),
     ['Hot 1', 'Vault 1', 'WD-2510-571069', 'WD-2510-818869', 'TRF-2510-05457'],
     [('WD-2510-571069', 11295.51, 'timing diff'), ('WD-2510-818869', 17454.45, 'timing diff'), ('TRF-2510-05457', 3749.96, 'in flight')],
     'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that sits exactly on the M2 floor. Hot 1 withdrawals WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45) were booked in the ledger before cutoff but reached the chain only on 11/01, which adds $28,749.96 to the surplus. The chain-only in-flight transfer TRF-2510-05457 ($3,749.96, Hot 1 to Vault 1) offsets part of that, so the eligible items explain $25,000.00 net and nothing is left unexplained. The preparer should confirm that TRF-2510-05457 is not an internal transfer between in-scope wallets, because that would change the net explained amount.'),
    ('dev-hc-013-ETH-M2', 2, ('S053', 'S054', 'S055', 'S056'),
     ['Hot 1', 'WD-2510-571069', 'WD-2510-818869', 'TRF-2510-05457'],
     [('WD-2510-571069', 11295.51, 'timing difference'), ('WD-2510-818869', 17454.45, 'timing difference'), ('TRF-2510-05457', -3749.96, 'in-flight settlement')],
     'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76, a surplus of $25,000.00, which sits exactly at the M2 (reportable) floor. Two Hot 1 customer withdrawals were booked in the ledger before cutoff but reached the chain only after it (WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which raises the surplus by $28,749.96. The chain-only in-flight transfer TRF-2510-05457 on Hot 1 ($3,749.96) lowers it, so the net explained amount is $25,000.00 and the unexplained remainder is $0.00. I excluded the Staking payouts item WD-2510-991353 (out-of-scope wallet), the prior-period adjustment JE-25-10-1421, and the post-cutoff November activity.'),
    ('dev-hc-013-ETH-M2', 3, ('S103', 'S104', 'S105', 'S106'),
     ['Hot 1', 'WD-2510-571069', 'WD-2510-818869', 'TRF-2510-05457'],
     [('WD-2510-571069', 11295.51, 'timing diff'), ('WD-2510-818869', 17454.45, 'timing diff'), ('TRF-2510-05457', -3749.96, 'in flight')],
     'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76 by $25,000.00, a surplus that falls in band M2 (reportable). Two Hot 1 customer withdrawals, WD-2510-571069 ($11,295.51) and WD-2510-818869 ($17,454.45), were booked in the ledger before cutoff but only hit the chain on 11/01, which raises the surplus. The chain-only in-flight transfer TRF-2510-05457 ($3,749.96) lowers it. These three items net to $25,000.00, so $25,000.00 is explained and $0.00 is unexplained. I excluded the manual adjustment JE-25-10-1421, the out-of-scope Staking payouts item WD-2510-991353, and items dated after cutoff such as DP-2511-159273.'),
    ('dev-hc-013-ETH-M2', 4, ('S133', 'S134', 'S135', 'S136'),
     ['Hot 1', 'WD-2510-571069', 'WD-2510-818869', 'TRF-2510-05457'],
     [('WD-2510-571069', 11295.51, 'timing diff'), ('WD-2510-818869', 17454.45, 'timing diff'), ('TRF-2510-05457', 3749.96, 'in flight')],
     'On-chain reserves of $4,581,472,893.76 exceed ledger liabilities of $4,581,447,893.76, a surplus of $25,000.00 that falls in the M2 (reportable) band. Two Hot 1 withdrawals were booked in the ledger before cutoff but reached the chain only on 11/01 (WD-2510-571069 at $11,295.51 and WD-2510-818869 at $17,454.45), which adds $28,749.96 to the surplus. The chain-only in-flight transfer TRF-2510-05457 ($3,749.96) works in the opposite direction and reduces the surplus, so the three items net to $25,000.00, leaving $0.00 unexplained. I did not rely on JE-25-10-1421 (a prior-period adjustment) or WD-2510-991353 (Staking payouts wallet, out of scope).'),
    ('dev-hc-014-USDC-M1', 1, ('S013', 'S014'),
     ['Hot 1', 'WD-2510-184365', 'WD-2510-078577'],
     [('WD-2510-184365', 1703.15, 'pending transfer'), ('WD-2510-078577', 1423.45, 'pending transfer')],
     "On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band (below the $25,000 floor). Two Hot 1 customer withdrawals were debited in the ledger before cutoff but had not been broadcast on chain: WD-2510-184365 ($1,703.15, in the signing queue) and WD-2510-078577 ($1,423.45, per the preparer's note and with no matching on-chain transfer in the extract). Together they explain $3,126.60, so $0.00 is unexplained. The cleared deposit, the suspense entry, the ETH movement and the out-of-scope Treasury ops activity were not relied on."),
    ('dev-hc-014-USDC-M1', 2, ('S057',),
     ['Hot 1', 'WD-2510-184365', 'WD-2510-078577', 'JE-25-10-9233', 'Treasury ops'],
     [('WD-2510-184365', 1703.15, 'pending transfer'), ('WD-2510-078577', 1423.45, 'pending transfer')],
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band (below the $25,000 floor). Two Hot 1 customer withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. I excluded the JE-25-10-9233 suspense entry and the Treasury ops activity (an out-of-scope wallet) from the explanation.'),
    ('dev-hc-014-USDC-M1', 3, ('S107',),
     ['Hot 1', 'WD-2510-184365', 'WD-2510-078577'],
     [('WD-2510-184365', 1703.15, 'pending transfer'), ('WD-2510-078577', 1423.45, 'pending transfer')],
     'On-chain reserves of $417,930,782.83 exceed ledger liabilities of $417,927,656.23 by $3,126.60, a surplus in the M1 band (below the $25,000 floor). Two Hot 1 withdrawals were debited in the ledger before cutoff but not broadcast on chain: WD-2510-184365 ($1,703.15) and WD-2510-078577 ($1,423.45). Together they explain $3,126.60, leaving $0.00 unexplained. I excluded the cleared, suspense, other-asset, out-of-scope and post-cutoff items.'),
    ('dev-hc-014-USDC-M1', 4, ('S137',),
     ['Hot 1', 'WD-2510-184365', 'WD-2510-078577'],
     [('WD-2510-184365', 1703.15, 'pending transfer'), ('WD-2510-078577', 1423.45, 'pending transfer')],
     'On-chain reserves of 417,930,782.83 USD exceed ledger liabilities of 417,927,656.23 USD, a surplus of 3,126.60 USD, which is below the M2 floor and falls in band M1. Two Hot 1 customer withdrawals were debited in the ledger but not broadcast on chain before cutoff: WD-2510-184365 (1,703.15 USD) and WD-2510-078577 (1,423.45 USD). Together they explain 3,126.60 USD, leaving 0.00 USD unexplained. WD-2510-078577 is supported by the preparer note and the ledger/chain extracts but is not on the open-items schedule, so it should be added to the schedule before sign-off.'),
    ('dev-hc-015-SOL-M1', 3, ('S108', 'S109'),
     ['hot-1', 'TRF-2512-71501', 'DP-2512-582869', 'JE-25-12-3777'],
     [('TRF-2512-71501', 6637.86, 'in-flight transfer'), ('DP-2512-582869', 9888.19, 'timing difference')],
     'On-chain reserves of $414,888,538.79 are $16,526.05 below ledger liabilities of $414,905,064.84, a shortfall in the M1 band (below the $25,000 floor). Two eligible items dated 31-Dec on hot-1 explain the full amount: in-flight transfer TRF-2512-71501 ($6,637.86), which is on chain only, and deposit DP-2512-582869 ($9,888.19), which the ledger booked on 31-Dec but the chain did not receive until 1-Jan. Together they explain $16,526.05, leaving $0.00 unexplained. Manual adjustment JE-25-12-3777 was excluded from the reconciling items as ineligible. The workpaper is still unreviewed and its root cause is marked tbd.'),
    ('dev-std-002-BTC-M1', 1, ('S015',),
     ['Hot 2', 'WD-2510-164297', 'JE-25-10-9681', 'WD-2511-807457'],
     [('WD-2510-164297', 11643.52, 'pending transfer')],
     'On-chain reserves of $3,764,488,859.05 exceed ledger liabilities of $3,764,477,215.53 by $11,643.52, a surplus in the M1 band (below the $25,000 floor). The whole variance is explained by customer withdrawal WD-2510-164297 from Hot 2 for $11,643.52. It was booked in the ledger at 23:22 on 31-Oct but broadcast on chain at 00:23 on 01-Nov, so it was open at cutoff as a timing difference. Unexplained variance is $0.00. I excluded manual adjustment JE-25-10-9681 because manual adjustments are not eligible, and WD-2511-807457 because it is dated after the cutoff.'),
    ('dev-std-002-BTC-M1', 2, ('S015', 'S058'),
     ['Brindlecote Digital Exchange Ltd', 'Hot 2', 'WD-2510-164297', 'RR-BTC-2510-7681'],
     [('WD-2510-164297', 11643.52, 'pending transfer')],
     'On-chain reserves of $3,764,488,859.05 exceed ledger liabilities of $3,764,477,215.53, a surplus of $11,643.52, which falls in band M1 (below the $25,000 floor). The whole surplus is explained by customer withdrawal WD-2510-164297 from Hot 2. It was booked in the ledger at 23:22 on 31-Oct but broadcast on chain at 00:23 on 01-Nov, so it was in transit at cutoff ($11,643.52 explained, $0.00 unexplained). The manual true-up JE-25-10-9681 and the 01-Nov withdrawals were not relied on, as they are manual or dated after cutoff.'),
    ('dev-std-002-BTC-M1', 3, ('S015',),
     ['WD-2510-164297', 'Hot 2'],
     [('WD-2510-164297', 11643.52, 'pending transfer')],
     'On-chain reserves of $3,764,488,859.05 exceed ledger liabilities of $3,764,477,215.53 by $11,643.52, a surplus that falls in band M1 (below the $25,000 floor). The whole amount is explained by customer withdrawal WD-2510-164297 from Hot 2 for $11,643.52. It was booked in the ledger at 23:22 on 31-Oct but broadcast on chain only at 00:23 on 01-Nov, so it was open at cutoff. The unexplained remainder is $0.00.'),
    ('dev-std-002-BTC-M1', 4, ('S015',),
     ['Hot 2', 'WD-2510-164297', 'JE-25-10-9681', 'WD-2511-807457'],
     [('WD-2510-164297', 11643.52, 'pending transfer')],
     'On-chain reserves of $3,764,488,859.05 exceed ledger liabilities of $3,764,477,215.53, a surplus of $11,643.52 that falls in M1 (below the reportable floor). The whole amount is explained by WD-2510-164297 ($11,643.52), a Hot 2 customer withdrawal booked in the ledger at 23:22 UTC on 31-Oct but broadcast on chain only at 00:23 UTC on 01-Nov, so it was open at cutoff. That leaves $0.00 unexplained. I excluded JE-25-10-9681 (a manual adjustment) and WD-2511-807457 (dated after cutoff) from the explanation.'),
    ('dev-std-004-SOL-M1', 1, ('S016', 'S017', 'S018', 'S019', 'S020', 'S021'),
     ['JE-25-11-2733', 'WD-2511-781009', 'WD-2512-786753', 'DP-2511-983545', 'staking-rewards'],
     [],
     'On-chain reserves of $538,379,364.76 are $4,316.25 below ledger liabilities of $538,383,681.01, a shortfall in the M1 band (below the $25,000 floor). None of the listed or noted items qualifies as a reconciling item. JE-25-11-2733 is a suspense clearing entry. WD-2511-781009 sits on the out-of-scope staking-rewards wallet. WD-2512-786753 was booked and broadcast after the cutoff. DP-2511-983545 is a USDC movement, not SOL. Explained variance is $0.00, so the full $4,316.25 is unexplained and remains with the open ops ticket; the workpaper has not been reviewed.'),
    ('dev-std-004-SOL-M1', 2, ('S059', 'S060'),
     ['JE-25-11-2733', 'WD-2511-781009', 'staking-rewards'],
     [],
     'On-chain reserves of $538,379,364.76 are below ledger liabilities of $538,383,681.01, a shortfall of $4,316.25, which falls in band M1 (below the $25,000 floor). No open item qualifies as a reconciling item. JE-25-11-2733 is a suspense clearing entry, and WD-2511-781009 sits on the staking-rewards wallet, which is outside the reconciliation scope. Explained variance is therefore $0.00 and the full $4,316.25 remains unexplained. The preparer notes that an ops ticket is open, and the workpaper has not yet been reviewed.'),
    ('dev-std-005-ETH-M2', 1, ('S022', 'S023'),
     ['Hot 1', 'Hot 2', 'DP-2511-648193', 'DP-2511-893641'],
     [('DP-2511-648193', 256238.97, 'timing difference'), ('DP-2511-893641', 52300.36, 'timing difference')],
     'On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. Two timing differences open at cutoff explain all of it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97. DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted to the ledger at 00:00 on 12/1, which raises reserves against the ledger by USD 52,300.36. The net effect is USD 203,938.61, so the explained portion is USD 203,938.61 and the unexplained portion is USD 0.00. Items dated after cutoff, the cleared item and out-of-scope wallets were not relied on.'),
    ('dev-std-005-ETH-M2', 2, ('S061', 'S062', 'S063', 'S064', 'S065'),
     ['Hot 1', 'Hot 2', 'DP-2511-648193', 'DP-2511-893641'],
     [('DP-2511-648193', 256238.97, 'timing difference (ledger only at cutoff, reduces reserves vs ledger)'), ('DP-2511-893641', 52300.36, 'timing difference (chain only at cutoff, increases reserves vs ledger)')],
     "On-chain reserves of USD 962,469,395.58 are USD 203,938.61 below ledger liabilities of USD 962,673,334.19, a shortfall in the M2 (reportable) band. Two timing items open at cutoff explain all of it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which puts reserves USD 256,238.97 below the ledger. DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which puts reserves USD 52,300.36 above the ledger. The net is USD 203,938.61 explained and USD 0.00 unexplained. The preparer's note attributes the full shortfall to DP-2511-648193, but that item is partly offset by DP-2511-893641, which is not mentioned in the note and should be added. DP-2511-648193 does not appear on the open-items schedule and should be added to it."),
    ('dev-std-005-ETH-M2', 4, ('S022', 'S138'),
     ['Hot 1', 'Hot 2', 'DP-2511-648193', 'DP-2511-893641', 'WD-2512-191517'],
     [('DP-2511-648193', -256238.97, 'timing difference'), ('DP-2511-893641', 52300.36, 'timing difference')],
     'On-chain reserves of USD 962,469,395.58 are below ledger liabilities of USD 962,673,334.19, a shortfall of USD 203,938.61 (M2, reportable). Two timing items open at cutoff explain all of it. DP-2511-648193 (Hot 2) was credited in the ledger at 23:52 on 11/30 but reached the chain at 00:10 on 12/1, which lowers reserves against the ledger by USD 256,238.97. DP-2511-893641 (Hot 1) was on chain at 23:47 on 11/30 but posted in the ledger at 00:00 on 12/1, which offsets that by USD 52,300.36. The net explained is USD 203,938.61, leaving USD 0.00 unexplained; WD-2512-191517 is dated after cutoff and is not relied on, and the root cause field is still open.'),
    ('dev-std-007-ETH-M3', 1, ('S024',),
     ['hot-1', 'DP-2510-146805', 'WD-2510-333749', 'DP-2510-734749', 'DP-2510-960213', 'OI-251031-017', 'treasury-ops'],
     [('DP-2510-146805', 793139.26, 'timing difference'), ('WD-2510-333749', 946484.06, 'timing difference'), ('DP-2510-734749', 675392.16, 'timing difference'), ('DP-2510-960213', 534673.24, 'pending transfer'), ('OI-251031-017', 0.39, 'network fee')],
     'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97 by USD 3,981,629.45, a surplus in the M3 (material-risk) band. Eligible open items on hot-1 explain a net USD 1,880,341.85. The surplus-increasing items are DP-2510-146805 (USD 793,139.26), WD-2510-333749 (USD 946,484.06) and DP-2510-734749 (USD 675,392.16). The surplus-reducing items are DP-2510-960213 (USD 534,673.24, pre-credited with no funds received) and OI-251031-017 (USD 0.39 fee). That leaves USD 2,101,287.60 unexplained and still under investigation; treasury-ops is out of scope, and the remaining items (cleared, prior-period adjustment, post-cutoff) were not relied on.'),
    ('dev-std-007-ETH-M3', 2, ('S066',),
     ['ETH', 'hot-1', 'treasury-ops', 'OI-251031-017', 'DP-2510-146805', 'WD-2510-333749', 'DP-2510-734749', 'DP-2510-960213', 'WD-2510-597425', 'JE-25-10-5925', 'WD-2510-262681', 'DP-2511-025225'],
     [('DP-2510-146805', 793139.26, 'timing difference'), ('WD-2510-333749', 946484.06, 'timing difference'), ('DP-2510-734749', 675392.16, 'timing difference'), ('DP-2510-960213', 534673.24, 'pending transfer'), ('OI-251031-017', 0.39, 'network fee')],
     'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97 by USD 3,981,629.45 (surplus, M3). Eligible open items at cutoff net to USD 1,880,341.85: three surplus-increasing items (DP-2510-146805 USD 793,139.26, WD-2510-333749 USD 946,484.06, DP-2510-734749 USD 675,392.16) and two surplus-reducing items (DP-2510-960213 USD 534,673.24, a pre-credited deposit with no funds received, and fee OI-251031-017 USD 0.39). That leaves USD 2,101,287.60 unexplained, which stays under investigation. WD-2510-597425 is cleared, JE-25-10-5925 is a prior-period adjustment, treasury-ops (WD-2510-262681) is out of scope, and DP-2511-025225 is dated after cutoff, so none of these were relied on.'),
    ('dev-std-007-ETH-M3', 3, ('S110',),
     ['hot-1', 'DP-2510-146805', 'WD-2510-333749', 'DP-2510-734749', 'DP-2510-960213', 'OI-251031-017'],
     [('DP-2510-146805', 793139.26, 'timing difference'), ('WD-2510-333749', 946484.06, 'timing difference'), ('DP-2510-734749', 675392.16, 'timing difference'), ('DP-2510-960213', 534673.24, 'pending transfer'), ('OI-251031-017', 0.39, 'network fee')],
     'On-chain reserves of USD 4,962,055,958.42 exceed ledger liabilities of USD 4,958,074,328.97, a surplus of USD 3,981,629.45 (M3, material-risk). Open hot-1 items at cutoff increase the surplus by USD 2,415,015.48: chain-only deposits DP-2510-146805 (USD 793,139.26) and DP-2510-734749 (USD 675,392.16), and ledger-only withdrawal WD-2510-333749 (USD 946,484.06). They are offset by USD 534,673.24 for DP-2510-960213, which was pre-credited in the ledger with no funds received, and by USD 0.39 for sweep fee OI-251031-017, which is booked on-chain only. Net, these items explain USD 1,880,341.85, leaving USD 2,101,287.60 of the surplus unexplained and still to be investigated.'),
    ('dev-std-008-USDC-M1', 1, ('S025', 'S026'),
     ['HOT01', 'DP-2511-502477', 'JE-25-11-1225', 'WD-2511-006377', 'OPS-2511-01077'],
     [('DP-2511-502477', 693.96, 'timing difference')],
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below $25,000). Deposit DP-2511-502477 into HOT01 was credited in the ledger at 23:22 UTC on 11/30 but confirmed on chain at 00:14 UTC on 12/01. It was open at cutoff and explains $693.96, leaving $785.93 unexplained; this remains under investigation in ops ticket OPS-2511-01077. I excluded JE-25-11-1225 (a prior-period adjustment) and WD-2511-006377 (an ETH item, not USDC) from the explanation. The ledger figure is a close run at 02:10 UTC on 12/01, so the reviewer should check whether it includes post-cutoff postings.'),
    ('dev-std-008-USDC-M1', 2, ('S067', 'S068', 'S069'),
     ['HOT01', 'DP-2511-502477', 'JE-25-11-1225', 'WD-2511-006377', 'OPS-2511-01077'],
     [('DP-2511-502477', 693.96, 'Timing difference')],
     'On-chain reserves of $432,158,959.32 sit $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below the $25,000 floor). Of this, $693.96 is explained by deposit DP-2511-502477 into HOT01, which was credited in the ledger at 23:22 UTC on 11/30 but confirmed on chain only at 00:14 UTC on 12/01, so it was open at cutoff. The remaining $785.93 is unexplained and is tracked under ops ticket OPS-2511-01077. I did not rely on JE-25-11-1225, a prior-period adjustment, or WD-2511-006377, an ETH item, because neither is eligible.'),
    ('dev-std-008-USDC-M1', 3, ('S111', 'S112', 'S113', 'S114', 'S115', 'S116'),
     ['HOT01', 'DP-2511-502477', 'OPS-2511-01077'],
     [('DP-2511-502477', 693.96, 'Timing difference')],
     'On-chain reserves of $432,158,959.32 sit $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in the M1 band (below $25,000). Deposit DP-2511-502477 for HOT01 was credited in the ledger at 23:22 UTC on 11/30 but confirmed on chain only at 00:14 UTC on 12/01, so it was open at cutoff and explains $693.96. That leaves $785.93 unexplained, which is still open under ops ticket OPS-2511-01077. I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and WD-2511-006377 is an ETH item, not USDC. Separately, the ledger figure comes from a close run at 02:10 UTC on 12/01, after cutoff, so the reviewer should confirm that post-cutoff postings are not included in it.'),
    ('dev-std-008-USDC-M1', 4, ('S139', 'S140', 'S141'),
     ['HOT01', 'DP-2511-502477', 'JE-25-11-1225', 'WD-2511-006377', 'OPS-2511-01077'],
     [('DP-2511-502477', 693.96, 'Timing difference')],
     'On-chain reserves of $432,158,959.32 are $1,479.89 below ledger liabilities of $432,160,439.21, a shortfall in band M1 (below the $25,000 floor). The deposit DP-2511-502477 into HOT01 was credited in the ledger at 23:22 UTC on 11/30 but confirmed on chain only at 00:14 UTC on 12/01, so it was open at cutoff and explains $693.96 of the shortfall. JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied on; this leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077.'),
    ('dev-std-010-BTC-M2', 1, ('S027',),
     ['HOT01', 'HOT02', 'DP-2512-574449', 'OI-251231-441'],
     [('DP-2512-574449', 66296.0, 'pending transfer'), ('OI-251231-441', 15.24, 'fee')],
     'On-chain reserves of $4,710,661,082.13 are below ledger liabilities of $4,710,727,393.37, a shortfall of $66,311.24 that falls in the M2 (reportable) band. Two open in-scope items dated 31-Dec-2025 explain all of it: DP-2512-574449 ($66,296.00), a HOT01 deposit credited in the ledger before the funds were received, and OI-251231-441 ($15.24), a HOT02 network fee that appears on chain only and was not booked in the ledger. Together they explain $66,311.24, leaving $0.00 unexplained. The cleared item, the TREAS01 balance and the post-cutoff postings were not relied on.'),
    ('dev-std-010-BTC-M2', 3, ('S117', 'S118'),
     ['DP-2512-574449', 'OI-251231-441', 'HOT01', 'HOT02'],
     [('DP-2512-574449', 66296.0, 'pending transfer'), ('OI-251231-441', 15.24, 'fee')],
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two eligible open items dated 31-Dec-2025 explain it in full. DP-2512-574449 ($66,296.00) is a HOT01 deposit that was pre-credited in the ledger with the funds not yet received on chain. OI-251231-441 ($15.24) is an unbooked chain-only network fee on the HOT02 sweep to COLD02. Together they explain $66,311.24, leaving $0.00 unexplained.'),
    ('dev-std-010-BTC-M2', 4, ('S142', 'S143', 'S144'),
     ['HOT01', 'HOT02', 'DP-2512-574449', 'OI-251231-441'],
     [('DP-2512-574449', 66296.0, 'pending transfer'), ('OI-251231-441', 15.24, 'fee')],
     'On-chain reserves of $4,710,661,082.13 are $66,311.24 below ledger liabilities of $4,710,727,393.37, a shortfall in the M2 (reportable) band. Two open items dated on or before the cutoff explain all of it. DP-2512-574449 ($66,296.00) is a HOT01 deposit pre-credited in the ledger with no funds on chain. OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep to COLD02, recorded on chain only. Together they explain $66,311.24, leaving $0.00 unexplained. I did not rely on the cleared DP-2512-039821, and WD-2601-039329 was broadcast after the cutoff.'),
    ('dev-std-012-SOL-M3', 1, ('S028', 'S029', 'S030', 'S031', 'S032', 'S033', 'S034', 'S035'),
     ['DP-2512-493757', 'DP-2512-052149', 'hot-1'],
     [('DP-2512-493757', 434042.88, 'pending transfer (chain only, increases surplus)'), ('DP-2512-052149', -91877.4, 'timing difference (ledger only, reduces surplus)')],
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10 by a surplus of $735,343.35, which is M3 (material-risk). Two eligible hot-1 items were open at cutoff: DP-2512-493757 was on chain only and adds $434,042.88 to the surplus, while DP-2512-052149 was ledger only (credited 23:48 on Dec 31, on chain after cutoff) and offsets it by $91,877.40. Net, they explain $342,165.48, leaving $393,177.87 of the surplus unexplained and with no root cause yet. I excluded the USDC movement (different asset), the Jan 1 deposit DP-2601-539265 (dated after cutoff), the suspense entry JE-25-12-5145 (suspense), the already-cleared DP-2512-310317, and the out-of-scope treasury-ops and staking-rewards wallets.'),
    ('dev-std-012-SOL-M3', 2, ('S070', 'S071', 'S072', 'S073', 'S074', 'S075', 'S076'),
     ['hot-1', 'DP-2512-493757', 'DP-2512-052149'],
     [('DP-2512-493757', 434042.88, 'pending transfer'), ('DP-2512-052149', -91877.4, 'timing difference')],
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two eligible items on hot-1 net to $342,165.48 explained: the 31 Dec on-chain deposit DP-2512-493757 not yet credited in the ledger (+$434,042.88), partly offset by DP-2512-052149, which was ledger-only at cutoff and reached the chain after it (-$91,877.40). That leaves $393,177.87 of the surplus unexplained and still under investigation. I excluded the USDC deposit (another asset), the cleared DP-2512-310317, the January item DP-2601-539265, the suspense entry JE-25-12-5145 and the out-of-scope wallets, and the review is still pending.'),
    ('dev-std-012-SOL-M3', 3, ('S119', 'S120', 'S121'),
     ['hot-1', 'DP-2512-493757', 'DP-2512-052149'],
     [('DP-2512-493757', 434042.88, 'pending transfer'), ('DP-2512-052149', -91877.4, 'timing difference')],
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 that falls in the M3 (material-risk) band. Two open hot-1 items dated on or before the cutoff are relied on. The DP-2512-493757 deposit, on chain but not yet credited in the ledger, adds $434,042.88 to the surplus. The DP-2512-052149 deposit, ledger-only at cutoff, offsets it by $91,877.40, so the net explained portion is $342,165.48. That leaves $393,177.87 unexplained, which stays under investigation with no root cause identified and no reviewer sign-off. The USDC movement (another asset), the suspense entry, the cleared item and the January-dated item are not relied on.'),
    ('dev-std-012-SOL-M3', 4, ('S145', 'S146'),
     ['SOL', 'hot-1', 'DP-2512-493757', 'DP-2512-052149'],
     [('DP-2512-493757', 434042.88, 'pending transfer'), ('DP-2512-052149', -91877.4, 'timing difference')],
     'On-chain reserves of $707,486,534.45 exceed ledger liabilities of $706,751,191.10, a surplus of $735,343.35 (M3, material-risk). Two open items on hot-1 are eligible. DP-2512-493757 is a chain-only deposit that adds $434,042.88 to the surplus. DP-2512-052149 is a ledger-only deposit that offsets it by $91,877.40, so the net explained amount is $342,165.48. This leaves $393,177.87 unexplained, and the root cause is still open and needs investigation before review.'),
    ('dev-std-013-USDC-M2', 2, ('S077', 'S078', 'S079', 'S080', 'S081', 'S082'),
     ['WD-2510-799057', 'HOT01'],
     [('WD-2510-799057', 34349.43, 'timing difference')],
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. The whole amount is explained by timing difference WD-2510-799057 on HOT01 for $34,349.43. The customer withdrawal was posted to the ledger on 31-Oct at 23:47:56 UTC but only went out on chain on 01-Nov, after the cutoff. The unexplained remainder is $0.00. I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the post-cutoff item WD-2511-221933.'),
    ('dev-std-013-USDC-M2', 3, ('S122',),
     ['HOT01', 'WD-2510-799057'],
     [('WD-2510-799057', 34349.43, 'timing difference')],
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the reportable (M2) band. The whole amount is explained by timing difference WD-2510-799057 on HOT01: the customer withdrawal was posted to the ledger on 31-Oct at 23:47:56 UTC, but it only went out on chain on 01-Nov, so the ledger was reduced before the reserves were. Explained portion is $34,349.43 and unexplained portion is $0.00. The manual adjustment, the cleared deposit, the post-cutoff item and the ETH deposit (a different asset) were not relied on.'),
    ('dev-std-013-USDC-M2', 4, ('S077', 'S078', 'S079', 'S080', 'S081', 'S082'),
     ['WD-2510-799057', 'HOT01'],
     [('WD-2510-799057', 34349.43, 'timing difference')],
     'On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by $34,349.43, a surplus in the M2 (reportable) band. The whole amount is explained by WD-2510-799057, a HOT01 customer withdrawal of $34,349.43 that was posted to the ledger at 23:47:56 UTC on 31-Oct but did not go on chain until 00:11 UTC on 01-Nov, after the cutoff. The unexplained remainder is $0.00. I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, or the post-cutoff item WD-2511-221933.'),
]

# dev-set row id: (kind, owner_answer, reference); the kind names the anchor that routed it
DEV_ROWS = {
    'S001': ('band_negation', 'fine', 'M1'),
    'S002': ('ineligible_mention', 'fine', 'JE-25-11-7405'),
    'S003': ('ineligible_mention', 'fine', 'DP-2511-680653'),
    'S004': ('ineligible_mention', 'fine', 'DP-2511-340361'),
    'S005': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S006': ('ineligible_mention', 'fine', 'DP-2511-582345'),
    'S007': ('ineligible_mention', 'fine', 'DP-2511-428189'),
    'S008': ('ineligible_mention', 'fine', 'DP-2511-679793'),
    'S009': ('ineligible_mention', 'fine', 'DP-2512-591373'),
    'S010': ('ineligible_mention', 'fine', 'JE-25-11-1801'),
    'S011': ('capitalized_phrase', 'fine', 'Three Hot'),
    'S012': ('amount_role', 'fine', '$28,749.96'),
    'S013': ('asset_negation', 'fine', 'ETH'),
    'S014': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S015': ('type_label', 'fine', 'WD-2510-164297'),
    'S016': ('amount_role', 'fine', '$0.00'),
    'S017': ('sentence_check', 'fine', 'Explained variance'),
    'S018': ('asset_negation', 'fine', 'USDC'),
    'S019': ('asset_negation', 'fine', 'SOL'),
    'S020': ('ineligible_mention', 'fine', 'DP-2511-983545'),
    'S021': ('ineligible_mention', 'fine', 'JE-25-11-2733'),
    'S022': ('amount_role', 'fine', '$256,238.97'),
    'S023': ('amount_role', 'fine', '$52,300.36'),
    'S024': ('amount_role', 'fine', '$1,880,341.85'),
    'S025': ('asset_negation', 'fine', 'ETH'),
    'S026': ('asset_negation', 'fine', 'USDC'),
    'S027': ('uncited_reference', 'error_citation_omission', 'TREAS01'),
    'S028': ('amount_role', 'fine', '$434,042.88'),
    'S029': ('amount_role', 'fine', '$91,877.40'),
    'S030': ('asset_negation', 'fine', 'USDC'),
    'S031': ('uncited_reference', 'error_citation_omission', 'DP-2601-539265'),
    'S032': ('uncited_reference', 'error_citation_omission', 'JE-25-12-5145'),
    'S033': ('uncited_reference', 'error_citation_omission', 'DP-2512-310317'),
    'S034': ('type_label', 'fine', 'DP-2512-493757'),
    'S035': ('type_label', 'fine', 'DP-2512-052149'),
    'S036': ('amount_role', 'fine', '$26,621.97'),
    'S037': ('amount_role', 'fine', '$29.32'),
    'S038': ('amount_role', 'fine', '$14,580.93'),
    'S039': ('amount_role', 'fine', '$7,163.92'),
    'S040': ('band_negation', 'fine', 'M1'),
    'S041': ('asset_negation', 'fine', 'USDC'),
    'S042': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S043': ('uncited_reference', 'error_citation_omission', 'JE-25-11-7121'),
    'S044': ('uncited_reference', 'error_citation_omission', 'WD-2512-156993'),
    'S045': ('uncited_reference', 'error_citation_omission', 'TREAS01'),
    'S046': ('amount_role', 'fine', '$0.00'),
    'S047': ('sentence_check', 'fine', 'Explained variance'),
    'S048': ('ineligible_mention', 'fine', 'DP-2511-582345'),
    'S049': ('ineligible_mention', 'fine', 'DP-2511-428189'),
    'S050': ('ineligible_mention', 'fine', 'DP-2511-679793'),
    'S051': ('ineligible_mention', 'fine', 'DP-2512-591373'),
    'S052': ('ineligible_mention', 'fine', 'JE-25-11-1801'),
    'S053': ('amount_role', 'fine', '$28,749.96'),
    'S054': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S055': ('uncited_reference', 'error_citation_omission', 'WD-2510-991353'),
    'S056': ('uncited_reference', 'error_citation_omission', 'JE-25-10-1421'),
    'S057': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S058': ('uncited_reference', 'error_citation_omission', 'JE-25-10-9681'),
    'S059': ('amount_role', 'fine', '$0.00'),
    'S060': ('sentence_check', 'fine', 'Explained variance'),
    'S061': ('amount_role', 'fine', '$256,238.97'),
    'S062': ('amount_role', 'fine', '$52,300.36'),
    'S063': ('sentence_check', 'fine', ''),
    'S064': ('type_label', 'fine', 'DP-2511-648193'),
    'S065': ('type_label', 'fine', 'DP-2511-893641'),
    'S066': ('amount_role', 'fine', '$1,880,341.85'),
    'S067': ('asset_negation', 'fine', 'ETH'),
    'S068': ('ineligible_mention', 'fine', 'JE-25-11-1225'),
    'S069': ('ineligible_mention', 'fine', 'WD-2511-006377'),
    'S070': ('amount_role', 'fine', '$342,165.48'),
    'S071': ('amount_role', 'fine', '$434,042.88'),
    'S072': ('amount_role', 'fine', '$91,877.40'),
    'S073': ('asset_negation', 'fine', 'USDC'),
    'S074': ('uncited_reference', 'error_citation_omission', 'DP-2512-310317'),
    'S075': ('uncited_reference', 'error_citation_omission', 'DP-2601-539265'),
    'S076': ('uncited_reference', 'error_citation_omission', 'JE-25-12-5145'),
    'S077': ('uncited_reference', 'error_citation_omission', 'JE-25-10-9025'),
    'S078': ('uncited_reference', 'error_citation_omission', 'DP-2510-831633'),
    'S079': ('uncited_reference', 'error_citation_omission', 'WD-2511-221933'),
    'S080': ('ineligible_mention', 'fine', 'JE-25-10-9025'),
    'S081': ('ineligible_mention', 'fine', 'DP-2510-831633'),
    'S082': ('ineligible_mention', 'fine', 'WD-2511-221933'),
    'S083': ('amount_role', 'fine', '$26,621.97'),
    'S084': ('amount_role', 'fine', '$29.32'),
    'S085': ('amount_role', 'fine', '$14,580.93'),
    'S086': ('amount_role', 'fine', '$7,163.92'),
    'S087': ('band_negation', 'fine', 'M1'),
    'S088': ('asset_negation', 'fine', 'USDC'),
    'S089': ('ineligible_mention', 'fine', 'JE-25-11-7405'),
    'S090': ('ineligible_mention', 'fine', 'DP-2511-680653'),
    'S091': ('ineligible_mention', 'fine', 'DP-2511-340361'),
    'S092': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S093': ('uncited_reference', 'error_citation_omission', 'WD-2512-156993'),
    'S094': ('uncited_reference', 'error_citation_omission', 'JE-25-11-7121'),
    'S095': ('uncited_reference', 'error_citation_omission', 'TREAS01'),
    'S096': ('amount_role', 'fine', '$0.00'),
    'S097': ('sentence_check', 'fine', 'Explained variance'),
    'S098': ('ineligible_mention', 'fine', 'DP-2511-582345'),
    'S099': ('ineligible_mention', 'fine', 'DP-2511-428189'),
    'S100': ('ineligible_mention', 'fine', 'DP-2511-679793'),
    'S101': ('ineligible_mention', 'fine', 'DP-2512-591373'),
    'S102': ('ineligible_mention', 'fine', 'JE-25-11-1801'),
    'S103': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S104': ('uncited_reference', 'error_citation_omission', 'JE-25-10-1421'),
    'S105': ('uncited_reference', 'error_citation_omission', 'WD-2510-991353'),
    'S106': ('uncited_reference', 'error_citation_omission', 'DP-2511-159273'),
    'S107': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S108': ('type_label', 'fine', 'TRF-2512-71501'),
    'S109': ('ineligible_mention', 'fine', 'JE-25-12-3777'),
    'S110': ('amount_role', 'fine', '$2,415,015.48'),
    'S111': ('asset_negation', 'fine', 'ETH'),
    'S112': ('asset_negation', 'fine', 'USDC'),
    'S113': ('uncited_reference', 'error_citation_omission', 'JE-25-11-1225'),
    'S114': ('uncited_reference', 'error_citation_omission', 'WD-2511-006377'),
    'S115': ('ineligible_mention', 'fine', 'JE-25-11-1225'),
    'S116': ('ineligible_mention', 'fine', 'WD-2511-006377'),
    'S117': ('sentence_check', 'fine', 'explain it in full'),
    'S118': ('uncited_reference', 'error_citation_omission', 'COLD02'),
    'S119': ('amount_role', 'fine', '$434,042.88'),
    'S120': ('amount_role', 'fine', '$91,877.40'),
    'S121': ('asset_negation', 'fine', 'USDC'),
    'S122': ('asset_negation', 'fine', 'ETH'),
    'S123': ('band_negation', 'fine', 'M1'),
    'S124': ('asset_negation', 'fine', 'USDC'),
    'S125': ('ineligible_mention', 'fine', 'JE-25-11-7405'),
    'S126': ('ineligible_mention', 'fine', 'DP-2511-680653'),
    'S127': ('ineligible_mention', 'fine', 'DP-2511-340361'),
    'S128': ('amount_role', 'fine', '$250,000.00'),
    'S129': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S130': ('uncited_reference', 'error_citation_omission', 'TREAS01'),
    'S131': ('ineligible_mention', 'fine', 'DP-2511-582345'),
    'S132': ('capitalized_phrase', 'fine', 'Three Hot'),
    'S133': ('amount_role', 'fine', '$28,749.96'),
    'S134': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S135': ('uncited_reference', 'error_citation_omission', 'JE-25-10-1421'),
    'S136': ('uncited_reference', 'error_citation_omission', 'WD-2510-991353'),
    'S137': ('capitalized_phrase', 'fine', 'Two Hot'),
    'S138': ('amount_role', 'fine', '$52,300.36'),
    'S139': ('asset_negation', 'fine', 'ETH'),
    'S140': ('ineligible_mention', 'fine', 'JE-25-11-1225'),
    'S141': ('ineligible_mention', 'fine', 'WD-2511-006377'),
    'S142': ('uncited_reference', 'error_citation_omission', 'COLD02'),
    'S143': ('uncited_reference', 'error_citation_omission', 'DP-2512-039821'),
    'S144': ('uncited_reference', 'error_citation_omission', 'WD-2601-039329'),
    'S145': ('amount_role', 'fine', '$434,042.88'),
    'S146': ('amount_role', 'fine', '$91,877.40'),
}

_ANCHOR_OF_KIND = {
    "amount_role": "correctness", "band_negation": "correctness", "asset_negation": "correctness",
    "sentence_check": "correctness", "ineligible_mention": "reconciling", "type_label": "reconciling",
    "capitalized_phrase": "fabrication", "uncited_reference": "fabrication",
}
# out of scope of the round (brief): these stay routed
KEEP_ROUTED = frozenset({"S063", "S128"})
# the brief's own pass-proofs do not reach these (see the module docstring)
STRETCH = frozenset({"S087", "S123", "S006", "S048", "S021"})


def _epoch_parsed(case_id, entities, items, text) -> dict:
    return answer(case_id, text, entities=entities, items=items)


def _epoch_id(row) -> str:
    return f"{row[0]}-e{row[1]}"


@pytest.mark.parametrize("case_id,epoch,rows,entities,items,text", [
    pytest.param(*row, id=_epoch_id(row)) for row in SONNET_EPOCHS
])
def test_whole_epoch_fires_no_fm(case_id, epoch, rows, entities, items, text):
    """Success bar: no owner-fine sentence may become a failure mode."""
    t = truth(case_id)
    parsed = _epoch_parsed(case_id, entities, items, text)
    for check in (check_correctness, check_reconciling_items, check_fabrication):
        result = check(parsed, t)
        assert result.failure_modes == (), (check.__name__, _debug(result))


def _after_fix_params():
    params = []
    for row in SONNET_EPOCHS:
        case_id, epoch, rows = row[0], row[1], row[2]
        for anchor in ("correctness", "reconciling", "fabrication"):
            mine = [r for r in rows if _ANCHOR_OF_KIND[DEV_ROWS[r][0]] == anchor]
            marks = ()
            if KEEP_ROUTED & set(mine):
                continue  # stays routed by the brief; covered by the no-FM test above
            if STRETCH & set(mine):
                marks = pytest.mark.xfail(strict=False, reason=f"stretch rows {sorted(STRETCH & set(mine))}")
            params.append(pytest.param(*row, anchor, id=f"{_epoch_id(row)}-{anchor}", marks=marks))
    return params


@pytest.mark.parametrize("case_id,epoch,rows,entities,items,text,anchor", _after_fix_params())
def test_whole_epoch_after_fix(case_id, epoch, rows, entities, items, text, anchor):
    """After the round: every owner-fine route of the anchor is decided by code and
    clears; every P12 row becomes a citation omission with the anchor at 0."""
    t = truth(case_id)
    parsed = _epoch_parsed(case_id, entities, items, text)
    if anchor == "correctness":
        expect(check_correctness(parsed, t), CLEAN)
    elif anchor == "reconciling":
        expect(check_reconciling_items(parsed, t), CLEAN)
    else:
        result = check_fabrication(parsed, t)
        omitted = sorted({DEV_ROWS[r][2] for r in rows if DEV_ROWS[r][0] == "uncited_reference"})
        assert result.failure_modes == () and result.review_reasons == (), result.details
        assert sorted(_omissions(result)) == omitted, result.details
        assert result.points == (0 if omitted else 3), result.details


def test_dev_rows_cover_every_epoch_row():
    listed = {r for row in SONNET_EPOCHS for r in row[2]}
    assert listed == set(DEV_ROWS), sorted(set(DEV_ROWS) ^ listed)
    assert sum(1 for k, a, _ in DEV_ROWS.values() if a == "fine") == 115
    assert sum(1 for k, a, _ in DEV_ROWS.values() if a == "error_citation_omission") == 31
