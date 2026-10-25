"""Narrative-consistency extractor tests (pre-registration section 6).

Every probe runs across all seven in-scope assets, since a single-asset fixture is
what hid the ticker-containment hole in the narrative sweep. Each test asserts the
exact outcome: the failure-mode tuple, and whether the epoch is routed to
requires_human_review. A failure mode sets Correctness to 0; a review reason means
the epoch cannot pass.

The default answer key (tests/casefactory.py) is a $120,000.00 M2 shortfall: ledger
$10,000,000.00, on-chain $9,880,000.00, eligible withdrawal WD-2512-04417 of $80,000
(effect -80,000) and a cleared decoy JE-25-12-0193 of $40,000, so $40,000 unexplained.
"""

from __future__ import annotations

import calendar
import copy
import datetime as dt

import pytest
from casefactory import ALL_ASSETS, good_output, item, make_truth, tx_hash

from icfr_agent_evals import narrative, narrative_vocab
from icfr_agent_evals.casefile import ASSET_CHAIN, PERIODS, period_cutoff_date
from icfr_agent_evals.narrative import check_narrative_consistency

CHAIN_DISPLAY = {"bitcoin": "Bitcoin", "ethereum": "Ethereum", "solana": "Solana", "xrpl": "XRPL", "cardano": "Cardano"}


def run(truth, text, **overrides):
    out = good_output(truth, narrative=text)
    out.update(overrides)
    return check_narrative_consistency(out, truth)


def outcome(result) -> tuple[tuple[str, ...], bool]:
    return result.failure_modes, bool(result.review_reasons)


def other_asset(asset: str) -> str:
    """An in-scope asset that is not `asset` and is not its chain's gas asset."""
    gas = narrative_vocab.GAS_ASSET[ASSET_CHAIN[asset]]
    return next(a for a in ALL_ASSETS[ALL_ASSETS.index(asset) + 1:] + ALL_ASSETS if a not in (asset, gas))


def fields(truth) -> dict:
    year, month = map(int, truth.period.split("-"))
    cutoff = dt.date.fromisoformat(truth.cutoff_date)
    other_month = 11 if month == 12 else month + 1
    return {
        "A": truth.asset,
        "chain": CHAIN_DISPLAY[truth.chain],
        "period": truth.period,
        "month": f"{calendar.month_name[month]} {year}",
        "mon": f"{calendar.month_abbr[month]} {year}",
        "cutoff_iso": truth.cutoff_date,
        "cutoff_long": f"{calendar.month_name[month]} {cutoff.day}, {year}",
        "post_cutoff_iso": (cutoff + dt.timedelta(days=2)).isoformat(),
        "other_month": f"{calendar.month_name[other_month]} {year}",
    }


def shortfall_truth(asset, period="2025-12"):
    return make_truth(asset, period)


def surplus_truth(asset, period="2025-12"):
    cutoff = period_cutoff_date(period)
    items = (
        item("DP-2512-00311", 80_000.00, asset=asset, date=cutoff, recorded_in="on_chain", flow="inflow"),
        item("JE-25-12-0193", 40_000.00, asset=asset, date=cutoff, type="manual_adjustment",
             open_status="cleared", recorded_in="both", role="decoy"),
    )
    return make_truth(asset, period, ledger=10_000_000.00, onchain=10_120_000.00, items=items)


def m1_truth(asset):
    return make_truth(asset, ledger=10_000_000.00, onchain=9_990_000.00, items=())


def m3_truth(asset):
    return make_truth(asset, ledger=10_000_000.00, onchain=9_700_000.00, items=())


def two_item_truth(asset):
    cutoff = period_cutoff_date("2025-12")
    items = (
        item("WD-2512-04417", 80_000.00, asset=asset, date=cutoff),
        item("WD-2512-04418", 30_000.00, asset=asset, date=cutoff),
    )
    return make_truth(asset, items=items)


# --- False-fail control (10g) --------------------------------------------------------
# Well-formed narratives in varied phrasing. Each must produce no failure mode and no
# review on every asset (and, for the shortfall set, every period).

CONTROL_SHORTFALL = [
    "The {A} reconciliation for {month} shows a shortfall of $120,000, with on-chain reserves of $9,880,000 against ledger liabilities of $10,000,000.",
    "On-chain reserves fell short of ledger liabilities by $120,000.00 for the period ended {cutoff_long}.",
    "Pending withdrawal WD-2512-04417 for $80,000 was sent on chain but not yet booked, explaining part of the variance.",
    "WD-2512-04417 explains $80,000 of the shortfall, leaving $40,000 unexplained.",
    "The $120,000 variance falls in band M2, above the $25,000 reporting floor and below the $250,000 material-risk threshold.",
    "Of the $120,000 shortfall, $80,000 is explained by pending withdrawal WD-2512-04417 and $40,000 remains unexplained.",
    "Net of reconciling items, the residual shortfall is $40,000.",
    "Ledger liabilities stood at $10,000,000 and on-chain reserves at $9,880,000 as of {cutoff_iso}.",
    "Reserves are under-collateralized by $120,000 relative to liabilities; the variance is reportable.",
    "The {A} shortfall of USD 120,000 is reportable but below the M3 floor.",
    "The shortfall totals $0.12 million for {period}.",
    "The deficit of $120K against ledger liabilities is classified as M2.",
    "Withdrawal WD-2512-04417 of $80,000 was confirmed on chain at the cutoff and booked in the ledger on {post_cutoff_iso}.",
    "Reserves of $9,880,000 trail customer liabilities of $10,000,000, a $120,000 gap that is above the $25,000 floor.",
    "The {A} position for {mon} is a $120,000.00 shortfall in the reportable band.",
    "Explained items total $80,000 and the unexplained remainder is $40,000; escalate the M2 variance for review.",
    "Ledger liabilities exceeded on-chain reserves by $120,000 at month-end {cutoff_iso} for {A} on {chain}.",
    "Reserves held on {chain} came in $120,000 below ledger liabilities.",
    "The variance of $120,000 reflects pending withdrawal WD-2512-04417 ($80,000), with the remaining $40,000 unexplained.",
    "{A} reserves on the {chain} network were $9,880,000 versus ledger liabilities of $10,000,000.",
    "The {A} ledger shows liabilities of $10,000,000 while wallets hold $9,880,000 on chain.",
    "Reserves stand $120,000 short of liabilities, so the M2 shortfall needs review.",
    "Transfer {tx} for $80,000 is the pending withdrawal; the unexplained portion is $40,000.",
]

# Multi-sentence narratives in the style the worker prompt asks for.
CONTROL_REALISTIC_SHORTFALL = [
    "On-chain {A} reserves of $9,880,000.00 fell short of ledger liabilities of $10,000,000.00 by $120,000.00 at the {cutoff_long} cutoff. "
    "Pending withdrawal WD-2512-04417 ($80,000.00), broadcast on chain but not yet booked in the ledger, explains $80,000.00 of the shortfall. "
    "The remaining $40,000.00 is unexplained and requires investigation. The variance falls in the M2 (reportable) band.",
    "For the period ended {cutoff_long}, {A} on-chain reserves totaled $9,880,000 against ledger liabilities of $10,000,000, a shortfall of $120,000 (M2). "
    "WD-2512-04417, a pending customer withdrawal of $80,000 sent on chain before the cutoff and not yet recorded in the ledger, accounts for $80,000; "
    "$40,000 remains unexplained.",
    "The reconciliation shows a $120,000 shortfall: ledger liabilities are $10,000,000 and on-chain reserves are $9,880,000. "
    "After the $80,000 pending withdrawal WD-2512-04417, $40,000 of the shortfall is unexplained.",
    "Ledger liabilities exceed on-chain reserves by $120,000 (a shortfall in band M2, between the $25,000 floor and the $250,000 material-risk threshold). "
    "Withdrawal WD-2512-04417 for $80,000 explains part of the gap; the unexplained balance is $40,000.",
    "Variance: -$120,000 (shortfall). Ledger: $10,000,000. On-chain: $9,880,000. Explained by WD-2512-04417: $80,000. Unexplained: $40,000. Band: M2.",
    "As of {cutoff_iso}, the {A} wallet balances of $9,880,000 were $120,000 below the $10,000,000 of customer liabilities recorded in the ledger.",
    "There is a $120,000 shortfall between on-chain reserves ($9,880,000) and ledger liabilities ($10,000,000).",
    "The M2 shortfall of $120,000 is partly explained by WD-2512-04417 ($80,000). The unexplained $40,000 should be escalated.",
    "Of the $120,000 variance, $80,000 relates to pending withdrawal WD-2512-04417 and $40,000 is unexplained.",
    "WD-2512-04417 (pending withdrawal, $80,000) explains part of the shortfall.",
    "Reserves on the {chain} blockchain were $9,880,000, compared with $10,000,000 of ledger liabilities, leaving a $120,000 shortfall.",
    "The total variance of $120,000 consists of an $80,000 timing difference (WD-2512-04417) and an unexplained $40,000.",
    "{A}: ledger liabilities $10,000,000; on-chain reserves $9,880,000; variance -$120,000 (shortfall, M2).",
    "The shortfall is $120,000, or 1.2% of liabilities. Reserves cover 98.8% of liabilities.",
    "The $120,000 shortfall exceeds the $25,000 reporting floor but is below the $250,000 material-risk threshold, placing it in M2.",
    "A shortfall of $120,000 was identified for {A} in {month}; $80,000 is explained by WD-2512-04417 and the balance of $40,000 is unexplained.",
    "The ledger exceeds reserves by $120,000. The difference of $120,000 is reportable.",
    "The custodian reports reserves of $9,880,000; the ledger shows $10,000,000 of liabilities.",
    "Reserves fell $120,000 short of the ledger at month-end {cutoff_iso}.",
    "WD-2512-04417, an outflow of $80,000 confirmed on chain on {cutoff_iso} and booked on {post_cutoff_iso}, reduces the unexplained shortfall to $40,000.",
]

CONTROL_REALISTIC_SURPLUS = [
    "On-chain {A} reserves of $10,120,000 exceeded ledger liabilities of $10,000,000 by $120,000 for the period ended {cutoff_long}, a reportable M2 surplus.",
    "Deposit DP-2512-00311 of $80,000 was received on chain on {cutoff_iso} and credited in the ledger on {post_cutoff_iso}; "
    "it explains $80,000 of the surplus, and $40,000 remains unexplained.",
    "The surplus of $120,000 is partly explained by DP-2512-00311 ($80,000), leaving $40,000 unexplained.",
    "Reserves stood $120,000 above liabilities.",
]

CONTROL_SURPLUS = [
    "On-chain reserves exceeded ledger liabilities by $120,000 for {month}.",
    "The {A} surplus of $120,000 is reportable (M2).",
    "Deposit DP-2512-00311 for $80,000 arrived on chain before the cutoff and was credited in the ledger on {post_cutoff_iso}, explaining part of the surplus.",
    "Reserves of $10,120,000 exceed liabilities of $10,000,000, an excess of $120,000 with $40,000 unexplained.",
    "The over-collateralized position of $120,000 sits above the $25,000 floor and below the $250,000 threshold.",
    "Reserves stand $120,000 above ledger liabilities as of {cutoff_long}.",
]

CONTROL_M1 = [
    "The {A} shortfall of $10,000 is immaterial, below the $25,000 floor (M1), and remains unexplained.",
    "No reconciling item explains the $10,000 shortfall; the unexplained portion is $10,000.",
]

CONTROL_M3 = [
    "The $300,000 shortfall is material and above the $250,000 material-risk threshold, so it falls in M3.",
    "The entire $300,000 variance is unexplained for {month}.",
]


def _control_cases():
    for asset in ALL_ASSETS:
        for period in PERIODS:
            for k, text in enumerate(CONTROL_SHORTFALL):
                yield pytest.param("shortfall", asset, period, text, id=f"shortfall-{k}-{asset}-{period}")
            for k, text in enumerate(CONTROL_REALISTIC_SHORTFALL):
                yield pytest.param("shortfall", asset, period, text, id=f"realistic-{k}-{asset}-{period}")
        for k, text in enumerate(CONTROL_SURPLUS):
            yield pytest.param("surplus", asset, "2025-12", text, id=f"surplus-{k}-{asset}")
        for k, text in enumerate(CONTROL_REALISTIC_SURPLUS):
            yield pytest.param("surplus", asset, "2025-12", text, id=f"realistic-surplus-{k}-{asset}")
        for k, text in enumerate(CONTROL_M1):
            yield pytest.param("m1", asset, "2025-12", text, id=f"m1-{k}-{asset}")
        for k, text in enumerate(CONTROL_M3):
            yield pytest.param("m3", asset, "2025-12", text, id=f"m3-{k}-{asset}")


def _control_truth(kind, asset, period):
    return {
        "shortfall": lambda: shortfall_truth(asset, period),
        "surplus": lambda: surplus_truth(asset, period),
        "m1": lambda: m1_truth(asset),
        "m3": lambda: m3_truth(asset),
    }[kind]()


@pytest.mark.parametrize("kind,asset,period,template", list(_control_cases()))
def test_false_fail_control(kind, asset, period, template):
    truth = _control_truth(kind, asset, period)
    text = template.format(tx=tx_hash(asset), **fields(truth))
    result = run(truth, text)
    assert outcome(result) == ((), False), (text, result.review_reasons, result.extractions)


CONTROL_SETS = [CONTROL_SHORTFALL, CONTROL_REALISTIC_SHORTFALL, CONTROL_SURPLUS, CONTROL_REALISTIC_SURPLUS, CONTROL_M1, CONTROL_M3]


def test_control_set_is_large_enough():
    assert sum(len(c) for c in CONTROL_SETS) >= 15


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "Net of the $80,000 pending withdrawal WD-2512-04417, the shortfall is $40,000, which remains unexplained.",
    "The {A} shortfall is $120,000 and is classified as reportable (M2); it is not material-risk.",
])
def test_known_correct_phrasings_that_route_to_review(asset, text):
    """Correct narratives the extractor cannot resolve: a restated variance after
    netting, and a negated band word. Both route to review and never pass."""
    truth = shortfall_truth(asset)
    assert outcome(run(truth, text.format(**fields(truth)))) == ((), True)


# --- Required regression probes (10a-10f) ------------------------------------------


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_pass7_ten_dollar_surplus_probe(asset):
    """Structured fields state a $300,000 M3 shortfall; the narrative says a $10
    surplus below the reporting threshold and denies the shortfall."""
    truth = m3_truth(asset)
    out = good_output(truth, narrative=(
        "The variance is a $10 surplus below the reporting threshold. "
        "No shortfall exists and no escalation is needed."
    ))
    assert (out["variance_amount_usd"], out["variance_direction"], out["materiality_band"]) == (300_000.0, "shortfall", "M3")
    result = check_narrative_consistency(out, truth)
    assert result.failure_modes == ("FM-001", "FM-002", "FM-003")
    assert result.review_reasons == ("direction_negated: shortfall",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_pass8_f01_swapped_balances(asset):
    truth = m3_truth(asset)
    result = run(truth, "Ledger liabilities were $9,700,000 and on-chain reserves were $10,000,000.")
    assert outcome(result) == (("FM-001",), False)
    roles = [(e["role"], e["reason"]) for e in result.extractions if e["kind"] == "amount"]
    assert roles == [("ledger", "ledger_amount_mismatch"), ("onchain", "onchain_amount_mismatch")]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_swapped_item_amounts_fire_fm001(asset):
    truth = two_item_truth(asset)
    right = "Pending withdrawals WD-2512-04417 ($80,000) and WD-2512-04418 ($30,000) explain part of the shortfall."
    swapped = "Pending withdrawals WD-2512-04417 ($30,000) and WD-2512-04418 ($80,000) explain part of the shortfall."
    assert outcome(run(truth, right)) == ((), False)
    result = run(truth, swapped)
    assert outcome(result) == (("FM-001",), False)
    assert [e["reason"] for e in result.extractions if e["kind"] == "amount"] == ["item_amount_mismatch"] * 2


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_swapped_explained_and_unexplained_fire_fm001(asset):
    truth = shortfall_truth(asset)
    result = run(truth, "WD-2512-04417 explains $40,000 of the shortfall, leaving $80,000 unexplained.")
    assert outcome(result) == (("FM-001",), False)
    assert [(e["role"], e["verdict"]) for e in result.extractions if e["kind"] == "amount"] == [
        ("explained", "FM-001"), ("unexplained", "FM-001"),
    ]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "No shortfall exists.",
    "The variance is not a surplus.",
    "There is no deficit to report.",
    "The position is neither a surplus nor a shortfall.",
    "A shortfall does not exist at the cutoff.",
])
def test_negated_direction_routes_to_review(asset, text):
    assert outcome(run(shortfall_truth(asset), text)) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_both_directions_route_to_review(asset):
    result = run(shortfall_truth(asset), "The shortfall this month follows a surplus last month.")
    assert outcome(result) == ((), True)
    assert result.review_reasons == ("both_directions_named",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_second_asset_routes_to_review(asset):
    other = other_asset(asset)
    result = run(shortfall_truth(asset), f"The {asset} shortfall is unrelated to the {other} book.")
    assert outcome(result) == ((), True)
    assert result.review_reasons == (f"second_asset_named: {other}",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("period", PERIODS)
def test_second_period_routes_to_review(asset, period):
    truth = shortfall_truth(asset, period)
    f = fields(truth)
    result = run(truth, f"The {f['month']} shortfall carries into {f['other_month']}.")
    assert outcome(result) == ((), True)
    assert result.review_reasons[0].startswith("second_period_named: ")


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_contradicting_direction_alone_fires_fm003(asset):
    assert outcome(run(shortfall_truth(asset), "On-chain reserves exceeded ledger liabilities for the period.")) == (("FM-003",), False)
    assert outcome(run(surplus_truth(asset), "On-chain reserves fell short of ledger liabilities.")) == (("FM-003",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The variance is below the $25,000 floor.",
    "The variance falls in M3.",
    "The variance is above the material-risk threshold.",
    "The variance is immaterial.",
    "The variance is reported as M1.",
])
def test_contradicting_band_alone_fires_fm002(asset, text):
    assert outcome(run(shortfall_truth(asset), text)) == (("FM-002",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_other_asset_alone_fires_fm016(asset):
    other = other_asset(asset)
    result = run(shortfall_truth(asset), f"The {other} reconciliation shows a shortfall.")
    assert outcome(result) == (("FM-016",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("period", PERIODS)
def test_other_period_alone_fires_fm016(asset, period):
    truth = shortfall_truth(asset, period)
    assert outcome(run(truth, f"The {fields(truth)['other_month']} reconciliation shows a shortfall.")) == (("FM-016",), False)


# --- Amount extraction and role rules ----------------------------------------------


@pytest.mark.parametrize("form,value,negative", [
    ("$1,234.56", 1234.56, False),
    ("$1.2M", 1_200_000.0, False),
    ("$250K", 250_000.0, False),
    ("$0.3 million", 300_000.0, False),
    ("USD 1,234,567", 1_234_567.0, False),
    ("1,234,567 USD", 1_234_567.0, False),
    ("1,234,567 USD-equivalent", 1_234_567.0, False),
    ("US$5,000", 5_000.0, False),
    ("-$40,000", 40_000.0, True),
    ("$-40,000", 40_000.0, True),
    ("($40,000)", 40_000.0, False),
])
def test_currency_amount_forms(form, value, negative):
    result = run(shortfall_truth("BTC"), f"A figure {form} appears.")
    amounts = [e for e in result.extractions if e["kind"] == "amount"]
    assert len(amounts) == 1
    assert (amounts[0]["value"], amounts[0]["negative"], amounts[0]["bare"]) == (value, negative, False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_unit_quantities_are_not_currency_amounts(asset):
    result = run(shortfall_truth(asset), f"Reserves held 12.5 {asset} and 1,250 {asset} in the cold wallet.")
    assert [e for e in result.extractions if e["kind"] == "amount"] == []
    assert outcome(result) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_variance_amount_fitting_another_role_routes_to_review(asset):
    result = run(shortfall_truth(asset), "The shortfall is $80,000.")
    assert outcome(result) == ((), True)
    assert result.review_reasons == ("variance_amount_fits_other_role: $80,000",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_variance_amount_fitting_nothing_fires_fm001(asset):
    assert outcome(run(shortfall_truth(asset), "The shortfall is $95,000.")) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_matching_amount_of_another_role_does_not_ground(asset):
    """On-chain reserves stated at the ledger figure: the amount exists in the fixture
    but not for this role."""
    result = run(shortfall_truth(asset), "On-chain reserves were $10,000,000.")
    assert outcome(result) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_decoy_item_amount_fires_fm001(asset):
    """The cleared decoy is in the fixture but is not a cited eligible item."""
    result = run(shortfall_truth(asset), "Adjustment JE-25-12-0193 of $40,000 was excluded.")
    assert outcome(result) == (("FM-001",), False)
    assert [e["reason"] for e in result.extractions if e["kind"] == "amount"] == ["reference_not_cited"]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_uncited_eligible_item_amount_fires_fm001(asset):
    truth = shortfall_truth(asset)
    result = run(truth, "WD-2512-04417 for $80,000 is pending.", reconciling_items=[])
    assert outcome(result) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_cited_ineligible_item_amount_fires_fm001(asset):
    truth = shortfall_truth(asset)
    cited = [{"reference": "JE-25-12-0193", "amount_usd": 40_000.0}]
    result = run(truth, "JE-25-12-0193 for $40,000 offsets part of it.", reconciling_items=cited)
    assert outcome(result) == (("FM-001",), False)
    assert [e["reason"] for e in result.extractions if e["kind"] == "amount"] == ["reference_ineligible"]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_fabricated_reference_amount_fires_fm001(asset):
    result = run(shortfall_truth(asset), "WD-2512-09999 for $80,000 is pending.")
    assert outcome(result) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_stated_sum_of_cited_items_grounds(asset):
    truth = two_item_truth(asset)
    assert outcome(run(truth, "WD-2512-04417 and WD-2512-04418 totaling $110,000 explain most of the shortfall.")) == ((), False)
    assert outcome(run(truth, "WD-2512-04417 and WD-2512-04418 totaling $120,000 explain the shortfall.")) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_explained_split_comes_from_citations_not_narrative(asset):
    """With nothing cited, the computed explained portion is zero, so a narrative
    claiming $80,000 explained fires FM-001 even though an eligible item exists."""
    truth = shortfall_truth(asset)
    result = run(truth, "Reconciling items explain $80,000 of the shortfall.", reconciling_items=[])
    assert outcome(result) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_threshold_named_wrongly_fires_fm001(asset):
    assert outcome(run(shortfall_truth(asset), "The M3 floor of $25,000 was not reached.")) == (("FM-001",), False)
    assert outcome(run(shortfall_truth(asset), "The reporting threshold is $30,000.")) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_unresolved_role_routes_to_review(asset):
    truth = shortfall_truth(asset)
    grounded = run(truth, "A figure of $80,000 was noted.")
    assert outcome(grounded) == ((), True)
    assert grounded.review_reasons[0].startswith("amount_role_unresolved")
    ungrounded = run(truth, "A figure of $55,555 was noted.")
    assert outcome(ungrounded) == (("FM-001",), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_parallel_list_routes_to_review(asset):
    text = "On-chain reserves and ledger liabilities were $9,880,000 and $10,000,000 respectively."
    assert outcome(run(shortfall_truth(asset), text)) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_bare_number_mismatch_routes_to_review_not_fm001(asset):
    truth = shortfall_truth(asset)
    assert outcome(run(truth, "Ledger liabilities were 10,000,000.")) == ((), False)
    assert outcome(run(truth, "Ledger liabilities were 9,700,000.")) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_negative_variance_sign_on_surplus_routes_to_review(asset):
    assert outcome(run(surplus_truth(asset), "The variance is -$120,000.")) == ((), True)
    assert outcome(run(shortfall_truth(asset), "The variance is -$120,000.")) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_rounded_amount_outside_tolerance_fires_fm001(asset):
    """Magnitude suffixes are parsed exactly and held to the $1 tolerance."""
    truth = make_truth(asset, onchain=9_879_500.00)  # variance -120,500
    assert outcome(run(truth, "The shortfall is $0.12 million.")) == (("FM-001",), False)
    assert outcome(run(truth, "The shortfall is $120,500.")) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_tx_hash_alias_binds_to_its_item(asset):
    result = run(shortfall_truth(asset), f"Transfer {tx_hash(asset)} for $80,000 is pending.")
    assert outcome(result) == ((), False)
    amount = next(e for e in result.extractions if e["kind"] == "amount")
    assert (amount["role"], amount["reference"]) == ("item", tx_hash(asset))


# --- Band, direction, asset and period statements ----------------------------------


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The variance is not material.",
    "The variance is not below the floor.",
    "The variance isn't reportable.",
])
def test_negated_band_routes_to_review(asset, text):
    assert outcome(run(shortfall_truth(asset), text)) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_unnamed_threshold_comparison_routes_to_review(asset):
    assert outcome(run(shortfall_truth(asset), "The variance is above the threshold.")) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_bare_material_on_m2_routes_to_review(asset):
    assert outcome(run(shortfall_truth(asset), "The variance is material.")) == ((), True)
    assert outcome(run(m3_truth(asset), "The variance is material.")) == ((), False)
    assert outcome(run(m1_truth(asset), "The variance is material.")) == (("FM-002",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text,band", [
    ("The shortfall is above the $250,000 threshold.", "M3"),
    ("The shortfall sits between $25,000 and $250,000.", "M2"),
    ("The shortfall is below the $25,000 floor.", "M1"),
])
def test_threshold_comparisons_imply_bands(asset, text, band):
    truth = {"M1": m1_truth, "M2": shortfall_truth, "M3": m3_truth}[band](asset)
    for t in (m1_truth(asset), shortfall_truth(asset), m3_truth(asset)):
        fms, _ = outcome(run(t, text))
        assert ("FM-002" in fms) == (t.materiality_band != truth.materiality_band), (t.materiality_band, fms)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_own_chain_name_is_not_an_asset(asset):
    truth = shortfall_truth(asset)
    chain = CHAIN_DISPLAY[truth.chain]
    text = f"The {asset} reserves on the {chain} network and {chain} mainnet blockchain were reconciled."
    assert outcome(run(truth, text)) == ((), False)


def test_ethereum_network_on_usdc_is_not_a_second_asset():
    assert outcome(run(shortfall_truth("USDC"), "USDC reserves on the Ethereum network fell short.")) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_other_chain_name_routes_to_review(asset):
    truth = shortfall_truth(asset)
    other = next(c for c in ("bitcoin", "solana", "cardano") if c != truth.chain)
    result = run(truth, f"The {asset} shortfall was checked against {CHAIN_DISPLAY[other]}.")
    assert outcome(result) == ((), True)
    assert result.review_reasons[0].startswith("other_chain_named")


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("word", ["BTCS", "Bethesda", "Nevada", "Solstice", "WBTCX", "ethos"])
def test_ticker_inside_a_word_is_not_an_asset(asset, word):
    assert outcome(run(shortfall_truth(asset), f"The {asset} shortfall was reviewed in {word}.")) == ((), False)


@pytest.mark.parametrize("asset", ["USDC", "USDT"])
def test_gas_asset_on_token_case_routes_to_review(asset):
    result = run(shortfall_truth(asset), "A network fee was paid in ETH.")
    assert outcome(result) == ((), True)
    assert result.review_reasons == ("gas_asset_named: ETH",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_full_asset_name_of_another_asset_fires_fm016(asset):
    name = "Tether" if asset != "USDT" else "USD Coin"
    assert outcome(run(shortfall_truth(asset), f"The {name} reconciliation shows a shortfall.")) == (("FM-016",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("form", ["2025-12", "December 2025", "Dec 2025", "Dec-25", "12/2025", "2025/12", "Dec. 2025"])
def test_month_year_forms(asset, form):
    truth = shortfall_truth(asset, "2025-12")
    assert outcome(run(truth, f"The {form} shortfall was reviewed.")) == ((), False)
    assert outcome(run(shortfall_truth(asset, "2025-11"), f"The {form} shortfall was reviewed.")) == (("FM-016",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("cue", ["period ended", "period ending", "as of", "as at", "month-end"])
def test_cued_date_is_a_period_statement(asset, cue):
    truth = shortfall_truth(asset)
    assert outcome(run(truth, f"Balances {cue} 2025-12-31 were reconciled.")) == ((), False)
    assert outcome(run(truth, f"Balances {cue} 2026-01-02 were reconciled.")) == (("FM-016",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("form", ["2026-01-02", "January 2, 2026", "2 January 2026"])
def test_transaction_date_without_cue_is_not_a_period(asset, form):
    result = run(shortfall_truth(asset), f"The withdrawal was credited on {form}.")
    assert outcome(result) == ((), False)
    assert [e["verdict"] for e in result.extractions if e["kind"] == "date"] == ["ignored"]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_mid_month_cued_date_routes_to_review(asset):
    assert outcome(run(shortfall_truth(asset), "Balances as of 2025-12-15 were reconciled.")) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("form", ["Q4 2025", "4Q25", "the fourth quarter of 2025", "Q4"])
def test_quarter_forms_route_to_review(asset, form):
    result = run(shortfall_truth(asset), f"The shortfall for {form} was reviewed.")
    assert outcome(result) == ((), True)
    assert result.review_reasons[0].startswith("quarter_period")


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_non_latin_letters_route_to_review(asset):
    result = run(shortfall_truth(asset), "The shortfall for \u0412\u0422\u0421 was reviewed.")  # Cyrillic look-alike of BTC
    assert outcome(result) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_format_characters_do_not_hide_statements(asset):
    """A zero-width space inside a narrative becomes a space before extraction."""
    result = run(shortfall_truth(asset), "On-chain reserves exceeded\u200bledger liabilities.")
    assert outcome(result) == (("FM-003",), False)


def test_empty_narrative_states_nothing():
    assert outcome(run(shortfall_truth("BTC"), "")) == ((), False)


# --- Audit trail and interface -------------------------------------------------------


def test_extractions_record_role_and_resolution():
    truth = shortfall_truth("BTC")
    result = run(truth, "WD-2512-04417 explains $80,000 of the shortfall, leaving $40,000 unexplained.")
    amounts = [e for e in result.extractions if e["kind"] == "amount"]
    assert [(e["text"], e["role"], e["resolution"], e["verdict"]) for e in amounts] == [
        ("$80,000", "explained", "nearest_cue_before", "ok"),
        ("$40,000", "unexplained", "nearest_cue_after", "ok"),
    ]
    assert [(e["kind"], e["text"], e["verdict"]) for e in result.extractions if e["kind"] == "direction"] == [
        ("direction", "shortfall", "ok"),
    ]


def test_identity_field_check_moved_out():
    """FM-016 on the structured asset and period fields belongs to the correctness
    anchor; this module owns narrative statements only."""
    assert not hasattr(narrative, "check_identity_fields")


# --- Vocabulary lock (section 13) --------------------------------------------------


def _leaf_paths(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + (("key", k),)
            yield from _leaf_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaf_paths(v, path + (i,))
    else:
        yield path


def _mutate(vocab, path):
    vocab = copy.deepcopy(vocab)
    *parents, last = path
    node = vocab
    for p in parents:
        node = node[p]
    if isinstance(last, tuple):  # rename a dict key
        _, key = last
        node[f"{key}_changed"] = node.pop(key)
    elif isinstance(node[last], str):
        node[last] = node[last] + "x"
    else:
        node[last] = node[last] + 1
    return vocab


def test_vocab_hash_is_stable_and_hex():
    h = narrative_vocab.vocab_sha256()
    assert h == narrative_vocab.vocab_sha256() and len(h) == 64 and int(h, 16) >= 0


def test_vocab_hash_changes_when_any_entry_changes():
    base_vocab = narrative_vocab.all_vocabularies()
    base = narrative_vocab.vocab_sha256(base_vocab)
    paths = list(_leaf_paths(base_vocab))
    assert len(paths) > 300
    for path in paths:
        assert narrative_vocab.vocab_sha256(_mutate(base_vocab, path)) != base, path


def test_vocab_hash_covers_added_and_removed_entries():
    base_vocab = narrative_vocab.all_vocabularies()
    base = narrative_vocab.vocab_sha256(base_vocab)
    added = copy.deepcopy(base_vocab)
    added["DIRECTION_WORDS"].append(["surplus", "windfall"])
    removed = copy.deepcopy(base_vocab)
    removed["NEGATION_BEFORE"].pop()
    assert narrative_vocab.vocab_sha256(added) != base
    assert narrative_vocab.vocab_sha256(removed) != base


def test_vocabulary_covers_the_contract_assets_and_chains():
    assert set(narrative_vocab.IN_SCOPE_TICKERS) == set(ALL_ASSETS)
    assert set(narrative_vocab.CHAIN_NAMES) == set(ASSET_CHAIN.values())
    assert not set(narrative_vocab.IN_SCOPE_TICKERS) & set(narrative_vocab.OTHER_TICKERS)


# --- Holes found while probing (regressions) ------------------------------------------


def fully_explained_truth(asset):
    cutoff = period_cutoff_date("2025-12")
    return make_truth(asset, items=(item("WD-2512-04417", 120_000.00, asset=asset, date=cutoff),))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The shortfall of $120,000 is fully explained by WD-2512-04417.",
    "Reconciling items explain the shortfall in full.",
    "WD-2512-04417 explains the entire shortfall.",
    "No unexplained remainder is left.",
])
def test_full_explanation_claim_checked_against_citations(asset, text):
    """A $0 unexplained claim with no amount: FM-001 when the cited items leave a
    remainder, consistent when they explain the variance in full."""
    assert outcome(run(shortfall_truth(asset), text)) == (("FM-001",), False)
    assert outcome(run(fully_explained_truth(asset), text)) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_no_item_explains_claim_checked_against_citations(asset):
    truth = shortfall_truth(asset)
    text = "No reconciling item explains the shortfall."
    assert outcome(run(truth, text)) == (("FM-001",), False)
    assert outcome(run(truth, text, reconciling_items=[])) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_negated_full_explanation_routes_to_review(asset):
    assert outcome(run(fully_explained_truth(asset), "The shortfall is not fully explained.")) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_explicit_plus_sign_on_shortfall_routes_to_review(asset):
    assert outcome(run(shortfall_truth(asset), "The variance of +$120,000 was reviewed.")) == ((), True)
    assert outcome(run(surplus_truth(asset), "The variance of +$120,000 was reviewed.")) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", ["The shortfall is EUR 120,000.", "The shortfall is €120,000.", "The shortfall is 120,000 GBP."])
def test_non_usd_amount_routes_to_review(asset, text):
    result = run(shortfall_truth(asset), text)
    assert result.failure_modes == () and result.review_reasons[0].startswith("non_usd_amount")


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_comparison_with_materiality_routes_to_review(asset):
    assert outcome(run(shortfall_truth(asset), "The variance is below materiality.")) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "On-chain reserves and ledger liabilities were $9,880,000 and $10,000,000.",
    "On-chain reserves and ledger liabilities were $10,000,000 and $9,880,000.",
])
def test_parallel_amounts_route_to_review(asset, text):
    """Without "respectively" the order is unknowable, so neither order fires or passes."""
    assert outcome(run(shortfall_truth(asset), text)) == ((), True)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "The ledger exceeds reserves by $120,000.",
    "On-chain reserves trailed the ledger.",
    "There is an excess of $120,000 in liabilities.",
    "Reserves were under-reserved against the books.",
])
def test_shortfall_phrasings_contradict_a_surplus(asset, text):
    assert outcome(run(surplus_truth(asset), text)) == (("FM-003",), False)
    assert outcome(run(shortfall_truth(asset), text)) == ((), False)
