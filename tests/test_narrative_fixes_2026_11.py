"""Regression tests for the narrative-extractor findings of the 2026-11-05 pre-pilot
review (fix lane W3): F3-01, F3-02, F3-04 (owner default D9), OpenAI R1 and R3.

Each test runs on all seven in-scope assets. A wrong statement must keep firing a
failure mode or route to requires_human_review; every fix here adds a review route
or corrects a role binding, and none turns a caught contradiction into a pass.
"""

from __future__ import annotations

import pytest
from casefactory import ALL_ASSETS, good_output, item, make_truth

from icfr_agent_evals.casefile import period_cutoff_date
from icfr_agent_evals.narrative import check_narrative_consistency


def run(truth, text, output=None):
    out = output if output is not None else good_output(truth, narrative=text)
    out["narrative"] = text
    return check_narrative_consistency(out, truth)


def outcome(result):
    return result.failure_modes, bool(result.review_reasons)


def reasons(result) -> list[str]:
    return [r.split(":", 1)[0] for r in result.review_reasons]


def U(x: float) -> str:
    return f"USD {abs(x):,.2f}"


def shortfall(asset):  # -120,000 M2; WD-2512-04417 explains 80,000
    return make_truth(asset)


def surplus(asset):  # +120,000 M2; DP-2512-00311 explains 80,000
    cutoff = period_cutoff_date("2025-12")
    items = (item("DP-2512-00311", 80_000.00, asset=asset, date=cutoff, recorded_in="on_chain", flow="inflow"),)
    return make_truth(asset, ledger=10_000_000.00, onchain=10_120_000.00, items=items)


def two_items(asset):  # -120,000; two eligible outflows of 80,000 and 30,000 explain 110,000
    cutoff = period_cutoff_date("2025-12")
    items = (
        item("WD-2512-04417", 80_000.00, asset=asset, date=cutoff),
        item("WD-2512-04418", 30_000.00, asset=asset, date=cutoff),
    )
    return make_truth(asset, items=items)


BALANCE_CASES = [shortfall, surplus]


def verb(truth) -> str:
    return "exceeded" if truth.direction == "surplus" else "fell short of"


# --- F3-01: nearest-cue binding at distance 1 or more takes the wrong balance cue ---

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", BALANCE_CASES)
@pytest.mark.parametrize("form", [
    "On-chain reserves with {c} were {r} against ledger liabilities of {l}.",
    "On-chain reserves held with {c} stood at {r} against ledger liabilities of {l}.",
    "On-chain reserves across the in-scope wallets of {r} {v} ledger liabilities of {l}.",
])
def test_true_balances_after_a_custodian_or_scope_phrase_route_to_review(asset, case, form):
    truth = case(asset)
    text = form.format(c=truth.custodian, r=U(truth.onchain_reserves_usd), l=U(truth.ledger_liabilities_usd), v=verb(truth))
    result = run(truth, text)
    assert outcome(result) == ((), True)
    assert {"role_amount_fits_competing_cue", "variance_amount_fits_other_role"} & set(reasons(result))


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", BALANCE_CASES)
def test_plain_balance_sentence_passes(asset, case):
    truth = case(asset)
    text = f"On-chain reserves were {U(truth.onchain_reserves_usd)} against ledger liabilities of {U(truth.ledger_liabilities_usd)}."
    assert outcome(run(truth, text)) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", BALANCE_CASES)
@pytest.mark.parametrize("form", [
    "On-chain reserves were {l} against ledger liabilities of {r}.",
    "On-chain reserves with {c} were {l} against ledger liabilities of {r}.",
    "On-chain reserves held with {c} stood at {l} against ledger liabilities of {r}.",
])
def test_swapped_balances_still_fire_fm001(asset, case, form):
    truth = case(asset)
    text = form.format(c=truth.custodian, r=U(truth.onchain_reserves_usd), l=U(truth.ledger_liabilities_usd))
    assert "FM-001" in run(truth, text).failure_modes


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_balance_binding_equal_to_the_variance_routes_to_review(asset):
    """V2's extension: no competing cue in the clause, the amount is |variance|."""
    truth = surplus(asset)
    result = run(truth, f"Reserves are ahead of ledger liabilities by {U(truth.variance_usd)}.")
    assert outcome(result) == ((), True)
    assert "balance_amount_equals_variance" in reasons(result)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "Ledger liabilities, as reported, were $9,880,000.",  # the reserve figure, no competing cue
    "On-chain reserves with BitGo Trust were USD 10,000,000.00.",  # the ledger figure, no ledger cue
    "Reserves are ahead of ledger liabilities by USD 150,000.00.",  # neither the variance nor a balance
])
def test_wrong_balance_without_a_competing_fit_keeps_fm001(asset, text):
    assert outcome(run(shortfall(asset), text)) == (("FM-001",), False)


# --- F3-02: "REF (type, USD X)" lists bind each amount to its reference -------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "Open items WD-2512-04417 (pending transfer, USD 80,000.00), WD-2512-04418 (timing difference, USD 30,000.00) "
    "explain USD 110,000.00 of it.",
    "Open items WD-2512-04417 (pending transfer, USD 80,000.00) and WD-2512-04418 (pending transfer, USD 30,000.00) "
    "explain USD 110,000.00 of the shortfall.",
    "WD-2512-04417 (USD 80,000.00), WD-2512-04418 (USD 30,000.00) explain USD 110,000.00.",
])
def test_parenthetical_item_list_passes(asset, text):
    result = run(two_items(asset), text)
    assert outcome(result) == ((), False)
    assert [e["resolution"] for e in result.extractions if e["kind"] == "amount"][:2] == [
        "parenthetical_of_reference", "parenthetical_of_reference"]


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "Open items WD-2512-04417 (pending transfer, USD 70,000.00), WD-2512-04418 (timing difference, USD 30,000.00) "
    "explain USD 110,000.00 of it.",
    "Open items WD-2512-04417 (pending transfer, USD 80,000.00), WD-2512-04418 (timing difference, USD 35,000.00) "
    "explain USD 110,000.00 of it.",
])
def test_parenthetical_wrong_item_amount_fires_fm001(asset, text):
    assert "FM-001" in run(two_items(asset), text).failure_modes


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_parenthetical_amount_of_an_uncited_reference_fires_fm001(asset):
    truth = two_items(asset)
    out = good_output(truth)
    out["reconciling_items"] = [r for r in out["reconciling_items"] if r["reference"] == "WD-2512-04417"]
    text = "Open items WD-2512-04417 (pending transfer, USD 80,000.00), WD-2512-04418 (pending transfer, USD 30,000.00) are open."
    assert "FM-001" in run(truth, text, out).failure_modes


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_parenthetical_amount_of_an_ineligible_reference_fires_fm001(asset):
    text = "Items WD-2512-04417 (pending transfer, USD 80,000.00), JE-25-12-0193 (manual adjustment, USD 40,000.00) are listed."
    assert "FM-001" in run(shortfall(asset), text).failure_modes


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_role_cue_inside_the_parenthetical_keeps_the_usual_binding(asset):
    """"WD-1 (the shortfall, $80,000)" states the shortfall, not the item: it never
    passes as the item's amount (review, since 80,000 fits the item)."""
    result = run(shortfall(asset), "WD-2512-04417 (the shortfall, USD 80,000.00) is pending.")
    assert outcome(result) == ((), True)
    assert all(e.get("resolution") != "parenthetical_of_reference" for e in result.extractions)
    assert "FM-001" in run(shortfall(asset), "WD-2512-04417 (the shortfall, USD 95,000.00) is pending.").failure_modes


# --- OpenAI R1: the clause-sum fallback needs an explicit aggregate word -------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_item_amount_equal_to_the_clause_sum_without_aggregate_word_routes_to_review(asset):
    result = run(two_items(asset), "WD-2512-04417 is USD 110,000.00 and WD-2512-04418 is USD 30,000.00.")
    assert outcome(result) == ((), True)
    assert "item_amount_equals_clause_sum" in reasons(result)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", [
    "WD-2512-04417 and WD-2512-04418 together explain USD 110,000.00.",
    "WD-2512-04417 and WD-2512-04418, totaling USD 110,000.00, are open.",
    "WD-2512-04417 and WD-2512-04418 are open, USD 110,000.00 in total.",
    "WD-2512-04417 is USD 80,000.00 and WD-2512-04418 is USD 30,000.00.",
])
def test_stated_sum_with_aggregate_word_and_individual_amounts_pass(asset, text):
    assert outcome(run(two_items(asset), text)) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("text", ["WD-2512-04417 is USD 90,000.00.", "WD-2512-04417 is USD 110,000.00."])
def test_wrong_single_item_amount_fires_fm001(asset, text):
    assert outcome(run(two_items(asset), text)) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_semicolon_list_total_equal_to_explained_routes_to_review(asset):
    """Extension: the stated total lands in the last item's clause after a semicolon."""
    truth = two_items(asset)
    ok = run(truth, "Items relied on: WD-2512-04417 80.0k; WD-2512-04418 30.0k, together $110,000.00.")
    assert outcome(ok) == ((), True)
    assert "item_amount_equals_explained_total" in reasons(ok)
    wrong = run(truth, "Items relied on: WD-2512-04417 80.0k; WD-2512-04418 30.0k, together $150,000.00.")
    assert "FM-001" in wrong.failure_modes


# --- OpenAI R3: signed suffixed amounts keep their sign ------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case,text", [
    (shortfall, "The variance is +120k."),
    (surplus, "The variance is -120k."),
    (surplus, "The variance is −120k."),
    (shortfall, "The variance is + 120k."),
])
def test_wrong_sign_on_suffixed_amount_routes_to_review(asset, case, text):
    result = run(case(asset), text)
    assert outcome(result) == ((), True)
    assert "amount_sign_disagrees" in reasons(result)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case,text", [
    (shortfall, "The variance is 120k."),
    (surplus, "The variance is 120k."),
    (shortfall, "The variance is -120k."),
    (surplus, "The variance is +120k."),
])
def test_right_or_no_sign_on_suffixed_amount_passes(asset, case, text):
    assert outcome(run(case(asset), text)) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_negative_suffixed_amount_is_extracted_and_checked(asset):
    assert outcome(run(shortfall(asset), "The shortfall is -10k.")) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_dash_in_identifier_or_range_is_not_a_sign(asset):
    result = run(shortfall(asset), "Batch FY-120k is closed. The band runs from 25k - 250k.")
    amounts = [e for e in result.extractions if e["kind"] == "amount"]
    assert all(e["value"] != 120_000 for e in amounts)
    assert not any(e["negative"] for e in amounts)


# --- F3-04 under D9: hedged or suffix-rounded figures within stated precision --------

def short_62k(asset):  # -62,878.92 M2, no items
    return make_truth(asset, ledger=10_000_000.00, onchain=9_937_121.08, items=())


def surplus_1m(asset):  # +1,103,947.93 M3, no items
    return make_truth(asset, ledger=10_000_000.00, onchain=11_103_947.93, items=())


VARIANCE_CASES = [shortfall, short_62k, surplus_1m]


def k_or_m(x: float, scale: float = 1.0) -> str:
    x = abs(x) * scale
    return f"{x / 1e6:.1f} million" if x >= 1e6 else f"{x / 1e3:.1f}k"


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", VARIANCE_CASES)
@pytest.mark.parametrize("form", [
    "The variance is a {d} of approximately USD {k}.",
    "The variance is a {d} of roughly USD {k}.",
    "The variance is a {d} of ~USD {k}.",
    "The variance is a {d} of USD {k}.",
    "The variance is a {d} of about USD {r}.",
])
def test_hedged_figure_within_stated_precision_routes_to_review(asset, case, form):
    truth = case(asset)
    v = truth.variance_usd
    text = form.format(d=truth.direction, k=k_or_m(v), r=f"{round(abs(v) * 0.97, -3):,.0f}")
    result = run(truth, text)
    if abs(round(abs(v) / 1e3, 1) * 1e3 - abs(v)) <= 1 and "{k}" in form:
        assert outcome(result) == ((), False)  # the rounded figure is exact within USD 1
    else:
        assert outcome(result) == ((), True)
        assert "hedged_amount" in reasons(result)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", VARIANCE_CASES)
@pytest.mark.parametrize("form", [
    "The variance is a {d} of approximately USD {k2}.",
    "The variance is a {d} of USD {k2}.",
    "The variance is a {d} of about USD {r}.",
])
def test_hedged_figure_outside_stated_precision_fires_fm001(asset, case, form):
    truth = case(asset)
    v = truth.variance_usd
    text = form.format(d=truth.direction, k2=k_or_m(v, 2.0), r=f"{round(abs(v) * 0.9, -3):,.0f}")
    assert outcome(run(truth, text)) == (("FM-001",), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
@pytest.mark.parametrize("case", VARIANCE_CASES)
def test_exact_figure_within_one_dollar_passes(asset, case):
    truth = case(asset)
    v = abs(truth.variance_usd)
    for text in (f"The variance is a {truth.direction} of {U(v)}.",
                 f"The variance is a {truth.direction} of approximately USD {round(v):,}."):
        assert outcome(run(truth, text)) == ((), False)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_hedged_item_amount_within_five_percent_routes_to_review(asset):
    truth = shortfall(asset)
    near = run(truth, "WD-2512-04417 for about USD 79,000 is pending.")
    assert outcome(near) == ((), True)
    assert "hedged_amount" in reasons(near)
    assert outcome(run(truth, "WD-2512-04417 for about USD 60,000 is pending.")) == (("FM-001",), False)
