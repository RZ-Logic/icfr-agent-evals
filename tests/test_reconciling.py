"""Reconciling-items anchor tests (pre-registration section 6, FM-015; design point D1).

An absent, ineligible (AC-6 decoy included), wrong-amount or repeated citation is
FM-015 and scores 0. Otherwise the explained split is computed by code from the
cited items: matching the truth scores 3; omitting an eligible item scores 2 when
at least one was cited and 1 when none was, with no FM code. Score levels are
provisional until rubric lock (section 13).
"""

import pytest
from casefactory import ALL_ASSETS, good_output, item, make_truth, other_wallet, tx_hash

from icfr_agent_evals.scoring import check_reconciling_items


def _cite(*pairs):
    return [{"reference": ref, "amount_usd": amt} for ref, amt in pairs]


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_correct_citation_scores_three(asset):
    t = make_truth(asset)
    r = check_reconciling_items(good_output(t), t)
    assert r.points == 3 and r.failure_modes == ()
    assert r.details["explained_usd"] == -80_000.00 and r.details["unexplained_usd"] == -40_000.00


def test_absent_reference_is_fm015():
    t = make_truth()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("WD-2512-99999", 80_000.0))}, t)
    assert r.points == 0 and r.failure_modes == ("FM-015",)


def test_reference_compared_exactly():
    t = make_truth()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("wd-2512-04417", 80_000.0))}, t)
    assert r.failure_modes == ("FM-015",)


def test_cleared_decoy_is_fm015():
    t = make_truth()
    out = {**good_output(t), "reconciling_items": _cite(("WD-2512-04417", 80_000.0), ("JE-25-12-0193", 40_000.0))}
    r = check_reconciling_items(out, t)
    assert r.points == 0 and r.failure_modes == ("FM-015",)
    assert any("ineligible" in p for p in r.details["problems"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"open_status": "cleared", "recorded_in": "both"},
        {"asset": "ETH"},
        {"wallet": "OTHER"},
        {"date": "2026-01-02"},
        {"type": "suspense_clearing"},
    ],
)
def test_ac6_decoys_summing_exactly_to_the_variance_are_fm015(overrides):
    """AC-6: real items that fail eligibility sum exactly to the variance."""
    kwargs = {"role": "decoy", **overrides}
    if kwargs.get("wallet") == "OTHER":
        kwargs["wallet"] = other_wallet("BTC")
    decoys = (item("WD-2512-70001", 60_000.00, **kwargs), item("WD-2512-70002", 60_000.00, **kwargs))
    t = make_truth(items=decoys)
    assert t.eligible_items() == () and t.explained_usd == 0.0
    out = {**good_output(t), "reconciling_items": _cite(("WD-2512-70001", 60_000.0), ("WD-2512-70002", 60_000.0))}
    r = check_reconciling_items(out, t)
    assert r.points == 0 and r.failure_modes == ("FM-015",)


def test_ac6_no_eligible_item_and_none_cited_scores_three():
    decoys = (item("WD-2512-70001", 120_000.00, open_status="cleared", recorded_in="both", role="decoy"),)
    t = make_truth(items=decoys)
    r = check_reconciling_items({**good_output(t), "reconciling_items": []}, t)
    assert r.points == 3 and r.failure_modes == ()


@pytest.mark.parametrize("amount", [79_998.0, 80_001.5, 8_000.0, 0.0])
def test_amount_mismatch_beyond_tolerance_is_fm015(amount):
    t = make_truth()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("WD-2512-04417", amount))}, t)
    assert r.failure_modes == ("FM-015",)


@pytest.mark.parametrize("amount", [80_000.0, 80_000.99, 79_999.01, -80_000.0])
def test_amount_within_tolerance_and_signed_form_accepted(amount):
    t = make_truth()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("WD-2512-04417", amount))}, t)
    assert r.points == 3 and r.failure_modes == ()


def test_same_item_cited_twice_is_fm015():
    t = make_truth()
    out = {**good_output(t), "reconciling_items": _cite(("WD-2512-04417", 80_000.0), ("WD-2512-04417", 80_000.0))}
    assert check_reconciling_items(out, t).failure_modes == ("FM-015",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_item_cited_by_reference_and_alias_is_fm015(asset):
    t = make_truth(asset)
    out = {**good_output(t), "reconciling_items": _cite(("WD-2512-04417", 80_000.0), (tx_hash(asset), 80_000.0))}
    assert check_reconciling_items(out, t).failure_modes == ("FM-015",)


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_item_cited_by_alias_alone_resolves(asset):
    t = make_truth(asset)
    out = {**good_output(t), "reconciling_items": _cite((tx_hash(asset), 80_000.0))}
    assert check_reconciling_items(out, t).points == 3


def _two_explaining():
    items = (
        item("WD-2512-04417", 80_000.00),
        item("WD-2512-04418", 30_000.00, channel="memo_only"),
    )
    return make_truth(items=items)


def test_omitting_one_eligible_item_scores_two_without_fm_code():
    t = _two_explaining()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("WD-2512-04417", 80_000.0))}, t)
    assert r.points == 2 and r.failure_modes == () and r.details["omitted"] == ["WD-2512-04418"]


def test_citing_no_item_when_eligible_items_exist_scores_one():
    t = _two_explaining()
    r = check_reconciling_items({**good_output(t), "reconciling_items": []}, t)
    assert r.points == 1 and r.failure_modes == ()


def test_full_citation_of_memo_only_item_scores_three():
    t = _two_explaining()
    r = check_reconciling_items(good_output(t), t)
    assert r.points == 3 and r.details["explained_usd"] == -110_000.00


def test_type_label_mismatch_routes_to_review_and_keeps_score():
    t = make_truth()
    out = {**good_output(t), "reconciling_items": [{"reference": "WD-2512-04417", "amount_usd": 80_000.0, "type": "fee"}]}
    r = check_reconciling_items(out, t)
    assert r.points == 3 and r.failure_modes == () and r.requires_human_review


@pytest.mark.parametrize("label", ["pending_transfer", "Pending transfer", "pending-transfer", "PENDING_TRANSFER"])
def test_type_label_normalized(label):
    t = make_truth()
    out = {**good_output(t), "reconciling_items": [{"reference": "WD-2512-04417", "amount_usd": 80_000.0, "type": label}]}
    assert not check_reconciling_items(out, t).requires_human_review


def test_explained_split_ignores_the_narrative():
    """The split is computed by code from the cited items, never from prose."""
    t = make_truth()
    out = good_output(t, narrative="The whole variance is explained by WD-2512-04417.")
    r = check_reconciling_items(out, t)
    assert r.details["unexplained_usd"] == -40_000.00


def test_fm015_instance_is_listed_for_review():
    t = make_truth()
    r = check_reconciling_items({**good_output(t), "reconciling_items": _cite(("NOPE-0000-01", 1.0))}, t)
    assert r.requires_human_review
