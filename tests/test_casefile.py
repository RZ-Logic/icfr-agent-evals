"""Contract tests: eligibility, effect sign, answer-key validation, serialization.

These pin the shared shapes every generator, scorer and baseline codes against.
"""

import dataclasses

import pytest
from casefactory import ALL_ASSETS, item, make_truth, other_wallet, scope_wallet, tx_hash

from icfr_agent_evals.casefile import (
    ADDRESS_FORMATS,
    ASSET_CHAIN,
    EVAL_SEED,
    SPLITS,
    TX_HASH_FORMATS,
    CaseTruth,
    TruthError,
    dataset_sha256,
    explained_from_citations,
    item_effect_usd,
    item_eligibility,
    matches_format,
    period_cutoff_date,
    split_seed,
    validate_truth,
    visible_structured,
)


@pytest.mark.parametrize(
    "recorded_in,flow,sign",
    [("on_chain", "inflow", 1), ("ledger", "outflow", 1), ("on_chain", "outflow", -1), ("ledger", "inflow", -1)],
)
def test_effect_sign_convention(recorded_in, flow, sign):
    it = item("WD-2512-00001", 1_000.00, recorded_in=recorded_in, flow=flow)
    assert item_effect_usd(it) == sign * 1_000.00


def test_effect_zero_when_both_or_missing():
    assert item_effect_usd(item("WD-2512-00001", 1_000.00, recorded_in="both", open_status="cleared")) == 0.0
    assert item_effect_usd(item("WD-2512-00001", None)) == 0.0


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"open_status": "cleared", "recorded_in": "both"}, "not_open"),
        ({"asset": "ETH"}, "wrong_asset"),
        ({"wallet": "bc1q09z23pa2tyjxzqhltaa5dfx0wkcdmxf69rlph9"}, "out_of_scope_wallet"),
        ({"date": "2026-01-02"}, "after_cutoff"),
        ({"type": "manual_adjustment"}, "disallowed_type"),
        ({"amount_usd": None}, "amount_missing"),
    ],
)
def test_eligibility_reasons(overrides, reason):
    base = item("WD-2512-00001", 1_000.00)
    it = dataclasses.replace(base, **overrides)
    e = item_eligibility(it, asset="BTC", scope_wallets=(scope_wallet("BTC"),), cutoff_date="2025-12-31")
    assert not e.eligible and reason in e.reasons


def test_eligible_item_dated_before_period_is_still_eligible():
    """An aged item still open at the cutoff explains the variance."""
    it = item("WD-2511-00001", 1_000.00, date="2025-11-03")
    e = item_eligibility(it, asset="BTC", scope_wallets=(scope_wallet("BTC"),), cutoff_date="2025-12-31")
    assert e.eligible


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_factory_truth_valid_for_every_asset(asset):
    t = make_truth(asset)
    assert t.variance_usd == -120_000.00 and t.materiality_band == "M2"
    assert t.explained_usd == -80_000.00 and t.unexplained_usd == -40_000.00
    assert t.evidence_state == "partly_unexplained"
    chain = ASSET_CHAIN[asset]
    assert matches_format(scope_wallet(asset), ADDRESS_FORMATS[chain])
    assert matches_format(other_wallet(asset), ADDRESS_FORMATS[chain])
    assert matches_format(tx_hash(asset), TX_HASH_FORMATS[chain])


def test_truth_round_trips_through_dict():
    t = make_truth("SOL")
    assert CaseTruth.from_dict(t.to_dict()) == t


def test_validate_truth_rejects_wrong_band():
    t = make_truth()
    bad = dataclasses.replace(t, materiality_band="M3")
    with pytest.raises(TruthError):
        validate_truth(bad)


def test_validate_truth_rejects_eligible_decoy():
    items = (item("WD-2512-04417", 80_000.00, role="decoy"),)
    with pytest.raises(TruthError):
        make_truth(items=items)


def test_validate_truth_rejects_memo_effects_that_net_to_zero():
    items = (
        item("WD-2512-00001", 5_000.00, channel="memo_only", recorded_in="on_chain", flow="outflow"),
        item("DP-2512-00002", 5_000.00, channel="memo_only", recorded_in="on_chain", flow="inflow"),
    )
    with pytest.raises(TruthError):
        make_truth(items=items)


def test_explained_from_citations_counts_each_item_once_and_resolves_alias():
    t = make_truth()
    assert explained_from_citations(["WD-2512-04417"], t) == (-80_000.00, -40_000.00)
    assert explained_from_citations(["WD-2512-04417", tx_hash("BTC")], t) == (-80_000.00, -40_000.00)
    assert explained_from_citations(["JE-25-12-0193"], t) == (0.0, -120_000.00)  # ineligible decoy
    assert explained_from_citations(["NOPE-0000-01"], t) == (0.0, -120_000.00)


def test_visible_structured_hides_memo_only_items_and_roles():
    items = (
        item("WD-2512-04417", 80_000.00),
        item("WD-2512-04418", 30_000.00, channel="memo_only"),
    )
    v = visible_structured(make_truth(items=items))
    assert [i.reference for i in v.open_items] == ["WD-2512-04417"]
    assert all(i.role == "noise" for i in v.open_items)


def test_split_seeds_are_distinct_and_eval_is_pinned():
    seeds = {s: split_seed(s) for s in SPLITS}
    assert seeds["eval"] == EVAL_SEED == 20260607
    assert len(set(seeds.values())) == len(SPLITS)


def test_period_cutoff_date():
    assert period_cutoff_date("2025-11") == "2025-11-30"
    assert period_cutoff_date("2025-12") == "2025-12-31"


def test_dataset_sha256_is_order_independent():
    from inspect_ai.dataset import Sample

    a = [Sample(input="x", id="1", metadata={"k": 1}), Sample(input="y", id="2", metadata={"k": 2})]
    assert dataset_sha256(a) == dataset_sha256(list(reversed(a)))
    b = [Sample(input="x", id="1", metadata={"k": 1}), Sample(input="y", id="2", metadata={"k": 3})]
    assert dataset_sha256(a) != dataset_sha256(b)
