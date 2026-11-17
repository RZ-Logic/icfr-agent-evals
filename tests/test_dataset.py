"""Dataset tests: reproducibility, ground-truth integrity and split composition.

Two properties matter most here. (1) Reproducibility: a fixed seed yields a
byte-identical set, so a pinned run is re-runnable. (2) Self-consistency: the
band recorded in each sample's ground truth equals the band the policy assigns to
its own variance; if the generator and the policy ever disagree, every downstream
correctness score is measured against a broken answer key.
"""

from collections import Counter
from functools import lru_cache

import pytest

from icfr_agent_evals.casefile import BANDS, DIRECTIONS, PERIODS, matches_format, split_seed, truth_of
from icfr_agent_evals.dataset import (
    ASSETS,
    CI_HARD_COMPOSITION,
    CUSTODIANS,
    DEV_HARD_COMPOSITION,
    STANDARD_CASES,
    build_cases,
    build_dataset,
)
from icfr_agent_evals.hardcases import build_hard_cases
from icfr_agent_evals.policy import materiality_band


@lru_cache(maxsize=None)
def _ds(split):
    return build_dataset(split=split)


def _truths(ds):
    return [s.metadata["truth"] for s in ds.samples]


def _standard(split):
    return [truth_of(s.metadata) for s in _ds(split).samples if s.metadata["truth"]["case_kind"] == "standard"]


def test_reproducible_ids_and_truth():
    a = build_dataset(split="eval", seed=20260607)
    b = _ds("eval")
    assert [s.id for s in a.samples] == [s.id for s in b.samples]
    assert _truths(a) == _truths(b)
    assert [s.input for s in a.samples] == [s.input for s in b.samples]


def test_default_sizes():
    """Pre-registration section 4: the evaluation split's 146 synthetic cases come
    from the generator (its 54 hand-written cases are written after the tag); the
    dev split is 30 cases; the CI regression split is 146 standard plus 54 generated
    hard cases. Hard cases come from hardcases.build_hard_cases."""
    assert len(_ds("eval").samples) == STANDARD_CASES == 146
    assert len(_ds("dev").samples) == 30 == 14 + sum(DEV_HARD_COMPOSITION.values())
    assert len(_ds("ci").samples) == 200 == 146 + sum(CI_HARD_COMPOSITION.values())
    assert len(_standard("dev")) == 14 and len(_standard("ci")) == 146


@pytest.mark.parametrize("split,composition", [("ci", CI_HARD_COMPOSITION), ("dev", DEV_HARD_COMPOSITION)])
def test_hard_case_composition_by_category(split, composition):
    """Section 4: the CI split's 54 hard cases are 30 edge-type (10 per edge template)
    and 24 adversarial-type (AC-1 to AC-6, 4 each); dev takes 16 by DEV_HARD_COMPOSITION."""
    ts = [truth_of(s.metadata) for s in _ds(split).samples if s.metadata["truth"]["case_kind"] != "standard"]
    assert Counter(t.category for t in ts) == composition
    kinds = Counter(t.case_kind for t in ts)
    edge = sum(composition[c] for c in ("band_boundary", "ambiguous_narrative", "incomplete_data"))
    assert kinds == {"edge": edge, "adversarial": len(ts) - edge}
    if split == "ci":
        assert kinds == {"edge": 30, "adversarial": 24}
        assert all(composition[f"AC-{i}"] == 4 for i in range(1, 7))
    hard = build_hard_cases(split, split_seed(split), composition)
    assert sorted(c.truth.case_id for c in hard) == sorted(t.case_id for t in ts)


def test_dev_pilot_subset_is_20():
    """8 standard pilot cases (D4) plus the 6 dev edge cases and one dev case per AC
    category, the 20-case pilot of section 13."""
    pilot = [truth_of(s.metadata) for s in _ds("dev").samples if s.metadata["truth"]["notes"].get("pilot")]
    assert len(pilot) == 20
    assert Counter(t.case_kind for t in pilot) == {"standard": 8, "edge": 6, "adversarial": 6}
    assert {t.category for t in pilot if t.case_kind == "adversarial"} == {f"AC-{i}" for i in range(1, 7)}


def test_ground_truth_band_is_self_consistent():
    for split in ("eval", "dev", "ci"):
        for t in _truths(_ds(split)):
            assert materiality_band(t["variance_usd"]) == t["materiality_band"]


def test_direction_matches_sign():
    for t in _truths(_ds("eval")):
        if t["direction"] == "surplus":
            assert t["variance_usd"] > 0
        else:
            assert t["variance_usd"] < 0


def test_all_bands_covered_in_eval():
    bands = {t["materiality_band"] for t in _truths(_ds("eval"))}
    assert bands == {"M1", "M2", "M3"}


def test_assets_are_in_scope():
    for t in _truths(_ds("eval")):
        assert t["asset"] in ASSETS


def test_every_cited_entity_class_is_in_valid_entities():
    """Replaces the two-entity check of the five-line record. Section 4: every
    item's reference joins the reference-number set the fabrication anchor checks;
    section 6: counterparties, wallet ids and references in entities_cited must
    appear in the fixture. So valid_entities holds the custodian, every wallet,
    every item reference and alias, and only strings the rendered case shows."""
    for s in _ds("eval").samples:
        t = truth_of(s.metadata)
        ents = set(t.valid_entities)
        assert t.custodian in ents and set(t.scope_wallets) <= ents
        for it in t.items:
            assert it.reference in ents and set(it.aliases) <= ents
        for e in t.valid_entities:
            assert isinstance(e, str) and e and e in s.input


def test_dev_and_eval_are_disjoint():
    """Different RNG streams -> the dev set (used for prompt elicitation) shares no
    record with the evaluation set. Prevents tuning on the test set."""
    eval_prompts = {s.input for s in _ds("eval").samples}
    dev_prompts = {s.input for s in _ds("dev").samples}
    assert eval_prompts.isdisjoint(dev_prompts)


@pytest.mark.parametrize("split", ["eval", "ci"])
def test_standard_split_has_full_factorial_coverage(split):
    ts = _standard(split)
    assert len(ts) == 146
    cells = Counter((t.asset, t.materiality_band, t.period, t.direction) for t in ts)
    assert len(cells) == len(ASSETS) * len(BANDS) * len(PERIODS) * len(DIRECTIONS) == 126
    # asset and band no longer share one rotation with period
    for band in BANDS:
        assert {t.period for t in ts if t.materiality_band == band} == set(PERIODS)


@pytest.mark.parametrize("split", ["eval", "ci"])
def test_memo_only_cases_are_balanced(split):
    ts = _standard(split)
    memo = [t for t in ts if t.has_memo_only_link]
    assert 0.20 <= len(memo) / len(ts) <= 0.30
    assert Counter(t.asset for t in memo) == {a: 6 for a in ASSETS}
    assert Counter(t.materiality_band for t in memo) == {b: 14 for b in BANDS}
    assert Counter(t.direction for t in memo) == {d: 21 for d in DIRECTIONS}


@pytest.mark.parametrize("split", ["eval", "ci"])
def test_evidence_mix(split):
    ts = _standard(split)
    mix = Counter(t.evidence_state for t in ts)
    assert mix == {"fully_explained": 73, "partly_unexplained": 49, "unexplained": 24}
    # decorrelated: every asset, band, period and direction holds every state
    for dim in ("asset", "materiality_band", "period", "direction"):
        for value in {getattr(t, dim) for t in ts}:
            assert {t.evidence_state for t in ts if getattr(t, dim) == value} == set(mix), (dim, value)


def test_dev_standard_composition():
    """Design point D4: 7 memo-only and 7 non-memo standard cases, one per asset in
    each group, every band in both groups; 4 + 4 pilot cases covering all bands."""
    ts = _standard("dev")
    memo = [t for t in ts if t.has_memo_only_link]
    plain = [t for t in ts if not t.has_memo_only_link]
    assert sorted(t.asset for t in memo) == sorted(ASSETS) == sorted(t.asset for t in plain)
    assert {t.materiality_band for t in memo} == set(BANDS) == {t.materiality_band for t in plain}
    assert len({t.period for t in ts}) == 3 and len({t.direction for t in ts}) == 2
    for group in (memo, plain):
        pilot = [t for t in group if t.notes.get("pilot")]
        assert len(pilot) == 4 and {t.materiality_band for t in pilot} == set(BANDS)


def test_case_ids_start_with_split_and_are_unique():
    for split in ("eval", "dev", "ci"):
        ids = [s.id for s in _ds(split).samples]
        assert len(ids) == len(set(ids))
        assert all(i.startswith(f"{split}-") for i in ids)


def test_n_returns_first_n_samples():
    head = build_dataset(split="dev", n=5)
    assert [s.id for s in head.samples] == [s.id for s in _ds("dev").samples[:5]]


def test_exports():
    assert CUSTODIANS[:5] == ["Anchorage Digital", "BitGo Trust", "Coinbase Custody", "Fireblocks", "Komainu"]
    assert ASSETS == ["BTC", "ETH", "SOL", "USDC", "USDT", "XRP", "ADA"]


def test_sample_input_is_the_rendered_case_and_target_the_band():
    for s in _ds("dev").samples:
        t = truth_of(s.metadata)
        assert s.target == t.materiality_band
        assert s.metadata["fixture"]["header"]["asset"] == t.asset
        assert not matches_format(s.input.split("\n")[0], r".*\b(eval|dev|ci)-std-.*")
