"""Dataset tests — reproducibility and ground-truth integrity.

Two properties matter most here. (1) Reproducibility: a fixed seed yields a
byte-identical set, so a pinned run is re-runnable. (2) Self-consistency: the
band recorded in each sample's ground truth equals the band the policy assigns to
its own variance — if the generator and the policy ever disagree, every downstream
correctness score is measured against a broken answer key.
"""

import pytest

from icfr_agent_evals.dataset import ASSETS, build_dataset
from icfr_agent_evals.policy import materiality_band


def _truths(ds):
    return [s.metadata["truth"] for s in ds.samples]


def test_reproducible_ids_and_truth():
    a = build_dataset(split="eval", seed=20260607)
    b = build_dataset(split="eval", seed=20260607)
    assert [s.id for s in a.samples] == [s.id for s in b.samples]
    assert _truths(a) == _truths(b)


def test_default_sizes():
    assert len(build_dataset(split="eval").samples) == 20
    assert len(build_dataset(split="dev").samples) == 30


def test_ground_truth_band_is_self_consistent():
    for t in _truths(build_dataset(split="eval")):
        assert materiality_band(t["variance_usd"]) == t["materiality_band"]
    for t in _truths(build_dataset(split="dev")):
        assert materiality_band(t["variance_usd"]) == t["materiality_band"]


def test_direction_matches_sign():
    for t in _truths(build_dataset(split="eval")):
        if t["direction"] == "surplus":
            assert t["variance_usd"] > 0
        else:
            assert t["variance_usd"] < 0


def test_all_bands_covered_in_eval():
    bands = {t["materiality_band"] for t in _truths(build_dataset(split="eval"))}
    assert bands == {"M1", "M2", "M3"}


def test_assets_are_in_scope():
    for t in _truths(build_dataset(split="eval")):
        assert t["asset"] in ASSETS


def test_each_sample_has_two_valid_entities():
    for t in _truths(build_dataset(split="eval")):
        ents = t["valid_entities"]
        assert len(ents) == 2 and all(isinstance(e, str) and e for e in ents)


def test_dev_and_eval_are_disjoint():
    """Different RNG streams -> the dev set (used for prompt elicitation) shares no
    record with the evaluation set. Prevents tuning on the test set."""
    eval_prompts = {s.input for s in build_dataset(split="eval").samples}
    dev_prompts = {s.input for s in build_dataset(split="dev").samples}
    assert eval_prompts.isdisjoint(dev_prompts)
