"""The pilot subset (pre-registration section 13, pilot row): 20 dev-split cases.

The pilot is selected by the flag in each dev case's truth.notes, never by a list
kept beside the data. These tests pin the selection to the committed dev file, its
composition (4 memo-only standard, 4 other standard, the 6 dev edge cases, one case
per AC category), the ValueError cases, and its trip through a real Inspect run:
task_args and Task metadata in the EvalLog header must say the pilot ran.

Dev split only. The evaluation split is never built: every check that pairs the
pilot with another split runs with the build stubbed to fail, so the check is shown
to fire first. Runs use mockllm with usage set, or the baseline's none provider.
Set ICFR_TEST_LOG_DIR to choose where the runs write logs; pytest's temp dir is
the default, never the repo.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals import baseline
from icfr_agent_evals import dataset as dataset_module
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import dataset_sha256, truth_of, visible_structured
from icfr_agent_evals.dataset import PILOT_CASES, build_dataset, is_pilot, select_subset
from icfr_agent_evals.rubric import WIRED_SCORERS

ROOT = Path(__file__).resolve().parents[1]
DEV_FILE = ROOT / "data" / "dev" / "dev.jsonl"
COMPOSITION_FILE = ROOT / "data" / "dev" / "COMPOSITION.json"
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
EDGE = ("band_boundary", "ambiguous_narrative", "incomplete_data")

# The 20 pilot ids as the generator emits them today. A generator change that moves
# the pilot fails here and must be re-registered, not absorbed.
PILOT_IDS = {
    "dev-hc-001-BTC-M2", "dev-hc-002-USDT-M2", "dev-hc-003-ETH-M1", "dev-hc-005-USDT-M3",
    "dev-hc-007-SOL-M2", "dev-hc-008-USDC-M3", "dev-hc-009-XRP-M3", "dev-hc-010-XRP-M3",
    "dev-hc-011-ADA-M1", "dev-hc-013-ETH-M2", "dev-hc-014-USDC-M1", "dev-hc-015-SOL-M1",
    "dev-std-002-BTC-M1", "dev-std-004-SOL-M1", "dev-std-005-ETH-M2", "dev-std-007-ETH-M3",
    "dev-std-008-USDC-M1", "dev-std-010-BTC-M2", "dev-std-012-SOL-M3", "dev-std-013-USDC-M2",
}


def _log_dir(tmp_path_factory, label: str) -> str:
    root = os.environ.get("ICFR_TEST_LOG_DIR")
    if root:
        path = Path(root) / f"{label}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(tmp_path_factory.mktemp(label))


@pytest.fixture(scope="module")
def dev():
    return list(build_dataset(split="dev"))


@pytest.fixture(scope="module")
def pilot():
    return list(build_dataset(split="dev", subset="pilot"))


@pytest.fixture
def no_build(monkeypatch):
    """Any build of any split fails the test: a subset check must fire before it."""

    def refuse(*args, **kwargs):
        raise AssertionError("a split was built before the subset check")

    monkeypatch.setattr(dataset_module, "build_cases", refuse)
    monkeypatch.setattr(task_module, "build_dataset", refuse)


# --- Selection ------------------------------------------------------------------------

def test_pilot_selects_exactly_the_flagged_ids(dev, pilot):
    assert len(pilot) == PILOT_CASES == 20
    ids = {s.id for s in pilot}
    assert ids == PILOT_IDS
    assert ids == {s.id for s in dev if truth_of(s.metadata).notes.get("pilot") is True}


def test_pilot_keeps_sample_order_and_content(dev, pilot):
    flagged = [s for s in dev if is_pilot(s)]
    assert [s.id for s in pilot] == [s.id for s in flagged]
    assert dataset_sha256(pilot) == dataset_sha256(flagged)


def test_committed_dev_file_flags_the_same_ids(dev):
    rows = [json.loads(line) for line in DEV_FILE.read_text(encoding="utf-8").splitlines() if line]
    assert {r["id"] for r in rows} == {s.id for s in dev}
    assert {r["id"] for r in rows if r["metadata"]["truth"]["notes"].get("pilot") is True} == PILOT_IDS
    # every non-pilot row carries an explicit False, so no case is out by omission
    assert Counter(repr(r["metadata"]["truth"]["notes"].get("pilot")) for r in rows) == {"True": 20, "False": 10}


def test_pilot_composition(pilot):
    ts = [truth_of(s.metadata) for s in pilot]
    std = [t for t in ts if t.case_kind == "standard"]
    assert Counter(t.has_memo_only_link for t in std) == {True: 4, False: 4}
    assert Counter(t.case_kind for t in ts) == {"standard": 8, "edge": 6, "adversarial": 6}
    assert Counter(t.category for t in ts if t.case_kind == "edge") == dict.fromkeys(EDGE, 2)
    assert Counter(t.category for t in ts if t.case_kind == "adversarial") == {f"AC-{i}": 1 for i in range(1, 7)}


def test_every_dev_edge_case_is_in_the_pilot(dev):
    edge = {s.id for s in dev if truth_of(s.metadata).case_kind == "edge"}
    assert len(edge) == 6 and edge <= PILOT_IDS


def test_is_pilot_accepts_only_a_literal_true(pilot):
    s = pilot[0].model_copy(deep=True)
    for value in ("true", 1, "yes", None):
        s.metadata["truth"]["notes"]["pilot"] = value
        assert not is_pilot(s)
    del s.metadata["truth"]["notes"]["pilot"]
    assert not is_pilot(s)


def test_a_moved_pilot_fails_the_count_guard(dev):
    with pytest.raises(ValueError, match="expected 20"):
        select_subset([s for s in dev if s.id != sorted(PILOT_IDS)[0]], "dev", "pilot")


def test_no_subset_is_backward_compatible(dev):
    ds = build_dataset(split="dev")
    assert ds.name == "variance-commentary-dev" and len(ds) == 30
    assert build_dataset("dev", None, 20260607).name == "variance-commentary-dev"
    assert select_subset(dev, "dev", None) == dev
    head = build_dataset(split="dev", subset="pilot", n=5)
    assert [s.id for s in head] == [s.id for s in build_dataset(split="dev", subset="pilot")][:5]
    assert build_dataset(split="dev", subset="pilot").name == "variance-commentary-dev-pilot"


# --- ValueError cases -------------------------------------------------------------------

@pytest.mark.parametrize("split", ["eval", "ci", "calibration"])
def test_pilot_off_the_dev_split_raises_before_any_build(no_build, split):
    with pytest.raises(ValueError, match="only on split 'dev'"):
        build_dataset(split=split, subset="pilot")
    with pytest.raises(ValueError, match="only on split 'dev'"):
        task_module.variance_commentary(split=split, subset="pilot")
    with pytest.raises(ValueError, match="only on split 'dev'"):
        baseline.variance_commentary_template_baseline(split=split, subset="pilot")


@pytest.mark.parametrize("subset", ["", "Pilot", "PILOT", "all", "eval", "dev", "pilot "])
def test_unknown_subset_raises_before_any_build(no_build, subset):
    with pytest.raises(ValueError, match="unknown subset"):
        build_dataset(split="dev", subset=subset)
    with pytest.raises(ValueError, match="unknown subset"):
        task_module.variance_commentary(split="dev", subset=subset)
    with pytest.raises(ValueError, match="unknown subset"):
        baseline.variance_commentary_template_baseline(split="dev", subset=subset)


def test_unknown_subset_is_named_before_the_split_check(no_build):
    with pytest.raises(ValueError, match="unknown subset"):
        build_dataset(split="eval", subset="all")


# --- Task metadata -----------------------------------------------------------------------

def test_task_metadata_hashes_what_runs(dev, pilot):
    t = task_module.variance_commentary(split="dev", epochs=1, subset="pilot")
    assert {s.id for s in t.dataset} == PILOT_IDS
    assert t.dataset.name == "variance-commentary-dev-pilot"
    assert t.metadata["subset"] == "pilot"
    assert t.metadata["dataset_sha256"] == dataset_sha256(pilot)
    assert t.metadata["split_dataset_sha256"] == dataset_sha256(dev)
    assert t.metadata["dataset_sha256"] != t.metadata["split_dataset_sha256"]
    # the split hash ties the pilot back to the committed dev export
    summary = json.loads(COMPOSITION_FILE.read_text(encoding="utf-8"))
    assert t.metadata["split_dataset_sha256"] == summary["dataset_sha256"]


def test_task_without_subset_keeps_one_hash(dev):
    t = task_module.variance_commentary(split="dev", epochs=1)
    assert t.metadata["subset"] is None
    assert t.metadata["dataset_sha256"] == t.metadata["split_dataset_sha256"] == dataset_sha256(dev)
    assert len(t.dataset) == 30 and t.dataset.name == "variance-commentary-dev"


def test_baseline_metadata_matches_the_main_task_on_the_pilot():
    main = task_module.variance_commentary(split="dev", epochs=1, subset="pilot")
    t = baseline.variance_commentary_template_baseline(split="dev", epochs=1, subset="pilot")
    assert {k: v for k, v in t.metadata.items() if k != "baseline"} == main.metadata
    assert [s.id for s in t.dataset] == [s.id for s in main.dataset]


# --- Real runs ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(pilot, tmp_path_factory):
    """The main task on the pilot at epochs=1, mockllm answering each case with the
    template baseline's output, keyed on the case's first in-scope wallet."""
    answers = {}
    for s in pilot:
        t = truth_of(s.metadata)
        answers[t.scope_wallets[0]] = json.dumps(baseline.template_output(visible_structured(t)))
    # each key must sit in its own case's text and in no other
    for key in answers:
        assert sum(key in str(s.input) for s in pilot) == 1, key

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in answers if w in blob), None)
        out = ModelOutput.from_content(model="mockllm/model", content=answers.get(key, "unmatched"))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    log = inspect_eval(
        task_module.variance_commentary(split="dev", epochs=1, subset="pilot"),
        model=get_model("mockllm/model", custom_outputs=outputs),
        model_roles=ROLES,
        log_dir=_log_dir(tmp_path_factory, "pilot-main"),
        display="none",
    )[0]
    return log


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    return inspect_eval(
        baseline.variance_commentary_template_baseline(split="dev", epochs=1, subset="pilot"),
        log_dir=_log_dir(tmp_path_factory, "pilot-baseline"),
        display="none",
    )[0]


def test_mock_run_records_the_subset_in_the_header(mock_run, pilot):
    log = mock_run
    assert log.status == "success", log.error
    assert log.eval.task_args["subset"] == "pilot"
    assert log.eval.task_args["split"] == "dev"
    assert log.eval.metadata["subset"] == "pilot"
    assert log.eval.metadata["dataset_sha256"] == dataset_sha256(pilot)
    assert log.eval.dataset.name == "variance-commentary-dev-pilot"
    assert log.eval.dataset.samples == 20
    assert {s.id for s in log.samples} == PILOT_IDS
    assert all(set(s.scores) == set(WIRED_SCORERS) for s in log.samples)


def test_mock_run_answers_every_pilot_case(mock_run):
    assert all("unmatched" not in (s.output.completion or "") for s in mock_run.samples)


def test_baseline_scores_the_20_pilot_cases(baseline_run):
    log = baseline_run
    assert log.status == "success", log.error
    assert log.eval.model == "none/none" and not log.stats.model_usage
    assert log.eval.task_args["subset"] == "pilot"
    assert log.eval.metadata["subset"] == "pilot"
    assert len(log.samples) == 20 and {s.id for s in log.samples} == PILOT_IDS
    assert all(set(s.scores) == set(WIRED_SCORERS) for s in log.samples)


def test_mock_template_answers_score_as_the_baseline_does(mock_run, baseline_run):
    """Same outputs, same scorers: the main task's scores equal the baseline row."""

    def scores(log):
        return {s.id: {k: s.scores[k].value for k in WIRED_SCORERS} for s in log.samples}

    assert scores(mock_run) == scores(baseline_run)
