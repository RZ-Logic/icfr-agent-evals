"""Task wiring tests (pre-registration section 5 run configuration).

The dataset generator is rebuilt in parallel, so these tests swap build_dataset for
a casefactory dataset and run the real task with Inspect's mockllm provider only.
"""

import json
import re
from pathlib import Path

import pytest
from casefactory import good_output, make_truth
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, ModelUsage, get_model

import icfr_agent_evals
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import DATASET_VERSION, SCHEMA_VERSION, CaseFile, split_seed, to_sample
from icfr_agent_evals.roster import (
    NOTICE_POLICIES,
    RETIREMENT_FLOORS,
    RETIREMENT_FLOORS_RECORDED,
    SERVING_PLATFORMS,
    WORKER,
)
from icfr_agent_evals.rubric import WIRED_SCORERS
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256, WORKER_PROMPT_VERSION

ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}


def _dataset(split="eval", seed=20260607, n=2):
    truths = [make_truth(a, case_id=f"{split}-unit-{i:03d}-{a}", split=split) for i, a in enumerate(["BTC", "SOL"][:n])]
    samples = [to_sample(CaseFile(t, {}, f"case {t.case_id}\nreconcile {t.asset} {t.period}")) for t in truths]
    return MemoryDataset(samples=samples, name=f"unit-{split}")


@pytest.fixture
def unit_dataset(monkeypatch):
    monkeypatch.setattr(task_module, "build_dataset", lambda split, seed: _dataset(split, seed))


def mock_worker(answers: dict[str, str]):
    """mockllm keyed on the case id in the prompt; usage is always set so Inspect
    never reaches for a tokenizer download."""

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        case = next((cid for cid in answers if cid in blob), None)
        out = ModelOutput.from_content(model="mockllm/model", content=answers.get(case, "unmatched"))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return get_model("mockllm/model", custom_outputs=outputs)


def _good_answers(ds):
    return {s.id: json.dumps(good_output(make_truth(s.metadata["truth"]["asset"]))) for s in ds}


def test_metadata_keys_and_values(unit_dataset):
    t = task_module.variance_commentary(split="dev", epochs=4)
    meta = t.metadata
    for key in (
        "worker_prompt_version", "worker_prompt_sha256", "dataset_version", "dataset_sha256",
        "roster", "retirement_floors", "split", "split_seed", "schema_version",
    ):
        assert key in meta, key
    assert meta["worker_prompt_version"] == WORKER_PROMPT_VERSION
    assert meta["worker_prompt_sha256"] == WORKER_PROMPT_SHA256
    assert meta["dataset_version"] == DATASET_VERSION == t.version
    assert re.fullmatch(r"[0-9a-f]{64}", meta["dataset_sha256"])
    assert meta["split"] == "dev" and meta["split_seed"] == split_seed("dev", 20260607)
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["roster"]["worker"] == WORKER
    assert meta["retirement_floors"]["floors"] == RETIREMENT_FLOORS
    # Section 13: retirement floors, notice policies and serving platform in the run config.
    assert meta["retirement_floors"]["recorded"] == RETIREMENT_FLOORS_RECORDED
    assert meta["retirement_floors"]["notice_policies"] == NOTICE_POLICIES
    assert meta["retirement_floors"]["serving_platforms"] == SERVING_PLATFORMS
    assert meta["roster"]["serving_platforms"] == SERVING_PLATFORMS


def test_task_binds_no_model_and_sets_no_sampling_parameter(unit_dataset):
    t = task_module.variance_commentary()
    assert t.model is None
    assert t.config.temperature is None and t.config.top_p is None and t.config.top_k is None
    assert t.config.reasoning_effort == "high"
    assert t.epochs == 4
    assert t.setup is not None


def test_grader_step_two_records_the_escalation_pin(unit_dataset):
    t = task_module.variance_commentary(grader_step=2)
    assert t.metadata["roster"]["roles"]["grader"] == "openai/gpt-6.1-sol"


def test_package_source_never_names_a_sampling_parameter():
    root = Path(icfr_agent_evals.__file__).parent
    pattern = re.compile(r"temperature|top_p|top_k")
    hits = [f"{p.name}:{n}" for p in sorted(root.rglob("*.py")) for n, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert hits == []


def test_superseded_roster_constants_are_gone():
    assert not hasattr(task_module, "DEFAULT_WORKER")
    assert not hasattr(task_module, "BASELINE_MODELS")


def test_mock_run_records_reducer_metadata_and_scores(unit_dataset, tmp_path):
    ds = _dataset()
    logs = inspect_eval(
        task_module.variance_commentary(epochs=2),
        model=mock_worker(_good_answers(ds)),
        model_roles=ROLES,
        log_dir=str(tmp_path),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert log.eval.config.epochs == 2
    assert log.eval.config.epochs_reducer is not None and log.eval.config.epochs_reducer == ["mean"]
    assert log.eval.task_args.get("seed") == 20260607
    assert log.eval.metadata["dataset_version"] == DATASET_VERSION
    assert log.eval.task_version == DATASET_VERSION
    assert len(log.samples) == 4
    for s in log.samples:
        assert {k: v.value for k, v in s.scores.items()} == {
            "correctness_anchor": 3, "fabrication_anchor": 3, "reconciling_items_anchor": 3,
            "faithfulness_anchor": 3, "completeness_anchor": 3, "injection_compliance": 3,
        }
        assert set(s.scores) == set(WIRED_SCORERS)
        for name, v in s.scores.items():
            assert {"failure_modes", "requires_human_review", "review_reasons", "dimension"} <= set(v.metadata)
            assert v.metadata["dimension"] == WIRED_SCORERS[name]


def test_format_break_scores_zero_everywhere_without_crashing(unit_dataset, tmp_path):
    ds = _dataset()
    answers = {sid: 'Sure! {"asset": "BTC"}' for sid in (s.id for s in ds)}
    log = inspect_eval(
        task_module.variance_commentary(epochs=1), model=mock_worker(answers), model_roles=ROLES,
        log_dir=str(tmp_path), display="none",
    )[0]
    assert log.status == "success", log.error
    for s in log.samples:
        for v in s.scores.values():
            assert v.value == 0 and "FM-000" in v.metadata["failure_modes"] and v.metadata["requires_human_review"]
            assert "FM-005" not in v.metadata["failure_modes"]


# The tests run with no API key, so a role bound to a real provider cannot be
# instantiated here; the "none" provider stands in for a real
# model (it is not offline mockllm), and an off-roster role name is the other case.
@pytest.mark.parametrize(
    "roles,needle",
    [
        ({"grader": "none/none", "calibrator": "mockllm/model"}, "mixing real and mockllm"),
        ({**ROLES, "scanner": "mockllm/model"}, "not on the frozen roster"),
    ],
)
def test_bad_role_fails_the_run(unit_dataset, tmp_path, roles, needle):
    ds = _dataset()
    log = inspect_eval(
        task_module.variance_commentary(epochs=1),
        model=mock_worker(_good_answers(ds)),
        model_roles=roles,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "error"
    assert "RosterError" in log.error.traceback and needle in log.error.message
    assert not log.samples or all(not s.scores for s in log.samples)


def test_unbound_role_fails_the_run(unit_dataset, tmp_path):
    ds = _dataset()
    log = inspect_eval(
        task_module.variance_commentary(epochs=1), model=mock_worker(_good_answers(ds)),
        model_roles={"grader": "mockllm/model"}, log_dir=str(tmp_path), display="none",
    )[0]
    assert log.status == "error"
    assert "calibrator" in log.error.message
    assert not log.samples or all(not s.scores for s in log.samples)
