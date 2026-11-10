"""Task wiring tests (pre-registration section 5 run configuration).

The dataset generator is rebuilt in parallel, so these tests swap build_dataset for
a casefactory dataset and run the real task with Inspect's mockllm provider only.
"""

import inspect as pyinspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from casefactory import good_output, make_truth
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ModelOutput,
    ModelUsage,
    get_model,
)

import icfr_agent_evals
from icfr_agent_evals import baseline
from icfr_agent_evals import dataset as dataset_module
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import ASSETS, DATASET_VERSION, SCHEMA_VERSION, CaseFile, split_seed, to_sample
from icfr_agent_evals.roster import (
    NOTICE_POLICIES,
    RETIREMENT_FLOORS,
    RETIREMENT_FLOORS_RECORDED,
    SERVING_PLATFORMS,
    WORKER,
    RosterError,
)
from icfr_agent_evals.rubric import WIRED_SCORERS
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256, WORKER_PROMPT_VERSION

ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}


def _dataset(split="dev", seed=20260607, n=2):
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
    t = task_module.variance_commentary(split="dev")
    assert t.model is None
    assert t.config.temperature is None and t.config.top_p is None and t.config.top_k is None
    assert t.config.reasoning_effort == "high"
    assert t.epochs == 4
    assert t.setup is not None


def test_grader_step_two_records_the_terra_pin(unit_dataset):
    t = task_module.variance_commentary(split="dev", grader_step=2)
    assert t.metadata["roster"]["roles"]["grader"] == "openai/gpt-6.1-terra"
    assert t.metadata["roster"]["grader_generate_config"] == {"batch": True}


def test_grader_step_three_records_the_sol_pin(unit_dataset):
    t = task_module.variance_commentary(split="dev", grader_step=3)
    assert t.metadata["roster"]["roles"]["grader"] == "openai/gpt-6.1-sol"
    assert t.metadata["roster"]["grader_generate_config"] == {"batch": True}
    assert t.config.batch is None  # the batch pin is the grader's, never the eval-wide config


def test_grader_step_one_records_luna_without_batch(unit_dataset):
    t = task_module.variance_commentary(split="dev")
    assert t.metadata["roster"]["roles"]["grader"] == "openai/gpt-6-luna"
    assert t.metadata["roster"]["grader_generate_config"] == {}


def test_grader_step_four_is_refused(unit_dataset):
    with pytest.raises(RosterError, match="grader_step"):
        task_module.variance_commentary(split="dev", grader_step=4)


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
        task_module.variance_commentary(split="dev", epochs=2),
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
    assert log.eval.task_args.get("allow_evaluation_split") is False
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
        task_module.variance_commentary(split="dev", epochs=1), model=mock_worker(answers), model_roles=ROLES,
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
        task_module.variance_commentary(split="dev", epochs=1),
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
        task_module.variance_commentary(split="dev", epochs=1), model=mock_worker(_good_answers(ds)),
        model_roles={"grader": "mockllm/model"}, log_dir=str(tmp_path), display="none",
    )[0]
    assert log.status == "error"
    assert "calibrator" in log.error.message
    assert not log.samples or all(not s.scores for s in log.samples)


# --- Evaluation-split lock (rule C3; findings F1-01, F5-01) ------------------------------

LOCK_TEXT = "refusing split 'eval' before the pre-registration tag"
REPO = Path(__file__).resolve().parents[1]

# The inspect CLI runs in a child process with every split build replaced by a
# sentinel, so even a broken lock can never build, generate on or score an
# evaluation case: the run stops at the sentinel instead.
_CLI_RUNNER = """
import sys
import icfr_agent_evals.dataset as d

def _sentinel(*args, **kwargs):
    raise AssertionError("SENTINEL: a split build was reached")

d.build_dataset = _sentinel
d.build_cases = _sentinel
from inspect_ai._cli.main import main
sys.argv = ["inspect"] + sys.argv[1:]
main()
"""


@pytest.fixture
def no_split_build(monkeypatch):
    """Any split build fails the test: the lock must fire before it."""

    def refuse(*args, **kwargs):
        raise AssertionError("a split was built before the evaluation-split lock")

    monkeypatch.setattr(dataset_module, "build_dataset", refuse)
    monkeypatch.setattr(dataset_module, "build_cases", refuse)
    monkeypatch.setattr(task_module, "build_dataset", refuse)


def _cli(tmp_path, *task_flags: str) -> subprocess.CompletedProcess:
    runner = tmp_path / "cli_runner.py"
    runner.write_text(_CLI_RUNNER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "INSPECT_EVAL_EXTRA_HEADERS"}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    cmd = [
        sys.executable, str(runner), "eval", "icfr_agent_evals/task.py@variance_commentary", *task_flags,
        "--model", "mockllm/model", "--model-role", "grader=mockllm/model",
        "--model-role", "calibrator=mockllm/model", "--log-dir", str(tmp_path / "logs"), "--display", "none",
    ]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=300, check=False)


def _no_logs(tmp_path) -> bool:
    return not (tmp_path / "logs").exists() or not list((tmp_path / "logs").rglob("*.eval"))


@pytest.mark.parametrize("fn", [task_module.variance_commentary, baseline.variance_commentary_template_baseline])
def test_split_defaults_to_eval_and_the_lock_defaults_to_closed(fn):
    sig = pyinspect.signature(fn)
    assert sig.parameters["split"].default == "eval"
    assert sig.parameters["allow_evaluation_split"].default is False


@pytest.mark.parametrize("fn", [task_module.variance_commentary, baseline.variance_commentary_template_baseline])
def test_no_argument_call_refuses_before_any_build(no_split_build, fn):
    with pytest.raises(ValueError, match=re.escape(LOCK_TEXT)) as err:
        fn()
    assert "allow_evaluation_split" in str(err.value)


@pytest.mark.parametrize("unlock", [False, None, 0, 1, "true", "True", "yes"])
def test_eval_split_refused_unless_unlock_is_exactly_true(no_split_build, unlock):
    with pytest.raises(ValueError, match=re.escape(LOCK_TEXT)):
        task_module.variance_commentary(split="eval", allow_evaluation_split=unlock)
    with pytest.raises(ValueError, match=re.escape(LOCK_TEXT)):
        baseline.variance_commentary_template_baseline(split="eval", allow_evaluation_split=unlock)


@pytest.mark.parametrize("split", ["dev", "ci", "calibration"])
def test_lock_leaves_every_other_split_alone(unit_dataset, split):
    assert task_module.variance_commentary(split=split, epochs=1).metadata["split"] == split
    assert baseline.variance_commentary_template_baseline(split=split, epochs=1).metadata["split"] == split


def test_lock_leaves_the_pilot_subset_alone(monkeypatch):
    calls = []

    def fake_build(split, seed):
        calls.append(split)
        return _dataset("dev")

    monkeypatch.setattr(task_module, "build_dataset", fake_build)
    monkeypatch.setattr(task_module, "select_subset", lambda full, split, subset: list(full))
    t = task_module.variance_commentary(split="dev", subset="pilot", epochs=1)
    b = baseline.variance_commentary_template_baseline(split="dev", subset="pilot", epochs=1)
    assert t.metadata["subset"] == b.metadata["subset"] == "pilot"
    assert calls == ["dev", "dev"]


def test_unlock_builds_the_eval_split(monkeypatch):
    """allow_evaluation_split=True reaches build_dataset with split 'eval'; the build
    is a casefactory stand-in, so no evaluation case is built or scored."""
    calls = []

    def fake_build(split, seed):
        calls.append(split)
        return _dataset("dev")

    monkeypatch.setattr(task_module, "build_dataset", fake_build)
    t = task_module.variance_commentary(split="eval", allow_evaluation_split=True)
    b = baseline.variance_commentary_template_baseline(split="eval", allow_evaluation_split=True)
    assert calls == ["eval", "eval"]
    assert t.metadata["split"] == b.metadata["split"] == "eval"


def test_inspect_eval_file_spec_with_split_eval_refuses_at_task_load(no_split_build, tmp_path):
    """The path the CLI takes (a file task spec plus task_args) raises while the task
    loads, before any sample exists; no log is written."""
    with pytest.raises(Exception) as err:
        inspect_eval(
            "icfr_agent_evals/task.py@variance_commentary", task_args={"split": "eval"},
            model="mockllm/model", model_roles=ROLES, log_dir=str(tmp_path / "logs"), display="none",
        )
    assert LOCK_TEXT in str(err.value) and "allow_evaluation_split" in str(err.value)
    assert _no_logs(tmp_path)


@pytest.mark.parametrize(
    "flags,needle",
    [
        (("-T", "split=eval"), LOCK_TEXT),
        (("-T", "splt=dev"), LOCK_TEXT),  # Inspect drops the mistyped key: split stays eval
        (("-T", "split=eval", "-T", "allow_evaluation_split=1"), LOCK_TEXT),  # YAML int, not True
        (("-T", "splt=dev", "-T", "subset=pilot"), "only on split 'dev'"),  # the subset check fires first
    ],
    ids=["split-eval", "mistyped-split-key", "unlock-1", "mistyped-split-with-pilot"],
)
def test_cli_eval_split_refuses_with_no_log(tmp_path, flags, needle):
    run = _cli(tmp_path, *flags)
    assert run.returncode != 0
    assert needle in run.stderr
    assert "SENTINEL" not in run.stderr
    assert _no_logs(tmp_path)


def test_cli_unlock_parses_to_true_and_reaches_the_build(tmp_path):
    """-T allow_evaluation_split=true passes the lock; the child stops at the build
    sentinel, so nothing is built."""
    run = _cli(tmp_path, "-T", "split=eval", "-T", "allow_evaluation_split=true")
    assert run.returncode != 0
    assert "SENTINEL" in run.stderr and LOCK_TEXT not in run.stderr


# --- Reasoning blocks never reach the parser (finding F1-06) -----------------------------

WRONG_DRAFT = json.dumps({"variance_usd": -1, "entities_cited": ["DP-0000-000000"], "narrative": "decoy"})


def _asset_dataset():
    truths = [make_truth(a, case_id=f"dev-unit-{i:03d}-{a}", split="dev") for i, a in enumerate(ASSETS)]
    samples = [to_sample(CaseFile(t, {}, f"case {t.case_id}\nreconcile {t.asset} {t.period}")) for t in truths]
    return MemoryDataset(samples=samples, name="unit-dev-assets")


def _content_worker(contents: dict[str, list]):
    """mockllm returning a content list (reasoning and text parts) keyed on the case
    id in the prompt; usage is set so no tokenizer is fetched."""

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        case = next(cid for cid in contents if cid in blob)
        message = ChatMessageAssistant(content=contents[case], model="mockllm/model")
        out = ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=message, stop_reason="stop")])
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return get_model("mockllm/model", custom_outputs=outputs)


def _asset_run(monkeypatch, log_dir, build_content):
    ds = _asset_dataset()
    monkeypatch.setattr(task_module, "build_dataset", lambda split, seed: ds)
    contents = {s.id: build_content(json.dumps(good_output(make_truth(s.metadata["truth"]["asset"])))) for s in ds}
    log = inspect_eval(
        task_module.variance_commentary(split="dev", epochs=1), model=_content_worker(contents),
        model_roles=ROLES, log_dir=str(log_dir), display="none",
    )[0]
    assert log.status == "success", log.error
    return {
        s.metadata["truth"]["asset"]: ({k: v.value for k, v in s.scores.items()}, s.output.completion)
        for s in log.samples
    }


@pytest.mark.parametrize(
    "reasoning",
    [
        ContentReasoning(reasoning="Draft answer: " + WRONG_DRAFT, summary="I first drafted " + WRONG_DRAFT),
        ContentReasoning(reasoning="opaque", redacted=True),
    ],
    ids=["summarized-with-wrong-draft", "redacted"],
)
def test_reasoning_block_ahead_of_the_json_scores_like_the_plain_completion(monkeypatch, tmp_path, reasoning):
    plain = _asset_run(monkeypatch, tmp_path / "plain", lambda good: [ContentText(text=good)])
    thought = _asset_run(monkeypatch, tmp_path / "thought", lambda good: [reasoning, ContentText(text=good)])
    assert set(plain) == set(thought) == set(ASSETS)
    for asset in ASSETS:
        plain_scores, plain_completion = plain[asset]
        thought_scores, thought_completion = thought[asset]
        assert thought_completion == plain_completion
        assert "DP-0000-000000" not in thought_completion
        assert thought_scores == plain_scores, asset
        assert set(plain_scores.values()) == {3}, asset


def test_reasoning_only_completion_is_a_format_break_not_a_crash(monkeypatch, tmp_path):
    runs = _asset_run(monkeypatch, tmp_path, lambda good: [ContentReasoning(reasoning=good)])
    assert set(runs) == set(ASSETS)
    for asset, (scores, completion) in runs.items():
        assert completion == "", asset
        assert set(scores.values()) == {0}, asset
