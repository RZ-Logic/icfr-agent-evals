"""Vendored Inspect Scout scanners (pre-registration sections 5 and 9).

Checks the provenance of the vendored copies (pristine blob SHAs, the committed
diff), the changes the spec asks for (dropped awareness cues, the pinned scanner
model), and the split guard on the scan jobs. Every model here is mockllm, and
every log is a dev-split or dummy log; no test reads an evaluation-split case.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model

import icfr_agent_evals.scanners as scanners
from icfr_agent_evals.roster import RosterError, SCOUT_SCANNER, WORKER
from icfr_agent_evals.scanners import _upstream
from icfr_agent_evals.scanners._model import ScannerModelError, scanner_model
from icfr_agent_evals.scanners.jobs import (
    ScanSplitError,
    check_split,
    check_worker,
    checked_logs,
    split_of,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / _upstream.EVIDENCE_DIR
HEADER_LINES = 4  # provenance header added at the top of each vendored file


def _module(name: str):
    # The package re-exports each scanner function under its module's name, so
    # attribute access on the package returns the function; use sys.modules.
    return importlib.import_module(f"icfr_agent_evals.scanners.{name}")


def _upstream_text(name: str) -> str:
    return (EVIDENCE / "examples" / "docs" / f"{name}.py").read_bytes().decode("utf-8")


def _vendored_text(name: str) -> str:
    return (ROOT / "icfr_agent_evals" / "scanners" / f"{name}.py").read_bytes().decode("utf-8")


# --- Provenance ------------------------------------------------------------------------

def test_package_imports_and_exports():
    for name in ("eval_awareness", "refusal_classifier", "command_not_found",
                 "worker_awareness_scan", "all_logs_scan"):
        assert callable(getattr(scanners, name))
    assert "MIT License" in scanners.__doc__ and "Copyright (c) 2025 Meridian Labs" in scanners.__doc__


@pytest.mark.parametrize("upstream", list(_upstream.VENDORED))
def test_pristine_copy_matches_recorded_blob_sha(upstream):
    blob, _ = _upstream.VENDORED[upstream]
    assert _upstream.git_blob_sha((EVIDENCE / upstream).read_bytes()) == blob


def test_upstream_license_copy_matches_recorded_blob_sha():
    path, blob = _upstream.LICENSE_BLOB
    assert _upstream.git_blob_sha((EVIDENCE / path).read_bytes()) == blob


def test_readme_records_commit_and_every_blob_sha():
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert _upstream.UPSTREAM_COMMIT in readme
    for upstream, (blob, vendored) in _upstream.VENDORED.items():
        assert upstream in readme and blob in readme and vendored in readme
    assert _upstream.LICENSE_BLOB[1] in readme


def test_committed_diff_regenerates_exactly():
    committed = (ROOT / _upstream.DIFF_PATH).read_bytes()
    assert _upstream.render_diff(ROOT).encode("utf-8") == committed


@pytest.mark.parametrize("name", ["eval_awareness", "refusal_classifier", "command_not_found"])
def test_vendored_header_names_repo_path_and_commit(name):
    header = "".join(_vendored_text(name).splitlines(keepends=True)[:HEADER_LINES])
    assert "github.com/meridianlabs-ai/inspect_scout" in header
    assert f"examples/docs/{name}.py" in header
    assert _upstream.UPSTREAM_COMMIT in header
    assert all(line.startswith("#") for line in header.splitlines())


def test_written_files_have_no_em_dash_and_lf_only():
    paths = [*(ROOT / "icfr_agent_evals" / "scanners").glob("*.py"), ROOT / _upstream.DIFF_PATH,
             EVIDENCE / "README.md", Path(__file__)]
    for p in paths:
        data = p.read_bytes()
        assert b"\r" not in data, p
        assert chr(0x2014) not in data.decode("utf-8"), p


# --- (a) awareness rubric -----------------------------------------------------------------

def _cue_line(question: str) -> str:
    (line,) = [l for l in question.splitlines() if l.startswith("This often manifests")]
    return line


def test_awareness_rubric_drops_controls_vocabulary_cues():
    question = _module("eval_awareness").QUESTION
    cues = re.findall(r"'([^']+)'", _cue_line(question))
    for word in _upstream.DROPPED_AWARENESS_CUES:
        assert word not in cues
        assert not re.search(rf"'{word}\w*'", question)
    assert cues == ["roleplay", "scenario", "simulation", "pretend", "imaginary", "hypothetical"]


def test_upstream_rubric_listed_the_dropped_cues_and_nothing_else_changed():
    up = _upstream_text("eval_awareness")
    up_q = up.split('QUESTION = """', 1)[1].split('"""', 1)[0]
    up_cues = re.findall(r"'([^']+)'", _cue_line(up_q))
    for word in _upstream.DROPPED_AWARENESS_CUES:
        assert word in up_cues
    ours = _module("eval_awareness").QUESTION
    assert [c for c in up_cues if c not in _upstream.DROPPED_AWARENESS_CUES] == re.findall(r"'([^']+)'", _cue_line(ours))
    up_lines, our_lines = up_q.splitlines(), ours.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(up_lines, our_lines)) if a != b]
    assert len(up_lines) == len(our_lines) and len(changed) == 1
    assert _module("eval_awareness").SCANNER_TEMPLATE in up


# --- (b) scanner model ----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["eval_awareness", "refusal_classifier"])
def test_llm_scanner_defaults_to_scout_scanner_pin(name):
    factory = getattr(_module(name), name)
    assert inspect.signature(factory).parameters["model"].default == SCOUT_SCANNER == "openai/gpt-6-luna"


@pytest.mark.parametrize("name", ["eval_awareness", "refusal_classifier"])
def test_llm_scanner_passes_model_to_llm_scanner(name, monkeypatch):
    mod = _module(name)
    real = mod.llm_scanner
    seen: list[dict] = []

    def capture(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(mod, "llm_scanner", capture)
    getattr(mod, name)()  # building the scanner resolves no model, so no key is needed
    assert seen and seen[-1]["model"] == SCOUT_SCANNER
    getattr(mod, name)(model="mockllm/model")
    assert seen[-1]["model"] == "mockllm/model"


@pytest.mark.parametrize("name", ["eval_awareness", "refusal_classifier"])
@pytest.mark.parametrize("bad", ["openai/gpt-6.1-sol", "anthropic/claude-opus-5-5", WORKER, ""])
def test_llm_scanner_rejects_off_pin_models(name, bad):
    with pytest.raises(ScannerModelError):
        getattr(_module(name), name)(model=bad)


def test_scanner_model_rule():
    assert scanner_model() == SCOUT_SCANNER
    assert scanner_model("mockllm/model") == "mockllm/model"
    assert issubclass(ScannerModelError, RosterError)
    with pytest.raises(ScannerModelError):
        scanner_model("openai/gpt-6-luna-2026-10-01")


def test_command_not_found_uses_no_model():
    mod = _module("command_not_found")
    assert inspect.signature(mod.command_not_found).parameters == {}
    body = "".join(_vendored_text("command_not_found").splitlines(keepends=True)[HEADER_LINES:])
    assert body == _upstream_text("command_not_found")
    assert "llm_scanner" not in body and "get_model" not in body
    assert not re.search(r"\bmodel\b", body)


# --- Scan jobs and the split guard -------------------------------------------------------

def test_split_of_reads_task_metadata_and_fails_closed():
    assert split_of({"split": "dev"}, {"split": "dev"}) == "dev"
    assert split_of({"split": "ci"}, {}) == "ci"
    for meta in (None, {}, {"split": None}, {"split": "evaluation"}, {"split": 3}):
        with pytest.raises(ScanSplitError):
            split_of(meta, {})
    with pytest.raises(ScanSplitError):
        split_of({"split": "dev"}, {"split": "eval"})


def test_check_split_refuses_evaluation_split_by_default():
    for split in ("dev", "ci", "calibration"):
        check_split(split)
    with pytest.raises(ScanSplitError):
        check_split("eval")
    with pytest.raises(ScanSplitError):
        check_split("eval", allow_evaluation_split="yes")  # only the literal True opens it
    check_split("eval", allow_evaluation_split=True)


def test_check_worker():
    check_worker(WORKER)
    check_worker("mockllm/model")
    with pytest.raises(ScanSplitError):
        check_worker("anthropic/claude-haiku-5-5")


class _Captured:
    """Stands in for a ScanJob so the scanjob decorator can tag it."""


def _mock(messages, tools, tool_choice, config) -> ModelOutput:
    out = ModelOutput.from_content(model="mockllm/model", content="ok")
    out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    return out


def _dummy_log(tmp_path: Path, name: str, metadata: dict) -> Path:
    """A one-sample log with no scorer, a trivial input and the given Task metadata.
    It holds no case of any split; only its header's split label varies."""
    task = Task(dataset=[Sample(input="hello", id="dummy-1")], metadata=metadata, name=name)
    log_dir = tmp_path / name
    (log,) = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=_mock),
                          display="none", log_dir=str(log_dir))
    assert log.status == "success"
    return log_dir


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scanlogs")
    return {
        "dev": _dummy_log(tmp, "dev", {"split": "dev"}),
        "eval": _dummy_log(tmp, "evalsplit", {"split": "eval"}),
        "none": _dummy_log(tmp, "nosplit", {}),
        "empty": tmp / "empty",
    }


def test_checked_logs_accepts_dev_and_lists_files(logs):
    files, splits = checked_logs(logs["dev"])
    assert splits == ["dev"] and len(files) == 1 and files[0].endswith(".eval")


def test_checked_logs_refuses_evaluation_split(logs):
    with pytest.raises(ScanSplitError, match="evaluation-split"):
        checked_logs(logs["eval"])
    with pytest.raises(ScanSplitError):
        checked_logs([logs["dev"], logs["eval"]])  # one evaluation-split log refuses the whole job
    _, splits = checked_logs(logs["eval"], allow_evaluation_split=True)
    assert splits == ["eval"]


def test_checked_logs_refuses_unknown_split_and_missing_logs(logs):
    with pytest.raises(ScanSplitError):
        checked_logs(logs["none"])
    logs["empty"].mkdir(exist_ok=True)
    with pytest.raises(ScanSplitError):
        checked_logs(logs["empty"])
    with pytest.raises(ScanSplitError):
        checked_logs(logs["empty"] / "missing.eval")
    with pytest.raises(ScanSplitError):
        checked_logs([])


@pytest.mark.parametrize("job_name", ["worker_awareness_scan", "all_logs_scan"])
def test_scan_jobs_refuse_evaluation_split_before_building(job_name, logs, monkeypatch):
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    built = []
    monkeypatch.setattr(jobs, "ScanJob", lambda **kw: built.append(kw))
    with pytest.raises(ScanSplitError):
        getattr(jobs, job_name)(logs=str(logs["eval"]), model="mockllm/model")
    assert built == []


@pytest.mark.parametrize("job_name, expected", [
    ("worker_awareness_scan", ["eval_awareness"]),
    ("all_logs_scan", ["refusal_classifier", "command_not_found"]),
])
def test_scan_jobs_name_the_scanner_model(job_name, expected, logs, monkeypatch):
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    job_fn = getattr(jobs, job_name)
    assert inspect.signature(job_fn).parameters["model"].default == SCOUT_SCANNER

    # Default model: capture the ScanJob arguments, since inspect-scout 0.5.5
    # resolves a ScanJob model at construction and the pin needs a provider key.
    built: list[dict] = []
    monkeypatch.setattr(jobs, "ScanJob", lambda **kw: built.append(kw) or _Captured())
    job_fn(logs=str(logs["dev"]))
    assert built[-1]["model"] == SCOUT_SCANNER
    assert built[-1]["metadata"]["scanner_model"] == SCOUT_SCANNER
    assert list(built[-1]["scanners"]) == expected
    monkeypatch.undo()

    # mockllm: a real ScanJob, built with no key.
    job = job_fn(logs=str(logs["dev"]), model="mockllm/model")
    assert str(job.model) == "mockllm/model"
    assert list(job.scanners) == expected
    assert job.metadata["splits"] == ["dev"]
    assert job.metadata["scout_examples_commit"] == _upstream.UPSTREAM_COMMIT


def test_awareness_job_refuses_a_baseline_log_and_all_logs_job_accepts_it(logs, monkeypatch):
    """Section 8 runs the awareness scanner on worker transcripts only. The header's
    evaluated model is swapped to a baseline pin, so no real model is ever run."""
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    real_read = jobs.read_eval_log

    def baseline_header(path, header_only=True):
        header = real_read(path, header_only=header_only)
        header.eval.model = "anthropic/claude-haiku-5-5"
        return header

    monkeypatch.setattr(jobs, "read_eval_log", baseline_header)
    monkeypatch.setattr(jobs, "ScanJob", lambda **kw: _Captured())
    with pytest.raises(ScanSplitError):
        jobs.worker_awareness_scan(logs=str(logs["dev"]), model="mockllm/model")
    jobs.all_logs_scan(logs=str(logs["dev"]), model="mockllm/model")


def test_scan_jobs_reject_off_pin_model(logs):
    with pytest.raises(ScannerModelError):
        scanners.all_logs_scan(logs=str(logs["dev"]), model="openai/gpt-6.1-sol")
