"""Command lines of the owner-facing scripts (F2-10, F5-02).

F2-10: rehearse_regression.py and consistency_mock.py took sys.argv[1] as their log
dir, so -h or --help became a folder. They now parse with argparse; -h prints help,
exits 0 and creates nothing, run from an empty temporary cwd.

F5-02 (D15): smoke_live.py is owner-run and keyed. These tests stay keyless: they
cover its parser, its default model list, the dev pilot sample check, and one
offline mockllm run of the fixed-prompt path. No real model is ever built here.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

from icfr_agent_evals import roster
from icfr_agent_evals.dataset import PILOT_CASES

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MOCK_SCRIPTS = ("rehearse_regression.py", "consistency_mock.py")


def _load(name: str):
    """Import a script by path; registered first because its dataclasses look it up."""
    key = f"w7_cli_{name.removesuffix('.py')}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
        "INSPECT_EVAL_MODEL_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS",
        "INSPECT_EVAL_EXTRA_HEADERS")}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), *filter(None, [env.get("PYTHONPATH")])])
    return env


def _run_help(script: str, flag: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / script), flag], cwd=cwd, env=_env(),
                          capture_output=True, text=True, check=False, timeout=600)


# --- F2-10: -h creates no folder ---------------------------------------------------------

@pytest.mark.parametrize("flag", ["-h", "--help"])
@pytest.mark.parametrize("script", MOCK_SCRIPTS)
def test_mock_script_help_creates_no_folder(script, flag, tmp_path):
    default_dir = _load(script).DEFAULT_LOG_DIR
    default_existed = os.path.exists(default_dir)
    done = _run_help(script, flag, tmp_path)
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout and "log_dir" in done.stdout
    assert list(tmp_path.iterdir()) == []
    assert os.path.exists(default_dir) == default_existed


@pytest.mark.parametrize("script", MOCK_SCRIPTS)
def test_mock_script_log_dir_is_optional_positional(script):
    mod = _load(script)
    assert mod.parse_args([]).log_dir == mod.DEFAULT_LOG_DIR
    assert mod.parse_args(["some/dir"]).log_dir == "some/dir"
    with pytest.raises(SystemExit) as exc:
        mod.parse_args(["a", "b"])
    assert exc.value.code == 2


@pytest.mark.parametrize("script", MOCK_SCRIPTS)
def test_mock_script_no_longer_reads_argv_directly(script):
    assert "sys.argv[1]" not in (SCRIPTS / script).read_text(encoding="utf-8")


# --- F5-02: smoke_live.py, keyless parts only -------------------------------------------------

smoke_live = _load("smoke_live.py")

EXPECTED_DEFAULT = [
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5-5",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-terra",
    "openai/gpt-6.1-sol",
]


def test_default_models_are_every_distinct_roster_model_in_stable_order():
    models = smoke_live.default_models()
    assert models == EXPECTED_DEFAULT
    assert len(models) == len(set(models))
    roles = {roster.WORKER, *roster.BASELINES, *roster.GRADER_BY_STEP.values(), roster.CALIBRATOR}
    assert set(models) == roles
    assert smoke_live.default_models() == models  # stable across calls


def test_parse_args_defaults_to_the_roster_and_requires_log_dir(tmp_path):
    args = smoke_live.parse_args(["--log-dir", str(tmp_path)])
    assert args.models == EXPECTED_DEFAULT
    assert args.sample_id is None
    assert args.log_dir == str(tmp_path)
    with pytest.raises(SystemExit) as exc:
        smoke_live.parse_args([])
    assert exc.value.code == 2


def test_models_list_is_stripped_and_deduplicated_in_order(tmp_path):
    args = smoke_live.parse_args(["--log-dir", str(tmp_path), "--models",
                                  " openai/gpt-6-luna,anthropic/claude-haiku-5-5,openai/gpt-6-luna, "])
    assert args.models == ["openai/gpt-6-luna", "anthropic/claude-haiku-5-5"]


@pytest.mark.parametrize("value", ["", " , ", "claude-haiku-5-5"])
def test_models_list_rejects_empty_or_unqualified(value, tmp_path):
    with pytest.raises(SystemExit) as exc:
        smoke_live.parse_args(["--log-dir", str(tmp_path), "--models", value])
    assert exc.value.code == 2


def test_sample_id_must_be_a_dev_pilot_case():
    ids = smoke_live.pilot_ids()
    assert len(ids) == PILOT_CASES and ids == sorted(ids)
    assert smoke_live.resolve_sample_id(None, ids) == ids[0]
    assert smoke_live.resolve_sample_id(ids[-1], ids) == ids[-1]
    with pytest.raises(ValueError, match="dev pilot subset"):
        smoke_live.resolve_sample_id("eval-0001", ids)


def test_non_pilot_sample_id_refuses_before_any_model_or_folder(tmp_path, capsys):
    log_dir = tmp_path / "logs"
    with pytest.raises(SystemExit) as exc:
        smoke_live.main(["--log-dir", str(log_dir), "--models", "mockllm/model", "--sample-id", "not-a-case"])
    assert exc.value.code == 2
    assert not log_dir.exists()


def test_smoke_live_help_creates_no_folder(tmp_path):
    done = _run_help("smoke_live.py", "-h", tmp_path)
    assert done.returncode == 0, done.stderr
    assert "--log-dir" in done.stdout and "--models" in done.stdout and "--sample-id" in done.stdout
    assert list(tmp_path.iterdir()) == []


def test_docstring_names_rows_212_and_220_and_the_dev_split_only():
    doc = smoke_live.__doc__
    assert "row 212" in doc and "row 220" in doc
    assert "never runs on the evaluation split" in doc


def test_offline_fixed_prompt_path_answers_and_prints_the_resolved_model(tmp_path, capsys):
    """mockllm only: the fixed-prompt path builds no case content and reads the
    resolved model string back from the log."""
    code = smoke_live.main(["--log-dir", str(tmp_path), "--models", "mockllm/model"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "PASS mockllm/model (fixed prompt): resolved mockllm;" in out  # mockllm reports its provider name
    assert "endpoint not recorded (offline)" in out
    assert "LIVE SMOKE RESULT: PASS" in out
    assert any((tmp_path / "mockllm_model").glob("*.eval"))


def test_extra_headers_line_names_sources_never_values(monkeypatch):
    monkeypatch.delenv(roster.EXTRA_HEADERS_ENV, raising=False)
    assert smoke_live.extra_headers_line() == "extra headers in force: none"
    monkeypatch.setenv(roster.EXTRA_HEADERS_ENV, "x-marker=secret-value")
    line = smoke_live.extra_headers_line()
    assert roster.EXTRA_HEADERS_ENV in line
    assert "secret-value" not in line and "x-marker" not in line


def test_main_prints_the_header_sources_before_each_model(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(roster.EXTRA_HEADERS_ENV, raising=False)
    code = smoke_live.main(["--log-dir", str(tmp_path), "--models", "mockllm/model"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "mockllm/model: extra headers in force: none" in out
    assert out.index("extra headers in force") < out.index("PASS mockllm/model")


def test_docstring_says_smoke_logs_never_reach_the_readers():
    doc = smoke_live.__doc__
    assert "never pass a smoke log" in doc
    assert "pilot_report.py" in doc and "compare_runs.py" in doc


def test_evaluated_model_path_passes_split_dev_explicitly(tmp_path, monkeypatch):
    """The C1 lock refuses a no-split build, so the pilot-task path must name split dev.
    Both the task build and eval() are stand-ins: nothing generates."""
    import inspect_ai

    from icfr_agent_evals import task as task_module

    seen: dict = {}

    def fake_task(**kwargs):
        seen.update(kwargs)
        return object()

    class _Log:
        status, samples, location = "error", [], "stand-in"

    monkeypatch.setattr(task_module, "variance_commentary", fake_task)
    monkeypatch.setattr(inspect_ai, "eval", lambda *a, **k: [_Log()])
    answered, line = smoke_live.smoke_one(roster.WORKER, "stand-in-id", str(tmp_path))
    assert seen == {"split": "dev", "subset": "pilot", "epochs": 1}
    assert answered is False and line.startswith("FAIL")


# --- compare_runs.py: the review-volume ladder rung (section 10(c), D-E5) ----------------

def test_compare_runs_ladder_offers_r1_r2_and_r3(capsys):
    compare_runs = _load("compare_runs.py")
    with pytest.raises(SystemExit) as stop:
        compare_runs.main(["--help"])
    assert stop.value.code == 3
    out = capsys.readouterr().out
    assert "R1" in out and "R2" in out and "R3" in out


def test_compare_runs_accepts_ladder_r3_and_reaches_the_inputs(tmp_path, capsys):
    """R3 passes the parser: the refusal is about the missing log, not the rung."""
    compare_runs = _load("compare_runs.py")
    missing = str(tmp_path / "missing.eval")
    no_record = str(tmp_path / "ladder.json")
    assert compare_runs.main(["--baseline", missing, "--candidate", missing, "--ladder", "R3",
                              "--ladder-record", no_record]) == 3
    err = capsys.readouterr().err
    assert "REFUSED" in err and "--ladder" not in err


@pytest.mark.parametrize("rung", ["R4", "r3", "R0"])
def test_compare_runs_refuses_an_unknown_ladder_rung_with_exit_3(rung, tmp_path, capsys):
    compare_runs = _load("compare_runs.py")
    missing = str(tmp_path / "missing.eval")
    with pytest.raises(SystemExit) as stop:
        compare_runs.main(["--baseline", missing, "--candidate", missing, "--ladder", rung])
    assert stop.value.code == 3
    assert "--ladder" in capsys.readouterr().err


def test_compare_runs_docstring_names_r3_and_the_route_blind_trial():
    doc = _load("compare_runs.py").__doc__
    assert "R1|R2|R3" in doc and "route-blind trial" in doc and "D-E5" in doc


def test_compare_runs_help_names_the_ladder_record(capsys):
    compare_runs = _load("compare_runs.py")
    with pytest.raises(SystemExit):
        compare_runs.main(["--help"])
    assert "--ladder-record" in capsys.readouterr().out
    assert compare_runs.DEFAULT_LADDER_RECORD == REPO / "docs" / "evidence" / "regression" / "ladder.json"


def test_compare_runs_docstring_says_the_recorded_rung_wins():
    doc = _load("compare_runs.py").__doc__
    assert "--ladder-record" in doc and "disagrees with a recorded" in doc and "no rung" in doc
