"""scripts/preflight.py tests (review finding F1-07, external review R4 pilot mitigation).

Keyless: the environment is a stub mapping, git runs against a temporary repository,
the .env lookup is stubbed or confined to tmp_path, and the models are built with a
dummy key and never asked to generate. HOME, USERPROFILE, APPDATA and
ANTHROPIC_CONFIG_DIR point at tmp_path, so the anthropic SDK never reads the real
home folder.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai.model import get_model

from icfr_agent_evals import roster
from icfr_agent_evals.roster import BASELINES, CALIBRATOR, GRADER, WORKER

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
DUMMY = "dummy-value-w2-never-printed"
CLEAN_ENV = {"ANTHROPIC_API_KEY": DUMMY, "OPENAI_API_KEY": DUMMY}


def _load():
    spec = importlib.util.spec_from_file_location("preflight_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


pf = _load()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "APPDATA", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    for name in (*pf.MUST_BE_UNSET, "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def _build(name):
    return get_model(name, api_key="test", memoize=False)


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=repo,
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    (path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "one")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    return path, head


def _by_name(checks):
    return {c.name: c for c in checks}


# --- Arguments ----------------------------------------------------------------------------

@pytest.mark.parametrize("model", [WORKER, *BASELINES])
def test_parser_accepts_each_evaluated_model(model):
    args = pf.build_parser().parse_args(["--model", model, "--pilot-commit", "d4e86ec"])
    assert (args.model, args.pilot_commit, args.log_dir) == (model, "d4e86ec", None)


@pytest.mark.parametrize("argv", [
    ["--model", GRADER, "--pilot-commit", "d4e86ec"],          # a judge, not an evaluated model
    ["--model", "mockllm/model", "--pilot-commit", "d4e86ec"],
    ["--model", WORKER],                                        # no commit
    ["--pilot-commit", "d4e86ec"],                              # no model
    ["--model", WORKER, "--pilot-commit", "main"],              # a branch name is not a pinned sha
    ["--model", WORKER, "--pilot-commit", "d4e8"],              # too short
])
def test_parser_refuses_bad_arguments(argv):
    with pytest.raises(SystemExit) as info:
        pf.build_parser().parse_args(argv)
    assert info.value.code == 2


def test_parser_takes_a_log_dir():
    args = pf.build_parser().parse_args(["--model", WORKER, "--pilot-commit", "a" * 40, "--log-dir", "x"])
    assert args.log_dir == "x"


# --- Git ----------------------------------------------------------------------------------

def test_git_checks_pass_on_a_clean_repo_at_the_pilot_commit(repo):
    path, head = repo
    checks = _by_name(pf.check_git(head, path))
    assert all(c.ok for c in checks.values()), checks
    assert checks["head"].detail == head


def test_a_short_pilot_sha_resolves(repo):
    path, head = repo
    assert _by_name(pf.check_git(head[:7], path))["head"].ok


def test_head_off_the_pilot_commit_fails(repo):
    path, head = repo
    (path / "b.txt").write_text("b\n", encoding="utf-8")
    _git(path, "add", "b.txt")
    _git(path, "commit", "-q", "-m", "two")
    check = _by_name(pf.check_git(head, path))["head"]
    assert not check.ok and "not " + head in check.detail


def test_unknown_pilot_commit_fails(repo):
    path, _ = repo
    assert not _by_name(pf.check_git("0" * 40, path))["head"].ok


@pytest.mark.parametrize("dirty", ["untracked", "modified"])
def test_a_dirty_tree_fails(repo, dirty):
    path, head = repo
    if dirty == "untracked":
        (path / "new.txt").write_text("n\n", encoding="utf-8")
    else:
        (path / "a.txt").write_text("changed\n", encoding="utf-8")
    check = _by_name(pf.check_git(head, path))["clean"]
    assert not check.ok and "1 uncommitted path(s)" in check.detail


def test_running_from_a_subfolder_fails_the_repo_check(repo):
    path, head = repo
    sub = path / "sub"
    sub.mkdir()
    assert not _by_name(pf.check_git(head, sub))["repo"].ok


def test_outside_any_repository_every_git_check_fails(tmp_path):
    checks = pf.check_git("a" * 40, tmp_path, git=lambda args, cwd: (128, ""))
    assert [c.ok for c in checks] == [False, False, False]


# --- Python and environment ---------------------------------------------------------------

def test_python_version_must_be_the_pin():
    assert pf.REQUIRED_PYTHON == "3.14.10"
    assert pf.check_python("3.14.10").ok
    assert not pf.check_python("3.12.16").ok


def test_the_unset_list_covers_the_serving_guard_variables():
    assert set(pf.MUST_BE_UNSET) == {
        "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS", "INSPECT_EVAL_EXTRA_HEADERS",
    }


def test_env_passes_with_both_keys_and_nothing_else():
    checks = _by_name(pf.check_env(CLEAN_ENV))
    assert checks["env"].ok and checks["keys"].ok
    assert checks["keys"].detail == "ANTHROPIC_API_KEY set: yes, OPENAI_API_KEY set: yes"


@pytest.mark.parametrize("name", pf.MUST_BE_UNSET)
@pytest.mark.parametrize("value", [DUMMY, ""])
def test_each_override_variable_fails_even_when_blank(name, value):
    check = _by_name(pf.check_env({**CLEAN_ENV, name: value}))["env"]
    assert not check.ok and name in check.detail and DUMMY not in check.detail


@pytest.mark.parametrize("missing", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
@pytest.mark.parametrize("value", [None, "  "])
def test_a_missing_or_blank_key_fails_and_prints_no(missing, value):
    env = dict(CLEAN_ENV)
    if value is None:
        del env[missing]
    else:
        env[missing] = value
    check = _by_name(pf.check_env(env))["keys"]
    assert not check.ok and f"{missing} set: no" in check.detail and DUMMY not in check.detail


def test_dotenv_setting_an_override_fails(tmp_path):
    (tmp_path / ".env").write_text(f"INSPECT_EVAL_EXTRA_HEADERS={DUMMY}\n", encoding="utf-8")
    assert pf.find_dotenv_upward(tmp_path) == str(tmp_path / ".env")
    check = pf.check_dotenv(tmp_path)
    assert not check.ok and "INSPECT_EVAL_EXTRA_HEADERS" in check.detail and DUMMY not in check.detail


def test_dotenv_without_an_override_passes(tmp_path):
    (tmp_path / ".env").write_text("SOME_OTHER_SETTING=1\n", encoding="utf-8")
    assert pf.check_dotenv(tmp_path).ok


def test_no_dotenv_passes(tmp_path):
    assert pf.check_dotenv(tmp_path, find=lambda start: "").ok


# --- Models (built, never asked to generate) -----------------------------------------------

@pytest.mark.parametrize("model", [WORKER, *BASELINES])
def test_models_construct_pass_the_roster_and_print_first_party_endpoints(model):
    checks = _by_name(pf.check_models(model, build=_build))
    assert all(c.ok for c in checks.values()), checks
    detail = checks["serving"].detail
    assert f"evaluated_model {model} -> https://api.anthropic.com" in detail
    assert f"grader {GRADER} -> https://api.openai.com/v1/" in detail
    assert f"calibrator {CALIBRATOR} -> https://api.anthropic.com" in detail


def test_a_model_that_cannot_construct_fails():
    def build(name):
        raise RuntimeError("no key\nsecond line")

    checks = pf.check_models(WORKER, build=build)
    assert [(c.name, c.ok) for c in checks] == [("construct", False), ("roster", False), ("serving", False)]
    assert "\n" not in checks[0].detail


def test_a_model_off_the_roster_fails_the_roster_check():
    checks = _by_name(pf.check_models(WORKER, build=lambda n: _build(GRADER if n == WORKER else n)))
    assert not checks["roster"].ok


@pytest.mark.parametrize("var, value", [
    ("ANTHROPIC_BASE_URL", "http://localhost:4000"),
    ("INSPECT_EVAL_EXTRA_HEADERS", '{"X-Review-Marker": "w2"}'),
    ("ANTHROPIC_CUSTOM_HEADERS", "X-Review-Marker: w2"),
])
def test_serving_guard_failures_surface(var, value, monkeypatch):
    monkeypatch.setenv(var, value)
    check = _by_name(pf.check_models(WORKER, build=_build))["serving"]
    assert not check.ok and "serving guard" in check.detail


# --- Log dir ------------------------------------------------------------------------------

def test_log_dir_outside_the_repo_passes(repo, tmp_path):
    path, _ = repo
    out = tmp_path / "logs"
    out.mkdir()
    assert pf.check_log_dir(str(out), path).ok


def test_log_dir_inside_the_repo_fails(repo):
    path, _ = repo
    inside = path / "runs"
    inside.mkdir()
    assert not pf.check_log_dir(str(inside), path).ok
    assert not pf.check_log_dir(str(path), path).ok


def test_missing_log_dir_fails(repo, tmp_path):
    path, _ = repo
    assert not pf.check_log_dir(str(tmp_path / "nope"), path).ok


def test_a_sibling_folder_sharing_the_name_prefix_is_outside(repo, tmp_path):
    path, _ = repo
    sibling = tmp_path / "repo-logs"
    sibling.mkdir()
    assert pf.check_log_dir(str(sibling), path).ok


def test_no_log_dir_means_no_check(repo):
    assert pf.check_log_dir(None, repo[0]) is None


# --- main ---------------------------------------------------------------------------------

def _main(repo, tmp_path, capsys, environ=CLEAN_ENV, python_version="3.14.10", extra=()):
    path, head = repo
    out = tmp_path / "logs"
    out.mkdir(exist_ok=True)
    code = pf.main(["--model", WORKER, "--pilot-commit", head, "--log-dir", str(out), *extra], repo=path,
                   environ=environ, python_version=python_version, build=_build, find_dotenv=lambda start: "")
    return code, capsys.readouterr().out


def test_main_passes_when_every_check_passes_and_never_generates(repo, tmp_path, capsys, monkeypatch):
    from inspect_ai.model import Model

    async def refuse(*a, **k):
        raise AssertionError("preflight called generate")

    monkeypatch.setattr(Model, "generate", refuse)
    code, out = _main(repo, tmp_path, capsys)
    lines = out.splitlines()
    assert code == 0, out
    assert [ln.split(" ", 2)[1].rstrip(":") for ln in lines[:-1]] == [
        "repo", "head", "clean", "python", "env", "keys", "dotenv", "construct", "roster", "serving", "log-dir",
    ]
    assert all(ln.startswith("PASS ") for ln in lines[:-1])
    assert lines[-1] == "preflight: PASS (11/11 checks)"
    assert DUMMY not in out


def test_main_fails_on_any_failing_check(repo, tmp_path, capsys):
    code, out = _main(repo, tmp_path, capsys, python_version="3.12.16")
    assert code == 1 and "FAIL python" in out and out.splitlines()[-1].startswith("preflight: FAIL")


def test_main_never_prints_a_value(repo, tmp_path, capsys):
    env = {**CLEAN_ENV, "ANTHROPIC_AUTH_TOKEN": DUMMY, "INSPECT_EVAL_EXTRA_HEADERS": DUMMY}
    code, out = _main(repo, tmp_path, capsys, environ=env)
    assert code == 1 and DUMMY not in out
    assert "ANTHROPIC_AUTH_TOKEN" in out and "INSPECT_EVAL_EXTRA_HEADERS" in out


def test_the_roster_exposes_the_names_the_preflight_uses():
    assert roster.EXTRA_HEADERS_ENV in pf.MUST_BE_UNSET
    assert roster.EVALUATED_KEY == "evaluated_model"
