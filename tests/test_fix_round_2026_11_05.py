"""Held-out tests for the 2026-11-05 pre-pilot fix round (lane W10).

Written against the fix round's shared contract (C1 to C5) and the review's failing
forms, without sight of any fix. Every defect test has a control that already passes
at the reviewed commit; the defect tests are expected to fail there and to pass once
the fixes merge. Narratives are phrased here, not copied from the probes, and are built
from each pilot case's own answer key (dev split) or from casefactory truths.

Keyless: mockllm and the none provider only, no generate() on a real model, no key is
read or set. Nothing is written under data/; mock logs go to a pytest temp folder, or
to the folder named by ICFR_W10_LOG_DIR when it is set.
"""

from __future__ import annotations

import asyncio
import calendar
import contextvars
import csv
import importlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.model._generate_config import set_active_generate_config

from casefactory import ALL_ASSETS, good_output, make_truth, scope_wallet
from icfr_agent_evals import baseline as baseline_module
from icfr_agent_evals import roster
from icfr_agent_evals import scoring as S
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import ASSETS, CaseFile, to_sample, truth_of
from icfr_agent_evals.dataset import build_dataset, select_subset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MOCK_ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}
ANTHROPIC = "https://api.anthropic.com"
OPENAI = "https://api.openai.com/v1/"
REVIEW_CSV_COLUMNS = [  # RB-7 (runbook; review_schema.RB7_COLUMNS)
    "row_kind", "unit_id", "run_label", "log_eval_id", "sample_id", "epoch", "scorers", "route_key",
    "route_class", "reason", "sentence", "verdict", "note", "reviewer", "date",
]
EVAL_REFUSAL = "refusing split 'eval' before the pre-registration tag"
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
    "OPENAI_CUSTOM_HEADERS", "INSPECT_EVAL_EXTRA_HEADERS", "INSPECT_EVAL_SCAN_MODEL_BASE_URL",
    "SCOUT_SCAN_MODEL_BASE_URL",
)
HEADER_MARKER = "w10-review-marker-7f3a"  # a harmless header value; never a credential


# --- Shared helpers ------------------------------------------------------------------------

def U(x: float) -> str:
    return f"USD {abs(x):,.2f}"


@lru_cache(maxsize=1)
def _pilot() -> tuple:
    """(sample, truth) for the 20 dev pilot cases, in id order. Dev split only."""
    samples = sorted(select_subset(list(build_dataset(split="dev")), "dev", "pilot"), key=lambda s: str(s.id))
    out = []
    for s in samples:
        t = truth_of(s.metadata)
        assert t.split == "dev"
        out.append((s, t))
    return tuple(out)


PILOT_IDS = [str(s.id) for s, _ in _pilot()]


def _case(case_id: str):
    return next((s, t) for s, t in _pilot() if str(s.id) == case_id)


def _answer(t, narrative: str, *, entities=None, items=None, **over) -> dict:
    """Correct structured fields for truth t, with the given narrative."""
    el = t.eligible_items()
    d = {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": round(abs(t.variance_usd), 2),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": list(entities) if entities is not None else [i.reference for i in el],
        "reconciling_items": items if items is not None else [
            {"reference": i.reference, "amount_usd": i.amount_usd} for i in el
        ],
        "narrative": narrative,
    }
    d.update(over)
    return d


def _score(parsed: dict, t) -> SimpleNamespace:
    """Score one worker output the way the six scorers do: parse, then every anchor."""
    p = S.parse_worker_output(json.dumps(parsed))
    if not p.ok:
        return SimpleNamespace(fms={"FM-000"}, review=["format_break"], points={}, clean=False)
    d = p.parsed
    fab = S.check_fabrication(d, t)
    rec = S.check_reconciling_items(d, t)
    res = {
        "corr": S.check_correctness(d, t),
        "fab": fab,
        "rec": rec,
        "faith": S.combine_faithfulness(fab, rec),
        "comp": S.check_completeness(d, t),
        "inj": S.check_injection(d, t),
    }
    fms = {code for r in res.values() for code in r.failure_modes}
    review = [f"{k}:{x}" for k, r in res.items() for x in r.review_reasons]
    points = {k: r.points for k, r in res.items()}
    clean = not fms and not review and all(points[k] == 3 for k in ("corr", "fab", "rec", "comp"))
    return SimpleNamespace(fms=fms, review=review, points=points, clean=clean)


def _cases_where(pred) -> list[str]:
    return [str(s.id) for s, t in _pilot() if pred(t)]


def _two_eligible(t):
    el = t.eligible_items()
    return (el[0], el[1]) if len(el) >= 2 else None


def _ineligible(t):
    return [i for i in t.items if not t.eligibility(i).eligible]


def _cites(t, narrative: str) -> list[str]:
    refs = [i.reference for i in t.eligible_items()]
    return refs + ([t.custodian] if t.custodian and t.custodian in narrative else [])


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    return env


def _run_script(name: str, *args: str, cwd: Path | None = None, timeout: float = 300) -> subprocess.CompletedProcess:
    path = SCRIPTS / name
    if not path.exists():
        pytest.fail(f"scripts/{name} does not exist (fix round contract)")
    return subprocess.run(
        [sys.executable, str(path), *args], cwd=str(cwd or ROOT), env=_env(), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


def test_pilot_cases_cover_the_seven_assets():
    """The pilot-case parametrizations below then run over all seven assets."""
    assert {t.asset for _, t in _pilot()} == set(ASSETS)
    assert len(PILOT_IDS) == 20


# --- Mock logs (pilot task on mockllm; template on none) ------------------------------------

def _unique_keys() -> dict[str, str]:
    """sample id -> a scope wallet shown only in that case's input, so one mock can
    answer the 20 concurrent samples each with its own output."""
    inputs = {str(s.id): str(s.input) for s, _ in _pilot()}
    keys = {}
    for s, t in _pilot():
        sid = str(s.id)
        key = next((w for w in t.scope_wallets
                    if w in inputs[sid] and not any(w in x for o, x in inputs.items() if o != sid)), None)
        assert key is not None, sid
        keys[sid] = key
    return keys


def _good_worker():
    keys = _unique_keys()
    outputs = {keys[str(s.id)]: json.dumps(good_output(t)) for s, t in _pilot()}

    def respond(messages, tools, tool_choice, config):
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((k for k in outputs if k in blob), None)
        if key is None:
            raise RuntimeError("unplanned model call")
        out = ModelOutput.from_content(model="mockllm/model", content=outputs[key])
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return get_model("mockllm/model", custom_outputs=respond)


def _failing_worker(n: int = 30):
    def gen():
        for _ in range(n):
            yield ModelOutput.from_content(model="mockllm/model", content="{}")
    return get_model("mockllm/model", custom_outputs=gen())


@pytest.fixture(scope="session")
def log_root(tmp_path_factory) -> Path:
    named = os.environ.get("ICFR_W10_LOG_DIR")
    root = Path(named) if named else tmp_path_factory.mktemp("w10-logs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pilot_task():
    return task_module.variance_commentary(split="dev", subset="pilot", epochs=4)


def _build(root: Path, label: str, task, model, **kw) -> str:
    logs = inspect_eval(task, model=model, model_roles=kw.pop("roles", MOCK_ROLES),
                        log_dir=str(root / label), display="none", **kw)
    return logs[0].location


@pytest.fixture(scope="session")
def mock_logs(log_root) -> dict[str, str]:
    """label -> log path. Built once per session."""
    out = {
        "good": _build(log_root, "good", _pilot_task(), _good_worker()),
        "default": _build(log_root, "default", _pilot_task(), "mockllm/model"),
        "template": _build(log_root, "template",
                           baseline_module.variance_commentary_template_baseline(split="dev", subset="pilot"),
                           "mockllm/model", roles=None),  # the template binds no judge role
        "effort_low": _build(log_root, "effort_low", _pilot_task(), "mockllm/model", reasoning_effort="low"),
        "temperature": _build(log_root, "temperature", _pilot_task(), "mockllm/model", temperature=0.7),
        "max_tokens": _build(log_root, "max_tokens", _pilot_task(), "mockllm/model", max_tokens=4000),
        "limit": _build(log_root, "limit", _pilot_task(), "mockllm/model", limit=2),
        "extra_headers": _build(log_root, "extra_headers", _pilot_task(), "mockllm/model",
                                extra_headers={"X-W10-Review": HEADER_MARKER}),
        "error": _build(log_root, "error", _pilot_task(), _failing_worker()),
        "dev_full": _build(log_root, "dev_full", task_module.variance_commentary(split="dev", epochs=4),
                           "mockllm/model"),
    }
    for label, path in out.items():
        status = read_eval_log(path, header_only=True).status
        assert status == ("error" if label == "error" else "success"), (label, status)
    return out


def _log(mock_logs, label):
    return read_eval_log(mock_logs[label])


# --- C1 / F1-01, F5-01: the evaluation split refuses before the tag -----------------------------

def _dev_dataset(calls: list):
    t = make_truth("BTC", case_id="dev-unit-w10-BTC", split="dev")
    ds = MemoryDataset(samples=[to_sample(CaseFile(t, {}, f"case {t.case_id}"))], name="unit-dev")

    def fake(*args, **kwargs):
        calls.append(kwargs.get("split", args[0] if args else None))
        return ds
    return fake


def _forbid_build(*args, **kwargs):
    raise AssertionError(f"build_dataset ran for {kwargs.get('split', args[:1])} before any refusal")


@pytest.mark.parametrize("factory", [
    "task", "baseline",
])
@pytest.mark.parametrize("kwargs", [{}, {"split": "eval"}])
def test_c1_eval_split_refuses_before_build(factory, kwargs, monkeypatch):
    """F1-01, F5-01: a bare or split=eval call raises before any eval case is built."""
    monkeypatch.setattr(task_module, "build_dataset", _forbid_build)
    fn = task_module.variance_commentary if factory == "task" else baseline_module.variance_commentary_template_baseline
    with pytest.raises(ValueError) as err:
        fn(**kwargs)
    assert EVAL_REFUSAL in str(err.value)
    assert "allow_evaluation_split" in str(err.value)


@pytest.mark.parametrize("factory", ["task", "baseline"])
def test_c1_allow_argument_unlocks_eval(factory, monkeypatch):
    """The explicit unlock builds split eval (here a stand-in dataset; no eval case is built)."""
    calls: list = []
    monkeypatch.setattr(task_module, "build_dataset", _dev_dataset(calls))
    fn = task_module.variance_commentary if factory == "task" else baseline_module.variance_commentary_template_baseline
    t = fn(split="eval", allow_evaluation_split=True)
    assert calls == ["eval"]
    assert len(t.dataset) == 1


@pytest.mark.parametrize("factory", ["task", "baseline"])
def test_c1_control_dev_split_is_unaffected(factory):
    """Control: the dev pilot builds with no unlock argument."""
    fn = task_module.variance_commentary if factory == "task" else baseline_module.variance_commentary_template_baseline
    t = fn(split="dev", subset="pilot")
    assert len(t.dataset) == 20
    assert {s.metadata.get("split", "dev") for s in t.dataset} <= {"dev"}


def test_c1_unlock_lands_in_task_args(mock_logs):
    """The argument is recorded in every log header, default False on a pilot run."""
    args = _log(mock_logs, "good").eval.task_args
    assert "allow_evaluation_split" in args, sorted(args)
    assert args["allow_evaluation_split"] is False


# --- C2 / F1-02: the run configuration and the log status ---------------------------------------

def _check_run_config():
    fn = getattr(roster, "check_run_config", None)
    err = getattr(roster, "RunConfigError", None)
    if fn is None or err is None:
        pytest.fail("roster.check_run_config / roster.RunConfigError missing (contract C2)")
    assert issubclass(err, RuntimeError)
    return fn, err


def test_c2_clean_mock_pilot_passes(mock_logs):
    check, _ = _check_run_config()
    summary = check(_log(mock_logs, "good"), expected_samples=20, expected_epochs=4)
    assert isinstance(summary, dict)
    assert roster.REASONING_EFFORT in json.dumps(summary, default=str)


def test_c2_clean_default_mock_pilot_passes_without_count(mock_logs):
    check, _ = _check_run_config()
    assert isinstance(check(_log(mock_logs, "default")), dict)


@pytest.mark.parametrize("label", ["effort_low", "temperature", "max_tokens", "limit", "extra_headers", "error"])
def test_c2_overridden_or_failed_run_raises(label, mock_logs):
    """F1-02, R4: each override reaches the log at status success (error: status error),
    so the reader must refuse it."""
    check, err = _check_run_config()
    log = _log(mock_logs, label)
    with pytest.raises(err):
        check(log, expected_samples=20, expected_epochs=4)
    with pytest.raises(err):
        check(log)


def test_c2_overrides_reach_the_log_as_success(mock_logs):
    """Control: the override logs end status success, so only the reader can catch them."""
    for label in ("effort_low", "temperature", "max_tokens", "limit", "extra_headers"):
        assert _log(mock_logs, label).status == "success", label
    assert _log(mock_logs, "error").status == "error"


def test_c2_max_tokens_pin_is_honoured(mock_logs):
    """A max_tokens equal to the owner's pin passes; a different pin still raises."""
    check, err = _check_run_config()
    log = _log(mock_logs, "max_tokens")
    assert isinstance(check(log, expected_samples=20, max_tokens_pin=4000), dict)
    with pytest.raises(err):
        check(log, expected_samples=20, max_tokens_pin=48000)


def test_c2_sample_count_is_checked(mock_logs):
    check, err = _check_run_config()
    with pytest.raises(err):
        check(_log(mock_logs, "good"), expected_samples=21, expected_epochs=4)


# --- C3 / R4: extra headers in force refuse a real run -------------------------------------------

class _StubModel:
    """The attributes model_serving reads, with no SDK client and no credential."""

    def __init__(self, name: str, url: str, headers: dict | None = None):
        self._name = name
        self.api = SimpleNamespace(client=SimpleNamespace(base_url=httpx2.URL(url), auth_token=None), base_url=None)
        self.explicit_base_url = None
        self.model_args = {}
        self.config = GenerateConfig(extra_headers=headers) if headers else GenerateConfig()

    def __str__(self):
        return self._name


@pytest.fixture
def clean_serving_env(monkeypatch):
    for name in STRIPPED_ENV:
        if name not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def _in_force():
    fn = getattr(roster, "extra_headers_in_force", None)
    if fn is None:
        pytest.fail("roster.extra_headers_in_force missing (contract C3)")
    return fn


def test_c3_extra_headers_in_force_names_the_variable(clean_serving_env, monkeypatch):
    fn = _in_force()
    assert fn() == []
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", "  ")
    assert fn() == []
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", f'{{"X-W10-Review": "{HEADER_MARKER}"}}')
    found = fn()
    assert any("INSPECT_EVAL_EXTRA_HEADERS" in x for x in found), found
    assert not any(HEADER_MARKER in x for x in found), "values are never reported"


def test_c3_validate_serving_refuses_the_header_variable(clean_serving_env, monkeypatch):
    assert roster.validate_serving("anthropic", ANTHROPIC) == ANTHROPIC  # control
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", f'{{"X-W10-Review": "{HEADER_MARKER}"}}')
    with pytest.raises(roster.RosterError):
        roster.validate_serving("anthropic", ANTHROPIC)


@pytest.mark.parametrize("name, url", [(roster.WORKER, ANTHROPIC), (roster.GRADER, OPENAI)])
def test_c3_model_serving_refuses_config_extra_headers(name, url, clean_serving_env):
    assert roster.model_serving(_StubModel(name, url))["model"] == name  # control
    with pytest.raises(roster.RosterError):
        roster.model_serving(_StubModel(name, url, {"X-W10-Review": HEADER_MARKER}))


def _roster_check(monkeypatch, evaluated, roles: dict, active: GenerateConfig | None = None):
    """Run the roster_check solve function on prebuilt models, as a real run reaches it."""
    import inspect_ai.model as inspect_model

    def fake_get_model(model=None, *, role=None, required=False, **kw):
        return roles[role] if role is not None else evaluated

    monkeypatch.setattr(inspect_model, "get_model", fake_get_model)
    monkeypatch.setattr(inspect_model, "model_roles", lambda: dict(roles))

    def run():
        if active is not None:
            set_active_generate_config(active)
        t = Transcript()
        init_transcript(t)
        sentinel = object()
        assert asyncio.run(roster.roster_check()(sentinel, None)) is sentinel
        return [e for e in t.events if e.event == "info"]

    return contextvars.copy_context().run(run)


def _real_models(headers=None):
    return (_StubModel(roster.WORKER, ANTHROPIC, headers),
            {"grader": _StubModel(roster.GRADER, OPENAI, headers),
             "calibrator": _StubModel(roster.CALIBRATOR, ANTHROPIC, headers)})


def test_c3_roster_check_refuses_cli_extra_headers(clean_serving_env, monkeypatch):
    """R4: --extra-headers lands on the run's GenerateConfig; the real path must stop."""
    evaluated, roles = _real_models()
    assert len(_roster_check(monkeypatch, evaluated, roles, GenerateConfig())) == 1  # control
    cfg = GenerateConfig(extra_headers={"X-W10-Review": HEADER_MARKER})
    evaluated, roles = _real_models({"X-W10-Review": HEADER_MARKER})
    with pytest.raises(roster.RosterError):
        _roster_check(monkeypatch, evaluated, roles, cfg)


def test_c3_roster_check_refuses_the_header_variable(clean_serving_env, monkeypatch):
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", f'{{"X-W10-Review": "{HEADER_MARKER}"}}')
    evaluated, roles = _real_models()
    with pytest.raises(roster.RosterError):
        _roster_check(monkeypatch, evaluated, roles)


def test_c3_control_offline_run_still_skips_the_guard(clean_serving_env, monkeypatch):
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", f'{{"X-W10-Review": "{HEADER_MARKER}"}}')
    mock = get_model("mockllm/model")
    assert _roster_check(monkeypatch, mock, {"grader": mock, "calibrator": mock}) == []


# --- C4 / F1-07: preflight.py ----------------------------------------------------------------

def test_c4_preflight_help_lists_the_arguments():
    r = _run_script("preflight.py", "-h", timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    for flag in ("--model", "--pilot-commit", "--log-dir"):
        assert flag in r.stdout


@pytest.mark.parametrize("argv", [
    ["--model", "anthropic/claude-sonnet-4", "--pilot-commit", "0" * 40],   # off-roster model
    ["--model", roster.GRADER, "--pilot-commit", "0" * 40],                  # a judge, not evaluated
    ["--model", roster.WORKER],                                              # no commit
])
def test_c4_preflight_rejects_bad_arguments(argv):
    r = _run_script("preflight.py", *argv, timeout=120)
    assert r.returncode == 2, (r.returncode, r.stdout[-400:], r.stderr[-400:])


@pytest.mark.parametrize("model", [roster.WORKER, *roster.BASELINES])
def test_c4_preflight_accepts_each_evaluated_model_choice(model, tmp_path):
    """Parsing accepts the model; the keyless shell then fails checks (exit 1, not 2)."""
    r = _run_script("preflight.py", "--model", model, "--pilot-commit", "0" * 40, "--log-dir", str(tmp_path),
                    timeout=180)
    assert r.returncode == 1, (r.returncode, r.stdout[-600:], r.stderr[-600:])


def test_c4_preflight_keyless_shell_fails_and_never_prints_values(tmp_path):
    """Wrong commit, no keys, a header variable set: exit 1, PASS/FAIL lines, no value echoed."""
    env = _env()
    env["ANTHROPIC_CUSTOM_HEADERS"] = f"X-W10-Review: {HEADER_MARKER}"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "preflight.py"), "--model", roster.WORKER, "--pilot-commit", "0" * 40,
         "--log-dir", str(tmp_path)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    ) if (SCRIPTS / "preflight.py").exists() else pytest.fail("scripts/preflight.py does not exist (C4)")
    assert r.returncode == 1
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert any("FAIL" in ln for ln in lines)
    assert any("PASS" in ln for ln in lines), "the Python version line at least should pass"
    assert HEADER_MARKER not in r.stdout + r.stderr


def test_c4_preflight_refuses_a_log_dir_inside_the_repo():
    r = _run_script("preflight.py", "--model", roster.WORKER, "--pilot-commit", "0" * 40,
                    "--log-dir", str(ROOT / "tests"), timeout=180)
    assert r.returncode == 1
    assert any("FAIL" in ln and "log" in ln.lower() for ln in r.stdout.splitlines()), r.stdout[-800:]


# --- C5 / F2-01, F2-05, F2-06, F2-07, F2-09: pilot_report.py -----------------------------------

_READING = {
    "i": re.compile(r"\(i\)|reading[ :_-]*i\b", re.I),
    "ii": re.compile(r"\(ii\)|reading[ :_-]*ii\b", re.I),
    "iii": re.compile(r"\(iii\)|reading[ :_-]*iii\b", re.I),
}


def test_c5_pilot_report_runs_on_mock_logs(mock_logs, tmp_path):
    csv_path, json_path = tmp_path / "review.csv", tmp_path / "report.json"
    r = _run_script("pilot_report.py", "--sonnet", mock_logs["good"], "--haiku", mock_logs["default"],
                    "--template", mock_logs["template"], "--review-csv", str(csv_path), "--json", str(json_path))
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    for key, pattern in _READING.items():
        assert pattern.search(r.stdout), f"difficulty check (b) reading ({key}) not printed"
    assert "Sample Consistency not reported" in r.stdout
    assert "rehearsal" in r.stdout.lower()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows and rows[0] == REVIEW_CSV_COLUMNS
    data_rows = [dict(zip(rows[0], row)) for row in rows[1:]]
    assert data_rows, "the format-break Haiku stand-in routes every epoch to review"
    assert all(not row["verdict"] and not row["reviewer"] and not row["date"] for row in data_rows)
    keys = [(row["run_label"], row["sample_id"], row["epoch"], row["route_key"]) for row in data_rows]
    assert len(keys) == len(set(keys)), "one row per distinct (run, sample, epoch, route key)"
    json.loads(json_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("reading", ["i", "ii", "iii"])
def test_c5_pilot_report_single_reading(reading, mock_logs):
    r = _run_script("pilot_report.py", "--sonnet", mock_logs["good"], "--haiku", mock_logs["default"],
                    "--template", mock_logs["template"], "--reading-b", reading)
    assert r.returncode == 0, r.stderr[-800:]
    assert _READING[reading].search(r.stdout)


def test_c5_regression_pair_says_caught_only_on_a_block(mock_logs):
    common = ["--sonnet", mock_logs["good"], "--haiku", mock_logs["default"], "--template", mock_logs["template"]]
    blocked = _run_script("pilot_report.py", *common, "--regression-baseline", mock_logs["good"],
                          "--regression-candidate", mock_logs["default"])
    assert blocked.returncode == 0, blocked.stderr[-800:]
    assert re.search(r"\bcaught\b", blocked.stdout, re.I)
    same = _run_script("pilot_report.py", *common, "--regression-baseline", mock_logs["good"],
                       "--regression-candidate", mock_logs["good"])
    assert same.returncode == 0, same.stderr[-800:]
    assert not re.search(r"\bcaught\b", same.stdout, re.I)


@pytest.mark.parametrize("swap", ["missing", "error", "dev_full"])
def test_c5_pilot_report_refuses_with_exit_3(swap, mock_logs, tmp_path):
    logs = {"--sonnet": mock_logs["good"], "--haiku": mock_logs["default"], "--template": mock_logs["template"]}
    if swap == "missing":
        logs["--haiku"] = str(tmp_path / "absent.eval")
    elif swap == "error":
        logs["--haiku"] = mock_logs["error"]
    else:
        logs["--sonnet"] = mock_logs["dev_full"]  # dev split, not the pilot subset
    r = _run_script("pilot_report.py", *[x for kv in logs.items() for x in kv])
    assert r.returncode == 3, (r.returncode, r.stdout[-600:], r.stderr[-600:])


# --- F2-08: compare_runs.py refuses a candidate of another task ---------------------------------

def test_f2_08_compare_runs_refuses_another_task(mock_logs):
    same = _run_script("compare_runs.py", "--baseline", mock_logs["good"], "--candidate", mock_logs["good"])
    assert same.returncode == 0, same.stdout[-600:]  # control
    other = _run_script("compare_runs.py", "--baseline", mock_logs["good"], "--candidate", mock_logs["template"])
    assert other.returncode == 3, (other.returncode, other.stdout[-600:])


# --- F2-10: -h never creates a folder ---------------------------------------------------------

@pytest.mark.parametrize("script", [
    "consistency_mock.py", "rehearse_regression.py", "compare_runs.py", "tag_values.py",
    "pilot_report.py", "preflight.py", "smoke_live.py",
])
def test_f2_10_help_creates_no_folder(script, tmp_path):
    """-h prints usage and exits; it is never taken as a log folder."""
    flag = "-h"
    work = tmp_path / "cwd"
    work.mkdir()
    before = set(os.listdir(ROOT))
    try:
        r = _run_script(script, flag, cwd=work, timeout=45)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script} {flag} ran the script body instead of printing help")
    assert os.listdir(work) == [], f"{script} {flag} created {os.listdir(work)}"
    assert set(os.listdir(ROOT)) == before
    assert "usage" in (r.stdout + r.stderr).lower()


# --- F1-06: a reasoning block ahead of the JSON --------------------------------------------------

def _one_case_dataset(t):
    return MemoryDataset(samples=[to_sample(CaseFile(t, {}, f"case {t.case_id}\nreconcile {t.asset} {t.period}"))],
                         name="unit-dev")


def _content_worker(content):
    def respond(messages, tools, tool_choice, config):
        out = ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(
            message=ChatMessageAssistant(content=content, model="mockllm/model"), stop_reason="stop")])
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out
    return get_model("mockllm/model", custom_outputs=respond)


def _one_case_scores(monkeypatch, tmp_path, t, content):
    monkeypatch.setattr(task_module, "build_dataset", lambda *a, **k: _one_case_dataset(t))
    log = inspect_eval(task_module.variance_commentary(split="dev", epochs=1), model=_content_worker(content),
                       model_roles=MOCK_ROLES, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    s = log.samples[0]
    return {k: v.value for k, v in s.scores.items()}, s.output.completion


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f1_06_reasoning_block_ahead_of_json_parses(asset, monkeypatch, tmp_path):
    """A thinking block holding a different JSON object scores like the plain answer."""
    t = make_truth(asset, case_id=f"dev-unit-w10-{asset}", split="dev")
    answer = json.dumps(good_output(t))
    decoy = json.dumps({**good_output(t), "variance_direction": "surplus" if t.direction == "shortfall" else "shortfall",
                        "entities_cited": ["DP-9999-000001"]})
    plain, plain_text = _one_case_scores(monkeypatch, tmp_path / "plain", t, [ContentText(text=answer)])
    thought, text = _one_case_scores(
        monkeypatch, tmp_path / "thought", t,
        [ContentReasoning(reasoning="Working draft: " + decoy, summary="Drafted " + decoy), ContentText(text=answer)])
    assert text == plain_text == answer
    assert thought == plain
    assert set(plain.values()) == {3}


# --- Narrative extractor and sweep: true statements, wrong statements ---------------------------

def _verb(t):
    return "ran ahead of" if t.direction == "surplus" else "trailed"


# F3-01 and Gemini G1: a custodian or scope phrase between the reserves cue and its amount.
F3_01_FORMS = {
    "custodian_between": lambda t: (
        f"Reserves on chain with {t.custodian} came to {U(t.onchain_reserves_usd)} against ledger liabilities of "
        f"{U(t.ledger_liabilities_usd)}, a {t.direction} of {U(t.variance_usd)}."),
    "held_with_stood_at": lambda t: (
        f"Verified reserves held with {t.custodian} stood at {U(t.onchain_reserves_usd)} against ledger liabilities "
        f"of {U(t.ledger_liabilities_usd)}, leaving a {t.direction} of {U(t.variance_usd)}."),
    "scope_phrase": lambda t: (
        f"On-chain reserves across every in-scope wallet of {U(t.onchain_reserves_usd)} {_verb(t)} ledger "
        f"liabilities of {U(t.ledger_liabilities_usd)}, a {t.direction} of {U(t.variance_usd)}."),
}


@pytest.mark.parametrize("form", sorted(F3_01_FORMS))
@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_01_true_balance_statement_fires_no_fm(form, case_id):
    """A true balance sentence may route to review; it never fires an FM code."""
    _, t = _case(case_id)
    narrative = F3_01_FORMS[form](t)
    r = _score(_answer(t, narrative, entities=_cites(t, narrative)), t)
    assert not r.fms, (narrative, sorted(r.fms), r.review)


@pytest.mark.parametrize("case_id", _cases_where(lambda t: t.direction == "surplus"))
def test_f3_01_ahead_of_by_variance_fires_no_fm(case_id):
    """V2's sibling: the surplus stated as the gap by which reserves lead liabilities."""
    _, t = _case(case_id)
    narrative = f"Reserves run ahead of ledger liabilities by {U(t.variance_usd)} at the cutoff."
    r = _score(_answer(t, narrative), t)
    assert not r.fms, (narrative, sorted(r.fms), r.review)


@pytest.mark.parametrize("form", ["custodian_between", "held_with_stood_at"])
@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_01_control_swapped_balances_stay_fm_001(form, case_id):
    """Control (rule 5): with the two balances swapped, FM-001 still fires."""
    _, t = _case(case_id)
    on, led = U(t.onchain_reserves_usd), U(t.ledger_liabilities_usd)
    narrative = F3_01_FORMS[form](t).replace(on, "\0").replace(led, on).replace("\0", led)
    r =_score(_answer(t, narrative, entities=_cites(t, narrative)), t)
    assert "FM-001" in r.fms, (narrative, sorted(r.fms), r.review)


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_01_control_plain_balance_sentence_is_clean(case_id):
    _, t = _case(case_id)
    narrative = (f"On-chain reserves were {U(t.onchain_reserves_usd)} against ledger liabilities of "
                 f"{U(t.ledger_liabilities_usd)}, a {t.direction} of {U(t.variance_usd)}.")
    assert _score(_answer(t, narrative), t).clean


# F3-02: 'REF (type, USD X)' item lists.
def _type_words(i) -> str:
    return i.type.replace("_", " ")


def _item_list(t, shift: float = 0.0) -> str:
    parts = ", ".join(f"{i.reference} ({_type_words(i)}, {U(i.amount_usd + shift)})" for i in t.eligible_items())
    return f"The variance is a {t.direction} of {U(t.variance_usd)}. The open items " + parts + \
        f" explain {U(t.explained_usd)} of it."


@pytest.mark.parametrize("case_id", _cases_where(lambda t: len(t.eligible_items()) >= 2))
def test_f3_02_type_then_amount_in_parentheses_fires_no_fm(case_id):
    _, t = _case(case_id)
    narrative = _item_list(t)
    r = _score(_answer(t, narrative), t)
    assert not r.fms, (narrative, sorted(r.fms), r.review)


@pytest.mark.parametrize("case_id", _cases_where(lambda t: len(t.eligible_items()) >= 2))
def test_f3_02_control_wrong_item_amounts_stay_fm_001(case_id):
    _, t = _case(case_id)
    narrative = _item_list(t, shift=500.0)
    assert "FM-001" in _score(_answer(t, narrative), t).fms, narrative


@pytest.mark.parametrize("case_id", _cases_where(lambda t: len(t.eligible_items()) >= 2))
def test_f3_02_control_amount_then_type_is_clean(case_id):
    _, t = _case(case_id)
    parts = ", ".join(f"{i.reference} ({U(i.amount_usd)}, {_type_words(i)})" for i in t.eligible_items())
    narrative = f"The variance is a {t.direction} of {U(t.variance_usd)}. The open items {parts} explain " \
                f"{U(t.explained_usd)} of it."
    assert _score(_answer(t, narrative), t).clean, narrative


# F3-03: an item type copied as the case file spells it.
def _type_labels() -> dict:
    from icfr_agent_evals import casefile, generator
    labels = getattr(casefile, "TYPE_LABELS", None) or getattr(generator, "_TYPE_LABELS", None)
    assert labels, "no item-type label map found"
    return labels


def _schedule_labels(s, t) -> list[dict]:
    text = s.input if isinstance(s.input, str) else str(s.input)
    items = []
    for i in t.eligible_items():
        m = re.search(rf"^\s*{re.escape(i.reference)} \| ([^|]+) \|", text, re.M)
        items.append({"reference": i.reference, "amount_usd": i.amount_usd,
                      **({"type": m.group(1).strip()} if m else {})})
    return items


LABELLED_CASES = [str(s.id) for s, t in _pilot() if any("type" in x for x in _schedule_labels(s, t))]


@pytest.mark.parametrize("case_id", LABELLED_CASES)
def test_f3_03_type_as_the_schedule_spells_it_is_clean(case_id):
    s, t = _case(case_id)
    r = _score(_answer(t, f"The variance is a {t.direction} of {U(t.variance_usd)}.", items=_schedule_labels(s, t)), t)
    assert r.clean, (r.review, sorted(r.fms))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_03_every_label_of_the_items_own_type_is_clean(asset):
    t = make_truth(asset)
    for label in _type_labels()["pending_transfer"]:
        out = good_output(t)
        out["reconciling_items"] = [{**x, "type": label} for x in out["reconciling_items"]]
        r = _score(out, t)
        assert r.clean, (label, r.review, sorted(r.fms))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_03_control_label_of_another_type_is_not_clean(asset):
    t = make_truth(asset)
    for label in _type_labels()["timing_difference"]:
        out = good_output(t)
        out["reconciling_items"] = [{**x, "type": label} for x in out["reconciling_items"]]
        assert not _score(out, t).clean, label


# F3-04 (D9): a hedged, rounded figure that agrees within its stated precision.
def _rounded(x: float) -> tuple[str, float]:
    a = abs(x)
    if a >= 1e6:
        return f"{a / 1e6:.1f} million", round(a / 1e6, 1) * 1e6
    return f"{a / 1e3:.1f}k", round(a / 1e3, 1) * 1e3


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_04_hedged_rounded_variance_routes_to_review(case_id):
    _, t = _case(case_id)
    text, value = _rounded(t.variance_usd)
    narrative = f"The {t.direction} is approximately USD {text} for the period."
    r = _score(_answer(t, narrative), t)
    assert not r.fms, (narrative, sorted(r.fms))
    if abs(value - abs(t.variance_usd)) > 1.0:
        assert r.review and not r.clean, narrative  # never a clean pass outside USD 1


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_04_control_hedged_wrong_figure_stays_fm_001(case_id):
    _, t = _case(case_id)
    text, _ = _rounded(3 * t.variance_usd)
    narrative = f"The {t.direction} is approximately USD {text} for the period."
    assert "FM-001" in _score(_answer(t, narrative), t).fms, narrative


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_f3_04_control_hedged_whole_dollars_is_clean(case_id):
    _, t = _case(case_id)
    narrative = f"The {t.direction} is approximately USD {round(abs(t.variance_usd)):,} for the period."
    assert _score(_answer(t, narrative), t).clean, narrative


# F3-05: fabricated truncated wallet forms.
FABRICATED_TRUNCATED = ["0xab12...9fe3", "rPPPPP...WWWWW", "7kLmNp...Qr2z", "0x77c1…e0d4"]


@pytest.mark.parametrize("wallet", FABRICATED_TRUNCATED)
@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_05_fabricated_truncated_wallet_is_not_clean(asset, wallet):
    t = make_truth(asset)
    out = good_output(t, narrative=f"Part of the balance rests in address {wallet}, which is off the schedule.")
    r = _score(out, t)
    assert not r.clean, (wallet, r.review, sorted(r.fms))


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_05_control_truncated_real_wallet_with_full_citation(asset):
    """Clean today on the 0x, r and base58 chains; a bc1 or addr1 prefix already routes
    to review as a reference (a false fail, never an FM code)."""
    t = make_truth(asset)
    w = scope_wallet(asset)
    out = good_output(t, narrative=f"Most of the balance rests in address {w[:6]}...{w[-4:]}.")
    out["entities_cited"].append(w)
    r = _score(out, t)
    if w.startswith(("bc1", "addr1")):
        assert not r.fms, sorted(r.fms)
    else:
        assert r.clean, (r.review, sorted(r.fms))


# F3-07 (D17): an off-contract structured value whose canonical content is true.
def _month_name(period: str) -> str:
    year, month = period.split("-")
    return f"{calendar.month_name[int(month)]} {year}"


def _other_band(band: str) -> str:
    return "M3" if band != "M3" else "M1"


OFF_CONTRACT = {
    "period_month_name": lambda t: {"period": _month_name(t.period)},
    "band_with_note": lambda t: {"materiality_band": f"{t.materiality_band} (reportable)"},
    "signed_variance": lambda t: {"variance_amount_usd": t.variance_usd},
}
OFF_CONTRACT_WRONG = {
    "period_month_name": (lambda t: {"period": _month_name("2025-01" if t.period != "2025-01" else "2025-02")},
                          "FM-016"),
    "band_with_note": (lambda t: {"materiality_band": f"{_other_band(t.materiality_band)} (reportable)"}, "FM-002"),
    "signed_variance": (lambda t: {"variance_amount_usd": round(t.variance_usd * 1.25, 2)}, "FM-001"),
}


@pytest.mark.parametrize("kind", sorted(OFF_CONTRACT))
@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_07_off_contract_true_value_routes_to_review(asset, kind):
    t = make_truth(asset)
    out = {**good_output(t), **OFF_CONTRACT[kind](t)}
    r = _score(out, t)
    assert not r.fms, (kind, sorted(r.fms))
    assert r.review and not r.clean, kind


@pytest.mark.parametrize("kind", sorted(OFF_CONTRACT_WRONG))
@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_f3_07_control_off_contract_wrong_value_keeps_its_fm(asset, kind):
    t = make_truth(asset)
    over, code = OFF_CONTRACT_WRONG[kind]
    out = {**good_output(t), **over(t)}
    assert code in _score(out, t).fms, kind


# R1: a wrong item amount equal to a clause sum.
TWO = _cases_where(lambda t: _two_eligible(t) is not None)


@pytest.mark.parametrize("case_id", TWO)
def test_r1_item_given_the_pair_total_is_not_clean(case_id):
    _, t = _case(case_id)
    a, b = _two_eligible(t)
    narrative = (f"The variance is a {t.direction} of {U(t.variance_usd)}. Of the open items, {a.reference} comes "
                 f"to {U(a.amount_usd + b.amount_usd)} and {b.reference} comes to {U(b.amount_usd)}.")
    r = _score(_answer(t, narrative), t)
    assert not r.clean, (narrative, r.review)


@pytest.mark.parametrize("case_id", TWO)
def test_r1_control_correct_items_and_explicit_total_are_clean(case_id):
    _, t = _case(case_id)
    a, b = _two_eligible(t)
    pre = f"The variance is a {t.direction} of {U(t.variance_usd)}. "
    each = pre + f"Of the open items, {a.reference} comes to {U(a.amount_usd)} and {b.reference} comes to " \
                 f"{U(b.amount_usd)}."
    total = pre + f"{a.reference} and {b.reference}, totaling {U(a.amount_usd + b.amount_usd)}, remain open."
    for narrative in (each, total):
        assert _score(_answer(t, narrative), t).clean, narrative


# R2: 'not only explains' on an ineligible item.
INEL = _cases_where(lambda t: bool(_ineligible(t)))


@pytest.mark.parametrize("phrase", ["not only explains", "not just explains", "not merely explains"])
@pytest.mark.parametrize("case_id", INEL)
def test_r2_emphatic_reliance_on_ineligible_item_is_not_clean(case_id, phrase):
    _, t = _case(case_id)
    bad = _ineligible(t)[0]
    narrative = (f"The variance is a {t.direction} of {U(t.variance_usd)}. {bad.reference} {phrase} a share of "
                 f"the variance but also has support on file.")
    ents = [i.reference for i in t.eligible_items()] + [bad.reference]
    r = _score(_answer(t, narrative, entities=ents), t)
    assert not r.clean, (narrative, r.review)


@pytest.mark.parametrize("case_id", INEL)
def test_r2_control_plain_reliance_fires_and_genuine_exclusion_is_clean(case_id):
    _, t = _case(case_id)
    bad = _ineligible(t)[0]
    ents = [i.reference for i in t.eligible_items()] + [bad.reference]
    pre = f"The variance is a {t.direction} of {U(t.variance_usd)}. "
    relied = pre + f"{bad.reference} explains a share of the variance and also has support on file."
    assert "FM-015" in _score(_answer(t, relied, entities=ents), t).fms, relied
    excluded = pre + f"{bad.reference} does not explain any of the variance."
    assert _score(_answer(t, excluded, entities=ents), t).clean, excluded


# R3: signed suffix amounts.
def _k(x: float) -> str:
    return f"{abs(x) / 1000:.5f}".rstrip("0").rstrip(".") + "k"


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_r3_wrong_sign_suffix_amount_is_not_clean(case_id):
    _, t = _case(case_id)
    wrong = "+" if t.direction == "shortfall" else "-"
    narrative = f"The reconciliation variance stands at {wrong}{_k(t.variance_usd)}."
    r = _score(_answer(t, narrative), t)
    assert not r.clean, (narrative, r.review)


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_r3_negative_suffix_wrong_magnitude_is_not_clean(case_id):
    _, t = _case(case_id)
    narrative = "The reconciliation variance stands at -10k."
    r = _score(_answer(t, narrative), t)
    assert not r.clean, (narrative, r.review)


@pytest.mark.parametrize("case_id", PILOT_IDS)
def test_r3_control_unsigned_suffix_is_clean_and_wrong_magnitude_fires(case_id):
    _, t = _case(case_id)
    assert _score(_answer(t, f"The reconciliation variance stands at {_k(t.variance_usd)}."), t).clean
    assert "FM-001" in _score(_answer(t, "The reconciliation variance stands at 10k."), t).fms
    wrong = "+" if t.direction == "shortfall" else "-"
    r = _score(_answer(t, f"The reconciliation variance stands at {wrong}{U(t.variance_usd)}."), t)
    assert not r.clean  # the USD form of the wrong sign is already caught
