"""The grader call and its first live check (pre-registration section 5 grader
configuration, section 7 test-retest, section 13 tooling row item 2; decisions D-T1,
D-R12 and D-R19).

Keyless: every model here is Inspect's mockllm or a monkeypatched stand-in, and no
test reaches a network. The mock logs carry made-up sample ids and no case content.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState, solver

from icfr_agent_evals import grader, roster
from icfr_agent_evals.grader import GraderError
from icfr_agent_evals.roster import RosterError, RunConfigError

REPO = Path(__file__).resolve().parents[1]
MOCK = "mockllm/model"
PINS = {1: "openai/gpt-6-luna", 2: "openai/gpt-6.1-terra", 3: "openai/gpt-6.1-sol"}
STEPS = (1, 2, 3)


def _load_script():
    key = "grader_live_check_under_test"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO / "scripts" / "grader_live_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


live = _load_script()


# --- Pins ------------------------------------------------------------------------------

@pytest.mark.parametrize("step", STEPS)
def test_pin_comes_from_the_roster_ladder(step):
    assert grader.grader_pin(step) == roster.GRADER_BY_STEP[step] == PINS[step]


@pytest.mark.parametrize("step", [0, 4, "1", None])
def test_a_step_off_the_ladder_is_refused(step):
    with pytest.raises(RosterError, match="grader_step"):
        grader.grader_pin(step)
    with pytest.raises(RosterError, match="grader_step"):
        grader.grader_config(step)


@pytest.mark.parametrize("step", STEPS)
def test_config_equals_the_pin_and_nothing_else(step):
    config = grader.grader_config(step)
    assert config.model_dump(exclude_none=True) == roster.grader_generate_config_args(step)
    assert config.reasoning_effort == "high"
    assert config.batch is (True if step in (2, 3) else None)
    assert config.temperature is None and config.top_p is None and config.top_k is None
    assert config.seed is None and config.max_tokens is None and config.cache is None


def test_tag_time_resolved_strings_match_the_section_13_record():
    assert grader.TAG_TIME_RESOLVED == {1: "gpt-6-luna", 2: "gpt-6.1-terra", 3: "gpt-6.1-sol"}


def test_grader_error_is_a_roster_error():
    assert issubclass(GraderError, RosterError)


# --- check_grader_config ---------------------------------------------------------------

@pytest.mark.parametrize("step", STEPS)
def test_check_accepts_the_pin(step):
    assert grader.check_grader_config(grader.grader_config(step), step) == roster.grader_generate_config_args(step)
    assert grader.check_grader_config(roster.grader_generate_config_args(step), step)


@pytest.mark.parametrize("step", STEPS)
def test_check_accepts_operational_fields(step):
    config = grader.grader_config(step).merge(GenerateConfig(max_retries=3, timeout=600, max_connections=4))
    assert grader.check_grader_config(config, step)["max_retries"] == 3


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("extra, needle", [
    ({"temperature": 0.0}, "temperature=0.0"),
    ({"top_p": 0.9}, "top_p=0.9"),
    ({"top_k": 5}, "top_k=5"),
    ({"seed": 7}, "seed=7"),
    ({"reasoning_tokens": 1000}, "reasoning_tokens=1000"),
    ({"max_tokens": 100}, "max_tokens=100"),
    ({"cache": True}, "cache=True"),
    ({"extra_headers": {"X-Probe": "1"}}, "extra_headers"),
    ({"system_message": "be kind"}, "system_message="),
])
def test_check_refuses_an_off_pin_field(step, extra, needle):
    config = {**roster.grader_generate_config_args(step), **extra}
    with pytest.raises(RunConfigError, match=needle.replace("(", r"\(").replace(".", r"\.")):
        grader.check_grader_config(config, step)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("effort", [None, "none", "low", "medium", "xhigh"])
def test_check_refuses_an_effort_off_the_pin(step, effort):
    config = {**roster.grader_generate_config_args(step), "reasoning_effort": effort}
    with pytest.raises(RunConfigError, match="reasoning_effort"):
        grader.check_grader_config(config, step)


def test_check_refuses_batch_at_step_1():
    with pytest.raises(RunConfigError, match="batch is not allowed"):
        grader.check_grader_config({"reasoning_effort": "high", "batch": True}, 1)


def test_check_accepts_batch_false_at_step_1():
    assert grader.check_grader_config({"reasoning_effort": "high", "batch": False}, 1)["batch"] is False


@pytest.mark.parametrize("step", [2, 3])
@pytest.mark.parametrize("batch", [None, False, 1, 50])
def test_check_refuses_steps_2_and_3_off_the_batch_pin(step, batch):
    config = {"reasoning_effort": "high", "batch": batch}
    with pytest.raises(RunConfigError, match="Batch API"):
        grader.check_grader_config(config, step)


def test_check_reads_none_as_unset():
    assert grader.check_grader_config({"reasoning_effort": "high", "temperature": None}, 1) == {"reasoning_effort": "high"}


# --- Offline pairing and model checks --------------------------------------------------

@pytest.mark.parametrize("grader_model, evaluated", [
    (MOCK, None), (MOCK, MOCK), (MOCK, grader.NO_MODEL),
    (PINS[1], None), (PINS[1], grader.NO_MODEL), (PINS[1], roster.WORKER),
])
def test_offline_pairing_accepts(grader_model, evaluated):
    grader.check_offline_pairing(grader_model, evaluated)


@pytest.mark.parametrize("grader_model, evaluated", [(MOCK, roster.WORKER), (PINS[1], MOCK)])
def test_offline_pairing_refuses_a_mixed_run(grader_model, evaluated):
    with pytest.raises(GraderError, match="mixing real and mockllm"):
        grader.check_offline_pairing(grader_model, evaluated)


@pytest.mark.parametrize("step", STEPS)
def test_model_check_accepts_the_pin_and_mockllm(step):
    grader.check_grader_model(PINS[step], step)
    grader.check_grader_model(MOCK, step)


@pytest.mark.parametrize("name", ["openai/gpt-6.1-terra", "openai/gpt-6-luna-2026", roster.CALIBRATOR,
                                  "openai/azure/gpt-6-luna"])
def test_model_check_refuses_another_model_at_step_1(name):
    with pytest.raises(GraderError, match="not the step 1 pin"):
        grader.check_grader_model(name, 1)


# --- grade(): the call -----------------------------------------------------------------

class _Recorder:
    """Stands in for an Inspect Model; records what generate() was given."""

    def __init__(self, name, role="grader", own=None):
        self.name, self.role, self.calls = name, role, []
        self.config = own or GenerateConfig()

    def __str__(self):
        return self.name

    def _resolve_config(self, config):
        return self.config.merge(config)

    async def generate(self, input, tools=(), tool_choice=None, config=GenerateConfig(), cache=None, **kw):
        self.calls.append({"input": input, "tools": tools, "config": config, "cache": cache, **kw})
        return ModelOutput.from_content(self.name, "OK")


def _patch_get_model(monkeypatch, model, seen=None):
    def fake_get_model(name=None, *, role=None, **kw):
        if seen is not None:
            seen.append((name, role, kw))
        return model

    monkeypatch.setattr("inspect_ai.model.get_model", fake_get_model)


@pytest.mark.parametrize("step", STEPS)
def test_grade_passes_exactly_the_pin_with_cache_off(step, monkeypatch):
    model = _Recorder(MOCK)
    seen = []
    _patch_get_model(monkeypatch, model, seen)
    output = asyncio.run(grader.grade("Reply with OK.", step))
    assert output.completion == "OK"
    assert seen == [(PINS[step], "grader", {})]  # resolved from the ladder, with the grader role
    (call,) = model.calls
    assert call["cache"] is False
    assert call["config"].model_dump(exclude_none=True) == roster.grader_generate_config_args(step)
    assert call["config"].temperature is None
    assert not call["tools"]
    assert call["input"] == "Reply with OK."


def test_grade_passes_the_callers_messages_unchanged(monkeypatch):
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    model = _Recorder(MOCK)
    _patch_get_model(monkeypatch, model)
    messages = [ChatMessageSystem(content="caller system"), ChatMessageUser(content="caller user")]
    asyncio.run(grader.grade(messages, 1))
    assert model.calls[0]["input"] == messages


@pytest.mark.parametrize("messages", ["", "   ", []])
def test_grade_refuses_empty_messages(messages, monkeypatch):
    model = _Recorder(MOCK)
    _patch_get_model(monkeypatch, model)
    with pytest.raises(GraderError, match="message"):
        asyncio.run(grader.grade(messages, 1))
    assert model.calls == []


def test_grade_refuses_a_model_off_the_pin_before_any_call(monkeypatch):
    model = _Recorder("openai/gpt-6.1-sol")
    _patch_get_model(monkeypatch, model)
    with pytest.raises(GraderError, match="not the step 1 pin"):
        asyncio.run(grader.grade("x", 1))
    assert model.calls == []


def test_grade_refuses_a_model_without_the_grader_role(monkeypatch):
    model = _Recorder(MOCK, role=None)
    _patch_get_model(monkeypatch, model)
    with pytest.raises(GraderError, match="carries role None"):
        asyncio.run(grader.grade("x", 1))
    assert model.calls == []


@pytest.mark.parametrize("own, needle", [
    (GenerateConfig(temperature=0.5), "temperature=0.5"),
    (GenerateConfig(seed=3), "seed=3"),
    (GenerateConfig(max_tokens=64), "max_tokens=64"),
    (GenerateConfig(extra_headers={"X-Probe": "1"}), "extra_headers"),
])
def test_grade_refuses_an_off_pin_model_config_before_any_call(own, needle, monkeypatch):
    # A role spec can carry its own config; the call refuses before it is made.
    model = _Recorder(MOCK, own=own)
    _patch_get_model(monkeypatch, model)
    with pytest.raises(RunConfigError, match=needle.replace(".", r"\.")):
        asyncio.run(grader.grade("x", 1))
    assert model.calls == []


def test_grade_refuses_a_real_grader_that_fails_the_serving_guard(monkeypatch):
    model = _Recorder(PINS[1])
    _patch_get_model(monkeypatch, model)

    def refuse(m):
        raise RosterError("serving guard: probe")

    monkeypatch.setattr(roster, "model_serving", refuse)
    with pytest.raises(RosterError, match="serving guard: probe"):
        asyncio.run(grader.grade("x", 1))
    assert model.calls == []


def test_grade_on_a_real_pin_applies_the_serving_guard_and_records_it(monkeypatch):
    from inspect_ai.log import transcript

    model = _Recorder(PINS[2])
    _patch_get_model(monkeypatch, model)
    checked = []

    def serving(m):
        checked.append(str(m))
        return {"model": str(m), "base_url": "https://api.openai.com/v1/"}

    monkeypatch.setattr(roster, "model_serving", serving)
    async def call_and_read():
        await grader.grade("x", 2)
        return list(transcript().events)

    events = asyncio.run(call_and_read())
    assert checked == [PINS[2]]
    assert model.calls[0]["config"].batch is True and model.calls[0]["cache"] is False
    infos = [e for e in events if getattr(e, "source", None) == grader.GRADER_EVENT_SOURCE]
    assert infos and infos[-1].data["serving_endpoint"]["model"] == PINS[2]
    assert infos[-1].data["grader_step"] == 2


def test_resolve_refuses_a_role_bound_to_a_list(monkeypatch):
    monkeypatch.setattr("inspect_ai.model.model_roles", lambda: {"grader": [object(), object()]})
    with pytest.raises(GraderError, match="more than one model"):
        grader.resolve_grader(1)


def test_resolve_outside_an_eval_builds_the_mock_with_the_grader_role_only_when_bound(monkeypatch):
    # Unbound, the pin itself is resolved (here intercepted, never built for real).
    seen = []
    _patch_get_model(monkeypatch, _Recorder(MOCK), seen)
    grader.resolve_grader(3)
    assert seen == [(PINS[3], "grader", {})]


# --- grade() inside a real mockllm eval -------------------------------------------------

def _grading_task(step):
    @solver(name="grade_once")
    def grade_once():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            state.output = await grader.grade(state.messages, step)
            return state

        return solve

    return Task(dataset=MemoryDataset([Sample(id="dev-mock-001", input="probe", metadata={"split": "dev"})]),
                solver=[grade_once()], epochs=1)


def _mock_grader_log(tmp_path, step, model=MOCK, roles=None, **kw):
    roles = {"grader": MOCK} if roles is None else roles
    return inspect_eval(_grading_task(step), model=model, model_roles=roles, log_dir=str(tmp_path),
                        display="none", **kw)[0]


def _model_events(log):
    return [e for s in log.samples for e in s.events if getattr(e, "event", None) == "model"]


@pytest.mark.parametrize("step", STEPS)
def test_mock_eval_records_the_pinned_config_on_the_grader_event(step, tmp_path):
    log = _mock_grader_log(tmp_path, step)
    assert log.status == "success", log.error
    (event,) = _model_events(log)
    assert event.role == "grader" and event.cache is None
    assert event.config.model_dump(exclude_none=True) == roster.grader_generate_config_args(step)


def test_mock_eval_eval_wide_cache_is_refused_before_the_grader_call(tmp_path):
    # An eval-wide cache setting is inherited by role models. The grader call passes
    # cache=False, which would win, but the config it would record still carries the
    # setting, so the call is refused before it is made and no ModelEvent is written.
    log = _mock_grader_log(tmp_path, 1, cache=True)
    assert log.status == "error"
    assert "cache=True" in str(log.error)
    assert _model_events(log) == []


def test_a_real_evaluated_model_refuses_a_mock_grader(monkeypatch):
    class Active:
        def __str__(self):
            return roster.WORKER

    monkeypatch.setattr("inspect_ai.model._model.active_model", lambda: Active())
    monkeypatch.setattr("inspect_ai.model.model_roles", lambda: {})
    _patch_get_model(monkeypatch, _Recorder(MOCK))
    with pytest.raises(GraderError, match="mixing real and mockllm"):
        grader.resolve_grader(1)


# --- check_grader_event / check_grader_log ---------------------------------------------

def _on_pin(log, step, resolved=None):
    """A copy of a mockllm grader log whose grader events name the step's pin and whose
    responses record a resolved string, as a live log would."""
    log = log.model_copy(deep=True)
    for s in log.samples:
        for e in s.events:
            if getattr(e, "event", None) == "model" and e.role == "grader":
                e.model = PINS[step]
                e.output.model = resolved if resolved is not None else grader.TAG_TIME_RESOLVED[step]
    return log


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory):
    return {step: inspect_eval(_grading_task(step), model=MOCK, model_roles={"grader": MOCK},
                               log_dir=str(tmp_path_factory.mktemp(f"grader-{step}")), display="none")[0]
            for step in STEPS}


@pytest.mark.parametrize("step", STEPS)
def test_log_check_accepts_a_mock_grader_log_on_the_pin(step, mock_logs):
    (summary,) = grader.check_grader_log(_on_pin(mock_logs[step], step), step)
    assert summary["model"] == PINS[step]
    assert summary["resolved"] == grader.TAG_TIME_RESOLVED[step]
    assert summary["config"] == roster.grader_generate_config_args(step)
    assert summary["sample_id"] == "dev-mock-001" and summary["epoch"] == 1


@pytest.mark.parametrize("step", STEPS)
def test_log_check_refuses_the_mock_model_name(step, mock_logs):
    with pytest.raises(RunConfigError, match="not the step"):
        grader.check_grader_log(mock_logs[step], step)


def test_log_check_refuses_a_step_1_log_read_as_step_2(mock_logs):
    log = _on_pin(mock_logs[1], 1)
    for e in _model_events(log):
        e.model = PINS[2]
    with pytest.raises(RunConfigError, match="Batch API"):
        grader.check_grader_log(log, 2)


def _edit_config(log, step, **fields):
    log = _on_pin(log, step)
    for e in _model_events(log):
        e.config = e.config.merge(GenerateConfig(**fields))
    return log


@pytest.mark.parametrize("fields, needle", [
    ({"temperature": 0.7}, "temperature=0.7"),
    ({"reasoning_effort": "low"}, "reasoning_effort='low'"),
    ({"max_tokens": 50}, "max_tokens=50"),
    ({"seed": 1}, "seed=1"),
    ({"batch": True}, "batch is not allowed"),
    ({"extra_headers": {"X-Probe": "1"}}, "extra_headers"),
])
def test_log_check_refuses_a_step_1_log_off_the_pin(fields, needle, mock_logs):
    with pytest.raises(RunConfigError, match=needle.replace(".", r"\.")):
        grader.check_grader_log(_edit_config(mock_logs[1], 1, **fields), 1)


@pytest.mark.parametrize("step", [2, 3])
def test_log_check_refuses_a_batch_step_run_without_batch(step, mock_logs):
    log = _on_pin(mock_logs[step], step)
    for e in _model_events(log):
        e.config = GenerateConfig(reasoning_effort="high")
    with pytest.raises(RunConfigError, match="Batch API"):
        grader.check_grader_log(log, step)


@pytest.mark.parametrize("cache", ["read", "write"])
def test_log_check_refuses_a_cached_grader_call(cache, mock_logs):
    log = _on_pin(mock_logs[1], 1)
    for e in _model_events(log):
        e.cache = cache
    with pytest.raises(RunConfigError, match="caching is disabled"):
        grader.check_grader_log(log, 1)


def test_log_check_refuses_an_errored_call(mock_logs):
    log = _on_pin(mock_logs[1], 1)
    for e in _model_events(log):
        e.error = "probe failure"
    with pytest.raises(RunConfigError, match="probe failure"):
        grader.check_grader_log(log, 1)


def test_log_check_refuses_a_log_with_no_grader_call(mock_logs):
    log = _on_pin(mock_logs[1], 1)
    for e in _model_events(log):
        e.role = None
    with pytest.raises(RunConfigError, match="no grader ModelEvent"):
        grader.check_grader_log(log, 1)


def test_log_check_refuses_an_unfinished_or_header_only_log(mock_logs):
    log = _on_pin(mock_logs[1], 1)
    with pytest.raises(RunConfigError, match="status"):
        grader.check_grader_log(log.model_copy(update={"status": "error"}), 1)
    with pytest.raises(RunConfigError, match="header_only"):
        grader.check_grader_log(log.model_copy(update={"samples": None}), 1)


@pytest.mark.parametrize("step", STEPS)
def test_log_check_flags_a_resolved_string_off_the_tag_time_one(step, mock_logs):
    log = _on_pin(mock_logs[step], step, resolved=grader.TAG_TIME_RESOLVED[step] + "-2027-01-01")
    with pytest.raises(RunConfigError, match="vendor-change event"):
        grader.check_grader_log(log, step)


def test_log_check_flags_a_call_with_no_resolved_string(mock_logs):
    log = _on_pin(mock_logs[1], 1, resolved="")
    with pytest.raises(RunConfigError, match="no resolved model string"):
        grader.check_grader_log(log, 1)


def test_log_check_flags_calls_that_resolve_to_different_strings(mock_logs):
    log = _on_pin(mock_logs[1], 1)
    sample = log.samples[0]
    extra = _model_events(log)[0].model_copy(deep=True)
    extra.output.model = "gpt-6-luna-2027-01-01"
    sample.events.append(extra)
    with pytest.raises(RunConfigError, match="more than one model string"):
        grader.check_grader_log(log, 1)


def test_resolved_check_on_made_up_summaries():
    good = {"model": PINS[2], "resolved": "gpt-6.1-terra", "sample_id": "dev-mock-001", "epoch": 1}
    grader.check_resolved([good, {**good, "epoch": 2}], 2)
    with pytest.raises(RunConfigError, match="more than one model string"):
        grader.check_resolved([good, {**good, "resolved": "gpt-6.1-terra-b"}], 2)
    with pytest.raises(RunConfigError, match="not 'gpt-6.1-sol'"):
        grader.check_resolved([good], 3)


def test_event_check_refuses_another_role():
    with pytest.raises(RunConfigError, match="not a grader call"):
        grader.check_grader_event(SimpleNamespace(role="calibrator", model=PINS[1]), 1)


def test_event_check_on_a_made_up_event():
    event = SimpleNamespace(role="grader", model=PINS[3], config={"reasoning_effort": "high", "batch": True},
                            cache=None, error=None,
                            output=SimpleNamespace(model="gpt-6.1-sol", usage=SimpleNamespace(input_tokens=3, output_tokens=1)))
    assert grader.check_grader_event(event, 3) == {
        "model": PINS[3], "resolved": "gpt-6.1-sol", "config": {"reasoning_effort": "high", "batch": True},
        "usage": (3, 1),
    }


def test_the_roster_run_config_check_still_refuses_a_grader_event_off_the_pin(mock_logs):
    # The grader branch reused here is the one check_run_config applies.
    log = _edit_config(mock_logs[1], 1, reasoning_effort="medium")
    fields = roster._generate_fields(_model_events(log)[0].config)
    with pytest.raises(RunConfigError, match="medium"):
        roster._check_generate("probe", fields, evaluated=True, allow_batch=False, max_tokens_pin=None, grader=True)


# --- The live check script: dry run ----------------------------------------------------

def _forbid_calls(monkeypatch):
    calls = []

    def boom(name):
        def fail(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} called in a dry run")
        return fail

    import inspect_ai
    from inspect_ai.model import Model

    monkeypatch.setattr("inspect_ai.model.get_model", boom("get_model"))
    monkeypatch.setattr(Model, "generate", boom("Model.generate"))
    monkeypatch.setattr(inspect_ai, "eval", boom("eval"))
    monkeypatch.setattr(live, "run_step", boom("run_step"))
    monkeypatch.setattr(grader, "grade", boom("grade"))
    monkeypatch.setattr(grader, "resolve_grader", boom("resolve_grader"))
    return calls


def test_dry_run_makes_no_call_and_prints_the_plan(monkeypatch, capsys):
    calls = _forbid_calls(monkeypatch)
    assert live.main([]) == 0
    assert calls == []
    out = capsys.readouterr().out
    for step in STEPS:
        assert f"step {step}: would call {PINS[step]}" in out
    assert "{'reasoning_effort': 'high'}" in out
    assert out.count("{'reasoning_effort': 'high', 'batch': True}") == 2
    assert "dry run: no call made" in out


def test_dry_run_with_a_log_dir_still_makes_no_call(monkeypatch, capsys, tmp_path):
    calls = _forbid_calls(monkeypatch)
    assert live.main(["--steps", "2", "--log-dir", str(tmp_path)]) == 0
    assert calls == [] and list(tmp_path.iterdir()) == []
    out = capsys.readouterr().out
    assert PINS[2] in out and PINS[1] not in out


def test_dry_run_in_a_subprocess_exits_0_and_creates_nothing(tmp_path):
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = str(REPO)
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "grader_live_check.py")], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert "dry run: no call made" in done.stdout
    assert list(tmp_path.iterdir()) == []


def test_plan_is_pure():
    assert live.plan(3) == {"step": 3, "model": PINS[3], "config": {"reasoning_effort": "high", "batch": True},
                            "cache": False, "batch_api": True, "tag_time_resolved": "gpt-6.1-sol"}


@pytest.mark.parametrize("value, steps", [("1,2,3", [1, 2, 3]), ("3", [3]), (" 2 , 2,1", [2, 1])])
def test_steps_parse(value, steps):
    assert live.parse_steps(value) == steps


@pytest.mark.parametrize("value", ["", "0", "4", "a", "1,,x"])
def test_bad_steps_exit_2(value):
    with pytest.raises(SystemExit) as exc:
        live.parse_args(["--steps", value])
    assert exc.value.code == 2


def test_live_needs_a_log_dir(monkeypatch):
    _forbid_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        live.main(["--live"])
    assert exc.value.code == 2


def test_live_refuses_a_log_dir_inside_the_repository(monkeypatch):
    _forbid_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        live.main(["--live", "--log-dir", str(REPO / "runs" / "probe")])
    assert exc.value.code == 2


def test_help_exits_0(capsys):
    with pytest.raises(SystemExit) as exc:
        live.parse_args(["-h"])
    assert exc.value.code == 0
    assert "--live" in capsys.readouterr().out


def test_script_never_names_a_key_variable():
    text = (REPO / "scripts" / "grader_live_check.py").read_text(encoding="utf-8")
    assert "API_KEY" not in text and "environ" not in text


# --- The live check script: the live path, offline -------------------------------------

@pytest.mark.parametrize("step", STEPS)
def test_live_task_offline_writes_a_log_the_check_accepts_on_the_pin(step, tmp_path):
    log = live.run_step(step, str(tmp_path), {"grader": MOCK})
    assert log.status == "success", log.error
    assert log.eval.model == grader.NO_MODEL
    assert Path(log.location).parent == tmp_path / f"step-{step}"
    assert log.eval.metadata["grader_live_check"] == step
    assert log.eval.metadata["roster"]["grader_generate_config"] == roster.grader_generate_config_args(step)
    with pytest.raises(RunConfigError, match="not the step"):
        live.check_log(log, step)  # the mock is not the pin
    summary = live.check_log(_on_pin(log, step), step)
    assert summary["resolved"] == grader.TAG_TIME_RESOLVED[step]
    assert summary["endpoint"] == "not recorded (offline)"
    assert summary["config"] == roster.grader_generate_config_args(step)
    # The whole-run check refuses a live-check log: it holds no evaluated-model call.
    with pytest.raises(RunConfigError):
        roster.check_run_config(log, expected_epochs=1)


def test_live_check_flags_a_changed_resolved_string(tmp_path):
    log = _on_pin(live.run_step(1, str(tmp_path), {"grader": MOCK}), 1, resolved="gpt-6-luna-2027-01-01")
    with pytest.raises(RunConfigError, match="vendor-change event"):
        live.check_log(log, 1)


def test_live_check_refuses_a_second_model_call(tmp_path):
    log = _on_pin(live.run_step(1, str(tmp_path), {"grader": MOCK}), 1)
    sample = log.samples[0]
    extra = [e for e in sample.events if getattr(e, "event", None) == "model"][0].model_copy(deep=True)
    extra.role = None
    sample.events.append(extra)
    with pytest.raises(RunConfigError, match="exactly one model call"):
        live.check_log(log, 1)


def test_live_check_refuses_an_empty_answer(tmp_path):
    log = _on_pin(live.run_step(1, str(tmp_path), {"grader": MOCK}), 1)
    log.samples[0].output = ModelOutput.from_content(PINS[1], "")
    with pytest.raises(RunConfigError, match="no answer"):
        live.check_log(log, 1)


def test_live_main_reports_pass_and_fail_per_step(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(roster, "extra_headers_in_force", lambda: [])
    original = live.run_step

    def fake_run_step(step, log_dir, model_roles=None):
        log = original(step, str(tmp_path), {"grader": MOCK})
        return _on_pin(log, step, resolved="gpt-6.1-sol-new" if step == 3 else None)

    monkeypatch.setattr(live, "run_step", fake_run_step)
    log_dir = tmp_path / "unused"  # outside the repository, never written
    code = live.main(["--live", "--log-dir", str(log_dir)])
    out = capsys.readouterr().out
    assert code == 1
    assert "PASS step 1 openai/gpt-6-luna: resolved gpt-6-luna; reasoning_effort high; batch unset" in out
    assert "PASS step 2 openai/gpt-6.1-terra: resolved gpt-6.1-terra; reasoning_effort high; batch True" in out
    assert "FAIL step 3" in out and "vendor-change event" in out
    assert "extra headers in force: none" in out
    assert "GRADER LIVE CHECK RESULT: FAIL" in out
    assert not log_dir.exists()


def test_live_main_passes_when_every_step_passes(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(roster, "extra_headers_in_force", lambda: [])
    original = live.run_step
    monkeypatch.setattr(live, "run_step",
                        lambda step, log_dir, model_roles=None: _on_pin(original(step, str(tmp_path), {"grader": MOCK}), step))
    code = live.main(["--live", "--steps", "1,3", "--log-dir", str(tmp_path / "unused")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.count("PASS step") == 2 and "GRADER LIVE CHECK RESULT: PASS" in out


def test_live_main_keeps_going_after_an_exception(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(roster, "extra_headers_in_force", lambda: [])

    def explode(step, log_dir, model_roles=None):
        raise RosterError(f"serving guard: probe {step}")

    monkeypatch.setattr(live, "run_step", explode)
    code = live.main(["--live", "--log-dir", str(tmp_path / "unused")])
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("FAIL step") == 3 and "serving guard: probe 3" in out


def test_live_run_step_binds_the_pin_and_no_evaluated_model(monkeypatch, tmp_path):
    seen = {}

    def fake_eval(task, **kw):
        seen.update(kw)
        return [SimpleNamespace(status="success")]

    import inspect_ai

    monkeypatch.setattr(inspect_ai, "eval", fake_eval)
    live.run_step(2, str(tmp_path))
    assert seen["model"] == "none/none"
    assert seen["model_roles"] == {"grader": PINS[2]}
    assert seen["epochs"] == 1 and seen["log_dir"].endswith("step-2")
