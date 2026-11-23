"""scripts/calibrator_call.py: the calibrator call of the full-run grading job and its log
checks, including the pre-registration section 5 checks (D-R23, calibrator).

Keyless: every model is Inspect's mockllm, a recorder stand-in, or a real Inspect model built
with a dummy key whose client is read and never called, as tests/test_serving_guard.py does.
HOME, USERPROFILE, APPDATA and ANTHROPIC_CONFIG_DIR point at tmp_path. No request is sent.
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import importlib.metadata
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ContentText, GenerateConfig, ModelOutput, StopDetails

import gradingfactory as gf
from icfr_agent_evals import grader, roster
from icfr_agent_evals.roster import RosterError, RunConfigError

gc = gf.load_script("grading_common")
cc = gf.load_script("calibrator_call")
PIN = roster.CALIBRATOR
MOCK = "mockllm/model"
ANTHROPIC = "https://api.anthropic.com"
RESOLVED = "claude-opus-5-5-20261101"
URL_ENV = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL")
HEADER_ENV = ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS")
COMPLETED = datetime.datetime(2026, 11, 23, 12, 0, tzinfo=datetime.timezone.utc)

# The request a live calibrator call records, frozen from the pinned Inspect. In Inspect
# 0.3.278 the Anthropic provider sets request["tools"] with no condition
# (model/_providers/anthropic.py:797), so every call records "tools": [], and the request is
# recorded at :930. A change of the Inspect version fails test_the_recorded_anthropic_request_passes,
# which flags this fixture for a new look.
ANTHROPIC_RECORDED_REQUEST = {   # Inspect 0.3.278, Anthropic provider, calibrator pin at effort high
    "messages": [{"role": "user", "content": "x"}],
    "cache_control": {"type": "ephemeral"},
    "tools": [],
    "extra_headers": {"anthropic-version": "2023-06-01", "anthropic-beta": "effort-2025-11-24"},
    "model": "claude-opus-5-5",
    "max_tokens": 48000,
    "output_config": {"effort": "high"},
    "thinking": {"type": "adaptive", "display": "summarized"},
}
RECORDED_MODE = 'output_config.effort="high", thinking={"display":"summarized","type":"adaptive"}'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "APPDATA", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    for name in (*URL_ENV, *HEADER_ENV, "INSPECT_EVAL_EXTRA_HEADERS", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)


# --- Pins and config --------------------------------------------------------------------------

def test_config_is_exactly_the_calibrator_pin():
    assert cc.CALIBRATOR_ROLE == roster.CALIBRATOR_ROLE == "calibrator"
    assert cc.CALIBRATOR_EVENT_SOURCE == "calibrator_serving"
    assert cc.FORBIDDEN_REQUEST_KEYS == ("temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks",
                                         "stop_sequences")
    assert cc.calibrator_pin() == PIN == roster.role_pins(1)["calibrator"]
    config = cc.calibrator_config()
    assert config.model_dump(exclude_none=True) == {"reasoning_effort": "high"} == roster.calibrator_generate_config_args()
    assert config.batch is None and config.cache is None and config.max_tokens is None
    assert cc.check_calibrator_config(config) == {"reasoning_effort": "high"}
    assert issubclass(cc.CalibratorError, RosterError)


@pytest.mark.parametrize("batch", [True, 1, 50])
def test_a_batched_calibrator_call_is_refused(batch, monkeypatch):
    # 1. the config check
    with pytest.raises(RunConfigError, match="batch"):
        cc.check_calibrator_config({"reasoning_effort": "high", "batch": batch})
    with pytest.raises(RunConfigError, match="batch"):
        cc.check_calibrator_config({"reasoning_effort": "high", "batch": batch}, recorded=True)
    # 2. a model whose own config sets batch is refused before the call
    model = _Recorder(MOCK, own=GenerateConfig(batch=batch))
    _patch_get_model(monkeypatch, model)
    with pytest.raises(RunConfigError, match="batch"):
        asyncio.run(cc.calibrate("x"))
    assert model.calls == []
    # 3. a recorded event with batch set
    with pytest.raises(RunConfigError, match="batch"):
        cc.check_calibrator_event(_event(config={"reasoning_effort": "high", "batch": batch}), offline=False)


def test_batch_false_or_unset_is_accepted():
    assert cc.check_calibrator_config({"reasoning_effort": "high", "batch": False})["batch"] is False
    assert cc.check_calibrator_config({"reasoning_effort": "high", "batch": None}) == {"reasoning_effort": "high"}


def test_max_tokens_is_refused_before_the_call_and_only_the_provider_default_is_accepted_after():
    fields = {"reasoning_effort": "high", "max_tokens": roster.DEFAULT_MAX_TOKENS}
    with pytest.raises(RunConfigError, match="max_tokens"):
        cc.check_calibrator_config(fields)
    assert cc.check_calibrator_config(fields, recorded=True)["max_tokens"] == 48_000
    with pytest.raises(RunConfigError, match="max_tokens"):
        cc.check_calibrator_config({"reasoning_effort": "high", "max_tokens": 1000}, recorded=True)
    with pytest.raises(RunConfigError, match="max_tokens"):
        cc.check_calibrator_config({"reasoning_effort": "high", "max_tokens": 1000})


@pytest.mark.parametrize("extra", [
    {"temperature": 0.0}, {"top_p": 0.9}, {"top_k": 5}, {"seed": 7}, {"reasoning_tokens": 1000},
    {"fallback_models": ["anthropic/claude-sonnet-5-5"]}, {"fail_on_refusal": True}, {"cache": True},
    {"extra_headers": {"X-Probe": "1"}}, {"system_message": "be kind"},
], ids=lambda e: next(iter(e)))
def test_sampling_fields_fallback_models_and_fail_on_refusal_are_refused(extra):
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_config({"reasoning_effort": "high", **extra})
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_config({"reasoning_effort": "high", **extra}, recorded=True)


@pytest.mark.parametrize("effort", [None, "low", "medium", "xhigh", "none"])
def test_an_effort_off_the_pin_is_refused(effort):
    with pytest.raises(RunConfigError, match="reasoning_effort"):
        cc.check_calibrator_config({"reasoning_effort": effort})


def test_operational_fields_are_accepted():
    got = cc.check_calibrator_config(GenerateConfig(reasoning_effort="high", max_retries=3, timeout=600))
    assert got["max_retries"] == 3


def test_model_and_pairing_checks():
    cc.check_calibrator_model(PIN)
    cc.check_calibrator_model(MOCK)
    for name in ("anthropic/claude-sonnet-5-5", "openai/gpt-6.1-terra", "anthropic/bedrock/claude-opus-5-5"):
        with pytest.raises(cc.CalibratorError):
            cc.check_calibrator_model(name)
    for evaluated in (None, grader.NO_MODEL, MOCK):
        cc.check_offline_pairing(MOCK, evaluated)
    cc.check_offline_pairing(PIN, grader.NO_MODEL)
    with pytest.raises(cc.CalibratorError, match="mixing real and mockllm"):
        cc.check_offline_pairing(MOCK, roster.WORKER)
    with pytest.raises(cc.CalibratorError, match="mixing real and mockllm"):
        cc.check_offline_pairing(PIN, MOCK)


# --- calibrate(): the call --------------------------------------------------------------------

class _Recorder:
    """Stands in for an Inspect Model; records what generate() was given."""

    def __init__(self, name, role="calibrator", own=None, output=None):
        self.name, self.role, self.calls = name, role, []
        self.config = own or GenerateConfig()
        self.output = output

    def __str__(self):
        return self.name

    def _resolve_config(self, config):
        return self.config.merge(config)

    async def generate(self, input, tools=(), tool_choice=None, config=GenerateConfig(), cache=None, **kw):
        self.calls.append({"input": input, "tools": tools, "tool_choice": tool_choice, "config": config,
                           "cache": cache, **kw})
        return self.output or ModelOutput.from_content(self.name, "OK")


def _patch(monkeypatch, name, value, where="inspect_ai.model"):
    """Patch an Inspect function where calibrator_call would read it: on the Inspect module and,
    when the script imported it by name, on the script too."""
    monkeypatch.setattr(f"{where}.{name}", value)
    if hasattr(cc, name):
        monkeypatch.setattr(cc, name, value)


def _patch_get_model(monkeypatch, model, seen=None, roles=None):
    def fake_get_model(name=None, *, role=None, **kw):
        if seen is not None:
            seen.append((name, role, kw))
        return model

    _patch(monkeypatch, "get_model", fake_get_model)
    _patch(monkeypatch, "model_roles", lambda: dict(roles or {}))


def test_calibrate_passes_cache_false_and_no_tools(monkeypatch):
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    model = _Recorder(MOCK)
    seen = []
    _patch_get_model(monkeypatch, model, seen)
    messages = [ChatMessageSystem(content="caller system"), ChatMessageUser(content="caller user")]
    out = asyncio.run(cc.calibrate(messages))
    assert out.completion == "OK"
    assert seen and seen[0][0] == PIN and seen[0][1] == "calibrator"
    (call,) = model.calls
    assert call["cache"] is False and not call["tools"] and call["tool_choice"] is None
    assert call["input"] == messages
    assert call["config"].model_dump(exclude_none=True) == {"reasoning_effort": "high"}
    assert "fallback_models" not in call["config"].model_dump(exclude_none=True)
    asyncio.run(cc.calibrate("Reply with OK."))
    assert model.calls[1]["input"] == "Reply with OK."


@pytest.mark.parametrize("messages", ["", "   ", []])
def test_calibrate_refuses_empty_messages_and_an_off_pin_model_before_any_call(messages, monkeypatch):
    model = _Recorder(MOCK)
    _patch_get_model(monkeypatch, model)
    with pytest.raises(cc.CalibratorError):
        asyncio.run(cc.calibrate(messages))
    assert model.calls == []
    off = _Recorder("anthropic/claude-sonnet-5-5")
    _patch_get_model(monkeypatch, off)
    with pytest.raises(cc.CalibratorError):
        asyncio.run(cc.calibrate("x"))
    assert off.calls == []
    no_role = _Recorder(MOCK, role="grader")
    _patch_get_model(monkeypatch, no_role)
    with pytest.raises(cc.CalibratorError):
        asyncio.run(cc.calibrate("x"))
    assert no_role.calls == []


@pytest.mark.parametrize("own", [GenerateConfig(temperature=0.5), GenerateConfig(seed=3),
                                 GenerateConfig(max_tokens=64), GenerateConfig(extra_headers={"X-Probe": "1"}),
                                 GenerateConfig(cache=True)])
def test_calibrate_refuses_an_off_pin_model_config_before_any_call(own, monkeypatch):
    model = _Recorder(MOCK, own=own)
    _patch_get_model(monkeypatch, model)
    with pytest.raises(RunConfigError):
        asyncio.run(cc.calibrate("x"))
    assert model.calls == []


def test_calibrate_fails_closed_without_resolve_config(monkeypatch):
    class Bare:
        """A model stand-in with no _resolve_config: the effective config cannot be read."""

        role = "calibrator"

        def __init__(self):
            self.calls = []

        def __str__(self):
            return MOCK

        async def generate(self, *a, **k):
            self.calls.append((a, k))
            return ModelOutput.from_content(MOCK, "OK")

    model = Bare()
    _patch_get_model(monkeypatch, model)
    with pytest.raises(cc.CalibratorError):
        asyncio.run(cc.calibrate("x"))
    assert model.calls == []


def test_a_refusal_is_recorded_as_a_refusal_not_an_error(tmp_path):
    out = ModelOutput.from_content(MOCK, "I will not grade this.", stop_reason="content_filter")
    summary = cc.check_calibrator_event(_event(output=out, model=MOCK, call=None), offline=True)
    assert summary["refused"] is True
    reply = gc.reply_record(SimpleNamespace(output=out), attempt=1, log_file="x.eval")
    assert reply["status"] == gc.REPLY_REFUSED and reply["scores"] == {}
    # through a real offline eval
    log = _mock_calibrator_eval(tmp_path, lambda text: out)
    assert log.status == "success", log.error
    calls, summaries = cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=True)
    assert set(calls["completed"]) == {"u0001", "u0002"}
    assert all(s["refused"] for s in summaries) and {s["sample_id"] for s in summaries} == {"u0001", "u0002"}


# --- An offline calibrator eval -----------------------------------------------------------------

def _calibrator_task(ids=("u0001", "u0002"), role="calibrator"):
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="calibrate_once")
    def calibrate_once():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            gc.check_eval_roles(role)
            state.output = await cc.calibrate(state.messages)
            return state

        return solve

    return Task(dataset=MemoryDataset([Sample(id=i, input=f"probe {i}", metadata={"split": "dev"}) for i in ids]),
                solver=[calibrate_once()], epochs=1, config=GenerateConfig())


def _mock_calibrator_eval(tmp_path, answer=None, *, roles=None, model=grader.NO_MODEL):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    def out(input, tools, tool_choice, config):
        text = "\n".join(str(m.text) for m in input)
        return answer(text) if answer else ModelOutput.from_content(MOCK, gf.reply(2, 1))

    mock = get_model(MOCK, custom_outputs=out, memoize=False)
    return inspect_eval(_calibrator_task(), model=model, model_roles=roles or {"calibrator": mock},
                        log_dir=str(tmp_path / "cal-logs"), display="none")[0]


def test_an_offline_calibrator_eval_passes_its_log_check(tmp_path):
    log = _mock_calibrator_eval(tmp_path)
    assert log.status == "success", log.error
    events = [e for _, e in cc.calibrator_events(log)]
    assert len(events) == 2 and all(e.role == "calibrator" and e.cache is None for e in events)
    assert all(not e.tools for e in events)
    for e in events:
        fields = cc.check_calibrator_config(e.config, recorded=True)
        assert fields.get("batch") in (None, False) and fields["reasoning_effort"] == "high"
    calls, summaries = cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=True)
    assert calls["skipped"] == 0 and not any(s["refused"] for s in summaries)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_log(log, asked=["u0001", "u0002", "u0003"], offline=True)


def test_calibrator_eval_binds_one_role_only(tmp_path):
    from inspect_ai.model import get_model

    mock = get_model(MOCK, memoize=False)
    two = _mock_calibrator_eval(tmp_path / "two", roles={"calibrator": mock, "grader": mock})
    assert two.status == "error" and "one role per grading eval" in str(two.error)
    assert cc.calibrator_events(two) == []
    evaluated = _mock_calibrator_eval(tmp_path / "evaluated", model=MOCK)
    assert evaluated.status == "error" and "none/none" in str(evaluated.error)
    assert cc.calibrator_events(evaluated) == []


# --- Section 5 serving checks on real Inspect models (dummy key, no request) ---------------------

def _real_resolver(monkeypatch, *, bound=None, **model_kw):
    """get_model, as calibrator_call calls it, builds the real pin with a dummy key; no request
    is ever sent. Returns a function that runs resolve_calibrator and the info events it wrote."""
    from inspect_ai.log._transcript import Transcript, init_transcript
    from inspect_ai.model import get_model as real_get_model

    def fake_get_model(name=None, *, role=None, **kw):
        return real_get_model(name, role=role, api_key="test", memoize=False, **model_kw)

    _patch(monkeypatch, "get_model", fake_get_model)
    _patch(monkeypatch, "model_roles", lambda: dict(bound or {}))

    def run():
        t = Transcript()
        init_transcript(t)
        model = cc.resolve_calibrator()
        return model, [e for e in t.events if getattr(e, "event", None) == "info"]

    return run


def test_calibrator_exact_endpoint(monkeypatch):
    model, infos = _real_resolver(monkeypatch)()
    assert str(model) == PIN and model.role == "calibrator"
    assert roster.model_serving(model) == {"model": PIN, "base_url": ANTHROPIC}


def test_calibrator_endpoint_is_recorded_in_the_transcript(monkeypatch):
    _, infos = _real_resolver(monkeypatch)()
    recorded = [e for e in infos if e.source == cc.CALIBRATOR_EVENT_SOURCE]
    assert len(recorded) == 1
    endpoint = recorded[0].data["serving_endpoint"]
    assert endpoint["model"] == PIN and endpoint["base_url"].rstrip("/") == ANTHROPIC
    sample = SimpleNamespace(id="u0001", events=recorded)
    assert gc.check_serving_recorded(sample, source=cc.CALIBRATOR_EVENT_SOURCE, provider="anthropic")["model"] == PIN


@pytest.mark.parametrize("var", ["ANTHROPIC_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL"])
def test_calibrator_refuses_a_base_url_variable(var, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    with pytest.raises(RosterError, match="serving guard"):
        _real_resolver(monkeypatch)()


def test_calibrator_refuses_an_explicit_base_url_override(monkeypatch):
    with pytest.raises(RosterError, match="explicit base URL override"):
        _real_resolver(monkeypatch, base_url="https://proxy.example")()


def test_calibrator_refuses_model_args(monkeypatch):
    with pytest.raises(RosterError, match="model_args"):
        _real_resolver(monkeypatch, default_headers={"x": "y"})()


def test_calibrator_refuses_the_auth_token_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    with pytest.raises(RosterError, match="ANTHROPIC_AUTH_TOKEN"):
        _real_resolver(monkeypatch)()


@pytest.mark.parametrize("var", HEADER_ENV)
def test_calibrator_refuses_custom_headers(var, monkeypatch):
    monkeypatch.setenv(var, "X-Api-Key: test")
    with pytest.raises(RosterError, match=var):
        _real_resolver(monkeypatch)()


def test_calibrator_refuses_extra_headers(monkeypatch):
    monkeypatch.setenv("INSPECT_EVAL_EXTRA_HEADERS", "X-Probe: 1")
    with pytest.raises(RosterError, match="INSPECT_EVAL_EXTRA_HEADERS"):
        _real_resolver(monkeypatch)()


def test_calibrator_role_bound_to_more_than_one_model_is_refused(monkeypatch):
    model = _Recorder(MOCK)
    _patch_get_model(monkeypatch, model, roles={"calibrator": [object(), object()]})
    with pytest.raises(cc.CalibratorError, match="more than one model"):
        cc.resolve_calibrator()
    with pytest.raises(cc.CalibratorError, match="more than one model"):
        asyncio.run(cc.calibrate("x"))
    assert model.calls == []


def test_a_real_calibrator_refuses_before_the_call_when_the_serving_guard_fails(monkeypatch):
    model = _Recorder(PIN)
    _patch_get_model(monkeypatch, model)

    def refuse(m):
        raise RosterError("serving guard: probe")

    monkeypatch.setattr(roster, "model_serving", refuse)
    with pytest.raises(RosterError, match="serving guard: probe"):
        asyncio.run(cc.calibrate("x"))
    assert model.calls == []


# --- The calibrator log check -------------------------------------------------------------------

def _event(**kw):
    """A calibrator ModelEvent as a completed live call records it: completed time set, not
    pending, and the frozen request of the pinned Anthropic provider (with "tools": [])."""
    request = copy.deepcopy(ANTHROPIC_RECORDED_REQUEST)
    base = dict(event="model", role="calibrator", model=PIN, cache=None, error=None, tools=[],
                completed=COMPLETED, pending=None,
                config={"reasoning_effort": "high", "max_tokens": 48_000},
                call=SimpleNamespace(request=request, response={}),
                output=SimpleNamespace(model=RESOLVED, usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                                       stop_reason="stop", completion=gf.reply(2, 2)))
    base.update(kw)
    return SimpleNamespace(**base)


def _info(base_url=ANTHROPIC):
    return SimpleNamespace(event="info", source="calibrator_serving",
                           data={"serving_endpoint": {"model": PIN, "base_url": base_url}})


def _live_log(*event_lists, status="success"):
    return SimpleNamespace(status=status, eval=None, location=None,
                           samples=[SimpleNamespace(id=f"u{i + 1:04d}", epoch=1, error=None, output=None,
                                                    events=list(evs)) for i, evs in enumerate(event_lists)])


def test_a_live_calibrator_log_on_the_pin_passes():
    log = _live_log([_info(), _event()], [_info(ANTHROPIC + "/"), _event()])
    calls, summaries = cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=False,
                                               allowed_resolved=(RESOLVED,))
    assert len(summaries) == 2 and {s["sample_id"] for s in summaries} == {"u0001", "u0002"}
    s = summaries[0]
    assert s["model"] == PIN and s["resolved"] == RESOLVED and s["refused"] is False
    assert s["request_mode"] == RECORDED_MODE and list(s["usage"]) == [100, 20]
    cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=False, allowed_resolved=None)


# --- The recorded request of the pinned provider (a forbidden key is refused only with a value) ---

def test_the_recorded_anthropic_request_passes():
    assert importlib.metadata.version("inspect_ai") == "0.3.278"
    assert set(ANTHROPIC_RECORDED_REQUEST) == {"cache_control", "extra_headers", "max_tokens", "messages", "model",
                                               "output_config", "thinking", "tools"}
    mode = cc._check_request(ANTHROPIC_RECORDED_REQUEST, "w")
    assert 'output_config.effort="high"' in mode
    assert mode == RECORDED_MODE
    summary = cc.check_calibrator_event(
        _event(call=SimpleNamespace(request=ANTHROPIC_RECORDED_REQUEST, response={})), offline=False)
    assert summary["request_mode"] == RECORDED_MODE and summary["refused"] is False
    assert _event().call.request == ANTHROPIC_RECORDED_REQUEST


def _recorded(**changes):
    request = copy.deepcopy(ANTHROPIC_RECORDED_REQUEST)
    request.update(changes)
    return request


@pytest.mark.parametrize("request_, key", [
    (_recorded(tools=[{"name": "t"}]), "tools"),
    (_recorded(temperature=0.0), "temperature"),
    (_recorded(top_k=0), "top_k"),
    (_recorded(tool_choice={"type": "auto"}), "tool_choice"),
    (_recorded(stop_sequences=["END"]), "stop_sequences"),
    (_recorded(fallbacks=False), "fallbacks"),
    (_recorded(extra_body={"tools": [{"name": "t"}]}), "tools in extra_body"),
    (_recorded(extra_body={"top_p": 0.5}), "top_p in extra_body"),
], ids=["tools", "temperature-0", "top_k-0", "tool_choice", "stop_sequences", "fallbacks-false",
        "extra-body-tools", "extra-body-top_p"])
def test_a_forbidden_key_with_a_value_is_refused(request_, key):
    with pytest.raises(RunConfigError, match=f"^run config: w sent {key}"):
        cc._check_request(request_, "w")
    log = _live_log([_info(), _event(call=SimpleNamespace(request=request_, response={}))])
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_log(log, asked=["u0001"], offline=False, allowed_resolved=(RESOLVED,))


@pytest.mark.parametrize("request_", [
    _recorded(extra_body={"tools": []}),
    _recorded(stop_sequences=None),
    _recorded(tool_choice={}, fallbacks=(), top_p=""),
    _recorded(extra_body={"tools": None, "stop_sequences": [], "tool_choice": {}}),
    _recorded(extra_body=None),
], ids=["extra-body-empty-tools", "stop_sequences-none", "empty-values", "extra-body-empty-values",
        "extra-body-none"])
def test_a_forbidden_key_sent_empty_is_accepted(request_):
    assert 'output_config.effort="high"' in cc._check_request(request_, "w")
    log = _live_log([_info(), _event(call=SimpleNamespace(request=request_, response={}))])
    cc.check_calibrator_log(log, asked=["u0001"], offline=False, allowed_resolved=(RESOLVED,))


def test_sent_treats_only_none_and_empty_containers_as_not_sent():
    for value in (None, [], (), {}, ""):
        assert cc._sent(value) is False
    for value in (0, 0.0, False, [None], {"a": None}, " ", "x", 1):
        assert cc._sent(value) is True


# --- One reasoning mode across the calibrator calls ---------------------------------------------

def test_two_reasoning_modes_are_refused():
    with pytest.raises(RunConfigError, match="^run config: the calibrator calls sent 2 reasoning modes, not one"):
        cc.check_calibrator_modes([{"request_mode": 'output_config.effort="high"'},
                                   {"request_mode": 'output_config.effort="low"'}])
    cc.check_calibrator_modes([{"request_mode": 'output_config.effort="high"'},
                               {"request_mode": 'output_config.effort="high"'}, {"request_mode": None}])
    cc.check_calibrator_modes([{"request_mode": None}, {"request_mode": None}])
    cc.check_calibrator_modes([])


def test_a_live_log_whose_calls_send_two_thinking_fields_is_refused():
    other = _recorded(thinking={"type": "enabled", "budget_tokens": 1024})
    log = _live_log(_with(), _with(call=SimpleNamespace(request=other, response={})))
    with pytest.raises(RunConfigError, match="^run config: the calibrator calls sent 2 reasoning modes"):
        cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=False, allowed_resolved=(RESOLVED,))
    # attempt admission leaves the mode check to the caller, which runs it over every attempt log
    _, summaries = cc.admit_calibrator_attempt_log(log, asked=["u0001", "u0002"], offline=False)
    with pytest.raises(RunConfigError, match="reasoning modes"):
        cc.check_calibrator_modes(summaries)


def _with(**kw):
    return [_info(), _event(**kw)]


@pytest.mark.parametrize("events, allowed", [
    (_with(role="grader"), (RESOLVED,)),                                                  # role
    (_with(model="anthropic/claude-sonnet-5-5"), (RESOLVED,)),                           # pin
    (_with(config={"reasoning_effort": "low"}), (RESOLVED,)),                           # config: effort
    (_with(config={"reasoning_effort": "high", "temperature": 0.0}), (RESOLVED,)),      # config: sampling
    (_with(config={"reasoning_effort": "high", "batch": True}), (RESOLVED,)),           # config: batch
    (_with(cache="read"), (RESOLVED,)),                                                   # cache
    (_with(error="boom"), (RESOLVED,)),                                                   # error
    (_with(tools=[{"name": "t"}]), (RESOLVED,)),                                          # tools
    (_with(call=SimpleNamespace(request={"output_config": {"effort": "low"}}, response={})), (RESOLVED,)),
    (_with(call=SimpleNamespace(request={"output_config": {"effort": "high"}, "temperature": 0.5}, response={})),
     (RESOLVED,)),
    (_with(call=SimpleNamespace(request={"output_config": {"effort": "high"}, "extra_body": {"top_k": 3}},
                                response={})), (RESOLVED,)),
    (_with(call=SimpleNamespace(request={"output_config": {"effort": "high"}, "fallbacks": ["x"]}, response={})),
     (RESOLVED,)),
    (_with(call=None), (RESOLVED,)),                                                      # no request
    ([_event()], (RESOLVED,)),                                                            # serving not recorded
    ([_info("http://localhost:4000"), _event()], (RESOLVED,)),                           # serving off first party
    (_with(output=SimpleNamespace(model="", usage=None, stop_reason="stop", completion="x")), None),  # no resolved
    (_with(), ("another-resolved-string",)),                                              # outside the Opus log's
], ids=["role", "pin", "effort", "sampling", "batch", "cache", "error", "tools", "request-effort",
        "forbidden-top-level", "forbidden-extra-body", "fallbacks", "missing-request", "serving-missing",
        "serving-off", "no-resolved", "resolved-outside"])
def test_calibrator_log_check_refuses_each_rule(events, allowed):
    log = _live_log(events)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_log(log, asked=["u0001"], offline=False, allowed_resolved=allowed)


def test_two_resolved_strings_are_refused():
    other = SimpleNamespace(model="claude-opus-5-5-20270101", usage=None, stop_reason="stop", completion="x")
    log = _live_log(_with(), _with(output=other))
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=False, allowed_resolved=None)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_resolved([{"resolved": RESOLVED}, {"resolved": "claude-opus-5-5-20270101"}], None)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_resolved([{"resolved": ""}], None)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_resolved([{"resolved": RESOLVED}], ("x",))
    cc.check_calibrator_resolved([{"resolved": RESOLVED}, {"resolved": RESOLVED}], (RESOLVED,))


def test_a_refusal_in_a_live_log_is_accepted_and_marked():
    refusal = SimpleNamespace(model=RESOLVED, usage=None, stop_reason="content_filter", completion="no")
    log = _live_log(_with(output=refusal))
    _, (summary,) = cc.check_calibrator_log(log, asked=["u0001"], offline=False, allowed_resolved=(RESOLVED,))
    assert summary["refused"] is True


@pytest.mark.parametrize("output, refused", [
    (ModelOutput.from_content(RESOLVED, [ContentText(text="no", refusal=True)]), True),
    (ModelOutput.from_content(RESOLVED, "no", stop_details=StopDetails(type="refusal")), True),
    (ModelOutput.from_content(RESOLVED, "no", stop_details=StopDetails(type="max_tokens")), False),
    (ModelOutput.from_content(RESOLVED, gf.reply(2, 2)), False),
], ids=["refusal-content", "refusal-stop-details", "max-tokens-stop-details", "plain"])
def test_refusal_content_and_stop_details_are_marked_as_refusals(output, refused):
    assert output.stop_reason != "content_filter"
    log = _live_log(_with(output=output))
    _, (summary,) = cc.check_calibrator_log(log, asked=["u0001"], offline=False, allowed_resolved=(RESOLVED,))
    assert summary["refused"] is refused
    assert summary["resolved"] == RESOLVED and summary["usage"] is None


def test_a_pending_or_uncompleted_event_is_not_a_completed_call():
    for event in (_event(pending=True, completed=None), _event(completed=None)):
        log = _live_log([_info(), event], status="error")
        calls, summaries = cc.admit_calibrator_attempt_log(log, asked=["u0001"], offline=False)
        assert calls["completed"] == {} and calls["missing"] == ["u0001"] and summaries == []
        with pytest.raises(RunConfigError, match="^run config: "):
            cc.check_calibrator_log(_live_log([_info(), event]), asked=["u0001"], offline=False,
                                    allowed_resolved=(RESOLVED,))


def test_an_offline_event_must_be_mockllm():
    cc.check_calibrator_event(_event(model=MOCK, call=None), offline=True)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_event(_event(model=PIN), offline=True)
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_event(_event(model=MOCK), offline=False)


def test_calibrator_log_check_refuses_a_non_success_log_and_attempt_admission_accepts_it():
    log = _live_log([_info(), _event()], [_info(), _event(error="batch died")], status="error")
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.check_calibrator_log(log, asked=["u0001", "u0002"], offline=False, allowed_resolved=(RESOLVED,))
    calls, summaries = cc.admit_calibrator_attempt_log(log, asked=["u0001", "u0002", "u0003"], offline=False)
    assert set(calls["completed"]) == {"u0001"} and sorted(calls["missing"]) == ["u0002", "u0003"]
    assert len(summaries) == 1
    bad = _live_log([_event()], status="error")                         # serving not recorded
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.admit_calibrator_attempt_log(bad, asked=["u0001"], offline=False)
    off = _live_log(_with(model="anthropic/claude-sonnet-5-5"), status="error")
    with pytest.raises(RunConfigError, match="^run config: "):
        cc.admit_calibrator_attempt_log(off, asked=["u0001"], offline=False)


def test_messages_name_no_model_literally():
    text = (cc.__doc__ or "") + json.dumps([cc.CALIBRATOR_EVENT_SOURCE])
    for model in roster.EVALUATED_MODELS:
        assert model not in text
