"""The calibrator call of the full-run grading job: the one place a calibrator model is
made and called, and the checks of what its calls recorded.

It mirrors icfr_agent_evals/grader.py for the calibrator role. It lives in scripts/
because the package must stay byte-identical to the run commit until the run is settled
(pre-registration section 8, package-hash rule), so no calibrator call can be added there.
Decision D-R23 records the job.

The rules it holds (D-R23, calibrator):
  - The calibrator is Opus 5.5 on the calibrator role (roster.CALIBRATOR). Its config is
    exactly roster.calibrator_generate_config_args(): reasoning_effort high and nothing
    else. No sampling field, no seed, no reasoning token budget, no max_tokens, no
    fallback models, no fail-on-refusal setting and no response cache setting.
  - Not batched. A batch setting of anything but unset or False is refused before the
    call and in every recorded event, since check_run_config lets batch through.
  - No fallback model and no tools. This module calls generate once, with cache=False.
    Inspect's own retries of transient errors apply, and each errored attempt is skipped
    by the log check.
  - The prompt is grader prompt gp-1 unchanged. This module adds nothing to the
    caller's messages.
  - Response caching is off, as for the grader.
  - A refusal (a refusal stop reason, refusal content or refusal stop details, read by
    grading_common.is_refusal) is recorded as a refusal, never as an error and never as
    a parsed reply.
  - A real calibrator passes the roster serving guard (roster.model_serving), and the
    endpoint it checked is recorded in the sample transcript as an info event with
    source CALIBRATOR_EVENT_SOURCE.

The log checks read what each call recorded: the role, the pin, the config, no cache,
no error, no tools, and on a live call the request as sent (effort high in
output_config, none of FORBIDDEN_REQUEST_KEYS with a value at the top level or in
extra_body) and a resolved model string. A forbidden key sent empty (None, or an empty
list, tuple, dict or string) carries no value and passes: the Anthropic provider of the
pinned Inspect records "tools": [] on every call. Every call of the calibrator must
resolve to one string, and that string must be one the Opus 5.5 worker log resolved to;
a change is a vendor-change event (pre-registration section 5). The calls must also
send one reasoning mode (check_calibrator_modes).

Offline mode: a mockllm calibrator is accepted for the unit tests. A run that mixes real
and mockllm models is refused. Offline calls skip the serving guard, the request checks
and the resolved check.

Call refusals raise CalibratorError. Log checks raise roster.RunConfigError.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader, roster  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gc = _load("grading_common")
rr = _load("run_report")

CALIBRATOR_ROLE = roster.CALIBRATOR_ROLE
# Source of the transcript info event that records the endpoint the guard checked.
CALIBRATOR_EVENT_SOURCE = "calibrator_serving"
# Request keys a live calibrator call must not send with a value, at the top level or in
# extra_body. An empty value (None, or an empty list, tuple, dict or string) is not sent.
FORBIDDEN_REQUEST_KEYS = ("temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks", "stop_sequences")

# The effort field the Anthropic provider sends for the pinned reasoning_effort, as
# run_report.request_mode renders it.
_EFFORT_SENT = f"output_config.effort={json.dumps(roster.REASONING_EFFORT)}"


class CalibratorError(roster.RosterError):
    """A calibrator call off the pin: wrong model, wrong role binding or no messages."""


# --- Pins ------------------------------------------------------------------------------

def calibrator_pin() -> str:
    """The calibrator model pin (roster.role_pins, equal to roster.CALIBRATOR)."""
    return roster.role_pins(1)[CALIBRATOR_ROLE]


def calibrator_config():
    """The GenerateConfig of the calibrator call, built from
    roster.calibrator_generate_config_args() and nothing else."""
    from inspect_ai.model import GenerateConfig

    return GenerateConfig(**roster.calibrator_generate_config_args())


# --- Pure checks -----------------------------------------------------------------------

def check_calibrator_config(config: Any, where: str = "calibrator call", *, recorded: bool = False) -> dict[str, Any]:
    """Raise roster.RunConfigError unless a calibrator call's config is on the pin.

    config: a GenerateConfig, or a mapping of its set fields (None means unset).
    where: names the call in the error message.
    recorded: False for the config a call would send (checked before the call), True for
        the config a ModelEvent recorded.

    The rules are roster._check_generate for an evaluated-model config with batch not
    allowed: reasoning_effort equals roster.REASONING_EFFORT, no sampling parameter, no
    seed, no reasoning token budget, no fallback models, no fail-on-refusal setting, no
    response cache setting and only the operational fields besides; batch set to
    anything but unset or False is refused; extra_headers is refused. Before the call any
    max_tokens is refused. In a recorded event only roster.DEFAULT_MAX_TOKENS is
    accepted, the provider default Inspect fills in.

    Returns the set fields. Pure (no Inspect import)."""
    fields = grader._fields(config)
    roster._check_generate(where, fields, evaluated=True, allow_batch=False, max_tokens_pin=None,
                           provider_default=recorded)
    return fields


def check_calibrator_model(name: str) -> None:
    """Raise CalibratorError unless a resolved calibrator name is the calibrator pin, or
    mockllm (offline). Pure (no Inspect import)."""
    if name != calibrator_pin() and not roster._is_offline(name):
        raise CalibratorError(
            f"the calibrator resolved to {name!r}, not the calibrator pin (pre-registration section 5); "
            f"a calibrator role bound to another model is refused"
        )


def check_offline_pairing(name: str, active: str | None) -> None:
    """Raise CalibratorError when the calibrator and the evaluated model mix real and
    mockllm.

    A mockllm calibrator beside a real evaluated model, or a real calibrator beside a
    mockllm evaluated model, is refused. Inspect's none provider (grader.NO_MODEL) and no
    evaluated model (None) pair with either, since neither makes a call. Pure (no
    Inspect import)."""
    if active is None or active == grader.NO_MODEL:
        return
    if roster._is_offline(name) != roster._is_offline(active):
        raise CalibratorError(
            f"a run mixing real and {roster.OFFLINE_PROVIDER} models is refused: "
            f"calibrator {name!r}, evaluated model {active!r}"
        )


# --- The call --------------------------------------------------------------------------

def resolve_calibrator():
    """The Inspect Model that calibrates.

    get_model(pin, role="calibrator"): inside an eval that binds the calibrator role this
    is the bound model, otherwise a model built from the pin with the calibrator role
    set. Refused: a role bound to more than one model, a model other than the pin
    (mockllm excepted, see check_offline_pairing), and a model that does not carry the
    calibrator role (its ModelEvents would not be checked as calibrator calls). A real
    calibrator then passes roster.model_serving (exact first-party endpoint, no explicit
    base URL, no model_args, no auth token, no custom headers, no extra headers), and the
    checked endpoint is recorded in the transcript as an info event with source
    CALIBRATOR_EVENT_SOURCE."""
    from inspect_ai.log import transcript
    from inspect_ai.model import get_model, model_roles
    from inspect_ai.model._model import active_model

    bound = model_roles().get(CALIBRATOR_ROLE)
    if isinstance(bound, list):
        raise CalibratorError(
            f"the calibrator role is bound to more than one model ({len(bound)}); the roster binds one"
        )
    model = get_model(calibrator_pin(), role=CALIBRATOR_ROLE)
    name = str(model)
    check_calibrator_model(name)
    active = active_model()
    check_offline_pairing(name, None if active is None else str(active))
    if getattr(model, "role", None) != CALIBRATOR_ROLE:
        raise CalibratorError(
            f"the calibrator model carries role {getattr(model, 'role', None)!r}, not {CALIBRATOR_ROLE!r}"
        )
    if not roster._is_offline(name):
        endpoint = roster.model_serving(model)
        transcript().info({"serving_endpoint": endpoint}, source=CALIBRATOR_EVENT_SOURCE)
    return model


async def calibrate(messages: str | Sequence[Any]):
    """Call the calibrator once on the caller's messages and return its ModelOutput.

    messages: a non-empty string or list of ChatMessage. This module adds nothing to
        them: no system message, no prompt, no rubric wording.

    Before the call, the config Inspect would send (the model's own config, the inherited
    operational fields and calibrator_config()) must pass check_calibrator_config, so an
    off-pin call (a batch setting, a max_tokens, a sampling field) is refused before it
    is made. That config is read with Model._resolve_config, a private Inspect method,
    safe under the exact pin; a missing method fails closed. The call passes no tools
    and cache=False, and there is no fallback. This module calls generate once; Inspect's
    own retries of transient errors apply, and each errored attempt is skipped by the log
    check."""
    if isinstance(messages, str):
        if not messages.strip():
            raise CalibratorError("the calibrator call needs a non-empty message")
    elif not messages:
        raise CalibratorError("the calibrator call needs at least one message")
    model = resolve_calibrator()
    config = calibrator_config()
    try:
        effective = model._resolve_config(config)
    except AttributeError as e:
        raise CalibratorError(f"cannot read the effective config of the calibrator ({e}); refused") from e
    check_calibrator_config(effective, where=f"calibrator call to {model} (before the call)")
    return await model.generate(list(messages) if not isinstance(messages, str) else messages,
                                config=config, cache=False)


# --- Log checks ------------------------------------------------------------------------

def calibrator_events(log: Any) -> list[tuple[Any, Any]]:
    """(sample, ModelEvent) for every ModelEvent of the calibrator role in a full EvalLog."""
    found = []
    for sample in getattr(log, "samples", None) or ():
        for event in sample.events or ():
            if getattr(event, "event", None) == "model" and getattr(event, "role", None) == CALIBRATOR_ROLE:
                found.append((sample, event))
    return found


def _sent(value: Any) -> bool:
    """Whether a request value is sent: False for None and for an empty list, tuple, dict
    or string; True for everything else (0, 0.0 and False are sent)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return False
    return True


def _forbidden_sent(request: Mapping[str, Any]) -> list[str]:
    """The forbidden keys a request sends with a value, sorted."""
    return sorted(k for k in FORBIDDEN_REQUEST_KEYS if k in request and _sent(request[k]))


def _check_request(request: Any, where: str) -> str:
    """The reasoning mode a live calibrator request records as sent, after refusing a
    request that is not recorded, does not send effort high, or sends a forbidden key
    with a value (see _sent) at the top level or inside extra_body."""
    if not isinstance(request, Mapping):
        raise roster.RunConfigError(f"run config: {where} recorded no model-call request")
    mode = rr.request_mode(request)
    if _EFFORT_SENT not in mode:
        raise roster.RunConfigError(
            f"run config: {where} sent reasoning mode {mode!r}, not {_EFFORT_SENT} "
            f"(the calibrator pin, pre-registration section 5)"
        )
    top = _forbidden_sent(request)
    if top:
        raise roster.RunConfigError(f"run config: {where} sent {', '.join(top)}; the calibrator call sends none of them")
    extra = request.get("extra_body") or {}
    if not isinstance(extra, Mapping):
        raise roster.RunConfigError(f"run config: {where} sent an extra_body that is not a mapping; refused")
    inner = _forbidden_sent(extra)
    if inner:
        raise roster.RunConfigError(
            f"run config: {where} sent {', '.join(inner)} in extra_body; the calibrator call sends none of them"
        )
    return mode


def check_calibrator_event(event: Any, *, offline: bool = False) -> dict[str, Any]:
    """Raise roster.RunConfigError unless one ModelEvent is a calibrator call on the pin.

    Checks: the role is the calibrator role; the model is the calibrator pin (offline: a
    mockllm model); the recorded config passes check_calibrator_config with recorded
    True; it read or wrote no response cache; it holds no error; it offered no tools.
    Live only: the model-call request is recorded, sends output_config.effort high (read
    with run_report.request_mode) and sends none of FORBIDDEN_REQUEST_KEYS with a value at
    the top level or inside extra_body; the response recorded a resolved model string.

    A refusal (grading_common.is_refusal: a refusal stop reason, refusal content or
    refusal stop details) passes and is marked, never an error.

    Returns {"model", "resolved", "config", "usage", "refused", "request_mode"}: the
    requested model, the resolved model string (empty when none was recorded), the set
    config fields, the token usage as (input, output) or None, whether the call was
    refused, and the reasoning mode sent (None offline). Pure (no Inspect import)."""
    role = getattr(event, "role", None)
    if role != CALIBRATOR_ROLE:
        raise roster.RunConfigError(f"run config: a ModelEvent with role {role!r} is not a calibrator call")
    model = getattr(event, "model", None)
    where = "calibrator ModelEvent"
    if offline:
        if not roster._is_offline(model):
            raise roster.RunConfigError(
                f"run config: {where} called a real model in an offline check; refused"
            )
    elif model != calibrator_pin():
        raise roster.RunConfigError(f"run config: {where} called {model!r}, not the calibrator pin")
    fields = check_calibrator_config(getattr(event, "config", None), where, recorded=True)
    cache = getattr(event, "cache", None)
    if cache is not None:
        raise roster.RunConfigError(
            f"run config: {where} has cache={cache!r}; response caching is disabled for the calibrator (D-R23, calibrator)"
        )
    error = getattr(event, "error", None)
    if error:
        raise roster.RunConfigError(f"run config: {where} ended in an error: {error}")
    if getattr(event, "tools", None):
        raise roster.RunConfigError(f"run config: {where} offered tools; the calibrator call offers none")
    output = getattr(event, "output", None)
    resolved = str(getattr(output, "model", "") or "")
    mode = None
    if not offline:
        call = getattr(event, "call", None)
        if call is None:
            raise roster.RunConfigError(f"run config: {where} recorded no model call")
        mode = _check_request(getattr(call, "request", None), where)
        if not resolved:
            raise roster.RunConfigError(
                f"run config: {where} recorded no resolved model string; section 5 logs it on every response, "
                f"so a vendor change cannot be ruled out"
            )
    usage = getattr(output, "usage", None)
    return {
        "model": model,
        "resolved": resolved,
        "config": fields,
        "usage": (usage.input_tokens, usage.output_tokens) if usage is not None else None,
        "refused": gc.is_refusal(output),
        "request_mode": mode,
    }


def check_calibrator_resolved(summaries: Sequence[Mapping[str, Any]], allowed: Sequence[str] | None) -> None:
    """Raise roster.RunConfigError unless every calibrator call resolved to one recorded
    string, and that string is in allowed when allowed is given.

    allowed: the resolved strings of the Opus 5.5 worker log (WorkerLog.resolved). A
    string outside it is a vendor-change event (pre-registration section 5). Pure."""
    for s in summaries:
        if not str(s.get("resolved") or ""):
            raise roster.RunConfigError(
                f"run config: calibrator call (unit {s.get('sample_id')}) recorded no resolved model string; "
                f"section 5 logs it on every response, so a vendor change cannot be ruled out"
            )
    seen = sorted({str(s.get("resolved")) for s in summaries})
    if len(seen) > 1:
        raise roster.RunConfigError(
            f"run config: the calibrator calls resolved to {len(seen)} model strings, not one; "
            f"a change is a vendor-change event (pre-registration section 5)"
        )
    if allowed is not None and seen and seen[0] not in set(allowed):
        raise roster.RunConfigError(
            "run config: the calibrator calls resolved to a string the calibrator's worker log never resolved to; "
            "a change is a vendor-change event (pre-registration section 5)"
        )


def check_calibrator_modes(summaries: Sequence[Mapping[str, Any]]) -> None:
    """Raise roster.RunConfigError when the calibrator calls sent more than one reasoning
    mode.

    summaries: check_calibrator_event summaries, from one attempt log or from every
    attempt log of the job. Their request_mode values that are not None (an offline call
    records None) must number one at most. Pure."""
    seen = sorted({str(s.get("request_mode")) for s in summaries if s.get("request_mode") is not None})
    if len(seen) > 1:
        raise roster.RunConfigError(
            f"run config: the calibrator calls sent {len(seen)} reasoning modes, not one; "
            f"every calibrator call sends the pinned mode ({'; '.join(seen)})"
        )


def _check_completed(calls: Mapping[str, Any], *, offline: bool) -> list[dict[str, Any]]:
    summaries = []
    for uid, event in calls["completed"].items():
        summary = check_calibrator_event(event, offline=offline)
        if not offline:
            gc.check_serving_recorded(calls["samples"][uid], source=CALIBRATOR_EVENT_SOURCE, provider="anthropic")
        summary["sample_id"] = uid
        summaries.append(summary)
    return summaries


def check_calibrator_log(log: Any, *, asked: Sequence[str], offline: bool = False,
                         allowed_resolved: Sequence[str] | None = None) -> tuple[dict, list[dict]]:
    """Raise roster.RunConfigError unless a finished calibrator attempt log is on the pin.

    log: a full EvalLog (samples and events read, not header_only).
    asked: the sample ids (unit ids or record names) the attempt was asked.

    The log must end in success and hold, for every asked id, exactly one completed
    calibrator call (grading_common.role_calls with require_all True: an errored call
    followed by a completed one is skipped; an error after it or a second completed call
    is refused). Every completed call passes check_calibrator_event. Live: every
    completed sample records the serving endpoint the guard checked, the calls pass
    check_calibrator_resolved against allowed_resolved, and they sent one reasoning mode
    (check_calibrator_modes).

    Returns (calls, summaries): the role_calls result and one check_calibrator_event
    summary per completed call, each with its sample_id added."""
    status = getattr(log, "status", None)
    if status != "success":
        raise roster.RunConfigError(f"run config: log status is {status!r}, not 'success'")
    calls = gc.role_calls(log, CALIBRATOR_ROLE, asked, require_all=True)
    summaries = _check_completed(calls, offline=offline)
    if not offline:
        check_calibrator_resolved(summaries, allowed_resolved)
        check_calibrator_modes(summaries)
    return calls, summaries


def admit_calibrator_attempt_log(log: Any, *, asked: Sequence[str], offline: bool = False) -> tuple[dict, list[dict]]:
    """Check a calibrator attempt log that did not end in success (D-R23, reply of record).

    The completed calls it holds are kept: grading_common.role_calls with require_all
    False, every completed call through check_calibrator_event, and live, the serving
    endpoint recorded on every completed sample. The resolved check and the mode check
    (check_calibrator_modes) are left to the caller, which runs them over every attempt
    log of the job together.

    Returns (calls, summaries) as check_calibrator_log does."""
    calls = gc.role_calls(log, CALIBRATOR_ROLE, asked, require_all=False)
    summaries = _check_completed(calls, offline=offline)
    return calls, summaries
