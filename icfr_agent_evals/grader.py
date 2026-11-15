"""The grader call: the one place a grader model is made and called.

Pre-registration section 5 fixes the grader ladder (GPT-6 Luna at step 1, GPT-6.1
Terra at step 2, GPT-6.1 Sol at step 3, decision D-T1) and the grader configuration:
reasoning_effort pinned for each grader with no sampling parameter set (decision
D-T1), high at every step (decision D-R12), and steps 2 and 3 on the OpenAI Batch
API, on the grader role only. Section 7 scores the test-retest replicates with
response caching disabled and the grader configuration pinned. Section 13 lists this
call as tooling built after the tag and before the first evaluation-split run
(decision D-R19, item 2).

What this module does:
  - resolve_grader(step) resolves the grader for a ladder step. The pin comes from
    roster.GRADER_BY_STEP. A bound grader role wins over the pin, so a role bound to
    anything else is refused. A role bound to more than one model is refused. A real
    grader then passes the roster serving guard (roster.model_serving), and the
    endpoint it checked is recorded in the sample transcript.
  - grader_config(step) builds the GenerateConfig from
    roster.grader_generate_config_args(step) and nothing else.
  - grade(messages, step) checks the config the call would send against the pin,
    then calls generate once, with no tools and with response caching disabled
    (cache=False). A role model inherits an eval-wide cache setting into the config
    it records, so a run with one is refused before the call.
  - check_grader_config, check_grader_event and check_grader_log check what a call
    recorded against the pin. They reuse the grader branch of roster's run-config
    check (roster._check_generate with grader=True) and add one rule the pin fixes:
    batch is True at steps 2 and 3 and unset at step 1.

This module holds no grader prompt and no rubric wording. The caller passes the
messages. The prompt is written in the judge-development step (pre-registration
section 7, judge remedy order, step 1).

Offline mode: a mockllm grader is accepted for the unit tests, as in
roster.validate_roster. A run that mixes real and mockllm models is refused (a mock
grader beside a real evaluated model, or a real grader beside a mock one). Inspect's
none provider, or no evaluated model at all, pairs with either. Offline graders skip
the serving guard.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from icfr_agent_evals import roster

GRADER_ROLE = roster.GRADER_ROLE
# Inspect's none provider: a run that binds it makes no call as the evaluated model.
NO_MODEL = "none/none"
# Source of the transcript info event that records the endpoint the guard checked.
GRADER_EVENT_SOURCE = "grader_serving"

# The resolved model string each grader alias returned at the tag-time live call
# (pre-registration section 13, grader aliases row; docs/evidence/smoke-2026-11-14.md).
# Section 5: any change is a vendor-change event that triggers an ADR and a re-run of
# the judge-trust gates before the next reported run.
TAG_TIME_RESOLVED: dict[int, str] = {
    1: "gpt-6-luna",
    2: "gpt-6.1-terra",
    3: "gpt-6.1-sol",
}


class GraderError(roster.RosterError):
    """A grader call off the ladder pin: wrong model, wrong role binding or no messages."""


# --- Pins ------------------------------------------------------------------------------

def grader_pin(step: int) -> str:
    """The grader model pin at a ladder step (roster.GRADER_BY_STEP). A step outside
    roster.GRADER_STEPS raises roster.RosterError."""
    return roster.role_pins(step)[GRADER_ROLE]


def grader_config(step: int):
    """The GenerateConfig of the grader call at a ladder step, built from
    roster.grader_generate_config_args(step) and nothing else."""
    from inspect_ai.model import GenerateConfig

    return GenerateConfig(**roster.grader_generate_config_args(step))


# --- Pure checks -----------------------------------------------------------------------

def _fields(config: Any) -> dict[str, Any]:
    if config is None or isinstance(config, Mapping):
        return {k: v for k, v in (config or {}).items() if v is not None}
    return roster._generate_fields(config)


def check_grader_config(config: Any, step: int, where: str = "grader call") -> dict[str, Any]:
    """Raise roster.RunConfigError unless a grader call's config is on the step's pin.

    config: a GenerateConfig, or a mapping of its set fields (None means unset).
    step: the ladder step the call was made at.
    where: names the call in the error message.

    The rules are the grader branch of roster.check_run_config (roster._check_generate
    with grader=True): reasoning_effort equals GRADER_REASONING_EFFORT, no sampling
    parameter, no seed, no reasoning token budget, no max_tokens, no extra_headers, no
    response cache setting, and only the operational fields besides. Batch is allowed
    only at steps 2 and 3. On top of that, batch must equal the pin: True at steps 2
    and 3 (an integer batch size is not the pin), unset or False at step 1.

    Returns the set fields. Pure (no Inspect import)."""
    pin = roster.grader_generate_config_args(step)
    fields = _fields(config)
    roster._check_generate(where, fields, evaluated=True, allow_batch=step in roster.GRADER_BATCH_STEPS,
                           max_tokens_pin=None, grader=True)
    batch = fields.get("batch")
    if "batch" in pin and batch is not pin["batch"]:
        raise roster.RunConfigError(
            f"run config: {where} has batch={batch!r}; the step {step} grader runs on the OpenAI Batch API "
            f"(batch={pin['batch']!r}, pre-registration section 5)"
        )
    return fields


def check_grader_event(event: Any, step: int) -> dict[str, Any]:
    """Raise roster.RunConfigError unless one ModelEvent is a grader call on the pin.

    Checks: the event's role is the grader role; its model is the step's pin; its config
    passes check_grader_config; it read or wrote no response cache; it holds no error.

    Returns {"model", "resolved", "config", "usage"}: the requested model, the resolved
    model string the response recorded (empty when none was recorded), the set config
    fields, and the token usage as (input, output) or None. Pure (no Inspect import)."""
    role = getattr(event, "role", None)
    if role != GRADER_ROLE:
        raise roster.RunConfigError(f"run config: a ModelEvent with role {role!r} is not a grader call")
    pin = grader_pin(step)
    model = getattr(event, "model", None)
    where = f"grader ModelEvent ({model})"
    if model != pin:
        raise roster.RunConfigError(f"run config: {where} called {model!r}, not the step {step} pin {pin!r}")
    fields = check_grader_config(getattr(event, "config", None), step, where)
    cache = getattr(event, "cache", None)
    if cache is not None:
        raise roster.RunConfigError(
            f"run config: {where} has cache={cache!r}; response caching is disabled for the grader "
            f"(pre-registration sections 5 and 7)"
        )
    error = getattr(event, "error", None)
    if error:
        raise roster.RunConfigError(f"run config: {where} ended in an error: {error}")
    output = getattr(event, "output", None)
    usage = getattr(output, "usage", None)
    return {
        "model": model,
        "resolved": str(getattr(output, "model", "") or ""),
        "config": fields,
        "usage": (usage.input_tokens, usage.output_tokens) if usage is not None else None,
    }


def grader_events(log: Any) -> list[tuple[Any, Any]]:
    """(sample, ModelEvent) for every ModelEvent of the grader role in a full EvalLog."""
    found = []
    for sample in getattr(log, "samples", None) or ():
        for event in sample.events or ():
            if getattr(event, "event", None) == "model" and getattr(event, "role", None) == GRADER_ROLE:
                found.append((sample, event))
    return found


def check_grader_log(log: Any, step: int) -> list[dict[str, Any]]:
    """Raise roster.RunConfigError unless a finished EvalLog holds grader calls and every
    one of them is on the step's pin (check_grader_event).

    log: a full EvalLog (samples and events read, not header_only).

    Returns one check_grader_event summary per grader call, each with the sample id and
    epoch added. Pure (no Inspect import)."""
    status = getattr(log, "status", None)
    if status != "success":
        raise roster.RunConfigError(f"run config: log status is {status!r}, not 'success'")
    if getattr(log, "samples", None) is None:
        raise roster.RunConfigError("run config: the log has no samples; read it in full, not header_only")
    events = grader_events(log)
    if not events:
        raise roster.RunConfigError("run config: the log holds no grader ModelEvent")
    summaries = []
    for sample, event in events:
        summary = check_grader_event(event, step)
        summary.update(sample_id=sample.id, epoch=sample.epoch)
        summaries.append(summary)
    return summaries


def _is_offline(model: str | None) -> bool:
    return roster._is_offline(model)


def check_offline_pairing(grader: str, evaluated: str | None) -> None:
    """Raise GraderError when a grader and the evaluated model mix real and mockllm.

    As in roster.validate_roster, a run mixing real and mockllm models is refused: a
    mockllm grader beside a real evaluated model, or a real grader beside a mockllm
    evaluated model. Inspect's none provider (NO_MODEL) and no evaluated model (None)
    pair with either, since neither makes a call. Pure (no Inspect import)."""
    if evaluated is None or evaluated == NO_MODEL:
        return
    if _is_offline(grader) != _is_offline(evaluated):
        raise GraderError(
            f"a run mixing real and {roster.OFFLINE_PROVIDER} models is refused: "
            f"grader {grader!r}, evaluated model {evaluated!r}"
        )


def check_grader_model(name: str, step: int) -> None:
    """Raise GraderError unless a resolved grader name is the step's pin, or mockllm
    (offline). Pure (no Inspect import)."""
    pin = grader_pin(step)
    if name != pin and not _is_offline(name):
        raise GraderError(
            f"the grader resolved to {name!r}, not the step {step} pin {pin!r} (pre-registration section 5); "
            f"a grader role bound to another model is refused"
        )


# --- The call --------------------------------------------------------------------------

def resolve_grader(step: int):
    """The Inspect Model that grades at a ladder step.

    get_model(pin, role="grader"): inside an eval that binds the grader role this is
    the bound model, otherwise a model built from the pin with the grader role set.
    Refused: a role bound to more than one model, a model other than the pin (mockllm
    excepted, see check_offline_pairing), and a model that does not carry the grader
    role (its ModelEvents would not be checked as grader calls). A real grader then
    passes roster.model_serving, and the checked endpoint is recorded in the
    transcript as an info event with source GRADER_EVENT_SOURCE."""
    from inspect_ai.log import transcript
    from inspect_ai.model import get_model, model_roles
    from inspect_ai.model._model import active_model

    pin = grader_pin(step)
    bound = model_roles().get(GRADER_ROLE)
    if isinstance(bound, list):
        raise GraderError(f"the grader role is bound to more than one model ({len(bound)}); the roster binds one")
    model = get_model(pin, role=GRADER_ROLE)
    name = str(model)
    check_grader_model(name, step)
    active = active_model()
    check_offline_pairing(name, None if active is None else str(active))
    if getattr(model, "role", None) != GRADER_ROLE:
        raise GraderError(f"the grader model {name!r} carries role {getattr(model, 'role', None)!r}, not {GRADER_ROLE!r}")
    if not _is_offline(name):
        endpoint = roster.model_serving(model)
        transcript().info({"grader_step": step, "serving_endpoint": endpoint}, source=GRADER_EVENT_SOURCE)
    return model


async def grade(messages: str | Sequence[Any], step: int):
    """Call the step's grader once on the caller's messages and return its ModelOutput.

    messages: a non-empty string or list of ChatMessage. This module adds nothing to
        them: no system message, no prompt, no rubric wording.
    step: the grader ladder step (1, 2 or 3).

    Before the call, the config Inspect would send (the model's own config, the
    inherited operational fields and grader_config(step)) must pass
    check_grader_config, so an off-pin call is refused before it is made. That config
    is read with Model._resolve_config, a private Inspect method, safe under the exact
    pin; a missing method fails closed. The call passes no tools and cache=False."""
    if isinstance(messages, str):
        if not messages.strip():
            raise GraderError("the grader call needs a non-empty message")
    elif not messages:
        raise GraderError("the grader call needs at least one message")
    model = resolve_grader(step)
    config = grader_config(step)
    try:
        effective = model._resolve_config(config)
    except AttributeError as e:
        raise GraderError(f"cannot read the effective config of the grader {model!r} ({e}); refused") from e
    check_grader_config(effective, step, where=f"grader call to {model} (before the call)")
    return await model.generate(messages if isinstance(messages, str) else list(messages),
                                config=config, cache=False)
