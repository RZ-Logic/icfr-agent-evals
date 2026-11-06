"""The frozen Phase 1 model roster and the setup check that enforces it.

Section 5 of the pre-registration fixes this roster. The evaluated model comes
from --model and is checked here at setup. It is never set on the Task, since a
model set there would take precedence over --model without a warning. The grader
and calibrator roles must both be bound. In the pinned inspect-ai, a get_model call
for a role nobody bound quietly hands back the evaluated model, so the setup solver
always asks with required=True. Each of these stops the run instead of falling
back: a role left unbound, a role bound to something other than its pin, a role
bound to more than one model (Inspect would otherwise check only the first), a
grader that is the evaluated model, or an evaluated model outside the roster.

Offline mode: the mock smoke and the unit tests run on Inspect's mockllm provider.
A run is offline only when the evaluated model and every role are mockllm; a run
that mixes real and mock models is rejected.

Serving guard: a model name says nothing about where its requests go. A base URL
set by ANTHROPIC_BASE_URL, OPENAI_BASE_URL or INSPECT_EVAL_MODEL_BASE_URL (from the
shell or a .env file) moves every request to another host, and neither the model
name nor the EvalLog shows it. So on a real run the setup check also reads each
model's resolved client endpoint and stops the run unless it is exactly the
vendor's first-party API (FIRST_PARTY_BASE_URLS). It also stops on an explicit
base URL override, on any model_args, on a blank endpoint, and when
ANTHROPIC_AUTH_TOKEN is set. It also stops when ANTHROPIC_CUSTOM_HEADERS or
OPENAI_CUSTOM_HEADERS is set to a non-blank value, since either can replace the
request credential while the host stays first-party. For the same reason it stops
when request headers from a GenerateConfig are in force: INSPECT_EVAL_EXTRA_HEADERS
set to a non-blank value (the CLI reads it as --extra-headers), extra_headers on the
active generate config (the eval config merged into the evaluated model's requests),
or extra_headers on a model's own config (a role spec). The SDKs layer request headers
after their auth headers, so any of these could swap the credential too. Each sample's
transcript records the endpoints it checked. This is what makes the fixed
SERVING_PLATFORMS claim true.

Run config check: check_run_config reads a finished EvalLog and raises
RunConfigError unless the run used the section 5 configuration. CLI generate flags
(--reasoning-effort, --max-tokens, --seed and the sampling flags)
replace the Task's GenerateConfig without a warning and the run still ends in
success, so the readers check the log itself: plan.config and every ModelEvent the
evaluated model sent.

validate_roster and validate_serving are pure (no Inspect import) so each
violation has a unit test; roster_check is the setup solver that resolves the
models and calls them.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

# --- Pins (section 5 table; frozen at the tag, section 13) ---------------------------

WORKER = "anthropic/claude-sonnet-5-5"
BASELINES: tuple[str, ...] = (
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5-5",
)
EVALUATED_MODELS: tuple[str, ...] = (WORKER, *BASELINES)

GRADER = "openai/gpt-6-luna"             # an alias (ADR-0004); each response records the model that served it
GRADER_ESCALATION = "openai/gpt-6.1-sol"  # step 2 of the judge remedy order (section 7)
CALIBRATOR = "anthropic/claude-opus-5-5"  # its sample skips the outputs of the Opus 5.5 baseline run
SCOUT_SCANNER = "openai/gpt-6-luna"       # named in each scan config; not an eval role

GRADER_ROLE = "grader"
CALIBRATOR_ROLE = "calibrator"
REQUIRED_ROLES: tuple[str, ...] = (GRADER_ROLE, CALIBRATOR_ROLE)
GRADER_STEPS: tuple[int, ...] = (1, 2)

# One effort setting for each Anthropic model in the roster, whatever role it holds.
REASONING_EFFORT = "high"

OFFLINE_PROVIDER = "mockllm"

# --- Reperformance window (section 5) ------------------------------------------------

# Date the floors were last read for the Claude API, from the model status table at
# https://platform.claude.com/docs/en/about-claude/model-deprecations ("Tentative
# retirement date: Not sooner than ..."), cross-checked against the Retirement row of
# https://platform.claude.com/docs/en/about-claude/models/overview. First read 2026-10-08.
RETIREMENT_FLOORS_RECORDED = "2026-11-04"
RETIREMENT_FLOORS: dict[str, str] = {
    "anthropic/claude-sonnet-5": "2027-06-30",
    "anthropic/claude-opus-5-5": "2027-09-22",
    "anthropic/claude-sonnet-5-5": "2027-09-28",
    "anthropic/claude-haiku-5-5": "2027-10-07",
}
# Wording follows each vendor's deprecation page as read on RETIREMENT_FLOORS_RECORDED:
# https://platform.claude.com/docs/en/about-claude/model-deprecations (Notifications) and
# https://developers.openai.com/api/docs/deprecations (Model deprecation notice periods).
# The two Anthropic bounds are separate: retirement no sooner than the floor, and no
# sooner than 60 days after notice. Notice can come before the floor (Opus 4.1 retired
# on its floor). The floors cover the Claude models only, so re-execution lasts until
# the first roster model retires, not until the Claude floors.
NOTICE_POLICIES: dict[str, str] = {
    "anthropic": (
        "each floor is a tentative retirement date (not sooner than the listed date); "
        "customers with active deployments get at least 60 days of notice before a publicly "
        "released model is retired; notice can come before the floor, so retirement can "
        "fall less than 60 days after it"
    ),
    "openai": (
        "at least 6 months of notice before retiring a generally available model, unless "
        "safety or compliance concerns require a faster timeline; no retirement date is "
        "listed for the grader and scanner models, so their availability rests on the "
        "notice period alone (their generally available status is inferred from the "
        "absence of a preview marker, not stated by OpenAI)"
    ),
}

# Serving platform per provider prefix (section 5: partner platforms set their own
# retirement schedules, so the run config names the platform). The pins name no
# service prefix (such as anthropic/bedrock/ or openai/azure/), and validate_roster
# rejects any name that differs from its pin. The name alone does not fix the host,
# so roster_check also enforces the endpoint below on every real run.
SERVING_PLATFORMS: dict[str, str] = {
    "anthropic": "Claude API (Anthropic first-party)",
    "openai": "OpenAI API (OpenAI first-party)",
}

# --- Serving guard (section 5 serving platform, section 13 roster check) -------------

# The only client endpoints a real run may use, compared exactly after stripping one
# trailing slash. They are the SDK defaults in anthropic 1.12.1 and openai 3.27.0.
FIRST_PARTY_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
}
# Inspect authenticates with this token instead of ANTHROPIC_API_KEY when it is set.
AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
# The SDKs add these headers after their own auth headers, so either can replace the
# request credential while the host stays first-party. Checked for every provider.
CUSTOM_HEADERS_ENV: tuple[str, ...] = ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS")
# Inspect's CLI reads this variable as --extra-headers into the eval GenerateConfig,
# whose extra_headers the providers copy into every request after the auth headers.
EXTRA_HEADERS_ENV = "INSPECT_EVAL_EXTRA_HEADERS"
ACTIVE_CONFIG_SOURCE = "active GenerateConfig.extra_headers"
SERVING_EVENT_SOURCE = "roster_check"


class RosterError(RuntimeError):
    """A run whose evaluated model or judge roles are off the frozen roster."""


def role_pins(grader_step: int = 1) -> dict[str, str]:
    if grader_step not in GRADER_STEPS:
        raise RosterError(f"grader_step must be one of {GRADER_STEPS}, got {grader_step!r}")
    return {
        GRADER_ROLE: GRADER if grader_step == 1 else GRADER_ESCALATION,
        CALIBRATOR_ROLE: CALIBRATOR,
    }


def calibrator_may_sample(evaluated_model: str) -> bool:
    """The calibrator never samples Opus 5.5 baseline generations (section 5)."""
    return evaluated_model != CALIBRATOR


def _is_offline(model: str | None) -> bool:
    return model is not None and model.split("/", 1)[0] == OFFLINE_PROVIDER


def validate_roster(evaluated_model: str, roles: Mapping[str, str | None], grader_step: int = 1) -> None:
    """Raise RosterError unless the run matches the frozen roster.

    evaluated_model: the --model the run evaluates, as "provider/name".
    roles: every role bound for the run (role -> "provider/name"); None or a
        missing key means unbound.
    grader_step: 1, or 2 after step 2 of the judge remedy order moves the grader.
    """
    pins = role_pins(grader_step)
    unbound = [r for r in REQUIRED_ROLES if not roles.get(r)]
    if unbound:
        raise RosterError(f"required judge role(s) unbound: {unbound}")
    unknown = sorted(r for r in roles if r not in REQUIRED_ROLES)
    if unknown:
        raise RosterError(f"role(s) not on the frozen roster: {unknown}")

    models = [evaluated_model, *(roles[r] for r in REQUIRED_ROLES)]
    offline = [_is_offline(m) for m in models]
    if any(offline):
        if not all(offline):
            raise RosterError(f"a run mixing real and {OFFLINE_PROVIDER} models is rejected: {models}")
        return  # offline smoke or unit test: no real model is reachable

    if evaluated_model not in EVALUATED_MODELS:
        raise RosterError(f"evaluated model {evaluated_model!r} is not on the roster {EVALUATED_MODELS}")
    if roles[GRADER_ROLE] == evaluated_model:
        raise RosterError(f"grader {roles[GRADER_ROLE]!r} equals the evaluated model")
    for role in REQUIRED_ROLES:
        if roles[role] != pins[role]:
            raise RosterError(f"role {role!r} is bound to {roles[role]!r}, not its pin {pins[role]!r}")


def _strip_one_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


def validate_serving(
    provider: str,
    effective_base_url: Any,
    explicit_base_url: Any = None,
    model_args: Any = None,
    auth_token_set: bool = False,
    model: str | None = None,
    custom_headers: Sequence[str] = (),
    extra_headers: Sequence[str] | None = None,
) -> str:
    """Raise RosterError unless a real model is served by its vendor's first-party API.

    provider: the model's provider prefix ("anthropic" or "openai").
    effective_base_url: the base URL the model's client will send requests to.
    explicit_base_url: a base URL set on the model itself (--model-base-url, a model
        spec, a role spec, or get_model(base_url=...)); None when there is none.
    model_args: the model's model_args. No roster run uses any, so any key fails.
        Several can move the host or transport (base_url, http_client, profile,
        config, credentials, provider, workload_identity, data_residency,
        websocket_base_url).
    auth_token_set: True when ANTHROPIC_AUTH_TOKEN is set and not blank.
    model: the "provider/name" string, for the error message only.
    custom_headers: the CUSTOM_HEADERS_ENV variables that are set and not blank;
        any one fails.
    extra_headers: the sources of GenerateConfig request headers in force for this
        model (extra_headers_in_force plus the model's own config); any one fails.
        None (the default) reads extra_headers_in_force(), so a direct call also
        refuses INSPECT_EVAL_EXTRA_HEADERS; an explicit empty list checks none.

    Returns the endpoint with one trailing slash stripped.
    """
    who = f"{model!r}" if model else f"the {provider!r} model"
    if extra_headers is None:
        extra_headers = extra_headers_in_force()
    allowed = FIRST_PARTY_BASE_URLS.get(provider)
    if allowed is None:
        raise RosterError(f"serving guard: no first-party endpoint is pinned for provider {provider!r} ({who})")
    if provider == "anthropic" and auth_token_set:
        raise RosterError(
            f"serving guard: {AUTH_TOKEN_ENV} is set, so Inspect would authenticate {who} "
            f"with it instead of ANTHROPIC_API_KEY; unset it"
        )
    if custom_headers:
        raise RosterError(
            f"serving guard: {', '.join(custom_headers)} is set, so the client of {who} would "
            f"send those headers, which can replace the request credential; unset it"
        )
    if extra_headers:
        raise RosterError(
            f"serving guard: request headers are in force for {who} ({', '.join(extra_headers)}); "
            f"the provider would send them after its auth headers, so they can replace the "
            f"request credential; remove them (no roster run sets extra_headers)"
        )
    if model_args is None:
        model_args = {}
    if not isinstance(model_args, Mapping):
        raise RosterError(f"serving guard: model_args of {who} is not a mapping ({type(model_args).__name__}); refused")
    if model_args:
        raise RosterError(
            f"serving guard: {who} has model_args {sorted(map(str, model_args))}; "
            f"no roster model takes model_args (-M or a role spec), since some move the host"
        )
    if explicit_base_url is not None:
        if not isinstance(explicit_base_url, str) or _strip_one_slash(explicit_base_url) != allowed:
            raise RosterError(
                f"serving guard: {who} has an explicit base URL override {explicit_base_url!r}; "
                f"only {allowed!r} is allowed"
            )
    if not isinstance(effective_base_url, str):
        raise RosterError(
            f"serving guard: the client endpoint of {who} is not a string "
            f"({type(effective_base_url).__name__}); refused"
        )
    if not effective_base_url.strip():
        raise RosterError(
            f"serving guard: the client endpoint of {who} is blank; a blank *_BASE_URL "
            f"variable (ANTHROPIC_BASE_URL, OPENAI_BASE_URL or INSPECT_EVAL_MODEL_BASE_URL) is set"
        )
    endpoint = _strip_one_slash(effective_base_url)
    if endpoint != allowed:
        raise RosterError(
            f"serving guard: {who} would send requests to {effective_base_url!r}, not the "
            f"first-party endpoint {allowed!r}; check ANTHROPIC_BASE_URL, OPENAI_BASE_URL and "
            f"INSPECT_EVAL_MODEL_BASE_URL in the shell and any .env file"
        )
    return endpoint


def _url_types() -> tuple[type, ...]:
    """The URL classes an SDK client's base_url may be: httpx2.URL (the pinned SDKs)
    or httpx.URL. Whichever of the two is not installed is left out."""
    found: list[type] = []
    for module in ("httpx2", "httpx"):
        try:
            found.append(__import__(module).URL)
        except (ImportError, AttributeError):
            pass
    return tuple(found)


def custom_headers_set() -> list[str]:
    """The CUSTOM_HEADERS_ENV variables set to a non-blank value."""
    return [name for name in CUSTOM_HEADERS_ENV if os.environ.get(name, "").strip()]


def extra_headers_in_force() -> list[str]:
    """Name each source of GenerateConfig request headers in force (empty when none).

    Sources: EXTRA_HEADERS_ENV set to a non-blank value, and non-empty extra_headers
    on Inspect's active generate config (the eval config, which Inspect merges into
    every request of the evaluated model; empty outside an eval). Names only, never
    the header values, since a header can hold a credential."""
    found = [EXTRA_HEADERS_ENV] if os.environ.get(EXTRA_HEADERS_ENV, "").strip() else []
    from inspect_ai.model._generate_config import active_generate_config

    if active_generate_config().extra_headers:
        found.append(ACTIVE_CONFIG_SOURCE)
    return found


def model_serving(model: Any) -> dict[str, str]:
    """Check one resolved Inspect Model with validate_serving and return what was
    checked: {"model": "provider/name", "base_url": the client's base URL as read}.

    The endpoint is read as str(model.api.client.base_url), the SDK's URL object
    (httpx2.URL in anthropic 1.12.1 and openai 3.27.0). That is a private Inspect
    attribute, safe under the exact pin; a missing attribute or an unexpected type
    fails closed. Request headers in force (extra_headers_in_force, or extra_headers
    on model.config) also fail."""
    url_types = _url_types()
    try:
        name = str(model)
        client = model.api.client
        url = client.base_url
        explicit = [model.explicit_base_url, model.api.base_url]
        model_args = model.model_args
        own_config = getattr(model, "config", None)
    except AttributeError as e:
        raise RosterError(f"serving guard: cannot read the client endpoint of {model!r} ({e}); refused") from e
    if not url_types or not isinstance(url, url_types):
        raise RosterError(
            f"serving guard: the client endpoint of {name!r} is a "
            f"{type(url).__module__}.{type(url).__name__}, not an httpx2 or httpx URL; refused"
        )
    provider = name.split("/", 1)[0]
    auth_token_set = bool(os.environ.get(AUTH_TOKEN_ENV)) or bool(getattr(client, "auth_token", None))
    observed = str(url)
    headers = custom_headers_set()
    extra = extra_headers_in_force()
    if getattr(own_config, "extra_headers", None):
        extra.append(f"{name} config.extra_headers")
    for explicit_url in explicit:
        validate_serving(provider, observed, explicit_url, model_args, auth_token_set, model=name,
                         custom_headers=headers, extra_headers=extra)
    return {"model": name, "base_url": observed}


def serving_endpoints(models: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """model_serving for each named model (for example "evaluated_model", "grader",
    "calibrator"); the result is what the sample transcript records."""
    return {key: model_serving(m) for key, m in models.items()}


def roster_metadata(grader_step: int = 1) -> dict:
    """The roster as recorded in Task metadata, so the run config is reproducible
    from the EvalLog header alone."""
    return {
        "worker": WORKER,
        "baselines": list(BASELINES),
        "roles": role_pins(grader_step),
        "grader_escalation": GRADER_ESCALATION,
        "grader_step": grader_step,
        "scout_scanner": SCOUT_SCANNER,
        "reasoning_effort": REASONING_EFFORT,
        "serving_platforms": dict(SERVING_PLATFORMS),
    }


def retirement_metadata() -> dict:
    return {
        "recorded": RETIREMENT_FLOORS_RECORDED,
        "platform": SERVING_PLATFORMS["anthropic"],
        "floors": dict(RETIREMENT_FLOORS),
        "notice_policies": dict(NOTICE_POLICIES),
        "serving_platforms": dict(SERVING_PLATFORMS),
    }


# --- Run config check (section 5 reasoning, sampling parameters and seed) -----------

# GenerateConfig fields a run may set: transport, retry and connection limits. None
# of them changes what the provider is asked to sample.
RUN_CONFIG_OPERATIONAL: tuple[str, ...] = (
    "max_retries", "timeout", "attempt_timeout", "stream_idle_timeout",
    "max_connections", "adaptive_connections",
)
# Every other set field refuses and is named with its value: section 5 sets no
# sampling parameter, no reasoning token budget and no seed. (The field names are not
# spelled out here; tests/test_task.py keeps sampling parameter names out of the package.)
# Fields with their own rule: reasoning_effort equals the pin, max_tokens equals an
# owner pin when one is given, batch only when allowed, extra_headers empty.
_RUN_CONFIG_RULED = ("reasoning_effort", "max_tokens", "batch", "extra_headers")


class RunConfigError(RuntimeError):
    """A finished run whose effective eval or generate config is off the section 5 pins."""


def _generate_fields(config: Any) -> dict[str, Any]:
    """The fields of a GenerateConfig that are set (None means unset)."""
    if config is None:
        return {}
    return {k: v for k, v in config.model_dump(exclude_none=True).items() if v is not None}


def _check_generate(where: str, fields: Mapping[str, Any], *, evaluated: bool, allow_batch: bool,
                    max_tokens_pin: int | None) -> None:
    """Raise RunConfigError on the first field of one GenerateConfig off the pins.

    evaluated: True for plan.config and the evaluated model's events, which must carry
    reasoning_effort == REASONING_EFFORT (and max_tokens equal to a given pin)."""
    if evaluated and fields.get("reasoning_effort") != REASONING_EFFORT:
        raise RunConfigError(
            f"run config: {where} has reasoning_effort={fields.get('reasoning_effort')!r}, "
            f"not the pin {REASONING_EFFORT!r} (a --reasoning-effort flag replaces the Task config)"
        )
    other = sorted(set(fields) - set(RUN_CONFIG_OPERATIONAL) - set(_RUN_CONFIG_RULED))
    if other:
        named = ", ".join(f"{k}={fields[k]!r}" for k in other)
        raise RunConfigError(
            f"run config: {where} sets {named}; section 5 sets no sampling parameter and no seed, "
            f"and only {list(RUN_CONFIG_OPERATIONAL)} may be set besides reasoning_effort"
        )
    if "max_tokens" in fields:
        if max_tokens_pin is None or fields["max_tokens"] != max_tokens_pin:
            raise RunConfigError(
                f"run config: {where} sets max_tokens={fields['max_tokens']!r}; "
                f"the max_tokens pin is {max_tokens_pin!r}"
            )
    elif evaluated and max_tokens_pin is not None:
        raise RunConfigError(
            f"run config: {where} leaves max_tokens unset; the max_tokens pin is {max_tokens_pin!r}"
        )
    if fields.get("batch") not in (None, False) and not allow_batch:
        raise RunConfigError(f"run config: {where} sets batch={fields['batch']!r}; batch is not allowed for this run")
    if fields.get("extra_headers"):
        raise RunConfigError(
            f"run config: {where} sets extra_headers ({len(fields['extra_headers'])} header(s)); "
            f"request headers can replace the credential, so no roster run sets them"
        )


def check_run_config(log: Any, *, expected_samples: int | None = None, expected_epochs: int | None = 4,
                     allow_batch: bool = True, max_tokens_pin: int | None = None) -> dict:
    """Raise RunConfigError unless a finished EvalLog ran the section 5 configuration.

    CLI generate flags replace the Task's GenerateConfig without a warning and the run
    still ends in success, so this reads what the log says was sent.

    log: a full EvalLog (samples and events read, not header_only), from a real run
        or an offline mockllm run alike.
    expected_samples: when given, the run must hold exactly expected_samples *
        expected_epochs scored sample-epochs (no error, at least one score).
    expected_epochs: config.epochs (and task_args epochs, when recorded) must equal it;
        None skips the epoch check.
    allow_batch: whether GenerateConfig.batch may be set (interactive or batch runs).
    max_tokens_pin: the owner's max_tokens pin; None means max_tokens must be unset.

    Checks: status success; eval config limit and sample_id None; plan.config and every
    ModelEvent of the evaluated model (role None) carry reasoning_effort ==
    REASONING_EFFORT; max_tokens per the pin; batch per allow_batch; extra_headers
    empty on every ModelEvent; any other set GenerateConfig field outside
    RUN_CONFIG_OPERATIONAL refuses (every sampling parameter, reasoning_tokens, seed). EvalConfig
    fields such as retry_on_error and fail_on_error are not checked.

    Returns a summary of the effective values (never a header value)."""
    status = getattr(log, "status", None)
    if status != "success":
        raise RunConfigError(f"run config: log status is {status!r}, not 'success'")
    try:
        spec = log.eval
        eval_config = spec.config
        plan_config = log.plan.config
    except AttributeError as e:
        raise RunConfigError(f"run config: cannot read the log header ({e})") from e
    for name in ("limit", "sample_id"):
        value = getattr(eval_config, name, None)
        if value is not None:
            raise RunConfigError(
                f"run config: eval config sets {name}={value!r}; a pre-registered run covers the whole dataset"
            )
    if expected_epochs is not None:
        if eval_config.epochs != expected_epochs:
            raise RunConfigError(
                f"run config: eval config epochs={eval_config.epochs!r}, expected {expected_epochs} "
                f"(an --epochs flag replaces the Task epochs)"
            )
        task_epochs = (spec.task_args or {}).get("epochs")
        if task_epochs is not None and task_epochs != expected_epochs:
            raise RunConfigError(f"run config: task_args epochs={task_epochs!r}, expected {expected_epochs}")

    plan = _generate_fields(plan_config)
    _check_generate("plan.config", plan, evaluated=True, allow_batch=allow_batch, max_tokens_pin=max_tokens_pin)

    samples = log.samples
    if samples is None:
        raise RunConfigError("run config: the log has no samples; read it in full, not header_only")
    evaluated_events = 0
    for sample in samples:
        for event in sample.events or ():
            if getattr(event, "event", None) != "model":
                continue
            role = getattr(event, "role", None)
            where = f"ModelEvent of sample {sample.id!r} epoch {sample.epoch}" + (f" role {role!r}" if role else "")
            fields = _generate_fields(event.config)
            if role is None:
                evaluated_events += 1
                _check_generate(where, fields, evaluated=True, allow_batch=allow_batch,
                                max_tokens_pin=max_tokens_pin)
            elif fields.get("extra_headers"):
                raise RunConfigError(f"run config: {where} sets extra_headers; no roster run sets them")
    if evaluated_events == 0:
        raise RunConfigError("run config: no ModelEvent of the evaluated model; what the provider was sent is unknown")

    scored = {(s.id, s.epoch) for s in samples if s.error is None and s.scores}
    if expected_samples is not None:
        epochs = expected_epochs if expected_epochs is not None else (eval_config.epochs or 1)
        want = expected_samples * epochs
        if len(scored) != want:
            raise RunConfigError(
                f"run config: {len(scored)} scored sample-epochs, expected exactly {want} "
                f"({expected_samples} samples x {epochs} epochs)"
            )
    return {
        "status": status,
        "model": spec.model,
        "epochs": eval_config.epochs,
        "samples": len({s.id for s in samples}),
        "scored_sample_epochs": len(scored),
        "evaluated_model_events": evaluated_events,
        "reasoning_effort": plan.get("reasoning_effort"),
        "max_tokens": plan.get("max_tokens"),
        "batch": plan.get("batch"),
        "operational": {k: plan[k] for k in RUN_CONFIG_OPERATIONAL if k in plan},
        "retry_on_error": getattr(eval_config, "retry_on_error", None),
        "fail_on_error": getattr(eval_config, "fail_on_error", None),
    }


# --- Setup solver --------------------------------------------------------------------

EVALUATED_KEY = "evaluated_model"  # key of the evaluated model in the serving record


def roster_check(grader_step: int = 1):
    """Setup solver: resolve the evaluated model with get_model() and every required
    role with get_model(role=..., required=True), then validate_roster. A role bound
    to more than one model fails first: get_model(role=...) hands back only the first
    model of a list, so the others would go unchecked. On a real run
    it then applies the serving guard to the evaluated model and every role and
    records the checked endpoints as one info event in the sample transcript:
    InfoEvent(source="roster_check", data={"serving_endpoints": {"evaluated_model":
    {"model": ..., "base_url": ...}, "grader": {...}, "calibrator": {...}}}).
    Offline (all-mockllm) runs skip the guard and record nothing."""
    from inspect_ai.log import transcript
    from inspect_ai.model import get_model, model_roles
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="roster_check")
    def _roster_check():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            raw = model_roles()
            for role, value in raw.items():
                if isinstance(value, list):
                    raise RosterError(
                        f"role {role!r} is bound to more than one model ({len(value)}); the roster binds one"
                    )
            bound: dict[str, Any] = {r: None for r in raw}
            for role in REQUIRED_ROLES:
                try:
                    bound[role] = get_model(role=role, required=True)
                except Exception as e:  # PrerequisiteError when the role is unbound
                    raise RosterError(f"required judge role {role!r} unbound: {e}") from e
            for role in bound:
                if bound[role] is None:
                    bound[role] = get_model(role=role, required=True)
            evaluated = get_model()
            validate_roster(str(evaluated), {r: str(m) for r, m in bound.items()}, grader_step)
            if not _is_offline(str(evaluated)):  # validate_roster rejected any mixed run
                endpoints = serving_endpoints({EVALUATED_KEY: evaluated, **bound})
                transcript().info({"serving_endpoints": endpoints}, source=SERVING_EVENT_SOURCE)
            return state

        return solve

    return _roster_check()
