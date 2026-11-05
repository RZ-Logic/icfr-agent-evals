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
request credential while the host stays first-party. Each sample's transcript
records the endpoints it checked. This is what makes the fixed SERVING_PLATFORMS claim true.

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

    Returns the endpoint with one trailing slash stripped.
    """
    who = f"{model!r}" if model else f"the {provider!r} model"
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


def model_serving(model: Any) -> dict[str, str]:
    """Check one resolved Inspect Model with validate_serving and return what was
    checked: {"model": "provider/name", "base_url": the client's base URL as read}.

    The endpoint is read as str(model.api.client.base_url), the SDK's URL object
    (httpx2.URL in anthropic 1.12.1 and openai 3.27.0). That is a private Inspect
    attribute, safe under the exact pin; a missing attribute or an unexpected type
    fails closed."""
    url_types = _url_types()
    try:
        name = str(model)
        client = model.api.client
        url = client.base_url
        explicit = [model.explicit_base_url, model.api.base_url]
        model_args = model.model_args
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
    for explicit_url in explicit:
        validate_serving(provider, observed, explicit_url, model_args, auth_token_set, model=name,
                         custom_headers=headers)
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
