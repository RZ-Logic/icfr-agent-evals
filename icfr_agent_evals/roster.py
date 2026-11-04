"""The frozen Phase 1 model roster and the setup check that enforces it.

Section 5 of the pre-registration fixes this roster. The evaluated model comes
from --model and is checked here at setup. It is never set on the Task, since a
model set there would take precedence over --model without a warning. The grader
and calibrator roles must both be bound. In the pinned inspect-ai, a get_model call
for a role nobody bound quietly hands back the evaluated model, so the setup solver
always asks with required=True. Each of these stops the run instead of falling
back: a role left unbound, a role bound to something other than its pin, a grader
that is the evaluated model, or an evaluated model outside the roster.

Offline mode: the mock smoke and the unit tests run on Inspect's mockllm provider.
A run is offline only when the evaluated model and every role are mockllm; a run
that mixes real and mock models is rejected.

validate_roster is pure (no Inspect import) so each violation has a unit test;
roster_check is the setup solver that resolves the models and calls it.
"""

from __future__ import annotations

from typing import Mapping

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
NOTICE_POLICIES: dict[str, str] = {
    "anthropic": (
        "not retired before the listed floor; at least 60 days of notice before retirement "
        "for publicly released models, sent to customers with active deployments"
    ),
    "openai": (
        "at least 6 months of notice before retirement for a generally available model, "
        "unless safety or compliance concerns require a faster timeline"
    ),
}

# Serving platform per provider prefix (section 5: partner platforms set their own
# retirement schedules, so the run config names the platform). The pins name no
# service prefix (such as anthropic/bedrock/ or openai/azure/), and validate_roster
# rejects any name that differs from its pin, so every roster model is served by its
# vendor's first-party API.
SERVING_PLATFORMS: dict[str, str] = {
    "anthropic": "Claude API (Anthropic first-party)",
    "openai": "OpenAI API (OpenAI first-party)",
}


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

def roster_check(grader_step: int = 1):
    """Setup solver: resolve the evaluated model with get_model() and every required
    role with get_model(role=..., required=True), then validate_roster."""
    from inspect_ai.model import get_model, model_roles
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="roster_check")
    def _roster_check():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            roles: dict[str, str | None] = {r: None for r in model_roles()}
            for role in REQUIRED_ROLES:
                try:
                    roles[role] = str(get_model(role=role, required=True))
                except Exception as e:  # PrerequisiteError when the role is unbound
                    raise RosterError(f"required judge role {role!r} unbound: {e}") from e
            for role in roles:
                if roles[role] is None:
                    roles[role] = str(get_model(role=role, required=True))
            validate_roster(str(get_model()), roles, grader_step)
            return state

        return solve

    return _roster_check()
