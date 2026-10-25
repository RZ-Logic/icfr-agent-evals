"""The frozen Phase 1 model roster and the setup check that enforces it.

Pre-registration section 5: the worker is chosen by --model and checked against
the roster at setup, never bound on the Task (a Task-level model silently
overrides --model). Judge roles are required roles. Under the pinned inspect-ai,
get_model(role=...) for an unbound role returns the model being evaluated unless
required=True is passed with no default, so the setup solver always passes it.
An unbound role, a role bound away from its pin, a grader equal to the evaluated
model, or an evaluated model off the roster is a run error, never a fallback.

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

GRADER = "openai/gpt-6-luna"             # alias-only, accepted; resolved string logged per response
GRADER_ESCALATION = "openai/gpt-6.1-sol"  # step 2 of the judge remedy order (section 7)
CALIBRATOR = "anthropic/claude-opus-5-5"  # never samples Opus 5.5 baseline generations
SCOUT_SCANNER = "openai/gpt-6-luna"       # named in each scan config; not an eval role

GRADER_ROLE = "grader"
CALIBRATOR_ROLE = "calibrator"
REQUIRED_ROLES: tuple[str, ...] = (GRADER_ROLE, CALIBRATOR_ROLE)
GRADER_STEPS: tuple[int, ...] = (1, 2)

# reasoning_effort is pinned high on every Anthropic roster model in every role.
REASONING_EFFORT = "high"

OFFLINE_PROVIDER = "mockllm"

# --- Reperformance window (section 5) ------------------------------------------------

RETIREMENT_FLOORS_RECORDED = "2026-10-08"   # on the Claude API
RETIREMENT_FLOORS: dict[str, str] = {
    "anthropic/claude-sonnet-5": "2027-06-30",
    "anthropic/claude-opus-5-5": "2027-09-22",
    "anthropic/claude-sonnet-5-5": "2027-09-28",
    "anthropic/claude-haiku-5-5": "2027-10-07",
}
NOTICE_POLICIES: dict[str, str] = {
    "anthropic": "tentative retirement no sooner than the listed floor, then at least 60 days' notice",
    "openai": "generally available models: at least 6 months' notice",
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
    }


def retirement_metadata() -> dict:
    return {
        "recorded": RETIREMENT_FLOORS_RECORDED,
        "platform": "Claude API",
        "floors": dict(RETIREMENT_FLOORS),
        "notice_policies": dict(NOTICE_POLICIES),
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
