"""Roster setup-check tests (pre-registration section 5 roles and roster, section 13).

One test per violation the setup check must reject, plus the offline mockllm mode
the smoke and unit tests run in.
"""

import pytest

from icfr_agent_evals.roster import (
    BASELINES,
    CALIBRATOR,
    EVALUATED_MODELS,
    GRADER,
    GRADER_ESCALATION,
    RETIREMENT_FLOORS,
    WORKER,
    RosterError,
    calibrator_may_sample,
    roster_metadata,
    validate_roster,
)

PINNED = {"grader": GRADER, "calibrator": CALIBRATOR}
MOCK = "mockllm/model"


def test_pins_match_the_pre_registration():
    assert WORKER == "anthropic/claude-sonnet-5-5"
    assert BASELINES == ("anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5", "anthropic/claude-opus-5-5")
    assert GRADER == "openai/gpt-6-luna" and GRADER_ESCALATION == "openai/gpt-6.1-sol"
    assert CALIBRATOR == "anthropic/claude-opus-5-5"
    assert RETIREMENT_FLOORS == {
        "anthropic/claude-sonnet-5": "2027-06-30",
        "anthropic/claude-opus-5-5": "2027-09-22",
        "anthropic/claude-sonnet-5-5": "2027-09-28",
        "anthropic/claude-haiku-5-5": "2027-10-07",
    }


@pytest.mark.parametrize("model", EVALUATED_MODELS)
def test_every_roster_model_with_pinned_roles_passes(model):
    validate_roster(model, PINNED)


def test_evaluated_model_off_the_roster_is_rejected():
    with pytest.raises(RosterError, match="not on the roster"):
        validate_roster("anthropic/claude-sonnet-4-6", PINNED)


def test_floating_alias_is_rejected():
    with pytest.raises(RosterError, match="not on the roster"):
        validate_roster("anthropic/claude-sonnet-latest", PINNED)


@pytest.mark.parametrize("role", ["grader", "calibrator"])
def test_unbound_required_role_is_rejected(role):
    roles = {**PINNED, role: None}
    with pytest.raises(RosterError, match="unbound"):
        validate_roster(WORKER, roles)
    with pytest.raises(RosterError, match="unbound"):
        validate_roster(WORKER, {k: v for k, v in PINNED.items() if k != role})


def test_grader_bound_away_from_its_pin_is_rejected():
    with pytest.raises(RosterError, match="not its pin"):
        validate_roster(WORKER, {**PINNED, "grader": "openai/gpt-6"})


def test_escalation_grader_only_at_step_two():
    with pytest.raises(RosterError, match="not its pin"):
        validate_roster(WORKER, {**PINNED, "grader": GRADER_ESCALATION}, grader_step=1)
    validate_roster(WORKER, {**PINNED, "grader": GRADER_ESCALATION}, grader_step=2)
    with pytest.raises(RosterError, match="not its pin"):
        validate_roster(WORKER, PINNED, grader_step=2)


def test_calibrator_bound_away_from_its_pin_is_rejected():
    with pytest.raises(RosterError, match="not its pin"):
        validate_roster(WORKER, {**PINNED, "calibrator": "anthropic/claude-sonnet-5-5"})


def test_grader_equal_to_evaluated_model_is_rejected():
    with pytest.raises(RosterError, match="equals the evaluated model"):
        validate_roster(WORKER, {**PINNED, "grader": WORKER})


def test_role_not_on_the_roster_is_rejected():
    with pytest.raises(RosterError, match="not on the frozen roster"):
        validate_roster(WORKER, {**PINNED, "scanner": GRADER})


@pytest.mark.parametrize("step", [0, 3, "2"])
def test_unknown_grader_step_is_rejected(step):
    with pytest.raises(RosterError, match="grader_step"):
        validate_roster(WORKER, PINNED, grader_step=step)


def test_offline_mode_accepts_an_all_mock_run():
    validate_roster(MOCK, {"grader": MOCK, "calibrator": MOCK})
    validate_roster(MOCK, {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"})


@pytest.mark.parametrize(
    "evaluated,roles",
    [
        (MOCK, PINNED),
        (MOCK, {"grader": MOCK, "calibrator": CALIBRATOR}),
        (WORKER, {"grader": MOCK, "calibrator": CALIBRATOR}),
        (WORKER, {"grader": MOCK, "calibrator": MOCK}),
        ("none/none", {"grader": MOCK, "calibrator": MOCK}),
    ],
)
def test_mixed_real_and_mock_run_is_rejected(evaluated, roles):
    with pytest.raises(RosterError, match="mixing real and mockllm"):
        validate_roster(evaluated, roles)


def test_offline_mode_still_requires_bound_roles():
    with pytest.raises(RosterError, match="unbound"):
        validate_roster(MOCK, {"grader": MOCK})


def test_calibrator_never_samples_opus_baseline():
    assert not calibrator_may_sample("anthropic/claude-opus-5-5")
    assert calibrator_may_sample(WORKER)


def test_roster_metadata_records_pins_and_effort():
    meta = roster_metadata()
    assert meta["roles"] == PINNED and meta["reasoning_effort"] == "high"
    assert roster_metadata(2)["roles"]["grader"] == GRADER_ESCALATION
