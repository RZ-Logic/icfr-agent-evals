"""Roster setup-check tests (pre-registration section 5 roles and roster, section 13).

One test per violation the setup check must reject, plus the offline mockllm mode
the smoke and unit tests run in.
"""

import re

import pytest

from icfr_agent_evals.roster import (
    BASELINES,
    CALIBRATOR,
    EVALUATED_MODELS,
    GRADER,
    GRADER_ESCALATION,
    NOTICE_POLICIES,
    RETIREMENT_FLOORS,
    RETIREMENT_FLOORS_RECORDED,
    SCOUT_SCANNER,
    SERVING_PLATFORMS,
    WORKER,
    RosterError,
    calibrator_may_sample,
    retirement_metadata,
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


@pytest.mark.parametrize("model", EVALUATED_MODELS)
@pytest.mark.parametrize("step", [1, 2])
def test_grader_equal_to_evaluated_model_is_rejected(model, step):
    with pytest.raises(RosterError, match="equals the evaluated model"):
        validate_roster(model, {**PINNED, "grader": model}, grader_step=step)


@pytest.mark.parametrize("role", ["grader", "calibrator"])
@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-6", "openai/gpt-6", "openai/gpt-6-luna-latest"])
def test_judge_role_bound_off_the_roster_is_rejected(role, model):
    with pytest.raises(RosterError, match="not its pin"):
        validate_roster(WORKER, {**PINNED, role: model})


@pytest.mark.parametrize(
    "evaluated,roles,needle",
    [
        ("anthropic/bedrock/claude-sonnet-5-5", PINNED, "not on the roster"),
        ("anthropic/vertex/claude-sonnet-5-5", PINNED, "not on the roster"),
        (WORKER, {**PINNED, "grader": "openai/azure/gpt-6-luna"}, "not its pin"),
        (WORKER, {**PINNED, "calibrator": "anthropic/bedrock/claude-opus-5-5"}, "not its pin"),
    ],
)
def test_partner_platform_service_prefix_is_rejected(evaluated, roles, needle):
    # Inspect routes anthropic/bedrock/, anthropic/vertex/ and openai/azure/ to a partner
    # platform with its own retirement schedule; the exact-pin match keeps them out.
    with pytest.raises(RosterError, match=needle):
        validate_roster(evaluated, roles)


def test_none_model_is_rejected_by_the_main_roster_check():
    # The template baseline's none/none exception lives only in baseline.baseline_check,
    # on its own Task; the main task's roster check never accepts it.
    with pytest.raises(RosterError, match="not on the roster"):
        validate_roster("none/none", PINNED)


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
    assert meta["serving_platforms"] == SERVING_PLATFORMS


# --- Reperformance window and serving platform (section 5, section 13) --------------------

def test_retirement_floors_cover_exactly_the_anthropic_roster_models():
    anthropic = {m for m in (*EVALUATED_MODELS, CALIBRATOR) if m.startswith("anthropic/")}
    assert set(RETIREMENT_FLOORS) == anthropic
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in RETIREMENT_FLOORS.values())
    assert min(RETIREMENT_FLOORS.values()) == RETIREMENT_FLOORS["anthropic/claude-sonnet-5"] == "2027-06-30"


def test_retirement_floors_recorded_date_is_iso_and_not_before_first_read():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", RETIREMENT_FLOORS_RECORDED)
    assert RETIREMENT_FLOORS_RECORDED >= "2026-10-08"


def test_every_roster_provider_has_a_serving_platform_and_notice_policy():
    providers = {m.split("/", 1)[0] for m in (*EVALUATED_MODELS, GRADER, GRADER_ESCALATION, CALIBRATOR, SCOUT_SCANNER)}
    assert providers == set(SERVING_PLATFORMS) == set(NOTICE_POLICIES) == {"anthropic", "openai"}
    assert "first-party" in SERVING_PLATFORMS["anthropic"] and "Claude API" in SERVING_PLATFORMS["anthropic"]
    assert "first-party" in SERVING_PLATFORMS["openai"] and "OpenAI API" in SERVING_PLATFORMS["openai"]
    assert "60 days" in NOTICE_POLICIES["anthropic"]
    assert "6 months" in NOTICE_POLICIES["openai"] and "generally available" in NOTICE_POLICIES["openai"]


def test_retirement_metadata_records_floors_policies_and_platforms():
    meta = retirement_metadata()
    assert meta["recorded"] == RETIREMENT_FLOORS_RECORDED
    assert meta["platform"] == SERVING_PLATFORMS["anthropic"]
    assert meta["floors"] == RETIREMENT_FLOORS
    assert meta["notice_policies"] == NOTICE_POLICIES
    assert meta["serving_platforms"] == SERVING_PLATFORMS
    meta["floors"]["anthropic/claude-sonnet-5"] = "changed"
    assert RETIREMENT_FLOORS["anthropic/claude-sonnet-5"] == "2027-06-30"  # a copy, not the constant
