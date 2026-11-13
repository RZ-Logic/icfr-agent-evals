"""roster.check_run_config tests (review finding F1-02; pre-registration section 5
reasoning, sampling parameters and seed).

CLI generate flags replace the Task's GenerateConfig and the run still ends in
success, so the check reads the finished log. Each case runs the dev pilot subset on
mockllm with the override the CLI flag would set (eval() takes the same fields as
keyword arguments) and checks that the run is refused with the field named. The
clean pilot run (20 cases x 4 epochs) passes. The override runs use one epoch to
keep the suite fast (each is checked with expected_epochs=1), except the --epochs
case itself. Dev split only; keyless.
"""

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from icfr_agent_evals import baseline, roster, task as task_module
from icfr_agent_evals.roster import (
    GRADER_REASONING_EFFORT,
    REASONING_EFFORT,
    RunConfigError,
    calibrator_generate_config_args,
    check_run_config,
    grader_generate_config_args,
)

ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}
PILOT_CASES = 20
EPOCHS = 4


def _pilot(tmp_path_factory, name, model="mockllm/model", task_epochs=EPOCHS, **kwargs):
    log_dir = tmp_path_factory.mktemp(f"run-config-{name}")
    return inspect_eval(
        task_module.variance_commentary(split="dev", epochs=task_epochs, subset="pilot"),
        model=model, model_roles=ROLES, log_dir=str(log_dir), display="none", **kwargs,
    )[0]


# One run per CLI override the finding names (the flag each keyword stands for).
OVERRIDES = {
    "reasoning_effort_low": ({"reasoning_effort": "low"}, "reasoning_effort='low'"),     # --reasoning-effort low
    "temperature": ({"temperature": 0.7}, "temperature=0.7"),                             # --temperature 0.7
    "max_tokens": ({"max_tokens": 100}, "max_tokens=100"),                                # --max-tokens 100
    "top_p": ({"top_p": 0.9}, "top_p=0.9"),                                               # --top-p 0.9
    "seed": ({"seed": 7}, "seed=7"),                                                      # --seed 7
    "limit": ({"limit": 2}, "limit=2"),                                                   # --limit 2
    "epochs": ({"epochs": 2}, "epochs=2"),                                                # --epochs 2
    "extra_headers": ({"extra_headers": {"X-Review-Marker": "w2"}}, "extra_headers"),     # --extra-headers
}


def _task_epochs(name):
    return EPOCHS if name == "epochs" else 1


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    log = _pilot(tmp_path_factory, "clean")
    assert log.status == "success", log.error
    return log


@pytest.fixture(scope="module")
def override_runs(tmp_path_factory):
    runs = {name: _pilot(tmp_path_factory, name, task_epochs=_task_epochs(name), **kw)
            for name, (kw, _) in OVERRIDES.items()}
    for name, log in runs.items():
        assert log.status == "success", (name, log.error)  # the defect: each run "succeeds"
    return runs


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    return _pilot(tmp_path_factory, "batch", task_epochs=1, batch=True)  # --batch


@pytest.fixture(scope="module")
def operational_run(tmp_path_factory):
    return _pilot(tmp_path_factory, "operational", task_epochs=1, max_connections=5, max_retries=2, timeout=600,
                  retry_on_error=1, fail_on_error=0.1)


def test_error_is_a_runtime_error():
    assert issubclass(RunConfigError, RuntimeError)


def test_clean_pilot_run_passes_and_reports_the_effective_values(clean_run):
    summary = check_run_config(clean_run, expected_samples=PILOT_CASES)
    assert summary["status"] == "success"
    assert summary["reasoning_effort"] == REASONING_EFFORT
    assert summary["scored_sample_epochs"] == PILOT_CASES * EPOCHS == 80
    assert summary["samples"] == PILOT_CASES
    assert summary["epochs"] == EPOCHS
    assert summary["evaluated_model_events"] == 80
    assert summary["max_tokens"] is None and summary["batch"] is None


def test_clean_run_passes_with_the_defaults(clean_run):
    assert check_run_config(clean_run)["scored_sample_epochs"] == 80


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_each_override_is_refused_with_the_field_named(name, override_runs):
    # Without the check each of these runs ends in success with the roster metadata
    # still recording reasoning_effort high.
    _, needle = OVERRIDES[name]
    with pytest.raises(RunConfigError, match=needle.replace("(", r"\(").replace(".", r"\.")):
        check_run_config(override_runs[name], expected_samples=PILOT_CASES, expected_epochs=_task_epochs(name))


def _with_event_max_tokens(log, value, *, only_first=False):
    """A copy of a log whose evaluated-model events record max_tokens=value, the way Inspect
    records the value it sent to a Claude model (plan.config stays unset)."""
    log = log.model_copy(deep=True)
    for i, s in enumerate(log.samples):
        if only_first and i > 0:
            break
        for ev in s.events:
            if getattr(ev, "event", None) == "model" and ev.role is None:
                ev.config.max_tokens = value
    return log


def test_provider_default_max_tokens_in_events_passes_without_a_pin(clean_run):
    """Sonnet pilot leg 2026-11-06: Inspect records max_tokens=48000 in every evaluated event
    while plan.config leaves it unset; with no pin that is the run the spec describes."""
    log = _with_event_max_tokens(clean_run, roster.DEFAULT_MAX_TOKENS)
    summary = check_run_config(log, expected_samples=PILOT_CASES)
    assert summary["max_tokens"] is None
    assert summary["max_tokens_effective"] == roster.DEFAULT_MAX_TOKENS == 48000
    assert check_run_config(clean_run, expected_samples=PILOT_CASES)["max_tokens_effective"] is None
    # a pin equal to the default still needs the --max-tokens flag in plan.config
    with pytest.raises(RunConfigError, match="leaves max_tokens unset"):
        check_run_config(log, max_tokens_pin=roster.DEFAULT_MAX_TOKENS)


def test_any_other_event_max_tokens_refuses_without_a_pin(clean_run):
    log = _with_event_max_tokens(clean_run, 16000)
    with pytest.raises(RunConfigError, match="max_tokens=16000.*documented default 48000"):
        check_run_config(log, expected_samples=PILOT_CASES)


def test_mixed_event_max_tokens_refuses(clean_run):
    log = _with_event_max_tokens(clean_run, roster.DEFAULT_MAX_TOKENS, only_first=True)
    with pytest.raises(RunConfigError, match="more than one max_tokens value"):
        check_run_config(log, expected_samples=PILOT_CASES)


def test_max_tokens_passes_only_when_it_equals_the_owner_pin(override_runs, clean_run):
    log = override_runs["max_tokens"]
    assert check_run_config(log, expected_samples=PILOT_CASES, expected_epochs=1, max_tokens_pin=100)["max_tokens"] == 100
    with pytest.raises(RunConfigError, match="max_tokens=100"):
        check_run_config(log, expected_epochs=1, max_tokens_pin=200)
    with pytest.raises(RunConfigError, match="leaves max_tokens unset"):
        check_run_config(clean_run, max_tokens_pin=100)


def test_batch_passes_when_allowed_and_is_refused_otherwise(batch_run):
    assert batch_run.status == "success", batch_run.error
    assert check_run_config(batch_run, expected_samples=PILOT_CASES, expected_epochs=1)["batch"] is not None
    with pytest.raises(RunConfigError, match="batch"):
        check_run_config(batch_run, expected_epochs=1, allow_batch=False)


def test_operational_settings_pass(operational_run):
    summary = check_run_config(operational_run, expected_samples=PILOT_CASES, expected_epochs=1)
    assert summary["operational"] == {"max_connections": 5, "max_retries": 2, "timeout": 600}
    assert summary["retry_on_error"] == 1


def test_error_status_log_is_refused(tmp_path_factory):
    def boom(messages, tools, tool_choice, config) -> ModelOutput:
        raise RuntimeError("mock failure")

    log = _pilot(tmp_path_factory, "error", task_epochs=1, model=get_model("mockllm/model", custom_outputs=boom))
    assert log.status == "error"
    with pytest.raises(RunConfigError, match="status is 'error'"):
        check_run_config(log)


@pytest.mark.parametrize("rows, ok", [(80, True), (34, False), (79, False)])
def test_expected_samples_needs_exactly_samples_times_epochs(clean_run, rows, ok):
    log = clean_run.model_copy(update={"samples": clean_run.samples[:rows]})
    if ok:
        assert check_run_config(log, expected_samples=PILOT_CASES)["scored_sample_epochs"] == rows
    else:
        with pytest.raises(RunConfigError, match=f"{rows} scored sample-epochs, expected exactly 80"):
            check_run_config(log, expected_samples=PILOT_CASES)


def test_an_errored_sample_does_not_count_as_scored(clean_run):
    from inspect_ai.log import EvalError

    first = clean_run.samples[0].model_copy(update={"error": EvalError(message="x", traceback="", traceback_ansi="")})
    log = clean_run.model_copy(update={"samples": [first, *clean_run.samples[1:]]})
    with pytest.raises(RunConfigError, match="79 scored sample-epochs"):
        check_run_config(log, expected_samples=PILOT_CASES)


def test_header_only_log_is_refused(clean_run):
    with pytest.raises(RunConfigError, match="header_only"):
        check_run_config(clean_run.model_copy(update={"samples": None}))


def _with_plan(log, **fields):
    return log.model_copy(update={"plan": log.plan.model_copy(update={"config": GenerateConfig(**fields)})})


@pytest.mark.parametrize("field, value", [
    ("top_k", 5), ("reasoning_tokens", 1000), ("stop_seqs", ["x"]), ("frequency_penalty", 0.5),
    ("num_choices", 2), ("cache", True), ("system_message", "x"),
])
def test_any_other_generate_field_is_refused(clean_run, field, value):
    log = _with_plan(clean_run, reasoning_effort=REASONING_EFFORT, **{field: value})
    with pytest.raises(RunConfigError, match=field):
        check_run_config(log)


def test_plan_config_without_the_pinned_effort_is_refused(clean_run):
    with pytest.raises(RunConfigError, match="reasoning_effort=None"):
        check_run_config(_with_plan(clean_run))


def _with_event_config(log, sample_index, **update):
    sample = log.samples[sample_index]
    events = [e.model_copy(update={"config": e.config.model_copy(update=update)}) if e.event == "model" else e
              for e in sample.events]
    samples = list(log.samples)
    samples[sample_index] = sample.model_copy(update={"events": events})
    return log.model_copy(update={"samples": samples})


def test_one_model_event_off_the_pin_is_refused_even_when_plan_config_is_clean(clean_run):
    # What the provider was sent wins over the plan: one event of one sample is enough.
    log = _with_event_config(clean_run, 7, reasoning_effort="low")
    with pytest.raises(RunConfigError, match="ModelEvent of sample .* reasoning_effort='low'"):
        check_run_config(log)


def _with_role_event(log, role, config, sample_index=0):
    """A copy of a log with one extra ModelEvent of the given role and config in one
    sample, the way Inspect records a call to a model bound to that role."""
    sample = log.samples[sample_index]
    (model_event,) = [e for e in sample.events if e.event == "model"]
    role_event = model_event.model_copy(update={"role": role, "config": config})
    samples = list(log.samples)
    samples[sample_index] = sample.model_copy(update={"events": [*sample.events, role_event]})
    return log.model_copy(update={"samples": samples})


@pytest.mark.parametrize("role, config", [
    ("calibrator", GenerateConfig(reasoning_effort=REASONING_EFFORT, extra_headers={"X": "y"})),
    ("grader", GenerateConfig(reasoning_effort=GRADER_REASONING_EFFORT, extra_headers={"X": "y"})),
])
def test_a_role_event_with_extra_headers_is_refused(clean_run, role, config):
    with pytest.raises(RunConfigError, match=f"role '{role}' sets extra_headers"):
        check_run_config(_with_role_event(clean_run, role, config))


# --- Grader reasoning_effort (section 13 grader configuration row; D-T1, D-R12) ------


def test_a_log_without_grader_events_passes_and_counts_none(clean_run):
    # No Phase 1 scorer calls the grader yet, so the dev pilot log holds no grader event.
    assert check_run_config(clean_run, expected_samples=PILOT_CASES)["grader_model_events"] == 0


@pytest.mark.parametrize("step", [1, 2, 3])
def test_a_grader_event_with_the_pinned_config_passes(clean_run, step):
    log = _with_role_event(clean_run, "grader", GenerateConfig(**grader_generate_config_args(step)))
    assert check_run_config(log, expected_samples=PILOT_CASES)["grader_model_events"] == 1


@pytest.mark.parametrize("effort", [None, "none", "low", "medium", "xhigh"])
def test_a_grader_event_off_the_effort_pin_is_refused(clean_run, effort):
    log = _with_role_event(clean_run, "grader", GenerateConfig(reasoning_effort=effort))
    with pytest.raises(RunConfigError, match=f"role 'grader' has reasoning_effort={effort!r}, not the pin 'high'"):
        check_run_config(log)


def test_one_grader_event_off_the_pin_among_clean_ones_is_refused(clean_run):
    log = _with_role_event(clean_run, "grader", GenerateConfig(**grader_generate_config_args(1)), sample_index=0)
    log = _with_role_event(log, "grader", GenerateConfig(reasoning_effort="low"), sample_index=5)
    with pytest.raises(RunConfigError, match="sample .* role 'grader' has reasoning_effort='low'"):
        check_run_config(log)


@pytest.mark.parametrize("field, value", [("temperature", 0.0), ("top_p", 0.9), ("seed", 7), ("reasoning_tokens", 1000)])
def test_a_grader_event_with_a_sampling_field_is_refused(clean_run, field, value):
    # D-T1: temperature unset on every grader; no sampling parameter or seed anywhere.
    log = _with_role_event(clean_run, "grader",
                           GenerateConfig(reasoning_effort=GRADER_REASONING_EFFORT, **{field: value}))
    with pytest.raises(RunConfigError, match=f"role 'grader' sets {field}="):
        check_run_config(log)


def test_a_grader_event_with_max_tokens_is_refused(clean_run):
    log = _with_role_event(clean_run, "grader",
                           GenerateConfig(reasoning_effort=GRADER_REASONING_EFFORT, max_tokens=4000))
    with pytest.raises(RunConfigError, match="role 'grader' sets max_tokens=4000; no grader max_tokens is pinned"):
        check_run_config(log)


def test_a_grader_batch_event_follows_allow_batch(clean_run):
    log = _with_role_event(clean_run, "grader", GenerateConfig(**grader_generate_config_args(2)))
    assert check_run_config(log)["grader_model_events"] == 1
    with pytest.raises(RunConfigError, match="role 'grader' sets batch=True"):
        check_run_config(log, allow_batch=False)


def test_the_calibrator_role_is_held_to_the_worker_effort_pin_not_the_grader_pin(clean_run):
    # D-R15 (2026-11-13): the calibrator (an Anthropic model) carries REASONING_EFFORT and
    # counts as a calibrator event, never as a grader event.
    log = _with_role_event(clean_run, "calibrator", GenerateConfig(reasoning_effort=REASONING_EFFORT))
    out = check_run_config(log)
    assert out["grader_model_events"] == 0 and out["calibrator_model_events"] == 1


def test_a_log_with_no_evaluated_model_event_is_refused(clean_run):
    samples = [s.model_copy(update={"events": [e for e in s.events if e.event != "model"]}) for s in clean_run.samples]
    with pytest.raises(RunConfigError, match="no ModelEvent of the evaluated model"):
        check_run_config(clean_run.model_copy(update={"samples": samples}))


def test_template_baseline_log_is_out_of_scope(tmp_path_factory):
    # The template baseline sends nothing to a provider, so it carries no reasoning
    # effort; the readers must not apply this check to the template log.
    log = inspect_eval(baseline.variance_commentary_template_baseline(split="dev", epochs=1, subset="pilot"),
                       log_dir=str(tmp_path_factory.mktemp("run-config-template")), display="none")[0]
    assert log.status == "success", log.error
    with pytest.raises(RunConfigError):
        check_run_config(log, expected_samples=PILOT_CASES, expected_epochs=1)


def test_allowlist_names_only_transport_fields():
    assert set(roster.RUN_CONFIG_OPERATIONAL) == {
        "max_retries", "timeout", "attempt_timeout", "stream_idle_timeout", "max_connections", "adaptive_connections",
    }
    assert not hasattr(roster, "RUN_CONFIG_FORBIDDEN")  # every field off the allowlist refuses


# --- Calibrator reasoning_effort (section 13 effort row; owner decision D-R15) -----------


def test_a_log_without_calibrator_events_counts_none(clean_run):
    assert check_run_config(clean_run, expected_samples=PILOT_CASES)["calibrator_model_events"] == 0


def test_a_calibrator_event_with_the_pinned_config_passes(clean_run):
    log = _with_role_event(clean_run, "calibrator", GenerateConfig(**calibrator_generate_config_args()))
    assert check_run_config(log, expected_samples=PILOT_CASES)["calibrator_model_events"] == 1


@pytest.mark.parametrize("effort", [None, "low", "medium", "max"])
def test_a_calibrator_event_off_the_effort_pin_is_refused(clean_run, effort):
    log = _with_role_event(clean_run, "calibrator", GenerateConfig(reasoning_effort=effort))
    with pytest.raises(RunConfigError, match=f"role 'calibrator' has reasoning_effort={effort!r}, not the pin 'high'"):
        check_run_config(log, expected_samples=PILOT_CASES)


def test_a_calibrator_event_with_a_sampling_field_is_refused(clean_run):
    log = _with_role_event(clean_run, "calibrator",
                           GenerateConfig(**calibrator_generate_config_args(), temperature=0.0))
    with pytest.raises(RunConfigError, match="temperature"):
        check_run_config(log, expected_samples=PILOT_CASES)


def test_calibrator_args_carry_the_worker_effort_pin():
    assert calibrator_generate_config_args() == {"reasoning_effort": REASONING_EFFORT}
