"""Serving guard tests (pre-registration section 5 serving platform, section 13 roster check).

A real run must send every request to its vendor's first-party API. The pure
validate_serving cases cover each rule; the adapter cases build real Inspect
models with a dummy key and read their clients, with the base-URL variables set,
blank or unset. No test ever calls generate, so no request is sent. HOME,
USERPROFILE, APPDATA and ANTHROPIC_CONFIG_DIR point at tmp_path, so the anthropic
SDK never reads the real home folder.
"""

import asyncio
import contextvars
import importlib
import os

import httpx2
import pytest
from inspect_ai._cli.util import parse_model_role_cli_args
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model import _model as inspect_model_module

from icfr_agent_evals import roster
from icfr_agent_evals.roster import (
    CALIBRATOR,
    FIRST_PARTY_BASE_URLS,
    GRADER,
    GRADER_STEP2,
    GRADER_STEP3,
    SCOUT_SCANNER,
    WORKER,
    RosterError,
    model_serving,
    roster_check,
    serving_endpoints,
    validate_serving,
)
from icfr_agent_evals.scanners._model import ScannerModelError, scanner_serving

ANTHROPIC = "https://api.anthropic.com"
OPENAI = "https://api.openai.com/v1"
URL_ENV = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL")
SCAN_ENV = ("SCOUT_SCAN_MODEL_BASE_URL", "INSPECT_EVAL_SCAN_MODEL_BASE_URL")
HEADER_ENV = ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS")
EXTRA_ENV = "INSPECT_EVAL_EXTRA_HEADERS"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE", "APPDATA", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    for name in (*URL_ENV, *SCAN_ENV, *HEADER_ENV, EXTRA_ENV, "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def _model(name: str, **kw):
    return get_model(name, api_key="test", memoize=False, **kw)


# --- Pure rule ---------------------------------------------------------------------------

def test_allowed_endpoints_are_pinned():
    assert FIRST_PARTY_BASE_URLS == {"anthropic": ANTHROPIC, "openai": OPENAI}


@pytest.mark.parametrize("provider, url", [("anthropic", ANTHROPIC), ("openai", OPENAI)])
@pytest.mark.parametrize("suffix", ["", "/"])
def test_first_party_endpoint_passes_with_or_without_trailing_slash(provider, url, suffix):
    assert validate_serving(provider, url + suffix) == url
    assert validate_serving(provider, url + suffix, explicit_base_url=url + suffix) == url


@pytest.mark.parametrize("provider, bad", [
    ("anthropic", "http://localhost:4000"),                 # proxy
    ("anthropic", "https://litellm.example.com"),           # proxy
    ("anthropic", "http://api.anthropic.com"),              # http://
    ("anthropic", "https://api.anthropic.com/v1"),          # different path
    ("anthropic", "https://api.anthropic.com.evil.example"),  # lookalike host
    ("anthropic", "https://api.anthropic.com//"),           # only one slash is stripped
    ("openai", "http://localhost:4000"),
    ("openai", "http://api.openai.com/v1"),
    ("openai", "https://api.openai.com"),                   # different path
    ("openai", "https://api.openai.com/v2"),
    ("openai", "https://api.openai.com.evil.example/v1"),
    ("openai", ANTHROPIC),                                  # the other vendor's endpoint
])
def test_any_other_endpoint_fails(provider, bad):
    with pytest.raises(RosterError, match="not the first-party endpoint"):
        validate_serving(provider, bad)


@pytest.mark.parametrize("blank", ["", " "])
def test_blank_endpoint_fails(blank):
    with pytest.raises(RosterError, match="blank"):
        validate_serving("anthropic", blank)


@pytest.mark.parametrize("bad", [None, 42, httpx2.URL(ANTHROPIC)])
def test_non_string_endpoint_fails(bad):
    with pytest.raises(RosterError, match="not a string"):
        validate_serving("anthropic", bad)


@pytest.mark.parametrize("explicit", ["http://localhost:4000", "", "https://api.anthropic.com.evil.example", 7])
def test_explicit_override_fails_even_when_the_client_looks_right(explicit):
    with pytest.raises(RosterError, match="explicit base URL override"):
        validate_serving("anthropic", ANTHROPIC, explicit_base_url=explicit)


@pytest.mark.parametrize("key", [
    "base_url", "http_client", "profile", "config", "credentials", "provider",
    "workload_identity", "data_residency", "websocket_base_url", "default_headers",
])
def test_any_model_args_fail(key):
    with pytest.raises(RosterError, match="model_args"):
        validate_serving("openai", OPENAI, model_args={key: "x"})


def test_non_mapping_model_args_fail():
    with pytest.raises(RosterError, match="not a mapping"):
        validate_serving("openai", OPENAI, model_args=["base_url"])


def test_auth_token_fails_for_anthropic():
    with pytest.raises(RosterError, match="ANTHROPIC_AUTH_TOKEN"):
        validate_serving("anthropic", ANTHROPIC, auth_token_set=True)


@pytest.mark.parametrize("provider, url", [("anthropic", ANTHROPIC), ("openai", OPENAI)])
@pytest.mark.parametrize("var", HEADER_ENV)
def test_custom_headers_fail_for_either_provider(provider, url, var):
    with pytest.raises(RosterError, match=var):
        validate_serving(provider, url, custom_headers=[var])


def test_unknown_provider_fails():
    with pytest.raises(RosterError, match="no first-party endpoint"):
        validate_serving("mockllm", "")


# --- Adapter on real Inspect models (dummy key, no request) -------------------------------

@pytest.mark.parametrize("name, url", [(WORKER, ANTHROPIC), (GRADER, OPENAI)])
def test_clean_env_resolves_first_party(name, url):
    record = model_serving(_model(name))
    assert record["model"] == name
    assert record["base_url"].rstrip("/") == url


@pytest.mark.parametrize("var, name", [
    ("ANTHROPIC_BASE_URL", WORKER),
    ("OPENAI_BASE_URL", GRADER),
    ("INSPECT_EVAL_MODEL_BASE_URL", WORKER),
    ("INSPECT_EVAL_MODEL_BASE_URL", GRADER),
])
def test_base_url_variable_set_to_a_proxy_fails(var, name, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    m = _model(name)
    assert str(m) == name  # the name alone hides the reroute
    with pytest.raises(RosterError, match="not the first-party endpoint"):
        model_serving(m)


@pytest.mark.parametrize("var, name", [
    ("ANTHROPIC_BASE_URL", WORKER),
    ("OPENAI_BASE_URL", GRADER),
    ("INSPECT_EVAL_MODEL_BASE_URL", WORKER),
    ("INSPECT_EVAL_MODEL_BASE_URL", GRADER),
])
def test_blank_base_url_variable_fails(var, name, monkeypatch):
    monkeypatch.setenv(var, "")
    with pytest.raises(RosterError, match="blank"):
        model_serving(_model(name))


@pytest.mark.parametrize("var, name, url", [
    ("ANTHROPIC_BASE_URL", WORKER, ANTHROPIC + "/"),
    ("OPENAI_BASE_URL", GRADER, OPENAI),
])
def test_base_url_variable_set_to_the_first_party_endpoint_passes(var, name, url, monkeypatch):
    monkeypatch.setenv(var, url)
    model_serving(_model(name))


def test_explicit_base_url_fails(monkeypatch):
    with pytest.raises(RosterError, match="explicit base URL override"):
        model_serving(_model(WORKER, base_url="https://proxy.example"))


def test_model_args_on_a_real_model_fail():
    with pytest.raises(RosterError, match="model_args"):
        model_serving(_model(GRADER, default_headers={"x": "y"}))


def test_role_spec_with_model_args_base_url_fails():
    roles = parse_model_role_cli_args(
        ("grader={model: openai/gpt-6-luna, model_args: {base_url: 'https://proxy.example/v1', api_key: test}}",)
    )
    grader = roles["grader"]
    assert str(grader) == GRADER  # validate_roster alone accepts it
    with pytest.raises(RosterError, match="explicit base URL override"):
        model_serving(grader)


def test_auth_token_variable_fails(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    with pytest.raises(RosterError, match="ANTHROPIC_AUTH_TOKEN"):
        model_serving(_model(WORKER))


def test_blank_auth_token_variable_is_harmless(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    model_serving(_model(WORKER))


@pytest.mark.parametrize("var, header", [
    ("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: test"),
    ("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer test"),
])
@pytest.mark.parametrize("name", [WORKER, GRADER])
def test_custom_headers_variable_fails(var, header, name, monkeypatch):
    monkeypatch.setenv(var, header)
    m = _model(name)
    assert str(m) == name and str(m.api.client.base_url).rstrip("/") in (ANTHROPIC, OPENAI)  # host unchanged
    with pytest.raises(RosterError, match=var):
        model_serving(m)


@pytest.mark.parametrize("var", HEADER_ENV)
@pytest.mark.parametrize("name", [WORKER, GRADER])
def test_blank_custom_headers_variable_passes(var, name, monkeypatch):
    monkeypatch.setenv(var, "")
    model_serving(_model(name))


class _Fake:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __str__(self):
        return self.__dict__.get("name", "anthropic/claude-sonnet-5-5")


@pytest.mark.parametrize("fake", [
    _Fake(),                                                            # no api
    _Fake(api=_Fake()),                                                 # no client
    _Fake(api=_Fake(client=_Fake(), base_url=None), explicit_base_url=None, model_args={}),  # no base_url
])
def test_unreadable_model_fails_closed(fake):
    with pytest.raises(RosterError, match="cannot read"):
        model_serving(fake)


def test_non_url_client_endpoint_fails_closed():
    fake = _Fake(api=_Fake(client=_Fake(base_url=ANTHROPIC), base_url=None), explicit_base_url=None, model_args={})
    with pytest.raises(RosterError, match="not an httpx2 or httpx URL"):
        model_serving(fake)


def test_serving_endpoints_records_each_model():
    record = serving_endpoints({"evaluated_model": _model(WORKER), "grader": _model(GRADER),
                                "calibrator": _model(CALIBRATOR)})
    assert set(record) == {"evaluated_model", "grader", "calibrator"}
    assert record["grader"] == {"model": GRADER, "base_url": "https://api.openai.com/v1/"}
    assert record["evaluated_model"] == {"model": WORKER, "base_url": ANTHROPIC}


# --- The setup solver ---------------------------------------------------------------------

def _run_roster_check(monkeypatch, evaluated, roles: dict, grader_step: int = 1):
    """Run the roster_check solve function with get_model and model_roles stubbed to
    hand back prebuilt models; returns the info events it recorded."""
    import inspect_ai.model as inspect_model

    def fake_get_model(model=None, *, role=None, required=False, **kw):
        if role is not None:
            if role not in roles:
                raise RuntimeError(f"role {role} unbound")
            return roles[role]
        return evaluated

    monkeypatch.setattr(inspect_model, "get_model", fake_get_model)
    monkeypatch.setattr(inspect_model, "model_roles", lambda: dict(roles))
    t = Transcript()
    init_transcript(t)
    solver = roster_check(grader_step=grader_step)
    sentinel = object()
    assert asyncio.run(solver(sentinel, None)) is sentinel
    return [e for e in t.events if e.event == "info"]


def test_roster_check_records_endpoints_once_on_a_real_run(monkeypatch):
    events = _run_roster_check(monkeypatch, _model(WORKER),
                               {"grader": _model(GRADER), "calibrator": _model(CALIBRATOR)})
    assert len(events) == 1
    (event,) = events
    assert event.source == "roster_check"
    assert event.data == {"serving_endpoints": {
        "evaluated_model": {"model": WORKER, "base_url": ANTHROPIC},
        "grader": {"model": GRADER, "base_url": "https://api.openai.com/v1/"},
        "calibrator": {"model": CALIBRATOR, "base_url": ANTHROPIC},
    }}


@pytest.mark.parametrize("var", ["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL"])
def test_roster_check_stops_a_rerouted_run(var, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    with pytest.raises(RosterError, match="serving guard"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(GRADER), "calibrator": _model(CALIBRATOR)})


@pytest.mark.parametrize("model", [GRADER_STEP2, GRADER_STEP3])
def test_terra_and_sol_are_served_exactly_like_luna(model):
    # Same provider, same first-party endpoint, same record shape.
    assert model_serving(_model(model)) == {"model": model, "base_url": "https://api.openai.com/v1/"}
    assert validate_serving("openai", OPENAI, model=model) == OPENAI


@pytest.mark.parametrize("step, model", [(2, GRADER_STEP2), (3, GRADER_STEP3)])
def test_roster_check_records_the_endpoint_of_a_step_two_or_three_grader(step, model, monkeypatch):
    events = _run_roster_check(monkeypatch, _model(WORKER),
                               {"grader": _model(model), "calibrator": _model(CALIBRATOR)}, grader_step=step)
    (event,) = events
    assert event.data["serving_endpoints"]["grader"] == {"model": model, "base_url": "https://api.openai.com/v1/"}


@pytest.mark.parametrize("step, model", [(2, GRADER_STEP2), (3, GRADER_STEP3)])
@pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL", "OPENAI_CUSTOM_HEADERS"])
def test_roster_check_still_stops_a_rerouted_step_two_or_three_grader(step, model, var, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    with pytest.raises(RosterError, match="serving guard"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(model), "calibrator": _model(CALIBRATOR)}, grader_step=step)


def test_roster_check_rejects_terra_at_step_one_and_luna_at_step_two(monkeypatch):
    with pytest.raises(RosterError, match="not its pin"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(GRADER_STEP2), "calibrator": _model(CALIBRATOR)}, grader_step=1)
    with pytest.raises(RosterError, match="not its pin"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(GRADER), "calibrator": _model(CALIBRATOR)}, grader_step=2)


def test_roster_check_rejects_a_step_four_grader(monkeypatch):
    with pytest.raises(RosterError, match="grader_step"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(GRADER_STEP3), "calibrator": _model(CALIBRATOR)}, grader_step=4)


def test_roster_check_rejects_a_grader_equal_to_the_evaluated_model(monkeypatch):
    # The workers' vendor is Anthropic, so only a mock-free equal-name case can arise;
    # the rule still fires at every ladder step.
    for step in (1, 2, 3):
        with pytest.raises(RosterError, match="equals the evaluated model"):
            _run_roster_check(monkeypatch, _model(WORKER),
                              {"grader": _model(WORKER), "calibrator": _model(CALIBRATOR)}, grader_step=step)


def test_roster_check_rejects_a_list_bound_role_with_the_real_role_lookup():
    # A grader bound to [pin, off-pin model]: get_model(role=...) alone would hand
    # back only the pin. The roles are set the way Inspect sets them for a task, in a
    # copied context so nothing leaks into other tests.
    from inspect_ai.model._model import init_model_roles

    roles = {"grader": [_model(GRADER), _model("openai/gpt-6.1-sol")], "calibrator": _model(CALIBRATOR)}

    def run():
        init_model_roles(roles)
        assert get_model(role="grader", required=True) is roles["grader"][0]
        return asyncio.run(roster_check()(object(), None))

    with pytest.raises(RosterError, match="'grader' is bound to more than one model"):
        contextvars.copy_context().run(run)


def test_roster_check_rejects_a_list_bound_role_before_any_get_model(monkeypatch):
    with pytest.raises(RosterError, match="more than one model"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": [_model(GRADER), _model(GRADER)], "calibrator": _model(CALIBRATOR)})


def test_roster_check_still_rejects_names_before_endpoints(monkeypatch):
    with pytest.raises(RosterError, match="not its pin"):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model("openai/gpt-6"), "calibrator": _model(CALIBRATOR)})


def test_roster_check_offline_run_is_unchanged(monkeypatch):
    # Even with a reroute set, an all-mockllm run passes and records nothing.
    monkeypatch.setenv("INSPECT_EVAL_MODEL_BASE_URL", "http://localhost:4000")
    mock = get_model("mockllm/model")
    assert _run_roster_check(monkeypatch, mock, {"grader": mock, "calibrator": mock}) == []


# --- Scout scan jobs ----------------------------------------------------------------------

def test_scanner_serving_skips_mockllm():
    assert scanner_serving("mockllm/model", object(), None) is None


def test_scanner_serving_checks_the_pin():
    record = scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER), None)
    assert record == {"model": SCOUT_SCANNER, "base_url": "https://api.openai.com/v1/"}


@pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "INSPECT_EVAL_MODEL_BASE_URL"])
def test_scanner_serving_rejects_a_rerouted_scanner(var, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    with pytest.raises(ScannerModelError, match="not the first-party endpoint"):
        scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER), None)


@pytest.mark.parametrize("var", HEADER_ENV)
def test_scanner_serving_rejects_custom_headers(var, monkeypatch):
    monkeypatch.setenv(var, "Authorization: Bearer test")
    with pytest.raises(ScannerModelError, match=var):
        scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER), None)


def test_scanner_serving_rejects_a_scan_model_base_url():
    with pytest.raises(ScannerModelError, match="scan model base URL override"):
        scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER), OPENAI)


@pytest.mark.parametrize("var", SCAN_ENV)
def test_scanner_serving_rejects_a_scan_base_url_variable(var, monkeypatch):
    monkeypatch.setenv(var, "http://localhost:4000")
    with pytest.raises(ScannerModelError, match=var):
        scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER), None)


def test_scanner_serving_rejects_a_different_resolved_model():
    with pytest.raises(ScannerModelError, match="not the scanner model"):
        scanner_serving(SCOUT_SCANNER, _model("openai/gpt-6.1-sol"), None)


def test_scanner_serving_fails_closed_on_an_unreadable_model():
    with pytest.raises(ScannerModelError, match="cannot read"):
        scanner_serving(SCOUT_SCANNER, _Fake(name=SCOUT_SCANNER), None)


class _Job:
    """Stands in for a ScanJob built on the pin (the pin needs a provider key)."""

    def __init__(self, resolved, model_base_url, kw):
        self.kw = kw
        self.model = resolved
        self.model_base_url = model_base_url
        self.metadata = kw["metadata"]


def _jobs_with_fake_scanjob(monkeypatch, model_base_url=None):
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    monkeypatch.setattr(jobs, "ScanJob", lambda **kw: _Job(_model(kw["model"]), model_base_url, kw))
    monkeypatch.setattr(jobs, "checked_logs", lambda logs, allow=False, worker_only=False: (["x.eval"], ["dev"]))
    monkeypatch.setattr(jobs, "transcripts_from", lambda files: files)
    return jobs


@pytest.mark.parametrize("job_name", ["worker_awareness_scan", "all_logs_scan"])
def test_scan_jobs_record_the_checked_scanner_endpoint(job_name, monkeypatch):
    jobs = _jobs_with_fake_scanjob(monkeypatch)
    job = getattr(jobs, job_name)(logs="x.eval")
    assert job.metadata["scanner_endpoint"] == {"model": SCOUT_SCANNER, "base_url": "https://api.openai.com/v1/"}


@pytest.mark.parametrize("job_name", ["worker_awareness_scan", "all_logs_scan"])
def test_scan_jobs_refuse_a_rerouted_scanner(job_name, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4000")
    jobs = _jobs_with_fake_scanjob(monkeypatch)
    with pytest.raises(ScannerModelError, match="serving guard"):
        getattr(jobs, job_name)(logs="x.eval")


def test_scan_jobs_refuse_a_job_model_base_url(monkeypatch):
    jobs = _jobs_with_fake_scanjob(monkeypatch, model_base_url="https://proxy.example/v1")
    with pytest.raises(ScannerModelError, match="scan model base URL override"):
        jobs.all_logs_scan(logs="x.eval")


def test_scan_jobs_fail_closed_when_the_job_has_no_model(monkeypatch):
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    monkeypatch.setattr(jobs, "ScanJob", lambda **kw: object())
    monkeypatch.setattr(jobs, "checked_logs", lambda logs, allow=False, worker_only=False: (["x.eval"], ["dev"]))
    monkeypatch.setattr(jobs, "transcripts_from", lambda files: files)
    with pytest.raises(ScannerModelError, match="cannot read the scan job"):
        jobs.all_logs_scan(logs="x.eval")


PROXY_DOTENV = "OPENAI_BASE_URL=http://proxy.example/v1\n"


@pytest.mark.parametrize("job_name", ["worker_awareness_scan", "all_logs_scan"])
def test_scan_jobs_load_the_project_dotenv_before_the_check(job_name, tmp_path, monkeypatch):
    # A real ScanJob on the pin, with a fake key and no transcripts: building it makes
    # a client but sends no request. The .env reroute must reach the check.
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    monkeypatch.setattr(jobs, "checked_logs", lambda logs, allow=False, worker_only=False: (["x.eval"], ["dev"]))
    monkeypatch.setattr(jobs, "transcripts_from", lambda files: None)
    monkeypatch.setattr(inspect_model_module, "_models", {})  # no memoized client from another test
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(PROXY_DOTENV, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    assert "OPENAI_BASE_URL" not in os.environ
    try:
        with pytest.raises(ScannerModelError, match="http://proxy.example/v1"):
            getattr(jobs, job_name)(logs="x.eval")
    finally:
        # load_dotenv set it outside monkeypatch, so remove it here.
        os.environ.pop("OPENAI_BASE_URL", None)


def test_scan_jobs_skip_the_dotenv_when_loading_is_disabled(tmp_path, monkeypatch):
    jobs = importlib.import_module("icfr_agent_evals.scanners.jobs")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(PROXY_DOTENV, encoding="utf-8")
    assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    try:
        jobs.load_scout_environment()
        assert "OPENAI_BASE_URL" not in os.environ
    finally:
        os.environ.pop("OPENAI_BASE_URL", None)


def test_roster_module_exposes_the_event_source():
    assert roster.SERVING_EVENT_SOURCE == "roster_check" and roster.EVALUATED_KEY == "evaluated_model"


# --- GenerateConfig request headers (external review R4) ---------------------------------
# --extra-headers, INSPECT_EVAL_EXTRA_HEADERS (the CLI reads it as --extra-headers) and a
# model's own config.extra_headers all reach the request after the SDK's auth headers,
# so each can replace the credential while the host stays first-party. The marker
# values below are harmless strings; no test sends a request.

MARKER = {"X-Review-Marker": "w2"}


def test_extra_headers_variable_name_is_pinned():
    assert roster.EXTRA_HEADERS_ENV == EXTRA_ENV


def test_no_extra_headers_in_force_by_default():
    assert roster.extra_headers_in_force() == []


@pytest.mark.parametrize("provider, url", [("anthropic", ANTHROPIC), ("openai", OPENAI)])
def test_validate_serving_refuses_any_extra_header_source(provider, url):
    with pytest.raises(RosterError, match="request headers are in force"):
        validate_serving(provider, url, extra_headers=[EXTRA_ENV])
    assert validate_serving(provider, url, extra_headers=[]) == url


@pytest.mark.parametrize("provider, url", [("anthropic", ANTHROPIC), ("openai", OPENAI)])
def test_validate_serving_reads_the_extra_headers_variable_by_default(provider, url, monkeypatch):
    """A direct call with no extra_headers argument reads extra_headers_in_force()."""
    assert validate_serving(provider, url) == url
    monkeypatch.setenv(EXTRA_ENV, '{"X-Review-Marker": "i3"}')
    with pytest.raises(RosterError, match=EXTRA_ENV) as exc:
        validate_serving(provider, url)
    assert "X-Review-Marker" not in str(exc.value)
    assert validate_serving(provider, url, extra_headers=[]) == url  # an explicit empty list checks none


@pytest.mark.parametrize("name", [WORKER, GRADER, CALIBRATOR])
def test_extra_headers_variable_fails(name, monkeypatch):
    monkeypatch.setenv(EXTRA_ENV, '{"X-Review-Marker": "w2"}')
    assert roster.extra_headers_in_force() == [EXTRA_ENV]
    with pytest.raises(RosterError, match=EXTRA_ENV):
        model_serving(_model(name))


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_extra_headers_variable_passes(blank, monkeypatch):
    monkeypatch.setenv(EXTRA_ENV, blank)
    assert roster.extra_headers_in_force() == []
    assert model_serving(_model(WORKER)) == {"model": WORKER, "base_url": ANTHROPIC}


@pytest.mark.parametrize("name", [WORKER, GRADER, CALIBRATOR])
def test_extra_headers_on_the_active_generate_config_fail(name):
    # The eval config (--extra-headers) reaches the setup solver as Inspect's active
    # generate config; it is set here the way Inspect sets it, in a copied context.
    from inspect_ai.model._model import init_active_model

    model = _model(name)

    def run():
        init_active_model(model, GenerateConfig(reasoning_effort="high", extra_headers=MARKER))
        assert roster.extra_headers_in_force() == [roster.ACTIVE_CONFIG_SOURCE]
        return model_serving(model)

    with pytest.raises(RosterError, match="active GenerateConfig.extra_headers"):
        contextvars.copy_context().run(run)


def test_empty_extra_headers_on_the_active_generate_config_pass():
    from inspect_ai.model._model import init_active_model

    model = _model(WORKER)

    def run():
        init_active_model(model, GenerateConfig(reasoning_effort="high", extra_headers={}))
        return model_serving(model)

    assert contextvars.copy_context().run(run) == {"model": WORKER, "base_url": ANTHROPIC}


@pytest.mark.parametrize("name", [WORKER, GRADER, CALIBRATOR])
def test_extra_headers_on_a_model_config_fail(name):
    with pytest.raises(RosterError, match="config.extra_headers"):
        model_serving(_model(name, config=GenerateConfig(extra_headers=MARKER)))


def test_extra_headers_on_a_stub_model_config_fail():
    # The external reviewer's stub: it passed before the fix.
    fake = _Fake(api=_Fake(client=_Fake(base_url=httpx2.URL(ANTHROPIC), auth_token=None), base_url=None),
                 explicit_base_url=None, model_args={}, config=GenerateConfig(extra_headers=MARKER))
    with pytest.raises(RosterError, match="request headers are in force"):
        model_serving(fake)
    fake.config = GenerateConfig()
    assert model_serving(fake) == {"model": WORKER, "base_url": ANTHROPIC}


def test_error_names_the_source_never_the_value(monkeypatch):
    monkeypatch.setenv(EXTRA_ENV, '{"X-Review-Marker": "secret-looking-w2"}')
    with pytest.raises(RosterError) as info:
        model_serving(_model(WORKER))
    assert "secret-looking-w2" not in str(info.value)


def test_roster_check_stops_a_run_with_the_extra_headers_variable(monkeypatch):
    monkeypatch.setenv(EXTRA_ENV, '{"X-Review-Marker": "w2"}')
    with pytest.raises(RosterError, match=EXTRA_ENV):
        _run_roster_check(monkeypatch, _model(WORKER),
                          {"grader": _model(GRADER), "calibrator": _model(CALIBRATOR)})


def test_roster_check_stops_a_run_with_eval_extra_headers(monkeypatch):
    from inspect_ai.model._model import init_active_model

    evaluated = _model(WORKER)

    def run():
        init_active_model(evaluated, GenerateConfig(reasoning_effort="high", extra_headers=MARKER))
        return _run_roster_check(monkeypatch, evaluated, {"grader": _model(GRADER), "calibrator": _model(CALIBRATOR)})

    with pytest.raises(RosterError, match="request headers are in force"):
        contextvars.copy_context().run(run)


def test_scanner_serving_rejects_extra_headers(monkeypatch):
    monkeypatch.setenv(EXTRA_ENV, '{"X-Review-Marker": "w2"}')
    with pytest.raises(ScannerModelError, match=EXTRA_ENV):
        scanner_serving(SCOUT_SCANNER, _model(SCOUT_SCANNER))


def test_eval_extra_headers_reach_the_setup_solver_and_offline_runs_still_skip_the_guard(tmp_path):
    # A mock eval with extra_headers (the in-process form of --extra-headers): the setup
    # stage sees them through extra_headers_in_force, and roster_check on an all-mockllm
    # run still returns before the guard, so the run succeeds as before.
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate, solver

    @solver
    def record_in_force():
        async def solve(state, generate):
            state.store.set("in_force", roster.extra_headers_in_force())
            return state

        return solve

    def run(**kw):
        task = Task(dataset=[Sample(input="x", target="y")], setup=[record_in_force(), roster_check()],
                    solver=generate())
        log = inspect_eval(task, model="mockllm/model", model_roles={"grader": "mockllm/model",
                           "calibrator": "mockllm/model"}, log_dir=str(tmp_path), display="none", **kw)[0]
        assert log.status == "success", log.error
        return log.samples[0].store["in_force"]

    assert run(extra_headers=MARKER) == [roster.ACTIVE_CONFIG_SOURCE]
    assert run() == []
