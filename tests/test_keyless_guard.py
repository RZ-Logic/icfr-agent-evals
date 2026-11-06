"""The keyless guard in conftest.py holds: no provider variable is set, and a .env
file in the working folder reaches neither Inspect's loader nor Scout's."""

import os

from dotenv import load_dotenv
from inspect_ai._util.dotenv import init_dotenv

PROBE = "ICFR_KEYLESS_GUARD_PROBE"


def test_no_provider_variable_is_set():
    assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
        "INSPECT_EVAL_MODEL_BASE_URL", "INSPECT_EVAL_SCAN_MODEL_BASE_URL", "SCOUT_SCAN_MODEL_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS", "INSPECT_EVAL_EXTRA_HEADERS",
    ):
        assert name not in os.environ, name


def test_a_dotenv_file_is_not_loaded(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{PROBE}=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(PROBE, raising=False)
    init_dotenv()  # the loader inspect_ai's eval() and CLI call
    load_dotenv(env_file)  # the call inspect_scout's scan makes
    assert PROBE not in os.environ
