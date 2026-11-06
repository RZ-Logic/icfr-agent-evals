"""Keep the suite keyless.

No test may reach a real provider, and nothing is scored on the evaluation split
before the tag. Inspect and Inspect Scout load the first .env file found at or above
the working folder on every eval() or scan call, so a filled key file would otherwise
reach the test process even when the shell has no keys. This module runs before any
test module is imported: it switches python-dotenv loading off and drops the
provider variables.
"""

import os

PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",  # Inspect prefers it over the API key when set
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "INSPECT_EVAL_MODEL_BASE_URL",  # reroutes every provider
    "INSPECT_EVAL_SCAN_MODEL_BASE_URL",
    "SCOUT_SCAN_MODEL_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",  # can replace the request credential
    "OPENAI_CUSTOM_HEADERS",
    "INSPECT_EVAL_EXTRA_HEADERS",  # request headers sent after the credential
)

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in PROVIDER_ENV:
    os.environ.pop(_name, None)
