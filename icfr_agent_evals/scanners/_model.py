"""The scanner model pin for the vendored Scout llm_scanner judges.

eval_awareness and refusal_classifier name roster.SCOUT_SCANNER explicitly; the
upstream examples pass no model and so fall back to whatever model the scan
defaults to. The only other model accepted is Inspect's mockllm provider, for the
offline wiring tests, matching the offline rule in roster.validate_roster.

Serving guard: the scanner is a real model call, so the scan jobs apply the
roster serving guard (roster.model_serving) to the scanner model the ScanJob
resolved. Its client endpoint must be exactly the OpenAI first-party API, and a
scan model base URL override is refused. The roster guard's other checks apply
too, the custom-header check included. mockllm scanner models skip the guard.
"""

from __future__ import annotations

import os
from typing import Any

from icfr_agent_evals.roster import OFFLINE_PROVIDER, SCOUT_SCANNER, RosterError, model_serving

# Scan-level base URL overrides: scout scan --model-base-url reads the first, and
# inspect eval --scan-model-base-url reads the second. They set only the scan's
# default model; the named llm scanners do not read them. They are refused as a
# precaution.
SCAN_BASE_URL_ENV: tuple[str, ...] = ("SCOUT_SCAN_MODEL_BASE_URL", "INSPECT_EVAL_SCAN_MODEL_BASE_URL")


class ScannerModelError(RosterError):
    """A Scout scanner bound to a model other than its pin (or mockllm offline)."""


def is_offline(model: str) -> bool:
    return model.split("/", 1)[0] == OFFLINE_PROVIDER


def scanner_model(model: str = SCOUT_SCANNER) -> str:
    """Return model unchanged if it is the scanner pin or a mockllm model; raise
    ScannerModelError otherwise."""
    if not isinstance(model, str) or not model:
        raise ScannerModelError(f"scanner model must be a 'provider/name' string, got {model!r}")
    if model != SCOUT_SCANNER and not is_offline(model):
        raise ScannerModelError(
            f"Scout scanner model {model!r} is not its pin {SCOUT_SCANNER!r} (section 5); "
            f"only {OFFLINE_PROVIDER} models are accepted besides the pin"
        )
    return model


def scanner_serving(model: str, resolved: Any, model_base_url: Any = None) -> dict[str, str] | None:
    """Apply the serving guard to a scan's resolved scanner model.

    model: the scanner model name the job was built on (it has passed scanner_model).
    resolved: the Inspect Model the ScanJob resolved for it (ScanJob.model).
    model_base_url: the ScanJob's model_base_url; any value other than None is refused.

    Returns None for a mockllm scanner, else what was checked, in the shape
    roster.model_serving returns. Raises ScannerModelError on any failure."""
    if is_offline(model):
        return None
    if model_base_url is not None:
        raise ScannerModelError(f"serving guard: scan model base URL override {model_base_url!r} is refused")
    for name in SCAN_BASE_URL_ENV:
        if os.environ.get(name):
            raise ScannerModelError(f"serving guard: {name} is set; a scan model base URL override is refused")
    try:
        if str(resolved) != model:
            raise ScannerModelError(
                f"serving guard: the scan resolved {str(resolved)!r}, not the scanner model {model!r}"
            )
        return model_serving(resolved)
    except ScannerModelError:
        raise
    except RosterError as e:
        raise ScannerModelError(str(e)) from e
