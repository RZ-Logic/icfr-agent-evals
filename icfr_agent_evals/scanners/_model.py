"""The scanner model pin for the vendored Scout llm_scanner judges.

eval_awareness and refusal_classifier name roster.SCOUT_SCANNER explicitly; the
upstream examples pass no model and so fall back to whatever model the scan
defaults to. The only other model accepted is Inspect's mockllm provider, for the
offline wiring tests, matching the offline rule in roster.validate_roster.
"""

from __future__ import annotations

from icfr_agent_evals.roster import OFFLINE_PROVIDER, SCOUT_SCANNER, RosterError


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
