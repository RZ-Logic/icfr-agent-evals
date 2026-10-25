"""Deterministic scoring logic for the Phase 1 variance-commentary content eval.

No Inspect imports, no LLM calls. These functions implement the deterministic
*anchor* checks for the SOX-consequential failure modes — wrong/hallucinated
number, fabricated counterparty, omitted required element. The defensibility
claim depends on these being code, not a stochastic grader: an auditor's question
is not "is the judge calibrated?" but "can bad grading hide a material weakness?"
If the consequential checks are deterministic, the answer is no.

Anchor robustness is itself a design obligation. Error direction is analyzed, not
assumed symmetric: a parser false-FAIL is conservative (it blocks a good output —
an operational cost); a false-PASS on fabrication extraction is the one direction
that would break the "incorrect grading cannot hide a material weakness" claim.
Accordingly:
  - cited entities are compared after canonicalization (casing, whitespace,
    0x-prefix normalization), so a case-normalizing worker is not falsely failed;
  - the free-text narrative is swept for entity-like tokens absent from the
    structured citations — matches are NOT auto-tagged FM-005 (the sweep is a
    heuristic, not a set-membership proof) but are routed to human review with a
    conservative provisional score, so they never silently pass;
  - unparseable output is a failure (format break), also routed to human review.

Subjective sub-dimensions (narrative grounding, coverage adequacy) are LLM-graded
and live elsewhere — they are NOT in this module by design.

Each check returns (points: int in 0..3, failure_modes: list[str]) — except
check_faithfulness, which returns a third element: the list of narrative tokens
requiring human review (empty when none).
"""

from __future__ import annotations

import json
import re

from .policy import materiality_band  # noqa: F401  (kept for callers that recompute)

USD_TOLERANCE = 1.0  # dollars; synthetic figures are exact, tolerance covers rounding

_WS_RE = re.compile(r"\s+")
_HEX_RE = re.compile(r"[0-9a-f]+")

# High-precision address-like token patterns for the narrative sweep.
_ADDRESS_PATTERNS = [
    re.compile(r"\b0x[0-9a-fA-F]{8,}\b"),                 # EVM-style hex
    re.compile(r"\bbc1[0-9a-z]{8,}\b"),                    # BTC bech32-style
    re.compile(r"\baddr1[0-9a-z]{20,}\b"),                 # ADA-style
    re.compile(r"\br[1-9A-HJ-NP-Za-km-z]{20,}\b"),         # XRP-style
    re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b"),        # base58-ish (SOL-style)
]

# TitleCase runs of 2-5 words are treated as org-name candidates. Leading filler
# words are stripped; candidates whose every word is generic reconciliation
# vocabulary are discarded. This is deliberately high-precision-first: the cost of
# a miss here is bounded because a missed narrative entity is still caught by the
# LLM-graded narrative-grounding sub-dimension and the read-the-transcripts
# protocol; the cost of over-flagging is a ballooning review queue.
_ORG_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.\-]*(?:\s+|$)){2,5}")
_LEADING_FILLER = {"the", "a", "an", "in", "on", "at", "per", "for", "as", "of", "and"}
_GENERIC_WORDS = {
    "reserve", "reserves", "reconciliation", "ledger", "liabilities", "liability",
    "on-chain", "chain", "variance", "commentary", "materiality", "band", "bands",
    "period", "surplus", "shortfall", "usd", "custodian", "wallet", "reference",
    "primary", "record", "asset", "assets", "tier", "report", "reporting",
    "m1", "m2", "m3", "btc", "eth", "sol", "usdc", "usdt", "xrp", "ada",
}


def canonicalize_entity(value: object) -> str:
    """Canonical comparison form: strip, collapse whitespace, casefold, and drop a
    0x prefix when the remainder is pure hex. Applied identically to both sides of
    every comparison, so normalization can never create an asymmetry."""
    text = _WS_RE.sub(" ", str(value).strip()).casefold()
    if text.startswith("0x") and _HEX_RE.fullmatch(text[2:] or "x") and len(text) > 2:
        text = text[2:]
    return text


def _balanced_json_objects(text: str):
    """Yield top-level {...} candidates using balanced-brace scanning. Unlike a
    greedy regex, prose containing multiple objects yields each candidate in turn.
    Braces inside JSON strings can confuse the scan; that failure direction is a
    false-fail (format break -> human review), never a false-pass."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            for j in range(i, n):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        i = j
                        break
            else:
                break  # unbalanced to end of text
        i += 1


def parse_worker_output(text: str | None) -> dict | None:
    """Extract the worker's JSON object from a completion.

    Tolerant of code fences and surrounding prose. Tries fenced content first,
    then each balanced top-level object in order, returning the first that parses
    to a dict. Returns None if nothing parses — an unparseable structured output
    is itself a failure (format break), which the caller records and routes to
    human review.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(_balanced_json_objects(text))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm_period(p: object) -> str:
    """Reduce a period to YYYY-MM for comparison (accepts 2025-12, 2025-12-31, ...)."""
    if p is None:
        return ""
    m = re.search(r"(20\d{2})\D?(\d{1,2})", str(p))
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return str(p).strip()


def check_correctness(parsed: dict, truth: dict) -> tuple[int, list[str]]:
    """Numerical facts: stated magnitude == input variance; direction matches;
    stated band == the policy band the inputs fall into. 1 point each, 0..3.

    Known rubric-lock item: FM-002 currently fires on any band mismatch, but an
    OVERSTATED band (e.g. M2 stated as M3) is a wrong value, not a "missed
    material variance" in the understatement sense. The point deduction is correct
    either way; whether overstatement keeps FM-002 or gets its own code is a
    taxonomy decision deferred to rubric lock, recorded here so it is not silently
    absorbed. The band-error direction is exposed in the explanation string.
    """
    fms: list[str] = []
    points = 0

    stated_mag = parsed.get("variance_amount_usd")
    if isinstance(stated_mag, (int, float)) and not isinstance(stated_mag, bool) and (
        abs(float(stated_mag) - abs(truth["variance_usd"])) <= USD_TOLERANCE
    ):
        points += 1
    else:
        fms.append("FM-001")  # hallucinated / wrong number

    if str(parsed.get("variance_direction", "")).strip().lower() == truth["direction"]:
        points += 1
    else:
        fms.append("FM-003")  # wrong sign / direction

    if parsed.get("materiality_band") == truth["materiality_band"]:
        points += 1
    else:
        fms.append("FM-002")  # mis-stated materiality band (see docstring)

    return points, fms


def _suspect_narrative_entities(narrative: str, allowed_canon: set[str]) -> list[str]:
    """Entity-like tokens in the narrative that are absent from the structured
    citations block. High-precision patterns only; results are review candidates,
    not fabrication verdicts."""
    suspects: list[str] = []
    seen: set[str] = set()

    def _flag(token: str) -> None:
        canon = canonicalize_entity(token)
        if not canon or canon in allowed_canon or canon in seen:
            return
        # a token that is a fragment of an allowed entity (or vice versa) is
        # grounded, not suspect — e.g. citing a wallet by its full form while the
        # narrative shows a canonicalized variant
        for ok in allowed_canon:
            if canon in ok or ok in canon:
                return
        seen.add(canon)
        suspects.append(token.strip())

    for pattern in _ADDRESS_PATTERNS:
        for m in pattern.finditer(narrative):
            _flag(m.group(0))

    for m in _ORG_RE.finditer(narrative):
        words = m.group(0).split()
        while words and words[0].casefold() in _LEADING_FILLER:
            words = words[1:]
        if len(words) < 2:
            continue
        if all(w.casefold().strip(".,&-") in _GENERIC_WORDS for w in words):
            continue
        _flag(" ".join(words))

    return suspects


def check_faithfulness(parsed: dict, truth: dict) -> tuple[int, list[str], list[str]]:
    """Fabrication: every entity the worker cites must appear in the input fixture
    (set-membership after canonicalization). Any fabricated citation is a
    consequential failure (FM-005) and drops the score to 0 — fabrication is a
    cliff, not a gradient.

    The narrative is additionally swept for entity-like tokens absent from the
    citations block. Those are NOT FM-005 (the sweep is heuristic); they return as
    the third element, score a conservative provisional 0, and route to
    `requires_human_review` — never a silent pass. Coverage adequacy ('did it cite
    enough?') remains a separate, LLM-graded sub-dimension.

    Returns (points, failure_modes, review_tokens).
    """
    valid_canon = {canonicalize_entity(e) for e in truth.get("valid_entities", [])}
    allowed = valid_canon | {canonicalize_entity(truth.get("asset", ""))}
    allowed.discard("")

    cited = [str(c) for c in (parsed.get("entities_cited") or []) if str(c).strip()]
    fabricated = [c for c in cited if canonicalize_entity(c) not in valid_canon]
    if fabricated:
        return 0, ["FM-005"], []  # fabricated counterparty — deterministic cliff

    review = _suspect_narrative_entities(str(parsed.get("narrative") or ""), allowed)
    if review:
        return 0, [], review  # provisional conservative score, pending human review
    return 3, [], []


def check_completeness(parsed: dict, truth: dict) -> tuple[int, list[str]]:
    """Required-element PRESENCE: asset, period, magnitude each present as a
    non-empty / parseable field. Presence only — value accuracy is Correctness,
    so this dimension stays orthogonal and FM-004 means a genuine omission, not a
    wrong value. 1 point each, 0..3.

    Known Phase-1 gap: value accuracy for asset/period is not yet scored anywhere
    (only magnitude/direction/band accuracy are, in Correctness). Echo-accuracy of
    the given asset/period is a candidate Correctness sub-check at rubric-lock.
    """
    fms: list[str] = []
    points = 0

    if str(parsed.get("asset", "")).strip():            # asset disclosed
        points += 1
    if _norm_period(parsed.get("period")).strip():       # period disclosed
        points += 1
    mag = parsed.get("variance_amount_usd")              # magnitude disclosed as a number
    if isinstance(mag, (int, float)) and not isinstance(mag, bool):
        points += 1

    if points < 3:
        fms.append("FM-004")  # omitted required disclosure element
    return points, fms
