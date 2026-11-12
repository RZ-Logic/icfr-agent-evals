"""CI regression gates: candidate runs paired by sample id against the baseline run.

Pre-registration section 10, on per (sample, epoch) records, never on Inspect's
reduced scores (section 6):
  (a) paired rule: a one-sided paired t-test on per-sample epoch-mean differences
      for each of the five rubric dimensions, and an exact one-sided McNemar test
      on per-sample content-pass^4, Holm-corrected as one family of six at
      FAMILY_ALPHA. A dimension no record scores (Calibration, reported and not
      scored in Phase 1, and Robustness) enters at p = 1 and still counts in the family.
      The content-pass^4 test reads the route-blind trial (section 6, owner
      decision D-E5, 2026-11-10; route_blind_epoch): no FM code, no unsettleable
      route, and every trial anchor 3 once each settleable route counts as passed.
      It never reads the code-only trial, in which every routed epoch fails.
  (b) fabrication rule: FM-005 and FM-015 each get a one-sided exact McNemar test
      on per-case indicators (the code fires on any epoch) at FABRICATION_ALPHA,
      outside the Holm family. Citation rule (owner decision D-R9, 2026-11-11): the
      per-case indicator "any epoch holds a citation omission code found"
      (EpochRecord.citation_omissions) gets the same test at CITATION_ALPHA, outside
      the Holm family, and blocks.
  (c) regression sign-off check: a case triggers when a listed FM code, or a
      code-check review route, fires on more epochs in the candidate than in the
      baseline, under the review-volume ladder: R1 triggers on any rise; R2 on a
      rise of two or more epochs, with FM-005, FM-015 and every narrative-sweep
      route kept on R1; R3 (owner decision D-E5, 2026-11-10) keeps FM-005, FM-015
      and every narrative-sweep route on R1, keeps the other FM codes on R2's
      rise of two, and triggers each code-check route key on a rise of three or
      more. The rung is chosen once from the A/A estimate (select_ladder,
      aa_estimate): R1 when its mean volume is at most REVIEW_VOLUME_CAP cases
      per clean PR, else R2 when its mean is at most the cap, else R3, which
      stands whatever its volume. aa_records_from_log reads the 8-epoch A/A
      log (CI regression split, caching off); write_ladder_record records the
      rung once in LADDER_RECORD_RELPATH and refuses to switch it; and
      resolve_ladder gives the sign-off check the recorded rung (R1 while none
      is recorded) and refuses an explicit rung that disagrees. Review
      routes are counted per reason (ratified 2026-11-03): each review reason
      (route_key) counts on its own, even on an epoch that also fired an FM code
      or carried another reason. FM-007 is
      sample-level (consistency.py) and is not a sign-off code. Sign-off cases
      never block; they wait for a recorded human sign-off. The citation-omission
      trigger (CITATION_OMISSION_TRIGGER, owner decision D-R9) counts the epochs
      holding a citation omission code found and takes the rise rule of the FM
      codes other than FM-005 and FM-015 (any rise under R1, a rise of two under R2
      and R3); its cases are kept out of the A/A review volume that picks the rung
      and reported apart (AAEstimate.citation_volume).
  (d) re-runs pool: every candidate run's epochs count together. The candidate's
      epoch means use all pooled epochs, it passes content-pass^4 only if every
      pooled epoch passes the route-blind trial, its fabrication indicator is any
      pooled epoch, and its sign-off counts are raw pooled counts against the
      baseline's count of k. Each asymmetry runs against the candidate.
  Every candidate run must hold the baseline's full sample id set and the same
  dataset content (dataset_version and both dataset SHA-256s, plus the other
  MATCHED_CONTENT_FIELDS), and a truncated (--limit or --sample-id) run is
  refused, so the comparison never runs on an overlap (ratified 2026-11-03).

Mapping from an Inspect (sample, epoch) record (records_from_samples):
  - success (the code-only trial, an extractor health figure only) and
    failure_modes come from reliability.evaluate_trial, unchanged;
  - route_blind is route_blind_epoch(sample), the trial section 10(a) reads;
  - dimensions come from rubric.WIRED_SCORERS: correctness <- correctness_anchor,
    faithfulness <- faithfulness_anchor (min of fabrication and reconciling items,
    rubric.RATIFIED), completeness <- completeness_anchor. Calibration (reported,
    not scored in Phase 1) and robustness (not wired yet) are absent from every record. An errored
    epoch, a missing score or a non-numeric value scores 0 on the wired dimension
    (strict, section 10(a): a format break scores 0 on every anchor);
  - review routes (section 10(c), ratified 2026-11-03): each review reason a
    scorer raises is its own route, keyed by route_key and counted on its own in
    the sign-off check. A reason attached to an FM code is not a route
    (scoring.FABRICATION_REVIEW, which rides with FM-005 and FM-015, and a format
    break, which rides with FM-000). Keys are deduplicated within an epoch, since
    faithfulness_anchor repeats the reasons of fabrication_anchor and
    reconciling_items_anchor. A narrative-sweep hit (scoring.SweepHit.reason) keys
    as "narrative_sweep: <reason>", which holds the hit kind and the token, so
    each suspect entity is its own route and stays on R1. Every other code-check
    reason keys as "code_review: <reason>" with its epoch-varying text dropped, so
    the same reason on two epochs is one route: a narrative extractor reason
    ("kind: matched text") keys by its kind, a reconciling-item type mismatch by
    the fixture type it differs from, an extractor exception by its exception
    type, and any other reason (an ineligible item named without a stated role, a
    blank citation) by its full text. A review flag with no reason and no FM code
    keys as "code_review: review flagged with no stated reason". A sample error
    is a failed trial, not a review route.

The route-blind trial on one epoch (route_blind_epoch; section 6, D-E5): every
settleable route counts as passed and every other rule stands. The epoch fails
on a sample error, any FM code, a format break, a missing or non-numeric trial
anchor, or any route key settlement.is_settleable marks unsettleable (given the
epoch's reasons, so each off-contract field takes its own row). Otherwise each
trial anchor takes the value a `correct` verdict on its settleable routes gives,
read from the settlement table's effects:
  - correctness_anchor: each withheld off-contract point (variance_amount_usd,
    materiality_band) is restored, and an off-contract period lifts the 0 to
    the structured score (details["structured_points"]); a narrative code route
    keeps the score it already has;
  - fabrication_anchor: a routed anchor goes back to 3 once every settleable route
    on it counts as passed (a citation omission no longer holds it at 0, D-R9);
  - reconciling_items_anchor: score kept, so an omission partial stands;
  - completeness_anchor: an off-contract period restores its point.
The epoch passes when every trial anchor then scores 3. Review never settles an
FM code, an error or a format break, so none of them is lifted.
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .reliability import (
    D_R9_RESCORE_HINT, FM002, FM002_UNRECORDED, FULL_MARKS, PREREGISTERED_K, QUEUE_CODE, QUEUE_SWEEP,
    QUEUE_UNSTATED_REASON, TRIAL_ANCHORS, GateError, _numeric, citation_omission_count, evaluate_trial,
    is_template_baseline_log, require_run_config, scored_before_d_r9,
)
from .rubric import WIRED_SCORERS
from .scoring import FABRICATION_REVIEW, SweepHit
from .scoring import SWEEP_KINDS as _SWEEP_KINDS  # every SweepHit kind, truncated_identifier included

DIMENSIONS: tuple[str, ...] = ("correctness", "faithfulness", "calibration", "completeness", "robustness")
PASS4 = "content_pass4"
FAMILY_ALPHA = 0.05          # Holm family-wise alpha, section 10(a)
FABRICATION_ALPHA = 0.025    # section 10(b), per code, outside the Holm family
FABRICATION_CODES: tuple[str, ...] = ("FM-005", "FM-015")
SIGNOFF_CODES: tuple[str, ...] = (
    "FM-000", "FM-001", "FM-002", "FM-003", "FM-004", "FM-005", "FM-006", "FM-015", "FM-016",
)
# Owner decision D-R9 (2026-11-11): the per-case count of epochs holding a citation
# omission code found is its own sign-off trigger and its own (b) test.
CITATION_OMISSION_TRIGGER = "citation_omission"
SIGNOFF_COUNTS: tuple[str, ...] = (CITATION_OMISSION_TRIGGER,)
CITATION_ALPHA = FABRICATION_ALPHA  # section 10(b) citation rule, outside the Holm family
SWEEP_ROUTE = "narrative_sweep"   # route class: holds unconfirmed fabrications, stays on R1 under R2
CODE_ROUTE = "code_review"        # route class: any other code-check review reason
ROUTE_CLASSES: tuple[str, ...] = (SWEEP_ROUTE, CODE_ROUTE)
assert (SWEEP_ROUTE, CODE_ROUTE) == (QUEUE_SWEEP, QUEUE_CODE), "reliability queue classes must match the route classes"
# Trigger families: the FM codes, then the two route classes. A route trigger is
# one route key ("<class>: <reason key>"), each counted on its own (route_key).
SIGNOFF_TRIGGERS: tuple[str, ...] = (*SIGNOFF_CODES, *ROUTE_CLASSES)
R1_ONLY: frozenset[str] = frozenset({"FM-005", "FM-015", SWEEP_ROUTE})
LADDERS: tuple[str, ...] = ("R1", "R2", "R3")
# The rise in a case's failing-epoch count each rule needs to trigger (section 10(c)).
RISE: dict[str, int] = {"R1": 1, "R2": 2, "R3": 3}
assert tuple(RISE) == LADDERS
REVIEW_VOLUME_CAP = 10        # mean A/A sign-off cases per clean PR a rung may hold and still stand
AA_EPOCHS = 8                 # the A/A run: one 8-epoch run of the tagged baseline configuration
AA_FALSE_BLOCK_CEILING = 0.09  # the top of the simulated 4 to 9 percent range; above it is a shortfall
HOLM_M = len(DIMENSIONS) + 1   # the section 10(a) family: five dimensions and content-pass^4
assert HOLM_M == 6
SCORE_RANGE = (0.0, 3.0)

# Wired dimension -> the scorer whose value is that dimension's score on an epoch.
DIMENSION_SCORERS: dict[str, str] = {
    "correctness": "correctness_anchor",
    "faithfulness": "faithfulness_anchor",
    "completeness": "completeness_anchor",
}
for _dim, _scorer in DIMENSION_SCORERS.items():
    assert WIRED_SCORERS.get(_scorer) == _dim, f"{_scorer} does not report {_dim} in rubric.WIRED_SCORERS"

# Sweep reasons, recognized by the prefix SweepHit.reason writes before the token.
_SENTINEL = "\x00sentinel\x00"
SWEEP_REASON_PREFIXES: tuple[str, ...] = tuple(
    SweepHit(_SENTINEL, kind).reason.split(repr(_SENTINEL))[0] for kind in _SWEEP_KINDS
)
FORMAT_BREAK_PREFIX = "format break:"
UNSTATED_REASON = "review flagged with no stated reason"
assert UNSTATED_REASON == QUEUE_UNSTATED_REASON
_NARRATIVE_KIND_RE = re.compile(r"^([a-z][a-z0-9_]*)(?::\s|$)")  # narrative.py writes "kind: matched text"
_TYPE_MISMATCH_RE = re.compile(r"^reconciling_items\[\d+\] .* type .* differs from (.+)$", re.DOTALL)
_EXTRACTOR_RAISED_RE = re.compile(r"^(narrative extractor raised [A-Za-z_][A-Za-z0-9_.]*)")


@dataclass(frozen=True)
class EpochRecord:
    sample_id: str
    epoch: int
    dims: Mapping[str, float]          # dimension -> 0..3 on this epoch; an unwired dimension is absent
    success: bool                      # code-only trial, as reliability.evaluate_trial decides (extractor health)
    failure_modes: tuple[str, ...]     # FM codes fired on this epoch, deduplicated
    review_routes: tuple[str, ...]     # route keys (route_key), deduplicated, sweep keys first
    fm002_direction: str | None = None  # reliability.Trial.fm002_direction, set when FM-002 fired
    # The route-blind trial (route_blind_epoch), the one section 10(a) reads. A record
    # built by hand without it takes `success`, which is exact when it has no
    # settleable route. A code-only pass is always a route-blind pass.
    route_blind: bool | None = None
    # D-R9: omitted tokens code found on the epoch (P12, other exact case-file names and
    # wallet labels, reliability.citation_omission_count); 0 on a hand-built record
    citation_omissions: int = 0

    def __post_init__(self) -> None:
        if self.route_blind is None:
            object.__setattr__(self, "route_blind", bool(self.success))
        if self.success and not self.route_blind:
            raise ValueError(f"{self.sample_id} epoch {self.epoch}: a code-only pass must pass the route-blind trial")


@dataclass(frozen=True)
class PairedTest:
    name: str
    p: float                     # raw one-sided p-value
    p_holm: float | None         # Holm-adjusted p for section 10(a), None for section 10(b)
    alpha: float
    rejected: bool               # True = this test blocks
    n: int                       # paired samples or cases
    detail: Mapping[str, float]


@dataclass(frozen=True)
class SignoffCase:
    sample_id: str
    trigger: str                 # an FM code from SIGNOFF_CODES, or one route key (route_key)
    baseline_count: int          # baseline epochs on which it fired (of k)
    candidate_count: int         # pooled candidate epochs on which it fired
    rule: str                    # the rise rule this trigger took: "R1" (1), "R2" (2) or "R3" (3)
    # FM-002 only: epochs by direction beside the counts (ratified 2026-11-04); reported, not gated
    baseline_by_direction: Mapping[str, int] = field(default_factory=dict, compare=False)
    candidate_by_direction: Mapping[str, int] = field(default_factory=dict, compare=False)
    # citation-omission trigger only (D-R9, m6): omitted tokens beside the epoch counts; reported, not gated
    baseline_tokens: int = field(default=0, compare=False)
    candidate_tokens: int = field(default=0, compare=False)


@dataclass(frozen=True)
class GateReport:
    paired: Mapping[str, PairedTest]        # the five DIMENSIONS and PASS4, m = 6 always
    fabrication: Mapping[str, PairedTest]   # FM-005, FM-015
    signoff: tuple[SignoffCase, ...]
    blocked: bool                           # any Holm rejection in paired, any fabrication or citation rejection
    notes: tuple[str, ...]
    citation: PairedTest | None = None      # section 10(b) citation rule (D-R9); None only on a hand-built report


# --- Statistics, pure Python -----------------------------------------------------------

_EPS = 1e-16
_FPMIN = 1e-300
_MAXIT = 100_000


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function, modified Lentz."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) >= _FPMIN else _FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) >= _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def _reg_inc_beta(a: float, b: float, x: float, y: float) -> float:
    """I_x(a, b) with y = 1 - x passed separately, so neither side loses digits."""
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, y) / b


def t_cdf(t: float, df: float) -> float:
    """Student t CDF via the regularized incomplete beta function."""
    if math.isnan(t) or math.isnan(df) or df <= 0:
        raise ValueError(f"t_cdf needs a number t and df > 0, got t={t}, df={df}")
    if t == 0:
        return 0.5
    if math.isinf(t):
        return 0.0 if t < 0 else 1.0
    t2 = t * t
    denom = df + t2
    tail = 0.5 * _reg_inc_beta(df / 2.0, 0.5, df / denom, t2 / denom)  # P(T <= -|t|)
    return tail if t < 0 else 1.0 - tail


def paired_t_one_sided(baseline: Sequence[float], candidate: Sequence[float]) -> tuple[float, dict]:
    """H1: mean(candidate - baseline) < 0. Returns (p, detail). A zero-variance
    difference gives p = 0 for a drop and 1 otherwise, so a constant drop blocks."""
    if len(baseline) != len(candidate):
        raise ValueError(f"paired samples differ in length: {len(baseline)} vs {len(candidate)}")
    n = len(baseline)
    if n < 2:
        raise ValueError(f"a paired t-test needs at least 2 pairs, got {n}")
    diffs = [float(c) - float(b) for b, c in zip(baseline, candidate)]
    if not all(math.isfinite(d) for d in diffs):
        raise ValueError("paired t-test values must be finite")
    mean = math.fsum(diffs) / n
    sd = math.sqrt(math.fsum((d - mean) ** 2 for d in diffs) / (n - 1))
    df = n - 1
    if sd == 0.0:
        p = 0.0 if mean < 0 else 1.0
        t = -math.inf if mean < 0 else (math.inf if mean > 0 else 0.0)
    else:
        t = mean / (sd / math.sqrt(n))
        p = t_cdf(t, df)
    return p, {"mean_diff": mean, "sd_diff": sd, "t": t, "df": float(df)}


def mcnemar_one_sided(worse: int, better: int) -> float:
    """Exact one-sided McNemar: P(X >= worse), X ~ Binomial(worse + better, 1/2)."""
    if worse < 0 or better < 0:
        raise ValueError(f"discordant counts must be non-negative, got worse={worse}, better={better}")
    n = worse + better
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(worse, n + 1)) / 2**n  # exact int ratio, correctly rounded


def holm(pvalues: Mapping[str, float], alpha: float = FAMILY_ALPHA) -> dict[str, tuple[float, bool]]:
    """Holm step-down: name -> (adjusted p capped at 1, rejected). Ties sort by name."""
    order = sorted(pvalues, key=lambda name: (pvalues[name], name))
    m = len(order)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    rejecting = True
    for i, name in enumerate(order, start=1):
        p = float(pvalues[name])
        running = max(running, min(1.0, (m - i + 1) * p))
        rejecting = rejecting and p <= alpha / (m - i + 1)
        out[name] = (running, rejecting)
    return out


def t_ppf(q: float, df: float) -> float:
    """Student t quantile by bisection on t_cdf; q in (0, 1)."""
    if not 0.0 < q < 1.0 or math.isnan(df) or df <= 0:
        raise ValueError(f"t_ppf needs 0 < q < 1 and df > 0, got q={q}, df={df}")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mcnemar_block_floor(m: int = HOLM_M, reversals: int = 0, *, alpha_rank: int = 1) -> int:
    """Smallest one-way discordant count (worse) whose exact one-sided McNemar p clears
    the Holm threshold FAMILY_ALPHA / (m - alpha_rank + 1) with `reversals` cases
    going the other way. At rank 1 with m = 6 it is 7 with no reversal and 10 with
    one (F2-02): content-pass^4 cannot block below it, and worse never exceeds the
    baseline's passing cases. A protocol helper only; compare() is unchanged."""
    if m < 1 or not 1 <= alpha_rank <= m:
        raise ValueError(f"need m >= 1 and 1 <= alpha_rank <= m, got m={m}, alpha_rank={alpha_rank}")
    if reversals < 0:
        raise ValueError(f"reversals must be non-negative, got {reversals}")
    threshold = FAMILY_ALPHA / (m - alpha_rank + 1)
    for worse in range(1, 10_000):
        if mcnemar_one_sided(worse, reversals) <= threshold:
            return worse
    raise ArithmeticError(f"no discordant count under 10000 blocks at threshold {threshold}")


def paired_t_mdd(sd: float, n: int = 20, alpha_rank: int = 1, *, m: int = HOLM_M, power: float = 0.8) -> float:
    """Minimal detectable mean drop of the one-sided paired t-test at Holm rank
    `alpha_rank` of m, by the central-t approximation (t_{1-a, n-1} + t_{power, n-1}) *
    sd / sqrt(n), sd being the sd of per-sample epoch-mean differences. Inferred, not
    measured: at n = 20, rank 1, 80 percent power it is 0.390 at sd 0.5 and 0.780 at
    sd 1.0 (F2-02). power = 0.5 gives the drop at which p just reaches the threshold."""
    if not math.isfinite(sd) or sd < 0:
        raise ValueError(f"sd must be finite and non-negative, got {sd}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if m < 1 or not 1 <= alpha_rank <= m:
        raise ValueError(f"need m >= 1 and 1 <= alpha_rank <= m, got m={m}, alpha_rank={alpha_rank}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie in (0, 1), got {power}")
    alpha = FAMILY_ALPHA / (m - alpha_rank + 1)
    df = n - 1
    return (t_ppf(1.0 - alpha, df) + t_ppf(power, df)) * sd / math.sqrt(n)


# --- Records from Inspect samples and logs -----------------------------------------------

def route_key(reason: str) -> str | None:
    """The route key one review reason counts under in section 10(c), or None for a
    reason that rides with an FM code. The key drops the text that varies from
    epoch to epoch, so the same reason on two epochs is one route."""
    reason = str(reason)
    if reason == FABRICATION_REVIEW or reason.startswith(FORMAT_BREAK_PREFIX):
        return None
    if reason.startswith(SWEEP_REASON_PREFIXES):
        return f"{SWEEP_ROUTE}: {reason}"  # hit kind and token: each suspect entity is its own route
    m = _NARRATIVE_KIND_RE.match(reason)
    if m:
        return f"{CODE_ROUTE}: {m.group(1)}"
    m = _TYPE_MISMATCH_RE.match(reason)
    if m:
        return f"{CODE_ROUTE}: reconciling item type differs from {m.group(1)}"
    m = _EXTRACTOR_RAISED_RE.match(reason)
    if m:
        return f"{CODE_ROUTE}: {m.group(1)}"
    return f"{CODE_ROUTE}: {reason}"


def route_class(key: str) -> str:
    """SWEEP_ROUTE or CODE_ROUTE for a route key; a bare class name is its own key."""
    for cls in ROUTE_CLASSES:
        if key == cls or key.startswith(cls + ": "):
            return cls
    raise GateError(f"review route {key!r} is not a {' or '.join(ROUTE_CLASSES)} route key")


def _route_order(key: str) -> tuple[int, str]:
    return ROUTE_CLASSES.index(route_class(key)), key


def _routes_of(scores: Mapping[str, Any]) -> tuple[str, ...]:
    routes: set[str] = set()
    for score in scores.values():
        meta = getattr(score, "metadata", None) or {}
        if not meta.get("requires_human_review") or meta.get("format_break"):
            continue
        reasons = [str(r) for r in meta.get("review_reasons") or []]
        if not reasons and not meta.get("failure_modes"):
            routes.add(f"{CODE_ROUTE}: {UNSTATED_REASON}")  # routed with no stated reason and no code
        for reason in reasons:
            key = route_key(reason)
            if key is not None:
                routes.add(key)
    return tuple(sorted(routes, key=_route_order))


# --- The route-blind trial (section 6, owner decision D-E5) ------------------------------

OFF_CONTRACT_WITHHELD_FIELDS: tuple[str, ...] = ("variance_amount_usd", "materiality_band")
OFF_CONTRACT_PERIOD_FIELD = "period"


def _reasons(meta: Mapping[str, Any]) -> list[str]:
    return [str(r) for r in meta.get("review_reasons") or []]


def _off_contract_fields(reasons: Sequence[str]) -> set[str]:
    """The off-contract fields a scorer's reasons name, through the settlement table
    rows of the one key `code_review: off_contract_value`."""
    from . import settlement  # settlement imports this module, so import it late

    if not any(route_key(r) == settlement.OFF_CONTRACT_ROUTE_KEY for r in reasons):
        return set()
    rows = settlement.rows_for(settlement.OFF_CONTRACT_ROUTE_KEY, list(reasons))
    return {r.key for r in rows if r.match == "off_contract_field"}


def _route_blind_anchor(name: str, value: float, meta: Mapping[str, Any]) -> float:
    """The anchor's value once every settleable route on it counts as passed (the
    settlement table's on_correct effects). Called only on an epoch with no FM code
    and no unsettleable route."""
    if value >= FULL_MARKS or not meta.get("requires_human_review"):
        return value  # nothing routed on this anchor, so review cannot move it
    reasons = _reasons(meta)
    details = meta.get("details") or {}
    if name == "correctness_anchor":
        fields = _off_contract_fields(reasons)
        restored = sum(f in fields for f in OFF_CONTRACT_WITHHELD_FIELDS)
        if OFF_CONTRACT_PERIOD_FIELD in fields:
            structured = _numeric(details.get("structured_points"))
            return value if structured is None else structured + restored
        return value + restored
    if name == "fabrication_anchor":
        return float(FULL_MARKS)  # D-R9: a citation omission no longer holds the anchor at 0
    if name == "completeness_anchor":
        return value + (OFF_CONTRACT_PERIOD_FIELD in _off_contract_fields(reasons))
    return value  # reconciling_items_anchor: score kept


def route_blind_epoch(sample: Any) -> bool:
    """The route-blind trial on one (sample, epoch) record: an Inspect EvalSample or
    anything with id, epoch, scores (name -> Score) and error. See the module
    docstring for the rule."""
    from . import settlement  # settlement imports this module, so import it late

    if getattr(sample, "error", None) is not None:
        return False
    scores = getattr(sample, "scores", None) or {}
    epoch_reasons: list[str] = []
    for score in scores.values():
        meta = getattr(score, "metadata", None) or {}
        if meta.get("failure_modes") or meta.get("format_break"):
            return False
        if meta.get("requires_human_review"):
            epoch_reasons.extend(_reasons(meta))
    for key in _routes_of(scores):
        if not settlement.is_settleable(key, epoch_reasons):
            return False
    for name in TRIAL_ANCHORS:
        score = scores.get(name)
        value = _numeric(getattr(score, "value", None)) if score is not None else None
        if value is None:
            return False
        if _route_blind_anchor(name, value, getattr(score, "metadata", None) or {}) != FULL_MARKS:
            return False
    return True


def route_blind_trial(source: Any, *, k: int = PREREGISTERED_K) -> dict[str, bool]:
    """Section 6 route-blind content-pass^k per sample: sample id -> True when every
    epoch 1..k passes route_blind_epoch. `source` is an EvalLog or a path to one
    (read through records_from_log without the run configuration check, so the
    evaluation split and partial runs are refused), or an iterable of Inspect
    samples or EpochRecords. A sample without exactly epochs 1..k is a GateError."""
    if isinstance(source, (str, Path)) or hasattr(source, "eval"):
        records = records_from_log(source, run_config=False)
    else:
        records = [s if isinstance(s, EpochRecord) else record_from_sample(s) for s in source]
    grouped = _by_sample(records, k, "route-blind trial")
    return {sid: all(r.route_blind for r in grouped[sid]) for sid in sorted(grouped)}


def record_from_sample(sample: Any) -> EpochRecord:
    """One (sample, epoch) record: an Inspect EvalSample or anything with id,
    epoch, scores (name -> Score) and error."""
    scores = getattr(sample, "scores", None) or {}
    if scored_before_d_r9(scores):
        raise GateError(f"{sample.id} epoch {sample.epoch}: the score was written before owner decision D-R9; "
                        f"{D_R9_RESCORE_HINT}")
    trial = evaluate_trial(sample)
    errored = getattr(sample, "error", None) is not None
    dims: dict[str, float] = {}
    for dim, name in DIMENSION_SCORERS.items():
        score = scores.get(name)
        value = _numeric(getattr(score, "value", None)) if score is not None else None
        dims[dim] = 0.0 if errored or value is None else value
    return EpochRecord(
        sample_id=trial.sample_id,
        epoch=trial.epoch,
        dims=dims,
        success=trial.success,
        failure_modes=trial.failure_modes,
        review_routes=_routes_of(scores),
        fm002_direction=trial.fm002_direction,
        route_blind=route_blind_epoch(sample),
        citation_omissions=citation_omission_count(scores),
    )


def records_from_samples(samples: Iterable[Any]) -> list[EpochRecord]:
    return [record_from_sample(s) for s in samples]


def run_split(log: Any) -> str | None:
    """The split a log ran on, from task_args, falling back to Task metadata."""
    split = (log.eval.task_args or {}).get("split")
    if split is None:
        split = (getattr(log.eval, "metadata", None) or {}).get("split")
    return None if split is None else str(split)


def refuse_evaluation_split(log: Any) -> None:
    """Section 10 gates run on the CI regression split, never the evaluation split.
    Any sign of the evaluation split, or no recorded split, refuses the log."""
    task_split = (log.eval.task_args or {}).get("split")
    meta_split = (getattr(log.eval, "metadata", None) or {}).get("split")
    dataset = getattr(getattr(log.eval, "dataset", None), "name", None) or ""
    if "eval" in (task_split, meta_split) or str(dataset).endswith("-eval"):
        raise GateError("refusing an evaluation-split log: the regression gates never read the evaluation split")
    if task_split is None and meta_split is None:
        raise GateError("log records no split; refusing it rather than risk reading the evaluation split")
    for s in log.samples or ():
        if str(s.id).startswith("eval-"):
            raise GateError(f"refusing a log holding evaluation-split sample {s.id}")


def load_log(log_or_path: Any) -> Any:
    if isinstance(log_or_path, (str, Path)):
        from inspect_ai.log import read_eval_log

        return read_eval_log(str(log_or_path))
    return log_or_path


def refuse_partial_run(log: Any) -> None:
    """Section 6: each gate asserts that scored samples equal the split's sample count.
    A run truncated by --limit or --sample-id, or whose samples differ from the
    dataset its header records, is refused."""
    cfg = log.eval.config
    if getattr(cfg, "limit", None) is not None:
        raise GateError(f"run used limit {cfg.limit!r}; the gates score the whole split only")
    if getattr(cfg, "sample_id", None) is not None:
        raise GateError(f"run used sample_id {cfg.sample_id!r}; the gates score the whole split only")
    dataset = getattr(log.eval, "dataset", None)
    expected = getattr(dataset, "samples", None)
    if expected is None:
        raise GateError("log header records no dataset sample count; cannot confirm the whole split was scored")
    ids = {str(s.id) for s in log.samples}
    if len(ids) != int(expected):
        raise GateError(f"scored samples {len(ids)} differ from the dataset's sample count {expected}")
    recorded = getattr(dataset, "sample_ids", None)
    if recorded is not None and {str(i) for i in recorded} != ids:
        raise GateError("scored sample ids differ from the sample ids the log header records")


# Header fields that fix the case content behind each sample id. Paired runs must
# agree on every one, each recorded, or pairing by id compares different cases.
# dataset_sha256 hashes the samples the task selects (a pilot run hashes its 20 cases) and
# split_dataset_sha256 hashes the whole split (ratified 2026-11-03).
MATCHED_CONTENT_FIELDS: tuple[str, ...] = (
    "dataset_version", "dataset_sha256", "split_dataset_sha256", "schema_version", "split", "split_seed", "subset",
)


def content_fingerprint(log: Any) -> dict[str, Any]:
    """Task version plus MATCHED_CONTENT_FIELDS from the log's Task metadata;
    a field the header lacks raises GateError."""
    meta = getattr(log.eval, "metadata", None) or {}
    out: dict[str, Any] = {"task_version": getattr(log.eval, "task_version", None)}
    for key in MATCHED_CONTENT_FIELDS:
        out[key] = meta.get(key)
    missing = [k for k, v in out.items() if v is None and k != "subset"]
    if "subset" not in meta:
        missing.append("subset")
    if missing:
        raise GateError(f"log header lacks {missing}; cannot confirm the runs scored the same cases")
    return out


def records_from_log(log_or_path: Any, *, run_config: bool = True,
                     max_tokens_pin: int | None = None, epochs: int = PREREGISTERED_K) -> list[EpochRecord]:
    """Records from an EvalLog (or a path to one) read with its samples. With
    run_config (the default) the log must also pass the contract C2 run
    configuration check at `epochs` epochs (reliability.require_run_config).
    `epochs` stays PREREGISTERED_K for every gate; only the A/A reader
    (aa_records_from_log) passes AA_EPOCHS."""
    log = load_log(log_or_path)
    refuse_evaluation_split(log)
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    refuse_partial_run(log)
    ran = int(log.eval.config.epochs or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if declared is not None and int(declared) != ran:
        raise GateError(f"run epochs {ran} differ from the task argument epochs {declared}")
    if run_config:
        require_run_config(log, expected_epochs=epochs, max_tokens_pin=max_tokens_pin)
    return records_from_samples(log.samples)


@dataclass(frozen=True)
class FabricationInstance:
    """One fabrication code firing on one candidate epoch, section 10(b) PR listing."""
    run: int          # 1-based candidate run index within the pool
    sample_id: str
    epoch: int
    code: str


def fabrication_instances(candidates: Sequence[Sequence[EpochRecord]]) -> tuple[FabricationInstance, ...]:
    """Section 10(b): every fabrication instance is listed in the PR report, whether
    or not it rises over the baseline."""
    out = [
        FabricationInstance(i + 1, str(r.sample_id), int(r.epoch), code)
        for i, run in enumerate(candidates)
        for r in run
        for code in FABRICATION_CODES
        if code in r.failure_modes
    ]
    return tuple(sorted(out, key=lambda f: (f.run, f.sample_id, f.epoch, f.code)))


# --- Section 10 comparison ---------------------------------------------------------------

def _by_sample(records: Sequence[EpochRecord], k: int, label: str) -> dict[str, list[EpochRecord]]:
    grouped: dict[str, list[EpochRecord]] = defaultdict(list)
    for r in records:
        grouped[str(r.sample_id)].append(r)
    bad = {sid: tuple(sorted(r.epoch for r in rs)) for sid, rs in grouped.items()
           if sorted(r.epoch for r in rs) != list(range(1, k + 1))}
    if bad:
        raise GateError(f"{label}: samples without exactly epochs 1..{k}: {dict(sorted(bad.items()))}")
    if not grouped:
        raise GateError(f"{label}: no records")
    return grouped


def _dimension_presence(records: Iterable[EpochRecord], label: str) -> dict[str, bool]:
    """dimension -> scored on every record (True) or on none (False); anything else is a GateError."""
    seen: dict[str, int] = dict.fromkeys(DIMENSIONS, 0)
    total = 0
    for r in records:
        total += 1
        unknown = set(r.dims) - set(DIMENSIONS)
        if unknown:
            raise GateError(f"{label}: unknown dimensions {sorted(unknown)} on {r.sample_id} epoch {r.epoch}")
        for dim, value in r.dims.items():
            v = _numeric(value)
            if v is None or not SCORE_RANGE[0] <= v <= SCORE_RANGE[1]:
                raise GateError(f"{label}: {dim} = {value!r} on {r.sample_id} epoch {r.epoch} is not a 0..3 score")
            seen[dim] += 1
    partial = [d for d, c in seen.items() if 0 < c < total]
    if partial:
        raise GateError(f"{label}: dimensions scored on some records and not others: {partial}")
    return {d: c == total for d, c in seen.items()}


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return math.fsum(vals) / len(vals)


# --- Section 10(c): the review-volume ladder ---------------------------------------------

ROUTE_BLIND_NOTE = ("content_pass4 reads the route-blind trial (section 6, D-E5): no FM code, no unsettleable "
                    "route, every anchor 3 once settleable routes count as passed")
LADDER_NOTES: dict[str, str] = {
    "R2": "ladder R2: a single-epoch rise triggers only on FM-005, FM-015 and the narrative-sweep routes",
    "R3": ("ladder R3: FM-005, FM-015 and the narrative-sweep routes trigger on any rise, the other FM codes "
           "on a rise of two epochs, each code-check route key on a rise of three"),
}


def trigger_rule(trigger: str, ladder: str) -> str:
    """The rise rule ("R1", "R2" or "R3") one sign-off trigger takes under a ladder
    rung. FM-005, FM-015 and every narrative-sweep route key stay on R1 under every
    rung; the other FM codes and the citation-omission trigger (D-R9) take R2 under
    R2 and R3; a code-check route key takes the rung itself."""
    if ladder not in LADDERS:
        raise ValueError(f"ladder must be one of {LADDERS}, got {ladder!r}")
    if ladder == "R1" or trigger in R1_ONLY:
        return "R1"
    if trigger in SIGNOFF_CODES or trigger in SIGNOFF_COUNTS:
        return "R2"
    return "R1" if route_class(trigger) == SWEEP_ROUTE else ladder


def signoff_check(
    base: Mapping[str, Sequence[EpochRecord]],
    pooled: Mapping[str, Sequence[EpochRecord]],
    ladder: str = "R1",
) -> tuple[SignoffCase, ...]:
    """Section 10(c) on records grouped by sample id (the same ids on both sides):
    every SIGNOFF_CODES code, the citation-omission count (SIGNOFF_COUNTS, D-R9) and
    every route key either side carries, each counted on its own, triggers when its
    candidate count rises over the baseline count by at least
    RISE[trigger_rule(trigger, ladder)] epochs."""
    if ladder not in LADDERS:
        raise ValueError(f"ladder must be one of {LADDERS}, got {ladder!r}")
    if set(base) != set(pooled):
        raise GateError("sign-off check: baseline and candidate sample ids differ")

    def fired(r: EpochRecord, trigger: str) -> bool:
        if trigger == CITATION_OMISSION_TRIGGER:
            return r.citation_omissions > 0
        return trigger in r.failure_modes or trigger in r.review_routes

    def by_direction(records: Iterable[EpochRecord]) -> dict[str, int]:
        found = Counter(r.fm002_direction or FM002_UNRECORDED for r in records if FM002 in r.failure_modes)
        return dict(sorted(found.items()))

    signoff: list[SignoffCase] = []
    for sid in sorted(base):
        keys = {key for r in (*base[sid], *pooled[sid]) for key in r.review_routes}
        for trigger in (*SIGNOFF_CODES, *SIGNOFF_COUNTS, *sorted(keys, key=_route_order)):
            bc = sum(fired(r, trigger) for r in base[sid])
            cc = sum(fired(r, trigger) for r in pooled[sid])
            rule = trigger_rule(trigger, ladder)
            if cc - bc >= RISE[rule]:
                if trigger == FM002:
                    signoff.append(SignoffCase(sid, trigger, bc, cc, rule,
                                               by_direction(base[sid]), by_direction(pooled[sid])))
                elif trigger == CITATION_OMISSION_TRIGGER:
                    signoff.append(SignoffCase(sid, trigger, bc, cc, rule,
                                               baseline_tokens=sum(r.citation_omissions for r in base[sid]),
                                               candidate_tokens=sum(r.citation_omissions for r in pooled[sid])))
                else:
                    signoff.append(SignoffCase(sid, trigger, bc, cc, rule))
    return tuple(signoff)


def select_ladder(mean_volume: Mapping[str, float], cap: float = REVIEW_VOLUME_CAP) -> str:
    """The rung section 10(c) fixes once from the A/A estimate: R1 when its mean
    review volume per clean PR is at most `cap`, else R2 when its mean is at most
    `cap`, else R3, which stands whatever its volume. mean_volume holds the A/A
    means by rung, counted by route key (AAEstimate.mean_volume)."""
    for rung in ("R1", "R2"):
        value = mean_volume.get(rung)
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"mean_volume[{rung!r}] must be a finite non-negative number, got {value!r}")
    if mean_volume["R1"] <= cap:
        return "R1"
    if mean_volume["R2"] <= cap:
        return "R2"
    return "R3"


# --- Section 10: the A/A estimate --------------------------------------------------------

def aa_splits(epochs: int = AA_EPOCHS, k: int = PREREGISTERED_K) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Every distinct k-and-k split of epochs 1..2k as (baseline half, candidate half),
    the half holding epoch 1 as the baseline: 35 splits for 8 epochs."""
    if epochs != 2 * k or k < 1:
        raise ValueError(f"the A/A estimate splits 2k epochs into two halves of k, got epochs={epochs}, k={k}")
    out = []
    for rest in itertools.combinations(range(2, epochs + 1), k - 1):
        half = (1, *rest)
        out.append((half, tuple(e for e in range(1, epochs + 1) if e not in half)))
    return tuple(out)


@dataclass(frozen=True)
class AASplit:
    baseline_epochs: tuple[int, ...]
    candidate_epochs: tuple[int, ...]
    blocked: bool                        # any automatic block under (a) or (b)
    rejected: tuple[str, ...]            # the (a) and (b) tests that rejected
    volume: Mapping[str, int]            # rung -> sign-off cases, counted by route key (trigger)
    volume_samples: Mapping[str, int]    # rung -> distinct samples with a sign-off case
    # D-R9 (M2): citation-omission sign-off cases per rung, kept out of `volume`
    citation_volume: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AAEstimate:
    splits: tuple[AASplit, ...]
    false_block_rate: float              # fraction of splits blocked by (a) or (b)
    false_block_shortfall: bool          # the rate exceeds AA_FALSE_BLOCK_CEILING
    mean_volume: Mapping[str, float]     # rung -> mean sign-off cases per split, counted by route key
    mean_volume_samples: Mapping[str, float]  # rung -> mean distinct samples per split, beside it
    ladder: str                          # select_ladder(mean_volume)
    # D-R9 (M2): mean citation-omission sign-off cases per split, reported apart, never in the rung choice
    mean_citation_volume: Mapping[str, float] = field(default_factory=dict)


def aa_estimate(records: Sequence[EpochRecord], *, epochs: int = AA_EPOCHS, k: int = PREREGISTERED_K) -> AAEstimate:
    """Section 10 A/A estimate on the records of one 8-epoch run: every split of
    aa_splits, each applied to every sample, is compared as a clean PR (epochs
    renumbered 1..k in order on each side). It reports the null false-block rate of
    (a) and (b) (the citation rule of (b) included, D-R9) and the review volume of (c)
    under R1, R2 and R3, counted by route key, and the rung select_ladder takes from
    it. Citation-omission sign-off cases (D-R9, M2) are counted apart, in
    citation_volume, and never in the volume the rung is chosen from."""
    grouped = _by_sample(records, epochs, "A/A run")
    splits: list[AASplit] = []
    for base_half, cand_half in aa_splits(epochs, k):
        def half(epoch_set: tuple[int, ...]) -> list[EpochRecord]:
            new = {e: i + 1 for i, e in enumerate(epoch_set)}
            return [replace(r, epoch=new[r.epoch]) for rs in grouped.values() for r in rs if r.epoch in new]

        base_recs, cand_recs = half(base_half), half(cand_half)
        report = compare(base_recs, [cand_recs], k=k, ladder="R1")
        base_g, cand_g = _by_sample(base_recs, k, "A/A baseline half"), _by_sample(cand_recs, k, "A/A candidate half")
        volume: dict[str, int] = {}
        volume_samples: dict[str, int] = {}
        citation_volume: dict[str, int] = {}
        for rung in LADDERS:
            every = report.signoff if rung == "R1" else signoff_check(base_g, cand_g, rung)
            cases = [c for c in every if c.trigger not in SIGNOFF_COUNTS]
            volume[rung] = len(cases)
            volume_samples[rung] = len({c.sample_id for c in cases})
            citation_volume[rung] = len(every) - len(cases)
        tests = (*report.paired.items(), *report.fabrication.items(),
                 *(((report.citation.name, report.citation),) if report.citation is not None else ()))
        rejected = tuple(name for name, t in tests if t.rejected)
        splits.append(AASplit(base_half, cand_half, report.blocked, rejected, volume, volume_samples,
                              citation_volume))
    n = len(splits)
    rate = sum(s.blocked for s in splits) / n
    mean_volume = {rung: math.fsum(s.volume[rung] for s in splits) / n for rung in LADDERS}
    mean_samples = {rung: math.fsum(s.volume_samples[rung] for s in splits) / n for rung in LADDERS}
    mean_citation = {rung: math.fsum(s.citation_volume.get(rung, 0) for s in splits) / n for rung in LADDERS}
    return AAEstimate(tuple(splits), rate, rate > AA_FALSE_BLOCK_CEILING, mean_volume, mean_samples,
                      select_ladder(mean_volume), mean_citation)


def format_aa_estimate(est: AAEstimate) -> str:
    """The A/A estimate as report lines: false-block rate and review volume by rung."""
    lines = [f"A/A estimate: {len(est.splits)} splits",
             f"  false-block rate {est.false_block_rate:.4f}"
             + ("  SHORTFALL (above 0.09)" if est.false_block_shortfall else ""),
             "  review volume per clean PR (mean sign-off cases per split, counted by route key):"]
    for rung in LADDERS:
        lines.append(f"    {rung}: {est.mean_volume[rung]:.3f}  (distinct samples {est.mean_volume_samples[rung]:.3f})")
    lines.append("  citation-omission sign-off volume per clean PR (D-R9; reported apart, not counted toward the rung):")
    for rung in LADDERS:
        lines.append(f"    {rung}: {est.mean_citation_volume.get(rung, 0.0):.3f}")
    lines.append(f"  rung the ladder takes (cap {REVIEW_VOLUME_CAP} per clean PR): {est.ladder}")
    return "\n".join(lines)


# --- Section 10: the A/A log and the recorded ladder rung --------------------------------

AA_SPLIT = "ci"   # the A/A run is on the CI regression split (section 10, A/A estimate)
LADDER_RULING = "D-E5"
LADDER_RECORD_SCHEMA = 1
# Where the rung is recorded, relative to the repository root; the file is committed.
LADDER_RECORD_RELPATH = "docs/evidence/regression/ladder.json"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_AA_LOG_FIELDS: tuple[str, ...] = ("file", "sha256", "eval_id", "run_commit", "split", "epochs", "samples", "model")
_LADDER_RECORD_FIELDS: tuple[str, ...] = (
    "schema", "ruling", "rung", "cap", "splits", "mean_volume", "mean_volume_samples",
    "false_block_rate", "false_block_shortfall", "aa_log", "recorded",
)


def refuse_cached_run(log: Any) -> None:
    """The A/A run is made with caching off: any model call the log shows as a
    cache read or write refuses it, since a cached reply makes epochs repeat."""
    for s in log.samples or ():
        for ev in getattr(s, "events", None) or ():
            if getattr(ev, "event", None) == "model" and getattr(ev, "cache", None) is not None:
                raise GateError(f"sample {s.id} epoch {s.epoch}: a model call shows cache {ev.cache!r}; "
                                "the A/A run is made with caching off")


def aa_records_from_log(log_or_path: Any, *, max_tokens_pin: int | None = None) -> list[EpochRecord]:
    """Records of the section 10 A/A run: one AA_EPOCHS-epoch run of the baseline
    configuration on the CI regression split, with caching off. The evaluation
    split, any other split, a subset, a template baseline log, a cached run and a
    partial run are refused, and the run configuration check runs at AA_EPOCHS."""
    log = load_log(log_or_path)
    refuse_evaluation_split(log)
    if is_template_baseline_log(log):
        raise GateError("the template baseline log has no evaluated model; the A/A run is of the baseline configuration")
    task_split = (log.eval.task_args or {}).get("split")
    meta = getattr(log.eval, "metadata", None) or {}
    for where, value in (("task_args", task_split), ("Task metadata", meta.get("split"))):
        if value is not None and str(value) != AA_SPLIT:
            raise GateError(f"{where} split is {value!r}; the A/A run is on the CI regression split {AA_SPLIT!r}")
    if meta.get("subset") is not None:
        raise GateError(f"run used subset {meta['subset']!r}; the A/A run covers the whole CI regression split")
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    stray = sorted({str(s.id) for s in log.samples if not str(s.id).startswith(f"{AA_SPLIT}-")})
    if stray:
        raise GateError(f"sample ids outside the CI regression split: {stray[:5]}")
    ran = int(log.eval.config.epochs or 1)
    if ran != AA_EPOCHS:
        raise GateError(f"run has {ran} epochs; the A/A run has {AA_EPOCHS}")
    refuse_cached_run(log)
    return records_from_log(log, epochs=AA_EPOCHS, max_tokens_pin=max_tokens_pin)


def ladder_record(est: AAEstimate, aa_log: Mapping[str, Any], recorded: str) -> dict[str, Any]:
    """The rung choice as the committed record (section 10(c), D-E5): the rung,
    the volumes it was chosen from, and the A/A log it was measured on. aa_log
    holds file, sha256, eval_id, run_commit, split, epochs, samples and model."""
    record = {
        "schema": LADDER_RECORD_SCHEMA,
        "ruling": LADDER_RULING,
        "rung": est.ladder,
        "cap": REVIEW_VOLUME_CAP,
        "splits": len(est.splits),
        "mean_volume": {rung: float(est.mean_volume[rung]) for rung in LADDERS},
        "mean_volume_samples": {rung: float(est.mean_volume_samples[rung]) for rung in LADDERS},
        "false_block_rate": float(est.false_block_rate),
        "false_block_shortfall": bool(est.false_block_shortfall),
        "aa_log": {key: aa_log.get(key) for key in _AA_LOG_FIELDS},
        "recorded": recorded,
    }
    return validate_ladder_record(record, "new ladder record")


def validate_ladder_record(record: Any, where: str) -> dict[str, Any]:
    """A ladder record with every field, the D-E5 ruling, a known rung, and the
    rung select_ladder takes from its own volumes; anything else is a GateError."""
    if not isinstance(record, dict):
        raise GateError(f"{where}: not a JSON object")
    missing = [k for k in _LADDER_RECORD_FIELDS if k not in record]
    if missing:
        raise GateError(f"{where}: missing {missing}")
    if record["schema"] != LADDER_RECORD_SCHEMA:
        raise GateError(f"{where}: schema {record['schema']!r}, expected {LADDER_RECORD_SCHEMA}")
    if record["ruling"] != LADDER_RULING:
        raise GateError(f"{where}: ruling {record['ruling']!r}, expected {LADDER_RULING!r}")
    if record["rung"] not in LADDERS:
        raise GateError(f"{where}: rung {record['rung']!r} is not one of {LADDERS}")
    if record["cap"] != REVIEW_VOLUME_CAP:
        raise GateError(f"{where}: cap {record['cap']!r}, expected {REVIEW_VOLUME_CAP}")
    volume = record["mean_volume"]
    if not isinstance(volume, dict) or set(volume) != set(LADDERS):
        raise GateError(f"{where}: mean_volume must hold {LADDERS}")
    try:
        chosen = select_ladder(volume)
    except (TypeError, ValueError) as e:
        raise GateError(f"{where}: {e}") from e
    if chosen != record["rung"]:
        raise GateError(f"{where}: rung {record['rung']} does not follow from its volumes (select_ladder gives {chosen})")
    aa_log = record["aa_log"]
    if not isinstance(aa_log, dict) or [k for k in _AA_LOG_FIELDS if k not in aa_log]:
        raise GateError(f"{where}: aa_log must hold {list(_AA_LOG_FIELDS)}")
    for key in ("file", "eval_id", "run_commit"):
        if not isinstance(aa_log[key], str) or not aa_log[key]:
            raise GateError(f"{where}: aa_log {key} must be a non-empty string")
    if not isinstance(aa_log["sha256"], str) or not _HEX64_RE.match(aa_log["sha256"]):
        raise GateError(f"{where}: aa_log sha256 must be 64 lowercase hex characters")
    if aa_log["split"] != AA_SPLIT or aa_log["epochs"] != AA_EPOCHS:
        raise GateError(f"{where}: aa_log must name split {AA_SPLIT!r} and epochs {AA_EPOCHS}")
    try:
        datetime.date.fromisoformat(str(record["recorded"]))
    except ValueError as e:
        raise GateError(f"{where}: recorded {record['recorded']!r} is not a YYYY-MM-DD date") from e
    return record


def read_ladder_record(path: str | Path) -> dict[str, Any] | None:
    """The recorded rung, or None when no record exists at path. A record that
    cannot be read or fails validate_ladder_record is a GateError."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise GateError(f"ladder record {p}: cannot read it ({type(e).__name__}: {e})") from e
    return validate_ladder_record(record, f"ladder record {p}")


def write_ladder_record(path: str | Path, record: Mapping[str, Any]) -> str:
    """Write the rung choice once. With no file at path it is written and
    "written" is returned. An existing record with the same rung from the same
    A/A log (by SHA-256) is left as it is ("unchanged"). Any other existing record
    is refused: the rung is chosen once and never switched (section 10(c))."""
    new = validate_ladder_record(dict(record), "new ladder record")
    p = Path(path)
    old = read_ladder_record(p)
    if old is not None:
        if old["rung"] != new["rung"]:
            raise GateError(f"ladder record {p} already holds rung {old['rung']} (recorded {old['recorded']}); "
                            f"the rung is chosen once and never switched (section 10(c), {LADDER_RULING}), "
                            f"so {new['rung']} is not recorded")
        if old["aa_log"]["sha256"] != new["aa_log"]["sha256"]:
            raise GateError(f"ladder record {p} already holds rung {old['rung']} from A/A log "
                            f"{old['aa_log']['file']} ({old['aa_log']['sha256']}); a record from another log "
                            "does not replace it")
        return "unchanged"
    body = json.dumps(new, indent=2, sort_keys=True) + "\n"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
    except OSError as e:
        raise GateError(f"cannot write ladder record {p}: {type(e).__name__}: {e}") from e
    return "written"


def resolve_ladder(requested: str | None, record_path: str | Path) -> tuple[str, bool, str]:
    """The rung the sign-off check uses, whether it was read from a record, and a
    report note naming where it came from. A recorded rung wins; an explicit `requested` rung that disagrees with
    it is a GateError. With no record, `requested` (default R1) stands and the
    note says no rung is recorded yet."""
    if requested is not None and requested not in LADDERS:
        raise ValueError(f"ladder must be one of {LADDERS}, got {requested!r}")
    record = read_ladder_record(record_path)
    if record is None:
        rung = requested or "R1"
        how = "the default R1" if requested is None else f"--ladder {rung} as given"
        return rung, False, (f"no ladder rung is recorded yet (no file at {record_path}); the sign-off check uses {how} "
                      f"until the A/A estimate records one (section 10(c), {LADDER_RULING})")
    rung = record["rung"]
    if requested is not None and requested != rung:
        raise GateError(f"--ladder {requested} disagrees with rung {rung} recorded in {record_path}; the rung is "
                        f"chosen once from the A/A estimate and never switched (section 10(c), {LADDER_RULING})")
    return rung, True, (f"ladder {rung} read from {record_path} (recorded {record['recorded']} from A/A log "
                  f"{record['aa_log']['file']}, {LADDER_RULING})")


def compare(
    baseline: Sequence[EpochRecord],
    candidates: Sequence[Sequence[EpochRecord]],
    *,
    k: int = PREREGISTERED_K,
    ladder: str = "R1",
) -> GateReport:
    """Section 10 (a) to (d): the candidate runs pool, paired by sample id with the baseline."""
    if ladder not in LADDERS:
        raise ValueError(f"ladder must be one of {LADDERS}, got {ladder!r}")
    if isinstance(candidates, Mapping) or not candidates or isinstance(candidates[0], EpochRecord):
        raise ValueError("candidates must be a non-empty sequence of runs, each a sequence of EpochRecord")

    base = _by_sample(baseline, k, "baseline")
    runs = [_by_sample(run, k, f"candidate run {i + 1}") for i, run in enumerate(candidates)]
    ids = sorted(base)
    for i, run in enumerate(runs):
        if set(run) != set(base):
            missing, extra = sorted(set(base) - set(run)), sorted(set(run) - set(base))
            raise GateError(f"candidate run {i + 1} sample ids differ from the baseline: missing {missing}, extra {extra}")
    pooled: dict[str, list[EpochRecord]] = {sid: [r for run in runs for r in run[sid]] for sid in ids}
    notes: list[str] = []
    if len(runs) > 1:
        notes.append(f"candidate pools {len(runs)} runs ({len(runs) * k} epochs per sample), section 10(d)")

    # (a) paired rule
    base_dims = _dimension_presence((r for sid in ids for r in base[sid]), "baseline")
    cand_dims = _dimension_presence((r for sid in ids for r in pooled[sid]), "candidate")
    raw: dict[str, tuple[float, int, dict]] = {}
    for dim in DIMENSIONS:
        if base_dims[dim] != cand_dims[dim]:
            which = "baseline" if base_dims[dim] else "candidate"
            raise GateError(f"{dim} is scored in the {which} run only; the runs are not comparable")
        if not base_dims[dim]:
            raw[dim] = (1.0, len(ids), {})
            notes.append(f"{dim}: not scored in either run; p = 1, still counted in the Holm family (m = 6)")
            continue
        b = [_mean(r.dims[dim] for r in base[sid]) for sid in ids]
        c = [_mean(r.dims[dim] for r in pooled[sid]) for sid in ids]
        p, detail = paired_t_one_sided(b, c)
        raw[dim] = (p, len(ids), detail)

    # content-pass^4 reads the route-blind trial (section 6, D-E5), never the code-only one
    base_pass = {sid: all(r.route_blind for r in base[sid]) for sid in ids}
    cand_pass = {sid: all(r.route_blind for r in pooled[sid]) for sid in ids}
    worse = sum(base_pass[s] and not cand_pass[s] for s in ids)
    better = sum(cand_pass[s] and not base_pass[s] for s in ids)
    raw[PASS4] = (
        mcnemar_one_sided(worse, better),
        len(ids),
        {"worse": float(worse), "better": float(better),
         "baseline_pass": float(sum(base_pass.values())), "candidate_pass": float(sum(cand_pass.values()))},
    )
    adjusted = holm({name: v[0] for name, v in raw.items()}, FAMILY_ALPHA)
    paired = {
        name: PairedTest(name, raw[name][0], adjusted[name][0], FAMILY_ALPHA, adjusted[name][1], raw[name][1], raw[name][2])
        for name in (*DIMENSIONS, PASS4)
    }

    # (b) fabrication rule
    fabrication: dict[str, PairedTest] = {}
    for code in FABRICATION_CODES:
        b_ind = {sid: any(code in r.failure_modes for r in base[sid]) for sid in ids}
        c_ind = {sid: any(code in r.failure_modes for r in pooled[sid]) for sid in ids}
        w = sum(c_ind[s] and not b_ind[s] for s in ids)
        bt = sum(b_ind[s] and not c_ind[s] for s in ids)
        p = mcnemar_one_sided(w, bt)
        fabrication[code] = PairedTest(code, p, None, FABRICATION_ALPHA, p <= FABRICATION_ALPHA, len(ids),
                                       {"worse": float(w), "better": float(bt)})

    # (b) citation rule (D-R9): the per-case indicator "any epoch holds a citation omission"
    b_cite = {sid: any(r.citation_omissions > 0 for r in base[sid]) for sid in ids}
    c_cite = {sid: any(r.citation_omissions > 0 for r in pooled[sid]) for sid in ids}
    w = sum(c_cite[s] and not b_cite[s] for s in ids)
    bt = sum(b_cite[s] and not c_cite[s] for s in ids)
    p = mcnemar_one_sided(w, bt)
    citation = PairedTest(CITATION_OMISSION_TRIGGER, p, None, CITATION_ALPHA, p <= CITATION_ALPHA, len(ids),
                          {"worse": float(w), "better": float(bt),
                           "baseline_cases": float(sum(b_cite.values())),
                           "candidate_cases": float(sum(c_cite.values()))})

    # (c) regression sign-off check: each FM code, and each route key, counts on its own
    signoff = signoff_check(base, pooled, ladder)
    if ladder in LADDER_NOTES:
        notes.append(LADDER_NOTES[ladder])
    notes.append(ROUTE_BLIND_NOTE)

    blocked = (any(t.rejected for t in paired.values()) or any(t.rejected for t in fabrication.values())
               or citation.rejected)
    return GateReport(paired=paired, fabrication=fabrication, signoff=tuple(signoff), blocked=blocked,
                      notes=tuple(notes), citation=citation)
