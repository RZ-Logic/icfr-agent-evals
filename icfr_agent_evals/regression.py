"""CI regression gates: candidate runs paired by sample id against the baseline run.

Pre-registration section 10, on per (sample, epoch) records, never on Inspect's
reduced scores (section 6):
  (a) paired rule: a one-sided paired t-test on per-sample epoch-mean differences
      for each of the five rubric dimensions, and an exact one-sided McNemar test
      on per-sample content-pass^4, Holm-corrected as one family of six at
      FAMILY_ALPHA. A dimension no record scores (Calibration, reported and not
      scored in Phase 1, and Robustness) enters at p = 1 and still counts in the family.
  (b) fabrication rule: FM-005 and FM-015 each get a one-sided exact McNemar test
      on per-case indicators (the code fires on any epoch) at FABRICATION_ALPHA,
      outside the Holm family.
  (c) regression sign-off check: a case triggers when a listed FM code, or a
      code-check review route, fires on more epochs in the
      candidate than in the baseline (R1), or on two or more extra epochs (R2,
      with FM-005, FM-015 and every narrative-sweep route kept on R1). Review
      routes are counted per reason (ratified 2026-11-03): each review reason
      (route_key) counts on its own, even on an epoch that also fired an FM code
      or carried another reason. FM-007 is
      sample-level (consistency.py) and is not a sign-off code. Sign-off cases
      never block; they wait for a recorded human sign-off.
  (d) re-runs pool: every candidate run's epochs count together. The candidate's
      epoch means use all pooled epochs, it passes content-pass^4 only if every
      pooled epoch succeeds, its fabrication indicator is any pooled epoch, and its
      sign-off counts are raw pooled counts against the baseline's count of k.
      Each asymmetry runs against the candidate.
  Every candidate run must hold the baseline's full sample id set and the same
  dataset content (dataset_version and both dataset SHA-256s, plus the other
  MATCHED_CONTENT_FIELDS), and a truncated (--limit or --sample-id) run is
  refused, so the comparison never runs on an overlap (ratified 2026-11-03).

Mapping from an Inspect (sample, epoch) record (records_from_samples):
  - success and failure_modes come from reliability.evaluate_trial, unchanged;
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
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .reliability import FM002, FM002_UNRECORDED, PREREGISTERED_K, GateError, _numeric, evaluate_trial
from .rubric import WIRED_SCORERS
from .scoring import FABRICATION_REVIEW, SweepHit

DIMENSIONS: tuple[str, ...] = ("correctness", "faithfulness", "calibration", "completeness", "robustness")
PASS4 = "content_pass4"
FAMILY_ALPHA = 0.05          # Holm family-wise alpha, section 10(a)
FABRICATION_ALPHA = 0.025    # section 10(b), per code, outside the Holm family
FABRICATION_CODES: tuple[str, ...] = ("FM-005", "FM-015")
SIGNOFF_CODES: tuple[str, ...] = (
    "FM-000", "FM-001", "FM-002", "FM-003", "FM-004", "FM-005", "FM-006", "FM-015", "FM-016",
)
SWEEP_ROUTE = "narrative_sweep"   # route class: holds unconfirmed fabrications, stays on R1 under R2
CODE_ROUTE = "code_review"        # route class: any other code-check review reason
ROUTE_CLASSES: tuple[str, ...] = (SWEEP_ROUTE, CODE_ROUTE)
# Trigger families: the FM codes, then the two route classes. A route trigger is
# one route key ("<class>: <reason key>"), each counted on its own (route_key).
SIGNOFF_TRIGGERS: tuple[str, ...] = (*SIGNOFF_CODES, *ROUTE_CLASSES)
R1_ONLY: frozenset[str] = frozenset({"FM-005", "FM-015", SWEEP_ROUTE})
LADDERS: tuple[str, ...] = ("R1", "R2")
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
_SWEEP_KINDS = ("non_latin_letter", "format_character", "custodian_vocabulary", "organization_suffix",
                "capitalized_run", "reference", "identifier")
_SENTINEL = "\x00sentinel\x00"
SWEEP_REASON_PREFIXES: tuple[str, ...] = tuple(
    SweepHit(_SENTINEL, kind).reason.split(repr(_SENTINEL))[0] for kind in _SWEEP_KINDS
)
FORMAT_BREAK_PREFIX = "format break:"
UNSTATED_REASON = "review flagged with no stated reason"
_NARRATIVE_KIND_RE = re.compile(r"^([a-z][a-z0-9_]*)(?::\s|$)")  # narrative.py writes "kind: matched text"
_TYPE_MISMATCH_RE = re.compile(r"^reconciling_items\[\d+\] .* type .* differs from (.+)$", re.DOTALL)
_EXTRACTOR_RAISED_RE = re.compile(r"^(narrative extractor raised [A-Za-z_][A-Za-z0-9_.]*)")


@dataclass(frozen=True)
class EpochRecord:
    sample_id: str
    epoch: int
    dims: Mapping[str, float]          # dimension -> 0..3 on this epoch; an unwired dimension is absent
    success: bool                      # content trial success, as reliability.evaluate_trial decides
    failure_modes: tuple[str, ...]     # FM codes fired on this epoch, deduplicated
    review_routes: tuple[str, ...]     # route keys (route_key), deduplicated, sweep keys first
    fm002_direction: str | None = None  # reliability.Trial.fm002_direction, set when FM-002 fired


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
    rule: str                    # "R1" or "R2"
    # FM-002 only: epochs by direction beside the counts (ratified 2026-11-04); reported, not gated
    baseline_by_direction: Mapping[str, int] = field(default_factory=dict, compare=False)
    candidate_by_direction: Mapping[str, int] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class GateReport:
    paired: Mapping[str, PairedTest]        # the five DIMENSIONS and PASS4, m = 6 always
    fabrication: Mapping[str, PairedTest]   # FM-005, FM-015
    signoff: tuple[SignoffCase, ...]
    blocked: bool                           # any Holm rejection in paired or any fabrication rejection
    notes: tuple[str, ...]


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


def record_from_sample(sample: Any) -> EpochRecord:
    """One (sample, epoch) record: an Inspect EvalSample or anything with id,
    epoch, scores (name -> Score) and error."""
    trial = evaluate_trial(sample)
    scores = getattr(sample, "scores", None) or {}
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


def records_from_log(log_or_path: Any) -> list[EpochRecord]:
    """Records from an EvalLog (or a path to one) read with its samples."""
    log = load_log(log_or_path)
    refuse_evaluation_split(log)
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    refuse_partial_run(log)
    ran = int(log.eval.config.epochs or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if declared is not None and int(declared) != ran:
        raise GateError(f"run epochs {ran} differ from the task argument epochs {declared}")
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

    base_pass = {sid: all(r.success for r in base[sid]) for sid in ids}
    cand_pass = {sid: all(r.success for r in pooled[sid]) for sid in ids}
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

    # (c) regression sign-off check: each FM code, and each route key, counts on its own
    def fired(r: EpochRecord, trigger: str) -> bool:
        return trigger in r.failure_modes or trigger in r.review_routes

    def by_direction(records: Iterable[EpochRecord]) -> dict[str, int]:
        found = Counter(r.fm002_direction or FM002_UNRECORDED for r in records if FM002 in r.failure_modes)
        return dict(sorted(found.items()))

    signoff: list[SignoffCase] = []
    for sid in ids:
        keys = {key for r in (*base[sid], *pooled[sid]) for key in r.review_routes}
        for trigger in (*SIGNOFF_CODES, *sorted(keys, key=_route_order)):
            bc = sum(fired(r, trigger) for r in base[sid])
            cc = sum(fired(r, trigger) for r in pooled[sid])
            sweep = trigger in keys and route_class(trigger) == SWEEP_ROUTE
            rule = "R1" if ladder == "R1" or trigger in R1_ONLY or sweep else "R2"
            if (cc > bc) if rule == "R1" else (cc - bc >= 2):
                if trigger == FM002:
                    signoff.append(SignoffCase(sid, trigger, bc, cc, rule,
                                               by_direction(base[sid]), by_direction(pooled[sid])))
                else:
                    signoff.append(SignoffCase(sid, trigger, bc, cc, rule))
    if ladder == "R2":
        notes.append("ladder R2: a single-epoch rise triggers only on FM-005, FM-015 and the narrative-sweep routes")

    blocked = any(t.rejected for t in paired.values()) or any(t.rejected for t in fabrication.values())
    return GateReport(paired=paired, fabrication=fabrication, signoff=tuple(signoff), blocked=blocked, notes=tuple(notes))
