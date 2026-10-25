"""Content-pass^k, the review queue and failure-mode counts, per (sample, epoch).

Pre-registration section 6: reliability is computed after the run from every
(sample, epoch) score in log.samples, never from Inspect's reduced scores (the
mean reducer keeps only the first epoch's metadata, so a review flag raised on a
later epoch would vanish). Inspect's pass_k and pass_at reducers are not used.

A content trial succeeds when the correctness, fabrication, reconciling-items and
completeness anchors each score 3 on that epoch, no scorer fires an FM code, and
no scorer routes the epoch to requires_human_review. A sample error, a missing
score or a non-numeric value is a failed trial (the strict form of AS 2315 .40).
A sample's content-pass^k is 1 only when exactly k epochs are present and all
k trials succeed.

FM-008 (inconsistency under repetition): at least one of a sample's trials
succeeded and its content-pass^k failed.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

TRIAL_ANCHORS: tuple[str, ...] = (
    "correctness_anchor",
    "fabrication_anchor",
    "reconciling_items_anchor",
    "completeness_anchor",
)
FULL_MARKS = 3
SAMPLE_ERROR_SCORER = "sample_error"


class GateError(AssertionError):
    """A gated run whose scored samples do not match the split."""


@dataclass(frozen=True)
class ReviewRow:
    sample_id: str
    epoch: int
    scorer: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Trial:
    sample_id: str
    epoch: int
    success: bool
    failure_modes: tuple[str, ...]
    review: tuple[ReviewRow, ...]
    failed_because: tuple[str, ...]


@dataclass(frozen=True)
class ReliabilityReport:
    k: int
    trials: tuple[Trial, ...]
    pass_k: dict[str, int]            # sample id -> 1 or 0
    review_queue: tuple[ReviewRow, ...]
    fm_counts: dict[str, int]         # code -> number of (sample, epoch) pairs firing it
    fm008: tuple[str, ...]            # sample ids
    epoch_anomalies: dict[str, tuple[int, ...]]  # sample id -> epochs seen, when not exactly 1..k

    @property
    def scored_samples(self) -> int:
        return len(self.pass_k)

    @property
    def content_pass_k_rate(self) -> float:
        return sum(self.pass_k.values()) / len(self.pass_k) if self.pass_k else 0.0


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def evaluate_trial(sample: Any) -> Trial:
    """One (sample, epoch) record: an Inspect EvalSample or anything with id,
    epoch, scores (name -> Score) and error."""
    sid, epoch = str(sample.id), int(sample.epoch)
    failed: list[str] = []
    fms: list[str] = []
    review: list[ReviewRow] = []

    if getattr(sample, "error", None) is not None:
        failed.append("sample error")
        review.append(ReviewRow(sid, epoch, SAMPLE_ERROR_SCORER, (str(getattr(sample.error, "message", sample.error)),)))
    scores = getattr(sample, "scores", None) or {}

    for name in TRIAL_ANCHORS:
        score = scores.get(name)
        value = _numeric(getattr(score, "value", None)) if score is not None else None
        if score is None:
            failed.append(f"{name} unscored")
        elif value is None:
            failed.append(f"{name} non-numeric value {getattr(score, 'value', None)!r}")
        elif value != FULL_MARKS:
            failed.append(f"{name} scored {value:g}")

    for name, score in scores.items():
        meta = getattr(score, "metadata", None) or {}
        codes = [str(c) for c in meta.get("failure_modes") or []]
        fms.extend(codes)
        if meta.get("requires_human_review"):
            review.append(ReviewRow(sid, epoch, name, tuple(str(r) for r in meta.get("review_reasons") or [])))
    if fms:
        failed.append(f"failure modes {sorted(set(fms))}")
    if any(r.scorer != SAMPLE_ERROR_SCORER for r in review):
        failed.append("routed to requires_human_review")

    return Trial(sid, epoch, not failed, tuple(dict.fromkeys(fms)), tuple(review), tuple(failed))


def reliability_report(samples: Iterable[Any], k: int) -> ReliabilityReport:
    """Every (sample, epoch) record, never reduced scores."""
    trials = [evaluate_trial(s) for s in samples]
    by_sample: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_sample[t.sample_id].append(t)

    pass_k: dict[str, int] = {}
    anomalies: dict[str, tuple[int, ...]] = {}
    fm008: list[str] = []
    for sid, ts in sorted(by_sample.items()):
        epochs = tuple(sorted(t.epoch for t in ts))
        complete = epochs == tuple(range(1, k + 1))
        if not complete:
            anomalies[sid] = epochs
        passed = complete and all(t.success for t in ts)
        pass_k[sid] = int(passed)
        if not passed and any(t.success for t in ts):
            fm008.append(sid)

    fm_counts = Counter(code for t in trials for code in t.failure_modes)
    queue = tuple(row for t in sorted(trials, key=lambda t: (t.sample_id, t.epoch)) for row in t.review)
    return ReliabilityReport(
        k=k,
        trials=tuple(trials),
        pass_k=pass_k,
        review_queue=queue,
        fm_counts=dict(sorted(fm_counts.items())),
        fm008=tuple(fm008),
        epoch_anomalies=anomalies,
    )


def report_from_log(log: Any, k: int | None = None) -> ReliabilityReport:
    """Report from an EvalLog read with its samples (header_only=False). k
    defaults to the run's configured epochs."""
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    if k is None:
        k = int(log.eval.config.epochs or 1)
    return reliability_report(log.samples, k)


def assert_gate(report: ReliabilityReport, expected_samples: int) -> None:
    """Section 6: scored samples must equal the split's sample count, and every
    sample must carry exactly k epochs. Any mismatch fails the gate."""
    if report.scored_samples != expected_samples:
        raise GateError(f"scored samples {report.scored_samples} != split sample count {expected_samples}")
    if report.epoch_anomalies:
        raise GateError(f"samples without exactly {report.k} epochs: {report.epoch_anomalies}")
