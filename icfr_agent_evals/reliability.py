"""Content-pass^k, the review queue and failure-mode counts, per (sample, epoch).

Pre-registration section 6: reliability is computed after the run from every
(sample, epoch) score in log.samples, never from Inspect's reduced scores (the
mean reducer keeps only the first epoch's metadata, so a review flag raised on a
later epoch would vanish). Inspect's pass_k and pass_at reducers are not used.

The trial evaluate_trial computes is the code-only trial: it succeeds when the
correctness, fabrication, reconciling-items and completeness anchors each score 3
on that epoch, no scorer fires an FM code, and no scorer routes the epoch to
requires_human_review. A sample error, a missing score or a non-numeric value is a
failed trial (the strict form of AS 2315 .40). A sample's content-pass^k is 1 only
when exactly k epochs are present and all k trials succeed. The code-only figure is
the extractor health figure (section 8), never a score. The review-settled,
audit-adjusted and route-blind trials live in trials.py, which reads this module.

Decision D-R20 (2026-11-15): an errored epoch is a format break. evaluate_trial gives
an epoch with a sample error FM-000 first among its failure modes, whatever its scorers
recorded (errored_format_break), so every FM count reads it from here: fm_counts, the
section 10 regression records and sign-off check, the run reports and the review packet.
An epoch with no sample error keeps exactly the codes its scorers recorded, so a log with
no errored epoch gives the same trials and counts as before.

Owner decision D-R9 (2026-11-11): a citation omission no longer lowers the fabrication
anchor, so an epoch whose only flaw is one passes the code-only trial too. The shared
helpers here (scored_before_d_r9, code_found_omissions) let trials.py and regression.py
refuse a log scored before D-R9 and count the omissions from the log alone.

FM-008 (inconsistency under repetition): at least one of a sample's trials
succeeded and its content-pass^k failed. Section 6 computes it on the
review-settled trial, so it moves with the review verdicts: given the run's frozen
review record (record=...), reliability_report and report_from_log read FM-008 from
trials.review_settled_trial and set fm008_basis to "review_settled". Without a
record, fm008 is the code-only reading, labeled fm008_basis "code_only"; that
reading is never the pre-registered FM-008. fm008_code_only always holds the
code-only reading.

FM-007 (over-confident incorrect) is sample-level too, but it is computed by
consistency.py from the canonical answers (all k epochs agree and the modal answer
is wrong), not here. It never enters a per-epoch trial or fm_counts, so it cannot
change content-pass^k: at least one epoch of such a sample fails its trial
(ratified 2026-11-03).

FM-002 counts are reported by direction beside the total (ratified 2026-11-04):
each (sample, epoch) firing FM-002 takes the single direction its scorers record
(understated, overstated, mixed, or unordered for a structured value outside the
enum), "mixed" when they differ, and "unrecorded" when no scorer recorded one, so
the counts by direction sum to fm_counts["FM-002"].

Queue volume (queue_volume) counts distinct (sample, epoch, reason) items and
distinct (sample, epoch) pairs by class, never len(review_queue): the queue keeps
one ReviewRow per routing scorer as the audit trail, so one format break on one
epoch is six rows and faithfulness_anchor repeats the fabrication and reconciling
reasons. Classes: format_break (FM-000), sample_error, narrative_sweep and
code_review (regression.route_key), plus fm_listed for a reason that rides with an
FM code (the section 10(b) fabrication listing), which is not a section 10(c) route.

Run configuration (contract C2): report_from_log refuses (GateError) a log whose
recorded configuration is off the section 5 pins, through roster.check_run_config
on the full log (require_run_config). The template baseline log is out of its scope:
it has no evaluated model, so no ModelEvent and no reasoning_effort.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

TRIAL_ANCHORS: tuple[str, ...] = (
    "correctness_anchor",
    "fabrication_anchor",
    "reconciling_items_anchor",
    "completeness_anchor",
)
FULL_MARKS = 3
PREREGISTERED_K = 4  # section 5: epochs = 4, the k of content-pass^4
SAMPLE_ERROR_SCORER = "sample_error"
FM002 = "FM-002"
FM002_DIRECTION_KEY = "fm002_direction"  # scoring.FM002_DIRECTION_KEY, kept free of the scorer imports
FM002_UNRECORDED = "unrecorded"
FM008_REVIEW_SETTLED = "review_settled"  # trials.REVIEW_SETTLED
FM008_CODE_ONLY = "code_only"            # trials.CODE_ONLY
FORMAT_BREAK_FM = "FM-000"  # scoring.FORMAT_BREAK_FM, kept free of the scorer imports


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
    fm002_direction: str | None = None  # set exactly when FM-002 is in failure_modes


@dataclass(frozen=True)
class ReliabilityReport:
    k: int
    trials: tuple[Trial, ...]
    pass_k: dict[str, int]            # sample id -> 1 or 0
    review_queue: tuple[ReviewRow, ...]
    fm_counts: dict[str, int]         # code -> number of (sample, epoch) pairs firing it
    fm008: tuple[str, ...]            # sample ids
    epoch_anomalies: dict[str, tuple[int, ...]]  # sample id -> epochs seen, when not exactly 1..k
    fm002_by_direction: dict[str, int] = field(default_factory=dict)  # direction -> (sample, epoch) pairs
    fm008_basis: str = FM008_CODE_ONLY      # which trial fm008 reads: "review_settled" or "code_only"
    fm008_code_only: tuple[str, ...] = ()   # FM-008 on the code-only trial, always computed
    settled: Any = None                     # trials.TrialReport of the review-settled trial, with a record

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


def errored_format_break(sample: Any, codes: Iterable[str]) -> tuple[str, ...]:
    """The FM codes of one (sample, epoch): `codes` deduplicated in order, with FM-000
    added first on an epoch with a sample error (decision D-R20: an errored epoch is a
    format break). Idempotent; an epoch with no sample error keeps its codes exactly."""
    codes = tuple(dict.fromkeys(str(c) for c in codes))
    if getattr(sample, "error", None) is None or FORMAT_BREAK_FM in codes:
        return codes
    return (FORMAT_BREAK_FM, *codes)


def evaluate_trial(sample: Any) -> Trial:
    """One (sample, epoch) record: an Inspect EvalSample or anything with id,
    epoch, scores (name -> Score) and error. An epoch with a sample error carries
    FM-000 (decision D-R20, errored_format_break)."""
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

    directions: set[str] = set()
    for name, score in scores.items():
        meta = getattr(score, "metadata", None) or {}
        codes = [str(c) for c in meta.get("failure_modes") or []]
        fms.extend(codes)
        if FM002 in codes and meta.get(FM002_DIRECTION_KEY):
            directions.add(str(meta[FM002_DIRECTION_KEY]))
        if meta.get("requires_human_review"):
            review.append(ReviewRow(sid, epoch, name, tuple(str(r) for r in meta.get("review_reasons") or [])))
    codes = errored_format_break(sample, fms)
    if codes:
        failed.append(f"failure modes {sorted(set(codes))}")
    if any(r.scorer != SAMPLE_ERROR_SCORER for r in review):
        failed.append("routed to requires_human_review")

    direction = None
    if FM002 in fms:
        direction = (directions.pop() if len(directions) == 1 else "mixed") if directions else FM002_UNRECORDED
    return Trial(sid, epoch, not failed, codes, tuple(review), tuple(failed), direction)


# --- Owner decision D-R9 (2026-11-11): citation omissions, read from the log -------------

FABRICATION_ANCHOR = "fabrication_anchor"
D_R9_RESCORE_HINT = ("re-score it at the D-R9 commit with scripts/rescore_log.py "
                     "(owner decision D-R9, 2026-11-11)")


def _fabrication_meta(scores: Any) -> tuple[Any, dict]:
    score = (scores or {}).get(FABRICATION_ANCHOR)
    meta = (getattr(score, "metadata", None) or {}) if score is not None else {}
    details = meta.get("details")
    return score, (details if isinstance(details, dict) else {})


def scored_before_d_r9(scores: Any) -> bool:
    """True when one (sample, epoch)'s scores were written by the scorer before owner
    decision D-R9: a fabrication anchor of 0 with details.citation_omissions filled and
    no FM code and no review reason (the D-R9 scorer scores that 3), or a sweep hit
    without its `in_case_file` flag. Settling such a log would pass the review-found
    omissions and still fail the code-found ones, so trials.py and regression.py
    refuse it."""
    score, details = _fabrication_meta(scores)
    if score is None:
        return False
    meta = getattr(score, "metadata", None) or {}
    value = _numeric(getattr(score, "value", None))
    omitted = details.get("citation_omissions") or []
    if (value == 0 and omitted and not meta.get("failure_modes") and not meta.get("review_reasons")):
        return True
    hits = details.get("sweep_hits") or []
    return any(isinstance(h, dict) and "in_case_file" not in h for h in hits)


def code_found_omissions(scores: Any) -> dict[str, list[str]]:
    """The citation omissions code found on one (sample, epoch), by source (D-R9,
    section 8): `p12` (a reference number or wallet label P12 matches), `case_file`
    (any other sweep token the case file holds exactly) and `wallet_label` (the
    add-only wallet-label matcher, choice 5). A field the log lacks reads as empty."""
    _, details = _fabrication_meta(scores)

    def listed(key: str) -> list[str]:
        value = details.get(key) or []
        return [str(v) for v in value] if isinstance(value, (list, tuple)) else []

    return {"p12": listed("citation_omissions"), "case_file": listed("case_file_names_uncited"),
            "wallet_label": listed("wallet_label_mentions_uncited")}


def citation_omission_count(scores: Any) -> int:
    """Every omitted token code_found_omissions lists on one (sample, epoch)."""
    return sum(len(v) for v in code_found_omissions(scores).values())


def reliability_report(samples: Iterable[Any], k: int, *, record: Any = None, run_id: str | None = None,
                       require_complete: bool = True) -> ReliabilityReport:
    """Every (sample, epoch) record, never reduced scores. With the run's frozen
    review record, FM-008 reads the review-settled trial (trials.py); run_id names
    the worker run in a record that holds several (trials.resolve_run_id)."""
    samples = list(samples)
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
    by_direction = Counter(t.fm002_direction for t in trials if t.fm002_direction is not None)
    assert sum(by_direction.values()) == fm_counts.get(FM002, 0), "FM-002 by direction must sum to its total"
    queue = tuple(row for t in sorted(trials, key=lambda t: (t.sample_id, t.epoch)) for row in t.review)
    settled = None
    fm008_of_record, basis = tuple(fm008), FM008_CODE_ONLY
    if record is not None:
        from .trials import review_settled_trial  # trials imports this module

        settled = review_settled_trial(samples, record, k=k, run_id=run_id, require_complete=require_complete)
        fm008_of_record, basis = settled.fm008, FM008_REVIEW_SETTLED
    return ReliabilityReport(
        k=k,
        trials=tuple(trials),
        pass_k=pass_k,
        review_queue=queue,
        fm_counts=dict(sorted(fm_counts.items())),
        fm008=fm008_of_record,
        epoch_anomalies=anomalies,
        fm002_by_direction=dict(sorted(by_direction.items())),
        fm008_basis=basis,
        fm008_code_only=tuple(fm008),
        settled=settled,
    )


# Queue classes, in print order. The two route classes match regression.SWEEP_ROUTE
# and regression.CODE_ROUTE (checked there); fm_listed is the reason route_key maps
# to None other than a format break.
QUEUE_FORMAT_BREAK = "format_break"
QUEUE_SAMPLE_ERROR = "sample_error"
QUEUE_SWEEP = "narrative_sweep"
QUEUE_CODE = "code_review"
QUEUE_FM_LISTED = "fm_listed"
QUEUE_CLASSES: tuple[str, ...] = (QUEUE_FORMAT_BREAK, QUEUE_SAMPLE_ERROR, QUEUE_SWEEP, QUEUE_CODE, QUEUE_FM_LISTED)
QUEUE_UNSTATED_REASON = "review flagged with no stated reason"  # regression.UNSTATED_REASON
_FORMAT_BREAK_PREFIX = "format break:"


@dataclass(frozen=True)
class QueueItem:
    """One distinct (sample, epoch, reason) review item, with every scorer that raised it."""
    sample_id: str
    epoch: int
    reason: str
    queue_class: str             # one of QUEUE_CLASSES
    route_key: str | None        # regression.route_key, None for format_break, sample_error, fm_listed
    scorers: tuple[str, ...]


@dataclass(frozen=True)
class QueueVolume:
    """Distinct review items and distinct (sample, epoch) pairs per queue class. A
    pair can sit in several classes, so total_sample_epochs is counted on its own."""
    items: dict[str, int]          # class -> distinct (sample, epoch, reason)
    sample_epochs: dict[str, int]  # class -> distinct (sample, epoch)
    total_items: int
    total_sample_epochs: int
    scorer_rows: int               # len(review_queue), the audit trail, never the volume


def _queue_class(row: ReviewRow, reason: str) -> tuple[str, str | None]:
    if row.scorer == SAMPLE_ERROR_SCORER:
        return QUEUE_SAMPLE_ERROR, None
    if reason.startswith(_FORMAT_BREAK_PREFIX):
        return QUEUE_FORMAT_BREAK, None
    from .regression import route_class, route_key  # regression imports this module

    key = route_key(reason)
    if key is None:
        return QUEUE_FM_LISTED, None
    return route_class(key), key


def queue_items(report: ReliabilityReport) -> tuple[QueueItem, ...]:
    """The review queue as distinct (sample, epoch, reason) items, in queue order.
    A row with no stated reason is one item under QUEUE_UNSTATED_REASON."""
    found: dict[tuple[str, int, str], tuple[str, str | None, list[str]]] = {}
    for row in report.review_queue:
        for reason in row.reasons or (QUEUE_UNSTATED_REASON,):
            ident = (row.sample_id, row.epoch, reason)
            if ident not in found:
                found[ident] = (*_queue_class(row, reason), [])
            if row.scorer not in found[ident][2]:
                found[ident][2].append(row.scorer)
    return tuple(QueueItem(sid, ep, reason, cls, key, tuple(scorers))
                 for (sid, ep, reason), (cls, key, scorers) in found.items())


def queue_volume(report: ReliabilityReport) -> QueueVolume:
    """Section 6 review volume by class (F2-07): distinct items and distinct
    (sample, epoch) pairs, so a scorer that repeats a reason never inflates it."""
    items = queue_items(report)
    per_items = Counter(i.queue_class for i in items)
    per_pairs = {cls: len({(i.sample_id, i.epoch) for i in items if i.queue_class == cls}) for cls in QUEUE_CLASSES}
    return QueueVolume(
        items={cls: per_items.get(cls, 0) for cls in QUEUE_CLASSES},
        sample_epochs=per_pairs,
        total_items=len(items),
        total_sample_epochs=len({(i.sample_id, i.epoch) for i in items}),
        scorer_rows=len(report.review_queue),
    )


def format_queue_volume(volume: QueueVolume) -> str:
    """One line: distinct items and (sample, epoch) pairs per class."""
    parts = "; ".join(f"{cls} {volume.items[cls]} items / {volume.sample_epochs[cls]} epochs" for cls in QUEUE_CLASSES)
    return (f"Review queue: {volume.total_items} distinct (sample, epoch, reason) items on "
            f"{volume.total_sample_epochs} distinct (sample, epoch) pairs; {parts}")


def format_fm_counts(report: ReliabilityReport) -> str:
    """FM counts per (sample, epoch), FM-002 by direction beside its total."""
    if not report.fm_counts:
        return "FM counts: none"
    parts = []
    for code, n in report.fm_counts.items():
        if code == FM002:
            split = ", ".join(f"{d} {c}" for d, c in report.fm002_by_direction.items())
            parts.append(f"{code} {n} ({split})")
        else:
            parts.append(f"{code} {n}")
    return "FM counts: " + "; ".join(parts)


# baseline.py's @task name. Its log has no evaluated model (none/none): no ModelEvent
# and no reasoning_effort, so roster.check_run_config would always refuse it.
TEMPLATE_BASELINE_TASK = "variance_commentary_template_baseline"


def is_template_baseline_log(log: Any) -> bool:
    """True for a log of the template baseline task (registry name, package prefix ignored)."""
    task = getattr(getattr(log, "eval", None), "task", None)
    return task is not None and str(task).rsplit("/", 1)[-1] == TEMPLATE_BASELINE_TASK


def require_run_config(log: Any, *, expected_samples: int | None = None,
                       expected_epochs: int | None = PREREGISTERED_K, allow_batch: bool = True,
                       max_tokens_pin: int | None = None) -> dict | None:
    """Contract C2 for the readers: roster.check_run_config on a full log (samples and
    events read), its RunConfigError raised as GateError with the message kept.
    The template baseline log is out of scope and returns None unchecked."""
    if is_template_baseline_log(log):
        return None
    from .roster import RunConfigError, check_run_config  # roster stays out of the module imports

    try:
        return check_run_config(log, expected_samples=expected_samples, expected_epochs=expected_epochs,
                                allow_batch=allow_batch, max_tokens_pin=max_tokens_pin)
    except RunConfigError as e:
        raise GateError(str(e)) from e


def report_from_log(log: Any, k: int | None = None, *, run_config: bool = True,
                    max_tokens_pin: int | None = None, record: Any = None, run_id: str | None = None,
                    require_complete: bool = True) -> ReliabilityReport:
    """Report from an EvalLog read with its samples (header_only=False). k
    defaults to the run's configured epochs. With run_config (the default) the log
    must pass require_run_config at k epochs (PREREGISTERED_K when k is None);
    run_config=False is for hand-built logs that make no model call. With the run's
    frozen review record, FM-008 reads the review-settled trial; the log's eval id
    finds its worker run in the record unless run_id is given."""
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    ran = int(log.eval.config.epochs or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if declared is not None and int(declared) != ran:
        # an --epochs override records the run's epochs in the config while
        # task_args keeps the task's own, so a gated k = 4 could silently become k = 1
        raise GateError(f"run epochs {ran} differ from the task argument epochs {declared}")
    if run_config:
        require_run_config(log, expected_epochs=PREREGISTERED_K if k is None else k, max_tokens_pin=max_tokens_pin)
    if k is None:
        k = ran
    if record is not None:
        from .trials import resolve_run_id  # trials imports this module

        run_id = resolve_run_id(record, log, run_id)
    return reliability_report(log.samples, k, record=record, run_id=run_id, require_complete=require_complete)


def content_pass_figures(log: Any, record: Any = None, *, k: int | None = None, run_id: str | None = None,
                         require_complete: bool = True) -> dict:
    """trials.figures: the review-settled, audit-adjusted and route-blind content-pass^4
    and the code-only extractor health figure, each with its Clopper-Pearson interval,
    and the anchor means review-settled and audit-adjusted side by side."""
    from .trials import figures  # trials imports this module

    return figures(log, record, k=k, run_id=run_id, require_complete=require_complete)


def assert_gate(report: ReliabilityReport, expected_samples: int, expected_k: int = PREREGISTERED_K) -> None:
    """Section 6: scored samples must equal the split's sample count, the report's
    k must be the pre-registered k, and every sample must carry exactly k epochs.
    Any mismatch fails the gate."""
    if report.k != expected_k:
        raise GateError(f"report k {report.k} != pre-registered k {expected_k}")
    if report.scored_samples != expected_samples:
        raise GateError(f"scored samples {report.scored_samples} != split sample count {expected_samples}")
    if report.epoch_anomalies:
        raise GateError(f"samples without exactly {report.k} epochs: {report.epoch_anomalies}")
