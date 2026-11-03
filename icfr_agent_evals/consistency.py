"""Sample Consistency across the k = 4 epoch resamples, per (sample, epoch).

Pre-registration section 6 (Calibration): Sample Consistency, not verbalized
confidence; the consistency measurement is deterministic, and discrimination and
calibration are tracked as separate properties. Section 2 H4 predicts the signal
is uninformative on the synthetic subset and informative on the difficulty-bearing
subsets, and section 8 reports calibration curves on both. This module is the
deterministic measurement only. calibration.semantic_agreement (the grader's LLM
agreement judgment) is a separate LLM-graded sub-dimension and is not built here.

Like reliability.py, everything is computed after the run from every (sample,
epoch) record in log.samples, never from reduced scores, and a sample without
exactly k epochs fails the gate (GateError).

Per epoch, a canonical answer is built from the parsed fields that decide
correctness, with scoring.py's own normalizers: direction and band
(normalize_enum), asset (normalize_asset), period (normalize_period), the
variance amount rounded half-up to whole dollars with its sign kept, and the cited
reconciling items as a sorted multiset of (canonical_reference, unsigned amount in
whole dollars) pairs, so order is ignored, a repeated citation stays visible and a
different item amount (FM-015 on a mismatch) is a different answer; the item sign
is dropped because check_reconciling_items compares the amount unsigned. Rounding
runs in a local decimal context wide enough for any finite amount the parser
accepts, so a huge amount is a wrong answer, never a crash. Rounding, not "within $1", is
used because a tolerance is not transitive and cannot define the classes a mode
needs; two amounts a cent apart across a 50-cent boundary therefore disagree.
A format break, a sample error or a missing completion is a parse failure, its
own value that agrees with nothing, another parse failure included.

An epoch's answer is correct when the Correctness anchor of record is 3 with no FM
code (check_correctness: magnitude within $1, direction, band, no FM-016, and no
narrative-consistency code, since any contradiction sets Correctness to 0 under
section 6) and check_reconciling_items is 3 with no FM code. The narrative is not
part of the canonical answer, so epochs that differ only in a contradicting
narrative share a class and that class is wrong. Review reasons do not decide
correctness here; reliability.py counts them.

Per sample: the modal answer and agreement = epochs agreeing with the mode / k
(1/k to 1). Epochs that share a canonical answer can still differ in correctness
(raw amounts on either side of the $1 tolerance round to one dollar), so the
modal answer is correct only when every epoch holding it is correct. Mode
tie-break, deterministic and conservative: among classes with the highest count,
an incorrect class is chosen before a correct one, then the class first seen at
the lowest epoch, so a tie is never resolved in the model's favor.

Per run, overall and per subset (synthetic: case_kind standard; difficulty-
bearing: edge or adversarial): discrimination as ROC AUC of agreement against
modal correctness (Mann-Whitney, ties count one half; None when either class is
empty) and a calibration table by agreement level (n, empirical accuracy, gap =
agreement level minus accuracy, so a positive gap is over-confidence).

Unanimous-but-wrong samples are listed as FM-007 candidates and reported only. No
scorer fires FM-007 from this module, and the Calibration dimension keeps no 0 to
3 score: the pre-registration has no calibration anchor scale yet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Iterable, Sequence

from .casefile import CaseTruth, canonical_reference, nfkc_spaced, truth_of
from .reliability import PREREGISTERED_K, GateError
from .scoring import (
    SCORE_MAX,
    check_correctness,
    check_identity_fields,
    check_reconciling_items,
    normalize_asset,
    normalize_enum,
    normalize_period,
    parse_worker_output,
)

SYNTHETIC = "synthetic"
DIFFICULTY_BEARING = "difficulty_bearing"
SUBSET_OF_KIND: dict[str, str] = {
    "standard": SYNTHETIC,
    "edge": DIFFICULTY_BEARING,
    "adversarial": DIFFICULTY_BEARING,
}
SUBSETS: tuple[str, ...] = (SYNTHETIC, DIFFICULTY_BEARING)
FM007_NOTE = "FM-007 candidate (unanimous, modal answer wrong): reported only, no scorer fires FM-007"
CALIBRATION_SCORE_NOTE = "Calibration dimension unscored: no 0 to 3 anchor scale is pre-registered yet"


# --- Canonical answer ------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalAnswer:
    """The correctness-deciding fields of one parsed output, in comparison form."""

    variance_direction: str
    materiality_band: str
    variance_amount_whole_usd: int
    asset: str
    period: str
    reconciling_references: tuple[tuple[str, int], ...]  # sorted multiset of (reference, whole dollars)


def whole_dollars(value: Any) -> int:
    """Half-up rounding to whole dollars, sign kept (a negative amount is a different answer).
    The local context holds every integer digit, so no finite amount raises."""
    d = Decimal(str(value))
    with localcontext() as ctx:
        ctx.prec = max(ctx.prec, d.adjusted() + 2)
        return int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _period_form(value: Any) -> str:
    p = normalize_period(value)
    return p if p is not None else "unparsed:" + nfkc_spaced(str(value))


def canonical_answer(parsed: dict) -> CanonicalAnswer:
    """parsed: a schema-valid output from scoring.parse_worker_output."""
    return CanonicalAnswer(
        variance_direction=normalize_enum(parsed["variance_direction"]),
        materiality_band=normalize_enum(parsed["materiality_band"]),
        variance_amount_whole_usd=whole_dollars(parsed["variance_amount_usd"]),
        asset=normalize_asset(parsed["asset"]),
        period=_period_form(parsed["period"]),
        reconciling_references=tuple(sorted(
            (canonical_reference(e["reference"]), abs(whole_dollars(e["amount_usd"])))
            for e in parsed["reconciling_items"]
        )),
    )


def answer_correct(parsed: dict, truth: CaseTruth) -> tuple[bool, tuple[str, ...]]:
    """(correct, reasons): Correctness anchor of record and reconciling items, each 3 with no FM code."""
    reasons: list[str] = []
    cor = check_correctness(parsed, truth)
    structured = cor.details.get("structured_points")
    if structured != SCORE_MAX:
        reasons.append(f"magnitude, direction or band: structured points {structured}/{SCORE_MAX}")
    identity = check_identity_fields(parsed, truth)
    reasons.extend(f"{fm} on asset or period" for fm in identity)
    # the anchor of record also carries narrative-consistency codes (section 6)
    if cor.points != SCORE_MAX or cor.failure_modes:
        reasons.append(f"correctness anchor {cor.points}/{SCORE_MAX} {list(cor.failure_modes)}")
    rec = check_reconciling_items(parsed, truth)
    if rec.points != SCORE_MAX or rec.failure_modes:
        reasons.append(f"reconciling items {rec.points}/{SCORE_MAX} {list(rec.failure_modes) or ''}".rstrip())
    return not reasons, tuple(reasons)


# --- Per epoch ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochAnswer:
    sample_id: str
    epoch: int
    answer: CanonicalAnswer | None   # None: parse failure, agrees with nothing
    parse_error: str | None
    correct: bool
    reasons: tuple[str, ...]

    @property
    def parse_failure(self) -> bool:
        return self.answer is None


def _completion(sample: Any) -> str | None:
    output = getattr(sample, "output", None)
    return getattr(output, "completion", None) if output is not None else None


def read_epoch(sample: Any, truth: CaseTruth | None = None) -> EpochAnswer:
    """One (sample, epoch) record: an Inspect EvalSample or anything with id, epoch,
    metadata (carrying the truth), output.completion and error."""
    sid, epoch = str(sample.id), int(sample.epoch)
    error = getattr(sample, "error", None)
    if error is not None:
        msg = f"sample error: {getattr(error, 'message', error)}"
        return EpochAnswer(sid, epoch, None, msg, False, (msg,))
    parse = parse_worker_output(_completion(sample))
    if not parse.ok:
        msg = f"format break: {parse.error}"
        return EpochAnswer(sid, epoch, None, msg, False, (msg,))
    truth = truth if truth is not None else truth_of(sample.metadata)
    correct, reasons = answer_correct(parse.parsed, truth)
    return EpochAnswer(sid, epoch, canonical_answer(parse.parsed), None, correct, reasons)


# --- Per sample --------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleConsistency:
    sample_id: str
    case_kind: str
    category: str | None
    subset: str
    k: int
    epochs: tuple[EpochAnswer, ...]   # sorted by epoch
    modal_answer: CanonicalAnswer | None
    modal_epochs: tuple[int, ...]
    modal_correct: bool
    tie: bool                         # another class shared the modal count

    @property
    def modal_count(self) -> int:
        return len(self.modal_epochs)

    @property
    def agreement(self) -> float:
        return self.modal_count / self.k

    @property
    def unanimous(self) -> bool:
        return self.modal_answer is not None and self.modal_count == self.k


def _classes(answers: Sequence[EpochAnswer]) -> list[list[EpochAnswer]]:
    groups: dict[Any, list[EpochAnswer]] = {}
    for a in sorted(answers, key=lambda a: a.epoch):
        key = ("parse_failure", a.epoch) if a.answer is None else a.answer
        groups.setdefault(key, []).append(a)
    return list(groups.values())


def modal_class(answers: Sequence[EpochAnswer]) -> tuple[list[EpochAnswer], bool]:
    """(modal class, tie). Highest count; ties go to an incorrect class, then to
    the class first seen at the lowest epoch."""
    if not answers:
        raise ValueError("no epochs to take a mode over")
    groups = _classes(answers)
    top = max(len(g) for g in groups)
    tied = [g for g in groups if len(g) == top]
    chosen = min(tied, key=lambda g: (all(a.correct for a in g), g[0].epoch))
    return chosen, len(tied) > 1


def sample_consistency(answers: Sequence[EpochAnswer], truth: CaseTruth, k: int) -> SampleConsistency:
    """Agreement and modal correctness for one sample's epochs; exactly epochs 1..k."""
    epochs = tuple(sorted(answers, key=lambda a: a.epoch))
    seen = tuple(a.epoch for a in epochs)
    if seen != tuple(range(1, k + 1)):
        raise GateError(f"sample {truth.case_id} has epochs {seen}, not exactly 1..{k}")
    chosen, tie = modal_class(epochs)
    if truth.case_kind not in SUBSET_OF_KIND:
        raise ValueError(f"unknown case_kind {truth.case_kind!r}")
    return SampleConsistency(
        sample_id=epochs[0].sample_id,
        case_kind=truth.case_kind,
        category=truth.category,
        subset=SUBSET_OF_KIND[truth.case_kind],
        k=k,
        epochs=epochs,
        modal_answer=chosen[0].answer,
        modal_epochs=tuple(a.epoch for a in chosen),
        modal_correct=all(a.correct for a in chosen),
        tie=tie,
    )


# --- Discrimination and calibration ------------------------------------------------------

@dataclass(frozen=True)
class Discrimination:
    auc: float | None
    n_correct: int
    n_wrong: int
    note: str | None = None


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> Discrimination:
    """ROC AUC of a score for a binary label by Mann-Whitney: the share of
    (correct, wrong) pairs in which the correct one scores higher, ties counted
    one half. None, with a note, when either class is empty."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        empty = "correct" if not pos else "wrong"
        return Discrimination(None, len(pos), len(neg), f"AUC undefined: no {empty} samples")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return Discrimination(wins / (len(pos) * len(neg)), len(pos), len(neg))


@dataclass(frozen=True)
class CalibrationRow:
    agreement: float
    n: int
    correct: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def gap(self) -> float | None:
        """Agreement level minus empirical accuracy; positive is over-confidence."""
        return self.agreement - self.accuracy if self.n else None


def calibration_table(samples: Iterable[SampleConsistency], k: int) -> tuple[CalibrationRow, ...]:
    """One row per attainable agreement level m/k, m = 1..k, empty levels included."""
    n: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for s in samples:
        n[s.modal_count] += 1
        correct[s.modal_count] += int(s.modal_correct)
    return tuple(CalibrationRow(m / k, n[m], correct[m]) for m in range(1, k + 1))


@dataclass(frozen=True)
class SubsetSummary:
    name: str
    n: int
    accuracy: float | None            # share of samples whose modal answer is correct
    mean_agreement: float | None
    discrimination: Discrimination
    calibration: tuple[CalibrationRow, ...]


def summarize(name: str, samples: Sequence[SampleConsistency], k: int) -> SubsetSummary:
    n = len(samples)
    return SubsetSummary(
        name=name,
        n=n,
        accuracy=sum(s.modal_correct for s in samples) / n if n else None,
        mean_agreement=sum(s.agreement for s in samples) / n if n else None,
        discrimination=roc_auc([s.agreement for s in samples], [s.modal_correct for s in samples]),
        calibration=calibration_table(samples, k),
    )


# --- Per run -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsistencyReport:
    k: int
    samples: tuple[SampleConsistency, ...]       # sorted by sample id
    overall: SubsetSummary
    by_subset: dict[str, SubsetSummary]
    fm007_candidates: tuple[str, ...]            # sample ids, reported only

    @property
    def scored_samples(self) -> int:
        return len(self.samples)

    def sample(self, sample_id: str) -> SampleConsistency:
        return next(s for s in self.samples if s.sample_id == sample_id)


def consistency_report(samples: Iterable[Any], k: int) -> ConsistencyReport:
    """Every (sample, epoch) record, never reduced scores. A sample without exactly
    epochs 1..k raises GateError, because agreement over k is then undefined."""
    by_sample: dict[str, list[EpochAnswer]] = defaultdict(list)
    truths: dict[str, CaseTruth] = {}
    for record in samples:
        sid = str(record.id)
        if sid not in truths:
            truths[sid] = truth_of(record.metadata)
        by_sample[sid].append(read_epoch(record, truths[sid]))
    anomalies = {
        sid: tuple(sorted(a.epoch for a in answers))
        for sid, answers in by_sample.items()
        if tuple(sorted(a.epoch for a in answers)) != tuple(range(1, k + 1))
    }
    if anomalies:
        raise GateError(f"samples without exactly {k} epochs: {dict(sorted(anomalies.items()))}")
    rows = tuple(sample_consistency(by_sample[sid], truths[sid], k) for sid in sorted(by_sample))
    return ConsistencyReport(
        k=k,
        samples=rows,
        overall=summarize("overall", rows, k),
        by_subset={name: summarize(name, [s for s in rows if s.subset == name], k) for name in SUBSETS},
        fm007_candidates=tuple(s.sample_id for s in rows if s.unanimous and not s.modal_correct),
    )


def report_from_log(log: Any, k: int | None = None) -> ConsistencyReport:
    """Report from an EvalLog read with its samples (header_only=False). k defaults
    to the run's configured epochs; the epochs override check matches reliability.py."""
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    ran = int(log.eval.config.epochs or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if declared is not None and int(declared) != ran:
        raise GateError(f"run epochs {ran} differ from the task argument epochs {declared}")
    return consistency_report(log.samples, ran if k is None else k)


def assert_gate(report: ConsistencyReport, expected_samples: int, expected_k: int = PREREGISTERED_K) -> None:
    """Section 6 gate, as in reliability.assert_gate: the pre-registered k and the
    split's sample count. Exactly k epochs per sample is enforced when the report is built."""
    if report.k != expected_k:
        raise GateError(f"report k {report.k} != pre-registered k {expected_k}")
    if report.scored_samples != expected_samples:
        raise GateError(f"scored samples {report.scored_samples} != split sample count {expected_samples}")


# --- Text report -------------------------------------------------------------------------

def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(report: ConsistencyReport) -> str:
    lines = [f"Sample Consistency, k = {report.k}, {report.scored_samples} samples"]
    lines.append("sample | subset | agreement | modal correct | tie | epochs (c correct, w wrong, p parse failure)")
    for s in report.samples:
        marks = " ".join(
            f"{a.epoch}:{'p' if a.parse_failure else 'c' if a.correct else 'w'}" for a in s.epochs
        )
        lines.append(f"{s.sample_id} | {s.subset} | {s.agreement:.2f} | {s.modal_correct} | {s.tie} | {marks}")
    for summary in (report.overall, *report.by_subset.values()):
        d = summary.discrimination
        lines.append("")
        lines.append(
            f"[{summary.name}] n={summary.n} accuracy={_fmt(summary.accuracy)} "
            f"mean agreement={_fmt(summary.mean_agreement)} AUC={_fmt(d.auc)} "
            f"(correct {d.n_correct}, wrong {d.n_wrong}){' ' + d.note if d.note else ''}"
        )
        lines.append("  agreement | n | accuracy | gap")
        for row in summary.calibration:
            lines.append(f"  {row.agreement:.2f} | {row.n} | {_fmt(row.accuracy)} | {_fmt(row.gap)}")
    lines.append("")
    lines.append(f"{FM007_NOTE}: {list(report.fm007_candidates) or 'none'}")
    lines.append(CALIBRATION_SCORE_NOTE)
    return "\n".join(lines)
