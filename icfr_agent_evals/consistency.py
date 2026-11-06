"""Sample Consistency across the k = 4 epoch resamples, per (sample, epoch).

Pre-registration section 6 (Calibration): Sample Consistency, not verbalized
confidence; the consistency measurement is deterministic, and discrimination and
calibration are tracked as separate properties. Section 2 H4 predicts the signal
is uninformative on the synthetic subset and informative on the difficulty-bearing
subsets, and section 8 reports calibration curves on both. Calibration is reported,
not scored, in Phase 1 (ratified 2026-11-04): this module is its whole measurement,
calibration.semantic_agreement is not graded, and an anchor enters only through a
new tagged pre-registration.

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
empty) with a two-sided 95% DeLong confidence interval, and a calibration table
by agreement level (n, empirical accuracy, gap = agreement level minus accuracy,
so a positive gap is over-confidence).

DeLong interval (ratified 2026-11-03), pure Python. With m correct samples X and
n wrong samples Y, midranks (ties share the mean rank) give the structural
components V10(X_i) = (R(X_i) - R_X(X_i)) / n and V01(Y_j) = 1 - (R(Y_j) -
R_Y(Y_j)) / m, where R ranks among all m + n samples and R_X, R_Y within one
class; V10(X_i) is the share of wrong samples X_i beats, ties one half. Var(AUC)
= S10 / m + S01 / n, S10 and S01 the sample variances (divisor m - 1, n - 1) of
V10 and V01 about the AUC; the interval is AUC +/- 1.959964 * sqrt(Var(AUC)),
clipped to [0, 1]. It is undefined, with a note, when either class has fewer
than 2 samples.

H4 (section 2, statistic ratified 2026-11-03), per model on the evaluation split:
an arm (synthetic; difficulty-bearing = edge plus adversarial) is informative
when its interval is defined and its lower bound exceeds 0.5. The synthetic arm
is uninformative when its AUC is undefined (no correct or no wrong samples), its
interval is undefined (fewer than 2 samples in either class) or its lower bound
is at most 0.5. H4 is supported when the synthetic arm is uninformative and the
difficulty-bearing arm is informative; otherwise it is not supported, with the
reason. A verdict is the pre-registered H4 test only for a whole evaluation-split
run of one roster worker at the pre-registered k: split eval, no --limit or
--sample-id, no subset, scored samples equal to the header's dataset count
(regression.refuse_partial_run), k = 4 and the model in roster.EVALUATED_MODELS.
Any other verdict is labeled as not the H4 test, with every reason. The model is
named as H1 to H3 name it (MODEL_DISPLAY_NAMES, from the section 5 roster).

FM-007 over-confident incorrect (ratified 2026-11-03): fires on a sample whose k
epochs all give the same canonical answer (unanimous) and whose modal answer is
wrong. It is sample-level, like FM-008, and is counted overall and per subset;
the run's model names it per model. It is not a section 10(c) sign-off code
(regression.SIGNOFF_CODES) and does not enter reliability.py's per-epoch trial,
so content-pass^k is unchanged: at least one epoch of such a sample fails its
content trial. Epochs sharing a canonical answer can differ in correctness (amounts
on both sides of the $1 tolerance), so FM-007 can fire beside FM-008.

The Calibration dimension keeps no 0 to 3 score: it is reported, not scored, in
Phase 1 (ratified 2026-11-04), with no anchor. In the section 10(a) paired tests the
unscored dimension enters at p = 1 and stays in the Holm family (m = 6).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Iterable, Sequence

from .casefile import CaseTruth, canonical_reference, nfkc_spaced, truth_of
from .regression import refuse_partial_run
from .reliability import PREREGISTERED_K, GateError, require_run_config
from .roster import EVALUATED_MODELS
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
FM007 = "FM-007"
FM007_NOTE = (
    "FM-007 over-confident incorrect (all k epochs give the same canonical answer and it is wrong; "
    "sample-level, not a section 10(c) sign-off code, content-pass^k unchanged)"
)
CALIBRATION_SCORE_NOTE = (
    "Calibration reported, not scored in Phase 1 (ratified 2026-11-04): no 0 to 3 anchor; "
    "section 10(a) enters it at p = 1, kept in the Holm family (m = 6); "
    "an anchor enters only through a new tagged pre-registration"
)
EVALUATION_SPLIT = "eval"
# Roster id -> the name H1 to H3 use (pre-registration sections 2 and 5).
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "anthropic/claude-haiku-5-5": "Haiku 5.5",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "anthropic/claude-sonnet-5-5": "Sonnet 5.5",
    "anthropic/claude-opus-5-5": "Opus 5.5",
}
assert set(MODEL_DISPLAY_NAMES) == set(EVALUATED_MODELS), "MODEL_DISPLAY_NAMES must name every roster worker"
NO_RUN_HEADER = "no run header checked (report not built by report_from_log)"


def model_display_name(model: str | None) -> str:
    """The H1 to H3 name for a roster worker, with its id; any other model by its raw id."""
    if model is None:
        return "unrecorded"
    name = MODEL_DISPLAY_NAMES.get(model)
    return f"{name} ({model})" if name else model


DELONG_Z = 1.959964          # two-sided 95%
H4_THRESHOLD = 0.5           # an arm is informative when its CI lower bound exceeds this


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
    note: str | None = None          # why the AUC is undefined
    se: float | None = None          # DeLong standard error, sqrt(Var(AUC))
    ci_low: float | None = None      # two-sided 95% DeLong interval, clipped to [0, 1]
    ci_high: float | None = None
    ci_note: str | None = None       # why the interval is undefined

    @property
    def ci_defined(self) -> bool:
        return self.ci_low is not None and self.ci_high is not None

    @property
    def informative(self) -> bool:
        """Section 2 H4: the interval is defined and its lower bound exceeds 0.5."""
        return self.ci_defined and self.ci_low > H4_THRESHOLD


def midranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; tied values share the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = mid
        i = j + 1
    return ranks


def delong_components(pos: Sequence[float], neg: Sequence[float]) -> tuple[list[float], list[float]]:
    """(V10 per correct sample, V01 per wrong sample) from midranks (DeLong 1988; Sun and Xu 2014)."""
    m, n = len(pos), len(neg)
    if not m or not n:
        raise ValueError("DeLong components need at least one sample in each class")
    r_all = midranks([*pos, *neg])
    r_pos, r_neg = midranks(pos), midranks(neg)
    v10 = [(r_all[i] - r_pos[i]) / n for i in range(m)]
    v01 = [1.0 - (r_all[m + j] - r_neg[j]) / m for j in range(n)]
    return v10, v01


def delong_ci(pos: Sequence[float], neg: Sequence[float], z: float = DELONG_Z) -> tuple[float, float, float]:
    """(se, low, high): AUC +/- z * sqrt(S10 / m + S01 / n), clipped to [0, 1]. Needs 2 or more per class."""
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        raise ValueError("a DeLong interval needs at least 2 samples in each class")
    v10, v01 = delong_components(pos, neg)
    auc = math.fsum(v10) / m
    s10 = math.fsum((v - auc) ** 2 for v in v10) / (m - 1)
    s01 = math.fsum((v - auc) ** 2 for v in v01) / (n - 1)
    se = math.sqrt(s10 / m + s01 / n)
    return se, max(0.0, auc - z * se), min(1.0, auc + z * se)


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> Discrimination:
    """ROC AUC of a score for a binary label by Mann-Whitney: the share of
    (correct, wrong) pairs in which the correct one scores higher, ties counted
    one half, with its two-sided 95% DeLong interval. The AUC is None, with a
    note, when either class is empty; the interval is None, with a note, when
    either class has fewer than 2 samples."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    pos = [float(s) for s, y in zip(scores, labels) if y]
    neg = [float(s) for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        empty = "correct" if not pos else "wrong"
        return Discrimination(None, len(pos), len(neg), f"AUC undefined: no {empty} samples",
                              ci_note=f"CI undefined: no {empty} samples")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    auc = wins / (len(pos) * len(neg))
    if len(pos) < 2 or len(neg) < 2:
        short = "correct" if len(pos) < 2 else "wrong"
        return Discrimination(auc, len(pos), len(neg), ci_note=f"CI undefined: fewer than 2 {short} samples")
    se, low, high = delong_ci(pos, neg)
    return Discrimination(auc, len(pos), len(neg), se=se, ci_low=low, ci_high=high)


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
    fm007: tuple[str, ...] = ()       # sample ids firing FM-007 in this subset

    @property
    def fm007_count(self) -> int:
        return len(self.fm007)


def fires_fm007(sample: SampleConsistency) -> bool:
    """FM-007 over-confident incorrect: unanimous over the k epochs and the modal answer wrong."""
    return sample.unanimous and not sample.modal_correct


def summarize(name: str, samples: Sequence[SampleConsistency], k: int) -> SubsetSummary:
    n = len(samples)
    return SubsetSummary(
        name=name,
        n=n,
        accuracy=sum(s.modal_correct for s in samples) / n if n else None,
        mean_agreement=sum(s.agreement for s in samples) / n if n else None,
        discrimination=roc_auc([s.agreement for s in samples], [s.modal_correct for s in samples]),
        calibration=calibration_table(samples, k),
        fm007=tuple(s.sample_id for s in samples if fires_fm007(s)),
    )


# --- Per run -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsistencyReport:
    k: int
    samples: tuple[SampleConsistency, ...]       # sorted by sample id
    overall: SubsetSummary
    by_subset: dict[str, SubsetSummary]
    fm007: tuple[str, ...]                       # sample ids firing FM-007, sorted
    model: str | None = None                     # the evaluated model, as the run records it
    split: str | None = None                     # the split the run scored
    run_scope_issues: tuple[str, ...] | None = None  # why the run is not a whole-split run; None: not checked

    @property
    def scored_samples(self) -> int:
        return len(self.samples)

    @property
    def fm007_count(self) -> int:
        return len(self.fm007)

    @property
    def fm007_by_subset(self) -> dict[str, int]:
        return {name: summary.fm007_count for name, summary in self.by_subset.items()}

    def sample(self, sample_id: str) -> SampleConsistency:
        return next(s for s in self.samples if s.sample_id == sample_id)


def consistency_report(samples: Iterable[Any], k: int, *, model: str | None = None,
                       split: str | None = None,
                       run_scope_issues: tuple[str, ...] | None = None) -> ConsistencyReport:
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
        fm007=tuple(s.sample_id for s in rows if fires_fm007(s)),
        model=model,
        split=split,
        run_scope_issues=run_scope_issues,
    )


def _log_split(log: Any) -> str | None:
    split = (log.eval.task_args or {}).get("split")
    if split is None:
        split = (getattr(log.eval, "metadata", None) or {}).get("split")
    return None if split is None else str(split)


def report_from_log(log: Any, k: int | None = None, *, run_config: bool = True,
                    max_tokens_pin: int | None = None) -> ConsistencyReport:
    """Report from an EvalLog read with its samples (header_only=False). k defaults
    to the run's configured epochs; the epochs override check and the run
    configuration check (contract C2, reliability.require_run_config, on by default)
    match reliability.report_from_log. The report carries the run's evaluated model
    and split for the H4 verdict."""
    if log.samples is None:
        raise GateError("EvalLog has no samples; read it with header_only=False and keep sample logging on")
    ran = int(log.eval.config.epochs or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if declared is not None and int(declared) != ran:
        raise GateError(f"run epochs {ran} differ from the task argument epochs {declared}")
    if run_config:
        require_run_config(log, expected_epochs=PREREGISTERED_K if k is None else k, max_tokens_pin=max_tokens_pin)
    model = getattr(log.eval, "model", None)
    return consistency_report(log.samples, ran if k is None else k,
                              model=None if model is None else str(model), split=_log_split(log),
                              run_scope_issues=run_scope_issues(log))


def run_scope_issues(log: Any) -> tuple[str, ...]:
    """Why a log is not a whole-split run: a --limit or --sample-id, a subset, or
    scored samples that differ from the header's dataset (regression.refuse_partial_run)."""
    issues: list[str] = []
    try:
        refuse_partial_run(log)
    except GateError as e:
        issues.append(str(e))
    subset = (log.eval.task_args or {}).get("subset")
    if subset is None:
        subset = (getattr(log.eval, "metadata", None) or {}).get("subset")
    if subset is not None:
        issues.append(f"run used subset {subset!r}; H4 scores the whole split")
    return tuple(issues)


def assert_gate(report: ConsistencyReport, expected_samples: int, expected_k: int = PREREGISTERED_K) -> None:
    """Section 6 gate, as in reliability.assert_gate: the pre-registered k and the
    split's sample count. Exactly k epochs per sample is enforced when the report is built."""
    if report.k != expected_k:
        raise GateError(f"report k {report.k} != pre-registered k {expected_k}")
    if report.scored_samples != expected_samples:
        raise GateError(f"scored samples {report.scored_samples} != split sample count {expected_samples}")


# --- H4 (section 2) ----------------------------------------------------------------------

@dataclass(frozen=True)
class H4Arm:
    subset: str
    discrimination: Discrimination
    informative: bool
    reason: str


@dataclass(frozen=True)
class H4Verdict:
    model: str | None
    split: str | None
    synthetic: H4Arm
    difficulty_bearing: H4Arm
    supported: bool
    reason: str
    scope_issues: tuple[str, ...] = ()   # why this verdict is not the pre-registered H4 test

    @property
    def is_h4_test(self) -> bool:
        """A whole evaluation-split run of a roster worker at k = 4; anything else is a rehearsal."""
        return not self.scope_issues


def h4_scope_issues(report: Any) -> tuple[str, ...]:
    """Every reason a report is not the pre-registered H4 test; empty when it is."""
    issues: list[str] = []
    split = getattr(report, "split", None)
    if split != EVALUATION_SPLIT:
        issues.append(f"split {split or 'unrecorded'}, not the evaluation split")
    k = getattr(report, "k", None)
    if k != PREREGISTERED_K:
        issues.append(f"k {k}, not the pre-registered k {PREREGISTERED_K}")
    model = getattr(report, "model", None)
    if model not in EVALUATED_MODELS:
        issues.append(f"model {model or 'unrecorded'} is not one of the four roster workers")
    scope = getattr(report, "run_scope_issues", None)
    issues.extend((NO_RUN_HEADER,) if scope is None else scope)
    return tuple(issues)


def h4_arm(subset: str, d: Discrimination) -> H4Arm:
    """Informative when the DeLong interval is defined and its lower bound exceeds 0.5."""
    if d.auc is None:
        return H4Arm(subset, d, False, f"uninformative: {d.note}")
    if not d.ci_defined:
        return H4Arm(subset, d, False, f"uninformative: {d.ci_note}")
    bound = f"CI lower bound {d.ci_low:.4f} (AUC {d.auc:.4f}, CI [{d.ci_low:.4f}, {d.ci_high:.4f}])"
    if d.ci_low > H4_THRESHOLD:
        return H4Arm(subset, d, True, f"informative: {bound} exceeds {H4_THRESHOLD}")
    return H4Arm(subset, d, False, f"uninformative: {bound} is at most {H4_THRESHOLD}")


def h4_verdict(report: ConsistencyReport) -> H4Verdict:
    """Section 2 H4 for one model's run: supported when the synthetic arm is
    uninformative and the difficulty-bearing arm is informative."""
    syn = h4_arm(SYNTHETIC, report.by_subset[SYNTHETIC].discrimination)
    hard = h4_arm(DIFFICULTY_BEARING, report.by_subset[DIFFICULTY_BEARING].discrimination)
    supported = not syn.informative and hard.informative
    if supported:
        reason = f"supported: synthetic arm {syn.reason}; difficulty-bearing arm {hard.reason}"
    else:
        failed = []
        if syn.informative:
            failed.append(f"synthetic arm {syn.reason}")
        if not hard.informative:
            failed.append(f"difficulty-bearing arm {hard.reason}")
        reason = "not supported: " + "; ".join(failed)
    return H4Verdict(report.model, report.split, syn, hard, supported, reason, h4_scope_issues(report))


# --- Text report -------------------------------------------------------------------------

def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(d: Discrimination) -> str:
    return f"[{_fmt(d.ci_low)}, {_fmt(d.ci_high)}]" if d.ci_defined else f"n/a ({d.ci_note})"


def format_h4(verdict: H4Verdict) -> str:
    scope = (
        "the pre-registered H4 test" if verdict.is_h4_test
        else "not the H4 test, a rehearsal: " + "; ".join(verdict.scope_issues)
    )
    return f"H4, model {model_display_name(verdict.model)} ({scope}): {verdict.reason}"


def format_report(report: ConsistencyReport) -> str:
    lines = [f"Sample Consistency, k = {report.k}, {report.scored_samples} samples, "
             f"model {report.model or 'unrecorded'}, split {report.split or 'unrecorded'}"]
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
            f"(correct {d.n_correct}, wrong {d.n_wrong}){' ' + d.note if d.note else ''} "
            f"95% DeLong CI={_ci(d)} {FM007}={summary.fm007_count}"
        )
        lines.append("  agreement | n | accuracy | gap")
        for row in summary.calibration:
            lines.append(f"  {row.agreement:.2f} | {row.n} | {_fmt(row.accuracy)} | {_fmt(row.gap)}")
    lines.append("")
    by_subset = ", ".join(f"{name} {n}" for name, n in report.fm007_by_subset.items())
    lines.append(f"{FM007_NOTE}: {report.fm007_count} sample(s) ({by_subset}): {list(report.fm007) or 'none'}")
    lines.append(format_h4(h4_verdict(report)))
    lines.append(CALIBRATION_SCORE_NOTE)
    return "\n".join(lines)
