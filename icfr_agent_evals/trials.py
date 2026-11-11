"""The four content-pass^4 trials of one run, per (sample, epoch) and per sample.

Pre-registration section 6 (Content-pass^4, Review settlement, Mandatory code audit)
and section 8 (Primary outcomes, Extractor health). Every figure is computed by code
from every (sample, epoch) score in log.samples, never from Inspect's reduced scores.

- review_settled_trial: the score of record (owner decision D-A). The anchors after
  the frozen verdicts are applied through the settlement table; per-epoch FM codes
  are final. A route on an FM-coded epoch is a finding with a verdict, and the epoch
  still fails on its code (D-H). A whole-narrative `wrong` fails the trial and zeroes
  an anchor by its error type (D-I). `cannot_settle` and `not_reviewed` zero the
  anchor the finding's row names and fail (D-K). The narrative-only units of the 54
  difficulty-bearing cases carry a whole-narrative verdict only (D-J).
- audit_adjusted_trial: the review-settled trial with every FM code the mandatory
  code audit (fm_audit.csv) rules `extractor_error` removed, the anchor that code
  lowered taking the score it would have had without it. FM-000 is never removed
  (D-R5): a parse failure fails this figure too. A sensitivity figure, never the
  score of record.
- route_blind_trial: no FM code, no unsettleable route, and every anchor 3 once
  settleable routes count as passed; every other rule stands. The most review can
  give, the trial section 10(a) reads (D-E5). It reads the log alone.
- code_only_trial: every routed epoch fails (reliability.evaluate_trial). The
  extractor health figure, labeled not a capability result, never a score.

Each public trial function returns a TrialReport whose `samples` maps a sample id to
pass or fail and the reason, so another module can be compared with it sample by
sample. figures() gives all four with Clopper-Pearson intervals and the per-anchor
means side by side.

The frozen review record is read duck-typed: anything with the fields of
scripts/review_freeze.py's Record (meta, runs, members, units, rows), as
load_frozen_record or load_record return it. The member map (members.csv) is read
here and nowhere else in the package: it ties a unit id to a worker run, sample and
epoch. This module never sets, suggests or edits a verdict.

Imports of settlement and regression are made inside functions, so regression.py may
import this module at load time without an import cycle.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import reliability as rel
from . import review_schema as rs

REVIEW_SETTLED = "review_settled"
AUDIT_ADJUSTED = "audit_adjusted"
ROUTE_BLIND = "route_blind"
CODE_ONLY = "code_only"
FIGURES: tuple[str, ...] = (REVIEW_SETTLED, AUDIT_ADJUSTED, ROUTE_BLIND, CODE_ONLY)
FIGURE_LABELS: dict[str, str] = {
    REVIEW_SETTLED: "review-settled content-pass^4, the score of record (D-A)",
    AUDIT_ADJUSTED: "audit-adjusted content-pass^4, a sensitivity figure, never the score of record",
    ROUTE_BLIND: "route-blind content-pass^4, an upper bound on the review-settled figure, not a capability result",
    CODE_ONLY: "code-only content-pass^4, the extractor health figure: not a capability result, never a score",
}

CORRECTNESS, FABRICATION, RECONCILING, COMPLETENESS = rel.TRIAL_ANCHORS
FULL_MARKS = rel.FULL_MARKS
FORMAT_BREAK_CODE = "FM-000"
CORRECTNESS_CODES: frozenset[str] = frozenset({"FM-001", "FM-002", "FM-003", "FM-016"})
# D-R5 (owner ruling 2026-11-10): the mandatory code audit never removes FM-000. A
# parse failure fails whatever review or the audit finds, so an audit row on FM-000
# ruled extractor_error leaves the epoch failing on FM-000 in the audit-adjusted figure,
# and its anchors stay as logged (with no parsed output there is no score to restore).
# audit_removed_codes applies this rule before any audit removal.
AUDIT_NEVER_REMOVED: frozenset[str] = frozenset({FORMAT_BREAK_CODE})
# The trial anchor each per-epoch code lowers. FM-006 has its own scorer outside the
# trial anchors and enters the trial through its code only. FM-000 zeroes every anchor
# and is never removed (D-R5, AUDIT_NEVER_REMOVED).
CODE_ANCHOR: dict[str, str] = {
    "FM-001": CORRECTNESS, "FM-002": CORRECTNESS, "FM-003": CORRECTNESS, "FM-016": CORRECTNESS,
    "FM-005": FABRICATION, "FM-015": RECONCILING, "FM-004": COMPLETENESS,
}
# A settlement row's anchor, as the trial anchors it names.
ROW_ANCHORS: dict[str, tuple[str, ...]] = {
    "correctness": (CORRECTNESS,),
    "fabrication": (FABRICATION,),
    "reconciling_items": (RECONCILING,),
    "correctness_and_completeness": (CORRECTNESS, COMPLETENESS),
    "any": (),
}
# D-I: the anchor a whole-narrative `wrong` sets to 0, by error type; "other" sets none.
ERROR_TYPE_ANCHOR: dict[str, str | None] = {
    "amount": CORRECTNESS, "direction": CORRECTNESS, "band": CORRECTNESS, "asset": CORRECTNESS,
    "period": CORRECTNESS, "entity": FABRICATION, "wallet_holder": FABRICATION, "item": RECONCILING,
    "wallet_scope": RECONCILING, "missing_required_element": COMPLETENESS, "other": None,
}
assert set(ERROR_TYPE_ANCHOR) == set(rs.ERROR_TYPES), "every error type needs an anchor rule"
# H2 (section 2): the codes it counts, and only those the audit rules real_error (D-N).
H2_CODES: tuple[str, ...] = ("FM-000", "FM-001", "FM-005", "FM-006", "FM-015")
H2_CASE_KIND = "adversarial"

CORRECT = "correct"
WRONG = "wrong"
CITATION_OMISSION = "citation_omission"
NOT_REVIEWED = "not_reviewed"
NONE_WRONG = "none_wrong"
REAL_ERROR, EXTRACTOR_ERROR = rs.AUDIT_VERDICTS
D_H_MODE = "findings"                      # review_schema.FM_EPOCH_ROUTE_MODES, ruled D-H
D_J_SCOPES: frozenset[str] = frozenset({"hard", "all"})  # whole-narrative scopes that meet D-J (b)
OFF_CONTRACT_AMOUNT, OFF_CONTRACT_BAND, OFF_CONTRACT_PERIOD = "variance_amount_usd", "materiality_band", "period"


class RecordError(ValueError):
    """A review record that does not cover the run: every gap is listed, not only the first."""

    def __init__(self, gaps: Sequence[str]):
        self.gaps = list(gaps)
        super().__init__("; ".join(self.gaps))


# --- Results -------------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochResult:
    """One (sample, epoch) trial under one figure. `anchors` are the anchors this figure
    reads (settled for the review figures, as logged for code-only); None is unscored."""
    sample_id: str
    epoch: int
    success: bool
    reasons: tuple[str, ...]              # why it failed; () on a pass
    anchors: Mapping[str, float | None]
    failure_modes: tuple[str, ...]        # per-epoch codes this figure counts
    removed_codes: tuple[str, ...] = ()   # audit-adjusted only: codes ruled extractor_error
    review_found: tuple[str, ...] = ()    # codes a `wrong` finding records (reported apart)
    review_found_omissions: int = 0       # findings ruled citation_omission
    narrative_errors: tuple[str, ...] = ()  # error types of a whole-narrative `wrong`


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    passed: bool
    reason: str                           # "" on a pass
    epochs: tuple[EpochResult, ...]


@dataclass(frozen=True)
class TrialReport:
    kind: str                             # one of FIGURES
    k: int
    samples: Mapping[str, SampleResult]   # sample id -> result, sorted by id

    @property
    def pass_k(self) -> dict[str, int]:
        return {sid: int(r.passed) for sid, r in self.samples.items()}

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.samples.values())

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return clopper_pearson(self.passed, self.n)

    @property
    def epochs(self) -> tuple[EpochResult, ...]:
        return tuple(e for r in self.samples.values() for e in r.epochs)

    @property
    def fm008(self) -> tuple[str, ...]:
        """Sample-level FM-008 on this figure's trial: at least one trial succeeded and
        content-pass^k failed."""
        return tuple(sid for sid, r in self.samples.items() if not r.passed and any(e.success for e in r.epochs))

    @property
    def label(self) -> str:
        return FIGURE_LABELS[self.kind]

    def anchor_means(self) -> dict[str, dict[str, float | None]]:
        return anchor_means(self)


# --- Statistics ------------------------------------------------------------------------------

def _bisect(f: Callable[[float], float], lo: float, hi: float, target: float, iters: int = 200) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _binom_cdf(x: int, n: int, p: float) -> float:
    return math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for x / n (the same method as scripts/pilot_report.py)."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if x == 0 else _bisect(lambda p: 1 - _binom_cdf(x - 1, n, p), 0.0, 1.0, alpha / 2)
    hi = 1.0 if x == n else _bisect(lambda p: -_binom_cdf(x, n, p), 0.0, 1.0, -alpha / 2)
    return lo, hi


def anchor_means(report: TrialReport) -> dict[str, dict[str, float | None]]:
    """Per trial anchor: the mean over every (sample, epoch), and the standard error of
    the per-sample epoch means with a normal 95% interval (section 8). An unscored
    anchor counts as 0."""
    out: dict[str, dict[str, float | None]] = {}
    for anchor in rel.TRIAL_ANCHORS:
        per_sample = []
        values = []
        for r in report.samples.values():
            vals = [float(e.anchors.get(anchor) or 0.0) for e in r.epochs]
            if vals:
                values.extend(vals)
                per_sample.append(sum(vals) / len(vals))
        mean = sum(values) / len(values) if values else None
        se = None
        if len(per_sample) >= 2:
            m = sum(per_sample) / len(per_sample)
            sd = math.sqrt(sum((v - m) ** 2 for v in per_sample) / (len(per_sample) - 1))
            se = sd / math.sqrt(len(per_sample))
        out[anchor] = {
            "mean": mean, "se": se,
            "ci95_low": None if mean is None or se is None else mean - 1.96 * se,
            "ci95_high": None if mean is None or se is None else mean + 1.96 * se,
        }
    return out


# --- Facts read from the log -------------------------------------------------------------------

@dataclass(frozen=True)
class _Facts:
    sample_id: str
    epoch: int
    sample: Any
    error: bool
    anchors: Mapping[str, float | None]
    codes: tuple[str, ...]
    findings: tuple[tuple[str, tuple[str, ...]], ...]   # settleable (route key, reasons)
    unsettleable: tuple[tuple[str, tuple[str, ...]], ...]
    code_trial: Any                                      # reliability.Trial
    structured_points: float | None
    citation_omissions: int
    completeness_period_withheld: bool
    extractions: tuple[Mapping[str, Any], ...]
    case_kind: str | None

    @property
    def routed(self) -> bool:
        return bool(self.findings or self.unsettleable)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _meta(score: Any) -> Mapping[str, Any]:
    return (getattr(score, "metadata", None) or {}) if score is not None else {}


def _details(score: Any) -> Mapping[str, Any]:
    d = _meta(score).get("details")
    return d if isinstance(d, Mapping) else {}


def _off_contract_fields(reasons: Iterable[str]) -> set[str]:
    """The off-contract fields these reasons name (settlement rows S11 to S13)."""
    from . import settlement

    reasons = [str(r) for r in reasons]
    mine = [r for r in reasons if _route_key(r) == settlement.OFF_CONTRACT_ROUTE_KEY]
    if not mine:
        return set()
    try:
        rows = settlement.rows_for(settlement.OFF_CONTRACT_ROUTE_KEY, mine)
    except ValueError:
        return set()
    return {r.key for r in rows if r.match == "off_contract_field"}


def _route_key(reason: str) -> str | None:
    from .regression import route_key

    return route_key(reason)


def _case_kind(sample: Any) -> str | None:
    meta = getattr(sample, "metadata", None) or {}
    truth = meta.get("truth") if isinstance(meta, Mapping) else None
    kind = truth.get("case_kind") if isinstance(truth, Mapping) else None
    return None if kind is None else str(kind)


def _is_hard(case_kind: str | None) -> bool:
    from .consistency import DIFFICULTY_BEARING, SUBSET_OF_KIND

    return case_kind is not None and SUBSET_OF_KIND.get(case_kind) == DIFFICULTY_BEARING


def _collect(samples: Iterable[Any], k: int) -> tuple[rel.ReliabilityReport, dict[tuple[str, int], _Facts]]:
    """The code-only report and the facts of every (sample, epoch), routes grouped per
    route key exactly as scripts/review_packet.py groups them."""
    from . import settlement

    samples = list(samples)
    report = rel.reliability_report(samples, k)
    by_ident = {(str(s.id), int(s.epoch)): s for s in samples}
    merged: dict[tuple[str, int], dict[str, list[str]]] = defaultdict(dict)
    for item in rel.queue_items(report):
        if item.route_key is None:
            continue
        reasons = merged[(item.sample_id, item.epoch)].setdefault(item.route_key, [])
        if item.reason not in reasons:
            reasons.append(item.reason)
    facts: dict[tuple[str, int], _Facts] = {}
    for t in report.trials:
        ident = (t.sample_id, t.epoch)
        s = by_ident[ident]
        scores = getattr(s, "scores", None) or {}
        anchors = {a: _numeric(getattr(scores.get(a), "value", None)) if scores.get(a) is not None else None
                   for a in rel.TRIAL_ANCHORS}
        findings, unsettleable = [], []
        for key in sorted(merged.get(ident, {})):
            reasons = tuple(sorted(merged[ident][key]))
            (findings if settlement.is_settleable(key, reasons) else unsettleable).append((key, reasons))
        cdet = _details(scores.get(CORRECTNESS))
        comp_reasons = [str(r) for r in _meta(scores.get(COMPLETENESS)).get("review_reasons") or []]
        omissions = _details(scores.get(FABRICATION)).get("citation_omissions") or []
        extractions = cdet.get("narrative_extractions") or []
        facts[ident] = _Facts(
            sample_id=t.sample_id, epoch=t.epoch, sample=s, error=getattr(s, "error", None) is not None,
            anchors=anchors, codes=tuple(t.failure_modes), findings=tuple(findings),
            unsettleable=tuple(unsettleable), code_trial=t,
            structured_points=_numeric(cdet.get("structured_points")),
            citation_omissions=len(omissions) if isinstance(omissions, (list, tuple)) else 0,
            completeness_period_withheld=OFF_CONTRACT_PERIOD in _off_contract_fields(comp_reasons),
            extractions=tuple(e for e in extractions if isinstance(e, Mapping)),
            case_kind=_case_kind(s),
        )
    return report, facts


def _samples_of(log_or_samples: Any) -> list[Any]:
    samples = getattr(log_or_samples, "samples", None)
    if samples is None and hasattr(log_or_samples, "eval"):
        raise rel.GateError("EvalLog has no samples; read it with header_only=False")
    return list(samples if samples is not None else log_or_samples)


def _k_of(log_or_samples: Any, k: int | None) -> int:
    if k is not None:
        return int(k)
    config = getattr(getattr(log_or_samples, "eval", None), "config", None)
    epochs = getattr(config, "epochs", None)
    return int(epochs) if epochs else rel.PREREGISTERED_K


# --- The review record -------------------------------------------------------------------------

@dataclass(frozen=True)
class _Finding:
    unit_id: str
    reasons: tuple[str, ...]
    verdict: str | None


@dataclass
class _RecordView:
    """One worker run's slice of the review record, keyed by (sample, epoch)."""
    run_id: str
    findings: dict[tuple[str, int, str], _Finding] = field(default_factory=dict)
    narratives: dict[tuple[str, int], tuple[str, tuple[str, ...]] | None] = field(default_factory=dict)
    audits: dict[tuple[str, int, str], str | None] = field(default_factory=dict)
    skipped: set[tuple[str, int, str]] = field(default_factory=set)
    gaps: list[str] = field(default_factory=list)


def _json_list(cell: Any) -> list:
    if isinstance(cell, (list, tuple)):
        return list(cell)
    try:
        value = json.loads(cell) if cell else []
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def resolve_run_id(record: Any, log: Any = None, run_id: str | None = None) -> str:
    """The masked run id (R1..) of this log in the record: the one given, else the run
    whose eval id is the log's, else the record's only run."""
    runs = dict(getattr(record, "runs", None) or {})
    if run_id is not None:
        if run_id not in runs:
            raise RecordError([f"run id {run_id!r} is not in the record (runs.json holds {sorted(runs)})"])
        return run_id
    eval_id = getattr(getattr(log, "eval", None), "eval_id", None)
    if eval_id is not None:
        hits = [rid for rid, info in runs.items() if str((info or {}).get("eval_id")) == str(eval_id)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RecordError([f"eval id {eval_id} names {len(hits)} runs in the record"])
        raise RecordError([f"the log's eval id {eval_id} is not in the record's runs.json"])
    if len(runs) == 1:
        return next(iter(runs))
    raise RecordError(["the record holds several runs; pass run_id or the EvalLog"])


def _record_view(record: Any, run_id: str) -> _RecordView:
    view = _RecordView(run_id=run_id)
    rows_by_unit: dict[str, dict] = {}
    for row in getattr(record, "rows", None) or ():
        uid = row.get("unit_id", "")
        if uid in rows_by_unit:
            view.gaps.append(f"unit {uid} has more than one verdict row")
            continue
        rows_by_unit[uid] = row
    units = getattr(record, "units", None) or {}
    for m in getattr(record, "members", None) or ():
        if m.get("run_id") != run_id:
            continue
        kind, uid = m.get("member_kind"), m.get("unit_id", "")
        try:
            ident = (str(m["sample_id"]), int(m["epoch"]))
        except (KeyError, TypeError, ValueError):
            view.gaps.append(f"member map row for unit {uid!r} has no readable sample and epoch")
            continue
        row = rows_by_unit.get(uid)
        if kind == "finding":
            reasons = tuple(str(r) for r in _json_list(m.get("reasons")))
            verdict = (row or {}).get("verdict") or None
            view.findings[(*ident, m.get("route_key", ""))] = _Finding(uid, reasons, verdict)
        elif kind == "narrative":
            if row is None or not row.get("verdict"):
                view.narratives[ident] = None
            else:
                types = row.get("reason", "")
                view.narratives[ident] = (row["verdict"], tuple(t for t in types.split(rs.RECORD_LIST_SEP) if t))
        elif kind == "fm_audit":
            codes = [str(c) for c in _json_list(m.get("fm_codes"))]
            code = (units.get(uid) or {}).get("fm_code") or (codes[0] if codes else "")
            view.audits[(*ident, code)] = ((row or {}).get("verdict") or None)
        elif kind == "fm_epoch_route_skipped":
            view.skipped.add((*ident, m.get("route_key", "")))
    return view


def _record_gaps(view: _RecordView, record: Any, facts: Mapping[tuple[str, int], _Facts]) -> list[str]:
    """Every way the record fails to cover this run's log (section 8: the settlement
    refuses a record unless every finding, every narrative in scope and every
    FM-coded (epoch, code) has exactly one row)."""
    from . import settlement

    gaps = list(view.gaps)
    meta = getattr(record, "meta", None) or {}
    if meta.get("fm_epoch_routes", D_H_MODE) != D_H_MODE:
        gaps.append(f"packet fm_epoch_routes is {meta.get('fm_epoch_routes')!r}; D-H needs {D_H_MODE!r}")
    if meta.get("scope") not in D_J_SCOPES:
        gaps.append(f"packet whole-narrative scope is {meta.get('scope')!r}; D-J (b) needs one of {sorted(D_J_SCOPES)}")
    expected: set[tuple[str, int, str]] = set()
    for ident, f in sorted(facts.items()):
        sid, ep = ident
        for key, reasons in f.findings:
            expected.add((sid, ep, key))
            got = view.findings.get((sid, ep, key))
            if got is None:
                gaps.append(f"{sid} epoch {ep}: no finding member for route {key!r}")
                continue
            if got.reasons != reasons:
                gaps.append(f"{sid} epoch {ep}: the member reasons for {key!r} differ from the log")
            if got.verdict is None:
                gaps.append(f"{sid} epoch {ep}: finding {got.unit_id} has no verdict")
            elif got.verdict not in settlement.allowed_verdicts(key, reasons):
                gaps.append(f"{sid} epoch {ep}: verdict {got.verdict!r} is not allowed for {key!r}")
        if f.routed or f.codes or _is_hard(f.case_kind):
            n = view.narratives.get(ident, "absent")
            if n == "absent":
                gaps.append(f"{sid} epoch {ep}: no whole-narrative member")
            elif n is None:
                gaps.append(f"{sid} epoch {ep}: the whole-narrative unit has no verdict")
            elif n[0] not in rs.NARRATIVE_VERDICTS:
                gaps.append(f"{sid} epoch {ep}: whole-narrative verdict {n[0]!r} is not none_wrong or wrong")
            elif any(t not in ERROR_TYPE_ANCHOR for t in n[1]):
                gaps.append(f"{sid} epoch {ep}: a whole-narrative error type is not one of ERROR_TYPES")
            elif n[0] == WRONG and not n[1]:
                gaps.append(f"{sid} epoch {ep}: a whole-narrative wrong names no error type")
        for code in f.codes:
            a = view.audits.get((sid, ep, code), "absent")
            if a == "absent":
                gaps.append(f"{sid} epoch {ep}: no audit row for {code}")
            elif a not in rs.AUDIT_VERDICTS:
                gaps.append(f"{sid} epoch {ep}: audit verdict {a!r} for {code} is not real_error or extractor_error")
    for (sid, ep, key) in sorted(set(view.findings) - expected):
        gaps.append(f"{sid} epoch {ep}: the record holds a finding on {key!r} that the log does not route")
    for (sid, ep) in sorted(set(view.narratives) - set(facts)):
        gaps.append(f"{sid} epoch {ep}: the record reads a narrative of an epoch the log does not hold")
    for (sid, ep, code) in sorted(view.audits):
        if (sid, ep) not in facts or code not in facts[(sid, ep)].codes:
            gaps.append(f"{sid} epoch {ep}: the record audits {code}, which the log does not fire there")
    return gaps


# --- Settlement of one epoch ------------------------------------------------------------------

def _review_found_code(row: Any, key: str, f: _Facts) -> str:
    """The code a `wrong` verdict records: the row's code, or for S69 the code the
    proven-contradiction exception downgraded."""
    if row.review_found_code != "downgraded_from":
        return str(row.review_found_code)
    kind = key.split(": ", 1)[1] if ": " in key else key
    codes = sorted({str(e["downgraded_from"]) for e in f.extractions
                    if e.get("downgrade_reason") == kind and e.get("downgraded_from")})
    return "/".join(codes) if codes else "downgraded_from"


def _reconciling_without_fm015(sample: Any) -> float | None:
    """The reconciling-items score with every FM-015 problem set aside: the split code
    computes from the cited items, then the omission rule (scoring.check_reconciling_items
    past its FM-015 return). None when the output or the case cannot be read."""
    from .casefile import USD_TOLERANCE, explained_from_citations, truth_of
    from .scoring import (
        RECONCILING_OMISSION_NONE_CITED,
        RECONCILING_OMISSION_PARTIAL,
        SCORE_MAX,
        parse_worker_output,
    )

    try:
        completion = getattr(getattr(sample, "output", None), "completion", None)
        parse = parse_worker_output(None if completion is None else str(completion))
        if not parse.ok:
            return None
        truth = truth_of(sample.metadata)
        refs = [e["reference"] for e in parse.parsed["reconciling_items"]]
        seen = {item.reference for item in (truth.item_by_reference(r) for r in refs) if item is not None}
        explained, _ = explained_from_citations(refs, truth)
        omitted = {i.reference for i in truth.eligible_items()} - seen
    except Exception:  # an unreadable case keeps the logged score
        return None
    if abs(explained - truth.explained_usd) <= USD_TOLERANCE:
        return float(SCORE_MAX)
    if omitted:
        return float(RECONCILING_OMISSION_PARTIAL if seen else RECONCILING_OMISSION_NONE_CITED)
    return 0.0


def audit_removed_codes(codes: Iterable[str], audit_verdict: Callable[[str], str | None]) -> frozenset[str]:
    """The codes of one epoch the audit-adjusted figure removes: those the audit rules
    extractor_error, less every code in AUDIT_NEVER_REMOVED. D-R5: FM-000 is never
    removed, whatever its audit row says, so the epoch keeps failing on it."""
    return frozenset(c for c in codes if c not in AUDIT_NEVER_REMOVED and audit_verdict(c) == EXTRACTOR_ERROR)


def _settle(f: _Facts, *, finding_verdict: Callable[[str], str], narrative: tuple[str, tuple[str, ...]] | None,
            narrative_required: bool, removed: frozenset[str] = frozenset()) -> EpochResult:
    """Apply verdicts to one epoch through the settlement table (section 6, Settlement by code)."""
    from . import settlement

    reasons: list[str] = []
    anchors: dict[str, float | None] = dict(f.anchors)
    # D-R5 again, so no caller can pass FM-000 in `removed`.
    removed_codes = tuple(c for c in f.codes if c in removed and c not in AUDIT_NEVER_REMOVED)
    remaining = tuple(c for c in f.codes if c not in removed_codes)
    review_found: list[str] = []
    omissions_found = 0
    narrative_errors: tuple[str, ...] = ()

    if f.error:
        reasons.append("sample error")
    for a in rel.TRIAL_ANCHORS:
        if anchors[a] is None:
            reasons.append(f"{a} unscored or non-numeric")
    if FORMAT_BREAK_CODE in f.codes:
        reasons.append("format break (FM-000): a parse failure fails whatever review finds")
    if remaining:
        reasons.append(f"FM codes {sorted(set(remaining))} (codes final)")

    corr_withheld = set()
    for key, rs_ in f.findings:
        if key == settlement.OFF_CONTRACT_ROUTE_KEY:
            corr_withheld = _off_contract_fields(rs_)
    withheld_points = len(corr_withheld & {OFF_CONTRACT_AMOUNT, OFF_CONTRACT_BAND})
    remaining_corr = set(remaining) & CORRECTNESS_CODES
    structured = f.structured_points if f.structured_points is not None else float(FULL_MARKS - withheld_points)
    fab_routed = any(FABRICATION in ROW_ANCHORS[r.anchor] for key, rs_ in (*f.findings, *f.unsettleable)
                     for r in settlement.rows_for(key, rs_))

    # Audit-adjusted: the anchor a removed code lowered takes the score it would have had.
    # On an FM-000 epoch nothing is restored (D-R5): a co-occurring code ruled
    # extractor_error is removed, but the anchors stay as logged and the epoch fails.
    if removed_codes and FORMAT_BREAK_CODE not in f.codes:
        removed_set = set(removed_codes)
        if removed_set & CORRECTNESS_CODES and not remaining_corr and anchors[CORRECTNESS] is not None:
            structured = float(FULL_MARKS - withheld_points)
            anchors[CORRECTNESS] = 0.0 if OFF_CONTRACT_PERIOD in corr_withheld else structured
        if "FM-005" in removed_set and anchors[FABRICATION] is not None:
            anchors[FABRICATION] = 0.0 if (fab_routed or f.citation_omissions) else float(FULL_MARKS)
        if "FM-015" in removed_set and anchors[RECONCILING] is not None:
            redo = _reconciling_without_fm015(f.sample)
            if redo is None:
                reasons.append("FM-015 removed but the reconciling-items score cannot be recomputed")
            else:
                anchors[RECONCILING] = redo
        if "FM-004" in removed_set and anchors[COMPLETENESS] is not None:
            anchors[COMPLETENESS] = float(FULL_MARKS - (1 if f.completeness_period_withheld else 0))

    for key, _ in f.unsettleable:
        reasons.append(f"unsettleable route {key!r}")

    zero: list[str] = []
    restore: set[str] = set()
    fab_findings, fab_all_correct = 0, True
    for key, rs_ in f.findings:
        verdict = finding_verdict(key)
        rows = settlement.rows_for(key, rs_)
        on_fab = any(FABRICATION in ROW_ANCHORS[r.anchor] for r in rows)
        fab_findings += on_fab
        if verdict == CORRECT:
            restore |= {r.key for r in rows if r.match == "off_contract_field"}
            continue
        if on_fab:
            fab_all_correct = False
        reasons.append(f"finding {key!r}: {verdict}")
        for r in rows:
            zero.extend(ROW_ANCHORS[r.anchor])
            if verdict == WRONG:
                review_found.append(_review_found_code(r, key, f))
        if verdict == CITATION_OMISSION:
            zero.append(FABRICATION)
            omissions_found += 1

    # `correct`: the table's on-correct effect.
    if restore and anchors[CORRECTNESS] is not None:
        restored = len(restore & {OFF_CONTRACT_AMOUNT, OFF_CONTRACT_BAND})
        if not remaining_corr:
            anchors[CORRECTNESS] = min(float(FULL_MARKS), structured + restored)
        elif anchors[CORRECTNESS] == structured and OFF_CONTRACT_PERIOD not in corr_withheld:
            anchors[CORRECTNESS] = min(float(FULL_MARKS), anchors[CORRECTNESS] + restored)
    if OFF_CONTRACT_PERIOD in restore and f.completeness_period_withheld and anchors[COMPLETENESS] is not None:
        anchors[COMPLETENESS] = min(float(FULL_MARKS), anchors[COMPLETENESS] + 1)
    fab_unsettleable = any(FABRICATION in ROW_ANCHORS[r.anchor] for key, rs_ in f.unsettleable
                           for r in settlement.rows_for(key, rs_))
    if (fab_findings and fab_all_correct and "FM-005" not in remaining and not f.citation_omissions
            and not fab_unsettleable and anchors[FABRICATION] is not None):
        anchors[FABRICATION] = float(FULL_MARKS)

    # D-I: a whole-narrative `wrong` fails the trial and zeroes an anchor by error type.
    if narrative is None:
        if narrative_required:
            reasons.append("whole narrative not read")
    elif narrative[0] not in rs.NARRATIVE_VERDICTS:
        reasons.append(f"whole-narrative record {narrative[0]!r} is not none_wrong or wrong")
    elif narrative[0] == WRONG:
        narrative_errors = tuple(narrative[1])
        reasons.append(f"whole-narrative wrong ({', '.join(narrative_errors) or 'no type'})")
        zero.extend(a for a in (ERROR_TYPE_ANCHOR.get(t) for t in narrative_errors) if a)

    for a in zero:
        if anchors.get(a) is not None:
            anchors[a] = 0.0
    for a in rel.TRIAL_ANCHORS:
        v = anchors[a]
        if v is not None and v != FULL_MARKS:
            reasons.append(f"{a} settled {v:g}")
    return EpochResult(
        sample_id=f.sample_id, epoch=f.epoch, success=not reasons, reasons=tuple(reasons), anchors=anchors,
        failure_modes=tuple(dict.fromkeys(remaining)), removed_codes=removed_codes,
        review_found=tuple(review_found), review_found_omissions=omissions_found, narrative_errors=narrative_errors,
    )


def _report(kind: str, k: int, results: Iterable[EpochResult]) -> TrialReport:
    by_sample: dict[str, list[EpochResult]] = defaultdict(list)
    for r in results:
        by_sample[r.sample_id].append(r)
    out: dict[str, SampleResult] = {}
    for sid in sorted(by_sample):
        eps = sorted(by_sample[sid], key=lambda e: e.epoch)
        seen = tuple(e.epoch for e in eps)
        why = []
        if seen != tuple(range(1, k + 1)):
            why.append(f"epochs {list(seen)}, not 1 to {k}")
        why += [f"epoch {e.epoch}: {'; '.join(e.reasons)}" for e in eps if not e.success]
        out[sid] = SampleResult(sid, not why, " | ".join(why), tuple(eps))
    return TrialReport(kind=kind, k=k, samples=out)


# --- Public trials -----------------------------------------------------------------------------

def code_only_trial(log_or_samples: Any, *, k: int | None = None) -> TrialReport:
    """Every routed epoch fails (reliability.evaluate_trial): the extractor health
    figure, never a score."""
    k = _k_of(log_or_samples, k)
    _, facts = _collect(_samples_of(log_or_samples), k)
    results = [EpochResult(f.sample_id, f.epoch, f.code_trial.success, tuple(f.code_trial.failed_because),
                           dict(f.anchors), tuple(f.codes)) for f in facts.values()]
    return _report(CODE_ONLY, k, results)


def route_blind_epoch(sample: Any) -> EpochResult:
    """One (sample, epoch) under the route-blind rule, from the log alone."""
    _, facts = _collect([sample], 1)
    f = next(iter(facts.values()))
    return _settle(f, finding_verdict=lambda key: CORRECT, narrative=None, narrative_required=False)


def route_blind_trial(log_or_samples: Any, *, k: int | None = None) -> TrialReport:
    """No FM code, no unsettleable route, and every anchor 3 once every settleable
    route counts as passed (every finding `correct`, no whole-narrative `wrong`); every
    other rule stands. Computed from the log alone, with no review record."""
    k = _k_of(log_or_samples, k)
    _, facts = _collect(_samples_of(log_or_samples), k)
    results = [_settle(f, finding_verdict=lambda key: CORRECT, narrative=None, narrative_required=False,
                       ) for f in facts.values()]
    return _report(ROUTE_BLIND, k, results)


def _reviewed(kind: str, log_or_samples: Any, record: Any, *, k: int | None, run_id: str | None,
              require_complete: bool) -> TrialReport:
    k = _k_of(log_or_samples, k)
    log = log_or_samples if hasattr(log_or_samples, "eval") else None
    rid = resolve_run_id(record, log, run_id)
    _, facts = _collect(_samples_of(log_or_samples), k)
    view = _record_view(record, rid)
    if require_complete:
        gaps = _record_gaps(view, record, facts)
        if gaps:
            raise RecordError(gaps)
    results = []
    for ident, f in facts.items():
        sid, ep = ident

        def verdict_of(key: str, sid: str = sid, ep: int = ep) -> str:
            got = view.findings.get((sid, ep, key))
            return got.verdict if got is not None and got.verdict else NOT_REVIEWED  # unreviewed fails (D-K)

        narrative = view.narratives.get(ident)
        required = f.routed or bool(f.codes) or _is_hard(f.case_kind)
        removed: frozenset[str] = frozenset()
        if kind == AUDIT_ADJUSTED:
            removed = audit_removed_codes(f.codes, lambda c, sid=sid, ep=ep: view.audits.get((sid, ep, c)))
        results.append(_settle(f, finding_verdict=verdict_of, narrative=narrative, narrative_required=required,
                               removed=removed))
    return _report(kind, k, results)


def review_settled_trial(log_or_samples: Any, record: Any, *, k: int | None = None, run_id: str | None = None,
                         require_complete: bool = True) -> TrialReport:
    """The score of record (D-A): the frozen verdicts applied through the settlement
    table, per-epoch FM codes final. With require_complete (the default) a record that
    does not cover every finding, every narrative in scope and every FM-coded (epoch,
    code) is refused (RecordError); without it a missing finding verdict counts as
    `not_reviewed` and a missing whole-narrative read fails the epoch."""
    return _reviewed(REVIEW_SETTLED, log_or_samples, record, k=k, run_id=run_id, require_complete=require_complete)


def audit_adjusted_trial(log_or_samples: Any, record: Any, *, k: int | None = None, run_id: str | None = None,
                         require_complete: bool = True) -> TrialReport:
    """The review-settled trial with every FM code fm_audit.csv rules `extractor_error`
    removed; the anchor the code lowered takes the score it would have had without it.
    FM-000 is never removed (D-R5, audit_removed_codes): an epoch with a parse failure
    fails here whatever the audit rules. A sensitivity figure, never the score of record."""
    return _reviewed(AUDIT_ADJUSTED, log_or_samples, record, k=k, run_id=run_id, require_complete=require_complete)


# --- All figures -------------------------------------------------------------------------------

def figure_row(report: TrialReport) -> dict[str, Any]:
    lo, hi = report.interval
    return {"kind": report.kind, "label": report.label, "passed": report.passed, "n": report.n,
            "rate": report.rate, "ci95": [lo, hi], "fm008": list(report.fm008), "per_sample": report.pass_k}


def figures(log_or_samples: Any, record: Any = None, *, k: int | None = None, run_id: str | None = None,
            require_complete: bool = True) -> dict[str, Any]:
    """Every figure of one run (section 6 and section 8): the review-settled,
    audit-adjusted and route-blind content-pass^4 (the three always reported together)
    and the code-only extractor health figure, each with its Clopper-Pearson interval,
    and the per-anchor means review-settled and audit-adjusted side by side. Without a
    record only the route-blind and code-only figures are given."""
    reports: dict[str, TrialReport] = {}
    if record is not None:
        reports[REVIEW_SETTLED] = review_settled_trial(log_or_samples, record, k=k, run_id=run_id,
                                                       require_complete=require_complete)
        reports[AUDIT_ADJUSTED] = audit_adjusted_trial(log_or_samples, record, k=k, run_id=run_id,
                                                       require_complete=require_complete)
    reports[ROUTE_BLIND] = route_blind_trial(log_or_samples, k=k)
    reports[CODE_ONLY] = code_only_trial(log_or_samples, k=k)
    out: dict[str, Any] = {"figures": {kind: figure_row(r) for kind, r in reports.items()}}
    if record is not None:
        settled, audit = reports[REVIEW_SETTLED].anchor_means(), reports[AUDIT_ADJUSTED].anchor_means()
        out["anchor_means"] = {a: {REVIEW_SETTLED: settled[a], AUDIT_ADJUSTED: audit[a]} for a in rel.TRIAL_ANCHORS}
        out["fm008"] = list(reports[REVIEW_SETTLED].fm008)
        found = reports[REVIEW_SETTLED].epochs
        out["review_found"] = {
            "codes": dict(sorted(Counter(c for e in found for c in e.review_found).items())),
            "citation_omissions": sum(e.review_found_omissions for e in found),
            "whole_narrative_wrong": sum(1 for e in found if e.narrative_errors),
        }
    out["reports"] = reports
    return out


def format_figures(result: Mapping[str, Any]) -> str:
    """One line per figure, then the anchor means side by side."""
    lines = []
    for kind in FIGURES:
        row = result["figures"].get(kind)
        if row is None:
            continue
        lo, hi = row["ci95"]
        lines.append(f"{row['label']}: {row['passed']}/{row['n']} = {row['rate']:.3f} "
                     f"(Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}])")
    for anchor, both in (result.get("anchor_means") or {}).items():
        s, a = both[REVIEW_SETTLED]["mean"], both[AUDIT_ADJUSTED]["mean"]
        lines.append(f"  {anchor}: review-settled {s:.3f}, audit-adjusted {a:.3f}")
    return "\n".join(lines)


# --- H2 (D-N) ----------------------------------------------------------------------------------

def audited_code_counts(log_or_samples: Any, record: Any, *, k: int | None = None, run_id: str | None = None,
                        codes: Sequence[str] = H2_CODES, case_kinds: Sequence[str] | None = (H2_CASE_KIND,),
                        require_complete: bool = True) -> dict[str, dict[str, int]]:
    """Per audit verdict, the (sample, epoch) count of each code (H2: only the codes
    the audit rules real_error count, D-N; extractor_error is reported beside). By
    default the H2 codes on adversarial cases; case_kinds=None counts every case."""
    k = _k_of(log_or_samples, k)
    log = log_or_samples if hasattr(log_or_samples, "eval") else None
    rid = resolve_run_id(record, log, run_id)
    _, facts = _collect(_samples_of(log_or_samples), k)
    view = _record_view(record, rid)
    if require_complete:
        gaps = [g for g in _record_gaps(view, record, facts) if "audit" in g]
        if gaps:
            raise RecordError(gaps)
    out: dict[str, Counter] = {REAL_ERROR: Counter(), EXTRACTOR_ERROR: Counter()}
    for (sid, ep), f in facts.items():
        if case_kinds is not None and f.case_kind not in case_kinds:
            continue
        for code in dict.fromkeys(f.codes):
            if code in codes:
                verdict = view.audits.get((sid, ep, code)) or REAL_ERROR
                out[verdict if verdict in out else REAL_ERROR][code] += 1
    return {v: dict(sorted(c.items())) for v, c in out.items()}
