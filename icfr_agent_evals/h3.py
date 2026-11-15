"""H3, the cross-model ordering on the 54 difficulty-bearing cases. No model call.

Pre-registration section 2 (H3), with the second-look gate of section 8 (decision
D-L, 2026-11-10; the empty-draw outcome, decision D-R19, 2026-11-14), built as
tooling item (7) of the section 13 row on tooling built after the tag.

The verdict, as section 2 fixes it:
- H3 is tested on the 54 difficulty-bearing cases pooled (30 edge plus 24
  adversarial) and on four anchors: correctness, fabrication, reconciling items and
  completeness.
- Three paired steps per anchor, twelve in all: Haiku 5.5 <= Sonnet 5, Sonnet 5 <=
  Sonnet 5.5 and Sonnet 5 <= Opus 5.5. A step holds when the mean over the 54 cases
  of the per-case difference of epoch means (higher-ordered model minus lower) is at
  or above 0. The mean is an exact fraction, so a zero mean is never moved below 0 by
  float error.
- H3 holds only when all twelve steps hold. Every failing step is named with its
  anchor.
- Each difference gets a paired 95% t interval: mean +/- t(0.975, 53) x the sample SD
  of the 54 per-case differences / sqrt(54). The interval never enters the verdict.

Reported beside, never in the verdict:
- edge and adversarial separately (the same steps on 30 and 24 cases);
- the faithfulness minimum (per epoch the smaller of the fabrication and
  reconciling-items anchors, the combination section 6 ratifies and
  scoring.check_faithfulness carries out), as model means and paired steps;
- the Sonnet 5.5 versus Opus 5.5 paired difference per anchor, flagged as a
  cross-model saturation finding when its mean lies within one standard error, that
  same SD / sqrt(54) (decision D-R19, 2026-11-14);
- the Sonnet 5 to Sonnet 5.5 delta per anchor as the generational comparison, with
  the reasoning mode beside it.

Robustness (robustness_labels): the review-settled ordering against the
audit-adjusted one ("not robust to extractor error") and against the ordering with
citation omissions scored 0, the rule before decision D-R9 ("not robust to the
citation rule"). An ordering here is the set of failing steps; the verdict follows
from it.

Second-look gate (second_look_gate): if any worker's second look result is "review
not confirmed", H3 is reported as not tested. "second look empty" is tested under
that label. "confirmed" is tested.

The input table maps each of the four worker models to its 54 cases, each case to
its case kind and, per trial anchor, its k epoch scores. table_from_reports builds it
from trials.TrialReport objects (review-settled or audit-adjusted); with
before_d_r9 it applies the rule before D-R9 the way trials.subset_anchor_means does.
Refused is raised on any table H3 cannot be computed on: a wrong case count, a case
set that differs between models, a wrong epoch count or an out-of-range score.

CLI (exit 0 on a report, 3 on a refusal):
  python -m icfr_agent_evals.h3 --settled S.json --second-look L.json
         [--audit A.json] [--before-d-r9 B.json] [--reasoning-modes M.json] [--json out.json]
Each table file holds {model id: {case id: {"case_kind": ..., "epochs": {anchor: [scores]}}}}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import reliability as rel
from . import roster
from .regression import t_ppf

EXIT_OK, EXIT_REFUSED = 0, 3

# --- The models and the twelve steps ------------------------------------------------------------

HAIKU_5_5, SONNET_5, OPUS_5_5 = roster.BASELINES
SONNET_5_5 = roster.WORKER
assert HAIKU_5_5.endswith("haiku-5-5") and SONNET_5.endswith("sonnet-5"), "roster order changed"
assert OPUS_5_5.endswith("opus-5-5") and SONNET_5_5.endswith("sonnet-5-5"), "roster order changed"
WORKERS: tuple[str, ...] = (HAIKU_5_5, SONNET_5, SONNET_5_5, OPUS_5_5)
MODEL_LABELS: dict[str, str] = {
    HAIKU_5_5: "Haiku 5.5", SONNET_5: "Sonnet 5", SONNET_5_5: "Sonnet 5.5", OPUS_5_5: "Opus 5.5",
}
# (lower-ordered, higher-ordered): the difference is higher minus lower.
STEPS: tuple[tuple[str, str], ...] = ((HAIKU_5_5, SONNET_5), (SONNET_5, SONNET_5_5), (SONNET_5, OPUS_5_5))
GENERATIONAL: tuple[str, str] = (SONNET_5, SONNET_5_5)
SATURATION_PAIR: tuple[str, str] = (SONNET_5_5, OPUS_5_5)  # difference Opus 5.5 minus Sonnet 5.5

CORRECTNESS, FABRICATION, RECONCILING, COMPLETENESS = rel.TRIAL_ANCHORS
ANCHORS: tuple[str, ...] = rel.TRIAL_ANCHORS
FAITHFULNESS = "faithfulness_minimum"
ANCHOR_LABELS: dict[str, str] = {
    CORRECTNESS: "correctness", FABRICATION: "fabrication", RECONCILING: "reconciling items",
    COMPLETENESS: "completeness", FAITHFULNESS: "faithfulness minimum",
}

# --- The cases ----------------------------------------------------------------------------------

EDGE, ADVERSARIAL = "edge", "adversarial"
POOLED = "difficulty_bearing"
CASE_COUNTS: dict[str, int] = {EDGE: 30, ADVERSARIAL: 24}   # pre-registration sections 2 and 4
N_CASES = sum(CASE_COUNTS.values())                         # 54
assert N_CASES == 54
T_LEVEL = 0.975

# --- Readings and labels ------------------------------------------------------------------------

REVIEW_SETTLED = "review_settled"
AUDIT_ADJUSTED = "audit_adjusted"
BEFORE_D_R9 = "review_settled_before_d_r9"
READINGS: tuple[str, ...] = (REVIEW_SETTLED, AUDIT_ADJUSTED, BEFORE_D_R9)
READING_LABELS: dict[str, str] = {
    REVIEW_SETTLED: "review-settled anchor scores, the reading H3 is evaluated on (decision D-A)",
    AUDIT_ADJUSTED: "audit-adjusted anchor scores, a sensitivity check, not the verdict",
    BEFORE_D_R9: "review-settled anchor scores with citation omissions scored 0, the rule before D-R9",
}
NOT_ROBUST_EXTRACTOR = "not robust to extractor error"
NOT_ROBUST_CITATION = "not robust to the citation rule"

CONFIRMED = "confirmed"
NOT_CONFIRMED = "review not confirmed"
SECOND_LOOK_EMPTY = "second look empty"
SECOND_LOOK_RESULTS: tuple[str, ...] = (CONFIRMED, NOT_CONFIRMED, SECOND_LOOK_EMPTY)
TESTED, NOT_TESTED = "tested", "not tested"


class Refused(Exception):
    """An input H3 cannot be computed on; the message says what is wrong."""


# --- Input --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseScores:
    """One case of one model: its case kind and, per trial anchor, the k epoch scores
    with epoch 1 first."""
    case_kind: str
    epochs: Mapping[str, Sequence[Any]]


ScoreTable = Mapping[str, Mapping[str, CaseScores]]   # model id -> case id -> scores


def _exact(value: Any, where: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise Refused(f"{where}: score {value!r} is not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise Refused(f"{where}: score {value!r} is not finite")
    exact = Fraction(value)
    if not 0 <= exact <= rel.FULL_MARKS:
        raise Refused(f"{where}: score {value!r} is outside 0 to {rel.FULL_MARKS}")
    return exact


@dataclass(frozen=True)
class _Checked:
    cases: tuple[str, ...]                                   # sorted case ids
    kinds: Mapping[str, str]                                 # case id -> edge or adversarial
    means: Mapping[str, Mapping[str, Mapping[str, Fraction]]]  # model -> case -> anchor -> epoch mean


def check_table(table: ScoreTable, *, k: int = rel.PREREGISTERED_K) -> _Checked:
    """Refuse unless the table holds exactly the four workers, the same 54 cases for
    each (30 edge and 24 adversarial, each case of one kind for every model), and k
    epoch scores per anchor per case, each a number from 0 to 3. Returns the exact
    epoch means, faithfulness minimum included."""
    if k < 1:
        raise Refused(f"k must be at least 1, got {k}")
    models = set(table)
    if models != set(WORKERS):
        missing = sorted(set(WORKERS) - models)
        extra = sorted(models - set(WORKERS))
        raise Refused(f"H3 needs exactly the four workers; missing {missing}, unexpected {extra}")
    first = sorted(table[WORKERS[0]])
    for m in WORKERS:
        got = sorted(table[m])
        if len(got) != N_CASES:
            raise Refused(f"{MODEL_LABELS[m]} has {len(got)} cases; H3 needs exactly {N_CASES}")
        if got != first:
            only_here = sorted(set(got) - set(first))[:3]
            only_there = sorted(set(first) - set(got))[:3]
            raise Refused(f"{MODEL_LABELS[m]} has a different case set from {MODEL_LABELS[WORKERS[0]]} "
                          f"(only here: {only_here}; only there: {only_there})")
    kinds: dict[str, str] = {}
    for cid in first:
        seen = {str(table[m][cid].case_kind) for m in WORKERS}
        if len(seen) != 1:
            raise Refused(f"case {cid} has case kinds {sorted(seen)} across models")
        kind = seen.pop()
        if kind not in CASE_COUNTS:
            raise Refused(f"case {cid} has case kind {kind!r}; H3 reads edge and adversarial cases only")
        kinds[cid] = kind
    for kind, want in CASE_COUNTS.items():
        have = sum(1 for v in kinds.values() if v == kind)
        if have != want:
            raise Refused(f"{have} {kind} cases; H3 needs {want} (30 edge and 24 adversarial)")
    means: dict[str, dict[str, dict[str, Fraction]]] = {}
    for m in WORKERS:
        means[m] = {}
        for cid in first:
            epochs = table[m][cid].epochs
            if set(epochs) != set(ANCHORS):
                raise Refused(f"{MODEL_LABELS[m]} case {cid}: anchors {sorted(epochs)}, need {sorted(ANCHORS)}")
            exact: dict[str, list[Fraction]] = {}
            for a in ANCHORS:
                vals = list(epochs[a])
                if len(vals) != k:
                    raise Refused(f"{MODEL_LABELS[m]} case {cid} {a}: {len(vals)} epoch scores, need k = {k}")
                exact[a] = [_exact(v, f"{MODEL_LABELS[m]} case {cid} {a} epoch {i}") for i, v in
                            enumerate(vals, start=1)]
            row = {a: sum(exact[a], Fraction(0)) / k for a in ANCHORS}
            row[FAITHFULNESS] = sum((min(f, r) for f, r in zip(exact[FABRICATION], exact[RECONCILING])),
                                    Fraction(0)) / k
            means[m][cid] = row
    return _Checked(cases=tuple(first), kinds=kinds, means=means)


# --- Paired differences -------------------------------------------------------------------------

@dataclass(frozen=True)
class Difference:
    """The paired difference higher minus lower over n cases: the exact mean, the
    sample SD of the per-case differences, SE = SD / sqrt(n) and the paired 95% t
    interval mean +/- t(0.975, n - 1) x SE."""
    n: int
    mean: Fraction
    sd: float
    se: float
    t_crit: float
    ci95: tuple[float, float]

    @property
    def mean_float(self) -> float:
        return float(self.mean)

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "mean": float(self.mean), "mean_exact": str(self.mean), "sd": self.sd, "se": self.se,
                "t_crit": self.t_crit, "ci95": list(self.ci95)}


def paired_difference(lower: Sequence[Fraction], higher: Sequence[Fraction]) -> Difference:
    """Per-case differences higher minus lower, paired by position."""
    if len(lower) != len(higher):
        raise Refused(f"paired values differ in length: {len(lower)} and {len(higher)}")
    n = len(lower)
    if n < 2:
        raise Refused(f"a paired t interval needs at least 2 cases, got {n}")
    diffs = [Fraction(h) - Fraction(lo) for lo, h in zip(lower, higher)]
    mean = sum(diffs, Fraction(0)) / n
    var = sum(((d - mean) ** 2 for d in diffs), Fraction(0)) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = t_ppf(T_LEVEL, n - 1)
    m = float(mean)
    return Difference(n=n, mean=mean, sd=sd, se=se, t_crit=t, ci95=(m - t * se, m + t * se))


def _diff(checked: _Checked, cases: Sequence[str], anchor: str, lower: str, higher: str) -> Difference:
    return paired_difference([checked.means[lower][c][anchor] for c in cases],
                             [checked.means[higher][c][anchor] for c in cases])


def step_name(anchor: str, lower: str, higher: str) -> str:
    return f"{MODEL_LABELS[lower]} <= {MODEL_LABELS[higher]} on {ANCHOR_LABELS[anchor]}"


@dataclass(frozen=True)
class Step:
    anchor: str
    lower: str
    higher: str
    difference: Difference

    @property
    def holds(self) -> bool:
        return self.difference.mean >= 0      # exact: a zero mean holds

    @property
    def name(self) -> str:
        return step_name(self.anchor, self.lower, self.higher)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.anchor, self.lower, self.higher)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "anchor": self.anchor, "lower": self.lower, "higher": self.higher,
                "holds": self.holds, **self.difference.as_dict()}


def _steps(checked: _Checked, cases: Sequence[str], anchors: Sequence[str]) -> tuple[Step, ...]:
    return tuple(Step(a, lo, hi, _diff(checked, cases, a, lo, hi)) for a in anchors for lo, hi in STEPS)


# --- Second-look gate ---------------------------------------------------------------------------

def second_look_gate(results: Mapping[str, str]) -> dict[str, Any]:
    """Section 2 and section 8 (decisions D-L and D-R19): every worker needs a second
    look result. Any "review not confirmed" makes H3 not tested; "second look empty"
    is tested under that label; otherwise H3 is tested on confirmed figures."""
    missing = [MODEL_LABELS[m] for m in WORKERS if m not in results]
    if missing:
        raise Refused(f"no second look result for {missing}; H3 needs one per worker")
    extra = sorted(set(results) - set(WORKERS))
    if extra:
        raise Refused(f"second look results for models outside the four workers: {extra}")
    bad = {MODEL_LABELS[m]: results[m] for m in WORKERS if results[m] not in SECOND_LOOK_RESULTS}
    if bad:
        raise Refused(f"unknown second look result {bad}; expected one of {list(SECOND_LOOK_RESULTS)}")
    failed = [m for m in WORKERS if results[m] == NOT_CONFIRMED]
    empty = [m for m in WORKERS if results[m] == SECOND_LOOK_EMPTY]
    per_model = {m: results[m] for m in WORKERS}
    if failed:
        names = ", ".join(MODEL_LABELS[m] for m in failed)
        return {"status": NOT_TESTED, "label": NOT_CONFIRMED, "per_model": per_model,
                "reason": f"second look failed its gate for {names} (review not confirmed, decision D-L)"}
    if empty:
        names = ", ".join(MODEL_LABELS[m] for m in empty)
        return {"status": TESTED, "label": SECOND_LOOK_EMPTY, "per_model": per_model,
                "reason": f"no second-look units for {names}: tested on figures labeled second look empty, "
                          "never confirmed (decision D-R19)"}
    return {"status": TESTED, "label": CONFIRMED, "per_model": per_model,
            "reason": "every worker's second look confirmed its review-settled figures"}


# --- H3 -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class H3Result:
    reading: str
    k: int
    cases: tuple[str, ...]
    gate: Mapping[str, Any]
    steps: tuple[Step, ...]                                 # the twelve, anchor by anchor
    model_means: Mapping[str, Mapping[str, Mapping[str, float]]]  # subset -> model -> anchor -> mean
    by_subset: Mapping[str, tuple[Step, ...]]               # edge, adversarial: beside only
    faithfulness_steps: tuple[Step, ...]                    # beside only
    saturation: Mapping[str, Mapping[str, Any]]             # anchor -> difference and flag
    generational: Mapping[str, Difference]                  # anchor -> Sonnet 5.5 minus Sonnet 5
    reasoning_modes: Mapping[str, str | None] = field(default_factory=dict)

    @property
    def tested(self) -> bool:
        return self.gate["status"] == TESTED

    @property
    def failing_steps(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if not s.holds)

    @property
    def ordering(self) -> frozenset[tuple[str, str, str]]:
        """The ordering this reading gives: the set of failing steps (empty when all
        twelve hold)."""
        return frozenset(s.key for s in self.failing_steps)

    @property
    def holds(self) -> bool | None:
        """True when all twelve steps hold; None when H3 is not tested."""
        return (not self.failing_steps) if self.tested else None

    @property
    def verdict(self) -> str:
        if not self.tested:
            return NOT_TESTED
        return "holds" if self.holds else "does not hold"


def _means(checked: _Checked, cases: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in WORKERS:
        out[m] = {a: float(sum((checked.means[m][c][a] for c in cases), Fraction(0)) / len(cases))
                  for a in (*ANCHORS, FAITHFULNESS)}
    return out


def evaluate_h3(table: ScoreTable, second_look: Mapping[str, str], *, reading: str = REVIEW_SETTLED,
                k: int = rel.PREREGISTERED_K, reasoning_modes: Mapping[str, str] | None = None) -> H3Result:
    """H3 on one reading of the anchor scores (section 2). The twelve steps decide it;
    everything else is reported beside. When the second-look gate fails, the figures
    are still computed and H3 is reported as not tested."""
    if reading not in READINGS:
        raise Refused(f"unknown reading {reading!r}; expected one of {list(READINGS)}")
    gate = second_look_gate(second_look)
    checked = check_table(table, k=k)
    cases = checked.cases
    subsets = {kind: tuple(c for c in cases if checked.kinds[c] == kind) for kind in CASE_COUNTS}
    lo, hi = SATURATION_PAIR
    saturation: dict[str, dict[str, Any]] = {}
    for a in ANCHORS:
        d = _diff(checked, cases, a, lo, hi)
        saturation[a] = {"difference": d, "within_one_se": abs(d.mean_float) <= d.se}
    modes = {m: (reasoning_modes or {}).get(m) for m in WORKERS}
    return H3Result(
        reading=reading, k=k, cases=cases, gate=gate,
        steps=_steps(checked, cases, ANCHORS),
        model_means={POOLED: _means(checked, cases), **{kind: _means(checked, subsets[kind]) for kind in subsets}},
        by_subset={kind: _steps(checked, subsets[kind], ANCHORS) for kind in subsets},
        faithfulness_steps=_steps(checked, cases, (FAITHFULNESS,)),
        saturation=saturation,
        generational={a: _diff(checked, cases, a, *GENERATIONAL) for a in ANCHORS},
        reasoning_modes=modes,
    )


def robustness_labels(settled: H3Result, audit: H3Result | None = None,
                      before_d_r9: H3Result | None = None) -> dict[str, Any]:
    """Compare the review-settled ordering (its set of failing steps, from which the
    verdict follows) with the audit-adjusted one and with the one under the rule
    before D-R9. A difference gives the section 2 label. A reading not given is
    reported as not compared."""
    if settled.reading != REVIEW_SETTLED:
        raise Refused(f"the first reading must be {REVIEW_SETTLED}, got {settled.reading}")
    out: dict[str, Any] = {"labels": []}
    for name, other, want, label in (("audit_adjusted", audit, AUDIT_ADJUSTED, NOT_ROBUST_EXTRACTOR),
                                     ("before_d_r9", before_d_r9, BEFORE_D_R9, NOT_ROBUST_CITATION)):
        if other is None:
            out[name] = {"compared": False}
            continue
        if other.reading != want:
            raise Refused(f"the {name} reading must be {want}, got {other.reading}")
        if other.cases != settled.cases or other.k != settled.k:
            raise Refused(f"the {name} reading covers other cases or another k than the review-settled one")
        only_settled = sorted(step_name(*key) for key in settled.ordering - other.ordering)
        only_other = sorted(step_name(*key) for key in other.ordering - settled.ordering)
        differs = bool(only_settled or only_other)
        out[name] = {"compared": True, "differs": differs,
                     "verdict_differs": (not settled.ordering) != (not other.ordering),
                     "failing_only_review_settled": only_settled, f"failing_only_{name}": only_other}
        if differs:
            out["labels"].append(label)
    return out


# --- Adapter from trials.TrialReport -------------------------------------------------------------

def table_from_reports(reports: Mapping[str, Any], *, before_d_r9: bool = False,
                       k: int = rel.PREREGISTERED_K) -> tuple[dict[str, dict[str, CaseScores]], dict[str, int]]:
    """The H3 input table from one trials.TrialReport per worker, all of one kind
    (review-settled or audit-adjusted). Keeps the edge and adversarial samples only.
    An unscored anchor counts as 0, as in trials.anchor_means and
    trials.subset_anchor_means; the count of such cells per model is returned beside
    the table. With before_d_r9, an epoch holding a P12 omission or a
    `citation_omission` verdict scores 0 on fabrication, the rule before D-R9 as
    trials.subset_anchor_means applies it. A sample with no case kind is refused.
    Missing epochs are left for check_table to refuse."""
    from . import trials

    kinds = {getattr(r, "kind", None) for r in reports.values()}
    if len(kinds) != 1 or not kinds <= {trials.REVIEW_SETTLED, trials.AUDIT_ADJUSTED}:
        raise Refused(f"H3 reads review-settled or audit-adjusted reports of one kind, got {sorted(map(str, kinds))}")
    table: dict[str, dict[str, CaseScores]] = {}
    unscored: dict[str, int] = {}
    for model, report in reports.items():
        if report.k != k:
            raise Refused(f"{model}: report has k = {report.k}, H3 needs k = {k}")
        cases: dict[str, CaseScores] = {}
        blank = 0
        for sid, sample in report.samples.items():
            kind = next((e.case_kind for e in sample.epochs if e.case_kind), None)
            if kind is None:
                raise Refused(f"{model}: sample {sid} has no case kind")
            if kind not in CASE_COUNTS:
                continue
            epochs: dict[str, list[float]] = {a: [] for a in ANCHORS}
            for e in sorted(sample.epochs, key=lambda e: e.epoch):
                for a in ANCHORS:
                    v = e.anchors.get(a)
                    if v is None:
                        blank += 1
                        v = 0.0
                    if a == FABRICATION and before_d_r9 and (e.p12_omissions or e.review_found_omissions):
                        v = 0.0
                    epochs[a].append(v)
            cases[sid] = CaseScores(case_kind=kind, epochs=epochs)
        table[model] = cases
        unscored[model] = blank
    return table, unscored


def h3_from_reports(settled: Mapping[str, Any], audit: Mapping[str, Any] | None, second_look: Mapping[str, str], *,
                    k: int = rel.PREREGISTERED_K, reasoning_modes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """H3 from the review-settled reports of the four workers, with the audit-adjusted
    reports (when given) and the reading before D-R9 beside it, and the robustness
    labels. `unscored` counts the anchor cells read as 0, per reading and model."""
    table, blank = table_from_reports(settled, k=k)
    old, old_blank = table_from_reports(settled, before_d_r9=True, k=k)
    result = evaluate_h3(table, second_look, reading=REVIEW_SETTLED, k=k, reasoning_modes=reasoning_modes)
    before = evaluate_h3(old, second_look, reading=BEFORE_D_R9, k=k, reasoning_modes=reasoning_modes)
    out: dict[str, Any] = {REVIEW_SETTLED: result, BEFORE_D_R9: before,
                           "unscored": {REVIEW_SETTLED: blank, BEFORE_D_R9: old_blank}}
    adjusted = None
    if audit is not None:
        atable, ablank = table_from_reports(audit, k=k)
        adjusted = evaluate_h3(atable, second_look, reading=AUDIT_ADJUSTED, k=k, reasoning_modes=reasoning_modes)
        out[AUDIT_ADJUSTED] = adjusted
        out["unscored"][AUDIT_ADJUSTED] = ablank
    out["robustness"] = robustness_labels(result, adjusted, before)
    return out


# --- Output -------------------------------------------------------------------------------------

def result_dict(result: H3Result) -> dict[str, Any]:
    """A JSON-ready view of one reading."""
    return {
        "reading": result.reading, "label": READING_LABELS[result.reading], "k": result.k,
        "n_cases": len(result.cases),
        "second_look": dict(result.gate), "verdict": result.verdict, "holds": result.holds,
        "failing_steps": [s.name for s in result.failing_steps],
        "steps": [s.as_dict() for s in result.steps],
        "beside": {
            "model_means": {sub: {MODEL_LABELS[m]: v for m, v in per.items()}
                            for sub, per in result.model_means.items()},
            "by_subset": {sub: [s.as_dict() for s in steps] for sub, steps in result.by_subset.items()},
            "faithfulness_steps": [s.as_dict() for s in result.faithfulness_steps],
            "saturation": {a: {"pair": f"{MODEL_LABELS[SATURATION_PAIR[1]]} minus {MODEL_LABELS[SATURATION_PAIR[0]]}",
                               "cross_model_saturation": v["within_one_se"], **v["difference"].as_dict()}
                           for a, v in result.saturation.items()},
            "generational": {a: {"pair": f"{MODEL_LABELS[GENERATIONAL[1]]} minus {MODEL_LABELS[GENERATIONAL[0]]}",
                                 "reasoning_mode": {MODEL_LABELS[m]: result.reasoning_modes.get(m)
                                                    for m in GENERATIONAL},
                                 **d.as_dict()} for a, d in result.generational.items()},
            "reasoning_modes": {MODEL_LABELS[m]: v for m, v in result.reasoning_modes.items()},
        },
    }


def _fmt_diff(d: Difference) -> str:
    lo, hi = d.ci95
    return f"{float(d.mean):+.4f} (paired 95% t [{lo:+.4f}, {hi:+.4f}], SE {d.se:.4f}, n = {d.n})"


def format_h3(result: H3Result) -> str:
    """One line per step, then the figures beside the verdict."""
    lines = [f"H3 on {READING_LABELS[result.reading]}: {result.verdict} "
             f"(second look: {result.gate['label']}; {result.gate['reason']})"]
    for s in result.steps:
        lines.append(f"  {'holds' if s.holds else 'FAILS'}  {s.name}: {_fmt_diff(s.difference)}")
    if result.failing_steps:
        lines.append("  failing steps: " + "; ".join(s.name for s in result.failing_steps))
    lines.append("Beside, not in the verdict:")
    for kind, steps in result.by_subset.items():
        for s in steps:
            lines.append(f"  {kind}: {s.name}: {_fmt_diff(s.difference)}")
    for s in result.faithfulness_steps:
        lines.append(f"  faithfulness: {s.name}: {_fmt_diff(s.difference)}")
    modes = ", ".join(f"{MODEL_LABELS[m]} {result.reasoning_modes.get(m) or 'not given'}" for m in GENERATIONAL)
    for a, d in result.generational.items():
        lines.append(f"  generational, Sonnet 5.5 minus Sonnet 5 on {ANCHOR_LABELS[a]}: {_fmt_diff(d)} "
                     f"(reasoning mode: {modes})")
    for a, v in result.saturation.items():
        flag = "cross-model saturation finding" if v["within_one_se"] else "not within one SE"
        lines.append(f"  Opus 5.5 minus Sonnet 5.5 on {ANCHOR_LABELS[a]}: {_fmt_diff(v['difference'])}: {flag}")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------------

def table_from_json(data: Any) -> dict[str, dict[str, CaseScores]]:
    if not isinstance(data, Mapping):
        raise Refused("a table file must hold an object keyed by model id")
    out: dict[str, dict[str, CaseScores]] = {}
    for model, cases in data.items():
        if not isinstance(cases, Mapping):
            raise Refused(f"{model}: cases must be an object keyed by case id")
        out[str(model)] = {}
        for cid, row in cases.items():
            if not isinstance(row, Mapping) or "case_kind" not in row or not isinstance(row.get("epochs"), Mapping):
                raise Refused(f"{model} case {cid}: needs case_kind and an epochs object")
            out[str(model)][str(cid)] = CaseScores(case_kind=str(row["case_kind"]), epochs=dict(row["epochs"]))
    return out


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {path}: {e}") from e


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m icfr_agent_evals.h3", description="H3 over the twelve paired steps "
                                "(pre-registration section 2). No model call.")
    p.add_argument("--settled", required=True, help="review-settled table (JSON)")
    p.add_argument("--audit", help="audit-adjusted table (JSON), the sensitivity check")
    p.add_argument("--before-d-r9", dest="before_d_r9", help="review-settled table with citation omissions scored 0")
    p.add_argument("--second-look", dest="second_look", required=True,
                   help="JSON object: model id -> confirmed | review not confirmed | second look empty")
    p.add_argument("--reasoning-modes", dest="reasoning_modes", help="JSON object: model id -> reasoning mode")
    p.add_argument("--k", type=int, default=rel.PREREGISTERED_K)
    p.add_argument("--json", dest="json_out", help="write the full result here")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        second_look = _read_json(args.second_look)
        modes = _read_json(args.reasoning_modes) if args.reasoning_modes else None
        if not isinstance(second_look, Mapping) or (modes is not None and not isinstance(modes, Mapping)):
            raise Refused("--second-look and --reasoning-modes must each hold a JSON object")
        settled = evaluate_h3(table_from_json(_read_json(args.settled)), second_look, reading=REVIEW_SETTLED,
                              k=args.k, reasoning_modes=modes)
        audit = None if not args.audit else evaluate_h3(
            table_from_json(_read_json(args.audit)), second_look, reading=AUDIT_ADJUSTED, k=args.k,
            reasoning_modes=modes)
        before = None if not args.before_d_r9 else evaluate_h3(
            table_from_json(_read_json(args.before_d_r9)), second_look, reading=BEFORE_D_R9, k=args.k,
            reasoning_modes=modes)
        robust = robustness_labels(settled, audit, before)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    text = [format_h3(settled)]
    text += [format_h3(r) for r in (audit, before) if r is not None]
    text.append("Robustness: " + ("; ".join(robust["labels"]) or "no label (every compared ordering agrees)"))
    print("\n".join(text))
    if args.json_out:
        payload = {r.reading: result_dict(r) for r in (settled, audit, before) if r is not None}
        payload["robustness"] = robust
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
