"""Judge-trust statistics: the gates of pre-registration section 7.

This module carries out the judge-trust rules fixed in pre-registration section 7
at the tag. It is tooling item (3) of the section 13 row "Tooling built after the
tag and before the first evaluation-split run" (decision D-R19, 2026-11-14). It
changes no rule. Every function is pure and deterministic: seeds are passed in and
recorded in the outputs, and nothing here calls a model or reads a log.

Input shape. A pass is a mapping from LLM-graded sub-dimension
(rubric.LLM_GRADED_SUBDIMENSIONS) to a mapping from output id (a string) to a raw
0 to 3 integer score. Two passes are paired by output id, and the id sets must
match exactly. Each statistic is computed per sub-dimension, and each gate applies
to the smaller value across sub-dimensions (for a lower-limit gate, the smaller
lower limit), never to a pooled or averaged statistic. The point estimate is
reported beside the lower limit.

Statistics:
  - icc_a1: ICC(A,1), two-way, absolute agreement, single rater (McGraw and Wong
    1996), with the two-sided McGraw-Wong interval on approximate degrees of
    freedom v.
  - icc_c1: ICC(C,1), two-way, consistency, single rater, with its exact F
    interval. Reported, never gated.
  - krippendorff_alpha: Krippendorff's alpha at the nominal, ordinal or interval
    level from the coincidence matrix, in exact integer arithmetic. Zero expected
    disagreement leaves alpha undefined (None), never 1.
  - bootstrap_alpha: the unit bootstrap of alpha, its percentile 95% interval and
    q, the share of bootstrap alphas below 0.80 (Hayes and Krippendorff 2007).
    Undefined resamples are dropped and counted.
  - paired_mean_difference: the signed mean of first minus second score with a
    paired 95% t interval.

Gates (pre-registration section 7):
  - validity_gate: ICC(A,1) grader versus human on the calibration subset passes
    only when the smaller lower limit is at least 0.60. The estimate is labeled
    stratified. The kappa gate applies only
    to binary checks the rubric defines at rubric lock; rubric.BINARY_CHECKS is
    empty, so it does not apply, and a non-empty set is refused here.
  - intra_rater_icc: the author's first-pass grades against the re-grade, ICC(A,1)
    with its McGraw-Wong interval, labeled intra-rater (test-retest) agreement.
  - reliability_gate: ICC(A,1) grader versus calibrator on the full calibrator
    sample passes only when the smaller lower limit is at least 0.60. The signed
    mean of grader minus calibrator with its paired 95% interval is reported and is
    not a gate.
  - reliability_exclusion: on a run whose reliability gate fails, each sub-dimension
    whose lower limit is below 0.60 is excluded from every reported claim for that
    run, with the exclusion and its statistic returned for disclosure. It is outside
    the judge remedy order and never triggers a grader change.
  - retest_alpha and run_test_retest: the grader scores the calibration subset
    exactly 3 times at an acceptance look, the acceptance-look scoring counted as
    the first, each pass with response caching recorded off and its grader
    configuration on the ladder step's pin (grader.check_grader_config). Ordinal
    alpha passes at 0.80 or above. An undefined alpha is reported
    as undefined beside raw agreement and fails the gate.
  - bias_audit: per sub-dimension, a perturbed pass's shift is the mean over the
    subset's outputs of the perturbed score minus the mean of that output's three
    test-retest scores. The baseline is the largest absolute difference between the
    subset means of any two of the three test-retest passes. The audit fails if any
    pass's absolute shift minus the baseline exceeds 0.10, or if the magnitude of
    leniency (grader acceptance-look score minus human score) exceeds 0.25. The gap
    between ICC(C,1) and ICC(A,1) is reported beside leniency. Shifts, baselines and
    leniency are exact fractions, so a value at the threshold is never flipped by
    float error.

A call that breaks a written rule (a pass count other than 3, cached responses, a
grader configuration off the step's pin, mismatched output ids, a score outside 0 to 3) raises
Refused and computes nothing.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

from .regression import _reg_inc_beta, t_ppf
from .rubric import BINARY_CHECKS, LLM_GRADED_SUBDIMENSIONS

CONFIDENCE = 0.95
SCORE_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
ICC_LOWER_LIMIT = 0.60                 # validity and reliability gates, section 7
RETEST_ALPHA_MIN = Fraction(4, 5)      # test-retest ordinal alpha, section 7
RETEST_PASSES = 3                      # exactly 3 passes per acceptance look, decision D-R19
SHIFT_MARGIN = Fraction(1, 10)         # bias audit: shift minus baseline, rubric points
LENIENCY_LIMIT = Fraction(1, 4)        # bias audit: magnitude of leniency, rubric points
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000   # the text fixes none for alpha; this matches the kappa rule's count
RELABEL: dict[int, str] = {0: "D", 1: "C", 2: "B", 3: "A"}
UNRELABEL: dict[str, int] = {label: score for score, label in RELABEL.items()}

OPTION_ORDER_REVERSED = "option_order_reversed"
OPTION_ORDER_PERMUTED = "option_order_permuted"
LABELS_RELABELED = "labels_relabeled"
DIMENSION_ORDER_PERMUTED = "dimension_order_permuted"
BIAS_PASSES: tuple[str, ...] = (OPTION_ORDER_REVERSED, OPTION_ORDER_PERMUTED, LABELS_RELABELED)

REMEDY_NOTE = ("outside the judge remedy order: computed from evaluation-split generations, it never "
               "triggers a grader change (pre-registration section 7, decision D-R19)")
KAPPA_NOTE = "does not apply: the rubric defines no binary checks at rubric lock (pre-registration section 7)"
STRATIFIED_LABEL = ("stratified: the calibration subset is stratified so every score level appears at "
                    "least 4 times per sub-dimension, so the estimate is labeled stratified "
                    "(pre-registration sections 4 and 7)")
INTRA_RATER_LABEL = "intra-rater (test-retest) agreement"

Pass = Mapping[str, Mapping[str, int]]


class Refused(ValueError):
    """A refusal: the inputs break a written rule, so no statistic is computed."""


# --- F distribution ---------------------------------------------------------------------------

def f_cdf(x: float, d1: float, d2: float) -> float:
    """F distribution CDF via the regularized incomplete beta function of regression.py."""
    if not (d1 > 0 and d2 > 0 and math.isfinite(d1) and math.isfinite(d2)) or math.isnan(x):
        raise ValueError(f"f_cdf needs a number x and finite d1, d2 > 0, got x={x}, d1={d1}, d2={d2}")
    if x <= 0:
        return 0.0
    if math.isinf(x):
        return 1.0
    den = d1 * x + d2
    return _reg_inc_beta(d1 / 2.0, d2 / 2.0, d1 * x / den, d2 / den)


def f_ppf(q: float, d1: float, d2: float) -> float:
    """F quantile by bisection on the beta variable z = d1 x / (d1 x + d2); q in (0, 1)."""
    if not 0.0 < q < 1.0 or not (d1 > 0 and d2 > 0 and math.isfinite(d1) and math.isfinite(d2)):
        raise ValueError(f"f_ppf needs 0 < q < 1 and finite d1, d2 > 0, got q={q}, d1={d1}, d2={d2}")
    a, b = d1 / 2.0, d2 / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(400):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if _reg_inc_beta(a, b, mid, 1.0 - mid) < q:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    return d2 * z / (d1 * (1.0 - z))


# --- Input checks -----------------------------------------------------------------------------

def _number(value: Any, where: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise Refused(f"{where}: a score must be a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise Refused(f"{where}: a score must be finite, got {value!r}")
    return Fraction(value)


def _score(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SCORE_LEVELS:
        raise Refused(f"{where}: a score must be a raw 0 to 3 integer, got {value!r}")
    return value


def _subdimensions(subdimensions: Sequence[str] | None) -> tuple[str, ...]:
    names = tuple(LLM_GRADED_SUBDIMENSIONS if subdimensions is None else subdimensions)
    if not names:
        raise Refused("no LLM-graded sub-dimension given")
    if len(set(names)) != len(names):
        raise Refused(f"a sub-dimension is listed twice: {list(names)}")
    unknown = [n for n in names if n not in LLM_GRADED_SUBDIMENSIONS]
    if unknown:
        raise Refused(f"not an LLM-graded sub-dimension of the rubric: {unknown}")
    return names


def check_pass(scores: Pass, subdimensions: Sequence[str] | None = None, *, label: str = "pass") -> dict[str, dict[str, int]]:
    """Validate one pass and return a plain copy: every listed sub-dimension present and no
    other, at least one output each, string output ids and raw 0 to 3 integer scores."""
    names = _subdimensions(subdimensions)
    if not isinstance(scores, Mapping):
        raise Refused(f"{label}: a pass maps sub-dimension to {{output id: score}}")
    if set(scores) != set(names):
        raise Refused(f"{label}: sub-dimensions {sorted(scores)} differ from {sorted(names)}")
    out: dict[str, dict[str, int]] = {}
    for name in names:
        per = scores[name]
        if not isinstance(per, Mapping) or not per:
            raise Refused(f"{label}: {name} holds no scored output")
        row: dict[str, int] = {}
        for oid, value in per.items():
            if not isinstance(oid, str) or not oid:
                raise Refused(f"{label}: {name}: an output id must be a non-empty string, got {oid!r}")
            row[oid] = _score(value, f"{label}: {name}: {oid}")
        out[name] = dict(sorted(row.items()))
    return out


def _paired(first: Mapping[str, int], second: Mapping[str, int], where: str) -> list[tuple[int, int]]:
    if set(first) != set(second):
        only_first = sorted(set(first) - set(second))
        only_second = sorted(set(second) - set(first))
        raise Refused(f"{where}: output ids differ (only in the first: {only_first[:5]}, "
                      f"only in the second: {only_second[:5]})")
    return [(first[oid], second[oid]) for oid in sorted(first)]


def score_distribution(scores: Iterable[int]) -> dict[str, int]:
    """Count of each 0 to 3 level, every level listed (pre-registration section 7 publishes it)."""
    counts = {str(level): 0 for level in SCORE_LEVELS}
    for value in scores:
        counts[str(_score(value, "score distribution"))] += 1
    return counts


def _exact(value: Fraction | None) -> str | None:
    return None if value is None else str(value)


# --- Two-way ANOVA and the ICC forms ----------------------------------------------------------

def two_way_anova(matrix: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Mean squares of the two-way model without replication on an n x k matrix (rows are the
    rated outputs, columns the raters), in exact fractions: msr (rows), msc (columns), mse."""
    rows = [list(r) for r in matrix]
    n = len(rows)
    if n < 2:
        raise Refused(f"ICC needs at least 2 rated outputs, got {n}")
    k = len(rows[0])
    if k < 2:
        raise Refused(f"ICC needs at least 2 raters, got {k}")
    if any(len(r) != k for r in rows):
        raise Refused("ICC needs a rectangular matrix: every output scored by every rater")
    x = [[_number(v, f"row {i}") for v in r] for i, r in enumerate(rows)]
    grand = sum(itertools.chain.from_iterable(x)) / (n * k)
    row_means = [sum(r) / k for r in x]
    col_means = [sum(x[i][j] for i in range(n)) / n for j in range(k)]
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((v - grand) ** 2 for v in itertools.chain.from_iterable(x))
    ss_error = ss_total - ss_rows - ss_cols
    return {"n": n, "k": k, "msr": ss_rows / (n - 1), "msc": ss_cols / (k - 1),
            "mse": ss_error / ((n - 1) * (k - 1))}


def _icc_result(form: str, anova: Mapping[str, Any], estimate: Fraction | None, lower: float | None,
                upper: float | None, confidence: float, note: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "form": form, "n": anova["n"], "k": anova["k"],
        "estimate": None if estimate is None else float(estimate), "estimate_exact": _exact(estimate),
        "ci_lower": lower, "ci_upper": upper, "confidence": confidence,
        "defined": estimate is not None and lower is not None,
        "msr": float(anova["msr"]), "msc": float(anova["msc"]), "mse": float(anova["mse"]),
        "note": note, **extra,
    }


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return float(confidence)


def icc_a1(matrix: Sequence[Sequence[Any]], *, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """ICC(A,1) with its two-sided McGraw-Wong interval (McGraw and Wong 1996, case 2A).

    estimate = (MSR - MSE) / (MSR + (k - 1) MSE + k (MSC - MSE) / n). With p the estimate,
    a = k p / (n (1 - p)), b = 1 + k p (n - 1) / (n (1 - p)) and
    v = (a MSC + b MSE)^2 / ((a MSC)^2 / (k - 1) + (b MSE)^2 / ((n - 1)(k - 1))):
      lower = n (MSR - F_L MSE) / (F_L (k MSC + (k n - k - n) MSE) + n MSR), F_L = F(1 - alpha/2; n - 1, v)
      upper = n (F_U MSR - MSE) / (k MSC + (k n - k - n) MSE + n F_U MSR), F_U = F(1 - alpha/2; v, n - 1)
    Every score equal leaves the estimate undefined (0 / 0). Two cases where v is 0 / 0 have
    an interval that does not depend on v: no residual and no rater variance (estimate 1,
    interval [1, 1]) and no residual and no output variance (estimate 0, interval [0, 0]).
    """
    confidence = _check_confidence(confidence)
    an = two_way_anova(matrix)
    n, k, msr, msc, mse = an["n"], an["k"], an["msr"], an["msc"], an["mse"]
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den == 0:
        return _icc_result("ICC(A,1)", an, None, None, None, confidence,
                           "undefined: every score is equal, so there is no variance", df_v=None)
    p = (msr - mse) / den
    if mse == 0 and msc == 0:
        return _icc_result("ICC(A,1)", an, p, 1.0, 1.0, confidence,
                           "no residual and no rater variance: the interval is [1, 1] for any v", df_v=None)
    if mse == 0 and msr == 0:
        return _icc_result("ICC(A,1)", an, p, 0.0, 0.0, confidence,
                           "no residual and no output variance: the interval is [0, 0] for any v", df_v=None)
    pf, msrf, mscf, msef = float(p), float(msr), float(msc), float(mse)
    a = k * pf / (n * (1.0 - pf))
    b = 1.0 + k * pf * (n - 1) / (n * (1.0 - pf))
    v_den = (a * mscf) ** 2 / (k - 1) + (b * msef) ** 2 / ((n - 1) * (k - 1))
    v = (a * mscf + b * msef) ** 2 / v_den if v_den > 0 else 0.0
    if not (v > 0 and math.isfinite(v)):
        return _icc_result("ICC(A,1)", an, p, None, None, confidence,
                           "interval undefined: the McGraw-Wong degrees of freedom v are not positive", df_v=None)
    q = 1.0 - (1.0 - confidence) / 2.0
    f_l = f_ppf(q, n - 1, v)
    f_u = f_ppf(q, v, n - 1)
    c = k * mscf + (k * n - k - n) * msef
    lower = n * (msrf - f_l * msef) / (f_l * c + n * msrf)
    upper = n * (f_u * msrf - msef) / (c + n * f_u * msrf)
    return _icc_result("ICC(A,1)", an, p, lower, upper, confidence, None, df_v=v, f_lower=f_l, f_upper=f_u)


def icc_c1(matrix: Sequence[Sequence[Any]], *, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """ICC(C,1) with its exact F interval (McGraw and Wong 1996, case 3).

    estimate = (MSR - MSE) / (MSR + (k - 1) MSE). With F0 = MSR / MSE and df2 = (n - 1)(k - 1):
    F_L = F0 / F(1 - alpha/2; n - 1, df2), F_U = F0 F(1 - alpha/2; df2, n - 1), and the limits
    are (F_L - 1) / (F_L + k - 1) and (F_U - 1) / (F_U + k - 1). No residual and no output
    variance leaves it undefined (0 / 0); no residual with output variance gives 1 on [1, 1].
    Reported, never gated (pre-registration section 7).
    """
    confidence = _check_confidence(confidence)
    an = two_way_anova(matrix)
    n, k, msr, mse = an["n"], an["k"], an["msr"], an["mse"]
    den = msr + (k - 1) * mse
    if den == 0:
        return _icc_result("ICC(C,1)", an, None, None, None, confidence,
                           "undefined: no residual and no output variance")
    p = (msr - mse) / den
    if mse == 0:
        return _icc_result("ICC(C,1)", an, p, 1.0, 1.0, confidence,
                           "no residual variance: the F ratio is infinite and the interval is [1, 1]")
    q = 1.0 - (1.0 - confidence) / 2.0
    df2 = (n - 1) * (k - 1)
    f0 = float(msr / mse)
    f_l = f0 / f_ppf(q, n - 1, df2)
    f_u = f0 * f_ppf(q, df2, n - 1)
    return _icc_result("ICC(C,1)", an, p, (f_l - 1.0) / (f_l + k - 1), (f_u - 1.0) / (f_u + k - 1),
                       confidence, None, f_ratio=f0)


# --- Paired mean difference -------------------------------------------------------------------

def paired_mean_difference(pairs: Sequence[tuple[Any, Any]], *, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Signed mean of first minus second over paired scores, with a paired two-sided t interval:
    mean +/- t(1 - alpha/2; n - 1) sd / sqrt(n). Fewer than 2 pairs leave the interval undefined;
    zero spread gives the interval [mean, mean]."""
    confidence = _check_confidence(confidence)
    diffs = [_number(a, "first") - _number(b, "second") for a, b in pairs]
    n = len(diffs)
    if n == 0:
        raise Refused("a paired mean difference needs at least one pair")
    mean = sum(diffs) / n
    out: dict[str, Any] = {"n": n, "mean": float(mean), "mean_exact": str(mean), "confidence": confidence,
                           "sd": None, "df": None, "ci_lower": None, "ci_upper": None}
    if n < 2:
        out["note"] = "interval undefined: fewer than 2 pairs"
        return out
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    half = 0.0 if var == 0 else t_ppf(1.0 - (1.0 - confidence) / 2.0, n - 1) * sd / math.sqrt(n)
    out.update(sd=sd, df=n - 1, ci_lower=float(mean) - half, ci_upper=float(mean) + half)
    return out


# --- Krippendorff's alpha ---------------------------------------------------------------------

_METRICS = ("nominal", "ordinal", "interval")


class _Coincidence:
    """Per-unit coincidence counts scaled by L = lcm(m_u - 1) so every entry is an integer."""

    def __init__(self, units: Sequence[Sequence[Any]], metric: str):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        self.metric = metric
        cleaned: list[list[int]] = []
        for i, unit in enumerate(units):
            values = []
            for v in unit:
                if v is None:
                    continue
                if isinstance(v, bool) or not isinstance(v, int):
                    raise Refused(f"unit {i}: alpha values must be integers or None, got {v!r}")
                values.append(v)
            cleaned.append(values)
        if not cleaned:
            raise Refused("alpha needs at least one unit")
        self.units = len(cleaned)
        pairable = [vals for vals in cleaned if len(vals) >= 2]
        self.cats = sorted({v for vals in pairable for v in vals})
        index = {c: i for i, c in enumerate(self.cats)}
        self.C = len(self.cats)
        self.L = math.lcm(*(len(vals) - 1 for vals in pairable)) if pairable else 1
        self.vectors: list[list[tuple[int, int]]] = []
        self.unanimous: list[bool] = []
        self.pairable: list[bool] = []
        for vals in cleaned:
            if len(vals) < 2:
                self.vectors.append([])
                self.unanimous.append(False)
                self.pairable.append(False)
                continue
            weight = self.L // (len(vals) - 1)
            cells: dict[int, int] = {}
            for a, b in itertools.permutations(range(len(vals)), 2):
                cell = index[vals[a]] * self.C + index[vals[b]]
                cells[cell] = cells.get(cell, 0) + weight
            self.vectors.append(sorted(cells.items()))
            self.unanimous.append(len(set(vals)) == 1)
            self.pairable.append(True)

    def totals(self, chosen: Iterable[int]) -> list[int]:
        o = [0] * (self.C * self.C)
        for u in chosen:
            for cell, count in self.vectors[u]:
                o[cell] += count
        return o

    def parts(self, o: Sequence[int]) -> tuple[int, int, int]:
        """(A, B, n') with alpha = 1 - (n' - L) A / B, all integers; B == 0 means undefined."""
        C, L = self.C, self.L
        marg = [sum(o[i * C:(i + 1) * C]) for i in range(C)]
        n_scaled = sum(marg)
        delta = [0] * (C * C)
        if self.metric == "ordinal":
            prefix = [0]
            for m in marg:
                prefix.append(prefix[-1] + m)
            for i in range(C):
                for j in range(i + 1, C):
                    d = (2 * (prefix[j + 1] - prefix[i]) - marg[i] - marg[j]) ** 2
                    delta[i * C + j] = delta[j * C + i] = d
        else:
            scale = 4 * L * L
            for i in range(C):
                for j in range(C):
                    if i != j:
                        delta[i * C + j] = scale * (1 if self.metric == "nominal" else (self.cats[i] - self.cats[j]) ** 2)
        a = sum(oc * dc for oc, dc in zip(o, delta) if oc and dc)
        b = sum(marg[i] * marg[j] * delta[i * C + j] for i in range(C) for j in range(C) if delta[i * C + j])
        return a, b, n_scaled


def krippendorff_alpha(units: Sequence[Sequence[Any]], *, metric: str = "ordinal") -> dict[str, Any]:
    """Krippendorff's alpha from the coincidence matrix, exact.

    units: one sequence per unit (rated output) holding each observer's (pass's) integer value,
    None where missing. A unit with fewer than 2 values is not pairable and contributes nothing.
    Ordinal metric: delta^2(c, k) = (sum of n_g for g from c to k - (n_c + n_k) / 2)^2.
    Zero expected disagreement (a single class, or no pairable value) leaves alpha undefined:
    alpha is None, never 1 (pre-registration section 7, decision D-R19).
    raw_agreement is the share of pairable value pairs that agree (the coincidence diagonal
    over n); unit_agreement is the share of pairable units whose values all agree.
    """
    co = _Coincidence(units, metric)
    o = co.totals(range(co.units))
    a, b, n_scaled = co.parts(o)
    pairable_units = sum(co.pairable)
    raw = Fraction(sum(o[i * co.C + i] for i in range(co.C)), n_scaled) if n_scaled else None
    unit_agree = Fraction(sum(co.unanimous), pairable_units) if pairable_units else None
    alpha = None if b == 0 else 1 - Fraction((n_scaled - co.L) * a, b)
    note = None
    if alpha is None:
        note = ("undefined: zero expected disagreement (a single class)" if n_scaled
                else "undefined: no pairable values")
    return {
        "metric": metric, "alpha": None if alpha is None else float(alpha), "alpha_exact": _exact(alpha),
        "defined": alpha is not None, "units": co.units, "pairable_units": pairable_units,
        "pairable_values": n_scaled // co.L if co.L else 0,
        "raw_agreement": None if raw is None else float(raw), "raw_agreement_exact": _exact(raw),
        "unit_agreement": None if unit_agree is None else float(unit_agree),
        "unit_agreement_exact": _exact(unit_agree), "note": note,
    }


def bootstrap_alpha(units: Sequence[Sequence[Any]], *, seed: int, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                    metric: str = "ordinal", threshold: Fraction = RETEST_ALPHA_MIN,
                    confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Unit bootstrap of alpha: units drawn with replacement (random.Random(seed)), alpha recomputed
    on each resample. Percentile interval: lower = sorted[floor(alpha/2 B)], upper =
    sorted[ceil((1 - alpha/2) B) - 1] over the B defined resamples. q is the share of defined
    resamples whose alpha is below threshold (Hayes and Krippendorff 2007), compared exactly.
    Undefined resamples (zero expected disagreement) are dropped and counted."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Refused(f"the bootstrap seed must be an integer, got {seed!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise Refused(f"resamples must be a positive integer, got {resamples!r}")
    confidence = _check_confidence(confidence)
    threshold = Fraction(threshold)
    co = _Coincidence(units, metric)
    rng = random.Random(seed)
    N = co.units
    values: list[float] = []
    below = undefined = 0
    num, den = threshold.numerator, threshold.denominator
    for _ in range(resamples):
        a, b, n_scaled = co.parts(co.totals(rng.choices(range(N), k=N)))
        if b == 0:
            undefined += 1
            continue
        r_num = (n_scaled - co.L) * a          # alpha = 1 - r_num / b
        values.append(1.0 - r_num / b)
        if r_num * den > (den - num) * b:      # alpha < num / den, exactly
            below += 1
    values.sort()
    defined = len(values)
    tail = (1.0 - confidence) / 2.0
    lower = values[math.floor(tail * defined)] if defined else None
    upper = values[math.ceil((1.0 - tail) * defined) - 1] if defined else None
    return {"seed": seed, "resamples": resamples, "defined_resamples": defined, "undefined_resamples": undefined,
            "ci_lower": lower, "ci_upper": upper, "confidence": confidence,
            "q": below / defined if defined else None, "q_threshold": str(threshold), "metric": metric}


# --- Gate helpers -----------------------------------------------------------------------------

def lower_limit_gate(per_subdimension: Mapping[str, Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Apply a lower-limit gate to the smaller lower limit across sub-dimensions. The gate passes
    only when every lower limit is defined and the smallest is at least threshold. Ties go to the
    sub-dimension listed first. The point estimate is reported beside the lower limit."""
    if not per_subdimension:
        raise Refused("a gate needs at least one sub-dimension")
    undefined = [name for name, s in per_subdimension.items() if s.get("ci_lower") is None]
    gating = None
    for name, s in per_subdimension.items():
        if s.get("ci_lower") is not None and (gating is None or s["ci_lower"] < per_subdimension[gating]["ci_lower"]):
            gating = name
    passed = not undefined and gating is not None and per_subdimension[gating]["ci_lower"] >= threshold
    return {
        "threshold": threshold, "passed": passed, "undefined": undefined,
        "gating_subdimension": undefined[0] if undefined else gating,
        "gating_lower_limit": None if undefined or gating is None else per_subdimension[gating]["ci_lower"],
        "gating_estimate": (None if undefined or gating is None else per_subdimension[gating].get("estimate")),
    }


def _two_rater(first: Pass, second: Pass, subdimensions: Sequence[str] | None, labels: tuple[str, str],
               confidence: float) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    names = _subdimensions(subdimensions)
    a = check_pass(first, names, label=labels[0])
    b = check_pass(second, names, label=labels[1])
    per: dict[str, dict[str, Any]] = {}
    for name in names:
        pairs = _paired(a[name], b[name], f"{name}: {labels[0]} versus {labels[1]}")
        per[name] = {
            "icc_a1": icc_a1(pairs, confidence=confidence),
            "icc_c1": icc_c1(pairs, confidence=confidence),
            "mean_difference": paired_mean_difference(pairs, confidence=confidence),
            "distribution": {labels[0]: score_distribution(x for x, _ in pairs),
                             labels[1]: score_distribution(y for _, y in pairs)},
        }
    return names, per


def validity_gate(grader: Pass, human: Pass, *, subdimensions: Sequence[str] | None = None,
                  confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Validity, grader versus human on the calibration subset (pre-registration section 7):
    ICC(A,1) on raw 0 to 3 scores passes only when the smaller two-sided 95% McGraw-Wong lower
    limit is at least 0.60. ICC(C,1) with its exact F interval and leniency are reported beside
    it. The kappa gate does not apply while rubric.BINARY_CHECKS is empty."""
    if BINARY_CHECKS:
        raise Refused("the rubric defines binary checks, and their kappa gate is not built here; "
                      "the validity gate cannot be computed without it (pre-registration section 7)")
    names, per = _two_rater(grader, human, subdimensions, ("grader", "human"), confidence)
    gate = lower_limit_gate({n: per[n]["icc_a1"] for n in names}, ICC_LOWER_LIMIT)
    return {"gate": "validity", "statistic": "ICC(A,1) grader versus human", "subdimensions": list(names),
            "per_subdimension": per, **gate, "kappa_gate": KAPPA_NOTE, "estimate_label": STRATIFIED_LABEL}


def intra_rater_icc(matrix: Sequence[Sequence[Any]], *, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """ICC(A,1) of the author's first-pass grades against the author's re-grade on one
    LLM-graded sub-dimension, with its two-sided McGraw-Wong interval (icc_a1). The intra-rater
    check of pre-registration section 7: reported as intra-rater (test-retest) agreement, never
    as agreement between two raters. matrix: one row per output, the first-pass grade then the
    re-grade."""
    return {**icc_a1(matrix, confidence=confidence), "agreement": INTRA_RATER_LABEL}


def reliability_gate(grader: Pass, calibrator: Pass, *, subdimensions: Sequence[str] | None = None,
                     confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Reliability, grader versus calibrator on a run's full calibrator sample (pre-registration
    section 7): ICC(A,1) passes only when the smaller lower limit is at least 0.60. The signed mean
    of grader minus calibrator with its paired 95% t interval is reported, not gated."""
    names, per = _two_rater(grader, calibrator, subdimensions, ("grader", "calibrator"), confidence)
    gate = lower_limit_gate({n: per[n]["icc_a1"] for n in names}, ICC_LOWER_LIMIT)
    return {"gate": "reliability", "statistic": "ICC(A,1) grader versus calibrator",
            "subdimensions": list(names), "per_subdimension": per, **gate, "remedy_order": REMEDY_NOTE}


def reliability_exclusion(grader: Pass, calibrator: Pass, *, subdimensions: Sequence[str] | None = None,
                          confidence: float = CONFIDENCE) -> dict[str, Any]:
    """The per-run reliability exclusion (pre-registration section 7, decision D-R19): each
    sub-dimension whose ICC(A,1) lower limit is below 0.60 is excluded from every reported claim
    for that run, and the deterministic anchors stand alone for it. Each exclusion is returned with
    its statistic for disclosure. It is outside the judge remedy order and never triggers a grader
    change. An undefined lower limit is refused: the text does not say whether it excludes."""
    gate = reliability_gate(grader, calibrator, subdimensions=subdimensions, confidence=confidence)
    if gate["undefined"]:
        raise Refused(f"ICC(A,1) lower limit undefined for {gate['undefined']}: pre-registration section 7 "
                      "excludes a sub-dimension whose lower limit is below 0.60 and does not fix this case")
    kept, excluded = [], []
    for name in gate["subdimensions"]:
        stat = gate["per_subdimension"][name]["icc_a1"]
        if stat["ci_lower"] < ICC_LOWER_LIMIT:
            excluded.append({"subdimension": name, "statistic": "ICC(A,1) grader versus calibrator",
                             "estimate": stat["estimate"], "ci_lower": stat["ci_lower"],
                             "ci_upper": stat["ci_upper"], "n": stat["n"], "threshold": ICC_LOWER_LIMIT,
                             "disclosure": (f"{name} excluded from every reported claim for this run: "
                                            f"ICC(A,1) {stat['estimate']:.3f}, lower limit {stat['ci_lower']:.3f} "
                                            f"below {ICC_LOWER_LIMIT:.2f}; the deterministic anchors stand alone")})
        else:
            kept.append(name)
    return {"gate_passed": gate["passed"], "kept": kept, "excluded": excluded,
            "triggers_grader_change": False, "remedy_order": REMEDY_NOTE, "gate": gate}


# --- Test-retest ------------------------------------------------------------------------------

def _three_passes(passes: Sequence[Pass], names: tuple[str, ...]) -> list[dict[str, dict[str, int]]]:
    if isinstance(passes, Mapping) or len(passes) != RETEST_PASSES:
        raise Refused(f"test-retest takes exactly {RETEST_PASSES} passes, the acceptance-look scoring "
                      f"counted as the first; got {len(passes)} (decision D-R19: no replicate is added or dropped)")
    checked = [check_pass(p, names, label=f"pass {i}") for i, p in enumerate(passes, start=1)]
    for name in names:
        for i, p in enumerate(checked[1:], start=2):
            _paired(checked[0][name], p[name], f"{name}: pass 1 versus pass {i}")
    return checked


def retest_alpha(passes: Sequence[Pass], *, bootstrap_seed: int, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                 subdimensions: Sequence[str] | None = None, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Ordinal alpha over the 3 test-retest passes per sub-dimension, with its unit bootstrap,
    q and raw agreement. The gate applies to the smaller alpha: it passes only when every alpha is
    defined and the smallest is at least 0.80, compared exactly. An undefined alpha is reported as
    undefined beside raw agreement and fails the gate, never alpha = 1."""
    names = _subdimensions(subdimensions)
    checked = _three_passes(passes, names)
    per: dict[str, dict[str, Any]] = {}
    for name in names:
        ids = sorted(checked[0][name])
        units = [[p[name][oid] for p in checked] for oid in ids]
        stat = krippendorff_alpha(units, metric="ordinal")
        stat["bootstrap"] = bootstrap_alpha(units, seed=bootstrap_seed, resamples=resamples, metric="ordinal",
                                            threshold=RETEST_ALPHA_MIN, confidence=confidence)
        stat["distribution"] = {f"pass {i}": score_distribution(p[name].values())
                                for i, p in enumerate(checked, start=1)}
        stat["passed"] = stat["defined"] and Fraction(stat["alpha_exact"]) >= RETEST_ALPHA_MIN
        per[name] = stat
    undefined = [n for n in names if not per[n]["defined"]]
    defined = [n for n in names if per[n]["defined"]]
    gating = min(defined, key=lambda n: Fraction(per[n]["alpha_exact"])) if defined else None
    gating = undefined[0] if undefined else gating
    passed = not undefined and all(per[n]["passed"] for n in names)
    return {"gate": "test-retest", "statistic": "Krippendorff's alpha, ordinal, over 3 grader passes",
            "threshold": float(RETEST_ALPHA_MIN), "threshold_exact": str(RETEST_ALPHA_MIN),
            "subdimensions": list(names), "per_subdimension": per, "passed": passed, "undefined": undefined,
            "gating_subdimension": gating,
            "gating_alpha": None if undefined or gating is None else per[gating]["alpha"],
            "bootstrap_seed": bootstrap_seed, "resamples": resamples,
            "note": "a stability screen: no reliance claim rests on it alone (pre-registration section 7)"}


CONFIG_CACHING_KEY = "response_caching"
CONFIG_MODEL_KEY = "model"


def check_retest_config(cfg: Any, step: Any, *, label: str = "pass") -> dict[str, Any]:
    """Refuse a test-retest pass's grader configuration that is off the section 5 pin.

    cfg: the configuration the pass recorded, as data. It must record response caching
    disabled ("response_caching": False). It may name the grader model ("model"), which must
    then be the step's pin (grader.grader_pin). Every other key is a GenerateConfig field and
    the set must pass grader.check_grader_config(fields, step), which compares it with
    roster.grader_generate_config_args(step): reasoning_effort on the pin, no sampling
    parameter, batch as the step pins it. Returns the GenerateConfig fields."""
    from . import grader, roster

    if isinstance(step, bool) or step not in roster.GRADER_STEPS:
        raise Refused(f"the grader ladder step must be one of {roster.GRADER_STEPS}, got {step!r} "
                      "(pre-registration section 5)")
    if not isinstance(cfg, Mapping):
        raise Refused(f"{label}: the grader configuration must be a mapping")
    if cfg.get(CONFIG_CACHING_KEY) is not False:
        raise Refused(f"{label}: response caching must be recorded as disabled (\"{CONFIG_CACHING_KEY}\": False) "
                      "for test-retest runs (pre-registration sections 5 and 7)")
    model = cfg.get(CONFIG_MODEL_KEY)
    if model is not None and model != grader.grader_pin(step):
        raise Refused(f"{label}: the grader model {model!r} is not the step {step} pin {grader.grader_pin(step)!r} "
                      "(pre-registration section 5)")
    fields = {k: v for k, v in cfg.items() if k not in (CONFIG_CACHING_KEY, CONFIG_MODEL_KEY)}
    try:
        return grader.check_grader_config(fields, step, where=f"test-retest {label}")
    except roster.RunConfigError as e:
        raise Refused(f"{label}: the grader configuration is off the step {step} pin "
                      f"{roster.grader_generate_config_args(step)} ({e})") from e


def run_test_retest(acceptance_look: Pass, score_pass: Callable[[int], Pass], *, step: int,
                    grader_configs: Sequence[Mapping[str, Any]], bootstrap_seed: int, passes: int = RETEST_PASSES,
                    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES, subdimensions: Sequence[str] | None = None,
                    confidence: float = CONFIDENCE) -> dict[str, Any]:
    """The exactly-3-pass test-retest runner (pre-registration section 7, decision D-R19).

    acceptance_look is pass 1. score_pass(2) and score_pass(3) score passes 2 and 3; it is called
    exactly twice, in that order. step is the grader ladder step of the acceptance look.
    grader_configs holds the grader configuration of each of the 3 passes as data; each must record
    response caching disabled ("response_caching": False) and be on the step's pin
    (check_retest_config: grader.check_grader_config against roster.grader_generate_config_args),
    and all 3 must be equal (the configuration pinned, pre-registration sections 5 and 7). Any pass
    count other than 3 is refused, as is a later pass whose output ids or sub-dimensions differ from
    pass 1. The record holds every pass, so nothing is added or dropped after the look."""
    if isinstance(passes, bool) or passes != RETEST_PASSES:
        raise Refused(f"test-retest runs exactly {RETEST_PASSES} passes at each acceptance look, got {passes!r} "
                      "(pre-registration section 7, decision D-R19)")
    names = _subdimensions(subdimensions)
    configs = list(grader_configs)
    if len(configs) != RETEST_PASSES:
        raise Refused(f"one grader configuration per pass is needed, {RETEST_PASSES} in all; got {len(configs)}")
    for i, cfg in enumerate(configs, start=1):
        check_retest_config(cfg, step, label=f"pass {i}")
    if any(dict(cfg) != dict(configs[0]) for cfg in configs[1:]):
        raise Refused("the grader configuration differs between passes; it is pinned for every pass "
                      "(pre-registration sections 5 and 7)")
    first = check_pass(acceptance_look, names, label="pass 1 (acceptance look)")
    scored = [first]
    for number in range(2, RETEST_PASSES + 1):
        result = check_pass(score_pass(number), names, label=f"pass {number}")
        for name in names:
            _paired(first[name], result[name], f"{name}: pass 1 versus pass {number}")
        scored.append(result)
    stats = retest_alpha(scored, bootstrap_seed=bootstrap_seed, resamples=resamples,
                         subdimensions=names, confidence=confidence)
    from . import roster

    return {"passes": RETEST_PASSES, "pass_1": "acceptance-look scoring", "scores": scored,
            "grader_step": step, "grader_pin": roster.grader_generate_config_args(step),
            "grader_configs": [dict(c) for c in configs], "response_caching_disabled": [True] * RETEST_PASSES,
            "bootstrap_seed": bootstrap_seed, "alpha": stats}


# --- Bias audit -------------------------------------------------------------------------------

def reversed_options(options: Sequence[Any]) -> list[Any]:
    """Rubric options in reversed order: perturbation (a), first pass."""
    if len(options) < 2:
        raise Refused("an option order perturbation needs at least 2 options")
    return list(reversed(options))


def permuted_options(options: Sequence[Any], seed: int) -> dict[str, Any]:
    """One random permutation of the rubric options, drawn from seed (recorded in the run config
    before the look): perturbation (a), second pass. The draw is not repeated, so an identity or a
    reversal draw stands and is flagged for disclosure."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Refused(f"the permutation seed must be an integer, got {seed!r}")
    if len(options) < 2:
        raise Refused("an option order perturbation needs at least 2 options")
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    return {"seed": seed, "order": order, "options": [options[i] for i in order],
            "is_identity": order == sorted(order), "is_reversal": order == sorted(order, reverse=True)}


def relabel_map() -> dict[str, Any]:
    """Score labels relabeled, 0 to 3 as D, C, B, A, with the map back: perturbation (b)."""
    return {"label_of": dict(RELABEL), "score_of": dict(UNRELABEL)}


def unrelabel_pass(labeled: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, int]]:
    """Map a relabeled pass's letter labels back to 0 to 3 scores; an unknown label is refused."""
    out: dict[str, dict[str, int]] = {}
    for name, per in labeled.items():
        row = {}
        for oid, label in per.items():
            if not isinstance(label, str) or label not in UNRELABEL:
                raise Refused(f"{name}: {oid}: not a relabeled score label {sorted(UNRELABEL)}: {label!r}")
            row[oid] = UNRELABEL[label]
        out[name] = row
    return out


def permuted_dimensions(dimensions: Sequence[str], seed: int) -> dict[str, Any]:
    """Dimension order permuted, perturbation (c). It applies only where several dimensions share
    one grader prompt, so fewer than 2 dimensions are refused."""
    if len(dimensions) < 2:
        raise Refused("the dimension order perturbation applies only where several dimensions share one grader prompt")
    drawn = permuted_options(list(dimensions), seed)
    return {"seed": seed, "order": drawn["order"], "dimensions": drawn["options"], "is_identity": drawn["is_identity"]}


def bias_audit(test_retest_passes: Sequence[Pass], perturbed: Mapping[str, Pass], human: Pass, *,
               dimension_order_applies: bool, permutation_seed: int, dimension_seed: int | None = None,
               subdimensions: Sequence[str] | None = None, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """The pointwise bias audit (pre-registration section 7, decision D-R19).

    test_retest_passes: the 3 test-retest passes, the first being the acceptance-look scoring.
    perturbed: pass name -> pass, scores mapped back to 0 to 3; the names are BIAS_PASSES, plus
    DIMENSION_ORDER_PERMUTED exactly when dimension_order_applies. human: the human reference.
    Per sub-dimension: shift = mean over outputs of (perturbed score - mean of the output's 3
    test-retest scores); baseline = largest absolute difference between the subset means of any
    two test-retest passes; a pass fails when |shift| - baseline > 0.10. Leniency = signed mean of
    acceptance-look grader minus human score with its paired 95% t interval; it fails when
    |leniency| > 0.25. All compared as exact fractions. ICC(C,1) - ICC(A,1) is reported beside
    leniency. A failure blocks the reliability claim."""
    names = _subdimensions(subdimensions)
    retest = _three_passes(test_retest_passes, names)
    required = set(BIAS_PASSES) | ({DIMENSION_ORDER_PERMUTED} if dimension_order_applies else set())
    if set(perturbed) != required:
        raise Refused(f"bias-audit passes {sorted(perturbed)} differ from the required {sorted(required)}")
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
        raise Refused("the option permutation seed, recorded before the look, must be an integer")
    if dimension_order_applies and (isinstance(dimension_seed, bool) or not isinstance(dimension_seed, int)):
        raise Refused("the dimension permutation seed must be an integer when the dimension perturbation applies")
    order = [p for p in (*BIAS_PASSES, DIMENSION_ORDER_PERMUTED) if p in required]
    checked = {p: check_pass(perturbed[p], names, label=p) for p in order}
    ref = check_pass(human, names, label="human")
    per: dict[str, dict[str, Any]] = {}
    for name in names:
        ids = sorted(retest[0][name])
        n = len(ids)
        for p in order:
            _paired(retest[0][name], checked[p][name], f"{name}: pass 1 versus {p}")
        pairs = _paired(retest[0][name], ref[name], f"{name}: acceptance look versus human")
        retest_mean = {oid: Fraction(sum(r[name][oid] for r in retest), RETEST_PASSES) for oid in ids}
        pass_means = [Fraction(sum(r[name].values()), n) for r in retest]
        baseline = max(abs(x - y) for x, y in itertools.combinations(pass_means, 2))
        shifts: dict[str, dict[str, Any]] = {}
        for p in order:
            shift = sum((checked[p][name][oid] - retest_mean[oid] for oid in ids), Fraction(0)) / n
            excess = abs(shift) - baseline
            shifts[p] = {"shift": float(shift), "shift_exact": str(shift), "excess": float(excess),
                         "excess_exact": str(excess), "failed": excess > SHIFT_MARGIN}
        lenient = paired_mean_difference(pairs, confidence=confidence)
        lenient_failed = abs(Fraction(lenient["mean_exact"])) > LENIENCY_LIMIT
        a1, c1 = icc_a1(pairs, confidence=confidence), icc_c1(pairs, confidence=confidence)
        gap = None if a1["estimate"] is None or c1["estimate"] is None else (
            Fraction(c1["estimate_exact"]) - Fraction(a1["estimate_exact"]))
        per[name] = {
            "n": n, "baseline": float(baseline), "baseline_exact": str(baseline),
            "retest_pass_means": [str(m) for m in pass_means], "passes": shifts,
            "leniency": {**lenient, "failed": lenient_failed},
            "icc_c1_minus_icc_a1": None if gap is None else float(gap), "icc_c1_minus_icc_a1_exact": _exact(gap),
            "failed": lenient_failed or any(s["failed"] for s in shifts.values()),
        }
    passed = not any(per[n]["failed"] for n in names)
    return {"gate": "bias audit", "subdimensions": list(names), "per_subdimension": per, "passed": passed,
            "reliability_claim_blocked": not passed, "passes": order,
            "shift_margin": str(SHIFT_MARGIN), "leniency_limit": str(LENIENCY_LIMIT),
            "permutation_seed": permutation_seed,
            "dimension_seed": dimension_seed if dimension_order_applies else None,
            "dimension_order_applies": bool(dimension_order_applies)}
