"""Pure statistics of the full-run grading job: gate, bootstrap, breaker, queue and means.

Decision D-R23. Every figure here is computed and sealed now, and reported only after
the second-look freeze, through the later unseal-and-report step (D-R23, unseal step).
Nothing here imports Inspect, reads or writes a file, or prints. Every function takes
plain dicts and returns plain dicts that are safe to write as JSON. A fraction is given
as a float beside its exact string ("x_exact"), so a value at a threshold is never
flipped by float error.

Input shapes (the grading job builds them). A Reply is one judge's reply of record on
one unit: status "parsed", "unparsed", "refused" or "failed", with scores,
low-confidence flags and reasons per sub-dimension when parsed, and the raw reply and
the error when not. A failed judgment (status "failed") is a unit asked three times
with no completed call; it has no reply and is treated exactly as a reply that did not
parse (D-R23, failed judgments). A Row is one drawn calibrator row that was not lost in
the draw: unit id, model, sample id, epoch, position, and the grader and calibrator
Replies. A Unit is one gradeable generation under its opaque id, with its grader Reply.
A Generation is one (model, sample, epoch) of the worker logs with its status, case
kind and defect-epoch flag.

Reliability gate (D-R23, reliability gate). The gate is the one the pre-registration
states: ICC(A,1) grader versus calibrator on the complete pairs of the calibrator
sample, through judge_trust.reliability_exclusion. A sub-dimension is kept when its
lower limit is defined and reaches 0.60; otherwise it is excluded for the full run with
the reason judge_trust gives. A pair where either reply did not parse, a refusal or a
failed judgment included, is dropped, never asked again and never refilled. A figure
that cannot be computed (fewer than 2 complete pairs) excludes every sub-dimension as
undefined. Two figures are sealed beside the gate, reported and not gated. The first is
a worst-case figure that puts each dropped pair back at the largest disagreement it can
have: when one reply parsed with score s, the missing score is 0 for s of 2 or 3 and 3
for s of 0 or 1; when neither parsed, the pair is 0 against 3, the grader taking 0 on
the first such row in unit-id order and the direction alternating after that. Only its
icc_a1 is meaningful. The second is the break-even count: the fewest added dropped
pairs at the largest disagreement that bring the complete-pairs lower limit below 0.60.
Both blocks carry the label "reported, not gated". A row lost in the draw (no gradeable epoch) is not a dropped pair and gets no fill. A
case-cluster percentile bootstrap lower limit on the complete pairs is also reported
beside the gate, never gated.

Circuit breaker (D-R23, circuit breaker). The unit is the full run. The pair rate is
the tagged complete pairs over the complete pairs. The generation rate is the tagged
units over the units whose grader reply parsed, all four worker runs pooled, Opus 5.5
included. A unit is tagged for the breaker when its grader reply parsed and either
flags low confidence on a sub-dimension, or its row is complete and the two judges
differ by more than 1 point on a sub-dimension or the calibrator flags low confidence.
A failed judgment is never a breaker tag; a dropped pair leaves the pair rate, and its
generation stays in the generation rate on its grader reply alone. The breaker trips
when either rate is strictly above one quarter, compared exactly. An undefined rate does
not trip. A trip excludes both LLM sub-dimensions from every claim of the run, and the
anchors stand alone. Per-worker rates are descriptive and never trip.

Queue (D-R23, queue). A unit goes in the queue when its row disagrees by more than 1
point, when either judge flags low confidence, or when either judge's reply did not
parse. Each item carries empty disposition fields and no model id. A reply that did not
parse keeps its raw reply and its error in the item. When both judges parsed, the item
allows grader-correct, calibrator-correct, both-wrong and
ambiguous-edge-case-add-to-Verified-subset. When the grader reply did not parse and a
calibrator score exists, it allows calibrator-correct, both-wrong and
ambiguous-edge-case-add-to-Verified-subset. With no calibrator score and a parsed grader
reply, it allows grader-correct, grader-wrong and
ambiguous-edge-case-add-to-Verified-subset. With no calibrator score and a grader reply
that did not parse, it allows only grader-wrong and
ambiguous-edge-case-add-to-Verified-subset. Grader
scores are of record: no function here reads a disposition, so no disposition changes a
grade, a mean, the gate or a breaker rate.

Means (D-R23, means). Per worker model, subset and sub-dimension, two means: the
primary mean over graded generations, conditional on a parsed worker output, and a
sensitivity mean over all generations with worker format breaks (errored epochs
included) counted as 0. A grader reply that did not parse stays missing in both and is
counted. The point estimate is the mean of per-sample epoch means. Its 95 percent
interval is normal, from the standard error of the per-sample epoch means (the
trials.anchor_means method), and is clipped to the 0 to 3 scale, which the block says.
Each block also gives the count of samples scored on fewer than 4 epochs, and the D-R20
row repeats both means with every defect epoch left out. Cross-model paired differences
use the samples with a value for both models and are labelled by reading. No composed
dimension score is computed.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import random
import sys
from fractions import Fraction
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader_prompt as gp  # noqa: E402
from icfr_agent_evals import judge_trust as jt  # noqa: E402

GATE_THRESHOLD = jt.ICC_LOWER_LIMIT                 # 0.60 (judge_trust.py:84)
BREAKER_LIMIT = Fraction(1, 4)                      # trips when a rate exceeds it (strictly greater)
DISAGREEMENT_POINTS = 1                             # tagged when |grader - calibrator| > 1 on a sub-dimension
BOOTSTRAP_RESAMPLES = jt.DEFAULT_BOOTSTRAP_RESAMPLES  # 10,000 (judge_trust.py:89)
Z95 = 1.96
SUBSETS = ("all", "synthetic", "edge", "adversarial", "difficulty_bearing")   # as run_report.subsets_of
TAG_DISAGREEMENT = "disagreement_over_1_point"
TAG_GRADER_LOW = "grader_low_confidence"
TAG_CALIBRATOR_LOW = "calibrator_low_confidence"
TAG_GRADER_FAILED = "grader_reply_failed"
TAG_CALIBRATOR_FAILED = "calibrator_reply_failed"
TAG_ORDER = (TAG_DISAGREEMENT, TAG_GRADER_LOW, TAG_CALIBRATOR_LOW, TAG_GRADER_FAILED, TAG_CALIBRATOR_FAILED)
DISPOSITIONS_WITH_CALIBRATOR = ("grader-correct", "calibrator-correct", "both-wrong",
                                "ambiguous-edge-case-add-to-Verified-subset")   # the four of the pre-registration
DISPOSITIONS_NO_CALIBRATOR = ("grader-correct", "grader-wrong",
                              "ambiguous-edge-case-add-to-Verified-subset")     # D-R23, queue
DISPOSITIONS_GRADER_FAILED = ("calibrator-correct", "both-wrong",
                              "ambiguous-edge-case-add-to-Verified-subset")     # D-R23, queue
DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR = ("grader-wrong",
                                            "ambiguous-edge-case-add-to-Verified-subset")  # no score to be correct
EXCLUSION_BREAKER = "circuit breaker tripped"
LABEL_PRIMARY = "conditional on a parsed worker output"
LABEL_SENSITIVITY = "all generations, worker format breaks counted as 0"
LABEL_BOOTSTRAP = "reported, not gated"
LABEL_WORST_CASE = "reported, not gated"            # same text as LABEL_BOOTSTRAP
LABEL_BREAK_EVEN = LABEL_WORST_CASE                 # the break-even block carries the same label
LABEL_PAIRED = {
    "primary": "first minus second, samples with a primary value for both models",
    "sensitivity": ("first minus second, samples with a sensitivity value for both models, worker format "
                    "breaks and errored epochs counted as 0"),
}
WORST_CASE_NOTE = ("only icc_a1 is meaningful in this figure; every other value in it is computed on "
                   "filled scores")
GATE_RULE = ("kept when the ICC(A,1) lower limit on complete pairs reaches 0.60, as the pre-registration "
             "says; the worst-case figure and the break-even count are reported, not gated (D-R23, reliability gate)")
BREAK_EVEN_RULE = ("the fewest added dropped pairs, each at the largest possible disagreement (0 against 3, the "
                   "grader taking 0 on the first and the direction alternating), that bring the ICC(A,1) lower "
                   "limit of the complete pairs plus those pairs below 0.60 or make it undefined")
BREAK_EVEN_EXCLUDED = "excluded on complete pairs"
BREAK_EVEN_NONE = "no break-even up to the number of complete pairs"
CI_NOTE = "normal 95 percent interval, clipped to the 0 to 3 scale"
SCORE_MIN, SCORE_MAX = 0.0, 3.0                     # the 0 to 3 scale of every LLM sub-dimension

# Status strings, equal to grading_common's REPLY_* and GEN_* constants. This module
# imports no sibling script, so the values are repeated here.
_PARSED = "parsed"
_GEN_GRADEABLE, _GEN_ERRORED, _GEN_UNPARSED = "gradeable", "errored", "unparsed"
_FULL_EPOCHS = 4                                    # worker epochs per sample (grading_common.WORKER_EPOCHS)
_READINGS = ("primary", "sensitivity")


# --- Reply helpers --------------------------------------------------------------------------

def is_parsed(reply: Mapping | None) -> bool:
    """True when the reply exists and parsed."""
    return reply is not None and reply["status"] == _PARSED


def any_low(reply: Mapping | None) -> bool:
    """True when the reply parsed and flags low confidence on any sub-dimension."""
    return is_parsed(reply) and any(bool(v) for v in reply["low_confidence"].values())


def disagrees(g: Mapping | None, c: Mapping | None, dims: Sequence[str]) -> bool:
    """True when both replies parsed and differ by more than 1 point on some sub-dimension."""
    if not (is_parsed(g) and is_parsed(c)):
        return False
    return any(abs(g["scores"][d] - c["scores"][d]) > DISAGREEMENT_POINTS for d in dims)


def _rate(tagged: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"tagged": tagged, "denominator": 0, "rate": None, "rate_exact": None}
    r = Fraction(tagged, denominator)
    return {"tagged": tagged, "denominator": denominator, "rate": float(r), "rate_exact": str(r)}


def _by_uid(items: Sequence[Mapping], key: str, what: str) -> dict[str, Mapping]:
    out: dict[str, Mapping] = {}
    for item in items:
        uid = item[key]
        if uid in out:
            raise jt.Refused(f"unit {uid} appears twice among the {what}")
        out[uid] = item
    return out


# --- Gate (D-R23, reliability gate) ---------------------------------------------------------

def complete_rows(rows: Sequence[Mapping]) -> list[Mapping]:
    """The rows whose grader and calibrator replies both parsed."""
    return [r for r in rows if is_parsed(r["grader"]) and is_parsed(r["calibrator"])]


def dropped_rows(rows: Sequence[Mapping]) -> list[Mapping]:
    """The rows where either reply did not parse: unparsed, refused or a failed judgment."""
    return [r for r in rows if not (is_parsed(r["grader"]) and is_parsed(r["calibrator"]))]


def _passes(rows: Sequence[Mapping], dims: Sequence[str]) -> tuple[dict, dict]:
    ordered = sorted(rows, key=lambda r: r["uid"])
    g = {d: {r["uid"]: r["grader"]["scores"][d] for r in ordered} for d in dims}
    c = {d: {r["uid"]: r["calibrator"]["scores"][d] for r in ordered} for d in dims}
    return g, c


def _far(score: int) -> int:
    """The level farthest from a parsed score: 0 for 2 or 3, 3 for 0 or 1."""
    return 0 if score in (2, 3) else 3


def fill_worst_case(rows: Sequence[Mapping], dims: Sequence[str] = gp.DIMENSIONS
                    ) -> tuple[dict, dict, list[dict]]:
    """The worst-case passes over every row: complete rows keep their scores, each dropped
    row is put back at the largest disagreement it can have. Returns (grader pass,
    calibrator pass, fills), each pass {dim: {uid: score}}; fills lists every filled value."""
    ordered = sorted(_by_uid(rows, "uid", "calibrator rows").values(), key=lambda r: r["uid"])
    g_pass: dict[str, dict[str, int]] = {d: {} for d in dims}
    c_pass: dict[str, dict[str, int]] = {d: {} for d in dims}
    fills: list[dict] = []
    neither_k = {d: 0 for d in dims}
    for r in ordered:
        uid, g, c = r["uid"], r["grader"], r["calibrator"]
        g_ok, c_ok = is_parsed(g), is_parsed(c)
        for d in dims:
            if g_ok and c_ok:
                gs_, cs_ = g["scores"][d], c["scores"][d]
            elif g_ok:
                gs_ = g["scores"][d]
                cs_ = _far(gs_)
                fills.append({"uid": uid, "dim": d, "grader": gs_, "calibrator": cs_, "rule": "one parsed"})
            elif c_ok:
                cs_ = c["scores"][d]
                gs_ = _far(cs_)
                fills.append({"uid": uid, "dim": d, "grader": gs_, "calibrator": cs_, "rule": "one parsed"})
            else:
                gs_, cs_ = (0, 3) if neither_k[d] % 2 == 0 else (3, 0)
                neither_k[d] += 1
                fills.append({"uid": uid, "dim": d, "grader": gs_, "calibrator": cs_, "rule": "neither parsed"})
            g_pass[d][uid] = gs_
            c_pass[d][uid] = cs_
    return g_pass, c_pass, fills


def one_figure(grader_pass: Mapping, calibrator_pass: Mapping, dims: Sequence[str]) -> dict[str, Any]:
    """judge_trust.reliability_exclusion on one pair of passes. A refusal there (no row, or
    fewer than 2 rows) gives a figure that was not computed, and every sub-dimension then
    counts as excluded as undefined."""
    try:
        exclusion = jt.reliability_exclusion(grader_pass, calibrator_pass, subdimensions=dims)
    except jt.Refused as e:
        return {"computed": False, "reason": str(e), "exclusion": None}
    return {"computed": True, "exclusion": exclusion}


def _figure_view(figure: Mapping, dim: str) -> tuple[bool, str | None, dict | None]:
    """(kept, reason when excluded, {"estimate", "ci_lower"} or None) of one sub-dimension."""
    if not figure["computed"]:
        return False, jt.EXCLUSION_UNDEFINED, None
    exclusion = figure["exclusion"]
    stat = exclusion["gate"]["per_subdimension"][dim]["icc_a1"]
    view = {"estimate": stat["estimate"], "ci_lower": stat["ci_lower"]}
    if dim in exclusion["kept"]:
        return True, None, view
    reason = next(e["reason"] for e in exclusion["excluded"] if e["subdimension"] == dim)
    return False, reason, view


def _lower_limit_or_none(matrix: Sequence[Sequence[int]]) -> float | None:
    """The ICC(A,1) lower limit of a matrix, None when it is undefined or cannot be computed."""
    try:
        return jt.icc_a1(matrix)["ci_lower"]
    except jt.Refused:
        return None


def break_even(complete: Sequence[Mapping], dim: str, *, kept_on_complete: bool) -> dict[str, Any]:
    """The break-even count of one sub-dimension (BREAK_EVEN_RULE): for k = 1 to the number
    of complete pairs, the complete pairs plus k rows at the largest disagreement, 0 against
    3 then 3 against 0 and so on, the grader first. The first k whose ICC(A,1) lower limit is
    undefined or below 0.60 is the count. Reported, not gated."""
    if not kept_on_complete:
        return {"count": 0, "searched_up_to": 0, "reason": BREAK_EVEN_EXCLUDED}
    base = [[r["grader"]["scores"][dim], r["calibrator"]["scores"][dim]]
            for r in sorted(complete, key=lambda r: r["uid"])]
    limit = len(base)
    added: list[list[int]] = []
    for k in range(1, limit + 1):
        added.append([0, 3] if k % 2 == 1 else [3, 0])
        lower = _lower_limit_or_none(base + added)
        if lower is None or lower < GATE_THRESHOLD:
            return {"count": k, "searched_up_to": k, "reason": None}
    return {"count": None, "searched_up_to": limit, "reason": BREAK_EVEN_NONE}


def reliability_gate(rows: Sequence[Mapping], dims: Sequence[str] = gp.DIMENSIONS) -> dict[str, Any]:
    """The reliability gate of the pre-registration: a sub-dimension is kept when the ICC(A,1)
    lower limit on complete pairs reaches 0.60. The worst-case figure and the break-even
    count of dropped pairs are computed and sealed beside it, reported and not gated
    (D-R23, reliability gate)."""
    dims = tuple(dims)
    _by_uid(rows, "uid", "calibrator rows")
    complete = complete_rows(rows)
    neither = sum(1 for r in rows if not is_parsed(r["grader"]) and not is_parsed(r["calibrator"]))
    complete_figure = one_figure(*_passes(complete, dims), dims)
    g_worst, c_worst, fills = fill_worst_case(rows, dims)
    worst_figure = {**one_figure(g_worst, c_worst, dims), "label": LABEL_WORST_CASE, "note": WORST_CASE_NOTE}
    kept: list[str] = []
    excluded: list[dict] = []
    even: dict[str, Any] = {"label": LABEL_BREAK_EVEN}
    for d in dims:
        c_kept, c_reason, c_view = _figure_view(complete_figure, d)
        even[d] = break_even(complete, d, kept_on_complete=c_kept)
        if c_kept:
            kept.append(d)
            continue
        excluded.append({"subdimension": d, "reason": c_reason, "complete": c_view})
    return {"rows": len(rows), "pairs_complete": len(complete), "pairs_dropped": len(rows) - len(complete),
            "neither_parsed": neither, "complete": complete_figure, "worst_case": worst_figure,
            "fills": fills, "break_even": even, "break_even_rule": BREAK_EVEN_RULE,
            "kept": kept, "excluded": excluded, "threshold": GATE_THRESHOLD, "rule": GATE_RULE}


# --- Case-cluster bootstrap (reported, not gated) -------------------------------------------

def icc_a1_estimate(matrix: Sequence[Sequence[Any]]) -> Fraction | None:
    """The ICC(A,1) point estimate in exact fractions, None when every score is equal.
    Equal to judge_trust.icc_a1's estimate_exact whenever that is defined."""
    an = jt.two_way_anova(matrix)
    n, k, msr, msc, mse = an["n"], an["k"], an["msr"], an["msc"], an["mse"]
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den == 0:
        return None
    return (msr - mse) / den


def case_cluster_bootstrap(rows: Sequence[Mapping], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES,
                           dims: Sequence[str] = gp.DIMENSIONS,
                           confidence: float = jt.CONFIDENCE) -> dict[str, dict[str, Any]]:
    """Percentile bootstrap lower limit of ICC(A,1) on the complete pairs, resampling cases
    with all their rows. One fresh random.Random(seed) per sub-dimension; each resample
    draws choices(range(C), k=C) over the C cases sorted by id. A resample with fewer than
    2 rows or an undefined estimate is dropped and counted. Reported, not gated."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise jt.Refused(f"the bootstrap seed must be an integer, got {seed!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise jt.Refused(f"resamples must be a positive integer, got {resamples!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    complete = sorted(complete_rows(rows), key=lambda r: r["uid"])
    clusters_by_id: dict[str, list[Mapping]] = {}
    for r in complete:
        clusters_by_id.setdefault(r["sample_id"], []).append(r)
    clusters = [clusters_by_id[s] for s in sorted(clusters_by_id)]
    n_clusters = len(clusters)
    tail = (1.0 - float(confidence)) / 2.0
    out: dict[str, dict[str, Any]] = {}
    for d in dims:
        cells = [[[r["grader"]["scores"][d], r["calibrator"]["scores"][d]] for r in cl] for cl in clusters]
        rng = random.Random(seed)
        values: list[float] = []
        undefined = 0
        for _ in range(resamples):
            picks = rng.choices(range(n_clusters), k=n_clusters)
            matrix = [row for p in picks for row in cells[p]]
            estimate = icc_a1_estimate(matrix) if len(matrix) >= 2 else None
            if estimate is None:
                undefined += 1
                continue
            values.append(float(estimate))
        values.sort()
        defined = len(values)
        out[d] = {"seed": seed, "resamples": resamples, "clusters": n_clusters, "rows": len(complete),
                  "defined_resamples": defined, "undefined_resamples": undefined,
                  "ci_lower": values[math.floor(tail * defined)] if defined else None,
                  "label": LABEL_BOOTSTRAP}
    return out


# --- Circuit breaker (D-R23, circuit breaker) ----------------------------------------------

def breaker_tagged(unit_grader: Mapping | None, row: Mapping | None, dims: Sequence[str]) -> bool:
    """Tagged for the breaker: the grader reply parsed, and it flags low confidence, or the
    row is complete and the judges differ by more than 1 point or the calibrator flags low
    confidence. A failed judgment is never a breaker tag."""
    if not is_parsed(unit_grader):
        return False
    if any_low(unit_grader):
        return True
    if row is None or not (is_parsed(row["grader"]) and is_parsed(row["calibrator"])):
        return False
    return disagrees(row["grader"], row["calibrator"], dims) or any_low(row["calibrator"])


def _breaker_rates(units: Sequence[Mapping], rows: Sequence[Mapping], rows_by_uid: Mapping[str, Mapping],
                   dims: Sequence[str]) -> tuple[dict, dict]:
    """(pair rate, generation rate). A dropped pair leaves the pair rate; its generation stays
    in the generation rate on its grader reply alone."""
    complete = complete_rows(rows)
    pair_tagged = sum(1 for r in complete if breaker_tagged(r["grader"], r, dims))
    graded = [u for u in units if is_parsed(u["grader"])]
    gen_tagged = sum(1 for u in graded if breaker_tagged(u["grader"], rows_by_uid.get(u["uid"]), dims))
    return _rate(pair_tagged, len(complete)), _rate(gen_tagged, len(graded))


def _trips(rate: Mapping) -> bool:
    return rate["rate_exact"] is not None and Fraction(rate["tagged"], rate["denominator"]) > BREAKER_LIMIT


def breaker(units: Sequence[Mapping], rows: Sequence[Mapping], dims: Sequence[str] = gp.DIMENSIONS) -> dict[str, Any]:
    """The full-run circuit breaker: the pair rate over the complete pairs, the generation
    rate over the units whose grader reply parsed, all worker runs pooled. It trips when
    either defined rate is strictly above one quarter. Per-worker rates are descriptive and
    never trip."""
    dims = tuple(dims)
    _by_uid(units, "uid", "units")
    rows_by_uid = _by_uid(rows, "uid", "calibrator rows")
    pair_rate, generation_rate = _breaker_rates(units, rows, rows_by_uid, dims)
    per_worker: dict[str, dict[str, Any]] = {}
    for model in sorted({u["model"] for u in units} | {r["model"] for r in rows}):
        m_units = [u for u in units if u["model"] == model]
        m_rows = [r for r in rows if r["model"] == model]
        m_pair, m_generation = _breaker_rates(m_units, m_rows, rows_by_uid, dims)
        per_worker[model] = {"pair_rate": m_pair, "generation_rate": m_generation, "descriptive": "never a trip"}
    return {"unit": "full run", "pair_rate": pair_rate, "generation_rate": generation_rate, "limit": "1/4",
            "trip": _trips(pair_rate) or _trips(generation_rate),
            "per_worker": per_worker,
            "note": ("an undefined rate does not trip; the reliability gate excludes a sub-dimension "
                     "with no complete pair")}


def claims(gate: Mapping, breaker_result: Mapping, dims: Sequence[str] = gp.DIMENSIONS) -> dict[str, Any]:
    """Which LLM sub-dimensions a claim may use: the gate's exclusions add their reasons, and
    a breaker trip excludes both. When none is kept, the anchors stand alone."""
    out: dict[str, Any] = {}
    gate_reasons = {e["subdimension"]: e["reason"] for e in gate["excluded"]}
    for d in dims:
        reasons: list[str] = []
        if d in gate_reasons:
            reasons.append(gate_reasons[d])
        if breaker_result["trip"]:
            reasons.append(EXCLUSION_BREAKER)
        out[d] = {"kept": not reasons, "reasons": reasons}
    out["anchors_stand_alone"] = not any(out[d]["kept"] for d in dims)
    return out


# --- Queue (D-R23, queue) ------------------------------------------------------------------

def queue_tags(grader_reply: Mapping | None, row: Mapping | None, dims: Sequence[str]) -> list[str]:
    """The queue tags of one unit, in TAG_ORDER."""
    tags: list[str] = []
    if row is not None and disagrees(row["grader"], row["calibrator"], dims):
        tags.append(TAG_DISAGREEMENT)
    if any_low(grader_reply):
        tags.append(TAG_GRADER_LOW)
    if row is not None and any_low(row["calibrator"]):
        tags.append(TAG_CALIBRATOR_LOW)
    if not is_parsed(grader_reply):
        tags.append(TAG_GRADER_FAILED)
    if row is not None and not is_parsed(row["calibrator"]):
        tags.append(TAG_CALIBRATOR_FAILED)
    return tags


def _reply_view(reply: Mapping | None) -> dict[str, Any] | None:
    """The queue view of one reply. A reply that did not parse keeps its raw reply and its
    error; a parsed reply has None in both."""
    if reply is None:
        return None
    parsed = is_parsed(reply)
    return {"status": reply["status"], "scores": dict(reply["scores"]),
            "low_confidence": dict(reply["low_confidence"]), "reasons": dict(reply["reasons"]),
            "reply": None if parsed else reply.get("reply"), "error": None if parsed else reply.get("error")}


def allowed_dispositions(grader_reply: Mapping | None, calibrator_reply: Mapping | None) -> tuple[str, ...]:
    """The dispositions a queue item offers. Both judges parsed: the four of the
    pre-registration. The grader reply did not parse and a calibrator score exists:
    calibrator-correct, both-wrong and ambiguous. No calibrator score and a parsed grader
    reply: grader-correct, grader-wrong and ambiguous (D-R23, queue). No calibrator score
    and a grader reply that did not parse, was refused or is a failed judgment: grader-wrong
    and ambiguous only, as there is no grader score to be correct about."""
    if not is_parsed(calibrator_reply):
        return DISPOSITIONS_NO_CALIBRATOR if is_parsed(grader_reply) else DISPOSITIONS_GRADER_FAILED_NO_CALIBRATOR
    return DISPOSITIONS_WITH_CALIBRATOR if is_parsed(grader_reply) else DISPOSITIONS_GRADER_FAILED


def queue_items(units: Sequence[Mapping], rows: Sequence[Mapping], dims: Sequence[str] = gp.DIMENSIONS) -> list[dict]:
    """One queue item per unit with at least one tag, in unit-id order, with empty
    disposition fields and no model id."""
    dims = tuple(dims)
    rows_by_uid = _by_uid(rows, "uid", "calibrator rows")
    items: list[dict] = []
    for unit in sorted(_by_uid(units, "uid", "units").values(), key=lambda u: u["uid"]):
        row = rows_by_uid.get(unit["uid"])
        tags = queue_tags(unit["grader"], row, dims)
        if not tags:
            continue
        calibrator = None if row is None else row["calibrator"]
        items.append({
            "unit_id": unit["uid"], "tags": tags, "grader": _reply_view(unit["grader"]),
            "calibrator": _reply_view(calibrator),
            "allowed_dispositions": list(allowed_dispositions(unit["grader"], calibrator)),
            "disposition": None, "disposition_note": None, "disposed_by": None, "disposed_on": None,
        })
    return items


def queue_counts(items: Sequence[Mapping]) -> dict[str, Any]:
    """The number of queue items and the number of items carrying each tag."""
    return {"items": len(items), "by_tag": {t: sum(1 for i in items if t in i["tags"]) for t in TAG_ORDER}}


# --- Means (D-R23, means) ------------------------------------------------------------------

def subsets_of(case_kind: str | None) -> tuple[str, ...]:
    """The report subsets a case belongs to: all, then synthetic, or its kind and the
    pooled difficulty-bearing subset (as run_report.subsets_of)."""
    if case_kind == "standard":
        return ("all", "synthetic")
    if case_kind in ("edge", "adversarial"):
        return ("all", case_kind, "difficulty_bearing")
    return ("all",)


def mean_block(values_by_sample: Mapping[str, list[int]]) -> dict[str, Any]:
    """The mean of the per-sample epoch means, and their standard error with a normal 95
    percent interval (the trials.anchor_means method), clipped to the 0 to 3 scale
    (D-R23, means). A sample with no value is left out; the caller counts it."""
    values: list[float] = []
    per_sample: list[float] = []
    fewer = 0
    for vals in values_by_sample.values():
        vals_f = [float(v) for v in vals]
        if not vals_f:
            continue
        values.extend(vals_f)
        per_sample.append(sum(vals_f) / len(vals_f))
        if len(vals_f) < _FULL_EPOCHS:
            fewer += 1
    mean = sum(per_sample) / len(per_sample) if per_sample else None
    se = None
    if mean is not None and len(per_sample) >= 2:
        sd = math.sqrt(sum((v - mean) ** 2 for v in per_sample) / (len(per_sample) - 1))
        se = sd / math.sqrt(len(per_sample))
    low = high = None
    clipped = False
    if mean is not None and se is not None:
        raw_low, raw_high = mean - Z95 * se, mean + Z95 * se
        low, high = max(SCORE_MIN, raw_low), min(SCORE_MAX, raw_high)
        clipped = low != raw_low or high != raw_high
    return {"mean": mean, "se": se, "ci95_low": low, "ci95_high": high, "ci95_clipped": clipped,
            "ci_note": CI_NOTE, "values": len(values), "samples": len(per_sample),
            "samples_fewer_than_4_epochs": fewer}


def _reading_value(gen: Mapping, grader_replies: Mapping[str, Mapping], dim: str, reading: str) -> int | None:
    """One generation's value under a reading, None when missing."""
    status = gen["status"]
    if status == _GEN_GRADEABLE:
        reply = grader_replies.get(gen["uid"]) if gen.get("uid") is not None else None
        return reply["scores"][dim] if is_parsed(reply) else None
    if status in (_GEN_ERRORED, _GEN_UNPARSED):
        return 0 if reading == "sensitivity" else None
    raise jt.Refused(f"a generation has an unknown status {status!r}")


def _ungraded(gen: Mapping, grader_replies: Mapping[str, Mapping]) -> bool:
    if gen["status"] != _GEN_GRADEABLE:
        return False
    reply = grader_replies.get(gen["uid"]) if gen.get("uid") is not None else None
    return not is_parsed(reply)


def _per_sample(gens: Sequence[Mapping], grader_replies: Mapping[str, Mapping], dim: str,
                reading: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for g in sorted(gens, key=lambda g: (g["sample_id"], g["epoch"])):
        vals = out.setdefault(g["sample_id"], [])
        value = _reading_value(g, grader_replies, dim, reading)
        if value is not None:
            vals.append(value)
    return out


def _block(gens: Sequence[Mapping], grader_replies: Mapping[str, Mapping], dim: str, reading: str,
           label: str) -> dict[str, Any]:
    by_sample = _per_sample(gens, grader_replies, dim, reading)
    return {**mean_block(by_sample), "label": label,
            "ungraded_generations": sum(1 for g in gens if _ungraded(g, grader_replies)),
            "samples_with_no_value": sum(1 for vals in by_sample.values() if not vals)}


def _model_subset(generations: Sequence[Mapping], model: str, subset: str) -> list[Mapping]:
    return [g for g in generations if g["model"] == model and subset in subsets_of(g["case_kind"])]


def llm_means(generations: Sequence[Mapping], grader_replies: Mapping[str, Mapping],
              dims: Sequence[str] = gp.DIMENSIONS) -> dict[str, Any]:
    """Per model, subset and sub-dimension: the primary and sensitivity means, and both again
    with every defect epoch left out (the D-R20 row)."""
    out: dict[str, Any] = {}
    for model in sorted({g["model"] for g in generations}):
        per_subset: dict[str, Any] = {}
        for subset in SUBSETS:
            gens = _model_subset(generations, model, subset)
            kept = [g for g in gens if not g["defect_epoch"]]
            per_subset[subset] = {d: {
                "primary": _block(gens, grader_replies, d, "primary", LABEL_PRIMARY),
                "sensitivity": _block(gens, grader_replies, d, "sensitivity", LABEL_SENSITIVITY),
                "primary_defect_dropped": _block(kept, grader_replies, d, "primary", LABEL_PRIMARY),
                "sensitivity_defect_dropped": _block(kept, grader_replies, d, "sensitivity", LABEL_SENSITIVITY),
            } for d in dims}
        out[model] = per_subset
    return out


def paired_differences(generations: Sequence[Mapping], grader_replies: Mapping[str, Mapping],
                       dims: Sequence[str] = gp.DIMENSIONS) -> list[dict]:
    """Cross-model paired differences of per-sample epoch means, on the samples with a value
    for both models, for the primary and the sensitivity readings, labelled by reading.
    first is the later model id in sorted order, second the earlier; the difference is
    first minus second."""
    out: list[dict] = []
    models = sorted({g["model"] for g in generations})
    for a, b in itertools.combinations(models, 2):
        for subset in SUBSETS:
            gens_a = _model_subset(generations, a, subset)
            gens_b = _model_subset(generations, b, subset)
            for d in dims:
                for reading in _READINGS:
                    sa = _per_sample(gens_a, grader_replies, d, reading)
                    sb = _per_sample(gens_b, grader_replies, d, reading)
                    both = sorted(s for s in sa if sa[s] and sb.get(s))
                    pairs = [(Fraction(sum(sb[s]), len(sb[s])), Fraction(sum(sa[s]), len(sa[s]))) for s in both]
                    out.append({"first": b, "second": a, "subset": subset, "dim": d, "reading": reading,
                                "samples": len(pairs),
                                "result": jt.paired_mean_difference(pairs) if pairs else None,
                                "label": LABEL_PAIRED[reading]})
    return out


# --- One call for the sealed aggregates -----------------------------------------------------

def aggregates(units: Sequence[Mapping], generations: Sequence[Mapping], rows: Sequence[Mapping],
               grader_replies: Mapping[str, Mapping], *, bootstrap_seed: int,
               resamples: int = BOOTSTRAP_RESAMPLES, dims: Sequence[str] = gp.DIMENSIONS) -> dict[str, Any]:
    """Every sealed aggregate of the run: the reliability gate with its worst-case figure and
    break-even counts, the bootstrap limit, the breaker, the claims, the means, the paired
    differences and the queue counts."""
    dims = tuple(dims)
    gate = reliability_gate(rows, dims)
    breaker_result = breaker(units, rows, dims)
    return {
        "gate": gate,
        "bootstrap": case_cluster_bootstrap(rows, seed=bootstrap_seed, resamples=resamples, dims=dims),
        "breaker": breaker_result,
        "claims": claims(gate, breaker_result, dims),
        "means": llm_means(generations, grader_replies, dims),
        "paired_differences": paired_differences(generations, grader_replies, dims),
        "queue_counts": queue_counts(queue_items(units, rows, dims)),
        "lost_rows_are_not_dropped_pairs": True,
    }
