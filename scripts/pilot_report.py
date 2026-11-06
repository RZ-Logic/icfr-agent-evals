"""The section 13 pilot row's quantities from saved logs. NO API key, no model call.

Usage:
  python scripts/pilot_report.py --sonnet S.eval --haiku H.eval --template T.eval
         [--reading-b i|ii|iii|all] [--regression-baseline B.eval --regression-candidate C.eval]
         [--json out.json] [--review-csv out.csv] [--max-tokens-pin N]

Reads the three pilot legs (pre-registration section 13, pilot row; 20 dev-split
cases, epochs = 4) and prints, per run: sample count and epochs; content-pass^4
per sample and overall; FM counts by code with FM-002 by direction; review-queue
volume as distinct (sample, epoch, reason) and distinct (sample, epoch), split by
class (never the raw scorer-row count, six rows per format break); the Sample
Consistency summary with the H4 line labeled as a rehearsal (never on the template
log: a deterministic run agrees with itself by construction, so every wrong case
would read as FM-007). Then the template margin per anchor, difficulty check (a)
from the template log, difficulty check (b) from the Haiku log under all three
readings, the McNemar block floor and the paired-t minimal detectable drop
(inferred), and, given both regression logs, compare_runs.py's verdict on the pair.

Every worker log (Sonnet, Haiku and both regression logs) must pass the run
configuration check (contract C2): roster.check_run_config with 20 samples, epochs 4
and batch allowed, max_tokens only at --max-tokens-pin. The template baseline log is
out of that check's scope (no evaluated model, so no ModelEvent and no
reasoning_effort). The Sonnet and Haiku logs must also pass
compare_runs.check_matching_runs (same task, worker prompt, split, seed, subset and
dataset content). Queue volume, the review CSV rows, the McNemar floor and the
paired-t drop come from reliability.queue_volume / queue_items and
regression.mcnemar_block_floor / paired_t_mdd.

Difficulty check (b) readings (owner decision D1: all three print; the reading of
record is chosen before the Haiku leg and passed as --reading-b):
  (i)   any trial anchor below 3, any FM code, or a missing or errored epoch;
  (ii)  as (i), excluding FM-000, errored or missing epochs and narrative-sweep
        provisional zeros (an anchor whose only review reasons are sweep hits);
  (iii) only a settled anchor below 3 or a non-format FM code; every anchor
        routed to review is excluded.
(b) passes when at least one hard case fails on at least one epoch.

Caught (owner decision D2): compare_runs.py exits 1 on the (pilot baseline,
regressed candidate) pair, with the rejecting tests named. Exit 2 is sign-off
volume, not caught; exit 0 is not caught. Only an exit 1 prints the word "caught";
exit 0 and exit 2 print "MISSED (no automatic block)".

The review CSV holds one row per distinct (run, sample, epoch, route key), with
verdict, reviewer and date blank for the owner (verdict: correct_route,
false_fail or confirmed_failure).

Exit codes:
  0  report printed
  3  refused: a missing or unreadable file, an evaluation-split log, a log that is
     not the 20-case dev pilot at epochs = 4 (or a partial run), a log whose status
     is not success, a worker log off the section 5 run configuration (contract C2),
     runs whose Task version or dataset content differ, Sonnet and Haiku logs on
     another task or worker prompt, a template
     log that is not the template baseline (or a worker log that is), a regression
     pair compare_runs.py refuses, a help request, an output path that cannot be
     written, or any other input the report cannot read
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import statistics
from functools import lru_cache
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from icfr_agent_evals import consistency as con
from icfr_agent_evals import reliability as rel
from icfr_agent_evals import roster
from icfr_agent_evals.regression import (
    CODE_ROUTE,
    DIMENSIONS,
    FAMILY_ALPHA,
    HOLM_M,
    SWEEP_REASON_PREFIXES,
    SWEEP_ROUTE,
    content_fingerprint,
    mcnemar_block_floor,
    paired_t_mdd,
    records_from_samples,
    refuse_evaluation_split,
    refuse_partial_run,
    t_ppf,
)
from icfr_agent_evals.reliability import GateError

EXIT_OK, EXIT_REFUSED = 0, 3
PILOT_SPLIT, PILOT_SUBSET, PILOT_SAMPLES, PILOT_K = "dev", "pilot", 20, rel.PREREGISTERED_K
READINGS: tuple[str, ...] = ("i", "ii", "iii")
TEMPLATE_TASK = "variance_commentary_template_baseline"
TEMPLATE_MODEL = "none/none"
HAIKU = next(m for m in roster.BASELINES if "haiku" in m)
ROLE_MODELS = {"sonnet": roster.WORKER, "haiku": HAIKU, "template": TEMPLATE_MODEL}
ANCHORS: tuple[str, ...] = (*rel.TRIAL_ANCHORS, "faithfulness_anchor")
MDD_POWER = 0.80  # regression.paired_t_mdd's default power
MDD_REFERENCE_SDS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25)

# Review-queue classes (reliability.QUEUE_CLASSES): a format break and a sample error
# are not section 10(c) routes, and fm_listed is a reason that rides with its FM code
# (the fabrication listing beside FM-005 / FM-015), printed on its own line.
FORMAT_BREAK = rel.QUEUE_FORMAT_BREAK
SAMPLE_ERROR = rel.QUEUE_SAMPLE_ERROR
FM_LISTED = rel.QUEUE_FM_LISTED
QUEUE_CLASSES: tuple[str, ...] = rel.QUEUE_CLASSES
QUEUE_CLASS_NOTES = {FM_LISTED: "  (rides with FM-005 / FM-015; not a section 10(c) route)"}
REVIEW_CSV_COLUMNS: tuple[str, ...] = (
    "run_label", "log_eval_id", "sample_id", "epoch", "scorers", "route_key", "reason", "verdict", "reviewer", "date",
)
VERDICTS: tuple[str, ...] = ("correct_route", "false_fail", "confirmed_failure")
TEMPLATE_CONSISTENCY_LINE = "template: deterministic, agreement 1 by construction; Sample Consistency not reported"
COMPARE_SCRIPT = Path(__file__).resolve().parent / "compare_runs.py"


class Refused(Exception):
    """An input the report must not read; exit 3."""


# --- Inputs ------------------------------------------------------------------------------

def run_config_check(log: Any, role: str, max_tokens_pin: int | None = None) -> dict | None:
    """Contract C2: roster.check_run_config on a full worker log (20 samples, epochs 4,
    batch allowed), its RunConfigError refused. The template role is out of scope and
    returns None unchecked (no evaluated model: no ModelEvent, no reasoning_effort)."""
    if role == "template":
        return None
    try:
        return roster.check_run_config(log, expected_samples=PILOT_SAMPLES, expected_epochs=PILOT_K,
                                       allow_batch=True, max_tokens_pin=max_tokens_pin)
    except roster.RunConfigError as e:
        raise Refused(f"{role} log: {e}") from e


def check_pilot_log(log: Any, role: str, max_tokens_pin: int | None = None) -> None:
    """Refuses a log that is not a successful whole 20-case dev pilot run at epochs = 4,
    and a template log that is not the template baseline (or a worker log that is)."""
    if log.status != "success":
        raise Refused(f"{role} log status is {log.status!r}, not 'success'")
    if log.samples is None:
        raise Refused(f"{role} log has no samples; read it with header_only=False and keep sample logging on")
    try:
        refuse_evaluation_split(log)
        refuse_partial_run(log)
    except GateError as e:
        raise Refused(f"{role} log: {e}") from e
    args = log.eval.task_args or {}
    meta = getattr(log.eval, "metadata", None) or {}
    if args.get("split") != PILOT_SPLIT or meta.get("split") != PILOT_SPLIT:
        raise Refused(f"{role} log is not the dev pilot: split {args.get('split')!r}")
    if args.get("subset") != PILOT_SUBSET or meta.get("subset") != PILOT_SUBSET:
        raise Refused(f"{role} log is not the dev pilot: subset {args.get('subset')!r}, not {PILOT_SUBSET!r}")
    samples = getattr(log.eval.dataset, "samples", None)
    if samples != PILOT_SAMPLES:
        raise Refused(f"{role} log is not the dev pilot: {samples} samples, not {PILOT_SAMPLES}")
    ran = int(log.eval.config.epochs or 1)
    declared = args.get("epochs")
    if ran != PILOT_K or (declared is not None and int(declared) != ran):
        raise Refused(f"{role} log ran epochs {ran} (task argument {declared}), not {PILOT_K}")
    is_template = str(log.eval.task).endswith(TEMPLATE_TASK)
    if (role == "template") != is_template:
        raise Refused(f"{role} log runs task {log.eval.task!r}")
    run_config_check(log, role, max_tokens_pin)


def load_pilot_log(path: str, role: str, max_tokens_pin: int | None = None) -> Any:
    if not Path(path).is_file():
        raise Refused(f"{role} log not found: {path}")
    from inspect_ai.log import read_eval_log

    try:
        log = read_eval_log(str(path))
    except Exception as e:
        raise Refused(f"{role} log unreadable: {type(e).__name__}: {e}") from e
    check_pilot_log(log, role, max_tokens_pin)
    return log


def check_same_content(logs: dict[str, Any]) -> dict[str, Any]:
    """Every run must carry the same Task version and dataset content (hashes)."""
    prints: dict[str, dict] = {}
    for role, log in logs.items():
        try:
            prints[role] = content_fingerprint(log)
        except GateError as e:
            raise Refused(f"{role} log: {e}") from e
    first_role, first = next(iter(prints.items()))
    for role, fp in prints.items():
        diff = {k: (first[k], fp[k]) for k in first if first[k] != fp[k]}
        if diff:
            raise Refused(f"{role} log differs from the {first_role} log in Task version or dataset content: {diff}")
    return first


# --- Small statistics --------------------------------------------------------------------

def _bisect(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    """x in [lo, hi] with f(x) = target, f increasing."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def _binom_cdf(x: int, n: int, p: float) -> float:
    return math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion x / n."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if x == 0 else _bisect(lambda p: 1 - _binom_cdf(x - 1, n, p), 0.0, 1.0, alpha / 2)
    hi = 1.0 if x == n else _bisect(lambda p: -_binom_cdf(x, n, p), 0.0, 1.0, -alpha / 2)
    return lo, hi


def noise_sd(log: Any) -> dict[str, float | None]:
    """Per wired dimension, the inferred sd of a no-change paired difference of
    per-sample epoch means: sqrt(2 * mean within-sample epoch variance / k)."""
    by_sample: dict[str, list] = defaultdict(list)
    for r in records_from_samples(log.samples):
        by_sample[r.sample_id].append(r)
    out: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        variances = []
        for recs in by_sample.values():
            vals = [r.dims[dim] for r in recs if dim in r.dims]
            if len(vals) >= 2:
                variances.append(statistics.variance(vals))
        if not variances:
            out[dim] = None
            continue
        k = max(len(recs) for recs in by_sample.values())
        out[dim] = math.sqrt(2 * statistics.fmean(variances) / k)
    return out


# --- Per epoch and per run ---------------------------------------------------------------

def is_hard(sample: Any) -> bool:
    return sample.metadata["truth"]["case_kind"] != "standard"


def subset_of(sample: Any) -> str:
    return con.SUBSET_OF_KIND[sample.metadata["truth"]["case_kind"]]


def _is_sweep(reason: str) -> bool:
    return str(reason).startswith(SWEEP_REASON_PREFIXES)


def epoch_fail(sample: Any, reading: str) -> bool:
    """Difficulty check (b): does this (sample, epoch) count as a hard-case failure
    under reading (i), (ii) or (iii)? See the module docstring."""
    if reading not in READINGS:
        raise ValueError(f"reading must be one of {READINGS}, got {reading!r}")
    if getattr(sample, "error", None) is not None:
        return reading == "i"
    scores = getattr(sample, "scores", None) or {}
    codes = {str(c) for sc in scores.values() for c in (getattr(sc, "metadata", None) or {}).get("failure_modes") or []}
    if reading == "i":
        if codes:
            return True
        return any(name not in scores or rel._numeric(scores[name].value) != rel.FULL_MARKS for name in rel.TRIAL_ANCHORS)
    if codes - {"FM-000"}:
        return True
    for name in rel.TRIAL_ANCHORS:
        sc = scores.get(name)
        if sc is None or rel._numeric(sc.value) == rel.FULL_MARKS:
            continue
        meta = getattr(sc, "metadata", None) or {}
        if meta.get("format_break"):
            continue
        routed = bool(meta.get("requires_human_review"))
        reasons = [str(r) for r in meta.get("review_reasons") or []]
        if reading == "ii" and routed and reasons and all(_is_sweep(r) for r in reasons):
            continue
        if reading == "iii" and routed:
            continue
        return True
    return False


def difficulty_b(log: Any) -> dict[str, Any]:
    """Failing hard cases and (case, epoch) pairs under each reading. A missing epoch
    counts under (i) only."""
    hard = [s for s in log.samples if is_hard(s)]
    seen: dict[str, set[int]] = defaultdict(set)
    for s in hard:
        seen[str(s.id)].add(int(s.epoch))
    missing = sorted((sid, e) for sid, eps in seen.items() for e in range(1, PILOT_K + 1) if e not in eps)
    out: dict[str, Any] = {
        "hard_cases": len(seen),
        "hard_epochs": len(hard),
        "errored_epochs": sum(1 for s in hard if getattr(s, "error", None) is not None),
        "missing_epochs": [list(x) for x in missing],
        "readings": {},
    }
    for reading in READINGS:
        fails = sorted({(str(s.id), int(s.epoch)) for s in hard if epoch_fail(s, reading)})
        if reading == "i":
            fails = sorted(set(fails) | set(missing))
        cases = sorted({sid for sid, _ in fails})
        out["readings"][reading] = {
            "verdict": "PASS" if cases else "FAIL",
            "failing_cases": cases,
            "failing_epochs": [list(x) for x in fails],
        }
    return out


def _codes(sample: Any) -> set[str]:
    return {str(c) for v in (sample.scores or {}).values() for c in (v.metadata or {}).get("failure_modes") or []}


def difficulty_a(log: Any) -> dict[str, Any]:
    """Every memo-only standard case fails on every epoch through the reconciling-items
    score only (that anchor below 3, the other trial anchors 3, no FM code); every other
    standard case is clean on every epoch."""
    memo: dict[str, list[bool]] = defaultdict(list)
    ctrl: dict[str, list[bool]] = defaultdict(list)
    hard_below: set[str] = set()
    for s in log.samples:
        truth = s.metadata["truth"]
        sc = s.scores or {}
        vals = {n: rel._numeric(sc[n].value) if n in sc else None for n in rel.TRIAL_ANCHORS}
        codes = _codes(s)
        review = any((v.metadata or {}).get("requires_human_review") for v in sc.values())
        errored = getattr(s, "error", None) is not None
        if truth["case_kind"] != "standard":
            if errored or any(v != rel.FULL_MARKS for v in vals.values()):
                hard_below.add(str(s.id))
            continue
        if truth["has_memo_only_link"]:
            ok = (not errored and not codes and vals["reconciling_items_anchor"] is not None
                  and vals["reconciling_items_anchor"] < rel.FULL_MARKS
                  and all(vals[n] == rel.FULL_MARKS for n in rel.TRIAL_ANCHORS if n != "reconciling_items_anchor"))
            memo[str(s.id)].append(ok)
        else:
            ctrl[str(s.id)].append(not errored and not codes and not review
                                   and all(v == rel.FULL_MARKS for v in vals.values()))
    memo_ok = sorted(sid for sid, v in memo.items() if all(v))
    ctrl_ok = sorted(sid for sid, v in ctrl.items() if all(v))
    passed = bool(memo) and len(memo_ok) == len(memo) and len(ctrl_ok) == len(ctrl)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "memo_only_cases": len(memo),
        "memo_only_failing_through_omission_only": len(memo_ok),
        "memo_only_not_so": sorted(set(memo) - set(memo_ok)),
        "other_standard_cases": len(ctrl),
        "other_standard_clean": len(ctrl_ok),
        "other_standard_not_clean": sorted(set(ctrl) - set(ctrl_ok)),
        "hard_cases_template_below_3": sorted(hard_below),
    }


def csv_route_key(item: rel.QueueItem) -> str:
    """The CSV route key of one queue item: its section 10(c) route key, else its class
    (format_break, sample_error), or "fm_listed: <reason>" for a reason riding with an FM code."""
    if item.route_key is not None:
        return item.route_key
    if item.queue_class == FM_LISTED:
        return f"{FM_LISTED}: {item.reason}"
    return item.queue_class


def review_rows(report: rel.ReliabilityReport, run_label: str, eval_id: str) -> list[dict[str, str]]:
    """One CSV row per distinct (run, sample, epoch, route key) from reliability.queue_items;
    verdict, reviewer and date blank."""
    grouped: dict[tuple[str, int, str], dict[str, list[str]]] = {}
    for item in rel.queue_items(report):
        g = grouped.setdefault((item.sample_id, item.epoch, csv_route_key(item)), {"scorers": [], "reasons": []})
        for scorer in item.scorers:
            if scorer not in g["scorers"]:
                g["scorers"].append(scorer)
        if item.reason not in g["reasons"]:
            g["reasons"].append(item.reason)
    return [
        {"run_label": run_label, "log_eval_id": eval_id, "sample_id": sid, "epoch": str(ep),
         "scorers": ";".join(sorted(g["scorers"])), "route_key": key, "reason": " | ".join(g["reasons"]),
         "verdict": "", "reviewer": "", "date": ""}
        for (sid, ep, key), g in sorted(grouped.items())
    ]


def queue_payload(volume: rel.QueueVolume) -> dict[str, dict[str, int]]:
    """Distinct (sample, epoch, reason) and distinct (sample, epoch), overall and per class
    (the scorer-row count is never reported)."""
    out = {"all": {"sample_epoch_reason": volume.total_items, "sample_epoch": volume.total_sample_epochs}}
    for cls in QUEUE_CLASSES:
        out[cls] = {"sample_epoch_reason": volume.items[cls], "sample_epoch": volume.sample_epochs[cls]}
    return out


def anchor_means(log: Any) -> dict[str, dict[str, float]]:
    """anchor -> sample id -> mean over epochs; an errored or missing score counts 0."""
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in log.samples:
        errored = getattr(s, "error", None) is not None
        for n in ANCHORS:
            v = None if errored or n not in (s.scores or {}) else rel._numeric(s.scores[n].value)
            per[n][str(s.id)].append(0.0 if v is None else v)
    return {n: {sid: statistics.fmean(v) for sid, v in d.items()} for n, d in per.items()}


def template_margin(worker: Any, template: Any) -> dict[str, dict[str, dict[str, float]]]:
    """subset -> anchor -> worker mean, template mean, paired margin and its 95% t CI."""
    w, t = anchor_means(worker), anchor_means(template)
    subsets = {str(s.id): subset_of(s) for s in worker.samples}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("overall", *con.SUBSETS):
        ids = sorted(sid for sid, sub in subsets.items() if name == "overall" or sub == name)
        out[name] = {}
        for anchor in ANCHORS:
            d = [w[anchor][sid] - t[anchor][sid] for sid in ids]
            m = statistics.fmean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            half = t_ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
            out[name][anchor] = {
                "n": len(d), "worker": statistics.fmean(w[anchor][sid] for sid in ids),
                "template": statistics.fmean(t[anchor][sid] for sid in ids),
                "margin": m, "ci_low": m - half, "ci_high": m + half,
            }
    return out


def consistency_summary(log: Any) -> list[str]:
    """Summary lines of consistency.format_report; the H4 line must read as a rehearsal."""
    report = con.report_from_log(log, run_config=False)  # checked in load_pilot_log (run_config_check)
    con.assert_gate(report, PILOT_SAMPLES, PILOT_K)
    lines = []
    for line in con.format_report(report).splitlines():
        if line.startswith("H4,"):
            if "not the H4 test" not in line:
                raise Refused("a pilot log's H4 line does not read 'not the H4 test'")
            lines.append("H4 line (rehearsal; a pilot log is never the H4 test): " + line)
        elif line.startswith(("[", con.FM007)):
            lines.append(line)
    return lines


def run_block(role: str, log: Any, path: str = "") -> tuple[list[str], dict[str, Any], list[dict[str, str]]]:
    """Printed lines, JSON payload and review rows for one run."""
    e = log.eval
    cr = load_compare_runs()
    expected = ROLE_MODELS[role]
    model_note = "roster model" if e.model == expected else f"STAND-IN, not the roster model {expected}"
    lines = [f"== Run {role}: {e.model} ({model_note})",
             cr.format_header(role, path, cr.run_header(log)) + f" | eval_id {e.eval_id}"]
    r = rel.report_from_log(log, run_config=False)  # checked in load_pilot_log (run_config_check)
    try:
        rel.assert_gate(r, PILOT_SAMPLES, PILOT_K)
        gate = f"assert_gate OK ({PILOT_SAMPLES} samples, k = {PILOT_K})"
    except GateError as err:
        gate = f"assert_gate FAILED: {err}"
    lines.append(f"samples {r.scored_samples}, epochs {int(e.config.epochs)}; {gate}")
    passed = sum(r.pass_k.values())
    lo, hi = clopper_pearson(passed, r.scored_samples)
    lines.append(f"content-pass^4 {passed}/{r.scored_samples} = {r.content_pass_k_rate:.3f} "
                 f"(Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}]); FM-008 {len(r.fm008)}")
    lines.append("  per sample: " + " ".join(f"{sid}:{v}" for sid, v in r.pass_k.items()))
    lines.append(rel.format_fm_counts(r))
    errored = sum(1 for s in log.samples if getattr(s, "error", None) is not None)
    fb = sum(1 for s in log.samples if getattr(s, "error", None) is None
             and any((v.metadata or {}).get("format_break") for v in (s.scores or {}).values()))
    lines.append(f"format-break epochs {fb}/{len(log.samples)}; errored epochs {errored}/{len(log.samples)}")
    volume = rel.queue_volume(r)
    vol = queue_payload(volume)
    lines.append(f"Review queue volume: {volume.total_items} distinct (sample, epoch, reason), "
                 f"{volume.total_sample_epochs} distinct (sample, epoch)")
    for cls in QUEUE_CLASSES:
        lines.append(f"  {cls:<16} {volume.items[cls]} (sample, epoch, reason), "
                     f"{volume.sample_epochs[cls]} (sample, epoch){QUEUE_CLASS_NOTES.get(cls, '')}")
    if role == "template":
        lines.append(TEMPLATE_CONSISTENCY_LINE)
        cons: Any = TEMPLATE_CONSISTENCY_LINE
    else:
        try:
            cons = consistency_summary(log)
        except GateError as err:
            cons = [f"Sample Consistency not computed: {err}"]
        lines.append("Sample Consistency (summary):")
        lines.extend("  " + x for x in cons)
    payload = {
        "path": None, "model": e.model, "stand_in": e.model != expected, "task": e.task, "eval_id": e.eval_id,
        "samples": r.scored_samples, "epochs": int(e.config.epochs), "gate": gate,
        "content_pass4": {"passed": passed, "n": r.scored_samples, "rate": r.content_pass_k_rate,
                          "ci95": [lo, hi], "per_sample": r.pass_k},
        "fm_counts": r.fm_counts, "fm002_by_direction": r.fm002_by_direction, "fm008": list(r.fm008),
        "format_break_epochs": fb, "errored_epochs": errored, "queue_volume": vol, "sample_consistency": cons,
    }
    return lines, payload, review_rows(r, role, str(e.eval_id))


def floors_block(sonnet: Any) -> tuple[list[str], dict[str, Any]]:
    """McNemar block floor under Holm (m = 6) and the inferred paired-t drop at n = 20."""
    p0, p1 = mcnemar_block_floor(), mcnemar_block_floor(reversals=1)
    passed = sum(rel.report_from_log(sonnet, run_config=False).pass_k.values())  # checked in load_pilot_log
    lines = ["== Regression power floors (section 10(a), Holm m = 6, rank 1 threshold "
             f"{FAMILY_ALPHA / HOLM_M:.5f})",
             f"McNemar block floor on content-pass^4: {p0} one-way discordant cases with no reversal, "
             f"{p1} with one reversal",
             f"Sonnet log passes content-pass^4 on {passed} of {PILOT_SAMPLES}; worse cannot exceed it: "
             + (f"content-pass^4 can block (at least {p0})" if passed >= p0
                else f"below {p0}, only the paired t-tests can catch a regression")]
    sds = noise_sd(sonnet)
    lines.append(f"Paired-t minimal detectable drop, n = {PILOT_SAMPLES}, rank 1, {MDD_POWER:.0%} power "
                 "(inferred: no-change noise sd = sqrt(2 * mean within-sample epoch variance / k) from the Sonnet log)")
    mdd: dict[str, Any] = {}
    for dim, sd in sds.items():
        if sd is None:
            lines.append(f"  {dim:<13} not scored")
            mdd[dim] = None
            continue
        drop = paired_t_mdd(sd)
        note = "  (sd 0: a deterministic run; any constant drop gives p = 0, so this floor says nothing)" if sd == 0 else ""
        lines.append(f"  {dim:<13} sd {sd:.3f} drop {drop:.3f}{note}")
        mdd[dim] = {"sd": sd, "drop": drop}
    ref = {sd: paired_t_mdd(sd) for sd in MDD_REFERENCE_SDS}
    lines.append("  reference (inferred, real sd unknown): " + ", ".join(f"sd {sd}: {d:.3f}" for sd, d in ref.items()))
    return lines, {"mcnemar_floor": {"no_reversal": p0, "one_reversal": p1}, "sonnet_pass4": passed,
                   "paired_t_mdd": mdd, "paired_t_mdd_reference": {str(k): v for k, v in ref.items()},
                   "label": "inferred"}


@lru_cache(maxsize=1)
def load_compare_runs() -> Any:
    spec = importlib.util.spec_from_file_location("compare_runs", COMPARE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regression_block(baseline: str, candidate: str,
                     max_tokens_pin: int | None = None) -> tuple[list[str], dict[str, Any]]:
    """compare_runs.py on the pair; caught only on its exit 1 (an automatic block)."""
    cli = load_compare_runs()
    with tempfile.TemporaryDirectory() as tmp:
        json_out = os.path.join(tmp, "compare.json")
        argv = ["--baseline", baseline, "--candidate", candidate, "--json", json_out]
        if max_tokens_pin is not None:
            argv += ["--max-tokens-pin", str(max_tokens_pin)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        if code == cli.EXIT_REFUSED:
            raise Refused(f"compare_runs.py refused the regression pair: {err.getvalue().strip()}")
        with open(json_out, encoding="utf-8") as fh:
            result = json.load(fh)
    rejecting = [n for n, t in result["paired"].items() if t["rejected"]]
    rejecting += [n for n, t in result["fabrication"].items() if t["rejected"]]
    base_pass = int(result["paired"]["content_pass4"]["detail"]["baseline_pass"])
    if code == cli.EXIT_BLOCK:
        verdict = f"caught: compare_runs.py exit 1, automatic block by {', '.join(rejecting)}"
    elif code == cli.EXIT_SIGNOFF:
        verdict = (f"MISSED (no automatic block): compare_runs.py exit 2, {len(result['signoff'])} sign-off case(s), "
                   "recorded as sign-off volume")
    else:
        verdict = "MISSED (no automatic block): compare_runs.py exit 0, no block and no sign-off case"
    lines = ["== Deliberate regression (compare_runs.py on the pair)"]
    lines.extend("  " + x for x in out.getvalue().rstrip().splitlines())
    lines.append(f"baseline passes content-pass^4 on {base_pass} of {PILOT_SAMPLES} against the McNemar floor "
                 f"of {mcnemar_block_floor()}")
    lines.append("Regression verdict: " + verdict)
    return lines, {"exit_code": code, "rejecting_tests": rejecting, "caught": code == cli.EXIT_BLOCK,
                   "baseline_pass4": base_pass, "signoff_cases": len(result["signoff"]), "verdict": verdict}


def _difficulty_b_lines(b: dict[str, Any], record: str) -> list[str]:
    lines = [f"== Difficulty check (b) from the Haiku log: {b['hard_cases']} hard cases, {b['hard_epochs']} hard epochs, "
             f"errored {b['errored_epochs']}, missing {len(b['missing_epochs'])}"]
    for reading in READINGS:
        r = b["readings"][reading]
        mark = "  (reading of record)" if record == reading else ""
        lines.append(f"  reading ({reading}): {r['verdict']}{mark}  failing hard cases {len(r['failing_cases'])}, "
                     f"failing hard epochs {len(r['failing_epochs'])}")
        if r["failing_epochs"]:
            per: dict[str, list[int]] = defaultdict(list)
            for sid, ep in r["failing_epochs"]:
                per[sid].append(ep)
            lines.append("    " + "; ".join(f"{sid} e{','.join(map(str, eps))}" for sid, eps in per.items()))
    if record == "all":
        lines.append("  reading of record: not chosen (the owner picks it before the Haiku leg; pass --reading-b)")
    else:
        lines.append(f"Difficulty check (b) verdict under reading ({record}): {b['readings'][record]['verdict']}")
    return lines


def _margin_lines(label: str, margin: dict[str, Any]) -> list[str]:
    lines = [f"== Template margin per anchor ({label} minus template; paired per-sample means, 95% t CI)"]
    for name, anchors in margin.items():
        for anchor, m in anchors.items():
            lines.append(f"  {name:<18} {anchor:<26} n={m['n']:<3} worker {m['worker']:.3f} template {m['template']:.3f} "
                         f"margin {m['margin']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]")
    return lines


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    if (args.regression_baseline is None) != (args.regression_candidate is None):
        raise Refused("--regression-baseline and --regression-candidate go together")
    pin = args.max_tokens_pin
    paths = {"sonnet": args.sonnet, "haiku": args.haiku, "template": args.template}
    logs = {role: load_pilot_log(p, role, pin) for role, p in paths.items()}
    content = {"sonnet": logs["sonnet"], "haiku": logs["haiku"], "template": logs["template"]}
    if args.regression_baseline is not None:
        content["regression-baseline"] = load_pilot_log(args.regression_baseline, "regression-baseline", pin)
        content["regression-candidate"] = load_pilot_log(args.regression_candidate, "regression-candidate", pin)
    fingerprint = check_same_content(content)
    try:  # the two worker legs run one task and worker prompt (the compare_runs F2-08 rule)
        load_compare_runs().check_matching_runs(logs["sonnet"], [logs["haiku"]], [f"haiku log {args.haiku}"])
    except GateError as e:
        raise Refused(f"sonnet and haiku logs do not pair: {e}") from e

    lines = [f"PILOT REPORT (section 13 pilot row), dev split, subset {PILOT_SUBSET}, "
             f"{PILOT_SAMPLES} cases, epochs {PILOT_K}; dataset_sha256 {fingerprint['dataset_sha256']}"]
    payload: dict[str, Any] = {"fingerprint": fingerprint, "runs": {}, "reading_of_record": args.reading_b}
    rows: list[dict[str, str]] = []
    for role, log in logs.items():
        block, data, run_rows = run_block(role, log, paths[role])
        data["path"] = paths[role]
        lines.extend(block)
        payload["runs"][role] = data
        rows.extend(run_rows)

    payload["template_margin"] = {}
    for role in ("sonnet", "haiku"):
        margin = template_margin(logs[role], logs["template"])
        lines.extend(_margin_lines(role, margin))
        payload["template_margin"][role] = margin

    a = difficulty_a(logs["template"])
    lines.append(f"== Difficulty check (a) from the template log: {a['verdict']}  memo-only cases failing through the "
                 f"reconciling-items score only {a['memo_only_failing_through_omission_only']}/{a['memo_only_cases']}; "
                 f"other standard cases clean {a['other_standard_clean']}/{a['other_standard_cases']}")
    if a["memo_only_not_so"] or a["other_standard_not_clean"]:
        lines.append(f"  not as expected: memo-only {a['memo_only_not_so']}; other standard {a['other_standard_not_clean']}")
    lines.append(f"  hard cases where the template scores below 3 (reported, not gated): {a['hard_cases_template_below_3'] or 'none'}")
    payload["difficulty_a"] = a

    b = difficulty_b(logs["haiku"])
    lines.extend(_difficulty_b_lines(b, args.reading_b))
    payload["difficulty_b"] = b

    floor_lines, floors = floors_block(logs["sonnet"])
    lines.extend(floor_lines)
    payload["floors"] = floors

    if args.regression_baseline is not None:
        reg_lines, reg = regression_block(args.regression_baseline, args.regression_candidate, pin)
        lines.extend(reg_lines)
        payload["regression"] = reg
    else:
        lines.append("== Deliberate regression: not given (pass --regression-baseline and --regression-candidate)")
        payload["regression"] = None
    lines.append(f"Review CSV rows (distinct run, sample, epoch, route key): {len(rows)}")
    lines.append("PILOT REPORT: DONE")
    payload["review_rows"] = len(rows)
    return "\n".join(lines), payload, rows


def write_outputs(args: argparse.Namespace, payload: dict[str, Any], rows: list[dict[str, str]]) -> None:
    try:
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        if args.review_csv:
            with open(args.review_csv, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(REVIEW_CSV_COLUMNS), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
    except OSError as e:
        raise Refused(f"cannot write an output file: {type(e).__name__}: {e}") from e


# --- CLI ---------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 is not a contract code
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


class _HelpAction(argparse.Action):
    """-h/--help prints usage and exits 3: argparse's own exit 0 would read as a report."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any) -> None:
        super().__init__(option_strings, dest=dest, default=argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        parser.print_help()
        print("REFUSED: help requested; no report ran", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="Section 13 pilot row quantities from saved pilot logs (no model call).", add_help=False)
    p.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (no report ran)")
    p.add_argument("--sonnet", required=True, help="Sonnet 5.5 pilot .eval log")
    p.add_argument("--haiku", required=True, help="Haiku 5.5 pilot .eval log")
    p.add_argument("--template", required=True, help="template baseline pilot .eval log")
    p.add_argument("--reading-b", default="all", choices=(*READINGS, "all"),
                   help="difficulty check (b) reading of record; all three always print")
    p.add_argument("--regression-baseline", help="deliberate regression: the pilot baseline .eval log")
    p.add_argument("--regression-candidate", help="deliberate regression: the regressed candidate .eval log")
    p.add_argument("--json", dest="json_out", help="write the report as JSON to this path")
    p.add_argument("--review-csv", help="write the review-verdict CSV (verdicts blank) to this path")
    p.add_argument("--max-tokens-pin", type=int, default=None,
                   help="the owner's max_tokens pin, when one is set (default: max_tokens must be unset)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        text, payload, rows = build_report(args)
        write_outputs(args, payload, rows)
    except (Refused, GateError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as a report
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    _emit(text)
    return EXIT_OK


def _emit(text: str) -> None:
    """Print the report; a console that cannot encode a token gets backslash escapes."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc))


if __name__ == "__main__":
    sys.exit(main())
