"""The run report of an evaluation run, from its logs and frozen records. NO API key, no model call.

Usage:
  python scripts/run_report.py --worker LABEL=PATH [--worker LABEL=PATH ...] --template PATH
         [--packet-dir D [--manifest M] [--freeze-commit SHA]]
         [--second-look-dir D2 [--second-look-manifest M2] [--second-look-commit SHA] [--blind-check-dir D3]]
         [--first-archive URL] [--first-archive-date YYYY-MM-DD]
         [--second-archive URL] [--second-archive-date YYYY-MM-DD]
         [--repo .] [--max-tokens-pin N] [--json out.json]

Pre-registration section 8 (Metrics and reporting commitments, Review record and freeze,
Second look on review verdicts, The run report) and section 2 (H3), built as tooling item
(1) of the section 13 row on tooling built after the tag (decision D-R19, 2026-11-14).

Logs. A worker log and the template baseline log are read on the evaluation split only
when the header task_args hold allow_evaluation_split exactly True and the header metadata
names the evaluation split (scripts/review_packet.py log_split). A set of dev-split logs is
read as a rehearsal and every heading says so. Logs on different splits are refused. Every
log must be a successful whole run (regression.refuse_partial_run) whose every sample holds
epochs 1 to 4 (k = 4 asserted), and the logs must carry one Task version and dataset
content. Every worker log passes roster.check_run_config at epochs 4 (contract C2), with no
expected sample count: that count check counts only epochs without a sample error, and an
errored epoch is a failed trial here, never a refusal (section 6). The template baseline
log is out of that check, as in scripts/pilot_report.py. Two worker logs of one model are
refused.

Order of checks (section 8: "No review-settled or audit-adjusted figure of a
pre-registered evaluation run or tagged release run is computed or read until its second
look is done and frozen by a second manifest commit"):
  1. the first record against its manifest and freeze commit, the path
     scripts/pilot_report.py load_review_record takes (review_freeze check_record_files,
     pilot_report freeze_commit_gaps, running_package_gap and package_code_gaps), plus:
     the record names every worker log by eval id, each log hashes to the SHA-256 the
     record holds for it, the record names no run whose log is not given, and an
     evaluation-split run is recorded as one;
  2. the second-look freeze (review_freeze check_second_look_freeze).
If a check fails, or no second look is given, the report prints only figures that read
the log alone: the route-blind upper bound, the code-only extractor health figure with
the routed-epoch rate and findings per routed epoch, citation-pass^4 with its counts, the
format-break and errored-epoch counts, the code-caught FM counts and each model's routing
rate. It states that the review-settled and audit-adjusted figures are withheld, and why,
and never computes them. With both checks passing it prints, per worker and subset, every
item of the section 8 run report list this repository can compute, the template baseline
as a reference row and H3 (icfr_agent_evals/h3.py) on the review-settled, audit-adjusted
and before-D-R9 readings with the second-look gate per worker. An item the repository
cannot compute yet is printed as not available, with the reason.

A frozen record that passes both checks but does not cover a worker log
(trials.RecordError) is refused, every gap counted.

Exit codes:
  0  report printed (the withheld path included)
  3  refused: a missing or unreadable log, an evaluation-split log without the unlock,
     logs on different splits, a log that is not a whole successful run at k = 4, a
     worker log off the section 5 run configuration, logs whose Task version or dataset
     content differ, two worker logs of one model, a template log that is not the
     template baseline (or a worker log that is), a record that passes both checks but
     does not cover a worker log, an output path that cannot be written, a help
     request, or any argument error
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from icfr_agent_evals import h3
from icfr_agent_evals import reliability as rel
from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import roster
from icfr_agent_evals import settlement
from icfr_agent_evals import trials
from icfr_agent_evals.reliability import GateError

EXIT_OK, EXIT_REFUSED = 0, 3
SCRIPTS_DIR = Path(__file__).resolve().parent
PILOT_REPORT_SCRIPT = SCRIPTS_DIR / "pilot_report.py"
K = rel.PREREGISTERED_K
EVAL_SPLIT, DEV_SPLIT = "eval", "dev"
ALL = "all"
SYNTHETIC, EDGE, ADVERSARIAL, POOLED = "synthetic", "edge", "adversarial", "difficulty_bearing"
SUBSETS: tuple[str, ...] = (ALL, SYNTHETIC, EDGE, ADVERSARIAL, POOLED)
SUBSET_NOTES = {POOLED: "the 54-case labeled subset, edge plus adversarial"}
ROUTE_CLASSES: tuple[str, ...] = ("narrative_sweep", "code_review")
FIGURE_NAMES: dict[str, str] = {
    trials.REVIEW_SETTLED: "review-settled", trials.AUDIT_ADJUSTED: "audit-adjusted",
    trials.ROUTE_BLIND: "route-blind", trials.CODE_ONLY: "code-only",
}
FIGURE_ROLES: dict[str, str] = {
    trials.REVIEW_SETTLED: "the score of record (decision D-A)",
    trials.AUDIT_ADJUSTED: "a sensitivity figure, never the score of record",
    trials.ROUTE_BLIND: "an upper bound on the review-settled figure, not a capability result",
    trials.CODE_ONLY: "extractor health, not a capability result and never a score",
}
RECORD_KINDS: tuple[str, ...] = (trials.REVIEW_SETTLED, trials.AUDIT_ADJUSTED)
LOG_KINDS: tuple[str, ...] = (trials.ROUTE_BLIND, trials.CODE_ONLY)
WITHHELD = ("withheld: no review-settled or audit-adjusted figure is computed or read until the first record "
            "and the second look are both frozen and checked (pre-registration section 8)")
REHEARSAL = "REHEARSAL on the dev split, not an evaluation run"
# Section 8 items this repository cannot compute yet, each with its reason.
NOT_AVAILABLE: dict[str, str] = {
    "judge-gate results per sub-dimension": (
        "no Phase 1 scorer calls the grader yet, so the logs hold no grader score; the judge-trust statistics "
        "(icfr_agent_evals/judge_trust.py) read calibration and acceptance-look data this report is not given"),
    "route judge exploratory agreement": (
        "the route judge has not run: it runs only after both freezes, through its own package job, and no such "
        "job exists in this repository yet"),
    "labeling kappa and the adjudicated-case count": (
        "computed from the Branch B labeling record by scripts/labeling_kappa.py, not from run logs"),
    "the name the 54-case labeled subset earns (Verified, Self-verified or neither)": (
        "it follows from the labeling record and its kappa, which this report does not read; the subset is printed "
        "as difficulty_bearing"),
    "the unsettleable-route sensitivity row": (
        "section 8 asks for a row that drops the extractor-crash and answer-key-mismatch epochs from every model's "
        "figures, but does not fix how a sample with a dropped epoch enters content-pass^4 at k = 4; the epochs "
        "such a row would drop are counted per model and route key instead"),
}
# Unsettleable rows filed as an extractor or an answer-key defect: the epochs the sensitivity
# row would drop (counted only; see NOT_AVAILABLE).
DEFECT_ON_CORRECT: tuple[str, ...] = (settlement.ON_CORRECT_EXTRACTOR, settlement.ON_CORRECT_ANSWER_KEY)


class Refused(Exception):
    """An input the report must not read; exit 3."""


# --- Sibling scripts, loaded by path ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_pilot_report() -> Any:
    """scripts/pilot_report.py under a private name: the first-record check path
    (freeze_commit_gaps, running_package_gap, package_code_gaps, load_review_freeze) and
    check_same_content. Its behaviour is not changed; it stays dev-pilot only."""
    name = "pilot_report_for_run_report"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, PILOT_REPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def load_review_freeze() -> Any:
    """scripts/review_freeze.py, as scripts/pilot_report.py loads it (one module for both)."""
    return load_pilot_report().load_review_freeze()


def load_review_packet() -> Any:
    """scripts/review_packet.py, as scripts/review_freeze.py loads it (log_split, carried_reasons)."""
    return load_review_freeze().load_review_packet()


# --- Logs ---------------------------------------------------------------------------------------

def _sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reasoning_mode(log: Any) -> str:
    """The reasoning mode a log records: reasoning_effort and every other reasoning or
    thinking field its plan config sets (section 2: the reasoning mode beside every delta).
    roster.check_run_config has already checked the evaluated model's events carry the
    same effort pin."""
    config = getattr(getattr(log, "plan", None), "config", None)
    fields = config.model_dump(exclude_none=True) if config is not None else {}
    parts = [f"{k}={fields[k]}" for k in sorted(fields) if k.startswith("reasoning") or "thinking" in k]
    return ", ".join(parts) if parts else "none recorded"


def check_log(log: Any, role: str, name: str, max_tokens_pin: int | None = None) -> tuple[str, dict | None]:
    """Refuse a log the report must not read; return (its split, the run-config summary
    or None for the template)."""
    rp = load_review_packet()
    if getattr(log, "status", None) != "success":
        raise Refused(f"{role} log {name} status is {getattr(log, 'status', None)!r}, not 'success'")
    if getattr(log, "samples", None) is None:
        raise Refused(f"{role} log {name} has no samples; read it with header_only=False")
    try:
        split = rp.log_split(log, name)
    except rp.Refused as e:
        raise Refused(f"{role} log: {e}") from e
    from icfr_agent_evals.regression import refuse_partial_run

    try:
        refuse_partial_run(log)
    except GateError as e:
        raise Refused(f"{role} log {name}: {e}") from e
    ran = int(getattr(log.eval.config, "epochs", None) or 1)
    declared = (log.eval.task_args or {}).get("epochs")
    if ran != K or (declared is not None and int(declared) != ran):
        raise Refused(f"{role} log {name} ran epochs {ran} (task argument {declared}); the report asserts k = {K}")
    epochs_of: dict[str, list[int]] = defaultdict(list)
    for s in log.samples:
        epochs_of[str(s.id)].append(int(s.epoch))
    for sid, eps in sorted(epochs_of.items()):
        if sorted(eps) != list(range(1, K + 1)):
            raise Refused(f"{role} log {name} sample {sid} holds epochs {sorted(eps)}, not 1 to {K}")
    is_template = rel.is_template_baseline_log(log)
    if (role == "template") != is_template:
        raise Refused(f"{role} log {name} runs task {log.eval.task!r}")
    if role == "template":
        return split, None
    try:
        summary = roster.check_run_config(log, expected_samples=None, expected_epochs=K, allow_batch=True,
                                          max_tokens_pin=max_tokens_pin)
    except roster.RunConfigError as e:
        raise Refused(f"{role} log {name}: {e}") from e
    return split, summary


def read_log(path: str, role: str) -> Any:
    if not Path(path).is_file():
        raise Refused(f"{role} log not found: {Path(path).name}")
    from inspect_ai.log import read_eval_log

    try:
        return read_eval_log(str(path))
    except Exception as e:  # an unreadable log must never read as a report
        raise Refused(f"{role} log {Path(path).name} unreadable: {type(e).__name__}") from e


class LogInput:
    """One log the report reads: its label, file, hash, the EvalLog and its checks."""

    def __init__(self, label: str, path: str, log: Any, role: str, sha256: str | None = None):
        self.label, self.path, self.log, self.role = label, str(path), log, role
        self.sha256 = sha256 if sha256 is not None else _sha_file(path)
        self.split = ""
        self.run_config: dict | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def model(self) -> str:
        return str(self.log.eval.model)

    @property
    def eval_id(self) -> str:
        return str(self.log.eval.eval_id)


def check_inputs(workers: Sequence[LogInput], template: LogInput | None,
                 max_tokens_pin: int | None = None) -> tuple[str, dict]:
    """Every log checked (check_log), one split for all, one Task version and dataset
    content (pilot_report check_same_content), one log per worker model. Returns (the
    split, the content fingerprint)."""
    if not workers:
        raise Refused("no --worker given")
    labels = [w.label for w in workers]
    if len(set(labels)) != len(labels):
        raise Refused(f"duplicate worker label(s): {sorted(l for l, n in Counter(labels).items() if n > 1)}")
    for item in (*workers, *([template] if template is not None else [])):
        item.split, item.run_config = check_log(item.log, item.role, item.name, max_tokens_pin)
    splits = {item.split for item in (*workers, *([template] if template is not None else []))}
    if len(splits) > 1:
        raise Refused(f"the logs sit on different splits {sorted(splits)}; one report reads one split, never a mix")
    models = Counter(w.model for w in workers)
    twice = sorted(m for m, n in models.items() if n > 1)
    if twice:
        raise Refused(f"two worker logs of one model: {twice}")
    ids = Counter(item.eval_id for item in (*workers, *([template] if template is not None else [])))
    if any(n > 1 for n in ids.values()):
        raise Refused("two logs share an eval id")
    pr = load_pilot_report()
    content = {w.label: w.log for w in workers}
    if template is not None:
        content["template"] = template.log
    try:
        fingerprint = pr.check_same_content(content)
    except pr.Refused as e:
        raise Refused(str(e)) from e
    return splits.pop(), fingerprint


# --- The two freeze checks ------------------------------------------------------------------------

class Checks:
    """What the two freeze checks found. `passed` only when both checks ran clean."""

    def __init__(self) -> None:
        self.record: Any = None
        self.first_manifest: dict | None = None
        self.first_path: Path | None = None
        self.first_sha256: str | None = None
        self.first_commit: str = ""
        self.first_gaps: list[str] = []
        self.second_manifest: dict | None = None
        self.second_info: dict = {}
        self.second_gaps: list[str] = []
        self.run_ids: dict[str, str | None] = {}
        self.settlement_commit: str = ""

    @property
    def passed(self) -> bool:
        return (self.record is not None and self.first_manifest is not None and self.second_manifest is not None
                and not self.first_gaps and not self.second_gaps)

    def reasons(self) -> list[str]:
        out = [f"first record: {g}" for g in self.first_gaps]
        out += [f"second look: {g}" for g in self.second_gaps]
        return out


def check_first_record(checks: Checks, packet_dir: str | None, *, repo: Path, manifest: str | None,
                       freeze_commit: str | None, workers: Sequence[LogInput], template: LogInput | None,
                       split: str) -> None:
    """The first record and its freeze, the scripts/pilot_report.py load_review_record path
    with this report's repository: check_record_files, freeze_commit_gaps,
    running_package_gap and package_code_gaps; then the logs it names."""
    if packet_dir is None:
        checks.first_gaps.append("no first review record given (--packet-dir)")
        return
    pr = load_pilot_report()
    rf = load_review_freeze()
    if not Path(packet_dir).is_dir():
        checks.first_gaps.append(f"{packet_dir} is not a packet directory")
        return
    try:
        record = rf.load_frozen_record(str(packet_dir))
    except rf.Refused as e:
        checks.first_gaps.extend(f"the frozen record cannot be read: {g}" for g in e.gaps)
        record = None
    try:
        man, mbytes, mpath, gaps = rf.check_record_files(str(packet_dir), repo=str(repo), manifest=manifest)
    except rf.Refused as e:
        checks.first_gaps.extend(e.gaps)
        return
    checks.first_manifest, checks.first_path = man, Path(mpath)
    checks.first_sha256 = hashlib.sha256(mbytes).hexdigest()
    more, commit = pr.freeze_commit_gaps(Path(repo), Path(mpath), mbytes, man, freeze_commit)
    checks.first_commit = commit
    gaps = [*gaps, *more, *pr.running_package_gap(Path(repo)),
            *pr.package_code_gaps(Path(repo), str(man.get("run_commit", "")))]
    head = pr._git(Path(repo), "rev-parse", "HEAD")
    checks.settlement_commit = head.stdout.decode("utf-8", errors="replace").strip() if head.returncode == 0 else ""
    if record is not None:
        runs = dict(record.runs or {})
        by_eval = defaultdict(list)
        for rid, info in runs.items():
            by_eval[str((info or {}).get("eval_id"))].append(rid)
        given = set()
        for item in (*workers, *([template] if template is not None else [])):
            hits = by_eval.get(item.eval_id, [])
            if len(hits) > 1:
                gaps.append(f"the record names {len(hits)} runs with the eval id of {item.label}")
                continue
            if not hits:
                checks.run_ids[item.label] = None
                if item.role == "worker":
                    gaps.append(f"the record names no run with the eval id of worker log {item.label}")
                continue
            rid = hits[0]
            given.add(rid)
            checks.run_ids[item.label] = rid
            info = runs[rid] or {}
            if str(info.get("sha256")) != item.sha256:
                gaps.append(f"worker log {item.label} does not hash to the SHA-256 the record holds for run {rid}"
                            if item.role == "worker" else
                            f"the template log does not hash to the SHA-256 the record holds for run {rid}")
            recorded_split = EVAL_SPLIT if info.get("split") == EVAL_SPLIT else DEV_SPLIT
            if recorded_split != split:
                gaps.append(f"the record holds run {rid} on the {recorded_split} split, the logs are on {split}")
            if split == EVAL_SPLIT and info.get("allow_evaluation_split") is not True:
                gaps.append(f"the record does not hold run {rid} as unlocked for the evaluation split")
        for rid in sorted(set(runs) - given):
            gaps.append(f"the record names run {rid}, whose log was not given")
    checks.record = record
    checks.first_gaps.extend(gaps)


def check_second_look(checks: Checks, second_look_dir: str | None, *, repo: Path, manifest: str | None,
                      commit: str | None, blind_check_dir: str | None) -> None:
    """The second-look freeze (review_freeze check_second_look_freeze)."""
    if second_look_dir is None:
        checks.second_gaps.append("no second look given (--second-look-dir)")
        return
    if checks.first_path is None or not checks.first_commit:
        checks.second_gaps.append("the second look cannot be checked without a first manifest frozen in a commit")
        return
    rf = load_review_freeze()
    man, info, gaps = rf.check_second_look_freeze(
        repo=str(repo), first_manifest_path=checks.first_path, first_freeze_commit=checks.first_commit,
        second_look_dir=second_look_dir, blind_check_dir=blind_check_dir, manifest=manifest, commit=commit)
    checks.second_manifest, checks.second_info = man, info
    checks.second_gaps.extend(gaps)


# --- Subsets ------------------------------------------------------------------------------------

def subsets_of(case_kind: str | None) -> tuple[str, ...]:
    """The report subsets a case belongs to: all, then synthetic, or its kind and the
    pooled difficulty-bearing subset."""
    if case_kind == "standard":
        return (ALL, SYNTHETIC)
    if case_kind in (EDGE, ADVERSARIAL):
        return (ALL, case_kind, POOLED)
    return (ALL,)


def _sample_kind(r: Any) -> str | None:
    return next((e.case_kind for e in r.epochs if e.case_kind), None)


def sub_report(report: trials.TrialReport, subset: str) -> trials.TrialReport:
    samples = {sid: r for sid, r in report.samples.items() if subset in subsets_of(_sample_kind(r))}
    return trials.TrialReport(kind=report.kind, k=report.k, samples=samples)


def _rate_row(passed: int, n: int) -> dict[str, Any]:
    lo, hi = trials.clopper_pearson(passed, n)
    return {"passed": passed, "n": n, "rate": passed / n if n else None, "ci95": [lo, hi]}


def _present(rows: Mapping[str, Any]) -> list[str]:
    return [s for s in SUBSETS if s in rows]


def pass_rows(report: trials.TrialReport) -> dict[str, dict[str, Any]]:
    """Per subset: content-pass^4 with its Clopper-Pearson interval, and the same figure
    under the rule before D-R9."""
    out: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        sub = sub_report(report, subset)
        if not sub.n:
            continue
        old = trials.before_d_r9_figures({report.kind: sub})[report.kind]
        out[subset] = {**_rate_row(sub.passed, sub.n), "before_d_r9": _rate_row(old["passed"], old["n"])}
    return out


def anchor_rows(report: trials.TrialReport) -> dict[str, dict[str, Any]]:
    """Per subset: trials.anchor_means (mean, SE of per-sample epoch means, normal 95% interval)."""
    out = {}
    for subset in SUBSETS:
        sub = sub_report(report, subset)
        if sub.n:
            out[subset] = trials.anchor_means(sub)
    return out


def citation_rows(route_blind: trials.TrialReport, settled: trials.TrialReport | None) -> dict[str, dict]:
    """Per subset: citation-pass^4 with its interval, the epoch and token counts, and the
    review-found count when the review-settled report is given (it never enters the figure)."""
    out = {}
    for subset in SUBSETS:
        sub = sub_report(route_blind, subset)
        if not sub.n:
            continue
        rs_sub = sub_report(settled, subset) if settled is not None else None
        c = trials.citation_omission_figures(sub, rs_sub)
        out[subset] = {k: c[k] for k in ("passed", "n", "rate", "ci95", "epochs_with_omission", "omitted_tokens",
                                         "p12_tokens", "wallet_label_tokens", "epochs_not_assessable",
                                         "review_found")}
    return out


# --- Log-only figures ------------------------------------------------------------------------------

def _kind_of_sample(sample: Any) -> str | None:
    meta = getattr(sample, "metadata", None) or {}
    truth = meta.get("truth") if isinstance(meta, Mapping) else None
    return str(truth.get("case_kind")) if isinstance(truth, Mapping) and truth.get("case_kind") else None


def epoch_counts(log: Any) -> dict[str, dict[str, Any]]:
    """Per subset, from the log alone: epochs, format-break epochs (an FM-000 parse failure
    without a sample error), errored epochs, carried epochs (review_packet carried_reasons:
    a sample error or an unscored trial anchor, a failed trial with no review unit), routed
    epochs (any section 10(c) route, settleable or not), settleable findings, the routing
    rate (routed epochs over epochs, also the extractor-health routed-epoch rate), findings
    per routed epoch and the code-caught FM counts by code, (epoch, code) once each."""
    rp = load_review_packet()
    _, facts = trials._collect(log.samples, K)  # the routes grouped exactly as the packet groups them
    out: dict[str, dict[str, Any]] = {}
    by_ident = {(str(s.id), int(s.epoch)): s for s in log.samples}
    for ident, f in facts.items():
        s = by_ident[ident]
        fb = (getattr(s, "error", None) is None
              and any((getattr(v, "metadata", None) or {}).get("format_break") for v in (s.scores or {}).values()))
        for subset in subsets_of(f.case_kind):
            row = out.setdefault(subset, {"epochs": 0, "format_break_epochs": 0, "errored_epochs": 0,
                                          "carried_epochs": 0, "routed_epochs": 0, "findings": 0,
                                          "fm_codes": Counter()})
            row["epochs"] += 1
            row["format_break_epochs"] += fb
            row["errored_epochs"] += f.error
            row["carried_epochs"] += bool(rp.carried_reasons(s))
            row["routed_epochs"] += f.routed
            row["findings"] += len(f.findings)
            row["fm_codes"].update(dict.fromkeys(f.codes, 1))
    for row in out.values():
        row["routing_rate"] = row["routed_epochs"] / row["epochs"] if row["epochs"] else None
        row["findings_per_routed_epoch"] = row["findings"] / row["routed_epochs"] if row["routed_epochs"] else None
        row["fm_codes"] = dict(sorted(row["fm_codes"].items()))
    return {s: out[s] for s in SUBSETS if s in out}


def log_only_block(item: LogInput) -> tuple[dict[str, Any], dict[str, trials.TrialReport]]:
    """Every figure that reads the log alone (no review record)."""
    res = trials.figures(item.log)
    reports = res["reports"]
    r = rel.report_from_log(item.log, run_config=False)  # the run config is checked in check_log
    payload = {
        "figures": {kind: pass_rows(reports[kind]) for kind in LOG_KINDS},
        "citation_pass4": citation_rows(reports[trials.ROUTE_BLIND], None),
        "epochs": epoch_counts(item.log),
        "fm_counts": r.fm_counts, "fm002_by_direction": r.fm002_by_direction,
    }
    return payload, reports


# --- Record figures ------------------------------------------------------------------------------

def _member_list(m: Mapping[str, Any], col: str) -> list:
    try:
        value = json.loads(m.get(col) or "[]")
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def record_tables(record: Any, run_id: str, kinds: Mapping[str, str | None]) -> dict[str, Any]:
    """Per subset, from the frozen record's members of one run: finding counts and finding
    units by route key and route class, the share of findings ruled `correct` (the
    extractor's false-alarm rate) by route key and route class, the verdict distribution by
    route class, unsettleable-route counts by route key and the defect epochs among them,
    narrative members and units by class, whole-narrative `wrong` counts (members and units)
    by class, and the audit verdicts by code."""
    verdict = {row.get("unit_id", ""): row.get("verdict", "") for row in record.rows or ()}
    members = [m for m in record.members or () if m.get("run_id") == run_id]
    out: dict[str, dict[str, Any]] = {}

    def row_for(subset: str) -> dict[str, Any]:
        return out.setdefault(subset, {
            "findings_by_route_key": Counter(), "finding_units_by_route_key": defaultdict(set),
            "findings_by_route_class": Counter(), "finding_units_by_route_class": defaultdict(set),
            "correct_by_route_key": Counter(), "correct_by_route_class": Counter(),
            "verdicts_by_route_class": defaultdict(Counter),
            "unsettleable_by_route_key": Counter(), "defect_epochs_by_route_key": Counter(),
            "narrative_members_by_class": Counter(), "narrative_units_by_class": defaultdict(set),
            "wrong_members_by_class": Counter(), "wrong_units_by_class": defaultdict(set),
            "audit_by_code": defaultdict(Counter),
        })

    for m in members:
        kind = m.get("member_kind")
        for subset in subsets_of(kinds.get(str(m.get("sample_id")))):
            row = row_for(subset)
            uid, key, cls = m.get("unit_id", ""), m.get("route_key", ""), m.get("route_class", "")
            v = verdict.get(uid, "")
            if kind == "finding":
                row["findings_by_route_key"][key] += 1
                row["finding_units_by_route_key"][key].add(uid)
                row["findings_by_route_class"][cls] += 1
                row["finding_units_by_route_class"][cls].add(uid)
                row["correct_by_route_key"][key] += v == trials.CORRECT
                row["correct_by_route_class"][cls] += v == trials.CORRECT
                row["verdicts_by_route_class"][cls][v or "none"] += 1
            elif kind == "unsettleable":
                row["unsettleable_by_route_key"][key] += 1
                try:
                    rows = settlement.rows_for(key, [str(r) for r in _member_list(m, "reasons")])
                except ValueError:
                    rows = ()
                if any(not r.settleable and r.on_correct in DEFECT_ON_CORRECT for r in rows):
                    row["defect_epochs_by_route_key"][key] += 1
            elif kind == "narrative":
                row["narrative_members_by_class"][cls] += 1
                row["narrative_units_by_class"][cls].add(uid)
                if v == trials.WRONG:
                    row["wrong_members_by_class"][cls] += 1
                    row["wrong_units_by_class"][cls].add(uid)
            elif kind == "fm_audit":
                codes = [str(c) for c in _member_list(m, "fm_codes")]
                code = (record.units.get(uid) or {}).get("fm_code") or (codes[0] if codes else "")
                row["audit_by_code"][code][v or "none"] += 1

    def finish(row: dict[str, Any]) -> dict[str, Any]:
        share_key = {k: row["correct_by_route_key"][k] / n for k, n in row["findings_by_route_key"].items() if n}
        share_cls = {k: row["correct_by_route_class"][k] / n for k, n in row["findings_by_route_class"].items() if n}
        return {
            "findings_by_route_key": dict(sorted(row["findings_by_route_key"].items())),
            "finding_units_by_route_key": {k: len(v) for k, v in sorted(row["finding_units_by_route_key"].items())},
            "findings_by_route_class": {c: row["findings_by_route_class"].get(c, 0) for c in ROUTE_CLASSES},
            "finding_units_by_route_class": {c: len(row["finding_units_by_route_class"].get(c, ()))
                                             for c in ROUTE_CLASSES},
            "share_ruled_correct_by_route_key": dict(sorted(share_key.items())),
            "share_ruled_correct_by_route_class": dict(sorted(share_cls.items())),
            "verdicts_by_route_class": {c: dict(sorted(row["verdicts_by_route_class"].get(c, Counter()).items()))
                                        for c in ROUTE_CLASSES},
            "unsettleable_by_route_key": dict(sorted(row["unsettleable_by_route_key"].items())),
            "defect_epochs_by_route_key": dict(sorted(row["defect_epochs_by_route_key"].items())),
            "narrative_members_by_class": {c: row["narrative_members_by_class"].get(c, 0)
                                           for c in rs.NARRATIVE_CLASSES},
            "narrative_units_by_class": {c: len(row["narrative_units_by_class"].get(c, ()))
                                         for c in rs.NARRATIVE_CLASSES},
            "whole_narrative_wrong_members_by_class": {c: row["wrong_members_by_class"].get(c, 0)
                                                       for c in rs.NARRATIVE_CLASSES},
            "whole_narrative_wrong_units_by_class": {c: len(row["wrong_units_by_class"].get(c, ()))
                                                     for c in rs.NARRATIVE_CLASSES},
            "audit_by_code": {c: dict(sorted(v.items())) for c, v in sorted(row["audit_by_code"].items())},
            "audit_extractor_error": sum(v.get(trials.EXTRACTOR_ERROR, 0) for v in row["audit_by_code"].values()),
        }

    return {s: finish(out[s]) for s in SUBSETS if s in out}


def review_found_rows(settled: trials.TrialReport) -> dict[str, dict[str, Any]]:
    """Per subset: review-found codes, citation omissions and whole-narrative wrong epochs
    on the review-settled trial, counted apart from the code-caught counts."""
    out = {}
    for subset in SUBSETS:
        sub = sub_report(settled, subset)
        if not sub.n:
            continue
        eps = sub.epochs
        out[subset] = {"codes": dict(sorted(Counter(c for e in eps for c in e.review_found).items())),
                       "citation_omissions": sum(e.review_found_omissions for e in eps),
                       "whole_narrative_wrong_epochs": sum(1 for e in eps if e.narrative_errors)}
    return out


def record_block(item: LogInput, record: Any, run_id: str,
                 log_reports: Mapping[str, trials.TrialReport]) -> tuple[dict[str, Any], dict[str, trials.TrialReport]]:
    """Every review-settled and audit-adjusted figure of one worker run. Called only after
    both freeze checks have passed. A record that does not cover the run is refused."""
    try:
        settled = trials.review_settled_trial(item.log, record, run_id=run_id)
        audit = trials.audit_adjusted_trial(item.log, record, run_id=run_id)
    except trials.RecordError as e:
        shown = "; ".join(e.gaps[:10])
        more = f"; and {len(e.gaps) - 10} more" if len(e.gaps) > 10 else ""
        raise Refused(f"the frozen record does not cover worker log {item.label} (run {run_id}, "
                      f"{len(e.gaps)} gap(s)): {shown}{more}") from e
    kinds = {sid: _sample_kind(r) for sid, r in settled.samples.items()}
    codes = sorted({c for e in settled.epochs for c in e.failure_modes} |
                   {c for e in log_reports[trials.CODE_ONLY].epochs for c in e.failure_modes})
    audited = trials.audited_code_counts(item.log, record, run_id=run_id, codes=tuple(codes), case_kinds=None)
    payload = {
        "run_id": run_id,
        "figures": {trials.REVIEW_SETTLED: pass_rows(settled), trials.AUDIT_ADJUSTED: pass_rows(audit)},
        "fm008_review_settled": list(settled.fm008),
        "anchor_means": {trials.REVIEW_SETTLED: anchor_rows(settled), trials.AUDIT_ADJUSTED: anchor_rows(audit)},
        "citation_pass4": citation_rows(log_reports[trials.ROUTE_BLIND], settled),
        "review_found": review_found_rows(settled),
        "record_tables": record_tables(record, run_id, kinds),
        "code_audit_counts": audited,
    }
    return payload, {trials.REVIEW_SETTLED: settled, trials.AUDIT_ADJUSTED: audit}


# --- Second look and run-level facts ----------------------------------------------------------------

def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def second_look_block(second_look_dir: str, checks: Checks) -> dict[str, Any]:
    """Per worker run id: the second-look result (N, the target n, the drawn counts per
    first-verdict class and route class, kappa with its bootstrap interval, the flips, the
    gate and the narrative-only `wrong` count labeled not second-looked), from the folder
    the second-look check has just hashed; the run's bootstrap seed and the blinding check."""
    rf = load_review_freeze()
    d2 = Path(second_look_dir)
    result = _read_json_file(d2 / rf.SECOND_LOOK_RESULT_FILE) or {}
    meta = _read_json_file(d2 / rf.SECOND_LOOK_META_FILE) or {}
    man = checks.second_manifest or {}
    runs = {}
    for rid, r in sorted((result.get("runs") or {}).items()):
        draw = (meta.get("runs") or {}).get(rid) or {}
        runs[rid] = {
            "N": r.get("N"), "target_n": draw.get("n"), "drawn": r.get("n"),
            "drawn_by_class_and_route_class": draw.get("strata"),
            "class_totals": {rf.FIRST_NOT_CORRECT: draw.get("not_correct_total"),
                             rf.FIRST_CORRECT: draw.get("correct_total")},
            "first_class_counts": r.get("first_class_counts"), "second_class_counts": r.get("second_class_counts"),
            "kappa": r.get("kappa"), "ci_lower": r.get("ci_lower"), "ci_upper": r.get("ci_upper"),
            "resamples": r.get("resamples"), "degenerate_resamples": r.get("degenerate_resamples"),
            "correct_to_not_correct": r.get("correct_to_not_correct"),
            "not_correct_to_correct": r.get("not_correct_to_correct"), "disagreements": r.get("disagreements"),
            "citation_omission_flips": {k: r.get(k) for k in (
                "citation_omission_to_wrong", "wrong_to_citation_omission", "citation_omission_to_correct",
                "correct_to_citation_omission")},
            "gate": r.get("gate"),
            "narrative_only_wrong_not_second_looked": r.get("narrative_only_wrong_not_second_looked"),
        }
    return {"agreement": man.get("agreement"), "bootstrap_seed": man.get("bootstrap_seed"),
            "bootstrap_seed_check": "derived from the first manifest, the run id and bootstrap; decision D-R19",
            "draw_seed": man.get("draw_seed"), "runs": runs, "blinding": man.get("blinding"),
            "reviewer": man.get("reviewer"), "date": man.get("date")}


def freeze_block(checks: Checks, args: argparse.Namespace | None) -> dict[str, Any]:
    """The freeze commits with their archive URLs (given as arguments) and every file hash."""
    a = args or argparse.Namespace()
    first = checks.first_manifest or {}
    second = checks.second_manifest or {}
    return {
        "first": {"manifest": str(checks.first_path) if checks.first_path else None,
                  "manifest_sha256": checks.first_sha256, "freeze_commit": checks.first_commit or None,
                  "archive_url": getattr(a, "first_archive", None), "archive_date": getattr(a, "first_archive_date", None),
                  "run_commit": first.get("run_commit"), "logs": first.get("logs"),
                  "private_files": first.get("private_files"), "vocab_sha256": first.get("vocab_sha256"),
                  "settlement_table_sha256": first.get("settlement_table_sha256")},
        "second": {"manifest": checks.second_info.get("path"), "manifest_sha256": checks.second_info.get("sha256"),
                   "commit": checks.second_info.get("commit"),
                   "archive_url": getattr(a, "second_archive", None),
                   "archive_date": getattr(a, "second_archive_date", None),
                   "private_files": second.get("private_files")},
        "settlement_commit": checks.settlement_commit or None,
    }


def carried_by_run(packet_dir: str | None) -> dict[str, int] | None:
    """Per masked run id, the carried failed-trial count sealed/summary.json records."""
    if packet_dir is None:
        return None
    data = _read_json_file(Path(packet_dir) / rs.PACKET_FILES["summary"])
    if not isinstance(data, dict):
        return None
    return {rid: int(v.get(trials.CARRIED_KEY, 0)) for rid, v in data.items()
            if rid != "totals" and isinstance(v, dict) and trials.CARRIED_KEY in v}


# --- H3 -------------------------------------------------------------------------------------------

def h3_block(workers: Sequence[LogInput], reports: Mapping[str, Mapping[str, trials.TrialReport]],
             gates: Mapping[str, str | None]) -> dict[str, Any]:
    """H3 through icfr_agent_evals/h3.py on the review-settled, audit-adjusted and
    before-D-R9 readings, with each worker's second-look gate result and its reasoning
    mode read from its log. A run H3 cannot be computed on (not the four roster workers,
    not 54 difficulty-bearing cases) prints the reason, never a figure."""
    models = sorted(w.model for w in workers)
    if models != sorted(h3.WORKERS):
        return {"computed": False, "reason": f"H3 reads the four roster workers {sorted(h3.WORKERS)}; this run holds "
                                             f"{models}"}
    settled = {w.model: reports[w.label][trials.REVIEW_SETTLED] for w in workers}
    audit = {w.model: reports[w.label][trials.AUDIT_ADJUSTED] for w in workers}
    second = {w.model: gates[w.label] for w in workers if gates.get(w.label)}
    modes = {w.model: reasoning_mode(w.log) for w in workers}
    try:
        out = h3.h3_from_reports(settled, audit, second, k=K, reasoning_modes=modes)
    except h3.Refused as e:
        return {"computed": False, "reason": str(e)}
    return {
        "computed": True,
        "readings": {name: h3.result_dict(out[name]) for name in h3.READINGS if name in out},
        "text": {name: h3.format_h3(out[name]) for name in h3.READINGS if name in out},
        "robustness": out["robustness"], "unscored": out["unscored"],
    }


# --- Text -----------------------------------------------------------------------------------------

def _ci(row: Mapping[str, Any]) -> str:
    lo, hi = row["ci95"]
    return f"{row['passed']}/{row['n']} (Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}])"


def _sub(subset: str) -> str:
    note = SUBSET_NOTES.get(subset)
    return f"{subset} ({note})" if note else subset


def _figure_lines(kind: str, rows: Mapping[str, Any]) -> list[str]:
    lines = [f"  {FIGURE_NAMES[kind]} content-pass^4, {FIGURE_ROLES[kind]}:"]
    for subset in _present(rows):
        r = rows[subset]
        lines.append(f"    {_sub(subset)}: {_ci(r)}; under the rule before D-R9 {_ci(r['before_d_r9'])}")
    return lines


def _citation_lines(rows: Mapping[str, Any]) -> list[str]:
    lines = ["  citation-pass^4 (decision D-R9: from the log alone, never a content failure):"]
    for subset in _present(rows):
        r = rows[subset]
        found = "withheld" if r["review_found"] is None else str(r["review_found"])
        lines.append(f"    {_sub(subset)}: {_ci(r)}; epochs with an omission {r['epochs_with_omission']}, "
                     f"omitted tokens {r['omitted_tokens']} (P12 {r['p12_tokens']}, wallet-label matcher "
                     f"{r['wallet_label_tokens']}), not assessable {r['epochs_not_assessable']}; review-found "
                     f"{found}")
    return lines


def _epoch_lines(rows: Mapping[str, Any]) -> list[str]:
    lines = ["  epochs (from the log alone):"]
    for subset in _present(rows):
        r = rows[subset]
        rate = "none" if r["routing_rate"] is None else f"{r['routing_rate']:.3f}"
        per = "none" if r["findings_per_routed_epoch"] is None else f"{r['findings_per_routed_epoch']:.3f}"
        codes = ", ".join(f"{c} {n}" for c, n in r["fm_codes"].items()) or "none"
        lines.append(f"    {_sub(subset)}: {r['epochs']} epochs; format breaks {r['format_break_epochs']}; errored "
                     f"{r['errored_epochs']}; carried failed trials {r['carried_epochs']}; routing rate {rate} "
                     f"({r['routed_epochs']} routed epochs); findings per routed epoch {per}; code-caught FM "
                     f"counts {codes}")
    return lines


def _mean_text(m: Mapping[str, Any]) -> str:
    if m.get("mean") is None:
        return "none"
    if m.get("ci95_low") is None:
        return f"{m['mean']:.3f}"
    return f"{m['mean']:.3f} [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}]"


def _record_lines(block: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for kind in RECORD_KINDS:
        lines.extend(_figure_lines(kind, block["figures"][kind]))
    lines.append(f"  FM-008 on the review-settled trial: {len(block['fm008_review_settled'])}")
    lines.append("  anchor means, review-settled beside audit-adjusted (mean [normal 95% CI on the SE of per-sample "
                 "epoch means]):")
    settled, audit = (block["anchor_means"][k] for k in RECORD_KINDS)
    for subset in _present(settled):
        for anchor in rel.TRIAL_ANCHORS:
            lines.append(f"    {_sub(subset)} {anchor}: review-settled {_mean_text(settled[subset][anchor])}, "
                         f"audit-adjusted {_mean_text(audit[subset][anchor])}")
    for subset in _present(block["review_found"]):
        r = block["review_found"][subset]
        codes = ", ".join(f"{c} {n}" for c, n in r["codes"].items()) or "none"
        lines.append(f"  review-found, {_sub(subset)} (counted apart from code-caught): codes {codes}; citation "
                     f"omissions {r['citation_omissions']}; whole-narrative wrong epochs "
                     f"{r['whole_narrative_wrong_epochs']}")
    tables = block["record_tables"]
    for subset in _present(tables):
        t = tables[subset]
        lines.append(f"  review queue, {_sub(subset)}:")
        lines.append(f"    findings by route class {t['findings_by_route_class']}; finding units by route class "
                     f"{t['finding_units_by_route_class']}")
        lines.append(f"    findings by route key {t['findings_by_route_key']}; finding units by route key "
                     f"{t['finding_units_by_route_key']}")
        lines.append(f"    verdict distribution by route class {t['verdicts_by_route_class']}")
        lines.append(f"    extractor health: share of findings ruled correct (false-alarm rate) by route class "
                     f"{_shares(t['share_ruled_correct_by_route_class'])}, by route key "
                     f"{_shares(t['share_ruled_correct_by_route_key'])}; code-audit extractor_error "
                     f"{t['audit_extractor_error']}")
        lines.append(f"    unsettleable routes by route key {t['unsettleable_by_route_key']}; of them extractor or "
                     f"answer-key defect epochs {t['defect_epochs_by_route_key']}")
        lines.append(f"    narrative units by class {t['narrative_units_by_class']}; whole-narrative wrong units "
                     f"{t['whole_narrative_wrong_units_by_class']} (members "
                     f"{t['whole_narrative_wrong_members_by_class']})")
        lines.append(f"    code-audit verdicts by code {t['audit_by_code']}")
    return lines


def _shares(d: Mapping[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in d.items()) + "}"


def _second_look_lines(sl: Mapping[str, Any], rid: str | None) -> list[str]:
    r = (sl.get("runs") or {}).get(rid or "")
    if r is None:
        return ["  second look: no result for this run in the second-look record"]
    gate = r.get("gate") or {}
    ci = ("none (kappa undefined)" if r.get("ci_lower") is None
          else f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
    kappa = "undefined" if r.get("kappa") is None else f"{r['kappa']:.3f}"
    return [f"  second look ({sl.get('agreement')} agreement, never inter-rater): N {r['N']}, target n "
            f"{r['target_n']}, drawn {r['drawn']}; drawn by first-verdict class and route class "
            f"{r['drawn_by_class_and_route_class']}; kappa {kappa}, bootstrap 95% CI {ci}; first correct now "
            f"otherwise {r['correct_to_not_correct']}, the other way {r['not_correct_to_correct']}; "
            f"citation-omission flips {r['citation_omission_flips']}",
            f"  second-look gate: {gate.get('result')} (branch {gate.get('branch')}, passed {gate.get('passed')}); "
            f"narrative-only units first ruled whole-narrative wrong {r['narrative_only_wrong_not_second_looked']}, "
            f"not second-looked"]


def format_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    head = "RUN REPORT" if payload["split"] == EVAL_SPLIT else f"RUN REPORT ({REHEARSAL})"
    lines.append(f"{head}: split {payload['split']}, {len(payload['workers'])} worker log(s), k = {K}; "
                 f"dataset_sha256 {payload['fingerprint'].get('dataset_sha256')}")
    if payload["withheld"]:
        lines.append("Review-settled and audit-adjusted figures: " + WITHHELD)
        lines.extend(f"  why: {r}" for r in payload["withheld_reasons"])
    else:
        fz = payload["freezes"]
        lines.append(f"First record frozen in {fz['first']['freeze_commit']}, second look frozen in "
                     f"{fz['second']['commit']}; both checked against the settlement commit "
                     f"{fz['settlement_commit']} (HEAD)")
    for label, w in payload["workers"].items():
        lines.append(f"== Worker {label}: {w['model']} | eval_id {w['eval_id']} | {w['file_name']} sha256 "
                     f"{w['sha256']} | reasoning mode {w['reasoning_mode']}")
        lines.extend(_figure_lines(trials.ROUTE_BLIND, w["log"]["figures"][trials.ROUTE_BLIND]))
        lines.append("  extractor health (not a capability result):")
        lines.extend("  " + x for x in _figure_lines(trials.CODE_ONLY, w["log"]["figures"][trials.CODE_ONLY]))
        lines.extend(_citation_lines(w["record"]["citation_pass4"] if w.get("record") else w["log"]["citation_pass4"]))
        lines.extend(_epoch_lines(w["log"]["epochs"]))
        lines.append(f"  FM-002 by direction (code-caught): {w['log']['fm002_by_direction']}")
        if w.get("record") is None:
            lines.append("  review-settled and audit-adjusted content-pass^4, anchor means, review queue tables, "
                         "review-found counts and code-audit counts: withheld")
            continue
        lines.extend(_record_lines(w["record"]))
        carried = w.get("carried_in_packet")
        lines.append(f"  carried failed trials: {w['log']['epochs'][ALL]['carried_epochs']} in the log, "
                     f"{'not recorded' if carried is None else carried} in the packet")
        lines.extend(_second_look_lines(payload["second_look"], w["record"]["run_id"]))
    t = payload.get("template")
    if t is not None:
        lines.append(f"== Template baseline (reference row, no model): eval_id {t['eval_id']} | {t['file_name']} "
                     f"sha256 {t['sha256']}")
        lines.extend(_figure_lines(trials.ROUTE_BLIND, t["log"]["figures"][trials.ROUTE_BLIND]))
        lines.extend(_figure_lines(trials.CODE_ONLY, t["log"]["figures"][trials.CODE_ONLY]))
        lines.extend(_citation_lines(t["log"]["citation_pass4"]))
        lines.append("  anchor means as the route-blind trial settles them (the template has no review):")
        for subset in _present(t["anchor_means_route_blind"]):
            for anchor, m in t["anchor_means_route_blind"][subset].items():
                lines.append(f"    {_sub(subset)} {anchor}: {_mean_text(m)}")
        lines.append("  Sample Consistency and FM-007 are not reported for the template baseline (section 8)")
    if not payload["withheld"]:
        run = payload["run"]
        lines.append("== Run")
        lines.append(f"review hours {run['review_hours']} over {run['sittings']} sitting(s) (the run's one record; "
                     f"worker runs share it)")
        sl = payload["second_look"]
        lines.append(f"second look: bootstrap seed {sl['bootstrap_seed']} ({sl['bootstrap_seed_check']}); draw seed "
                     f"{sl['draw_seed']}; reviewer {sl['reviewer']}, {sl['date']}")
        b = sl.get("blinding")
        if b is None:
            lines.append("blinding check: not run (only a full run of four worker runs has one)")
        else:
            lines.append(f"blinding check: named correctly {b['named_correctly']} of {b['n']} drawn, share "
                         f"{b['share']:.3f}, Clopper-Pearson 95% [{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
                         f"against chance {b['chance']}: {b['status']}")
        fz = payload["freezes"]
        for name in ("first", "second"):
            f = fz[name]
            commit = f.get("freeze_commit") if name == "first" else f.get("commit")
            lines.append(f"{name} freeze: commit {commit}, archive {f['archive_url'] or 'not given'} "
                         f"({f['archive_date'] or 'no date given'}); manifest {f['manifest']} sha256 "
                         f"{f['manifest_sha256']}")
            for rel_path, digest in sorted((f.get("private_files") or {}).items()):
                lines.append(f"  {rel_path} {digest}")
        for log in fz["first"].get("logs") or ():
            lines.append(f"  log {log.get('file_name')} sha256 {log.get('sha256')} eval_id {log.get('eval_id')}")
        hb = payload["h3"]
        lines.append("== H3 (section 2)")
        if not hb["computed"]:
            lines.append(f"H3 not computed: {hb['reason']}")
        else:
            for name in h3.READINGS:
                if name in hb["text"]:
                    lines.append(hb["text"][name])
            labels = hb["robustness"]["labels"]
            lines.append("H3 robustness: " + (", ".join(labels) if labels else "the three readings give one ordering"))
            lines.append(f"H3 anchor cells read as 0 because unscored (an errored or unscored epoch), per reading "
                         f"and model: {hb['unscored']}")
    lines.append("== Not available in this repository yet")
    for item, why in payload["not_available"].items():
        lines.append(f"  {item}: not available; {why}")
    lines.append("RUN REPORT: DONE")
    return "\n".join(lines)


# --- Build ------------------------------------------------------------------------------------------

def build_report(workers: Sequence[LogInput], template: LogInput | None, *, packet_dir: str | None = None,
                 manifest: str | None = None, freeze_commit: str | None = None,
                 second_look_dir: str | None = None, second_look_manifest: str | None = None,
                 second_look_commit: str | None = None, blind_check_dir: str | None = None,
                 repo: str | Path | None = None, max_tokens_pin: int | None = None,
                 args: argparse.Namespace | None = None) -> tuple[str, dict[str, Any]]:
    """The report text and its JSON payload. Every input check runs first; then the two
    freeze checks; the record figures only when both pass."""
    repo_path = Path(repo) if repo is not None else load_pilot_report().REPO_ROOT
    split, fingerprint = check_inputs(workers, template, max_tokens_pin)
    checks = Checks()
    check_first_record(checks, packet_dir, repo=repo_path, manifest=manifest, freeze_commit=freeze_commit,
                       workers=workers, template=template, split=split)
    check_second_look(checks, second_look_dir, repo=repo_path, manifest=second_look_manifest,
                      commit=second_look_commit, blind_check_dir=blind_check_dir)
    withheld = not checks.passed
    payload: dict[str, Any] = {
        "split": split, "rehearsal": split != EVAL_SPLIT, "fingerprint": fingerprint, "withheld": withheld,
        "withheld_reasons": checks.reasons() if withheld else [], "workers": {}, "template": None,
        "not_available": dict(NOT_AVAILABLE),
    }
    reports: dict[str, dict[str, trials.TrialReport]] = {}
    carried = carried_by_run(packet_dir) if not withheld else None
    second_look = second_look_block(second_look_dir, checks) if not withheld else None
    gates: dict[str, str | None] = {}
    for w in workers:
        log_payload, log_reports = log_only_block(w)
        entry: dict[str, Any] = {"model": w.model, "eval_id": w.eval_id, "file_name": w.name, "sha256": w.sha256,
                                 "reasoning_mode": reasoning_mode(w.log), "run_config": w.run_config,
                                 "log": log_payload, "record": None}
        if not withheld:
            rid = checks.run_ids[w.label]
            rec_payload, rec_reports = record_block(w, checks.record, rid, log_reports)
            entry["record"] = rec_payload
            entry["carried_in_packet"] = (carried or {}).get(rid)
            entry["second_look"] = (second_look["runs"] or {}).get(rid)
            gates[w.label] = ((entry["second_look"] or {}).get("gate") or {}).get("result")
            reports[w.label] = rec_reports
        payload["workers"][w.label] = entry
    if template is not None:
        log_payload, log_reports = log_only_block(template)
        payload["template"] = {"model": template.model, "eval_id": template.eval_id, "file_name": template.name,
                               "sha256": template.sha256, "log": log_payload,
                               "anchor_means_route_blind": anchor_rows(log_reports[trials.ROUTE_BLIND])}
    if not withheld:
        first = checks.first_manifest or {}
        payload["run"] = {"review_hours": first.get("review_hours"), "sittings": len(first.get("sittings") or ()),
                          "counts": first.get("counts"),
                          "verdict_distribution_by_route_class": first.get("verdict_distribution_by_route_class")}
        payload["second_look"] = second_look
        payload["freezes"] = freeze_block(checks, args)
        payload["h3"] = h3_block(workers, reports, gates)
    else:
        payload["freezes"] = freeze_block(checks, args) if checks.first_manifest is not None else None
    return format_report(payload), payload


# --- CLI ------------------------------------------------------------------------------------------

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


def _label_path(text: str) -> tuple[str, str]:
    label, sep, path = text.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {text!r}")
    return label, path


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="The run report of an evaluation run from its logs and frozen records (no model call).",
                add_help=False)
    p.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (no report ran)")
    p.add_argument("--worker", action="append", type=_label_path, required=True, metavar="LABEL=PATH",
                   help="a worker log; repeat once per worker run")
    p.add_argument("--template", required=True, help="the template baseline log (the reference row)")
    p.add_argument("--packet-dir", default=None, help="the first review record: the frozen packet directory")
    p.add_argument("--manifest", default=None,
                   help="the first manifest (default docs/evidence/reviews/<run-id>-manifest.json in --repo)")
    p.add_argument("--freeze-commit", default=None, help="the first freeze commit, checked when given")
    p.add_argument("--second-look-dir", default=None, help="the second-look folder")
    p.add_argument("--second-look-manifest", default=None,
                   help="the second-look manifest (default docs/evidence/reviews/<run-id>-second-look-manifest.json)")
    p.add_argument("--second-look-commit", default=None, help="the second-look freeze commit, checked when given")
    p.add_argument("--blind-check-dir", default=None, help="the blinding-check folder (a full run has one)")
    p.add_argument("--first-archive", default=None, help="the archive URL of the first freeze commit's page")
    p.add_argument("--first-archive-date", default=None, help="the date of that archive")
    p.add_argument("--second-archive", default=None, help="the archive URL of the second-look commit's page")
    p.add_argument("--second-archive-date", default=None, help="the date of that archive")
    p.add_argument("--repo", default=None, help="the repository holding the manifests (default: this repository)")
    p.add_argument("--max-tokens-pin", type=int, default=None, help="the max_tokens pin, when one is set")
    p.add_argument("--json", dest="json_out", default=None, help="write the report as JSON to this path")
    return p


def _json_safe(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        workers = [LogInput(label, path, read_log(path, f"worker {label}"), "worker") for label, path in args.worker]
        template = LogInput("template", args.template, read_log(args.template, "template"), "template")
        text, payload = build_report(
            workers, template, packet_dir=args.packet_dir, manifest=args.manifest, freeze_commit=args.freeze_commit,
            second_look_dir=args.second_look_dir, second_look_manifest=args.second_look_manifest,
            second_look_commit=args.second_look_commit, blind_check_dir=args.blind_check_dir, repo=args.repo,
            max_tokens_pin=args.max_tokens_pin, args=args)
        if args.json_out:
            try:
                with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(_json_safe(payload))
            except OSError as e:
                raise Refused(f"cannot write {args.json_out}: {type(e).__name__}") from e
    except (Refused, GateError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as a report
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
