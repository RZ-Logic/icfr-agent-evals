"""The section 13 pilot row's quantities from saved logs. NO API key, no model call.

Usage:
  python scripts/pilot_report.py --sonnet S.eval --haiku H.eval --template T.eval
         [--reading-b i|ii|iii|all] [--regression-baseline B.eval --regression-candidate C.eval]
         [--json out.json] [--review-csv out.csv] [--max-tokens-pin N]
         [--review-record PACKET_DIR [--review-manifest MANIFEST] [--freeze-commit SHA]]

Reads the three pilot legs (pre-registration section 13, pilot row; 20 dev-split
cases, epochs = 4) and prints, per run: sample count and epochs; the content-pass^4
figures (below); FM counts by code with FM-002 by direction; review-queue
volume as distinct (sample, epoch, reason) and distinct (sample, epoch), split by
class (never the raw scorer-row count, six rows per format break); the Sample
Consistency summary with the H4 line labeled as a rehearsal (never on the template
log: a deterministic run agrees with itself by construction, so every wrong case
would read as FM-007). Then the template margin per anchor, difficulty check (a)
from the template log, difficulty check (b) from the Haiku log under all three
readings, the McNemar block floor and the paired-t minimal detectable drop
(inferred), and, given both regression logs, compare_runs.py's verdict on the pair.

Content-pass^4 figures (section 6, Content-pass^4; section 8, Primary outcomes and
Extractor health), each with its Clopper-Pearson 95% interval, from
icfr_agent_evals/trials.py figures():
  review-settled  the score of record (owner decision D-A);
  audit-adjusted  sensitivity only, never the score of record;
  route-blind     an upper bound on the review-settled figure, not a capability result;
  code-only       extractor health: every routed epoch fails; not a capability result
                  and never a score.
With --review-record (a frozen packet directory: the packet plus its frozen
verdicts.csv and fm_audit.csv, read through scripts/review_freeze.py
load_frozen_record), every run whose eval id the record's runs.json names gets all
four, FM-008 on the review-settled trial, and the per-anchor means review-settled and
audit-adjusted side by side. A run the record does not name, and every run when no
record is given, gets the route-blind and code-only figures (they read the log alone),
and the report says that the review-settled and audit-adjusted figures need a frozen
review record. The code-only FM-008 is printed as extractor health only. The McNemar
floor line and the regression line count the route-blind content-pass^4, the trial
section 10(a) reads (owner decision D-E5).

The record is checked against its freeze manifest before any figure is computed, through
scripts/review_freeze.py check_record_files (the check second-look-draw runs too). The
manifest must be this packet's review_freeze manifest and list every packet and record
file, and every file it lists must hash to it. The manifest is --review-manifest, by
default docs/evidence/reviews/<run-id>-manifest.json in this repo. The freeze commit is
checked too (freeze_commit_gaps): --freeze-commit, by default the one commit on HEAD's
history that added the manifest, must hold the manifest byte for byte, carry the
reviewer's trailer and be an ancestor of HEAD, the settlement commit (section 8; its
age is not checked here, only by the second look). The package-hash check of
pre-registration section 8 (Review record and freeze) is: the package code this report
runs (icfr_agent_evals in this tree, the settlement code) must hash the same, file by
file as git blob ids, as at the manifest's run commit (the commit that scored the logs);
any difference, or a run commit git cannot read here, refuses the record
(package_code_gaps; owner decision D-R9, choice 6).

Citation-pass^4 (owner decision D-R9, 2026-11-11): every run prints it beside the
content-pass^4 figures, computed by code from the log alone (trials.citation_omission_figures),
with its interval, the epoch and token counts, the wallet-label matcher's added tokens,
the epochs not assessable and, with a record, the review-found count. Every figure is also
printed under the rule before D-R9, and the fabrication and faithfulness means by subset
are printed under both rules (H1 and H3).

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

The review CSV (--review-csv) is a queue listing, not the review record. It holds one
row per distinct (run, sample, epoch, route key) in the RB-7 header
(review_schema.RB7_COLUMNS), with row_kind "finding" only for a settleable route key
(settlement.is_settleable) and blank for every other row, unit_id, sentence, verdict,
note, reviewer and date blank. Verdicts are given in the grouped review packet
(scripts/review_packet.py); the member-level review record is the output of
scripts/review_freeze.py freeze --expanded-csv.

Exit codes:
  0  report printed
  3  refused: a missing or unreadable file, an evaluation-split log, a log that is
     not the 20-case dev pilot at epochs = 4 (or a partial run), a log whose status
     is not success, a worker log off the section 5 run configuration (contract C2),
     runs whose Task version or dataset content differ, Sonnet and Haiku logs on
     another task or worker prompt, a template
     log that is not the template baseline (or a worker log that is), a regression
     pair compare_runs.py refuses, a help request, an output path that cannot be
     written, a review record that cannot be read, that has no freeze manifest or
     fails the check against it (every gap counted), that names none of the three
     pilot logs or that does not cover a run it names (trials.RecordError, every gap
     listed), or any other input the report cannot read
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import statistics
import subprocess
from functools import lru_cache
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from icfr_agent_evals import consistency as con
from icfr_agent_evals import reliability as rel
from icfr_agent_evals import review_schema
from icfr_agent_evals import roster
from icfr_agent_evals import trials
from icfr_agent_evals.settlement import is_settleable
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
REVIEW_CSV_COLUMNS: tuple[str, ...] = review_schema.RB7_COLUMNS
VERDICTS: tuple[str, ...] = review_schema.FINDING_VERDICTS
TEMPLATE_CONSISTENCY_LINE = "template: deterministic, agreement 1 by construction; Sample Consistency not reported"
COMPARE_SCRIPT = Path(__file__).resolve().parent / "compare_runs.py"
REVIEW_FREEZE_SCRIPT = Path(__file__).resolve().parent / "review_freeze.py"
REPO_ROOT = Path(__file__).resolve().parent.parent  # the default manifest is under its docs/evidence/reviews

# The four content-pass^4 figures (section 6 and section 8), in print order, each with
# its short name and the role it plays. Only the review-settled figure is the score of
# record; the code-only figure is extractor health and never a score.
FIGURE_ORDER: tuple[str, ...] = (trials.REVIEW_SETTLED, trials.AUDIT_ADJUSTED, trials.ROUTE_BLIND, trials.CODE_ONLY)
RECORD_FIGURES: tuple[str, ...] = (trials.REVIEW_SETTLED, trials.AUDIT_ADJUSTED)
FIGURE_NAMES: dict[str, str] = {
    trials.REVIEW_SETTLED: "review-settled", trials.AUDIT_ADJUSTED: "audit-adjusted",
    trials.ROUTE_BLIND: "route-blind", trials.CODE_ONLY: "code-only",
}
FIGURE_ROLES: dict[str, str] = {
    trials.REVIEW_SETTLED: "SCORE OF RECORD (owner decision D-A)",
    trials.AUDIT_ADJUSTED: "SENSITIVITY only, never the score of record",
    trials.ROUTE_BLIND: "UPPER BOUND on the review-settled figure, not a capability result",
    trials.CODE_ONLY: "EXTRACTOR HEALTH only: not a capability result, never a score",
}
NEEDS_RECORD = "needs a frozen review record (pass --review-record PACKET_DIR)"
MAX_GAPS_SHOWN = 10
PACKAGE_DIR = "icfr_agent_evals"  # the package code section 8's package-hash check compares
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


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


# --- The frozen review record --------------------------------------------------------------

@lru_cache(maxsize=1)
def load_review_freeze() -> Any:
    """scripts/review_freeze.py, for load_frozen_record and check_record_files only (under a
    private module name, so the module other callers load by path stays untouched)."""
    name = "review_freeze_for_pilot_report"
    spec = importlib.util.spec_from_file_location(name, REVIEW_FREEZE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module by name
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def load_review_record(packet_dir: str, manifest: str | None = None,
                       freeze_commit: str | None = None) -> tuple[Any, dict[str, str]]:
    """The frozen review record: the packet and its frozen verdicts.csv and fm_audit.csv,
    read through review_freeze.load_frozen_record, then checked against its freeze
    manifest through review_freeze.check_record_files (the manifest is this packet's, lists
    every packet and record file, and every file it lists hashes to it). The manifest is
    `manifest`, or by default docs/evidence/reviews/<run-id>-manifest.json in this repo.
    The freeze commit is checked too (freeze_commit_gaps). Returns (record, {"path",
    "sha256", "freeze_commit"} of the manifest). A record that cannot be read, has no
    manifest or fails a check is refused, every gap counted and the first ones listed."""
    if not Path(packet_dir).is_dir():
        raise Refused(f"review record not found: {packet_dir} is not a packet directory")
    rf = load_review_freeze()
    try:
        record = rf.load_frozen_record(str(packet_dir))
    except rf.Refused as e:
        raise Refused(f"review record {packet_dir} unreadable: {e}") from e
    try:
        man, mbytes, mpath, gaps = rf.check_record_files(str(packet_dir), repo=str(REPO_ROOT), manifest=manifest)
    except rf.Refused as e:
        raise Refused(f"review record {packet_dir} fails the frozen-record check: {e}") from e
    more_gaps, commit = freeze_commit_gaps(REPO_ROOT, Path(mpath), mbytes, man, freeze_commit)
    gaps = [*gaps, *more_gaps, *running_package_gap(REPO_ROOT),
            *package_code_gaps(REPO_ROOT, str(man.get("run_commit", "")))]
    if gaps:
        shown = "; ".join(gaps[:MAX_GAPS_SHOWN])
        more = f"; and {len(gaps) - MAX_GAPS_SHOWN} more" if len(gaps) > MAX_GAPS_SHOWN else ""
        raise Refused(f"review record {packet_dir} fails the frozen-record check against {mpath} "
                      f"({len(gaps)} gap(s)): {shown}{more}")
    return record, {"path": str(mpath), "sha256": hashlib.sha256(mbytes).hexdigest(), "freeze_commit": commit}


def freeze_commit_gaps(repo: Path, mpath: Path, mbytes: bytes, man: dict,
                       freeze_commit: str | None) -> tuple[list[str], str]:
    """Pre-registration section 8 (Review record and freeze): the settlement script refuses
    a record unless its manifest was frozen in a commit that is an ancestor of the
    settlement commit. The settlement commit is HEAD of this repository (the commit whose
    code applies the verdicts; package_code_gaps checks that code against the run commit).
    The freeze commit is `freeze_commit`, or by default the one commit on HEAD's history
    that added the manifest file. That commit must hold the manifest byte for byte and carry
    the reviewer's trailer in its closing trailer block (review_freeze.check_freeze_commit,
    with no minimum age: settlement may follow the freeze at once), and `git merge-base
    --is-ancestor` must place it on HEAD's history (HEAD itself counts). Returns (gaps, the
    freeze commit's full hash or "" when none was found). Runs git log, git show and git
    merge-base, and nothing else."""
    rf = load_review_freeze()
    try:
        rel = mpath.resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return [f"the manifest {mpath} is outside this repository, so its freeze commit cannot be checked"], ""
    sha = str(freeze_commit or "")
    if not sha:
        added = _git(repo, "log", "--diff-filter=A", "--format=%H", "HEAD", "--", rel)
        found = added.stdout.decode("utf-8", errors="replace").split() if added.returncode == 0 else []
        if not found:
            return [f"the manifest {rel} is in no commit on HEAD's history: it is not frozen; commit it with "
                    f"the reviewer's trailer before settling"], ""
        if len(found) > 1:
            return [f"the manifest {rel} was added in {len(found)} commits on HEAD's history; pass "
                    f"--freeze-commit"], ""
        sha = found[0]
    if not _COMMIT_RE.match(sha):
        return [f"--freeze-commit {sha!r} is not a hex commit hash"], ""
    resolved = _git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    if resolved.returncode != 0:
        return [f"freeze commit {sha} cannot be read in {repo}"], ""
    full = resolved.stdout.decode("utf-8", errors="replace").strip()
    gaps, _when = rf.check_freeze_commit(mbytes, man, repo=str(repo), sha=full, rel_path=rel,
                                         now=datetime.now(timezone.utc), min_gap=None)
    ancestor = _git(repo, "merge-base", "--is-ancestor", full, "HEAD")
    if ancestor.returncode == 1:
        gaps.append(f"freeze commit {full[:12]} is not an ancestor of the settlement commit (HEAD)")
    elif ancestor.returncode != 0:
        gaps.append(f"git cannot tell whether freeze commit {full[:12]} is an ancestor of HEAD")
    return gaps, full


def _git(repo: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], input=None if stdin is None else stdin.encode("utf-8"),
                          capture_output=True, check=False, timeout=120)


def running_package_gap(repo: Path) -> list[str]:
    """The package-hash check hashes repo/icfr_agent_evals; it is only meaningful when
    that is the package this process imported (verifier finding, 2026-11-11: without
    PYTHONPATH the venv's editable install of another checkout runs instead)."""
    import icfr_agent_evals as running
    ran = Path(running.__file__).resolve().parent
    root = (Path(repo) / PACKAGE_DIR).resolve()
    if ran != root:
        return [f"the package that runs is {ran}, not the repository's {root}; "
                f"set PYTHONPATH to the repository so the hashed code is the code that settles"]
    return []


def package_code_gaps(repo: Path, run_commit: str) -> list[str]:
    """Pre-registration section 8 (Review record and freeze): the package code at the
    settlement commit must hash the same as at the run commit. The settlement code is
    the package this report runs, the working tree's icfr_agent_evals; the run commit is
    the manifest's run_commit. Each file's git blob id is compared: `git ls-tree` at the
    run commit, `git hash-object` (with the repository's own filters) on the working
    tree, so a line-ending checkout difference is no difference. Every file missing on
    one side or with another blob id is a gap, and so is a run commit git cannot read.
    Runs git ls-tree and git hash-object and nothing else (owner decision D-R9, choice 6)."""
    commit = str(run_commit or "")
    if not _COMMIT_RE.match(commit):
        return [f"the manifest's run commit {commit!r} is not a hex commit hash"]
    listed = _git(repo, "ls-tree", "-r", "--full-tree", f"{commit}^{{commit}}", "--", PACKAGE_DIR)
    if listed.returncode != 0:
        return [f"run commit {commit} cannot be read in {repo}; the package-hash check needs it"]
    at_run: dict[str, str] = {}
    for line in listed.stdout.decode("utf-8", errors="replace").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            at_run[path] = parts[2]
    root = Path(repo) / PACKAGE_DIR
    paths = sorted(f.relative_to(repo).as_posix() for f in root.rglob("*")
                   if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc")
    here: dict[str, str] = {}
    if paths:
        hashed = _git(repo, "hash-object", "--stdin-paths", stdin="\n".join(paths) + "\n")
        blobs = hashed.stdout.decode("ascii", errors="replace").split()
        if hashed.returncode != 0 or len(blobs) != len(paths):
            return [f"git hash-object could not hash the package files in {repo}"]
        here = dict(zip(paths, blobs))
    gaps = []
    for path in sorted(set(at_run) | set(here)):
        if path not in here:
            gaps.append(f"package file {path} is at the run commit {commit} but not in the settlement code")
        elif path not in at_run:
            gaps.append(f"package file {path} is in the settlement code but not at the run commit {commit}")
        elif here[path] != at_run[path]:
            gaps.append(f"package file {path} differs between the run commit {commit} and the settlement code")
    return gaps


def record_run_id(record: Any, log: Any) -> str | None:
    """The record's run id (R1, ...) whose eval id is this log's, or None when the record
    names no run of this log."""
    eval_id = str(log.eval.eval_id)
    hits = sorted(rid for rid, info in dict(getattr(record, "runs", None) or {}).items()
                  if str((info or {}).get("eval_id")) == eval_id)
    if len(hits) > 1:
        raise Refused(f"the review record names {len(hits)} runs with eval id {eval_id}: {hits}")
    return hits[0] if hits else None


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
    """One RB-7 row per distinct (run, sample, epoch, CSV route key) from reliability.queue_items.

    This is a queue listing, not the review record: verdicts are given in the grouped
    review packet (scripts/review_packet.py) and the member-level record is
    scripts/review_freeze.py --expanded-csv. row_kind is "finding" only for a route key
    settlement.is_settleable accepts; it is blank for an unsettleable route (not a finding;
    it fails on code) and for a format_break, fm_listed or sample_error row (neither a
    finding nor a per-(epoch, FM code) audit row, which exists only in the packet).
    route_class is the queue class; unit_id, sentence, verdict, note, reviewer and date
    are blank."""
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in rel.queue_items(report):
        g = grouped.setdefault((item.sample_id, item.epoch, csv_route_key(item)),
                               {"scorers": [], "reasons": [], "class": item.queue_class,
                                "settleable_key": item.route_key is not None})
        for scorer in item.scorers:
            if scorer not in g["scorers"]:
                g["scorers"].append(scorer)
        if item.reason not in g["reasons"]:
            g["reasons"].append(item.reason)
    rows: list[dict[str, str]] = []
    for (sid, ep, key), g in sorted(grouped.items()):
        finding = g["settleable_key"] and is_settleable(key, tuple(g["reasons"]))
        row = {col: "" for col in REVIEW_CSV_COLUMNS}
        row.update({"row_kind": "finding" if finding else "", "run_label": run_label, "log_eval_id": eval_id,
                    "sample_id": sid, "epoch": str(ep), "scorers": ";".join(sorted(g["scorers"])),
                    "route_key": key, "route_class": g["class"],
                    "reason": review_schema.RB7_LIST_SEP.join(g["reasons"])})
        rows.append(row)
    return rows


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


def generation_stops(log: Any) -> dict[str, Any]:
    """Stop reasons and output tokens of the evaluated model's last generation per sample-epoch.

    A generation that stops on max_tokens (thinking plus answer at the 48,000 default, or
    the D6 pin) is cut off: the strict parser scores it FM-000, so the count tells a
    ceiling cut-off apart from a format break the model produced. Output tokens come
    from the logged usage of the same event, thinking included where the provider bills it.
    """
    reasons: dict[str, int] = {}
    cut_and_format_break = 0
    tokens: list[int] = []
    for s in log.samples:
        last = None
        for ev in s.events or ():
            if getattr(ev, "event", None) == "model" and getattr(ev, "role", None) is None:
                last = ev
        if last is None or last.output is None:
            continue
        reason = str(last.output.stop_reason or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        usage = last.output.usage
        if usage is not None and usage.output_tokens is not None:
            tokens.append(int(usage.output_tokens))
        if reason == "max_tokens" and getattr(s, "error", None) is None and any(
                (v.metadata or {}).get("format_break") for v in (s.scores or {}).values()):
            cut_and_format_break += 1
    return {
        "stop_reasons": dict(sorted(reasons.items())),
        "max_tokens_stops": reasons.get("max_tokens", 0),
        "max_tokens_stops_format_break": cut_and_format_break,
        "output_tokens": {
            "n": len(tokens),
            "max": max(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "total": sum(tokens),
        },
    }


def format_generation_stops(stops: dict[str, Any], n: int) -> list[str]:
    other = ", ".join(f"{k} {v}" for k, v in stops["stop_reasons"].items() if k != "max_tokens") or "none"
    tok = stops["output_tokens"]
    tok_line = ("output tokens per generation: none logged" if tok["n"] == 0 else
                f"output tokens per generation (thinking included where billed): max {tok['max']:,}, "
                f"median {tok['median']:,.0f}, total {tok['total']:,} over {tok['n']} generations")
    return [
        f"generation stops: max_tokens {stops['max_tokens_stops']}/{n} "
        f"(format breaks among them {stops['max_tokens_stops_format_break']}); other stop reasons: {other}",
        tok_line,
    ]


def content_figures(log: Any, record: Any = None, run_id: str | None = None) -> dict[str, Any]:
    """trials.figures() on one run: all four figures with a record that names the run,
    route-blind and code-only without one. A record that does not cover the run is
    refused, every gap counted and the first ones listed."""
    try:
        if record is None:
            return trials.figures(log)
        return trials.figures(log, record, run_id=run_id)
    except trials.RecordError as e:
        shown = "; ".join(e.gaps[:MAX_GAPS_SHOWN])
        more = f"; and {len(e.gaps) - MAX_GAPS_SHOWN} more" if len(e.gaps) > MAX_GAPS_SHOWN else ""
        raise Refused(f"the frozen review record does not cover run {run_id} ({len(e.gaps)} gap(s)): "
                      f"{shown}{more}") from e


def _figure_line(kind: str, row: Mapping[str, Any] | None) -> str:
    name = f"{FIGURE_NAMES[kind]} content-pass^4"
    if row is None:
        return f"  {name:<30} not computed: {NEEDS_RECORD}"
    lo, hi = row["ci95"]
    return (f"  {name:<30} {row['passed']}/{row['n']} = {row['rate']:.3f} "
            f"(Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}])  {FIGURE_ROLES[kind]}")


def _mean_text(m: Mapping[str, Any]) -> str:
    if m.get("mean") is None:
        return "none"
    if m.get("ci95_low") is None:
        return f"{m['mean']:.3f}"
    return f"{m['mean']:.3f} [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}]"


def figures_block(res: Mapping[str, Any], *, record_dir: str | None = None, run_id: str | None = None,
                  why_no_record: str = "") -> tuple[list[str], dict[str, Any]]:
    """Printed lines and JSON payload of one run's content-pass^4 figures (section 6 and 8).

    The review-settled figure is the only one labeled the score of record; the code-only
    figure is labeled extractor health and never a score. Without a record that names the
    run the two record figures print as not computed, never as a number."""
    rows = res["figures"]
    has_record = trials.REVIEW_SETTLED in rows
    source = (f"frozen review record {record_dir}, run {run_id}" if has_record
              else f"no frozen review record{why_no_record}")
    lines = [f"Content-pass^4 figures (section 6, section 8; {source}):"]
    lines.extend(_figure_line(kind, rows.get(kind)) for kind in FIGURE_ORDER)
    if not has_record:
        lines.append("  review-settled and audit-adjusted content-pass^4, FM-008 on the review-settled trial and the "
                     f"per-anchor means review-settled and audit-adjusted: not computed; each {NEEDS_RECORD}")
    shown = [k for k in FIGURE_ORDER if k in rows]
    per = rows[shown[0]]["per_sample"]
    lines.append("  per sample (" + " / ".join(FIGURE_NAMES[k] for k in shown) + "): "
                 + " ".join(f"{sid}:{'/'.join(str(rows[k]['per_sample'][sid]) for k in shown)}" for sid in per))
    code_fm008 = rows[trials.CODE_ONLY]["fm008"]
    if has_record:
        fm008 = list(res["fm008"])
        lines.append(f"  FM-008 on the review-settled trial: {len(fm008)}"
                     + (f" ({', '.join(fm008)})" if fm008 else ""))
        lines.append("  per-anchor means (section 8), review-settled beside audit-adjusted "
                     "(mean [normal 95% CI on the SE of per-sample epoch means]):")
        for anchor, both in res["anchor_means"].items():
            lines.append(f"    {anchor:<26} review-settled {_mean_text(both[trials.REVIEW_SETTLED])}, "
                         f"audit-adjusted {_mean_text(both[trials.AUDIT_ADJUSTED])}")
        found = res["review_found"]
        codes = ", ".join(f"{c} {n}" for c, n in found["codes"].items()) or "none"
        lines.append(f"  review-found (counted apart from code-found): codes {codes}; citation omissions "
                     f"{found['citation_omissions']}; whole-narrative wrong {found['whole_narrative_wrong']}")
    lines.append(f"  FM-008 on the code-only trial (extractor health only, not the FM-008 count): {len(code_fm008)}")
    lines.extend(citation_lines(res))
    payload: dict[str, Any] = {
        "record": {"packet_dir": record_dir, "run_id": run_id} if has_record else None,
        "score_of_record": trials.REVIEW_SETTLED if has_record else None,
        "figures": {kind: ({**rows[kind], "name": FIGURE_NAMES[kind], "role": FIGURE_ROLES[kind]} if kind in rows
                           else {"computed": False, "name": FIGURE_NAMES[kind], "role": FIGURE_ROLES[kind],
                                 "note": NEEDS_RECORD})
                    for kind in FIGURE_ORDER},
        "fm008": ({"trial": trials.REVIEW_SETTLED, "samples": list(res["fm008"])} if has_record
                  else {"trial": None, "samples": None, "note": NEEDS_RECORD}),
        "fm008_code_only_extractor_health": list(code_fm008),
        "anchor_means": res.get("anchor_means"),
        "review_found": res.get("review_found"),
        "citation_pass4": res.get("citation_pass4"),
        "content_pass4_before_d_r9": res.get("before_d_r9"),
        "h1_h3_means": res.get("h1_h3_means"),
    }
    return lines, payload


def citation_lines(res: Mapping[str, Any]) -> list[str]:
    """Citation-pass^4 beside the content figures, every figure under the rule before
    D-R9, and the fabrication and faithfulness means by subset under both rules (owner
    decision D-R9, 2026-11-11)."""
    lines: list[str] = []
    cite = res.get("citation_pass4")
    if cite is not None:
        lo, hi = cite["ci95"]
        found = ("needs a frozen review record" if cite["review_found"] is None else str(cite["review_found"]))
        subsets = ", ".join(f"{name} {v['passed']}/{v['n']}" for name, v in cite["per_subset"].items())
        lines.append(f"  {cite['label']} {cite['passed']}/{cite['n']} = {cite['rate']:.3f} "
                     f"(Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}]); epochs with an omission "
                     f"{cite['epochs_with_omission']} ({cite['omitted_tokens']} tokens, of them wallet-label matcher "
                     f"{cite['wallet_label_tokens']}), not assessable {cite['epochs_not_assessable']}, review-found "
                     f"{found}; {subsets}")
    for kind, old in (res.get("before_d_r9") or {}).items():
        lo, hi = old["ci95"]
        lines.append(f"  {old['label']}, {FIGURE_NAMES[kind]}: {old['passed']}/{old['n']} "
                     f"(Clopper-Pearson 95% [{lo:.3f}, {hi:.3f}])")
    for kind, both in (res.get("h1_h3_means") or {}).items():
        for rule, label in (("d_r9", "D-R9"), ("before_d_r9", "rule before D-R9")):
            parts = []
            for subset in ("overall", "synthetic", "difficulty_bearing"):
                m = both[rule].get(subset) or {}
                if m.get("fabrication_anchor") is None:
                    continue
                parts.append(f"{subset} fabrication {m['fabrication_anchor']:.3f}, "
                             f"faithfulness {m['faithfulness_anchor']:.3f}")
            lines.append(f"  means, {FIGURE_NAMES[kind]}, {label} (H1, H3): " + "; ".join(parts))
    return lines


def run_block(role: str, log: Any, path: str = "", record: Any = None, run_id: str | None = None,
              record_dir: str | None = None) -> tuple[list[str], dict[str, Any], list[dict[str, str]]]:
    """Printed lines, JSON payload and review rows for one run. With a record that names
    this run (run_id), all four content-pass^4 figures; otherwise route-blind and code-only."""
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
    named = record is not None and run_id is not None
    why = "" if record is None or named else f" for this run: the record names no run with eval id {e.eval_id}"
    fig_lines, figs = figures_block(content_figures(log, record if named else None, run_id if named else None),
                                    record_dir=record_dir if named else None, run_id=run_id if named else None,
                                    why_no_record=why)
    lines.extend(fig_lines)
    lines.append(rel.format_fm_counts(r))
    errored = sum(1 for s in log.samples if getattr(s, "error", None) is not None)
    fb = sum(1 for s in log.samples if getattr(s, "error", None) is None
             and any((v.metadata or {}).get("format_break") for v in (s.scores or {}).values()))
    lines.append(f"format-break epochs {fb}/{len(log.samples)}; errored epochs {errored}/{len(log.samples)}")
    stops = generation_stops(log)
    lines.extend(format_generation_stops(stops, len(log.samples)))
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
        "content_pass4_figures": figs,
        "fm_counts": r.fm_counts, "fm002_by_direction": r.fm002_by_direction,
        "format_break_epochs": fb, "errored_epochs": errored, "generation_stops": stops,
        "queue_volume": vol, "sample_consistency": cons,
    }
    return lines, payload, review_rows(r, role, str(e.eval_id))


def floors_block(sonnet: Any) -> tuple[list[str], dict[str, Any]]:
    """McNemar block floor under Holm (m = 6) and the inferred paired-t drop at n = 20."""
    p0, p1 = mcnemar_block_floor(), mcnemar_block_floor(reversals=1)
    # Section 10(a) reads the route-blind trial (owner decision D-E5), never the code-only one.
    passed = trials.route_blind_trial(sonnet).passed
    lines = ["== Regression power floors (section 10(a), Holm m = 6, rank 1 threshold "
             f"{FAMILY_ALPHA / HOLM_M:.5f})",
             f"McNemar block floor on content-pass^4: {p0} one-way discordant cases with no reversal, "
             f"{p1} with one reversal",
             f"Sonnet log passes the route-blind content-pass^4 (the trial section 10(a) reads, D-E5) on {passed} "
             f"of {PILOT_SAMPLES}; worse cannot exceed it: "
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
                   "sonnet_pass4_trial": trials.ROUTE_BLIND,
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
    lines.append(f"baseline passes the route-blind content-pass^4 (the trial section 10(a) reads, D-E5) on "
                 f"{base_pass} of {PILOT_SAMPLES} against the McNemar floor "
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
    if args.review_manifest is not None and args.review_record is None:
        raise Refused("--review-manifest needs --review-record")
    if args.freeze_commit is not None and args.review_record is None:
        raise Refused("--freeze-commit needs --review-record")
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
    record_dir = args.review_record
    record, manifest = (load_review_record(record_dir, args.review_manifest, args.freeze_commit) if record_dir is not None
                        else (None, None))
    run_ids = {role: record_run_id(record, log) if record is not None else None for role, log in logs.items()}
    if record is not None and not any(run_ids.values()):
        raise Refused(f"the review record {record_dir} names none of the sonnet, haiku or template logs "
                      f"(its runs.json eval ids: {sorted(str((i or {}).get('eval_id')) for i in record.runs.values())})")
    if record is not None:
        lines.append(f"Frozen review record {record_dir}: every file it lists hashes to its manifest "
                     f"{manifest['path']} (sha256 {manifest['sha256']}), frozen in commit "
                     f"{manifest['freeze_commit'][:12]}, an ancestor of HEAD")
    payload["review_record"] = ({"packet_dir": record_dir, "run_ids": run_ids, "manifest": manifest}
                                if record is not None else None)
    for role, log in logs.items():
        block, data, run_rows = run_block(role, log, paths[role], record=record, run_id=run_ids[role],
                                          record_dir=record_dir)
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
    p.add_argument("--review-csv", help="write the review queue listing (RB-7 header, verdicts blank; not the "
                                        "review record) to this path")
    p.add_argument("--max-tokens-pin", type=int, default=None,
                   help="the owner's max_tokens pin, when one is set (default: no --max-tokens flag; the evaluated "
                             "model's events may carry Inspect's documented default 48,000)")
    p.add_argument("--review-record", default=None, metavar="PACKET_DIR",
                   help="the frozen review record: a packet directory holding its frozen verdicts.csv and "
                        "fm_audit.csv (scripts/review_freeze.py load_frozen_record). With it, every run the record "
                        "names gets the review-settled figure (the score of record) and the audit-adjusted figure; "
                        "without it only the route-blind and code-only figures print. The record is refused "
                        "unless every file its freeze manifest lists hashes to it")
    p.add_argument("--freeze-commit", default=None, metavar="SHA",
                   help="the commit that froze the record's manifest (default: the one commit on HEAD's history "
                        "that added the manifest file); it must hold the manifest, carry the reviewer's trailer and "
                        "be an ancestor of HEAD; needs --review-record")
    p.add_argument("--review-manifest", default=None, metavar="MANIFEST",
                   help="the record's freeze manifest (default: docs/evidence/reviews/<run-id>-manifest.json in "
                        "this repo); needs --review-record")
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
