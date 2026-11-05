"""Scan jobs for the vendored Scout scanners (pre-registration sections 5 and 9).

Two jobs, as section 9 splits them:
  - worker_awareness_scan: eval_awareness on worker transcripts;
  - all_logs_scan: refusal_classifier and command_not_found on any EvalLog.

Each job names the scanner model in its ScanJob (roster.SCOUT_SCANNER, section 5),
and each llm scanner also names it, so a scan-level --model cannot rebind it. The
only other accepted model is mockllm, for the offline tests. In inspect-scout
0.5.5, ScanJob resolves its model when it is built, so building a job on the pin
needs the OpenAI provider configured; building one on mockllm needs nothing.

Split guard: the scanners are exploratory and never run on the evaluation split
before the pre-registration tag (section 4, evaluation-split generation). Before
a job is built, every log it would scan has its header read (header_only, no
sample is read) and its split taken from the Task metadata the task records
(eval.metadata.split, checked against eval.task_args.split when present). A log
on the evaluation split is refused unless allow_evaluation_split=True, which is
for post-tag runs only. A log whose split is missing or unknown is refused too,
so a log from outside this harness never passes by default. The job scans exactly
the files that were checked, never the directory they were listed from.

Limits of the guards: in inspect-scout 0.5.5, a transcripts argument given to
scan() (or -T on the CLI) replaces the job's transcripts after the job is built,
so it bypasses the split guard. Pass logs to the job only, never transcripts to scan().
Run the scanners only through these scan jobs: running a scanner file directly
(scout scan <scanner>.py, inspect eval --scanner, INSPECT_EVAL_SCANNER) builds no
scan job and skips both the split guard and the serving guard. Never pass
--metadata or scan(metadata=...): either replaces the job metadata, the endpoint
record included.

Serving guard: once a job on the scanner pin is built, the scanner model it
resolved goes through roster.model_serving (via _model.scanner_serving). The job
is refused unless that model's client endpoint is exactly the OpenAI first-party
API, with no explicit base URL, no model_args and no scan model base URL
(SCOUT_SCAN_MODEL_BASE_URL or INSPECT_EVAL_SCAN_MODEL_BASE_URL). The checked
endpoint is recorded in the job metadata as scanner_endpoint (None on mockllm).
The check reads the environment when the job is built; the llm scanners resolve
the same pinned model by name from that environment when they run. Scout's scan()
loads the project .env only after the job is built, and its worker processes
inherit what it loaded. The scan jobs therefore load the project .env the way
Scout does before the check, so the check sees the environment the scan and its
workers use (python-dotenv, find_dotenv(usecwd=True); PYTHON_DOTENV_DISABLED
switches it off, as it does for Scout).

CLI: scout scan icfr_agent_evals/scanners/jobs.py@all_logs_scan -S logs=<path>
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import find_dotenv, load_dotenv
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_scout import ScanJob, scanjob, transcripts_from

# Absolute imports: the scout CLI may load this file by path.
from icfr_agent_evals.casefile import SPLITS
from icfr_agent_evals.roster import SCOUT_SCANNER, WORKER
from icfr_agent_evals.scanners._model import ScannerModelError, is_offline, scanner_model, scanner_serving
from icfr_agent_evals.scanners._upstream import BACKPORT, UPSTREAM_COMMIT
from icfr_agent_evals.scanners.command_not_found import command_not_found
from icfr_agent_evals.scanners.eval_awareness import eval_awareness
from icfr_agent_evals.scanners.refusal_classifier import refusal_classifier

EVALUATION_SPLIT = "eval"


class ScanSplitError(RuntimeError):
    """A scan that would read an evaluation-split log, or a log whose split is unknown."""


def split_of(eval_metadata: dict[str, Any] | None, task_args: dict[str, Any] | None) -> str:
    """The split a log was run on, from its header. Raise ScanSplitError when the
    Task metadata records none, records an unknown one, or disagrees with task_args."""
    split = (eval_metadata or {}).get("split")
    if not isinstance(split, str) or split not in SPLITS:
        raise ScanSplitError(f"log header records no known split (eval.metadata.split={split!r}); refused")
    arg = (task_args or {}).get("split")
    if arg is not None and arg != split:
        raise ScanSplitError(f"log header split {split!r} disagrees with task_args split {arg!r}; refused")
    return split


def check_split(split: str, allow_evaluation_split: bool = False) -> None:
    if split == EVALUATION_SPLIT and allow_evaluation_split is not True:
        raise ScanSplitError(
            "refusing to scan an evaluation-split log: Scout scanners never run on the "
            "evaluation split before the pre-registration tag (allow_evaluation_split=True "
            "is for post-tag runs only)"
        )


def check_worker(evaluated_model: str) -> None:
    """The awareness scan reads worker transcripts only (section 9); mockllm logs
    pass, as in roster.validate_roster's offline mode."""
    if evaluated_model != WORKER and not is_offline(evaluated_model):
        raise ScanSplitError(
            f"log evaluates {evaluated_model!r}; the awareness scan runs on worker "
            f"transcripts ({WORKER!r}) only"
        )


def _log_files(logs: str | PathLike[str] | Sequence[str | PathLike[str]]) -> list[str]:
    locations: Iterable[str | PathLike[str]] = [logs] if isinstance(logs, (str, PathLike)) else logs
    files: list[str] = []
    for loc in locations:
        p = Path(loc)
        if p.is_dir():
            found = [info.name for info in list_eval_logs(p.as_posix(), recursive=True)]
            if not found:
                raise ScanSplitError(f"no eval logs under {p.as_posix()!r}; refused")
            files.extend(found)
        elif p.is_file():
            files.append(p.as_posix())
        else:
            raise ScanSplitError(f"log location {str(loc)!r} is not a local file or directory; refused")
    if not files:
        raise ScanSplitError("no logs given; refused")
    return files


def checked_logs(
    logs: str | PathLike[str] | Sequence[str | PathLike[str]],
    allow_evaluation_split: bool = False,
    worker_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Read every log header and apply the split guard (and, for the awareness
    scan, the worker check). Returns (log files, distinct splits)."""
    files = _log_files(logs)
    splits: set[str] = set()
    for f in files:
        header = read_eval_log(f, header_only=True)
        split = split_of(header.eval.metadata, header.eval.task_args)
        check_split(split, allow_evaluation_split)
        if worker_only:
            check_worker(header.eval.model)
        splits.add(split)
    return files, sorted(splits)


def load_scout_environment() -> None:
    """Load the project .env as inspect_scout 0.5.5 init_environment does. Scout's own
    loader runs once per process behind a flag, so it is not called here: this call
    runs on every job build. load_dotenv never overrides a variable already set."""
    load_dotenv(find_dotenv(usecwd=True))


def _job(files: list[str], splits: list[str], scanners: dict, model: str, name: str,
         allow_evaluation_split: bool, scans: str | None) -> ScanJob:
    metadata: dict[str, Any] = {
        "scanner_model": model,
        "splits": splits,
        "allow_evaluation_split": allow_evaluation_split,
        "scout_examples_commit": UPSTREAM_COMMIT,
        # the awareness rubric is not the pure 0.5.5 text: PR #679 is backported
        "scout_awareness_backport": BACKPORT.merge_commit,
        "scanner_endpoint": None,
    }
    load_scout_environment()
    job = ScanJob(
        transcripts=transcripts_from(files),
        scanners=scanners,
        model=model,
        scans=scans,
        name=name,
        metadata=metadata,
    )
    # ScanJob keeps the metadata dict it was given, so the record lands in the job.
    metadata["scanner_endpoint"] = scanner_job_serving(model, job)
    return job


def scanner_job_serving(model: str, job: Any) -> dict[str, str] | None:
    """The serving guard on a built job: None for a mockllm scanner, else the checked
    endpoint. A job whose model cannot be read fails closed."""
    if is_offline(model):
        return None
    try:
        resolved, base_url = job.model, job.model_base_url
    except AttributeError as e:
        raise ScannerModelError(f"serving guard: cannot read the scan job's model ({e}); refused") from e
    return scanner_serving(model, resolved, base_url)


@scanjob
def worker_awareness_scan(
    logs: str | list[str],
    model: str = SCOUT_SCANNER,
    allow_evaluation_split: bool = False,
    scans: str | None = None,
) -> ScanJob:
    """eval_awareness on worker transcripts (section 9). Exploratory."""
    model = scanner_model(model)
    files, splits = checked_logs(logs, allow_evaluation_split, worker_only=True)
    return _job(files, splits, {"eval_awareness": eval_awareness(model=model)}, model,
                "worker_awareness_scan", allow_evaluation_split, scans)


@scanjob
def all_logs_scan(
    logs: str | list[str],
    model: str = SCOUT_SCANNER,
    allow_evaluation_split: bool = False,
    scans: str | None = None,
) -> ScanJob:
    """refusal_classifier and command_not_found on EvalLogs (section 9). Exploratory.
    command_not_found is a regex scanner and uses no model."""
    model = scanner_model(model)
    files, splits = checked_logs(logs, allow_evaluation_split)
    scanners = {
        "refusal_classifier": refusal_classifier(model=model),
        "command_not_found": command_not_found(),
    }
    return _job(files, splits, scanners, model, "all_logs_scan", allow_evaluation_split, scans)
