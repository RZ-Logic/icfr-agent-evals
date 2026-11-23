"""The full-run grading job: the grader of record and the calibrator on the evaluation run.

Decision D-R23 (2026-11-22, amended 2026-11-23) records the rules of the grading job as
named items. This script runs that job from scripts, not as a package job, so
icfr_agent_evals stays as it was at the worker run commit (the package-hash rule).

What it does:
  - It admits the four worker logs of the evaluation run read-only, one per evaluated
    model, through grading_common.admit_worker_logs (status, split and unlock, the 200
    cases, whole run at 4 epochs, run config, reasoning mode, worker prompt, header
    grader step 1, revision commit, dataset hash, every sample's input and the parse
    rule). A worker log is never written, re-scored or appended to.
  - It builds the plan (grading_common.build_plan): one unit per generation whose worker
    output parsed and did not error, all four worker runs, Opus 5.5 included. Unparsed
    and errored epochs are counted, never graded; an errored epoch whose completion
    parses is still skipped (D-R23, units). Units get opaque ids in one pooled order fixed
    by a rule seed and are split into four grader parts by position (D-R23, grading
    order). The calibrator sample is drawn by case rotation over the three other worker
    runs, one gradeable epoch per (model, case), no refill, before any grader output
    exists (D-R23, draw).
  - Before the first call it runs the start checks (clean tree, HEAD an ancestor of the
    local origin/main ref, package unchanged since the run commit, the step 2 look file
    reads PASS, the archive record and any worker-log and plan-file hashes it lists),
    requires a passed calibrator plumbing check at this code (D-R23, plumbing check),
    checks the folders (D-R23, sealed folder), takes the ledger lock, writes the three
    plan files to the sealed folder and creates the attempt ledger under the git common
    directory with the provenance, the seeds, the seed rule and the exact sampling calls.
  - It grades each part through Inspect evals with the evaluated model none/none and one
    role bound: the grader of record (GPT-6.1 Terra, ladder step 2, OpenAI Batch,
    reasoning effort high, cache off) through grader.grade, then the calibrator (Opus
    5.5, effort high, cache off, not batched, no fallback, no tools) through
    calibrator_call.calibrate. The prompt is gp-1 with default settings (look pass_1)
    for both. Until a grader attempt log ends in success, the first 12 units of grader part 1
    run first as their own small eval, a rehearsal. Every Batch id is saved to the ledger the moment Batch
    accepts it, and Batch failure lines are recorded as failures.
  - Every attempt log is checked and its SHA-256 recorded. A log that ends in success
    is checked in full. A log with any other status is admitted by the partial check:
    its completed calls are kept and the rest count as missing. A log that fails its
    check is recorded in the ledger as rejected and stops the job; its replies are never
    used (D-R23, rejected logs).
  - The reply of record of a unit, for each judge, is its first completed call in any
    attempt log, parsed or not (D-R23, reply of record). A reply that does not parse, and
    a refusal, are never asked again. --resume re-checks every recorded attempt log,
    admits a log of this job that the ledger missed, and asks only units with no reply
    of record. Outputs of a dead Batch that no log recorded count as missing.
  - A unit asked in two attempt logs with no completed call is next asked in an isolated
    eval that holds only such units and runs one sample at a time. A unit asked in three
    attempt logs with no completed call, one of them isolated, is a failed judgment with
    no reply: never asked again, queued, counted, and treated as a reply that does not
    parse (D-R23, failed judgments). A unit counts as asked in a log only when the log
    holds its sample with a judge call; an eval cut off before its samples reached the
    judge asks nothing.
  - An outage of the whole account (a budget cap, a spend limit, a key error, the provider
    down) fails every call. When an eval ends with units left and some of them are now due
    for their final ask, the job warns that the next resume asks them one at a time and
    records any with no reply as failed judgments. Before an isolated eval, a final ask
    guard stops the job with no call when more than 12 units of the part are due, or when
    the role's last attempt log that sent a unit completed no call and sent more units than
    are now due. A resume with --allow-final-ask runs the isolated eval past the guard; the
    author passes it only after finding the cause and logging a deviation.
  - A resolved model string outside the pin stops the job, and a resume refuses while that
    stop is recorded. A resume checks the resolved strings of every re-admitted attempt log
    before any new eval.
  - When every unit has a reply of record it writes the sealed results (every reply,
    the queue with empty disposition fields and the aggregates: the reliability gate on
    complete pairs with the worst-case figure and break-even counts reported beside it,
    the case-cluster bootstrap limit, the circuit breaker, the claims, the means and the
    paired differences), marks the ledger completed, then writes the committed evidence
    file docs/evidence/grading/grading-2026-11-21.json with provenance, rules, hashes and
    pooled counts only.

Blinding (D-R23, sealed folder). Until the second-look freeze, stdout and the evidence
file hold counts pooled over all logs together: no worker model id beside a count, unit
or score, no per-model figure, no score, no tag count and no queue item. Everything else
goes to the sealed folder outside the repository, with its SHA-256 recorded. No eval
name, task argument, metadata field, sample id or ledger key names a worker model. The
attempt logs sit in a folder inside the sealed folder. The attempt logs, the ledger and
the provider's Batch files hold per-record content and are never opened before the
freeze.

Run:
  python scripts/grading_job.py                    (plan only: no call, no file written)
  python scripts/calibrator_plumbing_check.py --live --log-dir GRADING_ROOT/plumbing-logs
  python scripts/grading_job.py --live --log-dir GRADING_ROOT/_sealed/logs --sealed-dir GRADING_ROOT/_sealed
  python scripts/grading_job.py --live --resume    (continue the job the ledger records)
  Option: --grader-max-samples N (default 200): the most grader requests in flight in one eval.
  Option: --allow-final-ask (with --live --resume only): run isolated evals past the final ask guard.

Before the paid run:
  1. Run the plan-only command. Put the printed worker-log and plan-file hashes in the archive record
     as worker_log_shas and plan_file_shas, then commit, push and archive.
  2. Check the organisation's Batch queue limit for the grader model against the queued prompt
     tokens the plan-only command prints for --grader-max-samples.
  3. Set the OpenAI project budget for the grader and the Anthropic Console spend limit for the
     calibrator for the run.
  4. Keep the machine awake and stop update restarts for the whole run.
  5. Give the plumbing check and the job separate log folders, as above.
  6. Never open the _sealed folder, the attempt logs, the ledger or the provider's Batch files before
     the second-look freeze.

Resume is tied to HEAD on purpose: the ledger's provenance holds the commit, so any code change after
the start is refused. A refused or unreadable attempt log is recorded as rejected and stops the job;
the author rules by a logged deviation.

Exit codes: 0 plan printed or job complete, 1 an eval left units without a reply (recorded in the
ledger; run again with --live --resume), 2 bad arguments, 3 refused (including a rejected attempt log,
a vendor-change event, the final ask guard and a held lock).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import platform
import re
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader, roster, rubric  # noqa: E402
from icfr_agent_evals import grader_prompt as gp  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


al = _load("acceptance_look")
rr = _load("run_report")
gc = _load("grading_common")
cc = _load("calibrator_call")
gs = _load("grading_stats")
cpc = _load("calibrator_plumbing_check")

EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED = 0, 1, 2, 3

GRADER_MAX_SAMPLES_DEFAULT = 200
EVAL_KIND_MAIN, EVAL_KIND_REHEARSAL, EVAL_KIND_ISOLATED = "main", "rehearsal", "isolated"
EVAL_KINDS = (EVAL_KIND_MAIN, EVAL_KIND_REHEARSAL, EVAL_KIND_ISOLATED)
EVAL_NAME_SUFFIX = {EVAL_KIND_MAIN: "", EVAL_KIND_REHEARSAL: "_rehearsal", EVAL_KIND_ISOLATED: "_isolated"}
REHEARSAL_UNITS = 12          # the first units of grader part 1, asked as their own eval until a grader log succeeds
ISOLATE_AFTER_ASKS = 2        # asked in this many admitted attempt logs with no reply: next ask is isolated
ISOLATED_MAX_SAMPLES = 1
FINAL_ASK_GUARD_MAX = 12      # more units of a part due for their final ask than this: the guard stops the job

STAGE_LOG_CHECK = "log check"
STAGE_RESOLVED = "resolved model strings"
STAGE_FINAL_ASK_GUARD = "final ask guard"
STAGE_BATCH = "batch"
REJECT_FAILED_CHECK = "failed its check"
REJECT_UNREADABLE = "unreadable"

BATCH_PROGRESS_SECONDS = 600              # a progress line at least this often, or when the counts change
BATCH_OLD_SECONDS = 25 * 3600
BATCH_OLD_WARNING = ("a batch is older than 25 hours; a batch that expired with no result file is polled forever; "
                     "stop the job and run --live --resume")
LOCK_HELD = ("another live grading job holds the lock file grading-2026-11-21.lock; if no job is running, the "
             "author removes it")
ARCHIVE_HASHES_DIFFER = "the archive record lists other worker-log or plan-file hashes than the logs and plan in use"
ARCHIVE_HASHES_ABSENT = "note: the archive record does not list the worker-log and plan-file hashes"
RESULT_FILES_EXIST = "a result file or the evidence file exists before the job started; it is never overwritten"
EVIDENCE_WRITTEN_AGAIN = ("the grading job completed; its evidence file was missing and was written again from the "
                          "ledger")
PLUMBING_EVIDENCE_KEYS = ("file", "sha256", "attempt", "calls", "parsed", "refused", "passed", "head")

_BATCH_CREATED = re.compile(r"^Created batch (\S+) with (\d+) requests$")
_RESUME_HINT = "Run again with --live --resume; units with a completed reply are never asked again"
STAGE_HINTS: dict[str, str] = {
    STAGE_LOG_CHECK: (
        "The attempt log was refused and recorded as rejected; its replies are not used. The job stops here. "
        "The author rules by a logged deviation; a resume refuses while a rejected log is recorded, and a code "
        "fix changes HEAD, which a resume also refuses."),
    STAGE_RESOLVED: (
        "A judge call resolved to a model string outside the pin: a vendor-change event. The job stops here. "
        "A resume refuses while this stop is recorded; the author rules by a logged deviation."),
    STAGE_FINAL_ASK_GUARD: (
        "The isolated eval was not run: no call was made and no unit was recorded as a failed judgment. The "
        "units it would ask are at their final ask, and a unit with no reply there becomes a failed judgment "
        "for good. Find the cause of the first error first, such as a budget cap, a spend limit, a key error "
        "or a provider outage. Then log a deviation and run again with --live --resume --allow-final-ask."),
}
_REFUSALS: tuple[type[BaseException], ...] = (gc.Refused, al.Refused, rr.Refused, roster.RosterError,
                                              roster.RunConfigError)

# The rules as recorded in the evidence file. The draw rule is read from the sealed draw-pool
# body (grading_common.sealed_plan_bodies), so both files carry the same text.
RULE_TEXTS: dict[str, str] = {
    "grader_order": (
        "every gradeable unit, sorted by model, sample id and epoch, is shuffled once with the grader-order "
        "seed; a unit's opaque id is its 1-based position in that order, and the four grader evals take the "
        "ids in parts of roughly equal size by position (D-R23, grading order)"),
    "calibrator_order": (
        "the drawn unit ids, sorted, are shuffled once with the calibrator-order seed, and the calibrator eval "
        "asks them in that order, not batched (D-R23, grading order)"),
    "reply_of_record": (
        "the reply of record of a unit for each judge is its first completed call in any attempt log, whatever "
        "that log's final status and whether or not the reply parses; a resume asks only units with no "
        "reply of record, and outputs of a dead Batch that no log recorded count as missing (D-R23, reply of "
        "record)"),
    "failed_judgment": (
        "a unit asked in three attempt logs of its role with no completed call, the third time in an isolated "
        "eval that holds only units at their third ask and runs one sample at a time, is recorded as a failed "
        "judgment with no reply; it is never asked again, goes to the queue, is counted as failed in the pooled "
        "counts and is treated as a reply that does not parse (grader: ungraded; calibrator: a dropped pair); "
        "a unit counts as asked in a log only when the log holds its sample with a judge call, so an eval cut "
        "off before its samples reached the judge asks nothing (D-R23, failed judgments)"),
    "gate": (
        "the reliability gate is the ICC(A,1) lower limit on complete pairs, kept at 0.60 or above, as the "
        "pre-registration says; a worst-case figure that puts each dropped pair back at its largest "
        "disagreement (0 for a parsed 2 or 3, 3 for a parsed 0 or 1, 0 against 3 alternating when neither "
        "parsed) and the break-even count of dropped pairs are published beside it, reported and not gated "
        "(D-R23, reliability gate)"),
    "breaker": (
        "the circuit breaker's unit is the full run: it trips when the pair rate (tagged complete pairs over "
        "complete pairs) or the generation rate (tagged units over units with a parsed grader reply, all four "
        "worker runs pooled) is above one quarter; a tag is a disagreement of more than 1 point on a "
        "sub-dimension or a low-confidence flag from either judge; a failed judgment is never a breaker tag; a "
        "dropped pair leaves the pair rate, and its generation stays in the generation rate on its grader reply "
        "alone; a trip excludes both LLM sub-dimensions from every claim of the run; per-worker rates are "
        "descriptive only (D-R23, circuit breaker)"),
    "means": (
        "per worker model, subset and sub-dimension: a primary mean over graded generations, conditional on a "
        "parsed worker output, and a sensitivity mean over all generations with worker format breaks, errored "
        "epochs included, counted as 0; a grader reply that does not parse is missing in both; the point "
        "estimate is the mean of per-sample epoch means, with a normal 95 percent interval from their standard "
        "error, clipped to the 0 to 3 scale; each mean also gives the count of samples with fewer than 4 epochs "
        "and the D-R20 defect-epoch row; paired differences use the samples with a value for both models "
        "(D-R23, means)"),
}
EVIDENCE_RULE_KEYS = ("seed", "draw", "grader_order", "calibrator_order", "reply_of_record", "failed_judgment",
                      "gate", "breaker", "means")

WITHHELD_UNTIL_FREEZE: tuple[str, ...] = (
    "reliability gate values, the worst-case figure, the break-even counts and the bootstrap limit",
    "circuit breaker rates and trip",
    "LLM sub-dimension means and paired differences",
    "queue items and every tag count",
    "the unit map, the draw pool and the drawn ids",
)


def _role_of_part(part: str) -> str:
    if part == gc.CALIBRATOR_PART_KEY:
        return cc.CALIBRATOR_ROLE
    if part in gc.GRADER_PART_KEYS:
        return grader.GRADER_ROLE
    raise gc.Refused(f"{part!r} is not a part of the grading job")


def _reply_key(role: str) -> str:
    if role == grader.GRADER_ROLE:
        return "grader"
    if role == cc.CALIBRATOR_ROLE:
        return "calibrator"
    raise roster.RunConfigError(f"run config: role {role!r} is not a grading role")


def eval_name(part: str, eval_kind: str = EVAL_KIND_MAIN) -> str:
    """The Inspect eval name of one eval of a part: the part's main name, plus _rehearsal or
    _isolated. Only grader part 1 has a rehearsal."""
    if eval_kind not in EVAL_KINDS:
        raise gc.Refused(f"{eval_kind!r} is not an eval kind of the grading job")
    if part == gc.CALIBRATOR_PART_KEY:
        base = gc.CALIBRATOR_EVAL_NAME
    elif part in gc.GRADER_PART_KEYS:
        base = gc.GRADER_EVAL_NAME.format(gc.GRADER_PART_KEYS.index(part) + 1)
    else:
        raise gc.Refused(f"{part!r} is not a part of the grading job")
    if eval_kind == EVAL_KIND_REHEARSAL and part != gc.GRADER_PART_KEYS[0]:
        raise gc.Refused("only grader part 1 has a rehearsal")
    return base + EVAL_NAME_SUFFIX[eval_kind]


# --- Inspect tasks ---------------------------------------------------------------------------

def grading_solver(role: str):
    """An Inspect solver for one grading role. Inside the eval it checks that exactly that
    role is bound and the evaluated model is none/none (grading_common.check_eval_roles),
    then makes the one judge call: the grader of record through grader.grade at ladder step
    2, or the calibrator through calibrator_call.calibrate."""
    from inspect_ai.solver import Generate, TaskState, solver

    if role not in (grader.GRADER_ROLE, cc.CALIBRATOR_ROLE):
        raise roster.RosterError(f"role {role!r} is not a grading role")

    @solver(name=f"grading_{role}")
    def grading():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            gc.check_eval_roles(role)
            if role == grader.GRADER_ROLE:
                state.output = await grader.grade(state.messages, gc.GRADER_STEP)
            else:
                state.output = await cc.calibrate(state.messages)
            return state

        return solve

    return grading()


def _task_metadata(role: str, part: str, uids: Sequence[str], provenance: Mapping[str, Any],
                   eval_kind: str = EVAL_KIND_MAIN) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "split": gc.EVAL_SPLIT, "grading_job": gc.job_id(provenance), "role": role, "part": part,
        "eval_kind": eval_kind, "parts": gc.GRADER_PARTS, "units": len(uids), "grader_step": gc.GRADER_STEP,
        "grader_model": grader.grader_pin(gc.GRADER_STEP),
        "grader_generate_config": roster.grader_generate_config_args(gc.GRADER_STEP),
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), "worker_run_commit": gc.RUN_COMMIT,
        "head": provenance["head"],
    }
    if role == cc.CALIBRATOR_ROLE:
        del meta["grader_model"]
        meta.update({"calibrator_model": cc.calibrator_pin(),
                     "calibrator_generate_config": roster.calibrator_generate_config_args(), "batched": False})
    return meta


def _task(name: str, role: str, part: str, uids: Sequence[str], messages: Mapping[str, list[dict]],
          provenance: Mapping[str, Any], eval_kind: str):
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig

    samples = [Sample(id=uid, input=gp.to_chat_messages(messages[uid]),
                      metadata={"split": gc.EVAL_SPLIT, "unit": uid}) for uid in uids]
    return Task(dataset=MemoryDataset(samples, name=name), solver=[grading_solver(role)], epochs=1,
                config=GenerateConfig(), metadata=_task_metadata(role, part, uids, provenance, eval_kind), name=name)


def grader_task(part: str, uids: Sequence[str], messages: Mapping[str, list[dict]], provenance: Mapping[str, Any],
                *, eval_kind: str = EVAL_KIND_MAIN):
    """The Inspect task of one grader eval: one sample per unit id (input the unit's gp-1
    messages), one grader call each. Its name and metadata carry the part, the eval kind,
    the job id and the grader pin, and no worker model id."""
    if part not in gc.GRADER_PART_KEYS:
        raise gc.Refused(f"{part!r} is not a grader part")
    return _task(eval_name(part, eval_kind), grader.GRADER_ROLE, part, uids, messages, provenance, eval_kind)


def calibrator_task(uids: Sequence[str], messages: Mapping[str, list[dict]], provenance: Mapping[str, Any],
                    *, eval_kind: str = EVAL_KIND_MAIN):
    """The Inspect task of one calibrator eval: one sample per drawn unit id in the
    calibrator order. The metadata records the calibrator pin as provenance and batched
    False, and holds no grader model key."""
    part = gc.CALIBRATOR_PART_KEY
    return _task(eval_name(part, eval_kind), cc.CALIBRATOR_ROLE, part, uids, messages, provenance, eval_kind)


def run_eval(task: Any, *, roles: Mapping[str, Any], log_dir: str, max_samples: int | None = None) -> Any:
    """Run one grading eval: evaluated model none/none, the one role bound, epochs 1. A
    failed eval is never retried with eval-retry; its units are asked again by --resume
    only when no completed call of theirs was logged.

    Every eval runs with continue_on_fail, so one failed sample does not end the eval; the
    log still ends in status error, and the partial check keeps every completed reply.
    max_samples comes from --grader-max-samples (default 200) for grader evals, 1 for an
    isolated eval, and Inspect's default for the calibrator."""
    from inspect_ai import eval as inspect_eval

    return inspect_eval(task, model=grader.NO_MODEL, model_roles=dict(roles), epochs=1, log_dir=log_dir,
                        display="none", max_samples=max_samples, continue_on_fail=True)[0]


def read_attempt_log(path: pathlib.Path) -> Any:
    """Read one attempt log in full."""
    from inspect_ai.log import read_eval_log

    return read_eval_log(str(path))


def _read_or_refuse(path: pathlib.Path) -> Any:
    """read_attempt_log, with a log that cannot be read (for example one a killed process left
    part-written) refused rather than raised as a crash."""
    try:
        return read_attempt_log(path)
    except Exception as e:  # noqa: BLE001 - any read failure is a refusal
        raise gc.Refused(f"the log {pathlib.Path(path).name} in the grading log folder cannot be read "
                         f"({type(e).__name__}); the author decides how to proceed") from e


# --- Batch log hooks -------------------------------------------------------------------------

def _batch_log_setters() -> tuple[Callable[[Any], None], Callable[[Any], None]]:
    """Inspect's batch log and batch status hooks; a missing hook is refused before any call."""
    try:
        from inspect_ai.model._providers.util.batch_log import set_batch_log_callback, set_batch_status_callback
    except ImportError as e:
        raise gc.Refused("the batch log hooks are not available in this Inspect version; batch ids cannot be "
                         "recorded") from e
    return set_batch_log_callback, set_batch_status_callback


def _is_batch_failure(message: str) -> bool:
    return message.startswith("Error") or " failed (" in message or message.startswith("Attempting to resolve")


@contextlib.contextmanager
def batch_log_capture(on_created: Callable[[dict], None], on_failure: Callable[[str], None],
                      on_progress: Callable[[str], None]) -> Iterator[None]:
    """Inside the block, Inspect's batch log lines and batch status reach the job, never the
    console directly. A "Created batch ID with N requests" line calls on_created with
    {"batch_id", "requests"}; a line that starts with Error, holds " failed (" or starts with
    "Attempting to resolve" calls on_failure; every other line calls on_progress. A status
    calls on_progress with the pooled counts when they changed or 10 minutes passed, and a
    batch older than 25 hours writes one warning to stderr. Both hooks are set back to None
    on exit, also on an exception."""
    set_log, set_status = _batch_log_setters()
    seen: dict[str, Any] = {"counts": None, "time": 0.0, "warned": False}

    def log_line(message: Any) -> None:
        text = str(message)
        found = _BATCH_CREATED.match(text)
        if found:
            on_created({"batch_id": found.group(1), "requests": int(found.group(2))})
        elif _is_batch_failure(text):
            on_failure(text)
        else:
            on_progress(text)

    def status(s: Any) -> None:
        counts = (s.batch_count, s.pending_requests, s.completed_requests, s.failed_requests)
        now = time.time()
        oldest = getattr(s, "oldest_created_at", None)
        age = int(now - oldest) if oldest else 0
        if counts != seen["counts"] or now - seen["time"] >= BATCH_PROGRESS_SECONDS:
            seen["counts"], seen["time"] = counts, now
            on_progress(f"batch progress: {s.batch_count} batches, requests pending {s.pending_requests}, "
                        f"completed {s.completed_requests}, failed {s.failed_requests}, oldest batch age {age} s")
        if oldest and age > BATCH_OLD_SECONDS and not seen["warned"]:
            seen["warned"] = True
            print(BATCH_OLD_WARNING, file=sys.stderr)

    set_log(log_line)
    set_status(status)
    try:
        yield
    finally:
        set_log(None)
        set_status(None)


# --- Attempt log checks ----------------------------------------------------------------------

def _check_attempt_metadata(log: Any, *, role: str, job_id: str | None, part: str | None,
                            eval_kind: str | None = None) -> None:
    meta = dict(getattr(getattr(log, "eval", None), "metadata", None) or {})
    if job_id is not None and meta.get("grading_job") != job_id:
        raise roster.RunConfigError(f"run config: the attempt log belongs to grading job "
                                    f"{meta.get('grading_job')!r}, not {job_id!r}")
    if meta.get("role") != role:
        raise roster.RunConfigError(f"run config: the attempt log is for role {meta.get('role')!r}, not {role!r}")
    if part is not None and meta.get("part") != part:
        raise roster.RunConfigError(f"run config: the attempt log is part {meta.get('part')!r}, not {part!r}")
    if eval_kind is not None and meta.get("eval_kind") != eval_kind:
        raise roster.RunConfigError(f"run config: the attempt log is a {meta.get('eval_kind')!r} eval, not "
                                    f"{eval_kind!r}")


def _check_inputs(samples: Mapping[str, Any], plan: Mapping[str, Any] | None, role: str) -> None:
    if plan is None:
        return
    for uid, sample in sorted(samples.items()):
        unit = plan["units"].get(uid)
        if unit is None:
            raise roster.RunConfigError(f"run config: unit {uid} is not in the plan")
        if gc.chat_messages_sha256(getattr(sample, "input", None) or []) != unit["messages_sha256"]:
            raise roster.RunConfigError(f"run config: the {role} input of unit {uid} differs from the plan")


def _offline_grader_summary(uid: str, event: Any) -> dict[str, Any]:
    grader.check_grader_config(getattr(event, "config", None), gc.GRADER_STEP)
    if getattr(event, "cache", None) is not None:
        raise roster.RunConfigError(f"run config: the grader call of unit {uid} has cache="
                                    f"{getattr(event, 'cache', None)!r}; response caching is off")
    output = getattr(event, "output", None)
    usage = getattr(output, "usage", None)
    return {"model": getattr(event, "model", None), "resolved": str(getattr(output, "model", "") or ""),
            "usage": (usage.input_tokens, usage.output_tokens) if usage is not None else None, "sample_id": uid}


def check_grader_attempt(log: Any, asked: Sequence[str], *, offline: bool, plan: Mapping[str, Any] | None = None,
                         job_id: str | None = None, part: str | None = None,
                         eval_kind: str | None = None) -> dict[str, Any]:
    """Raise roster.RunConfigError unless one grader attempt log is admissible.

    A log that ended in success must hold exactly the asked units, each with one completed
    grader call (acceptance_look.check_pass_samples, which skips an errored call followed
    by a completed one), and pass grader.check_grader_log at step 2 live (pin, batch, no
    cache, resolved string) or the offline check. A log with any other status is admitted
    by grading_common.role_calls: each completed call passes grader.check_grader_event
    live, or the config and cache checks offline, and the units it lacks count as missing.
    Live, each completed sample must record the exact first-party serving endpoint, and no
    call may carry tools. Each completed sample's input must hash to the plan's
    messages_sha256, and the log's metadata must name this job, the grader role, the part
    and the eval kind. plan, job_id, part and eval_kind may be left out only where a
    caller checks them itself.

    Returns {"status", "completed": {uid: event}, "samples": {uid: sample}, "summaries",
    "skipped"}."""
    role = grader.GRADER_ROLE
    _check_attempt_metadata(log, role=role, job_id=job_id, part=part, eval_kind=eval_kind)
    status = getattr(log, "status", None)
    if status == "success":
        view, skipped = al.check_pass_samples(log, [{"_file": str(u)} for u in asked])
        if offline:
            al.offline_log_check(view, gc.GRADER_STEP)
            summaries = [_offline_grader_summary(str(s.id), e) for s, e in grader.grader_events(view)]
        else:
            summaries = grader.check_grader_log(view, gc.GRADER_STEP)
        completed = {str(s.id): e for s, e in grader.grader_events(view)}
        samples = {str(s.id): s for s in log.samples if str(s.id) in completed}
    else:
        calls = gc.role_calls(log, role, asked, require_all=False)
        completed, samples, skipped = dict(calls["completed"]), dict(calls["samples"]), int(calls["skipped"])
        summaries = []
        for uid, event in sorted(completed.items()):
            if offline:
                summary = _offline_grader_summary(uid, event)
            else:
                summary = grader.check_grader_event(event, gc.GRADER_STEP)
                summary["sample_id"] = uid
            summaries.append(summary)
    for uid, event in sorted(completed.items()):
        if getattr(event, "tools", None):
            raise roster.RunConfigError(f"run config: the grader call of unit {uid} carries tools")
        if not offline:
            gc.check_serving_recorded(samples[uid], source=grader.GRADER_EVENT_SOURCE, provider="openai")
    _check_inputs(samples, plan, "grader")
    return {"status": status, "completed": completed, "samples": samples, "summaries": summaries,
            "skipped": skipped}


def check_calibrator_attempt(log: Any, asked: Sequence[str], *, offline: bool, allowed_resolved: Any,
                             plan: Mapping[str, Any] | None = None, job_id: str | None = None,
                             part: str | None = None, eval_kind: str | None = None) -> dict[str, Any]:
    """Raise roster.RunConfigError unless one calibrator attempt log is admissible: a
    success log through calibrator_call.check_calibrator_log (every asked unit completed,
    the calibrator pin and config, no batch, no cache, no tools, effort high as sent,
    serving endpoint recorded, one resolved string within the Opus worker log's, one
    reasoning mode), any other status through calibrator_call.admit_calibrator_attempt_log.
    The same input-hash and metadata checks as check_grader_attempt. Same return shape."""
    role = cc.CALIBRATOR_ROLE
    _check_attempt_metadata(log, role=role, job_id=job_id, part=part, eval_kind=eval_kind)
    status = getattr(log, "status", None)
    if status == "success":
        calls, summaries = cc.check_calibrator_log(log, asked=list(asked), offline=offline,
                                                   allowed_resolved=allowed_resolved)
    else:
        calls, summaries = cc.admit_calibrator_attempt_log(log, asked=list(asked), offline=offline)
    completed, samples = dict(calls["completed"]), dict(calls["samples"])
    _check_inputs(samples, plan, "calibrator")
    return {"status": status, "completed": completed, "samples": samples, "summaries": list(summaries),
            "skipped": int(calls["skipped"])}


def check_all_resolved(grader_summaries: Sequence[Mapping[str, Any]],
                       calibrator_summaries: Sequence[Mapping[str, Any]], *, allowed_resolved: Any,
                       offline: bool) -> None:
    """Over all attempt logs together: the calibrator calls sent one reasoning mode, and,
    live, every grader call resolved to the step 2 tag-time string and the calibrator
    calls to one string within the Opus worker log's. A change is a vendor-change event."""
    cc.check_calibrator_modes(calibrator_summaries)
    if offline:
        return
    grader.check_resolved(grader_summaries, gc.GRADER_STEP)
    cc.check_calibrator_resolved(calibrator_summaries, allowed_resolved)


def replies_of_record(attempts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict]]:
    """The reply of record per role and unit (D-R23, reply of record): the completed call in
    the attempt logs, taken in ledger order (attempt number, then the order the logs were
    recorded).

    attempts: one item per admitted attempt log, {"attempt", "role", "log_file",
    "completed": {uid: ModelEvent}}. A unit with a completed call in two attempt logs is
    refused (roster.RunConfigError): a completed reply is never asked again.

    Returns {"grader": {uid: Reply}, "calibrator": {uid: Reply}}, built with
    grading_common.reply_record."""
    out: dict[str, dict[str, dict]] = {"grader": {}, "calibrator": {}}
    for a in sorted(attempts, key=lambda a: int(a["attempt"])):
        key = _reply_key(a["role"])
        for uid, event in a["completed"].items():
            if uid in out[key]:
                raise roster.RunConfigError(f"run config: unit {uid} was asked twice; a completed reply is never "
                                            f"asked again (D-R23, reply of record)")
            out[key][uid] = gc.reply_record(event, attempt=int(a["attempt"]), log_file=str(a["log_file"]))
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def sent_units(log: Any, role: str, asked: Sequence[str]) -> list[str]:
    """The asked unit ids whose sample the attempt log holds with at least one model event of
    the judge role, completed, errored or pending, in asked order. Only these count as asked
    for the failed-judgment rule: a unit in the eval's dataset that never reached the judge
    (an eval cut off before its sample started or was written) was not asked."""
    reached = {str(s.id) for s in getattr(log, "samples", None) or ()
               if any(getattr(e, "event", None) == "model" and getattr(e, "role", None) == role
                      for e in getattr(s, "events", None) or ())}
    return [str(u) for u in asked if str(u) in reached]


def ask_counts(attempt_logs: Sequence[Mapping[str, Any]], role: str) -> dict[str, dict]:
    """Per unit id, how often the admitted attempt logs of one role asked it, in ledger order:
    {"asks", "isolated" (asked at least once in an isolated eval), "last_isolated" (the
    attempt and log file of the last isolated eval that asked it, or None)}. A unit counts
    as asked in a log only when it is in that log's "sent" list (sent_units): its sample is
    in the log with a judge call. A unit in the eval's dataset with no such sample is not
    counted, so an eval cut off before its samples reached the judge never moves a unit
    toward isolation or a failed judgment."""
    out: dict[str, dict] = {}
    for e in attempt_logs:
        if e.get("role") != role:
            continue
        isolated = e.get("eval_kind") == EVAL_KIND_ISOLATED
        for uid in e.get("sent") or ():
            c = out.setdefault(str(uid), {"asks": 0, "isolated": False, "last_isolated": None})
            c["asks"] += 1
            if isolated:
                c["isolated"] = True
                c["last_isolated"] = (int(e["attempt"]), str(e["log_file"]))
    return out


def with_failed_judgments(replies: Mapping[str, Mapping[str, dict]],
                          attempt_logs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict]]:
    """The replies of record plus a failed judgment (grading_common.failed_judgment_record)
    for each unit with no reply that was asked in at least three admitted attempt logs of
    its role, at least once in an isolated eval (D-R23, failed judgments). Asked means sent
    (ask_counts): the log holds the unit's sample with a judge call. The record names the
    last isolated log that asked it."""
    out: dict[str, dict[str, dict]] = {}
    for key, role in (("grader", grader.GRADER_ROLE), ("calibrator", cc.CALIBRATOR_ROLE)):
        have = dict(replies.get(key) or {})
        for uid, c in ask_counts(attempt_logs, role).items():
            if uid in have or c["asks"] < gc.FAILED_JUDGMENT_ASKS or not c["isolated"]:
                continue
            attempt, log_file = c["last_isolated"]
            have[uid] = gc.failed_judgment_record(attempt=attempt, log_file=log_file)
        out[key] = dict(sorted(have.items()))
    return out


def plan_evals(part: str, todo: Sequence[str], counts: Mapping[str, Mapping[str, Any]], *,
               rehearsal: bool) -> list[tuple[str, list[str]]]:
    """The evals that ask a part's missing units, in order, as (eval kind, unit ids): the
    rehearsal (only when rehearsal is True and the part is grader part 1: the first 12 of
    the units not due for isolation), then the isolated eval (units asked in 2 or more
    admitted attempt logs with no reply), then the main eval (the rest). Empty groups are
    left out; unit ids keep part order."""
    isolated = due_for_final_ask(todo, counts)
    iso = set(isolated)
    rest = [u for u in todo if u not in iso]
    groups: list[tuple[str, list[str]]] = []
    if rehearsal and part == gc.GRADER_PART_KEYS[0]:
        groups.append((EVAL_KIND_REHEARSAL, rest[:REHEARSAL_UNITS]))
        rest = rest[REHEARSAL_UNITS:]
    groups.append((EVAL_KIND_ISOLATED, isolated))
    groups.append((EVAL_KIND_MAIN, rest))
    return [(kind, list(uids)) for kind, uids in groups if uids]


def rehearsal_due(attempt_logs: Sequence[Mapping[str, Any]]) -> bool:
    """True while no admitted grader attempt log has status success: grader part 1 then
    runs its rehearsal first, so a resume after a failed rehearsal rehearses again."""
    return not any(e.get("role") == grader.GRADER_ROLE and e.get("status") == "success" for e in attempt_logs)


def due_for_final_ask(todo: Sequence[str], counts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The units of a part with no reply that the next isolated eval asks: asked in
    ISOLATE_AFTER_ASKS or more admitted attempt logs. A unit with no reply after that ask is
    a failed judgment."""
    return [u for u in todo if int((counts.get(u) or {}).get("asks", 0)) >= ISOLATE_AFTER_ASKS]


def final_ask_guard(attempt_logs: Sequence[Mapping[str, Any]], role: str, due: Sequence[str]) -> str | None:
    """Why the isolated eval of a part must not run without --allow-final-ask, or None.

    It stops when more than FINAL_ASK_GUARD_MAX units of the part are due, or when the
    role's most recent admitted attempt log that sent at least one unit completed no call
    and sent more units than are now due: a whole eval failed, not only the bad units. A
    single unit asked alone that completed nothing still reaches its isolated eval."""
    if len(due) > FINAL_ASK_GUARD_MAX:
        return (f"{len(due)} units of the part are due for their final ask, more than {FINAL_ASK_GUARD_MAX}; an "
                f"outage of the whole account may have failed their calls")
    last = next((e for e in reversed(attempt_logs) if e.get("role") == role and e.get("sent")), None)
    if last is not None and int(last.get("completed") or 0) == 0 and len(last["sent"]) > len(due):
        return (f"the last attempt log of the role ({last.get('log_file')}) completed no call of the "
                f"{len(last['sent'])} units it sent; an outage of the whole account may have failed every call")
    return None


def final_ask_warning(due: int, part: str) -> str:
    """The warning after an eval that left units of a part with no reply while some of them
    are now due for their final ask."""
    units = "1 unit" if due == 1 else f"{due} units"
    verb = "is" if due == 1 else "are"
    return (f"warning: {units} of {part} {verb} now due for the final ask. The next resume that reaches them asks "
            f"them one at a time and records any with no reply as a failed judgment. Find the cause of the first "
            f"error before you resume.")


def missing_units(plan: Mapping[str, Any], replies: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """Per part key, the asked unit ids with no reply of record, in part order. A failed
    judgment is a reply of record here."""
    out: dict[str, list[str]] = {}
    for part in (*gc.GRADER_PART_KEYS, gc.CALIBRATOR_PART_KEY):
        have = replies.get(_reply_key(_role_of_part(part))) or {}
        out[part] = [u for u in plan["parts"][part] if u not in have]
    return out


def discover_orphan_logs(log_dir: pathlib.Path, ledger: Mapping[str, Any]) -> list[tuple[pathlib.Path, Any, Any]]:
    """The top-level .eval files in the grading log folder that the ledger lists neither as
    attempt logs nor as rejected logs, sorted by name, as (path, log, error). A file that
    cannot be read comes back with log None and its refusal as error; the caller records it
    as rejected. A readable log of any other job is refused (gc.Refused): a foreign log is
    in the grading log folder."""
    known = {str(e.get("log_file")) for e in ledger.get("attempt_logs") or ()}
    known |= {str(e.get("log_file")) for e in ledger.get("rejected_logs") or ()}
    found: list[tuple[pathlib.Path, Any, Any]] = []
    folder = pathlib.Path(log_dir)
    for path in sorted(folder.glob("*.eval")) if folder.is_dir() else ():
        if path.name in known:
            continue
        try:
            log = _read_or_refuse(path)
        except gc.Refused as e:
            found.append((path, None, e))
            continue
        meta = dict(getattr(getattr(log, "eval", None), "metadata", None) or {})
        if meta.get("grading_job") != ledger.get("job_id"):
            raise gc.Refused(f"a foreign log is in the grading log folder ({path.name}); the folder holds this "
                             f"job's attempt logs only")
        found.append((path, log, None))
    return found


# --- Results ---------------------------------------------------------------------------------

def build_rows(plan: Mapping[str, Any], replies: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """One Row per drawn calibrator row (lost rows are not rows), in unit id order."""
    rows = []
    for r in sorted(plan["draw"]["rows"], key=lambda r: r["uid"]):
        uid = r["uid"]
        rows.append({"uid": uid, "model": r["model"], "sample_id": r["sample_id"], "epoch": r["epoch"],
                     "position": r["position"], "grader": replies["grader"][uid],
                     "calibrator": replies["calibrator"][uid]})
    return rows


def build_units(plan: Mapping[str, Any], replies: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """Each unit of the plan with its grader reply of record, in unit id order."""
    return [{**dict(unit), "grader": replies["grader"][uid]} for uid, unit in sorted(plan["units"].items())]


def result_bodies(plan: Mapping[str, Any], replies: Mapping[str, Mapping[str, dict]], ledger: Mapping[str, Any], *,
                  bootstrap_seed: int, resamples: int = gs.BOOTSTRAP_RESAMPLES) -> dict[str, dict]:
    """The four sealed result files, file name to body. Deterministic: no time stamp."""
    units, rows = build_units(plan, replies), build_rows(plan, replies)
    return {
        gc.SEALED_GRADER_REPLIES: {"run": gc.RUN_LABEL, "replies": dict(replies["grader"])},
        gc.SEALED_CALIBRATOR_REPLIES: {"run": gc.RUN_LABEL, "replies": dict(replies["calibrator"])},
        gc.SEALED_QUEUE: {"run": gc.RUN_LABEL, "items": gs.queue_items(units, rows),
                          "note": ("opened only after the second-look freeze, through the unseal step (D-R23, "
                                   "unseal step); dispositions go in runs/reviews/<run-id>-judge-queue/, keyed "
                                   "by unit id")},
        gc.SEALED_AGGREGATES: {"run": gc.RUN_LABEL,
                               **gs.aggregates(units, plan["generations"], rows, replies["grader"],
                                               bootstrap_seed=bootstrap_seed, resamples=resamples)},
    }


def parts_sizes(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The part sizes: the four grader parts in order and the calibrator part."""
    return {"grader": [len(plan["parts"][k]) for k in gc.GRADER_PART_KEYS],
            "calibrator": len(plan["parts"][gc.CALIBRATOR_PART_KEY])}


def evidence_body(*, plan: Mapping[str, Any], ledger: Mapping[str, Any], state: Mapping[str, Any],
                  worker_logs: Sequence[Any], replies: Mapping[str, Mapping[str, dict]], plumbing: Mapping[str, Any],
                  sealed_files: Mapping[str, str], calibrator_summaries: Sequence[Mapping[str, Any]],
                  offline: bool) -> dict[str, Any]:
    """The committed evidence file: provenance, the rules, the seeds and sampling calls, the
    attempts (with --grader-max-samples) and attempt logs, the Batch ids, the pooled counts
    and the SHA-256 of every sealed file. The calibrator pin at calibrator.model is the
    only worker model id in it. The plumbing check block never holds a local path."""
    archive = state["archive"]
    look = state.get("look") or {}
    steps = sorted({w.header_grader_step for w in worker_logs})
    if len(steps) != 1:
        raise gc.Refused(f"the worker log headers record {len(steps)} grader steps, not one")
    resolved = sorted({str(s.get("resolved") or "") for s in calibrator_summaries})
    draw_rule = gc.sealed_plan_bodies(plan, worker_logs)[gc.SEALED_DRAW_POOL]["rule"]
    rule_texts = {"seed": gc.SEED_RULE_TEXT, "draw": draw_rule, **RULE_TEXTS}
    return {
        "evidence": "full-run grading", "run": gc.RUN_LABEL, "job_id": ledger["job_id"],
        "ledger_started": ledger["started"], "worker_run_commit": gc.RUN_COMMIT, "head": state["head"],
        "origin_main": state["origin_main"],
        "archive": {k: archive.get(k) for k in ("file", "sha256", "commit", "url", "date")},
        "python_version": platform.python_version(), "python_pin": gc.PYTHON_PIN,
        "inspect_ai_version": _inspect_version(),
        "worker_logs": sorted(({"file": w.file, "sha256": w.sha256, "eval_id": w.eval_id} for w in worker_logs),
                              key=lambda e: e["sha256"]),
        "worker_header_grader_step": steps[0],
        "grader": {"model": grader.grader_pin(gc.GRADER_STEP), "step": gc.GRADER_STEP,
                   "generate_config": roster.grader_generate_config_args(gc.GRADER_STEP), "batch": True,
                   "cache": False, "prompt_version": gp.PROMPT_VERSION, "prompt_sha256": gp.prompt_sha256(),
                   "anchors_sha256": rubric.llm_anchor_sha256(), "prompt_settings": "default (look pass_1)",
                   "resolved": grader.TAG_TIME_RESOLVED[gc.GRADER_STEP],
                   "look": {k: look.get(k) for k in ("file", "sha256", "verdict")} if look else None},
        "calibrator": {"model": cc.calibrator_pin(), "role": cc.CALIBRATOR_ROLE,
                       "generate_config": roster.calibrator_generate_config_args(), "batch": None, "cache": False,
                       "fallback": None, "tools": None, "prompt_version": gp.PROMPT_VERSION,
                       "prompt_sha256": gp.prompt_sha256(),
                       "resolved": resolved[0] if (not offline and len(resolved) == 1) else None,
                       "reasoning_modes_sent": sorted({str(s["request_mode"]) for s in calibrator_summaries
                                                       if s.get("request_mode")})},
        "plumbing_check": {k: plumbing.get(k) for k in PLUMBING_EVIDENCE_KEYS},
        "rules": {k: rule_texts[k] for k in EVIDENCE_RULE_KEYS},
        "sampling_calls": list(gc.SAMPLING_CALLS), "seeds": dict(plan["seeds"]),
        "parts": parts_sizes(plan),
        "attempts": list(ledger["attempts"]),
        "attempt_logs": [{k: v for k, v in e.items() if k not in ("asked", "sent")} for e in ledger["attempt_logs"]],
        "batch_ids": list(ledger["batch_ids"]),
        "counts": gc.pooled_counts(plan, replies["grader"], replies["calibrator"], ledger),
        "sealed": {"folder": gc.SEALED_FOLDER_NAME,
                   "files": {name: sealed_files[name] for name in (*gc.PLAN_FILES, *gc.RESULT_FILES)}},
        "ledger": {"file": gc.LEDGER_FILE, "folder": f"<git common dir>/{gc.LEDGER_FOLDER}"},
        "withheld_until_freeze": list(WITHHELD_UNTIL_FREEZE),
        "decision": "D-R23",
        "note": "pooled counts and hashes only until the second-look freeze (D-R23, sealed folder)",
    }


def summary_lines(counts: Mapping[str, Any], parts_sizes_: Mapping[str, Any], *,
                  evidence_path: pathlib.Path | str | None = None,
                  sealed_dir: pathlib.Path | str | None = None) -> list[str]:
    """The lines the finished job prints: pooled counts only. Never a model id, a worker log
    folder name, a score or a tag count."""
    g, c = counts["grader"], counts["calibrator"]
    lines = [
        f"grading job: {counts['worker_logs']} worker logs, {counts['generations']} generations, "
        f"{counts['skipped_errored']} errored and {counts['skipped_unparsed']} unparsed skipped, "
        f"{counts['units']} units",
        f"grader parts: {', '.join(str(n) for n in parts_sizes_['grader'])} units; calibrator part: "
        f"{parts_sizes_['calibrator']} units",
        f"grader: {g['replies_of_record']} replies of record ({g['parsed']} parsed, {g['unparsed']} unparsed, "
        f"{g['refused']} refused, {g['failed']} failed), {g['errored_calls_skipped']} errored calls skipped, "
        f"{g['attempt_logs']} attempt logs",
        f"calibrator: {c['rows_drawn']} rows drawn, {c['rows_lost']} lost; {c['replies_of_record']} replies of "
        f"record ({c['parsed']} parsed, {c['unparsed']} unparsed, {c['refused']} refused, {c['failed']} failed), "
        f"{c['errored_calls_skipped']} errored calls skipped, {c['attempt_logs']} attempt logs; "
        f"{c['pairs_complete']} complete pairs, {c['pairs_dropped']} dropped",
    ]
    if evidence_path is not None:
        lines.append(f"evidence: {evidence_path}")
    if sealed_dir is not None:
        lines.append(f"sealed results in {sealed_dir} (SHA-256 recorded)")
    return lines


def dry_run_lines(plan: Mapping[str, Any], worker_logs: Sequence[Any], *, grader_max_samples: int) -> list[str]:
    """The plan-only lines on prompt size and the hashes that go in the archive record before
    the first live call. Hashes and pooled counts only."""
    pc = plan["counts"]
    chars = int(pc["grader_prompt_chars"])
    per_unit = chars // max(int(pc["units"]), 1) // 4
    plan_hashes = gc.sealed_plan_hashes(plan, worker_logs)
    return [
        f"grader prompts: {chars} characters in all, about {per_unit} tokens per unit on average; with "
        f"--grader-max-samples {grader_max_samples}, up to about {grader_max_samples * per_unit} prompt tokens are "
        f"queued at once (check the org's Batch queue limit for the grader model)",
        "worker logs (SHA-256, in hash order): " + ", ".join(sorted(w.sha256 for w in worker_logs)),
        "plan files (SHA-256): " + ", ".join(f"{name} {plan_hashes[name]}" for name in gc.PLAN_FILES),
        "before the first live call, add these to the archive record as worker_log_shas and plan_file_shas, then "
        "commit, push and archive",
    ]


def _inspect_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("inspect_ai")
    except Exception:  # noqa: BLE001 - a missing distribution record falls back to the module
        try:
            import inspect_ai

            return str(getattr(inspect_ai, "__version__", None))
        except ImportError:
            return None


def _errored_samples(log: Any) -> list[Any]:
    return [s for s in getattr(log, "samples", None) or () if getattr(s, "error", None)]


def missing_after_eval_error(log: Any, *, status: Any, completed: int, asked: int, left: int) -> str:
    """The failure line of an eval that left units with no reply of record: the log status,
    the completed count and the first errored sample's message, cut to 200 characters."""
    if status == "success":
        return f"run config: {left} units have no completed reply after a success log"
    errored = _errored_samples(log)
    msg = str(getattr(errored[0].error, "message", "") or "") if errored else ""
    return (f"run config: the eval ended with status {status!r}; {completed} of {asked} units completed; "
            f"{len(errored)} samples errored; first error: {msg[:200]}")


# --- Main ------------------------------------------------------------------------------------

def _roles_for(model_roles: Any, part: str, role: str) -> dict[str, Any]:
    if model_roles is None:
        pin = grader.grader_pin(gc.GRADER_STEP) if role == grader.GRADER_ROLE else cc.calibrator_pin()
        return {role: pin}
    bound = model_roles(part) if callable(model_roles) else model_roles
    if role not in bound:
        raise roster.RosterError(f"the offline roles bind no {role} model")
    return {role: bound[role]}


def _err(message: str, code: int) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return code


def _asked_of(log: Any) -> list[str]:
    ids = getattr(getattr(getattr(log, "eval", None), "dataset", None), "sample_ids", None)
    if ids:
        return [str(i) for i in ids]
    return sorted(str(s.id) for s in (getattr(log, "samples", None) or ()))


def _max_samples_of(log: Any) -> Any:
    return getattr(getattr(getattr(log, "eval", None), "config", None), "max_samples", None)


def _release_lock(lock: pathlib.Path) -> None:
    """Remove the ledger lock this process created."""
    try:
        body = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(body, dict) and body.get("pid") == os.getpid():
        with contextlib.suppress(OSError):
            lock.unlink()


def main(argv: Sequence[str] | None = None, *, model_roles: Any = None, repo: pathlib.Path = REPO,
         ledger_dir: pathlib.Path | None = None, out_dir: pathlib.Path = gc.EVIDENCE_DIR,
         cases: Mapping[str, Any] | None = None, models: Sequence[str] = roster.EVALUATED_MODELS,
         calibrator_model: str = roster.CALIBRATOR, composition: Mapping[str, int] = gc.EVAL_COMPOSITION,
         per_model: int = gc.CALIBRATOR_PER_MODEL, run_commit: str = gc.RUN_COMMIT,
         look_file: pathlib.Path | None = gc.LOOK_FILE, archive_file: pathlib.Path = gc.ARCHIVE_FILE,
         running_package: bool = True, plumbing_dir: pathlib.Path | None = None,
         resamples: int = gs.BOOTSTRAP_RESAMPLES) -> int:
    """model_roles (tests only): a mapping {"grader": model, "calibrator": model} of offline
    (mockllm) role models, or a callable from part key to such a mapping; the job is then
    offline (offline log checks, no serving or resolved-string checks). Every other keyword
    is a test injection: the repository, the ledger folder, the evidence folder, the cases,
    the evaluated models, the calibrator model, the composition, the rows per model, the
    run commit, the look file, the archive record, the running-package check, the plumbing
    record folder and the bootstrap resamples."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true", help="make the grader and calibrator calls (keyed, paid)")
    ap.add_argument("--resume", action="store_true", help="continue the job the attempt ledger records")
    ap.add_argument("--log-dir", default=None,
                    help="folder for the grading Inspect logs: a folder inside the sealed folder, outside the "
                         "repository, apart from the plumbing check's log folder")
    ap.add_argument("--sealed-dir", default=None,
                    help=f"the sealed folder outside the repository, named {gc.SEALED_FOLDER_NAME}")
    ap.add_argument("--worker-logs", default=str(gc.WORKER_LOGS_DIR),
                    help="the folder of the worker logs, one subfolder per evaluated model")
    ap.add_argument("--grader-max-samples", type=int, default=GRADER_MAX_SAMPLES_DEFAULT,
                    help="the most grader requests in flight in one grader eval (default 200); recorded per "
                         "attempt, so a resume may use another value")
    ap.add_argument("--allow-final-ask", action="store_true",
                    help="with --live --resume only: run isolated evals past the final ask guard, after the cause "
                         "of the first error is found and a deviation is logged")
    a = ap.parse_args(argv)
    offline = model_roles is not None
    worker_folder = pathlib.Path(a.worker_logs)
    if a.resume and not a.live:
        return _err("--resume continues a live job; pass --live with it", EXIT_ARGS)
    if a.allow_final_ask and not (a.live and a.resume):
        return _err("--allow-final-ask is valid only with --live --resume", EXIT_ARGS)
    if a.grader_max_samples < 1:
        return _err("--grader-max-samples must be 1 or more", EXIT_ARGS)

    # 1. Admit the worker logs and build the plan.
    try:
        cases = gc.eval_cases() if cases is None else cases
        paths = gc.find_worker_logs(worker_folder, models, repo=repo)
        logs = gc.admit_worker_logs(paths, cases=cases, composition=composition, models=models,
                                    run_commit=run_commit)
        plan = gc.build_plan(logs, cases, calibrator_model=calibrator_model, per_model=per_model)
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    logs_by_model = {w.model: w for w in logs}
    sizes = parts_sizes(plan)
    pc = plan["counts"]

    # 2. Plan only.
    if not a.live:
        print(f"grading job plan: {len(logs)} worker logs admitted, {pc['generations']} generations, "
              f"{pc['skipped_errored']} errored and {pc['skipped_unparsed']} unparsed skipped, "
              f"{pc['units']} units")
        print(f"grader: {grader.grader_pin(gc.GRADER_STEP)}, ladder step {gc.GRADER_STEP}, batch, cache off; "
              f"parts of {', '.join(str(n) for n in sizes['grader'])} units")
        print(f"calibrator: role calibrator, not batched, cache off; {pc['rows_drawn']} rows drawn, "
              f"{pc['rows_lost']} lost")
        print(f"prompt {gp.PROMPT_VERSION} ({gp.prompt_sha256()[:12]}), default settings (look pass_1)")
        try:
            print("\n".join(dry_run_lines(plan, logs, grader_max_samples=a.grader_max_samples)))
        except _REFUSALS as e:
            return _err(str(e), EXIT_REFUSED)
        try:
            started = gc.ledger_path(repo, ledger_dir).exists()
        except _REFUSALS:
            started = False
        print("a grading job was started: its ledger exists; --live --resume continues it" if started
              else "no grading job has started (no ledger)")
        print("dry run: no call made (pass --live with --log-dir and --sealed-dir to make the calls)")
        return EXIT_OK

    # 3. The ledger and its lock: one live process per ledger.
    try:
        ledger_file = gc.ledger_path(repo, ledger_dir)
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    lock = gc.ledger_lock_path(ledger_file)
    try:
        gc.write_new_file(lock, gc.canonical_bytes({"pid": os.getpid(), "started": gc._now()}))
    except FileExistsError:
        return _err(LOCK_HELD, EXIT_REFUSED)
    try:
        return _live_job(a, offline=offline, worker_folder=worker_folder, ledger_file=ledger_file, plan=plan,
                         logs=logs, logs_by_model=logs_by_model, sizes=sizes, cases=cases, model_roles=model_roles,
                         repo=repo, ledger_dir=ledger_dir, out_dir=out_dir, calibrator_model=calibrator_model,
                         run_commit=run_commit, look_file=look_file, archive_file=archive_file,
                         running_package=running_package, plumbing_dir=plumbing_dir, resamples=resamples)
    finally:
        _release_lock(lock)


def _complete_again(ledger: Mapping[str, Any], target: pathlib.Path) -> int:
    """A resume of a completed job: write a missing evidence file from the ledger's evidence
    body when it matches the recorded SHA-256, else refuse."""
    done = ledger.get("completed") or {}
    if target.exists():
        return _err(f"the grading job completed ({done.get('time')}); it is not run again", EXIT_REFUSED)
    body = done.get("evidence_body")
    if not isinstance(body, Mapping):
        return _err("the grading job completed, its evidence file is missing and the ledger holds no evidence "
                    "body; the author decides how to proceed", EXIT_REFUSED)
    data = gc.canonical_bytes(body)
    if gc.sha256_bytes(data) != done.get("evidence_sha256"):
        return _err("the ledger's evidence body does not match its recorded SHA-256; the author decides how to "
                    "proceed", EXIT_REFUSED)
    try:
        gc.write_or_match_bytes(target, data, what="evidence file")
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    print(EVIDENCE_WRITTEN_AGAIN)
    return EXIT_OK


def _live_job(a: Any, *, offline: bool, worker_folder: pathlib.Path, ledger_file: pathlib.Path,
              plan: Mapping[str, Any], logs: Sequence[Any], logs_by_model: Mapping[str, Any],
              sizes: Mapping[str, Any], cases: Mapping[str, Any], model_roles: Any, repo: pathlib.Path,
              ledger_dir: pathlib.Path | None, out_dir: pathlib.Path, calibrator_model: str, run_commit: str,
              look_file: pathlib.Path | None, archive_file: pathlib.Path, running_package: bool,
              plumbing_dir: pathlib.Path | None, resamples: int) -> int:
    """The live job, run while this process holds the ledger lock."""
    target = pathlib.Path(out_dir) / gc.EVIDENCE_FILE

    # 4. Resume: read the ledger; a completed job or a recorded rejected log stops here.
    ledger: dict[str, Any] | None = None
    if a.resume:
        if not ledger_file.exists():
            return _err("no grading ledger exists; there is no job to resume", EXIT_REFUSED)
        try:
            ledger = gc.read_ledger(ledger_file)
        except _REFUSALS as e:
            return _err(str(e), EXIT_REFUSED)
        for flag, key in ((a.log_dir, "log_dir"), (a.sealed_dir, "sealed_dir")):
            if flag and pathlib.Path(flag).resolve() != pathlib.Path(str(ledger.get(key))).resolve():
                return _err(f"--{key.replace('_', '-')} differs from the one the ledger records", EXIT_ARGS)
        if ledger.get("completed"):
            return _complete_again(ledger, target)
        rejected = ledger.get("rejected_logs") or []
        if rejected:
            return _err(f"the ledger records {len(rejected)} rejected attempt logs; the author rules by a logged "
                        f"deviation before the job goes on", EXIT_REFUSED)
        if any(f.get("stage") == STAGE_RESOLVED for f in ledger.get("failures") or ()):
            return _err("the ledger records a resolved model strings stop (a vendor-change event); the author "
                        "rules by a logged deviation before the job goes on", EXIT_REFUSED)
        log_dir_s, sealed_dir_s = ledger.get("log_dir"), ledger.get("sealed_dir")
    else:
        if ledger_file.exists():
            return _err("a job was already started (its ledger exists); pass --resume", EXIT_REFUSED)
        log_dir_s, sealed_dir_s = a.log_dir, a.sealed_dir

    # 5. Folders, start checks, archive hashes and the plumbing check.
    for flag, value in (("--log-dir", log_dir_s), ("--sealed-dir", sealed_dir_s)):
        if not value or al._inside(value, REPO) or al._inside(value, repo):
            return _err(f"--live needs {flag} outside the repository", EXIT_ARGS)
    fresh = ledger is None
    try:
        sealed_dir = gc.check_sealed_dir(pathlib.Path(sealed_dir_s), repo)
        log_dir = pathlib.Path(log_dir_s).resolve()
        state = gc.start_checks(repo, run_commit=run_commit, look_file=look_file, archive_file=archive_file,
                                folders={"--log-dir": log_dir, "--sealed-dir": sealed_dir,
                                         "--worker-logs": worker_folder},
                                running_package=running_package)
        archive = state["archive"]
        listed_logs, listed_plan = archive.get("worker_log_shas"), archive.get("plan_file_shas")
        if listed_logs is not None and list(listed_logs) != sorted(w.sha256 for w in logs):
            raise gc.Refused(ARCHIVE_HASHES_DIFFER)
        if listed_plan is not None and dict(listed_plan) != gc.sealed_plan_hashes(plan, logs):
            raise gc.Refused(ARCHIVE_HASHES_DIFFER)
        if listed_logs is None or listed_plan is None:
            print(ARCHIVE_HASHES_ABSENT, file=sys.stderr)
        opus_log = logs_by_model.get(calibrator_model)
        if opus_log is None and not offline:
            raise gc.Refused("no admitted worker log is the calibrator's own run, so its resolved string cannot be "
                             "checked")
        allowed_resolved = opus_log.resolved if opus_log is not None else None
        plumbing = cpc.check_plumbing_passed(plumbing_dir or cpc.record_dir(repo, ledger_dir), repo=repo,
                                             offline=offline, allowed_resolved=allowed_resolved)
        gc.check_job_folders(log_dir=log_dir, sealed_dir=sealed_dir, worker_logs=worker_folder, fresh=fresh,
                             plumbing_log_dir=plumbing.get("log_dir"))
        _batch_log_setters()
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)

    # 6. Provenance (no worker model id).
    look = state.get("look")
    provenance = {
        "head": state["head"], "origin_main": state["origin_main"], "run_commit": run_commit,
        "archive_commit": state["archive"]["commit"], "archive_url": state["archive"]["url"],
        "look_sha256": look["sha256"] if look else None, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), "worker_log_shas": sorted(w.sha256 for w in logs),
        "seeds": dict(plan["seeds"]), "python_version": platform.python_version(),
        "inspect_ai_version": _inspect_version(), "plumbing_record_sha256": plumbing["sha256"],
    }
    job = gc.job_id(provenance)
    plan_bodies = gc.sealed_plan_bodies(plan, logs)
    checked: list[dict[str, Any]] = []

    def admit(entry_role: str, log: Any, asked: Sequence[str], part: str, eval_kind: str | None) -> dict[str, Any]:
        if entry_role == grader.GRADER_ROLE:
            chk = check_grader_attempt(log, asked, offline=offline, plan=plan, job_id=job, part=part,
                                       eval_kind=eval_kind)
        else:
            chk = check_calibrator_attempt(log, asked, offline=offline, allowed_resolved=allowed_resolved,
                                           plan=plan, job_id=job, part=part, eval_kind=eval_kind)
        return {**chk, "sent": sent_units(log, entry_role, asked)}

    # 7. Fresh start or resume.
    resume_entry: dict[str, Any] | None = None
    replies: dict[str, dict[str, dict]] = {"grader": {}, "calibrator": {}}
    if ledger is None:
        if ledger_file.exists():
            return _err("a job was already started (its ledger exists); pass --resume", EXIT_REFUSED)
        if target.exists() or any((sealed_dir / name).exists() for name in gc.RESULT_FILES):
            return _err(RESULT_FILES_EXIST, EXIT_REFUSED)
        try:
            sealed_dir.mkdir(parents=True, exist_ok=True)
            plan_hashes = {name: gc.write_or_match_sealed(sealed_dir, name, plan_bodies[name])
                           for name in gc.PLAN_FILES}
        except _REFUSALS as e:
            return _err(str(e), EXIT_REFUSED)
        ledger = gc.new_ledger(provenance=provenance, plan=plan, sealed_plan_hashes=plan_hashes, log_dir=log_dir,
                               sealed_dir=sealed_dir)
        ledger.setdefault("rejected_logs", [])
        ledger["attempts"][0]["grader_max_samples"] = a.grader_max_samples
        try:
            gc.create_ledger(ledger_file, ledger)
        except FileExistsError:
            return _err("a job was already started; pass --resume", EXIT_REFUSED)
    else:
        ledger.setdefault("rejected_logs", [])
        off = sorted(k for k in set(provenance) | set(ledger.get("provenance") or {})
                     if (ledger.get("provenance") or {}).get(k) != provenance.get(k))
        if off:
            return _err(f"the job differs from the one the ledger started on {off}; resume is tied to HEAD on "
                        f"purpose", EXIT_REFUSED)
        if ledger.get("job_id") != job or ledger.get("seeds") != plan["seeds"] or ledger.get("parts") != plan["parts"]:
            return _err("the ledger's job id, seeds or parts differ from the recomputed plan", EXIT_REFUSED)
        try:
            for name in gc.PLAN_FILES:
                sha = gc.write_or_match_sealed(sealed_dir, name, plan_bodies[name])
                if sha != (ledger.get("sealed_plan_files") or {}).get(name):
                    raise gc.Refused(f"the sealed file {name} does not match the SHA-256 the ledger records")
        except _REFUSALS as e:
            return _err(str(e), EXIT_REFUSED)
        attempt_no = len(ledger["attempts"]) + 1
        resume_entry = {"attempt": attempt_no, "started": gc._now(), "resume": True,
                        "grader_max_samples": a.grader_max_samples, "kept": 0,
                        "allow_final_ask": bool(a.allow_final_ask)}
        try:
            for entry in ledger["attempt_logs"]:
                path = log_dir / str(entry["log_file"])
                if not path.is_file():
                    raise gc.Refused(f"the attempt log {entry['log_file']} is not in the grading log folder")
                if gc.sha256_file(path) != entry["log_sha256"]:
                    raise gc.Refused(f"the attempt log {path.name} does not match its recorded SHA-256")
                chk = admit(entry["role"], _read_or_refuse(path), entry["asked"], entry["part"],
                            entry.get("eval_kind"))
                if chk["status"] != entry["status"]:
                    raise gc.Refused(f"the attempt log {path.name} has status {chk['status']!r}, not the recorded "
                                     f"{entry['status']!r}")
                if list(entry.get("sent") or ()) != chk["sent"]:
                    raise gc.Refused(f"the attempt log {path.name} sent other units to the judge than the ledger "
                                     f"records")
                checked.append({"attempt": entry["attempt"], "part": entry["part"], "role": entry["role"],
                                "log_file": path.name, "completed": chk["completed"], "summaries": chk["summaries"]})
            replies = replies_of_record(checked)
            orphans = discover_orphan_logs(log_dir, ledger)
        except _REFUSALS as e:
            return _err(f"resume: {e}", EXIT_REFUSED)
    attempt = ledger["attempts"][-1]["attempt"] if resume_entry is None else resume_entry["attempt"]

    def save() -> None:
        gc.save_ledger(ledger_file, ledger)

    def open_resume_entry() -> None:
        if resume_entry is not None and resume_entry not in ledger["attempts"]:
            resume_entry["kept"] = len(_ror_index(with_failed_judgments(replies, ledger["attempt_logs"])))
            ledger["attempts"].append(resume_entry)

    def fail(stage: str, part: str | None, err: BaseException, log: Any = None, *,
             eval_kind: str | None = None) -> int:
        open_resume_entry()
        entry: dict[str, Any] = {"attempt": attempt, "time": gc._now(), "stage": stage, "part": part,
                                 "eval_kind": eval_kind, "error": f"{type(err).__name__}: {err}"}
        loc = getattr(log, "location", None)
        if loc and pathlib.Path(str(loc)).is_file():
            entry["log_file"] = pathlib.Path(str(loc)).name
            entry["log_sha256"] = gc.sha256_file(pathlib.Path(str(loc)))
        ledger["failures"].append(entry)
        save()
        print(f"FAIL: {stage}{f' ({part})' if part else ''}: {err}", file=sys.stderr)
        print(STAGE_HINTS.get(stage, _RESUME_HINT), file=sys.stderr)
        return EXIT_REFUSED if stage in STAGE_HINTS else EXIT_CHECK

    def reject(path: pathlib.Path, *, part: str | None, role: str | None, eval_kind: str | None, reason: str,
               err: BaseException, found_on_resume: bool) -> int:
        """Record a refused or unreadable attempt log as rejected (D-R23, rejected logs) and stop."""
        open_resume_entry()
        now, sha = gc._now(), gc.sha256_file(path)
        ledger["rejected_logs"].append({"attempt": attempt, "time": now, "part": part, "role": role,
                                        "eval_kind": eval_kind, "log_file": path.name, "log_sha256": sha,
                                        "stage": STAGE_LOG_CHECK, "reason": reason,
                                        "found_on_resume": found_on_resume})
        ledger["failures"].append({"attempt": attempt, "time": now, "stage": STAGE_LOG_CHECK, "part": part,
                                   "eval_kind": eval_kind, "error": f"{type(err).__name__}: {err}",
                                   "log_file": path.name, "log_sha256": sha})
        save()
        print(f"FAIL: {STAGE_LOG_CHECK}{f' ({part})' if part else ''}: {err}", file=sys.stderr)
        print(STAGE_HINTS[STAGE_LOG_CHECK], file=sys.stderr)
        return EXIT_REFUSED

    # 8. Resume: orphan logs of this job are admitted, or recorded as rejected.
    if resume_entry is not None:
        for path, log, read_error in orphans:
            if log is None:
                return reject(path, part=None, role=None, eval_kind=None, reason=REJECT_UNREADABLE, err=read_error,
                              found_on_resume=True)
            meta = dict(getattr(log.eval, "metadata", None) or {})
            part, kind = meta.get("part"), meta.get("eval_kind")
            role = None
            try:
                if part not in plan["parts"]:
                    raise gc.Refused(f"the log {path.name} names no part of this job")
                role = _role_of_part(part)
                if kind not in EVAL_KINDS:
                    raise gc.Refused(f"the log {path.name} names no eval kind of this job")
                if kind == EVAL_KIND_ISOLATED and _max_samples_of(log) != ISOLATED_MAX_SAMPLES:
                    raise gc.Refused(f"the isolated log {path.name} did not run one sample at a time")
                asked = _asked_of(log)
                if any(u not in plan["parts"][part] for u in asked):
                    raise gc.Refused(f"the log {path.name} asks units outside its part {part}")
                chk = admit(role, log, asked, part, kind)
                new = {"attempt": attempt, "part": part, "role": role, "log_file": path.name,
                       "completed": chk["completed"], "summaries": chk["summaries"]}
                new_replies = replies_of_record([*checked, new])
            except Exception as e:  # noqa: BLE001 - any refusal of an orphan of this job is a rejected log
                return reject(path, part=part if part in plan["parts"] else None, role=role,
                              eval_kind=kind if kind in EVAL_KINDS else None, reason=REJECT_FAILED_CHECK, err=e,
                              found_on_resume=True)
            checked.append(new)
            replies = new_replies
            ledger["attempt_logs"].append({
                "attempt": attempt, "part": part, "role": role, "eval_kind": kind, "log_file": path.name,
                "log_sha256": gc.sha256_file(path), "status": chk["status"], "asked": list(asked),
                "sent": list(chk["sent"]), "completed": len(chk["completed"]), "errored_calls_skipped": chk["skipped"],
                "max_samples": _max_samples_of(log), "found_on_resume": True})
        replies = with_failed_judgments(replies, ledger["attempt_logs"])
        ledger["replies_of_record"] = _ror_index(replies)
        open_resume_entry()
        save()
        # The resolved strings of every re-admitted attempt log, before any new eval. Offline,
        # check_all_resolved checks the calibrator's reasoning mode only.
        try:
            check_all_resolved([s for c in checked if c["role"] == grader.GRADER_ROLE for s in c["summaries"]],
                               [s for c in checked if c["role"] == cc.CALIBRATOR_ROLE for s in c["summaries"]],
                               allowed_resolved=allowed_resolved, offline=offline)
        except Exception as e:  # noqa: BLE001 - a vendor-change event stops the job
            return fail(STAGE_RESOLVED, None, e)

    # 9. The parts in order: grader-1 to grader-4, then the calibrator. Each part runs as its
    # rehearsal (grader part 1 only, until a grader log ends in success), its isolated eval
    # and its main eval.
    current: dict[str, Any] = {"part": None, "eval_kind": None}

    def on_created(b: Mapping[str, Any]) -> None:
        ledger["batch_ids"].append({"attempt": attempt, "part": current["part"], "eval_kind": current["eval_kind"],
                                    "batch_id": b["batch_id"], "requests": b["requests"], "time": gc._now()})
        save()

    def on_failure(message: str) -> None:
        ledger["failures"].append({"attempt": attempt, "time": gc._now(), "stage": STAGE_BATCH,
                                   "part": current["part"], "eval_kind": current["eval_kind"], "error": message})
        save()
        print(f"batch: {message[:200]}", file=sys.stderr)

    def on_progress(message: str) -> None:
        print(message, flush=True)

    for part in (*gc.GRADER_PART_KEYS, gc.CALIBRATOR_PART_KEY):
        role = _role_of_part(part)
        key = _reply_key(role)
        todo = missing_units(plan, replies)[part]
        if not todo:
            continue
        evals = plan_evals(part, todo, ask_counts(ledger["attempt_logs"], role),
                           rehearsal=rehearsal_due(ledger["attempt_logs"]))
        for kind, uids in evals:
            current.update(part=part, eval_kind=kind)
            if kind == EVAL_KIND_ISOLATED:
                why = final_ask_guard(ledger["attempt_logs"], role, uids)
                if why is not None and not a.allow_final_ask:
                    return fail(STAGE_FINAL_ASK_GUARD, part, roster.RunConfigError(f"run config: {why}"),
                                eval_kind=kind)
                max_samples = ISOLATED_MAX_SAMPLES
            elif role == grader.GRADER_ROLE:
                max_samples = a.grader_max_samples
            else:
                max_samples = None
            try:
                messages = {}
                for uid in uids:
                    msgs = gc.unit_messages(plan["units"][uid], logs_by_model, cases)
                    if gc.messages_sha256(msgs) != plan["units"][uid]["messages_sha256"]:
                        raise roster.RunConfigError(f"run config: the {key} input of unit {uid} differs from the "
                                                    f"plan")
                    messages[uid] = msgs
                task = (grader_task(part, uids, messages, provenance, eval_kind=kind) if role == grader.GRADER_ROLE
                        else calibrator_task(uids, messages, provenance, eval_kind=kind))
                roles = _roles_for(model_roles, part, role)
            except Exception as e:  # noqa: BLE001 - any failure before the eval stops the job and is recorded
                return fail("messages", part, e, eval_kind=kind)
            log = None
            try:
                hooks = (batch_log_capture(on_created, on_failure, on_progress) if role == grader.GRADER_ROLE
                         else contextlib.nullcontext())
                with hooks:
                    log = run_eval(task, roles=roles, log_dir=str(log_dir), max_samples=max_samples)
            except Exception as e:  # noqa: BLE001 - an eval that raises is recorded; its log, if any, is found on resume
                return fail(f"{key} eval", part, e, log, eval_kind=kind)
            try:
                chk = admit(role, log, uids, part, kind)
                name, sha = al.log_file_of(log)
                new = {"attempt": attempt, "part": part, "role": role, "log_file": name,
                       "completed": chk["completed"], "summaries": chk["summaries"]}
                new_replies = replies_of_record([*checked, new])
            except Exception as e:  # noqa: BLE001 - a refused attempt log is recorded as rejected and stops the job
                loc = getattr(log, "location", None)
                if loc and pathlib.Path(str(loc)).is_file():
                    return reject(pathlib.Path(str(loc)), part=part, role=role, eval_kind=kind,
                                  reason=REJECT_FAILED_CHECK, err=e, found_on_resume=False)
                return fail(STAGE_LOG_CHECK, part, e, log, eval_kind=kind)
            checked.append(new)
            ledger["attempt_logs"].append({
                "attempt": attempt, "part": part, "role": role, "eval_kind": kind, "log_file": name,
                "log_sha256": sha, "status": chk["status"], "asked": list(uids), "sent": list(chk["sent"]),
                "completed": len(chk["completed"]),
                "errored_calls_skipped": chk["skipped"], "max_samples": max_samples, "found_on_resume": False})
            replies = with_failed_judgments(new_replies, ledger["attempt_logs"])
            ledger["replies_of_record"] = _ror_index(replies)
            save()
            if kind == EVAL_KIND_REHEARSAL and not offline:
                try:
                    grader.check_resolved(chk["summaries"], gc.GRADER_STEP)
                except Exception as e:  # noqa: BLE001 - a vendor-change event stops the job
                    return fail(STAGE_RESOLVED, part, e, log, eval_kind=kind)
            left = [u for u in uids if u not in replies[key]]
            if left:
                code = fail(f"{key} eval", part,
                            roster.RunConfigError(missing_after_eval_error(
                                log, status=chk["status"], completed=len(chk["completed"]), asked=len(uids),
                                left=len(left))), log, eval_kind=kind)
                due = due_for_final_ask(missing_units(plan, replies)[part], ask_counts(ledger["attempt_logs"], role))
                if due:
                    print(final_ask_warning(len(due), part), file=sys.stderr)
                return code

    # 10. Every unit has a reply of record: results, sealed files, the ledger mark and the evidence file.
    if any(missing_units(plan, replies).values()):
        return fail("results", None, roster.RunConfigError("run config: units are still missing a reply of record"))
    grader_summaries = [s for c in checked if c["role"] == grader.GRADER_ROLE for s in c["summaries"]]
    calibrator_summaries = [s for c in checked if c["role"] == cc.CALIBRATOR_ROLE for s in c["summaries"]]
    try:
        check_all_resolved(grader_summaries, calibrator_summaries, allowed_resolved=allowed_resolved,
                           offline=offline)
    except Exception as e:  # noqa: BLE001 - a vendor-change event stops the job
        return fail(STAGE_RESOLVED, None, e)
    try:
        bodies = result_bodies(plan, replies, ledger, bootstrap_seed=plan["seeds"]["bootstrap"], resamples=resamples)
    except Exception as e:  # noqa: BLE001 - a result that cannot be computed stops the job
        return fail("results", None, e)
    try:
        result_hashes = {name: gc.write_or_match_sealed(sealed_dir, name, bodies[name]) for name in gc.RESULT_FILES}
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    sealed_files = {**dict(ledger["sealed_plan_files"]), **result_hashes}
    try:
        evidence = evidence_body(plan=plan, ledger=ledger, state=state, worker_logs=logs, replies=replies,
                                 plumbing=plumbing, sealed_files=sealed_files,
                                 calibrator_summaries=calibrator_summaries, offline=offline)
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    data = gc.canonical_bytes(evidence)
    if target.exists() and target.read_bytes() != data:
        return _err(f"the evidence file {target.name} exists and differs from the recomputed one; it is never "
                    f"overwritten", EXIT_REFUSED)
    evidence_sha = gc.sha256_bytes(data)
    ledger["completed"] = {"time": gc._now(), "evidence_file": target.name, "evidence_sha256": evidence_sha,
                           "sealed_files": sealed_files, "evidence_body": evidence}
    save()
    try:
        gc.write_or_match_bytes(target, data, what="evidence file")
    except _REFUSALS as e:
        return _err(str(e), EXIT_REFUSED)
    print("\n".join(summary_lines(evidence["counts"], sizes, evidence_path=target, sealed_dir=sealed_dir)))
    return EXIT_OK


def _ror_index(replies: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, dict]]:
    """The ledger's replies_of_record: per unit id, the attempt and log file of each judge's
    reply of record, failed judgments included."""
    index: dict[str, dict[str, dict]] = {}
    for key in ("grader", "calibrator"):
        for uid, r in (replies.get(key) or {}).items():
            index.setdefault(uid, {})[key] = {"attempt": r["attempt"], "log_file": r["log_file"]}
    return dict(sorted(index.items()))


if __name__ == "__main__":
    sys.exit(main())
