"""Shared parts of the full-run grading job: constants, worker-log admission, units,
seeds, the calibrator draw, the grading order, grader messages, the attempt ledger,
sealed-file IO, pooled counts and the start checks.

Decision D-R23 (2026-11-22). The grader of record (ladder step 2, on Batch, prompt gp-1
with default settings) grades every generation whose worker output parsed and did not
error, in all four worker runs. The calibrator (Opus 5.5) grades a drawn sample. This
module holds no Inspect task and makes no model call; scripts/grading_job.py runs the
evals, and scripts/calibrator_call.py is the calibrator call.

What this module holds:
  - Worker-log admission (find_worker_logs, admit_worker_log, admit_worker_logs). Each
    worker log of the run is read without rewriting it and refused unless it is a whole
    successful evaluation-split run at k = 4 on the section 5 configuration, at the run
    commit, on the locked worker prompt, with header grader_step 1, on the 200 cases of
    section 4 as build_dataset(split="eval") rebuilds them, and with a parse verdict that
    agrees with the scorers' format-break record on every epoch.
  - Generations and units (classify_generation, build_generations, assign_unit_ids). An
    errored epoch is never graded, even when its completion parses (D-R23, units). An
    epoch whose worker output does not parse is never graded. Both are counted. Every
    other generation is a unit with an opaque id.
  - Seeds (rule_seed, plan_seeds). Each seed is fixed by rule, not chosen: the first 16
    hex digits, read as an integer, of the SHA-256 of the worker-log SHA-256s sorted by
    model id, joined with "|", followed by a suffix that names its use.
  - The calibrator draw (draw_calibrator_sample, D-R23, draw). The 200 case ids, sorted,
    are shuffled with the "|calibrator" seed. The worker runs other than the calibrator's
    own model, sorted by model id, take the shuffled positions in rotation: 1 to 80, 81
    to 160, then 161 to 200 and 1 to 40. For each (model, case) one gradeable epoch is
    picked at random with the same generator. A (model, case) with no gradeable epoch
    loses its row: there is no refill and no draw is made for it. The draw reads no
    grader output, so it is fixed before any grader output exists.
  - The grading order (assign_unit_ids, split_parts, calibrator_order, D-R23, grading
    order). The units, sorted by (model, sample id, epoch), are shuffled with the
    "|grader-order" seed, and a unit's id is its 1-based position in that pooled order.
    The grader evals take contiguous runs of positions in roughly equal parts. The drawn
    ids, sorted, are shuffled with the "|calibrator-order" seed for the calibrator eval.
    So no eval name, sample id or ledger key names a worker model.
  - Grader messages (unit_messages, messages_sha256). The acceptance-look recipe on the
    evaluation split, with the case file rebuilt and cross-checked against the worker
    sample's input. The plan keeps each unit's message hash, never the messages.
  - The attempt ledger (ledger_path, new_ledger, read_ledger, ledger_lock_path), under
    the git common directory. The reply of record of a unit is its first completed judge
    call in any attempt log (D-R23, reply of record); role_calls and reply_record read
    those calls. A unit asked three times with no completed call gets a failed judgment
    with no reply (failed_judgment_record, D-R23, failed judgments). The ledger holds
    per-record content, like the attempt logs, so it is never opened before the
    second-look freeze (D-R23, sealed folder).
  - Folder and file IO (check_sealed_dir, check_job_folders, write_new_file,
    write_or_match_bytes, write_sealed, write_or_match_sealed). Sealed files are written
    all or nothing in canonical bytes, outside the repository, in a folder named _sealed,
    and are never overwritten with other bytes. The attempt logs sit in a folder inside
    _sealed.
  - Pooled counts (pooled_counts), the only counts the evidence file holds before the
    second-look freeze.
  - Start checks (start_checks, look_preconditions, archive_record, check_eval_roles).

Nothing here prints, and nothing here returns a model id beside a count: every count is
pooled over all worker logs together. Error text names a unit by its opaque id and a
worker log by its file name, never by a worker model id.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import re
import sys
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import casefile, dataset, grader, regression, reliability, roster, rubric  # noqa: E402
from icfr_agent_evals import scoring, solver, trials  # noqa: E402
from icfr_agent_evals import grader_prompt as gp  # noqa: E402
from icfr_agent_evals import review_facts as rf  # noqa: E402
from icfr_agent_evals.reliability import GateError  # noqa: E402


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
rp = rr.load_review_packet()
pr = rr.load_pilot_report()

# --- Fixed constants -------------------------------------------------------------------------

RUN_LABEL = "2026-11-21"
RUN_COMMIT = "0c745021275f67d8a3349f0ff0e71bada299b07b"   # worker run commit
GRADER_STEP = 2
WORKER_EPOCHS = 4
EVAL_SPLIT = "eval"
EVAL_COMPOSITION: dict[str, int] = {"standard": 146, "edge": 30, "adversarial": 24}   # equals rp.EVAL_COMPOSITION
CALIBRATOR_PER_MODEL = 80
GRADER_PARTS = 4
WORKER_LOGS_DIR = REPO.parent / "eval-logs-2026-11-21"
# suggested --log-dir GRADING_ROOT/"_sealed"/"logs", --sealed-dir GRADING_ROOT/"_sealed"; the plumbing check uses GRADING_ROOT/"plumbing-logs"
GRADING_ROOT = REPO.parent / "grading-2026-11-21"
SEALED_FOLDER_NAME = "_sealed"                               # the sealed folder's basename must be exactly this
LEDGER_FOLDER = "icfr-grading"                               # under the git common dir
LEDGER_FILE = "grading-2026-11-21.json"
LEDGER_LOCK_FILE = "grading-2026-11-21.lock"                 # beside the ledger; one live job per ledger
PLUMBING_RECORD_GLOB = "calibrator-plumbing-*.json"          # same folder as the ledger
LOOK_FILE = REPO / "docs" / "evidence" / "calibration" / "acceptance-look-step-2.json"
EVIDENCE_DIR = REPO / "docs" / "evidence" / "grading"
EVIDENCE_FILE = "grading-2026-11-21.json"
ARCHIVE_FILE = EVIDENCE_DIR / "archive-2026-11-21.json"
ANCHORS_PREFIX = "405813382714b30a"

# Seed recipe (D-R23, draw and grading order): material = "|".join(log SHA-256s sorted by model id) + suffix
SEED_SUFFIX_CALIBRATOR = "|calibrator"
SEED_SUFFIX_GRADER_ORDER = "|grader-order"
SEED_SUFFIX_CALIBRATOR_ORDER = "|calibrator-order"
SEED_SUFFIX_BOOTSTRAP = "|bootstrap"
SEED_HEX_DIGITS = 16
SEED_SUFFIXES: tuple[str, ...] = (SEED_SUFFIX_CALIBRATOR, SEED_SUFFIX_GRADER_ORDER, SEED_SUFFIX_CALIBRATOR_ORDER,
                                  SEED_SUFFIX_BOOTSTRAP)

UNIT_ID_FORMAT = "u{:04d}"          # 1-based position in the pooled grader order
MAX_UNITS = 9_999                   # the most ids UNIT_ID_FORMAT gives at a fixed width
GRADER_EVAL_NAME = "grading_grader_part_{}"   # 1 to GRADER_PARTS
CALIBRATOR_EVAL_NAME = "grading_calibrator"
GRADER_PART_KEYS = tuple(f"grader-{i}" for i in range(1, GRADER_PARTS + 1))
CALIBRATOR_PART_KEY = "calibrator"

# Generation status (one per (model, sample, epoch) of the worker logs)
GEN_GRADEABLE, GEN_ERRORED, GEN_UNPARSED = "gradeable", "errored", "unparsed"
# Judge reply status (one per reply of record)
REPLY_PARSED, REPLY_UNPARSED, REPLY_REFUSED = "parsed", "unparsed", "refused"
REPLY_FAILED = "failed"                     # a failed judgment: no completed call after three asks, no reply
REPLY_STATUSES = (REPLY_PARSED, REPLY_UNPARSED, REPLY_REFUSED, REPLY_FAILED)
REFUSAL_STOP_REASON = "content_filter"
REFUSAL_ERROR = f"refused (stop reason {REFUSAL_STOP_REASON})"
REFUSAL_ERROR_CONTENT = "refused (refusal content)"
REFUSAL_ERROR_DETAILS = "refused (refusal stop details)"
REFUSAL_DETAIL_TYPES = ("refusal", "content_filter")
FAILED_JUDGMENT_ASKS = 3
FAILED_JUDGMENT_ERROR = ("no completed call in 3 attempt logs, the third in an eval of units at their third ask, "
                         "one sample at a time")

# Sealed files (exclusive create, canonical bytes, SHA-256 recorded)
SEALED_UNIT_MAP = "unit-map.json"
SEALED_DRAW_POOL = "draw-pool.json"
SEALED_DRAWN = "drawn.json"
SEALED_GRADER_REPLIES = "grader-replies.json"
SEALED_CALIBRATOR_REPLIES = "calibrator-replies.json"
SEALED_QUEUE = "queue.json"
SEALED_AGGREGATES = "aggregates.json"
PLAN_FILES = (SEALED_UNIT_MAP, SEALED_DRAW_POOL, SEALED_DRAWN)            # written before the first call
RESULT_FILES = (SEALED_GRADER_REPLIES, SEALED_CALIBRATOR_REPLIES, SEALED_QUEUE, SEALED_AGGREGATES)

PYTHON_PIN = "3.14.10"              # recorded beside platform.python_version(); not enforced

SEED_RULE_TEXT = ("the first 16 hex digits, read as an integer, of the SHA-256 of the worker-log SHA-256s sorted by "
                  "model id, joined with '|', followed by the suffix (|calibrator, |grader-order, |calibrator-order "
                  "or |bootstrap)")

# The draw rule in words, written into the sealed draw-pool file (D-R23, draw).
DRAW_RULE_TEXT = ("the case ids, sorted, are shuffled with the |calibrator seed; the worker runs other than the "
                  "calibrator's own model, sorted by model id, take the shuffled positions in rotation, "
                  f"{CALIBRATOR_PER_MODEL} each (1 to 80, 81 to 160, then 161 to 200 and 1 to 40); one gradeable "
                  "epoch is picked at random per (model, case) with the same generator; a (model, case) with no "
                  "gradeable epoch loses its row, with no refill and no draw made for it")

SAMPLING_CALLS: tuple[str, ...] = (
    "random.Random(calibrator_draw).shuffle(case_ids), case_ids sorted first",
    "the same generator: .choice(gradeable_epochs) per (model, case) in rotation order, no call when a (model, case) "
    "has no gradeable epoch",
    "random.Random(grader_order).shuffle(units), units sorted by (model, sample_id, epoch) first",
    "random.Random(calibrator_order).shuffle(drawn_uids), drawn uids sorted first",
    "random.Random(bootstrap).choices(range(clusters), k=clusters) per resample, one fresh generator per sub-dimension",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ARCHIVE_URL_PREFIX = "https://web.archive.org/web/"
# The paths whose change since the archived commit makes the archive record stale.
ARCHIVE_GUARDED_PATHS: tuple[str, ...] = ("scripts", "tests", "icfr_agent_evals", "DEVIATIONS.md",
                                          "docs/DECISIONS.md", "PRE-REGISTRATION-PHASE-1.md")


class Refused(Exception):
    """An input, plan, folder or git state the grading job will not use; exit 3."""


# --- Small helpers ---------------------------------------------------------------------------

def _now() -> str:
    return al._now()


def sha256_bytes(data: bytes) -> str:
    return al._sha256_bytes(data)


def sha256_file(path: Any) -> str:
    return al._sha256_file(pathlib.Path(path))


def canonical_bytes(obj: Any) -> bytes:
    """The one byte form of every sealed file and of the evidence file, so a recompute
    gives the same bytes: sorted keys, indent 1, UTF-8, one final LF."""
    return (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def inside(path: Any, root: Any) -> bool:
    return al._inside(path, pathlib.Path(root))


def messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    """SHA-256 of role/content message dicts as compact JSON (UTF-8, key order role then
    content)."""
    rows = [{"role": m["role"], "content": m["content"]} for m in messages]
    return sha256_bytes(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def chat_messages_sha256(chat: Sequence[Any]) -> str:
    """messages_sha256 over Inspect ChatMessages, each read as its role and text. A logged
    sample input is checked against the plan with it."""
    return messages_sha256([{"role": m.role, "content": m.text} for m in chat])


def input_text(sample_input: Any) -> str:
    """A sample input as text: a string as is, else the user messages' text joined by
    blank lines."""
    if isinstance(sample_input, str):
        return sample_input
    return "\n\n".join(str(getattr(m, "text", "") or "") for m in sample_input or () if getattr(m, "role", None) == "user")


def completion(sample: Any) -> str:
    """A sample's completion text, or "" when it has none."""
    out = getattr(sample, "output", None)
    text = getattr(out, "completion", None) if out is not None else None
    return "" if text is None else str(text)


def refusal_signal(output: Any) -> str | None:
    """Which refusal a model output records, as its error text, or None. Checked in this
    order: the refusal stop reason; stop details of the first choice whose type is a
    refusal; refusal content in the first choice's message (how an OpenAI refusal shows,
    with stop reason "stop"). The choices are read first. An output whose choices list is
    empty records no refusal, and its stop reason is never read, since an Inspect
    ModelOutput reads that from the first choice. Every read has a default, so an output
    with no choices never raises."""
    choices = getattr(output, "choices", None)
    if choices is not None and len(choices) == 0:
        return None
    try:
        stop_reason = getattr(output, "stop_reason", None)
    except (IndexError, AttributeError):
        stop_reason = None
    if stop_reason == REFUSAL_STOP_REASON:
        return REFUSAL_ERROR
    choices = choices or []
    choice = choices[0] if choices else None
    details = getattr(choice, "stop_details", None)
    if getattr(details, "type", None) in REFUSAL_DETAIL_TYPES:
        return REFUSAL_ERROR_DETAILS
    content = getattr(getattr(choice, "message", None), "content", None)
    if isinstance(content, list) and any(getattr(part, "refusal", None) is True for part in content):
        return REFUSAL_ERROR_CONTENT
    return None


def is_refusal(output: Any) -> bool:
    """True when a model output records a refusal (refusal_signal)."""
    return refusal_signal(output) is not None


# --- Seeds -----------------------------------------------------------------------------------

def rule_seed(log_shas_by_model: Mapping[str, str], suffix: str) -> int:
    """The seed fixed by rule for one use (SEED_RULE_TEXT). Refused: a suffix that is not
    one of the four seed suffixes, a SHA-256 that is not 64 lower-case hex digits, or
    fewer than 2 logs."""
    if suffix not in SEED_SUFFIXES:
        raise Refused(f"the seed suffix {suffix!r} is not one of {list(SEED_SUFFIXES)}")
    if len(log_shas_by_model) < 2:
        raise Refused(f"a seed needs at least 2 worker logs, got {len(log_shas_by_model)}")
    for sha in log_shas_by_model.values():
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise Refused("a worker-log SHA-256 is not 64 lower-case hex digits")
    material = "|".join(log_shas_by_model[m] for m in sorted(log_shas_by_model)) + suffix
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:SEED_HEX_DIGITS], 16)


def plan_seeds(log_shas_by_model: Mapping[str, str]) -> dict[str, int]:
    """The four seeds of the job: the calibrator draw, the grader order, the calibrator
    order and the case-cluster bootstrap."""
    return {"calibrator_draw": rule_seed(log_shas_by_model, SEED_SUFFIX_CALIBRATOR),
            "grader_order": rule_seed(log_shas_by_model, SEED_SUFFIX_GRADER_ORDER),
            "calibrator_order": rule_seed(log_shas_by_model, SEED_SUFFIX_CALIBRATOR_ORDER),
            "bootstrap": rule_seed(log_shas_by_model, SEED_SUFFIX_BOOTSTRAP)}


# --- Worker-log admission --------------------------------------------------------------------

@dataclass
class WorkerLog:
    """One admitted worker log: its file, hash, header facts and samples by (id, epoch)."""

    path: pathlib.Path
    file: str
    sha256: str
    eval_id: str
    model: str
    log: Any
    run_config: dict
    reasoning_mode: str
    revision_commit: str
    revision_dirty: bool | None
    header_grader_step: int
    resolved: tuple[str, ...]
    by_key: dict[tuple[str, int], Any] = field(default_factory=dict)


def find_worker_logs(folder: pathlib.Path, models: Sequence[str], *, repo: Any = REPO) -> list[pathlib.Path]:
    """One .eval file per worker model, in sorted model order. Each model's log sits in the
    subfolder named after the part of its id after the provider. Refused: a folder inside
    the given repository or this one, or a subfolder that does not hold exactly one .eval
    file."""
    folder = pathlib.Path(folder)
    if inside(folder, repo) or inside(folder, REPO):
        raise Refused("the worker log folder is inside the repository")
    paths = []
    for n, model in enumerate(sorted(models), start=1):
        parts = str(model).split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise Refused(f"worker model {n} in sorted order has no provider prefix, so its log folder is unknown")
        found = sorted((folder / parts[1]).glob("*.eval")) if (folder / parts[1]).is_dir() else []
        if len(found) != 1:
            raise Refused(f"worker log folder {n} holds {len(found)} .eval files, not one")
        paths.append(found[0])
    return paths


def read_worker_log(path: pathlib.Path) -> Any:
    """The full worker log (samples and events), read without writing it."""
    try:
        return rr.read_log(str(path), "worker")
    except rr.Refused as e:
        raise Refused(str(e)) from e


def eval_cases() -> dict[str, Any]:
    """The evaluation split's cases by id, rebuilt with build_dataset (a pure build, no
    call)."""
    return {str(s.id): s for s in dataset.build_dataset(split=EVAL_SPLIT)}


def composition_gaps(samples: Sequence[Any], composition: Mapping[str, int]) -> list[str]:
    """How these samples depart from the given composition of case kinds; [] when they hold
    exactly that composition. A case whose epochs name different kinds, or no kind, is a
    gap."""
    kinds: dict[str, set] = {}
    for s in samples:
        kinds.setdefault(str(s.id), set()).add(trials._case_kind(s))
    gaps = [f"case {sid} names case kinds {sorted(str(k) for k in ks)}" for sid, ks in sorted(kinds.items())
            if len(ks) != 1 or None in ks]
    counts: dict[str, int] = {}
    for ks in kinds.values():
        if len(ks) == 1 and None not in ks:
            kind = next(iter(ks))
            counts[kind] = counts.get(kind, 0) + 1
    want_total = sum(composition.values())
    if len(kinds) != want_total:
        gaps.append(f"it holds {len(kinds)} cases, not {want_total}")
    for kind in sorted(set(composition) | set(counts)):
        if counts.get(kind, 0) != composition.get(kind, 0):
            gaps.append(f"it holds {counts.get(kind, 0)} {kind} cases, not {composition.get(kind, 0)}")
    return gaps


def _model_events(log: Any, role: str | None):
    for sample in getattr(log, "samples", None) or ():
        for event in getattr(sample, "events", None) or ():
            if getattr(event, "event", None) == "model" and getattr(event, "role", None) == role:
                yield sample, event


def resolved_strings(log: Any) -> tuple[str, ...]:
    """The distinct non-empty resolved model strings of the evaluated model's calls (role
    None), sorted."""
    seen = set()
    for _, event in _model_events(log, None):
        text = str(getattr(getattr(event, "output", None), "model", None) or "")
        if text:
            seen.add(text)
    return tuple(sorted(seen))


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def admit_worker_log(path: pathlib.Path, log: Any, *, cases: Mapping[str, Any],
                     composition: Mapping[str, int] = EVAL_COMPOSITION,
                     models: Sequence[str] = roster.EVALUATED_MODELS, run_commit: str = RUN_COMMIT) -> WorkerLog:
    """Admit one worker log of the run, or refuse it. The checks run in this order: status
    success with samples; the evaluation split with the unlock; the case composition; a
    whole run; epochs 1 to 4; not the template baseline; the section 5 run configuration;
    one reasoning mode sent; the locked worker prompt; header grader_step 1; an evaluated
    model; the run commit; the split dataset hash; every sample's input and metadata
    against the rebuilt case; and a worker-output parse that agrees with the scorers'
    format-break record on every epoch without a sample error. The log is never written."""
    path = pathlib.Path(path)

    def refuse(reason: str) -> Refused:
        return Refused(f"worker log {path.name}: {reason}")

    # 1. A finished log read in full.
    if getattr(log, "status", None) != "success":
        raise refuse(f"status is {getattr(log, 'status', None)!r}, not 'success'")
    if getattr(log, "samples", None) is None:
        raise refuse("the log has no samples; read it in full, not header_only")
    # 2. The evaluation split, unlocked.
    try:
        split = rp.log_split(log, path.name)
    except rp.Refused as e:
        raise refuse(str(e)) from e
    if split != EVAL_SPLIT:
        raise refuse(f"the log is on the {split!r} split, not the evaluation split")
    # 3. The case composition of section 4.
    gaps = composition_gaps(log.samples, composition)
    if gaps:
        raise refuse(f"the cases differ from the evaluation split's composition: {'; '.join(gaps[:5])}")
    # 4. A whole run.
    try:
        regression.refuse_partial_run(log)
    except GateError as e:
        raise refuse(str(e)) from e
    # 5. Epochs 1 to k on every sample.
    ran = getattr(log.eval.config, "epochs", None)
    declared = (log.eval.task_args or {}).get("epochs")
    if ran != WORKER_EPOCHS or (declared is not None and declared != ran):
        raise refuse(f"it ran epochs {ran!r} (task argument {declared!r}), not {WORKER_EPOCHS}")
    epochs_of: dict[str, list[int]] = {}
    for s in log.samples:
        epochs_of.setdefault(str(s.id), []).append(int(s.epoch))
    for sid, eps in sorted(epochs_of.items()):
        if sorted(eps) != list(range(1, WORKER_EPOCHS + 1)):
            raise refuse(f"sample {sid} holds epochs {sorted(eps)}, not 1 to {WORKER_EPOCHS}")
    # 6. A worker run, not the template baseline.
    if reliability.is_template_baseline_log(log):
        raise refuse("it is the template baseline log, which is not graded")
    # 7. The section 5 run configuration.
    try:
        run_config = roster.check_run_config(log, expected_samples=None, expected_epochs=WORKER_EPOCHS,
                                             allow_batch=True)
    except roster.RunConfigError as e:
        raise refuse(str(e)) from e
    # 8. One reasoning mode sent.
    try:
        mode = rr.reasoning_mode(log, path.name)
    except rr.Refused as e:
        raise refuse(str(e)) from e
    metadata = dict(getattr(log.eval, "metadata", None) or {})
    # 9. The locked worker prompt.
    if metadata.get("worker_prompt_sha256") != solver.WORKER_PROMPT_SHA256:
        raise refuse("its worker_prompt_sha256 is not the locked worker prompt's")
    # 10. The header grader step (D-R23, header step).
    step = (log.eval.task_args or {}).get("grader_step")
    if not _plain_int(step) or step != 1:
        raise refuse(f"its header task_args record grader_step {step!r}, not 1")
    # 11. An evaluated model of the roster.
    if str(log.eval.model) not in models:
        raise refuse("its model is not one of the evaluated models")
    # 12. The run commit.
    revision = getattr(log.eval, "revision", None)
    commit = str(getattr(revision, "commit", "") or "") if revision is not None else ""
    if revision is None or not commit:
        raise refuse("it records no revision commit")
    if not rr._same_commit(commit, run_commit):
        raise refuse(f"its revision commit {commit[:12]} is not the run commit {str(run_commit)[:12]}")
    dirty = getattr(revision, "dirty", None)
    # 13. The split dataset hash.
    if metadata.get("split_dataset_sha256") != casefile.dataset_sha256(list(cases.values())):
        raise refuse("its split_dataset_sha256 differs from the rebuilt evaluation split")
    # 14. Every sample against its rebuilt case.
    for s in log.samples:
        sid, epoch = str(s.id), int(s.epoch)
        case = cases.get(sid)
        if case is None:
            raise refuse(f"sample {sid} epoch {epoch} is not a case of the rebuilt evaluation split")
        if input_text(s.input) != case.input:
            raise refuse(f"sample {sid} epoch {epoch}: its input differs from the rebuilt case")
        if (json.dumps(s.metadata, sort_keys=True, default=str)
                != json.dumps(case.metadata, sort_keys=True, default=str)):
            raise refuse(f"sample {sid} epoch {epoch}: its metadata differs from the rebuilt case")
    # 15. Parse consistency with the scorers' FM-000 record.
    for s in log.samples:
        if getattr(s, "error", None) is not None:
            continue
        parses = scoring.parse_worker_output(completion(s)).ok
        if parses != (not rr._parse_failure(s)):
            raise refuse(f"sample {s.id} epoch {s.epoch}: the worker output {'parses' if parses else 'does not parse'}, "
                         f"but the scorers recorded {'a' if parses else 'no'} format break")
    return WorkerLog(path=path, file=path.name, sha256=sha256_file(path), eval_id=str(log.eval.eval_id),
                     model=str(log.eval.model), log=log, run_config=run_config, reasoning_mode=mode,
                     revision_commit=commit, revision_dirty=dirty, header_grader_step=step,
                     resolved=resolved_strings(log),
                     by_key={(str(s.id), int(s.epoch)): s for s in log.samples})


def admit_worker_logs(paths: Sequence[pathlib.Path], *, cases: Mapping[str, Any] | None = None,
                      composition: Mapping[str, int] = EVAL_COMPOSITION,
                      models: Sequence[str] = roster.EVALUATED_MODELS, run_commit: str = RUN_COMMIT,
                      read=read_worker_log) -> list[WorkerLog]:
    """Read and admit every worker log (admit_worker_log), then refuse unless there is one
    log per evaluated model and no other, the eval ids and file hashes are distinct, the
    logs carry one Task version and dataset content, and every log holds the same sample
    ids. The number of logs is checked before any log is read. Returns the logs sorted by
    model id."""
    if len(paths) != len(models):
        raise Refused(f"{len(paths)} worker logs given for {len(models)} evaluated models")
    if cases is None:
        cases = eval_cases()
    logs = [admit_worker_log(pathlib.Path(p), read(pathlib.Path(p)), cases=cases, composition=composition,
                             models=models, run_commit=run_commit) for p in paths]
    by_model: dict[str, WorkerLog] = {}
    for w in logs:
        if w.model in by_model:
            raise Refused(f"worker logs {by_model[w.model].file} and {w.file} record one model; one log per model")
        by_model[w.model] = w
    if set(by_model) != set(models):
        raise Refused(f"the worker logs cover {len(set(by_model) & set(models))} of the {len(set(models))} "
                      f"evaluated models")
    ids: dict[str, str] = {}
    for w in logs:
        if w.eval_id in ids:
            raise Refused(f"worker logs {ids[w.eval_id]} and {w.file} share one eval id")
        ids[w.eval_id] = w.file
    shas: dict[str, str] = {}
    for w in logs:
        if w.sha256 in shas:
            raise Refused(f"worker logs {shas[w.sha256]} and {w.file} have the same SHA-256")
        shas[w.sha256] = w.file
    try:
        pr.check_same_content({w.file: w.log for w in logs})
    except pr.Refused as e:
        raise Refused(str(e)) from e
    first = logs[0] if logs else None
    if first is not None:
        want = {str(s.id) for s in first.log.samples}
        for w in logs[1:]:
            if {str(s.id) for s in w.log.samples} != want:
                raise Refused(f"worker logs {first.file} and {w.file} hold different sample ids")
    return sorted(logs, key=lambda w: w.model)


# --- Units, draw and order -------------------------------------------------------------------

def classify_generation(sample: Any) -> str:
    """errored when the epoch ended in a sample error, even if its completion parses (D-R23, units);
    unparsed when the worker output does not parse (FM-000); gradeable otherwise."""
    if getattr(sample, "error", None) is not None:
        return GEN_ERRORED
    if not scoring.parse_worker_output(completion(sample)).ok:
        return GEN_UNPARSED
    return GEN_GRADEABLE


def build_generations(worker_logs: Sequence[WorkerLog], cases: Mapping[str, Any]) -> list[dict]:
    """One generation per (model, sample, epoch) of the admitted logs, sorted by (model,
    sample id, epoch), with its case kind, status and whether it is a defect epoch of the
    D-R20 sensitivity row. No unit id yet."""
    out = []
    for w in worker_logs:
        defect = {(str(sid), int(ep)) for sid, ep in trials.defect_epochs(w.log)}
        for (sid, epoch), s in w.by_key.items():
            out.append({"model": w.model, "sample_id": sid, "epoch": epoch,
                        "case_kind": str(cases[sid].metadata["truth"]["case_kind"]),
                        "status": classify_generation(s), "defect_epoch": (sid, epoch) in defect, "uid": None})
    out.sort(key=lambda g: (g["model"], g["sample_id"], g["epoch"]))
    return out


def assign_unit_ids(generations: Sequence[dict], seed: int) -> dict[str, dict]:
    """Sort the gradeable generations by (model, sample id, epoch), then shuffle them with
    the grader-order seed; a unit's id is its 1-based position in that pooled order
    (UNIT_ID_FORMAT). The sort is done here, so the ids do not depend on the caller's
    order. The id is written back into each gradeable generation. Returns the units in id
    order."""
    gradeable = sorted((g for g in generations if g["status"] == GEN_GRADEABLE),
                       key=lambda g: (g["model"], g["sample_id"], g["epoch"]))
    if len(gradeable) > MAX_UNITS:
        raise Refused(f"{len(gradeable)} units, more than the {MAX_UNITS} unit ids allow")
    random.Random(seed).shuffle(gradeable)
    units: dict[str, dict] = {}
    for p, g in enumerate(gradeable, start=1):
        uid = UNIT_ID_FORMAT.format(p)
        g["uid"] = uid
        units[uid] = {"uid": uid, "model": g["model"], "sample_id": g["sample_id"], "epoch": g["epoch"],
                      "case_kind": g["case_kind"], "defect_epoch": g["defect_epoch"]}
    return units


def split_parts(uids: Sequence[str], parts: int = GRADER_PARTS) -> dict[str, list[str]]:
    """The grader parts: the ids in ascending order, cut into contiguous parts whose sizes
    differ by at most one, the larger parts first. Keys grader-1 to grader-N
    (GRADER_PART_KEYS at the default of 4 parts)."""
    if not _plain_int(parts) or parts < 1:
        raise Refused(f"the grader evals take a positive number of parts, not {parts!r}")
    if len(uids) < parts:
        raise Refused(f"{len(uids)} units cannot fill {parts} grader parts")
    ordered = sorted(uids)
    q, r = divmod(len(ordered), parts)
    out: dict[str, list[str]] = {}
    start = 0
    for i in range(1, parts + 1):
        size = q + (1 if i <= r else 0)
        out[f"grader-{i}"] = ordered[start:start + size]
        start += size
    return out


def draw_calibrator_sample(generations: Sequence[dict], *, seed: int, calibrator_model: str = roster.CALIBRATOR,
                           per_model: int = CALIBRATOR_PER_MODEL) -> dict:
    """The calibrator sample by case rotation (D-R23, draw). The case ids, sorted, are
    shuffled with the seed. The models other than the calibrator's own, sorted, take
    per_model shuffled positions each in rotation, wrapping round the case list. For each
    (model, case), in model order and then rotation order, one gradeable epoch is picked
    with the same generator; a (model, case) with no gradeable epoch is lost, with no
    refill and no draw. The pool is every gradeable generation outside the calibrator's
    own model. Unit ids must be assigned first, and the calibrator's own model must be
    among the generation models."""
    by_key = {(g["model"], g["sample_id"], g["epoch"]): g for g in generations}
    if any(g["status"] == GEN_GRADEABLE and not g.get("uid") for g in generations):
        raise Refused("the units have no ids yet; assign_unit_ids runs before the draw")
    if calibrator_model not in {g["model"] for g in generations}:
        raise Refused("the calibrator's own model is not among the worker runs, so it cannot be set aside")
    case_ids = sorted({g["sample_id"] for g in generations})
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    models = sorted({g["model"] for g in generations} - {calibrator_model})
    if not _plain_int(per_model) or per_model < 1:
        raise Refused(f"the draw takes a positive number of cases per model, not {per_model!r}")
    if per_model > len(case_ids):
        raise Refused(f"the draw takes {per_model} cases per model, more than the {len(case_ids)} cases")
    if not models:
        raise Refused("no worker run is left for the calibrator sample once its own model is set aside")
    gradeable: dict[tuple[str, str], list[int]] = {}
    for g in generations:
        if g["status"] == GEN_GRADEABLE:
            gradeable.setdefault((g["model"], g["sample_id"]), []).append(g["epoch"])
    rows, lost = [], []
    n = len(case_ids)
    for i, model in enumerate(models):
        for j in range(per_model):
            index = (i * per_model + j) % n
            case = case_ids[index]
            epochs = sorted(gradeable.get((model, case), ()))
            if not epochs:
                lost.append({"model": model, "sample_id": case, "position": index + 1})
                continue
            epoch = rng.choice(epochs)
            rows.append({"uid": by_key[(model, case, epoch)]["uid"], "model": model, "sample_id": case,
                         "epoch": epoch, "position": index + 1})
    pool = sorted(g["uid"] for g in generations if g["status"] == GEN_GRADEABLE and g["model"] != calibrator_model)
    return {"case_order": case_ids, "pool": pool, "rows": rows, "lost": lost}


def calibrator_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[str]:
    """The calibrator eval's order: the drawn ids, sorted, shuffled with the seed."""
    uids = sorted(r["uid"] for r in rows)
    random.Random(seed).shuffle(uids)
    return uids


def unit_messages(unit: Mapping[str, Any], worker_logs_by_model: Mapping[str, WorkerLog],
                  cases: Mapping[str, Any]) -> list[dict]:
    """The judge's messages for one unit, the acceptance-look recipe on the evaluation
    split: the parsed narrative, the case input, the shared facts sheet and the default
    prompt settings. The logged input is cross-checked against the rebuilt case."""
    uid = unit.get("uid")
    w = worker_logs_by_model.get(unit["model"])
    s = w.by_key.get((unit["sample_id"], unit["epoch"])) if w is not None else None
    case = cases.get(unit["sample_id"])
    if s is None or case is None:
        raise Refused(f"unit {uid}: its generation is not in the admitted worker logs and the rebuilt cases")
    if input_text(s.input) != case.input:
        raise Refused(f"unit {uid}: the logged input differs from the rebuilt case")
    parse = scoring.parse_worker_output(completion(s))
    if not parse.ok:
        raise Refused(f"unit {uid}: the worker output does not parse, so it is never graded")
    meta = case.metadata
    return gp.build_messages(str(parse.parsed["narrative"]), case.input, rf.case_facts(meta),
                             rf.output_facts(meta, parse.parsed), gp.PromptSettings())


def build_plan(worker_logs: Sequence[WorkerLog], cases: Mapping[str, Any], *,
               calibrator_model: str = roster.CALIBRATOR, per_model: int = CALIBRATOR_PER_MODEL,
               parts: int = GRADER_PARTS) -> dict:
    """The whole plan before any call: generations, units with their message hashes, the
    grader parts and the calibrator order, the draw, the seeds and the pooled counts. The
    counts also hold grader_prompt_chars, the characters of every grader message of every
    unit, so the plan-only run can size the Batch queue. It reads no grader or calibrator
    output."""
    shas = {w.model: w.sha256 for w in worker_logs}
    seeds = plan_seeds(shas)
    generations = build_generations(worker_logs, cases)
    units = assign_unit_ids(generations, seeds["grader_order"])
    by_model = {w.model: w for w in worker_logs}
    prompt_chars = 0
    for uid, unit in units.items():
        messages = unit_messages(unit, by_model, cases)
        prompt_chars += sum(len(m["content"]) for m in messages)
        unit["messages_sha256"] = messages_sha256(messages)
    draw = draw_calibrator_sample(generations, seed=seeds["calibrator_draw"], calibrator_model=calibrator_model,
                                  per_model=per_model)
    parts_ = split_parts(list(units), parts)
    parts_[CALIBRATOR_PART_KEY] = calibrator_order(draw["rows"], seeds["calibrator_order"])
    counts = {"generations": len(generations),
              "skipped_errored": sum(1 for g in generations if g["status"] == GEN_ERRORED),
              "skipped_unparsed": sum(1 for g in generations if g["status"] == GEN_UNPARSED),
              "units": len(units), "rows_drawn": len(draw["rows"]), "rows_lost": len(draw["lost"]),
              "grader_prompt_chars": prompt_chars}
    return {"generations": generations, "units": units, "parts": parts_, "draw": draw, "seeds": seeds,
            "seed_material_shas": [shas[m] for m in sorted(shas)], "counts": counts}


def sealed_plan_bodies(plan: Mapping[str, Any], worker_logs: Sequence[WorkerLog]) -> dict[str, dict]:
    """The three sealed plan files, written before the first call: the unit map, the draw
    pool and the drawn rows."""
    logs = [{"model": w.model, "file": w.file, "sha256": w.sha256, "eval_id": w.eval_id,
             "revision_commit": w.revision_commit, "revision_dirty": w.revision_dirty,
             "reasoning_mode": w.reasoning_mode, "header_grader_step": w.header_grader_step,
             "resolved": list(w.resolved)} for w in sorted(worker_logs, key=lambda w: w.model)]
    draw = plan["draw"]
    return {
        SEALED_UNIT_MAP: {"run": RUN_LABEL, "worker_logs": logs, "generations": plan["generations"],
                          "units": plan["units"]},
        SEALED_DRAW_POOL: {"run": RUN_LABEL, "rule": DRAW_RULE_TEXT, "seed": plan["seeds"]["calibrator_draw"],
                           "seed_material_shas": plan["seed_material_shas"], "pool": draw["pool"]},
        SEALED_DRAWN: {"run": RUN_LABEL, "case_order": draw["case_order"], "rows": draw["rows"],
                       "lost": draw["lost"], "calibrator_order": plan["parts"][CALIBRATOR_PART_KEY]},
    }


# --- Sealed folder IO ------------------------------------------------------------------------

def check_sealed_dir(path: Any, repo: Any) -> pathlib.Path:
    """The sealed folder, resolved. Refused inside the repository or when its name is not
    exactly _sealed (a standing never-open name, D-R23, sealed folder)."""
    resolved = pathlib.Path(path).resolve()
    if inside(resolved, repo) or inside(resolved, REPO):
        raise Refused("the sealed folder is inside the repository")
    if resolved.name != SEALED_FOLDER_NAME:
        raise Refused(f"the sealed folder must be named {SEALED_FOLDER_NAME}, not {resolved.name!r}")
    return resolved


def _nested(a: pathlib.Path, b: pathlib.Path) -> bool:
    """True when a equals, contains or sits inside b (both resolved)."""
    return a == b or b in a.parents or a in b.parents


def check_job_folders(*, log_dir: Any, sealed_dir: Any, worker_logs: Any, fresh: bool,
                      plumbing_log_dir: Any = None) -> None:
    """The grading job's folder rules (D-R23, sealed folder). Every path is resolved, and
    the rules run in this order. The attempt logs sit in a folder inside the sealed
    folder (a path component named _sealed), so they are never opened before the freeze,
    and that folder is not the sealed folder itself. Neither folder equals, contains or
    sits inside the worker log folder. At a fresh start the log folder holds no .eval file
    and is not the plumbing check's log folder, so a resume never meets a log of another
    run. Refused otherwise; no path is printed."""
    log_dir = pathlib.Path(log_dir).resolve()
    sealed_dir = pathlib.Path(sealed_dir).resolve()
    if SEALED_FOLDER_NAME not in log_dir.parts:
        raise Refused("--log-dir must sit inside the sealed folder (a path component named _sealed), so the attempt "
                      "logs are never opened before the freeze")
    if log_dir == sealed_dir:
        raise Refused("--log-dir must be a folder inside the sealed folder, not the sealed folder itself")
    if worker_logs is not None:
        workers = pathlib.Path(worker_logs).resolve()
        if _nested(log_dir, workers) or _nested(sealed_dir, workers):
            raise Refused("--log-dir and --sealed-dir must not equal, contain or sit inside the worker log folder")
    if fresh and log_dir.is_dir() and any(p.is_file() for p in log_dir.glob("*.eval")):
        raise Refused("the grading log folder already holds .eval files; a fresh start needs an empty log folder")
    if fresh and plumbing_log_dir is not None and pathlib.Path(plumbing_log_dir).resolve() == log_dir:
        raise Refused("the grading log folder is the plumbing check's log folder; give each its own folder")


def write_new_file(path: pathlib.Path, data: bytes) -> None:
    """Write a new file all or nothing: the bytes go to a temporary file beside it, which is
    flushed, synced and then hard-linked to the name, so a kill never leaves a part-written
    file under the name. An existing file raises FileExistsError. On a file system with no
    hard links the file is created exclusively and written. The temporary file is always
    removed. The parent folder is created."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
        except OSError:
            with open(path, "xb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_or_match_bytes(path: pathlib.Path, data: bytes, *, what: str) -> str:
    """Write a new file (write_new_file), or accept an existing one only when its bytes equal
    these; returns the SHA-256. A file with other bytes is never overwritten."""
    path = pathlib.Path(path)
    try:
        write_new_file(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise Refused(f"the {what} {path.name} exists and differs from the recomputed one; "
                          f"it is never overwritten") from None
    return sha256_bytes(data)


def write_sealed(sealed_dir: Any, name: str, body: Mapping[str, Any]) -> str:
    """Write one new sealed file in canonical bytes, all or nothing; returns their SHA-256.
    An existing file raises FileExistsError."""
    data = canonical_bytes(body)
    write_new_file(pathlib.Path(sealed_dir) / name, data)
    return sha256_bytes(data)


def write_or_match_sealed(sealed_dir: Any, name: str, body: Mapping[str, Any]) -> str:
    """Write a sealed file, or accept an existing one only when its bytes equal the
    recomputed ones; returns the SHA-256. A sealed file is never overwritten."""
    return write_or_match_bytes(pathlib.Path(sealed_dir) / name, canonical_bytes(body), what="sealed file")


def sealed_plan_hashes(plan: Mapping[str, Any], worker_logs: Sequence[WorkerLog]) -> dict[str, str]:
    """The SHA-256 of each sealed plan file's canonical bytes, keyed by file name, without
    writing any file. The plan-only run prints them for the archive record."""
    return {name: sha256_bytes(canonical_bytes(body))
            for name, body in sealed_plan_bodies(plan, worker_logs).items()}


def sealed_hashes(sealed_dir: Any, names: Sequence[str]) -> dict[str, str]:
    """The SHA-256 of each named sealed file; a missing file is refused."""
    out = {}
    for name in names:
        path = pathlib.Path(sealed_dir) / name
        if not path.is_file():
            raise Refused(f"the sealed file {name} is missing")
        out[name] = sha256_file(path)
    return out


# --- Ledger ----------------------------------------------------------------------------------

def ledger_path(repo: Any = REPO, ledger_dir: Any = None) -> pathlib.Path:
    """The attempt ledger: <git common dir>/icfr-grading/grading-2026-11-21.json, seen by
    every checkout of the repository."""
    if ledger_dir is not None:
        return pathlib.Path(ledger_dir) / LEDGER_FILE
    try:
        return al.git_common_dir(pathlib.Path(repo)) / LEDGER_FOLDER / LEDGER_FILE
    except al.Refused as e:
        raise Refused(str(e)) from e


def ledger_lock_path(ledger_file: pathlib.Path) -> pathlib.Path:
    """The lock file beside the ledger. A live job creates it before anything else and
    removes it at the end, so two live jobs never share one ledger (D-R23, ledger)."""
    return pathlib.Path(ledger_file).with_name(LEDGER_LOCK_FILE)


def read_ledger(path: pathlib.Path) -> dict:
    """The ledger as a JSON object. Refused when the file cannot be read, is not UTF-8 JSON,
    or is not a JSON object. The file's content is never printed."""
    path = pathlib.Path(path)
    try:
        body = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise Refused(f"the grading ledger {path.name} cannot be read ({type(e).__name__}); "
                      f"the author decides how to proceed") from None
    if not isinstance(body, dict):
        raise Refused(f"the grading ledger {path.name} cannot be read ({type(body).__name__}); "
                      f"the author decides how to proceed")
    return body


def create_ledger(path: pathlib.Path, body: Mapping[str, Any]) -> None:
    """Create the ledger with exclusive create; an existing ledger raises FileExistsError."""
    al.create_ledger(pathlib.Path(path), body)


def save_ledger(path: pathlib.Path, body: Mapping[str, Any]) -> None:
    al.save_ledger(pathlib.Path(path), body)


def job_id(provenance: Mapping[str, Any]) -> str:
    """The job's id: the first 16 hex digits of the SHA-256 of its canonical provenance."""
    return sha256_bytes(canonical_bytes(provenance))[:16]


def new_ledger(*, provenance: Mapping[str, Any], plan: Mapping[str, Any], sealed_plan_hashes: Mapping[str, str],
               log_dir: Any, sealed_dir: Any) -> dict:
    """The ledger body written before the first call: provenance, seeds and their rule,
    the exact sampling calls, the sealed plan hashes, the parts and both folders, with
    empty lists for the attempt logs, Batch ids, failures and rejected logs."""
    return {
        "ledger": "full-run grading attempt ledger", "run": RUN_LABEL, "job_id": job_id(provenance),
        "started": _now(), "provenance": dict(provenance), "seeds": dict(plan["seeds"]), "seed_rule": SEED_RULE_TEXT,
        "sampling_calls": list(SAMPLING_CALLS), "sealed_plan_files": dict(sealed_plan_hashes),
        "parts": {k: list(v) for k, v in plan["parts"].items()},
        "log_dir": str(pathlib.Path(log_dir).resolve()), "sealed_dir": str(pathlib.Path(sealed_dir).resolve()),
        "attempts": [{"attempt": 1, "started": _now(), "resume": False}],
        "attempt_logs": [], "replies_of_record": {}, "batch_ids": [], "failures": [], "rejected_logs": [],
    }


# --- Calls in a log --------------------------------------------------------------------------

def is_completed_call(event: Any) -> bool:
    """True only for a model event with no error, not pending, and with a completed time.
    Inspect sets the completed time after a successful call and clears pending when the
    call ends, so an in-flight event left in an interrupted log is never a completed call."""
    return (not getattr(event, "error", None) and not getattr(event, "pending", None)
            and getattr(event, "completed", None) is not None)


def role_calls(log: Any, role: str, asked: Sequence[str], *, require_all: bool) -> dict:
    """The completed calls of one judge role in an attempt log, finished or not.

    A call is completed when is_completed_call holds; every other event of the role
    (errored, pending, or with no completed time) is skipped. Raises roster.RunConfigError
    when the log has no samples, a sample is not an asked unit or appears twice, a sample
    holds more than one completed call of the role or a skipped call after the completed
    one, or (require_all) an asked unit has no completed call. A skipped call before the
    completed one is counted in "skipped". A unit with no completed call is listed in
    "missing". "samples" and the view hold the samples with a completed call, each without
    its skipped calls (D-R23, reply of record)."""
    if getattr(log, "samples", None) is None:
        raise roster.RunConfigError("run config: the log has no samples; read it in full, not header_only")
    want = {str(u) for u in asked}
    seen: set[str] = set()
    completed: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    views: list[Any] = []
    skipped = 0
    for s in log.samples:
        uid = str(s.id)
        if uid not in want:
            raise roster.RunConfigError(f"run config: the log holds unit {uid}, which this attempt did not ask")
        if uid in seen:
            raise roster.RunConfigError(f"run config: unit {uid} appears twice in the log")
        seen.add(uid)
        events = list(getattr(s, "events", None) or ())
        calls = [(i, e) for i, e in enumerate(events)
                 if getattr(e, "event", None) == "model" and getattr(e, "role", None) == role]
        done = [i for i, e in calls if is_completed_call(e)]
        other = [i for i, e in calls if not is_completed_call(e)]
        if len(done) > 1:
            raise roster.RunConfigError(f"run config: unit {uid} holds {len(done)} completed {role} calls, not one")
        if not done:
            continue
        if any(i > done[0] for i in other):
            raise roster.RunConfigError(f"run config: unit {uid} holds an errored or unfinished {role} call after "
                                        f"its completed one")
        skipped += len(other)
        drop = set(other)
        completed[uid] = events[done[0]]
        samples[uid] = s
        views.append(SimpleNamespace(id=s.id, epoch=getattr(s, "epoch", 1), error=getattr(s, "error", None),
                                     output=getattr(s, "output", None), input=getattr(s, "input", None),
                                     events=[e for i, e in enumerate(events) if i not in drop]))
    missing = [str(u) for u in asked if str(u) not in completed]
    if require_all and missing:
        raise roster.RunConfigError(f"run config: {len(missing)} asked units have no completed {role} call "
                                    f"({', '.join(missing[:5])})")
    view = SimpleNamespace(status=getattr(log, "status", None), samples=views, eval=getattr(log, "eval", None),
                           location=getattr(log, "location", None))
    return {"completed": completed, "samples": samples, "skipped": skipped, "missing": missing, "view": view}


def check_serving_recorded(sample: Any, *, source: str, provider: str) -> dict:
    """The serving endpoint a live judge call recorded in its transcript: an info event of
    the given source whose base URL, one trailing slash stripped, is the provider's exact
    first-party URL. Raises roster.RunConfigError otherwise. Offline calls record none."""
    want = roster.FIRST_PARTY_BASE_URLS.get(provider)
    if want is None:
        raise roster.RunConfigError(f"run config: provider {provider!r} has no first-party base URL")
    found = [e for e in getattr(sample, "events", None) or ()
             if getattr(e, "event", None) == "info" and getattr(e, "source", None) == source]
    if not found:
        raise roster.RunConfigError(f"run config: unit {sample.id} records no serving endpoint ({source})")
    endpoint = None
    for event in found:
        data = getattr(event, "data", None)
        serving = data.get("serving_endpoint") if isinstance(data, Mapping) else None
        url = serving.get("base_url") if isinstance(serving, Mapping) else None
        url = url[:-1] if isinstance(url, str) and url.endswith("/") else url
        if url != want:
            raise roster.RunConfigError(f"run config: unit {sample.id} recorded a serving endpoint that is not "
                                        f"the first-party {provider} URL")
        if endpoint is None:
            endpoint = dict(serving)
    return endpoint


def reply_record(event: Any, *, attempt: int, log_file: str) -> dict:
    """The reply of record of one completed judge call. A refusal (refusal_signal: the
    refusal stop reason, refusal stop details or refusal content) is recorded as refused,
    with the signal as its error, and never parsed; a reply gp.parse_reply cannot read is
    unparsed, with the parse error; a parsed reply keeps its levels, flags and reasons."""
    output = getattr(event, "output", None)
    text = str(getattr(output, "completion", None) or "")
    usage = getattr(output, "usage", None)
    usage_pair = [int(usage.input_tokens), int(usage.output_tokens)] if usage is not None else None
    base = {"attempt": attempt, "log_file": log_file, "usage": usage_pair}
    refusal = refusal_signal(output)
    if refusal is not None:
        return {"status": REPLY_REFUSED, "scores": {}, "low_confidence": {}, "reasons": {}, "reply": text,
                "error": refusal, **base}
    try:
        parsed = gp.parse_reply(text, gp.PromptSettings())
    except gp.GraderReplyError as e:
        return {"status": REPLY_UNPARSED, "scores": {}, "low_confidence": {}, "reasons": {}, "reply": text,
                "error": str(e), **base}
    return {"status": REPLY_PARSED,
            "scores": {d: parsed[d]["level"] for d in gp.DIMENSIONS},
            "low_confidence": {d: parsed[d]["low_confidence"] for d in gp.DIMENSIONS},
            "reasons": {d: parsed[d]["reason"] for d in gp.DIMENSIONS},
            "reply": None, "error": None, **base}


def failed_judgment_record(*, attempt: int, log_file: str) -> dict:
    """The reply of record of a failed judgment (D-R23, failed judgments): a unit asked in
    three attempt logs with no completed call, the third time in an eval of units at their
    third ask, one sample at a time. It has no reply, is never asked again, and is treated
    as a reply that does not parse. attempt and log_file name that third attempt log."""
    return {"status": REPLY_FAILED, "scores": {}, "low_confidence": {}, "reasons": {}, "reply": None,
            "error": FAILED_JUDGMENT_ERROR, "attempt": attempt, "log_file": log_file, "usage": None}


# --- Pooled counts ---------------------------------------------------------------------------

def _status_counts(replies: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    statuses = [r.get("status") for r in replies.values()]
    return {"replies_of_record": len(statuses), "parsed": statuses.count(REPLY_PARSED),
            "unparsed": statuses.count(REPLY_UNPARSED), "refused": statuses.count(REPLY_REFUSED),
            "failed": statuses.count(REPLY_FAILED)}


def _attempt_counts(ledger: Mapping[str, Any], role: str) -> dict[str, int]:
    entries = [e for e in ledger.get("attempt_logs") or () if e.get("role") == role]
    return {"errored_calls_skipped": sum(int(e.get("errored_calls_skipped") or 0) for e in entries),
            "attempt_logs": len(entries)}


def pooled_counts(plan: Mapping[str, Any], grader_replies: Mapping[str, dict],
                  calibrator_replies: Mapping[str, dict], ledger: Mapping[str, Any]) -> dict:
    """The evidence counts block. Every figure is over all worker logs together; the pool
    size, any per-model figure and any tag count are never in it. A failed judgment is
    counted under failed, and a pair is complete only when both replies parsed, so a
    failed calibrator reply is a dropped pair."""
    counts = plan["counts"]
    rows = plan["draw"]["rows"]
    complete = sum(1 for r in rows
                   if (grader_replies.get(r["uid"]) or {}).get("status") == REPLY_PARSED
                   and (calibrator_replies.get(r["uid"]) or {}).get("status") == REPLY_PARSED)
    return {
        "worker_logs": len(plan["seed_material_shas"]), "generations": counts["generations"],
        "skipped_errored": counts["skipped_errored"], "skipped_unparsed": counts["skipped_unparsed"],
        "units": counts["units"],
        "grader": {**_status_counts(grader_replies), **_attempt_counts(ledger, roster.GRADER_ROLE)},
        "calibrator": {"rows_drawn": counts["rows_drawn"], "rows_lost": counts["rows_lost"],
                       **_status_counts(calibrator_replies), **_attempt_counts(ledger, roster.CALIBRATOR_ROLE),
                       "pairs_complete": complete, "pairs_dropped": len(rows) - complete},
    }


# --- Start checks ----------------------------------------------------------------------------

def _git_out(repo: Any, *args: str) -> str:
    try:
        return al._git_out(pathlib.Path(repo), *args)
    except al.Refused as e:
        raise Refused(str(e)) from e


def _git_code(repo: Any, *args: str) -> int:
    try:
        return al._git(pathlib.Path(repo), *args).returncode
    except al.Refused as e:
        raise Refused(str(e)) from e


def _tracked_clean(path: pathlib.Path, repo: Any, what: str) -> str:
    """The repository-relative path of a file tracked in HEAD with no uncommitted change."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise Refused(f"the {what} {p.name} does not exist")
    try:
        rel = al._rel(p, pathlib.Path(repo))
    except al.Refused as e:
        raise Refused(f"the {what} {p.name} is not inside the repository") from e
    if rel not in set(_git_out(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", rel).splitlines()):
        raise Refused(f"the {what} {rel} is not tracked in HEAD")
    if _git_out(repo, "status", "--porcelain", "--", rel).strip():
        raise Refused(f"the {what} {rel} differs from HEAD")
    return rel


def _read_json(path: pathlib.Path, rel: str, what: str) -> Any:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"the {what} {rel} is not a JSON file ({e})") from e


def look_preconditions(look_file: pathlib.Path, repo: Any) -> dict:
    """Refuse unless the grader's acceptance look is committed and reads PASS for the
    grader of record: ladder step 2, the step 2 pin and generate config, a live look, the
    gp-1 prompt and its hash, the committed anchors, and pass 1 on the default settings.
    Reads only those keys."""
    rel = _tracked_clean(look_file, repo, "acceptance look file")
    body = _read_json(look_file, rel, "acceptance look file")
    if not isinstance(body, Mapping):
        raise Refused(f"the acceptance look file {rel} is not a JSON object")
    passes = body.get("passes")
    first = passes[0] if isinstance(passes, list) and passes and isinstance(passes[0], Mapping) else {}
    want_settings = al.PassSpec("pass_1", 1, gp.PromptSettings()).settings_record()
    anchors = rubric.llm_anchor_sha256()
    verdict = body.get("verdict")
    checks = [
        ("verdict.overall", verdict.get("overall") if isinstance(verdict, Mapping) else None, "PASS"),
        ("grader_step", body.get("grader_step"), GRADER_STEP),
        ("grader_model", body.get("grader_model"), grader.grader_pin(GRADER_STEP)),
        ("grader_generate_config", body.get("grader_generate_config"),
         roster.grader_generate_config_args(GRADER_STEP)),
        ("offline", body.get("offline"), False),
        ("grader_prompt_version", body.get("grader_prompt_version"), gp.PROMPT_VERSION),
        ("grader_prompt_sha256", body.get("grader_prompt_sha256"), gp.prompt_sha256()),
        ("anchors_sha256", body.get("anchors_sha256"), anchors),
        ("passes[0].name", first.get("name"), "pass_1"),
        ("passes[0].settings", first.get("settings"), want_settings),
    ]
    for key, got, want in checks:
        if type(got) is not type(want) or got != want:
            raise Refused(f"the acceptance look file {rel} does not hold {key} of the grader of record")
    if gp.PROMPT_VERSION != "gp-1":
        raise Refused(f"the prompt version is {gp.PROMPT_VERSION!r}, not gp-1")
    if not anchors.startswith(ANCHORS_PREFIX):
        raise Refused(f"the committed anchors hash does not start with {ANCHORS_PREFIX}")
    return {"file": rel, "sha256": sha256_file(look_file), "verdict": "PASS"}


def archive_record(archive_file: pathlib.Path, repo: Any) -> dict:
    """Refuse unless the archive record is committed and clean, holds the archived commit
    (40 hex), its Wayback Machine URL and date, the commit is an ancestor of HEAD, and no
    code, test, deviation, decision or pre-registration file changed since it. Extra keys
    are allowed. worker_log_shas (the worker-log SHA-256s, sorted, no duplicate) and
    plan_file_shas (one SHA-256 per sealed plan file) are returned when present and
    checked for shape; each is None when absent, and the job compares them with the logs
    and plan in use."""
    rel = _tracked_clean(archive_file, repo, "archive record")
    body = _read_json(archive_file, rel, "archive record")
    if not isinstance(body, Mapping):
        raise Refused(f"the archive record {rel} is not a JSON object")
    commit, url, date = body.get("commit"), body.get("url"), body.get("date")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise Refused(f"the archive record {rel} holds no 40-digit lower-case hex commit")
    if not isinstance(url, str) or not url.startswith(ARCHIVE_URL_PREFIX) or len(url) <= len(ARCHIVE_URL_PREFIX):
        raise Refused(f"the archive record {rel} holds no URL starting {ARCHIVE_URL_PREFIX}")
    try:
        ok_date = isinstance(date, str) and bool(_DATE_RE.fullmatch(date)) and bool(_dt.date.fromisoformat(date))
    except ValueError:
        ok_date = False
    if not ok_date:
        raise Refused(f"the archive record {rel} holds no date in the form YYYY-MM-DD")
    if _git_code(repo, "merge-base", "--is-ancestor", commit, "HEAD") != 0:
        raise Refused(f"the archived commit {commit[:12]} is not an ancestor of HEAD")
    changed = _git_out(repo, "diff", "--name-only", commit, "HEAD", "--", *ARCHIVE_GUARDED_PATHS).strip()
    if changed:
        names = changed.splitlines()
        raise Refused(f"{len(names)} guarded files changed since the archived commit {commit[:12]} "
                      f"({', '.join(names[:5])}); push and archive again")
    worker_log_shas = body.get("worker_log_shas")
    if worker_log_shas is not None:
        if (not isinstance(worker_log_shas, list)
                or not all(isinstance(s, str) and _SHA256_RE.fullmatch(s) for s in worker_log_shas)
                or worker_log_shas != sorted(set(worker_log_shas))):
            raise Refused(f"the archive record {rel} holds worker_log_shas that are not sorted SHA-256 values")
        worker_log_shas = list(worker_log_shas)
    plan_file_shas = body.get("plan_file_shas")
    if plan_file_shas is not None:
        if (not isinstance(plan_file_shas, Mapping) or set(plan_file_shas) != set(PLAN_FILES)
                or not all(isinstance(s, str) and _SHA256_RE.fullmatch(s) for s in plan_file_shas.values())):
            raise Refused(f"the archive record {rel} holds plan_file_shas that are not one SHA-256 per plan file")
        plan_file_shas = dict(plan_file_shas)
    return {"file": rel, "sha256": sha256_file(archive_file), "commit": commit, "url": url, "date": date,
            "worker_log_shas": worker_log_shas, "plan_file_shas": plan_file_shas}


def start_checks(repo: Any = REPO, *, run_commit: str = RUN_COMMIT, look_file: pathlib.Path | None = LOOK_FILE,
                 archive_file: pathlib.Path = ARCHIVE_FILE, folders: Mapping[str, Any] | None = None,
                 running_package: bool = True) -> dict:
    """The checks before the first live call: a clean tree, HEAD an ancestor of the local
    origin/main ref, the package as at the run commit (and the package that runs is the
    repository's), the acceptance look (when given), the archive record, and every folder
    outside the repository (folders None means no folder). Returns HEAD, origin/main, the
    run commit, the look and the archive record (archive_record's dict, unchanged)."""
    repo = pathlib.Path(repo)
    status = _git_out(repo, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        lines = status.strip().splitlines()
        raise Refused(f"git status is not clean ({len(lines)} entries, first {'; '.join(lines[:3])}); commit first")
    if _git_code(repo, "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main") != 0:
        raise Refused("HEAD is not an ancestor of the local origin/main ref; push first")
    gaps = list(pr.package_code_gaps(repo, run_commit))
    if running_package:
        gaps += list(pr.running_package_gap(repo))
    if gaps:
        raise Refused(f"the package is not as at the run commit ({len(gaps)} gaps: {'; '.join(gaps[:3])})")
    look = look_preconditions(pathlib.Path(look_file), repo) if look_file is not None else None
    archive = archive_record(pathlib.Path(archive_file), repo)
    for name, folder in (folders or {}).items():
        if folder is None:
            continue
        if inside(folder, repo) or inside(folder, REPO):
            raise Refused(f"the folder given as {name} is inside the repository")
    return {"head": _git_out(repo, "rev-parse", "HEAD").strip(),
            "origin_main": _git_out(repo, "rev-parse", "refs/remotes/origin/main").strip(),
            "run_commit": run_commit, "look": look, "archive": archive}


def check_eval_roles(role: str) -> None:
    """Inside a grading solver, before the call: the eval binds exactly one role, this one,
    to one model, and the evaluated model is none/none. Raises roster.RosterError
    otherwise."""
    from inspect_ai.model import model_roles
    from inspect_ai.model._model import active_model

    roles = model_roles()
    if set(roles) != {role} or isinstance(roles.get(role), list):
        raise roster.RosterError(f"one role per grading eval: this eval must bind the {role} role alone, to one model")
    if str(active_model()) != grader.NO_MODEL:
        raise roster.RosterError("the evaluated model is not none/none")
