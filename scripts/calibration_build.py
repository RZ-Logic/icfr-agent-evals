"""Build the calibration subset's blind grading sheet and sealed plan. NO API key, no model call.

Usage:
  python scripts/calibration_build.py --log PATH --log PATH --log PATH [--cases PATH]
      [--selection PATH] [--out DIR] [--max-tokens-pin N] [--force]

Pre-registration section 4 (calibration subset: 40 cases outside the evaluation and dev
splits, graded by hand and by the grader, stratified so every score level 0 to 3 appears
at least 4 times per LLM-graded sub-dimension, with seeded degraded responses where
natural outputs do not reach a level; seeds for narrative coverage are made two ways,
omission seeds and addition seeds) and section 7 (the grader gets one acceptance look on
the subset, which is never used for tuning). The targets need a margin above the floor
of 4: every level at least 6 times per sub-dimension, so that author grades a level away
from a target can still meet the floor at import. The floor of 4 stays the hard rule
on the author's grades (scripts/calibration_import.py). Decision D-R10 (C5: an output that did not parse is not graded; C6: seeds
follow section 4's wording; C7: the grader and the human see one shared facts sheet).

Design set by the author on 2026-11-17 (not in the pre-registration text): 40 cases of
the calibration split (data/calibration/calibration.jsonl); natural outputs from three
workers, Sonnet 5.5, Sonnet 5 and Haiku 5.5, each answering all 40 cases once (epochs 1)
under the locked worker prompt; one output per case is picked, and a picked output may
be degraded by one mechanical seed rule of scripts/judge_dev_build.py (RULES).

Inputs:
  --log        three worker logs on the calibration split, one per worker. Each must be
               status success, split calibration, hold all 40 cases at epoch 1, carry the
               locked worker_prompt_sha256 and pass roster.check_run_config for its model.
  --cases      the 40 calibration cases (JSONL, as data/dev/dev.jsonl).
  --selection  the sealed selection input, outside the repo. Per case: the worker whose
               output is used, the target level per sub-dimension (the stratum), and for a
               seeded record one rule of RULES with its exact edits; also the shuffle seed.
               This script reads it and never writes it. Shape: {"shuffle_seed": int,
               "picks": [{"sample_id", "worker" (the model id as the log names it),
               "grounding", "coverage", and optionally "epoch" (1), "rule", "basis",
               "edits" ([{"find", "replace"}]), "note_words"}]}.

Files written under --out (outside the repo):
  calibration-sheet.html            the only file the author opens: one record at a time, blind
  _sealed/calibration-plan.json     the plan: per record the code, case id, worker model,
                                    source, seed note, stratum, worker output; read by
                                    scripts/calibration_import.py

Sheet id: a SHA-256 over the records the sheet shows and the plan. It sits in the sheet
data, in the browser storage key and in a sheet_id column of every row of the saved
grades file, and the plan records it. A rebuilt sheet that shows other records gets
another id, so grades saved from an earlier sheet never attach to outputs the author
did not grade: the page will not load them and the import refuses them.

The sheet shows what the judge-development sheet shows: the narrative, the facts sheet,
the case file as the worker saw it, and the committed scope and level texts. It never
shows the worker model, the case id, the source, the seed note, the stratum, the log or
any score. Records appear under a random code in an order shuffled by the recorded seed.
Same inputs give the same bytes. Record codes follow from the shuffle seed alone, so with
--force a selection that changed must also change its shuffle seed.

The screen never shows target levels: stdout prints totals only, and a refusal for
targets that miss the margin says only that. The per-level detail goes to
calibration-build-shortfall.txt in the sealed folder that holds the selection input.

Exit codes:
  0  written; stdout prints totals only (records, unparsed outputs, sheet id, sheet hash)
  3  refused (nothing written): a missing or unreadable log; not exactly one log per
     worker; a log that is not a successful whole calibration-split run at epochs 1 under
     the locked prompt, or that fails the run-config check; a case file that is not the 40
     calibration cases; a missing or malformed selection input; a selection that is not
     40 cases with one output each, whose targets miss the margin of 6 per level per
     sub-dimension (the one file written on a refusal is the per-level detail beside the
     selection input), that seeds one way only, or whose target does not follow from its rule; a
     picked output that does not parse (D-R10 C5) or ended in an error; a seed whose rule
     does not apply to its base output or whose edit changes a protected token or a code
     score; an output folder or selection inside the repo; an existing output without
     --force; with --force, an existing sheet without its plan, an existing plan that
     cannot be read, or a selection that changed while its shuffle seed did not (the
     record codes would repeat); or a sheet that would show sealed text
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location, and dataclasses resolve string annotations
# through sys.modules.
import argparse
import hashlib
import html
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _p in (REPO, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import icfr_agent_evals  # noqa: E402
import judge_dev_build as jdb  # noqa: E402
from icfr_agent_evals import casefile, roster, rubric, scoring  # noqa: E402
from icfr_agent_evals import review_facts as rf  # noqa: E402
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
PLAN_VERSION = "1"
CAL_SPLIT = "calibration"
CASE_PREFIX = "calibration-"
SET_SIZE = 40              # section 4: 40 cases
MIN_PER_LEVEL = 4          # section 4: every level 0 to 3 at least 4 times per sub-dimension
TARGET_MIN_PER_LEVEL = 6   # margin on the targets: every level at least 6 times per sub-dimension
EPOCH = 1                  # each worker answers each case once
GROUNDING, COVERAGE = jdb.GROUNDING, jdb.COVERAGE
SUBDIMS = jdb.SUBDIMS
LEVELS = jdb.LEVELS
SOURCES = jdb.SOURCES
LOCKED_ANCHOR_SHA256 = jdb.LOCKED_ANCHOR_SHA256
# The three workers whose natural outputs fill the subset: every evaluated model except
# the one that also holds the calibrator role (Sonnet 5.5, Sonnet 5 and Haiku 5.5).
CAL_WORKERS: tuple[str, ...] = tuple(m for m in roster.EVALUATED_MODELS if m != roster.CALIBRATOR)

DEFAULT_CASES = REPO / "data" / "calibration" / "calibration.jsonl"
DEFAULT_OUT = REPO.parent / "calibration-subset"
SEALED_DIR = jdb.SEALED_DIR
SELECTION_NAME = "calibration-selection.json"
DEFAULT_SELECTION = DEFAULT_OUT / SEALED_DIR / SELECTION_NAME
PLAN_NAME = "calibration-plan.json"
SHEET_NAME = "calibration-sheet.html"
SHEET_TITLE = "Calibration subset grading"
GRADES_FILE = "calibration-grades.csv"
SITTINGS_FILE = "calibration-sittings.csv"
STORAGE_PREFIX = "calibration-sheet-"
SHORTFALL_NAME = "calibration-build-shortfall.txt"
SHEET_ID_RE = re.compile(r"^[0-9a-f]{64}$")

Refused = jdb.Refused
SeedError = jdb.SeedError


class ShortRefused(Refused):
    """A refusal for levels that fall short. The message never names a level or a count;
    the per-level detail travels in `detail` and is written only to a sealed folder."""

    def __init__(self, message: str, detail: Sequence[str]):
        super().__init__(message)
        self.detail = list(detail)


def write_detail(folder: Path, name: str, heading: str, detail: Sequence[str]) -> Path:
    """Write the per-level detail of a refusal into a sealed folder, never to the screen."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(("\n".join([heading, *detail]) + "\n").encode("utf-8"))
    return path


@dataclass(frozen=True)
class CalPick:
    """One case of the subset: the worker whose output is used, and the judge-development
    pick (case id, epoch 1, targets, optional rule with its exact edits)."""
    worker: str
    pick: jdb.Pick

    @property
    def sample_id(self) -> str:
        return self.pick.sample_id

    @property
    def source(self) -> str:
        """natural, seeded_omission or seeded_addition; "unknown" for a rule RULES does not hold."""
        return "natural" if self.pick.rule is None else jdb.RULE_SOURCE.get(self.pick.rule, "unknown")

    @property
    def stratum(self) -> dict[str, int]:
        return self.pick.stratum


@dataclass(frozen=True)
class CalSelection:
    """The sealed build input: the picks and the shuffle seed."""
    picks: tuple[CalPick, ...]
    shuffle_seed: int
    sha256: str


# --- Sealed selection input -------------------------------------------------------------

_PICK_KEYS = ("sample_id", "worker", "epoch", "grounding", "coverage", "rule", "basis", "edits", "note_words")
_REQUIRED_KEYS = {"sample_id", "worker", "grounding", "coverage"}


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_selection(path: Path) -> CalSelection:
    """Read the sealed selection input. Refuses a missing or malformed file."""
    if not path.is_file():
        raise Refused(f"selection input not found: {path.name}")
    try:
        data = path.read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the selection input: {type(e).__name__}") from e
    if not isinstance(doc, dict):
        raise Refused("the selection input is not a JSON object")
    seed, rows = doc.get("shuffle_seed"), doc.get("picks")
    if not _int(seed):
        raise Refused("the selection input has no integer shuffle_seed")
    if not isinstance(rows, list):
        raise Refused("the selection input has no list of picks")
    picks: list[CalPick] = []
    for n, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) - set(_PICK_KEYS) or not _REQUIRED_KEYS <= set(row):
            raise Refused(f"pick {n} of the selection input is malformed")
        epoch = row.get("epoch", EPOCH)
        if (not isinstance(row["sample_id"], str) or not isinstance(row["worker"], str)
                or not all(_int(v) for v in (epoch, row["grounding"], row["coverage"]))):
            raise Refused(f"pick {n} of the selection input has a field of the wrong type")
        if epoch != EPOCH:
            raise Refused(f"pick {n} of the selection input names epoch {epoch}; every worker answers at epoch {EPOCH}")
        rule = row.get("rule")
        if rule is not None and not isinstance(rule, str):
            raise Refused(f"pick {n} of the selection input has a rule that is not a string")
        pick = jdb.Pick(row["sample_id"], EPOCH, row["grounding"], row["coverage"], rule, row.get("basis") or "",
                        jdb._edits_of(row.get("edits"), n), row.get("note_words"))
        picks.append(CalPick(row["worker"], pick))
    return CalSelection(tuple(picks), seed, hashlib.sha256(data).hexdigest())


# --- Stratification and selection rules -------------------------------------------------


def level_counts(picks: Sequence[CalPick]) -> dict[str, dict[int, int]]:
    """Records per target level per sub-dimension; a target out of range counts nowhere
    (selection_problems names it)."""
    counts = {d: {lv: 0 for lv in LEVELS} for d in SUBDIMS}
    for p in picks:
        for d, lv in p.stratum.items():
            if lv in counts[d]:
                counts[d][lv] += 1
    return counts


def _rule_problems(p: CalPick) -> list[str]:
    """The judge-development rule checks for one pick: a natural record carries no edit,
    a seeded record names a known rule, its targets follow from the rule and its edits
    are as many as the rule takes."""
    pick, out = p.pick, []
    if pick.rule is None:
        if pick.edits or pick.note_words:
            out.append(f"{pick.sample_id}: a natural record carries edits or note words")
        return out
    if pick.rule not in jdb.RULES:
        return [f"{pick.sample_id}: unknown rule {pick.rule}"]
    for d, lv in jdb.RULE_TARGET[pick.rule].items():
        if pick.stratum[d] != lv:
            out.append(f"{pick.sample_id}: rule {pick.rule} makes {d} level {lv}, not {pick.stratum[d]}")
    if pick.rule in jdb.EDIT_COUNT:
        lo, hi = jdb.EDIT_COUNT[pick.rule]
        if not lo <= len(pick.edits) <= hi:
            out.append(f"{pick.sample_id}: rule {pick.rule} takes {lo} to {hi} edits, not {len(pick.edits)}")
    elif pick.edits:
        out.append(f"{pick.sample_id}: an omission rule takes no edits")
    if (pick.rule == "G-ADOPT") != bool(pick.note_words):
        out.append(f"{pick.sample_id}: note words go with G-ADOPT only, and G-ADOPT needs them")
    return out


def stratification_shortfalls(counts: Mapping[str, Mapping[int, int]], minimum: int = MIN_PER_LEVEL) -> list[str]:
    """Each sub-dimension level with fewer than `minimum` records, and by how much."""
    out = []
    for d in SUBDIMS:
        for lv in LEVELS:
            n = counts.get(d, {}).get(lv, 0)
            if n < minimum:
                out.append(f"{d} level {lv} has {n} records, {minimum - n} short of {minimum}")
    return out


def target_shortfalls(picks: Sequence[CalPick]) -> list[str]:
    """Each sub-dimension level whose targets fall below the margin of
    TARGET_MIN_PER_LEVEL. Sealed detail: never printed to the screen."""
    return stratification_shortfalls(level_counts(picks), TARGET_MIN_PER_LEVEL)


def seed_types(picks: Sequence[CalPick]) -> dict[str, int]:
    return {s: sum(1 for p in picks if p.source == s) for s in SOURCES}


def selection_problems(picks: Sequence[CalPick], case_ids: Sequence[str] | None = None) -> list[str]:
    """Every rule the selection must meet, as plain messages; empty when it meets them all.
    The target margin is checked apart (target_shortfalls), since its detail is sealed."""
    problems: list[str] = []
    if len(picks) != SET_SIZE:
        problems.append(f"the subset holds {len(picks)} records, not {SET_SIZE}")
    per_case = Counter(p.sample_id for p in picks)
    for case, n in sorted(per_case.items()):
        if n > 1:
            problems.append(f"{case} is picked {n} times; the subset takes one output per case")
    if case_ids is not None:
        missing = sorted(set(case_ids) - set(per_case))
        unknown = sorted(set(per_case) - set(case_ids))
        if missing:
            problems.append(f"case(s) of the case file with no pick: {', '.join(missing)}")
        if unknown:
            problems.append(f"picked case(s) not in the case file: {', '.join(unknown)}")
    for p in picks:
        if not p.sample_id.startswith(CASE_PREFIX):
            problems.append(f"{p.sample_id} is not a calibration-split case")
        if p.worker not in CAL_WORKERS:
            problems.append(f"{p.sample_id}: worker {p.worker!r} is not one of the three calibration workers")
        for d, lv in p.stratum.items():
            if lv not in LEVELS:
                problems.append(f"{p.sample_id}: target {lv} for {d} out of range")
        problems.extend(_rule_problems(p))
    kinds = seed_types(picks)
    seeded = kinds["seeded_omission"] + kinds["seeded_addition"]
    if seeded and not (kinds["seeded_omission"] and kinds["seeded_addition"]):
        problems.append("the seeds are made one way only; section 4 seeds two ways, by omission and by addition")
    return problems


# --- Case file and logs -------------------------------------------------------------------


def load_cases(path: Path) -> dict[str, dict]:
    """The 40 calibration cases, by id. Refuses any other file."""
    if not path.is_file():
        raise Refused(f"case file not found: {path.name}")
    cases: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in cases:
                        raise Refused("the case file has a row without an id, or an id twice")
                    cases[row["id"]] = row
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the case file: {type(e).__name__}") from e
    if len(cases) != SET_SIZE:
        raise Refused(f"the case file holds {len(cases)} cases, not {SET_SIZE}")
    for cid, row in cases.items():
        meta = row.get("metadata")
        if not cid.startswith(CASE_PREFIX) or not isinstance(meta, dict) or meta.get("split") != CAL_SPLIT:
            raise Refused(f"{cid} is not a calibration-split case")
        if not isinstance(row.get("input"), str):
            raise Refused(f"{cid} has no case file text")
    return cases


def read_log(path: Path) -> Any:
    """Read one full worker log (samples and events); refuses a missing or unreadable file."""
    return jdb._read_log(path)


def check_worker_log(log: Any, model: str, case_ids: Sequence[str], max_tokens_pin: int | None = None) -> dict:
    """Refuses a log that is not a successful whole calibration-split run of `model` at
    epochs 1 under the locked prompt. Returns the run-config summary."""
    if getattr(log, "status", None) != "success":
        raise Refused(f"{model} log status is {getattr(log, 'status', None)!r}, not 'success'")
    spec = log.eval
    args = spec.task_args or {}
    meta = spec.metadata or {}
    if args.get("split") != CAL_SPLIT or args.get("allow_evaluation_split") or meta.get("split", CAL_SPLIT) != CAL_SPLIT:
        raise Refused(f"{model} log is not a calibration-split run: split {args.get('split')!r}")
    if spec.model != model:
        raise Refused(f"the log for {model} ran {spec.model!r}")
    if meta.get("worker_prompt_sha256") != WORKER_PROMPT_SHA256:
        raise Refused(f"{model} log: worker_prompt_sha256 is not the locked prompt's")
    try:
        summary = roster.check_run_config(log, expected_samples=SET_SIZE, expected_epochs=EPOCH, allow_batch=True,
                                          max_tokens_pin=max_tokens_pin)
    except roster.RunConfigError as e:
        raise Refused(f"{model} log: {e}") from e
    keys = [(str(s.id), int(s.epoch)) for s in log.samples]
    if len(keys) != len(set(keys)) or set(keys) != {(c, EPOCH) for c in case_ids}:
        raise Refused(f"{model} log does not hold each of the {SET_SIZE} calibration cases once at epoch {EPOCH}")
    return summary


def index_logs(logs: Sequence[Any]) -> dict[str, Any]:
    """The logs by worker model: exactly one per calibration worker."""
    by_model: dict[str, Any] = {}
    for log in logs:
        model = getattr(getattr(log, "eval", None), "model", None)
        if model not in CAL_WORKERS:
            raise Refused(f"a log ran {model!r}, which is not one of the three calibration workers")
        if model in by_model:
            raise Refused(f"two logs ran {model}; give one log per worker")
        by_model[model] = log
    missing = [m for m in CAL_WORKERS if m not in by_model]
    if missing:
        raise Refused(f"no log for {', '.join(missing)}")
    return by_model


def unparsed_outputs(logs_by_model: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every output in the three logs that did not parse (D-R10 C5: never graded)."""
    out = []
    for model in CAL_WORKERS:
        for s in sorted(logs_by_model[model].samples, key=lambda s: str(s.id)):
            if s.error is not None or not scoring.parse_worker_output(jdb._completion(s)).ok:
                out.append({"sample_id": str(s.id), "worker_model": model})
    return out


# --- Records ------------------------------------------------------------------------------


def build_records(logs_by_model: Mapping[str, Any], cases: Mapping[str, dict], selection: CalSelection) -> list[dict]:
    problems = selection_problems(selection.picks, list(cases))
    if problems:
        raise Refused("selection: " + "; ".join(problems))
    short = target_shortfalls(selection.picks)
    if short:
        raise ShortRefused(f"selection: the targets miss the margin of {TARGET_MIN_PER_LEVEL} records per level per "
                           "sub-dimension", short)
    records: list[dict] = []
    for cp in selection.picks:
        pick, model = cp.pick, cp.worker
        log = logs_by_model[model]
        by_id = {str(s.id): s for s in log.samples if int(s.epoch) == EPOCH}
        sample = by_id.get(pick.sample_id)
        if sample is None:
            raise Refused(f"{pick.sample_id} is not in the {model} log")
        if getattr(sample, "error", None):
            raise Refused(f"{pick.sample_id} from {model} ended in an error")
        raw = jdb._completion(sample)
        parse = scoring.parse_worker_output(raw)
        if not parse.ok:
            raise Refused(f"{pick.sample_id} from {model} did not parse (D-R10 C5: not graded)")
        case = cases[pick.sample_id]
        case_input = jdb._input_text(sample.input)
        if case["input"] != case_input:
            raise Refused(f"{pick.sample_id}: the {model} log's case file differs from the case file")
        sample_truth = dict(getattr(sample, "metadata", None) or {}).get("truth")
        if sample_truth != case["metadata"].get("truth"):
            raise Refused(f"{pick.sample_id}: the {model} log's answer key differs from the case file")
        metadata = dict(case["metadata"])
        base_narr = str(parse.parsed["narrative"])
        seeded = None
        if pick.rule is None:
            worker_output, narrative, note = raw, base_narr, None
        else:
            try:
                seeded = jdb.apply_rule(pick, base_narr, metadata, case_input)
                worker_output = jdb.replace_narrative(raw, base_narr, seeded.narrative)
            except SeedError as e:
                raise Refused(f"{pick.sample_id} from {model}, rule {pick.rule}: {e}") from e
            narrative, note = seeded.narrative, seeded.note
            if pick.rule in jdb.EDIT_COUNT:
                truth = casefile.truth_of(metadata)
                before = jdb.code_scores(parse.parsed, truth)
                after = jdb.code_scores(scoring.parse_worker_output(worker_output).parsed, truth)
                if before != after:
                    changed = sorted(k for k in before if before[k] != after[k])
                    raise Refused(f"{pick.sample_id} from {model}, rule {pick.rule}: the edit changes a code score "
                                  f"({', '.join(changed)})")
        parsed = scoring.parse_worker_output(worker_output).parsed
        records.append({
            "sample_id": pick.sample_id,
            "epoch": EPOCH,
            "worker_model": model,
            "log_eval_id": str(log.eval.eval_id),
            "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "worker_output": worker_output,
            "source": pick.source,
            "seed_note": note,
            "stratum": pick.stratum,
            "seed_rule": pick.rule,
            "stratum_basis": pick.basis,
            "seed_edits": list(seeded.edits) if seeded else [],
            "seed_deleted": list(seeded.deleted) if seeded else [],
            "base_worker_output": raw,
            "file_name": f"{pick.sample_id}.json",
            # sheet-only fields, left out of the plan
            "_narrative": narrative,
            "_case_input": case_input,
            "_case_facts": rf.facts_payload(rf.case_facts(metadata)),
            "_output_facts": rf.facts_payload(rf.output_facts(metadata, parsed)),
        })
    rng = random.Random(f"{selection.shuffle_seed}|codes")
    taken: set[str] = set()
    for r in records:
        r["code"] = jdb.record_code(rng, taken)
    order = list(range(len(records)))
    random.Random(f"{selection.shuffle_seed}|order").shuffle(order)
    for pos, i in enumerate(order, start=1):
        records[i]["position"] = pos
    return records


# --- Sheet ----------------------------------------------------------------------------------


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_sheet_id(records: Sequence[Mapping[str, Any]], plan_core: Mapping[str, Any]) -> str:
    """SHA-256 over the records the sheet shows (the judge-development sheet data: codes,
    narratives, facts, case files and rubric texts) and the plan without its sheet fields."""
    return hashlib.sha256(_canonical({"plan": dict(plan_core), "sheet": jdb.sheet_data(records)})).hexdigest()


def storage_key(sheet_id: str) -> str:
    """The browser storage key: one per sheet id, so a rebuilt sheet never loads the
    answers kept for an earlier one."""
    return STORAGE_PREFIX + sheet_id[:32]


def sheet_data(records: Sequence[Mapping[str, Any]], sheet_id: str) -> dict:
    """The judge-development sheet's data, with this subset's file names, sheet id and browser key."""
    if not SHEET_ID_RE.match(sheet_id or ""):
        raise Refused("the sheet id is not a SHA-256 hex digest")
    data = jdb.sheet_data(records)
    data.update(storage_key=storage_key(sheet_id), grades_file=GRADES_FILE, sittings_file=SITTINGS_FILE,
                sheet_id=sheet_id)
    return data


# The judge-development page script, with a sheet_id column in the saved grades file and
# a page that will not load a grades file saved from another sheet.
_JS_EDITS = (
    ('var rows = [["record_code"].concat(DIMS).concat(["note", "graded_at"])];',
     'var rows = [["record_code"].concat(DIMS).concat(["note", "graded_at", "sheet_id"])];'),
    ('.concat([x.note || "", x.at || ""])); });',
     '.concat([x.note || "", x.at || "", D.sheet_id])); });'),
    ('if (ci < 0 || di.indexOf(-1) >= 0) { status("That file is not a grades file from this page."); return; }',
     'var si = head.indexOf("sheet_id");\n'
     '  if (ci < 0 || si < 0 || di.indexOf(-1) >= 0) { status("That file is not a grades file from this page."); '
     'return; }\n'
     '  if (rows.some(function(row){ return row.length > 1 && row[si] !== D.sheet_id; })) '
     '{ status("That grades file was saved from another sheet. It is not loaded."); return; }'),
)


def sheet_script() -> str:
    """The judge-development script with the sheet id edits; refuses when an edit no
    longer finds its text exactly once."""
    js = jdb._JS.strip()
    for old, new in _JS_EDITS:
        if js.count(old) != 1:
            raise Refused("the judge-development sheet script changed; the sheet id edits no longer apply")
        js = js.replace(old, new)
    return js


def render_sheet(records: Sequence[Mapping[str, Any]], sheet_id: str) -> str:
    """The judge-development sheet page and script, under this subset's title, carrying the sheet id."""
    page = jdb._PAGE.replace("__TITLE__", html.escape(SHEET_TITLE)).replace("__CSS__", jdb._CSS.strip())
    page = page.replace("__JS__", sheet_script())
    return page.replace("__DATA__", jdb._embed_json(sheet_data(records, sheet_id)))


def worker_words() -> list[str]:
    """Each calibration worker's model id, its name without the provider, and each word of
    four letters or more in it (provider, family and tier words)."""
    out: list[str] = []
    for m in CAL_WORKERS:
        out += [m, m.split("/", 1)[-1]]
        out += [w for w in re.split(r"[/\-.]", m) if len(w) >= 4 and w.isalpha()]
    return list(dict.fromkeys(out))


def blindness_hits(sheet: str, records: Sequence[Mapping[str, Any]], log_eval_ids: Sequence[str] = ()) -> list[str]:
    """Everything sealed that the sheet must not hold. Empty when the sheet is blind."""
    hits = jdb.blindness_hits(sheet, records, "")
    low = sheet.lower()
    if re.search(r"calibration-(?:hc|std|ac|edge|adv|hw)-", low):
        hits.append("a calibration case id prefix")
    for eid in log_eval_ids:
        if eid and eid in sheet:
            hits.append("a log eval id")
    for w in worker_words():
        if re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", low):
            hits.append(f"worker model text {w!r}")
    if "worker_model" in low:
        hits.append("the key worker_model")
    return hits


# --- Plan ---------------------------------------------------------------------------------

PLAN_FIELDS = (
    "code", "position", "file_name", "sample_id", "epoch", "worker_model", "log_eval_id", "worker_prompt_sha256",
    "source", "seed_rule", "seed_note", "seed_edits", "seed_deleted",
    "stratum", "stratum_basis", "worker_output", "base_worker_output",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SHEET_FIELDS = ("sheet_file", "sheet_id", "sheet_sha256")


def plan_payload(records: Sequence[Mapping[str, Any]], *, logs: Sequence[Mapping[str, str]], cases_path: Path,
                 cases_sha: str, selection: CalSelection, selection_path: Path, unparsed: Sequence[Mapping[str, str]],
                 sheet_id: str, sheet_sha: str) -> dict:
    return {
        "plan_version": PLAN_VERSION,
        "what": "Calibration subset plan (sealed). Do not open before grading is finished.",
        "logs": list(logs),
        "cases_file": cases_path.name,
        "cases_sha256": cases_sha,
        "worker_prompt_sha256": WORKER_PROMPT_SHA256,
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "selection_file": selection_path.name,
        "selection_sha256": selection.sha256,
        "shuffle_seed": selection.shuffle_seed,
        "min_per_level": MIN_PER_LEVEL,
        "target_min_per_level": TARGET_MIN_PER_LEVEL,
        "unparsed": list(unparsed),
        "rules": dict(jdb.RULES),
        "level_counts": {d: {str(lv): n for lv, n in c.items()} for d, c in level_counts(selection.picks).items()},
        "source_counts": seed_types(selection.picks),
        "worker_counts": {m: sum(1 for p in selection.picks if p.worker == m) for m in CAL_WORKERS},
        "sheet_file": SHEET_NAME,
        "sheet_id": sheet_id,
        "sheet_sha256": sheet_sha,
        "records": [{k: r[k] for k in PLAN_FIELDS} for r in sorted(records, key=lambda r: r["position"])],
    }


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def check_rebuild(sheet_path: Path, plan_path: Path, selection: CalSelection) -> None:
    """With --force: refuses a rebuild whose record codes would repeat for other outputs.
    Codes follow from the shuffle seed alone, so a selection that changed must change its
    shuffle seed too. An existing sheet without its plan, or a plan that cannot be read,
    cannot be compared and is refused."""
    if not plan_path.exists():
        if sheet_path.exists():
            raise Refused("a sheet exists without its plan, so the rebuild cannot be checked; move the old sheet away")
        return
    try:
        old = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the existing plan to check the rebuild: {type(e).__name__}") from e
    if not isinstance(old, dict):
        raise Refused("the existing plan is not a JSON object, so the rebuild cannot be checked")
    if old.get("selection_sha256") != selection.sha256 and old.get("shuffle_seed") == selection.shuffle_seed:
        raise Refused("the selection changed but its shuffle seed did not, so the rebuilt sheet would reuse record "
                      "codes for other outputs; give the changed selection a new shuffle_seed")


def build(log_paths: Sequence[Path], out_dir: Path, cases_path: Path, selection_path: Path, force: bool,
          max_tokens_pin: int | None = None) -> dict:
    if rubric.llm_anchor_sha256() != LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    if _inside_repo(out_dir) or _inside_repo(selection_path):
        raise Refused("the output folder and the selection input stay outside the repo; they hold sealed text")
    if len(log_paths) != len(CAL_WORKERS):
        raise Refused(f"give {len(CAL_WORKERS)} logs, one per worker, not {len(log_paths)}")
    sheet_path = out_dir / SHEET_NAME
    plan_path = out_dir / SEALED_DIR / PLAN_NAME
    if not force and (sheet_path.exists() or plan_path.exists()):
        raise Refused("the sheet or the plan already exists; pass --force to rebuild")
    selection = load_selection(selection_path)
    if force:
        check_rebuild(sheet_path, plan_path, selection)
    cases = load_cases(cases_path)
    paths_of: dict[int, Path] = {}
    logs = []
    for p in log_paths:
        log = read_log(Path(p))
        paths_of[id(log)] = Path(p)
        logs.append(log)
    by_model = index_logs(logs)
    for model, log in by_model.items():
        check_worker_log(log, model, list(cases), max_tokens_pin)
    try:
        records = build_records(by_model, cases, selection)
    except ShortRefused as e:
        where = write_detail(selection_path.parent, SHORTFALL_NAME,
                             f"Targets below the margin of {TARGET_MIN_PER_LEVEL} per level per sub-dimension:",
                             e.detail)
        raise ShortRefused(f"{e}; the per-level detail is in {where.name} beside the selection input", e.detail) from e
    log_rows = [{"model": m, "file": paths_of[id(by_model[m])].name, "sha256": _sha256_file(paths_of[id(by_model[m])]),
                 "eval_id": str(by_model[m].eval.eval_id)} for m in CAL_WORKERS]
    plan_args = dict(logs=log_rows, cases_path=cases_path, cases_sha=_sha256_file(cases_path), selection=selection,
                     selection_path=selection_path, unparsed=unparsed_outputs(by_model))
    core = {k: v for k, v in plan_payload(records, **plan_args, sheet_id="", sheet_sha="").items()
            if k not in SHEET_FIELDS}
    sheet_id = compute_sheet_id(records, core)
    sheet = render_sheet(records, sheet_id)
    eval_ids = [str(by_model[m].eval.eval_id) for m in CAL_WORKERS]
    hits = blindness_hits(sheet, records, eval_ids)
    if hits:
        raise Refused("the sheet would show sealed text: " + "; ".join(hits))
    sheet_bytes = sheet.encode("utf-8")
    plan = plan_payload(records, **plan_args, sheet_id=sheet_id, sheet_sha=hashlib.sha256(sheet_bytes).hexdigest())
    (out_dir / SEALED_DIR).mkdir(parents=True, exist_ok=True)
    sheet_path.write_bytes(sheet_bytes)
    plan_path.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the calibration subset's sealed plan and blind grading sheet.")
    ap.add_argument("--log", type=Path, action="append", default=[], required=True,
                    help="a worker log on the calibration split; give one per worker")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-tokens-pin", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return EXIT_REFUSED
    print(f"module: {icfr_agent_evals.__file__}")
    try:
        plan = build(args.log, args.out, args.cases, args.selection, args.force, args.max_tokens_pin)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    # Totals only: never a level, source or worker count, since the author grades this sheet.
    print(f"records: {len(plan['records'])}")
    print(f"unparsed outputs (never graded): {len(plan['unparsed'])}")
    print(f"sheet id: {plan['sheet_id']}")
    print(f"sheet sha256: {plan['sheet_sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
