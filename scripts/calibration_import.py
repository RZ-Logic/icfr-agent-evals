"""Write the author's calibration subset grades into data/calibration/records/. NO API key, no model call.

Usage:
  python scripts/calibration_import.py --grades CSV [--plan PATH] [--out DIR]

Reads the grades file saved from calibration-sheet.html (calibration-grades.csv) and the
sealed plan written by scripts/calibration_build.py, and writes one JSON record per
case in the format of data/calibration/README.md: the 12 fields of the judge-development
record format in their order, then worker_model, UTF-8, LF line endings, named
<case id>.json. anchors_sha256 is the committed llm_anchor_sha256; graded_by is the
author's initials, as for the judge-development set; graded_on is the date of the
grade's graded_at stamp.

Grades are the author's. This script never sets, fills or changes one: it copies the
two 0 to 3 grades per record exactly as the file gives them, or refuses.

Pre-registration section 4: every score level 0 to 3 appears at least 4 times per
LLM-graded sub-dimension. The build checks the targets with a margin (at least 6 per
level per sub-dimension) and this import checks the plan's targets the same way. On the
author's grades the floor of 4 is the hard rule: the import refuses grades that miss it.
The subset then needs more seeds before the grader's acceptance look.

The screen never shows a level count or a shortfall. A refusal for levels that fall
short says only that the stratification is not met; the per-level detail is written to
calibration-import-shortfall.txt in the sealed folder beside the plan.

Sheet id: the grades file must carry a sheet_id column whose value, on every row, is
the plan's sheet_id. A grades file saved from another build of the sheet is refused,
since its record codes can name outputs the author never graded.

Exit codes:
  0  all 40 records written; stdout prints the file names only
  3  refused (nothing written): a missing or unreadable file; a plan that is not the
     40-record plan, whose anchors hash or prompt hash is not the committed one, whose
     worker outputs do not parse, that holds an output listed as unparsed, a case twice,
     a case outside the calibration split, an epoch other than 1, a worker outside the
     three calibration workers or a log eval id that is not its worker's, in which a
     worker output differs from its base output outside the narrative, whose targets
     miss the margin of 6, that has no sheet id, or whose sheet_sha256 is not the hash of
     the sheet beside it; a grades file without the required columns, without a sheet_id
     column, with a sheet id that is not the plan's, or with a column name given twice; a
     record code that is missing, repeated or not in the plan; a grade that is missing or
     not one of 0, 1, 2, 3; a missing or malformed graded_at; author grades that miss the
     floor of 4; an output folder whose parent is missing; or an output file that
     already exists. A refusal for levels that fall short writes one file, the per-level
     detail beside the plan, and nothing in the output folder
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location.
import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _p in (REPO, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import calibration_build as cb  # noqa: E402
import judge_dev_import as jdi  # noqa: E402
from icfr_agent_evals import rubric, scoring  # noqa: E402
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
GROUNDING, COVERAGE = jdi.GROUNDING, jdi.COVERAGE
SUBDIMS = jdi.SUBDIMS
LEVELS = (0, 1, 2, 3)
SET_SIZE = cb.SET_SIZE
MIN_PER_LEVEL = cb.MIN_PER_LEVEL
TARGET_MIN_PER_LEVEL = cb.TARGET_MIN_PER_LEVEL
SHORTFALL_NAME = "calibration-import-shortfall.txt"
LOCKED_ANCHOR_SHA256 = jdi.LOCKED_ANCHOR_SHA256
GRADED_BY = jdi.GRADED_BY
SOURCES = jdi.SOURCES
DEFAULT_PLAN = cb.DEFAULT_OUT / cb.SEALED_DIR / cb.PLAN_NAME
DEFAULT_OUT = REPO / "data" / "calibration" / "records"
# The judge-development record format, then the worker model of the output.
RECORD_FIELDS = (*jdi.RECORD_FIELDS, "worker_model")
FILE_NAME_RE = re.compile(r"^calibration-[A-Za-z0-9-]+\.json$")

Refused = jdi.Refused


class ShortRefused(Refused):
    """A refusal for levels that fall short. The message never names a level or a count;
    the per-level detail travels in `detail` and is written only beside the plan."""

    def __init__(self, message: str, detail: Sequence[str]):
        super().__init__(message)
        self.detail = list(detail)


def _counts(levels_by_record: Sequence[Mapping[str, int]]) -> dict[str, dict[int, int]]:
    out = {d: {lv: 0 for lv in LEVELS} for d in SUBDIMS}
    for g in levels_by_record:
        for d in SUBDIMS:
            out[d][g[d]] += 1
    return out


def grade_shortfalls(grades: Sequence[Mapping[str, int]]) -> list[str]:
    """Each sub-dimension level the author's grades give fewer than MIN_PER_LEVEL times,
    and by how much."""
    return cb.stratification_shortfalls(_counts(grades), MIN_PER_LEVEL)


def _logs_by_model(plan: Mapping[str, Any]) -> dict[str, str]:
    logs = plan.get("logs")
    if not isinstance(logs, list) or not all(
            isinstance(x, dict) and isinstance(x.get("model"), str) and isinstance(x.get("eval_id"), str) for x in logs):
        raise Refused("the plan's log list is malformed")
    by_model = {x["model"]: x["eval_id"] for x in logs}
    if len(by_model) != len(logs) or set(by_model) != set(cb.CAL_WORKERS):
        raise Refused("the plan does not list one log per calibration worker")
    return by_model


def _unparsed_keys(plan: Mapping[str, Any]) -> set[tuple[str, str]]:
    listed = plan.get("unparsed") or []
    if not isinstance(listed, list) or not all(
            isinstance(e, dict) and isinstance(e.get("sample_id"), str) and isinstance(e.get("worker_model"), str)
            for e in listed):
        raise Refused("the plan's unparsed list is malformed")
    return {(e["sample_id"], e["worker_model"]) for e in listed}


def load_plan(path: Path) -> dict:
    if not path.is_file():
        raise Refused(f"plan not found: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the plan: {type(e).__name__}") from e
    if not isinstance(plan, dict):
        raise Refused("the plan is not a JSON object")
    if rubric.llm_anchor_sha256() != LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    if plan.get("anchors_sha256") != LOCKED_ANCHOR_SHA256:
        raise Refused("the plan was built against other anchors")
    if plan.get("worker_prompt_sha256") != WORKER_PROMPT_SHA256:
        raise Refused("the plan's worker_prompt_sha256 is not the locked prompt's")
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != SET_SIZE:
        raise Refused(f"the plan does not hold {SET_SIZE} records")
    eval_ids = _logs_by_model(plan)
    unparsed = _unparsed_keys(plan)
    codes, cases = set(), set()
    for r in records:
        if not isinstance(r, dict):
            raise Refused("the plan has a record that is not a JSON object")
        code, name, case = r.get("code"), r.get("file_name"), r.get("sample_id")
        if not isinstance(code, str) or code in codes:
            raise Refused(f"the plan has a missing or repeated record code: {code!r}")
        codes.add(code)
        if not isinstance(case, str) or not case.startswith(cb.CASE_PREFIX):
            raise Refused(f"record {code}: {case!r} is not a calibration-split case")
        if case in cases:
            raise Refused(f"record {code}: case {case} appears twice; the subset takes one output per case")
        cases.add(case)
        if not isinstance(name, str) or not FILE_NAME_RE.match(name) or name != f"{case}.json":
            raise Refused(f"record {code}: file name {name!r} is not <case id>.json")
        if r.get("epoch") != cb.EPOCH or isinstance(r.get("epoch"), bool):
            raise Refused(f"record {code}: epoch {r.get('epoch')!r}, not {cb.EPOCH}")
        model = r.get("worker_model")
        if model not in cb.CAL_WORKERS:
            raise Refused(f"record {code}: worker {model!r} is not one of the three calibration workers")
        if r.get("log_eval_id") != eval_ids[model]:
            raise Refused(f"record {code}: its log eval id is not the {model} log's")
        if (case, model) in unparsed:
            raise Refused(f"record {code}: an output that did not parse is in the plan (D-R10 C5)")
        if r.get("source") not in SOURCES:
            raise Refused(f"record {code}: unknown source {r.get('source')!r}")
        if (r.get("source") == "natural") != (r.get("seed_note") is None):
            raise Refused(f"record {code}: a seed note goes with a seeded record only")
        stratum = r.get("stratum")
        if not isinstance(stratum, dict) or set(stratum) != set(SUBDIMS) or not all(
                stratum[d] in LEVELS and not isinstance(stratum[d], bool) for d in SUBDIMS):
            raise Refused(f"record {code}: malformed stratum")
        if r.get("worker_prompt_sha256") != WORKER_PROMPT_SHA256:
            raise Refused(f"record {code}: prompt hash differs")
        parsed = scoring.parse_worker_output(r.get("worker_output"))
        if not parsed.ok:
            raise Refused(f"record {code}: the worker output does not parse")
        base = scoring.parse_worker_output(r.get("base_worker_output"))
        if not base.ok:
            raise Refused(f"record {code}: the base worker output does not parse")
        if not jdi._same_but_narrative(dict(parsed.parsed), dict(base.parsed)):
            raise Refused(f"record {code}: the worker output differs from its base output outside the narrative")
    sheet_id = plan.get("sheet_id")
    if not isinstance(sheet_id, str) or not cb.SHEET_ID_RE.match(sheet_id):
        raise Refused("the plan has no sheet id; rebuild the sheet and the plan")
    short = cb.stratification_shortfalls(_counts([r["stratum"] for r in records]), TARGET_MIN_PER_LEVEL)
    if short:
        raise ShortRefused("the plan's targets miss the stratification margin", short)
    jdi._check_sheet(plan, path)
    return plan


def load_grades(path: Path) -> dict[str, dict[str, Any]]:
    """The grades file, read exactly as the judge-development import reads it."""
    return jdi.load_grades(path)


def grades_sheet_id(path: Path) -> str:
    """The sheet id the grades file was saved from: its sheet_id column, one value on
    every row. Refuses a file without the column, with a blank value or with two ids."""
    if not path.is_file():
        raise Refused(f"grades file not found: {path}")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read the grades file: {type(e).__name__}") from e
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames or []
    if header.count("sheet_id") != 1:
        raise Refused("the grades file has no single sheet_id column; save it again from the current sheet")
    ids = {(row.get("sheet_id") or "").strip() for row in reader}
    if not ids or "" in ids:
        raise Refused("the grades file has a row without a sheet id")
    if len(ids) != 1:
        raise Refused("the grades file mixes rows from more than one sheet")
    return ids.pop()


def check_sheet_id(plan: Mapping[str, Any], grades_path: Path) -> None:
    """Refuses a grades file saved from a sheet other than the plan's: its record codes
    can name outputs the author never graded."""
    got = grades_sheet_id(grades_path)
    if got != plan["sheet_id"]:
        raise Refused(f"the grades file comes from sheet {got[:12]}, not from the plan's sheet "
                      f"{plan['sheet_id'][:12]}; grades from another sheet are never imported")


def build_records(plan: Mapping[str, Any], grades: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, dict]]:
    """The judge-development records plus the worker model; refuses author grades that
    miss the stratification."""
    base = jdi.build_records(plan, grades)
    short = grade_shortfalls([g["grades"] for g in grades.values()])
    if short:
        raise ShortRefused("the author's grades do not meet the section 4 stratification (every level 0 to 3 at "
                           f"least {MIN_PER_LEVEL} times per sub-dimension). The subset needs more seeds before "
                           "the acceptance look", short)
    model_of = {r["file_name"]: r["worker_model"] for r in plan["records"]}
    out = []
    for name, record in base:
        record = {**record, "worker_model": model_of[name]}
        assert tuple(record) == RECORD_FIELDS
        out.append((name, record))
    return out


def run(grades_path: Path, plan_path: Path, out_dir: Path) -> list[str]:
    try:
        plan = load_plan(plan_path)
        grades = load_grades(grades_path)
        check_sheet_id(plan, grades_path)
        records = build_records(plan, grades)
    except ShortRefused as e:
        where = cb.write_detail(plan_path.parent, SHORTFALL_NAME, f"{e}.", e.detail)
        raise ShortRefused(f"{e}; the per-level detail is sealed in {where.name} beside the plan", e.detail) from e
    if not out_dir.is_dir() and not out_dir.parent.is_dir():
        raise Refused(f"output folder not found, and neither is its parent: {out_dir}")
    clash = [name for name, _ in records if (out_dir / name).exists()]
    if clash:
        raise Refused(f"output file(s) already exist: {', '.join(clash)}")
    out_dir.mkdir(exist_ok=True)
    written: list[str] = []
    for name, record in records:
        (out_dir / name).write_bytes(jdi.record_bytes(record))
        written.append(name)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the author's calibration subset grades as records.")
    ap.add_argument("--grades", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return EXIT_REFUSED
    try:
        written = run(args.grades, args.plan, args.out)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    for name in written:
        print(name)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
