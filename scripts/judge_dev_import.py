"""Write the owner's judge-development grades into data/dev/judge-dev/. NO API key, no model call.

Usage:
  python scripts/judge_dev_import.py --grades CSV [--plan PATH] [--out DIR]

Reads the grades file saved from judge-dev-sheet.html (judge-dev-grades.csv) and the
sealed plan written by scripts/judge_dev_build.py, and writes one JSON record per
graded output in the format of data/dev/judge-dev/README.md: the 12 fields in the
README's order, UTF-8, LF line endings, named <sample_id>-e<epoch>.json (a seeded
record adds -seed-<n>). anchors_sha256 is the committed llm_anchor_sha256; graded_by
is RA; graded_on is the date of the grade's graded_at stamp.

Grades are the owner's. This script never sets, fills or changes one: it copies the
two 0 to 3 grades per record exactly as the file gives them, or refuses.

Exit codes:
  0  all 20 records written; stdout prints the file names only
  3  refused (nothing written): a missing or unreadable file; a plan that is not the
     20-record plan, whose anchors hash or prompt hash is not the committed one, whose
     worker outputs do not parse, that holds an excluded output (the FM-000 epoch), in
     which a worker output differs from its base output in a field other than the
     narrative, or whose sheet_sha256 is not the hash of the sheet file beside it; a
     grades file without the required columns or with a column name given twice; a
     record code that is missing, repeated or not in the plan; a grade that is missing
     or not one of 0, 1, 2, 3; a missing or malformed graded_at; or an output file that
     already exists
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location.
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import rubric, scoring  # noqa: E402
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
GROUNDING = "faithfulness.narrative_grounding"
COVERAGE = "completeness.narrative_coverage"
SUBDIMS = (GROUNDING, COVERAGE)
LOCKED_ANCHOR_SHA256 = "405813382714b30acfd0a5c51123b02bad382695222e98c580245522957565e5"
GRADED_BY = "RA"
SET_SIZE = 20
SOURCES = ("natural", "seeded_omission", "seeded_addition")
DEFAULT_PLAN = REPO.parent / "pilot-logs-2026-11-11" / "judge-dev" / "_sealed" / "judge-dev-plan.json"
SEALED_DIR = "_sealed"
# D-R10 C5: unparsed outputs are not graded and stay out of the set (same list as the build).
EXCLUDED: dict[tuple[str, int], str] = {
    ("dev-std-007-ETH-M3", 2): "FM-000: the output did not parse (D-R10 C5)",
}
DEFAULT_OUT = REPO / "data" / "dev" / "judge-dev"
REQUIRED_COLUMNS = ("record_code", GROUNDING, COVERAGE, "note", "graded_at")
# The README's record format, in its table order.
RECORD_FIELDS = (
    "sample_id", "epoch", "log_eval_id", "worker_prompt_sha256", "worker_output", "source", "seed_note",
    "stratum", "grades", "anchors_sha256", "graded_by", "graded_on",
)
FILE_NAME_RE = re.compile(r"^dev-[A-Za-z0-9-]+-e[1-4](?:-seed-[1-9][0-9]*)?\.json$")
GRADED_AT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?$")


class Refused(Exception):
    """An input the import will not use; exit 3, nothing written."""


def _excluded_keys(plan: Mapping[str, Any]) -> set[tuple[str, int]]:
    """The excluded outputs: this script's list plus any the plan itself lists."""
    keys = set(EXCLUDED)
    listed = plan.get("excluded") or []
    if not isinstance(listed, list):
        raise Refused("the plan's excluded list is malformed")
    for e in listed:
        if not isinstance(e, dict) or not isinstance(e.get("sample_id"), str) or not isinstance(e.get("epoch"), int):
            raise Refused("the plan's excluded list is malformed")
        keys.add((e["sample_id"], e["epoch"]))
    return keys


def _check_sheet(plan: Mapping[str, Any], path: Path) -> None:
    """When the sheet sits beside the plan (in the plan's folder, or in the folder above
    a _sealed folder), its SHA-256 must equal the plan's sheet_sha256."""
    name = plan.get("sheet_file")
    if name is None:
        return
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise Refused(f"the plan's sheet_file is not a bare file name: {name!r}")
    places = [path.parent / name]
    if path.parent.name == SEALED_DIR:
        places.append(path.parent.parent / name)
    for sheet in places:
        if sheet.is_file():
            got = hashlib.sha256(sheet.read_bytes()).hexdigest()
            if plan.get("sheet_sha256") != got:
                raise Refused(f"the sheet beside the plan ({name}) does not match the plan's sheet_sha256")
            return


def _same_but_narrative(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return {k: v for k, v in a.items() if k != "narrative"} == {k: v for k, v in b.items() if k != "narrative"}


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
    excluded = _excluded_keys(plan)
    codes, names = set(), set()
    for r in records:
        code, name = r.get("code"), r.get("file_name")
        if not isinstance(code, str) or code in codes:
            raise Refused(f"the plan has a missing or repeated record code: {code!r}")
        if not isinstance(name, str) or not FILE_NAME_RE.match(name) or name in names:
            raise Refused(f"the plan has a malformed or repeated file name: {name!r}")
        codes.add(code)
        names.add(name)
        if r.get("source") not in SOURCES:
            raise Refused(f"record {code}: unknown source {r.get('source')!r}")
        if (r.get("source") == "natural") != (r.get("seed_note") is None):
            raise Refused(f"record {code}: a seed note goes with a seeded record only")
        stratum = r.get("stratum")
        if not isinstance(stratum, dict) or set(stratum) != set(SUBDIMS) or not all(
                stratum[d] in (0, 1, 2, 3) for d in SUBDIMS):
            raise Refused(f"record {code}: malformed stratum")
        if r.get("worker_prompt_sha256") != WORKER_PROMPT_SHA256:
            raise Refused(f"record {code}: prompt hash differs")
        if not isinstance(r.get("epoch"), int) or not 1 <= r["epoch"] <= 4:
            raise Refused(f"record {code}: epoch out of range")
        if (r.get("sample_id"), r["epoch"]) in excluded:
            raise Refused(f"record {code}: an excluded output (FM-000, D-R10 C5) is in the plan")
        parsed = scoring.parse_worker_output(r.get("worker_output"))
        if not parsed.ok:
            raise Refused(f"record {code}: the worker output does not parse")
        base = scoring.parse_worker_output(r.get("base_worker_output"))
        if not base.ok:
            raise Refused(f"record {code}: the base worker output does not parse")
        if not _same_but_narrative(dict(parsed.parsed), dict(base.parsed)):
            raise Refused(f"record {code}: the worker output differs from its base output outside the narrative")
        expected = f"{r.get('sample_id')}-e{r.get('epoch')}"
        if not name.startswith(expected):
            raise Refused(f"record {code}: file name does not match its sample id and epoch")
    _check_sheet(plan, path)
    return plan


def load_grades(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise Refused(f"grades file not found: {path}")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read the grades file: {type(e).__name__}") from e
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames or []
    twice = sorted({c for c in header if header.count(c) > 1})
    if twice:
        raise Refused(f"the grades file gives the column name(s) more than once: {', '.join(twice)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise Refused(f"the grades file lacks the column(s): {', '.join(missing)}")
    out: dict[str, dict[str, Any]] = {}
    for n, row in enumerate(reader, start=2):
        code = (row.get("record_code") or "").strip()
        if not code:
            raise Refused(f"line {n}: no record code")
        if code in out:
            raise Refused(f"record {code} appears twice")
        grades: dict[str, int] = {}
        for d in SUBDIMS:
            raw = (row.get(d) or "").strip()
            if raw == "":
                raise Refused(f"record {code}: no grade for {d}")
            if raw not in ("0", "1", "2", "3"):
                raise Refused(f"record {code}: grade {raw!r} for {d} is not 0, 1, 2 or 3")
            grades[d] = int(raw)
        stamp = (row.get("graded_at") or "").strip()
        m = GRADED_AT_RE.match(stamp)
        if m is None:
            raise Refused(f"record {code}: graded_at {stamp!r} is missing or malformed")
        try:
            dt.date.fromisoformat(m.group(1))
        except ValueError as e:
            raise Refused(f"record {code}: graded_at {stamp!r} is not a date") from e
        out[code] = {"grades": grades, "graded_on": m.group(1), "note": row.get("note") or ""}
    return out


def build_records(plan: Mapping[str, Any], grades: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, dict]]:
    plan_codes = {r["code"] for r in plan["records"]}
    unknown = sorted(set(grades) - plan_codes)
    if unknown:
        raise Refused(f"record code(s) not in the plan: {', '.join(unknown)}")
    missing = sorted(plan_codes - set(grades))
    if missing:
        raise Refused(f"record code(s) with no grades: {', '.join(missing)}")
    out: list[tuple[str, dict]] = []
    for r in sorted(plan["records"], key=lambda r: r["file_name"]):
        g = grades[r["code"]]
        record = {
            "sample_id": r["sample_id"],
            "epoch": r["epoch"],
            "log_eval_id": r["log_eval_id"],
            "worker_prompt_sha256": r["worker_prompt_sha256"],
            "worker_output": r["worker_output"],
            "source": r["source"],
            "seed_note": r["seed_note"],
            "stratum": {d: r["stratum"][d] for d in SUBDIMS},
            "grades": {d: g["grades"][d] for d in SUBDIMS},
            "anchors_sha256": LOCKED_ANCHOR_SHA256,
            "graded_by": GRADED_BY,
            "graded_on": g["graded_on"],
        }
        assert tuple(record) == RECORD_FIELDS
        out.append((r["file_name"], record))
    return out


def record_bytes(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def run(grades_path: Path, plan_path: Path, out_dir: Path) -> list[str]:
    plan = load_plan(plan_path)
    grades = load_grades(grades_path)
    records = build_records(plan, grades)
    if not out_dir.is_dir():
        raise Refused(f"output folder not found: {out_dir}")
    clash = [name for name, _ in records if (out_dir / name).exists()]
    if clash:
        raise Refused(f"output file(s) already exist: {', '.join(clash)}")
    written: list[str] = []
    for name, record in records:
        (out_dir / name).write_bytes(record_bytes(record))
        written.append(name)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the owner's judge-development grades as records.")
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
