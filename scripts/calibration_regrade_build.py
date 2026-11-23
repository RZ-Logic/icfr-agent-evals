"""Build the blind re-grade sheet for the intra-rater check. NO API key, no model call.

Usage:
  python scripts/calibration_regrade_build.py [--sheet PATH] [--plan PATH] [--first-grades PATH]
      [--first-sittings PATH] [--out DIR]

Pre-registration section 7, judge remedy order step 2: before any grader change, the
author re-grades the calibration subset at least 48 hours after the first grading,
without access to the first-pass grades. This script makes the sheet for that re-grade.

It shows the same 40 records the first sheet showed, byte for byte: the narrative, the
facts sheet, the case file and the committed level texts. It takes them from the first
sheet itself (calibration-sheet.html), after checking that sheet's SHA-256 against the
sealed plan, so nothing is rebuilt and nothing can drift. Each record gets a new code
(prefix Q-, where the first sheet used R-) and the records come in a new order, both
drawn from a seed that follows from the first sheet's id. The sheet gets its own sheet
id and its own browser storage key, so the first sheet's answers and notes never load
into it, and a grades file saved from the first sheet is refused by the page.

The page refuses a grade and refuses to start a sitting before the earliest start (the
sheet data carries it as not_before), so no grade can be made early and re-stamped later.
The sheet holds no grade, no note and no first-pass record code. It also holds nothing
the first sheet's blindness check forbids (worker model, case id, source, seed note,
stratum).

Earliest start: 48 hours after the first grading ended, which is the later of the latest
graded_at stamp in the first-pass grades file and the latest sitting end in the first-pass
sittings file. The first-pass grades file must hold exactly the grades committed in
data/calibration/records (the file the import read), so a stale or edited copy cannot set
an early end.
The sheet's introduction states it; scripts/intra_rater_check.py refuses a re-grade
whose grades carry an earlier stamp.

Files written under --out (outside the repo):
  calibration-regrade-sheet.html      the only file the author opens
  _sealed/regrade-plan.json           new code to first-pass code and record file, the
                                      sheet ids and hashes, the earliest start; read by
                                      scripts/intra_rater_check.py

stdout prints totals only: records, the sheet id, the sheet hash and the earliest start.

Exit codes:
  0  written
  3  refused (nothing written): a missing or unreadable input; a first sheet whose
     SHA-256 or sheet id is not the plan's, or whose records are not the plan's 40
     codes; a first-pass grades file from another sheet, or one without a full set of
     graded_at stamps; an output folder inside the repo; an existing output; or a sheet
     that would show sealed text or a first-pass code
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location.
import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import random
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
import judge_dev_build as jdb  # noqa: E402
from icfr_agent_evals import rubric  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
REGRADE_ROUND = 1
GAP = dt.timedelta(hours=48)
CODE_PREFIX = "Q-"
CODE_ALPHABET = "ACDEFHJKLMNPRTUVWXY3479"
FIRST_CODE_RE = re.compile(r"\bR-[ACDEFHJKLMNPRTUVWXY3479]{4}\b")
SHEET_NAME = "calibration-regrade-sheet.html"
SHEET_TITLE = "Calibration subset re-grade"
PLAN_NAME = "regrade-plan.json"
GRADES_FILE = "calibration-regrade-grades.csv"
SITTINGS_FILE = "calibration-regrade-sittings.csv"
STORAGE_PREFIX = "calibration-regrade-"
DEFAULT_SOURCE = cb.DEFAULT_OUT
DEFAULT_SHEET = DEFAULT_SOURCE / cb.SHEET_NAME
DEFAULT_PLAN = DEFAULT_SOURCE / cb.SEALED_DIR / cb.PLAN_NAME
DEFAULT_FIRST_GRADES = DEFAULT_SOURCE / cb.GRADES_FILE
DEFAULT_FIRST_SITTINGS = DEFAULT_SOURCE / cb.SITTINGS_FILE
DEFAULT_OUT = DEFAULT_SOURCE / "regrade"
DEFAULT_RECORDS = REPO / "data" / "calibration" / "records"
# The page refuses grades and sittings before the earliest start.
_NOT_BEFORE_JS = (
    'function notYet(){ if (D.not_before && Date.now() < Date.parse(D.not_before)) { '
    'status("The re-grade opens at " + D.not_before_text + ". Nothing was recorded."); return true; } return false; }\n')
_SCOPE_OPEN = '(function(){\n"use strict";\n'
_REGRADE_JS_EDITS = (
    ('function setGrade(dim, v){ var r = R[state.pos], x = g(r.code);',
     'function setGrade(dim, v){ if (notYet()) return; var r = R[state.pos], x = g(r.code);'),
    ('function start(){ if (open())',
     'function start(){ if (notYet()) return; if (open())'),
)
_DATA_RE = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)


class Refused(Exception):
    """A refusal: nothing is written."""


REFUSALS = (Refused, cb.Refused, jdb.Refused)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise Refused(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refused(f"cannot read {what}: {type(e).__name__}") from e


def sheet_payload(sheet_text: str) -> dict:
    """The data object embedded in a grading sheet."""
    found = _DATA_RE.findall(sheet_text)
    if len(found) != 1:
        raise Refused("the first sheet does not hold exactly one data block")
    try:
        data = json.loads(found[0])
    except json.JSONDecodeError as e:
        raise Refused("the first sheet's data block is not JSON") from e
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise Refused("the first sheet's data block has no records list")
    return data


def check_source(sheet_path: Path, plan: Mapping[str, Any]) -> tuple[dict, str]:
    """The first sheet's data, checked against the sealed plan: same SHA-256, same sheet id,
    and records that are exactly the plan's codes."""
    if not sheet_path.is_file():
        raise Refused(f"the first sheet not found: {sheet_path}")
    raw = sheet_path.read_bytes()
    sha = _sha256_bytes(raw)
    if sha != plan.get("sheet_sha256"):
        raise Refused("the first sheet's SHA-256 is not the one the sealed plan records; it is not the sheet "
                      "that was graded")
    data = sheet_payload(raw.decode("utf-8"))
    if data.get("sheet_id") != plan.get("sheet_id") or not cb.SHEET_ID_RE.match(str(plan.get("sheet_id") or "")):
        raise Refused("the first sheet's id is not the sealed plan's sheet id")
    plan_codes = sorted(r["code"] for r in plan.get("records") or [])
    sheet_codes = sorted(r.get("code") for r in data["records"])
    if len(plan_codes) != cb.SET_SIZE or sheet_codes != plan_codes:
        raise Refused(f"the first sheet's records are not the plan's {cb.SET_SIZE} record codes")
    if data.get("anchors_sha256") != rubric.llm_anchor_sha256():
        raise Refused("the first sheet was built on anchors other than the committed ones")
    return data, sha


def check_first_grades(grades_path: Path, plan: Mapping[str, Any], records_dir: Path) -> None:
    """The first-pass grades file holds exactly the grades committed in records_dir for the
    plan's records (the file the import read), so its stamps date the real first grading."""
    grades = {}
    for row in csv.DictReader(io.StringIO(grades_path.read_bytes().decode("utf-8-sig"), newline="")):
        grades[(row.get("record_code") or "").strip()] = row
    for r in plan["records"]:
        p = records_dir / r["file_name"]
        if not p.is_file():
            raise Refused(f"the committed first-pass record {r['file_name']} not found")
        committed = (json.loads(p.read_text(encoding="utf-8")).get("grades") or {})
        row = grades.get(r["code"])
        if row is None or any(str(committed.get(d)) != (row.get(d) or "").strip() for d in cb.SUBDIMS):
            raise Refused("the first-pass grades file does not hold the grades committed in "
                          f"{records_dir.name}; it is not the file the import read")


def first_grading_end(grades_path: Path, sheet_id: str) -> dt.datetime:
    """The latest graded_at stamp in the first-pass grades file (a timezone-aware time).
    Refuses a file from another sheet, a row without a full timestamp with an offset, or a
    file that does not grade every record."""
    if not grades_path.is_file():
        raise Refused(f"the first-pass grades file not found: {grades_path}")
    try:
        text = grades_path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read the first-pass grades file: {type(e).__name__}") from e
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if len(rows) != cb.SET_SIZE:
        raise Refused(f"the first-pass grades file has {len(rows)} rows, not {cb.SET_SIZE}")
    stamps = []
    for row in rows:
        if (row.get("sheet_id") or "").strip() != sheet_id:
            raise Refused("the first-pass grades file was not saved from the first sheet")
        stamps.append(parse_stamp(row.get("graded_at"), "the first-pass grades file"))
    return max(stamps)


def last_sitting_end(sittings_path: Path) -> dt.datetime:
    """The latest sitting end in the first-pass sittings file. Refuses a missing file, a
    file with no sitting, or a sitting left open (no end)."""
    if not sittings_path.is_file():
        raise Refused(f"the first-pass sittings file not found: {sittings_path}")
    try:
        text = sittings_path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read the first-pass sittings file: {type(e).__name__}") from e
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        raise Refused("the first-pass sittings file records no sitting")
    ends = []
    for row in rows:
        if not (row.get("end") or "").strip():
            raise Refused(f"first-pass sitting {row.get('sitting')} has no end")
        ends.append(parse_stamp(row.get("end"), "the first-pass sittings file"))
    return max(ends)


def parse_stamp(value: Any, where: str) -> dt.datetime:
    """A graded_at stamp as the sheet writes it: an ISO date and time with a UTC offset."""
    s = (value or "").strip()
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise Refused(f"{where}: graded_at {s!r} is not a date and time") from e
    if t.tzinfo is None:
        raise Refused(f"{where}: graded_at {s!r} has no UTC offset, so the 48-hour gap cannot be checked")
    return t


def regrade_seed(source_sheet_id: str) -> str:
    return f"{source_sheet_id}|regrade-{REGRADE_ROUND}"


def new_codes(first_codes: Sequence[str], seed: str) -> dict[str, str]:
    """First-pass code to re-grade code. Re-grade codes start with Q-, so no first-pass code
    (R-) is reused."""
    rng = random.Random(f"{seed}|codes")
    taken: set[str] = set()
    out: dict[str, str] = {}
    for code in sorted(first_codes):
        while True:
            c = CODE_PREFIX + "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
            if c not in taken:
                taken.add(c)
                out[code] = c
                break
    return out


def regrade_records(first_records: Sequence[Mapping[str, Any]], seed: str) -> list[dict]:
    """The first sheet's records, unchanged but for the code, in a new order."""
    codes = new_codes([r["code"] for r in first_records], seed)
    recs = [{**r, "code": codes[r["code"]]} for r in sorted(first_records, key=lambda r: r["code"])]
    random.Random(f"{seed}|order").shuffle(recs)
    return recs


def storage_key(sheet_id: str) -> str:
    return STORAGE_PREFIX + sheet_id[:32]


def _local(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M") + " (UTC" + t.strftime("%z")[:3] + ":" + t.strftime("%z")[3:] + ")"


def regrade_intro(not_before: dt.datetime) -> list[str]:
    return [
        "This is the intra-rater re-grade of the calibration subset (pre-registration section 7). Grade every "
        "record again from the level texts alone, as if for the first time. Do not look at your first grades, "
        "your first notes or any grader output before you finish.",
        f"Start no earlier than {_local(not_before)}, 48 hours after your first grading ended. A grade saved "
        "before then is refused.",
        "Records carry new codes and come in a new order. Start the sitting timer before the first record and end "
        "it when you stop. Answers stay in this browser as you go. When you finish, save the grades file and keep "
        "it; the intra-rater check reads that file.",
    ]


def sheet_data(records: Sequence[Mapping[str, Any]], first_data: Mapping[str, Any], sheet_id: str,
               not_before: dt.datetime) -> dict:
    data = {k: v for k, v in first_data.items()}
    data.update(records=[dict(r) for r in records], intro=regrade_intro(not_before), sheet_id=sheet_id,
                storage_key=storage_key(sheet_id), grades_file=GRADES_FILE, sittings_file=SITTINGS_FILE,
                not_before=not_before.isoformat(), not_before_text=_local(not_before))
    return data


def regrade_script() -> str:
    """The first sheet's page script plus the earliest-start guard on grading and on
    starting a sitting; refuses when an edit no longer finds its text exactly once."""
    js = cb.sheet_script()
    for old, new in _REGRADE_JS_EDITS:
        if js.count(old) != 1:
            raise Refused("the sheet script changed; the earliest-start edits no longer apply")
        js = js.replace(old, new)
    # The guard reads D and status, which live inside the page's function, so it goes in there.
    if js.count(_SCOPE_OPEN) != 1:
        raise Refused("the sheet script changed; the earliest-start guard has no place to go")
    return js.replace(_SCOPE_OPEN, _SCOPE_OPEN + _NOT_BEFORE_JS)


def compute_sheet_id(records: Sequence[Mapping[str, Any]], source_sheet_id: str, seed: str) -> str:
    return _sha256_bytes(cb._canonical({"source_sheet_id": source_sheet_id, "seed": seed,
                                        "records": [dict(r) for r in records]}))


def render_sheet(data: Mapping[str, Any]) -> str:
    page = jdb._PAGE.replace("__TITLE__", html.escape(SHEET_TITLE)).replace("__CSS__", jdb._CSS.strip())
    page = page.replace("__JS__", regrade_script())
    return page.replace("__DATA__", jdb._embed_json(dict(data)))


def blindness_hits(sheet: str, plan_records: Sequence[Mapping[str, Any]], source_sheet_id: str) -> list[str]:
    hits = cb.blindness_hits(sheet, plan_records)
    if FIRST_CODE_RE.search(sheet):
        hits.append("a first-pass record code")
    if source_sheet_id in sheet:
        hits.append("the first sheet's id")
    if cb.storage_key(source_sheet_id) in sheet:
        hits.append("the first sheet's storage key")
    return hits


def build(sheet_path: Path, plan_path: Path, grades_path: Path, out_dir: Path,
          sittings_path: Path = DEFAULT_FIRST_SITTINGS, records_dir: Path = DEFAULT_RECORDS) -> dict:
    if rubric.llm_anchor_sha256() != cb.LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    if _inside_repo(out_dir):
        raise Refused("the output folder stays outside the repo; it holds sealed text")
    out_sheet, out_plan = out_dir / SHEET_NAME, out_dir / cb.SEALED_DIR / PLAN_NAME
    for p in (out_sheet, out_plan):
        if p.exists():
            raise Refused(f"{p.name} exists; the re-grade sheet is built once")
    plan = _read_json(plan_path, "the sealed plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("records"), list):
        raise Refused("the sealed plan has no records list")
    first_data, first_sha = check_source(sheet_path, plan)
    source_id = plan["sheet_id"]
    check_first_grades(grades_path, plan, records_dir)
    first_end = max(first_grading_end(grades_path, source_id), last_sitting_end(sittings_path))
    not_before = first_end + GAP
    seed = regrade_seed(source_id)
    records = regrade_records(first_data["records"], seed)
    sheet_id = compute_sheet_id(records, source_id, seed)
    sheet = render_sheet(sheet_data(records, first_data, sheet_id, not_before))
    hits = blindness_hits(sheet, plan["records"], source_id)
    if hits:
        raise Refused("the re-grade sheet would show sealed text: " + "; ".join(sorted(set(hits))))
    first_by_code = {r["code"]: r for r in plan["records"]}
    back = {nr: old for old, nr in new_codes([r["code"] for r in first_data["records"]], seed).items()}
    sheet_bytes = sheet.encode("utf-8")
    regrade_plan = {
        "what": "Intra-rater re-grade plan (sealed). Do not open before the re-grade is finished.",
        "regrade_round": REGRADE_ROUND,
        "source_sheet_file": sheet_path.name,
        "source_sheet_id": source_id,
        "source_sheet_sha256": first_sha,
        "source_plan_sha256": _sha256_bytes(plan_path.read_bytes()),
        "first_grades_sha256": _sha256_bytes(grades_path.read_bytes()),
        "first_sittings_sha256": _sha256_bytes(sittings_path.read_bytes()),
        "first_grading_ended": first_end.isoformat(),
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "seed": seed,
        "gap_hours": int(GAP.total_seconds() // 3600),
        "not_before": not_before.isoformat(),
        "sheet_file": SHEET_NAME,
        "sheet_id": sheet_id,
        "sheet_sha256": _sha256_bytes(sheet_bytes),
        "grades_file": GRADES_FILE,
        "records": [{"code": r["code"], "position": i, "first_code": back[r["code"]],
                     "file_name": first_by_code[back[r["code"]]]["file_name"]}
                    for i, r in enumerate(records, start=1)],
    }
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sheet, "xb") as f:
        f.write(sheet_bytes)
    with open(out_plan, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(regrade_plan, indent=1, ensure_ascii=False) + "\n")
    return {"records": len(records), "sheet_id": sheet_id, "sheet_sha256": regrade_plan["sheet_sha256"],
            "not_before": _local(not_before), "sheet": str(out_sheet)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--first-grades", type=Path, default=DEFAULT_FIRST_GRADES)
    ap.add_argument("--first-sittings", type=Path, default=DEFAULT_FIRST_SITTINGS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    try:
        res = build(a.sheet, a.plan, a.first_grades, a.out, a.first_sittings)
    except REFUSALS as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"re-grade sheet written: {res['sheet']}")
    print(f"records {res['records']}, sheet id {res['sheet_id'][:12]}, sheet sha256 {res['sheet_sha256'][:12]}")
    print(f"earliest start {res['not_before']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
