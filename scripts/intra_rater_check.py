"""The intra-rater check: the author's first-pass calibration grades against the re-grade.
NO API key, no model call.

Usage:
  python scripts/intra_rater_check.py --grades CSV --sittings CSV --sealed-dir DIR
      [--plan PATH] [--records DIR] [--look PATH] [--out PATH]

Pre-registration section 7, judge remedy order step 2. Before any grader change the author
re-grades the calibration subset at least 48 hours after the first grading, without
access to the first-pass grades (sheet: scripts/calibration_regrade_build.py). Per
LLM-graded sub-dimension this computes author-versus-author ICC(A,1) with its two-sided
McGraw-Wong 95% CI (judge_trust.intra_rater_icc), reported as intra-rater (test-retest)
agreement, never as agreement between two raters. ICC(C,1), the signed mean of re-grade
minus first pass with a paired 95% CI, exact agreement and both score distributions are
reported beside it. No binary check exists at rubric lock, so no kappa is computed.

The decision per sub-dimension follows section 7:
  lower limit at least 0.60   the first-pass grades stay the human reference for the
                              GPT-6.1 Terra acceptance look (and GPT-6.1 Sol's, if step 3
                              runs); the re-grade is reported only as intra-rater agreement
  lower limit below 0.60      that sub-dimension's rubric wording is revised on the
  (or undefined)              judge-development set, never on the calibration subset, and
                              the author grades the subset again under the revised rubric,
                              blind to GPT-6 Luna's grades and to both earlier gradings;
                              those grades become the human reference for Terra's look.
                              An undefined interval counts as below 0.60 (as for the gates,
                              decision D-R20)

Inputs and checks (any failure refuses and writes nothing):
  --grades     the re-grade file saved from calibration-regrade-sheet.html: the regrade
               plan's sheet id on every row, exactly the plan's 40 codes, both grades 0 to
               3 on every row, and every graded_at stamp at or after the plan's earliest
               start (48 hours after the first grading ended)
  --sittings   the re-grade sittings file (required): every sitting starts at or after the
               earliest start and is closed, and every grade's graded_at falls inside a
               sitting, so a grade made early and re-clicked later is caught
  --plan       the sealed regrade plan (calibration-subset/regrade/_sealed/regrade-plan.json)
  --records    data/calibration/records: the first-pass records, every file tracked in HEAD
               with no change, graded against the committed anchors, whose set hash is the
               one the step 1 look recorded
  --look       docs/evidence/calibration/acceptance-look-step-1.json: must exist with
               verdict FAIL; a step 2 look must not exist yet (the check runs before the
               first grader change)

Output: --out, default docs/evidence/calibration/intra-rater-step-1.json, written with
exclusive create. Commit it before the GPT-6.1 Terra look, which takes it as
--intra-rater. When every sub-dimension keeps its first-pass reference, the per-record
pairs stay in that file. When any sub-dimension needs a revision, the per-record pairs go
to a sealed file in --sealed-dir (outside the repo), because the grading that follows
must stay blind to both earlier gradings; the output keeps the aggregates and the sealed
file's SHA-256.

stdout prints aggregates only: no record and no grade.

Exit codes: 0 written, 2 bad arguments, 3 refused.
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location.
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _p in (REPO, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import calibration_import as ci  # noqa: E402
import calibration_regrade_build as rb  # noqa: E402
from icfr_agent_evals import judge_trust as jt  # noqa: E402
from icfr_agent_evals import rubric  # noqa: E402

EXIT_OK, EXIT_ARGS, EXIT_REFUSED = 0, 2, 3
SUBDIMS = tuple(ci.SUBDIMS)
SET_SIZE = ci.SET_SIZE
THRESHOLD = jt.ICC_LOWER_LIMIT
DEFAULT_PLAN = rb.DEFAULT_OUT / ci.cb.SEALED_DIR / rb.PLAN_NAME
DEFAULT_RECORDS = REPO / "data" / "calibration" / "records"
EVIDENCE_DIR = REPO / "docs" / "evidence" / "calibration"
DEFAULT_LOOK = EVIDENCE_DIR / "acceptance-look-step-1.json"
DEFAULT_OUT = EVIDENCE_DIR / "intra-rater-step-1.json"
SEALED_NAME = "intra-rater-step-1-rows.json"
KEEP = "keep"
REVISE = "revise"
KEEP_TEXT = ("the first-pass grades stay the human reference for the GPT-6.1 Terra acceptance look and, if step 3 "
             "runs, the GPT-6.1 Sol look; the re-grade is reported only as intra-rater agreement")
REVISE_TEXT = ("revise this sub-dimension's rubric wording on the judge-development set (never on the calibration "
               "subset), log it in DEVIATIONS.md with these statistics, then grade the calibration subset under the "
               "revised rubric, blind to GPT-6 Luna's grades and to both earlier gradings; those grades are the human "
               "reference for the GPT-6.1 Terra look")

Refused = rb.Refused


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_set_sha256(folder: Path) -> str:
    """The step 1 look's record-set hash: one line "<file name> <SHA-256>" per record file,
    sorted by name, joined with LF (acceptance_look.records_set_sha256)."""
    lines = [f"{p.name} {_sha256_file(p)}\n" for p in sorted(folder.glob("*.json"))]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        cp = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise Refused(f"git {' '.join(args[:2])} failed: {type(e).__name__}") from e
    return cp.stdout


def _rel(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as e:
        raise Refused(f"{path} is not inside the repository") from e


def _inside(path: Path, repo: Path) -> bool:
    try:
        path.resolve().relative_to(repo.resolve())
        return True
    except ValueError:
        return False


def load_plan(path: Path) -> dict:
    plan = rb._read_json(path, "the sealed regrade plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("records"), list):
        raise Refused("the regrade plan has no records list")
    codes = [r.get("code") for r in plan["records"]]
    files = [r.get("file_name") for r in plan["records"]]
    if len(codes) != SET_SIZE or len(set(codes)) != SET_SIZE or len(set(files)) != SET_SIZE:
        raise Refused(f"the regrade plan does not hold {SET_SIZE} distinct records")
    for key in ("sheet_id", "not_before", "anchors_sha256"):
        if not plan.get(key):
            raise Refused(f"the regrade plan has no {key}")
    rb.parse_stamp(plan["not_before"], "the regrade plan")
    if plan["anchors_sha256"] != rubric.llm_anchor_sha256():
        raise Refused("the regrade plan was built on anchors other than the committed ones")
    return plan


def load_regrade(grades_path: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The re-grade, per plan code: both grades and the graded_at time. Refuses a file from
    another sheet, a missing or extra code, a missing or invalid grade and a stamp before
    the earliest start."""
    if ci.grades_sheet_id(grades_path) != plan["sheet_id"]:
        raise Refused("the grades file was not saved from the re-grade sheet; grades from another sheet are never "
                      "read")
    grades = ci.load_grades(grades_path)
    want = {r["code"] for r in plan["records"]}
    missing, extra = sorted(want - set(grades)), sorted(set(grades) - want)
    if missing or extra:
        raise Refused(f"the grades file does not grade exactly the plan's {SET_SIZE} records "
                      f"({len(missing)} missing, {len(extra)} not in the plan)")
    text = grades_path.read_bytes().decode("utf-8-sig")
    not_before = rb.parse_stamp(plan["not_before"], "the regrade plan")
    out: dict[str, dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(text, newline="")):
        code = (row.get("record_code") or "").strip()
        stamp = rb.parse_stamp(row.get("graded_at"), f"record {code}")
        if stamp < not_before:
            raise Refused(f"record {code} was graded at {stamp.isoformat()}, before the earliest start "
                          f"{not_before.isoformat()} (48 hours after the first grading)")
        out[code] = {"grades": grades[code]["grades"], "graded_at": stamp.isoformat()}
    return out


def load_sittings(path: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], list]:
    """The re-grade sittings: every one closed and started at or after the earliest start.
    Returns the summary and the (start, end) windows."""
    if not path.is_file():
        raise Refused(f"the sittings file not found: {path}")
    rows = list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline="")))
    if not rows:
        raise Refused("the sittings file records no sitting")
    not_before = rb.parse_stamp(plan["not_before"], "the regrade plan")
    minutes, windows = 0.0, []
    for row in rows:
        start = rb.parse_stamp(row.get("start"), f"sitting {row.get('sitting')}")
        if start < not_before:
            raise Refused(f"sitting {row.get('sitting')} started before the earliest start {not_before.isoformat()}")
        if not (row.get("end") or "").strip():
            raise Refused(f"sitting {row.get('sitting')} has no end; end it and save the sittings file again")
        end = rb.parse_stamp(row.get("end"), f"sitting {row.get('sitting')}")
        minutes += max(0.0, (end - start).total_seconds() / 60)
        windows.append((start, end))
    summary = {"file_sha256": _sha256_file(path), "sittings": len(rows), "minutes": round(minutes, 1),
               "first_start": min(w[0] for w in windows).isoformat()}
    return summary, windows


def check_inside_sittings(regrade: Mapping[str, Mapping[str, Any]], windows: Sequence[tuple]) -> None:
    """Every grade's graded_at falls inside a sitting (one minute of slack each side for the
    timestamp rounding)."""
    slack = dt.timedelta(minutes=1)
    for code, g in regrade.items():
        t = rb.parse_stamp(g["graded_at"], f"record {code}")
        if not any(s - slack <= t <= e + slack for s, e in windows):
            raise Refused(f"record {code} was graded at {t.isoformat()}, outside every recorded sitting")


def load_first_pass(records_dir: Path, plan: Mapping[str, Any], repo: Path, look: Mapping[str, Any]) -> dict[str, dict]:
    """The committed first-pass records the plan names, per file name: tracked in HEAD with
    no change, graded against the committed anchors, and the same set the step 1 look used."""
    rel = _rel(records_dir, repo)
    tracked = set(_git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", rel).splitlines())
    if _git(repo, "status", "--porcelain", "--", rel).strip():
        raise Refused(f"{rel} has uncommitted changes; the first-pass grades are the committed ones")
    on_disk = sorted(records_dir.glob("*.json"))
    if len(on_disk) != SET_SIZE:
        raise Refused(f"{rel} holds {len(on_disk)} records, not {SET_SIZE}")
    if records_set_sha256(records_dir) != look.get("records_sha256"):
        raise Refused("the first-pass records are not the set the step 1 acceptance look recorded")
    out: dict[str, dict] = {}
    for r in plan["records"]:
        p = records_dir / r["file_name"]
        if f"{rel}/{p.name}" not in tracked or not p.is_file():
            raise Refused(f"the first-pass record {p.name} is not tracked in HEAD")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("anchors_sha256") != rubric.llm_anchor_sha256():
            raise Refused(f"{p.name} was graded against anchors other than the committed ones")
        g = rec.get("grades") or {}
        if any(isinstance(g.get(d), bool) or g.get(d) not in (0, 1, 2, 3) for d in SUBDIMS):
            raise Refused(f"{p.name} does not hold a 0 to 3 grade on both sub-dimensions")
        out[r["file_name"]] = rec
    return out


def load_look(path: Path, step2_path: Path) -> dict:
    look = rb._read_json(path, "the step 1 acceptance look")
    if not isinstance(look, dict) or look.get("grader_step") != 1:
        raise Refused("the look file is not the step 1 acceptance look")
    if (look.get("verdict") or {}).get("overall") != "FAIL":
        raise Refused("the step 1 acceptance look did not fail; the intra-rater check runs only after a failed look")
    if step2_path.exists():
        raise Refused("the step 2 acceptance look exists; the intra-rater check runs before the first grader change")
    return look


def compute(pairs: Mapping[str, Sequence[tuple[int, int]]]) -> dict[str, Any]:
    """Per sub-dimension: pairs are (first pass, re-grade), one per record."""
    per: dict[str, Any] = {}
    for d in SUBDIMS:
        p = [tuple(x) for x in pairs[d]]
        a1 = jt.intra_rater_icc(p)
        lower = a1.get("ci_lower")
        decision = KEEP if lower is not None and lower >= THRESHOLD else REVISE
        per[d] = {
            "icc_a1": a1,
            "icc_c1": jt.icc_c1(p),
            "mean_difference_regrade_minus_first": jt.paired_mean_difference([(b, a) for a, b in p]),
            "exact_agreement": sum(1 for a, b in p if a == b),
            "within_one": sum(1 for a, b in p if abs(a - b) <= 1),
            "n": len(p),
            "distribution": {"first_pass": jt.score_distribution(a for a, _ in p),
                             "regrade": jt.score_distribution(b for _, b in p)},
            "decision": decision,
            "next": KEEP_TEXT if decision == KEEP else REVISE_TEXT,
        }
    return per


def run(grades_path: Path, sealed_dir: Path, sittings_path: Path, plan_path: Path, records_dir: Path,
        look_path: Path, out_path: Path, repo: Path = REPO) -> dict:
    if out_path.exists():
        raise Refused(f"{out_path.name} exists; the intra-rater check runs once")
    if _inside(sealed_dir, repo):
        raise Refused("--sealed-dir stays outside the repository")
    plan = load_plan(plan_path)
    look = load_look(look_path, look_path.parent / "acceptance-look-step-2.json")
    first = load_first_pass(records_dir, plan, repo, look)
    regrade = load_regrade(grades_path, plan)
    sittings, windows = load_sittings(sittings_path, plan)
    check_inside_sittings(regrade, windows)
    rows = []
    for r in sorted(plan["records"], key=lambda r: r["file_name"]):
        f, g = first[r["file_name"]]["grades"], regrade[r["code"]]
        rows.append({"file_name": r["file_name"], "first_pass": {d: f[d] for d in SUBDIMS},
                     "regrade": {d: g["grades"][d] for d in SUBDIMS}, "regraded_at": g["graded_at"]})
    per = compute({d: [(row["first_pass"][d], row["regrade"][d]) for row in rows] for d in SUBDIMS})
    revise = [d for d in SUBDIMS if per[d]["decision"] == REVISE]
    sealed_file = sealed_dir / SEALED_NAME
    if revise and sealed_file.exists():
        raise Refused(f"{sealed_file} exists; the intra-rater check runs once")
    stamps = [rb.parse_stamp(row["regraded_at"], "re-grade") for row in rows]
    result: dict[str, Any] = {
        "check": "intra-rater check, judge remedy order step 2 (pre-registration section 7)",
        "agreement": jt.INTRA_RATER_LABEL,
        "statistic": "ICC(A,1), author first pass versus author re-grade, two-sided 95% McGraw-Wong CI",
        "threshold": THRESHOLD,
        "kappa": "does not apply: the rubric defines no binary checks at rubric lock (pre-registration section 7)",
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "head": _git(repo, "rev-parse", "HEAD").strip(),
        "look_file": look_path.name,
        "look_sha256": _sha256_file(look_path),
        "records_sha256": look["records_sha256"],
        "regrade_plan_sha256": _sha256_file(plan_path),
        "regrade_sheet_id": plan["sheet_id"],
        "regrade_grades_sha256": _sha256_file(grades_path),
        "first_grading_ended": plan.get("first_grading_ended"),
        "earliest_start": plan["not_before"],
        "regrade_first_stamp": min(stamps).isoformat(),
        "regrade_last_stamp": max(stamps).isoformat(),
        "sittings": sittings,
        "records": len(rows),
        "per_subdimension": per,
        "revise": revise,
        "next": ("every sub-dimension keeps its first-pass reference: move the grader to GPT-6.1 Terra on the Batch "
                 "API and run its acceptance look with --intra-rater on this file"
                 if not revise else
                 "revise the wording of " + ", ".join(revise) + " on the judge-development set, then grade the "
                 "calibration subset blind under the revised rubric before the GPT-6.1 Terra look"),
    }
    if revise:
        body = json.dumps({"what": "intra-rater per-record pairs (sealed): the grading under the revised rubric "
                                   "stays blind to them", "rows": rows}, indent=1, ensure_ascii=False) + "\n"
        result["rows"] = {"sealed": True, "file": SEALED_NAME,
                          "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "records": len(rows)}
    else:
        body = None
        result["rows"] = rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if body is not None:
        sealed_dir.mkdir(parents=True, exist_ok=True)
        with open(sealed_file, "x", encoding="utf-8", newline="\n") as f:
            f.write(body)
    with open(out_path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return result


def _f(x: Any) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", type=Path, required=True)
    ap.add_argument("--sealed-dir", type=Path, required=True)
    ap.add_argument("--sittings", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--look", type=Path, default=DEFAULT_LOOK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return EXIT_ARGS
    try:
        res = run(a.grades, a.sealed_dir, a.sittings, a.plan, a.records, a.look, a.out, repo=REPO)
    except (*rb.REFUSALS, ci.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"intra-rater check written: {a.out}")
    for d in SUBDIMS:
        s = res["per_subdimension"][d]
        print(f"  {d}: ICC(A,1) {_f(s['icc_a1'].get('estimate'))}, lower limit {_f(s['icc_a1'].get('ci_lower'))} "
              f"(bar {THRESHOLD}); exact agreement {s['exact_agreement']}/{s['n']}; decision {s['decision']}")
    print(f"next: {res['next']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
