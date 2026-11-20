"""The intra-rater re-grade sheet (scripts/calibration_regrade_build.py) and the intra-rater
check (scripts/intra_rater_check.py), judge remedy order step 2 of pre-registration section 7.

Keyless and offline: synthetic sheets, plans and records in temporary folders; the check's
git conditions run against a temporary git repository.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("calibration_regrade_build")
ir = _load("intra_rater_check")
cb = rb.cb
jdb = rb.jdb
from icfr_agent_evals import rubric  # noqa: E402

G, C = ir.SUBDIMS
N = 40
TZ = dt.timezone(dt.timedelta(hours=-5))
FIRST_END = dt.datetime(2026, 11, 19, 19, 36, tzinfo=TZ)
SITTING_END = dt.datetime(2026, 11, 19, 19, 52, tzinfo=TZ)


# --- synthetic first sheet, plan and grades -------------------------------------------------

def _first_codes():
    rng = __import__("random").Random(7)
    taken: set[str] = set()
    return [jdb.record_code(rng, taken) for _ in range(N)]


def _first(i):
    return {G: i % 4, C: (i // 4) % 4}


def _sheet_records(codes):
    return [{"code": c, "narrative": f"Narrative number {i} for the shortfall.", "case_input": f"Case file {i}",
             "case_facts": [], "output_facts": []} for i, c in enumerate(codes)]


def _render(records, sheet_id):
    data = {"records": records, "dims": [G, C], "anchors_sha256": rubric.llm_anchor_sha256(),
            "intro": ["Grade each record."], "sheet_id": sheet_id, "storage_key": cb.storage_key(sheet_id),
            "grades_file": cb.GRADES_FILE, "sittings_file": cb.SITTINGS_FILE}
    page = jdb._PAGE.replace("__TITLE__", "t").replace("__CSS__", "").replace("__JS__", cb.sheet_script())
    return page.replace("__DATA__", jdb._embed_json(data))


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "calibration-subset"
    (src / "_sealed").mkdir(parents=True)
    codes = _first_codes()
    sheet_id = hashlib.sha256(b"first sheet").hexdigest()
    sheet = _render(_sheet_records(codes), sheet_id)
    (src / "calibration-sheet.html").write_bytes(sheet.encode("utf-8"))
    plan = {"sheet_id": sheet_id, "sheet_sha256": hashlib.sha256(sheet.encode("utf-8")).hexdigest(),
            "records": [{"code": c, "file_name": f"calibration-case-{i:03d}.json", "sample_id": f"case-{i:03d}",
                         "seed_note": None, "stratum_basis": None} for i, c in enumerate(codes)]}
    (src / "_sealed" / "calibration-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    stamps = [FIRST_END - dt.timedelta(minutes=i) for i in range(N)]
    _write_csv(src / "calibration-grades.csv", ["record_code", G, C, "note", "graded_at", "sheet_id"],
               [[c, _first(i)[G], _first(i)[C], "", s.isoformat(), sheet_id] for i, (c, s) in enumerate(zip(codes, stamps))])
    recs = src / "records"
    recs.mkdir()
    for i, r in enumerate(plan["records"]):
        (recs / r["file_name"]).write_text(json.dumps({"grades": _first(i), "anchors_sha256": rubric.llm_anchor_sha256()}),
                                           encoding="utf-8")
    _write_csv(src / "calibration-sittings.csv", ["sitting", "start", "end"],
               [[1, (SITTING_END - dt.timedelta(hours=2)).isoformat(), SITTING_END.isoformat()]])
    return src, plan


def _build(src, out=None):
    out = out or src / "regrade"
    return rb.build(src / "calibration-sheet.html", src / "_sealed" / "calibration-plan.json",
                    src / "calibration-grades.csv", out, src / "calibration-sittings.csv", src / "records"), out


# --- the re-grade sheet ---------------------------------------------------------------------

def test_build_makes_a_blind_sheet_of_the_same_records(source):
    src, plan = source
    res, out = _build(src)
    sheet = (out / rb.SHEET_NAME).read_text(encoding="utf-8")
    data = rb.sheet_payload(sheet)
    first = rb.sheet_payload((src / "calibration-sheet.html").read_text(encoding="utf-8"))
    assert len(data["records"]) == N and all(r["code"].startswith("Q-") for r in data["records"])
    strip = lambda recs: sorted(json.dumps({k: v for k, v in r.items() if k != "code"}, sort_keys=True) for r in recs)
    assert strip(data["records"]) == strip(first["records"])
    assert [r["narrative"] for r in data["records"]] != [r["narrative"] for r in first["records"]]
    assert not re.search(r"\bR-[A-Z0-9]{4}\b", sheet) and plan["sheet_id"] not in sheet
    assert data["storage_key"] != first["storage_key"] and data["grades_file"] == rb.GRADES_FILE
    rp = json.loads((out / "_sealed" / rb.PLAN_NAME).read_text(encoding="utf-8"))
    assert rp["sheet_id"] == data["sheet_id"] == res["sheet_id"]
    assert {r["first_code"] for r in rp["records"]} == {r["code"] for r in plan["records"]}
    assert rp["sheet_sha256"] == hashlib.sha256((out / rb.SHEET_NAME).read_bytes()).hexdigest()


def test_earliest_start_is_48_hours_after_the_later_of_last_grade_and_last_sitting(source):
    src, _ = source
    _, out = _build(src)
    rp = json.loads((out / "_sealed" / rb.PLAN_NAME).read_text(encoding="utf-8"))
    assert dt.datetime.fromisoformat(rp["not_before"]) == SITTING_END + dt.timedelta(hours=48)


def test_build_is_deterministic(source, tmp_path):
    src, _ = source
    a, _ = _build(src, tmp_path / "a")
    b, _ = _build(src, tmp_path / "b")
    assert (tmp_path / "a" / rb.SHEET_NAME).read_bytes() == (tmp_path / "b" / rb.SHEET_NAME).read_bytes()


def test_build_refuses_a_sheet_that_is_not_the_graded_one(source):
    src, _ = source
    p = src / "calibration-sheet.html"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(rb.Refused, match="SHA-256"):
        _build(src)


def test_build_refuses_grades_from_another_sheet(source):
    src, _ = source
    p = src / "calibration-grades.csv"
    p.write_text(p.read_text(encoding="utf-8").replace(source[1]["sheet_id"], "f" * 64), encoding="utf-8")
    with pytest.raises(rb.Refused, match="not saved from the first sheet"):
        _build(src)


def test_build_refuses_an_open_sitting_and_a_stamp_without_offset(source):
    src, _ = source
    _write_csv(src / "calibration-sittings.csv", ["sitting", "start", "end"], [[1, SITTING_END.isoformat(), ""]])
    with pytest.raises(rb.Refused, match="has no end"):
        _build(src)
    with pytest.raises(rb.Refused, match="no UTC offset"):
        rb.parse_stamp("2026-11-19T10:00:00", "x")


def test_build_refuses_an_existing_output_and_an_output_in_the_repo(source):
    src, _ = source
    _build(src)
    with pytest.raises(rb.Refused, match="built once"):
        _build(src)
    with pytest.raises(rb.Refused, match="outside the repo"):
        _build(src, REPO / "regrade-test-never-written")


# --- the intra-rater check -------------------------------------------------------------------

def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def checked(source, tmp_path):
    """A temporary repository with the first-pass records committed and a failed step 1 look."""
    src, plan = source
    _, out = _build(src)
    rp = json.loads((out / "_sealed" / rb.PLAN_NAME).read_text(encoding="utf-8"))
    repo = tmp_path / "repo"
    recs = repo / "data" / "calibration" / "records"
    recs.mkdir(parents=True)
    first = {}
    for i, r in enumerate(plan["records"]):
        g = _first(i)
        first[r["file_name"]] = g
        (recs / r["file_name"]).write_text(json.dumps({"grades": g, "anchors_sha256": rubric.llm_anchor_sha256()}),
                                           encoding="utf-8")
    ev = repo / "docs" / "evidence" / "calibration"
    ev.mkdir(parents=True)
    look = {"grader_step": 1, "verdict": {"overall": "FAIL"}, "records_sha256": ir.records_set_sha256(recs)}
    (ev / "acceptance-look-step-1.json").write_text(json.dumps(look), encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    return {"repo": repo, "recs": recs, "ev": ev, "rp": rp, "plan_path": out / "_sealed" / rb.PLAN_NAME,
            "first": first, "tmp": tmp_path}


def _regrade_csv(ctx, change=lambda name, g: g, stamp=None, sheet_id=None, drop=0):
    rp = ctx["rp"]
    stamp = stamp or dt.datetime.fromisoformat(rp["not_before"]) + dt.timedelta(hours=1)
    rows = []
    for r in rp["records"][drop:]:
        g = change(r["file_name"], dict(ctx["first"][r["file_name"]]))
        rows.append([r["code"], g[G], g[C], "", stamp.isoformat(), sheet_id or rp["sheet_id"]])
    path = ctx["tmp"] / "regrade-grades.csv"
    _write_csv(path, ["record_code", G, C, "note", "graded_at", "sheet_id"], rows)
    return path


def _sittings(ctx, start=None, hours=3):
    start = start or dt.datetime.fromisoformat(ctx["rp"]["not_before"])
    path = ctx["tmp"] / "regrade-sittings.csv"
    _write_csv(path, ["sitting", "start", "end"], [[1, start.isoformat(), (start + dt.timedelta(hours=hours)).isoformat()]])
    return path


def _run(ctx, grades, out=None, sittings=None):
    sittings = sittings or _sittings(ctx)
    return ir.run(grades, ctx["tmp"] / "sealed", sittings, ctx["plan_path"], ctx["recs"],
                  ctx["ev"] / "acceptance-look-step-1.json", out or ctx["ev"] / "intra-rater-step-1.json",
                  repo=ctx["repo"])


def test_identical_regrade_keeps_both_references_and_keeps_rows(checked):
    res = _run(checked, _regrade_csv(checked))
    for d in (G, C):
        s = res["per_subdimension"][d]
        assert s["decision"] == ir.KEEP and s["icc_a1"]["estimate"] == 1.0 and s["exact_agreement"] == N
    assert res["revise"] == [] and isinstance(res["rows"], list) and len(res["rows"]) == N
    assert not (checked["tmp"] / "sealed").exists()
    assert (checked["ev"] / "intra-rater-step-1.json").exists()


def test_a_disagreeing_regrade_calls_for_revision_and_seals_the_rows(checked):
    flip = {0: 3, 1: 2, 2: 1, 3: 0}
    res = _run(checked, _regrade_csv(checked, change=lambda n, g: {G: flip[g[G]], C: g[C]}))
    assert res["per_subdimension"][G]["decision"] == ir.REVISE
    assert res["per_subdimension"][C]["decision"] == ir.KEEP
    assert res["revise"] == [G] and res["rows"]["sealed"] is True
    sealed = checked["tmp"] / "sealed" / ir.SEALED_NAME
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == res["rows"]["sha256"]
    body = (checked["ev"] / "intra-rater-step-1.json").read_text(encoding="utf-8")
    assert "regraded_at" not in body and "file_name" not in body


def test_check_refuses_a_regrade_before_the_earliest_start(checked):
    early = dt.datetime.fromisoformat(checked["rp"]["not_before"]) - dt.timedelta(minutes=1)
    with pytest.raises(rb.Refused, match="before the earliest start"):
        _run(checked, _regrade_csv(checked, stamp=early))


def test_check_refuses_another_sheet_and_a_missing_record(checked):
    with pytest.raises(rb.Refused, match="re-grade sheet"):
        _run(checked, _regrade_csv(checked, sheet_id="e" * 64))
    with pytest.raises(rb.Refused, match="exactly the plan"):
        _run(checked, _regrade_csv(checked, drop=1))


def test_check_refuses_a_sitting_before_the_earliest_start(checked):
    nb = dt.datetime.fromisoformat(checked["rp"]["not_before"])
    path = checked["tmp"] / "sittings.csv"
    _write_csv(path, ["sitting", "start", "end"], [[1, (nb - dt.timedelta(hours=1)).isoformat(), nb.isoformat()]])
    with pytest.raises(rb.Refused, match="started before"):
        _run(checked, _regrade_csv(checked), sittings=path)


def test_check_refuses_uncommitted_or_changed_first_pass_records(checked):
    p = sorted(checked["recs"].glob("*.json"))[0]
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(rb.Refused, match="uncommitted"):
        _run(checked, _regrade_csv(checked))


def test_check_refuses_without_a_failed_step_1_look_or_after_step_2(checked):
    look = checked["ev"] / "acceptance-look-step-1.json"
    (checked["ev"] / "acceptance-look-step-2.json").write_text("{}", encoding="utf-8")
    with pytest.raises(rb.Refused, match="before the first grader change"):
        _run(checked, _regrade_csv(checked))
    (checked["ev"] / "acceptance-look-step-2.json").unlink()
    body = json.loads(look.read_text(encoding="utf-8"))
    body["verdict"]["overall"] = "PASS"
    look.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(rb.Refused, match="did not fail"):
        _run(checked, _regrade_csv(checked))


def test_check_runs_once_and_keeps_the_sealed_dir_outside_the_repo(checked):
    g = _regrade_csv(checked)
    _run(checked, g)
    with pytest.raises(rb.Refused, match="runs once"):
        _run(checked, g)
    with pytest.raises(rb.Refused, match="outside the repository"):
        ir.run(g, checked["repo"] / "sealed", _sittings(checked), checked["plan_path"], checked["recs"],
               checked["ev"] / "acceptance-look-step-1.json", checked["ev"] / "other.json", repo=checked["repo"])


def test_main_prints_aggregates_only(checked, capsys, monkeypatch):
    monkeypatch.setattr(ir, "REPO", checked["repo"])
    out = checked["ev"] / "intra-rater-step-1.json"
    code = ir.main(["--grades", str(_regrade_csv(checked)), "--sealed-dir", str(checked["tmp"] / "sealed"),
                    "--sittings", str(_sittings(checked)),
                    "--plan", str(checked["plan_path"]), "--records", str(checked["recs"]),
                    "--look", str(checked["ev"] / "acceptance-look-step-1.json"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == ir.EXIT_OK and "decision keep" in printed
    for r in checked["rp"]["records"]:
        assert r["code"] not in printed and r["file_name"] not in printed and r["first_code"] not in printed


def test_main_refuses_with_exit_3(checked, capsys, monkeypatch):
    monkeypatch.setattr(ir, "REPO", checked["repo"])
    code = ir.main(["--grades", str(checked["tmp"] / "missing.csv"), "--sealed-dir", str(checked["tmp"] / "s"),
                    "--sittings", str(_sittings(checked)),
                    "--plan", str(checked["plan_path"]), "--records", str(checked["recs"]),
                    "--look", str(checked["ev"] / "acceptance-look-step-1.json"),
                    "--out", str(checked["ev"] / "x.json")])
    assert code == ir.EXIT_REFUSED and "refused" in capsys.readouterr().err


# --- review findings ----------------------------------------------------------------------------

def test_the_page_refuses_grades_and_sittings_before_the_earliest_start(source):
    src, _ = source
    _, out = _build(src)
    sheet = (out / rb.SHEET_NAME).read_text(encoding="utf-8")
    data = rb.sheet_payload(sheet)
    assert dt.datetime.fromisoformat(data["not_before"]) == SITTING_END + dt.timedelta(hours=48)
    assert "function setGrade(dim, v){ if (notYet()) return;" in sheet
    assert "function start(){ if (notYet()) return;" in sheet and "function notYet()" in sheet


def test_a_later_last_grade_than_sitting_end_sets_the_earliest_start(source):
    src, _ = source
    late = SITTING_END + dt.timedelta(hours=1)
    _write_csv(src / "calibration-sittings.csv", ["sitting", "start", "end"],
               [[1, (FIRST_END - dt.timedelta(hours=3)).isoformat(), (FIRST_END - dt.timedelta(hours=2)).isoformat()]])
    rows = list(__import__("csv").reader(open(src / "calibration-grades.csv", encoding="utf-8")))
    rows[1][4] = late.isoformat()
    _write_csv(src / "calibration-grades.csv", rows[0], rows[1:])
    _, out = _build(src)
    rp = json.loads((out / "_sealed" / rb.PLAN_NAME).read_text(encoding="utf-8"))
    assert dt.datetime.fromisoformat(rp["not_before"]) == late + dt.timedelta(hours=48)


def test_build_refuses_a_first_grades_file_that_is_not_the_imported_one(source):
    src, plan = source
    p = src / "records" / plan["records"][0]["file_name"]
    body = json.loads(p.read_text(encoding="utf-8"))
    body["grades"][G] = (body["grades"][G] + 1) % 4
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(rb.Refused, match="not the file the import read"):
        _build(src)


def test_each_new_code_shows_the_record_its_plan_entry_names(source):
    src, plan = source
    _, out = _build(src)
    data = rb.sheet_payload((out / rb.SHEET_NAME).read_text(encoding="utf-8"))
    rp = json.loads((out / "_sealed" / rb.PLAN_NAME).read_text(encoding="utf-8"))
    narrative = {r["code"]: r["narrative"] for r in data["records"]}
    for r in rp["records"]:
        i = int(r["file_name"].split("-")[-1].split(".")[0])
        assert narrative[r["code"]] == f"Narrative number {i} for the shortfall."


def test_a_grade_outside_every_sitting_is_refused(checked):
    nb = dt.datetime.fromisoformat(checked["rp"]["not_before"])
    sittings = _sittings(checked, start=nb + dt.timedelta(hours=5), hours=1)
    with pytest.raises(rb.Refused, match="outside every recorded sitting"):
        _run(checked, _regrade_csv(checked), sittings=sittings)


def test_an_open_regrade_sitting_is_refused(checked):
    nb = dt.datetime.fromisoformat(checked["rp"]["not_before"])
    path = checked["tmp"] / "open.csv"
    _write_csv(path, ["sitting", "start", "end"], [[1, nb.isoformat(), ""]])
    with pytest.raises(rb.Refused, match="has no end"):
        _run(checked, _regrade_csv(checked), sittings=path)


def test_one_early_row_among_valid_ones_is_refused(checked):
    path = _regrade_csv(checked)
    rows = list(__import__("csv").reader(open(path, encoding="utf-8")))
    rows[5][4] = (dt.datetime.fromisoformat(checked["rp"]["not_before"]) - dt.timedelta(seconds=1)).isoformat()
    _write_csv(path, rows[0], rows[1:])
    with pytest.raises(rb.Refused, match="before the earliest start"):
        _run(checked, path)


def test_an_undefined_interval_calls_for_revision():
    res = ir.compute({G: [(2, 2)] * N, C: [(i % 4, i % 4) for i in range(N)]})
    assert res[G]["icc_a1"]["ci_lower"] is None and res[G]["decision"] == ir.REVISE
    assert res[C]["decision"] == ir.KEEP


def test_the_step_2_look_refuses_a_revise_record_while_the_anchors_are_unchanged(checked):
    al = _load("acceptance_look")
    flip = {0: 3, 1: 2, 2: 1, 3: 0}
    res = _run(checked, _regrade_csv(checked, change=lambda n, g: {G: flip[g[G]], C: g[C]}))
    assert res["revise"] == [G]
    _git(checked["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(checked["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "y")
    rec = checked["ev"] / "intra-rater-step-1.json"
    with pytest.raises(al.Refused, match="calls for a rubric revision"):
        al.check_intra_rater(str(rec), 2, checked["repo"])


def test_the_step_2_look_accepts_a_keep_record(checked):
    al = _load("acceptance_look")
    _run(checked, _regrade_csv(checked))
    _git(checked["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(checked["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "y")
    got = al.check_intra_rater(str(checked["ev"] / "intra-rater-step-1.json"), 2, checked["repo"])
    assert got["file"] == "docs/evidence/calibration/intra-rater-step-1.json"
