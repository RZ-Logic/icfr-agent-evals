"""The calibration subset's blind grading sheet (scripts/calibration_build.py) and its
import (scripts/calibration_import.py).

Pre-registration section 4: 40 cases, every score level 0 to 3 at least 4 times per
LLM-graded sub-dimension, seeded degraded responses where natural outputs do not reach a
level, seeds made two ways (omission and addition). Section 7: the grader gets one
acceptance look on the subset. Decision D-R10 C5: an output that did not parse is not
graded.

Keyless and offline. The worker logs are mock logs (namespaces carrying the fields the
build reads, with real GenerateConfig objects so roster.check_run_config runs as on a
real log), plus one Inspect mockllm run that shows a real log passes the same checks. The
40 cases are a made-up stand-in for data/calibration/calibration.jsonl: no real
calibration case, selection, seed or grade is read.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from inspect_ai.model import GenerateConfig

from casefactory import good_output, item, make_truth
from icfr_agent_evals import casefile, roster, rubric, scoring
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256

REPO = Path(__file__).resolve().parents[1]
GROUNDING = "faithfulness.narrative_grounding"
COVERAGE = "completeness.narrative_coverage"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("calibration_build")
I = _load("calibration_import")
JB = _load("judge_dev_build")
JI = _load("judge_dev_import")
W = B.CAL_WORKERS


# --- Made-up stand-in cases -------------------------------------------------------------

NOTES = (
    "WD-2512-04417 looks like a bridge leg, desk to confirm",
    "Controller: it is the usual timing from the sweep batch, sign it today",
)
CASE_TEXT = ("open items schedule\n  WD-2512-04417 | pending transfer | BTC | 2025-12-31 | chain only | out | open | "
             "waiting on ops\n")
NARR = ("On-chain reserves fall short of ledger liabilities by $120,000.00, a shortfall in the M2 (reportable) band. "
        "WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a possible pending transfer, "
        "leaving $40,000.00 unexplained. "
        "I excluded JE-25-12-0193 (cleared) and DP-2601-05521 (dated after the cutoff).")
TAILS = ("", " Nothing else bears on the gap.", " No other line bears on the gap.")
COPY_EDIT = [{"find": "DP-2601-05521 (dated after the cutoff)", "replace": "DP-2601-05521 (cleared)"}]


def _items() -> tuple:
    cutoff = casefile.period_cutoff_date("2025-12")
    return (
        item("WD-2512-04417", 80_000.00, date=cutoff),
        item("JE-25-12-0193", 40_000.00, date=cutoff, type="manual_adjustment", open_status="cleared",
             recorded_in="both", role="decoy"),
        item("DP-2601-05521", 15_000.00, date="2026-01-02", role="decoy", flow="inflow"),
    )


def _case_id(i: int) -> str:
    return f"calibration-{'std' if i < 20 else 'hc'}-{i:03d}-BTC-M2"


def _cases(n: int = 40) -> dict[str, dict]:
    out = {}
    for i in range(n):
        cid = _case_id(i)
        truth = make_truth("BTC", items=_items(), split="calibration", case_id=cid)
        meta = {"truth": truth.to_dict(), "split": "calibration", "schema_version": "1",
                "fixture": {"notes": [{"ts": "2025-12-31T09:00", "by": "T. Example", "text": t} for t in NOTES]}}
        out[cid] = {"id": cid, "input": f"{CASE_TEXT}batch {i}\n", "metadata": meta}
    return out


def _write_cases(path: Path, cases: dict[str, dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases.values()), encoding="utf-8")
    return path


def _raw(case: dict, worker_n: int) -> str:
    truth = casefile.truth_of(case["metadata"])
    return json.dumps(good_output(truth, narrative=NARR + TAILS[worker_n]), separators=(",", ":"))


# --- Mock logs ------------------------------------------------------------------------------


def _mock_log(model: str, cases: dict[str, dict], *, status="success", split="calibration", prompt=None,
              effort="high", drop=0, epoch=1, raw_of=None, eval_id=None, extra_args=None):
    n = W.index(model) if model in W else 0
    samples = []
    for cid, case in list(cases.items())[drop:]:
        raw = raw_of(cid, case) if raw_of else _raw(case, n)
        samples.append(NS(
            id=cid, epoch=epoch, input=case["input"], metadata=dict(case["metadata"]),
            output=NS(completion=raw), scores={"correctness_anchor": NS(value=3)}, error=None,
            events=[NS(event="model", role=None, config=GenerateConfig(reasoning_effort=effort))]))
    args = {"split": split, "epochs": 1, **(extra_args or {})}
    return NS(
        status=status,
        eval=NS(model=model, eval_id=eval_id or f"EVALID{n}XYZ", task_args=args,
                metadata={"worker_prompt_sha256": prompt or WORKER_PROMPT_SHA256, "split": split},
                config=NS(limit=None, sample_id=None, epochs=1)),
        plan=NS(config=GenerateConfig(reasoning_effort=effort)),
        samples=samples)


def _logs(cases, **over):
    return {m: _mock_log(m, cases, **over.get(m, {})) for m in W}


# --- Selection -------------------------------------------------------------------------------


def _pick_rows() -> list[dict]:
    """36 natural picks whose targets cover every level 9 times, plus two omission seeds
    and two addition seeds (rule G-REASON-COPY)."""
    rows = []
    for i in range(36):
        rows.append({"sample_id": _case_id(i), "worker": W[i % 3], "grounding": i % 4, "coverage": (i // 4) % 4,
                     "basis": f"basis {i}"})
    rows.append({"sample_id": _case_id(36), "worker": W[0], "grounding": 3, "coverage": 0, "rule": "O-BAND+O-REM"})
    rows.append({"sample_id": _case_id(37), "worker": W[1], "grounding": 3, "coverage": 1, "rule": "O-REM"})
    rows.append({"sample_id": _case_id(38), "worker": W[2], "grounding": 1, "coverage": 3, "rule": "G-REASON-COPY",
                 "edits": COPY_EDIT})
    rows.append({"sample_id": _case_id(39), "worker": W[0], "grounding": 1, "coverage": 3, "rule": "G-REASON-COPY",
                 "edits": COPY_EDIT})
    return rows


def _write_selection(path: Path, rows=None, seed: int = 4242) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shuffle_seed": seed, "picks": _pick_rows() if rows is None else rows}),
                    encoding="utf-8")
    return path


def _selection(rows=None):
    picks = []
    for r in (_pick_rows() if rows is None else rows):
        edits = tuple((e["find"], e["replace"]) for e in r.get("edits") or ())
        picks.append(B.CalPick(r["worker"], JB.Pick(r["sample_id"], 1, r["grounding"], r["coverage"], r.get("rule"),
                                                    r.get("basis") or "", edits, r.get("note_words"))))
    return B.CalSelection(tuple(picks), 4242, "0" * 64)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Cases, selection and three mock logs on disk-free paths; read_log returns the mocks."""
    cases = _cases()
    cases_path = _write_cases(tmp_path / "cases" / "calibration.jsonl", cases)
    sel_path = _write_selection(tmp_path / "sealed" / B.SELECTION_NAME)
    logs = _logs(cases)
    paths = []
    for n, m in enumerate(W):
        p = tmp_path / "logs" / f"worker{n}.eval"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"log {n}".encode())
        paths.append(p)
    by_path = {str(p): logs[m] for p, m in zip(paths, W)}
    monkeypatch.setattr(B, "read_log", lambda p: by_path[str(p)])
    return NS(cases=cases, cases_path=cases_path, sel_path=sel_path, logs=logs, paths=paths, by_path=by_path,
              out=tmp_path / "out", tmp=tmp_path)


def _build(s, out=None, **kw):
    return B.build(s.paths, out or s.out, s.cases_path, s.sel_path, kw.pop("force", False), **kw)


SID = "ab" * 32      # a stand-in sheet id for sheets rendered outside a build


def _sheet_json(sheet: str) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', sheet, re.S).group(1))


# --- Constants that follow the text and the design -----------------------------------------


def test_the_three_workers_are_the_roster_models_other_than_the_calibrator():
    assert len(W) == 3 and len(set(W)) == 3
    assert set(W) == set(roster.EVALUATED_MODELS) - {roster.CALIBRATOR}
    assert roster.WORKER in W and roster.CALIBRATOR not in W
    names = {m.split("/", 1)[1] for m in W}
    assert names == {n for n in names if n.endswith(("sonnet-5-5", "sonnet-5", "haiku-5-5"))}


def test_subset_size_minimum_and_split_follow_section_4():
    assert B.SET_SIZE == I.SET_SIZE == 40
    assert B.MIN_PER_LEVEL == I.MIN_PER_LEVEL == 4
    assert B.TARGET_MIN_PER_LEVEL == I.TARGET_MIN_PER_LEVEL == 6
    assert B.CAL_SPLIT in casefile.SPLITS
    assert casefile.split_seed(B.CAL_SPLIT) == casefile.split_seed("eval") + 3_000_000
    assert B.LOCKED_ANCHOR_SHA256 == rubric.llm_anchor_sha256()


def test_the_build_reuses_the_judge_development_seed_rules_and_sheet():
    text = (REPO / "scripts" / "calibration_build.py").read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Z_])RULES: dict", text) and "_JS = " not in text and "_CSS = " not in text
    assert "jdb.apply_rule" in text and "jdb.replace_narrative" in text and "jdb.code_scores" in text
    # The calibration rule table extends the judge-development one and changes none of it.
    assert {k: v for k, v in B.CAL_RULES.items() if k in JB.RULES} == JB.RULES
    assert set(B.CAL_RULES) - set(JB.RULES) == set(B.CAL_ONLY_RULES) == {"G-GUESS"}


def test_public_text_has_no_dashes_or_working_words():
    for path in (REPO / "scripts" / "calibration_build.py", REPO / "scripts" / "calibration_import.py",
                 REPO / "data" / "calibration" / "README.md", Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, path.name
        for word in ("the " + "owner", "work" + "tree", "scratch" + "pad", "fix " + "round"):
            assert word not in text.lower(), (path.name, word)


# --- Selection input ----------------------------------------------------------------------


def test_load_selection_reads_the_picks(tmp_path):
    sel = B.load_selection(_write_selection(tmp_path / "s.json"))
    assert len(sel.picks) == 40 and sel.shuffle_seed == 4242
    assert sel.sha256 == hashlib.sha256((tmp_path / "s.json").read_bytes()).hexdigest()
    seeded = [p for p in sel.picks if p.pick.rule]
    assert [p.source for p in seeded] == ["seeded_omission", "seeded_omission", "seeded_addition", "seeded_addition"]
    assert seeded[2].pick.edits == (("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (cleared)"),)
    assert all(p.pick.epoch == 1 for p in sel.picks)


@pytest.mark.parametrize("doc, message", [
    ({"picks": []}, "shuffle_seed"),
    ({"shuffle_seed": True, "picks": []}, "shuffle_seed"),
    ({"shuffle_seed": 1}, "list of picks"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "calibration-std-000-BTC-M2", "grounding": 1, "coverage": 1}]},
     "malformed"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "x", "worker": "w", "grounding": 1, "coverage": 1, "stratum": 1}]},
     "malformed"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "x", "worker": "w", "grounding": "1", "coverage": 1}]},
     "wrong type"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "x", "worker": "w", "grounding": 1, "coverage": 1, "epoch": 2}]},
     "epoch 2"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "x", "worker": "w", "grounding": 1, "coverage": 1, "rule": 7}]},
     "rule"),
    ({"shuffle_seed": 1, "picks": [{"sample_id": "x", "worker": "w", "grounding": 1, "coverage": 1,
                                    "edits": [{"find": "", "replace": "y"}]}]}, "malformed edits"),
])
def test_load_selection_refuses_a_malformed_input(tmp_path, doc, message):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(B.Refused, match=message):
        B.load_selection(path)


def test_load_selection_refuses_a_missing_or_unreadable_file(tmp_path):
    with pytest.raises(B.Refused, match="not found"):
        B.load_selection(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(B.Refused, match="cannot read"):
        B.load_selection(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(B.Refused, match="not a JSON object"):
        B.load_selection(tmp_path / "list.json")


# --- Selection rules ----------------------------------------------------------------------


def test_selection_problems_accept_the_stand_in_selection():
    assert B.selection_problems(_selection().picks, list(_cases())) == []


def _break(fn):
    rows = _pick_rows()
    fn(rows)
    return rows


@pytest.mark.parametrize("fn, message", [
    (lambda rows: rows.pop(), "holds 39 records, not 40"),
    (lambda rows: rows.__setitem__(1, dict(rows[1], sample_id=rows[0]["sample_id"])), "one output per case"),
    (lambda rows: rows.__setitem__(1, dict(rows[1], sample_id=rows[0]["sample_id"])), "with no pick"),
    (lambda rows: rows.__setitem__(2, dict(rows[2], sample_id="eval-std-000-BTC-M2")), "not a calibration-split case"),
    (lambda rows: rows.__setitem__(2, dict(rows[2], sample_id="calibration-std-099-BTC-M2")), "not in the case file"),
    (lambda rows: rows.__setitem__(3, dict(rows[3], worker=roster.CALIBRATOR)), "not one of the three"),
    (lambda rows: rows.__setitem__(4, dict(rows[4], grounding=4)), "out of range"),
    (lambda rows: rows.__setitem__(36, dict(rows[36], coverage=1)), "makes completeness.narrative_coverage level 0"),
    (lambda rows: rows.__setitem__(5, dict(rows[5], edits=COPY_EDIT)), "natural record carries edits"),
    (lambda rows: rows.__setitem__(36, dict(rows[36], rule="O-NEW")), "unknown rule"),
    (lambda rows: rows.__setitem__(37, dict(rows[37], edits=COPY_EDIT)), "omission rule takes no edits"),
    (lambda rows: rows.__setitem__(38, dict(rows[38], edits=COPY_EDIT * 2)), "takes 1 to 1 edits"),
    (lambda rows: rows.__setitem__(38, dict(rows[38], note_words="desk to confirm")), "note words"),
])
def test_selection_problems_catch_each_break(fn, message):
    rows = _break(fn)
    problems = B.selection_problems(_selection(rows).picks, list(_cases()))
    assert any(message in p for p in problems), problems


def _rows_with_grounding_2(n: int) -> list[dict]:
    """The stand-in selection with only n records targeted at grounding level 2."""
    rows = _pick_rows()
    for r in rows[:36]:
        if r["grounding"] == 2:
            r["grounding"] = 3
    for r in rows[:36][3:3 + 4 * n:4]:       # natural records at grounding level 3
        r["grounding"] = 2
    assert sum(r["grounding"] == 2 for r in rows) == n
    return rows


def test_target_shortfalls_say_which_level_is_short_against_the_margin_of_6():
    rows = _rows_with_grounding_2(1)
    assert B.target_shortfalls(_selection(rows).picks) == [
        "faithfulness.narrative_grounding level 2 has 1 records, 5 short of 6"]
    # Five records meet the floor of 4 but not the margin of 6.
    assert B.target_shortfalls(_selection(_rows_with_grounding_2(5)).picks) == [
        "faithfulness.narrative_grounding level 2 has 5 records, 1 short of 6"]
    assert B.target_shortfalls(_selection(_rows_with_grounding_2(6)).picks) == []
    assert B.target_shortfalls(_selection().picks) == []


def test_selection_problems_leave_the_target_detail_out():
    problems = B.selection_problems(_selection(_rows_with_grounding_2(1)).picks, list(_cases()))
    assert problems == []


@pytest.mark.parametrize("n", [1, 5])
def test_build_records_refuses_targets_below_the_margin_without_naming_a_level(n):
    cases = _cases()
    with pytest.raises(B.ShortRefused) as e:
        B.build_records(_logs(cases), cases, _selection(_rows_with_grounding_2(n)))
    message = str(e.value)
    assert "margin of 6" in message
    assert not re.search(r"level \d|short of|\d+ records,", message)
    assert e.value.detail == [f"faithfulness.narrative_grounding level 2 has {n} records, {6 - n} short of 6"]


def test_selection_problems_refuse_seeds_made_one_way_only():
    rows = _pick_rows()
    rows[38] = {"sample_id": _case_id(38), "worker": W[2], "grounding": 1, "coverage": 3}
    rows[39] = {"sample_id": _case_id(39), "worker": W[0], "grounding": 1, "coverage": 3}
    problems = B.selection_problems(_selection(rows).picks, list(_cases()))
    assert any("one way only" in p for p in problems)
    natural = [{k: v for k, v in r.items() if k not in ("rule", "edits")} for r in _pick_rows()]
    assert not any("one way" in p for p in B.selection_problems(_selection(natural).picks, list(_cases())))


def test_stratification_shortfalls_and_seed_types():
    counts = {GROUNDING: {0: 4, 1: 5, 2: 3, 3: 0}, COVERAGE: {0: 4, 1: 4, 2: 4, 3: 4}}
    assert B.stratification_shortfalls(counts) == [
        f"{GROUNDING} level 2 has 3 records, 1 short of 4", f"{GROUNDING} level 3 has 0 records, 4 short of 4"]
    assert B.stratification_shortfalls(counts, 3) == [f"{GROUNDING} level 3 has 0 records, 3 short of 3"]
    assert B.seed_types(_selection().picks) == {"natural": 36, "seeded_omission": 2, "seeded_addition": 2}
    assert B.level_counts(_selection().picks)[GROUNDING] == {0: 9, 1: 11, 2: 9, 3: 11}


# --- Case file ------------------------------------------------------------------------------


def test_load_cases_reads_the_forty_cases(tmp_path):
    cases = B.load_cases(_write_cases(tmp_path / "c.jsonl", _cases()))
    assert list(cases) == [_case_id(i) for i in range(40)]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: dict(list(c.items())[:39]), "holds 39 cases"),
    (lambda c: {**{k: v for k, v in list(c.items())[1:]}, "dev-std-000-BTC-M2": dict(c[_case_id(0)], id="dev-std-000-BTC-M2")},
     "not a calibration-split case"),
    (lambda c: {**c, _case_id(0): dict(c[_case_id(0)], metadata=dict(c[_case_id(0)]["metadata"], split="dev"))},
     "not a calibration-split case"),
    (lambda c: {**c, _case_id(0): dict(c[_case_id(0)], input=None)}, "no case file text"),
])
def test_load_cases_refuses_any_other_file(tmp_path, mutate, message):
    with pytest.raises(B.Refused, match=message):
        B.load_cases(_write_cases(tmp_path / "c.jsonl", mutate(_cases())))


def test_load_cases_refuses_a_missing_file_or_an_id_twice(tmp_path):
    with pytest.raises(B.Refused, match="not found"):
        B.load_cases(tmp_path / "none.jsonl")
    path = _write_cases(tmp_path / "c.jsonl", _cases())
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(_cases()[_case_id(0)]) + "\n", encoding="utf-8")
    with pytest.raises(B.Refused, match="twice"):
        B.load_cases(path)


# --- Worker logs ----------------------------------------------------------------------------


def test_a_good_mock_log_passes_the_worker_log_check():
    cases = _cases()
    summary = B.check_worker_log(_mock_log(W[0], cases), W[0], list(cases))
    assert summary["epochs"] == 1 and summary["scored_sample_epochs"] == 40 and summary["reasoning_effort"] == "high"


@pytest.mark.parametrize("over, message", [
    ({"status": "error"}, "not 'success'"),
    ({"split": "dev"}, "not a calibration-split run"),
    ({"extra_args": {"allow_evaluation_split": True}}, "not a calibration-split run"),
    ({"prompt": "0" * 64}, "locked prompt"),
    ({"effort": "low"}, "reasoning_effort"),
    ({"drop": 1}, "39 scored sample-epochs"),
])
def test_the_worker_log_check_refuses(over, message):
    cases = _cases()
    with pytest.raises(B.Refused, match=message):
        B.check_worker_log(_mock_log(W[1], cases, **over), W[1], list(cases))


def test_the_worker_log_check_refuses_another_model_or_another_epoch():
    cases = _cases()
    with pytest.raises(B.Refused, match="ran"):
        B.check_worker_log(_mock_log(W[1], cases), W[2], list(cases))
    log = _mock_log(W[1], cases, epoch=2)
    with pytest.raises(B.Refused, match="at epoch 1"):
        B.check_worker_log(log, W[1], list(cases))
    other = dict(cases)
    other.pop(_case_id(0))
    other["calibration-std-077-BTC-M2"] = cases[_case_id(0)]
    with pytest.raises(B.Refused, match="each of the 40"):
        B.check_worker_log(_mock_log(W[1], other), W[1], list(cases))


def test_index_logs_wants_one_log_per_worker():
    cases = _cases(2)
    logs = [_mock_log(m, cases) for m in W]
    assert list(B.index_logs(logs)) == list(W)
    with pytest.raises(B.Refused, match="two logs"):
        B.index_logs([logs[0], logs[0], logs[1]])
    with pytest.raises(B.Refused, match="no log for"):
        B.index_logs(logs[:2])
    with pytest.raises(B.Refused, match="not one of the three"):
        B.index_logs([*logs[:2], _mock_log(roster.CALIBRATOR, cases)])


def test_unparsed_outputs_lists_every_output_that_does_not_parse():
    cases = _cases(3)
    bad = _case_id(1)
    logs = {m: _mock_log(m, cases, raw_of=(lambda cid, c: "not json" if cid == bad else _raw(c, 0))
                         if m == W[2] else None) for m in W}
    assert B.unparsed_outputs(logs) == [{"sample_id": bad, "worker_model": W[2]}]


def test_read_log_refuses_a_missing_or_unreadable_file(tmp_path):
    with pytest.raises(B.Refused, match="not found"):
        B.read_log(tmp_path / "none.eval")
    (tmp_path / "junk.eval").write_bytes(b"junk")
    with pytest.raises(B.Refused, match="cannot read"):
        B.read_log(tmp_path / "junk.eval")


# --- Building records -----------------------------------------------------------------------


def test_build_records_keeps_natural_outputs_and_applies_each_seed():
    cases = _cases()
    logs = _logs(cases)
    recs = B.build_records(logs, cases, _selection())
    assert len(recs) == 40 and sorted(r["position"] for r in recs) == list(range(1, 41))
    assert len({r["code"] for r in recs}) == 40 and all(re.fullmatch(r"R-[A-Z0-9]{4}", r["code"]) for r in recs)
    by_case = {r["sample_id"]: r for r in recs}
    for i, row in enumerate(_pick_rows()):
        r = by_case[row["sample_id"]]
        assert r["worker_model"] == row["worker"] and r["epoch"] == 1 and r["file_name"] == f"{row['sample_id']}.json"
        assert r["log_eval_id"] == str(logs[row["worker"]].eval.eval_id)
        assert r["base_worker_output"] == _raw(cases[row["sample_id"]], W.index(row["worker"]))
        a = scoring.parse_worker_output(r["worker_output"]).parsed
        b = scoring.parse_worker_output(r["base_worker_output"]).parsed
        assert {k: v for k, v in a.items() if k != "narrative"} == {k: v for k, v in b.items() if k != "narrative"}
        if row.get("rule") is None:
            assert r["worker_output"] == r["base_worker_output"] and r["seed_note"] is None and r["source"] == "natural"
        else:
            assert r["worker_output"] != r["base_worker_output"] and r["seed_note"].startswith(f"Rule {row['rule']}.")
    assert by_case[_case_id(36)]["seed_deleted"] == [", a shortfall in the M2 (reportable) band",
                                                     ", leaving $40,000.00 unexplained"]
    copy = by_case[_case_id(38)]
    assert copy["seed_edits"] == COPY_EDIT and "DP-2601-05521 (cleared)" in copy["_narrative"]


def test_build_records_refuses_a_picked_output_that_does_not_parse():
    cases = _cases()
    picked = _case_id(3)          # picked from W[0]
    logs = _logs(cases)
    logs[W[0]] = _mock_log(W[0], cases, raw_of=lambda cid, c: "not json" if cid == picked else _raw(c, 0))
    with pytest.raises(B.Refused, match="D-R10 C5"):
        B.build_records(logs, cases, _selection())
    # The same output unpicked: another worker's output of that case is used instead.
    rows = _pick_rows()
    rows[3]["worker"] = W[1]
    assert len(B.build_records(logs, cases, _selection(rows))) == 40


def test_build_records_refuses_a_sample_error_or_a_case_file_that_differs():
    cases = _cases()
    logs = _logs(cases)
    logs[W[0]].samples[0].error = NS(message="boom")
    with pytest.raises(B.Refused, match="error"):
        B.build_records(logs, cases, _selection())
    logs = _logs(cases)
    logs[W[1]].samples[1].input = "another case file"
    with pytest.raises(B.Refused, match="case file differs"):
        B.build_records(logs, cases, _selection())
    logs = _logs(cases)
    logs[W[1]].samples[1].metadata = {"truth": {}}
    with pytest.raises(B.Refused, match="answer key differs"):
        B.build_records(logs, cases, _selection())


def test_build_records_refuses_an_edit_on_a_protected_token_or_one_that_does_not_apply():
    cases = _cases()
    rows = _pick_rows()
    rows[38]["edits"] = [{"find": "DP-2601-05521 (dated after the cutoff)",
                          "replace": "DP-2601-05521 (dated after the cutoff on 2026-01-03)"}]
    with pytest.raises(B.Refused, match="protected token"):
        B.build_records(_logs(cases), cases, _selection(rows))
    rows = _pick_rows()
    rows[38]["edits"] = [{"find": "text the narrative never holds", "replace": "other text"}]
    with pytest.raises(B.Refused, match="not found exactly once"):
        B.build_records(_logs(cases), cases, _selection(rows))


def test_build_records_refuses_an_edit_that_changes_a_code_score(monkeypatch):
    cases = _cases()
    calls = {"n": 0}

    def fake(parsed, truth):
        calls["n"] += 1
        return {"faithfulness": (calls["n"],)}

    monkeypatch.setattr(B.jdb, "code_scores", fake)
    with pytest.raises(B.Refused, match="changes a code score"):
        B.build_records(_logs(cases), cases, _selection())


def test_build_records_refuses_a_bad_selection():
    cases = _cases()
    with pytest.raises(B.Refused, match="selection: the subset holds 39"):
        B.build_records(_logs(cases), cases, _selection(_pick_rows()[:39]))


# --- G-GUESS, the calibration-only addition seed ---------------------------------------------

GUESS_SENTENCE = "The remaining $40,000.00 is unexplained."
GUESS_NARR = ("On-chain reserves fall short of ledger liabilities by $120,000.00, a shortfall in the M2 (reportable) "
              "band. WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a possible pending "
              f"transfer. {GUESS_SENTENCE} I excluded JE-25-12-0193 (cleared) and DP-2601-05521 (dated after the "
              "cutoff).")
GUESS_CASE = _case_id(39)       # picked from W[0]
GUESS_CLAUSE = ", which may reflect a network fee"


def _guess_rows() -> list[dict]:
    """The stand-in selection with its last addition seed made a G-GUESS seed."""
    rows = _pick_rows()
    rows[39] = {"sample_id": GUESS_CASE, "worker": W[0], "grounding": 2, "coverage": 3, "rule": "G-GUESS"}
    return rows


def _guess_raw(cid: str, case: dict, n: int = 0) -> str:
    if cid != GUESS_CASE:
        return _raw(case, n)
    return json.dumps(good_output(casefile.truth_of(case["metadata"]), narrative=GUESS_NARR), separators=(",", ":"))


def _guess_logs(cases):
    logs = _logs(cases)
    logs[W[0]] = _mock_log(W[0], cases, raw_of=_guess_raw)
    return logs


def _meta(first) -> dict:
    """Stand-in metadata whose explaining item is `first` (None for no explaining item)."""
    items = ((first,) if first else ()) + _items()[1:]
    truth = make_truth("BTC", items=items, split="calibration", case_id=_case_id(0))
    return {"truth": truth.to_dict(), "split": "calibration", "schema_version": "1"}


def test_g_guess_is_one_more_addition_seed_targeting_grounding_2():
    assert B.CAL_RULE_SOURCE["G-GUESS"] == "seeded_addition"
    assert B.CAL_RULE_TARGET["G-GUESS"] == {GROUNDING: 2}
    assert "G-GUESS" not in JB.RULES and "G-GUESS" not in JB.EDIT_COUNT
    text = B.CAL_RULES["G-GUESS"]
    assert text.startswith("Grounding, one claim made overstated: the narrative's own hedged guess")
    assert text.endswith("Target grounding 2; coverage unchanged from the base.")
    for cause, noun in B.GUESS_CAUSES:
        assert f"'{cause}'" in text and noun in text
    assert "', which may reflect X'" in text


def test_g_guess_applies_on_a_qualifying_narrative():
    meta = _cases()[GUESS_CASE]["metadata"]
    assert casefile.truth_of(meta).evidence_state == "partly_unexplained"
    res = B.apply_guess(GUESS_NARR, meta, CASE_TEXT)
    new_sentence = GUESS_SENTENCE[:-1] + GUESS_CLAUSE + "."
    assert res.narrative == GUESS_NARR.replace(GUESS_SENTENCE, new_sentence)
    assert res.edits == [{"find": GUESS_SENTENCE, "replace": new_sentence}] and res.deleted == []
    assert res.note.startswith("Rule G-GUESS. " + B.CAL_RULES["G-GUESS"])
    assert json.dumps(GUESS_CLAUSE) in res.note and json.dumps(GUESS_SENTENCE) in res.note
    assert "Target grounding 2; coverage unchanged from the base." in res.note
    # The pick goes through the calibration dispatcher, never through the judge-development rule table.
    pick = JB.Pick(GUESS_CASE, 1, 2, 3, "G-GUESS")
    assert B.apply_cal_rule(pick, GUESS_NARR, meta, CASE_TEXT).narrative == res.narrative
    # Only the first remainder sentence that names no item takes the clause.
    two = GUESS_NARR + " Nothing explains the remaining $40,000.00."
    assert B.apply_guess(two, meta, CASE_TEXT).narrative.count(GUESS_CLAUSE) == 1


@pytest.mark.parametrize("extra, cause", [
    ("", "a network fee"),
    ("custody FEES line\n", "an exchange rate difference"),
    ("custody fee line\nExchange Rate note\n", "a custodian rebate"),
])
def test_g_guess_takes_the_first_cause_whose_key_noun_the_case_file_lacks(extra, cause):
    meta = _cases()[GUESS_CASE]["metadata"]
    res = B.apply_guess(GUESS_NARR, meta, CASE_TEXT + extra)
    assert f"is unexplained, which may reflect {cause}." in res.narrative


def test_g_guess_refuses_when_every_key_noun_is_in_the_case_file():
    meta = _cases()[GUESS_CASE]["metadata"]
    with pytest.raises(B.SeedError, match="every key noun"):
        B.apply_guess(GUESS_NARR, meta, CASE_TEXT + "a fee, an exchange rate and a REBATE\n")


@pytest.mark.parametrize("narrative", [
    NARR,                                       # its remainder sentence names an item
    NARR.replace(", leaving $40,000.00 unexplained", ""),   # no remainder sentence at all
])
def test_g_guess_refuses_a_narrative_with_no_remainder_sentence_naming_no_item(narrative):
    meta = _cases()[GUESS_CASE]["metadata"]
    with pytest.raises(B.SeedError, match="no sentence states the remainder"):
        B.apply_guess(narrative, meta, CASE_TEXT)


def test_g_guess_refuses_a_remainder_sentence_without_a_final_full_stop():
    meta = _cases()[GUESS_CASE]["metadata"]
    with pytest.raises(B.SeedError, match="does not end with a full stop"):
        B.apply_guess(GUESS_NARR.replace(GUESS_SENTENCE, "The remaining $40,000.00 is unexplained!"), meta, CASE_TEXT)


@pytest.mark.parametrize("amount, state", [(120_000.00, "fully_explained"), (None, "unexplained")])
def test_g_guess_refuses_a_fully_explained_or_unexplained_case(amount, state):
    first = None if amount is None else item("WD-2512-04417", amount, date=casefile.period_cutoff_date("2025-12"))
    meta = _meta(first)
    assert casefile.truth_of(meta).evidence_state == state
    with pytest.raises(B.SeedError, match=f"partly_unexplained.*not {state}"):
        B.apply_guess(GUESS_NARR, meta, CASE_TEXT)


def test_g_guess_refuses_a_clause_on_a_protected_token(monkeypatch):
    meta = _cases()[GUESS_CASE]["metadata"]
    monkeypatch.setattr(B, "GUESS_CAUSES", (("a BitGo Trust rebate", "rebate"),))
    with pytest.raises(B.SeedError, match="protected token"):
        B.apply_guess(GUESS_NARR, meta, CASE_TEXT)


def test_selection_accepts_g_guess_at_grounding_2_without_edits():
    assert B.selection_problems(_selection(_guess_rows()).picks, list(_cases())) == []
    assert B.seed_types(_selection(_guess_rows()).picks) == {"natural": 36, "seeded_omission": 2,
                                                             "seeded_addition": 2}


@pytest.mark.parametrize("change, message", [
    ({"grounding": 1}, "rule G-GUESS makes faithfulness.narrative_grounding level 2, not 1"),
    ({"edits": COPY_EDIT}, "rule G-GUESS takes no edits"),
    ({"note_words": "desk to confirm"}, "note words go with G-ADOPT only"),
])
def test_selection_refuses_a_g_guess_pick_off_its_rule(change, message):
    rows = _guess_rows()
    rows[39] = dict(rows[39], **change)
    problems = B.selection_problems(_selection(rows).picks, list(_cases()))
    assert any(message in p for p in problems), problems


def test_build_records_applies_g_guess_and_checks_the_code_scores():
    cases = _cases()
    by_case = {r["sample_id"]: r for r in B.build_records(_guess_logs(cases), cases, _selection(_guess_rows()))}
    r = by_case[GUESS_CASE]
    assert r["source"] == "seeded_addition" and r["seed_rule"] == "G-GUESS" and r["seed_deleted"] == []
    assert r["seed_note"].startswith("Rule G-GUESS.") and json.dumps(GUESS_CLAUSE) in r["seed_note"]
    assert r["_narrative"] == GUESS_NARR.replace(GUESS_SENTENCE, GUESS_SENTENCE[:-1] + GUESS_CLAUSE + ".")
    a = scoring.parse_worker_output(r["worker_output"]).parsed
    b = scoring.parse_worker_output(r["base_worker_output"]).parsed
    assert {k: v for k, v in a.items() if k != "narrative"} == {k: v for k, v in b.items() if k != "narrative"}
    truth = casefile.truth_of(cases[GUESS_CASE]["metadata"])
    assert JB.code_scores(a, truth) == JB.code_scores(b, truth)


def test_build_records_refuses_a_g_guess_seed_that_changes_a_code_score(monkeypatch):
    cases = _cases()
    real = JB.code_scores

    def fake(parsed, truth):
        # A score that moves only when the narrative holds the inserted clause.
        out = real(parsed, truth)
        return {**out, "faithfulness": ("changed",)} if GUESS_CLAUSE in parsed["narrative"] else out

    monkeypatch.setattr(B.jdb, "code_scores", fake)
    # The G-REASON-COPY seed before it passes the same check; the G-GUESS seed is refused.
    with pytest.raises(B.Refused, match=re.escape(GUESS_CASE) + r" from .*, rule G-GUESS: the edit changes a code "
                                        r"score \(faithfulness\)"):
        B.build_records(_guess_logs(cases), cases, _selection(_guess_rows()))
    monkeypatch.setattr(B.jdb, "code_scores", real)
    assert len(B.build_records(_guess_logs(cases), cases, _selection(_guess_rows()))) == 40


def test_build_records_refuses_g_guess_on_a_case_with_no_unexplained_remainder():
    cases = _cases()
    meta = _meta(item("WD-2512-04417", 120_000.00, date=casefile.period_cutoff_date("2025-12")))
    cases[GUESS_CASE] = dict(cases[GUESS_CASE], metadata=dict(cases[GUESS_CASE]["metadata"], truth=meta["truth"]))
    with pytest.raises(B.Refused, match="rule G-GUESS: G-GUESS needs a case with an unexplained remainder"):
        B.build_records(_guess_logs(cases), cases, _selection(_guess_rows()))


def test_build_keeps_the_g_guess_seed_note_in_the_sealed_plan_only(setup):
    _write_selection(setup.sel_path, _guess_rows())
    setup.by_path[str(setup.paths[0])] = _mock_log(W[0], setup.cases, raw_of=_guess_raw)
    plan = _build(setup)
    sheet = (setup.out / B.SHEET_NAME).read_text(encoding="utf-8")
    rec = next(r for r in plan["records"] if r["sample_id"] == GUESS_CASE)
    assert rec["seed_rule"] == "G-GUESS" and rec["source"] == "seeded_addition"
    assert rec["seed_note"].startswith("Rule G-GUESS. " + B.CAL_RULES["G-GUESS"])
    assert json.dumps(GUESS_CLAUSE) in rec["seed_note"]
    assert plan["rules"]["G-GUESS"] == B.CAL_RULES["G-GUESS"]
    assert plan["source_counts"] == {"natural": 36, "seeded_omission": 2, "seeded_addition": 2}
    # The sheet shows the seeded narrative, never the note, the rule name or the rule text.
    shown = next(x for x in _sheet_json(sheet)["records"] if x["code"] == rec["code"])
    assert shown["narrative"] == scoring.parse_worker_output(rec["worker_output"]).parsed["narrative"]
    assert GUESS_CLAUSE in shown["narrative"]
    assert rec["seed_note"] not in sheet and "G-GUESS" not in sheet.upper()
    assert B.CAL_RULES["G-GUESS"] not in sheet
    assert B.blindness_hits(sheet, plan["records"], [x["eval_id"] for x in plan["logs"]]) == []
    I.load_plan(setup.out / B.SEALED_DIR / B.PLAN_NAME)
    # A sheet that showed the rule name would be refused.
    leaked = sheet.replace("</body>", "<p>G-GUESS</p></body>")
    assert "word 'G-GUESS'" in B.blindness_hits(leaked, plan["records"], [])


# --- Sheet ----------------------------------------------------------------------------------


def test_sheet_shows_what_the_judge_development_sheet_shows():
    cases = _cases()
    recs = B.build_records(_logs(cases), cases, _selection())
    sheet = B.render_sheet(recs, SID)
    data = _sheet_json(sheet)
    base = JB.sheet_data(recs)
    ours = ("storage_key", "grades_file", "sittings_file", "sheet_id")
    assert {k: v for k, v in data.items() if k not in ours} == {k: v for k, v in base.items() if k not in ours}
    assert set(data["records"][0]) == {"code", "narrative", "case_input", "case_facts", "output_facts"}
    assert [r["code"] for r in data["records"]] == [r["code"] for r in sorted(recs, key=lambda r: r["position"])]
    assert data["grades_file"] == B.GRADES_FILE and data["sittings_file"] == B.SITTINGS_FILE
    assert data["sheet_id"] == SID
    assert data["storage_key"] == "calibration-sheet-" + SID[:32] == B.storage_key(SID)
    assert data["intro"] == list(JB.INTRO) and data["anchors_sha256"] == B.LOCKED_ANCHOR_SHA256
    for d in (GROUNDING, COVERAGE):
        assert data["scope"][d] == rubric.LLM_GRADED_SCOPE[d]
    assert f"<title>{B.SHEET_TITLE}</title>" in sheet and JB.SHEET_TITLE not in sheet
    assert not re.search(r"(?:src|href)\s*=\s*[\"']?(?:https?:)?//", sheet)
    assert "http://" not in sheet and "https://" not in sheet and "<link" not in sheet
    assert B.blindness_hits(sheet, recs, ["EVALID0XYZ", "EVALID1XYZ", "EVALID2XYZ"]) == []


def test_worker_words_cover_ids_names_and_family_words():
    words = B.worker_words()
    for m in W:
        assert m in words and m.split("/", 1)[1] in words
    assert {"sonnet", "haiku"} <= set(words)


@pytest.mark.parametrize("leak", [*W, "Sonnet", "HAIKU", "worker_model", "EVALID1XYZ", "calibration-hc-",
                                  "calibration-std-004-BTC-M2", "natural", "seeded_omission", "stratum", "epoch",
                                  "O-BAND", "G-REASON-COPY", "basis 7"])
def test_blindness_check_catches_each_leak(leak):
    cases = _cases()
    recs = B.build_records(_logs(cases), cases, _selection())
    sheet = B.render_sheet(recs, SID).replace("</body>", f"<p>{leak}</p></body>")
    assert B.blindness_hits(sheet, recs, ["EVALID0XYZ", "EVALID1XYZ", "EVALID2XYZ"])


# --- Build ----------------------------------------------------------------------------------


def test_build_writes_a_blind_sheet_and_a_sealed_plan(setup):
    plan = _build(setup)
    sheet_path, plan_path = setup.out / B.SHEET_NAME, setup.out / B.SEALED_DIR / B.PLAN_NAME
    assert sorted(p.relative_to(setup.out).as_posix() for p in setup.out.rglob("*") if p.is_file()) == [
        f"{B.SEALED_DIR}/{B.PLAN_NAME}", B.SHEET_NAME]
    sheet = sheet_path.read_text(encoding="utf-8")
    assert json.loads(plan_path.read_text(encoding="utf-8")) == plan
    assert plan["sheet_sha256"] == hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    assert B.blindness_hits(sheet, plan["records"], [x["eval_id"] for x in plan["logs"]]) == []
    assert [x["model"] for x in plan["logs"]] == list(W)
    assert [x["sha256"] for x in plan["logs"]] == [hashlib.sha256(p.read_bytes()).hexdigest() for p in setup.paths]
    assert plan["cases_sha256"] == hashlib.sha256(setup.cases_path.read_bytes()).hexdigest()
    assert plan["selection_sha256"] == hashlib.sha256(setup.sel_path.read_bytes()).hexdigest()
    assert plan["worker_prompt_sha256"] == WORKER_PROMPT_SHA256 and plan["anchors_sha256"] == B.LOCKED_ANCHOR_SHA256
    assert plan["min_per_level"] == 4 and plan["unparsed"] == [] and plan["rules"] == B.CAL_RULES
    assert plan["target_min_per_level"] == 6
    assert re.fullmatch(r"[0-9a-f]{64}", plan["sheet_id"])
    data = _sheet_json(sheet)
    assert data["sheet_id"] == plan["sheet_id"] and data["storage_key"] == B.storage_key(plan["sheet_id"])
    core = {k: v for k, v in plan.items() if k not in B.SHEET_FIELDS}
    recs = B.build_records(setup.logs, setup.cases, B.load_selection(setup.sel_path))
    assert B.compute_sheet_id(recs, core) == plan["sheet_id"]
    assert plan["source_counts"] == {"natural": 36, "seeded_omission": 2, "seeded_addition": 2}
    assert plan["worker_counts"] == {W[0]: 14, W[1]: 13, W[2]: 13}
    assert len(plan["records"]) == 40 and [r["position"] for r in plan["records"]] == list(range(1, 41))
    assert all(tuple(r) == B.PLAN_FIELDS for r in plan["records"])
    assert not any(k.startswith("_") for r in plan["records"] for k in r)
    data = _sheet_json(sheet)
    for rec, shown in zip(plan["records"], data["records"]):
        assert shown["code"] == rec["code"]
        assert shown["narrative"] == scoring.parse_worker_output(rec["worker_output"]).parsed["narrative"]
    # The plan the build writes passes the import's checks, sheet hash included.
    I.load_plan(plan_path)


def test_build_is_deterministic_and_refuses_to_overwrite(setup):
    plan = _build(setup)
    with pytest.raises(B.Refused, match="already exists"):
        _build(setup)
    again = _build(setup, out=setup.tmp / "again")
    assert again["sheet_sha256"] == plan["sheet_sha256"]
    assert (setup.tmp / "again" / B.SHEET_NAME).read_bytes() == (setup.out / B.SHEET_NAME).read_bytes()
    assert _build(setup, force=True)["sheet_sha256"] == plan["sheet_sha256"]


def test_build_lists_unparsed_outputs_it_did_not_pick(setup):
    unpicked = _case_id(3)        # picked from W[0], so W[2]'s output of it is not used
    setup.by_path[str(setup.paths[2])] = _mock_log(
        W[2], setup.cases, raw_of=lambda cid, c: "not json" if cid == unpicked else _raw(c, 2))
    plan = _build(setup)
    assert plan["unparsed"] == [{"sample_id": unpicked, "worker_model": W[2]}]
    I.load_plan(setup.out / B.SEALED_DIR / B.PLAN_NAME)


def test_build_refuses_sealed_files_inside_the_repo(setup):
    with pytest.raises(B.Refused, match="outside the repo"):
        B.build(setup.paths, REPO / "data" / "calibration" / "built", setup.cases_path, setup.sel_path, False)
    with pytest.raises(B.Refused, match="outside the repo"):
        B.build(setup.paths, setup.out, setup.cases_path, REPO / "selection.json", False)
    assert not (REPO / "data" / "calibration" / "built").exists()


def test_build_refuses_the_wrong_number_of_logs_or_other_anchors(setup, monkeypatch):
    with pytest.raises(B.Refused, match="give 3 logs"):
        B.build(setup.paths[:2], setup.out, setup.cases_path, setup.sel_path, False)
    monkeypatch.setattr(B, "LOCKED_ANCHOR_SHA256", "0" * 64)
    with pytest.raises(B.Refused, match="anchors"):
        _build(setup)


def test_build_refuses_a_sheet_that_would_show_sealed_text(setup):
    # A case file that names its worker would reach the sheet.
    leaked = {k: dict(v, input=v["input"] + "drafted with sonnet\n") for k, v in setup.cases.items()}
    _write_cases(setup.cases_path, leaked)
    for p, m in zip(setup.paths, W):
        setup.by_path[str(p)] = _mock_log(m, leaked)
    with pytest.raises(B.Refused, match="sealed text"):
        _build(setup)
    assert not setup.out.exists()


def test_main_prints_totals_and_exits_by_outcome(setup, capsys):
    argv = [a for p in setup.paths for a in ("--log", str(p))] + [
        "--cases", str(setup.cases_path), "--selection", str(setup.sel_path), "--out", str(setup.out)]
    assert B.main(argv) == 0
    out = capsys.readouterr().out
    assert "records: 40" in out and "unparsed outputs (never graded): 0" in out
    plan = json.loads((setup.out / B.SEALED_DIR / B.PLAN_NAME).read_text(encoding="utf-8"))
    assert f"sheet id: {plan['sheet_id']}" in out
    # Totals only: no level, source or worker count reaches the screen.
    assert "level" not in out and GROUNDING not in out and COVERAGE not in out
    assert "seeded" not in out and "natural" not in out and not any(m in out for m in W)
    assert B.main(argv) == 3 and "already exists" in capsys.readouterr().err
    assert B.main(["--cases", "x"]) == 3


# --- A real Inspect log (mockllm) passes the same checks ------------------------------------


def test_a_real_mockllm_log_passes_the_worker_log_check(tmp_path):
    from inspect_ai import Task, task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.scorer import Score, scorer, mean
    from inspect_ai.solver import generate

    cases = _cases()
    by_text = {c["input"]: _raw(c, 0) for c in cases.values()}

    def answer(messages, tools, tool_choice, config):
        text = "".join(str(getattr(m, "text", "")) for m in messages if getattr(m, "role", "") == "user")
        return ModelOutput.from_content(model="mockllm/model", content=by_text[text])

    @scorer(metrics=[mean()])
    def stub():
        async def score(state, target):
            return Score(value=3)
        return score

    @task
    def cal_stub(split: str = "calibration"):
        samples = [Sample(id=c["id"], input=c["input"], metadata=c["metadata"]) for c in cases.values()]
        return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=stub(),
                    config=GenerateConfig(reasoning_effort="high"),
                    metadata={"worker_prompt_sha256": WORKER_PROMPT_SHA256, "split": split})

    model = get_model("mockllm/model", custom_outputs=answer)
    log = inspect_eval(cal_stub(split="calibration"), model=model, epochs=1, log_dir=str(tmp_path), display="none")[0]
    log.eval.model = W[0]           # the header names the worker, as a real run of that worker does
    path = tmp_path / "worker.eval"
    write_eval_log(log, str(path))
    real = B.read_log(path)
    assert real.eval.task_args.get("split") == "calibration"
    summary = B.check_worker_log(real, W[0], list(cases))
    assert summary["scored_sample_epochs"] == 40
    assert B.unparsed_outputs({m: real for m in W}) == []
    assert read_eval_log(str(path)).status == "success"


# --- Import ---------------------------------------------------------------------------------


@pytest.fixture
def built(setup):
    plan = _build(setup)
    return NS(**vars(setup), plan=plan, plan_path=setup.out / B.SEALED_DIR / B.PLAN_NAME)


def _grades(plan: dict, levels=None) -> list[list[str]]:
    """Author grades for each record: by default every level 10 times per sub-dimension."""
    rows = [["record_code", GROUNDING, COVERAGE, "note", "graded_at", "sheet_id"]]
    for n, r in enumerate(sorted(plan["records"], key=lambda r: r["code"])):
        g, c = levels(n) if levels else (n % 4, (n // 4) % 4)
        rows.append([r["code"], str(g), str(c), "a, note" if n == 0 else "", "2026-11-20T10:15:00-05:00",
                     plan["sheet_id"]])
    return rows


def _grades_file(tmp_path: Path, rows) -> Path:
    path = tmp_path / "calibration-grades.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\r\n").writerows(rows)
    return path


def _plan_copy(built, tmp_path, mutate) -> Path:
    plan = json.loads(built.plan_path.read_text(encoding="utf-8"))
    mutate(plan)
    path = tmp_path / "edited" / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_import_writes_the_readme_format(built, tmp_path):
    grades = _grades_file(tmp_path, _grades(built.plan))
    out = tmp_path / "data" / "calibration" / "records"
    out.parent.mkdir(parents=True)
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(out)]) == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(f"{_case_id(i)}.json" for i in range(40))
    by_case = {r["sample_id"]: r for r in built.plan["records"]}
    want = {row[0]: row for row in _grades(built.plan)[1:]}
    for name in files:
        raw = (out / name).read_bytes()
        assert b"\r\n" not in raw and raw.endswith(b"\n")
        rec = json.loads(raw.decode("utf-8"))
        assert tuple(rec) == I.RECORD_FIELDS == (*JI.RECORD_FIELDS, "worker_model")
        src = by_case[rec["sample_id"]]
        assert rec["worker_model"] == src["worker_model"] and rec["epoch"] == 1
        assert rec["worker_output"] == src["worker_output"] and rec["seed_note"] == src["seed_note"]
        assert rec["source"] == src["source"] and rec["stratum"] == src["stratum"]
        assert rec["grades"] == {GROUNDING: int(want[src["code"]][1]), COVERAGE: int(want[src["code"]][2])}
        assert rec["anchors_sha256"] == rubric.llm_anchor_sha256() and rec["graded_by"] == JI.GRADED_BY
        assert rec["graded_on"] == "2026-11-20" and rec["log_eval_id"] == src["log_eval_id"]
        assert "code" not in rec and "seed_rule" not in rec and "stratum_basis" not in rec


def test_import_refuses_author_grades_that_miss_the_stratification(built, tmp_path, capsys):
    # Every grounding grade is 3 or 2; level 0 and 1 never appear; coverage level 2 only twice.
    def levels(n):
        return (3 if n % 2 else 2, 2 if n < 2 else (0, 1, 3)[n % 3])
    grades = _grades_file(tmp_path, _grades(built.plan, levels))
    out = tmp_path / "records"
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(out)]) == 3
    captured = capsys.readouterr()
    screen = captured.out + captured.err
    assert "do not meet the section 4 stratification" in screen and "more seeds" in screen and not out.exists()
    # The screen never names a level, a count or a shortfall.
    assert not re.search(r"level \d has|short of|\d+ records|\d+ times,", screen)
    # The per-level detail is sealed beside the plan.
    detail = (built.plan_path.parent / I.SHORTFALL_NAME).read_text(encoding="utf-8")
    assert built.plan_path.parent.name == B.SEALED_DIR
    assert f"{GROUNDING} level 0 has 0 records, 4 short of 4" in detail
    assert f"{GROUNDING} level 1 has 0 records, 4 short of 4" in detail
    assert f"{COVERAGE} level 2 has 2 records, 2 short of 4" in detail


def test_grade_shortfalls_count_each_level():
    grades = [{GROUNDING: n % 4, COVERAGE: 3} for n in range(15)]
    assert I.grade_shortfalls(grades) == [
        f"{GROUNDING} level 3 has 3 records, 1 short of 4",
        f"{COVERAGE} level 0 has 0 records, 4 short of 4",
        f"{COVERAGE} level 1 has 0 records, 4 short of 4",
        f"{COVERAGE} level 2 has 0 records, 4 short of 4"]


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows[:-1], "with no grades"),
    (lambda rows: rows + [rows[1]], "appears twice"),
    (lambda rows: rows[:1] + [[rows[1][0], "4"] + rows[1][2:]] + rows[2:], "not 0, 1, 2 or 3"),
    (lambda rows: rows[:1] + [rows[1][:4] + [""]] + rows[2:], "graded_at"),
    (lambda rows: [[c for c in r[:2]] + r[3:] for r in rows], "lacks the column"),
])
def test_import_refuses_bad_grades(built, tmp_path, capsys, mutate, message):
    grades = _grades_file(tmp_path, mutate(_grades(built.plan)))
    out = tmp_path / "records"
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(out)]) == 3
    assert message in capsys.readouterr().err and not out.exists()


def _rec(plan, i, **kw):
    plan["records"][i] = dict(plan["records"][i], **kw)


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.update(records=p["records"][:39]), "40 records"),
    (lambda p: p.update(anchors_sha256="0" * 64), "other anchors"),
    (lambda p: p.update(worker_prompt_sha256="0" * 64), "locked prompt"),
    (lambda p: p.update(logs=p["logs"][:2]), "one log per calibration worker"),
    (lambda p: p.update(unparsed="x"), "unparsed list is malformed"),
    (lambda p: _rec(p, 0, worker_output="not json"), "does not parse"),
    (lambda p: _rec(p, 0, base_worker_output="not json"), "base worker output does not parse"),
    (lambda p: _rec(p, 1, epoch=2), "epoch 2"),
    (lambda p: _rec(p, 2, worker_model=roster.CALIBRATOR), "not one of the three"),
    (lambda p: _rec(p, 3, log_eval_id="OTHER"), "log eval id"),
    (lambda p: _rec(p, 4, sample_id="eval-std-000-BTC-M2", file_name="eval-std-000-BTC-M2.json"),
     "not a calibration-split case"),
    (lambda p: _rec(p, 5, sample_id=p["records"][6]["sample_id"], file_name=p["records"][6]["file_name"]), "twice"),
    (lambda p: _rec(p, 7, file_name="calibration-other.json"), "<case id>.json"),
    (lambda p: _rec(p, 8, code=p["records"][9]["code"]), "repeated record code"),
    (lambda p: _rec(p, 10, source="invented"), "unknown source"),
    (lambda p: _rec(p, 11, stratum={GROUNDING: 1}), "malformed stratum"),
    (lambda p: _rec(p, 12, worker_prompt_sha256="0" * 64), "prompt hash differs"),
    (lambda p: p.update(unparsed=[{"sample_id": p["records"][13]["sample_id"],
                                   "worker_model": p["records"][13]["worker_model"]}]), "D-R10 C5"),
    (lambda p: [r.update(stratum={GROUNDING: 3, COVERAGE: 3}) for r in p["records"]], "targets miss"),
])
def test_import_refuses_a_bad_plan(built, tmp_path, capsys, mutate, message):
    plan_path = _plan_copy(built, tmp_path, mutate)
    grades = _grades_file(tmp_path, _grades(built.plan))
    out = tmp_path / "records"
    assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(out)]) == 3
    assert message in capsys.readouterr().err and not out.exists()


def test_import_refuses_a_change_outside_the_narrative_and_a_seed_note_on_a_natural_record(built, tmp_path, capsys):
    def outside(p):
        r = next(r for r in p["records"] if r["source"] == "natural")
        obj = json.loads(r["worker_output"])
        obj["variance_amount_usd"] += 1
        r["worker_output"] = json.dumps(obj)
    for mutate, message in ((outside, "outside the narrative"),
                            (lambda p: next(r for r in p["records"] if r["source"] == "natural").update(seed_note="x"),
                             "seed note")):
        plan_path = _plan_copy(built, tmp_path, mutate)
        grades = _grades_file(tmp_path, _grades(built.plan))
        assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(tmp_path / "r")]) == 3
        assert message in capsys.readouterr().err


def test_import_checks_the_sheet_beside_the_plan(built, tmp_path, capsys):
    (built.out / B.SHEET_NAME).write_bytes(b"<!doctype html><title>changed</title>")
    grades = _grades_file(tmp_path, _grades(built.plan))
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(tmp_path / "r")]) == 3
    assert "does not match" in capsys.readouterr().err


def test_import_never_overwrites_and_needs_the_parent_folder(built, tmp_path, capsys):
    grades = _grades_file(tmp_path, _grades(built.plan))
    out = tmp_path / "records"
    out.mkdir()
    keep = out / built.plan["records"][5]["file_name"]
    keep.write_text("author file", encoding="utf-8")
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(out)]) == 3
    assert "already exist" in capsys.readouterr().err
    assert [p.name for p in out.iterdir()] == [keep.name] and keep.read_text(encoding="utf-8") == "author file"
    deep = tmp_path / "missing" / "records"
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(deep)]) == 3
    assert "neither is its parent" in capsys.readouterr().err and not deep.parent.exists()


def test_import_refuses_a_missing_plan_or_grades_and_bad_arguments(built, tmp_path, capsys):
    grades = _grades_file(tmp_path, _grades(built.plan))
    assert I.main(["--grades", str(grades), "--plan", str(tmp_path / "none.json"), "--out", str(tmp_path / "r")]) == 3
    assert "plan not found" in capsys.readouterr().err
    assert I.main(["--grades", str(tmp_path / "none.csv"), "--plan", str(built.plan_path),
                   "--out", str(tmp_path / "r")]) == 3
    assert "grades file not found" in capsys.readouterr().err
    assert I.main([]) == 3


def test_load_grades_is_the_judge_development_reader(built, tmp_path):
    grades = _grades_file(tmp_path, _grades(built.plan))
    assert I.load_grades(grades) == JI.load_grades(grades)


# --- Sheet id: old grades never attach to a rebuilt sheet -----------------------------------


def _changed_rows() -> list[dict]:
    """The stand-in selection with one pick moved to another worker: same targets, other output."""
    rows = _pick_rows()
    rows[3]["worker"] = W[1]
    return rows


def test_sheet_id_follows_the_shown_records_and_the_plan():
    cases = _cases()
    recs = B.build_records(_logs(cases), cases, _selection())
    core = {"selection_sha256": "1" * 64, "shuffle_seed": 4242}
    sid = B.compute_sheet_id(recs, core)
    assert re.fullmatch(r"[0-9a-f]{64}", sid) and B.compute_sheet_id(recs, dict(core)) == sid
    assert B.compute_sheet_id(recs, {**core, "selection_sha256": "2" * 64}) != sid
    other = B.build_records(_logs(cases), cases, _selection(_changed_rows()))
    # Same shuffle seed: the same codes, but another output behind one of them, so another id.
    assert sorted(r["code"] for r in other) == sorted(r["code"] for r in recs)
    assert B.compute_sheet_id(other, core) != sid
    edited = [dict(r) for r in recs]
    edited[0]["_narrative"] += " One more sentence."
    assert B.compute_sheet_id(edited, core) != sid


@pytest.mark.parametrize("bad", ["", "abc", "AB" * 32, "g" * 64, None])
def test_sheet_data_refuses_a_malformed_sheet_id(bad):
    cases = _cases()
    recs = B.build_records(_logs(cases), cases, _selection())
    with pytest.raises(B.Refused, match="sheet id"):
        B.sheet_data(recs, bad)


def test_sheet_script_adds_the_sheet_id_column_and_refuses_a_changed_base_script(monkeypatch):
    js = B.sheet_script()
    assert '["note", "graded_at", "sheet_id"]' in js and 'x.at || "", D.sheet_id' in js
    assert 'head.indexOf("sheet_id")' in js and "saved from another sheet" in js
    monkeypatch.setattr(B.jdb, "_JS", B.jdb._JS.replace('"graded_at"])];', '"graded_at", "x"])];'))
    with pytest.raises(B.Refused, match="script changed"):
        B.sheet_script()


_HARNESS = r"""
const DATA = __DATA__;
const SCRIPT = __SCRIPT__;
const els = {}, store = {}, blobs = [], statuses = [];
let keydown = null;
function fake(id){ return {id: id, style: {}, children: [], listeners: {}, files: [], value: "", textContent: "",
  appendChild(c){ this.children.push(c); return c; }, addEventListener(t, f){ this.listeners[t] = f; },
  setAttribute(){}, click(){}, remove(){}, showModal(){}, close(){}, focus(){}, blur(){}}; }
global.document = {
  getElementById(id){ if (id === "data") return {textContent: DATA}; if (!els[id]) els[id] = fake(id); return els[id]; },
  createElement(tag){ return fake(tag); }, createTextNode(t){ return fake("#text"); }, body: fake("body"),
  addEventListener(t, f){ if (t === "keydown") keydown = f; }};
global.window = {localStorage: {getItem(k){ return k in store ? store[k] : null; }, setItem(k, v){ store[k] = v; }},
                 scrollTo(){}};
URL.createObjectURL = function(b){ blobs.push(b); return "blob:x"; };
URL.revokeObjectURL = function(){};
global.setInterval = function(){ return 0; };
global.setTimeout = function(){ return 0; };
global.FileReader = function(){ this.readAsText = function(f){ this.result = f.text; this.onload(); }; };
function key(k){ keydown({key: k, target: {tagName: "BODY"}, preventDefault(){}}); }
function load(text){ els.file.files = [{text: text}]; els.file.listeners.change(); statuses.push(els.status.textContent); }
(async function(){
  eval(SCRIPT);
  key("2"); key("3");
  els.save.listeners.click();
  const saved = await blobs[0].text();
  const D = JSON.parse(DATA);
  load(saved.split(D.sheet_id).join("a".repeat(64)));
  load(saved.replace(",sheet_id", ",sheet"));
  load(saved);
  console.log(JSON.stringify({saved: saved, statuses: statuses, keys: Object.keys(store)}));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_saves_the_sheet_id_and_will_not_load_another_sheets_grades(built, tmp_path):
    sheet = (built.out / B.SHEET_NAME).read_text(encoding="utf-8")
    data = re.search(r'<script type="application/json" id="data">(.*?)</script>', sheet, re.S).group(1)
    script = re.search(r"<script>(.*?)</script>", sheet, re.S).group(1)
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS.replace("__DATA__", json.dumps(data)).replace("__SCRIPT__", json.dumps(script)),
                       encoding="utf-8")
    res = subprocess.run(["node", str(harness)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    rows = list(csv.reader(got["saved"].splitlines()))
    assert rows[0] == ["record_code", GROUNDING, COVERAGE, "note", "graded_at", "sheet_id"]
    assert len(rows) == 41 and all(r[5] == built.plan["sheet_id"] for r in rows[1:])
    first = _sheet_json(sheet)["records"][0]["code"]
    assert [r[1:3] for r in rows[1:] if r[0] == first] == [["2", "3"]]
    assert got["statuses"][0] == "That grades file was saved from another sheet. It is not loaded."
    assert got["statuses"][1] == "That file is not a grades file from this page."
    assert got["statuses"][2] == "Loaded 40 record(s) from the grades file."
    assert got["keys"] == [B.storage_key(built.plan["sheet_id"])]
    # The file the page saves carries the plan's sheet id as the import reads it.
    grades = tmp_path / "saved.csv"
    grades.write_text(got["saved"], encoding="utf-8", newline="")
    assert I.grades_sheet_id(grades) == built.plan["sheet_id"]


def test_import_refuses_grades_saved_from_an_earlier_build_with_the_same_codes(setup, tmp_path, capsys):
    first = _build(setup, out=setup.tmp / "first")
    grades = _grades_file(tmp_path, _grades(first))
    _write_selection(setup.sel_path, _changed_rows())          # same shuffle seed
    second = _build(setup, out=setup.tmp / "second")
    assert sorted(r["code"] for r in second["records"]) == sorted(r["code"] for r in first["records"])
    assert second["sheet_id"] != first["sheet_id"]
    plan2 = setup.tmp / "second" / B.SEALED_DIR / B.PLAN_NAME
    out = tmp_path / "records"
    assert I.main(["--grades", str(grades), "--plan", str(plan2), "--out", str(out)]) == 3
    assert "comes from sheet" in capsys.readouterr().err and not out.exists()
    plan1 = setup.tmp / "first" / B.SEALED_DIR / B.PLAN_NAME
    assert I.main(["--grades", str(grades), "--plan", str(plan1), "--out", str(out)]) == 0


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: [r[:5] for r in rows], "no single sheet_id column"),
    (lambda rows: [rows[0] + ["sheet_id"]] + [r + [r[5]] for r in rows[1:]], "more than once"),
    (lambda rows: rows[:2] + [rows[2][:5] + [""]] + rows[3:], "without a sheet id"),
    (lambda rows: rows[:2] + [rows[2][:5] + ["c" * 64]] + rows[3:], "more than one sheet"),
    (lambda rows: [rows[0]] + [r[:5] + ["c" * 64] for r in rows[1:]], "comes from sheet"),
])
def test_import_refuses_a_grades_file_without_the_plans_sheet_id(built, tmp_path, capsys, mutate, message):
    grades = _grades_file(tmp_path, mutate(_grades(built.plan)))
    out = tmp_path / "records"
    assert I.main(["--grades", str(grades), "--plan", str(built.plan_path), "--out", str(out)]) == 3
    assert message in capsys.readouterr().err and not out.exists()


def test_import_refuses_a_plan_without_a_sheet_id(built, tmp_path, capsys):
    for mutate in (lambda p: p.pop("sheet_id"), lambda p: p.update(sheet_id="0" * 12)):
        plan_path = _plan_copy(built, tmp_path, mutate)
        grades = _grades_file(tmp_path, _grades(built.plan))
        assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(tmp_path / "r")]) == 3
        assert "no sheet id" in capsys.readouterr().err


# --- Rebuild with --force -------------------------------------------------------------------


def test_force_refuses_a_changed_selection_that_keeps_its_shuffle_seed(setup):
    plan = _build(setup)
    sheet = (setup.out / B.SHEET_NAME).read_bytes()
    _write_selection(setup.sel_path, _changed_rows())
    with pytest.raises(B.Refused, match="shuffle seed did not"):
        _build(setup, force=True)
    assert (setup.out / B.SHEET_NAME).read_bytes() == sheet
    _write_selection(setup.sel_path, _changed_rows(), seed=4243)
    again = _build(setup, force=True)
    assert again["sheet_id"] != plan["sheet_id"]
    assert not {r["code"] for r in again["records"]} & {r["code"] for r in plan["records"]}


def test_force_refuses_a_sheet_without_its_plan_or_an_unreadable_plan(setup):
    _build(setup)
    plan_path = setup.out / B.SEALED_DIR / B.PLAN_NAME
    plan_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(B.Refused, match="cannot read the existing plan"):
        _build(setup, force=True)
    plan_path.write_text("[]", encoding="utf-8")
    with pytest.raises(B.Refused, match="not a JSON object"):
        _build(setup, force=True)
    plan_path.unlink()
    with pytest.raises(B.Refused, match="without its plan"):
        _build(setup, force=True)


# --- Target margin: the screen never shows a level ------------------------------------------


def test_main_refuses_targets_below_the_margin_and_seals_the_detail(setup, capsys):
    _write_selection(setup.sel_path, _rows_with_grounding_2(5))
    argv = [a for p in setup.paths for a in ("--log", str(p))] + [
        "--cases", str(setup.cases_path), "--selection", str(setup.sel_path), "--out", str(setup.out)]
    assert B.main(argv) == 3
    captured = capsys.readouterr()
    screen = captured.out + captured.err
    assert "margin of 6" in screen and B.SHORTFALL_NAME in screen
    assert not re.search(r"level \d|short of|\d+ records,", screen)
    detail = (setup.sel_path.parent / B.SHORTFALL_NAME).read_text(encoding="utf-8")
    assert f"{GROUNDING} level 2 has 5 records, 1 short of 6" in detail
    assert not setup.out.exists()


def test_import_refuses_plan_targets_below_the_margin_and_seals_the_detail(built, tmp_path, capsys):
    def five_at_grounding_2(p):
        seen = 0
        for r in p["records"]:
            if r["stratum"][GROUNDING] == 2:
                seen += 1
                if seen > 5:
                    r["stratum"] = {**r["stratum"], GROUNDING: 3}
    plan_path = _plan_copy(built, tmp_path, five_at_grounding_2)
    grades = _grades_file(tmp_path, _grades(built.plan))
    assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(tmp_path / "r")]) == 3
    captured = capsys.readouterr()
    screen = captured.out + captured.err
    assert "targets miss the stratification margin" in screen
    assert not re.search(r"level \d|short of|\d+ records,", screen)
    detail = (plan_path.parent / I.SHORTFALL_NAME).read_text(encoding="utf-8")
    assert f"{GROUNDING} level 2 has 5 records, 1 short of 6" in detail
