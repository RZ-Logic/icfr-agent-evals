"""The regression sign-off file (pre-registration section 10(c), section 13 row 26).

Part A, the loader and validator (icfr_agent_evals/signoff.py) on hand-built cases:
the happy path, then one test per refusal. Part B, compare_runs.py on two mockllm
runs of the 20-case dev pilot (no API key, dev split only): a correct baseline and
a candidate degraded on two cases, which raises sign-off cases and no automatic
block. A pending sign-off is cleared only by a valid file for that exact run pair.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import signoff as so
from icfr_agent_evals.regression import SignoffCase, compare, records_from_log
from icfr_agent_evals.reliability import GateError

REPO = Path(__file__).resolve().parents[1]
KEY = "ab" * 32
OTHER_KEY = "cd" * 32
PAIR = so.RunPair(pr=12, content_key=KEY, baseline_eval_id="BASE-1", candidate_eval_ids=("CAND-2", "CAND-1"))
SWEEP = "narrative_sweep: unknown_counterparty: Phantom Custody Group"
CASES = (
    SignoffCase("dev-001", "FM-003", 0, 4, "R1"),
    SignoffCase("dev-002", "FM-005", 0, 1, "R1"),
    SignoffCase("dev-003", SWEEP, 1, 2, "R1"),
)
FILLED = {
    ("dev-001", "FM-003"): ("a1_fixed", "direction rule fixed in the prompt"),
    ("dev-002", "FM-005"): ("a3_accepted_trade_off", "policy: shorter citations accepted"),
    ("dev-003", SWEEP): ("a2_noise", "single-epoch rise, change touches only the band table"),
}


def _rows(pair=PAIR, cases=CASES):
    rows = so.template_rows(pair, cases)
    for row in rows:
        verdict, note = FILLED.get((row["sample_id"], row["route_key"]), ("b_false_positive", "extractor defect #7"))
        row.update(verdict=verdict, note=note, reviewer="Rizwan Ahmed", date="2026-11-11")
    return rows


def _write(tmp_path, rows=None, text=None, name="pr-12.csv") -> Path:
    path = tmp_path / name
    body = so.signoff_text(_rows() if rows is None else rows) if text is None else text
    path.write_bytes(body.encode("utf-8"))
    return path


def _refused(path, match, pair=PAIR, cases=CASES):
    with pytest.raises(GateError, match=match):
        so.check_signoff_file(path, pair, cases)


# --- Part A: the format ----------------------------------------------------------------

def test_format_is_the_code_audit_record_plus_a_purpose_field():
    assert so.SIGNOFF_COLUMNS[:len(rs.RB7_COLUMNS)] == rs.RB7_COLUMNS
    assert so.SIGNOFF_COLUMNS[len(rs.RB7_COLUMNS)] == "purpose"
    assert so.PURPOSES == ("code_audit", "regression_signoff") and so.PURPOSE == "regression_signoff"
    assert so.OUTCOMES == ("a1_fixed", "a2_noise", "a3_accepted_trade_off", "b_false_positive")


def test_signoff_path_is_per_pr_and_gated_content_key():
    assert so.signoff_path(REPO, 12, KEY) == REPO / "docs/evidence/regression/signoffs" / f"pr-12-{KEY[:12]}.csv"
    for bad in ("AB" * 32, KEY[:-1], "zz" * 32, None):
        with pytest.raises(GateError, match="64 lowercase hex"):
            so.signoff_path(REPO, 12, bad)
    for bad in (0, -1, True, "12"):
        with pytest.raises(GateError, match="positive integer"):
            so.signoff_path(REPO, bad, KEY)


def test_happy_path_accepts_a_file_signing_every_case(tmp_path):
    rec = so.check_signoff_file(_write(tmp_path), PAIR, CASES)
    assert rec.outcomes == {k: v[0] for k, v in FILLED.items()}
    assert rec.counts() == {"a1_fixed": 1, "a2_noise": 1, "a3_accepted_trade_off": 1, "b_false_positive": 0}
    assert rec.policy_decisions() == (("dev-002", "FM-005"),)
    assert all(r["log_eval_id"] == "CAND-1 | CAND-2" for r in rec.rows)


def test_template_rows_fill_everything_but_the_reviewer_cells():
    rows = so.template_rows(PAIR, CASES)
    assert [r["route_class"] for r in rows] == ["fm_code", "fm_code", "narrative_sweep"]
    assert all((r["verdict"], r["note"], r["reviewer"], r["date"]) == ("", "", "", "") for r in rows)
    assert [(r["baseline_count"], r["candidate_count"]) for r in rows] == [("0", "4"), ("0", "1"), ("1", "2")]


# --- Part A: refusals, one per check ----------------------------------------------------

def test_refuses_a_missing_file(tmp_path):
    _refused(tmp_path / "absent.csv", "cannot read it")


@pytest.mark.parametrize("mangle, match", [
    (lambda t: "﻿" + t, "byte order mark"),
    (lambda t: t.replace("\n", "\r\n"), "carriage return"),
    (lambda t: t.replace(",purpose,", ",goal,", 1), "header"),
    (lambda t: t.rstrip("\n") + ",extra\n", "cells"),
], ids=["bom", "crlf", "wrong-header", "ragged-row"])
def test_refuses_a_malformed_file(tmp_path, mangle, match):
    _refused(_write(tmp_path, text=mangle(so.signoff_text(_rows()))), match)


def test_refuses_a_file_with_no_rows(tmp_path):
    _refused(_write(tmp_path, rows=[]), "holds no sign-off rows")


def _edit(col, value, row=0):
    def go(rows):
        rows[row][col] = value
        return rows
    return go


@pytest.mark.parametrize("edit, match", [
    (_edit("verdict", "approved"), "is not one of the four outcomes"),
    (_edit("verdict", "a1"), "is not one of the four outcomes"),
    (_edit("verdict", ""), "is not one of the four outcomes"),
    (_edit("purpose", "code_audit"), "purpose 'code_audit'"),
    (_edit("purpose", ""), "purpose ''"),
    (_edit("row_kind", "fm_audit"), "row_kind 'fm_audit'"),
    (_edit("note", "  "), "note is blank"),
    (_edit("reviewer", ""), "reviewer is blank"),
    (_edit("date", "11/11/2026"), "is not a YYYY-MM-DD date"),
    (_edit("epoch", "1"), "epoch must be blank"),
    (_edit("unit_id", "A-1"), "unit_id must be blank"),
    (_edit("sample_id", ""), "sample_id is blank"),
    (_edit("route_key", "FM-007"), "neither a sign-off FM code nor a route key"),
    (_edit("route_class", "code_review"), "route_class 'code_review', expected 'fm_code'"),
    (_edit("baseline_count", "x"), "is not a whole number"),
], ids=["unknown-outcome", "short-outcome", "blank-outcome", "code-audit-purpose", "blank-purpose",
        "row-kind", "note", "reviewer", "date", "epoch", "unit-id", "sample-id", "unknown-trigger",
        "route-class", "count-not-number"])
def test_refuses_a_malformed_row(tmp_path, edit, match):
    _refused(_write(tmp_path, rows=edit(_rows())), match)


@pytest.mark.parametrize("edit, match", [
    (_edit("gated_content_sha256", OTHER_KEY), "does not match the run pair's gated-content key"),
    (_edit("gated_content_sha256", KEY.upper()), "does not match the run pair's gated-content key"),
    (_edit("run_label", "pr-13"), "signs another PR"),
    (_edit("run_label", "12"), "is not pr-<N>"),
    (_edit("baseline_eval_id", "BASE-9"), "is not the baseline run's"),
    (_edit("log_eval_id", "CAND-1"), "is not the candidate run"),
], ids=["other-key", "key-case", "other-pr", "pr-shape", "other-baseline", "other-candidates"])
def test_refuses_a_file_for_another_run_pair(tmp_path, edit, match):
    _refused(_write(tmp_path, rows=edit(_rows())), match)


def test_refuses_a_whole_file_for_another_content_key(tmp_path):
    other = so.RunPair(12, OTHER_KEY, PAIR.baseline_eval_id, PAIR.candidate_eval_ids)
    _refused(_write(tmp_path, rows=_rows(pair=other)), "does not match the run pair's gated-content key")


def test_refuses_a_missing_case(tmp_path):
    _refused(_write(tmp_path, rows=_rows()[:2]), "unsigned case\\(s\\): dev-003 narrative_sweep")


def test_refuses_a_case_the_run_pair_does_not_trigger(tmp_path):
    extra = _rows(cases=(*CASES, SignoffCase("dev-004", "FM-001", 0, 1, "R1")))
    _refused(_write(tmp_path, rows=extra), "dev-004 FM-001 is not a sign-off case of this run pair")


def test_refuses_a_case_signed_twice(tmp_path):
    rows = _rows()
    _refused(_write(tmp_path, rows=[*rows, dict(rows[0])]), "dev-001 FM-003 is signed twice")


def test_refuses_counts_other_than_the_run_pair(tmp_path):
    _refused(_write(tmp_path, rows=_edit("candidate_count", "3")(_rows())),
             "counts baseline 0 -> candidate 3, but this run pair has 0 -> 4")


def test_refuses_noise_on_a_rise_of_more_than_one_epoch(tmp_path):
    _refused(_write(tmp_path, rows=_edit("verdict", "a2_noise")(_rows())), "single-epoch rise; this case rose 4")


def test_refusal_names_every_problem_at_once(tmp_path):
    rows = _edit("note", "")(_edit("verdict", "ok", row=1)(_rows()))
    with pytest.raises(GateError) as e:
        so.check_signoff_file(_write(tmp_path, rows=rows), PAIR, CASES)
    assert "row 2: note is blank" in str(e.value) and "row 3: verdict 'ok'" in str(e.value)


def test_format_spec_names_every_column_and_outcome():
    spec = (REPO / "docs/evidence/regression/SIGNOFF-FORMAT.md").read_text(encoding="utf-8")
    for name in (*so.SIGNOFF_COLUMNS, *so.OUTCOMES, so.SIGNOFF_DIR_RELPATH):
        assert f"`{name}`" in spec, name
    assert "\u2014" not in spec and "\u2013" not in spec


# --- Part B: compare_runs.py on a run pair with sign-off cases ---------------------------

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    rh = _load("rehearse_regression")

    def two_case_worker(truth, index):
        # case 0 states the wrong direction (FM-003), case 1 cites a fabricated counterparty (FM-005)
        direction = rh._flip(truth.direction) if index == 0 else truth.direction
        return rh._answer(truth, direction=direction, fabricate=index == 1)

    log_dir = str(tmp_path_factory.mktemp("signoff-runs"))
    baseline = rh.run_pilot("baseline", rh.correct_worker, log_dir)
    candidate = rh.run_pilot("candidate", two_case_worker, log_dir)
    regression = rh.run_pilot("regression", rh.degraded_worker, log_dir)
    from inspect_ai.log import read_eval_log

    ids = {r.label: read_eval_log(r.path, header_only=True).eval.eval_id for r in (baseline, candidate, regression)}
    report = compare(records_from_log(baseline.path), [records_from_log(candidate.path)])
    blocked = compare(records_from_log(baseline.path), [records_from_log(regression.path)])
    return baseline, candidate, regression, ids, report, blocked


@pytest.fixture()
def cli(tmp_path, monkeypatch):
    mod = _load("compare_runs")
    monkeypatch.setattr(mod, "DEFAULT_LADDER_RECORD", tmp_path / "no-ladder" / "ladder.json")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path / "repo")
    return mod


def _cli(cli, argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def _pair(runs, label="candidate", key=KEY):
    return so.RunPair(12, key, runs[3]["baseline"], (runs[3][label],))


def _commit(cli, runs, rows) -> Path:
    path = so.signoff_path(cli.REPO_ROOT, 12, KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(so.signoff_text(rows).encode("utf-8"))
    return path


def _argv(runs, *extra, label="candidate"):
    path = runs[1].path if label == "candidate" else runs[2].path
    return ["--baseline", runs[0].path, "--candidate", path, *extra]


def test_the_pair_raises_signoff_cases_and_no_block(runs):
    report = runs[4]
    assert not report.blocked
    assert {(c.trigger, c.baseline_count, c.candidate_count) for c in report.signoff} >= {("FM-003", 0, 4), ("FM-005", 0, 4)}


def test_without_pr_and_key_signoff_stays_pending_exit_2(runs, cli):
    code, out, _ = _cli(cli, _argv(runs))
    assert code == 2 and "RESULT: SIGN-OFF PENDING" in out


def test_no_file_for_the_pair_stays_pending_exit_2(runs, cli):
    code, out, _ = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY))
    assert code == 2 and "RESULT: SIGN-OFF PENDING" in out and "no sign-off file at" in out


def test_a_valid_file_for_the_exact_pair_clears_it_exit_0(runs, cli, tmp_path):
    rows = _rows(pair=_pair(runs), cases=runs[4].signoff)
    path = _commit(cli, runs, rows)
    out_json = tmp_path / "report.json"
    code, out, err = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY, "--json", str(out_json)))
    assert code == 0, err
    assert "RESULT: PASS (every sign-off case signed)" in out
    assert "signed: b_false_positive" in out
    assert f"sign-off file {path}" in out
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0 and payload["signoff_record"]["gated_content_sha256"] == KEY
    assert sum(payload["signoff_record"]["outcome_counts"].values()) == len(runs[4].signoff)


def test_an_accepted_trade_off_on_fabrication_is_reported_as_policy(runs, cli):
    rows = _rows(pair=_pair(runs), cases=runs[4].signoff)
    for row in rows:
        if row["route_key"] == "FM-005":
            row["verdict"] = "a3_accepted_trade_off"
    _commit(cli, runs, rows)
    code, out, _ = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY))
    assert code == 0 and "policy decision: accepted trade-off on fabrication code FM-005" in out


def test_an_explicit_signoff_path_is_read(runs, cli, tmp_path):
    path = _write(tmp_path, rows=_rows(pair=_pair(runs), cases=runs[4].signoff), name="elsewhere.csv")
    code, out, _ = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY, "--signoff", str(path)))
    assert code == 0 and "RESULT: PASS (every sign-off case signed)" in out


def test_a_file_for_another_content_key_is_refused_exit_3(runs, cli, tmp_path):
    rows = _rows(pair=_pair(runs, key=OTHER_KEY), cases=runs[4].signoff)
    path = _write(tmp_path, rows=rows)
    code, out, err = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY, "--signoff", str(path)))
    assert code == 3 and out == "" and "gated-content key" in err


def test_a_file_for_another_candidate_run_is_refused_exit_3(runs, cli, tmp_path):
    rows = _rows(pair=_pair(runs, label="regression"), cases=runs[4].signoff)
    path = _write(tmp_path, rows=rows)
    code, _, err = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY, "--signoff", str(path)))
    assert code == 3 and "is not the candidate run" in err


def test_a_file_missing_a_case_is_refused_exit_3(runs, cli):
    _commit(cli, runs, _rows(pair=_pair(runs), cases=runs[4].signoff)[1:])
    code, _, err = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY))
    assert code == 3 and "unsigned case(s)" in err


@pytest.mark.parametrize("extra, match", [
    (["--pr", "12"], "go together"),
    (["--content-key", KEY], "go together"),
    (["--signoff", "x.csv"], "needs --pr and --content-key"),
    (["--pr", "12", "--content-key", KEY.upper()], "64 lowercase hex"),
    (["--pr", "0", "--content-key", KEY], "positive integer"),
])
def test_incomplete_or_bad_pair_arguments_are_refused_exit_3(runs, cli, extra, match):
    code, out, err = _cli(cli, _argv(runs, *extra))
    assert code == 3 and out == "" and match in err


def test_a_signoff_file_never_clears_an_automatic_block(runs, cli, tmp_path):
    pair = so.RunPair(12, KEY, runs[3]["baseline"], (runs[3]["regression"],))
    rows = _rows(pair=pair, cases=runs[5].signoff)
    path = _write(tmp_path, rows=rows)
    code, out, _ = _cli(cli, _argv(runs, "--pr", "12", "--content-key", KEY, "--signoff", str(path),
                                   label="regression"))
    assert code == 1 and "RESULT: BLOCKED (automatic)" in out
