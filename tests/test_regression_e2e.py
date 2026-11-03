"""End-to-end rehearsal of the deliberate-regression check (pre-registration section
13 pilot row, section 10 gates).

Three mockllm runs of the 20-case dev pilot with epochs = 4, built by
scripts/rehearse_regression.py: a correct baseline, a deliberately degraded
worker (wrong direction on 16 cases, a fabricated counterparty on 10) and a second
correct run. compare_runs.py must block the regression through 10(a) and 10(b)
(exit 1) and pass the no-change run with no sign-off case (exit 0). The outputs
are deterministic, so this proves plumbing and gate arithmetic, not power. The
evaluation split is never built. Set ICFR_TEST_LOG_DIR to choose where the runs
write logs; pytest's temp dir is the default, never the repo.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals.regression import DIMENSIONS, PASS4, compare, records_from_log

REPO = Path(__file__).resolve().parents[1]
K = 4
PILOT = 20


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _log_dir(tmp_path_factory, label: str) -> str:
    root = os.environ.get("ICFR_TEST_LOG_DIR")
    if root:
        path = Path(root) / f"{label}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(tmp_path_factory.mktemp(label))


@pytest.fixture(scope="module")
def rh():
    return _load("rehearse_regression")


@pytest.fixture(scope="module")
def cli():
    return _load("compare_runs")


@pytest.fixture(scope="module")
def runs(rh, tmp_path_factory):
    log_dir = _log_dir(tmp_path_factory, "rehearsal")
    baseline = rh.run_pilot("baseline", rh.correct_worker, log_dir)
    regression = rh.run_pilot("regression", rh.degraded_worker, log_dir)
    no_change = rh.run_pilot("no-change", rh.correct_worker, log_dir)
    ids = list(rh.pilot_truths())
    return SimpleNamespace(
        dir=log_dir, ids=ids, plan=rh.degraded_plan(ids),
        baseline=baseline, regression=regression, no_change=no_change,
        records={r.label: records_from_log(r.path) for r in (baseline, regression, no_change)},
    )


def _cli(cli, argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _by_case(records):
    out = defaultdict(list)
    for r in records:
        out[r.sample_id].append(r)
    return out


# --- The runs themselves -------------------------------------------------------------

def test_degraded_plan_is_the_deliberate_regression(runs):
    flipped = [sid for sid, (flip, _) in runs.plan.items() if flip]
    fabricated = [sid for sid, (_, fab) in runs.plan.items() if fab]
    assert len(runs.ids) == PILOT
    assert (len(flipped), len(fabricated)) == (16, 10)
    assert len(fabricated) >= 6  # section 10(b): fewer than 6 discordant cases can never block


@pytest.mark.parametrize("label", ["baseline", "regression", "no-change"])
def test_each_run_is_the_dev_pilot_with_four_epochs(runs, label):
    from inspect_ai.log import read_eval_log

    path = {"baseline": runs.baseline, "regression": runs.regression, "no-change": runs.no_change}[label].path
    log = read_eval_log(path, header_only=True)
    assert log.eval.task_args["split"] == "dev"
    assert log.eval.task_args["subset"] == "pilot"
    assert int(log.eval.config.epochs) == K
    cases = _by_case(runs.records[label])
    assert sorted(cases) == sorted(runs.ids)
    assert all(sorted(r.epoch for r in rs) == list(range(1, K + 1)) for rs in cases.values())


@pytest.mark.parametrize("label", ["baseline", "no-change"])
def test_correct_runs_pass_every_trial_with_full_marks(runs, label):
    for r in runs.records[label]:
        assert r.success, (r.sample_id, r.epoch, r.failure_modes, r.review_routes)
        assert r.failure_modes == () and r.review_routes == ()
        assert dict(r.dims) == {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}


def test_regression_records_fire_the_planned_codes(runs):
    for r in runs.records["regression"]:
        flip, fab = runs.plan[r.sample_id]
        assert ("FM-003" in r.failure_modes) == flip, r
        assert ("FM-005" in r.failure_modes) == fab, r
        assert r.dims["correctness"] == (0.0 if flip else 3.0)
        assert r.dims["faithfulness"] == (0.0 if fab else 3.0)
        assert r.success == (not flip and not fab)


# --- compare_runs.py verdicts ---------------------------------------------------------

def test_regression_blocks_through_10a_and_10b_exit_1(runs, cli, tmp_path):
    out_json = tmp_path / "regression.json"
    code, report = _cli(cli, ["--baseline", runs.baseline.path, "--candidate", runs.regression.path,
                              "--json", str(out_json)])
    assert code == 1
    assert "RESULT: BLOCKED (automatic)" in report
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 1 and payload["blocked"] is True
    blocking = {name for name, t in payload["paired"].items() if t["rejected"]}
    assert blocking == {"correctness", "faithfulness", PASS4}
    assert payload["fabrication"]["FM-005"]["rejected"] is True
    assert payload["fabrication"]["FM-015"]["rejected"] is False


def test_regression_gate_numbers(runs):
    rep = compare(runs.records["baseline"], [runs.records["regression"]])
    assert rep.blocked
    corr, faith, p4 = rep.paired["correctness"], rep.paired["faithfulness"], rep.paired[PASS4]
    # 16 of 20 cases drop 3 -> 0; 10 of 20 cases fabricate.
    assert corr.detail["mean_diff"] == pytest.approx(-2.4)
    assert faith.detail["mean_diff"] == pytest.approx(-1.5)
    # scipy 1.18.1: ttest_rel(candidate, baseline, alternative='less')
    assert corr.p == pytest.approx(2.2857190144936008e-08, rel=1e-9)
    assert faith.p == pytest.approx(0.0001689408190154883, rel=1e-9)
    assert (p4.detail["worse"], p4.detail["better"]) == (18, 0)
    assert p4.p == pytest.approx(2.0**-18)
    # Holm, m = 6: ranks 1..3 are correctness, content_pass4, faithfulness.
    assert corr.p_holm == pytest.approx(6 * corr.p)
    assert p4.p_holm == pytest.approx(5 * p4.p)
    assert faith.p_holm == pytest.approx(4 * faith.p)
    fab = rep.fabrication["FM-005"]
    assert (fab.detail["worse"], fab.detail["better"], fab.p) == (10, 0, 2.0**-10)
    assert fab.p_holm is None and fab.rejected
    for dim in ("calibration", "robustness"):
        assert rep.paired[dim].p == 1.0 and not rep.paired[dim].rejected


def test_regression_signoff_lists_every_planned_case(runs):
    rep = compare(runs.records["baseline"], [runs.records["regression"]])
    expected = {(sid, "FM-003") for sid, (flip, _) in runs.plan.items() if flip}
    expected |= {(sid, "FM-005") for sid, (_, fab) in runs.plan.items() if fab}
    assert {(c.sample_id, c.trigger) for c in rep.signoff} == expected
    assert all((c.baseline_count, c.candidate_count, c.rule) == (0, K, "R1") for c in rep.signoff)


def test_no_change_passes_with_no_signoff_exit_0(runs, cli, tmp_path):
    out_json = tmp_path / "no-change.json"
    code, report = _cli(cli, ["--baseline", runs.baseline.path, "--candidate", runs.no_change.path,
                              "--json", str(out_json)])
    assert code == 0
    assert "RESULT: PASS" in report
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["blocked"] is False and payload["signoff"] == []
    rep = compare(runs.records["baseline"], [runs.records["no-change"]])
    assert not rep.blocked and rep.signoff == ()
    assert all(not t.rejected and t.p == 1.0 for t in rep.paired.values())
    assert all(not t.rejected and t.p == 1.0 for t in rep.fabrication.values())


@pytest.mark.parametrize("ladder", ["R1", "R2"])
def test_pooling_a_clean_rerun_does_not_clear_the_regression(runs, cli, ladder):
    # Section 10(d): re-runs pool and never replace; a clean redraw cannot buy a pass.
    code, report = _cli(cli, ["--baseline", runs.baseline.path, "--candidate", runs.regression.path,
                              "--candidate", runs.no_change.path, "--ladder", ladder])
    assert code == 1 and "RESULT: BLOCKED (automatic)" in report
    rep = compare(runs.records["baseline"], [runs.records["regression"], runs.records["no-change"]], ladder=ladder)
    assert rep.paired[PASS4].rejected and rep.fabrication["FM-005"].rejected
    assert rep.paired["correctness"].rejected
    assert {(c.candidate_count, c.baseline_count) for c in rep.signoff} == {(K, 0)}


def test_rehearse_prints_both_reports(rh, tmp_path_factory, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rehearse_regression.py", _log_dir(tmp_path_factory, "rehearsal-main")])
    assert rh.main() == 0
    out = capsys.readouterr().out
    assert "=== regression vs baseline (compare_runs.py exit 1, expected 1) ===" in out
    assert "=== no-change vs baseline (compare_runs.py exit 0, expected 0) ===" in out
    assert out.count("Section 10(a) paired rule") == 2
    assert "REHEARSAL RESULT: PASS" in out


def test_dimensions_contract_unchanged():
    assert DIMENSIONS == ("correctness", "faithfulness", "calibration", "completeness", "robustness")
