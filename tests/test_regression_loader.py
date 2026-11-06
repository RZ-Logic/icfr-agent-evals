"""Regression-gate loader and CLI tests (pre-registration section 10).

Covers how per (sample, epoch) Inspect records become regression.EpochRecord
(dimension mapping, strict zeros, trial success, review routes), one real mockllm
run of variance_commentary on the dev split with epochs = 4, the compare_runs.py
exit codes, and the refusal of evaluation-split logs. The evaluation split is
never built: its refusal is tested on a dev-split log whose header is edited.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.regression import (
    CODE_ROUTE,
    DIMENSION_SCORERS,
    SWEEP_ROUTE,
    UNSTATED_REASON,
    _SWEEP_KINDS,
    compare,
    record_from_sample,
    records_from_log,
    records_from_samples,
    route_class,
)
from icfr_agent_evals.reliability import GateError, evaluate_trial
from icfr_agent_evals.rubric import WIRED_SCORERS
from icfr_agent_evals.scoring import FABRICATION_REVIEW, SweepHit, check_fabrication, check_reconciling_items
from test_reliability import c2_logs  # noqa: F401  (contract C2 mock logs: clean, overrides, template)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compare_runs.py"
SCORERS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
           "completeness_anchor", "injection_compliance")
WIRED = {"correctness", "faithfulness", "completeness"}


def _score(value=3, fms=(), reasons=(), **extra):
    meta = {"failure_modes": list(fms), "requires_human_review": bool(reasons), "review_reasons": list(reasons), **extra}
    return SimpleNamespace(value=value, metadata=meta)


def _sample(sid="dev-x", epoch=1, overrides=None, error=None, drop=()):
    scores = {name: _score() for name in SCORERS}
    scores.update(overrides or {})
    for name in drop:
        scores.pop(name)
    return SimpleNamespace(id=sid, epoch=epoch, scores=scores, error=error)


# --- Mapping on fake samples -----------------------------------------------------------

def test_dimension_scorers_agree_with_the_rubric_registry():
    assert set(DIMENSION_SCORERS) == WIRED
    for dim, scorer in DIMENSION_SCORERS.items():
        assert WIRED_SCORERS[scorer] == dim


def test_clean_epoch_maps_to_full_marks_and_no_routes():
    r = record_from_sample(_sample())
    assert r.sample_id == "dev-x" and r.epoch == 1
    assert dict(r.dims) == {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}
    assert "calibration" not in r.dims and "robustness" not in r.dims
    assert r.success and r.failure_modes == () and r.review_routes == ()


def test_faithfulness_reads_the_combined_anchor_not_its_parts():
    r = record_from_sample(_sample(overrides={"fabrication_anchor": _score(1), "reconciling_items_anchor": _score(0),
                                              "faithfulness_anchor": _score(2)}))
    assert r.dims["faithfulness"] == 2.0


@pytest.mark.parametrize("value", [None, "C", float("nan"), float("inf"), True])
def test_non_numeric_value_scores_zero(value):
    r = record_from_sample(_sample(overrides={"correctness_anchor": _score(value)}))
    assert r.dims["correctness"] == 0.0 and r.dims["completeness"] == 3.0
    assert not r.success


def test_missing_score_scores_zero():
    r = record_from_sample(_sample(drop=("completeness_anchor",)))
    assert r.dims["completeness"] == 0.0 and not r.success


def test_errored_epoch_scores_zero_everywhere_and_is_not_a_review_route():
    r = record_from_sample(_sample(error=SimpleNamespace(message="boom")))
    assert set(r.dims.values()) == {0.0}
    assert not r.success and r.review_routes == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"correctness_anchor": _score(2, fms=("FM-002",))},
        {"fabrication_anchor": _score(0, fms=("FM-005",), reasons=(FABRICATION_REVIEW,))},
        {"injection_compliance": _score(0, fms=("FM-006",))},
        {"correctness_anchor": _score(3, reasons=("narrative direction word unresolved",))},
    ],
)
def test_success_and_failure_modes_match_evaluate_trial(overrides):
    s = _sample(overrides=overrides)
    r, t = record_from_sample(s), evaluate_trial(s)
    assert (r.success, r.failure_modes) == (t.success, t.failure_modes)
    assert not r.success


def test_format_break_routes_through_fm000_not_a_review_route():
    fb = {name: _score(0, fms=("FM-000",), reasons=("format break: no JSON object",), format_break=True) for name in SCORERS}
    r = record_from_sample(_sample(overrides=fb))
    assert r.failure_modes == ("FM-000",)
    assert r.review_routes == ()
    assert set(r.dims.values()) == {0.0}


@pytest.mark.parametrize("kind", _SWEEP_KINDS)
def test_every_sweep_hit_reason_maps_to_the_sweep_route(kind):
    reason = SweepHit("Phantom Custody Group", kind).reason
    r = record_from_sample(_sample(overrides={"fabrication_anchor": _score(0, reasons=(reason,)),
                                              "faithfulness_anchor": _score(0, reasons=(reason,))}))
    assert r.review_routes == (f"{SWEEP_ROUTE}: {reason}",)  # repeated by faithfulness_anchor, counted once
    assert route_class(r.review_routes[0]) == SWEEP_ROUTE
    assert r.failure_modes == () and not r.success


def test_other_code_check_reasons_map_to_the_code_route():
    r = record_from_sample(_sample(overrides={"correctness_anchor": _score(3, reasons=("amount role unresolved",))}))
    assert r.review_routes == (f"{CODE_ROUTE}: amount role unresolved",)
    assert route_class(r.review_routes[0]) == CODE_ROUTE


def test_review_flag_without_reasons_or_codes_is_a_code_route():
    s = _sample(overrides={"completeness_anchor": SimpleNamespace(value=3, metadata={"requires_human_review": True})})
    assert record_from_sample(s).review_routes == (f"{CODE_ROUTE}: {UNSTATED_REASON}",)


def test_fm_attached_review_reason_is_not_a_route():
    r = record_from_sample(_sample(overrides={"reconciling_items_anchor": _score(0, fms=("FM-015",), reasons=(FABRICATION_REVIEW,)),
                                              "faithfulness_anchor": _score(0, fms=("FM-015",), reasons=(FABRICATION_REVIEW,))}))
    assert r.failure_modes == ("FM-015",)
    assert r.review_routes == ()


def test_sweep_hit_beside_a_fired_code_still_counts_as_a_route():
    """Reasons are read one by one: a sweep hit riding with FM-005 is a second,
    unconfirmed entity, so it still counts on the sweep route (section 10(c), each
    review reason counts on its own, ratified 2026-11-03)."""
    sweep = SweepHit("Phantom Custody", "organization_suffix").reason
    r = record_from_sample(_sample(overrides={"fabrication_anchor": _score(0, fms=("FM-005",), reasons=(FABRICATION_REVIEW, sweep)),
                                              "correctness_anchor": _score(3, reasons=("band statement unresolved",))}))
    assert r.failure_modes == ("FM-005",)
    assert r.review_routes == (f"{SWEEP_ROUTE}: {sweep}", f"{CODE_ROUTE}: band statement unresolved")


def test_real_sweep_and_code_reasons_from_scoring():
    truth = make_truth("SOL", case_id="dev-route")
    swept = check_fabrication(good_output(truth, narrative="Balances were also confirmed by Phantom Custody Group."), truth)
    assert swept.review_reasons and not swept.failure_modes
    out = good_output(truth)
    assert out["reconciling_items"], "make_truth must carry an eligible item"
    out["reconciling_items"][0]["type"] = "dividend"
    typed = check_reconciling_items(out, truth)
    assert typed.review_reasons and not typed.failure_modes
    s = _sample(overrides={"fabrication_anchor": _score(0, reasons=swept.review_reasons),
                           "reconciling_items_anchor": _score(3, reasons=typed.review_reasons)})
    routes = record_from_sample(s).review_routes
    fixture_type = truth.item_by_reference(out["reconciling_items"][0]["reference"]).type
    assert routes == (*sorted(f"{SWEEP_ROUTE}: {r}" for r in set(swept.review_reasons)),
                      f"{CODE_ROUTE}: reconciling item type differs from {fixture_type!r}")
    assert [route_class(k) for k in routes] == [SWEEP_ROUTE] * (len(routes) - 1) + [CODE_ROUTE]


def test_records_from_samples_keeps_every_epoch():
    recs = records_from_samples([_sample("a", e) for e in (1, 2, 3, 4)])
    assert [(r.sample_id, r.epoch) for r in recs] == [("a", 1), ("a", 2), ("a", 3), ("a", 4)]


# --- One real mockllm run on the dev split, epochs = 4 ----------------------------------

FABRICATED = "Phantom Settlement Partners LLC"
SWEEP_NARRATIVE = "Reserves were confirmed with Phantom Custody Group for the period."


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    """Three standard dev cases, four epochs each: A clean on every epoch, B cites
    a fabricated entity on exactly one generation, C names an uncited custodian in
    its narrative on exactly two generations. Generations are counted per case, so
    the per-case totals are fixed whatever order Inspect runs the epochs in."""
    from icfr_agent_evals.task import variance_commentary

    task = variance_commentary(split="dev", epochs=4)
    truths = {s.id: truth_of(s.metadata) for s in task.dataset}
    std = [sid for sid, t in truths.items() if t.case_kind == "standard" and not t.has_memo_only_link]
    picked = dict(zip("ABC", std[:3]))
    key_of = {truths[sid].scope_wallets[0]: name for name, sid in picked.items()}
    assert len(key_of) == 3
    calls: Counter[str] = Counter()

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        name = next(n for w, n in key_of.items() if w in blob)
        calls[name] += 1
        answer = good_output(truths[picked[name]])
        if name == "B" and calls[name] == 1:
            answer["entities_cited"] = [*answer["entities_cited"], FABRICATED]
        if name == "C" and calls[name] <= 2:
            answer["narrative"] = SWEEP_NARRATIVE
        out = ModelOutput.from_content(model="mockllm/model", content=json.dumps(answer))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    chosen = set(picked.values())
    task.dataset = task.dataset.filter(lambda s: s.id in chosen)
    log_dir = tmp_path_factory.mktemp("regression-dev-run")
    log = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=mock_outputs),
        model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
        log_dir=str(log_dir),
        display="none",
    )[0]
    assert log.status == "success", log.error
    return SimpleNamespace(log=log, path=log.location, picked=picked, dir=log_dir)


def _by_case(records):
    out = defaultdict(list)
    for r in records:
        out[r.sample_id].append(r)
    return out


def test_dev_run_records_map_every_epoch(dev_run):
    recs = records_from_log(dev_run.path)
    assert len(recs) == 12
    cases = _by_case(recs)
    assert set(cases) == set(dev_run.picked.values())
    for rs in cases.values():
        assert sorted(r.epoch for r in rs) == [1, 2, 3, 4]
        assert all(set(r.dims) == WIRED for r in rs)

    a = cases[dev_run.picked["A"]]
    assert all(r.success and not r.failure_modes and not r.review_routes for r in a)
    assert all(set(r.dims.values()) == {3.0} for r in a)

    b = cases[dev_run.picked["B"]]
    fab = [r for r in b if "FM-005" in r.failure_modes]
    assert len(fab) == 1 and not fab[0].success and fab[0].dims["faithfulness"] == 0.0
    assert fab[0].dims["correctness"] == 3.0
    assert sum(r.success for r in b) == 3

    c = cases[dev_run.picked["C"]]
    swept = [r for r in c if any(route_class(k) == SWEEP_ROUTE for k in r.review_routes)]
    assert len(swept) == 2
    assert all(not r.failure_modes and not r.success and r.dims["faithfulness"] == 0.0 for r in swept)
    assert sum(r.success for r in c) == 2


def test_dev_run_records_match_between_path_and_log_object(dev_run):
    assert records_from_log(dev_run.path) == records_from_log(dev_run.log)


def test_dev_run_against_itself_neither_blocks_nor_triggers(dev_run):
    recs = records_from_log(dev_run.log)
    rep = compare(recs, [recs])
    assert not rep.blocked and rep.signoff == ()
    assert rep.paired["calibration"].p == 1.0 and rep.paired["robustness"].p == 1.0


# --- Derived logs for the CLI ---------------------------------------------------------

def _derive(dev_run, name, edit):
    log = read_eval_log(dev_run.path)
    edit(log)
    path = dev_run.dir / f"{name}.eval"
    write_eval_log(log, str(path))
    return str(path)


def _zero_correctness(log):
    for s in log.samples:
        s.scores["correctness_anchor"].value = 0


def _one_fm016_on_a(dev_run):
    def edit(log):
        s = next(s for s in log.samples if s.id == dev_run.picked["A"] and s.epoch == 2)
        s.scores["correctness_anchor"].metadata["failure_modes"] = ["FM-016"]
    return edit


def _set_task_arg(key, value):
    def edit(log):
        log.eval.task_args = {**(log.eval.task_args or {}), key: value}
    return edit


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_identical_runs_pass(dev_run, cli, capsys):
    assert cli.main(["--baseline", dev_run.path, "--candidate", dev_run.path]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "content_pass4" in out


def test_cli_blocks_a_correctness_collapse(dev_run, cli, capsys):
    bad = _derive(dev_run, "zero-correctness", _zero_correctness)
    assert cli.main(["--baseline", dev_run.path, "--candidate", bad]) == 1
    assert "RESULT: BLOCKED" in capsys.readouterr().out


def test_cli_unwritable_json_path_is_refused_with_exit_3(dev_run, cli, capsys, tmp_path):
    """An unwritable --json path is bad input: exit 3 and no RESULT line, never a
    printed PASS followed by Python's exit 1 (which reads as a block)."""
    out_json = tmp_path / "no" / "such" / "dir" / "out.json"
    assert cli.main(["--baseline", dev_run.path, "--candidate", dev_run.path, "--json", str(out_json)]) == 3
    captured = capsys.readouterr()
    assert "RESULT" not in captured.out and "REFUSED: cannot write --json" in captured.err
    assert not out_json.exists()


def test_cli_uncaught_error_exits_3(cli, monkeypatch, capsys):
    def boom(argv=None):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(cli, "main", boom)
    assert cli._run([]) == 3
    assert "REFUSED: RuntimeError" in capsys.readouterr().err


def test_cli_signoff_pending_without_a_block(dev_run, cli, capsys, tmp_path):
    cand = _derive(dev_run, "one-fm016", _one_fm016_on_a(dev_run))
    out_json = tmp_path / "report.json"
    assert cli.main(["--baseline", dev_run.path, "--candidate", cand, "--json", str(out_json)]) == 2
    assert "SIGN-OFF PENDING" in capsys.readouterr().out
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 2 and payload["blocked"] is False
    assert [(c["sample_id"], c["trigger"], c["baseline_count"], c["candidate_count"]) for c in payload["signoff"]] == [
        (dev_run.picked["A"], "FM-016", 0, 1)
    ]


@pytest.mark.parametrize("ladder", ["R1", "R2"])
def test_cli_pooled_rerun_counts_run_against_the_candidate(dev_run, cli, tmp_path, ladder):
    """Two identical candidate runs pool to eight epochs, so the raw counts double:
    B's FM-005 (1 -> 2) and C's sweep route (2 -> 4) trigger, and both stay on R1."""
    out_json = tmp_path / "pooled.json"
    code = cli.main(["--baseline", dev_run.path, "--candidate", dev_run.path, "--candidate", dev_run.path,
                     "--ladder", ladder, "--json", str(out_json)])
    assert code == 2
    cases = {(c["sample_id"], c["trigger"]): (c["baseline_count"], c["candidate_count"], c["rule"])
             for c in json.loads(out_json.read_text(encoding="utf-8"))["signoff"]}
    assert cases[(dev_run.picked["B"], "FM-005")] == (1, 2, "R1")
    swept = {trigger: v for (sid, trigger), v in cases.items()
             if sid == dev_run.picked["C"] and route_class(trigger) == SWEEP_ROUTE}
    assert swept and set(swept.values()) == {(2, 4, "R1")}


@pytest.mark.parametrize(
    "name,edit",
    [
        ("split-eval", _set_task_arg("split", "eval")),
        ("seed-differs", _set_task_arg("seed", 1)),
        ("subset-differs", _set_task_arg("subset", "pilot")),
    ],
)
def test_cli_refuses_unmatched_or_evaluation_split_candidates(dev_run, cli, capsys, name, edit):
    cand = _derive(dev_run, name, edit)
    assert cli.main(["--baseline", dev_run.path, "--candidate", cand]) == 3
    assert "REFUSED" in capsys.readouterr().err


def test_cli_refuses_an_evaluation_split_baseline(dev_run, cli):
    base = _derive(dev_run, "baseline-split-eval", _set_task_arg("split", "eval"))
    assert cli.main(["--baseline", base, "--candidate", dev_run.path]) == 3


def test_cli_refuses_a_missing_file(dev_run, cli, tmp_path):
    assert cli.main(["--baseline", dev_run.path, "--candidate", str(tmp_path / "absent.eval")]) == 3


def test_cli_usage_error_exits_3_not_the_signoff_code(dev_run):
    proc = subprocess.run([sys.executable, str(SCRIPT), "--baseline", dev_run.path], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    proc = subprocess.run([sys.executable, str(SCRIPT), "--baseline", dev_run.path, "--candidate", dev_run.path],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --- Evaluation-split refusal on edited dev-split headers --------------------------------

def _edited(dev_run, edit):
    log = read_eval_log(dev_run.path)
    edit(log)
    return log


@pytest.mark.parametrize(
    "edit",
    [
        _set_task_arg("split", "eval"),
        lambda log: setattr(log.eval, "metadata", {**(log.eval.metadata or {}), "split": "eval"}),
        lambda log: setattr(log.eval.dataset, "name", "variance-commentary-eval"),
        lambda log: setattr(log.samples[0], "id", "eval-std-000-BTC-M1"),
    ],
    ids=["task_args", "task_metadata", "dataset_name", "sample_id"],
)
def test_records_from_log_refuses_any_evaluation_split_sign(dev_run, edit):
    with pytest.raises(GateError, match="evaluation"):
        records_from_log(_edited(dev_run, edit))


def test_records_from_log_refuses_a_log_with_no_recorded_split(dev_run):
    def edit(log):
        log.eval.task_args = {k: v for k, v in (log.eval.task_args or {}).items() if k != "split"}
        log.eval.metadata = {k: v for k, v in (log.eval.metadata or {}).items() if k != "split"}
    with pytest.raises(GateError, match="no split"):
        records_from_log(_edited(dev_run, edit))


def test_records_from_log_refuses_a_header_only_read(dev_run):
    with pytest.raises(GateError, match="no samples"):
        records_from_log(read_eval_log(dev_run.path, header_only=True))


# --- Review round 1: whole-split, matched-content, fabrication listing, help ----------

@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda log: setattr(log.eval.config, "limit", 3), "limit"),
        (lambda log: setattr(log.eval.config, "sample_id", ["x"]), "sample_id"),
        (lambda log: setattr(log, "samples", [s for s in log.samples if s.id != log.samples[0].id]), "sample count"),
        (lambda log: setattr(log.eval.dataset, "samples", 4), "sample count"),
        (lambda log: setattr(log.eval.dataset, "samples", None), "no dataset sample count"),
        (lambda log: setattr(log.eval.dataset, "sample_ids", ["dev-other-1", "dev-other-2", "dev-other-3"]),
         "sample ids differ"),
    ],
    ids=["limit", "sample_id", "dropped_case", "header_count", "no_header_count", "header_ids"],
)
def test_records_from_log_refuses_a_partial_run(dev_run, edit, match):
    """Section 6: each gate asserts scored samples equal the split's sample count."""
    with pytest.raises(GateError, match=match):
        records_from_log(_edited(dev_run, edit))


def _truncate_to_two_cases(dev_run):
    def edit(log):
        keep = {dev_run.picked["A"], dev_run.picked["B"]}
        log.samples = [s for s in log.samples if s.id in keep]
        log.eval.config.limit = 2
    return edit


def test_cli_refuses_a_truncated_pair(dev_run, cli, capsys):
    """Both runs cut to the same two cases pair cleanly and would pass; the gate must refuse."""
    base = _derive(dev_run, "trunc-base", _truncate_to_two_cases(dev_run))
    cand = _derive(dev_run, "trunc-cand", _truncate_to_two_cases(dev_run))
    assert cli.main(["--baseline", base, "--candidate", cand]) == 3
    assert "REFUSED" in capsys.readouterr().err


def _set_meta(key, value):
    def edit(log):
        log.eval.metadata = {**(log.eval.metadata or {}), key: value}
    return edit


def _drop_meta(key):
    def edit(log):
        log.eval.metadata = {k: v for k, v in (log.eval.metadata or {}).items() if k != key}
    return edit


@pytest.mark.parametrize(
    "name,edit",
    [
        ("dataset-sha", _set_meta("dataset_sha256", "0" * 64)),
        ("split-dataset-sha", _set_meta("split_dataset_sha256", "1" * 64)),
        ("dataset-version", _set_meta("dataset_version", "9.9.9")),
        ("schema-version", _set_meta("schema_version", "9.9.9")),
        ("split-seed", _set_meta("split_seed", 1)),
        ("task-version", lambda log: setattr(log.eval, "task_version", "9.9.9")),
        ("no-dataset-sha", _drop_meta("dataset_sha256")),
        ("no-subset", _drop_meta("subset")),
    ],
)
def test_cli_refuses_a_candidate_on_other_case_content(dev_run, cli, capsys, name, edit):
    """Pairing by sample id means nothing when the case content behind an id changed."""
    cand = _derive(dev_run, f"content-{name}", edit)
    assert cli.main(["--baseline", dev_run.path, "--candidate", cand]) == 3
    assert "REFUSED" in capsys.readouterr().err
    assert cli.main(["--baseline", cand, "--candidate", dev_run.path]) == 3


def test_cli_lists_every_fabrication_instance_even_without_a_rise(dev_run, cli, capsys, tmp_path):
    """Section 10(b): B's single FM-005 epoch is in both runs, so nothing rises and the
    gate passes, yet the instance is listed in the report and the JSON."""
    out_json = tmp_path / "fab.json"
    assert cli.main(["--baseline", dev_run.path, "--candidate", dev_run.path, "--json", str(out_json)]) == 0
    out = capsys.readouterr().out
    b = dev_run.picked["B"]
    assert "fabrication instances in the candidate run(s): 1" in out
    assert any(b in line and "FM-005" in line for line in out.splitlines() if line.startswith("  run 1"))
    listed = json.loads(out_json.read_text(encoding="utf-8"))["fabrication_instances"]
    assert [(f["run"], f["sample_id"], f["code"]) for f in listed] == [(1, b, "FM-005")]


def test_fabrication_instances_list_every_pooled_epoch():
    from icfr_agent_evals.regression import EpochRecord, fabrication_instances

    def rec(sid, epoch, fms=()):
        return EpochRecord(sid, epoch, {"correctness": 3.0}, not fms, tuple(fms), ())

    run1 = [rec("s1", 1, ("FM-005",)), rec("s1", 2), rec("s2", 3, ("FM-015", "FM-005", "FM-003"))]
    run2 = [rec("s2", 1, ("FM-016",)), rec("s1", 4, ("FM-015",))]
    got = [(f.run, f.sample_id, f.epoch, f.code) for f in fabrication_instances([run1, run2])]
    assert got == [(1, "s1", 1, "FM-005"), (1, "s2", 3, "FM-005"), (1, "s2", 3, "FM-015"), (2, "s1", 4, "FM-015")]


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_cli_help_exits_3_not_the_pass_code(dev_run, flag):
    proc = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert "--baseline" in proc.stdout
    proc = subprocess.run([sys.executable, str(SCRIPT), "--baseline", dev_run.path, "--candidate", dev_run.path, flag],
                          capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr


# --- Run headers: task, model, revision and worker prompt (F2-08) ---------------------

def _set_spec(**fields):
    def edit(log):
        for key, value in fields.items():
            setattr(log.eval, key, value)
    return edit


def _set_revision(commit, dirty):
    from inspect_ai.log import EvalRevision

    def edit(log):
        log.eval.revision = EvalRevision(type="git", origin="https://example.invalid/repo.git", commit=commit, dirty=dirty)
    return edit


@pytest.mark.parametrize(
    "name,edit,field",
    [
        ("other-task", _set_spec(task="variance_commentary_template_baseline"), "task"),
        ("other-prompt", _set_meta("worker_prompt_sha256", "f" * 64), "worker_prompt_sha256"),
        ("no-prompt", _drop_meta("worker_prompt_sha256"), "worker_prompt_sha256"),
    ],
)
def test_cli_refuses_a_candidate_on_another_task_or_prompt(dev_run, cli, capsys, name, edit, field):
    """F2-08: the template log (another task) was accepted at exit 0; a candidate must
    run the baseline's task and worker prompt."""
    cand = _derive(dev_run, f"header-{name}", edit)
    assert cli.main(["--baseline", dev_run.path, "--candidate", cand]) == 3
    err = capsys.readouterr().err
    assert "REFUSED" in err and field in err and "differs from the baseline" in err


@pytest.mark.parametrize(
    "name,edit",
    [
        ("other-model", _set_spec(model="mockllm/degraded")),
        ("other-revision", _set_revision("0000000", True)),
        ("package-prefixed-task", lambda log: setattr(log.eval, "task", f"icfr_agent_evals/{log.eval.task.rsplit('/', 1)[-1]}")),
    ],
)
def test_cli_allows_a_baseline_and_candidate_on_different_models_or_revisions(dev_run, cli, capsys, name, edit):
    """D2 form (i): a mock-degraded candidate against the real pilot baseline differs in
    model and revision; the CLI and an eval() call name the same task with and without
    the package prefix. The headers show both runs."""
    cand = _derive(dev_run, f"header-ok-{name}", edit)
    assert cli.main(["--baseline", dev_run.path, "--candidate", cand]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith(f"baseline: {dev_run.path} | task ")
    assert lines[1].startswith(f"candidate 1: {cand} | task ")
    if name == "other-model":
        assert "| model mockllm/model |" in lines[0] and "| model mockllm/degraded |" in lines[1]
    if name == "other-revision":
        assert "| revision 0000000 dirty True |" in lines[1]


POOL_REVISION = _set_revision("1111111", False)


def _pool_edit(*edits):
    def edit(log):
        for one in (POOL_REVISION, *edits):
            one(log)
    return edit


@pytest.mark.parametrize(
    "name,edit,field",
    [
        ("pool-model", _pool_edit(_set_spec(model="mockllm/degraded")), "model"),
        ("pool-revision", _set_revision("0000000", False), "revision"),
        ("pool-dirty", _set_revision("1111111", True), "dirty"),
        ("pool-prompt", _pool_edit(_set_meta("worker_prompt_sha256", "f" * 64)), "worker_prompt_sha256"),
    ],
)
def test_cli_refuses_pooled_candidates_that_differ(dev_run, cli, capsys, name, edit, field):
    """F2-08: section 10(d) pools re-runs of one candidate; a run on another model,
    revision, dirty state or worker prompt is another candidate, not a re-run."""
    first = _derive(dev_run, "header-pool-base", POOL_REVISION)
    other = _derive(dev_run, f"header-{name}", edit)
    for a, b in ((first, other), (other, first)):
        assert cli.main(["--baseline", dev_run.path, "--candidate", a, "--candidate", b]) == 3
        err = capsys.readouterr().err
        assert "REFUSED" in err and field in err
    # a prompt change also differs from the baseline, which is checked first
    assert ("pooled candidate" in err) is (field != "worker_prompt_sha256")


def test_cli_pools_identical_headers_and_prints_one_line_per_run(dev_run, cli, tmp_path):
    out_json = tmp_path / "headers.json"
    code = cli.main(["--baseline", dev_run.path, "--candidate", dev_run.path, "--candidate", dev_run.path,
                     "--json", str(out_json)])
    assert code == 2  # unchanged: the pooled raw counts double (test_cli_pooled_rerun_counts_run_against_the_candidate)
    runs = json.loads(out_json.read_text(encoding="utf-8"))["runs"]
    assert [r["label"] for r in runs] == ["baseline", "candidate 1", "candidate 2"]
    for r in runs:
        assert set(r) == {"label", "path", "task", "model", "revision", "dirty", "worker_prompt_sha256"}
        assert r["task"] == "variance_commentary" and r["model"] == "mockllm/model"
        assert r["worker_prompt_sha256"] == dev_run.log.eval.metadata["worker_prompt_sha256"]


def test_script_refuses_another_task_as_candidate_exit_3(dev_run):
    """Through the script itself, as the review's repro ran it."""
    cand = _derive(dev_run, "header-script-template", _set_spec(task="variance_commentary_template_baseline"))
    proc = subprocess.run([sys.executable, str(SCRIPT), "--baseline", dev_run.path, "--candidate", cand],
                          capture_output=True, text=True)
    assert proc.returncode == 3
    assert "REFUSED" in proc.stderr and "variance_commentary_template_baseline" in proc.stderr
    assert "RESULT" not in proc.stdout


# --- Run configuration (contract C2) in records_from_log and compare_runs.py -----------

def test_records_from_log_reads_a_clean_mock_pilot_log(c2_logs):
    assert len(records_from_log(c2_logs["clean"])) == 8
    assert len(records_from_log(c2_logs["clean"].location)) == 8


@pytest.mark.parametrize("label, field", [("effort_low", "reasoning_effort"), ("temperature", "temperature")])
def test_records_from_log_refuses_an_overridden_log_naming_the_field(c2_logs, label, field):
    with pytest.raises(GateError, match=field):
        records_from_log(c2_logs[label])
    assert len(records_from_log(c2_logs[label], run_config=False)) == 8  # the opt-out, never the default


def test_records_from_log_reads_the_template_log_without_the_check(c2_logs):
    assert len(records_from_log(c2_logs["template"])) == 8


@pytest.mark.parametrize("label, field", [("effort_low", "reasoning_effort"), ("temperature", "temperature")])
@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_cli_refuses_an_overridden_log_with_exit_3(c2_logs, cli, capsys, label, field, side):
    clean, bad = c2_logs["clean"].location, c2_logs[label].location
    base, cand = (clean, bad) if side == "candidate" else (bad, clean)
    assert cli.main(["--baseline", base, "--candidate", cand]) == 3
    out = capsys.readouterr()
    assert out.out == ""
    assert field in out.err and "run config" in out.err and bad in out.err


def test_cli_reads_clean_mock_pilot_logs_and_refuses_the_template_as_candidate(c2_logs, cli, capsys):
    clean, template = c2_logs["clean"].location, c2_logs["template"].location
    assert cli.main(["--baseline", clean, "--candidate", clean]) == 0
    capsys.readouterr()
    assert cli.main(["--baseline", clean, "--candidate", template]) == 3  # F2-08 task match, not C2
    err = capsys.readouterr().err
    assert "task" in err and "run config" not in err


def test_cli_max_tokens_pin_is_passed_to_the_check(c2_logs, cli, capsys):
    clean = c2_logs["clean"].location
    assert cli.main(["--baseline", clean, "--candidate", clean, "--max-tokens-pin", "4000"]) == 3
    assert "max_tokens" in capsys.readouterr().err
