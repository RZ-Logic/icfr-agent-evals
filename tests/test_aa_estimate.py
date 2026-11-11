"""The A/A estimate command and the recorded ladder rung (section 10(c) and (d), D-E5).

scripts/aa_estimate.py reads the 8-epoch A/A log on the CI regression split, prints
the null false-block rate and the review volume under R1, R2 and R3 counted by
route key, and the rung select_ladder takes. --record writes the choice once and
refuses to switch it (pre-registration section 10(c): chosen once, never
switched). regression.resolve_ladder gives compare_runs.py the recorded rung.

No model call: the log-level tests use one mockllm run of two dev cases at 8
epochs, then a copy whose header and sample ids are relabelled to the CI split
(the CI split is never built here). Every ladder record is written under tmp_path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import good_output
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalRevision, read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.regression import (
    AA_EPOCHS,
    LADDER_RECORD_RELPATH,
    EpochRecord,
    aa_estimate,
    aa_records_from_log,
    ladder_record,
    read_ladder_record,
    records_from_log,
    resolve_ladder,
    validate_ladder_record,
    write_ladder_record,
)
from icfr_agent_evals.reliability import PREREGISTERED_K, GateError

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "aa_estimate.py"
FULL = {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}
CODE_KEY = "code_review: amount_role_unresolved"
SHA = "a" * 64
COMMIT = "1" * 40


def _aa_log(**over):
    base = {"file": "aa.eval", "sha256": SHA, "eval_id": "E1", "run_commit": COMMIT, "split": "ci",
            "epochs": AA_EPOCHS, "samples": 40, "model": "mockllm/model"}
    return {**base, **over}


def run(n: int, fires: dict | None = None, epochs: int = AA_EPOCHS) -> list[EpochRecord]:
    """n clean samples x epochs; fires maps (sample index, trigger) -> the epochs it fires on."""
    out = []
    for i in range(n):
        for e in range(1, epochs + 1):
            codes, routes = [], []
            for (j, trigger), on in (fires or {}).items():
                if j == i and e in on:
                    (codes if trigger.startswith("FM-") else routes).append(trigger)
            out.append(EpochRecord(f"ci-{i:03d}", e, dict(FULL), not codes and not routes, tuple(codes), tuple(routes)))
    return out


def _fires_for(rung: str) -> dict:
    """Synthetic 8-epoch tables whose A/A estimate takes each rung."""
    if rung == "R1":
        return {(0, "FM-016"): {5}}
    fires = {}
    for i in range(40):
        fires[(i, "FM-001")] = {5}
        if rung == "R3":
            fires[(i, CODE_KEY)] = {5, 6}
    return fires


@pytest.fixture(scope="module")
def estimates():
    return {rung: aa_estimate(run(40, _fires_for(rung))) for rung in ("R1", "R2", "R3")}


# --- Records level: the estimate and the record -------------------------------------------

def test_synthetic_tables_take_each_rung(estimates):
    assert {rung: est.ladder for rung, est in estimates.items()} == {"R1": "R1", "R2": "R2", "R3": "R3"}
    # one FM-016 on epoch 5 of one case: it lands in the candidate half on 20 of 35 splits
    assert estimates["R1"].mean_volume == {"R1": pytest.approx(20 / 35), "R2": 0.0, "R3": 0.0}


def test_ladder_record_holds_rung_volumes_log_and_ruling(estimates):
    rec = ladder_record(estimates["R2"], _aa_log(), "2026-11-10")
    assert rec["rung"] == "R2" and rec["ruling"] == "D-E5" and rec["splits"] == 35 and rec["cap"] == 10
    assert rec["mean_volume"] == dict(estimates["R2"].mean_volume)
    assert set(rec["mean_volume"]) == {"R1", "R2", "R3"}
    assert rec["aa_log"] == _aa_log() and rec["recorded"] == "2026-11-10"
    # 40 cases failing on epoch 5 block content-pass^4 whenever epoch 5 is a candidate epoch
    assert rec["false_block_rate"] == pytest.approx(20 / 35) and rec["false_block_shortfall"] is True
    assert ladder_record(estimates["R1"], _aa_log(), "2026-11-10")["false_block_rate"] == 0.0


@pytest.mark.parametrize("edit, match", [
    (lambda r: r.update(ruling="D-E6"), "ruling"),
    (lambda r: r.update(rung="R2"), "does not follow from its volumes"),
    (lambda r: r.update(rung="R4"), "rung"),
    (lambda r: r.update(cap=12), "cap"),
    (lambda r: r["aa_log"].update(sha256="xyz"), "sha256"),
    (lambda r: r["aa_log"].update(split="dev"), "split"),
    (lambda r: r["aa_log"].update(epochs=4), "epochs"),
    (lambda r: r["aa_log"].update(run_commit=None), "run_commit"),
    (lambda r: r.update(recorded="10 Nov 2026"), "YYYY-MM-DD"),
    (lambda r: r.pop("mean_volume"), "missing"),
])
def test_validate_refuses_a_bad_record(estimates, edit, match):
    rec = json.loads(json.dumps(ladder_record(estimates["R1"], _aa_log(), "2026-11-10")))
    edit(rec)
    with pytest.raises(GateError, match=match):
        validate_ladder_record(rec, "test record")


def test_write_once_then_unchanged_then_refuse_a_switch(estimates, tmp_path):
    path = tmp_path / "evidence" / "ladder.json"
    first = ladder_record(estimates["R2"], _aa_log(), "2026-11-10")
    assert write_ladder_record(path, first) == "written"
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert read_ladder_record(path) == first
    # the same rung from the same log, on a later date: left as it is
    assert write_ladder_record(path, ladder_record(estimates["R2"], _aa_log(), "2026-11-12")) == "unchanged"
    assert path.read_bytes() == raw
    # another rung: refused, file untouched
    for rung in ("R1", "R3"):
        with pytest.raises(GateError, match="chosen once and never switched"):
            write_ladder_record(path, ladder_record(estimates[rung], _aa_log(), "2026-11-12"))
        assert path.read_bytes() == raw
    # the same rung from another A/A log: refused too
    with pytest.raises(GateError, match="another log"):
        write_ladder_record(path, ladder_record(estimates["R2"], _aa_log(sha256="b" * 64), "2026-11-12"))
    assert path.read_bytes() == raw


def test_read_ladder_record_absent_and_unreadable(tmp_path):
    assert read_ladder_record(tmp_path / "none.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GateError, match="cannot read"):
        read_ladder_record(bad)


def test_resolve_ladder_without_a_record_keeps_r1_and_says_so(tmp_path):
    path = tmp_path / "ladder.json"
    rung, recorded, note = resolve_ladder(None, path)
    assert (rung, recorded) == ("R1", False)
    assert note.startswith("no ladder rung is recorded yet") and "default R1" in note
    rung, recorded, note = resolve_ladder("R3", path)
    assert (rung, recorded) == ("R3", False) and "--ladder R3 as given" in note


def test_resolve_ladder_reads_the_record_and_refuses_a_disagreeing_rung(estimates, tmp_path):
    path = tmp_path / "ladder.json"
    write_ladder_record(path, ladder_record(estimates["R2"], _aa_log(), "2026-11-10"))
    assert resolve_ladder(None, path)[:2] == ("R2", True)
    rung, recorded, note = resolve_ladder("R2", path)
    assert (rung, recorded) == ("R2", True) and "ladder R2 read from" in note and "aa.eval" in note
    for other in ("R1", "R3"):
        with pytest.raises(GateError, match="disagrees with rung R2"):
            resolve_ladder(other, path)


def test_resolve_ladder_refuses_an_unknown_rung(tmp_path):
    with pytest.raises(ValueError):
        resolve_ladder("R4", tmp_path / "ladder.json")


def test_records_from_log_keeps_four_epochs_by_default():
    import inspect as pyinspect

    assert pyinspect.signature(records_from_log).parameters["epochs"].default == PREREGISTERED_K == 4


def test_default_record_path_is_under_docs_evidence(script):
    assert LADDER_RECORD_RELPATH == "docs/evidence/regression/ladder.json"
    assert script.DEFAULT_RECORD == REPO / "docs" / "evidence" / "regression" / "ladder.json"


# --- Log level: one mockllm run at 8 epochs, relabelled to the CI split -------------------

@pytest.fixture(scope="module")
def aa_run(tmp_path_factory):
    """Two standard dev cases at 8 epochs, every answer correct, so no epoch fires."""
    from icfr_agent_evals.task import variance_commentary

    task = variance_commentary(split="dev", epochs=AA_EPOCHS)
    truths = {s.id: truth_of(s.metadata) for s in task.dataset}
    std = [sid for sid, t in truths.items() if t.case_kind == "standard" and not t.has_memo_only_link][:2]
    key_of = {truths[sid].scope_wallets[0]: sid for sid in std}
    assert len(key_of) == 2

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        sid = next(s for w, s in key_of.items() if w in blob)
        out = ModelOutput.from_content(model="mockllm/model", content=json.dumps(good_output(truths[sid])))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    task.dataset = task.dataset.filter(lambda s: s.id in set(std))
    log_dir = tmp_path_factory.mktemp("aa-run")
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=mock_outputs),
                       model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
                       log_dir=str(log_dir), display="none")[0]
    assert log.status == "success", log.error
    return SimpleNamespace(path=log.location, dir=log_dir, ids=sorted(std))


def _derive(aa_run, name, *edits):
    """A copy relabelled to the CI split, on a clean commit, then each edit applied."""
    log = read_eval_log(aa_run.path)
    log.eval.task_args = {**(log.eval.task_args or {}), "split": "ci"}
    log.eval.metadata = {**(log.eval.metadata or {}), "split": "ci"}
    for s in log.samples:
        s.id = "ci-" + str(s.id).removeprefix("dev-")
    if getattr(log.eval.dataset, "sample_ids", None) is not None:
        log.eval.dataset.sample_ids = ["ci-" + str(i).removeprefix("dev-") for i in log.eval.dataset.sample_ids]
    log.eval.revision = EvalRevision(type="git", origin="local", commit=COMMIT, dirty=False)
    for edit in edits:
        edit(log)
    path = aa_run.dir / f"{name}.eval"
    write_eval_log(log, str(path))
    return str(path)


def _fm016_on_epoch_5(log):
    first = sorted({str(s.id) for s in log.samples})[0]
    s = next(s for s in log.samples if str(s.id) == first and s.epoch == 5)
    s.scores["correctness_anchor"].metadata["failure_modes"] = ["FM-016"]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("aa_estimate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aa_estimate_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ci_log(aa_run):
    return _derive(aa_run, "aa-ci", _fm016_on_epoch_5)


def test_aa_records_read_eight_epochs_on_the_ci_split(ci_log):
    recs = aa_records_from_log(ci_log)
    assert len(recs) == 2 * AA_EPOCHS and {r.epoch for r in recs} == set(range(1, 9))
    fired = [(r.sample_id, r.epoch) for r in recs if r.failure_modes]
    assert len(fired) == 1 and fired[0][1] == 5
    # the gates' own reader still pins 4 epochs
    with pytest.raises(GateError, match="epochs"):
        records_from_log(ci_log)


def test_script_prints_the_estimate_and_the_rung(script, ci_log, capsys):
    assert script.main([ci_log]) == 0
    out = capsys.readouterr().out
    assert "A/A estimate: 35 splits" in out and "false-block rate 0.0000" in out
    assert "counted by route key" in out
    assert "R1: 0.571" in out and "R2: 0.000" in out and "R3: 0.000" in out
    assert "chosen rung: R1" in out and "not recorded" in out
    assert f"run commit {COMMIT}" in out and "split ci" in out and "epochs 8" in out
    sha = hashlib.sha256(Path(ci_log).read_bytes()).hexdigest()
    assert f"sha256 {sha}" in out


def test_script_record_writes_once_and_refuses_a_switch(script, ci_log, aa_run, tmp_path, capsys):
    path = tmp_path / "docs" / "evidence" / "regression" / "ladder.json"
    assert script.main([ci_log, "--record", str(path), "--date", "2026-11-10"]) == 0
    assert f"rung R1 written to {path}" in capsys.readouterr().out
    rec = json.loads(path.read_text(encoding="utf-8"))
    log = read_eval_log(ci_log, header_only=True)
    assert rec["rung"] == "R1" and rec["ruling"] == "D-E5" and rec["recorded"] == "2026-11-10"
    assert rec["mean_volume"] == {"R1": pytest.approx(20 / 35), "R2": 0.0, "R3": 0.0}
    assert rec["aa_log"]["file"] == Path(ci_log).name
    assert rec["aa_log"]["sha256"] == hashlib.sha256(Path(ci_log).read_bytes()).hexdigest()
    assert rec["aa_log"]["eval_id"] == log.eval.eval_id and rec["aa_log"]["run_commit"] == COMMIT
    assert rec["aa_log"]["samples"] == 2 and rec["aa_log"]["split"] == "ci" and rec["aa_log"]["epochs"] == 8
    raw = path.read_bytes()

    assert script.main([ci_log, "--record", str(path), "--date", "2026-11-11"]) == 0
    assert "left unchanged" in capsys.readouterr().out
    assert path.read_bytes() == raw

    # an earlier recording of another rung (hand-edited here) is never switched
    switched = dict(rec, rung="R2", mean_volume={"R1": 11.0, "R2": 3.0, "R3": 1.0})
    path.write_text(json.dumps(switched, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    before = path.read_bytes()
    assert script.main([ci_log, "--record", str(path)]) == 3
    assert "never switched" in capsys.readouterr().err
    assert path.read_bytes() == before

    # the same rung from another A/A log is refused as well
    path.write_bytes(raw)
    other = _derive(aa_run, "aa-ci-other", _fm016_on_epoch_5, lambda log: setattr(log.eval, "eval_id", "other-id"))
    assert script.main([other, "--record", str(path)]) == 3
    assert "another log" in capsys.readouterr().err
    assert path.read_bytes() == raw


def test_script_record_refuses_a_dirty_or_commitless_run(script, aa_run, tmp_path, capsys):
    dirty = _derive(aa_run, "aa-dirty", lambda log: setattr(log.eval.revision, "dirty", True))
    path = tmp_path / "ladder.json"
    assert script.main([dirty, "--record", str(path)]) == 3
    assert "dirty" in capsys.readouterr().err and not path.exists()
    assert script.main([dirty]) == 0  # printing alone is allowed
    capsys.readouterr()
    bare = _derive(aa_run, "aa-no-revision", lambda log: setattr(log.eval, "revision", None))
    assert script.main([bare, "--record", str(path)]) == 3
    assert "commit" in capsys.readouterr().err and not path.exists()


def _set_split(value):
    def edit(log):
        log.eval.task_args = {**log.eval.task_args, "split": value}
        log.eval.metadata = {**log.eval.metadata, "split": value}
    return edit


def _cache_read(log):
    s = log.samples[0]
    ev = next(e for e in s.events if getattr(e, "event", None) == "model")
    ev.cache = "read"


def _epochs(n):
    def edit(log):
        log.eval.config.epochs = n
    return edit


@pytest.mark.parametrize("name, edit, match", [
    ("aa-eval", _set_split("eval"), "evaluation"),
    ("aa-calibration", _set_split("calibration"), "CI regression split"),
    ("aa-subset", lambda log: log.eval.metadata.update(subset="pilot"), "subset"),
    ("aa-cached", _cache_read, "caching off"),
    ("aa-four-epochs", _epochs(4), "epochs"),
    ("aa-limit", lambda log: setattr(log.eval.config, "limit", 1), "limit"),
])
def test_script_refuses_logs_off_the_aa_design(script, aa_run, tmp_path, capsys, name, edit, match):
    bad = _derive(aa_run, name, edit)
    path = tmp_path / "ladder.json"
    assert script.main([bad, "--record", str(path)]) == 3
    assert match in capsys.readouterr().err and not path.exists()


def test_script_refuses_the_dev_split_log_as_run(script, aa_run, capsys):
    assert script.main([aa_run.path]) == 3
    assert "CI regression split" in capsys.readouterr().err


def test_script_refuses_a_missing_file_and_a_bad_date(script, ci_log, tmp_path, capsys):
    assert script.main([str(tmp_path / "absent.eval")]) == 3
    with pytest.raises(SystemExit) as stop:
        script.main([ci_log, "--record", str(tmp_path / "x.json"), "--date", "Nov 10"])
    assert stop.value.code == 3
    assert not (tmp_path / "x.json").exists()


def test_script_docstring_names_the_design():
    doc = SCRIPT.read_text(encoding="utf-8")
    for phrase in ("8-epoch", "CI regression split", "caching", "35", "R1, R2 and R3", "counted by route key",
                   "D-E5", "never switched", "docs/evidence/regression/ladder.json", "no model call"):
        assert phrase in doc, phrase
