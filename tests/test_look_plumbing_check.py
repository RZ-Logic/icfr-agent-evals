"""The parse-only plumbing check of the perturbed prompt settings (scripts/look_plumbing_check.py).

Keyless: the only model is Inspect's mockllm with canned replies; no grader call reaches a
provider. The inputs are the committed judge-development records (dev split); no
calibration record is read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from icfr_agent_evals import grader
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import judge_trust as jt

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS
SECRET = "SECRET-REASON-TEXT"


def _load():
    spec = importlib.util.spec_from_file_location("look_plumbing_check", REPO / "scripts" / "look_plumbing_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["look_plumbing_check"] = mod
    spec.loader.exec_module(mod)
    return mod


pc = _load()


def roles(bad_setting: str | None = None):
    """A mockllm grader that answers every prompt in its own labels, with a reason text the
    check must never print; on bad_setting it answers in the default labels (not offered)."""
    from inspect_ai.model import ModelOutput, get_model

    def out(input, tools, tool_choice, config):
        text = "\n".join(str(m.content) for m in input)
        relabeled = "- Level D:" in text
        lab = gp.RELABEL_DCBA if relabeled and bad_setting != jt.LABELS_RELABELED else gp.DEFAULT_LABELS
        if bad_setting == jt.LABELS_RELABELED and relabeled:
            lab = {0: "7", 1: "8", 2: "9", 3: "6"}
        body = {G: {"reason": SECRET, "level": lab[2], "low_confidence": False},
                C: {"reason": SECRET, "level": lab[1], "low_confidence": True}}
        return ModelOutput.from_content("mockllm/model", json.dumps(body))

    return {"grader": get_model("mockllm/model", custom_outputs=out, memoize=False)}


def _evidence_files() -> list[str]:
    return sorted(str(p.relative_to(REPO)) for p in (REPO / "docs").rglob("*") if p.is_file())


def test_the_four_settings_are_the_looks_perturbed_passes():
    specs = pc.settings()
    assert [s.name for s in specs] == list(pc.al.BIAS_NAMES)
    by = {s.name: s.settings for s in specs}
    assert by[jt.OPTION_ORDER_REVERSED].option_order == (3, 2, 1, 0)
    assert by[jt.OPTION_ORDER_PERMUTED].option_order == (3, 1, 2, 0)
    assert by[jt.LABELS_RELABELED].label_map() == gp.RELABEL_DCBA
    assert by[jt.DIMENSION_ORDER_PERMUTED].dimension_order == (C, G)


def test_dry_run_makes_no_call(monkeypatch, capsys):
    monkeypatch.setattr(pc, "run_check", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert pc.main(["--step", "1"]) == pc.EXIT_OK
    out = capsys.readouterr().out
    assert "4 settings x 3 judge-development outputs = 12 calls, parse only" in out
    assert grader.grader_pin(1) in out and "dry run: no call made" in out


def test_every_reply_parses_and_no_grade_is_printed_or_written(tmp_path, capsys):
    before = _evidence_files()
    code = pc.main(["--step", "1", "--live", "--log-dir", str(tmp_path / "logs")], model_roles=roles())
    out = capsys.readouterr().out
    assert code == pc.EXIT_OK, out
    assert "12 of 12 replies parse" in out and out.count(": parses") == 12
    assert SECRET not in out and '"level"' not in out
    assert _evidence_files() == before
    logs = list((tmp_path / "logs").glob("*.eval"))
    assert len(logs) == 1
    from inspect_ai.log import read_eval_log

    meta = read_eval_log(str(logs[0]), header_only=True).eval.metadata
    assert meta["split"] == "dev" and meta["parse_only"] is True and meta["permutation_seed"] == 23260607


def test_a_reply_that_does_not_parse_is_reported(tmp_path, capsys):
    code = pc.main(["--step", "2", "--live", "--log-dir", str(tmp_path / "logs")],
                   model_roles=roles(bad_setting=jt.LABELS_RELABELED))
    out = capsys.readouterr().out
    assert code == pc.EXIT_CHECK
    assert "9 of 12 replies parse" in out and out.count("labels_relabeled") >= 3
    assert "does not parse" in out and SECRET not in out


def test_calibration_inputs_are_refused(tmp_path, capsys):
    assert pc.main(["--step", "1", "--records-dir", str(REPO / "data" / "calibration" / "records")]) == pc.EXIT_REFUSED
    assert "calibration inputs are refused" in capsys.readouterr().err
    records = sorted((REPO / "data" / "dev" / "judge-dev").glob("*.json"))
    first = json.loads(records[0].read_text(encoding="utf-8"))["sample_id"]
    cases = tmp_path / "calibration.jsonl"
    cases.write_text(json.dumps({"id": first, "input": "", "metadata": {"split": "calibration"}}) + "\n",
                     encoding="utf-8")
    assert pc.main(["--step", "1"], cases_file=cases) == pc.EXIT_REFUSED
    assert "is a calibration case" in capsys.readouterr().err


def test_records_must_be_three_named_dev_records(capsys):
    names = [p.stem for p in sorted((REPO / "data" / "dev" / "judge-dev").glob("*.json"))]
    assert pc.main(["--step", "1", "--record", names[0], "--record", names[1]]) == pc.EXIT_REFUSED
    assert pc.main(["--step", "1", "--record", names[0], "--record", names[0], "--record", names[1]]) == pc.EXIT_REFUSED
    assert pc.main(["--step", "1", "--record", names[4], "--record", names[5], "--record", names[6]]) == pc.EXIT_OK
    out = capsys.readouterr().out
    assert f"outputs: {names[4]}, {names[5]}, {names[6]}" in out


def test_live_needs_a_log_dir_outside_the_repo():
    assert pc.main(["--step", "1", "--live"]) == pc.EXIT_ARGS
    assert pc.main(["--step", "1", "--live", "--log-dir", str(REPO / "logs")]) == pc.EXIT_ARGS


def test_a_failed_log_check_is_reported(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(pc, "run_check", lambda *a, **k: SimpleNamespace(status="error", samples=[]))
    assert pc.main(["--step", "1", "--live", "--log-dir", str(tmp_path)]) == pc.EXIT_CHECK
    assert "FAIL: the log check" in capsys.readouterr().err
