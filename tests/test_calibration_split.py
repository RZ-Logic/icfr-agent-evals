"""Calibration split tests: the 40-case calibration subset of pre-registration section 4.

The subset is 20 standard cases drawn from the eval split's factorial design under the
calibration split's own stream, plus 20 generated hard cases (10 edge-type, 10
adversarial-type). The properties that matter: it builds the same way every time, its
committed file is exactly what the generator produces, it shares no case id, template
instance, wallet, reference, transaction hash or canary with the evaluation, CI or dev
splits, and the task runs on it offline. The evaluation split is built in memory only,
and a failing check reports counts, never its content.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import random
import subprocess
import sys
from collections import Counter
from functools import lru_cache

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from test_generator import MARKER_SUBSTRINGS, MARKER_WORDS, PROBE_ENTITIES
from test_hardcases import REVEALING, SECTIONS, _keys, _sections

from icfr_agent_evals import dataset as dataset_module
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import BANDS, DIRECTIONS, PERIODS, split_seed, truth_of
from icfr_agent_evals.dataset import (
    ASSETS,
    CALIBRATION_CAPS,
    CALIBRATION_EVIDENCE,
    CALIBRATION_HARD_COMPOSITION,
    CALIBRATION_STANDARD_CASES,
    HARD_COMPOSITIONS,
    build_cases,
    build_dataset,
    calibration_cells,
    split_jsonl,
    standard_specs,
)
from icfr_agent_evals.hardcases import BOUNDARY_VALUES_CI, HARD_SPLITS, build_hard_cases
from icfr_agent_evals.policy import materiality_band
from icfr_agent_evals.rubric import WIRED_SCORERS

REPO = pathlib.Path(__file__).resolve().parents[1]
CAL_DIR = REPO / "data" / "calibration"
SCRIPT = REPO / "scripts" / "export_calibration.py"
ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}
FACTORS = ("asset", "band", "period", "direction")

# sha256 of split_jsonl("ci") before the calibration split existed: adding the
# calibration split must leave the CI regression split byte for byte as it was.
CI_SPLIT_JSONL_SHA256 = "2a5e61e0e551f40bad8de7ec303a4658bd853e9fea301d9b481cc4369892267d"


def _load_script():
    key = "calibration_export_script"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def cal_cases():
    return tuple(build_cases("calibration"))


@lru_cache(maxsize=None)
def cal_samples():
    return tuple(build_dataset(split="calibration"))


def _standard():
    return [c.truth for c in cal_cases() if c.truth.case_kind == "standard"]


def _hard():
    return [c for c in cal_cases() if c.truth.case_kind != "standard"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One export into a temporary folder; it builds eval, ci and dev in memory."""
    out = tmp_path_factory.mktemp("calibration")
    summary = _load_script().export(out)
    return out, summary


# --- composition ------------------------------------------------------------------------


def test_split_is_40_cases_with_the_decided_composition():
    ts = [c.truth for c in cal_cases()]
    assert len(ts) == 40 == CALIBRATION_STANDARD_CASES + sum(CALIBRATION_HARD_COMPOSITION.values())
    assert CALIBRATION_HARD_COMPOSITION == {
        "band_boundary": 4, "ambiguous_narrative": 3, "incomplete_data": 3,
        "AC-1": 2, "AC-2": 2, "AC-3": 2, "AC-4": 1, "AC-5": 1, "AC-6": 2,
    }
    assert HARD_COMPOSITIONS["calibration"] is CALIBRATION_HARD_COMPOSITION
    assert Counter(t.category for t in ts if t.case_kind != "standard") == CALIBRATION_HARD_COMPOSITION
    assert Counter(t.case_kind for t in ts) == {"standard": 20, "edge": 10, "adversarial": 10}
    assert all(t.split == "calibration" for t in ts)
    assert all(not t.notes.get("pilot") for t in ts)


def test_standard_cases_balance_every_factor_and_the_evidence_mix():
    std = _standard()
    assert len(std) == CALIBRATION_STANDARD_CASES
    assert set(Counter(t.asset for t in std).values()) <= {2, 3} and {t.asset for t in std} == set(ASSETS)
    for f, levels in (("materiality_band", BANDS), ("period", PERIODS)):
        counts = Counter(getattr(t, f) for t in std)
        assert set(counts) == set(levels) and set(counts.values()) <= {6, 7}, (f, counts)
    assert Counter(t.direction for t in std) == {d: 10 for d in DIRECTIONS}
    assert Counter(t.evidence_state for t in std) == {"fully_explained": 10, "partly_unexplained": 7,
                                                      "unexplained": 3}
    assert Counter((t.evidence_state, t.has_memo_only_link) for t in std) == CALIBRATION_EVIDENCE
    for t in std:
        assert t.materiality_band == materiality_band(t.variance_usd)


def test_calibration_cells_come_from_the_factorial_grid_within_the_caps():
    cells = calibration_cells(random.Random(split_seed("calibration")))
    grid = dataset_module._factorial_grid()
    assert len(cells) == len(set(cells)) == CALIBRATION_STANDARD_CASES
    assert set(cells) <= set(grid)
    for names, cap in CALIBRATION_CAPS.items():
        idx = [FACTORS.index(n) for n in names]
        counts = Counter(tuple(c[i] for i in idx) for c in cells)
        assert max(counts.values()) <= cap, (names, counts.most_common(1))
    # each asset meets each band at most once
    assert len({(c[0], c[1]) for c in cells}) == 20


def test_factorial_grid_is_the_eval_design():
    grid = dataset_module._factorial_grid()
    assert len(grid) == dataset_module.FACTORIAL_CASES == 126
    assert len({c[:4] for c in grid}) == 126
    pool = dataset_module._factorial_cells(random.Random(0))
    assert pool[:126] == grid and len(pool) == dataset_module.STANDARD_CASES


def test_standard_specs_use_the_calibration_stream_and_are_reproducible():
    specs = standard_specs("calibration")
    assert [s.case_id for s in specs] == [s.case_id for s in standard_specs("calibration")]
    assert all(s.case_id.startswith("calibration-std-") and s.split == "calibration" for s in specs)
    cells = calibration_cells(random.Random(split_seed("calibration")))
    assert [(s.asset, s.template_params["band"], s.period, s.template_params["direction"], s.evidence_state,
             s.memo_only) for s in specs] == cells
    other = standard_specs("calibration", seed=20260608)
    assert [s.case_id for s in other] != [s.case_id for s in specs]


def test_calibration_cells_raise_when_no_draw_completes(monkeypatch):
    monkeypatch.setattr(dataset_module, "CALIBRATION_DRAW_ATTEMPTS", 0)
    with pytest.raises(ValueError, match="no balanced draw"):
        calibration_cells(random.Random(1))
    monkeypatch.setattr(dataset_module, "CALIBRATION_DRAW_ATTEMPTS", 5)
    monkeypatch.setitem(dataset_module.CALIBRATION_CAPS, ("direction",), 1)
    with pytest.raises(ValueError, match="no balanced draw"):
        calibration_cells(random.Random(1))


def test_build_is_byte_identical_on_rebuild():
    again = build_cases("calibration")
    assert [c.rendered for c in again] == [c.rendered for c in cal_cases()]
    assert [json.dumps(c.truth.to_dict(), sort_keys=True) for c in again] == [
        json.dumps(c.truth.to_dict(), sort_keys=True) for c in cal_cases()]
    assert split_jsonl("calibration") == split_jsonl("calibration")


def test_samples_are_shuffled_across_kinds():
    kinds = [c.truth.case_kind for c in cal_cases()]
    assert kinds != sorted(kinds) and kinds[:20] != ["standard"] * 20


# --- hard cases -------------------------------------------------------------------------


def test_build_hard_cases_accepts_the_calibration_split():
    assert HARD_SPLITS == ("ci", "dev", "calibration")
    hard = build_hard_cases("calibration", split_seed("calibration"), CALIBRATION_HARD_COMPOSITION)
    assert sorted(c.truth.case_id for c in hard) == sorted(c.truth.case_id for c in _hard())
    assert all(c.truth.case_id.startswith("calibration-hc-") for c in hard)
    with pytest.raises(ValueError, match="ci, dev and calibration"):
        build_hard_cases("eval", split_seed("eval"), CALIBRATION_HARD_COMPOSITION)


def test_band_boundary_takes_each_named_value_once_in_alternating_directions():
    cs = [c for c in _hard() if c.truth.category == "band_boundary"]
    got = Counter((round(abs(c.truth.variance_usd) * 100), c.truth.direction) for c in cs)
    assert got == {(BOUNDARY_VALUES_CI[0], "surplus"): 1, (BOUNDARY_VALUES_CI[1], "shortfall"): 1,
                   (BOUNDARY_VALUES_CI[2], "surplus"): 1, (BOUNDARY_VALUES_CI[3], "shortfall"): 1}
    expected = {2_500_000: "M2", 2_499_999: "M1", 25_000_000: "M2", 25_000_001: "M3"}
    for c in cs:
        assert c.truth.materiality_band == expected[round(abs(c.truth.variance_usd) * 100)]


def test_hard_cases_have_the_standard_shape():
    std = [c for c in cal_cases() if c.truth.case_kind == "standard"]
    assert {tuple(_sections(c.rendered)) for c in std} == {SECTIONS}
    for c in _hard():
        assert tuple(_sections(c.rendered)) == SECTIONS, c.truth.case_id
        assert set(c.fixture) == set(std[0].fixture)


def test_no_rendered_text_reveals_the_construction_or_the_split():
    for c in cal_cases():
        t, text = c.truth, c.rendered
        low = text.lower()
        assert not REVEALING.search(text), (t.case_id, REVEALING.search(text).group(0))
        assert not MARKER_WORDS.search(text), (t.case_id, MARKER_WORDS.search(text).group(0))
        assert not any(s in low for s in MARKER_SUBSTRINGS)
        assert "calibration" not in low and t.case_id not in text
        for probe in PROBE_ENTITIES:
            assert probe.lower() not in low


# --- disjointness -----------------------------------------------------------------------


def test_disjoint_from_eval_ci_and_dev(exported):
    _, summary = exported
    rows = summary["disjointness"]
    assert set(rows) == {"eval", "ci", "dev"}
    for name, row in rows.items():
        assert row["seed"] == split_seed(name)
        shared = {k: v for k, v in row.items() if k.startswith("shared_")}
        assert len(shared) == 7 and set(shared.values()) == {0}, (name, shared)


def test_disjoint_from_dev_and_ci_hard_cases_by_every_rendered_key():
    from test_hardcases import hard

    mine = _keys(_hard())
    for split in ("dev", "ci"):
        theirs = _keys(hard(split))
        for name in mine:
            assert mine[name].isdisjoint(theirs[name]), (split, name, len(mine[name] & theirs[name]))
    assert mine["canary"] and mine["reference"] and mine["wallet"]


def test_wide_keys_cover_wallets_references_transactions_and_canaries():
    keys = _load_script().wide_keys(cal_samples())
    assert set(keys) == {"wallets", "references", "tx_hashes", "canaries"}
    assert all(keys[k] for k in keys)
    assert len(keys["canaries"]) == CALIBRATION_HARD_COMPOSITION["AC-1"]


# --- export script and committed files ----------------------------------------------------


def test_export_writes_the_split_and_its_composition(exported):
    out, summary = exported
    blob = (out / "calibration.jsonl").read_bytes()
    assert blob == split_jsonl("calibration")
    assert summary["file_sha256"] == hashlib.sha256(blob).hexdigest()
    assert summary["calibration_seed"] == split_seed("calibration") == 23260607
    comp = summary["composition"]
    assert comp["n"] == 40 and comp["case_kind"] == {"adversarial": 10, "edge": 10, "standard": 20}
    assert comp["standard"]["n"] == 20 and comp["standard"]["direction"] == {"shortfall": 10, "surplus": 10}
    assert summary["hard_composition"] == dict(sorted(CALIBRATION_HARD_COMPOSITION.items()))
    on_disk = json.loads((out / "COMPOSITION.json").read_text(encoding="utf-8"))
    assert on_disk == summary
    ids = [json.loads(line)["id"] for line in blob.decode("utf-8").splitlines()]
    assert ids == sorted(ids) and len(ids) == 40


def test_export_refuses_a_split_of_the_wrong_size(monkeypatch, tmp_path):
    mod = _load_script()
    monkeypatch.setattr(mod, "build_dataset", lambda split, seed: list(build_dataset(split="dev"))[:3])
    with pytest.raises(ValueError, match="expected 40"):
        mod.export(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_committed_calibration_file_matches_the_generator():
    assert (CAL_DIR / "calibration.jsonl").read_bytes() == split_jsonl("calibration")


def test_committed_composition_matches_the_committed_file():
    summary = json.loads((CAL_DIR / "COMPOSITION.json").read_text(encoding="utf-8"))
    assert summary["file_sha256"] == hashlib.sha256((CAL_DIR / "calibration.jsonl").read_bytes()).hexdigest()
    assert summary["composition"]["n"] == 40
    for other in ("eval", "ci", "dev"):
        shared = {k: v for k, v in summary["disjointness"][other].items() if k.startswith("shared_")}
        assert set(shared.values()) == {0}, (other, shared)


def test_script_help_exits_before_any_build_or_write(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_AUTH_TOKEN", "_BASE_URL"))}
    env.update(PYTHON_DOTENV_DISABLED="1", PYTHONPATH=str(REPO))
    done = subprocess.run([sys.executable, str(SCRIPT), "-h"], cwd=tmp_path, env=env, capture_output=True,
                          text=True, check=False, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout and "--out-dir" in done.stdout
    assert list(tmp_path.iterdir()) == []


def test_parse_args_defaults_to_the_data_folder(tmp_path):
    mod = _load_script()
    assert mod.parse_args([]).out_dir == CAL_DIR
    assert mod.parse_args(["--out-dir", str(tmp_path)]).out_dir == tmp_path


def test_main_writes_into_the_given_folder(monkeypatch, tmp_path, capsys):
    mod = _load_script()
    seen = []
    monkeypatch.setattr(mod, "export", lambda out: seen.append(out) or {"n": 40})
    assert mod.main(["--out-dir", str(tmp_path)]) == 0
    assert seen == [tmp_path] and json.loads(capsys.readouterr().out) == {"n": 40}


# --- other splits stay as they were ---------------------------------------------------------


def test_ci_split_builds_byte_for_byte_as_before():
    assert hashlib.sha256(split_jsonl("ci")).hexdigest() == CI_SPLIT_JSONL_SHA256


# --- task --------------------------------------------------------------------------------


def _mock_worker():
    """mockllm with usage set, so Inspect never reaches for a tokenizer download."""

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        out = ModelOutput.from_content(model="mockllm/model", content="The variance is under review.")
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return get_model("mockllm/model", custom_outputs=outputs)


def test_task_builds_and_runs_the_calibration_split_on_mockllm(tmp_path):
    t = task_module.variance_commentary(split="calibration", epochs=1)
    assert t.metadata["split"] == "calibration" and t.metadata["split_seed"] == 23260607
    assert t.metadata["dataset_sha256"] == t.metadata["split_dataset_sha256"]
    assert t.dataset.name == "variance-commentary-calibration" and len(t.dataset) == 40
    log = inspect_eval(t, model=_mock_worker(), model_roles=ROLES, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    assert log.eval.task_args.get("split") == "calibration"
    assert len(log.samples) == 40
    assert {str(s.id) for s in log.samples} == {str(s.id) for s in cal_samples()}
    for s in log.samples:
        assert set(s.scores) == set(WIRED_SCORERS)
