"""The acceptance look on the calibration subset (scripts/acceptance_look.py), pre-registration
section 7.

Keyless: the only model is Inspect's mockllm with canned replies; no grader call reaches a
provider. The calibration cases and records are made-up stand-ins built from dev-split rows
in a temporary folder; nothing from the evaluation split is read.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals import grader, rubric, roster
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import judge_trust as jt

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS
N = 40
RESAMPLES = 200


def _load():
    spec = importlib.util.spec_from_file_location("acceptance_look", REPO / "scripts" / "acceptance_look.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acceptance_look"] = mod
    spec.loader.exec_module(mod)
    return mod


al = _load()


# --- stand-ins ---------------------------------------------------------------------------------

def author(i: int) -> tuple[int, int]:
    """The stand-in author grades: every level 10 times per sub-dimension."""
    return i % 4, (i // 4) % 4


def _write_inputs(root: Path, *, n: int = N, grades=author, split: str = "calibration") -> tuple[Path, Path]:
    dev_rows = [json.loads(line) for line in (REPO / "data" / "dev" / "dev.jsonl").read_text(encoding="utf-8")
                .splitlines() if line.strip()]
    outputs = [json.loads(p.read_text(encoding="utf-8"))["worker_output"]
               for p in sorted((REPO / "data" / "dev" / "judge-dev").glob("*.json"))]
    cases_file = root / "calibration.jsonl"
    records_dir = root / "records"
    records_dir.mkdir(parents=True)
    lines = []
    sources = ("natural", "seeded_omission", "seeded_addition")
    workers = ("anthropic/claude-sonnet-5-5", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-5-5")
    for i in range(n):
        row = dev_rows[i % len(dev_rows)]
        cid = f"cal-{i:02d}"
        meta = {**row["metadata"], "split": split}
        lines.append(json.dumps({"id": cid, "input": row["input"], "metadata": meta}))
        out = json.loads(outputs[i % len(outputs)])
        out["narrative"] = f"Stand-in narrative REC{i:02d}. " + str(out["narrative"])
        g, c = grades(i)
        rec = {"sample_id": cid, "epoch": 1, "log_eval_id": "standin", "worker_prompt_sha256": "0" * 64,
               "worker_model": workers[i % 3], "worker_output": json.dumps(out), "source": sources[i % 3],
               "seed_note": None, "stratum": {G: g, C: c}, "grades": {G: g, C: c},
               "anchors_sha256": rubric.llm_anchor_sha256(), "graded_by": "XX", "graded_on": "2026-11-20"}
        (records_dir / f"{cid}-e1.json").write_text(json.dumps(rec), encoding="utf-8")
    cases_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cases_file, records_dir


def _write_judge_dev(folder: Path, **kw) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    body = {"round": 0, "offline": False, "stop_criteria_met": True, "grader_prompt_sha256": gp.prompt_sha256(), **kw}
    (folder / f"round-{body['round']}.json").write_text(json.dumps(body), encoding="utf-8")
    return folder


@pytest.fixture
def env(tmp_path):
    cases_file, records_dir = _write_inputs(tmp_path / "data")
    return SimpleNamespace(cases_file=cases_file, records_dir=records_dir, out=tmp_path / "evidence",
                           judge_dev=_write_judge_dev(tmp_path / "judge-dev"), logs=tmp_path / "logs")


def _kw(env, **extra):
    return dict(out_dir=env.out, records_dir=env.records_dir, cases_file=env.cases_file,
                judge_dev_dir=env.judge_dev, **extra)


def reply(g, c, *, labels=None, low=False):
    lab = labels or gp.DEFAULT_LABELS
    return json.dumps({G: {"reason": "r", "level": lab[g], "low_confidence": low},
                       C: {"reason": "r", "level": lab[c], "low_confidence": low}})


def mock_roles(scores):
    """Per pass, a mockllm grader whose reply to each record comes from scores(pass, i) ->
    (grounding, coverage) or a raw reply string. Relabeled prompts are answered in letters."""
    from inspect_ai.model import ModelOutput, get_model

    def roles(name):
        def out(input, tools, tool_choice, config):
            text = "\n".join(str(m.content) for m in input)
            i = int(re.search(r"REC(\d\d)", text).group(1))
            labels = gp.RELABEL_DCBA if "- Level D:" in text else None
            got = scores(name, i)
            content = got if isinstance(got, str) else reply(*got[:2], labels=labels,
                                                              low=len(got) > 2 and got[2])
            return ModelOutput.from_content("mockllm/model", content)

        return {"grader": get_model("mockllm/model", custom_outputs=out, memoize=False)}

    return roles


def near(name, i):
    """Pass 1 agrees with the author except four balanced 1-point misses; pass 2 moves one more."""
    g, c = author(i)
    if i in (1, 9):
        g += 1
    if i in (2, 10):
        g -= 1
    if i in (5, 6):
        c = c + 1 if c < 3 else c - 1
    if name == "pass_2" and i == 20:
        g = 1 if g != 1 else 2
    low = name == "pass_1" and i in (3, 4)
    return g, c, low


def lenient(name, i):
    g, c = author(i)
    return min(3, g + 1), min(3, c + 1)


def shifted(name, i):
    if name == jt.LABELS_RELABELED:
        return lenient(name, i)
    return near(name, i)


def identical(name, i):
    return 2, 2


def _run(env, scores, step=1, seed=11, **kw):
    argv = ["--step", str(step), "--permutation-seed", str(seed), "--resamples", str(RESAMPLES), "--live",
            "--log-dir", str(env.logs)]
    code = al.main(argv, model_roles=mock_roles(scores), **_kw(env, **kw))
    path = env.out / f"acceptance-look-step-{step}.json"
    return code, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)


# --- full offline runs ---------------------------------------------------------------------------

def test_agreement_passes_every_gate(env, capsys):
    code, res = _run(env, near)
    assert code == al.EXIT_OK, capsys.readouterr().err
    v = res["verdict"]
    assert v == {**v, "validity": "PASS", "test_retest": "PASS", "bias_audit": "PASS", "overall": "PASS"}
    assert [p["name"] for p in res["passes"]] == list(al.PASS_NAMES) and res["records"] == N
    assert res["gates"]["validity"]["gating_lower_limit"] >= 0.60
    assert res["gates"]["test_retest"]["alpha"]["bootstrap_seed"] == al.DEFAULT_BOOTSTRAP_SEED
    assert res["low_confidence"]["pass_1_outputs_flagged"] == 2
    assert res["low_confidence"]["per_pass"]["pass_1"] == {G: 2, C: 2}
    assert res["score_distribution"]["author"][G] == {"0": 10, "1": 10, "2": 10, "3": 10}
    assert sum(res["score_distribution"]["by_worker_model"].values()) == N
    assert set(res["seed_types"]["per_source"]) == {"natural", "seeded_omission", "seeded_addition"}
    assert all(p["grader_config"] == {"response_caching": False, "reasoning_effort": "high"} for p in res["passes"])
    assert res["passes"][0]["log_check"] == "offline_log_check passed; 40 grader calls, one per output"
    rel = next(p for p in res["passes"] if p["name"] == jt.LABELS_RELABELED)
    assert rel["settings"]["labels"] == {"0": "D", "1": "C", "2": "B", "3": "A"}
    out = capsys.readouterr().out
    assert "(limit 0.60)" in out and "(limit 0.80)" in out and "excess limit 0.10" in out
    assert "(limit 0.25 in magnitude)" in out and "OVERALL PASS" in out


def test_every_pass_records_the_permutation_seed_in_task_metadata(env):
    from inspect_ai.log import read_eval_log

    code, _ = _run(env, near, seed=42)
    assert code == al.EXIT_OK
    metas = [read_eval_log(str(p), header_only=True).eval.metadata for p in sorted(env.logs.glob("*.eval"))]
    assert len(metas) == 7 and {m["pass"] for m in metas} == set(al.PASS_NAMES)
    assert all(m["permutation_seed"] == 42 and m["split"] == "calibration" and m["grader_step"] == 1 for m in metas)
    perm = next(m for m in metas if m["pass"] == jt.OPTION_ORDER_PERMUTED)
    assert perm["prompt_settings"]["option_order"] == jt.permuted_options([0, 1, 2, 3], 42)["options"]


def test_a_lenient_grader_fails_leniency(env, capsys):
    code, res = _run(env, lenient)
    assert code == al.EXIT_GATE_FAIL
    bias = res["gates"]["bias_audit"]
    assert res["verdict"]["bias_audit"] == "FAIL" and res["verdict"]["overall"] == "FAIL"
    assert all(bias["per_subdimension"][d]["leniency"]["failed"] for d in gp.DIMENSIONS)
    assert bias["per_subdimension"][G]["leniency"]["mean"] == pytest.approx(0.75)
    assert "intra-rater check first" in res["verdict"]["next"]
    assert "OVERALL FAIL" in capsys.readouterr().out


def test_a_perturbed_pass_that_shifts_fails_the_bias_audit(env):
    code, res = _run(env, shifted)
    assert code == al.EXIT_GATE_FAIL
    v = res["verdict"]
    assert v["validity"] == "PASS" and v["test_retest"] == "PASS" and v["bias_audit"] == "FAIL"
    per = res["gates"]["bias_audit"]["per_subdimension"][G]
    assert per["passes"][jt.LABELS_RELABELED]["failed"] is True
    assert not per["passes"][jt.OPTION_ORDER_REVERSED]["failed"] and not per["leniency"]["failed"]


def test_identical_replies_leave_alpha_undefined_and_fail(env):
    code, res = _run(env, identical)
    assert code == al.EXIT_GATE_FAIL
    alpha = res["gates"]["test_retest"]["alpha"]
    assert res["verdict"]["test_retest"] == "FAIL" and alpha["passed"] is False
    assert alpha["undefined"] == list(gp.DIMENSIONS)
    assert all(alpha["per_subdimension"][d]["alpha"] is None for d in gp.DIMENSIONS)
    assert alpha["per_subdimension"][G]["raw_agreement"] == 1.0


def test_an_unparsed_reply_fails_the_look_and_computes_no_gate(env):
    code, res = _run(env, lambda name, i: "no json" if (name, i) == ("pass_2", 7) else near(name, i))
    assert code == al.EXIT_GATE_FAIL
    assert res["gates"] is None and res["unparsed_count"] == 1 and res["verdict"]["overall"] == "FAIL"
    assert res["passes"][1]["unparsed"][0]["record"] == "cal-07-e1"


def test_a_failed_log_check_writes_nothing_and_names_the_pass(env, capsys):
    seen = []

    def check(log, step):
        name = log.eval.metadata["pass"]
        seen.append(name)
        if name == jt.LABELS_RELABELED:
            raise roster.RunConfigError("run config: off the pin")
        return []

    code, res = _run(env, near, log_check=check)
    assert code == al.EXIT_CHECK and res is None
    assert not env.out.exists() or not list(env.out.iterdir())
    assert "pass 6 of 7 (labels_relabeled)" in capsys.readouterr().err
    assert seen == list(al.PASS_NAMES[:6])


def test_live_runs_check_every_log_with_check_grader_log(env, monkeypatch, capsys):
    calls = []
    real = grader.check_grader_log

    def fake_check(log, step):
        calls.append(step)
        return real(log, step)

    monkeypatch.setattr(al.grader, "check_grader_log", fake_check)
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: SimpleNamespace(status="success", samples=[]))
    code = al.main(["--step", "1", "--permutation-seed", "1", "--live", "--log-dir", str(env.logs)], **_kw(env))
    assert code == al.EXIT_CHECK and calls == [1]
    assert "holds no grader ModelEvent" in capsys.readouterr().err
    assert not env.out.exists()


# --- one look per grader ----------------------------------------------------------------------

def test_one_look_per_grader(env, monkeypatch, capsys):
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    env.out.mkdir(parents=True)
    (env.out / "acceptance-look-step-1.json").write_text(json.dumps({"verdict": {"overall": "FAIL"}}), encoding="utf-8")
    argv = ["--step", "1", "--permutation-seed", "1", "--live", "--log-dir", str(env.logs)]
    assert al.main(argv, **_kw(env)) == al.EXIT_REFUSED
    assert "exactly one acceptance look" in capsys.readouterr().err
    assert al.main(["--step", "1", "--permutation-seed", "1"], **_kw(env)) == al.EXIT_REFUSED


def test_ladder_order(env):
    with pytest.raises(al.Refused, match="step 1 acceptance look has not run"):
        al.check_look_order(2, env.out, env.judge_dev)
    env.out.mkdir(parents=True)
    (env.out / "acceptance-look-step-1.json").write_text(json.dumps({"verdict": {"overall": "PASS"}}), encoding="utf-8")
    with pytest.raises(al.Refused, match="did not fail"):
        al.check_look_order(2, env.out, env.judge_dev)
    (env.out / "acceptance-look-step-1.json").write_text(json.dumps({"verdict": {"overall": "FAIL"}}), encoding="utf-8")
    al.check_look_order(2, env.out, env.judge_dev)
    with pytest.raises(al.Refused, match="step 2 acceptance look has not run"):
        al.check_look_order(3, env.out, env.judge_dev)
    with pytest.raises(al.Refused, match="must be one of"):
        al.check_look_order(4, env.out, env.judge_dev)


@pytest.mark.parametrize("round_body, match", [
    (None, "no judge-development round"),
    ({"stop_criteria_met": False}, "has not ended"),
    ({"offline": True}, "offline round"),
    ({"grader_prompt_sha256": "f" * 64}, "not the wording step 1 ended on"),
])
def test_step_1_waits_for_judge_development(tmp_path, round_body, match):
    folder = tmp_path / "jd"
    folder.mkdir()
    if round_body is not None:
        _write_judge_dev(folder, **round_body)
    with pytest.raises(al.Refused, match=match):
        al.check_look_order(1, tmp_path / "out", folder)


def test_step_1_after_round_3_without_the_stop_criteria(tmp_path):
    folder = _write_judge_dev(tmp_path / "jd", round=3, stop_criteria_met=False)
    _write_judge_dev(folder, round=0, stop_criteria_met=False, grader_prompt_sha256="e" * 64)
    al.check_look_order(1, tmp_path / "out", folder)


# --- dry run and arguments --------------------------------------------------------------------

def test_dry_run_makes_no_call(env, monkeypatch, capsys):
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert al.main(["--step", "1", "--permutation-seed", "7"], **_kw(env)) == al.EXIT_OK
    out = capsys.readouterr().out
    assert "7 passes of 40 calls" in out and grader.grader_pin(1) in out and "permutation seed 7" in out
    assert str(roster.grader_generate_config_args(1)) in out and "option order [3, 1, 0, 2]" in out
    assert "dry run: no call made" in out
    assert not env.out.exists()


def test_dry_run_at_step_2_shows_the_batch_pin(env, capsys):
    env.out.mkdir(parents=True)
    (env.out / "acceptance-look-step-1.json").write_text(json.dumps({"verdict": {"overall": "FAIL"}}), encoding="utf-8")
    assert al.main(["--step", "2", "--permutation-seed", "3"], **_kw(env)) == al.EXIT_OK
    out = capsys.readouterr().out
    assert grader.grader_pin(2) in out and "'batch': True" in out and "intra-rater check" in out


def test_live_needs_a_log_dir_outside_the_repo(env):
    base = ["--step", "1", "--permutation-seed", "1", "--live"]
    assert al.main(base, **_kw(env)) == al.EXIT_ARGS
    assert al.main(base + ["--log-dir", str(REPO / "logs")], **_kw(env)) == al.EXIT_ARGS
    assert al.main(["--step", "1", "--permutation-seed", "1", "--resamples", "0"], **_kw(env)) == al.EXIT_ARGS


def test_the_seed_is_required():
    with pytest.raises(SystemExit):
        al.main(["--step", "1"])


# --- inputs -----------------------------------------------------------------------------------

def test_inputs_are_refused_when_off_the_subset_rules(tmp_path):
    cases_file, records_dir = _write_inputs(tmp_path / "a", n=39)
    with pytest.raises(al.Refused, match="39 calibration records"):
        al.load_records(records_dir)
    cases_file, records_dir = _write_inputs(tmp_path / "b", grades=lambda i: (min(i % 4, 2), (i // 4) % 4))
    with pytest.raises(al.Refused, match="not stratified"):
        al.check_subset(al.load_records(records_dir), al.load_cases(cases_file))
    cases_file, records_dir = _write_inputs(tmp_path / "c", split="dev")
    with pytest.raises(al.Refused, match="not a calibration-split case"):
        al.load_cases(cases_file)
    with pytest.raises(al.Refused, match="not found"):
        al.load_cases(tmp_path / "missing.jsonl")


def test_records_are_checked_one_by_one(tmp_path):
    cases_file, records_dir = _write_inputs(tmp_path)
    first = sorted(records_dir.glob("*.json"))[0]
    body = json.loads(first.read_text(encoding="utf-8"))
    first.write_text(json.dumps({**body, "anchors_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(al.Refused, match="graded against anchors"):
        al.load_records(records_dir)
    first.write_text(json.dumps({**body, "grades": {G: 4, C: 0}}), encoding="utf-8")
    with pytest.raises(al.Refused, match="not a 0 to 3 integer"):
        al.load_records(records_dir)
    first.write_text(json.dumps({**body, "grades": {G: 1}}), encoding="utf-8")
    with pytest.raises(al.Refused, match="do not cover both"):
        al.load_records(records_dir)
    first.write_text(json.dumps({**body, "sample_id": "cal-01"}), encoding="utf-8")
    with pytest.raises(al.Refused, match="more than one record"):
        al.check_subset(al.load_records(records_dir), al.load_cases(cases_file))
    first.write_text(json.dumps({**body, "sample_id": "cal-99"}), encoding="utf-8")
    with pytest.raises(al.Refused, match="not in the calibration file"):
        al.check_subset(al.load_records(records_dir), al.load_cases(cases_file))


def test_record_messages_follow_the_pass_settings(env):
    cases, records = al.load_cases(env.cases_file), al.load_records(env.records_dir)
    specs = {s.name: s for s in al.plan_passes(5)}
    default = al.record_messages(records[0], cases)[1]["content"]
    assert "REC00" in default and "# FACTS SHEET" in default and "- Level 0:" in default
    relabeled = al.record_messages(records[0], cases, specs[jt.LABELS_RELABELED].settings)[1]["content"]
    assert "- Level D:" in relabeled and "- Level 0:" not in relabeled
    swapped = al.record_messages(records[0], cases, specs[jt.DIMENSION_ORDER_PERMUTED].settings)[1]["content"]
    assert swapped.index("Narrative coverage (key") < swapped.index("Narrative grounding (key")
    with pytest.raises(al.Refused, match="not in the calibration file"):
        al.record_messages({**records[0], "sample_id": "nope"}, cases)
    with pytest.raises(al.Refused, match="does not parse"):
        al.record_messages({**records[0], "worker_output": "not json"}, cases)


# --- helpers ----------------------------------------------------------------------------------

def test_plan_passes():
    specs = al.plan_passes(42)
    assert [s.name for s in specs] == list(al.PASS_NAMES) and [s.number for s in specs] == list(range(1, 8))
    by = {s.name: s.settings for s in specs}
    assert by["pass_1"] == by["pass_2"] == by["pass_3"] == gp.PromptSettings()
    assert by[jt.OPTION_ORDER_REVERSED].option_order == (3, 2, 1, 0)
    assert list(by[jt.OPTION_ORDER_PERMUTED].option_order) == jt.permuted_options([0, 1, 2, 3], 42)["options"]
    assert by[jt.LABELS_RELABELED].label_map() == gp.RELABEL_DCBA
    assert by[jt.DIMENSION_ORDER_PERMUTED].dimension_order == (C, G) == al.swapped_dimensions()
    assert specs[3].settings_record()["option_order"] == [3, 2, 1, 0]


def test_pass_task_metadata(env):
    cases, records = al.load_cases(env.cases_file), al.load_records(env.records_dir)
    spec = al.plan_passes(9)[4]
    task = al.pass_task(spec, records, cases, step=2, permutation_seed=9, bootstrap_seed=5)
    assert len(task.dataset) == N and task.dataset[0].metadata == {"split": "calibration"}
    m = task.metadata
    assert m["pass"] == jt.OPTION_ORDER_PERMUTED and m["permutation_seed"] == 9 and m["bootstrap_seed"] == 5
    assert m["split"] == "calibration" and m["roster"]["grader_step"] == 2
    assert m["grader_prompt_sha256"] == gp.prompt_sha256()


def test_offline_log_check_and_sample_check():
    with pytest.raises(roster.RunConfigError, match="not 'success'"):
        al.offline_log_check(SimpleNamespace(status="error", samples=[]), 1)
    with pytest.raises(roster.RunConfigError, match="no samples"):
        al.offline_log_check(SimpleNamespace(status="success", samples=None), 1)
    assert al.offline_log_check(SimpleNamespace(status="success", samples=[]), 1) == []
    records = [{"_file": "a"}, {"_file": "b"}]
    with pytest.raises(roster.RunConfigError, match="do not match"):
        al.check_pass_samples(SimpleNamespace(samples=[SimpleNamespace(id="a", events=[])]), records)
    log = SimpleNamespace(samples=[SimpleNamespace(id="a", events=[]), SimpleNamespace(id="b", events=[])])
    with pytest.raises(roster.RunConfigError, match="exactly one grader call"):
        al.check_pass_samples(log, records)


def _event(cache=None, config=None, model="openai/gpt-6-luna"):
    return SimpleNamespace(event="model", role="grader", model=model, cache=cache,
                           config=config if config is not None else {"reasoning_effort": "high"})


def test_pass_config():
    log = SimpleNamespace(samples=[SimpleNamespace(id="a", events=[_event()]), SimpleNamespace(id="b", events=[_event()])])
    assert al.pass_config(log, offline=False) == {"response_caching": False, "model": "openai/gpt-6-luna",
                                                   "reasoning_effort": "high"}
    assert al.pass_config(log, offline=True) == {"response_caching": False, "reasoning_effort": "high"}
    cached = SimpleNamespace(samples=[SimpleNamespace(id="a", events=[_event(cache="read")])])
    assert al.pass_config(cached, offline=True)["response_caching"] is True
    mixed = SimpleNamespace(samples=[SimpleNamespace(id="a", events=[_event(), _event(config={"reasoning_effort": "low"})])])
    with pytest.raises(roster.RunConfigError, match="2 configurations"):
        al.pass_config(mixed, offline=True)
    two = SimpleNamespace(samples=[SimpleNamespace(id="a", events=[_event(), _event(model="x/y")])])
    with pytest.raises(roster.RunConfigError, match="2 grader models"):
        al.pass_config(two, offline=False)


def test_parse_pass_maps_labels_back():
    spec = al.plan_passes(1)[5]
    records = [{"_file": "a"}, {"_file": "b"}, {"_file": "c"}]
    sample = lambda i, text, err=None: SimpleNamespace(id=i, error=err, output=SimpleNamespace(completion=text))
    log = SimpleNamespace(samples=[sample("a", reply(3, 0, labels=gp.RELABEL_DCBA, low=True)),
                                   sample("b", reply(1, 1)), sample("c", "", err="boom")])
    out = al.parse_pass(spec, log, records)
    assert out["scores"][G] == {"a": 3} and out["scores"][C] == {"a": 0} and out["low_confidence"][G] == {"a": True}
    assert [u["record"] for u in out["unparsed"]] == ["b", "c"]


def test_low_confidence_counts():
    parsed = {"pass_1": {"low_confidence": {G: {"a": True, "b": False}, C: {"a": True, "b": True}}},
              "pass_2": {"low_confidence": {G: {"a": False, "b": False}, C: {"a": False, "b": True}}}}
    out = al.low_confidence_counts(parsed)
    assert out["per_pass"] == {"pass_1": {G: 1, C: 2}, "pass_2": {G: 0, C: 1}}
    assert out["pass_1_outputs_flagged"] == 2 and out["total"] == 4


def test_subset_distribution_and_seed_types():
    records = [{"_file": "a", "grades": {G: 0, C: 3}, "source": "natural", "worker_model": "m1"},
               {"_file": "b", "grades": {G: 2, C: 3}, "source": "seeded_omission", "worker_model": "m2"}]
    pass_1 = {G: {"a": 0, "b": 1}, C: {"a": 3, "b": 3}}
    dist = al.subset_distribution(records, pass_1)
    assert dist["author"][G] == {"0": 1, "1": 0, "2": 1, "3": 0} and dist["grader_pass_1"][G]["1"] == 1
    assert dist["by_source"] == {"natural": 1, "seeded_omission": 1} and dist["by_worker_model"] == {"m1": 1, "m2": 1}
    st = al.seed_types(records, pass_1)["per_source"]
    assert st["seeded_omission"][G] == {"n": 1, "author_mean": 2.0, "grader_mean": 1.0, "exact_matches": 0}


def test_verdict_names_the_next_remedy_step():
    assert "intra-rater check first" in al.verdict(None, 1)["next"]
    assert "GPT-6.1 Sol" in al.verdict(None, 2)["next"]
    assert "excluded from every reported claim" in al.verdict(None, 3)["next"]
    ok = {"validity": {"passed": True}, "test_retest": {"alpha": {"passed": True}}, "bias_audit": {"passed": True}}
    assert al.verdict(ok, 1)["overall"] == "PASS"
    bad = {**ok, "bias_audit": {"passed": False}}
    assert al.verdict(bad, 1) == {**al.verdict(bad, 1), "bias_audit": "FAIL", "overall": "FAIL"}


def test_roles_for_and_formatting():
    assert al._roles_for(None, "pass_1") is None
    assert al._roles_for({"grader": 1}, "pass_1") == {"grader": 1}
    assert al._roles_for(lambda n: {"grader": n}, "pass_2") == {"grader": "pass_2"}
    assert al._f(None) == "undefined" and al._f(0.12345) == "0.123"
    assert al.look_file(Path("x"), 2).name == "acceptance-look-step-2.json"
