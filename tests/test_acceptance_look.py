"""The acceptance look on the calibration subset (scripts/acceptance_look.py), pre-registration
section 7.

Keyless: the only model is Inspect's mockllm with canned replies; no grader call reaches a
provider. The calibration cases and records are made-up stand-ins built from dev-split rows
and committed to a throwaway git repository in a temporary folder; nothing from the
evaluation split is read.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals import casefile, grader, rubric, roster
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import judge_trust as jt

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS
N = 40
FIXED_SEED = casefile.split_seed("calibration")


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
        (records_dir / f"{cid}-e1.json").write_text(json.dumps(rec), encoding="utf-8", newline="\n")
    cases_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return cases_file, records_dir


def _write_judge_dev(folder: Path, **kw) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    body = {"round": 0, "offline": False, "stop_criteria_met": True, "grader_prompt_sha256": gp.prompt_sha256(), **kw}
    (folder / f"round-{body['round']}.json").write_text(json.dumps(body), encoding="utf-8")
    return folder


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def commit_all(repo: Path, message: str = "inputs") -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    cases_file, records_dir = _write_inputs(repo / "data" / "calibration")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "x.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    (repo / "intra-rater.json").write_text(json.dumps({"intra_rater": "stand-in"}), encoding="utf-8", newline="\n")
    git(repo, "init", "-q")
    commit_all(repo)
    return SimpleNamespace(repo=repo, cases_file=cases_file, records_dir=records_dir, out=tmp_path / "evidence",
                           judge_dev=_write_judge_dev(tmp_path / "judge-dev"), logs=tmp_path / "logs",
                           sealed=tmp_path / "sealed", ledger=Path(git(repo, "rev-parse", "--absolute-git-dir")
                                                                   .strip()) / al.LEDGER_FOLDER,
                           intra=repo / "intra-rater.json")


def _kw(env, **extra):
    return dict(out_dir=env.out, records_dir=env.records_dir, cases_file=env.cases_file,
                judge_dev_dir=env.judge_dev, repo=env.repo, **extra)


def _live(env, step=1, *extra):
    return ["--step", str(step), "--live", "--log-dir", str(env.logs), "--sealed-dir", str(env.sealed), *extra]


def reply(g, c, *, labels=None, low=False):
    lab = labels or gp.DEFAULT_LABELS
    return json.dumps({G: {"reason": "r", "level": lab[g], "low_confidence": low},
                       C: {"reason": "r", "level": lab[c], "low_confidence": low}})


def mock_roles(scores, calls=None):
    """Per pass, a mockllm grader whose reply to each record comes from scores(pass, i) ->
    (grounding, coverage) or a raw reply string. Relabeled prompts are answered in letters.
    calls, when given, counts the calls per (pass, record)."""
    from inspect_ai.model import ModelOutput, get_model

    def roles(name):
        def out(input, tools, tool_choice, config):
            text = "\n".join(str(m.content) for m in input)
            i = int(re.search(r"REC(\d\d)", text).group(1))
            if calls is not None:
                calls[(name, i)] += 1
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


def _run(env, scores, step=1, extra=(), calls=None, **kw):
    code = al.main(_live(env, step, *extra), model_roles=mock_roles(scores, calls), **_kw(env, **kw))
    path = env.out / f"acceptance-look-step-{step}.json"
    return code, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)


def _ledger(env, step=1):
    return json.loads((env.ledger / f"step-{step}.json").read_text(encoding="utf-8"))


def _sealed(env, step=1):
    path = env.sealed / f"acceptance-look-step-{step}-rows.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


# --- full offline runs ---------------------------------------------------------------------------

def test_agreement_passes_every_gate(env, capsys):
    code, res = _run(env, near)
    assert code == al.EXIT_OK, capsys.readouterr().err
    v = res["verdict"]
    assert v == {**v, "validity": "PASS", "test_retest": "PASS", "bias_audit": "PASS", "overall": "PASS"}
    assert [p["name"] for p in res["passes"]] == list(al.PASS_NAMES) and res["records"] == N
    assert res["gates"]["validity"]["gating_lower_limit"] >= 0.60
    alpha = res["gates"]["test_retest"]["alpha"]
    assert alpha["bootstrap_seed"] == FIXED_SEED and alpha["resamples"] == 10_000
    assert res["low_confidence"]["pass_1_outputs_flagged"] == 2
    assert res["low_confidence"]["per_pass"]["pass_1"] == {G: 2, C: 2}
    assert res["score_distribution"]["author"][G] == {"0": 10, "1": 10, "2": 10, "3": 10}
    assert sum(res["score_distribution"]["by_worker_model"].values()) == N
    assert set(res["seed_types"]["per_source"]) == {"natural", "seeded_omission", "seeded_addition"}
    assert all(p["grader_config"] == {"response_caching": False, "reasoning_effort": "high"} for p in res["passes"])
    assert res["passes"][0]["log_check"] == "offline_log_check passed; 40 completed grader calls, one per output"
    assert all(p["errored_calls_skipped"] == 0 for p in res["passes"])
    rel = next(p for p in res["passes"] if p["name"] == jt.LABELS_RELABELED)
    assert rel["settings"]["labels"] == {"0": "D", "1": "C", "2": "B", "3": "A"}
    # a passing look keeps its rows in the evidence file and writes no sealed file
    assert len(res["rows"]) == N and res["rows"][0]["reasons"]["pass_1"] == {G: "r", C: "r"}
    assert not env.sealed.exists()
    # seeds fixed by rule, provenance recorded
    assert res["permutation_seed"] == FIXED_SEED and res["option_permutation"]["order"] == [3, 1, 2, 0]
    assert res["head"] == git(env.repo, "rev-parse", "HEAD").strip()
    assert res["calibration_jsonl_sha256"] == hashlib.sha256(env.cases_file.read_bytes()).hexdigest()
    assert res["records_sha256"] == al.records_set_sha256(env.records_dir)
    assert len(res["attempts"]) == 1 and res["failures"] == [] and res["intra_rater"] is None
    ledger = _ledger(env)
    assert [p["name"] for p in ledger["passes"]] == list(al.PASS_NAMES)
    assert ledger["completed"]["look_sha256"] == hashlib.sha256(
        (env.out / "acceptance-look-step-1.json").read_bytes()).hexdigest()
    out = capsys.readouterr().out
    assert "(limit 0.60)" in out and "(limit 0.80)" in out and "excess limit 0.10" in out
    assert "(limit 0.25 in magnitude)" in out and "OVERALL PASS" in out


def test_every_pass_records_the_fixed_seeds_and_provenance_in_task_metadata(env):
    from inspect_ai.log import read_eval_log

    code, res = _run(env, near)
    assert code == al.EXIT_OK
    metas = [read_eval_log(str(p), header_only=True).eval.metadata for p in sorted(env.logs.glob("*.eval"))]
    assert len(metas) == 7 and {m["pass"] for m in metas} == set(al.PASS_NAMES)
    assert all(m["permutation_seed"] == FIXED_SEED == 23260607 and m["split"] == "calibration"
               and m["grader_step"] == 1 and m["bootstrap_seed"] == FIXED_SEED and m["resamples"] == 10_000
               for m in metas)
    assert all(m["head"] == res["head"] and m["calibration_jsonl_sha256"] == res["calibration_jsonl_sha256"]
               and m["records_sha256"] == res["records_sha256"] for m in metas)
    perm = next(m for m in metas if m["pass"] == jt.OPTION_ORDER_PERMUTED)
    assert perm["prompt_settings"]["option_order"] == [3, 1, 2, 0]


def test_a_lenient_grader_fails_leniency_and_the_rows_are_sealed(env, capsys):
    code, res = _run(env, lenient)
    assert code == al.EXIT_GATE_FAIL
    bias = res["gates"]["bias_audit"]
    assert res["verdict"]["bias_audit"] == "FAIL" and res["verdict"]["overall"] == "FAIL"
    assert all(bias["per_subdimension"][d]["leniency"]["failed"] for d in gp.DIMENSIONS)
    assert bias["per_subdimension"][G]["leniency"]["mean"] == pytest.approx(0.75)
    assert "intra-rater check first" in res["verdict"]["next"]
    out = capsys.readouterr().out
    assert "OVERALL FAIL" in out
    # M2: the per-record rows are sealed outside the repo; the evidence keeps aggregates and gate values
    path, sealed = _sealed(env)
    assert res["rows"]["sealed"] is True and res["rows"]["file"] == path.name
    assert res["rows"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(sealed["rows"]) == N and sealed["rows"][0]["grader"]["pass_1"]
    assert "scores" not in res["gates"]["test_retest"] and len(sealed["test_retest_scores"]) == 3
    assert "pass_1_flagged_records" not in res["low_confidence"]
    assert res["gates"]["validity"]["per_subdimension"][G]["icc_a1"]["estimate"] is not None
    text = (env.out / "acceptance-look-step-1.json").read_text(encoding="utf-8")
    assert '"reasons"' not in text and '"low_confidence": {"pass_1"' not in text and '"reply"' not in text
    assert _ledger(env)["completed"]["sealed_file"]["sha256"] == res["rows"]["sha256"]


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


def test_an_unparsed_reply_keeps_every_gate_whose_inputs_are_complete(env, capsys):
    calls = collections.Counter()
    code, res = _run(env, lambda name, i: "no json" if (name, i) == ("pass_2", 7) else near(name, i), calls=calls)
    assert code == al.EXIT_GATE_FAIL
    assert res["unparsed_count"] == 1 and res["verdict"]["overall"] == "FAIL"
    # validity needs pass 1 only, which parsed in full: computed and stored
    assert res["verdict"]["validity"] == "PASS" and res["gates"]["validity"]["gating_lower_limit"] >= 0.60
    assert res["score_distribution"] is not None
    # test-retest and the bias audit need pass 2: FAIL with the reason, never computed from a partial pass
    assert res["gates"]["test_retest"] is None and res["gates"]["bias_audit"] is None
    assert res["verdict"]["test_retest"] == "FAIL" and res["verdict"]["bias_audit"] == "FAIL"
    assert "pass_2 (1 unparsed)" in res["verdict"]["not_computed"]["test_retest"]
    assert "not asked again" in res["gates"]["not_computed"]["bias_audit"]
    # the grader is never asked again: one call per pass and record
    assert set(calls.values()) == {1} and len(calls) == 7 * N
    # sealed at step 1: the evidence names the record, the sealed file holds the reply
    assert res["passes"][1]["unparsed"] == [{"record": "cal-07-e1"}]
    assert _sealed(env)[1]["unparsed"]["pass_2"][0]["reply"] == "no json"
    assert "test-retest: FAIL, not computed" in capsys.readouterr().out


def test_an_unparsed_pass_1_reply_fails_every_gate(env):
    code, res = _run(env, lambda name, i: "```json\n{}\n```" if (name, i) == ("pass_1", 0) else near(name, i))
    assert code == al.EXIT_GATE_FAIL
    assert res["gates"]["validity"] is None and res["score_distribution"] is None
    assert set(res["gates"]["not_computed"]) == {"validity", "test_retest", "bias_audit"}


def test_a_failed_pass_is_recorded_and_resume_keeps_the_finished_passes(env, capsys):
    seen = []

    def failing(log, step):
        name = log.eval.metadata["pass"]
        seen.append(name)
        if name == jt.LABELS_RELABELED:
            raise roster.RunConfigError("run config: off the pin")
        return []

    calls = collections.Counter()
    code, res = _run(env, near, log_check=failing, calls=calls)
    assert code == al.EXIT_CHECK and res is None
    assert not env.out.exists() or not list(env.out.iterdir())
    err = capsys.readouterr().err
    assert "pass 6 of 7" in err and "labels_relabeled" in err and "--resume" in err
    assert seen == list(al.PASS_NAMES[:6])
    ledger = _ledger(env)
    assert [p["name"] for p in ledger["passes"]] == list(al.PASS_NAMES[:5])
    assert all(len(p["log_sha256"]) == 64 for p in ledger["passes"])
    assert ledger["failures"][0]["pass"] == jt.LABELS_RELABELED and "off the pin" in ledger["failures"][0]["error"]
    assert ledger["provenance"]["permutation_seed"] == FIXED_SEED
    # a fresh start is refused while the ledger exists, and makes no call
    before = sum(calls.values())
    code, _ = _run(env, near, calls=calls)
    assert code == al.EXIT_REFUSED and "already started" in capsys.readouterr().err
    assert sum(calls.values()) == before
    # --resume re-checks passes 1 to 5 from their logs and runs only 6 and 7
    calls.clear()
    code, res = _run(env, near, extra=("--resume",), calls=calls)
    assert code == al.EXIT_OK, capsys.readouterr().err
    assert {name for name, _ in calls} == set(al.PASS_NAMES[5:])
    assert [a["resume"] for a in res["attempts"]] == [False, True]
    assert res["attempts"][1]["kept"] == list(al.PASS_NAMES[:5])
    assert len(res["failures"]) == 1 and res["failures"][0]["attempt"] == 1
    assert "re-checked on resume" in res["passes"][0]["log_check"]
    assert res["verdict"]["overall"] == "PASS"


def test_resume_refuses_a_log_that_changed(env, capsys):
    def failing(log, step):
        if log.eval.metadata["pass"] == "pass_3":
            raise roster.RunConfigError("run config: stop")
        return []

    assert _run(env, near, log_check=failing)[0] == al.EXIT_CHECK
    first = env.logs / _ledger(env)["passes"][0]["log_file"]
    first.write_bytes(first.read_bytes() + b"x")
    code, res = _run(env, near, extra=("--resume",))
    assert code == al.EXIT_CHECK and res is None
    assert "does not match its recorded SHA-256" in capsys.readouterr().err
    assert _ledger(env)["failures"][-1]["stage"].startswith("re-check of pass 1")


def test_resume_needs_a_ledger_live_and_the_same_folders(env, capsys):
    assert al.main(["--step", "1", "--resume"], **_kw(env)) == al.EXIT_ARGS
    assert al.main(_live(env, 1, "--resume"), **_kw(env)) == al.EXIT_REFUSED
    assert "no look to resume" in capsys.readouterr().err

    def failing(log, step):
        raise roster.RunConfigError("run config: stop")

    assert _run(env, near, log_check=failing)[0] == al.EXIT_CHECK
    other = ["--step", "1", "--live", "--resume", "--log-dir", str(env.logs.parent / "other")]
    assert al.main(other, model_roles=mock_roles(near), **_kw(env)) == al.EXIT_ARGS


def test_resume_refuses_a_changed_input(env, capsys):
    def failing(log, step):
        raise roster.RunConfigError("run config: stop")

    assert _run(env, near, log_check=failing)[0] == al.EXIT_CHECK
    (env.repo / "scripts" / "y.py").write_text("y = 2\n", encoding="utf-8")
    commit_all(env.repo, "another commit")
    code, _ = _run(env, near, extra=("--resume",))
    assert code == al.EXIT_REFUSED and "['head']" in capsys.readouterr().err


def test_the_ledger_lives_under_the_git_common_dir(env, tmp_path):
    common = Path(git(env.repo, "rev-parse", "--absolute-git-dir").strip())
    assert al.ledger_path(2, env.repo) == common / "icfr-acceptance-looks" / "step-2.json"
    assert al.git_common_dir(env.repo) == common.resolve()
    assert al.ledger_path(1, env.repo, tmp_path / "l") == tmp_path / "l" / "step-1.json"


def test_ledger_and_look_files_are_created_exclusively(tmp_path):
    path = tmp_path / "led" / "step-1.json"
    al.create_ledger(path, {"a": 1})
    with pytest.raises(FileExistsError):
        al.create_ledger(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_a_look_file_that_appears_during_the_look_is_never_overwritten(env, monkeypatch, capsys):
    real = al.look_result

    def plant(*a, **k):
        env.out.mkdir(parents=True, exist_ok=True)
        (env.out / "acceptance-look-step-1.json").write_text("{}", encoding="utf-8")
        return real(*a, **k)

    monkeypatch.setattr(al, "look_result", plant)
    code, res = _run(env, near)
    assert code == al.EXIT_REFUSED and res == {}
    assert "never overwritten" in capsys.readouterr().err


def test_live_runs_check_every_log_with_check_grader_log(env, monkeypatch, capsys):
    calls = []
    real = grader.check_grader_log

    def fake_check(log, step):
        calls.append(step)
        return real(log, step)

    def event():
        return SimpleNamespace(event="model", role="grader", model=grader.grader_pin(1), cache=None, error=None,
                               config={"reasoning_effort": "high"},
                               output=SimpleNamespace(model="not-the-resolved-string", usage=None))

    records = al.load_records(env.records_dir)
    fake_log = SimpleNamespace(status="success", eval=None, location=None,
                               samples=[SimpleNamespace(id=r["_file"], epoch=1, error=None, output=None,
                                                        events=[event()]) for r in records])
    monkeypatch.setattr(al.grader, "check_grader_log", fake_check)
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: fake_log)
    code = al.main(_live(env), **_kw(env))
    assert code == al.EXIT_CHECK and calls == [1]
    assert "resolved to 'not-the-resolved-string'" in capsys.readouterr().err
    assert not env.out.exists()


# --- completed grader calls -------------------------------------------------------------------

def _gev(error=None, model="openai/gpt-6-luna", resolved="gpt-6-luna"):
    return SimpleNamespace(event="model", role="grader", model=model, cache=None, error=error,
                           config={"reasoning_effort": "high"},
                           output=SimpleNamespace(model=resolved, usage=None))


def _log(*event_lists):
    return SimpleNamespace(status="success", eval=None, location=None,
                           samples=[SimpleNamespace(id=f"r{i}", epoch=1, error=None, output=None, events=list(evs))
                                    for i, evs in enumerate(event_lists)])


def test_an_errored_grader_call_followed_by_a_completed_one_is_skipped():
    records = [{"_file": "r0"}, {"_file": "r1"}]
    log = _log([_gev(error="rate limit"), _gev()], [_gev()])
    view, skipped = al.check_pass_samples(log, records)
    assert skipped == 1 and [len(s.events) for s in view.samples] == [1, 1]
    with pytest.raises(roster.RunConfigError, match="ended in an error"):
        grader.check_grader_log(log, 1)
    assert len(grader.check_grader_log(view, 1)) == 2
    assert al.pass_config(view, offline=False)["model"] == "openai/gpt-6-luna"


@pytest.mark.parametrize("events", [
    [_gev(error="boom")],                       # no completed call
    [_gev(), _gev(error="boom")],               # an error after the completed call
    [_gev(), _gev()],                           # two completed calls
    [],                                         # no call
])
def test_a_sample_without_exactly_one_completed_grader_call_fails_the_pass(events):
    with pytest.raises(roster.RunConfigError, match="exactly one completed grader call"):
        al.check_pass_samples(_log(events), [{"_file": "r0"}])


# --- one look per grader ----------------------------------------------------------------------

def _prev_look(env, verdict="FAIL", **kw):
    env.out.mkdir(parents=True, exist_ok=True)
    body = {"verdict": {"overall": verdict}, "grader_prompt_sha256": gp.prompt_sha256(),
            "anchors_sha256": rubric.llm_anchor_sha256(), **kw}
    (env.out / "acceptance-look-step-1.json").write_text(json.dumps(body), encoding="utf-8")


def test_one_look_per_grader(env, monkeypatch, capsys):
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    _prev_look(env)
    assert al.main(_live(env), **_kw(env)) == al.EXIT_REFUSED
    assert "exactly one acceptance look" in capsys.readouterr().err
    assert al.main(["--step", "1"], **_kw(env)) == al.EXIT_REFUSED


def test_ladder_order(env):
    with pytest.raises(al.Refused, match="step 1 acceptance look has not run"):
        al.check_look_order(2, env.out, env.judge_dev)
    _prev_look(env, verdict="PASS")
    with pytest.raises(al.Refused, match="did not fail"):
        al.check_look_order(2, env.out, env.judge_dev)
    _prev_look(env)
    al.check_look_order(2, env.out, env.judge_dev)
    with pytest.raises(al.Refused, match="step 2 acceptance look has not run"):
        al.check_look_order(3, env.out, env.judge_dev)
    with pytest.raises(al.Refused, match="must be one of"):
        al.check_look_order(4, env.out, env.judge_dev)


def test_above_step_1_the_prompt_is_the_previous_looks_unless_the_anchors_changed(env):
    _prev_look(env, grader_prompt_sha256="f" * 64)
    with pytest.raises(al.Refused, match="anchors did not change"):
        al.check_look_order(2, env.out, env.judge_dev)
    _prev_look(env, grader_prompt_sha256="f" * 64, anchors_sha256="e" * 64)
    al.check_look_order(2, env.out, env.judge_dev)
    env.out.joinpath("acceptance-look-step-1.json").write_text(json.dumps({"verdict": {"overall": "FAIL"}}),
                                                               encoding="utf-8")
    with pytest.raises(al.Refused, match="records no prompt or anchors hash"):
        al.check_look_order(2, env.out, env.judge_dev)


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


# --- the intra-rater record -------------------------------------------------------------------

def test_steps_2_and_3_need_a_committed_intra_rater_record(env, capsys):
    _prev_look(env)
    assert al.main(["--step", "2"], **_kw(env)) == al.EXIT_REFUSED
    assert "needs --intra-rater" in capsys.readouterr().err
    got = al.check_intra_rater(str(env.intra), 2, env.repo)
    assert got == {"file": "intra-rater.json", "sha256": hashlib.sha256(env.intra.read_bytes()).hexdigest()}
    with pytest.raises(al.Refused, match="above step 1 only"):
        al.check_intra_rater(str(env.intra), 1, env.repo)
    assert al.check_intra_rater(None, 1, env.repo) is None
    env.intra.write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(al.Refused, match="differs from HEAD"):
        al.check_intra_rater(str(env.intra), 2, env.repo)
    new = env.repo / "new.json"
    new.write_text("{}", encoding="utf-8")
    with pytest.raises(al.Refused, match="not tracked in HEAD"):
        al.check_intra_rater(str(new), 3, env.repo)
    bad = env.repo / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(al.Refused, match="not a JSON file"):
        al.check_intra_rater(str(bad), 2, env.repo)
    with pytest.raises(al.Refused, match="not inside the repository"):
        al.check_intra_rater(str(env.out.parent / "judge-dev" / "round-0.json"), 2, env.repo)
    with pytest.raises(al.Refused, match="does not exist"):
        al.check_intra_rater(str(env.repo / "missing.json"), 2, env.repo)


def test_a_step_2_look_records_the_intra_rater_hash(env, capsys):
    _prev_look(env)
    code, res = _run(env, lenient, step=2, extra=("--intra-rater", str(env.intra)))
    assert code == al.EXIT_GATE_FAIL, capsys.readouterr().err
    assert res["intra_rater"]["sha256"] == hashlib.sha256(env.intra.read_bytes()).hexdigest()
    assert res["intra_rater_sha256"] == res["intra_rater"]["sha256"] and res["rows"]["sealed"] is True
    assert "GPT-6.1 Sol" in res["verdict"]["next"]


def test_a_failed_step_3_look_keeps_its_rows(env):
    _prev_look(env)
    (env.out / "acceptance-look-step-2.json").write_text(json.dumps(
        {"verdict": {"overall": "FAIL"}, "grader_prompt_sha256": gp.prompt_sha256(),
         "anchors_sha256": rubric.llm_anchor_sha256()}), encoding="utf-8")
    code, res = _run(env, lenient, step=3, extra=("--intra-rater", str(env.intra)))
    assert code == al.EXIT_GATE_FAIL and len(res["rows"]) == N and not env.sealed.exists()


# --- repository state -------------------------------------------------------------------------

def test_live_refuses_uncommitted_calibration_inputs(env, monkeypatch, capsys):
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    first = sorted(env.records_dir.glob("*.json"))[0]
    body = first.read_text(encoding="utf-8")
    first.write_text(body.replace('"graded_by": "XX"', '"graded_by": "YY"'), encoding="utf-8", newline="\n")
    assert al.main(_live(env), **_kw(env)) == al.EXIT_REFUSED
    assert "git status is not clean" in capsys.readouterr().err
    first.write_text(body, encoding="utf-8", newline="\n")
    (env.repo / "scripts" / "x.py").write_text("x = 2\n", encoding="utf-8")
    assert al.main(_live(env), **_kw(env)) == al.EXIT_REFUSED
    assert "git status is not clean" in capsys.readouterr().err
    assert not env.ledger.exists()


def test_live_refuses_inputs_not_tracked_in_head(tmp_path):
    repo = tmp_path / "r"
    cases_file, records_dir = _write_inputs(repo / "data" / "calibration")
    git(repo, "init", "-q")
    (repo / "README").write_text("x\n", encoding="utf-8")
    git(repo, "add", "README")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "add", "data/calibration/calibration.jsonl")
    git(repo, "commit", "-q", "-m", "cases")
    (repo / ".gitignore").write_text("data/calibration/records/\n", encoding="utf-8")
    commit_all(repo, "ignore records")
    with pytest.raises(al.Refused, match="record files are not all tracked in HEAD"):
        al.repo_state(repo, cases_file, records_dir)
    (repo / ".gitignore").unlink()
    commit_all(repo, "records")
    state = al.repo_state(repo, cases_file, records_dir)
    assert state["records_files"] == N and len(state["head"]) == 40
    git(repo, "rm", "-q", "--cached", "data/calibration/calibration.jsonl")
    git(repo, "commit", "-q", "-m", "untrack")
    (repo / ".git" / "info" / "exclude").write_text("data/calibration/calibration.jsonl\n", encoding="utf-8")
    with pytest.raises(al.Refused, match="not tracked in HEAD"):
        al.repo_state(repo, cases_file, records_dir)


def test_the_records_hash_covers_every_file(tmp_path):
    folder = tmp_path / "recs"
    folder.mkdir()
    (folder / "a.json").write_text("{}", encoding="utf-8")
    one = al.records_set_sha256(folder)
    (folder / "b.json").write_text("{}", encoding="utf-8")
    two = al.records_set_sha256(folder)
    (folder / "b.json").write_text("{ }", encoding="utf-8")
    assert len({one, two, al.records_set_sha256(folder)}) == 3


# --- dry run and arguments --------------------------------------------------------------------

def test_dry_run_makes_no_call(env, monkeypatch, capsys):
    monkeypatch.setattr(al, "run_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert al.main(["--step", "1"], **_kw(env)) == al.EXIT_OK
    out = capsys.readouterr().out
    assert "7 passes of 40 calls" in out and grader.grader_pin(1) in out
    assert f"permutation seed {FIXED_SEED}" in out and "fixed by rule" in out
    assert str(roster.grader_generate_config_args(1)) in out and "option order [3, 1, 2, 0]" in out
    assert "dry run: no call made" in out
    assert not env.out.exists() and not env.ledger.exists()


def test_dry_run_at_step_2_shows_the_batch_pin(env, capsys):
    _prev_look(env)
    assert al.main(["--step", "2", "--intra-rater", str(env.intra)], **_kw(env)) == al.EXIT_OK
    out = capsys.readouterr().out
    assert grader.grader_pin(2) in out and "'batch': True" in out and "intra-rater record intra-rater.json" in out


def test_live_needs_log_and_sealed_folders_outside_the_repo(env, capsys):
    base = ["--step", "1", "--live"]
    sealed = ["--sealed-dir", str(env.sealed)]
    logs = ["--log-dir", str(env.logs)]
    assert al.main(base + sealed, **_kw(env)) == al.EXIT_ARGS
    assert al.main(base + ["--log-dir", str(REPO / "logs")] + sealed, **_kw(env)) == al.EXIT_ARGS
    assert al.main(base + ["--log-dir", str(env.repo / "logs")] + sealed, **_kw(env)) == al.EXIT_ARGS
    assert al.main(base + logs, **_kw(env)) == al.EXIT_ARGS
    assert al.main(base + logs + ["--sealed-dir", str(REPO / "sealed")], **_kw(env)) == al.EXIT_ARGS
    assert "--sealed-dir outside the repository" in capsys.readouterr().err
    env.sealed.mkdir()
    (env.sealed / "acceptance-look-step-1-rows.json").write_text("{}", encoding="utf-8")
    assert al.main(base + logs + sealed, **_kw(env)) == al.EXIT_REFUSED
    assert not env.ledger.exists()


def test_the_seeds_are_fixed_by_rule():
    assert al.PERMUTATION_SEED == al.BOOTSTRAP_SEED == FIXED_SEED == 23260607 and al.RESAMPLES == 10_000
    draw = jt.permuted_options([0, 1, 2, 3], al.PERMUTATION_SEED)
    assert draw["order"] == [3, 1, 2, 0] and not draw["is_identity"] and not draw["is_reversal"]
    for flag in ("--permutation-seed", "--bootstrap-seed", "--resamples"):
        with pytest.raises(SystemExit):
            al.main(["--step", "1", flag, "11"])
    with pytest.raises(al.Refused, match="fixed by rule"):
        al.plan_passes(11)


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
    specs = {s.name: s for s in al.plan_passes()}
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
    specs = al.plan_passes()
    assert [s.name for s in specs] == list(al.PASS_NAMES) and [s.number for s in specs] == list(range(1, 8))
    by = {s.name: s.settings for s in specs}
    assert by["pass_1"] == by["pass_2"] == by["pass_3"] == gp.PromptSettings()
    assert by[jt.OPTION_ORDER_REVERSED].option_order == (3, 2, 1, 0)
    assert by[jt.OPTION_ORDER_PERMUTED].option_order == (3, 1, 2, 0)
    assert by[jt.LABELS_RELABELED].label_map() == gp.RELABEL_DCBA
    assert by[jt.DIMENSION_ORDER_PERMUTED].dimension_order == (C, G) == al.swapped_dimensions()
    assert specs[3].settings_record()["option_order"] == [3, 2, 1, 0]


def test_pass_task_metadata(env):
    cases, records = al.load_cases(env.cases_file), al.load_records(env.records_dir)
    spec = al.plan_passes()[4]
    task = al.pass_task(spec, records, cases, step=2, provenance={"head": "abc", "records_sha256": "d" * 64})
    assert len(task.dataset) == N and task.dataset[0].metadata == {"split": "calibration"}
    m = task.metadata
    assert m["pass"] == jt.OPTION_ORDER_PERMUTED and m["permutation_seed"] == FIXED_SEED
    assert m["bootstrap_seed"] == FIXED_SEED and m["resamples"] == 10_000
    assert m["permutation_seed_rule"] == "casefile.split_seed('calibration')"
    assert m["split"] == "calibration" and m["roster"]["grader_step"] == 2
    assert m["grader_prompt_sha256"] == gp.prompt_sha256() and m["head"] == "abc"


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
    with pytest.raises(roster.RunConfigError, match="exactly one completed grader call"):
        al.check_pass_samples(log, records)
    with pytest.raises(roster.RunConfigError, match="no samples"):
        al.check_pass_samples(SimpleNamespace(samples=None), records)


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
    spec = al.plan_passes()[5]
    records = [{"_file": "a"}, {"_file": "b"}, {"_file": "c"}, {"_file": "d"}]
    sample = lambda i, text, err=None: SimpleNamespace(id=i, error=err, output=SimpleNamespace(completion=text))
    lower = reply(2, 1, labels={k: v.lower() for k, v in gp.RELABEL_DCBA.items()})
    log = SimpleNamespace(samples=[sample("a", reply(3, 0, labels=gp.RELABEL_DCBA, low=True)),
                                   sample("b", reply(1, 1)), sample("c", "", err="boom"), sample("d", lower)])
    out = al.parse_pass(spec, log, records)
    assert out["scores"][G] == {"a": 3, "d": 2} and out["scores"][C] == {"a": 0, "d": 1}
    assert out["low_confidence"][G] == {"a": True, "d": False}
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
    partial = {**ok, "test_retest": None, "not_computed": {"test_retest": "FAIL: pass_2 incomplete"}}
    v = al.verdict(partial, 1)
    assert v["validity"] == "PASS" and v["test_retest"] == "FAIL" and v["overall"] == "FAIL"
    assert v["not_computed"] == {"test_retest": "FAIL: pass_2 incomplete"}


def test_seal_split_keeps_aggregates_only():
    res = {"look": "acceptance look", "grader_step": 1, "rows": [{"record": "a", "reasons": {"pass_1": "x"}}],
           "passes": [{"name": "pass_1", "unparsed": [{"record": "a", "error": "e", "reply": "R"}]}],
           "low_confidence": {"total": 1, "pass_1_flagged_records": ["a"]},
           "gates": {"test_retest": {"scores": [1], "alpha": {}}, "validity": None}}
    public, sealed = al.seal_split(res)
    assert "rows" not in public and public["passes"][0]["unparsed"] == [{"record": "a"}]
    assert "scores" not in public["gates"]["test_retest"] and "pass_1_flagged_records" not in public["low_confidence"]
    assert sealed["rows"] == res["rows"] and sealed["unparsed"]["pass_1"][0]["reply"] == "R"
    assert sealed["test_retest_scores"] == [1] and sealed["pass_1_flagged_records"] == ["a"]
    assert res["rows"] and res["gates"]["test_retest"]["scores"] == [1]


def test_roles_for_and_formatting():
    assert al._roles_for(None, "pass_1") is None
    assert al._roles_for({"grader": 1}, "pass_1") == {"grader": 1}
    assert al._roles_for(lambda n: {"grader": n}, "pass_2") == {"grader": "pass_2"}
    assert al._f(None) == "undefined" and al._f(0.12345) == "0.123"
    assert al.look_file(Path("x"), 2).name == "acceptance-look-step-2.json"
    assert al.sealed_file(Path("x"), 2).name == "acceptance-look-step-2-rows.json"


# --- the reply parser ---------------------------------------------------------------------------

def test_a_label_parses_in_any_letter_case_and_nothing_more_lenient():
    settings = gp.PromptSettings(labels=dict(gp.RELABEL_DCBA))
    lower = json.dumps({G: {"reason": "r", "level": "b", "low_confidence": False},
                        C: {"reason": "r", "level": "d", "low_confidence": True}})
    out = gp.parse_reply(lower, settings)
    assert out[G]["level"] == 2 and out[C]["level"] == 0
    assert gp.parse_reply("```json\n" + lower + "\n```", settings)[G]["level"] == 2
    for bad in (lower.replace('"b"', '" B"'), lower.replace('"b"', '"Level B"'), "Here it is: " + lower,
                lower + "\n" + lower, "```json\n" + lower + "\n```\nDone."):
        with pytest.raises(gp.GraderReplyError):
            gp.parse_reply(bad, settings)
    with pytest.raises(ValueError, match="any letter case"):
        gp.PromptSettings(labels={0: "a", 1: "A", 2: "b", 3: "c"}).check()


def test_round_0_replies_parse_the_same():
    body = json.loads((REPO / "docs" / "evidence" / "judge-dev" / "round-0.json").read_text(encoding="utf-8"))
    assert body["unparsed"] == [] and len(body["rows"]) == 20
    for row in body["rows"]:
        out = gp.parse_reply(row["reply"])
        assert {d: out[d]["level"] for d in gp.DIMENSIONS} == row["grader"]
        assert {d: out[d]["low_confidence"] for d in gp.DIMENSIONS} == row["low_confidence"]
        assert {d: out[d]["reason"] for d in gp.DIMENSIONS} == row["reasons"]
