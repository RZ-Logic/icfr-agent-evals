"""scripts/grading_job.py end to end, offline: the full-run grading job (decision D-R23).

Keyless: the grader and calibrator are Inspect's mockllm, answering from the narrative marker
of each mock worker output (tests/gradingfactory.py); the evaluated model of every grading eval
is none/none. The four worker logs are mock copies of one small mockllm run of the 20 dev pilot
cases, with headers that claim the evaluation split; no evaluation case is built, read or
scored. The start checks run on scratch git repositories under tmp_path, the ledger and the
plumbing record live in tmp_path, and nothing touches this repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import shutil
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.log import EvalError, read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.model._providers.util import batch_log

import gradingfactory as gf
from icfr_agent_evals import grader, reliability, roster, rubric
from icfr_agent_evals import grader_prompt as gp

REPO = Path(__file__).resolve().parents[1]
gc = gf.load_script("grading_common")
cc = gf.load_script("calibrator_call")
gs = gf.load_script("grading_stats")
cpc = gf.load_script("calibrator_plumbing_check")
gj = gf.load_script("grading_job")
REAL_RUN_EVAL = gj.run_eval

UNITS = gf.EXPECTED["units"]           # 314
ROWS_DRAWN, ROWS_LOST = 59, 1          # 3 sampled models x 20 cases, one (model, case) with no gradeable epoch
REHEARSAL = gj.REHEARSAL_UNITS
REHEARSAL_TASK = gc.GRADER_EVAL_NAME.format(1) + "_rehearsal"
EVIDENCE_KEYS = {"evidence", "run", "job_id", "ledger_started", "worker_run_commit", "head", "origin_main", "archive",
                 "python_version", "python_pin", "inspect_ai_version", "worker_logs", "worker_header_grader_step",
                 "grader", "calibrator", "plumbing_check", "rules", "sampling_calls", "seeds", "parts", "attempts",
                 "attempt_logs", "batch_ids", "counts", "sealed", "ledger", "withheld_until_freeze", "decision", "note"}
RULE_KEYS = ["seed", "draw", "grader_order", "calibrator_order", "reply_of_record", "failed_judgment", "gate",
             "breaker", "means"]
WITHHELD = ["reliability gate values, the worst-case figure, the break-even counts and the bootstrap limit",
            "circuit breaker rates and trip", "LLM sub-dimension means and paired differences",
            "queue items and every tag count", "the unit map, the draw pool and the drawn ids"]
ATTEMPT_LOG_KEYS = {"attempt", "part", "role", "eval_kind", "log_file", "log_sha256", "status", "asked", "sent",
                    "completed", "errored_calls_skipped", "max_samples", "found_on_resume"}
PLUMBING_KEYS = {"file", "sha256", "attempt", "calls", "parsed", "refused", "passed", "head"}
LOG_CHECK_HINT = gj.STAGE_HINTS["log check"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_of(chat) -> str:
    return chat if isinstance(chat, str) else "\n".join(str(m.text) for m in chat)


def assert_quiet(res) -> None:
    """No worker model id or stand-in folder name in what a run printed, on any path."""
    printed = res.out + res.err
    for s in (*gf.LEAK_STRINGS, *roster.EVALUATED_MODELS):
        assert s not in printed, s


# --- The world: worker logs, a scratch repository and a passed plumbing record ----------------

def _plumbing_roles():
    def out(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", gf.reply(2, 1))

    return {"calibrator": get_model("mockllm/model", custom_outputs=out, memoize=False)}


def run_plumbing(root: Path, env, *, log_dir: Path, record_dir: Path) -> int:
    return cpc.main(["--live", "--log-dir", str(log_dir)], model_roles=_plumbing_roles(), repo=env.repo,
                    ledger_dir=record_dir, run_commit=env.run_commit, archive_file=env.archive_file,
                    running_package=False, cases_file=root / "no-calibration-cases.jsonl")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    good, bad = gf.base_runs(root / "base")
    env = gf.make_repo(root)
    worker_dir = root / "worker-logs"
    paths = gf.write_worker_logs(good, bad, worker_dir, run_commit=env.run_commit)
    plumbing = root / "plumbing"
    assert run_plumbing(root, env, log_dir=root / "plumbing-logs", record_dir=plumbing) == cpc.EXIT_OK
    trials = {m: [repr(reliability.evaluate_trial(s)) for s in read_eval_log(str(p)).samples] for m, p in paths.items()}
    return SimpleNamespace(root=root, repo=env, worker_dir=worker_dir, paths=paths, plumbing=plumbing,
                           shas={m: sha(p) for m, p in paths.items()}, trials=trials, good=good, bad=bad)


@pytest.fixture(scope="module")
def planned(world):
    """The plan the job builds from the world's worker logs, and the admitted logs."""
    paths = gc.find_worker_logs(world.worker_dir, gf.MODELS)
    logs = gc.admit_worker_logs(paths, cases=gf.dev_cases(), composition=gf.composition(), models=gf.MODELS,
                                run_commit=world.repo.run_commit)
    plan = gc.build_plan(logs, gf.dev_cases(), calibrator_model=gf.DELTA, per_model=20)
    return SimpleNamespace(plan=plan, logs=logs)


def job_env(root: Path) -> SimpleNamespace:
    return SimpleNamespace(root=root, ledger=root / "ledger", logs=root / "grading" / "_sealed" / "logs",
                           sealed=root / "grading" / "_sealed", out=root / "evidence")


def live(world, env, *extra, logs=None) -> list[str]:
    return ["--live", "--log-dir", str(logs or env.logs), "--sealed-dir", str(env.sealed),
            "--worker-logs", str(world.worker_dir), *extra]


def kw(world, env, repo=None, **extra) -> dict:
    r = repo or world.repo
    base = dict(repo=r.repo, ledger_dir=env.ledger, out_dir=env.out, cases=gf.dev_cases(), models=gf.MODELS,
                calibrator_model=gf.DELTA, composition=gf.composition(), per_model=20, run_commit=r.run_commit,
                look_file=r.look_file, archive_file=r.archive_file, running_package=False,
                plumbing_dir=world.plumbing, resamples=200)
    base.update(extra)
    return base


def ledger_of(env) -> dict:
    return json.loads((env.ledger / gc.LEDGER_FILE).read_text(encoding="utf-8"))


def lock_of(env) -> Path:
    return env.ledger / gc.LEDGER_LOCK_FILE


def sealed_of(env, name) -> dict:
    return json.loads((env.sealed / name).read_text(encoding="utf-8"))


def evidence_path(env) -> Path:
    return env.out / gc.EVIDENCE_FILE


def unit_keys(env) -> dict[str, tuple]:
    units = sealed_of(env, gc.SEALED_UNIT_MAP)["units"]
    return {u: (x["model"], x["sample_id"], x["epoch"]) for u, x in units.items()}


class Spy:
    """Stands in for grading_job.run_eval: records each eval's task name, the roles it bound and
    the ledger and plan files as they stood before the call, then runs the eval (or a hook)."""

    def __init__(self, env, hook=None):
        self.env, self.hook, self.seen = env, hook, []

    def __call__(self, task, *a, **k):
        roles = k.get("roles", a[0] if a else None)
        path = self.env.ledger / gc.LEDGER_FILE
        self.seen.append({"task": task.name, "roles": sorted(roles or {}), "ids": [str(s.id) for s in task.dataset],
                          "ledger": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
                          "plan_files": {n: (self.env.sealed / n).exists() for n in gc.PLAN_FILES},
                          "metadata": dict(task.metadata or {}), "max_samples": k.get("max_samples")})
        if self.hook is not None:
            return self.hook(task, *a, **k)
        return REAL_RUN_EVAL(task, *a, **k)


def run_main(argv, roles, kwargs, monkeypatch_ctx=None):
    out, err = io.StringIO(), io.StringIO()
    raised = None
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = gj.main(argv, model_roles=roles, **kwargs)
        except RuntimeError as e:          # a hook that kills the process mid-eval
            code, raised = None, e
    return SimpleNamespace(code=code, out=out.getvalue(), err=err.getvalue(), raised=raised)


def crash_after(part_task: str, keep: int, *, role: str = "grader", forced: set | None = None):
    """A run_eval hook: the first eval named part_task keeps only its first `keep` samples and ends
    with status error, as a Batch job that died; the first kept unit's judge reply is forced
    unparsed. Every other eval runs as normal."""
    state = {"done": False, "kept": [], "dropped": [], "forced_key": None, "log": None}

    def hook(task, *a, **k):
        if task.name != part_task or state["done"]:
            return REAL_RUN_EVAL(task, *a, **k)
        state["done"] = True
        ids = [str(s.id) for s in task.dataset]
        key = gf.unmark(text_of(task.dataset[0].input))
        state["forced_key"] = key
        if forced is not None:
            forced.add((role, *key))
        log = REAL_RUN_EVAL(task, *a, **k)
        state["kept"], state["dropped"] = ids[:keep], ids[keep:]
        log.samples = [s for s in log.samples if str(s.id) in state["kept"]]
        log.status = "error"
        log.error = EvalError(message="the batch job died", traceback="", traceback_ansi="")
        write_eval_log(log, log.location)
        state["log"] = Path(str(log.location))
        return log

    return hook, state


def forced_plan(forced: set):
    def plan(role, model, sid, epoch):
        if (role, model, sid, epoch) in forced:
            return "garbled reply, not JSON"
        return gf.default_plan(role, model, sid, epoch)

    return plan


def guarded_evaluate_trial(spy, seen: list):
    """reliability.evaluate_trial as the job may use it: on worker samples only, before the
    first grading eval (admission reads the defect epochs through it). A call after grading
    started, or on a grading sample, raises."""
    real = reliability.evaluate_trial

    def guarded(sample, *a, **k):
        if spy.seen or re.fullmatch(r"u\d{4}", str(getattr(sample, "id", ""))):
            raise AssertionError("reliability.evaluate_trial was called after grading started")
        seen.append(str(sample.id))
        return real(sample, *a, **k)

    return guarded


# --- One full offline job -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def done(world, tmp_path_factory):
    env = job_env(tmp_path_factory.mktemp("done"))
    calls = Counter()
    spy = Spy(env)
    trial_calls: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        mp.setattr(reliability, "evaluate_trial", guarded_evaluate_trial(spy, trial_calls))   # a tag never reaches it
        res = run_main(live(world, env), gf.judge_roles(calls=calls), kw(world, env))
    ev = json.loads(evidence_path(env).read_text(encoding="utf-8")) if evidence_path(env).exists() else None
    return SimpleNamespace(env=env, res=res, calls=calls, spy=spy, evidence=ev, trial_calls=trial_calls)


def test_a_full_offline_job_writes_sealed_results_and_pooled_evidence(done, world):
    assert done.res.code == gj.EXIT_OK, done.res.err
    env, ev = done.env, done.evidence
    assert ev is not None and set(ev) == EVIDENCE_KEYS
    assert evidence_path(env).read_bytes() == gc.canonical_bytes(ev)
    led = ledger_of(env)
    assert ev["evidence"] == "full-run grading" and ev["run"] == "2026-11-21"
    assert ev["job_id"] == led["job_id"] and ev["ledger_started"] == led["started"]
    assert ev["head"] == world.repo.head and ev["origin_main"] == world.repo.head
    assert ev["archive"]["commit"] == world.repo.run_commit and ev["archive"]["sha256"] == sha(world.repo.archive_file)
    assert ev["python_version"] == platform.python_version() and ev["python_pin"] == "3.14.10"
    import inspect_ai

    assert ev["inspect_ai_version"] == inspect_ai.__version__
    want_logs = sorted(({"file": p.name, "sha256": world.shas[m], "eval_id": read_eval_log(str(p), header_only=True)
                         .eval.eval_id} for m, p in world.paths.items()), key=lambda x: x["sha256"])
    assert ev["worker_logs"] == want_logs and ev["worker_header_grader_step"] == 1
    g = ev["grader"]
    assert g["model"] == grader.grader_pin(2) and g["step"] == 2 and g["batch"] is True and g["cache"] is False
    assert g["generate_config"] == roster.grader_generate_config_args(2)
    assert g["prompt_version"] == "gp-1" and g["prompt_sha256"] == gp.prompt_sha256()
    assert g["anchors_sha256"] == rubric.llm_anchor_sha256() and g["resolved"] == "gpt-6.1-terra"
    assert g["look"]["verdict"] == "PASS" and g["look"]["sha256"] == sha(world.repo.look_file)
    c = ev["calibrator"]
    assert c["model"] == cc.calibrator_pin() and c["role"] == "calibrator" and c["batch"] is None
    assert c["cache"] is False and c["fallback"] is None and c["tools"] is None
    assert c["generate_config"] == roster.calibrator_generate_config_args()
    assert c["prompt_version"] == "gp-1" and c["prompt_sha256"] == gp.prompt_sha256()
    assert set(ev["plumbing_check"]) == PLUMBING_KEYS                     # never the plumbing log folder
    assert ev["plumbing_check"]["passed"] is True and ev["plumbing_check"]["calls"] == 12
    assert ev["plumbing_check"]["head"] == world.repo.head
    assert ev["seeds"] == gc.plan_seeds(world.shas) and ev["sampling_calls"] == list(gc.SAMPLING_CALLS)
    assert ev["parts"] == {"grader": [79, 79, 78, 78], "calibrator": ROWS_DRAWN}
    assert ev["batch_ids"] == [] and ev["decision"] == "D-R23" and ev["withheld_until_freeze"] == WITHHELD
    assert set(ev["rules"]) == set(RULE_KEYS) and "worst_case" not in ev["rules"]
    assert ev["rules"]["gate"] == gj.RULE_TEXTS["gate"] and "not gated" in ev["rules"]["gate"]
    assert ev["rules"]["failed_judgment"].endswith("(D-R23, failed judgments)")
    assert "mean of per-sample epoch means" in ev["rules"]["means"]
    assert "pair rate" in ev["rules"]["breaker"] and "generation rate" in ev["rules"]["breaker"]
    assert "never a breaker tag" in ev["rules"]["breaker"]
    assert ev["ledger"] == {"file": gc.LEDGER_FILE, "folder": "<git common dir>/icfr-grading"}
    assert ev["attempts"] == led["attempts"] and ev["attempts"][0]["grader_max_samples"] == 200
    counts = ev["counts"]
    assert counts["worker_logs"] == 4 and counts["generations"] == 320 and counts["units"] == UNITS
    assert counts["skipped_errored"] == 1 and counts["skipped_unparsed"] == 5
    assert counts["grader"] == {"replies_of_record": UNITS, "parsed": UNITS - 4, "unparsed": 4, "refused": 0,
                                "failed": 0, "errored_calls_skipped": 0, "attempt_logs": 5}
    assert counts["calibrator"] == {"rows_drawn": ROWS_DRAWN, "rows_lost": ROWS_LOST, "replies_of_record": ROWS_DRAWN,
                                    "parsed": ROWS_DRAWN - 2, "unparsed": 1, "refused": 1, "failed": 0,
                                    "errored_calls_skipped": 0, "attempt_logs": 1, "pairs_complete": ROWS_DRAWN - 3,
                                    "pairs_dropped": 3}
    assert ev["sealed"]["folder"] == "_sealed"
    files = ev["sealed"]["files"]
    assert set(files) == set(gc.PLAN_FILES + gc.RESULT_FILES)
    assert all(files[n] == sha(env.sealed / n) for n in files)
    done_mark = led["completed"]
    assert done_mark["evidence_sha256"] == sha(evidence_path(env)) and done_mark["sealed_files"] == files
    assert done_mark["evidence_body"] == ev and led["rejected_logs"] == []
    assert "314" in done.res.out and "sealed results in" in done.res.out and "0 failed" in done.res.out
    assert not lock_of(env).exists()                                     # the lock is released


def test_the_sealed_results_hold_grades_queue_and_aggregates(done):
    env = done.env
    for name in gc.PLAN_FILES + gc.RESULT_FILES:
        body = sealed_of(env, name)
        assert (env.sealed / name).read_bytes() == gc.canonical_bytes(body) and body["run"] == "2026-11-21"
    um = sealed_of(env, gc.SEALED_UNIT_MAP)
    assert len(um["units"]) == UNITS and len(um["generations"]) == 320
    drawn = sealed_of(env, gc.SEALED_DRAWN)
    assert len(drawn["rows"]) == ROWS_DRAWN and len(drawn["lost"]) == ROWS_LOST
    gr = sealed_of(env, gc.SEALED_GRADER_REPLIES)["replies"]
    assert set(gr) == set(um["units"]) and Counter(r["status"] for r in gr.values()) == {"parsed": UNITS - 4,
                                                                                         "unparsed": 4}
    keys = unit_keys(env)
    for u, r in gr.items():
        assert r["attempt"] == 1
        if r["status"] == "unparsed":
            m, sid, _ = keys[u]
            assert (m, gf.case_index(sid)) == (gf.CHARLIE, 3) and r["reply"] == "no json here" and r["scores"] == {}
    cr = sealed_of(env, gc.SEALED_CALIBRATOR_REPLIES)["replies"]
    assert set(cr) == {r["uid"] for r in drawn["rows"]}
    assert Counter(r["status"] for r in cr.values()) == {"parsed": ROWS_DRAWN - 2, "unparsed": 1, "refused": 1}
    queue = sealed_of(env, gc.SEALED_QUEUE)
    assert "after the second-look freeze" in queue["note"] and "(D-R23, unseal step)" in queue["note"]
    by_tag = Counter(t for item in queue["items"] for t in item["tags"])
    assert by_tag == {gs.TAG_DISAGREEMENT: 2, gs.TAG_GRADER_LOW: 4, gs.TAG_CALIBRATOR_LOW: 1,
                      gs.TAG_GRADER_FAILED: 4, gs.TAG_CALIBRATOR_FAILED: 2}
    assert all(i["disposition"] is None for i in queue["items"])
    assert not any(s in json.dumps(queue) for s in gf.LEAK_STRINGS)
    agg = sealed_of(env, gc.SEALED_AGGREGATES)
    assert agg["gate"]["pairs_complete"] == ROWS_DRAWN - 3 and agg["gate"]["pairs_dropped"] == 3
    assert agg["breaker"]["unit"] == "full run" and agg["lost_rows_are_not_dropped_pairs"] is True
    assert agg["queue_counts"]["items"] == len(queue["items"])
    assert set(agg["means"]) == set(gf.MODELS)


def test_each_judge_is_asked_once_per_unit(done):
    grader_calls = {k: v for k, v in done.calls.items() if k[0] == "grader"}
    cal_calls = {k: v for k, v in done.calls.items() if k[0] == "calibrator"}
    assert len(grader_calls) == UNITS and set(grader_calls.values()) == {1}
    assert len(cal_calls) == ROWS_DRAWN and set(cal_calls.values()) == {1}
    assert not any(k[1] == gf.DELTA for k in cal_calls)                    # never the Opus stand-in
    m, ci, ep = gf.ERRORED_EPOCH
    assert ("grader", m, gf.pilot_ids()[ci], ep) not in done.calls          # an errored epoch is never graded


def test_every_grading_log_passes_its_checks_and_records_its_sha(done):
    env = done.env
    led = ledger_of(env)
    entries = led["attempt_logs"]
    assert all(set(e) == ATTEMPT_LOG_KEYS for e in entries)
    assert [e["part"] for e in entries] == ["grader-1", *gc.GRADER_PART_KEYS, "calibrator"]
    assert [e["eval_kind"] for e in entries] == ["rehearsal"] + ["main"] * 5
    assert [e["role"] for e in entries] == ["grader"] * 5 + ["calibrator"]
    assert sorted(p.name for p in env.logs.glob("*.eval")) == sorted(e["log_file"] for e in entries)
    um = sealed_of(env, gc.SEALED_UNIT_MAP)["units"]
    part_1 = led["parts"]["grader-1"]
    assert entries[0]["asked"] == part_1[:REHEARSAL] and entries[1]["asked"] == part_1[REHEARSAL:]
    for e in entries[2:]:
        assert e["asked"] == led["parts"][e["part"]]
    assert all(e["sent"] == e["asked"] for e in entries)              # every asked unit reached the judge
    for e in entries:
        path = env.logs / e["log_file"]
        assert e["log_sha256"] == sha(path) and e["status"] == "success" and e["attempt"] == 1
        assert e["completed"] == len(e["asked"])
        assert e["errored_calls_skipped"] == 0 and e["found_on_resume"] is False
        log = read_eval_log(str(path))
        assert log.eval.metadata["grading_job"] == led["job_id"] and log.eval.metadata["role"] == e["role"]
        assert log.eval.metadata["eval_kind"] == e["eval_kind"]
        assert sorted(str(s.id) for s in log.samples) == sorted(e["asked"])
        for s in log.samples:
            assert gc.chat_messages_sha256(s.input) == um[str(s.id)]["messages_sha256"]
    assert [s["task"] for s in done.spy.seen] == [REHEARSAL_TASK, *(gc.GRADER_EVAL_NAME.format(i) for i in range(1, 5)),
                                                  gc.CALIBRATOR_EVAL_NAME]
    assert done.spy.seen[0]["ids"] == part_1[:REHEARSAL] and done.spy.seen[1]["ids"] == part_1[REHEARSAL:]
    ror = led["replies_of_record"]
    assert set(ror) == set(um)
    drawn = set(led["parts"]["calibrator"])
    for u, v in ror.items():
        assert v["grader"]["attempt"] == 1 and v["grader"]["log_file"] in {e["log_file"] for e in entries[:5]}
        assert ("calibrator" in v) == (u in drawn)


def test_a_tag_never_reaches_evaluate_trial(done, world):
    assert done.res.code == gj.EXIT_OK              # evaluate_trial raised if called after grading started
    assert done.spy.seen and not any(re.fullmatch(r"u\d{4}", i) for i in done.trial_calls)
    for m, p in world.paths.items():
        assert sha(p) == world.shas[m]
        assert [repr(reliability.evaluate_trial(s)) for s in read_eval_log(str(p)).samples] == world.trials[m]
    assert sealed_of(done.env, gc.SEALED_QUEUE)["items"]
    for m, p in world.paths.items():
        for s in read_eval_log(str(p)).samples:
            assert not any(t in json.dumps(s.scores or {}, default=str) for t in gs.TAG_ORDER)


def test_stdout_and_evidence_name_no_worker_model_beside_a_count(done):
    printed = done.res.out + done.res.err
    text = evidence_path(done.env).read_text(encoding="utf-8")
    for s in gf.LEAK_STRINGS:
        assert s not in printed and s not in text
    for m in roster.EVALUATED_MODELS:
        assert m not in printed
    assert text.count(roster.CALIBRATOR) == 1 and done.evidence["calibrator"]["model"] == roster.CALIBRATOR
    for m in set(roster.EVALUATED_MODELS) - {roster.CALIBRATOR}:
        assert m not in text
    assert all(set(x) == {"file", "sha256", "eval_id"} for x in done.evidence["worker_logs"])
    assert all("asked" not in x and "sent" not in x for x in done.evidence["attempt_logs"])
    assert "tag" not in json.dumps(done.evidence["counts"])


def test_no_eval_name_metadata_sample_id_or_ledger_key_names_a_worker_model(done):
    env = done.env
    for path in env.logs.glob("*.eval"):
        log = read_eval_log(str(path))
        meta = {k: v for k, v in (log.eval.metadata or {}).items() if k != "calibrator_model"}
        header = json.dumps({"task": log.eval.task, "task_args": log.eval.task_args, "metadata": meta}, default=str)
        assert log.eval.task in {REHEARSAL_TASK, *(gc.GRADER_EVAL_NAME.format(i) for i in range(1, 5)),
                                 gc.CALIBRATOR_EVAL_NAME}
        for s in log.samples:
            assert re.fullmatch(r"u\d{4}", str(s.id)) and s.metadata == {"split": "eval", "unit": str(s.id)}
        for leak in (*gf.LEAK_STRINGS, *roster.EVALUATED_MODELS):
            assert leak not in header
    led = ledger_of(env)
    text = json.dumps(led)
    assert not any(s in text for s in gf.LEAK_STRINGS)
    assert all(re.fullmatch(r"u\d{4}", u) for u in led["replies_of_record"])
    assert all(re.fullmatch(r"u\d{4}", u) for p in led["parts"].values() for u in p)
    prov = json.dumps(led["provenance"])
    assert not any(m in prov for m in (*gf.LEAK_STRINGS, *roster.EVALUATED_MODELS))


def test_grader_evals_run_with_the_max_samples_option_and_continue_on_fail(done):
    seen = {s["task"]: s for s in done.spy.seen}
    for name in (REHEARSAL_TASK, *(gc.GRADER_EVAL_NAME.format(i) for i in range(1, 5))):
        assert seen[name]["max_samples"] == gj.GRADER_MAX_SAMPLES_DEFAULT == 200
    assert seen[gc.CALIBRATOR_EVAL_NAME]["max_samples"] is None
    for e in ledger_of(done.env)["attempt_logs"]:
        header = read_eval_log(str(done.env.logs / e["log_file"]), header_only=True).eval.config
        assert header.continue_on_fail is True
        want = None if e["role"] == "calibrator" else 200
        assert header.max_samples == want and e["max_samples"] == want


def test_the_calibrator_eval_is_not_batched_and_binds_only_the_calibrator_role(done):
    roles = {s["task"]: s["roles"] for s in done.spy.seen}
    assert roles[gc.CALIBRATOR_EVAL_NAME] == ["calibrator"]
    assert all(roles[gc.GRADER_EVAL_NAME.format(i)] == ["grader"] for i in range(1, 5))
    assert roles[REHEARSAL_TASK] == ["grader"]
    meta = next(s["metadata"] for s in done.spy.seen if s["task"] == gc.CALIBRATOR_EVAL_NAME)
    assert meta["batched"] is False and meta["role"] == "calibrator" and meta["part"] == "calibrator"
    assert meta["eval_kind"] == "main" and meta["units"] == ROWS_DRAWN
    assert meta["calibrator_model"] == cc.calibrator_pin() and "grader_model" not in meta
    assert meta["calibrator_generate_config"] == {"reasoning_effort": "high"}
    entry = next(e for e in ledger_of(done.env)["attempt_logs"] if e["role"] == "calibrator")
    log = read_eval_log(str(done.env.logs / entry["log_file"]))
    assert sorted(log.eval.model_roles or {}) == ["calibrator"] and str(log.eval.model) == grader.NO_MODEL
    events = [e for _, e in cc.calibrator_events(log)]
    assert len(events) == ROWS_DRAWN
    for e in events:
        fields = cc.check_calibrator_config(e.config, recorded=True)
        assert fields.get("batch") in (None, False) and e.cache is None and not e.tools


def test_the_grading_logs_record_grader_step_2_for_the_run_report_cross_check(done):
    led = ledger_of(done.env)
    for e in led["attempt_logs"]:
        log = read_eval_log(str(done.env.logs / e["log_file"]), header_only=True)
        m = log.eval.metadata
        assert m["split"] == "eval" and m["grading_job"] == led["job_id"]
        assert m["grader_prompt_version"] == "gp-1" and m["grader_prompt_sha256"] == gp.prompt_sha256()
        assert m["anchors_sha256"] == rubric.llm_anchor_sha256() and m["worker_run_commit"] == gc.RUN_COMMIT
        if e["role"] == "grader":
            assert m["grader_step"] == 2 and m["grader_model"] == grader.grader_pin(2)
            assert m["grader_generate_config"] == roster.grader_generate_config_args(2)
            assert m["part"] == e["part"] and m["parts"] == 4 and m["units"] == len(e["asked"])
            assert sorted(log.eval.model_roles or {}) == ["grader"]
    grader_log = read_eval_log(str(done.env.logs / led["attempt_logs"][0]["log_file"]))
    events = [e for _, e in grader.grader_events(grader_log)]
    assert events and all(grader.check_grader_config(e.config, 2)["batch"] is True for e in events)


def test_ledger_records_provenance_seeds_sampling_calls_batch_ids_and_python_version_before_the_first_call(done, world):
    first = done.spy.seen[0]
    assert first["task"] == REHEARSAL_TASK and all(first["plan_files"].values())
    led = first["ledger"]
    assert led is not None and led["ledger"] == "full-run grading attempt ledger" and led["run"] == "2026-11-21"
    prov = led["provenance"]
    assert prov["python_version"] == platform.python_version()
    assert prov["head"] == world.repo.head and prov["run_commit"] == world.repo.run_commit
    assert prov["archive_commit"] == world.repo.run_commit and prov["look_sha256"] == sha(world.repo.look_file)
    assert prov["worker_log_shas"] == sorted(world.shas.values()) and prov["seeds"] == gc.plan_seeds(world.shas)
    assert prov["grader_prompt_sha256"] == gp.prompt_sha256() and prov["anchors_sha256"] == rubric.llm_anchor_sha256()
    assert len(prov["plumbing_record_sha256"]) == 64 and prov["inspect_ai_version"]
    assert led["seeds"] == gc.plan_seeds(world.shas) and led["seed_rule"] == gc.SEED_RULE_TEXT
    assert led["sampling_calls"] == list(gc.SAMPLING_CALLS)
    assert led["batch_ids"] == [] and led["attempt_logs"] == [] and led["replies_of_record"] == {}
    assert led["rejected_logs"] == []
    assert led["attempts"] == [{**led["attempts"][0], "attempt": 1, "resume": False, "grader_max_samples": 200}]
    assert set(led["sealed_plan_files"]) == set(gc.PLAN_FILES)
    assert all(led["sealed_plan_files"][n] == sha(done.env.sealed / n) for n in gc.PLAN_FILES)
    assert led["job_id"] == gc.job_id(prov)
    assert led["log_dir"] == str(done.env.logs.resolve()) and led["sealed_dir"] == str(done.env.sealed.resolve())


# --- Pure helpers: asks, failed judgments and the eval plan ------------------------------------

def _entry(attempt, uids, kind="main", role="grader", log_file=None, sent=None):
    return {"attempt": attempt, "role": role, "eval_kind": kind, "asked": list(uids),
            "sent": list(uids if sent is None else sent), "log_file": log_file or f"a{attempt}-{kind}.eval"}


def test_only_units_sent_to_the_judge_count_as_asked():
    logs = [_entry(1, ["u0001", "u0002"], sent=[]), _entry(2, ["u0001", "u0002"], sent=["u0002"]),
            _entry(3, ["u0001"], "isolated", sent=[]), _entry(4, ["u0001"], "isolated", sent=[])]
    counts = gj.ask_counts(logs, "grader")
    assert "u0001" not in counts                                   # in four datasets, never sent: never asked
    assert counts["u0002"] == {"asks": 1, "isolated": False, "last_isolated": None}
    assert gj.with_failed_judgments({"grader": {}, "calibrator": {}}, logs)["grader"] == {}
    assert gj.plan_evals("grader-2", ["u0001", "u0002"], counts, rehearsal=False) == [("main", ["u0001", "u0002"])]
    legacy = [{k: v for k, v in e.items() if k != "sent"} for e in logs]
    assert gj.ask_counts(legacy, "grader") == {}                   # an entry with no sent list asks nothing


def _ev(event, role):
    return SimpleNamespace(event=event, role=role)


def test_sent_units_needs_the_sample_with_a_judge_role_model_event():
    log = SimpleNamespace(samples=[
        SimpleNamespace(id="u0003", events=[_ev("model", "grader")]),            # completed, errored or pending
        SimpleNamespace(id="u0001", events=[_ev("info", None), _ev("model", "grader")]),
        SimpleNamespace(id="u0002", events=[_ev("model", "calibrator")]),        # another role
        SimpleNamespace(id="u0004", events=[_ev("sample_init", None)]),          # started, never reached the judge
        SimpleNamespace(id="u0005", events=None)])
    asked = ["u0001", "u0002", "u0003", "u0004", "u0005", "u0006"]
    assert gj.sent_units(log, "grader", asked) == ["u0001", "u0003"]             # asked order
    assert gj.sent_units(log, "calibrator", asked) == ["u0002"]
    assert gj.sent_units(SimpleNamespace(samples=[]), "grader", asked) == []
    assert gj.sent_units(SimpleNamespace(samples=None), "grader", asked) == []


def test_ask_counts_and_failed_judgments_follow_the_three_asks_rule():
    logs = [_entry(1, ["u0001", "u0002"]), _entry(2, ["u0001"]), _entry(3, ["u0001"], "isolated"),
            _entry(1, ["u0003"], role="calibrator"), _entry(2, ["u0003"], role="calibrator"),
            _entry(3, ["u0003"], role="calibrator")]
    counts = gj.ask_counts(logs, "grader")
    assert counts["u0001"] == {"asks": 3, "isolated": True, "last_isolated": (3, "a3-isolated.eval")}
    assert counts["u0002"] == {"asks": 1, "isolated": False, "last_isolated": None} and "u0003" not in counts
    replies = gj.with_failed_judgments({"grader": {"u0002": gf.parsed(1, 1)}, "calibrator": {}}, logs)
    assert replies["grader"]["u0002"]["status"] == "parsed"
    assert replies["grader"]["u0001"] == gc.failed_judgment_record(attempt=3, log_file="a3-isolated.eval")
    assert replies["grader"]["u0001"] == {**gf.failed("failed", attempt=3, log_file="a3-isolated.eval")}
    assert replies["calibrator"] == {}                             # three asks, none isolated: asked again
    two = gj.with_failed_judgments({"grader": {}, "calibrator": {}}, logs[:2])
    assert two["grader"] == {}


def test_plan_evals_orders_rehearsal_isolated_and_main():
    todo = [f"u{i:04d}" for i in range(1, 21)]
    counts = {"u0003": {"asks": 2}, "u0004": {"asks": 1}, "u0019": {"asks": 3}}
    got = gj.plan_evals("grader-1", todo, counts, rehearsal=True)
    rest = [u for u in todo if u not in ("u0003", "u0019")]
    assert got == [("rehearsal", rest[:12]), ("isolated", ["u0003", "u0019"]), ("main", rest[12:])]
    assert gj.plan_evals("grader-2", todo, counts, rehearsal=True)[0][0] == "isolated"
    assert gj.plan_evals("grader-1", todo, {}, rehearsal=False) == [("main", todo)]
    assert gj.plan_evals("calibrator", ["u0003"], counts, rehearsal=False) == [("isolated", ["u0003"])]
    assert gj.eval_name("grader-2", "isolated") == "grading_grader_part_2_isolated"
    assert gj.eval_name("grader-1", "rehearsal") == REHEARSAL_TASK
    assert gj.eval_name("calibrator", "isolated") == "grading_calibrator_isolated"
    with pytest.raises(gc.Refused):
        gj.eval_name("grader-2", "rehearsal")


# --- Batch log hooks ----------------------------------------------------------------------------

def test_the_batch_log_capture_routes_created_failed_and_status_lines_and_resets_the_hooks():
    created, failures, progress = [], [], []
    with pytest.raises(RuntimeError):
        with gj.batch_log_capture(created.append, failures.append, progress.append):
            batch_log.log_batch("Created batch b-1 with 3 requests")
            batch_log.log_batch("Batch b-1 failed (obtaining results), failing all 3 requests in batch")
            batch_log.log_batch("Error 1 checking batch b-1. Error: boom")
            batch_log.log_batch("Batch b-1 completed")
            batch_log.emit_batch_status(batch_log.BatchStatus(1, 2, 1, 0, None))
            batch_log.emit_batch_status(batch_log.BatchStatus(1, 2, 1, 0, None))   # same counts: no new line
            raise RuntimeError("the eval died")
    assert created == [{"batch_id": "b-1", "requests": 3}]
    assert failures == ["Batch b-1 failed (obtaining results), failing all 3 requests in batch",
                        "Error 1 checking batch b-1. Error: boom"]
    assert progress[0] == "Batch b-1 completed" and len(progress) == 2
    assert "1 batches" in progress[1] and "pending 2" in progress[1] and "completed 1" in progress[1]
    assert "failed 0" in progress[1]
    assert batch_log._batch_log_callback is None and batch_log._batch_status_callback is None


def test_a_batch_older_than_25_hours_warns_once(capsys):
    import time

    old = int(time.time()) - 26 * 3600
    with gj.batch_log_capture(lambda b: None, lambda m: None, lambda m: None):
        batch_log.emit_batch_status(batch_log.BatchStatus(1, 1, 0, 0, old))
        batch_log.emit_batch_status(batch_log.BatchStatus(1, 1, 0, 0, old))
    assert capsys.readouterr().err.count("older than 25 hours") == 1


def test_batch_ids_and_batch_failures_are_saved_the_moment_they_are_logged(world, tmp_path):
    env = job_env(tmp_path)
    on_disk = {}

    def hook(task, *a, **k):
        batch_log.log_batch("Created batch b-9 with 5 requests")
        on_disk["batch_ids"] = ledger_of(env)["batch_ids"]
        batch_log.log_batch("Batch b-9 failed (obtaining results), failing all 5 requests in batch")
        raise RuntimeError("the process died while the batch was in flight")

    calls = Counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env, hook))
        res = run_main(live(world, env), gf.judge_roles(calls=calls), kw(world, env))
    assert res.code == gj.EXIT_CHECK, res.err
    (saved,) = on_disk["batch_ids"]                               # in the ledger before the eval returned
    assert {k: saved[k] for k in ("batch_id", "requests", "part", "eval_kind", "attempt")} == {
        "batch_id": "b-9", "requests": 5, "part": "grader-1", "eval_kind": "rehearsal", "attempt": 1}
    assert set(saved) == {"attempt", "part", "eval_kind", "batch_id", "requests", "time"}
    led = ledger_of(env)
    assert led["batch_ids"] == [saved]
    batch_failures = [f for f in led["failures"] if f["stage"] == "batch"]
    assert len(batch_failures) == 1 and "b-9 failed" in batch_failures[0]["error"]
    assert batch_failures[0]["part"] == "grader-1" and batch_failures[0]["eval_kind"] == "rehearsal"
    assert "batch: Batch b-9 failed" in res.err and sum(calls.values()) == 0
    assert batch_log._batch_log_callback is None and batch_log._batch_status_callback is None
    assert not lock_of(env).exists()
    assert_quiet(res)


# --- Dry run and arguments ----------------------------------------------------------------------

def _no_call(*a, **k):
    raise AssertionError("an eval was run")


def test_dry_run_makes_no_call_and_writes_nothing(world, planned, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    monkeypatch.setattr(gj, "run_eval", _no_call)
    code = gj.main(["--worker-logs", str(world.worker_dir)], model_roles=gf.judge_roles(), **kw(world, env))
    out = capsys.readouterr().out
    assert code == gj.EXIT_OK
    assert "dry run: no call made" in out and grader.grader_pin(2) in out and str(UNITS) in out
    assert not any(s in out for s in gf.LEAK_STRINGS)
    assert not env.ledger.exists() and not env.out.exists() and not env.logs.exists() and not env.sealed.exists()
    (line,) = [x for x in out.splitlines() if x.startswith("worker logs (SHA-256, in hash order): ")]
    shown = line.split(": ", 1)[1].split(", ")
    assert shown == sorted(world.shas.values()) and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shown)
    want = gc.sealed_plan_hashes(planned.plan, planned.logs)
    (line,) = [x for x in out.splitlines() if x.startswith("plan files (SHA-256): ")]
    assert line == "plan files (SHA-256): " + ", ".join(f"{n} {want[n]}" for n in gc.PLAN_FILES)
    assert "worker_log_shas and plan_file_shas" in out
    chars = planned.plan["counts"]["grader_prompt_chars"]
    per_unit = chars // UNITS // 4
    assert f"grader prompts: {chars} characters in all, about {per_unit} tokens per unit" in out
    assert f"--grader-max-samples 200, up to about {200 * per_unit} prompt tokens" in out
    assert gj.main(["--resume", "--worker-logs", str(world.worker_dir)], model_roles=gf.judge_roles(),
                   **kw(world, env)) == gj.EXIT_ARGS


def test_live_needs_both_folders_outside_the_repo(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    monkeypatch.setattr(gj, "run_eval", _no_call)
    k = kw(world, env)
    roles = gf.judge_roles()
    wl = ["--worker-logs", str(world.worker_dir)]
    assert gj.main(["--live", "--log-dir", str(env.logs), *wl], model_roles=roles, **k) == gj.EXIT_ARGS
    assert gj.main(["--live", "--sealed-dir", str(env.sealed), *wl], model_roles=roles, **k) == gj.EXIT_ARGS
    inside = ["--live", "--log-dir", str(world.repo.repo / "logs"), "--sealed-dir", str(env.sealed), *wl]
    assert gj.main(inside, model_roles=roles, **k) == gj.EXIT_ARGS
    inside = ["--live", "--log-dir", str(env.logs), "--sealed-dir", str(world.repo.repo / "x" / "_sealed"), *wl]
    assert gj.main(inside, model_roles=roles, **k) == gj.EXIT_ARGS
    named = ["--live", "--log-dir", str(env.logs), "--sealed-dir", str(tmp_path / "grading" / "sealed"), *wl]
    assert gj.main(named, model_roles=roles, **k) in (gj.EXIT_ARGS, gj.EXIT_REFUSED)
    assert not (env.ledger / gc.LEDGER_FILE).exists() and not lock_of(env).exists()


def test_grader_max_samples_below_one_is_a_bad_argument(world, tmp_path):
    env = job_env(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", _no_call)
        res = run_main(live(world, env, "--grader-max-samples", "0"), gf.judge_roles(), kw(world, env))
    assert res.code == gj.EXIT_ARGS and "--grader-max-samples" in res.err
    assert not (env.ledger / gc.LEDGER_FILE).exists()


def test_the_grader_max_samples_option_reaches_inspect_and_the_ledger(world, tmp_path):
    env = job_env(tmp_path)
    seen_logs = []

    def stop_after_rehearsal(task, *a, **k):
        if task.name != REHEARSAL_TASK:
            raise RuntimeError("stopped after the rehearsal")
        log = REAL_RUN_EVAL(task, *a, **k)
        seen_logs.append(Path(str(log.location)))
        return log

    spy = Spy(env, stop_after_rehearsal)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        res = run_main(live(world, env, "--grader-max-samples", "7"), gf.judge_roles(), kw(world, env))
    assert res.code == gj.EXIT_CHECK, res.err
    assert [s["max_samples"] for s in spy.seen] == [7, 7]
    assert read_eval_log(str(seen_logs[0]), header_only=True).eval.config.max_samples == 7
    led = ledger_of(env)
    assert led["attempts"][0]["grader_max_samples"] == 7
    assert led["attempt_logs"][0]["max_samples"] == 7 and led["attempt_logs"][0]["eval_kind"] == "rehearsal"
    assert_quiet(res)


# --- Start refusals -----------------------------------------------------------------------------

def _refused_before_any_call(world, env, monkeypatch, capsys, argv=None, **extra):
    monkeypatch.setattr(gj, "run_eval", _no_call)
    code = gj.main(argv or live(world, env), model_roles=gf.judge_roles(), **kw(world, env, **extra))
    captured = capsys.readouterr()
    assert code == gj.EXIT_REFUSED, captured.err
    assert not (env.ledger / gc.LEDGER_FILE).exists() and not lock_of(env).exists()
    assert not env.sealed.exists() or not [p for p in env.sealed.iterdir() if p.is_file()]
    assert_quiet(SimpleNamespace(out=captured.out, err=captured.err))
    return captured.err


def test_the_job_refuses_to_start_without_a_passed_plumbing_check(world, tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty-plumbing"
    empty.mkdir()
    err = _refused_before_any_call(world, job_env(tmp_path / "a"), monkeypatch, capsys, plumbing_dir=empty)
    assert "plumbing" in err
    failed = tmp_path / "failed-plumbing"
    shutil.copytree(world.plumbing, failed)
    (record,) = sorted(failed.glob("calibrator-plumbing-*.json"))
    body = json.loads(record.read_text(encoding="utf-8"))
    record.write_text(json.dumps({**body, "passed": False}), encoding="utf-8")
    _refused_before_any_call(world, job_env(tmp_path / "b"), monkeypatch, capsys, plumbing_dir=failed)


def test_the_job_refuses_to_start_without_the_start_checks(world, tmp_path, monkeypatch, capsys):
    dirty = gf.make_repo(tmp_path / "dirty")
    assert dirty.run_commit == world.repo.run_commit                   # the worker logs name this commit too
    (dirty.repo / "scripts" / "new.py").write_text("y = 1\n", encoding="utf-8")
    _refused_before_any_call(world, job_env(tmp_path / "e1"), monkeypatch, capsys, repo=dirty)
    ahead = gf.make_repo(tmp_path / "ahead")
    (ahead.repo / "README").write_text("r\n", encoding="utf-8", newline="\n")
    gf.commit_all(ahead.repo, "not pushed")
    _refused_before_any_call(world, job_env(tmp_path / "e2"), monkeypatch, capsys, repo=ahead)
    plain = gf.make_repo(tmp_path / "plain")
    _refused_before_any_call(world, job_env(tmp_path / "e3"), monkeypatch, capsys, repo=plain,
                             archive_file=plain.repo / "docs" / "evidence" / "grading" / "missing.json")
    failing = gf.make_repo(tmp_path / "look")
    failing.look_file.write_text(json.dumps(gf.look_body(verdict={"overall": "FAIL"})), encoding="utf-8",
                                 newline="\n")
    gf.commit_all(failing.repo, "a look file that did not pass")
    gf.git(failing.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _refused_before_any_call(world, job_env(tmp_path / "e4"), monkeypatch, capsys, repo=failing)


def test_a_worker_log_folder_with_a_changed_log_is_refused(world, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "wl"
    shutil.copytree(world.worker_dir, folder)
    victim = next((folder / gf.FOLDERS[0]).glob("*.eval"))
    log = read_eval_log(str(victim))
    log.eval.metadata = {**log.eval.metadata, "worker_prompt_sha256": "0" * 64}
    write_eval_log(log, str(victim))
    env = job_env(tmp_path / "env")
    monkeypatch.setattr(gj, "run_eval", _no_call)
    argv = ["--live", "--log-dir", str(env.logs), "--sealed-dir", str(env.sealed), "--worker-logs", str(folder)]
    assert gj.main(argv, model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refused" in err and victim.name in err and not any(s in err for s in gf.MODELS)
    assert not (env.ledger / gc.LEDGER_FILE).exists()


def test_the_grading_logs_must_sit_inside_the_sealed_folder(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    argv = live(world, env, logs=env.root / "grading" / "logs")
    err = _refused_before_any_call(world, env, monkeypatch, capsys, argv=argv)
    assert "_sealed" in err


def test_a_log_folder_shared_with_the_plumbing_check_or_holding_eval_files_is_refused(world, tmp_path, monkeypatch,
                                                                                         capsys):
    env = job_env(tmp_path / "shared")
    records = tmp_path / "shared-plumbing"
    assert run_plumbing(tmp_path, world.repo, log_dir=env.logs, record_dir=records) == cpc.EXIT_OK
    err = _refused_before_any_call(world, env, monkeypatch, capsys, plumbing_dir=records)
    assert "already holds .eval files" in err or "plumbing check's log folder" in err
    env = job_env(tmp_path / "stray")
    env.logs.mkdir(parents=True)
    (env.logs / "x.eval").write_bytes(b"not a log")
    err = _refused_before_any_call(world, env, monkeypatch, capsys)
    assert "already holds .eval files" in err


def test_a_sealed_folder_that_holds_the_worker_logs_is_refused(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    worker_copy = env.sealed / "worker-logs"
    shutil.copytree(world.worker_dir, worker_copy)
    argv = ["--live", "--log-dir", str(env.logs), "--sealed-dir", str(env.sealed), "--worker-logs", str(worker_copy)]
    monkeypatch.setattr(gj, "run_eval", _no_call)
    assert gj.main(argv, model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "worker log folder" in err and not (env.ledger / gc.LEDGER_FILE).exists()
    assert not any((env.sealed / n).exists() for n in gc.PLAN_FILES)


def test_a_held_lock_refuses_a_second_live_job(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    env.ledger.mkdir(parents=True)
    lock_of(env).write_bytes(b'{"pid": 1}\n')
    monkeypatch.setattr(gj, "run_eval", _no_call)
    assert gj.main(live(world, env), model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "grading-2026-11-21.lock" in err and "author removes it" in err
    assert not (env.ledger / gc.LEDGER_FILE).exists() and lock_of(env).read_bytes() == b'{"pid": 1}\n'
    assert gj.main(live(world, env, "--resume"), model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED


def _archive_with(repo_env, **extra) -> None:
    body = json.loads(repo_env.archive_file.read_text(encoding="utf-8"))
    repo_env.archive_file.write_text(json.dumps({**body, **extra}), encoding="utf-8", newline="\n")
    gf.commit_all(repo_env.repo, "archive record with hashes")
    gf.git(repo_env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")


def test_an_archive_record_with_other_worker_log_hashes_is_refused(world, tmp_path, monkeypatch, capsys):
    scratch = gf.make_repo(tmp_path / "r")
    assert scratch.head == world.repo.head
    _archive_with(scratch, worker_log_shas=sorted(f"{i:064x}" for i in range(1, 5)))
    err = _refused_before_any_call(world, job_env(tmp_path / "e"), monkeypatch, capsys, repo=scratch)
    assert "other worker-log or plan-file hashes" in err


def test_an_archive_record_with_the_right_hashes_is_accepted_without_a_note(world, planned, tmp_path):
    scratch = gf.make_repo(tmp_path / "r")
    _archive_with(scratch, worker_log_shas=sorted(world.shas.values()),
                  plan_file_shas=gc.sealed_plan_hashes(planned.plan, planned.logs))
    env = job_env(tmp_path / "e")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env, _no_call))
        res = run_main(live(world, env), gf.judge_roles(), kw(world, env, repo=scratch))
    assert res.code == gj.EXIT_CHECK, res.err                  # past every start check, stopped at the first eval
    assert "does not list the worker-log and plan-file hashes" not in res.err
    assert ledger_of(env)["failures"][0]["stage"] == "grader eval"


# --- Crash and resume (D-R23, reply of record) --------------------------------------------------

@pytest.fixture(scope="module")
def crashed_grader(world, tmp_path_factory):
    """Grader part 1 (after its rehearsal) dies after 5 completed replies, the first of them
    unparsed. Then: a fresh start, a resume with another log folder, and the real resume."""
    env = job_env(tmp_path_factory.mktemp("crash-grader"))
    forced: set = set()
    calls = Counter()
    roles = gf.judge_roles(forced_plan(forced), calls)
    hook, state = crash_after(gc.GRADER_EVAL_NAME.format(1), 5, role="grader", forced=forced)
    spy = Spy(env, hook)
    out = SimpleNamespace(env=env, state=state, forced=forced, spy=spy)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        out.first = run_main(live(world, env), roles, kw(world, env))
        out.lock_after_crash = lock_of(env).exists()
        out.ledger_after_crash = ledger_of(env)
        before = sum(calls.values())
        out.fresh = run_main(live(world, env), roles, kw(world, env))
        out.other = run_main(live(world, env, "--resume", logs=env.root / "other-logs"), roles, kw(world, env))
        out.calls_between = sum(calls.values()) - before
        calls.clear()
        out.seen_before_resume = len(spy.seen)
        out.resume = run_main(live(world, env, "--resume"), roles, kw(world, env))
    out.calls = calls
    return out


def test_resume_does_not_reask_an_unparsed_reply_from_a_crashed_grader_attempt(crashed_grader):
    cg = crashed_grader
    assert cg.first.code == gj.EXIT_CHECK and "FAIL" in cg.first.err and "--resume" in cg.first.err
    assert_quiet(cg.first)
    assert not cg.lock_after_crash
    led = cg.ledger_after_crash
    rehearsal, entry = led["attempt_logs"]
    assert rehearsal["eval_kind"] == "rehearsal" and rehearsal["status"] == "success"
    assert entry["status"] != "success" and entry["part"] == "grader-1" and entry["completed"] == 5
    assert entry["eval_kind"] == "main" and entry["log_sha256"] == sha(cg.state["log"])
    assert led["failures"] and led["failures"][0]["part"] == "grader-1" and led["failures"][0]["attempt"] == 1
    assert led["failures"][0]["eval_kind"] == "main" and "the eval ended with status 'error'" in led["failures"][0]["error"]
    assert set(led["replies_of_record"]) == set(cg.state["kept"]) | set(rehearsal["asked"])
    assert cg.resume.code == gj.EXIT_OK, cg.resume.err
    assert_quiet(cg.resume)
    keys = unit_keys(cg.env)
    for u in cg.state["kept"] + rehearsal["asked"]:
        assert cg.calls[("grader", *keys[u])] == 0                      # never asked again
    replies = sealed_of(cg.env, gc.SEALED_GRADER_REPLIES)["replies"]
    forced_uid = next(u for u, k in keys.items() if k == cg.state["forced_key"])
    assert forced_uid == cg.state["kept"][0]
    assert replies[forced_uid]["status"] == "unparsed" and replies[forced_uid]["attempt"] == 1
    assert replies[forced_uid]["reply"] == "garbled reply, not JSON"
    assert all(replies[u]["attempt"] == 1 for u in cg.state["kept"])
    final = ledger_of(cg.env)
    assert [a["resume"] for a in final["attempts"]][:1] == [False] and final["attempts"][-1]["resume"] is True
    assert final["attempts"][-1]["kept"] == 5 + REHEARSAL and final["attempts"][-1]["grader_max_samples"] == 200
    extra = 0 if (gf.case_index(cg.state["forced_key"][1]), cg.state["forced_key"][0]) == (3, gf.CHARLIE) else 1
    ev = json.loads(evidence_path(cg.env).read_text(encoding="utf-8"))
    assert ev["counts"]["grader"]["unparsed"] == 4 + extra and ev["counts"]["grader"]["attempt_logs"] == 6
    assert len(ev["attempts"]) == len(final["attempts"])


def test_a_resume_makes_no_rehearsal(crashed_grader):
    cg = crashed_grader
    resumed = [s["task"] for s in cg.spy.seen[cg.seen_before_resume:]]
    assert REHEARSAL_TASK not in resumed and resumed[0] == gc.GRADER_EVAL_NAME.format(1)
    kinds = [(e["attempt"], e["eval_kind"]) for e in ledger_of(cg.env)["attempt_logs"]]
    assert [k for k in kinds if k[1] == "rehearsal"] == [(1, "rehearsal")]


def test_outputs_of_a_dead_batch_that_no_log_recorded_are_asked_on_resume(crashed_grader):
    cg = crashed_grader
    keys = unit_keys(cg.env)
    replies = sealed_of(cg.env, gc.SEALED_GRADER_REPLIES)["replies"]
    assert cg.state["dropped"]
    for u in cg.state["dropped"]:
        assert cg.calls[("grader", *keys[u])] == 1 and replies[u]["attempt"] > 1
    asked_again = [e for e in ledger_of(cg.env)["attempt_logs"] if e["part"] == "grader-1" and e["attempt"] > 1]
    assert len(asked_again) == 1 and asked_again[0]["asked"] == cg.state["dropped"]
    (crashed,) = [e for e in ledger_of(cg.env)["attempt_logs"] if e["part"] == "grader-1" and e["attempt"] == 1
                  and e["eval_kind"] == "main"]
    assert crashed["sent"] == cg.state["kept"]                        # dropped units were not sent by that log
    grader_calls = {k: v for k, v in cg.calls.items() if k[0] == "grader"}
    assert sum(grader_calls.values()) == UNITS - 5 - REHEARSAL


def test_a_fresh_start_is_refused_while_a_ledger_exists_and_resume_needs_the_same_folders(crashed_grader):
    cg = crashed_grader
    assert cg.fresh.code == gj.EXIT_REFUSED
    assert "already started" in cg.fresh.err or "--resume" in cg.fresh.err or "never overwritten" in cg.fresh.err
    assert cg.other.code == gj.EXIT_ARGS
    assert cg.calls_between == 0
    assert_quiet(cg.fresh)
    assert_quiet(cg.other)


@pytest.fixture(scope="module")
def crashed_calibrator(world, tmp_path_factory):
    env = job_env(tmp_path_factory.mktemp("crash-calibrator"))
    forced: set = set()
    calls = Counter()
    roles = gf.judge_roles(forced_plan(forced), calls)
    hook, state = crash_after(gc.CALIBRATOR_EVAL_NAME, 5, role="calibrator", forced=forced)
    out = SimpleNamespace(env=env, state=state)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env, hook))
        out.first = run_main(live(world, env), roles, kw(world, env))
        calls.clear()
        out.resume = run_main(live(world, env, "--resume"), roles, kw(world, env))
    out.calls = calls
    return out


def test_resume_does_not_reask_an_unparsed_reply_from_a_crashed_calibrator_attempt(crashed_calibrator):
    cc_ = crashed_calibrator
    assert cc_.first.code == gj.EXIT_CHECK, cc_.first.err
    assert cc_.resume.code == gj.EXIT_OK, cc_.resume.err
    assert_quiet(cc_.first)
    keys = unit_keys(cc_.env)
    assert not any(k[0] == "grader" for k in cc_.calls)                   # every grader part was done
    for u in cc_.state["kept"]:
        assert cc_.calls[("calibrator", *keys[u])] == 0
    for u in cc_.state["dropped"]:
        assert cc_.calls[("calibrator", *keys[u])] == 1
    replies = sealed_of(cc_.env, gc.SEALED_CALIBRATOR_REPLIES)["replies"]
    first = cc_.state["kept"][0]
    assert replies[first]["status"] == "unparsed" and replies[first]["attempt"] == 1
    assert set(replies) == set(cc_.state["kept"]) | set(cc_.state["dropped"])
    assert len(replies) == ROWS_DRAWN


def _crash_part_1(world, env, roles=None, repo=None):
    hook, state = crash_after(gc.GRADER_EVAL_NAME.format(1), 5)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env, hook))
        first = run_main(live(world, env), roles or gf.judge_roles(), kw(world, env, repo=repo))
    assert first.code == gj.EXIT_CHECK, first.err
    return state


def _resume_no_call(world, env, calls, repo=None):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", _no_call)
        return run_main(live(world, env, "--resume"), gf.judge_roles(calls=calls), kw(world, env, repo=repo))


def test_a_unit_completed_in_two_attempt_logs_is_refused(world, tmp_path):
    env = job_env(tmp_path)
    state = _crash_part_1(world, env)
    copy = env.logs / ("copy-" + state["log"].name)
    shutil.copyfile(state["log"], copy)
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED and "asked twice" in res.err and "FAIL: log check" in res.err
    assert sum(calls.values()) == 0 and not evidence_path(env).exists()
    (rejected,) = ledger_of(env)["rejected_logs"]
    assert rejected["log_file"] == copy.name and rejected["log_sha256"] == sha(copy)
    assert rejected["reason"] == "failed its check" and rejected["found_on_resume"] is True
    assert_quiet(res)


def test_an_orphan_log_of_this_job_is_admitted_and_a_foreign_log_is_refused(world, tmp_path):
    env = job_env(tmp_path / "orphan")
    died = []

    def die(task, *a, **k):
        log = REAL_RUN_EVAL(task, *a, **k)
        if task.name == gc.GRADER_EVAL_NAME.format(2) and not died:
            died.append(Path(str(log.location)).name)
            raise RuntimeError("the process died after the log was written")
        return log

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env, die))
        first = run_main(live(world, env), gf.judge_roles(), kw(world, env))
    assert first.code != gj.EXIT_OK and died
    assert_quiet(first)
    assert died[0] not in {e["log_file"] for e in ledger_of(env)["attempt_logs"]}
    calls = Counter()
    res = run_main(live(world, env, "--resume"), gf.judge_roles(calls=calls), kw(world, env))
    assert res.code == gj.EXIT_OK, res.err
    led = ledger_of(env)
    (orphan,) = [e for e in led["attempt_logs"] if e["found_on_resume"]]
    assert orphan["log_file"] == died[0] and orphan["part"] == "grader-2" and orphan["attempt"] == 2
    assert orphan["eval_kind"] == "main" and orphan["max_samples"] == 200 and set(orphan) == ATTEMPT_LOG_KEYS
    keys = unit_keys(env)
    assert all(calls[("grader", *keys[u])] == 0 for u in led["parts"]["grader-2"])
    assert all(calls[("grader", *keys[u])] == 0 for u in led["parts"]["grader-1"])
    assert all(calls[("grader", *keys[u])] == 1 for k in ("grader-3", "grader-4") for u in led["parts"][k])
    # a log of another job in the folder is refused
    env = job_env(tmp_path / "foreign")
    _crash_part_1(world, env)
    shutil.copyfile(world.paths[gf.ALPHA], env.logs / "foreign.eval")
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code in (gj.EXIT_CHECK, gj.EXIT_REFUSED) and "foreign log" in res.err
    assert sum(calls.values()) == 0 and ledger_of(env)["rejected_logs"] == []


def test_an_orphan_log_that_asks_units_outside_its_part_is_rejected(world, tmp_path):
    env = job_env(tmp_path)
    state = _crash_part_1(world, env)
    log = read_eval_log(str(state["log"]))
    log.eval.metadata = {**log.eval.metadata, "part": "grader-2"}
    moved = env.logs / "moved-part.eval"
    write_eval_log(log, str(moved))
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED and "outside its part" in res.err and LOG_CHECK_HINT in res.err
    assert sum(calls.values()) == 0
    (rejected,) = ledger_of(env)["rejected_logs"]
    assert rejected["log_file"] == moved.name and rejected["log_sha256"] == sha(moved)
    assert rejected["part"] == "grader-2" and rejected["stage"] == "log check"
    again = _resume_no_call(world, env, calls)
    assert again.code == gj.EXIT_REFUSED and "rejected attempt logs" in again.err and sum(calls.values()) == 0
    assert_quiet(res)


def test_a_part_written_log_in_the_folder_is_rejected_on_resume_not_a_crash(world, tmp_path):
    env = job_env(tmp_path)
    _crash_part_1(world, env)
    bad = env.logs / "part-written.eval"
    bad.write_bytes(b"PK\x03\x04 a log a killed process left part-written")
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED, res.err
    assert "part-written.eval" in res.err and "cannot be read" in res.err
    assert sum(calls.values()) == 0 and not evidence_path(env).exists()
    led = ledger_of(env)
    (rejected,) = led["rejected_logs"]
    assert rejected["log_file"] == "part-written.eval" and rejected["reason"] == "unreadable"
    assert rejected["log_sha256"] == sha(bad) and rejected["part"] is None and rejected["role"] is None
    assert led["failures"][-1]["stage"] == "log check" and led["failures"][-1]["log_sha256"] == sha(bad)
    assert led["attempts"][-1]["resume"] is True and rejected["attempt"] == led["attempts"][-1]["attempt"]
    again = _resume_no_call(world, env, calls)
    assert again.code == gj.EXIT_REFUSED and "1 rejected attempt logs" in again.err and sum(calls.values()) == 0
    assert_quiet(res)
    assert_quiet(again)


def test_resume_refuses_a_changed_attempt_log_or_plan_file(world, tmp_path):
    env = job_env(tmp_path / "log")
    state = _crash_part_1(world, env)
    state["log"].write_bytes(state["log"].read_bytes() + b"x")
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code in (gj.EXIT_CHECK, gj.EXIT_REFUSED) and sum(calls.values()) == 0
    assert_quiet(res)
    env = job_env(tmp_path / "plan")
    _crash_part_1(world, env)
    path = env.sealed / gc.SEALED_DRAWN
    planted = path.read_bytes().replace(b'"rows"', b'"rows_changed"')
    path.write_bytes(planted)
    res = _resume_no_call(world, env, calls)
    assert res.code in (gj.EXIT_CHECK, gj.EXIT_REFUSED) and sum(calls.values()) == 0
    assert path.read_bytes() == planted


def test_resume_under_a_changed_head_is_refused_with_no_call(world, tmp_path):
    scratch = gf.make_repo(tmp_path / "r")
    assert scratch.head == world.repo.head
    env = job_env(tmp_path / "e")
    _crash_part_1(world, env, repo=scratch)
    (scratch.repo / "README").write_text("a later fix\n", encoding="utf-8", newline="\n")
    gf.commit_all(scratch.repo, "a code fix after the start")
    gf.git(scratch.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    calls = Counter()
    res = _resume_no_call(world, env, calls, repo=scratch)
    assert res.code == gj.EXIT_REFUSED and "head" in res.err and "tied to HEAD" in res.err
    assert sum(calls.values()) == 0
    assert_quiet(res)


def test_a_fresh_start_refuses_an_existing_result_or_evidence_file(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path / "queue")
    env.sealed.mkdir(parents=True)
    queue = env.sealed / gc.SEALED_QUEUE
    queue.write_bytes(b"{}\n")
    monkeypatch.setattr(gj, "run_eval", _no_call)
    assert gj.main(live(world, env), model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    assert "never overwritten" in capsys.readouterr().err
    assert queue.read_bytes() == b"{}\n" and not (env.ledger / gc.LEDGER_FILE).exists()
    assert not any((env.sealed / n).exists() for n in gc.PLAN_FILES)
    env = job_env(tmp_path / "evidence")
    env.out.mkdir(parents=True)
    evidence_path(env).write_bytes(b"{}\n")
    assert gj.main(live(world, env), model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    assert evidence_path(env).read_bytes() == b"{}\n" and not (env.ledger / gc.LEDGER_FILE).exists()


def test_an_unreadable_ledger_is_refused_on_resume(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    env.ledger.mkdir(parents=True)
    (env.ledger / gc.LEDGER_FILE).write_bytes(b"{")
    monkeypatch.setattr(gj, "run_eval", _no_call)
    assert gj.main(live(world, env, "--resume"), model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_REFUSED
    assert "cannot be read" in capsys.readouterr().err and not lock_of(env).exists()


def test_a_missing_evidence_file_is_written_again_from_the_ledger(world, planned, tmp_path):
    env = job_env(tmp_path)
    env.sealed.mkdir(parents=True)
    plan_file = env.sealed / gc.SEALED_UNIT_MAP                      # left by a kill before the ledger existed
    plan_file.write_bytes(gc.canonical_bytes(gc.sealed_plan_bodies(planned.plan, planned.logs)[gc.SEALED_UNIT_MAP]))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", Spy(env))
        res = run_main(live(world, env), gf.judge_roles(), kw(world, env))
    assert res.code == gj.EXIT_OK, res.err
    first = evidence_path(env).read_bytes()
    evidence_path(env).unlink()
    calls = Counter()
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_OK, res.err
    assert "written again from the ledger" in res.out and evidence_path(env).read_bytes() == first
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED and "not run again" in res.err and sum(calls.values()) == 0
    evidence_path(env).write_bytes(b"{}\n")
    res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED and evidence_path(env).read_bytes() == b"{}\n"
    assert_quiet(res)


def test_a_grader_input_that_differs_from_the_plan_is_rejected(world, tmp_path):
    env = job_env(tmp_path)

    def tamper(task, *a, **k):
        if task.name == REHEARSAL_TASK:
            s = task.dataset[0]
            s.input = [*s.input, ChatMessageUser(content="an extra message the plan never built")]
        return REAL_RUN_EVAL(task, *a, **k)

    spy = Spy(env, tamper)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        res = run_main(live(world, env), gf.judge_roles(), kw(world, env))
    assert res.code == gj.EXIT_REFUSED and "FAIL: log check" in res.err and LOG_CHECK_HINT in res.err
    assert [s["task"] for s in spy.seen] == [REHEARSAL_TASK]
    led = ledger_of(env)
    assert led["attempt_logs"] == [] and led["replies_of_record"] == {}
    (rejected,) = led["rejected_logs"]
    (log_file,) = env.logs.glob("*.eval")
    assert rejected["log_file"] == log_file.name and rejected["log_sha256"] == sha(log_file)
    assert rejected["stage"] == "log check" and rejected["reason"] == "failed its check"
    assert rejected["part"] == "grader-1" and rejected["eval_kind"] == "rehearsal" and rejected["found_on_resume"] is False
    assert led["failures"][-1]["part"] == "grader-1" and led["failures"][-1]["stage"] == "log check"
    assert not evidence_path(env).exists() and not lock_of(env).exists()
    assert_quiet(res)
    calls = Counter()
    again = _resume_no_call(world, env, calls)
    assert again.code == gj.EXIT_REFUSED and "rejected attempt logs" in again.err and sum(calls.values()) == 0


# --- Failed judgments (D-R23, failed judgments) -------------------------------------------------

def _raising_plan(role: str, key: tuple):
    def plan(r, model, sid, epoch):
        if r == role and (model, sid, epoch) == key:
            return gf.RAISE
        return gf.default_plan(r, model, sid, epoch)

    return plan


def _parses_by_default(role: str, key: tuple) -> bool:
    return not isinstance(gf.default_plan(role, *key), str)


def _unit_key(plan, uid) -> tuple:
    u = plan["units"][uid]
    return (u["model"], u["sample_id"], u["epoch"])


def _never_completes(world, env, role: str, key: tuple):
    calls = Counter()
    roles = gf.judge_roles(_raising_plan(role, key), calls)
    spy = Spy(env)
    out = SimpleNamespace(env=env, calls=calls, spy=spy, key=key, runs=[], ledgers=[], seen=[])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        for extra in ((), ("--resume",), ("--resume",)):
            start = len(spy.seen)
            out.runs.append(run_main(live(world, env, *extra), roles, kw(world, env)))
            out.ledgers.append(ledger_of(env))
            out.seen.append(spy.seen[start:])
        out.after = _resume_no_call(world, env, Counter())
    return out


@pytest.fixture(scope="module")
def grader_never_completes(world, planned, tmp_path_factory):
    uid = next(u for u in planned.plan["parts"]["grader-2"]
               if _parses_by_default("grader", _unit_key(planned.plan, u)))
    out = _never_completes(world, job_env(tmp_path_factory.mktemp("never-grader")), "grader",
                           _unit_key(planned.plan, uid))
    out.uid = uid
    return out


@pytest.fixture(scope="module")
def calibrator_never_completes(world, planned, tmp_path_factory):
    uid = next(u for u in planned.plan["parts"]["calibrator"]
               if _parses_by_default("grader", _unit_key(planned.plan, u))
               and _parses_by_default("calibrator", _unit_key(planned.plan, u)))
    out = _never_completes(world, job_env(tmp_path_factory.mktemp("never-calibrator")), "calibrator",
                           _unit_key(planned.plan, uid))
    out.uid = uid
    return out


def test_a_grader_call_that_always_raises_ends_as_a_failed_judgment(grader_never_completes, planned):
    g = grader_never_completes
    uid, part = g.uid, planned.plan["parts"]["grader-2"]
    first, second, third = g.runs
    assert first.code == gj.EXIT_CHECK and "1 samples errored" in first.err and "mock judge call failed" in first.err
    ror = g.ledgers[0]["replies_of_record"]
    assert all("grader" in ror[u] for u in part if u != uid) and uid not in ror     # the rest of the part is kept
    assert second.code == gj.EXIT_CHECK
    assert [(s["task"], s["ids"]) for s in g.seen[1]] == [(gc.GRADER_EVAL_NAME.format(2), [uid])]
    assert g.ledgers[1]["attempt_logs"][-1]["eval_kind"] == "main"
    assert "warning" not in first.err and gj.final_ask_warning(1, "grader-2") in second.err
    assert third.code == gj.EXIT_OK, third.err
    assert not any(f["stage"] == gj.STAGE_FINAL_ASK_GUARD for f in ledger_of(g.env)["failures"])
    assert [a.get("allow_final_ask") for a in ledger_of(g.env)["attempts"]] == [None, False, False]
    iso = g.seen[2][0]
    assert iso["task"] == "grading_grader_part_2_isolated" and iso["ids"] == [uid] and iso["max_samples"] == 1
    assert iso["metadata"]["eval_kind"] == "isolated" and iso["metadata"]["units"] == 1
    led = ledger_of(g.env)
    (iso_entry,) = [e for e in led["attempt_logs"] if e["eval_kind"] == "isolated"]
    assert iso_entry["asked"] == [uid] and iso_entry["max_samples"] == 1 and iso_entry["status"] != "success"
    assert iso_entry["sent"] == [uid]                                   # the isolated ask reached the judge
    assert sum(uid in e["sent"] for e in led["attempt_logs"] if e["role"] == "grader") == 3
    assert g.calls[("grader", *g.key)] == 3
    keys = unit_keys(g.env)
    assert all(g.calls[("grader", *keys[u])] == 1 for u in part if u != uid)        # never asked again
    reply = sealed_of(g.env, gc.SEALED_GRADER_REPLIES)["replies"][uid]
    assert reply["status"] == "failed" and reply["reply"] is None and reply["error"] == gf.FAILED_JUDGMENT_ERROR
    assert reply["attempt"] == 3 and reply["log_file"] == iso_entry["log_file"]
    assert led["replies_of_record"][uid]["grader"] == {"attempt": 3, "log_file": iso_entry["log_file"]}
    item = next(i for i in sealed_of(g.env, gc.SEALED_QUEUE)["items"] if i["unit_id"] == uid)
    assert gs.TAG_GRADER_FAILED in item["tags"]
    ev = json.loads(evidence_path(g.env).read_text(encoding="utf-8"))
    assert ev["counts"]["grader"]["failed"] == 1 and ev["counts"]["grader"]["parsed"] == UNITS - 5
    assert "1 failed" in third.out
    assert g.after.code == gj.EXIT_REFUSED and "not run again" in g.after.err
    for res in (*g.runs, g.after):
        assert_quiet(res)


def test_a_calibrator_call_that_always_raises_ends_as_a_failed_judgment_and_a_dropped_pair(
        calibrator_never_completes):
    c = calibrator_never_completes
    first, second, third = c.runs
    assert first.code == gj.EXIT_CHECK and second.code == gj.EXIT_CHECK and third.code == gj.EXIT_OK, third.err
    iso = c.seen[2][0]
    assert iso["task"] == "grading_calibrator_isolated" and iso["ids"] == [c.uid] and iso["max_samples"] == 1
    assert c.calls[("calibrator", *c.key)] == 3
    reply = sealed_of(c.env, gc.SEALED_CALIBRATOR_REPLIES)["replies"][c.uid]
    assert reply["status"] == "failed" and reply["reply"] is None and reply["error"] == gf.FAILED_JUDGMENT_ERROR
    agg = sealed_of(c.env, gc.SEALED_AGGREGATES)
    assert agg["gate"]["pairs_dropped"] == 4 and agg["gate"]["pairs_complete"] == ROWS_DRAWN - 4
    ev = json.loads(evidence_path(c.env).read_text(encoding="utf-8"))
    assert ev["counts"]["calibrator"]["failed"] == 1 and ev["counts"]["calibrator"]["pairs_dropped"] == 4
    item = next(i for i in sealed_of(c.env, gc.SEALED_QUEUE)["items"] if i["unit_id"] == c.uid)
    assert gs.TAG_CALIBRATOR_FAILED in item["tags"]
    for res in (*c.runs, c.after):
        assert_quiet(res)


def test_evals_cut_off_before_any_sample_reached_the_judge_never_make_a_failed_judgment(world, tmp_path):
    env = job_env(tmp_path)
    part_task = gc.GRADER_EVAL_NAME.format(2)

    def interrupted(task, *a, **k):
        if not task.name.startswith(part_task):
            return REAL_RUN_EVAL(task, *a, **k)
        log = REAL_RUN_EVAL(task, *a, **k)
        log.samples = []                     # cut off (Ctrl+C, sleep, restart) before any sample was written
        log.status = "cancelled"
        write_eval_log(log, log.location)
        raise RuntimeError("interrupted")

    spy = Spy(env, interrupted)
    runs = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        for extra in ((), ("--resume",), ("--resume",)):
            runs.append(run_main(live(world, env, *extra), gf.judge_roles(), kw(world, env)))
    assert [r.code for r in runs] == [gj.EXIT_CHECK] * 3
    part = ledger_of(env)["parts"]["grader-2"]
    tried = [s for s in spy.seen if s["task"].startswith(part_task)]
    assert [(s["task"], s["ids"], s["max_samples"]) for s in tried] == [(part_task, part, 200)] * 3   # never isolated
    calls = Counter()
    spy = Spy(env)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        final = run_main(live(world, env, "--resume"), gf.judge_roles(calls=calls), kw(world, env))
    assert final.code == gj.EXIT_OK, final.err
    assert [(s["task"], s["ids"]) for s in spy.seen] == [(part_task, part), *(
        (gc.GRADER_EVAL_NAME.format(i), ledger_of(env)["parts"][f"grader-{i}"]) for i in (3, 4)),
        (gc.CALIBRATOR_EVAL_NAME, ledger_of(env)["parts"]["calibrator"])]
    led = ledger_of(env)
    cut = [e for e in led["attempt_logs"] if e["part"] == "grader-2" and e["completed"] == 0]
    assert len(cut) == 3 and all(e["asked"] == part and e["sent"] == [] and e["eval_kind"] == "main" for e in cut)
    assert not any(e["eval_kind"] == "isolated" for e in led["attempt_logs"])
    replies = sealed_of(env, gc.SEALED_GRADER_REPLIES)["replies"]
    assert not any(r["status"] == "failed" for r in replies.values())
    keys = unit_keys(env)
    assert all(calls[("grader", *keys[u])] == 1 for u in part)        # each unit of the part asked once, at last
    ev = json.loads(evidence_path(env).read_text(encoding="utf-8"))
    assert ev["counts"]["grader"]["failed"] == 0 and ev["counts"]["calibrator"]["failed"] == 0
    for res in (*runs, final):
        assert_quiet(res)


# --- The final ask guard (D-R23, failed judgments) ----------------------------------------------

def _log_entry(uids, *, completed, role="grader", log_file="a.eval", sent=None, status="error"):
    return {"role": role, "asked": list(uids), "sent": list(uids if sent is None else sent), "completed": completed,
            "log_file": log_file, "status": status}


def test_the_final_ask_guard_stops_on_many_due_units_or_a_whole_failed_eval():
    part = [f"u{i:04d}" for i in range(1, 80)]
    normal = [_log_entry(part, completed=60)]
    assert gj.FINAL_ASK_GUARD_MAX == 12
    assert gj.final_ask_guard(normal, "grader", part[:12]) is None                   # 12 due, a normal last eval
    assert "more than 12" in gj.final_ask_guard(normal, "grader", part[:13])
    whole = [_log_entry(part, completed=0)]                                          # every call of the part failed
    assert "completed no call" in gj.final_ask_guard(whole, "grader", part[:12])
    assert gj.final_ask_guard(whole, "grader", part) is not None
    alone = [_log_entry(part, completed=78), _log_entry(["u0007"], completed=0)]     # one bad unit asked alone
    assert gj.final_ask_guard(alone, "grader", ["u0007"]) is None
    three = [_log_entry(part, completed=76), _log_entry(part[:3], completed=0)]
    assert gj.final_ask_guard(three, "grader", part[:3]) is None
    cut = [*whole, _log_entry(part, completed=0, sent=[])]          # a log that sent nothing is passed over
    assert gj.final_ask_guard(cut, "grader", part[:3]) is not None
    other_role = [*whole, _log_entry(part[:5], completed=5, role="calibrator")]
    assert gj.final_ask_guard(other_role, "grader", part[:3]) is not None
    assert gj.final_ask_guard(other_role, "calibrator", part[:3]) is None
    assert gj.final_ask_guard([], "calibrator", part[:3]) is None
    assert gj.STAGE_FINAL_ASK_GUARD in gj.STAGE_HINTS and "--allow-final-ask" in gj.STAGE_HINTS["final ask guard"]


def test_the_rehearsal_is_due_until_a_grader_log_ends_in_success():
    assert gj.rehearsal_due([]) is True
    assert gj.rehearsal_due([_log_entry(["u0001"], completed=0)]) is True
    assert gj.rehearsal_due([_log_entry(["u0001"], completed=1, role="calibrator", status="success")]) is True
    assert gj.rehearsal_due([_log_entry(["u0001"], completed=1, status="success")]) is False


def test_allow_final_ask_needs_live_and_resume(world, tmp_path, monkeypatch, capsys):
    env = job_env(tmp_path)
    monkeypatch.setattr(gj, "run_eval", _no_call)
    for argv in (["--allow-final-ask", "--worker-logs", str(world.worker_dir)], live(world, env, "--allow-final-ask")):
        assert gj.main(argv, model_roles=gf.judge_roles(), **kw(world, env)) == gj.EXIT_ARGS
        assert "--allow-final-ask is valid only with --live --resume" in capsys.readouterr().err
    assert not (env.ledger / gc.LEDGER_FILE).exists() and not env.sealed.exists()


@pytest.fixture(scope="module")
def outage(world, planned, tmp_path_factory):
    """Every grader call of grader part 2 raises on runs 1 and 2 (an outage of the account).
    Run 3 resumes with the outage still on; run 4 resumes with --allow-final-ask after it ends."""
    env = job_env(tmp_path_factory.mktemp("outage"))
    part = planned.plan["parts"]["grader-2"]
    keys = {_unit_key(planned.plan, u) for u in part}
    down = {"on": True}

    def plan(r, model, sid, epoch):
        if r == "grader" and down["on"] and (model, sid, epoch) in keys:
            return gf.RAISE
        return gf.default_plan(r, model, sid, epoch)

    calls = Counter()
    roles = gf.judge_roles(plan, calls)
    spy = Spy(env)
    out = SimpleNamespace(env=env, part=part, runs=[], ledgers=[], seen=[], calls=[])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        for extra in ((), ("--resume",), ("--resume",), ("--resume", "--allow-final-ask")):
            if "--allow-final-ask" in extra:
                down["on"] = False
            start, before = len(spy.seen), sum(calls.values())
            out.runs.append(run_main(live(world, env, *extra), roles, kw(world, env)))
            out.ledgers.append(ledger_of(env))
            out.seen.append(spy.seen[start:])
            out.calls.append(sum(calls.values()) - before)
    return out


def test_an_outage_of_a_whole_part_stops_at_the_final_ask_guard(outage):
    o = outage
    first, second, third, _ = o.runs
    assert first.code == gj.EXIT_CHECK and second.code == gj.EXIT_CHECK
    assert "warning" not in first.err
    assert gj.final_ask_warning(len(o.part), "grader-2") in second.err                # the next resume is the final ask
    assert third.code == gj.EXIT_REFUSED
    assert o.seen[2] == [] and o.calls[2] == 0                                          # no call
    assert f"FAIL: {gj.STAGE_FINAL_ASK_GUARD} (grader-2)" in third.err and gj.STAGE_HINTS["final ask guard"] in third.err
    led = o.ledgers[2]
    guard = led["failures"][-1]
    assert guard["stage"] == "final ask guard" and guard["part"] == "grader-2" and guard["eval_kind"] == "isolated"
    assert guard["attempt"] == 3 and "more than 12" in guard["error"]
    assert not any("grader" in led["replies_of_record"].get(u, {}) for u in o.part)    # no failed judgment
    assert led == {**o.ledgers[1], "attempts": led["attempts"], "failures": led["failures"]}
    assert led["attempts"][-1]["resume"] is True and led["attempts"][-1]["allow_final_ask"] is False
    for res in o.runs:
        assert_quiet(res)


def test_a_resume_with_allow_final_ask_runs_the_isolated_eval_past_the_guard(outage):
    o = outage
    fourth = o.runs[3]
    assert fourth.code == gj.EXIT_OK, fourth.err
    iso = o.seen[3][0]
    assert iso["task"] == "grading_grader_part_2_isolated" and iso["ids"] == o.part and iso["max_samples"] == 1
    led = ledger_of(o.env)
    assert led["attempts"][-1]["attempt"] == 4 and led["attempts"][-1]["allow_final_ask"] is True
    assert [a.get("allow_final_ask") for a in led["attempts"]] == [None, False, False, True]
    replies = sealed_of(o.env, gc.SEALED_GRADER_REPLIES)["replies"]
    assert all(replies[u]["attempt"] == 4 for u in o.part) and not any(r["status"] == "failed" for r in replies.values())
    ev = json.loads(evidence_path(o.env).read_text(encoding="utf-8"))
    assert ev["counts"]["grader"]["failed"] == 0


# --- The rehearsal and the resolved-string stop on resume ---------------------------------------

def test_a_resume_after_a_rehearsal_that_did_not_succeed_rehearses_again(world, tmp_path):
    env = job_env(tmp_path)
    hook, state = crash_after(REHEARSAL_TASK, 5)
    spy = Spy(env, hook)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "run_eval", spy)
        first = run_main(live(world, env), gf.judge_roles(), kw(world, env))
        start = len(spy.seen)
        resume = run_main(live(world, env, "--resume"), gf.judge_roles(), kw(world, env))
    assert first.code == gj.EXIT_CHECK and resume.code == gj.EXIT_OK, resume.err
    again = spy.seen[start]
    assert again["task"] == REHEARSAL_TASK and len(again["ids"]) == REHEARSAL
    assert again["ids"][:REHEARSAL - 5] == state["dropped"]
    kinds = [(e["attempt"], e["eval_kind"], e["status"]) for e in ledger_of(env)["attempt_logs"]
             if e["eval_kind"] == "rehearsal"]
    assert kinds == [(1, "rehearsal", "error"), (2, "rehearsal", "success")]
    assert_quiet(first)
    assert_quiet(resume)


def test_a_resolved_string_stop_on_resume_is_recorded_and_blocks_the_next_resume(world, tmp_path):
    env = job_env(tmp_path)
    _crash_part_1(world, env)
    seen = []

    def vendor_change(grader_summaries, calibrator_summaries, *, allowed_resolved, offline):
        seen.append((len(grader_summaries), len(calibrator_summaries)))
        raise gc.Refused("a grader call resolved to a model string outside the pin")

    calls = Counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gj, "check_all_resolved", vendor_change)
        res = _resume_no_call(world, env, calls)
    assert res.code == gj.EXIT_REFUSED and sum(calls.values()) == 0
    assert seen and seen[0][0] > 0                                  # the re-admitted grader summaries were checked
    assert f"FAIL: {gj.STAGE_RESOLVED}" in res.err and gj.STAGE_HINTS["resolved model strings"] in res.err
    assert "A resume refuses while this stop is recorded" in gj.STAGE_HINTS["resolved model strings"]
    assert "repeats this check" not in gj.STAGE_HINTS["resolved model strings"]
    led = ledger_of(env)
    assert led["failures"][-1]["stage"] == "resolved model strings" and led["failures"][-1]["attempt"] == 2
    assert led["attempts"][-1]["resume"] is True
    again = _resume_no_call(world, env, calls)
    assert again.code == gj.EXIT_REFUSED and "resolved model strings stop" in again.err and sum(calls.values()) == 0
    assert ledger_of(env) == led                                    # the refusal records nothing
    assert_quiet(res)
    assert_quiet(again)
