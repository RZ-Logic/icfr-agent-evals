"""The calibrator plumbing check (scripts/calibrator_plumbing_check.py; D-R23, plumbing check).

Keyless: the only calibrator is Inspect's mockllm with canned replies, and the evaluated model is
none/none; no call reaches a provider. The inputs are the committed judge-development records
(dev split). No calibration record or case is read: the calibration case file is a stand-in
under tmp_path. The start checks run on scratch git repositories under tmp_path, and the
records go to a folder under tmp_path, so nothing touches this repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gradingfactory as gf
from icfr_agent_evals import roster, scoring

REPO = Path(__file__).resolve().parents[1]
cpc = gf.load_script("calibrator_plumbing_check")
gc = gf.load_script("grading_common")
cc = gf.load_script("calibrator_call")

RECORDS = sorted((REPO / "data" / "dev" / "judge-dev").glob("*.json"))
NAMES = [p.stem for p in RECORDS]


def narrative_of(name: str) -> str:
    record = json.loads((REPO / "data" / "dev" / "judge-dev" / f"{name}.json").read_text(encoding="utf-8"))
    return str(scoring.parse_worker_output(record["worker_output"]).parsed["narrative"])


def roles(bad: str | None = None):
    """A mockllm calibrator that answers every prompt with a reply holding a reason text the
    check must never print or store; on the narrative of the record named bad it answers with
    text that does not parse."""
    from inspect_ai.model import ModelOutput, get_model

    bad_text = narrative_of(bad) if bad else None

    def out(input, tools, tool_choice, config):
        text = "\n".join(str(m.text) for m in input)
        if bad_text is not None and bad_text in text:
            return ModelOutput.from_content("mockllm/model", "this is not a reply")
        return ModelOutput.from_content("mockllm/model", gf.reply(2, 1, low=True, reason=gf.SECRET))

    return {cc.CALIBRATOR_ROLE: get_model("mockllm/model", custom_outputs=out, memoize=False)}


def world(tmp_path: Path) -> SimpleNamespace:
    env = gf.make_repo(tmp_path)
    return SimpleNamespace(env=env, records=tmp_path / "records", logs=tmp_path / "logs",
                           cases=tmp_path / "no-calibration-cases.jsonl")


def run(w: SimpleNamespace, argv: list[str], *, model_roles=None, **extra) -> int:
    kwargs = dict(model_roles=model_roles, repo=w.env.repo, ledger_dir=w.records, run_commit=w.env.run_commit,
                  archive_file=w.env.archive_file, running_package=False, cases_file=w.cases)
    kwargs.update(extra)
    return cpc.main(argv, **kwargs)


def live(w: SimpleNamespace, *extra: str) -> list[str]:
    return ["--live", "--log-dir", str(w.logs), *extra]


def records_of(w: SimpleNamespace) -> list[dict]:
    return cpc.previous_records(w.records)


def _docs_files() -> list[str]:
    return sorted(str(p.relative_to(REPO)) for p in (REPO / "docs").rglob("*") if p.is_file())


def test_there_are_enough_judge_development_records():
    assert len(NAMES) >= cpc.OUTPUTS


def test_dry_run_makes_no_call(tmp_path, monkeypatch, capsys):
    w = world(tmp_path)
    monkeypatch.setattr(cpc, "run_check", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert run(w, []) == cpc.EXIT_OK
    out = capsys.readouterr().out
    assert "calibrator plumbing check: 12 judge-development outputs, default prompt gp-1" in out
    assert "calibrator role, not batched, parse only" in out
    assert "dry run: no call made" in out
    assert f"outputs: {', '.join(NAMES[:12])}" in out
    assert cc.calibrator_pin() not in out
    assert not w.records.exists() and not w.logs.exists()


def test_every_reply_parses_and_no_grade_is_printed_or_written(tmp_path, capsys):
    w = world(tmp_path)
    before = _docs_files()
    code = run(w, live(w), model_roles=roles())
    out = capsys.readouterr().out
    assert code == cpc.EXIT_OK, out
    assert "12 of 12 replies parse; 0 refused" in out and out.count(": parses;") == 12
    assert "the calibrator plumbing check passed" in out
    assert gf.SECRET not in out and '"level"' not in out and "low_confidence" not in out
    assert _docs_files() == before
    [record] = records_of(w)
    assert record["_file"] == "calibrator-plumbing-1.json" and record["attempt"] == 1
    assert record["passed"] is True and record["calls"] == 12 and record["parsed"] == 12 and record["refused"] == 0
    assert record["records"] == NAMES[:12]
    assert record["head"] == w.env.head and record["archive_commit"] == w.env.run_commit
    raw = (w.records / "calibrator-plumbing-1.json").read_text(encoding="utf-8")
    assert gf.SECRET not in raw and '"level"' not in raw and "low_confidence" not in raw and '"scores"' not in raw
    assert set(record["per_call"][0]) == {"record", "parses", "refused", "input_tokens", "output_tokens"}
    assert record["reasoning_modes_sent"] == []          # offline: no request recorded
    from inspect_ai.log import read_eval_log

    logs = list(w.logs.glob("*.eval"))
    assert len(logs) == 1 and logs[0].name == record["log_file"]
    assert gc.sha256_file(logs[0]) == record["log_sha256"]
    meta = read_eval_log(str(logs[0]), header_only=True).eval.metadata
    assert meta["split"] == "dev" and meta["parse_only"] is True and meta["batched"] is False
    assert meta["calibrator_model"] == cc.calibrator_pin()
    assert meta["calibrator_generate_config"] == roster.calibrator_generate_config_args()


def test_a_reply_that_does_not_parse_fails_the_check_and_records_it(tmp_path, capsys):
    w = world(tmp_path)
    bad = NAMES[3]
    hit = sum(1 for n in NAMES[:12] if narrative_of(bad) in narrative_of(n))
    code = run(w, live(w), model_roles=roles(bad=bad))
    out = capsys.readouterr().out
    assert code == cpc.EXIT_CHECK
    assert f"{12 - hit} of 12 replies parse" in out and "does not parse (not one JSON object)" in out
    assert "the evaluation-split calibrator run does not start" in out
    assert gf.SECRET not in out and "Expecting value" not in out
    [record] = records_of(w)
    assert record["passed"] is False and record["parsed"] == 12 - hit
    rows = {r["record"]: r for r in record["per_call"]}
    assert rows[bad]["parses"] is False and rows[bad]["refused"] is False
    assert rows[bad]["why"] == "not one JSON object" and rows[bad]["why"] in cpc.FAILURE_CATEGORIES
    raw = (w.records / "calibrator-plumbing-1.json").read_text(encoding="utf-8")
    assert "Expecting value" not in raw and "the reply is not" not in raw
    with pytest.raises(gc.Refused, match="has not passed"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)


def test_a_refused_reply_does_not_pass(tmp_path, capsys):
    from inspect_ai.model import ModelOutput, get_model

    def out(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", "I will not grade this.", stop_reason="content_filter")

    w = world(tmp_path)
    code = run(w, live(w), model_roles={cc.CALIBRATOR_ROLE: get_model("mockllm/model", custom_outputs=out,
                                                                        memoize=False)})
    text = capsys.readouterr().out
    assert code == cpc.EXIT_CHECK
    assert "0 of 12 replies parse; 12 refused" in text and text.count(": refused;") == 12
    [record] = records_of(w)
    assert record["passed"] is False and record["refused"] == 12
    assert {r["why"] for r in record["per_call"]} == {"refused"}


def test_calibration_inputs_are_refused(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, ["--records-dir", str(REPO / "data" / "calibration" / "records")]) == cpc.EXIT_REFUSED
    assert "calibration inputs are refused" in capsys.readouterr().err
    first = json.loads(RECORDS[0].read_text(encoding="utf-8"))["sample_id"]
    w.cases.write_text(json.dumps({"id": first, "input": "", "metadata": {"split": "calibration"}}) + "\n",
                       encoding="utf-8")
    assert run(w, []) == cpc.EXIT_REFUSED
    assert "is a calibration case" in capsys.readouterr().err


def test_records_must_be_twelve_named_dev_records(tmp_path, capsys):
    w = world(tmp_path)

    def named(names):
        return [x for n in names for x in ("--record", n)]

    assert run(w, named(NAMES[:11])) == cpc.EXIT_REFUSED
    assert run(w, named([NAMES[0], *NAMES[:11]])) == cpc.EXIT_REFUSED
    assert run(w, named([*NAMES[:11], "no-such-record"])) == cpc.EXIT_REFUSED
    capsys.readouterr()
    chosen = NAMES[-12:]
    assert run(w, named(chosen)) == cpc.EXIT_OK
    assert f"outputs: {', '.join(chosen)}" in capsys.readouterr().out


def test_a_rerun_must_use_the_same_outputs_and_a_passed_check_is_not_rerun(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, live(w), model_roles=roles(bad=NAMES[0])) == cpc.EXIT_CHECK
    other = [x for n in NAMES[-12:] for x in ("--record", n)]
    capsys.readouterr()
    assert run(w, live(w, *other), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "same outputs" in capsys.readouterr().err
    assert len(records_of(w)) == 1
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    first, second = records_of(w)
    assert (first["attempt"], first["passed"]) == (1, False)
    assert (second["_file"], second["attempt"], second["passed"]) == ("calibrator-plumbing-2.json", 2, True)
    assert second["records"] == first["records"]
    capsys.readouterr()
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "the check passed at this code; it is not run again" in capsys.readouterr().err
    assert len(records_of(w)) == 2


def test_live_needs_a_log_dir_outside_the_repo_and_the_start_checks(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, ["--live"], model_roles=roles()) == cpc.EXIT_ARGS
    assert run(w, ["--live", "--log-dir", str(REPO / "logs")], model_roles=roles()) == cpc.EXIT_ARGS
    assert run(w, ["--live", "--log-dir", str(w.env.repo / "logs")], model_roles=roles()) == cpc.EXIT_ARGS
    capsys.readouterr()
    (w.env.repo / "untracked.txt").write_text("x\n", encoding="utf-8")
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "git status is not clean" in capsys.readouterr().err
    (w.env.repo / "untracked.txt").unlink()
    assert run(w, live(w), model_roles=roles(), archive_file=w.env.repo / "docs" / "missing.json") == cpc.EXIT_REFUSED
    assert "archive record" in capsys.readouterr().err
    assert not w.records.exists() or not records_of(w)
    assert not w.logs.exists() or not list(w.logs.glob("*.eval"))


def test_check_plumbing_passed_refuses_a_failed_or_changed_record(tmp_path, capsys):
    w = world(tmp_path)
    with pytest.raises(gc.Refused, match="has not passed"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    got = cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    [log_file] = [p.name for p in w.logs.glob("*.eval")]
    assert got == {"file": "calibrator-plumbing-1.json",
                   "sha256": gc.sha256_file(w.records / "calibrator-plumbing-1.json"),
                   "attempt": 1, "calls": 12, "parsed": 12, "refused": 0, "passed": True,
                   "head": w.env.head, "log_dir": str(w.logs.resolve()), "log_file": log_file}
    # an offline log never passes the live check (the calibrator pin and the request are required)
    with pytest.raises(gc.Refused, match="fails its check"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=False)
    # a later record that did not pass is the latest record
    later = w.records / "calibrator-plumbing-2.json"
    later.write_bytes(gc.canonical_bytes({**json.loads((w.records / "calibrator-plumbing-1.json")
                                                       .read_text(encoding="utf-8")), "attempt": 2, "passed": False}))
    with pytest.raises(gc.Refused, match="did not pass"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    later.unlink()
    # a record whose attempt field differs from its name
    odd = w.records / "calibrator-plumbing-7.json"
    odd.write_bytes(gc.canonical_bytes({"attempt": 6, "passed": True}))
    with pytest.raises(gc.Refused, match="has not passed"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    odd.unlink()
    # the log changed since the record was written
    [log] = list(w.logs.glob("*.eval"))
    log.write_bytes(log.read_bytes() + b"\0")
    with pytest.raises(gc.Refused, match="changed since the record was written"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    log.unlink()
    with pytest.raises(gc.Refused, match="not on disk"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)


def test_a_failed_log_check_writes_no_record(tmp_path, monkeypatch, capsys):
    w = world(tmp_path)
    monkeypatch.setattr(cpc, "run_check", lambda *a, **k: SimpleNamespace(status="error", samples=[]))
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_CHECK
    assert "FAIL: the log check" in capsys.readouterr().err
    assert not w.records.exists() or not records_of(w)


# --- Fixed failure categories ----------------------------------------------------------------

def _unparsed(text: str) -> dict:
    from icfr_agent_evals import grader_prompt as gp

    with pytest.raises(gp.GraderReplyError) as e:
        gp.parse_reply(text, gp.PromptSettings())
    return {**gf.failed("unparsed"), "error": str(e.value)}


def _reply_with(**dim_changes) -> str:
    body = json.loads(gf.reply(2, 1))
    first = sorted(body)[0]
    body[first] = {**body[first], **dim_changes}
    return json.dumps(body)


def test_failure_category_names_a_fixed_category_and_never_parser_text():
    two = sorted(json.loads(gf.reply(2, 1)))
    cases = [
        (None, "no call"),
        ({**gf.failed("unparsed"), "status": "failed", "reply": None}, "no call"),
        (gf.failed("refused"), "refused"),
        (_unparsed("this is not a reply"), "not one JSON object"),
        (_unparsed("[1, 2]"), "not one JSON object"),
        (_unparsed(json.dumps({"x": 1})), "wrong keys"),
        (_unparsed(json.dumps({two[0]: "text", two[1]: "text"})), "wrong keys"),
        (_unparsed(_reply_with(level="Z")), "label not offered"),
        (_unparsed(_reply_with(low_confidence="yes")), "flag not boolean"),
        # reply text that imitates another message does not move the category
        (_unparsed(_reply_with(level="x are not the sub-dimensions y")), "label not offered"),
        (_unparsed(json.dumps({"k is not one of the offered labels": 1})), "wrong keys"),
        ({**gf.failed("unparsed"), "error": "some other parser message"}, "does not parse"),
    ]
    for reply, want in cases:
        got = cpc.failure_category(reply)
        assert got == want, (reply, got)
        assert got in cpc.FAILURE_CATEGORIES


def test_a_label_not_offered_is_reported_by_category_and_the_label_never_shown(tmp_path, capsys):
    from inspect_ai.model import ModelOutput, get_model

    label = "ZQX-LABEL"

    def out(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", _reply_with(level=label, reason=gf.SECRET))

    w = world(tmp_path)
    code = run(w, live(w), model_roles={cc.CALIBRATOR_ROLE: get_model("mockllm/model", custom_outputs=out,
                                                                        memoize=False)})
    captured = capsys.readouterr()
    assert code == cpc.EXIT_CHECK
    assert captured.out.count(": does not parse (label not offered);") == 12
    for text in (captured.out, captured.err):
        assert label not in text and gf.SECRET not in text and "offered labels" not in text
    [record] = records_of(w)
    raw = (w.records / "calibrator-plumbing-1.json").read_text(encoding="utf-8")
    assert label not in raw and gf.SECRET not in raw and "offered labels" not in raw
    assert {r["why"] for r in record["per_call"]} == {"label not offered"}


# --- The pass is tied to the calibrator-path code (D-R23, plumbing check) --------------------

def _rearchive(env) -> str:
    """Commit an archive record naming HEAD, then move the local origin/main ref to it."""
    commit = gf.git(env.repo, "rev-parse", "HEAD").strip()
    env.archive_file.write_text(json.dumps({"commit": commit, "url": "https://web.archive.org/web/20261123/x",
                                            "date": "2026-11-23"}), encoding="utf-8", newline="\n")
    head = gf.commit_all(env.repo, "archive record again")
    gf.git(env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return head


def test_a_change_to_the_calibrator_path_voids_the_pass_and_allows_a_rerun(tmp_path, capsys):
    w = world(tmp_path)
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    assert cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)["attempt"] == 1
    # a change outside the guarded paths keeps the pass, and the check is not rerun
    (w.env.repo / "scripts" / "x.py").write_text("x = 2\n", encoding="utf-8", newline="\n")
    gf.commit_all(w.env.repo, "other script")
    _rearchive(w.env)
    assert cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)["attempt"] == 1
    capsys.readouterr()
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "the check passed at this code; it is not run again" in capsys.readouterr().err
    # a change to calibrator_call.py voids the pass
    (w.env.repo / "scripts" / "calibrator_call.py").write_text("changed = True\n", encoding="utf-8", newline="\n")
    gf.commit_all(w.env.repo, "calibrator path change")
    head = _rearchive(w.env)
    with pytest.raises(gc.Refused, match="changed since the record's commit; run the check again"):
        cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    # the rerun must use the same outputs as attempt 1
    assert NAMES[-12:] != NAMES[:12]
    other = [x for n in NAMES[-12:] for x in ("--record", n)]
    assert run(w, live(w, *other), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "same outputs" in capsys.readouterr().err
    # the rerun is the next numbered attempt on the same outputs, and then the check passes again
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    first, second = records_of(w)
    assert (second["_file"], second["attempt"], second["passed"]) == ("calibrator-plumbing-2.json", 2, True)
    assert second["records"] == first["records"] and second["head"] == head
    got = cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)
    assert (got["file"], got["attempt"], got["head"]) == ("calibrator-plumbing-2.json", 2, head)
    assert got["log_file"] == second["log_file"] and got["log_dir"] == str(w.logs.resolve())
    capsys.readouterr()
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_REFUSED
    assert "it is not run again" in capsys.readouterr().err
    assert len(records_of(w)) == 2


def test_each_guarded_path_voids_the_pass(tmp_path):
    for i, rel in enumerate(cpc.PLUMBING_GUARDED_PATHS):
        w = world(tmp_path / f"w{i}")
        assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
        target = w.env.repo / rel
        if rel == "icfr_agent_evals":
            target = target / "__init__.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"changed = {i}\n", encoding="utf-8", newline="\n")
        gf.commit_all(w.env.repo, "guarded change")
        with pytest.raises(gc.Refused, match="changed since the record's commit"):
            cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)


def test_a_record_head_that_is_no_ancestor_of_head_is_refused(tmp_path):
    w = world(tmp_path)
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    path = w.records / "calibrator-plumbing-1.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    for bad in ("f" * 40, "abc", None):
        path.write_bytes(gc.canonical_bytes({**body, "head": bad}))
        with pytest.raises(gc.Refused, match="changed since the record's commit"):
            cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True)


def test_a_log_that_is_not_a_dev_split_plumbing_check_is_refused(tmp_path):
    from inspect_ai.log import read_eval_log

    w = world(tmp_path)
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK

    def reader(change):
        def read(p):
            log = read_eval_log(str(p))
            change(log)
            return log

        return read

    def task(log):
        log.eval.task = "another_task"

    def split(log):
        log.eval.metadata = {**log.eval.metadata, "split": "eval"}

    def flag(log):
        log.eval.metadata = {k: v for k, v in log.eval.metadata.items() if k != "calibrator_plumbing_check"}

    for change in (task, split, flag):
        with pytest.raises(gc.Refused, match="is not a dev-split plumbing check"):
            cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True, read_log=reader(change))
    assert cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True,
                                     read_log=reader(lambda log: None))["passed"] is True
    assert read_eval_log(str(next(w.logs.glob("*.eval"))), header_only=True).eval.task == cpc.PLUMBING_TASK


def test_the_resolved_strings_reach_the_log_check(tmp_path, monkeypatch):
    w = world(tmp_path)
    assert run(w, live(w), model_roles=roles()) == cpc.EXIT_OK
    seen = []
    real = cc.check_calibrator_log

    def spy(log, **kwargs):
        seen.append(kwargs)
        return real(log, **kwargs)

    monkeypatch.setattr(cpc.cc, "check_calibrator_log", spy)
    cpc.check_plumbing_passed(w.records, repo=w.env.repo, offline=True, allowed_resolved=["resolved-a"])
    assert seen and seen[-1]["allowed_resolved"] == ["resolved-a"] and seen[-1]["offline"] is True


def test_public_text_of_the_check():
    import re

    src = (REPO / "scripts" / "calibrator_plumbing_check.py").read_text(encoding="utf-8")
    assert not re.search("[" + chr(0x2013) + chr(0x2014) + "]", src)
    assert not re.search(r"\bD[0-9]{1,2}a?\b|requirements? [0-9]", src)
    assert "(D-R23, plumbing check)" in src
