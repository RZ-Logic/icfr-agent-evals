"""scripts/review_freeze.py: second-look scorer and blinding-check pool, tooling built after the tag.

Pre-registration section 8 ("Second look on review verdicts") and the section 13 row on
tooling built after the tag, items (4) and (5), decision D-R19:
(4a) a worker run with no second-look units (N = 0) gets the gate's empty branch and is
     labeled second look empty, never confirmed and never review not confirmed;
(4b) beside the gate, each worker run carries the count of its narrative-only units
     (decision D-J) first ruled whole-narrative wrong, labeled not second-looked;
(4c) the bootstrap seed is the derived one: left out it is derived, any other value is refused;
(5)  the blinding-check pool is judged on the narrative text the reviewer is shown, for
     the same case, and no text is shown twice; fewer than 40 qualifying texts are all drawn.

Every packet and record here is synthetic, built in tmp_path or in memory with made-up
unit ids, with reviewer SYNTHETIC-TEST. No test runs git, calls a model or reads a log.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals import review_schema as rs
from test_review_freeze import (AMOUNT, LABELS, MODELS, NOTE, REVIEWER, SEED, _exit_code, _finding, _git,
                                _member, _narrative, _second_look_fill, blind_argv, build, classify, do_freeze,
                                draw_argv, fill, mark_template, mdir, rf, rp, score_argv)

WRONG = ("wrong", "1", "amount")


# --- Helpers ----------------------------------------------------------------------------------

def custom_packet(tmp_path, narratives, *, n_runs=2, template=None):
    """A synthetic packet from a list of (name, case, text, placements), each placement a
    (run index, epoch, routed) triple. A routed placement adds an amount finding (one
    finding unit per narrative unit). template names a run index marked template baseline."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs, units, members, ids = [], [], [], {}
    for i in range(n_runs):
        f = tmp_path / f"{LABELS[i]}-{i}.eval"
        f.write_bytes(f"synthetic log {i}".encode())
        runs.append(rp.RunInput(run_id=f"R{i + 1}", label=LABELS[i], path=str(f), sha256=rs.sha256_file(f),
                                eval_id=f"EVAL{i + 1}ID", model=MODELS[i], revision_commit="a783322"))
    for name, case, text, placements in narratives:
        n = _narrative(name, case, text)
        units.append(n)
        ids[f"N_{name}"] = n.unit_id
        finding = None
        for run_index, epoch, routed in placements:
            run = runs[run_index].run_id
            members.append(_member(n.unit_id, "narrative", run, case, epoch))
            if routed:
                if finding is None:
                    finding = _finding(name, case, AMOUNT, "code_review", rs.narrative_sentences(text)[0],
                                       [n.unit_id])
                    units.append(finding)
                    ids[f"F_{name}"] = finding.unit_id
                members.append(_member(finding.unit_id, "finding", run, case, epoch, AMOUNT, "code_review"))
    units, members = classify(units, members)
    cases = tuple(sorted({u.case_id for u in units}))
    packet = rp.Packet(run_id="dev-test-run", seed=SEED, scope="hard", fm_epoch_routes="findings", runs=tuple(runs),
                       case_order=cases, units=tuple(units), members=rp._sort_members(members),
                       case_facts={c: () for c in cases}, case_inputs={c: f"Case file text for {c}." for c in cases})
    out = tmp_path / "packet"
    rp.write_packet(packet, str(out))
    env = SimpleNamespace(dir=out, packet=packet, ids=ids, runs=runs, tmp=tmp_path)
    if template is not None:
        mark_template(env, runs[template].run_id)
    return env


def text(k):
    return f"The variance is $5,000 on chain. Item {k} is open."


def empty_run_env(tmp_path):
    """R1 holds two routed narratives and a narrative-only one; R2 holds only narrative-only
    units, so R2 has no second-look unit (N = 0)."""
    return custom_packet(tmp_path, [
        ("A", "case-a", text("a"), [(0, 1, True)]),
        ("B", "case-b", text("b"), [(0, 1, True)]),
        ("H", "case-h", text("h"), [(0, 1, False)]),
        ("J", "case-j", text("j"), [(1, 1, False)]),
        ("K", "case-k", text("k"), [(1, 1, False)]),
    ])


def frozen(env, monkeypatch, capsys, verdicts=None):
    code, _, err = do_freeze(monkeypatch, capsys, env, fill(env, verdicts=verdicts))
    assert code == 0, err
    return env


def pool_record(entries, *, runs=("R1", "R2", "R3"), template=()):
    """An in-memory record for the pool: entries are (unit id, case id, text, [run ids])."""
    members, units = [], {}
    for uid, case, narrative, on_runs in entries:
        units[uid] = {"row_kind": "narrative", "case_id": case, "narrative": narrative, "route_class": "routed"}
        for run in on_runs:
            members.append({"unit_id": uid, "member_kind": "narrative", "run_id": run})
    run_info = {r: {"label": r.lower(), "template_baseline": r in template} for r in runs}
    return SimpleNamespace(runs=run_info, members=members, units=units, rows=[], meta={"run_id": "mock"})


def _result(**kw):
    base = {"N": 120, "sample_first_correct": 90, "sample_first_not_correct": 12, "run_first_not_correct": 12,
            "ci_lower": 0.7, "correct_to_not_correct": 0}
    base.update(kw)
    return base


# --- (4a) the empty branch -------------------------------------------------------------------

def test_gate_empty_branch_for_a_run_with_no_second_look_units():
    empty = _result(N=0, sample_first_correct=0, sample_first_not_correct=0, run_first_not_correct=0,
                    ci_lower=None)
    g = rf.gate(empty)
    assert g == {"branch": "empty", "passed": None, "result": "second look empty"}
    assert tuple(g) == rf.GATE_KEYS
    assert g["result"] not in ("confirmed", "review not confirmed") and g["passed"] is not True
    # read from the sample when N is absent: an empty sample means N = 0
    no_n = {k: v for k, v in empty.items() if k != "N"}
    assert rf.gate(no_n)["branch"] == "empty"


def test_gate_branches_with_units_are_unchanged():
    assert rf.gate(_result()) == {"branch": "kappa", "passed": True, "result": "confirmed"}
    assert rf.gate(_result(ci_lower=0.59)) == {"branch": "kappa", "passed": False, "result": "review not confirmed"}
    low = _result(N=5, sample_first_correct=1, sample_first_not_correct=4, run_first_not_correct=4)
    assert rf.gate(low) == {"branch": "no_overturn", "passed": True, "result": "confirmed"}
    assert rf.gate(dict(low, correct_to_not_correct=1))["result"] == "review not confirmed"
    one = _result(N=1, sample_first_correct=1, sample_first_not_correct=0, run_first_not_correct=0)
    assert rf.gate(one)["branch"] == "no_overturn"
    few = _result(sample_first_correct=8)
    assert rf.gate(few) == {"branch": "too_few_per_class", "passed": False, "result": "review not confirmed"}


def test_second_look_result_gives_the_empty_run_its_label(tmp_path):
    env = empty_run_env(tmp_path)
    rec = rf.load_record(str(env.dir), str(fill(env)))
    assert rf.validate_record(rec) == []
    units = rf.second_look_units(rec)
    assert len(units["R1"]) == 2 and units["R2"] == []
    details = rf._draw_details(units, 5)
    assert details["R2"]["N"] == 0 and details["R2"]["n"] == 0 and details["R2"]["drawn"] == []
    sl_map = [{"sl_unit_id": f"S-{u}", "first_unit_id": u, "row_kind": "narrative", "run_id": "R1",
               "stratum": "code_review", "first_class": "correct"} for u in details["R1"]["drawn"]]
    sl_map += [{"sl_unit_id": f"S-{f}", "first_unit_id": f, "row_kind": "finding", "run_id": "", "stratum": "",
                "first_class": ""} for u in units["R1"] for f in u["finding_unit_ids"]]
    sl_rows = [{"unit_id": r["sl_unit_id"], "verdict": "none_wrong" if r["row_kind"] == "narrative" else "correct"}
               for r in sl_map]
    result = rf.second_look_result(rec, sl_rows, sl_map, bootstrap_seed=1)["runs"]
    r2 = result["R2"]
    assert tuple(r2) == rf.RUN_RESULT_KEYS
    assert (r2["N"], r2["n"], r2["kappa"], r2["ci_lower"]) == (0, 0, None, None)
    assert r2["gate"] == {"branch": "empty", "passed": None, "result": "second look empty"}
    # a run with units keeps its branch: two correct units, none overturned
    assert result["R1"]["gate"] == {"branch": "no_overturn", "passed": True, "result": "confirmed"}


def test_score_cli_labels_an_empty_run_and_prints_it(tmp_path, monkeypatch, capsys):
    env = frozen(empty_run_env(tmp_path), monkeypatch, capsys, verdicts={})
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    capsys.readouterr()
    filled = _second_look_fill(env, tmp_path / "second-look")
    assert rf.main(score_argv(env, filled)) == 0, capsys.readouterr().err
    stdout = capsys.readouterr().out
    beta_line = next(line for line in stdout.splitlines() if line.startswith("beta:"))
    assert "n 0 of N 0" in beta_line and "gate empty: second look empty (passed None)" in beta_line
    assert "confirmed" not in beta_line
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["runs"]["beta"]["gate"] == {"branch": "empty", "passed": None, "result": "second look empty"}
    assert man["runs"]["alpha"]["gate"]["result"] == "confirmed"


# --- (4b) narrative-only wrong, not second-looked ---------------------------------------------

def test_narrative_only_wrong_count_per_worker_run(tmp_path):
    env = build(tmp_path, n_runs=2, narr_only=True)
    ids = env.ids
    # N_H is one narrative-only unit on R1 (two epochs) and R2; N_I is on R1 only
    rec = rf.load_record(str(env.dir), str(fill(env, verdicts={ids["N_H"]: WRONG, ids["N_I"]: WRONG})))
    assert rf.validate_record(rec) == []
    assert rf.narrative_only_wrong(rec) == {"R1": 2, "R2": 1}
    only_i = rf.load_record(str(env.dir), str(fill(env, verdicts={ids["N_I"]: WRONG})))
    assert rf.narrative_only_wrong(only_i) == {"R1": 1, "R2": 0}
    # a routed narrative ruled wrong is a second-look unit and is not counted here
    routed = rf.load_record(str(env.dir), str(fill(env, verdicts={ids["N_U1"]: WRONG, ids["N_A"]: WRONG})))
    assert rf.narrative_only_wrong(routed) == {"R1": 0, "R2": 0}
    none = rf.load_record(str(env.dir), str(fill(env)))
    assert rf.narrative_only_wrong(none) == {"R1": 0, "R2": 0}


def test_narrative_only_wrong_count_skips_the_template_baseline(tmp_path):
    env = custom_packet(tmp_path, [
        ("A", "case-a", text("a"), [(0, 1, True), (1, 1, True)]),
        ("H", "case-h", text("h"), [(0, 1, False), (2, 1, False)]),
        ("T", "case-t", text("t"), [(2, 1, False)]),
    ], n_runs=3, template=2)
    rec = rf.load_record(str(env.dir), str(fill(env, verdicts={env.ids["N_H"]: WRONG, env.ids["N_T"]: WRONG})))
    assert rf.narrative_only_wrong(rec) == {"R1": 1, "R2": 0}


def test_narrative_only_wrong_count_reaches_the_result_and_the_public_manifest(tmp_path, monkeypatch, capsys):
    env = empty_run_env(tmp_path)
    frozen(env, monkeypatch, capsys, verdicts={env.ids["N_H"]: WRONG, env.ids["N_J"]: WRONG,
                                               env.ids["N_K"]: WRONG})
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    capsys.readouterr()
    out = tmp_path / "second-look"
    sl_map = rs.read_csv(out / rf.SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)
    assert not {r["first_unit_id"] for r in sl_map} & {env.ids["N_H"], env.ids["N_J"], env.ids["N_K"]}
    assert rf.main(score_argv(env, _second_look_fill(env, out))) == 0, capsys.readouterr().err
    stdout = capsys.readouterr().out
    assert "narrative-only wrong 1 (not second-looked)" in next(l for l in stdout.splitlines() if l.startswith("alpha:"))
    assert "narrative-only wrong 2 (not second-looked)" in next(l for l in stdout.splitlines() if l.startswith("beta:"))
    assert "narrative_only_wrong_not_second_looked" in rf.PUBLIC_RUN_RESULT_KEYS
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["runs"]["alpha"]["narrative_only_wrong_not_second_looked"] == 1
    assert man["runs"]["beta"]["narrative_only_wrong_not_second_looked"] == 2
    assert tuple(man["runs"]["beta"]) == rf.PUBLIC_RUN_RESULT_KEYS
    # the count stays beside the gate: the empty run is still empty, the other still confirmed
    assert man["runs"]["beta"]["gate"]["result"] == "second look empty"
    assert man["runs"]["alpha"]["gate"]["result"] == "confirmed"
    result = json.loads((out / rf.SECOND_LOOK_RESULT_FILE).read_text(encoding="utf-8"))
    assert result["runs"]["R2"]["narrative_only_wrong_not_second_looked"] == 2


# --- (4c) the bootstrap seed is the derived one ------------------------------------------------

def test_derived_seed_follows_the_written_rule():
    sha = "ab" * 32
    want = int(hashlib.sha256(f"{sha}|run-x|bootstrap".encode("utf-8")).hexdigest()[:16], 16)
    assert rf.derived_seed(sha, "run-x", "bootstrap") == want
    # pinned value, so a change to the rule cannot pass unseen
    assert rf.derived_seed("0" * 64, "dev-test-run", "bootstrap") == PINNED_BOOTSTRAP_SEED
    assert rf.derived_seed(sha, "run-x", "bootstrap") != rf.derived_seed(sha, "run-y", "bootstrap")
    assert rf.derived_seed(sha, "run-x", "bootstrap") != rf.derived_seed(sha, "run-x", "second_look")
    assert 0 <= want < 16 ** 16


# int(sha256(b"000...0|dev-test-run|bootstrap").hexdigest()[:16], 16), worked out once by hand
PINNED_BOOTSTRAP_SEED = 1598748189817831262


def test_check_bootstrap_seed_derives_accepts_and_refuses():
    sha = "cd" * 32
    want = rf.derived_seed(sha, "run-x", "bootstrap")
    assert rf.check_bootstrap_seed(None, sha, "run-x") == (want, [])
    assert rf.check_bootstrap_seed(want, sha, "run-x") == (want, [])
    seed, gaps = rf.check_bootstrap_seed(want + 1, sha, "run-x")
    assert seed == want and len(gaps) == 1 and "is not the seed" in gaps[0] and "never chosen" in gaps[0]
    assert rf.check_bootstrap_seed(0, sha, "run-x")[1]
    assert rf.check_bootstrap_seed(-want, sha, "run-x")[1]


def _drawn_env(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    frozen(env, monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    capsys.readouterr()
    return env, tmp_path / "second-look"


def _want_seed(env):
    return rf.derived_seed(rs.sha256_file(mdir(env) / "dev-test-run-manifest.json"), "dev-test-run", "bootstrap")


def test_score_derives_the_seed_when_the_flag_is_left_out(tmp_path, monkeypatch, capsys):
    env, out = _drawn_env(tmp_path, monkeypatch, capsys)
    want = _want_seed(env)
    assert rf.main(score_argv(env, _second_look_fill(env, out))) == 0, capsys.readouterr().err
    assert f"bootstrap seed {want} " in capsys.readouterr().out
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out / rf.SECOND_LOOK_RESULT_FILE).read_text(encoding="utf-8"))
    assert man["bootstrap_seed"] == want and result["bootstrap_seed"] == want


def test_score_accepts_the_matching_seed(tmp_path, monkeypatch, capsys):
    env, out = _drawn_env(tmp_path, monkeypatch, capsys)
    want = _want_seed(env)
    argv = score_argv(env, _second_look_fill(env, out), "--bootstrap-seed", str(want))
    assert rf.main(argv) == 0, capsys.readouterr().err
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["bootstrap_seed"] == want


@pytest.mark.parametrize("given", ["99", "0", "-1"])
def test_score_refuses_another_seed_and_writes_nothing(tmp_path, monkeypatch, capsys, given):
    env, out = _drawn_env(tmp_path, monkeypatch, capsys)
    argv = score_argv(env, _second_look_fill(env, out), "--bootstrap-seed", given)
    assert rf.main(argv) == rf.EXIT_REFUSED
    err = capsys.readouterr().err
    assert f"--bootstrap-seed {given} is not the seed" in err and str(_want_seed(env)) in err
    assert not (out / rf.SECOND_LOOK_RESULT_FILE).exists() and not (out / rf.SECOND_LOOK_VERDICTS_FILE).exists()
    assert not (mdir(env) / "dev-test-run-second-look-manifest.json").exists()
    assert _exit_code(score_argv(env, out / "x.csv", "--bootstrap-seed", "not-an-int")) == rf.EXIT_REFUSED
    capsys.readouterr()


# --- (5) the blinding-check pool ---------------------------------------------------------------

def test_pool_excludes_a_text_two_runs_share_on_one_case():
    rec = pool_record([("u1", "case-a", "X", ["R1"]), ("u2", "case-a", "X", ["R2"]),
                       ("u3", "case-a", "Y", ["R1"])])
    assert rf.blind_check_pool(rec) == [("u3", "R1")]


def test_pool_excludes_one_unit_two_runs_carry():
    rec = pool_record([("u1", "case-a", "X", ["R1", "R2"]), ("u2", "case-b", "Y", ["R2"])])
    assert rf.blind_check_pool(rec) == [("u2", "R2")]


def test_pool_keeps_a_text_one_run_wrote_twice_once():
    rec = pool_record([("u2", "case-a", "X", ["R1"]), ("u1", "case-a", "X", ["R1"]),
                       ("u3", "case-a", "Z", ["R2"])])
    assert rf.blind_check_pool(rec) == [("u1", "R1"), ("u3", "R2")]


def test_pool_judges_the_same_text_on_different_cases_apart():
    # case-a: X is shared by two runs, so it names no one model there; case-b: X is R1's alone
    rec = pool_record([("u1", "case-a", "X", ["R1"]), ("u2", "case-a", "X", ["R2"]),
                       ("u3", "case-b", "X", ["R1"])])
    assert rf.blind_check_pool(rec) == [("u3", "R1")]
    # X qualifies on two cases: the page shows no case, so the text is shown once, from the lowest case
    rec = pool_record([("u9", "case-a", "X", ["R2"]), ("u3", "case-b", "X", ["R1"]), ("u5", "case-c", "W", ["R1"])])
    assert rf.blind_check_pool(rec) == [("u5", "R1"), ("u9", "R2")]


def test_pool_compares_the_displayed_text_exactly():
    rec = pool_record([("u1", "case-a", "X.", ["R1"]), ("u2", "case-a", "X. ", ["R2"])])
    assert rf.blind_check_pool(rec) == [("u1", "R1"), ("u2", "R2")]


def test_pool_never_counts_the_template_baseline():
    rec = pool_record([("u1", "case-a", "X", ["R1", "R3"]), ("u2", "case-a", "Y", ["R3"]),
                       ("u3", "case-b", "Z", ["R2"]), ("u4", "case-b", "Z", ["R3"])], template=("R3",))
    assert rf.worker_run_ids(rec) == ["R1", "R2"]
    assert rf.blind_check_pool(rec) == [("u1", "R1"), ("u3", "R2")]


def test_pool_refuses_a_member_naming_no_packet_unit():
    rec = pool_record([("u1", "case-a", "X", ["R1"])])
    rec.members.append({"unit_id": "ghost", "member_kind": "narrative", "run_id": "R1"})
    with pytest.raises(rf.Refused, match="not in the packet"):
        rf.blind_check_pool(rec)


@pytest.mark.parametrize("n_texts, want", [(0, 0), (5, 5), (39, 39), (40, 40), (45, 40)])
def test_draw_takes_all_when_fewer_than_40_qualify(n_texts, want):
    entries = [(f"u{i:03d}", f"case-{i % 7}", f"text {i}", [("R1", "R2")[i % 2]]) for i in range(n_texts)]
    entries += [("dup-a", "case-x", "shared", ["R1"]), ("dup-b", "case-x", "shared", ["R2"])]
    rec = pool_record(entries)
    pool, checks = rf.blind_check_checks(rec, seed=11, salt="0" * 32)
    assert len(pool) == n_texts and len(checks) == want
    texts = [c["narrative"] for c in checks]
    assert len(set(texts)) == len(texts) and "shared" not in texts
    if n_texts and n_texts <= 40:
        assert {c["narrative"] for c in checks} == {f"text {i}" for i in range(n_texts)}
    again = rf.blind_check_checks(rec, seed=11, salt="0" * 32)[1]
    assert again == checks
    assert all(c["run_label"] == c["run_id"].lower() and c["guess"] == "" for c in checks)


def dup_text_env(tmp_path):
    """Two distinct narrative units with one text on case-s, one in each worker run, plus
    the same text held once by R1 on case-t, and two texts R1 wrote once each."""
    shared = text("shared")
    return custom_packet(tmp_path, [
        ("A", "case-a", text("a"), [(0, 1, True)]),
        ("B", "case-b", text("b"), [(1, 1, True)]),
        ("S1", "case-s", shared, [(0, 1, True)]),
        ("S2", "case-s", shared, [(1, 1, False)]),
        ("T", "case-t", shared, [(0, 2, False)]),
        ("P", "case-p", text("p"), [(0, 1, False)]),
        ("Q", "case-q", text("p"), [(1, 1, False)]),
    ])


def test_blind_check_draw_shows_no_text_twice(tmp_path, monkeypatch, capsys):
    env = frozen(dup_text_env(tmp_path), monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B")) == 0, capsys.readouterr().err
    d3 = tmp_path / "blind-check"
    sheet = rs.read_csv(d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
    texts = [r["narrative"] for r in sheet]
    assert len(texts) == len(set(texts))
    # case-s names no one model, case-t names R1; text p qualifies on case-p and case-q and is shown once
    assert sorted(texts) == sorted([text("a"), text("b"), text("shared"), text("p")])
    mapping = {m["check_id"]: m for m in rs.read_csv(d3 / rf.BLIND_CHECK_MAP, rf.BLIND_CHECK_MAP_COLUMNS)}
    by_text = {r["narrative"]: mapping[r["check_id"]]["run_label"] for r in sheet}
    assert by_text[text("shared")] == "alpha" and by_text[text("p")] == "alpha"
    meta = json.loads((d3 / rf.BLIND_CHECK_META).read_text(encoding="utf-8"))
    assert meta["pool"] == 4 and meta["drawn"] == 4
    assert "only 4 narrative text(s)" in capsys.readouterr().err


def _scored_blind_check(env, tmp_path, capsys):
    d3 = tmp_path / "blind-check"
    mapping = rs.read_csv(d3 / rf.BLIND_CHECK_MAP, rf.BLIND_CHECK_MAP_COLUMNS)
    sheet = rs.read_csv(d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
    truth = {m["check_id"]: m["display_name"] for m in mapping}
    for row in sheet:
        row["guess"] = truth[row["check_id"]]
    answers = tmp_path / "blind-answers.csv"
    rs.write_csv(answers, rf.BLIND_CHECK_COLUMNS, sheet)
    assert rf.main(["blind-check-score", "--packet-dir", str(env.dir), "--blind-check-dir", str(d3),
                    "--filled", str(answers)]) == 0, capsys.readouterr().err
    capsys.readouterr()
    return d3


def _rehash(d3, rel):
    meta_path = d3 / rf.BLIND_CHECK_META
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"][rel] = rs.sha256_file(d3 / rel)
    rs.write_lf(meta_path, json.dumps(meta, indent=2) + "\n")


def test_check_blind_check_folder_rebuilds_the_pool(tmp_path, monkeypatch, capsys):
    env = frozen(dup_text_env(tmp_path), monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B")) == 0, capsys.readouterr().err
    d3 = _scored_blind_check(env, tmp_path, capsys)
    first = rf.load_frozen_record(str(env.dir))
    first_sha = rs.sha256_file(mdir(env) / "dev-test-run-manifest.json")
    gaps, result = rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")
    assert gaps == [] and result["n"] == 4
    meta_path, map_path, csv_path = d3 / rf.BLIND_CHECK_META, d3 / rf.BLIND_CHECK_MAP, d3 / rf.BLIND_CHECK_CSV
    kept = {p: p.read_bytes() for p in (meta_path, map_path, csv_path)}
    # a map row moved to the other run, rehashed in the meta: refused by the rebuild
    rows = rs.read_csv(map_path, rf.BLIND_CHECK_MAP_COLUMNS)
    rows[0] = dict(rows[0], run_id="R2" if rows[0]["run_id"] == "R1" else "R1",
                   run_label="beta" if rows[0]["run_label"] == "alpha" else "alpha")
    rs.write_csv(map_path, rf.BLIND_CHECK_MAP_COLUMNS, rows)
    _rehash(d3, rf.BLIND_CHECK_MAP)
    gaps, result = rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")
    assert result is None and any("blind-check-map.csv differs from the rebuild" in g for g in gaps)
    for p, b in kept.items():
        p.write_bytes(b)
    # a reviewer sheet holding a text twice: refused by the rebuild
    sheet = rs.read_csv(csv_path, rf.BLIND_CHECK_COLUMNS)
    sheet[1] = dict(sheet[1], narrative=sheet[0]["narrative"])
    rs.write_csv(csv_path, rf.BLIND_CHECK_COLUMNS, sheet)
    _rehash(d3, rf.BLIND_CHECK_CSV)
    gaps, _ = rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")
    assert any("blind-check.csv differs from the rebuild" in g for g in gaps)
    for p, b in kept.items():
        p.write_bytes(b)
    # an edited pool count
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rs.write_lf(meta_path, json.dumps(dict(meta, pool=meta["pool"] + 1), indent=2) + "\n")
    gaps, _ = rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")
    assert any("pool or drawn count differs" in g for g in gaps)
    rs.write_lf(meta_path, json.dumps(dict(meta, id_salt="nope"), indent=2) + "\n")
    gaps, _ = rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")
    assert gaps == ["blind-check-meta.json records no id salt"]
    for p, b in kept.items():
        p.write_bytes(b)
    assert rf.check_blind_check_folder(first, d3, first_sha, "abcdef1")[0] == []


def test_second_look_score_refuses_a_blind_check_drawn_from_an_edited_map(tmp_path, monkeypatch, capsys):
    env = frozen(dup_text_env(tmp_path), monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B")) == 0, capsys.readouterr().err
    d3 = _scored_blind_check(env, tmp_path, capsys)
    map_path = d3 / rf.BLIND_CHECK_MAP
    kept = map_path.read_bytes()
    rows = rs.read_csv(map_path, rf.BLIND_CHECK_MAP_COLUMNS)
    rs.write_csv(map_path, rf.BLIND_CHECK_MAP_COLUMNS, rows[:-1])
    meta_kept = (d3 / rf.BLIND_CHECK_META).read_bytes()
    _rehash(d3, rf.BLIND_CHECK_MAP)
    filled = _second_look_fill(env, tmp_path / "second-look")
    assert rf.main(score_argv(env, filled, "--blind-check-dir", str(d3))) == rf.EXIT_REFUSED
    assert "differs from the rebuild" in capsys.readouterr().err
    map_path.write_bytes(kept)
    (d3 / rf.BLIND_CHECK_META).write_bytes(meta_kept)
    assert rf.main(score_argv(env, filled, "--blind-check-dir", str(d3))) == 0, capsys.readouterr().err
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["blinding"]["n"] == 4


def test_new_source_text_follows_the_house_style():
    for path in (rf.SCRIPTS_DIR / "review_freeze.py", Path(__file__)):
        raw = path.read_bytes()
        assert b"\r" not in raw
        body = raw.decode("utf-8")
        assert chr(0x2014) not in body and chr(0x2013) not in body
    assert NOTE.startswith("synthetic test row") and REVIEWER == "SYNTHETIC-TEST"
