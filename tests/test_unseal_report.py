"""scripts/unseal_report.py: the unseal-and-report step of decision D-R23 (unseal step).

The review records come from tests/test_run_report.py: two mock worker logs on a claimed
evaluation split, a first record frozen and a second look scored in a scratch repository,
every verdict synthetic. The sealed folder is synthetic too: a small hand-built plan, its
replies, and result files written by the committed grading job code. No evaluation case is
built, read or scored, no API key is read, no model is called, and this repository's own
sealed folder, ledger and logs are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradingfactory as gf
import pytest
from test_run_report import (  # noqa: F401 - fixtures used by name
    LATER, _package_checks_pass, _small_eval_logs_allowed, base_log, commit_second, git, rf, small,
    small_items, small_logs, small_scored)

ur = gf.load_script("unseal_report")
gc, gs, gj = ur.gc, ur.gs, ur.gj
P, F = gf.parsed, gf.failed
RESAMPLES = 50
BOOT_SEED, DRAW_SEED = 11, 7
EVIDENCE_REL = f"docs/evidence/grading/{gc.EVIDENCE_FILE}"
JUDGE_ITEMS = ("judge-gate results per sub-dimension", "disagreement rate and resolution rate")


def _plan(models: dict[str, str]):
    """Two worker models, six cases, four epochs: units, generations, drawn rows and replies."""
    units, gens, rows, grader, calibrator = {}, [], [], {}, {}
    n = 0
    for mi, model in enumerate(sorted(models)):
        for si in range(6):
            kind = ("standard", "edge", "adversarial")[si % 3]
            for e in range(1, 5):
                n += 1
                uid = f"u{n:04d}"
                sid = f"s{si}"
                units[uid] = {"uid": uid, "model": model, "sample_id": sid, "epoch": e, "case_kind": kind,
                              "defect_epoch": False, "messages_sha256": "0" * 64}
                gens.append({"model": model, "sample_id": sid, "epoch": e, "case_kind": kind, "status": "gradeable",
                             "defect_epoch": False, "uid": uid})
                grader[uid] = F() if (mi, si, e) == (1, 2, 3) else P((si + e) % 4, (si * 2 + e) % 4,
                                                                      low_g=(si, e) == (3, 1))
                if e == 1 + si % 4:
                    rows.append({"uid": uid, "model": model, "sample_id": sid, "epoch": e, "position": len(rows) + 1})
                    calibrator[uid] = P((si + e + (si == 4) * 2) % 4, (si * 2 + e) % 4)
    return {"units": units, "generations": gens, "draw": {"rows": rows}}, {"grader": grader, "calibrator": calibrator}


def write_sealed(env, small_logs, folder: Path) -> dict:
    """A sealed folder for the small run and the evidence body that records it."""
    runs = rf.load_frozen_record(str(env.packet)).runs
    logs = []
    for label in ("alpha", "beta"):
        info = next(i for i in runs.values() if i.get("label") == label)
        logs.append({"model": f"mockllm/{label}", "file": info["file_name"], "sha256": info["sha256"],
                     "eval_id": info["eval_id"]})
    plan, replies = _plan({w["model"]: w["sha256"] for w in logs})
    folder.mkdir(parents=True)
    bodies = {
        gc.SEALED_UNIT_MAP: {"run": gc.RUN_LABEL, "worker_logs": logs, "generations": plan["generations"],
                             "units": plan["units"]},
        gc.SEALED_DRAW_POOL: {"run": gc.RUN_LABEL, "rule": "test", "seed": DRAW_SEED, "seed_material_shas": {},
                              "pool": []},
        gc.SEALED_DRAWN: {"run": gc.RUN_LABEL, "case_order": [], "rows": plan["draw"]["rows"], "lost": [],
                          "calibrator_order": []},
        **gj.result_bodies(plan, replies, {}, bootstrap_seed=BOOT_SEED, resamples=RESAMPLES),
    }
    files = {name: gc.write_sealed(folder, name, body) for name, body in bodies.items()}
    return {"evidence": "full-run grading", "run": gc.RUN_LABEL, "worker_run_commit": "a783322",
            "worker_logs": [{k: w[k] for k in ("file", "sha256", "eval_id")} for w in logs],
            "seeds": {"bootstrap": BOOT_SEED, "calibrator_draw": DRAW_SEED},
            "sealed": {"folder": gc.SEALED_FOLDER_NAME, "files": files}}


def commit_evidence(env, body: dict) -> None:
    path = env.repo / EVIDENCE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gc.canonical_bytes(body))
    git(env.repo, "add", EVIDENCE_REL)
    git(env.repo, "commit", "-q", "-m", "Grading evidence", when=LATER)


@pytest.fixture
def unsealed(small, small_logs):
    commit_second(small)
    sealed = small.root / "grading" / gc.SEALED_FOLDER_NAME
    evidence = write_sealed(small, small_logs, sealed)
    commit_evidence(small, evidence)
    return SimpleNamespace(env=small, sealed=sealed, evidence=evidence, logs=small_logs)


def argv(u, *extra) -> list[str]:
    env = u.env if hasattr(u, "env") else u
    logs = u.logs
    out = ["--sealed-dir", str(u.sealed), "--worker", f"alpha={logs['alpha']}", "--worker", f"beta={logs['beta']}",
           "--template", str(logs["template"]), "--packet-dir", str(env.packet), "--second-look-dir", str(env.d2),
           "--repo", str(env.repo)]
    if env.d3.exists():
        out += ["--blind-check-dir", str(env.d3)]
    return out + list(extra)


def run(u, capsys, *extra):
    code = ur.main(argv(u, *extra), resamples=RESAMPLES)
    out = capsys.readouterr()
    return SimpleNamespace(code=code, out=out.out, err=out.err)


def test_both_freezes_checked_then_the_judge_block_replaces_the_not_available_entries(unsealed, capsys, tmp_path):
    js = tmp_path / "report.json"
    r = run(unsealed, capsys, "--json", str(js))
    assert r.code == ur.EXIT_OK, r.err
    assert "== Judge results (unsealed" in r.out and "recomputed results equal the sealed ones" in r.out
    assert "reliability gate (ICC(A,1) lower limit on complete pairs" in r.out
    assert "grader minus calibrator " in r.out and "grader minus calibrator n/a" not in r.out
    assert "circuit breaker (full run, limit 1/4)" in r.out and "disagreement queue:" in r.out
    assert "alpha (mockllm/alpha)" in r.out and "beta (mockllm/beta)" in r.out
    na = r.out.split("== Not available", 1)[1]
    for item in JUDGE_ITEMS:
        assert f"  {item}: not available" not in na
    assert "  resolution rate: not available" in na
    payload = json.loads(js.read_text(encoding="utf-8"))
    sealed_agg = json.loads((unsealed.sealed / gc.SEALED_AGGREGATES).read_text(encoding="utf-8"))
    assert payload["judge"]["aggregates"] == sealed_agg
    assert payload["judge"]["queue_volume"] == sealed_agg["queue_counts"]["items"] > 0
    assert payload["judge"]["units"] == 48
    assert set(payload["not_available"]) & set(JUDGE_ITEMS) == set()


def test_refused_and_the_sealed_folder_untouched_before_the_second_freeze(small, small_logs, capsys, monkeypatch):
    sealed = small.root / "grading" / gc.SEALED_FOLDER_NAME
    commit_evidence(small, write_sealed(small, small_logs, sealed))   # the second look is scored, not committed

    def boom(*a, **k):
        raise AssertionError("the sealed folder was opened before both freezes checked")

    for name in ("check_sealed_files", "recompute", "committed_evidence"):
        monkeypatch.setattr(ur, name, boom)
    r = run(SimpleNamespace(env=small, sealed=sealed, logs=small_logs), capsys)
    assert r.code == ur.EXIT_REFUSED and r.out == ""
    assert "both run-report freeze checks must pass" in r.err and "second look" in r.err


def _resign(u, name: str, body: dict) -> None:
    """Rewrite a sealed file and record its new hash in a newly committed evidence file."""
    path = u.sealed / name
    path.chmod(0o644)
    path.write_bytes(gc.canonical_bytes(body))
    u.evidence["sealed"]["files"][name] = gc.sha256_bytes(gc.canonical_bytes(body))
    commit_evidence(u.env, u.evidence)


def test_a_sealed_result_that_the_code_does_not_recompute_is_refused(unsealed, capsys):
    agg = json.loads((unsealed.sealed / gc.SEALED_AGGREGATES).read_text(encoding="utf-8"))
    agg["breaker"]["trip"] = not agg["breaker"]["trip"]
    _resign(unsealed, gc.SEALED_AGGREGATES, agg)
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and r.out == ""
    assert f"the recomputed {gc.SEALED_AGGREGATES} differs from the sealed one" in r.err


def test_a_queue_that_the_code_does_not_recompute_is_refused(unsealed, capsys):
    q = json.loads((unsealed.sealed / gc.SEALED_QUEUE).read_text(encoding="utf-8"))
    q["items"] = q["items"][1:]
    _resign(unsealed, gc.SEALED_QUEUE, q)
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and f"the recomputed {gc.SEALED_QUEUE} differs" in r.err


def test_a_sealed_file_that_does_not_match_its_recorded_hash_is_refused(unsealed, capsys):
    path = unsealed.sealed / gc.SEALED_GRADER_REPLIES
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED
    assert f"the sealed file {gc.SEALED_GRADER_REPLIES} does not match the SHA-256" in r.err


def test_a_missing_sealed_file_is_refused(unsealed, capsys):
    (unsealed.sealed / gc.SEALED_DRAWN).unlink()
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and f"the sealed file {gc.SEALED_DRAWN} is missing" in r.err


def test_an_uncommitted_change_to_the_evidence_is_refused(unsealed, capsys):
    path = unsealed.env.repo / EVIDENCE_REL
    path.write_bytes(path.read_bytes() + b"\n")
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and "differs from the HEAD commit" in r.err


def test_evidence_that_names_another_worker_log_is_refused(unsealed, capsys):
    unsealed.evidence["worker_logs"][0]["sha256"] = "0" * 64
    commit_evidence(unsealed.env, unsealed.evidence)
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED
    assert "names 1 worker log(s) the frozen manifest does not and leaves out 1" in r.err


def test_evidence_of_another_run_commit_is_refused(unsealed, capsys):
    unsealed.evidence["worker_run_commit"] = "ffffff0"
    commit_evidence(unsealed.env, unsealed.evidence)
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and "worker run commit is not the frozen manifest's run commit" in r.err


def test_a_unit_without_a_grader_reply_of_record_is_refused(unsealed, capsys):
    body = json.loads((unsealed.sealed / gc.SEALED_GRADER_REPLIES).read_text(encoding="utf-8"))
    body["replies"].pop(sorted(body["replies"])[0])
    _resign(unsealed, gc.SEALED_GRADER_REPLIES, body)
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and "1 unit(s) have no grader reply of record" in r.err


def test_a_sealed_folder_with_another_name_is_refused(unsealed, capsys):
    other = unsealed.sealed.with_name("sealed")
    unsealed.sealed.rename(other)
    unsealed.sealed = other
    r = run(unsealed, capsys)
    assert r.code == ur.EXIT_REFUSED and "must be named _sealed" in r.err


def test_run_report_without_the_unseal_step_still_lists_the_judge_items(small, small_logs):
    commit_second(small)
    workers, template = small_items(small_logs)
    from test_run_report import report
    _, payload = report(small, workers, template)
    assert payload["withheld"] is False and "judge" not in payload
    assert set(JUDGE_ITEMS) <= set(payload["not_available"])
