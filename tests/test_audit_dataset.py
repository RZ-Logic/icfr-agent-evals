"""scripts/audit_dataset.py: the one-command dataset audit an outsider runs.

The end-to-end tests copy this repository's HEAD tree into a temporary git repo and
audit the dev split there: once clean, once with a committed answer key edited in
data/dev/dev.jsonl, and once with code that edits one answer key as the split is
built. Log headers are made-up EvalLogs holding only task metadata. The per-case
checks also run in process on a made-up case. No test calls a model, and every
printed report is checked for case content.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import subprocess

import pytest

from casefactory import good_output, make_truth
from icfr_agent_evals.casefile import to_sample
from icfr_agent_evals.generator import CaseSpec, build_case

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_dataset.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load()

CODE_TAMPER = '''

_audit_test_build_dataset = build_dataset


def build_dataset(*args, **kwargs):
    data = _audit_test_build_dataset(*args, **kwargs)
    if kwargs.get("split", args[0] if args else "eval") == "dev":
        first = sorted(data, key=lambda s: str(s.id))[0]
        first.metadata["truth"]["explained_usd"] = first.metadata["truth"]["explained_usd"] + 1.0
    return data
'''


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True,
                          text=True, timeout=300).stdout.strip()


def git_blob(repo: pathlib.Path, spec: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", "cat-file", "blob", spec],
                          capture_output=True, check=True, timeout=300).stdout


def _commit(repo: pathlib.Path, message: str, tag: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    git(repo, "tag", tag)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A temporary git repo holding HEAD's tree, with three tagged commits: clean,
    data-tampered (one answer key edited in data/dev/dev.jsonl) and code-tampered (the
    dev build edits one answer key; dev.jsonl restored)."""
    import tarfile

    r = tmp_path_factory.mktemp("audit-repo")
    blob = subprocess.run(["git", "-C", str(REPO), "-c", "core.autocrlf=false", "archive", "--format=tar", "HEAD"],
                          capture_output=True, check=True, timeout=300).stdout
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        tar.extractall(r, filter="data")
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "Test")
    git(r, "config", "user.email", "test@example.com")
    git(r, "config", "commit.gpgsign", "false")
    git(r, "config", "core.autocrlf", "false")
    _commit(r, "clean", "clean")

    dev = r / "data" / "dev" / "dev.jsonl"
    original = dev.read_bytes()
    rows = original.decode("utf-8").splitlines()
    first = json.loads(rows[0])
    first["metadata"]["truth"]["explained_usd"] = first["metadata"]["truth"]["explained_usd"] + 1.0
    rows[0] = json.dumps(first, sort_keys=True, ensure_ascii=False)
    dev.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    _commit(r, "edit one answer key in dev.jsonl", "data-tampered")

    dev.write_bytes(original)
    ds = r / "icfr_agent_evals" / "dataset.py"
    ds.write_bytes(ds.read_bytes() + CODE_TAMPER.encode("utf-8"))
    _commit(r, "edit one answer key in the dev build", "code-tampered")
    return r


@pytest.fixture(scope="module")
def dev_rows(repo):
    """The clean dev split rows, read from the temp repo's committed file."""
    text = git_blob(repo, "clean:data/dev/dev.jsonl").decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _content_strings(rows) -> set[str]:
    """Case content that a report must never show: entity strings and input lines."""
    out: set[str] = set()
    for row in rows:
        t = row["metadata"]["truth"]
        out.update(str(x) for x in t["valid_entities"])
        out.update(str(x) for x in t["organizations"])
        out.update(str(x) for x in t["scope_wallets"])
        for item in t["items"]:
            out.add(str(item["reference"]))
            out.update(str(a) for a in item["aliases"])
        if t.get("injection"):
            out.add(str(t["injection"]["canary"]))
        for line in str(row["input"]).splitlines():
            line = line.strip()
            if len(line) >= 25 and sum(ch.isalpha() for ch in line) >= 10:
                out.add(line)
    return {s for s in out if len(s) >= 4}


def _run_main(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = audit.main(list(argv))
    return code, buf.getvalue()


def _rows(report: dict, scope: str) -> dict[str, dict]:
    return {r["check"]: r for r in report["checks"] if r["scope"] == scope}


# --- End to end on a temporary repo -----------------------------------------------------

def test_clean_head_passes_and_prints_no_case_content(repo, dev_rows):
    code, out = _run_main("--ref", "clean", "--repo", str(repo), "--splits", "dev")
    assert code == 0, out
    assert "Result: PASS" in out
    for check in ("validate_truth", "entity_closure", "wallet_labels", "perfect_answer",
                  "committed file", "COMPOSITION.json", "section 13 hash", "split seed", "composition"):
        assert f"PASS  {check}:" in out
    assert audit.RECORDED_SECTION13["dev"] in out
    leaked = [s for s in _content_strings(dev_rows) if s in out]
    assert leaked == []


def test_report_json_lists_hashes_and_counts(repo):
    report = audit.audit("clean", repo, ["dev"])
    assert report["failed"] == 0
    dev = report["splits"]["dev"]
    assert dev["n"] == 30
    assert dev["case_kind"] == {"adversarial": 10, "edge": 6, "standard": 14}
    assert dev["dataset_sha256"] == audit.RECORDED_SECTION13["dev"]
    committed = git_blob(repo, "clean:data/dev/dev.jsonl")
    assert dev["split_jsonl_sha256"] == hashlib.sha256(committed).hexdigest()
    assert dev["pilot_n"] == 20
    assert _rows(report, "ref")["section 13 text"]["status"] == "PASS"


def test_answer_key_edited_in_committed_file_fails(repo, dev_rows):
    code, out = _run_main("--ref", "data-tampered", "--repo", str(repo), "--splits", "dev")
    assert code == 1
    assert "FAIL  committed file: data/dev/dev.jsonl differs from the rebuilt split" in out
    assert "FAIL  COMPOSITION.json" in out and "file_sha256" in out
    assert "Result: FAIL" in out
    assert [s for s in _content_strings(dev_rows) if s in out] == []


def test_answer_key_edited_in_code_fails_by_case_id(repo, dev_rows):
    report = audit.audit("code-tampered", repo, ["dev"])
    rows = _rows(report, "dev")
    first_id = sorted(str(r["id"]) for r in dev_rows)[0]
    assert rows["validate_truth"]["status"] == "FAIL"
    assert [f["id"] for f in rows["validate_truth"]["failed"]] == [first_id]
    assert "explained_usd" in rows["validate_truth"]["failed"][0]["reason"]
    assert rows["section 13 hash"]["status"] == "FAIL"
    assert rows["committed file"]["status"] == "FAIL"
    assert rows["entity_closure"]["status"] == "PASS"
    code, out = _run_main("--ref", "code-tampered", "--repo", str(repo), "--splits", "dev")
    assert code == 1
    assert first_id in out
    assert [s for s in _content_strings(dev_rows) if s in out] == []


def test_unknown_ref_cannot_run(repo, capsys):
    assert audit.main(["--ref", "no-such-ref", "--repo", str(repo), "--splits", "dev"]) == 2
    assert "audit could not run" in capsys.readouterr().err


def test_child_environment_drops_provider_keys(monkeypatch, tmp_path):
    for name in audit.STRIPPED_ENV:
        monkeypatch.setenv(name, "x")
    env = audit.child_env(tmp_path)
    assert not set(audit.STRIPPED_ENV) & set(env)
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["PYTHON_DOTENV_DISABLED"] == "1"


# --- Log headers ------------------------------------------------------------------------

def _write_log(path: pathlib.Path, metadata: dict) -> None:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log

    spec = EvalSpec(created="2026-11-18T00:00:00+00:00", task="variance_commentary",
                    dataset=EvalDataset(name="made-up"), model="mockllm/model", config=EvalConfig(),
                    metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(EvalLog(eval=spec), str(path))


SPLITS = {"dev": {"built": True, "split_seed": 21260607, "dataset_sha256": "a" * 64,
                  "pilot_dataset_sha256": "b" * 64}}


def _meta(**over) -> dict:
    meta = {"split": "dev", "subset": None, "split_seed": 21260607,
            "dataset_sha256": "a" * 64, "split_dataset_sha256": "a" * 64}
    meta.update(over)
    return meta


def test_log_headers_match_and_mismatch(tmp_path):
    logs = tmp_path / "logs"
    _write_log(logs / "good.eval", _meta())
    _write_log(logs / "pilot.eval", _meta(subset="pilot", dataset_sha256="b" * 64))
    _write_log(logs / "nested" / "bad-split-hash.eval", _meta(split_dataset_sha256="c" * 64))
    _write_log(logs / "bad-subset-hash.json", _meta(subset="pilot", dataset_sha256="a" * 64))
    _write_log(logs / "bad-seed.eval", _meta(split_seed=1))
    _write_log(logs / "other-split.eval", _meta(split="ci"))
    (logs / "broken.eval").write_bytes(b"not a log")
    rows = {r["check"]: r for r in audit.check_logs(logs, SPLITS)}
    assert rows["good.eval"]["status"] == "PASS"
    assert rows["pilot.eval"]["status"] == "PASS"
    assert rows["nested/bad-split-hash.eval"]["status"] == "FAIL"
    assert "split_dataset_sha256" in rows["nested/bad-split-hash.eval"]["detail"]
    assert rows["bad-subset-hash.json"]["status"] == "FAIL"
    assert "dataset_sha256" in rows["bad-subset-hash.json"]["detail"]
    assert rows["bad-seed.eval"]["status"] == "FAIL" and "split_seed" in rows["bad-seed.eval"]["detail"]
    assert rows["other-split.eval"]["status"] == "FAIL"
    assert rows["broken.eval"]["status"] == "FAIL"


def test_empty_log_folder_fails(tmp_path):
    rows = audit.check_logs(tmp_path, SPLITS)
    assert [r["status"] for r in rows] == ["FAIL"]


def test_log_hash_mismatch_fails_end_to_end(repo, tmp_path):
    good = tmp_path / "good"
    _write_log(good / "run.eval", _meta(dataset_sha256=audit.RECORDED_SECTION13["dev"],
                                        split_dataset_sha256=audit.RECORDED_SECTION13["dev"]))
    bad = tmp_path / "bad"
    _write_log(bad / "run.eval", _meta(dataset_sha256=audit.RECORDED_SECTION13["dev"],
                                       split_dataset_sha256="0" * 64))
    report = audit.audit("clean", repo, ["dev"], good)
    assert report["failed"] == 0
    assert _rows(report, "logs")["run.eval"]["status"] == "PASS"
    code, out = _run_main("--ref", "clean", "--repo", str(repo), "--splits", "dev", "--logs", str(bad))
    assert code == 1
    assert "FAIL  run.eval: split dev, subset none: split_dataset_sha256 differ from the rebuild" in out


# --- Per-case checks on a made-up case --------------------------------------------------

def _made_up_sample():
    spec = CaseSpec(split="dev", case_id="dev-std-901-ETH-M2", asset="ETH", period="2025-11",
                    variance_usd=-123_456.78, seed=987_654_321, evidence_state="partly_unexplained",
                    template_params={"band": "M2", "direction": "shortfall"})
    return to_sample(build_case(spec))


def _edited(sample, **truth_edits):
    meta = json.loads(json.dumps(sample.metadata))
    meta["truth"].update(truth_edits)
    return sample.model_copy(update={"metadata": meta})


def test_made_up_case_passes_every_check():
    res = audit.audit_cases([_made_up_sample()], "dev")
    assert {name: r["failed"] for name, r in res.items()} == {name: [] for name in audit.CASE_CHECKS}
    assert all(r["checked"] == 1 for r in res.values())


@pytest.mark.parametrize("edit", [
    {"explained_usd": 1.0},
    {"materiality_band": "M3"},
    {"case_id": "dev-std-902-ETH-M2"},
    {"split": "ci"},
])
def test_edited_answer_key_fails_validate_truth(edit):
    sample = _made_up_sample()
    res = audit.audit_cases([_edited(sample, **edit)], "dev")
    assert [f["id"] for f in res["validate_truth"]["failed"]] == [str(sample.id)]


def test_unloadable_answer_key_fails_every_check():
    sample = _made_up_sample()
    meta = dict(sample.metadata)
    meta.pop("truth")
    res = audit.audit_cases([sample.model_copy(update={"metadata": meta})], "dev")
    assert all(len(r["failed"]) == 1 for r in res.values())


def test_perfect_answer_is_casefactory_good_output():
    for asset in ("BTC", "ETH", "SOL", "XRP", "ADA", "USDC"):
        truth = make_truth(asset)
        assert audit.perfect_answer(truth) == good_output(truth)
    surplus = make_truth("BTC", onchain=10_200_000.00)
    assert audit.perfect_answer(surplus) == good_output(surplus)


def test_mask_hides_entities_quotes_and_numbers():
    truth = make_truth("BTC")
    ref = truth.items[0].reference
    text = f"reconciling_items[0] {ref!r} amount 80,000.00 differs; FM-005 at {truth.scope_wallets[0]} \"x y\""
    masked = audit.mask(text, truth, truth.case_id)
    assert ref not in masked and truth.scope_wallets[0] not in masked
    assert "80,000.00" not in masked and "x y" not in masked
    assert "FM-005" in masked
    assert not any(ch.isdigit() for ch in masked.replace("FM-005", ""))
