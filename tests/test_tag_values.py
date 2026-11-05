"""scripts/tag_values.py: the section 13 tag-time values, computed read-only.

The script builds splits by code and hashes them; it never scores and never calls
a model. These tests run it twice in fresh interpreters for determinism, check the
CI composition against the pre-registered numbers, check the generator source set
against the modules a fresh interpreter loads, and check that the combined source
hash moves when one file's bytes move (on a temp copy, never the real files).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tag_values.py"


def _load():
    spec = importlib.util.spec_from_file_location("tag_values", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tv = _load()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, capture_output=True, check=False,
                          timeout=600)


@pytest.fixture(scope="module")
def two_runs():
    first, second = _run(), _run()
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    assert second.returncode == 0, second.stderr.decode("utf-8", "replace")
    return first.stdout, second.stdout


def test_output_is_deterministic_across_two_runs(two_runs):
    first, second = two_runs
    assert first == second
    assert json.loads(first)["checks"]["mismatches"] == []


def test_ci_composition_equals_preregistered(two_runs):
    values = json.loads(two_runs[0])
    ci = values["splits"]["ci"]
    assert ci["n"] == 200
    assert ci["case_kind"] == {"standard": 146, "edge": 30, "adversarial": 24}
    assert values["splits"]["ci_matches_preregistered"] is True
    for cat in ("band_boundary", "ambiguous_narrative", "incomplete_data"):
        assert ci["category"][cat] == 10
    for n in range(1, 7):
        assert ci["category"][f"AC-{n}"] == 4
    assert values["seeds"]["eval_seed"] == 20260607
    assert values["seeds"]["split_seeds"]["ci"] == 22260607


def test_values_cover_section13_fields(two_runs):
    values = json.loads(two_runs[0])
    g = values["generator"]
    assert g["generator_version"] and g["hard_generator_version"]
    assert len(g["source_sha256"]) == 64
    assert g["source_sha256"] == tv.combine_hashes(g["source_files"])
    for split in ("dev", "ci", "eval"):
        assert len(values["splits"][split]["dataset_sha256"]) == 64
    assert values["splits"]["eval"]["n"] == 146
    assert values["dataset"]["dataset_version"] and values["dataset"]["schema_version"]
    prompt = values["worker_prompt"]
    assert prompt["worker_prompt_sha256"] == prompt["recomputed_sha256"]
    env = values["environment"]
    assert env["python_build"] and env["openssl"]


def test_environment_records_the_interpreter_build_and_openssl():
    import ssl

    env = tv._versions()
    assert env["python_build"] == sys.version
    assert env["openssl"] == ssl.OPENSSL_VERSION


def test_output_holds_no_case_content(two_runs):
    """Only hashes, counts and versions leave the script: no rendered fixture text, no case ids."""
    text = two_runs[0].decode("utf-8")
    assert "eval-std-" not in text and "ci-std-" not in text and "dev-std-" not in text
    assert "Brindlecote" not in text


def test_closure_includes_generator_and_hardcases():
    files = tv.generator_source_files()
    assert "icfr_agent_evals/generator.py" in files
    assert "icfr_agent_evals/hardcases.py" in files
    assert "icfr_agent_evals/dataset.py" in files
    assert "icfr_agent_evals/__init__.py" in files
    # Scoring code is not generator source.
    assert "icfr_agent_evals/scoring.py" not in files
    assert "icfr_agent_evals/solver.py" not in files


def test_closure_matches_the_modules_a_fresh_build_loads():
    code = (
        "import sys, pathlib, json; sys.path.insert(0, sys.argv[1]);"
        "from icfr_agent_evals.dataset import build_cases;"
        "build_cases('dev'); build_cases('ci');"
        "root = pathlib.Path(sys.argv[1]).resolve();"
        "print(json.dumps(sorted(pathlib.Path(m.__file__).resolve().relative_to(root).as_posix()"
        " for n, m in list(sys.modules.items()) if n.split('.')[0] == 'icfr_agent_evals')))"
    )
    out = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, check=True, timeout=600)
    assert json.loads(out.stdout) == tv.generator_source_files()


def test_combined_hash_changes_when_one_generator_file_changes(tmp_path):
    files = tv.generator_source_files()
    for rel in files:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dest)
    assert tv.generator_source_files(tmp_path) == files
    before = tv.source_sha256(files, tmp_path)
    assert before == tv.source_sha256(files, REPO)
    target = tmp_path / "icfr_agent_evals" / "generator.py"
    target.write_bytes(target.read_bytes() + b"\n")
    after = tv.source_sha256(files, tmp_path)
    assert after != before
    # Only the changed file's own hash moved.
    old, new = tv.file_hashes(files, REPO), tv.file_hashes(files, tmp_path)
    assert {p for p in files if old[p] != new[p]} == {"icfr_agent_evals/generator.py"}


def test_combine_rule_is_sorted_path_sha_lines():
    import hashlib

    hashes = {"b.py": "2" * 64, "a.py": "1" * 64}
    expected = hashlib.sha256(f"a.py {'1' * 64}\nb.py {'2' * 64}\n".encode()).hexdigest()
    assert tv.combine_hashes(hashes) == expected


def test_markdown_mode_drafts_section13_rows():
    out = _run("--markdown")
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    text = out.stdout.decode("utf-8")
    assert text.startswith("| Item | Value at tag (draft) |")
    assert "Generator version G" in text and "22260607" in text
    assert chr(0x2014) not in text  # no em dash in the drafted rows
