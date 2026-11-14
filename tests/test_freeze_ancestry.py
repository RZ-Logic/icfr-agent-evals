"""Section 8 freeze-commit check at settlement (scripts/pilot_report.py freeze_commit_gaps).

The settlement script refuses a review record unless its manifest was frozen in a commit
that holds the manifest byte for byte, carries the reviewer's trailer in its closing trailer
block, and is an ancestor of HEAD, the settlement commit. Each test builds a scratch git
repository; nothing here touches this repository's history. No API key, no model call.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pilot_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("pilot_report_ancestry", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _load()
REVIEWER = "Test Reviewer"
DAY = date.today().isoformat()
TRAILER = f"Reviewed-by: {REVIEWER}, {DAY}"
CO = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
REL = "docs/evidence/reviews/run-x-manifest.json"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "Test")
    git(r, "config", "user.email", "test@example.com")
    git(r, "config", "commit.gpgsign", "false")
    git(r, "config", "core.autocrlf", "false")
    (r / "README.md").write_text("start\n", encoding="utf-8")
    git(r, "add", "README.md")
    git(r, "commit", "-q", "-m", "start")
    return r


def manifest_bytes() -> bytes:
    man = {"manifest_version": "1", "kind": "review_freeze", "run_id": "run-x",
           "reviewer": REVIEWER, "freeze_date": DAY}
    return (json.dumps(man, indent=2) + "\n").encode("utf-8")


def write_manifest(repo: Path) -> Path:
    path = repo / REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest_bytes())
    return path


def freeze(repo: Path, message: str) -> str:
    write_manifest(repo)
    git(repo, "add", REL)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def check(repo: Path, commit: str | None = None, path: Path | None = None):
    mpath = path or (repo / REL)
    mbytes = mpath.read_bytes()
    return pr.freeze_commit_gaps(repo, mpath, mbytes, json.loads(mbytes), commit)


def later_commit(repo: Path, name: str = "later.txt") -> None:
    (repo / name).write_text("later\n", encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", "later work")


def test_a_frozen_manifest_on_heads_history_passes(repo):
    sha = freeze(repo, f"Freeze review record\n\n{TRAILER}\n{CO}\n")
    later_commit(repo)
    gaps, found = check(repo)
    assert gaps == [] and found == sha


def test_the_freeze_commit_may_be_head_itself(repo):
    sha = freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    assert check(repo) == ([], sha)


def test_an_explicit_freeze_commit_passes_when_it_is_the_right_one(repo):
    sha = freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    later_commit(repo)
    assert check(repo, sha[:7]) == ([], sha)


def test_an_uncommitted_manifest_is_not_frozen(repo):
    write_manifest(repo)
    gaps, found = check(repo)
    assert found == "" and len(gaps) == 1 and "it is not frozen" in gaps[0]


def test_a_freeze_commit_without_the_reviewer_trailer_refuses(repo):
    freeze(repo, "Freeze review record\n")
    gaps, _ = check(repo)
    assert any("closing trailer block" in g for g in gaps)


def test_a_freeze_commit_with_another_reviewer_refuses(repo):
    freeze(repo, f"Freeze review record\n\nReviewed-by: Someone Else, {DAY}\n")
    gaps, _ = check(repo)
    assert any("closing trailer block" in g for g in gaps)


def test_a_manifest_changed_after_the_freeze_refuses(repo):
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    path = repo / REL
    path.write_bytes(path.read_bytes().replace(b'"run-x"', b'"run-x" '))
    gaps, _ = check(repo)
    assert any("differs from the manifest file" in g for g in gaps)


def test_a_freeze_commit_off_heads_history_refuses(repo):
    git(repo, "checkout", "-q", "-b", "side")
    sha = freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    git(repo, "checkout", "-q", "main")
    write_manifest(repo)  # the same bytes, uncommitted on main
    gaps, found = check(repo, sha)
    assert found == sha
    assert any("is not an ancestor of the settlement commit (HEAD)" in g for g in gaps)


def test_a_commit_that_does_not_hold_the_manifest_refuses(repo):
    other = git(repo, "rev-parse", "HEAD")
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    gaps, _ = check(repo, other)
    assert any("does not hold" in g for g in gaps)


def test_a_manifest_added_twice_needs_an_explicit_commit(repo):
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    git(repo, "rm", "-q", REL)
    git(repo, "commit", "-q", "-m", "remove")
    sha = freeze(repo, f"Freeze again\n\n{TRAILER}\n")
    gaps, found = check(repo)
    assert found == "" and "added in 2 commits" in gaps[0]
    assert check(repo, sha) == ([], sha)


@pytest.mark.parametrize("bad", ["HEAD", "xyz1234", "abc"])
def test_a_malformed_freeze_commit_refuses(repo, bad):
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    gaps, found = check(repo, bad)
    assert found == "" and "is not a hex commit hash" in gaps[0]


def test_an_unknown_freeze_commit_refuses(repo):
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    gaps, found = check(repo, "0" * 40)
    assert found == "" and "cannot be read" in gaps[0]


def test_a_manifest_outside_the_repository_refuses(repo, tmp_path):
    outside = tmp_path / "elsewhere" / "run-x-manifest.json"
    outside.parent.mkdir()
    outside.write_bytes(manifest_bytes())
    gaps, found = check(repo, path=outside)
    assert found == "" and "outside this repository" in gaps[0]


def test_settlement_has_no_minimum_age(repo):
    """The 48-hour rule belongs to the second look; settlement may follow the freeze at once."""
    freeze(repo, f"Freeze review record\n\n{TRAILER}\n")
    gaps, _ = check(repo)
    assert not any("48 hours" in g for g in gaps)
