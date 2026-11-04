"""Provenance of the vendored Inspect Scout scanners, and the committed diff.

The three scanners ship in inspect_scout as documentation examples under
examples/docs, not as library code, so they are copied from the commit the 0.5.5
release tag points at. The pristine copies sit in docs/evidence/scout-upstream/
at their upstream relative paths; the working copies sit in this package.

render_diff() rebuilds docs/evidence/scout-scanners.diff from those two sets of
files: a unified diff (difflib, 3 lines of context) of each full file, the
provenance header lines included, in VENDORED order. Regenerate it with

    python -m icfr_agent_evals.scanners

from a repository checkout; tests/test_scanners.py checks that the committed
file matches this output byte for byte.
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

UPSTREAM_REPO = "https://github.com/meridianlabs-ai/inspect_scout"
UPSTREAM_TAG = "0.5.5"
UPSTREAM_COMMIT = "2db653d5fbd1352201b31a74557f612ee78a058c"
RETRIEVED = "2026-11-04"
UPSTREAM_LICENSE = "MIT"

# upstream path -> (git blob SHA-1 at UPSTREAM_COMMIT, vendored module path)
VENDORED: dict[str, tuple[str, str]] = {
    "examples/docs/eval_awareness.py": (
        "185489cfe2383198e3a9155dc291802b622b084f",
        "icfr_agent_evals/scanners/eval_awareness.py",
    ),
    "examples/docs/refusal_classifier.py": (
        "dac129c783dbec35c13c35162df52747b0adf122",
        "icfr_agent_evals/scanners/refusal_classifier.py",
    ),
    "examples/docs/command_not_found.py": (
        "06177a5f2aaaaf4a62d97b822dcdb03ef94c01d3",
        "icfr_agent_evals/scanners/command_not_found.py",
    ),
}
LICENSE_BLOB = ("LICENSE", "4f47c9420169a79f1b64036506d27d83df5306f1")

EVIDENCE_DIR = "docs/evidence/scout-upstream"
DIFF_PATH = "docs/evidence/scout-scanners.diff"

# Cue words removed from the eval_awareness rubric: ordinary vocabulary in a
# controls-testing task (section 5, Read the transcripts). The upstream rubric
# lists no inflections of either word.
DROPPED_AWARENESS_CUES: tuple[str, ...] = ("test", "audit")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_blob_sha(data: bytes) -> str:
    """The SHA-1 git assigns a blob with this content (git hash-object)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8").splitlines(keepends=True)


def render_diff(root: Path | None = None) -> str:
    root = root or repo_root()
    out: list[str] = []
    for upstream, (_, vendored) in VENDORED.items():
        out.extend(
            difflib.unified_diff(
                _lines(root / EVIDENCE_DIR / upstream),
                _lines(root / vendored),
                fromfile=f"a/{upstream}",
                tofile=f"b/{vendored}",
                n=3,
            )
        )
    return "".join(out)


def main() -> int:
    root = repo_root()
    (root / DIFF_PATH).write_bytes(render_diff(root).encode("utf-8"))
    print(f"wrote {DIFF_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
