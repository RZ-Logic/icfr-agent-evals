# Inspect Scout eval_awareness: reference copy at PR #679

This is a byte-for-byte copy of one upstream file, kept so the backport in the
vendored awareness scanner can be checked. It is not imported by the package.

| Field | Value |
|---|---|
| Repository | https://github.com/meridianlabs-ai/inspect_scout |
| Path | `examples/docs/eval_awareness.py` (kept here at the same relative path) |
| Commit | `2653f933d634f3f7e8f20c7d49ba3ff26f30fd16`, the merge commit of PR #679 ("fix(examples): eval_awareness no longer counts topic vocabulary as awareness", merged 2026-10-20) |
| First release | inspect-scout 0.6.1 (2026-10-21); 0.6.0 still has the 0.5.5 blob |
| git blob SHA-1 | `858cf99c747feb00530176ab4e594ae808ab298d` |
| Retrieved | 2026-11-05, with the GitHub contents API (`gh api`, raw media type) |
| Upstream license | MIT, Copyright (c) 2025 Meridian Labs. `LICENSE` at this commit has the same blob as at 0.5.5 (`4f47c9420169a79f1b64036506d27d83df5306f1`), so the full notice is the copy in `docs/evidence/scout-upstream/LICENSE`, also reproduced in `icfr_agent_evals/scanners/__init__.py` |

Why it is kept: inspect-scout stays pinned at 0.5.5, so the vendored scanners
come from the 0.5.5 release commit (`docs/evidence/scout-upstream/`). The rubric
text of PR #679 is backported into `icfr_agent_evals/scanners/eval_awareness.py`
by hand, with one local sentence added after it. `tests/test_scanners.py` checks
that this file still has the blob SHA above and that the vendored rubric equals
the rubric here plus exactly that one sentence. `docs/evidence/scout-upstream/README.md`
describes the backport and the local sentence.
