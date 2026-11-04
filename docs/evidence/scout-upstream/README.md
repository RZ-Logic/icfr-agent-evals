# Inspect Scout scanners: pristine upstream copies

These are byte-for-byte copies of the three Inspect Scout scanners this harness
vendors (pre-registration section 5, Scout scanner model row; section 9,
exploratory scanners). Inspect Scout ships them as documentation examples under
`examples/docs`, not as library code: the installed inspect-scout 0.5.5 package
contains none of the three names.

| Field | Value |
|---|---|
| Repository | https://github.com/meridianlabs-ai/inspect_scout |
| Release tag | `0.5.5` (a lightweight tag pointing straight at the commit) |
| Commit | `2db653d5fbd1352201b31a74557f612ee78a058c` ("chore(main): release 0.5.5 (#691)", 2026-10-11T13:19:01Z) |
| Retrieved | 2026-11-04, with the GitHub contents API (`gh api`, raw media type) |
| Upstream license | MIT, Copyright (c) 2025 Meridian Labs (`LICENSE` in this directory, copied from the same commit) |

| Upstream path (kept here at the same relative path) | git blob SHA-1 at the commit | Vendored copy |
|---|---|---|
| `examples/docs/eval_awareness.py` | `185489cfe2383198e3a9155dc291802b622b084f` | `icfr_agent_evals/scanners/eval_awareness.py` |
| `examples/docs/refusal_classifier.py` | `dac129c783dbec35c13c35162df52747b0adf122` | `icfr_agent_evals/scanners/refusal_classifier.py` |
| `examples/docs/command_not_found.py` | `06177a5f2aaaaf4a62d97b822dcdb03ef94c01d3` | `icfr_agent_evals/scanners/command_not_found.py` |
| `LICENSE` | `4f47c9420169a79f1b64036506d27d83df5306f1` | notice reproduced in `icfr_agent_evals/scanners/__init__.py` |

The blob SHAs are the ones the GitHub git trees API reports for the commit, and
`git hash-object` on each file here gives the same value.

The 0.5.5 tag commit is the release commit: PyPI lists the 0.5.5 wheel and sdist
as uploaded at 2026-10-11T13:20:24Z and 13:20:26Z, about a minute after it, and
the installed package's `_scanjob.py`, `_scanjob_config.py`, `_scan.py`,
`__init__.py` and `_llm_scanner/_llm_scanner.py` have the same blob SHAs as the
files at the commit.

## Changes in the vendored copies

The complete diff from these files to the vendored copies is
`docs/evidence/scout-scanners.diff`. It is a unified diff (Python difflib, three
lines of context) of each full file, the added provenance header included, and
`python -m icfr_agent_evals.scanners` regenerates it. `tests/test_scanners.py`
checks that the regenerated diff matches the committed file byte for byte and
that each file here still has the blob SHA recorded above. The changes:

1. `eval_awareness.py`: the rubric's list of awareness cue words drops `'test'`
   and `'audit'`. Both are ordinary vocabulary in a controls-testing task (a
   worker reconciling reserves describes control tests and audit evidence), so
   as cues they would mark routine task language as evaluation awareness. The
   upstream list has no inflections of either word. The rest of the rubric is
   unchanged, including the definition sentence ("being evaluated, tested, or
   placed in an artificial scenario"), which describes the construct rather
   than listing cue words.
2. `eval_awareness.py` and `refusal_classifier.py`: the upstream examples pass no
   model to `llm_scanner`, so they would use whatever model the scan defaults to.
   Each now takes `model` with the default `roster.SCOUT_SCANNER`
   (`openai/gpt-6-luna`) and passes it to `llm_scanner` explicitly. Only the pin
   or a `mockllm` model (offline tests) is accepted.
3. All three files: a four-line provenance header, and in the two llm scanners
   the two imports change 2 needs. `command_not_found.py` is otherwise unchanged;
   it is a regex scanner and uses no model.

Upstream code is otherwise kept as is, including the `awareness_to_float` error
message that lacks its `f` prefix.
