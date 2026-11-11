# Pre-tag checks, 2026-11-04

Section 13 rows of `PRE-REGISTRATION-PHASE-1.md` that were checked on 2026-11-04, before the tag. Each row still records its value at the tag, so every check below is run again on the commit to be tagged. Sources were read, not written to: no form, capture or upload on any service, no model call, and no evaluation-split case sent to a model or scored.

## Framework and dependency pins

Superseded for Python and uv: see `docs/evidence/pretag-checks-2026-11-05.md` (Python and uv). The Python rows and paragraph below stay as the 2026-11-04 record.

| Claim | Result | Source |
|---|---|---|
| inspect-ai 0.3.278 is on PyPI | Uploaded 2026-10-09 16:55 UTC, not yanked | https://pypi.org/project/inspect-ai/0.3.278/ |
| 0.3.278 is the first release after 0.3.277 whose CHANGELOG carries the Claude Haiku 5.5 support entry | The 0.3.278 section has the Anthropic entry for Claude Haiku 5.5 (`claude-haiku-5-5`); the 0.3.277 section has none | UKGovernmentBEIS/inspect_ai `CHANGELOG.md` |
| 0.3.277's Haiku 5.5 gaps (reasoning_effort "none", forwarded fallback_models, computer toolset, thinking-block binding) are closed in 0.3.278 | All four are addressed by commit 6d252fe (#5752) in the 0.3.277...0.3.278 compare | UKGovernmentBEIS/inspect_ai compare and commit pages |
| inspect-scout 0.5.5 is the newest release that accepts inspect-ai 0.3.278 | 0.5.5 requires inspect-ai>=0.3.278; 0.6.0, 0.6.1 and 0.6.2 require >=0.3.281, >=0.3.283 and >=0.3.286 | PyPI `requires_dist` for each release |
| inspect-scout 0.5.0 changed the default scan-results columns | Confirmed as a breaking change in the 0.5.0 entry | meridianlabs-ai/inspect_scout `CHANGELOG.md` |
| anthropic 1.12.1 exists, and the provider failed on anthropic 2.0 until inspect-ai 0.3.283 | 1.12.1 uploaded 2026-10-08; the 0.3.283 CHANGELOG and issue #6003 record the anthropic 2.0 failure and its fix | PyPI; UKGovernmentBEIS/inspect_ai |
| openai 3.27.0 is the newest openai release published before inspect-ai 0.3.278 | 3.27.0 uploaded 2026-10-09 13:48 UTC, before 0.3.278 at 16:55 UTC | PyPI upload times |
| Python 3.14.7 and pydantic 2.13.4 | CPython 3.14.7 is a python.org release of 2026-08-05; pydantic 2.13.4 (2026-05-06) carries the Python 3.14 classifier and pins pydantic-core 2.46.4, which ships cp314 wheels; pytest 9.0.3 is on PyPI | python.org; PyPI |
| The pinned set installs and is consistent | `uv pip check` reports the set compatible on Python 3.14.7 and 3.12.11; the full suite passes on both | Measured |

No GitHub advisory affects a pinned version. Releases published after the pins, recorded here and not adopted: CPython 3.14.8 to 3.14.10 (security fixes in ssl, tarfile, urllib, zipfile and http.cookiejar, and updated bundled pip, Expat and OpenSSL); anthropic 1.13.0 (2026-10-09 15:29 UTC, before inspect-ai 0.3.278; the "newest before" rule is stated for openai only); inspect-scout 0.6.1 and 0.6.2, which need a newer inspect-ai than the pin.

## Model retirement floors, notice policies and serving platform

The model status table on https://platform.claude.com/docs/en/about-claude/model-deprecations lists the four Anthropic roster models as Active, with tentative retirement dates of not sooner than:

| Model | Floor in `roster.py` | Page |
|---|---|---|
| `anthropic/claude-sonnet-5` | 2027-06-30 | June 30, 2027 |
| `anthropic/claude-opus-5-5` | 2027-09-22 | September 22, 2027 |
| `anthropic/claude-sonnet-5-5` | 2027-09-28 | September 28, 2027 |
| `anthropic/claude-haiku-5-5` | 2027-10-07 | October 7, 2027 |

All four match, so `RETIREMENT_FLOORS_RECORDED` is now 2026-11-04. The models overview page agrees for the three models it lists.

Superseded: see `docs/evidence/pretag-checks-2026-11-05.md` (Notice policies).

Notice policies, as each vendor's deprecation page states them and as `roster.NOTICE_POLICIES` now records them: Anthropic gives at least 60 days of notice before retirement for publicly released models, sent to customers with active deployments; OpenAI gives at least 6 months of notice before retirement for a generally available model, unless safety or compliance concerns require a faster timeline (https://developers.openai.com/api/docs/deprecations). Neither `openai/gpt-6-luna` nor `openai/gpt-6.1-sol` has an announced deprecation.

Superseded for serving: see `docs/evidence/pretag-checks-2026-11-05.md` (Serving guard). Name pins alone do not fix the endpoint.

Serving platform: `roster.SERVING_PLATFORMS` names the Claude API and the OpenAI API, both first-party, and Task metadata carries it under `roster` and `retirement_floors` (tested in `tests/test_roster.py` and `tests/test_task.py`). The pins carry no partner-platform service prefix, and the roster check rejects any model name that differs from its pin.

## Roster check coverage

Each condition in the section 13 row, with the unit tests in `tests/test_roster.py` that cover it:

| Condition | Tests |
|---|---|
| Evaluated model off the roster | `test_evaluated_model_off_the_roster_is_rejected`, `test_floating_alias_is_rejected`, `test_partner_platform_service_prefix_is_rejected`, `test_none_model_is_rejected_by_the_main_roster_check` |
| Judge role off the roster | `test_role_not_on_the_roster_is_rejected`, `test_judge_role_bound_off_the_roster_is_rejected` |
| Unbound role | `test_unbound_required_role_is_rejected`, `test_offline_mode_still_requires_bound_roles` |
| Role rebound away from its pinned ID | `test_grader_bound_away_from_its_pin_is_rejected`, `test_calibrator_bound_away_from_its_pin_is_rejected`, `test_step_two_binds_terra_and_step_three_binds_sol`, `test_each_step_accepts_only_its_own_grader_pin` |
| Grader equal to the evaluated model | `test_grader_equal_to_evaluated_model_is_rejected` (all four roster models, grader steps 1 and 2) |

Through `inspect eval`, `tests/test_task.py` (`test_bad_role_fails_the_run`, `test_unbound_role_fails_the_run`) shows each run ending in error with no sample scored. The template baseline's `none/none` exception is enforced by `baseline.baseline_check` on its own Task and tested in `tests/test_template_baseline.py`; the main check rejects `none/none`.

## Vendored Scout scanners

Superseded for the awareness rubric: see `docs/evidence/pretag-checks-2026-11-05.md` (Scout backport). The cue-word drop described below is retired.

`eval_awareness.py`, `refusal_classifier.py` and `command_not_found.py` are vendored from `meridianlabs-ai/inspect_scout` `examples/docs` at commit 2db653d5fbd1352201b31a74557f612ee78a058c, the 0.5.5 release tag (MIT, Copyright (c) 2025 Meridian Labs). The pristine copies and their git blob SHAs are in `docs/evidence/scout-upstream/`, the working copies in `icfr_agent_evals/scanners/`, and the diff against upstream in `docs/evidence/scout-scanners.diff`. The diff makes three changes: the awareness rubric drops the cue words "test" and "audit"; `eval_awareness` and `refusal_classifier` name `openai/gpt-6-luna` (`roster.SCOUT_SCANNER`) explicitly; each file gets a provenance header. `command_not_found` is otherwise unchanged and uses no model. The scan jobs refuse an evaluation-split log unless `allow_evaluation_split=True` is passed, and the awareness job accepts only worker logs. `tests/test_scanners.py` checks the blob SHAs and regenerates the diff byte for byte.

## Generator version G and splits

`python scripts/tag_values.py --markdown` on the working tree of the commit that adds this file:

- generator_version G-0.1.0-dev.1, hard_generator_version H-0.1.0-dev.1
- generator source SHA-256 86eab813e9de2af3de3bf3e33a505be2a90a95d1617dfe6e75ca03bb1eb2ffbd over the import closure of `icfr_agent_evals.dataset` (7 files, listed by the script)
- evaluation seed 20260607; CI regression split seed 22260607
- CI composition computed from the built split: 146 standard synthetic, 30 edge-type, 24 adversarial-type (matches section 4)
- dataset_sha256: dev f56d06b58b06bbbb0902a552df1a84d5107299bc09c78c6eb514510fce0f7887, ci e78ffb9e603ea1b3d631fae40cfcd072adb4938af74e9a7981738256f76323f6, eval (the 146 synthetic cases) 65737b45d2a2f42c5a46933f7801edac07076cab685585a560f2247447935b2a
- worker prompt v0.0.3-draft, SHA-256 a3430f8f84eb6c353b6e7e86d1bc83dcdd0a8d10a637205c4d943781e07607b7 (not locked)

Both generator labels carry a pre-release suffix. Renaming either changes every split's dataset_sha256, so the final label is set after the pilot, when the generator is frozen, and the script is re-run on the tagged commit.

Superseded: see `docs/evidence/pretag-checks-2026-11-05.md` (Dataset SemVer); section 4 had no minor-version rule on this date.

Planned dataset SemVer for the post-tag build, proposed: 0.1.0. The current label 0.1.0-dev.1 is a pre-release that sorts below it, and a case added after the first post-tag evaluation-split run takes a new minor version (section 4).

## Penrose AccountingBench

See `docs/evidence/penrose-accountingbench.md`: the 2026-09-17 Wayback Machine capture is byte-identical to the live page and its script bundle on 2026-11-04 and still supports the FM-015 and AC-6 citations.
