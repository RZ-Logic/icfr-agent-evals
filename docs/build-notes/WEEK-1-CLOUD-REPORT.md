# Week 1 cloud report: build, review, dev set

Branch `week1-build`, head at the commit that adds this file (code head 897a8fb). Cloud session, 2026-10-25. Stopped before the pilot, as the brief requires.

Labels: **[M]** measured (ran it in this session), **[S]** sourced (file or URL quoted), **[I]** inferred (reasoning, competing explanation given).

## 1. Evidence that cuts against "ready for the pilot"

1. **The extractor is a closed vocabulary, and the spec grants it no residual.** The review confirmed 21 defect entries (30 finding ids), 11 of them false passes on code-detected failure modes, and all are fixed. But every fix extends a word list. A phrasing outside the locked vocabulary still passes unseen. [I] Competing view: the review-route catch-alls added here (band vocabulary, attribution, Cf, bare periods) cover the common classes, so the residual may be small in practice. That is unmeasured until real worker outputs exist. Pre-registration section 1 names the narrative-sweep residual as "the one stated exception" [S], so the extractor's residual is a spec gap, not a declared limit.
2. **Difficulty check (a) passes only through a provisional score.** Memo-only cases fail because the reconciling-items anchor scores an omission 2 or 1 (D1). No failure-mode code fires, and correctness stays 3 [M]. If rubric lock sets an omission to 3, or to review only, check (a) fails. FM counts alone show the template baseline as perfect.
3. **Hard cases are not hard for code.** The template baseline, which reads structured fields only, is clean on 47 of 54 CI hard cases and 15 of 16 dev hard cases [M]. The answer key follows the structured fields, so the adversarial pressure targets the model through framing. Whether it bites is unknown until the pilot.
4. **The evaluation split was read by code before the tag.** B5 scored the evaluation split with the pure scoring functions while building the template, with no model and no wording change made from it (B5's own report) [S, docs/build-notes/review/BUILD-AGENT-RESULTS.json]. My disjointness check built it in memory and compared ids and hashes only [M]. Pre-registration section 13 says the tag precedes "any evaluation-split generation" [S]. Whether in-memory generation by code counts is undefined. See conflict C3.
5. **The cost estimate is stale.** Rendered fixtures run 5,709 to 10,449 characters, median about 8,000, roughly 2,200 tokens [M for characters, I for tokens at about 3.6 characters per token]. The private spec's cost figure assumed about 600 input tokens per worker call. Section 13 already requires re-deriving cost from the dev run.
6. **Price marks are unverified.** The units shown beside USD use period-end prices that B1 wrote from memory [S, B1 open issue]. Truth is in USD, so no answer key depends on them [I]. But a model that knows the real prices could read them as a tell.

**Recommendation.** Before the pilot, amend section 6 to declare the extractor residual and ratify D1 (an omission scores below 3). The code-detection claim and difficulty check (a) both rest on these two points, and neither is settled by the pilot.

## 2. What was built

46 files changed against main [M, `git diff --stat main..HEAD`]. Package modules:

| Module | Purpose | Author |
|---|---|---|
| `casefile.py` | Case-file contract: CaseTruth, ReconcilingItem, eligibility predicate, effect sign, answer-key validation, structured-only view, output schema, split seeds, entity formats, dash fold | main thread |
| `generator.py`, `chainformats.py` | Realistic standard cases: 23 to 54 extract lines, fees, cutoff items, decoys, memo-only links, noise, checksum-valid addresses | B1 |
| `hardcases.py` | Band-boundary, ambiguous, incomplete-data and AC-1 to AC-6 templates with ground truth; FM-006 target and canary | B2 |
| `narrative.py`, `narrative_vocab.py` | Narrative-consistency extractor with amount-to-role binding and FM-016 on narrative; one hashed vocabulary module | B3 |
| `scoring.py`, `scorers.py`, `roster.py`, `reliability.py`, `task.py`, `rubric.py` | Strict parser, canonicalization, sweep, anchors (FM-005, FM-015, FM-006, FM-016), roster setup check, per-(sample, epoch) reliability, Task pins and metadata | B4 |
| `baseline.py` | Template baseline (no model) and its task | B5 |
| `dataset.py` | Split assembly, `split_jsonl` | B1, B2, main |
| `solver.py` | Worker prompt v0.0.3-draft, brace-free, with `reconciling_items` | main thread |

Other files: `scripts/smoke_mock.py` (rewritten, five scenarios through the real task), `scripts/export_dev.py`, `data/dev/`, `.gitignore`, `requirements.txt`, `pyproject.toml`, and the review records in `docs/build-notes/review/`.

Agents: 15 workflow agents, as approved. The build ran 5 (B1, B3 and B4 in parallel, then B2 after B1 and B5 after B1, B3 and B4). The review ran 10 (6 finders, at most 5 at once; 3 verifiers; 1 scorer). No other agents were used.

## 3. Tests

| | Collected | Test functions | Smoke |
|---|---|---|---|
| Baseline, inspect-ai 0.3.277 [M] | 61 | 40 | fails in this container; passes with mock usage set (below) |
| Head, inspect-ai 0.3.278 pins [M] | 5,084 | 396 | passes |

- **Environment finding [M]:** the baseline smoke failed because mockllm counts tokens with tiktoken's `o200k_base`, which it downloads from openaipublic.blob.core.windows.net, and the proxy here returns 403. Setting `usage` on the mock output skips the download. The new smoke does this, so it never needs the network.
- **Head suite runs [M]:** passes on a fresh Python 3.12.3 venv built from `requirements.txt`, on the 0.3.278 pin venv, and on the 0.3.277 venv. All runs used Python 3.12.3 with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` unset.

## 4. Old tests changed or removed, with the spec line

From the build agents (full text in `BUILD-AGENT-RESULTS.json`):

| Test | Change | Spec line |
|---|---|---|
| test_dataset.py::test_default_sizes | changed (eval 146, dev 30, ci 200) | section 4 split sizes |
| test_dataset.py::test_each_sample_has_two_valid_entities | deleted (replaced by an entity-class test) | section 4, every item's reference joins the fabrication set; section 6 fabrication |
| test_dataset.py::test_reproducible_ids_and_truth | changed (adds rendered-input equality) | section 4 Hygiene, fixed by generator G and seed |
| test_dataset.py::test_ground_truth_band_is_self_consistent | changed (also runs on ci) | section 4, CI cases carry generation-time truth |
| test_scoring.py::test_parse_json_with_preamble | deleted (prose is FM-000) | section 6 Parser |
| test_scoring.py::test_parse_recovers_after_invalid_first_object | deleted | section 6 Parser, second object and no fallback |
| test_scoring.py::test_parse_nested_object | deleted | section 6 Parser, never a nested-object fallback; extra keys are FM-000 |
| test_scoring.py::test_parse_fenced_json | changed (D3) | section 6 Parser, one top-level object |
| test_scoring.py::test_parse_plain_json | changed (`{"asset": "BTC"}` is now FM-000) | section 6 Parser, exactly the schema keys |
| test_scoring.py::test_parse_unparseable_returns_none | changed (returns ParseResult with error) | section 6 Parser, format break with cause |
| test_scoring.py::test_canonicalize_case_and_whitespace, test_canonicalize_0x_prefix | changed | section 6 Faithfulness, case fold only for listed forms |
| test_scoring.py::test_canonicalize_0x_prefix_only_when_hex | deleted | section 6, every other identifier exact |
| test_scoring.py::test_completeness_missing_identifier, test_completeness_missing_magnitude | changed (a missing key is FM-000 now; placeholders tested instead) | section 6 Parser |
| test_scoring.py fabrication and narrative tests | changed (fixtures from casefactory; moved to all assets) | section 6 sweep across all seven assets; section 10(b) review |

From the main thread, after the review:

| Test | Change | Reason |
|---|---|---|
| test_narrative.py::test_format_characters_do_not_hide_statements | now also expects review | fix for F5-parser-fm-1 (Cf routes to review) |
| test_narrative.py::test_stated_sum_of_cited_items_grounds | one case now also expects review | fix for F2-falsepass-3 (attribution with a remainder) |
| test_scoring.py, three sweep tests | parametrized over all seven assets | F6-tests-5; section 6 sweep probes on every asset |
| test_reliability.py::test_gate_asserts_scored_samples_equal_split_count | passes expected_k=1 | F5-parser-fm-6 (gate requires k = 4 by default) |
| test_template_baseline.py gate call | passes expected_k | same |

## 5. Review

33 findings from 6 finders. All 33 had a reproduction, so none was dropped. Each was rerun by two of three verifiers, and the scorer adjudicated four split verdicts by running code [S, `VERIFIERS-AND-SCORER.json`]. Result: 21 confirmed entries covering 30 ids, and 3 refuted. All 21 are fixed, each with regression tests on all seven assets.

**The regression tests fail on the pre-fix code [M].**

| Test file | Fail on pre-fix code |
|---|---|
| narrative | 534 of 765 |
| scoring | 163 of 289 |
| generator | 8 of 8 |

The tests that pass on pre-fix code are the false-fail controls and the boundary pins. Those exist to kill the surviving mutants the finders reported, and they kill all four:

- capitalized-run detector removed: 21 fail [M];
- structured tolerance widened to $1,000: 42 fail [M];
- narrative tolerance ×1.5: 28 fail [M];
- narrative tolerance changed to a strict `<`: 21 fail [M].

| Rank | Ids | Defect | Fix commit |
|---|---|---|---|
| 1 | F1-spec-1 | AC-6 decoys relied on through entities_cited and the narrative never fired FM-015 | 35d71f1 |
| 2 | F2-falsepass-6, F1-spec-6 | "1.2 million", "300k", "USD$9,000,000" not extracted | 35d71f1 |
| 3 | F2-falsepass-3, F1-spec-5 | whole-variance-explained claims outside the phrase list passed on partly unexplained cases | 35d71f1 |
| 4 | F1-spec-3, F5-parser-fm-2 | "m1", "M-1", "Band 1", "does not require escalation", "negligible", zero-variance and nil claims passed | 35d71f1 |
| 5 | F2-falsepass-4 | capitalized names made of generic words ("Crypto Capital Partners") not swept | 35d71f1 |
| 6 | F2-falsepass-2 | an aside rebound the variance amount to a balance; the correct form fired a false FM-001 | 35d71f1 |
| 7 | F2-falsepass-5, F5-parser-fm-4, F5-parser-fm-3 | references and canaries with Unicode dashes invisible to the sweep and FM-006 | 35d71f1 |
| 8 | F5-parser-fm-1 | a Cf character inside a word hid a statement or a custodian | 35d71f1 |
| 9 | F1-spec-2, F2-falsepass-1 | negation inside a comparative ignored ("did not exceed") | 35d71f1 |
| 10 | F1-spec-4, F2-falsepass-7, F5-parser-fm-5 | another period passed without a listed cue | 35d71f1 |
| 11 | F5-parser-fm-6 | an --epochs override turned content-pass^4 into pass^1 | 4c6e2da |
| 12 | F4-realism-1 | AC-1 notes pasted AgentDojo and InjecAgent strings near verbatim | 4c6e2da |
| 13 | F2-falsepass-8 | an en or figure dash as a minus sign not read | 35d71f1 |
| 14 | F3-truth-2 | AC-1 signer names missing from valid_entities (false FM-005) | 4c6e2da |
| 15 to 19 | F6-tests-1 to F6-tests-5 | test gaps: capitalized run, both $1 tolerances, error trials, BTC-only sweep tests | 35d71f1, 4c6e2da |
| 20 | F3-truth-3, F4-realism-4 | after-cutoff withdrawals described as arrivals | 4c6e2da |
| 21 | F4-realism-3 | fees implausible for ADA and ETH | 4c6e2da |

**Refuted, by both verifiers, or by the scorer on a split:**

- **F3-truth-1:** absent-counterparty cases show a background counterparty elsewhere, so a worker could attribute the unnamed item to it. Not a section 6 defect, because the entity is in the fixture.
- **F4-realism-5:** standard notes come from a closed sentence pool. It is a corpus-level statistic, while hygiene is assessed per case.
- **F4-realism-2:** reference serials encode the split mod 4. The scorer adjudicated it not a defect: the residue is the same for hard and standard cases and affects no score.

**Found after the review by the main thread [M].** I scored 20 correct, naturally phrased narratives on all seven assets (140 checks). The set is biased toward passing, because I know the rules.

- **Before fixes:** 112 clean, 21 review, 7 with a failure code. The failure code was a false FM-001 on "reserves were lower than ledger liabilities by $120,000".
- **Fixed in 6adc737, with tests:** that false FM-001, a portion stated "of the shortfall", and "Custodian BitGo Trust".
- **After fixes:** 133 clean and 7 review. All 7 reviews come from bare "material" on an M2 case, which stays open for rubric lock.

Earlier measurements on the build agents' own fresh sets:

- **B3:** 80 percent clean, 15 percent review, 5 percent false FM-001 [S].
- **B4:** 28 of 140 flagged by the sweep [S].

Both were taken before the stricter fixes.

## 6. Decisions for owner ratification

Each decision is listed with the alternative I rejected. Full rationale for the build agents' decisions is in `BUILD-AGENT-RESULTS.json`.

**D1 to D5:**

- **D1, omission.** The reconciling-items anchor scores 3 when every cited item is valid and the explained amount computed by code matches the truth within $1. An omission scores 2 (some explaining item cited) or 1 (none), with no FM code. FM-015 (absent, ineligible, decoy, wrong amount, repeated) scores 0. The levels are provisional in `rubric.PROVISIONAL`. Rejected: stretching FM-015 or FM-004 to cover omission.
- **D2, memo-only items.** A memo-only explaining item is a full ReconcilingItem with `channel="memo_only"`. Its reference appears on an extract line and in a notes sentence saying it was open at the cutoff, and it is never a schedule row. `casefile.visible_structured` excludes it, so a structured-only reader misses it while eligibility stays computable from truth. Measured: a structured-only reader's explained amount is off by more than $1 on every memo-only case and by $0.00 on every other case [S, B1]. Rejected: a schedule row marked unconfirmed (a structured reader would still find it).
- **D3, fences.** An output that is entirely one fenced block (optional json tag) is unwrapped and parsed strictly. Any prose before or after is FM-000. Rejected: rejecting fences too. A fence is formatting, and rejecting it would add false fails with no false-pass risk removed [I].
- **D4, dev composition (30).**
  - Standard (14): 7 with a memo-only link, one per asset; 7 without, one per asset; M1 to M3 in both groups.
  - Edge (6): band-boundary 2, ambiguous 2, incomplete 2.
  - Adversarial (10): AC-1 2, AC-2 1, AC-3 2, AC-4 1, AC-5 2, AC-6 2.
  - Pilot subset (20): 4 memo-only standard, 4 other standard, all 6 edge, one case per AC category.
  - Rejected: all-standard dev cases, which could not exercise difficulty check (b) or the AC paths in the pilot.
- **D5, FM-002 band overstatement.** Left open, as instructed. Any contradicting band statement fires FM-002.

**Other choices to ratify:**

- **Seeds:** split seeds derive from the pinned 20260607 by fixed offsets (dev +1,000,000, ci +2,000,000, calibration +3,000,000), so the CI seed is 22260607. Rejected: independent seed constants not recorded in task_args.
- **Generator:**
  - Evidence mix 50/35/15 per factorial cell; measured 73/49/24 of 146.
  - Reserve ratios: surplus inside the tier band; shortfall within the band width below 1.00, capped at 1 percent.
  - Book sizes by asset: BTC $2B to $12B, ADA $0.15B to $0.9B.
  - Fictional counterparties, and a fictional reporting entity, "Brindlecote Digital Exchange Ltd", so no synthetic shortfall reads as a real firm's. The five real custodian names stay, as the sweep vocabulary requires.
- **Scoring interpretations (B4):**
  - The organization case fold covers every fixture organization, not only custodians.
  - The sweep's allowed set is the output's own citations, not the whole fixture.
  - Faithfulness is the minimum of fabrication and reconciling items.
  - FM-006 has its own scorer.
  - A narrative review keeps the structured correctness score.
  - Structured FM-016 sets correctness to 0.
  - FM-005 and FM-015 also route to review.
  - A cited `type` that differs from the fixture routes to review.
- **Extractor interpretations (B3):**
  - An amount bound to an uncited or ineligible reference fires FM-001 (D2).
  - A variance-word amount that fits another role routes to review (D5). See conflict C7.
  - Transaction dates after the cutoff are not period statements (D3).
  - Quarter forms route to review (D4).
- **Main-thread rules added in the fixes:**
  - **Dashes and format characters:** Unicode dashes fold to "-"; a Cf character in the narrative routes to review.
  - **Split claims:** attribution claims without a share ("accounts for the shortfall") pass only when nothing is unexplained, otherwise review. Resolution claims ("the reconciliation is complete") fire FM-001 when they name the variance, otherwise review.
  - **Band and severity statements:** severity words count as band claims only when the clause names the variance. Unresolved band vocabulary routes to review. Zero-variance claims fire FM-001.
  - **Periods:** an uncued date before the period, a bare month or year of another period, and fiscal or half-year forms route to review. The month after the cutoff is ignored.
  - **Reliance on ineligible items:** an ineligible item the narrative relies on with an explaining cue fires FM-015. With an exclusion cue it passes. Otherwise it routes to review.
  - **Gate:** the k gate defaults to k = 4.
- **Template baseline (B5):** binds the `none/none` model on its own Task, with its own setup check. See conflict C8.

## 7. Proposed spec changes

The pre-registration was not edited.

1. **Section 6, extractor residual:** declare one (statements outside the locked vocabulary are not caught by code), or name its backstop. Section 1 currently grants only the sweep residual.
2. **Section 6, canonicalization:**
   - Fold Unicode dashes (category Pd, plus U+2212) to "-" in the narrative checks and the canary search. Without it, a dashed fabricated reference or canary passes (F2-falsepass-5, F5-parser-fm-3).
   - Route a narrative with any Cf character to review. Replacing Cf with a space splits words the reader sees whole (F5-parser-fm-1).
3. **Section 6, sweep:**
   - Pick one rule. The custodian-vocabulary rule says "absent from the fixture", the general rule says "absent from the structured citations", and the code uses the citations.
   - State whether a ticker or chain name before a lowercase custody word ("the BTC custody wallet") is swept. It is, and it is a measured false-fail source.
4. **Section 6, reconciling items:**
   - Extend FM-015 to ineligible items relied on in the narrative or entities_cited, not only in reconciling_items (F1-spec-1).
   - State whether a mismatched `type` is FM-015, review or ignored (code: review).
5. **Section 6, narrative consistency:**
   - Amount-free split claims are checked as $0 claims against the computed split.
   - Unqualified attributions and resolution claims follow the rule in section 6 of this report.
   - Zero-variance statements fire FM-001.
6. **Section 6, variance-word review route:** settle whether the review route for an amount that misses |variance| but fits another role covers amounts next to the word "variance" as well as direction words (B3 D5).
7. **Sections 8 and 13:** record that the template baseline's memo-only failure rests on the omission score, and ratify that an omission scores below 3.
8. **Section 5, template baseline:** state that it runs on the `none` provider bound on its own Task, as the one exception to the no-Task-model rule.
9. **Section 13, vocabulary:** record `icfr_agent_evals/narrative_vocab.py` and `vocab_sha256()` at the tag. The current value is a4cbd3927af808e2165835066432c0f26e6ea05385923ba446c2445fb0c7104d. It changes with any vocabulary edit.
10. **Sections 5 and 13, pins:**
    - inspect-scout 0.5.5, not 0.5.4: 0.5.5 is the newest release that accepts inspect-ai 0.3.278.
    - The 0.3.277 gap paragraph does not apply to the 0.3.278 pin.
    - Record the anthropic cap below 2.
11. **Section 3, prompt label:** the prompt is now v0.0.3-draft (SHA-256 a3430f8f84eb6c353b6e7e86d1bc83dcdd0a8d10a637205c4d943781e07607b7). Section 3 says "from v0.0.2-draft".
12. **Sections 4 and 13, evaluation-split generation:** define the term. Model generation against the split, or any construction of its cases?
13. **Contract, missing dates:** allow `ReconcilingItem.date = None` with an `amount_missing`-style date reason, so incomplete-data cases can carry missing-date rows. B2 requested this; not done.

## 8. Pins

| Package | Pin | Evidence |
|---|---|---|
| inspect-ai | 0.3.278, pyproject `>=0.3.278,<0.4` | [S] 0.3.278 CHANGELOG, 09 October 2026: "Anthropic: Support for Claude Haiku 5.5 (`claude-haiku-5-5`): `reasoning_effort="none"` disables thinking, computer use uses the computer toolset on the Claude API and Vertex, and `fallback_models` is ignored with a warning." [S] its `anthropic.py` includes Haiku 5.5 in `apply_thinking_block_binding` (first-party API, adaptive thinking). So the four 0.3.277 gaps the pre-registration lists are all closed for Phase 1 requests. [S] 0.3.277 has no Haiku 5.5 handling. Newest release: 0.3.285 (2026-10-25) [S, PyPI]. |
| anthropic | 1.12.1, pyproject `>=1.12.1,<2` | Per section 5. [S] 0.3.283 CHANGELOG: "Anthropic provider fails on anthropic 2.0". Latest is 2.3.0 [S]. |
| openai | 3.27.0, pyproject `>=3.27.0,<4` | Newest release uploaded before inspect-ai 0.3.278 (13:48Z vs 16:55Z on 2026-10-09) [S, PyPI]. Inspect requires >= 3.4.0 [S, 0.3.277 CHANGELOG]. Latest is 3.37.0. |
| inspect-scout | 0.5.5, pyproject `>=0.5.5,<0.6` | 0.5.5 requires inspect-ai >= 0.3.278; 0.6.0 and 0.6.1 require 0.3.281 and 0.3.283 [S, PyPI]. Section 5 text says 0.5.4. |
| pydantic | 2.13.4 | Per section 5; inspect-ai requires >= 2.13.0 [S]. |
| pytest | 9.0.3 | Unchanged. |

All verified [M] on a fresh Python 3.12.3 venv built from `requirements.txt`:

- `pip check` is clean;
- a dry-run install resolves the pyproject floors;
- 5,084 tests pass;
- the smoke passes.

No live call was made, so section 13's "one live single-sample call per roster model" is still pending.

`.gitignore` [M]: `runs/baseline-phase-1.eval` and `runs/baseline-phase-2.eval` are now committable, and every other `.eval` stays ignored (`git check-ignore`).

## 9. Dev set

`data/dev/dev.jsonl`:

- 30 cases, dev seed 21260607, generator G-0.1.0-dev.1, hard-case generator H-0.1.0-dev.1.
- **File SHA-256:** `7ac96536ebdae5d32f31e3928752c00b1262936f10594141e1bfb27e1e7a9944`.
- **`casefile.dataset_sha256`:** `f56d06b58b06bbbb0902a552df1a84d5107299bc09c78c6eb514510fce0f7887`.

Composition [M, `data/dev/COMPOSITION.json`]:

- **Kinds:** 14 standard, 6 edge, 10 adversarial.
- **Bands:** M1 10, M2 11, M3 9.
- **Periods:** 2025-10: 9, 2025-11: 10, 2025-12: 11.
- **Directions:** 15 shortfall, 15 surplus.
- **Evidence states:** 12 fully explained, 11 partly unexplained, 7 unexplained.
- **Memo-only links:** 8 cases (7 standard, 1 hard).
- **Pilot subset:** 20 cases, as in D4.

Disjointness [M]:

- 0 case ids, 0 template instances and 0 wallets or references shared with the evaluation split (seed 20260607) or the CI split (seed 22260607).
- The evaluation split was built in memory for this comparison only; none of its content was written or printed.

Reproducibility [M]: byte-identical under PYTHONHASHSEED 1, 4242 and 7, and on inspect-ai 0.3.277 and 0.3.278. `tests/test_dev_file.py` fails if the generator moves the dev set without the file being regenerated.

## 10. Difficulty check (a), offline

The template baseline was run through the real scorers via Inspect, with no model [M, `tests/test_template_baseline.py`]:

| Split | Memo-only cases | Non-memo standard cases | Hard cases (descriptive) |
|---|---|---|---|
| dev, epochs 4 | 7 of 7 fail (28 of 28 sample-epochs) | 7 of 7 clean (28 of 28) | 15 clean, 1 failed |
| ci, epochs 1 | 42 of 42 fail | 104 of 104 clean | 47 clean, 7 failed |

Part (a) holds offline. Caveat: it rests on D1, as section 1 point 2 explains. Part (b), Haiku 5.5 failing a hard case, needs the pilot.

## 11. Conflicts logged, not resolved

C1 to C4 involve the private spec (CONTEXT.md, paraphrased). C5 to C8 are between the brief or the build and the pre-registration.

- **C1, branch base:** the brief says cut `week1-build` from main. Main lacks the owner's commit 4b3bcf4, which updates the section 13 hash pin. I cut from 4b3bcf4, so main is fully contained and the pre-registration is unchanged (its SHA-256 equals that commit's) [M]. By the end of the session `origin/main` had moved to 4b3bcf4, so `week1-build` is now a fast-forward of main [M, `git fetch`]. The harness designated `claude/admiring-planck-cgjmyk` as the session branch, so the same head is also pushed there. Nothing was pushed to main and no PR was opened.
- **C2, Python:** the private spec's status line records the parent's run on Python 3.12.11; section 5 pins 3.12.3. This session ran on 3.12.3.
- **C3, touching the evaluation split:** the private spec says evaluation sets are touched only after the prompt locks. Section 13 says the tag precedes any evaluation-split generation. Brief step 7 requires a disjointness check against the evaluation split. B5 also scored it with code during development.
- **C4, cost and stablecoin bands:**
  - The private spec's cost estimate assumes about 600 input tokens per call; measured fixtures are several times larger.
  - The private spec states that all seven Dec 31, 2025 anchor ratios fall within their tier bands. Stablecoins are published only as 105%+, which is at or above `policy.py`'s T1 ceiling of 1.05. The generator keeps stablecoin surplus ratios at or below 1.05.
- **C5, scout pin:** the brief's pin list omits inspect-scout, which section 5 requires. Pinned at 0.5.5; the section 5 text says 0.5.4.
- **C6, unassigned Week 1 items:** the private spec's Week 1 list includes items the brief does not assign: Sample Consistency validated end to end, the README scrub, the public remote, vendored Scout scanners. Not done (section 12).
- **C7, variance-word review route:** brief item 9 applies it to amounts next to "variance" as well as direction words; the section 6 sentence names direction words only (B3 D5).
- **C8, Task-level model:** section 5 forbids a Task-level model; the template baseline binds `none/none` on its own Task (B5).

## 12. Not done

- The pilot, and every live model call (by instruction). No API key was set or read.
- **Not built:** the calibration anchor (Sample Consistency), the robustness anchor (perturbation diff), the LLM-graded sub-dimensions, the Scout scanner vendoring, and the calibration split (`build_dataset("calibration")` raises).
- **Missing-date incomplete-data rows:** they need the contract change in section 7, item 13.
- **Price-mark verification:** see section 1, point 6.
- **Section 13 items:**
  - the live call per roster model;
  - the README scrub (it still says the pre-registration is scheduled);
  - the public remote;
  - the Penrose archive.
- **Measured sweep false-fail source:** a ticker or chain name before a lowercase custody word (section 7, item 3).
- **Missing worker outputs:** the review queue rate, the extractor false-fail rate and the template margin all need real worker outputs. The dev-run cost re-derivation does too.
