# Pilot runbook: Phase 1, dev split

This runbook turns the §13 pilot row of [`PRE-REGISTRATION-PHASE-1.md`](../../PRE-REGISTRATION-PHASE-1.md) into commands, and then lays out the sequence from the pilot to the `prereg-phase1-v1.0` tag. Every command runs on the dev split; nothing here generates on, scores or reads an evaluation-split case. The pilot's results are dev results and are never reported as findings (§4).

Bracketed ids such as [F1-03] name the finding of the pre-pilot review of d4e86ec that a rule answers. That review is not part of this repository; each rule stands on its own here.

## Placeholders

| Placeholder | Meaning |
|---|---|
| `<SHA>` | The pilot commit: the commit the fix round's re-verify passes. Checked out, clean tree. |
| `<PILOT_LOGS>` | A directory outside the repository that holds every pilot log while the legs run (D5). |
| `<SCRATCH>` | A scratch directory outside the repository for the mock dress rehearsal. |
| `<date>` | The pilot date, `YYYY-MM-DD`. |
| `<log>` | The one `.eval` file a leg wrote into its directory. |
| `python` | The pilot venv's Python 3.14.10 interpreter (§5). |

Commands are single lines that run in Git Bash or PowerShell from the repository root. Steps marked **keyed** run in the author's keyed shell with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set; every other step is keyless. No step prints a key.

**Why logs stay outside the repository while the legs run.** Inspect sets the log header's `revision.dirty` from `git status --porcelain`, which lists untracked files that `.gitignore` does not exclude. `runs/pilot-*/` is re-allowed so the logs can be committed as evidence, so a log written there during the Sonnet leg would make the Haiku leg's header read dirty and fail the pre-flight's clean-tree check. Write every leg to `<PILOT_LOGS>`; copy the logs into `runs/pilot-<date>/` once the last leg and the pilot report are done (step 12). Do not name a leg directory `logs`: `.gitignore` excludes any `logs/` directory, `runs/pilot-*/logs/` included.

## Rules for every command

- No `--epochs`, `--limit`, `--sample-id`, `--reasoning-effort`, `--temperature`, `--top-p`, `--top-k`, `--seed`, or `--max-tokens` (unless D6 pins a value, and then exactly that value). [F1-02]
- No `--model-role` beyond the two pins: `grader=openai/gpt-6-luna` and `calibrator=anthropic/claude-opus-5-5`.
- Never `--no-fail-on-error` and never a fractional `--fail-on-error`. A fractional threshold leaves errored epochs inside a status-success log as failed trials, which pass^4 and reading (i) of difficulty check (b) both count. [F1-03]
- Always `-T split=dev -T subset=pilot`. The task and the template baseline refuse `split=eval` unless `-T allow_evaluation_split=true` is passed, and only the tagged evaluation run passes it. The refusal is raised while Inspect loads the task, before any case is built: a command that reaches `split=eval` without the unlock exits non-zero with the `ValueError` on stderr and writes no log. Inspect parses `-T` values as YAML, so only `true`, `yes` or `on` unlock; `1`, a YAML-quoted string or a mistyped key stays locked. Inspect drops an unknown `-T` key, so a typo such as `-T splt=dev` falls back to the default split and is refused at load the same way, and a mistyped subset runs the whole dev split, which `pilot_report.py` refuses as not the pilot subset. [F1-01, F5-01]
- Both paid legs carry `--max-connections 4 --max-retries 6 --timeout 1800 --retry-on-error 2`. Inspect's defaults retry a 429 without end; these bound the retries, the request time, the sample retries and the in-flight loss when `fail_on_error` cancels a run. [F1-03, F4-05]
- Every real run passes an explicit `--log-dir` under `<PILOT_LOGS>`, never the default `./logs`. [F1-05]
- No pooled re-run in the regression comparison: one baseline log, one candidate log. If candidates are ever pooled (`--candidate` repeated), they must be re-runs of one configuration, and a correct run and a degraded run of the same model, task, revision and prompt are never passed together as candidates. `compare_runs.py` refuses (exit 3) pooled candidates that differ in model, task, revision, dirty flag or `worker_prompt_sha256`, and a candidate whose task or `worker_prompt_sha256` differs from the baseline's, but it cannot tell a correct run from a degraded one when the headers are identical. [F2-04, F2-08]
- A log whose status is not `success` is never a pilot log. `roster.check_run_config` and the readers refuse it; recover it with `eval-retry` (step 5) or rerun the leg.
- Keep the generator labels `G-0.1.0-dev.1` and `H-0.1.0-dev.1` through the tag. `scripts/tag_values.py` warns on both, and those two warnings are expected. Dropping the labels moves the dev split's `dataset_sha256` (f56d06b5 to 4312345a), which strands the pilot logs and the judge-development set on a hash the tagged commit no longer produces. [F5-09, D8]

## Decisions

Each decision is set before the step that needs it and written into the pilot record. The default is what the fix round built against; every one can be flipped.

| ID | Decision | Default in force | Before step | Author's choice |
|---|---|---|---|---|
| D1 | Reading of record for difficulty check (b): (i) literal; (ii) excluding format breaks, errored or missing epochs and narrative-sweep provisional zeros; (iii) settled anchor scores and non-format FM codes only | `pilot_report.py` prints all three; the reading of record is chosen before the Haiku leg | 6 | |
| D2 | What "caught" means for the deliberate regression, and its form: (i) the real Sonnet baseline against the mock-degraded candidate from `rehearse_regression.py`, $0; (ii) a second real Sonnet run on a weakened prompt | Caught: `compare_runs.py` exits 1 on (pilot baseline, regressed candidate) with the rejecting test named (text in step 8). Form: not defaulted; (ii) needs a weakened-prompt code path that is not built | 8 |Form (i), keyless: compare_runs.py exit 1 on (pilot Sonnet log, rehearse_regression.py degraded candidate) with the rejecting test named. Decided 2026-11-06. |
| D3 | Evaluation-split unlock | The task argument `allow_evaluation_split`, default `False` | 0 | |
| D4 | Timing of the extractor review-route fixes | Land before the pilot | 0 | |
| D5 | Log and record paths | Logs in `<PILOT_LOGS>` outside the repository during the legs, then copied to `runs/pilot-<date>/` (re-allowed by `.gitignore`); record at `docs/evidence/pilot-<date>.md`, review CSV at `docs/evidence/pilot-<date>-review.csv`, judge-development set under `data/dev/judge-dev/` | 3 | |
| D6 | `max_tokens` | No pin in code; the cost table shows the 48,000 default and a 16,000 pin | 5 |No pin; the 48,000 default is recorded; pilot_report.py's generation-stop and output-token lines decide whether it stands for the full run. Decided 2026-11-06. |
| D7 | `--batch` or interactive | Interactive; `check_run_config` allows batch (step 5 has the option) | 5 |Interactive. Decided 2026-11-06. |
| D8 | Generator labels | Keep `-dev.1` through the tag | 0 | |
| D9 | A hedged or suffix-rounded figure ("approximately USD 34.3k", "USD 1.1 million") that agrees with its target within its stated precision | Routes to review: never a pass, never FM-001 (needs a §6 sentence) | 0 | |
| D10 | Elicitation before or after the pilot | None; a pilot on the draft prompt leaves (b), the false-fail rate and the judge-development outputs on the draft prompt hash | 5 |Pilot first on v0.0.3-draft; elicitation after the pilot; the judge-development set is regenerated under the locked prompt if its hash differs (sequence step 8). Decided 2026-11-06. |
| D11 | Scope of "every transcript read" | None; step 11 states the proposed scope | 11 | |
| D12 | Judge-development set: pilot outputs, or regenerated under the locked prompt | None; the proposed rule is in the sequence, step 8 | sequence 8 | |
| D13 | 0 to 3 anchors for the two LLM-graded sub-dimensions written before the judge-development grading | None; proposed as sequence step 7 | sequence 7 | |
| D14 | Row 227 clause recording the review-routing and false-fail rate per run | None; proposed text awaits approval | sequence 12 | |
| D15 | Smoke scope ("one live single-sample call per roster model") | `smoke_live.py` accepts any model list; its default is every distinct pinned model | sequence 11 | |
| D16 | Row 228 calibrator cost | None; inferred from the pilot's per-call tokens at Opus 5.5 prices unless a paid dev run measures it | sequence 3 | |
| D17 | An off-contract structured value whose canonical content equals the truth ("November 2025" for `2025-11`, "M2 (reportable)", a signed variance) | Routes to review | 0 | |
| D18 | An exclusion sentence that states the decoy's amount | Keeps firing FM-001 by the §6 rule, disclosed | 0 | |

## Steps

**0. Fix round and re-verify (keyless).** The fix branches merge, the integration step wires the cross-file calls, and the re-verify runs the full suite on 3.14.10 and 3.12.16, every confirmed reproduction, the held-out probe set and step 4's mock command. The commit it passes is `<SHA>`.

**1. Decisions.** Fill the author's-choice column above and copy it into the pilot record. D5 before step 3, D1 before step 6, D2 before step 8.

**2. Console, by hand (author).** [F4-01, F4-05]
- (a) Credit balance and the expiry date of each credit grant, recorded in the pilot record.
- (b) Billing: the tier's monthly cap minus month-to-date spend is at least $80 at list price for the three runs ($40 with `--batch`; $41 for the two runs of form (i)).
- (c) Optional: a user spend limit at that headroom. Reaching it returns HTTP 400, which Inspect does not retry, so the sample errors and the eval stops; a tier spend-cap 429 is retried instead.
- (d) Rate limits page: the tier name and the Sonnet 5.5 and Haiku 5.5 output-tokens-per-minute limits, recorded in the pilot record. In the Evaluation tier, adaptive concurrency never drops below 10 streams; `--max-connections 4` caps it.

**3. Pre-flight (keyed, `<SHA>` checked out, repository root).**

```
python scripts/preflight.py --model anthropic/claude-sonnet-5-5 --pilot-commit <SHA> --log-dir <PILOT_LOGS>
```

Create `<PILOT_LOGS>` first. Every line must read PASS and the exit code must be 0; it is 1 when any check fails and 2 on a usage error. It checks that HEAD equals `<SHA>` and the tree is clean, Python 3.14.10, the override variables unset (`INSPECT_EVAL_EXTRA_HEADERS` among them) and both keys set (yes or no, never a value), that no `.env` file Inspect would load sets an override, that the model and both roles construct and pass the roster and serving checks with their endpoints printed, and that `<PILOT_LOGS>` exists outside the repository. A folder inside the checkout, `runs/pilot-<date>/` included, fails the log-dir check, for the dirty-tree reason above. `--model` takes only the worker or one of the three baselines; a judge model such as `openai/gpt-6-luna` is a usage error. It constructs the models and never calls one. In a keyless shell it prints FAIL on keys and construct and exits 1. Run it before every keyed leg: again with `--model anthropic/claude-haiku-5-5` before step 6, before the form (ii) leg of step 8, and before the live smoke (sequence step 11).

**4. Mock dress rehearsal (keyless, `<SHA>`).**

```
python -m inspect_ai eval icfr_agent_evals/task.py@variance_commentary -T split=dev -T subset=pilot -T epochs=4 --model mockllm/model --model-role grader=mockllm/model --model-role calibrator=mockllm/model --max-connections 4 --max-retries 6 --timeout 1800 --retry-on-error 2 --log-dir <SCRATCH>/mock --display none
```

Measured on the merged fix round with these exact flags: exit 0 in about 6 s, empty stderr, status success, task_args {split dev, seed 20260607, epochs 4, grader_step 1, subset pilot, allow_evaluation_split false}, limit and sample_id None, fail_on_error True, retry_on_error 2, plan.config {max_retries 6, timeout 1800, max_connections 4, reasoning_effort high} and effort high on all 80 model events, 80 sample-epochs with 0 errors; `roster.check_run_config` with `expected_samples=20` passes (80 scored sample-epochs, 80 evaluated-model events, effort high), `reliability.report_from_log` with `assert_gate(20, 4)` passes, `regression.records_from_log` returns 80, and `compare_runs.py` with this log as baseline against the rehearsal's degraded candidate prints `BLOCKED (automatic)` and exits 1. A mock run skips the roster pins and the serving guard (every model is offline), so those rest on the unit tests and step 3.

**5. Sonnet leg (keyed; after D5, D6, D7, D10).**

```
python -m inspect_ai eval icfr_agent_evals/task.py@variance_commentary -T split=dev -T subset=pilot -T epochs=4 --model anthropic/claude-sonnet-5-5 --model-role grader=openai/gpt-6-luna --model-role calibrator=anthropic/claude-opus-5-5 --max-connections 4 --max-retries 6 --timeout 1800 --retry-on-error 2 --log-dir <PILOT_LOGS>/sonnet
```

- 80 worker calls. The OpenAI key constructs the grader role only; no grader or calibrator call is made, since the LLM-graded sub-dimensions are not wired.
- Runtime [inferred]: 20 waves at 4 connections; about 30 to 45 minutes at 8,000 output tokens and 60 to 90 tokens per second, up to about 4 hours at the 48,000 ceiling.
- If D6 pins `max_tokens`, add `--max-tokens <pin>` to this and step 6, and pass the same pin as `--max-tokens-pin <pin>` to `pilot_report.py` and `compare_runs.py` (their readers refuse any other `max_tokens`). Thinking counts toward `max_tokens`, so a low pin turns a cut-off answer into FM-000. `pilot_report.py` prints per leg how many generations stopped on `max_tokens` (and how many of the format breaks they are) with the maximum, median and total output tokens, which is the measurement D6 is confirmed or revisited on before the tag. Without a pin, Inspect records the 48,000 it sent in every model event of the evaluated model while `plan.config` leaves `max_tokens` unset; the readers accept exactly that value and report it as `max_tokens_effective` (protocol fix from the 2026-11-06 Sonnet leg, which the readers refused before it).
- A trace line naming `enforced_spend_limit_reached` means a spend cap was hit: stop with Ctrl+C. [F4-01]
- If the run ends with status error, recover the same log, never read it as a pilot log:

```
python -m inspect_ai eval-retry <PILOT_LOGS>/sonnet/<log>.eval --max-connections 4 --max-retries 6 --timeout 1800 --retry-on-error 2 --log-dir <PILOT_LOGS>/sonnet
```

- **Batch option (D7).** [F4-04] Adding `--batch` halves the price ($39.93 worst case for three runs) and keeps epochs, per-sample logs and the first-party endpoint. It turns off adaptive concurrency and streaming, adds a 15 s send delay and poll tick on top of an unmeasured vendor queue, and adds a run-config field that `check_run_config` allows. Whether `--timeout 1800` leaves a queued batch request alone was not verified: with `--batch`, drop `--timeout` or verify it on a single dev sample first.

**6. Haiku leg (keyed; after D1).** Step 3 with `--model anthropic/claude-haiku-5-5`, then step 5's command with `--model anthropic/claude-haiku-5-5 --log-dir <PILOT_LOGS>/haiku`. Same flags, same recovery.

**7. Template baseline (keyless).**

```
python -m inspect_ai eval icfr_agent_evals/baseline.py@variance_commentary_template_baseline -T split=dev -T subset=pilot --log-dir <PILOT_LOGS>/template --display none
```

Bind no `--model-role` and no `--model` here: the template baseline calls no model, and a run with judge roles bound ends in status error from the roster check. Measured on the merged fix round: exit 0, empty stderr, status success, task_args {split dev, seed 20260607, epochs 4, subset pilot, allow_evaluation_split false}, 80 rows. On mock stand-ins: content-pass^4 15/20, difficulty check (a) passing (4 of 4 memo-only standard cases failing through the reconciling-items omission score only, 4 of 4 other standard cases clean). Epochs stay at 4 and every epoch is identical by construction.

**8. Deliberate regression (after D2).**

Form (i), keyless:

```
python scripts/rehearse_regression.py <PILOT_LOGS>/rehearsal
python scripts/compare_runs.py --baseline <PILOT_LOGS>/sonnet/<log>.eval --candidate <PILOT_LOGS>/rehearsal/regression/<log>.eval
```

Form (ii), keyed: step 3's pre-flight, then a second Sonnet leg on a weakened-prompt code path (not built at d4e86ec) into `<PILOT_LOGS>/weakened`, then the same `compare_runs.py` call with that log as candidate.

`rehearse_regression.py` takes its log folder as an optional positional argument (default `icfr-agent-evals-logs/rehearsal` under the system temp folder); `-h` prints usage and creates nothing. Measured on the merged fix round: the rehearsal prints `REHEARSAL RESULT: PASS`, and `compare_runs.py` with step 4's mock log as baseline and the rehearsal's regression log as candidate prints one header line per run, then `RESULT: BLOCKED (automatic)`, and exits 1.

**Caught (D2 default).** [F2-02, F2-03, F2-04]
- Caught means `compare_runs.py` exits 1 on the pair (pilot Sonnet baseline, regressed candidate), with the rejecting test named in its output. Exit 0 is not caught, and exit 2 is not caught; `pilot_report.py` prints those two as `MISSED (no automatic block)` and prints `caught` only on exit 1.
- Size the regression from the baseline's own per-case results. The exact one-sided McNemar test on content-pass^4 under Holm with m = 6 blocks only from 7 one-way discordant cases with no reversal, 10 with one; a regression that turns fewer passing cases can be caught only by the paired t-tests, at the floor `pilot_report.py` prints.
- A baseline passing pass^4 on fewer than 7 of 20 cases is recorded as "only the t-tests can catch it".
- A regression confined to reconciling-item omission fires no FM code and no review route, so at n = 20 it can pass the gates with no sign-off case. It is a declared blind spot of the regression at pilot size.
- Exit 2 on a no-change pair or a pooled pair is a measurement of review volume, not a gate failure. At a 2 percent per-epoch route rate, a no-change pair of real runs exits 2 with probability about 0.78 on 20 cases [inferred]; pooling identical runs makes exit 2 certain whenever a code fires.

**9. Pilot report (keyless).**

```
python scripts/pilot_report.py --sonnet <PILOT_LOGS>/sonnet/<log>.eval --haiku <PILOT_LOGS>/haiku/<log>.eval --template <PILOT_LOGS>/template/<log>.eval --reading-b <record> --regression-baseline <PILOT_LOGS>/sonnet/<log>.eval --regression-candidate <PILOT_LOGS>/rehearsal/regression/<log>.eval --json runs/pilot-<date>/report.json --review-csv docs/evidence/pilot-<date>-review.csv
```

`<record>` is the reading of record set by D1: `i`, `ii` or `iii` (`all` marks none); all three readings always print. Exit 0 on success, 3 on refusal (a missing file, a log that is not the dev pilot subset, mismatched dataset hashes, a log whose status is not success, swapped roles or half a regression pair). Like `compare_runs.py`, `-h` and an argument error exit 3 and run no report. A smoke log (sequence step 11) is never passed to it. Paste its output verbatim into the pilot record. The review CSV gets one row per distinct (run, sample, epoch, route key) with `verdict`, `reviewer` and `date` blank for step 11. The template log carries no Sample Consistency summary: it is deterministic, with agreement 1 by construction.

**10. Scout scan of the Sonnet pilot log (keyed, OpenAI key).** [F5-07] Records the GPT-6 Luna scanner usage for row 228. Run the scan jobs on the Sonnet log only, with results outside the repository; this command was not run by the review:

```
scout scan icfr_agent_evals/scanners/jobs.py@worker_awareness_scan -S logs=<PILOT_LOGS>/sonnet --scans <PILOT_LOGS>/scans
scout scan icfr_agent_evals/scanners/jobs.py@all_logs_scan -S logs=<PILOT_LOGS>/sonnet --scans <PILOT_LOGS>/scans
```

Record the scanner model's input and output tokens and calls in the pilot record. Scanner counts stay exploratory.

**11. Transcript reading (author).** [F4-03, F5-10, D11] Proposed scope:
- the 20 case files, once (about 2 hours);
- all 80 Sonnet transcripts with their thinking summaries (about 5.3 hours);
- every Haiku hard-case epoch scoring below 3 on an anchor or firing an FM code, since (b) rests on Haiku failing a hard case and a reader must confirm the failure is not a format break (up to 80 epochs, about 4 hours);
- the 20 distinct template outputs (its 4 epochs are identical);
- a verdict on every row of the review CSV: `correct_route`, `false_fail` or `confirmed_failure`, with reviewer and date (about 5 minutes each).

Expect 9 to 18 hours over at least two sittings [inferred]. Use the idiom checklist for statements the locked vocabulary does not catch (approximate amounts in words, coverage and reserve ratios, "fully backed" or "in the red" against the direction, nil-remainder idioms, a period named in prose). The row stays pending until this is done; record the hours, the transcript count per run and the route count. The false-fail rate per run and route class is `false_fail` rows over all rows.

**12. Record (author).** Fill [`pilot-TEMPLATE.md`](pilot-TEMPLATE.md) as `docs/evidence/pilot-<date>.md`, with the review CSV beside it. Copy the logs into `runs/pilot-<date>/sonnet`, `haiku`, `template` and `rehearsal` and record each file's SHA-256. If difficulty check (b) fails, ceiling scores on all three runs included, harden the case generator and repeat the pilot before tagging, and report each pilot round.

## Cost at list price

Inputs at 3.0 characters per token on the measured 10,445.6 characters per prompt, about 3,482 input tokens per generation. Prices: Sonnet 5.5 $2 / $10, Haiku 5.5 $0.10 / $0.50, Opus 5.5 $4 / $20 per million tokens in / out; batch 50 percent off; thinking billed as output in full, even when summarized, and counted toward `max_tokens`.

| Run (80 generations) | Worst, 48,000 out (no pin) | 16,000 pin | Expected, 8,000 out |
|---|---|---|---|
| Sonnet 5.5 pilot | $38.96 | $13.36 | $6.96 |
| Haiku 5.5 pilot | $1.95 | $0.67 | $0.35 |
| Sonnet 5.5 weakened-prompt run (form ii) | $38.96 | $13.36 | $6.96 |
| Grader, calibrator, template, mock regression | $0 | $0 | $0 |
| Total, one attempt per generation | $79.86 | $27.38 | $14.26 |
| Total with `--batch` | $39.93 | $13.69 | $7.13 |
| Added: a (b)-fail rerun (Sonnet, Haiku, template) | $40.91 | $14.03 | $7.31 |
| Added: one cancelled Sonnet run's in-flight loss | $1.95 at 4 streams | | $0.35 at 4 streams |

The worst case is a one-attempt figure, not a bound: `--retry-on-error 2` can bill an errored sample-epoch up to three times (Sonnet $0.49 each at the ceiling), and streamed partial generations cancelled by `fail_on_error` are billed. 8,000 output tokens is a guess for high-effort adaptive thinking on a single-turn case of 3,000 to 3,500 tokens plus a JSON answer of about 700.

## Sequence to the tag

| # | Step | §13 item | Artifact | Who |
|---|---|---|---|---|
| 1 | Elicitation on the dev split (D10 sets whether it precedes the pilot) | prompt lock | `solver.py` edits | author, keyed |
| 2 | Pilot: Sonnet, Haiku, template, deliberate regression (steps 1 to 12 above) | pilot | logs, pilot record | author, keyed |
| 3 | Scout scan of the Sonnet pilot log (step 10) | cost cap | pilot record | author, OpenAI key |
| 4 | Transcript reading and review verdicts (step 11) | pilot | review CSV | author |
| 5 | Protocol fixes from the pilot | pilot | fix commit, listed in the pilot record | keyless |
| 6 | Prompt lock: version and hash | prompt lock | `solver.py`, Task metadata | keyless, author ratifies |
| 7 | 0 to 3 anchors for `faithfulness.narrative_grounding` and `completeness.narrative_coverage`, committed with their SHA-256 before any judge-development grading (D13) | rubric | rubric text | author |
| 8 | Judge-development set (D12): if the locked prompt's `worker_prompt_sha256` equals the pilot logs', use 20 pilot outputs; otherwise regenerate the 20 under the locked prompt at one epoch (about 20 Sonnet calls) before grading. Grades are committed before any grader iteration | judge-development set | `data/dev/judge-dev/` | author grades |
| 9 | Full suite on 3.14.10 and 3.12.16 | runtime | evidence file | keyless |
| 10 | Pre-tag spec-against-docs review; the CONTEXT.md SHA-256 re-checked after any edit | counts match | evidence file | keyless |
| 11 | Live smoke on the commit to be tagged: `python scripts/preflight.py --model anthropic/claude-sonnet-5-5 --pilot-commit <commit to be tagged> --log-dir <PILOT_LOGS>/smoke` with every line PASS, then `python scripts/smoke_live.py --sample-id <dev case id> --log-dir <PILOT_LOGS>/smoke` (D15). Before each model it prints the request-header sources in force, which must read `none`; per model it prints the resolved model string, the checked endpoint, the usage and the log path, and it exits 0 only when every model answered. Smoke logs set `sample_id` and epochs 1, so `roster.check_run_config` refuses them: never pass one to `pilot_report.py` or `compare_runs.py` | roster, smoke | the printed lines in `docs/evidence/smoke-<date>.md` | author, both keys |
| 12 | §13 fill from `scripts/tag_values.py` and the smoke, row 227's clause included (D14) | all | pre-registration | keyless, author ratifies |
| 13 | Public remote | public remote | remote URL | author |
| 14 | Tag `prereg-phase1-v1.0` | tag | tag | author |
| 15 | Tag pushed to the public remote, before any evaluation-split generation | tag | remote tag | author |

The smoke follows the last code change, and the §13 fill follows the smoke it records. After the tag, an evaluation-split command adds `-T split=eval -T allow_evaluation_split=true`.

## Records

- [`pilot-TEMPLATE.md`](pilot-TEMPLATE.md): the pilot record, copied to `docs/evidence/pilot-<date>.md`.
- [`pilot-TEMPLATE-review.csv`](pilot-TEMPLATE-review.csv): the review CSV header; `pilot_report.py --review-csv` writes the rows.
- [`data/dev/judge-dev/README.md`](../../data/dev/judge-dev/README.md): the judge-development record format.
