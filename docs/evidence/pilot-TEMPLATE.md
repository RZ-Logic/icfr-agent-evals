# Pilot record: <date>

Copy this file to `docs/evidence/pilot-<date>.md` and fill every field. The steps are in [`PILOT-RUNBOOK.md`](PILOT-RUNBOOK.md). Dev split only; these results are never reported as findings (§4 of the pre-registration).

## Run identity

- Pilot commit (`<SHA>`):
- `scripts/preflight.py` exit code, Sonnet model:
- `scripts/preflight.py` exit code, Haiku model:
- Python (`platform.python_version()`):
- Console at launch: credit balance and grant expiry dates; tier; monthly cap minus month-to-date spend; user spend limit (if set); Sonnet 5.5 and Haiku 5.5 output-tokens-per-minute limits:
- Batch or interactive (D7):
- `max_tokens` pin, or "none, 48,000 default" (D6):

## Logs

Logs were written outside the repository while the legs ran and copied afterwards under `runs/pilot-<date>/`.

| Run | File name under `runs/pilot-<date>/` | SHA-256 | Status | worker_prompt_sha256 | dataset_sha256 |
|---|---|---|---|---|---|
| Sonnet 5.5 | | | | | |
| Haiku 5.5 | | | | | |
| Template baseline | | | | | |
| Regression candidate | | | | | |
| Mock dress rehearsal (step 4, not committed) | | | | | |

The Haiku log's `worker_prompt_sha256` equals the Sonnet log's (a weakened prompt does not count toward check (b)): yes / no.

## Pilot report output

`scripts/pilot_report.py` command line, then its output pasted verbatim:

```
```

## Pilot row clause by clause

| Clause | Artifact that satisfies it | Met |
|---|---|---|
| 20 dev-split cases | Pilot report: sample count per run | |
| Sonnet 5.5 | Sonnet log header, model and run config | |
| epochs = 4 | Pilot report: epochs per run; `check_run_config` passed | |
| every transcript read | Transcript reading table below | |
| the template baseline beside it | Template log; pilot report margin per anchor | |
| one deliberate regression caught end to end on the dev split | Regression block below; `compare_runs.py` exit code and rejecting test | |
| the protocol fixes it produced listed | Protocol fixes below | |
| difficulty check (a): the template baseline failing memo-only cases through the reconciling-items omission score | Pilot report: check (a) | |
| difficulty check (b): Haiku 5.5 at its roster configuration failing at least one hard case on a deterministic anchor | Pilot report: check (b) under the reading of record, with the failing hard cases and epochs | |
| a deliberately weakened prompt not counting | Prompt hash equality above | |
| review-routing and false-fail rate per run (proposed clause, D14) | Review CSV and the rates below | |

## Difficulty check (b)

- Reading of record (D1), chosen before the Haiku leg on (date):
- Result under (i):
- Result under (ii):
- Result under (iii):
- Failing hard cases and epochs under the reading of record:

## Deliberate regression

- Caught definition (D2): `compare_runs.py` exits 1 on (pilot Sonnet baseline, regressed candidate) with the rejecting test named; exit 0 and exit 2 are not caught.
- Form (D2): (i) mock-degraded candidate from `rehearse_regression.py` / (ii) weakened-prompt Sonnet run
- Baseline content-pass^4 count (against the McNemar floor of 7 one-way discordant cases, 10 with one reversal):
- `compare_runs.py` command, exit code and rejecting tests:
- Caught: yes / no

## Transcript reading

| Run | Transcripts or epochs read | Scope (D11) |
|---|---|---|
| Sonnet 5.5 | | |
| Haiku 5.5 | | |
| Template baseline | | |

- Case files read once: yes / no
- Author reading hours, in total and per sitting:
- Review routes given a verdict (rows in `pilot-<date>-review.csv`):
- False-fail rate per run and route class (`false_fail` rows over all rows):
- Statements outside the locked vocabulary noted while reading:

## Scout scan (row 228)

- Scan jobs run, scanner model and its calls, input and output tokens:
- Calibrator cost: measured / inferred (D16), figure and basis:

## Decisions in force

| ID | Choice | Date set |
|---|---|---|
| D1 | | |
| D2 | | |
| D3 | | |
| D4 | | |
| D5 | | |
| D6 | | |
| D7 | | |
| D8 | | |
| D9 | | |
| D10 | | |
| D11 | | |
| D12 | | |
| D13 | | |
| D14 | | |
| D15 | | |
| D16 | | |
| D17 | | |
| D18 | | |

## Protocol fixes

| # | What the pilot showed (run, sample, epoch) | Fix | Commit |
|---|---|---|---|
| 1 | | | |

## Deviations

Every departure from the runbook or the pre-registration in this pilot, with its reason. "None" if none.

-
