# Regression sign-off file format

Status: draft for the owner to ratify. It answers section 13 row 26 of the pre-registration: the sign-off file format, how it records one of the four sign-off outcomes, and the repository path.

## What it is for

Section 10(c) lists the cases on which a sign-off code or a review route fires on more epochs in the candidate than in the baseline. `scripts/compare_runs.py` exits 2 while those cases wait. A reviewer clears them by committing one sign-off file. The file gives every triggered case one of the four outcomes, with a name, a date and a note.

## Where it lives

One file per pull request and gated-content key, in the folder `docs/evidence/regression/signoffs`. The file name is `pr-<N>-<key12>.csv`, where key12 is the first 12 hex characters of the key.

For example, PR 12 with key `abab...ab` is `docs/evidence/regression/signoffs/pr-12-abababababab.csv`.

The gated-content key is the SHA-256 of the gated content in section 10(d). A new commit that changes gated content gets a new key, so it needs a new file. A re-run under the same key pools with the earlier runs. Its candidate eval ids and counts change, so the reviewer updates the file.

## The format

The file shares its format with the section 6 code-audit record (`fm_audit.csv`). It is a CSV file in UTF-8 with no byte order mark and LF line ends. The header comes first and must match exactly. Every row has one cell per column. The same reader checks both files.

The columns are the 15 code-audit columns in the same order, then a purpose field, then four columns that tie the file to one run pair. There is one row per triggered case, that is per case id and trigger.

| Column | What goes in it |
|---|---|
| `row_kind` | Always `signoff`. |
| `unit_id` | Blank. |
| `run_label` | The PR, as `pr-<N>`. The same in every row. |
| `log_eval_id` | The candidate run's eval id. Pooled re-runs give every eval id, sorted, joined by " \| ". The same in every row. |
| `sample_id` | The case id. |
| `epoch` | Blank. A sign-off is per case, not per epoch. |
| `scorers` | Blank. |
| `route_key` | The trigger, exactly as `compare_runs.py` prints it: an FM code or one route key. |
| `route_class` | `fm_code` for an FM code. For a route key, its class: `narrative_sweep` or `code_review`. |
| `reason` | Blank. |
| `sentence` | Blank. |
| `verdict` | One of the four outcomes below. |
| `note` | Required for every outcome. |
| `reviewer` | Required. The reviewer's name. |
| `date` | Required, as YYYY-MM-DD. |
| `purpose` | Always `regression_signoff`. The code-audit record's purpose is `code_audit`. |
| `gated_content_sha256` | The candidate's gated-content key: 64 lowercase hex characters. The same in every row. |
| `baseline_eval_id` | The baseline run's eval id. The same in every row. |
| `baseline_count` | The baseline's failing-epoch count for this trigger, as the report prints it. |
| `candidate_count` | The candidate's pooled failing-epoch count for this trigger, as the report prints it. |

## The four outcomes

These are the outcomes section 10(c) names, one per case:

| Value | Section 10(c) outcome | What the note says |
|---|---|---|
| `a1_fixed` | (a1) fixed | What was fixed and where. |
| `a2_noise` | (a2) noise: a single-epoch rise on a change unrelated to the failing behavior | Why the change is unrelated. Refused unless the candidate count is exactly one above the baseline count. |
| `a3_accepted_trade_off` | (a3) accepted trade-off with a stated reason | The reason. On FM-005 or FM-015 the report lists it as a policy decision. |
| `b_false_positive` | (b) false positive | The parser or extractor defect, and where it is filed. |

## What clears a pending sign-off

`compare_runs.py` reads the file when it is given `--pr` and `--content-key`. It looks at the path above, or at `--signoff PATH`. The cases clear, and the script exits 0, only when there is no automatic block and the file passes every check:

1. The file reads cleanly: no byte order mark, no carriage return, the exact header, one cell per column, at least one row.
2. Every row has `row_kind` `signoff`, purpose `regression_signoff`, a known outcome, a note, a reviewer and a valid date, and the blank columns are blank.
3. Every row names this PR, this gated-content key, this baseline eval id and these candidate eval ids.
4. Every triggered case has exactly one row. No row names a case the run pair does not trigger.
5. Every row's counts equal the run pair's counts.
6. No `a2_noise` row sits on a rise of more than one epoch.

A file that fails any check is refused with exit 3, and every problem is named. With no file, the cases stay pending and the script exits 2. A sign-off file never clears an automatic block under 10(a) or 10(b). The report prints each case's outcome and the count per outcome.

## How the reviewer writes it

Run `compare_runs.py` with `--json`. Write one row per sign-off case listed there. Fill in `verdict`, `note`, `reviewer` and `date`. `icfr_agent_evals/signoff.py` has `template_rows`, which fills every other cell from the run pair. Commit the file on the PR branch.

## Not checked by code

- The gated-content key is taken from `--content-key` as given. No log records it yet. The Actions workflow (week 3) computes it.
- Whether the note is true: that the fix exists, that the change is unrelated, or that the defect is filed.
