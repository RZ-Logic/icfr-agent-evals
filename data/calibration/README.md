# Calibration subset

The 40 worker outputs that the grader is checked against by hand (§4 and §7 of [`PRE-REGISTRATION-PHASE-1.md`](../../PRE-REGISTRATION-PHASE-1.md)). The author grades them blind against the rubric. The grader then gets exactly one acceptance look at the subset, with its test-retest replicates and bias-audit runs. The subset is never used to tune the grader prompt or the rubric wording; that happens on the judge-development set ([`data/dev/judge-dev/`](../dev/judge-dev/README.md)).

Calibration split only. The cases sit outside the 200-case evaluation split and the dev split. An evaluation-split output never enters this directory.

## Cases and outputs

- `calibration.jsonl`: the 40 calibration-split cases, generated at the calibration seed (the evaluation seed 20260607 plus 3,000,000).
- Outputs: three workers (Sonnet 5.5, Sonnet 5 and Haiku 5.5) answer all 40 cases once (epochs 1) under the locked worker prompt. One output per case is picked. The intent is about 13 or 14 per worker, but the build does not enforce that split: it checks only that each pick names one of the three workers. An output that did not parse is never graded (decision D-R10 C5).

## Stratification

As in §4: every score level 0 to 3 appears at least 4 times per LLM-graded sub-dimension. Where natural outputs do not reach a level, a picked output is degraded by one mechanical seed rule of `scripts/judge_dev_build.py` (the same rules as the judge-development set). Seeds are made two ways: omission seeds delete required content, and addition seeds add unsupported content by editing words inside the output's own sentences. Every seed changes only the narrative field. Two further checks run on addition seeds only: no edit may change an amount, reference, band, direction, asset, period, date, time or case entity name, and no edit may change a code score. Omission seeds delete whole clauses and get neither check, so an omission seed can remove an amount or a reference along with its clause.

The build checks the target levels with a margin: every level at least 6 times per sub-dimension, so that an author grade one level away from its target can still leave the floor met. The import checks the plan's targets against the same margin, and the author's grades against the floor of 4 from §4, and refuses when a level falls short; the subset then needs more seeds before the acceptance look. Neither script prints a level count or a shortfall to the screen. The build prints totals only. A refusal says only that the stratification is not met, and the per-level detail goes to a file in the sealed folder (beside the selection input for the build, beside the plan for the import). The rubric defines no binary checks (decision D-R10 C4), so no binary check needs 10 cases of each class.

## How the grades are made

1. `scripts/calibration_build.py` reads the three worker logs, the cases and a sealed selection input kept outside the repo. It writes the grading sheet and a sealed plan, both outside the repo.
2. The author grades every record in the sheet. The sheet shows what the judge-development sheet shows: the narrative, the facts sheet the grader also sees (decision D-R10 C7), the case file as the worker saw it, and the committed scope and level texts. It never shows the worker model, the case id, whether a record is natural or seeded, the seed note or the target level. Records carry a random code and come in a random order.
3. `scripts/calibration_import.py` reads the saved grades file and the sealed plan and writes one record per case to `records/`.

Each build gives the sheet a sheet id, a SHA-256 over the records the sheet shows and the plan. The id is in the sheet data, in the browser storage key, in the plan and in a `sheet_id` column of every row of the saved grades file. A rebuilt sheet that shows other records gets another id. The page will not load a grades file from another sheet, and the import refuses one. Record codes follow from the shuffle seed alone, so a rebuild with `--force` is refused when the selection changed but its shuffle seed did not.

## Record format

One JSON file per case in `records/`, named `<case id>.json`, UTF-8, LF line endings. The fields are the judge-development record format in its order, then `worker_model`.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Calibration-split case id from the log |
| `epoch` | integer | Epoch of the output in its log, always 1 |
| `log_eval_id` | string | `eval.eval_id` of the worker log the output was read from |
| `worker_prompt_sha256` | string | The prompt hash in that log's Task metadata |
| `worker_output` | string | The worker's completion, verbatim; for a seeded record, the degraded text |
| `source` | string | `natural`, `seeded_omission` or `seeded_addition` |
| `seed_note` | string or null | For a seeded record, the rule and exactly what was deleted or edited; otherwise null |
| `stratum` | object | The target level per sub-dimension that this output fills in the §4 stratification |
| `grades` | object | The author's 0 to 3 grade per LLM-graded sub-dimension: `faithfulness.narrative_grounding` and `completeness.narrative_coverage` |
| `anchors_sha256` | string | SHA-256 of the committed 0 to 3 anchor text the grades were made against |
| `graded_by` | string | The author's initials |
| `graded_on` | string | Date graded, `YYYY-MM-DD` |
| `worker_model` | string | The model that wrote the output, as the log names it |

Example (shape only):

```json
{
  "sample_id": "<calibration case id>",
  "epoch": 1,
  "log_eval_id": "<eval_id>",
  "worker_prompt_sha256": "<sha256>",
  "worker_output": "<completion text>",
  "source": "natural",
  "seed_note": null,
  "stratum": {"faithfulness.narrative_grounding": 2, "completeness.narrative_coverage": 3},
  "grades": {"faithfulness.narrative_grounding": 2, "completeness.narrative_coverage": 3},
  "anchors_sha256": "<sha256>",
  "graded_by": "<initials>",
  "graded_on": "<YYYY-MM-DD>",
  "worker_model": "<provider/model>"
}
```

## Rules

- All 40 grades are committed in one commit before the grader's acceptance look. A committed grade is not edited.
- Grades are made without access to any grader score of the same output.
- The subset's score distribution is published beside every agreement statistic, and each estimate is labeled stratified (§4).
- If the intra-rater check of §7 runs, the re-grade is made at least 48 hours after the first grading, without access to these grades, and is reported as intra-rater agreement only.
