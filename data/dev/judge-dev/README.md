# Judge-development set

The 20 dev-split worker outputs that step 1 of the judge remedy order (§7 of [`PRE-REGISTRATION-PHASE-1.md`](../../../PRE-REGISTRATION-PHASE-1.md)) iterates the grader prompt and rubric wording on. The author grades them against the rubric, and the grades are committed here before any grader iteration, so no grader round can shape them. The 40-case calibration subset is never used for tuning and never lands here.

Dev split only. An evaluation-split output never enters this directory.

## Source of the outputs

Set by D12 in [`docs/evidence/PILOT-RUNBOOK.md`](../../../docs/evidence/PILOT-RUNBOOK.md) (sequence, step 8): if the locked worker prompt's `worker_prompt_sha256` equals the pilot logs', the outputs come from the pilot logs; otherwise the 20 are regenerated under the locked prompt at one epoch before grading. Every record carries the prompt hash, so the rule can be checked file by file.

## Stratification

As in §7: every score level 0 to 3 appears at least 3 times per LLM-graded sub-dimension, with seeded degraded responses where natural outputs do not reach a level. Degraded responses for narrative coverage are seeded by deleting required content (omission seeds) or adding unsupported content (addition seeds).

## Record format

One JSON file per graded output, named `<sample_id>-e<epoch>.json` (a seeded variant adds `-seed-<n>`), UTF-8, LF line endings.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Dev-split case id from the log |
| `epoch` | integer | Epoch of the output in its log, 1 to 4 |
| `log_eval_id` | string | `eval.eval_id` of the log the output was read from |
| `worker_prompt_sha256` | string | The prompt hash in that log's Task metadata |
| `worker_output` | string | The worker's completion, verbatim; for a seeded record, the degraded text |
| `source` | string | `natural`, `seeded_omission` or `seeded_addition` (`seeded_addition` also covers in-sentence word edits that make a claim unsupported, overstated or adopted; the rule and the exact edits are in `seed_note`) |
| `seed_note` | string or null | For a seeded record, what was deleted or added; otherwise null |
| `stratum` | object | The target level per sub-dimension that this output fills in the §7 stratification, for example `{"faithfulness.narrative_grounding": 1, "completeness.narrative_coverage": 3}` |
| `grades` | object | The author's 0 to 3 grade per LLM-graded sub-dimension: `faithfulness.narrative_grounding` and `completeness.narrative_coverage` |
| `anchors_sha256` | string | SHA-256 of the committed 0 to 3 anchor text the grades were made against |
| `graded_by` | string | The author's initials |
| `graded_on` | string | Date graded, `YYYY-MM-DD` |

Example (shape only):

```json
{
  "sample_id": "<dev case id>",
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
  "graded_on": "<YYYY-MM-DD>"
}
```

## Rules

- The 0 to 3 anchors for both sub-dimensions are written and committed, with their SHA-256, before the first grade (runbook sequence, step 7).
- All 20 grades are committed in one commit before the first grader round. A committed grade is not edited; a rubric revision under the judge remedy order (§7) is logged with its statistics.
- Grades are made without access to any grader score of the same output.
