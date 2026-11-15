# Grader live check, 2026-11-15

Pre-registration §5: "The grader configuration is first checked live by the grader build (decision D-R12), after the tag and before any evaluation-split run." This is that check. It is tooling item (2) of the §13 tooling row (decision D-R19).

Run by the author with their own OpenAI key: `scripts/grader_live_check.py --live` on commit 7297be8782367b07514e752b6deb3ee39a1af13f (every log header reads that commit, clean). Each step is one call on the fixed prompt "Reply with the single word OK.", with no case content. The evaluated model is Inspect's `none/none`, so no other model was called. Result: `GRADER LIVE CHECK RESULT: PASS`.

| Step | Grader | Resolved model | Sent | Batch API | Answer | Tokens in / out |
|---|---|---|---|---|---|---|
| 1 | `openai/gpt-6-luna` | `gpt-6-luna` | reasoning effort high, no sampling parameter, no max_tokens | no | OK | 13 / 17 |
| 2 | `openai/gpt-6.1-terra` | `gpt-6.1-terra` | reasoning effort high, no sampling parameter, no max_tokens | yes | OK | 13 / 24 |
| 3 | `openai/gpt-6.1-sol` | `gpt-6.1-sol` | reasoning effort high, no sampling parameter, no max_tokens | yes | OK | 13 / 17 |

Each log passed `grader.check_grader_log`: the grader branch of `roster.check_run_config` (reasoning_effort high, no sampling parameter set, no max_tokens, no extra headers, batch only at steps 2 and 3, no response cache), exactly one grader call to the step's pin, an answer, and a resolved model string equal to the one recorded at the tag, so no vendor-change event. The recorded requests carry `reasoning: {effort: high}` and no temperature. The serving guard checked the endpoint `https://api.openai.com/v1/` at every step.

The check was read again offline from the saved logs on the same day, with no key, and gave the same result.

| Step | Log file | SHA-256 |
|---|---|---|
| 1 | `2026-11-15T20-28-48-00-00_task_7SX8aR3TPois2hjD3yFAKP.eval` | ec8e9e36365358659930e0e8cc10a9ee930bf866277ea2f7c054118aaf123a36 |
| 2 | `2026-11-15T20-28-55-00-00_task_FMHFArL3qmYjfa9hxQqPvd.eval` | a46ff7d6342b5f1d39f39720cb60699ef34d2ef5a2d5f00cb6ddb037fcea7f4c |
| 3 | `2026-11-15T20-32-30-00-00_task_hcAyems9r6HdHoiqPbdj22.eval` | 188c260bdc2bc3f3f7a64045e2256ce8adc152f4b10b8ffbc716ed11aee099ed |

The logs sit in the author's private pilot archive, outside the repository. A live-check log is a record of the grader configuration only: it holds no case and no score.
