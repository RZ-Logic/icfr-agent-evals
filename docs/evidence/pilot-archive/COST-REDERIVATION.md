# Cost per full run, re-derived from the dev logs

Date: 2026-11-11. Status: prep for the section 13 cost row and section 5 Cost. The pre-registration is not edited. No model was called and no key was read.

This is the working derivation of 2026-11-11, copied from the author's private pilot archive; only its wording was edited for publication, and every figure is unchanged. Decision D-R12 (2026-11-11) later put calibration-time work outside the per-run cap and recorded the Opus 5.5 and Sonnet 5 figures as measured; the pre-registration's §13 cost row holds the confirmed figures. CONTEXT.md (the author's private design file, whose SHA-256 the pre-registration's §13 pins) and LOAD.md (a private review-load working note) are cited below by name; neither is published.

Script: `scripts/cost_estimate.py` (written on a branch based on commit 5807a98). Tests: `tests/test_cost_estimate.py`.

Revisions to the first draft (2026-11-11): calibration-time work is now counted the same way as CONTEXT.md's full-run list; the cache-write assumption is stated with a sensitivity line; a measured Scout scan is scaled by transcript length, not count; the review-load model applies LOAD.md section 3's per-sentence rule; the review-load script no longer counts bulk-filled answers as reviewed work.

## Result

A full run is the worker and the three baselines on the 200-case evaluation split at epochs 4 (3,200 generations), with the grader, calibrator and Scout scanner calls on them, and the calibration-time grader work CONTEXT.md lists in the same full-run count.

| Grader step | Expected total | Against the caps | With the route judge (high end, $19) |
|---|---:|---|---:|
| Step 1, GPT-6 Luna | $150.50 | over the $150 soft cap by $0.50, under the $300 hard cap | $169.50 |
| Step 2, GPT-6.1 Terra on Batch | $189.87 | over the soft cap, under the hard cap | $208.87 |
| Step 3, GPT-6.1 Sol on Batch | $239.08 | over the soft cap, under the hard cap | $258.08 |
| Sol at standard price (reference only) | $337.49 | over the hard cap | $356.49 |

These totals count the calibration-time work at its upper reading (see below). The same totals at the lower reading, and with no calibration-time work at all:

| Grader step | Lower reading | No calibration-time work |
|---|---:|---:|
| Step 1, Luna | $147.88, under the soft cap | $143.94, under the soft cap |
| Step 2, Terra on Batch | $176.74 | $157.06 |
| Step 3, Sol on Batch | $212.83 | $173.46 |
| Sol at standard price | $285.00, under the hard cap | $206.27 |

So the expected run sits at the soft cap at step 1. It is just over with all calibration-time work at the upper reading and just under at the lower reading or without it. It is over the soft cap at every other step and once the route judge runs. Nothing expected at steps 1 to 3 reaches the hard cap; Sol at standard price does once calibration-time work is counted at the upper reading. The worst case at max_tokens 48,000 is $2,146 and is far over both caps.

At step 2 or 3, the earlier steps' acceptance looks have already run at their own prices. The totals leave those out. At step 2, add Luna's look, $6.56. At step 3, add Luna's and Terra's looks, $6.56 and $32.81, so $39.37.

## Each component

| Component | Basis | Calls | Input per call | Output per call | Cost |
|---|---|---:|---:|---:|---:|
| Sonnet 5.5 worker | measured | 800 | 5,742 | 1,398 | $20.37 |
| Haiku 5.5 baseline | measured | 800 | 5,742 | 5,112 | $2.50 |
| Sonnet 5 baseline | measured | 800 | 5,740 | 3,381 | $54.35 |
| Opus 5.5 baseline | measured | 800 | 5,742 | 1,491 | $42.23 |
| Calibrator, Opus 5.5 | inferred | 240 | 6,243 | 1,491 | $13.15 |
| Grader, GPT-6 Luna, on the 3,200 generations | assumed | 3,200 | 6,252 | 800 | $3.28 |
| Grader, calibration subset, first scoring | assumed, upper reading | 800 | 6,252 | 800 | $0.82 |
| Grader, test-retest replicates | assumed, upper reading | 2,400 | 6,252 | 800 | $2.46 |
| Grader, bias-audit passes | assumed, upper reading | 3,200 | 6,252 | 800 | $3.28 |
| Scout scanners, GPT-6 Luna | placeholder | 6,400 | 8,587 | 800 | $8.06 |
| Route judge, GPT-6 Luna, if it runs | stated in section 5 | | | | $9 to $19 |

Output tokens include billed thinking. Inspect's `output_tokens` holds the thinking; `reasoning_tokens` is a part of it and is never added on top.

What each basis means:

- **Measured.** Each worker's cost comes from its own dev log. The mean cost per call is taken per stratum (8 standard and 12 hard dev cases) and scaled to the evaluation split's 146 standard and 54 hard cases at 4 epochs (584 and 216 generations per worker). Every one of the 320 dev calls stopped on `stop`; none hit max_tokens.
- **Inferred.** The calibrator is about 10 percent of the 2,400 non-Opus generations, so 240 calls at Opus 5.5 prices. Its input is Opus 5.5's mean dev input plus the mean visible answer of the three non-Opus workers. Its output is Opus 5.5's mean dev output. No calibrator call has run.
- **Assumed.** No grader call has run. Each grader call reads the mean worker input (a stand-in for the case file and instructions) plus the mean visible answer, and writes 800 tokens (the judge call as CONTEXT.md budgets it, reasoning included). The calibration-time calls have the same shape.
- **Placeholder.** The Scout scanner figure waits on the author's scan of the Sonnet 5.5 pilot log (runbook step 10). The placeholder is 2 model scanners on 3,200 transcripts, each reading the worker's input plus its whole output and writing 800 tokens. Rerun the script with `--scout-scan CALLS,INPUT,OUTPUT` from that scan and the line becomes measured (see Scout below).
- **Stated.** The route judge figure is carried from section 5 as written. It is not re-derived here.

## Calibration-time work

CONTEXT.md's full-run list (Cost section, Phase 1 bullet) counts the grader on every generation "plus at least 3 test-retest replicates on the 40-case calibration subset", and says "The calibration subset adds ~800 graded both ways". The Autograder calibration bullet says the subset is graded "both manually (against rubric) and by autograder", so "both ways" is read as by hand and by the grader. The hand grades are human time. The bias audit runs at the same acceptance look ("its test-retest replicates and bias-audit runs included"): (a) option order reversed and one random permutation, (b) score labels relabeled, (c) dimension order permuted where dimensions share a prompt. It is not on the full-run list.

| Item | On the full-run list | Upper reading (counted) | Lower reading | Inside the dollar total |
|---|---|---:|---:|---|
| First scoring of the calibration subset | yes ("~800 graded both ways") | 800 calls, $0.82 | 640 calls, $0.66 | yes |
| Test-retest replicates | yes ("at least 3") | 2,400 calls, $2.46 | 1,280 calls, $1.31 | yes |
| Bias-audit passes | no, same look | 3,200 calls, $3.28 | 1,920 calls, $1.97 | yes, for consistency |
| Hand grades of the subset | yes (the other way) | human time | human time | no |
| Intra-rater check, if it runs | yes ("human time only") | human time | human time | no |

Costs are at step 1 (Luna). At step 3 (Sol on Batch) the counted calibration-time work is $65.61 at the upper reading and $39.37 at the lower.

- **Upper reading:** 800 outputs per pass (CONTEXT.md's figure), 3 test-retest replicates on top of the first scoring, 4 bias-audit passes ((a) twice, (b), (c)). 6,400 grader calls.
- **Lower reading:** 640 outputs per pass (40 cases x 4 epochs x 4 workers), the first scoring counted as replicate 1, 3 bias-audit passes ((c) not applying). 3,840 grader calls.
- **Not counted:** if "both ways" meant grader and calibrator, the 800 outputs would add 800 Opus 5.5 calls, $43.83. That would put step 1 at about $194.

The earlier draft counted only the test-retest calls (1,920, at 640 outputs per replicate) and left the ~800 and the bias audit out. All three are now counted alike.

## Scout

With `--scout-scan`, the scan's own totals replace the placeholder. The scan reads 80 Sonnet 5.5 transcripts. Haiku 5.5 and Sonnet 5 transcripts are longer (mean output 5,112 and 3,381 tokens against 1,398). A scanner's input grows with the transcript, so the script scales the scan's input to each worker's 800 transcripts by that worker's measured mean input plus output against Sonnet 5.5's (ratios average 1.20: 8,587 against 7,140 tokens). The scanner's output is scaled by transcript count only. The report prints the count-scaled figure beside it. A scan that matches the placeholder's shape gives the same $8.06; the count-scaled figure would be $7.13.

## Worker detail

| Worker | Dev log, private pilot archive (SHA-256) | Prompt | Mean thinking | Max output |
|---|---|---|---:|---:|
| Sonnet 5.5 | `sonnet-v004/...HVs4KEaDhcWASS77s48w63.eval` (9a54ace02a9bc5dc75f50dd1c2cab6f21be4e346e14610b0ebaa4660478b547a) | v0.0.4-draft, 8eee5a57bafb | 932 | 2,649 |
| Haiku 5.5 | `haiku55/...9Xod6GaskGma3Cni6DR9GL.eval` (554d46988110cadedce4506aaf88a70ee08382337281286a8cd7baba0fc639e7) | v0.0.4, 8eee5a57bafb | 4,604 | 10,150 |
| Sonnet 5 | `sonnet5/...mhTHNhGPnNYX3EB5kcYDjb.eval` (f904c85310a51daffebc3d6877436395bde365fac01894904b8952b2dc5704e3) | v0.0.4-draft, 8eee5a57bafb | 2,852 | 12,701 |
| Opus 5.5 | `opus55/...XY6rSSHF7XdEwnanzMBebF.eval` (159cfc9328eacb0d55fce87c69a33758f29647d8f488146ec15260304391f4bf) | v0.0.4-draft, 8eee5a57bafb | 953 | 2,760 |

All four logs carry the same worker prompt hash, so they are one prompt text. Sonnet 5 is the largest cost: it thinks about three times as long as Sonnet 5.5 and is priced at $15 per million output tokens.

## Grader output: where the caps bind

No grader has run, so 800 output tokens per call is a guess. The total is most sensitive to it. Grader cost below covers the 3,200 generations and the calibration-time work at the upper reading (9,600 calls).

| Output tokens per grader call | Step 1 grader | Step 2 grader | Step 3 grader | Sol standard |
|---:|---:|---:|---:|---:|
| 800 | $9.84 | $49.21 | $98.42 | $196.84 |
| 2,000 | $15.60 | $78.01 | $156.02 | $312.04 |
| 8,000 | $44.40 | $222.01 | $444.02 | $888.04 |

Output tokens per grader call at which the total reaches each cap, route judge left out:

| Grader step | Reaches $150 at | Reaches $300 at |
|---|---:|---:|
| Step 1, Luna | 696 | 31,946 |
| Step 2, Terra on Batch | over already | 5,389 |
| Step 3, Sol on Batch | over already | 2,069 |
| Sol at standard price | over already | 409 |

This supports the D-T1 reasoning. Sol at standard price passes the hard cap once a grader call writes more than about 400 tokens. Sol on Batch needs about 2,100.

## Worst case at max_tokens 48,000

No max_tokens pin is set, so every Claude call may run to 48,000 output tokens.

| Part | At 3,482 input (the cost row) | At the measured input |
|---|---:|---:|
| Sonnet 5.5 | $389.57 | $393.19 |
| Haiku 5.5 | $19.48 | $19.66 |
| Sonnet 5 | $584.36 | $589.78 |
| Opus 5.5 | $779.14 | $786.37 |
| All workers | $1,772.55 | $1,789.00 |
| Calibrator at 48,000 out | $236.39 | $236.39 |
| Route judge worst (stated) | $103.00 | $103.00 |
| Grader step 1 and Scout, expected | $17.90 | $17.90 |
| Total | $2,129.84 | $2,146.29 |

The worker figures at 3,482 input reproduce the cost row exactly (a test checks this against the pre-registration text). The measured input is 5,742 tokens per call, not 3,482: the 3.0 characters per token guess undercounts. At 48,000 output tokens the input is about 2 percent of a call, so the measured input adds only $16.45 to the workers' worst case. The worst case is one attempt per generation. `--retry-on-error 2` can bill an errored sample-epoch up to three times, and a cancelled streamed generation is billed.

## Prices and where they are pinned

| Model | In / out per million tokens | Source |
|---|---|---|
| Sonnet 5.5 | $2 / $10 | CONTEXT.md "Prices per MTok" bullet; PILOT-RUNBOOK.md "Cost at list price" |
| Haiku 5.5 | $0.10 / $0.50 | same bullet (prompts up to 100K tokens); PILOT-RUNBOOK.md |
| Sonnet 5 | $3 / $15 | same bullet (budgeted at an earlier steady-state price, $3/$15) |
| Opus 5.5 | $4 / $20 | same bullet; calibrator role bullet; PILOT-RUNBOOK.md |
| GPT-6 Luna | $0.10 / $0.50 | same bullet; grader role bullet |
| GPT-6.1 Terra on Batch | $0.50 / $2.50 | CONTEXT.md ADR-0006 bullet (D-T1) and the "Prices per MTok" bullet |
| GPT-6.1 Sol on Batch | $1 / $5 | same |
| GPT-6.1 Sol, standard | $2 / $10 | same |

No price was fetched. The logs show prompt-cache reads and writes on every worker. Cache reads are 75 to 78 percent of dev input tokens and cache writes 22 to 25 percent (on Sonnet 5.5, 99,911 of 456,732 input tokens are writes). No cache price is pinned in the repository's documents, so every input token is billed at the base input price. Claude input (workers and calibrator) is $47.79 of the $150.50 step 1 total.

Cache reads are usually billed below base price and cache writes above it. So the figures are an upper estimate on input only if the read discount outweighs the write premium. Neither price is pinned, so the figures may not be an upper bound. Cache writes are $10.84 of the Claude input at base price. With reads left at base, a write premium adds:

| Cache-write price, times base input (assumed, not pinned) | Added | Step 1 total |
|---|---:|---:|
| 1.25 | $2.71 | $153.21 |
| 2 | $10.84 | $161.34 |

The calibrator's cache writes are inferred at the Opus 5.5 worker's write share.

## Recommendation

Confirm the caps as they stand ($150 soft, $300 hard) and record in the cost row: expected $150.50 per full run at grader step 1 with all calibration-time work at the upper reading ($147.88 at the lower reading, $143.94 without it, $169.50 with the route judge), worker figures measured from all four dev logs, the calibrator inferred, the cache-price assumption stated, the Scout figure filled from the author's scan, and the worst case of $1,772.55 at the row's input ($1,789.00 at the measured 5,742 input).

## Review load (ruling D-G), demonstrated

Script: `scripts/review_load.py`, tests `tests/test_review_load.py`. It reads the packet's `reviewer/packet.csv` and `sealed/summary.json` and the sittings file the review page saves. It reuses `review_freeze.py`'s own sittings reader, checks and hour count. No sittings file exists yet, so it ran on the author's packet `runs/reviews/pilot-2026-11-11` (private, untracked; read only) twice:

- **Without sittings (the D-G assumptions on this packet's counts):** about 91 to 276 hours per full run, against the stated 78 to 259. The D-J narrative-only read is about 306 narratives and 5.6 to 11.1 hours, against the stated 5 to 11.
- **On a synthetic sittings file (two sittings, 4.25 hours; not real data):** about 153 to 207 hours. This only shows the mechanics. The real figure comes from the author's sittings file.

The model now follows LOAD.md section 3 for finding verdicts: 30 or 90 s per flagged sentence plus 10 or 20 s for each further key on that sentence. Finding units are grouped by their narrative unit and routed sentences as `packet.csv` writes them; a unit that routes no sentence is its own group. This packet's 103 finding units sit on 75 flagged sentences, so 28 are further keys. Case facts are charged per case with a route (19 of the 20 dev cases), as section 5 says. The earlier draft charged 30 or 90 s per finding unit and case facts per case with any unit, which gave 98 to 307 hours. The range is still above D-G's because the D-D second look samples 100 narrative units rather than 100 findings, and D-G grouped 77 items where this packet has 75 sentences.

The sittings file gives one total time. The script shares it out over finding, narrative, audit and case-fact time by the D-G weights, so per-kind minutes are apportioned, not timed one by one. With the filled answers file it also tries a fit by date. That works only when the review spans more dates than unit kinds and the kinds differ from day to day.

Answers filled in bulk are not reviewed work. The review desk's "Mark the rest not_reviewed" button stamps that day's date on every unanswered finding at once. Rows marked not_reviewed, and rows left blank, are left out of the fit by date and out of the dev weights the measured time is shared over. They stay in the projection, since a full run reviews every unit. The desk also used to fill "correct" on a narrative's unanswered findings when the narrative was answered none_wrong. That was removed on 2026-11-11, but answers made before the fix may carry it. Those rows have a date and no review time of their own, and the answers file cannot tell them apart from hand answers. So the fit by date is a weak diagnostic. It is shown beside the range and never used in it. The script reads only whether a verdict is blank, not_reviewed or something else, and prints no verdict value.

Commands, from the repository root (`python` is the Python 3.14.10 venv interpreter; `<logs>` is the private log folder):

```
env -u ANTHROPIC_BASE_URL -u ANTHROPIC_API_KEY -u OPENAI_API_KEY PYTHON_DOTENV_DISABLED=1 PYTHONPATH=. \
  python scripts/cost_estimate.py \
  <logs>/sonnet-v004/*.eval <logs>/haiku55/*.eval \
  <logs>/opus55/*.eval <logs>/sonnet5/*.eval \
  [--scout-scan CALLS,INPUT,OUTPUT]

env -u ANTHROPIC_BASE_URL -u ANTHROPIC_API_KEY -u OPENAI_API_KEY PYTHON_DOTENV_DISABLED=1 PYTHONPATH=. \
  python scripts/review_load.py \
  --packet-dir runs/reviews/pilot-2026-11-11 [--sittings sittings.csv [--answers answers.csv]]
```

Both exit 0. The cost script refuses any log that is not a successful dev-split log before it opens a sample, and it never reads a score.

## Open questions for the author (as of 2026-11-11)

1. The cost row says the Opus 5.5 and Sonnet 5 worker figures are inferred from the Sonnet 5.5 and Haiku 5.5 logs. Both now have their own dev logs, so their figures are measured. Should the row say so?
2. Sonnet 5 is priced at that earlier budget figure, $3 / $15. It is now the largest single cost ($54). Is that still the price to record?
3. Is calibration-time work inside the per-run cap? CONTEXT.md lists the test-retest replicates and the ~800 calibration-subset grades in the full-run count; the bias audit runs at the same look but is not listed. At step 1 they add $6.56 at the upper reading (800 outputs per pass, 3 replicates beyond the first scoring, 4 bias-audit passes) and $3.94 at the lower (640 outputs, first scoring as replicate 1, 3 passes); at step 3, $65.61 and $39.37. Counting them moves step 1 from $143.94 to $150.50, over the soft cap. Which pass size is right, 800 or 40 x 4 x 4 = 640? And does "graded both ways" mean by hand and by the grader (read here), or grader and calibrator (another $43.83)?
4. No prompt-cache price is pinned. Cache reads are about three quarters of input tokens and cache writes about a fifth. Should cache-read and cache-write prices be pinned in CONTEXT.md so the estimate can use them? Until then every input token is billed at base price, which is an upper estimate only if the read discount outweighs the write premium. At a write premium of 1.25 times base, step 1 rises by $2.71.
5. The Sonnet 5.5 log is the 2026-11-08 v0.0.4-draft run at 59128ba. Same prompt hash as the lock. Use it, or a Sonnet 5.5 leg run at the locked commit if one exists?
