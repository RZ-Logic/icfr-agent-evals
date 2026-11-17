# Decision log

The author's rulings on the Phase 1 pre-registration ([`PRE-REGISTRATION-PHASE-1.md`](../PRE-REGISTRATION-PHASE-1.md)) from 2026-11-09 to 2026-11-15, grouped by topic, one plain sentence each. The tag and date in brackets are the ruling's name and date in the pre-registration, which states each rule in full; where this log and the pre-registration differ, the pre-registration governs. Earlier D-numbered decisions (D1 to D18, from 2026-11-05 on; the D1 of 2026-11-11 on difficulty check (b) is listed below), P12 of 2026-11-07 and the ratifications of 2026-11-03 and 2026-11-04 are recorded in §13 of the pre-registration; the few that still shape today's rules are noted where they matter.

Written on 2026-11-13 from the author's private decision record as it stood then: 37 rulings, with D-R16 added the same evening, D-R17 and D-R18 on 2026-11-14, and D-R19 (the outside pre-tag review's fixes) the same day (41). D-R20, the first ruling after the tag, was added on 2026-11-15, and D-R21 on 2026-11-17 (section 13).

## 1. What the score of record is

- **The score that counts is the review-settled one.** Code flags, the author settles the flags, and per-epoch failure codes stay final. The audit-adjusted figure is shown beside it as a sensitivity check only. [D-A, 2026-11-09]
- **The route-blind figure is an upper bound,** and PR CI's McNemar test reads it. A third sign-off rule, R3, is pre-registered now. [D-E5, 2026-11-10]
- **A parse failure (FM-000) is never removed** by the code audit. The audit-adjusted figure keeps it whatever the audit rules. [D-R5, 2026-11-10]
- **Frozen verdicts stay of record.** If anything is amended later, the amended figures are shown beside the frozen ones, labelled amended. [D-M, 2026-11-10]
- **H2 counts only the code-caught failures** that the code audit rules a real error. [D-N, 2026-11-10]
- **FM-000 supports H2 only when the code audit rules it a real error,** like every H2 code; D-R5 keeps FM-000 in the audit-adjusted figure only. [D-R15, 2026-11-13]
- **H3's step rule:** H3 is tested on the 54 difficulty-bearing cases pooled, on four anchors (correctness, fabrication, reconciling items, completeness) and three steps (Haiku 5.5 ≤ Sonnet 5, Sonnet 5 ≤ Sonnet 5.5, Sonnet 5 ≤ Opus 5.5), and holds only when all twelve steps hold. A step holds when the mean per-case paired difference is at or above 0; its paired 95% t interval is reported and does not enter the verdict. H3 is not robust if the two orderings disagree, and not tested if a second look fails. [D-L, 2026-11-10; D-R19, 2026-11-14]

## 2. Citation omissions (naming a case-file entity but leaving it out of the structured citations)

- **A citation omission no longer fails a trial.** It is still detected and counted, and reported as its own measure, citation-pass^4, beside the score of record. Changed after the pilot, because of it, and disclosed. [D-R9, 2026-11-11]
- **CI still guards it:** its own per-case McNemar test (alpha 0.025, outside the Holm family) plus a sign-off case on a rise, kept out of the review volume that sets the ladder rung. [D-R9, 2026-11-11]
- **A reviewer's citation_omission verdict passes only if code confirms the name really is in the case file,** so a made-up name can never pass that way. [D-R9, 2026-11-11]
- **A reviewer's citation_omission verdict is counted apart as review-found.** It never enters citation-pass^4, which code computes from the log alone. [D-R19, 2026-11-14]
- **A reference in reconciling_items counts as cited,** like an entry of entities_cited, so naming it in the narrative is not an omission. An uncited organization, person, address or hash the case file holds exactly is counted as an omission by code, with no review needed. [D-R19, 2026-11-14]
- **Every epoch with an omission the P12 rule finds gets a whole-narrative read,** so none passes unread. An uncited organization, person, address or hash keeps its review route and is read that way; a mention only the wallet-label matcher finds is counted but adds no read. [D-R9, 2026-11-11]
- **Wallet labels are matched directly** (ignoring case, spaces and hyphens), so a missed citation of a wallet is counted. This only adds catches. [D-R9, 2026-11-11]
- Background: P12 (2026-11-07) made an uncited case-file reference or wallet label a citation omission; D-R9 keeps the detection and changes only its effect.

## 3. Amounts and what a narrative may say

- **A band threshold written in dollars is allowed** ("below the $25,000 floor"): it is a fact of the policy. [D-R1, 2026-11-10]
- **A case-file unit price used to restate an allowed figure is allowed** ("6.264955 ETH at 2,991.00 USD/ETH"): it is a rate, not a new dollar amount. [D-R3, 2026-11-10]
- **Stating an excluded item's amount fails** (FM-001), as before. [D-E6, 2026-11-10]
- **A hedged amount passes** once a frozen review verdict says it is correct. [D-E2, 2026-11-09]
- **A hedged or suffixed amount routes to review within its stated precision:** half a unit of the last stated digit for a suffixed figure ("62.9k" within $50), hedged or not, and 5 percent of the target for a hedged figure written out in full, never less than $1. Outside it the $1 rule stands (FM-001). [D-R19, 2026-11-14]

## 4. The extractor and its known gaps

- **Extractor revisions 4, 4b and 5 turn a code into a route only where the reading is ambiguous in a named, tested shape, never into a pass.** The pre-registration says so. [D-R4, 2026-11-10]
- **Known gaps are disclosed,** not hidden: F2-a04, r1:C6, and wallet holder and scope claims. [D-F, 2026-11-10]

## 5. The author's review of flagged narratives

- **A flagged sweep token is "correct"** when it is not an entity, or is a case-file entity cited in another form. [D-E1, 2026-11-09]
- **Review settles four structured flags:** an off-contract value, a type label, a cited ticker, and an ineligible item named in entities_cited but not discussed. [D-E3, 2026-11-09]
- **A flag on an epoch that already has a failure code is still a finding:** it gets a verdict and a whole-narrative read, the epoch still fails on its code, and the author's verdicts decide the audit-adjusted figure. [D-H, 2026-11-10]
- **On the 54 hard cases the author also reads every unflagged, uncoded narrative.** About 79 epochs per worker, 5 to 11 hours per full run; silent passes are only caught by reading. [D-J, 2026-11-10]
- **cannot_settle and not_reviewed fail the trial,** zeroing the anchor the settlement row names. [D-K, 2026-11-10]
- **A whole-narrative "wrong" fails the trial** and zeroes an anchor by error type. [D-I, 2026-11-10]
- **Review load** was stated as about 78 to 259 hours per full run, to be re-derived from the timed pilot review. Re-derived 2026-11-12: about 94 to 126 hours. [D-G, 2026-11-10]

## 6. Checks on the author's own review

- **Second look:** at least 48 hours after the freeze, the author re-reviews a blinded sample per worker run: the larger of 100 units and 10 percent rounded up, capped at the run's units, with every not-correct unit up to 50 counted inside it and correct units filling the rest, never refilled when too few exist. A run is confirmed when each class holds at least 10 sampled units and the kappa lower limit is at least 0.60, or, when the run has fewer than 10 not-correct units in all, when no correct verdict is overturned; otherwise it is labeled review not confirmed. A run with no second-look units is labeled second look empty, never confirmed. The hard-case reads are not re-read; their whole-narrative wrong count is reported beside the gate. The bootstrap seed is derived from the frozen first manifest, never chosen. No outside reviewer gives the second look in Phase 1. [D-D, 2026-11-09; D-R19, 2026-11-14]
- **Blinding check:** 40 units, drawn from every narrative the author read (flagged, coded and the hard-case reads alike), where the author guesses the model. Only a text that no other worker run wrote for the same case qualifies, and no text is shown twice; when fewer than 40 qualify, all are drawn. [D-D, 2026-11-09; D-R6, 2026-11-10; D-R19, 2026-11-14]
- **The freeze commit's Reviewed-by line may sit beside Co-Authored-By attribution lines** in the message's closing trailer block; another reviewer's line, other text or another kind of trailer there still refuses the freeze. The attribution credits the tooling that built the manifest and drafted the commit; every verdict in the record is the named reviewer's. Changed after the pilot's second-look draw refused both pilot freeze commits for this order, and disclosed in the pilot record. [D-R16, 2026-11-13]
- **A second-look unit's class follows its trial effect under D-R9:** it is `correct` when the record is `none_wrong` and every finding passes, a `citation_omission` that code confirms in the case file included. Section 8 had said every finding must be `correct`; the scoring code already applied D-R9 (since 4c083e1, before 271f2fa, the pilot's second freeze, which froze the record the second look drew on). A flip between `citation_omission` and `correct` keeps the class, so it is counted apart, each way, and reported beside κ. Aligned after the pilot's second look showed the two readings differ on one unit, and disclosed in the pilot record. [D-R17, 2026-11-14]

## 7. The route judge (an LLM that would sort flags)

- **Shadow only:** it runs after the review record and its second look are frozen, and changes no score. [D-B, 2026-11-09]
- **No route-judge work before the tag;** its test sets come from real runs' frozen records. [D-C, 2026-11-09]
- **The route judge job's roster, role and serving checks and their unit test are built after the tag,** before the job first runs; their absence at the tag is recorded. [D-R15, 2026-11-13]

## 8. The grader (judge) and its anchors

- **Grader ladder:** GPT-6 Luna, then GPT-6.1 Terra, then GPT-6.1 Sol, then exclude the failing sub-dimension. Terra and Sol run on the OpenAI Batch API. Each gets one look at the 40-case calibration set; never a same-vendor swap. [D-T1, 2026-11-09]
- **The full-run reliability gate never moves the grader:** grader-versus-calibrator ICC(A,1) comes from evaluation-split generations, so it sits outside the remedy order, and a sub-dimension that fails it is excluded for that run only. [D-R19, 2026-11-14]
- **Test-retest and the bias audit are pinned:** the grader scores the calibration set exactly 3 times at each acceptance look, the look's own scoring counted as the first; an undefined α fails the gate; and each bias pass fails when its absolute shift, minus the largest gap between two replicate means, exceeds 0.10 points. [D-R19, 2026-11-14]
- **The grader and calibrator disagreement queue never fails a content trial.** [D-E4, 2026-11-09]
- **Anchors:** the short post-lock version (about 500 words of scope) the author approved on 2026-11-11 replaces the 2026-11-05 text. A narrative with no explanatory claim scores 3; a band given one way is generic; no binary checks, so no kappa in the judge gate; unparsed outputs are not graded; seeds follow the pre-registration's wording; grader and human see one shared facts sheet. [D-R10, 2026-11-11]
- **Grader effort is pinned to high** at every ladder step. [D-R12, 2026-11-11]
- **A calibrator effort check is added before the last code commit,** so calibrator calls carry reasoning_effort high. [D-R15, 2026-11-13]
- **The judge-threshold and A/A-output text of §13 is ratified.** [D-R15, 2026-11-13]

## 9. Prompt, pilot and difficulty

- **The worker prompt is locked as v0.0.4,** the same text as v0.0.4-draft. The pilot of record is the 2026-11-08 Sonnet 5.5 run. [D-R8, 2026-11-11]
- **Difficulty check (b) is read by reading (iii):** a hard case counts as failed only on a code-proven anchor below 3 or a failure code other than FM-000. Recorded before the Haiku 5.5 leg of record ran. [D1, 2026-11-11]
- **dev-hc-000-USDT-M2 stays in the later held-out check,** with the exposure disclosed. [D-R2, 2026-11-10]
- **The deliberate regression is the keyless mock one,** recorded as the pilot's. [D-R12, 2026-11-11]
- **Transcript reading scope (D11):** the 77 reviewed Sonnet 5.5 narratives, the 3 remaining ones after the second look, and every Haiku 5.5 hard-case epoch that fails check (b) under reading (iii); thinking summaries and template outputs are not read, and that is disclosed. [D-R15, 2026-11-13]
- **The pilot record's list of protocol fixes is confirmed;** the type label on unscheduled items stays open after the tag. [D-R15, 2026-11-13]

## 10. CI regression gate

- **Sign-off file:** the format and the path docs/evidence/regression/signoffs/ are ratified ([`SIGNOFF-FORMAT.md`](evidence/regression/SIGNOFF-FORMAT.md)); "noise" means a rise of exactly one epoch. [D-R11, 2026-11-11]
- **The A/A run uses the worker, Sonnet 5.5.** [D-R12, 2026-11-11]
- **Robustness is not scored in Phase 1,** like Calibration: no run generates perturbed inputs, it has no anchor, and it enters the §10(a) Holm family at p = 1 and counts in m = 6. An anchor needs a new tagged pre-registration. [D-R19, 2026-11-14]

## 11. Data, labeling and licences

- **Labeling the 54 hand-written cases:** one rater (Branch B, the pre-registered fallback). No external reviewer is named, so the author labels the 54 cases and re-labels them after at least 48 hours; the statistic is intra-rater kappa on per-item eligibility, with a case bootstrap CI, and the subset earns the name Self-verified only when kappa is at least 0.70; an undefined kappa earns neither name. Branch A (two raters) was the plan from 2026-10-09; the rulings that kept it with a reviewer whose commitment was not yet confirmed are superseded. [D-R7, 2026-11-11; D-R7a, 2026-11-12; D-R18, 2026-11-14; D-R19, 2026-11-14]
- **"Hand-written" means the author writes each case's scenario;** balances, items and arithmetic may be built and checked by code. [D-R13, 2026-11-11]
- **Licences:** code Apache 2.0, datasets CC BY 4.0, a canary GUID in the 0.1.0 dataset files, and a private held-out slice from v0.2. [D-R14, 2026-11-12]

## 12. Cost

- **Calibration-time grading is a one-off cost outside the per-run cap;** Opus 5.5 and Sonnet 5 costs are measured from their own dev logs. Confirmed 2026-11-12: about $140 per full run at grader step 1. [D-R12, 2026-11-11]

## 13. After the tag

D-R20 (2026-11-15) is the first ruling after the tag. It settles points the tagged text leaves open for the tooling built after the tag, and changes no rule, threshold or criterion of the pre-registration. It is logged in DEVIATIONS.md.

- **An errored epoch is a format break (FM-000):** an epoch that ended in an error fails its trial, scores 0 on every anchor, and counts as FM-000 in every FM count, the format-break rate and the CI regression gates. It gets its FM-000 code audit and its whole-narrative read, stays in the audit-adjusted figure, and supports H2 only when the audit rules it `real_error`. The audit sees the error's class name, never its message. An epoch with a missing score and no error stays a failed trial. This follows §9: a missing or refused generation stays a format break. [D-R20, 2026-11-15]
- **The sensitivity row drops only the defect epochs:** an epoch hit by an extractor crash or an answer-key mismatch leaves the row; a case passes when all its remaining epochs pass; a case with every epoch dropped leaves the row; the count of cases scored on fewer than four epochs is given per model and subset. The row is given for all three figures, never as the score of record, and an errored epoch is never dropped from it. [D-R20, 2026-11-15]
- **Blinding check pool:** a text that another worker run also wrote, on any case, leaves the pool, since it has no single right answer; an empty text never enters it. [D-R20, 2026-11-15]
- **H3 robustness** compares the set of failing steps across the readings, and a gap of exactly one standard error counts as saturation. [D-R20, 2026-11-15]
- **An undefined reliability limit fails:** when the full-run ICC(A,1) lower limit cannot be computed, that sub-dimension is excluded for the run and the other is decided on its own value. [D-R20, 2026-11-15]
- **Test-retest and bias-audit details are kept as built:** the alpha bootstrap uses 10,000 unit resamples with undefined resamples dropped and counted, and the one seeded option permutation is kept, and flagged, when it comes out as the identity or the reversal. [D-R20, 2026-11-15]
- **Labeling checks:** the 48-hour gap is shown on every case from the sheet's timer, or the subset gets no name; the published adjudicated-case count is the count of cases differing on any label, a label given in one pass and left blank in the other counting as differing. [D-R20, 2026-11-15]
- **The template baseline goes in the review packet,** so its reference row gets all three figures; an evaluation-split packet without it is refused. [D-R20, 2026-11-15]
- **Grader development rounds (D-R21):** in step 1 of the judge remedy order, the first grading of the judge-development set with the first prompt wording is round 0 and targets no pattern; at most 3 rounds follow, each after an edit for a named disagreement pattern, so the 3-round cap counts edits. [D-R21, 2026-11-17]

## Still open

Nothing at the tag. The author ratified §13 of the pre-registration on 2026-11-14; the tooling the rules need after the tag is listed in its §13 tooling row.
