# Decision log

The author's rulings on the Phase 1 pre-registration ([`PRE-REGISTRATION-PHASE-1.md`](../PRE-REGISTRATION-PHASE-1.md)) from 2026-11-09 to 2026-11-13, grouped by topic, one plain sentence each. The tag and date in brackets are the ruling's name and date in the pre-registration, which states each rule in full; where this log and the pre-registration differ, the pre-registration governs. Earlier D-numbered decisions (D1 to D18, from 2026-11-05 on; the D1 of 2026-11-11 on difficulty check (b) is listed below), P12 of 2026-11-07 and the ratifications of 2026-11-03 and 2026-11-04 are recorded in §13 of the pre-registration; the few that still shape today's rules are noted where they matter.

Written on 2026-11-13 from the author's private decision record as it stood then: 37 rulings, with D-R16 added the same evening (38).

## 1. What the score of record is

- **The score that counts is the review-settled one.** Code flags, the author settles the flags, and per-epoch failure codes stay final. The audit-adjusted figure is shown beside it as a sensitivity check only. [D-A, 2026-11-09]
- **The route-blind figure is an upper bound,** and PR CI's McNemar test reads it. A third sign-off rule, R3, is pre-registered now. [D-E5, 2026-11-10]
- **A parse failure (FM-000) is never removed** by the code audit. The audit-adjusted figure keeps it whatever the audit rules. [D-R5, 2026-11-10]
- **Frozen verdicts stay of record.** If anything is amended later, the amended figures are shown beside the frozen ones, labelled amended. [D-M, 2026-11-10]
- **H2 counts only the code-caught failures** that the code audit rules a real error. [D-N, 2026-11-10]
- **FM-000 supports H2 only when the code audit rules it a real error,** like every H2 code; D-R5 keeps FM-000 in the audit-adjusted figure only. [D-R15, 2026-11-13]
- **H3's step rule:** a step holds when its point estimate is at or above 0 with a paired 95% interval; it is not robust if the two orderings disagree, and not tested if a second look fails. [D-L, 2026-11-10]

## 2. Citation omissions (naming an entity but leaving it out of entities_cited)

- **A citation omission no longer fails a trial.** It is still detected and counted, and reported as its own measure, citation-pass^4, beside the score of record. Changed after the pilot, because of it, and disclosed. [D-R9, 2026-11-11]
- **CI still guards it:** its own per-case McNemar test (alpha 0.025, outside the Holm family) plus a sign-off case on a rise, kept out of the review volume that sets the ladder rung. [D-R9, 2026-11-11]
- **A reviewer's citation_omission verdict passes only if code confirms the name really is in the case file,** so a made-up name can never pass that way. [D-R9, 2026-11-11]
- **Every epoch with a code-found omission gets a whole-narrative read,** so nothing passes unread. [D-R9, 2026-11-11]
- **Wallet labels are matched directly** (ignoring case, spaces and hyphens), so a missed citation of a wallet is counted. This only adds catches. [D-R9, 2026-11-11]
- Background: P12 (2026-11-07) made an uncited case-file reference or wallet label a citation omission; D-R9 keeps the detection and changes only its effect.

## 3. Amounts and what a narrative may say

- **A band threshold written in dollars is allowed** ("below the $25,000 floor"): it is a fact of the policy. [D-R1, 2026-11-10]
- **A case-file unit price used to restate an allowed figure is allowed** ("6.264955 ETH at 2,991.00 USD/ETH"): it is a rate, not a new dollar amount. [D-R3, 2026-11-10]
- **Stating an excluded item's amount fails** (FM-001), as before. [D-E6, 2026-11-10]
- **A hedged amount passes** once a frozen review verdict says it is correct. [D-E2, 2026-11-09]

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

- **Second look:** at least 48 hours after the freeze, the author re-reviews a blinded sample (100 units or 10 percent per worker run, plus every not-correct unit up to 50). If agreement falls below kappa 0.60, no correct verdict can be overturned. An outside reviewer is optional. [D-D, 2026-11-09]
- **Blinding check:** 40 units, drawn from every narrative the author read (flagged, coded and the hard-case reads alike), where the author guesses the model. [D-D, 2026-11-09; D-R6, 2026-11-10]
- **The freeze commit's Reviewed-by line may sit beside Co-Authored-By attribution lines** in the message's closing trailer block; another reviewer's line, other text or another kind of trailer there still refuses the freeze. The attribution credits the tooling that built the manifest and drafted the commit; every verdict in the record is the named reviewer's. Changed after the pilot's second-look draw refused both pilot freeze commits for this order, and disclosed in the pilot record. [D-R16, 2026-11-13]

## 7. The route judge (an LLM that would sort flags)

- **Shadow only:** it runs after the review record and its second look are frozen, and changes no score. [D-B, 2026-11-09]
- **No route-judge work before the tag;** its test sets come from real runs' frozen records. [D-C, 2026-11-09]
- **The route judge job's roster, role and serving checks and their unit test are built after the tag,** before the job first runs; their absence at the tag is recorded. [D-R15, 2026-11-13]

## 8. The grader (judge) and its anchors

- **Grader ladder:** GPT-6 Luna, then GPT-6.1 Terra, then GPT-6.1 Sol, then exclude the failing sub-dimension. Terra and Sol run on the OpenAI Batch API. Each gets one look at the 40-case calibration set; never a same-vendor swap. [D-T1, 2026-11-09]
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

## 11. Data, labeling and licences

- **Labeling the 54 hand-written cases:** two raters (Branch A). [name withheld] is named as the external SOX-experienced reviewer, although his commitment to label the 54 cases and to be named is not yet confirmed; it is checked before the repository is made public. If he cannot complete the labeling, Branch B (the author relabels after 48 hours) applies and is logged in DEVIATIONS.md. [D-R7, 2026-11-11; D-R7a, 2026-11-12]
- **"Hand-written" means the author writes each case's scenario;** balances, items and arithmetic may be built and checked by code. [D-R13, 2026-11-11]
- **Licences:** code Apache 2.0, datasets CC BY 4.0, a canary GUID in the 0.1.0 dataset files, and a private held-out slice from v0.2. [D-R14, 2026-11-12]

## 12. Cost

- **Calibration-time grading is a one-off cost outside the per-run cap;** Opus 5.5 and Sonnet 5 costs are measured from their own dev logs. Confirmed 2026-11-12: about $140 per full run at grader step 1. [D-R12, 2026-11-11]

## Still open

- The 3 remaining Sonnet 5.5 transcript reads, after the second look (§13 pilot row).
- The external reviewer's confirmation (§13 labeling row).
- The author's ratification of the §13 decision row.
