# Pilot archive files

Small files from the author's private pilot archive that the pre-registration and the pilot record ask a reader to check. Everything else in that archive stays private: the `.eval` pilot logs (they hold worker text), the review notes and the frozen review record. The private files are cited elsewhere by file name and SHA-256 only.

Each file below was copied from the archive on 2026-11-13. "Source" is the SHA-256 of the archive file; "Published" is the SHA-256 of the file in this folder, with LF line endings. Where the two differ, only wording was edited for publication; no figure, date, hash or decision tag changed.

| File | What it is | Source SHA-256 | Published SHA-256 | Edits |
|---|---|---|---|---|
| [`haiku-check-b-answers.csv`](haiku-check-b-answers.csv) | The author's reads of the 9 Haiku 5.5 hard-case epochs that fail difficulty check (b) under reading (iii), 2026-11-13: one row per case and epoch with verdict (`real` or `not_real`), note and time. Dev split only | b5c41929e8f611c6aa712b05b296f4009937f62798d24e2abbe7de313a2d5239 | the same | none |
| [`sonnet-remaining-reads.csv`](sonnet-remaining-reads.csv) | The author's whole-narrative reads of the 3 Sonnet 5.5 pilot narratives no review record covers (D11, decision D-R15), 2026-11-14, after the second look: one row per case and epoch with verdict (`none_wrong` or `wrong`), note and time. Dev split only. Copied 2026-11-14 | 5d9d2284c436ea3187109f0f4b380862aa68354cc257444be25f13bc5731a56d | the same | none |
| [`REVIEW-LOAD-7dcb85b.txt`](REVIEW-LOAD-7dcb85b.txt) | Output of `scripts/review_load.py` on the first-freeze review record (freeze commit 7dcb85b): the review load re-derived from the two timed sittings, about 94 to 126 hours per full run, with the author's confirmation line added by hand | fe61997cdf8d740033ec4867f1f3c140281d2337d2d6c6e1fcd365e7ea85aee9 | bb439c48897b642e9ce8a9ba1da549a96e36cdb4271f7581ea0447c487a5c2ca | The confirmation line now opens "Author's confirmation"; line endings set to LF |
| [`COST-REDERIVATION.md`](COST-REDERIVATION.md) | The 2026-11-11 working derivation of the cost per full run from the dev logs, behind the §13 cost row | 2f72d12d16646e8a1f33c8ce47f554d24e703be874c10bc10724fe03594414a0 | cf0dd735d2c369f74b4fdad831036f7c4886e51287c28ae19204059901cd6d8e | A status paragraph added under the date line; private folder, branch and interpreter paths removed or shortened to `<logs>`; "owner" reads "author"; "Round 1 fixes" reads "Revisions to the first draft"; two references to earlier design rounds reworded |
| [`fab_gate_per_case.py`](fab_gate_per_case.py) | The simulation behind the operating characteristics of the fabrication rule, §10(b) of the pre-registration: per-case exact McNemar against a pooled-count binomial on the CI split composition, seed 23, 4,000 simulations by default | 8d159282e8a16e4f2d33d53dd7cd2ec1e3f1d705818ace79cd6c3be4bed379d2 | the same | none: copied byte for byte |

Notes on `fab_gate_per_case.py`:

- Its docstring names an earlier design round ("pass-6") and a script from it (`s03_ci_gate.py`) that is not published; the world model it describes is stated in full in the docstring and the code.
- It needs numpy and scipy, which are not in `requirements.txt`. Run it as `python fab_gate_per_case.py [SIMS]`.
- The output recorded beside the original script, which the pre-registration's figures come from (one-sided alpha 0.025; "indep" and "corr" are the two epoch-dependence regimes; "fab x2" is a doubled fabrication rate):

```
scale 0.25 indep  null: pooled-binomial 0.002  per-case McNemar 0.002   | fab x2: pooled-binomial 0.048  per-case McNemar 0.039
scale 0.25 corr   null: pooled-binomial 0.118  per-case McNemar 0.003   | fab x2: pooled-binomial 0.370  per-case McNemar 0.043
scale  1.0 indep  null: pooled-binomial 0.010  per-case McNemar 0.009   | fab x2: pooled-binomial 0.312  per-case McNemar 0.245
scale  1.0 corr   null: pooled-binomial 0.105  per-case McNemar 0.013   | fab x2: pooled-binomial 0.594  per-case McNemar 0.233
```

The pilot report of record (`PILOT-REPORT-dr9-final.txt`) is not copied here: its output is pasted in full in [`../pilot-2026-11-11.md`](../pilot-2026-11-11.md).
