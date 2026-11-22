# Labeling and adjudication of the 54 hand-written cases (dataset 0.1.0)

This is the methodology record that section 4 of the pre-registration asks for: the labeling kappa with its 95% CI, the adjudicated-case count, and the reasoning for every adjudicated case. Branch B is in force (decision D-R18): the author labeled every case twice, the second time at least 48 hours after the first and without access to the first-pass labels, with the answer key withheld both times.

## Result

| Figure | Value |
|---|---|
| Intra-rater kappa on per-item eligibility | 1.0 (95% case-bootstrap CI 1.0 to 1.0) |
| Items compared | 203 (202 labeled in both passes, 1 in the second pass only) |
| Name earned | Self-verified (the point estimate is at least 0.70) |
| 48-hour gap shown on every case | yes; smallest gap 112.6 hours |
| Adjudicated-case count | 26 |
| Beside the kappa, not gating | variance direction 1.0; band 0.944; evidence state 0.976 |

The adjudicated-case count counts every case where the two passes differ on any label, a label given in one pass and left blank in the other counting as a difference (decision D-R20). The bootstrap seed was recorded in `data/eval/DATASET-0.1.0.json` before the passes were compared.

## Files

- `adjudication.csv`: one row per disagreement, pass against pass first, then each pass against the answer key, with the author's decision and reasoning on every row.
- `summary.json` and `summary.txt`: the output of `scripts/labeling_compare.py`, with the answer key's SHA-256.
- `inputs/`: the two label exports and the key file of each labeling sheet, so the comparison can be run again.

The answer key compared is `data/eval/handwritten.jsonl`, SHA-256 `5c22fe7877465a65298b7ea0527f7031b5ce1b494dc7a0ee61c156e87a2398ad`. It was rebuilt from its sources by `scripts/build_handwritten.py` at the freeze and matched byte for byte.

## Reasoning by case

**22 cases, a blank where 0 was meant.** In the first pass the author left the explained amount blank on cases labeled unexplained, or the unexplained amount blank on cases labeled fully explained. The second pass records 0, as the answer key does. Decision: 0.

**eval-hw-024.** The first pass gave the variance as a surplus of $17,203.66; the second pass and the key give -$17,203.66. A sign slip in the first pass. Its blank unexplained amount is 0, as above. Decision: -17,203.66.

**eval-hw-048.** The second pass labeled the evidence state partly unexplained with an unexplained amount of -$1.00. The labeling rules call a case fully explained when the unexplained amount is within one dollar of zero, so the first pass (fully explained) and the key are right. The first pass left the unexplained amount blank; the second pass and the key give -1. Decision: fully explained, unexplained -1.

**eval-hw-010.** The first pass gave band M1; the second pass and the key give M3. A slip in the first pass. Decision: M3.

**eval-hw-047.** DP-2512-198640 was labeled not eligible in the second pass and not labeled in the first, which counts as not eligible. The key agrees. Decision: not eligible.

**Two cases where both passes differ from the key (eval-hw-010 and eval-hw-040).** In each, both passes marked one item eligible (DP-2511-318764 and DP-2510-447316) that is recorded on both the chain and the ledger and is not open, so it fails two eligibility conditions. The author overlooked the booking columns in both passes. With the item out, no item is eligible: the explained amount is 0, the whole variance is unexplained (-$470,884.34 and $143,512.68) and the evidence state is unexplained, as the key holds. Decision: the key.

No answer-key value changed: every decision agrees with the key the code computed.
