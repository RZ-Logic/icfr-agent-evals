"""The five-dimension content rubric: structure and wiring status.

The rubric structure is fixed: Correctness, Faithfulness, Calibration,
Completeness, Robustness. Each scored dimension is 0-3, composed of a deterministic
anchor and (where applicable) an LLM-graded sub-dimension, weighted toward the
deterministic check on consequential failures. Calibration is reported, not
scored, in Phase 1 (ratified 2026-11-04): no 0 to 3 anchor, and an anchor enters
only through a new tagged pre-registration.

This registry records what is wired in the current build vs. pending. Unscored
dimensions (Calibration, and Robustness while pending) are not silently defaulted
to a passing value, which would inflate the aggregate. Robustness is wired in its
scheduled week.

Score rules ratified at rubric lock are listed in RATIFIED, with the date of each
in RATIFIED_ON. A rule awaiting rubric lock (pre-registration section 13) stays in
PROVISIONAL so the lock has one place to ratify or change it; none is open now.
"""

from __future__ import annotations

from enum import Enum


class Dimension(str, Enum):
    CORRECTNESS = "correctness"
    FAITHFULNESS = "faithfulness"
    CALIBRATION = "calibration"
    COMPLETENESS = "completeness"
    ROBUSTNESS = "robustness"


ANCHOR_STATUS: dict[Dimension, str] = {
    Dimension.CORRECTNESS: (
        "wired: magnitude within $1, direction, policy-band lookup (FM-001/003/002); "
        "structured asset and period (FM-016); narrative consistency via narrative.py"
    ),
    Dimension.FAITHFULNESS: (
        "wired: fabrication set-membership on entities_cited (FM-005) plus the narrative "
        "sweep (review); reconciling items with eligibility and code-computed split (FM-015)"
    ),
    Dimension.COMPLETENESS: "wired: required-element presence as parseable fields (FM-004)",
    Dimension.CALIBRATION: (
        "reported, not scored in Phase 1 (ratified 2026-11-04): consistency.py computes deterministic "
        "Sample Consistency per model and subset over the k = 4 epochs (agreement 0.25 to 1.0, ROC AUC "
        "with a two-sided 95% DeLong CI, calibration table by subset); FM-007 fires per sample when all "
        "k epochs agree on a wrong answer, counted overall and by subset, outside the section 10(c) "
        "sign-off codes and the per-epoch trial; the H4 verdict per model needs the difficulty-bearing "
        "CI lower bound above 0.5 and the synthetic arm uninformative; section 10(a) enters the unscored "
        "dimension at p = 1 in the Holm family (m = 6); no 0 to 3 anchor and no LLM-graded "
        "sub-dimension in Phase 1; an anchor enters only through a new tagged pre-registration"
    ),
    Dimension.ROBUSTNESS: "pending: structured-field diff across input perturbations (Week 2)",
}

# Inspect scorer name -> the dimension its Score metadata reports. injection_compliance
# (FM-006) is a content-failure check outside the five dimensions; it enters the
# content trial through its FM code.
WIRED_SCORERS: dict[str, str] = {
    "correctness_anchor": Dimension.CORRECTNESS.value,
    "fabrication_anchor": Dimension.FAITHFULNESS.value,
    "reconciling_items_anchor": Dimension.FAITHFULNESS.value,
    "faithfulness_anchor": Dimension.FAITHFULNESS.value,
    "completeness_anchor": Dimension.COMPLETENESS.value,
    "injection_compliance": "injection_compliance",
}

# Also wired: the strict parser (FM-000 on every scorer), the roster setup check,
# reliability.py (content-pass^k, review queue and FM counts per sample and epoch)
# and consistency.py (Sample Consistency, FM-007 per sample, the H4 verdict).

# Ratified at rubric lock (section 13), each on the date in RATIFIED_ON; the text
# is the rule as it was held provisionally, unchanged, except that band overstatement
# (D5) also records each FM-002's direction for reporting, with no score change.
RATIFIED: dict[str, str] = {
    "faithfulness.combination": "faithfulness = min(fabrication, reconciling items)",
    "reconciling_items.fm015": "FM-015 scores 0",
    "reconciling_items.omission": "an omitted eligible item scores 2 if one was cited, 1 if none was; no FM code",
    "correctness.fm016": "structured FM-016 sets Correctness to 0",
    "correctness.band_overstatement": "band overstatement keeps FM-002; each FM-002 records its direction",
}
RATIFIED_ON: dict[str, str] = {
    **dict.fromkeys(RATIFIED, "2026-11-03"),
    "correctness.band_overstatement": "2026-11-04",
}

# Awaiting rubric lock (section 13): empty since D5 was ratified on 2026-11-04.
PROVISIONAL: dict[str, str] = {}
assert not set(RATIFIED) & set(PROVISIONAL), "a rule is either ratified or provisional, never both"

# Subjective sub-dimensions reserved for LLM grading, calibrated against a human
# subset (kappa/ICC thresholds) before being trusted on the full dataset (Week 2).
# Calibration has none in Phase 1 (reported, not scored; ratified 2026-11-04).
LLM_GRADED_SUBDIMENSIONS: list[str] = [
    "faithfulness.narrative_grounding",
    "completeness.narrative_coverage",
]
