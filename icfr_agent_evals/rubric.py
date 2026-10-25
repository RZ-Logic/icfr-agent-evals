"""The five-dimension content rubric: structure and wiring status.

The rubric structure is fixed: Correctness, Faithfulness, Calibration,
Completeness, Robustness, each 0-3. Each dimension is composed of a deterministic
anchor and (where applicable) an LLM-graded sub-dimension, weighted toward the
deterministic check on consequential failures.

This registry records what is wired in the current build vs. pending. Pending
dimensions are NOT scored yet; they are not silently defaulted to a passing value,
which would inflate the aggregate. Each is wired in its scheduled week.

Provisional score rules awaiting rubric lock (pre-registration section 13) are
listed in PROVISIONAL so the lock has one place to ratify or change them.
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
    Dimension.CALIBRATION: "pending: Sample Consistency via epochs = 4 (Week 1 derisk)",
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
# and reliability.py (content-pass^k, review queue and FM counts per sample and epoch).

PROVISIONAL: dict[str, str] = {
    "faithfulness.combination": "faithfulness = min(fabrication, reconciling items)",
    "reconciling_items.fm015": "FM-015 scores 0",
    "reconciling_items.omission": "an omitted eligible item scores 2 if one was cited, 1 if none was; no FM code",
    "correctness.fm016": "structured FM-016 sets Correctness to 0",
    "correctness.band_overstatement": "band overstatement keeps FM-002",
}

# Subjective sub-dimensions reserved for LLM grading, calibrated against a human
# subset (kappa/ICC thresholds) before being trusted on the full dataset (Week 2).
LLM_GRADED_SUBDIMENSIONS: list[str] = [
    "faithfulness.narrative_grounding",
    "completeness.narrative_coverage",
    "calibration.semantic_agreement",
]
