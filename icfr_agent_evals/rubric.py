"""The five-dimension content rubric — structure and wiring status.

The rubric structure is fixed: Correctness, Faithfulness, Calibration,
Completeness, Robustness, each 0-3. Each dimension is composed of a deterministic
anchor and (where applicable) an LLM-graded sub-dimension, weighted toward the
deterministic check on consequential failures.

This registry records what is wired in the current (v0.0.x) skeleton vs. pending.
Pending dimensions are NOT scored yet — they are not silently defaulted to a
passing value, which would inflate the aggregate. Each is wired in its scheduled
week.
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
    Dimension.CORRECTNESS:  "wired — numeric equality, direction, policy-band lookup",
    Dimension.FAITHFULNESS: "wired — fabrication set-membership",
    Dimension.COMPLETENESS: "wired — required-element presence",
    Dimension.CALIBRATION:  "pending — Sample Consistency via epochs = 4 (Week 1 derisk)",
    Dimension.ROBUSTNESS:   "pending — structured-field diff across input perturbations (Week 2)",
}

# Subjective sub-dimensions reserved for LLM grading, calibrated against a human
# subset (kappa/ICC thresholds) before being trusted on the full dataset (Week 2).
LLM_GRADED_SUBDIMENSIONS: list[str] = [
    "faithfulness.narrative_grounding",
    "completeness.narrative_coverage",
    "calibration.semantic_agreement",
]
