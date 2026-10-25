"""Narrative-consistency extractor and asset/period value checks (section 6).

Interface stub. The extractor that implements section 6 "Correctness, narrative
consistency" (amount-to-role binding, direction, band, asset and period statements)
and FM-016 replaces the bodies below; the signatures and result type are the
contract the correctness anchor calls.

`parsed` is always a schema-valid worker output (the strict parser has already
rejected anything else as FM-000), so every key in casefile.OUTPUT_FIELDS is
present with the right type.
"""

from __future__ import annotations

from dataclasses import dataclass

from .casefile import CaseTruth


@dataclass(frozen=True)
class NarrativeResult:
    """failure_modes: codes the narrative fires (FM-001, FM-002, FM-003, FM-016).
    Any code here sets Correctness to 0.
    review_reasons: non-empty routes the (sample, epoch) to requires_human_review;
    the epoch cannot pass.
    extractions: audit trail of what was extracted and how each item was resolved."""

    failure_modes: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    extractions: tuple[dict, ...] = ()


def check_identity_fields(parsed: dict, truth: CaseTruth) -> tuple[str, ...]:
    """FM-016 on the structured asset and period fields. Returns ("FM-016",) on a
    mismatch, () otherwise."""
    return ()


def check_narrative_consistency(parsed: dict, truth: CaseTruth) -> NarrativeResult:
    """Section 6 narrative consistency over parsed["narrative"]."""
    return NarrativeResult()
