"""The 54 hand-written cases of the evaluation split (pre-registration section 4).

The author writes each case's scenario; the balances, items and arithmetic of the
answer key are computed by code (decision D-R13) with the same functions every other
split uses (casefile.item_eligibility, item_effect_usd, evidence_state_for,
policy.materiality_band), by scripts/build_handwritten.py. That script writes
data/eval/handwritten.jsonl: one record per case with its id, the case text the worker
reads (input) and metadata holding the answer key (truth) and the structured fixture.

This module only loads that file and checks it. It is absent until the dataset
freeze (section 4, Hygiene): the cases are labeled twice under Branch B (decision
D-R18) with the answer key withheld from the rater, and the file is committed at the
freeze of dataset version 0.1.0, before the first post-tag evaluation-split run.
Until then the evaluation split holds the 146 synthetic cases only, as at the tag.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .casefile import (
    ADVERSARIAL_CATEGORIES, EDGE_CATEGORIES, SCHEMA_VERSION, CaseFile, CaseTruth, validate_truth,
)

HANDWRITTEN_FILE = Path(__file__).resolve().parents[1] / "data" / "eval" / "handwritten.jsonl"
HANDWRITTEN_ID_PREFIX = "eval-hw-"
HANDWRITTEN_EDGE = 30
HANDWRITTEN_PER_AC = 4  # section 4: six adversarial categories, 4 cases each
HANDWRITTEN_CASES = HANDWRITTEN_EDGE + HANDWRITTEN_PER_AC * len(ADVERSARIAL_CATEGORIES)
assert HANDWRITTEN_CASES == 54


class HandwrittenError(ValueError):
    """The hand-written case file breaks the section 4 composition or the answer-key contract."""


def composition_gaps(truths: list[CaseTruth], *, edge: int = HANDWRITTEN_EDGE,
                     per_ac: int = HANDWRITTEN_PER_AC) -> list[str]:
    """Section 4: 30 edge cases and 24 adversarial cases, 4 per AC category, every one on
    the evaluation split with a hand-written id."""
    gaps: list[str] = []
    kinds = Counter(t.case_kind for t in truths)
    if kinds.get("edge", 0) != edge:
        gaps.append(f"{kinds.get('edge', 0)} edge cases, not {edge}")
    if kinds.get("standard", 0):
        gaps.append(f"{kinds['standard']} standard cases among the hand-written cases")
    ac = Counter(t.category for t in truths if t.case_kind == "adversarial")
    for cat in ADVERSARIAL_CATEGORIES:
        if ac.get(cat, 0) != per_ac:
            gaps.append(f"{ac.get(cat, 0)} {cat} cases, not {per_ac}")
    for t in truths:
        if t.split != "eval":
            gaps.append(f"{t.case_id}: split {t.split!r}, not 'eval'")
        if not t.case_id.startswith(HANDWRITTEN_ID_PREFIX):
            gaps.append(f"{t.case_id}: not a hand-written case id ({HANDWRITTEN_ID_PREFIX}...)")
        if t.case_kind == "edge" and t.category not in EDGE_CATEGORIES:
            gaps.append(f"{t.case_id}: edge category {t.category!r}")
    ids = [t.case_id for t in truths]
    if len(ids) != len(set(ids)):
        gaps.append("duplicate case ids")
    return gaps


def load_handwritten(path: str | Path = HANDWRITTEN_FILE, *, edge: int = HANDWRITTEN_EDGE,
                     per_ac: int = HANDWRITTEN_PER_AC) -> list[CaseFile] | None:
    """The hand-written cases as CaseFile objects in file order, or None when the file is
    absent (before the freeze). Every answer key must pass validate_truth and the file
    must hold the section 4 composition; otherwise HandwrittenError."""
    path = Path(path)
    if not path.is_file():
        return None
    cases: list[CaseFile] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = rec.get("metadata") or {}
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise HandwrittenError(f"line {n}: schema_version {meta.get('schema_version')!r}, not {SCHEMA_VERSION!r}")
        truth = CaseTruth.from_dict(meta["truth"])
        if rec.get("id") != truth.case_id:
            raise HandwrittenError(f"line {n}: id {rec.get('id')!r} is not the answer key's case id {truth.case_id!r}")
        if not isinstance(rec.get("input"), str) or not rec["input"].strip():
            raise HandwrittenError(f"line {n}: no case text")
        validate_truth(truth)
        cases.append(CaseFile(truth=truth, fixture=meta["fixture"], rendered=rec["input"]))
    gaps = composition_gaps([c.truth for c in cases], edge=edge, per_ac=per_ac)
    if gaps:
        raise HandwrittenError(f"{path.name}: " + "; ".join(gaps[:8]))
    return cases
