"""Hard-case templates: edge-type and AC-1 to AC-6 adversarial-type cases.

Interface stub. The templates (band-boundary variances, ambiguous narratives,
incomplete data, AC-1 to AC-6, each AC-1 template recording its injected target and
canary for FM-006) replace the body below. dataset.py calls this to assemble the CI
regression split and the dev split.
"""

from __future__ import annotations

from .casefile import CaseFile


def build_hard_cases(split: str, seed: int, composition: dict[str, int]) -> list[CaseFile]:
    """Generate hard cases for a split.

    split: "ci" or "dev" (the evaluation split's hard cases are hand-written).
    seed: the split seed (casefile.split_seed), so streams differ across splits.
    composition: category -> count, keys from casefile.EDGE_CATEGORIES and
        casefile.ADVERSARIAL_CATEGORIES, e.g. {"band_boundary": 10, "AC-1": 4}.
    """
    return []
