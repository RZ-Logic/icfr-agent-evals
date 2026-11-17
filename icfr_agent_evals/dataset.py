"""Seeded case-file dataset for the Phase 1 content eval (pre-registration section 4).

Ground truth is computed at generation time and stored in each Sample's metadata
(casefile.to_sample). The generator emits structured facts (variance, direction,
band, eligible reconciling items, valid entities) and no reference narrative, so
the LLM-graded sub-dimensions never score against the generator's own prose.

Splits, each from its own RNG stream (casefile.split_seed):
  - eval: the 146 standard synthetic cases, plus the 30 hand-written edge and 24
    hand-written adversarial cases of data/eval/handwritten.jsonl once that file is
    committed at the dataset 0.1.0 freeze (icfr_agent_evals.handwritten; section 4).
    Before the freeze the file is absent and the split is the 146 synthetic cases.
  - ci: 146 standard cases with the same factorial composition plus the generated
    hard cases in CI_HARD_COMPOSITION (30 edge-type, 24 adversarial-type).
  - dev: 14 standard cases (design point D4) plus the hard cases in
    DEV_HARD_COMPOSITION, 30 in all. Used only for elicitation and the pilot.
  - calibration: built later.

Subsets (pre-registration section 13, pilot row): "pilot" keeps the 20 dev cases
whose truth.notes flag pilot (8 standard per D4, 4 memo-only and 4 not, the 6 dev
edge cases and one case per AC category). It exists only on the dev split.

Standard composition: 126 = 7 assets x 3 bands x 3 periods x 2 directions, one
each, plus 20 seeded draws, so asset, band, period and direction are decorrelated.
Within each asset x band cell the six period x direction slots take the evidence
pattern in CELL_PATTERN, rotated by cell, which gives 2 memo-only cases per cell
(42 of 146) and an evidence mix near EVIDENCE_MIX.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable

from inspect_ai.dataset import MemoryDataset, Sample

from .casefile import ASSETS as _ASSETS
from .casefile import BANDS, DIRECTIONS, EVIDENCE_STATES, SPLITS, CaseFile, split_seed, to_sample
from .casefile import PERIODS as _PERIODS
from .generator import BAND_RANGES, CUSTODIANS, CaseSpec, build_case, draw_variance_cents
from .handwritten import HANDWRITTEN_FILE, load_handwritten

ASSETS = list(_ASSETS)
PERIODS = list(_PERIODS)
__all__ = [
    "ASSETS", "PERIODS", "CUSTODIANS", "BAND_RANGES", "EVIDENCE_MIX", "CI_HARD_COMPOSITION",
    "DEV_HARD_COMPOSITION", "PILOT_CASES", "STANDARD_CASES", "SUBSETS", "build_cases", "build_dataset",
    "check_subset", "dataset_name", "is_pilot", "select_subset", "standard_specs",
]

STANDARD_CASES = 146
FACTORIAL_CASES = len(ASSETS) * len(BANDS) * len(PERIODS) * len(DIRECTIONS)  # 126
EXTRA_DRAWS = STANDARD_CASES - FACTORIAL_CASES  # 20

# Target evidence-state mix for standard cases (fully explained, partly unexplained,
# unexplained). CELL_PATTERN realizes it per asset x band cell as 3/2/1 of 6, and the
# 20 extra draws take EXTRA_STATES, for 73/49/24 of 146.
EVIDENCE_MIX = {"fully_explained": 0.50, "partly_unexplained": 0.35, "unexplained": 0.15}
CELL_PATTERN = (
    ("fully_explained", True),
    ("partly_unexplained", True),
    ("fully_explained", False),
    ("fully_explained", False),
    ("partly_unexplained", False),
    ("unexplained", False),
)
EXTRA_STATES = ("fully_explained",) * 10 + ("partly_unexplained",) * 7 + ("unexplained",) * 3

CI_HARD_COMPOSITION = {
    "band_boundary": 10, "ambiguous_narrative": 10, "incomplete_data": 10,
    "AC-1": 4, "AC-2": 4, "AC-3": 4, "AC-4": 4, "AC-5": 4, "AC-6": 4,
}
DEV_HARD_COMPOSITION = {
    "band_boundary": 2, "ambiguous_narrative": 2, "incomplete_data": 2,
    "AC-1": 2, "AC-2": 1, "AC-3": 2, "AC-4": 1, "AC-5": 2, "AC-6": 2,
}
DEV_STANDARD_CASES = 14
DEV_PILOT_PER_GROUP = 4  # memo-only and non-memo dev standard cases flagged for the pilot

SUBSETS = {"pilot": "dev"}  # subset -> the only split it is defined on
PILOT_CASES = 20  # section 13 pilot row: 20 dev-split cases

assert sum(CI_HARD_COMPOSITION.values()) == 54
assert DEV_STANDARD_CASES + sum(DEV_HARD_COMPOSITION.values()) == 30
assert set(EVIDENCE_MIX) == set(EVIDENCE_STATES)


def _spec(rng: random.Random, split: str, idx: int, asset: str, band: str, period: str, direction: str,
          state: str, memo_only: bool, pilot: bool = False) -> CaseSpec:
    return CaseSpec(
        split=split,
        case_id=f"{split}-std-{idx:03d}-{asset}-{band}",
        asset=asset,
        period=period,
        variance_usd=draw_variance_cents(rng, band, direction) / 100,
        seed=rng.getrandbits(63),
        evidence_state=state,
        memo_only=memo_only,
        notes={"pilot": True} if pilot else {},
        template_params={"band": band, "direction": direction},
    )


def _factorial_cells(rng: random.Random) -> list[tuple[str, str, str, str, str, bool]]:
    cells = []
    for a, asset in enumerate(ASSETS):
        for b, band in enumerate(BANDS):
            c = a * len(BANDS) + b
            for p, period in enumerate(PERIODS):
                for d, direction in enumerate(DIRECTIONS):
                    k = p * len(DIRECTIONS) + d
                    state, memo = CELL_PATTERN[(k + c) % len(CELL_PATTERN)]
                    cells.append((asset, band, period, direction, state, memo))
    extra_states = list(EXTRA_STATES)
    rng.shuffle(extra_states)
    for state in extra_states:
        cells.append((rng.choice(ASSETS), rng.choice(BANDS), rng.choice(PERIODS), rng.choice(DIRECTIONS), state, False))
    return cells


def _dev_cells() -> list[tuple[str, str, str, str, str, bool, bool]]:
    """D4: one memo-only and one non-memo case per asset, bands rotated so M1, M2 and
    M3 appear in both groups, period and direction rotated independently of band."""
    cells = []
    for i, asset in enumerate(ASSETS):
        band = BANDS[i % 3]
        period = PERIODS[(i + i // 3) % 3]
        direction = DIRECTIONS[i % 2]
        state = ("fully_explained", "partly_unexplained")[(i // 2) % 2]
        cells.append((asset, band, period, direction, state, True, i < DEV_PILOT_PER_GROUP))
    for i, asset in enumerate(ASSETS):
        band = BANDS[(i + 1) % 3]
        period = PERIODS[(i + 2 + i // 3) % 3]
        direction = DIRECTIONS[(i + 1) % 2]
        state = ("fully_explained", "partly_unexplained", "unexplained")[i % 3]
        cells.append((asset, band, period, direction, state, False, i < DEV_PILOT_PER_GROUP))
    return cells


def standard_specs(split: str, seed: int = 20260607) -> list[CaseSpec]:
    """The standard-case specs of a split, in sample order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "calibration":
        raise NotImplementedError("the calibration subset is built later (section 4)")
    rng = random.Random(split_seed(split, seed))
    if split == "dev":
        cells = _dev_cells()
        order = list(range(len(cells)))
        rng.shuffle(order)
        return [_spec(rng, split, n, *cells[i]) for n, i in enumerate(order)]
    cells = _factorial_cells(rng)
    rng.shuffle(cells)
    return [_spec(rng, split, n, *cell) for n, cell in enumerate(cells)]


def build_cases(split: str = "eval", seed: int = 20260607, handwritten_path=None) -> list[CaseFile]:
    """Every case of a split as CaseFile objects, in sample order. On the eval split the
    hand-written cases (icfr_agent_evals.handwritten) join the synthetic ones when their
    file exists, and the 200 cases are shuffled under a seed derived from the split's."""
    cases = [build_case(s) for s in standard_specs(split, seed)]
    if split == "eval":
        hw = load_handwritten(HANDWRITTEN_FILE if handwritten_path is None else handwritten_path)
        if hw:
            cases = cases + list(hw)
            random.Random(split_seed(split, seed) ^ 0x5EED).shuffle(cases)
    if split in ("ci", "dev"):
        from .hardcases import build_hard_cases

        composition = CI_HARD_COMPOSITION if split == "ci" else DEV_HARD_COMPOSITION
        hard = build_hard_cases(split, split_seed(split, seed), composition)
        if hard:
            cases = cases + list(hard)
            random.Random(split_seed(split, seed) ^ 0x5EED).shuffle(cases)
    ids = [c.truth.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in split {split!r}")
    return cases


def check_subset(split: str, subset: str | None) -> None:
    """Raise ValueError unless subset is None or a known subset of this split. Runs
    before any case is built, so a bad pairing never builds the split."""
    if subset is None:
        return
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; known: {sorted(SUBSETS)}")
    if split != SUBSETS[subset]:
        raise ValueError(f"subset {subset!r} is defined only on split {SUBSETS[subset]!r}, not {split!r}")


def is_pilot(sample: Sample) -> bool:
    """True when the sample's answer key flags it for the pilot. Only a literal True
    counts, so a missing or malformed flag keeps the case out."""
    truth = (sample.metadata or {}).get("truth") or {}
    return (truth.get("notes") or {}).get("pilot") is True


def select_subset(samples: Iterable[Sample], split: str, subset: str | None) -> list[Sample]:
    """The samples of a subset, in sample order; every sample when subset is None.
    The pilot must come to exactly PILOT_CASES, or the dev build has moved and the
    pilot is not the one section 13 registers."""
    check_subset(split, subset)
    samples = list(samples)
    if subset is None:
        return samples
    chosen = [s for s in samples if is_pilot(s)]
    if len(chosen) != PILOT_CASES:
        raise ValueError(f"subset 'pilot' selected {len(chosen)} cases, expected {PILOT_CASES}")
    return chosen


def dataset_name(split: str, subset: str | None = None) -> str:
    return f"variance-commentary-{split}" + (f"-{subset}" if subset else "")


def build_dataset(split: str = "eval", n: int | None = None, seed: int = 20260607,
                  subset: str | None = None) -> MemoryDataset:
    """The Inspect dataset for a split; subset, when given, keeps that subset's
    samples (checked before anything is built); n, when given, then keeps the
    first n samples."""
    check_subset(split, subset)
    cases = build_cases(split, seed)
    samples = select_subset((to_sample(c) for c in cases), split, subset)
    if n is not None:
        samples = samples[:n]
    return MemoryDataset(samples=samples, name=dataset_name(split, subset))


def split_jsonl(split: str, seed: int = 20260607) -> bytes:
    """The split as JSON Lines, one sample per line sorted by id, with id, input and
    metadata (answer key included). The committed data/dev/dev.jsonl is exactly
    split_jsonl("dev"), so a generator change that moves the dev set fails a test."""
    samples = sorted(build_dataset(split=split, seed=seed), key=lambda s: str(s.id))
    rows = [json.dumps({"id": s.id, "input": s.input, "metadata": s.metadata}, sort_keys=True, ensure_ascii=False)
            for s in samples]
    return ("\n".join(rows) + "\n").encode("utf-8")
