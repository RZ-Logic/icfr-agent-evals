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
  - calibration: the 40-case calibration subset of section 4, used only for grader
    versus human alignment: 20 standard cases (calibration_cells) plus the hard
    cases in CALIBRATION_HARD_COMPOSITION (10 edge-type, 10 adversarial-type).
    It sits outside the evaluation and dev splits.

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
from collections import Counter
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
    "ASSETS", "PERIODS", "CUSTODIANS", "BAND_RANGES", "EVIDENCE_MIX", "CALIBRATION_HARD_COMPOSITION",
    "CALIBRATION_STANDARD_CASES", "CI_HARD_COMPOSITION", "DEV_HARD_COMPOSITION", "HARD_COMPOSITIONS",
    "calibration_cells", "PILOT_CASES", "STANDARD_CASES", "SUBSETS", "build_cases", "build_dataset",
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
CALIBRATION_HARD_COMPOSITION = {
    "band_boundary": 4, "ambiguous_narrative": 3, "incomplete_data": 3,
    "AC-1": 2, "AC-2": 2, "AC-3": 2, "AC-4": 1, "AC-5": 1, "AC-6": 2,
}
DEV_STANDARD_CASES = 14
DEV_PILOT_PER_GROUP = 4  # memo-only and non-memo dev standard cases flagged for the pilot

# Calibration standard cases: 20, with EVIDENCE_MIX realized exactly (10/7/3) and a
# memo-only share near the eval split's 42 of 146 (6 of 20, 3 fully explained and 3
# partly unexplained, as CELL_PATTERN puts memo-only links in those two states only).
CALIBRATION_STANDARD_CASES = 20
CALIBRATION_EVIDENCE: dict[tuple[str, bool], int] = {
    ("fully_explained", True): 3, ("fully_explained", False): 7,
    ("partly_unexplained", True): 3, ("partly_unexplained", False): 4,
    ("unexplained", False): 3,
}
# Most cases each factor level and each pair of factors may take among the 20, so no
# level and no pairing dominates: ceil(20 / levels) for one factor and
# ceil(20 / cells) for a pair of factors, except asset x period at 2: with 1 there,
# about 1 shuffle in 200 completes and a stream can need over 2,000 shuffles, while
# at 2 about 1 in 33 completes. Asset x band stays at 1, since the evidence pattern
# rotates by asset x band cell.
CALIBRATION_CAPS: dict[tuple[str, ...], int] = {
    ("asset",): 3, ("band",): 7, ("period",): 7, ("direction",): 10,
    ("asset", "band"): 1, ("asset", "period"): 2, ("asset", "direction"): 2,
    ("band", "period"): 3, ("band", "direction"): 4, ("period", "direction"): 4,
}
CALIBRATION_DRAW_ATTEMPTS = 2_000

# Splits with generated hard cases, and their compositions.
HARD_COMPOSITIONS: dict[str, dict[str, int]] = {
    "ci": CI_HARD_COMPOSITION, "dev": DEV_HARD_COMPOSITION, "calibration": CALIBRATION_HARD_COMPOSITION,
}

SUBSETS = {"pilot": "dev"}  # subset -> the only split it is defined on
PILOT_CASES = 20  # section 13 pilot row: 20 dev-split cases

assert sum(CI_HARD_COMPOSITION.values()) == 54
assert DEV_STANDARD_CASES + sum(DEV_HARD_COMPOSITION.values()) == 30
assert CALIBRATION_STANDARD_CASES + sum(CALIBRATION_HARD_COMPOSITION.values()) == 40
assert sum(CALIBRATION_EVIDENCE.values()) == CALIBRATION_STANDARD_CASES
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


def _factorial_grid() -> list[tuple[str, str, str, str, str, bool]]:
    """The 126 factorial cells (asset, band, period, direction, state, memo_only), one
    per asset x band x period x direction, with CELL_PATTERN rotated by cell. No RNG."""
    cells = []
    for a, asset in enumerate(ASSETS):
        for b, band in enumerate(BANDS):
            c = a * len(BANDS) + b
            for p, period in enumerate(PERIODS):
                for d, direction in enumerate(DIRECTIONS):
                    k = p * len(DIRECTIONS) + d
                    state, memo = CELL_PATTERN[(k + c) % len(CELL_PATTERN)]
                    cells.append((asset, band, period, direction, state, memo))
    return cells


def _factorial_cells(rng: random.Random) -> list[tuple[str, str, str, str, str, bool]]:
    cells = _factorial_grid()
    extra_states = list(EXTRA_STATES)
    rng.shuffle(extra_states)
    for state in extra_states:
        cells.append((rng.choice(ASSETS), rng.choice(BANDS), rng.choice(PERIODS), rng.choice(DIRECTIONS), state, False))
    return cells


_FACTOR_INDEX = {"asset": 0, "band": 1, "period": 2, "direction": 3}


def _cap_keys(cell: tuple) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    return [(names, tuple(cell[_FACTOR_INDEX[n]] for n in names)) for names in CALIBRATION_CAPS]


def calibration_cells(rng: random.Random) -> list[tuple[str, str, str, str, str, bool]]:
    """The 20 calibration standard cells, drawn from the eval split's factorial design.

    The pool is the same 126-cell grid the eval and CI splits use (_factorial_grid:
    every asset x band x period x direction once, with the CELL_PATTERN evidence and
    memo-only rotation). The calibration split's own stream (split_seed('calibration'))
    shuffles the pool and a pass takes cells in that order, skipping a cell that would
    exceed its CALIBRATION_EVIDENCE quota or any CALIBRATION_CAPS limit. A pass that
    ends short of 20 is discarded and the stream reshuffles, up to
    CALIBRATION_DRAW_ATTEMPTS times. So, as in the eval split, no factor is tied to
    another: each asset appears 2 or 3 times, bands and periods 6 or 7 times,
    directions 10 each, each asset meets each band at most once, and no other pair
    of factors piles up in one cell. The evidence mix is EVIDENCE_MIX exactly
    (10/7/3) with 6 memo-only links. The pool holds no seeded extra draws, since 20
    cells cannot carry the eval split's 20 extras."""
    grid = _factorial_grid()
    for _ in range(CALIBRATION_DRAW_ATTEMPTS):
        order = list(grid)
        rng.shuffle(order)
        counts: Counter = Counter()
        evidence: Counter = Counter()
        chosen = []
        for cell in order:
            ev = (cell[4], cell[5])
            keys = _cap_keys(cell)
            if evidence[ev] >= CALIBRATION_EVIDENCE.get(ev, 0):
                continue
            if any(counts[k] >= CALIBRATION_CAPS[k[0]] for k in keys):
                continue
            evidence[ev] += 1
            counts.update(keys)
            chosen.append(cell)
            if len(chosen) == CALIBRATION_STANDARD_CASES:
                return chosen
    raise ValueError(f"no balanced draw of {CALIBRATION_STANDARD_CASES} calibration cells "
                     f"in {CALIBRATION_DRAW_ATTEMPTS} attempts")


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
    """The standard-case specs of a split, in sample order. The calibration split's 20
    come from calibration_cells under the split's own stream."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    rng = random.Random(split_seed(split, seed))
    if split == "calibration":
        return [_spec(rng, split, n, *cell) for n, cell in enumerate(calibration_cells(rng))]
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
    file exists, and the 200 cases are shuffled under a seed derived from the split's.
    On the ci, dev and calibration splits the generated hard cases of HARD_COMPOSITIONS
    join the standard ones, shuffled the same way."""
    cases = [build_case(s) for s in standard_specs(split, seed)]
    if split == "eval":
        hw = load_handwritten(HANDWRITTEN_FILE if handwritten_path is None else handwritten_path)
        if hw:
            cases = cases + list(hw)
            random.Random(split_seed(split, seed) ^ 0x5EED).shuffle(cases)
    if split in HARD_COMPOSITIONS:
        from .hardcases import build_hard_cases

        hard = build_hard_cases(split, split_seed(split, seed), HARD_COMPOSITIONS[split])
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
