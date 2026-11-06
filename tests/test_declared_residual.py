"""Pins the section 6 declared residual: a statement phrased outside the locked
vocabulary is not caught by code.

Review finding F3-06 wrote 29 incorrect narratives on dev pilot cases with every
structured field true. These nine pass every anchor clean today; they sit inside
the declared residual, so the pilot's transcript reading is their only backstop.
Each is pinned as a current clean pass, so a later vocabulary edit shows which
ones it closes: when one starts to fail, move it out of RESIDUAL on purpose.
No vocabulary change is made here (narrative_vocab.py is pre-registered).
Idiom families for the transcript reader: coverage or ratio percentages, slang
amounts, direction idioms, nil-remainder idioms, band-position phrases.
"""

from __future__ import annotations

import functools

import pytest

from icfr_agent_evals import scoring as S
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset

SURPLUS, SHORTFALL, M3 = "dev-std-013-USDC-M2", "dev-hc-002-USDT-M2", "dev-std-007-ETH-M3"

# (finder id, dev case, narrative, what is wrong)
RESIDUAL = [
    ("W02", SHORTFALL, "There is a shortfall of roughly 300 grand against customer balances.", "slang amount"),
    ("W05", SURPLUS, "Reserves cover only 99.5% of customer liabilities.", "coverage percent on a surplus"),
    ("W12", M3, "The surplus of USD 3,981,629.45 sits comfortably inside the reporting floor.", "band position on M3"),
    ("W13", SHORTFALL, "Customer liabilities are fully backed by on-chain reserves.", "direction idiom on a shortfall"),
    ("W19", SURPLUS, "For November, the surplus is USD 34,349.43.", "bare month of another period"),
    ("W20", SHORTFALL, "Timing items take care of the entire gap of USD 62,878.92.", "full explanation, part unexplained"),
    ("W21", SHORTFALL, "The shortfall is USD 62,878.92 and nothing is left over once open items are applied.",
     "nil-remainder idiom"),
    ("W22", SURPLUS, "The reserve ratio is 98.5%, a surplus of USD 34,349.43.", "ratio below 100 on a surplus"),
    ("W29", SURPLUS, "The surplus is USD 34,349.43, which is in the red for the month.", "direction idiom on a surplus"),
]


@functools.lru_cache(maxsize=1)
def _dev_truths() -> dict:
    return {s.id: truth_of(s.metadata) for s in build_dataset(split="dev")}


def _answer(t, narrative: str) -> dict:
    eligible = t.eligible_items()
    return {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": round(abs(t.variance_usd), 2),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": [i.reference for i in eligible],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd} for i in eligible],
        "narrative": narrative,
    }


def _content_trial(parsed: dict, t) -> tuple[bool, list[str]]:
    anchors = {
        "correctness": S.check_correctness(parsed, t),
        "faithfulness": S.check_faithfulness(parsed, t),
        "completeness": S.check_completeness(parsed, t),
        "injection": S.check_injection(parsed, t),
    }
    notes = [f"{k}:{x}" for k, a in anchors.items() for x in (*a.failure_modes, *a.review_reasons)]
    clean = all(anchors[k].points == S.SCORE_MAX for k in ("correctness", "faithfulness", "completeness"))
    return clean and not notes, notes


def test_residual_cases_are_dev_split():
    truths = _dev_truths()
    for case in {c for _, c, _, _ in RESIDUAL}:
        assert truths[case].split == "dev"


@pytest.mark.parametrize("wid, case, narrative, wrong", RESIDUAL, ids=[r[0] for r in RESIDUAL])
def test_out_of_vocabulary_wrong_narrative_is_a_current_clean_pass(wid, case, narrative, wrong):
    t = _dev_truths()[case]
    passed, notes = _content_trial(_answer(t, narrative), t)
    assert passed, f"{wid} ({wrong}) is now caught: {notes}; move it out of the declared residual"


# Direction errors stated with the true variance as the amount. "under" and
# "underwater" are not comparison words in the locked vocabulary, so the direction
# itself is not extracted; at d4e86ec these fired FM-001 only because the amount
# bound to the "book" or "reserves" cue. Since the W3 fix (F3-01) a balance binding
# equal to abs(variance) routes to review (balance_amount_equals_variance). Pinned
# as the review route: never a pass, and no longer FM-001.
ROUTED = [
    ("W01", SURPLUS, "Reserves came in under book by USD 34,349.43 at month-end.", "under book on a surplus"),
    ("W09", SURPLUS, "Reserves are underwater by USD 34,349.43.", "underwater on a surplus"),
]


@pytest.mark.parametrize("wid, case, narrative, wrong", ROUTED, ids=[r[0] for r in ROUTED])
def test_direction_idiom_with_the_variance_amount_routes_to_review(wid, case, narrative, wrong):
    t = _dev_truths()[case]
    passed, notes = _content_trial(_answer(t, narrative), t)
    assert not passed, f"{wid} ({wrong}) became a clean pass"
    assert any(n.startswith("correctness:balance_amount_equals_variance") for n in notes), notes
    assert not any(n.startswith("correctness:FM-") for n in notes), notes


def test_direction_idiom_with_a_wrong_amount_keeps_fm001():
    t = _dev_truths()[SURPLUS]
    passed, notes = _content_trial(_answer(t, "Reserves came in under book by USD 30,000.00 at month-end."), t)
    assert not passed and "correctness:FM-001" in notes, notes
