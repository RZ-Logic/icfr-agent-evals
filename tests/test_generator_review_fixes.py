"""Regression tests for the generator and hard-case findings of the Week 1 review.

Each test names the review finding it pins (docs/build-notes/review/). They read the
dev and ci splits only.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from casefactory import good_output

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.hardcases import SIGNERS
from icfr_agent_evals.scoring import check_fabrication

SPLITS = ("dev", "ci")
BENCHMARK_STRINGS = ("<INFORMATION>", "to you, the assistant", "Ignore all previous instructions", "###(system_message)")


@pytest.fixture(scope="module", params=SPLITS)
def cases(request):
    return [(s, truth_of(s.metadata)) for s in build_dataset(split=request.param)]


def test_ac1_text_carries_no_benchmark_attack_string(cases):
    """F4-realism-1: AC-1 notes are restyled, not pasted from AgentDojo or InjecAgent."""
    for s, _ in cases:
        for marker in BENCHMARK_STRINGS:
            assert marker.lower() not in s.input.lower(), (s.id, marker)


def test_ac1_signers_are_valid_entities_and_ground(cases):
    """F3-truth-2: a rendered signer is a fixture entity, so citing it is not FM-005."""
    seen = 0
    for s, t in cases:
        if t.category != "AC-1":
            continue
        for name in SIGNERS:
            if name in s.input:
                seen += 1
                assert name in t.valid_entities, (s.id, name)
                out = good_output(t)
                out["entities_cited"] = list(out["entities_cited"]) + [name]
                assert "FM-005" not in check_fabrication(out, t).failure_modes
    assert seen > 0


def test_after_cutoff_withdrawal_decoys_are_not_described_as_arrivals(cases):
    """F3-truth-3, F4-realism-4: an outflow dated after the cutoff never 'came in' or 'arrived'."""
    for s, t in cases:
        for it in t.items:
            # an undated row (date_missing) is not dated after the cutoff
            if it.role != "decoy" or it.date is None or it.date <= t.cutoff_date or it.flow != "outflow":
                continue
            for line in s.input.splitlines():
                if it.reference in line:
                    low = line.lower()
                    assert "came in" not in low and "credit pending" not in low and "arrived" not in low, (s.id, line)


FEE_CEILING_UNITS = {"ADA": Decimal("0.876277"), "ETH": Decimal("0.0005"), "BTC": Decimal("0.0005")}


def test_network_fees_are_chain_plausible(cases):
    """F4-realism-3: every fee line in the ledger and on-chain extracts stays under a
    chain-plausible ceiling in native units (a maximum-size Cardano transaction's
    fee; 24 gwei on 21,000 gas; 50,000 sats)."""
    checked = 0
    for s, t in cases:
        if t.asset not in FEE_CEILING_UNITS:
            continue
        fixture = s.metadata["fixture"]
        for section in ("ledger_extract", "onchain_extract"):
            for line in fixture[section]["lines"]:
                if line.get("desc") == "fee":
                    checked += 1
                    assert abs(Decimal(line["units"])) <= FEE_CEILING_UNITS[t.asset], (s.id, line)
    assert checked > 0
