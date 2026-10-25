"""Seeded synthetic dataset for the Phase 1 content eval (smallest version).

Ground truth is computed at generation time and stored in each Sample's metadata.
The generator emits only *structured facts* (variance, direction, band, valid
entities) — it does NOT emit a reference narrative. This separation is deliberate:
the subjective LLM-graded sub-dimensions (Week 2) must score against an
independently authored reference, never against the generator's own output, or the
eval becomes circular.

This is the 20-example skeleton. Week 2 expands to 200 evaluation examples + 30 dev
+ 25 calibration, adds the dual-labeled edge/adversarial Verified subset, and
deepens the evaluation-awareness mitigations. The dev/test split mechanism exists
here so prompt elicitation never touches the evaluation set.
"""

from __future__ import annotations

import random

from inspect_ai.dataset import MemoryDataset, Sample

from .policy import asset_tier, materiality_band

ASSETS = ["BTC", "ETH", "SOL", "USDC", "USDT", "XRP", "ADA"]
PERIODS = ["2025-10", "2025-11", "2025-12"]
CUSTODIANS = ["Anchorage Digital", "BitGo Trust", "Coinbase Custody", "Fireblocks", "Komainu"]

# magnitude ranges that fall strictly inside each band; boundaries are reserved for
# the Week-2 edge-case set, not the clean synthetic set.
BAND_RANGES = {
    "M1": (1_000.0, 24_500.0),
    "M2": (26_000.0, 245_000.0),
    "M3": (260_000.0, 5_000_000.0),
}


def _wallet_for(asset: str, rng: random.Random) -> str:
    def h(n: int) -> str:
        return "".join(rng.choice("0123456789abcdef") for _ in range(n))

    if asset == "BTC":
        return "bc1q" + h(38)
    if asset in {"ETH", "USDC", "USDT"}:
        return "0x" + h(40)
    if asset == "SOL":
        b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        return "".join(rng.choice(b58) for _ in range(44))
    if asset == "XRP":
        alnum = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        return "r" + "".join(rng.choice(alnum) for _ in range(33))
    return "addr1" + h(50)  # ADA


def _make_sample(rng: random.Random, idx: int, split: str) -> Sample:
    asset = ASSETS[idx % len(ASSETS)]
    tier = asset_tier(asset)
    band = ["M1", "M2", "M3"][idx % 3]            # rotate to cover all bands
    period = PERIODS[idx % len(PERIODS)]
    direction = rng.choice(["surplus", "shortfall"])

    ledger = round(rng.uniform(5_000_000, 200_000_000), 2)
    lo, hi = BAND_RANGES[band]
    mag = round(rng.uniform(lo, hi), 2)
    signed = mag if direction == "surplus" else -mag
    onchain = round(ledger + signed, 2)

    custodian = rng.choice(CUSTODIANS)
    wallet = _wallet_for(asset, rng)

    # internal consistency guard: the generated band must equal the policy band
    assert materiality_band(signed) == band, "generator produced an inconsistent band"

    truth = {
        "asset": asset,
        "asset_tier": tier,
        "period": period,
        "variance_usd": signed,
        "direction": direction,
        "materiality_band": band,
        "reserve_ratio": round(onchain / ledger, 6),
        "valid_entities": [custodian, wallet],
    }

    prompt = (
        f"Reserve reconciliation record — {asset} (tier {tier}) — period {period}\n"
        f"Ledger liabilities (USD-equivalent): ${ledger:,.2f}\n"
        f"On-chain reserves (USD-equivalent): ${onchain:,.2f}\n"
        f"Primary custodian: {custodian}\n"
        f"Reference wallet: {wallet}\n\n"
        f"Produce variance commentary for this reconciliation."
    )

    return Sample(
        input=prompt,
        target=band,  # not used by the deterministic anchors; recorded for convenience
        id=f"{split}-{idx:03d}-{asset}-{band}",
        metadata={"truth": truth, "split": split},
    )


def build_dataset(split: str = "eval", n: int | None = None, seed: int = 20260607) -> MemoryDataset:
    """Build the synthetic dataset.

    split='eval' -> evaluation set (default n=20 in this skeleton)
    split='dev'  -> dev set for prompt elicitation (default n=30), disjoint seed
    """
    if n is None:
        n = 20 if split == "eval" else 30
    salt = {"eval": 0, "dev": 1_000_000}.get(split, 7)
    rng = random.Random(seed + salt)
    samples = [_make_sample(rng, i, split) for i in range(n)]
    return MemoryDataset(samples=samples, name=f"variance-commentary-{split}")
