"""Asset tiers and materiality bands: the deterministic policy lookup.

Phase 1 uses the materiality-band lookup inside the Correctness anchor (the
worker's stated band must equal the band the inputs actually fall into). Phase 2
will add the 9-cell action matrix on top of these same tiers and bands.

Thresholds and ratio bands are NOTIONAL for the eval. A production deployment
requires entity-specific materiality determination by the auditor; this matrix is
a methodology-demonstration vehicle, not a deployable policy.
"""

from __future__ import annotations

from typing import Iterable

# --- Asset tiers (Kraken Dec 31, 2025 PoR scope) ---------------------------------
ASSET_TIERS: dict[str, str] = {
    "USDC": "T1", "USDT": "T1",              # T1 stablecoins (ceiling = Kraken's 105%+ display cap)
    "BTC": "T2", "ETH": "T2", "SOL": "T2",   # T2 major coins (modest over-collateralization)
    "XRP": "T3", "ADA": "T3",                # T3 mid-tail (slightly wider tolerance)
}

EXPECTED_RATIO_BAND: dict[str, tuple[float, float]] = {
    "T1": (1.00, 1.05),
    "T2": (1.00, 1.015),
    "T3": (1.00, 1.03),
}

# --- Materiality bands (notional; anchored to AS 2201 severity definitions) -------
# M1 below floor: < $25K ; M2 reportable: $25K-$250K ; M3 material-risk: > $250K
M1_CEILING = 25_000.0
M3_FLOOR = 250_000.0


def asset_tier(asset: str) -> str:
    try:
        return ASSET_TIERS[asset.strip().upper()]
    except KeyError as e:
        raise KeyError(
            f"asset {asset!r} not in Dec-31-2025 PoR scope {sorted(ASSET_TIERS)}"
        ) from e


def materiality_band(variance_usd: float) -> str:
    """Map a USD-equivalent variance to a materiality band on |variance|.

    Boundaries: |v| < 25,000 -> M1 ; 25,000 <= |v| <= 250,000 -> M2 ; > 250,000 -> M3.
    """
    mag = abs(float(variance_usd))
    if mag < M1_CEILING:
        return "M1"
    if mag <= M3_FLOOR:
        return "M2"
    return "M3"


# --- FM-002 direction (ratified 2026-11-04, D5) -----------------------------------
# A misstated band fires FM-002 in either direction and scores the same; each firing
# records its direction, so FM-002 counts are reported by direction beside the total.
BAND_ORDER: tuple[str, ...] = ("M1", "M2", "M3")  # M1 < M2 < M3


def band_error(claimed: Iterable[str], true_band: str) -> str:
    """Direction of one FM-002 firing: "understated" when every claimed band sits
    below the true band, "overstated" when every one sits above it, "mixed" when
    the claimed bands straddle it. The true band must not be among the claimed."""
    claimed = list(claimed)
    rank = {b: i for i, b in enumerate(BAND_ORDER)}
    truth = rank[true_band]
    ranks = {rank[b] for b in claimed}
    if not ranks or truth in ranks:
        raise ValueError(f"no misstatement: claimed {sorted(claimed)} against {true_band}")
    if max(ranks) < truth:
        return "understated"
    if min(ranks) > truth:
        return "overstated"
    return "mixed"


# --- Phase 2 placeholder ----------------------------------------------------------
# The 9-cell action matrix (3 asset tiers x 3 materiality bands) and the
# action-vs-policy lookup land here in Week 5, AFTER CPA review of policy.md.
# Not built yet on purpose: the contested stablecoin cell (M1xT1) is a genuine
# industry-practice judgment that gets external review before it is encoded.
