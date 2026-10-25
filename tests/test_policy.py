"""Policy lookup tests — asset tiers and materiality-band boundaries.

The band boundaries are a defensibility-critical detail: an off-by-one at a
threshold is exactly the kind of silent error that lets a material variance be
mis-tiered. These tests pin the boundaries explicitly.
"""

import pytest

from icfr_agent_evals.policy import (
    M1_CEILING,
    M3_FLOOR,
    asset_tier,
    materiality_band,
)


@pytest.mark.parametrize(
    "asset,tier",
    [
        ("USDC", "T1"), ("USDT", "T1"),
        ("BTC", "T2"), ("ETH", "T2"), ("SOL", "T2"),
        ("XRP", "T3"), ("ADA", "T3"),
        ("btc", "T2"), ("  eth  ", "T2"),  # normalization
    ],
)
def test_asset_tier_known(asset, tier):
    assert asset_tier(asset) == tier


def test_asset_tier_unknown_raises():
    with pytest.raises(KeyError):
        asset_tier("DOGE")


@pytest.mark.parametrize(
    "variance,band",
    [
        (0.0, "M1"),
        (24_999.99, "M1"),
        (M1_CEILING, "M2"),          # 25,000 exactly -> M2 (lower bound inclusive)
        (100_000.0, "M2"),
        (M3_FLOOR, "M2"),            # 250,000 exactly -> M2 (upper bound inclusive)
        (250_000.01, "M3"),
        (5_000_000.0, "M3"),
    ],
)
def test_materiality_band_boundaries(variance, band):
    assert materiality_band(variance) == band


@pytest.mark.parametrize(
    "variance,band",
    [
        (-24_999.99, "M1"),
        (-25_000.0, "M2"),
        (-300_000.0, "M3"),
    ],
)
def test_materiality_band_uses_absolute_value(variance, band):
    """Direction does not change severity; magnitude does. A shortfall and a
    surplus of equal size sit in the same band."""
    assert materiality_band(variance) == band
