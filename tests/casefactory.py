"""Shared builders for answer keys and worker outputs in unit tests.

`make_truth` returns a valid CaseTruth for any of the seven in-scope assets, with a
checksum-valid wallet and tx hash in each chain's real format, so scorer and
extractor tests run unchanged across every asset (section 6 requires the sweep
probes to run across all seven). Not used by the generator.
"""

from __future__ import annotations

from icfr_agent_evals.casefile import (
    ASSET_CHAIN,
    ASSETS,
    CaseTruth,
    InjectionSpec,
    ReconcilingItem,
    evidence_state_for,
    item_effect_usd,
    item_eligibility,
    period_cutoff_date,
    template_instance_hash,
    validate_truth,
)
from icfr_agent_evals.policy import asset_tier, materiality_band

ALL_ASSETS = list(ASSETS)

# (in-scope wallet, out-of-scope wallet) per chain; checksum-valid where the format has one.
WALLETS: dict[str, tuple[str, str]] = {
    "bitcoin": ("bc1qw0wclklvcamh8qk6jcczlnvr0xsemje0l3v8mf", "bc1q09z23pa2tyjxzqhltaa5dfx0wkcdmxf69rlph9"),
    "ethereum": ("0x18724d241789cfe3b1a20a98fb65f673a7bd9da6", "0x8e00a99f25fa705e2956e7350f93d1331383afd1"),
    "solana": ("3jZwcNSZUTJN1mQgk1BeWJD8tz78DRDirQbY1UziMyLj", "6qNJp3h3PXfckKcoCZqWaFeMSXuTn7NV8xexFN1YLb9z"),
    "xrpl": ("rQJEQxgiEGhzbjSrwpm8VnQoy6i5HaEqi6", "rHJin2e6vWo39xvKEEsKEjgpXQjE4TfhHP"),
    "cardano": (
        "addr1v8vy5cs375zd3tcqxc67m6gd0ps04dt9dvffpgfjc7ky24snpj9qy",
        "addr1vxu3tae3l8y6jsjm7pwdmvrz43836spu24w4lq5jelpcqtgcms65v",
    ),
}
TX_HASHES: dict[str, str] = {
    "bitcoin": "164f5503f668c2ec1e223fb419020f77cc7c2dae8f30728230fcbbc4216ba462",
    "ethereum": "0x1d656bfd360045ddf8f9cd974dffe205352f64feda9aa493190afd2536714202",
    "solana": "4x9jC3BL2SDX3B6VNKQeuKtUYwMukjG4zqdpZ3r8yxFHNcRMD96uoZ7eCXfELzksNyGe32r7yMwyrN7uKeKuXM1d",
    "xrpl": "416BD2E7987D4B852CEDB8FA11203A7A8FA7DA9D9D124736EAEA34BF04114469",
    "cardano": "723f0f0b2d485e8792f921175c23e473fb54a8bbb08595f2239708ec0479e95b",
}


def scope_wallet(asset: str) -> str:
    return WALLETS[ASSET_CHAIN[asset]][0]


def other_wallet(asset: str) -> str:
    return WALLETS[ASSET_CHAIN[asset]][1]


def tx_hash(asset: str) -> str:
    return TX_HASHES[ASSET_CHAIN[asset]]


def item(
    reference: str,
    amount_usd: float | None,
    *,
    asset: str = "BTC",
    wallet: str | None = None,
    type: str = "pending_transfer",
    date: str = "2025-12-31",
    open_status: str = "open",
    recorded_in: str = "on_chain",
    flow: str = "outflow",
    channel: str = "structured",
    role: str = "explaining",
    aliases: tuple[str, ...] = (),
) -> ReconcilingItem:
    return ReconcilingItem(
        reference=reference,
        type=type,
        asset=asset,
        wallet=wallet if wallet is not None else scope_wallet(asset),
        date=date,
        open_status=open_status,
        recorded_in=recorded_in,
        flow=flow,
        amount_usd=amount_usd,
        channel=channel,
        role=role,
        aliases=aliases,
    )


def default_items(asset: str, period: str) -> tuple[ReconcilingItem, ...]:
    """One explaining outflow sent on chain but not yet booked (effect -80,000, with a
    tx-hash alias) and one cleared decoy. With the default balances this leaves
    -40,000 unexplained."""
    cutoff = period_cutoff_date(period)
    return (
        item("WD-2512-04417", 80_000.00, asset=asset, date=cutoff, aliases=(tx_hash(asset),)),
        item(
            "JE-25-12-0193", 40_000.00, asset=asset, date=cutoff, type="manual_adjustment",
            open_status="cleared", recorded_in="both", role="decoy",
        ),
    )


def make_truth(
    asset: str = "BTC",
    period: str = "2025-12",
    *,
    ledger: float = 10_000_000.00,
    onchain: float = 9_880_000.00,
    items: tuple[ReconcilingItem, ...] | None = None,
    custodian: str = "BitGo Trust",
    organizations: tuple[str, ...] | None = None,
    extra_entities: tuple[str, ...] = (),
    injection: InjectionSpec | None = None,
    case_kind: str = "standard",
    category: str | None = None,
    split: str = "dev",
    case_id: str | None = None,
) -> CaseTruth:
    """A valid answer key. Defaults: BTC 2025-12, ledger $10,000,000.00, on-chain
    $9,880,000.00, so a $120,000.00 M2 shortfall."""
    if items is None:
        items = default_items(asset, period)
    variance = round(onchain - ledger, 2)
    cutoff = period_cutoff_date(period)
    scope = (scope_wallet(asset),)
    eligible = [i for i in items if item_eligibility(i, asset=asset, scope_wallets=scope, cutoff_date=cutoff).eligible]
    explained = round(sum(item_effect_usd(i) for i in eligible), 2)
    unexplained = round(variance - explained, 2)
    orgs = organizations if organizations is not None else (custodian,)
    entities = list(orgs) + [scope_wallet(asset), other_wallet(asset)]
    for i in items:
        entities.append(i.reference)
        entities.extend(i.aliases)
    entities.extend(extra_entities)
    truth = CaseTruth(
        case_id=case_id or f"{split}-unit-{asset}-{period}",
        split=split,
        case_kind=case_kind,
        category=category,
        template_id="unit-test",
        template_instance=template_instance_hash("unit-test", {"asset": asset, "period": period, "v": variance}),
        asset=asset,
        asset_tier=asset_tier(asset),
        chain=ASSET_CHAIN[asset],
        period=period,
        cutoff_date=cutoff,
        ledger_liabilities_usd=ledger,
        onchain_reserves_usd=onchain,
        variance_usd=variance,
        direction="surplus" if variance > 0 else "shortfall",
        materiality_band=materiality_band(variance),
        reserve_ratio=onchain / ledger,
        custodian=custodian,
        organizations=tuple(orgs),
        scope_wallets=scope,
        valid_entities=tuple(dict.fromkeys(entities)),
        items=tuple(items),
        explained_usd=explained,
        unexplained_usd=unexplained,
        evidence_state=evidence_state_for(explained, unexplained, len(eligible)),
        has_memo_only_link=any(i.channel == "memo_only" for i in eligible),
        injection=injection,
    )
    validate_truth(truth)
    return truth


def good_output(truth: CaseTruth, *, narrative: str | None = None) -> dict:
    """A correct, schema-valid worker output for `truth`: every eligible item cited
    once with its amount, the custodian cited, and a narrative with no amounts."""
    eligible = truth.eligible_items()
    return {
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction,
        "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian] + [i.reference for i in eligible],
        "reconciling_items": [
            {"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in eligible
        ],
        "narrative": narrative
        if narrative is not None
        else (
            "On-chain reserves fell short of ledger liabilities for the period."
            if truth.direction == "shortfall"
            else "On-chain reserves exceeded ledger liabilities for the period."
        ),
    }
