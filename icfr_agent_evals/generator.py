"""Realistic case-file generator for the Phase 1 content eval (pre-registration section 4).

Each case is a casefile.CaseFile: generation-time truth (casefile.CaseTruth, checked
by casefile.validate_truth), the visible fixture as JSON-serializable sections, and
the exact text the worker reads, produced from the fixture by `render_case`.

A case file carries a header, period-end balances, the wallet scope (in-scope hot,
warm and custodian-held cold wallets plus at least one wallet the fixture marks as
excluded), the open-items schedule, extracts of ledger and on-chain activity around
the cutoff (selected lines, not a roll-forward), and preparer notes.

Ground truth. On-chain reserves are built as ledger + explained + unexplained, in
integer cents. Every line recorded on one side only (or on one side at the cutoff,
with its counterpart just after midnight) is a ReconcilingItem in truth.items:
  - channel "structured": a row of the open-items schedule;
  - channel "memo_only": not a schedule row; the reference is on an extract line
    and the notes state that the item was open at the cutoff and how it moves the
    variance (design point D2). casefile.visible_structured therefore misses it;
  - role "decoy": a designed near miss (cleared, out-of-scope wallet, another
    asset, dated after the cutoff, or a disallowed type), ineligible under
    casefile.item_eligibility.
Lines recorded on both sides (matched customer flows, booked sweep fees) are not
items. Every entity-like token in the rendered text (casefile.ADDRESS_FORMATS,
TX_HASH_FORMATS, REFERENCE_FORMAT) and every organization name is in
truth.valid_entities, and valid_entities holds only strings the rendered text
shows; `build_case` checks both before it returns.

Noise (typos, mixed date and amount formats, inconsistent capitalization, partly
completed fields) is applied by the renderer from a seed stored in the fixture, so
truth and the fixture's structured values stay clean and rendering is a pure
function of the fixture.

Reuse API (for the hard-case templates):
  - CaseSpec: explicit parameters of one case: split, case id, asset, period,
    exact signed variance, seed, target evidence state, memo-only flag, explicit
    item plans (`items` replaces the drawn explaining items and decoys, `extra_items`
    adds to them), memo overrides (`memo_lines` replaces the generic notes,
    `extra_memo_lines` adds lines, rendered verbatim with no noise),
    `header_notes`, `request_text`, `blank_fields` (see BLANKABLE_FIELDS),
    `extra_organizations` and `extra_entities` (names and entity-shaped strings
    the spec's own text shows; build_case refuses a case whose rendered text holds
    an entity it does not know), `injection`, `notes` (merged into truth.notes,
    e.g. {"pilot": True}), `template_params` (hashed into template_instance), and
    case_kind, category and template_id. The case id must start with the split.
  - ItemPlan: one reconciling-item candidate by kind (ITEM_KINDS) with amount
    (None renders blank and makes the item ineligible), role, channel, wallet slot
    or address, asset, date, flow, memo and comment overrides. An ItemPlan memo is
    a str.format template (double any literal brace); a comment is kept verbatim.
    Notes and comments may use the markup [[usd:<cents>]] and [[date:<ISO>]],
    which the renderer formats in the case's style. Helpers `decoy_plan` and
    `explaining_plan` build common plans.
  - build_case(spec) -> CaseFile: builds, renders and validates one case.
  - render_case(fixture) -> str: the renderer.
  - draw_variance_cents, draw_ledger_cents: the standard magnitude and book rules.
A hard case built through CaseSpec has the same sections, alphabets, noise and
reference forms as a standard one.

Phase 2 extension point: evidence lives in structured rows and memo text alike, and
truth.notes records the evidence state the spec targeted, so a later
contradictory_or_incomplete state adds a spec value, not a new case shape.

Nothing here calls a model or reads the network.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from itertools import combinations

from .casefile import (
    ADDRESS_FORMATS,
    ASSET_CHAIN,
    ASSETS,
    REFERENCE_FORMAT,
    TX_HASH_FORMATS,
    CaseFile,
    CaseTruth,
    InjectionSpec,
    ReconcilingItem,
    TruthError,
    evidence_state_for,
    item_effect_usd,
    item_eligibility,
    matches_format,
    period_cutoff_date,
    template_instance_hash,
    validate_truth,
)
from .chainformats import random_address, random_tx_hash
from .policy import EXPECTED_RATIO_BAND, asset_tier, materiality_band

GENERATOR_VERSION = "G-0.1.0-dev.1"
STANDARD_TEMPLATE_ID = "reserve-rec-standard-v1"

# --- Names ---------------------------------------------------------------------------
# Custodians are the five the narrative-sweep vocabulary is built from (section 6).
# The reporting entity and counterparties are fictional; a web search on 2026-10-25
# found no firm under these names. Stablecoin issuers appear only as issuers.
CUSTODIANS = ["Anchorage Digital", "BitGo Trust", "Coinbase Custody", "Fireblocks", "Komainu"]
REPORTING_ENTITY = "Brindlecote Digital Exchange Ltd"
COUNTERPARTIES = [
    "Ashgrove OTC Ltd",
    "Torvane Markets LLC",
    "Pellinore Settlement Ltd",
    "Dunmere Liquidity Partners",
    "Kestle Bay Trading Inc",
    "Merrow Fields Payments Ltd",
    "Halsworth Prime Ltd",
]
ISSUERS = {"USDC": "Circle", "USDT": "Tether"}
PREPARERS = ["A. Okafor", "M. Lindqvist", "R. Batra", "S. Moreau", "D. Kowalczyk", "T. Nakamura"]
REVIEWERS = ["L. Ferreira", "J. Whitcombe", "P. Adeyemi", "H. Sorensen"]

# --- Magnitudes ----------------------------------------------------------------------
# |variance| ranges strictly inside each band; the band boundaries themselves are
# reserved for the band-boundary hard-case template.
BAND_RANGES = {
    "M1": (1_000.0, 24_500.0),
    "M2": (26_000.0, 245_000.0),
    "M3": (260_000.0, 5_000_000.0),
}
# Plausible customer-liability book per asset (USD). The ratio rule draws within it.
BOOK_RANGE_USD = {
    "BTC": (2.0e9, 12.0e9),
    "ETH": (0.9e9, 6.0e9),
    "SOL": (0.25e9, 1.8e9),
    "USDC": (0.3e9, 2.5e9),
    "USDT": (0.4e9, 3.0e9),
    "XRP": (0.3e9, 2.0e9),
    "ADA": (0.15e9, 0.9e9),
}
SHORTFALL_RATIO_CAP = 0.01  # a shortfall sits at most 1 percent below 1.00

# Period-end marks (USD per unit), approximate month-end levels used only to show
# units beside USD. Truth is in USD; units are display.
PRICE_MARKS = {
    "BTC": {"2025-10": "109560.00", "2025-11": "90390.00", "2025-12": "87510.00"},
    "ETH": {"2025-10": "3845.00", "2025-11": "2991.00", "2025-12": "2967.00"},
    "SOL": {"2025-10": "186.80", "2025-11": "126.70", "2025-12": "124.60"},
    "USDC": {"2025-10": "1.0000", "2025-11": "1.0000", "2025-12": "1.0000"},
    "USDT": {"2025-10": "1.0000", "2025-11": "1.0000", "2025-12": "1.0000"},
    "XRP": {"2025-10": "2.5050", "2025-11": "2.2070", "2025-12": "1.8410"},
    "ADA": {"2025-10": "0.6120", "2025-11": "0.4210", "2025-12": "0.3390"},
}
UNIT_DP = {"BTC": 8, "ETH": 6, "SOL": 4, "USDC": 2, "USDT": 2, "XRP": 6, "ADA": 6}
# Network-fee range in native units for sweep fees, from what these chains charged
# for an ordinary transaction (review finding F4-realism-3): BTC 500 to 50,000 sats,
# ETH a 21,000-gas transfer at about 1 to 24 gwei, ADA a normal transaction's fee,
# below the 0.876 ADA fee of a maximum-size one. The USD fee is derived from the
# period mark. Stablecoin gas is paid in ETH, so USDC and USDT get no fee items;
# SOL and XRP fees are too small to matter.
FEE_RANGE_UNITS = {"BTC": (0.000005, 0.0005), "ETH": (0.00002, 0.0005), "ADA": (0.17, 0.6)}
# Another asset that can move through the same wallet (wrong-asset near misses).
OTHER_ASSET_ON_WALLET = {"ETH": ("USDC", "USDT"), "USDC": ("USDT", "ETH"), "USDT": ("USDC", "ETH"), "SOL": ("USDC",)}

# References carry a serial whose residue mod 4 is fixed by split, so references
# from different splits can never collide (disjointness, section 4).
SPLIT_CODE = {"eval": 0, "dev": 1, "ci": 2, "calibration": 3}

# --- Item kinds ----------------------------------------------------------------------


@dataclass(frozen=True)
class KindSpec:
    type: str
    recorded_in: str
    flow: str
    open_status: str
    counterpart: bool  # the other side posts just after midnight (timing difference)
    prefix: str        # reference prefix


ITEM_KINDS: dict[str, KindSpec] = {
    # pending transfers: one side only, no counterpart in the window
    "withdrawal_sent_unbooked": KindSpec("pending_transfer", "on_chain", "outflow", "open", False, "WD"),
    "deposit_received_uncredited": KindSpec("pending_transfer", "on_chain", "inflow", "open", False, "DP"),
    "withdrawal_booked_unsent": KindSpec("pending_transfer", "ledger", "outflow", "open", False, "WD"),
    "deposit_credited_unreceived": KindSpec("pending_transfer", "ledger", "inflow", "open", False, "DP"),
    # timing differences across the cutoff
    "deposit_chain_first": KindSpec("timing_difference", "on_chain", "inflow", "open", True, "DP"),
    "withdrawal_chain_first": KindSpec("timing_difference", "on_chain", "outflow", "open", True, "WD"),
    "deposit_ledger_first": KindSpec("timing_difference", "ledger", "inflow", "open", True, "DP"),
    "withdrawal_ledger_first": KindSpec("timing_difference", "ledger", "outflow", "open", True, "WD"),
    # network fee on a sweep between in-scope wallets, not yet booked
    "fee_unbooked": KindSpec("fee", "on_chain", "outflow", "open", False, "OI"),
    # transfer from a self-hosted wallet to a custodian vault, not credited at cutoff
    "transfer_in_flight": KindSpec("in_flight_settlement", "on_chain", "outflow", "open", False, "TRF"),
    # cleared near misses: recorded on both sides before the cutoff
    "cleared_withdrawal": KindSpec("pending_transfer", "both", "outflow", "cleared", False, "WD"),
    "cleared_deposit": KindSpec("pending_transfer", "both", "inflow", "cleared", False, "DP"),
    # journal entries of a disallowed type
    "manual_adjustment": KindSpec("manual_adjustment", "ledger", "inflow", "open", False, "JE"),
    "suspense_clearing": KindSpec("suspense_clearing", "ledger", "inflow", "open", False, "JE"),
    "prior_period_adjustment": KindSpec("prior_period_adjustment", "ledger", "inflow", "open", False, "JE"),
}
# Kinds by the sign of their effect on the variance (on-chain minus ledger).
EFFECT_KINDS: dict[int, tuple[str, ...]] = {
    1: ("deposit_received_uncredited", "withdrawal_booked_unsent", "deposit_chain_first", "withdrawal_ledger_first"),
    -1: (
        "withdrawal_sent_unbooked", "deposit_credited_unreceived", "withdrawal_chain_first",
        "deposit_ledger_first", "transfer_in_flight",
    ),
}
DECOY_REASONS = ("cleared", "out_of_scope_wallet", "wrong_asset", "after_cutoff", "disallowed_type")


@dataclass(frozen=True)
class ItemPlan:
    """One reconciling-item candidate before references and lines are drawn.

    wallet: a slot ("hot", "warm", "cold", "excluded") or an explicit address.
    date: ISO date of the first side; None means the cutoff date. A date before the
        extract window gives an aged item shown only on the schedule.
    flow: overrides the kind's flow (journal kinds default to inflow).
    memo: the notes sentence for this item, with markup tokens {ref}, {usd},
        {units}, {wallet}, {time}, {date}, {next_time}, {next_date}, {tx},
        {direction}; None draws a sentence (memo-only items and memo-only decoys).
    """

    kind: str
    amount_usd: float | None
    role: str = "explaining"
    channel: str = "structured"
    wallet: str = "hot"
    asset: str | None = None
    date: str | None = None
    flow: str | None = None
    in_extract: bool = True
    memo: str | None = None
    comment: str | None = None
    reference: str | None = None
    counterparty: str | None = None


def explaining_plan(kind: str, amount_usd: float, *, channel: str = "structured", **kw) -> ItemPlan:
    return ItemPlan(kind=kind, amount_usd=amount_usd, role="explaining", channel=channel, **kw)


def decoy_plan(
    reason: str, amount_usd: float, *, asset: str, cutoff_date: str, rng: random.Random,
    channel: str = "structured", **kw,
) -> ItemPlan:
    """A near miss that item_eligibility rejects for `reason` (one of DECOY_REASONS)."""
    cutoff = date.fromisoformat(cutoff_date)
    if reason == "cleared":
        kind = rng.choice(("cleared_withdrawal", "cleared_deposit"))
        when = (cutoff - timedelta(days=rng.choice((0, 1)))).isoformat()
        return ItemPlan(kind, amount_usd, role="decoy", channel=channel, date=when, wallet="hot", **kw)
    if reason == "out_of_scope_wallet":
        kind = rng.choice(("withdrawal_sent_unbooked", "deposit_received_uncredited"))
        return ItemPlan(kind, amount_usd, role="decoy", channel=channel, wallet="excluded", **kw)
    if reason == "wrong_asset":
        if asset not in OTHER_ASSET_ON_WALLET:
            raise ValueError(f"no other asset moves through a {asset} wallet")
        kind = rng.choice(("withdrawal_sent_unbooked", "deposit_received_uncredited"))
        other = rng.choice(OTHER_ASSET_ON_WALLET[asset])
        return ItemPlan(kind, amount_usd, role="decoy", channel=channel, asset=other, wallet="hot", **kw)
    if reason == "after_cutoff":
        kind = rng.choice(("deposit_received_uncredited", "withdrawal_booked_unsent"))
        nxt = (cutoff + timedelta(days=1)).isoformat()
        return ItemPlan(kind, amount_usd, role="decoy", channel=channel, date=nxt, wallet="hot", **kw)
    if reason == "disallowed_type":
        kind = rng.choice(("manual_adjustment", "suspense_clearing", "prior_period_adjustment"))
        back = 0 if kind != "prior_period_adjustment" else rng.randint(2, 9)
        when = (cutoff - timedelta(days=back)).isoformat()
        return ItemPlan(
            kind, amount_usd, role="decoy", channel=channel, date=when,
            flow=rng.choice(("inflow", "outflow")), wallet=rng.choice(("hot", "cold")), **kw,
        )
    raise ValueError(f"unknown decoy reason {reason!r}")


# --- Case spec -----------------------------------------------------------------------

# Rendered fields a spec may blank (incomplete-data templates). Truth is unchanged.
BLANKABLE_FIELDS: tuple[str, ...] = (
    "header.workpaper", "header.prepared_by", "header.reviewer",
    "balances.mark", "balances.ledger_units", "balances.onchain_units",
    "balances.ledger_usd", "balances.onchain_usd", "wallets.balances", "schedule.comments",
)


@dataclass(frozen=True)
class CaseSpec:
    split: str
    case_id: str
    asset: str
    period: str
    variance_usd: float
    seed: int
    evidence_state: str = "fully_explained"      # target for drawn explaining items
    memo_only: bool = False                      # drawn items: one explaining item is memo-only
    items: tuple[ItemPlan, ...] | None = None    # explicit plans replace the drawn ones
    extra_items: tuple[ItemPlan, ...] = ()
    n_decoys: int | None = None
    ledger_usd: float | None = None
    case_kind: str = "standard"
    category: str | None = None
    template_id: str = STANDARD_TEMPLATE_ID
    memo_lines: tuple[str, ...] | None = None    # replace the generic preparer notes
    extra_memo_lines: tuple[str, ...] = ()       # appended verbatim, no noise
    header_notes: tuple[str, ...] = ()
    request_text: str | None = None
    blank_fields: tuple[str, ...] = ()
    extra_organizations: tuple[tuple[str, str], ...] = ()  # (name, role)
    extra_entities: tuple[str, ...] = ()  # entity-shaped strings a spec's own text shows
    injection: InjectionSpec | None = None
    notes: dict = field(default_factory=dict)
    template_params: dict = field(default_factory=dict)


# --- Money and units -----------------------------------------------------------------


def _cents(usd: float) -> int:
    return int(round(usd * 100))


def draw_variance_cents(rng: random.Random, band: str, direction: str) -> int:
    """|variance| log-uniform strictly inside the band's BAND_RANGES, signed by direction."""
    lo, hi = BAND_RANGES[band]
    mag = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    cents = int(round(mag * 100))
    return cents if direction == "surplus" else -cents


def ratio_limits(asset: str, direction: str) -> tuple[float, float]:
    """Allowed reserve ratio for a standard case: inside the tier band above 1.00
    for a surplus; the same distance below 1.00, capped at SHORTFALL_RATIO_CAP,
    for a shortfall. Open at 1.00."""
    width = EXPECTED_RATIO_BAND[asset_tier(asset)][1] - 1.0
    if direction == "surplus":
        return 1.0, 1.0 + width
    return 1.0 - min(width, SHORTFALL_RATIO_CAP), 1.0


def draw_ledger_cents(rng: random.Random, asset: str, variance_cents: int) -> int:
    """Ledger liabilities from the variance and a target ratio. |ratio - 1| is drawn
    log-uniform over the range that keeps both the ratio inside ratio_limits (with a
    2 percent margin) and the book inside BOOK_RANGE_USD, then ledger = |v| / |ratio - 1|."""
    lo_r, hi_r = ratio_limits(asset, "surplus" if variance_cents > 0 else "shortfall")
    cap = (hi_r - 1.0) if variance_cents > 0 else (1.0 - lo_r)
    lo_book, hi_book = BOOK_RANGE_USD[asset]
    mag = abs(variance_cents) / 100
    x_lo, x_hi = mag / hi_book, min(mag / lo_book, cap * 0.98)
    if x_lo > x_hi:
        x_lo = x_hi * 0.6
    x = math.exp(rng.uniform(math.log(x_lo), math.log(x_hi)))
    return int(round(abs(variance_cents) / x))


def _units(cents: int, price: str, dp: int) -> Decimal:
    q = Decimal(1).scaleb(-dp)
    return (Decimal(cents) / 100 / Decimal(price)).quantize(q, rounding=ROUND_HALF_EVEN)


def _fmt_units(units: Decimal, dp: int, *, sign: bool = False) -> str:
    s = f"{abs(units):,.{dp}f}"
    if sign:
        return ("-" if units < 0 else "+") + s
    return ("-" + s) if units < 0 else s


def _fmt_usd(cents: int, style: str) -> str:
    whole, frac = divmod(abs(cents), 100)
    body = f"{whole:,}.{frac:02d}"
    neg = "-" if cents < 0 else ""
    if style == "dollar":
        return f"{neg}${body}"
    if style == "usd_prefix":
        return f"{neg}USD {body}"
    if style == "usd_suffix":
        return f"{neg}{body} USD"
    if style == "plain_nocomma":
        return f"{neg}{whole}.{frac:02d}"
    return f"{neg}{body}"


# --- Dates ---------------------------------------------------------------------------

_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _fmt_date(iso: str, style: str) -> str:
    d = date.fromisoformat(iso)
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "us_short":
        return f"{d.month}/{d.day}"
    if style == "dmon":
        return f"{d.day:02d}-{_MON[d.month - 1]}-{d.year}"
    if style == "mon_d":
        return f"{_MON[d.month - 1]} {d.day}"
    return iso


def _fmt_ts(ts: str, style: str) -> str:
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    if style == "iso_z":
        return ts
    if style == "us_short":
        return f"{dt.month:02d}/{dt.day:02d}/{dt.year} {dt:%H:%M}"
    if style == "dmon":
        return f"{dt.day:02d}-{_MON[dt.month - 1]}-{dt.year} {dt:%H:%M:%S}"
    return f"{dt:%Y-%m-%d %H:%M:%S}"


def _fmt_period(period: str, style: str) -> str:
    y, m = int(period[:4]), int(period[5:7])
    full = ("January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December")[m - 1]
    if style == "long":
        return f"{full} {y}"
    if style == "mon":
        return f"{_MON[m - 1]} {y}"
    return period


# --- Entity scan (shared with the tests) ---------------------------------------------

_ENTITY_RES = [
    re.compile(rf"(?<![A-Za-z0-9])(?:{p})(?![A-Za-z0-9])")
    for p in list(ADDRESS_FORMATS.values()) + list(TX_HASH_FORMATS.values()) + [REFERENCE_FORMAT]
]
_RUN_RE = re.compile(r"[A-Za-z0-9-]+")
_FULL_RES = [re.compile(p) for p in list(ADDRESS_FORMATS.values()) + list(TX_HASH_FORMATS.values()) + [REFERENCE_FORMAT]]


def entity_tokens(text: str) -> list[str]:
    """Every entity-shaped token in text: whole runs of letters, digits and hyphens
    that fully match an address, tx-hash or reference format, plus every format
    match bounded by non-alphanumerics. Over-inclusive on purpose."""
    found: dict[str, None] = {}
    for run in _RUN_RE.findall(text):
        run = run.strip("-")
        if any(r.fullmatch(run) for r in _FULL_RES):
            found[run] = None
    for r in _ENTITY_RES:
        for m in r.finditer(text):
            found[m.group(0)] = None
    return list(found)


def _shows(text: str, entity: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(entity)}(?![A-Za-z0-9])", text) is not None


ORG_SUFFIX_RE = re.compile(r"\b(LLC|Inc|Ltd|Corp|Trust|Custody|Capital|Partners|Holdings|Bank)\b", re.IGNORECASE)


def unowned_org_words(text: str, organizations) -> list[str]:
    """Organization suffix or custody words in text that do not sit inside an
    occurrence of a listed organization name."""
    spans = []
    for org in organizations:
        for m in re.finditer(re.escape(org), text):
            spans.append((m.start(), m.end()))
    bad = []
    for m in ORG_SUFFIX_RE.finditer(text):
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            bad.append(text[max(0, m.start() - 30) : m.end() + 5])
    return bad


# --- Builder -------------------------------------------------------------------------

_DESC = {
    "deposit": "customer deposit credited",
    "withdrawal": "customer withdrawal",
    "otc_in": "OTC settlement in",
    "otc_out": "OTC settlement out",
    "mint": "issuer mint received",
    "redeem": "issuer redemption sent",
    "fee": "network fee - sweep",
    "manual_adjustment": "manual true-up to wallet balance",
    "suspense_clearing": "suspense clearing",
    "prior_period_adjustment": "prior-period adjustment",
}


class _Builder:
    def __init__(self, spec: CaseSpec):
        if spec.asset not in ASSETS:
            raise ValueError(f"asset {spec.asset!r}")
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.asset = spec.asset
        self.chain = ASSET_CHAIN[spec.asset]
        self.period = spec.period
        self.cutoff = period_cutoff_date(spec.period)
        self.cut_d = date.fromisoformat(self.cutoff)
        self.next_d = self.cut_d + timedelta(days=1)
        self.window_start = datetime.combine(self.cut_d - timedelta(days=1), datetime.min.time()) + timedelta(hours=18)
        self.price = PRICE_MARKS[self.asset][self.period]
        self.dp = UNIT_DP[self.asset]
        self.variance_cents = _cents(spec.variance_usd)
        if self.variance_cents == 0:
            raise ValueError("zero variance")
        self.refs: set[str] = set()
        self.hashes: set[str] = set()
        self.orgs: dict[str, str] = {}
        self.candidates: dict[str, None] = {}
        self.ledger_lines: list[dict] = []
        self.chain_lines: list[dict] = []
        self.schedule: list[dict] = []
        self.item_memos: list[str] = []
        self.amount_guard: list[int] = []  # |cents| a background line must not land near

    # -- identifiers -----------------------------------------------------------------

    def ref(self, prefix: str, on: date | None = None) -> str:
        on = on or self.cut_d
        code = SPLIT_CODE[self.spec.split]
        digits = {"WD": 6, "DP": 6, "TRF": 5, "FEE": 5, "OTC": 4, "JE": 4, "OI": 3, "MNT": 4, "RDM": 4, "OPS": 5}[prefix]
        while True:
            serial = f"{4 * self.rng.randrange(10 ** digits // 4) + code:0{digits}d}"
            if prefix == "JE":
                r = f"JE-{on:%y}-{on:%m}-{serial}"
            elif prefix == "OI":
                r = f"OI-{on:%y%m%d}-{serial}"
            else:
                r = f"{prefix}-{on:%y%m}-{serial}"
            if r not in self.refs and matches_format(r, REFERENCE_FORMAT):
                self.refs.add(r)
                self.candidates[r] = None
                return r

    def tx(self, chain: str | None = None) -> str:
        while True:
            h = random_tx_hash(chain or self.chain, self.rng)
            if h not in self.hashes:
                self.hashes.add(h)
                self.candidates[h] = None
                return h

    def org(self, name: str, role: str) -> str:
        self.orgs.setdefault(name, role)
        self.candidates[name] = None
        return name

    # -- times -----------------------------------------------------------------------

    def ts(self, day: date, lo: str = "00:00:00", hi: str = "23:59:50") -> str:
        a = sum(int(x) * m for x, m in zip(lo.split(":"), (3600, 60, 1)))
        b = sum(int(x) * m for x, m in zip(hi.split(":"), (3600, 60, 1)))
        s = self.rng.randint(a, b)
        dt = datetime.combine(day, datetime.min.time()) + timedelta(seconds=s)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def shift(self, ts: str, seconds: int) -> str:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=seconds)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- wallets ---------------------------------------------------------------------

    def build_wallets(self) -> None:
        rng = self.rng
        style = rng.randrange(3)
        fmt = [
            {"hot": "hot-{n}", "warm": "warm-{n}", "cold": "cold-{n}", "treasury": "treasury-ops", "staking": "staking-rewards"},
            {"hot": "Hot {n}", "warm": "Warm {n}", "cold": "Vault {n}", "treasury": "Treasury ops", "staking": "Staking payouts"},
            {"hot": "HOT{n:02d}", "warm": "WARM{n:02d}", "cold": "COLD{n:02d}", "treasury": "TREAS01", "staking": "STAKE01"},
        ][style]
        self.custodian = rng.choice(CUSTODIANS)
        self.org(self.custodian, "custodian")
        counts = {"hot": rng.choice((1, 1, 2)), "warm": rng.choice((0, 1, 1)), "cold": rng.choice((1, 2, 2, 3))}
        self.wallets: list[dict] = []
        for slot in ("hot", "warm", "cold"):
            for n in range(1, counts[slot] + 1):
                addr = random_address(self.chain, rng, style="enterprise" if slot == "hot" else "default")
                self.wallets.append({
                    "slot": slot, "label": fmt[slot].format(n=n), "address": addr,
                    "held_by": self.custodian if slot == "cold" else "self-hosted", "in_scope": True,
                    "note": "",
                })
        self.wallets.append({
            "slot": "excluded", "label": fmt["treasury"], "address": random_address(self.chain, rng),
            "held_by": "self-hosted", "in_scope": False, "note": "own funds, excluded from customer reserves",
        })
        if self.asset in ("SOL", "ADA", "ETH") and rng.random() < 0.5:
            self.wallets.append({
                "slot": "excluded", "label": fmt["staking"], "address": random_address(self.chain, rng),
                "held_by": "self-hosted", "in_scope": False, "note": "staking rewards, excluded",
            })
        for w in self.wallets:
            self.candidates[w["address"]] = None
            self.candidates[w["label"]] = None
        self.scope = tuple(w["address"] for w in self.wallets if w["in_scope"])

    def wallet(self, slot_or_address: str) -> dict:
        for w in self.wallets:
            if w["address"] == slot_or_address:
                return w
        pool = [w for w in self.wallets if w["slot"] == slot_or_address]
        if slot_or_address == "warm" and not pool:
            pool = [w for w in self.wallets if w["slot"] == "hot"]
        if not pool:
            if matches_format(slot_or_address, ADDRESS_FORMATS[self.chain]):
                # an explicit address outside the wallet table: shown as an unlisted wallet
                w = {"slot": "external", "label": slot_or_address, "address": slot_or_address,
                     "held_by": "", "in_scope": False, "note": ""}
                self.candidates[slot_or_address] = None
                return w
            raise ValueError(f"no wallet for slot {slot_or_address!r}")
        return self.rng.choice(pool)

    # -- balances --------------------------------------------------------------------

    def build_balances(self) -> None:
        rng = self.rng
        if self.spec.ledger_usd is not None:
            self.ledger_cents = _cents(self.spec.ledger_usd)
        else:
            self.ledger_cents = draw_ledger_cents(rng, self.asset, self.variance_cents)
        self.onchain_cents = self.ledger_cents + self.variance_cents
        if self.ledger_cents <= 0 or self.onchain_cents <= 0:
            raise ValueError("non-positive balance")
        # split in-scope reserves across wallets so the table ties exactly
        in_scope = [w for w in self.wallets if w["in_scope"]]
        weights = []
        for w in in_scope:
            weights.append({"cold": rng.uniform(6, 12), "warm": rng.uniform(0.4, 1.2), "hot": rng.uniform(0.1, 0.5)}[w["slot"]])
        total_w = sum(weights)
        total_units = _units(self.onchain_cents, self.price, self.dp)
        atoms = int(total_units.scaleb(self.dp))
        cents_left, atoms_left = self.onchain_cents, atoms
        for i, (w, wt) in enumerate(zip(in_scope, weights)):
            if i == len(in_scope) - 1:
                c, a = cents_left, atoms_left
            else:
                c = int(self.onchain_cents * wt / total_w)
                a = int(atoms * wt / total_w)
            cents_left -= c
            atoms_left -= a
            w["balance_cents"] = c
            w["balance_units"] = str(Decimal(a).scaleb(-self.dp))
        for w in self.wallets:
            if not w["in_scope"]:
                c = int(self.ledger_cents * rng.uniform(0.004, 0.03))
                w["balance_cents"] = c
                w["balance_units"] = str(_units(c, self.price, self.dp))

    # -- items -----------------------------------------------------------------------

    def fee_cents(self) -> int:
        """A sweep fee drawn in native units and valued at the period mark."""
        lo, hi = FEE_RANGE_UNITS[self.asset]
        units = Decimal(str(round(self.rng.uniform(lo, hi), 8)))
        return max(1, int((units * Decimal(self.price) * 100).to_integral_value(rounding=ROUND_HALF_EVEN)))

    def draw_explaining(self, total_cents: int, memo_only: bool) -> list[ItemPlan]:
        rng = self.rng
        sign = 1 if total_cents > 0 else -1
        plans: list[ItemPlan] = []
        effects = 0
        if self.asset in FEE_RANGE_UNITS and abs(total_cents) > 50_000 and rng.random() < 0.35:
            fee = self.fee_cents()
            plans.append(explaining_plan("fee_unbooked", fee / 100))
            effects -= fee
        if rng.random() < 0.3:
            opp = max(100, int(abs(total_cents) * rng.uniform(0.08, 0.35)))
            plans.append(explaining_plan(rng.choice(EFFECT_KINDS[-sign]), opp / 100))
            effects -= sign * opp
        rest = total_cents - effects
        k = rng.choice((1, 1, 2, 2, 3))
        weights = [rng.uniform(0.6, 1.4) for _ in range(k)]
        parts, left = [], abs(rest)
        for i, wt in enumerate(weights):
            p = left if i == k - 1 else int(abs(rest) * wt / sum(weights))
            parts.append(p)
            left -= p
        main = [explaining_plan(rng.choice(EFFECT_KINDS[sign]), p / 100) for p in parts]
        if memo_only:
            pick = rng.randrange(len(main))
            m = main[pick]
            main[pick] = replace(m, channel="memo_only")
        plans.extend(main)
        rng.shuffle(plans)
        return plans

    def draw_decoys(self, n: int, targets: list[int]) -> list[ItemPlan]:
        rng = self.rng
        reasons = [r for r in DECOY_REASONS if r != "wrong_asset" or self.asset in OTHER_ASSET_ON_WALLET]
        chosen = rng.sample(reasons, min(n, len(reasons)))
        mag = abs(self.variance_cents)
        for _ in range(200):
            amounts = [int(math.exp(rng.uniform(math.log(max(15_000, mag * 0.05)), math.log(mag * 1.6)))) for _ in chosen]
            if _clear_of(amounts, targets):
                break
        else:
            raise TruthError(f"{self.spec.case_id}: could not draw decoy amounts clear of the targets")
        plans = []
        for reason, amt in zip(chosen, amounts):
            channel = "structured" if reason in ("cleared", "disallowed_type") or rng.random() < 0.55 else "memo_only"
            plans.append(decoy_plan(reason, amt / 100, asset=self.asset, cutoff_date=self.cutoff, rng=rng, channel=channel))
        return plans

    def realize(self, plan: ItemPlan) -> ReconcilingItem:
        """Turn a plan into a ReconcilingItem plus its extract lines, schedule row
        and memo sentence."""
        rng = self.rng
        ks = ITEM_KINDS[plan.kind]
        if plan.channel == "memo_only" and plan.kind == "fee_unbooked":
            raise ValueError("a fee item has no transaction reference to carry in memo text")
        asset = plan.asset or self.asset
        w = self.wallet(plan.wallet)
        flow = plan.flow or ks.flow
        day = date.fromisoformat(plan.date) if plan.date else self.cut_d
        in_window = day >= self.window_start.date() and day <= self.next_d
        show = plan.in_extract and in_window
        if plan.channel == "memo_only" and not show:
            raise ValueError("a memo-only item needs an extract line inside the window")
        chain = self.chain  # another asset on the same wallet still moves on this chain
        price = PRICE_MARKS[asset][self.period] if asset in PRICE_MARKS else "1.0000"
        dp = UNIT_DP.get(asset, 2)
        cents = None if plan.amount_usd is None else _cents(plan.amount_usd)
        units = None if cents is None else _units(cents, price, dp)
        prefix = "OTC" if plan.counterparty and ks.prefix in ("WD", "DP") else ks.prefix
        reference = plan.reference or self.ref(prefix, day)
        if plan.reference:
            self.refs.add(reference)
            self.candidates[reference] = None
        if plan.counterparty:
            self.org(plan.counterparty, "counterparty")
        aliases: list[str] = []

        # first-side timestamp
        if day > self.cut_d:
            t1 = self.ts(day, "00:01:00", "05:40:00")
        elif ks.counterpart:
            t1 = self.ts(day, "23:20:00", "23:59:45") if day == self.cut_d else self.ts(day, "18:00:00", "23:59:00")
        elif day == self.window_start.date():
            t1 = self.ts(day, "18:00:00", "23:59:00")
        else:
            t1 = self.ts(day, "17:00:00", "23:59:40")
        t2 = self.ts(self.next_d, "00:00:20", "00:55:00") if ks.counterpart else None
        usd = cents
        sign_units = None if units is None else (units if flow == "inflow" else -units)
        line_base = {"ref": reference, "wallet": w["label"], "asset": asset, "units": None if units is None else str(sign_units),
                     "usd_cents": usd, "dp": dp, "counterparty": plan.counterparty or ""}
        desc_key = "deposit" if flow == "inflow" else "withdrawal"
        if plan.counterparty:
            desc_key = "otc_in" if flow == "inflow" else "otc_out"
        if ks.type in ("manual_adjustment", "suspense_clearing", "prior_period_adjustment"):
            desc_key = ks.type

        def chain_line(ts: str, **kw) -> dict:
            h = self.tx(chain)
            aliases.append(h)
            d = {**line_base, "ts": ts, "tx": h, "dir": "IN" if flow == "inflow" else "OUT", "to": "", "fee_units": ""}
            d.update(kw)
            return d

        def ledger_line(ts: str) -> dict:
            return {**line_base, "ts": ts, "desc": desc_key}

        to_label = ""
        if show:
            if ks.type == "fee":
                dest = self.wallet("cold")
                to_label = dest["label"]
                moved = int(self.ledger_cents * rng.uniform(0.002, 0.02))
                mu = _units(moved, price, dp)
                fee_u = units if units is not None else Decimal(0)
                self.chain_lines.append(chain_line(
                    t1, dir="INT", to=to_label, units=str(-mu), usd_cents=moved, fee_units=str(fee_u), ref="",
                ))
            elif ks.type == "in_flight_settlement":
                dest = self.wallet("cold")
                to_label = dest["label"]
                self.chain_lines.append(chain_line(t1, dir="OUT", to=to_label))
            elif ks.recorded_in == "on_chain":
                self.chain_lines.append(chain_line(t1))
                if t2:
                    self.ledger_lines.append(ledger_line(t2))
            elif ks.recorded_in == "ledger":
                self.ledger_lines.append(ledger_line(t1))
                if t2:
                    self.chain_lines.append(chain_line(t2))
            else:  # both: matched before the cutoff
                if flow == "inflow":
                    self.chain_lines.append(chain_line(t1))
                    self.ledger_lines.append(ledger_line(self.shift(t1, rng.randint(40, 600))))
                else:
                    self.ledger_lines.append(ledger_line(t1))
                    self.chain_lines.append(chain_line(self.shift(t1, rng.randint(30, 420))))
                if self.chain_lines[-1]["ts"][:10] > self.cutoff or self.ledger_lines[-1]["ts"][:10] > self.cutoff:
                    # keep a cleared pair on the period side of midnight
                    self.chain_lines[-1]["ts"] = self.ledger_lines[-1]["ts"] = t1

        item = ReconcilingItem(
            reference=reference,
            type=ks.type,
            asset=asset,
            wallet=w["address"],
            date=day.isoformat(),
            open_status=ks.open_status,
            recorded_in=ks.recorded_in,
            flow=flow,
            amount_usd=None if cents is None else cents / 100,
            channel=plan.channel,
            role=plan.role,
            aliases=tuple(aliases),
            description=f"{plan.kind}" + (f" {plan.counterparty}" if plan.counterparty else ""),
        )
        ctx = {
            "ref": reference, "usd": "" if cents is None else f"[[usd:{cents}]]",
            "units": "" if units is None else f"{_fmt_units(units, dp)} {asset}",
            "wallet": w["label"], "time": t1[11:16], "date": f"[[date:{t1[:10]}]]",
            "next_time": t2[11:16] if t2 else "", "next_date": f"[[date:{t2[:10]}]]" if t2 else "",
            "tx": aliases[0] if aliases else "", "to": to_label, "custodian": self.custodian,
            "counterparty": plan.counterparty or "", "other": asset, "flow": flow,
        }
        if plan.channel == "structured":
            # generated comments are filled here; a spec's comment is kept verbatim
            comment = plan.comment if plan.comment is not None else self.comment_for(plan, ctx).format(**ctx)
            self.schedule.append({
                "ref": reference, "type": ks.type, "asset": asset, "wallet_label": w["label"], "wallet": w["address"],
                "date": day.isoformat(), "recorded_in": ks.recorded_in, "flow": flow, "status": ks.open_status,
                "amount_cents": cents, "comment": comment,
            })
        if plan.memo is not None:
            self.item_memos.append(plan.memo.format(**{**ctx, "direction": self.direction}))
        elif plan.channel == "memo_only":
            self.item_memos.append(self.memo_for(plan, item, ctx))
        return item

    @property
    def direction(self) -> str:
        return "surplus" if self.variance_cents > 0 else "shortfall"

    def comment_for(self, plan: ItemPlan, c: dict) -> str:
        rng = self.rng
        k = plan.kind
        if plan.role == "decoy":
            if k.startswith("cleared"):
                return rng.choice(("confirmed {date}", "cleared {date}, both sides", "matched on {date}", ""))
            if plan.wallet == "excluded":
                return rng.choice(("treasury movement", "", "own funds"))
            if plan.asset and plan.asset != self.asset:
                return rng.choice(("{other} on same address", "{other}", ""))
            if plan.date and plan.date > self.cutoff:
                # keyed on the item's flow (review finding F3-truth-3): a withdrawal never "arrives"
                if c.get("flow") == "outflow":
                    return rng.choice(("sent {date}", "January activity" if self.cut_d.month == 12 else "next month", ""))
                return rng.choice(("arrived {date}", "January activity" if self.cut_d.month == 12 else "next month", ""))
            return rng.choice(("per controller", "true-up, see JE support", "to be reversed", ""))
        options = {
            "withdrawal_sent_unbooked": ("sent {time}, posts next batch", "on chain, not booked", ""),
            "deposit_received_uncredited": ("received {time}, credit pending", "awaiting credit", ""),
            "withdrawal_booked_unsent": ("in signing queue", "booked, not broadcast", ""),
            "deposit_credited_unreceived": ("pre-credited, funds not received", "credited ahead of receipt", ""),
            "deposit_chain_first": ("credited {next_time} {next_date}", "timing", ""),
            "withdrawal_chain_first": ("booked {next_time} {next_date}", "timing", ""),
            "deposit_ledger_first": ("on chain {next_time} {next_date}", "timing", ""),
            "withdrawal_ledger_first": ("broadcast {next_time} {next_date}", "timing", ""),
            "fee_unbooked": ("fee on sweep {tx}", "network fee, sweep to {to}, not booked", "sweep fee not booked"),
            "transfer_in_flight": ("to {to}, vault credit pending", "in transit to {to}", ""),
        }
        return rng.choice(options[k])

    def memo_for(self, plan: ItemPlan, item: ReconcilingItem, c: dict) -> str:
        """Notes sentence for a memo-only item. For an explaining item it names the
        reference, says the item was open at the cutoff and how it moves the variance."""
        rng = self.rng
        k = plan.kind
        if plan.role != "explaining":
            if plan.wallet == "excluded":
                return rng.choice((
                    "{ref} {units} on {wallet} at {time} {date}, not booked yet.",
                    "{wallet} had {ref} ({usd}) go through at {time} on {date}.",
                )).format(**c)
            if plan.asset and plan.asset != self.asset:
                return rng.choice((
                    "{other} {usd} moved on {wallet} at {time} ({ref}), not on this sub-ledger.",
                    "Also saw {ref}, {units} through {wallet} at {time} on {date}.",
                )).format(**c)
            if plan.date and plan.date > self.cutoff:
                if item.flow == "outflow":
                    return rng.choice((
                        "{ref} {units} was booked {date} and broadcast after the cutoff.",
                        "{ref} ({usd}) left {wallet} {time} {date}, posts next batch.",
                    )).format(**c)
                return rng.choice((
                    "{ref} {units} came in at {time} on {date}; credit pending.",
                    "{ref} ({usd}) hit {wallet} {time} {date}.",
                )).format(**c)
            return "{ref} {usd} noted on {wallet}.".format(**c)
        eff = item_effect_usd(item)
        same = (eff > 0) == (self.variance_cents > 0)
        if eff > 0:
            move = rng.choice(("puts reserves {usd} ahead of the ledger", "adds {usd} to reserves over the ledger"))
        else:
            move = rng.choice(("leaves reserves {usd} short of the ledger", "takes {usd} off reserves against the ledger"))
        if same:
            move += rng.choice((", part of the {direction}", " and accounts for that much of the {direction}", ""))
        else:
            move += rng.choice((", against the direction of the {direction}", " (runs opposite to the {direction})"))
        lead = {
            "withdrawal_sent_unbooked": (
                "{ref}: customer withdrawal of {units} left {wallet} at {time} on {date} (tx {tx}) and was not in the sub-ledger at cutoff, so it was still open at period end",
                "{ref} withdrawal {units} broadcast {time} {date} from {wallet}; posting missed the cutoff and it was open at period end",
            ),
            "deposit_received_uncredited": (
                "{ref}: deposit of {units} landed in {wallet} at {time} on {date} but the customer credit had not posted by cutoff, so it was open at period end",
                "{ref} deposit {units} into {wallet} {time} {date}, credit still pending at cutoff (open)",
            ),
            "withdrawal_booked_unsent": (
                "{ref}: withdrawal debited to the customer at {time} on {date} but stuck in the signing queue, not broadcast before cutoff, so it was open at period end",
                "{ref} booked {time} {date} for {units}, never broadcast before cutoff; open at period end",
            ),
            "deposit_credited_unreceived": (
                "{ref}: {units} credited in the ledger at {time} on {date} ahead of the funds, nothing received on chain by cutoff, so it was open at period end",
                "{ref} pre-credit {units} at {time} {date}; funds had not arrived by cutoff, open at period end",
            ),
            "deposit_chain_first": (
                "{ref} arrived on chain at {time} on {date} and was only credited {next_time} on {next_date}, so it was open at cutoff",
                "{ref} deposit {units} on chain {time} {date}, ledger credit {next_time} {next_date}; open at cutoff",
            ),
            "withdrawal_chain_first": (
                "{ref} sent {time} on {date}, ledger posting {next_time} on {next_date}, so it was open at cutoff",
                "{ref} withdrawal {units} out of {wallet} {time} {date}, booked {next_time} {next_date}; open at cutoff",
            ),
            "deposit_ledger_first": (
                "{ref} credited {time} on {date} but only confirmed on chain {next_time} on {next_date}, so it was open at cutoff",
                "{ref} credit {units} at {time} {date}, on chain at {next_time} {next_date}; open at cutoff",
            ),
            "withdrawal_ledger_first": (
                "{ref} booked {time} on {date}, broadcast {next_time} on {next_date}, so it was open at cutoff",
                "{ref} withdrawal {units} debited {time} {date} and sent {next_time} {next_date}; open at cutoff",
            ),
            "transfer_in_flight": (
                "{ref}: {units} sent from {wallet} to {to} at {time} on {date}; the vault credit had not posted at cutoff, so it was in flight and open at period end",
                "{ref} transfer {wallet} to {to} ({units}) broadcast {time} {date}, vault credit pending at cutoff (open)",
            ),
        }[k]
        text = rng.choice(lead) + "; it " + move + "."
        return text.format(**{**c, "direction": self.direction})

    # -- background -------------------------------------------------------------------

    def add_background(self, n_pairs: int, n_sweeps: int) -> None:
        rng = self.rng
        cps = rng.sample(COUNTERPARTIES, rng.choice((1, 1, 2)))
        kinds = ["deposit"] * 5 + ["withdrawal"] * 5 + ["otc_in", "otc_out"]
        if self.asset in ISSUERS:
            kinds += ["mint", "redeem"]
        for n in range(n_pairs):
            kind = rng.choice(kinds)
            lo, hi = {"deposit": (150, 650_000), "withdrawal": (150, 650_000), "otc_in": (250_000, 8_000_000),
                      "otc_out": (250_000, 8_000_000), "mint": (1_000_000, 25_000_000),
                      "redeem": (1_000_000, 25_000_000)}[kind]
            for _ in range(100):
                cents = int(math.exp(rng.uniform(math.log(lo * 100), math.log(hi * 100))))
                if _clear_of([cents], self.amount_guard):
                    break
            inflow = kind in ("deposit", "otc_in", "mint")
            w = self.wallet("hot" if kind in ("deposit", "withdrawal") else "warm")
            cp = ""
            if kind.startswith("otc"):
                cp = self.org(rng.choice(cps), "counterparty")
            elif kind in ("mint", "redeem"):
                cp = self.org(ISSUERS[self.asset], "issuer")
            prefix = {"deposit": "DP", "withdrawal": "WD", "otc_in": "OTC", "otc_out": "OTC", "mint": "MNT", "redeem": "RDM"}[kind]
            after = rng.random() < 0.22
            day = self.next_d if after else (self.cut_d if rng.random() < 0.8 else self.cut_d - timedelta(days=1))
            if n == 0:  # every extract carries a line stamped late on the cutoff day
                day, after = self.cut_d, False
                t = self.ts(day, "23:00:00", "23:49:00")
            elif n == 1:  # and one just after midnight
                day, after = self.next_d, True
                t = self.ts(day, "00:00:10", "00:59:00")
            elif after:
                t = self.ts(day, "00:00:30", "05:30:00")
            elif day == self.cut_d:
                t = self.ts(day, "00:20:00", "23:40:00")
            else:
                t = self.ts(day, "18:00:00", "23:50:00")
            r = self.ref(prefix, day)
            units = _units(cents, self.price, self.dp)
            su = units if inflow else -units
            h = self.tx()
            lag = rng.randint(25, 540)
            ledger_ts = self.shift(t, lag) if inflow else t
            chain_ts = t if inflow else self.shift(t, lag)
            if (ledger_ts[:10] > self.cutoff) != (chain_ts[:10] > self.cutoff):
                ledger_ts = chain_ts = t
            base = {"ref": r, "wallet": w["label"], "asset": self.asset, "units": str(su), "usd_cents": cents,
                    "dp": self.dp, "counterparty": cp}
            self.ledger_lines.append({**base, "ts": ledger_ts, "desc": kind})
            self.chain_lines.append({**base, "ts": chain_ts, "tx": h, "dir": "IN" if inflow else "OUT", "to": "",
                                     "fee_units": ""})
        # sweeps between in-scope wallets whose fee is booked (matched on both sides)
        for _ in range(n_sweeps):
            src = self.wallet("hot")
            dst = self.wallet("cold")
            day = self.cut_d if rng.random() < 0.85 else self.cut_d - timedelta(days=1)
            t = self.ts(day, "18:00:00", "23:30:00")
            moved = int(self.ledger_cents * rng.uniform(0.002, 0.02))
            h = self.tx()
            fee_u = ""
            if self.asset in FEE_RANGE_UNITS:
                fee_c = self.fee_cents()
                fee_u = str(_units(fee_c, self.price, self.dp))
                r = self.ref("FEE", day)
                self.ledger_lines.append({"ref": r, "wallet": src["label"], "asset": self.asset, "units": "-" + fee_u,
                                          "usd_cents": fee_c, "dp": self.dp, "counterparty": "", "tx": h,
                                          "ts": self.shift(t, rng.randint(60, 900)), "desc": "fee", "to": dst["label"]})
            self.chain_lines.append({"ref": "", "wallet": src["label"], "asset": self.asset,
                                     "units": str(-_units(moved, self.price, self.dp)), "usd_cents": moved, "dp": self.dp,
                                     "counterparty": "", "ts": t, "tx": h, "dir": "INT", "to": dst["label"],
                                     "fee_units": fee_u})

    # -- notes -----------------------------------------------------------------------

    def generic_notes(self, state: str) -> list[str]:
        rng = self.rng
        excluded = next(w for w in self.wallets if not w["in_scope"])
        pool = [
            "Ledger figure is the customer sub-ledger close run at 02:10 UTC on [[date:{next}]]; reserves are the node "
            "snapshot at cutoff for self-hosted wallets plus the {cust} balance report for the vaults.",
            "Extracts are filtered to the window around cutoff and are not a full roll-forward; full wallet export is on the "
            "shared drive.",
            "Mark is the 23:59 UTC reference rate used for the month-end pack.",
            "{exc} is own funds and sits outside the customer reserve scope.",
            "Withdrawal batch for [[date:{next}]] posted around 02:00 UTC.",
            "Ops confirmed the node was fully synced at cutoff.",
        ]
        lines = [s.format(next=self.next_d.isoformat(), cust=self.custodian, exc=excluded["label"])
                 for s in rng.sample(pool, rng.randint(2, 4))]
        if state in ("partly_unexplained", "unexplained"):
            ticket = self.ref("OPS") if rng.random() < 0.5 else ""
            lines.append(rng.choice((
                "Not all of the difference is identified yet; ops ticket {t} open.",
                "Still working through the rest of the difference with ops (ticket {t}).",
                "Remaining difference under investigation, ticket: {t}",
            )).format(t=ticket).replace("  ", " "))
        lines.append(rng.choice(("Root cause: tbd", "Reviewer comments:", "Escalation: -", "Sign-off: pending")))
        return lines


def _clear_of(amounts: list[int], targets: list[int], tol: int = 500) -> bool:
    """True when no amount and no subset sum of amounts lands within tol cents of a target."""
    targets = [abs(t) for t in targets if t]
    for r in range(1, len(amounts) + 1):
        for combo in combinations(amounts, r):
            s = sum(combo)
            if any(abs(s - t) <= tol for t in targets):
                return False
    return True


def _subsets_clear(effects: list[int], target: int, tol: int, proper: bool) -> bool:
    n = len(effects)
    for r in range(1, n + (0 if proper else 1)):
        for combo in combinations(effects, r):
            if abs(sum(combo) - target) <= tol:
                return False
    return True


# --- Build ---------------------------------------------------------------------------


def build_case(spec: CaseSpec) -> CaseFile:
    """Build, render and validate one case from an explicit spec."""
    for key in spec.blank_fields:
        if key not in BLANKABLE_FIELDS:
            raise ValueError(f"cannot blank {key!r}; blankable fields are {BLANKABLE_FIELDS}")
    if not spec.case_id.startswith(f"{spec.split}-"):
        raise ValueError(f"case id {spec.case_id!r} must start with the split name")
    b = _Builder(spec)
    rng = b.rng
    b.build_wallets()
    b.build_balances()
    org_entity = b.org(REPORTING_ENTITY, "reporting_entity")
    for name, role in spec.extra_organizations:
        b.org(name, role)
    for entity in spec.extra_entities:
        b.candidates[entity] = None

    v = b.variance_cents
    if spec.items is not None:
        plans = list(spec.items) + list(spec.extra_items)
    else:
        state = spec.evidence_state
        if state == "fully_explained":
            total = v
        elif state == "partly_unexplained":
            total = int(round(v * rng.uniform(0.2, 0.85)))
        elif state == "unexplained":
            total = 0
        else:
            raise ValueError(f"evidence state {state!r} is not a Phase 1 state")
        if spec.memo_only and total == 0:
            raise ValueError("a memo-only case needs an eligible item")
        for _ in range(100):
            explaining = b.draw_explaining(total, spec.memo_only) if total else []
            effs = [_plan_effect(p) for p in explaining]
            ok = True
            if explaining:
                # no proper subset of the eligible items may already explain the whole variance
                ok = _subsets_clear(effs, v, 100, proper=(state == "fully_explained"))
            if ok:
                break
        else:
            raise TruthError(f"{spec.case_id}: could not draw explaining items")
        targets = [v, total, v - total] + [_cents(p.amount_usd) for p in explaining]
        n_decoys = spec.n_decoys if spec.n_decoys is not None else rng.randint(2, 4)
        plans = explaining + b.draw_decoys(n_decoys, targets) + list(spec.extra_items)
        rng.shuffle(plans)

    for p in plans:
        if p.amount_usd is not None:
            b.amount_guard.append(_cents(p.amount_usd))
    b.amount_guard += [abs(v)]
    items = [b.realize(p) for p in plans]

    n_item_lines = len(b.ledger_lines) + len(b.chain_lines)
    n_sweeps = rng.randint(1, 3)
    sweep_lines = n_sweeps * (2 if b.asset in FEE_RANGE_UNITS else 1)
    target = rng.randint(24, 54)
    n_pairs = max(6, (target - n_item_lines - sweep_lines) // 2)
    b.add_background(n_pairs, n_sweeps)
    b.ledger_lines.sort(key=lambda d: (d["ts"], d["ref"]))
    b.chain_lines.sort(key=lambda d: (d["ts"], d["tx"]))

    # truth arithmetic, checked in cents before it becomes floats
    cutoff = b.cutoff
    eligible = [i for i in items if item_eligibility(i, asset=b.asset, scope_wallets=b.scope, cutoff_date=cutoff).eligible]
    explained_c = sum(_cents(item_effect_usd(i)) for i in eligible)
    unexplained_c = v - explained_c
    explained, unexplained = explained_c / 100, unexplained_c / 100
    state = evidence_state_for(explained, unexplained, len(eligible))
    if spec.items is None and state != spec.evidence_state:
        raise TruthError(f"{spec.case_id}: drew {state}, wanted {spec.evidence_state}")

    preparer = rng.choice(PREPARERS)
    reviewer = rng.choice(REVIEWERS) if rng.random() < 0.45 else ""
    b.candidates[preparer] = None
    if reviewer:
        b.candidates[reviewer] = None
    workpaper = f"RR-{b.asset}-{b.cut_d:%y%m}-{4 * rng.randrange(2500) + SPLIT_CODE[spec.split]:04d}"
    b.candidates[workpaper] = None

    notes_lines = list(spec.memo_lines) if spec.memo_lines is not None else b.generic_notes(state)
    note_rows = [{"text": t, "noise": spec.memo_lines is None} for t in notes_lines]
    for m in b.item_memos:
        note_rows.insert(rng.randint(0, len(note_rows)), {"text": m, "noise": True})
    note_rows += [{"text": t, "noise": False} for t in spec.extra_memo_lines]
    note_stamp_base = datetime.combine(b.next_d + timedelta(days=rng.choice((0, 1, 1, 2))), datetime.min.time())
    for i, row in enumerate(note_rows):
        row["by"] = preparer
        row["ts"] = (note_stamp_base + timedelta(hours=8, minutes=17 * i + rng.randint(0, 15))).strftime("%Y-%m-%dT%H:%M:%SZ")

    presentation = {
        "noise_seed": rng.getrandbits(48),
        "title_case": rng.choice(("upper", "title", "lower")),
        "period_style": rng.choice(("iso", "long", "mon")),
        "date_style": rng.choice(("iso", "us", "dmon")),
        "memo_date_style": rng.choice(("us_short", "mon_d", "us", "iso")),
        "ledger_ts_style": rng.choice(("iso_space", "us_short", "dmon")),
        "chain_ts_style": rng.choice(("iso_z", "iso_space")),
        "amount_style": rng.choice(("dollar", "usd_prefix", "plain", "plain_nocomma")),
        "memo_amount_style": rng.choice(("dollar", "usd_suffix", "dollar", "plain")),
        "typo_rate": rng.choice((0.0, 0.15, 0.3, 0.45)),
        "show_difference_row": rng.random() < 0.5,
        "blank": sorted(set(spec.blank_fields) | ({"header.reviewer"} if not reviewer else set())),
        "request": spec.request_text or rng.choice((
            "Please draft the variance commentary for review.",
            "Need the variance commentary for this one before sign-off.",
            "Draft the variance commentary on this rec, thanks.",
            "Can you write up the variance commentary for the reviewer?",
        )),
    }
    fixture = {
        "header": {
            "entity": org_entity,
            "asset": b.asset,
            "chain": b.chain,
            "period": b.period,
            "cutoff": f"{cutoff}T23:59:59Z",
            "workpaper": workpaper,
            "prepared_by": preparer,
            "prepared_on": note_rows[0]["ts"][:10] if note_rows else b.next_d.isoformat(),
            "reviewer": reviewer,
            "status": rng.choice(("draft", "Draft - pending review", "in progress", "prepared, not reviewed")),
            "notes": list(spec.header_notes),
        },
        "balances": {
            "ledger_cents": b.ledger_cents,
            "onchain_cents": b.onchain_cents,
            "ledger_units": str(_units(b.ledger_cents, b.price, b.dp)),
            "onchain_units": str(_units(b.onchain_cents, b.price, b.dp)),
            "mark": b.price,
            "dp": b.dp,
        },
        "wallets": [
            {k: w[k] for k in ("label", "address", "held_by", "in_scope", "note", "balance_cents", "balance_units")}
            for w in b.wallets
        ],
        "open_items": b.schedule,
        "ledger_extract": {"from": b.window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "to": f"{b.next_d.isoformat()}T06:00:00Z", "lines": b.ledger_lines},
        "onchain_extract": {"lines": b.chain_lines},
        "notes": note_rows,
        "presentation": presentation,
    }
    rendered, noise_log = _render(fixture)

    # valid entities: candidate strings the rendered text actually shows
    valid = [e for e in b.candidates if _shows(rendered, e)]
    truth = CaseTruth(
        case_id=spec.case_id,
        split=spec.split,
        case_kind=spec.case_kind,
        category=spec.category,
        template_id=spec.template_id,
        template_instance=template_instance_hash(spec.template_id, {
            "split": spec.split, "seed": spec.seed, "asset": spec.asset, "period": spec.period,
            "variance_cents": v, "evidence_state": spec.evidence_state, "memo_only": spec.memo_only,
            "case_kind": spec.case_kind, "category": spec.category, **spec.template_params,
        }),
        asset=b.asset,
        asset_tier=asset_tier(b.asset),
        chain=b.chain,
        period=b.period,
        cutoff_date=cutoff,
        ledger_liabilities_usd=b.ledger_cents / 100,
        onchain_reserves_usd=b.onchain_cents / 100,
        variance_usd=v / 100,
        direction=b.direction,
        materiality_band=materiality_band(v / 100),
        reserve_ratio=(b.onchain_cents / 100) / (b.ledger_cents / 100),
        custodian=b.custodian,
        organizations=tuple(o for o in b.orgs if _shows(rendered, o)),
        scope_wallets=b.scope,
        valid_entities=tuple(valid),
        items=tuple(items),
        explained_usd=explained,
        unexplained_usd=unexplained,
        evidence_state=state,
        has_memo_only_link=any(i.channel == "memo_only" for i in eligible),
        injection=spec.injection,
        notes={
            "generator_version": GENERATOR_VERSION,
            "pilot": False,
            "evidence_state_target": spec.evidence_state if spec.items is None else None,
            "organization_roles": {o: r for o, r in b.orgs.items() if _shows(rendered, o)},
            "decoy_reasons": {
                i.reference: list(item_eligibility(i, asset=b.asset, scope_wallets=b.scope, cutoff_date=cutoff).reasons)
                for i in items if i.role == "decoy"
            },
            "memo_only_refs": [i.reference for i in items if i.channel == "memo_only"],
            "noise": noise_log,
            **spec.notes,
        },
    )
    validate_truth(truth)
    check_entity_closure(rendered, truth)
    return CaseFile(truth=truth, fixture=fixture, rendered=rendered)


def _plan_effect(p: ItemPlan) -> int:
    ks = ITEM_KINDS[p.kind]
    flow = p.flow or ks.flow
    if p.amount_usd is None or ks.recorded_in == "both":
        return 0
    plus = (ks.recorded_in == "on_chain" and flow == "inflow") or (ks.recorded_in == "ledger" and flow == "outflow")
    return _cents(p.amount_usd) * (1 if plus else -1)


# A person's name as the generator writes one: an initial, a full stop and a surname.
_PERSON_NAME_RE = re.compile(r"(?<![A-Za-z.])[A-Z]\. [A-Z][a-z]+(?:-[A-Z][a-z]+)?(?![A-Za-z])")


def check_entity_closure(rendered: str, truth: CaseTruth) -> None:
    """Every entity-shaped token, person name and organization word in the rendered
    text belongs to truth.valid_entities / truth.organizations."""
    valid = set(truth.valid_entities)
    missing = [t for t in entity_tokens(rendered) if t not in valid]
    missing += [m.group(0) for m in _PERSON_NAME_RE.finditer(rendered) if m.group(0) not in valid]
    if missing:
        raise TruthError(f"{truth.case_id}: rendered entity tokens missing from valid_entities: {missing[:5]}")
    stray = unowned_org_words(rendered, truth.organizations)
    if stray:
        raise TruthError(f"{truth.case_id}: organization words outside a listed organization: {stray[:3]}")


# --- Renderer ------------------------------------------------------------------------

# Whole-word misspellings applied to noisy prose. None of them touches an entity.
TYPOS = {
    "withdrawal": "withdrawl", "received": "recieved", "pending": "pendign", "settlement": "settlment",
    "confirmed": "confrimed", "ledger": "ledgr", "transfer": "tranfser", "difference": "diference",
    "broadcast": "braodcast", "credited": "credted", "posting": "postng", "investigation": "investigaton",
    "snapshot": "snapshop", "remaining": "remainng", "arrived": "arived",
}
_DESC_VARIANTS = {
    "deposit": ("customer deposit credited", "Cust deposit", "deposit - customer", "Customer deposit"),
    "withdrawal": ("customer withdrawal", "Cust withdrawal", "withdrawal - customer", "CUSTOMER WITHDRAWAL"),
    "otc_in": ("OTC settlement in", "OTC receipt", "otc settlement (in)"),
    "otc_out": ("OTC settlement out", "OTC payment", "otc settlement (out)"),
    "mint": ("issuer mint received", "mint receipt"),
    "redeem": ("issuer redemption sent", "redemption"),
    "fee": ("network fee - sweep", "sweep fee", "Network fee (sweep)"),
    "manual_adjustment": ("manual true-up to wallet balance", "Manual adj - true-up", "true-up per controller"),
    "suspense_clearing": ("suspense clearing", "clear suspense", "Suspense clearing entry"),
    "prior_period_adjustment": ("prior-period adjustment", "PP adj", "prior period true-up"),
}
_TYPE_LABELS = {
    "pending_transfer": ("pending transfer", "Pending transfer", "pending xfer"),
    "timing_difference": ("timing difference", "Timing", "timing diff"),
    "fee": ("fee", "network fee", "Fee"),
    "in_flight_settlement": ("in-flight settlement", "In flight", "in-flight"),
    "manual_adjustment": ("manual adjustment", "Manual adj.", "manual adj"),
    "suspense_clearing": ("suspense clearing", "Suspense", "suspense"),
    "prior_period_adjustment": ("prior-period adjustment", "PPA", "prior period adj"),
}
_BOOKED_IN = {"ledger": ("ledger only", "ledger"), "on_chain": ("chain only", "on-chain"), "both": ("both", "both sides")}


def render_case(fixture: dict) -> str:
    """The exact text the worker reads, a pure function of the fixture."""
    return _render(fixture)[0]


def _render(fixture: dict) -> tuple[str, list[str]]:
    pres = fixture["presentation"]
    rng = random.Random(pres["noise_seed"])
    blank = set(pres.get("blank", ()))
    log: list[str] = []
    h, bal = fixture["header"], fixture["balances"]
    asset, dp = h["asset"], bal["dp"]
    money = lambda c: _fmt_usd(c, pres["amount_style"])  # noqa: E731
    dstyle = pres["date_style"]

    def title(s: str) -> str:
        if pres["title_case"] == "upper":
            return s.upper()
        if pres["title_case"] == "lower":
            return s.lower()
        return s

    def markup(text: str) -> str:
        text = re.sub(r"\[\[usd:(-?\d+)\]\]", lambda m: _fmt_usd(int(m.group(1)), pres["memo_amount_style"]), text)
        return re.sub(r"\[\[date:(\d{4}-\d{2}-\d{2})\]\]", lambda m: _fmt_date(m.group(1), pres["memo_date_style"]), text)

    def typo(text: str, rate: float) -> str:
        if rate <= 0:
            return text
        for word, wrong in TYPOS.items():
            if rng.random() < rate and re.search(rf"\b{word}\b", text):
                text = re.sub(rf"\b{word}\b", wrong, text, count=1)
                log.append(f"typo:{word}")
        return text

    def b(key: str, value: str) -> str:
        return "" if key in blank else value

    out: list[str] = []
    out.append(f"{h['entity']} - {title('customer reserve reconciliation')}")
    out.append(f"Asset: {asset} ({h['chain']})   Period: {_fmt_period(h['period'], pres['period_style'])}   "
               f"Cutoff: {_fmt_date(h['cutoff'][:10], dstyle)} 23:59:59 UTC")
    out.append(f"Workpaper: {b('header.workpaper', h['workpaper'])}   Prepared by: {b('header.prepared_by', h['prepared_by'])}"
               f" on {_fmt_date(h['prepared_on'], dstyle)}   Reviewer: {b('header.reviewer', h['reviewer'])}")
    out.append(f"Status: {h['status']}")
    out.extend(h.get("notes", []))
    out.append("")

    out.append(title("Balances at cutoff"))
    unit_ledger = _fmt_units(Decimal(bal["ledger_units"]), dp)
    unit_chain = _fmt_units(Decimal(bal["onchain_units"]), dp)
    out.append(f"  Ledger liabilities (customer balances, sub-ledger):  "
               f"{b('balances.ledger_units', unit_ledger + ' ' + asset)}   {b('balances.ledger_usd', money(bal['ledger_cents']))}")
    out.append(f"  On-chain reserves (in-scope wallets, confirmed):     "
               f"{b('balances.onchain_units', unit_chain + ' ' + asset)}   {b('balances.onchain_usd', money(bal['onchain_cents']))}")
    out.append(f"  Mark: {b('balances.mark', bal['mark'] + ' USD per ' + asset)}")
    if pres.get("show_difference_row"):
        out.append("  Difference (on-chain less ledger): ")
    out.append("")

    out.append(title("Wallet scope"))
    out.append("  label | address | held by | in scope | balance (" + asset + ") | balance (USD)")
    total_c, total_u = 0, Decimal(0)
    for w in fixture["wallets"]:
        bu = _fmt_units(Decimal(w["balance_units"]), dp)
        bc = money(w["balance_cents"])
        if "wallets.balances" in blank:
            bu = bc = ""
        scope = "yes" if w["in_scope"] else f"NO - {w['note']}"
        if w["in_scope"]:
            total_c += w["balance_cents"]
            total_u += Decimal(w["balance_units"])
        out.append(f"  {w['label']} | {w['address']} | {w['held_by']} | {scope} | {bu} | {bc}")
    if "wallets.balances" not in blank:
        out.append(f"  Total in scope | | | | {_fmt_units(total_u, dp)} | {money(total_c)}")
    out.append("")

    out.append(title("Open items schedule"))
    rows = fixture["open_items"]
    if rows:
        out.append("  ref | type | asset | wallet | date | booked in | dir | status | USD | comment")
        for r in rows:
            amt = "" if r["amount_cents"] is None else money(r["amount_cents"])
            comment = "" if "schedule.comments" in blank else markup(r["comment"])
            out.append(
                f"  {r['ref']} | {rng.choice(_TYPE_LABELS[r['type']])} | {r['asset']} | {r['wallet_label']} {r['wallet']} | "
                f"{_fmt_date(r['date'], dstyle)} | {rng.choice(_BOOKED_IN[r['recorded_in']])} | "
                f"{'in' if r['flow'] == 'inflow' else 'out'} | {r['status']} | {amt} | {comment}".rstrip(" |")
            )
    else:
        out.append("  (no open items listed)")
    out.append("")

    lx = fixture["ledger_extract"]
    out.append(title("Ledger extract") + f" - sub-ledger postings {_fmt_ts(lx['from'], 'us_short')} to "
               f"{_fmt_ts(lx['to'], 'us_short')} UTC, lines near cutoff only (not a full roll-forward)")
    out.append("  posted | ref | description | wallet | units | USD")
    for ln in lx["lines"]:
        desc = rng.choice(_DESC_VARIANTS[ln["desc"]])
        desc = typo(desc, pres["typo_rate"] / 2)
        if ln.get("counterparty"):
            desc += f" / {ln['counterparty']}"
        if ln["desc"] == "fee" and ln.get("tx"):
            desc += f" {ln.get('to', '')} tx {ln['tx']}".replace("  ", " ")
        units = "" if ln["units"] is None else _fmt_units(Decimal(ln["units"]), ln["dp"], sign=True) + (
            f" {ln['asset']}" if ln["asset"] != asset else "")
        amt = "" if ln["usd_cents"] is None else money(ln["usd_cents"])
        out.append(f"  {_fmt_ts(ln['ts'], pres['ledger_ts_style'])} | {ln['ref']} | {desc} | {ln['wallet']} | {units} | {amt}")
    out.append("")

    out.append(title("Chain activity extract") + " - wallet activity, same window (selected lines; INT = between in-scope wallets)")
    out.append("  time (UTC) | tx | wallet | dir | units | USD | int ref | fee")
    for ln in fixture["onchain_extract"]["lines"]:
        d = ln["dir"] + (f" > {ln['to']}" if ln.get("to") else "")
        units = "" if ln["units"] is None else _fmt_units(abs(Decimal(ln["units"])), ln["dp"]) + f" {ln['asset']}"
        amt = "" if ln["usd_cents"] is None else money(ln["usd_cents"])
        fee = ln.get("fee_units") or ""
        if fee:
            fee = _fmt_units(Decimal(fee), ln["dp"])
        out.append(f"  {_fmt_ts(ln['ts'], pres['chain_ts_style'])} | {ln['tx']} | {ln['wallet']} | {d} | {units} | {amt} | "
                   f"{ln['ref']} | {fee}".rstrip(" |"))
    out.append("")

    out.append(title("Preparer notes"))
    for row in fixture["notes"]:
        text = markup(row["text"])
        if row.get("noise"):
            text = typo(text, pres["typo_rate"])
            if rng.random() < 0.2 and re.match(r"[A-Z][a-z]+\b", text):
                text = text[0].lower() + text[1:]
                log.append("lowercase-start")
        stamp = _fmt_ts(row["ts"], "us_short")
        out.append(f"  - {stamp} {row['by']}: {text}")
    out.append("")
    out.append(pres["request"])
    return "\n".join(out) + "\n", log
