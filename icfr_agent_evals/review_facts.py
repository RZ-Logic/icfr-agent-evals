"""Looked-up facts for the grouped review packet.

Each fact block holds values read from the case file the worker saw, values the
answer key stores, or a small computation the build spec names (item eligibility
and why not, each item's effect on the variance, explained and unexplained from
the items an output cites). A block never compares a worker claim with the case
file, never picks case-file rows by resemblance to a worker token, and never
carries a label that reads as a verdict. Rows keep case-file order.

Allowed reads only. From the answer key: asset, chain, period, cutoff_date,
ledger_liabilities_usd, onchain_reserves_usd, variance_usd, direction,
materiality_band, explained_usd, unexplained_usd, organizations, valid_entities,
scope_wallets and items (reference, type, asset, wallet, date, open_status,
recorded_in, flow, amount_usd, aliases), plus its eligibility rule. From the
fixture: the header, balances, ledger_extract, onchain_extract, open_items,
wallets and notes blocks (for a note: ts, by and text). Nothing else.

Wallet facts are plain lookups of the fixture's wallet table: a label found as a
whole word in a sentence, and that wallet's row. No wallet rule is applied.

Fields the renderer may leave blank never appear in a fact. A case can blank
wallet balances and schedule comments (generator.BLANKABLE_FIELDS), and the
renderer shows a wallet note only beside an out-of-scope wallet. So a wallet row
shows label, address, held by and in scope only (pre-registration section 6:
each wallet label with its holder and scope), and an open-items line never shows
its comment. The worker's own view of those fields is the rendered case file.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import casefile, policy, review_schema, scoring

BLOCK_IDS: tuple[str, ...] = (
    "answer_key", "thresholds", "items", "wallets", "organizations", "notes",
    "worker_fields", "worker_items", "worker_split", "named_references", "fixture_rows",
    "named_wallets", "role_targets", "period_dates", "entities", "routed_field",
    "type_labels", "case_file_exact", "uncited_names",
)

TITLES: dict[str, str] = {
    "answer_key": "Answer key values",
    "thresholds": "Materiality band thresholds",
    "items": "Items in the answer key, in case-file order",
    "wallets": "Wallet table from the case file",
    "organizations": "Organizations the case file shows",
    "notes": "Notes in the case file, as stored",
    "worker_fields": "Structured fields of the output, as written",
    "worker_items": "Reconciling items of the output, as written",
    "worker_split": "Explained and unexplained, computed by code from the items this output cites",
    "named_references": "References and aliases the routed text names, with their item rows",
    "fixture_rows": "Case-file lines that hold a named reference, as stored before rendering",
    "named_wallets": "Wallet labels the routed text names, with their wallet rows",
    "role_targets": "Amounts in the answer key and the output, by role",
    "period_dates": "Item dates by reference",
    "entities": "Entity strings the case file shows",
    "routed_field": "Structured field named in the route, as stated and in the answer key",
    "type_labels": "Labels the schedule may show for the item's type",
    "case_file_exact": "Flagged names the case file holds exactly as written",
    "uncited_names": "Case-file names the narrative gives that entities_cited leaves out, as code records them",
}
assert set(TITLES) == set(BLOCK_IDS)

ELIGIBILITY_LABELS: dict[str, str] = {
    "not_open": "status not open",
    "recorded_both": "recorded in ledger and on chain",
    "wrong_asset": "asset other than the case asset",
    "out_of_scope_wallet": "wallet outside the stated scope",
    "date_missing": "no date",
    "after_cutoff": "dated after the cutoff",
    "disallowed_type": "type not in the allowed list",
    "amount_missing": "no amount",
}

FACTS_FAMILIES: tuple[str, ...] = (
    "amount", "split", "direction", "band", "asset", "period", "statement", "sweep",
    "reference_item", "structured_field", "none",
)

NOT_PARSED = "output did not parse"
CELL_SEP = "; "

# --- Block --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactBlock:
    """One table of looked-up facts. Every cell is a string; no cell is a comparison."""

    block_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    verbatim_columns: frozenset[str]  # columns copied unchanged from the fixture or the worker output

    def __post_init__(self) -> None:
        if self.block_id not in BLOCK_IDS:
            raise ValueError(f"unknown fact block id {self.block_id!r}")
        if not isinstance(self.title, str):
            raise ValueError("a fact block title is a string")
        if not all(isinstance(c, str) for c in self.columns):
            raise ValueError("fact block columns are strings")
        unknown = set(self.verbatim_columns) - set(self.columns)
        if unknown:
            raise ValueError(f"verbatim columns not in the block: {sorted(unknown)}")
        width = len(self.columns)
        for row in self.rows:
            if len(row) != width:
                raise ValueError(f"block {self.block_id}: a row has {len(row)} cells, the block {width} columns")
            if not all(isinstance(cell, str) for cell in row):
                raise ValueError(f"block {self.block_id}: every cell is a string")

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "verbatim_columns": sorted(self.verbatim_columns),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FactBlock":
        return cls(
            block_id=str(d["block_id"]),
            title=str(d["title"]),
            columns=tuple(d["columns"]),
            rows=tuple(tuple(r) for r in d["rows"]),
            verbatim_columns=frozenset(d.get("verbatim_columns") or ()),
        )


def _block(block_id: str, columns: Sequence[str], rows: Iterable[Sequence[str]],
           verbatim: Iterable[str] = ()) -> FactBlock:
    return FactBlock(
        block_id=block_id,
        title=TITLES[block_id],
        columns=tuple(columns),
        rows=tuple(tuple(r) for r in rows),
        verbatim_columns=frozenset(verbatim),
    )


def facts_payload(blocks: Sequence[FactBlock]) -> list[dict]:
    return [b.to_dict() for b in blocks]


def verdict_word_hits(blocks: Sequence[FactBlock]) -> list[str]:
    """review_schema.forbidden_word_hits on every title, column name and cell of a
    column that is not verbatim; sorted, distinct."""
    hits: set[str] = set()
    for b in blocks:
        hits.update(review_schema.forbidden_word_hits(b.title))
        for col in b.columns:
            hits.update(review_schema.forbidden_word_hits(col))
        for row in b.rows:
            for col, cell in zip(b.columns, row):
                if col not in b.verbatim_columns:
                    hits.update(review_schema.forbidden_word_hits(cell))
    return sorted(hits)


# --- Rendering ----------------------------------------------------------------------


def _usd(value: float | None) -> str:
    if value is None:
        return "no amount"
    return f"${abs(float(value)):,.2f}"


def _signed_usd(value: float | None) -> str:
    """Signed form, used for the variance, an item's effect and the explained and
    unexplained sums (each a sum of signed effects)."""
    if value is None:
        return "no amount"
    v = round(float(value), 2)
    if v == 0:
        return "$0.00"
    return f"{'+' if v > 0 else '-'}${abs(v):,.2f}"


def _as_written(value: Any) -> str:
    """A worker or fixture value unchanged: a string as is, anything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


# --- Sources ------------------------------------------------------------------------


def _fixture(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    fx = (metadata or {}).get("fixture")
    return fx if isinstance(fx, Mapping) else {}


def _wallet_rows(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _fixture(metadata).get("wallets") or []
    return [w for w in rows if isinstance(w, Mapping)]


def _note_rows(metadata: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Each fixture note as (ts, by, text); no other field of a note is read."""
    out: list[tuple[str, str, str]] = []
    for n in _fixture(metadata).get("notes") or []:
        if isinstance(n, Mapping):
            out.append((_as_written(n.get("ts", "")), _as_written(n.get("by", "")), _as_written(n.get("text", ""))))
    return out


def _extract_lines(metadata: Mapping[str, Any], block: str) -> list[Mapping[str, Any]]:
    section = _fixture(metadata).get(block) or {}
    lines = section.get("lines") if isinstance(section, Mapping) else None
    return [ln for ln in (lines or []) if isinstance(ln, Mapping)]


def _open_item_rows(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in (_fixture(metadata).get("open_items") or []) if isinstance(r, Mapping)]


# Fields shown for each fixture line kind, in this order (an explicit allow-list).
_LEDGER_FIELDS = ("ts", "ref", "tx", "wallet", "asset", "counterparty", "desc", "to", "units", "usd_cents", "dp")
_ONCHAIN_FIELDS = ("ts", "ref", "tx", "wallet", "asset", "counterparty", "dir", "to", "units", "fee_units",
                   "usd_cents", "dp")
# No "comment": a case may blank schedule comments (generator.BLANKABLE_FIELDS).
_OPEN_ITEM_FIELDS = ("ref", "type", "asset", "wallet_label", "wallet", "date", "status", "recorded_in", "flow",
                     "amount_cents")
_SOURCE_NAMES = {
    "ledger_extract": "ledger extract",
    "onchain_extract": "on-chain extract",
    "open_items": "open items schedule",
    "items": "items list",
    "notes": "notes",
}


def _line_text(line: Mapping[str, Any], fields: Sequence[str]) -> str:
    picked = {k: line[k] for k in fields if k in line}
    return json.dumps(picked, ensure_ascii=False, separators=(", ", ": "))


def _label_of_address(metadata: Mapping[str, Any], address: str) -> str:
    for w in _wallet_rows(metadata):
        if w.get("address") == address:
            return _as_written(w.get("label", ""))
    return ""


# --- Text lookups -------------------------------------------------------------------

_REF_BEFORE = r"(?<![A-Za-z0-9_-])"
_REF_AFTER = r"(?![A-Za-z0-9_]|-[A-Za-z0-9])"
_WORD_BEFORE = r"(?<![^\W_])"
_WORD_AFTER = r"(?![^\W_])"


def _spans(text: str, candidates: Iterable[str], before: str, after: str,
           flags: int = 0) -> list[tuple[int, int, str]]:
    """Non-overlapping exact occurrences of the candidates in text, leftmost first,
    the longer candidate first at one position."""
    found: list[tuple[int, int, str]] = []
    for cand in {c for c in candidates if c}:
        for m in re.finditer(f"{before}{re.escape(cand)}{after}", text, flags):
            found.append((m.start(), m.end(), cand))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0]), t[2]))
    kept: list[tuple[int, int, str]] = []
    end = -1
    for start, stop, cand in found:
        if start >= end:
            kept.append((start, stop, cand))
            end = stop
    return kept


def _reference_forms(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Canonical form -> the case-file form, for every item reference and alias and
    every ref and tx value of the ledger, on-chain and open-items rows."""
    forms: dict[str, str] = {}

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            forms.setdefault(casefile.canonical_reference(value), value)

    truth = casefile.truth_of(dict(metadata))
    for item in truth.items:
        add(item.reference)
        for a in item.aliases:
            add(a)
    for block in ("ledger_extract", "onchain_extract"):
        for line in _extract_lines(metadata, block):
            add(line.get("ref"))
            add(line.get("tx"))
    for row in _open_item_rows(metadata):
        add(row.get("ref"))
    return forms


def named_references(text: str, metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Case-file references and aliases found in text by exact canonical lookup
    (NFKC, format characters spaced, case kept), in text order, distinct."""
    forms = _reference_forms(metadata)
    canon = casefile.canonical_reference(text or "")
    out: list[str] = []
    for _, _, cand in _spans(canon, forms, _REF_BEFORE, _REF_AFTER):
        name = forms[cand]
        if name not in out:
            out.append(name)
    return tuple(out)


def named_wallet_labels(text: str, metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Fixture wallet labels found in text as whole words, case kept, in text order."""
    labels = [w.get("label") for w in _wallet_rows(metadata)]
    out: list[str] = []
    for _, _, label in _spans(str(text or ""), [x for x in labels if isinstance(x, str)], _WORD_BEFORE, _WORD_AFTER):
        if label not in out:
            out.append(label)
    return tuple(out)


# --- Reason parsing (the token a block concerns comes only from the reason text) ----

_SENTINEL = "\x00sentinel\x00"
SWEEP_REASON_PREFIXES: tuple[str, ...] = tuple(
    scoring.SweepHit(_SENTINEL, kind).reason.split(repr(_SENTINEL))[0] for kind in scoring.SWEEP_KINDS
)
_QUOTED = r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
_ITEM_ENTRY_RE = re.compile(r"^reconciling_items\[\d+\] " + _QUOTED, re.DOTALL)
_INELIGIBLE_RE = re.compile(r"ineligible item (\S+)")
_ENTITIES_CITED_RE = re.compile(r"^entities_cited names " + _QUOTED, re.DOTALL)
_NARRATIVE_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*: ")
_OFF_CONTRACT_RE = re.compile("^" + re.escape(scoring.OFF_CONTRACT_REASON) + r": (\S+)")
_OFF_CONTRACT_FIELDS = ("variance_amount_usd", "materiality_band", "period")


def _literal(text: str) -> str | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, str) else None


def _sweep_token(reason: str) -> str | None:
    for prefix in SWEEP_REASON_PREFIXES:
        if prefix and reason.startswith(prefix) and len(reason) > len(prefix):
            return _literal(reason[len(prefix):])
    return None


def _item_reference_of(reason: str) -> str | None:
    """The reference an S14, S15 or S16 reason names."""
    m = _ITEM_ENTRY_RE.match(reason)
    if m:
        return _literal(m.group(1))
    m = _INELIGIBLE_RE.search(reason)
    if m and not reason.startswith(SWEEP_REASON_PREFIXES):
        return m.group(1)
    return None


def _reason_text(reason: str) -> str:
    """The token or text one reason names: a sweep token, a quoted reference, an
    ineligible item's reference, or a narrative kind's matched text."""
    token = _sweep_token(reason)
    if token is not None:
        return token
    ref = _item_reference_of(reason)
    if ref is not None:
        return ref
    m = _ENTITIES_CITED_RE.match(reason)
    if m:
        return _literal(m.group(1)) or ""
    if _NARRATIVE_KIND_RE.match(reason):
        return reason.split(": ", 1)[1]
    return ""


def _off_contract_fields(reasons: Sequence[str]) -> list[str]:
    named: list[str] = []
    for r in reasons:
        m = _OFF_CONTRACT_RE.match(str(r))
        if m and m.group(1) not in named:
            named.append(m.group(1))
    order = {f: i for i, f in enumerate(_OFF_CONTRACT_FIELDS)}
    return sorted(named, key=lambda f: (order.get(f, len(order)), f))


# --- Answer-key rows ----------------------------------------------------------------


def _answer_key_rows(truth: casefile.CaseTruth) -> list[tuple[str, str, str]]:
    """(key, quantity, value) in a fixed order."""
    return [
        ("asset", "asset", truth.asset),
        ("chain", "chain", truth.chain),
        ("period", "period", truth.period),
        ("cutoff", "cutoff date", truth.cutoff_date),
        ("ledger", "ledger liabilities", _usd(truth.ledger_liabilities_usd)),
        ("onchain", "on-chain reserves", _usd(truth.onchain_reserves_usd)),
        ("variance", "variance, on-chain minus ledger", f"{_signed_usd(truth.variance_usd)} ({truth.direction})"),
        ("direction", "direction", truth.direction),
        ("band", "materiality band", truth.materiality_band),
        ("explained", "explained by eligible items", _signed_usd(truth.explained_usd)),
        ("unexplained", "unexplained by eligible items", _signed_usd(truth.unexplained_usd)),
    ]


def _answer_key_block(truth: casefile.CaseTruth, keys: Sequence[str] | None = None) -> FactBlock:
    rows = _answer_key_rows(truth)
    if keys is not None:
        wanted = set(keys)
        rows = [r for r in rows if r[0] in wanted]
    return _block("answer_key", ("quantity", "value"), [(q, v) for _, q, v in rows])


def _thresholds_block() -> FactBlock:
    lo, hi = _usd(policy.M1_CEILING), _usd(policy.M3_FLOOR)
    return _block("thresholds", ("band", "range of the variance magnitude"), [
        ("M1", f"below {lo}"),
        ("M2", f"from {lo} to {hi}, both included"),
        ("M3", f"above {hi}"),
    ])


_ITEM_COLUMNS = ("reference", "type", "asset", "wallet label", "wallet address", "date", "status",
                 "recorded in", "flow", "amount", "effect on variance", "eligible", "reasons not eligible",
                 "aliases")
_ITEM_VERBATIM = ("reference", "wallet label", "wallet address", "aliases")


def _item_cells(item: casefile.ReconcilingItem, truth: casefile.CaseTruth,
                metadata: Mapping[str, Any]) -> tuple[str, ...]:
    elig = truth.eligibility(item)
    return (
        item.reference,
        item.type,
        item.asset,
        _label_of_address(metadata, item.wallet),
        item.wallet,
        item.date if item.date else "no date",
        item.open_status,
        item.recorded_in,
        item.flow,
        _usd(item.amount_usd),
        _signed_usd(casefile.item_effect_usd(item)),
        _yes_no(elig.eligible),
        CELL_SEP.join(ELIGIBILITY_LABELS.get(r, r) for r in elig.reasons),
        CELL_SEP.join(item.aliases),
    )


def _items_block(truth: casefile.CaseTruth, metadata: Mapping[str, Any]) -> FactBlock:
    return _block("items", _ITEM_COLUMNS, [_item_cells(i, truth, metadata) for i in truth.items], _ITEM_VERBATIM)


# No balance or note: a case may blank wallet balances, and the renderer shows a
# note only beside an out-of-scope wallet. Pre-registration section 6 lists label,
# holder and scope.
_WALLET_COLUMNS = ("label", "address", "held by", "in scope")
_WALLET_VERBATIM = ("label", "address", "held by")


def _wallet_cells(w: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _as_written(w.get("label", "")),
        _as_written(w.get("address", "")),
        _as_written(w.get("held_by", "")),
        _yes_no(bool(w.get("in_scope"))),
    )


def _wallets_block(metadata: Mapping[str, Any]) -> FactBlock:
    return _block("wallets", _WALLET_COLUMNS, [_wallet_cells(w) for w in _wallet_rows(metadata)], _WALLET_VERBATIM)


def _organizations_block(truth: casefile.CaseTruth) -> FactBlock:
    return _block("organizations", ("organization",), [(o,) for o in truth.organizations], ("organization",))


def _notes_block(metadata: Mapping[str, Any]) -> FactBlock:
    cols = ("timestamp", "by", "text")
    return _block("notes", cols, _note_rows(metadata), cols)


def _entities_block(truth: casefile.CaseTruth) -> FactBlock:
    return _block("entities", ("entity",), [(e,) for e in truth.valid_entities], ("entity",))


# --- Worker-output rows -------------------------------------------------------------


def _worker_entries(parsed: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(parsed, Mapping):
        return []
    items = parsed.get("reconciling_items")
    return list(items) if isinstance(items, list) else []


def _cited_references(parsed: Mapping[str, Any] | None) -> list[str]:
    refs: list[str] = []
    for entry in _worker_entries(parsed):
        if isinstance(entry, Mapping) and "reference" in entry:
            refs.append(str(entry["reference"]))
    return refs


_WORKER_ITEM_COLUMNS = ("entry", "reference", "amount_usd", "type")
_WORKER_ITEM_VERBATIM = ("reference", "amount_usd", "type")


def _worker_item_cells(n: int, entry: Any) -> tuple[str, ...]:
    if not isinstance(entry, Mapping):
        return (f"reconciling_items[{n}]", _as_written(entry), "", "")
    return (
        f"reconciling_items[{n}]",
        _as_written(entry["reference"]) if "reference" in entry else "",
        _as_written(entry["amount_usd"]) if "amount_usd" in entry else "",
        _as_written(entry["type"]) if "type" in entry else "",
    )


def _worker_items_block(parsed: Mapping[str, Any] | None, only_reference: str | None = None) -> FactBlock:
    rows = []
    want = casefile.canonical_reference(only_reference) if only_reference is not None else None
    for n, entry in enumerate(_worker_entries(parsed)):
        if want is not None:
            if not (isinstance(entry, Mapping) and "reference" in entry
                    and casefile.canonical_reference(entry["reference"]) == want):
                continue
        rows.append(_worker_item_cells(n, entry))
    return _block("worker_items", _WORKER_ITEM_COLUMNS, rows, _WORKER_ITEM_VERBATIM)


def _worker_split_values(truth: casefile.CaseTruth, parsed: Mapping[str, Any] | None) -> tuple[str, str]:
    if parsed is None:
        return NOT_PARSED, NOT_PARSED
    explained, unexplained = casefile.explained_from_citations(_cited_references(parsed), truth)
    return _signed_usd(explained), _signed_usd(unexplained)


def _worker_split_block(truth: casefile.CaseTruth, parsed: Mapping[str, Any] | None) -> FactBlock:
    explained, unexplained = _worker_split_values(truth, parsed)
    return _block("worker_split", ("quantity", "amount"), [("explained amount", explained), ("unexplained amount", unexplained)])


def _worker_fields_block(parsed: Mapping[str, Any] | None, only: Sequence[str] | None = None) -> FactBlock:
    rows: list[tuple[str, str]] = []
    keys = [k for k in scoring.OUTPUT_FIELDS if k != "narrative"]
    if only is not None:
        keys = [k for k in keys if k in only]
    for key in keys:
        if isinstance(parsed, Mapping) and key in parsed:
            rows.append((key, _as_written(parsed[key])))
    return _block("worker_fields", ("field", "value"), rows, ("value",))


def _entities_cited_block(parsed: Mapping[str, Any] | None) -> FactBlock:
    """The worker's entities_cited, one entry per row, as written."""
    rows: list[tuple[str, str]] = []
    cited = parsed.get("entities_cited") if isinstance(parsed, Mapping) else None
    if parsed is None:
        rows.append(("entities_cited", NOT_PARSED))
    elif isinstance(cited, list):
        rows.extend((f"entities_cited[{n}]", _as_written(e)) for n, e in enumerate(cited))
    return _block("worker_fields", ("field", "value"), rows, ("value",))


# --- Public builders ----------------------------------------------------------------


def case_facts(metadata: Mapping[str, Any]) -> tuple[FactBlock, ...]:
    """Shown once per case: answer key, thresholds, items, wallets, organizations, notes."""
    truth = casefile.truth_of(dict(metadata))
    return (
        _answer_key_block(truth),
        _thresholds_block(),
        _items_block(truth, metadata),
        _wallets_block(metadata),
        _organizations_block(truth),
        _notes_block(metadata),
    )


def output_facts(metadata: Mapping[str, Any], parsed: Mapping[str, Any] | None) -> tuple[FactBlock, ...]:
    """Per narrative unit: the output's fields and items as written, and the split
    computed from the items it cites. An output that did not parse gets one row."""
    if parsed is None:
        return (_block("worker_fields", ("output",), [(NOT_PARSED,)]),)
    truth = casefile.truth_of(dict(metadata))
    return (
        _worker_fields_block(parsed),
        _worker_items_block(parsed),
        _worker_split_block(truth, parsed),
        _uncited_names_block(truth, parsed),
    )


def _uncited_names_block(truth: casefile.CaseTruth, parsed: Mapping[str, Any]) -> FactBlock:
    """Owner request (2026-11-11): the citation omissions code records on this output
    (P12, the case-file-exact sweep names and the wallet-label matcher), so a reviewer
    sees what code already counted. A lookup of code's own record, never a reading of
    a sentence; "none" when code records none."""
    details = scoring.check_fabrication(dict(parsed), truth).details
    seen: dict[str, str] = {}
    for key, how in (("citation_omissions", "reference number or wallet label (P12)"),
                     ("case_file_names_uncited", "name the case file holds exactly"),
                     ("wallet_label_mentions_uncited", "wallet label, any case, space or hyphen form")):
        for name in details.get(key, []) or []:
            seen.setdefault(str(name), how)
    rows = [(name, how) for name, how in seen.items()] or [("none", "")]
    return _block("uncited_names", ("name", "recorded as"), rows, ("name",))  # the name as the narrative gives it


def _named_references_block(names: Sequence[str], truth: casefile.CaseTruth,
                            metadata: Mapping[str, Any]) -> FactBlock:
    cols = ("named in text", "found in") + _ITEM_COLUMNS
    rows = []
    for name in names:
        key = casefile.canonical_reference(name)
        item = truth.item_by_reference(name)
        where: list[str] = []
        if item is not None:
            where.append(_SOURCE_NAMES["items"])
        for block in ("ledger_extract", "onchain_extract"):
            if any(casefile.canonical_reference(ln.get(f)) == key
                   for ln in _extract_lines(metadata, block) for f in ("ref", "tx") if isinstance(ln.get(f), str)):
                where.append(_SOURCE_NAMES[block])
        if any(isinstance(r.get("ref"), str) and casefile.canonical_reference(r["ref"]) == key
               for r in _open_item_rows(metadata)):
            where.append(_SOURCE_NAMES["open_items"])
        if any(_spans(casefile.canonical_reference(text), {key}, _REF_BEFORE, _REF_AFTER)
               for _, _, text in _note_rows(metadata)):
            where.append(_SOURCE_NAMES["notes"])
        cells = _item_cells(item, truth, metadata) if item is not None else ("",) * len(_ITEM_COLUMNS)
        rows.append((name, CELL_SEP.join(where)) + cells)
    return _block("named_references", cols, rows, ("named in text",) + _ITEM_VERBATIM)


def _named_wallets_block(labels: Sequence[str], metadata: Mapping[str, Any]) -> FactBlock:
    rows = []
    for label in labels:
        for w in _wallet_rows(metadata):
            if w.get("label") == label:
                rows.append(_wallet_cells(w))
    return _block("named_wallets", _WALLET_COLUMNS, rows, _WALLET_VERBATIM)


def _lookup_forms(names: Sequence[str], truth: casefile.CaseTruth) -> set[str]:
    """Canonical forms to look up: each named reference, plus the reference and
    aliases of the item it resolves to."""
    forms: set[str] = set()
    for name in names:
        forms.add(casefile.canonical_reference(name))
        item = truth.item_by_reference(name)
        if item is not None:
            forms.add(casefile.canonical_reference(item.reference))
            forms.update(casefile.canonical_reference(a) for a in item.aliases)
    return forms


def _fixture_rows_block(names: Sequence[str], truth: casefile.CaseTruth,
                        metadata: Mapping[str, Any]) -> FactBlock:
    forms = _lookup_forms(names, truth)
    rows: list[tuple[str, str, str]] = []
    for block, fields in (("ledger_extract", _LEDGER_FIELDS), ("onchain_extract", _ONCHAIN_FIELDS)):
        for n, line in enumerate(_extract_lines(metadata, block), start=1):
            if any(isinstance(line.get(f), str) and casefile.canonical_reference(line[f]) in forms
                   for f in ("ref", "tx")):
                rows.append((_SOURCE_NAMES[block], str(n), _line_text(line, fields)))
    for n, row in enumerate(_open_item_rows(metadata), start=1):
        if isinstance(row.get("ref"), str) and casefile.canonical_reference(row["ref"]) in forms:
            rows.append((_SOURCE_NAMES["open_items"], str(n), _line_text(row, _OPEN_ITEM_FIELDS)))
    for n, (ts, by, text) in enumerate(_note_rows(metadata), start=1):
        if _spans(casefile.canonical_reference(text), forms, _REF_BEFORE, _REF_AFTER):
            rows.append((_SOURCE_NAMES["notes"], str(n),
                         json.dumps({"ts": ts, "by": by, "text": text}, ensure_ascii=False, separators=(", ", ": "))))
    return _block("fixture_rows", ("source", "row", "line"), rows, ("line",))


def _role_targets_block(truth: casefile.CaseTruth, parsed: Mapping[str, Any] | None) -> FactBlock:
    w_explained, w_unexplained = _worker_split_values(truth, parsed)
    rows = [
        ("variance magnitude", "", _usd(abs(truth.variance_usd))),
        ("ledger liabilities", "", _usd(truth.ledger_liabilities_usd)),
        ("on-chain reserves", "", _usd(truth.onchain_reserves_usd)),
        ("M1 ceiling", "", _usd(policy.M1_CEILING)),
        ("M3 floor", "", _usd(policy.M3_FLOOR)),
        ("explained by eligible items", "", _signed_usd(truth.explained_usd)),
        ("unexplained by eligible items", "", _signed_usd(truth.unexplained_usd)),
        ("explained from the items the output cites", "", w_explained),
        ("unexplained from the items the output cites", "", w_unexplained),
    ]
    rows += [("item amount", i.reference, _usd(i.amount_usd)) for i in truth.items]
    return _block("role_targets", ("quantity", "reference", "amount"), rows, ("reference",))


def _period_dates_block(truth: casefile.CaseTruth) -> FactBlock:
    return _block("period_dates", ("reference", "date"),
                  [(i.reference, i.date if i.date else "no date") for i in truth.items], ("reference",))


def _type_labels_block(refs: Sequence[str], truth: casefile.CaseTruth) -> FactBlock:
    rows = []
    for ref in refs:
        item = truth.item_by_reference(ref)
        if item is None:
            continue
        for label in casefile.TYPE_LABELS.get(item.type, ()):
            rows.append((ref, item.type, label))
    return _block("type_labels", ("reference", "type", "label"), rows, ("reference",))


def _routed_field_block(truth: casefile.CaseTruth, parsed: Mapping[str, Any] | None,
                        reasons: Sequence[str]) -> FactBlock:
    key_values = {
        "variance_amount_usd": truth.variance_usd,
        "materiality_band": truth.materiality_band,
        "period": truth.period,
    }
    rows = []
    for f in _off_contract_fields(reasons):
        if parsed is None:
            stated = NOT_PARSED
        elif isinstance(parsed, Mapping) and f in parsed:
            stated = _as_written(parsed[f])
        else:
            stated = ""
        answer = _as_written(key_values[f]) if f in key_values else ""
        rows.append((f, stated, answer))
    return _block("routed_field", ("field", "stated in the output", "answer key"), rows, ("stated in the output",))


CASE_FILE_EXACT_YES, CASE_FILE_EXACT_NO, CASE_FILE_EXACT_UNRECORDED = "yes", "no", "not recorded"


def case_file_exact_block(reasons: Sequence[str], sweep_hits: Sequence[Mapping[str, Any]]) -> FactBlock:
    """Owner decision D-R9 (2026-11-11), red-team B1: for each token a sweep finding's
    reasons name, whether the case file holds it exactly as written, the `in_case_file`
    flag the scorer records on its sweep hit (the test FM-005 applies to a cited
    entity). A token with no recorded flag shows "not recorded". A looked-up fact, no
    reading of any sentence."""
    flag = {str(h.get("token")): h.get("in_case_file") for h in sweep_hits if isinstance(h, Mapping)}
    rows: list[tuple[str, str]] = []
    for r in reasons:
        token = _sweep_token(str(r))
        if token is None or any(token == row[0] for row in rows):
            continue
        value = flag.get(token)
        rows.append((token, CASE_FILE_EXACT_YES if value is True else
                     CASE_FILE_EXACT_NO if value is False else CASE_FILE_EXACT_UNRECORDED))
    return _block("case_file_exact", ("flagged name", "in the case file exactly as written"), rows,
                  ("flagged name",))


def finding_facts(metadata: Mapping[str, Any], parsed: Mapping[str, Any] | None, facts_family: str,
                  reasons: Sequence[str], routed_sentences: Sequence[str]) -> tuple[FactBlock, ...]:
    """Facts for one finding unit, by facts family. Every family but `none` starts
    with the references and wallet labels the routed text and the reason name."""
    if facts_family not in FACTS_FAMILIES:
        raise ValueError(f"unknown facts family {facts_family!r}")
    if facts_family == "none":
        return ()
    truth = casefile.truth_of(dict(metadata))
    reasons = [str(r) for r in reasons]
    reason_texts = [t for t in (_reason_text(r) for r in reasons) if t]
    search_parts = [str(s) for s in routed_sentences] + reason_texts

    names: list[str] = []
    labels: list[str] = []
    for part in search_parts:
        for n in named_references(part, metadata):
            if n not in names:
                names.append(n)
        for lb in named_wallet_labels(part, metadata):
            if lb not in labels:
                labels.append(lb)

    blocks: list[FactBlock] = [
        _named_references_block(names, truth, metadata),
        _named_wallets_block(labels, metadata),
    ]
    if facts_family == "amount":
        blocks += [_role_targets_block(truth, parsed), _fixture_rows_block(names, truth, metadata)]
    elif facts_family == "split":
        blocks += [_role_targets_block(truth, parsed), _worker_split_block(truth, parsed)]
    elif facts_family == "direction":
        blocks += [_answer_key_block(truth, ("variance", "direction", "ledger", "onchain"))]
    elif facts_family == "band":
        blocks += [_thresholds_block(), _answer_key_block(truth, ("band", "variance")), _notes_block(metadata)]
    elif facts_family == "asset":
        blocks += [
            _answer_key_block(truth, ("asset", "chain")),
            _block("items", ("reference", "asset"), [(i.reference, i.asset) for i in truth.items], ("reference",)),
            _wallets_block(metadata),
        ]
    elif facts_family == "period":
        blocks += [_answer_key_block(truth, ("period", "cutoff")), _period_dates_block(truth)]
    elif facts_family == "statement":
        blocks += [
            _answer_key_block(truth, ("variance", "direction", "band", "asset", "chain", "period", "cutoff")),
            _thresholds_block(),
            _notes_block(metadata),
        ]
    elif facts_family == "sweep":
        blocks += [_entities_block(truth), _organizations_block(truth), _entities_cited_block(parsed)]
    elif facts_family == "reference_item":
        refs: list[str] = []
        for r in reasons:
            ref = _item_reference_of(r)
            if ref is not None and ref not in refs:
                refs.append(ref)
        blocks.append(_type_labels_block(refs, truth))
        for ref in refs:
            blocks.append(_worker_items_block(parsed, only_reference=ref))
    elif facts_family == "structured_field":
        blocks.append(_routed_field_block(truth, parsed, reasons))
    return tuple(blocks)
