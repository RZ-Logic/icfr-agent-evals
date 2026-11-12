"""The review settlement table: which route keys a reviewer may settle, and how.

Pre-registration section 6 (Review settlement): review settles routes, and only a
person settles a route. This module is pure data plus lookup. It never sets, implies
or applies a verdict; `on_correct` and `review_found_code` are data for the later
settlement step, which is not built here.

Every route key `regression.route_key` can give maps to exactly one row; a key no
row takes maps to the catch-all S74, which is unsettleable (it fails on code).
`tests/test_settlement.py` scans every emit site in `narrative.py` and `scoring.py`
so a new reason kind fails the tests until it has a row.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Sequence

from . import regression
from . import scoring
from .review_schema import FINDING_VERDICTS

MATCH_KINDS: tuple[str, ...] = ("exact", "prefix", "pattern", "off_contract_field", "catch_all")
ROUTE_CLASSES: tuple[str, ...] = ("narrative_sweep", "code_review", "any")
SOURCES: tuple[str, ...] = ("narrative", "structured", "system", "any")
ANCHORS: tuple[str, ...] = ("correctness", "fabrication", "reconciling_items", "correctness_and_completeness", "any")
FACTS_FAMILIES: tuple[str, ...] = (
    "amount", "split", "direction", "band", "asset", "period", "statement", "sweep",
    "reference_item", "structured_field", "none",
)


@dataclass(frozen=True)
class SettlementRow:
    row_id: str                         # "S01" .. "S77", stable, in table order
    match: str                          # one of MATCH_KINDS
    key: str                            # full route key, key prefix, regex (fullmatch) or field name
    route_class: str                    # "narrative_sweep" | "code_review" | "any"
    source: str                         # one of SOURCES
    anchor: str                         # one of ANCHORS
    settleable: bool
    correct_means: str
    on_correct: str                     # effect text (data only; no effect is applied here)
    review_found_code: str | None       # "FM-001" ...; "downgraded_from" for S69; None if unsettleable
    allowed_verdicts: tuple[str, ...]   # () when unsettleable
    facts_family: str                   # one of FACTS_FAMILIES; "none" when unsettleable
    unit_test: str                      # name of its test in tests/test_settlement.py
    note: str


SWEEP = regression.SWEEP_ROUTE
CODE = regression.CODE_ROUTE
_CR = f"{CODE}: "
OFF_CONTRACT_ROUTE_KEY = f"{CODE}: {scoring.OFF_CONTRACT_REASON}"
OFF_CONTRACT_FIELDS: tuple[str, ...] = ("variance_amount_usd", "materiality_band", "period")
_OFF_CONTRACT_FIELD_RE = re.compile(rf"^{re.escape(scoring.OFF_CONTRACT_REASON)}: (\S+) states(?: |$)")
CATCH_ALL_ROW_ID = "S77"
CATCH_ALL_KEY = "*"

_SENTINEL = "\x00sentinel\x00"


def sweep_reason_prefix(kind: str) -> str:
    """The text a SweepHit of this kind writes before its token (the whole reason for
    a kind that writes no token), built from scoring.SweepHit, never retyped."""
    return scoring.SweepHit(_SENTINEL, kind).reason.split(repr(_SENTINEL))[0]


assert set(map(sweep_reason_prefix, scoring.SWEEP_KINDS)) == set(regression.SWEEP_REASON_PREFIXES)

# --- Row texts ----------------------------------------------------------------------------

ALL_FINDING_VERDICTS: tuple[str, ...] = FINDING_VERDICTS
CODE_ROW_VERDICTS: tuple[str, ...] = tuple(v for v in FINDING_VERDICTS if v != "citation_omission")

CORRECT_MEANS_SWEEP = "the flagged token is not an entity, or is a case-file entity cited in another form (D-E1)"
CORRECT_MEANS_NARRATIVE_CODE = ("the routed sentence states the case file truly for what the key checks; "
                                "the route was a false alarm")
CORRECT_MEANS_SCHEDULE_TICKER = "the cited ticker is the asset of a schedule item and naming it misstates nothing"
CORRECT_MEANS_OFF_CONTRACT = "the stated value is the true value written outside the output contract (D17, D-E3)"
CORRECT_MEANS_TYPE_LABEL = "the cited type names the item's own type in other words"
CORRECT_MEANS_INELIGIBLE_MENTION = "the narrative names the ineligible item without relying on it to explain the variance"
CORRECT_MEANS_INELIGIBLE_CITED = "entities_cited lists the item without relying on it"
CORRECT_MEANS_LANE_F_AMOUNT = ("the amount is the named item's own amount and the clause states its effect "
                               "and type truly")
CORRECT_MEANS_LANE_F_REFUSED = "the statement is the object of a refused request and claims nothing false"
CORRECT_MEANS_LANE_F_EXCLUSION = "the ticker is the excluded item's own asset inside an exclusion clause"

ON_CORRECT_SWEEP = ("fabrication anchor back to 3 if every finding on the anchor is correct, or citation_omission "
                    "on a name code confirms the case file holds, and no cited entity is fabricated (D-R9)")
ON_CORRECT_NARRATIVE_CODE = "structured Correctness kept"
ON_CORRECT_SCHEDULE_TICKER = "fabrication anchor back to 3 as for sweep"
ON_CORRECT_WITHHELD = "withheld point restored"
ON_CORRECT_PERIOD = "Correctness 0 lifted to the structured score; completeness point restored"
ON_CORRECT_SCORE_KEPT = "score kept"
ON_CORRECT_FAILS = "fails on code"
ON_CORRECT_ANSWER_KEY = "fails on code and is filed as an answer-key defect"
ON_CORRECT_EXTRACTOR = "fails on code and is filed as an extractor defect"

NOTE_LANE_F = "lane F, a783322"
NOTE_DRAFT_LACKED = "code emits it; draft lacked it"
NOTE_FIX4 = "fix round 4, 2026-11-09"
CORRECT_MEANS_BAND_SUBJECT = ("the band or threshold comparison is about the unexplained remainder or another "
                              "non-variance figure (including a rounded rendering of the variance) and is true of "
                              "that figure; the narrative states no wrong band for the variance itself")
CORRECT_MEANS_DIRECTION_THRESHOLD = ("'short of' compares the variance with a threshold or band; the narrative "
                                     "states no wrong direction")
CORRECT_MEANS_ASSET_WORD = ("the words that spell an asset name are plain words ('the true USD shortfall'), "
                            "not an asset claim; the narrative states no wrong asset")


def _settleable(row_id: str, match: str, key: str, route_class: str, source: str, anchor: str,
                code: str, family: str, test: str, correct_means: str, on_correct: str,
                verdicts: tuple[str, ...] = CODE_ROW_VERDICTS, note: str = "") -> SettlementRow:
    return SettlementRow(row_id, match, key, route_class, source, anchor, True, correct_means, on_correct,
                         code, verdicts, family, test, note)


def _unsettleable(row_id: str, match: str, key: str, route_class: str, source: str, anchor: str,
                  test: str, on_correct: str = ON_CORRECT_FAILS, note: str = "") -> SettlementRow:
    return SettlementRow(row_id, match, key, route_class, source, anchor, False, "", on_correct,
                         None, (), "none", test, note)


def _sweep_row(row_id: str, kind: str, note: str = "") -> SettlementRow:
    return _settleable(row_id, "prefix", f"{SWEEP}: {sweep_reason_prefix(kind)}", SWEEP, "narrative",
                       "fabrication", "FM-005", "sweep", f"test_settle_sweep_{kind}",
                       CORRECT_MEANS_SWEEP, ON_CORRECT_SWEEP, ALL_FINDING_VERDICTS, note)


def _narrative_code_row(row_id: str, kind: str, code: str, family: str, note: str = "",
                        match: str = "exact", test: str | None = None) -> SettlementRow:
    return _settleable(row_id, match, f"{_CR}{kind}", CODE, "narrative", "correctness", code, family,
                       test or f"test_settle_{kind}", CORRECT_MEANS_NARRATIVE_CODE, ON_CORRECT_NARRATIVE_CODE,
                       note=note)


def _pattern(*parts: str) -> str:
    """A regex from literal parts and the markers ANY (".+") and QUOTED (either repr quote)."""
    out = []
    for p in parts:
        if p == _ANY:
            out.append(".+")
        elif p == _QUOTED:
            out.append("(?:'.+'|\".+\")")
        else:
            out.append(re.escape(p))
    return "".join(out)


_ANY = "\x00any\x00"
_QUOTED = "\x00quoted\x00"

# Narrative code kinds by family, in table order (S21 .. S67).
AMOUNT_KINDS: tuple[str, ...] = (
    "amount_role_unresolved", "amount_role_unresolved_and_ungrounded", "aggregate_equals_item_subset_sum",
    "aggregate_total_role_cue_follows", "aggregate_total_unconfirmed", "balance_amount_equals_variance",
    "item_amount_equals_clause_sum", "item_amount_equals_explained_total", "item_effect_sum_unverified",
    "item_effect_unnamed_item", "item_effect_unverified", "parenthetical_item_amount",
    "role_amount_fits_competing_cue", "variance_amount_fits_other_role", "hedged_amount", "bare_number_",
    "negative_balance", "negative_threshold", "non_usd_amount", "amount_in_words", "unknown_currency_marker",
)
BARE_NUMBER_PREFIX = "bare_number_"
SPLIT_KINDS: tuple[str, ...] = (
    "split_attribution_with_remainder", "split_resolution_with_remainder", "split_claim_negated",
    "split_claim_subject_unresolved",
)
DIRECTION_KINDS: tuple[str, ...] = ("amount_sign_disagrees", "direction_negated", "both_directions_named")
BAND_KINDS: tuple[str, ...] = (
    "band_negated", "band_threshold_unresolved", "band_vocabulary_unresolved", "band_word_ambiguous",
    "severity_word_subject_unresolved", "escalation_claim_unresolved",
)
ASSET_KINDS: tuple[str, ...] = ("asset_negated", "gas_asset_named", "other_chain_named", "second_asset_named")
PERIOD_KINDS: tuple[str, ...] = (
    "period_negated", "period_date_ambiguous", "period_date_not_cutoff", "period_date_uncued_before_period",
    "quarter_period", "fiscal_or_half_year_period", "bare_month_other_period", "bare_year_other_period",
    "second_period_named",
)
NARRATIVE_CODE_FAMILIES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (AMOUNT_KINDS, "FM-001", "amount"),
    (SPLIT_KINDS, "FM-001", "split"),
    (DIRECTION_KINDS, "FM-003", "direction"),
    (BAND_KINDS, "FM-002", "band"),
    (ASSET_KINDS, "FM-016", "asset"),
    (PERIOD_KINDS, "FM-016", "period"),
)
# The draft settlement map's note column, for the rows that carried one.
_DRAFT_NOTES: dict[str, str] = {
    "item_amount_equals_clause_sum": "R1 fix round 2026-11-05",
    "hedged_amount": ("owner decision D9: a hedged figure within its stated precision becomes passable on a "
                      "correct verdict; before this change it always failed as a route"),
    "amount_sign_disagrees": "R3 fix round 2026-11-05",
}


def _build_table() -> tuple[SettlementRow, ...]:
    rows: list[SettlementRow] = [
        _sweep_row("S01", "capitalized_run",
                   note="correct means: not an entity or a grounded case-file entity cited in another form"),
        _sweep_row("S02", "organization_suffix"),
        _sweep_row("S03", "custodian_vocabulary"),
        _sweep_row("S04", "reference", note="an exact case-file reference is P12 and never routes"),
        _sweep_row("S05", "identifier"),
        _sweep_row("S06", "truncated_identifier"),
        _unsettleable("S07", "exact", f"{SWEEP}: {sweep_reason_prefix('format_character')}", SWEEP, "narrative",
                      "fabrication", "test_unsettleable_format_character",
                      note="hidden-text defense; a reader cannot see what the code saw"),
        _unsettleable("S08", "prefix", f"{SWEEP}: {sweep_reason_prefix('non_latin_letter')}", SWEEP, "narrative",
                      "fabrication", "test_unsettleable_non_latin_letter", note="homoglyph defense"),
        _unsettleable("S09", "exact", f"{_CR}entities_cited holds a blank entry", CODE, "structured",
                      "fabrication", "test_unsettleable_blank_entry",
                      note="a blank entry can be a string of format characters"),
        _settleable("S10", "pattern",
                    _pattern(f"{_CR}entities_cited names ", _QUOTED,
                             ", the asset of a schedule item, not the case's asset or a counterparty"),
                    CODE, "structured", "fabrication", "FM-016", "asset", "test_settle_schedule_ticker",
                    CORRECT_MEANS_SCHEDULE_TICKER, ON_CORRECT_SCHEDULE_TICKER,
                    note="owner decision: structured route review may settle"),
        _settleable("S11", "off_contract_field", "variance_amount_usd", CODE, "structured", "correctness",
                    "FM-001", "structured_field", "test_settle_off_contract_amount",
                    CORRECT_MEANS_OFF_CONTRACT, ON_CORRECT_WITHHELD,
                    note="code already proved the content equals the truth (D17)"),
        _settleable("S12", "off_contract_field", "materiality_band", CODE, "structured", "correctness",
                    "FM-002", "structured_field", "test_settle_off_contract_band",
                    CORRECT_MEANS_OFF_CONTRACT, ON_CORRECT_WITHHELD, note="D17"),
        _settleable("S13", "off_contract_field", "period", CODE, "structured", "correctness_and_completeness",
                    "FM-016", "structured_field", "test_settle_off_contract_period",
                    CORRECT_MEANS_OFF_CONTRACT, ON_CORRECT_PERIOD,
                    note="D17; period off-contract sets Correctness 0 in code today"),
        _settleable("S14", "pattern", _pattern(f"{_CR}reconciling item type differs from ", _ANY),
                    CODE, "structured", "reconciling_items", "FM-015", "reference_item", "test_settle_type_label",
                    CORRECT_MEANS_TYPE_LABEL, ON_CORRECT_SCORE_KEPT,
                    note="owner decision: optional type label (F3-03); 6 labeled rows ruled fine"),
        _settleable("S15", "pattern",
                    _pattern(f"{_CR}narrative names ineligible item ", _ANY,
                             " without stating whether it explains the variance"),
                    CODE, "narrative", "reconciling_items", "FM-015", "reference_item",
                    "test_settle_ineligible_mention", CORRECT_MEANS_INELIGIBLE_MENTION, ON_CORRECT_SCORE_KEPT),
        _settleable("S16", "pattern",
                    _pattern(f"{_CR}entities_cited names ineligible item ", _ANY,
                             " that the narrative does not discuss"),
                    CODE, "structured", "reconciling_items", "FM-015", "reference_item",
                    "test_settle_ineligible_cited_only", CORRECT_MEANS_INELIGIBLE_CITED, ON_CORRECT_SCORE_KEPT,
                    note="new line 37 asks for excluded items to be cited"),
        _unsettleable("S17", "exact",
                      f"{_CR}computed explained split disagrees with truth although every eligible item is cited",
                      CODE, "system", "reconciling_items", "test_unsettleable_answer_key", ON_CORRECT_ANSWER_KEY,
                      note="an eval defect not a model claim"),
        _unsettleable("S18", "prefix", f"{_CR}narrative extractor raised", CODE, "system", "correctness",
                      "test_unsettleable_extractor_raised", ON_CORRECT_EXTRACTOR,
                      note="no narrative claim was checked"),
        _unsettleable("S19", "exact", f"{_CR}{regression.UNSTATED_REASON}", CODE, "system", "any",
                      "test_unsettleable_unstated"),
        _unsettleable("S20", "exact", f"{_CR}non_latin_characters", CODE, "narrative", "correctness",
                      "test_unsettleable_non_latin_characters", note="homoglyph defense"),
    ]
    n = 21
    for kinds, code, family in NARRATIVE_CODE_FAMILIES:
        for kind in kinds:
            if kind == BARE_NUMBER_PREFIX:
                row = _narrative_code_row(f"S{n:02d}", kind, code, family, match="prefix",
                                          test="test_settle_bare_number")
            else:
                row = _narrative_code_row(f"S{n:02d}", kind, code, family, note=_DRAFT_NOTES.get(kind, ""))
            rows.append(row)
            n += 1
    rows += [
        _settleable("S68", "exact", f"{_CR}lane_f_amount_is_preceding_item_amount", CODE, "narrative",
                    "correctness", "FM-001", "amount", "test_settle_lane_f_amount_is_preceding_item_amount",
                    CORRECT_MEANS_LANE_F_AMOUNT, ON_CORRECT_NARRATIVE_CODE, note=NOTE_LANE_F),
        _settleable("S69", "exact", f"{_CR}lane_f_refused_request_object", CODE, "narrative", "correctness",
                    "downgraded_from", "statement", "test_settle_lane_f_refused_request_object",
                    CORRECT_MEANS_LANE_F_REFUSED, ON_CORRECT_NARRATIVE_CODE, note=NOTE_LANE_F),
        _settleable("S70", "exact", f"{_CR}lane_f_exclusion_context", CODE, "narrative", "correctness",
                    "FM-016", "asset", "test_settle_lane_f_exclusion_context",
                    CORRECT_MEANS_LANE_F_EXCLUSION, ON_CORRECT_NARRATIVE_CODE, note=NOTE_LANE_F),
        _unsettleable("S71", "exact", f"{_CR}narrative_too_long", CODE, "system", "correctness",
                      "test_unsettleable_narrative_too_long", note=NOTE_DRAFT_LACKED),
        _unsettleable("S72", "exact", f"{_CR}extractor_error", CODE, "system", "correctness",
                      "test_unsettleable_extractor_error", ON_CORRECT_EXTRACTOR, note=NOTE_DRAFT_LACKED),
        _unsettleable("S73", "exact", f"{_CR}format_characters_in_narrative", CODE, "narrative", "correctness",
                      "test_unsettleable_format_characters_in_narrative", note=NOTE_DRAFT_LACKED),
        _settleable("S74", "exact", f"{_CR}band_statement_subject_unresolved", CODE, "narrative", "correctness",
                    "FM-002", "band", "test_settle_band_statement_subject_unresolved",
                    CORRECT_MEANS_BAND_SUBJECT, ON_CORRECT_NARRATIVE_CODE, note=NOTE_FIX4),
        _settleable("S75", "exact", f"{_CR}direction_word_threshold_object", CODE, "narrative", "correctness",
                    "FM-003", "direction", "test_settle_direction_word_threshold_object",
                    CORRECT_MEANS_DIRECTION_THRESHOLD, ON_CORRECT_NARRATIVE_CODE, note=NOTE_FIX4),
        _settleable("S76", "exact", f"{_CR}asset_word_ambiguous", CODE, "narrative", "correctness",
                    "FM-016", "asset", "test_settle_asset_word_ambiguous",
                    CORRECT_MEANS_ASSET_WORD, ON_CORRECT_NARRATIVE_CODE, note=NOTE_FIX4),
        _unsettleable(CATCH_ALL_ROW_ID, "catch_all", CATCH_ALL_KEY, "any", "any", "any",
                      "test_every_route_key_has_a_settlement_row",
                      note="a meta-test scans every review() call site and fails when a key has no row"),
    ]
    return tuple(rows)


TABLE: tuple[SettlementRow, ...] = _build_table()
_BY_ID: dict[str, SettlementRow] = {r.row_id: r for r in TABLE}


def _check_table() -> None:
    assert [r.row_id for r in TABLE] == [f"S{i:02d}" for i in range(1, 78)], "row ids must run S01..S77"
    assert TABLE[-1].row_id == CATCH_ALL_ROW_ID and TABLE[-1].match == "catch_all"
    assert len({r.unit_test for r in TABLE}) == len(TABLE), "each row has its own unit test"
    for r in TABLE:
        assert r.match in MATCH_KINDS and r.route_class in ROUTE_CLASSES, r.row_id
        assert r.source in SOURCES and r.anchor in ANCHORS and r.facts_family in FACTS_FAMILIES, r.row_id
        if r.settleable:
            assert r.allowed_verdicts and set(r.allowed_verdicts) <= set(FINDING_VERDICTS), r.row_id
            assert r.review_found_code and r.correct_means and r.facts_family != "none", r.row_id
        else:
            assert r.allowed_verdicts == () and r.review_found_code is None and r.facts_family == "none", r.row_id
        if r.match == "off_contract_field":
            assert r.key in OFF_CONTRACT_FIELDS, r.row_id


_check_table()

# --- Lookup -----------------------------------------------------------------------------------


def row(row_id: str) -> SettlementRow:
    """The row with this id."""
    return _BY_ID[row_id]


def _key_hits(route_key: str) -> list[SettlementRow]:
    hits: list[SettlementRow] = []
    for r in TABLE[:-1]:
        if r.match == "exact":
            hit = route_key == r.key
        elif r.match == "off_contract_field":
            hit = route_key == OFF_CONTRACT_ROUTE_KEY
        elif r.match == "prefix":
            hit = route_key.startswith(r.key)
        else:  # pattern
            hit = re.fullmatch(r.key, route_key, re.DOTALL) is not None
        if hit:
            hits.append(r)
    return hits


def _off_contract_rows(reasons: Sequence[str]) -> tuple[SettlementRow, ...]:
    """One row per off-contract field the reasons name, in OFF_CONTRACT_FIELDS order; a
    reason on this key naming any other field (or none) adds the catch-all row."""
    named: set[str] = set()
    unknown = False
    for reason in reasons:
        reason = str(reason)
        if regression.route_key(reason) != OFF_CONTRACT_ROUTE_KEY:
            continue  # a reason of another key says nothing about this one
        m = _OFF_CONTRACT_FIELD_RE.match(reason)
        if m and m.group(1) in OFF_CONTRACT_FIELDS:
            named.add(m.group(1))
        else:
            unknown = True
    if not named and not unknown:
        raise ValueError(f"{OFF_CONTRACT_ROUTE_KEY!r} needs reasons that name the off-contract field")
    by_field = {r.key: r for r in TABLE if r.match == "off_contract_field"}
    out = [by_field[f] for f in OFF_CONTRACT_FIELDS if f in named]
    if unknown:
        out.append(TABLE[-1])
    return tuple(out)


def rows_for(route_key: str, reasons: Sequence[str] = ()) -> tuple[SettlementRow, ...]:
    """The settlement rows one route key takes. Exactly one row, except the
    off-contract key, which takes one row per field its reasons name. A key no row
    takes returns the catch-all alone. A key two rows take raises ValueError."""
    route_key = str(route_key)
    if isinstance(reasons, str):
        raise TypeError("reasons must be a sequence of reason strings, not one string")
    hits = _key_hits(route_key)
    groups = {("off_contract" if r.match == "off_contract_field" else r.row_id) for r in hits}
    if len(groups) > 1:
        raise ValueError(f"route key {route_key!r} hits more than one settlement row: "
                         f"{[r.row_id for r in hits]}")
    if not hits:
        return (TABLE[-1],)
    if hits[0].match == "off_contract_field":
        return _off_contract_rows(reasons)
    return (hits[0],)


def is_settleable(route_key: str, reasons: Sequence[str] = ()) -> bool:
    """True when every row the key takes is settleable (the catch-all never is)."""
    return all(r.settleable for r in rows_for(route_key, reasons))


def allowed_verdicts(route_key: str, reasons: Sequence[str] = ()) -> tuple[str, ...]:
    """The verdicts every row the key takes allows, in FINDING_VERDICTS order; () when
    any row is unsettleable."""
    rows = rows_for(route_key, reasons)
    return tuple(v for v in FINDING_VERDICTS if all(v in r.allowed_verdicts for r in rows))


def facts_family(route_key: str, reasons: Sequence[str] = ()) -> str:
    """The facts family of the first row the key takes."""
    return rows_for(route_key, reasons)[0].facts_family


def counts() -> dict[str, int]:
    settleable = [r for r in TABLE if r.settleable]
    return {
        "rows": len(TABLE),
        "settleable": len(settleable),
        "settleable_narrative": sum(r.source == "narrative" for r in settleable),
        "settleable_structured": sum(r.source == "structured" for r in settleable),
        "unsettleable": len(TABLE) - len(settleable),
    }


def table_rows_as_dicts() -> list[dict]:
    """Every row as a plain dict (tuples as lists), in table order."""
    out = []
    for r in TABLE:
        d = asdict(r)
        d["allowed_verdicts"] = list(r.allowed_verdicts)
        out.append(d)
    return out


def table_sha256() -> str:
    """sha256 of the canonical JSON of the table, recorded in packet-meta.json and the manifest."""
    text = json.dumps(table_rows_as_dicts(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
