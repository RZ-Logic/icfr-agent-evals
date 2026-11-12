"""Settlement table tests (pre-registration section 6, Review settlement).

One named test per row: each builds a representative reason, runs
regression.route_key on it and checks rows_for returns exactly that row with the
settleable flag, allowed verdicts, anchor, review-found code and facts family the
spec gives. The coverage test scans every reason emit site in narrative.py and
scoring.py by AST, so a new reason kind fails here until it has a row.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from icfr_agent_evals import narrative, regression, scoring, settlement
from icfr_agent_evals.review_schema import FINDING_VERDICTS
from icfr_agent_evals.scoring import SweepHit

NARRATIVE_SRC = Path(inspect.getfile(narrative)).read_text(encoding="utf-8")
SCORING_SRC = Path(inspect.getfile(scoring)).read_text(encoding="utf-8")

FIVE = ("correct", "wrong", "citation_omission", "cannot_settle", "not_reviewed")
FOUR = ("correct", "wrong", "cannot_settle", "not_reviewed")

# Typed from the spec's row list, independently of settlement.py:
# row id -> (settleable, source, anchor, review-found code, facts family, unit test)
_N, _S, _Y = "narrative", "structured", "system"
_C, _F, _R = "correctness", "fabrication", "reconciling_items"
SPEC_ROWS: dict[str, tuple] = {
    "S01": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_capitalized_run"),
    "S02": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_organization_suffix"),
    "S03": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_custodian_vocabulary"),
    "S04": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_reference"),
    "S05": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_identifier"),
    "S06": (True, _N, _F, "FM-005", "sweep", "test_settle_sweep_truncated_identifier"),
    "S07": (False, _N, _F, None, "none", "test_unsettleable_format_character"),
    "S08": (False, _N, _F, None, "none", "test_unsettleable_non_latin_letter"),
    "S09": (False, _S, _F, None, "none", "test_unsettleable_blank_entry"),
    "S10": (True, _S, _F, "FM-016", "asset", "test_settle_schedule_ticker"),
    "S11": (True, _S, _C, "FM-001", "structured_field", "test_settle_off_contract_amount"),
    "S12": (True, _S, _C, "FM-002", "structured_field", "test_settle_off_contract_band"),
    "S13": (True, _S, "correctness_and_completeness", "FM-016", "structured_field",
            "test_settle_off_contract_period"),
    "S14": (True, _S, _R, "FM-015", "reference_item", "test_settle_type_label"),
    "S15": (True, _N, _R, "FM-015", "reference_item", "test_settle_ineligible_mention"),
    "S16": (True, _S, _R, "FM-015", "reference_item", "test_settle_ineligible_cited_only"),
    "S17": (False, _Y, _R, None, "none", "test_unsettleable_answer_key"),
    "S18": (False, _Y, _C, None, "none", "test_unsettleable_extractor_raised"),
    "S19": (False, _Y, "any", None, "none", "test_unsettleable_unstated"),
    "S20": (False, _N, _C, None, "none", "test_unsettleable_non_latin_characters"),
    "S68": (True, _N, _C, "FM-001", "amount", "test_settle_lane_f_amount_is_preceding_item_amount"),
    "S69": (True, _N, _C, "downgraded_from", "statement", "test_settle_lane_f_refused_request_object"),
    "S70": (True, _N, _C, "FM-016", "asset", "test_settle_lane_f_exclusion_context"),
    "S71": (False, _Y, _C, None, "none", "test_unsettleable_narrative_too_long"),
    "S72": (False, _Y, _C, None, "none", "test_unsettleable_extractor_error"),
    "S73": (False, _N, _C, None, "none", "test_unsettleable_format_characters_in_narrative"),
    "S74": (True, _N, _C, "FM-002", "band", "test_settle_band_statement_subject_unresolved"),
    "S75": (True, _N, _C, "FM-003", "direction", "test_settle_direction_word_threshold_object"),
    "S76": (True, _N, _C, "FM-016", "asset", "test_settle_asset_word_ambiguous"),
    "S77": (False, "any", "any", None, "none", "test_every_route_key_has_a_settlement_row"),
}
SPEC_NARRATIVE_FAMILIES = (
    (("amount_role_unresolved", "amount_role_unresolved_and_ungrounded", "aggregate_equals_item_subset_sum",
      "aggregate_total_role_cue_follows", "aggregate_total_unconfirmed", "balance_amount_equals_variance",
      "item_amount_equals_clause_sum", "item_amount_equals_explained_total", "item_effect_sum_unverified",
      "item_effect_unnamed_item", "item_effect_unverified", "parenthetical_item_amount",
      "role_amount_fits_competing_cue", "variance_amount_fits_other_role", "hedged_amount", "bare_number_",
      "negative_balance", "negative_threshold", "non_usd_amount", "amount_in_words", "unknown_currency_marker"),
     "FM-001", "amount"),
    (("split_attribution_with_remainder", "split_resolution_with_remainder", "split_claim_negated",
      "split_claim_subject_unresolved"), "FM-001", "split"),
    (("amount_sign_disagrees", "direction_negated", "both_directions_named"), "FM-003", "direction"),
    (("band_negated", "band_threshold_unresolved", "band_vocabulary_unresolved", "band_word_ambiguous",
      "severity_word_subject_unresolved", "escalation_claim_unresolved"), "FM-002", "band"),
    (("asset_negated", "gas_asset_named", "other_chain_named", "second_asset_named"), "FM-016", "asset"),
    (("period_negated", "period_date_ambiguous", "period_date_not_cutoff", "period_date_uncued_before_period",
      "quarter_period", "fiscal_or_half_year_period", "bare_month_other_period", "bare_year_other_period",
      "second_period_named"), "FM-016", "period"),
)
NARRATIVE_KIND_OF_ROW: dict[str, str] = {}
_n = 21
for _kinds, _code, _family in SPEC_NARRATIVE_FAMILIES:
    for _kind in _kinds:
        _rid = f"S{_n:02d}"
        _test = "test_settle_bare_number" if _kind == "bare_number_" else f"test_settle_{_kind}"
        SPEC_ROWS[_rid] = (True, _N, _C, _code, _family, _test)
        NARRATIVE_KIND_OF_ROW[_rid] = _kind
        _n += 1
assert _n == 68 and len(SPEC_ROWS) == 77


def _representative_reasons(row_id: str) -> tuple[str, ...]:
    """Reasons as the code writes them, one or more per row."""
    sweep_kinds = {"S01": "capitalized_run", "S02": "organization_suffix", "S03": "custodian_vocabulary",
                   "S04": "reference", "S05": "identifier", "S06": "truncated_identifier"}
    if row_id in sweep_kinds:
        kind = sweep_kinds[row_id]
        return (SweepHit("Acme Holdings", kind).reason, SweepHit("O'Neil: Trust", kind).reason)
    if row_id in NARRATIVE_KIND_OF_ROW:
        kind = NARRATIVE_KIND_OF_ROW[row_id]
        if kind == "bare_number_":
            return ("bare_number_variance_amount_mismatch: 5000", "bare_number_amount_role_unresolved: 12")
        if kind == "both_directions_named":
            return (kind,)
        return (f"{kind}: $5,000", f"{kind}: two parts")
    fixed = {
        "S07": (SweepHit("\u200b", "format_character").reason,),
        "S08": (SweepHit("\u0430", "non_latin_letter").reason,),
        "S09": ("entities_cited holds a blank entry",),
        "S10": ("entities_cited names 'SOL', the asset of a schedule item, not the case's asset or a counterparty",
                "entities_cited names \"O'SOL\", the asset of a schedule item, not the case's asset or a "
                "counterparty"),
        "S11": (scoring._off_contract("variance_amount_usd"),),
        "S12": (scoring._off_contract("materiality_band"),),
        "S13": (scoring._off_contract("period"),),
        "S14": ("reconciling_items[0] 'DP-1' type 'deposit' differs from 'deposit_in_transit'",
                "reconciling_items[12] 'WD-2' type 'x' differs from \"o'clock\""),
        "S15": ("narrative names ineligible item DP-3 without stating whether it explains the variance",),
        "S16": ("entities_cited names ineligible item WD-9 that the narrative does not discuss",),
        "S17": ("computed explained split disagrees with truth although every eligible item is cited",),
        "S18": ("narrative extractor raised KeyError: 'x'", "narrative extractor raised ValueError: bad: worse"),
        "S19": (regression.UNSTATED_REASON,),
        "S20": ("non_latin_characters: \u0430\u0435",),
        "S68": ("lane_f_amount_is_preceding_item_amount: $1,250",),
        "S69": ("lane_f_refused_request_object: material",),
        "S70": ("lane_f_exclusion_context: USDC",),
        "S71": ("narrative_too_long: 9000 characters",),
        "S72": ("extractor_error: KeyError",),
        "S73": ("format_characters_in_narrative",),
        "S74": ("band_statement_subject_unresolved: unexplained remainder below the reportable floor",),
        "S75": ("direction_word_threshold_object: short of the M3 threshold",),
        "S76": ("asset_word_ambiguous: true USD",),
    }
    return fixed[row_id]


def _check_row(row_id: str) -> None:
    settleable, source, anchor, code, family, test = SPEC_ROWS[row_id]
    row = settlement.row(row_id)
    assert row.unit_test == test
    for reason in _representative_reasons(row_id):
        key = regression.route_key(reason)
        assert key is not None, reason
        got = settlement.rows_for(key, [reason])
        assert got == (row,), (reason, key, [r.row_id for r in got])
        assert row.settleable is settleable
        assert settlement.is_settleable(key, [reason]) is settleable
        if not settleable:
            expected_verdicts: tuple[str, ...] = ()
        elif row_id in ("S01", "S02", "S03", "S04", "S05", "S06"):
            expected_verdicts = FIVE
        else:
            expected_verdicts = FOUR
        assert row.allowed_verdicts == expected_verdicts
        assert settlement.allowed_verdicts(key, [reason]) == expected_verdicts
        assert row.source == source and row.anchor == anchor
        assert row.review_found_code == code
        assert row.facts_family == family == settlement.facts_family(key, [reason])
        assert regression.route_class(key) == row.route_class


def _make_row_test(row_id: str):
    def test() -> None:
        _check_row(row_id)
    test.__name__ = SPEC_ROWS[row_id][5]
    test.__doc__ = f"Settlement row {row_id}."
    return test


for _rid in sorted(SPEC_ROWS):
    if _rid != settlement.CATCH_ALL_ROW_ID:
        globals()[SPEC_ROWS[_rid][5]] = _make_row_test(_rid)


# --- Table shape --------------------------------------------------------------------------


def test_counts() -> None:
    assert settlement.counts() == {"rows": 77, "settleable": 66, "settleable_narrative": 60,
                                   "settleable_structured": 6, "unsettleable": 11}
    assert len({r.row_id for r in settlement.TABLE}) == 77
    assert len({r.unit_test for r in settlement.TABLE}) == 77
    assert [r.row_id for r in settlement.TABLE] == [f"S{i:02d}" for i in range(1, 78)]
    assert settlement.TABLE[-1].row_id == settlement.CATCH_ALL_ROW_ID == "S77"


def test_rows_follow_the_spec() -> None:
    for r in settlement.TABLE:
        settleable, source, anchor, code, family, test = SPEC_ROWS[r.row_id]
        assert (r.settleable, r.source, r.anchor, r.review_found_code, r.facts_family, r.unit_test) == (
            settleable, source, anchor, code, family, test), r.row_id
    for rid, kind in NARRATIVE_KIND_OF_ROW.items():
        r = settlement.row(rid)
        assert r.key == f"code_review: {kind}"
        assert r.match == ("prefix" if kind == "bare_number_" else "exact")


def test_every_row_test_is_collected() -> None:
    names = {r.unit_test for r in settlement.TABLE}
    module = globals()
    for name in names:
        assert callable(module.get(name)), name


def test_correct_means_and_on_correct_texts() -> None:
    sweep = "the flagged token is not an entity, or is a case-file entity cited in another form (D-E1)"
    code = ("the routed sentence states the case file truly for what the key checks; the route was a false alarm")
    for r in settlement.TABLE:
        if r.row_id in ("S01", "S02", "S03", "S04", "S05", "S06"):
            assert r.correct_means == sweep
            assert r.on_correct == ("fabrication anchor back to 3 if every finding on the anchor is correct, or "
                                    "citation_omission on a name code confirms the case file holds, and no cited "
                                    "entity is fabricated (D-R9)")
        elif r.row_id in NARRATIVE_KIND_OF_ROW:
            assert r.correct_means == code
            assert r.on_correct == "structured Correctness kept"
        elif not r.settleable:
            assert r.correct_means == ""
            assert r.on_correct.startswith("fails on code")
    assert settlement.row("S17").on_correct == "fails on code and is filed as an answer-key defect"
    assert settlement.row("S18").on_correct == settlement.row("S72").on_correct == (
        "fails on code and is filed as an extractor defect")
    assert settlement.row("S10").on_correct == "fabrication anchor back to 3 as for sweep"
    assert settlement.row("S11").on_correct == settlement.row("S12").on_correct == "withheld point restored"
    assert settlement.row("S13").on_correct == (
        "Correctness 0 lifted to the structured score; completeness point restored")
    for rid in ("S14", "S15", "S16"):
        assert settlement.row(rid).on_correct == "score kept"
    for rid in ("S68", "S69", "S70"):
        assert settlement.row(rid).on_correct == "structured Correctness kept"
        assert settlement.row(rid).note == "lane F, a783322"
    for rid in ("S71", "S72", "S73"):
        assert settlement.row(rid).note == "code emits it; draft lacked it"
    for rid in ("S74", "S75", "S76"):
        assert settlement.row(rid).on_correct == "structured Correctness kept"
        assert settlement.row(rid).note == "fix round 4, 2026-11-09"
    assert settlement.row("S10").correct_means == (
        "the cited ticker is the asset of a schedule item and naming it misstates nothing")
    assert settlement.row("S14").correct_means == "the cited type names the item's own type in other words"
    assert settlement.row("S69").correct_means == (
        "the statement is the object of a refused request and claims nothing false")


def test_no_dash_characters_in_table() -> None:
    for d in settlement.table_rows_as_dicts():
        for v in d.values():
            text = str(v)
            assert "\u2014" not in text and "\u2013" not in text


def test_sweep_keys_come_from_sweephit() -> None:
    # Built from scoring.SweepHit, the same way regression.SWEEP_REASON_PREFIXES is.
    for kind in scoring.SWEEP_KINDS:
        prefix = settlement.sweep_reason_prefix(kind)
        assert prefix in regression.SWEEP_REASON_PREFIXES
    rows = {r.row_id: r for r in settlement.TABLE}
    assert rows["S07"].key == "narrative_sweep: " + SweepHit("x", "format_character").reason
    src = Path(inspect.getfile(settlement)).read_text(encoding="utf-8")
    assert "not in the structured citations" not in src  # never retyped


# --- Lookup rules -------------------------------------------------------------------------


def _off(field: str) -> str:
    return scoring._off_contract(field)


def test_off_contract_one_row_per_field_in_field_order() -> None:
    key = settlement.OFF_CONTRACT_ROUTE_KEY
    reasons = [_off("period"), _off("variance_amount_usd"), _off("materiality_band"), _off("period")]
    got = settlement.rows_for(key, reasons)
    assert [r.row_id for r in got] == ["S11", "S12", "S13"]
    assert settlement.is_settleable(key, reasons)
    assert settlement.allowed_verdicts(key, reasons) == FOUR
    assert settlement.facts_family(key, reasons) == "structured_field"


def test_off_contract_other_field_adds_catch_all() -> None:
    key = settlement.OFF_CONTRACT_ROUTE_KEY
    reasons = [_off("period"), _off("asset")]
    got = settlement.rows_for(key, reasons)
    assert [r.row_id for r in got] == ["S13", "S77"]
    assert not settlement.is_settleable(key, reasons)
    assert settlement.allowed_verdicts(key, reasons) == ()
    # an off-contract reason that names no field in the known form is fail safe too
    assert [r.row_id for r in settlement.rows_for(key, ["off_contract_value: garbled"])] == ["S77"]


def test_off_contract_without_field_raises() -> None:
    key = settlement.OFF_CONTRACT_ROUTE_KEY
    with pytest.raises(ValueError):
        settlement.rows_for(key)
    with pytest.raises(ValueError):
        settlement.rows_for(key, ["hedged_amount: $5"])  # a reason of another key names no field


def test_unlisted_key_takes_catch_all() -> None:
    for key in ("code_review: some_new_kind", "narrative_sweep: something new", "not a route key", ""):
        got = settlement.rows_for(key)
        assert got == (settlement.TABLE[-1],)
        assert not settlement.is_settleable(key)
        assert settlement.allowed_verdicts(key) == ()
        assert settlement.facts_family(key) == "none"


def test_near_keys_do_not_hit_exact_rows() -> None:
    # exact rows need the whole key; a longer or shorter key is unlisted
    for key in ("code_review: hedged_amount_extra", "code_review: hedged", "code_review: hedged_amount "):
        assert settlement.rows_for(key) == (settlement.TABLE[-1],)


def test_reasons_must_be_a_sequence() -> None:
    with pytest.raises(TypeError):
        settlement.rows_for(settlement.OFF_CONTRACT_ROUTE_KEY, _off("period"))


def test_two_row_hit_raises(monkeypatch) -> None:
    extra = dataclasses.replace(settlement.row("S35"), row_id="S35b", match="prefix", key="code_review: hedged")
    monkeypatch.setattr(settlement, "TABLE", settlement.TABLE[:-1] + (extra, settlement.TABLE[-1]))
    with pytest.raises(ValueError):
        settlement.rows_for("code_review: hedged_amount")


def test_no_row_overlap() -> None:
    for r in settlement.TABLE[:-1]:
        for reason in _representative_reasons(r.row_id) if r.row_id in SPEC_ROWS else ():
            key = regression.route_key(reason)
            hits = settlement._key_hits(key)
            if r.match == "off_contract_field":
                assert {h.match for h in hits} == {"off_contract_field"}, key
            else:
                assert hits == [r], (key, [h.row_id for h in hits])


def test_table_sha_stable(monkeypatch) -> None:
    sha = settlement.table_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert settlement.table_sha256() == sha
    assert settlement._build_table() == settlement.TABLE
    target = settlement.TABLE[20]
    for f in dataclasses.fields(settlement.SettlementRow):
        value = getattr(target, f.name)
        if isinstance(value, bool):
            new = not value
        elif isinstance(value, tuple):
            new = value + ("x",)
        elif value is None:
            new = "x"
        else:
            new = value + "x"
        changed = dataclasses.replace(target, **{f.name: new})
        monkeypatch.setattr(settlement, "TABLE",
                            settlement.TABLE[:20] + (changed,) + settlement.TABLE[21:])
        assert settlement.table_sha256() != sha, f.name
        monkeypatch.undo()
    assert settlement.table_sha256() == sha


def test_table_rows_as_dicts_is_plain_json() -> None:
    import json
    rows = settlement.table_rows_as_dicts()
    assert len(rows) == 77
    assert json.loads(json.dumps(rows)) == rows
    assert set(rows[0]) == {f.name for f in dataclasses.fields(settlement.SettlementRow)}


def test_settlement_sets_no_verdict() -> None:
    # pure data plus lookup: no function takes or returns a verdict for a finding
    src = Path(inspect.getfile(settlement)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not any("verdict" in f and f != "allowed_verdicts" for f in funcs)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imported <= {"annotations", "regression", "scoring", "FINDING_VERDICTS", "asdict", "dataclass",
                        "Sequence"}


# --- Coverage: every reason kind the code can emit has one row ------------------------------

# FM-only reason values: they ride with FM-001 and reach review only as bare_number_<value>.
FM_ONLY_REASONS = frozenset({
    "variance_amount_mismatch", "threshold_amount_mismatch", "item_amount_mismatch",
    "ledger_amount_mismatch", "onchain_amount_mismatch", "explained_amount_mismatch",
    "unexplained_amount_mismatch",
    "reference_not_in_fixture", "reference_not_cited", "reference_ineligible",
})
FM_ONLY_ROLE_FSTRING = "{role}_amount_mismatch"
FM_ONLY_ROLES = ("ledger", "onchain", "explained", "unexplained")
# Measured at a783322. A new emit site fails here until it is classified.
NARRATIVE_REVIEW_CALL_SITES = 36  # 28 at a783322; fix round 4 added 8
NARRATIVE_VARIABLE_HEADS = ("rec['reason']", "rec['reason']", "b['reason']", "why",
                            # fix round 4 and 4b guards: their keys are read from the guards' returns
                            "y1", "x1", "why", "x1", "x2[0]", "y2", "why", "why")
# Fix round 4 and 4b guard functions; every string they return is a reason kind, except the
# internal marker below, which the band guard returns with emit=False.
FIX4_GUARD_PREFIXES = ("_x1_", "_x2_", "_y1_", "_y2_")
FIX4_INTERNAL_MARKERS = frozenset({"gloss_of_label"})
SCORING_REASON_EMIT_SITES = 14


def _head(text: str) -> str:
    """The reason kind: the text before the first ": ", or the whole text."""
    return text.split(": ", 1)[0]


def _const(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _fstring_head(node: ast.JoinedStr) -> str | None:
    """The leading constant of an f-string up to ": ", or None when it starts with a value."""
    first = node.values[0] if node.values else None
    text = _const(first) if first is not None else None
    return _head(text) if text is not None else None


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _narrative_kinds() -> dict[str, set[str]]:
    """Reason kinds narrative.py can emit, by source of the evidence."""
    tree = ast.parse(NARRATIVE_SRC)
    out: dict[str, set[str]] = {"review_heads": set(), "reason_literals": set(), "route_reason": set(),
                                "lane_f": set(), "result_heads": set(), "bare_prefix": set(),
                                "fstring_reasons": set(), "why_literals": set(), "fix4_guards": set()}
    variable_heads: list[str] = []
    review_sites = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "review" and isinstance(node.func, ast.Name):
                review_sites += 1
                arg = node.args[0]
                if _const(arg) is not None:
                    out["review_heads"].add(_head(_const(arg)))
                elif isinstance(arg, ast.JoinedStr):
                    head = _fstring_head(arg)
                    if head is None:
                        variable_heads.append(ast.unparse(arg.values[0].value))
                    else:
                        out["review_heads"].add(head)
                elif isinstance(arg, ast.BinOp) and _const(arg.left) is not None:
                    out["review_heads"].add(_head(_const(arg.left)))
                else:
                    raise AssertionError(f"unclassified review() argument: {ast.unparse(arg)}")
            for kw in node.keywords:
                if kw.arg == "reason":
                    if _const(kw.value) is not None:
                        out["reason_literals"].add(_const(kw.value))
                    elif isinstance(kw.value, ast.JoinedStr):
                        out["fstring_reasons"].add(ast.unparse(kw.value))
                    elif isinstance(kw.value, ast.BinOp) and _const(kw.value.left) is not None:
                        out["bare_prefix"].add(_const(kw.value.left))
                if kw.arg == "route_reason" and _const(kw.value) is not None:
                    out["route_reason"].add(_const(kw.value))
                if kw.arg == "review_reasons" and name == "NarrativeResult" and isinstance(kw.value, ast.Tuple):
                    for elt in kw.value.elts:
                        if _const(elt) is not None:
                            out["result_heads"].add(_head(_const(elt)))
                        elif isinstance(elt, ast.JoinedStr) and _fstring_head(elt) is not None:
                            out["result_heads"].add(_fstring_head(elt))
            if name == "NarrativeResult":
                for kw in node.keywords:
                    if kw.arg == "review_reasons" and isinstance(kw.value, ast.BinOp):
                        left = kw.value.left
                        if isinstance(left, ast.Tuple):
                            for elt in left.elts:
                                if _const(elt) is not None:
                                    out["result_heads"].add(_head(_const(elt)))
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None and _const(k) == "reason" and _const(v) is not None:
                    out["reason_literals"].add(_const(v))
        elif isinstance(node, ast.FunctionDef) and node.name.startswith(FIX4_GUARD_PREFIXES):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    val = sub.value
                    if isinstance(val, ast.Tuple) and val.elts:
                        val = val.elts[0]
                    vals = [val.body, val.orelse] if isinstance(val, ast.IfExp) else [val]
                    for v in vals:
                        if _const(v) is not None and _const(v) not in FIX4_INTERNAL_MARKERS:
                            out["fix4_guards"].add(_const(v))
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("_lane_f_"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    val = sub.value
                    if isinstance(val, ast.Tuple) and val.elts:
                        val = val.elts[0]
                    if _const(val) is not None:
                        out["lane_f"].add(_const(val))
        elif isinstance(node, ast.Assign):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "why" and _const(node.value) is not None):
                out["why_literals"].add(_const(node.value))
    assert review_sites == NARRATIVE_REVIEW_CALL_SITES, review_sites
    assert sorted(variable_heads) == sorted(NARRATIVE_VARIABLE_HEADS), variable_heads
    return out


def _render(node: ast.JoinedStr) -> str:
    """An f-string rendered with a plausible value per field, conversion kept."""
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value))
            continue
        src = ast.unparse(v.value)
        if src == "n":
            value = "3"
        elif "__name__" in src:
            value = "KeyError"
        else:
            value = "DP-1"
        parts.append(repr(value) if v.conversion == ord("r") else value)
    return "".join(parts)


def _scoring_reasons() -> tuple[list[str], set[str], int]:
    """(rendered reason strings, off-contract field names, emit-site count) from scoring.py."""
    tree = ast.parse(SCORING_SRC)
    reasons: list[str] = []
    fields: set[str] = set()
    sites = 0

    def collect(expr: ast.AST) -> None:
        if _const(expr) is not None:
            reasons.append(_const(expr))
        elif isinstance(expr, ast.JoinedStr):
            reasons.append(_render(expr))
        elif isinstance(expr, (ast.List, ast.Tuple)):
            for elt in expr.elts:
                collect(elt)
        elif isinstance(expr, ast.ListComp):
            collect(expr.elt)
        elif isinstance(expr, ast.IfExp):
            collect(expr.body)
            collect(expr.orelse)
        elif isinstance(expr, ast.Starred):
            collect(expr.value)
        elif isinstance(expr, ast.Call) and _call_name(expr) == "_off_contract":
            if expr.args and _const(expr.args[0]) is not None:
                fields.add(_const(expr.args[0]))
        # names, attributes and other calls are resolved by their own emit sites:
        # h.reason (SweepHit kinds), unclear (ineligible_reliance), nar.review_reasons (narrative)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if isinstance(target, ast.Name) and target.id == "reasons" and node.func.attr in (
                    "append", "insert", "extend"):
                sites += 1
                collect(node.args[-1])
            if isinstance(target, ast.Name) and target.id == "off_contract" and node.func.attr == "append":
                if _const(node.args[0]) is not None:
                    fields.add(_const(node.args[0]))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "reasons":
            sites += 1
            collect(node.value)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "reasons" for t in targets) and node.value is not None:
                sites += 1
                collect(node.value)
    return reasons, fields, sites


def _assert_one_row(reason: str, *, reasons: tuple[str, ...] | None = None) -> settlement.SettlementRow:
    key = regression.route_key(reason)
    assert key is not None, reason
    rows = settlement.rows_for(key, list(reasons or (reason,)))
    assert len(rows) == 1 and rows[0].row_id != settlement.CATCH_ALL_ROW_ID, (reason, key, rows)
    return rows[0]


def test_every_route_key_has_a_settlement_row() -> None:
    kinds = _narrative_kinds()
    # narrative kinds: review heads, reason literals, route reasons, lane F returns, result heads
    narrative_kinds = (kinds["review_heads"] | kinds["reason_literals"] | kinds["route_reason"]
                       | kinds["lane_f"] | kinds["result_heads"] | kinds["fix4_guards"])
    assert {"band_statement_subject_unresolved", "direction_word_threshold_object",
            "asset_word_ambiguous"} <= kinds["fix4_guards"]
    assert {"aggregate_total_role_cue_follows", "aggregate_total_unconfirmed"} <= kinds["route_reason"]
    assert {"narrative_too_long", "extractor_error", "format_characters_in_narrative"} <= kinds["result_heads"]
    assert {"lane_f_amount_is_preceding_item_amount", "lane_f_refused_request_object",
            "lane_f_exclusion_context"} <= kinds["lane_f"]
    assert kinds["bare_prefix"] == {"bare_number_"}
    assert kinds["fstring_reasons"] == {"f'{role}_amount_mismatch'"}
    assert kinds["why_literals"] == {"reference_not_in_fixture", "reference_not_cited", "reference_ineligible"}
    seen_rows: set[str] = set()
    fm_only_seen: set[str] = set()
    for kind in sorted(narrative_kinds):
        if kind in FM_ONLY_REASONS:
            fm_only_seen.add(kind)
            continue
        for reason in (f"{kind}: $5,000", kind):
            seen_rows.add(_assert_one_row(reason).row_id)
    # FM-only values and the f-string role values reach review only under bare_number_
    fm_only = set(FM_ONLY_REASONS) | kinds["why_literals"] | {f"{r}_amount_mismatch" for r in FM_ONLY_ROLES}
    assert fm_only == FM_ONLY_REASONS
    assert fm_only_seen <= FM_ONLY_REASONS
    for value in sorted(fm_only | narrative_kinds):
        assert _assert_one_row(f"bare_number_{value}: 5000").row_id == "S36"
        seen_rows.add("S36")
    # scoring.py reasons, off-contract fields and sweep kinds
    reasons, fields, sites = _scoring_reasons()
    assert sites == SCORING_REASON_EMIT_SITES, sites
    assert fields == {"variance_amount_usd", "materiality_band", "period"}
    for reason in reasons:
        seen_rows.add(_assert_one_row(reason).row_id)
    for field in sorted(fields):
        seen_rows.add(_assert_one_row(scoring._off_contract(field)).row_id)
    for kind in scoring.SWEEP_KINDS:
        for token in ("Acme Holdings", "\u200b", "\u0430"):
            seen_rows.add(_assert_one_row(SweepHit(token, kind).reason).row_id)
    seen_rows.add(_assert_one_row(regression.UNSTATED_REASON).row_id)
    seen_rows.add(_assert_one_row("narrative extractor raised KeyError: 'x'").row_id)
    # review=True rides only beside amount_role_unresolved_and_ungrounded
    tree = ast.parse(NARRATIVE_SRC)
    review_true = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kws = {kw.arg: kw.value for kw in node.keywords}
            v = kws.get("review")
            if isinstance(v, ast.Constant) and v.value is True:
                review_true.append(_const(kws.get("reason")) if "reason" in kws else None)
    assert review_true == ["amount_role_unresolved_and_ungrounded"]
    # every non-catch-all row is reached by some emit site
    assert seen_rows == {r.row_id for r in settlement.TABLE[:-1]}, sorted(
        {r.row_id for r in settlement.TABLE[:-1]} - seen_rows)


def test_catch_all_reason_shapes_never_settle() -> None:
    # fail safe: reasons that do not fit a row land on S77 and fail on code
    for reason in ("brand_new_kind: x", "something the code never wrote"):
        key = regression.route_key(reason)
        assert settlement.rows_for(key, [reason]) == (settlement.TABLE[-1],)
        assert not settlement.is_settleable(key, [reason])
