"""Looked-up facts for the grouped review packet (build spec section 6).

Runs on the 20 pilot cases of data/dev/dev.jsonl plus hand-built worker outputs.
No log, no model, no network.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import re
from pathlib import Path

import pytest
from inspect_ai.dataset import Sample

from icfr_agent_evals import casefile, generator, regression, review_facts as rf, review_schema, scoring
from icfr_agent_evals.dataset import is_pilot

ROOT = Path(__file__).resolve().parents[1]
DEV_FILE = ROOT / "data" / "dev" / "dev.jsonl"
SOURCE = Path(rf.__file__).read_text(encoding="utf-8")


def _pilot_cases() -> list[dict]:
    rows = [json.loads(line) for line in DEV_FILE.read_text(encoding="utf-8").splitlines() if line]
    samples = [Sample(input=r["input"], id=r["id"], metadata=r["metadata"]) for r in rows]
    return [s.metadata for s in samples if is_pilot(s)]


PILOT = _pilot_cases()
PILOT_IDS = [md["truth"]["case_id"] for md in PILOT]


def _parsed(md: dict) -> dict:
    """A hand-built output: true fields, the eligible items cited, one organization."""
    t = casefile.truth_of(md)
    label = md["fixture"]["wallets"][0]["label"]
    items = [
        {"reference": i.reference, "amount_usd": i.amount_usd or 0.0, "type": casefile.TYPE_LABELS[i.type][0]}
        for i in t.eligible_items()
    ]
    if not items:
        i = t.items[0]
        items = [{"reference": i.reference, "amount_usd": i.amount_usd or 0.0}]
    return {
        "asset": t.asset,
        "period": t.period,
        "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction,
        "materiality_band": t.materiality_band,
        "entities_cited": [t.organizations[0], label],
        "reconciling_items": items,
        "narrative": f"The variance sits in {label}. Item {t.items[0].reference} is open.",
    }


def _family_inputs(md: dict) -> dict[str, tuple[list[str], list[str]]]:
    """Representative reasons and routed sentences for every facts family."""
    t = casefile.truth_of(md)
    ref = t.items[0].reference
    label = md["fixture"]["wallets"][0]["label"]
    sentence = f"The variance of $1,000.00 is explained by {ref} held in {label}."
    return {
        "amount": ([f"amount_role_unresolved: $1,000.00"], [sentence]),
        "split": (["split_attribution_with_remainder: Explained variance"], [sentence]),
        "direction": (["direction_negated: surplus"], [sentence]),
        "band": (["band_negated: M1"], [sentence]),
        "asset": (["asset_negated: USDC",
                   "entities_cited names 'USDC', the asset of a schedule item, not the case's asset or a counterparty"],
                  [sentence]),
        "period": (["period_negated: 2025-12"], [sentence]),
        "statement": (["lane_f_refused_request_object: M1"], [sentence]),
        "sweep": ([scoring.SweepHit("Two Hot", "capitalized_run").reason], [sentence]),
        "reference_item": ([f"reconciling_items[0] {ref!r} type 'fee' differs from {t.items[0].type!r}",
                            f"narrative names ineligible item {ref} without stating whether it explains the variance",
                            f"entities_cited names ineligible item {ref} that the narrative does not discuss"], []),
        "structured_field": ([f"{scoring.OFF_CONTRACT_REASON}: {f} states the true value in a form outside "
                              "the output contract" for f in ("period", "variance_amount_usd", "materiality_band")],
                             []),
        "none": (["narrative_too_long: 4000 characters"], []),
    }


def _all_blocks(md: dict) -> list[rf.FactBlock]:
    parsed = _parsed(md)
    blocks = list(rf.case_facts(md)) + list(rf.output_facts(md, parsed)) + list(rf.output_facts(md, None))
    for family, (reasons, sentences) in _family_inputs(md).items():
        for p in (parsed, None):
            blocks += rf.finding_facts(md, p, family, reasons, sentences)
    return blocks


def _check_shape(blocks) -> None:
    for b in blocks:
        assert isinstance(b, rf.FactBlock)
        assert b.block_id in rf.BLOCK_IDS
        assert b.title == rf.TITLES[b.block_id]
        assert all(isinstance(c, str) for c in b.columns)
        assert set(b.verbatim_columns) <= set(b.columns)
        for row in b.rows:
            assert len(row) == len(b.columns)
            assert all(isinstance(cell, str) for cell in row)


# --- Shape --------------------------------------------------------------------------


def test_twenty_pilot_cases():
    assert len(PILOT) == 20


def test_families_constant():
    assert rf.FACTS_FAMILIES == ("amount", "split", "direction", "band", "asset", "period", "statement",
                                 "sweep", "reference_item", "structured_field", "none")
    assert set(_family_inputs(PILOT[0])) == set(rf.FACTS_FAMILIES)


@pytest.mark.parametrize("md", PILOT, ids=PILOT_IDS)
def test_every_family_returns_string_cells(md):
    parsed = _parsed(md)
    _check_shape(rf.case_facts(md))
    _check_shape(rf.output_facts(md, parsed))
    for family, (reasons, sentences) in _family_inputs(md).items():
        for p in (parsed, None):
            blocks = rf.finding_facts(md, p, family, reasons, sentences)
            _check_shape(blocks)
            if family == "none":
                assert blocks == ()
            else:
                assert [b.block_id for b in blocks[:2]] == ["named_references", "named_wallets"]
    json.dumps(rf.facts_payload(_all_blocks(md)))


def test_case_facts_block_order():
    assert [b.block_id for b in rf.case_facts(PILOT[0])] == [
        "answer_key", "thresholds", "items", "wallets", "organizations", "notes"]


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        rf.finding_facts(PILOT[0], None, "wallet", [], [])


def test_block_rejects_bad_rows():
    with pytest.raises(ValueError):
        rf.FactBlock("items", "x", ("a", "b"), (("1",),), frozenset())
    with pytest.raises(ValueError):
        rf.FactBlock("items", "x", ("a",), ((1,),), frozenset())
    with pytest.raises(ValueError):
        rf.FactBlock("nope", "x", ("a",), (), frozenset())
    with pytest.raises(ValueError):
        rf.FactBlock("items", "x", ("a",), (), frozenset({"b"}))


@pytest.mark.parametrize("md", PILOT[:3], ids=PILOT_IDS[:3])
def test_from_dict_round_trip(md):
    for b in _all_blocks(md):
        again = rf.FactBlock.from_dict(json.loads(json.dumps(b.to_dict())))
        assert again == b


# --- No verdicts ------------------------------------------------------------------------


@pytest.mark.parametrize("md", PILOT, ids=PILOT_IDS)
def test_no_verdict_words(md):
    assert rf.verdict_word_hits(_all_blocks(md)) == []


def test_no_verdict_words_static_strings():
    for title in rf.TITLES.values():
        assert review_schema.forbidden_word_hits(title) == [], title
    for label in rf.ELIGIBILITY_LABELS.values():
        assert review_schema.forbidden_word_hits(label) == [], label
    assert review_schema.forbidden_word_hits(rf.NOT_PARSED) == []


def test_verbatim_note_allowed_tool_title_caught():
    md = json.loads(json.dumps(PILOT[0]))
    md["fixture"]["notes"].append({"by": "A. Reviewer", "text": "This figure looks wrong to me.",
                                   "ts": "2026-01-02T09:00:00Z"})
    blocks = rf.case_facts(md)
    notes = [b for b in blocks if b.block_id == "notes"][0]
    assert any("wrong" in cell for row in notes.rows for cell in row)
    assert rf.verdict_word_hits(blocks) == []
    bad = rf.FactBlock("answer_key", "Values that match the narrative", ("quantity",), (("asset",),), frozenset())
    assert rf.verdict_word_hits([bad]) == ["match"]
    bad_col = rf.FactBlock("answer_key", "Answer key values", ("closest amount",), (("x",),), frozenset())
    assert rf.verdict_word_hits([bad_col]) == ["closest"]
    bad_cell = rf.FactBlock("answer_key", "Answer key values", ("quantity",), (("likely",),), frozenset())
    assert rf.verdict_word_hits([bad_cell]) == ["likely"]


COMPARISON_PARTS = tuple(review_schema.FORBIDDEN_FACT_WORDS) + ("diff", "delta", "gap", "vs", "against",
                                                                 "near", "similar")


def test_no_comparison_fields():
    seen_cols: set[str] = set()
    for md in PILOT:
        for b in _all_blocks(md):
            seen_cols.update(b.columns)
    names = set(rf.BLOCK_IDS) | seen_cols
    for name in names:
        low = name.lower()
        hits = [p for p in COMPARISON_PARTS if p in low]
        assert not hits, (name, hits)


# --- Never shown ------------------------------------------------------------------------

NEVER_SHOWN = ("role", "channel", "description", "injection", "category", "evidence_state",
               "has_memo_only_link", "template_id", "template_instance", "asset_tier", "reserve_ratio",
               "presentation", "noise", "pressed_claim", "decoy_reasons", "truth_rule")


def test_never_shown_fields_ast():
    tree = ast.parse(SOURCE)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in NEVER_SHOWN + ("notes",):
            bad.append(f"attribute .{node.attr} at line {node.lineno}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in NEVER_SHOWN:
            bad.append(f"string {node.value!r} at line {node.lineno}")
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
            bad.append(f"getattr at line {node.lineno}")
    assert not bad, bad


def _leaves(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _leaves(k) + _leaves(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _leaves(v)]
    return []


@pytest.mark.parametrize("md", PILOT, ids=PILOT_IDS)
def test_never_shown_values(md):
    roles = set(casefile.ITEM_ROLES)
    note_leaves = {s for s in _leaves(md["truth"]["notes"]) if s.strip()}
    # Generator identifiers (with an underscore) must not appear inside any tool-written cell.
    identifiers = {s for s in note_leaves if "_" in s and " " not in s}
    for b in _all_blocks(md):
        for row in b.rows:
            for col, cell in zip(b.columns, row):
                if col in b.verbatim_columns:
                    continue
                assert cell not in roles, (b.block_id, col, cell)
                assert cell not in note_leaves, (b.block_id, col, cell)
                for ident in identifiers:
                    assert not re.search(rf"(?<![\w-]){re.escape(ident)}(?![\w-])", cell), (b.block_id, col, ident)


def test_never_shown_values_are_present_to_hide():
    # The test above is not vacuous: the pilot answer keys do carry these labels.
    md = PILOT[0]
    assert any(i["role"] == "decoy" for i in md["truth"]["items"]) or md["truth"]["notes"].get("decoy_reasons")
    joined = json.dumps(rf.facts_payload(_all_blocks(md)))
    assert '"decoy"' not in joined and "explaining" not in joined


def test_note_fields_limited_to_ts_by_text():
    md = json.loads(json.dumps(PILOT[0]))
    md["fixture"]["notes"] = [{"by": "X", "text": "plain note", "ts": "t", "extra_field": "HIDDEN-VALUE"}]
    joined = json.dumps(rf.facts_payload(_all_blocks(md)))
    assert "HIDDEN-VALUE" not in joined


# --- Order and lookups ------------------------------------------------------------------


@pytest.mark.parametrize("md", PILOT, ids=PILOT_IDS)
def test_fixture_order(md):
    t = casefile.truth_of(md)
    want = [i.reference for i in t.items]
    items = [b for b in rf.case_facts(md) if b.block_id == "items"][0]
    assert [r[0] for r in items.rows] == want
    biggest = max((i.amount_usd or 0.0) for i in t.items)
    for amount in (0.01, biggest, 1e12):
        reasons = [f"amount_role_unresolved: ${amount:,.2f}"]
        blocks = rf.finding_facts(md, _parsed(md), "amount", reasons, [f"The amount is ${amount:,.2f}."])
        roles = [b for b in blocks if b.block_id == "role_targets"][0]
        assert [r[1] for r in roles.rows if r[0] == "item amount"] == want
        period = rf.finding_facts(md, None, "period", ["period_negated: 2025-12"], [])
        assert [r[0] for r in [b for b in period if b.block_id == "period_dates"][0].rows] == want


def _case(case_id: str) -> dict:
    return next(md for md in PILOT if md["truth"]["case_id"] == case_id)


def test_wallet_lookup_only():
    md = _case("dev-hc-007-SOL-M2")
    labels = [w["label"] for w in md["fixture"]["wallets"]]
    assert "HOT01" in labels
    assert rf.named_wallet_labels("HOT01 holds the deposit.", md) == ("HOT01",)
    assert rf.named_wallet_labels("HOT011 holds the deposit.", md) == ()
    assert rf.named_wallet_labels("hot01 holds it", md) == ()
    assert rf.named_wallet_labels("XHOT01 and HOT01x", md) == ()
    text = "Moved from WARM01 to HOT01, then back to WARM01."
    assert rf.named_wallet_labels(text, md) == ("WARM01", "HOT01")
    low = SOURCE.lower()
    for word in ("proven", "contradiction", "settle"):
        assert word not in low, word
    for rule in ("W1", "W2", "W3", "W4", "W5", "W6"):
        assert not re.search(rf"\b{rule}\b", SOURCE), rule
    assert "wallets.py" not in SOURCE


def test_wallet_labels_with_spaces_and_hyphens():
    spaced = _case("dev-hc-001-BTC-M2")
    assert rf.named_wallet_labels("Hot 1 sent to Vault 1 and Hot 10.", spaced) == ("Hot 1", "Vault 1")
    hyphen = _case("dev-std-004-SOL-M1")
    assert rf.named_wallet_labels("from hot-1 to cold-1; not hot-10 or hot-1x", hyphen) == ("hot-1", "cold-1")


def test_named_wallets_block_rows():
    md = _case("dev-hc-007-SOL-M2")
    blocks = rf.finding_facts(md, None, "direction", ["direction_negated: surplus"], ["HOT01 and WARM01 hold it."])
    wallets = blocks[1]
    assert wallets.block_id == "named_wallets"
    assert [r[0] for r in wallets.rows] == ["HOT01", "WARM01"]
    hot = [w for w in md["fixture"]["wallets"] if w["label"] == "HOT01"][0]
    assert wallets.rows[0][1] == hot["address"]


def test_wallet_rows_show_label_address_holder_scope_only():
    """Pre-registration section 6: each wallet label with its holder and scope."""
    assert rf._WALLET_COLUMNS == ("label", "address", "held by", "in scope")
    assert "comment" not in rf._OPEN_ITEM_FIELDS
    for md in PILOT:
        wallets = [b for b in rf.case_facts(md) if b.block_id == "wallets"][0]
        assert wallets.columns == ("label", "address", "held by", "in scope")
        assert len(wallets.rows) == len(md["fixture"]["wallets"])


def _blanked(md: dict) -> dict:
    """A copy of a pilot case with every renderer-blankable field blanked, as the
    incomplete_data variant blank_memo_fields does (truth unchanged)."""
    out = copy.deepcopy(md)
    out["fixture"]["presentation"]["blank"] = sorted(generator.BLANKABLE_FIELDS)
    return out


def _hidden_values(md: dict) -> list[str]:
    """Wallet balances and schedule comments of a blanked case that its rendered
    text does not hold."""
    rendered = generator.render_case(md["fixture"])
    values = [str(w["balance_units"]) for w in md["fixture"]["wallets"] if w.get("balance_units")]
    values += [str(w["balance_cents"]) for w in md["fixture"]["wallets"] if w.get("balance_cents")]
    values += [r["comment"] for r in md["fixture"]["open_items"] if r.get("comment")]
    return [v for v in values if v and v not in rendered]


@pytest.mark.parametrize("md", PILOT, ids=PILOT_IDS)
def test_blanked_case_fields_never_appear_in_a_fact(md):
    """Red team H2: a case that blanks wallet balances and schedule comments never
    shows them in a fact block, case facts and every family included."""
    blanked = _blanked(md)
    hidden = _hidden_values(blanked)
    t = casefile.truth_of(blanked)
    every_ref = " ".join(i.reference for i in t.items)
    every_label = " ".join(w["label"] for w in blanked["fixture"]["wallets"])
    blocks = _all_blocks(blanked)
    for family in ("amount", "asset"):
        blocks += rf.finding_facts(blanked, _parsed(blanked), family, ["amount_role_unresolved: $1.00"],
                                   [f"Items {every_ref} sit in {every_label}."])
    cells = [cell for b in blocks for row in b.rows for cell in row]
    for value in hidden:
        assert not any(value in cell for cell in cells), value
    schedule_lines = [r[2] for b in blocks if b.block_id == "fixture_rows" for r in b.rows
                      if r[0] == "open items schedule"]
    assert not any('"comment"' in line for line in schedule_lines)


def test_blanked_case_hides_something_to_test():
    """The H2 test has values to look for: some pilot case has a wallet balance or
    a schedule comment that blanking removes from the rendered text."""
    assert any(_hidden_values(_blanked(md)) for md in PILOT)
    md = next(md for md in PILOT if any(r.get("comment") for r in md["fixture"]["open_items"]))
    ref = next(r["ref"] for r in md["fixture"]["open_items"] if r.get("comment"))
    blocks = rf.finding_facts(md, None, "amount", ["amount_role_unresolved: $1.00"], [f"Item {ref} is open."])
    rows = [b for b in blocks if b.block_id == "fixture_rows"][0].rows
    assert any(r[0] == "open items schedule" for r in rows)
    assert not any('"comment"' in r[2] for r in rows)


def test_named_references_exact_and_in_text_order():
    md = PILOT[0]
    t = casefile.truth_of(md)
    a, b = t.items[0].reference, t.items[1].reference
    assert rf.named_references(f"First {b}, then {a}, then {b} again.", md) == (b, a)
    assert rf.named_references(f"{a}9 and X{a}", md) == ()
    assert rf.named_references(f"({a}).", md) == (a,)
    alias_items = [i for i in t.items if i.aliases]
    if alias_items:
        alias = alias_items[0].aliases[0]
        assert rf.named_references(f"tx {alias} landed", md) == (alias,)


def test_named_references_block_resolves_alias_to_item_row():
    md = next(m for m in PILOT if any(i["aliases"] for i in m["truth"]["items"]))
    t = casefile.truth_of(md)
    item = next(i for i in t.items if i.aliases)
    blocks = rf.finding_facts(md, None, "amount", ["item_effect_unverified: $1.00"], [f"See {item.aliases[0]}."])
    refs = blocks[0]
    assert len(refs.rows) == 1
    row = dict(zip(refs.columns, refs.rows[0]))
    assert row["named in text"] == item.aliases[0]
    assert row["reference"] == item.reference
    assert "items list" in row["found in"]


def test_amount_fixture_rows_by_exact_reference():
    md = PILOT[0]
    line = md["fixture"]["ledger_extract"]["lines"][0]
    ref = line["ref"]
    blocks = rf.finding_facts(md, None, "amount", ["amount_role_unresolved: $1.00"], [f"Line {ref} moved."])
    rows = [b for b in blocks if b.block_id == "fixture_rows"][0]
    assert rows.rows, "the ledger line holding the reference is shown"
    assert all(ref in r[2] for r in rows.rows)
    assert {r[0] for r in rows.rows} <= {"ledger extract", "on-chain extract", "open items schedule", "notes"}
    none = rf.finding_facts(md, None, "amount", ["amount_role_unresolved: $1.00"], ["No reference here."])
    assert [b for b in none if b.block_id == "fixture_rows"][0].rows == ()


def test_eligibility_labels_cover_casefile():
    assert set(rf.ELIGIBILITY_LABELS) == set(casefile.INELIGIBLE_REASONS)
    tree = ast.parse(inspect.getsource(casefile.item_eligibility))
    emitted = {
        n.args[0].value for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "append"
        and n.args and isinstance(n.args[0], ast.Constant)
    }
    assert emitted and emitted <= set(rf.ELIGIBILITY_LABELS)


def test_format_break_output():
    blocks = rf.output_facts(PILOT[0], None)
    assert len(blocks) == 1
    assert blocks[0].rows == (("output did not parse",),)


def test_output_facts_parsed():
    md = PILOT[0]
    parsed = _parsed(md)
    blocks = rf.output_facts(md, parsed)
    assert [b.block_id for b in blocks] == ["worker_fields", "worker_items", "worker_split"]
    fields = dict(blocks[0].rows)
    assert "narrative" not in fields
    assert fields["asset"] == parsed["asset"]
    assert fields["variance_amount_usd"] == json.dumps(parsed["variance_amount_usd"])
    assert len(blocks[1].rows) == len(parsed["reconciling_items"])
    t = casefile.truth_of(md)
    e, u = casefile.explained_from_citations([i["reference"] for i in parsed["reconciling_items"]], t)
    assert blocks[2].title == "Explained and unexplained, computed by code from the items this output cites"
    assert dict(blocks[2].rows)["explained amount"] == rf._signed_usd(e)
    assert dict(blocks[2].rows)["unexplained amount"] == rf._signed_usd(u)


def test_answer_key_values():
    md = PILOT[0]
    t = casefile.truth_of(md)
    ak = dict(rf.case_facts(md)[0].rows)
    assert ak["asset"] == t.asset and ak["materiality band"] == t.materiality_band
    assert ak["variance, on-chain minus ledger"].endswith(f"({t.direction})")
    assert ak["variance, on-chain minus ledger"].startswith("-" if t.variance_usd < 0 else "+")
    assert rf._usd(1234.5) == "$1,234.50"
    assert rf._signed_usd(-1234.5) == "-$1,234.50"
    assert rf._signed_usd(0) == "$0.00"


# --- Reason parsing ---------------------------------------------------------------------


def test_sweep_prefixes_equal_regression():
    assert rf.SWEEP_REASON_PREFIXES == regression.SWEEP_REASON_PREFIXES


def test_sweep_token_may_hold_colon_space():
    reason = scoring.SweepHit("Acme: Holdings", "organization_suffix").reason
    assert rf._reason_text(reason) == "Acme: Holdings"
    assert rf._reason_text(scoring.SweepHit("x", "format_character").reason) == ""


def test_reference_item_parsing():
    assert rf._item_reference_of("reconciling_items[3] 'WD-1' type 'fee' differs from 'timing_difference'") == "WD-1"
    assert rf._item_reference_of("reconciling_items[0] \"WD-'2\" type 'fee' differs from 'fee'") == "WD-'2"
    assert rf._item_reference_of(
        "narrative names ineligible item JE-25-11-7405 without stating whether it explains the variance"
    ) == "JE-25-11-7405"
    assert rf._item_reference_of("entities_cited names ineligible item DP-1 that the narrative does not discuss") == "DP-1"
    assert rf._reason_text("amount_role_unresolved: $34,068.30") == "$34,068.30"


def test_reference_item_blocks():
    md = PILOT[0]
    t = casefile.truth_of(md)
    item = t.items[0]
    parsed = _parsed(md)
    parsed["reconciling_items"] = [
        {"reference": "OTHER-1", "amount_usd": 1.0},
        {"reference": item.reference, "amount_usd": 2.0, "type": "pending xfer"},
    ]
    reason = f"reconciling_items[1] {item.reference!r} type 'pending xfer' differs from {item.type!r}"
    blocks = rf.finding_facts(md, parsed, "reference_item", [reason], [])
    ids = [b.block_id for b in blocks]
    assert ids == ["named_references", "named_wallets", "type_labels", "worker_items"]
    assert blocks[0].rows[0][0] == item.reference
    assert [r[2] for r in blocks[2].rows] == list(casefile.TYPE_LABELS[item.type])
    assert blocks[3].rows == (("reconciling_items[1]", item.reference, "2.0", "pending xfer"),)


def test_structured_field_rows():
    md = PILOT[0]
    t = casefile.truth_of(md)
    parsed = _parsed(md)
    parsed["variance_amount_usd"] = -abs(t.variance_usd)
    reasons = [f"{scoring.OFF_CONTRACT_REASON}: {f} states the true value in a form outside the output contract"
               for f in ("period", "variance_amount_usd")]
    blocks = rf.finding_facts(md, parsed, "structured_field", reasons, [])
    field = blocks[-1]
    assert field.block_id == "routed_field"
    assert field.columns == ("field", "stated in the output", "answer key")
    assert [r[0] for r in field.rows] == ["variance_amount_usd", "period"]
    assert field.rows[0][1] == json.dumps(parsed["variance_amount_usd"])
    assert field.rows[0][2] == json.dumps(t.variance_usd)
    assert field.rows[1][1:] == (parsed["period"], t.period)


def test_sweep_blocks_list_every_entity_in_case_order():
    md = PILOT[0]
    t = casefile.truth_of(md)
    blocks = rf.finding_facts(md, _parsed(md), "sweep", [scoring.SweepHit("Two Hot", "capitalized_run").reason], [])
    ents = [b for b in blocks if b.block_id == "entities"][0]
    assert [r[0] for r in ents.rows] == list(t.valid_entities)
    cited = [b for b in blocks if b.block_id == "worker_fields"][0]
    assert [r[1] for r in cited.rows] == _parsed(md)["entities_cited"]


def test_no_dashes_in_module_or_tests():
    for text in (SOURCE, Path(__file__).read_text(encoding="utf-8")):
        assert chr(0x2014) not in text and chr(0x2013) not in text


def test_imports_stay_inside_the_allowed_graph():
    tree = ast.parse(SOURCE)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("icfr_agent_evals"):
            mods.add(node.module.split(".")[-1])
    assert mods <= {"casefile", "policy", "review_schema", "scoring"}, mods
