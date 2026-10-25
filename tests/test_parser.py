"""Strict parser tests (pre-registration section 6 Parser, section 3 schema).

Exactly one top-level JSON object with exactly the schema keys and types decodes.
Everything else is a format break (FM-000) with a specific reason: never a
fabrication verdict, never an exception, never a fallback to a nested object.
"""

import json

import pytest
from casefactory import good_output, make_truth

from icfr_agent_evals.casefile import OUTPUT_FIELDS
from icfr_agent_evals.scoring import parse_worker_output


def _good() -> dict:
    return good_output(make_truth("BTC"))


def _text(obj: dict) -> str:
    return json.dumps(obj)


def test_plain_schema_valid_object_parses():
    result = parse_worker_output(_text(_good()))
    assert result.ok and result.error is None
    assert set(result.parsed) == set(OUTPUT_FIELDS)


def test_surrounding_whitespace_is_allowed():
    assert parse_worker_output("\n  " + _text(_good()) + "\n\n").ok


# D3: an output that is entirely one fenced block is unwrapped; prose outside is FM-000.

@pytest.mark.parametrize("tag", ["json", "JSON", ""])
def test_entire_output_as_one_fenced_block_parses(tag):
    text = f"```{tag}\n{json.dumps(_good(), indent=2)}\n```"
    assert parse_worker_output(text).ok


@pytest.mark.parametrize(
    "text",
    [
        "Here you go:\n```json\n{obj}\n```",
        "```json\n{obj}\n```\nLet me know if you need more.",
        "```json {obj} ```",
        "```json\n{obj}\n",
    ],
)
def test_fence_with_prose_or_malformed_fence_is_format_break(text):
    result = parse_worker_output(text.replace("{obj}", _text(_good())))
    assert not result.ok and result.error


def test_preamble_prose_is_format_break():
    result = parse_worker_output("Sure! " + _text(_good()))
    assert not result.ok and "not one JSON object" in result.error


def test_trailing_text_is_format_break():
    result = parse_worker_output(_text(_good()) + " done.")
    assert not result.ok


def test_second_object_is_format_break():
    result = parse_worker_output(_text(_good()) + "\n" + _text(_good()))
    assert not result.ok


@pytest.mark.parametrize("stray", ["}", "{", "]"])
def test_stray_brace_is_format_break(stray):
    assert not parse_worker_output(_text(_good()) + stray).ok
    assert not parse_worker_output(stray + _text(_good())).ok


def test_p1_brace_inside_string_never_yields_a_nested_object():
    """P1: the old balanced-brace scanner stopped at a '}' inside a string and
    returned the inner object while the outer object cited Ghost Custody LLC."""
    inner = _good()
    outer = {**_good(), "entities_cited": ["BitGo Trust", "Ghost Custody LLC"]}
    outer["narrative"] = "See note } and " + _text(inner)
    text = "Result: " + _text(outer)
    result = parse_worker_output(text)
    assert not result.ok
    # the whole text, decoded strictly, is the outer object or nothing
    whole = parse_worker_output(_text(outer))
    assert whole.ok and "Ghost Custody LLC" in whole.parsed["entities_cited"]


def test_p3_duplicate_entities_cited_key_is_format_break():
    """P3: a decoder that keeps the last duplicate would read a clean citation list
    while the first one named a fabricated entity."""
    good = _text(_good())
    text = good[:-1] + ', "entities_cited": ["BitGo Trust"]}'
    text = text.replace('"entities_cited": [', '"entities_cited": ["Ghost Custody LLC", ', 1)
    result = parse_worker_output(text)
    assert not result.ok and "duplicate key 'entities_cited'" in result.error


def test_duplicate_key_inside_an_item_is_format_break():
    text = _text(_good()).replace('"amount_usd": 80000.0', '"amount_usd": 80000.0, "amount_usd": 1.0')
    result = parse_worker_output(text)
    assert not result.ok and "duplicate key" in result.error


def test_p4_extra_key_naming_a_fabricated_entity_is_format_break():
    obj = {**_good(), "counterparty": "Ghost Custody LLC"}
    result = parse_worker_output(_text(obj))
    assert not result.ok and "extra keys" in result.error


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400", "-1e999"])
def test_p5_non_finite_numbers_are_format_break(literal):
    text = _text(_good()).replace('"variance_amount_usd": 120000.0', f'"variance_amount_usd": {literal}')
    assert literal in text
    result = parse_worker_output(text)
    assert not result.ok and "non-finite" in result.error


def test_non_finite_item_amount_is_format_break():
    text = _text(_good()).replace('"amount_usd": 80000.0', '"amount_usd": NaN')
    assert not parse_worker_output(text).ok


@pytest.mark.parametrize("key", list(OUTPUT_FIELDS))
def test_missing_key_is_format_break(key):
    obj = {k: v for k, v in _good().items() if k != key}
    result = parse_worker_output(_text(obj))
    assert not result.ok and "missing keys" in result.error


@pytest.mark.parametrize(
    "key,value",
    [
        ("entities_cited", "BitGo Trust"),             # not a list
        ("entities_cited", ["BitGo Trust", 7]),         # a non-string entry
        ("entities_cited", [{"name": "BitGo Trust"}]),
        ("variance_amount_usd", "120000"),
        ("variance_amount_usd", True),                  # bool is not a number
        ("variance_amount_usd", None),
        ("asset", 7),
        ("period", None),
        ("narrative", ["text"]),
        ("reconciling_items", {"reference": "WD-2512-04417"}),
        ("reconciling_items", ["WD-2512-04417"]),
        ("reconciling_items", [{"reference": "WD-2512-04417"}]),                      # amount missing
        ("reconciling_items", [{"reference": "WD-2512-04417", "amount_usd": "80000"}]),
        ("reconciling_items", [{"reference": 4417, "amount_usd": 80000.0}]),
        ("reconciling_items", [{"reference": "WD-2512-04417", "amount_usd": 80000.0, "wallet": "x"}]),
        ("reconciling_items", [{"reference": "WD-2512-04417", "amount_usd": 80000.0, "type": 3}]),
    ],
)
def test_schema_type_error_is_format_break(key, value):
    obj = {**_good(), key: value}
    result = parse_worker_output(_text(obj))
    assert not result.ok and result.error


def test_item_with_optional_type_parses():
    obj = _good()
    obj["reconciling_items"] = [{"reference": "WD-2512-04417", "amount_usd": 80000.0}]
    assert parse_worker_output(_text(obj)).ok


@pytest.mark.parametrize("bad", [None, "", "   ", "no json here", "[1, 2, 3]", '"text"', "{not valid}", "{}"])
def test_unparseable_inputs_are_format_breaks_not_exceptions(bad):
    result = parse_worker_output(bad)
    assert not result.ok and result.parsed is None and result.error


def test_deep_nesting_does_not_crash():
    result = parse_worker_output("[" * 100_000 + "]" * 100_000)
    assert not result.ok
