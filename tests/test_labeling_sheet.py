"""The labeling sheet (scripts/labeling_sheet.py) must never carry the answer key.

For every dev case, on its own sheet and on the 3-case trial sheet: no answer-key
number (variance, explained, unexplained) appears in the HTML unless that exact
string is in the case file text; no item eligibility reason or item role string
appears unless it is in the case file text; no metadata.truth key name appears as
data; every string the page carries for a case is cut from that case's text; and
the case id, which names the band, never appears. Dev split only.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from icfr_agent_evals.casefile import CaseTruth, item_eligibility, truth_of
from icfr_agent_evals.solver import SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "labeling_sheet.py"
DEV = REPO / "data" / "dev" / "dev.jsonl"
TRIAL = ["dev-hc-010-XRP-M3", "dev-hc-008-USDC-M3", "dev-hc-003-ETH-M1"]


def _load_script():
    spec = importlib.util.spec_from_file_location("labeling_sheet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = _load_script()


def _records() -> dict[str, dict]:
    out = {}
    with open(DEV, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


RECORDS = _records()
ALL_IDS = sorted(RECORDS)
HARD_IDS = [i for i in ALL_IDS if RECORDS[i]["metadata"]["truth"]["case_kind"] != "standard"]


def _sheet(ids: list[str]) -> str:
    return ls.build_sheet(ls.load_cases(ids, DEV), "Labeling sheet")


def _data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="sheet-data">(.*?)</script>', html, re.S)
    assert m, "sheet data block missing"
    return json.loads(m.group(1))


def _truth(cid: str) -> CaseTruth:
    return truth_of(RECORDS[cid]["metadata"])


def _number_forms(x: float) -> set[str]:
    a = abs(round(float(x), 2))
    forms = {f"{a:.2f}", f"{a:,.2f}", str(int(round(a * 100)))}
    return {f for f in forms if f}


def _occurs(s: str, hay: str) -> bool:
    return re.search(r"(?<![\d.,])" + re.escape(s) + r"(?![\d])", hay) is not None


def _truth_strings(t: CaseTruth) -> set[str]:
    """Item eligibility reasons and roles, plus the code-like labels of the key."""
    out: set[str] = set()
    for item in t.items:
        out.add(item.role)
        out.update(item_eligibility(item, asset=t.asset, scope_wallets=t.scope_wallets,
                                    cutoff_date=t.cutoff_date).reasons)
    for reasons in (t.notes.get("decoy_reasons") or {}).values():
        out.update(reasons)
    for v in (t.category, t.evidence_state, t.notes.get("truth_rule"), t.notes.get("template_variant"),
              t.notes.get("pressed_claim")):
        if isinstance(v, str) and "_" in v:  # code-like labels; plain words such as "timing" are case-file words
            out.add(v)
    return {s for s in out if s}


def _truth_key_names() -> set[str]:
    names: set[str] = set()
    for rec in RECORDS.values():
        t = rec["metadata"]["truth"]
        names.update(t)
        for item in t["items"]:
            names.update(item)
        names.update(t.get("notes") or {})
        if t.get("injection"):
            names.update(t["injection"])
    return names


TRUTH_KEYS = _truth_key_names()


def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        out = set(obj)
        for v in obj.values():
            out |= _keys(v)
        return out
    if isinstance(obj, list):
        out = set()
        for v in obj:
            out |= _keys(v)
        return out
    return set()


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _assert_no_leak(html: str, ids: list[str]) -> None:
    texts = [RECORDS[c]["input"] for c in ids]
    for cid, text in zip(ids, texts):
        t = _truth(cid)
        assert cid not in html, f"{cid}: case id on the page"
        for name, value in (("variance", t.variance_usd), ("explained", t.explained_usd),
                            ("unexplained", t.unexplained_usd)):
            if abs(value) < 0.005:
                continue  # zero is covered by the no-prefill and text-only checks
            for form in _number_forms(value):
                if _occurs(form, html):
                    assert any(_occurs(form, x) for x in texts), f"{cid}: {name} {form} leaks"
        for s in _truth_strings(t):
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", html):
                assert any(s in x for x in texts), f"{cid}: answer-key string {s!r} leaks"


@pytest.mark.parametrize("cid", ALL_IDS)
def test_single_case_sheet_leaks_nothing(cid):
    _assert_no_leak(_sheet([cid]), [cid])


def test_trial_sheet_leaks_nothing():
    _assert_no_leak(_sheet(TRIAL), TRIAL)


@pytest.mark.parametrize("ids", [TRIAL, HARD_IDS, ALL_IDS], ids=["trial", "hard", "all"])
def test_no_truth_key_name_as_data(ids):
    html = _sheet(ids)
    data = _data(html)
    assert not (_keys(data) & TRUTH_KEYS), _keys(data) & TRUTH_KEYS
    # The data block's keys are checked above and its strings are cut from the case
    # text (an AC-4 case file may itself hold a JSON fragment). The page code around it
    # must not use a truth key name as a key, field name or attribute.
    code = re.sub(r'(<script type="application/json" id="sheet-data">).*?(</script>)', r"\1\2", html, flags=re.S)
    for k in sorted(TRUTH_KEYS):
        for pat in (rf'"{re.escape(k)}"\s*:', rf"'{re.escape(k)}'\s*:", rf'\b(?:name|id)="{re.escape(k)}"',
                    rf"\bdata-{re.escape(k)}\b", rf"\.{re.escape(k)}\b"):
            assert not re.search(pat, code), f"truth key {k!r} used as data ({pat})"
        if "_" in k:
            assert k not in code, f"truth key name {k!r} appears in the page code"


@pytest.mark.parametrize("cid", ALL_IDS)
def test_every_case_string_is_cut_from_the_case_text(cid):
    data = _data(_sheet([cid]))
    (case,) = data["cases"]
    text = RECORDS[cid]["input"]
    assert case["text"] == text
    assert case["warnings"] == []
    for s in _strings({k: v for k, v in case.items() if k not in ("code", "warnings")}):
        assert s in text, f"{cid}: {s!r} is not from the case file"
    assert case["code"] == ls.case_code(cid) and cid not in json.dumps(data)


@pytest.mark.parametrize("cid", ALL_IDS)
def test_aids_match_the_case_file(cid):
    """The parsed aids agree with the fixture the text was rendered from."""
    rec = RECORDS[cid]
    fx = rec["metadata"]["fixture"]
    case = _data(_sheet([cid]))["cases"][0]
    assert [r["cells"][0] for r in case["rows"]] == [r["ref"] for r in fx["open_items"]]
    assert all(len(r["cells"]) == ls.SCHEDULE_COLUMNS for r in case["rows"])
    want = [(w["label"], w["address"]) for w in fx["wallets"] if w["in_scope"]]
    assert [(w["label"], w["address"]) for w in case["inscope"]] == want
    assert case["cutoff"].endswith("23:59:59 UTC")
    assert case["assetline"].startswith(fx["header"]["asset"] + " (")
    for r, o in zip(case["rows"], fx["open_items"]):
        assert (r["cells"][8] == "") == (o["amount_cents"] is None)


def test_no_prefilled_answers_and_no_network():
    html = _sheet(TRIAL)
    body = html.split('<script type="application/json"')[0]
    assert not re.search(r"<input[^>]*\bvalue=\"[^\"]", body), "an input is prefilled"
    assert not re.search(r"\b(checked|selected)\b", body)
    for bad in ("http://", "https://", "src=", "<link", "@import", "XMLHttpRequest", "fetch(", "WebSocket",
                "sendBeacon", "url("):
        assert bad not in html, bad


def test_reason_keys_cover_the_eligibility_codes():
    html = _sheet(TRIAL)
    codes = {v for v in ls.REASON_CODES.values() if v}
    assert codes == {"not_open", "recorded_both", "wrong_asset", "out_of_scope_wallet", "date_missing",
                     "after_cutoff", "disallowed_type", "amount_missing"}
    for key in ls.REASON_CODES:
        assert f'"{key}"' in html
    for code in codes:
        assert code not in html


def test_rules_are_the_worker_prompt_verbatim():
    rules = _data(_sheet(TRIAL))["rules"]
    assert SYSTEM_PROMPT.startswith(rules)
    assert "Respond with ONLY" not in rules and "M1" in rules and "cutoff" in rules


def test_case_order_and_codes_are_neutral():
    data = _data(_sheet(TRIAL))
    codes = [c["code"] for c in data["cases"]]
    assert codes == sorted(codes) and [c["n"] for c in data["cases"]] == [1, 2, 3]
    assert {ls.case_code(c) for c in TRIAL} == set(codes)


def test_no_long_dashes():
    html = _sheet(ALL_IDS)
    assert chr(0x2013) not in html and chr(0x2014) not in html


def test_refuses_unknown_repeated_and_non_dev(tmp_path):
    with pytest.raises(ls.SheetError):
        ls.load_cases(["no-such-case"], DEV)
    with pytest.raises(ls.SheetError):
        ls.load_cases([TRIAL[0], TRIAL[0]], DEV)
    rec = dict(RECORDS[TRIAL[0]])
    rec["metadata"] = dict(rec["metadata"], split="eval")
    src = tmp_path / "other.jsonl"
    src.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ls.SheetError):
        ls.load_cases([TRIAL[0]], src)


def test_cli_writes_the_sheet(tmp_path, capsys):
    out = tmp_path / "sheet.html"
    assert ls.main(["--cases", *TRIAL, "--out", str(out), "--show-key"]) == 0
    html = out.read_text(encoding="utf-8")
    assert len(_data(html)["cases"]) == 3
    printed = capsys.readouterr().out
    assert all(c in printed for c in TRIAL)  # the key goes to the owner's terminal, never the page
    _assert_no_leak(html, TRIAL)


@pytest.mark.parametrize("plant", ["variance", "explained", "unexplained", "role", "reason"])
def test_leak_check_catches_a_planted_value(plant):
    """The checker itself works: a planted answer-key value makes it fail."""
    cid = "dev-hc-010-XRP-M3"  # partly unexplained: all three amounts are non-zero
    t = _truth(cid)
    html = _sheet([cid])
    value = {
        "variance": f"{abs(t.variance_usd):,.2f}",
        "explained": f"{abs(t.explained_usd):.2f}",
        "unexplained": str(int(round(abs(t.unexplained_usd) * 100))),
        "role": t.items[0].role,
        "reason": next(r for i in t.items for r in item_eligibility(
            i, asset=t.asset, scope_wallets=t.scope_wallets, cutoff_date=t.cutoff_date).reasons),
    }[plant]
    assert value not in RECORDS[cid]["input"]
    planted = html.replace("<main>", f"<main><p>{value}</p>", 1)
    with pytest.raises(AssertionError):
        _assert_no_leak(planted, [cid])
