"""The labeling sheet (scripts/labeling_sheet.py) must never carry the answer key.

For every dev case, on its own sheet and on the 3-case trial sheet:
  - the page is a function of the case text alone: removing or swapping
    metadata.truth leaves the HTML byte-identical (codes held fixed);
  - outside the case text, the page holds no answer-key amount in any common
    form (exact, rounded, in thousands or millions), no evidence state, case kind,
    category, band label, item role, eligibility reason or truth note beyond the
    copies the fixed page already holds, and no extra comment, title or data
    attribute;
  - no metadata.truth key name appears as data; every string the page carries
    for a case is cut from that case's text; and the case id, which names the
    band, never appears. Case codes are random per build and the key goes to a
    file outside the sheet's folder.
Dev split only.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from icfr_agent_evals.casefile import CaseTruth, item_eligibility, truth_of
from icfr_agent_evals.solver import SYSTEM_PROMPT

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "labeling_sheet.py"
DEV = REPO / "data" / "dev" / "dev.jsonl"
TRIAL = ["dev-hc-010-XRP-M3", "dev-hc-008-USDC-M3", "dev-hc-003-ETH-M1"]
DATA_RE = re.compile(r'(<script type="application/json" id="sheet-data">)(.*?)(</script>)', re.S)


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


def _fixed_codes(ids: list[str]) -> dict[str, str]:
    return {c: f"c0de{i:04x}" for i, c in enumerate(ids)}


def _sheet(ids: list[str], codes: dict[str, str] | None = None, source: Path = DEV) -> str:
    return ls.build_sheet(ls.load_cases(ids, source), "Labeling sheet", codes)


def _data(html: str) -> dict:
    m = DATA_RE.search(html)
    assert m, "sheet data block missing"
    return json.loads(m.group(2))


def _truth(cid: str) -> CaseTruth:
    return truth_of(RECORDS[cid]["metadata"])


def _write_jsonl(path: Path, recs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


# ---------- leak checker ----------

def _number_forms(x: float) -> set[str]:
    """Exact, rounded and scaled forms of an amount: 1103947.93, 1,103,947.93,
    110394793 (cents), 1,103,947.9, 1,103,948, 1,104k, 1,103.9 thousand, 1.10
    million, 1.1m and so on."""
    a = abs(round(float(x), 2))
    whole = int(a + 0.5)
    forms = {f"{a:.2f}", f"{a:,.2f}", str(int(round(a * 100))), f"{a:.1f}", f"{a:,.1f}", str(whole), f"{whole:,}"}
    if a >= 1_000:
        for k in (f"{a / 1e3:,.1f}", f"{a / 1e3:,.2f}", f"{int(a / 1e3 + 0.5):,}"):
            forms |= {k + "k", k + " k", k + " thousand"}
    if a >= 100_000:
        for m in (f"{a / 1e6:,.1f}", f"{a / 1e6:,.2f}", f"{a / 1e6:,.3f}"):
            forms |= {m + " million", m + "m", m + " m", m + "mn"}
    return forms


def _count_number(form: str, hay: str) -> int:
    return len(re.findall(r"(?<![\d.,])" + re.escape(form) + r"(?!\d)", hay, re.I))


def _count_word(word: str, hay: str) -> int:
    return len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", hay, re.I))


def _leak_words(t: CaseTruth) -> set[str]:
    """Answer-key labels, as codes and as plain words: evidence state, case kind,
    category, band, item roles, eligibility reasons and truth notes."""
    raw: set[str] = {t.evidence_state, t.case_kind, t.materiality_band}
    if t.category:
        raw.add(t.category)
    for item in t.items:
        raw.add(item.role)
        raw.update(item_eligibility(item, asset=t.asset, scope_wallets=t.scope_wallets,
                                    cutoff_date=t.cutoff_date).reasons)
    for reasons in (t.notes.get("decoy_reasons") or {}).values():
        raw.update(reasons)
    for v in (t.notes.get("truth_rule"), t.notes.get("template_variant"), t.notes.get("pressed_claim")):
        if isinstance(v, str):
            raw.add(v)
    out = set()
    for s in raw:
        if s:
            out |= {s, s.replace("_", " ")}
    return out


MARKERS = {"comment": r"<!--", "title attribute": r"\btitle\s*=", "data attribute": r"\bdata-[\w-]*\s*="}


def _residue(html: str, ids: list[str]) -> tuple[str, str]:
    """(residue, reference). The residue is the page with every string cut from a
    case text blanked, and the random codes and sheet id blanked. The reference is
    the same page with no cases at all: the fixed template, title and rules. A
    leak is anything the residue holds more often than the reference."""
    m = DATA_RE.search(html)
    assert m, "sheet data block missing"
    data = json.loads(m.group(2))
    texts = {RECORDS[c]["input"] for c in ids}

    def blank(obj, text):
        if isinstance(obj, str):
            return "" if obj in text else obj
        if isinstance(obj, dict):
            return {k: blank(v, text) for k, v in obj.items()}
        if isinstance(obj, list):
            return [blank(v, text) for v in obj]
        return obj

    assert re.fullmatch(r"[0-9a-f]{10}", data["sheet"]["id"]), "sheet id is not a bare hex id"
    cases = []
    for case in data["cases"]:
        assert case.get("text") in texts, "a case text on the page is not a requested case's input"
        assert re.fullmatch(r"[0-9a-f]{8}", case["code"]), "a case code is not a bare hex code"
        assert isinstance(case["n"], int)
        cases.append({k: blank(v, case["text"]) for k, v in case.items() if k not in ("code", "n")})
    rest = {k: v for k, v in data.items() if k not in ("sheet", "cases")}
    sheet = dict(data["sheet"], id="")
    pre, post = html[: m.start(2)], html[m.end(2):]
    residue = pre + ls._json_for_html({**rest, "sheet": sheet, "cases": cases}) + post
    reference = ls.render({**rest, "sheet": sheet, "cases": []})  # the fixed template, not this page's
    return residue, reference


def _assert_no_leak(html: str, ids: list[str]) -> None:
    for cid in ids:
        assert cid not in html, f"{cid}: case id on the page"
    residue, reference = _residue(html, ids)
    for what, pat in MARKERS.items():
        assert len(re.findall(pat, residue, re.I)) <= len(re.findall(pat, reference, re.I)), f"an extra {what}"
    for cid in ids:
        t = _truth(cid)
        for name, value in (("variance", t.variance_usd), ("explained", t.explained_usd),
                            ("unexplained", t.unexplained_usd)):
            if abs(value) < 0.005:
                continue  # zero is covered by the no-prefill and text-only checks
            for form in _number_forms(value):
                assert _count_number(form, residue) <= _count_number(form, reference), f"{cid}: {name} {form!r} leaks"
        for w in _leak_words(t):
            assert _count_word(w, residue) <= _count_word(w, reference), f"{cid}: answer-key word {w!r} leaks"


@pytest.mark.parametrize("cid", ALL_IDS)
def test_single_case_sheet_leaks_nothing(cid):
    _assert_no_leak(_sheet([cid]), [cid])


def test_trial_sheet_leaks_nothing():
    _assert_no_leak(_sheet(TRIAL), TRIAL)


@pytest.mark.parametrize("change", ["metadata_removed", "truth_rotated"])
def test_page_is_a_function_of_the_case_text_alone(change, tmp_path):
    """Metamorphic: the answer key cannot reach the page if changing it changes nothing."""
    codes = _fixed_codes(ALL_IDS)
    base = _sheet(ALL_IDS, codes)
    recs = [json.loads(json.dumps(RECORDS[c])) for c in ALL_IDS]
    if change == "metadata_removed":
        for r in recs:
            r["metadata"] = {"split": "dev"}
    else:
        metas = [r["metadata"] for r in recs]
        for r, m in zip(recs, metas[1:] + metas[:1]):
            r["metadata"] = dict(m, split="dev")
    src = _write_jsonl(tmp_path / "changed.jsonl", recs)
    assert _sheet(ALL_IDS, codes, src) == base


@pytest.mark.parametrize("ids", [TRIAL, HARD_IDS, ALL_IDS], ids=["trial", "hard", "all"])
def test_page_outside_the_data_block_is_the_fixed_template(ids):
    html = _sheet(ids)
    m = DATA_RE.search(html)
    assert html[: m.start(2)] + "__DATA__" + html[m.end(2):] == ls.PAGE.replace("__TITLE__", "Labeling sheet")


# ---------- the checker itself ----------

PLANT_CASE = "dev-hc-010-XRP-M3"  # edge, ambiguous_narrative, partly unexplained, M3; all three amounts non-zero


def _plants() -> dict[str, str]:
    t = _truth(PLANT_CASE)
    v, e, u = abs(t.variance_usd), abs(t.explained_usd), abs(t.unexplained_usd)
    reason = next(r for i in t.items for r in item_eligibility(
        i, asset=t.asset, scope_wallets=t.scope_wallets, cutoff_date=t.cutoff_date).reasons)
    return {
        # written out by hand, not through _number_forms, so the test is not circular
        "variance exact": f"{v:,.2f}",
        "variance rounded": "1,103,948",
        "variance in millions": "1.10 million",
        "variance short": "1.1m",
        "explained in thousands": "700k",
        "explained plain": "699991.74",
        "unexplained cents": str(int(round(u * 100))),
        "unexplained rounded": "403,956",
        "evidence state": "Partly unexplained",
        "case kind": "edge case",
        "category plain words": "ambiguous narrative",
        "item role": t.items[0].role,
        "reason": reason,
        "title attribute": '<span title="M3"></span>',
        "data attribute": '<span data-band="M3"></span>',
        "comment": "<!-- band M3 -->",
        "comment, no band": "<!-- x -->",
    }


@pytest.mark.parametrize("plant", sorted(_plants()))
def test_leak_check_catches_a_planted_value_on_the_page(plant):
    value = _plants()[plant]
    html = _sheet([PLANT_CASE])
    _assert_no_leak(html, [PLANT_CASE])
    planted = html.replace("<main>", f"<main><p>{value}</p>", 1)
    with pytest.raises(AssertionError):
        _assert_no_leak(planted, [PLANT_CASE])


@pytest.mark.parametrize("plant", ["variance rounded", "variance short", "evidence state", "category plain words"])
def test_leak_check_catches_a_planted_value_in_the_data(plant):
    value = _plants()[plant]
    codes = _fixed_codes([PLANT_CASE])
    data = ls.sheet_data(ls.load_cases([PLANT_CASE], DEV), "Labeling sheet", codes)
    _assert_no_leak(ls.render(data), [PLANT_CASE])
    data["cases"][0]["hint"] = value
    with pytest.raises(AssertionError):
        _assert_no_leak(ls.render(data), [PLANT_CASE])


# ---------- data block, codes, key ----------

@pytest.mark.parametrize("ids", [TRIAL, HARD_IDS, ALL_IDS], ids=["trial", "hard", "all"])
def test_no_truth_key_name_as_data(ids):
    html = _sheet(ids)
    data = _data(html)
    assert not (_keys(data) & TRUTH_KEYS), _keys(data) & TRUTH_KEYS
    # The data block's keys are checked above and its strings are cut from the case
    # text (an AC-4 case file may itself hold a JSON fragment). The page code around it
    # must not use a truth key name as a key, field name or attribute.
    code = DATA_RE.sub(r"\1\3", html)
    for k in sorted(TRUTH_KEYS):
        for pat in (rf'"{re.escape(k)}"\s*:', rf"'{re.escape(k)}'\s*:", rf'\b(?:name|id)="{re.escape(k)}"',
                    rf"\bdata-{re.escape(k)}\b", rf"\.{re.escape(k)}\b"):
            assert not re.search(pat, code), f"truth key {k!r} used as data ({pat})"
        if "_" in k:
            assert k not in code, f"truth key name {k!r} appears in the page code"


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


@pytest.mark.parametrize("cid", ALL_IDS)
def test_every_case_string_is_cut_from_the_case_text(cid):
    data = _data(_sheet([cid]))
    (case,) = data["cases"]
    text = RECORDS[cid]["input"]
    assert case["text"] == text
    assert case["warnings"] == []
    for s in _strings({k: v for k, v in case.items() if k not in ("code", "warnings")}):
        assert s in text, f"{cid}: {s!r} is not from the case file"
    assert re.fullmatch(r"[0-9a-f]{8}", case["code"]) and cid not in json.dumps(data)


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


def test_markup_in_case_text_stays_inside_the_data_block(tmp_path):
    """A '<!--' or '</script>' in a case text must neither end the data block nor
    break its JSON (the old escape turned '<!--' into the invalid JSON '<\\!--')."""
    rec = json.loads(json.dumps(RECORDS[TRIAL[0]]))
    rec["input"] += "\nForwarded: <!-- x <script>alert(1)</script> --> </SCRIPT > <!-- y\n"
    src = _write_jsonl(tmp_path / "markup.jsonl", [rec])
    html = _sheet([rec["id"]], source=src)
    m = DATA_RE.search(html)
    block = m.group(2)
    assert "<" not in block
    assert json.loads(block)["cases"][0]["text"] == rec["input"]
    assert len(re.findall(r"</script", html, re.I)) == len(re.findall(r"</script", ls.PAGE, re.I))
    assert html.count("<!--") == ls.PAGE.count("<!--")
    assert html[m.end(2):].startswith("</script>\n<script>")


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
    assert len(set(codes)) == 3


def test_codes_are_random_per_build():
    """No fixed salt: two builds of the same cases share no code and no sheet id."""
    a, b = _data(_sheet(TRIAL)), _data(_sheet(TRIAL))
    assert not ({c["code"] for c in a["cases"]} & {c["code"] for c in b["cases"]})
    assert a["sheet"]["id"] != b["sheet"]["id"]
    codes = ls.new_codes(ALL_IDS)
    assert sorted(codes) == ALL_IDS and len(set(codes.values())) == len(ALL_IDS)
    with pytest.raises(ls.SheetError):
        ls.sheet_data(ls.load_cases(TRIAL, DEV), "x", dict.fromkeys(TRIAL, "00000000"))


def test_effect_sign_help_matches_the_answer_key():
    """The page defines the effect sign the way casefile.item_effect_usd does:
    from where the item is booked and which way it moves, eligible or not."""
    html = _sheet(TRIAL)
    help_p = re.search(r"<p><b>Effect sign</b>:(.*?)</p>", html, re.S).group(1)
    for must in ("eligible or not", "<b>+</b> when it is booked on chain only and comes in, or booked in the "
                 "ledger only and goes out", "<b>-</b> when it is booked on chain only and goes out, or booked "
                 "in the ledger only and comes in", "<b>0</b> only when it has no amount or is booked on both sides"):
        assert must in help_p, must
    assert "accounts for none" not in html


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
    src = _write_jsonl(tmp_path / "other.jsonl", [rec])
    with pytest.raises(ls.SheetError):
        ls.load_cases([TRIAL[0]], src)


def test_cli_writes_the_sheet_and_the_key_outside_its_folder(tmp_path, capsys):
    out = tmp_path / "rater" / "sheet.html"
    assert ls.main(["--cases", *TRIAL, "--out", str(out), "--show-key"]) == 0
    html = out.read_text(encoding="utf-8")
    data = _data(html)
    assert len(data["cases"]) == 3
    assert sorted(p.name for p in out.parent.iterdir()) == ["sheet.html"]
    (key_path,) = (out.parent.parent / "rater-key").iterdir()
    assert key_path.name == f"sheet.html.{data['sheet']['id']}.key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["sheet_id"] == data["sheet"]["id"]
    assert {(r["code"], r["n"]) for r in [dict(code=c["code"], n=c["n"]) for c in data["cases"]]} == {
        (r["code"], r["case"]) for r in key["cases"]}
    assert sorted(r["case_id"] for r in key["cases"]) == sorted(TRIAL)
    printed = capsys.readouterr().out
    assert all(c in printed for c in TRIAL) and str(key_path) in printed
    _assert_no_leak(html, TRIAL)

    other = tmp_path / "keys"
    out2 = tmp_path / "rater2" / "sheet.html"
    assert ls.main(["--cases", *TRIAL, "--out", str(out2), "--key-dir", str(other)]) == 0
    assert [p.name for p in out2.parent.iterdir()] == ["sheet.html"]
    assert len(list(other.iterdir())) == 1
    assert ls.main(["--cases", *TRIAL, "--out", str(out2), "--key-dir", str(out2.parent)]) == 2
    assert ls.main(["--cases", *TRIAL]) == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_item_check_flags_contradictory_entries():
    """Eligible Yes with a failing condition ticked, or No with none ticked, is flagged."""
    m = re.search(r"function itemCheck\(store\) \{.*?\n\}", ls.PAGE, re.S)
    assert m, "itemCheck missing"
    assert 'tr.addEventListener("change", paintCheck)' in ls.PAGE
    cases = [({"r_eligible": "yes", "r_reasons": ["r_open"]}, True),
             ({"r_eligible": "yes", "r_reasons": ["r_other"]}, True),
             ({"r_eligible": "no", "r_reasons": []}, True),
             ({"r_eligible": "no"}, True),
             ({"r_eligible": "yes", "r_reasons": []}, False),
             ({"r_eligible": "yes"}, False),
             ({"r_eligible": "no", "r_reasons": ["r_late"]}, False),
             ({}, False),
             ({"r_reasons": ["r_late"]}, False)]
    js = m.group(0) + "\nconsole.log(JSON.stringify(" + json.dumps([c for c, _ in cases]) + ".map(itemCheck)));"
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60, check=True)
    got = json.loads(res.stdout)
    assert [bool(g) for g in got] == [w for _, w in cases]


def test_cli_refuses_a_key_dir_inside_the_sheet_folder(tmp_path):
    out = tmp_path / "rater" / "sheet.html"
    for bad in (out.parent, out.parent / "_key"):
        assert ls.main(["--cases", *TRIAL, "--out", str(out), "--key-dir", str(bad)]) == 2
