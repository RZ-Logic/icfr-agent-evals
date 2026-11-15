"""The Branch B labeling kappa (scripts/labeling_kappa.py; pre-registration section 4,
decision D-R18, gate pinned by decision D-R19).

Every input is a synthetic CSV export and key file on made-up case ids
(mock-case-001 and so on). No case file, dataset or answer key is read.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import random
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "labeling_kappa.py"
SHEET_SCRIPT = REPO / "scripts" / "labeling_sheet.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LK = _load("labeling_kappa", SCRIPT)

BASE_FIELDS = {"r_variance": "-1000.00", "r_dir": "shortfall", "r_band": "M1", "r_explained": "-600.00",
               "r_unexplained": "-400.00", "r_evidence": "part", "r_flags": "", "r_passage": "",
               "r_handling": "", "r_note": "", "r_done": "yes"}


def case(case_id: str, items, **fields) -> dict:
    """items: (ref, eligible) or (ref, eligible, source); eligible is 'yes', 'no' or ''."""
    f = dict(BASE_FIELDS)
    f.update(fields)
    norm = [(t[0], t[1], t[2] if len(t) > 2 else "schedule") for t in items]
    return {"case_id": case_id, "fields": f, "items": norm}


def csv_text(sheet_id: str, rater: str, codes: dict[str, str], cases: list[dict],
             exported: str = "2026-11-20T10:00:00.000Z", first_start: str = "2026-11-23T10:00:00.000Z",
             extra_rows: list[list[str]] | None = None) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    w.writerow(LK.CSV_COLUMNS)
    for c in cases:
        code = codes[c["case_id"]]
        for fld in LK.CASE_FIELDS:
            w.writerow([sheet_id, rater, code, "", "case", fld, c["fields"].get(fld, ""), exported])
        for ref, elig, src in c["items"]:
            for fld in LK.ITEM_FIELDS:
                w.writerow([sheet_id, rater, code, ref, src, fld, elig if fld == "r_eligible" else "", exported])
        for fld in LK.TIMER_FIELDS:
            val = first_start if fld == "first_start_utc" else "0"
            w.writerow([sheet_id, rater, code, "", "timer", fld, val, exported])
    for r in extra_rows or []:
        w.writerow(r)
    return buf.getvalue()


def write_pass(tmp: Path, tag: str, sheet_id: str, cases: list[dict], rater: str = "AB",
               csv_override: str | None = None, **kw) -> tuple[Path, Path]:
    codes = {c["case_id"]: f"{tag}{i:03d}" for i, c in enumerate(cases)}
    key = {"owner_only": "Never send this file to a rater.", "sheet_id": sheet_id, "sheet_version": "1",
           "sheet_file": f"{tag}.html", "source": "made-up",
           "cases": [{"case": i + 1, "code": codes[c["case_id"]], "case_id": c["case_id"]}
                     for i, c in enumerate(cases)]}
    kp = tmp / f"{tag}.html.{sheet_id}.key.json"
    kp.write_text(json.dumps(key), encoding="utf-8")
    cp = tmp / f"labels-{sheet_id}-{tag}.csv"
    cp.write_bytes((csv_override if csv_override is not None else csv_text(sheet_id, rater, codes, cases, **kw))
                   .encode("utf-8"))
    return cp, kp


def pad54(cases: list[dict]) -> list[dict]:
    """The cases plus filler cases with no reconciling item, up to the 54 cases of
    section 4. A filler case adds no item, so the kappa is unchanged."""
    have = {c["case_id"] for c in cases}
    fill = [case(f"mock-fill-{i:03d}", []) for i in range(54)]
    return cases + [c for c in fill if c["case_id"] not in have][:54 - len(cases)]


def run_pair(tmp: Path, first_cases, second_cases, *, seed: int = 7, resamples: int = 500, **kw):
    a = LK.read_export(*write_pass(tmp, "p1", "aaaa111111", first_cases))
    b = LK.read_export(*write_pass(tmp, "p2", "bbbb222222", second_cases, **kw))
    return LK.compare(a, b, seed=seed, resamples=resamples)


def version_file(tmp: Path, seed=424242, version="0.1.0") -> Path:
    p = tmp / "dataset-version.json"
    p.write_text(json.dumps({"dataset_version": version, "labeling_kappa_seed": seed}), encoding="utf-8")
    return p


# --- The export format matches the sheet --------------------------------------------------

def test_fields_match_the_sheet_export():
    page = _load("labeling_sheet_for_kappa", SHEET_SCRIPT).PAGE
    case_fields = re.search(r"var CASE_FIELDS = \[(.*?)\];", page, re.S).group(1)
    item_fields = re.search(r"var ITEM_FIELDS = \[(.*?)\];", page, re.S).group(1)
    header = re.search(r'var rows = \[\[(.*?)\]\];', page).group(1)
    assert tuple(re.findall(r'"(\w+)"', case_fields)) == LK.CASE_FIELDS
    assert tuple(re.findall(r'"(\w+)"', item_fields)) == LK.ITEM_FIELDS
    assert tuple(re.findall(r'"(\w+)"', header)) == LK.CSV_COLUMNS
    for fld in LK.TIMER_FIELDS:
        assert f'"timer", "{fld}"' in page
    for sym in ('["surplus", ', '["shortfall", ', '["M1", ', '["M2", ', '["M3", ', '["full", ', '["part", ',
                '["none", ', '["yes", "Yes"]', '["no", "No"]'):
        assert sym in page


def test_constants_follow_the_text():
    assert LK.BOOTSTRAP_RESAMPLES == 10_000
    assert LK.NAME_THRESHOLD == Fraction(7, 10)
    assert LK.SUBSET_NAME == "Self-verified"
    assert LK.STATISTIC_LABEL == "intra-rater (test-retest) Cohen's kappa"
    assert LK.LABELED_SUBSET_CASES == 54


def test_script_reads_no_dataset():
    src = SCRIPT.read_text(encoding="utf-8")
    for banned in ("build_dataset", "build_cases", "split_jsonl", "data/eval", "metadata", "truth",
                   "icfr_agent_evals", "dev.jsonl"):
        assert banned not in src
    for banned in (chr(0x2014), chr(0x2013), "inter-rater"):
        assert banned not in src


# --- Binary kappa -------------------------------------------------------------------------

def test_binary_kappa_hand_values():
    assert LK.binary_kappa(4, 2, 2, 4) == 1
    assert LK.binary_kappa(4, 2, 2, 0) == -1
    assert LK.binary_kappa(0, 0, 0, 0) is None
    assert LK.binary_kappa(5, 5, 5, 5) is None
    assert LK.binary_kappa(5, 0, 0, 5) is None
    # first all eligible, second none: p_e = 0, kappa 0, defined
    assert LK.binary_kappa(3, 3, 0, 0) == 0


def test_perfect_agreement(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("R-2", "no")]),
             case("mock-case-002", [("R-3", "yes"), ("R-4", "no"), ("R-5", "no")])]
    r = run_pair(tmp_path, cases, cases)
    g = r["gate"]
    assert g["kappa"] == 1.0 and g["kappa_exact"] == "1"
    # Two cases, not the 54 of section 4: kappa is printed, the name is withheld.
    assert g["name"] is None and g["name_on_kappa_alone"] == "Self-verified"
    assert "the passes hold 2 cases, not the 54 cases" in g["name_reason"]
    assert g["items"] == 5 and g["items_labeled_in_both"] == 5 and g["agreed_items"] == 5
    assert g["ci"]["ci_upper"] == 1.0
    assert r["disagreed_cases"]["on_eligibility"]["count"] == 0
    assert r["disagreed_cases"]["on_any_compared_label"]["count"] == 0
    assert r["case_count"] == 2
    assert r["beside_not_gating"]["amounts_within_1_usd"]["share"] == 1.0


def test_omitted_item_counts_as_not_eligible(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes"), ("R-2", "no"), ("MEMO-9", "yes", "added")]),
             case("mock-case-002", [("R-3", "yes"), ("R-4", "no")])]
    # the second pass leaves R-1 blank and does not add MEMO-9; it adds X-7 the first never labels
    second = [case("mock-case-001", [("R-1", ""), ("R-2", "no"), ("X-7", "yes", "added")]),
              case("mock-case-002", [("R-3", "yes"), ("R-4", "no")])]
    r = run_pair(tmp_path, first, second)
    g = r["gate"]
    items = {i["ref"]: i for i in r["cases"][0]["items"]}
    assert set(items) == {"R-1", "R-2", "MEMO-9", "X-7"}
    assert (items["R-1"]["first"], items["R-1"]["second"], items["R-1"]["labeled_in"]) == (True, False, "first_only")
    assert (items["MEMO-9"]["first"], items["MEMO-9"]["second"]) == (True, False)
    assert (items["X-7"]["first"], items["X-7"]["second"], items["X-7"]["labeled_in"]) == (False, True, "second_only")
    assert g["items"] == 6
    assert (g["items_labeled_in_both"], g["items_labeled_in_first_only"], g["items_labeled_in_second_only"]) == (3, 2, 1)
    # 6 items: first yes R-1, MEMO-9, R-3 (3); second yes X-7, R-3 (2); agree R-2, R-3, R-4 (3)
    assert g["kappa_exact"] == str(LK.binary_kappa(6, 3, 2, 3))
    assert r["disagreed_cases"]["on_eligibility"]["cases"] == ["mock-case-001"]


def test_item_neither_pass_labels_is_left_out(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes"), ("R-2", ""), ("R-3", "no")])]
    second = [case("mock-case-001", [("R-1", "yes"), ("R-2", ""), ("R-3", "no")])]
    r = run_pair(tmp_path, first, second)
    assert r["gate"]["items"] == 2


def test_undefined_when_every_item_in_one_class(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("R-2", "yes")]), case("mock-case-002", [("R-3", "yes")])]
    r = run_pair(tmp_path, cases, cases)
    g = r["gate"]
    assert g["kappa"] is None and g["name"] is None
    assert "one class (eligible)" in g["undefined_reason"]
    assert "neither name" in g["name_reason"] and "undefined" in g["name_reason"]
    assert g["ci"]["ci_lower"] is None and g["ci"]["ci_upper"] is None
    assert g["ci"]["undefined_resamples_dropped"] == 500

    none = [case("mock-case-001", [("R-1", "no")]), case("mock-case-002", [("R-3", "no")])]
    g = run_pair(tmp_path, none, none)["gate"]
    assert g["kappa"] is None and "one class (not eligible)" in g["undefined_reason"]


def test_undefined_when_no_item_is_labeled(tmp_path):
    cases = [case("mock-case-001", [("R-1", "")]), case("mock-case-002", [])]
    g = run_pair(tmp_path, cases, cases)["gate"]
    assert g["kappa"] is None and g["name"] is None
    assert g["undefined_reason"] == "no item is labeled"
    assert g["items"] == 0


def test_opposite_single_classes_give_kappa_zero(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes"), ("R-2", "yes")])]
    second = [case("mock-case-001", [("R-1", "no"), ("R-2", "no")])]
    g = run_pair(tmp_path, first, second)["gate"]
    assert g["kappa"] == 0.0 and g["undefined_reason"] is None and g["name"] is None


def _items(yy: int, yn: int, ny: int, nn: int):
    first, second, k = [], [], 0
    for n, (a, b) in ((yy, ("yes", "yes")), (yn, ("yes", "no")), (ny, ("no", "yes")), (nn, ("no", "no"))):
        for _ in range(n):
            k += 1
            first.append((f"R-{k}", a))
            second.append((f"R-{k}", b))
    return first, second


def test_threshold_exactly_070_earns_the_name(tmp_path):
    f, s = _items(4, 1, 1, 9)  # kappa = 70 / 100 exactly
    first = pad54([case("mock-case-001", f[:8]), case("mock-case-002", f[8:])])
    second = pad54([case("mock-case-001", s[:8]), case("mock-case-002", s[8:])])
    g = run_pair(tmp_path, first, second)["gate"]
    assert g["kappa_exact"] == "7/10"
    assert g["name"] == "Self-verified" and g["name_withheld_because"] == []


def test_just_below_070_keeps_neither_name(tmp_path):
    f, s = _items(4, 2, 1, 8)
    first = pad54([case("mock-case-001", f)])
    second = pad54([case("mock-case-001", s)])
    g = run_pair(tmp_path, first, second)["gate"]
    assert Fraction(g["kappa_exact"]) < Fraction(7, 10)
    assert g["name"] is None and "below 0.70" in g["name_reason"]


def test_subset_name_never_verified():
    for k in (Fraction(1), Fraction(7, 10), Fraction(69, 100), Fraction(-1)):
        assert LK.subset_name(k, None)["name"] in ("Self-verified", None)
    assert LK.subset_name(Fraction(699999, 1000000), None)["name"] is None


# --- Bootstrap ----------------------------------------------------------------------------

COUNTS = [(3, 2, 1, 2), (2, 1, 1, 2), (4, 0, 1, 3), (1, 1, 1, 1), (2, 0, 0, 2)]


def _replay(counts, seed, resamples):
    rng = random.Random(seed)
    vals, undefined = [], 0
    for _ in range(resamples):
        tot = [0, 0, 0, 0]
        for i in rng.choices(range(len(counts)), k=len(counts)):
            tot = [x + y for x, y in zip(tot, counts[i])]
        k = LK.binary_kappa(*tot)
        if k is None:
            undefined += 1
        else:
            vals.append(float(k))
    vals.sort()
    return vals, undefined


def test_bootstrap_is_deterministic():
    a = LK.bootstrap_case_kappa(COUNTS, seed=99, resamples=2000)
    b = LK.bootstrap_case_kappa(COUNTS, seed=99, resamples=2000)
    c = LK.bootstrap_case_kappa(COUNTS, seed=100, resamples=2000)
    assert a == b
    assert c["seed"] == 100 and c == LK.bootstrap_case_kappa(COUNTS, seed=100, resamples=2000)
    vals, undefined = _replay(COUNTS, 99, 2000)
    assert a["defined_resamples"] == len(vals)
    assert a["ci_lower"] == vals[math.floor(0.025 * len(vals))]
    assert a["ci_upper"] == vals[math.ceil(0.975 * len(vals)) - 1]
    assert a["seed"] == 99 and a["resamples"] == 2000


def test_bootstrap_default_is_10000_resamples():
    r = LK.bootstrap_case_kappa(COUNTS, seed=5)
    assert r["resamples"] == 10_000
    assert r["defined_resamples"] + r["undefined_resamples_dropped"] == 10_000


def test_undefined_resamples_are_dropped_and_counted():
    # a draw holding only cases 4 and 5 puts every item in one class: undefined
    counts = [(2, 1, 1, 2), (1, 1, 1, 1), (2, 0, 0, 2)]
    r = LK.bootstrap_case_kappa(counts, seed=3, resamples=3000)
    vals, undefined = _replay(counts, 3, 3000)
    assert undefined > 0
    assert r["undefined_resamples_dropped"] == undefined
    assert r["defined_resamples"] == len(vals) == 3000 - undefined
    assert r["ci_lower"] == vals[math.floor(0.025 * len(vals))]


def test_cases_move_together(tmp_path):
    # one case holds every disagreement: a resample without it is perfect, with it is not
    first = [case("mock-case-001", [("R-1", "yes"), ("R-2", "no")]),
             case("mock-case-002", [("R-3", "yes"), ("R-4", "yes")])]
    second = [case("mock-case-001", [("R-1", "yes"), ("R-2", "no")]),
              case("mock-case-002", [("R-3", "no"), ("R-4", "no")])]
    r = run_pair(tmp_path, first, second, resamples=400)
    counts = [(2, 1, 1, 2), (2, 2, 0, 0)]
    assert r["gate"]["ci"] == LK.bootstrap_case_kappa(counts, seed=7, resamples=400)


def test_bootstrap_uses_the_given_seed(tmp_path):
    cases1 = [case(f"mock-case-{i:03d}", [("R-1", "yes"), ("R-2", "no")]) for i in range(6)]
    cases2 = [case(f"mock-case-{i:03d}", [("R-1", "yes" if i % 3 else "no"), ("R-2", "no")]) for i in range(6)]
    r1 = run_pair(tmp_path, cases1, cases2, seed=11)
    r2 = run_pair(tmp_path, cases1, cases2, seed=11)
    assert r1["gate"]["ci"] == r2["gate"]["ci"] and r1["gate"]["ci"]["seed"] == 11


# --- Weighted kappas ----------------------------------------------------------------------

def test_quadratic_weighted_band_hand_example():
    # o: (M1,M1) (M2,M3) (M3,M3) (M1,M2); weighted disagreement 1/8; expected 13/32
    pairs = [("M1", "M1"), ("M2", "M3"), ("M3", "M3"), ("M1", "M2")]
    assert LK.weighted_kappa(pairs, LK.BAND_SCALE, "quadratic") == Fraction(9, 13)


def test_quadratic_weighted_evidence_hand_example():
    pairs = [("full", "none"), ("none", "full"), ("part", "part")]
    assert LK.weighted_kappa(pairs, LK.EVIDENCE_SCALE, "quadratic") == Fraction(-1)


def test_unweighted_direction_hand_example():
    pairs = [("surplus", "surplus"), ("surplus", "shortfall"), ("shortfall", "shortfall"),
             ("shortfall", "shortfall")]
    assert LK.weighted_kappa(pairs, LK.DIRECTION_SCALE, "unweighted") == Fraction(1, 2)


def test_weighted_kappa_edges():
    assert LK.weighted_kappa([], LK.BAND_SCALE) is None
    assert LK.weighted_kappa([("M2", "M2"), ("M2", "M2")], LK.BAND_SCALE) is None
    with pytest.raises(ValueError):
        LK.weighted_kappa([("M4", "M1")], LK.BAND_SCALE)
    # with three categories quadratic and unweighted differ
    pairs = [("M1", "M2"), ("M2", "M2"), ("M3", "M3"), ("M1", "M1")]
    assert LK.weighted_kappa(pairs, LK.BAND_SCALE, "quadratic") != LK.weighted_kappa(pairs, LK.BAND_SCALE, "unweighted")


def test_side_kappas_in_report(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes"), ("R-0", "no")], r_band="M1", r_evidence="full", r_dir="surplus"),
             case("mock-case-002", [("R-2", "yes")], r_band="M2", r_evidence="none", r_dir="surplus"),
             case("mock-case-003", [("R-3", "no")], r_band="M3", r_evidence="part", r_dir="shortfall"),
             case("mock-case-004", [("R-4", "no")], r_band="M1", r_evidence="part", r_dir="shortfall")]
    second = [case("mock-case-001", [("R-1", "yes"), ("R-0", "no")], r_band="M1", r_evidence="none", r_dir="surplus"),
              case("mock-case-002", [("R-2", "yes")], r_band="M3", r_evidence="full", r_dir="shortfall"),
              case("mock-case-003", [("R-3", "no")], r_band="M3", r_evidence="part", r_dir="shortfall"),
              case("mock-case-004", [("R-4", "no")], r_band="M2", r_evidence="", r_dir="shortfall")]
    r = run_pair(tmp_path, first, second)
    side = r["beside_not_gating"]
    assert side["band"]["kappa_exact"] == "9/13" and side["band"]["weights"] == "quadratic"
    assert side["variance_direction"]["kappa_exact"] == "1/2" and side["variance_direction"]["weights"] == "unweighted"
    assert side["evidence_state"]["kappa_exact"] == "-1"
    assert side["evidence_state"]["cases"] == 3
    assert side["evidence_state"]["cases_left_out_for_a_missing_value"] == 1
    by = r["disagreed_cases"]["by_label"]
    assert by["band"] == 2 and by["variance_direction"] == 1 and by["evidence_state"] == 3
    assert by["eligibility"] == 0
    assert r["disagreed_cases"]["on_eligibility"]["count"] == 0
    assert r["disagreed_cases"]["on_any_compared_label"]["cases"] == ["mock-case-001", "mock-case-002",
                                                                       "mock-case-004"]


# --- Amounts ------------------------------------------------------------------------------

@pytest.mark.parametrize("text,cents", [
    ("1,234.56", 123456), ("-1,234.56", -123456), ("(1,234.56)", -123456), ("$-5", -500), ("USD -5.10", -510),
    ("USD 5.105", 511), ("0.5", 50), (".5", 50), ("12", 1200), ("", None), ("  ", None), ("abc", None),
    ("1.2.3", None), (None, None),
])
def test_parse_amount(text, cents):
    assert LK.parse_amount(text) == cents


def test_amounts_within_one_dollar(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes")], r_variance="-1000.00", r_explained="-600.00", r_unexplained="-400.00"),
             case("mock-case-002", [("R-2", "no")], r_variance="-1000.00", r_explained="-600.00", r_unexplained="-400.00"),
             case("mock-case-003", [("R-3", "no")], r_variance="", r_explained="-600.00", r_unexplained="-400.00")]
    second = [case("mock-case-001", [("R-1", "yes")], r_variance="-1001.00", r_explained="-599.00", r_unexplained="-400.00"),
              case("mock-case-002", [("R-2", "no")], r_variance="-1001.01", r_explained="-600.00", r_unexplained="-400.00"),
              case("mock-case-003", [("R-3", "no")], r_variance="-1000.00", r_explained="-600.00", r_unexplained="-400.00")]
    r = run_pair(tmp_path, first, second)
    am = r["beside_not_gating"]["amounts_within_1_usd"]
    assert am["cases_matching"] == 1 and am["cases"] == 3
    assert am["share"] == pytest.approx(1 / 3)
    assert am["cases_with_a_missing_or_unreadable_amount"] == 1
    labels = {c["case_id"]: c["labels"]["amounts"] for c in r["cases"]}
    assert labels == {"mock-case-001": "agree", "mock-case-002": "differ", "mock-case-003": "missing"}


# --- Refusals -----------------------------------------------------------------------------

def _cases():
    return [case("mock-case-001", [("R-1", "yes"), ("R-2", "no")]), case("mock-case-002", [("R-3", "no")])]


def test_refuses_the_same_sheet_twice(tmp_path):
    a = LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", _cases()))
    with pytest.raises(LK.Refused, match="both exports come from sheet"):
        LK.compare(a, a, seed=1, resamples=10)


def test_refuses_a_rater_mismatch(tmp_path):
    a = LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", _cases(), rater="AB"))
    b = LK.read_export(*write_pass(tmp_path, "p2", "bbbb222222", _cases(), rater="CD"))
    with pytest.raises(LK.Refused, match="different raters"):
        LK.compare(a, b, seed=1, resamples=10)


def test_refuses_different_case_sets(tmp_path):
    a = LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", _cases()))
    b = LK.read_export(*write_pass(tmp_path, "p2", "bbbb222222", _cases()[:1]))
    with pytest.raises(LK.Refused, match="different cases"):
        LK.compare(a, b, seed=1, resamples=10)


def test_refuses_an_unknown_case_code(tmp_path):
    cases = _cases()
    codes = {"mock-case-001": "p1000", "mock-case-002": "p1001"}
    text = csv_text("aaaa111111", "AB", codes, cases,
                    extra_rows=[["aaaa111111", "AB", "zz999", "", "case", "r_note", "", "2026-11-20T10:00:00.000Z"]])
    with pytest.raises(LK.Refused, match="not in the key file"):
        LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases, csv_override=text))


def test_refuses_a_key_of_another_sheet(tmp_path):
    cp, _ = write_pass(tmp_path, "p1", "aaaa111111", _cases())
    _, kp = write_pass(tmp_path, "p2", "bbbb222222", _cases())
    with pytest.raises(LK.Refused, match="the key file decodes sheet"):
        LK.read_export(cp, kp)


def test_refuses_a_case_missing_from_the_export(tmp_path):
    cases = _cases()
    codes = {"mock-case-001": "p1000", "mock-case-002": "p1001"}
    text = csv_text("aaaa111111", "AB", codes, cases[:1])
    with pytest.raises(LK.Refused, match="the export lacks"):
        LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases, csv_override=text))


def _good_text():
    return csv_text("aaaa111111", "AB", {"mock-case-001": "p1000", "mock-case-002": "p1001"}, _cases())


@pytest.mark.parametrize("mutate,match", [
    (lambda t: t.replace('"sheet_id","rater"', '"sheet","rater"', 1), "header"),
    (lambda t: t.replace('"r_eligible","yes"', '"r_eligible","maybe"', 1), "eligibility"),
    (lambda t: t.replace('"r_band","M1"', '"r_band","M4"', 1), "r_band"),
    (lambda t: t.replace('"case","r_note"', '"case","r_other"', 1), "not a case field"),
    (lambda t: t.replace('"schedule","r_effect"', '"schedule","r_bogus"', 1), "not an item field"),
    (lambda t: t.replace('"timer","timer_seconds"', '"timer","timer_bogus"', 1), "not a timer field"),
    (lambda t: t.replace('"case","r_note"', '"elsewhere","r_note"', 1), "item source"),
    (lambda t: t.replace('"p1000","","case","r_note"', '"p1000","R-9","case","r_note"', 1), "carries an item reference"),
    (lambda t: t.replace('"aaaa111111"', '"cccc333333"', 1), "more than one sheet"),
    (lambda t: t.replace('"AB"', '"XY"', 1), "more than one rater"),
    (lambda t: t.replace('"AB"', '""'), "no rater initials"),
    (lambda t: t.replace('2026-11-20T10:00:00.000Z', '2026-11-21T10:00:00.000Z', 1), "more than one export time"),
    (lambda t: t + '"aaaa111111","AB","p1000"\r\n', "cells"),
    (lambda t: t.replace('"schedule","r_reasons"', '"schedule","r_effect"', 1), "out of order"),
    (lambda t: t.replace('"p1000","","case","r_note","",', '"p1000","","case","r_dir","",', 1), "given twice"),
    (lambda t: t.split("\r\n")[0] + "\r\n", "no rows"),
])
def test_refuses_malformed_rows(tmp_path, mutate, match):
    text = mutate(_good_text())
    assert text != _good_text()
    with pytest.raises(LK.Refused, match=match):
        LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", _cases(), csv_override=text))


def test_refuses_a_conflicting_duplicate_reference(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("R-1", "no", "added")])]
    with pytest.raises(LK.Refused, match="both eligible and not"):
        LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases))


def test_same_answer_duplicate_counts_once(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("R-1", "yes", "added"), ("R-2", "no")])]
    p = LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases))
    assert p.cases["mock-case-001"].items == {"R-1": True, "R-2": False}
    assert any("twice with the same answer" in w for w in p.warnings)


def test_refuses_a_labeled_item_without_reference(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("", "yes", "added")])]
    with pytest.raises(LK.Refused, match="no reference"):
        LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases))


def test_unlabeled_added_row_without_reference_is_ignored(tmp_path):
    cases = [case("mock-case-001", [("R-1", "yes"), ("", "", "added")])]
    p = LK.read_export(*write_pass(tmp_path, "p1", "aaaa111111", cases))
    assert p.cases["mock-case-001"].items == {"R-1": True}


def test_references_matched_as_written_with_a_warning(tmp_path):
    first = [case("mock-case-001", [("R-1", "yes"), ("memo-9", "yes", "added"), ("R-2", "no")])]
    second = [case("mock-case-001", [("R-1", "yes"), ("MEMO-9", "yes", "added"), ("R-2", "no")])]
    r = run_pair(tmp_path, first, second)
    assert r["gate"]["items"] == 4  # memo-9 and MEMO-9 are two references


def test_refuses_bad_key_files(tmp_path):
    p = tmp_path / "k.json"
    for payload, match in (("[]", "not a JSON object"), ("{}", "no sheet id"),
                           ('{"sheet_id": "s", "cases": []}', "lists no cases"),
                           ('{"sheet_id": "s", "cases": [{"code": "a"}]}', "lacks a code or a case id"),
                           ('{"sheet_id": "s", "cases": [{"code": "a", "case_id": "x"}, {"code": "a", "case_id": "y"}]}',
                            "listed twice"),
                           ('{"sheet_id": "s", "cases": [{"code": "a", "case_id": "x"}, {"code": "b", "case_id": "x"}]}',
                            "two codes"),
                           ("not json", "not a readable key file")):
        p.write_text(payload, encoding="utf-8")
        with pytest.raises(LK.Refused, match=match):
            LK.read_key(p)


def test_dataset_version_file(tmp_path):
    v = LK.read_dataset_version(version_file(tmp_path, seed=31))
    assert v["seed"] == 31 and v["dataset_version"] == "0.1.0" and len(v["sha256"]) == 64
    p = tmp_path / "v.json"
    for payload in ('{"dataset_version": "0.1.0"}', '{"dataset_version": "0.1.0", "labeling_kappa_seed": true}',
                    '{"dataset_version": "0.1.0", "labeling_kappa_seed": -1}',
                    '{"dataset_version": "0.1.0", "labeling_kappa_seed": "5"}',
                    '{"labeling_kappa_seed": 5}', "[]"):
        p.write_text(payload, encoding="utf-8")
        with pytest.raises(LK.Refused):
            LK.read_dataset_version(p)


# --- Warnings -----------------------------------------------------------------------------

def test_case_count_warning_and_relabel_gap(tmp_path):
    r = run_pair(tmp_path, _cases(), _cases(), first_start="2026-11-21T10:00:00.000Z")
    assert any("holds 54" in w for w in r["warnings"])
    assert r["timing"]["hours_from_first_export_to_second_start"] == 24.0
    assert any("48 hours" in w for w in r["warnings"])
    r = run_pair(tmp_path, _cases(), _cases(), first_start="2026-11-22T10:00:00.000Z")
    assert r["timing"]["hours_from_first_export_to_second_start"] == 48.0
    assert not any("48 hours" in w for w in r["warnings"])


def test_name_withheld_when_the_second_pass_started_within_48_hours(tmp_path):
    cases = pad54(_cases())
    r = run_pair(tmp_path, cases, cases, first_start="2026-11-21T10:00:00.000Z")
    g = r["gate"]
    assert g["kappa"] == 1.0                                   # still computed and printed
    assert g["name"] is None and g["name_on_kappa_alone"] == "Self-verified"
    assert "started 24.0 hours after the first pass was exported" in g["name_reason"]
    assert len(g["name_withheld_because"]) == 1 and r["timing"]["gap_verifiable"] is True
    r = run_pair(tmp_path, cases, cases, first_start="2026-11-22T10:00:00.000Z")
    assert r["gate"]["name"] == "Self-verified" and r["gate"]["name_withheld_because"] == []


def test_short_case_count_and_short_gap_both_give_their_reasons(tmp_path):
    g = run_pair(tmp_path, _cases(), _cases(), first_start="2026-11-20T12:00:00.000Z")["gate"]
    assert g["name"] is None and len(g["name_withheld_because"]) == 2
    assert "2 cases" in g["name_reason"] and "2.0 hours" in g["name_reason"]


def test_unverifiable_gap_warns_and_does_not_withhold_the_name_alone(tmp_path):
    cases = pad54(_cases())
    r = run_pair(tmp_path, cases, cases, first_start="")
    assert r["timing"]["hours_from_first_export_to_second_start"] is None
    assert r["timing"]["gap_verifiable"] is False
    assert LK.GAP_NOT_VERIFIABLE in r["warnings"]
    assert "the 48-hour gap is not verifiable" in LK.GAP_NOT_VERIFIABLE
    assert r["gate"]["name"] == "Self-verified"
    assert "warning: the 48-hour gap is not verifiable" in LK.format_summary(r)


def _rewrite_source(key_path: Path, source):
    data = json.loads(key_path.read_text(encoding="utf-8"))
    if source is None:
        data.pop("source")
    else:
        data["source"] = source
    key_path.write_text(json.dumps(data), encoding="utf-8")


def test_refuses_key_files_that_name_different_source_files(tmp_path):
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", _cases())
    c2, k2 = write_pass(tmp_path, "p2", "bbbb222222", _cases())
    _rewrite_source(k2, "made-up-other")
    a, b = LK.read_export(c1, k1), LK.read_export(c2, k2)
    assert a.key_source == "made-up" and b.key_source == "made-up-other"
    with pytest.raises(LK.Refused, match="different source files"):
        LK.compare(a, b, seed=1, resamples=10)


@pytest.mark.parametrize("source", [None, "", 5])
def test_refuses_a_key_file_that_names_no_source(tmp_path, source):
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", _cases())
    _rewrite_source(k1, source)
    with pytest.raises(LK.Refused, match="names no source file"):
        LK.read_export(c1, k1)


def test_no_warning_at_54_cases(tmp_path):
    cases = [case(f"mock-case-{i:03d}", [("R-1", "yes" if i % 2 else "no")]) for i in range(54)]
    r = run_pair(tmp_path, cases, cases, resamples=50)
    assert r["case_count"] == 54
    assert not any("holds 54" in w for w in r["warnings"])


# --- Command line -------------------------------------------------------------------------

def _cli(args, cwd):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", timeout=300)


def test_cli_end_to_end(tmp_path):
    first = [case(f"mock-case-{i:03d}", [("R-1", "yes"), ("R-2", "no" if i % 4 else "yes")]) for i in range(8)]
    second = [case(f"mock-case-{i:03d}", [("R-1", "yes"), ("R-2", "no")]) for i in range(8)]
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", first)
    c2, k2 = write_pass(tmp_path, "p2", "bbbb222222", second)
    vf = version_file(tmp_path, seed=8080)
    out = tmp_path / "out" / "kappa.json"
    proc = _cli(["--first", c1, k1, "--second", c2, k2, "--dataset-version", vf, "--seed", "8080",
                 "--out", out], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "intra-rater (test-retest)" in proc.stdout
    assert "inter-rater" not in proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "inter-rater" not in json.dumps(data)
    assert data["gate"]["ci"]["resamples"] == 10_000 and data["gate"]["ci"]["seed"] == 8080
    assert data["dataset_version"]["dataset_version"] == "0.1.0"
    assert data["gate"]["name"] in ("Self-verified", None)
    assert data["statistic"] == "intra-rater (test-retest) Cohen's kappa"
    # 16 items; first marks 10 eligible, second 8; they agree on 14
    assert data["gate"]["kappa_exact"] == str(LK.binary_kappa(16, 10, 8, 14))


def test_cli_refuses_a_seed_that_differs(tmp_path):
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", _cases())
    c2, k2 = write_pass(tmp_path, "p2", "bbbb222222", _cases())
    vf = version_file(tmp_path, seed=8080)
    proc = _cli(["--first", c1, k1, "--second", c2, k2, "--dataset-version", vf, "--seed", "8081"], tmp_path)
    assert proc.returncode == LK.EXIT_REFUSED
    assert "differs from the seed 8080" in proc.stderr


def test_cli_refuses_the_same_export_twice(tmp_path):
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", _cases())
    vf = version_file(tmp_path)
    proc = _cli(["--first", c1, k1, "--second", c1, k1, "--dataset-version", vf], tmp_path)
    assert proc.returncode == LK.EXIT_REFUSED and "sheet aaaa111111" in proc.stderr


def test_main_in_process(tmp_path, capsys):
    c1, k1 = write_pass(tmp_path, "p1", "aaaa111111", pad54(_cases()))
    c2, k2 = write_pass(tmp_path, "p2", "bbbb222222", pad54(_cases()))
    vf = version_file(tmp_path, seed=1)
    assert LK.main(["--first", str(c1), str(k1), "--second", str(c2), str(k2), "--dataset-version", str(vf)]) == 0
    out = capsys.readouterr().out
    assert "kappa 1.000" in out and "name: Self-verified" in out
    assert "Disagreed cases: 0 on eligibility" in out
