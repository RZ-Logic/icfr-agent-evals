"""The labeling comparison (scripts/labeling_compare.py; pre-registration section 4,
Branch B in force, decision D-R18): pass against pass, then labels against the key.

Every input is made up: 54 answer keys built with tests/casefactory.py under
hand-written ids, CSV exports in the labeling sheet's format and key files for them.
No case file, dataset or real answer key is read.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from icfr_agent_evals.casefile import SCHEMA_VERSION, InjectionSpec, period_cutoff_date

from casefactory import default_items, item, make_truth

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "labeling_compare.py"
KAPPA_SCRIPT = REPO / "scripts" / "labeling_kappa.py"


def _load():
    name = "labeling_compare"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LC = _load()
LK = LC.LK

FIRST_EXPORTED = "2026-11-20T10:00:00.000Z"
SECOND_START = "2026-11-23T10:00:00.000Z"
SECOND_EXPORTED = "2026-11-24T10:00:00.000Z"
CUTOFF = period_cutoff_date("2025-12")
EXPLAINING = "WD-2512-04417"
DECOY = "JE-25-12-0193"


# --- Made-up answer keys ------------------------------------------------------------------

def case_ids() -> list[str]:
    return [f"eval-hw-{n:03d}" for n in range(1, 55)]


def kinds() -> list[tuple[str, str]]:
    edge = ["band_boundary", "ambiguous_narrative", "incomplete_data"]
    out = [("edge", edge[n % 3]) for n in range(30)]
    for k in range(1, 7):
        out += [("adversarial", f"AC-{k}")] * 4
    return out


def truths(overrides: dict | None = None, ids: list[str] | None = None) -> list:
    """54 answer keys in the section 4 composition. overrides maps a case id to the items
    of that case; every other case holds the default items (one eligible outflow of
    $80,000 and one cleared decoy) against a $120,000 M2 shortfall."""
    out = []
    for cid, (kind, cat) in zip(ids or case_ids(), kinds()):
        inj = None
        if cat == "AC-1":
            inj = InjectionSpec(target_field=None, demanded_value=None, truth_value=None,
                                canary="MADE-UP-CANARY", instruction_text="made-up instruction")
        out.append(make_truth(items=(overrides or {}).get(cid), case_kind=kind, category=cat, split="eval",
                              case_id=cid, injection=inj))
    return out


def write_key_file(tmp: Path, ts, name: str = "answer-key.jsonl") -> Path:
    p = tmp / name
    p.write_text("".join(json.dumps({"id": t.case_id, "input": "made-up case text",
                                     "metadata": {"schema_version": SCHEMA_VERSION, "split": "eval",
                                                  "truth": t.to_dict(), "fixture": {}}}) + "\n" for t in ts),
                 encoding="utf-8")
    return p


# --- Made-up exports ----------------------------------------------------------------------

def labels_from(t) -> dict:
    """Labels that agree with an answer key, in the sheet's vocabulary."""
    ev = {v: k for k, v in LC.EVIDENCE_FROM_SHEET.items()}[t.evidence_state]
    fields = {"r_variance": f"{t.variance_usd:.2f}", "r_dir": t.direction, "r_band": t.materiality_band,
              "r_explained": f"{t.explained_usd:.2f}", "r_unexplained": f"{t.unexplained_usd:.2f}",
              "r_evidence": ev, "r_flags": "", "r_passage": "", "r_handling": "", "r_note": "", "r_done": "yes"}
    items = [(i.reference, "yes" if t.eligibility(i).eligible else "no", "schedule") for i in t.items]
    return {"case_id": t.case_id, "fields": fields, "items": items}


def csv_text(sheet_id: str, rater: str, codes: dict[str, str], cases: list[dict], exported: str) -> str:
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
            val = {"first_start_utc": SECOND_START if exported == SECOND_EXPORTED else "2026-11-19T10:00:00.000Z",
                   "last_stop_utc": "", "timer_running_at_export": "no"}.get(fld, "0")
            w.writerow([sheet_id, rater, code, "", "timer", fld, val, exported])
    return buf.getvalue()


def write_pass(tmp: Path, tag: str, sheet_id: str, cases: list[dict], rater: str = "AB",
               exported: str = FIRST_EXPORTED) -> tuple[Path, Path]:
    codes = {c["case_id"]: f"{tag}{i:03d}" for i, c in enumerate(cases)}
    key = {"owner_only": "Never send this file to a rater.", "sheet_id": sheet_id, "sheet_version": "1",
           "sheet_file": f"{tag}.html", "source": "made-up-source",
           "cases": [{"case": i + 1, "code": codes[c["case_id"]], "case_id": c["case_id"]}
                     for i, c in enumerate(cases)]}
    kp = tmp / f"{tag}.html.{sheet_id}.key.json"
    kp.write_text(json.dumps(key), encoding="utf-8")
    cp = tmp / f"labels-{sheet_id}-{tag}.csv"
    cp.write_bytes(csv_text(sheet_id, rater, codes, cases, exported).encode("utf-8"))
    return cp, kp


def version_file(tmp: Path, seed: int | None = 424242) -> Path:
    p = tmp / "dataset-version.json"
    data = {"dataset_version": "0.1.0"}
    if seed is not None:
        data["labeling_kappa_seed"] = seed
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def edit(cases: list[dict], cid: str, *, fields: dict | None = None, items: dict | None = None,
         drop: tuple[str, ...] = (), add: tuple[tuple[str, str], ...] = ()) -> list[dict]:
    """A copy of the cases with one case changed: fields replaced, item answers set by
    reference (a blank answer leaves the item unlabeled), items dropped, items added."""
    out = []
    for c in cases:
        if c["case_id"] != cid:
            out.append(c)
            continue
        f = {**c["fields"], **(fields or {})}
        its = [(r, (items or {}).get(r, e), s) for r, e, s in c["items"] if r not in drop]
        its += [(r, e, "added") for r, e in add]
        out.append({"case_id": cid, "fields": f, "items": its})
    return out


def setup(tmp: Path, first_cases=None, second_cases=None, ts=None, *, first_sheet="aaaa111111",
          second_sheet="bbbb222222", second_rater="AB") -> list[str]:
    ts = ts if ts is not None else truths()
    base = [labels_from(t) for t in ts]
    p1 = write_pass(tmp, "p1", first_sheet, first_cases if first_cases is not None else base)
    p2 = write_pass(tmp, "p2", second_sheet, second_cases if second_cases is not None else base,
                    rater=second_rater, exported=SECOND_EXPORTED)
    return ["--first", str(p1[0]), str(p1[1]), "--second", str(p2[0]), str(p2[1]),
            "--dataset-version", str(version_file(tmp)), "--answer-key", str(write_key_file(tmp, ts)),
            "--out", str(tmp / "out")]


def run(tmp: Path, argv: list[str], capsys=None):
    code = LC.main(argv, resamples=200)
    out = tmp / "out"
    rows = []
    summary = None
    if (out / LC.ADJUDICATION_FILE).exists():
        with open(out / LC.ADJUDICATION_FILE, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        summary = json.loads((out / LC.SUMMARY_JSON).read_text(encoding="utf-8"))
    return code, rows, summary


def base_cases(ts=None) -> list[dict]:
    return [labels_from(t) for t in (ts if ts is not None else truths())]


# --- Agreement everywhere -----------------------------------------------------------------

def test_agreement_everywhere(tmp_path, capsys):
    code, rows, summary = run(tmp_path, setup(tmp_path))
    assert code == LC.EXIT_OK
    assert rows == []
    with open(tmp_path / "out" / LC.ADJUDICATION_FILE, encoding="utf-8-sig", newline="") as fh:
        assert next(csv.reader(fh)) == list(LC.ADJUDICATION_COLUMNS)
    gate = summary["pass_vs_pass"]["kappa"]["gate"]
    assert gate["kappa"] == 1.0 and gate["items"] == 108 and gate["name"] == LK.SUBSET_NAME
    assert summary["pass_vs_pass"]["disagreements"]["rows"] == 0
    assert summary["pass_vs_pass"]["kappa"]["disagreed_cases"]["adjudicated_case_count"]["count"] == 0
    assert summary["label_vs_key"]["rows"] == 0
    assert summary["label_vs_key"]["key_items"] == 108 and summary["label_vs_key"]["key_items_eligible"] == 54
    assert summary["dataset_version"]["seed"] == 424242
    assert summary["pass_vs_pass"]["kappa"]["gate"]["ci"]["seed"] == 424242
    text = (tmp_path / "out" / LC.SUMMARY_TEXT).read_text(encoding="utf-8")
    assert "difference rows: 0" in text and "disagreement rows: 0" in text


def test_text_summary_holds_counts_only(tmp_path, capsys):
    first = edit(base_cases(), "eval-hw-007", fields={"r_band": "M3"}, items={EXPLAINING: "no"})
    code, rows, _ = run(tmp_path, setup(tmp_path, first_cases=first))
    assert code == LC.EXIT_OK and rows
    shown = capsys.readouterr().out
    text = (tmp_path / "out" / LC.SUMMARY_TEXT).read_text(encoding="utf-8")
    for s in (text, shown.split("wrote ")[0]):
        for leak in ("eval-hw", EXPLAINING, DECOY, "M3", "M2", "120000", "80000", "shortfall"):
            assert leak not in s


# --- (1) Pass against pass ----------------------------------------------------------------

def test_pass_vs_pass_eligibility_flip(tmp_path):
    second = edit(base_cases(), "eval-hw-005", items={EXPLAINING: "no"})
    code, rows, summary = run(tmp_path, setup(tmp_path, second_cases=second))
    assert code == LC.EXIT_OK
    pvp = [r for r in rows if r["kind"] == LC.PASS_VS_PASS]
    assert [(r["case_id"], r["field"], r["item_ref"], r["pass1_value"], r["pass2_value"], r["key_value"])
            for r in pvp] == [("eval-hw-005", "eligibility", EXPLAINING, "yes", "no", "")]
    lvk = [r for r in rows if r["kind"] == LC.LABEL_VS_KEY]
    assert [(r["field"], r["item_ref"], r["pass1_value"], r["pass2_value"], r["key_value"], r["differs_from_key"])
            for r in lvk] == [("eligibility", EXPLAINING, "yes", "no", "yes", "second")]
    # pass-against-pass rows come first
    assert [r["kind"] for r in rows] == [LC.PASS_VS_PASS, LC.LABEL_VS_KEY]
    assert all(r["author_decision"] == "" and r["author_reasoning"] == "" for r in rows)
    d = summary["pass_vs_pass"]["disagreements"]
    assert d["cases"] == ["eval-hw-005"] and d["same_cases_as_the_adjudicated_case_count"]
    assert summary["pass_vs_pass"]["kappa"]["disagreed_cases"]["adjudicated_case_count"]["cases"] == ["eval-hw-005"]
    assert summary["pass_vs_pass"]["kappa"]["gate"]["agreed_items"] == 107
    assert summary["label_vs_key"]["by_pass"]["second"]["rows"] == 1
    assert summary["label_vs_key"]["by_pass"]["first"]["rows"] == 0


def test_pass_vs_pass_choices_and_amounts(tmp_path):
    second = edit(base_cases(), "eval-hw-010",
                  fields={"r_dir": "", "r_evidence": "full", "r_explained": "-80000.90", "r_unexplained": "-39998.50"})
    code, rows, summary = run(tmp_path, setup(tmp_path, second_cases=second))
    pvp = {(r["field"]): (r["pass1_value"], r["pass2_value"]) for r in rows if r["kind"] == LC.PASS_VS_PASS}
    # the explained amounts are within $1 of each other; the unexplained ones are not
    assert pvp == {"variance_direction": ("shortfall", ""), "evidence_state": ("part", "full"),
                   "unexplained_usd": ("-40000.00", "-39998.50")}
    assert summary["pass_vs_pass"]["disagreements"]["same_cases_as_the_adjudicated_case_count"]


# --- (2) Labels against the key -----------------------------------------------------------

def test_label_vs_key_band_difference(tmp_path):
    both = edit(base_cases(), "eval-hw-031", fields={"r_band": "M3"})
    code, rows, summary = run(tmp_path, setup(tmp_path, first_cases=both, second_cases=both))
    assert code == LC.EXIT_OK
    assert [(r["kind"], r["case_id"], r["field"], r["pass1_value"], r["pass2_value"], r["key_value"],
             r["differs_from_key"]) for r in rows] == [
        (LC.LABEL_VS_KEY, "eval-hw-031", "band", "M3", "M3", "M2", "both")]
    assert summary["pass_vs_pass"]["disagreements"]["rows"] == 0


def test_omitted_item_counts_as_not_eligible(tmp_path):
    # pass 1 leaves the eligible item blank on one case and drops the decoy on another
    first = edit(base_cases(), "eval-hw-012", items={EXPLAINING: ""})
    first = edit(first, "eval-hw-013", drop=(DECOY,))
    code, rows, summary = run(tmp_path, setup(tmp_path, first_cases=first))
    got = [(r["kind"], r["case_id"], r["item_ref"], r["pass1_value"], r["pass2_value"], r["key_value"],
            r["differs_from_key"]) for r in rows]
    assert got == [
        (LC.PASS_VS_PASS, "eval-hw-012", EXPLAINING, "not labeled", "yes", "", ""),
        (LC.PASS_VS_PASS, "eval-hw-013", DECOY, "not labeled", "no", "", ""),
        # omitted counts as not eligible: a difference from the key on the eligible item only
        (LC.LABEL_VS_KEY, "eval-hw-012", EXPLAINING, "not labeled", "yes", "yes", "first"),
    ]
    gate = summary["pass_vs_pass"]["kappa"]["gate"]
    assert gate["items_labeled_in_second_only"] == 2 and gate["agreed_items"] == 107


def test_key_item_neither_pass_labeled(tmp_path):
    extra_yes = item("TD-2512-00099", 5_000.00, type="timing_difference", recorded_in="on_chain", flow="inflow")
    extra_no = item("TD-2601-00007", 9_000.00, type="timing_difference", date="2026-01-05", role="decoy")
    ts = truths({"eval-hw-040": default_items("BTC", "2025-12") + (extra_yes, extra_no)})
    t40 = next(t for t in ts if t.case_id == "eval-hw-040")
    assert t40.explained_usd == -75_000.00 and t40.unexplained_usd == -45_000.00
    cases = [labels_from(t) for t in ts]
    cases = edit(cases, "eval-hw-040", drop=("TD-2512-00099", "TD-2601-00007"))  # neither pass labels them
    code, rows, summary = run(tmp_path, setup(tmp_path, cases, cases, ts))
    assert code == LC.EXIT_OK
    assert [(r["kind"], r["field"], r["item_ref"], r["pass1_value"], r["pass2_value"], r["key_value"],
             r["differs_from_key"]) for r in rows] == [
        (LC.LABEL_VS_KEY, "eligibility", "TD-2512-00099", "not labeled", "not labeled", "yes", "both")]
    k = summary["label_vs_key"]
    assert k["key_items"] == 110 and k["key_items_labeled_in_neither_pass"] == 2
    assert summary["pass_vs_pass"]["kappa"]["gate"]["items"] == 108  # the kappa covers labeled items only


def test_reference_the_key_does_not_hold(tmp_path):
    first = edit(base_cases(), "eval-hw-020", add=(("WD-9999-00001", "yes"), ("WD-9999-00002", "no")))
    code, rows, summary = run(tmp_path, setup(tmp_path, first_cases=first))
    lvk = [(r["item_ref"], r["pass1_value"], r["pass2_value"], r["key_value"], r["differs_from_key"])
           for r in rows if r["kind"] == LC.LABEL_VS_KEY]
    assert lvk == [("WD-9999-00001", "yes", "not labeled", LC.NOT_IN_KEY, "first")]
    assert summary["label_vs_key"]["labeled_references_not_in_the_key"] == {"first": 2, "second": 0}


def test_references_match_by_canonical_form(tmp_path):
    # a full-width letter folds to its plain form under NFKC; the item is the key's item
    first = edit(base_cases(), "eval-hw-021", drop=(EXPLAINING,), add=(("ＷD-2512-04417", "yes"),))
    code, rows, _ = run(tmp_path, setup(tmp_path, first_cases=first))
    assert not [r for r in rows if r["kind"] == LC.LABEL_VS_KEY]


def test_variance_sign_and_evidence_vocabulary(tmp_path):
    both = edit(base_cases(), "eval-hw-050", fields={"r_variance": "120000.00", "r_evidence": "full"})
    code, rows, _ = run(tmp_path, setup(tmp_path, first_cases=both, second_cases=both))
    got = {r["field"]: (r["pass1_value"], r["key_value"], r["differs_from_key"]) for r in rows}
    assert got == {"variance_usd": ("120000.00", "-120000.00", "both"),
                   "evidence_state": ("fully_explained", "partly_unexplained", "both")}


def test_amounts_within_one_dollar_match_the_key(tmp_path):
    first = edit(base_cases(), "eval-hw-051", fields={"r_variance": "(119,999.10)", "r_explained": "-80001.00"})
    code, rows, _ = run(tmp_path, setup(tmp_path, first_cases=first))
    assert not [r for r in rows if r["kind"] == LC.LABEL_VS_KEY]


# --- Refusals -----------------------------------------------------------------------------

def _refused(tmp_path, argv, capsys, words: str):
    code = LC.main(argv, resamples=50)
    err = capsys.readouterr().err
    assert code == LC.EXIT_REFUSED, err
    assert words in err, err
    assert not (tmp_path / "out" / LC.ADJUDICATION_FILE).exists()


def test_refuses_a_missing_export(tmp_path, capsys):
    argv = setup(tmp_path)
    i = argv.index("--second")
    _refused(tmp_path, argv[:i] + argv[i + 3:], capsys, "give both passes")


def test_refuses_an_unreadable_export(tmp_path, capsys):
    argv = setup(tmp_path)
    argv[argv.index("--second") + 1] = str(tmp_path / "missing.csv")
    _refused(tmp_path, argv, capsys, "not a readable CSV file")


def test_refuses_one_sheet_twice(tmp_path, capsys):
    _refused(tmp_path, setup(tmp_path, second_sheet="aaaa111111"), capsys, "must be two sheets")


def test_refuses_another_rater(tmp_path, capsys):
    _refused(tmp_path, setup(tmp_path, second_rater="CD"), capsys, "different raters")


def test_refuses_a_key_with_other_case_ids(tmp_path, capsys):
    argv = setup(tmp_path)
    ids = case_ids()[:-1] + ["eval-hw-099"]
    write_key_file(tmp_path, truths(ids=ids), name="other-key.jsonl")
    argv[argv.index("--answer-key") + 1] = str(tmp_path / "other-key.jsonl")
    _refused(tmp_path, argv, capsys, "do not hold the same case ids")


def test_refuses_passes_short_of_54_cases(tmp_path, capsys):
    short = base_cases()[:-1]
    _refused(tmp_path, setup(tmp_path, short, short), capsys, "holds 53 cases, not the 54")


def test_refuses_a_missing_or_broken_answer_key(tmp_path, capsys):
    argv = setup(tmp_path)
    argv[argv.index("--answer-key") + 1] = str(tmp_path / "no-such-key.jsonl")
    _refused(tmp_path, argv, capsys, "the answer-key file is not there")
    (tmp_path / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
    argv[argv.index("--answer-key") + 1] = str(tmp_path / "broken.jsonl")
    _refused(tmp_path, argv, capsys, "not a readable answer-key file")


def test_refuses_a_dataset_version_without_the_seed(tmp_path, capsys):
    argv = setup(tmp_path)
    version_file(tmp_path, seed=None)
    _refused(tmp_path, argv, capsys, "labeling_kappa_seed")


def test_refuses_an_out_folder_inside_the_repository(tmp_path, capsys):
    argv = setup(tmp_path)
    inside = REPO / "labeling-compare-out-must-not-exist"
    argv[argv.index("--out") + 1] = str(inside)
    _refused(tmp_path, argv, capsys, "inside the repository")
    assert not inside.exists()


def test_refuses_to_write_over_an_adjudication_sheet(tmp_path, capsys):
    argv = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / LC.ADJUDICATION_FILE).write_text("the author's decisions\n", encoding="utf-8")
    assert LC.main(argv, resamples=50) == LC.EXIT_REFUSED
    assert "already holds adjudication.csv" in capsys.readouterr().err
    assert (out / LC.ADJUDICATION_FILE).read_text(encoding="utf-8") == "the author's decisions\n"
    assert not (out / LC.SUMMARY_JSON).exists()


def test_never_edits_an_input(tmp_path):
    argv = setup(tmp_path)
    inputs = [Path(a) for a in argv if a.endswith((".csv", ".json", ".jsonl"))]
    before = {p: p.read_bytes() for p in inputs}
    assert LC.main(argv, resamples=50) == LC.EXIT_OK
    assert {p: p.read_bytes() for p in inputs} == before


# --- Reuse and public text ----------------------------------------------------------------

def test_reuses_the_kappa_script():
    assert Path(LK.__file__).resolve() == KAPPA_SCRIPT.resolve()
    src = SCRIPT.read_text(encoding="utf-8")
    for copied in ("def compare", "def binary_kappa", "def read_export", "def bootstrap_case_kappa",
                   "def parse_amount", "def weighted_kappa"):
        assert copied not in src


def test_public_text():
    # the words are built from parts, so this file does not hold them either
    banned = [chr(0x2014), chr(0x2013), "Cl" + "aude", "Fa" + "ble", "the " + "owner", "work" + "tree",
              "scratch" + "pad", "fix " + "round", "inter" + "-rater"]
    for path in (SCRIPT, Path(__file__)):
        src = path.read_text(encoding="utf-8")
        for word in banned:
            assert word not in src, (path.name, word)


def test_command_line_run(tmp_path):
    argv = setup(tmp_path)
    proc = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True,
                          cwd=str(REPO), timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "Labeling comparison" in proc.stdout
    summary = json.loads((tmp_path / "out" / LC.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["pass_vs_pass"]["kappa"]["gate"]["ci"]["resamples"] == LK.BOOTSTRAP_RESAMPLES
