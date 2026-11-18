"""Compare the two Branch B labeling passes with each other, then with the answer key
(pre-registration section 4, Labeling protocol; Branch B in force, decision D-R18).

Usage:
  python scripts/labeling_compare.py --first CSV KEY --second CSV KEY --dataset-version FILE
                                     --answer-key FILE --out DIR

When it runs. After both labeling passes on the 54 hand-written evaluation cases are
exported. It reads the two CSV exports and the key file of each sheet (the formats of
scripts/labeling_sheet.py, read by scripts/labeling_kappa.py), the dataset-version file
that records the bootstrap seed, and the answer-key file (one JSON record per case with
id, input, metadata.truth and metadata.fixture, the format of
scripts/build_handwritten.py, loaded by icfr_agent_evals.handwritten.load_handwritten).

It does two separate things, in this order, and keeps them apart in its output.

(1) Pass against pass. It calls compare in scripts/labeling_kappa.py (loaded by path,
not copied) for the gated intra-rater kappa on per-item eligibility, its case
bootstrap, the name (Self-verified or neither), the side statistics and the
adjudicated-case list. It then lists every disagreement between the two passes, per
case and label: eligibility per item (matched by reference as written, as the kappa
matches them), variance direction, band, evidence state, the variance, explained and
unexplained amounts when they are more than $1 apart, and every label given in one
pass only (an item labeled in one pass and absent or blank in the other; a choice or
an amount given in one pass and blank or unreadable in the other). No answer-key value
enters this part.

(2) Labels against the key. For each case and each pass it compares the labels with
the answer key: the variance amount within $1, signed as the sheet records it (on-chain
less ledger, with its sign), direction, band, evidence state (the sheet's full, part
and none read as fully_explained, partly_unexplained and unexplained, the order of
casefile.EVIDENCE_STATES), eligibility per item, and the explained and unexplained
amounts within $1. Key eligibility is casefile.item_eligibility through
CaseTruth.eligibility. Items are matched by casefile.canonical_reference of the
reference. An item the key holds that a pass did not label counts as not eligible in
that pass, as section 4 does for the kappa. A reference a pass labels that the key does
not hold counts as not eligible in the key, so it is listed only when a pass marks it
eligible. A blank or unreadable label differs from the key.

Output, written to the --out folder:
  adjudication.csv  one row per disagreement: case_id, kind (pass_vs_pass or
                    label_vs_key), field, item_ref, pass1_value, pass2_value,
                    key_value, differs_from_key (first, second or both; blank on a
                    pass_vs_pass row), and two empty columns, author_decision and
                    author_reasoning. Pass-against-pass rows come first.
  summary.json      the full labeling_kappa result, the disagreement counts of both
                    parts, the input hashes and the dataset version.
  summary.txt       a short summary of counts only, also printed.
The folder must be outside the repository, because these files hold labels and
answer-key values. The tool refuses to write over any of the three files, so an
adjudication sheet the author has started is never replaced. It never edits a label:
the decision and reasoning columns are for the author.

Refused: an export or key file not given or not readable, two exports of one sheet,
raters that differ, passes and an answer key that do not hold the same 54 case ids, an
answer-key file that is missing or fails its loader's checks, a dataset-version file
with no recorded seed, an --out folder inside the repository, and an --out folder that
already holds an output file.

Exit codes: 0 written; 3 refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals.casefile import (  # noqa: E402
    EVIDENCE_STATES, CaseTruth, TruthError, canonical_reference,
)
from icfr_agent_evals.handwritten import HANDWRITTEN_CASES, HandwrittenError, load_handwritten  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
KAPPA_SCRIPT = REPO / "scripts" / "labeling_kappa.py"


def _load_kappa():
    """scripts/labeling_kappa.py, loaded by path and registered under its own name, so
    this tool and the kappa report share one copy of its code."""
    name = "labeling_kappa"
    if name in sys.modules and Path(getattr(sys.modules[name], "__file__", "")).resolve() == KAPPA_SCRIPT.resolve():
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, KAPPA_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module by name
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


LK = _load_kappa()
Refused = LK.Refused

assert LK.LABELED_SUBSET_CASES == HANDWRITTEN_CASES == 54

ADJUDICATION_FILE = "adjudication.csv"
SUMMARY_JSON = "summary.json"
SUMMARY_TEXT = "summary.txt"
OUTPUT_FILES = (ADJUDICATION_FILE, SUMMARY_JSON, SUMMARY_TEXT)
ADJUDICATION_COLUMNS: tuple[str, ...] = (
    "case_id", "kind", "field", "item_ref", "pass1_value", "pass2_value", "key_value",
    "differs_from_key", "author_decision", "author_reasoning",
)
PASS_VS_PASS, LABEL_VS_KEY = "pass_vs_pass", "label_vs_key"

# The sheet's evidence options, in the order of casefile.EVIDENCE_STATES.
EVIDENCE_FROM_SHEET: dict[str, str] = {"full": "fully_explained", "part": "partly_unexplained", "none": "unexplained"}
assert tuple(EVIDENCE_FROM_SHEET) == LK.EVIDENCE_SCALE
assert tuple(EVIDENCE_FROM_SHEET.values()) == EVIDENCE_STATES

CHOICE_LABELS: tuple[tuple[str, str], ...] = (
    ("variance_direction", "r_dir"), ("band", "r_band"), ("evidence_state", "r_evidence"),
)
AMOUNT_LABELS: tuple[tuple[str, str], ...] = (
    ("variance_usd", "r_variance"), ("explained_usd", "r_explained"), ("unexplained_usd", "r_unexplained"),
)
assert tuple(f for _, f in AMOUNT_LABELS) == LK.AMOUNT_FIELDS
FIELD_ORDER: tuple[str, ...] = ("eligibility",) + tuple(lab for lab, _ in CHOICE_LABELS) + tuple(
    lab for lab, _ in AMOUNT_LABELS)

NOT_LABELED = "not labeled"
BOTH_ANSWERS = "yes and no"
NOT_IN_KEY = "not in the key"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cents(value: float) -> int:
    return int((Decimal(repr(float(value))) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _money(value: float) -> str:
    return f"{Decimal(_cents(value)) / 100:.2f}"


def _yes_no(value: bool | None) -> str:
    return NOT_LABELED if value is None else ("yes" if value else "no")


def _row(case_id: str, kind: str, field: str, ref: str, first: str, second: str, key: str = "",
         differs: str = "") -> dict[str, str]:
    return {"case_id": case_id, "kind": kind, "field": field, "item_ref": ref, "pass1_value": first,
            "pass2_value": second, "key_value": key, "differs_from_key": differs,
            "author_decision": "", "author_reasoning": ""}


# --- Inputs -------------------------------------------------------------------------------

def read_answer_key(path: str | Path) -> dict[str, CaseTruth]:
    """case id -> answer key, from a hand-written answer-key file. The loader checks every
    key with validate_truth and the section 4 composition."""
    path = Path(path)
    try:
        cases = load_handwritten(path)
    except (HandwrittenError, TruthError) as e:
        raise Refused(f"{path.name}: the answer-key file fails its checks ({e})") from e
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise Refused(f"{path.name}: not a readable answer-key file ({type(e).__name__})") from e
    if cases is None:
        raise Refused(f"{path.name}: the answer-key file is not there")
    return {c.truth.case_id: c.truth for c in cases}


def check_inputs(first, second, truths: Mapping[str, CaseTruth]) -> None:
    """The refusals that come before any comparison."""
    if first.sheet_id == second.sheet_id:
        raise Refused(f"both exports come from sheet {first.sheet_id}: the two passes must be two sheets")
    if first.rater != second.rater:
        raise Refused(f"the passes have different raters ({first.rater!r}, {second.rater!r}); "
                      "Branch B is one rater labeling twice")
    sets = {"the first pass": set(first.cases), "the second pass": set(second.cases),
            "the answer key": set(truths)}
    for name, ids in sets.items():
        if len(ids) != LK.LABELED_SUBSET_CASES:
            raise Refused(f"{name} holds {len(ids)} cases, not the {LK.LABELED_SUBSET_CASES} hand-written "
                          "cases of pre-registration section 4")
    if not (sets["the first pass"] == sets["the second pass"] == sets["the answer key"]):
        parts = []
        for a, b in (("the first pass", "the second pass"), ("the first pass", "the answer key"),
                     ("the second pass", "the answer key")):
            only_a = sorted(sets[a] - sets[b])
            only_b = sorted(sets[b] - sets[a])
            if only_a or only_b:
                parts.append(f"{a} only against {b}: {len(only_a)}; {b} only: {len(only_b)}")
        raise Refused("the passes and the answer key do not hold the same case ids (" + "; ".join(parts) + ")")


# --- (1) Pass against pass ----------------------------------------------------------------

def pass_disagreements(first, second, kappa: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every disagreement between the two passes, per case and label. Choices and items
    follow the statuses labeling_kappa.compare gives; amounts are listed per field."""
    counted = set(LK.COUNTED_STATUSES)
    rows: list[dict[str, str]] = []
    for case in kappa["cases"]:
        cid = case["case_id"]
        a, b = first.cases[cid], second.cases[cid]
        for it in case["items"]:
            if it["labeled_in"] == "both" and it["first"] == it["second"]:
                continue
            v1 = it["first"] if it["labeled_in"] in ("both", "first_only") else None
            v2 = it["second"] if it["labeled_in"] in ("both", "second_only") else None
            rows.append(_row(cid, PASS_VS_PASS, "eligibility", it["ref"], _yes_no(v1), _yes_no(v2)))
        for label, fld in CHOICE_LABELS:
            if case["labels"][label] in counted:
                rows.append(_row(cid, PASS_VS_PASS, label, "", a.fields[fld], b.fields[fld]))
        for label, fld in AMOUNT_LABELS:
            x, y = LK.parse_amount(a.fields[fld]), LK.parse_amount(b.fields[fld])
            apart = x is not None and y is not None and abs(x - y) > LK.AMOUNT_TOLERANCE_CENTS
            if apart or (x is None) != (y is None):
                rows.append(_row(cid, PASS_VS_PASS, label, "", a.fields[fld], b.fields[fld]))
    return rows


# --- (2) Labels against the key -----------------------------------------------------------

def _canonical_items(labels) -> tuple[dict[str, bool | str], dict[str, str]]:
    """canonical reference -> eligible (or BOTH_ANSWERS when two spellings that fold to one
    canonical reference carry different answers), and canonical reference -> the
    reference as written."""
    out: dict[str, bool | str] = {}
    written: dict[str, str] = {}
    for ref, value in labels.items.items():
        key = canonical_reference(ref)
        written.setdefault(key, ref)
        if key in out and out[key] != value:
            out[key] = BOTH_ANSWERS
        elif key not in out:
            out[key] = value
    return out, written


def _show(value: bool | str | None) -> str:
    return value if isinstance(value, str) else _yes_no(value)


def key_differences(first, second, truths: Mapping[str, CaseTruth]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Every difference between a pass's labels and the answer key, one row per case,
    field and item where at least one pass differs, with counts."""
    rows: list[dict[str, str]] = []
    unlabeled_key_items = 0
    key_items = 0
    eligible_key_items = 0
    extra_refs = {"first": 0, "second": 0}
    passes = (("first", first), ("second", second))

    def add(cid: str, field: str, ref: str, values: Sequence[str], oks: Sequence[bool], key: str) -> None:
        if all(oks):
            return
        differs = "both" if not any(oks) else ("first" if not oks[0] else "second")
        rows.append(_row(cid, LABEL_VS_KEY, field, ref, values[0], values[1], key, differs))

    for cid in sorted(truths):
        t = truths[cid]
        labs = [p.cases[cid] for _, p in passes]

        # eligibility per item
        canon = [_canonical_items(lab) for lab in labs]
        key_refs: set[str] = set()
        for item in t.items:
            ck = canonical_reference(item.reference)
            key_refs.add(ck)
            elig = t.eligibility(item)
            key_items += 1
            eligible_key_items += elig.eligible
            got = [c[0].get(ck) for c in canon]
            if all(g is None for g in got):
                unlabeled_key_items += 1
            # an item the key holds that a pass did not label counts as not eligible in that pass
            oks = [g is not BOTH_ANSWERS and bool(g) == elig.eligible for g in got]
            key_show = "yes" if elig.eligible else "no (" + ", ".join(elig.reasons) + ")"
            add(cid, "eligibility", item.reference, [_show(g) for g in got], oks, key_show)
        extra: dict[str, str] = {}
        for c in canon:
            for ck, ref in c[1].items():
                if ck not in key_refs:
                    extra.setdefault(ck, ref)
        for n, (name, _) in enumerate(passes):
            extra_refs[name] += sum(1 for ck in canon[n][0] if ck not in key_refs)
        for ck in sorted(extra):
            got = [c[0].get(ck) for c in canon]
            # a reference the key does not hold is not eligible in the key
            oks = [g is not BOTH_ANSWERS and not g for g in got]
            alias = t.item_by_reference(extra[ck])
            key_show = NOT_IN_KEY + (f" as a reference; an alias of {alias.reference}" if alias is not None else "")
            add(cid, "eligibility", extra[ck], [_show(g) for g in got], oks, key_show)

        # choices
        for label, fld in CHOICE_LABELS:
            raw = [lab.fields[fld] for lab in labs]
            if label == "evidence_state":
                values = [EVIDENCE_FROM_SHEET.get(v, v) for v in raw]
                key = t.evidence_state
            elif label == "variance_direction":
                values, key = raw, t.direction
            else:
                values, key = raw, t.materiality_band
            add(cid, label, "", values, [v == key for v in values], key)

        # amounts, signed as the sheet records them
        for label, fld in AMOUNT_LABELS:
            key_value = {"variance_usd": t.variance_usd, "explained_usd": t.explained_usd,
                         "unexplained_usd": t.unexplained_usd}[label]
            want = _cents(key_value)
            raw = [lab.fields[fld] for lab in labs]
            got = [LK.parse_amount(v) for v in raw]
            oks = [g is not None and abs(g - want) <= LK.AMOUNT_TOLERANCE_CENTS for g in got]
            add(cid, label, "", raw, oks, _money(key_value))

    stats = {"key_items": key_items, "key_items_eligible": eligible_key_items,
             "key_items_labeled_in_neither_pass": unlabeled_key_items,
             "labeled_references_not_in_the_key": extra_refs}
    return rows, stats


# --- Counting and writing -----------------------------------------------------------------

def _by_field(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {f: sum(1 for r in rows if r["field"] == f) for f in FIELD_ORDER}


def _cases(rows: Sequence[Mapping[str, str]]) -> list[str]:
    return sorted({r["case_id"] for r in rows})


def run(first, second, truths: Mapping[str, CaseTruth], *, seed: int,
        resamples: int | None = None) -> dict[str, Any]:
    """Both parts, in order: pass against pass, then labels against the key."""
    check_inputs(first, second, truths)
    kw = {} if resamples is None else {"resamples": resamples}
    kappa = LK.compare(first, second, seed=seed, **kw)
    p_rows = pass_disagreements(first, second, kappa)
    k_rows, k_stats = key_differences(first, second, truths)
    adjudicated = kappa["disagreed_cases"]["adjudicated_case_count"]["cases"]
    k_by_pass = {name: [r for r in k_rows if r["differs_from_key"] in (name, "both")] for name in ("first", "second")}
    summary = {
        "pass_vs_pass": {
            "kappa": kappa,
            "disagreements": {
                "rows": len(p_rows), "by_field": _by_field(p_rows), "cases": _cases(p_rows),
                "case_count": len(_cases(p_rows)),
                "same_cases_as_the_adjudicated_case_count": _cases(p_rows) == sorted(adjudicated),
            },
        },
        "label_vs_key": {
            "rows": len(k_rows), "by_field": _by_field(k_rows),
            "by_pass": {name: {"rows": len(rs), "by_field": _by_field(rs), "cases": _cases(rs),
                               "case_count": len(_cases(rs))} for name, rs in k_by_pass.items()},
            "cases": _cases(k_rows), "case_count": len(_cases(k_rows)),
            **k_stats,
        },
    }
    return {"rows": p_rows + k_rows, "summary": summary}


def format_counts(summary: Mapping[str, Any], dataset_version: str) -> str:
    """Counts only: no label, no key value, no case id, no reference."""
    p = summary["pass_vs_pass"]
    k = summary["label_vs_key"]
    g = p["kappa"]["gate"]
    adj = p["kappa"]["disagreed_cases"]["adjudicated_case_count"]["count"]

    def fields(d: Mapping[str, int]) -> str:
        return ", ".join(f"{f} {n}" for f, n in d.items())

    lines = [
        f"Labeling comparison, dataset version {dataset_version}",
        f"Cases: {p['kappa']['case_count']} in each pass and in the answer key",
        "1. Pass against pass",
        f"  items compared for the kappa: {g['items']} (labeled in both {g['items_labeled_in_both']}, "
        f"first only {g['items_labeled_in_first_only']}, second only {g['items_labeled_in_second_only']})",
        f"  adjudicated-case count: {adj}",
        f"  disagreement rows: {p['disagreements']['rows']} ({fields(p['disagreements']['by_field'])})",
        f"  cases with a disagreement: {p['disagreements']['case_count']}",
        "  the kappa, its CI and the name are in " + SUMMARY_JSON,
        "2. Labels against the answer key",
        f"  key items: {k['key_items']} ({k['key_items_eligible']} eligible); labeled in neither pass: "
        f"{k['key_items_labeled_in_neither_pass']}",
        f"  labeled references not in the key: first {k['labeled_references_not_in_the_key']['first']}, "
        f"second {k['labeled_references_not_in_the_key']['second']}",
        f"  difference rows: {k['rows']} ({fields(k['by_field'])})",
    ]
    for name in ("first", "second"):
        bp = k["by_pass"][name]
        lines.append(f"  {name} pass differs from the key: {bp['rows']} rows in {bp['case_count']} cases "
                     f"({fields(bp['by_field'])})")
    lines.append(f"  cases with a difference in either pass: {k['case_count']}")
    lines.append(f"Warnings: {len(p['kappa']['warnings'])} (listed in {SUMMARY_JSON})")
    return "\n".join(lines)


def check_out_dir(out: Path) -> Path:
    out = Path(out)
    resolved = out.resolve()
    repo = REPO.resolve()
    if resolved == repo or repo in resolved.parents:
        raise Refused(f"the --out folder {out} is inside the repository; its files hold labels and answer-key "
                      "values, so write them outside it")
    if resolved.exists() and not resolved.is_dir():
        raise Refused(f"the --out path {out} is not a folder")
    there = [n for n in OUTPUT_FILES if (resolved / n).exists()]
    if there:
        raise Refused(f"the --out folder already holds {', '.join(there)}; move them first, so an adjudication "
                      "sheet in progress is never replaced")
    return resolved


def write_outputs(out: Path, result: Mapping[str, Any], summary_json: Mapping[str, Any], text: str) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / name for name in OUTPUT_FILES}
    with open(paths[ADJUDICATION_FILE], "x", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ADJUDICATION_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        for r in result["rows"]:
            w.writerow(r)
    with open(paths[SUMMARY_JSON], "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary_json, indent=2) + "\n")
    with open(paths[SUMMARY_TEXT], "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return paths


def main(argv: Sequence[str] | None = None, *, resamples: int | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--first", nargs=2, metavar=("CSV", "KEY"),
                    help="the first pass: its CSV export and the key file of its sheet")
    ap.add_argument("--second", nargs=2, metavar=("CSV", "KEY"),
                    help="the second pass: its CSV export and the key file of its sheet")
    ap.add_argument("--dataset-version", type=Path,
                    help=f"JSON file holding {LK.VERSION_KEY} and the {LK.SEED_KEY} recorded with it")
    ap.add_argument("--answer-key", type=Path, help="the hand-written answer-key file (JSON Lines)")
    ap.add_argument("--out", type=Path, help="folder for the outputs, outside the repository")
    args = ap.parse_args(argv)
    try:
        if not args.first or not args.second:
            raise Refused("give both passes: --first CSV KEY and --second CSV KEY")
        if args.dataset_version is None:
            raise Refused("give --dataset-version: the file that records the bootstrap seed")
        if args.answer_key is None:
            raise Refused("give --answer-key: the hand-written answer-key file")
        if args.out is None:
            raise Refused("give --out: a folder outside the repository")
        out = check_out_dir(args.out)
        version = LK.read_dataset_version(args.dataset_version)
        first = LK.read_export(*args.first)
        second = LK.read_export(*args.second)
        truths = read_answer_key(args.answer_key)
        result = run(first, second, truths, seed=version["seed"], resamples=resamples)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    summary = {
        "tool": "scripts/labeling_compare.py",
        "rule": "pre-registration section 4, Branch B (decision D-R18); part 1 is labeling_kappa.compare",
        "dataset_version": version,
        "inputs": {
            "first": {"csv": Path(args.first[0]).name, "key": Path(args.first[1]).name},
            "second": {"csv": Path(args.second[0]).name, "key": Path(args.second[1]).name},
            "answer_key": {"file": args.answer_key.name, "sha256": _sha256(args.answer_key),
                           "cases": len(truths)},
        },
        "adjudication_csv": {"file": ADJUDICATION_FILE, "columns": list(ADJUDICATION_COLUMNS),
                             "rows": len(result["rows"]),
                             "note": "author_decision and author_reasoning are for the author; "
                                     "this tool never edits a label"},
        **result["summary"],
    }
    text = format_counts(result["summary"], version["dataset_version"])
    try:
        paths = write_outputs(out, result, summary, text)
    except FileExistsError as e:
        print(f"refused: {Path(e.filename).name} appeared in the --out folder while the tool ran", file=sys.stderr)
        return EXIT_REFUSED
    print(text)
    for p in paths.values():
        print(f"wrote {p}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
