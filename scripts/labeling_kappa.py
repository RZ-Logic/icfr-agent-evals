"""The Branch B labeling kappa of pre-registration section 4 (Labeling protocol;
Branch B in force, decision D-R18; gate pinned by decision D-R19), reported as
section 8 asks: the labeling kappa with its 95% CI and the adjudicated-case count.

Usage:
  python scripts/labeling_kappa.py --first CSV KEY --second CSV KEY --dataset-version FILE
                                   [--seed N] [--out JSON]

Inputs. The author labels every case twice, on two sheets built by
scripts/labeling_sheet.py. Each pass is one CSV export from its sheet and the key
file of that sheet (<sheet>.<sheet id>.key.json), which decodes the case codes to
case ids. The script reads those four files and the dataset-version file. It reads
no case file, no dataset, no answer key and nothing under data/.

The gated statistic. Unweighted Cohen's kappa between the two passes on per-item
eligibility (eligible or not). It covers every reconciling item either pass labels
on the cases, matched by reference within a case. An item one pass labels and the
other omits counts as not eligible in the pass that omits it. An item counts as
labeled in a pass when its eligibility answer is yes or no; a row left blank is
an omission. The 95% CI is a percentile bootstrap of 10,000 resamples of cases,
drawn with replacement, in which all items of a drawn case move together:
lower = sorted[floor(0.025 B)], upper = sorted[ceil(0.975 B) - 1] over the B
defined resamples. A resample whose kappa is undefined is dropped and the count is
reported.

The seed. Section 4 records the bootstrap seed with the dataset version before the
two passes are compared. The script reads it from the dataset-version file, a JSON
object holding "dataset_version" and the integer "labeling_kappa_seed". A --seed
that differs from the recorded seed is refused.

The name. The subset earns the name Self-verified only when the point estimate is
at least 0.70, and never the name Verified. When kappa is undefined (both passes
put every item in one class, or no item is labeled) the subset keeps neither name
and the reason is given. The statistic is labeled intra-rater (test-retest) kappa:
it shows label stability only.

Reported beside the gate, not gating: kappa for variance direction (unweighted)
and for band and evidence state (quadratic-weighted, over the sheet's three ordered
options), each over the cases where both passes give a value; the share of cases
whose variance, explained and unexplained amounts all match within $1; the cases
whose labels differ (the queue for adjudication), by label; and the case count.

Refused: two exports of one sheet (the same sheet id), a rater that differs
between the passes, a case code the key does not hold, a key that does not match
its export, passes over different cases, a malformed row and a seed that differs
from the recorded one.

Exit codes: 0 the result printed (and written, with --out); 3 refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_OK, EXIT_REFUSED = 0, 3

# The CSV export of scripts/labeling_sheet.py (its buildCsv function).
CSV_COLUMNS: tuple[str, ...] = (
    "sheet_id", "rater", "case", "item_ref", "item_source", "field", "value", "exported_at_utc",
)
CASE_FIELDS: tuple[str, ...] = (
    "r_variance", "r_dir", "r_band", "r_explained", "r_unexplained", "r_evidence",
    "r_flags", "r_passage", "r_handling", "r_note", "r_done",
)
ITEM_FIELDS: tuple[str, ...] = ("r_eligible", "r_reasons", "r_effect", "r_item_note")
TIMER_FIELDS: tuple[str, ...] = (
    "timer_seconds", "timer_sessions", "timer_running_at_export", "timer_interrupted",
    "first_start_utc", "last_stop_utc",
)
ITEM_SOURCES: tuple[str, ...] = ("schedule", "added")
ELIGIBLE_VALUES: tuple[str, ...] = ("yes", "no", "")

# The ordered options of the sheet. Evidence: fully explained, partly unexplained,
# unexplained, the order of casefile.EVIDENCE_STATES.
DIRECTION_SCALE: tuple[str, ...] = ("surplus", "shortfall")
BAND_SCALE: tuple[str, ...] = ("M1", "M2", "M3")
EVIDENCE_SCALE: tuple[str, ...] = ("full", "part", "none")
CHOICE_FIELDS: dict[str, tuple[str, ...]] = {
    "r_dir": DIRECTION_SCALE, "r_band": BAND_SCALE, "r_evidence": EVIDENCE_SCALE,
}
AMOUNT_FIELDS: tuple[str, ...] = ("r_variance", "r_explained", "r_unexplained")

BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95
NAME_THRESHOLD = Fraction(7, 10)
SUBSET_NAME = "Self-verified"
STATISTIC_LABEL = "intra-rater (test-retest) Cohen's kappa"
SEED_KEY = "labeling_kappa_seed"
VERSION_KEY = "dataset_version"
LABELED_SUBSET_CASES = 54
AMOUNT_TOLERANCE_CENTS = 100  # $1
RELABEL_GAP_HOURS = 48


class Refused(Exception):
    """An input the computation cannot use."""


# --- Reading the inputs -------------------------------------------------------------------

@dataclass
class CaseLabels:
    fields: dict[str, str] = field(default_factory=dict)
    items: dict[str, bool] = field(default_factory=dict)  # labeled items only: ref -> eligible
    first_start_utc: str = ""


@dataclass
class PassLabels:
    sheet_id: str
    rater: str
    exported_at_utc: str
    cases: dict[str, CaseLabels]  # case id -> labels
    csv_sha256: str
    key_sha256: str
    warnings: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_key(path: str | Path) -> tuple[str, dict[str, str]]:
    """(sheet id, code -> case id) from a key file written by labeling_sheet.py."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path.name}: not a readable key file ({type(e).__name__})") from e
    if not isinstance(data, dict):
        raise Refused(f"{path.name}: the key file is not a JSON object")
    sid = data.get("sheet_id")
    cases = data.get("cases")
    if not isinstance(sid, str) or not sid:
        raise Refused(f"{path.name}: the key file has no sheet id")
    if not isinstance(cases, list) or not cases:
        raise Refused(f"{path.name}: the key file lists no cases")
    codes: dict[str, str] = {}
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("code"), str) or not isinstance(row.get("case_id"), str):
            raise Refused(f"{path.name}: a key row lacks a code or a case id")
        if not row["code"] or not row["case_id"]:
            raise Refused(f"{path.name}: a key row has an empty code or case id")
        if row["code"] in codes:
            raise Refused(f"{path.name}: code {row['code']} is listed twice")
        codes[row["code"]] = row["case_id"]
    if len(set(codes.values())) != len(codes):
        raise Refused(f"{path.name}: a case id is listed under two codes")
    return sid, codes


def read_export(csv_path: str | Path, key_path: str | Path) -> PassLabels:
    """One pass: its CSV export decoded by its key file. Malformed rows are refused."""
    csv_path, key_path = Path(csv_path), Path(key_path)
    key_sid, codes = read_key(key_path)
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"{csv_path.name}: not a readable CSV file ({type(e).__name__})") from e
    rows = list(csv.reader(text.splitlines()))
    rows = [r for r in rows if r]  # a trailing empty line is not a row
    if not rows or tuple(rows[0]) != CSV_COLUMNS:
        raise Refused(f"{csv_path.name}: the header is not the labeling sheet's ({', '.join(CSV_COLUMNS)})")
    body = rows[1:]
    if not body:
        raise Refused(f"{csv_path.name}: no rows under the header")
    where = csv_path.name
    sheet_ids, raters, stamps = set(), set(), set()
    for n, r in enumerate(body, 2):
        if len(r) != len(CSV_COLUMNS):
            raise Refused(f"{where} line {n}: {len(r)} cells, not {len(CSV_COLUMNS)}")
        sheet_ids.add(r[0])
        raters.add(r[1].strip())
        stamps.add(r[7])
    if len(sheet_ids) != 1:
        raise Refused(f"{where}: rows from more than one sheet ({', '.join(sorted(sheet_ids))})")
    sid = sheet_ids.pop()
    if sid != key_sid:
        raise Refused(f"{where}: sheet {sid}, but the key file decodes sheet {key_sid}")
    if len(raters) != 1:
        raise Refused(f"{where}: rows from more than one rater")
    rater = raters.pop()
    if not rater:
        raise Refused(f"{where}: no rater initials in the export")
    if len(stamps) != 1:
        raise Refused(f"{where}: rows from more than one export time")
    stamp = stamps.pop()

    cases: dict[str, CaseLabels] = {}
    item_rows: dict[str, list[tuple[int, str, str, dict[str, str]]]] = {}
    pending: tuple[int, str, str, str, dict[str, str]] | None = None  # line, code, src, ref, fields

    def close_item() -> None:
        nonlocal pending
        if pending is None:
            return
        line, code, src, ref, got = pending
        if tuple(got) != ITEM_FIELDS:
            raise Refused(f"{where} line {line}: item {ref or '(no reference)'} does not carry the fields "
                          f"{', '.join(ITEM_FIELDS)} in order")
        item_rows.setdefault(code, []).append((line, src, ref, got))
        pending = None

    for n, r in enumerate(body, 2):
        code, ref, src, fld, value = r[2], r[3], r[4], r[5], r[6]
        if code not in codes:
            raise Refused(f"{where} line {n}: case code {code!r} is not in the key file")
        cid = codes[code]
        lab = cases.setdefault(cid, CaseLabels())
        if src in ITEM_SOURCES:
            if fld not in ITEM_FIELDS:
                raise Refused(f"{where} line {n}: {fld!r} is not an item field")
            if fld == ITEM_FIELDS[0]:
                close_item()
                if value not in ELIGIBLE_VALUES:
                    raise Refused(f"{where} line {n}: eligibility {value!r} is not yes, no or blank")
                pending = (n, code, src, ref.strip(), {fld: value})
            else:
                if pending is None or pending[1] != code or pending[2] != src or pending[3] != ref.strip() \
                        or fld in pending[4]:
                    raise Refused(f"{where} line {n}: item field {fld} is out of order")
                pending[4][fld] = value
            continue
        close_item()
        if src not in ("case", "timer"):
            raise Refused(f"{where} line {n}: item source {src!r} is not case, schedule, added or timer")
        if ref:
            raise Refused(f"{where} line {n}: a {src} row carries an item reference")
        if src == "case":
            if fld not in CASE_FIELDS:
                raise Refused(f"{where} line {n}: {fld!r} is not a case field")
            if fld in lab.fields:
                raise Refused(f"{where} line {n}: case field {fld} is given twice")
            allowed = CHOICE_FIELDS.get(fld)
            if allowed is not None and value not in allowed + ("",):
                raise Refused(f"{where} line {n}: {fld} {value!r} is not one of {', '.join(allowed)} or blank")
            lab.fields[fld] = value
        else:
            if fld not in TIMER_FIELDS:
                raise Refused(f"{where} line {n}: {fld!r} is not a timer field")
            if fld == "first_start_utc":
                lab.first_start_utc = value
    close_item()

    missing_cases = sorted(c for c in codes if codes[c] not in cases)
    if missing_cases:
        raise Refused(f"{where}: the key holds case codes the export lacks ({', '.join(missing_cases)})")
    for cid, lab in cases.items():
        absent = [f for f in CASE_FIELDS if f not in lab.fields]
        if absent:
            raise Refused(f"{where}: case {cid} lacks the case fields {', '.join(absent)}")

    warnings: list[str] = []
    by_case = {codes[c]: rows_ for c, rows_ in item_rows.items()}
    for cid, items in by_case.items():
        labeled = cases[cid].items
        for line, _src, ref, got in items:
            ans = got["r_eligible"]
            if not ans:
                continue  # not labeled in this pass
            if not ref:
                raise Refused(f"{where} line {line}: case {cid} has a labeled item with no reference")
            value = ans == "yes"
            if ref in labeled:
                if labeled[ref] != value:
                    raise Refused(f"{where} line {line}: case {cid} labels reference {ref} both eligible and not")
                warnings.append(f"{where}: case {cid} labels reference {ref} twice with the same answer; "
                                "it counts once")
                continue
            labeled[ref] = value
    for cid, lab in cases.items():
        folded: dict[str, list[str]] = {}
        for ref in lab.items:
            folded.setdefault(ref.casefold(), []).append(ref)
        for refs in folded.values():
            if len(refs) > 1:
                warnings.append(f"{where}: case {cid} holds references that differ only in letter case "
                                f"({', '.join(sorted(refs))}); they are matched as written")
    return PassLabels(sheet_id=sid, rater=rater, exported_at_utc=stamp, cases=cases,
                      csv_sha256=_sha256(csv_path), key_sha256=_sha256(key_path), warnings=warnings)


def read_dataset_version(path: str | Path) -> dict[str, Any]:
    """The dataset version and the bootstrap seed recorded with it."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path.name}: not a readable dataset-version file ({type(e).__name__})") from e
    if not isinstance(data, dict):
        raise Refused(f"{path.name}: the dataset-version file is not a JSON object")
    version = data.get(VERSION_KEY)
    seed = data.get(SEED_KEY)
    if not isinstance(version, str) or not version:
        raise Refused(f"{path.name}: no {VERSION_KEY}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Refused(f"{path.name}: no non-negative integer {SEED_KEY} is recorded with the dataset version")
    return {"dataset_version": version, "seed": seed, "file": path.name, "sha256": _sha256(path)}


# --- Amounts ------------------------------------------------------------------------------

def parse_amount(text: str | None) -> int | None:
    """Cents from a typed amount, as the sheet's own check reads it: a leading minus
    sign (after an optional $ or USD) or parentheses make it negative, and every
    character but digits and one point is dropped. None when it holds no digit or
    more than one point."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    neg = bool(re.match(r"^\(.*\)$", t) or t.startswith("-") or re.match(r"^\$?\s*-", t)
               or re.match(r"^USD\s*-", t, re.I))
    d = re.sub(r"[^0-9.]", "", t)
    if not re.search(r"[0-9]", d) or d.count(".") > 1:
        return None
    v = int((Decimal(d) * 100).to_integral_value(rounding=ROUND_HALF_UP))
    return -v if neg else v


# --- Kappa --------------------------------------------------------------------------------

def binary_kappa(n: int, yes_first: int, yes_second: int, agree: int) -> Fraction | None:
    """Unweighted Cohen's kappa from counts: n items, the items each pass marks
    eligible, and the items the passes agree on. None when it is undefined: no item,
    or p_e = 1 (both passes put every item in the same class)."""
    if n == 0:
        return None
    expected = yes_first * yes_second + (n - yes_first) * (n - yes_second)  # p_e * n^2
    denom = n * n - expected
    if denom == 0:
        return None
    return Fraction(agree * n - expected, denom)


def _counts(pairs: Sequence[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    return (len(pairs), sum(1 for a, _ in pairs if a), sum(1 for _, b in pairs if b),
            sum(1 for a, b in pairs if a == b))


def weighted_kappa(pairs: Sequence[tuple[str, str]], scale: Sequence[str],
                   weights: str = "quadratic") -> Fraction | None:
    """Cohen's kappa over the ordered options `scale`: 1 - sum(w o) / sum(w e), with
    quadratic weights (i - j)^2 / (k - 1)^2 or unweighted ones (0 on the diagonal,
    1 off it). None when no pair is given or the expected disagreement is 0."""
    if weights not in ("quadratic", "unweighted"):
        raise ValueError(f"unknown weights {weights!r}")
    k = len(scale)
    if k < 2:
        raise ValueError("a scale needs at least two options")
    index = {v: i for i, v in enumerate(scale)}
    n = len(pairs)
    if n == 0:
        return None
    for a, b in pairs:
        if a not in index or b not in index:
            raise ValueError(f"value outside the scale: {a!r}, {b!r}")

    def w(i: int, j: int) -> Fraction:
        if weights == "unweighted":
            return Fraction(0 if i == j else 1)
        return Fraction((i - j) ** 2, (k - 1) ** 2)

    row = [0] * k
    col = [0] * k
    observed = Fraction(0)
    for a, b in pairs:
        i, j = index[a], index[b]
        row[i] += 1
        col[j] += 1
        observed += w(i, j)
    observed /= n
    expected = sum((w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)), Fraction(0)) / (n * n)
    if expected == 0:
        return None
    return 1 - observed / expected


def _as_float(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def eligibility_pairs(first: PassLabels, second: PassLabels) -> dict[str, list[tuple[str, bool, bool, str]]]:
    """Per case id: (reference, first eligible, second eligible, how labeled) for
    every item either pass labels, matched by reference. An item one pass omits is
    not eligible in that pass. How labeled: both, first_only or second_only."""
    out: dict[str, list[tuple[str, bool, bool, str]]] = {}
    for cid in sorted(first.cases):
        a, b = first.cases[cid].items, second.cases[cid].items
        rows = []
        for ref in sorted(set(a) | set(b)):
            how = "both" if ref in a and ref in b else ("first_only" if ref in a else "second_only")
            rows.append((ref, a.get(ref, False), b.get(ref, False), how))
        out[cid] = rows
    return out


def bootstrap_case_kappa(case_counts: Sequence[tuple[int, int, int, int]], *, seed: int,
                         resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Percentile bootstrap of the binary kappa over cases drawn with replacement,
    all items of a drawn case together. case_counts holds per case (items, first
    eligible, second eligible, agreed). Undefined resamples are dropped and counted."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    m = len(case_counts)
    rng = random.Random(seed)
    values: list[float] = []
    undefined = 0
    if m:
        idx = range(m)
        for _ in range(resamples):
            n = ya = yb = ag = 0
            for i in rng.choices(idx, k=m):
                c = case_counts[i]
                n += c[0]
                ya += c[1]
                yb += c[2]
                ag += c[3]
            k = binary_kappa(n, ya, yb, ag)
            if k is None:
                undefined += 1
            else:
                values.append(float(k))
    else:
        undefined = resamples
    values.sort()
    kept = len(values)
    if kept:
        lower = values[math.floor(0.025 * kept)]
        upper = values[math.ceil(0.975 * kept) - 1]
    else:
        lower = upper = None
    return {"method": "percentile bootstrap over cases", "level": CI_LEVEL, "seed": seed,
            "resamples": resamples, "defined_resamples": kept, "undefined_resamples_dropped": undefined,
            "ci_lower": lower, "ci_upper": upper}


def subset_name(kappa: Fraction | None, undefined_reason: str | None) -> dict[str, Any]:
    """Self-verified when the point estimate is at least 0.70; never Verified."""
    if kappa is None:
        return {"name": None, "reason": f"kappa is undefined: {undefined_reason}; the subset keeps neither name"}
    if kappa >= NAME_THRESHOLD:
        return {"name": SUBSET_NAME, "reason": "the point estimate is at least 0.70"}
    return {"name": None, "reason": "the point estimate is below 0.70; the subset keeps neither name"}


def _undefined_reason(pairs: Sequence[tuple[bool, bool]]) -> str | None:
    if not pairs:
        return "no item is labeled"
    if all(a for a, b in pairs) and all(b for a, b in pairs):
        return "both passes put every item in one class (eligible)"
    if not any(a for a, b in pairs) and not any(b for a, b in pairs):
        return "both passes put every item in one class (not eligible)"
    return None


def _choice_status(a: str, b: str) -> str:
    if a and b:
        return "agree" if a == b else "differ"
    if a or b:
        return "missing_first" if not a else "missing_second"
    return "missing_both"


def compare(first: PassLabels, second: PassLabels, *, seed: int,
            resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Every figure of the report from two decoded passes."""
    if first.sheet_id == second.sheet_id:
        raise Refused(f"both exports come from sheet {first.sheet_id}: the two passes must be two sheets")
    if first.rater != second.rater:
        raise Refused(f"the passes have different raters ({first.rater!r}, {second.rater!r}); "
                      "Branch B is one rater labeling twice")
    if set(first.cases) != set(second.cases):
        only1 = sorted(set(first.cases) - set(second.cases))
        only2 = sorted(set(second.cases) - set(first.cases))
        raise Refused("the passes cover different cases (first only: " + (", ".join(only1) or "none")
                      + "; second only: " + (", ".join(only2) or "none") + ")")

    per_case_items = eligibility_pairs(first, second)
    all_pairs = [(a, b) for rows in per_case_items.values() for _, a, b, _ in rows]
    kappa = binary_kappa(*_counts(all_pairs))
    reason = _undefined_reason(all_pairs) if kappa is None else None
    case_ids = sorted(per_case_items)
    case_counts = [_counts([(a, b) for _, a, b, _ in per_case_items[c]]) for c in case_ids]
    boot = bootstrap_case_kappa(case_counts, seed=seed, resamples=resamples)
    if kappa is None:
        boot["ci_lower"] = boot["ci_upper"] = None
    how = [h for rows in per_case_items.values() for *_, h in rows]

    side: dict[str, Any] = {}
    statuses: dict[str, dict[str, str]] = {c: {} for c in case_ids}
    for fld, label, scale, weights in (("r_dir", "variance_direction", DIRECTION_SCALE, "unweighted"),
                                       ("r_band", "band", BAND_SCALE, "quadratic"),
                                       ("r_evidence", "evidence_state", EVIDENCE_SCALE, "quadratic")):
        pairs = []
        for c in case_ids:
            a, b = first.cases[c].fields[fld], second.cases[c].fields[fld]
            statuses[c][label] = _choice_status(a, b)
            if a and b:
                pairs.append((a, b))
        k = weighted_kappa(pairs, scale, weights)
        side[label] = {"kappa": _as_float(k), "kappa_exact": None if k is None else str(k),
                       "weights": weights, "scale": list(scale), "cases": len(pairs),
                       "cases_left_out_for_a_missing_value": len(case_ids) - len(pairs)}

    match, unreadable = 0, 0
    for c in case_ids:
        got = [(parse_amount(first.cases[c].fields[f]), parse_amount(second.cases[c].fields[f]))
               for f in AMOUNT_FIELDS]
        if any(a is None or b is None for a, b in got):
            unreadable += 1
            none_given = all(a is None and b is None for a, b in got)
            statuses[c]["amounts"] = "missing_both" if none_given else "missing"
        elif all(abs(a - b) <= AMOUNT_TOLERANCE_CENTS for a, b in got):
            match += 1
            statuses[c]["amounts"] = "agree"
        else:
            statuses[c]["amounts"] = "differ"
    for c in case_ids:
        statuses[c]["eligibility"] = ("differ" if any(a != b for _, a, b, _ in per_case_items[c])
                                      else "agree")

    elig_queue = [c for c in case_ids if statuses[c]["eligibility"] == "differ"]
    any_queue = [c for c in case_ids if any(s != "agree" and s != "missing_both" for s in statuses[c].values())]
    by_label = {lab: sum(1 for c in case_ids if statuses[c][lab] not in ("agree", "missing_both"))
                for lab in ("eligibility", "variance_direction", "band", "evidence_state", "amounts")}

    warnings = list(first.warnings) + list(second.warnings)
    if len(case_ids) != LABELED_SUBSET_CASES:
        warnings.append(f"the passes hold {len(case_ids)} cases; the labeled subset of pre-registration "
                        f"section 4 holds {LABELED_SUBSET_CASES}")
    timing = _timing(first, second)
    if timing["hours_from_first_export_to_second_start"] is not None \
            and timing["hours_from_first_export_to_second_start"] < RELABEL_GAP_HOURS:
        warnings.append("the second pass started less than 48 hours after the first pass was exported "
                        "(pre-registration section 4 asks for at least 48 hours)")

    named = subset_name(kappa, reason)
    return {
        "statistic": STATISTIC_LABEL,
        "rule": "pre-registration section 4, Branch B (decision D-R18; decision D-R19)",
        "gate": {
            "kappa": _as_float(kappa), "kappa_exact": None if kappa is None else str(kappa),
            "undefined_reason": reason, "threshold": float(NAME_THRESHOLD),
            "name": named["name"], "name_reason": named["reason"],
            "items": len(all_pairs), "items_labeled_in_both": how.count("both"),
            "items_labeled_in_first_only": how.count("first_only"),
            "items_labeled_in_second_only": how.count("second_only"),
            "eligible_in_first": sum(1 for a, _ in all_pairs if a),
            "eligible_in_second": sum(1 for _, b in all_pairs if b),
            "agreed_items": sum(1 for a, b in all_pairs if a == b),
            "ci": boot,
        },
        "beside_not_gating": {
            **side,
            "amounts_within_1_usd": {
                "share": (match / len(case_ids)) if case_ids else None,
                "cases_matching": match, "cases": len(case_ids),
                "cases_with_a_missing_or_unreadable_amount": unreadable,
                "fields": list(AMOUNT_FIELDS), "tolerance_usd": AMOUNT_TOLERANCE_CENTS / 100,
            },
        },
        "disagreed_cases": {
            "on_eligibility": {"count": len(elig_queue), "cases": elig_queue},
            "on_any_compared_label": {"count": len(any_queue), "cases": any_queue},
            "by_label": by_label,
        },
        "case_count": len(case_ids),
        "cases": [{"case_id": c, "labels": statuses[c],
                   "items": [{"ref": r, "first": a, "second": b, "labeled_in": h}
                             for r, a, b, h in per_case_items[c]]} for c in case_ids],
        "passes": {
            name: {"sheet_id": p.sheet_id, "rater": p.rater, "exported_at_utc": p.exported_at_utc,
                   "csv_sha256": p.csv_sha256, "key_sha256": p.key_sha256}
            for name, p in (("first", first), ("second", second))
        },
        "timing": timing,
        "warnings": warnings,
    }


def _parse_utc(s: str) -> datetime | None:
    if not s:
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _timing(first: PassLabels, second: PassLabels) -> dict[str, Any]:
    """Hours from the first pass's export to the earliest case start of the second
    pass, from the sheet's own timer rows. Shown, not enforced."""
    exported = _parse_utc(first.exported_at_utc)
    starts = [t for t in (_parse_utc(c.first_start_utc) for c in second.cases.values()) if t is not None]
    hours = None
    if exported is not None and starts:
        hours = round((min(starts) - exported).total_seconds() / 3600, 2)
    return {"first_exported_at_utc": first.exported_at_utc,
            "second_earliest_case_start_utc": min(starts).strftime("%Y-%m-%dT%H:%M:%SZ") if starts else None,
            "hours_from_first_export_to_second_start": hours}


# --- Report -------------------------------------------------------------------------------

def _fmt(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def format_summary(result: Mapping[str, Any]) -> str:
    g = result["gate"]
    ci = g["ci"]
    side = result["beside_not_gating"]
    dis = result["disagreed_cases"]
    lines = [
        f"Labeling kappa, Branch B: {result['statistic']} on per-item eligibility "
        "(label stability only)",
        f"  cases {result['case_count']}, items {g['items']} (labeled in both {g['items_labeled_in_both']}, "
        f"first only {g['items_labeled_in_first_only']}, second only {g['items_labeled_in_second_only']})",
        f"  kappa {_fmt(g['kappa'])}" + (f" ({g['undefined_reason']})" if g["undefined_reason"] else ""),
        f"  95% CI {_fmt(ci['ci_lower'])} to {_fmt(ci['ci_upper'])}: {ci['method']}, {ci['resamples']} resamples, "
        f"seed {ci['seed']}, {ci['undefined_resamples_dropped']} undefined resamples dropped",
        f"  name: {g['name'] or 'neither name'} ({g['name_reason']})",
        "Beside the gate, not gating:",
        f"  variance direction kappa (unweighted) {_fmt(side['variance_direction']['kappa'])} "
        f"over {side['variance_direction']['cases']} cases",
        f"  band kappa (quadratic-weighted) {_fmt(side['band']['kappa'])} over {side['band']['cases']} cases",
        f"  evidence state kappa (quadratic-weighted) {_fmt(side['evidence_state']['kappa'])} "
        f"over {side['evidence_state']['cases']} cases",
    ]
    am = side["amounts_within_1_usd"]
    share = "undefined" if am["share"] is None else f"{am['share']:.3f}"
    lines.append(f"  variance, explained and unexplained amounts all within $1: {am['cases_matching']} of "
                 f"{am['cases']} cases ({share}); {am['cases_with_a_missing_or_unreadable_amount']} with a "
                 "missing or unreadable amount")
    lines.append(f"Disagreed cases: {dis['on_eligibility']['count']} on eligibility, "
                 f"{dis['on_any_compared_label']['count']} on any compared label")
    for w in result["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--first", nargs=2, metavar=("CSV", "KEY"), required=True,
                    help="the first pass: its CSV export and the key file of its sheet")
    ap.add_argument("--second", nargs=2, metavar=("CSV", "KEY"), required=True,
                    help="the second pass: its CSV export and the key file of its sheet")
    ap.add_argument("--dataset-version", required=True, type=Path,
                    help=f"JSON file holding {VERSION_KEY} and the {SEED_KEY} recorded with it")
    ap.add_argument("--seed", type=int, help="optional; refused unless it equals the recorded seed")
    ap.add_argument("--out", type=Path, help="write the full result as JSON here")
    args = ap.parse_args(argv)
    try:
        version = read_dataset_version(args.dataset_version)
        if args.seed is not None and args.seed != version["seed"]:
            raise Refused(f"seed {args.seed} differs from the seed {version['seed']} recorded with dataset "
                          f"version {version['dataset_version']}")
        first = read_export(*args.first)
        second = read_export(*args.second)
        result = compare(first, second, seed=version["seed"])
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    result["dataset_version"] = version
    print(format_summary(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
