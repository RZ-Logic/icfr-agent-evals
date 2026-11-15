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
on the 54 cases, matched by reference within a case. An item one pass labels and the
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
and the reason is given. Kappa is always computed and printed, but the subset also
keeps neither name, with the reason, when the passes do not hold the 54 cases the
gated kappa covers, or when the sheet's timer data does not show the 48-hour gap on
every case (section 4: the author re-labels after at least 48 hours; decision D-R20,
ruling 10). The statistic is labeled intra-rater (test-retest) kappa: it shows label
stability only.

The 48-hour gap, case by case (decision D-R20). On each case the second pass's start
on that case (its first_start_utc timer row) must come at least 48 hours after the
first pass's reference time on that case. The reference time is the first pass's
export time, or the first pass's last recorded time on that case (its last_stop_utc
timer row) when that is later. The export time is the moment the compared first-pass
labels were fixed, so the gap runs from it; a last stop after the export can only come
from a clock change, and taking the later of the two never shortens the gap. A case
lacks timer data when the first pass's export time is missing or unreadable, the
first pass's last stop on the case is given but unreadable, or the second pass gives
no readable start on the case. When any case lacks timer data, or any case shows a gap
under 48 hours, the name is withheld with the reason and the cases listed.

The adjudicated-case count (decision D-R20). Section 8 publishes the number of
adjudicated cases. It is the count of cases whose two passes differ on any compared
label: eligibility (any item), variance direction, band or evidence state (both passes
give a value and the values differ), or an amount (variance, explained or unexplained)
whose two readable values are more than $1 apart. The count of cases that differ on
eligibility is reported beside it as a breakdown, with the count per label.

Reported beside the gate, not gating: kappa for variance direction (unweighted)
and for band and evidence state (quadratic-weighted, over the sheet's three ordered
options), each over the cases where both passes give a value; the share of cases
whose variance, explained and unexplained amounts all match within $1; the
adjudicated-case count and its cases, with the breakdown by label; and the case count.

Refused: two exports of one sheet (the same sheet id), a rater that differs
between the passes, two key files that name different source files (or a key file
that names none), a case code the key does not hold, a key that does not match its
export, passes over different cases, a malformed row and a seed that differs from
the recorded one.

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
from datetime import datetime, timedelta, timezone
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
REFERENCE_EXPORT = "first_pass_export"
REFERENCE_LAST_STOP = "first_pass_last_stop_on_case"
GAP_RULE = ("on each case the second pass's start on that case is at least 48 hours after the first pass's "
            "export time, or after the first pass's last recorded time on that case when that is later "
            "(pre-registration section 4; decision D-R20)")
ADJUDICATED_DEFINITION = ("cases whose two passes differ on eligibility, variance direction, band or evidence "
                          "state, or on an amount whose two readable values are more than $1 apart "
                          "(pre-registration section 8; decision D-R20)")


class Refused(Exception):
    """An input the computation cannot use."""


# --- Reading the inputs -------------------------------------------------------------------

@dataclass
class CaseLabels:
    fields: dict[str, str] = field(default_factory=dict)
    items: dict[str, bool] = field(default_factory=dict)  # labeled items only: ref -> eligible
    first_start_utc: str = ""
    last_stop_utc: str = ""


@dataclass
class PassLabels:
    sheet_id: str
    rater: str
    exported_at_utc: str
    cases: dict[str, CaseLabels]  # case id -> labels
    csv_sha256: str
    key_sha256: str
    warnings: list[str] = field(default_factory=list)
    key_source: str = ""  # the case source file the key file names


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


def read_key_source(path: str | Path) -> str:
    """The case source file a key file names (its "source" entry, as labeling_sheet.py
    writes it). A key file that names none is refused."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path.name}: not a readable key file ({type(e).__name__})") from e
    source = data.get("source") if isinstance(data, dict) else None
    if not isinstance(source, str) or not source.strip():
        raise Refused(f"{path.name}: the key file names no source file")
    return source


def read_export(csv_path: str | Path, key_path: str | Path) -> PassLabels:
    """One pass: its CSV export decoded by its key file. Malformed rows are refused."""
    csv_path, key_path = Path(csv_path), Path(key_path)
    key_sid, codes = read_key(key_path)
    key_source = read_key_source(key_path)
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
                lab.first_start_utc = value.strip()
            elif fld == "last_stop_utc":
                lab.last_stop_utc = value.strip()
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
                      csv_sha256=_sha256(csv_path), key_sha256=_sha256(key_path), warnings=warnings,
                      key_source=key_source)


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
    if first.key_source != second.key_source:
        raise Refused(f"the two key files name different source files ({first.key_source!r}, "
                      f"{second.key_source!r}); both passes must label the same cases from one source")
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
    one_sided_amount: set[str] = set()
    for c in case_ids:
        got = [(parse_amount(first.cases[c].fields[f]), parse_amount(second.cases[c].fields[f]))
               for f in AMOUNT_FIELDS]
        apart = any(a is not None and b is not None and abs(a - b) > AMOUNT_TOLERANCE_CENTS for a, b in got)
        if any((a is None) != (b is None) for a, b in got):
            one_sided_amount.add(c)
        if any(a is None or b is None for a, b in got):
            unreadable += 1
            none_given = all(a is None and b is None for a, b in got)
            # an amount whose two readable values are more than $1 apart is a difference even
            # when another amount is missing (decision D-R20)
            statuses[c]["amounts"] = "differ" if apart else ("missing_both" if none_given else "missing")
        elif not apart:
            match += 1
            statuses[c]["amounts"] = "agree"
        else:
            statuses[c]["amounts"] = "differ"
    for c in case_ids:
        statuses[c]["eligibility"] = ("differ" if any(a != b for _, a, b, _ in per_case_items[c])
                                      else "agree")

    compared = ("eligibility", "variance_direction", "band", "evidence_state", "amounts")
    elig_queue = [c for c in case_ids if statuses[c]["eligibility"] == "differ"]
    adjudicated = [c for c in case_ids if any(statuses[c][lab] == "differ" for lab in compared)]
    by_label = {lab: sum(1 for c in case_ids if statuses[c][lab] == "differ") for lab in compared}
    one_sided = [c for c in case_ids if c not in adjudicated and (
        c in one_sided_amount
        or any(statuses[c][lab] in ("missing_first", "missing_second") for lab in compared))]

    warnings = list(first.warnings) + list(second.warnings)
    withheld: list[str] = []
    if len(case_ids) != LABELED_SUBSET_CASES:
        warnings.append(f"the passes hold {len(case_ids)} cases; the labeled subset of pre-registration "
                        f"section 4 holds {LABELED_SUBSET_CASES}")
        withheld.append(f"the passes hold {len(case_ids)} cases, not the {LABELED_SUBSET_CASES} cases the "
                        "gated kappa covers (pre-registration section 4)")
    timing = _timing(first, second, case_ids)
    lacking, short = timing["cases_without_timer_data"], timing["cases_under_48_hours"]
    if lacking:
        withheld.append(f"the 48-hour gap is not shown on every case: {len(lacking)} case(s) lack timer data "
                        f"({', '.join(lacking)}) (pre-registration section 4; decision D-R20)")
    if short:
        withheld.append(f"the sheet's timer data shows the second pass started less than 48 hours after the "
                        f"first pass on {len(short)} case(s) ({', '.join(short)}) (pre-registration section 4; "
                        "decision D-R20)")

    named = subset_name(kappa, reason)
    on_kappa = named["name"]
    if withheld:
        parts = list(withheld) + [named["reason"] if on_kappa is None else "the subset keeps neither name"]
        named = {"name": None, "reason": "; ".join(parts)}
    return {
        "statistic": STATISTIC_LABEL,
        "rule": "pre-registration section 4, Branch B (decision D-R18; decision D-R19; decision D-R20)",
        "gate": {
            "kappa": _as_float(kappa), "kappa_exact": None if kappa is None else str(kappa),
            "undefined_reason": reason, "threshold": float(NAME_THRESHOLD),
            "name": named["name"], "name_reason": named["reason"],
            "name_on_kappa_alone": on_kappa, "name_withheld_because": withheld,
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
            "adjudicated_case_count": {"count": len(adjudicated), "cases": adjudicated,
                                       "definition": ADJUDICATED_DEFINITION},
            "breakdown": {
                "on_eligibility": {"count": len(elig_queue), "cases": elig_queue},
                "by_label": by_label,
            },
            "label_given_in_one_pass_only": {"count": len(one_sided), "cases": one_sided,
                                             "note": "not in the adjudicated-case count"},
        },
        "case_count": len(case_ids),
        "cases": [{"case_id": c, "labels": statuses[c],
                   "items": [{"ref": r, "first": a, "second": b, "labeled_in": h}
                             for r, a, b, h in per_case_items[c]]} for c in case_ids],
        "passes": {
            name: {"sheet_id": p.sheet_id, "rater": p.rater, "exported_at_utc": p.exported_at_utc,
                   "csv_sha256": p.csv_sha256, "key_sha256": p.key_sha256, "key_source": p.key_source}
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


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timing(first: PassLabels, second: PassLabels, case_ids: Sequence[str]) -> dict[str, Any]:
    """The 48-hour gap on every case, from the sheet's own timer rows (decision D-R20).
    Per case: the reference time is the first pass's export time, or the first pass's
    last stop on the case when that is later; the gap runs from it to the second pass's
    start on the case. A case lacks timer data when a time it needs is missing or
    unreadable. compare withholds the name when any case lacks timer data or shows a
    gap under 48 hours."""
    exported = _parse_utc(first.exported_at_utc)
    rows: list[dict[str, Any]] = []
    lacking: list[str] = []
    short: list[str] = []
    for c in case_ids:
        f, s = first.cases[c], second.cases[c]
        last = _parse_utc(f.last_stop_utc)
        start = _parse_utc(s.first_start_utc)
        missing = None
        if exported is None:
            missing = "the first pass's export time is missing or unreadable"
        elif f.last_stop_utc and last is None:
            missing = "the first pass's last stop time on the case is unreadable"
        elif not s.first_start_utc:
            missing = "the second pass records no start time on the case"
        elif start is None:
            missing = "the second pass's start time on the case is unreadable"
        ref, source = exported, REFERENCE_EXPORT
        if exported is not None and last is not None and last > exported:
            ref, source = last, REFERENCE_LAST_STOP
        row: dict[str, Any] = {"case_id": c, "first_reference_utc": _iso(ref) if missing is None else None,
                               "reference": source if missing is None else None,
                               "second_start_utc": _iso(start), "hours": None}
        if missing is not None:
            row.update(status="no_timer_data", reason=missing)
            lacking.append(c)
        else:
            gap = start - ref
            row["hours"] = round(gap.total_seconds() / 3600, 2)
            if gap < timedelta(hours=RELABEL_GAP_HOURS):
                row["status"] = "under_48_hours"
                short.append(c)
            else:
                row["status"] = "at_least_48_hours"
        rows.append(row)
    hours = [r["hours"] for r in rows if r["hours"] is not None]
    return {"rule": GAP_RULE, "first_exported_at_utc": first.exported_at_utc,
            "gap_shown_on_every_case": bool(rows) and not lacking and not short,
            "cases_without_timer_data": lacking, "cases_under_48_hours": short,
            "smallest_gap_hours": min(hours) if hours else None, "cases": rows}


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
    adj, brk = dis["adjudicated_case_count"], dis["breakdown"]
    lines.append(f"Adjudicated-case count: {adj['count']} ({adj['definition']})")
    lines.append(f"  breakdown: {brk['on_eligibility']['count']} on eligibility; by label "
                 + ", ".join(f"{lab} {n}" for lab, n in brk["by_label"].items()))
    one = dis["label_given_in_one_pass_only"]
    if one["count"]:
        lines.append(f"  {one['count']} other case(s) hold a label given in one pass only, not in the "
                     f"adjudicated-case count: {', '.join(one['cases'])}")
    t = result["timing"]
    lines.append("48-hour gap: " + ("shown on every case" if t["gap_shown_on_every_case"] else "not shown on every case")
                 + f"; smallest gap {'undefined' if t['smallest_gap_hours'] is None else t['smallest_gap_hours']} "
                 f"hours; {len(t['cases_without_timer_data'])} case(s) without timer data; "
                 f"{len(t['cases_under_48_hours'])} case(s) under 48 hours")
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
