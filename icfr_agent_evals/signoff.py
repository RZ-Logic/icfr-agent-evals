"""The regression sign-off file: pre-registration section 10(c) and section 13 row 26.

Section 10(c) keeps a required status check red until a reviewer records one of
four sign-off outcomes for every triggered case. This module defines the file
that records them, reads it, and refuses any file that does not sign exactly the
cases of one run pair. docs/evidence/regression/SIGNOFF-FORMAT.md is the plain
English spec the owner ratifies.

Format. The file shares its format with the section 6 code-audit record
(fm_audit.csv, review_schema.RB7_COLUMNS): utf-8, no byte order mark, LF line
ends, the exact header first, one cell per column, read by review_schema.read_csv.
It has the RB-7 columns in order, then a `purpose` column (PURPOSE, which tells it
apart from a code-audit row) and four sign-off columns that bind it to one run
pair. One row per triggered case, that is per (sample_id, trigger):

  row_kind              "signoff"
  unit_id               blank
  run_label             "pr-<N>", the pull request number; the same in every row
  log_eval_id           the candidate run's eval_id; pooled re-runs (section 10(d))
                        give every eval_id, sorted, joined by " | "; the same in every row
  sample_id             the case id
  epoch                 blank (a sign-off is per case, not per epoch)
  scorers               blank
  route_key             the trigger: an FM code from regression.SIGNOFF_CODES, or one
                        route key, exactly as compare_runs.py prints it
  route_class           "fm_code" for an FM code, else regression.route_class(route_key)
  reason, sentence      blank
  verdict               one of OUTCOMES
  note                  required for every outcome
  reviewer              required
  date                  YYYY-MM-DD
  purpose               "regression_signoff"
  gated_content_sha256  the candidate's gated-content key (section 10(d)), 64 lowercase
                        hex; the same in every row
  baseline_eval_id      the baseline run's eval_id; the same in every row
  baseline_count        the baseline's failing-epoch count for the trigger
  candidate_count       the candidate's pooled failing-epoch count for the trigger

The committed path is SIGNOFF_DIR_RELPATH/pr-<N>-<first 12 hex of the key>.csv
(signoff_path). check_signoff_file refuses (GateError) a file that cannot be read,
a wrong header, a malformed or duplicate row, an unknown outcome or purpose, a
missing note, reviewer or date, a PR, gated-content key, baseline or candidate
eval_id other than the run pair's, a case the run pair does not trigger, a case
it triggers that the file leaves out, and counts other than the run pair's.
Outcome (a2) noise is defined as a single-epoch rise, so it is refused on a case
whose candidate count exceeds the baseline count by more than one.
"""

from __future__ import annotations

import datetime
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import review_schema as rs
from .regression import FABRICATION_CODES, SIGNOFF_CODES, SignoffCase, route_class
from .reliability import GateError

SIGNOFF_ROW_KIND = "signoff"
PURPOSE = "regression_signoff"
# The purposes of the shared format: the section 6 code audit and the section 10(c) sign-off.
PURPOSES: tuple[str, ...] = ("code_audit", PURPOSE)
FM_CODE_CLASS = "fm_code"
# The four section 10(c) outcomes: (a1) fixed, (a2) noise, (a3) accepted trade-off, (b) false positive.
OUTCOMES: tuple[str, ...] = ("a1_fixed", "a2_noise", "a3_accepted_trade_off", "b_false_positive")
NOISE, TRADE_OFF = OUTCOMES[1], OUTCOMES[2]
SIGNOFF_COLUMNS: tuple[str, ...] = (
    *rs.RB7_COLUMNS, "purpose", "gated_content_sha256", "baseline_eval_id", "baseline_count", "candidate_count",
)
BLANK_COLUMNS: tuple[str, ...] = ("unit_id", "epoch", "scorers", "reason", "sentence")
# Where sign-off files are committed, relative to the repository root.
SIGNOFF_DIR_RELPATH = "docs/evidence/regression/signoffs"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PR_RE = re.compile(r"^pr-([1-9][0-9]*)$")
_COUNT_RE = re.compile(r"^(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class RunPair:
    """What one sign-off file must name: the PR, the candidate's gated-content key,
    and the eval_ids of the baseline run and of every pooled candidate run."""
    pr: int
    content_key: str
    baseline_eval_id: str
    candidate_eval_ids: tuple[str, ...]

    @property
    def run_label(self) -> str:
        return f"pr-{self.pr}"

    @property
    def log_eval_id(self) -> str:
        return rs.RB7_LIST_SEP.join(sorted(self.candidate_eval_ids))


@dataclass(frozen=True)
class SignoffRecord:
    """A sign-off file that check_signoff_file accepted for one run pair."""
    path: str
    pair: RunPair
    outcomes: Mapping[tuple[str, str], str]   # (sample_id, trigger) -> outcome
    rows: tuple[Mapping[str, str], ...]

    def counts(self) -> dict[str, int]:
        """Cases per outcome, every outcome listed (section 10(c): counts per outcome are reported)."""
        found = Counter(self.outcomes.values())
        return {o: found.get(o, 0) for o in OUTCOMES}

    def policy_decisions(self) -> tuple[tuple[str, str], ...]:
        """(sample_id, code) of every accepted trade-off on a fabrication code, which
        section 10(c) reports as a policy decision, not free text."""
        return tuple(sorted(key for key, o in self.outcomes.items() if o == TRADE_OFF and key[1] in FABRICATION_CODES))


def check_content_key(key: str) -> str:
    if not isinstance(key, str) or not _HEX64_RE.match(key):
        raise GateError(f"gated-content key {key!r} is not 64 lowercase hex characters (a SHA-256)")
    return key


def check_pr(pr: int) -> int:
    if isinstance(pr, bool) or not isinstance(pr, int) or pr < 1:
        raise GateError(f"PR number {pr!r} is not a positive integer")
    return pr


def signoff_path(root: str | Path, pr: int, content_key: str) -> Path:
    """The committed sign-off file of one PR and gated-content key under the repository root."""
    return Path(root) / SIGNOFF_DIR_RELPATH / f"pr-{check_pr(pr)}-{check_content_key(content_key)[:12]}.csv"


def trigger_class(trigger: str) -> str:
    """route_class column value: fm_code for an FM code, else the route key's class."""
    return FM_CODE_CLASS if trigger in SIGNOFF_CODES else route_class(trigger)


def template_rows(pair: RunPair, cases: Sequence[SignoffCase]) -> list[dict[str, str]]:
    """One row per triggered case with every column but verdict, note, reviewer and
    date filled in, in the order compare_runs.py lists the cases."""
    return [{
        "row_kind": SIGNOFF_ROW_KIND, "run_label": pair.run_label, "log_eval_id": pair.log_eval_id,
        "sample_id": c.sample_id, "route_key": c.trigger, "route_class": trigger_class(c.trigger),
        "verdict": "", "note": "", "reviewer": "", "date": "", "purpose": PURPOSE,
        "gated_content_sha256": pair.content_key, "baseline_eval_id": pair.baseline_eval_id,
        "baseline_count": str(c.baseline_count), "candidate_count": str(c.candidate_count),
    } for c in cases]


def signoff_text(rows: Sequence[Mapping[str, str]]) -> str:
    """The exact bytes (as text) of a sign-off file holding rows."""
    return rs.csv_text(SIGNOFF_COLUMNS, rows)


def _row_errors(n: int, row: Mapping[str, str], pair: RunPair) -> list[str]:
    where = f"row {n}"
    errs: list[str] = []
    expected = {"row_kind": SIGNOFF_ROW_KIND, "purpose": PURPOSE}
    for col in ("row_kind", "purpose"):
        if row[col] != expected[col]:
            errs.append(f"{where}: {col} {row[col]!r}, expected {expected[col]!r}")
    if row["run_label"] != pair.run_label:
        if not _PR_RE.match(row["run_label"]):
            errs.append(f"{where}: run_label {row['run_label']!r} is not pr-<N>")
        else:
            errs.append(f"{where}: run_label {row['run_label']!r} signs another PR; this run pair is {pair.run_label}")
    if row["gated_content_sha256"] != pair.content_key:
        errs.append(f"{where}: gated_content_sha256 {row['gated_content_sha256']!r} does not match the run pair's "
                    f"gated-content key {pair.content_key}")
    if row["baseline_eval_id"] != pair.baseline_eval_id:
        errs.append(f"{where}: baseline_eval_id {row['baseline_eval_id']!r} is not the baseline run's "
                    f"{pair.baseline_eval_id!r}")
    if row["log_eval_id"] != pair.log_eval_id:
        errs.append(f"{where}: log_eval_id {row['log_eval_id']!r} is not the candidate run(s) "
                    f"{pair.log_eval_id!r}")
    for col in BLANK_COLUMNS:
        if row[col] != "":
            errs.append(f"{where}: {col} must be blank in a sign-off row")
    if not row["sample_id"].strip():
        errs.append(f"{where}: sample_id is blank")
    trigger = row["route_key"]
    try:
        cls = trigger_class(trigger)
    except GateError:
        errs.append(f"{where}: route_key {trigger!r} is neither a sign-off FM code nor a route key")
    else:
        if row["route_class"] != cls:
            errs.append(f"{where}: route_class {row['route_class']!r}, expected {cls!r} for {trigger!r}")
    if row["verdict"] not in OUTCOMES:
        errs.append(f"{where}: verdict {row['verdict']!r} is not one of the four outcomes {OUTCOMES}")
    for col in ("note", "reviewer"):
        if not row[col].strip():
            errs.append(f"{where}: {col} is blank")
    try:
        datetime.date.fromisoformat(row["date"])
    except ValueError:
        errs.append(f"{where}: date {row['date']!r} is not a YYYY-MM-DD date")
    for col in ("baseline_count", "candidate_count"):
        if not _COUNT_RE.match(row[col]):
            errs.append(f"{where}: {col} {row[col]!r} is not a whole number")
    return errs


def check_signoff_file(path: str | Path, pair: RunPair, cases: Sequence[SignoffCase]) -> SignoffRecord:
    """Read the sign-off file at path and accept it only if it signs exactly `cases`
    (the run pair's section 10(c) triggers) for exactly `pair`. Every problem found
    is named in one GateError."""
    check_pr(pair.pr)
    check_content_key(pair.content_key)
    p = Path(path)
    try:
        rows = rs.read_csv(p, SIGNOFF_COLUMNS)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise GateError(f"sign-off file {p}: cannot read it ({type(e).__name__}: {e})") from e
    if not rows:
        raise GateError(f"sign-off file {p}: holds no sign-off rows")
    errs: list[str] = []
    for n, row in enumerate(rows, start=2):
        errs.extend(_row_errors(n, row, pair))
    if errs:
        raise GateError(f"sign-off file {p}: " + "; ".join(errs))

    wanted = {(c.sample_id, c.trigger): c for c in cases}
    seen: dict[tuple[str, str], int] = {}
    for n, row in enumerate(rows, start=2):
        key = (row["sample_id"], row["route_key"])
        if key in seen:
            errs.append(f"row {n}: {key[0]} {key[1]} is signed twice (first in row {seen[key]})")
            continue
        seen[key] = n
        case = wanted.get(key)
        if case is None:
            errs.append(f"row {n}: {key[0]} {key[1]} is not a sign-off case of this run pair")
            continue
        counts = (int(row["baseline_count"]), int(row["candidate_count"]))
        if counts != (case.baseline_count, case.candidate_count):
            errs.append(f"row {n}: {key[0]} {key[1]} counts baseline {counts[0]} -> candidate {counts[1]}, "
                        f"but this run pair has {case.baseline_count} -> {case.candidate_count}")
        elif row["verdict"] == NOISE and case.candidate_count - case.baseline_count != 1:
            errs.append(f"row {n}: {key[0]} {key[1]} is recorded as {NOISE}, which section 10(c) defines as a "
                        f"single-epoch rise; this case rose {case.candidate_count - case.baseline_count} epochs")
    missing = [k for k in wanted if k not in seen]
    if missing:
        errs.append("unsigned case(s): " + ", ".join(f"{s} {t}" for s, t in missing))
    if errs:
        raise GateError(f"sign-off file {p}: " + "; ".join(errs))
    outcomes = {(r["sample_id"], r["route_key"]): r["verdict"] for r in rows}
    return SignoffRecord(path=str(p), pair=pair, outcomes=outcomes, rows=tuple(rows))
