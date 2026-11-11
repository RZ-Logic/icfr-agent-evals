"""Review record: validate, freeze, second look and blinding check. NO API key, no model call.

Usage:
  python scripts/review_freeze.py validate --packet-dir D --filled F.csv
  python scripts/review_freeze.py freeze --packet-dir D --filled F.csv --run LABEL=PATH [...]
         --reviewer NAME --date YYYY-MM-DD --run-commit SHA (--sittings S.csv | --sitting START/END [...])
         [--manifest-dir docs/evidence/reviews] [--expanded-csv PATH]
  python scripts/review_freeze.py second-look-draw --packet-dir D --freeze-commit SHA
         [--out D-second-look] [--repo .] [--manifest docs/evidence/reviews/<run-id>-manifest.json]
  python scripts/review_freeze.py blind-check-draw --packet-dir D --freeze-commit SHA
         --models LABEL=NAME,LABEL=NAME,... [--out D-blind-check] [--repo .] [--manifest ...]
  python scripts/review_freeze.py blind-check-score --packet-dir D --blind-check-dir D3 --filled B.csv
  python scripts/review_freeze.py second-look-score --packet-dir D --second-look-dir D2 --filled SL.csv
         --reviewer NAME --date YYYY-MM-DD --bootstrap-seed INT [--blind-check-dir D3]
         [--manifest-dir docs/evidence/reviews] [--repo .]

The draw seeds are not chosen by anyone: each is derived from the SHA-256 of the frozen
first manifest, so a draw cannot be repeated until it suits (red-team H1). Second-look and
blinding-check ids are salted with a random value kept only under that folder's sealed/,
so they cannot be traced back to first ids or runs from the public seeds (red-team M2).
`second-look-score` rebuilds the whole second-look folder from the frozen record and the
recorded salt and refuses any file that differs, rechecks the freeze commit, and rechecks
the blinding check from its own map and the kept answers.

Pre-registration section 8 ("Review record and freeze", "Second look on review
verdicts", owner decision D-D kept exactly as written). The packet folder is the one
scripts/review_packet.py writes. `validate` checks a filled RB-7 answers file against
the packet; `freeze` also rebuilds the packet from the run logs byte for byte, splits the
answers into verdicts.csv and fm_audit.csv, writes sittings.csv and the public manifest,
and prints the commit trailer. It never runs git and never commits: the owner commits
and pushes the manifest. At least 48 hours later `second-look-draw` checks the freeze
commit (through git_commit_info, which runs only `git show`) and writes a second-look
packet with fresh unit ids; `second-look-score` reports intra-rater agreement only.
`validate` and `freeze` refuse a packet whose recorded whole-narrative scope is not hard
or all (D-J, option b): a routed-scope packet is never frozen.

This script never sets, suggests or edits a verdict, never computes a review-settled,
audit-adjusted, route-blind or code-only figure, and reports no score. Its own source
imports nothing from reliability, regression or scoring.

Exit codes: 0 done; 3 refused (every gap found is listed, not only the first).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import secrets
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import settlement

EXIT_OK, EXIT_REFUSED = 0, 3
SCRIPTS_DIR = Path(__file__).resolve().parent
REVIEW_PACKET_SCRIPT = SCRIPTS_DIR / "review_packet.py"
PILOT_REPORT_SCRIPT = SCRIPTS_DIR / "pilot_report.py"

MASKED_COLUMNS: tuple[str, ...] = ("run_label", "log_eval_id", "sample_id", "epoch", "scorers")
SITTINGS_COLUMNS: tuple[str, ...] = ("start", "end")
SECOND_LOOK_MAP_FILE = "sealed/second-look-map.csv"
SECOND_LOOK_META_FILE = "sealed/second-look-meta.json"
SECOND_LOOK_RESULT_FILE = "sealed/second-look-result.json"
SECOND_LOOK_VERDICTS_FILE = "verdicts.csv"
BLIND_CHECK_CSV = "reviewer/blind-check.csv"
BLIND_CHECK_HTML = "reviewer/blind-check.html"
BLIND_CHECK_MAP = "sealed/blind-check-map.csv"
BLIND_CHECK_META = "sealed/blind-check-meta.json"
BLIND_CHECK_RESULT = "sealed/blind-check-result.json"
BLIND_CHECK_ANSWERS = "sealed/blind-check-answers.csv"
# A calendar date is "local": a day runs from UTC-12 to UTC+14 somewhere on earth. Date
# bounds below allow that spread and nothing more.
EARLIEST_OFFSET, LATEST_OFFSET = timedelta(hours=12), timedelta(hours=14)
BLIND_CHECK_COLUMNS: tuple[str, ...] = ("check_id", "narrative", "guess")
BLIND_CHECK_MAP_COLUMNS: tuple[str, ...] = ("check_id", "run_id", "run_label", "display_name")
DEFAULT_MANIFEST_DIR = "docs/evidence/reviews"

ROUTE_CLASSES: tuple[str, ...] = ("narrative_sweep", "code_review")
STRATUM_BOTH = "both"
FIRST_CORRECT, FIRST_NOT_CORRECT = "correct", "not_correct"
MIN_GAP = timedelta(hours=48)
SECOND_LOOK_FLOOR, SECOND_LOOK_SHARE, NOT_CORRECT_CAP = 100, Fraction(1, 10), 50
BOOTSTRAP_RESAMPLES = 10_000
GATE_MIN_PER_CLASS, GATE_KAPPA_LOWER = 10, 0.60
BLIND_CHECK_UNITS, BLIND_CHECK_CHANCE = 40, 0.25
FULL_RUN_WORKER_RUNS = 4
# D-J (option b, ruled 2026-11-10): a record is frozen only from a packet whose whole-narrative
# scope reads every narrative of a difficulty-bearing case. Scope routed stays a packet option
# for exploration, but its record is never frozen (trials.py refuses it for settlement too).
FREEZE_SCOPES: tuple[str, ...] = ("hard", "all")
AGREEMENT_LABEL = "intra-rater"
SECOND_LOOK_STATEMENT = ("The second look was made without access to the first verdicts and with the "
                         "worker model hidden. It reports agreement only.")

# Fixed key sets of what this script writes (test_no_score_reported pins them).
MANIFEST_KEYS: tuple[str, ...] = (
    "manifest_version", "kind", "run_id", "run_commit", "vocab_sha256", "settlement_table_sha256",
    "settlement_py_sha256", "tool_sha256", "logs", "private_files", "packet_seed", "whole_narrative_scope",
    "fm_epoch_routes", "counts", "verdict_distribution_by_route_class", "narrative_records", "fm_audit",
    "reviewer", "freeze_date", "sittings", "review_hours", "expanded_csv_sha256", "statement",
)
RUN_RESULT_KEYS: tuple[str, ...] = (
    "N", "n", "strata", "first_class_counts", "second_class_counts", "sample_first_correct",
    "sample_first_not_correct", "run_first_not_correct", "kappa", "ci_lower", "ci_upper", "resamples",
    "degenerate_resamples", "correct_to_not_correct", "not_correct_to_correct", "disagreements",
    "disagreeing_first_unit_ids", "gate",
)
PUBLIC_RUN_RESULT_KEYS: tuple[str, ...] = tuple(k for k in RUN_RESULT_KEYS if k != "disagreeing_first_unit_ids")
SECOND_LOOK_RESULT_KEYS: tuple[str, ...] = ("agreement", "bootstrap_seed", "runs")
SECOND_LOOK_MANIFEST_KEYS: tuple[str, ...] = (
    "manifest_version", "kind", "run_id", "agreement", "first_manifest_sha256", "freeze_commit",
    "draw_seed", "bootstrap_seed", "private_files", "runs", "blinding", "reviewer", "date", "statement",
)
BLINDING_KEYS: tuple[str, ...] = ("n", "named_correctly", "share", "ci_lower", "ci_upper", "chance", "status")
GATE_KEYS: tuple[str, ...] = ("branch", "passed", "result")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
_BOM = b"\xef\xbb\xbf"


class Refused(Exception):
    """A refusal; carries every gap found."""

    def __init__(self, gaps: Sequence[str] | str):
        self.gaps = [gaps] if isinstance(gaps, str) else list(gaps)
        super().__init__("; ".join(self.gaps))


# --- Sibling scripts, loaded by path (the pilot_report.load_compare_runs pattern) ---------

def _load_script(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None and Path(getattr(existing, "__file__", "") or "").resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {path.name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses in the script resolve their module by name
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def load_review_packet() -> Any:
    """scripts/review_packet.py (B3): build_packet, write_packet, load_packet, subset_packet, load_runs."""
    if not REVIEW_PACKET_SCRIPT.is_file():
        raise Refused(f"{REVIEW_PACKET_SCRIPT.name} is not present")
    return _load_script("review_packet", REVIEW_PACKET_SCRIPT)


@lru_cache(maxsize=1)
def load_pilot_report() -> Any:
    """scripts/pilot_report.py, for clopper_pearson only."""
    return _load_script("pilot_report", PILOT_REPORT_SCRIPT)


def _utcnow() -> datetime:
    """The clock; tests monkeypatch it."""
    return datetime.now(timezone.utc)


def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def latest_local_date(t: datetime) -> date:
    """The latest calendar date anywhere on earth at instant t."""
    return (_utc(t) + LATEST_OFFSET).date()


def earliest_local_date(t: datetime) -> date:
    """The earliest calendar date anywhere on earth at instant t."""
    return (_utc(t) - EARLIEST_OFFSET).date()


def derived_seed(first_manifest_sha256: str, run_id: str, purpose: str) -> int:
    """A draw seed fixed by the frozen first manifest: nobody picks it, and the same
    frozen record always gives the same draw (red-team H1)."""
    digest = hashlib.sha256(f"{first_manifest_sha256}|{run_id}|{purpose}".encode()).hexdigest()
    return int(digest[:16], 16)


# --- Record --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    packet_dir: str
    meta: dict
    runs: dict
    members: tuple[dict, ...]
    units: dict[str, dict]            # unit_id -> packet.csv row
    rows: tuple[dict, ...]            # filled RB-7 rows
    filled_paths: tuple[str, ...] = ()


def _packet_path(packet_dir: str | Path, rel: str) -> Path:
    return Path(packet_dir) / rel


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refused(f"cannot read {path.name}: {type(e).__name__}: {e}") from e


def _lenient_rb7_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a filled answers file, read leniently so every other gap can still be listed;
    the byte rules (header, LF, no BOM) are checked apart, in validate_record item 1."""
    try:
        data = path.read_bytes()
    except OSError as e:
        raise Refused(f"cannot read the answers file {path.name}: {type(e).__name__}") from e
    if data.startswith(_BOM):
        data = data[len(_BOM):]
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        return []
    rows = []
    for cells in reader:
        if not any(c.strip() for c in cells):
            continue
        row = {col: "" for col in rs.RB7_COLUMNS}
        for name, cell in zip(header, cells):
            if name in row:
                row[name] = cell
        rows.append(row)
    return rows


def _load(packet_dir: str, filled: Sequence[str]) -> Record:
    d = Path(packet_dir)
    meta = _read_json(_packet_path(d, rs.PACKET_FILES["meta"]))
    runs = _read_json(_packet_path(d, rs.PACKET_FILES["runs"]))
    try:
        members = rs.read_csv(_packet_path(d, rs.PACKET_FILES["members"]), rs.MEMBER_COLUMNS)
        packet_rows = rs.read_csv(_packet_path(d, rs.PACKET_FILES["packet"]), rs.PACKET_COLUMNS)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the packet: {type(e).__name__}: {e}") from e
    units: dict[str, dict] = {}
    for row in packet_rows:
        if row["unit_id"] in units:
            raise Refused(f"packet.csv repeats unit {row['unit_id']}")
        units[row["unit_id"]] = row
    rows: list[dict] = []
    for f in filled:
        rows.extend(_lenient_rb7_rows(Path(f)))
    return Record(packet_dir=str(d), meta=meta, runs=runs, members=tuple(members), units=units,
                  rows=tuple(rows), filled_paths=tuple(str(f) for f in filled))


def load_record(packet_dir: str, filled: str) -> Record:
    """The packet (meta, runs, member map, units) and one filled RB-7 answers file."""
    return _load(packet_dir, [filled])


def load_frozen_record(packet_dir: str) -> Record:
    """The packet and its frozen verdicts.csv and fm_audit.csv."""
    d = Path(packet_dir)
    return _load(packet_dir, [str(d / rs.RECORD_FILES["verdicts"]), str(d / rs.RECORD_FILES["fm_audit"])])


def _json_list(cell: str) -> list:
    value = json.loads(cell) if cell else []
    if not isinstance(value, list):
        raise ValueError("not a JSON array")
    return value


def _join(cell: str) -> str:
    return rs.RB7_LIST_SEP.join(str(x) for x in _json_list(cell))


def _check_file_bytes(path: Path) -> list[str]:
    try:
        data = path.read_bytes()
    except OSError as e:
        return [f"item 1: cannot read {path.name}: {type(e).__name__}"]
    gaps = []
    if data.startswith(_BOM):
        gaps.append(f"item 1: {path.name} starts with a byte order mark")
    crs = data.count(b"\r")
    if crs:
        gaps.append(f"item 1: {path.name} holds {crs} carriage return byte(s); LF line ends only")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        gaps.append(f"item 1: {path.name} is not utf-8")
        text = data.decode("utf-8", errors="replace")
    first = text.lstrip("\ufeff").split("\n", 1)[0].rstrip("\r")
    if first != ",".join(rs.RB7_COLUMNS) or not text.lstrip("\ufeff").startswith(",".join(rs.RB7_COLUMNS) + "\n"):
        gaps.append(f"item 1: {path.name} header is not the RB-7 header byte for byte")
    try:
        parsed = list(csv.reader(io.StringIO(text.lstrip("\ufeff"), newline="")))
    except csv.Error as e:
        gaps.append(f"item 1: {path.name} is not readable CSV: {e}")
        parsed = []
    width = len(parsed[0]) if parsed else 0
    for n, cells in enumerate(parsed[1:], start=2):
        if any(c.strip() for c in cells) and len(cells) != width:
            gaps.append(f"item 1: {path.name} record {n} has {len(cells)} cells; the header has {width}")
    return gaps


def _sentences(unit: dict) -> list:
    try:
        return _json_list(unit.get("sentences_json", ""))
    except (ValueError, json.JSONDecodeError):
        return []


def _validate_finding(uid: str, unit: dict, row: dict) -> list[str]:
    gaps = []
    verdict = row["verdict"]
    try:
        reasons = [str(r) for r in _json_list(unit["reasons"])]
        unit_allowed = tuple(_json_list(unit["allowed_verdicts"]))
    except (ValueError, json.JSONDecodeError):
        return [f"item 3: finding {uid}: packet lists are not JSON arrays"]
    table_allowed = settlement.allowed_verdicts(unit["route_key"], reasons)
    if unit_allowed != tuple(table_allowed):
        gaps.append(f"item 3: finding {uid}: the packet's allowed verdicts differ from the settlement table's")
    if not verdict:
        gaps.append(f"item 3: finding {uid}: verdict blank")
    elif verdict not in unit_allowed or verdict not in table_allowed:
        gaps.append(f"item 3: finding {uid}: verdict {verdict!r} not allowed for this route key")
    if verdict in rs.FINDING_NOTE_REQUIRED and not row["note"].strip():
        gaps.append(f"item 3: finding {uid}: note required for {verdict}")
    for col in ("route_key", "route_class"):
        if row[col] != unit[col]:
            gaps.append(f"item 3: finding {uid}: {col} differs from the packet")
    if row["reason"] != rs.RB7_LIST_SEP.join(reasons):
        gaps.append(f"item 3: finding {uid}: reason differs from the packet")
    try:
        if row["sentence"] != _join(unit["routed_sentences"]):
            gaps.append(f"item 3: finding {uid}: sentence differs from the packet")
    except (ValueError, json.JSONDecodeError):
        gaps.append(f"item 3: finding {uid}: packet routed_sentences is not a JSON array")
    return gaps


def _validate_narrative(uid: str, unit: dict, row: dict) -> list[str]:
    gaps = []
    verdict = row["verdict"]
    if row["route_key"] or row["route_class"]:
        gaps.append(f"item 4: narrative {uid}: route_key and route_class must be blank")
    if verdict not in rs.NARRATIVE_VERDICTS:
        gaps.append(f"item 4: narrative {uid}: verdict {verdict!r} is not none_wrong or wrong"
                    if verdict else f"item 4: narrative {uid}: verdict blank")
        return gaps
    if verdict == "none_wrong":
        if row["sentence"] or row["reason"]:
            gaps.append(f"item 4: narrative {uid}: none_wrong takes a blank sentence and reason")
        return gaps
    n_sent = len(_sentences(unit))
    numbers = row["sentence"].split(rs.RECORD_LIST_SEP) if row["sentence"] else []
    types = row["reason"].split(rs.RECORD_LIST_SEP) if row["reason"] else []
    if not numbers:
        gaps.append(f"item 4: narrative {uid}: wrong needs at least one sentence number")
    if len(numbers) != len(types):
        gaps.append(f"item 4: narrative {uid}: {len(numbers)} sentence number(s) but {len(types)} error type(s)")
    seen: set[tuple[str, str]] = set()
    for num, etype in zip(numbers, types):
        if etype not in rs.ERROR_TYPES:
            gaps.append(f"item 4: narrative {uid}: error type {etype!r} is not one of ERROR_TYPES")
        if num == rs.WHOLE_NARRATIVE_SENTENCE:
            if etype != "missing_required_element":
                gaps.append(f"item 4: narrative {uid}: sentence 0 is allowed only with missing_required_element")
        elif not (num.isdigit() and num == str(int(num)) and 1 <= int(num) <= n_sent):
            gaps.append(f"item 4: narrative {uid}: sentence number {num!r} outside 1..{n_sent}")
        if (num, etype) in seen:
            gaps.append(f"item 4: narrative {uid}: sentence {num} repeated with the same error type")
        seen.add((num, etype))
    if not row["note"].strip():
        gaps.append(f"item 4: narrative {uid}: note required for wrong")
    return gaps


def _validate_audit(uid: str, unit: dict, row: dict) -> list[str]:
    gaps = []
    verdict = row["verdict"]
    if verdict not in rs.AUDIT_VERDICTS:
        gaps.append(f"item 5: audit {uid}: verdict {verdict!r} is not real_error or extractor_error"
                    if verdict else f"item 5: audit {uid}: verdict blank")
    if not row["note"].strip():
        gaps.append(f"item 5: audit {uid}: note required")
    if row["route_key"] != unit["fm_code"]:
        gaps.append(f"item 5: audit {uid}: route_key is not the unit's FM code")
    try:
        if row["reason"] != _join(unit["fm_reasons"]):
            gaps.append(f"item 5: audit {uid}: reason differs from the packet's audit evidence")
    except (ValueError, json.JSONDecodeError):
        gaps.append(f"item 5: audit {uid}: packet fm_reasons is not a JSON array")
    if row["route_class"] != unit.get("route_class", "") or row["sentence"]:
        gaps.append(f"item 5: audit {uid}: route_class and sentence must be as the packet gives them")
    return gaps


def _validate_common(uid: str, row: dict, reviewer: str | None, freeze: date | None,
                     not_before: date | None = None) -> list[str]:
    gaps = []
    who = row["reviewer"].strip()
    if not who:
        gaps.append(f"item 6: {uid}: reviewer blank")
    elif reviewer is not None and row["reviewer"] != reviewer:
        gaps.append(f"item 6: {uid}: reviewer is not {reviewer!r}")
    when = row["date"]
    if not when:
        gaps.append(f"item 6: {uid}: date blank")
    else:
        parsed = _parse_date(when)
        if parsed is None:
            gaps.append(f"item 6: {uid}: date {when!r} is not YYYY-MM-DD")
        elif freeze is not None and parsed > freeze:
            gaps.append(f"item 6: {uid}: dated {when}, after the freeze date {freeze.isoformat()}")
        elif not_before is not None and parsed < not_before:
            gaps.append(f"item 6: {uid}: dated {when}, before {not_before.isoformat()}")
    filled = [c for c in MASKED_COLUMNS if row[c]]
    if filled:
        gaps.append(f"item 6: {uid}: masked column(s) {', '.join(filled)} must be blank")
    return gaps


def _parse_date(text: str) -> date | None:
    if not _DATE_RE.match(text or ""):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _epoch_key(m: Mapping[str, str]) -> tuple[str, str, str]:
    return (m["run_id"], m["sample_id"], str(m["epoch"]))


def _member_list(m: Mapping[str, str], col: str) -> list:
    try:
        return _json_list(m.get(col, ""))
    except (ValueError, json.JSONDecodeError):
        return []


def _validate_members(rec: Record) -> list[str]:
    gaps = []
    kinds_of_unit = {uid: u["row_kind"] for uid, u in rec.units.items()}
    narrative_epochs = {_epoch_key(m) for m in rec.members if m["member_kind"] == "narrative"}
    fm_pairs: set[tuple[tuple[str, str, str], str]] = set()
    audited: set[tuple[tuple[str, str, str], str]] = set()
    used: Counter = Counter()
    for m in rec.members:
        kind, uid, ek = m["member_kind"], m["unit_id"], _epoch_key(m)
        if kind not in rs.MEMBER_KINDS:
            gaps.append(f"item 7: member map: unknown member kind {kind!r}")
            continue
        for code in _member_list(m, "fm_codes"):
            fm_pairs.add((ek, str(code)))
        if kind in ("finding", "narrative", "fm_audit"):
            if uid not in rec.units:
                gaps.append(f"item 7: member map: a {kind} member names unit {uid!r}, not in the packet")
            elif kinds_of_unit[uid] != kind:
                gaps.append(f"item 7: member map: a {kind} member names {kinds_of_unit[uid]} unit {uid}")
            else:
                used[uid] += 1
                if kind == "fm_audit":
                    audited.add((ek, rec.units[uid]["fm_code"]))
        elif uid:
            gaps.append(f"item 7: member map: an {kind} member carries a unit id")
        if kind != "narrative" and ek not in narrative_epochs:
            gaps.append(f"item 7: member map: an epoch carrying a {kind} member is in no narrative unit")
    for pair in sorted(fm_pairs - audited):
        gaps.append(f"item 7: member map: an FM-coded epoch has no audit unit for {pair[1]}")
    gaps.extend(_validate_narrative_classes(rec))
    for uid in rec.units:
        if not used[uid]:
            gaps.append(f"item 7: unit {uid} has no member in the member map")
    return gaps


def _validate_narrative_classes(rec: Record) -> list[str]:
    """Item 7: every narrative member and narrative unit says what it is (D-J, option b).
    An epoch is routed when it carries a finding or unsettleable member, fm_coded when it
    carries an FM code and no such route, narrative_only otherwise; a unit takes the first
    of routed, fm_coded, narrative_only that any of its member epochs has. A narrative-only
    unit therefore has no finding and no audit unit on any of its epochs."""
    gaps: list[str] = []
    routed_epochs = {_epoch_key(m) for m in rec.members if m["member_kind"] in ("finding", "unsettleable")}
    coded_epochs = {_epoch_key(m) for m in rec.members if _member_list(m, "fm_codes")}
    member_classes: dict[str, set[str]] = {}
    for m in rec.members:
        if m["member_kind"] != "narrative" or m["unit_id"] not in rec.units:
            continue
        ek = _epoch_key(m)
        want = rs.epoch_narrative_class(ek in routed_epochs, ek in coded_epochs)
        if m["route_class"] != want:
            gaps.append(f"item 7: member map: a narrative member of unit {m['unit_id']} has class "
                        f"{m['route_class']!r}, expected {want!r}")
        member_classes.setdefault(m["unit_id"], set()).add(want)
    for uid, unit in rec.units.items():
        if unit["row_kind"] != "narrative" or uid not in member_classes:
            continue
        want = rs.unit_narrative_class(member_classes[uid])
        if unit["route_class"] != want:
            gaps.append(f"item 7: narrative unit {uid} has class {unit['route_class']!r}, expected {want!r}")
    return gaps


def check_packet_scope(rec: Record) -> list[str]:
    """D-J (option b): the packet's recorded whole-narrative scope is hard or all. A packet built
    with --whole-narrative-scope routed leaves the narrative-only units of the difficulty-bearing
    cases unread, so its record is never frozen and never settled."""
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    scope = meta.get("scope")
    if scope in FREEZE_SCOPES:
        return []
    return [f"D-J: the packet's whole-narrative scope is {scope!r}; a record is frozen only from a packet "
            f"built with scope {' or '.join(FREEZE_SCOPES)} (rebuild it with --whole-narrative-scope hard)"]


def check_packet_hashes(rec: Record) -> list[str]:
    """Item 8: every file packet-meta.json lists exists and hashes to it, and the list
    covers every packet file but the meta file itself."""
    gaps = []
    files = rec.meta.get("files") if isinstance(rec.meta, dict) else None
    if not isinstance(files, dict):
        return ["item 8: packet-meta.json lists no files"]
    for key, rel in rs.PACKET_FILES.items():
        if key != "meta" and rel not in files:
            gaps.append(f"item 8: packet-meta.json does not list {rel}")
    for rel, digest in sorted(files.items()):
        path = _packet_path(rec.packet_dir, rel)
        if not path.is_file():
            gaps.append(f"item 8: {rel} is missing")
        elif rs.sha256_file(path) != digest:
            gaps.append(f"item 8: {rel} does not hash to packet-meta.json")
    return gaps


def validate_record(rec: Record, *, reviewer: str | None = None, freeze_date: str | None = None,
                    packet_checks: bool = True, not_before: date | None = None) -> list[str]:
    """Every gap of the record against section 8.2 items 1 to 8; [] when complete.
    packet_checks=False skips items 7 and 8 and the D-J scope check (the second-look packet
    has no audit units, and it is drawn from a first packet whose scope was checked at freeze).
    not_before refuses a row dated before that date (the second look's draw)."""
    gaps: list[str] = []
    freeze = None
    if freeze_date is not None:
        freeze = _parse_date(freeze_date)
        if freeze is None:
            gaps.append(f"freeze date {freeze_date!r} is not YYYY-MM-DD")
    for f in rec.filled_paths:
        gaps.extend(_check_file_bytes(Path(f)))
    seen: Counter = Counter()
    for row in rec.rows:
        uid = row["unit_id"]
        seen[uid] += 1
        if seen[uid] == 2:
            gaps.append(f"item 2: unit {uid!r} has more than one row")
        if seen[uid] > 1:
            continue
        unit = rec.units.get(uid)
        if unit is None:
            gaps.append(f"item 2: unit {uid!r} is not in the packet")
            continue
        if row["row_kind"] != unit["row_kind"]:
            gaps.append(f"item 2: unit {uid}: row_kind {row['row_kind']!r}, the packet says {unit['row_kind']!r}")
            continue
        kind = unit["row_kind"]
        if kind == "finding":
            gaps.extend(_validate_finding(uid, unit, row))
        elif kind == "narrative":
            gaps.extend(_validate_narrative(uid, unit, row))
        elif kind == "fm_audit":
            gaps.extend(_validate_audit(uid, unit, row))
        else:
            gaps.append(f"item 2: unit {uid}: unknown row kind {kind!r}")
            continue
        gaps.extend(_validate_common(uid, row, reviewer, freeze, not_before))
    for uid, unit in rec.units.items():
        if not seen[uid]:
            gaps.append(f"item 2: {unit['row_kind']} unit {uid} has no row")
    if packet_checks:
        gaps.extend(check_packet_scope(rec))
        gaps.extend(_validate_members(rec))
        gaps.extend(check_packet_hashes(rec))
    return gaps


# --- Freeze -------------------------------------------------------------------------------

def _run_ids_by_label(rec: Record) -> dict[str, str]:
    return {str(info.get("label")): rid for rid, info in rec.runs.items()}


def rederive_check(rec: Record, runs: Sequence[tuple[str, str]]) -> list[str]:
    """Each --run file hashes to sealed/runs.json under the same label, and review_packet
    rebuilds every packet file byte for byte from those logs with the recorded seed, scope
    and FM-epoch mode (packet.html and packet-meta.json included)."""
    gaps: list[str] = []
    by_label = _run_ids_by_label(rec)
    given = [label for label, _ in runs]
    for label in sorted({x for x in given if given.count(x) > 1}):
        gaps.append(f"rederive: run label {label!r} given more than once")
    for label, path in runs:
        rid = by_label.get(label)
        if rid is None:
            gaps.append(f"rederive: run label {label!r} is not in the packet")
            continue
        p = Path(path)
        if not p.is_file():
            gaps.append(f"rederive: run file for {label!r} is missing")
        elif rs.sha256_file(p) != rec.runs[rid].get("sha256"):
            gaps.append(f"rederive: run file for {label!r} does not hash to sealed/runs.json")
    for label in sorted(set(by_label) - set(given)):
        gaps.append(f"rederive: run label {label!r} of the packet was not given")
    if gaps:
        return gaps
    meta = rec.meta
    try:
        rp = load_review_packet()
        run_inputs = rp.load_runs([(label, path) for label, path in runs], int(meta["seed"]))
        packet = rp.build_packet(run_inputs, run_id=meta["run_id"], seed=int(meta["seed"]), scope=meta["scope"],
                                 fm_epoch_routes=meta["fm_epoch_routes"])
        with tempfile.TemporaryDirectory(prefix="review-rederive-") as tmp:
            rp.write_packet(packet, tmp)
            for rel in rs.PACKET_FILES.values():
                built, kept = Path(tmp) / rel, _packet_path(rec.packet_dir, rel)
                if not built.is_file():
                    gaps.append(f"rederive: the rebuild wrote no {rel}")
                elif not kept.is_file() or built.read_bytes() != kept.read_bytes():
                    gaps.append(f"rederive: {rel} differs from the rebuild")
    except Refused as e:
        gaps.extend(f"rederive: {g}" for g in e.gaps)
    except SystemExit as e:
        gaps.append(f"rederive: the rebuild refused (exit {e.code})")
    except Exception as e:  # any failure to rebuild is a gap, never a pass
        gaps.append(f"rederive: the rebuild failed: {type(e).__name__}: {e}")
    return gaps


def parse_sittings(pairs: Sequence[tuple[str, str]]) -> tuple[list[tuple[datetime, datetime]], list[str]]:
    gaps, parsed = [], []
    for start, end in pairs:
        try:
            s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
        except ValueError:
            gaps.append(f"sitting {start}/{end}: not ISO date-times")
            continue
        if s.tzinfo is None or e.tzinfo is None:
            gaps.append(f"sitting {start}/{end}: needs a UTC offset")
            continue
        parsed.append((s, e))
    return parsed, gaps


def check_sittings(pairs: Sequence[tuple[str, str]], freeze_date: str | None,
                   now: datetime | None = None) -> list[str]:
    """At least one sitting; each start before its end; none overlapping; none after the
    freeze date; none ending after `now` when it is given."""
    if not pairs:
        return ["sittings: at least one sitting is required"]
    parsed, gaps = parse_sittings(pairs)
    freeze = _parse_date(freeze_date) if freeze_date else None
    for s, e in parsed:
        if not s < e:
            gaps.append(f"sitting {s.isoformat()}/{e.isoformat()}: start is not before end")
        if freeze is not None and (s.date() > freeze or e.date() > freeze):
            gaps.append(f"sitting {s.isoformat()}/{e.isoformat()}: after the freeze date {freeze.isoformat()}")
        if now is not None and e > now:
            gaps.append(f"sitting {s.isoformat()}/{e.isoformat()}: ends after the time of this freeze")
    ordered = sorted(parsed, key=lambda p: p[0])
    for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
        if s2 < e1:
            gaps.append(f"sittings {s1.isoformat()}/{e1.isoformat()} and {s2.isoformat()}/{e2.isoformat()} overlap")
    return gaps


def _review_hours(pairs: Sequence[tuple[str, str]]) -> float:
    parsed, _ = parse_sittings(pairs)
    seconds = sum((e - s).total_seconds() for s, e in parsed)
    return round(seconds / 3600, 3)


def _row_kind_order(kind: str) -> int:
    return rs.ROW_KINDS.index(kind) if kind in rs.ROW_KINDS else len(rs.ROW_KINDS)


def record_rows(rec: Record, kinds: Sequence[str]) -> list[dict[str, str]]:
    """The answers rows of the given kinds, every RB-7 cell, in (row kind, unit id) order."""
    rows = [{c: row.get(c, "") for c in rs.RB7_COLUMNS} for row in rec.rows if row["row_kind"] in kinds]
    return sorted(rows, key=lambda r: (_row_kind_order(r["row_kind"]), r["unit_id"]))


def _verdict_of(rec: Record) -> dict[str, str]:
    return {row["unit_id"]: row["verdict"] for row in rec.rows}


def _unsettleable_row_id(m: Mapping[str, str]) -> str:
    reasons = [str(r) for r in _member_list(m, "reasons")]
    try:
        rows = settlement.rows_for(m["route_key"], reasons)
    except ValueError:
        return settlement.CATCH_ALL_ROW_ID
    for row in rows:
        if not row.settleable:
            return row.row_id
    return rows[0].row_id if rows else settlement.CATCH_ALL_ROW_ID


def _counter_dict(counter: Mapping[str, int], keys: Sequence[str] = ()) -> dict[str, int]:
    out = {k: 0 for k in keys}
    for k, v in counter.items():
        out[k] = out.get(k, 0) + v
    return {k: out[k] for k in sorted(out)}


def build_manifest(rec: Record, *, run_id: str, run_commit: str, reviewer: str,
                   sittings: Sequence[tuple[str, str]], freeze_date: str = "",
                   expanded_csv_sha256: str | None = None) -> dict:
    """The public manifest: hashes, counts and aggregate verdict counts only; no sentence,
    no narrative, no unit id, no per-unit verdict, no route key."""
    meta, verdicts = rec.meta, _verdict_of(rec)
    finding_units = [u for u in rec.units.values() if u["row_kind"] == "finding"]
    finding_members = [m for m in rec.members if m["member_kind"] == "finding"]
    distribution: dict[str, Counter] = {cls: Counter() for cls in ROUTE_CLASSES}
    for m in finding_members:
        distribution.setdefault(m["route_class"], Counter())[verdicts.get(m["unit_id"], "")] += 1
    narrative_rows = [r for r in rec.rows if r["row_kind"] == "narrative"]
    audit_rows = [r for r in rec.rows if r["row_kind"] == "fm_audit"]
    d = Path(rec.packet_dir)
    private = {rel: rs.sha256_file(d / rel) for rel in rs.PACKET_FILES.values()}
    private.update({rel: rs.sha256_file(d / rel) for rel in rs.RECORD_FILES.values()})
    logs = sorted(
        ({"file_name": info.get("file_name", ""), "sha256": info.get("sha256", ""), "eval_id": info.get("eval_id", ""),
          "log_revision_commit": info.get("revision_commit", "")} for info in rec.runs.values()),
        key=lambda x: (x["file_name"], x["sha256"]))
    manifest = {
        "manifest_version": "1",
        "kind": "review_freeze",
        "run_id": run_id,
        "run_commit": run_commit,
        "vocab_sha256": meta.get("vocab_sha256", ""),
        "settlement_table_sha256": meta.get("settlement_table_sha256", ""),
        "settlement_py_sha256": meta.get("settlement_py_sha256", ""),
        "tool_sha256": {
            "review_schema.py": meta.get("review_schema_py_sha256", ""),
            "review_packet.py": meta.get("review_packet_py_sha256", ""),
            "review_facts.py": meta.get("review_facts_py_sha256", ""),
            "review_freeze.py": rs.sha256_file(Path(__file__)),
        },
        "logs": logs,
        "private_files": private,
        "packet_seed": meta.get("seed"),
        "whole_narrative_scope": meta.get("scope", ""),
        "fm_epoch_routes": meta.get("fm_epoch_routes", ""),
        "counts": {
            "finding_units_by_route_class": _counter_dict(Counter(u["route_class"] for u in finding_units),
                                                          ROUTE_CLASSES),
            "findings_by_route_class": _counter_dict(Counter(m["route_class"] for m in finding_members),
                                                     ROUTE_CLASSES),
            "narrative_units": sum(1 for u in rec.units.values() if u["row_kind"] == "narrative"),
            "narrative_members": sum(1 for m in rec.members if m["member_kind"] == "narrative"),
            "narrative_units_by_class": _counter_dict(
                Counter(u["route_class"] for u in rec.units.values() if u["row_kind"] == "narrative"),
                rs.NARRATIVE_CLASSES),
            "narrative_members_by_class": _counter_dict(
                Counter(m["route_class"] for m in rec.members if m["member_kind"] == "narrative"),
                rs.NARRATIVE_CLASSES),
            "fm_audit_rows": sum(1 for u in rec.units.values() if u["row_kind"] == "fm_audit"),
            "unsettleable_by_row": _counter_dict(Counter(
                _unsettleable_row_id(m) for m in rec.members if m["member_kind"] == "unsettleable")),
        },
        "verdict_distribution_by_route_class": {
            cls: _counter_dict(c, rs.FINDING_VERDICTS) for cls, c in sorted(distribution.items())},
        "narrative_records": _counter_dict(Counter(r["verdict"] for r in narrative_rows), rs.NARRATIVE_VERDICTS),
        "fm_audit": _counter_dict(Counter(r["verdict"] for r in audit_rows), rs.AUDIT_VERDICTS),
        "reviewer": reviewer,
        "freeze_date": freeze_date,
        "sittings": [{"start": s, "end": e} for s, e in sittings],
        "review_hours": _review_hours(sittings),
        "expanded_csv_sha256": expanded_csv_sha256,
        "statement": rs.STATEMENT,
    }
    assert tuple(manifest) == MANIFEST_KEYS
    return manifest


def expanded_rows(rec: Record) -> list[dict[str, str]]:
    """One RB-7 row per finding, narrative and audit member, with run label, eval id,
    sample, epoch and scorers filled from the member map; the unit's verdict row otherwise."""
    by_unit = {row["unit_id"]: row for row in rec.rows}
    out = []
    for m in rec.members:
        if m["member_kind"] not in rs.ROW_KINDS or m["unit_id"] not in by_unit:
            continue
        row = {c: by_unit[m["unit_id"]].get(c, "") for c in rs.RB7_COLUMNS}
        info = rec.runs.get(m["run_id"], {})
        row.update({"run_label": str(info.get("label", "")), "log_eval_id": str(info.get("eval_id", "")),
                    "sample_id": m["sample_id"], "epoch": str(m["epoch"]),
                    "scorers": ";".join(sorted(str(s) for s in _member_list(m, "scorers")))})
        out.append(row)

    def order(r: dict[str, str]) -> tuple:
        ep = int(r["epoch"]) if r["epoch"].isdigit() else 0
        return (r["run_label"], r["sample_id"], ep, _row_kind_order(r["row_kind"]), r["route_key"], r["unit_id"])

    return sorted(out, key=order)


def trailer(reviewer: str, date: str) -> str:
    """The last line of the freeze commit message."""
    return f"Reviewed-by: {reviewer}, {date}"


def _write_json(path: Path, payload: Any) -> None:
    rs.write_lf(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def freeze(packet_dir: str, filled: str, runs: Sequence[tuple[str, str]], *, reviewer: str, freeze_date: str,
           run_commit: str, sittings: Sequence[tuple[str, str]], manifest_dir: str = DEFAULT_MANIFEST_DIR,
           expanded_csv: str | None = None) -> tuple[Path, str]:
    """Check everything, then write the record files, the manifest and (optionally) the
    expanded CSV. Nothing is written when any gap is found. Returns (manifest path, trailer)."""
    gaps: list[str] = []
    now = _utcnow()
    if not reviewer.strip():
        gaps.append("--reviewer is blank")
    parsed_date = _parse_date(freeze_date)
    if parsed_date is None:
        gaps.append(f"--date {freeze_date!r} is not YYYY-MM-DD")
    elif parsed_date > latest_local_date(now):
        gaps.append(f"--date {freeze_date} is after today")
    if not _SHA_RE.match(run_commit or ""):
        gaps.append(f"--run-commit {run_commit!r} is not a hex commit hash")
    rec = load_record(packet_dir, filled)
    gaps.extend(validate_record(rec, reviewer=reviewer, freeze_date=freeze_date))
    gaps.extend(check_sittings(sittings, freeze_date, now))
    d = Path(packet_dir)
    for rel in rs.RECORD_FILES.values():
        if (d / rel).exists():
            gaps.append(f"{rel} already exists in the packet folder; a frozen record is never overwritten")
    run_id = str(rec.meta.get("run_id", ""))
    mdir = Path(manifest_dir)
    manifest_path = mdir / f"{run_id}-manifest.json"
    if _inside(mdir, d):
        gaps.append("--manifest-dir is inside the packet folder")
    if manifest_path.exists():
        gaps.append(f"{manifest_path.name} already exists; a manifest is never overwritten")
    if expanded_csv is not None and Path(expanded_csv).exists():
        gaps.append("--expanded-csv path already exists")
    gaps.extend(rederive_check(rec, runs))
    if gaps:
        raise Refused(gaps)
    rs.write_rb7_csv(d / rs.RECORD_FILES["verdicts"], record_rows(rec, ("finding", "narrative")))
    rs.write_rb7_csv(d / rs.RECORD_FILES["fm_audit"], record_rows(rec, ("fm_audit",)))
    rs.write_csv(d / rs.RECORD_FILES["sittings"], SITTINGS_COLUMNS,
                 [{"start": s, "end": e} for s, e in sittings])
    expanded_sha = None
    if expanded_csv is not None:  # written and hashed before the manifest, so the manifest binds it
        rs.write_rb7_csv(expanded_csv, expanded_rows(rec))
        expanded_sha = rs.sha256_file(expanded_csv)
    manifest = build_manifest(rec, run_id=run_id, run_commit=run_commit, reviewer=reviewer, sittings=sittings,
                              freeze_date=freeze_date, expanded_csv_sha256=expanded_sha)
    mdir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest_path, trailer(reviewer, freeze_date)


# --- Freeze commit checks -------------------------------------------------------------------

def git_commit_info(repo: str, sha: str, path: str) -> tuple[str, str, bytes | None]:
    """(committer ISO time, message, the file's bytes at that commit or None). The only git
    reader in this script; it runs `git show` and nothing else."""
    head = subprocess.run(["git", "-C", repo, "show", "-s", "--format=%cI%n%B", sha], capture_output=True,
                          check=False, timeout=60)
    if head.returncode != 0:
        raise Refused(f"git show could not read commit {sha}")
    text = head.stdout.decode("utf-8", errors="replace")
    committer, _, message = text.partition("\n")
    blob = subprocess.run(["git", "-C", repo, "show", f"{sha}:{path}"], capture_output=True, check=False,
                          timeout=60)
    return committer.strip(), message, blob.stdout if blob.returncode == 0 else None


def check_freeze_commit(manifest_bytes: bytes, manifest: Mapping[str, Any], *, repo: str, sha: str, rel_path: str,
                        now: datetime) -> tuple[list[str], datetime | None]:
    """The freeze commit is at least 48 hours old, its message ends with the trailer for the
    manifest's reviewer and date, and it holds the manifest byte for byte."""
    gaps: list[str] = []
    try:
        committer, message, blob = git_commit_info(repo, sha, rel_path)
    except Refused as e:
        return list(e.gaps), None
    try:
        freeze_time = datetime.fromisoformat(committer)
        if freeze_time.tzinfo is None:
            freeze_time = freeze_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return [f"freeze commit {sha}: committer time {committer!r} is not ISO"], None
    if now - freeze_time < MIN_GAP:
        gaps.append(f"freeze commit {sha} is less than 48 hours old")
    stated = _parse_date(str(manifest.get("freeze_date", "")))
    if stated is not None and stated > latest_local_date(freeze_time):
        gaps.append(f"freeze commit {sha}: the manifest's freeze date {stated.isoformat()} is after the commit")
    lines = message.rstrip("\n").split("\n")
    want = trailer(str(manifest.get("reviewer", "")), str(manifest.get("freeze_date", "")))
    if not lines or lines[-1].rstrip("\r") != want:
        gaps.append(f"freeze commit {sha}: the message's last line is not {want!r}")
    if blob is None:
        gaps.append(f"freeze commit {sha} does not hold {rel_path}")
    elif blob != manifest_bytes:
        gaps.append(f"freeze commit {sha}: {rel_path} differs from the manifest file")
    return gaps, freeze_time


def _default_manifest(repo: str, run_id: str) -> Path:
    return Path(repo) / DEFAULT_MANIFEST_DIR / f"{run_id}-manifest.json"


def check_private_files(packet_dir: str, manifest: Mapping[str, Any]) -> list[str]:
    gaps = []
    files = manifest.get("private_files")
    if not isinstance(files, dict) or not files:
        return ["the manifest lists no private files"]
    for rel, digest in sorted(files.items()):
        p = Path(packet_dir) / rel
        if not p.is_file():
            gaps.append(f"private file {rel} is missing")
        elif rs.sha256_file(p) != digest:
            gaps.append(f"private file {rel} does not hash to the manifest")
    return gaps


def check_record_files(packet_dir: str, *, repo: str, manifest: str | None) -> tuple[dict, bytes, Path, list[str]]:
    """The packet's freeze manifest and every gap between the frozen record's files and it:
    the manifest is this packet's review_freeze manifest, it lists every packet file and
    every record file, and each file it lists exists and hashes to it. Returns (manifest,
    its bytes, its path, gaps); refuses only when the manifest cannot be found or read.
    Shared by check_frozen and scripts/pilot_report.py --review-record."""
    meta = _read_json(Path(packet_dir) / rs.PACKET_FILES["meta"])
    run_id = str(meta.get("run_id", "")) if isinstance(meta, dict) else ""
    mpath = Path(manifest) if manifest else _default_manifest(repo, run_id)
    if not mpath.is_file():
        raise Refused(f"manifest {mpath.name} not found")
    mbytes = mpath.read_bytes()
    man = _read_json(mpath)
    if not isinstance(man, dict):
        raise Refused(f"manifest {mpath.name} is not a JSON object")
    gaps: list[str] = []
    if man.get("kind") != "review_freeze" or man.get("run_id") != run_id:
        gaps.append("the manifest is not this packet's freeze manifest")
    listed = man.get("private_files")
    if isinstance(listed, dict) and listed:
        for rel in (*rs.PACKET_FILES.values(), *rs.RECORD_FILES.values()):
            if rel not in listed:
                gaps.append(f"the manifest does not list private file {rel}")
    gaps.extend(check_private_files(packet_dir, man))
    return man, mbytes, mpath, gaps


def check_frozen(packet_dir: str, *, freeze_commit: str, repo: str, manifest: str | None,
                 now: datetime) -> tuple[Record, dict, datetime, Path]:
    """Shared by second-look-draw and blind-check-draw. Refuses with every gap."""
    man, mbytes, mpath, gaps = check_record_files(packet_dir, repo=repo, manifest=manifest)
    if not _SHA_RE.match(freeze_commit or ""):
        gaps.append(f"--freeze-commit {freeze_commit!r} is not a hex commit hash")
        freeze_time = None
    elif not _inside(mpath, Path(repo)):
        gaps.append("the manifest is not inside --repo")
        freeze_time = None
    else:
        rel = mpath.resolve().relative_to(Path(repo).resolve()).as_posix()
        more, freeze_time = check_freeze_commit(mbytes, man, repo=repo, sha=freeze_commit, rel_path=rel, now=now)
        gaps.extend(more)
    if gaps:
        raise Refused(gaps)
    rec = load_frozen_record(packet_dir)
    return rec, man, freeze_time, mpath  # type: ignore[return-value]


def _out_dir_free(path: Path, packet_dir: str | Path | None = None) -> list[str]:
    gaps = []
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        gaps.append(f"output folder {path} exists and is not empty")
    if packet_dir is not None and _inside(path, Path(packet_dir)):
        gaps.append(f"output folder {path} is inside the packet folder, beside the first verdicts")
    return gaps


def _default_out(packet_dir: str, suffix: str) -> Path:
    """A sibling of the packet folder, never inside it (red-team L3)."""
    p = Path(packet_dir)
    return p.parent / f"{p.name}-{suffix}"


# --- Second look ----------------------------------------------------------------------------

def worker_run_ids(rec: Record) -> list[str]:
    """Worker runs: every run but a template baseline."""
    return sorted(rid for rid, info in rec.runs.items() if not info.get("template_baseline"))


def second_look_units(rec: Record) -> dict[str, list[dict]]:
    """Per worker run: one unit per narrative unit with at least one routed member epoch in
    that run (an epoch carrying a finding or unsettleable member), with every finding unit
    holding a member on those epochs, its first class and its route stratum.

    This is the D-D second-look population (pre-registration section 8, "a second-look unit is
    one distinct routed narrative"). A narrative-only unit (an unrouted, uncoded narrative of
    a difficulty-bearing case, D-J) is never a unit here: it has no routed member epoch, so it
    is neither counted in N nor sampled. An FM-coded narrative with no route is not a unit
    either; with the D-H default its routes are findings, so an FM-coded epoch that routes is."""
    verdicts = _verdict_of(rec)
    out: dict[str, list[dict]] = {}
    for run in worker_run_ids(rec):
        run_members = [m for m in rec.members if m["run_id"] == run]
        narrative_of = {_epoch_key(m): m["unit_id"] for m in run_members if m["member_kind"] == "narrative"}
        routed = [m for m in run_members if m["member_kind"] in ("finding", "unsettleable")]
        by_unit: dict[str, dict[str, set]] = {}
        for m in routed:
            nu = narrative_of.get(_epoch_key(m))
            if nu is None:
                raise Refused(f"a routed epoch of run {run} is in no narrative unit")
            if rec.units[nu].get("route_class") != rs.NARRATIVE_CLASS_ROUTED:
                raise Refused(f"narrative unit {nu} holds a routed epoch of run {run} but is classed "
                              f"{rec.units[nu].get('route_class')!r}, not {rs.NARRATIVE_CLASS_ROUTED!r}")
            slot = by_unit.setdefault(nu, {"findings": set(), "classes": set(), "epochs": set()})
            slot["epochs"].add(_epoch_key(m))
            slot["classes"].add(m["route_class"])
            if m["member_kind"] == "finding":
                slot["findings"].add(m["unit_id"])
        units = []
        for nu in sorted(by_unit):
            slot = by_unit[nu]
            classes = slot["classes"]
            if not classes <= set(ROUTE_CLASSES):
                raise Refused(f"unknown route class in run {run}: {sorted(classes - set(ROUTE_CLASSES))}")
            stratum = STRATUM_BOTH if len(classes) == 2 else next(iter(classes))
            findings = sorted(slot["findings"])
            correct = verdicts.get(nu) == "none_wrong" and all(verdicts.get(f) == "correct" for f in findings)
            units.append({"first_unit_id": nu, "run_id": run, "finding_unit_ids": findings,
                          "stratum": stratum, "first_class": FIRST_CORRECT if correct else FIRST_NOT_CORRECT})
        out[run] = units
    return out


def blind_check_pool(rec: Record) -> list[tuple[str, str]]:
    """The blinding check population: (narrative unit id, worker run) for every narrative the
    reviewer read, routed, FM-coded and narrative-only (D-J hard-case reads) alike (owner
    decision D-R6, 2026-11-10). A unit counts only when it occurs in exactly one worker run,
    since a narrative several workers wrote has no one model to name. Unlike the second look,
    the pool is not limited to routed narratives."""
    worker_set = set(worker_run_ids(rec))
    runs_of_unit: dict[str, set[str]] = {}
    for m in rec.members:
        if m["member_kind"] == "narrative" and m["run_id"] in worker_set:
            runs_of_unit.setdefault(m["unit_id"], set()).add(m["run_id"])
    narrative_units = {uid for uid, u in rec.units.items() if u.get("row_kind") == "narrative"}
    missing = sorted(set(runs_of_unit) - narrative_units)
    if missing:
        raise Refused(f"member map names narrative unit(s) not in the packet: {missing[:5]}")
    return sorted((uid, next(iter(runs))) for uid, runs in runs_of_unit.items() if len(runs) == 1)


def sample_size(n_units: int) -> int:
    """n = min(N, max(100, ceil(0.10 * N)))."""
    return min(n_units, max(SECOND_LOOK_FLOOR, math.ceil(SECOND_LOOK_SHARE * n_units)))


def _quotas(sizes: Mapping[str, int], slots: int) -> dict[str, int]:
    """Largest-remainder allocation of slots over strata in proportion to size (ties by name)."""
    total = sum(sizes.values())
    if slots >= total:
        return dict(sizes)
    exact = {k: Fraction(slots * v, total) for k, v in sizes.items()}
    quotas = {k: math.floor(x) for k, x in exact.items()}
    left = slots - sum(quotas.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - quotas[k]), k))[:left]:
        quotas[k] += 1
    return quotas


def _stratified(units: Sequence[dict], slots: int, *, seed: int, run_id: str, cls: str) -> tuple[list[str], dict]:
    strata: dict[str, list[str]] = {}
    for u in units:
        strata.setdefault(u["stratum"], []).append(u["first_unit_id"])
    quotas = _quotas({k: len(v) for k, v in strata.items()}, max(slots, 0))
    drawn: list[str] = []
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        q = quotas[stratum]
        drawn.extend(ids if q >= len(ids) else random.Random(f"{seed}|{run_id}|{stratum}|{cls}").sample(ids, q))
    return sorted(drawn), {k: quotas[k] for k in sorted(quotas)}


def _draw_details(units_by_run: Mapping[str, Sequence[dict]], seed: int) -> dict[str, dict]:
    details = {}
    for run, units in sorted(units_by_run.items()):
        not_correct = [u for u in units if u["first_class"] == FIRST_NOT_CORRECT]
        correct = [u for u in units if u["first_class"] == FIRST_CORRECT]
        n = sample_size(len(units))
        if len(not_correct) <= NOT_CORRECT_CAP:
            nc_ids = sorted(u["first_unit_id"] for u in not_correct)
            nc_strata = dict(sorted(Counter(u["stratum"] for u in not_correct).items()))
        else:
            nc_ids, nc_strata = _stratified(not_correct, NOT_CORRECT_CAP, seed=seed, run_id=run, cls=FIRST_NOT_CORRECT)
        c_ids, c_strata = _stratified(correct, n - len(nc_ids), seed=seed, run_id=run, cls=FIRST_CORRECT)
        details[run] = {"N": len(units), "n": n, "not_correct_total": len(not_correct), "correct_total": len(correct),
                        "drawn": sorted(nc_ids + c_ids),
                        "strata": {FIRST_NOT_CORRECT: nc_strata, FIRST_CORRECT: c_strata}}
    return details


def draw_second_look(rec: Record, *, seed: int, freeze_time: datetime, now: datetime) -> dict[str, list[str]]:
    """Per worker run, the drawn first narrative unit ids (D-D). Refuses within 48 hours of the freeze."""
    if now - freeze_time < MIN_GAP:
        raise Refused("the second look is drawn at least 48 hours after the freeze")
    return {run: d["drawn"] for run, d in _draw_details(second_look_units(rec), seed).items()}


def write_second_look(rec: Record, packet_dir: str, out: Path, *, seed: int, salt: str) -> tuple[dict, list[str]]:
    """Write the second-look packet (fresh ids salted with `salt`) and its map under out.
    Deterministic given the frozen record, seed and salt, so second-look-score can rebuild
    it and compare byte for byte. Returns (draw details per run, drawn first unit ids)."""
    units_by_run = second_look_units(rec)
    details = _draw_details(units_by_run, seed)
    drawn_ids = sorted({uid for d in details.values() for uid in d["drawn"]})
    rp = load_review_packet()
    packet = rp.load_packet(packet_dir)
    subset, new_to_old = rp.subset_packet(packet, drawn_ids, seed=seed, id_salt=salt)
    rp.write_packet(subset, str(out), mode="second_look")
    old_to_new = {old: new for new, old in new_to_old.items()}
    map_rows = []
    by_first = {(u["first_unit_id"], u["run_id"]): u for units in units_by_run.values() for u in units}
    for run, d in details.items():
        for uid in d["drawn"]:
            u = by_first[(uid, run)]
            map_rows.append({"sl_unit_id": old_to_new[uid], "first_unit_id": uid, "row_kind": "narrative",
                             "run_id": run, "stratum": u["stratum"], "first_class": u["first_class"]})
    for new, old in sorted(new_to_old.items()):
        kind = rec.units.get(old, {}).get("row_kind", "")
        if kind == "finding":
            map_rows.append({"sl_unit_id": new, "first_unit_id": old, "row_kind": "finding", "run_id": "",
                             "stratum": "", "first_class": ""})
    map_rows.sort(key=lambda r: (_row_kind_order(r["row_kind"]), r["sl_unit_id"], r["run_id"]))
    (out / "sealed").mkdir(parents=True, exist_ok=True)
    rs.write_csv(out / SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS, map_rows)
    return details, drawn_ids


def _files_of(folder: Path, skip: Sequence[str] = ()) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): rs.sha256_file(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.relative_to(folder).as_posix() not in skip}


def _draw_runs(details: Mapping[str, dict]) -> dict:
    return {run: {k: v for k, v in d.items() if k != "drawn"} for run, d in details.items()}


def _second_look_draw(args: argparse.Namespace) -> int:
    now = _utcnow()
    out = Path(args.out) if args.out else _default_out(args.packet_dir, "second-look")
    gaps = _out_dir_free(out, args.packet_dir)
    try:
        rec, _man, freeze_time, mpath = check_frozen(args.packet_dir, freeze_commit=args.freeze_commit,
                                                     repo=args.repo, manifest=args.manifest, now=now)
    except Refused as e:
        raise Refused(gaps + e.gaps) from e
    if gaps:
        raise Refused(gaps)
    second_look_units(rec)  # refuses a malformed record before anything is written
    if now - freeze_time < MIN_GAP:
        raise Refused("the second look is drawn at least 48 hours after the freeze")
    first_sha = rs.sha256_file(mpath)
    seed = derived_seed(first_sha, str(rec.meta.get("run_id", "")), "second_look")
    salt = secrets.token_hex(16)
    details, drawn_ids = write_second_look(rec, args.packet_dir, out, seed=seed, salt=salt)
    meta = {"seed": seed, "id_salt": salt, "freeze_commit": args.freeze_commit,
            "freeze_time": freeze_time.isoformat(), "draw_time": now.isoformat(), "first_manifest_sha256": first_sha,
            "runs": _draw_runs(details), "files": _files_of(out)}
    _write_json(out / SECOND_LOOK_META_FILE, meta)
    # No strata, classes or per-run figures here: the reviewer runs this and must not see them.
    print(f"second look drawn: {len(drawn_ids)} narrative unit(s); open {out / rs.PACKET_FILES['html']}")
    return EXIT_OK


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float | None:
    """Cohen's kappa on two boolean ratings; None when p_e == 1 (or no units)."""
    n = len(first)
    if n != len(second):
        raise ValueError("rating lists differ in length")
    if n == 0:
        return None
    agree = sum(1 for a, b in zip(first, second) if a == b)
    pa, pb = Fraction(sum(map(bool, first)), n), Fraction(sum(map(bool, second)), n)
    p_o, p_e = Fraction(agree, n), pa * pb + (1 - pa) * (1 - pb)
    if p_e == 1:
        return None
    return float((p_o - p_e) / (1 - p_e))


def bootstrap_kappa(first: Sequence[bool], second: Sequence[bool], *, seed: int,
                    resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Percentile bootstrap CI of kappa over units resampled with replacement:
    lower = sorted[floor(0.025 B)], upper = sorted[ceil(0.975 B) - 1]. A resample with
    p_e = 1 takes kappa 0.0 and is counted."""
    pairs = list(zip(map(bool, first), map(bool, second)))
    point = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])
    if not pairs:
        return {"kappa": None, "ci_lower": None, "ci_upper": None, "resamples": resamples,
                "degenerate_resamples": 0, "seed": seed}
    rng = random.Random(seed)
    values, degenerate = [], 0
    for _ in range(resamples):
        draw = rng.choices(pairs, k=len(pairs))
        k = cohen_kappa([a for a, _ in draw], [b for _, b in draw])
        if k is None:
            degenerate += 1
            k = 0.0
        values.append(k)
    values.sort()
    lower = values[math.floor(0.025 * resamples)]
    upper = values[math.ceil(0.975 * resamples) - 1]
    if point is None:
        # kappa is undefined on the sample itself (one class only), so no interval is
        # reported: a CI of [0.0, 0.0] would read as a measured bound (demo defect 4)
        lower = upper = None
    return {"kappa": point, "ci_lower": lower, "ci_upper": upper, "resamples": resamples,
            "degenerate_resamples": degenerate, "seed": seed}


def gate(result_for_run: Mapping[str, Any]) -> dict:
    """D-D gate. kappa: both first classes hold at least 10 sampled units and the kappa
    lower limit is at least 0.60. no_overturn: the run holds fewer than 10 first
    not_correct units and no unit first ruled correct is ruled otherwise. Else
    too_few_per_class, failed."""
    if (result_for_run["sample_first_correct"] >= GATE_MIN_PER_CLASS
            and result_for_run["sample_first_not_correct"] >= GATE_MIN_PER_CLASS):
        lower = result_for_run.get("ci_lower")
        branch, passed = "kappa", lower is not None and lower >= GATE_KAPPA_LOWER
    elif result_for_run["run_first_not_correct"] < GATE_MIN_PER_CLASS:
        branch, passed = "no_overturn", result_for_run["correct_to_not_correct"] == 0
    else:
        branch, passed = "too_few_per_class", False
    return {"branch": branch, "passed": passed, "result": "confirmed" if passed else "review not confirmed"}


def second_look_result(rec: Record, sl_rows: Sequence[dict], sl_map: Sequence[dict], *, bootstrap_seed: int) -> dict:
    """Per worker run: first and second class per drawn unit, kappa with its bootstrap CI,
    flips both ways, the disagreement count and ids (private), and the gate. Agreement only."""
    units_by_run = second_look_units(rec)
    sl_verdict = {r["unit_id"]: r["verdict"] for r in sl_rows}
    finding_sl = {r["first_unit_id"]: r["sl_unit_id"] for r in sl_map if r["row_kind"] == "finding"}
    runs: dict[str, dict] = {}
    for run, units in sorted(units_by_run.items()):
        by_first = {u["first_unit_id"]: u for u in units}
        drawn = sorted((r for r in sl_map if r["row_kind"] == "narrative" and r["run_id"] == run),
                       key=lambda r: r["first_unit_id"])
        first, second, ids = [], [], []
        strata: Counter = Counter()
        for r in drawn:
            u = by_first.get(r["first_unit_id"])
            if u is None or u["first_class"] != r["first_class"] or u["stratum"] != r["stratum"]:
                raise Refused(f"second-look map row for {r['first_unit_id']} does not agree with the frozen record")
            missing = [f for f in u["finding_unit_ids"] if f not in finding_sl]
            if missing:
                raise Refused(f"second-look map lacks {len(missing)} finding unit(s) of {r['first_unit_id']}")
            ok = sl_verdict.get(r["sl_unit_id"]) == "none_wrong" and all(
                sl_verdict.get(finding_sl[f]) == "correct" for f in u["finding_unit_ids"])
            first.append(u["first_class"] == FIRST_CORRECT)
            second.append(ok)
            ids.append(u["first_unit_id"])
            strata[u["stratum"]] += 1
        boot = bootstrap_kappa(first, second, seed=bootstrap_seed)
        c2n = sum(1 for a, b in zip(first, second) if a and not b)
        n2c = sum(1 for a, b in zip(first, second) if b and not a)
        result = {
            "N": len(units), "n": len(drawn), "strata": dict(sorted(strata.items())),
            "first_class_counts": {FIRST_CORRECT: sum(first), FIRST_NOT_CORRECT: len(first) - sum(first)},
            "second_class_counts": {FIRST_CORRECT: sum(second), FIRST_NOT_CORRECT: len(second) - sum(second)},
            "sample_first_correct": sum(first), "sample_first_not_correct": len(first) - sum(first),
            "run_first_not_correct": sum(1 for u in units if u["first_class"] == FIRST_NOT_CORRECT),
            "kappa": boot["kappa"], "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
            "resamples": boot["resamples"], "degenerate_resamples": boot["degenerate_resamples"],
            "correct_to_not_correct": c2n, "not_correct_to_correct": n2c, "disagreements": c2n + n2c,
            "disagreeing_first_unit_ids": [i for i, a, b in zip(ids, first, second) if a != b],
        }
        result["gate"] = gate(result)
        assert tuple(result) == RUN_RESULT_KEYS
        runs[run] = result
    return {"agreement": AGREEMENT_LABEL, "bootstrap_seed": bootstrap_seed, "runs": runs}


def check_second_look_folder(first: Record, packet_dir: str, d2: Path, sl_meta: Mapping[str, Any],
                             first_sha: str) -> list[str]:
    """The second-look folder is the one the D-D draw of this frozen record gives: the seed
    is the derived one, and a rebuild from the frozen record with the recorded salt writes
    every file byte for byte (red-team H1). Nothing but the meta file and the files
    second-look-score writes may sit beside them."""
    gaps: list[str] = []
    run_id = str(first.meta.get("run_id", ""))
    want_seed = derived_seed(first_sha, run_id, "second_look")
    if sl_meta.get("seed") != want_seed:
        return ["second-look-meta.json records a draw seed that is not the one the frozen manifest gives"]
    salt = sl_meta.get("id_salt")
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        return ["second-look-meta.json records no id salt"]
    try:
        with tempfile.TemporaryDirectory(prefix="second-look-rebuild-") as tmp:
            details, _ = write_second_look(first, packet_dir, Path(tmp), seed=want_seed, salt=salt)
            built = _files_of(Path(tmp))
            if sl_meta.get("runs") != json.loads(json.dumps(_draw_runs(details))):
                gaps.append("second-look-meta.json per-run draw figures differ from the rebuild")
            if sl_meta.get("files") != built:
                gaps.append("second-look-meta.json file list differs from the rebuild")
            for rel in sorted(built):
                p = d2 / rel
                if not p.is_file():
                    gaps.append(f"second-look file {rel} is missing")
                elif p.read_bytes() != (Path(tmp) / rel).read_bytes():
                    gaps.append(f"second-look file {rel} differs from the rebuild of the D-D draw")
    except Refused as e:
        gaps.extend(f"second-look rebuild: {g}" for g in e.gaps)
    except Exception as e:  # any failure to rebuild is a gap, never a pass
        gaps.append(f"second-look rebuild failed: {type(e).__name__}: {e}")
    allowed = set(sl_meta.get("files") or {}) | {SECOND_LOOK_META_FILE, SECOND_LOOK_RESULT_FILE,
                                                  SECOND_LOOK_VERDICTS_FILE}
    for rel in sorted(_files_of(d2)):
        if rel not in allowed:
            gaps.append(f"second-look folder holds a file the draw did not write: {rel}")
    return gaps


def check_blind_check_folder(first: Record, d3: Path, first_sha: str,
                             freeze_commit: str) -> tuple[list[str], dict | None]:
    """The blinding check is bound to this frozen record: its meta names the same first
    manifest and freeze commit and the derived seed, every file still hashes to its meta,
    and the result is recomputed from the map and the kept answers (red-team M4).
    Returns (gaps, result)."""
    gaps: list[str] = []
    meta_path = d3 / BLIND_CHECK_META
    if not meta_path.is_file():
        return ["blind-check-meta.json not found"], None
    meta = _read_json(meta_path)
    run_id = str(first.meta.get("run_id", ""))
    if meta.get("first_manifest_sha256") != first_sha:
        gaps.append("the blinding check was not drawn from this frozen manifest")
    if meta.get("freeze_commit") != freeze_commit:
        gaps.append("the blinding check names another freeze commit than the second look")
    if meta.get("seed") != derived_seed(first_sha, run_id, "blind_check"):
        gaps.append("blind-check-meta.json records a draw seed that is not the one the frozen manifest gives")
    for rel, digest in sorted((meta.get("files") or {}).items()):
        p = d3 / rel
        if not p.is_file() or rs.sha256_file(p) != digest:
            gaps.append(f"blind-check file {rel} is missing or does not hash to blind-check-meta.json")
    for rel in (BLIND_CHECK_ANSWERS, BLIND_CHECK_RESULT):
        if not (d3 / rel).is_file():
            gaps.append(f"blind-check file {rel} not found")
    if gaps:
        return gaps, None
    mapping = rs.read_csv(d3 / BLIND_CHECK_MAP, BLIND_CHECK_MAP_COLUMNS)
    answers = rs.read_csv(d3 / BLIND_CHECK_ANSWERS, BLIND_CHECK_COLUMNS)
    try:
        again = blinding_result([{"check_id": r["check_id"], "guess": r["guess"]} for r in answers],
                                {m["check_id"]: m["display_name"] for m in mapping}, chance_of(meta))
    except Refused as e:
        return [f"blinding check: {g}" for g in e.gaps], None
    if json.loads(json.dumps(again)) != _read_json(d3 / BLIND_CHECK_RESULT):
        gaps.append("blind-check-result.json differs from the result recomputed from the kept answers")
    return gaps, again


def _second_look_score(args: argparse.Namespace) -> int:
    d, d2 = Path(args.packet_dir), Path(args.second_look_dir)
    now = _utcnow()
    gaps: list[str] = []
    meta = _read_json(d / rs.PACKET_FILES["meta"])
    run_id = str(meta.get("run_id", ""))
    mdir = Path(args.manifest_dir)
    first_manifest = mdir / f"{run_id}-manifest.json"
    out_manifest = mdir / f"{run_id}-second-look-manifest.json"
    if not first_manifest.is_file():
        raise Refused(f"first manifest {first_manifest.name} not found")
    first_bytes = first_manifest.read_bytes()
    first_sha = rs.sha256_file(first_manifest)
    man = _read_json(first_manifest)
    gaps.extend(check_private_files(str(d), man))
    if args.reviewer != man.get("reviewer"):
        gaps.append(f"--reviewer {args.reviewer!r} is not the first record's reviewer; the second look is "
                    "reported as intra-rater agreement")
    sl_meta_path = d2 / SECOND_LOOK_META_FILE
    freeze_time: datetime | None = None
    draw_time: datetime | None = None
    sl_meta: dict = {}
    if not sl_meta_path.is_file():
        gaps.append("second-look-meta.json not found")
    else:
        sl_meta = _read_json(sl_meta_path)
        if sl_meta.get("first_manifest_sha256") != first_sha:
            gaps.append("the first manifest does not hash to what second-look-meta.json recorded at the draw")
        commit = str(sl_meta.get("freeze_commit", ""))
        if not _SHA_RE.match(commit):
            gaps.append("second-look-meta.json records no freeze commit")
        elif not _inside(first_manifest, Path(args.repo)):
            gaps.append("the first manifest is not inside --repo")
        else:
            rel = first_manifest.resolve().relative_to(Path(args.repo).resolve()).as_posix()
            more, freeze_time = check_freeze_commit(first_bytes, man, repo=args.repo, sha=commit, rel_path=rel,
                                                    now=now)
            gaps.extend(more)
        try:
            draw_time = _utc(datetime.fromisoformat(str(sl_meta.get("draw_time", ""))))
        except ValueError:
            gaps.append("second-look-meta.json records no draw time")
        if draw_time is not None and freeze_time is not None and draw_time - freeze_time < MIN_GAP:
            gaps.append("second-look-meta.json records a draw less than 48 hours after the freeze commit")
        if draw_time is not None and draw_time > now:
            gaps.append("second-look-meta.json records a draw time after now")
    for p in (d2 / SECOND_LOOK_RESULT_FILE, d2 / SECOND_LOOK_VERDICTS_FILE, out_manifest):
        if p.exists():
            gaps.append(f"{p.name} already exists; it is never overwritten")
    if _inside(mdir, d):
        gaps.append("--manifest-dir is inside the packet folder")
    if _inside(d2, d):
        gaps.append("--second-look-dir is inside the packet folder, beside the first verdicts")
    first = load_frozen_record(str(d))
    if sl_meta:
        gaps.extend(check_second_look_folder(first, str(d), d2, sl_meta, first_sha))
    blinding = None
    blind_dir = Path(args.blind_check_dir) if args.blind_check_dir else None
    if len(worker_run_ids(first)) >= FULL_RUN_WORKER_RUNS and blind_dir is None:
        gaps.append("a full run needs --blind-check-dir holding blind-check-result.json")
    if blind_dir is not None:
        more, blinding = check_blind_check_folder(first, blind_dir, first_sha, str(sl_meta.get("freeze_commit", "")))
        gaps.extend(more)
    parsed_date = _parse_date(args.date)
    not_before = earliest_local_date(draw_time) if draw_time is not None else None
    if parsed_date is None:
        gaps.append(f"--date {args.date!r} is not YYYY-MM-DD")
    elif parsed_date > latest_local_date(now):
        gaps.append(f"--date {args.date} is after today")
    elif not_before is not None and parsed_date < not_before:
        gaps.append(f"--date {args.date} is before the second-look draw")
    sl_rec = load_record(str(d2), args.filled)
    gaps.extend(validate_record(sl_rec, reviewer=args.reviewer, freeze_date=args.date, packet_checks=False,
                                not_before=not_before))
    if gaps:
        raise Refused(gaps)
    sl_map = rs.read_csv(d2 / SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)
    result = second_look_result(first, sl_rec.rows, sl_map, bootstrap_seed=args.bootstrap_seed)
    rs.write_rb7_csv(d2 / SECOND_LOOK_VERDICTS_FILE, record_rows(sl_rec, ("finding", "narrative")))
    _write_json(d2 / SECOND_LOOK_RESULT_FILE, result)
    private = _files_of(d2)
    if blind_dir is not None:
        private.update({f"blind-check/{rel}": digest for rel, digest in _files_of(blind_dir).items()})
    labels = {rid: str(first.runs[rid].get("label", rid)) for rid in result["runs"]}
    manifest = {
        "manifest_version": "1", "kind": "second_look", "run_id": run_id, "agreement": AGREEMENT_LABEL,
        "first_manifest_sha256": first_sha, "freeze_commit": sl_meta.get("freeze_commit", ""),
        "draw_seed": sl_meta.get("seed"), "bootstrap_seed": args.bootstrap_seed, "private_files": private,
        "runs": {labels[rid]: {k: r[k] for k in PUBLIC_RUN_RESULT_KEYS} for rid, r in sorted(result["runs"].items())},
        "blinding": blinding, "reviewer": args.reviewer, "date": args.date, "statement": SECOND_LOOK_STATEMENT,
    }
    assert tuple(manifest) == SECOND_LOOK_MANIFEST_KEYS
    mdir.mkdir(parents=True, exist_ok=True)
    _write_json(out_manifest, manifest)
    for label, r in manifest["runs"].items():
        ci = "none (kappa undefined)" if r["ci_lower"] is None else f"[{r['ci_lower']}, {r['ci_upper']}]"
        print(f"{label}: {AGREEMENT_LABEL} agreement, n {r['n']} of N {r['N']}, kappa {r['kappa']}, "
              f"CI {ci}, disagreements {r['disagreements']}, gate {r['gate']['branch']}: {r['gate']['result']}")
    print(out_manifest)
    print(trailer(args.reviewer, args.date))
    return EXIT_OK


# --- Blinding check -------------------------------------------------------------------------

def blinding_result(guesses: Sequence[dict], truth: Mapping[str, str], chance: float = BLIND_CHECK_CHANCE) -> dict:
    """Share of checks named correctly with a two-sided 95% Clopper-Pearson interval;
    `blinding not maintained` when the lower limit is above chance."""
    answered = {g["check_id"]: g["guess"] for g in guesses}
    gaps = [f"check {cid} has no guess" for cid in sorted(truth) if not answered.get(cid)]
    gaps += [f"guess for unknown check {cid}" for cid in sorted(answered) if cid not in truth]
    if gaps:
        raise Refused(gaps)
    n = len(truth)
    x = sum(1 for cid, label in truth.items() if answered[cid] == label)
    lo, hi = load_pilot_report().clopper_pearson(x, n)
    result = {"n": n, "named_correctly": x, "share": x / n if n else None, "ci_lower": lo, "ci_upper": hi,
              "chance": chance, "status": "blinding not maintained" if lo > chance else "lower limit not above chance"}
    assert tuple(result) == BLINDING_KEYS
    return result


def chance_of(meta: Mapping[str, Any]) -> float:
    """Chance is one over the number of names offered: 0.25 on a full run of four worker
    runs, as pre-registered; 0.5 with two (red-team M4)."""
    k = len(meta.get("choices") or [])
    return 1 / k if k else BLIND_CHECK_CHANCE


def parse_models(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (text or "").split(","):
        label, sep, name = part.partition("=")
        if not sep or not label.strip() or not name.strip():
            raise Refused(f"--models entry {part!r} is not LABEL=NAME")
        if label.strip() in out:
            raise Refused(f"--models names {label.strip()!r} twice")
        out[label.strip()] = name.strip()
    return out


def _blind_check_html(checks: Sequence[dict], choices: Sequence[str]) -> str:
    data = json.dumps({"checks": [{"check_id": c["check_id"], "narrative": c["narrative"]} for c in checks],
                       "choices": list(choices)}, ensure_ascii=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blinding check</title>
<style>
:root { --bg: #f7f7f5; --card: #ffffff; --ink: #1d1d1b; --muted: #5d5d58; --line: #d9d9d4; --accent: #2f5d8a; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --card: #20201e; --ink: #ececea; --muted: #a5a5a0; --line: #3a3a37; --accent: #8db4dc; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }
header { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line); padding: 12px 16px; }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin: 0 0 16px; }
.id { color: var(--muted); font-size: 13px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
select, button { font: inherit; padding: 6px 10px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); max-width: 100%; }
button { cursor: pointer; }
button.primary { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header><strong>Blinding check</strong> <span id="progress"></span>
<button class="primary" id="save" type="button">Save answers file</button></header>
<main>
<p class="card">For each narrative, pick the worker model you think wrote it. Every narrative is shown
once. Save the answers file when you are done.</p>
<div id="list"></div>
</main>
<script type="application/json" id="data">""" + data + """</script>
<script>
(function () {
  var data = JSON.parse(document.getElementById("data").textContent);
  var answers = {};
  var list = document.getElementById("list");
  function progress() {
    var n = 0;
    for (var k in answers) { if (answers[k]) { n += 1; } }
    document.getElementById("progress").textContent = n + " of " + data.checks.length + " answered";
  }
  data.checks.forEach(function (c) {
    var card = document.createElement("div");
    card.className = "card";
    var id = document.createElement("div");
    id.className = "id";
    id.textContent = c.check_id;
    var text = document.createElement("div");
    text.className = "text";
    text.textContent = c.narrative;
    var sel = document.createElement("select");
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "choose a model";
    sel.appendChild(blank);
    data.choices.forEach(function (name) {
      var o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      sel.appendChild(o);
    });
    sel.addEventListener("change", function () { answers[c.check_id] = sel.value; progress(); });
    card.appendChild(id);
    card.appendChild(text);
    card.appendChild(sel);
    list.appendChild(card);
  });
  function cell(v) {
    v = String(v).replace(/\\r/g, "");
    return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  document.getElementById("save").addEventListener("click", function () {
    var lines = ["check_id,narrative,guess"];
    data.checks.forEach(function (c) {
      lines.push([cell(c.check_id), cell(c.narrative), cell(answers[c.check_id] || "")].join(","));
    });
    var blob = new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "blind-check.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
  progress();
})();
</script>
</body>
</html>
"""


def _blind_check_draw(args: argparse.Namespace) -> int:
    now = _utcnow()
    out = Path(args.out) if args.out else _default_out(args.packet_dir, "blind-check")
    gaps = _out_dir_free(out, args.packet_dir)
    try:
        rec, _man, _, mpath = check_frozen(args.packet_dir, freeze_commit=args.freeze_commit, repo=args.repo,
                                       manifest=args.manifest, now=now)
    except Refused as e:
        raise Refused(gaps + e.gaps) from e
    workers = worker_run_ids(rec)
    if len(workers) < 2:
        gaps.append("the blinding check needs at least two worker runs")
    models = parse_models(args.models)
    labels = {rid: str(rec.runs[rid].get("label", "")) for rid in workers}
    templates = {str(info.get("label", "")) for rid, info in rec.runs.items() if info.get("template_baseline")}
    for label in sorted(set(models) - set(labels.values())):
        gaps.append(f"--models names {label!r}, which is not a worker run of this packet"
                    + (" (a template baseline is never offered)" if label in templates else ""))
    for label in sorted(set(labels.values()) - set(models)):
        gaps.append(f"--models does not name worker run {label!r}")
    if len(set(models.values())) != len(models):
        gaps.append("--models gives two runs the same display name")
    if gaps:
        raise Refused(gaps)
    pool = blind_check_pool(rec)
    first_sha = rs.sha256_file(mpath)
    seed = derived_seed(first_sha, str(rec.meta.get("run_id", "")), "blind_check")
    salt = secrets.token_hex(16)  # kept under sealed/ only: a check id cannot be traced to its run
    k = min(BLIND_CHECK_UNITS, len(pool))
    drawn = random.Random(f"{seed}|blind_check").sample(pool, k)
    checks = []
    for uid, run in drawn:
        cid = "BC-" + hashlib.sha256(f"{salt}|blind_check|{uid}|{run}".encode()).hexdigest()[:12]
        checks.append({"check_id": cid, "narrative": rec.units[uid]["narrative"], "guess": "", "run_id": run,
                       "run_label": labels[run], "display_name": models[labels[run]]})
    choices = sorted(models.values())
    random.Random(f"{seed}|blind_check|choices").shuffle(choices)
    (out / "reviewer").mkdir(parents=True, exist_ok=True)
    (out / "sealed").mkdir(parents=True, exist_ok=True)
    rs.write_csv(out / BLIND_CHECK_CSV, BLIND_CHECK_COLUMNS, [{c: ch[c] for c in BLIND_CHECK_COLUMNS} for ch in checks])
    rs.write_lf(out / BLIND_CHECK_HTML, _blind_check_html(checks, choices))
    rs.write_csv(out / BLIND_CHECK_MAP, BLIND_CHECK_MAP_COLUMNS,
                 [{c: ch[c] for c in BLIND_CHECK_MAP_COLUMNS} for ch in checks])
    files = {p.relative_to(out).as_posix(): rs.sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    _write_json(out / BLIND_CHECK_META, {"seed": seed, "id_salt": salt, "freeze_commit": args.freeze_commit,
                                         "first_manifest_sha256": first_sha, "draw_time": now.isoformat(),
                                         "pool": len(pool), "drawn": k, "choices": choices, "files": files})
    if k < BLIND_CHECK_UNITS:
        print(f"only {k} narrative unit(s) occur in exactly one worker run; all drawn", file=sys.stderr)
    print(f"blinding check drawn: {k} narrative(s); open {out / BLIND_CHECK_HTML}")
    return EXIT_OK


def _blind_check_score(args: argparse.Namespace) -> int:
    d3 = Path(args.blind_check_dir)
    gaps: list[str] = []
    meta_path = d3 / BLIND_CHECK_META
    if not meta_path.is_file():
        raise Refused("blind-check-meta.json not found")
    meta = _read_json(meta_path)
    for rel, digest in sorted((meta.get("files") or {}).items()):
        p = d3 / rel
        if not p.is_file() or rs.sha256_file(p) != digest:
            gaps.append(f"blind-check file {rel} is missing or does not hash to blind-check-meta.json")
    for rel in (BLIND_CHECK_RESULT, BLIND_CHECK_ANSWERS):
        if (d3 / rel).exists():
            gaps.append(f"{Path(rel).name} already exists; it is never overwritten")
    if gaps:
        raise Refused(gaps)
    try:
        filled = rs.read_csv(args.filled, BLIND_CHECK_COLUMNS)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the answers file: {e}") from e
    mapping = rs.read_csv(d3 / BLIND_CHECK_MAP, BLIND_CHECK_MAP_COLUMNS)
    names = {m["display_name"] for m in mapping} | set(meta.get("choices") or [])
    for row in filled:
        if row["guess"] and row["guess"] not in names:
            gaps.append(f"check {row['check_id']}: guess {row['guess']!r} is not one of the offered names")
    ids = [row["check_id"] for row in filled]
    for cid in sorted({c for c in ids if ids.count(c) > 1}):
        gaps.append(f"check {cid} answered more than once")
    if gaps:
        raise Refused(gaps)
    result = blinding_result([{"check_id": r["check_id"], "guess": r["guess"]} for r in filled],
                             {m["check_id"]: m["display_name"] for m in mapping}, chance_of(meta))
    # the guesses are kept, so second-look-score can recompute this result (red-team M4)
    rs.write_csv(d3 / BLIND_CHECK_ANSWERS, BLIND_CHECK_COLUMNS, filled)
    _write_json(d3 / BLIND_CHECK_RESULT, result)
    print(f"named correctly {result['named_correctly']} of {result['n']}, 95% CI "
          f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] against chance {result['chance']}: {result['status']}")
    return EXIT_OK


# --- CLI ------------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 is not a contract code
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)

    def exit(self, status: int = 0, message: str | None = None) -> None:  # help: no record work ran
        if message:
            print(message, file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def _label_path(text: str) -> tuple[str, str]:
    label, sep, path = text.partition("=")
    if not sep or not _LABEL_RE.match(label) or not path:
        raise argparse.ArgumentTypeError(f"{text!r} is not LABEL=PATH")
    return label, path


def _sitting(text: str) -> tuple[str, str]:
    parts = text.split("/")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"{text!r} is not START/END")
    return parts[0], parts[1]


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="Review record: validate, freeze, second look, blinding check (no model call).")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    v = sub.add_parser("validate", help="check a filled answers file against the packet")
    v.add_argument("--packet-dir", required=True)
    v.add_argument("--filled", required=True)
    f = sub.add_parser("freeze", help="validate, rebuild the packet, write the record and the manifest")
    f.add_argument("--packet-dir", required=True)
    f.add_argument("--filled", required=True)
    f.add_argument("--run", dest="runs", action="append", type=_label_path, required=True, metavar="LABEL=PATH")
    f.add_argument("--reviewer", required=True)
    f.add_argument("--date", required=True)
    f.add_argument("--run-commit", required=True)
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--sittings", help="sittings.csv saved from the review page (columns start,end)")
    g.add_argument("--sitting", dest="sitting", action="append", type=_sitting, metavar="START/END")
    f.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    f.add_argument("--expanded-csv")
    for name, default_out in (("second-look-draw", "second-look"), ("blind-check-draw", "blind-check")):
        s = sub.add_parser(name)
        s.add_argument("--packet-dir", required=True)
        s.add_argument("--freeze-commit", required=True)
        s.add_argument("--out", help=f"default: <packet-dir>-{default_out}, beside the packet folder")
        s.add_argument("--repo", default=".")
        s.add_argument("--manifest", help="default: <repo>/docs/evidence/reviews/<run-id>-manifest.json")
        if name == "blind-check-draw":
            s.add_argument("--models", required=True, help="LABEL=NAME,LABEL=NAME,... for every worker run")
    b = sub.add_parser("blind-check-score")
    b.add_argument("--packet-dir", required=True)
    b.add_argument("--blind-check-dir", required=True)
    b.add_argument("--filled", required=True)
    s2 = sub.add_parser("second-look-score")
    s2.add_argument("--packet-dir", required=True)
    s2.add_argument("--second-look-dir", required=True)
    s2.add_argument("--filled", required=True)
    s2.add_argument("--reviewer", required=True)
    s2.add_argument("--date", required=True)
    s2.add_argument("--bootstrap-seed", type=int, required=True)
    s2.add_argument("--blind-check-dir")
    s2.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    s2.add_argument("--repo", default=".", help="the repository holding the freeze commit")
    return p


def _read_sittings_file(path: str) -> list[tuple[str, str]]:
    try:
        rows = rs.read_csv(path, SITTINGS_COLUMNS)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the sittings file: {e}") from e
    return [(r["start"], r["end"]) for r in rows]


def _validate(args: argparse.Namespace) -> int:
    rec = load_record(args.packet_dir, args.filled)
    gaps = validate_record(rec)
    if gaps:
        raise Refused(gaps)
    kinds = Counter(u["row_kind"] for u in rec.units.values())
    print(f"record complete: {kinds.get('finding', 0)} finding unit(s), {kinds.get('narrative', 0)} narrative "
          f"unit(s), {kinds.get('fm_audit', 0)} audit unit(s)")
    return EXIT_OK


def _freeze(args: argparse.Namespace) -> int:
    sittings = _read_sittings_file(args.sittings) if args.sittings else list(args.sitting)
    path, line = freeze(args.packet_dir, args.filled, args.runs, reviewer=args.reviewer, freeze_date=args.date,
                        run_commit=args.run_commit, sittings=sittings, manifest_dir=args.manifest_dir,
                        expanded_csv=args.expanded_csv)
    print(path)
    print(line)
    return EXIT_OK


COMMANDS = {"validate": _validate, "freeze": _freeze, "second-look-draw": _second_look_draw,
            "blind-check-draw": _blind_check_draw, "blind-check-score": _blind_check_score,
            "second-look-score": _second_look_score}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except Refused as e:
        print(f"REFUSED: {len(e.gaps)} gap(s)", file=sys.stderr)
        for gap in e.gaps:
            print(f"  - {gap}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as a record
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
