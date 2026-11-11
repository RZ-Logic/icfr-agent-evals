"""Review load per full run of four workers, re-derived from the timed dev rehearsal. NO API key, no model call.

Usage:
  python scripts/review_load.py --packet-dir D [--sittings S.csv [--answers F.csv]]
         [--dev-cases standard=8,hard=12] [--json OUT.json]

Pre-registration section 5 (expected review load, owner decision D-G, 2026-11-10) and the
section 13 pilot row: the estimate of about 78 to 259 hours per full run of the four
workers is re-derived from the timed dev rehearsal before the tag. The D-J whole-narrative
read of the unrouted, uncoded epochs of the 54 difficulty-bearing cases (about 5 to 11
hours, owner decision D-J, 2026-11-10) is shown beside that range, as section 5 states it.

Inputs, all read only:
  --packet-dir  the packet folder scripts/review_packet.py wrote: reviewer/packet.csv
                (unit counts by kind and narrative class, words per narrative, case ids)
                and sealed/summary.json (the totals, cross-checked against packet.csv).
  --sittings    sittings.csv saved from the review page (columns start,end), read and
                checked with scripts/review_freeze.py's own sittings code. Without it the
                script prints the D-G time model on this packet's counts, marked not measured.
  --answers     optional: the filled answers file. Only the row kind, the unit id, the date
                and whether a verdict is blank, not_reviewed or something else are read;
                no other verdict value is read, and no verdict value is printed. Rows left
                blank and finding rows marked not_reviewed (the review desk's "Mark the rest
                not_reviewed" button stamps that day's date on every unanswered finding at
                once) are not reviewed work: they are left out of the fit by date and out
                of the dev packet weights the measured time is shared over. The file is
                used to fit minutes per finding, narrative and audit unit from the time on
                each date, when the dates allow it. Answers saved in the review desk before
                its 2026-11-11 fix may hold findings the desk filled automatically when
                their narrative was answered; those carry a date but no review time of
                their own, and the file cannot tell them apart, so the fit by date is a
                weak diagnostic and is never used in the range.

How the measured rate is used. The sittings give one total time. The D-G time model
(section 5 and LOAD.md section 3 assumptions: 15 to 30 seconds plus 60 to 120 words a
minute per narrative; 30 to 90 seconds per flagged sentence, that is per group of finding
units on one narrative unit and one routed sentence set, plus 10 to 20 seconds for each
further finding unit on it; 90 to 180 seconds of case facts per case with a route; 30 to
120 seconds per audit row) gives each unit kind a weight. The measured total is shared out
by those weights, once with the low weights and once with the high ones, so both splits
add up to the measured time on the dev packet. Each split is then scaled to the full run:
per stratum (standard, hard) from the dev case counts to the evaluation split's 146 and 54
cases, with the D-G volume assumptions (every worker routing like the dev worker in the
low case; Haiku 5.5 1.05, Sonnet 5 and Opus 5.5 1.25 times in the high case; case facts
read once per case for all four workers in the low case and once per case per worker in
the high case; the second look at max(100, 10 percent) routed narrative units per worker
run, each with its findings, D-D). A fit by date, when it is possible, is printed beside
it and not used in the range.

This script reads no log, no score and no verdict value, and computes no score.

Exit codes: 0 printed; 3 refused (a packet that does not match its summary, a case off the
dev split, a bad sittings file, a help request or an argument error).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from icfr_agent_evals import review_schema as rs

EXIT_OK, EXIT_REFUSED = 0, 3
SCRIPTS_DIR = Path(__file__).resolve().parent
REVIEW_FREEZE_SCRIPT = SCRIPTS_DIR / "review_freeze.py"

# Evaluation split per worker run (section 13 generator row): 146 standard, 30 edge and
# 24 adversarial cases; edge and adversarial are the hard (difficulty-bearing) stratum.
EVAL_CASES: dict[str, int] = {"standard": 146, "hard": 54}
DEV_CASES_DEFAULT: dict[str, int] = {"standard": 8, "hard": 12}   # the 20-case pilot, 12 hard (section 13)
STATED_RANGE_HOURS = (78.0, 259.0)          # section 5, D-G
STATED_DJ_HOURS = (5.0, 11.0)               # section 5, D-J

# Section 5 (D-G) time assumptions, in seconds, low and high (LOAD.md section 3): "finding"
# is the verdict per flagged sentence (one group of finding units), "further_key" each
# further finding unit on that same sentence.
TIME_MODEL: dict[str, dict[str, float]] = {
    "low": {"read_fixed": 15, "wpm": 120, "finding": 30, "further_key": 10, "case": 90, "audit": 30},
    "high": {"read_fixed": 30, "wpm": 60, "finding": 90, "further_key": 20, "case": 180, "audit": 120},
}
# D-G volume assumptions per worker (LOAD.md section 4): route volume against the dev
# worker, and the high audit rows per worker run.
WORKERS: tuple[str, ...] = ("Sonnet 5.5", "Haiku 5.5", "Sonnet 5", "Opus 5.5")
ROUTE_MULTIPLIER: dict[str, dict[str, float]] = {
    "low": {w: 1.0 for w in WORKERS},
    "high": {"Sonnet 5.5": 1.0, "Haiku 5.5": 1.05, "Sonnet 5": 1.25, "Opus 5.5": 1.25},
}
HIGH_AUDIT_ROWS: dict[str, int] = {"Sonnet 5.5": 70, "Haiku 5.5": 200, "Sonnet 5": 70, "Opus 5.5": 70}
SECOND_LOOK_FLOOR, SECOND_LOOK_SHARE = 100, 0.10   # D-D: 100 units or 10 percent per worker run
UNIT_KINDS: tuple[str, ...] = ("finding", "narrative", "fm_audit")
ROUTED_CLASSES = (rs.NARRATIVE_CLASS_ROUTED, rs.NARRATIVE_CLASS_FM_CODED)


class Refused(Exception):
    """A refusal: nothing is printed as an estimate."""


@lru_cache(maxsize=1)
def load_review_freeze() -> Any:
    """scripts/review_freeze.py, for its sittings reader and checks (private module name)."""
    name = "review_freeze_for_review_load"
    spec = importlib.util.spec_from_file_location(name, REVIEW_FREEZE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module by name
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


# --- Packet counts ------------------------------------------------------------------------

@dataclass(frozen=True)
class PacketCounts:
    run_id: str
    scope: str
    runs: int
    # stratum -> finding, finding_groups, routed, only, audit, cases, cases_routed, words_routed, words_only
    by_stratum: dict[str, dict[str, float]]
    findings: int                             # finding members (before grouping), from summary.json
    rows: tuple[Mapping[str, str], ...] = ()  # the packet.csv rows the counts come from

    def total(self, key: str) -> float:
        return sum(s[key] for s in self.by_stratum.values())

    def without(self, unit_ids: frozenset[str]) -> "PacketCounts":
        """The same packet with the given units left out (units not reviewed)."""
        kept = tuple(r for r in self.rows if r["unit_id"] not in unit_ids)
        by, _, _ = count_rows(kept)
        return PacketCounts(run_id=self.run_id, scope=self.scope, runs=self.runs, by_stratum=by,
                            findings=self.findings, rows=kept)


def finding_group_key(row: Mapping[str, str]) -> tuple[str, str]:
    """LOAD.md section 3 groups finding units by flagged sentence: the narrative unit(s) and the
    routed sentence(s) of the unit, as packet.csv writes them. A unit that routes no sentence is
    its own group."""
    sentences = row.get("routed_sentences", "").strip()
    narratives = row.get("narrative_unit_ids", "").strip()
    if not sentences or sentences == "[]" or not narratives or narratives == "[]":
        return ("unit", row["unit_id"])
    return (narratives, sentences)


def count_rows(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, dict[str, float]], dict[str, set[str]], Counter]:
    """Counts by stratum, cases with any unit by stratum, and units by row kind."""
    blank = {"finding": 0.0, "finding_groups": 0.0, "routed": 0.0, "only": 0.0, "audit": 0.0,
             "words_routed": 0.0, "words_only": 0.0}
    by: dict[str, dict[str, float]] = {s: dict(blank) for s in EVAL_CASES}
    cases: dict[str, set[str]] = defaultdict(set)
    routed_cases: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    kinds: Counter = Counter()
    for row in rows:
        kind = row["row_kind"]
        if kind not in UNIT_KINDS:
            raise Refused(f"packet.csv row {row['unit_id']}: unknown row kind {kind!r}")
        kinds[kind] += 1
        s = stratum_of(row["case_id"])
        cases[s].add(row["case_id"])
        if kind == "finding":
            by[s]["finding"] += 1
            groups[s].add(finding_group_key(row))
            routed_cases[s].add(row["case_id"])
        elif kind == "fm_audit":
            by[s]["audit"] += 1
            routed_cases[s].add(row["case_id"])
        else:
            words = len(row["narrative"].split())
            if row["route_class"] in ROUTED_CLASSES:
                by[s]["routed"] += 1
                by[s]["words_routed"] += words
                routed_cases[s].add(row["case_id"])
            elif row["route_class"] == rs.NARRATIVE_CLASS_ONLY:
                by[s]["only"] += 1
                by[s]["words_only"] += words
            else:
                raise Refused(f"packet.csv row {row['unit_id']}: unknown narrative class {row['route_class']!r}")
    for s in by:
        by[s]["cases"] = float(len(cases[s]))
        by[s]["cases_routed"] = float(len(routed_cases[s]))
        by[s]["finding_groups"] = float(len(groups[s]))
    return by, cases, kinds


def stratum_of(case_id: str) -> str:
    if not case_id.startswith("dev-"):
        raise Refused(f"case {case_id} is not a dev case; this script reads dev packets only")
    if "-hc-" in case_id:
        return "hard"
    if "-std-" in case_id:
        return "standard"
    raise Refused(f"case {case_id}: no stratum in its id (expected dev-hc-* or dev-std-*)")


def read_packet(packet_dir: str | Path) -> PacketCounts:
    root = Path(packet_dir)
    try:
        summary = json.loads((root / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
        meta = json.loads((root / rs.PACKET_FILES["meta"]).read_text(encoding="utf-8"))
        rows = rs.read_csv(root / rs.PACKET_FILES["packet"], rs.PACKET_COLUMNS)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the packet in {root}: {e}") from e
    totals = summary.get("totals") or {}
    by, cases, kinds = count_rows(rows)
    gaps = []
    for kind, key in (("finding", "finding_units"), ("narrative", "narrative_units"), ("fm_audit", "audit_units")):
        if totals.get(key) != kinds[kind]:
            gaps.append(f"{key}: summary.json {totals.get(key)}, packet.csv {kinds[kind]}")
    nclass = totals.get("narrative_units_by_class") or {}
    routed = sum(int(nclass.get(c, 0)) for c in ROUTED_CLASSES)
    if routed != sum(by[s]["routed"] for s in by) or int(nclass.get(rs.NARRATIVE_CLASS_ONLY, 0)) != sum(
            by[s]["only"] for s in by):
        gaps.append(f"narrative_units_by_class: summary.json {nclass} does not match packet.csv")
    if totals.get("cases_with_units") != sum(len(c) for c in cases.values()):
        gaps.append(f"cases_with_units: summary.json {totals.get('cases_with_units')}, "
                    f"packet.csv {sum(len(c) for c in cases.values())}")
    if gaps:
        raise Refused("the packet does not match its summary: " + "; ".join(gaps))
    runs = len([k for k in summary if k != "totals"])
    return PacketCounts(run_id=str(meta.get("run_id")), scope=str(meta.get("scope")), runs=runs,
                        by_stratum=by, findings=int(totals.get("findings") or 0), rows=tuple(rows))


# --- Sittings and the optional answers file ------------------------------------------------

def read_sittings(path: str | Path) -> tuple[list[tuple[str, str]], float]:
    """The sitting pairs and total hours, through review_freeze's own reader and checks."""
    rf = load_review_freeze()
    try:
        pairs = rf._read_sittings_file(str(path))
    except rf.Refused as e:
        raise Refused(str(e)) from e
    gaps = rf.check_sittings(pairs, None)
    if gaps:
        raise Refused("the sittings file fails review_freeze's checks: " + "; ".join(gaps))
    return pairs, rf._review_hours(pairs)


NOT_REVIEWED = "not_reviewed"


@dataclass(frozen=True)
class Answers:
    """What the answers file says about review work: reviewed rows per date and kind, and the
    unit ids that were not reviewed (left blank, or a finding marked not_reviewed)."""
    by_date: dict[str, Counter]
    not_reviewed_ids: frozenset[str]
    blank: int
    marked_not_reviewed: int


def read_answers(path: str | Path) -> Answers:
    """Reads row_kind, unit_id, date and whether verdict is blank or not_reviewed; nothing else of
    the verdict. A finding marked not_reviewed is not reviewed work (the desk's bulk button
    stamps that day's date on every unanswered finding at once), so it is not counted."""
    try:
        rows = rs.read_csv(path, rs.RB7_COLUMNS)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the answers file: {e}") from e
    out: dict[str, Counter] = defaultdict(Counter)
    skipped: set[str] = set()
    blank = marked = 0
    for row in rows:
        if row["row_kind"] not in UNIT_KINDS:
            continue
        verdict = row["verdict"].strip()
        if not verdict:
            blank += 1
            skipped.add(row["unit_id"])
        elif verdict == NOT_REVIEWED:
            marked += 1
            skipped.add(row["unit_id"])
        else:
            out[row["date"]][row["row_kind"]] += 1
    return Answers(by_date=dict(out), not_reviewed_ids=frozenset(skipped), blank=blank, marked_not_reviewed=marked)


def answered_by_date(path: str | Path) -> dict[str, Counter]:
    """Reviewed rows per date and row kind: blank and not_reviewed rows are not counted."""
    return read_answers(path).by_date


def seconds_by_date(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Sitting seconds per local start date (the date the review page records, in its own offset)."""
    out: dict[str, float] = defaultdict(float)
    for start, end in pairs:
        s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
        out[s.date().isoformat()] += (e - s).total_seconds()
    return dict(out)


def fit_by_date(pairs: Sequence[tuple[str, str]], answered: Mapping[str, Counter]) -> dict[str, Any]:
    """Minutes per unit kind by least squares over dates: time on a date = sum of rate x units
    answered that date. Allowed only when every answered date has a sitting, there are more
    dates than kinds, the system has full rank and every fitted rate is above zero."""
    import numpy as np

    secs = seconds_by_date(pairs)
    kinds = [k for k in UNIT_KINDS if sum(c.get(k, 0) for c in answered.values()) > 0]
    dates = sorted(set(secs) | set(answered))
    no_sitting = sorted(d for d in answered if d not in secs)
    if no_sitting:
        return {"allowed": False, "why": f"answers dated {', '.join(no_sitting)} have no sitting on that date"}
    if not kinds:
        return {"allowed": False, "why": "no answered row"}
    if len(dates) <= len(kinds):
        return {"allowed": False, "why": f"{len(dates)} dated sittings for {len(kinds)} unit kinds; "
                                         "a fit needs more dates than kinds"}
    a = np.array([[float(answered.get(d, Counter()).get(k, 0)) for k in kinds] for d in dates])
    b = np.array([secs.get(d, 0.0) for d in dates])
    coef, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < len(kinds):
        return {"allowed": False, "why": "the unit counts per date do not separate the kinds (rank too low)"}
    if any(c <= 0 for c in coef):
        return {"allowed": False, "why": "a fitted rate is zero or negative; the dates do not separate the kinds"}
    resid = b - a @ coef
    return {"allowed": True, "dates": len(dates),
            "minutes_per_unit": {k: float(c) / 60 for k, c in zip(kinds, coef)},
            "rms_residual_minutes": float(np.sqrt(np.mean(resid ** 2))) / 60}


# --- The time model and its projection ----------------------------------------------------

def narrative_seconds(words: float, units: float, m: Mapping[str, float]) -> float:
    """Seconds to read `units` narratives holding `words` words in all."""
    return units * m["read_fixed"] + words / m["wpm"] * 60


def finding_seconds(pc: PacketCounts, m: Mapping[str, float]) -> float:
    """LOAD.md section 3: the verdict rate per flagged sentence (group), plus the further-key
    rate for each further finding unit on that sentence."""
    groups, units = pc.total("finding_groups"), pc.total("finding")
    return groups * m["finding"] + (units - groups) * m["further_key"]


def dev_model_seconds(pc: PacketCounts, m: Mapping[str, float]) -> dict[str, float]:
    """The D-G model's seconds for the dev packet as one sitting set: case facts once per case
    with a route."""
    return {
        "narrative_routed": narrative_seconds(pc.total("words_routed"), pc.total("routed"), m),
        "narrative_only": narrative_seconds(pc.total("words_only"), pc.total("only"), m),
        "finding": finding_seconds(pc, m),
        "case": pc.total("cases_routed") * m["case"],
        "audit": pc.total("audit") * m["audit"],
    }


def unit_seconds(pc: PacketCounts, m: Mapping[str, float], scale: float) -> dict[str, float]:
    """Seconds per unit of each kind: the model's, times the measured scale (1 when unmeasured).
    A finding unit's seconds are the packet's mean under the per-sentence rule, so the projection
    keeps this packet's share of further keys."""
    def per(words_key: str, units_key: str) -> float:
        n = pc.total(units_key)
        return narrative_seconds(pc.total(words_key), n, m) / n if n else m["read_fixed"]
    n_find = pc.total("finding")
    return {
        "narrative_routed": per("words_routed", "routed") * scale,
        "narrative_only": per("words_only", "only") * scale,
        "finding": (finding_seconds(pc, m) / n_find if n_find else m["finding"]) * scale,
        "case": m["case"] * scale,
        "audit": m["audit"] * scale,
    }


def projected_per_worker(pc: PacketCounts, dev_cases: Mapping[str, int]) -> dict[str, float]:
    """Units per worker run on the evaluation split: each stratum scaled from its dev cases."""
    out: dict[str, float] = defaultdict(float)
    for s, counts in pc.by_stratum.items():
        if dev_cases.get(s, 0) <= 0:
            raise Refused(f"--dev-cases gives no {s} cases")
        if counts["cases"] > dev_cases[s]:
            raise Refused(f"the packet holds {int(counts['cases'])} {s} cases, more than --dev-cases {dev_cases[s]}")
        f = EVAL_CASES[s] / dev_cases[s]
        for key in ("finding", "routed", "only", "audit", "cases", "cases_routed"):
            out[key] += counts[key] * f
    return dict(out)


def project(pc: PacketCounts, per_unit: Mapping[str, Mapping[str, float]],
            dev_cases: Mapping[str, int]) -> dict[str, Any]:
    """Hours per full run of four workers, low and high, and the D-J narrative-only read beside it."""
    p = projected_per_worker(pc, dev_cases)
    fpn = pc.total("finding") / pc.total("routed") if pc.total("routed") else 0.0
    out: dict[str, Any] = {"per_worker_units": p, "findings_per_routed_narrative": fpn}
    for case in ("low", "high"):
        t = per_unit[case]
        rows = {}
        for w in WORKERS:
            k = ROUTE_MULTIPLIER[case][w]
            routed, findings = p["routed"] * k, p["finding"] * k
            if case == "low":
                case_s = p["cases_routed"] * t["case"] / len(WORKERS)   # once per routed case for all four
                audit_rows = p["audit"]
            else:
                case_s = p["cases_routed"] * t["case"]                  # once per routed case per worker
                audit_rows = max(p["audit"] * k, HIGH_AUDIT_ROWS[w])
            n2 = max(SECOND_LOOK_FLOOR, math.ceil(SECOND_LOOK_SHARE * routed))
            parts = {
                "narrative_read": routed * t["narrative_routed"],
                "finding_verdicts": findings * t["finding"],
                "case_facts": case_s,
                "audit": audit_rows * t["audit"],
                "second_look": n2 * (t["narrative_routed"] + fpn * t["finding"]),
            }
            rows[w] = {"hours": {k_: v / 3600 for k_, v in parts.items()},
                       "total_hours": sum(parts.values()) / 3600, "routed_narratives": routed,
                       "finding_units": findings, "audit_rows": audit_rows, "second_look_units": n2}
        dj = p["only"] * t["narrative_only"] * len(WORKERS) / 3600
        out[case] = {"workers": rows, "total_hours": sum(r["total_hours"] for r in rows.values()),
                     "dj_hours": dj, "dj_epochs_per_full_run": p["only"] * len(WORKERS)}
    return out


def derive(pc: PacketCounts, sittings: Sequence[tuple[str, str]] | None, hours: float | None,
           dev_cases: Mapping[str, int], answers: Answers | None = None) -> dict[str, Any]:
    measured = sittings is not None
    # The measured time is shared over the work done: units the answers file leaves blank or
    # marks not_reviewed are left out of the dev weights (never out of the projection).
    reviewed = pc.without(answers.not_reviewed_ids) if (measured and answers is not None) else pc
    units_dev = (reviewed.total("finding") + reviewed.total("routed") + reviewed.total("only")
                 + reviewed.total("audit"))
    if measured and units_dev <= 0:
        raise Refused("the answers file leaves every packet unit unreviewed; no measured rate can be derived")
    per_unit, scales, dev_model_hours = {}, {}, {}
    for case, m in TIME_MODEL.items():
        dev_s = dev_model_seconds(reviewed, m)
        dev_model_hours[case] = sum(dev_s.values()) / 3600
        scale = (hours * 3600 / sum(dev_s.values())) if measured else 1.0
        scales[case] = scale
        per_unit[case] = unit_seconds(pc, m, scale)
    result: dict[str, Any] = {
        "basis": "measured" if measured else "model only (no sittings: the D-G assumptions, not measured)",
        "packet": {"run_id": pc.run_id, "scope": pc.scope, "worker_runs": pc.runs,
                   "finding_units": pc.total("finding"), "findings": pc.findings,
                   "finding_groups": pc.total("finding_groups"), "cases_routed": pc.total("cases_routed"),
                   "narrative_units_routed": pc.total("routed"), "narrative_units_only": pc.total("only"),
                   "audit_units": pc.total("audit"), "cases": pc.total("cases"),
                   "mean_words_routed": pc.total("words_routed") / pc.total("routed") if pc.total("routed") else 0,
                   "mean_words_only": pc.total("words_only") / pc.total("only") if pc.total("only") else 0,
                   "by_stratum": pc.by_stratum},
        "dev_cases": dict(dev_cases),
        "dev_model_hours": dev_model_hours,
        "measured_hours": hours, "sittings": len(sittings) if sittings is not None else 0,
        "measured_to_model": scales if measured else None,
        "blended_minutes_per_unit": (hours * 60 / units_dev) if measured and units_dev else None,
        "minutes_per_unit": {case: {k: v / 60 for k, v in pu.items()} for case, pu in per_unit.items()},
        "projection": project(pc, per_unit, dev_cases),
        "stated": {"range_hours": STATED_RANGE_HOURS, "dj_hours": STATED_DJ_HOURS},
        "fit": None,
        "answers": None,
    }
    if measured and answers is not None:
        result["fit"] = fit_by_date(sittings, answers.by_date)
        result["answers"] = {"blank": answers.blank, "marked_not_reviewed": answers.marked_not_reviewed,
                             "units_left_out_of_weights": len(pc.rows) - len(reviewed.rows),
                             "reviewed_units": units_dev}
    return result


# --- Report -------------------------------------------------------------------------------

def _h(x: float) -> str:
    return f"{x:,.1f}"


def format_report(r: Mapping[str, Any]) -> str:
    pk, proj = r["packet"], r["projection"]
    L: list[str] = ["# Review load per full run of four workers", ""]
    L.append(f"Basis: {r['basis']}.")
    L.append("")
    L.append("## The dev packet")
    L.append("")
    L.append(f"Packet {pk['run_id']}, whole-narrative scope {pk['scope']}, {pk['worker_runs']} worker run(s), "
             f"{int(pk['cases'])} cases with units (dev cases given: standard {r['dev_cases']['standard']}, "
             f"hard {r['dev_cases']['hard']}).")
    L.append("")
    L.append("| Unit kind | Standard | Hard | All |")
    L.append("|---|---:|---:|---:|")
    names = (("finding", "Finding units"), ("finding_groups", "Flagged sentences (finding units grouped)"),
             ("routed", "Narrative units, routed or FM-coded"),
             ("only", "Narrative units, narrative-only (D-J)"), ("audit", "Audit units"), ("cases", "Cases"),
             ("cases_routed", "Cases with a route"))
    for key, label in names:
        st, hd = pk["by_stratum"]["standard"][key], pk["by_stratum"]["hard"][key]
        L.append(f"| {label} | {st:.0f} | {hd:.0f} | {st + hd:.0f} |")
    L.append("")
    L.append(f"{pk['findings']} findings sit in {pk['finding_units']:.0f} finding units on "
             f"{pk['finding_groups']:.0f} flagged sentences (narrative unit and routed sentences, as packet.csv "
             "writes them; a unit that routes no sentence is its own). Mean words per narrative: "
             f"{pk['mean_words_routed']:.1f} routed, {pk['mean_words_only']:.1f} narrative-only.")
    L.append("")
    L.append("## Rates")
    L.append("")
    if r["measured_hours"] is not None:
        L.append(f"Measured: {r['sittings']} sitting(s), {r['measured_hours']:.2f} hours in all. "
                 f"The D-G model gives {r['dev_model_hours']['low']:.2f} hours (low) to "
                 f"{r['dev_model_hours']['high']:.2f} hours (high) for this packet, so the measured time is "
                 f"{r['measured_to_model']['low']:.2f} times the low model and "
                 f"{r['measured_to_model']['high']:.2f} times the high model.")
        L.append("")
        L.append(f"Blended: {r['blended_minutes_per_unit']:.2f} minutes per unit over every reviewed finding, "
                 "narrative and audit unit, case facts included.")
        ans = r.get("answers")
        if ans is not None:
            L.append("")
            L.append(f"Answers file: {ans['marked_not_reviewed']} finding row(s) marked not reviewed and "
                     f"{ans['blank']} row(s) left blank are not reviewed work. They are left out of the fit by date "
                     f"and out of the dev weights the measured time is shared over ({ans['reviewed_units']:.0f} "
                     "reviewed units). They stay in the projection, since a full run reviews every unit.")
        else:
            L.append("")
            L.append("No answers file: every packet unit is taken as reviewed in the sittings' time.")
    else:
        L.append(f"No sittings file: no time is measured yet. The D-G model gives "
                 f"{r['dev_model_hours']['low']:.2f} hours (low) to {r['dev_model_hours']['high']:.2f} hours "
                 "(high) for this packet.")
    L.append("")
    L.append("| Minutes per unit | Low split | High split |")
    L.append("|---|---:|---:|")
    labels = (("finding", "Finding unit (verdict; mean of sentence and further-key rates)"),
              ("narrative_routed", "Narrative unit, routed (whole read)"),
              ("narrative_only", "Narrative unit, narrative-only (D-J)"), ("audit", "Audit unit"),
              ("case", "Case facts, per case with a route"))
    for key, label in labels:
        L.append(f"| {label} | {r['minutes_per_unit']['low'][key]:.2f} | {r['minutes_per_unit']['high'][key]:.2f} |")
    L.append("")
    if r["measured_hours"] is not None:
        L.append("The measured total is shared out by the D-G weights; the two splits add up to the same measured "
                 "time. Per-kind minutes are apportioned, not timed one by one.")
    else:
        L.append("These are the section 5 assumptions on this packet, not measured.")
    fit = r.get("fit")
    if fit is not None:
        L.append("")
        if fit["allowed"]:
            per = ", ".join(f"{k} {v:.2f}" for k, v in fit["minutes_per_unit"].items())
            L.append(f"Fit by date ({fit['dates']} dates): minutes per unit {per}; root mean square residual "
                     f"{fit['rms_residual_minutes']:.1f} minutes per date. Shown beside the range, not used in it.")
        else:
            L.append(f"Fit by date not possible: {fit['why']}.")
        L.append("")
        L.append("The fit is a weak diagnostic. Answers saved in the review desk before its 2026-11-11 fix may "
                 "hold findings the desk filled automatically when their narrative was answered. Those rows carry "
                 "the day's date but no review time of their own, and the answers file cannot tell them apart from "
                 "findings answered by hand.")
    L.append("")
    L.append("## Scaled to a full run (four workers, 200 cases, 4 epochs)")
    L.append("")
    p = proj["per_worker_units"]
    L.append(f"Per worker run, at the dev worker's routing: {p['finding']:,.0f} finding units, "
             f"{p['routed']:,.0f} routed narratives, {p['only']:,.0f} narrative-only narratives, "
             f"{p['audit']:,.0f} audit rows, {p['cases']:,.0f} cases with units, {p['cases_routed']:,.0f} of them "
             "with a route.")
    L.append("")
    L.append("| Worker | Low hours | High hours |")
    L.append("|---|---:|---:|")
    for w in WORKERS:
        L.append(f"| {w} | {_h(proj['low']['workers'][w]['total_hours'])} "
                 f"| {_h(proj['high']['workers'][w]['total_hours'])} |")
    lo, hi = proj["low"]["total_hours"], proj["high"]["total_hours"]
    L.append(f"| Four workers | {_h(lo)} | {_h(hi)} |")
    L.append("")
    L.append("| Part, four workers | Low hours | High hours |")
    L.append("|---|---:|---:|")
    for part in ("narrative_read", "finding_verdicts", "case_facts", "audit", "second_look"):
        L.append(f"| {part.replace('_', ' ')} | {_h(sum(x['hours'][part] for x in proj['low']['workers'].values()))} "
                 f"| {_h(sum(x['hours'][part] for x in proj['high']['workers'].values()))} |")
    L.append("")
    r_lo, r_hi = sorted((lo, hi))
    dj_lo, dj_hi = sorted((proj["low"]["dj_hours"], proj["high"]["dj_hours"]))
    L.append(f"Re-derived range: about {r_lo:,.0f} to {r_hi:,.0f} hours per full run, against the stated "
             f"{STATED_RANGE_HOURS[0]:.0f} to {STATED_RANGE_HOURS[1]:.0f} (D-G).")
    L.append(f"Beside it (D-J): the narrative-only read of about {proj['low']['dj_epochs_per_full_run']:,.0f} "
             f"narratives per full run, {dj_lo:,.1f} to {dj_hi:,.1f} hours, "
             f"against the stated {STATED_DJ_HOURS[0]:.0f} to {STATED_DJ_HOURS[1]:.0f}.")
    L.append("")
    L.append("Assumptions: each stratum scaled from its dev cases to 146 standard and 54 hard evaluation cases; "
             "route volume 1.00 for every worker (low) or 1.00, 1.05, 1.25 and 1.25 for Sonnet 5.5, Haiku 5.5, "
             "Sonnet 5 and Opus 5.5 (high); finding verdicts at 30 or 90 s per flagged sentence plus 10 or 20 s "
             "per further key on it (LOAD.md section 3), at this packet's share of further keys; case facts once "
             "per case with a route for all four workers (low) or per worker "
             "(high); audit rows as scaled (low) or at least 70, 200, 70 and 70 (high); second look max(100, 10 "
             "percent) routed narrative units per worker run, each with its findings; narrative-only reads not "
             "scaled by route volume. Not included: transcript reading, breaks, the review tool's code work.")
    return "\n".join(L)


# --- CLI ----------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 is not a contract code
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


class _HelpAction(argparse.Action):
    """-h/--help prints usage and exits 3: argparse's own exit 0 would read as an estimate."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any) -> None:
        super().__init__(option_strings, dest=dest, default=argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        parser.print_help()
        print("REFUSED: help requested; no estimate ran", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def _dev_cases(text: str) -> dict[str, int]:
    out = dict(DEV_CASES_DEFAULT)
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep or key not in EVAL_CASES or not value.isdigit() or int(value) <= 0:
            raise argparse.ArgumentTypeError(f"{text!r} is not standard=N,hard=N")
        out[key] = int(value)
    return out


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="Review load per full run from the timed dev rehearsal (no model call).", add_help=False)
    p.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (no estimate ran)")
    p.add_argument("--packet-dir", required=True, help="the packet folder scripts/review_packet.py wrote")
    p.add_argument("--sittings", help="sittings.csv saved from the review page (columns start,end)")
    p.add_argument("--answers", help="the filled answers file, for the fit by date (needs --sittings)")
    p.add_argument("--dev-cases", type=_dev_cases, default=dict(DEV_CASES_DEFAULT),
                   help="dev cases per stratum the packet was built on (default standard=8,hard=12)")
    p.add_argument("--json", dest="json_out", help="also write the result as JSON to this path")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.answers and not args.sittings:
            raise Refused("--answers needs --sittings")
        pc = read_packet(args.packet_dir)
        pairs, hours = read_sittings(args.sittings) if args.sittings else (None, None)
        answers = read_answers(args.answers) if args.answers else None
        result = derive(pc, pairs, hours, args.dev_cases, answers)
        text = format_report(result)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n",
                                           encoding="utf-8")
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as an estimate
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(text)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
