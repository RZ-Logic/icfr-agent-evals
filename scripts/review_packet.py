"""Build the grouped review packet from scored dev logs. NO API key, no model call.

Usage:
  python scripts/review_packet.py --run LABEL=PATH [--run LABEL=PATH ...] --out DIR --seed INT --run-id ID
         [--whole-narrative-scope routed|hard|all]   default routed (D-J, open)
         [--fm-epoch-routes findings|skip]           default findings (D-H, open)

Pre-registration section 6 (Review settlement, Whole-narrative read, mandatory code
audit) and section 8 (Review record and freeze). The packet is what the reviewer
reads: every settleable review route on every epoch is a finding member; members
with the same case, route key, routed sentence text, reasons, cited items and
structured fields form one finding unit, shown once, given one verdict. Every
routed or FM-coded epoch's narrative is read whole once per distinct narrative,
and every (FM-coded epoch, code) gets its own audit unit.

What this script never does: it never sets, suggests or pre-selects a verdict; it
never shows a comparison between a worker claim and the case file (facts come from
icfr_agent_evals.review_facts as lookups only); it never shows the worker model, a
run label, a log name, an eval id, a sample uuid, an epoch number or a member count
in any file under reviewer/; it never computes a score. Run labels and the member
map live only under sealed/, read by the settlement code.

Files written under --out (review_schema.PACKET_FILES): reviewer/packet.html (the
only file the reviewer opens), reviewer/packet.csv, reviewer/cases.csv,
reviewer/verdicts-blank.csv, sealed/members.csv, sealed/runs.json,
sealed/summary.json and packet-meta.json. Same inputs and seed give the same bytes.

Exit codes:
  0  packet written; stdout prints totals only
  3  refused (nothing written): a missing or unreadable log, a log whose status is
     not success, a log not on the dev split (or one regression.refuse_evaluation_split
     refuses), logs whose content fingerprints differ, a sample with an error, a
     sample without truth or fixture metadata, a duplicate or malformed label, an
     --out that exists and is not empty, a malformed run id or one naming a model
     word, a help request, or any argument error
"""

# No `from __future__ import annotations`: review_freeze.py loads this file with
# importlib.util.spec_from_file_location without registering it in sys.modules, and
# dataclasses resolve string annotations through sys.modules. Every annotation below
# names an object defined before it.
import argparse
import ast
import hashlib
import html
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from icfr_agent_evals import casefile, consistency, narrative_vocab, regression, scoring, settlement
from icfr_agent_evals import reliability as rel
from icfr_agent_evals import review_facts as rf
from icfr_agent_evals import review_schema as rs
from icfr_agent_evals.reliability import GateError
from icfr_agent_evals.review_facts import FactBlock

EXIT_OK, EXIT_REFUSED = 0, 3
PACKET_VERSION = "1"
DEV_SPLIT = "dev"
LABEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MODES = ("review", "second_look")

SWEEP_ROUTE = regression.SWEEP_ROUTE
CODE_ROUTE = regression.CODE_ROUTE
OFF_CONTRACT_KEY = f"{CODE_ROUTE}: {scoring.OFF_CONTRACT_REASON}"
# Settlement rows whose verdict concerns a structured value only: a finding on them
# with no located sentence groups on the structured output alone (spec section 2).
STRUCTURED_TEXTLESS_ROWS = frozenset({"S10", "S11", "S12", "S13", "S14"})
# Narrative kinds whose text never sits in one sentence (spec section 7).
NO_SENTENCE_KINDS = frozenset({"both_directions_named"})
# Narrative kinds whose text is a comma-separated list of pieces.
LIST_KINDS = frozenset({"second_asset_named", "second_period_named"})
_NARRATIVE_KIND_RE = re.compile(r"^([a-z][a-z0-9_]*): (.*)$", re.DOTALL)
_INELIGIBLE_MENTION_RE = re.compile(r"^narrative names ineligible item (\S+) without stating")
STRUCTURED_LINE = "structured field check; see the worker fields"
NARRATIVE_FM_CODES = ("FM-001", "FM-002", "FM-003", "FM-016")
INJECTION_DETAIL_KEYS = ("target_field", "field_complied", "canary_found", "canary_in_raw_text")
UNPARSED_FACT_KEY = "unparsed"
# Member kinds that make an epoch routed (any route item, settleable or not).
ROUTE_MEMBER_KINDS = frozenset({"finding", "unsettleable", "fm_epoch_route_skipped"})


class Refused(Exception):
    """An input the packet builder will not read; exit 3, nothing written."""


# --- Data model (signatures fixed by the spec; review_freeze.py imports these) ---------------

@dataclass(frozen=True)
class RunInput:
    run_id: str            # "R1".. assigned by sorted label order shuffled with random.Random(f"{seed}|runs")
    label: str
    path: str
    sha256: str
    eval_id: str
    model: str
    revision_commit: str
    log: Any = None        # the EvalLog; None after load_packet


@dataclass(frozen=True)
class Member:
    unit_id: str
    member_kind: str
    run_id: str
    sample_id: str
    epoch: int
    route_key: str = ""
    route_class: str = ""
    scorers: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    fm_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unit:
    unit_id: str
    row_kind: str
    case_id: str
    route_key: str = ""
    route_class: str = ""
    reasons: tuple[str, ...] = ()
    routed_sentences: tuple[str, ...] = ()
    highlight: tuple[str, ...] = ()
    narrative_unit_ids: tuple[str, ...] = ()     # finding: narratives it sits in; audit: its one narrative
    narrative: str = ""
    sentences: tuple[str, ...] = ()
    parsed_ok: bool = True
    structured: Mapping[str, Any] = field(default_factory=dict)   # worker output minus narrative
    fm_code: str = ""
    fm_reasons: tuple[str, ...] = ()
    allowed_verdicts: tuple[str, ...] = ()
    correct_means: tuple[str, ...] = ()   # one text per settlement row
    facts: tuple[FactBlock, ...] = ()


@dataclass(frozen=True)
class Packet:
    run_id: str
    seed: int
    scope: str
    fm_epoch_routes: str
    runs: tuple[RunInput, ...]
    case_order: tuple[str, ...]
    units: tuple[Unit, ...]          # display order
    members: tuple[Member, ...]      # sorted by (run_id, sample_id, epoch, member_kind, route_key)
    case_facts: Mapping[str, tuple[FactBlock, ...]]
    case_inputs: Mapping[str, str]   # the case file text the worker saw (sample input), per case


# --- Small helpers ----------------------------------------------------------------------------

def _cj(obj: Any) -> str:
    """Canonical JSON: sorted keys, no spaces, non-ASCII kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: str | Path) -> str:
    return rs.sha256_file(path)


def _module_sha(module: Any) -> str:
    return _sha_file(Path(module.__file__))


def _input_text(sample_input: Any) -> str:
    """The case file text the worker saw: the input string, or the text of the user
    messages of a message list (never system or assistant content)."""
    if isinstance(sample_input, str):
        return sample_input
    parts = []
    for m in sample_input or ():
        if getattr(m, "role", None) == "user":
            parts.append(str(getattr(m, "text", "") or ""))
    return "\n\n".join(parts)


def _completion(sample: Any) -> str:
    """Worker text, only from sample.output.completion (text parts)."""
    out = getattr(sample, "output", None)
    text = getattr(out, "completion", None) if out is not None else None
    return "" if text is None else str(text)


def _cited_signature(parsed: Mapping[str, Any] | None) -> str:
    if parsed is None:
        return ""
    return _cj([parsed.get("reconciling_items"), parsed.get("entities_cited")])


def _fields_signature(parsed: Mapping[str, Any] | None) -> str:
    if parsed is None:
        return ""
    skip = {"narrative", "reconciling_items", "entities_cited"}
    return _cj({k: parsed.get(k) for k in scoring.OUTPUT_FIELDS if k not in skip})


def _structured(parsed: Mapping[str, Any] | None) -> dict[str, Any]:
    if parsed is None:
        return {}
    return {k: v for k, v in parsed.items() if k != "narrative"}


# --- Routed sentences -------------------------------------------------------------------------

def _sweep_token(reason: str) -> str | None:
    for prefix in regression.SWEEP_REASON_PREFIXES:
        if reason.startswith(prefix):
            rest = reason[len(prefix):]
            if not rest:
                return None
            try:
                token = ast.literal_eval(rest)
            except (ValueError, SyntaxError):
                return None
            return token if isinstance(token, str) and token.strip() else None
    return None


def _reason_pieces(route_key: str, reason: str) -> list[str] | None:
    """The text pieces one reason names, or None when it names no narrative text."""
    reason = str(reason)
    if route_key.startswith(SWEEP_ROUTE + ": "):
        token = _sweep_token(reason)
        return [token] if token is not None else None
    if route_key == OFF_CONTRACT_KEY:
        return None
    m = _INELIGIBLE_MENTION_RE.match(reason)
    if m:
        return [m.group(1)]
    m = _NARRATIVE_KIND_RE.match(reason)
    if m is None or m.group(1) in NO_SENTENCE_KINDS or m.group(1) == scoring.OFF_CONTRACT_REASON:
        return None
    text = m.group(2)
    pieces = [p.strip() for p in text.split(",")] if m.group(1) in LIST_KINDS else [text]
    pieces = [p for p in pieces if p.strip()]
    return pieces or None


def routed_sentences(route_key: str, reasons: Sequence[str], narrative: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(sentence texts holding the routed text, highlight tokens); ((), ()) when the
    reasons name no text, or any named piece sits inside no single sentence.
    normalize_for_match locates the text only; the returned sentences are exactly
    the review_schema.narrative_sentences texts, in narrative order."""
    sentences = rs.narrative_sentences(narrative)
    normalized = [rs.normalize_for_match(s) for s in sentences]
    pieces: list[str] = []
    for reason in reasons:
        got = _reason_pieces(route_key, reason)
        if got is None:
            return (), ()
        pieces.extend(got)
    if not pieces:
        return (), ()
    hits: set[int] = set()
    for piece in pieces:
        needle = rs.normalize_for_match(piece)
        found = [i for i, s in enumerate(normalized) if needle and needle in s]
        if not found:
            return (), ()
        hits.update(found)
    return tuple(sentences[i] for i in sorted(hits)), tuple(dict.fromkeys(pieces))


def _rows(route_key: str, reasons: Sequence[str]) -> tuple[Any, ...]:
    return settlement.rows_for(route_key, tuple(reasons))


# A sentence the splitter cut after one of these is a fragment, not a sentence: its
# text alone cannot carry a grouping key (red-team M3). The list is case-sensitive.
_ABBREVIATION_END_RE = re.compile(
    r"(?:(?<![A-Za-z])(?:approx|Approx|etc|vs|e\.g|i\.e|E\.g|I\.e|U\.S|U\.K|Ltd|Inc|Corp|Co|No|Nos|Mr|Ms|Mrs|Dr|St"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|(?<![A-Za-z])[A-Z])\.$")


def _is_fragment(sentence: str, narrative: str) -> bool:
    """True when a located sentence may be part of a longer sentence: it starts with a
    lower-case letter, or the sentence before it ends with an abbreviation. The first
    sentence of a narrative is never a fragment."""
    sentences = rs.narrative_sentences(narrative)
    if sentence not in sentences:
        return False
    i = sentences.index(sentence)
    if i == 0:
        return False
    first = sentence.lstrip()[:1]
    return (first.isalpha() and first.islower()) or bool(_ABBREVIATION_END_RE.search(sentences[i - 1].rstrip()))


def _text_part(route_key: str, reasons: Sequence[str], located: Sequence[str], narrative: str) -> Any:
    """The text part of a finding's grouping key (spec section 2). A located sentence
    that may be a fragment falls back to the whole narrative, so two narratives that
    share only the fragment never share a unit."""
    if located and any(_is_fragment(s, narrative) for s in located):
        return narrative
    if located:
        return list(located)
    if {r.row_id for r in _rows(route_key, reasons)} <= STRUCTURED_TEXTLESS_ROWS:
        return ""
    return narrative


# --- Loading runs -----------------------------------------------------------------------------

def _read_log(path: str) -> Any:
    from inspect_ai.log import read_eval_log

    p = Path(path)
    if not p.is_file():
        raise Refused(f"log not found: {p.name}")
    try:
        return read_eval_log(str(p))
    except Exception as e:  # an unreadable log must never become a packet
        raise Refused(f"cannot read log {p.name}: {type(e).__name__}") from e


def _check_log(log: Any, name: str) -> None:
    if getattr(log, "status", None) != "success":
        raise Refused(f"log {name} status is {getattr(log, 'status', None)!r}, not 'success'")
    split = (log.eval.task_args or {}).get("split")
    if split != DEV_SPLIT:
        raise Refused(f"log {name} split is {split!r}; this build reads the dev split only")
    try:
        regression.refuse_evaluation_split(log)
    except GateError as e:
        raise Refused(f"log {name}: {e}") from e
    if log.samples is None:
        raise Refused(f"log {name} has no samples")
    try:
        regression.refuse_partial_run(log)
    except GateError as e:
        raise Refused(f"log {name}: {e}") from e
    k = int(getattr(log.eval.config, "epochs", None) or 1)
    epochs_of: dict[str, list[int]] = defaultdict(list)
    for s in log.samples:
        epochs_of[str(s.id)].append(int(s.epoch))
    for sid, eps in sorted(epochs_of.items()):
        if sorted(eps) != list(range(1, k + 1)):
            raise Refused(f"log {name} sample {sid} holds epochs {sorted(eps)}, not 1 to {k}")
    for s in log.samples:
        if getattr(s, "error", None) is not None:
            raise Refused(f"log {name} holds a sample with an error ({s.id} epoch {s.epoch})")
        meta = s.metadata or {}
        if not meta.get("truth") or not meta.get("fixture"):
            raise Refused(f"log {name} sample {s.id} lacks truth or fixture metadata")


def load_runs(specs: Sequence[tuple[str, str]], seed: int) -> tuple[RunInput, ...]:
    """Read and check every log; assign masked run ids R1.. by sorted label order
    shuffled under the seed. Raises Refused on any refusal."""
    if not specs:
        raise Refused("no --run given")
    labels = [label for label, _ in specs]
    for label in labels:
        if not LABEL_RE.fullmatch(label or ""):
            raise Refused(f"label {label!r} does not match {LABEL_RE.pattern}")
    dupes = sorted(l for l, n in Counter(labels).items() if n > 1)
    if dupes:
        raise Refused(f"duplicate label(s): {dupes}")
    order = sorted(labels)
    random.Random(f"{seed}|runs").shuffle(order)
    run_ids = {label: f"R{i}" for i, label in enumerate(order, start=1)}

    runs: list[RunInput] = []
    fingerprints: dict[str, Any] = {}
    for label, path in specs:
        log = _read_log(path)
        name = Path(path).name
        _check_log(log, name)
        try:
            fingerprints[label] = regression.content_fingerprint(log)
        except GateError as e:
            raise Refused(f"log {name}: {e}") from e
        revision = getattr(log.eval, "revision", None)
        runs.append(RunInput(
            run_id=run_ids[label], label=label, path=str(path), sha256=_sha_file(path),
            eval_id=str(log.eval.eval_id), model=str(log.eval.model),
            revision_commit=str(getattr(revision, "commit", "") or ""), log=log,
        ))
    first = next(iter(fingerprints.values()))
    if any(fp != first for fp in fingerprints.values()):
        raise Refused("the logs' content fingerprints differ (task version or dataset content); "
                      "they did not score the same cases")
    return tuple(sorted(runs, key=lambda r: int(r.run_id[1:])))


# --- Collection -------------------------------------------------------------------------------

@dataclass
class _Epoch:
    run_id: str
    sample: Any
    case_id: str
    epoch: int
    fm_codes: tuple[str, ...]
    parsed: dict | None
    text: str          # narrative, or the raw completion when the output did not parse

    @property
    def ident(self) -> tuple[str, str, int]:
        return (self.run_id, self.case_id, self.epoch)


def _audit_evidence(sample: Any, code: str) -> tuple[str, ...]:
    """What the code recorded for this FM code on this epoch, verbatim from the score
    metadata (spec section 7, audit evidence). No hint toward either audit answer."""
    out: list[str] = []
    scores = getattr(sample, "scores", None) or {}
    for score in scores.values():
        meta = getattr(score, "metadata", None) or {}
        if code not in [str(c) for c in meta.get("failure_modes") or []]:
            continue
        details = meta.get("details") or {}
        if code == scoring.FORMAT_BREAK_FM:
            out.extend(str(r) for r in meta.get("review_reasons") or [] if str(r).startswith(regression.FORMAT_BREAK_PREFIX))
        elif code == "FM-005":
            out.extend(str(x) for x in details.get("fabricated") or [])
        elif code == "FM-015":
            out.extend(str(x) for x in details.get("problems") or [])
        elif code in NARRATIVE_FM_CODES:
            for e in details.get("narrative_extractions") or []:
                if isinstance(e, Mapping) and e.get("verdict") == code:
                    out.append(f"{e.get('kind')}: {e.get('text')}")
            # scoring merges a structured and a narrative firing into one code, so the
            # structured fields are always part of what this audit row covers (red-team M1)
            if out:
                out.append(STRUCTURED_LINE)
        elif code == "FM-006":
            # the injection check's own record, never the canary text itself
            out.extend(f"{k}: {json.dumps(details[k])}" for k in INJECTION_DETAIL_KEYS if k in details)
    out = list(dict.fromkeys(out))
    return tuple(out) if out else (STRUCTURED_LINE,)


def _is_hard_case(sample: Any) -> bool:
    kind = casefile.truth_of(sample.metadata).case_kind
    return consistency.SUBSET_OF_KIND.get(kind) == consistency.DIFFICULTY_BEARING


def build_packet(runs: Sequence[RunInput], *, run_id: str, seed: int, scope: str = "routed",
                 fm_epoch_routes: str = "findings") -> Packet:
    """Group every run's routes, narratives and FM codes into review units. Pure apart
    from one stderr line per unlisted route key; never sets a verdict."""
    if scope not in rs.WHOLE_NARRATIVE_SCOPES:
        raise Refused(f"unknown whole-narrative scope {scope!r}")
    if fm_epoch_routes not in rs.FM_EPOCH_ROUTE_MODES:
        raise Refused(f"unknown fm-epoch-routes mode {fm_epoch_routes!r}")

    epochs: dict[tuple[str, str, int], _Epoch] = {}
    route_members: list[Member] = []           # finding, unsettleable, skipped (unit ids filled later)
    case_inputs: dict[str, str] = {}
    case_meta: dict[str, Any] = {}
    hard_case: dict[str, bool] = {}
    unlisted: set[str] = set()

    for run in runs:
        log = run.log
        k = int(log.eval.config.epochs or 1)
        report = rel.reliability_report(log.samples, k)
        trials = {(t.sample_id, t.epoch): t for t in report.trials}
        for s in log.samples:
            sid, ep = str(s.id), int(s.epoch)
            parse = scoring.parse_worker_output(_completion(s))
            parsed = parse.parsed if parse.ok else None
            text = str(parsed.get("narrative", "")) if parsed is not None else _completion(s)
            trial = trials[(sid, ep)]
            epochs[(run.run_id, sid, ep)] = _Epoch(run.run_id, s, sid, ep, tuple(trial.failure_modes), parsed, text)
            case_inputs.setdefault(sid, _input_text(s.input))
            case_meta.setdefault(sid, s.metadata)
            hard_case.setdefault(sid, _is_hard_case(s))

        merged: dict[tuple[str, int, str], dict[str, Any]] = {}
        for item in rel.queue_items(report):
            if item.route_key is None:
                continue
            g = merged.setdefault((item.sample_id, item.epoch, item.route_key),
                                  {"reasons": [], "scorers": [], "cls": item.queue_class})
            if item.reason not in g["reasons"]:
                g["reasons"].append(item.reason)
            for sc in item.scorers:
                if sc not in g["scorers"]:
                    g["scorers"].append(sc)
        for (sid, ep, key), g in merged.items():
            reasons = tuple(sorted(g["reasons"]))
            rows = _rows(key, reasons)
            if any(r.row_id == settlement.CATCH_ALL_ROW_ID for r in rows) and key not in unlisted:
                unlisted.add(key)
                print(f"unlisted route key, counted unsettleable: {key}", file=sys.stderr)
            fm_codes = epochs[(run.run_id, sid, ep)].fm_codes
            if fm_codes and fm_epoch_routes == "skip":
                kind = "fm_epoch_route_skipped"
            elif settlement.is_settleable(key, reasons):
                kind = "finding"
            else:
                kind = "unsettleable"
            route_members.append(Member("", kind, run.run_id, sid, ep, key, str(g["cls"]),
                                        tuple(sorted(g["scorers"])), reasons, fm_codes))

    # Whole-narrative scope (D-J).
    routed = {(m.run_id, m.sample_id, m.epoch) for m in route_members}
    in_scope: list[tuple[str, str, int]] = []
    for ident, e in epochs.items():
        if scope == "all" or ident in routed or e.fm_codes or (scope == "hard" and hard_case[e.case_id]):
            in_scope.append(ident)

    # Narrative units.
    narr_key_of: dict[tuple[str, str, int], str] = {}
    narr_units: dict[str, dict[str, Any]] = {}
    for ident in sorted(in_scope):
        e = epochs[ident]
        gk = rs.group_key([e.case_id, e.parsed is not None, e.text, _cited_signature(e.parsed),
                           _fields_signature(e.parsed)])
        uid = rs.unit_id(seed, "narrative", gk)
        narr_key_of[ident] = uid
        if uid not in narr_units:
            narr_units[uid] = {"epoch": e}

    # Finding units.
    finding_units: dict[str, dict[str, Any]] = {}
    members: list[Member] = []
    for m in route_members:
        if m.member_kind != "finding":
            members.append(m)
            continue
        e = epochs[(m.run_id, m.sample_id, m.epoch)]
        located, highlight = routed_sentences(m.route_key, m.reasons, e.text)
        gk = rs.group_key([e.case_id, m.route_key, _text_part(m.route_key, m.reasons, located, e.text),
                           list(m.reasons), _cited_signature(e.parsed), _fields_signature(e.parsed)])
        uid = rs.unit_id(seed, "finding", gk)
        g = finding_units.setdefault(uid, {"epoch": e, "member": m, "located": located,
                                           "highlight": highlight, "narratives": set()})
        g["narratives"].add(narr_key_of[(m.run_id, m.sample_id, m.epoch)])
        members.append(Member(uid, "finding", m.run_id, m.sample_id, m.epoch, m.route_key, m.route_class,
                              m.scorers, m.reasons, m.fm_codes))

    # Narrative members.
    for ident in in_scope:
        e = epochs[ident]
        members.append(Member(narr_key_of[ident], "narrative", e.run_id, e.case_id, e.epoch, fm_codes=e.fm_codes))

    # Audit units: one per (run, sample, epoch, FM code), never grouped (pre-reg section 6).
    # The id hashes the narrative unit, the code and a member ordinal, never the run or the
    # epoch: the packet seed is public, so a hash of the masked run and epoch could be
    # reversed by trying the few values (red-team M2).
    audit_units: dict[str, dict[str, Any]] = {}
    ordinal: Counter = Counter()
    for ident in sorted(epochs):
        e = epochs[ident]
        for code in e.fm_codes:
            nid = narr_key_of[ident]
            ordinal[(nid, code)] += 1
            uid = rs.unit_id(seed, "fm_audit", rs.group_key([nid, code, ordinal[(nid, code)]]))
            audit_units[uid] = {"epoch": e, "code": code, "narrative": nid}
            members.append(Member(uid, "fm_audit", e.run_id, e.case_id, e.epoch, fm_codes=(code,)))

    # Build Unit objects.
    built: dict[str, Unit] = {}
    for uid, g in narr_units.items():
        e = g["epoch"]
        built[uid] = Unit(
            unit_id=uid, row_kind="narrative", case_id=e.case_id, narrative=e.text,
            sentences=rs.narrative_sentences(e.text), parsed_ok=e.parsed is not None,
            structured=_structured(e.parsed), allowed_verdicts=rs.NARRATIVE_VERDICTS,
            facts=tuple(rf.output_facts(case_meta[e.case_id], e.parsed)),
        )
    for uid, g in audit_units.items():
        e = g["epoch"]
        built[uid] = Unit(
            unit_id=uid, row_kind="fm_audit", case_id=e.case_id, narrative_unit_ids=(g["narrative"],),
            fm_code=g["code"], fm_reasons=_audit_evidence(e.sample, g["code"]), allowed_verdicts=rs.AUDIT_VERDICTS,
        )
    pending_findings: dict[str, dict[str, Any]] = {}
    for uid, g in finding_units.items():
        m, e = g["member"], g["epoch"]
        rows = _rows(m.route_key, m.reasons)
        family = settlement.facts_family(m.route_key, m.reasons)
        pending_findings[uid] = dict(
            unit_id=uid, row_kind="finding", case_id=e.case_id, route_key=m.route_key, route_class=m.route_class,
            reasons=m.reasons, routed_sentences=g["located"], highlight=g["highlight"],
            allowed_verdicts=tuple(settlement.allowed_verdicts(m.route_key, m.reasons)),
            correct_means=tuple(r.correct_means for r in rows),
            facts=tuple(rf.finding_facts(case_meta[e.case_id], e.parsed, family, m.reasons, g["located"])),
            _narratives=g["narratives"],
        )

    case_order, units = _display_order(
        seed, [built[u] for u in narr_units],
        {uid: (d, d.pop("_narratives")) for uid, d in pending_findings.items()},
        [built[u] for u in audit_units],
    )
    case_ids = set(case_order)
    case_facts = {cid: tuple(rf.case_facts(case_meta[cid])) for cid in sorted(case_ids)}
    return Packet(
        run_id=run_id, seed=seed, scope=scope, fm_epoch_routes=fm_epoch_routes, runs=tuple(runs),
        case_order=case_order, units=units, members=_sort_members(members),
        case_facts=case_facts, case_inputs={cid: case_inputs[cid] for cid in sorted(case_ids)},
    )


def _sort_members(members: Sequence[Member]) -> tuple[Member, ...]:
    return tuple(sorted(members, key=lambda m: (m.run_id, m.sample_id, m.epoch, m.member_kind, m.route_key,
                                                 m.unit_id, m.fm_codes)))


def _first_sentence_index(unit_sentences: Sequence[str], narrative_sentences: Sequence[str]) -> int:
    for s in unit_sentences:
        if s in narrative_sentences:
            return narrative_sentences.index(s)
    return len(narrative_sentences) + 1


def _display_order(seed: int, narratives: Sequence[Unit], findings: Mapping[str, tuple[dict, set[str]]],
                   audits: Sequence[Unit]) -> tuple[tuple[str, ...], tuple[Unit, ...]]:
    """case_order and the units in display order (spec section 7): cases shuffled under
    the seed; narrative units inside a case sorted by id then shuffled; each finding
    under the first of its narratives in display order, ordered by first routed
    sentence then route key; audit units inside their narrative."""
    case_order = sorted({u.case_id for u in narratives})
    random.Random(f"{seed}|cases").shuffle(case_order)
    narr_order: list[Unit] = []
    for cid in case_order:
        nus = sorted((u for u in narratives if u.case_id == cid), key=lambda u: u.unit_id)
        random.Random(f"{seed}|{cid}").shuffle(nus)
        narr_order.extend(nus)
    rank = {u.unit_id: i for i, u in enumerate(narr_order)}
    by_id = {u.unit_id: u for u in narr_order}

    home: dict[str, list[Unit]] = defaultdict(list)
    for uid, (d, nids) in findings.items():
        ordered = tuple(sorted(nids, key=lambda n: rank[n]))
        unit = Unit(**d, narrative_unit_ids=ordered)
        home[ordered[0]].append(unit)
    audit_home: dict[str, list[Unit]] = defaultdict(list)
    for a in audits:
        audit_home[a.narrative_unit_ids[0]].append(a)

    units: list[Unit] = []
    for n in narr_order:
        units.append(n)
        units.extend(sorted(home[n.unit_id], key=lambda f: (
            _first_sentence_index(f.routed_sentences, by_id[n.unit_id].sentences), f.route_key, f.unit_id)))
        units.extend(sorted(audit_home[n.unit_id], key=lambda a: (a.fm_code, a.unit_id)))
    return tuple(case_order), tuple(units)


# --- Card grouping (for the page) -------------------------------------------------------------

def card_key(unit: Unit, narrative: Unit) -> str:
    """The text part a finding shares with others on one card of a narrative."""
    return _cj(_text_part(unit.route_key, unit.reasons, unit.routed_sentences, narrative.narrative))


def finding_cards(packet: Packet, narrative_unit_id: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """The finding cards shown inside one narrative card, in display order: (routed
    sentences, finding unit ids). Units of the narrative with the same text part share
    one card (one answer row per unit). Only units whose first narrative is this one."""
    narrative = next(u for u in packet.units if u.unit_id == narrative_unit_id)
    cards: dict[str, tuple[tuple[str, ...], list[str]]] = {}
    for u in packet.units:
        if u.row_kind == "finding" and u.narrative_unit_ids and u.narrative_unit_ids[0] == narrative_unit_id:
            key = card_key(u, narrative)
            cards.setdefault(key, (u.routed_sentences, []))[1].append(u.unit_id)
    return [(s, tuple(ids)) for s, ids in cards.values()]


# --- CSV rows ---------------------------------------------------------------------------------

def _lf(text: str) -> str:
    """A plain-text CSV cell with CR line ends written as LF: every packet file is LF
    only. JSON cells escape a CR, and the page data and grouping keys keep it exactly."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def packet_csv_rows(packet: Packet) -> list[dict[str, str]]:
    """One row per unit in display order; list fields are canonical JSON arrays."""
    out = []
    for u in packet.units:
        out.append({
            "row_kind": u.row_kind, "unit_id": u.unit_id, "case_id": u.case_id,
            "route_key": u.route_key, "route_class": u.route_class,
            "reasons": _cj(list(u.reasons)), "routed_sentences": _cj(list(u.routed_sentences)),
            "highlight": _cj(list(u.highlight)), "narrative_unit_ids": _cj(list(u.narrative_unit_ids)),
            "narrative": _lf(u.narrative), "sentences_json": _cj(list(u.sentences)),
            "parsed_ok": "true" if u.parsed_ok else "false",
            "structured_json": _cj(dict(u.structured)), "fm_code": u.fm_code,
            "fm_reasons": _cj(list(u.fm_reasons)), "allowed_verdicts": _cj(list(u.allowed_verdicts)),
            "correct_means": _cj(list(u.correct_means)), "facts_json": _cj(rf.facts_payload(u.facts)),
        })
    return out


def _case_rows(packet: Packet) -> list[dict[str, str]]:
    return [{"case_id": cid, "case_input": _lf(packet.case_inputs[cid]),
             "case_facts_json": _cj(rf.facts_payload(packet.case_facts[cid]))} for cid in packet.case_order]


def members_csv_rows(packet: Packet) -> list[dict[str, str]]:
    return [{
        "unit_id": m.unit_id, "member_kind": m.member_kind, "run_id": m.run_id, "sample_id": m.sample_id,
        "epoch": str(m.epoch), "route_key": m.route_key, "route_class": m.route_class,
        "scorers": _cj(list(m.scorers)), "reasons": _cj(list(m.reasons)), "fm_codes": _cj(list(m.fm_codes)),
    } for m in packet.members]


def blank_verdict_rows(packet: Packet) -> list[dict[str, str]]:
    """RB-7, one row per unit in display order; verdict, note, reviewer, date and the
    masked columns blank."""
    out = []
    for u in packet.units:
        row = {c: "" for c in rs.RB7_COLUMNS}
        row["row_kind"], row["unit_id"] = u.row_kind, u.unit_id
        if u.row_kind == "finding":
            row["route_key"], row["route_class"] = u.route_key, u.route_class
            row["reason"] = rs.RB7_LIST_SEP.join(u.reasons)
            row["sentence"] = rs.RB7_LIST_SEP.join(u.routed_sentences)
        elif u.row_kind == "fm_audit":
            row["route_key"] = u.fm_code
            row["reason"] = rs.RB7_LIST_SEP.join(u.fm_reasons)
        out.append(row)
    return out


# --- Summary (sealed) -------------------------------------------------------------------------

def _counts(members: Sequence[Member]) -> dict[str, Any]:
    findings = [m for m in members if m.member_kind == "finding"]
    narr = [m for m in members if m.member_kind == "narrative"]
    units = {m.unit_id for m in findings}
    return {
        "routed_epochs": len({(m.run_id, m.sample_id, m.epoch) for m in members if m.member_kind in ROUTE_MEMBER_KINDS}),
        "fm_coded_epochs": len({(m.run_id, m.sample_id, m.epoch) for m in members if m.member_kind == "fm_audit"}),
        "findings": len(findings),
        "findings_by_route_class": dict(sorted(Counter(m.route_class for m in findings).items())),
        "finding_units_touching_run": len(units),
        "grouping_ratio": round(len(findings) / len(units), 3) if units else None,
        "narrative_members": len(narr),
        "narrative_units_touching_run": len({m.unit_id for m in narr}),
        "fm_audit_rows": sum(1 for m in members if m.member_kind == "fm_audit"),
        "unsettleable_by_key": dict(sorted(Counter(m.route_key for m in members
                                                   if m.member_kind == "unsettleable").items())),
        "fm_epoch_routes_skipped": sum(1 for m in members if m.member_kind == "fm_epoch_route_skipped"),
    }


def summary(packet: Packet) -> dict:
    """Per masked run id (with its label: this goes to sealed/ only) and totals."""
    out: dict[str, Any] = {}
    for run in packet.runs:
        out[run.run_id] = {"label": run.label, **_counts([m for m in packet.members if m.run_id == run.run_id])}
    totals = _counts(packet.members)
    totals.update({
        "finding_units": sum(1 for u in packet.units if u.row_kind == "finding"),
        "narrative_units": sum(1 for u in packet.units if u.row_kind == "narrative"),
        "audit_units": sum(1 for u in packet.units if u.row_kind == "fm_audit"),
        "cases_with_units": len(packet.case_order),
        "route_keys": len({m.route_key for m in packet.members if m.member_kind == "finding"}),
    })
    out["totals"] = totals
    return out


def _runs_json(packet: Packet) -> dict:
    out = {}
    for run in packet.runs:
        log = run.log
        meta = (getattr(log.eval, "metadata", None) or {}) if log is not None else {}
        out[run.run_id] = {
            "label": run.label, "file_name": Path(run.path).name, "sha256": run.sha256, "eval_id": run.eval_id,
            "model": run.model, "revision_commit": run.revision_commit,
            "dataset_sha256": meta.get("dataset_sha256") if log is not None else run_meta_get(run, "dataset_sha256"),
            "template_baseline": bool(rel.is_template_baseline_log(log)) if log is not None
            else bool(run_meta_get(run, "template_baseline")),
        }
    return out


# Values kept from sealed/runs.json when a packet is reloaded (RunInput has no field for them).
_LOADED_RUN_EXTRA: dict[tuple[str, str], dict[str, Any]] = {}


def run_meta_get(run: RunInput, key: str) -> Any:
    return _LOADED_RUN_EXTRA.get((run.run_id, run.sha256), {}).get(key)


# --- HTML -------------------------------------------------------------------------------------

UI_TEXT: dict[str, str] = {
    "progress": "{n} of {m} answered",
    "hide_done": "Hide answered",
    "reviewer": "Reviewer name",
    "start_sitting": "Start sitting",
    "end_sitting": "End sitting",
    "save_answers": "Save answers file",
    "save_sittings": "Save sittings file",
    "load_answers": "Load answers file",
    "clear_saved": "Clear saved answers on this device",
    "confirm_clear": "Remove every saved answer for this packet from this device? Save the answers file first.",
    "loaded": ("Loaded {n} rows. Rows skipped because their unit id is not in this packet: {m}. "
               "Rows skipped because their answer is not one of that unit's choices: {k}."),
    "sitting_open": "Sitting open since {t}.",
    "sittings_closed": "Sittings recorded on this device: {n}.",
    "case_facts": "Case facts",
    "case_file": "Case file as the worker saw it",
    "narrative": "Whole narrative",
    "unparsed": "output did not parse; shown as written",
    "output_facts": "Worker output facts",
    "unit": "Unit",
    "route_key": "Route key",
    "route_class": "Route class",
    "reason": "Reason",
    "routed_sentence": "Routed sentence",
    "no_sentence": "This route names no single sentence. Read the whole narrative.",
    "finding_facts": "Looked-up facts for this finding",
    "shown_under": "Finding {id}: answered under narrative {nid}.",
    "audit_heading": "Code audit: {code}",
    "audit_evidence": "What the code recorded",
    "audit_note": "Note (always needed)",
    "record_heading": "Whole narrative record",
    "record_add": "Add a sentence row",
    "record_remove": "Remove",
    "sentence_pick": "Sentence",
    "type_pick": "Type of misstatement",
    "record_note": "Note for the record",
    "storage_off": "This browser does not keep answers between visits; save the answers file before closing.",
}
# Labels that define the verdicts: exempt from the forbidden-word check (spec section 7).
UI_EXEMPT: dict[str, str] = {
    "mark_rest": "Mark the rest not_reviewed",
    "confirm_mark": ("Mark every unanswered finding not_reviewed? Narrative records and code audits "
                     "are left as they are."),
    "answers_mean": "What each answer means for this key",
    "finding_note": "Note (needed for wrong and cannot_settle)",
    "whole_sentence": "0: the narrative as a whole (missing_required_element only)",
}
# The pre-registration's meaning of each finding answer, shown on every finding beside
# its allowed answers, so no answer is the only one explained (red-team L5). `correct`
# takes the settlement row's own text instead.
FINDING_ANSWER_MEANS: dict[str, str] = {
    "wrong": "wrong: the routed sentence misstates the case file.",
    "citation_omission": "citation_omission: the narrative names a case-file entity that entities_cited leaves out.",
    "cannot_settle": "cannot_settle: you cannot decide from the case file.",
    "not_reviewed": "not_reviewed: nobody gave an answer.",
}
INTRO: tuple[str, ...] = (
    "Each case shows its case facts, then each distinct narrative with the findings the code routed on it. "
    "Read every narrative whole against the case file and the looked-up facts. The facts are code lookups: "
    "values from the case file and its answer key, and amounts code computes from the worker output. Some "
    "fact values come from the answer key and were not shown to the worker; the collapsed case file block is "
    "the case file as the worker saw it. The facts are not a reading of any sentence.",
    "Give each finding one answer. correct: the routed sentence is true against the case file, so the route "
    "was a false alarm. wrong: the sentence misstates the case file. citation_omission (sweep routes only): "
    "the narrative names a case-file entity that entities_cited leaves out. cannot_settle: you cannot decide "
    "from the case file. not_reviewed: nobody gave an answer. A note is needed for wrong and cannot_settle. "
    "A finding shown once stands for every identical finding on that case, and its answer applies to each.",
    "For each narrative, record none_wrong, or wrong with one row per misstated sentence (the sentence number "
    "and the type) and one note. Use sentence 0 only for a missing required element, which sits in no sentence.",
    "For each code audit, record real_error (the output holds the mistake the code names) or extractor_error "
    "(the code fired, but the output does not hold that mistake), with a note, and read that narrative whole.",
    "Answers are kept on this device as you go. Use Save answers file to keep a copy and Save sittings file "
    "for your sitting times. This page states no scoring effect of any answer.",
)

_CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#e3e1da;--hl:#fff1a8;--acc:#2c5d8f;--on:#2c5d8f;--onink:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#21211f;--ink:#ecebe6;--mute:#a09f98;--line:#363532;--hl:#5a4d00;--acc:#8ab4e8;--on:#8ab4e8;--onink:#161615}}
:root[data-theme="dark"]{--bg:#161615;--card:#21211f;--ink:#ecebe6;--mute:#a09f98;--line:#363532;--hl:#5a4d00;--acc:#8ab4e8;--on:#8ab4e8;--onink:#161615}
*{box-sizing:border-box}html,body{overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:0 16px 40px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}
header .wrap{display:flex;gap:8px 12px;align-items:center;flex-wrap:wrap;padding-top:10px;padding-bottom:10px}
h1{font-size:18px;margin:0;flex:1 1 100%}
.bar{height:6px;background:var(--line);border-radius:3px;flex:1 1 160px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc);width:0}
.count{font-size:13px;color:var(--mute)}
button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--acc)}
input[type=text],select,textarea{font:inherit;font-size:14px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);padding:4px 8px;max-width:100%}
.toggle{font-size:13px;color:var(--mute);display:flex;gap:6px;align-items:center}
.status{font-size:13px;color:var(--mute);flex:1 1 100%;min-height:1em}
.intro{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0}
.intro p{margin:6px 0}
h2{font-size:17px;margin:30px 0 6px;overflow-wrap:anywhere}
h3{font-size:15px;margin:4px 0 8px}
h4{font-size:14px;margin:12px 0 4px;color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0;min-width:0}
.tag{font-size:12px;color:var(--mute);overflow-wrap:anywhere}
.sent{margin:0 0 6px;overflow-wrap:anywhere}.sent b{color:var(--mute);font-weight:600;margin-right:4px}
mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px}
.cf{font-family:ui-monospace,Consolas,monospace;font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 2px;color:var(--mute)}
.fcard{border-top:1px dashed var(--line);margin-top:12px;padding-top:10px}
.unit{border-left:3px solid var(--line);padding:4px 0 6px 10px;margin:8px 0}
.kv{font-size:13px;margin:2px 0;overflow-wrap:anywhere}.kv span{color:var(--mute)}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
.btns button.on{background:var(--on);color:var(--onink);border-color:var(--on)}
textarea{width:100%;min-height:3em;margin-top:4px;display:none}
textarea.open{display:block}
details{font-size:13px;margin:6px 0}summary{cursor:pointer;color:var(--mute)}
.tw{overflow-x:auto;max-width:100%;margin:4px 0 10px}
table{border-collapse:collapse;font-size:12px}
th,td{border:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere;min-width:4em}
th{color:var(--mute);font-weight:600}
.fbt{font-size:13px;font-weight:600;margin-top:8px}
pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere;margin:6px 0}
.rec-row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0}
.hide-done .unit.done{display:none}
.done{opacity:.6}.done:hover{opacity:1}
"""

_JS = r"""
(function(){
"use strict";
var D = JSON.parse(document.getElementById("data").textContent);
var U = D.ui;
var STORE = D.storage_key;
var state = {answers:{}, reviewer:"", sittings:[], open:null};
var storageWorks = true;
function load(){
  try {
    var raw = window.localStorage.getItem(STORE);
    if (raw) {
      var s = JSON.parse(raw);
      if (s && typeof s === "object") {
        state.answers = s.answers || {}; state.reviewer = s.reviewer || "";
        state.sittings = s.sittings || []; state.open = s.open || null;
      }
    }
  } catch (e) { storageWorks = false; }
}
function save(){ try { window.localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { storageWorks = false; } }
function el(tag, cls, text){ var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
function fmt(s, vals){ return s.replace(/\{(\w+)\}/g, function(_, k){ return vals[k] == null ? "" : String(vals[k]); }); }
function pad(n){ return (n < 10 ? "0" : "") + n; }
function today(){ var d = new Date(); return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()); }
function isoLocal(d){
  var off = -d.getTimezoneOffset(), sign = off >= 0 ? "+" : "-"; off = Math.abs(off);
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + "T" + pad(d.getHours()) + ":" +
    pad(d.getMinutes()) + ":" + pad(d.getSeconds()) + sign + pad(Math.floor(off / 60)) + ":" + pad(off % 60);
}
var CF = /\p{Cf}/u;
function addText(parent, text){
  var buf = "";
  for (var ch of text) {
    if (CF.test(ch)) {
      if (buf) { parent.appendChild(document.createTextNode(buf)); buf = ""; }
      var hex = ch.codePointAt(0).toString(16).toUpperCase(); while (hex.length < 4) hex = "0" + hex;
      parent.appendChild(el("span", "cf", "[U+" + hex + "]"));
    } else { buf += ch; }
  }
  if (buf) parent.appendChild(document.createTextNode(buf));
}
function addMarked(parent, text, tokens){
  var marks = [];
  (tokens || []).forEach(function(t){
    if (!t) return;
    var i = text.indexOf(t);
    while (i >= 0) { marks.push([i, i + t.length]); i = text.indexOf(t, i + t.length); }
  });
  marks.sort(function(a, b){ return a[0] - b[0] || b[1] - a[1]; });
  var merged = [];
  marks.forEach(function(m){ var last = merged[merged.length - 1]; if (last && m[0] <= last[1]) { last[1] = Math.max(last[1], m[1]); } else merged.push([m[0], m[1]]); });
  var pos = 0;
  merged.forEach(function(m){
    if (m[0] > pos) addText(parent, text.slice(pos, m[0]));
    var mk = el("mark"); addText(mk, text.slice(m[0], m[1])); parent.appendChild(mk); pos = m[1];
  });
  if (pos < text.length) addText(parent, text.slice(pos));
}
function factBlocks(parent, blocks){
  (blocks || []).forEach(function(b){
    parent.appendChild(el("div", "fbt", b.title));
    var tw = el("div", "tw"), t = el("table"), thead = el("thead"), hr = el("tr");
    (b.columns || []).forEach(function(c){ hr.appendChild(el("th", null, c)); });
    thead.appendChild(hr); t.appendChild(thead);
    var tb = el("tbody");
    (b.rows || []).forEach(function(r){ var tr = el("tr"); r.forEach(function(c){ var td = el("td"); addText(td, String(c)); tr.appendChild(td); }); tb.appendChild(tr); });
    t.appendChild(tb); tw.appendChild(t); parent.appendChild(tw);
  });
}
function ans(uid){ if (!state.answers[uid]) state.answers[uid] = {verdict:"", note:"", date:"", rows:[]}; return state.answers[uid]; }
function complete(uid){
  var u = D.units[uid], a = state.answers[uid];
  if (!a || !a.verdict) return false;
  if (u.kind === "finding") return D.note_required.indexOf(a.verdict) < 0 || !!(a.note && a.note.trim());
  if (u.kind === "fm_audit") return !!(a.note && a.note.trim());
  if (a.verdict === "none_wrong") return true;
  if (!(a.note && a.note.trim()) || !a.rows || !a.rows.length) return false;
  return a.rows.every(function(r){ return r[0] !== "" && r[1] !== "" && (r[0] !== D.whole_sentence || r[1] === "missing_required_element"); });
}
var refreshers = {};
function progress(){
  var n = 0; D.order.forEach(function(uid){ if (complete(uid)) n++; });
  document.getElementById("count").textContent = fmt(U.progress, {n:n, m:D.order.length});
  document.getElementById("bar").style.width = (D.order.length ? (100 * n / D.order.length) : 0) + "%";
}
function sittingStatus(){
  var s = state.open ? fmt(U.sitting_open, {t:state.open}) : fmt(U.sittings_closed, {n:state.sittings.length});
  if (!storageWorks) s = U.storage_off + " " + s;
  document.getElementById("sitting").textContent = s;
}
function changed(uid){ save(); if (refreshers[uid]) refreshers[uid](); progress(); }
function verdictButtons(parent, uid, allowed, onPick){
  var box = el("div", "btns"), buttons = [];
  allowed.forEach(function(v){
    var b = el("button", null, v); b.type = "button"; b.setAttribute("data-v", v);
    b.addEventListener("click", function(){ var a = ans(uid); a.verdict = v; a.date = today(); if (onPick) onPick(v); changed(uid); });
    buttons.push(b); box.appendChild(b);
  });
  parent.appendChild(box);
  return function(){ var v = (state.answers[uid] || {}).verdict; buttons.forEach(function(b){ b.classList.toggle("on", b.getAttribute("data-v") === v); }); };
}
function noteBox(parent, uid, placeholder){
  var t = el("textarea"); t.placeholder = placeholder; t.value = (state.answers[uid] || {}).note || "";
  t.addEventListener("input", function(){ ans(uid).note = t.value; save(); if (refreshers[uid]) refreshers[uid](); progress(); });
  parent.appendChild(t); return t;
}
function findingUnit(parent, uid){
  var u = D.units[uid], box = el("div", "unit");
  box.appendChild(el("div", "kv", U.unit + ": " + uid));
  var rk = el("div", "kv"); rk.appendChild(el("span", null, U.route_key + ": ")); addText(rk, u.route_key); box.appendChild(rk);
  box.appendChild(el("div", "kv", U.route_class + ": " + u.route_class));
  u.reasons.forEach(function(r){ var d = el("div", "kv"); d.appendChild(el("span", null, U.reason + ": ")); addText(d, r); box.appendChild(d); });
  box.appendChild(el("h4", null, D.exempt.answers_mean));
  u.allowed.forEach(function(v){
    if (v === "correct") { u.correct_means.forEach(function(c){ box.appendChild(el("div", "kv", v + ": " + c)); }); }
    else if (D.answer_means[v]) { box.appendChild(el("div", "kv", D.answer_means[v])); }
  });
  if (u.facts.length) { var det = el("details"); det.appendChild(el("summary", null, U.finding_facts)); factBlocks(det, u.facts); box.appendChild(det); }
  var note;
  var paint = verdictButtons(box, uid, u.allowed, null);
  note = noteBox(box, uid, D.exempt.finding_note);
  refreshers[uid] = function(){
    paint(); var a = state.answers[uid] || {};
    note.classList.toggle("open", D.note_required.indexOf(a.verdict) >= 0 || !!(a.note));
    box.classList.toggle("done", complete(uid));
  };
  refreshers[uid](); parent.appendChild(box);
}
function auditUnit(parent, uid){
  var u = D.units[uid], box = el("div", "unit");
  box.appendChild(el("h4", null, fmt(U.audit_heading, {code:u.fm_code})));
  box.appendChild(el("div", "kv", U.unit + ": " + uid));
  box.appendChild(el("div", "tag", U.audit_evidence));
  u.fm_reasons.forEach(function(r){ var d = el("div", "kv"); addText(d, r); box.appendChild(d); });
  var paint = verdictButtons(box, uid, u.allowed, null);
  var note = noteBox(box, uid, U.audit_note); note.classList.add("open");
  refreshers[uid] = function(){ paint(); box.classList.toggle("done", complete(uid)); };
  refreshers[uid](); parent.appendChild(box);
}
function recordUnit(parent, uid){
  var u = D.units[uid], box = el("div", "unit");
  box.appendChild(el("h4", null, U.record_heading));
  var rowsBox = el("div"), addBtn = el("button", null, U.record_add); addBtn.type = "button";
  function drawRows(){
    while (rowsBox.firstChild) rowsBox.removeChild(rowsBox.firstChild);
    var a = state.answers[uid] || {rows:[]};
    (a.rows || []).forEach(function(r, i){
      var line = el("div", "rec-row");
      var s = el("select"); s.setAttribute("aria-label", U.sentence_pick);
      s.appendChild(new Option(U.sentence_pick, ""));
      s.appendChild(new Option(D.exempt.whole_sentence, D.whole_sentence));
      for (var n = 1; n <= u.sentences.length; n++) s.appendChild(new Option(String(n), String(n)));
      s.value = r[0];
      s.addEventListener("change", function(){ ans(uid).rows[i][0] = s.value; changed(uid); });
      var t = el("select"); t.setAttribute("aria-label", U.type_pick);
      t.appendChild(new Option(U.type_pick, ""));
      D.error_types.forEach(function(e){ t.appendChild(new Option(e, e)); });
      t.value = r[1];
      t.addEventListener("change", function(){ ans(uid).rows[i][1] = t.value; changed(uid); });
      var rm = el("button", null, U.record_remove); rm.type = "button";
      rm.addEventListener("click", function(){ ans(uid).rows.splice(i, 1); drawRows(); changed(uid); });
      line.appendChild(s); line.appendChild(t); line.appendChild(rm); rowsBox.appendChild(line);
    });
  }
  addBtn.addEventListener("click", function(){ var a = ans(uid); a.rows = a.rows || []; a.rows.push(["", ""]); drawRows(); changed(uid); });
  var paint = verdictButtons(box, uid, u.allowed, function(v){ var a = ans(uid); if (v === "wrong" && !(a.rows && a.rows.length)) a.rows = [["", ""]]; drawRows(); });
  var wrongBox = el("div"); wrongBox.appendChild(rowsBox); wrongBox.appendChild(addBtn);
  var note = noteBox(wrongBox, uid, U.record_note); note.classList.add("open");
  box.appendChild(wrongBox);
  drawRows();
  refreshers[uid] = function(){ paint(); var a = state.answers[uid] || {}; wrongBox.style.display = a.verdict === "wrong" ? "" : "none"; box.classList.toggle("done", complete(uid)); };
  refreshers[uid](); parent.appendChild(box);
}
function narrativeCard(parent, nid){
  var n = D.units[nid], card = el("div", "card ncard");
  card.appendChild(el("h3", null, U.narrative));
  card.appendChild(el("div", "tag", U.unit + ": " + nid));
  if (!n.parsed_ok) card.appendChild(el("div", "tag", U.unparsed));
  var tokensBySentence = {};
  function markUnit(fid){ var f = D.units[fid]; f.routed_sentences.forEach(function(s){ (tokensBySentence[s] = tokensBySentence[s] || []).push.apply(tokensBySentence[s], f.highlight); }); }
  n.findings.forEach(function(c){ c.units.forEach(markUnit); });
  n.refs.forEach(function(r){ markUnit(r[0]); });
  n.sentences.forEach(function(s, i){ var p = el("p", "sent"); p.appendChild(el("b", null, "[" + (i + 1) + "]")); addMarked(p, s, tokensBySentence[s]); card.appendChild(p); });
  if (n.facts.length) { var det = el("details"); det.appendChild(el("summary", null, U.output_facts)); factBlocks(det, n.facts); card.appendChild(det); }
  n.findings.forEach(function(c){
    var fc = el("div", "fcard");
    if (c.sentences.length) {
      fc.appendChild(el("div", "tag", U.routed_sentence));
      c.sentences.forEach(function(s){ var p = el("p", "sent"); var toks = []; c.units.forEach(function(fid){ toks = toks.concat(D.units[fid].highlight); }); addMarked(p, s, toks); fc.appendChild(p); });
    } else { fc.appendChild(el("div", "tag", U.no_sentence)); }
    c.units.forEach(function(fid){ findingUnit(fc, fid); });
    card.appendChild(fc);
  });
  n.refs.forEach(function(r){
    var fc = el("div", "fcard"), f = D.units[r[0]];
    fc.appendChild(el("div", "tag", fmt(U.shown_under, {id:r[0], nid:r[1]})));
    f.routed_sentences.forEach(function(s){ var p = el("p", "sent"); addMarked(p, s, f.highlight); fc.appendChild(p); });
    card.appendChild(fc);
  });
  n.audits.forEach(function(aid){ auditUnit(card, aid); });
  recordUnit(card, nid);
  parent.appendChild(card);
}
function build(){
  var root = document.getElementById("cases");
  while (root.firstChild) root.removeChild(root.firstChild);
  refreshers = {};
  D.cases.forEach(function(c){
    root.appendChild(el("h2", null, c.case_id));
    var cc = el("div", "card");
    cc.appendChild(el("h3", null, U.case_facts)); factBlocks(cc, c.facts);
    var det = el("details"); det.appendChild(el("summary", null, U.case_file)); var pre = el("pre"); addText(pre, c.input); det.appendChild(pre); cc.appendChild(det);
    root.appendChild(cc);
    c.narratives.forEach(function(nid){ narrativeCard(root, nid); });
  });
  document.getElementById("reviewer").value = state.reviewer;
  progress(); sittingStatus();
}
function csvField(v){ v = v == null ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function csvText(cols, rows){ var out = cols.join(",") + "\n"; rows.forEach(function(r){ out += cols.map(function(c){ return csvField(r[c]); }).join(",") + "\n"; }); return out; }
function exportRows(){
  return D.order.map(function(uid){
    var u = D.units[uid], a = state.answers[uid] || {}, row = {};
    D.columns.forEach(function(c){ row[c] = ""; });
    row.row_kind = u.kind; row.unit_id = uid;
    if (u.kind === "finding") { row.route_key = u.route_key; row.route_class = u.route_class; row.reason = u.reasons.join(D.list_sep); row.sentence = u.routed_sentences.join(D.list_sep); }
    if (u.kind === "fm_audit") { row.route_key = u.fm_code; row.reason = u.fm_reasons.join(D.list_sep); }
    if (a.verdict) {
      row.verdict = a.verdict; row.note = (a.note || "").replace(/\r/g, ""); row.reviewer = state.reviewer; row.date = a.date || "";
      if (u.kind === "narrative" && a.verdict === "wrong") {
        row.sentence = (a.rows || []).map(function(r){ return r[0]; }).join(D.record_sep);
        row.reason = (a.rows || []).map(function(r){ return r[1]; }).join(D.record_sep);
      }
    }
    return row;
  });
}
function download(name, text){
  var blob = new Blob([text], {type:"text/csv;charset=utf-8"});
  var link = document.createElement("a");
  var made = URL.createObjectURL.call(URL, blob);
  link.download = name; link.href = made; document.body.appendChild(link); link.click();
  setTimeout(function(){ document.body.removeChild(link); URL.revokeObjectURL.call(URL, made); }, 0);
}
function parseCsv(text){
  var rows = [], row = [], f = "", i = 0, q = false;
  text = text.replace(/^\uFEFF/, "");
  while (i < text.length) {
    var c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i += 2; continue; } q = false; i++; continue; } f += c; i++; continue; }
    if (c === '"') { q = true; i++; continue; }
    if (c === ",") { row.push(f); f = ""; i++; continue; }
    if (c === "\r") { i++; continue; }
    if (c === "\n") { row.push(f); rows.push(row); row = []; f = ""; i++; continue; }
    f += c; i++;
  }
  if (f !== "" || row.length) { row.push(f); rows.push(row); }
  return rows;
}
function restore(text){
  var rows = parseCsv(text); if (!rows.length) return;
  var head = rows[0], idx = {}; head.forEach(function(h, i){ idx[h] = i; });
  var n = 0, skipped = 0, refused = 0;
  rows.slice(1).forEach(function(r){
    var uid = r[idx.unit_id]; var u = D.units[uid];
    if (!u) { skipped++; return; }
    var v = r[idx.verdict] || "";
    if (v && u.allowed.indexOf(v) < 0) { refused++; return; }
    var a = ans(uid); a.verdict = v; a.note = r[idx.note] || ""; a.date = r[idx.date] || "";
    if (r[idx.reviewer] && !state.reviewer) state.reviewer = r[idx.reviewer];
    if (u.kind === "narrative") {
      a.rows = [];
      if (v === "wrong") {
        var ss = (r[idx.sentence] || "").split(D.record_sep), ts = (r[idx.reason] || "").split(D.record_sep);
        for (var k = 0; k < Math.max(ss.length, ts.length); k++) a.rows.push([ss[k] || "", ts[k] || ""]);
      }
    }
    n++;
  });
  save(); build();
  document.getElementById("status").textContent = fmt(U.loaded, {n:n, m:skipped, k:refused});
}
document.getElementById("reviewer").addEventListener("input", function(e){ state.reviewer = e.target.value; save(); });
document.getElementById("hide").addEventListener("change", function(e){ document.body.classList.toggle("hide-done", e.target.checked); });
document.getElementById("start").addEventListener("click", function(){ if (!state.open) { state.open = isoLocal(new Date()); save(); } sittingStatus(); });
document.getElementById("end").addEventListener("click", function(){ if (state.open) { state.sittings.push({start:state.open, end:isoLocal(new Date())}); state.open = null; save(); } sittingStatus(); });
document.getElementById("save").addEventListener("click", function(){ download(D.answers_file, csvText(D.columns, exportRows())); });
document.getElementById("save-sittings").addEventListener("click", function(){ download(D.sittings_file, csvText(["start", "end"], state.sittings)); });
var fileIn = document.getElementById("file");
document.getElementById("load").addEventListener("click", function(){ fileIn.click(); });
fileIn.addEventListener("change", function(){
  var f = fileIn.files && fileIn.files[0]; if (!f) return;
  var rd = new FileReader(); rd.onload = function(){ restore(String(rd.result)); fileIn.value = ""; }; rd.readAsText(f);
});
document.getElementById("mark").addEventListener("click", function(){
  if (!window.confirm(D.exempt.confirm_mark)) return;
  D.order.forEach(function(uid){ if (D.units[uid].kind !== "finding") return; var a = ans(uid); if (!a.verdict) { a.verdict = "not_reviewed"; a.date = today(); } });
  save(); build();
});
document.getElementById("clear").addEventListener("click", function(){
  if (!window.confirm(U.confirm_clear)) return;
  try { window.localStorage.removeItem(STORE); } catch (e) {}
  state = {answers:{}, reviewer:"", sittings:[], open:null}; build();
});
load(); build();
})();
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style></head><body>
<header><div class="wrap">
<h1>__TITLE__</h1>
<div class="bar"><i id="bar"></i></div><span class="count" id="count"></span>
<label class="toggle"><input type="checkbox" id="hide"> __HIDE__</label>
<input type="text" id="reviewer" placeholder="__REVIEWER__" aria-label="__REVIEWER__">
<button type="button" id="start">__START__</button><button type="button" id="end">__END__</button>
<button type="button" id="save">__SAVE__</button><button type="button" id="save-sittings">__SAVE_SITTINGS__</button>
<button type="button" id="load">__LOAD__</button><input type="file" id="file" accept=".csv,text/csv" hidden>
<button type="button" id="mark">__MARK__</button><button type="button" id="clear">__CLEAR__</button>
<div class="status" id="sitting"></div><div class="status" id="status"></div>
</div></header>
<main class="wrap">
<div class="intro">__INTRO__</div>
<div id="cases"></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>__JS__</script>
</body></html>
"""


def _embed_json(obj: Any) -> str:
    """JSON safe inside a script block: <, > and & written as unicode escapes."""
    return _cj(obj).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _packet_csv_sha(packet: Packet) -> str:
    return _sha_text(rs.csv_text(rs.PACKET_COLUMNS, packet_csv_rows(packet)))


def _page_data(packet: Packet, title: str, mode: str) -> dict:
    units = [u for u in packet.units if not (mode == "second_look" and u.row_kind == "fm_audit")]
    by_id = {u.unit_id: u for u in units}
    data_units: dict[str, Any] = {}
    cases = []
    for cid in packet.case_order:
        narr_ids = [u.unit_id for u in units if u.row_kind == "narrative" and u.case_id == cid]
        cases.append({"case_id": cid, "input": packet.case_inputs.get(cid, ""),
                      "facts": rf.facts_payload(packet.case_facts.get(cid, ())), "narratives": narr_ids})
    for u in units:
        if u.row_kind == "finding":
            data_units[u.unit_id] = {
                "kind": "finding", "route_key": u.route_key, "route_class": u.route_class,
                "reasons": list(u.reasons), "routed_sentences": list(u.routed_sentences),
                "highlight": list(u.highlight), "correct_means": list(u.correct_means),
                "allowed": list(u.allowed_verdicts), "facts": rf.facts_payload(u.facts),
            }
        elif u.row_kind == "fm_audit":
            data_units[u.unit_id] = {"kind": "fm_audit", "fm_code": u.fm_code, "fm_reasons": list(u.fm_reasons),
                                     "allowed": list(u.allowed_verdicts)}
    for u in units:
        if u.row_kind != "narrative":
            continue
        cards = [{"sentences": list(s), "units": list(ids)} for s, ids in finding_cards(packet, u.unit_id)
                 if all(i in by_id for i in ids)]
        refs = [[f.unit_id, f.narrative_unit_ids[0]] for f in units
                if f.row_kind == "finding" and u.unit_id in f.narrative_unit_ids[1:]]
        audits = [a.unit_id for a in units if a.row_kind == "fm_audit" and a.narrative_unit_ids[:1] == (u.unit_id,)]
        data_units[u.unit_id] = {
            "kind": "narrative", "narrative": u.narrative, "sentences": list(u.sentences), "parsed_ok": u.parsed_ok,
            "facts": rf.facts_payload(u.facts), "allowed": list(u.allowed_verdicts),
            "findings": cards, "refs": refs, "audits": audits,
        }
    return {
        "title": title, "mode": mode, "storage_key": "review:" + _packet_csv_sha(packet),
        "ui": UI_TEXT, "exempt": UI_EXEMPT, "answer_means": FINDING_ANSWER_MEANS,
        "columns": list(rs.RB7_COLUMNS), "error_types": list(rs.ERROR_TYPES),
        "whole_sentence": rs.WHOLE_NARRATIVE_SENTENCE, "list_sep": rs.RB7_LIST_SEP, "record_sep": rs.RECORD_LIST_SEP,
        "note_required": sorted(rs.FINDING_NOTE_REQUIRED), "answers_file": "answers.csv",
        "sittings_file": rs.RECORD_FILES["sittings"],
        "cases": cases, "units": data_units, "order": [u.unit_id for u in units],
    }


def render_html(packet: Packet, *, title: str = "Grouped review", mode: str = "review") -> str:
    """The self-contained reviewer page: no external asset, data as escaped JSON,
    every data string inserted with textContent. second_look drops audit units."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "second_look":
        title = f"Second look: {title}"
    esc = html.escape
    intro = "".join(f"<p>{esc(p)}</p>" for p in INTRO)
    page = _PAGE
    for marker, value in (
        ("__CSS__", _CSS.strip()), ("__HIDE__", esc(UI_TEXT["hide_done"])),
        ("__REVIEWER__", esc(UI_TEXT["reviewer"])), ("__START__", esc(UI_TEXT["start_sitting"])),
        ("__END__", esc(UI_TEXT["end_sitting"])), ("__SAVE_SITTINGS__", esc(UI_TEXT["save_sittings"])),
        ("__SAVE__", esc(UI_TEXT["save_answers"])), ("__LOAD__", esc(UI_TEXT["load_answers"])),
        ("__MARK__", esc(UI_EXEMPT["mark_rest"])), ("__CLEAR__", esc(UI_TEXT["clear_saved"])),
        ("__INTRO__", intro), ("__TITLE__", esc(title)),
    ):
        page = page.replace(marker, value)
    # data and script last, so no marker inside data is ever replaced
    page = page.replace("__JS__", _JS.strip())
    return page.replace("__DATA__", _embed_json(_page_data(packet, title, mode)))


# --- Writing and loading ----------------------------------------------------------------------

def _file_texts(packet: Packet, mode: str) -> dict[str, str]:
    title = "Grouped review"
    texts = {
        rs.PACKET_FILES["packet"]: rs.csv_text(rs.PACKET_COLUMNS, packet_csv_rows(packet)),
        rs.PACKET_FILES["cases"]: rs.csv_text(rs.CASE_COLUMNS, _case_rows(packet)),
        rs.PACKET_FILES["blank"]: rs.csv_text(rs.RB7_COLUMNS, blank_verdict_rows(packet)),
        rs.PACKET_FILES["members"]: rs.csv_text(rs.MEMBER_COLUMNS, members_csv_rows(packet)),
        rs.PACKET_FILES["runs"]: _pretty_json(_runs_json(packet)),
        rs.PACKET_FILES["summary"]: _pretty_json(summary(packet)),
        rs.PACKET_FILES["html"]: render_html(packet, title=title, mode=mode),
    }
    meta = {
        "packet_version": PACKET_VERSION, "run_id": packet.run_id, "seed": packet.seed, "scope": packet.scope,
        "fm_epoch_routes": packet.fm_epoch_routes, "mode": mode,
        "settlement_table_sha256": settlement.table_sha256(),
        "settlement_py_sha256": _module_sha(settlement), "review_facts_py_sha256": _module_sha(rf),
        "review_packet_py_sha256": _sha_file(Path(__file__)), "review_schema_py_sha256": _module_sha(rs),
        "narrative_py_sha256": _sha_file(Path(casefile.__file__).with_name("narrative.py")),
        "vocab_sha256": narrative_vocab.vocab_sha256(),
        "files": {path: _sha_text(text) for path, text in sorted(texts.items())},
    }
    texts[rs.PACKET_FILES["meta"]] = _pretty_json(meta)
    return texts


def write_packet(packet: Packet, out_dir: str, *, mode: str = "review") -> dict[str, str]:
    """Write review_schema.PACKET_FILES under out_dir (created when missing; refused
    when it holds anything); returns relative path -> sha256."""
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refused(f"--out {out} exists and is not empty")
    texts = _file_texts(packet, mode)
    created = not out.exists()
    try:
        for rel_path, text in texts.items():
            p = out / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            rs.write_lf(p, text)
    except OSError:
        if created:
            shutil.rmtree(out, ignore_errors=True)
        else:
            for child in list(out.iterdir()):
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        raise
    return {rel_path: _sha_text(text) for rel_path, text in sorted(texts.items())}


def _json_list(text: str) -> tuple:
    return tuple(json.loads(text)) if text else ()


def _blocks(text: str) -> tuple[FactBlock, ...]:
    return tuple(FactBlock.from_dict(d) for d in (json.loads(text) if text else []))


def load_packet(packet_dir: str) -> Packet:
    """Rebuild a Packet from reviewer/packet.csv, reviewer/cases.csv, sealed/* and
    packet-meta.json (RunInput.log is None; RunInput.path is the file name)."""
    root = Path(packet_dir)
    meta = json.loads((root / rs.PACKET_FILES["meta"]).read_text(encoding="utf-8"))
    runs_json = json.loads((root / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    runs = []
    for rid, r in runs_json.items():
        run = RunInput(run_id=rid, label=r["label"], path=r["file_name"], sha256=r["sha256"], eval_id=r["eval_id"],
                       model=r["model"], revision_commit=r["revision_commit"], log=None)
        _LOADED_RUN_EXTRA[(rid, r["sha256"])] = {"dataset_sha256": r.get("dataset_sha256"),
                                                "template_baseline": r.get("template_baseline")}
        runs.append(run)
    runs.sort(key=lambda r: int(r.run_id[1:]) if r.run_id[1:].isdigit() else 0)
    units = []
    for row in rs.read_csv(root / rs.PACKET_FILES["packet"], rs.PACKET_COLUMNS):
        units.append(Unit(
            unit_id=row["unit_id"], row_kind=row["row_kind"], case_id=row["case_id"], route_key=row["route_key"],
            route_class=row["route_class"], reasons=_json_list(row["reasons"]),
            routed_sentences=_json_list(row["routed_sentences"]), highlight=_json_list(row["highlight"]),
            narrative_unit_ids=_json_list(row["narrative_unit_ids"]), narrative=row["narrative"],
            sentences=_json_list(row["sentences_json"]), parsed_ok=row["parsed_ok"] == "true",
            structured=json.loads(row["structured_json"]) if row["structured_json"] else {},
            fm_code=row["fm_code"], fm_reasons=_json_list(row["fm_reasons"]),
            allowed_verdicts=_json_list(row["allowed_verdicts"]), correct_means=_json_list(row["correct_means"]),
            facts=_blocks(row["facts_json"]),
        ))
    case_order, case_facts, case_inputs = [], {}, {}
    for row in rs.read_csv(root / rs.PACKET_FILES["cases"], rs.CASE_COLUMNS):
        case_order.append(row["case_id"])
        case_facts[row["case_id"]] = _blocks(row["case_facts_json"])
        case_inputs[row["case_id"]] = row["case_input"]
    members = [Member(
        unit_id=row["unit_id"], member_kind=row["member_kind"], run_id=row["run_id"], sample_id=row["sample_id"],
        epoch=int(row["epoch"]), route_key=row["route_key"], route_class=row["route_class"],
        scorers=_json_list(row["scorers"]), reasons=_json_list(row["reasons"]), fm_codes=_json_list(row["fm_codes"]),
    ) for row in rs.read_csv(root / rs.PACKET_FILES["members"], rs.MEMBER_COLUMNS)]
    return Packet(
        run_id=meta["run_id"], seed=int(meta["seed"]), scope=meta["scope"], fm_epoch_routes=meta["fm_epoch_routes"],
        runs=tuple(runs), case_order=tuple(case_order), units=tuple(units), members=_sort_members(members),
        case_facts={k: case_facts[k] for k in sorted(case_facts)},
        case_inputs={k: case_inputs[k] for k in sorted(case_inputs)},
    )


# --- Second-look subset -----------------------------------------------------------------------

def second_look_id(seed: int, row_kind: str, first_unit_id: str, salt: str = "") -> str:
    """A fresh second-look id. With a salt (kept only under the second look's sealed/), the
    id cannot be traced back to the first id from the public seed (red-team M2)."""
    digest = hashlib.sha256(f"{salt}|{seed}|second_look|{first_unit_id}".encode()).hexdigest()[:12]
    return f"S{rs.UNIT_PREFIX[row_kind]}-{digest}"


def subset_packet(packet: Packet, narrative_unit_ids: Sequence[str], *, seed: int,
                  id_salt: str = "") -> tuple[Packet, dict[str, str]]:
    """The drawn narrative units plus every finding unit with a member on any of their
    member epochs; fresh ids (S + prefix, salted with id_salt); no fm_audit units; case
    and unit order reshuffled under seed. Returns (subset, new id -> old id)."""
    by_id = {u.unit_id: u for u in packet.units}
    drawn = list(dict.fromkeys(narrative_unit_ids))
    for nid in drawn:
        if nid not in by_id or by_id[nid].row_kind != "narrative":
            raise ValueError(f"{nid!r} is not a narrative unit of this packet")
    drawn_set = set(drawn)
    epochs = {(m.run_id, m.sample_id, m.epoch) for m in packet.members
              if m.member_kind == "narrative" and m.unit_id in drawn_set}
    finding_ids = {m.unit_id for m in packet.members
                   if m.member_kind == "finding" and (m.run_id, m.sample_id, m.epoch) in epochs}
    new_of = {old: second_look_id(seed, by_id[old].row_kind, old, id_salt) for old in (*drawn, *sorted(finding_ids))}
    old_of = {new: old for old, new in new_of.items()}

    narratives = [_renamed(by_id[n], new_of) for n in drawn]
    findings: dict[str, tuple[dict, set[str]]] = {}
    for fid in sorted(finding_ids):
        u = by_id[fid]
        nids = {new_of[n] for n in u.narrative_unit_ids if n in drawn_set}
        d = {f: getattr(u, f) for f in Unit.__dataclass_fields__ if f != "narrative_unit_ids"}
        d["unit_id"] = new_of[fid]
        findings[new_of[fid]] = (d, nids)
    case_order, units = _display_order(seed, narratives, findings, [])
    members = []
    for m in packet.members:
        if (m.run_id, m.sample_id, m.epoch) not in epochs or m.member_kind == "fm_audit":
            continue
        if m.member_kind in ("finding", "narrative"):
            if m.unit_id not in new_of:
                continue
            m = Member(new_of[m.unit_id], *[getattr(m, f) for f in Member.__dataclass_fields__ if f != "unit_id"])
        members.append(m)
    cases = set(case_order)
    sub = Packet(
        run_id=packet.run_id, seed=seed, scope=packet.scope, fm_epoch_routes=packet.fm_epoch_routes, runs=packet.runs,
        case_order=case_order, units=units, members=_sort_members(members),
        case_facts={k: v for k, v in packet.case_facts.items() if k in cases},
        case_inputs={k: v for k, v in packet.case_inputs.items() if k in cases},
    )
    return sub, {new: old_of[new] for new in sorted(old_of)}


def _renamed(u: Unit, new_of: Mapping[str, str]) -> Unit:
    d = {f: getattr(u, f) for f in Unit.__dataclass_fields__}
    d["unit_id"] = new_of[u.unit_id]
    d["narrative_unit_ids"] = tuple(new_of[n] for n in u.narrative_unit_ids if n in new_of)
    return Unit(**d)


# --- CLI --------------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 is not a contract code
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


class _HelpAction(argparse.Action):
    """-h/--help prints usage and exits 3: nothing was built."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any) -> None:
        super().__init__(option_strings, dest=dest, default=argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        parser.print_help()
        print("REFUSED: help requested; no packet built", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="Grouped review packet from scored dev logs (no model call).", add_help=False)
    p.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (nothing built)")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                   help="a scored .eval log under a label (repeat for several runs)")
    p.add_argument("--out", required=True, help="packet folder; must not exist or be empty")
    p.add_argument("--seed", type=int, required=True, help="packet seed (case order, unit order, unit ids)")
    p.add_argument("--run-id", required=True, help="record id; names no model")
    p.add_argument("--whole-narrative-scope", default=rs.DEFAULT_WHOLE_NARRATIVE_SCOPE,
                   choices=rs.WHOLE_NARRATIVE_SCOPES, help="D-J (open): which epochs get the whole-narrative read")
    p.add_argument("--fm-epoch-routes", default=rs.DEFAULT_FM_EPOCH_ROUTES, choices=rs.FM_EPOCH_ROUTE_MODES,
                   help="D-H (open): routes on an FM-coded epoch are findings, or skipped")
    return p


def _split_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise Refused(f"--run {spec!r} is not LABEL=PATH")
    label, path = spec.split("=", 1)
    return label, path


def run_id_model_words(run_id: str) -> list[str]:
    """MODEL_HINT_WORDS found anywhere in a run id, as a case-insensitive substring: a run
    id joins words with digits ("gpt5-eval", "sonnet45-run"), so a whole-word test misses
    them (red-team M5). Worker text keeps the whole-word test."""
    low = (run_id or "").casefold()
    return sorted({w for w in rs.MODEL_HINT_WORDS if w.casefold() in low})


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id or ""):
        raise Refused(f"run id {run_id!r} does not match {RUN_ID_RE.pattern}")
    if rs.model_hint_hits(run_id) or run_id_model_words(run_id):
        raise Refused("run id names a model word; it sits in packet-meta.json and the record folder name")


def _model_word_texts(packet: Packet) -> int:
    """Worker texts in reviewer files that name a model word (never redacted)."""
    n = 0
    for u in packet.units:
        if u.row_kind == "narrative" and (rs.model_hint_hits(u.narrative) or rs.model_hint_hits(_cj(dict(u.structured)))):
            n += 1
    return n


def _totals_lines(packet: Packet) -> list[str]:
    t = summary(packet)["totals"]
    unsettleable = sum(t["unsettleable_by_key"].values())
    return [
        f"findings: {t['findings']} on {t['route_keys']} route keys; finding units: {t['finding_units']}; "
        f"grouping ratio: {t['grouping_ratio']}",
        f"unsettleable route members: {unsettleable}; skipped FM-epoch route members: {t['fm_epoch_routes_skipped']}",
        f"narrative members: {t['narrative_members']}; narrative units: {t['narrative_units']}",
        f"audit units: {t['audit_units']}; cases with units: {t['cases_with_units']}",
        f"scope: {packet.scope}; fm-epoch routes: {packet.fm_epoch_routes}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _check_run_id(args.run_id)
        out = Path(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise Refused(f"--out {out} exists and is not empty")
        specs = [_split_spec(s) for s in args.run]
        runs = load_runs(specs, args.seed)
        packet = build_packet(runs, run_id=args.run_id, seed=args.seed, scope=args.whole_narrative_scope,
                              fm_epoch_routes=args.fm_epoch_routes)
        hits = _model_word_texts(packet)
        write_packet(packet, str(out))
    except (Refused, GateError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never become a packet
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if hits:
        print(f"worker texts naming a model word: {hits}", file=sys.stderr)
    print("review packet written")
    for line in _totals_lines(packet):
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
