"""Shared names for the grouped review tool: CSV columns, file names, verdict lists,
the forbidden-word and model-word helpers, group and unit ids, and LF-only CSV I/O.

Every name two review modules share lives here and nowhere else. Pure constants and
small helpers; nothing here reads a log, a case or a verdict.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .casefile import fold_dashes, nfkc_spaced

RB7_COLUMNS: tuple[str, ...] = (
    "row_kind", "unit_id", "run_label", "log_eval_id", "sample_id", "epoch", "scorers",
    "route_key", "route_class", "reason", "sentence", "verdict", "note", "reviewer", "date",
)
ROW_KINDS: tuple[str, ...] = ("finding", "narrative", "fm_audit")
FINDING_VERDICTS: tuple[str, ...] = ("correct", "wrong", "citation_omission", "cannot_settle", "not_reviewed")
FINDING_NOTE_REQUIRED: frozenset[str] = frozenset({"wrong", "cannot_settle"})
NARRATIVE_VERDICTS: tuple[str, ...] = ("none_wrong", "wrong")
ERROR_TYPES: tuple[str, ...] = (
    "amount", "direction", "band", "asset", "period", "entity", "item",
    "wallet_holder", "wallet_scope", "missing_required_element", "other",
)
AUDIT_VERDICTS: tuple[str, ...] = ("real_error", "extractor_error")  # note always required
WHOLE_NARRATIVE_SCOPES: tuple[str, ...] = ("routed", "hard", "all")
DEFAULT_WHOLE_NARRATIVE_SCOPE = "routed"  # D-J, open
FM_EPOCH_ROUTE_MODES: tuple[str, ...] = ("findings", "skip")
DEFAULT_FM_EPOCH_ROUTES = "findings"  # D-H, open
MEMBER_KINDS: tuple[str, ...] = ("finding", "narrative", "fm_audit", "unsettleable", "fm_epoch_route_skipped")
MEMBER_COLUMNS: tuple[str, ...] = (
    "unit_id", "member_kind", "run_id", "sample_id", "epoch", "route_key",
    "route_class", "scorers", "reasons", "fm_codes",
)
UNIT_PREFIX: dict[str, str] = {"finding": "F", "narrative": "N", "fm_audit": "A"}

FORBIDDEN_FACT_WORDS: frozenset[str] = frozenset({
    "correct", "correctly", "incorrect", "wrong", "right", "true", "false", "likely", "unlikely",
    "probable", "probably", "proven", "unproven", "prove", "proves", "contradiction", "contradicts",
    "contradict", "contradictory", "match", "matches", "matched", "matching", "mismatch", "consistent",
    "inconsistent", "agree", "agrees", "disagree", "disagrees", "valid", "invalid", "error", "ok",
    "pass", "passes", "fail", "fails", "verdict", "confirmed", "confirms", "supported", "unsupported",
    "grounded", "ungrounded", "accurate", "inaccurate", "plausible", "suspicious", "expected", "differs",
    "equal", "equals", "same", "closest", "nearest", "best", "near", "similar", "resembles", "decoy",
})
FORBIDDEN_FACT_SYMBOLS: tuple[str, ...] = ("\u2713", "\u2717", "==", "!=")
MODEL_HINT_WORDS: tuple[str, ...] = (
    "claude", "sonnet", "haiku", "opus", "gpt", "anthropic", "openai", "luna", "gemini", "mockllm",
)
STATEMENT = ("Every verdict in this record was recorded before any review-settled figure "
             "of the run was computed.")
RB7_LIST_SEP = " | "            # joins reasons and routed sentences in the RB-7 `reason` and `sentence` cells
RECORD_LIST_SEP = ";"           # joins sentence numbers and error types in a narrative `wrong` row
WHOLE_NARRATIVE_SENTENCE = "0"  # sentence number allowed only with error type missing_required_element
REVIEWER_DIR = "reviewer"       # the only folder the reviewer opens
SEALED_DIR = "sealed"           # member map, run labels, per-run counts, second-look map: settlement code only
PACKET_COLUMNS: tuple[str, ...] = (
    "row_kind", "unit_id", "case_id", "route_key", "route_class", "reasons", "routed_sentences",
    "highlight", "narrative_unit_ids", "narrative", "sentences_json", "parsed_ok",
    "structured_json", "fm_code", "fm_reasons", "allowed_verdicts", "correct_means", "facts_json",
)
CASE_COLUMNS: tuple[str, ...] = ("case_id", "case_input", "case_facts_json")
SECOND_LOOK_MAP_COLUMNS: tuple[str, ...] = (
    "sl_unit_id", "first_unit_id", "row_kind", "run_id", "stratum", "first_class",
)
PACKET_FILES: dict[str, str] = {
    "html": "reviewer/packet.html", "packet": "reviewer/packet.csv", "cases": "reviewer/cases.csv",
    "blank": "reviewer/verdicts-blank.csv", "members": "sealed/members.csv",
    "runs": "sealed/runs.json", "summary": "sealed/summary.json", "meta": "packet-meta.json",
}
RECORD_FILES: dict[str, str] = {"verdicts": "verdicts.csv", "fm_audit": "fm_audit.csv", "sittings": "sittings.csv"}

_BOM = b"\xef\xbb\xbf"
_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# A whole word: no letter or digit on either side ([^\W_] is a letter or digit, any script).
_EDGE_BEFORE = r"(?<![^\W_])"
_EDGE_AFTER = r"(?![^\W_])"


def _word_re(words: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w)))
    return re.compile(f"{_EDGE_BEFORE}(?:{alternation}){_EDGE_AFTER}", re.IGNORECASE)


_FORBIDDEN_RE = _word_re(FORBIDDEN_FACT_WORDS)
_MODEL_HINT_RE = _word_re(MODEL_HINT_WORDS)


def group_key(parts: Sequence[object]) -> str:
    """Canonical JSON of the list of parts: the grouping key string."""
    return json.dumps(list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def forbidden_word_hits(text: str) -> list[str]:
    """FORBIDDEN_FACT_WORDS found as whole words (case-insensitive, given lower-case) and
    the forbidden symbols found anywhere; sorted, distinct."""
    text = str(text)
    hits = {m.group(0).casefold() for m in _FORBIDDEN_RE.finditer(text)}
    hits.update(s for s in FORBIDDEN_FACT_SYMBOLS if s in text)
    return sorted(hits)


def model_hint_hits(text: str) -> list[str]:
    """MODEL_HINT_WORDS found as whole words (no letter or digit either side),
    case-insensitive; given lower-case, sorted, distinct."""
    return sorted({m.group(0).casefold() for m in _MODEL_HINT_RE.finditer(str(text))})


def unit_id(seed: int, row_kind: str, group_key: str) -> str:
    """The stable unit id for one group key: prefix by row kind, then 12 hex digits."""
    digest = hashlib.sha256(f"{seed}|{row_kind}|{group_key}".encode()).hexdigest()[:12]
    return f"{UNIT_PREFIX[row_kind]}-{digest}"


def normalize_for_match(text: str) -> str:
    """The form used only to locate routed text: NFKC with format characters spaced,
    dashes folded, whitespace collapsed, stripped. Never part of a group key."""
    return _WS_RE.sub(" ", fold_dashes(nfkc_spaced(text))).strip()


def narrative_sentences(text: str) -> tuple[str, ...]:
    """The raw text (whitespace collapsed, stripped) split after ".", "!" or "?" and
    the whitespace that follows. Sentence numbers are 1-based indexes into this tuple."""
    collapsed = _WS_RE.sub(" ", str(text)).strip()
    if not collapsed:
        return ()
    return tuple(s for s in _SENTENCE_SPLIT_RE.split(collapsed) if s)


def _check_bytes(data: bytes, columns: Sequence[str], path: object) -> None:
    if data.startswith(_BOM):
        raise ValueError(f"{path}: starts with a byte order mark")
    if b"\r" in data:
        raise ValueError(f"{path}: holds a carriage return; LF line ends only")
    header = ",".join(columns)
    first = data.split(b"\n", 1)[0].decode("utf-8")
    if first != header:
        raise ValueError(f"{path}: first line is not the header {header!r}")


def read_csv(path: str | Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read a CSV written by write_csv. Raises ValueError unless the bytes have no BOM
    and no carriage return, the first line is exactly the header, and every row has
    exactly one cell per column. A quoted cell may hold "\\n"."""
    path = Path(path)
    data = path.read_bytes()
    _check_bytes(data, columns, path)
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    next(reader)  # header, already checked
    rows: list[dict[str, str]] = []
    for n, cells in enumerate(reader, start=2):
        if len(cells) != len(columns):
            raise ValueError(f"{path}: row {n} has {len(cells)} cells, expected {len(columns)}")
        rows.append(dict(zip(columns, cells)))
    return rows


def read_rb7_csv(path: str | Path) -> list[dict[str, str]]:
    """Read an RB-7 review CSV (read_csv with RB7_COLUMNS)."""
    return read_csv(path, RB7_COLUMNS)


def csv_text(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> str:
    """The exact text write_csv writes: header first, LF line ends, QUOTE_MINIMAL,
    missing keys blank. A key outside the columns, or a carriage return in any
    cell, raises ValueError, so the reader never refuses what the writer wrote."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)
    known = set(columns)
    for row in rows:
        extra = [k for k in row if k not in known]
        if extra:
            raise ValueError(f"row holds keys outside the columns: {extra}")
        cells = []
        for col in columns:
            value = row.get(col)
            cell = "" if value is None else str(value)
            if "\r" in cell:
                raise ValueError(f"cell {col!r} holds a carriage return; LF line ends only")
            cells.append(cell)
        writer.writerow(cells)
    return buf.getvalue()


def write_csv(path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    """Write a CSV: utf-8, no BOM, LF line ends, QUOTE_MINIMAL, header first."""
    write_lf(path, csv_text(columns, rows))


def write_rb7_csv(path: str | Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Write an RB-7 review CSV (write_csv with RB7_COLUMNS)."""
    write_csv(path, RB7_COLUMNS, rows)


def write_lf(path: str | Path, text: str) -> None:
    """Write text as utf-8 bytes, unchanged, so LF line ends stay LF on Windows."""
    Path(path).write_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
