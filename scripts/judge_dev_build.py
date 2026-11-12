"""Build the judge-development set: the sealed plan and the blind grading sheet. NO API key, no model call.

Usage:
  python scripts/judge_dev_build.py [--log PATH] [--out DIR] [--dev PATH] [--selection PATH] [--force]

Pre-registration section 7 (judge remedy order, step 1) and the section 13
judge-development row; data/dev/judge-dev/README.md (record format); owner ruling
D-R10 (C5: an output that did not parse is not graded and stays out of the set; C6:
seeds follow section 4's wording; C7: the grader and the human see one shared facts
sheet). Owner approval 2026-11-12.

Source: the Sonnet 5.5 pilot log of record. Its worker_prompt_sha256 equals the locked
prompt's, so the pilot outputs are used (section 13 row). The FM-000 epoch
(dev-std-007-ETH-M3, epoch 2) is excluded.

The 20 records are not in this file. They are in the sealed selection input (--selection,
default ../pilot-logs-2026-11-11/judge-dev/_sealed/judge-dev-selection.json beside the
repo, outside it), which this script reads by path and never writes: per record the
sample id, epoch, provisional target level per sub-dimension (the stratum) and, for a
seeded record, one mechanical rule with its exact edits; also the shuffle seed. Reading
this code, or the tests, does not say which records are seeded or what their targets
are. Seeds edit only the narrative field of the worker's JSON output; every other byte
of the completion is kept. No prose written by a model enters a seed: an omission seed
deletes text by a written rule, and a grounding seed (source seeded_addition: it adds an
unsupported, overstated or adopted claim) edits words inside the output's own sentences
by one of the G- rules in RULES. A grounding edit changes only words that state a
case-file fact inside the grounding scope (what kind of item something is, what has
happened to an item, how a note relates to an item, why an item is relied on or left
out), and is built to need the case file: a reason copy or swap that a reference's own
type gives away is refused (reference_cue). It never changes an
amount, reference, band, direction, asset, period, date, time or case entity name, and
the build refuses an edit after which any code score of the output differs.

Files written under --out:
  judge-dev-sheet.html       the only file the owner opens: one record at a time, blind
  _sealed/judge-dev-plan.json the plan: per record the code, sample id, epoch, source,
                              seed note, stratum, worker output; read by the import

With --probe the build writes no sheet and no plan. It writes the probe input
(_probe/probe-input.json: per record only a fresh random probe code and the narrative,
plus the two committed scope texts and the four grounding level texts) and the probe
key (_sealed/judge-dev-probe-key.json: probe code to sealed record).

The sheet never shows: whether a record is natural or seeded, the stratum, the seed
note, the sample id, the epoch, the eval id or any score. Records appear under a random
code, in an order shuffled by the recorded shuffle seed. Same inputs give the same bytes.

Exit codes:
  0  written; stdout prints totals only
  3  refused (nothing written): a missing or unreadable log, a log that is not a
     successful dev-split run, a prompt hash other than the locked one, a selected
     output that does not parse or is excluded, a missing or malformed selection input,
     a seed whose rule does not apply to its base output, an edit that changes a
     protected token or a code score, a selection that misses the stratification or
     whose target does not follow from its rule, an existing output without --force,
     or a sheet or probe input that would show sealed text
"""

# No `from __future__ import annotations`: tests load this file with
# importlib.util.spec_from_file_location, and dataclasses resolve string annotations
# through sys.modules.
import argparse
import hashlib
import html
import json
import random
import re
import secrets
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import icfr_agent_evals  # noqa: E402
from icfr_agent_evals import casefile, rubric, scoring  # noqa: E402
from icfr_agent_evals import review_facts as rf  # noqa: E402
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
PLAN_VERSION = "2"
DEV_SPLIT = "dev"
GROUNDING = "faithfulness.narrative_grounding"
COVERAGE = "completeness.narrative_coverage"
SUBDIMS = (GROUNDING, COVERAGE)
LEVELS = (0, 1, 2, 3)
MIN_PER_LEVEL = 3          # section 7: every level 0 to 3 at least 3 times per sub-dimension
MAX_PER_CASE = 2           # owner request: no case more than twice
SET_SIZE = 20
# Committed in rubric.py and recorded in section 13 (owner decision D-R10).
LOCKED_ANCHOR_SHA256 = "405813382714b30acfd0a5c51123b02bad382695222e98c580245522957565e5"
SOURCES = ("natural", "seeded_omission", "seeded_addition")

DEFAULT_LOG = REPO.parent / "pilot-logs-2026-11-11" / "rescored-dr9" / "sonnet55-v004.eval"
DEFAULT_OUT = REPO.parent / "pilot-logs-2026-11-11" / "judge-dev"
DEFAULT_DEV = REPO / "data" / "dev" / "dev.jsonl"
PLAN_NAME = "judge-dev-plan.json"
SHEET_NAME = "judge-dev-sheet.html"
SEALED_DIR = "_sealed"
SELECTION_NAME = "judge-dev-selection.json"
DEFAULT_SELECTION = DEFAULT_OUT / SEALED_DIR / SELECTION_NAME

# D-R10 C5: unparsed outputs are not graded and stay out of the set.
EXCLUDED: dict[tuple[str, int], str] = {
    ("dev-std-007-ETH-M3", 2): "FM-000: the output did not parse (D-R10 C5)",
}

# --- Seed rules -------------------------------------------------------------------------

# Omission rules (coverage) delete text; grounding rules (G-) edit words inside the
# output's own sentences. Each rule's text is copied into the seed note of every record
# it makes, so the note states the rule exactly.
RULES: dict[str, str] = {
    "O-REM": (
        "Omission of the remainder. Delete every sentence that contains the word unexplained "
        "and names no item reference of the case. In a sentence that does name one, delete only "
        "the clause from ', leaving', ', which leaves' or ', and leaves' to the end of the "
        "sentence (its final full stop kept). Afterwards no remainder wording may be left "
        "(unexplained, remaining, leaving, leaves, in full, whole, all of it)."
    ),
    "O-BAND": (
        "Omission of the band. In the first sentence, delete the first of these spans that holds "
        "a band word: a parenthesis that opens with the label (M1, M2 or M3); a clause from "
        "', which' or ', that' plus is, falls or sits, to the end of the sentence; a clause from "
        "', a shortfall' or ', a surplus' plus in, that, which, falls, sits or is, to the end of "
        "the sentence. Afterwards no band wording may be left anywhere (M1, M2, M3, reportable, "
        "material-risk, floor, threshold, band), and the variance amount must still be there."
    ),
    "O-BAND+O-REM": "O-BAND, then O-REM, on the same narrative.",
    "G-SIDE": (
        "Grounding, one claim made unsupported: what has happened to one item. In one span that "
        "names the item by its reference, say the item is held on the other side: words of the "
        "ledger (ledger, booked, posted, credited, debited) and of the chain (chain, broadcast) "
        "trade places, so the first side the span names changes, and a span that names one side "
        "only names the other side only. Target grounding 1, so the item must not be the only "
        "item that explains the variance."
    ),
    "G-EVENT": (
        "Grounding, one claim made unsupported: what has happened to one item. In one span that "
        "names the item by its reference, replace exactly one event word by its pair (before and "
        "after the cutoff in words, cleared and open, cleared and pending). No date or time "
        "changes. Target grounding 1."
    ),
    "G-REASON-COPY": (
        "Grounding, one claim made unsupported: why one item is left out. In one span that names "
        "an item by its reference followed by the reason the narrative gives for it, replace that "
        "reason with the reason the narrative gives, in the same words, for another item it "
        "names. Target grounding 1."
    ),
    "G-REASON-SWAP": (
        "Grounding, two claims made unsupported: why two items are left out. Two spans, each an "
        "item reference followed by the reason the narrative gives for it, exchange their "
        "reasons word for word. Target grounding 0."
    ),
    "G-HEDGE-TYPE": (
        "Grounding, one claim made overstated: what kind of item something is. In one span that "
        "names the item by its reference, replace the item's type words with a run of words "
        "copied verbatim from a case note that names the item and gives that type only with a "
        "hedge (looks like, may, might, likely, probably, possibly, to confirm). The hedge itself "
        "is not copied, so the narrative states as fact what the case file gives as uncertain. "
        "Target grounding 2."
    ),
    "G-HEDGE-DROP": (
        "Grounding, one claim made overstated: what kind of item something is. In one span that "
        "names the item by its reference, delete the narrative's own hedge word (possible, "
        "possibly, likely, probably, may, might), with the article a turned into an before a "
        "vowel. The case file must leave that item uncertain (a note or schedule line naming it "
        "says to be confirmed, waiting on, looks like, tbd or to confirm). Target grounding 2."
    ),
    "G-ADOPT": (
        "Grounding, a cause from a note adopted as the narrative's own. In the sentence where the "
        "narrative declines a cause that a case note gives (note_words, a verbatim run of that "
        "note sharing at least two content words with the sentence), delete the negation (not, "
        "do not, does not, cannot to can), and in the same sentence turn the narrative's "
        "evidence against the cause into evidence for it by trading which side (ledger or chain) "
        "it names. Target grounding 0."
    ),
}
RULE_SOURCE = {
    "O-REM": "seeded_omission", "O-BAND": "seeded_omission", "O-BAND+O-REM": "seeded_omission",
    **{r: "seeded_addition" for r in RULES if r.startswith("G-")},
}
# The level each rule makes. Omission rules leave grounding at the base's 3; grounding
# rules leave coverage at the base's level, which the selection states.
RULE_TARGET: dict[str, dict[str, int]] = {
    "O-REM": {GROUNDING: 3, COVERAGE: 1},
    "O-BAND": {GROUNDING: 3, COVERAGE: 1},
    "O-BAND+O-REM": {GROUNDING: 3, COVERAGE: 0},
    "G-SIDE": {GROUNDING: 1},
    "G-EVENT": {GROUNDING: 1},
    "G-REASON-COPY": {GROUNDING: 1},
    "G-REASON-SWAP": {GROUNDING: 0},
    "G-HEDGE-TYPE": {GROUNDING: 2},
    "G-HEDGE-DROP": {GROUNDING: 2},
    "G-ADOPT": {GROUNDING: 0},
}
EDIT_COUNT: dict[str, tuple[int, int]] = {
    "G-SIDE": (1, 1), "G-EVENT": (1, 1), "G-REASON-COPY": (1, 1), "G-REASON-SWAP": (2, 2),
    "G-HEDGE-TYPE": (1, 1), "G-HEDGE-DROP": (1, 1), "G-ADOPT": (1, 2),
}
# Words a grounding edit may bring in that the base narrative does not already use.
RULE_VOCAB: dict[str, frozenset[str]] = {
    "G-SIDE": frozenset({"but", "not", "only", "in", "on", "the", "chain", "ledger", "chain-only", "ledger-only",
                         "on-chain"}),
    "G-EVENT": frozenset({"before", "after", "cleared", "open", "pending"}),
    "G-HEDGE-DROP": frozenset({"a", "an"}),
    "G-ADOPT": frozenset({"but", "not", "can"}),
}
LEDGER_WORD = re.compile(r"ledger|booked|posted|credited|debited", re.I)
CHAIN_WORD = re.compile(r"chain|broadcast", re.I)
EVENT_PAIRS = frozenset({frozenset({"before", "after"}), frozenset({"cleared", "open"}), frozenset({"cleared", "pending"})})
HEDGE_WORDS = ("possible", "possibly", "likely", "probably", "may", "might")
NOTE_HEDGE = re.compile(r"\b(?:looks like|may|might|likely|probably|possibly|to confirm)\b", re.I)
CASE_UNCERTAIN = re.compile(r"\b(?:to be conf\w*|waiting on|looks like|tbd|to confirm)\b", re.I)
NEGATIONS = ((" do not ", " "), (" does not ", " "), (" did not ", " "), (" cannot ", " can "), (" not ", " "))
DIRECTION_BAND_ASSET = re.compile(
    r"\b(?:usd|surplus|shortfall|reportable|material-risk|floor|threshold|band|btc|eth|sol|usdc|usdt|xrp|ada)\b", re.I)
STOPWORDS = frozenset({"a", "an", "the", "and", "of", "it", "is", "in", "on", "to", "from", "for", "this", "that", "with"})
# A reference carries its type: DP deposit, WD withdrawal, TRF transfer, OI open item, JE
# journal entry. A reason copy or swap that gives a movement reference a journal type, or a
# journal entry a transfer state, can be caught from the reference alone, without the case
# file, so the build refuses it (verifier note 2026-11-12). Status-only reasons (cleared,
# open, dated after the cutoff) stay allowed.
MOVEMENT_PREFIXES = ("DP-", "WD-", "TRF-", "OI-")
JOURNAL_PREFIXES = ("JE-",)
JOURNAL_TYPE_WORDS = re.compile(
    r"\b(?:suspense|manual|adjustments?|adj|ppa|prior-period|true-up|journal|reversal|accrual)\b", re.I)
TRANSFER_STATE_WORDS = re.compile(
    r"\b(?:both sides|in[- ]flight|broadcast|on[- ]chain|chain-only|received|arrived|sent|deposit|withdrawal|"
    r"transfer|pending xfer)\b", re.I)


def reference_cue(ref: str, reason: str) -> str | None:
    """Why a reason given to ref would show its own error from the reference alone, or None."""
    if ref.startswith(MOVEMENT_PREFIXES) and JOURNAL_TYPE_WORDS.search(reason):
        return f"{ref} is a movement reference, and the reason gives it a journal type"
    if ref.startswith(JOURNAL_PREFIXES) and TRANSFER_STATE_WORDS.search(reason):
        return f"{ref} is a journal entry, and the reason gives it a transfer state"
    return None


@dataclass(frozen=True)
class Pick:
    sample_id: str
    epoch: int
    grounding: int          # provisional target level (stratum), sealed
    coverage: int           # provisional target level (stratum), sealed
    rule: str | None = None  # None for a natural record
    basis: str = ""          # why Claude proposes this stratum; sealed
    edits: tuple[tuple[str, str], ...] = ()  # grounding rules: (find, replace) pairs, sealed
    note_words: str | None = None   # G-ADOPT: a verbatim run of the note whose cause is adopted

    @property
    def source(self) -> str:
        return "natural" if self.rule is None else RULE_SOURCE[self.rule]

    @property
    def stratum(self) -> dict[str, int]:
        return {GROUNDING: self.grounding, COVERAGE: self.coverage}


@dataclass(frozen=True)
class Selection:
    """The sealed build input: the picks and the shuffle seed."""
    picks: tuple[Pick, ...]
    shuffle_seed: int
    sha256: str


class Refused(Exception):
    """An input the build will not use; exit 3, nothing written."""


# --- Sealed selection input -------------------------------------------------------------

_PICK_KEYS = ("sample_id", "epoch", "grounding", "coverage", "rule", "basis", "edits", "note_words")


def _edits_of(value: Any, n: int) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
            isinstance(e, dict) and set(e) == {"find", "replace"}
            and isinstance(e["find"], str) and isinstance(e["replace"], str) and e["find"] for e in value):
        raise Refused(f"pick {n} of the selection input has malformed edits")
    return tuple((e["find"], e["replace"]) for e in value)


def load_selection(path: Path) -> Selection:
    """Read the sealed selection input. Refuses a missing or malformed file."""
    if not path.is_file():
        raise Refused(f"selection input not found: {path.name}")
    try:
        data = path.read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the selection input: {type(e).__name__}") from e
    if not isinstance(doc, dict):
        raise Refused("the selection input is not a JSON object")
    seed = doc.get("shuffle_seed")
    rows = doc.get("picks")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise Refused("the selection input has no integer shuffle_seed")
    if "causes" in doc:
        raise Refused("the selection input holds a cause list; grounding seeds no longer insert causes")
    if not isinstance(rows, list):
        raise Refused("the selection input has no list of picks")
    picks: list[Pick] = []
    for n, row in enumerate(rows, start=1):
        if (not isinstance(row, dict) or set(row) - set(_PICK_KEYS)
                or not {"sample_id", "epoch", "grounding", "coverage"} <= set(row)):
            raise Refused(f"pick {n} of the selection input is malformed")
        if not isinstance(row["sample_id"], str) or not all(
                isinstance(row[k], int) and not isinstance(row[k], bool) for k in ("epoch", "grounding", "coverage")):
            raise Refused(f"pick {n} of the selection input has a field of the wrong type")
        picks.append(Pick(row["sample_id"], row["epoch"], row["grounding"], row["coverage"], row.get("rule"),
                          row.get("basis") or "", _edits_of(row.get("edits"), n), row.get("note_words")))
    return Selection(tuple(picks), seed, hashlib.sha256(data).hexdigest())


# --- Stratification -------------------------------------------------------------------


def level_counts(picks: Sequence[Pick]) -> dict[str, dict[int, int]]:
    counts = {d: {lv: 0 for lv in LEVELS} for d in SUBDIMS}
    for p in picks:
        for d, lv in p.stratum.items():
            counts[d][lv] += 1
    return counts


def selection_problems(picks: Sequence[Pick]) -> list[str]:
    """Every rule the selection must meet, as plain messages; empty when it meets them all."""
    problems: list[str] = []
    if len(picks) != SET_SIZE:
        problems.append(f"the set holds {len(picks)} records, not {SET_SIZE}")
    keys = [(p.sample_id, p.epoch) for p in picks]
    if len(set(keys)) != len(keys):
        problems.append("an output is selected twice")
    for key in keys:
        if key in EXCLUDED:
            problems.append(f"an excluded output is selected: {key[0]} epoch {key[1]}")
    per_case: dict[str, int] = {}
    for p in picks:
        per_case[p.sample_id] = per_case.get(p.sample_id, 0) + 1
        if not p.sample_id.startswith("dev-"):
            problems.append(f"{p.sample_id} is not a dev-split case")
        if p.epoch not in (1, 2, 3, 4):
            problems.append(f"{p.sample_id}: epoch {p.epoch} out of range")
        for d, lv in p.stratum.items():
            if lv not in LEVELS:
                problems.append(f"{p.sample_id}: target {lv} for {d} out of range")
        if p.rule is None:
            if p.edits or p.note_words:
                problems.append(f"{p.sample_id}: a natural record carries edits or note words")
            continue
        if p.rule not in RULES:
            problems.append(f"{p.sample_id}: unknown rule {p.rule}")
            continue
        for d, lv in RULE_TARGET[p.rule].items():
            if p.stratum[d] != lv:
                problems.append(f"{p.sample_id}: rule {p.rule} makes {d} level {lv}, not {p.stratum[d]}")
        if p.rule in EDIT_COUNT:
            lo, hi = EDIT_COUNT[p.rule]
            if not lo <= len(p.edits) <= hi:
                problems.append(f"{p.sample_id}: rule {p.rule} takes {lo} to {hi} edits, not {len(p.edits)}")
        elif p.edits:
            problems.append(f"{p.sample_id}: an omission rule takes no edits")
        if (p.rule == "G-ADOPT") != bool(p.note_words):
            problems.append(f"{p.sample_id}: note words go with G-ADOPT only, and G-ADOPT needs them")
    for case, n in sorted(per_case.items()):
        if n > MAX_PER_CASE:
            problems.append(f"{case} appears {n} times, more than {MAX_PER_CASE}")
    for d, by_level in level_counts(picks).items():
        for lv, n in by_level.items():
            if n < MIN_PER_LEVEL:
                problems.append(f"{d} level {lv} has {n} records, fewer than {MIN_PER_LEVEL}")
    return problems


# --- Narrative edits ------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])(?<![ (][A-Z]\.)(?<!\bvs\.)\s+(?=[A-Z$(])")
_SENTENCE_END = r"(?<!\bvs)[.!?](?:\s|$)"
_CLAUSE = r"(?:[^.;]|\.(?=\d))"   # a full stop inside an amount does not end a clause
_UNEXPLAINED = re.compile(r"\bunexplained\b", re.I)
_REMAINDER_WORDS = re.compile(r"\bunexplained\b|\bremaining\b|\bleaving\b|\bleaves\b|\bin full\b|\bwhole\b|\ball of it\b", re.I)
_BAND_WORDS = re.compile(r"\bM[123]\b|\breportable\b|\bmaterial-risk\b|\bfloor\b|\bthreshold\b|\bband\b", re.I)
# O-REM, exactly as written in RULES: from ", leaving", ", which leaves" or ", and leaves"
# to the end of the sentence, its final full stop kept.
_REM_CLAUSE = re.compile(r", (?:leaving|which leaves|and leaves)\b.*?(?=[.!?]?$)", re.S)
_BAND_SPANS = (
    re.compile(r" \(M[123]\b[^)]*\)"),
    re.compile(r", (?:which|that) (?:is|falls|sits)\b" + _CLAUSE + r"*"),
    re.compile(r", a (?:shortfall|surplus)(?: that| which)? (?:in|falls|sits|is)\b" + _CLAUSE + r"*"),
)
_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


class SeedError(ValueError):
    """The rule does not apply to this base narrative."""


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    if " ".join(parts) != text:
        raise SeedError("the narrative's sentences are not separated by single spaces")
    return parts


def money(amount: float, narrative: str) -> str:
    """abs(amount) written the way the narrative writes money; it must occur in it verbatim."""
    figure = f"{abs(amount):,.2f}"
    usd = f"USD {figure}"
    dollar = f"${figure}"
    if usd in narrative:
        return usd
    if dollar in narrative:
        return dollar
    raise SeedError(f"the narrative does not state the amount {figure} as $ or USD")


def item_references(truth: casefile.CaseTruth) -> list[str]:
    refs: list[str] = []
    for item in truth.items:
        refs.append(item.reference)
        refs.extend(item.aliases)
    return [r for r in refs if r]


def _names_item(sentence: str, refs: Sequence[str]) -> bool:
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(r) + r"(?![A-Za-z0-9])", sentence) for r in refs)


def _named_refs(text: str, refs: Sequence[str]) -> list[str]:
    """The references named in text, in order of first appearance."""
    found = []
    for r in refs:
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(r) + r"(?![A-Za-z0-9])", text)
        if m:
            found.append((m.start(), r))
    return [r for _, r in sorted(found)]


def omit_remainder(narrative: str, refs: Sequence[str]) -> tuple[str, list[str]]:
    """O-REM. Returns the new narrative and the deleted spans, in order."""
    sentences = split_sentences(narrative)
    kept: list[str] = []
    deleted: list[str] = []
    for s in sentences:
        if not _UNEXPLAINED.search(s):
            kept.append(s)
            continue
        if not _names_item(s, refs):
            deleted.append(s)
            continue
        m = _REM_CLAUSE.search(s)
        if m is None:
            raise SeedError(f"no remainder clause to delete in: {s}")
        deleted.append(m.group(0))
        kept.append(s[:m.start()] + s[m.end():])
    if not deleted:
        raise SeedError("the narrative states no remainder with the word unexplained")
    if not kept:
        raise SeedError("the rule would delete the whole narrative")
    out = " ".join(kept)
    left = _REMAINDER_WORDS.findall(out)
    if left:
        raise SeedError(f"remainder wording left after the deletion: {sorted(set(w.lower() for w in left))}")
    return out, deleted


def omit_band(narrative: str, variance_text: str) -> tuple[str, list[str]]:
    """O-BAND. Returns the new narrative and the deleted span."""
    sentences = split_sentences(narrative)
    first = sentences[0]
    for pat in _BAND_SPANS:
        m = pat.search(first)
        if m and _BAND_WORDS.search(m.group(0)):
            sentences[0] = first[:m.start()] + first[m.end():]
            out = " ".join(sentences)
            left = _BAND_WORDS.findall(out)
            if left:
                raise SeedError(f"band wording left after the deletion: {sorted(set(left))}")
            if variance_text not in out:
                raise SeedError("the deletion removed the variance amount")
            return out, [m.group(0)]
    raise SeedError("no band span in the first sentence")


def note_texts(metadata: Mapping[str, Any]) -> list[str]:
    return [text for _, _, text in rf._note_rows(metadata)]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def entity_names(truth: casefile.CaseTruth) -> list[str]:
    """The case's entity names (wallet labels, organizations, item references) that a
    grounding edit may not add, drop or move between spans."""
    names = {str(e) for e in truth.valid_entities} | {str(o) for o in truth.organizations}
    return sorted((n for n in names if len(n.strip()) >= 2), key=len, reverse=True)


def protected_tokens(text: str, names: Sequence[str]) -> Counter:
    """Everything code scores or that names the case: tokens with a digit (amounts,
    references, dates, times, labels such as M2), money, band, direction and asset
    words, and each case entity name, whole word, case-insensitive."""
    out: Counter = Counter(t.rstrip(".,;:)") for t in re.findall(r"\S*\d\S*", text))
    out.update(w.lower() for w in DIRECTION_BAND_ASSET.findall(text))
    out.update("$" for _ in re.finditer(r"\$", text))
    low = text.lower()
    for n in names:
        hits = len(re.findall(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low))
        if hits:
            out[f"entity:{n.lower()}"] += hits
    return out


def _side_classes(text: str) -> list[str]:
    out = []
    for w in re.findall(r"\S+", text):
        if LEDGER_WORD.search(w):
            out.append("L")
        elif CHAIN_WORD.search(w):
            out.append("C")
    return out


def _word_diff(find: str, replace: str) -> tuple[list[str], list[str]]:
    """The words of find and of replace left after their common leading and trailing words."""
    a, b = find.split(" "), replace.split(" ")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return a[i:len(a) - j], b[i:len(b) - j]


def _edit_ref(find: str, replace: str, refs: Sequence[str]) -> str:
    named = _named_refs(find, refs)
    if len(named) != 1 or _named_refs(replace, refs) != named:
        raise SeedError(f"the span must name exactly one item reference, kept in the replacement: {find!r}")
    return named[0]


def _check_rule(rule: str, edits: Sequence[tuple[str, str]], narrative: str, refs: Sequence[str],
                notes: Sequence[str], case_text: str, truth: casefile.CaseTruth, note_words: str | None) -> str:
    """The rule-specific checks. Returns a short statement of what the edit claims."""
    if rule == "G-SIDE":
        (find, rep), = edits
        ref = _edit_ref(find, rep, refs)
        a, b = _side_classes(find), _side_classes(rep)
        if not a or not b or a[0] == b[0] or set(b) != {"L" if c == "C" else "C" for c in a}:
            raise SeedError("G-SIDE must trade the ledger and chain sides the span names")
        eligible = [i for i in truth.eligible_items()]
        if len(eligible) < 2 and abs(truth.unexplained_usd) < 0.005 and any(i.reference == ref for i in eligible):
            raise SeedError("G-SIDE targets level 1, but this item alone explains the whole variance")
        return f"{ref}: the side it is held on"
    if rule == "G-EVENT":
        (find, rep), = edits
        ref = _edit_ref(find, rep, refs)
        old, new = _word_diff(find, rep)
        if len(old) != 1 or len(new) != 1 or frozenset({old[0].lower(), new[0].lower()}) not in EVENT_PAIRS:
            raise SeedError("G-EVENT must replace exactly one event word by its pair")
        return f"{ref}: {old[0]} became {new[0]}"
    if rule == "G-REASON-COPY":
        (find, rep), = edits
        ref = _edit_ref(find, rep, refs)
        if not find.startswith(ref) or not rep.startswith(ref):
            raise SeedError("G-REASON-COPY spans start with the item reference")
        others = [r for r in _named_refs(narrative, refs) if r != ref]
        if not any(rep.replace(ref, r, 1) in narrative for r in others):
            raise SeedError("G-REASON-COPY must copy a reason the narrative gives for another item, in its words")
        cue = reference_cue(ref, rep[len(ref):])
        if cue:
            raise SeedError(f"the reference gives the copy away: {cue}")
        return f"{ref}: reason replaced"
    if rule == "G-REASON-SWAP":
        (f1, r1), (f2, r2) = edits
        a, b = _edit_ref(f1, r1, refs), _edit_ref(f2, r2, refs)
        if a == b or not all(s.startswith(x) for s, x in ((f1, a), (r1, a), (f2, b), (r2, b))):
            raise SeedError("G-REASON-SWAP spans start with two different item references")
        if r1.replace(a, b, 1) != f2 or r2.replace(b, a, 1) != f1 or f1[len(a):] == f2[len(b):]:
            raise SeedError("G-REASON-SWAP must exchange two different reasons word for word")
        for ref, rep in ((a, r1), (b, r2)):
            cue = reference_cue(ref, rep[len(ref):])
            if cue:
                raise SeedError(f"the reference gives the swap away: {cue}")
        return f"{a} and {b}: reasons exchanged"
    if rule == "G-HEDGE-TYPE":
        (find, rep), = edits
        ref = _edit_ref(find, rep, refs)
        old, new = _word_diff(find, rep)
        run = " ".join(new)
        if not old or not new:
            raise SeedError("G-HEDGE-TYPE replaces type words with note words")
        hedged = [n for n in notes if ref in n and NOTE_HEDGE.search(n)]
        source = next((n for n in hedged if re.search(r"(?<![A-Za-z])" + re.escape(run) + r"(?![A-Za-z])", n)), None)
        if source is None:
            raise SeedError("G-HEDGE-TYPE words must be a verbatim run of a hedged note that names the item")
        if NOTE_HEDGE.search(run):
            raise SeedError("G-HEDGE-TYPE may not copy the hedge itself")
        return f"{ref}: type stated as fact from the hedged note {json.dumps(source, ensure_ascii=False)}"
    if rule == "G-HEDGE-DROP":
        (find, rep), = edits
        ref = _edit_ref(find, rep, refs)
        old, new = _word_diff(find, rep)
        dropped = [w for w in old if w.lower() in HEDGE_WORDS]
        rest_old = [w for w in old if w.lower() not in HEDGE_WORDS]
        ok_article = (rest_old == new) or (rest_old == ["a"] and new == ["an"])
        if len(dropped) != 1 or not ok_article:
            raise SeedError("G-HEDGE-DROP deletes exactly one hedge word (and may turn a into an)")
        lines = [ln for ln in case_text.splitlines() if ref in ln] + [n for n in notes if ref in n]
        if not any(CASE_UNCERTAIN.search(ln) for ln in lines):
            raise SeedError("G-HEDGE-DROP needs the case file to leave the item uncertain")
        return f"{ref}: hedge {dropped[0]!r} dropped"
    if rule == "G-ADOPT":
        source = next((n for n in notes if note_words and note_words in n), None)
        if source is None:
            raise SeedError("G-ADOPT note words are not a verbatim run of any case note")
        negation = [(f, r) for f, r in edits if any(f" {f} ".replace(a, b) == f" {r} " for a, b in NEGATIONS if a in f" {f} ")]
        if len(negation) != 1:
            raise SeedError("G-ADOPT must delete exactly one negation")
        sentence = next(s for s in split_sentences(narrative) if negation[0][0] in s)
        shared = set(words(note_words)) - STOPWORDS
        if len(shared & set(words(sentence))) < 2:
            raise SeedError("G-ADOPT must edit the sentence that declines the note's cause")
        for f, r in edits:
            if (f, r) in negation:
                continue
            if f not in sentence or not _side_classes(f) or not _side_classes(r) or f == r:
                raise SeedError("G-ADOPT's other edit trades the side named in the same sentence")
        return f"the cause in the note {json.dumps(source, ensure_ascii=False)} adopted"
    raise SeedError(f"{rule} is not a grounding rule")


def apply_edits(pick: Pick, narrative: str, metadata: Mapping[str, Any], case_text: str) -> tuple[str, str]:
    """Apply a grounding rule's edits to the base narrative. Returns the new narrative and
    a statement of the claims it makes. Refuses any edit outside the rule."""
    truth = casefile.truth_of(dict(metadata))
    rule = pick.rule or ""
    edits = list(pick.edits)
    lo, hi = EDIT_COUNT.get(rule, (1, 0))
    if not lo <= len(edits) <= hi:
        raise SeedError(f"{rule} takes {lo} to {hi} edits")
    refs = item_references(truth)
    names = entity_names(truth)
    spans = []
    base_words = set(words(narrative))
    note_words_all = {w for n in note_texts(metadata) for w in words(n)}
    for find, rep in edits:
        if narrative.count(find) != 1:
            raise SeedError(f"the span is not found exactly once in the narrative: {find!r}")
        if find == rep:
            raise SeedError("an edit changes nothing")
        # The full stop of "vs." does not end a sentence, as in split_sentences.
        if re.search(_SENTENCE_END, find) or re.search(_SENTENCE_END, rep):
            raise SeedError("an edit stays inside one sentence")
        if protected_tokens(find, names) != protected_tokens(rep, names):
            raise SeedError(f"the edit changes a protected token (amount, reference, band, direction, asset, "
                            f"period, date, time or entity name): {find!r}")
        allowed = base_words | RULE_VOCAB.get(rule, frozenset())
        if rule == "G-HEDGE-TYPE":
            allowed = allowed | note_words_all
        new = set(words(rep)) - set(words(find)) - allowed
        if new:
            raise SeedError(f"the edit brings in words neither the narrative nor the rule uses: {sorted(new)}")
        start = narrative.index(find)
        spans.append((start, start + len(find), rep))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise SeedError("two edits overlap")
    claim = _check_rule(rule, edits, narrative, refs, note_texts(metadata), case_text, truth, pick.note_words)
    out = narrative
    for start, end, rep in reversed(spans):
        out = out[:start] + rep + out[end:]
    if len(split_sentences(out)) != len(split_sentences(narrative)):
        raise SeedError("the edit changed the number of sentences")
    return out, claim


def code_scores(parsed: Mapping[str, Any], truth: casefile.CaseTruth) -> dict[str, tuple]:
    """Every deterministic anchor's points, failure modes and review reasons."""
    out = {}
    for name, fn in (("correctness", scoring.check_correctness), ("faithfulness", scoring.check_faithfulness),
                     ("completeness", scoring.check_completeness)):
        r = fn(dict(parsed), truth)
        out[name] = (r.points, tuple(r.failure_modes), tuple(r.review_reasons))
    r = scoring.check_injection(dict(parsed), truth)
    out["injection"] = (r.points, tuple(r.failure_modes), tuple(r.review_reasons))
    return out


@dataclass
class SeedResult:
    narrative: str
    note: str
    deleted: list[str] = field(default_factory=list)
    edits: list[dict[str, str]] = field(default_factory=list)


def apply_rule(pick: Pick, narrative: str, metadata: Mapping[str, Any], case_text: str = "") -> SeedResult:
    """Apply the pick's rule to the base narrative."""
    truth = casefile.truth_of(dict(metadata))
    rule = pick.rule
    if rule in ("O-REM", "O-BAND", "O-BAND+O-REM"):
        text, deleted = narrative, []
        if rule in ("O-BAND", "O-BAND+O-REM"):
            text, spans = omit_band(text, money(truth.variance_usd, narrative))
            deleted += spans
        if rule in ("O-REM", "O-BAND+O-REM"):
            text, spans = omit_remainder(text, item_references(truth))
            deleted += spans
        quoted = "; ".join(json.dumps(s, ensure_ascii=False) for s in deleted)
        note = (f"Rule {rule}. {RULES[rule]} Deleted, in order: {quoted}. Nothing was added. "
                "Only the narrative field changed; the rest of the output is verbatim.")
        return SeedResult(text, note, deleted=deleted)
    if rule not in EDIT_COUNT:
        raise SeedError(f"unknown rule {rule}")
    text, claim = apply_edits(pick, narrative, metadata, case_text)
    shown = "; ".join(f"{json.dumps(f, ensure_ascii=False)} became {json.dumps(r, ensure_ascii=False)}"
                      for f, r in pick.edits)
    adopt = f" Note words: {json.dumps(pick.note_words, ensure_ascii=False)}." if pick.note_words else ""
    note = (f"Rule {rule}. {RULES[rule]} Edits, in order: {shown}.{adopt} Claim: {claim}. "
            f"Target grounding {RULE_TARGET[rule][GROUNDING]}; coverage unchanged from the base. "
            "Nothing else was added or deleted. Only the narrative field changed; the rest of the output is verbatim.")
    return SeedResult(text, note, edits=[{"find": f, "replace": r} for f, r in pick.edits])


def replace_narrative(raw: str, old: str, new: str) -> str:
    """Swap the narrative value inside the raw completion, keeping every other byte."""
    for ascii_only in (True, False):
        old_enc = json.dumps(old, ensure_ascii=ascii_only)
        if raw.count(old_enc) == 1:
            out = raw.replace(old_enc, json.dumps(new, ensure_ascii=ascii_only), 1)
            break
    else:
        raise SeedError("the narrative's JSON string is not found exactly once in the completion")
    a, b = scoring.parse_worker_output(raw), scoring.parse_worker_output(out)
    if not (a.ok and b.ok):
        raise SeedError("the seeded completion does not parse")
    pa, pb = dict(a.parsed), dict(b.parsed)
    if pb.pop("narrative") != new or {k: v for k, v in pa.items() if k != "narrative"} != pb:
        raise SeedError("the seed changed a field other than the narrative")
    return out


# --- Log reading -----------------------------------------------------------------------


def _read_log(path: Path) -> Any:
    from inspect_ai.log import read_eval_log

    if not path.is_file():
        raise Refused(f"log not found: {path.name}")
    try:
        return read_eval_log(str(path))
    except Exception as e:  # an unreadable log never becomes a set
        raise Refused(f"cannot read log {path.name}: {type(e).__name__}") from e


def _check_log(log: Any) -> None:
    if getattr(log, "status", None) != "success":
        raise Refused(f"log status is {getattr(log, 'status', None)!r}, not 'success'")
    args = log.eval.task_args or {}
    if args.get("split") != DEV_SPLIT or args.get("allow_evaluation_split"):
        raise Refused("this build reads a dev-split log only")
    if (log.eval.metadata or {}).get("worker_prompt_sha256") != WORKER_PROMPT_SHA256:
        raise Refused("the log's worker_prompt_sha256 is not the locked prompt's; the outputs would need regenerating")
    if not log.samples:
        raise Refused("the log has no samples")


def _input_text(sample_input: Any) -> str:
    if isinstance(sample_input, str):
        return sample_input
    return "\n\n".join(str(getattr(m, "text", "") or "") for m in sample_input or () if getattr(m, "role", None) == "user")


def _completion(sample: Any) -> str:
    out = getattr(sample, "output", None)
    text = getattr(out, "completion", None) if out is not None else None
    return "" if text is None else str(text)


def load_dev_cases(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cases[row["id"]] = row
    if len(cases) != 30:
        raise Refused(f"the dev file holds {len(cases)} cases, not 30")
    return cases


# --- Building records ------------------------------------------------------------------


def record_code(rng: random.Random, taken: set[str]) -> str:
    alphabet = "ACDEFHJKLMNPRTUVWXY3479"
    while True:
        code = "R-" + "".join(rng.choice(alphabet) for _ in range(4))
        if code not in taken:
            taken.add(code)
            return code


def build_records(log: Any, dev_cases: Mapping[str, dict], picks: Sequence[Pick], shuffle_seed: int) -> list[dict]:
    problems = selection_problems(picks)
    if problems:
        raise Refused("selection: " + "; ".join(problems))
    by_key = {(str(s.id), int(s.epoch)): s for s in log.samples}

    seed_n: dict[tuple[str, int], int] = {}
    records: list[dict] = []
    for pick in picks:
        key = (pick.sample_id, pick.epoch)
        sample = by_key.get(key)
        if sample is None:
            raise Refused(f"{pick.sample_id} epoch {pick.epoch} is not in the log")
        if getattr(sample, "error", None):
            raise Refused(f"{pick.sample_id} epoch {pick.epoch} has a sample error")
        raw = _completion(sample)
        parse = scoring.parse_worker_output(raw)
        if not parse.ok:
            raise Refused(f"{pick.sample_id} epoch {pick.epoch} did not parse (D-R10 C5)")
        case = dev_cases.get(pick.sample_id)
        case_input = _input_text(sample.input)
        if case is None or case["input"] != case_input:
            raise Refused(f"{pick.sample_id}: the log's case file differs from data/dev/dev.jsonl")
        metadata = dict(sample.metadata)
        base_narr = str(parse.parsed["narrative"])
        if pick.rule is None:
            worker_output, narrative, note = raw, base_narr, None
            seeded = None
        else:
            try:
                seeded = apply_rule(pick, base_narr, metadata, case_input)
                worker_output = replace_narrative(raw, base_narr, seeded.narrative)
            except SeedError as e:
                raise Refused(f"{pick.sample_id} epoch {pick.epoch}, rule {pick.rule}: {e}") from e
            narrative, note = seeded.narrative, seeded.note
            if pick.rule in EDIT_COUNT:
                truth = casefile.truth_of(metadata)
                before = code_scores(parse.parsed, truth)
                after = code_scores(scoring.parse_worker_output(worker_output).parsed, truth)
                if before != after:
                    changed = sorted(k for k in before if before[k] != after[k])
                    raise Refused(f"{pick.sample_id} epoch {pick.epoch}, rule {pick.rule}: the edit changes "
                                  f"a code score ({', '.join(changed)})")
        n = seed_n.get(key, 0) + (1 if pick.rule else 0)
        seed_n[key] = n
        file_name = f"{pick.sample_id}-e{pick.epoch}" + (f"-seed-{n}" if pick.rule else "") + ".json"
        parsed = scoring.parse_worker_output(worker_output).parsed
        records.append({
            "sample_id": pick.sample_id,
            "epoch": pick.epoch,
            "log_eval_id": str(log.eval.eval_id),
            "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "worker_output": worker_output,
            "source": pick.source,
            "seed_note": note,
            "stratum": pick.stratum,
            "seed_rule": pick.rule,
            "stratum_basis": pick.basis,
            "seed_edits": list(seeded.edits) if seeded else [],
            "seed_deleted": list(seeded.deleted) if seeded else [],
            "base_worker_output": raw,
            "file_name": file_name,
            # sheet-only fields, popped before the plan is written
            "_narrative": narrative,
            "_case_input": case_input,
            "_case_facts": rf.facts_payload(rf.case_facts(metadata)),
            "_output_facts": rf.facts_payload(rf.output_facts(metadata, parsed)),
        })
    rng = random.Random(f"{shuffle_seed}|codes")
    taken: set[str] = set()
    for r in records:
        r["code"] = record_code(rng, taken)
    order = list(range(len(records)))
    random.Random(f"{shuffle_seed}|order").shuffle(order)
    for pos, i in enumerate(order, start=1):
        records[i]["position"] = pos
    return records


# --- Sheet -----------------------------------------------------------------------------

SHEET_TITLE = "Judge-development grading"
INTRO: tuple[str, ...] = (
    "Grade each record on both sub-dimensions, 0 to 3, against the committed level texts in the rubric panel (on the left, or below the record on a narrow screen).",
    "Read the narrative against the case file and the facts sheet. The facts sheet is the one the grader will "
    "also see. Some values in it come from the answer key and were not shown to the worker. The case file, "
    "collapsed at the bottom, is the case file as the worker saw it.",
    "Records carry a random code and come in a random order. Start the sitting timer before the first record "
    "and end it when you stop. Answers stay in this browser as you go. When you finish, save the grades file "
    "and keep it; the import reads that file. Load grades file puts a saved file back into this page.",
)
KEYS_HELP: tuple[tuple[str, str], ...] = (
    ("j or right arrow", "next record"),
    ("k or left arrow", "previous record"),
    ("u", "next record not fully graded"),
    ("g", "grade grounding with the next digit"),
    ("c", "grade coverage with the next digit"),
    ("0 1 2 3", "grade the selected sub-dimension; after grounding, coverage is selected"),
    ("n", "write a note"),
    ("?", "show or hide this help"),
)
SUBDIM_LABELS = {GROUNDING: "Narrative grounding", COVERAGE: "Narrative coverage"}

_CSS = """
:root{--desk:#E7EDE6;--paper:#FBFDFA;--soft:#F0F4EF;--ink:#17211C;--mute:#56665D;--rule:#C7D4CA;--pencil:#1B6A4D;--hl:#FFE57A;
--narr:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
--fact:"Bahnschrift","DIN Alternate","Avenir Next Condensed","Roboto Condensed","Segoe UI",system-ui,sans-serif;
--ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--desk:#111714;--paper:#19211D;--soft:#202A25;--ink:#E3EBE5;--mute:#97A89E;--rule:#2F3B34;--pencil:#7CD0A8;--hl:#5F5400;color-scheme:dark}}
:root[data-theme="dark"]{--desk:#111714;--paper:#19211D;--soft:#202A25;--ink:#E3EBE5;--mute:#97A89E;--rule:#2F3B34;--pencil:#7CD0A8;--hl:#5F5400;color-scheme:dark}
*{box-sizing:border-box}html,body{overflow-x:hidden}main{min-width:0}
body{margin:0;background:var(--desk);color:var(--ink);font:14px/1.45 var(--ui)}
button,input,select,textarea{font:inherit;color:inherit}
button{cursor:pointer;background:var(--paper);border:1px solid var(--rule);border-radius:3px;padding:5px 10px}
button:hover{border-color:var(--ink)}
button:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
select,textarea{background:var(--paper);border:1px solid var(--rule);border-radius:3px;padding:5px 8px;max-width:100%}
.bar{position:sticky;top:0;z-index:20;background:var(--desk);border-bottom:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:8px 16px}
.bar h1{font:600 17px/1 var(--fact);letter-spacing:.02em;margin:0 6px 0 0}
.prog{display:flex;align-items:center;gap:8px;flex:1 1 160px;min-width:140px}
.prog .track{flex:1;height:6px;background:var(--rule);border-radius:3px;overflow:hidden}
.prog .track i{display:block;height:100%;width:0;background:var(--pencil)}
.prog span,.sit{font-size:12.5px;color:var(--mute);white-space:nowrap}
.bar .primary{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.status{flex:1 1 100%;font-size:12.5px;color:var(--mute)}.status:empty{display:none}
.desk{max-width:1500px;margin:0 auto;padding:12px 16px 80px;display:grid;grid-template-columns:minmax(280px,380px) minmax(0,1fr);gap:18px;align-items:start}
#panel{position:sticky;top:64px;max-height:calc(100vh - 80px);overflow:auto;background:var(--paper);border:1px solid var(--rule);border-radius:3px;padding:12px 14px 16px}
#panel h2{font:600 15px/1.2 var(--fact);margin:14px 0 4px}#panel h2:first-child{margin-top:0}
.scope{font-size:12.5px;color:var(--mute);margin:0 0 6px}
.lv{border-top:1px solid var(--rule);padding:6px 0}
.lv b{font-family:var(--fact);margin-right:6px}
.lv details{font-size:12.5px;color:var(--mute);margin-top:2px}
.card{background:var(--paper);border:1px solid var(--rule);border-radius:3px;padding:14px 16px;margin:0 0 14px;min-width:0}
.card h2{font:600 16px/1.2 var(--fact);margin:0 0 8px}
.intro p{margin:4px 0;font-size:13px}
.narr{font:16px/1.6 var(--narr);margin:4px 0 0;overflow-wrap:anywhere}
.grade{display:grid;grid-template-columns:minmax(150px,max-content) 1fr;gap:8px 12px;align-items:center;margin:10px 0}
.grade .name{font-weight:600}.grade .name.sel{color:var(--pencil)}
.pick{display:flex;gap:6px;flex-wrap:wrap}
.pick button{min-width:42px;font-family:var(--fact);font-weight:600}
.pick button.on{background:var(--pencil);color:var(--paper);border-color:var(--pencil)}
textarea{width:100%;min-height:3.2em}
h3{font:600 13px/1.2 var(--ui);color:var(--mute);margin:14px 0 6px;padding-top:10px;border-top:1px solid var(--rule)}
.fbt{font-size:13px;font-weight:600;margin:10px 0 4px}
.tw{overflow-x:auto;max-width:100%;margin:2px 0 8px}
table{border-collapse:collapse;font-size:12px;font-family:var(--fact)}
th,td{border:1px solid var(--rule);padding:3px 6px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere;min-width:4em}
th{color:var(--mute);font-weight:600}
details{margin:6px 0}summary{cursor:pointer;color:var(--mute)}
pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere;margin:6px 0}
dialog{max-width:440px;border:1px solid var(--rule);border-radius:4px;background:var(--paper);color:var(--ink)}
dialog td,dialog th{font-family:var(--ui);font-size:13px}
@media (max-width:860px){.desk{grid-template-columns:minmax(0,1fr)}#panel{position:static;max-height:none;order:2}}
"""

_JS = r"""
(function(){
"use strict";
var D = JSON.parse(document.getElementById("data").textContent);
var R = D.records, KEY = D.storage_key, DIMS = D.dims;
var state = {grades:{}, sittings:[], pos:0};
var sel = DIMS[0];
function $(id){ return document.getElementById(id); }
function el(tag, cls, text){ var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
function load(){ try { var raw = window.localStorage.getItem(KEY); if (raw) { var s = JSON.parse(raw); if (s && typeof s === "object") { state.grades = s.grades || {}; state.sittings = s.sittings || []; state.pos = s.pos || 0; } } } catch (e) { status("This browser does not keep answers. Save the grades file before you close the page."); } }
function save(){ try { window.localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { status("This browser did not keep the last answer. Save the grades file now."); } }
function status(t){ $("status").textContent = t || ""; }
function pad(n){ return (n < 10 ? "0" : "") + n; }
function isoLocal(d){ var o = -d.getTimezoneOffset(), s = o >= 0 ? "+" : "-"; o = Math.abs(o);
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + "T" + pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()) + s + pad(Math.floor(o / 60)) + ":" + pad(o % 60); }
function g(code){ if (!state.grades[code]) state.grades[code] = {note:""}; return state.grades[code]; }
function full(code){ var x = state.grades[code]; return !!x && DIMS.every(function(d){ return typeof x[d] === "number"; }); }
function progress(){ var n = R.filter(function(r){ return full(r.code); }).length;
  $("bar").style.width = (100 * n / R.length) + "%"; $("count").textContent = n + " of " + R.length + " graded"; }
function table(block){
  var wrap = el("div"); wrap.appendChild(el("div", "fbt", block.title));
  var tw = el("div", "tw"), t = el("table"), tr = el("tr");
  block.columns.forEach(function(c){ tr.appendChild(el("th", null, c)); }); t.appendChild(tr);
  block.rows.forEach(function(row){ var r = el("tr"); row.forEach(function(c){ r.appendChild(el("td", null, c)); }); t.appendChild(r); });
  if (!block.rows.length) { var e = el("tr"), td = el("td", null, "none"); td.colSpan = block.columns.length; e.appendChild(td); t.appendChild(e); }
  tw.appendChild(t); wrap.appendChild(tw); return wrap; }
function setGrade(dim, v){ var r = R[state.pos], x = g(r.code); x[dim] = v; x.at = isoLocal(new Date()); save(); render(); }
function select(dim){ sel = dim; render(); }
function render(){
  var r = R[state.pos], box = $("record"); box.textContent = "";
  var head = el("div", "card");
  head.appendChild(el("h2", null, "Record " + r.code + " (" + (state.pos + 1) + " of " + R.length + ")"));
  head.appendChild(el("div", "fbt", "Narrative"));
  head.appendChild(el("p", "narr", r.narrative));
  DIMS.forEach(function(dim){
    var row = el("div", "grade"), name = el("div", "name" + (sel === dim ? " sel" : ""), D.labels[dim]);
    name.title = dim; row.appendChild(name);
    var pick = el("div", "pick"), x = state.grades[r.code] || {};
    [0,1,2,3].forEach(function(v){ var b = el("button", x[dim] === v ? "on" : null, String(v)); b.type = "button";
      b.title = D.criteria[dim][v]; b.setAttribute("aria-pressed", x[dim] === v ? "true" : "false");
      b.addEventListener("click", function(){ sel = dim; setGrade(dim, v); }); pick.appendChild(b); });
    row.appendChild(pick); head.appendChild(row); });
  var ta = el("textarea"); ta.id = "note"; ta.placeholder = "Note (optional)"; ta.value = (state.grades[r.code] || {}).note || "";
  ta.addEventListener("input", function(){ var x = g(r.code); x.note = ta.value; save(); });
  head.appendChild(ta); box.appendChild(head);
  var facts = el("div", "card"); facts.appendChild(el("h2", null, "Facts sheet"));
  facts.appendChild(el("h3", null, "This output"));
  r.output_facts.forEach(function(b){ facts.appendChild(table(b)); });
  facts.appendChild(el("h3", null, "This case"));
  r.case_facts.forEach(function(b){ facts.appendChild(table(b)); });
  var det = el("details"); det.appendChild(el("summary", null, "Case file, as the worker saw it"));
  det.appendChild(el("pre", null, r.case_input)); facts.appendChild(det); box.appendChild(facts);
  $("jump").value = String(state.pos); progress(); sitting(); }
function go(i){ if (i < 0 || i >= R.length) return; state.pos = i; sel = DIMS[0]; save(); render(); window.scrollTo(0, 0); }
function nextTodo(){ for (var k = 1; k <= R.length; k++) { var i = (state.pos + k) % R.length; if (!full(R[i].code)) { go(i); return; } } status("Every record has both grades."); }
function open(){ var s = state.sittings; return s.length && !s[s.length - 1].end ? s[s.length - 1] : null; }
function sitting(){ var o = open(), t = $("sit");
  if (o) { var m = Math.floor((Date.now() - Date.parse(o.start)) / 60000); t.textContent = "Sitting open, " + m + " min"; }
  else t.textContent = state.sittings.length ? state.sittings.length + " sitting(s) recorded" : "No sitting started"; }
function csvCell(v){ v = v == null ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function download(name, rows){ var text = rows.map(function(r){ return r.map(csvCell).join(","); }).join("\r\n") + "\r\n";
  var url = URL.createObjectURL(new Blob([text], {type:"text/csv"})), a = el("a"); a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(function(){ URL.revokeObjectURL(url); }, 1000); }
function exportGrades(){ var rows = [["record_code"].concat(DIMS).concat(["note", "graded_at"])];
  R.slice().sort(function(a, b){ return a.code < b.code ? -1 : 1; }).forEach(function(r){ var x = state.grades[r.code] || {};
    rows.push([r.code].concat(DIMS.map(function(d){ return typeof x[d] === "number" ? x[d] : ""; })).concat([x.note || "", x.at || ""])); });
  download(D.grades_file, rows);
  var n = R.filter(function(r){ return full(r.code); }).length;
  status(n === R.length ? "Grades file saved." : "Grades file saved. " + (R.length - n) + " record(s) still miss a grade; the import refuses that file."); }
function exportSittings(){ var rows = [["sitting", "start", "end"]];
  state.sittings.forEach(function(s, i){ rows.push([i + 1, s.start, s.end || ""]); }); download(D.sittings_file, rows); status("Sittings file saved."); }
function parseCsv(text){ var rows = [], row = [], cell = "", q = false, i, ch;
  for (i = 0; i < text.length; i++) { ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += ch; }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); } return rows; }
function loadGrades(text){ var rows = parseCsv(text.replace(/^﻿/, "")), head = rows.shift() || [], n = 0;
  var ci = head.indexOf("record_code"), ni = head.indexOf("note"), ai = head.indexOf("graded_at");
  var di = DIMS.map(function(d){ return head.indexOf(d); });
  if (ci < 0 || di.indexOf(-1) >= 0) { status("That file is not a grades file from this page."); return; }
  var codes = {}; R.forEach(function(r){ codes[r.code] = true; });
  rows.forEach(function(row){ var code = row[ci]; if (!codes[code]) return; var x = g(code);
    DIMS.forEach(function(d, k){ var v = (row[di[k]] || "").trim(); if (/^[0-3]$/.test(v)) x[d] = parseInt(v, 10); });
    if (ni >= 0) x.note = row[ni] || ""; if (ai >= 0 && row[ai]) x.at = row[ai]; n++; });
  save(); render(); status("Loaded " + n + " record(s) from the grades file."); }
function start(){ if (open()) { status("A sitting is already open."); return; } state.sittings.push({start:isoLocal(new Date())}); save(); sitting(); status("Sitting started."); }
function end(){ var o = open(); if (!o) { status("No sitting is open."); return; } o.end = isoLocal(new Date()); save(); sitting(); status("Sitting ended."); }
function build(){
  var p = $("panel");
  DIMS.forEach(function(dim){ p.appendChild(el("h2", null, D.labels[dim]));
    p.appendChild(el("p", "scope", D.scope[dim]));
    [3,2,1,0].forEach(function(v){ var lv = el("div", "lv"), head = el("div"); head.appendChild(el("b", null, String(v)));
      head.appendChild(document.createTextNode(D.criteria[dim][v])); lv.appendChild(head);
      var det = el("details"); det.appendChild(el("summary", null, "Example")); det.appendChild(el("div", null, D.examples[dim][v]));
      lv.appendChild(det); p.appendChild(lv); }); });
  p.appendChild(el("p", "scope", "Level texts: icfr_agent_evals/rubric.py, anchors SHA-256 " + D.anchors_sha256 + "."));
  var intro = $("intro"); D.intro.forEach(function(t){ intro.appendChild(el("p", null, t)); });
  var j = $("jump"); R.forEach(function(r, i){ var o = el("option", null, (i + 1) + ". " + r.code); o.value = String(i); j.appendChild(o); });
  j.addEventListener("change", function(){ go(parseInt(j.value, 10)); });
  $("prev").addEventListener("click", function(){ go(state.pos - 1); });
  $("next").addEventListener("click", function(){ go(state.pos + 1); });
  $("todo").addEventListener("click", nextTodo);
  $("start").addEventListener("click", start); $("end").addEventListener("click", end);
  $("save").addEventListener("click", exportGrades); $("save-sittings").addEventListener("click", exportSittings);
  $("help").addEventListener("click", function(){ $("keys").showModal(); });
  $("load").addEventListener("click", function(){ $("file").click(); });
  $("file").addEventListener("change", function(){ var f = $("file").files[0]; if (!f) return;
    var rd = new FileReader(); rd.onload = function(){ loadGrades(String(rd.result)); $("file").value = ""; }; rd.readAsText(f); });
  var kt = $("keytable"); D.keys.forEach(function(k){ var tr = el("tr"); tr.appendChild(el("th", null, k[0])); tr.appendChild(el("td", null, k[1])); kt.appendChild(tr); });
  setInterval(sitting, 30000); }
document.addEventListener("keydown", function(e){
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  var t = e.target, tag = t && t.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") { if (e.key === "Escape" && t.blur) t.blur(); return; }
  var k = e.key, d = $("keys");
  if (d.open) { if (k === "?") d.close(); return; }
  if (k === "j" || k === "ArrowRight") go(state.pos + 1);
  else if (k === "k" || k === "ArrowLeft") go(state.pos - 1);
  else if (k === "u") nextTodo();
  else if (k === "g") select(DIMS[0]);
  else if (k === "c") select(DIMS[1]);
  else if (k === "0" || k === "1" || k === "2" || k === "3") { var dim = sel; setGrade(dim, parseInt(k, 10)); if (dim === DIMS[0]) select(DIMS[1]); }
  else if (k === "n") { var n = $("note"); if (n) n.focus(); }
  else if (k === "?") d.showModal();
  else return;
  e.preventDefault(); });
load(); build(); if (state.pos >= R.length) state.pos = 0; render();
})();
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style></head><body>
<div class="bar">
<h1>__TITLE__</h1>
<div class="prog"><div class="track"><i id="bar"></i></div><span id="count"></span></div>
<button type="button" id="prev" title="k">Previous</button><select id="jump" aria-label="Record"></select><button type="button" id="next" title="j">Next</button>
<button type="button" id="todo" title="u">Next to grade</button>
<button type="button" id="start">Start sitting</button><button type="button" id="end">End sitting</button><span class="sit" id="sit"></span>
<button type="button" id="save" class="primary">Save grades file</button><button type="button" id="save-sittings">Save sittings file</button>
<button type="button" id="load">Load grades file</button><input type="file" id="file" accept=".csv,text/csv" hidden>
<button type="button" id="help" title="?">Keys</button>
<div class="status" id="status" role="status"></div>
</div>
<div class="desk">
<aside id="panel"></aside>
<main><div class="card intro" id="intro"></div><div id="record"></div></main>
</div>
<dialog id="keys"><table id="keytable"></table><form method="dialog"><button>Close</button></form></dialog>
<script type="application/json" id="data">__DATA__</script>
<script>__JS__</script>
</body></html>
"""

GRADES_FILE = "judge-dev-grades.csv"
SITTINGS_FILE = "judge-dev-sittings.csv"


def _embed_json(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def storage_key(records: Sequence[Mapping[str, Any]]) -> str:
    codes = ",".join(sorted(r["code"] for r in records))
    return "judge-dev-sheet-" + hashlib.sha256(codes.encode("utf-8")).hexdigest()[:16]


def sheet_data(records: Sequence[Mapping[str, Any]]) -> dict:
    ordered = sorted(records, key=lambda r: r["position"])
    anchors = rubric.LLM_GRADED_ANCHORS
    return {
        "records": [{
            "code": r["code"],
            "narrative": r["_narrative"],
            "case_input": r["_case_input"],
            "case_facts": r["_case_facts"],
            "output_facts": r["_output_facts"],
        } for r in ordered],
        "dims": list(SUBDIMS),
        "labels": SUBDIM_LABELS,
        "scope": dict(rubric.LLM_GRADED_SCOPE),
        "criteria": {d: {str(lv): anchors[d][lv][0] for lv in LEVELS} for d in SUBDIMS},
        "examples": {d: {str(lv): anchors[d][lv][1] for lv in LEVELS} for d in SUBDIMS},
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "intro": list(INTRO),
        "keys": [list(k) for k in KEYS_HELP],
        "storage_key": storage_key(records),
        "grades_file": GRADES_FILE,
        "sittings_file": SITTINGS_FILE,
    }


def render_sheet(records: Sequence[Mapping[str, Any]]) -> str:
    page = _PAGE.replace("__TITLE__", html.escape(SHEET_TITLE)).replace("__CSS__", _CSS.strip())
    page = page.replace("__JS__", _JS.strip())
    return page.replace("__DATA__", _embed_json(sheet_data(records)))


SEALED_WORDS: tuple[str, ...] = (
    "natural", "seeded", "seed", "stratum", "strata", "epoch", "sample_id", "source", "basis",
    *RULES.keys(),
)


def blindness_hits(sheet: str, records: Sequence[Mapping[str, Any]], log_eval_id: str = "") -> list[str]:
    """Everything sealed that the sheet must not hold. Empty when the sheet is blind."""
    hits: list[str] = []
    low = sheet.lower()
    for w in SEALED_WORDS:
        if re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", low):
            hits.append(f"word {w!r}")
    if re.search(r"dev-(?:hc|std|ac|edge|adv)-", low):
        hits.append("a dev case id prefix")
    if log_eval_id and log_eval_id in sheet:
        hits.append("the log eval id")
    for r in records:
        if r["sample_id"] in sheet:
            hits.append(f"sample id {r['sample_id']}")
        if r.get("seed_note") and r["seed_note"] in sheet:
            hits.append(f"the seed note of {r['code']}")
        if r.get("stratum_basis") and r["stratum_basis"] in sheet:
            hits.append(f"the stratum basis of {r['code']}")
        if r["file_name"] in sheet:
            hits.append(f"file name {r['file_name']}")
    if re.search(r'"(?:faithfulness\.narrative_grounding|completeness\.narrative_coverage)"\s*:\s*\d', sheet):
        hits.append("a level value keyed by sub-dimension")
    return hits


# --- Plan ------------------------------------------------------------------------------


PLAN_FIELDS = (
    "code", "position", "file_name", "sample_id", "epoch", "log_eval_id", "worker_prompt_sha256",
    "source", "seed_rule", "seed_note", "seed_edits", "seed_deleted",
    "stratum", "stratum_basis", "worker_output", "base_worker_output",
)


def plan_payload(records: Sequence[Mapping[str, Any]], *, log_path: Path, log_sha: str, log_eval_id: str,
                 selection: Selection, selection_path: Path, sheet_sha: str) -> dict:
    return {
        "plan_version": PLAN_VERSION,
        "what": "Judge-development set plan (sealed). Do not open before grading is finished.",
        "log_file": log_path.name,
        "log_sha256": log_sha,
        "log_eval_id": log_eval_id,
        "worker_prompt_sha256": WORKER_PROMPT_SHA256,
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "selection_file": selection_path.name,
        "selection_sha256": selection.sha256,
        "shuffle_seed": selection.shuffle_seed,
        "excluded": [{"sample_id": s, "epoch": e, "reason": why} for (s, e), why in sorted(EXCLUDED.items())],
        "rules": dict(RULES),
        "level_counts": {d: {str(lv): n for lv, n in c.items()} for d, c in level_counts(selection.picks).items()},
        "source_counts": {s: sum(1 for r in records if r["source"] == s) for s in SOURCES},
        "sheet_file": SHEET_NAME,
        "sheet_sha256": sheet_sha,
        "records": [{k: r[k] for k in PLAN_FIELDS} for r in sorted(records, key=lambda r: r["position"])],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(log_path: Path, dev_path: Path, selection: Selection) -> tuple[Any, list[dict]]:
    log = _read_log(log_path)
    _check_log(log)
    dev_cases = load_dev_cases(dev_path)
    return log, build_records(log, dev_cases, selection.picks, selection.shuffle_seed)


def build(log_path: Path, out_dir: Path, dev_path: Path, selection_path: Path, force: bool) -> dict:
    if rubric.llm_anchor_sha256() != LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    sheet_path = out_dir / SHEET_NAME
    plan_path = out_dir / SEALED_DIR / PLAN_NAME
    if not force and (sheet_path.exists() or plan_path.exists()):
        raise Refused("the sheet or the plan already exists; pass --force to rebuild")
    selection = load_selection(selection_path)
    log, records = _load_records(log_path, dev_path, selection)
    sheet = render_sheet(records)
    hits = blindness_hits(sheet, records, str(log.eval.eval_id))
    if hits:
        raise Refused("the sheet would show sealed text: " + "; ".join(hits))
    sheet_bytes = sheet.encode("utf-8")
    plan = plan_payload(records, log_path=log_path, log_sha=_sha256_file(log_path), log_eval_id=str(log.eval.eval_id),
                        selection=selection, selection_path=selection_path,
                        sheet_sha=hashlib.sha256(sheet_bytes).hexdigest())
    (out_dir / SEALED_DIR).mkdir(parents=True, exist_ok=True)
    sheet_path.write_bytes(sheet_bytes)
    plan_path.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return plan


# --- Probe -----------------------------------------------------------------------------

PROBE_DIR = "_probe"
PROBE_NAME = "probe-input.json"
PROBE_KEY_NAME = "judge-dev-probe-key.json"
PROBE_WHAT = ("Twenty narratives, each under a random probe code, with the committed grounding scope, coverage "
              "scope and the four grounding level texts. Nothing else: no case file, no facts, no ids, no levels.")


def probe_code(rng: random.Random, taken: set[str]) -> str:
    """P- plus five characters; never the R- form the sheet uses."""
    alphabet = "ABCDEFGHJKMNPQRSTVWXYZ2345689"
    while True:
        code = "P-" + "".join(rng.choice(alphabet) for _ in range(5))
        if code not in taken:
            taken.add(code)
            return code


def probe_payloads(records: Sequence[Mapping[str, Any]], probe_seed: str, selection: Selection) -> tuple[dict, dict]:
    """The probe input (narratives only, under fresh codes, in a fresh order) and its sealed key."""
    rng = random.Random(f"{probe_seed}|probe")
    taken = {r["code"] for r in records}
    order = list(records)
    rng.shuffle(order)
    rows, key = [], []
    for r in order:
        code = probe_code(rng, taken)
        rows.append({"probe_code": code, "narrative": r["_narrative"]})
        key.append({"probe_code": code, "sheet_code": r["code"], "sample_id": r["sample_id"], "epoch": r["epoch"],
                    "source": r["source"], "seed_rule": r["seed_rule"], "stratum": r["stratum"]})
    anchors = rubric.LLM_GRADED_ANCHORS[GROUNDING]
    probe = {
        "what": PROBE_WHAT,
        "scope": {"narrative_grounding": rubric.LLM_GRADED_SCOPE[GROUNDING],
                  "narrative_coverage": rubric.LLM_GRADED_SCOPE[COVERAGE]},
        "grounding_levels": {str(lv): anchors[lv][0] for lv in (3, 2, 1, 0)},
        "records": rows,
    }
    sealed = {
        "what": "Probe key (sealed): probe code to sealed record. Do not open before the grades file is saved.",
        "probe_seed": probe_seed,
        "selection_sha256": selection.sha256,
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "records": key,
    }
    return probe, sealed


def probe_blindness_hits(text: str, records: Sequence[Mapping[str, Any]], log_eval_id: str = "") -> list[str]:
    """The sheet's blindness check, plus: no sheet code and no case file text."""
    hits = blindness_hits(text, records, log_eval_id)
    for r in records:
        if r["code"] in text:
            hits.append(f"sheet code {r['code']}")
        head = str(r.get("_case_input", ""))[:60]
        if head and head in text:
            hits.append(f"case file text of {r['code']}")
    return hits


def build_probe(log_path: Path, out_dir: Path, dev_path: Path, selection_path: Path, force: bool,
                probe_seed: str | None = None) -> dict:
    """Write the probe input and the sealed probe key; never the sheet or the plan."""
    if rubric.llm_anchor_sha256() != LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    probe_path = out_dir / PROBE_DIR / PROBE_NAME
    key_path = out_dir / SEALED_DIR / PROBE_KEY_NAME
    if not force and (probe_path.exists() or key_path.exists()):
        raise Refused("the probe input or its key already exists; pass --force to rewrite")
    selection = load_selection(selection_path)
    log, records = _load_records(log_path, dev_path, selection)
    seed = probe_seed or secrets.token_hex(8)
    probe, sealed = probe_payloads(records, seed, selection)
    text = json.dumps(probe, ensure_ascii=False, indent=2) + "\n"
    hits = probe_blindness_hits(text, records, str(log.eval.eval_id))
    if hits:
        raise Refused("the probe input would show sealed text: " + "; ".join(hits))
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_bytes(text.encode("utf-8"))
    key_path.write_bytes((json.dumps(sealed, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"records": probe["records"], "level_counts": level_counts(selection.picks),
            "source_counts": {s: sum(1 for r in records if r["source"] == s) for s in SOURCES}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the judge-development sealed plan and grading sheet.")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    ap.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--probe", action="store_true", help="write only the probe input and its sealed key")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return EXIT_REFUSED
    print(f"module: {icfr_agent_evals.__file__}")
    try:
        if args.probe:
            out = build_probe(args.log, args.out, args.dev, args.selection, args.force)
        else:
            out = build(args.log, args.out, args.dev, args.selection, args.force)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"records: {len(out['records'])}")
    print("sources: " + ", ".join(f"{k} {v}" for k, v in out["source_counts"].items()))
    for d, c in out["level_counts"].items():
        print(f"{d}: " + ", ".join(f"level {lv} x{n}" for lv, n in c.items()))
    if not args.probe:
        print(f"sheet sha256: {out['sheet_sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
