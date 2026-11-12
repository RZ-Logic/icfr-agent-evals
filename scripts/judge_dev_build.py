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
seeded record, one mechanical rule; also the cause list and the shuffle seed. Reading
this code, or the tests, does not say which records are seeded or what their targets are. Seeds edit only the narrative field of the worker's JSON output;
every other byte of the completion is kept. No prose written by a model enters a seed:
an omission seed deletes text by a written rule, and an addition seed inserts one
sentence built from a fixed template (TEMPLATES) and the sealed cause list, or from a
case note's own words (adoption).

Files written under --out:
  judge-dev-sheet.html       the only file the owner opens: one record at a time, blind
  _sealed/judge-dev-plan.json the plan: per record the code, sample id, epoch, source,
                              seed note, stratum, worker output; read by the import

The sheet never shows: whether a record is natural or seeded, the stratum, the seed
note, the sample id, the epoch, the eval id or any score. Records appear under a random
code, in an order shuffled by the recorded shuffle seed. Same inputs give the same bytes.

Exit codes:
  0  written; stdout prints totals only
  3  refused (nothing written): a missing or unreadable log, a log that is not a
     successful dev-split run, a prompt hash other than the locked one, a selected
     output that does not parse or is excluded, a missing or malformed selection input,
     a seed whose rule does not apply to its base output, a cause or phrase that
     collides with a dev case file, the anchors or a pilot output, a template whose
     fixed words occur in the anchors, a selection that misses the stratification, an
     existing output without --force, or a sheet that would show sealed text
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
import sys
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
PLAN_VERSION = "1"
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

# --- Seed material ----------------------------------------------------------------------

# The cause list is sealed (in the selection input), used once each in selection order.
# The build refuses any cause, or content word of one, that appears in a dev case file,
# the anchors or a pilot output (collision_hits).
STOPWORDS = frozenset({"a", "an", "the", "and", "of"})
# Assets whose chain has staking: no staking-adjacent cause may land on one of their cases.
STAKING_ASSETS = frozenset({"SOL", "ADA", "ETH"})
STAKING_WORDS = frozenset({
    "stake", "staking", "staked", "validator", "slashing", "slash", "slashed", "delegation", "delegator",
    "delegated", "reward", "unbonding", "bonding", "epoch", "pool",
})
# Words that give a cause a direction (money in or money out). A cause given for the whole
# variance (A-WHOLE) must have none, so it fits a surplus and a shortfall alike.
DIRECTIONAL_WORDS = frozenset({
    "loss", "gain", "profit", "deficit", "excess", "refund", "reimbursement", "penalty", "fee", "bonus",
    "cashback", "chargeback", "rebate", "overpayment", "underpayment", "windfall", "donation", "theft",
    "hack", "write-off", "writedown", "income", "expense", "payout", "deposit", "withdrawal",
})

# Addition templates. {amount} is the remainder and {variance} the variance, written as
# the base narrative writes money ("$" or "USD "), and found in it verbatim. No fixed word
# of a template may occur in the anchors or scopes of rubric.py (template_anchor_hits),
# except the words in TEMPLATE_SHARED_WORDS, which every narrative already uses.
TEMPLATES: dict[str, str] = {
    "A-HEDGE": "Possibly {cause} lies behind the unexplained {amount}.",
    "A-PART": "The unexplained {amount} arose from {cause}.",
    "A-PART2": "The unexplained {amount} arose from {cause1} plus {cause2}.",
    "A-WHOLE": "The {variance} {direction} arose entirely from {cause}.",
    "A-ADOPT": "{note_words}.",
}
TEMPLATE_SHARED_WORDS = frozenset({"the", "from", "unexplained", "surplus", "shortfall", "usd"})
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
    "A-HEDGE": (
        f"Addition at the overstated level: the narrative's own hedged guess. Insert the sentence "
        f"'{TEMPLATES['A-HEDGE']}' directly after the last sentence that contains the word "
        f"unexplained. Needs a remainder other than zero."
    ),
    "A-PART": (
        f"Addition of one unsupported claim about part of the variance, stated as fact. Insert "
        f"'{TEMPLATES['A-PART']}' directly after the last sentence that contains the word "
        f"unexplained. Needs a case with both an explained and an unexplained part."
    ),
    "A-PART2": (
        f"Addition of two unsupported claims, stated as fact. Insert '{TEMPLATES['A-PART2']}' "
        f"directly after the last sentence that contains the word unexplained. Needs a case with "
        f"both an explained and an unexplained part."
    ),
    "A-WHOLE": (
        f"Addition of one unsupported claim given as the cause of the whole variance, stated as "
        f"fact. Insert '{TEMPLATES['A-WHOLE']}' directly after the last sentence that contains "
        f"the word unexplained. The cause has no direction word, so it fits a surplus and a "
        f"shortfall alike."
    ),
    "A-ADOPT": (
        "Addition by adoption. Insert, as the narrative's own sentence, a run of words copied "
        "verbatim from one case note that gives a cause or a conclusion, with its first letter "
        "capitalized and a full stop, directly after the last sentence that contains the word "
        "unexplained. The base narrative must not mention the note's subject."
    ),
}
RULE_SOURCE = {
    "O-REM": "seeded_omission", "O-BAND": "seeded_omission", "O-BAND+O-REM": "seeded_omission",
    "A-HEDGE": "seeded_addition", "A-PART": "seeded_addition", "A-PART2": "seeded_addition",
    "A-WHOLE": "seeded_addition", "A-ADOPT": "seeded_addition",
}
CAUSES_PER_RULE = {"A-HEDGE": 1, "A-PART": 1, "A-PART2": 2, "A-WHOLE": 1, "A-ADOPT": 0}


@dataclass(frozen=True)
class Pick:
    sample_id: str
    epoch: int
    grounding: int          # provisional target level (stratum), sealed
    coverage: int           # provisional target level (stratum), sealed
    rule: str | None = None  # None for a natural record
    basis: str = ""          # why Claude proposes this stratum; sealed
    note_words: str | None = None   # A-ADOPT: the verbatim run of note words
    absent_word: str | None = None  # A-ADOPT: the note's subject, absent from the base narrative

    @property
    def source(self) -> str:
        return "natural" if self.rule is None else RULE_SOURCE[self.rule]

    @property
    def stratum(self) -> dict[str, int]:
        return {GROUNDING: self.grounding, COVERAGE: self.coverage}


@dataclass(frozen=True)
class Selection:
    """The sealed build input: the picks, the cause list and the shuffle seed."""
    picks: tuple[Pick, ...]
    causes: tuple[str, ...]
    shuffle_seed: int
    sha256: str


class Refused(Exception):
    """An input the build will not use; exit 3, nothing written."""


# --- Sealed selection input -------------------------------------------------------------

_PICK_KEYS = ("sample_id", "epoch", "grounding", "coverage", "rule", "basis", "note_words", "absent_word")


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
    causes = doc.get("causes")
    rows = doc.get("picks")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise Refused("the selection input has no integer shuffle_seed")
    if not isinstance(causes, list) or not all(isinstance(c, str) and c.strip() for c in causes):
        raise Refused("the selection input's causes are not a list of phrases")
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
                          row.get("basis") or "", row.get("note_words"), row.get("absent_word")))
    return Selection(tuple(picks), tuple(causes), seed, hashlib.sha256(data).hexdigest())


# --- Stratification -------------------------------------------------------------------


def level_counts(picks: Sequence[Pick]) -> dict[str, dict[int, int]]:
    counts = {d: {lv: 0 for lv in LEVELS} for d in SUBDIMS}
    for p in picks:
        for d, lv in p.stratum.items():
            counts[d][lv] += 1
    return counts


def selection_problems(picks: Sequence[Pick], causes: Sequence[str]) -> list[str]:
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
        if p.rule is not None and p.rule not in RULES:
            problems.append(f"{p.sample_id}: unknown rule {p.rule}")
        if p.rule == "A-ADOPT" and not (p.note_words and p.absent_word):
            problems.append(f"{p.sample_id}: an adoption seed needs its note words and subject word")
    for case, n in sorted(per_case.items()):
        if n > MAX_PER_CASE:
            problems.append(f"{case} appears {n} times, more than {MAX_PER_CASE}")
    for d, by_level in level_counts(picks).items():
        for lv, n in by_level.items():
            if n < MIN_PER_LEVEL:
                problems.append(f"{d} level {lv} has {n} records, fewer than {MIN_PER_LEVEL}")
    used = sum(CAUSES_PER_RULE.get(p.rule or "", 0) for p in picks)
    if used != len(causes):
        problems.append(f"the seeds need {used} causes and the cause list holds {len(causes)}")
    if len(set(causes)) != len(causes):
        problems.append("a cause is listed twice")
    return problems


# --- Collision check ------------------------------------------------------------------


def content_words(phrase: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", phrase.lower()) if w not in STOPWORDS]


def collision_hits(phrases: Sequence[str], texts: Mapping[str, str]) -> list[str]:
    """Each phrase, and each content word of it (with an s or es plural), searched as a
    whole word, case-insensitively, in every named text. Returns 'phrase | word | text'."""
    hits: list[str] = []
    lowered = {name: t.lower() for name, t in texts.items()}
    for phrase in phrases:
        needles = [phrase.lower()] + content_words(phrase)
        for needle in dict.fromkeys(needles):
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(needle) + r"(?:s|es)?(?![a-z0-9])")
            for name, t in lowered.items():
                if pat.search(t):
                    hits.append(f"{phrase} | {needle} | {name}")
    return hits


def anchor_text() -> str:
    """The committed anchors and scopes, as one text, for the collision check."""
    parts: list[str] = list(rubric.LLM_GRADED_SCOPE.values())
    for levels in rubric.LLM_GRADED_ANCHORS.values():
        for crit, example in levels.values():
            parts.extend([crit, example])
    return "\n".join(parts)


def template_fixed_words(template: str) -> list[str]:
    """The fixed words of a template: every word outside a {placeholder}, lower case,
    minus TEMPLATE_SHARED_WORDS."""
    bare = re.sub(r"\{[a-z_0-9]+\}", " ", template)
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", bare.lower())
    return [w for w in dict.fromkeys(words) if w not in TEMPLATE_SHARED_WORDS]


def template_anchor_hits(templates: Mapping[str, str], anchors: str) -> list[str]:
    """Each fixed word of each template, searched as a whole word (with an s or es plural)
    in the anchor text. Returns 'rule | word'; empty when no template echoes the anchors."""
    hits: list[str] = []
    low = anchors.lower()
    for rule, template in templates.items():
        for w in template_fixed_words(template):
            if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?:s|es)?(?![a-z0-9])", low):
                hits.append(f"{rule} | {w}")
    return hits


def cause_fit_problem(cause: str, rule: str, asset: str) -> str | None:
    """Why a cause does not fit this rule on this asset's case, or None."""
    words = set(content_words(cause))
    if asset in STAKING_ASSETS and words & STAKING_WORDS:
        return f"the cause {cause!r} is staking-adjacent and the case asset is {asset}"
    if rule == "A-WHOLE" and words & DIRECTIONAL_WORDS:
        return f"the cause {cause!r} has a direction and is given for the whole variance"
    return None


# --- Narrative edits ------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])(?<![ (][A-Z]\.)(?<!\bvs\.)\s+(?=[A-Z$(])")
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


def insert_after_remainder(narrative: str, sentence: str) -> str:
    """Insert one sentence directly after the last sentence that contains 'unexplained'."""
    sentences = split_sentences(narrative)
    at = None
    for i, s in enumerate(sentences):
        if _UNEXPLAINED.search(s):
            at = i
    if at is None:
        raise SeedError("no sentence contains the word unexplained")
    sentences.insert(at + 1, sentence)
    return " ".join(sentences)


def note_texts(metadata: Mapping[str, Any]) -> list[str]:
    return [text for _, _, text in rf._note_rows(metadata)]


@dataclass
class SeedResult:
    narrative: str
    note: str
    inserted: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)


def apply_rule(pick: Pick, narrative: str, metadata: Mapping[str, Any], causes: list[str]) -> SeedResult:
    """Apply the pick's rule to the base narrative. `causes` is consumed in order."""
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

    if rule == "A-ADOPT":
        words = pick.note_words or ""
        notes = note_texts(metadata)
        source = next((n for n in notes if words in n), None)
        if source is None:
            raise SeedError("the note words are not a verbatim run of any case note")
        if (pick.absent_word or "").lower() in narrative.lower():
            raise SeedError(f"the base narrative mentions {pick.absent_word!r}")
        sentence = TEMPLATES[rule].format(note_words=words[0].upper() + words[1:])
        used: list[str] = []
    else:
        need = CAUSES_PER_RULE[rule]
        if len(causes) < need:
            raise SeedError("the cause list is used up")
        used = [causes.pop(0) for _ in range(need)]
        for c in used:
            problem = cause_fit_problem(c, rule, truth.asset)
            if problem:
                raise SeedError(problem)
        if rule in ("A-PART", "A-PART2") and (abs(truth.explained_usd) < 0.005 or abs(truth.unexplained_usd) < 0.005):
            raise SeedError(f"{rule} needs both an explained and an unexplained part")
        if rule == "A-HEDGE" and abs(truth.unexplained_usd) < 0.005:
            raise SeedError("A-HEDGE needs a remainder other than zero")
        fill: dict[str, str] = {}
        if rule in ("A-HEDGE", "A-PART", "A-PART2"):
            fill["amount"] = money(truth.unexplained_usd, narrative)
        if rule == "A-WHOLE":
            fill["variance"] = money(truth.variance_usd, narrative)
            fill["direction"] = truth.direction
        if need == 1:
            fill["cause"] = used[0]
        else:
            fill["cause1"], fill["cause2"] = used
        sentence = TEMPLATES[rule].format(**fill)
    text = insert_after_remainder(narrative, sentence)
    detail = f"Template: {json.dumps(TEMPLATES[rule])}." if rule != "A-ADOPT" else (
        f"Note words copied from the case note {json.dumps(source, ensure_ascii=False)}.")
    cause_part = f" Causes from the sealed cause list:{', '.join(used)}." if used else ""
    note = (f"Rule {rule}. {RULES[rule]} {detail}{cause_part} Inserted: {json.dumps(sentence, ensure_ascii=False)}. "
            "Nothing was deleted. Only the narrative field changed; the rest of the output is verbatim.")
    return SeedResult(text, note, inserted=[sentence], causes=used)


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


def build_records(log: Any, dev_cases: Mapping[str, dict], picks: Sequence[Pick], cause_list: Sequence[str],
                  shuffle_seed: int) -> list[dict]:
    problems = selection_problems(picks, cause_list)
    if problems:
        raise Refused("selection: " + "; ".join(problems))
    hits = template_anchor_hits(TEMPLATES, anchor_text())
    if hits:
        raise Refused("template words found in the anchors: " + "; ".join(hits))
    texts = {cid: row["input"] for cid, row in dev_cases.items()}
    texts.update({f"{cid} (metadata)": json.dumps(row["metadata"], ensure_ascii=False) for cid, row in dev_cases.items()})
    texts["rubric.py anchors and scopes"] = anchor_text()
    hits = collision_hits(cause_list, texts)
    if hits:
        raise Refused("cause collisions: " + "; ".join(hits))
    by_key = {(str(s.id), int(s.epoch)): s for s in log.samples}
    completions = {f"{sid} epoch {ep}": _completion(s) for (sid, ep), s in by_key.items()}
    hits = collision_hits(cause_list, completions)
    if hits:
        raise Refused("cause found in a pilot output: " + "; ".join(hits))

    causes = list(cause_list)
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
                seeded = apply_rule(pick, base_narr, metadata, causes)
                worker_output = replace_narrative(raw, base_narr, seeded.narrative)
            except SeedError as e:
                raise Refused(f"{pick.sample_id} epoch {pick.epoch}, rule {pick.rule}: {e}") from e
            narrative, note = seeded.narrative, seeded.note
        if seeded is not None and seeded.inserted:
            if pick.rule != "A-ADOPT":
                for s in seeded.inserted:
                    for cid, t in texts.items():
                        if s.lower() in t.lower():
                            raise Refused(f"inserted sentence found in {cid}")
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
            "seed_inserted": list(seeded.inserted) if seeded else [],
            "seed_deleted": list(seeded.deleted) if seeded else [],
            "seed_causes": list(seeded.causes) if seeded else [],
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
    "source", "seed_rule", "seed_note", "seed_inserted", "seed_deleted", "seed_causes",
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
        "causes": list(selection.causes),
        "templates": dict(TEMPLATES),
        "rules": dict(RULES),
        "level_counts": {d: {str(lv): n for lv, n in c.items()} for d, c in level_counts(selection.picks).items()},
        "source_counts": {s: sum(1 for r in records if r["source"] == s) for s in SOURCES},
        "sheet_file": SHEET_NAME,
        "sheet_sha256": sheet_sha,
        "records": [{k: r[k] for k in PLAN_FIELDS} for r in sorted(records, key=lambda r: r["position"])],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(log_path: Path, out_dir: Path, dev_path: Path, selection_path: Path, force: bool) -> dict:
    if rubric.llm_anchor_sha256() != LOCKED_ANCHOR_SHA256:
        raise Refused("rubric.py anchors differ from the committed llm_anchor_sha256")
    sheet_path = out_dir / SHEET_NAME
    plan_path = out_dir / SEALED_DIR / PLAN_NAME
    if not force and (sheet_path.exists() or plan_path.exists()):
        raise Refused("the sheet or the plan already exists; pass --force to rebuild")
    selection = load_selection(selection_path)
    log = _read_log(log_path)
    _check_log(log)
    dev_cases = load_dev_cases(dev_path)
    records = build_records(log, dev_cases, selection.picks, selection.causes, selection.shuffle_seed)
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the judge-development sealed plan and grading sheet.")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    ap.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--force", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return EXIT_REFUSED
    print(f"module: {icfr_agent_evals.__file__}")
    try:
        plan = build(args.log, args.out, args.dev, args.selection, args.force)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"records: {len(plan['records'])}")
    print("sources: " + ", ".join(f"{k} {v}" for k, v in plan["source_counts"].items()))
    for d, c in plan["level_counts"].items():
        print(f"{d}: " + ", ".join(f"level {lv} x{n}" for lv, n in c.items()))
    print(f"sheet sha256: {plan['sheet_sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
