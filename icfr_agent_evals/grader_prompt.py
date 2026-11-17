"""The grader prompt for the two LLM-graded sub-dimensions, and the parser of its reply.

Pre-registration section 6 (LLM-graded sub-dimensions) and section 7 (judge-trust
gates; judge remedy order step 1). Decision D-R10 C7: the grader and the human see one
shared facts sheet. So the grader reads exactly what the author read when grading the
judge-development set (scripts/judge_dev_build.py): the narrative, the facts sheet
(review_facts.case_facts and output_facts, the same blocks under the same titles), the
case file as the worker saw it, and the committed scope and level texts of each
sub-dimension (rubric.LLM_GRADED_SCOPE and LLM_GRADED_ANCHORS, unchanged).

The bias audit of section 7 perturbs the rubric-option order (reversed, or one seeded
permutation), the score labels (0 to 3 shown as D, C, B, A) and the order of the
dimensions in this shared prompt. build_messages takes all three as parameters, and
parse_reply maps a label back to its 0 to 3 level, so a perturbed pass is scored on the
same scale. Each reply gives, per sub-dimension, a short reason, the level label and a
low-confidence flag; section 7's per-case defense routes a flagged (sample, epoch) to
requires_human_review.

PROMPT_VERSION names the wording. Any edit to the instructions or to the rubric texts
changes prompt_sha256(), which every judge-development round records. Step 1 iterates
this wording for at most 3 rounds on the judge-development set, each round logged with
the disagreement pattern it targets (section 7).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import rubric
from .review_facts import FactBlock

PROMPT_VERSION = "gp-1"
LEVELS: tuple[int, ...] = (0, 1, 2, 3)
DIMENSIONS: tuple[str, ...] = tuple(rubric.LLM_GRADED_SUBDIMENSIONS)
DIMENSION_NAMES: dict[str, str] = {
    "faithfulness.narrative_grounding": "Narrative grounding",
    "completeness.narrative_coverage": "Narrative coverage",
}
assert set(DIMENSION_NAMES) == set(DIMENSIONS)
DEFAULT_LABELS: dict[int, str] = {lv: str(lv) for lv in LEVELS}
RELABEL_DCBA: dict[int, str] = {0: "D", 1: "C", 2: "B", 3: "A"}  # section 7 (b)

SYSTEM_TEXT = (
    "You grade one narrative from a crypto reserve reconciliation commentary against a fixed rubric. "
    "You are not the author of the narrative and you do not rewrite it. You grade it on each "
    "sub-dimension the instructions name, using only that sub-dimension's scope and level texts, "
    "and you answer in the JSON format the instructions give, with nothing before or after it."
)

INSTRUCTIONS = (
    "Grade the narrative below on each sub-dimension listed under RUBRIC, from its level texts. "
    "Read the narrative against the case file and the facts sheet. Some values in the facts sheet come "
    "from the answer key and were not shown to the writer of the narrative. The case file is the case "
    "file as the writer saw it.",
    "Grade each sub-dimension on its own scope only. A claim or an element outside a sub-dimension's "
    "scope does not move that sub-dimension's level. Pick the level whose text describes the narrative; "
    "the examples illustrate a level and are not the only way to reach it.",
    "Set low_confidence to true for a sub-dimension when the case file, the facts sheet or the level "
    "texts leave the right level unclear between two levels; otherwise false.",
)


class GraderReplyError(ValueError):
    """A grader reply that does not hold one valid level per sub-dimension."""


@dataclass(frozen=True)
class PromptSettings:
    """The perturbable presentation of the rubric (section 7 bias audit)."""

    option_order: tuple[int, ...] = (0, 1, 2, 3)   # order the level texts are listed in
    labels: Mapping[int, str] | None = None        # level -> label shown; default "0".."3"
    dimension_order: tuple[str, ...] = DIMENSIONS  # order the sub-dimensions are listed and answered in

    def label_map(self) -> dict[int, str]:
        return dict(self.labels) if self.labels else dict(DEFAULT_LABELS)

    def check(self) -> None:
        if sorted(self.option_order) != list(LEVELS):
            raise ValueError(f"option_order must be a permutation of {LEVELS}, got {self.option_order}")
        labels = self.label_map()
        if sorted(labels) != list(LEVELS) or len({str(v).casefold() for v in labels.values()}) != len(LEVELS):
            raise ValueError("labels must give each level 0 to 3 one distinct label, in any letter case")
        if any(not str(v).strip() or re.search(r'["\\\s]', str(v)) for v in labels.values()):
            raise ValueError("a label is one short token with no quote, backslash or space")
        if sorted(self.dimension_order) != sorted(DIMENSIONS):
            raise ValueError(f"dimension_order must be a permutation of {DIMENSIONS}")


def render_block(block: FactBlock) -> str:
    """A fact block as a plain-text table: title, header row, one line per row."""
    lines = [f"### {block.title}", " | ".join(block.columns)]
    lines += [" | ".join(cell.replace("\n", " ") for cell in row) for row in block.rows]
    if not block.rows:
        lines.append("(none)")
    return "\n".join(lines)


def rubric_text(settings: PromptSettings) -> str:
    labels = settings.label_map()
    parts = []
    for dim in settings.dimension_order:
        parts.append(f"## {DIMENSION_NAMES[dim]} (key {dim})")
        parts.append(f"Scope: {rubric.LLM_GRADED_SCOPE[dim]}")
        parts.append("Levels:")
        for lv in settings.option_order:
            criterion, example = rubric.LLM_GRADED_ANCHORS[dim][lv]
            parts.append(f"- Level {labels[lv]}: {criterion}\n  Example: {example}")
    return "\n".join(parts)


def answer_format(settings: PromptSettings) -> str:
    labels = settings.label_map()
    allowed = ", ".join(f'"{labels[lv]}"' for lv in settings.option_order)
    body = ",\n".join(
        f'  "{dim}": {{"reason": "<one or two sentences>", "level": <one of {allowed}>, "low_confidence": <true or false>}}'
        for dim in settings.dimension_order)
    return "Answer with exactly this JSON object, the level as a string:\n{\n" + body + "\n}"


def build_messages(narrative: str, case_input: str, case_blocks: Sequence[FactBlock],
                   output_blocks: Sequence[FactBlock], settings: PromptSettings = PromptSettings()) -> list[dict]:
    """The grader's messages for one narrative, as role/content dicts."""
    settings.check()
    facts = "\n\n".join(render_block(b) for b in (*case_blocks, *output_blocks))
    user = "\n\n".join([
        "\n\n".join(INSTRUCTIONS),
        "# RUBRIC\n" + rubric_text(settings),
        "# NARRATIVE\n" + narrative,
        "# FACTS SHEET\n" + facts,
        "# CASE FILE\n" + case_input,
        "# ANSWER\n" + answer_format(settings),
    ])
    return [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": user}]


def to_chat_messages(messages: Sequence[Mapping[str, str]]) -> list[Any]:
    """The role/content dicts as Inspect chat messages."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    out = []
    for m in messages:
        cls = ChatMessageSystem if m["role"] == "system" else ChatMessageUser
        out.append(cls(content=m["content"]))
    return out


def _json_object(text: str) -> dict:
    s = (text or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
    if fence:
        s = fence.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise GraderReplyError(f"the reply is not one JSON object ({e.msg})") from e
    if not isinstance(obj, dict):
        raise GraderReplyError("the reply is not a JSON object")
    return obj


def parse_reply(text: str, settings: PromptSettings = PromptSettings()) -> dict[str, dict[str, Any]]:
    """Per sub-dimension: level (0 to 3), low_confidence (bool) and reason (str). Strict:
    the reply is one JSON object, bare or wrapped in one code fence; it holds exactly the
    sub-dimension keys, each with a label the prompt offered and a boolean flag. A label
    matches in any letter case ("b" reads as "B"), and nothing else is loosened: no
    padding, no other label form. Anything else is a GraderReplyError (the reply is not
    scored, and the grader is not asked again)."""
    obj = _json_object(text)
    if set(obj) != set(DIMENSIONS):
        raise GraderReplyError(f"the reply's keys {sorted(obj)} are not the sub-dimensions {sorted(DIMENSIONS)}")
    back = {str(v).casefold(): lv for lv, v in settings.label_map().items()}
    out: dict[str, dict[str, Any]] = {}
    for dim in DIMENSIONS:
        entry = obj[dim]
        if not isinstance(entry, dict) or set(entry) - {"reason", "level", "low_confidence"}:
            raise GraderReplyError(f"{dim}: not an object with reason, level and low_confidence")
        label = entry.get("level")
        if isinstance(label, bool) or str(label).casefold() not in back:
            raise GraderReplyError(f"{dim}: level {label!r} is not one of the offered labels")
        flag = entry.get("low_confidence")
        if not isinstance(flag, bool):
            raise GraderReplyError(f"{dim}: low_confidence is not true or false")
        out[dim] = {"level": back[str(label).casefold()], "low_confidence": flag, "reason": str(entry.get("reason") or "")}
    return out


def prompt_sha256() -> str:
    """SHA-256 of the wording that is not case content: the version, the system text, the
    instructions and the rubric texts the prompt shows (rubric.llm_anchor_sha256 covers
    the anchors and scopes)."""
    payload = json.dumps({"version": PROMPT_VERSION, "system": SYSTEM_TEXT, "instructions": list(INSTRUCTIONS),
                          "answer_format": answer_format(PromptSettings()), "anchors": rubric.llm_anchor_sha256()},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
