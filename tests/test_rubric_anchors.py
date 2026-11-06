"""Level anchors for the two LLM-graded sub-dimensions (rubric lock, section 13 rows 205 and 226).

Shape checks on rubric.LLM_GRADED_ANCHORS, LLM_GRADED_SCOPE and BINARY_CHECKS, and the content
hash llm_anchor_sha256(): a 64-hex string, stable across calls, and moved by any anchor edit
(computed on a modified copy; the module is never edited). No model is called.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

from icfr_agent_evals import rubric

SUBDIMENSIONS = {"faithfulness.narrative_grounding", "completeness.narrative_coverage"}


def _hash(anchors, scope, binary_checks) -> str:
    payload = json.dumps(
        {"anchors": anchors, "scope": scope, "binary_checks": binary_checks},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_subdimension_names_match_the_llm_graded_list():
    assert set(rubric.LLM_GRADED_SUBDIMENSIONS) == SUBDIMENSIONS
    assert set(rubric.LLM_GRADED_ANCHORS) == set(rubric.LLM_GRADED_SUBDIMENSIONS)
    assert set(rubric.LLM_GRADED_SCOPE) == set(rubric.LLM_GRADED_SUBDIMENSIONS)


def test_every_subdimension_has_levels_0_to_3_with_criterion_and_example():
    for name, levels in rubric.LLM_GRADED_ANCHORS.items():
        assert sorted(levels) == [0, 1, 2, 3], name
        for level, entry in levels.items():
            assert isinstance(entry, tuple) and len(entry) == 2, (name, level)
            criterion, example = entry
            assert isinstance(criterion, str) and criterion.strip(), (name, level)
            assert isinstance(example, str) and example.strip(), (name, level)
    for name, text in rubric.LLM_GRADED_SCOPE.items():
        assert isinstance(text, str) and text.strip(), name


def test_binary_checks_is_a_dict_tied_to_llm_graded_subdimensions():
    assert isinstance(rubric.BINARY_CHECKS, dict)
    assert set(rubric.BINARY_CHECKS.values()) <= set(rubric.LLM_GRADED_SUBDIMENSIONS)


def test_anchor_hash_is_64_hex_and_stable():
    first = rubric.llm_anchor_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert rubric.llm_anchor_sha256() == first
    assert _hash(rubric.LLM_GRADED_ANCHORS, rubric.LLM_GRADED_SCOPE, rubric.BINARY_CHECKS) == first


def test_anchor_hash_moves_when_an_anchor_text_changes():
    base = rubric.llm_anchor_sha256()
    anchors = copy.deepcopy(rubric.LLM_GRADED_ANCHORS)
    criterion, example = anchors["faithfulness.narrative_grounding"][2]
    anchors["faithfulness.narrative_grounding"][2] = (criterion + " (edited)", example)
    assert _hash(anchors, rubric.LLM_GRADED_SCOPE, rubric.BINARY_CHECKS) != base
    anchors = copy.deepcopy(rubric.LLM_GRADED_ANCHORS)
    criterion, example = anchors["completeness.narrative_coverage"][0]
    anchors["completeness.narrative_coverage"][0] = (criterion, example + " ")
    assert _hash(anchors, rubric.LLM_GRADED_SCOPE, rubric.BINARY_CHECKS) != base
    assert rubric.llm_anchor_sha256() == base  # the module itself is unchanged
