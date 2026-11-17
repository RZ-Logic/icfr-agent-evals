"""The grader prompt (icfr_agent_evals.grader_prompt) and the judge-development round runner
(scripts/judge_dev_round.py), judge remedy order step 1 of pre-registration section 7.

Keyless: the only model is Inspect's mockllm; no grader call reaches a provider.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import rubric
from icfr_agent_evals.review_facts import FactBlock

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS


def _load_round():
    spec = importlib.util.spec_from_file_location("judge_dev_round", REPO / "scripts" / "judge_dev_round.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_dev_round"] = mod
    spec.loader.exec_module(mod)
    return mod


jr = _load_round()

BLOCK = FactBlock(block_id="answer_key", title="Answer key values", columns=("field", "value"),
                  rows=(("band", "M2"),), verbatim_columns=frozenset())


def reply(g=3, c=3, *, low=False, labels=None):
    lab = labels or gp.DEFAULT_LABELS
    return json.dumps({G: {"reason": "r", "level": lab[g], "low_confidence": low},
                       C: {"reason": "r", "level": lab[c], "low_confidence": low}})


# --- the prompt -----------------------------------------------------------------------------

def test_prompt_holds_what_the_author_saw():
    msgs = gp.build_messages("THE NARRATIVE", "THE CASE FILE", [BLOCK], [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    for dim in gp.DIMENSIONS:
        assert rubric.LLM_GRADED_SCOPE[dim] in user
        for lv in gp.LEVELS:
            criterion, example = rubric.LLM_GRADED_ANCHORS[dim][lv]
            assert criterion in user and example in user
    for text in ("THE NARRATIVE", "THE CASE FILE", "### Answer key values", "band | M2"):
        assert text in user


def test_option_order_labels_and_dimension_order_are_parameters():
    settings = gp.PromptSettings(option_order=(3, 2, 1, 0), labels=gp.RELABEL_DCBA, dimension_order=(C, G))
    user = gp.build_messages("n", "c", [], [], settings)[1]["content"]
    assert user.index("Level A:") < user.index("Level B:") < user.index("Level C:") < user.index("Level D:")
    assert user.index("Narrative coverage (key") < user.index("Narrative grounding (key")
    assert '"level": <one of "A", "B", "C", "D">' in user
    parsed = gp.parse_reply(reply(3, 0, labels=gp.RELABEL_DCBA), settings)
    assert parsed[G]["level"] == 3 and parsed[C]["level"] == 0


@pytest.mark.parametrize("bad", [
    dict(option_order=(0, 1, 2)), dict(option_order=(0, 1, 2, 2)),
    dict(labels={0: "A", 1: "A", 2: "B", 3: "C"}), dict(labels={0: "a b", 1: "c", 2: "d", 3: "e"}),
    dict(dimension_order=(G,)),
])
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        gp.build_messages("n", "c", [], [], gp.PromptSettings(**bad))


def test_parse_reply_reads_levels_flags_and_reasons():
    out = gp.parse_reply(reply(1, 2, low=True))
    assert out[G] == {"level": 1, "low_confidence": True, "reason": "r"}
    assert gp.parse_reply("```json\n" + reply(0, 3) + "\n```")[C]["level"] == 3


@pytest.mark.parametrize("text", [
    "not json", "[]", json.dumps({G: {"level": "1", "low_confidence": False}}),
    json.dumps({G: {"level": "4", "low_confidence": False}, C: {"level": "1", "low_confidence": False}}),
    json.dumps({G: {"level": True, "low_confidence": False}, C: {"level": "1", "low_confidence": False}}),
    json.dumps({G: {"level": "1", "low_confidence": "no"}, C: {"level": "1", "low_confidence": False}}),
    json.dumps({G: {"level": "1", "low_confidence": False, "extra": 1}, C: {"level": "1", "low_confidence": False}}),
    json.dumps({G: {"level": "1", "low_confidence": False}, C: {"level": "1", "low_confidence": False}, "x": {}}),
])
def test_parse_reply_is_strict(text):
    with pytest.raises(gp.GraderReplyError):
        gp.parse_reply(text)


def test_prompt_hash_moves_with_the_wording(monkeypatch):
    before = gp.prompt_sha256()
    assert before == gp.prompt_sha256()
    monkeypatch.setattr(gp, "INSTRUCTIONS", gp.INSTRUCTIONS + ("One more sentence.",))
    assert gp.prompt_sha256() != before


# --- the round runner -------------------------------------------------------------------------

def test_the_committed_records_build_prompts():
    records, dev = jr.load_records(), jr.load_dev()
    assert len(records) == 20
    for r in records:
        msgs = jr.record_messages(r, dev)
        assert "# FACTS SHEET" in msgs[1]["content"] and "# CASE FILE" in msgs[1]["content"]


def test_dry_run_makes_no_call(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(jr, "run_round", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert jr.main(["--round", "0"], out_dir=tmp_path) == jr.EXIT_OK
    assert "dry run: no call made" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def _write_round(out_dir, n, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {"round": n, "grader_prompt_sha256": "x" * 64, "stop_criteria_met": False, **kw}
    (out_dir / f"round-{n}.json").write_text(json.dumps(body), encoding="utf-8")


def test_round_order_rules(tmp_path):
    with pytest.raises(jr.Refused, match="targets no pattern"):
        jr.check_round_order(0, "a pattern", tmp_path)
    with pytest.raises(jr.Refused, match="must name the disagreement pattern"):
        jr.check_round_order(1, None, tmp_path)
    with pytest.raises(jr.Refused, match="round 0 has not run"):
        jr.check_round_order(1, "p", tmp_path)
    with pytest.raises(jr.Refused, match="outside 0 to 3"):
        jr.check_round_order(4, "p", tmp_path)
    _write_round(tmp_path, 0)
    with pytest.raises(jr.Refused, match="never re-run"):
        jr.check_round_order(0, None, tmp_path)
    jr.check_round_order(1, "p", tmp_path)  # new wording, a pattern, round 0 present
    _write_round(tmp_path, 1, grader_prompt_sha256=gp.prompt_sha256())
    with pytest.raises(jr.Refused, match="same as an earlier round"):
        jr.check_round_order(2, "p", tmp_path)


def test_no_round_after_the_stop_criteria_are_met(tmp_path):
    _write_round(tmp_path, 0, stop_criteria_met=True)
    with pytest.raises(jr.Refused, match="step 1 has stopped"):
        jr.check_round_order(1, "p", tmp_path)


def _fake_log(records, replies):
    samples = [SimpleNamespace(id=r["_file"], error=None, output=SimpleNamespace(completion=rep))
               for r, rep in zip(records, replies)]
    return SimpleNamespace(samples=samples, location=None)


def test_agreement_meets_the_stop_criteria():
    records = jr.load_records()
    replies = [reply(r["grades"][G], r["grades"][C]) for r in records]
    res = jr.round_result(records, _fake_log(records, replies), 0, None, offline=True)
    assert res["stop_criteria_met"] is True and not res["unparsed"]
    for d in gp.DIMENSIONS:
        assert res["stats"][d]["icc_a1"]["estimate"] == 1.0 and res["stats"][d]["leniency"]["mean"] == 0.0


def test_a_lenient_grader_misses_the_stop_criteria_and_lists_targets():
    records = jr.load_records()
    replies = [reply(min(3, r["grades"][G] + 1), r["grades"][C]) for r in records]
    res = jr.round_result(records, _fake_log(records, replies), 0, None, offline=True)
    assert res["stats"][G]["stop_met"] is False and res["stats"][C]["stop_met"] is True
    assert res["stop_criteria_met"] is False
    dis = res["stats"][G]["disagreements"]
    assert dis["basis"] == "1 point" and dis["count_1"] == len(dis["cases"]) > 0


def test_an_unparsed_reply_keeps_the_stop_criteria_unmet():
    records = jr.load_records()
    replies = [reply(r["grades"][G], r["grades"][C]) for r in records]
    replies[0] = "no json here"
    res = jr.round_result(records, _fake_log(records, replies), 0, None, offline=True)
    assert len(res["unparsed"]) == 1 and res["stop_criteria_met"] is False
    assert res["stats"][G]["n_graded"] == 19


def test_live_path_runs_one_eval_of_twenty_grader_calls_offline(tmp_path):
    from inspect_ai.model import ModelOutput, get_model

    mock = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", reply(2, 2))
                                                      for _ in range(20)])
    out = tmp_path / "evidence"
    code = jr.main(["--round", "0", "--live", "--log-dir", str(tmp_path / "logs")],
                   model_roles={"grader": mock}, out_dir=out)
    assert code == jr.EXIT_OK
    res = json.loads((out / "round-0.json").read_text(encoding="utf-8"))
    assert res["offline"] is True and res["records"] == 20 and not res["unparsed"]
    assert all(row["grader"] == {G: 2, C: 2} for row in res["rows"])
    assert res["grader_prompt_sha256"] == gp.prompt_sha256()


def test_live_needs_a_log_dir_outside_the_repo(tmp_path, capsys):
    assert jr.main(["--round", "0", "--live", "--log-dir", str(REPO / "logs")], out_dir=tmp_path) == jr.EXIT_ARGS
    assert jr.main(["--round", "0", "--live"], out_dir=tmp_path) == jr.EXIT_ARGS
