"""scripts/rescore_log.py: keyless re-scoring of a saved log reproduces an Inspect run's scores
on the same code, keeps the header, and writes a log the readers accept."""

import importlib.util
import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals import reliability, roster
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.task import variance_commentary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rescore_log.py"
ROLES = {"grader": "mockllm/model", "calibrator": "mockllm/model"}


def _load():
    spec = importlib.util.spec_from_file_location("rescore_log", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _outputs():
    samples = sorted(build_dataset(split="dev", subset="pilot"), key=lambda s: str(s.id))
    inputs = {str(s.id): str(s.input) for s in samples}
    answers = {}
    for s in samples:
        t = truth_of(s.metadata)
        key = next(w for w in t.scope_wallets
                   if w in inputs[str(s.id)] and not any(w in v for o, v in inputs.items() if o != str(s.id)))
        items = t.eligible_items()
        answers[key] = json.dumps({
            "asset": t.asset, "period": t.period, "variance_amount_usd": abs(t.variance_usd),
            "variance_direction": t.direction, "materiality_band": t.materiality_band,
            "entities_cited": [t.custodian, *(i.reference for i in items)],
            "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
            "narrative": "On-chain reserves differ from ledger liabilities for the period.",
        })

    def fn(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in answers if w in blob), None)
        out = ModelOutput.from_content(model="mockllm/model", content=answers.get(key, "unmatched"))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return fn


@pytest.fixture(scope="module")
def pilot_log(tmp_path_factory):
    log = inspect_eval(
        variance_commentary(split="dev", subset="pilot", epochs=4),
        model=get_model("mockllm/model", custom_outputs=_outputs()),
        model_roles=ROLES, log_dir=str(tmp_path_factory.mktemp("rescore")), display="none",
    )[0]
    assert log.status == "success", log.error
    return log.location


def test_check_reproduces_the_inspect_scores(pilot_log, capsys):
    mod = _load()
    assert mod.main([pilot_log, "--check"]) == 0
    assert "differences from the log's scores: 0" in capsys.readouterr().out


def test_output_keeps_the_header_and_the_readers_accept_it(pilot_log, tmp_path):
    mod = _load()
    out = tmp_path / "rescored.eval"
    assert mod.main([pilot_log, "--output", str(out)]) == 0
    src, new = read_eval_log(pilot_log), read_eval_log(str(out))
    assert new.eval.eval_id == src.eval.eval_id and new.eval.model == src.eval.model
    assert new.eval.task_args == src.eval.task_args and new.status == src.status
    assert roster.check_run_config(new, expected_samples=20)["scored_sample_epochs"] == 80
    a, b = reliability.report_from_log(new), reliability.report_from_log(src)
    assert a.pass_k == b.pass_k and a.fm_counts == b.fm_counts
    assert read_eval_log(pilot_log).samples[0].scores.keys() == new.samples[0].scores.keys()


def test_changed_scoring_code_changes_the_written_scores(pilot_log, tmp_path, monkeypatch, capsys):
    """Fix-round 2 verifier finding (2026-11-06): resolve_sample_attachments returns a copy, and the
    first version scored the copy, so --output returned the input unchanged and --check always
    reported 0. A scorer that disagrees with the log must show up as differences and in the file."""
    from inspect_ai.scorer import Score
    from icfr_agent_evals import scorers as scorer_module

    def disagreeing_anchor():
        async def score(state, target):
            return Score(value=0, answer="", explanation="forced", metadata={"failure_modes": ["FM-999"]})
        return score

    monkeypatch.setattr(scorer_module, "correctness_anchor", disagreeing_anchor)
    mod = _load()
    assert mod.main([pilot_log, "--check"]) == 1
    out = capsys.readouterr().out
    # value, failure_modes, review_reasons and requires_human_review differ on 80 sample-epochs
    assert "differences from the log's scores: 320" in out
    rescored = tmp_path / "rescored.eval"
    assert mod.main([pilot_log, "--output", str(rescored)]) == 0
    new = read_eval_log(str(rescored))
    assert all(s.scores["correctness_anchor"].value == 0 for s in new.samples)
    assert all(s.scores["correctness_anchor"].metadata["failure_modes"] == ["FM-999"] for s in new.samples)
    assert all(s.scores["correctness_anchor"].value == 3 for s in read_eval_log(pilot_log).samples)


def test_check_without_output_is_allowed_and_output_is_required_otherwise(pilot_log):
    mod = _load()
    with pytest.raises(SystemExit):
        mod.main([pilot_log])
