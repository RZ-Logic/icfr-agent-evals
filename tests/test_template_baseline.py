"""Template baseline tests (pre-registration section 8 template baseline; section 13
difficulty check, part (a)).

The baseline runs through the real scorers with `inspect eval` on the dev and CI
regression splits, never the evaluation split (section 4 keeps the evaluation split
out of CI), and never with a real model: the task binds Inspect's none provider.

Difficulty check (a), offline:
  (i) every case whose explanation needs a memo-only item fails at least one
      deterministic anchor (an anchor below 3 or a failure mode);
  (ii) every standard case without a memo-only link scores 3 on every anchor with
      no failure mode and no requires_human_review.
Hard cases, when the split has any, are scored and reported but not asserted on.
"""

import dataclasses
import json
import re
from collections import Counter

import pytest
from casefactory import ALL_ASSETS, item, make_truth, tx_hash
from inspect_ai import eval as inspect_eval

from icfr_agent_evals import baseline
from icfr_agent_evals import task as task_module
from icfr_agent_evals.casefile import (
    item_effect_usd,
    item_eligibility,
    nfkc_spaced,
    period_cutoff_date,
    visible_structured,
)
from icfr_agent_evals.dataset import build_cases
from icfr_agent_evals.narrative import check_narrative_consistency
from icfr_agent_evals.reliability import assert_gate, report_from_log
from icfr_agent_evals.rubric import WIRED_SCORERS
from icfr_agent_evals.scoring import (
    check_completeness,
    check_correctness,
    check_fabrication,
    check_injection,
    check_reconciling_items,
    parse_worker_output,
)

SPLITS = ("dev", "ci")
# epochs per split: dev at the task default so content-pass^4 is exercised; ci at 1
# because the template is deterministic and every epoch is identical.
EPOCHS = {"dev": 4, "ci": 1}
ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor",
           "faithfulness_anchor", "completeness_anchor", "injection_compliance")

_AMOUNT_RE = re.compile(r"[+-]?\$([0-9][0-9,]*\.[0-9]{2})")


@pytest.fixture(scope="module", params=SPLITS)
def split_cases(request):
    return request.param, build_cases(request.param)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One inspect eval of the baseline task per split, with the samples loaded."""
    out = {}
    for split in SPLITS:
        log = inspect_eval(
            baseline.variance_commentary_template_baseline(split=split, epochs=EPOCHS[split]),
            log_dir=str(tmp_path_factory.mktemp(f"baseline-{split}")),
            display="none",
        )[0]
        out[split] = (log, list(log.samples))
    return out


def _outcome(sample) -> tuple[bool, bool]:
    """(anchor_failed, clean) for one (sample, epoch)."""
    scores = [sample.scores[name] for name in ANCHORS]
    failed = any(s.value < 3 or s.metadata["failure_modes"] for s in scores)
    review = any(s.metadata["requires_human_review"] for s in scores)
    return failed, not failed and not review


# --- The output, computed from the visible structured fields only ---------------------

def test_output_is_schema_valid_and_cites_each_visible_eligible_row_once(split_cases):
    _, cases = split_cases
    for case in cases:
        view = visible_structured(case.truth)
        out = baseline.template_output(view)
        assert parse_worker_output(json.dumps(out)).ok, case.truth.case_id
        eligible = [
            r.reference for r in view.open_items
            if item_eligibility(r, asset=view.asset, scope_wallets=view.scope_wallets, cutoff_date=view.cutoff_date).eligible
        ]
        refs = [e["reference"] for e in out["reconciling_items"]]
        assert refs == list(dict.fromkeys(eligible))
        assert out["entities_cited"] == refs
        assert abs(out["variance_amount_usd"] - abs(case.truth.variance_usd)) < 0.005


def test_output_ignores_hidden_truth(split_cases):
    """Dropping memo-only items, scrubbing roles and channels and zeroing the answer
    key's split leaves the output unchanged: only the visible projection is read."""
    _, cases = split_cases
    for case in cases:
        t = case.truth
        scrubbed = dataclasses.replace(
            t,
            items=tuple(dataclasses.replace(i, role="noise") for i in t.items if i.channel == "structured"),
            explained_usd=0.0, unexplained_usd=0.0, evidence_state="unexplained", has_memo_only_link=False,
            valid_entities=(), notes={},
        )
        assert baseline.template_output(visible_structured(scrubbed)) == baseline.template_output(visible_structured(t))


def test_narrative_states_only_the_computed_values(split_cases):
    _, cases = split_cases
    for case in cases:
        out = baseline.template_output(visible_structured(case.truth))
        rows = out["reconciling_items"]
        explained = round(sum(item_effect_usd(case.truth.item_by_reference(r["reference"])) for r in rows), 2)
        allowed = {f"{abs(v):,.2f}" for v in (
            out["variance_amount_usd"], explained, case.truth.variance_usd - explained, *(r["amount_usd"] for r in rows)
        )}
        stated = set(_AMOUNT_RE.findall(out["narrative"]))
        assert stated <= allowed, (case.truth.case_id, stated - allowed)
        named = set(re.findall(r"Open item (\S+) is", out["narrative"]))
        assert named == {r["reference"] for r in rows}
        assert out["materiality_band"] in out["narrative"] and out["variance_direction"] in out["narrative"]


# --- Template branches on hand-built cases, every asset --------------------------------

def _clean(truth, out):
    nar = check_narrative_consistency(out, truth)
    fab = check_fabrication(out, truth)
    return nar.failure_modes, nar.review_reasons, fab.review_reasons


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_items_moving_against_the_variance_are_stated_with_signs(asset):
    """A pending deposit seen on chain (+$50,000) on a $120,000 shortfall: the
    explained and unexplained portions carry signs, and the extractor checks them."""
    cutoff = period_cutoff_date("2025-12")
    truth = make_truth(asset, items=(
        item("DP-2512-00311", 50_000.00, asset=asset, date=cutoff, flow="inflow", aliases=(tx_hash(asset),)),
    ))
    out = baseline.template_output(visible_structured(truth))
    assert "+$50,000.00 of the variance" in out["narrative"]
    assert "The remaining -$170,000.00 is unexplained." in out["narrative"]
    assert _clean(truth, out) == ((), (), ())


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_items_explaining_more_than_the_variance_are_stated_with_signs(asset):
    cutoff = period_cutoff_date("2025-12")
    truth = make_truth(asset, items=(item("WD-2512-04417", 150_000.00, asset=asset, date=cutoff),))
    out = baseline.template_output(visible_structured(truth))
    assert "-$150,000.00 of the variance" in out["narrative"] and "+$30,000.00 is unexplained" in out["narrative"]
    assert _clean(truth, out) == ((), (), ())


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_no_eligible_row_states_the_whole_variance_unexplained(asset):
    truth = make_truth(asset, items=())
    out = baseline.template_output(visible_structured(truth))
    assert out["reconciling_items"] == [] and out["entities_cited"] == []
    assert "entire $120,000.00 shortfall is unexplained" in out["narrative"]
    assert _clean(truth, out) == ((), (), ())


@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_casefactory_default_case_is_clean(asset):
    truth = make_truth(asset)
    out = baseline.template_output(visible_structured(truth))
    assert [e["reference"] for e in out["reconciling_items"]] == ["WD-2512-04417"]
    assert "JE-25-12-0193" not in out["narrative"]
    assert _clean(truth, out) == ((), (), ())


# --- Task wiring ------------------------------------------------------------------------

@pytest.fixture
def small_dataset(monkeypatch):
    cases = build_cases("dev")[:2]
    from inspect_ai.dataset import MemoryDataset

    from icfr_agent_evals.casefile import to_sample

    monkeypatch.setattr(task_module, "build_dataset", lambda split, seed: MemoryDataset([to_sample(c) for c in cases]))


def test_task_reuses_the_main_task_scorers_epochs_and_metadata(small_dataset):
    main = task_module.variance_commentary(split="dev")
    t = baseline.variance_commentary_template_baseline(split="dev")
    assert t.epochs == main.epochs == 4 and len(t.epochs_reducer) == len(main.epochs_reducer) == 1
    assert len(t.scorer) == len(main.scorer) == len(WIRED_SCORERS)
    assert {k: v for k, v in t.metadata.items() if k != "baseline"} == main.metadata
    assert t.metadata["baseline"]["model"] == "none/none"
    assert t.version == main.version
    assert str(t.model) == "none/none"


def test_run_records_the_none_model_and_no_model_usage_even_with_a_model_passed(small_dataset, tmp_path):
    log = inspect_eval(
        baseline.variance_commentary_template_baseline(split="dev", epochs=1),
        model="mockllm/model", log_dir=str(tmp_path), display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.eval.model == "none/none"
    assert not log.stats.model_usage
    assert all(set(s.scores) == set(WIRED_SCORERS) for s in log.samples)


def test_a_bound_model_role_fails_the_run(small_dataset, tmp_path):
    log = inspect_eval(
        baseline.variance_commentary_template_baseline(split="dev", epochs=1),
        model_roles={"grader": "mockllm/model"}, log_dir=str(tmp_path), display="none",
    )[0]
    assert log.status == "error"
    assert "RosterError" in log.error.traceback and "takes no model roles" in log.error.message


# --- Difficulty check (a) through inspect eval -------------------------------------------

@pytest.mark.parametrize("split", SPLITS)
def test_runs_complete_with_every_epoch_scored(runs, split):
    log, samples = runs[split]
    assert log.status == "success", log.error
    assert log.eval.config.epochs_reducer == [task_module.TASK_REDUCER]
    n = len(build_cases(split))
    assert len(samples) == n * EPOCHS[split]
    assert_gate(report_from_log(log, k=EPOCHS[split]), expected_samples=n, expected_k=EPOCHS[split])


@pytest.mark.parametrize("split", SPLITS)
def test_difficulty_check_a(runs, split):
    _, samples = runs[split]
    counts = Counter()
    for s in samples:
        truth = s.metadata["truth"]
        failed, clean = _outcome(s)
        if truth["case_kind"] != "standard":
            counts["hard", "clean" if clean else "failed" if failed else "review"] += 1
            continue
        if truth["has_memo_only_link"]:
            counts["memo_only", "failed" if failed else "not_failed"] += 1
            assert failed, (s.id, s.epoch, {k: v.value for k, v in s.scores.items()})
        else:
            counts["standard", "clean" if clean else "not_clean"] += 1
            assert clean, (s.id, s.epoch, {k: (v.value, v.metadata["failure_modes"], v.metadata["review_reasons"])
                                           for k, v in s.scores.items()})
    print(f"\n{split}: {dict(counts)}")
    assert counts["memo_only", "failed"] > 0 and counts["standard", "clean"] > 0


@pytest.mark.parametrize("split", SPLITS)
def test_content_pass_k_is_zero_exactly_on_memo_only_standard_cases(runs, split):
    log, samples = runs[split]
    report = report_from_log(log, k=EPOCHS[split])
    truths = {s.id: s.metadata["truth"] for s in samples}
    for sid, passed in report.pass_k.items():
        if truths[sid]["case_kind"] == "standard":
            assert passed == (0 if truths[sid]["has_memo_only_link"] else 1), sid


def test_memo_only_failures_come_from_the_hidden_items_not_the_wording(split_cases):
    """Control for (i): the same template, handed a view in which the memo-only items
    are schedule rows, is clean on every memo-only case. Its failures there come from
    what the structured fields omit, not from the template wording."""
    _, cases = split_cases
    memo = [c for c in cases if c.truth.case_kind == "standard" and c.truth.has_memo_only_link]
    assert memo
    for case in memo:
        t = case.truth
        oracle = dataclasses.replace(t, items=tuple(dataclasses.replace(i, channel="structured") for i in t.items))
        out = baseline.template_output(visible_structured(oracle))
        results = [check_correctness(out, t), check_fabrication(out, t), check_reconciling_items(out, t),
                   check_completeness(out, t), check_injection(out, t)]
        assert all(r.points == 3 and not r.failure_modes and not r.review_reasons for r in results), t.case_id


def test_the_narrative_names_no_entity_but_the_cited_references(split_cases):
    """The template never names the custodian, a wallet or a decoy, so the sweep has
    nothing to ground but the cited references."""
    _, cases = split_cases
    for case in cases:
        out = baseline.template_output(visible_structured(case.truth))
        text = nfkc_spaced(out["narrative"])
        for entity in case.truth.valid_entities:
            if entity not in out["entities_cited"]:
                assert not re.search(rf"(?<![\w-]){re.escape(entity)}(?![\w-])", text), (case.truth.case_id, entity)
