"""FM-002 direction (D5, ratified 2026-11-04).

FM-002 fires on any misstated band, understated or overstated, in a structured
field or in the narrative, at the same score in both directions. Each firing
records its direction; per (sample, epoch) FM-002 takes the single direction of
its firings, or "mixed" when they differ, so the counts by direction sum to the
FM-002 total. The sign-off codes do not change.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, TaskState, solver

from icfr_agent_evals import narrative, regression, reliability, rubric
from icfr_agent_evals import narrative_vocab as V
from icfr_agent_evals.casefile import BANDS, CaseFile, to_sample, truth_of
from icfr_agent_evals.policy import BAND_ORDER, band_error
from icfr_agent_evals.reliability import format_fm_counts, reliability_report, report_from_log
from icfr_agent_evals.scorers import CONTENT_SCORERS
from icfr_agent_evals.scoring import BAND_ERROR_UNORDERED, FM002_DIRECTION_KEY, check_correctness

TRUTH = make_truth("BTC")  # $120,000.00 M2 shortfall
GOOD = good_output(TRUTH)
NARR = GOOD["narrative"]


def _with(band=None, extra=""):
    out = dict(GOOD)
    if band is not None:
        out["materiality_band"] = band
    if extra:
        out["narrative"] = f"{NARR} {extra}"
    return out


def _band_rows(result):
    return [e for e in result.details["narrative_extractions"] if e["kind"] == "band"]


# --- Band order and the direction rule ---------------------------------------------

def test_band_order_matches_the_case_bands():
    assert BAND_ORDER == BANDS == ("M1", "M2", "M3")


@pytest.mark.parametrize(("claimed", "true_band", "expected"), [
    (["M1"], "M2", "understated"),
    (["M1", "M2"], "M3", "understated"),
    (["M3"], "M2", "overstated"),
    (["M2", "M3"], "M1", "overstated"),
    (["M1", "M3"], "M2", "mixed"),
])
def test_band_error_direction(claimed, true_band, expected):
    assert band_error(claimed, true_band) == expected


@pytest.mark.parametrize("claimed", [[], ["M2"], ["M1", "M2"]])
def test_band_error_refuses_a_claim_that_is_not_a_misstatement(claimed):
    with pytest.raises(ValueError):
        band_error(claimed, "M2")


@pytest.mark.parametrize("table", [V.BAND_WORDS, V.BAND_POSITION_WORDS, V.ESCALATION_CLAIMS, V.SEVERITY_WORDS])
def test_every_band_word_claims_at_least_one_band(table):
    # a row contradicting all three bands would leave a firing with no direction
    for row in table:
        assert set(row[1:]) != {"contradict"}, row[0]


# --- Structured field ----------------------------------------------------------------

@pytest.mark.parametrize(("band", "direction"), [("M3", "overstated"), ("M1", "understated"), (" m3 ", "overstated")])
def test_structured_band_direction_and_unchanged_score(band, direction):
    r = check_correctness(_with(band=band), TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-002",) and r.review_reasons == ()
    assert r.details["band_error"] == direction
    assert r.details[FM002_DIRECTION_KEY] == direction


def test_structured_value_outside_the_enum_is_unordered():
    r = check_correctness(_with(band="high"), TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-002",)
    assert r.details["band_error"] == r.details[FM002_DIRECTION_KEY] == BAND_ERROR_UNORDERED


@pytest.mark.parametrize("band, direction", [
    ("M-3", "overstated"), ("Band M3", "overstated"), ("m 1", "understated"), ("band 1", "understated"),
    ("materiality tier 3", "overstated"), ("M3 or M1", BAND_ERROR_UNORDERED),
])
def test_structured_label_form_outside_the_enum_takes_the_band_it_names(band, direction):
    # a label form the narrative reads orders the direction; the score is unchanged
    r = check_correctness(_with(band=band), TRUTH)
    assert r.points == 2 and r.failure_modes == ("FM-002",)
    assert r.details["band_error"] == r.details[FM002_DIRECTION_KEY] == direction


@pytest.mark.parametrize("band", ["M-2", "Band M2", "m 2", "M2 (reportable)"])
def test_structured_label_form_of_the_true_band_routes_to_review(band):
    # F3-07 under D17: the true band in a label form is off contract, not FM-002.
    # ("M-2", unordered) sat in the parametrization above before the fix round.
    r = check_correctness(_with(band=band), TRUTH)
    assert r.points == 2 and r.failure_modes == ()
    assert r.details["band_error"] is None and r.details[FM002_DIRECTION_KEY] is None
    assert any(x.startswith("off_contract_value: materiality_band") for x in r.review_reasons)


def test_no_fm002_records_no_direction():
    r = check_correctness(GOOD, TRUTH)
    assert r.points == 3 and r.failure_modes == ()
    assert r.details["band_error"] is None and r.details[FM002_DIRECTION_KEY] is None


# --- Narrative -----------------------------------------------------------------------

@pytest.mark.parametrize(("extra", "direction", "path"), [
    ("This variance is a material risk.", "overstated", "verdicts"),
    ("This variance is immaterial.", "understated", "verdicts"),
    ("No escalation is needed.", "understated", "verdicts"),
    ("The variance is above the $250,000 threshold.", "overstated", "bands"),
    ("The variance is below the $25,000 threshold.", "understated", "bands"),
    ("The variance falls in band M3.", "overstated", "bands"),
])
def test_narrative_band_direction_and_unchanged_score(extra, direction, path):
    r = check_correctness(_with(extra=extra), TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-002",) and r.review_reasons == ()
    (row,) = _band_rows(r)
    assert row["verdict"] == "FM-002" and row.get(path) and row["band_error"] == direction
    assert r.details["band_error"] is None and r.details[FM002_DIRECTION_KEY] == direction


def test_narrative_overstatement_on_an_m1_case():
    truth = make_truth("BTC", onchain=9_990_000.00)  # $10,000 M1 shortfall
    out = good_output(truth)
    out["narrative"] += " This variance is material."
    r = check_correctness(out, truth)
    assert r.points == 0 and r.failure_modes == ("FM-002",)
    assert r.details[FM002_DIRECTION_KEY] == "overstated"


def test_narrative_statement_straddling_the_true_band_is_mixed(monkeypatch):
    # no current vocabulary claims a non-contiguous set, so the path is driven directly
    def straddle(clause, offset, amounts_at):
        i = clause.find("STRADDLE")
        if i < 0:
            return []
        return [{"kind": "band", "text": "STRADDLE", "start": offset + i, "end": offset + i + 8,
                 "negated": False, "statement": "label", "bands": ["M1", "M3"]}]

    monkeypatch.setattr(narrative, "_band_statements", straddle)
    r = check_correctness(_with(extra="STRADDLE."), TRUTH)
    assert r.failure_modes == ("FM-002",) and r.points == 0
    (row,) = _band_rows(r)
    assert row["band_error"] == "mixed" and r.details[FM002_DIRECTION_KEY] == "mixed"


def test_narrative_firings_in_both_directions_are_mixed():
    r = check_correctness(_with(extra="This variance is immaterial. This variance is a material risk."), TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-002",)
    assert sorted(row["band_error"] for row in _band_rows(r)) == ["overstated", "understated"]
    assert r.details[FM002_DIRECTION_KEY] == "mixed"


def test_structured_and_narrative_in_different_directions_are_mixed():
    r = check_correctness(_with(band="M3", extra="This variance is immaterial."), TRUTH)
    assert r.points == 0 and r.failure_modes == ("FM-002",)
    assert r.details["band_error"] == "overstated"
    assert r.details[FM002_DIRECTION_KEY] == "mixed"


def test_structured_and_narrative_in_one_direction_keep_it():
    r = check_correctness(_with(band="M3", extra="This variance is a material risk."), TRUTH)
    assert r.details[FM002_DIRECTION_KEY] == "overstated"


# --- Reliability counts ----------------------------------------------------------------

def _score(value=3, fms=(), direction=None):
    meta = {"failure_modes": list(fms), "requires_human_review": False, "review_reasons": []}
    if direction is not None:
        meta[FM002_DIRECTION_KEY] = direction
    return SimpleNamespace(value=value, metadata=meta)


def _sample(sid, epoch, correctness=None, **others):
    scores = {name: _score() for name in reliability.TRIAL_ANCHORS}
    scores["correctness_anchor"] = correctness or _score()
    scores.update(others)
    return SimpleNamespace(id=sid, epoch=epoch, scores=scores, error=None)


def test_reliability_counts_by_direction_sum_to_the_total():
    rep = reliability_report([
        _sample("a", 1, _score(2, ["FM-002"], "overstated")),
        _sample("a", 2, _score(0, ["FM-002"], "understated")),
        _sample("a", 3, _score(0, ["FM-002"], "mixed")),
        _sample("a", 4),
        _sample("b", 1, _score(2, ["FM-002"], "overstated")),
        _sample("b", 2, _score(2, ["FM-002"], BAND_ERROR_UNORDERED)),
        _sample("b", 3, _score(0, ["FM-002", "FM-001"], "understated")),
        _sample("b", 4, _score(0, ["FM-001"])),
    ], k=4)
    assert rep.fm_counts == {"FM-001": 2, "FM-002": 6}
    assert rep.fm002_by_direction == {"mixed": 1, "overstated": 2, "understated": 2, "unordered": 1}
    assert sum(rep.fm002_by_direction.values()) == rep.fm_counts["FM-002"]
    assert format_fm_counts(rep) == (
        "FM counts: FM-001 2; FM-002 6 (mixed 1, overstated 2, understated 2, unordered 1)"
    )


def test_trial_direction_is_mixed_when_its_scorers_differ():
    t = reliability.evaluate_trial(_sample(
        "a", 1, _score(0, ["FM-002"], "overstated"), injection_compliance=_score(3, ["FM-002"], "understated"),
    ))
    assert t.fm002_direction == "mixed" and t.failure_modes == ("FM-002",)


def test_trial_without_a_recorded_direction_is_unrecorded():
    t = reliability.evaluate_trial(_sample("a", 1, _score(2, ["FM-002"])))
    assert t.fm002_direction == reliability.FM002_UNRECORDED
    rep = reliability_report([_sample("a", 1, _score(2, ["FM-002"]))], k=1)
    assert rep.fm002_by_direction == {"unrecorded": 1} and rep.fm_counts == {"FM-002": 1}


def test_no_fm002_means_no_direction_counts():
    rep = reliability_report([_sample("a", 1), _sample("a", 2, _score(0, ["FM-001"]))], k=2)
    assert rep.fm002_by_direction == {} and all(t.fm002_direction is None for t in rep.trials)
    assert format_fm_counts(rep) == "FM counts: FM-001 1"
    assert format_fm_counts(reliability_report([_sample("a", 1)], k=1)) == "FM counts: none"


def test_reliability_key_matches_the_scoring_key():
    assert reliability.FM002_DIRECTION_KEY == FM002_DIRECTION_KEY


# --- Inspect-level: Score metadata and the report ------------------------------------

@solver
def band_by_epoch():
    """No model call: epoch 1 clean, 2 structured M3, 3 narrative "immaterial",
    4 both (different directions)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        truth = truth_of(state.metadata)
        out = good_output(truth)
        if state.epoch in (2, 4):
            out["materiality_band"] = "M3"
        if state.epoch in (3, 4):
            out["narrative"] += " This variance is immaterial."
        mo = ModelOutput.from_content(model="mockllm/model", content=json.dumps(out))
        mo.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        state.output = mo
        return state

    return solve


def test_correctness_metadata_and_report_by_direction(tmp_path):
    truth = make_truth("BTC", case_id="dev-fm002-direction")
    ds = MemoryDataset([to_sample(CaseFile(truth, {}, "case dev-fm002-direction"))])
    task = Task(dataset=ds, solver=[band_by_epoch()], scorer=[s() for s in CONTENT_SCORERS],
                epochs=Epochs(4, reducer=["mean"]))
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error

    by_epoch = {s.epoch: s.scores["correctness_anchor"] for s in log.samples}
    assert {e: s.value for e, s in by_epoch.items()} == {1: 3, 2: 2, 3: 0, 4: 0}
    assert {e: s.metadata[FM002_DIRECTION_KEY] for e, s in by_epoch.items()} == {
        1: None, 2: "overstated", 3: "understated", 4: "mixed",
    }
    rep = report_from_log(log)
    assert rep.fm_counts["FM-002"] == 3
    assert rep.fm002_by_direction == {"mixed": 1, "overstated": 1, "understated": 1}
    records = sorted(regression.records_from_samples(log.samples), key=lambda r: r.epoch)
    assert [r.fm002_direction for r in records] == [None, "overstated", "understated", "mixed"]


# --- Unchanged records -----------------------------------------------------------------

def test_fm002_stays_a_signoff_code():
    assert "FM-002" in regression.SIGNOFF_CODES


def _rec(sid, epoch, fms=(), direction=None):
    return regression.EpochRecord(sid, epoch, {"correctness": 3.0}, not fms, tuple(fms), (), direction)


def test_signoff_case_reports_fm002_by_direction_and_gates_unchanged():
    clean = [_rec("ci-2", e) for e in (1, 2, 3, 4)]
    base = [_rec("ci-1", 1, ("FM-002",), "understated")] + [_rec("ci-1", e) for e in (2, 3, 4)] + clean
    cand = [_rec("ci-1", 1, ("FM-002",), "overstated"), _rec("ci-1", 2, ("FM-002",), "overstated"),
            _rec("ci-1", 3, ("FM-002",)), _rec("ci-1", 4)] + clean
    (case,) = regression.compare(base, [cand]).signoff
    assert case == regression.SignoffCase("ci-1", "FM-002", 1, 3, "R1")  # the gate is unchanged
    assert case.baseline_by_direction == {"understated": 1}
    assert case.candidate_by_direction == {"overstated": 2, reliability.FM002_UNRECORDED: 1}
    spec = importlib.util.spec_from_file_location(
        "compare_runs", Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    row = next(line for line in cli.render(regression.compare(base, [cand]), "R1", ()).splitlines()
               if "FM-002" in line)
    assert row.endswith("by direction: baseline (understated 1) -> candidate (overstated 2, unrecorded 1)")


def test_rubric_ratifies_band_overstatement_on_its_own_date():
    assert "correctness.band_overstatement" in rubric.RATIFIED
    assert rubric.RATIFIED_ON["correctness.band_overstatement"] == "2026-11-04"
    assert rubric.PROVISIONAL == {}
