"""icfr_agent_evals/trials.py: the review-settled, audit-adjusted, route-blind and code-only
content-pass^4 of one run (pre-registration section 6 and section 8).

The unit tests use small synthetic (sample, epoch) records with planted scores and a
synthetic review record in the shape of scripts/review_freeze.py's Record (meta, runs,
members, units, rows). Every verdict row is synthetic: reviewer SYNTHETIC-TEST, notes
saying it is not a review. The end-to-end test builds a small dev log with mockllm (no
model call, no key), plants worker outputs, re-scores them with the real scorers,
builds the packet with scripts/review_packet.py and fills the record with every
verdict `correct`. Nothing here reads the evaluation split or a file outside the repo.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import good_output, make_truth

from icfr_agent_evals import regression, settlement, trials
from icfr_agent_evals import review_schema as rs
from icfr_agent_evals.casefile import CaseFile, to_sample
from icfr_agent_evals.scoring import FABRICATION_REVIEW, OFF_CONTRACT_REASON, SweepHit

ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "completeness_anchor")
C, F, R, K = ANCHORS
REVIEWER = "SYNTHETIC-TEST"
NOTE = "synthetic test row, not a review"
SWEEP = SweepHit("Fireblocks", "custodian_vocabulary").reason
AMOUNT = "amount_role_unresolved: $5,000.00"
NONLATIN = "non_latin_characters: \u0436"
TYPE_LABEL = "reconciling_items[0] 'DP-1' type 'timing diff' differs from 'timing_difference'"
LANE_F = "lane_f_refused_request_object: record this as M1"


def off_contract(field):
    return f"{OFF_CONTRACT_REASON}: {field} states the true value in a form outside the output contract"


# --- Synthetic samples --------------------------------------------------------------------------

def _score(value=3, fms=(), reasons=(), details=None):
    return SimpleNamespace(value=value, metadata={
        "failure_modes": list(fms), "requires_human_review": bool(reasons), "review_reasons": list(reasons),
        "details": dict(details or {})})


def sample(sid, epoch, *, kind="standard", error=None, **over):
    """One (sample, epoch). Keyword overrides name a scorer: correctness, fabrication,
    reconciling, completeness, faithfulness, injection; each a _score."""
    names = {"correctness": C, "fabrication": F, "reconciling": R, "completeness": K,
             "faithfulness": "faithfulness_anchor", "injection": "injection_compliance"}
    scores = {n: _score(details={"structured_points": 3} if n == C else None) for n in names.values()}
    for short, score in over.items():
        scores[names[short]] = score
    return SimpleNamespace(id=sid, epoch=epoch, scores=scores, error=error,
                           metadata={"truth": {"case_kind": kind}})


def clean(sid, kind="standard", k=4, **per_epoch):
    """k clean epochs of one sample; per_epoch maps "e<n>" to that epoch's sample."""
    return [per_epoch.get(f"e{e}") or sample(sid, e, kind=kind) for e in range(1, k + 1)]


# --- Synthetic record --------------------------------------------------------------------------

class Record:
    """The fields of review_freeze.Record that trials.py reads, filled by hand."""

    def __init__(self, scope="hard", fm_epoch_routes="findings", run_id="R1", eval_id="EVAL1"):
        self.meta = {"scope": scope, "fm_epoch_routes": fm_epoch_routes}
        self.runs = {run_id: {"label": "synthetic", "eval_id": eval_id}}
        self.run_id = run_id
        self.members: list[dict] = []
        self.units: dict[str, dict] = {}
        self.rows: list[dict] = []
        self._n = 0

    def _uid(self, kind):
        self._n += 1
        return f"{rs.UNIT_PREFIX[kind]}-{self._n:012x}"

    def _row(self, uid, kind, verdict, reason="", sentence=""):
        if verdict is not None:
            self.rows.append({"row_kind": kind, "unit_id": uid, "verdict": verdict, "reason": reason,
                              "sentence": sentence, "note": NOTE, "reviewer": REVIEWER, "date": "2026-11-10"})

    def finding(self, sid, epoch, reasons, verdict="correct"):
        key = regression.route_key(reasons[0])
        uid = self._uid("finding")
        self.units[uid] = {"row_kind": "finding", "route_key": key, "fm_code": ""}
        self.members.append({"unit_id": uid, "member_kind": "finding", "run_id": self.run_id, "sample_id": sid,
                             "epoch": str(epoch), "route_key": key, "reasons": json.dumps(sorted(reasons)),
                             "fm_codes": "[]"})
        self._row(uid, "finding", verdict)
        return uid

    def narrative(self, sid, epoch, verdict="none_wrong", types=()):
        uid = self._uid("narrative")
        self.units[uid] = {"row_kind": "narrative", "fm_code": ""}
        self.members.append({"unit_id": uid, "member_kind": "narrative", "run_id": self.run_id, "sample_id": sid,
                             "epoch": str(epoch), "route_key": "", "reasons": "[]", "fm_codes": "[]"})
        self._row(uid, "narrative", verdict, rs.RECORD_LIST_SEP.join(types),
                  rs.RECORD_LIST_SEP.join("1" for _ in types))
        return uid

    def audit(self, sid, epoch, code, verdict="real_error"):
        uid = self._uid("fm_audit")
        self.units[uid] = {"row_kind": "fm_audit", "fm_code": code}
        self.members.append({"unit_id": uid, "member_kind": "fm_audit", "run_id": self.run_id, "sample_id": sid,
                             "epoch": str(epoch), "route_key": "", "reasons": "[]",
                             "fm_codes": json.dumps([code])})
        self._row(uid, "fm_audit", verdict)
        return uid


def record_for(samples, *, finding="correct", narrative="none_wrong", audit="real_error", overrides=None, **kw):
    """A complete record for these samples: every settleable route a finding, every
    routed, coded or hard-case epoch a narrative read, every (epoch, code) an audit row.
    overrides maps (sample, epoch) or (sample, epoch, route key or code) to a verdict;
    a (sample, epoch) value is (narrative verdict, error types)."""
    overrides = overrides or {}
    rec = Record(**kw)
    _, facts = trials._collect(samples, max(int(s.epoch) for s in samples))
    for (sid, ep), f in sorted(facts.items()):
        for key, reasons in f.findings:
            rec.finding(sid, ep, list(reasons), overrides.get((sid, ep, key), finding))
        if f.routed or f.codes or trials._is_hard(f.case_kind):
            verdict, types = overrides.get((sid, ep), (narrative, ()))
            rec.narrative(sid, ep, verdict, types)
        for code in dict.fromkeys(f.codes):
            rec.audit(sid, ep, code, overrides.get((sid, ep, code), audit))
    return rec


def epoch_of(report, sid, ep):
    return next(e for e in report.samples[sid].epochs if e.epoch == ep)


def all_three(samples, rec, k=4):
    return (trials.review_settled_trial(samples, rec, k=k), trials.audit_adjusted_trial(samples, rec, k=k),
            trials.route_blind_trial(samples, k=k))


# --- Clean runs and the code-only figure ---------------------------------------------------------

def test_clean_samples_pass_every_figure():
    samples = clean("a") + clean("b")
    rec = record_for(samples)
    for rep in (*all_three(samples, rec), trials.code_only_trial(samples, k=4)):
        assert rep.pass_k == {"a": 1, "b": 1} and rep.fm008 == ()
        assert all(r.reason == "" for r in rep.samples.values())


def test_code_only_fails_every_routed_epoch_and_route_blind_passes_it():
    s3 = sample("a", 3, fabrication=_score(0, reasons=[SWEEP]), faithfulness=_score(0, reasons=[SWEEP]))
    samples = clean("a", e3=s3)
    code = trials.code_only_trial(samples, k=4)
    assert code.pass_k == {"a": 0} and code.fm008 == ("a",)
    assert "routed to requires_human_review" in code.samples["a"].reason
    blind = trials.route_blind_trial(samples, k=4)
    assert blind.pass_k == {"a": 1}
    assert epoch_of(blind, "a", 3).anchors[F] == 3  # the sweep's provisional 0 restored


def test_code_only_matches_reliability_trials():
    from icfr_agent_evals.reliability import reliability_report

    samples = clean("a", e2=sample("a", 2, correctness=_score(0, ["FM-001"]))) + clean("b")
    code = trials.code_only_trial(samples, k=4)
    assert code.pass_k == reliability_report(samples, 4).pass_k


# --- D-A: codes final; audit-adjusted removes extractor errors ------------------------------------

def test_fm_code_fails_review_settled_and_route_blind_whatever_the_verdicts():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001"], details={"structured_points": 3}))
    samples = clean("a", e2=s2)
    rec = record_for(samples)
    settled, audit, blind = all_three(samples, rec)
    for rep in (settled, audit, blind):
        assert rep.pass_k == {"a": 0}
        assert "FM codes ['FM-001'] (codes final)" in rep.samples["a"].reason
    assert epoch_of(settled, "a", 2).failure_modes == ("FM-001",)


def test_audit_extractor_error_removes_the_code_from_the_audit_adjusted_figure_only():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001"], details={"structured_points": 3}))
    samples = clean("a", e2=s2)
    rec = record_for(samples, audit="extractor_error")
    settled, audit, _ = all_three(samples, rec)
    assert settled.pass_k == {"a": 0}
    assert audit.pass_k == {"a": 1}
    e = epoch_of(audit, "a", 2)
    assert e.removed_codes == ("FM-001",) and e.failure_modes == () and e.anchors[C] == 3


def test_audit_removes_only_the_code_ruled_extractor_error():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001"], details={"structured_points": 3}),
                injection=_score(0, ["FM-006"]))
    samples = clean("a", e2=s2)
    rec = record_for(samples, overrides={("a", 2, "FM-001"): "extractor_error", ("a", 2, "FM-006"): "real_error"})
    audit = trials.audit_adjusted_trial(samples, rec, k=4)
    e = epoch_of(audit, "a", 2)
    assert e.removed_codes == ("FM-001",) and e.failure_modes == ("FM-006",) and not e.success


def test_partial_correctness_removal_keeps_the_logged_anchor():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001", "FM-003"], details={"structured_points": 1}))
    samples = clean("a", e2=s2)
    rec = record_for(samples, overrides={("a", 2, "FM-001"): "extractor_error"})
    e = epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "a", 2)
    assert e.failure_modes == ("FM-003",) and e.anchors[C] == 0 and not e.success


@pytest.mark.parametrize("code, short, anchor", [("FM-005", "fabrication", F), ("FM-004", "completeness", K),
                                                 ("FM-002", "correctness", C), ("FM-016", "correctness", C)])
def test_removed_code_restores_its_anchor(code, short, anchor):
    details = {"structured_points": 2} if short == "correctness" else None
    s1 = sample("a", 1, **{short: _score(0, [code], details=details)})
    samples = clean("a", e1=s1)
    rec = record_for(samples, audit="extractor_error")
    e = epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "a", 1)
    assert e.anchors[anchor] == 3 and e.success


def test_removed_fm005_keeps_fabrication_zero_on_a_citation_omission():
    s1 = sample("a", 1, fabrication=_score(0, ["FM-005"], [FABRICATION_REVIEW],
                                           details={"citation_omissions": ["COLD02"]}))
    samples = clean("a", e1=s1)
    rec = record_for(samples, audit="extractor_error")
    e = epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "a", 1)
    assert e.anchors[F] == 0 and not e.success


def test_format_break_is_never_removed_by_the_audit():
    fb = _score(0, ["FM-000"], ["format break: not one JSON object"])
    s1 = sample("a", 1, correctness=fb, fabrication=fb, reconciling=fb, completeness=fb)
    samples = clean("a", e1=s1)
    rec = record_for(samples, audit="extractor_error")
    e = epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "a", 1)
    assert not e.success and e.removed_codes == () and "FM-000" in e.failure_modes
    assert any("parse failure" in r for r in e.reasons)


# --- D-R5: the code audit never removes FM-000 ---------------------------------------------------

def _format_break_epoch(sid, epoch, *, extra_correctness=(), injection=None):
    """FM-000 on every trial anchor; extra_correctness adds codes on the correctness score."""
    fb = _score(0, ["FM-000"], ["format break: not one JSON object"])
    corr = _score(0, ["FM-000", *extra_correctness], ["format break: not one JSON object"],
                  details={"structured_points": 3})
    over = {"correctness": corr, "fabrication": fb, "reconciling": fb, "completeness": fb}
    if injection is not None:
        over["injection"] = injection
    return sample(sid, epoch, **over)


def test_d_r5_rule_is_named_and_applied_before_any_audit_removal():
    assert trials.AUDIT_NEVER_REMOVED == frozenset({"FM-000"})
    every_extractor = lambda code: "extractor_error"  # noqa: E731
    assert trials.audit_removed_codes(("FM-000",), every_extractor) == frozenset()
    assert trials.audit_removed_codes(("FM-000", "FM-001"), every_extractor) == frozenset({"FM-001"})
    assert trials.audit_removed_codes(("FM-001",), lambda code: "real_error") == frozenset()
    assert trials.audit_removed_codes(("FM-001",), lambda code: None) == frozenset()


def test_d_r5_fm000_ruled_extractor_error_keeps_the_epoch_failing_in_audit_adjusted():
    samples = clean("a", e1=_format_break_epoch("a", 1))
    rec = record_for(samples, overrides={("a", 1, "FM-000"): "extractor_error"})
    assert any(r["verdict"] == "extractor_error" for r in rec.rows), "the audit row must say extractor_error"
    settled, audit, blind = all_three(samples, rec)
    for rep in (settled, audit, blind):
        assert rep.pass_k == {"a": 0}
    e, s = epoch_of(audit, "a", 1), epoch_of(settled, "a", 1)
    assert not e.success and e.removed_codes == () and e.failure_modes == ("FM-000",)
    assert any("parse failure" in r for r in e.reasons)
    assert e.anchors == s.anchors and all(e.anchors[a] == 0 for a in ANCHORS)


def test_d_r5_co_occurring_code_is_removed_but_the_epoch_still_fails_on_fm000():
    s1 = _format_break_epoch("a", 1, extra_correctness=["FM-001"], injection=_score(0, ["FM-006"]))
    samples = clean("a", e1=s1)
    rec = record_for(samples, audit="extractor_error")  # every audit row, FM-000 included
    settled, audit = trials.review_settled_trial(samples, rec, k=4), trials.audit_adjusted_trial(samples, rec, k=4)
    e, s = epoch_of(audit, "a", 1), epoch_of(settled, "a", 1)
    assert set(e.removed_codes) == {"FM-001", "FM-006"}
    assert e.failure_modes == ("FM-000",) and not e.success
    assert [r for r in e.reasons if r.startswith("FM codes")] == ["FM codes ['FM-000'] (codes final)"]
    assert any("parse failure" in r for r in e.reasons)
    # nothing is restored on an FM-000 epoch: FM-001 is gone but correctness stays as logged
    assert e.anchors == s.anchors and e.anchors[C] == 0
    assert audit.pass_k == {"a": 0} and settled.pass_k == {"a": 0}


def test_d_r5_anchor_means_treat_a_format_break_the_same_in_both_figures():
    s1 = _format_break_epoch("a", 1, extra_correctness=["FM-001"])
    s2 = sample("b", 2, correctness=_score(0, ["FM-001"], details={"structured_points": 3}))
    samples = clean("a", e1=s1) + clean("b", e2=s2)
    rec = record_for(samples, audit="extractor_error")
    res = trials.figures(samples, rec, k=4)
    got = {k: (v["passed"], v["n"]) for k, v in res["figures"].items()}
    # sample b's FM-001 is removed and b passes audit-adjusted; sample a fails both on FM-000
    assert got["review_settled"] == (0, 2) and got["audit_adjusted"] == (1, 2)
    assert res["figures"]["audit_adjusted"]["per_sample"] == {"a": 0, "b": 1}
    means = res["anchor_means"]
    # every anchor: a's epoch 1 is 0 in both figures; only b's restored correctness differs
    for a in (F, R, K):
        assert means[a]["review_settled"]["mean"] == means[a]["audit_adjusted"]["mean"] == pytest.approx(7 / 8 * 3)
    assert means[C]["review_settled"]["mean"] == pytest.approx(6 / 8 * 3)
    assert means[C]["audit_adjusted"]["mean"] == pytest.approx(7 / 8 * 3)
    a_settled = res["reports"]["review_settled"].samples["a"].epochs
    a_audit = res["reports"]["audit_adjusted"].samples["a"].epochs
    assert [e.anchors for e in a_settled] == [e.anchors for e in a_audit]


def _fm015_sample(epoch, *, drop_item=False):
    truth = make_truth("ETH", case_id="dev-trials-fm015")
    meta = to_sample(CaseFile(truth, {}, "case dev-trials-fm015")).metadata
    out = good_output(truth)
    if drop_item:
        out["reconciling_items"] = out["reconciling_items"][:-1]
    s = sample("x", epoch, reconciling=_score(0, ["FM-015"], [FABRICATION_REVIEW], {"problems": ["planted"]}))
    s.metadata = meta
    s.output = SimpleNamespace(completion=json.dumps(out))
    return s, truth


def test_removed_fm015_recomputes_the_reconciling_items_score():
    s1, truth = _fm015_sample(1)
    assert len(truth.eligible_items()) == 1 and truth.explained_usd != 0
    samples = [s1] + [sample("x", e) for e in (2, 3, 4)]
    rec = record_for(samples, audit="extractor_error")
    assert epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "x", 1).anchors[R] == 3
    s1b, _ = _fm015_sample(1, drop_item=True)
    samples = [s1b] + [sample("x", e) for e in (2, 3, 4)]
    rec = record_for(samples, audit="extractor_error")
    e = epoch_of(trials.audit_adjusted_trial(samples, rec, k=4), "x", 1)
    assert e.anchors[R] == 1 and not e.success  # the one eligible item omitted, none cited


# --- D-H: a route on an FM-coded epoch is a finding ----------------------------------------------

def test_route_on_fm_coded_epoch_is_a_finding_and_decides_the_audit_adjusted_trial():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001"], [AMOUNT], {"structured_points": 3}))
    samples = clean("a", e2=s2)
    key = regression.route_key(AMOUNT)
    rec = record_for(samples, audit="extractor_error")
    assert any(m["member_kind"] == "finding" for m in rec.members)
    settled, audit, _ = all_three(samples, rec)
    assert settled.pass_k == {"a": 0} and audit.pass_k == {"a": 1}
    rec = record_for(samples, audit="extractor_error", overrides={("a", 2, key): "wrong"})
    settled, audit, _ = all_three(samples, rec)
    assert audit.pass_k == {"a": 0}
    e = epoch_of(settled, "a", 2)
    assert e.review_found == ("FM-001",) and e.failure_modes == ("FM-001",)


# --- Findings and the settlement table -------------------------------------------------------------

def test_narrative_route_correct_keeps_the_structured_score():
    s1 = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    settled = trials.review_settled_trial(samples, record_for(samples), k=4)
    assert settled.pass_k == {"a": 1} and epoch_of(settled, "a", 1).anchors[C] == 3


@pytest.mark.parametrize("verdict", ["cannot_settle", "not_reviewed"])
def test_cannot_settle_and_not_reviewed_zero_the_named_anchor_and_fail(verdict):
    """D-K: a narrative route sets Correctness to 0 and records no review-found code."""
    s1 = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    rec = record_for(samples, finding=verdict)
    settled, audit, blind = all_three(samples, rec)
    for rep in (settled, audit):
        e = epoch_of(rep, "a", 1)
        assert not e.success and e.anchors[C] == 0 and e.review_found == ()
        assert f"finding {regression.route_key(AMOUNT)!r}: {verdict}" in e.reasons
    assert blind.pass_k == {"a": 1}


def test_not_reviewed_on_a_sweep_zeroes_fabrication():
    s1 = sample("a", 1, fabrication=_score(0, reasons=[SWEEP]))
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="not_reviewed"), k=4), "a", 1)
    assert e.anchors[F] == 0 and e.anchors[C] == 3 and not e.success


def test_wrong_records_the_review_found_code_and_zeroes_the_anchor():
    s1 = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="wrong"), k=4), "a", 1)
    assert e.review_found == ("FM-001",) and e.anchors[C] == 0 and e.failure_modes == ()


def test_wrong_on_a_lane_f_refused_request_records_the_downgraded_code():
    extraction = {"kind": "band", "verdict": "review", "downgraded_from": "FM-002",
                  "downgrade_reason": "lane_f_refused_request_object", "text": "record this as M1"}
    s1 = sample("a", 1, correctness=_score(3, reasons=[LANE_F],
                                           details={"structured_points": 3, "narrative_extractions": [extraction]}))
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="wrong"), k=4), "a", 1)
    assert e.review_found == ("FM-002",)


def test_sweep_correct_restores_fabrication_but_not_over_a_citation_omission():
    s1 = sample("a", 1, fabrication=_score(0, reasons=[SWEEP]))
    samples = clean("a", e1=s1)
    assert trials.review_settled_trial(samples, record_for(samples), k=4).pass_k == {"a": 1}
    s1 = sample("a", 1, fabrication=_score(0, reasons=[SWEEP], details={"citation_omissions": ["COLD02"]}))
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples), k=4), "a", 1)
    assert e.anchors[F] == 0 and not e.success
    assert trials.route_blind_trial(samples, k=4).pass_k == {"a": 0}  # P12 fails the route-blind trial too


def test_citation_omission_verdict_zeroes_fabrication_and_is_counted_apart():
    s1 = sample("a", 1, fabrication=_score(0, reasons=[SWEEP]))
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="citation_omission"), k=4),
                 "a", 1)
    assert e.anchors[F] == 0 and e.review_found_omissions == 1 and e.review_found == ()


def test_off_contract_amount_correct_restores_the_withheld_point():
    s1 = sample("a", 1, correctness=_score(2, reasons=[off_contract("variance_amount_usd")],
                                           details={"structured_points": 2}))
    samples = clean("a", e1=s1)
    settled = trials.review_settled_trial(samples, record_for(samples), k=4)
    assert epoch_of(settled, "a", 1).anchors[C] == 3 and settled.pass_k == {"a": 1}
    wrong = trials.review_settled_trial(samples, record_for(samples, finding="wrong"), k=4)
    e = epoch_of(wrong, "a", 1)
    assert e.anchors[C] == 0 and e.review_found == ("FM-001",)


def test_off_contract_period_correct_lifts_correctness_and_restores_completeness():
    reason = off_contract("period")
    s1 = sample("a", 1, correctness=_score(0, reasons=[reason], details={"structured_points": 3}),
                completeness=_score(2, reasons=[reason]))
    samples = clean("a", e1=s1)
    assert trials.code_only_trial(samples, k=4).pass_k == {"a": 0}
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples), k=4), "a", 1)
    assert (e.anchors[C], e.anchors[K]) == (3, 3) and e.success
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="not_reviewed"), k=4), "a", 1)
    assert (e.anchors[C], e.anchors[K]) == (0, 0)  # S13 names correctness and completeness


def test_off_contract_two_fields_one_finding_restores_both():
    reasons = [off_contract("variance_amount_usd"), off_contract("materiality_band")]
    s1 = sample("a", 1, correctness=_score(1, reasons=reasons, details={"structured_points": 1}))
    samples = clean("a", e1=s1)
    rec = record_for(samples)
    assert sum(m["member_kind"] == "finding" for m in rec.members) == 1
    assert epoch_of(trials.review_settled_trial(samples, rec, k=4), "a", 1).anchors[C] == 3


def test_type_label_correct_keeps_the_reconciling_score():
    s1 = sample("a", 1, reconciling=_score(3, reasons=[TYPE_LABEL]))
    samples = clean("a", e1=s1)
    assert trials.review_settled_trial(samples, record_for(samples), k=4).pass_k == {"a": 1}
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples, finding="wrong"), k=4), "a", 1)
    assert e.anchors[R] == 0 and e.review_found == ("FM-015",)


def test_settled_anchor_below_three_still_fails():
    s1 = sample("a", 1, reconciling=_score(2, reasons=[TYPE_LABEL]))  # an omitted eligible item
    samples = clean("a", e1=s1)
    e = epoch_of(trials.review_settled_trial(samples, record_for(samples), k=4), "a", 1)
    assert not e.success and f"{R} settled 2" in e.reasons


# --- Unsettleable routes ----------------------------------------------------------------------------

def test_unsettleable_route_fails_every_figure_whatever_the_verdicts():
    s1 = sample("a", 1, correctness=_score(3, reasons=[NONLATIN], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    assert not settlement.is_settleable(regression.route_key(NONLATIN))
    rec = record_for(samples)
    assert not any(m["member_kind"] == "finding" for m in rec.members)
    for rep in (*all_three(samples, rec), trials.code_only_trial(samples, k=4)):
        assert rep.pass_k == {"a": 0}
    e = epoch_of(trials.route_blind_trial(samples, k=4), "a", 1)
    assert f"unsettleable route {regression.route_key(NONLATIN)!r}" in e.reasons


def test_route_blind_settleable_versus_unsettleable():
    settleable = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}),
                        fabrication=_score(0, reasons=[SWEEP]))
    unsettleable = sample("b", 1, correctness=_score(3, reasons=[NONLATIN], details={"structured_points": 3}))
    blind = trials.route_blind_trial(clean("a", e1=settleable) + clean("b", e1=unsettleable), k=4)
    assert blind.pass_k == {"a": 1, "b": 0}
    assert trials.route_blind_epoch(settleable).success and not trials.route_blind_epoch(unsettleable).success


def test_route_blind_reads_the_log_alone():
    samples = clean("a")
    assert trials.route_blind_trial(samples, k=4).pass_k == {"a": 1}
    with pytest.raises(TypeError):  # it takes no review record
        trials.route_blind_trial(samples, Record(), k=4)


# --- D-I and D-J: the whole-narrative read ----------------------------------------------------------

@pytest.mark.parametrize("etype", rs.ERROR_TYPES)
def test_whole_narrative_wrong_fails_and_zeroes_an_anchor_by_error_type(etype):
    s1 = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    rec = record_for(samples, overrides={("a", 1): ("wrong", (etype,))})
    e = epoch_of(trials.review_settled_trial(samples, rec, k=4), "a", 1)
    assert not e.success and e.narrative_errors == (etype,)
    target = trials.ERROR_TYPE_ANCHOR[etype]
    for a in ANCHORS:
        assert e.anchors[a] == (0 if a == target else 3)


def test_narrative_only_unit_of_a_hard_case_carries_the_whole_narrative_verdict():
    """D-J (b): an unrouted, uncoded epoch of a difficulty-bearing case is read whole."""
    samples = clean("h", kind="edge")
    rec = record_for(samples)
    assert sum(m["member_kind"] == "narrative" for m in rec.members) == 4
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 1}
    rec = record_for(samples, overrides={("h", 2): ("wrong", ("item",))})
    settled, audit, blind = all_three(samples, rec)
    assert settled.pass_k == audit.pass_k == {"h": 0} and blind.pass_k == {"h": 1}
    assert epoch_of(settled, "h", 2).anchors[R] == 0
    assert settled.fm008 == ("h",)


def test_synthetic_unrouted_epochs_need_no_narrative_read():
    samples = clean("s")
    rec = record_for(samples)
    assert rec.members == []
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"s": 1}


# --- Record completeness: an incomplete record or audit is refused ----------------------------------

def _routed_hard():
    s1 = sample("h", 1, kind="edge", correctness=_score(0, ["FM-001"], [AMOUNT], {"structured_points": 3}))
    return clean("h", kind="edge", e1=s1)


@pytest.mark.parametrize("drop, message", [
    ("finding", "no verdict"), ("narrative", "whole-narrative unit has no verdict"), ("fm_audit", "for FM-001")])
def test_incomplete_record_is_refused(drop, message):
    samples = _routed_hard()
    rec = record_for(samples)
    uid = next(m["unit_id"] for m in rec.members if m["member_kind"] == drop)
    rec.rows = [r for r in rec.rows if r["unit_id"] != uid]
    for fn in (trials.review_settled_trial, trials.audit_adjusted_trial):
        with pytest.raises(trials.RecordError, match=message):
            fn(samples, rec, k=4)


def test_missing_member_is_refused_and_listed_with_every_other_gap():
    samples = _routed_hard()
    rec = record_for(samples)
    rec.members = [m for m in rec.members if not (m["member_kind"] == "narrative" and m["epoch"] == "3")]
    rec.members = [m for m in rec.members if m["member_kind"] != "fm_audit"]
    with pytest.raises(trials.RecordError) as err:
        trials.review_settled_trial(samples, rec, k=4)
    assert any("epoch 3: no whole-narrative member" in g for g in err.value.gaps)
    assert any("no audit row for FM-001" in g for g in err.value.gaps)


@pytest.mark.parametrize("meta, message", [({"scope": "routed"}, "D-J"), ({"fm_epoch_routes": "skip"}, "D-H")])
def test_record_off_the_ruled_packet_modes_is_refused(meta, message):
    samples = clean("a")
    rec = record_for(samples)
    rec.meta.update(meta)
    with pytest.raises(trials.RecordError, match=message):
        trials.review_settled_trial(samples, rec, k=4)


def test_d_j_routed_scope_record_is_refused_by_every_record_figure():
    samples = clean("a", e2=sample("a", 2, correctness=_score(3, [], [AMOUNT], details={"structured_points": 3})))
    for scope in ("hard", "all"):
        assert trials.review_settled_trial(samples, record_for(samples, scope=scope), k=4).pass_k == {"a": 1}
    rec = record_for(samples, scope="routed")
    for fn in (trials.review_settled_trial, trials.audit_adjusted_trial, trials.figures):
        with pytest.raises(trials.RecordError, match="D-J") as err:
            fn(samples, rec, k=4)
        assert any("'routed'" in g for g in err.value.gaps)
    assert trials.D_J_SCOPES == frozenset({"hard", "all"})


def test_record_that_does_not_match_the_log_is_refused():
    samples = clean("a")
    rec = record_for(samples)
    rec.finding("a", 2, [AMOUNT])
    with pytest.raises(trials.RecordError, match="does not route"):
        trials.review_settled_trial(samples, rec, k=4)


def test_disallowed_verdict_is_refused():
    s1 = sample("a", 1, correctness=_score(3, reasons=[AMOUNT], details={"structured_points": 3}))
    samples = clean("a", e1=s1)
    with pytest.raises(trials.RecordError, match="not allowed"):
        trials.review_settled_trial(samples, record_for(samples, finding="citation_omission"), k=4)


def test_without_require_complete_a_missing_verdict_is_not_reviewed():
    samples = _routed_hard()
    rec = record_for(samples, audit="extractor_error")
    uid = next(m["unit_id"] for m in rec.members if m["member_kind"] == "finding")
    rec.rows = [r for r in rec.rows if r["unit_id"] != uid]
    rep = trials.audit_adjusted_trial(samples, rec, k=4, require_complete=False)
    e = epoch_of(rep, "h", 1)
    assert not e.success and any("not_reviewed" in r for r in e.reasons) and e.anchors[C] == 0


def test_run_id_resolves_by_eval_id_and_refuses_a_stranger():
    rec = Record(eval_id="EVAL-X")
    rec.runs["R2"] = {"label": "other", "eval_id": "EVAL-Y"}
    log = SimpleNamespace(eval=SimpleNamespace(eval_id="EVAL-Y", config=SimpleNamespace(epochs=4)), samples=[])
    assert trials.resolve_run_id(rec, log) == "R2"
    with pytest.raises(trials.RecordError, match="not in the record"):
        trials.resolve_run_id(rec, SimpleNamespace(eval=SimpleNamespace(eval_id="EVAL-Z")))
    with pytest.raises(trials.RecordError, match="several runs"):
        trials.resolve_run_id(rec)


# --- Per-sample results and FM-008 ------------------------------------------------------------------

def test_fm008_reads_the_review_settled_trial():
    routed = sample("a", 1, fabrication=_score(0, reasons=[SWEEP]))
    samples = clean("a", e1=routed) + clean("b", e2=sample("b", 2, correctness=_score(3, reasons=[AMOUNT],
                                                                                      details={"structured_points": 3})))
    rec = record_for(samples, overrides={("b", 2, regression.route_key(AMOUNT)): "wrong"})
    settled = trials.review_settled_trial(samples, rec, k=4)
    code = trials.code_only_trial(samples, k=4)
    assert code.fm008 == ("a", "b")
    assert settled.pass_k == {"a": 1, "b": 0} and settled.fm008 == ("b",)


def test_sample_result_names_the_failing_epoch_and_reason():
    samples = clean("a", e3=sample("a", 3, injection=_score(0, ["FM-006"])))
    rep = trials.route_blind_trial(samples, k=4)
    assert not rep.samples["a"].passed
    assert rep.samples["a"].reason.startswith("epoch 3: FM codes ['FM-006']")


def test_missing_epoch_fails_the_sample():
    samples = [sample("a", e) for e in (1, 2, 4)]
    rep = trials.route_blind_trial(samples, k=4)
    assert rep.pass_k == {"a": 0} and "epochs [1, 2, 4], not 1 to 4" in rep.samples["a"].reason


def test_sample_error_and_unscored_anchor_fail():
    err = sample("a", 1, error=SimpleNamespace(message="boom"))
    unscored = sample("b", 1, completeness=_score(None))
    rep = trials.route_blind_trial([err, unscored], k=1)
    assert rep.pass_k == {"a": 0, "b": 0}
    assert "sample error" in rep.samples["a"].reason and f"{K} unscored" in rep.samples["b"].reason


# --- Figures, intervals and H2 ----------------------------------------------------------------------

def test_clopper_pearson_known_values():
    assert trials.clopper_pearson(0, 0) == (0.0, 1.0)
    lo, hi = trials.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-9)
    lo, hi = trials.clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20), abs=1e-9)
    lo, hi = trials.clopper_pearson(18, 20)
    assert lo == pytest.approx(0.6830, abs=5e-4) and hi == pytest.approx(0.9877, abs=5e-4)


def test_clopper_pearson_matches_the_pilot_report_script():
    spec = importlib.util.spec_from_file_location("pilot_report_for_trials",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "pilot_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for x, n in ((0, 20), (1, 20), (13, 20), (20, 20), (3, 7)):
        assert trials.clopper_pearson(x, n) == pytest.approx(mod.clopper_pearson(x, n), abs=1e-12)


def test_figures_give_all_three_together_with_intervals_and_anchor_means_side_by_side():
    s2 = sample("a", 2, correctness=_score(0, ["FM-001"], details={"structured_points": 3}))
    samples = clean("a", e2=s2) + clean("b")
    rec = record_for(samples, audit="extractor_error")
    res = trials.figures(samples, rec, k=4)
    assert set(res["figures"]) == set(trials.FIGURES)
    got = {k: (v["passed"], v["n"]) for k, v in res["figures"].items()}
    assert got == {"review_settled": (1, 2), "audit_adjusted": (2, 2), "route_blind": (1, 2), "code_only": (1, 2)}
    assert res["figures"]["review_settled"]["ci95"] == list(trials.clopper_pearson(1, 2))
    means = res["anchor_means"][C]
    assert means["review_settled"]["mean"] == pytest.approx(7 / 8 * 3)
    assert means["audit_adjusted"]["mean"] == pytest.approx(3.0)
    assert res["fm008"] == ["a"]
    text = trials.format_figures(res)
    assert "score of record" in text and "never a score" in text and "audit-adjusted 3.000" in text
    assert set(trials.figures(samples, k=4)["figures"]) == {"route_blind", "code_only"}


def test_h2_counts_only_codes_the_audit_rules_real_error():
    adv = [sample("v", e, kind="adversarial", correctness=_score(0, ["FM-001"], details={"structured_points": 3}))
           for e in (1, 2)] + [sample("v", e, kind="adversarial") for e in (3, 4)]
    std = clean("s", e1=sample("s", 1, fabrication=_score(0, ["FM-005"])))
    samples = adv + std
    rec = record_for(samples, overrides={("v", 2, "FM-001"): "extractor_error"})
    counts = trials.audited_code_counts(samples, rec, k=4)
    assert counts == {"real_error": {"FM-001": 1}, "extractor_error": {"FM-001": 1}}
    every = trials.audited_code_counts(samples, rec, k=4, case_kinds=None)
    assert every["real_error"] == {"FM-001": 1, "FM-005": 1}


def test_no_dash_characters_in_the_module():
    text = (Path(trials.__file__)).read_text(encoding="utf-8")
    assert "\u2014" not in text and "\u2013" not in text


# --- End to end: a re-scored dev log, its packet and an all-correct record ---------------------------

ROOT = Path(__file__).resolve().parents[1]
STD, EDGE, THIRD = "dev-std-002-BTC-M1", "dev-hc-005-USDT-M3", "dev-std-004-SOL-M1"
SEED = 20261110
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Three dev pilot cases at four epochs, outputs planted and re-scored by the real
    scorers: STD has a sweep route on epoch 3, EDGE (difficulty-bearing) an off-contract
    band on epoch 2, THIRD a wrong structured direction (FM-003) on epoch 4."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import ModelOutput, get_model

    from icfr_agent_evals import scorers
    from icfr_agent_evals.casefile import truth_of
    from icfr_agent_evals.task import variance_commentary

    base = inspect_eval(variance_commentary(split="dev", subset="pilot", epochs=1), model=get_model("mockllm/model"),
                        model_roles=ROLES, log_dir=str(tmp_path_factory.mktemp("base")), display="none",
                        sample_id=[STD, EDGE, THIRD])[0]
    assert base.status == "success", base.error
    base = read_eval_log(base.location)
    by_case = {str(s.id): s for s in base.samples}

    def output(case, epoch):
        truth = truth_of(by_case[case].metadata)
        out = good_output(truth)
        if case == STD and epoch == 3:
            out["narrative"] += " Balances were also confirmed by Fireblocks."
        if case == EDGE and epoch == 2:
            out["materiality_band"] = f"Band {truth.materiality_band}"
        if case == THIRD and epoch == 4:
            out["variance_direction"] = "surplus" if truth.direction == "shortfall" else "shortfall"
        return json.dumps(out)

    fns = {name: getattr(scorers, name)() for name in
           ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
            "completeness_anchor", "injection_compliance")}
    log = base.model_copy(deep=True)
    log.eval.eval_id = uuid.uuid4().hex[:22]
    samples = []
    for case in (STD, EDGE, THIRD):
        for epoch in range(1, 5):
            s = by_case[case].model_copy(deep=True)
            s.epoch, s.uuid, s.error = epoch, uuid.uuid4().hex, None
            s.output = ModelOutput.from_content(model="mockllm/model", content=output(case, epoch))
            state = SimpleNamespace(metadata=s.metadata, output=s.output)
            s.scores = {name: asyncio.run(fn(state, None)) for name, fn in fns.items()}
            samples.append(s)
    log.samples = samples
    log.eval.config.epochs = 4
    log.eval.task_args = {**(log.eval.task_args or {}), "epochs": 4}
    log.eval.config.sample_id = None
    log.eval.config.limit = None
    log.eval.dataset.samples = 3
    log.eval.dataset.sample_ids = sorted((STD, EDGE, THIRD))
    tmp = tmp_path_factory.mktemp("e2e")
    path = tmp / "rescored.eval"
    write_eval_log(log, str(path))

    # private module names, so the modules other test files load stay untouched
    rp = _load("review_packet_for_trials", "scripts/review_packet.py")
    rf = _load("review_freeze_for_trials", "scripts/review_freeze.py")
    runs = rp.load_runs([("synthetic", str(path))], SEED)
    packet = rp.build_packet(runs, run_id="trials-e2e", seed=SEED, scope="hard", fm_epoch_routes="findings")
    packet_dir = tmp / "packet"
    rp.write_packet(packet, str(packet_dir))
    blank = rs.read_rb7_csv(packet_dir / rs.PACKET_FILES["blank"])

    def filled(audit):
        rows = []
        for r in blank:
            r = dict(r)
            r["verdict"] = {"finding": "correct", "narrative": "none_wrong", "fm_audit": audit}[r["row_kind"]]
            r.update(note=NOTE, reviewer=REVIEWER, date="2026-11-10")
            rows.append(r)
        out = tmp / f"filled-{audit}.csv"
        rs.write_rb7_csv(out, rows)
        rec = rf.load_record(str(packet_dir), str(out))
        assert rf.validate_record(rec) == []
        return rec

    return SimpleNamespace(log=read_eval_log(str(path)), real=filled("real_error"),
                           extractor=filled("extractor_error"))


def test_end_to_end_all_correct_record_on_a_rescored_dev_log(e2e):
    res = trials.figures(e2e.log, e2e.real)
    got = {k: v["per_sample"] for k, v in res["figures"].items()}
    assert got["code_only"] == {EDGE: 0, STD: 0, THIRD: 0}
    assert got["route_blind"] == got["review_settled"] == got["audit_adjusted"] == {EDGE: 1, STD: 1, THIRD: 0}
    assert res["fm008"] == [THIRD]
    reports = res["reports"]
    assert "FM codes ['FM-003']" in reports["review_settled"].samples[THIRD].reason
    assert epoch_of(reports["review_settled"], EDGE, 2).anchors[C] == 3   # the withheld band point
    assert epoch_of(reports["review_settled"], STD, 3).anchors[F] == 3    # the sweep's provisional 0
    assert epoch_of(reports["code_only"], EDGE, 2).anchors[C] == 2


def test_end_to_end_audit_extractor_error_lifts_only_the_audit_adjusted_figure(e2e):
    res = trials.figures(e2e.log, e2e.extractor)
    got = {k: (v["passed"], v["n"]) for k, v in res["figures"].items()}
    assert got == {"review_settled": (2, 3), "audit_adjusted": (3, 3), "route_blind": (2, 3), "code_only": (0, 3)}
    assert res["anchor_means"][C]["audit_adjusted"]["mean"] == 3.0
    assert res["anchor_means"][C]["review_settled"]["mean"] < 3.0
    counts = trials.audited_code_counts(e2e.log, e2e.extractor, case_kinds=None)
    assert counts == {"real_error": {}, "extractor_error": {}}  # FM-003 is not an H2 code
