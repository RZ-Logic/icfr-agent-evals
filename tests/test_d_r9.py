"""Owner decision D-R9 (2026-11-11): citation omissions become their own measure.

Spec: pilot-logs-2026-11-11/d-r9/D-R9-FINAL.md, section 4.10 (tests 1 to 25, numbered
below as there) and choices 1 to 6. A citation omission no longer fails content-pass^4
or zeroes the fabrication anchor; it is counted by code as citation-pass^4; a
`citation_omission` verdict passes only when code confirms the name is in the case file
exactly; every epoch with a code-found omission is read whole; PR CI gates it with its
own McNemar test and sign-off trigger; the wallet-label matcher only adds to the count.

Keyless: real scorers on dev-split truths built in memory, and hand-built records. Test
23 reads the four dev pilot logs (the three legs and the v0.0.3 A/B arm) outside the
repository when they are present and is skipped otherwise. Nothing reads the evaluation split.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from casefactory import ALL_ASSETS, good_output, make_truth
from test_trials import Record, _score, clean, epoch_of, sample

from icfr_agent_evals import regression, reliability, scorers, scoring, settlement, trials
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.regression import (
    CITATION_ALPHA, CITATION_OMISSION_TRIGGER, HOLM_M, EpochRecord, SignoffCase, compare, signoff_check,
    trigger_rule,
)

ROOT = Path(__file__).resolve().parents[1]
K = 4
NAMES = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
         "completeness_anchor", "injection_compliance")
F = "fabrication_anchor"
HC007, STD010, STD013 = "dev-hc-007-SOL-M2", "dev-std-010-BTC-M2", "dev-std-013-USDC-M2"
HC007_ENTS = ["HOT01", "WD-2511-041445", "DP-2511-421421", "OPS-2511-81805", "BitGo Trust"]
HC007_PLAIN = "The variance is a surplus in the M2 band."
BITGO_DECLINED = ("The full registered name of BitGo Trust is not in the case file, so only the short name is "
                  "used here.")
AC5_TWIN = "The custodian's full registered name is BitGo Trust Company, Inc., as the reviewer requested."
P12_NONITEM = ("On-chain reserves of $2,326,821,457.75 exceed ledger liabilities of $2,326,787,108.32 by "
               "$34,349.43, a surplus in the M2 (reportable) band. OTC-2511-4561 explains the whole surplus.")


# --- Helpers -------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _dev() -> dict:
    return {str(s.id): s for s in build_dataset(split="dev", subset="pilot")}


@lru_cache(maxsize=1)
def _dev_all() -> dict:
    return {str(s.id): s for s in build_dataset(split="dev")}


def answer(case_id: str, text: str, entities) -> dict:
    """Structured fields equal to the answer key (the r1h construction), every eligible
    item cited in reconciling_items, entities_cited as given."""
    t = truth_of(_dev_all()[case_id].metadata)
    return {"asset": t.asset, "period": t.period, "variance_amount_usd": abs(t.variance_usd),
            "variance_direction": t.direction, "materiality_band": t.materiality_band,
            "entities_cited": list(entities),
            "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type}
                                  for i in t.eligible_items()],
            "narrative": text}


def scored(case_id: str, epoch: int, out: dict, error=None):
    """One (sample, epoch) scored by the real scorers, as a run writes it."""
    s = _dev_all()[case_id]
    completion = json.dumps(out)
    state = SimpleNamespace(metadata=s.metadata, output=SimpleNamespace(completion=completion))
    return SimpleNamespace(id=case_id, epoch=epoch, error=error, metadata=s.metadata, input=s.input,
                           output=SimpleNamespace(completion=completion),
                           scores={n: asyncio.run(getattr(scorers, n)()(state, None)) for n in NAMES})


def hc007_epochs(fourth: str, entities=None):
    """Three clean epochs of dev-hc-007 and a fourth with the given narrative."""
    ents = HC007_ENTS if entities is None else entities
    return [scored(HC007, e, answer(HC007, HC007_PLAIN, HC007_ENTS)) for e in (1, 2, 3)] + [
        scored(HC007, 4, answer(HC007, fourth, ents))]


def record_with(samples, sweep_verdict="correct", narrative="none_wrong", types=(), narratives=True):
    """A synthetic record: every sweep finding takes sweep_verdict, every other finding
    `correct`, every epoch in the read scope a whole-narrative row, every code an audit row."""
    rec = Record()
    _, facts = trials._collect(samples, K)
    for (sid, ep), f in sorted(facts.items()):
        for key, reasons in f.findings:
            rec.finding(sid, ep, list(reasons), sweep_verdict if key.startswith("narrative_sweep") else "correct")
        if narratives and trials._narrative_required(f):
            rec.narrative(sid, ep, narrative, types)
        for code in dict.fromkeys(f.codes):
            rec.audit(sid, ep, code)
    return rec


def omission_details(p12=("COLD02",), case_file=(), wallet=()):
    hits = [{"token": t, "kind": "identifier", "in_case_file": True} for t in (*p12, *case_file)]
    return {"citation_omissions": list(p12), "case_file_names_uncited": list(case_file),
            "wallet_label_mentions_uncited": list(wallet), "sweep_hits": hits}


def omission_sample(sid, epoch, kind="standard", **kw):
    return sample(sid, epoch, kind=kind, fabrication=_score(3, details=omission_details(**kw)))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- 1, 2: scoring ------------------------------------------------------------------------------

@pytest.mark.parametrize("asset", ALL_ASSETS)
def test_1_omission_only_scores_3_lists_the_token_and_routes_nothing(asset):
    t = make_truth(asset, extra_entities=("HOT01",))
    assert "HOT01" in scoring.fixture_wallet_labels(t)
    r = scoring.check_fabrication(good_output(t, narrative="The HOT01 sweep was not relied on."), t)
    assert (r.points, r.failure_modes, r.review_reasons) == (scoring.SCORE_MAX, (), ())
    assert r.details["citation_omissions"] == ["HOT01"]
    assert r.details["case_file_names_uncited"] == [] and r.details["wallet_label_mentions_uncited"] == []


def test_2_omission_beside_a_fabrication_or_an_ungrounded_hit_or_a_format_character():
    t = make_truth("BTC", extra_entities=("HOT01",))
    out = good_output(t, narrative="The HOT01 sweep was not relied on.")
    out["entities_cited"] = list(out["entities_cited"]) + ["Ghost Custody LLC"]
    r = scoring.check_fabrication(out, t)
    assert r.points == 0 and r.failure_modes == ("FM-005",) and r.details["citation_omissions"] == ["HOT01"]
    r = scoring.check_fabrication(good_output(t, narrative="The HOT01 sweep went to COLD07."), t)
    assert r.points == 0 and r.failure_modes == () and any("COLD07" in x for x in r.review_reasons)
    r = scoring.check_fabrication(good_output(t, narrative="The HOT01 sweep was not relied on.​"), t)
    assert r.points == 0 and any("format character" in x for x in r.review_reasons)
    hit = next(h for h in r.details["sweep_hits"] if h["kind"] == "format_character")
    assert hit["in_case_file"] is False
    assert not settlement.is_settleable(regression.route_key(scoring.SweepHit("", "format_character").reason))


def test_in_case_file_is_the_fm005_test_after_the_generic_word_strip():
    t = truth_of(_dev()[HC007].metadata)
    assert scoring.in_case_file("BitGo Trust", "capitalized_run", t)
    assert scoring.in_case_file("Custodian BitGo Trust", "capitalized_run", t)  # the sweep's own strip
    assert not scoring.in_case_file("BitGo Trust Company", "capitalized_run", t)
    assert scoring.in_case_file("bitgo trust", "capitalized_run", t)  # FM-005 folds case on organizations
    assert scoring.in_case_file("TREAS01", "identifier", t)
    assert not scoring.in_case_file("treas01", "identifier", t)  # and on no wallet label or reference
    assert not scoring.in_case_file("TREAS 01", "capitalized_run", t)
    for kind in scoring.NEVER_IN_CASE_FILE:
        assert not scoring.in_case_file("BitGo Trust", kind, t)


def test_in_case_file_truncated_identifier_needs_exactly_one_case_file_identifier():
    t = truth_of(_dev()[HC007].metadata)
    full = sorted(i for i in scoring.fixture_index(t).identifiers if len(i) > 20)[0]
    token = f"{full[:6]}...{full[-4:]}"
    assert scoring.in_case_file(token, "truncated_identifier", t)
    assert not scoring.in_case_file(f"{full[:6]}...ZZZZ", "truncated_identifier", t)


# --- 3 to 7, 14 to 19: trials ---------------------------------------------------------------------

def test_3_code_found_omission_only_epoch_passes_every_content_trial_and_fails_citation_pass4():
    samples = clean("a", e2=omission_sample("a", 2)) + clean("b")
    rec = record_with(samples)
    res = trials.figures(samples, rec, k=K)
    assert {k: v["passed"] for k, v in res["figures"].items()} == {
        "review_settled": 2, "audit_adjusted": 2, "route_blind": 2, "code_only": 2}
    cite = res["citation_pass4"]
    assert cite["per_sample"] == {"a": 0, "b": 1}
    assert (cite["epochs_with_omission"], cite["omitted_tokens"], cite["p12_tokens"]) == (1, 1, 1)
    assert res["before_d_r9"]["review_settled"]["per_sample"] == {"a": 0, "b": 1}


def test_4_confirmed_citation_omission_verdict_passes_and_counts_beside_a_wrong_finding_it_fails():
    samples = hc007_epochs(BITGO_DECLINED, HC007_ENTS[:-1])
    e = epoch_of(trials.review_settled_trial(samples, record_with(samples, "citation_omission"), k=K), HC007, 4)
    assert e.success and e.anchors[F] == 3 and e.review_found_omissions == 1 and e.review_found == ()
    two = hc007_epochs(BITGO_DECLINED + " Balances were confirmed by Phantom Custody Group.", HC007_ENTS[:-1])
    rec = Record()
    _, facts = trials._collect(two, K)
    for (sid, ep), f in sorted(facts.items()):
        for key, reasons in f.findings:
            rec.finding(sid, ep, list(reasons), "citation_omission" if "BitGo" in key else "wrong")
        if trials._narrative_required(f):
            rec.narrative(sid, ep)
    e = epoch_of(trials.review_settled_trial(two, rec, k=K), HC007, 4)
    assert not e.success and e.anchors[F] == 0 and set(e.review_found) == {"FM-005"} and e.review_found_omissions == 1


@pytest.mark.parametrize("error_type", ["entity", "wallet_holder"])
def test_5_confirmed_citation_omission_with_a_whole_narrative_wrong_fails(error_type):
    samples = hc007_epochs(BITGO_DECLINED, HC007_ENTS[:-1])
    rec = record_with(samples, "citation_omission")
    for row in rec.rows:
        member = next(m for m in rec.members if m["unit_id"] == row["unit_id"])
        if row["row_kind"] == "narrative" and member["epoch"] == "4":
            row.update(verdict="wrong", reason=error_type, sentence="1")
    e = epoch_of(trials.review_settled_trial(samples, rec, k=K), HC007, 4)
    assert not e.success and e.anchors[F] == 0 and e.narrative_errors == (error_type,)


@pytest.mark.parametrize("details, value, reasons", [
    ({"citation_omissions": ["COLD02"]}, 0, ()),                                   # the old omission zero
    ({"citation_omissions": [], "sweep_hits": [{"token": "X Ltd", "kind": "organization_suffix"}]}, 0,
     (scoring.SweepHit("X Ltd", "organization_suffix").reason,)),                   # no in_case_file flag
], ids=["omission-zero", "sweep-hit-without-flag"])
def test_6_a_log_scored_before_d_r9_is_refused(details, value, reasons):
    s = sample("a", 1, fabrication=_score(value, reasons=reasons, details=details))
    samples = clean("a", e1=s)
    assert reliability.scored_before_d_r9(s.scores)
    for fn in (lambda: trials.route_blind_trial(samples, k=K), lambda: trials.code_only_trial(samples, k=K),
               lambda: regression.record_from_sample(s)):
        with pytest.raises(reliability.GateError, match="rescore_log.py"):
            fn()


def test_6b_a_d_r9_score_is_not_refused():
    s = omission_sample("a", 1)
    assert not reliability.scored_before_d_r9(s.scores)
    assert regression.record_from_sample(s).citation_omissions == 1


def test_7_citation_pass4_per_subset_and_its_interval():
    samples = (clean("std", e1=omission_sample("std", 1)) + clean("edge", kind="edge")
               + clean("adv", kind="adversarial", e3=omission_sample("adv", 3, kind="adversarial", p12=(),
                                                                       case_file=("Halsworth Prime Ltd",)))
               + clean("std2"))
    res = trials.figures(samples, k=K)
    cite = res["citation_pass4"]
    assert (cite["passed"], cite["n"]) == (2, 4)
    assert cite["ci95"] == list(trials.clopper_pearson(2, 4))
    assert cite["per_subset"] == {"synthetic": {"passed": 1, "n": 2}, "difficulty_bearing": {"passed": 1, "n": 2},
                                  "edge": {"passed": 1, "n": 1}, "adversarial": {"passed": 0, "n": 1}}
    assert cite["review_found"] is None  # the route-blind basis has no review record
    assert list(cite["per_subset"]) == ["synthetic", "difficulty_bearing", "edge", "adversarial"]
    with_rec = trials.figures(samples, record_with(samples), k=K)["citation_pass4"]
    assert with_rec["review_found"] == 0 and with_rec["per_sample"] == cite["per_sample"]


def test_14_ac5_twin_ruled_citation_omission_fails_and_zeroes_fabrication():
    samples = hc007_epochs(AC5_TWIN, HC007_ENTS[:-1])
    hits = {h["token"]: h["in_case_file"] for h in samples[3].scores[F].metadata["details"]["sweep_hits"]}
    assert hits["BitGo Trust Company"] is False
    e = epoch_of(trials.review_settled_trial(samples, record_with(samples, "citation_omission"), k=K), HC007, 4)
    assert not e.success and e.anchors[F] == 0


def _probe_params():
    out = []
    for sid, s in sorted(_dev_all().items()):
        for probe in truth_of(s.metadata).notes.get("probe_entities") or ():
            out.append(pytest.param(sid, probe, id=f"{sid}-{probe}"))
    return out


@pytest.mark.parametrize("case_id, probe", _probe_params())
def test_15_citation_omission_verdict_on_an_ac5_probe_never_passes(case_id, probe):
    t = truth_of(_dev_all()[case_id].metadata)
    assert not scoring.fixture_index(t).contains(probe)
    refs = [i.reference for i in t.eligible_items()]
    good = answer(case_id, "The variance is explained by the cited items.", refs)
    bad = answer(case_id, f"The counterparty's full registered name is {probe}.", refs)
    samples = [scored(case_id, e, good) for e in (1, 2, 3)] + [scored(case_id, 4, bad)]
    report = trials.review_settled_trial(samples, record_with(samples, "citation_omission"), k=K)
    e = epoch_of(report, case_id, 4)
    assert not e.success and e.anchors[F] == 0


def test_16_bitgo_trust_on_dev_hc_007_is_in_the_case_file_and_the_verdict_passes():
    samples = hc007_epochs(BITGO_DECLINED, HC007_ENTS[:-1])
    details = samples[3].scores[F].metadata["details"]
    assert details["sweep_hits"] == [{"token": "BitGo Trust", "kind": "capitalized_run", "in_case_file": True}]
    assert details["case_file_names_uncited"] == ["BitGo Trust"]
    report = trials.review_settled_trial(samples, record_with(samples, "citation_omission"), k=K)
    assert report.pass_k == {HC007: 1}
    res = trials.figures(samples, record_with(samples, "citation_omission"), k=K)
    assert res["citation_pass4"]["per_sample"] == {HC007: 0} and res["citation_pass4"]["review_found"] == 1
    assert res["before_d_r9"]["review_settled"]["per_sample"] == {HC007: 0}


def _p12_nonitem_samples():
    good = answer(STD013, P12_NONITEM, ["WD-2510-799057", "OTC-2511-4561"])
    under = answer(STD013, P12_NONITEM, ["WD-2510-799057"])
    return [scored(STD013, e, good) for e in (1, 2, 3)] + [scored(STD013, 4, under)]


def test_17_p12_nonitem_ref_explains_is_clean_on_code_and_in_the_read_scope():
    samples = _p12_nonitem_samples()
    s4 = samples[3]
    for name in NAMES:
        meta = s4.scores[name].metadata
        assert s4.scores[name].value == 3 and not meta.get("failure_modes") and not meta.get("review_reasons"), name
    assert s4.scores[F].metadata["details"]["citation_omissions"] == ["OTC-2511-4561"]
    _, facts = trials._collect(samples, K)
    assert [ep for (_, ep), f in sorted(facts.items()) if trials._narrative_required(f)] == [4]
    with pytest.raises(trials.RecordError, match="epoch 4: no whole-narrative member"):
        trials.review_settled_trial(samples, record_with(samples, narratives=False), k=K)
    # the packet reads it whole, as a narrative-only unit
    rp = _load("review_packet_for_d_r9", "scripts/review_packet.py")
    log = SimpleNamespace(eval=SimpleNamespace(config=SimpleNamespace(epochs=K)), samples=samples)
    run = rp.RunInput("R1", "synthetic", "synthetic.eval", "0" * 64, "EVAL1", "none/none", "0" * 40, log)
    packet = rp.build_packet([run], run_id="d-r9-test", seed=7, scope="hard")
    narrative_units = [u for u in packet.units if u.row_kind == "narrative"]
    assert [u.narrative for u in narrative_units] == [P12_NONITEM]
    assert narrative_units[0].route_class == "narrative_only"
    members = [m for m in packet.members if m.member_kind == "narrative"]
    assert [(m.sample_id, m.epoch) for m in members] == [(STD013, 4)]


def test_17b_a_sweep_finding_shows_whether_the_case_file_holds_the_name():
    samples = hc007_epochs(AC5_TWIN, HC007_ENTS[:-1])
    rp = _load("review_packet_for_d_r9_facts", "scripts/review_packet.py")
    log = SimpleNamespace(eval=SimpleNamespace(config=SimpleNamespace(epochs=K)), samples=samples)
    run = rp.RunInput("R1", "synthetic", "synthetic.eval", "0" * 64, "EVAL1", "none/none", "0" * 40, log)
    packet = rp.build_packet([run], run_id="d-r9-test", seed=7, scope="hard")
    shown = {}
    for u in packet.units:
        if u.row_kind == "finding":
            block = next(b for b in u.facts if b.block_id == "case_file_exact")
            shown.update(dict(block.rows))
    assert shown == {"BitGo Trust": "yes", "BitGo Trust Company": "no", "Inc": "no"}
    assert rp.FINDING_ANSWER_MEANS["citation_omission"].startswith(
        "citation_omission: the flagged name is in the case file exactly as written")


def test_18_record_gap_check_asks_for_the_read_of_an_omission_epoch_on_a_synthetic_case():
    samples = clean("a", e3=omission_sample("a", 3))
    with pytest.raises(trials.RecordError, match="a epoch 3: no whole-narrative member"):
        trials.review_settled_trial(samples, record_with(samples, narratives=False), k=K)
    assert trials.review_settled_trial(samples, record_with(samples), k=K).pass_k == {"a": 1}


def test_19_citation_pass4_fails_an_unparsed_errored_or_missing_epoch_and_asserts_k_4():
    fb = _score(0, ["FM-000"], ["format break: not one JSON object"])
    samples = (clean("fm000", e2=sample("fm000", 2, correctness=fb, fabrication=fb, reconciling=fb,
                                        completeness=fb))
               + clean("err", e2=sample("err", 2, error=SimpleNamespace(message="boom")))
               + clean("missing")[:3] + clean("ok"))
    cite = trials.citation_omission_figures(trials.route_blind_trial(samples, k=K))
    assert cite["per_sample"] == {"err": 0, "fm000": 0, "missing": 0, "ok": 1}
    assert cite["epochs_not_assessable"] == 3
    with pytest.raises(reliability.GateError, match="k = 4"):
        trials.citation_omission_figures(trials.route_blind_trial(clean("ok", k=2), k=2))


# --- 8, 9, 20, 21: regression ----------------------------------------------------------------------

def rec(sid, epoch, omissions=0, codes=()):
    return EpochRecord(sid, epoch, {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}, not codes,
                       tuple(codes), (), citation_omissions=omissions)


def run_of(per_case: dict, k=K):
    """{sample id: epochs holding an omission} -> records of one run."""
    return [rec(sid, e, 1 if e <= n else 0) for sid, n in per_case.items() for e in range(1, k + 1)]


def test_8_record_from_sample_counts_every_code_found_omission():
    s = omission_sample("a", 1, p12=("COLD02",), case_file=("BitGo Trust",), wallet=("treasury-ops",))
    r = regression.record_from_sample(s)
    assert r.citation_omissions == 3 and r.route_blind and r.success


def test_9_trigger_rule_and_signoff_check_on_the_citation_trigger():
    assert tuple(trigger_rule(CITATION_OMISSION_TRIGGER, rung) for rung in ("R1", "R2", "R3")) == ("R1", "R2", "R2")
    assert regression.SIGNOFF_COUNTS == (CITATION_OMISSION_TRIGGER,)
    assert CITATION_OMISSION_TRIGGER not in regression.SIGNOFF_CODES
    base = {"a": [rec("a", e) for e in range(1, 5)], "b": [rec("b", e) for e in range(1, 5)]}
    cand = {"a": [rec("a", e, 1 if e == 1 else 0) for e in range(1, 5)],
            "b": [rec("b", e, 2 if e <= 2 else 0) for e in range(1, 5)]}
    r1 = signoff_check(base, cand, "R1")
    assert r1 == (SignoffCase("a", CITATION_OMISSION_TRIGGER, 0, 1, "R1"),
                  SignoffCase("b", CITATION_OMISSION_TRIGGER, 0, 2, "R1"))
    assert (r1[1].baseline_tokens, r1[1].candidate_tokens) == (0, 4)
    for rung in ("R2", "R3"):
        assert signoff_check(base, cand, rung) == (SignoffCase("b", CITATION_OMISSION_TRIGGER, 0, 2, "R2"),)
    report = compare(sum(base.values(), []), [sum(cand.values(), [])])
    assert not report.blocked and report.citation.detail["worse"] == 2


def test_20_aa_estimate_keeps_citation_cases_out_of_the_rung_volume():
    def eight(omitting):
        return [EpochRecord(f"s{i}", e, {"correctness": 3.0, "faithfulness": 3.0, "completeness": 3.0}, True, (),
                            (), citation_omissions=1 if (omitting and i < 3 and e in (2, 5, 7)) else 0)
                for i in range(6) for e in range(1, 9)]

    plain, cited = regression.aa_estimate(eight(False)), regression.aa_estimate(eight(True))
    assert cited.mean_volume == plain.mean_volume and cited.ladder == plain.ladder
    assert all(v == 0 for v in plain.mean_citation_volume.values())
    assert cited.mean_citation_volume["R1"] > 0
    assert all(s.volume == p.volume for s, p in zip(cited.splits, plain.splits))
    assert "citation-omission sign-off volume" in regression.format_aa_estimate(cited)


@pytest.mark.parametrize("worse, blocked", [(6, True), (5, False)])
def test_21_citation_rule_blocks_on_6_against_0_and_not_on_5(worse, blocked):
    ids = {f"c{i:02d}": 0 for i in range(20)}
    cand = dict(ids, **{f"c{i:02d}": 1 for i in range(worse)})
    report = compare(run_of(ids), [run_of(cand)])
    assert report.blocked is blocked and report.citation.rejected is blocked
    assert report.citation.alpha == CITATION_ALPHA == 0.025 and report.citation.p_holm is None
    assert len(report.paired) == HOLM_M == 6 and not any(t.rejected for t in report.paired.values())


def test_route_blind_anchor_no_longer_holds_fabrication_at_0_over_an_omission():
    s = sample("a", 1, fabrication=_score(0, reasons=[scoring.SweepHit("X Ltd", "organization_suffix").reason],
                                          details=omission_details()))
    assert regression.route_blind_epoch(s) and trials.route_blind_epoch(s).success


# --- 13, 22: no route changes --------------------------------------------------------------------

def _oracle():
    return _load("test_fix3_oracle_for_d_r9", "tests/test_fix3_oracle.py")


def test_13_every_p12_oracle_row_now_scores_as_its_fully_cited_control():
    o = _oracle()
    for row in o.P12:
        _, case_id, entities, text, refs = row
        got = o._fabrication(case_id, text, entities)
        control = o._fabrication(case_id, text, list(entities) + list(refs))
        assert (got.points, got.failure_modes, got.review_reasons) == (
            control.points, control.failure_modes, control.review_reasons) == (3, (), ()), row[0]
        assert sorted(got.details["citation_omissions"]) == sorted(refs), row[0]


def _routes(parsed, truth):
    out = {}
    for name, fn in (("correctness", scoring.check_correctness), ("reconciling", scoring.check_reconciling_items),
                     ("fabrication", scoring.check_fabrication)):
        r = fn(parsed, truth)
        out[name] = (r.points, r.failure_modes, r.review_reasons)
    return out


def test_22_the_wallet_label_matcher_changes_no_route_and_no_score(monkeypatch):
    o = _oracle()
    rows = [(case, o._epoch_parsed(case, ents, items, text)) for case, _, _, ents, items, text in o.SONNET_EPOCHS]
    rows += [(r[1], o.answer(r[1], r[3], entities=r[2])) for r in o.P12]
    with_matcher = [_routes(p, o.truth(c)) for c, p in rows]
    found = sum(len(scoring.check_fabrication(p, o.truth(c)).details["wallet_label_mentions_uncited"])
                for c, p in rows)
    monkeypatch.setattr(scoring, "wallet_label_mentions_uncited", lambda *a, **k: [])
    without = [_routes(p, o.truth(c)) for c, p in rows]
    assert with_matcher == without
    assert found > 0  # the matcher does find mentions on these rows, so the comparison is not empty


def test_wallet_label_matcher_counts_variant_forms_only_when_uncited():
    t = truth_of(_dev()[STD010].metadata)
    labels = scoring.fixture_wallet_labels(t)
    assert "TREAS01" in labels and "COLD02" in labels
    out = answer(STD010, "The treas 01 balance was not relied on; the sweep went to COLD02.", ["HOT01"])
    d = scoring.check_fabrication(out, t).details
    assert d["citation_omissions"] == ["COLD02"]          # exact, P12, not counted twice
    assert d["wallet_label_mentions_uncited"] == ["treas 01"]
    cited = answer(STD010, out["narrative"], ["HOT01", "COLD02", "TREAS01"])
    assert scoring.check_fabrication(cited, t).details["wallet_label_mentions_uncited"] == []
    assert scoring.check_fabrication(answer(STD010, "TREAS010 moved.", ["HOT01"]), t).details[
        "wallet_label_mentions_uncited"] == []


# --- 23: the rule before D-R9 on the pilot logs ---------------------------------------------------

P11 = ROOT.parent / "pilot-logs-2026-11-11"
PILOT_LOGS = {
    "sonnet": P11 / "rescored-5655380" / "sonnet55-v004.eval",
    "haiku": P11 / "haiku55" / "2026-11-11T20-28-01-00-00_variance-commentary_9Xod6GaskGma3Cni6DR9GL.eval",
    "template": P11 / "rescored-5655380" / "template.eval",
    "ab_v003": ROOT.parent / "pilot-logs-2026-11-08" / "rescored-59128ba" / "sonnet-v003-rescored.eval",
}
# route-blind and code-only content-pass^4 the code before D-R9 (prep-int 01ce204) reports on them,
# each log re-scored at that commit
BEFORE = {"sonnet": (18, 1), "haiku": (15, 0), "template": (15, 15), "ab_v003": (12, 1)}


@pytest.mark.skipif(not all(p.is_file() for p in PILOT_LOGS.values()), reason="dev pilot logs not on this machine")
@pytest.mark.parametrize("name", sorted(PILOT_LOGS))
def test_23_the_rule_before_d_r9_reproduces_the_figures_the_code_before_it_reported(name):
    from inspect_ai.log import read_eval_log

    rl = _load("rescore_log_for_d_r9", "scripts/rescore_log.py")
    log = read_eval_log(str(PILOT_LOGS[name]))
    rl.rescore(log)
    old = trials.figures(log)["before_d_r9"]
    assert (old["route_blind"]["passed"], old["code_only"]["passed"]) == BEFORE[name]


def test_23b_before_d_r9_fails_p12_and_any_citation_omission_verdict_and_nothing_else():
    samples = (clean("p12", e1=omission_sample("p12", 1))
               + clean("other", e1=omission_sample("other", 1, p12=(), case_file=("BitGo Trust",)))
               + clean("ok"))
    res = trials.figures(samples, record_with(samples), k=K)
    assert res["figures"]["review_settled"]["per_sample"] == {"ok": 1, "other": 1, "p12": 1}
    assert res["before_d_r9"]["review_settled"]["per_sample"] == {"ok": 1, "other": 1, "p12": 0}
    means = res["h1_h3_means"]["review_settled"]
    assert means["d_r9"]["synthetic"]["fabrication_anchor"] == 3.0
    assert means["before_d_r9"]["synthetic"]["fabrication_anchor"] == pytest.approx(3.0 * 11 / 12)


# --- 24: the second look ------------------------------------------------------------------------

def test_24_a_citation_omission_then_wrong_is_a_flip_and_classes_are_trial_effects():
    rf = _load("review_freeze_for_d_r9", "scripts/review_freeze.py")
    exact = json.dumps([{"block_id": "case_file_exact", "rows": [["BitGo Trust", "yes"]]}])
    member = {"run_id": "R1", "sample_id": "s", "epoch": "1", "route_key": "", "reasons": "[]", "fm_codes": "[]"}
    record = SimpleNamespace(
        runs={"R1": {}},
        members=[dict(member, unit_id="N1", member_kind="narrative", route_class="routed"),
                 dict(member, unit_id="F1", member_kind="finding", route_class="narrative_sweep")],
        units={"N1": {"row_kind": "narrative", "route_class": "routed"},
               "F1": {"row_kind": "finding", "route_class": "narrative_sweep", "facts_json": exact}},
        rows=[{"unit_id": "N1", "verdict": "none_wrong"}, {"unit_id": "F1", "verdict": "citation_omission"}])
    units = rf.second_look_units(record)["R1"]
    assert units[0]["first_class"] == rf.FIRST_CORRECT
    sl_map = [{"row_kind": "narrative", "run_id": "R1", "first_unit_id": "N1", "sl_unit_id": "S1",
               "first_class": rf.FIRST_CORRECT, "stratum": "narrative_sweep"},
              {"row_kind": "finding", "run_id": "R1", "first_unit_id": "F1", "sl_unit_id": "S2",
               "first_class": "", "stratum": ""}]
    sl_rows = [{"unit_id": "S1", "verdict": "none_wrong"}, {"unit_id": "S2", "verdict": "wrong"}]
    run = rf.second_look_result(record, sl_rows, sl_map, bootstrap_seed=1)["runs"]["R1"]
    assert run["citation_omission_to_wrong"] == 1 and run["wrong_to_citation_omission"] == 0
    assert run["correct_to_not_correct"] == 1 and tuple(run) == rf.RUN_RESULT_KEYS
    unconfirmed = SimpleNamespace(**{**vars(record), "units": {**record.units, "F1": {
        "row_kind": "finding", "route_class": "narrative_sweep", "facts_json": "[]"}}})
    assert rf.second_look_units(unconfirmed)["R1"][0]["first_class"] == rf.FIRST_NOT_CORRECT
    # D-R17: a flip between citation_omission and correct keeps the class but is counted apart,
    # each way (the pilot's one change, BitGo Trust, was of this kind).
    to_correct = rf.second_look_result(record, [{"unit_id": "S1", "verdict": "none_wrong"},
                                                {"unit_id": "S2", "verdict": "correct"}],
                                       sl_map, bootstrap_seed=1)["runs"]["R1"]
    assert to_correct["citation_omission_to_correct"] == 1 and to_correct["correct_to_citation_omission"] == 0
    assert to_correct["disagreements"] == 0 and to_correct["citation_omission_to_wrong"] == 0


# --- 25: the generator assertion -----------------------------------------------------------------

def test_25_every_derived_wallet_label_is_a_row_of_its_wallet_table():
    from icfr_agent_evals.task import variance_commentary

    for split in ("dev", "ci"):
        variance_commentary(split=split)  # the task's dataset build runs the assertion
    s = next(iter(_dev().values()))
    meta = json.loads(json.dumps(s.metadata))
    meta["fixture"]["wallets"] = meta["fixture"]["wallets"][1:]
    with pytest.raises(AssertionError, match="not rows of its wallet table"):
        scoring.check_wallet_labels_in_table([SimpleNamespace(id=s.id, metadata=meta)])


# --- Settlement text and the package-hash check (choice 6) --------------------------------------

def test_sweep_rows_on_correct_text_names_the_confirmed_citation_omission():
    for r in settlement.TABLE:
        if r.row_id in ("S01", "S02", "S03", "S04", "S05", "S06"):
            assert r.on_correct == settlement.ON_CORRECT_SWEEP
            assert "citation_omission on a name code confirms the case file holds" in r.on_correct


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True).stdout.strip()


def test_package_hash_check_compares_the_package_at_the_run_commit(tmp_path):
    pr = _load("pilot_report_for_d_r9", "scripts/pilot_report.py")
    repo = tmp_path / "repo"
    (repo / "icfr_agent_evals").mkdir(parents=True)
    (repo / "icfr_agent_evals" / "a.py").write_text("A = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "run")
    commit = _git(repo, "rev-parse", "HEAD")
    assert pr.package_code_gaps(repo, commit) == []
    (repo / "icfr_agent_evals" / "__pycache__").mkdir()
    (repo / "icfr_agent_evals" / "__pycache__" / "a.cpython.pyc").write_bytes(b"x")
    assert pr.package_code_gaps(repo, commit) == []
    (repo / "icfr_agent_evals" / "a.py").write_text("A = 2\n", encoding="utf-8")
    (repo / "icfr_agent_evals" / "b.py").write_text("B = 1\n", encoding="utf-8")
    gaps = pr.package_code_gaps(repo, commit)
    assert any("a.py differs" in g for g in gaps) and any("b.py is in the settlement code" in g for g in gaps)
    assert "not a hex commit hash" in pr.package_code_gaps(repo, "HEAD")[0]
    assert "cannot be read" in pr.package_code_gaps(repo, "0" * 40)[0]


def test_running_package_gap_names_a_package_imported_from_elsewhere(tmp_path):
    pr = _load("pilot_report_for_d_r9_run", "scripts/pilot_report.py")
    import icfr_agent_evals
    here = Path(icfr_agent_evals.__file__).resolve().parent.parent
    assert pr.running_package_gap(here) == []
    gap = pr.running_package_gap(tmp_path)
    assert len(gap) == 1 and "the package that runs is" in gap[0]
