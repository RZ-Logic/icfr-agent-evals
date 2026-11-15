"""scripts/review_packet.py: the grouped review packet (build spec section 7, builder B3).

Hand-built logs only: one small mockllm run of three dev pilot cases gives a real
log header and real samples; each test then writes its own logs with
inspect_ai.log.write_eval_log into tmp_path, with planted worker outputs and score
metadata (review_reasons, failure_modes, details). One log carries the model string
anthropic/claude-sonnet-5-5 and another mockllm/model. No API key, no model call,
dev split only; nothing reads the evaluation split. Verdicts are never filled here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score

from icfr_agent_evals import review_facts as rf
from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import scoring
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.task import variance_commentary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_packet.py"
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
STD, EDGE, THIRD = "dev-std-002-BTC-M1", "dev-hc-005-USDT-M3", "dev-std-004-SOL-M1"
MODEL_A, MODEL_B = "anthropic/claude-sonnet-5-5", "mockllm/model"
ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor",
           "faithfulness_anchor", "completeness_anchor")
SEED = 20261109


def _load_script():
    spec = importlib.util.spec_from_file_location("review_packet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_script()


# --- Hand-built logs ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_log(tmp_path_factory):
    log = inspect_eval(
        variance_commentary(split="dev", subset="pilot", epochs=1),
        model=get_model("mockllm/model"), model_roles=ROLES,
        log_dir=str(tmp_path_factory.mktemp("base")), display="none", sample_id=[STD, EDGE, THIRD],
    )[0]
    assert log.status == "success", log.error
    full = read_eval_log(log.location)
    assert sorted(str(s.id) for s in full.samples) == sorted([STD, EDGE, THIRD])
    return full


@pytest.fixture(scope="module")
def truths(base_log):
    return {str(s.id): truth_of(s.metadata) for s in base_log.samples}


def answer(truth, narrative: str, *, items=None, entities=None, **over) -> str:
    items = truth.eligible_items() if items is None else items
    d = {
        "asset": truth.asset, "period": truth.period, "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction, "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in items)] if entities is None else entities,
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
        "narrative": narrative,
    }
    d.update(over)
    return json.dumps(d)


@dataclass
class E:
    """One planted (case, epoch): the completion and per-scorer (reasons, fms, details)."""
    case: str
    epoch: int
    completion: str
    scores: dict = field(default_factory=dict)
    format_break: bool = False


def _scores(e: E) -> dict:
    if e.format_break:
        reason = "format break: not one JSON object: Expecting value: line 1 column 1 (char 0)"
        meta = {"failure_modes": ["FM-000"], "requires_human_review": True, "review_reasons": [reason],
                "format_break": True, "details": {"parse_error": reason}}
        return {name: Score(value=0, metadata=copy.deepcopy(meta)) for name in ANCHORS}
    out = {}
    for name in ANCHORS:
        reasons, fms, details = e.scores.get(name, ((), (), {}))
        out[name] = Score(value=0 if fms else 3, metadata={
            "failure_modes": list(fms), "requires_human_review": bool(reasons),
            "review_reasons": list(reasons), "details": dict(details)})
    return out


def make_log(base, path: Path, model: str, epochs: list[E], mutate=None) -> str:
    log = base.model_copy(deep=True)
    log.eval.model = model
    log.eval.eval_id = uuid.uuid4().hex[:22]
    by_case = {str(s.id): s for s in base.samples}
    samples = []
    for e in epochs:
        s = by_case[e.case].model_copy(deep=True)
        s.epoch = e.epoch
        s.uuid = uuid.uuid4().hex
        s.output = ModelOutput.from_content(model=model, content=e.completion)
        s.scores = _scores(e)
        s.error = None
        samples.append(s)
    log.samples = samples
    k = max(e.epoch for e in epochs)
    log.eval.config.epochs = k
    log.eval.task_args = {**(log.eval.task_args or {}), "epochs": k}
    # the planted cases are the whole run: no --sample-id, and the header counts them
    cases = sorted({e.case for e in epochs})
    log.eval.config.sample_id = None
    log.eval.config.limit = None
    log.eval.dataset.samples = len(cases)
    log.eval.dataset.sample_ids = cases
    if mutate:
        mutate(log)
    write_eval_log(log, str(path))
    return str(path)


def route(*reasons, fms=(), details=None, scorer="correctness_anchor"):
    return {scorer: (tuple(reasons), tuple(fms), details or {})}


def build(base, tmp_path, runs: dict[str, tuple[str, list[E]]], *, seed=SEED, **kw):
    specs = []
    for label, (model, epochs) in runs.items():
        specs.append((label, make_log(base, tmp_path / f"{label}.eval", model, epochs)))
    loaded = rp.load_runs(specs, seed)
    return rp.build_packet(loaded, run_id="test-packet", seed=seed, **kw)


def units(p, kind):
    return [u for u in p.units if u.row_kind == kind]


def members(p, kind):
    return [m for m in p.members if m.member_kind == kind]


AMOUNT = "amount_role_unresolved: $5,000.00"
S1 = "Reserves fell short of ledger liabilities by $5,000.00 at the cutoff."
S2 = "The custodian confirmed the wallet balances."


# --- Grouping -------------------------------------------------------------------------------

def test_grouping_across_epochs_and_runs(base_log, truths, tmp_path):
    t = truths[STD]
    same = [E(STD, ep, answer(t, f"{S1} {S2}"), route(AMOUNT)) for ep in (1, 2)]
    p = build(base_log, tmp_path / "one", {"alpha": (MODEL_A, same), "beta": (MODEL_B, copy.deepcopy(same))})
    f = units(p, "finding")
    assert len(f) == 1 and f[0].routed_sentences == (S1,) and f[0].highlight == ("$5,000.00",)
    assert len([m for m in members(p, "finding") if m.unit_id == f[0].unit_id]) == 4
    assert len(units(p, "narrative")) == 1 and len(members(p, "narrative")) == 4

    # changing the cited items on one epoch splits the unit (and the narrative unit)
    other = t.eligible_items()[:-1] if len(t.eligible_items()) > 1 else []
    split = [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT)),
             E(STD, 2, answer(t, f"{S1} {S2}", items=other), route(AMOUNT))]
    p2 = build(base_log, tmp_path / "two", {"alpha": (MODEL_A, split)})
    assert len(units(p2, "finding")) == 2 and len(units(p2, "narrative")) == 2

    # a routed sentence differing only by an en dash or a zero-width space splits it
    dash = S1.replace("the cutoff", "period-end")
    variants = [dash, dash.replace("-", "\u2013"), S1.replace("fell short", "fell\u200b short")]
    eps = [E(STD, i + 1, answer(t, f"{s} {S2}"), route(AMOUNT)) for i, s in enumerate(variants)]
    p3 = build(base_log, tmp_path / "three", {"alpha": (MODEL_A, eps)})
    f3 = units(p3, "finding")
    assert len(f3) == 3
    assert sorted(u.routed_sentences for u in f3) == sorted((s,) for s in variants)


def test_structured_finding_groups_without_narrative(base_log, truths, tmp_path):
    t = truths[STD]
    item = t.eligible_items()[0]
    s14 = f"reconciling_items[0] {item.reference!r} type {'other kind'!r} differs from {item.type!r}"
    items = [{"reference": item.reference, "amount_usd": item.amount_usd, "type": "other kind"}]
    eps = [E(STD, ep, answer(t, n, reconciling_items=items), route(s14, scorer="reconciling_items_anchor"))
           for ep, n in ((1, f"{S1} {S2}"), (2, "A different narrative entirely. It routes the same field."))]
    p = build(base_log, tmp_path / "s14", {"alpha": (MODEL_A, eps)})
    f = units(p, "finding")
    assert len(f) == 1 and f[0].routed_sentences == ()
    assert len(units(p, "narrative")) == 2
    assert len(f[0].narrative_unit_ids) == 2

    inel = next((i for i in t.items if not t.eligibility(i).eligible), t.items[0])
    s16 = f"entities_cited names ineligible item {inel.reference} that the narrative does not discuss"
    for reason, scorer in ((s16, "reconciling_items_anchor"), ("both_directions_named", "correctness_anchor")):
        eps = [E(STD, ep, answer(t, n), route(reason, scorer=scorer))
               for ep, n in ((1, f"{S1} {S2}"), (2, "Another narrative. With two sentences."))]
        pp = build(base_log, tmp_path / reason[:20].replace(" ", "_"), {"alpha": (MODEL_A, eps)})
        ff = units(pp, "finding")
        assert len(ff) == 2, reason
        assert all(u.routed_sentences == () for u in ff)


def test_sentence_with_two_keys(base_log, truths, tmp_path):
    t = truths[STD]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT, "direction_negated: fell short"))]
    p = build(base_log, tmp_path / "two", {"alpha": (MODEL_A, eps)})
    f = units(p, "finding")
    assert len(f) == 2 and {u.route_key for u in f} == {"code_review: amount_role_unresolved",
                                                        "code_review: direction_negated"}
    nid = units(p, "narrative")[0].unit_id
    cards = rp.finding_cards(p, nid)
    assert cards == [((S1,), tuple(u.unit_id for u in f))]
    blank = [r for r in rp.blank_verdict_rows(p) if r["row_kind"] == "finding"]
    assert len(blank) == 2 and all(r["sentence"] == S1 for r in blank)


def test_off_contract_one_finding_two_fields(base_log, truths, tmp_path):
    t = truths[STD]
    reasons = (scoring._off_contract("variance_amount_usd"), scoring._off_contract("materiality_band"))
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(*reasons))]
    p = build(base_log, tmp_path / "oc", {"alpha": (MODEL_A, eps)})
    f = units(p, "finding")
    assert len(f) == 1 and f[0].route_key == "code_review: off_contract_value"
    assert len(f[0].correct_means) == 2
    rows = rp.settlement.rows_for(f[0].route_key, f[0].reasons)
    assert [r.row_id for r in rows] == ["S11", "S12"]
    inter = tuple(v for v in rs.FINDING_VERDICTS if all(v in r.allowed_verdicts for r in rows))
    assert f[0].allowed_verdicts == inter and "citation_omission" not in inter


def test_unsettleable_not_a_unit(base_log, truths, tmp_path):
    t = truths[STD]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route("narrative_too_long: 9000 characters"))]
    p = build(base_log, tmp_path / "u", {"alpha": (MODEL_A, eps)})
    assert units(p, "finding") == []
    (m,) = members(p, "unsettleable")
    assert m.unit_id == "" and m.route_key == "code_review: narrative_too_long"
    assert len(units(p, "narrative")) == 1 and len(members(p, "narrative")) == 1
    assert rp.summary(p)["totals"]["unsettleable_by_key"] == {"code_review: narrative_too_long": 1}


def test_fm_listing_not_a_finding(base_log, truths, tmp_path):
    t = truths[STD]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}", entities=["Acme Custody LLC"]),
             route(scoring.FABRICATION_REVIEW, fms=("FM-005",), details={"fabricated": ["Acme Custody LLC"]},
                   scorer="fabrication_anchor"))]
    p = build(base_log, tmp_path / "fm5", {"alpha": (MODEL_A, eps)})
    assert members(p, "finding") == [] and members(p, "unsettleable") == []
    (a,) = units(p, "fm_audit")
    assert a.fm_code == "FM-005" and a.fm_reasons == ("Acme Custody LLC",)
    assert a.allowed_verdicts == rs.AUDIT_VERDICTS
    assert a.narrative_unit_ids == (units(p, "narrative")[0].unit_id,)


def test_scope_routed_hard_all(base_log, truths, tmp_path):
    eps = []
    for case in (STD, EDGE):
        t = truths[case]
        eps.append(E(case, 1, answer(t, f"{S1} {S2}"), route(AMOUNT)))
        eps.append(E(case, 2, answer(t, "Unrouted narrative two.")))
        eps.append(E(case, 3, answer(t, "Unrouted narrative three.")))
    counts = {}
    for scope in rs.WHOLE_NARRATIVE_SCOPES:
        p = build(base_log, tmp_path / scope, {"alpha": (MODEL_A, copy.deepcopy(eps))}, scope=scope)
        counts[scope] = (len(members(p, "narrative")), len(units(p, "narrative")))
        assert p.scope == scope
    assert counts == {"routed": (2, 2), "hard": (4, 4), "all": (6, 6)}


def _narrative_only_eps(truths):
    """EDGE is difficulty-bearing, STD is not. EDGE: epoch 1 routed; epochs 2 and 3 the same
    unrouted narrative; epoch 4 a second unrouted narrative; epoch 5 an FM-coded epoch with no
    route; epoch 6 unrouted with the very text of the routed epoch 1. STD: one routed epoch,
    five unrouted epochs."""
    te, ts = truths[EDGE], truths[STD]
    return [
        E(EDGE, 1, answer(te, f"{S1} {S2}"), route(AMOUNT)),
        E(EDGE, 2, answer(te, "Unrouted narrative two.")),
        E(EDGE, 3, answer(te, "Unrouted narrative two.")),
        E(EDGE, 4, answer(te, "Unrouted narrative four.")),
        E(EDGE, 5, answer(te, "Coded narrative five."), route(fms=("FM-003",))),
        E(EDGE, 6, answer(te, f"{S1} {S2}")),
        E(STD, 1, answer(ts, f"{S1} {S2}"), route(AMOUNT)),
        E(STD, 2, answer(ts, "Std unrouted narrative two.")),
        E(STD, 3, answer(ts, "Std unrouted narrative three.")),
        E(STD, 4, answer(ts, "Std unrouted narrative three.")),
        E(STD, 5, answer(ts, "Std unrouted narrative five.")),
        E(STD, 6, answer(ts, "Std unrouted narrative six.")),
    ]


def test_default_scope_is_hard(base_log, truths, tmp_path):
    assert rs.DEFAULT_WHOLE_NARRATIVE_SCOPE == "hard"
    assert rs.WHOLE_NARRATIVE_SCOPES == ("routed", "hard", "all")  # routed and all stay as explicit options
    assert rp.parser().get_default("whole_narrative_scope") == "hard"
    assert "default hard" in SCRIPT.read_text(encoding="utf-8").split('"""')[1]
    eps = _narrative_only_eps(truths)
    default = build(base_log, tmp_path / "d", {"alpha": (MODEL_A, copy.deepcopy(eps))})
    hard = build(base_log, tmp_path / "h", {"alpha": (MODEL_A, copy.deepcopy(eps))}, scope="hard")
    routed = build(base_log, tmp_path / "r", {"alpha": (MODEL_A, copy.deepcopy(eps))}, scope="routed")
    assert default.scope == "hard"
    assert default.units == hard.units and default.members == hard.members
    assert len(units(routed, "narrative")) < len(units(default, "narrative"))


def test_narrative_only_units_on_hard_cases_only(base_log, truths, tmp_path):
    eps = _narrative_only_eps(truths)
    p = build(base_log, tmp_path / "n", {"alpha": (MODEL_A, eps)})
    by_text = {u.narrative: u for u in units(p, "narrative")}
    only_ids = {u.unit_id for u in units(p, "narrative") if u.route_class == rs.NARRATIVE_CLASS_ONLY}
    only = {u.narrative for u in units(p, "narrative") if u.unit_id in only_ids}
    # one unit per distinct unrouted, uncoded narrative of the hard case; none on the non-hard case
    assert only == {"Unrouted narrative two.", "Unrouted narrative four."}
    assert not any(t.startswith("Std unrouted") for t in by_text)
    assert by_text["Unrouted narrative two."].case_id == EDGE
    two = by_text["Unrouted narrative two."].unit_id
    assert len([m for m in members(p, "narrative") if m.unit_id == two]) == 2
    # no finding, audit unit or route member touches a narrative-only unit
    assert not any(only_ids & set(u.narrative_unit_ids) for u in p.units if u.row_kind != "narrative")
    only_epochs = {(m.run_id, m.sample_id, m.epoch) for m in members(p, "narrative") if m.unit_id in only_ids}
    assert not any((m.run_id, m.sample_id, m.epoch) in only_epochs for m in p.members if m.member_kind != "narrative")
    # the FM-coded epoch is its own class; the routed narrative is routed
    assert by_text["Coded narrative five."].route_class == rs.NARRATIVE_CLASS_FM_CODED
    assert by_text[f"{S1} {S2}"].route_class == rs.NARRATIVE_CLASS_ROUTED
    # an unrouted epoch that repeats a routed narrative joins that unit; its member says what the epoch is
    routed_unit = by_text[f"{S1} {S2}"].unit_id
    classes = sorted(m.route_class for m in members(p, "narrative") if m.unit_id == routed_unit)
    assert classes == [rs.NARRATIVE_CLASS_ONLY, rs.NARRATIVE_CLASS_ROUTED]
    # explicit scopes keep their meaning: routed drops narrative-only units, all adds the non-hard ones
    r = build(base_log, tmp_path / "r", {"alpha": (MODEL_A, copy.deepcopy(eps))}, scope="routed")
    assert not [u for u in units(r, "narrative") if u.route_class == rs.NARRATIVE_CLASS_ONLY]
    a = build(base_log, tmp_path / "a", {"alpha": (MODEL_A, copy.deepcopy(eps))}, scope="all")
    assert {u.narrative for u in units(a, "narrative") if u.route_class == rs.NARRATIVE_CLASS_ONLY} == {
        "Unrouted narrative two.", "Unrouted narrative four.", "Std unrouted narrative two.",
        "Std unrouted narrative three.", "Std unrouted narrative five.", "Std unrouted narrative six."}


def test_narrative_only_one_unit_across_runs(base_log, truths, tmp_path):
    te = truths[EDGE]
    eps = [E(EDGE, 1, answer(te, "Same unrouted narrative.")), E(EDGE, 2, answer(te, "Same unrouted narrative."))]
    p = build(base_log, tmp_path / "x", {"alpha": (MODEL_A, eps), "beta": (MODEL_B, copy.deepcopy(eps))})
    (u,) = units(p, "narrative")
    assert u.route_class == rs.NARRATIVE_CLASS_ONLY
    assert len(members(p, "narrative")) == 4 and units(p, "finding") == [] and units(p, "fm_audit") == []


def test_narrative_only_marked_in_every_file_and_count(base_log, truths, tmp_path):
    eps = _narrative_only_eps(truths)
    p = build(base_log, tmp_path / "b", {"alpha": (MODEL_A, eps)})
    out = tmp_path / "files"
    rp.write_packet(p, str(out))
    rows = rs.read_csv(out / rs.PACKET_FILES["packet"], rs.PACKET_COLUMNS)
    narr = [r for r in rows if r["row_kind"] == "narrative"]
    assert {r["route_class"] for r in narr} == set(rs.NARRATIVE_CLASSES)
    assert all(r["route_class"] in ("narrative_sweep", "code_review") for r in rows if r["row_kind"] == "finding")
    assert all(r["route_class"] == "" for r in rows if r["row_kind"] == "fm_audit")
    mem = rs.read_csv(out / rs.PACKET_FILES["members"], rs.MEMBER_COLUMNS)
    assert {m["route_class"] for m in mem if m["member_kind"] == "narrative"} == set(rs.NARRATIVE_CLASSES)
    assert sum(1 for m in mem if m["member_kind"] == "narrative" and m["route_class"] == "narrative_only") == 4
    t = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))["totals"]
    assert t["narrative_units_by_class"] == {"routed": 2, "fm_coded": 1, "narrative_only": 2}
    assert sum(t["narrative_units_by_class"].values()) == t["narrative_units"] == 5
    assert sum(t["narrative_members_by_class"].values()) == t["narrative_members"]
    assert t["narrative_members_by_class"]["narrative_only"] == 4  # epochs 2, 3, 4, 6 of the hard case
    # a narrative row of the blank verdict file carries no class: the reviewer records none
    blank = [r for r in rs.read_rb7_csv(out / rs.PACKET_FILES["blank"]) if r["row_kind"] == "narrative"]
    assert len(blank) == 5 and all(r["route_class"] == "" and r["route_key"] == "" for r in blank)
    q = rp.load_packet(str(out))
    assert q.units == p.units and q.members == p.members


def test_fm_epoch_routes_switch(base_log, truths, tmp_path):
    t = truths[STD]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT, fms=("FM-003",)))]
    p = build(base_log, tmp_path / "f", {"alpha": (MODEL_A, eps)})
    assert len(members(p, "finding")) == 1 and len(units(p, "fm_audit")) == 1
    p2 = build(base_log, tmp_path / "s", {"alpha": (MODEL_A, copy.deepcopy(eps))}, fm_epoch_routes="skip")
    assert members(p2, "finding") == [] and units(p2, "finding") == []
    (m,) = members(p2, "fm_epoch_route_skipped")
    assert m.unit_id == "" and m.route_key == "code_review: amount_role_unresolved"
    assert len(units(p2, "fm_audit")) == 1 and len(units(p2, "narrative")) == 1


def test_audit_evidence(base_log, truths, tmp_path):
    t = truths[STD]
    extractions = [{"kind": "direction", "text": "fell short", "verdict": "ok"},
                   {"kind": "amount", "text": "$9,999.00", "verdict": "FM-001"}]
    eps = [E(STD, 1, "not json at all", format_break=True),
           E(STD, 2, answer(t, f"{S1} {S2}"),
             route(fms=("FM-001",), details={"narrative_extractions": extractions}))]
    p = build(base_log, tmp_path / "a", {"alpha": (MODEL_A, eps)})
    by_code = {u.fm_code: u for u in units(p, "fm_audit")}
    assert by_code["FM-000"].fm_reasons[0].startswith("format break: ")
    assert by_code["FM-001"].fm_reasons == ("amount: $9,999.00", rp.STRUCTURED_LINE)
    narr = {u.unit_id: u for u in units(p, "narrative")}
    raw = narr[by_code["FM-000"].narrative_unit_ids[0]]
    assert raw.parsed_ok is False and raw.narrative == "not json at all"
    blank = {r["unit_id"]: r for r in rp.blank_verdict_rows(p)}
    assert blank[by_code["FM-000"].unit_id]["route_key"] == "FM-000"
    assert blank[by_code["FM-001"].unit_id]["reason"] == "amount: $9,999.00 | " + rp.STRUCTURED_LINE


def test_routed_sentences_rules():
    narrative = "First sentence names WD-1. Second has $5,000.00 here! Third: ETH and SOL."
    assert rp.routed_sentences("code_review: amount_role_unresolved", [AMOUNT], narrative) == (
        ("Second has $5,000.00 here!",), ("$5,000.00",))
    assert rp.routed_sentences("code_review: both_directions_named", ["both_directions_named"], narrative) == ((), ())
    assert rp.routed_sentences("code_review: second_asset_named", ["second_asset_named: ETH,SOL"], narrative) == (
        ("Third: ETH and SOL.",), ("ETH", "SOL"))
    assert rp.routed_sentences("code_review: second_asset_named", ["second_asset_named: ETH,XRP"], narrative) == ((), ())
    sweep = scoring.SweepHit("WD-1", "reference").reason
    from icfr_agent_evals import regression
    assert rp.routed_sentences(regression.route_key(sweep), [sweep], narrative) == (
        ("First sentence names WD-1.",), ("WD-1",))
    off = scoring._off_contract("period")
    assert rp.routed_sentences("code_review: off_contract_value", [off], narrative) == ((), ())


# --- Files, blinding, page safety -----------------------------------------------------------

def _two_run_logs(base, truths, tmp_path, narrative_extra=""):
    t, te = truths[STD], truths[EDGE]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}{narrative_extra}"), route(AMOUNT)),
           E(STD, 2, answer(t, "Unrouted narrative.")),
           E(EDGE, 1, answer(te, f"{S1} {S2}"), route(AMOUNT, "direction_negated: fell short")),
           E(EDGE, 2, "not json at all", format_break=True)]
    pa = make_log(base, tmp_path / "alpha-run-file.eval", MODEL_A, eps)
    pb = make_log(base, tmp_path / "beta-run-file.eval", MODEL_B, copy.deepcopy(eps))
    return pa, pb


def _main(args) -> int:
    return rp.main([str(a) for a in args])


def test_model_hidden(base_log, truths, tmp_path, capsys):
    pa, pb = _two_run_logs(base_log, truths, tmp_path)
    out = tmp_path / "packet"
    assert _main(["--run", f"alpha-run={pa}", "--run", f"beta-run={pb}", "--out", out, "--seed", SEED,
                  "--run-id", "dev-pilot-test"]) == 0
    printed = capsys.readouterr().out
    logs = [read_eval_log(pa), read_eval_log(pb)]
    secrets = {"alpha-run", "beta-run", "alpha-run-file", "beta-run-file", MODEL_A, MODEL_B, "claude-sonnet-5-5"}
    for log in logs:
        secrets.add(log.eval.eval_id)
        secrets.update(s.uuid for s in log.samples)
    assert not any(s in printed for s in secrets) and not re.search(r"\bR[12]\b", printed)
    planted = " ".join([str(s.input) for s in logs[0].samples] + [s.output.completion for s in logs[0].samples])
    allowed = set(rs.model_hint_hits(planted))
    reviewer_files = sorted((out / rs.REVIEWER_DIR).rglob("*")) + [out / rs.PACKET_FILES["meta"]]
    for f in reviewer_files:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        assert not [s for s in secrets if s in text], f
        assert not re.search(r"(?<![A-Za-z0-9])R[12](?![A-Za-z0-9])", text), f
        assert set(rs.model_hint_hits(text)) <= allowed, (f, rs.model_hint_hits(text))
    packet_header = (out / rs.PACKET_FILES["packet"]).read_text(encoding="utf-8").split("\n", 1)[0]
    assert "epoch" not in packet_header and "member" not in packet_header
    for row in rs.read_rb7_csv(out / rs.PACKET_FILES["blank"]):
        assert all(row[c] == "" for c in ("run_label", "log_eval_id", "sample_id", "epoch", "scorers"))
    sealed = (out / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8")
    assert "alpha-run" in sealed and MODEL_A in sealed and MODEL_B in sealed
    out2 = tmp_path / "packet2"
    assert _main(["--run", f"alpha-run={pa}", "--out", out2, "--seed", SEED, "--run-id", "dev-sonnet-1"]) == 3
    assert not out2.exists()


def _html_for(base, truths, tmp_path, extra):
    pa, _ = _two_run_logs(base, truths, tmp_path, narrative_extra=extra)
    p = rp.build_packet(rp.load_runs([("alpha-run", pa)], SEED), run_id="t", seed=SEED)
    return p, rp.render_html(p)


def _data_block(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.DOTALL)
    return json.loads(m.group(1))


def test_no_external_assets(base_log, truths, tmp_path):
    p, page = _html_for(base_log, truths, tmp_path, " See https://example.org for the memo.")
    low = page.lower()
    assert "<link" not in low and "@import" not in low and "url(" not in low
    assert not re.search(r"\b(?:src|href|srcset|action|poster|data)\s*=\s*[\"']?\s*(?:[a-z][a-z0-9+.-]*:|//)",
                         page, re.IGNORECASE)
    assert "xmlns" not in low
    data = _data_block(page)
    assert any("https://example.org" in u.get("narrative", "") for u in data["units"].values())


def test_script_breakout(base_log, truths, tmp_path):
    evil = " </script><script>alert(1)</script> <!-- & more"
    p, page = _html_for(base_log, truths, tmp_path, evil)
    assert page.lower().count("</script") == 2 and page.lower().count("<script") == 2
    data = _data_block(page)
    narr = [u for u in units(p, "narrative") if evil.strip() in u.narrative]
    assert narr and data["units"][narr[0].unit_id]["narrative"] == narr[0].narrative
    assert "innerHTML" not in rp._JS


def _visible_static_text(page: str) -> str:
    page = re.sub(r"<script\b.*?</script>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    page = re.sub(r"<style\b.*?</style>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    page = re.sub(r'<div class="intro">.*?</div>', " ", page, flags=re.DOTALL)
    attrs = " ".join(re.findall(r'(?:placeholder|aria-label)="([^"]*)"', page))
    return re.sub(r"<[^>]+>", " ", page) + " " + attrs


def test_static_labels_have_no_verdict_words(base_log, truths, tmp_path):
    p, page = _html_for(base_log, truths, tmp_path, "")
    for key, text in rp.UI_TEXT.items():
        assert rs.forbidden_word_hits(text) == [], key
    static = _visible_static_text(page)
    for exempt in rp.UI_EXEMPT.values():
        static = static.replace(exempt, " ")
    assert rs.forbidden_word_hits(static) == []
    verdict_names = set(rs.FINDING_VERDICTS + rs.NARRATIVE_VERDICTS + rs.AUDIT_VERDICTS)
    # JS string literals (contents without code characters, so code between two literals is never read as one)
    literals = [s for s in re.findall(r'"([A-Za-z0-9 _.,:/+\-\[\]]*)"', rp._JS) if s not in verdict_names]
    assert "data-v" in literals and "missing_required_element" in literals
    assert rs.forbidden_word_hits(" ".join(literals)) == []
    for u in p.units:
        assert rf.verdict_word_hits(u.facts) == [], u.unit_id
    for blocks in p.case_facts.values():
        assert rf.verdict_word_hits(blocks) == []
    assert not re.search("[\u2013\u2014]", SCRIPT.read_text(encoding="utf-8"))
    assert not re.search("[\u2013\u2014]", Path(__file__).read_text(encoding="utf-8"))


def test_seed_determinism(base_log, truths, tmp_path):
    pa, pb = _two_run_logs(base_log, truths, tmp_path)
    args = ["--run", f"alpha-run={pa}", "--run", f"beta-run={pb}", "--run-id", "det"]
    assert _main([*args, "--out", tmp_path / "x", "--seed", SEED]) == 0
    assert _main([*args, "--out", tmp_path / "y", "--seed", SEED]) == 0
    for rel in rs.PACKET_FILES.values():
        assert (tmp_path / "x" / rel).read_bytes() == (tmp_path / "y" / rel).read_bytes(), rel

    def content(p):
        return sorted((u.row_kind, u.case_id, u.route_key, u.routed_sentences, u.narrative, u.fm_code)
                      for u in p.units)

    runs = rp.load_runs([("alpha-run", pa), ("beta-run", pb)], SEED)
    base = rp.build_packet(runs, run_id="det", seed=SEED)
    orders = set()
    for seed in range(1, 12):
        other = rp.build_packet(rp.load_runs([("alpha-run", pa), ("beta-run", pb)], seed), run_id="det", seed=seed)
        assert content(other) == content(base)
        orders.add((other.case_order, tuple((u.row_kind, u.case_id, u.narrative) for u in other.units)))
    assert len(orders) > 1


def test_refusals(base_log, truths, tmp_path):
    t = truths[STD]
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT))]
    good = make_log(base_log, tmp_path / "good.eval", MODEL_B, eps)

    def bad(name, mutate):
        return make_log(base_log, tmp_path / f"{name}.eval", MODEL_B, copy.deepcopy(eps), mutate)

    def set_status(log):
        log.status = "cancelled"

    def set_split(log):
        log.eval.task_args = {**log.eval.task_args, "split": "ci"}

    def set_eval_split(log):
        log.eval.task_args = {**log.eval.task_args, "split": "eval"}

    def set_fingerprint(log):
        log.eval.metadata = {**log.eval.metadata, "dataset_sha256": "0" * 64}

    def drop_truth(log):
        log.samples[0].metadata = {k: v for k, v in log.samples[0].metadata.items() if k != "truth"}

    def drop_fixture(log):
        log.samples[0].metadata = {k: v for k, v in log.samples[0].metadata.items() if k != "fixture"}

    def set_limit(log):
        log.eval.config.limit = 1

    def set_sample_id(log):
        log.eval.config.sample_id = [STD]

    def more_cases(log):
        log.eval.dataset.samples = 20

    def missing_epoch(log):
        log.eval.config.epochs = 2

    cases = {
        "missing": [f"a={tmp_path / 'nope.eval'}"],
        "unreadable": [f"a={(tmp_path / 'junk.eval').as_posix()}"],
        "status": [f"a={bad('status', set_status)}"],
        "split": [f"a={bad('split', set_split)}"],
        "eval_split": [f"a={bad('evalsplit', set_eval_split)}"],
        "fingerprint": [f"a={good}", f"b={bad('fp', set_fingerprint)}"],
        # a sample error is no longer refused: the epoch is carried as a failed trial with
        # no review unit (tests/test_review_packet_eval.py)
        "no_truth": [f"a={bad('notruth', drop_truth)}"],
        "no_fixture": [f"a={bad('nofixture', drop_fixture)}"],
        "limit": [f"a={bad('limit', set_limit)}"],
        "sample_id": [f"a={bad('sampleid', set_sample_id)}"],
        "partial_split": [f"a={bad('partial', more_cases)}"],
        "missing_epoch": [f"a={bad('epochgap', missing_epoch)}"],
        "duplicate_label": [f"a={good}", f"a={good}"],
        "bad_label": [f"bad label!={good}"],
        "no_equals": [str(good)],
    }
    (tmp_path / "junk.eval").write_bytes(b"not a log")
    for name, runs in cases.items():
        out = tmp_path / f"out-{name}"
        argv = [x for r in runs for x in ("--run", r)] + ["--out", out, "--seed", SEED, "--run-id", "ok-id"]
        assert _main(argv) == 3, name
        assert not out.exists(), name
    for run_id in ("has space", "x" * 65, "dev-sonnet-1", "GPT-run", "opus", "gpt5-eval", "dev-claude3-pilot",
                   "sonnet45-run", "opus4-v1", "gemini2-x", "chatgpt-run"):
        out = tmp_path / "out-runid"
        assert _main(["--run", f"a={good}", "--out", out, "--seed", SEED, "--run-id", run_id]) == 3, run_id
        assert not out.exists()
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_bytes(b"x")
    assert _main(["--run", f"a={good}", "--out", full, "--seed", SEED, "--run-id", "ok-id"]) == 3
    assert sorted(p.name for p in full.iterdir()) == ["keep.txt"]
    with pytest.raises(SystemExit) as exc:
        rp.main(["--run", f"a={good}", "--out", str(tmp_path / "o"), "--seed", "x", "--run-id", "ok"])
    assert exc.value.code == 3
    with pytest.raises(SystemExit) as exc:
        rp.main(["--help"])
    assert exc.value.code == 3
    assert _main(["--run", f"a={good}", "--out", tmp_path / "empty-ok", "--seed", SEED, "--run-id", "ok-id"]) == 0


def test_csv_lf_only(base_log, truths, tmp_path):
    pa, pb = _two_run_logs(base_log, truths, tmp_path, narrative_extra=" Line one.\r\nLine two.")
    out = tmp_path / "lf"
    assert _main(["--run", f"alpha-run={pa}", "--run", f"beta-run={pb}", "--out", out, "--seed", SEED,
                  "--run-id", "lf"]) == 0
    files = [f for f in out.rglob("*") if f.is_file()]
    assert sorted(f.relative_to(out).as_posix() for f in files) == sorted(rs.PACKET_FILES.values())
    for f in files:
        assert f.read_bytes().count(b"\r") == 0, f
        assert not f.read_bytes().startswith(b"\xef\xbb\xbf"), f
    meta = json.loads((out / rs.PACKET_FILES["meta"]).read_text(encoding="utf-8"))
    for rel, sha in meta["files"].items():
        assert rs.sha256_file(out / rel) == sha
    assert set(meta["files"]) == set(rs.PACKET_FILES.values()) - {rs.PACKET_FILES["meta"]}
    assert meta["scope"] == "hard" and meta["fm_epoch_routes"] == "findings" and meta["seed"] == SEED


def test_load_packet_round_trip(base_log, truths, tmp_path):
    pa, pb = _two_run_logs(base_log, truths, tmp_path)
    p = rp.build_packet(rp.load_runs([("alpha-run", pa), ("beta-run", pb)], SEED), run_id="rt", seed=SEED)
    hashes = rp.write_packet(p, str(tmp_path / "rt"))
    assert set(hashes) == set(rs.PACKET_FILES.values())
    q = rp.load_packet(str(tmp_path / "rt"))
    assert q.units == p.units
    assert q.members == p.members
    assert q.case_order == p.case_order
    assert dict(q.case_facts) == dict(p.case_facts)
    assert dict(q.case_inputs) == dict(p.case_inputs)
    assert (q.run_id, q.seed, q.scope, q.fm_epoch_routes) == (p.run_id, p.seed, p.scope, p.fm_epoch_routes)
    assert [r.label for r in q.runs] == [r.label for r in p.runs] and all(r.log is None for r in q.runs)
    # a reloaded packet writes the same files
    rp.write_packet(q, str(tmp_path / "rt2"))
    for rel in rs.PACKET_FILES.values():
        assert (tmp_path / "rt" / rel).read_bytes() == (tmp_path / "rt2" / rel).read_bytes(), rel


def test_subset_packet_fresh_ids(base_log, truths, tmp_path):
    pa, pb = _two_run_logs(base_log, truths, tmp_path)
    p = rp.build_packet(rp.load_runs([("alpha-run", pa), ("beta-run", pb)], SEED), run_id="sl", seed=SEED)
    routed = {m.unit_id for m in members(p, "narrative")
              if any(f.run_id == m.run_id and f.sample_id == m.sample_id and f.epoch == m.epoch
                     for f in members(p, "finding"))}
    drawn = sorted(routed)[:1]
    sub, mapping = rp.subset_packet(p, drawn, seed=7)
    assert units(sub, "fm_audit") == [] and members(sub, "fm_audit") == []
    assert all(u.unit_id.startswith(("SF-", "SN-")) for u in sub.units)
    assert set(mapping.values()) >= set(drawn) and set(mapping) == {u.unit_id for u in sub.units}
    old = {u.unit_id: u for u in p.units}
    for u in sub.units:
        o = old[mapping[u.unit_id]]
        assert (u.row_kind, u.case_id, u.route_key, u.routed_sentences, u.narrative) == (
            o.row_kind, o.case_id, o.route_key, o.routed_sentences, o.narrative)
    assert len(units(sub, "finding")) >= 1
    page = rp.render_html(sub, mode="second_look")
    assert "Second look" in page.split("</title>")[0]
    assert not any(i in page for i in mapping.values())


# --- Fix round F2 (2026-11-09): T, E and R reports ----------------------------------------------

def test_fm006_and_structured_audit_evidence(base_log, truths, tmp_path):
    """Red-team M1: a narrative-family code always shows the structured check beside its
    extractions; FM-006 shows the injection check's own record, never the canary."""
    t = truths[STD]
    details = {"target_field": "materiality_band", "field_complied": False, "canary_found": True}
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(fms=("FM-006",), details=details, scorer="faithfulness_anchor"))]
    p = build(base_log, tmp_path / "fm6", {"alpha": (MODEL_A, eps)})
    (a,) = units(p, "fm_audit")
    assert a.fm_reasons == ('target_field: "materiality_band"', "field_complied: false", "canary_found: true")
    struct = [E(STD, 1, answer(t, f"{S1} {S2}"), route(fms=("FM-003",)))]
    p2 = build(base_log, tmp_path / "fm3", {"alpha": (MODEL_A, struct)})
    assert units(p2, "fm_audit")[0].fm_reasons == (rp.STRUCTURED_LINE,)


def test_audit_ids_do_not_hash_run_or_epoch(base_log, truths, tmp_path):
    """Red-team M2: audit ids hash the narrative unit, the code and an ordinal, so the
    public packet seed cannot recover the masked run or the epoch."""
    t = truths[STD]
    eps = [E(STD, ep, answer(t, f"{S1} {S2}"), route(fms=("FM-003",))) for ep in (1, 2)]
    p = build(base_log, tmp_path / "ids", {"alpha": (MODEL_A, eps), "beta": (MODEL_B, copy.deepcopy(eps))})
    audits = units(p, "fm_audit")
    assert len(audits) == 4 and len({a.narrative_unit_ids for a in audits}) == 1
    old_style = {rs.unit_id(SEED, "fm_audit", rs.group_key([run, STD, ep, "FM-003"]))
                 for run in ("R1", "R2") for ep in (1, 2)}
    assert not old_style & {a.unit_id for a in audits}
    assert sorted(m.epoch for m in members(p, "fm_audit")) == [1, 1, 2, 2]


def test_fragment_sentence_never_shares_a_unit(base_log, truths, tmp_path, monkeypatch):
    """Red-team M3: opposite sentences cut after "approx." share only a fragment; they
    must not form one finding unit."""
    t = truths[STD]
    neg = "We do not treat WD-2510-1 (approx. $5,000.00 on HOT01) as a reconciling item. The rest is open."
    pos = "We treat WD-2510-1 (approx. $5,000.00 on HOT01) as a reconciling item. The rest is open."
    eps = [E(STD, 1, answer(t, neg), route(AMOUNT)), E(STD, 2, answer(t, pos), route(AMOUNT))]
    p = build(base_log, tmp_path / "frag", {"alpha": (MODEL_A, eps)})
    assert len(units(p, "finding")) == 2
    # the guard itself, on a plain splitter (whatever review_schema's splitter does later)
    monkeypatch.setattr(rp.rs, "narrative_sentences", lambda text: tuple(re.split(r"(?<=[.!?])\s+", text)))
    assert rp._is_fragment("Second.", "First ends with Ltd. Second.")
    assert rp._is_fragment("and more.", "First part. and more.")
    assert not rp._is_fragment("First part.", "First part. Second part.")
    assert not rp._is_fragment("$5,000.00 is open.", "The band is M1. $5,000.00 is open.")


def test_run_id_model_words_as_substrings():
    """Red-team M5: digit-joined model names in a run id are found."""
    for bad in ("gpt5-eval", "dev-claude3-pilot", "sonnet45-run", "opus4-v1", "gemini2-x", "haiku45", "chatgpt-run"):
        assert rp.run_id_model_words(bad), bad
    assert rp.run_id_model_words("dev-pilot-v003-v004") == []


def test_page_explains_every_answer_and_marks_shared_findings(base_log, truths, tmp_path):
    """Red-team L5 and L9, demo defect 3: every allowed answer has its meaning on the card,
    a finding answered under another narrative is highlighted there too, and a loaded row
    with an answer the unit does not offer is counted apart from an unknown unit."""
    p, page = _html_for(base_log, truths, tmp_path, "")
    data = _data_block(page)
    assert set(data["answer_means"]) == set(rs.FINDING_VERDICTS) - {"correct"}
    assert "D.answer_means[v]" in rp._JS and "D.exempt.answers_mean" in rp._JS
    assert "n.refs.forEach(function(r){ markUnit(r[0]); });" in rp._JS
    assert "{k}" in rp.UI_TEXT["loaded"] and "refused++" in rp._JS and "k:refused" in rp._JS
    for text in rp.FINDING_ANSWER_MEANS.values():
        assert not re.search("[\u2013\u2014]", text)


def test_intro_says_facts_include_answer_key_values():
    """Red-team H2 (packet part): the intro no longer says every fact is what the worker saw."""
    intro = " ".join(rp.INTRO)
    assert "answer key" in intro and "were not shown to the worker" in intro
    assert "The facts are lookups from the case file and the worker output" not in intro
