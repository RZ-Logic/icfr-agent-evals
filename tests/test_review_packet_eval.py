"""scripts/review_packet.py: the evaluation-split path and carried epochs.

Pre-registration section 13 (tooling built after the tag, item 1), section 4 (the
allow_evaluation_split unlock in every log header) and section 6 (errors and
unscored epochs count as failed trials, and review never settles them).

Hand-built logs only: one small mockllm run of three dev pilot cases gives a real
log header and real samples; each test writes its own logs into tmp_path. A log
that claims the evaluation split is a mock: its header is edited to say so and its
sample ids are made up (eval-mock-001 and so on). No evaluation case is built, read
or scored, no API key is read and no model is called.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalError, read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score

from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import trials
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.reliability import TRIAL_ANCHORS
from icfr_agent_evals.task import variance_commentary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_packet.py"
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
STD, EDGE, THIRD = "dev-std-002-BTC-M1", "dev-hc-005-USDT-M3", "dev-std-004-SOL-M1"
MODEL_A, MODEL_B = "mockllm/model", "mockllm/other"
ANCHORS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor",
           "faithfulness_anchor", "completeness_anchor")
SEED = 20261114
AMOUNT = "amount_role_unresolved: $5,000.00"
S1 = "Reserves fell short of ledger liabilities by $5,000.00 at the cutoff."
S2 = "The custodian confirmed the wallet balances."
MARK = "Carried epoch marker sentence that review must never see."
ABSENT = object()


def _load_script():
    spec = importlib.util.spec_from_file_location("review_packet_eval_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_script()
REAL_COMPOSITION_CHECK = rp.check_eval_composition
REAL_TEMPLATE_CHECK = rp.check_eval_template


@pytest.fixture(scope="module", autouse=True)
def _small_eval_logs_allowed():
    """The mock evaluation-split logs here hold two or three cases, not the 200 of section
    4, so the composition check is set aside for them; the composition tests below call the
    real check (REAL_COMPOSITION_CHECK). Most hold one worker run and no template run, so the
    template check (decision D-R20) is set aside too; the template tests put it back
    (REAL_TEMPLATE_CHECK)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(rp, "check_eval_composition", lambda log, name: None)
    mp.setattr(rp, "check_eval_template", lambda runs, split: None)
    yield
    mp.undo()


# --- Hand-built logs ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_log(tmp_path_factory):
    log = inspect_eval(
        variance_commentary(split="dev", subset="pilot", epochs=1),
        model=get_model("mockllm/model"), model_roles=ROLES,
        log_dir=str(tmp_path_factory.mktemp("base")), display="none", sample_id=[STD, EDGE, THIRD],
    )[0]
    assert log.status == "success", log.error
    return read_eval_log(log.location)


@pytest.fixture(scope="module")
def truths(base_log):
    return {str(s.id): truth_of(s.metadata) for s in base_log.samples}


def answer(truth, narrative: str) -> str:
    items = truth.eligible_items()
    return json.dumps({
        "asset": truth.asset, "period": truth.period, "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction, "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in items)],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
        "narrative": narrative,
    })


@dataclass
class E:
    """One planted (case, epoch): the completion, per-scorer (reasons, fms, details), and
    how the epoch fails to be scored (a sample error, dropped anchors, odd values)."""
    case: str
    epoch: int
    completion: str
    scores: dict = field(default_factory=dict)
    error: str | None = None
    drop: tuple[str, ...] = ()
    values: dict = field(default_factory=dict)
    no_scores: bool = False


def _scores(e: E) -> dict | None:
    if e.no_scores:
        return None
    out = {}
    for name in ANCHORS:
        if name in e.drop:
            continue
        reasons, fms, details = e.scores.get(name, ((), (), {}))
        out[name] = Score(value=e.values.get(name, 0 if fms else 3), metadata={
            "failure_modes": list(fms), "requires_human_review": bool(reasons),
            "review_reasons": list(reasons), "details": dict(details)})
    return out


def route(*reasons, fms=(), scorer="correctness_anchor"):
    return {scorer: (tuple(reasons), tuple(fms), {})}


def to_eval(log, *, flag=True, meta_split="eval"):
    """Make a mock log claim the evaluation split, with made-up sample ids."""
    args = {**(log.eval.task_args or {}), "split": "eval"}
    if flag is ABSENT:
        args.pop("allow_evaluation_split", None)
    else:
        args["allow_evaluation_split"] = flag
    log.eval.task_args = args
    log.eval.metadata = {**(log.eval.metadata or {}), "split": meta_split}
    names = {cid: f"eval-mock-{i:03d}" for i, cid in enumerate(sorted({str(s.id) for s in log.samples}), start=1)}
    for s in log.samples:
        s.id = names[str(s.id)]
    log.eval.dataset.sample_ids = sorted(names.values())


def make_log(base, path: Path, model: str, epochs: list[E], *, eval_split=False, mutate=None, **eval_kw) -> str:
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
        s.error = None if e.error is None else EvalError(message=e.error, traceback="", traceback_ansi="")
        samples.append(s)
    log.samples = samples
    k = max(e.epoch for e in epochs)
    log.eval.config.epochs = k
    log.eval.task_args = {**(log.eval.task_args or {}), "epochs": k}
    cases = sorted({e.case for e in epochs})
    log.eval.config.sample_id = None
    log.eval.config.limit = None
    log.eval.dataset.samples = len(cases)
    log.eval.dataset.sample_ids = cases
    if eval_split:
        to_eval(log, **eval_kw)
    if mutate:
        mutate(log)
    write_eval_log(log, str(path))
    return str(path)


def _main(args) -> int:
    return rp.main([str(a) for a in args])


def _files(out: Path) -> dict[str, bytes]:
    return {rel: (out / rel).read_bytes() for rel in rs.PACKET_FILES.values()}


def _reviewer_texts(out: Path) -> dict[str, str]:
    return {p.relative_to(out).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted((out / rs.REVIEWER_DIR).rglob("*")) if p.is_file()}


def _good_epochs(truths):
    t, te = truths[STD], truths[EDGE]
    return [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT)),
            E(STD, 2, answer(t, "Unrouted narrative.")),
            E(EDGE, 1, answer(te, f"{S1} {S2}"), route(AMOUNT)),
            E(EDGE, 2, answer(te, "A hard case narrative read whole."))]


# --- log_split: the unlock, read from the header ---------------------------------------------

def _header(split, *, flag=ABSENT, meta_split=ABSENT, sample_ids=("dev-std-001",)):
    args = {"split": split} if split is not ABSENT else {}
    if flag is not ABSENT:
        args["allow_evaluation_split"] = flag
    meta = {"split": meta_split} if meta_split is not ABSENT else {}
    return SimpleNamespace(eval=SimpleNamespace(task_args=args, metadata=meta, dataset=SimpleNamespace(name="ds")),
                           samples=[SimpleNamespace(id=i) for i in sample_ids])


def test_log_split_dev_and_unlocked_eval():
    assert rp.log_split(_header("dev", meta_split="dev"), "a.eval") == "dev"
    assert rp.log_split(_header("eval", flag=True, meta_split="eval", sample_ids=("eval-mock-001",)), "a.eval") == "eval"


@pytest.mark.parametrize("flag", [ABSENT, False, None, "true", "True", 1, 1.0, [True], {"x": True}])
def test_log_split_refuses_eval_without_literal_true(flag):
    with pytest.raises(rp.Refused, match="allow_evaluation_split"):
        rp.log_split(_header("eval", flag=flag, meta_split="eval"), "a.eval")


@pytest.mark.parametrize("meta_split", [ABSENT, "dev", None, "EVAL"])
def test_log_split_refuses_eval_header_that_disagrees(meta_split):
    with pytest.raises(rp.Refused, match="disagrees"):
        rp.log_split(_header("eval", flag=True, meta_split=meta_split), "a.eval")


@pytest.mark.parametrize("split", [ABSENT, None, "ci", "Eval", "EVAL", "dev ", ""])
def test_log_split_refuses_other_splits(split):
    with pytest.raises(rp.Refused, match="split"):
        rp.log_split(_header(split, flag=True, meta_split="eval"), "a.eval")


def test_log_split_dev_still_meets_regression_refusal():
    # a dev log carrying any sign of the evaluation split is refused as before
    with pytest.raises(rp.Refused, match="evaluation"):
        rp.log_split(_header("dev", meta_split="eval"), "a.eval")
    with pytest.raises(rp.Refused, match="evaluation-split sample"):
        rp.log_split(_header("dev", meta_split="dev", sample_ids=("eval-mock-001",)), "a.eval")
    # the unlock never turns a dev log into an evaluation log
    assert rp.log_split(_header("dev", flag=True, meta_split="dev"), "a.eval") == "dev"


def test_eval_path_never_calls_the_regression_refusal(monkeypatch):
    def boom(log):
        raise AssertionError("regression.refuse_evaluation_split called on the eval path")

    monkeypatch.setattr(rp.regression, "refuse_evaluation_split", boom)
    assert rp.log_split(_header("eval", flag=True, meta_split="eval"), "a.eval") == "eval"
    with pytest.raises(AssertionError):
        rp.log_split(_header("dev", meta_split="dev"), "a.eval")


# --- carried_reasons: what counts as errored or unscored --------------------------------------

def _sample(values=None, *, error=None, drop=(), scores=True):
    values = values or {}
    if not scores:
        return SimpleNamespace(error=error, scores=None)
    sc = {a: SimpleNamespace(value=values.get(a, 3)) for a in ANCHORS if a not in drop}
    return SimpleNamespace(error=error, scores=sc)


def test_carried_reasons_clean_epoch_is_reviewed():
    assert rp.carried_reasons(_sample()) == ()
    assert rp.carried_reasons(_sample({"correctness_anchor": 0, "completeness_anchor": 2.0})) == ()
    # faithfulness_anchor is not a trial anchor: dropping it carries nothing
    assert rp.carried_reasons(_sample(drop=("faithfulness_anchor",))) == ()


def test_carried_reasons_error_and_unscored():
    assert rp.carried_reasons(_sample(error="boom")) == ("sample error",)
    assert rp.carried_reasons(_sample(error="boom", scores=False)) == (
        "sample error", *(f"{a} unscored or non-numeric" for a in TRIAL_ANCHORS))
    assert rp.carried_reasons(_sample(drop=("completeness_anchor",))) == (
        "completeness_anchor unscored or non-numeric",)
    for bad in (math.nan, math.inf, -math.inf, None, True, False, "3", [3]):
        assert rp.carried_reasons(_sample({"fabrication_anchor": bad})) == (
            "fabrication_anchor unscored or non-numeric",), bad


def test_carried_reasons_match_trials_numeric_test():
    for value in (3, 0, 2.5, math.nan, math.inf, True, "3", None):
        carried = bool(rp.carried_reasons(_sample({"correctness_anchor": value})))
        assert carried == (trials._numeric(value) is None), value


# --- The evaluation-split path on mock logs ---------------------------------------------------

def test_eval_log_with_unlock_accepted(base_log, truths, tmp_path, capsys):
    path = make_log(base_log, tmp_path / "ev.eval", MODEL_A, _good_epochs(truths), eval_split=True)
    runs = rp.load_runs([("alpha", path)], SEED)
    p = rp.build_packet(runs, run_id="eval-mock-run", seed=SEED)
    assert rp.packet_split(p) == "eval" and p.carried == ()
    assert {u.case_id for u in p.units} == {"eval-mock-001", "eval-mock-002"}
    assert [u for u in p.units if u.row_kind == "finding"]
    out = tmp_path / "packet"
    assert _main(["--run", f"alpha={path}", "--out", out, "--seed", SEED, "--run-id", "eval-mock-run"]) == 0
    assert "carried failed trials (errored or unscored epochs, no finding unit): 0" in capsys.readouterr().out
    runs_json = json.loads((out / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    assert runs_json["R1"]["split"] == "eval" and runs_json["R1"]["allow_evaluation_split"] is True
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"]["carried_failed_trials"] == [] and summary["R1"]["carried_failed_trials"] == 0


@pytest.mark.parametrize("flag", [ABSENT, False, "true", 1])
def test_eval_log_without_unlock_refused(base_log, truths, tmp_path, flag):
    path = make_log(base_log, tmp_path / "ev.eval", MODEL_A, _good_epochs(truths), eval_split=True, flag=flag)
    with pytest.raises(rp.Refused, match="allow_evaluation_split"):
        rp.load_runs([("alpha", path)], SEED)
    out = tmp_path / "out"
    assert _main(["--run", f"alpha={path}", "--out", out, "--seed", SEED, "--run-id", "eval-mock-run"]) == 3
    assert not out.exists()


def test_eval_log_whose_metadata_says_dev_refused(base_log, truths, tmp_path):
    path = make_log(base_log, tmp_path / "ev.eval", MODEL_A, _good_epochs(truths), eval_split=True, meta_split="dev")
    with pytest.raises(rp.Refused, match="disagrees"):
        rp.load_runs([("alpha", path)], SEED)


def test_mixed_splits_refused(base_log, truths, tmp_path):
    dev = make_log(base_log, tmp_path / "dev.eval", MODEL_A, _good_epochs(truths))
    ev = make_log(base_log, tmp_path / "ev.eval", MODEL_B, _good_epochs(truths), eval_split=True)
    for specs in ([("alpha", dev), ("beta", ev)], [("alpha", ev), ("beta", dev)]):
        with pytest.raises(rp.Refused, match="different splits"):
            rp.load_runs(specs, SEED)
    out = tmp_path / "out"
    assert _main(["--run", f"alpha={dev}", "--run", f"beta={ev}", "--out", out, "--seed", SEED,
                  "--run-id", "mixed"]) == 3
    assert not out.exists()
    # build_packet refuses a mix too, when handed runs load_runs read one at a time
    mixed = (*rp.load_runs([("alpha", dev)], SEED), *rp.load_runs([("beta", ev)], SEED))
    mixed = tuple(rp.RunInput(**{**r.__dict__, "run_id": f"R{i}"}) for i, r in enumerate(mixed, start=1))
    with pytest.raises(rp.Refused, match="different splits"):
        rp.build_packet(mixed, run_id="mixed", seed=SEED)


def test_eval_log_partial_or_missing_epoch_still_refused(base_log, truths, tmp_path):
    def set_limit(log):
        log.eval.config.limit = 1

    def more_cases(log):
        log.eval.dataset.samples = 20

    def gap(log):
        log.eval.config.epochs = 3

    for name, mutate in (("limit", set_limit), ("partial", more_cases), ("gap", gap)):
        path = make_log(base_log, tmp_path / f"{name}.eval", MODEL_A, _good_epochs(truths), eval_split=True,
                        mutate=mutate)
        with pytest.raises(rp.Refused):
            rp.load_runs([("alpha", path)], SEED)


def test_two_eval_runs_accepted(base_log, truths, tmp_path):
    a = make_log(base_log, tmp_path / "a.eval", MODEL_A, _good_epochs(truths), eval_split=True)
    b = make_log(base_log, tmp_path / "b.eval", MODEL_B, copy.deepcopy(_good_epochs(truths)), eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", a), ("beta", b)], SEED), run_id="eval-two", seed=SEED)
    assert {m.run_id for m in p.members} == {"R1", "R2"}


# --- Carried epochs ---------------------------------------------------------------------------

def _with_errored(truths, fms=("FM-001",), **errored):
    """The good epochs plus epoch 4 of each case: errored, holding a planted route and the
    marker sentence, so any finding unit built from it would show. With fms the standard
    case's errored epoch also holds those FM codes, which the code audit must still cover."""
    t, te = truths[STD], truths[EDGE]
    return _good_epochs(truths) + [
        E(STD, 3, answer(t, f"{S1} {S2}"), route(AMOUNT)),
        E(EDGE, 3, answer(te, "A hard case narrative read whole.")),
        E(STD, 4, answer(t, f"{MARK} {S1}"), route(AMOUNT, "direction_negated: fell short", fms=fms),
          **errored),
        E(EDGE, 4, answer(te, MARK), **errored),
    ]


@pytest.mark.parametrize("eval_split", [True, False])
def test_errored_epoch_is_a_format_break_with_its_audit_and_narrative_read(base_log, truths, tmp_path, eval_split):
    # Decision D-R20 (section 9: "A missing or refused generation stays a format break, and
    # review never settles it"): an errored epoch carries FM-000. It stays carried as a failed
    # trial and its routes build no finding, but it gets its FM-000 audit unit and its
    # whole-narrative read like any FM-000 epoch. Its output is shown as its real parse gives
    # it (here it parses: the narrative and the fields), under its own page tag.
    path = make_log(base_log, tmp_path / "err.eval", MODEL_A,
                    _with_errored(truths, fms=(), error="boom: provider error"), eval_split=eval_split)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    log = read_eval_log(path)
    sids = sorted({str(s.id) for s in log.samples})
    assert [(c.run_id, c.sample_id, c.epoch, c.reason) for c in p.carried] == [
        ("R1", sid, 4, "sample error") for sid in sids]
    on4 = sorted((m.sample_id, m.member_kind, m.fm_codes, m.route_class) for m in p.members if m.epoch == 4)
    assert on4 == sorted([(sid, "fm_audit", ("FM-000",), "") for sid in sids]
                         + [(sid, "narrative", ("FM-000",), rs.NARRATIVE_CLASS_FM_CODED) for sid in sids])
    assert not [u for u in p.units if u.row_kind == "finding" and "direction_negated" in " ".join(u.reasons)]
    audits = [u for u in p.units if u.row_kind == "fm_audit"]
    assert [u.fm_code for u in audits] == ["FM-000", "FM-000"]
    assert {u.fm_reasons for u in audits} == {(rp.ERRORED_EVIDENCE,)}
    parsed = {str(s.id): json.loads(s.output.completion) for s in log.samples if s.epoch == 4}
    for a in audits:
        narrative = next(u for u in p.units if u.unit_id == a.narrative_unit_ids[0])
        assert narrative.parsed_ok is True and narrative.errored is True
        assert narrative.structured["asset"] == parsed[narrative.case_id]["asset"]
        assert narrative.narrative == parsed[narrative.case_id]["narrative"] and MARK in narrative.narrative
        assert narrative.facts[0].block_id == "worker_fields" and len(narrative.facts) > 1
    # every other narrative unit is untagged
    assert sum(u.errored for u in p.units if u.row_kind == "narrative") == 2
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    page = (out / rs.PACKET_FILES["html"]).read_text(encoding="utf-8")
    assert page.count(rp.ERRORED_TAG) == 2
    assert 'if (n.errored_tag) card.appendChild(el("div", "tag", n.errored_tag));' in page
    # the tag survives a reload of the packet (derived from the carried epochs and the members)
    loaded = rp.load_packet(str(out))
    assert {u.unit_id for u in loaded.units if u.errored} == {u.unit_id for u in p.units if u.errored}
    assert rp.render_html(loaded) == page
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"]["carried_failed_trials"] == [
        {"run_id": "R1", "sample_id": sid, "epoch": 4, "reason": "sample error"} for sid in sids]
    assert summary["R1"]["carried_failed_trials"] == 2 and summary["R1"]["fm_audit_rows"] == 2
    blank = rs.read_rb7_csv(out / rs.PACKET_FILES["blank"])
    assert [r["route_key"] for r in blank if r["row_kind"] == "fm_audit"] == ["FM-000", "FM-000"]


@pytest.mark.parametrize("eval_split", [True, False])
def test_fm_coded_carried_epoch_gets_its_code_audit_and_no_finding(base_log, truths, tmp_path, eval_split):
    # Section 8: the settlement refuses a record unless every FM-coded (epoch, code) has an
    # audit row; section 6: the audit reads that epoch's narrative whole. The epoch is still
    # carried as a failed trial and its routes build no finding. An errored epoch also
    # carries FM-000 (decision D-R20), beside the codes its scorers recorded.
    path = make_log(base_log, tmp_path / "err.eval", MODEL_A, _with_errored(truths, error="boom: provider error"),
                    eval_split=eval_split)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    edge, std = sorted({str(s.id) for s in read_eval_log(path).samples})
    assert [(c.sample_id, c.epoch, c.reason) for c in p.carried] == [
        (edge, 4, "sample error"), (std, 4, "sample error")]
    on4 = sorted((m.sample_id, m.member_kind, m.fm_codes) for m in p.members if m.epoch == 4)
    assert on4 == [(edge, "fm_audit", ("FM-000",)), (edge, "narrative", ("FM-000",)),
                   (std, "fm_audit", ("FM-000",)), (std, "fm_audit", ("FM-001",)),
                   (std, "narrative", ("FM-000", "FM-001"))]
    audits = [u for u in p.units if u.row_kind == "fm_audit"]
    assert sorted(u.fm_code for u in audits) == ["FM-000", "FM-000", "FM-001"]
    fm001 = next(u for u in audits if u.fm_code == "FM-001")
    narrative = next(u for u in p.units if u.unit_id == fm001.narrative_unit_ids[0])
    assert MARK in narrative.narrative and narrative.route_class == rs.NARRATIVE_CLASS_FM_CODED
    assert not [u for u in p.units if u.row_kind == "finding" and "direction_negated" in " ".join(u.reasons)]
    assert not [m for m in p.members if m.epoch == 4 and m.member_kind in rp.ROUTE_MEMBER_KINDS]
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["R1"]["carried_failed_trials"] == 2 and summary["R1"]["fm_audit_rows"] == 3
    blank = rs.read_rb7_csv(out / rs.PACKET_FILES["blank"])
    assert sorted(r["route_key"] for r in blank if r["row_kind"] == "fm_audit") == ["FM-000", "FM-000", "FM-001"]


def test_errored_epoch_shows_its_completion_but_never_its_error(base_log, truths, tmp_path):
    # Decision D-R20: the reviewer reads the errored epoch's raw completion whole, as for any
    # format break; the error text (which can name the model), its routes and the masked run
    # never reach reviewer/.
    err = make_log(base_log, tmp_path / "err.eval", MODEL_A,
                   _with_errored(truths, fms=(), error="boom: provider error"), eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", err)], SEED), run_id="carry", seed=SEED)
    rp.write_packet(p, str(tmp_path / "err"))
    texts = _reviewer_texts(tmp_path / "err")
    assert texts and any(MARK in t for t in texts.values())
    assert any(rp.ERRORED_EVIDENCE in t for t in texts.values())
    for rel, text in texts.items():
        for trace in ("boom", "provider error", "direction_negated", MODEL_A):
            assert trace not in text, (rel, trace)
        assert not re.search(r"(?<![A-Za-z0-9])R1(?![A-Za-z0-9])", text), rel


def test_unscored_epoch_carried(base_log, truths, tmp_path):
    t, te = truths[STD], truths[EDGE]
    eps = _good_epochs(truths) + [
        E(STD, 3, answer(t, MARK), route(AMOUNT), drop=("completeness_anchor",)),
        E(EDGE, 3, answer(te, MARK), values={"fabrication_anchor": math.nan}),
    ]
    path = make_log(base_log, tmp_path / "uns.eval", MODEL_A, eps, eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED, scope="all")
    by_sid = {c.sample_id: c.reason for c in p.carried}
    names = {str(s.id): truth_of(s.metadata) for s in read_eval_log(path).samples}
    std_id = next(sid for sid, tr in names.items() if tr.case_id == STD)
    edge_id = next(sid for sid, tr in names.items() if tr.case_id == EDGE)
    assert by_sid == {std_id: "completeness_anchor unscored or non-numeric",
                      edge_id: "fabrication_anchor unscored or non-numeric"}
    assert not [m for m in p.members if m.epoch == 3]
    assert not [u for u in p.units if MARK in u.narrative]


def test_errored_epoch_with_no_scores_carried(base_log, truths, tmp_path):
    eps = _good_epochs(truths) + [E(STD, 3, "", error="timeout", no_scores=True),
                                  E(EDGE, 3, answer(truths[EDGE], "A hard case narrative read whole."))]
    path = make_log(base_log, tmp_path / "ns.eval", MODEL_A, eps, eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    assert len(p.carried) == 1
    assert p.carried[0].reason == "sample error; " + "; ".join(f"{a} unscored or non-numeric" for a in TRIAL_ANCHORS)
    # decision D-R20: nothing came back, so its FM-000 audit reads an empty narrative
    on3 = [m for m in p.members if m.epoch == 3 and m.sample_id == p.carried[0].sample_id]
    assert sorted((m.member_kind, m.fm_codes) for m in on3) == [("fm_audit", ("FM-000",)), ("narrative", ("FM-000",))]
    narrative = next(u for u in p.units if u.unit_id == next(m.unit_id for m in on3 if m.member_kind == "narrative"))
    assert narrative.narrative == "" and narrative.parsed_ok is False and narrative.errored is True


def test_carried_reasons_agree_with_trials(base_log, truths, tmp_path):
    t, te = truths[STD], truths[EDGE]
    eps = _good_epochs(truths) + [
        E(STD, 3, answer(t, MARK), error="boom"),
        E(EDGE, 3, answer(te, MARK), drop=("reconciling_items_anchor",)),
    ]
    path = make_log(base_log, tmp_path / "agree.eval", MODEL_A, eps, eval_split=True)
    log = read_eval_log(path)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    blind = trials.route_blind_trial(log)
    for c in p.carried:
        result = next(e for e in blind.samples[c.sample_id].epochs if e.epoch == c.epoch)
        assert not result.success
        assert set(c.reason.split("; ")) <= set(result.reasons), (c, result.reasons)
    # every epoch trials fails on an error or an unscored anchor is carried, and no other
    carried = {(c.sample_id, c.epoch) for c in p.carried}
    for e in blind.epochs:
        flagged = any(r == "sample error" or r.endswith("unscored or non-numeric") for r in e.reasons)
        assert flagged == ((e.sample_id, e.epoch) in carried), e


def test_case_with_every_epoch_unscored_is_not_shown(base_log, truths, tmp_path):
    # an unscored epoch with no sample error is not a format break (decision D-R20): it builds
    # no unit, so a case with nothing else is not shown
    t = truths[THIRD]
    eps = _good_epochs(truths) + [E(THIRD, 1, answer(t, MARK), drop=("completeness_anchor",)),
                                  E(THIRD, 2, answer(t, MARK), values={"correctness_anchor": math.nan})]
    path = make_log(base_log, tmp_path / "all.eval", MODEL_A, eps, eval_split=True)
    log = read_eval_log(path)
    third = next(str(s.id) for s in log.samples if truth_of(s.metadata).case_id == THIRD)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED, scope="all")
    assert third not in p.case_order and third not in p.case_inputs and third not in p.case_facts
    assert {(c.sample_id, c.epoch) for c in p.carried} == {(third, 1), (third, 2)}
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    for rel, text in _reviewer_texts(out).items():
        assert third not in text and MARK not in text, rel


def test_case_with_every_epoch_errored_shows_its_format_breaks_only(base_log, truths, tmp_path):
    # decision D-R20: every errored epoch is a format break with its FM-000 audit and its read
    t = truths[THIRD]
    eps = _good_epochs(truths) + [E(THIRD, 1, answer(t, MARK), error="boom"), E(THIRD, 2, answer(t, MARK), error="boom")]
    path = make_log(base_log, tmp_path / "all.eval", MODEL_A, eps, eval_split=True)
    log = read_eval_log(path)
    third = next(str(s.id) for s in log.samples if truth_of(s.metadata).case_id == THIRD)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED, scope="all")
    assert third in p.case_order
    assert {(c.sample_id, c.epoch) for c in p.carried} == {(third, 1), (third, 2)}
    kinds = sorted((m.member_kind, m.fm_codes) for m in p.members if m.sample_id == third)
    assert kinds == [("fm_audit", ("FM-000",))] * 2 + [("narrative", ("FM-000",))] * 2
    units = [u for u in p.units if u.case_id == third]
    assert sorted(u.row_kind for u in units) == ["fm_audit", "fm_audit", "narrative"]  # one text, read once


def test_format_break_is_not_carried(base_log, truths, tmp_path):
    # FM-000 keeps its audit unit and its narrative read (section 6 and section 8)
    reason = "format break: not one JSON object: Expecting value: line 1 column 1 (char 0)"
    fb = {name: ((reason,), ("FM-000",), {}) for name in ANCHORS}
    eps = _good_epochs(truths) + [E(STD, 3, "not json at all", fb), E(EDGE, 3, answer(truths[EDGE], "Read."))]
    path = make_log(base_log, tmp_path / "fb.eval", MODEL_A, eps, eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    assert p.carried == ()
    assert [u.fm_code for u in p.units if u.row_kind == "fm_audit"] == ["FM-000"]


# --- Composition: the evaluation split's 200 cases (section 4) ---------------------------------

def _kind_samples(counts, k=4):
    out = []
    for kind, n in counts.items():
        for i in range(n):
            out += [SimpleNamespace(id=f"eval-mock-{kind}-{i:03d}", epoch=e, metadata={"truth": {"case_kind": kind}})
                    for e in range(1, k + 1)]
    return out


def test_composition_of_the_evaluation_split():
    assert rp.EVAL_COMPOSITION == {"standard": 146, "edge": 30, "adversarial": 24}
    assert rp.eval_composition_gaps(_kind_samples({"standard": 146, "edge": 30, "adversarial": 24})) == []
    short = rp.eval_composition_gaps(_kind_samples({"standard": 146, "edge": 29, "adversarial": 24}))
    assert "it holds 199 cases, not 200" in short and "it holds 29 edge cases, not 30" in short
    swapped = rp.eval_composition_gaps(_kind_samples({"standard": 145, "edge": 31, "adversarial": 24}))
    assert swapped == ["it holds 31 edge cases, not 30", "it holds 145 standard cases, not 146"]
    odd = rp.eval_composition_gaps(_kind_samples({"standard": 146, "edge": 30, "adversarial": 23, "other": 1}))
    assert "it holds 1 other cases, not 0" in odd and "it holds 23 adversarial cases, not 24" in odd
    mixed = _kind_samples({"standard": 146, "edge": 30, "adversarial": 24})
    mixed[0].metadata = {"truth": {"case_kind": "edge"}}
    assert any("names case kinds ['edge', 'standard']" in g for g in rp.eval_composition_gaps(mixed))


def test_evaluation_log_without_the_200_cases_is_refused(base_log, truths, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "check_eval_composition", REAL_COMPOSITION_CHECK)
    path = make_log(base_log, tmp_path / "ev.eval", MODEL_A, _good_epochs(truths), eval_split=True)
    with pytest.raises(rp.Refused, match="does not hold its 200 cases") as e:
        rp.load_runs([("alpha", path)], SEED)
    assert "it holds 2 cases, not 200" in str(e.value)
    out = tmp_path / "out"
    assert _main(["--run", f"alpha={path}", "--out", out, "--seed", SEED, "--run-id", "eval-mock-run"]) == 3
    assert not out.exists() and "200 cases" in capsys.readouterr().err
    # a dev log keeps its own counts
    dev = make_log(base_log, tmp_path / "dev.eval", MODEL_A, _good_epochs(truths))
    assert rp.load_runs([("alpha", dev)], SEED)[0].run_id == "R1"


# --- Narrative-only units outside the difficulty-bearing cases (section 8, D-J) ------------------

def test_summary_lists_narrative_only_units_outside_the_hard_cases(base_log, truths, tmp_path):
    # a P12 citation omission on an unrouted, uncoded epoch of a synthetic case adds a
    # narrative-only read (D-R9); it is not one of the D-J units section 8 counts
    t, te = truths[STD], truths[EDGE]
    p12 = {"fabrication_anchor": ((), (), {"citation_omissions": ["W-1"]})}
    eps = [E(STD, 1, answer(t, f"{S1} {S2}"), route(AMOUNT)), E(STD, 2, answer(t, "Names W-1 uncited."), p12),
           E(EDGE, 1, answer(te, f"{S1} {S2}"), route(AMOUNT)), E(EDGE, 2, answer(te, "A hard case narrative."))]
    path = make_log(base_log, tmp_path / "p12.eval", MODEL_A, eps)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="dev-run", seed=SEED)
    only = {u.case_id: u.unit_id for u in p.units if u.row_kind == "narrative" and u.route_class == "narrative_only"}
    assert set(only) == {STD, EDGE}
    assert p.narrative_only_outside == (only[STD],)
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"][rp.NARRATIVE_ONLY_OUTSIDE_KEY] == [only[STD]]
    assert rp.load_packet(str(out)).narrative_only_outside == (only[STD],)
    assert rp.load_narrative_only_outside(out) == (only[STD],)


def test_summary_has_no_outside_key_without_such_units(base_log, truths, tmp_path):
    path = make_log(base_log, tmp_path / "dev.eval", MODEL_A, _good_epochs(truths))
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="dev-run", seed=SEED)
    assert p.narrative_only_outside == ()
    assert rp.NARRATIVE_ONLY_OUTSIDE_KEY not in rp.summary(p)["totals"]


# --- Dev behaviour and determinism ------------------------------------------------------------

DEV_RUN_KEYS = {"label", "file_name", "sha256", "eval_id", "model", "revision_commit", "dataset_sha256",
                "template_baseline"}


def test_dev_packet_keeps_its_layout(base_log, truths, tmp_path):
    path = make_log(base_log, tmp_path / "dev.eval", MODEL_A, _good_epochs(truths))
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="dev-run", seed=SEED)
    assert rp.packet_split(p) == "dev" and p.carried == ()
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    runs_json = json.loads((out / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    assert set(runs_json["R1"]) == DEV_RUN_KEYS
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert "carried_failed_trials" not in summary["totals"] and "carried_failed_trials" not in summary["R1"]
    assert set(summary) == {"R1", "totals"}
    assert rp.summary(p) == summary


def test_dev_log_with_error_now_carried(base_log, truths, tmp_path, capsys):
    t = truths[STD]
    eps = _good_epochs(truths) + [E(STD, 3, answer(t, MARK), error="boom"),
                                  E(EDGE, 3, answer(truths[EDGE], "Read whole."))]
    path = make_log(base_log, tmp_path / "dev-err.eval", MODEL_A, eps)
    out = tmp_path / "packet"
    assert _main(["--run", f"alpha={path}", "--out", out, "--seed", SEED, "--run-id", "dev-run"]) == 0
    assert "carried failed trials (errored or unscored epochs, no finding unit): 1" in capsys.readouterr().out
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"]["carried_failed_trials"] == [
        {"run_id": "R1", "sample_id": STD, "epoch": 3, "reason": "sample error"}]
    runs_json = json.loads((out / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    assert set(runs_json["R1"]) == DEV_RUN_KEYS


def test_eval_determinism_and_reload(base_log, truths, tmp_path):
    a = make_log(base_log, tmp_path / "a.eval", MODEL_A, _with_errored(truths, error="boom"), eval_split=True)
    b = make_log(base_log, tmp_path / "b.eval", MODEL_B, _with_errored(truths, error="boom"), eval_split=True)
    args = ["--run", f"alpha={a}", "--run", f"beta={b}", "--seed", SEED, "--run-id", "eval-det"]
    assert _main([*args, "--out", tmp_path / "x"]) == 0
    assert _main([*args, "--out", tmp_path / "y"]) == 0
    assert _files(tmp_path / "x") == _files(tmp_path / "y")
    loaded = rp.load_packet(str(tmp_path / "x"))
    assert len(loaded.carried) == 4 and rp.packet_split(loaded) == "eval"
    rp.write_packet(loaded, str(tmp_path / "z"))
    for rel in (rs.PACKET_FILES["summary"], rs.PACKET_FILES["runs"], rs.PACKET_FILES["members"],
                rs.PACKET_FILES["packet"]):
        assert (tmp_path / "x" / rel).read_bytes() == (tmp_path / "z" / rel).read_bytes(), rel


def test_second_look_subset_carries_nothing(base_log, truths, tmp_path):
    path = make_log(base_log, tmp_path / "a.eval", MODEL_A, _with_errored(truths, error="boom"), eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="sl", seed=SEED)
    narr = [u.unit_id for u in p.units if u.row_kind == "narrative"][:1]
    sub, _ = rp.subset_packet(p, narr, seed=SEED + 1)
    assert sub.carried == ()
    rp.write_packet(sub, str(tmp_path / "sub"), mode="second_look")
    summary = json.loads((tmp_path / "sub" / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"]["carried_failed_trials"] == []


def test_load_carried_tolerates_absent_key(tmp_path):
    assert rp.load_carried(str(tmp_path)) == ()
    (tmp_path / "sealed").mkdir()
    (tmp_path / rs.PACKET_FILES["summary"]).write_text(json.dumps({"totals": {}}), encoding="utf-8")
    assert rp.load_carried(str(tmp_path)) == ()
    rows = [{"run_id": "R2", "sample_id": "eval-mock-002", "epoch": 1, "reason": "sample error"},
            {"run_id": "R1", "sample_id": "eval-mock-009", "epoch": 2, "reason": "sample error"}]
    (tmp_path / rs.PACKET_FILES["summary"]).write_text(json.dumps({"totals": {"carried_failed_trials": rows}}),
                                                       encoding="utf-8")
    got = rp.load_carried(str(tmp_path))
    assert [(c.run_id, c.sample_id) for c in got] == [("R1", "eval-mock-009"), ("R2", "eval-mock-002")]
    assert got[0].as_dict() == rows[1]


# --- The template baseline in the packet (decision D-R20) ----------------------------------------

TEMPLATE_TASK = "icfr_agent_evals/variance_commentary_template_baseline"


def _as_template(log):
    log.eval.task = TEMPLATE_TASK


def test_template_run_gets_units_like_a_worker_run(base_log, truths, tmp_path):
    # The template run's epochs build findings, narrative reads (the narrative-only reads of
    # the difficulty-bearing cases included) and audit units exactly as a worker run's do.
    eps = _good_epochs(truths)
    worker = make_log(base_log, tmp_path / "w.eval", MODEL_A, eps, eval_split=True)
    tmpl = make_log(base_log, tmp_path / "t.eval", "none/none", copy.deepcopy(eps), eval_split=True,
                    mutate=_as_template)
    p = rp.build_packet(rp.load_runs([("alpha", worker), ("template", tmpl)], SEED), run_id="tmpl", seed=SEED)
    rid = {r.label: r.run_id for r in p.runs}
    assert [rp.is_template_run(r) for r in p.runs if r.label == "template"] == [True]

    def shape(run_id):
        return sorted((m.sample_id, m.epoch, m.member_kind, m.route_class, m.fm_codes, m.unit_id)
                      for m in p.members if m.run_id == run_id)

    assert shape(rid["template"]) == shape(rid["alpha"])  # one log's twin: the same units
    hard = next(str(s.id) for s in read_eval_log(tmpl).samples if truth_of(s.metadata).case_id == EDGE)
    only = [m for m in p.members if m.run_id == rid["template"] and m.route_class == rs.NARRATIVE_CLASS_ONLY]
    assert [(m.sample_id, m.epoch) for m in only] == [(hard, 2)]
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    runs_json = json.loads((out / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    assert runs_json[rid["template"]]["template_baseline"] is True
    assert runs_json[rid["alpha"]]["template_baseline"] is False
    assert [rp.is_template_run(r) for r in rp.load_packet(str(out)).runs if r.label == "template"] == [True]


def test_second_look_subset_never_brings_in_template_epochs(base_log, truths, tmp_path):
    # A drawn narrative unit the template run shares with a worker run (the same text and
    # fields) brings the worker's epochs into the second look, never the template's.
    eps = _good_epochs(truths)
    worker = make_log(base_log, tmp_path / "w.eval", MODEL_A, eps, eval_split=True)
    tmpl = make_log(base_log, tmp_path / "t.eval", "none/none", copy.deepcopy(eps), eval_split=True,
                    mutate=_as_template)
    p = rp.build_packet(rp.load_runs([("alpha", worker), ("template", tmpl)], SEED), run_id="tmpl", seed=SEED)
    rid = {r.label: r.run_id for r in p.runs}
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    for packet in (p, rp.load_packet(str(out))):
        shared = sorted({m.unit_id for m in packet.members if m.member_kind == "narrative"})
        assert {m.run_id for m in packet.members if m.unit_id in shared} == {rid["alpha"], rid["template"]}
        sub, _ = rp.subset_packet(packet, shared, seed=SEED + 1)
        assert sub.members and {m.run_id for m in sub.members} == {rid["alpha"]}
        assert [u for u in sub.units if u.row_kind == "finding"]


def test_evaluation_packet_needs_exactly_one_template_run(base_log, truths, tmp_path, monkeypatch, capsys):
    # Decision D-R20 (ruling 11): on the evaluation split the template baseline goes in the
    # review packet, so a packet without exactly one template run is refused (exit 3).
    monkeypatch.setattr(rp, "check_eval_template", REAL_TEMPLATE_CHECK)
    eps = _good_epochs(truths)
    worker = make_log(base_log, tmp_path / "w.eval", MODEL_A, eps, eval_split=True)
    other = make_log(base_log, tmp_path / "o.eval", MODEL_B, copy.deepcopy(eps), eval_split=True)
    tmpl = make_log(base_log, tmp_path / "t.eval", "none/none", copy.deepcopy(eps), eval_split=True,
                    mutate=_as_template)
    tmpl2 = make_log(base_log, tmp_path / "t2.eval", "none/none", copy.deepcopy(eps), eval_split=True,
                     mutate=_as_template)
    for specs, n in (([("alpha", worker)], 0), ([("alpha", worker), ("beta", other)], 0),
                     ([("alpha", worker), ("t1", tmpl), ("t2", tmpl2)], 2)):
        with pytest.raises(rp.Refused, match=rf"exactly one --run that is the template baseline log .*; {n} given"):
            rp.load_runs(specs, SEED)
    out = tmp_path / "out"
    assert _main(["--run", f"alpha={worker}", "--out", out, "--seed", SEED, "--run-id", "eval-mock-run"]) == 3
    assert not out.exists() and "template baseline" in capsys.readouterr().err
    runs = rp.load_runs([("alpha", worker), ("beta", other), ("template", tmpl)], SEED)
    assert sum(rp.is_template_run(r) for r in runs) == 1
    good = tmp_path / "good"
    assert _main(["--run", f"alpha={worker}", "--run", f"template={tmpl}", "--out", good, "--seed", SEED,
                  "--run-id", "eval-mock-run"]) == 0
    # a dev packet needs no template run, as before
    dev = make_log(base_log, tmp_path / "dev.eval", MODEL_A, _good_epochs(truths))
    assert rp.load_runs([("alpha", dev)], SEED)[0].run_id == "R1"


def test_errored_epoch_audit_evidence_and_tag_wording():
    # Decision D-R20: the evidence says the epoch ended in an error, never that no output was
    # scored, and never shows the error text; the page tag is data on the unit, not a label.
    assert rp.ERRORED_EVIDENCE == "sample error: the epoch ended in an error (decision D-R20)"
    assert rp.ERRORED_TAG == "the epoch ended in an error; output shown as written"
    assert rp.ERRORED_TAG not in rp.UI_TEXT.values()
    errored = SimpleNamespace(error=SimpleNamespace(message="boom: provider error"), scores={})
    assert rp._audit_evidence(errored, "FM-000") == (rp.ERRORED_EVIDENCE,)
    assert rp._audit_evidence(SimpleNamespace(error=None, scores={}), "FM-000") == (rp.STRUCTURED_LINE,)


def test_errored_epoch_that_does_not_parse_shows_its_own_tag(base_log, truths, tmp_path):
    # An errored epoch whose completion does not parse is shown as written, with the errored
    # tag in place of the unparsed tag; a parse failure with no sample error keeps the unparsed tag.
    t = truths[STD]
    eps = [E(STD, 1, answer(t, "Plain narrative.")), E(STD, 2, "not json, errored", error="boom"),
           E(STD, 3, "not json, parse failure", scores={"correctness_anchor": ((), ("FM-000",), {})})]
    path = make_log(base_log, tmp_path / "np.eval", MODEL_A, eps)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="np", seed=SEED, scope="all")
    by_text = {u.narrative: u for u in p.units if u.row_kind == "narrative"}
    err, broke = by_text["not json, errored"], by_text["not json, parse failure"]
    assert (err.parsed_ok, err.errored, broke.parsed_ok, broke.errored) == (False, True, False, False)
    data = rp._page_data(p, "t", "review")["units"]
    assert data[err.unit_id]["errored_tag"] == rp.ERRORED_TAG and "errored_tag" not in data[broke.unit_id]
    assert "errored_tag" not in data[by_text["Plain narrative."].unit_id]


def test_errored_narrative_is_never_grouped_with_the_same_text_unerrored(base_log, truths, tmp_path):
    # The errored tag sits on the unit, so the same completion on an epoch that did not error
    # is its own unit; two errored epochs with one text still share one unit.
    t = truths[STD]
    same = answer(t, "One text on every epoch.")
    eps = [E(STD, 1, same), E(STD, 2, same), E(STD, 3, same, error="boom"), E(STD, 4, same, error="boom")]
    path = make_log(base_log, tmp_path / "grp.eval", MODEL_A, eps)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="grp", seed=SEED, scope="all")
    unit_of = {m.epoch: m.unit_id for m in p.members if m.member_kind == "narrative"}
    assert unit_of[1] == unit_of[2] and unit_of[3] == unit_of[4] and unit_of[1] != unit_of[3]
    by_id = {u.unit_id: u for u in p.units}
    assert (by_id[unit_of[1]].errored, by_id[unit_of[3]].errored) == (False, True)


def test_public_text_rules():
    for path in (SCRIPT, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert not re.search("[\u2013\u2014]", text), path
