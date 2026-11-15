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
    assert "carried failed trials (errored or unscored epochs, no review unit): 0" in capsys.readouterr().out
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

def _with_errored(truths, **errored):
    """The good epochs plus epoch 3 of each case: errored, holding a planted route, an FM
    code and the marker sentence, so any unit built from it would show."""
    t, te = truths[STD], truths[EDGE]
    return _good_epochs(truths) + [
        E(STD, 3, answer(t, f"{S1} {S2}"), route(AMOUNT)),
        E(EDGE, 3, answer(te, "A hard case narrative read whole.")),
        E(STD, 4, answer(t, f"{MARK} {S1}"), route(AMOUNT, "direction_negated: fell short", fms=("FM-001",)),
          **errored),
        E(EDGE, 4, answer(te, MARK), **errored),
    ]


def _copy_epochs(truths):
    """The same log with each errored epoch 4 replaced by a copy of epoch 1: the units it
    builds are exactly the units of the errored log, if the errored epochs build none."""
    t, te = truths[STD], truths[EDGE]
    eps = _good_epochs(truths)
    eps += [E(STD, 3, answer(t, f"{S1} {S2}"), route(AMOUNT)), E(EDGE, 3, answer(te, "A hard case narrative read whole."))]
    eps += [E(STD, 4, answer(t, f"{S1} {S2}"), route(AMOUNT)), E(EDGE, 4, answer(te, f"{S1} {S2}"), route(AMOUNT))]
    return eps


@pytest.mark.parametrize("eval_split", [True, False])
def test_errored_epoch_carried_with_no_unit(base_log, truths, tmp_path, eval_split):
    path = make_log(base_log, tmp_path / "err.eval", MODEL_A, _with_errored(truths, error="boom: provider error"),
                    eval_split=eval_split)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    sids = sorted({str(s.id) for s in read_eval_log(path).samples})
    assert [(c.run_id, c.sample_id, c.epoch, c.reason) for c in p.carried] == [
        ("R1", sid, 4, "sample error") for sid in sids]
    assert not [m for m in p.members if m.epoch == 4]
    assert not [u for u in p.units if MARK in u.narrative or "direction_negated" in " ".join(u.reasons)]
    assert not [u for u in p.units if u.row_kind == "fm_audit"]
    # the hard case's epochs 1 to 3 and the standard case's routed epochs still get their
    # narrative reads; epoch 4 adds none (sids[0] is the hard case in both namings)
    edge, std = sids
    assert {(m.sample_id, m.epoch) for m in p.members if m.member_kind == "narrative"} == {
        (edge, 1), (edge, 2), (edge, 3), (std, 1), (std, 3)}
    out = tmp_path / "packet"
    rp.write_packet(p, str(out))
    summary = json.loads((out / rs.PACKET_FILES["summary"]).read_text(encoding="utf-8"))
    assert summary["totals"]["carried_failed_trials"] == [
        {"run_id": "R1", "sample_id": sid, "epoch": 4, "reason": "sample error"} for sid in sids]
    assert summary["R1"]["carried_failed_trials"] == 2
    rows = rs.read_csv(out / rs.PACKET_FILES["members"], rs.MEMBER_COLUMNS)
    assert rows and not [r for r in rows if r["epoch"] == "4"]


def test_errored_epoch_leaves_no_trace_in_reviewer_files(base_log, truths, tmp_path):
    err = make_log(base_log, tmp_path / "err.eval", MODEL_A, _with_errored(truths, error="boom: provider error"),
                   eval_split=True)
    same = make_log(base_log, tmp_path / "same.eval", MODEL_A, _copy_epochs(truths), eval_split=True)
    built = {}
    for name, path in (("err", err), ("same", same)):
        p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
        rp.write_packet(p, str(tmp_path / name))
        built[name] = _reviewer_texts(tmp_path / name)
    # byte for byte the reviewer side of a log whose errored epochs are clean copies
    assert built["err"] == built["same"]
    assert built["err"]
    for rel, text in built["err"].items():
        for trace in (MARK, "boom", "provider error", "direction_negated", MODEL_A):
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
    assert not [m for m in p.members if m.epoch == 3 and m.sample_id == p.carried[0].sample_id]


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


def test_case_with_every_epoch_carried_is_not_shown(base_log, truths, tmp_path):
    t = truths[THIRD]
    eps = _good_epochs(truths) + [E(THIRD, 1, answer(t, MARK), error="boom"), E(THIRD, 2, answer(t, MARK), error="boom")]
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


def test_format_break_is_not_carried(base_log, truths, tmp_path):
    # FM-000 keeps its audit unit and its narrative read (section 6 and section 8)
    reason = "format break: not one JSON object: Expecting value: line 1 column 1 (char 0)"
    fb = {name: ((reason,), ("FM-000",), {}) for name in ANCHORS}
    eps = _good_epochs(truths) + [E(STD, 3, "not json at all", fb), E(EDGE, 3, answer(truths[EDGE], "Read."))]
    path = make_log(base_log, tmp_path / "fb.eval", MODEL_A, eps, eval_split=True)
    p = rp.build_packet(rp.load_runs([("alpha", path)], SEED), run_id="carry", seed=SEED)
    assert p.carried == ()
    assert [u.fm_code for u in p.units if u.row_kind == "fm_audit"] == ["FM-000"]


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
    assert "carried failed trials (errored or unscored epochs, no review unit): 1" in capsys.readouterr().out
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


def test_public_text_rules():
    for path in (SCRIPT, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert not re.search("[\u2013\u2014]", text), path
