"""scripts/grading_common.py: worker-log admission, units, seeds, the calibrator draw, the
pooled order, grader messages, calls in a log, failed judgments, folder rules, sealed files,
the ledger and the start checks of the full-run grading job (decision D-R23).

Keyless and offline. The worker logs are mock copies of one small mockllm run of the 20 dev
pilot cases, with headers that claim the evaluation split (tests/gradingfactory.py). No
evaluation case is built, read or scored, and no model is called. The start checks run on
scratch git repositories under tmp_path.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import gradingfactory as gf
from icfr_agent_evals import roster, scoring
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals import review_facts as rf

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS
gc = gf.load_script("grading_common")
SHA = {m: hashlib.sha256(m.encode("utf-8")).hexdigest() for m in gf.MODELS}


# --- Fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return gf.base_runs(tmp_path_factory.mktemp("base"))


@pytest.fixture(scope="module")
def wl(runs, tmp_path_factory):
    """The four mock worker logs on disk, admitted, and the plan built from them."""
    good, bad = runs
    folder = tmp_path_factory.mktemp("worker-logs")
    run_commit = "a" * 40
    paths = gf.write_worker_logs(good, bad, folder, run_commit=run_commit)
    found = gc.find_worker_logs(folder, gf.MODELS)
    logs = gc.admit_worker_logs(found, cases=gf.dev_cases(), composition=gf.composition(), models=gf.MODELS,
                                run_commit=run_commit)
    plan = gc.build_plan(logs, gf.dev_cases(), calibrator_model=gf.DELTA, per_model=20)
    return SimpleNamespace(folder=folder, paths=paths, found=found, logs=logs, run_commit=run_commit, plan=plan,
                           by_model={w.model: w for w in logs})


def _kw(wl, **extra):
    return {"cases": gf.dev_cases(), "composition": gf.composition(), "models": gf.MODELS,
            "run_commit": wl.run_commit, **extra}


def _gen(model, sid, epoch, status="gradeable", kind="standard", defect=False):
    return {"model": model, "sample_id": sid, "epoch": epoch, "case_kind": kind, "status": status,
            "defect_epoch": defect, "uid": None}


def _synthetic(models, n_cases, *, unparsed=()):
    gens = []
    for m in models:
        for i in range(n_cases):
            sid = f"c{i:03d}"
            for e in range(1, 5):
                status = gc.GEN_UNPARSED if (m, sid) in unparsed else gc.GEN_GRADEABLE
                gens.append(_gen(m, sid, e, status))
    return sorted(gens, key=lambda g: (g["model"], g["sample_id"], g["epoch"]))


# --- Constants --------------------------------------------------------------------------------

def test_fixed_constants():
    assert gc.RUN_LABEL == "2026-11-21" and gc.RUN_COMMIT == "0c745021275f67d8a3349f0ff0e71bada299b07b"
    assert gc.GRADER_STEP == 2 and gc.WORKER_EPOCHS == 4 and gc.EVAL_SPLIT == "eval"
    assert gc.EVAL_COMPOSITION == {"standard": 146, "edge": 30, "adversarial": 24}
    assert gc.CALIBRATOR_PER_MODEL == 80 and gc.GRADER_PARTS == 4
    assert gc.SEALED_FOLDER_NAME == "_sealed" and gc.LEDGER_FOLDER == "icfr-grading"
    assert gc.LEDGER_FILE == "grading-2026-11-21.json" and gc.EVIDENCE_FILE == "grading-2026-11-21.json"
    assert gc.ANCHORS_PREFIX == "405813382714b30a" and gc.SEED_HEX_DIGITS == 16
    assert gc.GRADER_PART_KEYS == ("grader-1", "grader-2", "grader-3", "grader-4")
    assert gc.CALIBRATOR_PART_KEY == "calibrator" and gc.UNIT_ID_FORMAT.format(7) == "u0007"
    assert gc.PLAN_FILES == ("unit-map.json", "draw-pool.json", "drawn.json")
    assert gc.RESULT_FILES == ("grader-replies.json", "calibrator-replies.json", "queue.json", "aggregates.json")
    assert gc.REFUSAL_STOP_REASON == "content_filter" and gc.PYTHON_PIN == "3.14.10"
    assert not gc.inside(gc.WORKER_LOGS_DIR, REPO) and not gc.inside(gc.GRADING_ROOT, REPO)
    assert len(gc.SAMPLING_CALLS) == 5 and all(isinstance(s, str) for s in gc.SAMPLING_CALLS)
    assert gc.REPLY_FAILED == "failed"
    assert gc.REPLY_STATUSES == ("parsed", "unparsed", "refused", "failed")
    assert gc.REFUSAL_ERROR == "refused (stop reason content_filter)"
    assert gc.REFUSAL_ERROR_CONTENT == "refused (refusal content)"
    assert gc.REFUSAL_ERROR_DETAILS == "refused (refusal stop details)"
    assert gc.REFUSAL_DETAIL_TYPES == ("refusal", "content_filter")
    assert gc.FAILED_JUDGMENT_ASKS == 3
    assert gc.FAILED_JUDGMENT_ERROR == ("no completed call in 3 attempt logs, the third in an eval of units at their "
                                        "third ask, one sample at a time")
    assert gc.LEDGER_LOCK_FILE == "grading-2026-11-21.lock"
    assert gc.DRAW_RULE_TEXT.startswith("the case ids, sorted, are shuffled with the |calibrator seed")


# --- Seeds ------------------------------------------------------------------------------------

def test_rule_seed_follows_the_recipe():
    shas = {"b/two": "2" * 64, "a/one": "1" * 64, "c/three": "3" * 64}
    material = "1" * 64 + "|" + "2" * 64 + "|" + "3" * 64 + "|calibrator"
    want = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    assert gc.rule_seed(shas, "|calibrator") == want
    assert gc.rule_seed(dict(reversed(list(shas.items()))), "|calibrator") == want  # sorted by model id, not order
    assert 0 <= want < 16 ** 16


def test_each_seed_has_its_own_suffix_and_unknown_suffixes_are_refused():
    seeds = gc.plan_seeds(SHA)
    assert set(seeds) == {"calibrator_draw", "grader_order", "calibrator_order", "bootstrap"}
    assert seeds["calibrator_draw"] == gc.rule_seed(SHA, gc.SEED_SUFFIX_CALIBRATOR)
    assert seeds["grader_order"] == gc.rule_seed(SHA, "|grader-order")
    assert seeds["calibrator_order"] == gc.rule_seed(SHA, "|calibrator-order")
    assert seeds["bootstrap"] == gc.rule_seed(SHA, "|bootstrap")
    assert len(set(seeds.values())) == 4
    for bad in ("|other", "", "calibrator", "|Calibrator"):
        with pytest.raises(gc.Refused):
            gc.rule_seed(SHA, bad)
    with pytest.raises(gc.Refused):
        gc.rule_seed({gf.ALPHA: SHA[gf.ALPHA]}, "|calibrator")       # fewer than 2 logs
    with pytest.raises(gc.Refused):
        gc.rule_seed({**SHA, gf.ALPHA: SHA[gf.ALPHA].upper()}, "|calibrator")
    with pytest.raises(gc.Refused):
        gc.rule_seed({**SHA, gf.ALPHA: "abc"}, "|calibrator")
    assert "|grader-order" in gc.SEED_RULE_TEXT and "16 hex digits" in gc.SEED_RULE_TEXT


# --- Helpers ----------------------------------------------------------------------------------

def test_canonical_bytes_and_message_hashes():
    body = {"b": 1, "a": ["é", 2]}
    assert gc.canonical_bytes(body) == (json.dumps(body, indent=1, ensure_ascii=False, sort_keys=True)
                                        + "\n").encode("utf-8")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "ü", "extra": "dropped"}]
    want = hashlib.sha256(json.dumps([{"role": "system", "content": "s"}, {"role": "user", "content": "ü"}],
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert gc.messages_sha256(msgs) == want
    assert gc.chat_messages_sha256(gp.to_chat_messages(msgs[:1] + [{"role": "user", "content": "ü"}])) == want
    assert gc.input_text("plain") == "plain"
    assert gc.completion(SimpleNamespace(output=None)) == ""
    assert gc.is_refusal(SimpleNamespace(stop_reason="content_filter"))
    assert not gc.is_refusal(SimpleNamespace(stop_reason="stop"))


def test_refusals_are_read_from_stop_reason_stop_details_and_refusal_content():
    from inspect_ai.model import ContentText, ModelOutput, StopDetails

    m = "mockllm/model"
    content = ModelOutput.from_content(m, [ContentText(text="no", refusal=True)])
    assert content.stop_reason == "stop"
    assert gc.refusal_signal(content) == gc.REFUSAL_ERROR_CONTENT and gc.is_refusal(content)
    details = ModelOutput.from_content(m, "no", stop_details=StopDetails(type="refusal"))
    assert gc.refusal_signal(details) == gc.REFUSAL_ERROR_DETAILS
    filtered = ModelOutput.from_content(m, "no", stop_details=StopDetails(type="content_filter"))
    assert gc.refusal_signal(filtered) == gc.REFUSAL_ERROR_DETAILS
    stop_reason = ModelOutput.from_content(m, "no", stop_reason="content_filter",
                                           stop_details=StopDetails(type="refusal"))
    assert gc.refusal_signal(stop_reason) == gc.REFUSAL_ERROR          # the stop reason is checked first
    for plain in (ModelOutput.from_content(m, "no", stop_details=StopDetails(type="max_tokens")),
                  ModelOutput.from_content(m, [ContentText(text="no", refusal=False)]),
                  ModelOutput.from_content(m, "no")):
        assert gc.refusal_signal(plain) is None and not gc.is_refusal(plain)
    # a stand-in output with no choices still works and never raises
    assert gc.refusal_signal(SimpleNamespace(stop_reason="content_filter")) == gc.REFUSAL_ERROR
    assert gc.refusal_signal(SimpleNamespace(stop_reason="stop")) is None
    assert gc.refusal_signal(None) is None
    # an output whose stop reason read raises still never raises here
    class _RaisingStop:
        choices = [SimpleNamespace(message=SimpleNamespace(content="text"))]

        @property
        def stop_reason(self):
            raise IndexError("list index out of range")
    assert gc.refusal_signal(_RaisingStop()) is None
    text_only =SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="text"))])
    assert gc.refusal_signal(text_only) is None
    for output, error in ((content, gc.REFUSAL_ERROR_CONTENT), (details, gc.REFUSAL_ERROR_DETAILS)):
        got = gc.reply_record(SimpleNamespace(output=output), attempt=1, log_file="r.eval")
        assert got["status"] == gc.REPLY_REFUSED and got["error"] == error
        assert got["scores"] == {} and got["reply"] == "no"


def test_an_output_with_no_choices_records_no_refusal_and_never_raises():
    from inspect_ai.model import ModelOutput

    empty = ModelOutput(model="mockllm/model", choices=[])
    with pytest.raises(IndexError):
        empty.stop_reason                                   # Inspect reads it from the first choice
    assert gc.refusal_signal(empty) is None and not gc.is_refusal(empty)
    got = gc.reply_record(SimpleNamespace(output=empty), attempt=1, log_file="r.eval")
    assert got["status"] == gc.REPLY_UNPARSED and got["error"]
    assert got["scores"] == {} and got["reply"] == "" and got["usage"] is None


# --- Admission --------------------------------------------------------------------------------

def test_find_worker_logs_needs_one_log_per_model_folder_outside_the_repo(wl, tmp_path):
    assert [p.parent.name for p in wl.found] == sorted(gf.FOLDERS)
    assert {p.name for p in wl.found} == {p.name for p in wl.paths.values()}
    folder = tmp_path / "wl"
    for name in gf.FOLDERS:
        (folder / name).mkdir(parents=True)
        (folder / name / "x.eval").write_bytes(b"x")
    (folder / gf.FOLDERS[1] / "y.eval").write_bytes(b"y")
    with pytest.raises(gc.Refused, match="holds 2 .eval files, not one"):
        gc.find_worker_logs(folder, gf.MODELS)
    (folder / gf.FOLDERS[1] / "y.eval").unlink()
    (folder / gf.FOLDERS[2] / "x.eval").unlink()
    with pytest.raises(gc.Refused, match="holds 0 .eval files, not one"):
        gc.find_worker_logs(folder, gf.MODELS)
    with pytest.raises(gc.Refused):
        gc.find_worker_logs(REPO / "logs", gf.MODELS)


def test_find_worker_logs_refuses_a_folder_inside_the_given_repo(tmp_path):
    repo = gf.make_repo(tmp_path).repo
    folder = repo / "wl"
    for name in gf.FOLDERS:
        (folder / name).mkdir(parents=True)
        (folder / name / "x.eval").write_bytes(b"x")
    assert len(gc.find_worker_logs(folder, gf.MODELS)) == 4                # outside this repository
    with pytest.raises(gc.Refused, match="inside the repository"):
        gc.find_worker_logs(folder, gf.MODELS, repo=repo)
    with pytest.raises(gc.Refused, match="inside the repository"):
        gc.find_worker_logs(REPO / "logs", gf.MODELS, repo=repo)          # this repository is always refused


def test_admitted_logs_record_file_hash_and_eval_id(wl):
    assert [w.model for w in wl.logs] == sorted(gf.MODELS)
    for w in wl.logs:
        path = wl.paths[w.model]
        assert w.file == path.name and w.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert w.eval_id == w.log.eval.eval_id and w.header_grader_step == 1
        assert w.revision_commit == wl.run_commit and w.revision_dirty is False
        assert len(w.by_key) == 80 and isinstance(w.run_config, dict)
        assert isinstance(w.resolved, tuple)


def _status(log):
    log.status = "error"


def _split(log):
    log.eval.task_args = {**log.eval.task_args, "split": "dev"}


def _unlock(log):
    log.eval.task_args = {**log.eval.task_args, "allow_evaluation_split": "true"}


def _partial(log):
    log.eval.dataset.samples = 21


def _epochs(log):
    log.eval.config.epochs = 3


def _template(log):
    log.eval.task = "variance_commentary_template_baseline"


def _run_config(log):
    log.plan.config.temperature = 0.5


def _reasoning(log):
    from inspect_ai.model import ModelCall

    event = next(e for e in log.samples[0].events if getattr(e, "event", None) == "model" and e.role is None)
    event.call = ModelCall(request={"output_config": {"effort": "low"}}, response={})


def _prompt(log):
    log.eval.metadata = {**log.eval.metadata, "worker_prompt_sha256": "0" * 64}


def _step(log):
    log.eval.task_args = {**log.eval.task_args, "grader_step": 2}


def _model(log):
    log.eval.model = "mockllm/not-on-the-list"


def _revision(log):
    log.eval.revision.commit = "f" * 40


def _no_revision(log):
    log.eval.revision = None


def _dataset_hash(log):
    log.eval.metadata = {**log.eval.metadata, "split_dataset_sha256": "0" * 64}


def _sample_input(log):
    log.samples[5].input = log.samples[5].input + " changed"


def _sample_meta(log):
    log.samples[5].metadata = {**log.samples[5].metadata, "extra": 1}


def _parse(log):
    from inspect_ai.model import ModelOutput

    s = next(s for s in log.samples if s.error is None and scoring.parse_worker_output(s.output.completion).ok)
    s.output = ModelOutput.from_content("mockllm/model", "not a worker output")


def _missing_epoch(log):
    # the first pilot case loses its epoch 4; the header still says 4 epochs
    sid = gf.pilot_ids()[0]
    log.samples = [s for s in log.samples if not (str(s.id) == sid and int(s.epoch) == 4)]


@pytest.mark.parametrize("mutate, extra, says", [
    (_status, {}, "status is"), (_split, {}, "split"), (_unlock, {}, "only the literal True unlocks it"),
    (None, {"composition": {"standard": 1}}, "composition"),
    (_partial, {}, "differ from the dataset's sample count"), (_epochs, {}, "epochs"),
    (_template, {}, "template baseline"), (_run_config, {}, "run config"),
    (_reasoning, {}, "different reasoning modes"), (_prompt, {}, "worker_prompt_sha256"),
    (_step, {}, "grader_step"), (_model, {}, "evaluated models"), (_revision, {}, "revision commit"),
    (_no_revision, {}, "records no revision"), (_dataset_hash, {}, "split_dataset_sha256"),
    (_sample_input, {}, "input differs"), (_sample_meta, {}, "metadata differs"), (_parse, {}, "format break"),
    (_missing_epoch, {}, "holds epochs [1, 2, 3], not 1 to 4"),
], ids=["status", "split", "unlock", "composition", "partial-run", "epochs", "template", "run-config",
        "reasoning-mode", "worker-prompt-sha", "header-grader-step", "model", "revision-commit", "no-revision",
        "dataset-hash", "sample-input", "sample-metadata", "parse-consistency", "missing-epoch"])
def test_admission_refuses_each_failed_check(wl, mutate, extra, says):
    path = wl.paths[gf.ALPHA]
    log = wl.by_model[gf.ALPHA].log.model_copy(deep=True)
    gc.admit_worker_log(path, log.model_copy(deep=True), **_kw(wl))            # the unmutated copy passes
    if mutate is not None:
        mutate(log)
    with pytest.raises(gc.Refused) as e:
        gc.admit_worker_log(path, log, **_kw(wl, **extra))
    text = str(e.value)
    assert text.startswith(f"worker log {path.name}: ")
    assert says in text, text                                                  # refused by its own check
    if mutate is _missing_epoch:
        assert gf.pilot_ids()[0] in text
    if mutate is not _model:
        assert not any(m in text for m in gf.MODELS)


def test_admission_names_a_sample_by_id_and_epoch(wl):
    path = wl.paths[gf.ALPHA]
    log = wl.by_model[gf.ALPHA].log.model_copy(deep=True)
    _sample_input(log)
    with pytest.raises(gc.Refused) as e:
        gc.admit_worker_log(path, log, **_kw(wl))
    assert str(log.samples[5].id) in str(e.value)


def test_admission_refuses_two_logs_of_one_model_and_shared_eval_ids(wl):
    paths = [wl.paths[m] for m in sorted(gf.MODELS)]
    base = {wl.paths[w.model]: w.log for w in wl.logs}

    def reader(overrides):
        def read(p):
            log = base[p].model_copy(deep=True)
            if p in overrides:
                overrides[p](log)
            return log
        return read

    ok = gc.admit_worker_logs(paths, read=reader({}), **_kw(wl))
    assert [w.model for w in ok] == sorted(gf.MODELS)

    def as_alpha(log):
        log.eval.model = gf.ALPHA
        log.eval.eval_id = "a-distinct-eval-id-0001"

    with pytest.raises(gc.Refused):
        gc.admit_worker_logs(paths, read=reader({wl.paths[gf.BRAVO]: as_alpha}), **_kw(wl))
    shared = wl.by_model[gf.ALPHA].eval_id

    def same_id(log):
        log.eval.eval_id = shared

    with pytest.raises(gc.Refused):
        gc.admit_worker_logs(paths, read=reader({wl.paths[gf.BRAVO]: same_id}), **_kw(wl))
    with pytest.raises(gc.Refused):
        gc.admit_worker_logs(paths[:3], read=reader({}), **_kw(wl))

    def fewer_samples(log):
        log.samples = [s for s in log.samples if str(s.id) != gf.pilot_ids()[0]]
        log.eval.dataset.samples = 19
        log.eval.dataset.sample_ids = sorted({str(s.id) for s in log.samples})

    with pytest.raises(gc.Refused):
        gc.admit_worker_logs(paths, read=reader({wl.paths[gf.CHARLIE]: fewer_samples}), **_kw(wl))


def test_admit_worker_logs_checks_the_log_count_before_reading_any_log(wl):
    def no_read(path):
        raise AssertionError("a worker log was read before the log count was checked")

    paths = [wl.paths[m] for m in sorted(gf.MODELS)]
    with pytest.raises(gc.Refused, match="3 worker logs given for 4 evaluated models"):
        gc.admit_worker_logs(paths[:3], read=no_read, **_kw(wl))
    with pytest.raises(gc.Refused, match="5 worker logs given for 4 evaluated models"):
        gc.admit_worker_logs(paths + paths[:1], read=no_read, **_kw(wl))


def test_the_worker_logs_are_never_written(wl):
    for m, p in wl.paths.items():
        assert wl.by_model[m].sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


# --- Generations, units, draw and order -------------------------------------------------------

def _planted(plan):
    pid = gf.pilot_ids()
    by = {(g["model"], g["sample_id"], g["epoch"]): g for g in plan["generations"]}
    return by, pid


def test_unparsed_and_errored_worker_outputs_are_skipped_and_counted(wl):
    plan = wl.plan
    by, pid = _planted(plan)
    m, ci, ep = gf.UNPARSED_EPOCH
    assert by[(m, pid[ci], ep)]["status"] == gc.GEN_UNPARSED and by[(m, pid[ci], ep)]["uid"] is None
    m, ci, ep = gf.ERRORED_EPOCH
    assert by[(m, pid[ci], ep)]["status"] == gc.GEN_ERRORED and by[(m, pid[ci], ep)]["uid"] is None
    m, ci = gf.NO_GRADEABLE
    assert all(by[(m, pid[ci], e)]["status"] == gc.GEN_UNPARSED for e in range(1, 5))
    counts = plan["counts"]
    assert counts == {**counts, **{k: gf.EXPECTED[k] for k in ("generations", "skipped_errored", "skipped_unparsed",
                                                                "units")}}
    assert len(plan["units"]) == gf.EXPECTED["units"] == 314
    assert all(g["uid"] is not None for g in plan["generations"] if g["status"] == gc.GEN_GRADEABLE)
    assert {u["model"] for u in plan["units"].values()} == set(gf.MODELS)      # Opus stand-in graded too
    assert [g for g in plan["generations"]] == sorted(plan["generations"],
                                                       key=lambda g: (g["model"], g["sample_id"], g["epoch"]))


def test_an_errored_epoch_whose_completion_parses_is_still_skipped(wl):
    m, ci, ep = gf.ERRORED_EPOCH
    sample = wl.by_model[m].by_key[(gf.pilot_ids()[ci], ep)]
    assert sample.error is not None and scoring.parse_worker_output(gc.completion(sample)).ok
    assert gc.classify_generation(sample) == gc.GEN_ERRORED
    out = SimpleNamespace(completion="nope")
    assert gc.classify_generation(SimpleNamespace(error=None, output=out)) == gc.GEN_UNPARSED
    assert gc.classify_generation(SimpleNamespace(error="x", output=out)) == gc.GEN_ERRORED
    good = wl.by_model[m].by_key[(gf.pilot_ids()[ci], 1)]
    assert gc.classify_generation(good) == gc.GEN_GRADEABLE


def test_unit_ids_are_opaque_positions_in_the_seeded_order(wl):
    gens = gc.build_generations(wl.logs, gf.dev_cases())
    assert all(g["uid"] is None for g in gens)
    units = gc.assign_unit_ids(gens, 4242)
    order = [g for g in sorted(gens, key=lambda g: (g["model"], g["sample_id"], g["epoch"]))
             if g["status"] == gc.GEN_GRADEABLE]
    want = [(g["model"], g["sample_id"], g["epoch"]) for g in order]
    random.Random(4242).shuffle(want)
    assert list(units) == [f"u{p:04d}" for p in range(1, len(want) + 1)]
    for p, key in enumerate(want, start=1):
        u = units[f"u{p:04d}"]
        assert (u["model"], u["sample_id"], u["epoch"]) == key and u["uid"] == f"u{p:04d}"
        assert set(u) == {"uid", "model", "sample_id", "epoch", "case_kind", "defect_epoch"}
    assert all(re.fullmatch(r"u\d{4}", g["uid"]) for g in gens if g["status"] == gc.GEN_GRADEABLE)
    with pytest.raises(gc.Refused):
        gc.assign_unit_ids([_gen("m/a", f"c{i}", e) for i in range(2500) for e in range(1, 5)], 1)


def test_unit_ids_do_not_depend_on_the_order_of_the_generations(wl):
    gens = gc.build_generations(wl.logs, gf.dev_cases())
    shuffled = copy.deepcopy(gens)
    random.Random(7).shuffle(shuffled)
    assert [g["model"] for g in shuffled] != [g["model"] for g in gens]
    want = gc.assign_unit_ids(copy.deepcopy(gens), 4242)
    got = gc.assign_unit_ids(shuffled, 4242)
    assert got == want
    by_key = {(g["model"], g["sample_id"], g["epoch"]): g["uid"] for g in shuffled}
    assert all(by_key[(u["model"], u["sample_id"], u["epoch"])] == uid for uid, u in got.items())


def test_split_parts_gives_roughly_equal_contiguous_parts():
    uids = [f"u{i:04d}" for i in range(1, 315)]
    parts = gc.split_parts(list(reversed(uids)))
    assert list(parts) == list(gc.GRADER_PART_KEYS)
    assert [len(parts[k]) for k in gc.GRADER_PART_KEYS] == [79, 79, 78, 78]
    assert sum((parts[k] for k in gc.GRADER_PART_KEYS), []) == uids
    assert [len(v) for v in gc.split_parts(uids[:8]).values()] == [2, 2, 2, 2]
    with pytest.raises(gc.Refused):
        gc.split_parts(uids[:3])


def _replica_draw(gens, seed, cal, per_model):
    ids = sorted({g["sample_id"] for g in gens})
    rng = random.Random(seed)
    rng.shuffle(ids)
    models = sorted({g["model"] for g in gens} - {cal})
    rows, lost = [], []
    for i, m in enumerate(models):
        for j in range(per_model):
            pos = (i * per_model + j) % len(ids)
            case = ids[pos]
            eps = sorted(g["epoch"] for g in gens if g["model"] == m and g["sample_id"] == case
                         and g["status"] == gc.GEN_GRADEABLE)
            if not eps:
                lost.append({"model": m, "sample_id": case, "position": pos + 1})
                continue
            e = rng.choice(eps)
            uid = next(g["uid"] for g in gens if (g["model"], g["sample_id"], g["epoch"]) == (m, case, e))
            rows.append({"uid": uid, "model": m, "sample_id": case, "epoch": e, "position": pos + 1})
    return ids, rows, lost


def test_case_rotation_assigns_positions_1_to_80_81_to_160_and_161_to_200_plus_1_to_40():
    models = ("m/one", "m/two", "m/three", "m/cal")
    gens = _synthetic(models, 200)
    gc.assign_unit_ids(gens, 99)
    draw = gc.draw_calibrator_sample(gens, seed=777, calibrator_model="m/cal", per_model=80)
    ids, rows, lost = _replica_draw(gens, 777, "m/cal", 80)
    assert draw["case_order"] == ids and draw["rows"] == rows and draw["lost"] == lost == []
    by_model = {}
    for r in draw["rows"]:
        by_model.setdefault(r["model"], []).append(r["position"])
    first, second, third = sorted(set(models) - {"m/cal"})
    assert by_model[first] == list(range(1, 81))
    assert by_model[second] == list(range(81, 161))
    assert by_model[third] == list(range(161, 201)) + list(range(1, 41))
    assert all(r["sample_id"] == ids[r["position"] - 1] for r in draw["rows"])
    assert len(draw["rows"]) == 240
    with pytest.raises(gc.Refused):
        gc.draw_calibrator_sample(gens, seed=1, calibrator_model="m/cal", per_model=201)


def test_draw_skips_the_calibrator_model_and_draws_one_gradeable_epoch_per_model_case(wl):
    plan = wl.plan
    draw = plan["draw"]
    by, pid = _planted(plan)
    assert {r["model"] for r in draw["rows"]} == {gf.ALPHA, gf.BRAVO, gf.CHARLIE}
    keys = [(r["model"], r["sample_id"]) for r in draw["rows"]]
    assert len(keys) == len(set(keys)) == 59
    for r in draw["rows"]:
        g = by[(r["model"], r["sample_id"], r["epoch"])]
        assert g["status"] == gc.GEN_GRADEABLE and g["uid"] == r["uid"]
        assert plan["units"][r["uid"]]["model"] == r["model"]
    m, ci = gf.NO_GRADEABLE
    assert [(x["model"], x["sample_id"]) for x in draw["lost"]] == [(m, pid[ci])]
    assert draw["pool"] == sorted(u for u, x in plan["units"].items() if x["model"] != gf.DELTA)
    assert plan["counts"]["rows_drawn"] == 59 and plan["counts"]["rows_lost"] == 1
    with pytest.raises(gc.Refused):
        gc.draw_calibrator_sample([{**_gen("m/a", "c1", 1), "uid": "u0001"}], seed=1, calibrator_model="m/a",
                                  per_model=1)                        # no model left once the calibrator's is set aside


def test_the_draw_refuses_a_calibrator_model_not_among_the_worker_runs():
    models = ("m/one", "m/two", "m/cal")
    gens = _synthetic(models, 4)
    gc.assign_unit_ids(gens, 3)
    assert gc.draw_calibrator_sample(gens, seed=1, calibrator_model="m/cal", per_model=2)["rows"]
    with pytest.raises(gc.Refused, match="the calibrator's own model is not among the worker runs"):
        gc.draw_calibrator_sample(gens, seed=1, calibrator_model="m/mistyped", per_model=2)


def test_a_model_case_with_no_gradeable_epoch_loses_its_row_with_no_refill_and_no_rng_call():
    models = ("m/one", "m/two", "m/cal")
    differs = 0
    for seed in range(50):
        ids = sorted(f"c{i:03d}" for i in range(20))
        random.Random(seed).shuffle(ids)
        lost_key = ("m/one", ids[1])
        gens = _synthetic(models, 20, unparsed={lost_key})
        gc.assign_unit_ids(gens, 5)
        draw = gc.draw_calibrator_sample(gens, seed=seed, calibrator_model="m/cal", per_model=5)
        _, rows, lost = _replica_draw(gens, seed, "m/cal", 5)
        assert draw["rows"] == rows
        assert draw["lost"] == lost == [{"model": "m/one", "sample_id": ids[1], "position": 2}]
        assert len(draw["rows"]) == 9                                   # no refill
        # the variant that draws an epoch for the lost row too
        rng = random.Random(seed)
        rng.shuffle(sorted(f"c{i:03d}" for i in range(20)))
        wrong = []
        for i, m in enumerate(("m/one", "m/two")):
            for j in range(5):
                case = ids[(i * 5 + j) % 20]
                e = rng.choice([1, 2, 3, 4])
                if (m, case) != lost_key:
                    wrong.append((m, case, e))
        differs += [(r["model"], r["sample_id"], r["epoch"]) for r in draw["rows"]] != wrong
    assert differs > 0                                                  # the rng call would show
    with pytest.raises(gc.Refused):
        gc.draw_calibrator_sample(_synthetic(models, 3), seed=1, calibrator_model="m/cal", per_model=1)  # no uid


def test_draw_is_fixed_before_any_grader_output(wl, monkeypatch):
    from icfr_agent_evals import grader

    def no_call(*a, **k):
        raise AssertionError("a grader or eval call was made while planning")

    monkeypatch.setattr(grader, "grade", no_call)
    monkeypatch.setattr("inspect_ai.eval", no_call)
    again = gc.build_plan(wl.logs, gf.dev_cases(), calibrator_model=gf.DELTA, per_model=20)
    assert again == wl.plan
    assert json.dumps(again, sort_keys=True) == json.dumps(wl.plan, sort_keys=True)
    shas = {w.model: w.sha256 for w in wl.logs}
    assert again["seeds"] == gc.plan_seeds(shas)
    assert again["seed_material_shas"] == [shas[m] for m in sorted(shas)]
    assert set(again) == {"generations", "units", "parts", "draw", "seeds", "seed_material_shas", "counts"}
    assert list(again["parts"]) == [*gc.GRADER_PART_KEYS, "calibrator"]


def test_calibrator_order_is_the_seeded_shuffle_of_sorted_drawn_ids(wl):
    rows = [{"uid": u} for u in ("u0300", "u0002", "u0150", "u0077")]
    want = ["u0002", "u0077", "u0150", "u0300"]
    random.Random(31).shuffle(want)
    assert gc.calibrator_order(rows, 31) == want
    plan = wl.plan
    want = sorted(r["uid"] for r in plan["draw"]["rows"])
    random.Random(plan["seeds"]["calibrator_order"]).shuffle(want)
    assert plan["parts"]["calibrator"] == want
    grader_uids = sum((plan["parts"][k] for k in gc.GRADER_PART_KEYS), [])
    assert grader_uids == sorted(plan["units"])


def test_unit_messages_follow_the_acceptance_look_recipe_and_cross_check_the_case(wl):
    uid = sorted(wl.plan["units"])[0]
    unit = wl.plan["units"][uid]
    cases = gf.dev_cases()
    got = gc.unit_messages(unit, wl.by_model, cases)
    s = wl.by_model[unit["model"]].by_key[(unit["sample_id"], unit["epoch"])]
    case = cases[unit["sample_id"]]
    parse = scoring.parse_worker_output(s.output.completion)
    want = gp.build_messages(str(parse.parsed["narrative"]), case.input, rf.case_facts(case.metadata),
                             rf.output_facts(case.metadata, parse.parsed), gp.PromptSettings())
    assert got == want
    assert unit["messages_sha256"] == gc.messages_sha256(want)
    assert gf.unmark("\n".join(m["content"] for m in got)) == (unit["model"], unit["sample_id"], unit["epoch"])
    changed = dict(cases)
    other = copy.deepcopy(case)
    other.input = case.input + " changed"
    changed[unit["sample_id"]] = other
    with pytest.raises(gc.Refused):
        gc.unit_messages(unit, wl.by_model, changed)
    m, ci, ep = gf.UNPARSED_EPOCH
    bad_unit = {**unit, "model": m, "sample_id": gf.pilot_ids()[ci], "epoch": ep}
    with pytest.raises(gc.Refused):
        gc.unit_messages(bad_unit, wl.by_model, cases)


def test_sealed_plan_bodies(wl):
    bodies = gc.sealed_plan_bodies(wl.plan, wl.logs)
    assert set(bodies) == set(gc.PLAN_FILES)
    um = bodies["unit-map.json"]
    assert um["run"] == gc.RUN_LABEL and um["units"] == wl.plan["units"]
    assert {w["model"] for w in um["worker_logs"]} == set(gf.MODELS)
    assert bodies["draw-pool.json"]["pool"] == wl.plan["draw"]["pool"]
    assert bodies["draw-pool.json"]["seed"] == wl.plan["seeds"]["calibrator_draw"]
    assert bodies["drawn.json"]["calibrator_order"] == wl.plan["parts"]["calibrator"]
    assert bodies["drawn.json"]["lost"] == wl.plan["draw"]["lost"]
    for body in bodies.values():
        json.dumps(body)
    assert bodies["draw-pool.json"]["rule"] == gc.DRAW_RULE_TEXT


def test_sealed_plan_hashes_are_the_hashes_of_the_written_plan_files(wl, tmp_path):
    hashes = gc.sealed_plan_hashes(wl.plan, wl.logs)
    assert list(hashes) == list(gc.PLAN_FILES)
    assert not list(tmp_path.iterdir())                                 # nothing written
    folder = tmp_path / "_sealed"
    bodies = gc.sealed_plan_bodies(wl.plan, wl.logs)
    for name in gc.PLAN_FILES:
        assert gc.write_sealed(folder, name, bodies[name]) == hashes[name]
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == hashes[name]


def test_the_plan_counts_the_characters_of_every_grader_prompt(wl):
    counts = wl.plan["counts"]
    want = sum(len(m["content"]) for unit in wl.plan["units"].values()
               for m in gc.unit_messages(unit, wl.by_model, gf.dev_cases()))
    assert counts["grader_prompt_chars"] == want > 0
    assert set(counts) == {"generations", "skipped_errored", "skipped_unparsed", "units", "rows_drawn", "rows_lost",
                           "grader_prompt_chars"}


def test_a_defect_epoch_reaches_the_unit_and_the_defect_dropped_row(wl, monkeypatch):
    """A planted defect epoch (through trials.defect_epochs) marks its generation and unit,
    and the defect-dropped mean leaves out exactly that one value."""
    sid, epoch = gf.pilot_ids()[3], 2
    real = gc.trials.defect_epochs

    def planted(log, *a, **k):
        if str(log.eval.model) == gf.ALPHA:
            return {(sid, epoch): ("route",)}
        return {}

    monkeypatch.setattr(gc.trials, "defect_epochs", planted)
    plan = gc.build_plan(wl.logs, gf.dev_cases(), calibrator_model=gf.DELTA, per_model=20)
    monkeypatch.setattr(gc.trials, "defect_epochs", real)
    flagged = [g for g in plan["generations"] if g["defect_epoch"]]
    assert [(g["model"], g["sample_id"], g["epoch"]) for g in flagged] == [(gf.ALPHA, sid, epoch)]
    assert flagged[0]["status"] == gc.GEN_GRADEABLE
    assert plan["units"][flagged[0]["uid"]]["defect_epoch"] is True
    assert sum(1 for u in plan["units"].values() if u["defect_epoch"]) == 1
    gs = gf.load_script("grading_stats")
    replies = {uid: gf.parsed(1, 2) for uid in plan["units"]}
    means = gs.llm_means(plan["generations"], replies)
    for d in (G, C):
        block = means[gf.ALPHA]["all"][d]
        assert block["primary_defect_dropped"]["values"] == block["primary"]["values"] - 1
        assert block["sensitivity_defect_dropped"]["values"] == block["sensitivity"]["values"] - 1
        other = means[gf.BRAVO]["all"][d]
        assert other["primary_defect_dropped"]["values"] == other["primary"]["values"]


# --- Calls in a log ---------------------------------------------------------------------------

DONE_AT = datetime.datetime(2026, 11, 23, tzinfo=datetime.timezone.utc)


def _ev(role="calibrator", error=None, text="ok", *, completed=DONE_AT, pending=None):
    # a completed call has a completed time and is not pending (Inspect sets both)
    return SimpleNamespace(event="model", role=role, error=error, model="mockllm/model", cache=None, config={},
                           completed=completed, pending=pending,
                           output=SimpleNamespace(completion=text, stop_reason="stop", model="m", usage=None))


def _calls_log(samples, status="success"):
    return SimpleNamespace(status=status, eval=None, location=None,
                           samples=[SimpleNamespace(id=i, epoch=1, error=None, output=None, events=list(evs))
                                    for i, evs in samples])


def test_role_calls_skips_an_errored_call_followed_by_a_completed_one():
    done = _ev()
    log = _calls_log([("u0001", [_ev(error="rate limit"), done]), ("u0002", [_ev(role="grader"), _ev()])])
    out = gc.role_calls(log, "calibrator", ["u0001", "u0002"], require_all=True)
    assert out["skipped"] == 1 and out["missing"] == []
    assert out["completed"]["u0001"] is done and set(out["samples"]) == {"u0001", "u0002"}
    view = {str(s.id): s for s in out["view"].samples}
    assert len([e for e in view["u0001"].events if e.role == "calibrator"]) == 1
    assert out["view"].status == "success"


@pytest.mark.parametrize("events", [
    [_ev(), _ev()],                       # two completed calls
    [_ev(), _ev(error="boom")],           # an error after the completed call
])
def test_role_calls_refuses_two_completed_calls_or_an_error_after_completion(events):
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(_calls_log([("u0001", events)]), "calibrator", ["u0001"], require_all=False)


def test_role_calls_refuses_a_sample_not_asked_or_listed_twice():
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(_calls_log([("u0009", [_ev()])]), "calibrator", ["u0001"], require_all=False)
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(_calls_log([("u0001", [_ev()]), ("u0001", [_ev()])]), "calibrator", ["u0001"],
                      require_all=False)
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(SimpleNamespace(status="success", samples=None), "calibrator", ["u0001"], require_all=False)


def test_role_calls_partial_log_lists_missing_units():
    log = _calls_log([("u0001", [_ev()]), ("u0002", [_ev(error="died")])], status="error")
    out = gc.role_calls(log, "calibrator", ["u0001", "u0002", "u0003"], require_all=False)
    assert set(out["completed"]) == {"u0001"} and sorted(out["missing"]) == ["u0002", "u0003"]
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(log, "calibrator", ["u0001", "u0002", "u0003"], require_all=True)


def test_a_pending_call_or_one_with_no_completed_time_is_not_a_completed_call():
    assert gc.is_completed_call(_ev())
    assert not gc.is_completed_call(_ev(error="boom"))
    assert not gc.is_completed_call(_ev(pending=True))
    assert not gc.is_completed_call(_ev(completed=None))
    assert not gc.is_completed_call(SimpleNamespace(event="model"))
    log = _calls_log([("u0001", [_ev(pending=True, completed=None)]), ("u0002", [_ev(completed=None)]),
                      ("u0003", [_ev()])], status="error")
    out = gc.role_calls(log, "calibrator", ["u0001", "u0002", "u0003"], require_all=False)
    assert out["missing"] == ["u0001", "u0002"] and set(out["completed"]) == {"u0003"}
    assert out["skipped"] == 0
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(log, "calibrator", ["u0001", "u0002", "u0003"], require_all=True)


def test_an_errored_then_a_pending_then_a_completed_call_keeps_the_completed_one():
    done = _ev(text="kept")
    log = _calls_log([("u0001", [_ev(error="rate limit"), _ev(pending=True, completed=None), done])])
    out = gc.role_calls(log, "calibrator", ["u0001"], require_all=True)
    assert out["completed"]["u0001"] is done and out["skipped"] == 2 and out["missing"] == []
    view = out["view"].samples[0]
    assert [e for e in view.events if e.role == "calibrator"] == [done]
    # a pending call after the completed one is refused, like an errored one
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.role_calls(_calls_log([("u0001", [_ev(), _ev(pending=True, completed=None)])]), "calibrator",
                      ["u0001"], require_all=False)


def test_failed_judgment_record_has_no_reply_and_the_fixed_error():
    got = gc.failed_judgment_record(attempt=3, log_file="iso.eval")
    assert got == {"status": "failed", "scores": {}, "low_confidence": {}, "reasons": {}, "reply": None,
                   "error": gc.FAILED_JUDGMENT_ERROR, "attempt": 3, "log_file": "iso.eval", "usage": None}
    assert set(got) == set(gc.reply_record(SimpleNamespace(output=None), attempt=1, log_file="x.eval"))
    assert got["status"] in gc.REPLY_STATUSES and got["status"] != gc.REPLY_PARSED


def test_reply_record_marks_refusals_and_never_parses_them():
    text = gf.reply(3, 1, low=True, reason="why")
    usage = SimpleNamespace(input_tokens=12, output_tokens=4)
    ok = gc.reply_record(SimpleNamespace(output=SimpleNamespace(completion=text, stop_reason="stop", usage=usage)),
                         attempt=2, log_file="x.eval")
    assert ok["status"] == gc.REPLY_PARSED and ok["scores"] == {G: 3, C: 1}
    assert ok["low_confidence"] == {G: True, C: True} and ok["reasons"] == {G: "why", C: "why"}
    assert ok["reply"] is None and ok["error"] is None and ok["attempt"] == 2 and ok["log_file"] == "x.eval"
    assert list(ok["usage"]) == [12, 4]
    refused = gc.reply_record(SimpleNamespace(output=SimpleNamespace(completion=text, stop_reason="content_filter",
                                                                     usage=None)), attempt=1, log_file="y.eval")
    assert refused["status"] == gc.REPLY_REFUSED and refused["scores"] == {} and refused["low_confidence"] == {}
    assert refused["reasons"] == {} and refused["reply"] == text
    assert refused["error"] == "refused (stop reason content_filter)" and refused["usage"] is None
    bad = gc.reply_record(SimpleNamespace(output=SimpleNamespace(completion="no json", stop_reason="stop",
                                                                 usage=None)), attempt=1, log_file="z.eval")
    assert bad["status"] == gc.REPLY_UNPARSED and bad["scores"] == {} and bad["reply"] == "no json"
    assert bad["error"]
    empty = gc.reply_record(SimpleNamespace(output=SimpleNamespace(completion=None, stop_reason="stop",
                                                                   usage=None)), attempt=1, log_file="z.eval")
    assert empty["status"] == gc.REPLY_UNPARSED and empty["reply"] == ""


def test_check_serving_recorded():
    info = SimpleNamespace(event="info", source="calibrator_serving",
                           data={"serving_endpoint": {"model": roster.CALIBRATOR, "base_url": "https://api.anthropic.com/"}})
    sample = SimpleNamespace(id="u0001", events=[info])
    got = gc.check_serving_recorded(sample, source="calibrator_serving", provider="anthropic")
    assert got["model"] == roster.CALIBRATOR
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.check_serving_recorded(sample, source="grader_serving", provider="anthropic")
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.check_serving_recorded(sample, source="calibrator_serving", provider="openai")
    info.data = {"serving_endpoint": {"model": roster.CALIBRATOR, "base_url": "http://localhost:4000"}}
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.check_serving_recorded(sample, source="calibrator_serving", provider="anthropic")
    with pytest.raises(roster.RunConfigError, match="^run config: "):
        gc.check_serving_recorded(SimpleNamespace(id="u0001", events=[]), source="calibrator_serving",
                                  provider="anthropic")


# --- Sealed files and the ledger --------------------------------------------------------------

def test_sealed_files_are_created_exclusively_and_matched_on_rewrite(tmp_path):
    folder = tmp_path / "g" / "_sealed"
    folder.mkdir(parents=True)
    body = {"run": "x", "rows": [3, 1, 2]}
    sha = gc.write_sealed(folder, "a.json", body)
    path = folder / "a.json"
    assert path.read_bytes() == gc.canonical_bytes(body) and sha == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        gc.write_sealed(folder, "a.json", body)
    assert gc.write_or_match_sealed(folder, "a.json", dict(reversed(list(body.items())))) == sha
    with pytest.raises(gc.Refused, match="never overwritten"):
        gc.write_or_match_sealed(folder, "a.json", {"run": "x", "rows": [3, 1]})
    assert path.read_bytes() == gc.canonical_bytes(body)
    assert gc.write_or_match_sealed(folder, "b.json", {"b": 1}) == gc.sha256_file(folder / "b.json")
    assert gc.sealed_hashes(folder, ["a.json", "b.json"]) == {"a.json": sha, "b.json": gc.sha256_file(folder / "b.json")}
    with pytest.raises(gc.Refused):
        gc.sealed_hashes(folder, ["a.json", "missing.json"])
    assert sorted(p.name for p in folder.iterdir()) == ["a.json", "b.json"]       # no temporary file is left


def test_write_new_file_is_all_or_nothing_and_never_overwrites(tmp_path):
    path = tmp_path / "new" / "deep" / "f.json"
    gc.write_new_file(path, b"one")                                     # creates the parent folder
    assert path.read_bytes() == b"one"
    with pytest.raises(FileExistsError):
        gc.write_new_file(path, b"two")
    assert path.read_bytes() == b"one"
    assert [p.name for p in path.parent.iterdir()] == ["f.json"]        # no .tmp file left, also after a refusal


def test_write_new_file_falls_back_to_exclusive_create_without_hard_links(tmp_path, monkeypatch):
    def no_link(src, dst):
        raise OSError("hard links are not supported")

    monkeypatch.setattr(gc.os, "link", no_link)
    path = tmp_path / "f.bin"
    gc.write_new_file(path, b"data")
    assert path.read_bytes() == b"data"
    with pytest.raises(FileExistsError):
        gc.write_new_file(path, b"other")
    assert path.read_bytes() == b"data" and [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_write_new_file_removes_its_temporary_file_when_the_write_fails(tmp_path, monkeypatch):
    def broken_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr(gc.os, "fsync", broken_fsync)
    with pytest.raises(OSError, match="disk full"):
        gc.write_new_file(tmp_path / "f.bin", b"data")
    assert list(tmp_path.iterdir()) == []


def test_write_or_match_bytes_matches_equal_bytes_and_refuses_others(tmp_path):
    path = tmp_path / "evidence.json"
    sha = gc.write_or_match_bytes(path, b"abc\n", what="evidence file")
    assert sha == hashlib.sha256(b"abc\n").hexdigest() == gc.write_or_match_bytes(path, b"abc\n", what="evidence file")
    with pytest.raises(gc.Refused, match="^the evidence file evidence.json exists and differs from the recomputed one; "
                                         "it is never overwritten$"):
        gc.write_or_match_bytes(path, b"abd\n", what="evidence file")
    assert path.read_bytes() == b"abc\n"


def test_sealed_dir_must_be_named_sealed_and_outside_the_repo(tmp_path):
    repo = gf.make_repo(tmp_path).repo
    ok = tmp_path / "grading" / "_sealed"
    assert gc.check_sealed_dir(ok, repo) == ok.resolve()
    with pytest.raises(gc.Refused):
        gc.check_sealed_dir(tmp_path / "grading" / "sealed", repo)
    with pytest.raises(gc.Refused):
        gc.check_sealed_dir(repo / "x" / "_sealed", repo)
    with pytest.raises(gc.Refused):
        gc.check_sealed_dir(REPO / "x" / "_sealed", repo)


def test_ledger_lives_under_the_git_common_dir(tmp_path):
    env = gf.make_repo(tmp_path)
    assert gc.ledger_path(env.repo) == env.git_common / "icfr-grading" / "grading-2026-11-21.json"
    assert gc.ledger_path(env.repo, tmp_path / "l") == tmp_path / "l" / "grading-2026-11-21.json"
    path = tmp_path / "l" / "grading-2026-11-21.json"
    gc.create_ledger(path, {"a": 1})
    with pytest.raises(FileExistsError):
        gc.create_ledger(path, {"a": 2})
    gc.save_ledger(path, {"a": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 3}
    assert gc.read_ledger(path) == {"a": 3}
    assert gc.ledger_lock_path(path) == tmp_path / "l" / "grading-2026-11-21.lock"


@pytest.mark.parametrize("data, kind", [
    (b"{", "JSONDecodeError"), (b"\xff\xfe{}", "UnicodeDecodeError"), (b"[1, 2]", "list"), (b"3", "int"),
], ids=["truncated", "not-utf8", "list", "number"])
def test_read_ledger_refuses_a_ledger_it_cannot_read_and_never_prints_it(tmp_path, data, kind):
    path = tmp_path / "grading-2026-11-21.json"
    path.write_bytes(data)
    with pytest.raises(gc.Refused) as e:
        gc.read_ledger(path)
    assert str(e.value) == (f"the grading ledger grading-2026-11-21.json cannot be read ({kind}); "
                            f"the author decides how to proceed")


def test_read_ledger_refuses_a_missing_ledger(tmp_path):
    with pytest.raises(gc.Refused, match=r"cannot be read \(FileNotFoundError\)"):
        gc.read_ledger(tmp_path / "grading-2026-11-21.json")
    path = tmp_path / "grading-2026-11-21.json"
    path.write_text('{"secret": "SECRET-REASON-TEXT"', encoding="utf-8")
    with pytest.raises(gc.Refused) as e:
        gc.read_ledger(path)
    assert "SECRET" not in str(e.value)


def test_new_ledger_has_exactly_the_listed_keys(wl, tmp_path):
    prov = {"head": "h", "worker_log_shas": sorted(w.sha256 for w in wl.logs)}
    led = gc.new_ledger(provenance=prov, plan=wl.plan, sealed_plan_hashes={"unit-map.json": "0" * 64},
                        log_dir=tmp_path / "logs", sealed_dir=tmp_path / "_sealed")
    assert set(led) == {"ledger", "run", "job_id", "started", "provenance", "seeds", "seed_rule", "sampling_calls",
                        "sealed_plan_files", "parts", "log_dir", "sealed_dir", "attempts", "attempt_logs",
                        "replies_of_record", "batch_ids", "failures", "rejected_logs"}
    assert led["job_id"] == gc.job_id(prov) == hashlib.sha256(gc.canonical_bytes(prov)).hexdigest()[:16]
    assert led["sampling_calls"] == list(gc.SAMPLING_CALLS) and led["seeds"] == wl.plan["seeds"]
    assert led["attempts"][0]["attempt"] == 1 and led["attempts"][0]["resume"] is False
    assert led["attempt_logs"] == [] and led["replies_of_record"] == {} and led["batch_ids"] == []
    assert led["failures"] == [] and led["rejected_logs"] == []
    assert led["log_dir"] == str((tmp_path / "logs").resolve())
    assert not any(s in json.dumps(led) for s in gf.LEAK_STRINGS)


# --- Pooled counts ----------------------------------------------------------------------------

def test_pooled_counts_hold_no_per_model_figure(wl):
    plan = wl.plan
    uids = sorted(plan["units"])
    grader = {u: (gf.failed() if i % 50 == 0 else gf.parsed(1, 2)) for i, u in enumerate(uids)}
    grader[uids[1]] = gf.failed("refused")
    rows = plan["draw"]["rows"]
    cal = {r["uid"]: gf.parsed(1, 2) for r in rows}
    cal[rows[0]["uid"]] = gf.failed("refused")
    cal[rows[1]["uid"]] = gf.failed()
    ledger = {"attempt_logs": [
        {"attempt": 1, "part": "grader-1", "role": "grader", "log_file": "a.eval", "log_sha256": "0" * 64,
         "status": "success", "asked": [], "completed": 1, "errored_calls_skipped": 2, "found_on_resume": False},
        {"attempt": 1, "part": "grader-2", "role": "grader", "log_file": "b.eval", "log_sha256": "1" * 64,
         "status": "error", "asked": [], "completed": 1, "errored_calls_skipped": 1, "found_on_resume": False},
        {"attempt": 1, "part": "calibrator", "role": "calibrator", "log_file": "c.eval", "log_sha256": "2" * 64,
         "status": "success", "asked": [], "completed": 1, "errored_calls_skipped": 3, "found_on_resume": False}]}
    counts = gc.pooled_counts(plan, grader, cal, ledger)
    assert set(counts) == {"worker_logs", "generations", "skipped_errored", "skipped_unparsed", "units", "grader",
                           "calibrator"}
    assert counts["worker_logs"] == 4 and counts["generations"] == 320 and counts["units"] == 314
    assert counts["skipped_errored"] == 1 and counts["skipped_unparsed"] == 5
    unparsed = sum(1 for r in grader.values() if r["status"] == "unparsed")
    assert counts["grader"] == {"replies_of_record": 314, "parsed": 314 - unparsed - 1, "unparsed": unparsed,
                                "refused": 1, "failed": 0, "errored_calls_skipped": 3, "attempt_logs": 2}
    complete = sum(1 for r in rows if grader[r["uid"]]["status"] == "parsed" and cal[r["uid"]]["status"] == "parsed")
    assert counts["calibrator"] == {"rows_drawn": 59, "rows_lost": 1, "replies_of_record": 59, "parsed": 57,
                                    "unparsed": 1, "refused": 1, "failed": 0, "errored_calls_skipped": 3,
                                    "attempt_logs": 1, "pairs_complete": complete, "pairs_dropped": 59 - complete}
    text = json.dumps(counts)
    assert not any(s in text for s in gf.LEAK_STRINGS) and "pool" not in text


def test_pooled_counts_count_a_failed_judgment_as_failed_and_a_failed_calibrator_reply_as_a_dropped_pair(wl):
    plan = wl.plan
    rows = plan["draw"]["rows"]
    grader = {u: gf.parsed(1, 2) for u in plan["units"]}
    cal = {r["uid"]: gf.parsed(1, 2) for r in rows}
    base = gc.pooled_counts(plan, grader, cal, {"attempt_logs": []})
    assert base["calibrator"]["pairs_complete"] == 59 and base["calibrator"]["pairs_dropped"] == 0
    other = next(u for u in sorted(plan["units"]) if u not in cal)
    grader[other] = gc.failed_judgment_record(attempt=3, log_file="g.eval")
    cal[rows[0]["uid"]] = gc.failed_judgment_record(attempt=3, log_file="c.eval")
    counts = gc.pooled_counts(plan, grader, cal, {"attempt_logs": []})
    assert counts["grader"]["failed"] == 1 and counts["grader"]["parsed"] == 313
    assert counts["grader"]["unparsed"] == counts["grader"]["refused"] == 0
    assert counts["calibrator"]["failed"] == 1 and counts["calibrator"]["parsed"] == 58
    assert counts["calibrator"]["pairs_complete"] == 58 and counts["calibrator"]["pairs_dropped"] == 1
    # a failed grader reply on a drawn unit also drops its pair
    grader[rows[1]["uid"]] = gc.failed_judgment_record(attempt=3, log_file="g.eval")
    counts = gc.pooled_counts(plan, grader, cal, {"attempt_logs": []})
    assert counts["grader"]["failed"] == 2 and counts["calibrator"]["pairs_dropped"] == 2


# --- Start checks -----------------------------------------------------------------------------

@pytest.fixture
def repo_env(tmp_path):
    return gf.make_repo(tmp_path)


def _start(env, **kw):
    args = dict(run_commit=env.run_commit, look_file=env.look_file, archive_file=env.archive_file,
                folders={"--log-dir": env.repo.parent / "logs"}, running_package=False)
    args.update(kw)
    return gc.start_checks(env.repo, **args)


def test_start_checks_pass_on_a_clean_archived_head(repo_env):
    state = _start(repo_env)
    assert state["head"] == repo_env.head and state["origin_main"] == repo_env.head
    assert state["run_commit"] == repo_env.run_commit
    assert state["look"]["verdict"] == "PASS" and state["look"]["file"] == "docs/evidence/calibration/acceptance-look-step-2.json"
    assert state["look"]["sha256"] == hashlib.sha256(repo_env.look_file.read_bytes()).hexdigest()
    assert state["archive"]["commit"] == repo_env.run_commit and state["archive"]["date"] == "2026-11-22"
    assert state["archive"]["url"].startswith("https://web.archive.org/web/")
    assert _start(repo_env, look_file=None)["look"] is None


def test_start_checks_refuse_a_dirty_tree(repo_env):
    (repo_env.repo / "scripts" / "new.py").write_text("y = 2\n", encoding="utf-8")
    with pytest.raises(gc.Refused):
        _start(repo_env)
    (repo_env.repo / "scripts" / "new.py").unlink()
    (repo_env.repo / "scripts" / "x.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(gc.Refused):
        _start(repo_env)


def test_start_checks_refuse_a_head_not_on_origin_main(repo_env):
    (repo_env.repo / "README").write_text("r\n", encoding="utf-8", newline="\n")
    gf.commit_all(repo_env.repo, "ahead of origin/main")
    with pytest.raises(gc.Refused):
        _start(repo_env)
    gf.git(repo_env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _start(repo_env)


def test_start_checks_refuse_a_package_change_since_the_run_commit(repo_env):
    (repo_env.repo / "icfr_agent_evals" / "__init__.py").write_text("v = 2\n", encoding="utf-8", newline="\n")
    gf.commit_all(repo_env.repo, "package change")
    gf.git(repo_env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(gc.Refused, match="package"):
        _start(repo_env)


@pytest.mark.parametrize("change", [
    {"verdict": {"overall": "FAIL"}},
    {"grader_step": 1},
    {"grader_model": "openai/gpt-6-luna"},
    {"grader_generate_config": {"reasoning_effort": "high"}},
    {"offline": True},
    {"grader_prompt_version": "gp-0"},
    {"grader_prompt_sha256": "f" * 64},
    {"anchors_sha256": "e" * 64},
    {"passes": [{"name": "pass_2", "settings": {}}]},
], ids=["verdict", "step", "model", "config", "offline", "prompt-version", "prompt-sha", "anchors", "pass-1"])
def test_start_checks_refuse_a_look_file_that_is_not_a_step_2_pass(tmp_path, change):
    env = gf.make_repo(tmp_path, look=gf.look_body(**change))
    with pytest.raises(gc.Refused):
        gc.look_preconditions(env.look_file, env.repo)
    with pytest.raises(gc.Refused):
        _start(env)


def test_the_look_file_must_be_committed_and_unchanged(repo_env):
    gc.look_preconditions(repo_env.look_file, repo_env.repo)
    repo_env.look_file.write_text(json.dumps(gf.look_body(), indent=2), encoding="utf-8")
    with pytest.raises(gc.Refused):
        gc.look_preconditions(repo_env.look_file, repo_env.repo)
    other = repo_env.repo.parent / "outside-look.json"
    other.write_text(json.dumps(gf.look_body()), encoding="utf-8")
    with pytest.raises(gc.Refused):
        gc.look_preconditions(other, repo_env.repo)


def test_the_committed_step_2_look_file_meets_the_preconditions():
    got = gc.look_preconditions(gc.LOOK_FILE, REPO)
    assert got["verdict"] == "PASS" and got["file"] == "docs/evidence/calibration/acceptance-look-step-2.json"


def _rewrite_archive(env, body):
    env.archive_file.write_text(json.dumps(body), encoding="utf-8", newline="\n")
    gf.commit_all(env.repo, "archive record changed")
    gf.git(env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")


def test_start_checks_refuse_a_missing_or_stale_archive_record(tmp_path):
    env = gf.make_repo(tmp_path / "a")
    with pytest.raises(gc.Refused):
        _start(env, archive_file=env.repo / "docs" / "evidence" / "grading" / "missing.json")
    good = {"commit": env.run_commit, "url": "https://web.archive.org/web/20261122/x", "date": "2026-11-22"}
    for bad in ({**good, "commit": "f" * 40}, {**good, "url": "https://example.com/x"}, {**good, "date": "22-11-2026"},
                {**good, "commit": "abc"}):
        env = gf.make_repo(tmp_path / f"b{len(list(tmp_path.iterdir()))}")
        _rewrite_archive(env, bad)
        with pytest.raises(gc.Refused):
            gc.archive_record(env.archive_file, env.repo)
    # stale: scripts changed after the archived commit
    env = gf.make_repo(tmp_path / "c")
    (env.repo / "scripts" / "x.py").write_text("x = 3\n", encoding="utf-8", newline="\n")
    gf.commit_all(env.repo, "scripts change after the archive")
    gf.git(env.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(gc.Refused):
        _start(env)
    # an extra key is kept
    env = gf.make_repo(tmp_path / "d")
    _rewrite_archive(env, {**good, "note": "kept"})
    assert gc.archive_record(env.archive_file, env.repo)["commit"] == env.run_commit
    # an uncommitted archive record is refused
    env.archive_file.write_text(json.dumps({**good, "note": "changed"}), encoding="utf-8")
    with pytest.raises(gc.Refused):
        gc.archive_record(env.archive_file, env.repo)


def test_start_checks_refuse_folders_inside_the_repo(repo_env):
    with pytest.raises(gc.Refused):
        _start(repo_env, folders={"--log-dir": repo_env.repo / "logs"})
    with pytest.raises(gc.Refused):
        _start(repo_env, folders={"--sealed-dir": REPO / "x" / "_sealed"})
    _start(repo_env, folders={"--log-dir": repo_env.repo.parent / "logs",
                              "--sealed-dir": repo_env.repo.parent / "g" / "_sealed"})


def test_start_checks_pass_with_no_folders(repo_env):
    args = dict(run_commit=repo_env.run_commit, look_file=repo_env.look_file, archive_file=repo_env.archive_file,
                running_package=False)
    assert gc.start_checks(repo_env.repo, **args)["head"] == repo_env.head
    assert gc.start_checks(repo_env.repo, folders=None, **args)["head"] == repo_env.head


# --- Archive record hashes --------------------------------------------------------------------

def test_archive_record_returns_the_worker_log_and_plan_file_hashes_or_none(tmp_path):
    env = gf.make_repo(tmp_path / "a")
    got = gc.archive_record(env.archive_file, env.repo)
    assert set(got) == {"file", "sha256", "commit", "url", "date", "worker_log_shas", "plan_file_shas"}
    assert got["worker_log_shas"] is None and got["plan_file_shas"] is None
    assert _start(env)["archive"] == got
    good = {"commit": env.run_commit, "url": "https://web.archive.org/web/20261122/x", "date": "2026-11-22"}
    shas = sorted(hashlib.sha256(m.encode("utf-8")).hexdigest() for m in gf.MODELS)
    plan_shas = {name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in gc.PLAN_FILES}
    env = gf.make_repo(tmp_path / "b")
    _rewrite_archive(env, {**good, "worker_log_shas": shas, "plan_file_shas": plan_shas})
    got = gc.archive_record(env.archive_file, env.repo)
    assert got["worker_log_shas"] == shas and got["plan_file_shas"] == plan_shas
    assert _start(env)["archive"] == got


@pytest.mark.parametrize("extra, says", [
    ({"worker_log_shas": "not a list"}, "worker_log_shas"),
    ({"worker_log_shas": ["a" * 64, "b" * 63]}, "worker_log_shas"),
    ({"worker_log_shas": ["b" * 64, "a" * 64]}, "worker_log_shas"),                 # not sorted
    ({"worker_log_shas": ["a" * 64, "a" * 64]}, "worker_log_shas"),                 # a duplicate
    ({"worker_log_shas": ["A" * 64]}, "worker_log_shas"),                           # upper case
    ({"plan_file_shas": ["a" * 64]}, "plan_file_shas"),
    ({"plan_file_shas": {"unit-map.json": "a" * 64, "draw-pool.json": "b" * 64}}, "plan_file_shas"),
    ({"plan_file_shas": {"unit-map.json": "a" * 64, "draw-pool.json": "b" * 64, "drawn.json": "c" * 64,
                         "other.json": "d" * 64}}, "plan_file_shas"),
    ({"plan_file_shas": {"unit-map.json": "a" * 64, "draw-pool.json": "b" * 64, "drawn.json": 3}},
     "plan_file_shas"),
], ids=["wl-not-list", "wl-short", "wl-unsorted", "wl-duplicate", "wl-upper", "pf-not-dict", "pf-missing",
        "pf-extra", "pf-not-hex"])
def test_archive_record_refuses_badly_shaped_hashes(tmp_path, extra, says):
    env = gf.make_repo(tmp_path)
    good = {"commit": env.run_commit, "url": "https://web.archive.org/web/20261122/x", "date": "2026-11-22"}
    _rewrite_archive(env, {**good, **extra})
    with pytest.raises(gc.Refused, match=says):
        gc.archive_record(env.archive_file, env.repo)


# --- Job folders ------------------------------------------------------------------------------

def _folders(root, **kw):
    args = dict(log_dir=root / "_sealed" / "logs", sealed_dir=root / "_sealed", worker_logs=root / "workers",
                fresh=True, plumbing_log_dir=root / "plumbing-logs")
    args.update(kw)
    return gc.check_job_folders(**args)


def test_job_folders_pass_on_the_suggested_layout(tmp_path):
    _folders(tmp_path)                                                  # nothing exists yet
    (tmp_path / "_sealed" / "logs").mkdir(parents=True)
    (tmp_path / "_sealed" / "logs" / "notes.txt").write_text("x", encoding="utf-8")
    _folders(tmp_path)
    _folders(tmp_path, worker_logs=None, plumbing_log_dir=None)
    _folders(tmp_path, log_dir=tmp_path / "_sealed" / "logs" / "deeper")


@pytest.mark.parametrize("change, says", [
    (lambda r: {"log_dir": r / "logs"}, "--log-dir must sit inside the sealed folder"),
    (lambda r: {"log_dir": r / "_sealed"}, "not the sealed folder itself"),
    (lambda r: {"worker_logs": r}, "the worker log folder"),                                 # contains both
    (lambda r: {"worker_logs": r / "_sealed" / "logs"}, "the worker log folder"),           # equals the log folder
    (lambda r: {"worker_logs": r / "_sealed" / "logs" / "w"}, "the worker log folder"),     # inside the log folder
    (lambda r: {"worker_logs": r / "w", "log_dir": r / "w" / "_sealed" / "logs",
                "sealed_dir": r / "s" / "_sealed"}, "the worker log folder"),              # log folder inside
    (lambda r: {"worker_logs": r / "w", "sealed_dir": r / "w" / "_sealed"}, "the worker log folder"),
    (lambda r: {"plumbing_log_dir": r / "_sealed" / "logs"}, "the plumbing check's log folder"),
], ids=["no-sealed-component", "is-the-sealed-folder", "worker-contains", "worker-equals", "worker-inside",
        "log-inside-worker", "sealed-inside-worker", "plumbing-folder"])
def test_job_folders_refuse_each_broken_rule(tmp_path, change, says):
    with pytest.raises(gc.Refused) as e:
        _folders(tmp_path, **change(tmp_path))
    assert says in str(e.value)
    assert str(tmp_path) not in str(e.value)                            # no path is printed


def test_a_fresh_start_needs_an_empty_log_folder_but_a_resume_does_not(tmp_path):
    logs = tmp_path / "_sealed" / "logs"
    logs.mkdir(parents=True)
    (logs / "x.eval").write_bytes(b"x")
    with pytest.raises(gc.Refused, match="already holds .eval files; a fresh start needs an empty log folder"):
        _folders(tmp_path)
    _folders(tmp_path, fresh=False)
    # the plumbing check's folder: refused at a fresh start (the .eval rule fires first when it holds a log)
    with pytest.raises(gc.Refused, match="already holds .eval files"):
        _folders(tmp_path, plumbing_log_dir=logs)
    (logs / "x.eval").unlink()
    with pytest.raises(gc.Refused, match="the plumbing check's log folder; give each its own folder"):
        _folders(tmp_path, plumbing_log_dir=logs)
    _folders(tmp_path, plumbing_log_dir=logs, fresh=False)
    (logs / "sub").mkdir()
    (logs / "sub" / "y.eval").write_bytes(b"y")                         # only top-level .eval files count
    _folders(tmp_path)
