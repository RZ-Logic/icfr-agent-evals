"""Shared builders for the grading job tests (not a test module).

Keyless and offline. One small mockllm run of the 20 dev pilot cases gives real worker log
headers, real samples and real scores. It is copied into four mock worker logs whose headers
claim the evaluation split, each with its own stand-in model name. The sample ids stay the dev
ids, so the job's case cross-check runs against build_dataset(split="dev"). No evaluation case
is built, read or scored, no API key is read and no model is called.

Each worker narrative carries a marker that names its stand-in worker, case and epoch, so a
mockllm judge can answer each grading prompt from a plan the test fixes. The marker never
names a model.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from icfr_agent_evals import grader, roster, rubric
from icfr_agent_evals import grader_prompt as gp
from icfr_agent_evals.casefile import truth_of

REPO = Path(__file__).resolve().parents[1]
G, C = gp.DIMENSIONS
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}

# Stand-in worker model names: none is a substring of a roster pin, so a search for them in
# printed or committed text finds only a leak.
ALPHA, BRAVO, CHARLIE, DELTA = MODELS = ("mockllm/wkr-alpha", "mockllm/wkr-bravo",
                                         "mockllm/wkr-charlie", "mockllm/wkr-delta")
CALIBRATOR_STANDIN = DELTA          # plays the Opus 5.5 run: never in the calibrator sample
FOLDERS = tuple(m.split("/", 1)[1] for m in MODELS)
LEAK_STRINGS = (*MODELS, *FOLDERS)
EPOCHS = 4

MARK_RE = re.compile(r"QQMK-(\d)-(\d\d)-(\d)-QQ")


def load_script(name: str):
    """A sibling script under its own module name, shared with the scripts' own loader."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Cases ------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def dev_cases() -> dict[str, Any]:
    from icfr_agent_evals.dataset import build_dataset

    return {str(s.id): s for s in build_dataset(split="dev")}


@functools.lru_cache(maxsize=1)
def pilot_ids() -> tuple[str, ...]:
    from icfr_agent_evals.dataset import build_dataset

    return tuple(sorted(str(s.id) for s in build_dataset(split="dev", subset="pilot")))


def composition() -> dict[str, int]:
    cases = dev_cases()
    return dict(Counter(cases[i].metadata["truth"]["case_kind"] for i in pilot_ids()))


def case_index(sample_id: str) -> int:
    return pilot_ids().index(sample_id)


# Planted worker generations (model, case index, epoch), fixed by index into pilot_ids().
UNPARSED_EPOCH = (ALPHA, 0, 2)          # one unparsed epoch in a sampled model
ERRORED_EPOCH = (ALPHA, 1, 3)           # one errored epoch whose completion parses (D-R23, units)
NO_GRADEABLE = (BRAVO, 2)               # a (model, case) with no gradeable epoch: all 4 unparsed
EXPECTED = {"generations": len(MODELS) * 20 * EPOCHS, "skipped_errored": 1, "skipped_unparsed": 1 + EPOCHS}
EXPECTED["units"] = EXPECTED["generations"] - EXPECTED["skipped_errored"] - EXPECTED["skipped_unparsed"]


def marker(model: str, ci: int, epoch: int) -> str:
    return f"QQMK-{MODELS.index(model)}-{ci:02d}-{epoch}-QQ"


def unmark(text: str) -> tuple[str, str, int] | None:
    m = MARK_RE.search(text)
    if m is None:
        return None
    return MODELS[int(m.group(1))], pilot_ids()[int(m.group(2))], int(m.group(3))


def answer(truth, narrative: str) -> str:
    items = truth.eligible_items()
    return json.dumps({
        "asset": truth.asset, "period": truth.period, "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction, "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in items)],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
        "narrative": narrative,
    })


def narrative(model: str, ci: int, epoch: int) -> str:
    return (f"Reserves differ from ledger liabilities at the cutoff. Review note {marker(model, ci, epoch)} "
            f"records this epoch.")


# --- Base runs --------------------------------------------------------------------------------

def _worker_model():
    """A mockllm worker that answers each case with a parseable output built from its key."""
    from inspect_ai.model import ModelOutput, get_model

    by_input = {s.input: truth_of(s.metadata) for s in dev_cases().values() if isinstance(s.input, str)}

    def out(input, tools, tool_choice, config):
        text = next((str(m.text) for m in input if m.role == "user"), "")
        truth = by_input.get(text)
        content = answer(truth, "Placeholder narrative.") if truth is not None else "no case"
        return ModelOutput.from_content("mockllm/model", content)

    return get_model("mockllm/model", custom_outputs=out, memoize=False)


def base_runs(folder: Path) -> tuple[Any, Any]:
    """(parsed, garbage): the pilot at 4 epochs with parseable outputs, and at 1 epoch with
    mockllm's default text, which every scorer records as a format break."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import get_model

    from icfr_agent_evals.task import variance_commentary

    good = inspect_eval(variance_commentary(split="dev", subset="pilot", epochs=EPOCHS), model=_worker_model(),
                        model_roles=ROLES, log_dir=str(folder / "good"), display="none")[0]
    assert good.status == "success", good.error
    bad = inspect_eval(variance_commentary(split="dev", subset="pilot", epochs=1), model=get_model("mockllm/model"),
                       model_roles=ROLES, log_dir=str(folder / "bad"), display="none")[0]
    assert bad.status == "success", bad.error
    return read_eval_log(good.location), read_eval_log(bad.location)


def worker_log(good: Any, bad: Any, model: str, *, run_commit: str) -> Any:
    """One mock worker log for a stand-in model, with the planted generations."""
    from inspect_ai.log import EvalError, EvalRevision
    from inspect_ai.model import ModelOutput

    log = good.model_copy(deep=True)
    log.eval.model = model
    log.eval.eval_id = uuid.uuid4().hex[:22]
    log.eval.run_id = uuid.uuid4().hex[:22]
    log.eval.task_args = {**(log.eval.task_args or {}), "split": "eval", "allow_evaluation_split": True,
                          "grader_step": 1, "epochs": EPOCHS}
    log.eval.metadata = {**(log.eval.metadata or {}), "split": "eval"}
    log.eval.revision = EvalRevision(type="git", origin="https://example.invalid/repo.git", commit=run_commit,
                                     dirty=False)
    garbage = {str(s.id): s for s in bad.samples}
    cases = dev_cases()
    samples = []
    for s in good.samples:
        sid, ep = str(s.id), int(s.epoch)
        ci = case_index(sid)
        if (model, ci, ep) == UNPARSED_EPOCH or (model, ci) == NO_GRADEABLE:
            new = garbage[sid].model_copy(deep=True)
            new.epoch, new.uuid = ep, uuid.uuid4().hex
            samples.append(new)
            continue
        new = s.model_copy(deep=True)
        new.uuid = uuid.uuid4().hex
        new.output = ModelOutput.from_content(model=model, content=answer(truth_of(cases[sid].metadata),
                                                                          narrative(model, ci, ep)))
        if (model, ci, ep) == ERRORED_EPOCH:
            new.error = EvalError(message="mock sample error", traceback="", traceback_ansi="")
            new.scores = None
        samples.append(new)
    log.samples = sorted(samples, key=lambda x: (str(x.id), int(x.epoch)))
    return log


def write_worker_logs(good: Any, bad: Any, folder: Path, *, run_commit: str,
                      mutate: Callable[[str, Any], None] | None = None) -> dict[str, Path]:
    """One folder per stand-in model (the part of its name after the slash), one .eval each."""
    from inspect_ai.log import write_eval_log

    out = {}
    for model in MODELS:
        log = worker_log(good, bad, model, run_commit=run_commit)
        if mutate is not None:
            mutate(model, log)
        sub = folder / model.split("/", 1)[1]
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{uuid.uuid4().hex[:12]}.eval"
        write_eval_log(log, str(path))
        out[model] = path
    return out


# --- Judges -----------------------------------------------------------------------------------

def reply(g: int, c: int, *, low: bool = False, reason: str = "r") -> str:
    return json.dumps({G: {"reason": reason, "level": gp.DEFAULT_LABELS[g], "low_confidence": low},
                       C: {"reason": reason, "level": gp.DEFAULT_LABELS[c], "low_confidence": low}})


REFUSE = "REFUSE"
RAISE = "RAISE"                         # a judge call that always raises
SECRET = "SECRET-REASON-TEXT"


def base_scores(model: str, sid: str, epoch: int) -> tuple[int, int]:
    ci = case_index(sid)
    return (ci + epoch) % 4, (2 * ci + MODELS.index(model) + epoch) % 4


def default_plan(role: str, model: str, sid: str, epoch: int):
    """The judges' replies: mostly in agreement, with planted disagreements, low-confidence
    flags, unparsed replies and a calibrator refusal."""
    ci = case_index(sid)
    g, c = base_scores(model, sid, epoch)
    if role == "grader":
        if (model, ci) == (CHARLIE, 3):
            return "no json here"                       # every epoch of this (model, case): unparsed
        return g, c, (ci == 4 and epoch == 1)          # low confidence on one case's first epoch
    if (model, ci) == (BRAVO, 4):
        return REFUSE
    if (model, ci) == (ALPHA, 5):
        return "not a reply either"
    if (model, ci) in ((ALPHA, 6), (CHARLIE, 7)):
        return (g + 2) % 4, c, False                    # a disagreement over 1 point
    return g, c, (model, ci) == (BRAVO, 8)


def judge_roles(plan: Callable = default_plan, calls: Counter | None = None) -> dict[str, Any]:
    """mockllm grader and calibrator whose reply to each grading prompt comes from
    plan(role, model, sample id, epoch) read from the narrative marker: (grounding, coverage,
    low) or a raw string, REFUSE for a content_filter stop, or RAISE for a call that raises
    RuntimeError("mock judge call failed"). calls counts calls per (role, model, sample id,
    epoch), a raising call included."""
    from inspect_ai.model import ModelOutput, get_model

    def make(role):
        def out(input, tools, tool_choice, config):
            text = "\n".join(str(m.text) for m in input)
            key = unmark(text)
            if calls is not None:
                calls[(role, *key) if key else (role, None)] += 1
            got = plan(role, *key) if key else (2, 2, False)
            if got == RAISE:
                raise RuntimeError("mock judge call failed")
            if got == REFUSE:
                return ModelOutput.from_content("mockllm/model", "I will not grade this.", stop_reason="content_filter")
            content = got if isinstance(got, str) else reply(got[0], got[1], low=got[2], reason=SECRET)
            return ModelOutput.from_content("mockllm/model", content)

        return get_model("mockllm/model", custom_outputs=out, memoize=False)

    return {"grader": make("grader"), "calibrator": make("calibrator")}


# --- Fixture repository -----------------------------------------------------------------------

# Fixed commit dates: every scratch repository built from the same files gets the same commit
# hashes, so one set of mock worker logs names the run commit of each of them.
GIT_DATES = {"GIT_AUTHOR_DATE": "2026-11-22T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-11-22T00:00:00+00:00"}


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
                           "-c", "init.defaultBranch=main", *args],
                          capture_output=True, text=True, check=True, env={**os.environ, **GIT_DATES})
    return done.stdout


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def look_body(**changes) -> dict:
    al = load_script("acceptance_look")
    body = {"verdict": {"overall": "PASS"}, "grader_step": 2, "grader_model": grader.grader_pin(2),
            "grader_generate_config": roster.grader_generate_config_args(2), "offline": False,
            "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
            "anchors_sha256": rubric.llm_anchor_sha256(),
            "passes": [{"name": "pass_1", "settings": al.PassSpec("pass_1", 1, gp.PromptSettings()).settings_record()}]}
    body.update(changes)
    return body


def make_repo(root: Path, *, look: dict | None = None) -> SimpleNamespace:
    """A scratch git repository: scripts/x.py, a tiny package, a committed stand-in look file
    (first commit, the run commit), then an archive record naming that commit; the local
    origin/main ref points at HEAD."""
    repo = root / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "x.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    (repo / "icfr_agent_evals").mkdir()
    (repo / "icfr_agent_evals" / "__init__.py").write_text("v = 1\n", encoding="utf-8", newline="\n")
    look_file = repo / "docs" / "evidence" / "calibration" / "acceptance-look-step-2.json"
    look_file.parent.mkdir(parents=True)
    look_file.write_text(json.dumps(look if look is not None else look_body(), indent=1), encoding="utf-8",
                         newline="\n")
    git(repo, "init", "-q")
    run_commit = commit_all(repo, "first")
    archive_file = repo / "docs" / "evidence" / "grading" / "archive-2026-11-21.json"
    archive_file.parent.mkdir(parents=True)
    archive_file.write_text(json.dumps({"commit": run_commit, "url": "https://web.archive.org/web/20261122/x",
                                        "date": "2026-11-22"}), encoding="utf-8", newline="\n")
    head = commit_all(repo, "archive record")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    common = Path(git(repo, "rev-parse", "--absolute-git-dir").strip()).resolve()
    return SimpleNamespace(repo=repo, run_commit=run_commit, head=head, look_file=look_file,
                           archive_file=archive_file, git_common=common)


# --- Small builders ---------------------------------------------------------------------------

def parsed(g: int, c: int, *, low_g: bool = False, low_c: bool = False, attempt: int = 1,
           log_file: str = "a.eval") -> dict:
    return {"status": "parsed", "scores": {G: g, C: c}, "low_confidence": {G: low_g, C: low_c},
            "reasons": {G: "r", C: "r"}, "reply": None, "error": None, "attempt": attempt, "log_file": log_file,
            "usage": [10, 5]}


FAILED_JUDGMENT_ERROR = ("no completed call in 3 attempt logs, the third in an eval of units at their third ask, "
                         "one sample at a time")


def failed(status: str = "unparsed", *, attempt: int = 1, log_file: str = "a.eval") -> dict:
    if status == "failed":
        return {"status": "failed", "scores": {}, "low_confidence": {}, "reasons": {}, "reply": None,
                "error": FAILED_JUDGMENT_ERROR, "attempt": attempt, "log_file": log_file, "usage": None}
    error = "refused (stop reason content_filter)" if status == "refused" else "not one JSON object"
    return {"status": status, "scores": {}, "low_confidence": {}, "reasons": {}, "reply": "raw text",
            "error": error, "attempt": attempt, "log_file": log_file, "usage": None}


def row(uid: str, g: dict, c: dict, *, model: str = ALPHA, sample_id: str | None = None, epoch: int = 1,
        position: int = 1) -> dict:
    return {"uid": uid, "model": model, "sample_id": sample_id or f"s-{uid}", "epoch": epoch,
            "position": position, "grader": g, "calibrator": c}


def by_uid(items) -> dict:
    out = defaultdict(list)
    for x in items:
        out[x["uid"]].append(x)
    return out
