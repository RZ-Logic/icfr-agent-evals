"""The acceptance look: one grader's single scoring of the 40-case calibration subset.

Pre-registration section 7. At the look the grader scores the calibration subset in
seven passes of 40 calls, each pass one Inspect eval through icfr_agent_evals.grader
with the grader role bound to the ladder step's pin and response caching off:

  pass 1   the acceptance-look scoring, default prompt settings
  pass 2   test-retest replicate, default settings, independent
  pass 3   test-retest replicate, default settings, independent
  option_order_reversed      rubric options listed 3, 2, 1, 0
  option_order_permuted      the one random permutation, drawn from the fixed seed
  labels_relabeled           score labels 0 to 3 shown as D, C, B, A
  dimension_order_permuted   the two sub-dimensions listed and answered in swapped order

Seeds are fixed by rule, not chosen. The permutation seed is
casefile.split_seed("calibration"), 23260607, which draws the order 3, 1, 2, 0 (neither
the identity nor the reversal). The alpha unit bootstrap uses the same seed and 10,000
resamples. Both are written into the attempt ledger before pass 1 and into the Inspect
task metadata of every pass, with HEAD and the SHA-256 of calibration.jsonl and of the
record set.

Every log must hold one completed grader call per calibration output and pass
grader.check_grader_log (pin, configuration, resolved model string). A grader call that
ended in an error and was followed by a later completed call on the same output is
skipped and counted; an output with no completed grader call fails the pass. Then
icfr_agent_evals.judge_trust computes:

  validity      ICC(A,1) grader (pass 1) versus the author's grades, per sub-dimension;
                the smaller lower limit of the McGraw-Wong 95% CI must be at least 0.60.
                ICC(C,1) is reported. The kappa gate does not apply (no binary checks).
  test-retest   ordinal Krippendorff alpha over passes 1 to 3 at least 0.80, with its
                unit-bootstrap CI, q and raw agreement; an undefined alpha fails.
  bias audit    each perturbed pass's shift minus the largest pairwise test-retest
                baseline at most 0.10; leniency on pass 1 within 0.25 in magnitude;
                the ICC(C,1) minus ICC(A,1) gap reported.

A reply that does not parse is never asked again. Every gate whose passes all parsed is
computed and stored; a gate that needs a pass with an unparsed reply is marked FAIL with
the reason. The score distribution and the low-confidence count are reported beside them.

One look per grader. Before pass 1 the script creates an attempt ledger with exclusive
create under the git common directory, so every checkout of the repository sees it:
<git common dir>/icfr-acceptance-looks/step-<N>.json. It records the start, HEAD, the
prompt and anchors hashes, the seeds, the input hashes and the log folder, then each
finished pass's log file and SHA-256 and each failure. A fresh start is refused while a
ledger for the step exists. --resume re-checks the passes the ledger holds (log hash and
log checks), keeps them, runs only the remaining passes and lists every attempt and
failure in the look file. The look file, docs/evidence/calibration/acceptance-look-step-
<N>.json, is written with exclusive create after every pass ran and every log passed.

Above step 1 the look runs on the prompt of the look before it, unless the anchors hash
changed (a rubric revision from the intra-rater check), and needs --intra-rater FILE: the
committed intra-rater record, a JSON file tracked in HEAD, whose SHA-256 the look records.

A look that fails at step 1 or 2 is followed by grading that must stay blind to this
grader's scores. Its per-record rows (scores, flags, reasons, unparsed replies) go to a
sealed file in --sealed-dir, outside the repository; the evidence file keeps the
aggregates, every gate value and the sealed file's SHA-256. A look that passes, or fails
at step 3, keeps the rows in the evidence file. A gate FAIL starts the judge remedy order
(step 2: the intra-rater check, then the next grader), which this script does not run.

Calibration split only. The script reads data/calibration/calibration.jsonl and
data/calibration/records/*.json and nothing from the evaluation split. With --live it
refuses unless git status is clean for data/calibration and scripts and both inputs are
tracked in HEAD. The subset is never used for tuning.

Run:
  python scripts/acceptance_look.py --step 1                       (plan only, no call)
  python scripts/acceptance_look.py --step 1 --live --log-dir DIR --sealed-dir SEALED   (paid: 280 calls)
  python scripts/acceptance_look.py --step 1 --live --resume       (continue a started look)
  python scripts/acceptance_look.py --step 2 --intra-rater FILE --live --log-dir DIR --sealed-dir SEALED

Exit codes: 0 plan printed or every gate PASS, 1 a pass or log check failed (recorded in
the ledger, no look file), 2 bad arguments, 3 refused, 4 the look ran and a gate FAILED.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import casefile, grader, roster, rubric, scoring  # noqa: E402
from icfr_agent_evals import grader_prompt as gp  # noqa: E402
from icfr_agent_evals import judge_trust as jt  # noqa: E402
from icfr_agent_evals import review_facts as rf  # noqa: E402


def _load_judge_dev_round():
    name = "judge_dev_round"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "judge_dev_round.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


jr = _load_judge_dev_round()

CALIBRATION_DIR = REPO / "data" / "calibration"
RECORDS_DIR = CALIBRATION_DIR / "records"
CASES_FILE = CALIBRATION_DIR / "calibration.jsonl"
OUT_DIR = REPO / "docs" / "evidence" / "calibration"
JUDGE_DEV_OUT = jr.OUT_DIR
SPLIT = "calibration"
EXPECTED_RECORDS = 40
MIN_PER_LEVEL = 4                      # section 4: every level at least 4 times per sub-dimension

# Fixed by rule (section 7: "drawn from a seed recorded in the run config before the look").
PERMUTATION_SEED = casefile.split_seed(SPLIT)
PERMUTATION_SEED_RULE = "casefile.split_seed('calibration')"
BOOTSTRAP_SEED = casefile.split_seed(SPLIT)
RESAMPLES = jt.DEFAULT_BOOTSTRAP_RESAMPLES
DEFAULT_BOOTSTRAP_SEED = BOOTSTRAP_SEED

SEALED_STEPS: tuple[int, ...] = (1, 2)  # a failed look here is followed by grading blind to its scores
LEDGER_FOLDER = "icfr-acceptance-looks"
EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED, EXIT_GATE_FAIL = 0, 1, 2, 3, 4

RETEST_NAMES: tuple[str, ...] = ("pass_1", "pass_2", "pass_3")
BIAS_NAMES: tuple[str, ...] = (*jt.BIAS_PASSES, jt.DIMENSION_ORDER_PERMUTED)
PASS_NAMES: tuple[str, ...] = (*RETEST_NAMES, *BIAS_NAMES)
GATE_NEEDS: dict[str, tuple[str, ...]] = {
    "validity": ("pass_1",),
    "test_retest": RETEST_NAMES,
    "bias_audit": PASS_NAMES,
}

NEXT_REMEDY: dict[int, str] = {
    1: ("judge remedy order step 2: the intra-rater check first, then the grader moves to GPT-6.1 Terra "
        "on the OpenAI Batch API (pre-registration section 7)"),
    2: ("judge remedy order step 3: the grader moves to GPT-6.1 Sol on the OpenAI Batch API; the "
        "intra-rater check is not run again (pre-registration section 7)"),
    3: ("judge remedy order step 4: each failing sub-dimension is excluded from every reported claim, "
        "with the exclusion disclosed, and the deterministic anchors stand alone (pre-registration section 7)"),
}


class Refused(Exception):
    pass


@dataclass(frozen=True)
class PassSpec:
    """One grader pass over the subset: its name, its place in the look and its prompt settings."""

    name: str
    number: int
    settings: gp.PromptSettings

    def settings_record(self) -> dict[str, Any]:
        return {"option_order": list(self.settings.option_order),
                "labels": {str(k): v for k, v in self.settings.label_map().items()},
                "dimension_order": list(self.settings.dimension_order)}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    return _sha256_bytes(pathlib.Path(path).read_bytes())


# --- Plan ------------------------------------------------------------------------------------

def swapped_dimensions() -> tuple[str, ...]:
    """The dimension order of the dimension pass. Both LLM-graded sub-dimensions share one
    prompt, and with two of them the swap is the one order that differs from the default."""
    if len(gp.DIMENSIONS) != 2:
        raise Refused(f"the dimension pass is built for two shared sub-dimensions, got {len(gp.DIMENSIONS)}")
    return tuple(reversed(gp.DIMENSIONS))


def plan_passes(permutation_seed: int = PERMUTATION_SEED) -> list[PassSpec]:
    """The 7 passes of the look in run order: 3 test-retest passes on the default settings,
    then the bias-audit passes (option order reversed, the one seeded permutation, the labels
    relabeled D, C, B, A, the dimension order swapped). The look itself only ever uses the
    fixed PERMUTATION_SEED; any other seed is refused."""
    if isinstance(permutation_seed, bool) or permutation_seed != PERMUTATION_SEED:
        raise Refused(f"the permutation seed is fixed by rule at {PERMUTATION_SEED} ({PERMUTATION_SEED_RULE}); "
                      f"{permutation_seed!r} is refused")
    default = gp.PromptSettings()
    permuted = tuple(jt.permuted_options(list(gp.LEVELS), permutation_seed)["options"])
    settings = {
        "pass_1": default, "pass_2": default, "pass_3": default,
        jt.OPTION_ORDER_REVERSED: gp.PromptSettings(option_order=tuple(jt.reversed_options(list(gp.LEVELS)))),
        jt.OPTION_ORDER_PERMUTED: gp.PromptSettings(option_order=permuted),
        jt.LABELS_RELABELED: gp.PromptSettings(labels=dict(gp.RELABEL_DCBA)),
        jt.DIMENSION_ORDER_PERMUTED: gp.PromptSettings(dimension_order=swapped_dimensions()),
    }
    specs = [PassSpec(name, i, settings[name]) for i, name in enumerate(PASS_NAMES, start=1)]
    for s in specs:
        s.settings.check()
    return specs


# --- Inputs ----------------------------------------------------------------------------------

def load_cases(path: pathlib.Path = CASES_FILE) -> dict[str, dict]:
    """The calibration cases by id. Every row must be a calibration-split case."""
    if not path.is_file():
        raise Refused(f"{path.name} not found; the calibration cases are built first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {}
    for r in rows:
        if (r.get("metadata") or {}).get("split") != SPLIT:
            raise Refused(f"case {r.get('id')!r} is not a calibration-split case")
        if r["id"] in cases:
            raise Refused(f"case {r['id']!r} is listed twice")
        cases[r["id"]] = r
    return cases


def load_records(folder: pathlib.Path = RECORDS_DIR) -> list[dict]:
    """The author-graded calibration records, sorted by file name: exactly 40, each graded on
    both sub-dimensions against the committed anchors."""
    recs = []
    for p in sorted(folder.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_file"] = p.stem
        recs.append(r)
    if len(recs) != EXPECTED_RECORDS:
        raise Refused(f"{len(recs)} calibration records, not {EXPECTED_RECORDS}")
    for r in recs:
        grades = r.get("grades") or {}
        if set(grades) != set(gp.DIMENSIONS):
            raise Refused(f"{r['_file']}: the author's grades do not cover both sub-dimensions")
        for d in gp.DIMENSIONS:
            if isinstance(grades[d], bool) or grades[d] not in gp.LEVELS:
                raise Refused(f"{r['_file']}: grade {grades[d]!r} on {d} is not a 0 to 3 integer")
        if r.get("anchors_sha256") != rubric.llm_anchor_sha256():
            raise Refused(f"{r['_file']}: graded against anchors {r.get('anchors_sha256')!r}, not the committed "
                          f"{rubric.llm_anchor_sha256()!r}")
    return recs


def records_set_sha256(folder: pathlib.Path) -> str:
    """SHA-256 of the record set: one line "<file name> <SHA-256 of its bytes>" per record
    file, sorted by name, joined with LF."""
    lines = [f"{p.name} {_sha256_file(p)}\n" for p in sorted(folder.glob("*.json"))]
    return _sha256_bytes("".join(lines).encode("utf-8"))


def check_subset(records: Sequence[Mapping[str, Any]], cases: Mapping[str, dict]) -> dict[str, dict[str, int]]:
    """Refuse a subset that is not the 40 calibration cases, one output each, stratified so every
    level 0 to 3 appears at least 4 times per sub-dimension in the author's grades (section 4).
    Returns the author's score distribution per sub-dimension."""
    ids = [r.get("sample_id") for r in records]
    if len(set(ids)) != len(ids):
        raise Refused("a calibration case has more than one record; the subset grades one output per case")
    missing = sorted(set(ids) - set(cases))
    if missing:
        raise Refused(f"records name cases not in the calibration file: {missing[:5]}")
    if set(cases) != set(ids):
        raise Refused(f"calibration cases without a record: {sorted(set(cases) - set(ids))[:5]}")
    dist = {d: jt.score_distribution(r["grades"][d] for r in records) for d in gp.DIMENSIONS}
    short = {d: [lv for lv, n in dist[d].items() if n < MIN_PER_LEVEL] for d in gp.DIMENSIONS}
    if any(short.values()):
        raise Refused(f"the subset is not stratified: levels below {MIN_PER_LEVEL} grades per sub-dimension "
                      f"{ {d: v for d, v in short.items() if v} } (pre-registration section 4)")
    return dist


def record_messages(record: Mapping[str, Any], cases: Mapping[str, dict],
                    settings: gp.PromptSettings = gp.PromptSettings()) -> list[dict]:
    """The grader's messages for one calibration record, built from the same inputs as
    judge_dev_round.record_messages: the narrative, the shared facts sheet and the case file."""
    case = cases.get(record["sample_id"])
    if case is None:
        raise Refused(f"{record['_file']}: case {record['sample_id']} is not in the calibration file")
    if case["metadata"].get("split") != SPLIT:
        raise Refused(f"{record['_file']}: not a calibration-split case")
    parse = scoring.parse_worker_output(record["worker_output"])
    if not parse.ok:
        raise Refused(f"{record['_file']}: the output does not parse (it is never graded, D-R10 C5)")
    meta = case["metadata"]
    return gp.build_messages(str(parse.parsed["narrative"]), case["input"], rf.case_facts(meta),
                             rf.output_facts(meta, parse.parsed), settings)


# --- Repository state ------------------------------------------------------------------------

def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except OSError as e:
        raise Refused(f"git could not run ({e})") from e


def _git_out(repo: pathlib.Path, *args: str) -> str:
    done = _git(repo, *args)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args[:2])} failed: {done.stderr.strip()[:200]}")
    return done.stdout


def _inside(path: Any, root: pathlib.Path) -> bool:
    target = pathlib.Path(path).resolve()
    base = pathlib.Path(root).resolve()
    return target == base or base in target.parents


def _rel(path: pathlib.Path, repo: pathlib.Path) -> str:
    target, base = pathlib.Path(path).resolve(), pathlib.Path(repo).resolve()
    if not _inside(target, base):
        raise Refused(f"{pathlib.Path(path).name} is not inside the repository")
    return target.relative_to(base).as_posix() or "."


def git_common_dir(repo: pathlib.Path = REPO) -> pathlib.Path:
    """The git common directory of the repository: shared by every checkout of it."""
    out = _git_out(repo, "rev-parse", "--git-common-dir").strip()
    path = pathlib.Path(out)
    return (path if path.is_absolute() else pathlib.Path(repo) / path).resolve()


def ledger_path(step: int, repo: pathlib.Path = REPO, ledger_dir: pathlib.Path | None = None) -> pathlib.Path:
    base = pathlib.Path(ledger_dir) if ledger_dir is not None else git_common_dir(repo) / LEDGER_FOLDER
    return base / f"step-{step}.json"


def repo_state(repo: pathlib.Path, cases_file: pathlib.Path, records_dir: pathlib.Path) -> dict[str, Any]:
    """Refuse unless git status is clean for the calibration folder, the records folder and
    scripts, and calibration.jsonl and every record file are tracked in HEAD. Returns HEAD,
    the SHA-256 of calibration.jsonl and of the record set (records_set_sha256)."""
    case_rel, cal_rel, rec_rel = _rel(cases_file, repo), _rel(cases_file.parent, repo), _rel(records_dir, repo)
    status = _git_out(repo, "status", "--porcelain", "--untracked-files=all", "--", cal_rel, rec_rel, "scripts")
    if status.strip():
        first = "; ".join(status.strip().splitlines()[:5])
        raise Refused(f"git status is not clean for {cal_rel} and scripts ({first}); commit before the look")
    tracked = set(_git_out(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", case_rel, rec_rel).splitlines())
    if case_rel not in tracked:
        raise Refused(f"{case_rel} is not tracked in HEAD")
    on_disk = {f"{rec_rel}/{p.name}" for p in records_dir.glob("*.json")}
    in_head = {t for t in tracked if pathlib.PurePosixPath(t).parent.as_posix() == rec_rel and t.endswith(".json")}
    if not on_disk or on_disk != in_head:
        raise Refused(f"the record files are not all tracked in HEAD ({len(on_disk)} on disk, {len(in_head)} in HEAD, "
                      f"{len(on_disk ^ in_head)} differ)")
    return {"head": _git_out(repo, "rev-parse", "HEAD").strip(),
            "calibration_jsonl_sha256": _sha256_file(cases_file),
            "records_sha256": records_set_sha256(records_dir), "records_files": len(on_disk)}


def check_intra_rater(path: str | None, step: int, repo: pathlib.Path = REPO) -> dict[str, Any] | None:
    """Above step 1: the committed intra-rater record, a JSON file inside the repository that
    is tracked in HEAD with no uncommitted change. Returns its path and SHA-256. At step 1
    no record is taken."""
    if step == 1:
        if path:
            raise Refused("--intra-rater applies above step 1 only; the intra-rater check runs after the first failed look")
        return None
    if not path:
        raise Refused(f"step {step} needs --intra-rater FILE, the committed intra-rater record (pre-registration "
                      f"section 7: the intra-rater check runs before the first grader change)")
    p = pathlib.Path(path)
    p = p if p.is_absolute() else pathlib.Path.cwd() / p
    if not p.is_file():
        raise Refused(f"the intra-rater record {p.name} does not exist")
    rel = _rel(p, repo)
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"the intra-rater record {rel} is not a JSON file ({e})") from e
    if rel not in set(_git_out(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", rel).splitlines()):
        raise Refused(f"the intra-rater record {rel} is not tracked in HEAD; commit it before the look")
    if _git_out(repo, "status", "--porcelain", "--", rel).strip():
        raise Refused(f"the intra-rater record {rel} differs from HEAD; commit it before the look")
    body = json.loads(p.read_text(encoding="utf-8"))
    revise = body.get("revise") if isinstance(body, dict) else None
    if revise and body.get("anchors_sha256") == rubric.llm_anchor_sha256():
        raise Refused(f"the intra-rater record {rel} calls for a rubric revision of {', '.join(revise)}, and the "
                      f"anchors have not changed since it; revise the wording and grade the subset again under it "
                      f"before the look (pre-registration section 7)")
    return {"file": rel, "sha256": _sha256_file(p)}


# --- Look order ------------------------------------------------------------------------------

def look_file(out_dir: pathlib.Path, step: int) -> pathlib.Path:
    return out_dir / f"acceptance-look-step-{step}.json"


def sealed_file(sealed_dir: pathlib.Path, step: int) -> pathlib.Path:
    return pathlib.Path(sealed_dir) / f"acceptance-look-step-{step}-rows.json"


def check_look_order(step: int, out_dir: pathlib.Path, judge_dev_dir: pathlib.Path) -> None:
    """One look per grader, in ladder order.

    Refused: a step outside 1 to 3; a step whose look file exists; step 1 before judge remedy
    order step 1 has ended (a live judge-development round met the stop criteria, or round 3
    ran) or on prompt wording other than the one step 1 ended on; step 2 or 3 unless the look
    before it exists and failed, and on a prompt other than that look's unless the anchors
    hash changed since it (a rubric revision from the intra-rater check)."""
    if isinstance(step, bool) or step not in roster.GRADER_STEPS:
        raise Refused(f"--step must be one of {roster.GRADER_STEPS}")
    if look_file(out_dir, step).exists():
        raise Refused(f"{look_file(out_dir, step).name} exists; each grader gets exactly one acceptance look "
                      f"(pre-registration section 7)")
    if step == 1:
        rounds = sorted(judge_dev_dir.glob("round-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        if not rounds:
            raise Refused("no judge-development round has run; step 1 runs before the subset is first scored")
        last = json.loads(rounds[-1].read_text(encoding="utf-8"))
        if last.get("offline"):
            raise Refused(f"{rounds[-1].name} is an offline round, not a grading by the step 1 grader")
        if not last.get("stop_criteria_met") and last.get("round") != jr.MAX_ROUND:
            raise Refused(f"judge remedy order step 1 has not ended ({rounds[-1].name}: stop criteria not met, "
                          f"fewer than {jr.MAX_ROUND} rounds)")
        if last.get("grader_prompt_sha256") != gp.prompt_sha256():
            raise Refused(f"the grader prompt ({gp.prompt_sha256()[:12]}) is not the wording step 1 ended on "
                          f"({str(last.get('grader_prompt_sha256'))[:12]}, {rounds[-1].name})")
        return
    prev = look_file(out_dir, step - 1)
    if not prev.exists():
        raise Refused(f"the step {step - 1} acceptance look has not run; the step {step} grader looks only after "
                      f"a gate fails there")
    body = json.loads(prev.read_text(encoding="utf-8"))
    verdict_ = (body.get("verdict") or {}).get("overall")
    if verdict_ != "FAIL":
        raise Refused(f"the step {step - 1} acceptance look did not fail (verdict {verdict_!r}); "
                      f"the step {step} grader does not grade")
    prev_prompt, prev_anchors = body.get("grader_prompt_sha256"), body.get("anchors_sha256")
    if not prev_prompt or not prev_anchors:
        raise Refused(f"{prev.name} records no prompt or anchors hash, so the prompt rule cannot be checked")
    if prev_anchors == rubric.llm_anchor_sha256() and prev_prompt != gp.prompt_sha256():
        raise Refused(f"the grader prompt ({gp.prompt_sha256()[:12]}) is not the one the step {step - 1} look used "
                      f"({prev_prompt[:12]}), and the anchors did not change; the subset is never used for tuning")


# --- Running a pass --------------------------------------------------------------------------

def pass_task(spec: PassSpec, records: Sequence[Mapping[str, Any]], cases: Mapping[str, dict], *,
              step: int, provenance: Mapping[str, Any] | None = None):
    """An Inspect task for one pass: one sample per record, one grader call each through
    grader.grade. The task metadata records the pass, its settings, the fixed seeds and the
    provenance (HEAD and the input hashes)."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="acceptance_look_grade")
    def acceptance_look_grade():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            state.output = await grader.grade(state.messages, step)
            return state

        return solve

    samples = [Sample(id=r["_file"], input=gp.to_chat_messages(record_messages(r, cases, spec.settings)),
                      metadata={"split": SPLIT}) for r in records]
    metadata = {
        "split": SPLIT, "acceptance_look": True, "pass": spec.name, "pass_number": spec.number,
        "passes_in_look": len(PASS_NAMES), "prompt_settings": spec.settings_record(),
        "permutation_seed": PERMUTATION_SEED, "permutation_seed_rule": PERMUTATION_SEED_RULE,
        "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "grader_step": step,
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), "roster": roster.roster_metadata(step),
        **dict(provenance or {}),
    }
    return Task(dataset=MemoryDataset(samples, name=f"acceptance_look_{spec.name}"),
                solver=[acceptance_look_grade()], epochs=1, config=GenerateConfig(), metadata=metadata,
                name=f"acceptance_look_{spec.name}")


def run_pass(spec: PassSpec, records, cases, *, step: int, log_dir: str,
             provenance: Mapping[str, Any] | None = None, model_roles: Mapping[str, Any] | None = None) -> Any:
    """Run one pass as its own Inspect eval, the grader role bound to the step's pin."""
    from inspect_ai import eval as inspect_eval

    roles = dict(model_roles) if model_roles is not None else {roster.GRADER_ROLE: grader.grader_pin(step)}
    task = pass_task(spec, records, cases, step=step, provenance=provenance)
    return inspect_eval(task, model=grader.NO_MODEL, model_roles=roles, epochs=1, log_dir=log_dir,
                        display="none")[0]


def offline_log_check(log: Any, step: int) -> list:
    """The check an offline (mockllm) pass gets in place of grader.check_grader_log, which
    requires the real pin: the log finished and holds its samples."""
    if getattr(log, "status", None) != "success":
        raise roster.RunConfigError(f"run config: log status is {getattr(log, 'status', None)!r}, not 'success'")
    if getattr(log, "samples", None) is None:
        raise roster.RunConfigError("run config: the log has no samples; read it in full, not header_only")
    return []


def _is_grader_event(event: Any) -> bool:
    return getattr(event, "event", None) == "model" and getattr(event, "role", None) == grader.GRADER_ROLE


def check_pass_samples(log: Any, records: Sequence[Mapping[str, Any]]) -> tuple[Any, int]:
    """Raise roster.RunConfigError unless the log holds one sample per record and exactly one
    completed grader call in each. A grader call that ended in an error and was followed by a
    later completed call on the same sample is skipped; an error after the completed call, a
    second completed call or no completed call fails the pass.

    Returns a view of the log without the skipped calls (what the log check and the
    configuration read) and the number of calls skipped."""
    if getattr(log, "samples", None) is None:
        raise roster.RunConfigError("run config: the log has no samples; read it in full, not header_only")
    ids = sorted(str(s.id) for s in log.samples)
    want = sorted(r["_file"] for r in records)
    if ids != want:
        raise roster.RunConfigError(f"run config: the log's samples {len(ids)} do not match the {len(want)} "
                                    f"calibration records")
    skipped, off, samples = 0, [], []
    for s in log.samples:
        events = list(s.events or ())
        calls = [(i, e) for i, e in enumerate(events) if _is_grader_event(e)]
        done = [i for i, e in calls if not getattr(e, "error", None)]
        errored = [i for i, e in calls if getattr(e, "error", None)]
        if len(done) != 1 or any(i > done[0] for i in errored):
            off.append(str(s.id))
            continue
        skipped += len(errored)
        drop = set(errored)
        samples.append(SimpleNamespace(id=s.id, epoch=getattr(s, "epoch", 1), error=getattr(s, "error", None),
                                       output=getattr(s, "output", None),
                                       events=[e for i, e in enumerate(events) if i not in drop]))
    if off:
        raise roster.RunConfigError(f"run config: samples without exactly one completed grader call: {sorted(off)[:5]}")
    view = SimpleNamespace(status=getattr(log, "status", None), samples=samples, eval=getattr(log, "eval", None),
                           location=getattr(log, "location", None))
    return view, skipped


def pass_config(log: Any, offline: bool) -> dict[str, Any]:
    """The grader configuration a pass recorded, as judge_trust.run_test_retest takes it:
    "response_caching" False only when no grader call read or wrote a cache, "model" the
    called model (left out offline, where it is mockllm), and the GenerateConfig fields,
    which must be the same on every call of the pass."""
    events = [e for _, e in grader.grader_events(log)]
    fields = {json.dumps(grader._fields(getattr(e, "config", None)), sort_keys=True, default=str) for e in events}
    if len(fields) != 1:
        raise roster.RunConfigError(f"run config: the pass's grader calls carry {len(fields)} configurations, not one")
    models = {str(getattr(e, "model", "")) for e in events}
    cfg: dict[str, Any] = {"response_caching": any(getattr(e, "cache", None) is not None for e in events)}
    if not offline:
        if len(models) != 1:
            raise roster.RunConfigError(f"run config: the pass called {len(models)} grader models")
        cfg["model"] = models.pop()
    cfg.update(json.loads(fields.pop()))
    return cfg


def log_file_of(log: Any) -> tuple[str, str]:
    """The pass's log file name and its SHA-256; the file must be on disk."""
    loc = getattr(log, "location", None)
    path = pathlib.Path(str(loc)) if loc else None
    if path is None or not path.is_file():
        raise roster.RunConfigError("run config: the pass's log file is not on disk, so it cannot be hashed")
    return path.name, _sha256_file(path)


def parse_pass(spec: PassSpec, log: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each record's reply in one pass, parsed with that pass's settings: levels mapped back to
    0 to 3, low-confidence flags and reasons. A reply that does not parse is listed in
    "unparsed" with its text; it is never asked again."""
    by_id = {str(s.id): s for s in log.samples}
    scores = {d: {} for d in gp.DIMENSIONS}
    low = {d: {} for d in gp.DIMENSIONS}
    reasons: dict[str, dict[str, str]] = {}
    unparsed = []
    for r in records:
        s = by_id.get(r["_file"])
        reply = (s.output.completion if s is not None and s.output else "") or ""
        try:
            if s is None or s.error is not None:
                raise gp.GraderReplyError("no reply (sample missing or ended in an error)")
            parsed = gp.parse_reply(reply, spec.settings)
        except gp.GraderReplyError as e:
            unparsed.append({"record": r["_file"], "error": str(e), "reply": reply})
            continue
        for d in gp.DIMENSIONS:
            scores[d][r["_file"]] = parsed[d]["level"]
            low[d][r["_file"]] = parsed[d]["low_confidence"]
        reasons[r["_file"]] = {d: parsed[d]["reason"] for d in gp.DIMENSIONS}
    return {"scores": scores, "low_confidence": low, "reasons": reasons, "unparsed": unparsed}


# --- Statistics ------------------------------------------------------------------------------

def low_confidence_counts(parsed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per pass and sub-dimension, the outputs the grader flagged low-confidence, and on pass 1
    the outputs flagged on any sub-dimension."""
    per_pass = {name: {d: sum(1 for v in p["low_confidence"][d].values() if v) for d in gp.DIMENSIONS}
                for name, p in parsed.items()}
    first = parsed["pass_1"]["low_confidence"]
    any_flag = sorted({oid for d in gp.DIMENSIONS for oid, v in first[d].items() if v})
    return {"per_pass": per_pass, "pass_1_outputs_flagged": len(any_flag), "pass_1_flagged_records": any_flag,
            "total": sum(sum(v.values()) for v in per_pass.values())}


def subset_distribution(records: Sequence[Mapping[str, Any]], pass_1: Mapping[str, Mapping[str, int]]) -> dict:
    """The subset's score distribution, published beside every agreement statistic (section 4):
    the author's grades and the pass 1 grader scores per sub-dimension, and the records per
    source and per worker model."""
    count: Callable[[str], dict[str, int]] = lambda key: dict(sorted(
        {str(v): sum(1 for r in records if str(r.get(key)) == str(v)) for v in {r.get(key) for r in records}}.items()))
    return {
        "author": {d: jt.score_distribution(r["grades"][d] for r in records) for d in gp.DIMENSIONS},
        "grader_pass_1": {d: jt.score_distribution(pass_1[d].values()) for d in gp.DIMENSIONS},
        "by_source": count("source"), "by_worker_model": count("worker_model"),
        "label": jt.STRATIFIED_LABEL,
    }


def seed_types(records: Sequence[Mapping[str, Any]], pass_1: Mapping[str, Mapping[str, int]]) -> dict:
    """Grader discrimination per seed type, descriptive and not a gate (section 7): per source
    and sub-dimension, the outputs, the mean author grade, the mean pass 1 grader score and the
    exact matches."""
    out: dict[str, dict] = {}
    for source in sorted({str(r.get("source")) for r in records}):
        group = [r for r in records if str(r.get("source")) == source]
        out[source] = {}
        for d in gp.DIMENSIONS:
            pairs = [(r["grades"][d], pass_1[d][r["_file"]]) for r in group]
            out[source][d] = {"n": len(pairs), "author_mean": sum(a for a, _ in pairs) / len(pairs),
                              "grader_mean": sum(g for _, g in pairs) / len(pairs),
                              "exact_matches": sum(1 for a, g in pairs if a == g)}
    return {"per_source": out, "note": "descriptive, not a gate (pre-registration section 7)"}


def compute_gates(records, parsed: Mapping[str, Mapping[str, Any]], configs: Sequence[Mapping[str, Any]], *,
                  step: int) -> dict[str, Any]:
    """The three gates of the look from the parsed passes and the author's grades. A gate is
    computed when every pass it needs parsed in full (validity: pass 1; test-retest: passes 1
    to 3; bias audit: all 7). Otherwise it is None, with the reason under "not_computed",
    and counts as FAIL."""
    incomplete = {n: len(parsed[n]["unparsed"]) for n in PASS_NAMES if parsed[n]["unparsed"]}
    not_computed: dict[str, str] = {}
    for gate, needs in GATE_NEEDS.items():
        short = [f"{n} ({incomplete[n]} unparsed)" for n in needs if n in incomplete]
        if short:
            not_computed[gate] = (f"FAIL: replies that do not parse leave {', '.join(short)} incomplete, and this "
                                  f"gate needs {'every pass' if needs == PASS_NAMES else ', '.join(needs)}; "
                                  f"the grader is not asked again")
    human = {d: {r["_file"]: r["grades"][d] for r in records} for d in gp.DIMENSIONS}
    retest = [parsed[n]["scores"] for n in RETEST_NAMES]
    out: dict[str, Any] = {"validity": None, "test_retest": None, "bias_audit": None}
    if "validity" not in not_computed:
        out["validity"] = jt.validity_gate(retest[0], human)
    if "test_retest" not in not_computed:
        out["test_retest"] = jt.run_test_retest(retest[0], lambda n: retest[n - 1], step=step, grader_configs=configs,
                                                bootstrap_seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    if "bias_audit" not in not_computed:
        bias = jt.bias_audit(retest, {n: parsed[n]["scores"] for n in BIAS_NAMES}, human,
                             dimension_order_applies=True, permutation_seed=PERMUTATION_SEED,
                             dimension_seed=PERMUTATION_SEED)
        bias["dimension_order"] = list(swapped_dimensions())
        bias["dimension_order_note"] = ("the two sub-dimensions are swapped, the one order that differs from the "
                                        "default; no seed is drawn for it, and dimension_seed repeats the "
                                        "permutation seed")
        out["bias_audit"] = bias
    out["not_computed"] = not_computed
    out["kappa"] = jt.KAPPA_NOTE
    return out


def verdict(gates: Mapping[str, Any] | None, step: int) -> dict[str, Any]:
    if gates is None:
        out = {"validity": "FAIL", "test_retest": "FAIL", "bias_audit": "FAIL", "overall": "FAIL",
               "reason": "no gate computed"}
    else:
        passed = {
            "validity": gates.get("validity") is not None and gates["validity"]["passed"],
            "test_retest": gates.get("test_retest") is not None and gates["test_retest"]["alpha"]["passed"],
            "bias_audit": gates.get("bias_audit") is not None and gates["bias_audit"]["passed"],
        }
        out = {k: "PASS" if v else "FAIL" for k, v in passed.items()}
        out["overall"] = "PASS" if all(passed.values()) else "FAIL"
        if gates.get("not_computed"):
            out["not_computed"] = dict(gates["not_computed"])
    out["next"] = ("LLM-graded results may be reported once the other section 7 rules hold"
                   if out["overall"] == "PASS" else NEXT_REMEDY[step] + "; this script does not run it")
    return out


def _f(x: Any, nd: int = 3) -> str:
    return "undefined" if x is None else f"{x:.{nd}f}"


def summary_lines(res: Mapping[str, Any]) -> list[str]:
    """Every gate's value beside its limit, PASS or FAIL per gate and overall."""
    v = res["verdict"]
    lines = [f"acceptance look, step {res['grader_step']} ({res['grader_model']}): {res['records']} outputs, "
             f"{len(res['passes'])} passes, {res['unparsed_count']} unparsed replies"]
    g = res.get("gates") or {}
    nc = g.get("not_computed") or {}
    val = g.get("validity")
    if val is None:
        lines.append(f"validity: FAIL, not computed: {nc.get('validity', 'no gate computed')}")
    else:
        for d in gp.DIMENSIONS:
            a1, c1 = val["per_subdimension"][d]["icc_a1"], val["per_subdimension"][d]["icc_c1"]
            lines.append(f"  validity {d}: ICC(A,1) {_f(a1['estimate'])}, lower limit {_f(a1['ci_lower'])} "
                         f"(limit 0.60); ICC(C,1) {_f(c1['estimate'])} reported")
        lines.append(f"validity: smaller lower limit {_f(val['gating_lower_limit'])} on "
                     f"{val['gating_subdimension']} versus 0.60: {v['validity']}")
    tr = g.get("test_retest")
    if tr is None:
        lines.append(f"test-retest: FAIL, not computed: {nc.get('test_retest', 'no gate computed')}")
    else:
        alpha = tr["alpha"]
        for d in gp.DIMENSIONS:
            s = alpha["per_subdimension"][d]
            b = s["bootstrap"]
            lines.append(f"  test-retest {d}: alpha {_f(s['alpha'])} (limit 0.80), 95% CI "
                         f"[{_f(b['ci_lower'])}, {_f(b['ci_upper'])}], q {_f(b['q'])}, raw agreement "
                         f"{_f(s['raw_agreement'])}" + (f", {s['note']}" if s.get("note") else ""))
        lines.append(f"test-retest: {v['test_retest']}")
    bias = g.get("bias_audit")
    if bias is None:
        lines.append(f"bias audit: FAIL, not computed: {nc.get('bias_audit', 'no gate computed')}")
    else:
        for d in gp.DIMENSIONS:
            s = bias["per_subdimension"][d]
            shifts = ", ".join(f"{p} {_f(x['shift'])} (excess {_f(x['excess'])})" for p, x in s["passes"].items())
            lines.append(f"  bias {d}: baseline {_f(s['baseline'])}; shifts {shifts}; excess limit 0.10")
            lean = s["leniency"]
            lines.append(f"  bias {d}: leniency {_f(lean['mean'])} [{_f(lean.get('ci_lower'))}, "
                         f"{_f(lean.get('ci_upper'))}] (limit 0.25 in magnitude); ICC(C,1) minus ICC(A,1) "
                         f"{_f(s['icc_c1_minus_icc_a1'])}")
        lines.append(f"bias audit: {v['bias_audit']}")
    if g.get("kappa"):
        lines.append(f"kappa: {g['kappa']}")
    low = res["low_confidence"]
    lines.append(f"low-confidence flags: {low['total']} across all passes; pass 1 outputs flagged "
                 f"{low['pass_1_outputs_flagged']}")
    lines.append(f"OVERALL {v['overall']}" + ("" if v["overall"] == "PASS" else f": {v['next']}"))
    return lines


def look_result(records, specs: Sequence[PassSpec], logs: Mapping[str, Any], configs: Mapping[str, dict], *,
                step: int, offline: bool, log_checks: Mapping[str, Any],
                skipped: Mapping[str, int] | None = None, provenance: Mapping[str, Any] | None = None,
                intra_rater: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The look's record: per pass its settings, log and parse; per output the author's grades
    and every pass's scores; every gate that could be computed, the distribution, the
    low-confidence count and the verdict."""
    parsed = {s.name: parse_pass(s, logs[s.name], records) for s in specs}
    unparsed = {name: p["unparsed"] for name, p in parsed.items() if p["unparsed"]}
    n_unparsed = sum(len(v) for v in unparsed.values())
    gates = compute_gates(records, parsed, [configs[n] for n in RETEST_NAMES], step=step)
    passes = []
    for s in specs:
        loc = getattr(logs[s.name], "location", None)
        path = pathlib.Path(str(loc)) if loc else None
        passes.append({"name": s.name, "number": s.number, "settings": s.settings_record(),
                       "grader_config": configs[s.name], "log_check": log_checks[s.name],
                       "errored_calls_skipped": (skipped or {}).get(s.name, 0),
                       "log_file": path.name if path else None,
                       "log_sha256": _sha256_file(path) if path and path.is_file() else None,
                       "unparsed": parsed[s.name]["unparsed"]})
    rows = [{"record": r["_file"], "sample_id": r["sample_id"], "worker_model": r.get("worker_model"),
             "source": r.get("source"), "author": dict(r["grades"]),
             "grader": {s.name: {d: parsed[s.name]["scores"][d].get(r["_file"]) for d in gp.DIMENSIONS}
                        for s in specs},
             "low_confidence": {s.name: {d: parsed[s.name]["low_confidence"][d].get(r["_file"])
                                         for d in gp.DIMENSIONS} for s in specs},
             "reasons": {s.name: parsed[s.name]["reasons"].get(r["_file"]) for s in specs}}
            for r in records]
    pass_1_ok = not parsed["pass_1"]["unparsed"]
    pass_1 = parsed["pass_1"]["scores"]
    permutation = jt.permuted_options(list(gp.LEVELS), PERMUTATION_SEED)
    return {
        "look": "acceptance look", "grader_step": step, "grader_model": grader.grader_pin(step),
        "grader_generate_config": roster.grader_generate_config_args(step), "offline": offline,
        "permutation_seed": PERMUTATION_SEED, "permutation_seed_rule": PERMUTATION_SEED_RULE,
        "option_permutation": {k: permutation[k] for k in ("order", "is_identity", "is_reversal")},
        "dimension_order": list(swapped_dimensions()), "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES,
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), **dict(provenance or {}),
        "intra_rater": dict(intra_rater) if intra_rater else None,
        "records": len(records), "passes": passes,
        "unparsed_count": n_unparsed, "gates": gates,
        "score_distribution": subset_distribution(records, pass_1) if pass_1_ok else None,
        "seed_types": seed_types(records, pass_1) if pass_1_ok else None,
        "low_confidence": low_confidence_counts(parsed), "verdict": verdict(gates, step),
        "rows": rows, "estimate_label": jt.STRATIFIED_LABEL,
        "note": "one acceptance look per grader (pre-registration section 7, decision D-T1)",
    }


def seal_split(res: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a look record into the evidence part (aggregates and every gate value) and the
    sealed part (the per-record rows: scores, flags, reasons, the unparsed replies, the
    flagged record ids and the per-record test-retest scores)."""
    public = json.loads(json.dumps(res))
    sealed: dict[str, Any] = {"look": res["look"], "grader_step": res["grader_step"], "rows": public.pop("rows")}
    sealed["unparsed"] = {}
    for p in public["passes"]:
        sealed["unparsed"][p["name"]] = p["unparsed"]
        p["unparsed"] = [{"record": u["record"]} for u in p["unparsed"]]
    sealed["pass_1_flagged_records"] = public["low_confidence"].pop("pass_1_flagged_records")
    tr = (public.get("gates") or {}).get("test_retest")
    if tr is not None:
        sealed["test_retest_scores"] = tr.pop("scores", None)
    return public, sealed


# --- Ledger ----------------------------------------------------------------------------------

def save_ledger(path: pathlib.Path, ledger: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def create_ledger(path: pathlib.Path, ledger: Mapping[str, Any]) -> None:
    """Create the ledger with exclusive create; an existing ledger raises FileExistsError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n")


def recheck_pass(entry: Mapping[str, Any], spec: PassSpec, records, *, log_dir: str, step: int,
                 provenance: Mapping[str, Any], check: Callable[[Any, int], Any], offline: bool):
    """Re-check a pass the ledger holds: its log file is in the log folder with the recorded
    SHA-256, its metadata names the pass and the look's provenance, and it passes the sample
    and log checks again. Returns (log, configuration, skipped calls, completed calls)."""
    from inspect_ai.log import read_eval_log

    path = pathlib.Path(log_dir) / str(entry.get("log_file"))
    if not path.is_file():
        raise roster.RunConfigError(f"run config: the log {entry.get('log_file')} of {spec.name} is not in the log folder")
    if _sha256_file(path) != entry.get("log_sha256"):
        raise roster.RunConfigError(f"run config: the log {path.name} of {spec.name} does not match its recorded SHA-256")
    log = read_eval_log(str(path))
    meta = dict(getattr(log.eval, "metadata", None) or {})
    if meta.get("pass") != spec.name:
        raise roster.RunConfigError(f"run config: the log {path.name} is pass {meta.get('pass')!r}, not {spec.name}")
    off = sorted(k for k, v in provenance.items() if meta.get(k) != v)
    if off:
        raise roster.RunConfigError(f"run config: the log {path.name} differs from the look on {off}")
    view, skipped = check_pass_samples(log, records)
    check(view, step)
    return log, pass_config(view, offline), skipped, len(grader.grader_events(view))


# --- Main ------------------------------------------------------------------------------------

def _roles_for(model_roles: Any, name: str) -> Mapping[str, Any] | None:
    if model_roles is None:
        return None
    return model_roles(name) if callable(model_roles) else model_roles


def _check_name(check: Callable) -> str:
    return "grader.check_grader_log" if check is grader.check_grader_log else getattr(check, "__name__", "log check")


def main(argv: Sequence[str] | None = None, *, model_roles: Any = None, out_dir: pathlib.Path = OUT_DIR,
         records_dir: pathlib.Path = RECORDS_DIR, cases_file: pathlib.Path = CASES_FILE,
         judge_dev_dir: pathlib.Path = JUDGE_DEV_OUT, log_check: Callable[[Any, int], Any] | None = None,
         repo: pathlib.Path = REPO, ledger_dir: pathlib.Path | None = None) -> int:
    """model_roles (tests only): a mapping, or a callable from pass name to a mapping, of
    offline (mockllm) role models; the look is then offline. log_check (tests only): replaces
    the log check; live, it is grader.check_grader_log. repo and ledger_dir (tests only): the
    repository whose git state is checked, and the ledger folder in place of the one under its
    git common directory."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=int, required=True, choices=roster.GRADER_STEPS)
    ap.add_argument("--live", action="store_true", help="make the 280 grader calls (keyed, paid)")
    ap.add_argument("--resume", action="store_true", help="continue the look the attempt ledger records")
    ap.add_argument("--log-dir", default=None, help="folder for the Inspect logs, outside the repository")
    ap.add_argument("--sealed-dir", default=None,
                    help="folder outside the repository for the per-record rows of a look that fails at step 1 or 2")
    ap.add_argument("--intra-rater", default=None, help="steps 2 and 3: the committed intra-rater record (JSON)")
    a = ap.parse_args(argv)
    offline = model_roles is not None
    check = log_check or (offline_log_check if offline else grader.check_grader_log)
    try:
        check_look_order(a.step, out_dir, judge_dev_dir)
        intra = check_intra_rater(a.intra_rater, a.step, repo)
        cases, records = load_cases(cases_file), load_records(records_dir)
        check_subset(records, cases)
        specs = plan_passes()
        sizes = [len(m["content"]) for s in specs for r in records for m in record_messages(r, cases, s.settings)]
    except (Refused, jt.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if a.resume and not a.live:
        print("refused: --resume continues a live look; pass --live with it", file=sys.stderr)
        return EXIT_ARGS
    perm = jt.permuted_options(list(gp.LEVELS), PERMUTATION_SEED)
    if not a.live:
        print(f"acceptance look plan, step {a.step}: {len(specs)} passes of {len(records)} calls, grader "
              f"{grader.grader_pin(a.step)} with {roster.grader_generate_config_args(a.step)}, cache off, prompt "
              f"{gp.PROMPT_VERSION} ({gp.prompt_sha256()[:12]}), about {sum(sizes) // 4 // (len(specs) * len(records))} input "
              f"tokens per call")
        print(f"permutation seed {PERMUTATION_SEED} ({PERMUTATION_SEED_RULE}, fixed by rule): option order "
              f"{perm['order']}; bootstrap seed {BOOTSTRAP_SEED}, {RESAMPLES} resamples")
        for s in specs:
            print(f"  {s.number}. {s.name}: {s.settings_record()}")
        if intra:
            print(f"intra-rater record {intra['file']} ({intra['sha256'][:12]})")
        try:
            started = ledger_path(a.step, repo, ledger_dir).exists()
        except Refused:
            started = False
        if started:
            print(f"a look at step {a.step} was started: its attempt ledger exists; --live --resume continues it")
        print("dry run: no call made (pass --live with --log-dir and --sealed-dir to make the calls)")
        return EXIT_OK

    try:
        ledger_file = ledger_path(a.step, repo, ledger_dir)
        state = repo_state(repo, cases_file, records_dir)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    ledger: dict[str, Any] | None = None
    if a.resume:
        if not ledger_file.exists():
            print(f"refused: no attempt ledger for step {a.step}; there is no look to resume", file=sys.stderr)
            return EXIT_REFUSED
        ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
        for flag, key in ((a.log_dir, "log_dir"), (a.sealed_dir, "sealed_dir")):
            if flag and pathlib.Path(flag).resolve() != pathlib.Path(ledger[key]).resolve():
                print(f"refused: --{key.replace('_', '-')} differs from the one the ledger records", file=sys.stderr)
                return EXIT_ARGS
        log_dir, sealed_dir = ledger["log_dir"], ledger["sealed_dir"]
    else:
        log_dir, sealed_dir = a.log_dir, a.sealed_dir
    for flag, value in (("--log-dir", log_dir), ("--sealed-dir", sealed_dir)):
        if not value or _inside(value, REPO) or _inside(value, repo):
            print(f"refused: --live needs {flag} outside the repository", file=sys.stderr)
            return EXIT_ARGS
    sealed_target = sealed_file(pathlib.Path(sealed_dir), a.step)
    if sealed_target.exists():
        print(f"refused: {sealed_target.name} exists in the sealed folder; it is never overwritten", file=sys.stderr)
        return EXIT_REFUSED
    provenance = {"head": state["head"], "calibration_jsonl_sha256": state["calibration_jsonl_sha256"],
                  "records_sha256": state["records_sha256"], "permutation_seed": PERMUTATION_SEED,
                  "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES,
                  "grader_prompt_sha256": gp.prompt_sha256(), "anchors_sha256": rubric.llm_anchor_sha256(),
                  "intra_rater_sha256": intra["sha256"] if intra else None}
    if ledger is None:
        ledger = {"look": "acceptance look attempt ledger", "step": a.step, "started": _now(),
                  "provenance": provenance, "log_dir": str(pathlib.Path(log_dir).resolve()),
                  "sealed_dir": str(pathlib.Path(sealed_dir).resolve()),
                  "attempts": [{"attempt": 1, "started": _now(), "resume": False}], "passes": [], "failures": []}
        try:
            create_ledger(ledger_file, ledger)
        except FileExistsError:
            print(f"refused: a look at step {a.step} was already started (its attempt ledger exists); it is never "
                  f"started again. Pass --resume to continue it", file=sys.stderr)
            return EXIT_REFUSED
    else:
        off = sorted(k for k, v in provenance.items() if ledger.get("provenance", {}).get(k) != v)
        if off:
            print(f"refused: the look differs from the one the ledger started on {off}", file=sys.stderr)
            return EXIT_REFUSED
        if [p["name"] for p in ledger["passes"]] != list(PASS_NAMES[:len(ledger["passes"])]):
            print("refused: the ledger's passes are not the look's passes in order", file=sys.stderr)
            return EXIT_REFUSED
        ledger["attempts"].append({"attempt": len(ledger["attempts"]) + 1, "started": _now(), "resume": True,
                                   "kept": [p["name"] for p in ledger["passes"]]})
        save_ledger(ledger_file, ledger)
    attempt = ledger["attempts"][-1]["attempt"]

    def fail(stage: str, name: str | None, err: Exception, log: Any = None) -> int:
        entry = {"attempt": attempt, "time": _now(), "stage": stage, "pass": name,
                 "error": f"{type(err).__name__}: {err}"}
        loc = getattr(log, "location", None)
        if loc and pathlib.Path(str(loc)).is_file():
            entry["log_file"], entry["log_sha256"] = pathlib.Path(str(loc)).name, _sha256_file(pathlib.Path(str(loc)))
        ledger["failures"].append(entry)
        save_ledger(ledger_file, ledger)
        print(f"FAIL: {stage}{f' ({name})' if name else ''}: {err}", file=sys.stderr)
        print(f"no look file written; the attempt ledger records the failure. Run again with --live --resume "
              f"to keep the passes that finished and run the rest", file=sys.stderr)
        return EXIT_CHECK

    logs: dict[str, Any] = {}
    configs: dict[str, dict] = {}
    checks: dict[str, Any] = {}
    skipped: dict[str, int] = {}
    by_name = {s.name: s for s in specs}
    for entry in ledger["passes"]:
        s = by_name[entry["name"]]
        try:
            log, cfg, n_skip, n_calls = recheck_pass(entry, s, records, log_dir=log_dir, step=a.step, provenance=provenance,
                                            check=check, offline=offline)
        except Exception as e:  # noqa: BLE001 - a kept pass that fails its re-check stops the look
            return fail(f"re-check of pass {s.number} of {len(specs)}", s.name, e)
        logs[s.name], configs[s.name], skipped[s.name] = log, cfg, n_skip
        checks[s.name] = (f"{_check_name(check)} passed (re-checked on resume); {n_calls} completed grader calls, "
                          f"one per output")
    for s in specs:
        if s.name in logs:
            continue
        log = None
        try:
            log = run_pass(s, records, cases, step=a.step, log_dir=log_dir, provenance=provenance,
                           model_roles=_roles_for(model_roles, s.name))
            view, n_skip = check_pass_samples(log, records)
            check(view, a.step)
            cfg = pass_config(view, offline)
            name, sha = log_file_of(log)
        except Exception as e:  # noqa: BLE001 - any failure of a pass stops the look and is recorded
            return fail(f"pass {s.number} of {len(specs)}", s.name, e, log)
        logs[s.name], configs[s.name], skipped[s.name] = log, cfg, n_skip
        checks[s.name] = (f"{_check_name(check)} passed; {len(grader.grader_events(view))} completed grader calls, "
                          f"one per output")
        ledger["passes"].append({"name": s.name, "number": s.number, "attempt": attempt, "finished": _now(),
                                 "log_file": name, "log_sha256": sha, "grader_calls": len(grader.grader_events(view)),
                                 "errored_calls_skipped": n_skip})
        save_ledger(ledger_file, ledger)
    try:
        res = look_result(records, specs, logs, configs, step=a.step, offline=offline, log_checks=checks,
                          skipped=skipped, provenance=provenance, intra_rater=intra)
    except (jt.Refused, roster.RunConfigError) as e:
        return fail("gates", None, e)
    res["attempts"] = ledger["attempts"]
    res["failures"] = ledger["failures"]
    res["ledger"] = {"file": ledger_file.name, "folder": f"<git common dir>/{LEDGER_FOLDER}"}
    target = look_file(out_dir, a.step)
    sealed = res["verdict"]["overall"] == "FAIL" and a.step in SEALED_STEPS
    sealed_info = None
    try:
        if sealed:
            public, private = seal_split(res)
            sealed_target.parent.mkdir(parents=True, exist_ok=True)
            body = (json.dumps(private, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
            with open(sealed_target, "xb") as fh:
                fh.write(body)
            sealed_info = {"sealed": True, "file": sealed_target.name, "sha256": _sha256_bytes(body),
                           "records": len(private["rows"]),
                           "note": ("the per-record rows are sealed outside the repository: the grading that "
                                    "follows a failed look at step 1 or 2 stays blind to this grader's scores "
                                    "(pre-registration section 7)")}
            public["rows"] = sealed_info
            res_out = public
        else:
            res_out = res
        out_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(res_out, indent=1, ensure_ascii=False) + "\n"
        with open(target, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except FileExistsError as e:
        print(f"refused: {pathlib.Path(str(e.filename)).name} appeared during the look; it is never overwritten",
              file=sys.stderr)
        return EXIT_REFUSED
    ledger["completed"] = {"time": _now(), "look_file": target.name, "look_sha256": _sha256_file(target),
                           "verdict": res["verdict"]["overall"], "sealed_file": sealed_info}
    save_ledger(ledger_file, ledger)
    print("\n".join(summary_lines(res_out)))
    if sealed_info:
        print(f"per-record rows sealed in {sealed_target} (SHA-256 {sealed_info['sha256'][:12]})")
    print(target)
    return EXIT_OK if res["verdict"]["overall"] == "PASS" else EXIT_GATE_FAIL


if __name__ == "__main__":
    sys.exit(main())
