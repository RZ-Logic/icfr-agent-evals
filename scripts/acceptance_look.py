"""The acceptance look: one grader's single scoring of the 40-case calibration subset.

Pre-registration section 7. At the look the grader scores the calibration subset in
seven passes of 40 calls, each pass one Inspect eval through icfr_agent_evals.grader
with the grader role bound to the ladder step's pin and response caching off:

  pass 1   the acceptance-look scoring, default prompt settings
  pass 2   test-retest replicate, default settings, independent
  pass 3   test-retest replicate, default settings, independent
  option_order_reversed      rubric options listed 3, 2, 1, 0
  option_order_permuted      one random permutation drawn from --permutation-seed
  labels_relabeled           score labels 0 to 3 shown as D, C, B, A
  dimension_order_permuted   the two sub-dimensions listed and answered in swapped order

The permutation seed is recorded in the Inspect task metadata of every pass. Every log
must pass grader.check_grader_log (pin, configuration, resolved model string) and hold
one grader call per calibration output. Then icfr_agent_evals.judge_trust computes:

  validity      ICC(A,1) grader (pass 1) versus the author's grades, per sub-dimension;
                the smaller lower limit of the McGraw-Wong 95% CI must be at least 0.60.
                ICC(C,1) is reported. The kappa gate does not apply (no binary checks).
  test-retest   ordinal Krippendorff alpha over passes 1 to 3 at least 0.80, with its
                unit-bootstrap CI, q and raw agreement; an undefined alpha fails.
  bias audit    each perturbed pass's shift minus the largest pairwise test-retest
                baseline at most 0.10; leniency on pass 1 within 0.25 in magnitude;
                the ICC(C,1) minus ICC(A,1) gap reported.

The score distribution of the subset and the low-confidence count are reported beside
them. Each grader gets one look: the script refuses when
docs/evidence/calibration/acceptance-look-step-<N>.json exists, and writes that file only
after every pass ran and every log passed its check. A failed log check writes nothing
and names the pass. A gate FAIL starts the judge remedy order (step 2: the intra-rater
check, then the next grader), which this script does not run.

Calibration split only. The script reads data/calibration/calibration.jsonl and
data/calibration/records/*.json and nothing from the evaluation split. The subset is
never used for tuning.

Run:
  python scripts/acceptance_look.py --step 1 --permutation-seed S              (plan only, no call)
  python scripts/acceptance_look.py --step 1 --permutation-seed S --live --log-dir DIR   (paid: 280 calls)

Exit codes: 0 plan printed or every gate PASS, 1 a pass or log check failed (nothing
written), 2 bad arguments, 3 refused, 4 the look ran and a gate FAILED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from dataclasses import dataclass
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
DEFAULT_BOOTSTRAP_SEED = casefile.split_seed(SPLIT)
EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED, EXIT_GATE_FAIL = 0, 1, 2, 3, 4

RETEST_NAMES: tuple[str, ...] = ("pass_1", "pass_2", "pass_3")
BIAS_NAMES: tuple[str, ...] = (*jt.BIAS_PASSES, jt.DIMENSION_ORDER_PERMUTED)
PASS_NAMES: tuple[str, ...] = (*RETEST_NAMES, *BIAS_NAMES)

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


# --- Plan ------------------------------------------------------------------------------------

def swapped_dimensions() -> tuple[str, ...]:
    """The dimension order of the dimension pass. Both LLM-graded sub-dimensions share one
    prompt, and with two of them the swap is the one order that differs from the default."""
    if len(gp.DIMENSIONS) != 2:
        raise Refused(f"the dimension pass is built for two shared sub-dimensions, got {len(gp.DIMENSIONS)}")
    return tuple(reversed(gp.DIMENSIONS))


def plan_passes(permutation_seed: int) -> list[PassSpec]:
    """The 7 passes of the look in run order: 3 test-retest passes on the default settings,
    then the bias-audit passes (option order reversed, the one seeded permutation, the labels
    relabeled D, C, B, A, the dimension order swapped)."""
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


# --- Look order ------------------------------------------------------------------------------

def look_file(out_dir: pathlib.Path, step: int) -> pathlib.Path:
    return out_dir / f"acceptance-look-step-{step}.json"


def check_look_order(step: int, out_dir: pathlib.Path, judge_dev_dir: pathlib.Path) -> None:
    """One look per grader, in ladder order.

    Refused: a step outside 1 to 3; a step whose look file exists; step 1 before judge remedy
    order step 1 has ended (a live judge-development round met the stop criteria, or round 3
    ran) or on prompt wording other than the one step 1 ended on; step 2 or 3 unless the look
    before it exists and failed."""
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
    verdict = (json.loads(prev.read_text(encoding="utf-8")).get("verdict") or {}).get("overall")
    if verdict != "FAIL":
        raise Refused(f"the step {step - 1} acceptance look did not fail (verdict {verdict!r}); "
                      f"the step {step} grader does not grade")


# --- Running a pass --------------------------------------------------------------------------

def pass_task(spec: PassSpec, records: Sequence[Mapping[str, Any]], cases: Mapping[str, dict], *,
              step: int, permutation_seed: int, bootstrap_seed: int):
    """An Inspect task for one pass: one sample per record, one grader call each through
    grader.grade. The task metadata records the pass, its settings and the permutation seed."""
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
        "permutation_seed": permutation_seed, "bootstrap_seed": bootstrap_seed, "grader_step": step,
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), "roster": roster.roster_metadata(step),
    }
    return Task(dataset=MemoryDataset(samples, name=f"acceptance_look_{spec.name}"),
                solver=[acceptance_look_grade()], epochs=1, config=GenerateConfig(), metadata=metadata,
                name=f"acceptance_look_{spec.name}")


def run_pass(spec: PassSpec, records, cases, *, step: int, permutation_seed: int, bootstrap_seed: int,
             log_dir: str, model_roles: Mapping[str, Any] | None = None) -> Any:
    """Run one pass as its own Inspect eval, the grader role bound to the step's pin."""
    from inspect_ai import eval as inspect_eval

    roles = dict(model_roles) if model_roles is not None else {roster.GRADER_ROLE: grader.grader_pin(step)}
    task = pass_task(spec, records, cases, step=step, permutation_seed=permutation_seed,
                     bootstrap_seed=bootstrap_seed)
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


def check_pass_samples(log: Any, records: Sequence[Mapping[str, Any]]) -> None:
    """Raise roster.RunConfigError unless the log holds one sample per record and exactly one
    grader call in each."""
    ids = sorted(str(s.id) for s in log.samples)
    want = sorted(r["_file"] for r in records)
    if ids != want:
        raise roster.RunConfigError(f"run config: the log's samples {len(ids)} do not match the {len(want)} "
                                    f"calibration records")
    per: dict[str, int] = {}
    for sample, _ in grader.grader_events(log):
        per[str(sample.id)] = per.get(str(sample.id), 0) + 1
    off = sorted(i for i in want if per.get(i, 0) != 1)
    if off:
        raise roster.RunConfigError(f"run config: samples without exactly one grader call: {off[:5]}")


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


def parse_pass(spec: PassSpec, log: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each record's reply in one pass, parsed with that pass's settings: levels mapped back to
    0 to 3, low-confidence flags and reasons. A reply that does not parse is listed in
    "unparsed" with its text."""
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
                  step: int, permutation_seed: int, bootstrap_seed: int, resamples: int) -> dict[str, Any]:
    """The three gates of the look from the 7 parsed passes and the author's grades."""
    human = {d: {r["_file"]: r["grades"][d] for r in records} for d in gp.DIMENSIONS}
    retest = [parsed[n]["scores"] for n in RETEST_NAMES]
    validity = jt.validity_gate(retest[0], human)
    test_retest = jt.run_test_retest(retest[0], lambda n: retest[n - 1], step=step, grader_configs=configs,
                                     bootstrap_seed=bootstrap_seed, resamples=resamples)
    bias = jt.bias_audit(retest, {n: parsed[n]["scores"] for n in BIAS_NAMES}, human,
                         dimension_order_applies=True, permutation_seed=permutation_seed,
                         dimension_seed=permutation_seed)
    bias["dimension_order"] = list(swapped_dimensions())
    bias["dimension_order_note"] = ("the two sub-dimensions are swapped, the one order that differs from the "
                                    "default; no seed is drawn for it, and dimension_seed repeats the permutation seed")
    return {"validity": validity, "test_retest": test_retest, "bias_audit": bias, "kappa": jt.KAPPA_NOTE}


def verdict(gates: Mapping[str, Any] | None, step: int) -> dict[str, Any]:
    if gates is None:
        out = {"validity": "FAIL", "test_retest": "FAIL", "bias_audit": "FAIL", "overall": "FAIL",
               "reason": "replies that do not parse leave the 40-case subset incomplete, so no gate is computed"}
    else:
        out = {"validity": "PASS" if gates["validity"]["passed"] else "FAIL",
               "test_retest": "PASS" if gates["test_retest"]["alpha"]["passed"] else "FAIL",
               "bias_audit": "PASS" if gates["bias_audit"]["passed"] else "FAIL"}
        out["overall"] = "PASS" if all(v == "PASS" for v in out.values()) else "FAIL"
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
    g = res.get("gates")
    if g is None:
        lines.append(f"no gate computed: {v['reason']}")
    else:
        val = g["validity"]
        for d in gp.DIMENSIONS:
            a1, c1 = val["per_subdimension"][d]["icc_a1"], val["per_subdimension"][d]["icc_c1"]
            lines.append(f"  validity {d}: ICC(A,1) {_f(a1['estimate'])}, lower limit {_f(a1['ci_lower'])} "
                         f"(limit 0.60); ICC(C,1) {_f(c1['estimate'])} reported")
        lines.append(f"validity: smaller lower limit {_f(val['gating_lower_limit'])} on "
                     f"{val['gating_subdimension']} versus 0.60: {v['validity']}")
        alpha = g["test_retest"]["alpha"]
        for d in gp.DIMENSIONS:
            s = alpha["per_subdimension"][d]
            b = s["bootstrap"]
            lines.append(f"  test-retest {d}: alpha {_f(s['alpha'])} (limit 0.80), 95% CI "
                         f"[{_f(b['ci_lower'])}, {_f(b['ci_upper'])}], q {_f(b['q'])}, raw agreement "
                         f"{_f(s['raw_agreement'])}" + (f", {s['note']}" if s.get("note") else ""))
        lines.append(f"test-retest: {v['test_retest']}")
        bias = g["bias_audit"]
        for d in gp.DIMENSIONS:
            s = bias["per_subdimension"][d]
            shifts = ", ".join(f"{p} {_f(x['shift'])} (excess {_f(x['excess'])})" for p, x in s["passes"].items())
            lines.append(f"  bias {d}: baseline {_f(s['baseline'])}; shifts {shifts}; excess limit 0.10")
            lean = s["leniency"]
            lines.append(f"  bias {d}: leniency {_f(lean['mean'])} [{_f(lean.get('ci_lower'))}, "
                         f"{_f(lean.get('ci_upper'))}] (limit 0.25 in magnitude); ICC(C,1) minus ICC(A,1) "
                         f"{_f(s['icc_c1_minus_icc_a1'])}")
        lines.append(f"bias audit: {v['bias_audit']}")
        lines.append(f"kappa: {g['kappa']}")
    low = res["low_confidence"]
    lines.append(f"low-confidence flags: {low['total']} across all passes; pass 1 outputs flagged "
                 f"{low['pass_1_outputs_flagged']}")
    lines.append(f"OVERALL {v['overall']}" + ("" if v["overall"] == "PASS" else f": {v['next']}"))
    return lines


def look_result(records, specs: Sequence[PassSpec], logs: Mapping[str, Any], configs: Mapping[str, dict], *,
                step: int, permutation_seed: int, bootstrap_seed: int, resamples: int, offline: bool,
                log_checks: Mapping[str, Any]) -> dict[str, Any]:
    """The look's record: per pass its settings, log and parse; per output the author's grades
    and every pass's scores; the gates, the distribution, the low-confidence count and the
    verdict."""
    parsed = {s.name: parse_pass(s, logs[s.name], records) for s in specs}
    unparsed = {name: p["unparsed"] for name, p in parsed.items() if p["unparsed"]}
    n_unparsed = sum(len(v) for v in unparsed.values())
    gates = None
    if not n_unparsed:
        gates = compute_gates(records, parsed, [configs[n] for n in RETEST_NAMES], step=step,
                              permutation_seed=permutation_seed, bootstrap_seed=bootstrap_seed, resamples=resamples)
    passes = []
    for s in specs:
        loc = getattr(logs[s.name], "location", None)
        path = pathlib.Path(str(loc)) if loc else None
        passes.append({"name": s.name, "number": s.number, "settings": s.settings_record(),
                       "grader_config": configs[s.name], "log_check": log_checks[s.name],
                       "log_file": path.name if path else None,
                       "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
                       "unparsed": parsed[s.name]["unparsed"]})
    rows = [{"record": r["_file"], "sample_id": r["sample_id"], "worker_model": r.get("worker_model"),
             "source": r.get("source"), "author": dict(r["grades"]),
             "grader": {s.name: {d: parsed[s.name]["scores"][d].get(r["_file"]) for d in gp.DIMENSIONS}
                        for s in specs},
             "low_confidence": {s.name: {d: parsed[s.name]["low_confidence"][d].get(r["_file"])
                                         for d in gp.DIMENSIONS} for s in specs},
             "reasons": {s.name: parsed[s.name]["reasons"].get(r["_file"]) for s in specs}}
            for r in records]
    pass_1 = parsed["pass_1"]["scores"]
    permutation = jt.permuted_options(list(gp.LEVELS), permutation_seed)
    return {
        "look": "acceptance look", "grader_step": step, "grader_model": grader.grader_pin(step),
        "grader_generate_config": roster.grader_generate_config_args(step), "offline": offline,
        "permutation_seed": permutation_seed,
        "option_permutation": {k: permutation[k] for k in ("order", "is_identity", "is_reversal")},
        "dimension_order": list(swapped_dimensions()), "bootstrap_seed": bootstrap_seed, "resamples": resamples,
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(), "records": len(records), "passes": passes,
        "unparsed_count": n_unparsed, "gates": gates,
        "score_distribution": subset_distribution(records, pass_1) if not n_unparsed else None,
        "seed_types": seed_types(records, pass_1) if not n_unparsed else None,
        "low_confidence": low_confidence_counts(parsed), "verdict": verdict(gates, step),
        "rows": rows, "estimate_label": jt.STRATIFIED_LABEL,
        "note": "one acceptance look per grader (pre-registration section 7, decision D-T1)",
    }


# --- Main ------------------------------------------------------------------------------------

def _roles_for(model_roles: Any, name: str) -> Mapping[str, Any] | None:
    if model_roles is None:
        return None
    return model_roles(name) if callable(model_roles) else model_roles


def main(argv: Sequence[str] | None = None, *, model_roles: Any = None, out_dir: pathlib.Path = OUT_DIR,
         records_dir: pathlib.Path = RECORDS_DIR, cases_file: pathlib.Path = CASES_FILE,
         judge_dev_dir: pathlib.Path = JUDGE_DEV_OUT, log_check: Callable[[Any, int], Any] | None = None) -> int:
    """model_roles (tests only): a mapping, or a callable from pass name to a mapping, of
    offline (mockllm) role models; the look is then offline. log_check (tests only): replaces
    the log check; live, it is grader.check_grader_log."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=int, required=True, choices=roster.GRADER_STEPS)
    ap.add_argument("--permutation-seed", type=int, required=True,
                    help="seed of the one random option permutation, recorded before the look")
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
                    help=f"seed of the alpha unit bootstrap (default {DEFAULT_BOOTSTRAP_SEED})")
    ap.add_argument("--resamples", type=int, default=jt.DEFAULT_BOOTSTRAP_RESAMPLES)
    ap.add_argument("--live", action="store_true", help="make the 280 grader calls (keyed, paid)")
    ap.add_argument("--log-dir", default=None, help="folder for the Inspect logs, outside the repository")
    a = ap.parse_args(argv)
    offline = model_roles is not None
    check = log_check or (offline_log_check if offline else grader.check_grader_log)
    try:
        check_look_order(a.step, out_dir, judge_dev_dir)
        cases, records = load_cases(cases_file), load_records(records_dir)
        check_subset(records, cases)
        specs = plan_passes(a.permutation_seed)
        sizes = [len(m["content"]) for s in specs for r in records for m in record_messages(r, cases, s.settings)]
    except (Refused, jt.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if a.resamples < 1:
        print("refused: --resamples must be positive", file=sys.stderr)
        return EXIT_ARGS
    perm = jt.permuted_options(list(gp.LEVELS), a.permutation_seed)
    if not a.live:
        print(f"acceptance look plan, step {a.step}: {len(specs)} passes of {len(records)} calls, grader "
              f"{grader.grader_pin(a.step)} with {roster.grader_generate_config_args(a.step)}, cache off, prompt "
              f"{gp.PROMPT_VERSION} ({gp.prompt_sha256()[:12]}), about {sum(sizes) // 4 // (len(specs) * len(records))} input "
              f"tokens per call")
        print(f"permutation seed {a.permutation_seed}: option order {perm['order']}"
              + (" (identity draw: it stands if the look runs)" if perm["is_identity"] else "")
              + (" (reversal draw: it stands if the look runs)" if perm["is_reversal"] else "")
              + f"; bootstrap seed {a.bootstrap_seed}, {a.resamples} resamples")
        for s in specs:
            print(f"  {s.number}. {s.name}: {s.settings_record()}")
        if a.step > 1:
            print("before this look: the intra-rater check of section 7 has run once, before the first grader change")
        print("dry run: no call made (pass --live with --log-dir to make the calls)")
        return EXIT_OK
    if not a.log_dir or jr._inside_repo(a.log_dir):
        print("refused: --live needs --log-dir outside the repository", file=sys.stderr)
        return EXIT_ARGS
    logs: dict[str, Any] = {}
    configs: dict[str, dict] = {}
    checks: dict[str, Any] = {}
    for s in specs:
        where = f"pass {s.number} of {len(specs)} ({s.name})"
        try:
            log = run_pass(s, records, cases, step=a.step, permutation_seed=a.permutation_seed,
                           bootstrap_seed=a.bootstrap_seed, log_dir=a.log_dir,
                           model_roles=_roles_for(model_roles, s.name))
            check(log, a.step)
            check_pass_samples(log, records)
            configs[s.name] = pass_config(log, offline)
        except Exception as e:  # noqa: BLE001 - any failure of a pass writes nothing
            print(f"FAIL: {where}: {e}", file=sys.stderr)
            print("nothing written; the look file is written only after every pass ran and every log passed "
                  "its check", file=sys.stderr)
            return EXIT_CHECK
        logs[s.name] = log
        name = "grader.check_grader_log" if check is grader.check_grader_log else getattr(check, "__name__", "log check")
        checks[s.name] = f"{name} passed; {len(grader.grader_events(log))} grader calls, one per output"
    try:
        res = look_result(records, specs, logs, configs, step=a.step, permutation_seed=a.permutation_seed,
                          bootstrap_seed=a.bootstrap_seed, resamples=a.resamples, offline=offline, log_checks=checks)
    except (jt.Refused, roster.RunConfigError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        print("nothing written", file=sys.stderr)
        return EXIT_CHECK
    out_dir.mkdir(parents=True, exist_ok=True)
    target = look_file(out_dir, a.step)
    if target.exists():
        print(f"refused: {target.name} appeared during the look; it is never overwritten", file=sys.stderr)
        return EXIT_REFUSED
    target.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(summary_lines(res)))
    print(target)
    return EXIT_OK if res["verdict"]["overall"] == "PASS" else EXIT_GATE_FAIL


if __name__ == "__main__":
    sys.exit(main())
