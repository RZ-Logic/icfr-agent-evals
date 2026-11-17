"""One round of grader development on the judge-development set (judge remedy order step 1).

Pre-registration section 7, step (1): "Iterate the grader prompt and rubric wording for
at most 3 rounds on a judge-development set of 20 dev-split worker outputs, graded by the
author against the rubric's 0 to 3 level anchors ... It stops when every LLM-graded
sub-dimension's ICC(A,1) point estimate is at least 0.78 and leniency (the signed mean
of grader minus author score) is within plus or minus 0.25 on the 20-output
judge-development set, or after 3 rounds ... Each round targets a named disagreement
pattern on a sub-dimension that misses the stop criteria, found in that sub-dimension's
cases where author and grader differ by 2 or more points, or, when it has no such case,
its most frequent named 1-point disagreement pattern; every round, a fallback round
included, counts toward the 3-round cap and is logged with the pattern it targets and
its statistics. These statistics are diagnostics, not gates."

Round 0 grades the 20 outputs with the prompt as first written; it targets no pattern.
Rounds 1 to 3 each follow an edit of the prompt or rubric wording made for the pattern
given with --pattern, and must run on wording that differs from every earlier round. A
round after one that met the stop criteria is refused, and so is a fourth round.

Each output is graded by GPT-6 Luna (ladder step 1) through icfr_agent_evals.grader with
the pinned configuration and caching off, on the prompt of icfr_agent_evals.grader_prompt:
the narrative, the shared facts sheet (decision D-R10 C7), the case file and the
committed scope and level texts. Dev split only: no evaluation-split or calibration
case is read. The 40-case calibration subset is never used here.

Without --live nothing is called: the script prints the plan and the prompt size.
With --live (the author's key in the environment) it runs one Inspect eval of 20 grader
calls, checks the log with grader.check_grader_log (pin, configuration, resolved model
string), parses each reply, and writes docs/evidence/judge-dev/round-<N>.json: per output
the author's grades, the grader's levels, low-confidence flags and reasons; per
sub-dimension ICC(A,1) with its McGraw-Wong interval, ICC(C,1) and leniency with its
paired interval; whether the stop criteria are met; and the disagreements a next round
would target. A reply that does not parse leaves its output ungraded and the stop
criteria unmet. An existing round file is never overwritten.

Run:
  python scripts/judge_dev_round.py --round 0                          (plan only, no call)
  python scripts/judge_dev_round.py --round 0 --live --log-dir DIR     (paid: 20 grader calls)
  python scripts/judge_dev_round.py --round 1 --pattern "grounding: ..." --live --log-dir DIR

Exit codes: 0 done (or plan printed), 1 a log or reply check failed, 2 bad arguments,
3 refused (round order, cap, overwrite, records).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from fractions import Fraction
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader, grader_prompt as gp, roster, rubric, scoring  # noqa: E402
from icfr_agent_evals import judge_trust as jt  # noqa: E402
from icfr_agent_evals import review_facts as rf  # noqa: E402

JUDGE_DEV_DIR = REPO / "data" / "dev" / "judge-dev"
DEV_FILE = REPO / "data" / "dev" / "dev.jsonl"
OUT_DIR = REPO / "docs" / "evidence" / "judge-dev"
STEP = 1
MAX_ROUND = 3
ICC_STOP = Fraction(78, 100)
LENIENCY_STOP = Fraction(1, 4)
EXPECTED_RECORDS = 20
EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED = 0, 1, 2, 3


class Refused(Exception):
    pass


def load_records(folder: pathlib.Path = JUDGE_DEV_DIR) -> list[dict]:
    """The committed judge-development records, sorted by file name."""
    recs = []
    for p in sorted(folder.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_file"] = p.stem
        recs.append(r)
    if len(recs) != EXPECTED_RECORDS:
        raise Refused(f"{len(recs)} judge-development records, not {EXPECTED_RECORDS}")
    for r in recs:
        if set(r.get("grades") or {}) != set(gp.DIMENSIONS):
            raise Refused(f"{r['_file']}: the author's grades do not cover both sub-dimensions")
        if r.get("anchors_sha256") != rubric.llm_anchor_sha256():
            raise Refused(f"{r['_file']}: graded against anchors {r.get('anchors_sha256')!r}, not the committed "
                          f"{rubric.llm_anchor_sha256()!r}")
    return recs


def load_dev(path: pathlib.Path = DEV_FILE) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def record_messages(record: Mapping[str, Any], dev: Mapping[str, dict],
                    settings: gp.PromptSettings = gp.PromptSettings()) -> list[dict]:
    """The grader's messages for one record: what the author saw when grading it."""
    case = dev.get(record["sample_id"])
    if case is None:
        raise Refused(f"{record['_file']}: case {record['sample_id']} is not in the dev split")
    if case["metadata"].get("split") != "dev":
        raise Refused(f"{record['_file']}: not a dev-split case")
    parse = scoring.parse_worker_output(record["worker_output"])
    if not parse.ok:
        raise Refused(f"{record['_file']}: the output does not parse (it is never graded, D-R10 C5)")
    meta = case["metadata"]
    return gp.build_messages(str(parse.parsed["narrative"]), case["input"], rf.case_facts(meta),
                             rf.output_facts(meta, parse.parsed), settings)


def round_task(records: Sequence[Mapping[str, Any]], dev: Mapping[str, dict]):
    """An Inspect task: one sample per record, one grader call each through grader.grade."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="judge_dev_grade")
    def judge_dev_grade():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            state.output = await grader.grade(state.messages, STEP)
            return state

        return solve

    samples = [Sample(id=r["_file"], input=gp.to_chat_messages(record_messages(r, dev)), metadata={"split": "dev"})
               for r in records]
    return Task(dataset=MemoryDataset(samples, name="judge_dev_round"), solver=[judge_dev_grade()], epochs=1,
                config=GenerateConfig(),
                metadata={"split": "dev", "judge_dev_round": True, "grader_prompt_version": gp.PROMPT_VERSION,
                          "grader_prompt_sha256": gp.prompt_sha256(), "roster": roster.roster_metadata(STEP)})


def run_round(records, dev, log_dir: str, model_roles: dict[str, str] | None = None) -> Any:
    from inspect_ai import eval as inspect_eval

    roles = model_roles if model_roles is not None else {roster.GRADER_ROLE: grader.grader_pin(STEP)}
    return inspect_eval(round_task(records, dev), model=grader.NO_MODEL, model_roles=roles, epochs=1,
                        log_dir=log_dir, display="none")[0]


def dim_stats(author: Sequence[int], graded: Sequence[int]) -> dict[str, Any]:
    """ICC(A,1), ICC(C,1) and leniency (grader minus author) on the graded pairs, and the
    stop test: ICC(A,1) point estimate at least 0.78 and |leniency| at most 0.25, exact."""
    matrix = [[a, g] for a, g in zip(author, graded)]
    icc = jt.icc_a1(matrix)
    iccc = jt.icc_c1(matrix)
    len_ = jt.paired_mean_difference([(g, a) for a, g in zip(author, graded)])
    est = Fraction(icc["estimate_exact"]) if icc.get("defined") and icc.get("estimate_exact") else None
    lean = Fraction(len_["mean_exact"])
    met = est is not None and est >= ICC_STOP and abs(lean) <= LENIENCY_STOP
    return {"icc_a1": icc, "icc_c1": iccc, "leniency": len_, "stop_met": met}


def disagreements(rows: Sequence[Mapping[str, Any]], dim: str) -> dict[str, Any]:
    """The cases a next round targets on a sub-dimension: those 2 or more points apart, or,
    when none is, every 1-point disagreement (the analyst names the most frequent pattern)."""
    diffs = [r for r in rows if r["grader"].get(dim) is not None and r["grader"][dim] != r["author"][dim]]
    two = [r for r in diffs if abs(r["grader"][dim] - r["author"][dim]) >= 2]
    pick = two or diffs
    return {"basis": "2+ points" if two else "1 point", "count_2plus": len(two), "count_1": len(diffs) - len(two),
            "cases": [{"record": r["record"], "author": r["author"][dim], "grader": r["grader"][dim],
                       "low_confidence": r["low_confidence"].get(dim), "reason": r["reasons"].get(dim, "")}
                      for r in pick]}


def round_result(records, log: Any, round_n: int, pattern: str | None, offline: bool) -> dict[str, Any]:
    if not offline:
        grader.check_grader_log(log, STEP)
    by_id = {str(s.id): s for s in log.samples}
    rows, unparsed = [], []
    for r in records:
        s = by_id.get(r["_file"])
        reply = (s.output.completion if s is not None and s.output else "") or ""
        try:
            if s is None or s.error is not None:
                raise gp.GraderReplyError("no reply (sample missing or ended in an error)")
            parsed = gp.parse_reply(reply)
            grades = {d: parsed[d]["level"] for d in gp.DIMENSIONS}
            low = {d: parsed[d]["low_confidence"] for d in gp.DIMENSIONS}
            reasons = {d: parsed[d]["reason"] for d in gp.DIMENSIONS}
        except gp.GraderReplyError as e:
            unparsed.append({"record": r["_file"], "error": str(e)})
            grades = {d: None for d in gp.DIMENSIONS}
            low, reasons = {}, {}
        rows.append({"record": r["_file"], "author": dict(r["grades"]), "grader": grades, "low_confidence": low,
                     "reasons": reasons, "reply": reply})
    stats = {}
    for d in gp.DIMENSIONS:
        pairs = [(row["author"][d], row["grader"][d]) for row in rows if row["grader"][d] is not None]
        stats[d] = dim_stats([a for a, _ in pairs], [g for _, g in pairs]) if len(pairs) >= 2 else None
        stats[d] = {**(stats[d] or {"stop_met": False}), "n_graded": len(pairs),
                    "disagreements": disagreements(rows, d)}
    stop = not unparsed and all(stats[d]["stop_met"] for d in gp.DIMENSIONS)
    log_path = pathlib.Path(str(log.location)) if getattr(log, "location", None) else None
    return {
        "round": round_n, "targets_pattern": pattern, "offline": offline,
        "grader_step": STEP, "grader_model": grader.grader_pin(STEP),
        "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
        "anchors_sha256": rubric.llm_anchor_sha256(),
        "log_file": log_path.name if log_path else None,
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path and log_path.is_file() else None,
        "records": len(records), "unparsed": unparsed, "stats": stats, "stop_criteria_met": stop,
        "rows": rows,
        "note": "Diagnostics, not gates (pre-registration section 7, judge remedy order step 1).",
    }


def check_round_order(round_n: int, pattern: str | None, out_dir: pathlib.Path) -> None:
    """Round 0 first; rounds 1 to 3 each after the one before, on new wording, naming a
    pattern; none after a round that met the stop criteria; never an overwrite."""
    if not 0 <= round_n <= MAX_ROUND:
        raise Refused(f"round {round_n} is outside 0 to {MAX_ROUND} (at most 3 rounds after the first grading)")
    if (out_dir / f"round-{round_n}.json").exists():
        raise Refused(f"round-{round_n}.json exists; a round is never re-run or overwritten")
    if round_n == 0:
        if pattern:
            raise Refused("round 0 grades the first wording and targets no pattern")
        return
    if not pattern or not pattern.strip():
        raise Refused("rounds 1 to 3 must name the disagreement pattern they target (--pattern)")
    earlier = []
    for n in range(round_n):
        p = out_dir / f"round-{n}.json"
        if not p.exists():
            raise Refused(f"round {n} has not run; rounds run in order")
        earlier.append(json.loads(p.read_text(encoding="utf-8")))
    if any(e.get("stop_criteria_met") for e in earlier):
        raise Refused("an earlier round met the stop criteria; step 1 has stopped")
    if any(e.get("grader_prompt_sha256") == gp.prompt_sha256() for e in earlier):
        raise Refused("the grader prompt wording is the same as an earlier round's; edit it for the pattern first")


def summary_lines(res: Mapping[str, Any]) -> list[str]:
    lines = [f"round {res['round']}: prompt {res['grader_prompt_version']} ({res['grader_prompt_sha256'][:12]}), "
             f"{res['records']} outputs, {len(res['unparsed'])} unparsed"]
    for d in gp.DIMENSIONS:
        s = res["stats"][d]
        icc = s.get("icc_a1") or {}
        lean = s.get("leniency") or {}
        lines.append(f"  {d}: ICC(A,1) {icc.get('estimate')} [{icc.get('ci_lower')}, {icc.get('ci_upper')}], "
                     f"leniency {lean.get('mean')}, stop {'met' if s['stop_met'] else 'not met'}; "
                     f"disagreements {s['disagreements']['count_2plus']} at 2+ points, "
                     f"{s['disagreements']['count_1']} at 1 point")
    lines.append("stop criteria " + ("MET: step 1 ends" if res["stop_criteria_met"] else "not met"))
    return lines


def _inside_repo(path: str) -> bool:
    target = pathlib.Path(path).resolve()
    return target == REPO or REPO in target.parents


def main(argv: Sequence[str] | None = None, *, model_roles: dict[str, str] | None = None,
         out_dir: pathlib.Path = OUT_DIR) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--pattern", default=None, help="rounds 1 to 3: the disagreement pattern this round's edit targets")
    ap.add_argument("--live", action="store_true", help="make the 20 grader calls (keyed, paid)")
    ap.add_argument("--log-dir", default=None, help="folder for the Inspect log, outside the repository")
    a = ap.parse_args(argv)
    offline = model_roles is not None
    try:
        check_round_order(a.round, a.pattern, out_dir)
        records, dev = load_records(), load_dev()
        sizes = [len(m["content"]) for r in records for m in record_messages(r, dev)]
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if not a.live:
        print(f"round {a.round} plan: {len(records)} judge-development outputs, grader {grader.grader_pin(STEP)} "
              f"with {roster.grader_generate_config_args(STEP)}, cache off, prompt {gp.PROMPT_VERSION} "
              f"({gp.prompt_sha256()[:12]}), about {sum(sizes) // 4 // len(records)} input tokens per call")
        print("dry run: no call made (pass --live with --log-dir to make the calls)")
        return EXIT_OK
    if not a.log_dir or _inside_repo(a.log_dir):
        print("refused: --live needs --log-dir outside the repository", file=sys.stderr)
        return EXIT_ARGS
    log = run_round(records, dev, a.log_dir, model_roles)
    try:
        res = round_result(records, log, a.round, a.pattern, offline)
    except roster.RunConfigError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return EXIT_CHECK
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"round-{a.round}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n",
                                                   encoding="utf-8", newline="\n")
    print("\n".join(summary_lines(res)))
    print(out_dir / f"round-{a.round}.json")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
