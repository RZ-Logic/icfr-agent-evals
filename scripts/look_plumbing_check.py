"""A parse-only plumbing check of the four perturbed grader prompt settings.

Before an acceptance look, this checks that the ladder step's grader answers each
perturbed prompt of the bias audit in a form grader_prompt.parse_reply reads: the rubric
options reversed, the options in the one permutation drawn from the fixed seed
(casefile.split_seed("calibration")), the score labels shown as D, C, B, A, and the two
sub-dimensions swapped. It sends each setting on 3 judge-development outputs (dev split
only), 12 grader calls in one Inspect eval through grader.grade, and reports only whether
each reply parses.

It writes no grade anywhere in the repository and prints none: no level, flag or reason.
The Inspect log goes to --log-dir, outside the repository. A calibration record or case is
refused: the 40-case calibration subset is scored only at the acceptance look
(pre-registration section 7), and this check never reads its records.

Run:
  python scripts/look_plumbing_check.py --step 1                        (plan only, no call)
  python scripts/look_plumbing_check.py --step 1 --live --log-dir DIR   (keyed: 12 grader calls)

Exit codes: 0 plan printed or every reply parses, 1 the log check failed or a reply does
not parse, 2 bad arguments, 3 refused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader, roster  # noqa: E402
from icfr_agent_evals import grader_prompt as gp  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


al = _load("acceptance_look")
jr = al.jr

OUTPUTS = 3
SPLIT = "dev"
EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED = 0, 1, 2, 3


class Refused(Exception):
    pass


def settings() -> list:
    """The four perturbed settings of the look, in its run order, with the fixed seed."""
    return [s for s in al.plan_passes() if s.name in al.BIAS_NAMES]


def calibration_ids(cases_file: pathlib.Path = al.CASES_FILE) -> set[str]:
    """The calibration case ids (ids only), so a calibration output is refused."""
    if not cases_file.is_file():
        return set()
    return {json.loads(line)["id"] for line in cases_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def pick_records(records_dir: pathlib.Path, dev_file: pathlib.Path, names: Sequence[str] | None = None, *,
                 cases_file: pathlib.Path = al.CASES_FILE) -> tuple[list[dict], dict[str, dict]]:
    """Three judge-development records (the first three by file name, or the ones named) and
    the dev cases. Refused: a folder inside data/calibration, a record whose case is a
    calibration case, or a case that is not in the dev split."""
    if al._inside(records_dir, al.CALIBRATION_DIR) or al._inside(dev_file, al.CALIBRATION_DIR):
        raise Refused("calibration inputs are refused; the check runs on judge-development outputs only")
    try:
        records = jr.load_records(records_dir)
    except jr.Refused as e:
        raise Refused(str(e)) from e
    dev = jr.load_dev(dev_file)
    if names:
        by = {r["_file"]: r for r in records}
        if len(set(names)) != OUTPUTS or any(n not in by for n in names):
            raise Refused(f"--record names {OUTPUTS} distinct judge-development records")
        chosen = [by[n] for n in names]
    else:
        chosen = records[:OUTPUTS]
    cal = calibration_ids(cases_file)
    for r in chosen:
        if r.get("sample_id") in cal:
            raise Refused(f"{r['_file']}: case {r.get('sample_id')} is a calibration case; calibration inputs are refused")
        case = dev.get(r.get("sample_id"))
        if case is None or (case.get("metadata") or {}).get("split") != SPLIT:
            raise Refused(f"{r['_file']}: not a dev-split case")
    return chosen, dev


def sample_id(setting: str, record: Mapping[str, Any]) -> str:
    return f"{setting}--{record['_file']}"


def check_task(records: Sequence[Mapping[str, Any]], dev: Mapping[str, dict], step: int):
    """One Inspect task: per setting and record, one sample and one grader call through
    grader.grade."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="look_plumbing_grade")
    def look_plumbing_grade():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            state.output = await grader.grade(state.messages, step)
            return state

        return solve

    specs = settings()
    samples = [Sample(id=sample_id(s.name, r), input=gp.to_chat_messages(jr.record_messages(r, dev, s.settings)),
                      metadata={"split": SPLIT, "setting": s.name, "record": r["_file"]})
               for s in specs for r in records]
    metadata = {"split": SPLIT, "look_plumbing_check": True, "parse_only": True, "grader_step": step,
                "settings": {s.name: s.settings_record() for s in specs},
                "permutation_seed": al.PERMUTATION_SEED, "permutation_seed_rule": al.PERMUTATION_SEED_RULE,
                "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
                "roster": roster.roster_metadata(step)}
    return Task(dataset=MemoryDataset(samples, name="look_plumbing_check"), solver=[look_plumbing_grade()],
                epochs=1, config=GenerateConfig(), metadata=metadata, name="look_plumbing_check")


def run_check(records, dev, step: int, log_dir: str, model_roles: Mapping[str, Any] | None = None) -> Any:
    from inspect_ai import eval as inspect_eval

    roles = dict(model_roles) if model_roles is not None else {roster.GRADER_ROLE: grader.grader_pin(step)}
    return inspect_eval(check_task(records, dev, step), model=grader.NO_MODEL, model_roles=roles, epochs=1,
                        log_dir=log_dir, display="none")[0]


def parse_report(log: Any, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per setting and record: whether the reply parses, and if not, why. The parsed levels,
    flags and reasons are dropped here and never returned."""
    by_id = {str(s.id): s for s in log.samples}
    out = []
    for s in settings():
        for r in records:
            sample = by_id.get(sample_id(s.name, r))
            reply = (sample.output.completion if sample is not None and sample.output else "") or ""
            try:
                if sample is None or sample.error is not None:
                    raise gp.GraderReplyError("no reply (sample missing or ended in an error)")
                gp.parse_reply(reply, s.settings)
                out.append({"setting": s.name, "record": r["_file"], "parses": True})
            except gp.GraderReplyError as e:
                out.append({"setting": s.name, "record": r["_file"], "parses": False, "why": str(e)})
    return out


def main(argv: Sequence[str] | None = None, *, model_roles: Mapping[str, Any] | None = None,
         records_dir: pathlib.Path = jr.JUDGE_DEV_DIR, dev_file: pathlib.Path = jr.DEV_FILE,
         cases_file: pathlib.Path = al.CASES_FILE) -> int:
    """model_roles (tests only): offline (mockllm) role models; the log check is then the
    offline one. records_dir, dev_file and cases_file (tests only) replace the inputs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=int, required=True, choices=roster.GRADER_STEPS)
    ap.add_argument("--record", action="append", default=None,
                    help=f"a judge-development record file name (stem); give {OUTPUTS}, or none for the first {OUTPUTS}")
    ap.add_argument("--records-dir", default=None, help="the judge-development records folder (default data/dev/judge-dev)")
    ap.add_argument("--live", action="store_true", help=f"make the {OUTPUTS * 4} grader calls (keyed)")
    ap.add_argument("--log-dir", default=None, help="folder for the Inspect log, outside the repository")
    a = ap.parse_args(argv)
    folder = pathlib.Path(a.records_dir) if a.records_dir else records_dir
    try:
        records, dev = pick_records(folder, dev_file, a.record, cases_file=cases_file)
        specs = settings()
    except (Refused, al.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"look plumbing check, step {a.step}: grader {grader.grader_pin(a.step)}, {len(specs)} settings x "
          f"{len(records)} judge-development outputs = {len(specs) * len(records)} calls, parse only")
    for s in specs:
        print(f"  {s.name}: {s.settings_record()}")
    print(f"  outputs: {', '.join(r['_file'] for r in records)}")
    if not a.live:
        print("dry run: no call made (pass --live with --log-dir to make the calls)")
        return EXIT_OK
    if not a.log_dir or al._inside(a.log_dir, REPO):
        print("refused: --live needs --log-dir outside the repository", file=sys.stderr)
        return EXIT_ARGS
    log = None
    try:
        log = run_check(records, dev, a.step, a.log_dir, model_roles)
        view, skipped = al.check_pass_samples(log, [{"_file": sample_id(s.name, r)} for s in specs for r in records])
        (al.offline_log_check if model_roles is not None else grader.check_grader_log)(view, a.step)
    except Exception as e:  # noqa: BLE001 - any failure of the run or its log check
        print(f"FAIL: the log check: {e}", file=sys.stderr)
        return EXIT_CHECK
    report = parse_report(log, records)
    for row in report:
        print(f"  {row['setting']} {row['record']}: " + ("parses" if row["parses"] else f"does not parse ({row['why']})"))
    bad = [row for row in report if not row["parses"]]
    if skipped:
        print(f"{skipped} errored grader calls skipped (each followed by a completed call)")
    print(f"{len(report) - len(bad)} of {len(report)} replies parse" + ("" if not bad else "; fix the plumbing before the look"))
    return EXIT_OK if not bad else EXIT_CHECK


if __name__ == "__main__":
    sys.exit(main())
