"""First live check of the grader configuration. KEYED with --live; makes paid calls.

Pre-registration section 5: "The grader configuration is first checked live by the
grader build (decision D-R12), after the tag and before any evaluation-split run."
Section 13 lists the grader call and this check as tooling built after the tag
(decision D-R19, item 2).

Without --live (the default) it makes no call and builds no model. It prints, per
requested ladder step, the model string it would call, the GenerateConfig it would
send (roster.grader_generate_config_args) and the tag-time resolved string it
compares against, then exits 0.

With --live it makes one call per requested step, through icfr_agent_evals.grader,
on a fixed one-line prompt with no case content. Each call runs as a one-sample
Inspect eval whose evaluated model is Inspect's none provider (none/none, so no
other model is called) with the grader role bound to the step's pin, and writes its
log under --log-dir. Each log is then checked:
  - the grader branch of roster.check_run_config, through grader.check_grader_log:
    reasoning_effort high, no sampling parameter set, no max_tokens, no extra
    headers, batch True at steps 2 and 3 and unset at step 1, no response cache;
  - exactly one model call, made by the grader role to the step's pin;
  - the call answered;
  - the resolved model string equals the one recorded at the tag-time live call
    (grader.TAG_TIME_RESOLVED). Section 5: any change is a vendor-change event.
It prints PASS or FAIL per step with the resolved string, the endpoint the serving
guard checked, the token usage and the log path. Steps 2 and 3 go through the OpenAI
Batch API, so each can wait on the batch for minutes or longer.

It never reads, prints or sets a key. Before each live step it prints the names of
the request-header sources in force (roster.extra_headers_in_force()), never a value;
the serving guard refuses the grader when any is in force.

A live-check log is a record of the grader configuration only. It holds no case and
no score, and roster.check_run_config refuses it (no evaluated-model call): never
pass it to scripts/pilot_report.py, scripts/compare_runs.py or any other reader.

Exit codes: 0 every requested step passed (or the dry run printed its plan), 1 a step
failed, 2 bad arguments.

Run: python scripts/grader_live_check.py [--steps 1,2,3]
     python scripts/grader_live_check.py --live --log-dir <folder outside the repo> [--steps 1,2,3]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

from icfr_agent_evals import grader, roster

FIXED_PROMPT = "Reply with the single word OK."
SPLIT = "dev"  # a label only: the prompt holds no case
REPO = pathlib.Path(__file__).resolve().parents[1]


def parse_steps(value: str) -> list[int]:
    """A comma list of ladder steps, deduplicated in order; each must be 1, 2 or 3."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--steps needs at least one ladder step")
    steps: list[int] = []
    for part in parts:
        try:
            step = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a ladder step: {part!r}") from None
        if step not in roster.GRADER_STEPS:
            raise argparse.ArgumentTypeError(f"ladder step {step} is not one of {roster.GRADER_STEPS}")
        if step not in steps:
            steps.append(step)
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First live check of the grader configuration. Without --live it makes no call "
                    "and prints what it would call; with --live it makes one paid call per ladder step.",
    )
    parser.add_argument("--steps", type=parse_steps, default=list(roster.GRADER_STEPS),
                        help="comma list of grader ladder steps (default: 1,2,3)")
    parser.add_argument("--live", action="store_true",
                        help="make the calls (keyed, paid); without it nothing is called")
    parser.add_argument("--log-dir", default=None,
                        help="folder for the logs, one subfolder per step, outside the repository (needed with --live)")
    return parser


def _inside_repo(path: str) -> bool:
    target = pathlib.Path(path).resolve()
    return target == REPO or REPO in target.parents


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live:
        if not args.log_dir:
            parser.error("--live needs --log-dir")
        if _inside_repo(args.log_dir):
            parser.error(f"--log-dir {args.log_dir!r} is inside the repository; use a folder outside it")
    return args


def plan(step: int) -> dict[str, Any]:
    """What a live call at a step would use. Pure: builds no model."""
    return {
        "step": step,
        "model": grader.grader_pin(step),
        "config": roster.grader_generate_config_args(step),
        "cache": False,
        "batch_api": step in roster.GRADER_BATCH_STEPS,
        "tag_time_resolved": grader.TAG_TIME_RESOLVED[step],
    }


def plan_line(step: int) -> str:
    p = plan(step)
    return (f"step {p['step']}: would call {p['model']} with GenerateConfig {p['config']}, cache off, "
            f"OpenAI Batch API {'yes' if p['batch_api'] else 'no'}; expects resolved {p['tag_time_resolved']!r}")


def extra_headers_line() -> str:
    """The request-header sources in force, by name only (never a value)."""
    sources = roster.extra_headers_in_force()
    return "extra headers in force: " + (", ".join(sources) if sources else "none")


def live_task(step: int):
    """One sample, the fixed prompt, one grader call through grader.grade."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="grader_live_call")
    def grader_live_call():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            state.output = await grader.grade(state.messages, step)
            return state

        return solve

    return Task(
        dataset=MemoryDataset([Sample(id=f"grader-live-step-{step}", input=FIXED_PROMPT, metadata={"split": SPLIT})],
                              name="grader_live_check"),
        solver=[grader_live_call()],
        epochs=1,
        config=GenerateConfig(),
        metadata={"split": SPLIT, "grader_live_check": step, "roster": roster.roster_metadata(step)},
    )


def _log_folder(log_dir: str, step: int) -> str:
    return os.path.join(log_dir, f"step-{step}")


def run_step(step: int, log_dir: str, model_roles: dict[str, str] | None = None) -> Any:
    """Run the one-call eval for a step and return its EvalLog. model_roles defaults to
    the grader role bound to the step's pin (the offline tests bind mockllm)."""
    from inspect_ai import eval as inspect_eval

    roles = model_roles if model_roles is not None else {roster.GRADER_ROLE: grader.grader_pin(step)}
    return inspect_eval(live_task(step), model=grader.NO_MODEL, model_roles=roles, epochs=1,
                        log_dir=_log_folder(log_dir, step), display="none")[0]


def _endpoint(sample: Any) -> str:
    for event in sample.events or ():
        if getattr(event, "source", None) == grader.GRADER_EVENT_SOURCE:
            data = getattr(event, "data", None) or {}
            return str((data.get("serving_endpoint") or {}).get("base_url", "not recorded"))
    return "not recorded (offline)"


def check_log(log: Any, step: int) -> dict[str, Any]:
    """Raise roster.RunConfigError unless a live-check log passes every check of its step;
    return the summary that is printed."""
    summaries = grader.check_grader_log(log, step)
    calls = [e for s in log.samples for e in (s.events or ()) if getattr(e, "event", None) == "model"]
    if len(calls) != 1 or len(summaries) != 1:
        raise roster.RunConfigError(
            f"live check: expected exactly one model call, by the grader role; the log holds {len(calls)} "
            f"model call(s), {len(summaries)} by the grader"
        )
    sample = log.samples[0]
    if sample.error is not None:
        raise roster.RunConfigError(f"live check: the sample ended in an error: {sample.error.message}")
    completion = (sample.output.completion or "").strip() if sample.output else ""
    if not completion:
        raise roster.RunConfigError("live check: the grader gave no answer")
    summary = summaries[0]
    expected = grader.TAG_TIME_RESOLVED[step]
    if summary["resolved"] != expected:
        raise roster.RunConfigError(
            f"live check: {summary['model']} resolved to {summary['resolved']!r}, not {expected!r} as at the "
            f"tag-time live call; a change is a vendor-change event (pre-registration section 5)"
        )
    summary["endpoint"] = _endpoint(sample)
    return summary


def report_line(step: int, summary: dict[str, Any], location: str) -> str:
    usage = summary["usage"]
    used = f"in {usage[0]} out {usage[1]}" if usage else "not recorded"
    config = summary["config"]
    return (f"PASS step {step} {summary['model']}: resolved {summary['resolved']}; "
            f"reasoning_effort {config.get('reasoning_effort')}; batch {config.get('batch', 'unset')}; "
            f"endpoint {summary['endpoint']}; usage {used}; log {location}")


def live_one(step: int, log_dir: str) -> tuple[bool, str]:
    """(passed, report line) for one step's live call."""
    log = run_step(step, log_dir)
    if log.status != "success":
        message = getattr(getattr(log, "error", None), "message", None)
        return False, f"FAIL step {step}: status {log.status}{'; ' + message if message else ''}; log {log.location}"
    try:
        summary = check_log(log, step)
    except (roster.RosterError, roster.RunConfigError) as e:
        return False, f"FAIL step {step}: {e}; log {log.location}"
    return True, report_line(step, summary, log.location)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.live:
        for step in args.steps:
            print(plan_line(step))
        print("dry run: no call made (pass --live with --log-dir to make the calls)")
        return 0
    print(f"grader live check on a fixed prompt ({len(args.steps)} step(s))")
    ok = True
    for step in args.steps:
        print(f"step {step}: {extra_headers_line()}")
        try:
            passed, line = live_one(step, args.log_dir)
        except Exception as e:  # one step's failure must not hide the others
            passed, line = False, f"FAIL step {step}: {type(e).__name__}: {e}"
        ok = ok and passed
        print(line)
    print("GRADER LIVE CHECK RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
