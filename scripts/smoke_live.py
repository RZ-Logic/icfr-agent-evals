"""Owner-run live smoke: one dev-split generation per model. KEYED; makes paid calls.

Fills pre-registration section 13 row 212 (the grader alias's resolved model string
seen at the tag-time live call) and row 220 (one live single-sample call per roster
model, on a dev-split sample, on the code of the commit to be tagged). Run it from the
repository root in the owner's keyed tab, after scripts/preflight.py passes on that
commit. It never runs on the evaluation split: the sample is a case of the dev pilot
subset, and any other id is refused before a model is built.

Per model:
  - an evaluated model (roster.EVALUATED_MODELS) runs the pilot task itself,
    variance_commentary(split="dev", subset="pilot", epochs=1), on the one sample,
    with the grader and calibrator roles bound to their step-1 pins, so the roster
    check and the serving guard run exactly as in the pilot (the roles are bound,
    never called);
  - any other model (the grader GPT-6 Luna, the escalation grader GPT-6.1 Sol, or a
    model the owner names, D15) fails the roster check as a worker, so it runs a
    one-sample task under the same sample id with a fixed one-line prompt that holds
    no case content, after roster.model_serving checks its endpoint.

It prints, per model, the resolved model string read from the log, the endpoint the
serving guard checked, the token usage and the log path; never a key value. Exit 0
when every model answered, 1 otherwise, 2 on bad arguments.

Run: python scripts/smoke_live.py --log-dir <folder outside the repo>
     [--models a/b,c/d] [--sample-id <dev pilot case id>]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from icfr_agent_evals import roster

FIXED_PROMPT = "Reply with the single word OK."
SPLIT, SUBSET = "dev", "pilot"


def default_models() -> list[str]:
    """Every distinct model of the roster's worker, baselines, grader, escalation
    grader and calibrator, in that order (D15 reading: six models)."""
    ordered = [roster.WORKER, *roster.BASELINES, roster.GRADER, roster.GRADER_ESCALATION, roster.CALIBRATOR]
    return list(dict.fromkeys(ordered))


def parse_models(value: str) -> list[str]:
    """A comma list of provider/name strings, stripped and deduplicated in order."""
    models = list(dict.fromkeys(m.strip() for m in value.split(",") if m.strip()))
    if not models:
        raise argparse.ArgumentTypeError("--models needs at least one provider/name")
    bad = [m for m in models if "/" not in m]
    if bad:
        raise argparse.ArgumentTypeError(f"not provider/name: {bad}")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Owner-run live smoke (keyed, paid): one dev-split generation per model, "
                    "printing the resolved model string read from each log.",
    )
    parser.add_argument("--models", type=parse_models, default=None,
                        help="comma list of provider/name (default: every distinct roster model)")
    parser.add_argument("--sample-id", default=None,
                        help="a case id of the dev pilot subset (default: the first in id order)")
    parser.add_argument("--log-dir", required=True, help="folder for the smoke logs, one subfolder per model")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.models is None:
        args.models = default_models()
    return args


def pilot_ids() -> list[str]:
    """The dev pilot case ids in id order. Builds the dev split only."""
    from icfr_agent_evals.dataset import build_dataset, select_subset

    return sorted(str(s.id) for s in select_subset(build_dataset(split=SPLIT), SPLIT, SUBSET))


def resolve_sample_id(requested: str | None, ids: list[str]) -> str:
    """The requested id when it is a dev pilot case, else ValueError; None picks the first."""
    if requested is None:
        return ids[0]
    if requested not in ids:
        raise ValueError(f"--sample-id {requested!r} is not a case of the dev pilot subset")
    return requested


def _log_folder(log_dir: str, model: str) -> str:
    return os.path.join(log_dir, model.replace("/", "_"))


def _direct_task(sample_id: str):
    """One sample, a fixed prompt, no case content; the serving guard runs at setup."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import transcript
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.solver import Generate, TaskState, generate, solver

    @solver(name="serving_check")
    def serving_check():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            model = get_model()
            if not roster._is_offline(str(model)):
                endpoints = roster.serving_endpoints({roster.EVALUATED_KEY: model})
                transcript().info({"serving_endpoints": endpoints}, source=roster.SERVING_EVENT_SOURCE)
            return state

        return solve

    return Task(
        dataset=MemoryDataset([Sample(id=sample_id, input=FIXED_PROMPT, metadata={"split": SPLIT})],
                              name="smoke_live_direct"),
        setup=serving_check(),
        solver=[generate()],
        epochs=1,
        config=GenerateConfig(),
        metadata={"split": SPLIT, "smoke": "live_direct"},
    )


def _endpoint(sample: Any) -> str:
    for event in sample.events or []:
        if getattr(event, "source", None) == roster.SERVING_EVENT_SOURCE:
            data = getattr(event, "data", None) or {}
            entry = data.get("serving_endpoints", {}).get(roster.EVALUATED_KEY, {})
            return str(entry.get("base_url", "not recorded"))
    return "not recorded (offline)"


def smoke_one(model: str, sample_id: str, log_dir: str) -> tuple[bool, str]:
    """(answered, report line) for one model's single generation."""
    from inspect_ai import eval as inspect_eval

    evaluated = model in roster.EVALUATED_MODELS
    if evaluated:
        from icfr_agent_evals.task import variance_commentary

        task = variance_commentary(split=SPLIT, subset=SUBSET, epochs=1)
        roles: dict[str, str] | None = roster.role_pins(1)
    else:
        task = _direct_task(sample_id)
        roles = None
    log = inspect_eval(task, model=model, model_roles=roles, sample_id=sample_id, epochs=1,
                       log_dir=_log_folder(log_dir, model), display="none")[0]
    path = "pilot task" if evaluated else "fixed prompt"
    if log.status != "success" or not log.samples:
        return False, f"FAIL {model} ({path}): status {log.status}; log {log.location}"
    sample = log.samples[0]
    output = sample.output
    completion = (output.completion or "").strip() if output else ""
    if sample.error is not None or not completion:
        return False, f"FAIL {model} ({path}): no answer; log {log.location}"
    usage = output.usage
    used = f"in {usage.input_tokens} out {usage.output_tokens}" if usage else "not recorded"
    return True, (f"PASS {model} ({path}): resolved {output.model}; endpoint {_endpoint(sample)}; "
                  f"usage {used}; log {log.location}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parse_args(argv)
    try:
        sample_id = resolve_sample_id(args.sample_id, pilot_ids())
    except ValueError as e:
        parser.error(str(e))
    print(f"live smoke on dev pilot case {sample_id} ({len(args.models)} models)")
    ok = True
    for model in args.models:
        try:
            answered, line = smoke_one(model, sample_id, args.log_dir)
        except Exception as e:  # one model's failure must not hide the others
            answered, line = False, f"FAIL {model}: {type(e).__name__}: {e}"
        ok = ok and answered
        print(line)
    print("LIVE SMOKE RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
