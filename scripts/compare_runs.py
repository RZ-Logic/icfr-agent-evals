"""Compare candidate runs with the baseline run under the section 10 CI regression gates.

Usage:
  python scripts/compare_runs.py --baseline A.eval --candidate B.eval [--candidate C.eval ...]
                                 [--ladder R1|R2] [--json out.json] [--max-tokens-pin N]

Every --candidate is a run of the same gated content; they pool (section 10(d)).
The ladder defaults to R1, the rule with the most triggers, until the A/A estimate
records the choice (section 10(c)).

Every candidate run must hold the baseline's full sample id set and the same
dataset content, so the comparison never runs on an overlap (ratified 2026-11-03).

The report opens with one header line per run: path, task, model, git revision
and dirty flag, and worker_prompt_sha256. A candidate must run the baseline's task
and worker prompt, and pooled candidates must agree with each other on model, task,
revision and worker prompt (F2-08). Baseline and candidate may differ in model and
revision, so a mock-degraded candidate can be compared with a real baseline. Task
names compare by registry name without the package prefix: the CLI records
"variance_commentary" and an eval() call "icfr_agent_evals/variance_commentary".

Every log must pass the run configuration check (contract C2,
roster.check_run_config through reliability.require_run_config) before any record
is read: status success, no --limit or --sample-id, epochs 4, and plan.config and
every request of the evaluated model on the section 5 pins (reasoning_effort high,
no sampling parameter, reasoning token budget, seed or request header; max_tokens
only at --max-tokens-pin when the owner pins it). The template baseline log is out
of that check's scope; a template candidate is refused by the task match above.

Exit codes (ratified 2026-11-03):
  0  pass: no automatic block and no sign-off case
  1  automatic block: a Holm rejection in 10(a) or a fabrication rejection in 10(b)
  2  no automatic block, sign-off cases pending under 10(c)
  3  refused: an evaluation-split log, a truncated run (--limit, --sample-id, or
     scored samples short of the dataset's count), runs whose split, seed, subset,
     Task version or dataset content (version and hashes) differ, a candidate on
     another task or worker prompt, pooled candidates that differ in model, task,
     revision or worker prompt, a log off the section 5 run configuration, a help request,
     a --json path that cannot be written, or any other input the gates cannot read
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from icfr_agent_evals.regression import (
    FABRICATION_CODES,
    LADDERS,
    FabricationInstance,
    PASS4,
    DIMENSIONS,
    GateReport,
    compare,
    content_fingerprint,
    fabrication_instances,
    load_log,
    records_from_log,
)
from icfr_agent_evals.reliability import PREREGISTERED_K, GateError, require_run_config

EXIT_PASS, EXIT_BLOCK, EXIT_SIGNOFF, EXIT_REFUSED = 0, 1, 2, 3
MATCHED_TASK_ARGS = ("split", "seed", "subset")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 would read as "sign-off pending"
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


class _HelpAction(argparse.Action):
    """-h/--help prints usage and exits 3: argparse's own exit 0 would read as a pass."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any) -> None:
        super().__init__(option_strings, dest=dest, default=argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        parser.print_help()
        print("REFUSED: help requested; no comparison ran", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


# run_header fields: a candidate must match the baseline on BASELINE_MATCHED, and
# pooled candidates must match each other on POOL_MATCHED (section 10(d) pools
# re-runs of one candidate, never two different candidates).
BASELINE_MATCHED = ("task", "worker_prompt_sha256")
POOL_MATCHED = ("model", "task", "revision", "dirty", "worker_prompt_sha256")


def run_header(log: Any) -> dict[str, Any]:
    """Task (registry name without the package prefix), model, git revision and dirty
    flag, and worker_prompt_sha256 from a log header; an absent field is None."""
    spec = log.eval
    task = getattr(spec, "task", None)
    revision = getattr(spec, "revision", None)
    return {
        "task": None if task is None else str(task).rsplit("/", 1)[-1],
        "model": getattr(spec, "model", None),
        "revision": getattr(revision, "commit", None),
        "dirty": getattr(revision, "dirty", None),
        "worker_prompt_sha256": (getattr(spec, "metadata", None) or {}).get("worker_prompt_sha256"),
    }


def format_header(label: str, path: str, header: dict[str, Any]) -> str:
    return (f"{label}: {path} | task {header['task']} | model {header['model']} | revision {header['revision']}"
            f" dirty {header['dirty']} | worker_prompt_sha256 {header['worker_prompt_sha256']}")


def check_run_headers(baseline: Any, candidates: Sequence[Any], paths: Sequence[str]) -> None:
    """F2-08: a candidate on another task or worker prompt is not a regression of the
    baseline, and pooled candidates must be re-runs of one configuration."""
    base = run_header(baseline)
    heads = [run_header(c) for c in candidates]
    for path, head in zip(paths, heads):
        diff = {k: (base[k], head[k]) for k in BASELINE_MATCHED if base[k] != head[k]}
        if diff:
            raise GateError(f"{path} differs from the baseline in {sorted(diff)} (baseline, candidate): {diff}")
    for path, head in zip(paths[1:], heads[1:]):
        diff = {k: (heads[0][k], head[k]) for k in POOL_MATCHED if heads[0][k] != head[k]}
        if diff:
            raise GateError(f"pooled candidate {path} differs from {paths[0]} in {sorted(diff)}"
                            f" (first, this): {diff}; pooled runs must be re-runs of one candidate")


def check_matching_runs(baseline: Any, candidates: Sequence[Any], paths: Sequence[str]) -> None:
    """Paired runs must share split, seed and subset, and the Task version and dataset
    content behind each sample id, or pairing by id compares different cases; the
    run headers must pass check_run_headers."""
    check_run_headers(baseline, candidates, paths)
    base_args = baseline.eval.task_args or {}
    base_content = content_fingerprint(baseline)
    for path, cand in zip(paths, candidates):
        cand_args = cand.eval.task_args or {}
        diff = {k: (base_args.get(k), cand_args.get(k)) for k in MATCHED_TASK_ARGS if base_args.get(k) != cand_args.get(k)}
        if diff:
            raise GateError(f"{path} differs from the baseline in task_args (baseline, candidate): {diff}")
        cand_content = content_fingerprint(cand)
        diff = {k: (v, cand_content[k]) for k, v in base_content.items() if v != cand_content[k]}
        if diff:
            raise GateError(f"{path} differs from the baseline in Task version or dataset content (baseline, candidate): {diff}")


def check_run_config(path: str, log: Any, max_tokens_pin: int | None = None) -> None:
    """Contract C2 on one full log, naming the file in the refusal; the template
    baseline log is skipped (require_run_config)."""
    try:
        require_run_config(log, expected_epochs=PREREGISTERED_K, max_tokens_pin=max_tokens_pin)
    except GateError as e:
        raise GateError(f"{path}: {e}") from e


def _fmt_p(p: float | None) -> str:
    return "-" if p is None else f"{p:.4g}"


def _fmt_directions(counts: dict[str, int]) -> str:
    return ", ".join(f"{d} {n}" for d, n in counts.items()) or "none"


def render(report: GateReport, ladder: str, instances: Sequence[FabricationInstance],
           headers: Sequence[str] = ()) -> str:
    lines = [*headers, "Section 10(a) paired rule (Holm, m = 6)"]
    for name in (*DIMENSIONS, PASS4):
        t = report.paired[name]
        verdict = "BLOCK" if t.rejected else "pass"
        extra = ", ".join(f"{k}={v:.4g}" for k, v in t.detail.items())
        lines.append(f"  {name:<15} p={_fmt_p(t.p):<10} holm p={_fmt_p(t.p_holm):<10} n={t.n:<4} {verdict:<5} {extra}")
    lines.append(f"Section 10(b) fabrication rule (exact McNemar, alpha {report.fabrication[FABRICATION_CODES[0]].alpha})")
    for code in FABRICATION_CODES:
        t = report.fabrication[code]
        verdict = "BLOCK" if t.rejected else "pass"
        lines.append(f"  {code:<15} p={_fmt_p(t.p):<10} n={t.n:<4} {verdict:<5} worse={t.detail['worse']:g} better={t.detail['better']:g}")
    lines.append(f"Section 10(b) fabrication instances in the candidate run(s): {len(instances)}")
    for f in instances:
        lines.append(f"  run {f.run}  {f.sample_id}  epoch {f.epoch}  {f.code}")
    lines.append(f"Section 10(c) regression sign-off check (ladder {ladder}): {len(report.signoff)} case(s)")
    for c in report.signoff:
        row = f"  {c.sample_id}  {c.trigger}  baseline {c.baseline_count} -> candidate {c.candidate_count}  ({c.rule})"
        if c.trigger == "FM-002":  # by direction beside the total (ratified 2026-11-04)
            row += (f"  by direction: baseline ({_fmt_directions(c.baseline_by_direction)})"
                    f" -> candidate ({_fmt_directions(c.candidate_by_direction)})")
        lines.append(row)
    if report.notes:
        lines.append("Notes")
        lines.extend(f"  {n}" for n in report.notes)
    if report.blocked:
        lines.append("RESULT: BLOCKED (automatic)")
    elif report.signoff:
        lines.append("RESULT: SIGN-OFF PENDING")
    else:
        lines.append("RESULT: PASS")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description="Section 10 CI regression gates: candidate runs against the baseline run.",
                     add_help=False)
    parser.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (no comparison ran)")
    parser.add_argument("--baseline", required=True, help="baseline .eval log")
    parser.add_argument("--candidate", required=True, action="append", help="candidate .eval log; repeat to pool re-runs")
    parser.add_argument("--ladder", default="R1", choices=LADDERS, help="review-volume ladder, section 10(c)")
    parser.add_argument("--json", dest="json_out", help="write the report as JSON to this path")
    parser.add_argument("--max-tokens-pin", type=int, default=None,
                        help="the owner's max_tokens pin, when one is set (default: max_tokens must be unset)")
    args = parser.parse_args(argv)

    # Everything runs inside the try, the --json write included, and the report
    # prints only after it: any failure exits 3, never 1 (a block) after a PASS line.
    try:
        base_log = load_log(args.baseline)
        cand_logs = [load_log(p) for p in args.candidate]
        for path, log in ((args.baseline, base_log), *zip(args.candidate, cand_logs)):
            check_run_config(path, log, args.max_tokens_pin)
        baseline = records_from_log(base_log, run_config=False)  # checked just above
        candidates = [records_from_log(log, run_config=False) for log in cand_logs]
        check_matching_runs(base_log, cand_logs, args.candidate)
        report = compare(baseline, candidates, k=PREREGISTERED_K, ladder=args.ladder)
        instances = fabrication_instances(candidates)
        run_headers = [("baseline", args.baseline, run_header(base_log))]
        run_headers += [(f"candidate {i + 1}", p, run_header(log)) for i, (p, log) in enumerate(zip(args.candidate, cand_logs))]
        text = render(report, args.ladder, instances, [format_header(*h) for h in run_headers])
        code = EXIT_BLOCK if report.blocked else EXIT_SIGNOFF if report.signoff else EXIT_PASS
        if args.json_out:
            payload = {
                "baseline": args.baseline,
                "candidates": list(args.candidate),
                "ladder": args.ladder,
                "exit_code": code,
                "runs": [{"label": label, "path": path, **head} for label, path, head in run_headers],
                **asdict(report),
                "fabrication_instances": [asdict(f) for f in instances],
            }
            body = json.dumps(payload, indent=2, sort_keys=True)
            try:
                with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(body)
            except OSError as e:
                raise GateError(f"cannot write --json {args.json_out}: {type(e).__name__}: {e}") from e
    except GateError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as a pass or a block
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED

    _emit(text)
    return code


def _emit(text: str) -> None:
    """Print the report; a console that cannot encode a route key's token gets
    backslash escapes rather than a traceback (exit 1 would read as a block)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc))


def _run(argv: Sequence[str] | None = None) -> int:
    """main, with any uncaught exception exiting 3 (refused), never Python's 1."""
    try:
        return main(argv)
    except SystemExit:
        raise
    except Exception as e:
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(_run())
