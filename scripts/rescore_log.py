"""Re-score a saved pilot log keyless on the checked-out scoring code.

`inspect score` constructs the log's model and needs its key, and `--model mockllm/model`
would rewrite the header's model. This script keeps the header and every sample's
transcript, re-runs the six deterministic scorers of scorers.py on each sample's
completion and metadata (the only fields they read), replaces the sample scores and
writes a new .eval file. The source log is never modified.

Run: python scripts/rescore_log.py <log.eval> --output <rescored.eval> [--check]

--check re-scores and compares with the scores already in the log (value, failure_modes,
review_reasons per sample and scorer) and exits 0 only when every one matches: the
self-test that the path reproduces an Inspect run on the same code.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from types import SimpleNamespace

from inspect_ai.log import read_eval_log, resolve_sample_attachments, write_eval_log

from icfr_agent_evals import scorers as scorer_module

SCORERS = (
    "correctness_anchor",
    "fabrication_anchor",
    "reconciling_items_anchor",
    "faithfulness_anchor",
    "completeness_anchor",
    "injection_compliance",
)


def rescore(log):
    fns = {name: getattr(scorer_module, name)() for name in SCORERS}
    changed = 0
    for sample in log.samples:
        # resolve_sample_attachments returns a copy: score on the copy, write to the log's sample
        resolved = resolve_sample_attachments(sample)
        state = SimpleNamespace(metadata=resolved.metadata, output=resolved.output)
        new = {}
        for name, fn in fns.items():
            new[name] = asyncio.run(fn(state, None))
        sample.scores = new
        changed += 1
    return changed


def compare(before: dict, after: dict) -> list[str]:
    diffs = []
    for name in SCORERS:
        a, b = before.get(name), after.get(name)
        if a is None or b is None:
            diffs.append(f"{name}: missing ({'before' if a is None else 'after'})")
            continue
        for field in ("value",):
            if getattr(a, field) != getattr(b, field):
                diffs.append(f"{name}.{field}: {getattr(a, field)!r} -> {getattr(b, field)!r}")
        ma, mb = a.metadata or {}, b.metadata or {}
        for key in ("failure_modes", "review_reasons", "requires_human_review"):
            if ma.get(key) != mb.get(key):
                diffs.append(f"{name}.{key}: {ma.get(key)!r} -> {mb.get(key)!r}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--output", help="path of the re-scored .eval (required unless --check)")
    ap.add_argument("--check", action="store_true", help="compare with the scores in the log instead of writing")
    args = ap.parse_args(argv)
    if not args.check and not args.output:
        ap.error("--output is required unless --check")
    log = read_eval_log(args.log)
    if log.samples is None:
        print("no samples in the log", file=sys.stderr)
        return 2
    before = {(s.id, s.epoch): dict(s.scores or {}) for s in log.samples}
    n = rescore(log)
    if args.check:
        total = 0
        for s in log.samples:
            diffs = compare(before[(s.id, s.epoch)], s.scores)
            for d in diffs:
                print(f"{s.id} epoch {s.epoch}: {d}")
            total += len(diffs)
        print(f"re-scored {n} sample-epochs; differences from the log's scores: {total}")
        return 0 if total == 0 else 1
    write_eval_log(log, args.output)
    print(f"re-scored {n} sample-epochs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
