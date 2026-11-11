"""The section 10 A/A estimate from the 8-epoch A/A log, and the ladder rung it fixes.

Usage:
  python scripts/aa_estimate.py AA.eval [--record [PATH]] [--date YYYY-MM-DD] [--max-tokens-pin N]

Pre-registration section 10(c) and (d), A/A estimate (owner decision D-E5,
2026-11-10): after the dataset build and before the first gated PR, one 8-epoch
run of the tagged baseline configuration on the CI regression split, with caching
off, is split into all 35 4-and-4 halves, the half holding epoch 1 as the
baseline. Each split is compared as a clean PR (regression.aa_estimate). The
script prints the null false-block rate of the automatic gates in (a) and (b),
the review volume per clean PR of the sign-off check in (c) under R1, R2 and R3,
counted by route key, and the rung regression.select_ladder takes: R1 when its
mean volume is at most 10, else R2 when its mean is at most 10, else R3, which
stands whatever its volume.

The log is read by regression.aa_records_from_log: the evaluation split, any
split but the CI regression split, a subset, a template baseline log, a cached
run, a partial run and a log off the section 5 run configuration (checked at 8
epochs) are refused. The script makes no model call and reads no API key.

--record writes the choice to a file meant to be committed (default
docs/evidence/regression/ladder.json under the repository root): the rung, the
volumes under R1, R2 and R3, the false-block rate, the A/A log file name and
SHA-256, its eval id, the run commit, the date and the ruling D-E5. The rung is
chosen once and never switched: an existing record with another rung, or from
another A/A log, is refused and left as it is; the same rung from the same log
leaves it unchanged. --record also refuses a log with no recorded commit or one
run from a dirty tree. scripts/compare_runs.py reads the record.

Exit codes: 0 the estimate printed (and recorded, with --record); 3 refused.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

from icfr_agent_evals.regression import (
    AA_EPOCHS,
    AA_SPLIT,
    LADDER_RECORD_RELPATH,
    aa_estimate,
    aa_records_from_log,
    format_aa_estimate,
    ladder_record,
    load_log,
    write_ladder_record,
)
from icfr_agent_evals.reliability import GateError

EXIT_OK, EXIT_REFUSED = 0, 3
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = REPO_ROOT / LADDER_RECORD_RELPATH


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aa_log_facts(path: str, log: Any, samples: int) -> dict[str, Any]:
    """What the record names about the A/A log; never a header value or a key."""
    revision = getattr(log.eval, "revision", None)
    return {
        "file": Path(path).name,
        "sha256": file_sha256(path),
        "eval_id": getattr(log.eval, "eval_id", None),
        "run_commit": getattr(revision, "commit", None),
        "run_dirty": getattr(revision, "dirty", None),
        "split": AA_SPLIT,
        "epochs": AA_EPOCHS,
        "samples": samples,
        "model": getattr(log.eval, "model", None),
    }


def _date(value: str) -> str:
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from e


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description="Section 10 A/A estimate from the 8-epoch A/A log on the CI regression split.")
    parser.add_argument("log", help="the 8-epoch A/A .eval log")
    parser.add_argument("--record", nargs="?", const=str(DEFAULT_RECORD), default=None, metavar="PATH",
                        help=f"write the rung choice to PATH (default {LADDER_RECORD_RELPATH}); "
                             "refuses to switch an existing record")
    parser.add_argument("--date", type=_date, default=None,
                        help="the date written into the record (default: today)")
    parser.add_argument("--max-tokens-pin", type=int, default=None,
                        help="the owner's max_tokens pin, when one is set")
    args = parser.parse_args(argv)

    try:
        log = load_log(args.log)
        records = aa_records_from_log(log, max_tokens_pin=args.max_tokens_pin)
        est = aa_estimate(records)
        facts = aa_log_facts(args.log, log, len({r.sample_id for r in records}))
        lines = [
            f"A/A log: {args.log} | file {facts['file']} | sha256 {facts['sha256']} | eval id {facts['eval_id']}"
            f" | run commit {facts['run_commit']} dirty {facts['run_dirty']} | model {facts['model']}"
            f" | split {facts['split']} | epochs {facts['epochs']} | samples {facts['samples']}",
            format_aa_estimate(est),
            f"chosen rung: {est.ladder} (section 10(c), D-E5)",
        ]
        if args.record is None:
            lines.append("not recorded: pass --record to write the choice for scripts/compare_runs.py")
        else:
            if not facts["run_commit"]:
                raise GateError("the A/A log records no git commit; --record needs the run commit")
            if facts["run_dirty"]:
                raise GateError("the A/A log was run from a dirty tree; --record needs a run of a committed configuration")
            recorded = args.date or datetime.date.today().isoformat()
            record = ladder_record(est, facts, recorded)
            status = write_ladder_record(args.record, record)
            if status == "written":
                lines.append(f"recorded: rung {est.ladder} written to {args.record}")
            else:
                lines.append(f"recorded: {args.record} already holds rung {est.ladder} from this log; left unchanged")
    except GateError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as an estimate
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED

    print("\n".join(lines))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
