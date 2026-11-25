"""The unseal-and-report step of the grading job (decision D-R23, unseal step). NO API key, no model call.

Usage:
  python scripts/unseal_report.py --sealed-dir GRADING_ROOT/_sealed [--grading-evidence PATH]
         --worker LABEL=PATH [--worker LABEL=PATH ...] --template PATH
         --packet-dir D --second-look-dir D2 [--blind-check-dir D3] [every other scripts/run_report.py option]

The sealed results of the grading job are opened here and nowhere else. The step refuses
unless, in this order:
  1. both run-report freeze checks pass (scripts/run_report.py, the first record and the
     second-look freeze), run through run_report.build_report with every argument it takes;
     until then the sealed folder is not touched;
  2. the grading evidence file is committed: tracked at HEAD in --repo with no change
     against HEAD (default docs/evidence/grading/grading-2026-11-21.json in --repo);
  3. the evidence names exactly the frozen manifest's worker logs, each by file name,
     SHA-256 and eval id (the template baseline run is left out, as the grading job graded
     worker runs only), and its worker run commit is the manifest's run commit;
  4. the sealed folder is named _sealed and sits outside the repository, and every sealed
     file the evidence lists (the three plan files and the four result files, no more and
     no fewer) matches its recorded SHA-256;
  5. the unit map names the same worker logs as the evidence, the draw pool holds the
     evidence's draw seed, every unit has a grader reply of record and every drawn row a
     reply of record from both judges;
  6. the four result files equal, byte for byte, what the committed grading job code
     (scripts/grading_job.py result_bodies) recomputes from the sealed plan files and the
     sealed replies with the evidence's bootstrap seed: the gate, the bootstrap, the
     breaker, the claims, the means, the paired differences and the queue.
Then it prints the run report with the judge block in place of the not-available entries
it fills: the reliability gate per sub-dimension (ICC(A,1) on complete pairs with its lower
limit, kept or excluded), the worst-case figure and break-even counts, the bootstrap limit,
the circuit breaker, the claims, the disagreement rate (the breaker's combined disagreement
and low-confidence rates, pre-registration section 7), the queue volume by tag, and the LLM
means and paired differences per worker. The resolution rate stays not available until
queue dispositions are recorded; no disposition changes any figure here.

Exit codes: 0 report printed, 3 refused, bad arguments included (nothing from the sealed folder
is printed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from icfr_agent_evals import grader_prompt as gp  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gj = _load("grading_job")
rr = gj.rr
gc = gj.gc
gs = gj.gs

EXIT_OK, EXIT_REFUSED = 0, 3
EVIDENCE_DEFAULT = pathlib.Path("docs") / "evidence" / "grading" / gc.EVIDENCE_FILE
SEALED_NAMES: tuple[str, ...] = (*gc.PLAN_FILES, *gc.RESULT_FILES)
RESOLUTION_NOT_AVAILABLE = {
    "resolution rate": ("no queue disposition is recorded yet; dispositions go in runs/reviews/<run-id>-judge-queue/, "
                        "keyed by unit id, and are reported as counts per value and as the resolution rate "
                        "(decision D-R23, queue)"),
    "acceptance-look judge statistics (test-retest alpha, bias audit, intra-rater ICC, calibration-subset "
    "distribution)": ("reported from the acceptance-look and intra-rater evidence under docs/evidence/calibration/, "
                      "not from the sealed full-run grades"),
}


class Refused(rr.Refused):
    """The unseal step refuses; nothing from the sealed folder is printed."""


# --- Checks -----------------------------------------------------------------------------------

def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def committed_evidence(repo: pathlib.Path, evidence: pathlib.Path) -> tuple[dict, str]:
    """The grading evidence body and its SHA-256, refused unless the file sits in the
    repository, is tracked at HEAD and has no change against HEAD."""
    repo, evidence = repo.resolve(), evidence.resolve()
    try:
        rel = evidence.relative_to(repo).as_posix()
    except ValueError:
        raise Refused("the grading evidence file is not inside the repository") from None
    if not evidence.is_file():
        raise Refused(f"the grading evidence file {rel} is missing")
    if _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        raise Refused(f"the grading evidence file {rel} is not committed")
    if _git(repo, "cat-file", "-e", f"HEAD:{rel}").returncode != 0:
        raise Refused(f"the grading evidence file {rel} is not in the HEAD commit")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise Refused(f"the grading evidence file {rel} differs from the HEAD commit")
    data = evidence.read_bytes()
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refused(f"the grading evidence file {rel} cannot be read: {type(e).__name__}") from None
    if not isinstance(body, dict) or body.get("evidence") != "full-run grading":
        raise Refused(f"{rel} is not a full-run grading evidence file")
    return body, gc.sha256_bytes(data)


def _log_set(entries: Sequence[Mapping[str, Any]], file_key: str) -> set[tuple[str, str, str]]:
    return {(str(e.get(file_key, "")), str(e.get("sha256", "")), str(e.get("eval_id", ""))) for e in entries}


def manifest_worker_logs(checks: Any) -> set[tuple[str, str, str]]:
    """The frozen manifest's worker logs, (file name, SHA-256, eval id): the record's runs
    that are not a template baseline, each also listed in the committed manifest."""
    rf = rr.load_review_freeze()
    record, manifest = checks.record, checks.first_manifest or {}
    workers = [record.runs[rid] for rid in rf.worker_run_ids(record)]
    found = _log_set(workers, "file_name")
    listed = _log_set(manifest.get("logs") or [], "file_name")
    if not found:
        raise Refused("the frozen record names no worker run")
    if not found <= listed:
        raise Refused("the frozen manifest's log list does not hold every worker run of the record")
    return found


def check_evidence_logs(evidence: Mapping[str, Any], checks: Any) -> None:
    want = manifest_worker_logs(checks)
    have = _log_set(evidence.get("worker_logs") or [], "file")
    if have != want:
        raise Refused(f"the grading evidence names {len(have - want)} worker log(s) the frozen manifest does not and "
                      f"leaves out {len(want - have)} it does; each is matched by file name, SHA-256 and eval id")
    run_commit = str((checks.first_manifest or {}).get("run_commit") or "")
    if not rr._same_commit(str(evidence.get("worker_run_commit") or ""), run_commit):
        raise Refused("the grading evidence's worker run commit is not the frozen manifest's run commit")


def check_sealed_files(sealed_dir: Any, evidence: Mapping[str, Any], repo: pathlib.Path) -> pathlib.Path:
    """The sealed folder, resolved, after every sealed file is checked against the SHA-256
    the evidence records for it."""
    try:
        folder = gc.check_sealed_dir(sealed_dir, repo)
    except gc.Refused as e:
        raise Refused(str(e)) from None
    recorded = dict(((evidence.get("sealed") or {}).get("files")) or {})
    if set(recorded) != set(SEALED_NAMES):
        raise Refused(f"the grading evidence lists sealed files {sorted(recorded)}, not {sorted(SEALED_NAMES)}")
    for name in SEALED_NAMES:
        path = folder / name
        if not path.is_file():
            raise Refused(f"the sealed file {name} is missing")
        if gc.sha256_file(path) != recorded[name]:
            raise Refused(f"the sealed file {name} does not match the SHA-256 the grading evidence records")
    return folder


def _read_sealed(folder: pathlib.Path, name: str) -> dict:
    body = json.loads((folder / name).read_text(encoding="utf-8"))
    if not isinstance(body, dict) or body.get("run") != evidence_run():
        raise Refused(f"the sealed file {name} is not a file of run {evidence_run()}")
    return body


def evidence_run() -> str:
    return gc.RUN_LABEL


def recompute(folder: pathlib.Path, evidence: Mapping[str, Any], *,
              resamples: int = gs.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Rebuild the plan and the replies from the sealed files, recompute the four result
    files with the committed job code and refuse unless each equals its sealed bytes."""
    unit_map = _read_sealed(folder, gc.SEALED_UNIT_MAP)
    pool = _read_sealed(folder, gc.SEALED_DRAW_POOL)
    drawn = _read_sealed(folder, gc.SEALED_DRAWN)
    grader_replies = _read_sealed(folder, gc.SEALED_GRADER_REPLIES)["replies"]
    calibrator_replies = _read_sealed(folder, gc.SEALED_CALIBRATOR_REPLIES)["replies"]
    if _log_set(unit_map.get("worker_logs") or [], "file") != _log_set(evidence.get("worker_logs") or [], "file"):
        raise Refused("the sealed unit map names other worker logs than the grading evidence")
    seeds = dict(evidence.get("seeds") or {})
    if pool.get("seed") != seeds.get("calibrator_draw"):
        raise Refused("the sealed draw pool's seed is not the evidence's calibrator draw seed")
    bootstrap_seed = seeds.get("bootstrap")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise Refused("the grading evidence holds no bootstrap seed")
    units = dict(unit_map["units"])
    rows = list(drawn["rows"])
    no_grader = [u for u in units if u not in grader_replies]
    no_row_reply = [r["uid"] for r in rows if r["uid"] not in grader_replies or r["uid"] not in calibrator_replies]
    if no_grader or no_row_reply:
        raise Refused(f"{len(no_grader)} unit(s) have no grader reply of record and {len(no_row_reply)} drawn row(s) "
                      f"lack a reply of record from a judge")
    extra = (set(grader_replies) - set(units)) | (set(calibrator_replies) - {r["uid"] for r in rows})
    if extra:
        raise Refused(f"the sealed replies hold {len(extra)} unit id(s) outside the plan")
    plan = {"units": units, "generations": unit_map["generations"], "draw": {"rows": rows}}
    replies = {"grader": grader_replies, "calibrator": calibrator_replies}
    try:
        bodies = gj.result_bodies(plan, replies, {}, bootstrap_seed=bootstrap_seed, resamples=resamples)
    except Exception as e:  # noqa: BLE001 - a recompute that fails is a refusal, never a report
        raise Refused(f"the sealed results cannot be recomputed: {type(e).__name__}: {e}") from None
    for name in gc.RESULT_FILES:
        if gc.canonical_bytes(bodies[name]) != (folder / name).read_bytes():
            raise Refused(f"the recomputed {name} differs from the sealed one")
    return {"plan": plan, "aggregates": bodies[gc.SEALED_AGGREGATES], "queue": bodies[gc.SEALED_QUEUE]["items"],
            "worker_logs": unit_map["worker_logs"]}


# --- The judge block --------------------------------------------------------------------------

def _rate_text(rate: Mapping[str, Any]) -> str:
    if rate["rate"] is None:
        return f"{rate['tagged']} of 0 (undefined)"
    return f"{rate['tagged']} of {rate['denominator']} ({rate['rate']:.4f})"


def _num(x: Any) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _names(worker_logs: Sequence[Mapping[str, Any]], workers: Sequence[Any]) -> dict[str, str]:
    """Model id to 'label (model id)', matched through the log SHA-256."""
    label_of = {w.sha256: w.label for w in workers}
    return {str(w["model"]): f"{label_of.get(w['sha256'], '?')} ({w['model']})" for w in worker_logs}


def _mean_difference(figure: Mapping[str, Any], dim: str) -> dict[str, Any]:
    """The signed grader-minus-calibrator mean of one sub-dimension on complete pairs, or {}."""
    if not figure.get("computed"):
        return {}
    per = ((figure.get("exclusion") or {}).get("gate") or {}).get("per_subdimension") or {}
    return dict((per.get(dim) or {}).get("mean_difference") or {})


def judge_lines(block: Mapping[str, Any]) -> list[str]:
    agg, names = block["aggregates"], block["names"]
    gate, br = agg["gate"], agg["breaker"]
    lines = ["== Judge results (unsealed, decision D-R23 unseal step; grader scores are of record)",
             f"  grading evidence {block['evidence_file']} sha256 {block['evidence_sha256']}; sealed files checked "
             f"{len(block['sealed_files'])}; recomputed results equal the sealed ones",
             f"  reliability gate (ICC(A,1) lower limit on complete pairs, threshold {gate['threshold']}): calibrator rows "
             f"{gate['rows']}, complete pairs {gate['pairs_complete']}, dropped {gate['pairs_dropped']}, neither "
             f"parsed {gate['neither_parsed']}"]
    for d in gp.DIMENSIONS:
        kept, reason, view = gs._figure_view(gate["complete"], d)
        boot = agg["bootstrap"].get(d, {})
        even = gate["break_even"].get(d, {})
        worst = gs._figure_view(gate["worst_case"], d)[2]
        state = "kept" if kept else f"excluded ({reason})"
        diff = _mean_difference(gate["complete"], d)
        lines.append(f"    {d}: {state}; ICC(A,1) {_num((view or {}).get('estimate'))}, lower limit "
                     f"{_num((view or {}).get('ci_lower'))}; grader minus calibrator {_num(diff.get('mean'))} "
                     f"[{_num(diff.get('ci_lower'))}, {_num(diff.get('ci_upper'))}] (paired 95% t interval, "
                     f"reported, not gated); case-cluster bootstrap lower limit "
                     f"{_num(boot.get('ci_lower'))} (reported, not gated); worst-case ICC(A,1) "
                     f"{_num((worst or {}).get('estimate'))} (reported, not gated); break-even dropped pairs "
                     f"{even.get('count')} ({even.get('reason') or 'first count below the threshold'})")
    lines.append(f"  circuit breaker (full run, limit {br['limit']}): pair rate {_rate_text(br['pair_rate'])}, "
                 f"generation rate {_rate_text(br['generation_rate'])}; trip {'yes' if br['trip'] else 'no'}")
    for model, r in sorted(br["per_worker"].items()):
        lines.append(f"    {names.get(model, model)}: pair rate {_rate_text(r['pair_rate'])}, generation rate "
                     f"{_rate_text(r['generation_rate'])} (descriptive, never a trip)")
    claims = agg["claims"]
    lines.append("  claims: " + "; ".join(
        f"{d} {'kept' if claims[d]['kept'] else 'excluded (' + ', '.join(claims[d]['reasons']) + ')'}"
        for d in gp.DIMENSIONS) + f"; anchors stand alone: {'yes' if claims['anchors_stand_alone'] else 'no'}")
    dis = block["disagreement"]
    lines.append(f"  disagreement rate (combined disagreement and low-confidence rate, section 7): pairs "
                 f"{_rate_text(dis['pair_rate'])}, generations {_rate_text(dis['generation_rate'])}")
    q = agg["queue_counts"]
    lines.append(f"  disagreement queue: {q['items']} item(s) of {block['units']} graded unit(s); by tag "
                 + ", ".join(f"{t} {n}" for t, n in q["by_tag"].items()))
    lines.append("  LLM means per worker, all cases, primary reading (conditional on a parsed worker output):")
    for model, per_subset in sorted(agg["means"].items()):
        cells = []
        for d in gp.DIMENSIONS:
            m = per_subset["all"][d]["primary"]
            cells.append(f"{d} {_num(m['mean'])} [{_num(m['ci95_low'])}, {_num(m['ci95_high'])}] "
                         f"(samples {m['samples']})")
        lines.append(f"    {names.get(model, model)}: " + "; ".join(cells))
    lines.append("  paired differences, all cases, primary reading (first minus second):")
    for p in agg["paired_differences"]:
        if p["subset"] != "all" or p["reading"] != "primary":
            continue
        res = p["result"] or {}
        lines.append(f"    {names.get(p['first'], p['first'])} minus {names.get(p['second'], p['second'])}, {p['dim']}: "
                     f"{_num(res.get('mean'))} [{_num(res.get('ci_lower'))}, {_num(res.get('ci_upper'))}] "
                     f"(samples {p['samples']})")
    lines.append("  every subset, the sensitivity reading and the defect-dropped rows are in the JSON (judge.aggregates)")
    return lines


def judge_block(checks: Any, workers: Sequence[Any], *, sealed_dir: Any, evidence_path: pathlib.Path,
                repo: pathlib.Path, resamples: int = gs.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Every check of steps 2 to 6, then the block run_report prints. Called by
    run_report.build_report only when both freeze checks pass."""
    if not checks.passed:  # build_report never calls it then; a second guard all the same
        raise Refused("both run-report freeze checks must pass before the sealed folder is opened")
    evidence, evidence_sha = committed_evidence(repo, evidence_path)
    check_evidence_logs(evidence, checks)
    folder = check_sealed_files(sealed_dir, evidence, repo)
    got = recompute(folder, evidence, resamples=resamples)
    agg = got["aggregates"]
    block = {
        "evidence_file": evidence_path.name, "evidence_sha256": evidence_sha,
        "sealed_files": dict(evidence["sealed"]["files"]), "bootstrap_seed": evidence["seeds"]["bootstrap"],
        "units": len(got["plan"]["units"]), "names": _names(got["worker_logs"], workers),
        "aggregates": agg,
        "disagreement": {"pair_rate": agg["breaker"]["pair_rate"],
                         "generation_rate": agg["breaker"]["generation_rate"]},
        "queue_volume": agg["queue_counts"]["items"],
        "resolution": None,
        "still_not_available": dict(RESOLUTION_NOT_AVAILABLE),
    }
    block["text"] = judge_lines(block)
    return block


# --- CLI ----------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = rr.parser()
    p.description = "The unseal-and-report step: run_report with the sealed grading results (no model call)."
    p.add_argument("--sealed-dir", required=True, help="the grading job's sealed folder, named _sealed")
    p.add_argument("--grading-evidence", default=None,
                   help=f"the committed grading evidence file (default {EVIDENCE_DEFAULT.as_posix()} in --repo)")
    return p


def main(argv: Sequence[str] | None = None, *, resamples: int = gs.BOOTSTRAP_RESAMPLES) -> int:
    args = parser().parse_args(argv)
    repo = pathlib.Path(args.repo) if args.repo else REPO
    evidence_path = pathlib.Path(args.grading_evidence) if args.grading_evidence else repo / EVIDENCE_DEFAULT

    def judge(checks: Any, workers: Sequence[Any]) -> dict[str, Any]:
        return judge_block(checks, workers, sealed_dir=args.sealed_dir, evidence_path=evidence_path, repo=repo,
                           resamples=resamples)

    try:
        workers = [rr.LogInput(label, path, rr.read_log(path, f"worker {label}"), "worker")
                   for label, path in args.worker]
        template = rr.LogInput("template", args.template, rr.read_log(args.template, "template"), "template")
        text, payload = rr.build_report(
            workers, template, packet_dir=args.packet_dir, manifest=args.manifest, freeze_commit=args.freeze_commit,
            second_look_dir=args.second_look_dir, second_look_manifest=args.second_look_manifest,
            second_look_commit=args.second_look_commit, blind_check_dir=args.blind_check_dir, repo=args.repo,
            max_tokens_pin=args.max_tokens_pin, args=args, judge=judge)
        if payload["withheld"]:
            reasons = "; ".join(payload["withheld_reasons"])
            raise Refused(f"both run-report freeze checks must pass before the sealed folder is opened: {reasons}")
        if args.json_out:
            try:
                with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(rr._json_safe(payload))
            except OSError as e:
                raise Refused(f"cannot write {args.json_out}: {type(e).__name__}") from e
    except (rr.Refused, rr.GateError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # noqa: BLE001 - an unreadable input must never read as a report
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "backslashreplace").decode(enc))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
