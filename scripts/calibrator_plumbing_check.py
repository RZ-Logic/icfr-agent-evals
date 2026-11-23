"""A parse-only plumbing check of the calibrator path, run before any evaluation-split
calibrator call (D-R23, plumbing check).

The calibrator answers grader prompt gp-1, with its default settings, on 12
judge-development outputs (dev split only). The 12 calls go through
calibrator_call.calibrate in one Inspect eval, not batched, with the evaluated model
none/none and only the calibrator role bound. calibrator_call.check_calibrator_log then
reads the log: role, pin, config as sent, no cache, no tools, endpoint recorded and one
resolved model string. Per call the check reports only whether the reply parses,
whether it was refused, its token use, and for a reply that does not parse a fixed
failure category (FAILURE_CATEGORIES). It never prints or stores the parser's message.

It prints and stores no grade: no level, flag or reason, here or in its record. The
Inspect log goes to --log-dir, outside the repository, in a folder of its own: the
grading job refuses a log folder shared with this check. A calibration record or case
is refused, as in look_plumbing_check: the calibration subset is scored only at an
acceptance look (pre-registration section 7).

Each live attempt writes one record, calibrator-plumbing-N.json (N the 1-based attempt
number), with exclusive create into the grading ledger folder under the git common
directory. The grading job starts its calibrator run only when the latest record passed,
its log still checks, its log is this check's (task name, dev split), and the calibrator
path (PLUMBING_GUARDED_PATHS) is unchanged between the record's commit and HEAD
(check_plumbing_passed).

Fail branch, fixed before the check runs: if any reply does not parse, the
evaluation-split calibrator run does not start. The author picks a scripts-only remedy
as a logged deviation, and the check is rerun on the same outputs. A check that passed
is not run again at the same code. When the calibrator path changed since the latest
passed record's commit, the check runs again as the next numbered attempt on the same
outputs as attempt 1.

Run:
  python scripts/calibrator_plumbing_check.py                        (plan only, no call)
  python scripts/calibrator_plumbing_check.py --live --log-dir GRADING_ROOT/plumbing-logs
                                                                     (keyed: 12 calibrator calls)

Exit codes: 0 plan printed or every reply parses, 1 the log check failed or a reply does
not parse, 2 bad arguments, 3 refused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
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
gc = _load("grading_common")
cc = _load("calibrator_call")
jr = al.jr

OUTPUTS = 12
SPLIT = "dev"
RECORD_NAME = "calibrator-plumbing-{}.json"
EXIT_OK, EXIT_CHECK, EXIT_ARGS, EXIT_REFUSED = 0, 1, 2, 3

PLUMBING_GUARDED_PATHS = ("scripts/calibrator_call.py", "scripts/grading_common.py",
                          "scripts/calibrator_plumbing_check.py", "icfr_agent_evals")
PLUMBING_TASK = "calibrator_plumbing_check"     # the task name and dataset name of the check's eval
FAILURE_CATEGORIES = ("not one JSON object", "wrong keys", "label not offered", "flag not boolean",
                      "refused", "no call", "does not parse")

_RECORD_RE = re.compile(r"^calibrator-plumbing-([1-9][0-9]*)\.json$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FAIL_BRANCH = ("the evaluation-split calibrator run does not start; the author picks a scripts-only remedy as a "
                "logged deviation and the check is rerun on the same outputs")


# --- Inputs ----------------------------------------------------------------------------

def pick_records(records_dir: pathlib.Path, dev_file: pathlib.Path, names: Sequence[str] | None = None, *,
                 cases_file: pathlib.Path = al.CASES_FILE) -> tuple[list[dict], dict[str, dict]]:
    """Twelve judge-development records (the first twelve by file name, or the ones named)
    and the dev cases.

    Refused (grading_common.Refused): a folder inside data/calibration, a record whose
    case is a calibration case (ids only are read from the calibration case file), a case
    that is not in the dev split, a record whose messages cannot be built, and names that
    are not twelve distinct records."""
    if al._inside(records_dir, al.CALIBRATION_DIR) or al._inside(dev_file, al.CALIBRATION_DIR):
        raise gc.Refused("calibration inputs are refused; the check runs on judge-development outputs only")
    try:
        records = jr.load_records(records_dir)
        dev = jr.load_dev(dev_file)
    except jr.Refused as e:
        raise gc.Refused(str(e)) from e
    except (OSError, ValueError) as e:
        raise gc.Refused(f"the judge-development inputs cannot be read ({type(e).__name__})") from e
    if names:
        by = {r["_file"]: r for r in records}
        if len(names) != OUTPUTS or len(set(names)) != OUTPUTS or any(n not in by for n in names):
            raise gc.Refused(f"--record names {OUTPUTS} distinct judge-development records")
        chosen = [by[n] for n in names]
    else:
        chosen = records[:OUTPUTS]
    if len(chosen) != OUTPUTS:
        raise gc.Refused(f"{len(chosen)} judge-development records, not {OUTPUTS}")
    calibration = set()
    if cases_file.is_file():
        calibration = {json.loads(line)["id"] for line in cases_file.read_text(encoding="utf-8").splitlines()
                       if line.strip()}
    for r in chosen:
        if r.get("sample_id") in calibration:
            raise gc.Refused(f"{r['_file']}: case {r.get('sample_id')} is a calibration case; "
                             f"calibration inputs are refused")
        case = dev.get(r.get("sample_id"))
        if case is None or (case.get("metadata") or {}).get("split") != SPLIT:
            raise gc.Refused(f"{r['_file']}: not a dev-split case")
        try:
            jr.record_messages(r, dev, gp.PromptSettings())
        except jr.Refused as e:
            raise gc.Refused(str(e)) from e
    return chosen, dev


def sample_id(record: Mapping[str, Any]) -> str:
    return record["_file"]


# --- The eval --------------------------------------------------------------------------

def check_task(records: Sequence[Mapping[str, Any]], dev: Mapping[str, dict]):
    """One Inspect task: per record, one sample and one calibrator call through
    calibrator_call.calibrate, with grader prompt gp-1 at its default settings."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import Generate, TaskState, solver

    @solver(name="calibrator_plumbing_calibrate")
    def calibrator_plumbing_calibrate():
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            gc.check_eval_roles(cc.CALIBRATOR_ROLE)
            state.output = await cc.calibrate(state.messages)
            return state

        return solve

    samples = [Sample(id=sample_id(r), input=gp.to_chat_messages(jr.record_messages(r, dev, gp.PromptSettings())),
                      metadata={"split": SPLIT, "record": r["_file"]})
               for r in records]
    metadata = {"split": SPLIT, "calibrator_plumbing_check": True, "parse_only": True,
                "calibrator_model": cc.calibrator_pin(),
                "calibrator_generate_config": roster.calibrator_generate_config_args(),
                "grader_prompt_version": gp.PROMPT_VERSION, "grader_prompt_sha256": gp.prompt_sha256(),
                "batched": False}
    return Task(dataset=MemoryDataset(samples, name=PLUMBING_TASK),
                solver=[calibrator_plumbing_calibrate()], epochs=1, config=GenerateConfig(), metadata=metadata,
                name=PLUMBING_TASK)


def run_check(records, dev, log_dir: str, model_roles: Mapping[str, Any] | None = None) -> Any:
    from inspect_ai import eval as inspect_eval

    roles = dict(model_roles) if model_roles is not None else {cc.CALIBRATOR_ROLE: cc.calibrator_pin()}
    return inspect_eval(check_task(records, dev), model=grader.NO_MODEL, model_roles=roles, epochs=1,
                        log_dir=log_dir, display="none")[0]


def failure_category(reply: Mapping[str, Any] | None) -> str:
    """The fixed failure category (one of FAILURE_CATEGORIES) of a reply that did not
    parse (D-R23, plumbing check). None (no completed call) and a failed judgment give
    "no call"; a refusal gives "refused". For an unparsed reply the parser's message is
    read here to pick the category and is never returned, so no key, label or other
    text of the reply reaches the output or the record."""
    if reply is None:
        return "no call"
    status = reply.get("status")
    if status == gc.REPLY_REFUSED:
        return "refused"
    if status == gc.REPLY_FAILED:
        return "no call"
    error = str(reply.get("error") or "")
    if error.startswith(("the reply is not one JSON object", "the reply is not a JSON object")):
        return "not one JSON object"
    if error.startswith("the reply's keys ") and " are not the sub-dimensions " in error:
        return "wrong keys"
    if error.endswith(" is not one of the offered labels"):
        return "label not offered"
    if error.endswith("low_confidence is not true or false"):
        return "flag not boolean"
    if error.endswith("not an object with reason, level and low_confidence"):
        return "wrong keys"
    return "does not parse"


def parse_report(calls: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per record: whether the reply parses, whether it was refused, its token use, and if
    it does not parse, its fixed failure category (failure_category), never the parser's
    message. The reply is read with grading_common.reply_record; its levels, flags,
    reasons and text are dropped here and never returned."""
    completed = calls.get("completed") or {}
    out = []
    for r in records:
        name = sample_id(r)
        event = completed.get(name)
        if event is None:
            out.append({"record": name, "parses": False, "refused": False, "input_tokens": None,
                        "output_tokens": None, "why": failure_category(None)})
            continue
        reply = gc.reply_record(event, attempt=0, log_file="")
        usage = reply.get("usage")
        row: dict[str, Any] = {"record": name, "parses": reply["status"] == gc.REPLY_PARSED,
                               "refused": reply["status"] == gc.REPLY_REFUSED,
                               "input_tokens": usage[0] if usage else None,
                               "output_tokens": usage[1] if usage else None}
        if not row["parses"]:
            row["why"] = failure_category(reply)
        out.append(row)
    return out


# --- Records ---------------------------------------------------------------------------

def record_dir(repo: pathlib.Path, ledger_dir: pathlib.Path | None = None) -> pathlib.Path:
    """The folder of the check's records: the grading ledger folder under the git common
    directory, or ledger_dir when given (tests)."""
    return pathlib.Path(ledger_dir) if ledger_dir is not None else al.git_common_dir(repo) / gc.LEDGER_FOLDER


def previous_records(folder: pathlib.Path) -> list[dict]:
    """Every calibrator-plumbing-N.json in the folder, sorted by N, each with its file name
    added under "_file". A file of that pattern with no valid attempt number, or whose
    attempt field differs from its number, is refused."""
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return []
    found = []
    for path in folder.glob(gc.PLUMBING_RECORD_GLOB):
        m = _RECORD_RE.match(path.name)
        if not m:
            raise gc.Refused(f"{path.name}: not a calibrator plumbing record name")
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise gc.Refused(f"{path.name}: unreadable ({type(e).__name__})") from e
        if not isinstance(body, dict) or body.get("attempt") != int(m.group(1)):
            raise gc.Refused(f"{path.name}: its attempt field does not match its file name")
        body["_file"] = path.name
        found.append((int(m.group(1)), body))
    return [body for _, body in sorted(found, key=lambda t: t[0])]


def calibrator_path_unchanged(repo: pathlib.Path, head: Any) -> bool:
    """True when head is a 40-hex commit, an ancestor of HEAD in repo, and no file under
    PLUMBING_GUARDED_PATHS differs between head and HEAD (D-R23, plumbing check)."""
    if not isinstance(head, str) or not _COMMIT_RE.fullmatch(head):
        return False
    if gc._git_code(repo, "merge-base", "--is-ancestor", head, "HEAD") != 0:
        return False
    return not gc._git_out(repo, "diff", "--name-only", head, "HEAD", "--", *PLUMBING_GUARDED_PATHS).strip()


def _latest_passed(previous: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passed = [p for p in previous if p.get("passed") is True]
    return passed[-1] if passed else None


def _usage_totals(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    return (sum(int(r.get("input_tokens") or 0) for r in rows), sum(int(r.get("output_tokens") or 0) for r in rows))


def _row_line(row: Mapping[str, Any]) -> str:
    if row["refused"]:
        status = "refused"
    elif row["parses"]:
        status = "parses"
    else:
        status = f"does not parse ({row['why']})"
    if row["input_tokens"] is None and row["output_tokens"] is None:
        tokens = "token use not recorded"
    else:
        tokens = f"{row['input_tokens']} input and {row['output_tokens']} output tokens"
    return f"  {row['record']}: {status}; {tokens}"


# --- main ------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, model_roles: Mapping[str, Any] | None = None,
         records_dir: pathlib.Path = jr.JUDGE_DEV_DIR, dev_file: pathlib.Path = jr.DEV_FILE,
         cases_file: pathlib.Path = al.CASES_FILE, repo: pathlib.Path = REPO,
         ledger_dir: pathlib.Path | None = None, run_commit: str = gc.RUN_COMMIT,
         archive_file: pathlib.Path = gc.ARCHIVE_FILE, running_package: bool = True) -> int:
    """model_roles (tests only): offline (mockllm) role models; the log check is then the
    offline one. Every other keyword (tests only) replaces an input, the repository, the
    record folder or a start-check input."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", action="append", default=None,
                    help=f"a judge-development record file name (stem); give {OUTPUTS}, or none for the first {OUTPUTS}")
    ap.add_argument("--records-dir", default=None, help="the judge-development records folder (default data/dev/judge-dev)")
    ap.add_argument("--live", action="store_true", help=f"make the {OUTPUTS} calibrator calls (keyed)")
    ap.add_argument("--log-dir", default=None, help="folder for the Inspect log, outside the repository")
    a = ap.parse_args(argv)
    folder = pathlib.Path(a.records_dir) if a.records_dir else records_dir
    try:
        records, dev = pick_records(folder, dev_file, a.record, cases_file=cases_file)
    except (gc.Refused, al.Refused, jr.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    names = [sample_id(r) for r in records]
    print(f"calibrator plumbing check: {len(records)} judge-development outputs, default prompt "
          f"{gp.PROMPT_VERSION} ({gp.prompt_sha256()[:12]}), calibrator role, not batched, parse only")
    print(f"  outputs: {', '.join(names)}")
    if not a.live:
        print("dry run: no call made (pass --live with --log-dir to make the calls)")
        return EXIT_OK
    if not a.log_dir or al._inside(a.log_dir, repo) or al._inside(a.log_dir, REPO):
        print("refused: --live needs --log-dir outside the repository", file=sys.stderr)
        return EXIT_ARGS
    log_dir = pathlib.Path(a.log_dir).resolve()
    try:
        state = gc.start_checks(repo, run_commit=run_commit, look_file=None, archive_file=archive_file,
                                folders={"--log-dir": log_dir}, running_package=running_package)
        out_folder = record_dir(repo, ledger_dir)
        previous = previous_records(out_folder)
        last_pass = _latest_passed(previous)
        if last_pass is not None and calibrator_path_unchanged(repo, last_pass.get("head")):
            raise gc.Refused("the check passed at this code; it is not run again")
        if previous and sorted(previous[0].get("records") or []) != sorted(names):
            raise gc.Refused("a rerun uses the same outputs as the first attempt; pass the same --record names")
    except (gc.Refused, al.Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    attempt = int(previous[-1]["attempt"]) + 1 if previous else 1
    try:
        log = run_check(records, dev, str(log_dir), model_roles)
        calls, summaries = cc.check_calibrator_log(log, asked=names, offline=model_roles is not None)
        log_file, log_sha = al.log_file_of(log)
    except Exception as e:  # noqa: BLE001 - any failure of the run or its log check
        print(f"FAIL: the log check: {e}", file=sys.stderr)
        return EXIT_CHECK
    report = parse_report(calls, records)
    for row in report:
        print(_row_line(row))
    if calls.get("skipped"):
        print(f"{calls['skipped']} errored calibrator calls skipped (each followed by a completed call)")
    parsed = sum(1 for row in report if row["parses"])
    refused = sum(1 for row in report if row["refused"])
    tokens_in, tokens_out = _usage_totals(report)
    print(f"{parsed} of {OUTPUTS} replies parse; {refused} refused; {tokens_in} input and {tokens_out} output tokens")
    passed = parsed == OUTPUTS
    body = {"check": "calibrator plumbing check", "attempt": attempt, "time": gc._now(), "head": state["head"],
            "archive_commit": state["archive"]["commit"], "log_dir": str(log_dir), "log_file": log_file,
            "log_sha256": log_sha, "records": names, "calls": OUTPUTS, "parsed": parsed, "refused": refused,
            "tokens": {"input": tokens_in, "output": tokens_out}, "per_call": report,
            "reasoning_modes_sent": sorted({s["request_mode"] for s in summaries if s.get("request_mode") is not None}),
            "passed": passed}
    path = out_folder / RECORD_NAME.format(attempt)
    try:
        out_folder.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as fh:
            fh.write(gc.canonical_bytes(body))
    except FileExistsError:
        print(f"refused: {path.name} exists; a record is never overwritten", file=sys.stderr)
        return EXIT_REFUSED
    print(f"record written: {path.name} in the grading ledger folder")
    if passed:
        print("the calibrator plumbing check passed; the evaluation-split calibrator run may start")
        return EXIT_OK
    print(_FAIL_BRANCH)
    return EXIT_CHECK


# --- Used by the grading job -----------------------------------------------------------

def check_plumbing_passed(folder: pathlib.Path, *, repo: pathlib.Path, offline: bool,
                          allowed_resolved: Sequence[str] | None = None, read_log=None) -> dict[str, Any]:
    """Raise grading_common.Refused unless the latest plumbing record passed, its log
    still checks, and it was made at the calibrator-path code that runs now (D-R23,
    plumbing check).

    The latest record (highest attempt number) must hold passed True with every one of
    its calls parsed; its log file, under the record's log_dir, must hash to the recorded
    SHA-256. The record's head must be a 40-hex commit, an ancestor of HEAD in repo, and
    no file under PLUMBING_GUARDED_PATHS may differ between it and HEAD. Read again
    (read_log, default inspect_ai.log.read_eval_log), the log must be this check's
    (task PLUMBING_TASK, dev split, the plumbing check flag in its metadata), pass
    calibrator_call.check_calibrator_log on the recorded outputs (live: against
    allowed_resolved, the resolved strings of the calibrator's own worker log), and
    every reply must parse.

    Returns {"file", "sha256", "attempt", "calls", "parsed", "refused", "passed", "head",
    "log_dir", "log_file"}: head is the record's commit, log_dir the record's resolved log
    folder (a local path, for the job's folder check only) and log_file its log file
    name."""
    lead = "the calibrator plumbing check has not passed"
    try:
        previous = previous_records(folder)
    except gc.Refused as e:
        raise gc.Refused(f"{lead} ({e})") from e
    if not previous:
        raise gc.Refused(f"{lead} (no record)")
    rec = previous[-1]
    file = rec["_file"]
    if rec.get("passed") is not True:
        raise gc.Refused(f"{lead} (the latest record, {file}, did not pass)")
    names = list(rec.get("records") or [])
    if len(names) != OUTPUTS or rec.get("calls") != OUTPUTS or rec.get("parsed") != OUTPUTS:
        raise gc.Refused(f"{lead} ({file} does not record {OUTPUTS} parsed calls)")
    log_path = pathlib.Path(str(rec.get("log_dir") or "")) / str(rec.get("log_file") or "")
    if not rec.get("log_dir") or not rec.get("log_file") or not log_path.is_file():
        raise gc.Refused(f"{lead} (the log of {file} is not on disk)")
    if gc.sha256_file(log_path) != rec.get("log_sha256"):
        raise gc.Refused(f"{lead} (the log of {file} changed since the record was written)")
    try:
        unchanged = calibrator_path_unchanged(pathlib.Path(repo), rec.get("head"))
    except gc.Refused as e:
        raise gc.Refused(f"{lead} ({e})") from e
    if not unchanged:
        raise gc.Refused(f"{lead} (the calibrator path changed since the record's commit; run the check again)")
    if read_log is None:
        from inspect_ai.log import read_eval_log

        def read_log(p):
            return read_eval_log(str(p))
    try:
        log = read_log(log_path)
    except Exception as e:  # noqa: BLE001 - an unreadable log never counts as passed
        raise gc.Refused(f"{lead} (the log of {file} fails its check: {e})") from e
    header = getattr(log, "eval", None)
    metadata = getattr(header, "metadata", None) or {}
    if (getattr(header, "task", None) != PLUMBING_TASK or metadata.get("split") != SPLIT
            or metadata.get("calibrator_plumbing_check") is not True):
        raise gc.Refused(f"{lead} (the log of {file} is not a dev-split plumbing check)")
    try:
        calls, _ = cc.check_calibrator_log(log, asked=names, offline=offline, allowed_resolved=allowed_resolved)
    except Exception as e:  # noqa: BLE001 - an off-pin log never counts as passed
        raise gc.Refused(f"{lead} (the log of {file} fails its check: {e})") from e
    rows = parse_report(calls, [{"_file": n} for n in names])
    if not all(row["parses"] for row in rows):
        raise gc.Refused(f"{lead} (a reply in the log of {file} does not parse)")
    return {"file": file, "sha256": gc.sha256_file(pathlib.Path(folder) / file),
            "attempt": rec["attempt"], "calls": rec["calls"], "parsed": rec["parsed"], "refused": rec.get("refused"),
            "passed": True, "head": rec["head"], "log_dir": str(pathlib.Path(str(rec["log_dir"])).resolve()),
            "log_file": str(rec["log_file"])}


if __name__ == "__main__":
    sys.exit(main())
