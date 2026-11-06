"""Pre-flight for a keyed pilot run: every check the setup solver cannot see, run
before the owner launches.

Usage (the owner runs it from the repository root in the keyed PowerShell tab, right
before the pilot command):
  python scripts/preflight.py --model anthropic/claude-sonnet-5-5 --pilot-commit <sha>
                              [--log-dir <path>]

It never calls generate(): models are only constructed, so their resolved client
endpoints can be read, and nothing is sent to any provider. Key and endpoint
variable values are never printed; key presence prints yes or no.

Checks, one line each, PASS or FAIL:
  repo        the working folder is the top level of this repository
  head        HEAD equals --pilot-commit
  clean       git status --porcelain is empty
  python      platform.python_version() is REQUIRED_PYTHON
  env         none of MUST_BE_UNSET is present in the environment
  keys        ANTHROPIC_API_KEY and OPENAI_API_KEY are set (yes or no only)
  dotenv      the .env file Inspect would load (if any) sets none of MUST_BE_UNSET
  construct   --model and both required roles (role_pins at grader step 1) construct
  roster      roster.validate_roster passes
  serving     roster.serving_endpoints passes (validate_serving on each model,
              including request headers in force); the endpoints print
  log-dir     --log-dir (when given) exists and lies outside the repository

Exit codes: 0 when every check passes, 1 otherwise, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from icfr_agent_evals import roster

REQUIRED_PYTHON = "3.14.10"
# Each one reroutes requests or swaps the request credential (section 5 serving guard).
MUST_BE_UNSET: tuple[str, ...] = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "INSPECT_EVAL_MODEL_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "OPENAI_CUSTOM_HEADERS",
    roster.EXTRA_HEADERS_ENV,
)
KEYS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
EXIT_PASS = 0
EXIT_FAIL = 1

_SHA = re.compile(r"[0-9a-f]{7,40}")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'} {self.name}" + (f": {self.detail}" if self.detail else "")


def _sha(value: str) -> str:
    if not _SHA.fullmatch(value):
        raise argparse.ArgumentTypeError(f"--pilot-commit must be a hex commit sha (7 to 40 characters), got {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="preflight.py", description="Call-free pre-flight for a keyed pilot run.")
    p.add_argument("--model", required=True, choices=[roster.WORKER, *roster.BASELINES],
                   help="the evaluated model of the run")
    p.add_argument("--pilot-commit", required=True, type=_sha, help="the commit the pilot runs on")
    p.add_argument("--log-dir", default=None, help="the run's --log-dir; must exist outside the repository")
    return p


def _git(args: Sequence[str], cwd: Path) -> tuple[int, str]:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def check_git(pilot_commit: str, repo: Path, git: Callable[[Sequence[str], Path], tuple[int, str]] = _git) -> list[Check]:
    """repo, head and clean. git(args, cwd) -> (returncode, stdout)."""
    code, top = git(["rev-parse", "--show-toplevel"], repo)
    if code != 0 or not top:
        return [Check("repo", False, f"{repo} is not inside a git repository"),
                Check("head", False, "no repository"), Check("clean", False, "no repository")]
    checks = [Check("repo", _same_path(Path(top), repo),
                    "" if _same_path(Path(top), repo) else f"run from the repository root {top}, not {repo}")]
    code, head = git(["rev-parse", "HEAD"], repo)
    rcode, want = git(["rev-parse", "--verify", "--quiet", f"{pilot_commit}^{{commit}}"], repo)
    if code != 0 or not head:
        checks.append(Check("head", False, "git rev-parse HEAD failed"))
    elif rcode != 0 or not want:
        checks.append(Check("head", False, f"--pilot-commit {pilot_commit} is not a commit here; HEAD is {head}"))
    else:
        checks.append(Check("head", head == want, head if head == want else f"HEAD is {head}, not {want}"))
    code, status = git(["status", "--porcelain"], repo)
    if code != 0:
        checks.append(Check("clean", False, "git status failed"))
    else:
        lines = [ln for ln in status.splitlines() if ln.strip()]
        checks.append(Check("clean", not lines, "" if not lines else f"{len(lines)} uncommitted path(s)"))
    return checks


def check_python(version: str | None = None) -> Check:
    found = platform.python_version() if version is None else version
    return Check("python", found == REQUIRED_PYTHON, found if found == REQUIRED_PYTHON else f"{found}, need {REQUIRED_PYTHON}")


def check_env(environ: Mapping[str, str] | None = None) -> list[Check]:
    """env and keys. A variable present with a blank value still fails env (a blank
    base URL breaks the client); the keys need a non-blank value. Names only."""
    env = os.environ if environ is None else environ
    present = [name for name in MUST_BE_UNSET if name in env]
    keys = {name: bool(env.get(name, "").strip()) for name in KEYS}
    return [
        Check("env", not present, "none set" if not present else f"set: {', '.join(present)}; unset them"),
        Check("keys", all(keys.values()), ", ".join(f"{k} set: {'yes' if v else 'no'}" for k, v in keys.items())),
    ]


def find_dotenv_upward(start: Path) -> str:
    """The first .env file at or above start, as python-dotenv's find_dotenv(usecwd=True)
    finds it from that folder; "" when there is none."""
    for folder in (start.resolve(), *start.resolve().parents):
        if (folder / ".env").is_file():
            return str(folder / ".env")
    return ""


def check_dotenv(start: Path, find: Callable[[Path], str] | None = None,
                 values: Callable[[str], Mapping[str, Any]] | None = None) -> Check:
    """The .env file Inspect's CLI would load from the working folder up, checked for
    MUST_BE_UNSET names (values stay in memory and are never printed)."""
    if values is None:
        from dotenv import dotenv_values as values
    path = (find or find_dotenv_upward)(start)
    if not path:
        return Check("dotenv", True, "no .env file found")
    found = [name for name in MUST_BE_UNSET if name in values(path)]
    return Check("dotenv", not found, f"{path} sets none" if not found else f"{path} sets {', '.join(found)}")


def check_models(model: str, build: Callable[[str], Any] | None = None) -> list[Check]:
    """construct, roster and serving. build(name) returns an Inspect Model; the default
    is get_model(name, memoize=False). Nothing is generated."""
    if build is None:
        from inspect_ai.model import get_model

        def build(name: str) -> Any:
            return get_model(name, memoize=False)

    pins = roster.role_pins(1)
    models: dict[str, Any] = {}
    try:
        models[roster.EVALUATED_KEY] = build(model)
        for role, pin in pins.items():
            models[role] = build(pin)
    except Exception as e:  # a missing key or package fails here
        return [Check("construct", False, f"{type(e).__name__}: {' '.join(str(e).split())}"),
                Check("roster", False, "not run"), Check("serving", False, "not run")]
    checks = [Check("construct", True, ", ".join(f"{k} {v}" for k, v in models.items()))]
    try:
        roster.validate_roster(str(models[roster.EVALUATED_KEY]), {r: str(models[r]) for r in pins}, 1)
        checks.append(Check("roster", True))
    except roster.RosterError as e:
        return [*checks, Check("roster", False, str(e)), Check("serving", False, "not run")]
    try:
        endpoints = roster.serving_endpoints(models)
    except roster.RosterError as e:
        return [*checks, Check("serving", False, str(e))]
    return [*checks, Check("serving", True, "; ".join(f"{k} {v['model']} -> {v['base_url']}" for k, v in endpoints.items()))]


def check_log_dir(log_dir: str | None, repo: Path) -> Check | None:
    """None when --log-dir is not given."""
    if log_dir is None:
        return None
    path = Path(log_dir)
    if not path.is_dir():
        return Check("log-dir", False, f"{log_dir} does not exist")
    a, b = (os.path.normcase(str(x.resolve())) for x in (path, repo))
    inside = os.path.commonpath([a, b]) == b
    return Check("log-dir", not inside, "outside the repository" if not inside else f"{log_dir} lies inside the repository")


def run_checks(args: argparse.Namespace, repo: Path | None = None, *, git=_git, environ=None,
               python_version: str | None = None, build=None, find_dotenv=None, dotenv_values=None) -> list[Check]:
    repo = Path.cwd() if repo is None else repo
    checks = [*check_git(args.pilot_commit, repo, git), check_python(python_version), *check_env(environ),
              check_dotenv(repo, find_dotenv, dotenv_values), *check_models(args.model, build)]
    log_dir = check_log_dir(args.log_dir, repo)
    return checks + ([log_dir] if log_dir else [])


def main(argv: Sequence[str] | None = None, **kw: Any) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(args, **kw)
    for c in checks:
        print(c.line())
    ok = all(c.ok for c in checks)
    print(f"preflight: {'PASS' if ok else 'FAIL'} ({sum(c.ok for c in checks)}/{len(checks)} checks)")
    return EXIT_PASS if ok else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
