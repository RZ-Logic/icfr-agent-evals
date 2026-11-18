"""Audit the dataset of a git ref in one command, without trusting the author's files.

An outsider runs this to check that the cases a ref's code builds are the cases the
pre-registration records, and that every answer key is consistent with its case and
with the scorer. It makes no model call, reads no API key and never prints case
content: only counts, hashes, case ids, check names and PASS, FAIL or SKIP.

Run:
  python scripts/audit_dataset.py                       audit the tag prereg-phase1-v1.0
  python scripts/audit_dataset.py --ref HEAD            audit the current commit
  python scripts/audit_dataset.py --logs runs/x         also check Inspect log headers
  python scripts/audit_dataset.py --splits dev ci       audit some splits only
  python scripts/audit_dataset.py --json                the report as JSON

How it works. The ref's committed tree is exported with git archive into a temporary
folder. A child Python process runs there with PYTHONPATH set to that folder, so it
imports the ref's own package and nothing from the working copy. It rebuilds each
split the ref's code builds (dev, ci, eval, and calibration when the ref has it; the
eval split takes data/eval/handwritten.jsonl only when the ref holds that file) and
runs the per-case checks below with this script's own code. The child's API keys are
removed from its environment.

Per split it reports the case count, the composition (case kinds and categories), the
SHA-256 of dataset.split_jsonl(split) and dataset_sha256 as
icfr_agent_evals.casefile.dataset_sha256 computes it. Then these checks:

  split seed        the split seed is the one section 4 fixes.
  composition       the case kinds match section 4 (dev 14/6/10, ci 146/30/24, eval
                    146 synthetic, plus 30 edge and 24 adversarial once the
                    hand-written file is in the ref; calibration 40 cases), and the
                    ci categories match section 4.
  validate_truth    casefile.validate_truth passes on every answer key, and the key's
                    case id and split match the sample.
  entity closure    generator.check_entity_closure passes on the rendered input.
  wallet labels     scoring.check_wallet_labels_in_table passes on every case.
  perfect answer    an answer built from the key (every eligible item cited once with
                    its amount, the custodian cited, a narrative with no amounts, as
                    tests/casefactory.good_output builds it) scores 3 with no FM code
                    and no review route on check_correctness, check_fabrication,
                    check_reconciling_items, check_faithfulness and
                    check_completeness, and on check_injection for every case with an
                    injection spec. An AC-1 case without an injection spec fails.
  committed file    the dev split equals data/dev/dev.jsonl byte for byte, and the
                    calibration split equals data/calibration/calibration.jsonl when
                    the ref holds it. The dataset_sha256, file_sha256, case kinds and
                    categories in each folder's COMPOSITION.json match the rebuild.
  section 13        dataset_sha256 of the dev and ci splits and of the 146 synthetic
                    eval cases equals the value section 13 records at the tag
                    (RECORDED_SECTION13 below, copied from the generator row, where
                    scripts/tag_values.py computed it the same way:
                    dataset_sha256(build_dataset(split, seed=EVAL_SEED))). The ref's
                    own section 13 text must state the same three values.
  tag commit        with the default ref, the tag points at TAG_COMMIT.

What a FAIL on "perfect answer" means. The scorer did not give full, clean credit to
an answer that copies the answer key. Either the key and the scorer disagree on that
case (a defect in one of them) or the case meets a documented limit of the narrative
extractor, such as the patterns the 17 known failing oracle tests record. A FAIL there
does not by itself say which; each case is listed by id with the anchor, its points,
its FM codes and its review reasons with entities, quoted text and numbers masked, so
the reader can look the case up. No case is hidden and no exception list excuses one:
the script still exits non-zero. On the eval split no score is read here except this
self-check of the scorer against the key.

With --logs DIR, every Inspect log under DIR (*.eval, *.json) is read header only. Its
task metadata split_dataset_sha256 must equal the rebuilt split's dataset_sha256, its
dataset_sha256 the rebuilt hash of the samples the run selected (the whole split, or
the dev pilot subset), and its split_seed the rebuilt split's seed. No sample is read.

Exit code: 0 when nothing fails, 1 when any check fails, 2 when the audit cannot run
(an unknown ref, no git, or a child process that crashes before reporting).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_REF = "prereg-phase1-v1.0"
TAG_COMMIT = "dc9c26e5315a50595de50e4de63964b9b177504a"
ALL_SPLITS = ("dev", "ci", "eval", "calibration")

# Section 13, generator row, recorded 2026-11-14 at the tag: dataset_sha256 per split.
RECORDED_SECTION13 = {
    "dev": "f56d06b58b06bbbb0902a552df1a84d5107299bc09c78c6eb514510fce0f7887",
    "ci": "e78ffb9e603ea1b3d631fae40cfcd072adb4938af74e9a7981738256f76323f6",
    "eval_synthetic": "65737b45d2a2f42c5a46933f7801edac07076cab685585a560f2247447935b2a",
}
SECTION13_RE = re.compile(
    r"dataset_sha256 dev ([0-9a-f]{64}), CI ([0-9a-f]{64}), evaluation \(146 synthetic\) ([0-9a-f]{64})")

# Section 4: split seeds and composition.
SECTION4_SEEDS = {"eval": 20260607, "dev": 21260607, "ci": 22260607, "calibration": 23260607}
SECTION4_KINDS = {
    "dev": {"adversarial": 10, "edge": 6, "standard": 14},
    "ci": {"adversarial": 24, "edge": 30, "standard": 146},
    "eval": {"standard": 146},
    "eval_handwritten": {"adversarial": 24, "edge": 30, "standard": 146},
}
SECTION4_CI_CATEGORIES = {
    "AC-1": 4, "AC-2": 4, "AC-3": 4, "AC-4": 4, "AC-5": 4, "AC-6": 4,
    "ambiguous_narrative": 10, "band_boundary": 10, "incomplete_data": 10, "standard": 146,
}
SECTION4_CALIBRATION_CASES = 40

COMMITTED_FILES = {
    "dev": ("data/dev/dev.jsonl", "data/dev/COMPOSITION.json"),
    "calibration": ("data/calibration/calibration.jsonl", "data/calibration/COMPOSITION.json"),
}
HANDWRITTEN_PATH = "data/eval/handwritten.jsonl"
PREREG_PATH = "PRE-REGISTRATION-PHASE-1.md"

ANCHORS = ("check_correctness", "check_fabrication", "check_reconciling_items",
           "check_faithfulness", "check_completeness")
CASE_CHECKS = ("validate_truth", "entity_closure", "wallet_labels", "perfect_answer")
SCORE_FULL = 3
# Provider variables dropped from the child's environment (the tests/conftest.py list).
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
    "INSPECT_EVAL_MODEL_BASE_URL", "INSPECT_EVAL_SCAN_MODEL_BASE_URL", "SCOUT_SCAN_MODEL_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS", "INSPECT_EVAL_EXTRA_HEADERS",
)


class AuditError(RuntimeError):
    """The audit cannot run (exit 2)."""


# --- Child side: per-case checks (run in the exported tree) -------------------------------

def perfect_answer(truth) -> dict:
    """The answer tests/casefactory.good_output builds from a key: every eligible item
    cited once with its amount, the custodian cited, a narrative with no amounts. Kept
    here so the audited ref cannot change it."""
    eligible = truth.eligible_items()
    return {
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction,
        "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian] + [i.reference for i in eligible],
        "reconciling_items": [
            {"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in eligible
        ],
        "narrative": (
            "On-chain reserves fell short of ledger liabilities for the period."
            if truth.direction == "shortfall"
            else "On-chain reserves exceeded ledger liabilities for the period."
        ),
    }


def _case_strings(truth, sample_id: str) -> list[str]:
    """Strings of a case that a printed reason must never show."""
    out = {sample_id}
    if truth is not None:
        out.add(str(truth.case_id))
        out.update(str(x) for x in truth.valid_entities)
        out.update(str(x) for x in truth.organizations)
        out.update(str(x) for x in truth.scope_wallets)
        out.add(str(truth.custodian))
        for item in truth.items:
            out.add(str(item.reference))
            out.add(str(item.wallet))
            out.update(str(a) for a in item.aliases)
        if truth.injection is not None:
            for value in (truth.injection.canary, truth.injection.instruction_text,
                          truth.injection.demanded_value, truth.injection.truth_value):
                if value is not None:
                    out.add(str(value))
    return sorted((s for s in out if s and len(s) >= 3), key=len, reverse=True)


_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_HAS_DIGIT = re.compile(r"\S*\d\S*")


def mask(text: str, truth=None, sample_id: str = "") -> str:
    """A reason with case entities, quoted text and anything holding a digit masked,
    cut to 160 characters."""
    text = str(text)
    for s in _case_strings(truth, sample_id):
        text = text.replace(s, "<entity>")
    text = _QUOTED.sub("<text>", text)
    text = _HAS_DIGIT.sub(lambda m: m.group(0) if m.group(0).startswith("FM-") else "<n>", text)
    text = " ".join(text.split())
    return text[:160]


def _exc(e: BaseException, truth, sid: str) -> str:
    return f"{type(e).__name__}: {mask(str(e), truth, sid)}"


def audit_cases(samples, split: str | None = None) -> dict:
    """Run the per-case checks on samples with the icfr_agent_evals package on sys.path.
    Returns {check: {"checked": n, "failed": [{"id", "reason"}]}}; no case content."""
    from icfr_agent_evals import scoring
    from icfr_agent_evals.casefile import truth_of, validate_truth
    from icfr_agent_evals.generator import check_entity_closure

    res = {name: {"checked": 0, "failed": []} for name in CASE_CHECKS}

    def fail(name, sid, reason):
        res[name]["failed"].append({"id": sid, "reason": reason})

    for s in samples:
        sid = str(s.id)
        meta = s.metadata or {}
        for name in CASE_CHECKS:
            res[name]["checked"] += 1
        try:
            truth = truth_of(meta)
        except Exception as e:  # noqa: BLE001 - a key that does not load fails every check
            for name in CASE_CHECKS:
                fail(name, sid, f"answer key does not load: {_exc(e, None, sid)}")
            continue

        try:
            validate_truth(truth)
            if str(truth.case_id) != sid:
                raise AssertionError("answer key case_id differs from the sample id")
            if split is not None and truth.split != split:
                raise AssertionError("answer key split differs from the split built")
        except Exception as e:  # noqa: BLE001
            fail("validate_truth", sid, _exc(e, truth, sid))

        try:
            check_entity_closure(str(s.input), truth)
        except Exception as e:  # noqa: BLE001
            fail("entity_closure", sid, _exc(e, truth, sid))

        try:
            scoring.check_wallet_labels_in_table([s])
        except Exception as e:  # noqa: BLE001
            fail("wallet_labels", sid, _exc(e, truth, sid))

        reasons = []
        try:
            answer = perfect_answer(truth)
            anchors = [(name, getattr(scoring, name)) for name in ANCHORS]
            if truth.injection is not None:
                anchors.append(("check_injection", scoring.check_injection))
            elif truth.category == "AC-1":
                reasons.append("AC-1 case has no injection spec")
            for name, fn in anchors:
                try:
                    r = fn(answer, truth)
                except Exception as e:  # noqa: BLE001
                    reasons.append(f"{name} raised {_exc(e, truth, sid)}")
                    continue
                if r.points != SCORE_FULL or r.failure_modes or r.review_reasons:
                    bits = [f"{name} points {r.points}"]
                    if r.failure_modes:
                        bits.append(" ".join(r.failure_modes))
                    bits += [f"review: {mask(x, truth, sid)}" for x in r.review_reasons]
                    reasons.append("; ".join(bits))
        except Exception as e:  # noqa: BLE001
            reasons.append(_exc(e, truth, sid))
        if reasons:
            fail("perfect_answer", sid, " | ".join(reasons))
    return res


def _counts(values) -> dict:
    return dict(sorted(Counter(values).items()))


def child_main(out_path: str, splits: list[str]) -> int:
    """Build each split from the package on PYTHONPATH and write a JSON report."""
    here = pathlib.Path.cwd().resolve()
    import icfr_agent_evals

    pkg = pathlib.Path(icfr_agent_evals.__file__).resolve()
    report: dict = {"package_inside_export": here in pkg.parents, "splits": {}}
    if not report["package_inside_export"]:
        pathlib.Path(out_path).write_text(json.dumps(report), encoding="utf-8")
        return 0

    from icfr_agent_evals import dataset as ds
    from icfr_agent_evals.casefile import EVAL_SEED, dataset_sha256, split_seed, truth_of

    report["eval_seed"] = EVAL_SEED
    for split in splits:
        entry: dict = {"built": False}
        report["splits"][split] = entry
        try:
            entry["split_seed"] = split_seed(split, EVAL_SEED)
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"the ref does not know this split ({type(e).__name__})"
            continue
        try:
            samples = list(ds.build_dataset(split=split, seed=EVAL_SEED))
        except NotImplementedError:
            entry["error"] = "the ref's code does not build this split"
            continue
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"build raised {type(e).__name__}"
            entry["build_failed"] = True
            continue
        entry["built"] = True
        truths = []
        for s in samples:
            try:
                truths.append(truth_of(s.metadata))
            except Exception:  # noqa: BLE001 - audit_cases reports it
                truths.append(None)
        entry["n"] = len(samples)
        entry["case_kind"] = _counts(t.case_kind if t else "unreadable" for t in truths)
        entry["category"] = _counts((t.category or "standard") if t else "unreadable" for t in truths)
        entry["dataset_sha256"] = dataset_sha256(samples)
        try:
            entry["split_jsonl_sha256"] = hashlib.sha256(ds.split_jsonl(split, EVAL_SEED)).hexdigest()
        except Exception as e:  # noqa: BLE001
            entry["split_jsonl_error"] = type(e).__name__
        if split == "eval":
            std = [s for s, t in zip(samples, truths) if t is not None and t.case_kind == "standard"]
            entry["synthetic_n"] = len(std)
            entry["synthetic_dataset_sha256"] = dataset_sha256(std)
        if split == "dev":
            try:
                pilot = ds.select_subset(samples, "dev", "pilot")
                entry["pilot_n"] = len(pilot)
                entry["pilot_dataset_sha256"] = dataset_sha256(pilot)
            except Exception as e:  # noqa: BLE001
                entry["pilot_error"] = type(e).__name__
        entry["checks"] = audit_cases(samples, split)
    pathlib.Path(out_path).write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return 0


# --- Parent side: export, run the child, compare ------------------------------------------

def _git(repo: pathlib.Path, *args: str) -> bytes:
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "-c", "core.autocrlf=false", *args],
                             capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        raise AuditError(f"git could not run: {type(e).__name__}") from e
    if out.returncode != 0:
        raise AuditError(f"git {args[0]} failed: {out.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return out.stdout


def resolve_ref(repo: pathlib.Path, ref: str) -> str:
    if ref.startswith("-"):
        raise AuditError(f"not a ref: {ref!r}")
    return _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").decode().strip()


def export_tree(repo: pathlib.Path, commit: str, dest: pathlib.Path) -> None:
    """The committed tree of commit, written under dest by git archive."""
    blob = _git(repo, "archive", "--format=tar", commit)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        tar.extractall(dest, filter="data")


def child_env(export: pathlib.Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    env["PYTHONPATH"] = str(export)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_child(export: pathlib.Path, splits: list[str], timeout: int = 3600) -> dict:
    out = export.parent / "audit-child.json"
    cmd = [sys.executable, "-P", str(pathlib.Path(__file__).resolve()), "--child", str(out), "--splits", *splits]
    try:
        proc = subprocess.run(cmd, cwd=str(export), env=child_env(export), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AuditError("the child process timed out") from e
    if proc.returncode != 0 or not out.is_file():
        last = (proc.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1]
        raise AuditError(f"the child process failed with exit code {proc.returncode} ({last.split(':')[0][:80]})")
    return json.loads(out.read_text(encoding="utf-8"))


def _check(rows: list, scope: str, name: str, status: str, detail: str = "", failed: list | None = None) -> None:
    rows.append({"scope": scope, "check": name, "status": status, "detail": detail, "failed": failed or []})


def _sha_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(child: dict, export: pathlib.Path, splits: list[str], *, ref: str, commit: str) -> list[dict]:
    """PASS, FAIL or SKIP rows from the child's report and the exported tree."""
    rows: list[dict] = []
    if ref == DEFAULT_REF:
        _check(rows, "ref", "tag commit", "PASS" if commit == TAG_COMMIT else "FAIL",
               f"{ref} is {commit}" + ("" if commit == TAG_COMMIT else f", expected {TAG_COMMIT}"))
    if not child.get("package_inside_export"):
        _check(rows, "ref", "package source", "FAIL", "icfr_agent_evals did not load from the exported tree")
        return rows

    prereg = export / PREREG_PATH
    m = SECTION13_RE.search(prereg.read_text(encoding="utf-8")) if prereg.is_file() else None
    if m is None:
        _check(rows, "ref", "section 13 text", "SKIP", "the ref's pre-registration states no dataset_sha256 row")
    else:
        stated = dict(zip(("dev", "ci", "eval_synthetic"), m.groups()))
        _check(rows, "ref", "section 13 text", "PASS" if stated == RECORDED_SECTION13 else "FAIL",
               "the ref's section 13 states the recorded dataset_sha256 values"
               if stated == RECORDED_SECTION13 else "the ref's section 13 states other dataset_sha256 values")

    hw_present = (export / HANDWRITTEN_PATH).is_file()
    for split in splits:
        e = child["splits"].get(split, {})
        if not e.get("built"):
            committed = COMMITTED_FILES.get(split, (None,))[0]
            if e.get("build_failed") or (committed and (export / committed).is_file()):
                _check(rows, split, "build", "FAIL", e.get("error", "not built"))
            else:
                _check(rows, split, "build", "SKIP", e.get("error", "not built"))
            continue

        want_seed = SECTION4_SEEDS.get(split)
        _check(rows, split, "split seed", "PASS" if e["split_seed"] == want_seed else "FAIL",
               f"{e['split_seed']}" + ("" if e["split_seed"] == want_seed else f", section 4 states {want_seed}"))

        if split == "calibration":
            ok = e["n"] == SECTION4_CALIBRATION_CASES
            _check(rows, split, "composition", "PASS" if ok else "FAIL",
                   f"{e['n']} cases" + ("" if ok else f", section 4 states {SECTION4_CALIBRATION_CASES}"))
        else:
            want = SECTION4_KINDS["eval_handwritten" if split == "eval" and hw_present else split]
            ok = e["case_kind"] == want
            if split == "ci":
                ok = ok and e["category"] == SECTION4_CI_CATEGORIES
            _check(rows, split, "composition", "PASS" if ok else "FAIL",
                   "case kinds" + (" and categories" if split == "ci" else "") + " match section 4"
                   if ok else f"case kinds {e['case_kind']} differ from section 4 {want}")
        if split == "eval":
            _check(rows, split, "hand-written file", "PASS" if hw_present or e["n"] == e["synthetic_n"] else "FAIL",
                   f"{HANDWRITTEN_PATH} {'present' if hw_present else 'absent'}; "
                   f"{e['n'] - e['synthetic_n']} non-synthetic cases built")

        for name in CASE_CHECKS:
            c = e["checks"][name]
            bad = c["failed"]
            _check(rows, split, name, "FAIL" if bad else "PASS",
                   f"{c['checked'] - len(bad)} of {c['checked']} cases pass", bad)

        if split in COMMITTED_FILES:
            data_rel, comp_rel = COMMITTED_FILES[split]
            data_file, comp_file = export / data_rel, export / comp_rel
            if data_file.is_file():
                same = _sha_file(data_file) == e.get("split_jsonl_sha256")
                _check(rows, split, "committed file", "PASS" if same else "FAIL",
                       f"{data_rel} {'equals' if same else 'differs from'} the rebuilt split")
            else:
                _check(rows, split, "committed file", "SKIP", f"{data_rel} not in the ref")
            if comp_file.is_file():
                try:
                    comp = json.loads(comp_file.read_text(encoding="utf-8"))
                except ValueError:
                    comp = None
                bad = []
                if not isinstance(comp, dict):
                    bad.append("not a JSON object")
                else:
                    if comp.get("dataset_sha256") != e["dataset_sha256"]:
                        bad.append("dataset_sha256")
                    if data_file.is_file() and comp.get("file_sha256") != _sha_file(data_file):
                        bad.append("file_sha256")
                    composition = comp.get("composition") or {}
                    for key in ("case_kind", "category"):
                        if key in composition and composition[key] != e[key]:
                            bad.append(key)
                _check(rows, split, "COMPOSITION.json", "FAIL" if bad else "PASS",
                       f"{comp_rel} " + (f"differs in {', '.join(bad)}" if bad else "matches the rebuild"))

        recorded_key = {"dev": "dev", "ci": "ci", "eval": "eval_synthetic"}.get(split)
        if recorded_key:
            got = e["synthetic_dataset_sha256"] if split == "eval" else e["dataset_sha256"]
            ok = got == RECORDED_SECTION13[recorded_key]
            what = "146 synthetic eval cases" if split == "eval" else "split"
            _check(rows, split, "section 13 hash", "PASS" if ok else "FAIL",
                   f"dataset_sha256 of the {what} {'equals' if ok else 'differs from'} the recorded "
                   f"{RECORDED_SECTION13[recorded_key]}")
    return rows


def _log_files(log_dir: pathlib.Path) -> list[pathlib.Path]:
    files = [p for p in log_dir.rglob("*") if p.is_file() and p.suffix in (".eval", ".json")
             and p.name != "logs.json"]
    return sorted(files)


def check_logs(log_dir: pathlib.Path, child_splits: dict) -> list[dict]:
    """One row per log under log_dir: its header hashes against the rebuilt splits."""
    from inspect_ai.log import read_eval_log

    rows: list[dict] = []
    files = _log_files(log_dir) if log_dir.is_dir() else []
    if not files:
        _check(rows, "logs", "log headers", "FAIL", f"no Inspect logs under {log_dir}")
        return rows
    for f in files:
        name = f.relative_to(log_dir).as_posix()
        try:
            header = read_eval_log(str(f), header_only=True)
            meta = dict(header.eval.metadata or {})
        except Exception as e:  # noqa: BLE001
            _check(rows, "logs", name, "FAIL", f"not a readable Inspect log ({type(e).__name__})")
            continue
        split, subset = meta.get("split"), meta.get("subset")
        e = child_splits.get(split) if isinstance(split, str) else None
        if not e or not e.get("built"):
            _check(rows, "logs", name, "FAIL", f"split {split!r} was not rebuilt")
            continue
        bad = []
        if meta.get("split_seed") != e["split_seed"]:
            bad.append("split_seed")
        if meta.get("split_dataset_sha256") != e["dataset_sha256"]:
            bad.append("split_dataset_sha256")
        if subset is None:
            want = e["dataset_sha256"]
        elif subset == "pilot" and split == "dev":
            want = e.get("pilot_dataset_sha256")
        else:
            want = None
        if want is None or meta.get("dataset_sha256") != want:
            bad.append("dataset_sha256")
        label = f"split {split}, subset {subset or 'none'}"
        _check(rows, "logs", name, "FAIL" if bad else "PASS",
               f"{label}: " + (f"{', '.join(bad)} differ from the rebuild" if bad
                               else "dataset_sha256, split_dataset_sha256 and split_seed match"))
    return rows


def audit(ref: str = DEFAULT_REF, repo: pathlib.Path = ROOT, splits: list[str] | None = None,
          logs: pathlib.Path | None = None) -> dict:
    """The full audit report (no case content)."""
    splits = list(splits or ALL_SPLITS)
    commit = resolve_ref(repo, ref)
    with tempfile.TemporaryDirectory(prefix="icfr-audit-", ignore_cleanup_errors=True) as tmp:
        export = pathlib.Path(tmp) / "tree"
        export.mkdir()
        export_tree(repo, commit, export)
        child = run_child(export, splits)
        rows = evaluate(child, export, splits, ref=ref, commit=commit)
    if logs is not None:
        rows += check_logs(logs, child.get("splits", {}))
    summary = {}
    for split in splits:
        e = child.get("splits", {}).get(split, {})
        if e.get("built"):
            summary[split] = {k: e.get(k) for k in (
                "n", "case_kind", "category", "split_seed", "dataset_sha256", "split_jsonl_sha256",
                "synthetic_n", "synthetic_dataset_sha256", "pilot_n", "pilot_dataset_sha256") if k in e}
    return {"ref": ref, "commit": commit, "splits": summary, "checks": rows,
            "failed": sum(r["status"] == "FAIL" for r in rows)}


def to_text(report: dict) -> str:
    out = [f"Dataset audit of {report['ref']} (commit {report['commit']})", ""]
    by_scope: dict[str, list] = {}
    for r in report["checks"]:
        by_scope.setdefault(r["scope"], []).append(r)
    for scope, rows in by_scope.items():
        s = report["splits"].get(scope)
        if s:
            kinds = ", ".join(f"{k} {v}" for k, v in s["case_kind"].items())
            cats = ", ".join(f"{k} {v}" for k, v in s["category"].items())
            out.append(f"split {scope}: {s['n']} cases; case kinds: {kinds}")
            out.append(f"  categories: {cats}")
            out.append(f"  split seed {s['split_seed']}")
            out.append(f"  dataset_sha256      {s['dataset_sha256']}")
            if "split_jsonl_sha256" in s:
                out.append(f"  split_jsonl SHA-256 {s['split_jsonl_sha256']}")
            if "synthetic_dataset_sha256" in s:
                out.append(f"  dataset_sha256 of the {s['synthetic_n']} synthetic cases {s['synthetic_dataset_sha256']}")
            if "pilot_dataset_sha256" in s:
                out.append(f"  dataset_sha256 of the {s['pilot_n']}-case pilot subset {s['pilot_dataset_sha256']}")
        else:
            out.append(f"{scope}:")
        for r in rows:
            out.append(f"  {r['status']:<4}  {r['check']}: {r['detail']}")
            for f in r["failed"]:
                out.append(f"          {f['id']}: {f['reason']}")
        out.append("")
    n = len(report["checks"])
    out.append(f"{n} checks, {report['failed']} FAIL, "
               f"{sum(r['status'] == 'SKIP' for r in report['checks'])} SKIP. "
               + ("Result: FAIL" if report["failed"] else "Result: PASS"))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=DEFAULT_REF, help=f"git ref to audit (default {DEFAULT_REF}; HEAD works)")
    parser.add_argument("--repo", type=pathlib.Path, default=ROOT, help="git repository (default: this one)")
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=list(ALL_SPLITS))
    parser.add_argument("--logs", type=pathlib.Path, help="folder of Inspect logs whose headers to check")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--child", metavar="OUT", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args.child, args.splits)
    try:
        report = audit(args.ref, args.repo, args.splits, args.logs)
    except AuditError as e:
        sys.stderr.write(f"audit could not run: {e}\n")
        return 2
    sys.stdout.write(json.dumps(report, indent=1, sort_keys=True) + "\n" if args.json else to_text(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
