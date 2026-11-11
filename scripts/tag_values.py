"""Print the values PRE-REGISTRATION-PHASE-1.md section 13 records at the tag.

Read-only. It builds the dev, CI and evaluation splits in memory by code, hashes
them, counts their composition from ground-truth fields and reads version
constants. It never scores anything, never makes a model call, never writes a
file and never prints case content. Building evaluation-split cases by code and
reading ids, hashes, counts, composition and ground-truth fields is allowed before
the tag (section 4, ratified 2026-11-04).

Run:
  python scripts/tag_values.py              JSON on stdout
  python scripts/tag_values.py --markdown   section 13 value-cell drafts

Exit code 0 when every pre-registered value checked here matches, 1 when any
mismatch is listed under checks.mismatches. Warnings (a draft prompt version,
uncommitted generator files, a pre-release dataset version, an environment off
its pin) are printed but do not change the exit code.

Generator source set and source SHA-256 (the "source SHA-256" of generator
version G in the section 13 generator row):
  - The set is the static import closure of the module icfr_agent_evals.dataset,
    the module that defines build_cases. Starting from that module, every
    `import` and `from ... import` statement anywhere in the file (function-level
    imports included, so the lazy `from .hardcases import build_hard_cases`
    inside build_cases is followed) that names a module of the icfr_agent_evals
    package is resolved to its file, and the walk repeats on each new file. The
    package's __init__.py is included, because importing any submodule executes
    it. Modules outside the package (inspect_ai, the standard library) are not in
    the set; their versions are pinned separately (section 13 pin rows).
  - The closure is computed, never hand-picked, and is a module-level superset of
    what build_cases executes. tests/test_tag_values.py checks it against the
    modules a fresh interpreter actually loads to build the dev and CI splits.
  - Each file's working-tree bytes are hashed with SHA-256. The source SHA-256 is
    SHA-256 over the UTF-8 bytes of the lines "<path> <sha256>\\n", one per file,
    sorted by path, where <path> is repo-relative with forward slashes (for
    example "icfr_agent_evals/generator.py") and <sha256> is lowercase hex.
  - head_source_sha256 applies the same rule to the blobs at git HEAD, so a clean
    tree gives the same value; generator_dirty lists set files with uncommitted
    or untracked changes, and last_generator_commit is the newest commit that
    touched any set file (stable when a later commit records section 13 values).

Scorer source set (the code the score of record depends on): the same rule from two
roots, icfr_agent_evals.scorers (the run-time scorers) and icfr_agent_evals.trials (the
settlement behind trials.figures). It is reported with its combined SHA-256, the files
outside the generator set, dirty files and the last commit, like the generator set. A
file whose SHA-256 is not the value section 13 records (RECORDED_FILE_SHA256) is a
warning, since section 13 re-records it at the tag.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import platform
import ssl
import subprocess
import sys
from collections import Counter
from importlib import metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "icfr_agent_evals"
CLOSURE_ROOT = "icfr_agent_evals.dataset"
SCORER_ROOTS = ("icfr_agent_evals.scorers", "icfr_agent_evals.trials")
# File SHA-256 values section 13 records before the tag, each re-recorded at the tag; a
# difference is a warning (re-record it), not a mismatch.
RECORDED_FILE_SHA256 = {
    "icfr_agent_evals/narrative.py": "985a01df13700c96959aaf2520269848100be38162521177c8f87d05698942e2",
}
SPLITS = ("dev", "ci", "eval")

# Values the pre-registration states, checked against what the code produces.
PREREGISTERED = {
    "eval_seed": 20260607,
    "split_seeds": {"eval": 20260607, "dev": 21260607, "ci": 22260607, "calibration": 23260607},
    "composition": {
        "eval": {"standard": 146},  # the 54 hand-written cases join in the post-tag build (section 4)
        "ci": {"standard": 146, "edge": 30, "adversarial": 24},
        "dev": {"standard": 14, "edge": 6, "adversarial": 10},
    },
    "ci_categories": {
        "band_boundary": 10, "ambiguous_narrative": 10, "incomplete_data": 10,
        "AC-1": 4, "AC-2": 4, "AC-3": 4, "AC-4": 4, "AC-5": 4, "AC-6": 4,
    },
    "dev_pilot_cases": 20,
    "draft_prompt": ("v0.0.4-draft", "8eee5a57bafbf6499dbed8a4c4e91807cb8ba6d0466346cba614bed5f7cae24d"),
}

PINS = {  # section 13 pin rows; a difference is a warning, not a mismatch
    # Pre-registered interpreter. The 3.12.16 cross-check leg and the uv 0.13.6 minimum
    # are stated in requirements.txt; this script checks only the interpreter it runs on.
    "python": "3.14.10",
    "inspect-ai": "0.3.278",
    "inspect-scout": "0.5.5",
    "anthropic": "1.12.1",
    "openai": "3.27.0",
    "pydantic": "2.13.4",
}


# --- Generator source set ---------------------------------------------------------------

def _module_file(module: str, root: pathlib.Path) -> pathlib.Path | None:
    """The file of a package module under root, or None when it is not one."""
    parts = module.split(".")
    if parts[0] != PACKAGE:
        return None
    base = root.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _imported_modules(path: pathlib.Path, module: str) -> set[str]:
    """Every absolute module name an import statement anywhere in the file names."""
    is_pkg = path.name == "__init__.py"
    package = module if is_pkg else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                if node.level > 1:
                    base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            found.add(target)
            # `from pkg import name` may name a submodule (`from . import narrative_vocab`).
            found.update(f"{target}.{alias.name}" for alias in node.names if alias.name != "*")
    return found


def generator_source_files(root: pathlib.Path = ROOT, start: str = CLOSURE_ROOT) -> list[str]:
    """Repo-relative paths of the generator source set (see the module docstring)."""
    seen: dict[str, pathlib.Path] = {}
    todo = [start]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        path = _module_file(module, root)
        if path is None:
            continue
        seen[module] = path
        # Importing a submodule executes every parent package's __init__.py.
        parts = module.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        todo.extend(_imported_modules(path, module))
    if start not in seen:
        raise FileNotFoundError(f"closure root {start!r} not found under {root}")
    return sorted(p.relative_to(root).as_posix() for p in seen.values())


def scorer_source_files(root: pathlib.Path = ROOT) -> list[str]:
    """Repo-relative paths of the scorer source set: the static import closures of the run-time
    scorers and of the settlement that computes the score of record (trials.figures), by the
    generator set's rule. It overlaps the generator set; scorer_only lists the files outside it."""
    return sorted(set().union(*(generator_source_files(root, start) for start in SCORER_ROOTS)))


def recorded_file_warnings(hashes: dict[str, str]) -> list[str]:
    """One warning per file whose SHA-256 is not the value section 13 records for it."""
    return [f"{rel} SHA-256 {hashes.get(rel)} is not the section 13 value {want}; re-record it at the tag"
            for rel, want in RECORDED_FILE_SHA256.items() if hashes.get(rel) != want]


def combine_hashes(file_hashes: dict[str, str]) -> str:
    """SHA-256 over sorted "path sha256" lines, one per file."""
    blob = "".join(f"{path} {digest}\n" for path, digest in sorted(file_hashes.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_hashes(paths: list[str], root: pathlib.Path = ROOT) -> dict[str, str]:
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def source_sha256(paths: list[str], root: pathlib.Path = ROOT) -> str:
    return combine_hashes(file_hashes(paths, root))


# --- git ----------------------------------------------------------------------------------

def _git(*args: str, root: pathlib.Path = ROOT) -> bytes | None:
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                             capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_facts(paths: list[str], root: pathlib.Path = ROOT) -> dict:
    head = _git("rev-parse", "HEAD", root=root)
    last = _git("log", "-1", "--format=%H", "--", *paths, root=root)
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *paths, root=root)
    head_hashes = {}
    for p in paths:
        blob = _git("cat-file", "blob", f"HEAD:{p}", root=root)
        if blob is None:
            head_hashes = None
            break
        head_hashes[p] = hashlib.sha256(blob).hexdigest()
    dirty = None
    if status is not None:
        dirty = sorted(line[3:].strip() for line in status.decode("utf-8", "replace").splitlines() if line.strip())
    return {
        "git_head": head.decode().strip() if head else None,
        "last_generator_commit": (last.decode().strip() or None) if last is not None else None,
        "generator_dirty": dirty,
        "head_source_sha256": combine_hashes(head_hashes) if head_hashes else None,
    }


# --- Values ---------------------------------------------------------------------------------

def _versions() -> dict:
    # The full build string and the linked OpenSSL show a different interpreter build at the tag,
    # even when the Python version number is the same.
    env = {"python": platform.python_version(), "python_build": sys.version, "openssl": ssl.OPENSSL_VERSION}
    for dist in ("inspect-ai", "inspect-scout", "anthropic", "openai", "pydantic"):
        try:
            env[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[dist] = None
    return env


def collect(root: pathlib.Path = ROOT) -> dict:
    """Every tag-time value, as one JSON-serializable dict. Nothing is scored."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from icfr_agent_evals import narrative_vocab, solver
    from icfr_agent_evals.casefile import DATASET_VERSION, EVAL_SEED, SCHEMA_VERSION, dataset_sha256
    from icfr_agent_evals.casefile import SPLITS as ALL_SPLITS
    from icfr_agent_evals.casefile import split_seed, truth_of
    from icfr_agent_evals.dataset import build_dataset, select_subset
    from icfr_agent_evals.generator import GENERATOR_VERSION
    from icfr_agent_evals.hardcases import HARD_GENERATOR_VERSION

    mismatches: list[str] = []
    warnings: list[str] = []

    paths = generator_source_files(root)
    hashes = file_hashes(paths, root)
    generator = {
        "generator_version": GENERATOR_VERSION,
        "hard_generator_version": HARD_GENERATOR_VERSION,
        "source_files": hashes,
        "source_sha256": combine_hashes(hashes),
        **git_facts(paths, root),
    }
    if generator["generator_dirty"]:
        warnings.append(f"generator source files have uncommitted changes: {generator['generator_dirty']}")
    if generator["head_source_sha256"] and generator["head_source_sha256"] != generator["source_sha256"]:
        warnings.append("working-tree generator source SHA-256 differs from HEAD")
    for name, value in (("GENERATOR_VERSION", GENERATOR_VERSION), ("HARD_GENERATOR_VERSION", HARD_GENERATOR_VERSION)):
        if "-" in value.partition("-")[2]:
            warnings.append(f"{name} {value} carries a pre-release label")

    spaths = scorer_source_files(root)
    shashes = file_hashes(spaths, root)
    sgit = git_facts(spaths, root)
    scorer = {"source_files": shashes, "source_sha256": combine_hashes(shashes),
              "scorer_only": sorted(set(spaths) - set(paths)), "last_scorer_commit": sgit["last_generator_commit"],
              "scorer_dirty": sgit["generator_dirty"], "head_source_sha256": sgit["head_source_sha256"]}
    if scorer["scorer_dirty"]:
        warnings.append(f"scorer source files have uncommitted changes: {scorer['scorer_dirty']}")
    if scorer["head_source_sha256"] and scorer["head_source_sha256"] != scorer["source_sha256"]:
        warnings.append("working-tree scorer source SHA-256 differs from HEAD")
    warnings.extend(recorded_file_warnings(shashes))

    seeds = {"eval_seed": EVAL_SEED, "split_seeds": {s: split_seed(s, EVAL_SEED) for s in ALL_SPLITS}}
    if EVAL_SEED != PREREGISTERED["eval_seed"]:
        mismatches.append(f"EVAL_SEED {EVAL_SEED} != pre-registered {PREREGISTERED['eval_seed']}")
    for s, want in PREREGISTERED["split_seeds"].items():
        if seeds["split_seeds"].get(s) != want:
            mismatches.append(f"{s} split seed {seeds['split_seeds'].get(s)} != pre-registered {want}")
    if seeds["split_seeds"]["ci"] == EVAL_SEED:
        mismatches.append("CI regression split seed equals the evaluation seed")

    splits: dict[str, dict] = {}
    for split in SPLITS:
        samples = list(build_dataset(split=split, seed=EVAL_SEED))
        truths = [truth_of(s.metadata) for s in samples]
        kinds = dict(sorted(Counter(t.case_kind for t in truths).items()))
        cats = dict(sorted(Counter(t.category or "standard" for t in truths).items()))
        splits[split] = {
            "n": len(samples),
            "dataset_sha256": dataset_sha256(samples),
            "case_kind": kinds,
            "category": cats,
        }
        if kinds != PREREGISTERED["composition"][split]:
            mismatches.append(f"{split} composition {kinds} != pre-registered {PREREGISTERED['composition'][split]}")
        if split == "ci":
            want = {"standard": 146, **PREREGISTERED["ci_categories"]}
            if cats != dict(sorted(want.items())):
                mismatches.append(f"ci categories {cats} != pre-registered {dict(sorted(want.items()))}")
        if split == "dev":
            pilot = select_subset(samples, "dev", "pilot")
            splits["dev_pilot"] = {"n": len(pilot), "dataset_sha256": dataset_sha256(pilot)}
            if len(pilot) != PREREGISTERED["dev_pilot_cases"]:
                mismatches.append(f"dev pilot has {len(pilot)} cases, expected {PREREGISTERED['dev_pilot_cases']}")
    ci = splits["ci"]["case_kind"]
    splits["ci_matches_preregistered"] = ci == PREREGISTERED["composition"]["ci"]

    dev_file = root / "data" / "dev" / "dev.jsonl"
    if dev_file.is_file():
        from icfr_agent_evals.dataset import split_jsonl

        committed = hashlib.sha256(dev_file.read_bytes()).hexdigest()
        built = hashlib.sha256(split_jsonl("dev", EVAL_SEED)).hexdigest()
        splits["dev_file"] = {"path": "data/dev/dev.jsonl", "file_sha256": committed, "matches_build": committed == built}
        if committed != built:
            mismatches.append("data/dev/dev.jsonl does not match the dev split the generator builds")

    recomputed = hashlib.sha256(solver.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    prompt = {
        "source": "icfr_agent_evals/solver.py",
        "worker_prompt_version": solver.WORKER_PROMPT_VERSION,
        "worker_prompt_sha256": solver.WORKER_PROMPT_SHA256,
        "recomputed_sha256": recomputed,
        "locked": not solver.WORKER_PROMPT_VERSION.endswith("-draft"),
        "matches_section3_draft": (solver.WORKER_PROMPT_VERSION, recomputed) == PREREGISTERED["draft_prompt"],
        "locked_from_section3_draft": recomputed == PREREGISTERED["draft_prompt"][1]
        and not solver.WORKER_PROMPT_VERSION.endswith("-draft"),
    }
    if prompt["locked"] and recomputed != PREREGISTERED["draft_prompt"][1]:
        mismatches.append("locked worker prompt text differs from the section 3 draft (v0.0.4-draft)")
    if recomputed != solver.WORKER_PROMPT_SHA256:
        mismatches.append("WORKER_PROMPT_SHA256 does not match the SHA-256 of SYSTEM_PROMPT")
    if not prompt["locked"]:
        warnings.append(f"worker prompt version {solver.WORKER_PROMPT_VERSION} is a draft; section 13 records the locked version")

    dataset = {"dataset_version": DATASET_VERSION, "schema_version": SCHEMA_VERSION,
               "dataset_version_prerelease": "-" in DATASET_VERSION}
    if dataset["dataset_version_prerelease"]:
        warnings.append(f"DATASET_VERSION {DATASET_VERSION} is a pre-release; the post-tag build records its own SemVer")

    related = {
        "vocab_sha256": narrative_vocab.vocab_sha256(),
        "file_sha256": file_hashes(["icfr_agent_evals/narrative_vocab.py", "icfr_agent_evals/rubric.py"], root),
    }

    env = _versions()
    for name, want in PINS.items():
        if env.get(name) != want:
            warnings.append(f"{name} {env.get(name)} is off its pin {want}")

    return {
        "generator": generator,
        "scorer": scorer,
        "seeds": seeds,
        "splits": splits,
        "dataset": dataset,
        "worker_prompt": prompt,
        "related": related,
        "environment": env,
        "checks": {"mismatches": mismatches, "warnings": warnings},
    }


def to_markdown(v: dict) -> str:
    g, s, sp, p, d = v["generator"], v["seeds"], v["splits"], v["worker_prompt"], v["dataset"]
    ci = sp["ci"]["case_kind"]
    files = ", ".join(f"`{f}`" for f in g["source_files"])
    rows = [
        ("Locked worker prompt and scaffold version, and the prompt's SHA-256",
         f"worker_prompt_version {p['worker_prompt_version']}, worker_prompt_sha256 {p['worker_prompt_sha256']} "
         f"(`{p['source']}`)" + ("" if p["locked"] else ". DRAFT: lock the version before the tag")),
        ("Generator version G",
         f"generator_version {g['generator_version']}, hard_generator_version {g['hard_generator_version']}; last generator commit "
         f"{g['last_generator_commit']}; source SHA-256 {g['source_sha256']} over {len(g['source_files'])} files "
         f"({files}), defined in `scripts/tag_values.py`; evaluation seed {s['eval_seed']}; CI regression split seed "
         f"{s['split_seeds']['ci']}; CI composition, computed from the built split, {ci.get('standard', 0)} standard "
         f"synthetic, {ci.get('edge', 0)} edge-type, {ci.get('adversarial', 0)} adversarial-type; dataset_sha256 at "
         f"this commit: dev {sp['dev']['dataset_sha256']}, ci {sp['ci']['dataset_sha256']}, eval (146 synthetic) "
         f"{sp['eval']['dataset_sha256']}"),
        ("Planned dataset SemVer for the post-tag build",
         f"current DATASET_VERSION {d['dataset_version']}, SCHEMA_VERSION {d['schema_version']}; the planned SemVer "
         f"is the owner's entry"),
        ("Extractor vocabulary (rubric row)",
         f"vocab_sha256() {v['related']['vocab_sha256']}"),
        ("Scorer source set (proposed rubric row sentence)",
         f"scorer source SHA-256 {v['scorer']['source_sha256']} over {len(v['scorer']['source_files'])} files, "
         f"defined in `scripts/tag_values.py`; last scorer commit {v['scorer']['last_scorer_commit']}; outside the "
         "generator set: " + ", ".join(f"`{f}` {v['scorer']['source_files'][f]}" for f in v["scorer"]["scorer_only"])),
    ]
    out = ["| Item | Value at tag (draft) |", "|---|---|"]
    out += [f"| {item} | {value} |" for item, value in rows]
    checks = v["checks"]
    out.append("")
    out.append(f"Git HEAD when computed: {g['git_head']}. Mismatches: {len(checks['mismatches'])}. "
               f"Warnings: {len(checks['warnings'])}.")
    out += [f"- mismatch: {m}" for m in checks["mismatches"]]
    out += [f"- warning: {w}" for w in checks["warnings"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", action="store_true", help="print section 13 value-cell drafts")
    args = parser.parse_args(argv)
    values = collect()
    if args.markdown:
        sys.stdout.write(to_markdown(values))
    else:
        sys.stdout.write(json.dumps(values, indent=1, sort_keys=True) + "\n")
    return 1 if values["checks"]["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
