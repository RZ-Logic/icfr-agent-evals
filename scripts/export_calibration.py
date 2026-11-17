"""Write the calibration split to data/calibration/ and record its SHA-256, composition and disjointness.

The calibration split is the 40-case calibration subset of pre-registration section
4: 20 standard cases and 20 generated hard cases, used only to align the grader with
human grades. It sits outside the evaluation and dev splits, so its cases may be
public. The evaluation, CI and dev splits are built in memory only to compare case
ids, template instances, wallets, references, transaction hashes and canaries as
sets; no case content from them is printed or written.

Run: python scripts/export_calibration.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icfr_agent_evals.casefile import (  # noqa: E402
    EVAL_SEED,
    REFERENCE_FORMAT,
    dataset_sha256,
    matches_format,
    split_seed,
    truth_of,
)
from icfr_agent_evals.dataset import (  # noqa: E402
    CALIBRATION_HARD_COMPOSITION,
    CALIBRATION_STANDARD_CASES,
    build_dataset,
    split_jsonl,
)
from icfr_agent_evals.generator import GENERATOR_VERSION  # noqa: E402
from icfr_agent_evals.hardcases import HARD_GENERATOR_VERSION  # noqa: E402
from icfr_agent_evals.scoring import check_wallet_labels_in_table  # noqa: E402

SPLIT = "calibration"
OTHER_SPLITS = ("eval", "ci", "dev")
DEFAULT_OUT_DIR = ROOT / "data" / "calibration"


def _export_dev():
    """scripts/export_dev.py, imported by path for its disjointness keys."""
    spec = importlib.util.spec_from_file_location("export_dev", ROOT / "scripts" / "export_dev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wide_keys(samples) -> dict[str, set[str]]:
    """Every identifier a case renders that another split must not share: the full
    wallet table, every reference among the valid entities, every on-chain transaction
    hash and the AC-1 canary."""
    keys: dict[str, set[str]] = {name: set() for name in ("wallets", "references", "tx_hashes", "canaries")}
    for s in samples:
        t = truth_of(s.metadata)
        fixture = s.metadata["fixture"]
        keys["wallets"] |= {w["address"] for w in fixture["wallets"]}
        keys["references"] |= {e for e in t.valid_entities if matches_format(e, REFERENCE_FORMAT)}
        keys["tx_hashes"] |= {ln["tx"] for ln in fixture["onchain_extract"]["lines"]}
        if t.injection is not None:
            keys["canaries"].add(t.injection.canary)
    return keys


def disjointness(samples, others: dict[str, list]) -> dict[str, dict[str, int]]:
    """Counts of what the calibration samples share with each other split, as sets."""
    narrow = _export_dev().keys
    mine, mine_wide = narrow(samples), wide_keys(samples)
    out = {}
    for name, other in others.items():
        theirs, theirs_wide = narrow(other), wide_keys(other)
        row = {
            "seed": split_seed(name, EVAL_SEED),
            "shared_case_ids": len(mine[0] & theirs[0]),
            "shared_template_instances": len(mine[1] & theirs[1]),
            "shared_wallets_or_references": len(mine[2] & theirs[2]),
        }
        for key in mine_wide:
            row[f"shared_{key}"] = len(mine_wide[key] & theirs_wide[key])
        out[name] = row
    return out


def composition(samples) -> dict:
    truths = [truth_of(s.metadata) for s in samples]

    def count(f, ts=truths):
        return dict(sorted(collections.Counter(f(t) for t in ts).items()))

    standard = [t for t in truths if t.case_kind == "standard"]
    return {
        "n": len(truths),
        "case_kind": count(lambda t: t.case_kind),
        "category": count(lambda t: t.category or "standard"),
        "asset": count(lambda t: t.asset),
        "band": count(lambda t: t.materiality_band),
        "period": count(lambda t: t.period),
        "direction": count(lambda t: t.direction),
        "evidence_state": count(lambda t: t.evidence_state),
        "memo_only_link": sum(t.has_memo_only_link for t in truths),
        "standard": {
            "n": len(standard),
            "asset": count(lambda t: t.asset, standard),
            "band": count(lambda t: t.materiality_band, standard),
            "period": count(lambda t: t.period, standard),
            "direction": count(lambda t: t.direction, standard),
            "evidence_state": count(lambda t: t.evidence_state, standard),
            "memo_only_link": sum(t.has_memo_only_link for t in standard),
        },
        "template_variant": count(lambda t: t.notes.get("template_variant") or "standard"),
    }


def export(out_dir: pathlib.Path = DEFAULT_OUT_DIR, seed: int = EVAL_SEED) -> dict:
    """Write calibration.jsonl and COMPOSITION.json into out_dir; return the summary."""
    cal = list(build_dataset(split=SPLIT, seed=seed))
    expected = CALIBRATION_STANDARD_CASES + sum(CALIBRATION_HARD_COMPOSITION.values())
    if len(cal) != expected:
        raise ValueError(f"calibration split has {len(cal)} cases, expected {expected}")
    check_wallet_labels_in_table(cal)  # D-R9 m7, on the split this script writes, before it is written
    blob = split_jsonl(SPLIT, seed)
    others = {name: list(build_dataset(split=name, seed=seed)) for name in OTHER_SPLITS}
    summary = {
        "file": "data/calibration/calibration.jsonl",
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "dataset_sha256": dataset_sha256(cal),
        "generator_version": GENERATOR_VERSION,
        "hard_generator_version": HARD_GENERATOR_VERSION,
        "calibration_seed": split_seed(SPLIT, seed),
        "hard_composition": dict(sorted(CALIBRATION_HARD_COMPOSITION.items())),
        "composition": composition(cal),
        "disjointness": disjointness(cal, others),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.jsonl").write_bytes(blob)
    (out_dir / "COMPOSITION.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8", newline="\n")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR,
                    help="folder to write calibration.jsonl and COMPOSITION.json (default: data/calibration)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    # argparse first, so -h prints this docstring and exits before any build or write.
    args = parse_args(argv)
    summary = export(args.out_dir)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
