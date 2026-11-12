"""Write the dev split to data/dev/ and record its SHA-256, composition and disjointness.

The dev split is the only split written to disk before the pre-registration tag. The
evaluation and CI splits are built in memory only to compare case ids, template
instances, wallets and references as sets; no case content from them is printed or
written.

Run: python scripts/export_dev.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import argparse
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icfr_agent_evals.casefile import EVAL_SEED, dataset_sha256, split_seed, truth_of  # noqa: E402
from icfr_agent_evals.dataset import build_dataset, split_jsonl  # noqa: E402
from icfr_agent_evals.generator import GENERATOR_VERSION  # noqa: E402
from icfr_agent_evals.scoring import check_wallet_labels_in_table  # noqa: E402


def keys(samples):
    ids, instances, entities = set(), set(), set()
    for s in samples:
        t = truth_of(s.metadata)
        ids.add(str(s.id))
        instances.add(t.template_instance)
        entities.update(t.scope_wallets)
        for item in t.items:
            entities.add(item.reference)
            entities.update(item.aliases)
    return ids, instances, entities


def main() -> int:
    out_dir = ROOT / "data" / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = list(build_dataset(split="dev", seed=EVAL_SEED))
    check_wallet_labels_in_table(dev)  # D-R9 m7, on the split this script writes, before it is written
    blob = split_jsonl("dev", EVAL_SEED)
    (out_dir / "dev.jsonl").write_bytes(blob)

    dev_keys = keys(dev)
    disjoint = {}
    for other in ("eval", "ci"):
        other_keys = keys(build_dataset(split=other, seed=EVAL_SEED))
        disjoint[other] = {
            "seed": split_seed(other, EVAL_SEED),
            "shared_case_ids": len(dev_keys[0] & other_keys[0]),
            "shared_template_instances": len(dev_keys[1] & other_keys[1]),
            "shared_wallets_or_references": len(dev_keys[2] & other_keys[2]),
        }

    truths = [truth_of(s.metadata) for s in dev]

    def count(f):
        return dict(sorted(collections.Counter(f(t) for t in truths).items()))

    summary = {
        "file": "data/dev/dev.jsonl",
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "dataset_sha256": dataset_sha256(dev),
        "generator_version": GENERATOR_VERSION,
        "dev_seed": split_seed("dev", EVAL_SEED),
        "composition": {
            "n": len(dev),
            "case_kind": count(lambda t: t.case_kind),
            "category": count(lambda t: t.category or "standard"),
            "asset": count(lambda t: t.asset),
            "band": count(lambda t: t.materiality_band),
            "period": count(lambda t: t.period),
            "direction": count(lambda t: t.direction),
            "evidence_state": count(lambda t: t.evidence_state),
            "memo_only_link": sum(t.has_memo_only_link for t in truths),
            "pilot": sum(bool(t.notes.get("pilot")) for t in truths),
            "pilot_by_kind": dict(sorted(collections.Counter(
                ("standard-memo" if t.has_memo_only_link else "standard") if t.case_kind == "standard" else t.category
                for t in truths if t.notes.get("pilot")
            ).items())),
        },
        "disjointness": disjoint,
    }
    (out_dir / "COMPOSITION.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline='\n')
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    # argparse so that -h prints this docstring and exits before any write (F2-10, N1).
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.exit(main())
