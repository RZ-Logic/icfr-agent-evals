"""The committed dev split is exactly what the generator produces.

data/dev/dev.jsonl is the dev split used for prompt elicitation and the pilot. A
generator change that moves it must regenerate the file (scripts/export_dev.py),
which this test enforces, so the dev set behind any pilot result is reproducible.
"""

import hashlib
import json
import pathlib

from icfr_agent_evals.dataset import split_jsonl

DEV_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "dev"


def test_committed_dev_file_matches_the_generator():
    committed = (DEV_DIR / "dev.jsonl").read_bytes()
    assert committed == split_jsonl("dev")


def test_recorded_sha256_matches_the_committed_file():
    summary = json.loads((DEV_DIR / "COMPOSITION.json").read_text(encoding="utf-8"))
    assert summary["file_sha256"] == hashlib.sha256((DEV_DIR / "dev.jsonl").read_bytes()).hexdigest()
    assert summary["composition"]["n"] == 30 and summary["composition"]["pilot"] == 20
    for other in ("eval", "ci"):
        d = summary["disjointness"][other]
        assert d["shared_case_ids"] == d["shared_template_instances"] == d["shared_wallets_or_references"] == 0
