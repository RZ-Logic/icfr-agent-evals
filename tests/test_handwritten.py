"""The hand-written evaluation cases: the loader (icfr_agent_evals.handwritten), the eval
split hook (dataset.build_cases) and the key builder (scripts/build_handwritten.py).

No evaluation-split case is read: the loader tests relabel the generated hard cases of
the CI regression split (the section 4 composition, 30 edge and 4 per AC category), and
the builder tests parse rendered dev-split cases and compare the key code computes from
their text with the generator's own answer key.
"""

from __future__ import annotations

import dataclasses
import html
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from icfr_agent_evals.casefile import SCHEMA_VERSION, CaseTruth, split_seed
from icfr_agent_evals.dataset import CI_HARD_COMPOSITION, build_cases
from icfr_agent_evals.handwritten import (
    HANDWRITTEN_CASES, HandwrittenError, composition_gaps, load_handwritten,
)
from icfr_agent_evals.hardcases import build_hard_cases

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stand_ins():
    """54 generated hard cases relabeled as hand-written evaluation cases."""
    cases = build_hard_cases("ci", split_seed("ci"), CI_HARD_COMPOSITION)
    out = []
    for i, c in enumerate(cases, start=1):
        t = dataclasses.replace(c.truth, split="eval", case_id=f"eval-hw-{i:03d}")
        out.append({"id": t.case_id, "input": c.rendered,
                    "metadata": {"schema_version": SCHEMA_VERSION, "split": "eval", "truth": t.to_dict(),
                                 "fixture": c.fixture}})
    assert len(out) == HANDWRITTEN_CASES
    return out


def write(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_absent_file_means_no_handwritten_cases(tmp_path):
    assert load_handwritten(tmp_path / "missing.jsonl") is None


def test_loader_reads_the_section_4_composition(stand_ins, tmp_path):
    cases = load_handwritten(write(tmp_path / "hw.jsonl", stand_ins))
    assert len(cases) == 54
    kinds = Counter(c.truth.case_kind for c in cases)
    assert kinds == {"edge": 30, "adversarial": 24}
    assert Counter(c.truth.category for c in cases if c.truth.case_kind == "adversarial") == {
        f"AC-{k}": 4 for k in range(1, 7)}


def test_loader_refuses_a_missing_ac_case(stand_ins, tmp_path):
    drop = next(i for i, r in enumerate(stand_ins) if r["metadata"]["truth"]["category"] == "AC-3")
    with pytest.raises(HandwrittenError, match="3 AC-3 cases, not 4"):
        load_handwritten(write(tmp_path / "hw.jsonl", stand_ins[:drop] + stand_ins[drop + 1:]))


def test_loader_refuses_an_id_that_is_not_the_key_case_id(stand_ins, tmp_path):
    bad = [dict(r) for r in stand_ins]
    bad[0] = {**bad[0], "id": "eval-hw-999"}
    with pytest.raises(HandwrittenError, match="is not the answer key's case id"):
        load_handwritten(write(tmp_path / "hw.jsonl", bad))


def test_composition_gaps_name_split_and_id_problems(stand_ins):
    truths = [CaseTruth.from_dict(r["metadata"]["truth"]) for r in stand_ins]
    truths[0] = dataclasses.replace(truths[0], split="ci", case_id="ci-hc-001")
    gaps = composition_gaps(truths)
    assert any("split 'ci'" in g for g in gaps) and any("not a hand-written case id" in g for g in gaps)


def test_eval_split_takes_the_handwritten_cases_when_the_file_exists(stand_ins, tmp_path):
    path = write(tmp_path / "hw.jsonl", stand_ins)
    cases = build_cases("eval", handwritten_path=path)
    assert len(cases) == 200
    assert Counter(c.truth.case_kind for c in cases) == {"standard": 146, "edge": 30, "adversarial": 24}
    assert [c.truth.case_id for c in cases] == [c.truth.case_id for c in build_cases("eval", handwritten_path=path)]


def test_eval_split_without_the_file_is_the_146_synthetic_cases(tmp_path):
    cases = build_cases("eval", handwritten_path=tmp_path / "missing.jsonl")
    assert len(cases) == 146 and {c.truth.case_kind for c in cases} == {"standard"}


# --- scripts/build_handwritten.py ------------------------------------------------------------

def _load_builder():
    spec = importlib.util.spec_from_file_location("build_handwritten", REPO / "scripts" / "build_handwritten.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_handwritten"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def dev_plain():
    """Rendered dev cases with no memo-only item and no injection: their keys follow from
    the schedule rows alone, as the builder computes them."""
    rows = [json.loads(line) for line in (REPO / "data" / "dev" / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows:
        t = CaseTruth.from_dict(r["metadata"]["truth"])
        if t.injection is None and not any(i.channel == "memo_only" for i in t.items):
            out.append((r["input"], t))
    assert len(out) >= 8
    return out


def test_builder_key_matches_the_generator_key_on_dev_cases(builder, dev_plain):
    for n, (text, gen) in enumerate(dev_plain, start=1):
        num = f"{n:03d}"
        p = builder.parse_case(num, text)
        p["text"] = text
        truth, _fixture, gaps = builder.build(num, p, "standard", None, {})
        assert not gaps, (gen.case_id, gaps)
        assert truth.variance_usd == gen.variance_usd, gen.case_id
        assert (truth.direction, truth.materiality_band) == (gen.direction, gen.materiality_band), gen.case_id
        assert (truth.explained_usd, truth.unexplained_usd, truth.evidence_state) == (
            gen.explained_usd, gen.unexplained_usd, gen.evidence_state), gen.case_id
        assert {i.reference for i in truth.eligible_items()} == {i.reference for i in gen.eligible_items()}, gen.case_id


def test_builder_reads_wallet_labels_with_spaces(builder, dev_plain):
    text = next(t for t, _ in dev_plain if "Hot 1 0x" in t or "Hot 1 bc1" in t or "Hot 1 addr1" in t)
    p = builder.parse_case("001", text)
    labels = {w["label"]: w["address"] for w in p["wallets"]}
    for row in p["open_items"]:
        assert row["wallet_label"] in labels and row["wallet_address"] == labels[row["wallet_label"]]


def test_builder_refuses_a_set_that_breaks_the_composition(builder, dev_plain, tmp_path, capsys):
    blocks = "".join(f'<details class="case" id="case-{n:03d}"><summary>Case {n:03d}</summary><pre>{html.escape(t)}</pre></details>'
                     for n, (t, _) in enumerate(dev_plain[:2], start=1))
    (tmp_path / "book.html").write_text(blocks, encoding="utf-8")
    (tmp_path / "cat.json").write_text(json.dumps({f"{n:03d}": {"case_kind": "edge", "category": "band_boundary"}
                                                   for n in (1, 2)}), encoding="utf-8")
    (tmp_path / "j.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "hw.jsonl"
    code = builder.main(["--casebook", str(tmp_path / "book.html"), "--categories", str(tmp_path / "cat.json"),
                         "--judgments", str(tmp_path / "j.json"), "--out", str(out)])
    assert code == builder.EXIT_REFUSED and not out.exists()
    err = capsys.readouterr().err
    assert "2 edge cases, not 30" in err
