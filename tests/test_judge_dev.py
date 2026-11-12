"""The judge-development set build (scripts/judge_dev_build.py) and import
(scripts/judge_dev_import.py).

Build: the selection meets the section 7 stratification (every level 0 to 3 at least
3 times per LLM-graded sub-dimension), keeps the FM-000 epoch out (D-R10 C5) and uses
no case more than twice; every seed rule is mechanical (an omission seed only deletes
text found by its rule, an addition seed only inserts one template sentence with causes
from the fixed list, or a note's own words) and edits only the narrative field; the
fixed cause list collides with no dev case file and no anchor text; the sheet is
offline and blind (no source, stratum, seed note, sample id, epoch, eval id or file
name). Import: the 20 records are written in the README format with the committed
anchors hash and graded_by RA, and a missing, repeated, unknown or out-of-range grade
is refused with nothing written. No grade is ever produced by code.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from casefactory import good_output, make_truth
from icfr_agent_evals import rubric, scoring
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "data" / "dev" / "dev.jsonl"
LOG = REPO.parent / "pilot-logs-2026-11-11" / "rescored-dr9" / "sonnet55-v004.eval"
GROUNDING = "faithfulness.narrative_grounding"
COVERAGE = "completeness.narrative_coverage"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("judge_dev_build")
I = _load("judge_dev_import")


def _dev_cases() -> dict[str, dict]:
    return {json.loads(line)["id"]: json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines() if line.strip()}


# --- Selection and stratification -----------------------------------------------------


def test_selection_meets_every_rule():
    assert B.selection_problems(B.SELECTION) == []


def test_every_level_appears_at_least_three_times_per_subdimension():
    counts = B.level_counts(B.SELECTION)
    assert set(counts) == set(rubric.LLM_GRADED_SUBDIMENSIONS)
    for d, by_level in counts.items():
        assert sorted(by_level) == [0, 1, 2, 3]
        assert all(n >= 3 for n in by_level.values()), (d, by_level)


def test_twenty_records_no_case_more_than_twice_and_fm000_out():
    assert len(B.SELECTION) == 20
    keys = [(p.sample_id, p.epoch) for p in B.SELECTION]
    assert len(set(keys)) == 20
    per_case: dict[str, int] = {}
    for p in B.SELECTION:
        per_case[p.sample_id] = per_case.get(p.sample_id, 0) + 1
    assert max(per_case.values()) <= 2
    assert ("dev-std-007-ETH-M3", 2) in B.EXCLUDED
    assert ("dev-std-007-ETH-M3", 2) not in keys
    assert all(p.sample_id in _dev_cases() for p in B.SELECTION)


def test_natural_records_carry_no_rule_and_seeds_use_the_minimum():
    natural = [p for p in B.SELECTION if p.source == "natural"]
    seeded = [p for p in B.SELECTION if p.source != "natural"]
    assert natural and all(p.rule is None for p in natural)
    # Each seeded record fills a level no natural record of the set reaches.
    natural_levels = {d: {p.stratum[d] for p in natural} for d in (GROUNDING, COVERAGE)}
    for p in seeded:
        if p.source == "seeded_addition":
            assert p.grounding not in natural_levels[GROUNDING]
        else:
            assert p.coverage not in natural_levels[COVERAGE]
    # Three records per level reached only by seeds: no more seeds than that.
    missing = sum(1 for d in natural_levels for lv in (0, 1, 2, 3) if lv not in natural_levels[d])
    assert len(seeded) == 3 * missing


def test_selection_problems_catch_each_break():
    P = B.Pick
    bad = list(B.SELECTION[:-1]) + [P("dev-std-007-ETH-M3", 2, 3, 0, "O-BAND+O-REM")]
    assert any("excluded" in m for m in B.selection_problems(bad))
    thrice = list(B.SELECTION[:-1]) + [P("dev-hc-005-USDT-M3", 3, 3, 0, "O-BAND+O-REM")]
    assert any("more than 2" in m for m in B.selection_problems(thrice))
    short = list(B.SELECTION[:-1])
    msgs = B.selection_problems(short)
    assert any("19 records" in m for m in msgs)
    assert any("level 0 has 2" in m for m in msgs)
    unknown = list(B.SELECTION[:-1]) + [P("dev-hc-010-XRP-M3", 2, 3, 0, "O-WHATEVER")]
    assert any("unknown rule" in m for m in B.selection_problems(unknown))


# --- Fixed cause list ------------------------------------------------------------------


def _collision_texts() -> dict[str, str]:
    cases = _dev_cases()
    texts = {cid: row["input"] for cid, row in cases.items()}
    texts.update({f"{cid} (metadata)": json.dumps(row["metadata"], ensure_ascii=False) for cid, row in cases.items()})
    texts["anchors"] = B.anchor_text()
    return texts


def test_causes_collide_with_no_dev_case_file_and_no_anchor_text():
    assert len(_dev_cases()) == 30
    assert B.collision_hits(B.CAUSES, _collision_texts()) == []


def test_causes_reuse_no_anchor_example_word():
    anchor_words = {"airdrop", "custodian", "rebate", "system", "migration", "harrow", "vale", "brokerage",
                    "finance", "director", "pending", "transfer", "timing"}
    for cause in B.CAUSES:
        assert not set(B.content_words(cause)) & anchor_words, cause
    assert len(set(B.CAUSES)) == len(B.CAUSES)


def test_the_selection_needs_exactly_the_fixed_list():
    used = sum(B.CAUSES_PER_RULE.get(p.rule or "", 0) for p in B.SELECTION)
    assert used == len(B.CAUSES)


def test_collision_check_finds_words_and_plurals():
    texts = {"case": "The Staking payouts wallet held a Bridge."}
    hits = B.collision_hits(["a staking payout", "a bridge refund", "a card chargeback"], texts)
    assert any("| staking |" in h for h in hits)
    assert any("| payout |" in h for h in hits)  # plural form
    assert any("| bridge |" in h for h in hits)
    assert not any(h.startswith("a card chargeback") for h in hits)
    # a word inside a longer word is not a hit
    assert B.collision_hits(["a card"], {"case": "cardano chain"}) == []


# --- Seed rules ------------------------------------------------------------------------

REFS = ("WD-2511-041445", "DP-2511-421421", "JE-25-11-7121")


def test_omit_remainder_drops_a_sentence_without_an_item():
    base = ("Reserves exceed liabilities by $26,889.19, a surplus in the M2 (reportable) band. "
            "WD-2511-041445 explains $19,125.36. That leaves $7,763.83 unexplained, and OPS-2511-81805 remains open. "
            "I excluded JE-25-11-7121.")
    out, deleted = B.omit_remainder(base, REFS)
    assert deleted == ["That leaves $7,763.83 unexplained, and OPS-2511-81805 remains open."]
    assert out == base.replace(" " + deleted[0], "")


def test_omit_remainder_cuts_only_the_clause_beside_an_item():
    base = ("Reserves exceed liabilities by $26,889.19 (M2, reportable). JE-25-11-7121 is a suspense entry "
            "and is excluded, leaving $7,763.83 unexplained under investigation (ticket OPS-1).")
    out, deleted = B.omit_remainder(base, REFS)
    assert deleted == [", leaving $7,763.83 unexplained under investigation (ticket OPS-1)"]
    assert out.endswith("JE-25-11-7121 is a suspense entry and is excluded.")


def test_omit_remainder_refuses_when_remainder_wording_is_left():
    base = "WD-2511-041445 explains the whole amount. That leaves $0.00 unexplained."
    with pytest.raises(B.SeedError):
        B.omit_remainder(base, REFS)
    with pytest.raises(B.SeedError):
        B.omit_remainder("WD-2511-041445 explains $1.00.", REFS)


@pytest.mark.parametrize("first, span", [
    ("Reserves are $62,878.92 below liabilities, a shortfall in the M2 (reportable) band.",
     ", a shortfall in the M2 (reportable) band"),
    ("Reserves exceed liabilities by a surplus of $62,878.92 (M3, material-risk).", " (M3, material-risk)"),
    ("Reserves are below liabilities, a shortfall of $62,878.92, which is M3 (just over the $250,000.00 threshold).",
     ", which is M3 (just over the $250,000.00 threshold)"),
])
def test_omit_band_deletes_the_band_span_only(first, span):
    base = first + " WD-2511-041445 explains $1.00, leaving $62,877.92 unexplained."
    out, deleted = B.omit_band(base, "$62,878.92")
    assert deleted == [span]
    assert out == base.replace(span, "", 1)
    assert not re.search(r"\bM[123]\b|reportable|material-risk|threshold|\bband\b", out)


def test_omit_band_refuses_a_second_band_mention_or_a_lost_variance():
    base = ("Reserves are $62,878.92 below liabilities, a shortfall in the M2 (reportable) band. "
            "The preparer asked for M1.")
    with pytest.raises(B.SeedError):
        B.omit_band(base, "$62,878.92")
    with pytest.raises(B.SeedError):
        B.omit_band("Reserves fell, a shortfall of $5.00 in band M1 (below the floor).", "$5.00")


def test_insert_goes_after_the_last_unexplained_sentence():
    base = "A is $1.00 unexplained. B. C is still unexplained. D."
    assert B.insert_after_remainder(base, "X.") == "A is $1.00 unexplained. B. C is still unexplained. X. D."
    with pytest.raises(B.SeedError):
        B.insert_after_remainder("Nothing here. At all.", "X.")


def test_money_uses_the_narrative_style_and_needs_the_figure():
    assert B.money(-7763.83, "leaving USD 7,763.83 unexplained") == "USD 7,763.83"
    assert B.money(7763.83, "leaving $7,763.83 unexplained") == "$7,763.83"
    with pytest.raises(B.SeedError):
        B.money(7763.83, "leaving 7,763.83 USD unexplained")


def _hc007():
    case = _dev_cases()["dev-hc-007-SOL-M2"]
    return case["metadata"]


HC007_NARR = ("On-chain reserves exceed ledger liabilities by $26,889.19, a surplus in the M2 (reportable) band. "
              "WD-2511-041445 and DP-2511-421421 explain $19,125.36. That leaves $7,763.83 unexplained. "
              "I excluded JE-25-11-7121.")


@pytest.mark.parametrize("rule, level_sentence", [
    ("A-HEDGE", "The unexplained $7,763.83 may reflect {c0}."),
    ("A-PART", "The remaining $7,763.83 reflects {c0}."),
    ("A-PART2", "The remaining $7,763.83 reflects {c0} and {c1}."),
    ("A-WHOLE", "The surplus of $26,889.19 reflects {c0}."),
])
def test_addition_seeds_insert_one_template_sentence(rule, level_sentence):
    causes = list(B.CAUSES)
    pick = B.Pick("dev-hc-007-SOL-M2", 1, 1, 2, rule)
    res = B.apply_rule(pick, HC007_NARR, _hc007(), causes)
    expected = level_sentence.format(c0=B.CAUSES[0], c1=B.CAUSES[1])
    assert res.inserted == [expected]
    assert res.deleted == []
    assert res.narrative.replace(" " + expected, "", 1) == HC007_NARR
    assert causes == list(B.CAUSES[B.CAUSES_PER_RULE[rule]:])  # consumed in order
    assert expected in res.note and rule in res.note


def test_partial_seeds_refuse_a_case_without_an_explained_part():
    meta = _dev_cases()["dev-hc-003-ETH-M1"]["metadata"]  # nothing explained
    narr = "Reserves exceed liabilities by USD 18,738.48. The full USD 18,738.48 is unexplained."
    with pytest.raises(B.SeedError):
        B.apply_rule(B.Pick("dev-hc-003-ETH-M1", 1, 1, 2, "A-PART"), narr, meta, list(B.CAUSES))
    res = B.apply_rule(B.Pick("dev-hc-003-ETH-M1", 1, 2, 3, "A-HEDGE"), narr, meta, list(B.CAUSES))
    assert res.inserted == [f"The unexplained USD 18,738.48 may reflect {B.CAUSES[0]}."]


def test_adoption_copies_note_words_verbatim_and_needs_a_silent_base():
    meta = _dev_cases()["dev-hc-001-BTC-M2"]["metadata"]
    pick = next(p for p in B.SELECTION if p.rule == "A-ADOPT")
    narr = "Reserves are $34,068.30 below liabilities. Four items explain it, leaving $0.00 unexplained. I excluded two."
    res = B.apply_rule(pick, narr, meta, list(B.CAUSES))
    words = pick.note_words
    assert res.inserted == [words[0].upper() + words[1:] + "."]
    assert any(words in n for n in B.note_texts(meta))
    assert res.causes == []
    with pytest.raises(B.SeedError):
        B.apply_rule(pick, narr + " The waiver was refused.", meta, list(B.CAUSES))
    other = B.Pick(pick.sample_id, 1, 0, 3, "A-ADOPT", note_words="words in no note", absent_word="zzz")
    with pytest.raises(B.SeedError):
        B.apply_rule(other, narr, meta, list(B.CAUSES))


def test_omission_seed_via_apply_rule_records_each_deletion():
    res = B.apply_rule(B.Pick("dev-hc-007-SOL-M2", 3, 3, 0, "O-BAND+O-REM"), HC007_NARR, _hc007(), list(B.CAUSES))
    assert res.inserted == [] and res.causes == []
    rebuilt = HC007_NARR
    for span in res.deleted:
        assert rebuilt.count(span) == 1
        rebuilt = rebuilt.replace(span, "", 1).replace("  ", " ")
    assert rebuilt == res.narrative
    for span in res.deleted:
        assert json.dumps(span) in res.note


def test_replace_narrative_keeps_every_other_byte():
    truth = make_truth()
    obj = good_output(truth, narrative="Old text. Leaving $1.00 unexplained.")
    raw = json.dumps(obj, separators=(",", ":"))
    new = "Old text."
    out = B.replace_narrative(raw, obj["narrative"], new)
    assert out == raw.replace(json.dumps(obj["narrative"]), json.dumps(new))
    a, b = scoring.parse_worker_output(raw).parsed, scoring.parse_worker_output(out).parsed
    assert {k: v for k, v in a.items() if k != "narrative"} == {k: v for k, v in b.items() if k != "narrative"}
    with pytest.raises(B.SeedError):
        B.replace_narrative(raw, "text that is not there", new)


# --- Sheet blindness -------------------------------------------------------------------


def _fake_records(n: int = 3) -> list[dict]:
    recs = []
    for i in range(n):
        recs.append({
            "code": f"R-AAA{i}", "position": n - i, "sample_id": f"dev-hc-00{i}-BTC-M2", "epoch": 1 + i,
            "source": "seeded_addition" if i else "natural", "seed_note": f"Rule A-PART. secret note {i}" if i else None,
            "stratum_basis": f"basis text {i}", "file_name": f"dev-hc-00{i}-BTC-M2-e{1 + i}.json",
            "_narrative": f"Narrative number {i}.", "_case_input": f"Case file text {i}",
            "_case_facts": [], "_output_facts": [],
        })
    return recs


def test_sheet_is_blind_offline_and_ordered_by_position():
    recs = _fake_records()
    sheet = B.render_sheet(recs)
    assert B.blindness_hits(sheet, recs, "EVALID123") == []
    assert not re.search(r"(?:src|href)\s*=\s*[\"']?(?:https?:)?//", sheet)
    assert "<link" not in sheet and "@import" not in sheet and "http://" not in sheet and "https://" not in sheet
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', sheet, re.S).group(1))
    assert [r["code"] for r in data["records"]] == ["R-AAA2", "R-AAA1", "R-AAA0"]
    assert set(data["records"][0]) == {"code", "narrative", "case_input", "case_facts", "output_facts"}
    assert data["anchors_sha256"] == B.LOCKED_ANCHOR_SHA256
    for d in (GROUNDING, COVERAGE):
        assert data["scope"][d] == rubric.LLM_GRADED_SCOPE[d]
        for lv in range(4):
            assert data["criteria"][d][str(lv)] == rubric.LLM_GRADED_ANCHORS[d][lv][0]


@pytest.mark.parametrize("leak", ["dev-hc-001-BTC-M2", "seeded_addition", "stratum", "Rule A-PART. secret note 1",
                                  "epoch", "EVALID123", "natural", "O-REM", "basis text 2"])
def test_blindness_check_catches_each_leak(leak):
    recs = _fake_records()
    sheet = B.render_sheet(recs).replace("</body>", f"<p>{leak}</p></body>")
    assert B.blindness_hits(sheet, recs, "EVALID123")


# --- Build on the pilot log (skipped where the log is not on disk) ---------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not LOG.is_file():
        pytest.skip("the pilot log of record is not on this machine")
    out = tmp_path_factory.mktemp("judge-dev")
    plan = B.build(LOG, out, DEV, B.DEFAULT_SHUFFLE_SEED, force=False)
    return out, plan


def test_build_writes_a_blind_sheet_and_a_full_plan(built):
    out, plan = built
    sheet = (out / B.SHEET_NAME).read_text(encoding="utf-8")
    assert B.blindness_hits(sheet, plan["records"], plan["log_eval_id"]) == []
    assert plan["worker_prompt_sha256"] == WORKER_PROMPT_SHA256
    assert plan["anchors_sha256"] == B.LOCKED_ANCHOR_SHA256
    assert plan["source_counts"] == {"natural": 5, "seeded_omission": 6, "seeded_addition": 9}
    assert len(plan["records"]) == 20
    assert sorted(r["position"] for r in plan["records"]) == list(range(1, 21))
    for r in plan["records"]:
        parsed = scoring.parse_worker_output(r["worker_output"]).parsed
        base = scoring.parse_worker_output(r["base_worker_output"]).parsed
        assert parsed["narrative"] in sheet or json.dumps(parsed["narrative"], ensure_ascii=False)[1:-1] in sheet
        assert {k: v for k, v in parsed.items() if k != "narrative"} == {k: v for k, v in base.items() if k != "narrative"}
        if r["source"] == "natural":
            assert r["worker_output"] == r["base_worker_output"] and r["seed_note"] is None
        else:
            assert r["worker_output"] != r["base_worker_output"] and r["seed_note"]
            for s in r["seed_inserted"]:
                assert s in parsed["narrative"] and s not in base["narrative"]
            for s in r["seed_deleted"]:
                assert s in base["narrative"] and s not in parsed["narrative"]


def test_build_refuses_to_overwrite_and_is_deterministic(built, tmp_path):
    out, plan = built
    with pytest.raises(B.Refused):
        B.build(LOG, out, DEV, B.DEFAULT_SHUFFLE_SEED, force=False)
    again = B.build(LOG, tmp_path, DEV, B.DEFAULT_SHUFFLE_SEED, force=False)
    assert again["sheet_sha256"] == plan["sheet_sha256"]
    assert (tmp_path / B.SHEET_NAME).read_bytes() == (out / B.SHEET_NAME).read_bytes()


# --- Import ----------------------------------------------------------------------------


def _plan_file(tmp_path: Path, **overrides) -> tuple[Path, dict]:
    truth = make_truth()
    raw = json.dumps(good_output(truth), separators=(",", ":"))
    records = []
    for i in range(20):
        seeded = i % 2 == 1
        records.append({
            "code": f"R-{i:04d}", "position": i + 1, "file_name": f"dev-std-{i:03d}-BTC-M1-e1" + ("-seed-1" if seeded else "") + ".json",
            "sample_id": f"dev-std-{i:03d}-BTC-M1", "epoch": 1, "log_eval_id": "EVALID", "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "source": "seeded_addition" if seeded else "natural", "seed_rule": "A-PART" if seeded else None,
            "seed_note": "Rule A-PART. note" if seeded else None, "seed_inserted": [], "seed_deleted": [], "seed_causes": [],
            "stratum": {GROUNDING: i % 4, COVERAGE: (i + 1) % 4}, "stratum_basis": "", "worker_output": raw,
            "base_worker_output": raw,
        })
    plan = {"anchors_sha256": I.LOCKED_ANCHOR_SHA256, "worker_prompt_sha256": WORKER_PROMPT_SHA256, "records": records}
    plan.update(overrides)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, plan


def _grades_file(tmp_path: Path, plan: dict, mutate=None) -> Path:
    rows = [["record_code", GROUNDING, COVERAGE, "note", "graded_at"]]
    for n, r in enumerate(plan["records"]):
        rows.append([r["code"], str(n % 4), str((n + 2) % 4), "a, note" if n == 0 else "", "2026-11-13T10:15:00-05:00"])
    if mutate:
        rows = mutate(rows)
    path = tmp_path / "grades.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\r\n").writerows(rows)
    return path


def test_import_writes_the_readme_format(tmp_path):
    plan_path, plan = _plan_file(tmp_path)
    grades = _grades_file(tmp_path, plan)
    out = tmp_path / "judge-dev"
    out.mkdir()
    assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(out)]) == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(r["file_name"] for r in plan["records"])
    by_code = {r["code"]: r for r in plan["records"]}
    for n, r in enumerate(plan["records"]):
        raw = (out / r["file_name"]).read_bytes()
        assert b"\r\n" not in raw and raw.endswith(b"\n")
        rec = json.loads(raw.decode("utf-8"))
        assert tuple(rec) == I.RECORD_FIELDS
        assert rec["grades"] == {GROUNDING: n % 4, COVERAGE: (n + 2) % 4}
        assert rec["anchors_sha256"] == I.LOCKED_ANCHOR_SHA256 == rubric.llm_anchor_sha256()
        assert rec["graded_by"] == "RA" and rec["graded_on"] == "2026-11-13"
        assert rec["stratum"] == by_code[r["code"]]["stratum"] and rec["source"] == r["source"]
        assert rec["worker_output"] == r["worker_output"] and rec["seed_note"] == r["seed_note"]
        assert "code" not in rec and "seed_rule" not in rec and "stratum_basis" not in rec


def _refused(tmp_path, capsys, plan_path, grades, out=None) -> str:
    if out is None:
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
    assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(out)]) == 3
    assert not any(out.iterdir()) or out.name == "full"
    return capsys.readouterr().err


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: [rows[0]] + [[r[0], "", r[2], r[3], r[4]] if i == 3 else r for i, r in enumerate(rows[1:], 1)], "no grade"),
    (lambda rows: rows[:1] + [[rows[1][0], "4"] + rows[1][2:]] + rows[2:], "not 0, 1, 2 or 3"),
    (lambda rows: rows[:1] + [[rows[1][0], "2.0"] + rows[1][2:]] + rows[2:], "not 0, 1, 2 or 3"),
    (lambda rows: rows[:1] + [[rows[1][0], rows[1][1], "-1"] + rows[1][3:]] + rows[2:], "not 0, 1, 2 or 3"),
    (lambda rows: rows[:-1], "with no grades"),
    (lambda rows: rows + [rows[1]], "appears twice"),
    (lambda rows: rows + [["R-9999", "1", "1", "", "2026-11-13T10:00:00"]], "not in the plan"),
    (lambda rows: [[c for c in r[:2]] + r[3:] for r in rows], "lacks the column"),
    (lambda rows: rows[:1] + [rows[1][:4] + [""]] + rows[2:], "graded_at"),
    (lambda rows: rows[:1] + [rows[1][:4] + ["2026-13-45T10:00:00"]] + rows[2:], "not a date"),
])
def test_import_refuses_bad_grades(tmp_path, capsys, mutate, message):
    plan_path, plan = _plan_file(tmp_path)
    grades = _grades_file(tmp_path, plan, mutate)
    err = _refused(tmp_path, capsys, plan_path, grades)
    assert message in err


@pytest.mark.parametrize("overrides, message", [
    ({"anchors_sha256": "0" * 64}, "other anchors"),
    ({"worker_prompt_sha256": "0" * 64}, "locked prompt"),
])
def test_import_refuses_a_plan_on_other_anchors_or_prompt(tmp_path, capsys, overrides, message):
    plan_path, plan = _plan_file(tmp_path, **overrides)
    err = _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))
    assert message in err


def test_import_refuses_a_short_plan_or_an_unparsed_output(tmp_path, capsys):
    plan_path, plan = _plan_file(tmp_path)
    short = dict(plan, records=plan["records"][:19])
    plan_path.write_text(json.dumps(short), encoding="utf-8")
    assert "20 records" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))
    broken = dict(plan, records=[dict(plan["records"][0], worker_output="not json")] + plan["records"][1:])
    plan_path.write_text(json.dumps(broken), encoding="utf-8")
    assert "does not parse" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))


def test_import_never_overwrites(tmp_path, capsys):
    plan_path, plan = _plan_file(tmp_path)
    grades = _grades_file(tmp_path, plan)
    out = tmp_path / "full"
    out.mkdir()
    keep = out / plan["records"][5]["file_name"]
    keep.write_text("owner file", encoding="utf-8")
    err = _refused(tmp_path, capsys, plan_path, grades, out)
    assert "already exist" in err
    assert keep.read_text(encoding="utf-8") == "owner file"
    assert sorted(p.name for p in out.iterdir()) == [keep.name]


def test_import_reads_a_bom_and_ignores_extra_columns(tmp_path):
    plan_path, plan = _plan_file(tmp_path)
    grades = _grades_file(tmp_path, plan, lambda rows: [r + ["x"] for r in rows])
    grades.write_bytes(b"\xef\xbb\xbf" + grades.read_bytes())
    out = tmp_path / "o"
    out.mkdir()
    assert I.main(["--grades", str(grades), "--plan", str(plan_path), "--out", str(out)]) == 0
    assert len(list(out.iterdir())) == 20

