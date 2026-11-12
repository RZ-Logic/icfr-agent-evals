"""The judge-development set build (scripts/judge_dev_build.py) and import
(scripts/judge_dev_import.py).

Blindness: the selection (which outputs, which are seeded, their rules and target
levels), the cause list and the shuffle seed live only in the sealed selection input
outside the repo. These tests hold no real sample id with a seed rule or a target: the
rule and import tests run on synthetic cases, and the tests of the real selection read
the sealed file when it is on this machine and skip otherwise.

Build: the selection meets the section 7 stratification (every level 0 to 3 at least
3 times per LLM-graded sub-dimension), keeps the FM-000 epoch out (D-R10 C5) and uses
no case more than twice; every seed rule is mechanical (an omission seed only deletes
text found by its rule, an addition seed only inserts one template sentence with causes
from the sealed list, or a note's own words) and edits only the narrative field; no
cause collides with a dev case file, the anchors or a pilot output; no template's fixed
words occur in the anchors; the sheet is offline and blind. Import: the 20 records are
written in the README format with the committed anchors hash and graded_by RA, and a
bad grades file or a bad plan is refused with nothing written. No grade is ever
produced by code.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from casefactory import good_output, item, make_truth
from icfr_agent_evals import casefile, rubric, scoring
from icfr_agent_evals.solver import WORKER_PROMPT_SHA256

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "data" / "dev" / "dev.jsonl"
LOG = REPO.parent / "pilot-logs-2026-11-11" / "rescored-dr9" / "sonnet55-v004.eval"
SEALED_SELECTION = REPO.parent / "pilot-logs-2026-11-11" / "judge-dev" / "_sealed" / "judge-dev-selection.json"
GROUNDING = "faithfulness.narrative_grounding"
COVERAGE = "completeness.narrative_coverage"
FM000 = ("dev-std-007-ETH-M3", 2)


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


@pytest.fixture(scope="module")
def sealed():
    if not SEALED_SELECTION.is_file():
        pytest.skip("the sealed selection input is not on this machine")
    return B.load_selection(SEALED_SELECTION)


# --- Synthetic material (no real record, rule or target) --------------------------------

SYN_CAUSES = ("a teapot audit", "a comet relay", "a lantern sweep")


def _synthetic_picks() -> list:
    return [B.Pick(f"dev-unit-{i:03d}-BTC-M1", 1, i % 4, (i + 1) % 4) for i in range(20)]


def _syn_meta(asset: str = "BTC", items=None) -> dict:
    """Metadata of a synthetic case: BTC 2025-12, a $120,000.00 M2 shortfall, WD-2512-04417
    explains $80,000.00, JE-25-12-0193 is a cleared decoy, $40,000.00 is left unexplained."""
    truth = make_truth(asset) if items is None else make_truth(asset, items=items)
    notes = [{"ts": "2025-12-31T09:00", "by": "T. Example",
              "text": "Ops thinks the gap is a ticket backlog; sign-off pending from the desk lead"}]
    return {"truth": truth.to_dict(), "fixture": {"notes": notes}}


SYN_REFS = ("WD-2512-04417", "JE-25-12-0193")
SYN_NARR = ("On-chain reserves fall short of ledger liabilities by $120,000.00, a shortfall in the M2 (reportable) band. "
            "WD-2512-04417 explains $80,000.00. That leaves $40,000.00 unexplained. I excluded JE-25-12-0193.")


# --- The code and tests hold no sealed data ---------------------------------------------


def test_code_and_tests_hold_no_selection_cause_or_seed(sealed):
    sources = {p.name: p.read_text(encoding="utf-8") for p in (
        REPO / "scripts" / "judge_dev_build.py", REPO / "scripts" / "judge_dev_import.py", Path(__file__))}
    excluded_ids = {k[0] for k in B.EXCLUDED}
    for name, text in sources.items():
        assert str(sealed.shuffle_seed) not in text, name
        for p in sealed.picks:
            if p.sample_id not in excluded_ids:   # the FM-000 id appears only as the exclusion
                assert p.sample_id not in text, (name, "a selected sample id")
            if p.basis:
                assert p.basis not in text, name
            if p.note_words:
                assert p.note_words not in text, name
        for c in sealed.causes:
            assert c not in text, (name, "a sealed cause")
    assert not hasattr(B, "SELECTION") and not hasattr(B, "CAUSES") and not hasattr(B, "DEFAULT_SHUFFLE_SEED")


# --- The sealed selection (skipped where the sealed file is not on disk) -----------------


def test_sealed_selection_meets_every_rule(sealed):
    assert B.selection_problems(sealed.picks, sealed.causes) == []


def test_every_level_appears_at_least_three_times_per_subdimension(sealed):
    counts = B.level_counts(sealed.picks)
    assert set(counts) == set(rubric.LLM_GRADED_SUBDIMENSIONS)
    for d, by_level in counts.items():
        assert sorted(by_level) == [0, 1, 2, 3]
        assert all(n >= 3 for n in by_level.values()), d


def test_twenty_records_no_case_more_than_twice_and_fm000_out(sealed):
    assert len(sealed.picks) == 20
    keys = [(p.sample_id, p.epoch) for p in sealed.picks]
    assert len(set(keys)) == 20
    per_case: dict[str, int] = {}
    for p in sealed.picks:
        per_case[p.sample_id] = per_case.get(p.sample_id, 0) + 1
    assert max(per_case.values()) <= 2
    assert FM000 in B.EXCLUDED and FM000 in I.EXCLUDED and set(B.EXCLUDED) == set(I.EXCLUDED)
    assert FM000 not in keys
    assert all(p.sample_id in _dev_cases() for p in sealed.picks)


def test_natural_records_carry_no_rule_and_seeds_use_the_minimum(sealed):
    natural = [p for p in sealed.picks if p.source == "natural"]
    seeded = [p for p in sealed.picks if p.source != "natural"]
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


def _collision_texts() -> dict[str, str]:
    cases = _dev_cases()
    texts = {cid: row["input"] for cid, row in cases.items()}
    texts.update({f"{cid} (metadata)": json.dumps(row["metadata"], ensure_ascii=False) for cid, row in cases.items()})
    texts["anchors"] = B.anchor_text()
    return texts


def test_sealed_causes_collide_with_no_dev_case_file_and_no_anchor_text(sealed):
    assert len(_dev_cases()) == 30
    assert B.collision_hits(sealed.causes, _collision_texts()) == []


def test_sealed_causes_reuse_no_anchor_example_word(sealed):
    anchor_words = {"airdrop", "custodian", "rebate", "system", "migration", "harrow", "vale", "brokerage",
                    "finance", "director", "pending", "transfer", "timing"}
    for cause in sealed.causes:
        assert not set(B.content_words(cause)) & anchor_words
    assert len(set(sealed.causes)) == len(sealed.causes)


def test_sealed_causes_fit_the_cases_they_land_on(sealed):
    """No staking-adjacent cause on a SOL, ADA or ETH case; the whole-variance cause has no direction."""
    cases = _dev_cases()
    causes = list(sealed.causes)
    for p in sealed.picks:
        need = B.CAUSES_PER_RULE.get(p.rule or "", 0)
        used, causes = causes[:need], causes[need:]
        asset = casefile.truth_of(cases[p.sample_id]["metadata"]).asset
        for c in used:
            assert B.cause_fit_problem(c, p.rule, asset) is None
    assert causes == []


# --- Selection rules on a synthetic selection ------------------------------------------


def test_selection_problems_accept_a_synthetic_selection():
    assert B.selection_problems(_synthetic_picks(), ()) == []


def test_selection_problems_catch_each_break(monkeypatch):
    P = B.Pick
    picks = _synthetic_picks()
    monkeypatch.setitem(B.EXCLUDED, (picks[-1].sample_id, picks[-1].epoch), "test exclusion")
    assert any("excluded" in m for m in B.selection_problems(picks, ()))
    monkeypatch.delitem(B.EXCLUDED, (picks[-1].sample_id, picks[-1].epoch))
    thrice = picks[:-2] + [P(picks[0].sample_id, 2, 3, 0), P(picks[0].sample_id, 3, 2, 1)]
    assert any("more than 2" in m for m in B.selection_problems(thrice, ()))
    assert any("19 records" in m for m in B.selection_problems(picks[:-1], ()))
    assert any("fewer than 3" in m for m in B.selection_problems(picks[:10], ()))
    unknown = picks[:-1] + [P(picks[-1].sample_id, 1, 3, 0, "O-WHATEVER")]
    assert any("unknown rule" in m for m in B.selection_problems(unknown, ()))
    hedge = picks[:-1] + [P(picks[-1].sample_id, 1, 2, 0, "A-HEDGE")]
    assert any("need 1 causes" in m for m in B.selection_problems(hedge, ()))
    assert B.selection_problems(hedge, SYN_CAUSES[:1]) == []
    assert any("listed twice" in m for m in B.selection_problems(picks, (SYN_CAUSES[0], SYN_CAUSES[0])))
    adopt = picks[:-1] + [P(picks[-1].sample_id, 1, 2, 0, "A-ADOPT")]
    assert any("note words" in m for m in B.selection_problems(adopt, ()))


def test_load_selection_reads_and_refuses(tmp_path):
    path = tmp_path / "selection.json"
    doc = {"shuffle_seed": 7, "causes": list(SYN_CAUSES[:1]),
           "picks": [{"sample_id": p.sample_id, "epoch": p.epoch, "grounding": p.grounding, "coverage": p.coverage,
                      "rule": None, "basis": "", "note_words": None, "absent_word": None} for p in _synthetic_picks()]}
    path.write_text(json.dumps(doc), encoding="utf-8")
    sel = B.load_selection(path)
    assert sel.shuffle_seed == 7 and sel.causes == SYN_CAUSES[:1] and list(sel.picks) == _synthetic_picks()
    assert sel.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(B.Refused):
        B.load_selection(tmp_path / "missing.json")
    for bad in ({**doc, "shuffle_seed": "7"}, {**doc, "causes": "x"}, {**doc, "picks": [{"sample_id": "x"}]},
                {**doc, "picks": [dict(doc["picks"][0], extra=1)]}, [1, 2]):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(B.Refused):
            B.load_selection(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(B.Refused):
        B.load_selection(path)


def test_build_refuses_without_the_selection_input(tmp_path):
    with pytest.raises(B.Refused, match="selection input not found"):
        B.build(tmp_path / "no.eval", tmp_path / "out", DEV, tmp_path / "missing.json", force=False)


# --- Collision check and templates -----------------------------------------------------


def test_collision_check_finds_words_and_plurals():
    texts = {"case": "The Lunar payouts wallet held a Comet."}
    hits = B.collision_hits(["a lunar payout", "a comet relay", "a teapot audit"], texts)
    assert any("| lunar |" in h for h in hits)
    assert any("| payout |" in h for h in hits)  # plural form
    assert any("| comet |" in h for h in hits)
    assert not any(h.startswith("a teapot audit") for h in hits)
    # a word inside a longer word is not a hit
    assert B.collision_hits(["a tea"], {"case": "teapot shelf"}) == []


def test_template_fixed_words_do_not_occur_in_the_anchors():
    assert B.template_anchor_hits(B.TEMPLATES, B.anchor_text()) == []


def test_every_addition_template_is_checked_word_by_word():
    for rule, template in B.TEMPLATES.items():
        words = B.template_fixed_words(template)
        assert not set(words) & B.TEMPLATE_SHARED_WORDS
        if rule != "A-ADOPT":
            assert words, rule
            assert template in B.RULES[rule], rule
    # Only words every narrative already uses are exempt.
    assert B.TEMPLATE_SHARED_WORDS == {"the", "from", "unexplained", "surplus", "shortfall", "usd"}


@pytest.mark.parametrize("template, word", [
    ("The remaining {amount} reflects {cause}.", "remaining"),
    ("The remaining {amount} reflects {cause}.", "reflects"),
    ("The unexplained {amount} may reflect {cause}.", "may"),
    ("The unexplained {amount} may reflect {cause}.", "reflect"),
    ("The {direction} of {variance} reflects {cause1} and {cause2}.", "and"),
])
def test_template_check_catches_an_anchor_shape(template, word):
    assert f"X | {word}" in B.template_anchor_hits({"X": template}, B.anchor_text())


@pytest.mark.parametrize("cause, rule, asset, bad", [
    ("a validator tip", "A-PART", "SOL", True),
    ("a staking pause", "A-HEDGE", "ETH", True),
    ("a delegation slip", "A-HEDGE", "ADA", True),
    ("a validator tip", "A-PART", "BTC", False),
    ("a ledger loss", "A-WHOLE", "BTC", True),
    ("a lantern refund", "A-WHOLE", "XRP", True),
    ("a ledger loss", "A-PART", "BTC", False),
    ("a formula slip", "A-WHOLE", "SOL", False),
])
def test_cause_fit(cause, rule, asset, bad):
    assert (B.cause_fit_problem(cause, rule, asset) is not None) is bad


# --- Seed rules (synthetic cases) -------------------------------------------------------


def test_omit_remainder_drops_a_sentence_without_an_item():
    base = SYN_NARR.replace("unexplained.", "unexplained, and OPS-1 remains open.")
    out, deleted = B.omit_remainder(base, SYN_REFS)
    assert deleted == ["That leaves $40,000.00 unexplained, and OPS-1 remains open."]
    assert out == base.replace(" " + deleted[0], "")


@pytest.mark.parametrize("trigger", [", leaving", ", which leaves", ", and leaves"])
def test_omit_remainder_cuts_only_the_written_clause_beside_an_item(trigger):
    base = (f"Reserves fall short of liabilities by $120,000.00 (M2, reportable). WD-2512-04417 explains "
            f"$80,000.00{trigger} $40,000.00 unexplained under review; ticket OPS-1 is open.")
    out, deleted = B.omit_remainder(base, SYN_REFS)
    assert deleted == [f"{trigger} $40,000.00 unexplained under review; ticket OPS-1 is open"]
    assert out.endswith("WD-2512-04417 explains $80,000.00.")


@pytest.mark.parametrize("trigger", [", leaves", ", and leaving", ", and which leaves"])
def test_omit_remainder_takes_no_trigger_outside_the_rule(trigger):
    base = f"WD-2512-04417 explains $80,000.00{trigger} $40,000.00 unexplained."
    with pytest.raises(B.SeedError, match="no remainder clause"):
        B.omit_remainder(base, SYN_REFS)


def test_omit_remainder_refuses_when_remainder_wording_is_left():
    base = "WD-2512-04417 explains the whole amount. That leaves $0.00 unexplained."
    with pytest.raises(B.SeedError):
        B.omit_remainder(base, SYN_REFS)
    with pytest.raises(B.SeedError):
        B.omit_remainder("WD-2512-04417 explains $1.00.", SYN_REFS)


@pytest.mark.parametrize("first, span", [
    ("Reserves are $120,000.00 below liabilities, a shortfall in the M2 (reportable) band.",
     ", a shortfall in the M2 (reportable) band"),
    ("Reserves fall short of liabilities by $120,000.00 (M2, reportable).", " (M2, reportable)"),
    ("Reserves are below liabilities, a shortfall of $120,000.00, which is M2 (over the $100,000.00 threshold).",
     ", which is M2 (over the $100,000.00 threshold)"),
])
def test_omit_band_deletes_the_band_span_only(first, span):
    base = first + " WD-2512-04417 explains $80,000.00, leaving $40,000.00 unexplained."
    out, deleted = B.omit_band(base, "$120,000.00")
    assert deleted == [span]
    assert out == base.replace(span, "", 1)
    assert not re.search(r"\bM[123]\b|reportable|material-risk|threshold|\bband\b", out)


def test_omit_band_refuses_a_second_band_mention_or_a_lost_variance():
    base = ("Reserves are $120,000.00 below liabilities, a shortfall in the M2 (reportable) band. "
            "The preparer asked for M1.")
    with pytest.raises(B.SeedError):
        B.omit_band(base, "$120,000.00")
    with pytest.raises(B.SeedError):
        B.omit_band("Reserves fell, a shortfall of $5.00 in band M1 (below the floor).", "$5.00")


def test_insert_goes_after_the_last_unexplained_sentence():
    base = "A is $1.00 unexplained. B. C is still unexplained. D."
    assert B.insert_after_remainder(base, "X.") == "A is $1.00 unexplained. B. C is still unexplained. X. D."
    with pytest.raises(B.SeedError):
        B.insert_after_remainder("Nothing here. At all.", "X.")


def test_money_uses_the_narrative_style_and_needs_the_figure():
    assert B.money(-40000.00, "leaving USD 40,000.00 unexplained") == "USD 40,000.00"
    assert B.money(40000.00, "leaving $40,000.00 unexplained") == "$40,000.00"
    with pytest.raises(B.SeedError):
        B.money(40000.00, "leaving 40,000.00 USD unexplained")


@pytest.mark.parametrize("rule", ["A-HEDGE", "A-PART", "A-PART2", "A-WHOLE"])
def test_addition_seeds_insert_one_template_sentence(rule):
    causes = list(SYN_CAUSES)
    pick = B.Pick("dev-unit-001-BTC-M2", 1, 1, 2, rule)
    res = B.apply_rule(pick, SYN_NARR, _syn_meta(), causes)
    expected = B.TEMPLATES[rule].format(amount="$40,000.00", variance="$120,000.00", direction="shortfall",
                                        cause=SYN_CAUSES[0], cause1=SYN_CAUSES[0], cause2=SYN_CAUSES[1])
    assert res.inserted == [expected]
    assert res.deleted == []
    assert res.narrative == SYN_NARR.replace("unexplained.", "unexplained. " + expected, 1)
    assert causes == list(SYN_CAUSES[B.CAUSES_PER_RULE[rule]:])  # consumed in order
    assert expected in res.note and rule in res.note


def test_addition_seeds_refuse_a_cause_that_does_not_fit():
    with pytest.raises(B.SeedError, match="staking"):
        B.apply_rule(B.Pick("dev-unit-001-SOL-M2", 1, 1, 2, "A-PART"), SYN_NARR, _syn_meta("SOL"),
                     ["a validator tip"])
    with pytest.raises(B.SeedError, match="direction"):
        B.apply_rule(B.Pick("dev-unit-001-BTC-M2", 1, 0, 2, "A-WHOLE"), SYN_NARR, _syn_meta(), ["a ledger loss"])


def test_partial_seeds_refuse_a_case_without_an_explained_part():
    decoy = item("JE-25-12-0193", 40_000.00, date=casefile.period_cutoff_date("2025-12"), type="manual_adjustment",
                 open_status="cleared", recorded_in="both", role="decoy")
    meta = _syn_meta(items=(decoy,))
    narr = "Reserves fall short of liabilities by USD 120,000.00. The full USD 120,000.00 is unexplained."
    with pytest.raises(B.SeedError):
        B.apply_rule(B.Pick("dev-unit-003-BTC-M2", 1, 1, 2, "A-PART"), narr, meta, list(SYN_CAUSES))
    res = B.apply_rule(B.Pick("dev-unit-003-BTC-M2", 1, 2, 3, "A-HEDGE"), narr, meta, list(SYN_CAUSES))
    assert res.inserted == [B.TEMPLATES["A-HEDGE"].format(cause=SYN_CAUSES[0], amount="USD 120,000.00")]


def test_adoption_copies_note_words_verbatim_and_needs_a_silent_base():
    meta = _syn_meta()
    words = "sign-off pending from the desk lead"
    pick = B.Pick("dev-unit-002-BTC-M2", 1, 0, 3, "A-ADOPT", note_words=words, absent_word="sign-off")
    res = B.apply_rule(pick, SYN_NARR, meta, list(SYN_CAUSES))
    assert res.inserted == ["Sign-off pending from the desk lead."]
    assert any(words in n for n in B.note_texts(meta))
    assert res.causes == []
    with pytest.raises(B.SeedError):
        B.apply_rule(pick, SYN_NARR + " The sign-off was refused.", meta, list(SYN_CAUSES))
    other = B.Pick(pick.sample_id, 1, 0, 3, "A-ADOPT", note_words="words in no note", absent_word="zzz")
    with pytest.raises(B.SeedError):
        B.apply_rule(other, SYN_NARR, meta, list(SYN_CAUSES))


def test_omission_seed_via_apply_rule_records_each_deletion():
    res = B.apply_rule(B.Pick("dev-unit-004-BTC-M2", 1, 3, 0, "O-BAND+O-REM"), SYN_NARR, _syn_meta(), list(SYN_CAUSES))
    assert res.inserted == [] and res.causes == []
    assert res.deleted == [", a shortfall in the M2 (reportable) band", "That leaves $40,000.00 unexplained."]
    rebuilt = SYN_NARR
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
            "code": f"R-AAA{i}", "position": n - i, "sample_id": f"dev-unit-00{i}-BTC-M2", "epoch": 1 + i,
            "source": "seeded_addition" if i else "natural", "seed_note": f"Rule A-PART. secret note {i}" if i else None,
            "stratum_basis": f"basis text {i}", "file_name": f"dev-unit-00{i}-BTC-M2-e{1 + i}.json",
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


@pytest.mark.parametrize("leak", ["dev-unit-001-BTC-M2", "dev-hc-999-ZZZ-M9", "seeded_addition", "stratum",
                                  "Rule A-PART. secret note 1", "epoch", "EVALID123", "natural", "O-REM",
                                  "basis text 2"])
def test_blindness_check_catches_each_leak(leak):
    recs = _fake_records()
    sheet = B.render_sheet(recs).replace("</body>", f"<p>{leak}</p></body>")
    assert B.blindness_hits(sheet, recs, "EVALID123")


# --- Build on the pilot log (skipped where the log or the sealed input is not on disk) --


@pytest.fixture(scope="module")
def built(tmp_path_factory, sealed):
    if not LOG.is_file():
        pytest.skip("the pilot log of record is not on this machine")
    out = tmp_path_factory.mktemp("judge-dev")
    plan = B.build(LOG, out, DEV, SEALED_SELECTION, force=False)
    return out, plan


def test_build_writes_a_blind_sheet_and_a_full_plan(built, sealed):
    out, plan = built
    sheet = (out / B.SHEET_NAME).read_text(encoding="utf-8")
    assert B.blindness_hits(sheet, plan["records"], plan["log_eval_id"]) == []
    assert plan["worker_prompt_sha256"] == WORKER_PROMPT_SHA256
    assert plan["anchors_sha256"] == B.LOCKED_ANCHOR_SHA256
    assert plan["selection_sha256"] == sealed.sha256 == hashlib.sha256(SEALED_SELECTION.read_bytes()).hexdigest()
    assert plan["shuffle_seed"] == sealed.shuffle_seed and plan["causes"] == list(sealed.causes)
    assert plan["source_counts"] == {s: sum(1 for p in sealed.picks if p.source == s) for s in B.SOURCES}
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
    # The plan the build writes passes the import's checks, sheet hash included.
    I.load_plan(out / B.SEALED_DIR / B.PLAN_NAME)


def test_build_refuses_to_overwrite_and_is_deterministic(built, tmp_path):
    out, plan = built
    with pytest.raises(B.Refused):
        B.build(LOG, out, DEV, SEALED_SELECTION, force=False)
    again = B.build(LOG, tmp_path, DEV, SEALED_SELECTION, force=False)
    assert again["sheet_sha256"] == plan["sheet_sha256"]
    assert (tmp_path / B.SHEET_NAME).read_bytes() == (out / B.SHEET_NAME).read_bytes()


# --- Import ----------------------------------------------------------------------------


def _plan_file(tmp_path: Path, folder: str = "", **overrides) -> tuple[Path, dict]:
    truth = make_truth()
    base = good_output(truth)
    raw = json.dumps(base, separators=(",", ":"))
    seeded_raw = json.dumps(dict(base, narrative=base["narrative"] + " An added sentence."), separators=(",", ":"))
    records = []
    for i in range(20):
        seeded = i % 2 == 1
        records.append({
            "code": f"R-{i:04d}", "position": i + 1,
            "file_name": f"dev-unit-{i:03d}-BTC-M1-e1" + ("-seed-1" if seeded else "") + ".json",
            "sample_id": f"dev-unit-{i:03d}-BTC-M1", "epoch": 1, "log_eval_id": "EVALID",
            "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "source": "seeded_addition" if seeded else "natural", "seed_rule": "A-PART" if seeded else None,
            "seed_note": "Rule A-PART. note" if seeded else None, "seed_inserted": [], "seed_deleted": [], "seed_causes": [],
            "stratum": {GROUNDING: i % 4, COVERAGE: (i + 1) % 4}, "stratum_basis": "",
            "worker_output": seeded_raw if seeded else raw, "base_worker_output": raw,
        })
    plan = {"anchors_sha256": I.LOCKED_ANCHOR_SHA256, "worker_prompt_sha256": WORKER_PROMPT_SHA256,
            "excluded": [{"sample_id": FM000[0], "epoch": FM000[1], "reason": "FM-000"}],
            "sheet_file": "judge-dev-sheet.html", "sheet_sha256": "0" * 64, "records": records}
    plan.update(overrides)
    where = tmp_path / folder if folder else tmp_path
    where.mkdir(parents=True, exist_ok=True)
    path = where / "plan.json"
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
    # A column name given twice: csv.DictReader would keep only the last one.
    (lambda rows: [rows[0] + [GROUNDING]] + [r + ["3"] for r in rows[1:]], "more than once"),
    (lambda rows: [rows[0] + ["note"]] + [r + ["x"] for r in rows[1:]], "more than once"),
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
    broken = dict(plan, records=[dict(plan["records"][0], base_worker_output="not json")] + plan["records"][1:])
    plan_path.write_text(json.dumps(broken), encoding="utf-8")
    assert "base worker output does not parse" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))


def test_import_refuses_an_excluded_output(tmp_path, capsys, monkeypatch):
    # Listed in the plan's own excluded list.
    plan_path, plan = _plan_file(tmp_path)
    first = plan["records"][3]
    listed = dict(plan, excluded=plan["excluded"] + [{"sample_id": first["sample_id"], "epoch": first["epoch"],
                                                      "reason": "test"}])
    plan_path.write_text(json.dumps(listed), encoding="utf-8")
    assert "excluded output" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))
    # Listed in the import's own list, even when the plan leaves it out.
    plan_path.write_text(json.dumps(dict(plan, excluded=[])), encoding="utf-8")
    monkeypatch.setitem(I.EXCLUDED, (first["sample_id"], first["epoch"]), "test")
    assert "excluded output" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))
    assert FM000 in I.EXCLUDED


def test_import_refuses_a_change_outside_the_narrative(tmp_path, capsys):
    plan_path, plan = _plan_file(tmp_path)
    obj = json.loads(plan["records"][4]["base_worker_output"])
    obj["variance_amount_usd"] = obj["variance_amount_usd"] + 1
    bad = dict(plan, records=plan["records"][:4] + [dict(plan["records"][4], worker_output=json.dumps(obj))]
               + plan["records"][5:])
    plan_path.write_text(json.dumps(bad), encoding="utf-8")
    assert "outside the narrative" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))


@pytest.mark.parametrize("folder", ["", "_sealed"])
def test_import_checks_the_sheet_beside_the_plan(tmp_path, capsys, folder):
    sheet = tmp_path / "judge-dev-sheet.html"
    sheet.write_bytes(b"<!doctype html><title>sheet</title>")
    good = hashlib.sha256(sheet.read_bytes()).hexdigest()
    plan_path, plan = _plan_file(tmp_path, folder, sheet_sha256=good)
    assert I.load_plan(plan_path)["sheet_sha256"] == good
    plan_path, plan = _plan_file(tmp_path, folder, sheet_sha256="f" * 64)
    assert "does not match" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))
    plan_path, plan = _plan_file(tmp_path, folder, sheet_file="../judge-dev-sheet.html")
    assert "bare file name" in _refused(tmp_path, capsys, plan_path, _grades_file(tmp_path, plan))


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
