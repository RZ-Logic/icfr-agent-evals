"""The judge-development set build (scripts/judge_dev_build.py) and import
(scripts/judge_dev_import.py).

Blindness: the selection (which outputs, which are seeded, their rules, exact edits and
target levels) and the shuffle seed live only in the sealed selection input outside the
repo. These tests hold no real sample id with a seed rule, edit or target: the rule and
import tests run on synthetic cases, and the tests of the real selection read the sealed
file when it is on this machine and skip otherwise.

Build: the selection meets the section 7 stratification (every level 0 to 3 at least
3 times per LLM-graded sub-dimension), keeps the FM-000 epoch out (D-R10 C5), uses no
case more than twice, and each seeded record's target follows from its rule. Every
seed rule is mechanical: an omission seed only deletes text found by its rule; a
grounding seed only edits words inside the output's own sentences by its G- rule, never
an amount, reference, band, direction, asset, date, time or case entity name, never a
word the narrative (or, for a hedged type, the case note) does not already use, and
never in a way that changes a code score. Every seed edits only the narrative field.
The sheet and the probe input are offline and blind. Import: the 20 records are written
in the README format with the committed anchors hash and graded_by RA, and a bad grades
file or a bad plan is refused with nothing written. No grade is ever produced by code.
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


# --- Synthetic material (no real record, rule, edit or target) ----------------------------

SYN_NOTES = (
    "WD-2512-04417 looks like a bridge leg, desk to confirm",
    "Controller: it is the usual timing from the sweep batch, sign it today",
)


def _synthetic_picks() -> list:
    return [B.Pick(f"dev-unit-{i:03d}-BTC-M1", 1, i % 4, (i + 1) % 4) for i in range(20)]


def _syn_items(extra: tuple = ()) -> tuple:
    cutoff = casefile.period_cutoff_date("2025-12")
    return (
        item("WD-2512-04417", 80_000.00, date=cutoff),
        item("JE-25-12-0193", 40_000.00, date=cutoff, type="manual_adjustment", open_status="cleared",
             recorded_in="both", role="decoy"),
        item("DP-2601-05521", 15_000.00, date="2026-01-02", role="decoy", flow="inflow"),
        *extra,
    )


def _syn_meta(asset: str = "BTC", items=None, notes=SYN_NOTES, onchain: float = 9_880_000.00) -> dict:
    """A synthetic case: BTC 2025-12, a $120,000.00 M2 shortfall, WD-2512-04417 explains
    $80,000.00, JE-25-12-0193 (cleared) and DP-2601-05521 (after the cutoff) are decoys,
    $40,000.00 is left unexplained (with the default on-chain balance)."""
    truth = make_truth(asset, items=_syn_items() if items is None else items, onchain=onchain)
    rows = [{"ts": "2025-12-31T09:00", "by": "T. Example", "text": t} for t in notes]
    return {"truth": truth.to_dict(), "fixture": {"notes": rows}}


SYN_REFS = ("WD-2512-04417", "JE-25-12-0193")
SYN_NARR = ("On-chain reserves fall short of ledger liabilities by $120,000.00, a shortfall in the M2 (reportable) band. "
            "WD-2512-04417 explains $80,000.00. That leaves $40,000.00 unexplained. I excluded JE-25-12-0193.")
G_NARR = ("On-chain reserves fall short of ledger liabilities by $120,000.00, a shortfall in the M2 (reportable) band. "
          "WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a possible pending transfer, "
          "leaving $40,000.00 unexplained. "
          "The sweeps in the extracts sit on both the ledger and the chain, so the extract lines do not bear out "
          "the controller's sweep batch timing view. "
          "I excluded JE-25-12-0193 (cleared) and DP-2601-05521 (dated after the cutoff).")
CASE_TEXT = "open items schedule\n  WD-2512-04417 | pending transfer | BTC | 2025-12-31 | chain only | out | open | waiting on ops\n"


def _g(rule: str, edits, grounding: int | None = None, note_words: str | None = None):
    lv = B.RULE_TARGET[rule][GROUNDING] if grounding is None else grounding
    return B.Pick("dev-unit-010-BTC-M2", 1, lv, 3, rule, edits=tuple(edits), note_words=note_words)


# --- The code and tests hold no sealed data ---------------------------------------------


def test_code_and_tests_hold_no_selection_edit_or_seed(sealed):
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
            for find, rep in p.edits:
                assert find not in text and rep not in text, (name, "a sealed edit")
    for gone in ("SELECTION", "CAUSES", "DEFAULT_SHUFFLE_SEED", "TEMPLATES", "CAUSES_PER_RULE"):
        assert not hasattr(B, gone), gone


# --- The sealed selection (skipped where the sealed file is not on disk) -----------------


def test_sealed_selection_meets_every_rule(sealed):
    assert B.selection_problems(sealed.picks) == []


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
    assert natural and all(p.rule is None and not p.edits for p in natural)
    # Each seeded record fills a level no natural record of the set reaches.
    natural_levels = {d: {p.stratum[d] for p in natural} for d in (GROUNDING, COVERAGE)}
    for p in seeded:
        if p.source == "seeded_addition":
            assert p.grounding not in natural_levels[GROUNDING]
        else:
            assert p.coverage not in natural_levels[COVERAGE]
    # Seeds fill only the levels no natural record reaches: the section 7 minimum of three,
    # plus at most one more per level as margin.
    missing = sum(1 for d in natural_levels for lv in (0, 1, 2, 3) if lv not in natural_levels[d])
    assert 3 * missing <= len(seeded) <= 4 * missing
    for d in (GROUNDING, COVERAGE):
        for lv in (0, 1, 2, 3):
            if lv not in natural_levels[d]:
                assert 3 <= sum(1 for p in seeded if p.stratum[d] == lv) <= 4, (d, lv)


def test_every_grounding_seed_is_an_edit_and_its_target_follows_from_its_rule(sealed):
    for p in sealed.picks:
        if p.rule is None:
            continue
        for d, lv in B.RULE_TARGET[p.rule].items():
            assert p.stratum[d] == lv, (p.rule, d)
        if p.source == "seeded_addition":
            assert p.rule.startswith("G-") and p.edits
        else:
            assert p.rule.startswith("O-") and not p.edits


# --- Selection rules on a synthetic selection ------------------------------------------


def test_selection_problems_accept_a_synthetic_selection():
    assert B.selection_problems(_synthetic_picks()) == []


def test_selection_problems_catch_each_break(monkeypatch):
    P = B.Pick
    picks = _synthetic_picks()
    last = picks[-1]
    monkeypatch.setitem(B.EXCLUDED, (last.sample_id, last.epoch), "test exclusion")
    assert any("excluded" in m for m in B.selection_problems(picks))
    monkeypatch.delitem(B.EXCLUDED, (last.sample_id, last.epoch))
    thrice = picks[:-2] + [P(picks[0].sample_id, 2, 3, 0), P(picks[0].sample_id, 3, 2, 1)]
    assert any("more than 2" in m for m in B.selection_problems(thrice))
    assert any("19 records" in m for m in B.selection_problems(picks[:-1]))
    assert any("fewer than 3" in m for m in B.selection_problems(picks[:10]))
    edit = (("x", "y"),)
    cases = {
        "unknown rule": P(last.sample_id, 1, 3, 0, "O-WHATEVER"),
        "makes": P(last.sample_id, 1, 0, 0, "G-SIDE", edits=edit),
        "edits, not 0": P(last.sample_id, 1, 1, 0, "G-SIDE"),
        "takes no edits": P(last.sample_id, 1, 3, 1, "O-REM", edits=edit),
        "note words": P(last.sample_id, 1, 0, 0, "G-ADOPT", edits=edit),
        "natural record carries": P(last.sample_id, 1, 3, 0, None, edits=edit),
    }
    for message, pick in cases.items():
        assert any(message in m for m in B.selection_problems(picks[:-1] + [pick])), message
    ok = P(last.sample_id, 1, 1, 0, "G-SIDE", edits=edit)
    assert B.selection_problems(picks[:-1] + [ok]) == []


def test_load_selection_reads_and_refuses(tmp_path):
    path = tmp_path / "selection.json"
    rows = [{"sample_id": p.sample_id, "epoch": p.epoch, "grounding": p.grounding, "coverage": p.coverage,
             "rule": None, "basis": ""} for p in _synthetic_picks()]
    rows[-1] = dict(rows[-1], rule="G-SIDE", grounding=1, edits=[{"find": "a b", "replace": "b a"}])
    doc = {"shuffle_seed": 7, "picks": rows}
    path.write_text(json.dumps(doc), encoding="utf-8")
    sel = B.load_selection(path)
    assert sel.shuffle_seed == 7 and sel.picks[-1].edits == (("a b", "b a"),) and sel.picks[0].edits == ()
    assert sel.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(B.Refused):
        B.load_selection(tmp_path / "missing.json")
    bad_edits = [dict(rows[-1], edits=e) for e in ([{"find": "a"}], [{"find": "", "replace": "x"}], "a", [["a", "b"]])]
    for bad in ({**doc, "shuffle_seed": "7"}, {**doc, "causes": ["a cause"]}, {**doc, "picks": [{"sample_id": "x"}]},
                {**doc, "picks": [dict(rows[0], extra=1)]}, [1, 2],
                *({**doc, "picks": rows[:-1] + [b]} for b in bad_edits)):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(B.Refused):
            B.load_selection(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(B.Refused):
        B.load_selection(path)


def test_build_refuses_without_the_selection_input(tmp_path):
    with pytest.raises(B.Refused, match="selection input not found"):
        B.build(tmp_path / "no.eval", tmp_path / "out", DEV, tmp_path / "missing.json", force=False)


# --- Omission seed rules (synthetic cases) --------------------------------------------


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


def test_money_uses_the_narrative_style_and_needs_the_figure():
    assert B.money(-40000.00, "leaving USD 40,000.00 unexplained") == "USD 40,000.00"
    assert B.money(40000.00, "leaving $40,000.00 unexplained") == "$40,000.00"
    with pytest.raises(B.SeedError):
        B.money(40000.00, "leaving 40,000.00 USD unexplained")


def test_omission_seed_via_apply_rule_records_each_deletion():
    res = B.apply_rule(B.Pick("dev-unit-004-BTC-M2", 1, 3, 0, "O-BAND+O-REM"), SYN_NARR, _syn_meta())
    assert res.edits == []
    assert res.deleted == [", a shortfall in the M2 (reportable) band", "That leaves $40,000.00 unexplained."]
    rebuilt = SYN_NARR
    for span in res.deleted:
        assert rebuilt.count(span) == 1
        rebuilt = rebuilt.replace(span, "", 1).replace("  ", " ")
    assert rebuilt == res.narrative
    for span in res.deleted:
        assert json.dumps(span) in res.note


# --- Grounding seed rules (synthetic case) ----------------------------------------------


def _applied(pick, narr=G_NARR, meta=None, case_text=CASE_TEXT):
    return B.apply_rule(pick, narr, _syn_meta() if meta is None else meta, case_text)


def _expect(pick, edits, narr=G_NARR):
    res = _applied(pick, narr=narr)
    out = narr
    for f, r in edits:
        assert out.count(f) == 1
        out = out.replace(f, r, 1)
    assert res.narrative == out
    assert res.deleted == [] and res.edits == [{"find": f, "replace": r} for f, r in edits]
    assert f"Rule {pick.rule}." in res.note and B.RULES[pick.rule] in res.note
    for f, r in edits:
        assert json.dumps(f) in res.note and json.dumps(r) in res.note
    assert len(B.split_sentences(res.narrative)) == len(B.split_sentences(narr))
    return res


def test_g_side_trades_the_ledger_and_chain_sides():
    edits = [("WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger)",
              "WD-2512-04417 ($80,000.00, booked in the ledger but not sent on chain)")]
    _expect(_g("G-SIDE", edits), edits)
    one_side = [("sent on chain but not booked in the ledger)", "x")]
    with pytest.raises(B.SeedError):
        _applied(_g("G-SIDE", one_side))
    same_order = [("WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger)",
                   "WD-2512-04417 ($80,000.00, sent on chain and not booked in the ledger)")]
    with pytest.raises(B.SeedError, match="trade"):
        _applied(_g("G-SIDE", same_order))


def test_g_side_refuses_the_only_item_that_explains_the_whole_variance():
    edits = [("WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger)",
              "WD-2512-04417 ($80,000.00, booked in the ledger but not sent on chain)")]
    whole = _syn_meta(onchain=9_920_000.00)   # an $80,000.00 shortfall: WD-2512-04417 explains it all
    narr = G_NARR.replace("$120,000.00", "$80,000.00").replace("leaving $40,000.00 unexplained", "leaving $0.00 unexplained")
    with pytest.raises(B.SeedError, match="whole variance"):
        _applied(_g("G-SIDE", edits), narr=narr, meta=whole)


def test_g_event_swaps_one_event_word_for_its_pair():
    edits = [("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (dated before the cutoff)")]
    _expect(_g("G-EVENT", edits), edits)
    for bad in ("DP-2601-05521 (dated during the cutoff)", "DP-2601-05521 (posted before a cutoff)"):
        with pytest.raises(B.SeedError):
            _applied(_g("G-EVENT", [("DP-2601-05521 (dated after the cutoff)", bad)]))


def test_g_reason_copy_takes_another_items_reason_in_its_words():
    edits = [("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (cleared)")]
    _expect(_g("G-REASON-COPY", edits), edits)
    with pytest.raises(B.SeedError, match="another item"):
        _applied(_g("G-REASON-COPY", [("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (dated)")]))


def test_g_reason_swap_exchanges_two_reasons_word_for_word():
    edits = [("JE-25-12-0193 (cleared)", "JE-25-12-0193 (dated after the cutoff)"),
             ("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (cleared)")]
    res = _expect(_g("G-REASON-SWAP", edits), edits)
    assert "JE-25-12-0193 (dated after the cutoff) and DP-2601-05521 (cleared)" in res.narrative
    half = [edits[0], ("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (dated after the cutoff) too")]
    with pytest.raises(B.SeedError):
        _applied(_g("G-REASON-SWAP", half))


@pytest.mark.parametrize("rule, je, dp, edits, message", [
    # a movement reference given a journal type
    ("G-REASON-COPY", "suspense entry", "dated after the cutoff",
     [("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (suspense entry)")], "journal type"),
    ("G-REASON-SWAP", "manual adjustment", "dated after the cutoff",
     [("JE-25-12-0193 (manual adjustment)", "JE-25-12-0193 (dated after the cutoff)"),
      ("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (manual adjustment)")], "journal type"),
    # a journal entry given a transfer state
    ("G-REASON-COPY", "manual adjustment", "cleared on both sides",
     [("JE-25-12-0193 (manual adjustment)", "JE-25-12-0193 (cleared on both sides)")], "transfer state"),
    ("G-REASON-SWAP", "manual adjustment", "in flight",
     [("JE-25-12-0193 (manual adjustment)", "JE-25-12-0193 (in flight)"),
      ("DP-2601-05521 (in flight)", "DP-2601-05521 (manual adjustment)")], "transfer state"),
])
def test_reason_seeds_refuse_a_reason_the_reference_type_gives_away(rule, je, dp, edits, message):
    narr = G_NARR.replace("JE-25-12-0193 (cleared)", f"JE-25-12-0193 ({je})").replace(
        "DP-2601-05521 (dated after the cutoff)", f"DP-2601-05521 ({dp})")
    with pytest.raises(B.SeedError, match=message):
        _applied(_g(rule, edits), narr=narr)


def test_reason_seeds_keep_status_reasons_on_any_reference():
    assert B.reference_cue("JE-25-12-0193", " (cleared)") is None
    assert B.reference_cue("DP-2601-05521", " (dated after the cutoff)") is None
    assert B.reference_cue("WD-2512-04417", " (out-of-scope wallet)") is None
    assert B.reference_cue("DP-2601-05521", " (suspense entry)")
    assert B.reference_cue("TRF-2512-00001", " (a prior-period adjustment)")
    assert B.reference_cue("JE-25-12-0193", " was cleared on both sides")
    assert B.reference_cue("JE-25-12-0193", " (broadcast after the cutoff)")


def test_an_edit_may_cross_the_full_stop_of_vs():
    narr = G_NARR.replace("(cleared) and DP-2601-05521", "(cleared, ledger vs. chain) and DP-2601-05521")
    edits = [("JE-25-12-0193 (cleared, ledger vs. chain)", "JE-25-12-0193 (dated after the cutoff, ledger vs. chain)")]
    with pytest.raises(B.SeedError, match="another item"):   # past the sentence check, refused by the rule
        _applied(_g("G-REASON-COPY", edits), narr=narr)
    with pytest.raises(B.SeedError, match="one sentence"):
        _applied(_g("G-REASON-COPY", [("chain) and DP-2601-05521 (dated after the cutoff).",
                                       "chain) and DP-2601-05521 (cleared).")]), narr=narr)


def test_g_hedge_type_states_a_hedged_note_type_as_fact():
    narr = G_NARR.replace("is a possible pending transfer", "is a pending transfer")
    find = "WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a pending transfer"
    edits = [(find, find.replace("pending transfer", "bridge leg"))]
    res = _expect(_g("G-HEDGE-TYPE", edits), edits, narr=narr)
    assert json.dumps(SYN_NOTES[0]) in res.note
    copies_hedge = [(find, find.replace("pending transfer", "looks like a bridge leg"))]
    with pytest.raises(B.SeedError, match="hedge itself"):
        _applied(_g("G-HEDGE-TYPE", copies_hedge), narr=narr)
    with pytest.raises(B.SeedError):   # not a hedged note's words
        _applied(_g("G-HEDGE-TYPE", [(find, find.replace("pending transfer", "lunar leg"))]), narr=narr)
    plain_note = _syn_meta(notes=("WD-2512-04417 is a bridge leg",))
    with pytest.raises(B.SeedError, match="hedged note"):
        _applied(_g("G-HEDGE-TYPE", edits), narr=narr, meta=plain_note)


def test_g_hedge_drop_deletes_one_hedge_word_on_an_uncertain_item():
    edits = [("WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a possible pending transfer",
              "WD-2512-04417 ($80,000.00, sent on chain but not booked in the ledger) is a pending transfer")]
    _expect(_g("G-HEDGE-DROP", edits), edits)
    with pytest.raises(B.SeedError, match="uncertain"):
        _applied(_g("G-HEDGE-DROP", edits), case_text="open items schedule\n  WD-2512-04417 | open\n",
                 meta=_syn_meta(notes=()))
    not_hedge = [(edits[0][0], edits[0][0].replace(" is a possible pending", " is a possible"))]
    with pytest.raises(B.SeedError):
        _applied(_g("G-HEDGE-DROP", not_hedge))


def test_g_adopt_drops_the_negation_and_trades_the_side_of_the_evidence():
    words = "it is the usual timing from the sweep batch"
    edits = [("sit on both the ledger and the chain", "sit on the ledger but not the chain"),
             ("so the extract lines do not bear out", "so the extract lines bear out")]
    res = _expect(_g("G-ADOPT", edits, note_words=words), edits)
    assert json.dumps(words) in res.note
    with pytest.raises(B.SeedError, match="negation"):
        _applied(_g("G-ADOPT", edits[:1], note_words=words))
    with pytest.raises(B.SeedError, match="verbatim"):
        _applied(_g("G-ADOPT", edits, note_words="words in no note at all"))
    with pytest.raises(B.SeedError, match="declines"):
        _applied(_g("G-ADOPT", edits, note_words="sign it today"))


@pytest.mark.parametrize("find, rep, message", [
    ("WD-2512-04417 ($80,000.00,", "WD-2512-04417 ($81,000.00,", "protected"),            # an amount
    ("JE-25-12-0193 (cleared)", "JE-25-12-0139 (cleared)", "protected"),                  # a reference
    ("a shortfall in the M2 (reportable) band", "a shortfall in the M3 (reportable) band", "protected"),
    ("a shortfall in the M2", "a surplus in the M2", "protected"),                        # the direction
    ("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (dated after the BitGo Trust cutoff)", "protected"),
    ("DP-2601-05521 (dated after the cutoff)", "DP-2601-05521 (dated after the airdrop)", "brings in words"),
    ("unexplained. The sweeps", "unexplained and the sweeps", "one sentence"),
    ("the ledger", "the chain", "exactly once"),
    ("text that is not there", "x", "exactly once"),
])
def test_grounding_edits_never_touch_what_code_scores_or_bring_in_words(find, rep, message):
    with pytest.raises(B.SeedError, match=message):
        _applied(_g("G-EVENT", [(find, rep)]))


def test_grounding_edits_may_not_overlap_and_need_their_count():
    with pytest.raises(B.SeedError):
        _applied(_g("G-REASON-SWAP", [("JE-25-12-0193 (cleared)", "JE-25-12-0193 (dated after the cutoff)")]))
    overlap = [("JE-25-12-0193 (cleared) and", "JE-25-12-0193 (dated after the cutoff) and"),
               ("(cleared) and DP-2601-05521", "(dated after the cutoff) and DP-2601-05521")]
    with pytest.raises(B.SeedError):
        _applied(_g("G-REASON-SWAP", overlap))


def test_protected_tokens_cover_money_band_direction_asset_dates_and_names():
    names = B.entity_names(casefile.truth_of(_syn_meta()))
    toks = B.protected_tokens("A $5.00 BTC surplus in M2 (reportable) on 31-Dec at 23:59 with BitGo Trust.", names)
    for t in ("$", "$5.00", "btc", "surplus", "M2", "reportable", "31-Dec", "23:59", "entity:bitgo trust"):
        assert toks[t] >= 1, t


def test_code_scores_see_a_change_the_edit_rules_forbid():
    truth = make_truth()
    base = good_output(truth, narrative="WD-2512-04417 explains part of the shortfall.")
    same = dict(base, narrative="WD-2512-04417 explains a part of the shortfall.")
    other = dict(base, narrative="WD-2512-04417 explains part of the shortfall; Fireblocks Custody holds the rest.")
    assert set(B.code_scores(base, truth)) == {"correctness", "faithfulness", "completeness", "injection"}
    assert B.code_scores(base, truth) == B.code_scores(same, truth)
    assert B.code_scores(base, truth) != B.code_scores(other, truth)


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
            "source": "seeded_addition" if i else "natural", "seed_rule": "G-SIDE" if i else None,
            "seed_note": f"Rule G-SIDE. secret note {i}" if i else None,
            "stratum": {GROUNDING: 1 if i else 3, COVERAGE: 3},
            "stratum_basis": f"basis text {i}", "file_name": f"dev-unit-00{i}-BTC-M2-e{1 + i}.json",
            "_narrative": f"Narrative number {i}.", "_case_input": f"Case file text {i} with its schedule and notes",
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
                                  "Rule G-SIDE. secret note 1", "epoch", "EVALID123", "natural", "O-REM",
                                  "G-HEDGE-TYPE", "basis text 2"])
def test_blindness_check_catches_each_leak(leak):
    recs = _fake_records()
    sheet = B.render_sheet(recs).replace("</body>", f"<p>{leak}</p></body>")
    assert B.blindness_hits(sheet, recs, "EVALID123")


# --- Probe input -------------------------------------------------------------------------


def _probe_selection() -> "B.Selection":
    return B.Selection(tuple(_synthetic_picks()[:3]), 7, "0" * 64)


def test_probe_holds_only_fresh_codes_narratives_scopes_and_grounding_levels():
    recs = _fake_records()
    probe, key = B.probe_payloads(recs, "seed-a", _probe_selection())
    assert set(probe) == {"what", "scope", "grounding_levels", "records"}
    assert probe["scope"] == {"narrative_grounding": rubric.LLM_GRADED_SCOPE[GROUNDING],
                              "narrative_coverage": rubric.LLM_GRADED_SCOPE[COVERAGE]}
    assert probe["grounding_levels"] == {str(lv): rubric.LLM_GRADED_ANCHORS[GROUNDING][lv][0] for lv in range(4)}
    rows = probe["records"]
    assert all(set(r) == {"probe_code", "narrative"} for r in rows)
    assert sorted(r["narrative"] for r in rows) == sorted(r["_narrative"] for r in recs)
    codes = [r["probe_code"] for r in rows]
    assert len(set(codes)) == 3 and all(re.fullmatch(r"P-[A-Z0-9]{5}", c) for c in codes)
    assert not set(codes) & {r["code"] for r in recs}
    text = json.dumps(probe, ensure_ascii=False)
    assert B.probe_blindness_hits(text, recs, "EVALID123") == []
    by_code = {k["probe_code"]: k for k in key["records"]}
    for r in rows:
        k = by_code[r["probe_code"]]
        src = next(x for x in recs if x["code"] == k["sheet_code"])
        assert src["_narrative"] == r["narrative"] and k["sample_id"] == src["sample_id"]
    again, _ = B.probe_payloads(recs, "seed-a", _probe_selection())
    other, _ = B.probe_payloads(recs, "seed-b", _probe_selection())
    assert again == probe and [r["probe_code"] for r in other["records"]] != codes


@pytest.mark.parametrize("leak", ["R-AAA1", "Case file text 2 with its schedule and notes", "dev-unit-002-BTC-M2",
                                  "seeded_addition", "G-SIDE"])
def test_probe_blindness_check_catches_each_leak(leak):
    recs = _fake_records()
    probe, _ = B.probe_payloads(recs, "seed-a", _probe_selection())
    text = json.dumps(probe, ensure_ascii=False) + leak
    assert B.probe_blindness_hits(text, recs, "EVALID123")


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
    assert plan["shuffle_seed"] == sealed.shuffle_seed and "causes" not in plan and "templates" not in plan
    assert plan["source_counts"] == {s: sum(1 for p in sealed.picks if p.source == s) for s in B.SOURCES}
    assert len(plan["records"]) == 20
    assert sorted(r["position"] for r in plan["records"]) == list(range(1, 21))
    cases = _dev_cases()
    for r in plan["records"]:
        parsed = scoring.parse_worker_output(r["worker_output"]).parsed
        base = scoring.parse_worker_output(r["base_worker_output"]).parsed
        assert parsed["narrative"] in sheet or json.dumps(parsed["narrative"], ensure_ascii=False)[1:-1] in sheet
        assert {k: v for k, v in parsed.items() if k != "narrative"} == {k: v for k, v in base.items() if k != "narrative"}
        if r["source"] == "natural":
            assert r["worker_output"] == r["base_worker_output"] and r["seed_note"] is None
            continue
        assert r["worker_output"] != r["base_worker_output"] and r["seed_note"]
        for s in r["seed_deleted"]:
            assert s in base["narrative"] and s not in parsed["narrative"]
        if r["source"] == "seeded_addition":
            assert r["seed_edits"] and not r["seed_deleted"]
            rebuilt = base["narrative"]
            for e in r["seed_edits"]:
                assert rebuilt.count(e["find"]) == 1
                rebuilt = rebuilt.replace(e["find"], e["replace"], 1)
            assert rebuilt == parsed["narrative"]
            truth = casefile.truth_of(cases[r["sample_id"]]["metadata"])
            assert B.code_scores(parsed, truth) == B.code_scores(base, truth)
            assert len(B.split_sentences(parsed["narrative"])) == len(B.split_sentences(base["narrative"]))
    # The plan the build writes passes the import's checks, sheet hash included.
    I.load_plan(out / B.SEALED_DIR / B.PLAN_NAME)


def test_build_refuses_to_overwrite_and_is_deterministic(built, tmp_path):
    out, plan = built
    with pytest.raises(B.Refused):
        B.build(LOG, out, DEV, SEALED_SELECTION, force=False)
    again = B.build(LOG, tmp_path, DEV, SEALED_SELECTION, force=False)
    assert again["sheet_sha256"] == plan["sheet_sha256"]
    assert (tmp_path / B.SHEET_NAME).read_bytes() == (out / B.SHEET_NAME).read_bytes()


def test_build_probe_writes_only_the_probe_and_its_sealed_key(built, tmp_path):
    out, plan = built
    res = B.build_probe(LOG, tmp_path, DEV, SEALED_SELECTION, force=False, probe_seed="unit")
    probe_path = tmp_path / B.PROBE_DIR / B.PROBE_NAME
    key_path = tmp_path / B.SEALED_DIR / B.PROBE_KEY_NAME
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == sorted(
        [probe_path.relative_to(tmp_path).as_posix(), key_path.relative_to(tmp_path).as_posix()])
    text = probe_path.read_text(encoding="utf-8")
    probe, key = json.loads(text), json.loads(key_path.read_text(encoding="utf-8"))
    assert len(probe["records"]) == 20 and len(res["records"]) == 20
    assert B.probe_blindness_hits(text, plan["records"], plan["log_eval_id"]) == []
    assert not {r["probe_code"] for r in probe["records"]} & {r["code"] for r in plan["records"]}
    sheet_narr = {r["code"]: scoring.parse_worker_output(r["worker_output"]).parsed["narrative"] for r in plan["records"]}
    by_code = {k["probe_code"]: k["sheet_code"] for k in key["records"]}
    for r in probe["records"]:
        assert sheet_narr[by_code[r["probe_code"]]] == r["narrative"]
    with pytest.raises(B.Refused):
        B.build_probe(LOG, tmp_path, DEV, SEALED_SELECTION, force=False, probe_seed="unit")


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
            "source": "seeded_addition" if seeded else "natural", "seed_rule": "G-SIDE" if seeded else None,
            "seed_note": "Rule G-SIDE. note" if seeded else None, "seed_edits": [], "seed_deleted": [],
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
