"""scripts/review_freeze.py: record validation, freeze, second look and blinding check.

Every packet here is synthetic and built in tmp_path with review_packet's own data model
and write_packet, so the files have the real packet format. Every verdict is filled on
that scratch copy only, with reviewer SYNTHETIC-TEST and every note starting
"synthetic test row, not a review". No test runs git (git_commit_info is monkeypatched),
calls a model, or reads a log: review_packet.load_runs and build_packet are monkeypatched
to return the synthetic packet, so the byte-identical rebuild is real write_packet output.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from icfr_agent_evals import regression, settlement
from icfr_agent_evals import review_schema as rs
from icfr_agent_evals.scoring import SweepHit

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_freeze.py"


def _load_freeze():
    spec = importlib.util.spec_from_file_location("review_freeze", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["review_freeze"] = mod
    spec.loader.exec_module(mod)
    return mod


rf = _load_freeze()
rp = rf.load_review_packet()

REVIEWER = "SYNTHETIC-TEST"
NOTE = "synthetic test row, not a review"
DAY = "2026-11-09"
SEED = 7
LABELS = ("alpha", "beta", "gamma", "delta")
MODELS = ("mockllm/model", "anthropic/claude-sonnet-5-5", "openai/gpt-5", "mockllm/other")
SWEEP = SweepHit("BTC custody wallet", "custodian_vocabulary").reason
AMOUNT = "amount_role_unresolved: $5,000"
AMOUNT_U = "amount_role_unresolved: $7,000"
NONLATIN = "non_latin_characters: \u0436"
FORMAT_BREAK = "format break: not one JSON object"
TEXT_A = "The variance is $5,000 on chain. The BTC custody wallet holds the reserve. The band is M2."
TEXT_U = "The variance is $7,000. Nothing else is open."
TEXT_B = "Funds sit in a wallet. The band is M1."
TEXT_C = "not json {"
TEXT_D = "The variance is $5,000. The band is M3."
SITTING = "2026-11-09T09:00:00-05:00/2026-11-09T10:30:00-05:00"


def key(reason):
    return regression.route_key(reason)


def _uid(kind, name):
    return rs.unit_id(SEED, kind, rs.group_key([name]))


def _finding(name, case, reason, cls, sentence, narratives, highlight=()):
    k = key(reason)
    return rp.Unit(unit_id=_uid("finding", name), row_kind="finding", case_id=case, route_key=k, route_class=cls,
                   reasons=(reason,), routed_sentences=(sentence,), highlight=highlight,
                   narrative_unit_ids=tuple(narratives), allowed_verdicts=settlement.allowed_verdicts(k, [reason]),
                   correct_means=tuple(r.correct_means for r in settlement.rows_for(k, [reason])))


def _narrative(name, case, text, parsed_ok=True):
    return rp.Unit(unit_id=_uid("narrative", name), row_kind="narrative", case_id=case, narrative=text,
                   sentences=rs.narrative_sentences(text), parsed_ok=parsed_ok,
                   structured={"variance_amount_usd": 5000} if parsed_ok else {})


def _audit(name, case, code, reasons, narrative):
    return rp.Unit(unit_id=_uid("fm_audit", name), row_kind="fm_audit", case_id=case, fm_code=code,
                   fm_reasons=tuple(reasons), narrative_unit_ids=(narrative,))


def _member(unit, kind, run, case, epoch, reason="", cls="", codes=()):
    return rp.Member(unit_id=unit, member_kind=kind, run_id=run, sample_id=case, epoch=epoch,
                     route_key=key(reason) if reason else "", route_class=cls,
                     scorers=("correctness_anchor",) if reason else (), reasons=(reason,) if reason else (),
                     fm_codes=tuple(codes))


def build(tmp_path, *, n_runs=2, fm_route="none", name="packet"):
    """A synthetic packet. Every run: case-a epoch 1 (narrative A with a sweep finding and an
    amount finding, shared by all runs) and its own case-u<i> narrative with one amount
    finding. Run R1 also: case-a epoch 2 (same narrative and findings), case-b epoch 1 (only
    an unsettleable route), case-c epoch 3 (format break, FM-000 audit). fm_route "findings"
    or "skip" adds case-d epoch 1 on R1: FM-001 with an amount route, as a finding or skipped."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs, units, members = [], [], []
    for i in range(n_runs):
        f = tmp_path / f"{LABELS[i]}-{i}.eval"
        f.write_bytes(f"synthetic log {i}".encode())
        runs.append(rp.RunInput(run_id=f"R{i + 1}", label=LABELS[i], path=str(f), sha256=rs.sha256_file(f),
                                eval_id=f"EVAL{i + 1}ID", model=MODELS[i], revision_commit="a783322"))
    n_a = _narrative("A", "case-a", TEXT_A)
    f_sweep = _finding("sweep", "case-a", SWEEP, "narrative_sweep", rs.narrative_sentences(TEXT_A)[1],
                       [n_a.unit_id], ("BTC custody wallet",))
    f_amt = _finding("amt", "case-a", AMOUNT, "code_review", rs.narrative_sentences(TEXT_A)[0], [n_a.unit_id])
    units += [n_a, f_sweep, f_amt]
    ids = {"N_A": n_a.unit_id, "F_SWEEP": f_sweep.unit_id, "F_AMT": f_amt.unit_id}
    for i, run in enumerate(runs):
        epochs = [1, 2] if i == 0 else [1]
        for ep in epochs:
            members += [_member(n_a.unit_id, "narrative", run.run_id, "case-a", ep),
                        _member(f_sweep.unit_id, "finding", run.run_id, "case-a", ep, SWEEP, "narrative_sweep"),
                        _member(f_amt.unit_id, "finding", run.run_id, "case-a", ep, AMOUNT, "code_review")]
        case = f"case-u{i + 1}"
        n_u = _narrative(f"U{i + 1}", case, TEXT_U + f" Run {i + 1}.")
        f_u = _finding(f"U{i + 1}", case, AMOUNT_U, "code_review", rs.narrative_sentences(TEXT_U)[0], [n_u.unit_id])
        units += [n_u, f_u]
        ids[f"N_U{i + 1}"], ids[f"F_U{i + 1}"] = n_u.unit_id, f_u.unit_id
        members += [_member(n_u.unit_id, "narrative", run.run_id, case, 1),
                    _member(f_u.unit_id, "finding", run.run_id, case, 1, AMOUNT_U, "code_review")]
    r1 = runs[0].run_id
    n_b = _narrative("B", "case-b", TEXT_B)
    n_c = _narrative("C", "case-c", TEXT_C, parsed_ok=False)
    a_c = _audit("C", "case-c", "FM-000", [FORMAT_BREAK], n_c.unit_id)
    units += [n_b, n_c, a_c]
    ids.update({"N_B": n_b.unit_id, "N_C": n_c.unit_id, "A_C": a_c.unit_id})
    members += [_member(n_b.unit_id, "narrative", r1, "case-b", 1),
                _member("", "unsettleable", r1, "case-b", 1, NONLATIN, "code_review"),
                _member(n_c.unit_id, "narrative", r1, "case-c", 3, codes=("FM-000",)),
                _member(a_c.unit_id, "fm_audit", r1, "case-c", 3, codes=("FM-000",))]
    if fm_route != "none":
        n_d = _narrative("D", "case-d", TEXT_D)
        a_d = _audit("D", "case-d", "FM-001", ["amount: $5,000"], n_d.unit_id)
        units += [n_d, a_d]
        ids.update({"N_D": n_d.unit_id, "A_D": a_d.unit_id})
        members += [_member(n_d.unit_id, "narrative", r1, "case-d", 1, codes=("FM-001",)),
                    _member(a_d.unit_id, "fm_audit", r1, "case-d", 1, codes=("FM-001",))]
        if fm_route == "findings":
            f_d = _finding("D", "case-d", AMOUNT, "code_review", rs.narrative_sentences(TEXT_D)[0], [n_d.unit_id])
            units.append(f_d)
            ids["F_D"] = f_d.unit_id
            members.append(_member(f_d.unit_id, "finding", r1, "case-d", 1, AMOUNT, "code_review", ("FM-001",)))
        else:
            members.append(_member("", "fm_epoch_route_skipped", r1, "case-d", 1, AMOUNT, "code_review",
                                   ("FM-001",)))
    cases = tuple(sorted({u.case_id for u in units}))
    packet = rp.Packet(run_id="dev-test-run", seed=SEED, scope="routed",
                       fm_epoch_routes="skip" if fm_route == "skip" else "findings", runs=tuple(runs),
                       case_order=cases, units=tuple(units), members=rp._sort_members(members),
                       case_facts={c: () for c in cases}, case_inputs={c: f"Case file text for {c}." for c in cases})
    out = tmp_path / name
    rp.write_packet(packet, str(out))
    return SimpleNamespace(dir=out, packet=packet, ids=ids, runs=runs, tmp=tmp_path)


def mark_template(env, run_id):
    """Mark one run of a written packet as a template baseline (sealed/runs.json), re-hashed in the meta."""
    runs_path = env.dir / rs.PACKET_FILES["runs"]
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    runs[run_id]["template_baseline"] = True
    rs.write_lf(runs_path, json.dumps(runs, indent=2) + "\n")
    meta_path = env.dir / rs.PACKET_FILES["meta"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"][rs.PACKET_FILES["runs"]] = rs.sha256_file(runs_path)
    rs.write_lf(meta_path, json.dumps(meta, indent=2) + "\n")


def fill(env, *, verdicts=None, edit=None, path=None):
    """Fill the scratch copy's blank file: findings correct, narratives none_wrong, audits
    real_error, unless `verdicts` (unit id -> verdict or (verdict, sentence, reason)) says otherwise."""
    rows = rs.read_rb7_csv(env.dir / rs.PACKET_FILES["blank"])
    defaults = {"finding": "correct", "narrative": "none_wrong", "fm_audit": "real_error"}
    for row in rows:
        v = (verdicts or {}).get(row["unit_id"], defaults[row["row_kind"]])
        if isinstance(v, tuple):
            row["verdict"], row["sentence"], row["reason"] = v
        else:
            row["verdict"] = v
        row.update({"note": NOTE, "reviewer": REVIEWER, "date": DAY})
    if edit:
        rows = edit(rows) or rows
    out = Path(path) if path else env.tmp / "filled.csv"
    rs.write_rb7_csv(out, rows)
    return out


FREEZE_NOW = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)  # after SITTING ends, on or after DAY


def patch_rebuild(monkeypatch, env):
    monkeypatch.setattr(rp, "load_runs", lambda specs, seed: env.packet.runs)
    monkeypatch.setattr(rp, "build_packet", lambda runs, **kw: env.packet)
    monkeypatch.setattr(rf, "_utcnow", lambda: FREEZE_NOW)


def run_args(env):
    return [a for r in env.runs for a in ("--run", f"{r.label}={r.path}")]


def mdir(env):
    return env.tmp / "repo" / "docs" / "evidence" / "reviews"


def freeze_argv(env, filled, *extra):
    return ["freeze", "--packet-dir", str(env.dir), "--filled", str(filled), *run_args(env), "--reviewer", REVIEWER,
            "--date", DAY, "--run-commit", "a783322", "--sitting", SITTING, "--manifest-dir", str(mdir(env)),
            *extra]


def do_freeze(monkeypatch, capsys, env, filled=None, *extra):
    patch_rebuild(monkeypatch, env)
    filled = filled or fill(env)
    code = rf.main(freeze_argv(env, filled, *extra))
    out, err = capsys.readouterr()
    return code, out, err


def gaps_of(env, filled, **kw):
    return rf.validate_record(rf.load_record(str(env.dir), str(filled)), **kw)


# --- Validation (section 8.2 items 1 to 8) -------------------------------------------------

def test_complete_record_validates(tmp_path, capsys):
    env = build(tmp_path)
    filled = fill(env)
    assert gaps_of(env, filled) == []
    assert gaps_of(env, filled, reviewer=REVIEWER, freeze_date=DAY) == []
    assert rf.main(["validate", "--packet-dir", str(env.dir), "--filled", str(filled)]) == 0
    assert "record complete" in capsys.readouterr().out


def _set(uid, **cells):
    def edit(rows):
        for r in rows:
            if r["unit_id"] == uid:
                r.update(cells)
    return edit


def _drop(uid):
    return lambda rows: [r for r in rows if r["unit_id"] != uid]


REFUSALS = {
    "blank_verdict": (lambda ids: _set(ids["F_SWEEP"], verdict=""), "item 3", "verdict blank"),
    "citation_omission_on_code_review": (lambda ids: _set(ids["F_AMT"], verdict="citation_omission"), "item 3",
                                         "not allowed"),
    "missing_narrative_row": (lambda ids: _drop(ids["N_A"]), "item 2", "has no row"),
    "missing_audit_row": (lambda ids: _drop(ids["A_C"]), "item 2", "has no row"),
    "missing_note": (lambda ids: _set(ids["F_AMT"], verdict="wrong", note=""), "item 3", "note required"),
    "sentence_out_of_range": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="9", reason="amount"),
                              "item 4", "outside 1..3"),
    "zero_beside_other_type": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="0", reason="amount"),
                               "item 4", "sentence 0"),
    "error_type_count": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="1;2", reason="amount"),
                         "item 4", "error type(s)"),
    "not_reviewed_narrative": (lambda ids: _set(ids["N_A"], verdict="not_reviewed"), "item 4", "not_reviewed"),
    "finding_sentence_differs": (lambda ids: _set(ids["F_SWEEP"], sentence="The BTC custody wallet holds it."),
                                 "item 3", "sentence differs"),
    "missing_reviewer": (lambda ids: _set(ids["N_B"], reviewer=""), "item 6", "reviewer blank"),
    "masked_column_filled": (lambda ids: _set(ids["F_AMT"], run_label="alpha"), "item 6", "masked"),
    "audit_code_differs": (lambda ids: _set(ids["A_C"], route_key="FM-001"), "item 5", "FM code"),
    "audit_note_missing": (lambda ids: _set(ids["A_C"], note=""), "item 5", "note required"),
    "audit_not_reviewed": (lambda ids: _set(ids["A_C"], verdict="not_reviewed"), "item 5", "not real_error"),
    "repeated_row": (lambda ids: (lambda rows: rows + [dict(rows[0])]), "item 2", "more than one row"),
    "unknown_unit": (lambda ids: _set(ids["F_AMT"], unit_id="F-000000000000"), "item 2", "not in the packet"),
    "row_kind_differs": (lambda ids: _set(ids["N_B"], row_kind="finding"), "item 2", "row_kind"),
    "repeated_number_and_type": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="1;1", reason="band;band"),
                                 "item 4", "repeated"),
    "unknown_error_type": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="2", reason="tone"),
                           "item 4", "ERROR_TYPES"),
    "narrative_wrong_without_note": (lambda ids: _set(ids["N_A"], verdict="wrong", sentence="2",
                                                      reason="wallet_holder", note=""), "item 4", "note required"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_record_refusals(name, tmp_path, capsys):
    env = build(tmp_path)
    make_edit, item, words = REFUSALS[name]
    filled = fill(env, edit=make_edit(env.ids))
    gaps = gaps_of(env, filled)
    assert any(g.startswith(item) and words in g for g in gaps), gaps
    assert rf.main(["validate", "--packet-dir", str(env.dir), "--filled", str(filled)]) == 3
    assert "REFUSED" in capsys.readouterr().err


def test_zero_with_missing_required_element_is_allowed(tmp_path):
    env = build(tmp_path)
    filled = fill(env, verdicts={env.ids["N_A"]: ("wrong", "0;2", "missing_required_element;wallet_holder")})
    assert gaps_of(env, filled) == []


def test_crlf_bom_and_header_refused(tmp_path):
    env = build(tmp_path)
    filled = fill(env)
    raw = filled.read_bytes()
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert any(g.startswith("item 1") and "carriage return" in g for g in gaps_of(env, crlf))
    bom = tmp_path / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbf" + raw)
    assert any(g.startswith("item 1") and "byte order mark" in g for g in gaps_of(env, bom))
    head, rest = raw.split(b"\n", 1)
    cols = head.split(b",")
    reordered = tmp_path / "reordered.csv"
    reordered.write_bytes(b",".join([cols[1], cols[0], *cols[2:]]) + b"\n" + rest)
    assert any(g.startswith("item 1") and "header" in g for g in gaps_of(env, reordered))


def test_row_dated_after_freeze_date_refused(tmp_path):
    env = build(tmp_path)
    filled = fill(env, edit=_set(env.ids["N_B"], date="2026-11-10"))
    assert gaps_of(env, filled) == []
    gaps = gaps_of(env, filled, freeze_date=DAY)
    assert any(g.startswith("item 6") and "after the freeze date" in g for g in gaps)
    assert any("is not 'someone'" in g for g in gaps_of(env, fill(env), reviewer="someone"))


def test_tampered_members_refused(tmp_path):
    env = build(tmp_path)
    filled = fill(env)
    path = env.dir / rs.PACKET_FILES["members"]
    rows = rs.read_csv(path, rs.MEMBER_COLUMNS)
    rs.write_csv(path, rs.MEMBER_COLUMNS, [r for r in rows if r["member_kind"] != "fm_audit"])
    gaps = gaps_of(env, filled)
    assert any(g.startswith("item 7") and "no audit unit for FM-000" in g for g in gaps)
    assert any(g.startswith("item 8") and "members.csv" in g for g in gaps)


def test_every_gap_is_listed(tmp_path):
    env = build(tmp_path)
    edit = lambda rows: [dict(r, verdict="") if r["row_kind"] == "finding" else r for r in rows]  # noqa: E731
    gaps = gaps_of(env, fill(env, edit=edit))
    findings = [u for u in env.packet.units if u.row_kind == "finding"]
    assert sum("verdict blank" in g for g in gaps) == len(findings) > 2


def test_packet_hash_check(tmp_path):
    env = build(tmp_path)
    rec = rf.load_record(str(env.dir), str(fill(env)))
    assert rf.check_packet_hashes(rec) == []
    (env.dir / rs.PACKET_FILES["cases"]).write_bytes(b"case_id,case_input,case_facts_json\n")
    assert any("cases.csv" in g for g in rf.check_packet_hashes(rec))


# --- Freeze ---------------------------------------------------------------------------------

def test_freeze_writes_record_and_manifest(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    code, out, err = do_freeze(monkeypatch, capsys, env)
    assert code == 0, err
    assert out.strip().splitlines()[-1] == f"Reviewed-by: {REVIEWER}, {DAY}"
    for rel in rs.RECORD_FILES.values():
        assert (env.dir / rel).is_file() and b"\r" not in (env.dir / rel).read_bytes()
    verdicts = rs.read_rb7_csv(env.dir / rs.RECORD_FILES["verdicts"])
    audits = rs.read_rb7_csv(env.dir / rs.RECORD_FILES["fm_audit"])
    assert {r["row_kind"] for r in verdicts} == {"finding", "narrative"}
    assert [r["row_kind"] for r in audits] == ["fm_audit"]
    man_path = mdir(env) / "dev-test-run-manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert tuple(man) == rf.MANIFEST_KEYS
    assert man["statement"] == rs.STATEMENT
    assert man["counts"]["findings_by_route_class"] == {"code_review": 5, "narrative_sweep": 3}
    assert man["counts"]["finding_units_by_route_class"] == {"code_review": 3, "narrative_sweep": 1}
    assert man["counts"]["unsettleable_by_row"] == {"S20": 1}
    assert man["counts"]["narrative_units"] == 5 and man["counts"]["narrative_members"] == 7
    assert man["verdict_distribution_by_route_class"]["narrative_sweep"]["correct"] == 3
    assert man["review_hours"] == 1.5 and man["fm_audit"] == {"extractor_error": 0, "real_error": 1}
    assert set(man["private_files"]) == set(rs.PACKET_FILES.values()) | set(rs.RECORD_FILES.values())
    for rel, digest in man["private_files"].items():
        assert rs.sha256_file(env.dir / rel) == digest


def test_rederive_catches_page_edit(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    html_path, meta_path = env.dir / rs.PACKET_FILES["html"], env.dir / rs.PACKET_FILES["meta"]
    rs.write_lf(html_path, html_path.read_text(encoding="utf-8").replace("<body", "<body data-edit=\"1\"", 1))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"][rs.PACKET_FILES["html"]] = rs.sha256_file(html_path)
    rs.write_lf(meta_path, json.dumps(meta, indent=2) + "\n")
    filled = fill(env)
    assert gaps_of(env, filled) == []  # hashes agree with each other; only the rebuild sees the edit
    code, _, err = do_freeze(monkeypatch, capsys, env, filled)
    assert code == 3
    assert "packet.html differs from the rebuild" in err and "packet-meta.json differs from the rebuild" in err
    assert not (env.dir / rs.RECORD_FILES["verdicts"]).exists() and not mdir(env).exists()


def test_rederive_checks_run_files(tmp_path, monkeypatch):
    env = build(tmp_path)
    patch_rebuild(monkeypatch, env)
    rec = rf.load_record(str(env.dir), str(fill(env)))
    good = [(r.label, r.path) for r in env.runs]
    assert rf.rederive_check(rec, good) == []
    Path(env.runs[0].path).write_bytes(b"another log")
    assert any("does not hash" in g for g in rf.rederive_check(rec, good))
    assert any("was not given" in g for g in rf.rederive_check(rec, good[1:]))
    assert any("not in the packet" in g for g in rf.rederive_check(rec, good + [("other", env.runs[1].path)]))


def test_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    code, _, err = do_freeze(monkeypatch, capsys, env)
    assert code == 0, err
    before = {p: p.read_bytes() for p in env.dir.rglob("*") if p.is_file()}
    code, _, err = do_freeze(monkeypatch, capsys, env)
    assert code == 3 and "never overwritten" in err
    assert {p: p.read_bytes() for p in env.dir.rglob("*") if p.is_file()} == before


def test_freeze_refuses_manifest_dir_inside_packet_and_bad_sittings(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    patch_rebuild(monkeypatch, env)
    argv = freeze_argv(env, fill(env))
    argv[argv.index("--manifest-dir") + 1] = str(env.dir / "reviews")
    assert rf.main(argv) == 3 and "inside the packet folder" in capsys.readouterr().err
    gaps = rf.check_sittings([("2026-11-09T10:00:00-05:00", "2026-11-09T09:00:00-05:00"),
                              ("2026-11-08T09:00:00-05:00", "2026-11-08T11:00:00-05:00"),
                              ("2026-11-08T10:00:00-05:00", "2026-11-08T12:00:00-05:00"),
                              ("2026-11-10T09:00:00-05:00", "2026-11-10T10:00:00-05:00"),
                              ("2026-11-09T09:00:00", "2026-11-09T10:00:00")], DAY)
    assert any("not before end" in g for g in gaps) and any("overlap" in g for g in gaps)
    assert any("after the freeze date" in g for g in gaps) and any("UTC offset" in g for g in gaps)
    assert rf.check_sittings([], DAY) == ["sittings: at least one sitting is required"]


def test_freeze_reads_a_sittings_file(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    patch_rebuild(monkeypatch, env)
    sittings = tmp_path / "sittings.csv"
    rs.write_csv(sittings, ("start", "end"), [{"start": "2026-11-09T09:00:00-05:00",
                                               "end": "2026-11-09T09:45:00-05:00"}])
    argv = freeze_argv(env, fill(env))
    i = argv.index("--sitting")
    argv[i:i + 2] = ["--sittings", str(sittings)]
    assert rf.main(argv) == 0, capsys.readouterr().err
    assert (env.dir / "sittings.csv").read_bytes() == sittings.read_bytes()


def test_manifest_has_no_content(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    code, _, err = do_freeze(monkeypatch, capsys, env)
    assert code == 0, err
    text = (mdir(env) / "dev-test-run-manifest.json").read_text(encoding="utf-8")
    man = json.loads(text)
    banned = [u.unit_id for u in env.packet.units] + [NOTE, "BTC custody wallet", TEXT_A, TEXT_B, TEXT_U]
    banned += list(rs.narrative_sentences(TEXT_A)) + [m.route_key for m in env.packet.members if m.route_key]
    banned += ["case-a", "case-b", FORMAT_BREAK, "custodian_vocabulary", "amount_role_unresolved", NONLATIN[-1]]
    for b in banned:
        assert b not in text, b
    rest = dict(man)
    rest.pop("logs")  # log file names may name the model; nothing else may
    rest_text = json.dumps(rest, ensure_ascii=False)
    assert rs.model_hint_hits(rest_text) == []
    for b in [r.label for r in env.runs] + [r.model for r in env.runs] + [r.run_id for r in env.runs]:
        assert f'"{b}"' not in rest_text, b


def test_expanded_rows_map_back(tmp_path, monkeypatch, capsys):
    env = build(tmp_path)
    expanded = tmp_path / "expanded.csv"
    filled = fill(env, verdicts={env.ids["F_SWEEP"]: "citation_omission"})
    code, _, err = do_freeze(monkeypatch, capsys, env, filled, "--expanded-csv", str(expanded))
    assert code == 0, err
    rows = rs.read_rb7_csv(expanded)
    sweep = [r for r in rows if r["unit_id"] == env.ids["F_SWEEP"]]
    assert len(sweep) == 3
    assert {(r["run_label"], r["log_eval_id"], r["sample_id"], r["epoch"]) for r in sweep} == {
        ("alpha", "EVAL1ID", "case-a", "1"), ("alpha", "EVAL1ID", "case-a", "2"), ("beta", "EVAL2ID", "case-a", "1")}
    assert {r["verdict"] for r in sweep} == {"citation_omission"}
    assert all(r["scorers"] == "correctness_anchor" for r in sweep)
    narrative_members = sum(1 for m in env.packet.members if m.member_kind == "narrative")
    assert sum(r["row_kind"] == "narrative" for r in rows) == narrative_members
    assert [r for r in rows if r["row_kind"] == "fm_audit"][0]["route_key"] == "FM-000"


def test_trailer_format():
    assert rf.trailer("Rizwan Ahmed", "2026-11-09") == "Reviewed-by: Rizwan Ahmed, 2026-11-09"


# --- Freeze commit checks and the second-look draw ------------------------------------------

def _frozen(tmp_path, monkeypatch, capsys, **build_kw):
    env = build(tmp_path, **build_kw)
    code, _, err = do_freeze(monkeypatch, capsys, env, fill(env))
    assert code == 0, err
    return env


def _git(monkeypatch, env, *, hours=49, message=None, blob=None):
    man = mdir(env) / "dev-test-run-manifest.json"
    when = (datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc) - timedelta(hours=hours)).isoformat()
    msg = message if message is not None else f"Freeze review record\n\nReviewed-by: {REVIEWER}, {DAY}\n"
    calls = []

    def fake(repo, sha, path):
        calls.append((repo, sha, path))
        return when, msg, man.read_bytes() if blob is None else blob

    monkeypatch.setattr(rf, "git_commit_info", fake)
    monkeypatch.setattr(rf, "_utcnow", lambda: datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc))
    return calls


def draw_argv(env, *extra, out=None):
    return ["second-look-draw", "--packet-dir", str(env.dir), "--freeze-commit", "abcdef1",
            "--repo", str(env.tmp / "repo"), "--out", str(out or env.tmp / "second-look"), *extra]


def test_freeze_commit_checks(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env, message="Freeze review record\n\nReviewed-by: someone else, 2026-11-09\n")
    assert rf.main(draw_argv(env)) == 3
    assert "last line" in capsys.readouterr().err
    _git(monkeypatch, env, blob=b"{}\n")
    assert rf.main(draw_argv(env)) == 3
    assert "differs from the manifest file" in capsys.readouterr().err
    calls = _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    assert calls[-1][2] == "docs/evidence/reviews/dev-test-run-manifest.json"
    assert not (env.tmp / "repo" / ".git").exists()


def test_second_look_48h(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    rec = rf.load_frozen_record(str(env.dir))
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(rf.Refused):
        rf.draw_second_look(rec, seed=1, freeze_time=now - timedelta(hours=47), now=now)
    assert rf.draw_second_look(rec, seed=1, freeze_time=now - timedelta(hours=49), now=now)
    _git(monkeypatch, env, hours=47)
    assert rf.main(draw_argv(env)) == 3
    assert "48 hours" in capsys.readouterr().err
    _git(monkeypatch, env, hours=49)
    assert rf.main(draw_argv(env)) == 0


def test_second_look_draw_refuses_tampered_record_and_full_out(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env)
    out = tmp_path / "busy"
    out.mkdir()
    (out / "x.txt").write_bytes(b"x")
    assert rf.main(draw_argv(env, out=out)) == 3
    assert "not empty" in capsys.readouterr().err
    v = env.dir / rs.RECORD_FILES["verdicts"]
    v.write_bytes(v.read_bytes().replace(b"none_wrong", b"wrong", 1))
    assert rf.main(draw_argv(env)) == 3
    assert "does not hash to the manifest" in capsys.readouterr().err


def test_second_look_packet_blind(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env)
    out = tmp_path / "second-look"
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    first_ids = [u.unit_id for u in env.packet.units]
    reviewer_files = [p for p in (out / rs.REVIEWER_DIR).rglob("*") if p.is_file()]
    assert reviewer_files
    for p in reviewer_files:
        text = p.read_text(encoding="utf-8")
        for banned in [*first_ids, NOTE, REVIEWER, "first_class", "stratum", "not_correct",
                       *(r.label for r in env.runs), *(r.model for r in env.runs)]:
            assert banned not in text, (p.name, banned)
        assert not re.search(r"\bR[0-9]+\b", text), p.name
        assert rs.model_hint_hits(text) == []
    sl_rows = rs.read_csv(out / rs.PACKET_FILES["packet"], rs.PACKET_COLUMNS)
    assert sl_rows and all(r["unit_id"].startswith(("SF-", "SN-")) for r in sl_rows)
    assert {r["row_kind"] for r in sl_rows} <= {"finding", "narrative"}
    sl_map = rs.read_csv(out / rf.SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)
    assert {r["first_unit_id"] for r in sl_map} <= set(first_ids)
    meta = json.loads((out / rf.SECOND_LOOK_META_FILE).read_text(encoding="utf-8"))
    assert set(meta["runs"]) == {"R1", "R2"} and all("strata" in r for r in meta["runs"].values())
    for rel, digest in meta["files"].items():
        assert rs.sha256_file(out / rel) == digest
    assert (out / "sealed").is_dir() and "strata" not in "".join(p.read_text(encoding="utf-8")
                                                                 for p in reviewer_files)


def test_second_look_unit_rules(tmp_path, monkeypatch, capsys):
    env = build(tmp_path, fm_route="skip")
    rec = rf.load_record(str(env.dir), str(fill(env, verdicts={env.ids["N_B"]: "none_wrong"})))
    units = rf.second_look_units(rec)
    r1 = {u["first_unit_id"]: u for u in units["R1"]}
    assert r1[env.ids["N_B"]]["finding_unit_ids"] == [] and r1[env.ids["N_B"]]["first_class"] == "correct"
    assert r1[env.ids["N_B"]]["stratum"] == "code_review"
    assert env.ids["N_D"] not in r1  # its only route was skipped under --fm-epoch-routes skip
    assert env.ids["N_C"] not in r1  # FM-coded, not routed: audited, not a second-look unit
    assert r1[env.ids["N_A"]]["stratum"] == "both"
    assert r1[env.ids["N_A"]]["finding_unit_ids"] == sorted([env.ids["F_SWEEP"], env.ids["F_AMT"]])
    env2 = build(tmp_path / "f", fm_route="findings")
    rec2 = rf.load_record(str(env2.dir), str(fill(env2, verdicts={env2.ids["F_D"]: "not_reviewed"})))
    r1b = {u["first_unit_id"]: u for u in rf.second_look_units(rec2)["R1"]}
    assert r1b[env2.ids["N_D"]]["first_class"] == "not_correct"
    rec3 = rf.load_record(str(env.dir), str(fill(env, verdicts={env.ids["N_B"]: ("wrong", "1", "wallet_holder")})))
    assert {u["first_unit_id"]: u for u in rf.second_look_units(rec3)["R1"]}[env.ids["N_B"]]["first_class"] == (
        "not_correct")


def _units(n_correct, n_not, strata=("narrative_sweep", "code_review", "both"), weights=(1, 1, 1)):
    out, cycle = [], [s for s, w in zip(strata, weights) for _ in range(w)]
    for i in range(n_correct + n_not):
        out.append({"first_unit_id": f"N-{i:05d}", "stratum": cycle[i % len(cycle)],
                    "first_class": "not_correct" if i < n_not else "correct", "finding_unit_ids": []})
    return out


def test_second_look_sample_size():
    assert rf.sample_size(57) == 57 and rf.sample_size(1200) == 120 and rf.sample_size(150) == 100
    assert rf.sample_size(1001) == 101 and rf.sample_size(0) == 0
    d = rf._draw_details({"R1": _units(1130, 70)}, 3)["R1"]
    assert d["n"] == 120 and d["N"] == 1200
    drawn = set(d["drawn"])
    not_correct_ids = {u["first_unit_id"] for u in _units(1130, 70) if u["first_class"] == "not_correct"}
    assert len(drawn & not_correct_ids) == 50 and len(drawn) == 120
    assert sum(d["strata"]["not_correct"].values()) == 50 and sum(d["strata"]["correct"].values()) == 70
    assert rf._quotas({"a": 30, "b": 20, "c": 20}, 50) == {"a": 22, "b": 14, "c": 14}
    assert rf._quotas({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}  # tie broken by stratum name
    assert rf._draw_details({"R1": _units(1130, 70)}, 3) == rf._draw_details({"R1": _units(1130, 70)}, 3)
    assert rf._draw_details({"R1": _units(1130, 70)}, 4)["R1"]["drawn"] != d["drawn"]
    small = rf._draw_details({"R1": _units(50, 7)}, 3)["R1"]
    assert small["n"] == 57 and len(small["drawn"]) == 57
    short = rf._draw_details({"R1": _units(20, 120)}, 3)["R1"]  # n = 100: 50 not_correct, then only 20 correct
    assert short["n"] == 100 and len(short["drawn"]) == 70  # the 30 empty slots are not refilled


def test_kappa_known_values():
    first = [True] * 5 + [False] * 5
    second = [True] * 4 + [False] + [True] + [False] * 4
    assert rf.cohen_kappa(first, second) == pytest.approx(0.6)
    # p_o = 0.7; p_e = 0.7 * 0.8 + 0.3 * 0.2 = 0.62 by hand
    assert rf.cohen_kappa([True] * 7 + [False] * 3, [True] * 6 + [False] + [True] * 2 + [False]) == pytest.approx(
        (0.7 - 0.62) / 0.38)
    assert rf.cohen_kappa([True] * 4, [True] * 4) is None
    assert rf.cohen_kappa([], []) is None


def test_bootstrap_percentiles():
    first = [True] * 30 + [False] * 12
    second = [True] * 27 + [False] * 3 + [False] * 10 + [True] * 2
    a = rf.bootstrap_kappa(first, second, seed=5)
    assert a == rf.bootstrap_kappa(first, second, seed=5)
    assert a["resamples"] == 10_000 and a["ci_lower"] <= a["kappa"] <= a["ci_upper"]
    assert a != rf.bootstrap_kappa(first, second, seed=6)
    d = rf.bootstrap_kappa([True, True, False], [True, True, False], seed=1, resamples=200)
    assert d["degenerate_resamples"] > 0 and d["ci_lower"] == 0.0


def _run_result(**kw):
    base = {"sample_first_correct": 90, "sample_first_not_correct": 12, "run_first_not_correct": 12,
            "ci_lower": 0.7, "correct_to_not_correct": 0}
    base.update(kw)
    return base


def test_gate_branches():
    assert rf.gate(_run_result()) == {"branch": "kappa", "passed": True, "result": "confirmed"}
    assert rf.gate(_run_result(ci_lower=0.59))["result"] == "review not confirmed"
    assert rf.gate(_run_result(ci_lower=None))["passed"] is False
    low = _run_result(sample_first_not_correct=4, run_first_not_correct=4)
    assert rf.gate(low) == {"branch": "no_overturn", "passed": True, "result": "confirmed"}
    assert rf.gate(dict(low, correct_to_not_correct=1)) == {"branch": "no_overturn", "passed": False,
                                                             "result": "review not confirmed"}
    few = _run_result(sample_first_correct=8, sample_first_not_correct=12, run_first_not_correct=12)
    assert rf.gate(few) == {"branch": "too_few_per_class", "passed": False, "result": "review not confirmed"}
    assert tuple(rf.gate(few)) == rf.GATE_KEYS


def _second_look_fill(env, out, flip_first=None):
    """Fill the second-look packet's blank file synthetically on the scratch copy."""
    rows = rs.read_rb7_csv(out / rs.PACKET_FILES["blank"])
    sl_map = {r["sl_unit_id"]: r for r in rs.read_csv(out / rf.SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)}
    for row in rows:
        row["verdict"] = "correct" if row["row_kind"] == "finding" else "none_wrong"
        if flip_first and sl_map[row["unit_id"]]["first_unit_id"] == flip_first:
            row["verdict"] = "wrong" if row["row_kind"] == "finding" else "wrong"
            if row["row_kind"] == "narrative":
                row["sentence"], row["reason"] = "1", "amount"
        row.update({"note": NOTE, "reviewer": REVIEWER, "date": "2026-11-12"})
    path = env.tmp / "second-look-filled.csv"
    rs.write_rb7_csv(path, rows)
    return path


def score_argv(env, filled, *extra):
    return ["second-look-score", "--packet-dir", str(env.dir), "--second-look-dir", str(env.tmp / "second-look"),
            "--filled", str(filled), "--reviewer", REVIEWER, "--date", "2026-11-12", "--bootstrap-seed", "99",
            "--manifest-dir", str(mdir(env)), "--repo", str(env.tmp / "repo"), *extra]


def test_second_look_score_end_to_end(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    out = tmp_path / "second-look"
    filled = _second_look_fill(env, out, flip_first=env.ids["N_U1"])
    code = rf.main(score_argv(env, filled))
    stdout, err = capsys.readouterr()
    assert code == 0, err
    assert stdout.strip().splitlines()[-1] == f"Reviewed-by: {REVIEWER}, 2026-11-12"
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert tuple(man) == rf.SECOND_LOOK_MANIFEST_KEYS and man["agreement"] == "intra-rater"
    assert "inter-rater" not in json.dumps(man)
    assert set(man["runs"]) == {"alpha", "beta"}
    alpha = man["runs"]["alpha"]
    assert tuple(alpha) == rf.PUBLIC_RUN_RESULT_KEYS
    assert alpha["correct_to_not_correct"] == 1 and alpha["disagreements"] == 1
    assert alpha["gate"]["branch"] == "no_overturn" and alpha["gate"]["result"] == "review not confirmed"
    assert man["runs"]["beta"]["gate"]["result"] == "confirmed"
    result = json.loads((out / rf.SECOND_LOOK_RESULT_FILE).read_text(encoding="utf-8"))
    assert tuple(result) == rf.SECOND_LOOK_RESULT_KEYS
    assert result["runs"]["R1"]["disagreeing_first_unit_ids"] == [env.ids["N_U1"]]
    assert env.ids["N_U1"] not in json.dumps(man)
    assert rf.main(score_argv(env, filled)) == 3  # never overwritten
    assert "never overwritten" in capsys.readouterr().err


def test_second_look_score_refuses_a_bad_second_look_file(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0
    capsys.readouterr()
    out = tmp_path / "second-look"
    rows = rs.read_rb7_csv(out / rs.PACKET_FILES["blank"])  # every verdict blank
    bad = tmp_path / "blank-second-look.csv"
    rs.write_rb7_csv(bad, rows)
    assert rf.main(score_argv(env, bad)) == 3
    assert "verdict blank" in capsys.readouterr().err
    good = _second_look_fill(env, out)
    man = mdir(env) / "dev-test-run-manifest.json"
    man.write_bytes(man.read_bytes() + b" ")
    assert rf.main(score_argv(env, good)) == 3
    assert "first manifest does not hash" in capsys.readouterr().err


# --- Blinding check -------------------------------------------------------------------------

def blind_argv(env, models, *extra):
    return ["blind-check-draw", "--packet-dir", str(env.dir), "--freeze-commit", "abcdef1",
            "--repo", str(env.tmp / "repo"), "--models", models, "--out", str(env.tmp / "blind-check"), *extra]


def test_blind_check_rules(tmp_path, monkeypatch, capsys):
    env = build(tmp_path, n_runs=3)
    mark_template(env, "R3")
    monkeypatch.setattr(rf, "rederive_check", lambda rec, runs: [])
    code, _, err = do_freeze(monkeypatch, capsys, env)
    assert code == 0, err
    _git(monkeypatch, env)
    assert rf.main(blind_argv(env, "alpha=Model A")) == 3
    assert "does not name worker run 'beta'" in capsys.readouterr().err
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B,gamma=Model C")) == 3
    assert "template baseline is never offered" in capsys.readouterr().err
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B")) == 0, capsys.readouterr().err
    d3 = tmp_path / "blind-check"
    mapping = rs.read_csv(d3 / rf.BLIND_CHECK_MAP, rf.BLIND_CHECK_MAP_COLUMNS)
    assert mapping and {m["run_label"] for m in mapping} <= {"alpha", "beta"}
    page = (d3 / rf.BLIND_CHECK_HTML).read_text(encoding="utf-8")
    assert "Model A" in page and "Model B" in page and "Model C" not in page and "gamma" not in page
    for bad in ("<link", "@import", "url(", "src=", "href=", "\u2014", "\u2013"):
        assert bad not in page
    sheet = rs.read_csv(d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
    assert {r["narrative"] for r in sheet} == {TEXT_U + " Run 1.", TEXT_U + " Run 2.", TEXT_B}
    assert all(r["guess"] == "" for r in sheet)
    for p in (d3 / "reviewer").rglob("*"):
        text = p.read_text(encoding="utf-8")
        assert "alpha" not in text and "R1" not in text and "template" not in text


def test_full_run_second_look_needs_blinding_result(tmp_path, monkeypatch, capsys):
    env = _frozen(tmp_path, monkeypatch, capsys, n_runs=4)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    filled = _second_look_fill(env, tmp_path / "second-look")
    assert rf.main(score_argv(env, filled)) == 3
    assert "blind-check-result.json" in capsys.readouterr().err
    models = ",".join(f"{label}=Name {i}" for i, label in enumerate(LABELS))
    assert rf.main(blind_argv(env, models)) == 0, capsys.readouterr().err
    d3 = tmp_path / "blind-check"
    mapping = rs.read_csv(d3 / rf.BLIND_CHECK_MAP, rf.BLIND_CHECK_MAP_COLUMNS)
    sheet = rs.read_csv(d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
    truth = {m["check_id"]: m["display_name"] for m in mapping}
    for row in sheet:
        row["guess"] = truth[row["check_id"]]
    answers = tmp_path / "blind-answers.csv"
    rs.write_csv(answers, rf.BLIND_CHECK_COLUMNS, sheet)
    assert rf.main(["blind-check-score", "--packet-dir", str(env.dir), "--blind-check-dir", str(d3),
                    "--filled", str(answers)]) == 0, capsys.readouterr().err
    assert rf.main(["blind-check-score", "--packet-dir", str(env.dir), "--blind-check-dir", str(d3),
                    "--filled", str(answers)]) == 3
    capsys.readouterr()
    assert rf.main(score_argv(env, filled, "--blind-check-dir", str(d3))) == 0, capsys.readouterr().err
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["blinding"]["n"] == len(sheet) and tuple(man["blinding"]) == rf.BLINDING_KEYS


def test_blinding_check(tmp_path, monkeypatch, capsys):
    truth = {f"BC-{i:03d}": ("A" if i % 2 else "B") for i in range(40)}
    ids = sorted(truth)
    right = [{"check_id": c, "guess": truth[c] if i < 18 else ("B" if truth[c] == "A" else "A")}
             for i, c in enumerate(ids)]
    r = rf.blinding_result(right, truth)
    assert r["named_correctly"] == 18 and r["ci_lower"] > 0.25 and r["status"] == "blinding not maintained"
    ten = [dict(g, guess=truth[g["check_id"]] if i < 10 else ("B" if truth[g["check_id"]] == "A" else "A"))
           for i, g in enumerate(right)]
    r10 = rf.blinding_result(ten, truth)
    assert r10["named_correctly"] == 10 and r10["ci_lower"] <= 0.25 and r10["status"] != "blinding not maintained"
    with pytest.raises(rf.Refused):
        rf.blinding_result(right[:-1], truth)
    env = _frozen(tmp_path, monkeypatch, capsys, n_runs=1)
    _git(monkeypatch, env)
    assert rf.main(blind_argv(env, "alpha=Model A")) == 3
    assert "at least two worker runs" in capsys.readouterr().err


# --- What the script never does --------------------------------------------------------------

def test_no_score_reported():
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("content_pass", "pass_k", "pass^", "mean("):
        assert banned not in source, banned
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    for mod in ("reliability", "regression", "scoring"):
        assert not any(name.split(".")[-1] == mod for name in imported), mod
    assert rf.MANIFEST_KEYS == (
        "manifest_version", "kind", "run_id", "run_commit", "vocab_sha256", "settlement_table_sha256",
        "settlement_py_sha256", "tool_sha256", "logs", "private_files", "packet_seed", "whole_narrative_scope",
        "fm_epoch_routes", "counts", "verdict_distribution_by_route_class", "narrative_records", "fm_audit",
        "reviewer", "freeze_date", "sittings", "review_hours", "expanded_csv_sha256", "statement")
    assert rf.SECOND_LOOK_RESULT_KEYS == ("agreement", "bootstrap_seed", "runs")
    assert rf.SECOND_LOOK_MANIFEST_KEYS == (
        "manifest_version", "kind", "run_id", "agreement", "first_manifest_sha256", "freeze_commit", "draw_seed",
        "bootstrap_seed", "private_files", "runs", "blinding", "reviewer", "date", "statement")
    assert "disagreeing_first_unit_ids" not in rf.PUBLIC_RUN_RESULT_KEYS
    for key_name in (*rf.MANIFEST_KEYS, *rf.RUN_RESULT_KEYS, *rf.SECOND_LOOK_MANIFEST_KEYS):
        assert not re.search(r"score|pass_rate|anchor|content", key_name), key_name


def test_source_has_no_dashes_and_lf_only():
    for path in (SCRIPT, Path(__file__)):
        raw = path.read_bytes()
        assert b"\r" not in raw
        assert "\u2014" not in raw.decode("utf-8") and "\u2013" not in raw.decode("utf-8")


def test_git_reader_runs_only_git_show():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "git_commit_info")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "run"]
    assert len(calls) == 2
    for call in calls:
        argv = call.args[0]
        assert isinstance(argv, ast.List)
        assert [ast.literal_eval(e) for e in argv.elts[:4] if isinstance(e, ast.Constant)][:1] == ["git"]
        assert any(isinstance(e, ast.Constant) and e.value == "show" for e in argv.elts)
    other = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "run"
             and n not in calls]
    assert other == []


def test_help_and_bad_arguments_exit_3(capsys):
    assert _exit_code(["--help"]) == 3
    assert _exit_code(["freeze", "--packet-dir", "x"]) == 3
    capsys.readouterr()


def _exit_code(argv):
    try:
        return rf.main(argv)
    except SystemExit as e:
        return e.code


# --- Fix round F2 (2026-11-09): T, E and R reports ----------------------------------------------

def _drawn(tmp_path, monkeypatch, capsys, **build_kw):
    env = _frozen(tmp_path, monkeypatch, capsys, **build_kw)
    _git(monkeypatch, env)
    assert rf.main(draw_argv(env)) == 0, capsys.readouterr().err
    capsys.readouterr()
    return env, tmp_path / "second-look"


def _map_rows(out):
    return rs.read_csv(out / rf.SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)


def test_draw_seed_is_fixed_by_the_manifest_and_ids_are_salted(tmp_path, monkeypatch, capsys):
    """Red-team H1 and M2: nobody picks the draw seed, a second draw gives the same units,
    and second-look ids cannot be recomputed from first ids and the public seed."""
    env, out = _drawn(tmp_path, monkeypatch, capsys)
    again = tmp_path / "again"
    assert rf.main(draw_argv(env, out=again)) == 0, capsys.readouterr().err
    m1, m2 = _map_rows(out), _map_rows(again)
    assert {(r["first_unit_id"], r["run_id"]) for r in m1} == {(r["first_unit_id"], r["run_id"]) for r in m2}
    assert {r["sl_unit_id"] for r in m1}.isdisjoint({r["sl_unit_id"] for r in m2})
    meta = json.loads((out / rf.SECOND_LOOK_META_FILE).read_text(encoding="utf-8"))
    man_sha = rs.sha256_file(mdir(env) / "dev-test-run-manifest.json")
    assert meta["seed"] == rf.derived_seed(man_sha, "dev-test-run", "second_look")
    unsalted = {rp.second_look_id(meta["seed"], r["row_kind"], r["first_unit_id"]) for r in m1}
    assert unsalted.isdisjoint({r["sl_unit_id"] for r in m1})
    capsys.readouterr()
    assert _exit_code(draw_argv(env, "--seed", "3", out=tmp_path / "third")) == 3
    capsys.readouterr()


def test_second_look_score_refuses_an_edited_draw(tmp_path, monkeypatch, capsys):
    """Red-team H1 reproduction: rows dropped from the sealed map (rehashed in the meta), or
    another seed, no longer score as confirmed; the rebuild of the D-D draw refuses them."""
    env, out = _drawn(tmp_path, monkeypatch, capsys)
    filled = _second_look_fill(env, out, flip_first=env.ids["N_U1"])
    map_path, meta_path = out / rf.SECOND_LOOK_MAP_FILE, out / rf.SECOND_LOOK_META_FILE
    original_map, original_meta = map_path.read_bytes(), meta_path.read_bytes()
    rows = [r for r in _map_rows(out) if r["first_unit_id"] != env.ids["N_U1"]]
    rs.write_csv(map_path, rs.SECOND_LOOK_MAP_COLUMNS, rows)
    meta = json.loads(original_meta)
    meta["files"][rf.SECOND_LOOK_MAP_FILE] = rs.sha256_file(map_path)
    rs.write_lf(meta_path, json.dumps(meta, indent=2) + "\n")
    assert rf.main(score_argv(env, filled)) == 3
    assert "differs from the rebuild" in capsys.readouterr().err
    map_path.write_bytes(original_map)
    meta = json.loads(original_meta)
    meta["seed"] = 1
    rs.write_lf(meta_path, json.dumps(meta, indent=2) + "\n")
    assert rf.main(score_argv(env, filled)) == 3
    assert "draw seed that is not the one" in capsys.readouterr().err
    meta_path.write_bytes(original_meta)
    (out / "extra.txt").write_bytes(b"x")
    assert rf.main(score_argv(env, filled)) == 3
    assert "did not write: extra.txt" in capsys.readouterr().err
    (out / "extra.txt").unlink()
    assert rf.main(score_argv(env, filled)) == 0, capsys.readouterr().err
    man_text = (mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8")
    assert "id_salt" not in man_text


def test_second_look_dates_are_bounded(tmp_path, monkeypatch, capsys):
    """Demo defect 1: second-look rows and --date may not fall before the draw or after today."""
    env, out = _drawn(tmp_path, monkeypatch, capsys)
    filled = _second_look_fill(env, out)
    early = tmp_path / "early.csv"
    early.write_bytes(filled.read_bytes().replace(b"2026-11-12", b"2026-11-01"))
    argv = score_argv(env, early)
    argv[argv.index("--date") + 1] = "2026-11-01"
    assert rf.main(argv) == 3
    err = capsys.readouterr().err
    assert "before the second-look draw" in err and "before 2026-11-12" in err
    argv = score_argv(env, filled)
    argv[argv.index("--date") + 1] = "2026-11-20"
    assert rf.main(argv) == 3
    assert "is after today" in capsys.readouterr().err


def test_freeze_refuses_future_dates(tmp_path, monkeypatch, capsys):
    """Demo defect 2 and red-team L2: no freeze date after today, no sitting ending after the
    freeze, and the freeze commit may not predate the manifest freeze date."""
    env = build(tmp_path)
    patch_rebuild(monkeypatch, env)
    argv = freeze_argv(env, fill(env))
    argv[argv.index("--date") + 1] = "2099-12-31"
    assert rf.main(argv) == 3
    assert "is after today" in capsys.readouterr().err
    argv = freeze_argv(env, fill(env))
    argv[argv.index("--sitting") + 1] = "2026-11-09T20:00:00-05:00/2026-11-09T23:00:00-05:00"
    assert rf.main(argv) == 3
    assert "ends after the time of this freeze" in capsys.readouterr().err
    assert not mdir(env).exists()
    monkeypatch.setattr(rf, "git_commit_info", lambda repo, sha, path: (
        "2026-11-10T11:00:00+00:00", "x\n\nReviewed-by: A, 2026-11-20\n", b"{}"))
    gaps, _ = rf.check_freeze_commit(b"{}", {"reviewer": "A", "freeze_date": "2026-11-20"}, repo=".",
                                     sha="abcdef1", rel_path="m.json",
                                     now=datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert any("is after the commit" in g for g in gaps)


def test_expanded_csv_is_hashed_in_the_manifest(tmp_path, monkeypatch, capsys):
    """Red-team L2: the expanded CSV is written before the manifest and bound by its hash."""
    env = build(tmp_path)
    expanded = tmp_path / "expanded.csv"
    code, _, err = do_freeze(monkeypatch, capsys, env, None, "--expanded-csv", str(expanded))
    assert code == 0, err
    man = json.loads((mdir(env) / "dev-test-run-manifest.json").read_text(encoding="utf-8"))
    assert man["expanded_csv_sha256"] == rs.sha256_file(expanded)


def test_kappa_interval_is_none_when_kappa_is_undefined():
    """Demo defect 4: no [0.0, 0.0] interval is reported beside an undefined kappa."""
    b = rf.bootstrap_kappa([True] * 6, [True] * 6, seed=1, resamples=200)
    assert b["kappa"] is None and b["ci_lower"] is None and b["ci_upper"] is None
    assert b["degenerate_resamples"] == 200
    assert rf.gate({"sample_first_correct": 6, "sample_first_not_correct": 0, "run_first_not_correct": 0,
                    "ci_lower": None, "correct_to_not_correct": 0})["result"] == "confirmed"


def test_extra_cells_refused(tmp_path):
    """Red-team L1: a row with more cells than the header is refused, not trimmed."""
    env = build(tmp_path)
    filled = fill(env)
    lines = filled.read_bytes().split(b"\n")
    lines[1] = lines[1] + b",EXTRA-CELL"
    bad = tmp_path / "extra.csv"
    bad.write_bytes(b"\n".join(lines))
    assert any(g.startswith("item 1") and "cells" in g for g in gaps_of(env, bad))


def test_second_look_output_stays_outside_the_packet(tmp_path, monkeypatch, capsys):
    """Red-team L3: the default output folder is a sibling of the packet folder, and an
    output folder inside the packet folder is refused."""
    env = _frozen(tmp_path, monkeypatch, capsys)
    _git(monkeypatch, env)
    inside = env.dir / "second-look"
    assert rf.main(draw_argv(env, out=inside)) == 3
    assert "inside the packet folder" in capsys.readouterr().err and not inside.exists()
    argv = draw_argv(env)
    i = argv.index("--out")
    del argv[i:i + 2]
    assert rf.main(argv) == 0, capsys.readouterr().err
    assert (env.dir.parent / "packet-second-look" / rs.PACKET_FILES["html"]).is_file()


def test_second_look_reviewer_must_be_the_first_reviewer(tmp_path, monkeypatch, capsys):
    """Red-team L10: intra-rater agreement needs the same reviewer."""
    env, out = _drawn(tmp_path, monkeypatch, capsys)
    argv = score_argv(env, _second_look_fill(env, out))
    argv[argv.index("--reviewer") + 1] = "someone else"
    assert rf.main(argv) == 3
    assert "is not the first record" in capsys.readouterr().err


def test_blinding_check_is_bound_and_chance_follows_the_choices(tmp_path, monkeypatch, capsys):
    """Red-team M4 and M2: chance is one over the names offered, the guesses are kept, an
    edited result is refused at second-look-score, and check ids are salted."""
    import hashlib

    env, out = _drawn(tmp_path, monkeypatch, capsys)
    assert rf.main(blind_argv(env, "alpha=Model A,beta=Model B")) == 0, capsys.readouterr().err
    d3 = tmp_path / "blind-check"
    meta = json.loads((d3 / rf.BLIND_CHECK_META).read_text(encoding="utf-8"))
    man_sha = rs.sha256_file(mdir(env) / "dev-test-run-manifest.json")
    assert meta["seed"] == rf.derived_seed(man_sha, "dev-test-run", "blind_check")
    assert rf.chance_of(meta) == 0.5
    mapping = rs.read_csv(d3 / rf.BLIND_CHECK_MAP, rf.BLIND_CHECK_MAP_COLUMNS)
    sheet = rs.read_csv(d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
    old_ids = {"BC-" + hashlib.sha256(f"{meta['seed']}|blind_check|{u}|{r}".encode()).hexdigest()[:12]
               for u in env.ids.values() for r in ("R1", "R2")}
    assert not old_ids & {m["check_id"] for m in mapping}
    truth = {m["check_id"]: m["display_name"] for m in mapping}
    for row in sheet:
        row["guess"] = truth[row["check_id"]]
    answers = tmp_path / "blind-answers.csv"
    rs.write_csv(answers, rf.BLIND_CHECK_COLUMNS, sheet)
    assert rf.main(["blind-check-score", "--packet-dir", str(env.dir), "--blind-check-dir", str(d3),
                    "--filled", str(answers)]) == 0, capsys.readouterr().err
    result = json.loads((d3 / rf.BLIND_CHECK_RESULT).read_text(encoding="utf-8"))
    assert result["chance"] == 0.5 and (d3 / rf.BLIND_CHECK_ANSWERS).read_bytes() == answers.read_bytes()
    kept = (d3 / rf.BLIND_CHECK_RESULT).read_bytes()
    edited = dict(result, named_correctly=0, share=0.0)
    rs.write_lf(d3 / rf.BLIND_CHECK_RESULT, json.dumps(edited, indent=2) + "\n")
    filled = _second_look_fill(env, out)
    assert rf.main(score_argv(env, filled, "--blind-check-dir", str(d3))) == 3
    assert "differs from the result recomputed" in capsys.readouterr().err
    (d3 / rf.BLIND_CHECK_RESULT).write_bytes(kept)
    assert rf.main(score_argv(env, filled, "--blind-check-dir", str(d3))) == 0, capsys.readouterr().err
    man = json.loads((mdir(env) / "dev-test-run-second-look-manifest.json").read_text(encoding="utf-8"))
    assert man["blinding"]["chance"] == 0.5
    assert any(k.startswith("blind-check/") for k in man["private_files"])
