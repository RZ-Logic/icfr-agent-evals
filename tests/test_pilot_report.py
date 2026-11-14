"""scripts/pilot_report.py (contract C5): the section 13 pilot row from saved logs.

Findings F2-05 (no report), F2-01 (difficulty check (b) readings), F2-02 (McNemar
floor and paired-t drop), F2-06 (review-verdict CSV), F2-07 (queue volume as
distinct rows) and F2-09 (no Sample Consistency on the template log).

Content-pass^4 figures (pre-registration section 6 and section 8): with a frozen
review record (--review-record) the review-settled (score of record), audit-adjusted
(sensitivity), route-blind (upper bound) and code-only (extractor health) figures, each
with its interval, and FM-008 on the review-settled trial; without one the route-blind
and code-only figures and a plain note that the other two need a record. The code-only
figure is never labeled the score.

Mock logs only: the pilot task on mockllm with scripted outputs keyed per case, and
the template baseline on none/none. Dev split pilot only; nothing reads the
evaluation split. Logs go to pytest's temp dir unless ICFR_TEST_LOG_DIR is set. The
review records are synthetic: every verdict row says it is not a review.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals import baseline
from icfr_agent_evals import consistency as con
from icfr_agent_evals import reliability as rel
from icfr_agent_evals import roster
from icfr_agent_evals.casefile import ASSETS, truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.scoring import SweepHit
from icfr_agent_evals.task import variance_commentary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pilot_report.py"
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
# RB-7 (runbook; review_schema.RB7_COLUMNS), typed out so a change to the shared tuple fails here.
CSV_COLUMNS = ["row_kind", "unit_id", "run_label", "log_eval_id", "sample_id", "epoch", "scorers", "route_key",
               "route_class", "reason", "sentence", "verdict", "note", "reviewer", "date"]


def _load_script():
    spec = importlib.util.spec_from_file_location("pilot_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _load_script()
_REAL_PACKAGE_CODE_GAPS = pr.package_code_gaps
FAKE_FREEZE = "f" * 40


@pytest.fixture(autouse=True)
def _package_check_passes(monkeypatch):
    """The synthetic manifests here record run commit 0000000, which no repository holds,
    so the section 8 package-hash check (owner decision D-R9, choice 6) is stubbed to pass;
    test_refuses_a_record_whose_run_commit_package_differs runs the real check and
    tests/test_d_r9.py tests it on a scratch repository."""
    monkeypatch.setattr(pr, "package_code_gaps", lambda repo, commit: [])
    # The synthetic manifests sit outside this repository and in no commit, so the section 8
    # freeze-commit check is stubbed too; tests/test_freeze_ancestry.py tests it on a scratch
    # repository.
    monkeypatch.setattr(pr, "freeze_commit_gaps", _record_freeze_call)


FREEZE_CALLS: list[dict] = []


def _record_freeze_call(repo, mpath, mbytes, man, commit):
    FREEZE_CALLS.append({"repo": repo, "mpath": mpath, "commit": commit})
    return [], FAKE_FREEZE


def _log_dir(tmp_path_factory, label: str) -> str:
    root = os.environ.get("ICFR_TEST_LOG_DIR")
    if root:
        path = Path(root) / f"{label}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(tmp_path_factory.mktemp(label))


# --- Scripted pilot runs ------------------------------------------------------------------

def _pilot():
    return sorted(build_dataset(split="dev", subset="pilot"), key=lambda s: str(s.id))


def _answer(truth, direction: str) -> str:
    items = truth.eligible_items()
    word = "fell short of" if direction == "shortfall" else "exceeded"
    return json.dumps({
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": direction,
        "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in items)],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
        "narrative": f"On-chain reserves {word} ledger liabilities for the period.",
    })


def _flip(direction: str) -> str:
    return "surplus" if direction == "shortfall" else "shortfall"


def _cut_outputs(answer_of):
    """Like _keyed_outputs, but every generation stops on max_tokens with 48,000 output tokens,
    the shape of a thinking run cut at the ceiling."""
    inner = _keyed_outputs(answer_of)

    def fn(messages, tools, tool_choice, config) -> ModelOutput:
        out = inner(messages, tools, tool_choice, config)
        out.choices[0].stop_reason = "max_tokens"  # stop_reason is a read-only view of choices[0]
        out.usage = ModelUsage(input_tokens=1, output_tokens=48000, total_tokens=48001)
        return out

    return fn


def _keyed_outputs(answer_of):
    """custom_outputs for mockllm: each case answered by answer_of(truth), keyed on a
    scope wallet that sits in that case's input and in no other pilot case."""
    samples = _pilot()
    inputs = {str(s.id): str(s.input) for s in samples}
    outputs = {}
    for s in samples:
        t = truth_of(s.metadata)
        key = next(w for w in t.scope_wallets
                   if w in inputs[str(s.id)] and not any(w in v for o, v in inputs.items() if o != str(s.id)))
        outputs[key] = answer_of(t)

    def fn(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in outputs if w in blob), None)
        out = ModelOutput.from_content(model="mockllm/model", content=outputs.get(key, "unmatched"))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    return fn


def _run_pilot(tmp_path_factory, label: str, answer_of, *, subset="pilot", epochs=4, outputs=_keyed_outputs,
               **eval_kw) -> str:
    log = inspect_eval(
        variance_commentary(split="dev", subset=subset, epochs=epochs),
        model=get_model("mockllm/model", custom_outputs=outputs(answer_of)),
        model_roles=ROLES,
        log_dir=_log_dir(tmp_path_factory, label),
        display="none",
        **eval_kw,
    )[0]
    assert log.status == "success", log.error
    return log.location


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    """sonnet: correct; haiku_fb: a format break on every epoch; haiku_settled: the
    wrong direction on every case (FM-003, a settled failure); template: the baseline;
    nonpilot: a dev run without the subset; effort_low: the correct run with an
    eval-level reasoning_effort override (what --reasoning-effort low does)."""
    template = inspect_eval(
        baseline.variance_commentary_template_baseline(split="dev", subset="pilot"),
        log_dir=_log_dir(tmp_path_factory, "report-template"),
        display="none",
    )[0]
    assert template.status == "success", template.error
    return {
        "sonnet": _run_pilot(tmp_path_factory, "report-sonnet", lambda t: _answer(t, t.direction)),
        "haiku_fb": _run_pilot(tmp_path_factory, "report-haiku-fb", lambda t: "not one JSON object"),
        "haiku_cut": _run_pilot(tmp_path_factory, "report-haiku-cut", lambda t: "{\"asset\": \"BTC\", \"per",
                                outputs=_cut_outputs),
        "haiku_settled": _run_pilot(tmp_path_factory, "report-haiku-settled", lambda t: _answer(t, _flip(t.direction))),
        "template": template.location,
        "nonpilot": _run_pilot(tmp_path_factory, "report-nonpilot", lambda t: _answer(t, t.direction),
                               subset=None, epochs=1),
        "effort_low": _run_pilot(tmp_path_factory, "report-effort-low", lambda t: _answer(t, t.direction),
                                 reasoning_effort="low"),
    }


def _main(capsys, *argv):
    code = pr.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def _base(logs, haiku="haiku_fb"):
    return ["--sonnet", logs["sonnet"], "--haiku", logs[haiku], "--template", logs["template"]]


def _rewrite(tmp_path, src: str, name: str, mutate) -> str:
    log = read_eval_log(src)
    mutate(log)
    path = tmp_path / name
    write_eval_log(log, str(path))
    return str(path)


# --- End to end ---------------------------------------------------------------------------

SECTIONS = (
    "PILOT REPORT", "== Run sonnet", "== Run haiku", "== Run template", "content-pass^4", "FM counts",
    "Review queue volume", "Sample Consistency (summary)", "H4 line (rehearsal", pr.TEMPLATE_CONSISTENCY_LINE,
    "== Template margin per anchor (sonnet", "== Template margin per anchor (haiku",
    "== Difficulty check (a)", "== Difficulty check (b)", "McNemar block floor", "Paired-t minimal detectable drop",
    "== Deliberate regression", "PILOT REPORT: DONE",
)


def test_report_prints_every_section(logs, capsys):
    code, out, err = _main(capsys, *_base(logs))
    assert code == pr.EXIT_OK, err
    for section in SECTIONS:
        assert section in out, section
    assert "len(review_queue)" not in out and "scorer rows" not in out


def test_difficulty_a_passes_on_the_template(logs, capsys):
    code, out, _ = _main(capsys, *_base(logs))
    assert code == 0
    assert ("== Difficulty check (a) from the template log: PASS  memo-only cases failing through the "
            "reconciling-items score only 4/4; other standard cases clean 4/4") in out


def test_format_break_haiku_passes_reading_i_only(logs, capsys):
    """F2-01: an all-format-break Haiku passes the literal reading and no other."""
    code, out, _ = _main(capsys, *_base(logs, "haiku_fb"))
    assert code == 0
    assert "reading (i): PASS  failing hard cases 12, failing hard epochs 48" in out
    assert "reading (ii): FAIL  failing hard cases 0, failing hard epochs 0" in out
    assert "reading (iii): FAIL  failing hard cases 0, failing hard epochs 0" in out
    assert "reading of record: not chosen" in out


def test_generation_stops_tell_a_ceiling_cut_from_a_format_break(logs, capsys):
    """D6: a generation that stops on max_tokens is FM-000 like any format break, and the
    report says how many format breaks were ceiling cut-offs and how long generations ran."""
    code, out, _ = _main(capsys, *_base(logs, "haiku_cut"))
    assert code == 0
    haiku = out.split("== Run haiku")[1].split("== Run template")[0]
    assert "format-break epochs 80/80; errored epochs 0/80" in haiku
    assert "generation stops: max_tokens 80/80 (format breaks among them 80); other stop reasons: none" in haiku
    assert ("output tokens per generation (thinking included where billed): max 48,000, median 48,000, "
            "total 3,840,000 over 80 generations") in haiku
    sonnet = out.split("== Run sonnet")[1].split("== Run haiku")[0]
    assert "generation stops: max_tokens 0/80 (format breaks among them 0); other stop reasons: stop 80" in sonnet
    assert "max 1, median 1, total 80 over 80 generations" in sonnet


def test_generation_stops_in_the_json_payload(logs, tmp_path):
    out = tmp_path / "report.json"
    code = pr.main([*_base(logs, "haiku_cut"), "--json", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    stops = payload["runs"]["haiku"]["generation_stops"]
    assert stops["max_tokens_stops"] == 80 and stops["max_tokens_stops_format_break"] == 80
    assert stops["stop_reasons"] == {"max_tokens": 80}
    assert stops["output_tokens"] == {"n": 80, "max": 48000, "median": 48000, "total": 3840000}
    assert payload["runs"]["sonnet"]["generation_stops"]["stop_reasons"] == {"stop": 80}


def test_settled_failure_haiku_passes_every_reading(logs, capsys):
    code, out, _ = _main(capsys, *_base(logs, "haiku_settled"), "--reading-b", "iii")
    assert code == 0
    for reading in ("i", "ii"):
        assert f"reading ({reading}): PASS  failing hard cases 12, failing hard epochs 48" in out
    assert "reading (iii): PASS  (reading of record)  failing hard cases 12, failing hard epochs 48" in out
    assert "Difficulty check (b) verdict under reading (iii): PASS" in out


@pytest.mark.parametrize("asset", ASSETS)
def test_settled_failure_counts_on_every_asset(logs, asset):
    """The settled FM-003 failure counts under all three readings on each asset's hard cases."""
    b = pr.difficulty_b(read_eval_log(logs["haiku_settled"]))
    hard = {str(s.id) for s in read_eval_log(logs["haiku_settled"]).samples
            if s.metadata["truth"]["case_kind"] != "standard" and s.metadata["truth"]["asset"] == asset}
    assert hard
    for reading in pr.READINGS:
        assert hard <= set(b["readings"][reading]["failing_cases"])


def test_template_margin_and_floors(logs, capsys):
    code, out, _ = _main(capsys, *_base(logs))
    assert code == 0
    assert ("overall            reconciling_items_anchor   n=20  worker 3.000 template 2.650 "
            "margin +0.350 [+0.036, +0.664]") in out
    assert "McNemar block floor on content-pass^4: 7 one-way discordant cases with no reversal, 10 with one reversal" in out
    assert ("Sonnet log passes the route-blind content-pass^4 (the trial section 10(a) reads, D-E5) "
            "on 20 of 20") in out
    assert "inferred" in out and "sd 0.5: 0.390" in out


def test_template_never_gets_sample_consistency(logs, capsys, monkeypatch):
    """F2-09: the template log is never passed to consistency.report_from_log."""
    seen = []
    real = con.report_from_log

    def spy(log, k=None, **kw):
        seen.append(log.eval.task)
        return real(log, k, **kw)

    monkeypatch.setattr(con, "report_from_log", spy)
    code, out, _ = _main(capsys, *_base(logs))
    assert code == 0
    assert seen and not any(str(t).endswith(pr.TEMPLATE_TASK) for t in seen)
    template_block = out.split("== Run template")[1].split("== Template margin")[0]
    assert pr.TEMPLATE_CONSISTENCY_LINE in template_block
    assert "FM-007" not in template_block and "H4" not in template_block


def test_queue_volume_counts_distinct_rows(logs, capsys):
    """F2-07: a format break is six scorer rows but one (sample, epoch, reason)."""
    code, out, _ = _main(capsys, *_base(logs))
    assert code == 0
    haiku = out.split("== Run haiku")[1].split("== Run template")[0]
    assert "Review queue volume: 80 distinct (sample, epoch, reason), 80 distinct (sample, epoch)" in haiku
    assert "format_break     80 (sample, epoch, reason), 80 (sample, epoch)" in haiku
    fm_line = next(line for line in haiku.splitlines() if line.strip().startswith(pr.FM_LISTED))
    assert "not a section 10(c) route" in fm_line  # its own line (W5 note)


def test_review_csv_columns_and_rows(logs, capsys, tmp_path):
    """F2-06: exactly the RB-7 columns, one row per distinct (run, sample, epoch, route
    key), verdict, reviewer and date blank. A format break is a queue row, not a finding
    and not an audit row: row_kind blank, route_class format_break."""
    path = tmp_path / "review.csv"
    code, _, err = _main(capsys, *_base(logs), "--review-csv", path)
    assert code == 0, err
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 80  # 20 cases x 4 epochs, one format-break route each, Haiku only
    keys = [(r["run_label"], r["sample_id"], r["epoch"], r["route_key"]) for r in rows]
    assert len(set(keys)) == len(keys)
    assert {r["run_label"] for r in rows} == {"haiku"}
    assert {r["route_key"] for r in rows} == {pr.FORMAT_BREAK}
    assert all(r["verdict"] == r["reviewer"] == r["date"] == "" for r in rows)
    assert all(r["unit_id"] == r["sentence"] == r["note"] == "" for r in rows)
    assert all(r["row_kind"] == "" and r["route_class"] == pr.FORMAT_BREAK for r in rows)
    assert all(len(r["scorers"].split(";")) == 6 for r in rows)
    assert tuple(CSV_COLUMNS) == pr.REVIEW_CSV_COLUMNS
    from icfr_agent_evals import review_schema

    assert pr.REVIEW_CSV_COLUMNS == review_schema.RB7_COLUMNS
    assert pr.VERDICTS == review_schema.FINDING_VERDICTS


def test_json_output(logs, capsys, tmp_path):
    path = tmp_path / "report.json"
    code, _, _ = _main(capsys, *_base(logs), "--json", path)
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    figs = data["runs"]["template"]["content_pass4_figures"]
    assert figs["figures"]["code_only"]["passed"] == 15
    assert figs["score_of_record"] is None and figs["figures"]["review_settled"]["computed"] is False
    assert "content_pass4" not in data["runs"]["template"]  # the old code-only field is gone, never read as a score
    assert data["floors"]["sonnet_pass4_trial"] == "route_blind"
    assert data["runs"]["template"]["sample_consistency"] == pr.TEMPLATE_CONSISTENCY_LINE
    assert data["difficulty_a"]["verdict"] == "PASS"
    assert {r: v["verdict"] for r, v in data["difficulty_b"]["readings"].items()} == {"i": "PASS", "ii": "FAIL", "iii": "FAIL"}
    assert data["floors"]["mcnemar_floor"] == {"no_reversal": 7, "one_reversal": 10}


def test_regression_caught_only_on_an_automatic_block(logs, capsys):
    """F2-02 under D2: caught means compare_runs.py exit 1, the rejecting tests named."""
    code, out, _ = _main(capsys, *_base(logs), "--regression-baseline", logs["sonnet"],
                         "--regression-candidate", logs["haiku_settled"])
    assert code == 0
    verdict = next(line for line in out.splitlines() if line.startswith("Regression verdict:"))
    assert verdict.startswith("Regression verdict: caught: compare_runs.py exit 1, automatic block by ")
    assert "correctness" in verdict and "content_pass4" in verdict
    code, out, _ = _main(capsys, *_base(logs), "--regression-baseline", logs["sonnet"],
                         "--regression-candidate", logs["sonnet"])
    assert code == 0
    assert "Regression verdict: MISSED (no automatic block): compare_runs.py exit 0" in out
    assert "caught" not in out.lower()  # the word appears only on an exit 1 (contract C5)


# --- Content-pass^4 figures (section 6, section 8) and the frozen review record ------------

REVIEW_NOTE = "synthetic test row, not a review"
REVIEW_SEED = 20261110
FIGURE_RE = re.compile(r"content-pass\^4\s+\d+/\d+")
NEEDS = "not computed: needs a frozen review record (pass --review-record PACKET_DIR)"


def _block(out: str, role: str) -> str:
    nxt = {"sonnet": "== Run haiku", "haiku": "== Run template", "template": "== Template margin"}[role]
    return out.split(f"== Run {role}")[1].split(nxt)[0]


def _figure_lines(text: str) -> dict[str, str]:
    """figure name -> its printed line (the four lines under 'Content-pass^4 figures')."""
    got = {}
    for line in text.splitlines():
        for name in pr.FIGURE_NAMES.values():
            if line.strip().startswith(f"{name} content-pass^4"):
                got[name] = line
    return got


SITTING = ("2026-11-10T09:00:00+00:00", "2026-11-10T10:00:00+00:00")


def _freeze(packet_dir: Path, rows: list[dict]) -> None:
    """Write verdicts.csv, fm_audit.csv and sittings.csv the way review_freeze.freeze does
    (the manifest is written apart, by _write_manifest, once every file is final)."""
    from icfr_agent_evals import review_schema as rs

    rf = pr.load_review_freeze()
    filled = packet_dir.parent / f"{packet_dir.name}-filled.csv"
    rs.write_rb7_csv(filled, rows)
    rec = rf.load_record(str(packet_dir), str(filled))
    rs.write_rb7_csv(packet_dir / rs.RECORD_FILES["verdicts"], rf.record_rows(rec, ("finding", "narrative")))
    rs.write_rb7_csv(packet_dir / rs.RECORD_FILES["fm_audit"], rf.record_rows(rec, ("fm_audit",)))
    rs.write_csv(packet_dir / rs.RECORD_FILES["sittings"], rf.SITTINGS_COLUMNS,
                 [{"start": SITTING[0], "end": SITTING[1]}])


def _write_manifest(packet_dir: Path, path: Path) -> None:
    """The public freeze manifest the way review_freeze.freeze writes it, outside the packet."""
    rf = pr.load_review_freeze()
    rec = rf.load_frozen_record(str(packet_dir))
    man = rf.build_manifest(rec, run_id=str(rec.meta["run_id"]), run_commit="0000000", reviewer="SYNTHETIC-TEST",
                            sittings=[SITTING], freeze_date="2026-11-10")
    rf._write_json(path, man)


def _rec(records, name: str) -> list[str]:
    """--review-record and --review-manifest for one of the fixture's records."""
    return ["--review-record", getattr(records, name), "--review-manifest", getattr(records, f"{name}_manifest")]


@pytest.fixture(scope="module")
def records(logs, tmp_path_factory):
    """Frozen review records on the correct Sonnet stand-in log, built with
    scripts/review_packet.py (scope hard, D-J): `correct` holds every narrative
    none_wrong; `wrong` rules one hard case's narrative unit wrong (error type other,
    D-I); `unfrozen` is the packet with no frozen files; `incomplete` leaves one verdict
    blank; `elsewhere` names another eval id in runs.json. Every row is synthetic."""
    from icfr_agent_evals import review_schema as rs

    spec = importlib.util.spec_from_file_location("review_packet_for_pilot_report_tests",
                                                  ROOT / "scripts" / "review_packet.py")
    rp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rp  # its dataclasses resolve their module by name
    spec.loader.exec_module(rp)
    root = tmp_path_factory.mktemp("records")
    runs = rp.load_runs([("sonnet", logs["sonnet"])], REVIEW_SEED)
    packet = rp.build_packet(runs, run_id="pilot-report-test", seed=REVIEW_SEED, scope="hard",
                             fm_epoch_routes="findings")
    out = {}
    for name in ("correct", "wrong", "unfrozen", "incomplete", "elsewhere"):
        rp.write_packet(packet, str(root / name))
        out[name] = root / name
    blank = rs.read_rb7_csv(out["correct"] / rs.PACKET_FILES["blank"])
    members = rs.read_csv(out["correct"] / rs.PACKET_FILES["members"], rs.MEMBER_COLUMNS)
    case_of = {m["unit_id"]: m["sample_id"] for m in members}

    def filled(verdict_of):
        rows = []
        for r in blank:
            r = dict(r)
            r.update(verdict_of(r))
            r.update(note=REVIEW_NOTE, reviewer="SYNTHETIC-TEST", date="2026-11-10")
            rows.append(r)
        return rows

    def all_correct(r):
        return {"verdict": {"finding": "correct", "narrative": "none_wrong", "fm_audit": "real_error"}[r["row_kind"]]}

    narratives = [r for r in blank if r["row_kind"] == "narrative"]
    assert narratives, "scope hard gives every hard case a narrative-only unit"
    wrong_unit = narratives[0]["unit_id"]

    def one_wrong(r):
        if r["unit_id"] == wrong_unit:
            return {"verdict": "wrong", "reason": "other", "sentence": "1"}
        return all_correct(r)

    _freeze(out["correct"], filled(all_correct))
    _freeze(out["wrong"], filled(one_wrong))
    _freeze(out["incomplete"], filled(lambda r: {"verdict": ""} if r["unit_id"] == wrong_unit else all_correct(r)))
    _freeze(out["elsewhere"], filled(all_correct))
    runs_path = out["elsewhere"] / rs.PACKET_FILES["runs"]
    data = json.loads(runs_path.read_text(encoding="utf-8"))
    for info in data.values():
        info["eval_id"] = "not-a-log-given-to-the-report"
    runs_path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    manifests = {}
    for name in ("correct", "wrong", "incomplete", "elsewhere"):  # `unfrozen` has no record files to hash
        manifests[f"{name}_manifest"] = root / f"{name}-manifest.json"
        _write_manifest(out[name], manifests[f"{name}_manifest"])
    return NS(**{k: str(v) for k, v in {**out, **manifests}.items()}, wrong_case=case_of[wrong_unit])


def test_without_a_record_prints_route_blind_and_code_only_and_says_the_record_is_needed(logs, capsys):
    code, out, err = _main(capsys, *_base(logs))
    assert code == pr.EXIT_OK, err
    sonnet = _block(out, "sonnet")
    assert "Content-pass^4 figures (section 6, section 8; no frozen review record):" in sonnet
    lines = _figure_lines(sonnet)
    assert set(lines) == set(pr.FIGURE_NAMES.values())
    for name in ("review-settled", "audit-adjusted"):
        assert lines[name].endswith(NEEDS) and not FIGURE_RE.search(lines[name]), lines[name]
    assert "20/20 = 1.000 (Clopper-Pearson 95% [0.832, 1.000])" in lines["route-blind"]
    assert lines["route-blind"].endswith("UPPER BOUND on the review-settled figure, not a capability result")
    assert lines["code-only"].endswith("EXTRACTOR HEALTH only: not a capability result, never a score")
    assert ("review-settled and audit-adjusted content-pass^4, FM-008 on the review-settled trial and the per-anchor "
            "means review-settled and audit-adjusted: not computed; each needs a frozen review record") in sonnet
    assert "SCORE OF RECORD" not in out and "FM-008 on the review-settled trial:" not in out
    assert "per sample (route-blind / code-only): " in sonnet
    assert "FM-008 on the code-only trial (extractor health only, not the FM-008 count): 0" in sonnet
    # D-R9: citation-pass^4 beside the figures, read from the log alone; no review-found without a record
    assert ("citation-pass^4 (D-R9: its own measure, never a content failure) 20/20 = 1.000 (Clopper-Pearson 95% "
            "[0.832, 1.000]); epochs with an omission 0 (0 tokens, of them wallet-label matcher 0), not assessable 0, "
            "review-found needs a frozen review record; synthetic") in sonnet
    assert "content-pass^4 under the rule before D-R9, route-blind: 20/20" in sonnet
    assert "content-pass^4 under the rule before D-R9, review-settled" not in sonnet


def test_with_a_record_prints_all_four_figures_labeled_and_fm008_on_the_review_settled_trial(logs, records, capsys,
                                                                                               tmp_path):
    path = tmp_path / "report.json"
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "correct"), "--json", path)
    assert code == pr.EXIT_OK, err
    sonnet = _block(out, "sonnet")
    assert f"frozen review record {records.correct}, run R1):" in sonnet
    lines = _figure_lines(sonnet)
    for name in pr.FIGURE_NAMES.values():
        assert "20/20 = 1.000 (Clopper-Pearson 95% [0.832, 1.000])" in lines[name], lines[name]
    assert lines["review-settled"].endswith("SCORE OF RECORD (owner decision D-A)")
    assert lines["audit-adjusted"].endswith("SENSITIVITY only, never the score of record")
    assert lines["route-blind"].endswith("UPPER BOUND on the review-settled figure, not a capability result")
    assert lines["code-only"].endswith("EXTRACTOR HEALTH only: not a capability result, never a score")
    assert "FM-008 on the review-settled trial: 0" in sonnet
    assert "per sample (review-settled / audit-adjusted / route-blind / code-only): " in sonnet
    for anchor in pr.rel.TRIAL_ANCHORS:
        assert re.search(rf"{anchor}\s+review-settled 3\.000 \[3\.000, 3\.000\], "
                         r"audit-adjusted 3\.000 \[3\.000, 3\.000\]", sonnet), anchor
    assert ("review-found (counted apart from code-found): codes none; citation omissions 0; "
            "whole-narrative wrong 0") in sonnet
    # The record names only the Sonnet run: the other runs say so and print no record figure.
    for role in ("haiku", "template"):
        block = _block(out, role)
        assert "no frozen review record for this run: the record names no run with eval id" in block
        assert _figure_lines(block)["review-settled"].endswith(NEEDS)
    data = json.loads(path.read_text(encoding="utf-8"))
    figs = data["runs"]["sonnet"]["content_pass4_figures"]
    assert figs["score_of_record"] == "review_settled"
    assert figs["record"] == {"packet_dir": records.correct, "run_id": "R1"}
    assert {k: v["passed"] for k, v in figs["figures"].items()} == {
        "review_settled": 20, "audit_adjusted": 20, "route_blind": 20, "code_only": 20}
    assert figs["fm008"] == {"trial": "review_settled", "samples": []}
    sha = hashlib.sha256(Path(records.correct_manifest).read_bytes()).hexdigest()
    assert data["review_record"] == {"packet_dir": records.correct,
                                     "run_ids": {"sonnet": "R1", "haiku": None, "template": None},
                                     "manifest": {"path": records.correct_manifest, "sha256": sha,
                                                  "freeze_commit": FAKE_FREEZE,
                                                  "settlement_commit": data["review_record"]["manifest"]["settlement_commit"]}}
    assert re.fullmatch(r"[0-9a-f]{40}", data["review_record"]["manifest"]["settlement_commit"])
    assert (f"Frozen review record {records.correct}: every file it lists hashes to its manifest "
            f"{records.correct_manifest} (sha256 {sha})") in out
    assert data["runs"]["haiku"]["content_pass4_figures"]["score_of_record"] is None
    # D-R9: the citation line, the figures under the rule before D-R9 and the JSON keys
    assert "review-found 0; synthetic" in sonnet
    for name in ("review-settled", "audit-adjusted", "route-blind", "code-only"):
        assert f"content-pass^4 under the rule before D-R9, {name}: 20/20" in sonnet
    assert "means, review-settled, rule before D-R9 (H1, H3): overall fabrication 3.000" in sonnet
    cite = figs["citation_pass4"]
    assert (cite["passed"], cite["n"], cite["review_found"]) == (20, 20, 0)
    assert {k: v["passed"] for k, v in figs["content_pass4_before_d_r9"].items()} == {
        "review_settled": 20, "audit_adjusted": 20, "route_blind": 20, "code_only": 20}


def test_review_settled_reads_the_verdicts_and_the_code_only_figure_is_not_the_score(logs, records, capsys, tmp_path):
    """A whole-narrative `wrong` (D-I) fails one hard case on the score of record; the
    route-blind and code-only figures, which read the log alone, do not move."""
    path = tmp_path / "report.json"
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "wrong"), "--json", path)
    assert code == pr.EXIT_OK, err
    lines = _figure_lines(_block(out, "sonnet"))
    assert "19/20 = 0.950" in lines["review-settled"] and "19/20 = 0.950" in lines["audit-adjusted"]
    assert "20/20 = 1.000" in lines["route-blind"] and "20/20 = 1.000" in lines["code-only"]
    figs = json.loads(path.read_text(encoding="utf-8"))["runs"]["sonnet"]["content_pass4_figures"]
    settled = figs["figures"]["review_settled"]["per_sample"]
    assert [sid for sid, v in settled.items() if not v] == [records.wrong_case]
    assert figs["review_found"]["whole_narrative_wrong"] >= 1
    # The printed figures are trials.figures() on the same log and frozen record.
    from icfr_agent_evals import trials

    record, _manifest = pr.load_review_record(records.wrong, records.wrong_manifest)
    ref = trials.figures(read_eval_log(logs["sonnet"]), record)
    for kind, row in ref["figures"].items():
        assert (figs["figures"][kind]["passed"], figs["figures"][kind]["ci95"]) == (row["passed"], row["ci95"])


@pytest.mark.parametrize("with_record", [False, True])
def test_code_only_figure_is_never_printed_as_the_score(logs, records, capsys, with_record):
    extra = _rec(records, "wrong") if with_record else []
    code, out, err = _main(capsys, *_base(logs), *extra)
    assert code == pr.EXIT_OK, err
    names = tuple(f"{n} content-pass^4" for n in pr.FIGURE_NAMES.values())
    for line in out.splitlines():
        if FIGURE_RE.search(line):  # every printed content-pass^4 count names its figure
            assert line.strip().startswith(names), line
        if line.strip().startswith("code-only"):
            assert "SCORE OF RECORD" not in line and "SENSITIVITY" not in line and "UPPER BOUND" not in line, line
        if "SCORE OF RECORD" in line:
            assert line.strip().startswith("review-settled content-pass^4"), line
    assert not any(line.startswith("content-pass^4 ") for line in out.splitlines())  # the old unlabeled line
    code_lines = [line for line in out.splitlines() if line.strip().startswith("code-only content-pass^4")]
    assert len(code_lines) == 3 and all(line.endswith("never a score") for line in code_lines)
    assert out.count("SCORE OF RECORD") == (1 if with_record else 0)


def test_refuses_a_missing_or_unfrozen_record(logs, capsys, records, tmp_path):
    code, out, err = _main(capsys, *_base(logs), "--review-record", tmp_path / "absent")
    assert code == pr.EXIT_REFUSED and out == "" and "not a packet directory" in err
    code, out, err = _main(capsys, *_base(logs), "--review-record", records.unfrozen)
    assert code == pr.EXIT_REFUSED and out == "" and "unreadable" in err and "verdicts.csv" in err


def _copy_record(records, tmp_path) -> tuple[Path, Path]:
    """A copy of the `correct` record and its manifest, safe to tamper with."""
    packet = tmp_path / "copy"
    shutil.copytree(records.correct, packet)
    manifest = tmp_path / "copy-manifest.json"
    shutil.copyfile(records.correct_manifest, manifest)
    return packet, manifest


def _redump_json(path: Path) -> None:
    """Same JSON content, different bytes."""
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1), encoding="utf-8",
                    newline="\n")


def _flip_a_verdict(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert ",none_wrong," in text
    path.write_text(text.replace(",none_wrong,", ",wrong,", 1), encoding="utf-8", newline="\n")


def test_a_clean_copy_of_a_frozen_record_passes_the_frozen_record_check(logs, capsys, records, tmp_path):
    packet, manifest = _copy_record(records, tmp_path)
    code, out, err = _main(capsys, *_base(logs), "--review-record", packet, "--review-manifest", manifest)
    assert code == pr.EXIT_OK, err
    assert f"Frozen review record {packet}: every file it lists hashes to its manifest {manifest}" in out
    assert "SCORE OF RECORD" in out


@pytest.mark.parametrize("rel, tamper", [
    ("verdicts.csv", _flip_a_verdict),
    ("sealed/runs.json", _redump_json),
    ("packet-meta.json", _redump_json),
])
def test_refuses_a_tampered_record(logs, capsys, records, tmp_path, rel, tamper):
    """review_freeze.check_record_files (the check second-look-draw runs) refuses a record
    whose files no longer hash to its manifest, before any figure is computed."""
    packet, manifest = _copy_record(records, tmp_path)
    tamper(packet / rel)
    code, out, err = _main(capsys, *_base(logs), "--review-record", packet, "--review-manifest", manifest)
    assert code == pr.EXIT_REFUSED and out == ""
    assert "fails the frozen-record check" in err
    assert f"private file {rel} does not hash to the manifest" in err


def test_refuses_a_record_whose_manifest_is_missing_or_not_its_own(logs, capsys, records, tmp_path):
    # No --review-manifest: the default docs/evidence/reviews/<run-id>-manifest.json is not there.
    assert not (pr.REPO_ROOT / "docs" / "evidence" / "reviews" / "pilot-report-test-manifest.json").exists()
    code, out, err = _main(capsys, *_base(logs), "--review-record", records.correct)
    assert code == pr.EXIT_REFUSED and out == ""
    assert "fails the frozen-record check" in err and "manifest pilot-report-test-manifest.json not found" in err
    # A manifest for another run.
    packet, manifest = _copy_record(records, tmp_path)
    man = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**man, "run_id": "another-run"}), encoding="utf-8", newline="\n")
    code, out, err = _main(capsys, *_base(logs), "--review-record", packet, "--review-manifest", manifest)
    assert code == pr.EXIT_REFUSED and out == "" and "the manifest is not this packet's freeze manifest" in err
    # A manifest that leaves the frozen verdicts out of its private files.
    files = {k: v for k, v in man["private_files"].items() if k != "verdicts.csv"}
    manifest.write_text(json.dumps({**man, "private_files": files}), encoding="utf-8", newline="\n")
    code, out, err = _main(capsys, *_base(logs), "--review-record", packet, "--review-manifest", manifest)
    assert code == pr.EXIT_REFUSED and out == "" and "the manifest does not list private file verdicts.csv" in err


def test_refuses_a_record_whose_run_commit_package_differs(logs, capsys, records, monkeypatch):
    """Section 8 package-hash check (owner decision D-R9, choice 6): the manifest's run
    commit must hold the package code the report runs; 0000000 is no commit here."""
    monkeypatch.setattr(pr, "package_code_gaps", _REAL_PACKAGE_CODE_GAPS)
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "correct"))
    assert code == pr.EXIT_REFUSED and out == ""
    assert "fails the frozen-record check" in err and "run commit 0000000 cannot be read" in err


def test_freeze_commit_needs_a_review_record(logs, capsys):
    code, out, err = _main(capsys, *_base(logs), "--freeze-commit", "abcdef1")
    assert code == pr.EXIT_REFUSED and out == "" and "--freeze-commit needs --review-record" in err


def test_freeze_commit_reaches_the_check(logs, capsys, records):
    FREEZE_CALLS.clear()
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "correct"), "--freeze-commit", "abcdef1")
    assert code == 0, err
    assert FREEZE_CALLS[-1]["commit"] == "abcdef1" and FREEZE_CALLS[-1]["repo"] == pr.REPO_ROOT
    assert str(FREEZE_CALLS[-1]["mpath"]) == records.correct_manifest
    assert f"frozen in commit {FAKE_FREEZE[:12]}, an ancestor of the settlement commit " in out


def test_refuses_a_record_whose_freeze_commit_fails(logs, capsys, records, monkeypatch):
    """Section 8: a freeze-commit gap refuses the record before any figure."""
    monkeypatch.setattr(pr, "freeze_commit_gaps",
                        lambda repo, mpath, mbytes, man, commit: (["freeze commit abc is not an ancestor of the "
                                                                   "settlement commit (HEAD)"], ""))
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "correct"))
    assert code == pr.EXIT_REFUSED and out == ""
    assert "fails the frozen-record check" in err and "not an ancestor of the settlement commit" in err


def test_review_manifest_needs_a_review_record(logs, capsys, records):
    code, out, err = _main(capsys, *_base(logs), "--review-manifest", records.correct_manifest)
    assert code == pr.EXIT_REFUSED and out == "" and "--review-manifest needs --review-record" in err


def test_refuses_a_record_that_names_none_of_the_logs(logs, capsys, records):
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "elsewhere"))
    assert code == pr.EXIT_REFUSED and out == ""
    assert "names none of the sonnet, haiku or template logs" in err and "not-a-log-given-to-the-report" in err


def test_refuses_a_record_that_does_not_cover_its_run(logs, capsys, records):
    code, out, err = _main(capsys, *_base(logs), *_rec(records, "incomplete"))
    assert code == pr.EXIT_REFUSED and out == ""
    assert "the frozen review record does not cover run R1 (" in err


def test_figure_labels_follow_the_preregistration():
    assert pr.FIGURE_ORDER == ("review_settled", "audit_adjusted", "route_blind", "code_only")
    assert "SCORE OF RECORD" in pr.FIGURE_ROLES["review_settled"]
    assert "never the score of record" in pr.FIGURE_ROLES["audit_adjusted"]
    assert "UPPER BOUND" in pr.FIGURE_ROLES["route_blind"]
    assert "never a score" in pr.FIGURE_ROLES["code_only"] and "EXTRACTOR HEALTH" in pr.FIGURE_ROLES["code_only"]
    assert sum("score of record" in r.lower() and "never" not in r.lower() for r in pr.FIGURE_ROLES.values()) == 1


def test_no_dash_or_cr_characters_in_the_script_and_tests():
    for path in (SCRIPT, Path(__file__)):
        raw = path.read_bytes()
        assert b"\r" not in raw
        text = raw.decode("utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text


# --- Refusals (exit 3) --------------------------------------------------------------------

def test_refuses_a_missing_file(logs, capsys, tmp_path):
    code, out, err = _main(capsys, "--sonnet", tmp_path / "absent.eval", "--haiku", logs["haiku_fb"],
                           "--template", logs["template"])
    assert code == pr.EXIT_REFUSED and "not found" in err and out == ""


def test_refuses_a_non_pilot_log(logs, capsys):
    code, out, err = _main(capsys, "--sonnet", logs["nonpilot"], "--haiku", logs["haiku_fb"],
                           "--template", logs["template"])
    assert code == 3 and "not the dev pilot" in err and out == ""


def test_refuses_a_status_error_log(logs, capsys, tmp_path):
    bad = _rewrite(tmp_path, logs["sonnet"], "error.eval", lambda log: setattr(log, "status", "error"))
    code, out, err = _main(capsys, "--sonnet", bad, "--haiku", logs["haiku_fb"], "--template", logs["template"])
    assert code == 3 and "status is 'error'" in err and out == ""


def test_refuses_mismatched_dataset_hashes(logs, capsys, tmp_path):
    def mutate(log):
        log.eval.metadata["dataset_sha256"] = "0" * 64

    bad = _rewrite(tmp_path, logs["haiku_fb"], "hash.eval", mutate)
    code, out, err = _main(capsys, "--sonnet", logs["sonnet"], "--haiku", bad, "--template", logs["template"])
    assert code == 3 and "dataset content" in err and out == ""


def test_refuses_swapped_roles_and_half_a_pair(logs, capsys):
    code, _, err = _main(capsys, "--sonnet", logs["template"], "--haiku", logs["haiku_fb"], "--template", logs["template"])
    assert code == 3 and "runs task" in err
    code, _, err = _main(capsys, "--sonnet", logs["sonnet"], "--haiku", logs["haiku_fb"], "--template", logs["sonnet"])
    assert code == 3 and "runs task" in err
    code, _, err = _main(capsys, *_base(logs), "--regression-baseline", logs["sonnet"])
    assert code == 3 and "go together" in err


@pytest.mark.parametrize("argv", [["--help"], [], ["--sonnet", "x.eval"], ["--reading-b", "iv"]])
def test_argument_errors_exit_3(argv):
    with pytest.raises(SystemExit) as e:
        pr.main(argv)
    assert e.value.code == pr.EXIT_REFUSED


def test_run_config_check_passes_a_clean_pilot_log(logs):
    """C2: the clean 20-case pilot log passes with 20 samples x 4 epochs and effort high."""
    summary = pr.run_config_check(read_eval_log(logs["sonnet"]), "sonnet")
    assert summary["reasoning_effort"] == "high" and summary["scored_sample_epochs"] == 80


def test_run_config_check_skips_the_template_log(logs):
    """The template baseline has no evaluated model, so check_run_config always refuses
    it (tests/test_run_config.py pins that); the template role is never checked."""
    log = read_eval_log(logs["template"])
    with pytest.raises(roster.RunConfigError):
        roster.check_run_config(log)
    assert pr.run_config_check(log, "template") is None


@pytest.mark.parametrize("role", ["--sonnet", "--haiku"])
def test_refuses_an_overridden_worker_log_naming_the_field(logs, capsys, role):
    argv = dict(zip(_base(logs)[::2], _base(logs)[1::2]))
    argv[role] = logs["effort_low"]
    code, out, err = _main(capsys, *[x for kv in argv.items() for x in kv])
    assert code == pr.EXIT_REFUSED and out == ""
    assert "reasoning_effort" in err and "run config" in err


def test_refuses_an_overridden_regression_log(logs, capsys):
    code, out, err = _main(capsys, *_base(logs), "--regression-baseline", logs["sonnet"],
                           "--regression-candidate", logs["effort_low"])
    assert code == pr.EXIT_REFUSED and out == "" and "reasoning_effort" in err


def test_max_tokens_pin_must_match_the_log(logs, capsys):
    """A pin the runs did not apply is a mismatch (C2): the clean logs set no max_tokens."""
    code, out, err = _main(capsys, *_base(logs), "--max-tokens-pin", "4000")
    assert code == pr.EXIT_REFUSED and "max_tokens" in err


def test_refuses_sonnet_and_haiku_on_another_worker_prompt(logs, capsys, tmp_path):
    def mutate(log):
        log.eval.metadata["worker_prompt_sha256"] = "f" * 64

    bad = _rewrite(tmp_path, logs["haiku_fb"], "prompt.eval", mutate)
    code, out, err = _main(capsys, "--sonnet", logs["sonnet"], "--haiku", bad, "--template", logs["template"])
    assert code == pr.EXIT_REFUSED and out == "" and "worker_prompt_sha256" in err


def test_review_csv_header_matches_the_evidence_template_byte_for_byte(logs, capsys, tmp_path):
    """docs/evidence/pilot-TEMPLATE-review.csv is the header the owner fills in."""
    template = (ROOT / "docs" / "evidence" / "pilot-TEMPLATE-review.csv").read_bytes()
    assert template == (",".join(CSV_COLUMNS) + "\n").encode("utf-8")
    path = tmp_path / "review.csv"
    code, _, err = _main(capsys, *_base(logs), "--review-csv", path)
    assert code == 0, err
    raw = path.read_bytes()
    assert raw.split(b"\n", 1)[0] + b"\n" == template


# --- Difficulty check (b) readings, one epoch at a time -----------------------------------

def _score(value, *, codes=(), reasons=(), review=None, format_break=False):
    meta = {"failure_modes": list(codes), "review_reasons": list(reasons),
            "requires_human_review": bool(reasons) if review is None else review}
    if format_break:
        meta["format_break"] = True
    return NS(value=value, metadata=meta)


def _epoch(scores, *, error=None, epoch=1, sid="dev-hc-001-BTC-M2", kind="edge"):
    return NS(id=sid, epoch=epoch, error=error, scores=scores, metadata={"truth": {"case_kind": kind}})


def _clean():
    return {n: _score(3) for n in pr.ANCHORS}


def test_correct_epoch_fails_no_reading():
    for reading in pr.READINGS:
        assert not pr.epoch_fail(_epoch(_clean()), reading)


def test_format_break_epoch_fails_reading_i_only():
    fb = {n: _score(0, codes=["FM-000"], reasons=["format break: not one JSON object"], format_break=True)
          for n in pr.ANCHORS}
    assert [pr.epoch_fail(_epoch(fb), r) for r in pr.READINGS] == [True, False, False]


@pytest.mark.parametrize("asset", ASSETS)
def test_sweep_provisional_zero_fails_reading_i_only(asset):
    """F2-01: a correct answer whose narrative trips the sweep ('the <asset> custody
    wallet') scores a provisional 0 with only a sweep reason."""
    scores = _clean()
    scores["fabrication_anchor"] = _score(0, reasons=[SweepHit(f"{asset} custody wallet", "custodian_vocabulary").reason])
    assert [pr.epoch_fail(_epoch(scores), r) for r in pr.READINGS] == [True, False, False]


@pytest.mark.parametrize("asset", ASSETS)
def test_sweep_route_beside_a_settled_code_still_fails(asset):
    scores = _clean()
    scores["fabrication_anchor"] = _score(0, reasons=[SweepHit(f"{asset} custody wallet", "custodian_vocabulary").reason])
    scores["correctness_anchor"] = _score(0, codes=["FM-003"])
    assert all(pr.epoch_fail(_epoch(scores), r) for r in pr.READINGS)


def test_code_review_route_fails_i_and_ii_not_iii():
    scores = _clean()
    scores["reconciling_items_anchor"] = _score(1, reasons=["reconciling_items[0] type 'x' differs from 'y'"])
    assert [pr.epoch_fail(_epoch(scores), r) for r in pr.READINGS] == [True, True, False]


def test_settled_low_anchor_fails_every_reading():
    scores = _clean()
    scores["completeness_anchor"] = _score(2)
    assert all(pr.epoch_fail(_epoch(scores), r) for r in pr.READINGS)


def test_errored_epoch_fails_reading_i_only():
    errored = _epoch({}, error=NS(message="boom"))
    assert [pr.epoch_fail(errored, r) for r in pr.READINGS] == [True, False, False]


def test_missing_epoch_counts_under_reading_i_only():
    log = NS(samples=[_epoch(_clean(), epoch=e) for e in (1, 2, 3)])
    b = pr.difficulty_b(log)
    assert b["missing_epochs"] == [["dev-hc-001-BTC-M2", 4]]
    assert b["readings"]["i"]["failing_epochs"] == [["dev-hc-001-BTC-M2", 4]]
    assert b["readings"]["ii"]["verdict"] == b["readings"]["iii"]["verdict"] == "FAIL"


def test_standard_cases_never_count_for_b():
    scores = _clean()
    scores["correctness_anchor"] = _score(0, codes=["FM-003"])
    log = NS(samples=[_epoch(scores, epoch=e, sid="dev-std-002-BTC-M1", kind="standard") for e in (1, 2, 3, 4)])
    assert all(v["verdict"] == "FAIL" for v in pr.difficulty_b(log)["readings"].values())


def test_unknown_reading_raises():
    with pytest.raises(ValueError):
        pr.epoch_fail(_epoch(_clean()), "iv")


# --- Queue and statistics -----------------------------------------------------------------

def test_queue_classes_and_route_keys():
    sweep = SweepHit("BTC custody wallet", "custodian_vocabulary").reason
    scores = _clean()
    scores["fabrication_anchor"] = _score(0, reasons=[sweep])
    scores["faithfulness_anchor"] = _score(0, reasons=[sweep])  # repeats its component's reason
    scores["reconciling_items_anchor"] = _score(1, reasons=["reconciling_items[0] type 'x' differs from 'y'"])
    report = rel.reliability_report([_epoch(scores), _epoch({}, error=NS(message="boom"), epoch=2)], k=2)
    vol = pr.queue_payload(rel.queue_volume(report))
    assert vol[pr.SWEEP_ROUTE] == {"sample_epoch_reason": 1, "sample_epoch": 1}
    assert vol[pr.CODE_ROUTE] == {"sample_epoch_reason": 1, "sample_epoch": 1}
    assert vol[pr.SAMPLE_ERROR] == {"sample_epoch_reason": 1, "sample_epoch": 1}
    assert vol["all"] == {"sample_epoch_reason": 3, "sample_epoch": 2}
    rows = pr.review_rows(report, "sonnet", "abc")
    assert len(rows) == 3
    assert all(tuple(r) == pr.REVIEW_CSV_COLUMNS for r in rows)
    sweep_row = next(r for r in rows if r["route_key"].startswith(pr.SWEEP_ROUTE))
    assert sweep_row["scorers"] == "fabrication_anchor;faithfulness_anchor"
    assert (sweep_row["row_kind"], sweep_row["route_class"]) == ("finding", pr.SWEEP_ROUTE)
    code_row = next(r for r in rows if r["route_key"].startswith(pr.CODE_ROUTE))
    # This reason lacks the item reference scoring writes, so regression.route_key keeps it
    # whole and only the settlement catch-all takes it: unsettleable, not a finding.
    assert (code_row["row_kind"], code_row["route_class"]) == ("", pr.CODE_ROUTE)
    s14 = rel.reliability_report([_epoch({**_clean(), "reconciling_items_anchor": _score(
        1, reasons=["reconciling_items[0] 'TX-1' type 'x' differs from 'y'"])})], k=1)
    (s14_row,) = pr.review_rows(s14, "sonnet", "abc")
    assert (s14_row["row_kind"], s14_row["route_class"]) == ("finding", pr.CODE_ROUTE)  # S14, settleable
    error_row = next(r for r in rows if r["epoch"] == "2")
    assert error_row["route_key"] == pr.SAMPLE_ERROR
    assert (error_row["row_kind"], error_row["route_class"]) == ("", pr.SAMPLE_ERROR)


def test_unsettleable_route_is_not_a_finding_row():
    """An unsettleable key (S20, non_latin_characters) keeps its queue row with row_kind blank:
    it is not a finding and fails on code."""
    scores = _clean()
    scores["correctness_anchor"] = _score(0, reasons=["non_latin_characters: ж"])
    report = rel.reliability_report([_epoch(scores)], k=1)
    (row,) = pr.review_rows(report, "sonnet", "abc")
    assert row["route_key"] == f"{pr.CODE_ROUTE}: non_latin_characters"
    assert (row["row_kind"], row["route_class"]) == ("", pr.CODE_ROUTE)


def test_fm_listed_reason_gets_its_own_route_key():
    from icfr_agent_evals.scoring import FABRICATION_REVIEW

    scores = _clean()
    scores["fabrication_anchor"] = _score(0, codes=["FM-005"], reasons=[FABRICATION_REVIEW])
    report = rel.reliability_report([_epoch(scores)], k=1)
    assert pr.queue_payload(rel.queue_volume(report))[pr.FM_LISTED] == {"sample_epoch_reason": 1, "sample_epoch": 1}
    (row,) = pr.review_rows(report, "haiku", "abc")
    assert row["route_key"] == f"{pr.FM_LISTED}: {FABRICATION_REVIEW}"
    assert (row["row_kind"], row["route_class"]) == ("", pr.FM_LISTED)  # rides with FM-005; not a finding


def test_statistics_helpers():
    assert pr.mcnemar_block_floor() == 7 and pr.mcnemar_block_floor(reversals=1) == 10
    assert pr.paired_t_mdd(0.5) == pytest.approx(0.390, abs=0.001)
    assert pr.paired_t_mdd(1.0) == pytest.approx(0.780, abs=0.001)
    lo, hi = pr.clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20), abs=1e-6)
    lo, hi = pr.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-6)
    assert pr.t_ppf(0.975, 19) == pytest.approx(2.093, abs=0.001)
