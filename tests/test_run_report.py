"""scripts/run_report.py and the carried-epoch path of icfr_agent_evals/trials.py.

Pre-registration section 8 (Review record and freeze, Second look on review verdicts, The
run report), section 6 (errored and unscored epochs fail whatever review finds) and the
section 13 row on tooling built after the tag, item (1) (decision D-R19): the run report
reads an evaluation run's logs, carries an errored epoch as a failed trial and computes no
review-settled or audit-adjusted figure until both the first-freeze and the second-look
freeze manifests check.

Hand-built logs only: one small mockllm run of three dev pilot cases gives a real log
header and real samples, copied into mock logs whose header claims the evaluation split
and whose sample ids are made up (eval-mock-001 and so on). No evaluation case is built,
read or scored, no API key is read and no model is called. The review records are built
by scripts/review_packet.py and scripts/review_freeze.py from those mock logs, every
verdict row synthetic (reviewer SYNTHETIC-TEST), and frozen in scratch git repositories
under tmp_path; nothing touches this repository's history. The package-hash checks are
stubbed, as in tests/test_pilot_report.py, since a scratch repository holds no package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalError, read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score
from test_trials import Record, record_for, sample

from icfr_agent_evals import review_schema as rs
from icfr_agent_evals import h3, roster, trials
from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.task import variance_commentary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_report.py"
ROLES = {"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"}
STD, EDGE, ADV = "dev-std-002-BTC-M1", "dev-hc-005-USDT-M3", "dev-hc-002-USDT-M2"
SCORERS = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "faithfulness_anchor",
           "completeness_anchor")
HAIKU, SONNET5, OPUS = roster.BASELINES
SONNET55 = roster.WORKER
REVIEWER = "SYNTHETIC-TEST"
NOTE = "synthetic test row, not a review"
SEED = 20261114
RUN_ID = "mock-eval-run"
AMOUNT = "amount_role_unresolved: $5,000.00"
S1 = "Reserves fell short of ledger liabilities by $5,000.00 at the cutoff."
NOW = datetime.now(timezone.utc).replace(microsecond=0)
DAY = NOW.date().isoformat()
LATER = NOW + timedelta(days=3)
LATER_DAY = LATER.date().isoformat()
SITTING = ((NOW - timedelta(hours=3)).isoformat(), (NOW - timedelta(hours=2)).isoformat())
CO = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
ABSENT = object()


def _load():
    spec = importlib.util.spec_from_file_location("run_report_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load()
pr = rr.load_pilot_report()
rf = rr.load_review_freeze()
rp = rr.load_review_packet()
REAL_COMPOSITION_CHECK = rp.check_eval_composition
REAL_TEMPLATE_CHECK = rp.check_eval_template


@pytest.fixture(scope="module", autouse=True)
def _small_eval_logs_allowed():
    """The mock evaluation-split logs here hold 3 or 55 cases, not the 200 of section 4, so
    the composition check is set aside for them; the composition tests below put the real
    check back (REAL_COMPOSITION_CHECK). Most packets here hold worker runs only, so the
    template check of decision D-R20 is set aside too; the template tests put it back
    (REAL_TEMPLATE_CHECK)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(rp, "check_eval_composition", lambda log, name: None)
    mp.setattr(rp, "check_eval_template", lambda runs, split: None)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _package_checks_pass(monkeypatch):
    """A scratch repository holds no package, so the section 8 package-hash checks are
    stubbed to pass (tests/test_pilot_report.py does the same);
    test_first_record_check_runs_the_package_checks shows they are called."""
    monkeypatch.setattr(pr, "package_code_gaps", lambda repo, commit: [])
    monkeypatch.setattr(pr, "running_package_gap", lambda repo: [])


# --- Mock logs ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_log(tmp_path_factory):
    log = inspect_eval(
        variance_commentary(split="dev", subset="pilot", epochs=1),
        model=get_model("mockllm/model"), model_roles=ROLES,
        log_dir=str(tmp_path_factory.mktemp("base")), display="none", sample_id=[STD, EDGE, ADV],
    )[0]
    assert log.status == "success", log.error
    return read_eval_log(log.location)


def answer(truth, narrative: str) -> str:
    items = truth.eligible_items()
    return json.dumps({
        "asset": truth.asset, "period": truth.period, "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction, "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in items)],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in items],
        "narrative": narrative,
    })


def scores(*, routed=False, values=None, fms=None):
    out = {}
    for name in SCORERS:
        reasons = [AMOUNT] if routed and name == "correctness_anchor" else []
        codes = list((fms or {}).get(name, ()))
        out[name] = Score(value=(values or {}).get(name, 0 if codes else 3), metadata={
            "failure_modes": codes, "requires_human_review": bool(reasons), "review_reasons": reasons,
            "details": {"structured_points": 3} if name == "correctness_anchor" else {}})
    return out


def make_log(base, path: Path, model: str, cases: dict[str, str], plan, *, split="eval", flag=True,
             task=None, epochs=4, mutate=None) -> str:
    """A mock log: cases maps a new case id to its base dev case; plan(case, epoch) gives
    (narrative, scores kwargs, error) per epoch."""
    log = base.model_copy(deep=True)
    log.eval.model = model
    log.eval.eval_id = uuid.uuid4().hex[:22]
    if task is not None:
        log.eval.task = task
    by_case = {str(s.id): s for s in base.samples}
    samples = []
    for cid, base_id in cases.items():
        truth = truth_of(by_case[base_id].metadata)
        for ep in range(1, epochs + 1):
            narrative, kw, error = plan(cid, ep)
            s = by_case[base_id].model_copy(deep=True)
            s.id, s.epoch, s.uuid = cid, ep, uuid.uuid4().hex
            s.output = ModelOutput.from_content(model=model, content=answer(truth, narrative))
            s.scores = scores(**kw) if error is None else None  # an errored sample carries no score
            s.error = None if error is None else EvalError(message=error, traceback="", traceback_ansi="")
            samples.append(s)
    log.samples = samples
    log.eval.config.epochs = epochs
    log.eval.config.sample_id = None
    log.eval.config.limit = None
    args = {**(log.eval.task_args or {}), "epochs": epochs, "split": split}
    if flag is ABSENT:
        args.pop("allow_evaluation_split", None)
    else:
        args["allow_evaluation_split"] = flag
    if split == "dev":
        args.pop("allow_evaluation_split", None)
    log.eval.task_args = args
    log.eval.metadata = {**(log.eval.metadata or {}), "split": split}
    log.eval.dataset.samples = len(cases)
    log.eval.dataset.sample_ids = sorted(cases)
    if mutate:
        mutate(log)
    write_eval_log(log, str(path))
    return str(path)


def cases_of(n_std: int, n_edge: int, n_adv: int, prefix: str = "eval-mock") -> dict[str, str]:
    bases = [STD] * n_std + [EDGE] * n_edge + [ADV] * n_adv
    return {f"{prefix}-{i:03d}": b for i, b in enumerate(bases, start=1)}


def dev_cases() -> dict[str, str]:
    return {STD: STD, EDGE: EDGE, ADV: ADV}


def plan_for(tag: str, *, routed_epochs=(1,), error_at=(), low_at=()):
    """Each narrative names the worker's tag, so no narrative is shared between workers."""
    def plan(cid, ep):
        narrative = f"{S1} The {tag} desk confirmed the wallet balances for case {cid} in pass {ep}."
        values = {"correctness_anchor": 2} if (cid, ep) in low_at else None
        error = "mock sample error" if (cid, ep) in error_at else None
        return narrative, {"routed": ep in routed_epochs, "values": values}, error
    return plan


def items(paths: dict[str, str], template: str | None = None):
    workers = [rr.LogInput(label, p, read_eval_log(p), "worker") for label, p in paths.items()]
    tmpl = rr.LogInput("template", template, read_eval_log(template), "template") if template else None
    return workers, tmpl


# --- Scratch git repositories and the review pipeline --------------------------------------------

def git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when.isoformat()
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True,
                          env=env).stdout.strip()


def init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    for k, v in (("user.name", "Test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false"),
                 ("core.autocrlf", "false")):
        git(repo, "config", k, v)
    (repo / "README.md").write_text("start\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "start")


def first_rel() -> str:
    return f"docs/evidence/reviews/{RUN_ID}-manifest.json"


def second_rel() -> str:
    return f"docs/evidence/reviews/{RUN_ID}-second-look-manifest.json"


def fill_first(packet: Path, out: Path) -> Path:
    rows = rs.read_rb7_csv(packet / rs.PACKET_FILES["blank"])
    defaults = {"finding": "correct", "narrative": "none_wrong", "fm_audit": "real_error"}
    for row in rows:
        row.update({"verdict": defaults[row["row_kind"]], "note": NOTE, "reviewer": REVIEWER, "date": DAY})
    rs.write_rb7_csv(out, rows)
    return out


def fill_second(d2: Path, out: Path, flip_run: str | None = None) -> Path:
    """Every second verdict agrees with the first, except one first-correct narrative unit
    of flip_run, ruled whole-narrative wrong."""
    rows = rs.read_rb7_csv(d2 / rs.PACKET_FILES["blank"])
    sl_map = {r["sl_unit_id"]: r for r in rs.read_csv(d2 / rf.SECOND_LOOK_MAP_FILE, rs.SECOND_LOOK_MAP_COLUMNS)}
    flipped = False
    for row in rows:
        row["verdict"] = "correct" if row["row_kind"] == "finding" else "none_wrong"
        m = sl_map[row["unit_id"]]
        if (flip_run and not flipped and row["row_kind"] == "narrative" and m["run_id"] == flip_run
                and m["first_class"] == "correct"):
            row["verdict"], row["sentence"], row["reason"] = "wrong", "1", "amount"
            flipped = True
        row.update({"note": NOTE, "reviewer": REVIEWER, "date": LATER_DAY})
    assert flip_run is None or flipped
    rs.write_rb7_csv(out, rows)
    return out


def freeze_first(root: Path, log_paths: dict[str, str]) -> SimpleNamespace:
    """Packet, frozen first record and its manifest commit in a scratch repository."""
    repo, packet = root / "repo", root / "packet"
    init_repo(repo)
    runs = rp.load_runs(list(log_paths.items()), SEED)
    rp.write_packet(rp.build_packet(runs, run_id=RUN_ID, seed=SEED), str(packet))
    filled = fill_first(packet, root / "filled.csv")
    rf.freeze(str(packet), str(filled), list(log_paths.items()), reviewer=REVIEWER, freeze_date=DAY,
              run_commit="a783322", sittings=[SITTING], manifest_dir=str(repo / "docs/evidence/reviews"))
    git(repo, "add", first_rel())
    git(repo, "commit", "-q", "-m", f"Freeze review record\n\nReviewed-by: {REVIEWER}, {DAY}\n{CO}\n")
    return SimpleNamespace(root=root, repo=repo, packet=packet, first=git(repo, "rev-parse", "HEAD"),
                           d2=root / "second-look", d3=root / "blind-check", logs=log_paths)


def draw_second(env: SimpleNamespace, models: str | None = None) -> None:
    mp = pytest.MonkeyPatch()
    mp.setattr(rf, "_utcnow", lambda: LATER)
    try:
        assert rf.main(["second-look-draw", "--packet-dir", str(env.packet), "--freeze-commit", env.first,
                        "--repo", str(env.repo), "--out", str(env.d2)]) == 0
        if models:
            assert rf.main(["blind-check-draw", "--packet-dir", str(env.packet), "--freeze-commit", env.first,
                            "--repo", str(env.repo), "--models", models, "--out", str(env.d3)]) == 0
            meta = json.loads((env.d3 / rf.BLIND_CHECK_META).read_text(encoding="utf-8"))
            sheet = rs.read_csv(env.d3 / rf.BLIND_CHECK_CSV, rf.BLIND_CHECK_COLUMNS)
            for row in sheet:
                row["guess"] = meta["choices"][0]
            answers = env.root / "blind-answers.csv"
            rs.write_csv(answers, rf.BLIND_CHECK_COLUMNS, sheet)
            assert rf.main(["blind-check-score", "--packet-dir", str(env.packet), "--blind-check-dir", str(env.d3),
                            "--filled", str(answers)]) == 0
    finally:
        mp.undo()


def score_second(env: SimpleNamespace, flip_run: str | None = None) -> None:
    filled = fill_second(env.d2, env.root / "second-filled.csv", flip_run)
    argv = ["second-look-score", "--packet-dir", str(env.packet), "--second-look-dir", str(env.d2),
            "--filled", str(filled), "--reviewer", REVIEWER, "--date", LATER_DAY,
            "--manifest-dir", str(env.repo / "docs/evidence/reviews"), "--repo", str(env.repo)]
    if env.d3.exists():
        argv += ["--blind-check-dir", str(env.d3)]
    mp = pytest.MonkeyPatch()
    mp.setattr(rf, "_utcnow", lambda: LATER)
    try:
        assert rf.main(argv) == 0
    finally:
        mp.undo()


def commit_second(env: SimpleNamespace, message: str | None = None, extra: str | None = None) -> str:
    git(env.repo, "add", second_rel())
    if extra:
        (env.repo / extra).write_text("another file\n", encoding="utf-8")
        git(env.repo, "add", extra)
    msg = message if message is not None else f"Freeze the second look\n\nReviewed-by: {REVIEWER}, {LATER_DAY}\n"
    git(env.repo, "commit", "-q", "-m", msg, when=LATER)
    return git(env.repo, "rev-parse", "HEAD")


def copy_env(env: SimpleNamespace, dest: Path) -> SimpleNamespace:
    shutil.copytree(env.root, dest)
    return SimpleNamespace(root=dest, repo=dest / "repo", packet=dest / "packet", first=env.first,
                           d2=dest / "second-look", d3=dest / "blind-check", logs=env.logs)


def report(env: SimpleNamespace, workers, template, **kw):
    args = dict(packet_dir=str(env.packet), second_look_dir=str(env.d2), repo=str(env.repo))
    if env.d3.exists():
        args["blind_check_dir"] = str(env.d3)
    args.update(kw)
    return rr.build_report(workers, template, **args)


# --- The small run: two workers, an errored epoch, three cases ----------------------------------

SMALL = cases_of(1, 1, 1)
ERRORED = ("eval-mock-002", 2)  # the edge case, epoch 2, of worker alpha


@pytest.fixture(scope="module")
def small_logs(base_log, tmp_path_factory):
    d = tmp_path_factory.mktemp("small-logs")
    return {
        "alpha": make_log(base_log, d / "alpha.eval", HAIKU, SMALL, plan_for("north", error_at={ERRORED})),
        "beta": make_log(base_log, d / "beta.eval", SONNET55, SMALL, plan_for("south")),
        "template": make_log(base_log, d / "template.eval", "none/none", SMALL, plan_for("west", routed_epochs=()),
                             task="icfr_agent_evals/variance_commentary_template_baseline"),
    }


@pytest.fixture(scope="module")
def small_scored(small_logs, tmp_path_factory):
    """First freeze committed, second look drawn and scored, its manifest written but not
    committed: each test copies this state and finishes it its own way."""
    env = freeze_first(tmp_path_factory.mktemp("small") / "run", {k: small_logs[k] for k in ("alpha", "beta")})
    draw_second(env)
    score_second(env)
    return env


@pytest.fixture
def small(small_scored, tmp_path):
    return copy_env(small_scored, tmp_path / "run")


def small_items(small_logs):
    return items({k: small_logs[k] for k in ("alpha", "beta")}, small_logs["template"])


def no_record_figures(monkeypatch):
    """Any call of a review-settled or audit-adjusted computation fails the test."""
    def boom(*a, **kw):
        raise AssertionError("a review-settled or audit-adjusted figure was computed on the withheld path")

    for name in ("review_settled_trial", "audit_adjusted_trial", "audited_code_counts", "_reviewed"):
        monkeypatch.setattr(rr.trials, name, boom)
    monkeypatch.setattr(rr.h3, "h3_from_reports", boom)
    monkeypatch.setattr(rr, "record_block", boom)


NAMES_NEVER_SHOWN = ("alpha", "beta", "gamma", HAIKU, SONNET5, SONNET55, OPUS)


def assert_withheld(text, payload, why):
    """The withheld path on the evaluation split (section 9): the reasons and counts over all
    logs together only, with no model name, no run label and no per-model rate or score."""
    assert payload["withheld"] is True
    assert "Review-settled and audit-adjusted figures: withheld" in text
    assert any(why in r for r in payload["withheld_reasons"]), payload["withheld_reasons"]
    shown = text.split("== Not available", 1)[0]  # the not-available list names items, never a figure
    for kind in ("review_settled", "audit_adjusted", "route_blind", "code_only"):
        assert f"{rr.FIGURE_NAMES[kind]} content-pass^4" not in shown
    assert "anchor means" not in shown and "second-look gate" not in shown and "== H3" not in shown
    assert "citation-pass^4" not in shown and "routing rate" not in shown and "== Worker" not in shown
    assert "Template baseline" not in shown
    assert payload["workers"] == {} and payload["template"] is None
    t = payload["totals"]
    assert t["epochs"] == 12 * t["logs"] and t["errored_epochs"] == 1 and t["carried_epochs"] == 1
    # decision D-R20: the errored epoch is a format break, counted beside the parse failures
    assert t["format_break_epochs"] == 1 and t["format_break_errored_epochs"] == 1 and t["unscored_epochs"] == 0
    assert "Counts only" in text and f"over all {t['logs']} log(s) together, {t['epochs']} epochs; errored 1" in text
    assert ("format breaks (FM-000) 1, of them errored epochs 1 (a sample error rather than a parse failure, "
            "decision D-R20); unscored epochs 0") in text
    body = json.dumps(payload, default=str).lower()
    for name in NAMES_NEVER_SHOWN:
        assert name.lower() not in text.lower() and name.lower() not in body, name


def test_full_path_with_both_checks_passing(small, small_logs):
    commit_second(small)
    workers, template = small_items(small_logs)
    text, payload = report(small, workers, template,
                           args=SimpleNamespace(first_archive="https://web.archive.org/web/1/x", first_archive_date=DAY))
    assert payload["withheld"] is False, payload["withheld_reasons"]
    assert payload["split"] == "eval" and "REHEARSAL" not in text
    alpha = payload["workers"]["alpha"]["record"]
    for kind in ("review_settled", "audit_adjusted"):
        assert set(alpha["figures"][kind]) == {"all", "synthetic", "edge", "adversarial", "difficulty_bearing"}
    # the errored epoch fails its case on every figure; review never read it
    rs_alpha = alpha["figures"]["review_settled"]
    assert rs_alpha["edge"]["passed"] == 0 and rs_alpha["synthetic"]["passed"] == 1
    assert payload["workers"]["beta"]["record"]["figures"]["review_settled"]["all"]["passed"] == 3
    assert "before_d_r9" in rs_alpha["all"]
    assert payload["workers"]["alpha"]["carried_in_packet"] == 1
    assert payload["workers"]["alpha"]["log"]["epochs"]["all"]["carried_epochs"] == 1
    assert payload["workers"]["beta"]["carried_in_packet"] == 0
    for section in ("review-settled content-pass^4", "audit-adjusted content-pass^4", "anchor means",
                    "review queue", "verdict distribution by route class", "second-look gate", "review hours",
                    "bootstrap seed", "blinding check: not run", "first freeze: commit", "== H3 (section 2)",
                    "H3 not computed", "Not available in this repository yet", "route judge",
                    "judge-gate results", "Template baseline (reference row", "RUN REPORT: DONE"):
        assert section in text, section
    assert "archive https://web.archive.org/web/1/x" in text
    assert payload["h3"]["computed"] is False and "four roster workers" in payload["h3"]["reason"]
    # the errored epoch carries no score: its anchors count 0 in the review-settled means
    assert alpha["anchor_means"]["review_settled"]["edge"]["correctness_anchor"]["mean"] == 2.25
    # decision D-R20: the errored epoch is a format break (FM-000) in every FM count and in
    # the format-break rate, with the errored count beside FM-000; its audit row is required
    epochs = payload["workers"]["alpha"]["log"]["epochs"]
    for subset in ("all", "edge", "difficulty_bearing"):
        assert epochs[subset]["fm_codes"] == {"FM-000": 1}
        assert epochs[subset]["format_break_epochs"] == epochs[subset]["format_break_errored_epochs"] == 1
    assert epochs["edge"]["format_break_rate"] == 0.25 and epochs["synthetic"]["format_break_epochs"] == 0
    assert payload["workers"]["alpha"]["log"]["fm_counts"] == {"FM-000": 1}
    assert payload["workers"]["beta"]["log"]["fm_counts"] == {}
    assert "code-caught FM counts FM-000 1 (of them errored epochs 1)" in text
    assert "format breaks (FM-000) 1, format-break rate 0.250, of them errored epochs 1" in text
    assert alpha["code_audit_counts"] == {"real_error": {"FM-000": 1}, "extractor_error": {}}
    assert alpha["record_tables"]["edge"]["audit_by_code"] == {"FM-000": {"real_error": 1}}
    assert alpha["figures"]["audit_adjusted"]["edge"]["passed"] == 0
    # decision D-R20: the sensitivity row is printed; this run has no defect epoch
    d = alpha["defect_sensitivity"]
    assert d["defect_epochs"] == 0 and d["label"] == trials.DEFECT_DROPPED_LABEL
    for kind in ("review_settled", "audit_adjusted"):
        for subset, row in d[kind].items():
            assert (row["passed"], row["n"]) == (alpha["figures"][kind][subset]["passed"],
                                                 alpha["figures"][kind][subset]["n"])
            assert row["samples_fewer_than_k"] == row["samples_left_row"] == 0
    assert "sensitivity row (decision D-R20)" in text and "the unsettleable-route sensitivity row" not in text
    assert "the unsettleable-route sensitivity row" not in payload["not_available"]
    sl = payload["second_look"]
    assert sl["bootstrap_seed"] == rf.derived_seed(rs.sha256_file(small.repo / first_rel()), RUN_ID, "bootstrap")
    assert {r["gate"]["result"] for r in sl["runs"].values()} == {"confirmed"}
    for r in sl["runs"].values():
        assert r["N"] == 3 and r["target_n"] == 3 and r["drawn"] == 3
        assert r["narrative_only_wrong_not_second_looked"] == 0
    assert "not second-looked" in text
    files = payload["freezes"]["first"]["private_files"]
    assert set(rs.RECORD_FILES.values()) <= set(files) and set(rs.PACKET_FILES.values()) <= set(files)
    assert payload["freezes"]["second"]["commit"] == git(small.repo, "rev-parse", "HEAD")
    json.dumps(payload, default=str)
    # section 8: each worker's gate result labels its review-settled and audit-adjusted figures
    assert alpha["second_look_label"] == "second look confirmed"
    for kind in ("review_settled", "audit_adjusted"):
        assert {r["second_look"] for r in alpha["figures"][kind].values()} == {"second look confirmed"}
        assert f"{rr.FIGURE_NAMES[kind]} content-pass^4, {rr.FIGURE_ROLES[kind]}; second look confirmed:" in text
    assert "epoch means]); second look confirmed:" in text
    assert "second_look" not in payload["workers"]["alpha"]["log"]["figures"]["route_blind"]["all"]
    # section 8: the archive URL and date go in the run report; the second freeze's are missing
    assert payload["incomplete"] == ["the second freeze commit's archive URL", "the second freeze commit's archive date"]
    assert text.splitlines()[1].startswith("INCOMPLETE: the run report lacks the second freeze commit's archive URL")
    # each log's header revision commit beside the manifest's run commit (a783322 here)
    revs = payload["revision_commits"]
    assert [r["file_name"] for r in revs] == ["alpha.eval", "beta.eval", "template.eval"]
    assert all(r["run_commit"] == "a783322" and r["differs"] is not None for r in revs)
    for r in revs:
        assert r["differs"] == (not r["revision_commit"].startswith("a783322"))
        assert f"log {r['file_name']} header revision commit" in text
    # the template row is not in this record and does not route: no review line for it
    assert payload["template"]["record"] is None and "not_review_settled" not in payload["template"]
    # section 8 primary outcomes this report does not print are listed, with the reason
    for item in ("the sample-level FM-007 count", "Sample Consistency and the H4 line (section 2)",
                 "the H1 verdict, and the H1 means under the rule before D-R9", "the H2 count (section 2)",
                 "disagreement rate and resolution rate"):
        assert item in payload["not_available"] and f"  {item}: not available; " in text


def test_full_path_with_every_archive_given_is_complete(small, small_logs):
    commit_second(small)
    args = SimpleNamespace(first_archive="https://web.archive.org/web/1/x", first_archive_date=DAY,
                           second_archive="https://web.archive.org/web/2/y", second_archive_date=LATER_DAY)
    text, payload = report(small, *small_items(small_logs), args=args)
    assert payload["withheld"] is False and payload["incomplete"] == [] and "the run report lacks" not in text
    assert "archive https://web.archive.org/web/2/y" in text
    # decision D-R20: this record does not name the template run, so the report says so
    assert payload["incomplete_template"] == rr.TEMPLATE_NOT_IN_RECORD
    assert [line for line in text.splitlines() if line.startswith("INCOMPLETE")] == [
        f"INCOMPLETE: {rr.TEMPLATE_NOT_IN_RECORD}"]


def test_cli_prints_and_writes_json(small, small_logs, capsys, tmp_path):
    commit_second(small)
    out = tmp_path / "report.json"
    code = rr.main(["--worker", f"alpha={small_logs['alpha']}", "--worker", f"beta={small_logs['beta']}",
                    "--template", small_logs["template"], "--packet-dir", str(small.packet),
                    "--second-look-dir", str(small.d2), "--repo", str(small.repo), "--json", str(out)])
    stdout, err = capsys.readouterr()
    assert code == 0, err
    assert stdout.startswith("RUN REPORT: split eval") and "RUN REPORT: DONE" in stdout
    assert json.loads(out.read_text(encoding="utf-8"))["withheld"] is False


def test_withheld_when_no_second_look_is_given(small, small_logs, monkeypatch):
    commit_second(small)
    no_record_figures(monkeypatch)
    workers, template = small_items(small_logs)
    text, payload = report(small, workers, template, second_look_dir=None)
    assert_withheld(text, payload, "no second look given")
    # decision D-R20: the record does not name the template run; the withheld report says so
    assert f"INCOMPLETE: {rr.TEMPLATE_NOT_IN_RECORD}" in text.splitlines()


def test_withheld_when_the_second_look_manifest_is_missing(small, small_logs, monkeypatch):
    (small.repo / second_rel()).unlink()
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "is not found: the second look is not done and frozen")


def test_withheld_when_the_second_look_manifest_is_uncommitted(small, small_logs, monkeypatch):
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "it is not frozen")


def test_withheld_when_the_second_look_commit_holds_another_file(small, small_logs, monkeypatch):
    commit_second(small, extra="notes.txt")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "not the second-look manifest alone")


def test_withheld_when_the_second_look_commit_lacks_the_trailer(small, small_logs, monkeypatch):
    commit_second(small, message="Second look\n")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "closing trailer block")


def test_withheld_when_the_second_look_names_another_reviewer(small, small_logs, monkeypatch):
    commit_second(small, message=f"Second look\n\nReviewed-by: Someone Else, {LATER_DAY}\n")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "closing trailer block")


def test_withheld_when_the_second_look_does_not_descend_from_the_first_freeze(small, small_logs, monkeypatch):
    path = small.repo / second_rel()
    data = path.read_bytes()
    path.unlink()
    git(small.repo, "checkout", "-q", "-b", "side", "HEAD~1")  # before the first freeze commit
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    commit_second(small)
    git(small.repo, "checkout", "-q", "main")
    git(small.repo, "merge", "-q", "--no-ff", "-m", "merge", "side")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "is not a descendant of the first freeze commit")


def test_withheld_when_the_second_look_names_another_first_manifest(small, small_logs, monkeypatch):
    path = small.repo / second_rel()
    man = json.loads(path.read_text(encoding="utf-8"))
    man["first_manifest_sha256"] = "0" * 64
    rs.write_lf(path, json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    commit_second(small)
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "names another first manifest")


def test_withheld_when_a_second_look_file_was_changed(small, small_logs, monkeypatch):
    commit_second(small)
    result = small.d2 / rf.SECOND_LOOK_RESULT_FILE
    result.write_bytes(result.read_bytes() + b" ")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "does not hash to the second-look manifest")


def test_withheld_when_the_first_record_fails_its_check(small, small_logs, monkeypatch):
    commit_second(small)
    verdicts = small.packet / rs.RECORD_FILES["verdicts"]
    verdicts.write_bytes(verdicts.read_bytes() + b"\n")
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert_withheld(text, payload, "does not hash to the manifest")


def test_withheld_without_a_record(small_logs, monkeypatch):
    no_record_figures(monkeypatch)
    text, payload = rr.build_report(*small_items(small_logs))
    assert_withheld(text, payload, "no first review record given")
    assert payload["freezes"] is None
    assert "incomplete_template" not in payload  # no record was read, so nothing to name


def test_withheld_when_the_record_does_not_name_a_worker_log(small, small_logs, base_log, tmp_path, monkeypatch):
    commit_second(small)
    other = make_log(base_log, tmp_path / "gamma.eval", SONNET5, SMALL, plan_for("east"))
    workers, template = items({"alpha": small_logs["alpha"], "beta": small_logs["beta"], "gamma": other},
                              small_logs["template"])
    no_record_figures(monkeypatch)
    text, payload = report(small, workers, template)
    # on the evaluation split a gap names the worker log by its place, never by its label
    assert_withheld(text, payload, "names no run with the eval id of worker log 3")


def test_first_record_check_runs_the_package_checks(small, small_logs, monkeypatch):
    commit_second(small)
    calls = []
    monkeypatch.setattr(pr, "package_code_gaps", lambda repo, commit: calls.append(commit) or ["package differs"])
    monkeypatch.setattr(pr, "running_package_gap", lambda repo: calls.append("running") or [])
    no_record_figures(monkeypatch)
    text, payload = report(small, *small_items(small_logs))
    assert calls == ["running", "a783322"]
    assert_withheld(text, payload, "package differs")


def test_a_record_that_does_not_cover_a_log_is_refused(small, small_logs, monkeypatch):
    commit_second(small)

    def short(log, record, **kw):
        raise trials.RecordError(["eval-mock-001 epoch 1: no whole-narrative member"])

    monkeypatch.setattr(rr.trials, "review_settled_trial", short)
    with pytest.raises(rr.Refused, match="does not cover worker log alpha"):
        report(small, *small_items(small_logs))


# --- The second-look freeze check, alone ------------------------------------------------------------

def check(env, **kw):
    return rf.check_second_look_freeze(repo=str(env.repo), first_manifest_path=env.repo / first_rel(),
                                       first_freeze_commit=env.first, second_look_dir=env.d2, **kw)


def test_second_look_check_passes_and_names_the_commit(small):
    sha = commit_second(small)
    man, info, gaps = check(small)
    assert gaps == [] and man["kind"] == "second_look" and info["commit"] == sha
    assert check(small, commit=sha[:9])[2] == []


def test_second_look_check_lists_every_gap(small):
    path = small.repo / second_rel()
    man = json.loads(path.read_text(encoding="utf-8"))
    man.update(kind="review_freeze", reviewer="Someone Else", bootstrap_seed=1, freeze_commit="abcdef1")
    rs.write_lf(path, json.dumps(man, indent=2) + "\n")
    commit_second(small, message="Second look\n")
    _, _, gaps = check(small)
    text = "\n".join(gaps)
    for want in ("kind is 'review_freeze'", "not the first record's reviewer", "--bootstrap-seed 1",
                 "not the first freeze commit", "closing trailer block"):
        assert want in text, want


def test_second_look_check_refuses_another_commit_named(small):
    commit_second(small)
    _, _, gaps = check(small, commit=small.first)
    assert any("is not the commit that froze" in g for g in gaps)


def test_second_look_check_before_any_second_look(small_logs, tmp_path):
    env = freeze_first(tmp_path / "run", {k: small_logs[k] for k in ("alpha", "beta")})
    _, _, gaps = check(env)
    assert any("is not found" in g for g in gaps)


def test_second_look_private_files_need_the_blinding_folder_when_listed(small):
    man = json.loads((small.repo / second_rel()).read_text(encoding="utf-8"))
    man["private_files"]["blind-check/sealed/blind-check-result.json"] = "0" * 64
    gaps = rf.check_second_look_private_files(small.d2, man)
    assert any("give the blinding-check folder" in g for g in gaps)
    del man["private_files"][rf.SECOND_LOOK_RESULT_FILE]
    assert any("does not list private file sealed/second-look-result.json" in g
               for g in rf.check_second_look_private_files(small.d2, man, small.d2))


def test_second_look_check_needs_the_blinding_check_on_a_full_run(small):
    # section 8: "On a full run, at that commit the reviewer names the likely worker model"
    commit_second(small)
    assert check(small)[2] == []
    _, _, gaps = check(small, worker_runs=4)
    assert any("holds 4 worker runs, a full run, but the second-look manifest holds no blinding result" in g
               for g in gaps), gaps
    assert any("lists no blinding-check file" in g for g in gaps), gaps


def test_second_look_check_refuses_a_draw_within_48_hours_of_the_first_freeze(small):
    # section 8: "At least 48 hours after the freeze", read from the hashed draw time
    meta_path = small.d2 / rf.SECOND_LOOK_META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    first_time = datetime.fromisoformat(git(small.repo, "show", "-s", "--format=%cI", small.first))
    meta["draw_time"] = (first_time + timedelta(hours=47)).isoformat()
    rs.write_lf(meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    path = small.repo / second_rel()
    man = json.loads(path.read_text(encoding="utf-8"))
    man["private_files"][rf.SECOND_LOOK_META_FILE] = rs.sha256_file(meta_path)
    rs.write_lf(path, json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    commit_second(small)
    _, _, gaps = check(small)
    assert any("draw less than 48 hours after the first freeze commit" in g for g in gaps), gaps
    assert not any("does not hash" in g for g in gaps), gaps
    # the drawn second look (three days after the freeze) passes: test_second_look_check_passes...


# --- The template baseline: a reference row beside every model (section 6 and section 8) ------------

def test_template_in_the_record_gets_its_review_settled_and_audit_adjusted_figures(small_logs, tmp_path):
    env = freeze_first(tmp_path / "run", {k: small_logs[k] for k in ("alpha", "beta", "template")})
    draw_second(env)
    score_second(env)
    commit_second(env)
    text, payload = report(env, *small_items(small_logs))
    assert payload["withheld"] is False, payload["withheld_reasons"]
    assert "incomplete_template" not in payload and rr.TEMPLATE_NOT_IN_RECORD not in text
    t = payload["template"]["record"]
    assert t is not None and t["second_look_label"] == "reference row, never a worker; not second-looked"
    # decision D-R20: the sensitivity row covers the template row too, in all three readings
    for kind in ("review_settled", "audit_adjusted", "route_blind"):
        assert set(t["defect_sensitivity"][kind]) == {"all", "synthetic", "edge", "adversarial", "difficulty_bearing"}
    for kind in ("review_settled", "audit_adjusted"):
        assert set(t["figures"][kind]) == {"all", "synthetic", "edge", "adversarial", "difficulty_bearing"}
    assert t["figures"]["review_settled"]["all"]["passed"] == 3
    block = text.split("== Template baseline", 1)[1].split("== Run", 1)[0]
    for kind in ("review_settled", "audit_adjusted", "route_blind", "code_only"):
        assert f"{rr.FIGURE_NAMES[kind]} content-pass^4" in block, kind
    assert "anchor means, review-settled beside audit-adjusted" in block
    # never a worker: no second look, no H3 entry, not among the workers
    assert set(payload["workers"]) == {"alpha", "beta"} and set(payload["second_look"]["runs"]) == {
        payload["workers"]["alpha"]["record"]["run_id"], payload["workers"]["beta"]["record"]["run_id"]}


def test_the_freeze_rebuild_needs_the_template_run_on_the_evaluation_split(small_logs, tmp_path, monkeypatch):
    # Decision D-R20 (ruling 11): review_packet refuses an evaluation-split packet without
    # exactly one template run, and the freeze rebuilds the packet through load_runs, so a
    # packet built without the template run cannot be frozen.
    logs = {k: small_logs[k] for k in ("alpha", "beta")}
    root = tmp_path / "run"
    repo, packet = root / "repo", root / "packet"
    init_repo(repo)
    rp.write_packet(rp.build_packet(rp.load_runs(list(logs.items()), SEED), run_id=RUN_ID, seed=SEED), str(packet))
    filled = fill_first(packet, root / "filled.csv")
    monkeypatch.setattr(rp, "check_eval_template", REAL_TEMPLATE_CHECK)
    with pytest.raises(rp.Refused, match="exactly one --run that is the template baseline log"):
        rp.load_runs(list(logs.items()), SEED)
    with pytest.raises(rf.Refused) as e:
        rf.freeze(str(packet), str(filled), list(logs.items()), reviewer=REVIEWER, freeze_date=DAY,
                  run_commit="a783322", sittings=[SITTING], manifest_dir=str(repo / "docs/evidence/reviews"))
    assert any(g.startswith("rederive:") and "template baseline" in g for g in e.value.gaps), e.value.gaps
    # with the template run in the packet the real check passes and the freeze goes through
    env = freeze_first(tmp_path / "ok", {k: small_logs[k] for k in ("alpha", "beta", "template")})
    assert env.first


def test_template_that_routes_outside_the_record_is_not_review_settled(small, small_logs, base_log, tmp_path):
    commit_second(small)
    routed = make_log(base_log, tmp_path / "template.eval", "none/none", SMALL, plan_for("west"),
                      task="icfr_agent_evals/variance_commentary_template_baseline")
    workers, _ = small_items(small_logs)
    template = rr.LogInput("template", routed, read_eval_log(routed), "template")
    text, payload = report(small, workers, template)
    assert payload["withheld"] is False, payload["withheld_reasons"]
    n = payload["template"]["not_review_settled"]
    assert payload["template"]["record"] is None and (n["routed_epochs"], n["findings"], n["epochs"]) == (3, 3, 12)
    assert ("this row is not review-settled; the frozen record does not name the template run, and its log routes "
            "3 epoch(s) with 3 finding(s) over 12 epochs") in text


# --- The defect sensitivity row (decision D-R20) -------------------------------------------------------

S72 = "extractor_error: KeyError"
S17 = "computed explained split disagrees with truth although every eligible item is cited"
NONLATIN = "non_latin_characters: ж"


def with_reasons(targets):
    """A mutate that plants code-review reasons on the correctness scorer of some epochs."""
    def mutate(log):
        for s in log.samples:
            reasons = targets.get((str(s.id), int(s.epoch)))
            if reasons and s.scores:
                meta = s.scores["correctness_anchor"].metadata
                meta["review_reasons"] = [*meta.get("review_reasons", []), *reasons]
                meta["requires_human_review"] = True
    return mutate


def test_sensitivity_row_drops_each_models_defect_epochs(small_logs, base_log, tmp_path):
    # The standard case loses one epoch to an extractor defect (S72) and is scored on 3; every
    # epoch of the adversarial case holds an answer-key defect (S17), so it leaves the row; a
    # non-Latin letter (S20) is unsettleable but no defect, so it is never dropped.
    targets = {("eval-mock-001", 2): [S72], **{("eval-mock-003", e): [S17] for e in range(1, 5)},
               ("eval-mock-002", 3): [NONLATIN]}
    gamma = make_log(base_log, tmp_path / "gamma.eval", SONNET5, SMALL, plan_for("east"), mutate=with_reasons(targets))
    logs = {"alpha": small_logs["alpha"], "gamma": gamma}
    env = freeze_first(tmp_path / "run", logs)
    draw_second(env)
    score_second(env)
    commit_second(env)
    workers, template = items(logs, small_logs["template"])
    text, payload = report(env, workers, template)
    assert payload["withheld"] is False, payload["withheld_reasons"]
    assert sorted(trials.defect_epochs(read_eval_log(gamma))) == [
        ("eval-mock-001", 2), *(("eval-mock-003", e) for e in range(1, 5))]
    g = payload["workers"]["gamma"]["record"]
    assert g["figures"]["review_settled"]["all"]["passed"] == 0  # every case fails on an unsettleable route
    d = g["defect_sensitivity"]
    assert d["defect_epochs"] == 5
    # section 6: every table that reports one of the three figures reports all three, so the
    # sensitivity row gives the route-blind reading too (from the log alone, no second-look label)
    for kind in ("review_settled", "audit_adjusted", "route_blind"):
        row = d[kind]["all"]
        assert (row["passed"], row["n"], row["samples_fewer_than_k"], row["samples_left_row"]) == (1, 2, 1, 1)
        assert row["dropped_epochs"] == 5
        assert row.get("second_look") == (None if kind == "route_blind" else "second look confirmed")
        lo, hi = trials.clopper_pearson(1, 2)
        assert row["ci95"] == [lo, hi] and row["before_d_r9"]["n"] == 2
        assert (d[kind]["synthetic"]["passed"], d[kind]["synthetic"]["n"]) == (1, 1)
        assert d[kind]["synthetic"]["samples_fewer_than_k"] == 1
        assert (d[kind]["edge"]["passed"], d[kind]["edge"]["n"]) == (0, 1)  # S20 stays
        assert (d[kind]["adversarial"]["n"], d[kind]["adversarial"]["samples_left_row"]) == (0, 1)
    # each model for its own defect epochs: alpha has none, so its row is its figure
    a = payload["workers"]["alpha"]["record"]["defect_sensitivity"]
    assert a["defect_epochs"] == 0 and a["review_settled"]["all"]["n"] == 3
    assert a["route_blind"]["all"]["n"] == 3
    block = text.split("== Worker gamma", 1)[1]
    assert (f"  {trials.DEFECT_DROPPED_LABEL}; 5 defect epoch(s) dropped from this run; review-settled and "
            "audit-adjusted second look confirmed:") in block
    for name in ("review-settled", "audit-adjusted", "route-blind"):
        assert f"    {name}, all: 1/2 (Clopper-Pearson 95% [" in block, name
    assert "samples scored on fewer than 4 epochs 1; samples that left the row 1" in block
    # the review-queue table counts its defect epochs from the same source as the row
    tables = g["record_tables"]
    assert tables["all"]["defect_epochs_by_route_key"] == {
        k: n for k, n in Counter(key for keys in trials.defect_epochs(read_eval_log(gamma)).values()
                                 for key in keys).items()}
    assert sum(tables["all"]["defect_epochs_by_route_key"].values()) == d["defect_epochs"]


def test_defect_dropped_rule():
    def ep(sid, e, ok=True):
        return trials.EpochResult(sid, e, ok, () if ok else ("x",), {}, ())

    def rep(*samples):
        return trials.TrialReport(kind=trials.REVIEW_SETTLED, k=4, samples={
            sid: trials.SampleResult(sid, all(e.success for e in eps), "", tuple(eps)) for sid, eps in samples})

    report_ = rep(("a", [ep("a", 1), ep("a", 2, False), ep("a", 3), ep("a", 4)]),
                  ("b", [ep("b", e, e != 1) for e in range(1, 5)]),
                  ("c", [ep("c", e) for e in range(1, 5)]))
    row = trials.defect_dropped(report_, {("a", 2), *(("b", e) for e in range(1, 5))})
    assert row["per_sample"] == {"a": 1, "c": 1} and (row["passed"], row["n"]) == (2, 2)
    assert (row["samples_fewer_than_k"], row["samples_left_row"], row["dropped_epochs"]) == (1, 1, 5)
    assert row["label"] == trials.DEFECT_DROPPED_LABEL and "never the score of record" in row["label"]
    assert trials.defect_dropped(report_, set())["per_sample"] == {"a": 0, "b": 0, "c": 1}


# --- The reasoning mode, read from the model-call requests (section 5) --------------------------------

def _model_event(request, role=None):
    return SimpleNamespace(event="model", role=role, call=SimpleNamespace(request=request))


SENT = {"model": "m", "max_tokens": 48000, "output_config": {"effort": "high"},
        "thinking": {"type": "adaptive", "display": "summarized"}, "messages": []}


def test_reasoning_mode_is_read_from_each_request_as_sent():
    log = SimpleNamespace(plan=SimpleNamespace(config=SimpleNamespace(reasoning_effort="low")), samples=[
        SimpleNamespace(events=[_model_event(SENT), SimpleNamespace(event="score")]),
        SimpleNamespace(events=[_model_event(dict(SENT)), _model_event({"model": "g"}, role="grader")]),
    ])
    want = 'output_config.effort="high", thinking={"display":"summarized","type":"adaptive"}'
    assert rr.reasoning_modes_sent(log) == [want]
    assert rr.reasoning_mode(log) == want  # the plan config's effort is never read
    assert rr.request_mode({"reasoning_effort": "high"}) == 'reasoning_effort="high"'
    assert rr.request_mode({"model": "m"}) == rr.NO_REASONING_FIELD
    assert rr.request_mode(None) == rr.NO_REQUEST


def test_a_worker_log_whose_events_disagree_is_refused(base_log, small_logs, tmp_path, capsys):
    unit = SimpleNamespace(samples=[SimpleNamespace(events=[_model_event(SENT)]),
                                    SimpleNamespace(events=[_model_event({**SENT, "thinking": {"type": "enabled"}})])])
    with pytest.raises(rr.Refused, match="2 different reasoning modes"):
        rr.reasoning_mode(unit, "x.eval")

    def one_thinks(log):
        event = next(e for e in log.samples[0].events if e.event == "model" and e.role is None)
        event.call.request = {**event.call.request, "thinking": {"type": "adaptive", "display": "summarized"}}

    bad = make_log(base_log, tmp_path / "mixed.eval", SONNET5, SMALL, plan_for("east"), mutate=one_thinks)
    assert rr.main(["--worker", f"alpha={bad}", "--template", small_logs["template"]]) == 3
    assert "different reasoning modes" in capsys.readouterr().err


# --- The evaluation split's 200 cases (section 4) -------------------------------------------------

def test_evaluation_logs_without_the_200_cases_are_refused(small_logs, base_log, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "check_eval_composition", REAL_COMPOSITION_CHECK)
    assert rr.main(["--worker", f"alpha={small_logs['alpha']}", "--template", small_logs["template"]]) == 3
    err = capsys.readouterr().err
    assert "does not hold its 200 cases" in err and "it holds 3 cases, not 200" in err
    workers, template = small_items(small_logs)
    with pytest.raises(rr.Refused, match="200 cases"):
        rr.check_inputs(workers[:1], None)
    with pytest.raises(rr.Refused, match="template log: log template.eval is on the evaluation split"):
        rr.check_log(template.log, "template", "template.eval")
    # dev rehearsal logs keep their own counts
    dev = make_log(base_log, tmp_path / "dev.eval", SONNET5, dev_cases(), plan_for("east"), split="dev")
    assert rr.check_inputs(items({"gamma": dev})[0], None)[0] == "dev"


# --- Logs the report refuses ---------------------------------------------------------------------

@pytest.mark.parametrize("flag", [ABSENT, False, "true", 1])
def test_eval_log_without_the_unlock_is_refused(base_log, small_logs, tmp_path, flag, capsys):
    bad = make_log(base_log, tmp_path / "bad.eval", SONNET5, SMALL, plan_for("east"), flag=flag)
    code = rr.main(["--worker", f"alpha={bad}", "--template", small_logs["template"]])
    assert code == 3 and "allow_evaluation_split" in capsys.readouterr().err


def test_mixed_splits_are_refused(base_log, small_logs, tmp_path, capsys):
    dev = make_log(base_log, tmp_path / "dev.eval", SONNET5, dev_cases(), plan_for("east"), split="dev")
    code = rr.main(["--worker", f"alpha={small_logs['alpha']}", "--worker", f"gamma={dev}",
                    "--template", small_logs["template"]])
    assert code == 3 and "different splits" in capsys.readouterr().err


def test_dev_logs_run_as_a_labeled_rehearsal(base_log, tmp_path, monkeypatch):
    dev = make_log(base_log, tmp_path / "dev.eval", SONNET5, dev_cases(), plan_for("east"), split="dev")
    tmpl = make_log(base_log, tmp_path / "tmpl.eval", "none/none", dev_cases(), plan_for("west", routed_epochs=()),
                    split="dev", task="icfr_agent_evals/variance_commentary_template_baseline")
    no_record_figures(monkeypatch)
    text, payload = rr.build_report(*items({"gamma": dev}, tmpl))
    assert payload["rehearsal"] is True and text.startswith(f"RUN REPORT ({rr.REHEARSAL})")
    # a dev rehearsal keeps the full block of figures that read the log alone
    assert payload["withheld"] is True and "totals" not in payload
    assert payload["workers"]["gamma"]["log"]["figures"]["route_blind"]["all"]["n"] == 3
    for section in ("== Worker gamma", "route-blind content-pass^4", "code-only content-pass^4", "citation-pass^4",
                    "routing rate", "Template baseline (reference row"):
        assert section in text, section
    assert "INCOMPLETE" not in text


def test_worker_log_off_the_run_configuration_is_refused(base_log, small_logs, tmp_path, capsys):
    def low(log):
        log.plan.config.reasoning_effort = "low"

    bad = make_log(base_log, tmp_path / "low.eval", SONNET5, SMALL, plan_for("east"), mutate=low)
    assert rr.main(["--worker", f"alpha={bad}", "--template", small_logs["template"]]) == 3
    assert "reasoning_effort" in capsys.readouterr().err


def test_k_other_than_four_is_refused(base_log, small_logs, tmp_path, capsys):
    bad = make_log(base_log, tmp_path / "k2.eval", SONNET5, SMALL, plan_for("east"), epochs=2)
    assert rr.main(["--worker", f"alpha={bad}", "--template", small_logs["template"]]) == 3
    assert "k = 4" in capsys.readouterr().err


def test_template_and_worker_roles_are_checked(small_logs, capsys):
    assert rr.main(["--worker", f"alpha={small_logs['alpha']}", "--template", small_logs["beta"]]) == 3
    assert "template log" in capsys.readouterr().err
    assert rr.main(["--worker", f"alpha={small_logs['template']}", "--template", small_logs["template"]]) == 3


def test_two_worker_logs_of_one_model_are_refused(base_log, small_logs, tmp_path, capsys):
    twin = make_log(base_log, tmp_path / "twin.eval", HAIKU, SMALL, plan_for("east"))
    assert rr.main(["--worker", f"alpha={small_logs['alpha']}", "--worker", f"gamma={twin}",
                    "--template", small_logs["template"]]) == 3
    assert "two worker logs of one model" in capsys.readouterr().err


def test_errored_epoch_passes_the_run_configuration_check(small_logs):
    workers, _ = items({"alpha": small_logs["alpha"]})
    split, _ = rr.check_inputs(workers, None)
    assert split == "eval" and workers[0].run_config["scored_sample_epochs"] == 11


# --- The full run: four workers, 54 difficulty-bearing cases --------------------------------------

FULL = cases_of(1, 30, 24)
FULL_MODELS = {"alpha": HAIKU, "beta": SONNET5, "gamma": SONNET55, "delta": OPUS}
TAGS = {"alpha": "north", "beta": "south", "gamma": "east", "delta": "west"}


@pytest.fixture(scope="module")
def full_logs(base_log, tmp_path_factory):
    d = tmp_path_factory.mktemp("full-logs")
    hard = [cid for cid, b in FULL.items() if b != STD]
    out = {}
    for label, model in FULL_MODELS.items():
        # delta never routes, so its run has no second-look unit (N = 0); alpha scores lower on a third
        routed = () if label == "delta" else (1,)
        low = {(cid, 2) for cid in hard[::3]} if label == "alpha" else set()
        out[label] = make_log(base_log, d / f"{label}.eval", model, FULL,
                              plan_for(TAGS[label], routed_epochs=routed, low_at=low))
    out["template"] = make_log(base_log, d / "template.eval", "none/none", FULL, plan_for("plain", routed_epochs=()),
                               task="icfr_agent_evals/variance_commentary_template_baseline")
    return out


@pytest.fixture(scope="module")
def full_drawn(full_logs, tmp_path_factory):
    env = freeze_first(tmp_path_factory.mktemp("full") / "run", {k: full_logs[k] for k in FULL_MODELS})
    draw_second(env, models=",".join(f"{label}=Model {label.upper()}" for label in FULL_MODELS))
    return env


def full_items(full_logs):
    return items({k: full_logs[k] for k in FULL_MODELS}, full_logs["template"])


def test_h3_is_tested_under_second_look_empty(full_drawn, full_logs, tmp_path):
    env = copy_env(full_drawn, tmp_path / "run")
    score_second(env)
    commit_second(env)
    text, payload = report(env, *full_items(full_logs))
    assert payload["withheld"] is False, payload["withheld_reasons"]
    gates = {label: w["second_look"]["gate"]["result"] for label, w in payload["workers"].items()}
    assert gates == {"alpha": "confirmed", "beta": "confirmed", "gamma": "confirmed", "delta": "second look empty"}
    assert payload["workers"]["delta"]["second_look"]["N"] == 0
    assert payload["workers"]["alpha"]["second_look"]["N"] == 55
    hb = payload["h3"]
    assert hb["computed"] is True
    settled = hb["readings"]["review_settled"]
    assert settled["second_look"]["status"] == "tested" and settled["second_look"]["label"] == "second look empty"
    assert settled["verdict"] == "holds" and settled["n_cases"] == 54
    assert set(hb["readings"]) == {"review_settled", "audit_adjusted", "review_settled_before_d_r9"}
    modes = settled["beside"]["reasoning_modes"]
    # read from each model-call request as sent; the mock requests carry no reasoning field
    assert set(modes.values()) == {rr.NO_REASONING_FIELD}
    assert payload["workers"]["alpha"]["reasoning_modes_sent"] == [rr.NO_REASONING_FIELD]
    # a full run of four worker runs carries its blinding check; the second-look check holds it
    assert payload["second_look"]["blinding"] is not None
    for label, w in payload["workers"].items():
        want = "labeled second look empty, never confirmed" if label == "delta" else "second look confirmed"
        assert w["record"]["second_look_label"] == want
        assert {r["second_look"] for r in w["record"]["figures"]["review_settled"].values()} == {want}
    assert "; labeled second look empty, never confirmed:" in text
    assert "H3 on review-settled anchor scores" in text and "second look empty" in text
    b = payload["second_look"]["blinding"]
    assert b["n"] == 40 and "blinding check: named correctly" in text
    # alpha's lower correctness shows in the review-settled anchor means beside the audit-adjusted ones
    means = payload["workers"]["alpha"]["record"]["anchor_means"]
    assert means["review_settled"]["edge"]["correctness_anchor"]["mean"] < 3.0
    assert payload["workers"]["delta"]["log"]["epochs"]["all"]["routing_rate"] == 0.0


def test_review_not_confirmed_makes_h3_not_tested(full_drawn, full_logs, tmp_path):
    env = copy_env(full_drawn, tmp_path / "run")
    labels = json.loads((env.packet / rs.PACKET_FILES["runs"]).read_text(encoding="utf-8"))
    beta = next(rid for rid, info in labels.items() if info["label"] == "beta")
    score_second(env, flip_run=beta)
    commit_second(env)
    text, payload = report(env, *full_items(full_logs))
    assert payload["withheld"] is False, payload["withheld_reasons"]
    assert payload["workers"]["beta"]["second_look"]["gate"]["result"] == "review not confirmed"
    beta_rec = payload["workers"]["beta"]["record"]
    assert beta_rec["second_look_label"] == "labeled review not confirmed"
    assert {r["second_look"] for r in beta_rec["figures"]["audit_adjusted"].values()} == {"labeled review not confirmed"}
    assert "; labeled review not confirmed:" in text
    settled = payload["h3"]["readings"]["review_settled"]
    assert settled["verdict"] == "not tested" and settled["holds"] is None
    assert "Sonnet 5" in settled["second_look"]["reason"] and "review not confirmed" in settled["second_look"]["reason"]
    assert "H3 on review-settled anchor scores, the reading H3 is evaluated on (decision D-A): not tested" in text
    assert f"H3 robustness: {h3.ROBUSTNESS_NOT_GIVEN}" in text
    assert "the three readings give one ordering" not in text


# --- trials.py: carried epochs need no finding row ----------------------------------------------------

def carried_record(samples, entries, *, errored_rows=True, **kw):
    """record_for on the samples the packet reads (carried epochs left out), with the carried
    list. With errored_rows, each errored epoch also gets the rows decision D-R20 asks of a
    format break: its FM-000 audit row and its whole-narrative read."""
    carried = [s for s in samples if trials.carried_reason(trials._collect([s], 1)[1][(str(s.id), int(s.epoch))])]
    rec = record_for([s for s in samples if s not in carried], **kw)
    if errored_rows:
        for s in carried:
            if s.error is not None:
                rec.audit(str(s.id), int(s.epoch), "FM-000")
                rec.narrative(str(s.id), int(s.epoch))
    rec.carried = entries
    return rec


def hard_samples(err_epoch=2, kind="edge", **err_kw):
    out = [sample("h", e, kind=kind) for e in range(1, 5) if e != err_epoch]
    out.append(sample("h", err_epoch, kind=kind, error=SimpleNamespace(message="boom"), **err_kw))
    return sorted(out, key=lambda s: s.epoch)


def entry(sid="h", epoch=2, reason="sample error", run_id="R1"):
    return {"run_id": run_id, "sample_id": sid, "epoch": epoch, "reason": reason}


ANCHOR_NAMES = ("correctness_anchor", "fabrication_anchor", "reconciling_items_anchor", "completeness_anchor")


def test_errored_epoch_is_a_format_break_and_a_failed_trial():
    # Decision D-R20: an errored epoch carries FM-000, fails its trial, scores 0 on every
    # anchor and keeps FM-000 in the audit-adjusted figure whatever the audit rules.
    samples = hard_samples()
    rec = carried_record(samples, [entry()])
    rep = trials.review_settled_trial(samples, rec, k=4)
    e2 = next(e for e in rep.samples["h"].epochs if e.epoch == 2)
    assert not e2.success and e2.reasons == ("sample error", trials.ERRORED_FORMAT_BREAK_REASON,
                                             "FM codes ['FM-000'] (codes final)")
    assert e2.failure_modes == ("FM-000",) and e2.anchors == {a: 0.0 for a in ANCHOR_NAMES}
    assert rep.pass_k == {"h": 0}
    assert all(e.success for e in rep.samples["h"].epochs if e.epoch != 2)
    assert rep.anchor_means()["correctness_anchor"]["mean"] == 2.25
    ruled_out = carried_record(samples, [entry()], errored_rows=False)
    ruled_out.audit("h", 2, "FM-000", verdict="extractor_error")
    ruled_out.narrative("h", 2)
    for record in (rec, ruled_out):
        audit = trials.audit_adjusted_trial(samples, record, k=4)
        a2 = next(e for e in audit.samples["h"].epochs if e.epoch == 2)
        assert audit.pass_k == {"h": 0} and a2.failure_modes == ("FM-000",) and a2.removed_codes == ()
        assert a2.anchors == {a: 0.0 for a in ANCHOR_NAMES}
    for trial in (trials.route_blind_trial, trials.code_only_trial):
        r = trial(samples, k=4)
        assert r.pass_k == {"h": 0}
        assert next(e for e in r.samples["h"].epochs if e.epoch == 2).failure_modes == ("FM-000",)
    blind = trials.route_blind_trial(samples, k=4)
    assert next(e for e in blind.samples["h"].epochs if e.epoch == 2).anchors == {a: 0.0 for a in ANCHOR_NAMES}
    # the code-only figure scores it 0 on every anchor too (decision D-R20)
    code = trials.code_only_trial(samples, k=4)
    assert next(e for e in code.samples["h"].epochs if e.epoch == 2).anchors == {a: 0.0 for a in ANCHOR_NAMES}
    assert code.anchor_means()["correctness_anchor"]["mean"] == 2.25


def test_errored_epoch_needs_its_fm000_audit_row_and_its_narrative_read():
    samples = hard_samples()
    rec = carried_record(samples, [entry()], errored_rows=False)
    with pytest.raises(trials.RecordError) as e:
        trials.review_settled_trial(samples, rec, k=4)
    assert "h epoch 2: no audit row for FM-000" in e.value.gaps
    assert "h epoch 2: no whole-narrative member" in e.value.gaps


def test_epoch_codes_adds_fm000_on_an_errored_epoch_only():
    err = SimpleNamespace(error=SimpleNamespace(message="boom"))
    assert trials.epoch_codes(err, ()) == ("FM-000",)
    assert trials.epoch_codes(err, ("FM-001",)) == ("FM-000", "FM-001")
    assert trials.epoch_codes(err, ("FM-000", "FM-005")) == ("FM-000", "FM-005")
    ok = SimpleNamespace(error=None)
    assert trials.epoch_codes(ok, ()) == () and trials.epoch_codes(ok, ("FM-001",)) == ("FM-001",)


def test_a_record_without_the_carried_list_reads_as_before():
    samples = hard_samples()
    rec = carried_record(samples, None, errored_rows=False)
    del rec.carried
    with pytest.raises(trials.RecordError, match="h epoch 2: no whole-narrative member"):
        trials.review_settled_trial(samples, rec, k=4)
    clean = [sample("c", e) for e in range(1, 5)]
    assert trials.review_settled_trial(clean, record_for(clean), k=4).pass_k == {"c": 1}


@pytest.mark.parametrize("entries,message", [
    ([], "the packet's carried list does not name it"),
    ([entry(), entry(epoch=3)], "the packet carried it as a failed trial (sample error), but the log scores it"),
    ([entry(reason="sample error; correctness_anchor unscored or non-numeric")], "the log gives 'sample error'"),
    ([entry(), entry(sid="zz", epoch=1)], "the packet carried an epoch the log does not hold"),
    ([entry(), entry()], "lists the carried epoch twice"),
    ([{"run_id": "R1"}], "has no readable run, sample and epoch"),
])
def test_carried_list_mismatch_is_a_record_gap(entries, message):
    samples = hard_samples()
    rec = carried_record(samples, entries)
    with pytest.raises(trials.RecordError) as e:
        trials.review_settled_trial(samples, rec, k=4)
    assert any(message in g for g in e.value.gaps), e.value.gaps


def test_carried_list_of_another_run_does_not_count():
    samples = hard_samples()
    rec = carried_record(samples, [entry(run_id="R2")])
    with pytest.raises(trials.RecordError, match="does not name it"):
        trials.review_settled_trial(samples, rec, k=4)


def unscored_samples():
    blank = SimpleNamespace(value=None, metadata={})
    out = [sample("h", e, kind="edge") for e in (1, 3, 4)]
    out.append(sample("h", 2, kind="edge", completeness=blank))
    return sorted(out, key=lambda s: s.epoch)


UNSCORED = "completeness_anchor unscored or non-numeric"


def test_a_narrative_member_on_an_unscored_carried_epoch_is_a_gap():
    # an unscored epoch with no sample error is not a format break (decision D-R20): the
    # packet reads none of it, so a narrative member on it is a gap
    samples = unscored_samples()
    rec = carried_record(samples, [entry(reason=UNSCORED)])
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 0}
    rec.narrative("h", 2)
    with pytest.raises(trials.RecordError, match="reads the narrative of an epoch the packet carried"):
        trials.review_settled_trial(samples, rec, k=4)


FM001 = SimpleNamespace(value=0, metadata={"failure_modes": ["FM-001"], "requires_human_review": False,
                                           "review_reasons": [], "details": {"structured_points": 3}})


def test_an_fm_coded_carried_epoch_needs_its_audit_row_and_its_narrative_read():
    # Section 8: the settlement refuses a record unless every FM-coded (epoch, code) has an
    # audit row; section 6: the audit reads that epoch's narrative whole. A carried epoch is
    # no exception, and it stays a failed trial whatever the rows say. Being errored, it
    # also carries FM-000 (decision D-R20), which the audit never removes (D-R5).
    samples = hard_samples(correctness=FM001)
    rec = carried_record(samples, [entry()], errored_rows=False)
    with pytest.raises(trials.RecordError) as e:
        trials.review_settled_trial(samples, rec, k=4)
    assert "h epoch 2: no audit row for FM-001" in e.value.gaps
    assert "h epoch 2: no audit row for FM-000" in e.value.gaps
    assert "h epoch 2: no whole-narrative member" in e.value.gaps
    rec.audit("h", 2, "FM-001", verdict=None)
    rec.audit("h", 2, "FM-000")
    rec.narrative("h", 2)
    with pytest.raises(trials.RecordError, match="h epoch 2: audit verdict None for FM-001"):
        trials.review_settled_trial(samples, rec, k=4)
    done = carried_record(samples, [entry()], errored_rows=False)
    done.audit("h", 2, "FM-001", verdict="extractor_error")
    done.audit("h", 2, "FM-000", verdict="extractor_error")
    done.narrative("h", 2, "wrong", ("amount",))
    for trial in (trials.review_settled_trial, trials.audit_adjusted_trial):
        rep = trial(samples, done, k=4)
        e2 = next(e for e in rep.samples["h"].epochs if e.epoch == 2)
        assert rep.pass_k == {"h": 0} and not e2.success and e2.reasons[0] == "sample error"
        # decision D-R20: an errored epoch is a format break, so its whole-narrative wrong is
        # recorded and counted as review-found, as on a parse-failure FM-000 epoch; its anchors
        # are 0 and it fails whatever the row says, so no figure moves
        assert "whole-narrative wrong (amount)" in e2.reasons
        assert e2.narrative_wrong and e2.narrative_errors == ("amount",)
        assert e2.anchors == {a: 0.0 for a in ANCHOR_NAMES}
        assert rep.anchor_means()["correctness_anchor"]["mean"] == 2.25
        assert "FM-000" in e2.failure_modes
    audit = trials.audit_adjusted_trial(samples, done, k=4)
    assert next(e for e in audit.samples["h"].epochs if e.epoch == 2).removed_codes == ("FM-001",)


FM000_PARSE = SimpleNamespace(value=0, metadata={"failure_modes": ["FM-000"], "requires_human_review": False,
                                                 "review_reasons": [], "details": {"structured_points": 0}})


def test_whole_narrative_wrong_lines_read_one_source():
    # Decision D-R20: a whole-narrative wrong on an errored FM-000 epoch counts as review-found,
    # the same as on a parse-failure FM-000 epoch, in the review-found line and in the
    # review-queue table alike. On an unscored carried epoch the record settles nothing, so the
    # wrong it holds counts in neither line.
    blank = SimpleNamespace(value=None, metadata={})
    samples = hard_samples(correctness=FM001)  # h epoch 2: errored, FM-001 recorded
    samples += [sample("u", e, kind="edge") for e in (1, 3, 4)]
    samples.append(sample("u", 2, kind="edge", correctness=FM001, completeness=blank))
    samples += [sample("p", e, kind="edge") for e in (1, 2, 4)]
    samples.append(sample("p", 3, kind="edge", correctness=FM000_PARSE))
    unscored = entry("u", 2, reason="completeness_anchor unscored or non-numeric")
    rec = carried_record(samples, [entry(), unscored], errored_rows=False,
                         overrides={("p", 3): ("wrong", ("amount",))})
    for code in ("FM-000", "FM-001"):
        rec.audit("h", 2, code)
    rec.narrative("h", 2, "wrong", ("amount",))
    rec.audit("u", 2, "FM-001")
    rec.narrative("u", 2, "wrong", ("amount",))
    for m in rec.members:
        if m["member_kind"] == "narrative":
            m["route_class"] = rs.NARRATIVE_CLASS_FM_CODED
    settled = trials.review_settled_trial(samples, rec, k=4)
    assert rr.narrative_wrong_epochs(settled) == {("h", 2), ("p", 3)}
    found = rr.review_found_rows(settled)
    assert found["all"]["whole_narrative_wrong_epochs"] == found["edge"]["whole_narrative_wrong_epochs"] == 2
    kinds = {sid: "edge" for sid in settled.samples}
    tables = rr.record_tables(rec, "R1", kinds, defects={}, narrative_wrong=rr.narrative_wrong_epochs(settled))
    assert tables["all"]["whole_narrative_wrong_members_by_class"][rs.NARRATIVE_CLASS_FM_CODED] == 2
    assert tables["all"]["whole_narrative_wrong_units_by_class"][rs.NARRATIVE_CLASS_FM_CODED] == 2
    # no figure of record moves: the errored and parse-failure epochs already fail at 0
    e_h = next(e for e in settled.samples["h"].epochs if e.epoch == 2)
    e_u = next(e for e in settled.samples["u"].epochs if e.epoch == 2)
    assert e_h.anchors == {a: 0.0 for a in ANCHOR_NAMES} and not e_u.narrative_wrong
    assert settled.pass_k == {"h": 0, "p": 0, "u": 0}


def test_audited_code_counts_never_reads_a_missing_carried_audit_row_as_real_error():
    samples = hard_samples(correctness=FM001)
    rec = carried_record(samples, [entry()], errored_rows=False)
    with pytest.raises(trials.RecordError, match="no audit row for FM-001"):
        trials.audited_code_counts(samples, rec, k=4, case_kinds=None)
    counts = trials.audited_code_counts(samples, rec, k=4, case_kinds=None, require_complete=False)
    assert counts == {"real_error": {}, "extractor_error": {}, trials.NOT_AUDITED: {"FM-000": 1, "FM-001": 1}}
    rec.audit("h", 2, "FM-001")
    rec.audit("h", 2, "FM-000")
    rec.narrative("h", 2)
    assert trials.audited_code_counts(samples, rec, k=4, case_kinds=None) == {
        "real_error": {"FM-000": 1, "FM-001": 1}, "extractor_error": {}}
    # an epoch nothing carried reads as before
    clean = [sample("c", e, kind="adversarial") for e in range(1, 5)]
    assert trials.audited_code_counts(clean, record_for(clean), k=4) == {"real_error": {}, "extractor_error": {}}


@pytest.mark.parametrize("verdict,h2", [("real_error", {"FM-000": 1}), ("extractor_error", {})])
def test_an_errored_epoch_supports_h2_only_when_the_audit_rules_it_real_error(verdict, h2):
    # Section 2 and decision D-R20: H2 counts FM-000 only when the audit rules it real_error;
    # the audit-adjusted figure keeps it either way (D-R5).
    samples = hard_samples(kind="adversarial")
    rec = carried_record(samples, [entry()], errored_rows=False)
    rec.audit("h", 2, "FM-000", verdict=verdict)
    rec.narrative("h", 2)
    counts = trials.audited_code_counts(samples, rec, k=4)
    assert counts["real_error"] == h2
    assert counts["extractor_error"] == ({} if h2 else {"FM-000": 1})
    assert trials.audit_adjusted_trial(samples, rec, k=4).pass_k == {"h": 0}


def test_unscored_epoch_is_carried_with_its_reason():
    samples = unscored_samples()
    rec = carried_record(samples, [entry(reason=UNSCORED)])
    rep = trials.review_settled_trial(samples, rec, k=4)
    e2 = next(e for e in rep.samples["h"].epochs if e.epoch == 2)
    assert e2.reasons == (UNSCORED,) and e2.failure_modes == ()  # not a format break (decision D-R20)
    assert e2.anchors["completeness_anchor"] is None


def test_carried_list_is_read_from_the_packet_summary(tmp_path):
    samples = hard_samples()
    rec = carried_record(samples, None)
    del rec.carried
    rec.packet_dir = str(tmp_path)
    summary = tmp_path / rs.PACKET_FILES["summary"]
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"totals": {trials.CARRIED_KEY: [entry()]}}), encoding="utf-8")
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 0}
    summary.write_text(json.dumps({"totals": {trials.CARRIED_KEY: [entry(), entry(epoch=3)]}}), encoding="utf-8")
    with pytest.raises(trials.RecordError, match="but the log scores it"):
        trials.review_settled_trial(samples, rec, k=4)
    summary.write_text(json.dumps({"totals": {}}), encoding="utf-8")
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 0}  # no list: read as before


# --- Public text ---------------------------------------------------------------------------------------

DASHES = "[" + chr(0x2013) + chr(0x2014) + "]"
BANNED = re.compile(r"Claude|Fable|the owner|worktree|scratchpad|Week \d|fix round", re.IGNORECASE)


def test_public_text_rules():
    for path in (SCRIPT, ROOT / "icfr_agent_evals" / "trials.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(DASHES, text), path
        assert b"\r" not in path.read_bytes(), path
    assert not BANNED.search(SCRIPT.read_text(encoding="utf-8"))
    assert not re.search(DASHES, Path(__file__).read_text(encoding="utf-8"))
