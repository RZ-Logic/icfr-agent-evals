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
from icfr_agent_evals import roster, trials
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


def assert_withheld(text, payload, why):
    assert payload["withheld"] is True
    assert "Review-settled and audit-adjusted figures: withheld" in text
    assert any(why in r for r in payload["withheld_reasons"]), payload["withheld_reasons"]
    for kind in ("review_settled", "audit_adjusted"):
        assert f"{rr.FIGURE_NAMES[kind]} content-pass^4, {rr.FIGURE_ROLES[kind]}" not in text
    assert "anchor means, review-settled" not in text and "second-look gate" not in text and "== H3" not in text
    for w in payload["workers"].values():
        assert w["record"] is None
        assert w["log"]["figures"]["route_blind"]["all"]["n"] == 3
    assert "route-blind content-pass^4" in text and "code-only content-pass^4" in text
    assert "citation-pass^4" in text and "format breaks" in text


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


def test_withheld_when_the_record_does_not_name_a_worker_log(small, small_logs, base_log, tmp_path, monkeypatch):
    commit_second(small)
    other = make_log(base_log, tmp_path / "gamma.eval", SONNET5, SMALL, plan_for("east"))
    workers, template = items({"alpha": small_logs["alpha"], "beta": small_logs["beta"], "gamma": other},
                              small_logs["template"])
    no_record_figures(monkeypatch)
    text, payload = report(small, workers, template)
    assert_withheld(text, payload, "names no run with the eval id of worker log gamma")


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
    assert set(modes.values()) == {"reasoning_effort=high"}
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
    settled = payload["h3"]["readings"]["review_settled"]
    assert settled["verdict"] == "not tested" and settled["holds"] is None
    assert "Sonnet 5" in settled["second_look"]["reason"] and "review not confirmed" in settled["second_look"]["reason"]
    assert "H3 on review-settled anchor scores, the reading H3 is evaluated on (decision D-A): not tested" in text


# --- trials.py: carried epochs need no review row ----------------------------------------------------

def carried_record(samples, entries, **kw):
    """record_for on the samples the packet reads (carried epochs left out), with the carried list."""
    errored = [s for s in samples if trials.carried_reason(trials._collect([s], 1)[1][(str(s.id), int(s.epoch))])]
    rec = record_for([s for s in samples if s not in errored], **kw)
    rec.carried = entries
    return rec


def hard_samples(err_epoch=2, **err_kw):
    out = [sample("h", e, kind="edge") for e in range(1, 5) if e != err_epoch]
    out.append(sample("h", err_epoch, kind="edge", error=SimpleNamespace(message="boom"), **err_kw))
    return sorted(out, key=lambda s: s.epoch)


def entry(sid="h", epoch=2, reason="sample error", run_id="R1"):
    return {"run_id": run_id, "sample_id": sid, "epoch": epoch, "reason": reason}


def test_errored_epoch_counts_as_a_failed_trial_with_no_review_row():
    samples = hard_samples()
    rec = carried_record(samples, [entry()])
    rep = trials.review_settled_trial(samples, rec, k=4)
    e2 = next(e for e in rep.samples["h"].epochs if e.epoch == 2)
    assert not e2.success and e2.reasons == ("sample error",)
    assert rep.pass_k == {"h": 0}
    assert all(e.success for e in rep.samples["h"].epochs if e.epoch != 2)
    assert trials.audit_adjusted_trial(samples, rec, k=4).pass_k == {"h": 0}


def test_a_record_without_the_carried_list_reads_as_before():
    samples = hard_samples()
    rec = carried_record(samples, None)
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


def test_a_narrative_member_on_a_carried_epoch_is_a_gap():
    samples = hard_samples()
    rec = carried_record(samples, [entry()])
    rec.narrative("h", 2)
    with pytest.raises(trials.RecordError, match="reads the narrative of an epoch the packet carried"):
        trials.review_settled_trial(samples, rec, k=4)


def test_an_fm_coded_carried_epoch_needs_an_audit_row_only_when_the_packet_built_one():
    fm =SimpleNamespace(value=0, metadata={"failure_modes": ["FM-001"], "requires_human_review": False,
                                            "review_reasons": [], "details": {"structured_points": 3}})
    samples = hard_samples(correctness=fm)
    rec = carried_record(samples, [entry()])
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 0}
    rec.audit("h", 2, "FM-001", verdict=None)
    with pytest.raises(trials.RecordError, match="h epoch 2: audit verdict None for FM-001"):
        trials.review_settled_trial(samples, rec, k=4)


def test_unscored_epoch_is_carried_with_its_reason():
    blank = SimpleNamespace(value=None, metadata={})
    samples = [sample("h", e, kind="edge") for e in (1, 2, 3)]
    samples.append(sample("h", 4, kind="edge", completeness=blank))
    rec = carried_record(samples, [entry(epoch=4, reason="completeness_anchor unscored or non-numeric")])
    rep = trials.review_settled_trial(samples, rec, k=4)
    e4 = next(e for e in rep.samples["h"].epochs if e.epoch == 4)
    assert e4.reasons == ("completeness_anchor unscored or non-numeric",)


def test_carried_list_is_read_from_the_packet_summary(tmp_path):
    samples = hard_samples()
    rec = carried_record(samples, None)
    del rec.carried
    rec.packet_dir = str(tmp_path)
    summary = tmp_path / rs.PACKET_FILES["summary"]
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"totals": {trials.CARRIED_KEY: [entry()]}}), encoding="utf-8")
    assert trials.review_settled_trial(samples, rec, k=4).pass_k == {"h": 0}
    summary.write_text(json.dumps({"totals": {}}), encoding="utf-8")
    with pytest.raises(trials.RecordError, match="no whole-narrative member"):
        trials.review_settled_trial(samples, rec, k=4)


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
