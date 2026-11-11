"""scripts/review_load.py: the section 5 review-load estimate (D-G, D-J) from the timed dev rehearsal.

No model call and no real packet: each test writes a small synthetic packet folder
(packet-meta.json, sealed/summary.json, reviewer/packet.csv) and a synthetic sittings
file under tmp_path. The sittings file is read and checked by review_freeze.py's own code.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from icfr_agent_evals import review_schema as rs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_load.py"


def _load():
    name = "review_load_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rl = _load()

TEN_WORDS = "one two three four five six seven eight nine ten"


def _narr(uid: str, case: str, cls: str, words: int = 120) -> dict:
    text = " ".join([TEN_WORDS] * (words // 10))
    return {"row_kind": "narrative", "unit_id": uid, "case_id": case, "route_class": cls, "narrative": text}


def write_packet(root: Path, *, rows: list[dict] | None = None, summary_override: dict | None = None,
                 scope: str = "hard") -> Path:
    """Standard stratum: 2 cases, 4 finding units, 2 routed narratives, 1 audit unit.
    Hard stratum: 1 case, 2 finding units, 1 FM-coded narrative, 2 narrative-only narratives."""
    if rows is None:
        rows = [
            {"row_kind": "finding", "unit_id": "F-1", "case_id": "dev-std-001-ETH-M1", "route_key": "k"},
            {"row_kind": "finding", "unit_id": "F-2", "case_id": "dev-std-001-ETH-M1", "route_key": "k"},
            {"row_kind": "finding", "unit_id": "F-3", "case_id": "dev-std-002-BTC-M2", "route_key": "k"},
            {"row_kind": "finding", "unit_id": "F-4", "case_id": "dev-std-002-BTC-M2", "route_key": "k"},
            {"row_kind": "finding", "unit_id": "F-5", "case_id": "dev-hc-001-SOL-M3", "route_key": "k"},
            {"row_kind": "finding", "unit_id": "F-6", "case_id": "dev-hc-001-SOL-M3", "route_key": "k"},
            _narr("N-1", "dev-std-001-ETH-M1", "routed"),
            _narr("N-2", "dev-std-002-BTC-M2", "routed"),
            _narr("N-3", "dev-hc-001-SOL-M3", "fm_coded"),
            _narr("N-4", "dev-hc-001-SOL-M3", "narrative_only", words=60),
            _narr("N-5", "dev-hc-001-SOL-M3", "narrative_only", words=60),
            {"row_kind": "fm_audit", "unit_id": "A-1", "case_id": "dev-std-001-ETH-M1", "fm_code": "FM-000"},
        ]
    kinds = {k: sum(1 for r in rows if r["row_kind"] == k) for k in ("finding", "narrative", "fm_audit")}
    by_class = {c: sum(1 for r in rows if r["row_kind"] == "narrative" and r["route_class"] == c)
                for c in rs.NARRATIVE_CLASSES}
    totals = {"finding_units": kinds["finding"], "narrative_units": kinds["narrative"],
              "audit_units": kinds["fm_audit"], "narrative_units_by_class": by_class,
              "cases_with_units": len({r["case_id"] for r in rows}), "findings": kinds["finding"] + 1}
    totals.update(summary_override or {})
    (root / "sealed").mkdir(parents=True, exist_ok=True)
    (root / "reviewer").mkdir(parents=True, exist_ok=True)
    (root / rs.PACKET_FILES["summary"]).write_text(json.dumps({"R1": {"label": "x"}, "totals": totals}),
                                                   encoding="utf-8")
    (root / rs.PACKET_FILES["meta"]).write_text(json.dumps({"run_id": "dev-test", "scope": scope}), encoding="utf-8")
    rs.write_csv(root / rs.PACKET_FILES["packet"], rs.PACKET_COLUMNS, rows)
    return root


def write_sittings(path: Path, pairs: list[tuple[str, str]]) -> Path:
    rs.write_csv(path, ("start", "end"), [{"start": s, "end": e} for s, e in pairs])
    return path


DEV = {"standard": 2, "hard": 1}


# --- Packet counts ------------------------------------------------------------------------

def test_read_packet_counts_by_stratum_and_words(tmp_path):
    pc = rl.read_packet(write_packet(tmp_path / "p"))
    st, hd = pc.by_stratum["standard"], pc.by_stratum["hard"]
    assert (st["finding"], st["routed"], st["only"], st["audit"], st["cases"]) == (4, 2, 0, 1, 2)
    assert (hd["finding"], hd["routed"], hd["only"], hd["audit"], hd["cases"]) == (2, 1, 2, 0, 1)
    assert (st["words_routed"], hd["words_routed"], hd["words_only"]) == (240, 120, 120)
    assert pc.findings == 7 and pc.run_id == "dev-test" and pc.scope == "hard"


def test_refuses_a_packet_that_does_not_match_its_summary(tmp_path):
    with pytest.raises(rl.Refused, match="finding_units"):
        rl.read_packet(write_packet(tmp_path / "p", summary_override={"finding_units": 99}))
    with pytest.raises(rl.Refused, match="narrative_units_by_class"):
        rl.read_packet(write_packet(tmp_path / "q", summary_override={
            "narrative_units_by_class": {"routed": 2, "fm_coded": 1, "narrative_only": 1}}))


def test_refuses_a_case_off_the_dev_split(tmp_path):
    rows = [{"row_kind": "finding", "unit_id": "F-1", "case_id": "eval-std-001-ETH-M1", "route_key": "k"}]
    with pytest.raises(rl.Refused, match="not a dev case"):
        rl.read_packet(write_packet(tmp_path / "p", rows=rows))


# --- Rates --------------------------------------------------------------------------------

def test_model_only_rates_are_the_section_5_assumptions(tmp_path):
    pc = rl.read_packet(write_packet(tmp_path / "p"))
    r = rl.derive(pc, None, None, DEV)
    assert r["basis"].startswith("model only") and r["measured_to_model"] is None
    low, high = r["minutes_per_unit"]["low"], r["minutes_per_unit"]["high"]
    assert (low["finding"], high["finding"]) == (0.5, 1.5)          # 30 and 90 seconds
    assert (low["case"], high["case"]) == (1.5, 3.0)                # 90 and 180 seconds
    assert (low["audit"], high["audit"]) == (0.5, 2.0)              # 30 and 120 seconds
    # 120 words: 15 s + 60 s at 120 wpm (low); 30 s + 120 s at 60 wpm (high).
    assert low["narrative_routed"] == pytest.approx(75 / 60)
    assert high["narrative_routed"] == pytest.approx(150 / 60)
    # Dev packet, low: 3 narratives x 75 s + 2 narrative-only x (15 + 30) s + 6 x 30 + 3 cases x 90 + 1 x 30.
    assert r["dev_model_hours"]["low"] == pytest.approx((3 * 75 + 2 * 45 + 6 * 30 + 3 * 90 + 30) / 3600)


def test_measured_time_is_shared_out_so_both_splits_add_up_to_it(tmp_path):
    pc = rl.read_packet(write_packet(tmp_path / "p"))
    sit = write_sittings(tmp_path / "s.csv", [("2026-11-12T09:00:00-05:00", "2026-11-12T09:30:00-05:00")])
    pairs, hours = rl.read_sittings(sit)
    assert hours == 0.5
    r = rl.derive(pc, pairs, hours, DEV)
    units = 6 + 3 + 2 + 1
    assert r["blended_minutes_per_unit"] == pytest.approx(30 / units)
    for case in ("low", "high"):
        mpu = r["minutes_per_unit"][case]
        total = (6 * mpu["finding"] + 3 * mpu["narrative_routed"] + 2 * mpu["narrative_only"]
                 + 3 * mpu["case"] + 1 * mpu["audit"])
        assert total == pytest.approx(30.0)
        assert r["measured_to_model"][case] == pytest.approx(0.5 / r["dev_model_hours"][case])


# --- Projection ---------------------------------------------------------------------------

def test_projection_scales_each_stratum_and_applies_the_volume_assumptions(tmp_path):
    pc = rl.read_packet(write_packet(tmp_path / "p"))
    r = rl.derive(pc, None, None, DEV)
    p = r["projection"]["per_worker_units"]
    # standard x 146/2, hard x 54/1
    assert p["finding"] == pytest.approx(4 * 73 + 2 * 54)
    assert p["routed"] == pytest.approx(2 * 73 + 1 * 54)
    assert p["only"] == pytest.approx(2 * 54)
    assert p["audit"] == pytest.approx(73)
    assert p["cases"] == pytest.approx(200)
    low, high = r["projection"]["low"], r["projection"]["high"]
    s55 = low["workers"]["Sonnet 5.5"]
    assert s55["second_look_units"] == max(100, math.ceil(0.1 * p["routed"]))
    # Low: case facts once per case for all four workers; high: per worker.
    assert s55["hours"]["case_facts"] == pytest.approx(200 * 90 / 4 / 3600)
    assert high["workers"]["Sonnet 5.5"]["hours"]["case_facts"] == pytest.approx(200 * 180 / 3600)
    # High route volume: Opus 5.5 at 1.25 times; audit at least the D-G rows.
    assert high["workers"]["Opus 5.5"]["finding_units"] == pytest.approx(1.25 * p["finding"])
    assert high["workers"]["Haiku 5.5"]["audit_rows"] == 200
    assert low["total_hours"] == pytest.approx(sum(w["total_hours"] for w in low["workers"].values()))
    # D-J beside the range: narrative-only reads of four workers, not scaled by route volume.
    assert low["dj_epochs_per_full_run"] == pytest.approx(4 * 108)
    assert low["dj_hours"] == pytest.approx(4 * 108 * 45 / 3600)


def test_dev_cases_must_cover_the_packet(tmp_path):
    pc = rl.read_packet(write_packet(tmp_path / "p"))
    with pytest.raises(rl.Refused, match="more than --dev-cases"):
        rl.derive(pc, None, None, {"standard": 1, "hard": 1})


# --- Sittings (review_freeze's own checks) and the fit by date -------------------------------

@pytest.mark.parametrize("pairs, needle", [
    ([("2026-11-12T09:00:00", "2026-11-12T10:00:00")], "UTC offset"),
    ([("2026-11-12T10:00:00-05:00", "2026-11-12T09:00:00-05:00")], "not before end"),
    ([("2026-11-12T09:00:00-05:00", "2026-11-12T10:00:00-05:00"),
      ("2026-11-12T09:30:00-05:00", "2026-11-12T11:00:00-05:00")], "overlap"),
    ([], "at least one sitting"),
])
def test_bad_sittings_are_refused_by_review_freezes_checks(tmp_path, pairs, needle):
    with pytest.raises(rl.Refused, match=needle):
        rl.read_sittings(write_sittings(tmp_path / "s.csv", pairs))


def test_sittings_hours_match_review_freeze(tmp_path):
    pairs = [("2026-11-12T09:00:00-05:00", "2026-11-12T10:15:00-05:00"),
             ("2026-11-13T09:00:00+01:00", "2026-11-13T09:45:00+01:00")]
    _, hours = rl.read_sittings(write_sittings(tmp_path / "s.csv", pairs))
    assert hours == rl.load_review_freeze()._review_hours(pairs) == 2.0


def _answers(path: Path, per_date: dict[str, dict[str, int]]) -> Path:
    rows, n = [], 0
    for date, kinds in per_date.items():
        for kind, count in kinds.items():
            for _ in range(count):
                n += 1
                verdict = {"finding": "correct", "narrative": "none_wrong", "fm_audit": "real_error"}[kind]
                rows.append({"row_kind": kind, "unit_id": f"U-{n}", "verdict": verdict, "date": date,
                             "note": "SECRET-NOTE"})
    rows.append({"row_kind": "finding", "unit_id": "U-blank", "verdict": "", "date": ""})
    rs.write_rb7_csv(path, rows)
    return path


def test_fit_by_date_recovers_rates_when_the_dates_separate_the_kinds(tmp_path):
    # True rates: finding 1 min, narrative 2 min.
    per_date = {"2026-11-12": {"finding": 30, "narrative": 10}, "2026-11-13": {"finding": 10, "narrative": 20},
                "2026-11-14": {"finding": 20, "narrative": 5}}
    pairs = []
    for d, k in per_date.items():
        minutes = k["finding"] * 1 + k["narrative"] * 2
        pairs.append((f"{d}T09:00:00-05:00", f"{d}T{9 + minutes // 60:02d}:{minutes % 60:02d}:00-05:00"))
    answered = rl.answered_by_date(_answers(tmp_path / "a.csv", per_date))
    assert "" not in answered  # a blank verdict is not counted
    fit = rl.fit_by_date(pairs, answered)
    assert fit["allowed"] is True
    assert fit["minutes_per_unit"]["finding"] == pytest.approx(1.0)
    assert fit["minutes_per_unit"]["narrative"] == pytest.approx(2.0)


def test_fit_by_date_is_refused_when_the_data_do_not_allow_it(tmp_path):
    one_day = {"2026-11-12": {"finding": 10, "narrative": 5}}
    answered = rl.answered_by_date(_answers(tmp_path / "a.csv", one_day))
    fit = rl.fit_by_date([("2026-11-12T09:00:00-05:00", "2026-11-12T10:00:00-05:00")], answered)
    assert fit["allowed"] is False and "more dates than kinds" in fit["why"]
    fit = rl.fit_by_date([("2026-11-13T09:00:00-05:00", "2026-11-13T10:00:00-05:00")], answered)
    assert fit["allowed"] is False and "no sitting" in fit["why"]


# --- Command line -------------------------------------------------------------------------

def test_cli_prints_the_estimate_and_never_a_verdict(tmp_path, capsys):
    packet = write_packet(tmp_path / "p")
    sit = write_sittings(tmp_path / "s.csv", [("2026-11-12T09:00:00-05:00", "2026-11-12T10:00:00-05:00"),
                                              ("2026-11-13T09:00:00-05:00", "2026-11-13T09:30:00-05:00")])
    ans = _answers(tmp_path / "a.csv", {"2026-11-12": {"finding": 4, "narrative": 3},
                                        "2026-11-13": {"finding": 2, "narrative": 2, "fm_audit": 1}})
    out = tmp_path / "load.json"
    code = rl.main(["--packet-dir", str(packet), "--sittings", str(sit), "--answers", str(ans),
                    "--dev-cases", "standard=2,hard=1", "--json", str(out)])
    assert code == rl.EXIT_OK
    text = capsys.readouterr().out
    for needle in ("Basis: measured.", "Minutes per unit", "Re-derived range", "78 to 259", "Beside it (D-J)",
                   "Fit by date not possible"):
        assert needle in text, needle
    for word in ("SECRET-NOTE", "none_wrong", "real_error", "correct"):
        assert word not in text, word
    assert chr(0x2013) not in text and chr(0x2014) not in text  # no en or em dash
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["measured_hours"] == 1.5


def test_cli_without_sittings_says_nothing_is_measured(tmp_path, capsys):
    assert rl.main(["--packet-dir", str(write_packet(tmp_path / "p")), "--dev-cases", "standard=2,hard=1"]) == 0
    assert "No sittings file: no time is measured yet." in capsys.readouterr().out


def test_cli_refusals_exit_3(tmp_path, capsys):
    packet = write_packet(tmp_path / "p")
    assert rl.main(["--packet-dir", str(packet), "--answers", "x.csv"]) == rl.EXIT_REFUSED
    assert rl.main(["--packet-dir", str(tmp_path / "missing")]) == rl.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
    for argv in (["-h"], ["--packet-dir", str(packet), "--dev-cases", "standard=0"]):
        with pytest.raises(SystemExit) as e:
            rl.main(argv)
        assert e.value.code == rl.EXIT_REFUSED
