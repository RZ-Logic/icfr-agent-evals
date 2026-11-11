"""scripts/cost_estimate.py: the section 5 Cost and section 13 cost-row re-derivation.

No model call and no real log: each test writes a small synthetic .eval zip (a
header.json and a few dev sample files with model events) under tmp_path. The price
table is tied to the pinned documents: the section 13 cost row's worst-case figures and
the runbook's price sentence must come out of PRICES unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cost_estimate.py"

SONNET55, HAIKU55 = "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"
SONNET5, OPUS55 = "anthropic/claude-sonnet-5", "anthropic/claude-opus-5-5"


def _load():
    name = "cost_estimate_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


ce = _load()


def _event(model: str, inp: int, out: int, reasoning: int, cache_write: int = 0, cache_read: int = 0) -> dict:
    return {
        "event": "model", "model": model,
        "output": {"usage": {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": reasoning,
                             "input_tokens_cache_write": cache_write, "input_tokens_cache_read": cache_read},
                   "stop_reason": "stop"},
        "call": {"request": {"max_tokens": 48000}},
    }


def write_log(path: Path, model: str, samples: list[tuple[str, int, list[dict]]], *, split: str | None = "dev",
              meta_split: str | None = "dev", status: str = "success", dataset: str = "variance-commentary-dev-pilot",
              extra_files: dict[str, bytes] | None = None) -> Path:
    header = {
        "status": status,
        "eval": {
            "model": model,
            "task_args": {} if split is None else {"split": split},
            "metadata": {"worker_prompt_version": "v0.0.4", "worker_prompt_sha256": "a" * 64,
                         **({} if meta_split is None else {"split": meta_split})},
            "dataset": {"name": dataset},
        },
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps(header))
        for sid, epoch, events in samples:
            z.writestr(f"samples/{sid}_epoch_{epoch}.json",
                       json.dumps({"id": sid, "epoch": epoch, "events": events, "scores": {"x": {"value": 3}}}))
        for name, data in (extra_files or {}).items():
            z.writestr(name, data)
    return path


def simple_log(tmp_path: Path, model: str, std=(1000, 2000, 500), hard=(1000, 4000, 3000), name=None) -> Path:
    """One standard and one hard dev sample, each with one call (input, output, reasoning)."""
    return write_log(tmp_path / (name or f"{model.split('/')[-1]}.eval"), model, [
        ("dev-std-001-ETH-M1", 1, [_event(model, std[0], std[1], std[2])]),
        ("dev-hc-001-ETH-M2", 1, [_event(model, hard[0], hard[1], hard[2])]),
    ])


# --- Prices are the pinned ones -----------------------------------------------------------

def test_prices_reproduce_the_cost_row_worst_case():
    """Section 13 cost row: at 48,000 out, 3,482 in and list prices, 800 generations each."""
    text = (ROOT / "PRE-REGISTRATION-PHASE-1.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Cost cap confirmed"))
    stated = {name: float(v.replace(",", "")) for name, v in
              re.findall(r"(Sonnet 5\.5|Opus 5\.5|Sonnet 5|Haiku 5\.5) \$([\d,]+\.\d\d)", row)}
    total = float(re.search(r"\$([\d,]+\.\d\d) in all", row).group(1).replace(",", ""))
    assert ce.ROW_INPUT_TOKENS == 3482 and ce.MAX_TOKENS_DEFAULT == 48000
    got = {
        "Sonnet 5.5": 800 * ce.PRICES[SONNET55].cost(3482, 48000),
        "Opus 5.5": 800 * ce.PRICES[OPUS55].cost(3482, 48000),
        "Sonnet 5": 800 * ce.PRICES[SONNET5].cost(3482, 48000),
        "Haiku 5.5": 800 * ce.PRICES[HAIKU55].cost(3482, 48000),
    }
    assert set(stated) == set(got)
    for k, v in got.items():
        assert round(v, 2) == stated[k], k
    assert round(sum(got.values()), 2) == total


def test_prices_match_the_runbook_sentence():
    text = (ROOT / "docs" / "evidence" / "PILOT-RUNBOOK.md").read_text(encoding="utf-8")
    line = next(x for x in text.splitlines() if x.startswith("Inputs at 3.0 characters per token"))
    for model, label in ((SONNET55, "Sonnet 5.5"), (HAIKU55, "Haiku 5.5"), (OPUS55, "Opus 5.5")):
        m = re.search(re.escape(label) + r" \$([\d.]+) / \$([\d.]+)", line)
        assert m, label
        assert (float(m.group(1)), float(m.group(2))) == (ce.PRICES[model].input_per_mtok,
                                                          ce.PRICES[model].output_per_mtok)
    assert "3,482 input tokens" in line


def test_ladder_prices_are_half_of_standard_on_batch():
    assert ce.PRICES["openai/gpt-6.1-sol@batch"].input_per_mtok * 2 == ce.PRICES["openai/gpt-6.1-sol"].input_per_mtok
    assert ce.PRICES["openai/gpt-6.1-sol@batch"].output_per_mtok * 2 == ce.PRICES["openai/gpt-6.1-sol"].output_per_mtok
    terra, sol = ce.PRICES["openai/gpt-6.1-terra@batch"], ce.PRICES["openai/gpt-6.1-sol@batch"]
    assert (terra.input_per_mtok * 2, terra.output_per_mtok * 2) == (sol.input_per_mtok, sol.output_per_mtok)
    for p in ce.PRICES.values():
        assert p.source.startswith("CONTEXT.md")


# --- Reading a dev log --------------------------------------------------------------------

def test_reads_every_input_class_and_keeps_thinking_inside_output(tmp_path):
    log = write_log(tmp_path / "s.eval", SONNET55, [
        ("dev-std-001-ETH-M1", 1, [_event(SONNET55, 4, 900, 600, cache_write=4500, cache_read=0)]),
        ("dev-hc-001-ETH-M2", 2, [_event(SONNET55, 4, 1200, 700, cache_write=0, cache_read=4500)]),
    ])
    u = ce.read_log_usage(log)
    assert u.model == SONNET55 and u.samples == 2 and len(u.calls) == 2
    std, hard = sorted(u.calls, key=lambda c: c.stratum, reverse=True)
    assert (std.stratum, std.input_tokens, std.output_tokens, std.visible_tokens) == ("standard", 4504, 900, 300)
    assert (hard.stratum, hard.input_tokens, hard.output_tokens, hard.visible_tokens) == ("hard", 4504, 1200, 500)
    assert u.prompt_sha256 == "a" * 64


def test_other_model_calls_are_counted_not_billed(tmp_path):
    log = write_log(tmp_path / "s.eval", SONNET55, [
        ("dev-std-001-ETH-M1", 1, [_event(SONNET55, 1000, 1000, 0), _event("openai/gpt-6-luna", 9, 9, 0)]),
        ("dev-hc-001-ETH-M2", 1, [_event(SONNET55, 1000, 1000, 0)]),
    ])
    u = ce.read_log_usage(log)
    assert len(u.calls) == 2 and u.other_model_calls == 1


@pytest.mark.parametrize("kwargs, needle", [
    ({"split": "eval", "meta_split": "eval"}, "not dev"),
    ({"split": "dev", "meta_split": "eval"}, "not dev"),
    ({"split": "ci", "meta_split": None}, "not dev"),
    ({"split": None, "meta_split": None}, "records no split"),
    ({"dataset": "variance-commentary-eval"}, "evaluation split"),
    ({"status": "error"}, "not success"),
])
def test_refuses_anything_but_a_successful_dev_log_before_reading_samples(tmp_path, kwargs, needle):
    # The sample file is not JSON: the refusal must come from the header, before any sample is read.
    log = write_log(tmp_path / "x.eval", SONNET55, [], extra_files={"samples/dev-std-001_epoch_1.json": b"{not json"},
                    **kwargs)
    with pytest.raises(ce.Refused, match=needle):
        ce.read_log_usage(log)


def test_refuses_an_evaluation_sample_file(tmp_path):
    log = write_log(tmp_path / "x.eval", SONNET55, [("dev-std-001-ETH-M1", 1, [_event(SONNET55, 1, 1, 0)])],
                    extra_files={"samples/eval-std-001_epoch_1.json": b"{not json"})
    with pytest.raises(ce.Refused, match="not a dev sample"):
        ce.read_log_usage(log)


def test_refuses_off_roster_model_and_unreadable_file(tmp_path):
    log = simple_log(tmp_path, "openai/gpt-6-luna")
    with pytest.raises(ce.Refused, match="not a roster worker"):
        ce.read_log_usage(log)
    bad = tmp_path / "bad.eval"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ce.Refused, match="not a readable"):
        ce.read_log_usage(bad)


def test_refuses_two_logs_of_one_model(tmp_path):
    a = ce.read_log_usage(simple_log(tmp_path, SONNET55, name="a.eval"))
    b = ce.read_log_usage(simple_log(tmp_path, SONNET55, name="b.eval"))
    with pytest.raises(ce.Refused, match="two logs"):
        ce.estimate([a, b])


def test_refuses_a_log_missing_a_stratum(tmp_path):
    log = write_log(tmp_path / "s.eval", SONNET55, [("dev-std-001-ETH-M1", 1, [_event(SONNET55, 1, 1, 0)])])
    with pytest.raises(ce.Refused, match="hard"):
        ce.estimate([ce.read_log_usage(log)])


# --- Arithmetic ---------------------------------------------------------------------------

def test_worker_cost_weights_each_stratum_by_its_evaluation_generations(tmp_path):
    u = ce.read_log_usage(simple_log(tmp_path, SONNET55))
    w = ce.worker_estimate(u)
    price = ce.PRICES[SONNET55]
    std_cost = price.cost(1000, 2000)
    hard_cost = price.cost(1000, 4000)
    assert ce.eval_generations() == {"standard": 584, "hard": 216}
    assert w["cost"] == pytest.approx(584 * std_cost + 216 * hard_cost)
    assert w["mean_output"] == pytest.approx((584 * 2000 + 216 * 4000) / 800)
    assert w["mean_visible"] == pytest.approx((584 * 1500 + 216 * 1000) / 800)
    assert w["worst_row_input"] == pytest.approx(800 * price.cost(3482, 48000))
    assert w["worst_measured_input"] == pytest.approx(800 * price.cost(1000, 48000))


def test_full_estimate_components_and_caps(tmp_path):
    usages = [ce.read_log_usage(simple_log(tmp_path, m)) for m in (SONNET55, HAIKU55, SONNET5, OPUS55)]
    est = ce.estimate(usages)
    assert est["complete"] and est["missing_workers"] == []
    # Calibrator: 10 percent of 2,400 non-Opus generations at Opus prices; input is Opus's mean input
    # plus the non-Opus workers' mean visible answer; output is Opus's mean output.
    cal = est["calibrator"]
    visible = (584 * 1500 + 216 * 1000) / 800
    out = (584 * 2000 + 216 * 4000) / 800
    assert cal["calls"] == 240
    assert cal["input_per_call"] == pytest.approx(1000 + visible)
    assert cal["cost"] == pytest.approx(240 * ce.PRICES[OPUS55].cost(1000 + visible, out))
    assert cal["worst"] == pytest.approx(240 * ce.PRICES[OPUS55].cost(1000 + visible, 48000))
    # Grader: 3,200 generations plus 3 x 40 x 4 x 4 test-retest calls, 800 out each.
    assert est["grader_calls"] == {"generations": 3200, "test_retest": 1920, "total": 5120}
    g1 = est["graders"]["step 1"]
    assert g1["cost"] == pytest.approx(5120 * ce.PRICES["openai/gpt-6-luna"].cost(1000 + visible, 800))
    # Ladder steps replace the grader: step 2 is five times step 1 on these prices, step 3 twice step 2.
    assert est["graders"]["step 2"]["cost"] == pytest.approx(5 * g1["cost"])
    assert est["graders"]["step 3"]["cost"] == pytest.approx(2 * est["graders"]["step 2"]["cost"])
    assert est["sol_standard"]["cost"] == pytest.approx(2 * est["graders"]["step 3"]["cost"])
    # Scout is a placeholder until the scan.
    assert est["scout"]["basis"] == "placeholder" and est["scout"]["calls"] == 6400
    t1 = est["totals"]["step 1"]
    assert t1["without_route_judge"] == pytest.approx(est["worker_cost"] + cal["cost"] + g1["cost"]
                                                      + est["scout"]["cost"])
    assert t1["with_route_judge"] == pytest.approx(t1["without_route_judge"] + 19.0)


def test_scout_scan_replaces_the_placeholder_and_scales_to_the_full_run(tmp_path):
    usages = [ce.read_log_usage(simple_log(tmp_path, SONNET55))]
    est = ce.estimate(usages, scan=(160, 1_000_000, 100_000), transcripts=80)
    s = est["scout"]
    assert s["basis"] == "measured"
    assert s["calls"] == pytest.approx(160 * 40)
    assert s["cost"] == pytest.approx(ce.PRICES["openai/gpt-6-luna"].cost(1_000_000, 100_000) * 40)


def test_missing_workers_mark_the_estimate_incomplete(tmp_path):
    est = ce.estimate([ce.read_log_usage(simple_log(tmp_path, SONNET55))])
    assert not est["complete"]
    assert est["missing_workers"] == [HAIKU55, SONNET5, OPUS55]
    assert est["calibrator"] is None
    assert "INCOMPLETE" in ce.format_report(est)


@pytest.mark.parametrize("total, status", [
    (149.99, "under the $150 soft cap"), (150.0, "under the $150 soft cap"),
    (150.01, "over the $150 soft cap, under the $300 hard cap"), (300.01, "over the $300 hard cap"),
])
def test_cap_status(total, status):
    assert ce.cap_status(total) == status


# --- Command line -------------------------------------------------------------------------

def test_cli_prints_the_table_and_writes_json(tmp_path, capsys):
    logs = [str(simple_log(tmp_path, m)) for m in (SONNET55, HAIKU55, SONNET5, OPUS55)]
    out = tmp_path / "cost.json"
    assert ce.main([*logs, "--json", str(out)]) == ce.EXIT_OK
    text = capsys.readouterr().out
    for needle in ("| Claude Sonnet 5.5 worker | measured", "| Calibrator, Opus 5.5 | inferred",
                   "| assumed |", "| Scout scanners, GPT-6 Luna | placeholder", "stated (section 5)",
                   "$1,772.55", "$150 soft", "$300 hard"):
        assert needle in text, needle
    assert chr(0x2013) not in text and chr(0x2014) not in text  # no en or em dash
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["complete"] is True and set(data["workers"]) == {SONNET55, HAIKU55, SONNET5, OPUS55}


def test_cli_refusals_exit_3(tmp_path, capsys):
    eval_log = write_log(tmp_path / "e.eval", SONNET55, [], split="eval", meta_split="eval")
    assert ce.main([str(eval_log)]) == ce.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        ce.main(["-h"])
    assert e.value.code == ce.EXIT_REFUSED
    with pytest.raises(SystemExit) as e:
        ce.main([str(eval_log), "--scout-scan", "0,1,1"])
    assert e.value.code == ce.EXIT_REFUSED


def test_grader_output_thresholds_land_on_the_caps(tmp_path):
    usages = [ce.read_log_usage(simple_log(tmp_path, m)) for m in (SONNET55, HAIKU55, SONNET5, OPUS55)]
    est = ce.estimate(usages)
    base = est["totals"]["step 1"]["without_route_judge"] - est["graders"]["step 1"]["cost"]
    workers = est["workers"]
    for step, key in (*ce.GRADER_STEPS, ("step 3 standard", "openai/gpt-6.1-sol")):
        for cap in (150.0, 300.0):
            out = est["grader_output_thresholds"][step][f"{cap:.0f}"]
            if out is None:
                assert base + ce.grader_estimate(workers, key, 0)["cost"] > cap
            else:
                assert base + ce.grader_estimate(workers, key, out)["cost"] == pytest.approx(cap)


def test_input_cost_and_cache_read_share_are_reported(tmp_path):
    log = write_log(tmp_path / "s.eval", SONNET55, [
        ("dev-std-001-ETH-M1", 1, [_event(SONNET55, 0, 1000, 0, cache_write=1000, cache_read=3000)]),
        ("dev-hc-001-ETH-M2", 1, [_event(SONNET55, 0, 1000, 0, cache_write=1000, cache_read=3000)]),
    ])
    w = ce.worker_estimate(ce.read_log_usage(log))
    assert w["cache_read_share"] == pytest.approx(0.75)
    assert w["input_cost"] == pytest.approx(800 * 4000 * 2.0 / 1_000_000)
