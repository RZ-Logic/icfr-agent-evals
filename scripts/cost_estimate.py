"""Cost per full run, re-derived from dev-run per-call tokens. NO API key, no model call, no network.

Usage:
  python scripts/cost_estimate.py LOG.eval [LOG.eval ...]
         [--scout-scan CALLS,INPUT_TOKENS,OUTPUT_TOKENS] [--scout-transcripts 80] [--json OUT.json]

Pre-registration section 5 (Cost) and the section 13 cost row. A full run is the worker
and the three cross-model baselines on the 200-case evaluation split at epochs = 4
(3,200 generations), with the grader, calibrator and Scout scanner calls on those
generations. This script re-derives its dollar cost from the per-call token counts the
dev logs record, billed thinking included, and sets it against the $150 soft and $300
hard cap.

Each log is one roster worker's dev run (Sonnet 5.5, Haiku 5.5, Sonnet 5 or Opus 5.5),
told apart by the model in its header. The log is read as a zip file. Before any sample
is opened, a log is refused unless its status is success and every split it records is
dev; a log that records no split, a dataset named for the evaluation split, or a sample
file outside dev is refused too. The script reads the usage of each model call and never
reads a score.

What it derives, and on what basis (each line of the output says which):
  measured  each worker's cost from its own dev log: mean cost per call per stratum
            (standard, hard) scaled to the evaluation split's 146 standard and 54 hard
            cases at 4 epochs. Billed output is Inspect's output_tokens, which holds the
            thinking tokens (reasoning_tokens is a part of it, never added to it).
  inferred  the calibrator: about 10 percent of the 2,400 non-Opus generations (240
            calls) at Opus 5.5 prices, each call's input taken as Opus 5.5's mean dev
            input plus the mean visible answer of the non-Opus workers, its output as
            Opus 5.5's mean dev output (section 5: no calibrator call runs before the tag).
  assumed   the grader on GPT-6 Luna, one call per generation plus the calibration-time
            work at one acceptance look: the calibration subset's first scoring (~800
            outputs) and its test-retest replicates (both on CONTEXT.md's full-run list),
            and the bias-audit passes run at the same look (not on that list). The upper
            reading is counted in the totals; the lower reading and the totals with no
            calibration-time work are printed beside them. Each call's input is taken as
            the mean dev worker input plus the mean visible answer, its output 800 tokens
            (CONTEXT.md pass-5 judge call, reasoning included). No grader call has run, so
            nothing here is measured.
  ladder    the same grader calls at the D-T1 steps: GPT-6.1 Terra and GPT-6.1 Sol on
            the OpenAI Batch API (half the standard price), and Sol at standard price for
            reference.
  placeholder  the Scout scanners until the owner's scan of the Sonnet 5.5 pilot log
            (runbook step 10). With --scout-scan the scan's own totals replace it: the
            scan's input is scaled to each worker's transcripts by that worker's measured
            mean input plus output against Sonnet 5.5's, its output by transcript count.
  stated    the exploratory route judge, $9 to $19 per full run and $103 at the worst
            case, carried from section 5 as stated, not re-derived.
  worst     each worker at max_tokens 48,000 (no pin is set), at the cost row's 3,482
            input tokens (this reproduces the row's $1,772.55) and at the measured mean
            input; the calibrator at 48,000 too.

Prices come only from the repository's pinned documents (PRICES below names the source
of each). Every input token is billed at the model's base input price: the logs show
prompt-cache writes and reads, and no cache price is pinned in those documents. Most input
tokens are cache reads and about a fifth are cache writes. Reads are usually billed below
base and writes above it, so the input cost is an upper estimate only if the read discount
outweighs the write premium. The report prints the input cost, both shares, and the effect
of cache writes at stated premiums (1.25 and 2 times base, assumptions, not pinned prices).

Exit codes: 0 printed; 3 refused (a log off the dev split, a model off the roster, two
logs of one model, an unreadable log, a help request or an argument error).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_OK, EXIT_REFUSED = 0, 3
MTOK = 1_000_000

SOFT_CAP, HARD_CAP = 150.0, 300.0
EPOCHS = 4
# Evaluation split composition (section 13 generator row): 146 standard synthetic, 30 edge
# and 24 adversarial cases. Edge and adversarial cases are the hard stratum.
EVAL_CASES: dict[str, int] = {"standard": 146, "hard": 30 + 24}
FULL_RUN_GENERATIONS = 3_200
MAX_TOKENS_DEFAULT = 48_000          # Inspect's default at effort high (section 5, Reasoning)
ROW_INPUT_TOKENS = 3_482             # the section 13 cost row and PILOT-RUNBOOK "Cost at list price"
CALIBRATOR_SHARE = 0.10              # section 5 roster table: about 10 percent of non-Opus generations
JUDGE_OUTPUT_TOKENS = 800            # CONTEXT.md pass-5 assumption: about 800 out per judge call
TEST_RETEST_REPLICATES, CALIBRATION_CASES = 3, 40   # section 7 and CONTEXT.md cost ceiling
# Calibration-time grader work at one acceptance look on the 40-case calibration subset.
# CONTEXT.md's full-run list: "The calibration subset adds ~800 graded both ways" (by hand and
# by the autograder, per the Autograder calibration bullet), "at least 3 test-retest
# replicates". The bias audit runs at the same look (CONTEXT.md judge replication bullet:
# (a) option order reversed and one random permutation, (b) score labels relabeled, (c)
# dimension order permuted where dimensions share a prompt); it is not on the full-run list.
# Two readings; the upper one is counted in the totals.
CALIBRATION_READINGS: dict[str, dict[str, int]] = {
    # outputs per pass, test-retest replicates beyond the first scoring, bias-audit passes
    "upper": {"outputs": 800, "retest_extra": TEST_RETEST_REPLICATES, "bias_passes": 4},
    "lower": {"outputs": CALIBRATION_CASES * EPOCHS * 4, "retest_extra": TEST_RETEST_REPLICATES - 1,
              "bias_passes": 3},
}
# Cache-write premiums for the sensitivity line: stated assumptions, not prices pinned in the
# repository and not fetched.
CACHE_WRITE_PREMIUMS: tuple[float, ...] = (1.25, 2.0)
SCOUT_MODEL_SCANNERS = 2             # eval_awareness and refusal_classifier; command_not_found is a regex
PILOT_TRANSCRIPTS = 80               # 20 dev cases x 4 epochs
ROUTE_JUDGE_LOW, ROUTE_JUDGE_HIGH, ROUTE_JUDGE_WORST = 9.0, 19.0, 103.0  # section 5, stated
SENSITIVITY_JUDGE_OUTPUTS: tuple[int, ...] = (800, 2_000, 8_000)


@dataclass(frozen=True)
class Price:
    model: str
    input_per_mtok: float
    output_per_mtok: float
    source: str

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / MTOK


_CTX_PRICES = "CONTEXT.md, bullet 'Prices per MTok (pass 5, Haiku 5.5 added pass 6)'"
_ADR6 = "CONTEXT.md, ADR-0006 bullet (owner decision D-T1) and the 'Prices per MTok' bullet"
PRICES: dict[str, Price] = {
    "anthropic/claude-sonnet-5-5": Price("Claude Sonnet 5.5", 2.0, 10.0,
                                         f"{_CTX_PRICES}; docs/evidence/PILOT-RUNBOOK.md 'Cost at list price'"),
    "anthropic/claude-haiku-5-5": Price("Claude Haiku 5.5", 0.10, 0.50,
                                        f"{_CTX_PRICES} (prompts up to 100K tokens); PILOT-RUNBOOK.md"),
    "anthropic/claude-sonnet-5": Price("Claude Sonnet 5", 3.0, 15.0,
                                       f"{_CTX_PRICES} (budgeted at the pass-4 steady-state $3/$15)"),
    "anthropic/claude-opus-5-5": Price("Claude Opus 5.5", 4.0, 20.0,
                                       f"{_CTX_PRICES}; calibrator role bullet; PILOT-RUNBOOK.md"),
    "openai/gpt-6-luna": Price("GPT-6 Luna", 0.10, 0.50, f"{_CTX_PRICES}; grader role bullet"),
    "openai/gpt-6.1-terra@batch": Price("GPT-6.1 Terra on Batch", 0.50, 2.50, _ADR6),
    "openai/gpt-6.1-sol@batch": Price("GPT-6.1 Sol on Batch", 1.0, 5.0, _ADR6),
    "openai/gpt-6.1-sol": Price("GPT-6.1 Sol, standard", 2.0, 10.0, _ADR6),
}

# Roster workers in report order (section 5 roster table).
WORKERS: tuple[str, ...] = (
    "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5",
    "anthropic/claude-sonnet-5", "anthropic/claude-opus-5-5",
)
CALIBRATOR = "anthropic/claude-opus-5-5"
GRADER_STEPS: tuple[tuple[str, str], ...] = (
    ("step 1", "openai/gpt-6-luna"),
    ("step 2", "openai/gpt-6.1-terra@batch"),
    ("step 3", "openai/gpt-6.1-sol@batch"),
)
SCOUT_MODEL = "openai/gpt-6-luna"


class Refused(Exception):
    """A refusal: nothing is printed as an estimate."""


# --- Reading a dev log --------------------------------------------------------------------

@dataclass(frozen=True)
class Call:
    sample_id: str
    epoch: int
    stratum: str
    input_tokens: int        # uncached + cache write + cache read: every input token billed
    output_tokens: int       # billed output, thinking included
    reasoning_tokens: int    # part of output_tokens
    cache_read_tokens: int   # part of input_tokens
    stop_reason: str | None
    max_tokens: int | None
    cache_write_tokens: int = 0  # part of input_tokens

    @property
    def visible_tokens(self) -> int:
        return max(self.output_tokens - self.reasoning_tokens, 0)


@dataclass(frozen=True)
class LogUsage:
    path: str
    model: str
    prompt_version: str | None
    prompt_sha256: str | None
    samples: int
    calls: tuple[Call, ...]
    other_model_calls: int


def stratum_of(sample_id: str) -> str:
    """Dev case ids carry their stratum: dev-hc-* (edge and adversarial) and dev-std-*."""
    if "-hc-" in sample_id:
        return "hard"
    if "-std-" in sample_id:
        return "standard"
    raise Refused(f"sample {sample_id}: no stratum in its id (expected dev-hc-* or dev-std-*)")


def check_dev_header(header: Mapping[str, Any], path: str) -> None:
    """Refuse anything but a successful dev-split log, before any sample is opened."""
    ev = header.get("eval") or {}
    if header.get("status") != "success":
        raise Refused(f"{path}: status {header.get('status')!r}, not success")
    splits = [s for s in ((ev.get("task_args") or {}).get("split"), (ev.get("metadata") or {}).get("split"))
              if s is not None]
    if not splits:
        raise Refused(f"{path}: the log records no split; refusing it rather than risk the evaluation split")
    if any(s != "dev" for s in splits):
        raise Refused(f"{path}: split {sorted(set(map(str, splits)))}, not dev; this script reads dev logs only")
    dataset = str((ev.get("dataset") or {}).get("name") or "")
    if dataset.endswith("-eval"):
        raise Refused(f"{path}: dataset {dataset!r} names the evaluation split")


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if value is not None else 0


def read_log_usage(path: str | Path) -> LogUsage:
    """Per-call usage of the evaluated model in one dev .eval log (a zip file)."""
    path = str(path)
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise Refused(f"{path}: not a readable .eval zip ({e})") from e
    with z:
        try:
            header = json.loads(z.read("header.json"))
        except KeyError as e:
            raise Refused(f"{path}: no header.json (an unfinished log?)") from e
        check_dev_header(header, path)
        ev = header["eval"]
        model = str(ev.get("model") or "")
        if model not in WORKERS:
            raise Refused(f"{path}: model {model!r} is not a roster worker ({', '.join(WORKERS)})")
        names = sorted(n for n in z.namelist() if n.startswith("samples/"))
        off_dev = [n for n in names if not n.startswith("samples/dev-")]
        if off_dev:
            raise Refused(f"{path}: sample file {off_dev[0]} is not a dev sample")
        if not names:
            raise Refused(f"{path}: no samples (sample logging off?)")
        calls: list[Call] = []
        other = 0
        for name in names:
            sample = json.loads(z.read(name))
            sid = str(sample.get("id"))
            if not sid.startswith("dev-"):
                raise Refused(f"{path}: sample {sid} is not a dev sample")
            stratum = stratum_of(sid)
            for event in sample.get("events") or ():
                if event.get("event") != "model":
                    continue
                if event.get("model") != model:
                    other += 1
                    continue
                output = event.get("output") or {}
                usage = output.get("usage") or {}
                request = (event.get("call") or {}).get("request") or {}
                choices = output.get("choices") or [{}]
                calls.append(Call(
                    sample_id=sid, epoch=int(sample.get("epoch") or 0), stratum=stratum,
                    input_tokens=(_usage_int(usage, "input_tokens") + _usage_int(usage, "input_tokens_cache_write")
                                  + _usage_int(usage, "input_tokens_cache_read")),
                    output_tokens=_usage_int(usage, "output_tokens"),
                    reasoning_tokens=_usage_int(usage, "reasoning_tokens"),
                    cache_read_tokens=_usage_int(usage, "input_tokens_cache_read"),
                    stop_reason=output.get("stop_reason") or choices[0].get("stop_reason"),
                    max_tokens=request.get("max_tokens"),
                    cache_write_tokens=_usage_int(usage, "input_tokens_cache_write"),
                ))
    if not calls:
        raise Refused(f"{path}: no model call of {model} carries usage")
    meta = ev.get("metadata") or {}
    return LogUsage(path=path, model=model, prompt_version=meta.get("worker_prompt_version"),
                    prompt_sha256=meta.get("worker_prompt_sha256"),
                    samples=len(names), calls=tuple(calls), other_model_calls=other)


# --- Arithmetic ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def eval_generations() -> dict[str, int]:
    return {s: n * EPOCHS for s, n in EVAL_CASES.items()}


def weighted_mean(calls: Sequence[Call], attr: str) -> float:
    """Per-call mean of `attr`, each stratum weighted by its evaluation-split generations."""
    gens = eval_generations()
    total = sum(gens.values())
    out = 0.0
    for stratum, n in gens.items():
        values = [getattr(c, attr) for c in calls if c.stratum == stratum]
        if not values:
            raise Refused(f"no {stratum} calls in the log: the projection needs both strata")
        out += _mean(values) * n / total
    return out


def worker_estimate(usage: LogUsage) -> dict[str, Any]:
    """Measured: the worker's expected cost on the evaluation split, and its worst case."""
    price = PRICES[usage.model]
    gens = eval_generations()
    by_stratum: dict[str, Any] = {}
    cost = 0.0
    for stratum, n in gens.items():
        calls = [c for c in usage.calls if c.stratum == stratum]
        if not calls:
            raise Refused(f"{usage.path}: no {stratum} calls; the projection needs both strata")
        per_call = _mean([price.cost(c.input_tokens, c.output_tokens) for c in calls])
        by_stratum[stratum] = {
            "dev_calls": len(calls), "eval_generations": n, "cost_per_call": per_call,
            "mean_input": _mean([c.input_tokens for c in calls]),
            "mean_output": _mean([c.output_tokens for c in calls]),
            "mean_reasoning": _mean([c.reasoning_tokens for c in calls]),
        }
        cost += per_call * n
    mean_in = weighted_mean(usage.calls, "input_tokens")
    total_gens = sum(gens.values())
    stops: dict[str, int] = {}
    for c in usage.calls:
        stops[str(c.stop_reason)] = stops.get(str(c.stop_reason), 0) + 1
    return {
        "model": usage.model, "label": price.model, "basis": "measured", "price_source": price.source,
        "log": Path(usage.path).name, "prompt_version": usage.prompt_version,
        "prompt_sha256": usage.prompt_sha256,
        "dev_samples": usage.samples, "dev_calls": len(usage.calls), "other_model_calls": usage.other_model_calls,
        "generations": total_gens, "by_stratum": by_stratum,
        "mean_input": mean_in,
        "mean_output": weighted_mean(usage.calls, "output_tokens"),
        "mean_reasoning": weighted_mean(usage.calls, "reasoning_tokens"),
        "mean_visible": weighted_mean(usage.calls, "visible_tokens"),
        "max_output": max(c.output_tokens for c in usage.calls),
        "max_tokens_sent": sorted({c.max_tokens for c in usage.calls if c.max_tokens is not None}),
        "stop_reasons": dict(sorted(stops.items())),
        "cost": cost,
        "input_cost": total_gens * price.cost(mean_in, 0),
        "cache_read_share": (sum(c.cache_read_tokens for c in usage.calls)
                             / max(sum(c.input_tokens for c in usage.calls), 1)),
        "cache_write_share": (sum(c.cache_write_tokens for c in usage.calls)
                              / max(sum(c.input_tokens for c in usage.calls), 1)),
        # Cache-write tokens per full run billed at the base input price (as in "cost").
        "cache_write_cost": total_gens * price.cost(weighted_mean(usage.calls, "cache_write_tokens"), 0),
        "worst_row_input": total_gens * price.cost(ROW_INPUT_TOKENS, MAX_TOKENS_DEFAULT),
        "worst_measured_input": total_gens * price.cost(mean_in, MAX_TOKENS_DEFAULT),
    }


def judge_call_shape(workers: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
    """Assumed input of one judge call: the mean worker input (a stand-in for the case file and
    the instructions) plus the mean visible answer it grades, over the workers measured."""
    rows = list(workers.values())
    return _mean([w["mean_input"] for w in rows]), _mean([w["mean_visible"] for w in rows])


def calibration_calls(reading: str = "upper") -> dict[str, int]:
    """Grader calls at one acceptance look on the calibration subset, under one reading.
    upper: ~800 outputs per pass (CONTEXT.md), 3 test-retest replicates on top of the first
    scoring, 4 bias-audit passes ((a) twice, (b), (c)). lower: 640 outputs per pass (40 cases x
    4 epochs x 4 workers), the first scoring counted as replicate 1, 3 bias-audit passes ((c)
    not applying)."""
    r = CALIBRATION_READINGS[reading]
    first, retest, bias = r["outputs"], r["retest_extra"] * r["outputs"], r["bias_passes"] * r["outputs"]
    return {"outputs_per_pass": r["outputs"], "first_scoring": first, "test_retest": retest,
            "bias_audit": bias, "total": first + retest + bias}


def grader_calls() -> dict[str, Any]:
    """Grader calls per full run: one per generation, plus the calibration-time work at the upper
    reading (counted in the totals; the lower reading is shown beside)."""
    cal = calibration_calls("upper")
    return {"generations": FULL_RUN_GENERATIONS, "calibration": cal,
            "calibration_lower": calibration_calls("lower"), "total": FULL_RUN_GENERATIONS + cal["total"]}


def grader_estimate(workers: Mapping[str, Mapping[str, Any]], price_key: str,
                    output_tokens: float = JUDGE_OUTPUT_TOKENS) -> dict[str, Any]:
    case_in, answer = judge_call_shape(workers)
    gc = grader_calls()
    calls = gc["total"]
    price = PRICES[price_key]
    per_call_in = case_in + answer
    per_call = price.cost(per_call_in, output_tokens)
    return {"model": price_key, "label": price.model, "calls": calls, "input_per_call": per_call_in,
            "output_per_call": output_tokens, "cost": calls * per_call, "cost_per_call": per_call,
            "generations_cost": gc["generations"] * per_call,
            "calibration_cost": gc["calibration"]["total"] * per_call,
            "calibration_lower_cost": gc["calibration_lower"]["total"] * per_call,
            "price_source": price.source}


def grader_output_threshold(workers: Mapping[str, Mapping[str, Any]], price_key: str, base: float,
                            cap: float) -> float | None:
    """Output tokens per grader call at which the total (the non-grader parts `base` plus the
    grader at `price_key`) reaches `cap`, route judge left out. None when it is over at zero output."""
    case_in, answer = judge_call_shape(workers)
    calls = grader_calls()["total"]
    price = PRICES[price_key]
    room = (cap - base) * MTOK / calls - (case_in + answer) * price.input_per_mtok
    return room / price.output_per_mtok if room >= 0 else None


def calibrator_estimate(workers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    opus = workers.get(CALIBRATOR)
    non_opus = [w for m, w in workers.items() if m != CALIBRATOR]
    if opus is None or not non_opus:
        return None
    non_opus_generations = FULL_RUN_GENERATIONS - EPOCHS * sum(EVAL_CASES.values())
    calls = round(CALIBRATOR_SHARE * non_opus_generations)
    per_call_in = opus["mean_input"] + _mean([w["mean_visible"] for w in non_opus])
    per_call_out = opus["mean_output"]
    price = PRICES[CALIBRATOR]
    alt_calls = CALIBRATION_READINGS["upper"]["outputs"]
    return {"calls": calls, "input_per_call": per_call_in, "output_per_call": per_call_out,
            "cost": calls * price.cost(per_call_in, per_call_out),
            "worst": calls * price.cost(per_call_in, MAX_TOKENS_DEFAULT), "price_source": price.source,
            "non_opus_workers_measured": len(non_opus),
            # Inferred: the calibrator's cache writes at the Opus 5.5 worker's measured write share.
            "cache_write_cost": calls * price.cost(per_call_in * opus["cache_write_share"], 0),
            # Not counted: the reading of "graded both ways" as grader and calibrator.
            "both_ways_calls": alt_calls, "both_ways_cost": alt_calls * price.cost(per_call_in, per_call_out)}


def scout_estimate(workers: Mapping[str, Mapping[str, Any]],
                   scan: tuple[int, int, int] | None, transcripts: int) -> dict[str, Any]:
    price = PRICES[SCOUT_MODEL]
    scale = FULL_RUN_GENERATIONS / transcripts
    if scan is not None:
        calls, tin, tout = scan
        count_scaled = price.cost(tin, tout) * scale
        # The scan reads Sonnet 5.5 transcripts. A scanner's input grows with the transcript, so
        # each worker's share of the scan input is scaled by its measured mean input plus output
        # against Sonnet 5.5's; the scanner's output is scaled by transcript count only.
        ref = workers.get("anthropic/claude-sonnet-5-5")
        if ref is not None:
            ref_len = ref["mean_input"] + ref["mean_output"]
            ratios = {m: (w["mean_input"] + w["mean_output"]) / ref_len for m, w in workers.items()}
            per_worker = FULL_RUN_GENERATIONS / len(WORKERS)
            in_tokens = sum(tin / transcripts * per_worker * r for r in ratios.values())
            in_tokens += tin / transcripts * per_worker * (len(WORKERS) - len(ratios))  # unmeasured: as Sonnet 5.5
            out_tokens = tout * scale
            cost = price.cost(in_tokens, out_tokens)
            how = (f"input scaled by each worker's measured mean input plus output against Sonnet 5.5's "
                   f"(mean ratio {_mean(list(ratios.values())):.2f}); count-scaled figure "
                   f"${count_scaled:,.2f}")
        else:
            in_tokens, out_tokens, cost = tin * scale, tout * scale, count_scaled
            how = "scaled by transcript count only: no Sonnet 5.5 log to scale the input by length"
        n = calls * scale
        return {"basis": "measured", "calls": n, "input_per_call": in_tokens / n,
                "output_per_call": out_tokens / n, "cost": cost, "count_scaled_cost": count_scaled,
                "price_source": price.source,
                "note": f"the owner's scan: {calls} calls, {tin:,} in, {tout:,} out on {transcripts} Sonnet 5.5 "
                        f"transcripts, scaled to {FULL_RUN_GENERATIONS:,} transcripts ({how})"}
    rows = list(workers.values())
    per_in = _mean([w["mean_input"] + w["mean_output"] for w in rows])
    calls = SCOUT_MODEL_SCANNERS * FULL_RUN_GENERATIONS
    return {"basis": "placeholder", "calls": calls, "input_per_call": per_in,
            "output_per_call": JUDGE_OUTPUT_TOKENS, "cost": calls * price.cost(per_in, JUDGE_OUTPUT_TOKENS),
            "price_source": price.source,
            "note": "not measured: waits on the owner's scan of the Sonnet 5.5 pilot log (runbook step 10); "
                    "2 model scanners x 3,200 transcripts, each read as the worker's input plus its whole "
                    "output, 800 out"}


def cap_status(total: float) -> str:
    if total > HARD_CAP:
        return "over the $300 hard cap"
    if total > SOFT_CAP:
        return "over the $150 soft cap, under the $300 hard cap"
    return "under the $150 soft cap"


def estimate(usages: Sequence[LogUsage], scan: tuple[int, int, int] | None = None,
             transcripts: int = PILOT_TRANSCRIPTS) -> dict[str, Any]:
    seen: dict[str, str] = {}
    for u in usages:
        if u.model in seen:
            raise Refused(f"two logs of {u.model}: {Path(seen[u.model]).name} and {Path(u.path).name}")
        seen[u.model] = u.path
    if not usages:
        raise Refused("no log given")
    workers = {m: worker_estimate(u) for m in WORKERS for u in usages if u.model == m}
    missing = [m for m in WORKERS if m not in workers]
    worker_cost = sum(w["cost"] for w in workers.values())
    calibrator = calibrator_estimate(workers)
    graders = {step: grader_estimate(workers, key) for step, key in GRADER_STEPS}
    sol_standard = grader_estimate(workers, "openai/gpt-6.1-sol")
    scout = scout_estimate(workers, scan, transcripts)
    base = worker_cost + (calibrator["cost"] if calibrator else 0.0) + scout["cost"]
    totals = {}
    for step, g in (*graders.items(), ("step 3 at standard price (reference only)", sol_standard)):
        t = base + g["cost"]
        lower = t - g["calibration_cost"] + g["calibration_lower_cost"]
        none = t - g["calibration_cost"]
        totals[step] = {"grader": g["label"], "without_route_judge": t, "with_route_judge": t + ROUTE_JUDGE_HIGH,
                        "status_without": cap_status(t), "status_with": cap_status(t + ROUTE_JUDGE_HIGH),
                        "calibration_lower_reading": lower, "status_lower": cap_status(lower),
                        "without_calibration": none, "status_none": cap_status(none)}
    # Cache writes: every input token is billed at base price above. If a cache write costs a
    # premium over base, the Claude input rises by (premium - 1) x the cache-write cost.
    write_cost = sum(w["cache_write_cost"] for w in workers.values()) + (
        calibrator["cache_write_cost"] if calibrator else 0.0)
    cache_write = {
        "cost_at_base": write_cost,
        "premiums": {f"{p:g}": {"extra": (p - 1) * write_cost,
                                "step 1 total": totals["step 1"]["without_route_judge"] + (p - 1) * write_cost}
                     for p in CACHE_WRITE_PREMIUMS},
    }
    worst_workers_row = sum(w["worst_row_input"] for w in workers.values())
    worst_workers_measured = sum(w["worst_measured_input"] for w in workers.values())
    worst_cal = calibrator["worst"] if calibrator else 0.0
    sensitivity = {
        str(out): {step: grader_estimate(workers, key, out)["cost"] for step, key in
                   (*GRADER_STEPS, ("step 3 standard", "openai/gpt-6.1-sol"))}
        for out in SENSITIVITY_JUDGE_OUTPUTS
    }
    thresholds = {
        step: {f"{cap:.0f}": grader_output_threshold(workers, key, base, cap) for cap in (SOFT_CAP, HARD_CAP)}
        for step, key in (*GRADER_STEPS, ("step 3 standard", "openai/gpt-6.1-sol"))
    }
    return {
        "complete": not missing and calibrator is not None,
        "missing_workers": missing,
        "workers": workers, "worker_cost": worker_cost,
        "calibrator": calibrator, "graders": graders, "sol_standard": sol_standard,
        "grader_calls": grader_calls(), "scout": scout,
        "route_judge": {"low": ROUTE_JUDGE_LOW, "high": ROUTE_JUDGE_HIGH, "worst": ROUTE_JUDGE_WORST},
        "totals": totals,
        "worst": {
            "workers_row_input": worst_workers_row, "workers_measured_input": worst_workers_measured,
            "calibrator": worst_cal, "route_judge": ROUTE_JUDGE_WORST,
            "total_row_input": worst_workers_row + worst_cal + ROUTE_JUDGE_WORST + graders["step 1"]["cost"]
            + scout["cost"],
            "total_measured_input": worst_workers_measured + worst_cal + ROUTE_JUDGE_WORST
            + graders["step 1"]["cost"] + scout["cost"],
        },
        "grader_output_sensitivity": sensitivity,
        "grader_output_thresholds": thresholds,
        "cache_write": cache_write,
        "caps": {"soft": SOFT_CAP, "hard": HARD_CAP},
    }


# --- Report -------------------------------------------------------------------------------

def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _tok(x: float) -> str:
    return f"{x:,.0f}"


def format_report(est: Mapping[str, Any]) -> str:
    L: list[str] = []
    L.append("# Cost per full run, re-derived from the dev logs")
    L.append("")
    L.append("A full run: 4 workers x 200 evaluation cases x 4 epochs = 3,200 generations, with the grader, "
             "calibrator and Scout scanner calls on them. Dev logs only; no model call was made.")
    if not est["complete"]:
        L.append("")
        L.append(f"INCOMPLETE: no dev log for {', '.join(est['missing_workers']) or 'none'}"
                 f"{'' if est['calibrator'] else '; the calibrator needs the Opus 5.5 log'}. "
                 "The totals below leave those parts out.")
    L.append("")
    L.append("## Workers (measured)")
    L.append("")
    L.append("| Worker | Dev log | Prompt | Dev calls | Mean input | Mean output (thinking incl.) | Mean thinking "
             "| Max output | Stops | Cost per full run |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|")
    for w in est["workers"].values():
        stops = ", ".join(f"{k} {v}" for k, v in w["stop_reasons"].items())
        prompt = f"{w['prompt_version']} {str(w['prompt_sha256'] or '')[:12]}"
        L.append(f"| {w['label']} | {w['log']} | {prompt} | {w['dev_calls']} | {_tok(w['mean_input'])} "
                 f"| {_tok(w['mean_output'])} | {_tok(w['mean_reasoning'])} | {_tok(w['max_output'])} | {stops} "
                 f"| {_usd(w['cost'])} |")
    L.append(f"| All workers | | | | | | | | | {_usd(est['worker_cost'])} |")
    L.append("")
    L.append("Means are per call, each stratum weighted by its evaluation-split share (584 standard and 216 hard "
             "generations per worker). Cost per full run is the mean dev cost per call of each stratum times its "
             "evaluation generations.")
    L.append("")
    L.append("## Every component (expected per full run)")
    L.append("")
    L.append("| Component | Basis | Calls | Input per call | Output per call | Cost |")
    L.append("|---|---|---:|---:|---:|---:|")
    for w in est["workers"].values():
        L.append(f"| {w['label']} worker | measured | {w['generations']:,} | {_tok(w['mean_input'])} "
                 f"| {_tok(w['mean_output'])} | {_usd(w['cost'])} |")
    c = est["calibrator"]
    if c:
        L.append(f"| Calibrator, Opus 5.5 | inferred | {c['calls']:,} | {_tok(c['input_per_call'])} "
                 f"| {_tok(c['output_per_call'])} | {_usd(c['cost'])} |")
    else:
        L.append("| Calibrator, Opus 5.5 | not derived (no Opus 5.5 log) | | | | |")
    g = est["graders"]["step 1"]
    gc = est["grader_calls"]
    cal = gc["calibration"]
    L.append(f"| Grader, GPT-6 Luna, on the {gc['generations']:,} generations | assumed | {gc['generations']:,} "
             f"| {_tok(g['input_per_call'])} | {_tok(g['output_per_call'])} | {_usd(g['generations_cost'])} |")
    for key, label in (("first_scoring", "calibration subset, first scoring (CONTEXT.md full-run list)"),
                       ("test_retest", "test-retest replicates (CONTEXT.md full-run list)"),
                       ("bias_audit", "bias-audit passes (same look; not on the full-run list)")):
        L.append(f"| Grader, GPT-6 Luna, {label} | assumed, upper reading | {cal[key]:,} "
                 f"| {_tok(g['input_per_call'])} | {_tok(g['output_per_call'])} "
                 f"| {_usd(cal[key] * g['cost_per_call'])} |")
    s = est["scout"]
    L.append(f"| Scout scanners, GPT-6 Luna | {s['basis']} | {s['calls']:,.0f} | {_tok(s['input_per_call'])} "
             f"| {_tok(s['output_per_call'])} | {_usd(s['cost'])} |")
    rj = est["route_judge"]
    L.append(f"| Route judge, GPT-6 Luna, if it runs | stated (section 5) | | | | {_usd(rj['low'])} to "
             f"{_usd(rj['high'])} |")
    L.append("")
    L.append(f"Scout: {s['note']}.")
    L.append("")
    lo = gc["calibration_lower"]
    L.append(f"Calibration-time work at one acceptance look on the 40-case calibration subset, upper reading "
             f"(counted): {cal['outputs_per_pass']:,} outputs per pass (CONTEXT.md \"~800 graded both ways\"), "
             f"{cal['test_retest'] // cal['outputs_per_pass']} test-retest replicates on top of the first scoring, "
             f"{cal['bias_audit'] // cal['outputs_per_pass']} bias-audit passes, {cal['total']:,} grader calls. "
             f"Lower reading: {lo['outputs_per_pass']:,} outputs per pass (40 cases x 4 epochs x 4 workers), the "
             f"first scoring as replicate 1, {lo['bias_audit'] // lo['outputs_per_pass']} bias-audit passes, "
             f"{lo['total']:,} calls. \"Both ways\" is read as by hand and by the grader (the Autograder "
             "calibration bullet), so the hand grades are human time. The intra-rater check is human time "
             "only. Neither is in the dollar total.")
    if c:
        L.append("")
        L.append(f"Not counted: if \"both ways\" meant grader and calibrator, the {c['both_ways_calls']:,} "
                 f"calibration-subset outputs would add {c['both_ways_calls']:,} Opus 5.5 calls, "
                 f"{_usd(c['both_ways_cost'])}.")
    L.append("")
    L.append("## Totals against the caps ($150 soft, $300 hard)")
    L.append("")
    L.append("| Grader step | Grader | Total | Status | With route judge (high end) | Status |")
    L.append("|---|---|---:|---|---:|---|")
    for step, t in est["totals"].items():
        L.append(f"| {step} | {t['grader']} | {_usd(t['without_route_judge'])} | {t['status_without']} "
                 f"| {_usd(t['with_route_judge'])} | {t['status_with']} |")
    L.append("")
    L.append("A grader step replaces GPT-6 Luna on every grader call; it does not add to it. The totals hold "
             "the calibration-time work at the upper reading. The same totals at the lower reading, and with no "
             "calibration-time work:")
    L.append("")
    L.append("| Grader step | Lower reading | Status | No calibration-time work | Status |")
    L.append("|---|---:|---|---:|---|")
    for step, t in est["totals"].items():
        L.append(f"| {step} | {_usd(t['calibration_lower_reading'])} | {t['status_lower']} "
                 f"| {_usd(t['without_calibration'])} | {t['status_none']} |")
    L.append("")
    L.append("## Grader cost by output tokens per call (sensitivity)")
    L.append("")
    sens = est["grader_output_sensitivity"]
    steps = list(next(iter(sens.values())).keys())
    L.append("| Output tokens per grader call | " + " | ".join(steps) + " |")
    L.append("|---:|" + "---:|" * len(steps))
    for out, row in sens.items():
        L.append(f"| {int(out):,} | " + " | ".join(_usd(row[s_]) for s_ in steps) + " |")
    L.append("")
    L.append("Output tokens per grader call at which the total reaches each cap (route judge left out):")
    L.append("")
    L.append("| Grader step | Reaches $150 at | Reaches $300 at |")
    L.append("|---|---:|---:|")
    for step_, caps in est["grader_output_thresholds"].items():
        cells = ["over already" if v is None else f"{v:,.0f} tokens" for v in (caps["150"], caps["300"])]
        L.append(f"| {step_} | {cells[0]} | {cells[1]} |")
    L.append("")
    L.append("## Worst case at max_tokens 48,000 (no pin is set)")
    L.append("")
    L.append("| Worker | At 3,482 input (cost row) | At measured mean input |")
    L.append("|---|---:|---:|")
    for w in est["workers"].values():
        L.append(f"| {w['label']} | {_usd(w['worst_row_input'])} | {_usd(w['worst_measured_input'])} |")
    wc = est["worst"]
    L.append(f"| All workers | {_usd(wc['workers_row_input'])} | {_usd(wc['workers_measured_input'])} |")
    if c:
        L.append(f"| Calibrator at 48,000 out | {_usd(wc['calibrator'])} | {_usd(wc['calibrator'])} |")
    L.append(f"| Route judge worst (stated) | {_usd(wc['route_judge'])} | {_usd(wc['route_judge'])} |")
    L.append(f"| Grader step 1 and Scout, expected (no max_tokens known) | "
             f"{_usd(est['graders']['step 1']['cost'] + est['scout']['cost'])} | "
             f"{_usd(est['graders']['step 1']['cost'] + est['scout']['cost'])} |")
    L.append(f"| Total | {_usd(wc['total_row_input'])} | {_usd(wc['total_measured_input'])} |")
    L.append("")
    L.append("The worst case is one attempt per generation: --retry-on-error 2 can bill an errored sample-epoch "
             "up to three times, and a cancelled streamed generation is billed.")
    L.append("")
    L.append("## Prices (per million tokens, in / out) and where they are pinned")
    L.append("")
    L.append("| Model | In | Out | Source |")
    L.append("|---|---:|---:|---|")
    for p in PRICES.values():
        L.append(f"| {p.model} | {_usd(p.input_per_mtok)} | {_usd(p.output_per_mtok)} | {p.source} |")
    L.append("")
    in_cost = sum(w["input_cost"] for w in est["workers"].values()) + (
        est["calibrator"]["calls"] * PRICES[CALIBRATOR].cost(est["calibrator"]["input_per_call"], 0)
        if est["calibrator"] else 0.0)
    shares = ", ".join(f"{w['label']} {100 * w['cache_read_share']:.0f} and "
                       f"{100 * w['cache_write_share']:.0f} percent"
                       for w in est["workers"].values())
    cw = est["cache_write"]
    L.append("Every input token, cached or not, is billed at the base input price: no cache price is pinned. "
             f"Claude input (workers and calibrator) is {_usd(in_cost)} of the step 1 total. Cache reads and "
             f"cache writes are these shares of dev input tokens: {shares}. Reads are usually billed below base "
             "and writes above it. The figures are an upper estimate on input only if the read discount outweighs "
             "the write premium; neither price is pinned, so they may not be an upper bound.")
    L.append("")
    L.append(f"Sensitivity (cache writes at a stated premium, reads still at base): cache writes are "
             f"{_usd(cw['cost_at_base'])} of the Claude input at base price.")
    L.append("")
    L.append("| Cache-write price, times base input (assumed, not pinned) | Added | Step 1 total |")
    L.append("|---:|---:|---:|")
    for p, row in cw["premiums"].items():
        L.append(f"| {p} | {_usd(row['extra'])} | {_usd(row['step 1 total'])} |")
    return "\n".join(L)


# --- CLI ----------------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse's own exit 2 is not a contract code
        self.print_usage(sys.stderr)
        print(f"REFUSED: {message}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


class _HelpAction(argparse.Action):
    """-h/--help prints usage and exits 3: argparse's own exit 0 would read as an estimate."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any) -> None:
        super().__init__(option_strings, dest=dest, default=argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        parser.print_help()
        print("REFUSED: help requested; no estimate ran", file=sys.stderr)
        sys.exit(EXIT_REFUSED)


def _scan(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not CALLS,INPUT,OUTPUT") from e
    if len(values) != 3 or any(v < 0 for v in values) or values[0] == 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not CALLS,INPUT,OUTPUT with CALLS above 0")
    return values  # type: ignore[return-value]


def parser() -> argparse.ArgumentParser:
    p = _Parser(description="Cost per full run from dev-log per-call tokens (no model call).", add_help=False)
    p.add_argument("-h", "--help", action=_HelpAction, help="show this help and exit 3 (no estimate ran)")
    p.add_argument("logs", nargs="+", help="dev .eval logs, one per roster worker")
    p.add_argument("--scout-scan", type=_scan, default=None, metavar="CALLS,INPUT,OUTPUT",
                   help="the owner's Scout scan totals on the pilot log (scanner model calls and tokens); "
                        "without it the Scout line is a placeholder")
    p.add_argument("--scout-transcripts", type=int, default=PILOT_TRANSCRIPTS,
                   help=f"transcripts the scan covered (default {PILOT_TRANSCRIPTS})")
    p.add_argument("--json", dest="json_out", help="also write the estimate as JSON to this path")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.scout_transcripts <= 0:
            raise Refused("--scout-transcripts must be above 0")
        est = estimate([read_log_usage(p) for p in args.logs], args.scout_scan, args.scout_transcripts)
        text = format_report(est)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(est, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:  # an unreadable input must never read as an estimate
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(text)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
