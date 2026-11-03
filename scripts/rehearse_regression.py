"""Offline rehearsal of the pilot's deliberate-regression check. NO API key.

Pre-registration section 13 (pilot row) asks for one deliberate regression caught
end to end on the dev split by a local run of the paired tests of section 10. This
script runs the 20-case dev pilot (variance_commentary(split="dev", subset="pilot"),
epochs = 4) three times through mockllm and compares them with
scripts/compare_runs.py:

  baseline    a correct answer per case, built from the case's answer key;
  regression  a deliberately degraded worker: on most cases the answer states the
              wrong variance direction (FM-003, Correctness 0), and on several cases
              it also cites a fabricated counterparty (FM-005, Faithfulness 0). It
              must block through section 10(a) and 10(b): exit code 1;
  no-change   a second correct run. It must neither block nor raise a sign-off
              case: exit code 0.

Every mock output is a fixed function of its case, so epoch order does not matter
and every mock output carries usage, so Inspect never downloads a tokenizer. The
dev split only; the evaluation split is never built.

A deterministic mock proves the plumbing and the gate arithmetic, not statistical
power; the A/A estimate of section 10 measures the gate's operating characteristics.

Run: python scripts/rehearse_regression.py [log_dir]
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals.casefile import CaseTruth, truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.task import variance_commentary

K = 4
SPLIT, SUBSET = "dev", "pilot"
FABRICATED = "Phantom Custody Group"
FLIP_EVERY = 5        # every case except each 5th one states the wrong direction (16 of 20)
FABRICATE_EVERY = 2   # every 2nd case also cites a fabricated counterparty (10 of 20)
DEFAULT_LOG_DIR = os.path.join(tempfile.gettempdir(), "icfr-agent-evals-logs", "rehearsal")
COMPARE_SCRIPT = Path(__file__).resolve().parent / "compare_runs.py"

Worker = Callable[[CaseTruth, int], str]


def _flip(direction: str) -> str:
    return "surplus" if direction == "shortfall" else "shortfall"


def _answer(truth: CaseTruth, *, direction: str, fabricate: bool) -> str:
    eligible = truth.eligible_items()
    narrative = (
        "On-chain reserves fell short of ledger liabilities for the period."
        if direction == "shortfall"
        else "On-chain reserves exceeded ledger liabilities for the period."
    )
    return json.dumps({
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": direction,
        "materiality_band": truth.materiality_band,
        "entities_cited": [truth.custodian, *(i.reference for i in eligible), *([FABRICATED] if fabricate else [])],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in eligible],
        "narrative": narrative,
    })


def correct_worker(truth: CaseTruth, index: int) -> str:
    return _answer(truth, direction=truth.direction, fabricate=False)


def degraded_worker(truth: CaseTruth, index: int) -> str:
    """Wrong direction on every case but each FLIP_EVERY-th; a fabricated
    counterparty on every FABRICATE_EVERY-th case. index is the case's position
    in sample-id order."""
    flip = index % FLIP_EVERY != FLIP_EVERY - 1
    return _answer(truth, direction=_flip(truth.direction) if flip else truth.direction,
                   fabricate=index % FABRICATE_EVERY == 0)


def degraded_plan(ids: list[str]) -> dict[str, tuple[bool, bool]]:
    """sample id -> (direction flipped, fabricated), as degraded_worker decides."""
    return {sid: (i % FLIP_EVERY != FLIP_EVERY - 1, i % FABRICATE_EVERY == 0) for i, sid in enumerate(ids)}


def pilot_truths() -> dict[str, CaseTruth]:
    """sample id -> answer key for the 20 pilot cases, in sample-id order."""
    samples = sorted(build_dataset(split=SPLIT, subset=SUBSET), key=lambda s: str(s.id))
    return {str(s.id): truth_of(s.metadata) for s in samples}


def mock_keys(truths: dict[str, CaseTruth]) -> dict[str, str]:
    """sample id -> a scope wallet that appears in that case's rendered input and in
    no other pilot case's, so the mock can tell the concurrent samples apart."""
    inputs = {str(s.id): str(s.input) for s in build_dataset(split=SPLIT, subset=SUBSET)}
    keys: dict[str, str] = {}
    for sid, truth in truths.items():
        key = next((w for w in truth.scope_wallets
                    if w in inputs[sid] and not any(w in text for o, text in inputs.items() if o != sid)), None)
        if key is None:
            raise RuntimeError(f"no unique mock key for {sid}")
        keys[sid] = key
    return keys


@dataclass(frozen=True)
class Run:
    label: str
    path: str


def run_pilot(label: str, worker: Worker, log_dir: str) -> Run:
    """One epochs = 4 pilot run with the given worker; returns its log path."""
    truths = pilot_truths()
    keys = mock_keys(truths)
    outputs = {keys[sid]: worker(truth, i) for i, (sid, truth) in enumerate(truths.items())}

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in outputs if w in blob), None)
        if key is None:
            raise RuntimeError("unplanned model call: no pilot case key in the prompt")
        out = ModelOutput.from_content(model="mockllm/model", content=outputs[key])
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    task = variance_commentary(split=SPLIT, subset=SUBSET, epochs=K)
    log = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=mock_outputs),
        model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
        log_dir=os.path.join(log_dir, label),
        display="none",
    )[0]
    if log.status != "success":
        raise RuntimeError(f"{label} run status {log.status}: {log.error}")
    return Run(label, log.location)


def load_compare_runs():
    spec = importlib.util.spec_from_file_location("compare_runs", COMPARE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_cli(baseline: Run, candidate: Run, json_out: str | None = None) -> tuple[int, str]:
    """(exit code, printed report) of compare_runs.py on one candidate run."""
    cli = load_compare_runs()
    argv = ["--baseline", baseline.path, "--candidate", candidate.path]
    if json_out:
        argv += ["--json", json_out]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def rehearse(log_dir: str = DEFAULT_LOG_DIR) -> dict[str, tuple[int, str]]:
    """Runs the three pilot runs and both comparisons: name -> (exit code, report)."""
    baseline = run_pilot("baseline", correct_worker, log_dir)
    regression = run_pilot("regression", degraded_worker, log_dir)
    no_change = run_pilot("no-change", correct_worker, log_dir)
    return {
        "regression": compare_cli(baseline, regression, os.path.join(log_dir, "regression.json")),
        "no-change": compare_cli(baseline, no_change, os.path.join(log_dir, "no-change.json")),
    }


EXPECTED = {"regression": 1, "no-change": 0}


def main() -> int:
    log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    results = rehearse(log_dir)
    ok = True
    for name, (code, report) in results.items():
        print(f"=== {name} vs baseline (compare_runs.py exit {code}, expected {EXPECTED[name]}) ===")
        print(report)
        ok = ok and code == EXPECTED[name]
    print(f"logs: {log_dir}")
    print("REHEARSAL RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
