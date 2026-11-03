"""Sample Consistency end to end with a mock model and epochs = 4. NO API key.

Pre-registration section 6 (Calibration) and section 2 H4: the deterministic
consistency measurement runs on every (sample, epoch) record of a real Inspect
run. Eight dev-split cases (five synthetic, three difficulty-bearing) each get a
planned multiset of four worker outputs, one per epoch:

  c correct, b wrong band, d wrong direction, a wrong amount, p prose (format break)

Epochs run concurrently, so only the multiset per case is controllable, not which
epoch receives which output. The mock is keyed on each case's first in-scope
wallet, as in smoke_mock.py, and counts calls per case. Every mock output carries
usage, so Inspect never downloads a tokenizer. The dev split only; the evaluation
split is never built.

Run: python scripts/consistency_mock.py [log_dir]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals.casefile import BANDS, CaseTruth, truth_of
from icfr_agent_evals.consistency import ConsistencyReport, assert_gate, format_report, report_from_log
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.task import variance_commentary

K = 4
DEFAULT_LOG_DIR = os.path.join(tempfile.gettempdir(), "icfr-agent-evals-logs", "consistency")


@dataclass(frozen=True)
class Plan:
    name: str
    subset: str            # "synthetic" or "difficulty_bearing"
    outputs: str           # one letter per epoch call: c, b, d, a or p
    agreement: float
    modal_correct: bool


# Ties go to the incorrect class (consistency.modal_class), so cc+dd is wrong at 0.50.
PLANS: tuple[Plan, ...] = (
    Plan("S1 four correct", "synthetic", "cccc", 1.00, True),
    Plan("S2 three correct, one wrong band", "synthetic", "cccb", 0.75, True),
    Plan("S3 two correct, two wrong direction (tie)", "synthetic", "ccdd", 0.50, False),
    Plan("S4 four identical wrong amounts", "synthetic", "aaaa", 1.00, False),
    Plan("S5 four format breaks", "synthetic", "pppp", 0.25, False),
    Plan("H1 two correct, one wrong band, one wrong direction", "difficulty_bearing", "ccbd", 0.50, True),
    Plan("H2 two wrong band, one correct, one format break", "difficulty_bearing", "bbcp", 0.50, False),
    Plan("H3 four different answers (tie)", "difficulty_bearing", "cbda", 0.25, False),
)


def answer(truth: CaseTruth, kind: str) -> str:
    """The worker output for one planned letter."""
    if kind == "p":
        return "Here is my commentary: the reserves look fine for the period."
    eligible = truth.eligible_items()
    direction = truth.direction
    band = truth.materiality_band
    amount = abs(truth.variance_usd)
    if kind == "b":
        band = BANDS[(BANDS.index(band) + 1) % len(BANDS)]
    elif kind == "d":
        direction = "surplus" if direction == "shortfall" else "shortfall"
    elif kind == "a":
        amount = round(amount + 5_000.0, 2)
    elif kind != "c":
        raise ValueError(f"unknown planned output {kind!r}")
    return json.dumps({
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": amount,
        "variance_direction": direction,
        "materiality_band": band,
        "entities_cited": [i.reference for i in eligible],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd} for i in eligible],
        "narrative": "Reserves were compared with ledger liabilities for the period.",
    })


def pick_cases() -> dict[str, Plan]:
    """Sample id -> plan: the first dev-split cases of each subset, in id order,
    whose mock key (first in-scope wallet) appears in no other dev case."""
    samples = sorted(build_dataset(split="dev"), key=lambda s: str(s.id))
    truths = {s.id: truth_of(s.metadata) for s in samples}

    def unique_key(s) -> bool:
        key = truths[s.id].scope_wallets[0]
        return not any(key in str(o.input) for o in samples if o.id != s.id)

    pools = {
        "synthetic": [s for s in samples if truths[s.id].case_kind == "standard" and unique_key(s)],
        "difficulty_bearing": [s for s in samples if truths[s.id].case_kind != "standard" and unique_key(s)],
    }
    picked: dict[str, Plan] = {}
    for plan in PLANS:
        s = pools[plan.subset].pop(0)
        picked[str(s.id)] = plan
    return picked


def run(log_dir: str = DEFAULT_LOG_DIR):
    """(EvalLog, ConsistencyReport, sample id -> Plan) for the planned mock run."""
    picked = pick_cases()
    dataset = build_dataset(split="dev")
    truths = {str(s.id): truth_of(s.metadata) for s in dataset if str(s.id) in picked}
    queue = {truths[sid].scope_wallets[0]: [answer(truths[sid], c) for c in plan.outputs] for sid, plan in picked.items()}
    calls: dict[str, int] = dict.fromkeys(queue, 0)

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in queue if w in blob), None)
        if key is None or calls[key] >= K:
            raise RuntimeError(f"unplanned model call (key {key!r})")
        content = queue[key][calls[key]]
        calls[key] += 1
        out = ModelOutput.from_content(model="mockllm/model", content=content)
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    task = variance_commentary(split="dev", epochs=K)
    task.dataset = task.dataset.filter(lambda s: str(s.id) in picked)
    log = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=mock_outputs),
        model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
        log_dir=log_dir,
        display="none",
    )[0]
    if log.status != "success":
        raise RuntimeError(f"eval status {log.status}: {log.error}")
    if any(n != K for n in calls.values()):
        raise RuntimeError(f"calls per case {calls}, expected {K} each")
    report = report_from_log(log)
    assert_gate(report, expected_samples=len(picked), expected_k=K)
    return log, report, picked


def check(report: ConsistencyReport, picked: dict[str, Plan]) -> list[str]:
    """Mismatches between the report and the plan; empty when they agree."""
    problems: list[str] = []
    for sid, plan in picked.items():
        s = report.sample(sid)
        if abs(s.agreement - plan.agreement) > 1e-9 or s.modal_correct != plan.modal_correct or s.subset != plan.subset:
            problems.append(
                f"{plan.name} ({sid}): agreement {s.agreement}, modal correct {s.modal_correct}, subset {s.subset}"
            )
    expected_fm007 = sorted(sid for sid, p in picked.items() if p.agreement == 1.0 and not p.modal_correct)
    if sorted(report.fm007_candidates) != expected_fm007:
        problems.append(f"FM-007 candidates {list(report.fm007_candidates)} != {expected_fm007}")
    return problems


def main() -> int:
    log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    log, report, picked = run(log_dir)
    print("planned multisets:")
    for sid, plan in sorted(picked.items()):
        print(f"  {sid}: {plan.outputs} ({plan.name})")
    print()
    print(format_report(report))
    problems = check(report, picked)
    for p in problems:
        print("MISMATCH", p)
    print(f"\nlog: {log.location}")
    print("CONSISTENCY RESULT:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
