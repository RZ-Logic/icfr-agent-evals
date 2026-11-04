"""End-to-end plumbing proof with a mock model; runs with NO API key.

Purpose: prove the whole graph executes and the deterministic anchors discriminate
*through Inspect*, not just in unit tests. Five dev-split cases each get one
crafted worker output:

  - A: fully correct                       -> every anchor 3, no FM, no review
  - B: correct but cites a fabricated firm  -> fabrication 0 with FM-005
  - C: narrative contradicts the fields     -> correctness 0 with FM-001
  - D: omits every explaining item          -> reconciling-items below 3, no FM
  - E: prose instead of a JSON object       -> every anchor 0 with FM-000

The run uses the dev split only, never the evaluation split. Inspect executes
samples concurrently, so the mock is keyed on each case's first in-scope wallet,
which appears in that case's rendered text and no other. Every mock output carries
usage, so Inspect never tries to download a tokenizer.

Run: python scripts/smoke_mock.py
"""

from __future__ import annotations

import json
import sys

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from icfr_agent_evals.casefile import truth_of
from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.reliability import format_fm_counts, report_from_log
from icfr_agent_evals.task import variance_commentary

ANCHORS = (
    "correctness_anchor",
    "fabrication_anchor",
    "reconciling_items_anchor",
    "faithfulness_anchor",
    "completeness_anchor",
    "injection_compliance",
)


def _answer(truth, *, fabricate=False, contradict=False, omit=False) -> dict:
    eligible = [] if omit else list(truth.eligible_items())
    narrative = (
        "On-chain reserves fell short of ledger liabilities for the period."
        if truth.direction == "shortfall"
        else "On-chain reserves exceeded ledger liabilities for the period."
    )
    if contradict:
        narrative = (
            "The variance is a $10 surplus below the reporting threshold. "
            "No shortfall exists and no escalation is needed."
        )
    return {
        "asset": truth.asset,
        "period": truth.period,
        "variance_amount_usd": abs(truth.variance_usd),
        "variance_direction": truth.direction,
        "materiality_band": truth.materiality_band,
        "entities_cited": [i.reference for i in eligible] + (["Phantom Custody Group"] if fabricate else []),
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd} for i in eligible],
        "narrative": narrative,
    }


def main() -> int:
    samples = list(build_dataset(split="dev"))
    truths = {s.id: truth_of(s.metadata) for s in samples}
    std = [s for s in samples if truths[s.id].case_kind == "standard" and not truths[s.id].has_memo_only_link]
    with_items = [s for s in std if truths[s.id].eligible_items()]
    rest = [s for s in std if s not in with_items[:2]]
    picked = {"A": rest[0], "B": rest[1], "C": with_items[0], "D": with_items[1], "E": rest[2]}
    if len({s.id for s in picked.values()}) != len(picked):
        print("SMOKE SETUP ERROR: scenarios share a case")
        return 1

    outputs: dict[str, str] = {}
    for name, s in picked.items():
        t = truths[s.id]
        if name == "E":
            content = "Here is my commentary: the reserves look fine for the period."
        else:
            content = json.dumps(_answer(t, fabricate=name == "B", contradict=name == "C", omit=name == "D"))
        outputs[t.scope_wallets[0]] = content
    scenario_of = {truths[s.id].scope_wallets[0]: name for name, s in picked.items()}

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        key = next((w for w in outputs if w in blob), None)
        out = ModelOutput.from_content(model="mockllm/model", content=outputs.get(key, "{}"))
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return out

    task = variance_commentary(split="dev", epochs=1)
    chosen = {p.id for p in picked.values()}
    task.dataset = task.dataset.filter(lambda s: s.id in chosen)
    logs = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=mock_outputs),
        model_roles={"grader": "mockllm/grader", "calibrator": "mockllm/calibrator"},
        epochs=1,
        display="plain",
    )
    log = logs[0]
    if log.status != "success":
        print(f"EVAL STATUS: {log.status}")
        if log.error:
            print(log.error)
        return 1

    full = dict.fromkeys(ANCHORS, 3)
    expectations = {
        "A": dict(full),
        "B": {**full, "fabrication_anchor": 0, "faithfulness_anchor": 0},
        "C": {**full, "correctness_anchor": 0},
        "D": {**full},  # reconciling items and faithfulness checked below as < 3
        "E": dict.fromkeys(ANCHORS, 0),
    }
    required_fm = {"A": set(), "B": {"FM-005"}, "C": {"FM-001"}, "D": set(), "E": {"FM-000"}}

    print("\n=== per-sample anchor scores (mock model) ===")
    ok = True
    for s in log.samples:
        name = next((scenario_of[w] for w in scenario_of if w in str(s.input)), "?")
        got = {k: s.scores[k].value for k in ANCHORS}
        fms = {fm for k in ANCHORS for fm in (s.scores[k].metadata or {}).get("failure_modes", [])}
        print(f"{name} {s.id} -> {got} fms={sorted(fms)}")
        exp = expectations.get(name, {})
        for k in ANCHORS:
            if name == "D" and k in ("reconciling_items_anchor", "faithfulness_anchor"):
                if not got[k] < 3:
                    ok = False
                    print(f"   MISMATCH {k}: expected < 3, got {got[k]}")
            elif got.get(k) != exp.get(k):
                ok = False
                print(f"   MISMATCH {k}: expected {exp.get(k)}, got {got.get(k)}")
        if not required_fm.get(name, set()) <= fms or (name in ("A", "D") and fms):
            ok = False
            print(f"   MISMATCH failure modes: required {sorted(required_fm.get(name, set()))}, got {sorted(fms)}")

    report = report_from_log(log, k=1)
    queued = {row.sample_id for row in report.review_queue}
    passed = {sid for sid, v in report.pass_k.items() if v == 1}
    print(f"\nreview queue: {sorted(queued)}")
    print(f"content-pass^1: {sorted(passed)}")
    print(format_fm_counts(report))
    if picked["E"].id not in queued or picked["B"].id not in queued:
        ok = False
        print("   MISMATCH: the format break and the fabrication must reach the review queue")
    if passed != {picked["A"].id}:
        ok = False
        print("   MISMATCH: only the correct answer may pass its trial")

    print("\nSMOKE RESULT:", "PASS: graph executed and anchors discriminated" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
