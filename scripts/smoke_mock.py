"""End-to-end plumbing proof with a mock model — runs with NO API key.

Purpose: prove the whole graph executes and the deterministic anchors discriminate
*through Inspect*, not just in unit tests. We craft two worker outputs:

  - sample A: fully correct      -> expect correctness 3, faithfulness 3, completeness 3
  - sample B: correct but cites a fabricated custodian
                                  -> expect correctness 3, completeness 3, faithfulness 0

Inspect executes samples concurrently, so we do NOT rely on output ordering: the
mock is a callable keyed on each sample's unique reference wallet, so the intended
output always reaches the intended sample.

Run: python scripts/smoke_mock.py
"""

from __future__ import annotations

import json
import re
import sys

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from icfr_agent_evals.dataset import build_dataset
from icfr_agent_evals.task import variance_commentary


def _wallet_in(text: str, wallets: list[str]) -> str | None:
    for w in wallets:
        if w in text:
            return w
    return None


def _answer_for(truth: dict, *, fabricate: bool) -> str:
    valid = list(truth["valid_entities"])
    cited = [valid[0]] + (["Phantom Custody Group"] if fabricate else [])
    return json.dumps(
        {
            "asset": truth["asset"],
            "period": truth["period"],
            "variance_amount_usd": abs(truth["variance_usd"]),
            "variance_direction": truth["direction"],
            "materiality_band": truth["materiality_band"],
            "entities_cited": cited,
            "narrative": "Reserves reconciled against ledger liabilities for the period.",
        }
    )


def main() -> int:
    ds = build_dataset(split="eval", seed=20260607)
    a, b = ds.samples[0], ds.samples[1]

    wallet_a = a.metadata["truth"]["valid_entities"][1]
    wallet_b = b.metadata["truth"]["valid_entities"][1]

    # wallet -> crafted completion: A correct, B fabricates a custodian
    answer_by_wallet = {
        wallet_a: _answer_for(a.metadata["truth"], fabricate=False),
        wallet_b: _answer_for(b.metadata["truth"], fabricate=True),
    }

    def mock_outputs(messages, tools, tool_choice, config) -> ModelOutput:
        blob = "\n".join(getattr(m, "text", "") or "" for m in messages)
        w = _wallet_in(blob, list(answer_by_wallet))
        content = answer_by_wallet.get(w, '{"narrative": "unmatched sample"}')
        return ModelOutput.from_content(model="mockllm/model", content=content)

    model = get_model("mockllm/model", custom_outputs=mock_outputs)

    logs = inspect_eval(
        variance_commentary(split="eval", epochs=1),
        model=model,
        limit=2,
        epochs=1,
        display="plain",
    )
    log = logs[0]

    if log.status != "success":
        print(f"EVAL STATUS: {log.status}")
        if log.error:
            print(log.error)
        return 1

    print("\n=== per-sample anchor scores (mock model) ===")
    expectations = {
        wallet_a: {"correctness_anchor": 3, "faithfulness_anchor": 3, "completeness_anchor": 3},
        wallet_b: {"correctness_anchor": 3, "faithfulness_anchor": 0, "completeness_anchor": 3},
    }
    ok = True
    for s in log.samples:
        blob = str(s.input)
        which = "A (correct)" if wallet_a in blob else "B (fabricated entity)" if wallet_b in blob else "?"
        got = {name: sc.value for name, sc in s.scores.items()}
        print(f"sample {s.id}  [{which}] -> {got}")
        w = wallet_a if wallet_a in blob else wallet_b if wallet_b in blob else None
        for name, expected in expectations.get(w, {}).items():
            if got.get(name) != expected:
                ok = False
                print(f"   MISMATCH {name}: expected {expected}, got {got.get(name)}")

    print("\nSMOKE RESULT:", "PASS — graph executed and anchors discriminated" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
