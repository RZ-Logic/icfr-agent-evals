# icfr-agent-evals

**A pre-registered, LLM-as-judge evaluation harness for measuring whether an AI agent is safe to be granted _action authority_ over financial controls** — not merely whether it writes plausible text.

Scope is deliberately narrow: crypto reserve reconciliation (a single asset's ledger liabilities vs. on-chain reserves for a reporting period). Severity is anchored to the PCAOB AS 2201 deficiency / significant-deficiency / material-weakness hierarchy. The unit of measurement is the **decision and its disclosure**, evaluated against a deterministic policy that serves as the answer key.

> **Status: in active development (v0.0.x skeleton).** The harness spine, the deterministic content anchors, and their tests run today. Calibration, robustness, the LLM-graded sub-dimensions, the action/disposition layer, the full dataset, and the pre-registration document are scheduled, not yet built. **No headline results are published here yet, by design — the public write-up is gated on completed, pre-registered runs.** A negative result is a valid and reportable outcome; the contribution is the measurement method, not a particular score.

## Design commitments

- **Consequential failures are checked deterministically, not by a model.** Wrong/hallucinated figure, fabricated counterparty, omitted required element — these are scored in plain code (`scoring.py`), unit-tested with no API key. The auditor's question is not "is the judge calibrated?" but "can bad grading hide a material weakness?" If the consequential checks are code, the answer is no.
- **Subjective sub-dimensions are LLM-graded but calibrated.** Narrative grounding, coverage adequacy, and rationale quality are graded by a model only after agreement against a human-labeled subset clears pre-set thresholds (Cohen's κ / ICC). They are not wired yet, and are not silently defaulted to a passing value in the meantime.
- **The generator never authors the reference it is graded against.** Synthetic samples emit structured ground-truth facts only; the reference narrative used for subjective grading is authored independently, to avoid a circular eval.
- **Reproducibility is non-negotiable.** Seeds and dependency versions are pinned; the worker model is a *documented pin*, never a default-to-latest alias.

## Layout

```
icfr_agent_evals/
  policy.py     deterministic policy: asset tiers + materiality bands (the answer key)
  scoring.py    pure deterministic scoring logic (no Inspect, no LLM) — unit-tested
  scorers.py    thin Inspect @scorer adapters around scoring.py
  solver.py     the worker (agent-under-eval) system prompt; version-pinned
  dataset.py    seeded synthetic dataset; ground truth in sample metadata
  rubric.py     the five-dimension rubric: what is wired vs. scheduled
  task.py       the Inspect task wiring dataset + worker + anchors
tests/          boundary-precise unit tests for policy, scoring, dataset
scripts/
  smoke_mock.py end-to-end plumbing proof using a mock model (no API key)
```

## Quickstart

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt

pytest                            # deterministic checks — runs with no API key
python scripts/smoke_mock.py      # end-to-end graph proof with a mock model

# real run (requires an Anthropic API key and a verified, pinned model string):
inspect eval icfr_agent_evals/task.py@variance_commentary --model <pinned-model>
```

## Roadmap (abridged)

- **Phase 1** — content eval: deterministic anchors (done) → Sample-Consistency calibration → robustness under input perturbation → calibrated LLM-graded sub-dimensions → prompt lock → pre-registration.
- **Phase 2** — action authority: the 9-cell (asset-tier × materiality) disposition matrix, action-correctness scoring with a hard-fail gate on any material-weakness-tier under-escalation, and CI regression gating.

## License

Apache-2.0.
