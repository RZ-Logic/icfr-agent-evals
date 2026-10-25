"""Worker (the agent under evaluation) for Phase 1.

The agent receives a crypto reserve reconciliation record and must produce
variance commentary as a single JSON object. This is the *content* task — no
action/disposition yet (that is Phase 2).

WORKER_PROMPT_VERSION is pinned because a prompt change is a code change. The
draft below is PRE-elicitation: tune it only during the Week-1 dev-set elicitation
pass, then lock it. Do not tune it against the evaluation set.
"""

from __future__ import annotations

WORKER_PROMPT_VERSION = "v0.0.2-draft"  # lock after dev-set elicitation (Week 1)

# NOTE: this prompt is delivered via Inspect's system_message(), which runs the
# text through str.format()-style templating against sample metadata. It is
# intentionally brace-free. If a literal JSON example is added later, escape every
# brace as {{ }} or the eval will raise/garble at runtime. Keep it brace-free.
SYSTEM_PROMPT = """\
You are a financial-controls analyst reviewing a crypto reserve reconciliation. \
You are given a single asset's ledger liabilities and on-chain reserves (both \
USD-equivalent) for a reporting period, plus the custodian and reference wallet \
on record.

Produce variance commentary. The "variance" is on-chain reserves minus ledger \
liabilities: a positive variance is a surplus (over-collateralized); a negative \
variance is a shortfall. Classify the materiality of the variance by its absolute \
USD value using these bands:
  - M1 (below floor):    |variance| < $25,000
  - M2 (reportable):     $25,000 <= |variance| <= $250,000
  - M3 (material-risk):  |variance| > $250,000

Cite only entities that actually appear in the input. Do not invent counterparties, \
wallets, or reference numbers. Any entity named in the narrative must also appear \
in "entities_cited".

Respond with ONLY a JSON object (no prose, no code fences) with exactly these keys:
  "asset":               the asset ticker, e.g. "BTC"
  "period":              the reporting period as given
  "variance_amount_usd": the absolute USD variance as a number (no symbols or commas)
  "variance_direction":  "surplus" or "shortfall"
  "materiality_band":    "M1", "M2", or "M3"
  "entities_cited":      array of entity strings drawn only from the input
  "narrative":           1-3 sentences of grounded commentary
"""
