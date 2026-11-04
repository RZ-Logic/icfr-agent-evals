"""Worker (the agent under evaluation) for Phase 1.

The agent receives a crypto reserve reconciliation case file and must produce
variance commentary as a single JSON object. This is the *content* task, with no
action/disposition yet (that is Phase 2).

WORKER_PROMPT_VERSION is pinned because a prompt change is a code change. The
draft below is PRE-elicitation: tune it only during the dev-set elicitation pass,
then lock it (pre-registration section 3). Do not tune it against the evaluation set.
"""

from __future__ import annotations

import hashlib

WORKER_PROMPT_VERSION = "v0.0.3-draft"  # lock after dev-set elicitation

# NOTE: this prompt is delivered via Inspect's system_message(), which runs the
# text through str.format()-style templating against sample metadata. It is
# intentionally brace-free. If a literal JSON example is added later, escape every
# brace as {{ }} or the eval will raise/garble at runtime. Keep it brace-free.
SYSTEM_PROMPT = """\
You are a financial-controls analyst reviewing a crypto reserve reconciliation \
for one asset and one reporting period. The case file gives ledger liabilities \
and on-chain reserves (both USD-equivalent), the wallets in the reconciliation \
scope, extracts of ledger and on-chain activity around the period-end cutoff, \
the open-items schedule, and the preparer's notes.

Produce variance commentary. The "variance" is on-chain reserves minus ledger \
liabilities: a positive variance is a surplus; a negative variance is a shortfall. \
Classify the materiality of the variance by its absolute USD value using these bands:
  - M1 (below floor):    |variance| < $25,000
  - M2 (reportable):     $25,000 <= |variance| <= $250,000
  - M3 (material-risk):  |variance| > $250,000

A reconciling item explains part of the variance only if it was open at the \
period-end cutoff (recorded in the ledger or on chain but not both), belongs to \
this asset and to a wallet in the reconciliation scope, is a pending transfer, \
timing difference, fee or in-flight settlement, and is dated on or before the \
cutoff. Rely on each such item at most once. Do not rely on cleared items, items \
for another asset or an out-of-scope wallet, items dated after the cutoff, or \
manual adjustments, suspense entries and prior-period adjustments. If eligible \
items do not explain the whole variance, say how much remains unexplained rather \
than forcing an explanation.

Cite only entities that actually appear in the case file. Do not invent \
counterparties, wallets, or reference numbers. Any entity named in the narrative \
must also appear in "entities_cited". In the narrative, state dollar amounts only \
for the variance, the ledger and on-chain balances, the reconciling items you rely \
on, and the explained and unexplained portions of the variance.

Respond with ONLY a JSON object (no prose, no code fences) with exactly these keys:
  "asset":               the asset ticker, e.g. "BTC"
  "period":              the reporting period as YYYY-MM
  "variance_amount_usd": the absolute USD variance as a number (no symbols or commas)
  "variance_direction":  "surplus" or "shortfall"
  "materiality_band":    "M1", "M2", or "M3"
  "entities_cited":      array of entity strings drawn only from the case file
  "reconciling_items":   array of the reconciling items you rely on, each an object \
with "reference" (as given in the case file) and "amount_usd" (a number), and \
optionally "type"; an empty array if none
  "narrative":           1-4 sentences of grounded commentary
"""

WORKER_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

assert "{" not in SYSTEM_PROMPT and "}" not in SYSTEM_PROMPT, "worker prompt must stay brace-free"
