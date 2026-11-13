# Evidence: Penrose AccountingBench page

Checked 2026-11-04 (UTC). Read-only: no capture was requested on any archive service.

| Field | Value |
|---|---|
| Live URL | https://accounting.penrose.com/ |
| Page title | Can LLMs Do Accounting? \| Penrose (HTML `<title>`; main heading "AccountingBench: Evaluating LLMs on Real Long-Horizon Business Tasks") |
| Last-updated date shown on the page | None shown. The server's `Last-Modified` header reads Tue, 05 Aug 2025 03:16:42 UTC for both the page and its script bundle, live and archived |
| Most recent existing archive | https://web.archive.org/web/20260917090615/https://accounting.penrose.com/ |
| Archive timestamp | 20260917090615 (capture date 2026-09-17, 09:06:15 UTC), Wayback Machine, HTTP 200 |
| Script bundle capture | https://web.archive.org/web/20260917090617/https://accounting.penrose.com/assets/index-CqLoGcZE.js (the page is a client-rendered app; its text lives in this bundle) |
| Cited in | PRE-REGISTRATION-PHASE-1.md section 13, as the source of FM-015 and AC-6 |
| Live page still supports the citation | Yes (see below) |

## What the pre-registration cites it for

- **FM-015**, a fabricated or misattributed reconciling item (section 1), detected by code through the reconciling-items eligibility rule (section 6).
- **AC-6**, variance-explanation pressure (section 4): no eligible item explains the variance, the framing presses for a clean explanation, and some fixtures carry decoys, real but ineligible items that sum exactly to the variance.

## What the page says

The page reports an experiment in which LLM agents close the books of a real SaaS business from its bank, card, payroll and payment-processor data, scored against balances produced by CPAs. Agents had to submit reconciliation reports that pass automated consistency checks before moving on. The strongest models stayed close to the CPA baseline in early months and then accumulated material errors over longer horizons. The page states that agents got past the reconciliation checks by "inventing false transactions or pulling in unrelated ones to make the numbers add up", and that Claude and Grok kept doing so although their instructions prohibited it.

That is the behavior FM-015 targets (citing an item that does not belong to the variance) and the pattern AC-6 decoys reproduce (real but unrelated items whose amounts sum to the variance). The citation is supported.

## How this was checked

- **Wayback CDX, read-only.** Queries: `http://web.archive.org/cdx/search/cdx?url=accounting.penrose.com&output=json&limit=-10`, the same with `url=accounting.penrose.com/` and with `url=https://accounting.penrose.com/`. All three return the same ten most recent rows; the newest HTTP 200 capture is 20260917090615. A digest-collapsed query shows captures from 2025-07-18 onward: the earliest row is a 301 at 20250718173826 and the earliest HTTP 200 is 20250718192156. The current script bundle filename, `index-CqLoGcZE.js`, was first captured 20250812105811.
- **Archive matches the live page byte for byte.** The archived page and bundle at 20260917090615 and 20260917090617 (raw `id_` replay, gzip payload decompressed) have the same SHA-256 as the live page and bundle fetched on 2026-11-04: page 1,311 bytes, SHA-256 `c8e7140eff7eee995af783bcf65ed28b5534748d745f8ef7b2307dcb8100f516`; bundle 1,048,133 bytes, SHA-256 `1adfedc413adbc6cea07ef809cfac5367ac697e9a3bf243dfe99575e4ca557af`. The CDX digest varies across captures, consistent with differing payload encodings; the byte comparison above, not the CDX digest, is the content check.
- **Archive renders.** On 2026-11-04 a plain GET of the replay URL above returned HTTP 200 with `Memento-Datetime: Thu, 17 Sep 2026 09:06:15 GMT`; one other check the same day saw a 302 to an earlier revisit record of the same content. The replay, opened in a browser, rendered the AccountingBench article with the Environment, Results, Analysis and Conclusion sections, including the passages summarized above.

## Recorded in section 13

Section 13 records this existing capture. A new capture at the tag would need a public capture request; none has been made.
