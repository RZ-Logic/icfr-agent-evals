"""Build data/eval/handwritten.jsonl: the 54 hand-written cases with their answer keys.

Pre-registration section 4: "Hand-written" means the author writes each case's
scenario; the case's balances, items and arithmetic may be built and checked by code
(decision D-R13, 2026-11-11). This script is that code. It makes no model call and
reads no synthetic evaluation-split case.

Inputs:
  --casebook   the case texts as the author wrote them (an HTML file of
               <details class="case" id="case-NNN"><pre>...</pre></details> blocks);
  --categories a JSON object, case number -> {"case_kind", "category"}: the author's
               list of edge and AC cases;
  --judgments  a JSON object, case number -> the judgment calls code cannot make (the
               edge category, memo-only items, the AC-1 injection, extra organization
               names, decoys), each fixed from exact quotes of the case text.

Every derived figure (variance, direction, band, ratio, item eligibility and effect,
explained and unexplained amounts, evidence state) comes from the package's own
functions (casefile.item_eligibility, item_effect_usd, evidence_state_for,
policy.materiality_band), never typed by hand. Each case must pass validate_truth,
the generator's entity-closure check and the wallet-label table check, a perfect
answer built from its key must score 3 on every anchor with no FM code, and the set
must hold the section 4 composition; otherwise nothing is written (exit 3).

Prints counts only, never a key value: the rater labels the cases twice with the
answer key withheld (Branch B, decision D-R18).

Run: python scripts/build_handwritten.py --casebook C --categories CAT --judgments J [--out data/eval/handwritten.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from icfr_agent_evals import scoring  # noqa: E402
from icfr_agent_evals.casefile import (  # noqa: E402
    ASSET_CHAIN, SCHEMA_VERSION, TYPE_LABELS, CaseTruth, InjectionSpec, ReconcilingItem, TruthError,
    evidence_state_for, item_effect_usd, item_eligibility, period_cutoff_date, validate_truth,
)
from icfr_agent_evals.generator import _PERSON_NAME_RE, check_entity_closure, entity_tokens  # noqa: E402
from icfr_agent_evals.handwritten import HANDWRITTEN_FILE, composition_gaps  # noqa: E402
from icfr_agent_evals.policy import asset_tier, materiality_band  # noqa: E402

EXIT_OK, EXIT_REFUSED = 0, 3
TEMPLATE_ID = "handwritten-v1"
CASE_RE = re.compile(r'<details class="case" id="case-(\d+)"><summary>Case \d+</summary><pre>(.*?)</pre></details>', re.S)

LABEL_TO_TYPE = {lab.lower(): typ for typ, labs in TYPE_LABELS.items() for lab in labs}
BOOKED = {"both": "both", "both sides": "both", "chain only": "on_chain", "on-chain": "on_chain",
          "ledger only": "ledger", "ledger": "ledger"}
DIRS = {"in": "inflow", "out": "outflow"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], start=1)}
FULL_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
               "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}


def iso_date(text: str) -> str | None:
    """A date as written in any of the casebook forms, to YYYY-MM-DD; None when blank."""
    s = text.strip()
    if not s:
        return None
    s = s.split(" ")[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError(f"unreadable date {text!r}")


def iso_ts(text: str) -> str:
    s = text.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%m/%d/%Y %H:%M",
                "%Y-%m-%d %H:%M", "%d-%b-%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    raise ValueError(f"unreadable timestamp {text!r}")


def period_of(text: str) -> str:
    s = text.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        return s
    a, b = s.split()
    mon = FULL_MONTHS.get(a.lower()) or MONTHS[a.lower()[:3]]
    return f"{int(b):04d}-{mon:02d}"


def money(text: str) -> str | None:
    """A USD amount as written, to a plain decimal string; None when blank."""
    s = text.strip().replace("USD", "").replace("$", "").replace(",", "").strip()
    if not s:
        return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        raise ValueError(f"unreadable amount {text!r}")
    return s


def units(text: str) -> str | None:
    s = text.strip()
    if not s:
        return None
    s = re.sub(r"\s+[A-Z]{3,4}$", "", s).replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        raise ValueError(f"unreadable units {text!r}")
    return s


def cells(line: str) -> list[str]:
    return [p.strip() for p in line.strip().split("|")]


def parse_case(num: str, text: str) -> dict:
    warn: list[str] = []
    lines = text.splitlines()
    out: dict = {"case": num, "warnings": warn}
    m = re.match(r"Asset: (\S+) \((\w+)\)\s+Period: (.*?)\s{2,}Cutoff: (.*?) UTC", lines[1])
    asset, chain, period, cutoff = m.group(1), m.group(2), period_of(m.group(3)), m.group(4)
    out["header"] = {"entity": lines[0].split(" - ")[0].strip(), "asset": asset, "chain": chain, "period": period,
                     "cutoff_as_written": cutoff, "cutoff_ts": iso_ts(cutoff)}
    if ASSET_CHAIN.get(asset) != chain:
        warn.append(f"chain {chain} is not the contract chain for {asset}")
    m = re.match(r"Workpaper: (\S+)\s+Prepared by: (.*?) on (\S+)\s+Reviewer:(.*)$", lines[2])
    out["header"].update({"workpaper": m.group(1), "prepared_by": m.group(2), "prepared_on": iso_date(m.group(3)),
                          "reviewer": m.group(4).strip()})
    out["header"]["status"] = lines[3].split(":", 1)[1].strip() if lines[3].startswith("Status") else ""

    sec = None
    wallets, oi, ledger, chain_lines, notes, tail = [], [], [], [], [], []
    bal: dict = {}
    sec_titles: dict = {}
    for ln in lines[4:]:
        low = ln.strip().lower()
        if not ln.strip():
            if sec in ("wallets", "oi", "ledger", "chain", "notes"):
                sec = "after" if sec == "notes" else None
            continue
        if not ln.startswith(" "):
            if low.startswith("balances at cutoff"):
                sec = "bal"; continue
            if low.startswith("wallet scope"):
                sec = "wallets"; sec_titles["wallets"] = ln; continue
            if low.startswith("open items schedule"):
                sec = "oi"; continue
            if low.startswith("ledger extract"):
                sec = "ledger"; sec_titles["ledger"] = ln; continue
            if low.startswith("chain activity extract") or low.startswith("on-chain extract"):
                sec = "chain"; sec_titles["chain"] = ln; continue
            if low.startswith("preparer notes"):
                sec = "notes"; continue
            tail.append(ln); continue
        if sec == "bal":
            s = ln.strip()
            mm = re.match(r"(Ledger liabilities|On-chain reserves)[^:]*:\s+(\S+)\s+([A-Z]+)\s+(.*)$", s)
            if mm:
                key = "ledger" if mm.group(1).startswith("Ledger") else "onchain"
                bal[key + "_units"] = units(mm.group(2))
                bal[key + "_usd_as_written"] = mm.group(4).strip()
                bal[key + "_usd"] = money(mm.group(4)) if mm.group(4).strip() else None
            elif re.match(r"(Ledger liabilities|On-chain reserves)[^:]*:\s+\S+$", s):
                # USD shown with no units column
                key = "ledger" if s.startswith("Ledger") else "onchain"
                val = s.split(":", 1)[1].strip()
                bal[key + "_units"] = None
                bal[key + "_usd_as_written"] = val
                bal[key + "_usd"] = money(val)
                warn.append(f"balance {key} shows USD only, no units")
            elif s.startswith("Mark:"):
                mk = re.match(r"Mark:\s*(\S*)", s)
                bal["mark"] = (mk.group(1) or "").replace(",", "") or None
                bal["mark_as_written"] = s
            elif s.startswith("Difference"):
                bal["difference_as_written"] = s.split(":", 1)[1].strip()
            else:
                bal.setdefault("other", []).append(s)
        elif sec == "wallets":
            c = cells(ln)
            if c[0] == "label":
                continue
            if c[0].lower().startswith("total in scope"):
                bal["total_in_scope_as_written"] = ln.strip()
                continue
            while len(c) < 6:
                c.append("")
            wallets.append({"label": c[0], "address": c[1], "held_by": c[2], "in_scope_as_written": c[3],
                            "in_scope": c[3].lower().startswith("yes"), "units": units(c[4]) if c[4] else None,
                            "usd": money(c[5]) if c[5] else None})
        elif sec == "oi":
            c = cells(ln)
            if c[0] == "ref":
                continue
            while len(c) < 10:
                c.append("")
            row = {"ref": c[0], "type_label": c[1], "asset": c[2], "wallet_as_written": c[3],
                   "date_as_written": c[4], "booked_in": c[5], "dir": c[6], "status": c[7],
                   "usd_as_written": c[8], "comment": " | ".join(c[9:]).strip()}
            row["type"] = LABEL_TO_TYPE.get(c[1].lower())
            if row["type"] is None:
                warn.append(f"{c[0]}: unknown type label {c[1]!r}")
            row["recorded_in"] = BOOKED.get(c[5].lower())
            if row["recorded_in"] is None:
                warn.append(f"{c[0]}: unknown booked-in {c[5]!r}")
            row["flow"] = DIRS.get(c[6].lower())
            row["open_status"] = c[7].lower() if c[7].lower() in ("open", "cleared") else None
            if row["open_status"] is None:
                warn.append(f"{c[0]}: unknown status {c[7]!r}")
            try:
                row["date"] = iso_date(c[4])
            except ValueError as e:
                warn.append(f"{c[0]}: {e}")
                row["date"] = "UNREADABLE"
            try:
                row["amount_usd"] = money(c[8])
            except ValueError as e:
                warn.append(f"{c[0]}: {e}")
                row["amount_usd"] = "UNREADABLE"
            # "label address": the address is the last token when it looks like one; a label
            # may hold spaces ("Hot 1"). A cell with no address keeps the whole cell as label.
            wl = c[3].split()
            if len(wl) > 1 and re.fullmatch(r"(0x[0-9a-fA-F]{40}|bc1[0-9a-z]{20,90}|addr1[0-9a-z]{20,110}|r[1-9A-HJ-NP-Za-km-z]{24,34}|[1-9A-HJ-NP-Za-km-z]{32,44})", wl[-1]):
                row["wallet_label"], row["wallet_address"] = " ".join(wl[:-1]), wl[-1]
            else:
                row["wallet_label"], row["wallet_address"] = c[3].strip(), ""
            oi.append(row)
        elif sec == "ledger":
            c = cells(ln)
            if c[0] == "posted":
                continue
            while len(c) < 6:
                c.append("")
            desc = c[2]
            cp = desc.split(" / ", 1)[1].strip() if " / " in desc else ""
            ledger.append({"ts": iso_ts(c[0]), "ref": c[1], "desc": desc, "counterparty": cp, "wallet": c[3],
                           "units": units(c[4]), "usd": money(c[5]), "extra": c[6:]})
        elif sec == "chain":
            c = cells(ln)
            if c[0].startswith("time"):
                continue
            while len(c) < 8:
                c.append("")
            chain_lines.append({"ts": iso_ts(c[0]), "tx": c[1], "wallet": c[2], "dir": c[3], "units": units(c[4]),
                                "usd": money(c[5]), "ref": c[6], "fee": c[7], "extra": c[8:]})
        elif sec == "notes":
            s = ln.strip()
            mm = re.match(r"- (\S+ \S+) (.*?): (.*)$", s)
            if mm:
                notes.append({"ts_as_written": mm.group(1), "by": mm.group(2), "text": mm.group(3)})
            elif notes:
                notes[-1]["text"] += "\n" + s
            else:
                notes.append({"ts_as_written": "", "by": "", "text": s})
        else:
            tail.append(ln)
    out.update({"balances": bal, "wallets": wallets, "open_items": oi, "ledger_lines": ledger,
                "chain_lines": chain_lines, "notes": notes, "request_lines": tail, "section_titles": sec_titles})
    for k in ("ledger_usd", "onchain_usd"):
        if bal.get(k) is None:
            warn.append(f"balance {k} missing")
    return out


def cents(s: str | None) -> int | None:
    if s is None:
        return None
    return int((Decimal(s) * 100).to_integral_value())


def case_id(num: str) -> str:
    return f"eval-hw-{num}"


def build(num: str, p: dict, kind: str, category: str | None, j: dict) -> tuple[CaseTruth, dict, list[str]]:
    gaps: list[str] = []
    h, bal = p["header"], p["balances"]
    asset, chain, period = h["asset"], h["chain"], h["period"]
    cutoff = period_cutoff_date(period)
    if p["header"]["cutoff_ts"][:10] != cutoff:
        gaps.append(f"cutoff {p['header']['cutoff_ts']} is not the period end {cutoff}")
    lc, oc = cents(bal.get("ledger_usd")), cents(bal.get("onchain_usd"))
    if lc is None or oc is None:
        raise TruthError(f"{num}: a balance in USD is missing")
    v = oc - lc
    label_to_addr = {w["label"]: w["address"] for w in p["wallets"]}
    scope = tuple(w["address"] for w in p["wallets"] if w["in_scope"] and w["address"])
    if any(w["in_scope"] and not w["address"] for w in p["wallets"]):
        gaps.append("an in-scope wallet shows no address")

    tx_by_ref: dict[str, list[str]] = {}
    for ln in p["chain_lines"]:
        if ln["ref"]:
            tx_by_ref.setdefault(ln["ref"], []).append(ln["tx"])

    items: list[ReconcilingItem] = []
    overrides = {(o["ref"], o["field"]): o["value"] for o in j.get("row_overrides", [])}
    for r in p["open_items"]:
        f = {
            "reference": r["ref"], "type": r["type"], "asset": r["asset"],
            "wallet": r["wallet_address"] or label_to_addr.get(r["wallet_label"], r["wallet_label"]),
            "date": r["date"], "open_status": r["open_status"], "recorded_in": r["recorded_in"],
            "flow": r["flow"],
            "amount_usd": float(Decimal(r["amount_usd"])) if r["amount_usd"] not in (None, "UNREADABLE") else None,
            "channel": "structured", "aliases": tuple(t for t in tx_by_ref.get(r["ref"], []) if t),
            "description": r["comment"],
        }
        for (ref, field_name), value in overrides.items():
            if ref == r["ref"]:
                f[field_name] = value
        items.append(ReconcilingItem(**f))
    for m in j.get("memo_only_items", []):
        items.append(ReconcilingItem(
            reference=m["reference"], type=m["type"], asset=m["asset"], wallet=m["wallet"], date=m["date"],
            open_status=m["open_status"], recorded_in=m["recorded_in"], flow=m["flow"],
            amount_usd=m["amount_usd"], channel="memo_only",
            aliases=tuple(m.get("aliases") or tx_by_ref.get(m["reference"], [])), description=m.get("basis", ""),
        ))
    decoys = set(j.get("decoys", []))
    final = []
    for it in items:
        el = item_eligibility(it, asset=asset, scope_wallets=scope, cutoff_date=cutoff)
        role = "explaining" if el.eligible else ("decoy" if it.reference in decoys else "noise")
        final.append(ReconcilingItem(**{**it.to_dict(), "aliases": tuple(it.aliases), "role": role}))
    eligible = [i for i in final if item_eligibility(i, asset=asset, scope_wallets=scope, cutoff_date=cutoff).eligible]
    explained_c = sum(int(round(item_effect_usd(i) * 100)) for i in eligible)
    unexplained_c = v - explained_c

    held = Counter(w["held_by"] for w in p["wallets"] if w["in_scope"] and w["held_by"] and w["held_by"] != "self-hosted")
    custodian = j.get("custodian") or (held.most_common(1)[0][0] if held else "")
    orgs: dict[str, None] = {}
    for w in p["wallets"]:
        if w["held_by"] and w["held_by"] != "self-hosted":
            orgs[w["held_by"]] = None
    for ln in p["ledger_lines"]:
        if ln["counterparty"]:
            orgs[ln["counterparty"]] = None
    for o in j.get("extra_organizations", []):
        orgs[o] = None
    orgs[h["entity"]] = None
    text = p["text"]
    organizations = tuple(o for o in orgs if o and o in text)

    valid: dict[str, None] = {}
    for t in entity_tokens(text):
        valid[t] = None
    for mm in _PERSON_NAME_RE.finditer(text):
        valid[mm.group(0)] = None
    for o in organizations:
        valid[o] = None
    for w in p["wallets"]:
        if w["label"]:
            valid[w["label"]] = None
    for it in final:
        valid[it.reference] = None
        for a in it.aliases:
            valid[a] = None
    for e in j.get("extra_entities", []):
        valid[e] = None

    inj = None
    if j.get("injection"):
        d = dict(j["injection"])
        if d.get("instruction_text") and d["instruction_text"] not in text:
            gaps.append("injection instruction_text is not an exact substring of the case text")
        if d.get("canary") and d["canary"] not in text:
            gaps.append("injection canary is not in the case text")
        tf = d.get("target_field")
        band, direction = materiality_band(v / 100), ("surplus" if v > 0 else "shortfall")
        d["truth_value"] = {
            None: None, "materiality_band": band, "variance_direction": direction,
            "variance_amount_usd": abs(v) / 100,
            "reconciling_items": [i.reference for i in eligible],
            "entities_cited": [],
        }.get(tf, d.get("truth_value"))
        inj = InjectionSpec(**d)

    truth = CaseTruth(
        case_id=case_id(num), split="eval", case_kind=kind, category=category,
        template_id=TEMPLATE_ID,
        template_instance=hashlib.sha256(f"{TEMPLATE_ID}|{num}|{text}".encode("utf-8")).hexdigest(),
        asset=asset, asset_tier=asset_tier(asset), chain=chain, period=period, cutoff_date=cutoff,
        ledger_liabilities_usd=lc / 100, onchain_reserves_usd=oc / 100, variance_usd=v / 100,
        direction="surplus" if v > 0 else "shortfall", materiality_band=materiality_band(v / 100),
        reserve_ratio=(oc / 100) / (lc / 100), custodian=custodian, organizations=organizations,
        scope_wallets=scope, valid_entities=tuple(valid), items=tuple(final),
        explained_usd=explained_c / 100, unexplained_usd=unexplained_c / 100,
        evidence_state=evidence_state_for(explained_c / 100, unexplained_c / 100, len(eligible)),
        has_memo_only_link=any(i.channel == "memo_only" for i in eligible), injection=inj,
        notes={"handwritten": True, "casebook_case": num,
               "decoy_reasons": {i.reference: list(item_eligibility(i, asset=asset, scope_wallets=scope,
                                                                     cutoff_date=cutoff).reasons)
                                 for i in final if i.role == "decoy"},
               "memo_only_refs": [i.reference for i in final if i.channel == "memo_only"],
               "judgment_notes": j.get("notes", "")},
    )
    fixture = {
        "header": {"entity": h["entity"], "asset": asset, "chain": chain, "period": period,
                   "cutoff": f"{cutoff}T23:59:59Z", "workpaper": h["workpaper"], "prepared_by": h["prepared_by"],
                   "prepared_on": h["prepared_on"], "reviewer": h["reviewer"], "status": h["status"], "notes": []},
        "balances": {"ledger_cents": lc, "onchain_cents": oc, "ledger_units": bal.get("ledger_units"),
                     "onchain_units": bal.get("onchain_units"), "mark": bal.get("mark")},
        "wallets": [{"label": w["label"], "address": w["address"], "held_by": w["held_by"], "in_scope": w["in_scope"],
                     "note": w["in_scope_as_written"], "balance_cents": cents(w["usd"]),
                     "balance_units": w["units"]} for w in p["wallets"]],
        "open_items": [{"ref": r["ref"], "type": r["type"], "type_label": r["type_label"], "asset": r["asset"],
                        "wallet_label": r["wallet_label"], "wallet": r["wallet_address"], "date": r["date"],
                        "status": r["open_status"], "recorded_in": r["recorded_in"], "flow": r["flow"],
                        "amount_cents": cents(r["amount_usd"]) if r["amount_usd"] not in (None, "UNREADABLE") else None,
                        "comment": r["comment"]} for r in p["open_items"]],
        "ledger_extract": {"lines": [{"ts": ln["ts"], "ref": ln["ref"], "wallet": ln["wallet"], "asset": asset,
                                      "counterparty": ln["counterparty"], "desc": ln["desc"], "units": ln["units"],
                                      "usd_cents": cents(ln["usd"])} for ln in p["ledger_lines"]]},
        "onchain_extract": {"lines": [{"ts": ln["ts"], "tx": ln["tx"], "ref": ln["ref"], "wallet": ln["wallet"],
                                       "asset": asset, "dir": ln["dir"], "units": ln["units"],
                                       "usd_cents": cents(ln["usd"]), "fee_units": ln["fee"]}
                                      for ln in p["chain_lines"]]},
        "notes": [{"ts": n["ts_as_written"], "by": n["by"], "text": n["text"]} for n in p["notes"]],
        "request": p["request_lines"],
    }
    return truth, fixture, gaps


def perfect_output(t: CaseTruth) -> dict:
    """A correct, schema-valid answer built from the key: every eligible item cited once
    with its amount, the custodian cited, a narrative with no amounts."""
    eligible = t.eligible_items()
    return {
        "asset": t.asset, "period": t.period, "variance_amount_usd": abs(t.variance_usd),
        "variance_direction": t.direction, "materiality_band": t.materiality_band,
        "entities_cited": [t.custodian] + [i.reference for i in eligible],
        "reconciling_items": [{"reference": i.reference, "amount_usd": i.amount_usd, "type": i.type} for i in eligible],
        "narrative": ("On-chain reserves fell short of ledger liabilities for the period." if t.direction == "shortfall"
                      else "On-chain reserves exceeded ledger liabilities for the period."),
    }


def case_gaps(t: CaseTruth, fixture: dict, text: str) -> list[str]:
    """validate_truth, entity closure, the wallet-label table check and the perfect answer."""
    gaps: list[str] = []
    for check in (lambda: validate_truth(t), lambda: check_entity_closure(text, t)):
        try:
            check()
        except TruthError as e:
            gaps.append(str(e))
    try:
        scoring.check_wallet_labels_in_table(
            [SimpleNamespace(id=t.case_id, metadata={"truth": t.to_dict(), "fixture": fixture})])
    except AssertionError as e:
        gaps.append(str(e))
    out = perfect_output(t)
    for name, fn in (("correctness", scoring.check_correctness), ("fabrication", scoring.check_fabrication),
                     ("reconciling items", scoring.check_reconciling_items),
                     ("faithfulness", scoring.check_faithfulness), ("completeness", scoring.check_completeness)):
        a = fn(out, t)
        if a.points != 3 or a.failure_modes or a.review_reasons:
            gaps.append(f"{t.case_id}: a perfect answer does not score clean on {name}")
    if t.injection is not None:
        a = scoring.check_injection(out, t, json.dumps(out))
        if a.failure_modes:
            gaps.append(f"{t.case_id}: a perfect answer fires FM-006")
    return gaps


def build_all(casebook: str, categories: dict, judgments: dict) -> tuple[list[dict], list[str]]:
    """Every case of the casebook as a JSON Lines record (id, input, metadata), sorted by id,
    and every gap found. A case that cannot be built is a gap; any gap refuses the file."""
    found = CASE_RE.findall(casebook)
    records, gaps = [], []
    if sorted(n for n, _ in found) != sorted(categories):
        gaps.append("the casebook cases and the category list differ")
    for num, raw in found:
        text = html.unescape(raw)
        try:
            p = parse_case(num, text)
            p["text"] = text
            cat = categories[num]
            kind, category = cat["case_kind"], cat.get("category")
            j = judgments.get(num, {})
            if kind == "edge":
                category = j.get("edge_category") or category
            truth, fixture, more = build(num, p, kind, category, j)
        except Exception as e:  # any case that cannot be built refuses the whole file
            gaps.append(f"case {num}: {type(e).__name__}: {e}")
            continue
        gaps.extend(f"case {num}: {g}" for g in more)
        gaps.extend(case_gaps(truth, fixture, text))
        records.append({"id": truth.case_id, "input": text,
                        "metadata": {"schema_version": SCHEMA_VERSION, "split": "eval", "truth": truth.to_dict(),
                                     "fixture": fixture}})
    gaps.extend(composition_gaps([CaseTruth.from_dict(r["metadata"]["truth"]) for r in records]))
    return sorted(records, key=lambda r: r["id"]), gaps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--casebook", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out", default=str(HANDWRITTEN_FILE))
    a = ap.parse_args(argv)
    records, gaps = build_all(Path(a.casebook).read_text(encoding="utf-8"),
                              json.loads(Path(a.categories).read_text(encoding="utf-8")),
                              json.loads(Path(a.judgments).read_text(encoding="utf-8")))
    if gaps:
        print(f"refused: {len(gaps)} gap(s); nothing written", file=sys.stderr)
        for g in gaps[:20]:
            print("  " + re.sub(r"-?\d[\d,]*\.\d\d", "<amount>", g), file=sys.stderr)
        return EXIT_REFUSED
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    out.write_bytes(body.encode("utf-8"))
    print(f"wrote {len(records)} cases to {out} (SHA-256 {hashlib.sha256(body.encode('utf-8')).hexdigest()})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
