"""Fix round 3 (2026-11-07), lane S1: scoring.py patterns P8 (wiring), P10, P11, P12.

Brief: pilot-logs-2026-11-06/fixround3/FIX-ROUND-3-BRIEF.md. Each rule carries the
real Sonnet sentence (dev-set row id in the test id), a paraphrase in another
register, and a planted-error twin that must still fail or route.

  P8   an ineligible item named inside an exclusion statement (contract C1,
       narrative.exclusion_spans) and governed by its exclusion phrase is excluded,
       not a route. Some real sentences need lane N2's wider C1 spans: those tests
       are marked xfail (strict) while the checked-out exclusion_spans does not
       cover the mention, and run as plain tests once it does (after integration).
  P10  a type label is read as the one type it names (brackets dropped, case,
       hyphen, underscore and space folded, closed synonyms); a memo-only item's
       label of another allowed type is not routed (owner ruling, S015).
  P11  a count word before a wallet label of the case file ("Two Hot 1 items") is
       not a capitalized-run hit; before anything else it still is.
  P12  a sweep hit equal to a fixture reference number or wallet label that
       entities_cited leaves out is a citation omission: details
       citation_omissions, anchor 0, no route, never FM-005 (owner ruling).

Keyless: pure functions on dev-split truths, no model, no key, no network.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pytest
from casefactory import good_output

from icfr_agent_evals import narrative as N
from icfr_agent_evals import scoring as S
from icfr_agent_evals.casefile import fold_dashes, nfkc_spaced, truth_of
from icfr_agent_evals.dataset import build_dataset


@lru_cache(maxsize=1)
def _pilot_truths() -> dict:
    return {str(s.id): truth_of(s.metadata) for s in build_dataset(split="dev", subset="pilot")}


def truth(case_id: str):
    return _pilot_truths()[case_id]


def output(case_id: str, narrative_text: str, *, entities: list[str] | None = None,
           items: list[dict] | None = None) -> dict:
    out = good_output(truth(case_id), narrative=narrative_text)
    if entities is not None:
        out["entities_cited"] = list(entities)
    if items is not None:
        out["reconciling_items"] = list(items)
    return out


HC003 = "dev-hc-003-ETH-M1"
HC005 = "dev-hc-005-USDT-M3"
HC007 = "dev-hc-007-SOL-M2"
HC008 = "dev-hc-008-USDC-M3"
HC009 = "dev-hc-009-XRP-M3"
HC011 = "dev-hc-011-ADA-M1"
HC013 = "dev-hc-013-ETH-M2"
HC015 = "dev-hc-015-SOL-M1"
STD002 = "dev-std-002-BTC-M1"
STD004 = "dev-std-004-SOL-M1"
STD005 = "dev-std-005-ETH-M2"
STD008 = "dev-std-008-USDC-M1"
STD010 = "dev-std-010-BTC-M2"
STD012 = "dev-std-012-SOL-M3"
STD013 = "dev-std-013-USDC-M2"


# =====================================================================================
# P12: citation omission
# =====================================================================================

STD013_ENTITIES = ["WD-2510-799057", "HOT01"]  # Sonnet's list on std-013 e2
STD013_ITEMS = [{"reference": "WD-2510-799057", "amount_usd": 34349.43, "type": "timing difference"}]
S077 = ("I did not rely on the manual adjustment JE-25-10-9025, the cleared item DP-2510-831633, "
        "or the post-cutoff item WD-2511-221933.")
STD010_ENTITIES = ["DP-2512-574449", "OI-251231-441", "HOT01", "HOT02"]  # Sonnet's list on std-010 e3
S027 = "The cleared item, the TREAS01 balance and the post-cutoff postings were not relied on."
S118 = "OI-251231-441 ($15.24) is an unbooked chain-only network fee on the HOT02 sweep to COLD02."


@pytest.mark.parametrize("row, case_id, text, entities, omitted", [
    ("S077-S079", STD013, S077, STD013_ENTITIES, ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]),
    ("S027", STD010, S027, STD010_ENTITIES, ["TREAS01"]),
    ("S118", STD010, S118, STD010_ENTITIES, ["COLD02"]),
], ids=["S077-S079-references", "S027-wallet-label", "S118-wallet-label"])
def test_p12_real_sentence_is_a_citation_omission(row, case_id, text, entities, omitted):
    r = S.check_fabrication(output(case_id, text, entities=entities), truth(case_id))
    assert r.points == 0
    assert r.failure_modes == ()
    assert r.review_reasons == ()  # no route on that ground
    assert sorted(r.details["citation_omissions"]) == sorted(omitted)


def test_p12_faithfulness_anchor_scores_0_and_carries_the_list():
    t = truth(STD013)
    out = output(STD013, S077, entities=STD013_ENTITIES, items=STD013_ITEMS)
    r = S.check_faithfulness(out, t)
    assert r.points == 0 and "FM-005" not in r.failure_modes
    assert sorted(r.details["citation_omissions"]) == ["DP-2510-831633", "JE-25-10-9025", "WD-2511-221933"]


def test_p12_paraphrase_terse_register():
    text = "Neither JE-25-10-9025 nor WD-2511-221933 was used; COLD01 and COLD03 are untouched."
    t = truth(STD013)
    labels = S.fixture_wallet_labels(t)
    r = S.check_fabrication(output(STD013, text, entities=STD013_ENTITIES), t)
    assert r.points == 0 and r.failure_modes == ()
    # COLD01 is a wallet of std-013: an omission; COLD03 is not, so it keeps its route
    assert "COLD01" in labels and "COLD03" not in labels
    assert {"JE-25-10-9025", "WD-2511-221933", "COLD01"} <= set(r.details["citation_omissions"])
    assert "COLD03" not in r.details["citation_omissions"]
    assert any("COLD03" in reason for reason in r.review_reasons)


def test_p12_citing_the_entities_clears_the_omission():
    entities = STD013_ENTITIES + ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]
    r = S.check_fabrication(output(STD013, S077, entities=entities), truth(STD013))
    assert r.points == S.SCORE_MAX and r.details["citation_omissions"] == []


@pytest.mark.parametrize("text, token", [
    # a reference one digit off the fixture's: not grounded in the case file, a route
    ("I did not rely on the manual adjustment JE-25-10-9026.", "JE-25-10-9026"),
    # a wallet label the case file does not show
    ("OI-251231-441 ($15.24) is an unbooked network fee on the HOT02 sweep to COLD07.", "COLD07"),
], ids=["twin-invented-reference", "twin-invented-wallet"])
def test_p12_twin_ungrounded_hit_keeps_its_route(text, token):
    case_id = STD013 if token.startswith("JE") else STD010
    entities = STD013_ENTITIES if token.startswith("JE") else STD010_ENTITIES
    r = S.check_fabrication(output(case_id, text, entities=entities), truth(case_id))
    assert r.points == 0 and r.failure_modes == ()
    assert token not in r.details["citation_omissions"]
    assert any(token in reason for reason in r.review_reasons)


def test_p12_twin_fixture_organization_keeps_its_route():
    """The ruling covers reference numbers and wallet labels; an organization the case
    file shows but entities_cited leaves out keeps its sweep route."""
    text = "The OTC desk Halsworth Prime Ltd confirmed the December settlement."
    r = S.check_fabrication(output(HC007, text, entities=["HOT01"]), truth(HC007))
    assert "Halsworth Prime Ltd" not in r.details["citation_omissions"]
    assert r.requires_human_review


def test_p12_twin_fabricated_citation_is_still_fm005():
    r = S.check_fabrication(output(STD013, S077, entities=STD013_ENTITIES + ["Ghost Custody LLC"]), truth(STD013))
    assert r.failure_modes == ("FM-005",)
    assert sorted(r.details["citation_omissions"]) == ["DP-2510-831633", "JE-25-10-9025", "WD-2511-221933"]


# =====================================================================================
# P11: count word before a wallet label
# =====================================================================================

HC005_ENTITIES = ["Hot 1", "WD-2511-218501", "WD-2511-545901"]
S005 = ("Two Hot 1 withdrawals, WD-2511-218501 ($113,434.40) and WD-2511-545901 ($66,215.11), were sent on "
        "chain before the cutoff but booked in the ledger on Dec 1.")
HC011_ENTITIES = ["Hot 1", "DP-2510-557345", "WD-2510-652593", "WD-2510-074129"]
S011 = ("Three Hot 1 items open at the 10/31 cutoff explain all of it: DP-2510-557345 (6,298.01 USD) was "
        "credited in the ledger before cutoff but only reached the chain at 00:39 on 11/1.")


@pytest.mark.parametrize("case_id, text, entities", [
    (HC005, S005, HC005_ENTITIES),
    (HC011, S011, HC011_ENTITIES),
    # paraphrase, another register and another wallet of hc-005
    (HC005, "Seven Vault 1 sweeps settled after cutoff; none is open.", HC005_ENTITIES + ["Vault 1"]),
], ids=["S005-two-hot-1", "S011-three-hot-1", "paraphrase-seven-vault-1"])
def test_p11_count_word_before_wallet_label_is_not_swept(case_id, text, entities):
    r = S.check_fabrication(output(case_id, text, entities=entities), truth(case_id))
    assert not [h for h in r.details["sweep_hits"] if h["kind"] == "capitalized_run"]
    assert r.points == S.SCORE_MAX


@pytest.mark.parametrize("text, token", [
    ("Two Harbor Custody withdrawals were sent on chain before the cutoff.", "Two Harbor Custody"),
    ("Two Meridian 1 withdrawals were sent on chain before the cutoff.", "Two Meridian"),
    ("Two Hot 3 withdrawals were sent on chain before the cutoff.", "Two Hot"),  # hc-005 has no Hot 3
], ids=["twin-not-a-label", "twin-r2-meridian", "twin-wrong-wallet-number"])
def test_p11_twin_count_word_before_other_text_is_still_swept(text, token):
    r = S.check_fabrication(output(HC005, text, entities=HC005_ENTITIES), truth(HC005))
    assert r.requires_human_review
    assert any(token in h["token"] for h in r.details["sweep_hits"])
    assert token not in r.details["citation_omissions"]


def test_p11_fixture_wallet_labels_follow_the_generator_styles():
    """CaseTruth has no wallet table: the labels are derived from valid_entities.
    Every derived label on the pilot dev cases is one of the generator's three styles."""
    style = re.compile(
        r"(?:hot|warm|cold)-\d|treasury-ops|staking-rewards|(?:Hot|Warm|Vault) \d|Treasury ops|Staking payouts"
        r"|(?:HOT|WARM|COLD)\d\d|TREAS01|STAKE01"
    )
    for case_id, t in _pilot_truths().items():
        labels = S.fixture_wallet_labels(t)
        assert labels, case_id
        assert all(style.fullmatch(label) for label in labels), (case_id, sorted(labels))


# =====================================================================================
# P10: type labels
# =====================================================================================

def _type_routes(case_id: str, items: list[dict]) -> list[str]:
    r = S.check_reconciling_items(output(case_id, "Commentary.", items=items), truth(case_id))
    return [x for x in r.review_reasons if " type " in x]


@pytest.mark.parametrize("case_id, ref, amount, label", [
    (STD002, "WD-2510-164297", 11643.52, "pending transfer"),  # S015: memo-only, owner ruling
    (STD012, "DP-2512-493757", 434042.88, "pending transfer (chain only, increases surplus)"),  # S034
    (STD012, "DP-2512-052149", -91877.40, "timing difference (ledger only, reduces surplus)"),  # S035
    (STD005, "DP-2511-648193", 256238.97, "timing difference (ledger only at cutoff, reduces reserves vs ledger)"),  # S064
    (STD005, "DP-2511-893641", 52300.36, "timing difference (chain only at cutoff, increases reserves vs ledger)"),  # S065
    (HC015, "TRF-2512-71501", 6637.86, "in-flight transfer"),  # S108
], ids=["S015", "S034", "S035", "S064", "S065", "S108"])
def test_p10_real_label_is_not_routed(case_id, ref, amount, label):
    assert _type_routes(case_id, [{"reference": ref, "amount_usd": amount, "type": label}]) == []


@pytest.mark.parametrize("case_id, ref, amount, label", [
    (STD012, "DP-2512-052149", -91877.40, "TIMING_DIFF"),
    (STD012, "DP-2512-493757", 434042.88, "Pending-Xfer (on-chain only)"),
    (HC015, "TRF-2512-71501", 6637.86, "In Flight Transfer"),
    (STD010, "OI-251231-441", 15.24, "Network-fee"),
], ids=["paraphrase-timing-diff", "paraphrase-pending-xfer", "paraphrase-in-flight", "paraphrase-network-fee"])
def test_p10_paraphrase_label_is_not_routed(case_id, ref, amount, label):
    assert _type_routes(case_id, [{"reference": ref, "amount_usd": amount, "type": label}]) == []


@pytest.mark.parametrize("case_id, ref, amount, label", [
    # a different type for an item whose schedule row states one
    (HC015, "TRF-2512-71501", 6637.86, "fee"),
    (STD012, "DP-2512-052149", -91877.40, "pending transfer (ledger only, reduces surplus)"),
    # a disallowed type for a memo-only item
    (STD002, "WD-2510-164297", 11643.52, "manual adjustment"),
    # the bracketed note contradicts the item's recording side or effect
    (STD012, "DP-2512-493757", 434042.88, "pending transfer (ledger only, increases surplus)"),
    (STD012, "DP-2512-493757", 434042.88, "pending transfer (chain only, reduces surplus)"),
    (STD005, "DP-2511-893641", 52300.36, "timing difference (chain only at cutoff, reduces reserves vs ledger)"),
    # a note that names another type
    (HC015, "TRF-2512-71501", 6637.86, "in-flight transfer (manual adjustment)"),
    # a label that names no single type
    (HC015, "TRF-2512-71501", 6637.86, "transfer"),
], ids=["twin-structured-other-type", "twin-structured-other-allowed-type", "twin-memo-only-disallowed",
        "twin-note-wrong-side", "twin-note-wrong-effect", "twin-note-wrong-effect-reserves",
        "twin-note-names-type", "twin-unreadable"])
def test_p10_twin_still_routes(case_id, ref, amount, label):
    assert _type_routes(case_id, [{"reference": ref, "amount_usd": amount, "type": label}])


def test_p10_memo_only_other_allowed_type_detail_is_recorded():
    r = S.check_reconciling_items(
        output(STD002, "Commentary.", items=[{"reference": "WD-2510-164297", "amount_usd": 11643.52,
                                              "type": "pending transfer"}]), truth(STD002))
    (check,) = r.details["type_checks"]
    assert check["routes"] is False and "memo-only" in check["why"]


# =====================================================================================
# P8: ineligible items in an exclusion statement (contract C1 wiring)
# =====================================================================================

def _c1_covers(case_id: str, text: str, refs: list[str]) -> bool:
    t = fold_dashes(nfkc_spaced(text))
    spans = N.exclusion_spans(t)
    for ref in refs:
        for m in re.finditer(re.escape(ref), t):
            if not any(lo <= m.start() and m.end() <= hi for lo, hi in spans):
                return False
    return True


def _needs_n2(case_id: str, text: str, refs: list[str]):
    return pytest.mark.xfail(
        not _c1_covers(case_id, text, refs), strict=True,
        reason="needs lane N2's wider C1 spans (narrative.exclusion_spans); runs as a plain test once they land",
    )


def _reliance(case_id: str, text: str, entities: list[str] | None = None, items: list[dict] | None = None):
    out = output(case_id, text, entities=entities, items=items if items is not None else [])
    return S.ineligible_reliance(out, truth(case_id))


def _clean_for(refs: list[str], problems: list[str], reasons: list[str]) -> bool:
    return not any(ref in x for ref in refs for x in problems + reasons)


def _not_clean_for(ref: str, problems: list[str], reasons: list[str]) -> bool:
    return any(ref in x for x in problems + reasons)


STD013_REFS = ["JE-25-10-9025", "DP-2510-831633", "WD-2511-221933"]
S089 = ("None of the open-items schedule qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment, "
        "DP-2511-680653 sits in the out-of-scope Staking payouts wallet, and DP-2511-340361 was cleared on both "
        "sides.")
S049 = ("No open item qualifies as a reconciling item, so $0.00 is explained and the full $3,397,826.64 is "
        "unexplained: DP-2511-428189 is cleared, DP-2511-679793 relates to out-of-scope TREAS01, DP-2512-591373 "
        "is dated after the cutoff, and JE-25-11-1801 is a suspense entry.")
S140 = ("JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 is an ETH item, so neither is relied on; "
        "this leaves $785.93 unexplained, which remains with open ticket OPS-2511-01077.")
S020 = "None of the listed or noted items qualifies as a reconciling item. DP-2511-983545 is a USDC movement, not SOL."
S109 = "Manual adjustment JE-25-12-3777 was excluded from the reconciling items as ineligible."
S098 = ("DP-2511-582345 (HOT01, ledger-only, dated 2025-11-30) is an eligible type, but its amount is blank in "
        "both the open-items schedule and the ledger extract, and no matching on-chain transaction appears, so it "
        "cannot be quantified or relied on.")
S115 = ("I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
        "WD-2511-006377 is an ETH item, not USDC.")

REAL_P8 = [
    ("S080-S082", STD013, S077, STD013_REFS),
    ("S089-S090", HC003, S089, ["JE-25-11-7405", "DP-2511-680653"]),
    ("S091", HC003, S089, ["DP-2511-340361"]),
    ("S049-S051", HC009, S049, ["DP-2511-428189", "DP-2511-679793", "DP-2512-591373"]),
    ("S052", HC009, S049, ["JE-25-11-1801"]),
    ("S140-S141", STD008, S140, ["JE-25-11-1225", "WD-2511-006377"]),
    ("S020", STD004, S020, ["DP-2511-983545"]),
    ("S109", HC015, S109, ["JE-25-12-3777"]),
    ("S115-S116", STD008, S115, ["JE-25-11-1225", "WD-2511-006377"]),
]


@pytest.mark.parametrize(
    "row, case_id, text, refs",
    [pytest.param(*row, marks=_needs_n2(row[1], row[2], row[3]), id=row[0]) for row in REAL_P8],
)
def test_p8_real_sentence_exclusion_is_decided(row, case_id, text, refs):
    problems, reasons = _reliance(case_id, text)
    assert _clean_for(refs, problems, reasons), (problems, reasons)


PARAPHRASE_P8 = [
    # each one routes on 45838b9 (the old sentence-level cue list does not read it)
    ("paraphrase-british-head", STD013, "I have not relied upon JE-25-10-9025, DP-2510-831633 or WD-2511-221933.",
     STD013_REFS),
    ("paraphrase-terse-backref", STD008, "JE-25-11-1225 is a PPA and WD-2511-006377 an ETH line, so neither qualifies.",
     ["JE-25-11-1225", "WD-2511-006377"]),
    ("paraphrase-colon-brackets", HC003,
     "No item qualifies: JE-25-11-7405 (manual adj.), DP-2511-680653 (Staking payouts wallet).",
     ["JE-25-11-7405", "DP-2511-680653"]),
    ("paraphrase-another-asset", STD004, "DP-2511-983545 belongs to another asset (USDC).", ["DP-2511-983545"]),
]


@pytest.mark.parametrize(
    "row, case_id, text, refs",
    [pytest.param(*row, marks=_needs_n2(row[1], row[2], row[3]), id=row[0]) for row in PARAPHRASE_P8],
)
def test_p8_paraphrase_exclusion_is_decided(row, case_id, text, refs):
    assert S._reliance_verdict(fold_dashes(text)) == "unclear"  # the old reading routes it
    problems, reasons = _reliance(case_id, text)
    assert _clean_for(refs, problems, reasons), (problems, reasons)


TWIN_P8 = [
    # a reliance in a list that opens like an exclusion
    ("twin-relied-in-list", STD013,
     "I did not rely on the manual adjustment JE-25-10-9025 or the cleared item DP-2510-831633, and relied on "
     "the post-cutoff item WD-2511-221933.", "WD-2511-221933"),
    ("twin-exception", STD013, "I did not rely on any open item except the manual adjustment JE-25-10-9025.",
     "JE-25-10-9025"),
    # the coarse span: the exclusion phrase belongs to the other member (a C1 phrase the
    # old sentence-level cue list does not hold, so 45838b9 routes it)
    ("twin-other-members-phrase", STD013,
     "JE-25-10-9025 is a timing item, DP-2510-645053 is an ETH deposit, not USDC.", "JE-25-10-9025"),
    ("twin-colon-head-then-explains", HC003,
     "None of the open items qualifies as a reconciling item: JE-25-11-7405 explains $8,555.30 of the variance.",
     "JE-25-11-7405"),
    ("twin-other-items-head", HC003,
     "None of the other items qualifies as a reconciling item: JE-25-11-7405 is a manual adjustment.",
     "JE-25-11-7405"),
    ("twin-not-every", HC003,
     "Not every open item qualifies as a reconciling item: JE-25-11-7405 does.", "JE-25-11-7405"),
    ("twin-trailing-with-offset", STD008,
     "JE-25-11-1225 is a prior-period adjustment and WD-2511-006377 offsets $409.20 of the gap, so neither is "
     "relied on.", "WD-2511-006377"),
    ("twin-other-asset-then-explains", STD004,
     "DP-2511-983545 is a USDC movement, not SOL, and it explains $859.01 of the variance.", "DP-2511-983545"),
    ("twin-split-with-agent", HC009,
     "No open item qualifies as a reconciling item, so $3,397,826.64 is explained by DP-2511-428189: "
     "DP-2511-679793 relates to out-of-scope TREAS01.", "DP-2511-428189"),
    ("twin-excluded-but-counted", HC015,
     "JE-25-12-3777 was excluded from the schedule but counted in the reconciling items.", "JE-25-12-3777"),
    ("twin-not-only", STD013,
     "Not only DP-2510-831633 but also JE-25-10-9025 was relied on.", "JE-25-10-9025"),
    ("twin-nothing-omitted", STD013,
     "Nothing was omitted: JE-25-10-9025 and DP-2510-831633 are in the bridge.", "JE-25-10-9025"),
    ("twin-usdc-item-in-usdc-case", STD008,
     "WD-2511-006377 is a USDC item and DP-2511-502477 is a timing difference.", "WD-2511-006377"),
    ("twin-none-of-list-omitted", STD013,
     "None of DP-2510-831633, WD-2511-221933 or JE-25-10-9025 was omitted.", "JE-25-10-9025"),
    ("twin-colon-head-member-verb", STD008,
     "I did not rely on the other two schedule entries: JE-25-11-1225 is a prior-period adjustment, and "
     "WD-2511-006377 bridges the gap.", "WD-2511-006377"),
    ("twin-backref-member-verb", STD008,
     "JE-25-11-1225 is a PPA and WD-2511-006377 settles the gap, so neither is relied on.", "WD-2511-006377"),
    ("twin-not-rely-then-kept", STD013, "I did not rely on DP-2510-831633 and kept JE-25-10-9025 in the bridge.",
     "JE-25-10-9025"),
]


@pytest.mark.parametrize("row, case_id, text, ref", TWIN_P8, ids=[r[0] for r in TWIN_P8])
def test_p8_twin_still_fails_or_routes(row, case_id, text, ref):
    problems, reasons = _reliance(case_id, text)
    assert _not_clean_for(ref, problems, reasons), (problems, reasons)


# The scope rules alone, under the widest span C1 could return (every sentence whole):
# the proof must still decide by its own scope reading, not by the span.

@pytest.fixture
def whole_sentence_spans(monkeypatch):
    monkeypatch.setattr(N, "exclusion_spans", lambda text: S._sentence_bounds(text))


@pytest.mark.parametrize("row, case_id, text, refs", [
    ("S091", HC003, S089, ["JE-25-11-7405", "DP-2511-680653", "DP-2511-340361"]),
    ("S052", HC009, S049, ["DP-2511-428189", "DP-2511-679793", "DP-2512-591373", "JE-25-11-1801"]),
    ("S115-S116", STD008, S115, ["JE-25-11-1225", "WD-2511-006377"]),
], ids=["S091", "S052", "S115-S116"])
def test_p8_scope_decides_under_whole_sentence_spans(whole_sentence_spans, row, case_id, text, refs):
    problems, reasons = _reliance(case_id, text)
    assert _clean_for(refs, problems, reasons), (problems, reasons)


@pytest.mark.parametrize("row, case_id, text, ref", TWIN_P8, ids=[r[0] for r in TWIN_P8])
def test_p8_twin_still_fails_or_routes_under_whole_sentence_spans(whole_sentence_spans, row, case_id, text, ref):
    problems, reasons = _reliance(case_id, text)
    assert _not_clean_for(ref, problems, reasons), (problems, reasons)


@pytest.mark.parametrize("text", [
    "JE-25-11-2733 is a suspense clearing entry.",  # S021: no exclusion phrase, stays a route
    "DP-2511-582345 (HOT01, ledger-only) is in scope and dated at cutoff, but it has no amount in the schedule "
    "or the ledger and no matching on-chain transaction, so it cannot be quantified.",  # S006: no exclusion phrase
    # S098: the phrase is there ("cannot be ... relied on") but the member calls the item "an eligible type"
    # and holds a contrast; the proof does not read past that, so it stays a route (measured, accepted)
    S098,
], ids=["S021-no-phrase", "S006-no-phrase", "S098-member-not-descriptive"])
def test_p8_undecided_sentence_stays_routed_under_whole_sentence_spans(whole_sentence_spans, text):
    case_id = STD004 if "2733" in text else HC008
    ref = "JE-25-11-2733" if "2733" in text else "DP-2511-582345"
    problems, reasons = _reliance(case_id, text)
    assert problems == [] and _not_clean_for(ref, problems, reasons)


def test_p8_item_in_reconciling_items_is_not_read_here():
    items = [{"reference": "JE-25-10-9025", "amount_usd": 6701.03}]
    problems, reasons = _reliance(STD013, S077, items=items)
    assert not any("JE-25-10-9025" in x for x in problems + reasons)
    r = S.check_reconciling_items(output(STD013, S077, items=items), truth(STD013))
    assert r.failure_modes == ("FM-015",)  # citing it there is the error itself


def test_p8_relied_verdict_is_never_downgraded(whole_sentence_spans):
    """A sentence the old reading finds relied on stays FM-015, span or not."""
    problems, _ = _reliance(STD013, "JE-25-10-9025 explains $6,701.03 of the variance.")
    assert any("JE-25-10-9025" in p for p in problems)


def test_p8_known_gap_kept_as_it_was():
    """Measured on 45838b9 and unchanged here: the old sentence-level exclusion cue
    ("out-of-scope") excludes every item the sentence names. P8 only turns an
    unclear verdict into an excluded one, so it neither opens nor closes this gap."""
    text = "JE-25-11-7405 is a timing item, DP-2511-680653 sits in the out-of-scope Staking payouts wallet."
    assert S._reliance_verdict(fold_dashes(text)) == "excluded"


def test_p8_details_list_the_span_decisions():
    r = S.check_reconciling_items(output(STD013, S077, entities=STD013_ENTITIES, items=STD013_ITEMS), truth(STD013))
    assert sorted(r.details["span_excluded"]) == sorted(STD013_REFS)
