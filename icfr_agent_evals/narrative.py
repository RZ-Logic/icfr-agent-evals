"""Narrative-consistency extractor (pre-registration section 6, "Correctness,
narrative consistency").

check_narrative_consistency reads parsed["narrative"] and returns the failure modes
its statements fire and the reasons it routes the (sample, epoch) to
requires_human_review. The correctness anchor calls it: any failure mode sets
Correctness to 0, any review reason means the epoch cannot pass. FM-016 on the
structured asset and period fields is the correctness anchor's; this module owns
FM-016 on narrative statements only.

`parsed` is always a schema-valid worker output (the strict parser has already
rejected anything else as FM-000), so every key in casefile.OUTPUT_FIELDS is
present with the right type.

What is extracted, after casefile.nfkc_spaced:
  - currency amounts, each bound to an accounting role by cue words in its own
    clause (variance, ledger, onchain, item, threshold, explained, unexplained);
  - direction statements (surplus or shortfall words and comparative phrases);
  - band statements (M1 to M3 labels, threshold comparisons, band words);
  - asset statements (tickers and full asset names; chain names are not assets);
  - period statements (month-year forms, and full dates after a period cue).

Error direction (section 6): anything the extractor cannot resolve routes to
review and never passes. A false fail is the accepted cost; a false pass on a
contradiction is the failure this module exists to prevent. No model is called.
Every vocabulary lives in narrative_vocab.py, hashed by vocab_sha256().
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field

from . import narrative_vocab as V
from .casefile import (
    ADDRESS_FORMATS,
    ASSET_CHAIN,
    ASSETS,
    REFERENCE_FORMAT,
    TX_HASH_FORMATS,
    USD_TOLERANCE,
    CaseTruth,
    canonical_reference,
    explained_from_citations,
    item_effect_usd,
    nfkc_spaced,
    period_cutoff_date,
)
from .policy import M1_CEILING, M3_FLOOR

assert set(V.IN_SCOPE_TICKERS) == set(ASSETS), "narrative vocabulary drifted from the contract asset list"
assert set(V.CHAIN_NAMES) == set(ASSET_CHAIN.values()), "chain vocabulary drifted from ASSET_CHAIN"


@dataclass(frozen=True)
class NarrativeResult:
    """failure_modes: codes the narrative fires (FM-001, FM-002, FM-003, FM-016).
    Any code here sets Correctness to 0.
    review_reasons: non-empty routes the (sample, epoch) to requires_human_review;
    the epoch cannot pass.
    extractions: audit trail of what was extracted and how each item was resolved."""

    failure_modes: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    extractions: tuple[dict, ...] = ()


# --- Compiled vocabulary ---------------------------------------------------------------

_MASK_AMT = V.AMOUNT_PLACEHOLDER  # stands in for an amount span in masked text
_MASK_OTHER = "\ue001"  # stands in for a reference, entity, date or period span

THRESHOLD_VALUES: dict[str, float] = {"M1_CEILING": M1_CEILING, "M3_FLOOR": M3_FLOOR}


def _alt(patterns) -> str:
    return "|".join(f"(?:{p})" for p in patterns)


def _w(pattern: str) -> str:
    """Whole-word form of a fragment: no letter or digit on either side."""
    return rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])"


def _groups(patterns: list[str]) -> str:
    """One alternation with a named group per entry, so lastgroup gives the index."""
    return "|".join(f"(?P<g{i}>{_w(p)})" for i, p in enumerate(patterns))


def _index(m: re.Match) -> int:
    return int(m.lastgroup[1:])


_I = re.IGNORECASE

# Role cues: the vocabulary's ROLE_CUES with the direction words joined to the variance role.
_CUES: list[tuple[str, str]] = [(r, p) for r, p in V.ROLE_CUES if r == "neutral"]
_CUES += [(r, p) for r, p in V.ROLE_CUES if r in ("unexplained", "explained")]
_CUES += [("variance", p) for _, p in V.DIRECTION_WORDS]
_CUES += [(r, p) for r, p in V.ROLE_CUES if r not in ("neutral", "unexplained", "explained")]
_CUE_RE = re.compile(_groups([p for _, p in _CUES]), _I)

_LINK_WORDS = frozenset(
    V.LINK_WORDS
    + [p.lower() for pats in V.CHAIN_NAMES.values() for p in pats if p.isalpha()]
    + [t.lower() for t in V.IN_SCOPE_TICKERS]
)
_LINK_PUNCT = frozenset(V.LINK_PUNCT)
_COMPARISON_RE = re.compile(_alt(V.COMPARISON_WORDS), _I)
_COMPARISON_AFTER_RE = re.compile(
    rf" (?:{_alt(V.COMPARISON_AFTER_AMOUNT)}) (?:the )?(?:{_MASK_AMT}+ (?:of |in )?(?:the )?)?"
    rf"(?:{_alt(V.DIRECTION_LIABILITY_NOUNS + V.DIRECTION_RESERVE_NOUNS)})(?![A-Za-z0-9])",
    _I,
)
_PARALLEL_RE = re.compile(_w(_alt(V.PARALLEL_LIST_WORDS)), _I)

_BAND_CMP = {k: re.compile(_alt(v), _I) for k, v in V.BAND_COMPARATORS.items()}
_BAND_CMP_ALT = _alt(V.BAND_COMPARATORS["above"] + V.BAND_COMPARATORS["below"])
_TREF_ALT = _alt([row[0] for row in V.THRESHOLD_NAMES])
_TREF_RE = re.compile(_w(_TREF_ALT), _I)
_BAND_STMT_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<cmp>{_BAND_CMP_ALT})(?: the| a)? "
    rf"(?:(?P<amt>{_MASK_AMT}+)(?: (?P<tref1>{_TREF_ALT})(?![A-Za-z0-9]))?|(?P<tref2>{_TREF_ALT})(?![A-Za-z0-9]))",
    _I,
)
_BAND_RANGE_RE = re.compile(
    rf"(?<![A-Za-z0-9]){V.BAND_RANGE_WORD} (?:the )?(?P<a1>{_MASK_AMT}+)(?: [A-Za-z-]+){{0,3}} and (?:the )?(?P<a2>{_MASK_AMT}+)",
    _I,
)
_BAND_LABEL_RE = re.compile(_w(V.BAND_LABEL))  # case-sensitive
_BAND_WORD_RE = re.compile(_groups([row[0] for row in V.BAND_WORDS]), _I)

_SPLIT_CLAIM_RE = re.compile(_groups([p for _, p in V.SPLIT_CLAIMS]), _I)
_SELF_NEGATED_RE = re.compile(_alt(V.SPLIT_CLAIMS_SELF_NEGATED), _I)
_NON_USD_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:{_alt(V.NON_USD_CURRENCY)}) ?[0-9][0-9,.]*|[0-9][0-9,.]* ?(?:{_alt(V.NON_USD_CURRENCY)})(?![A-Za-z0-9])"
)
_NEG_BEFORE_RE = re.compile(_w(_alt(V.NEGATION_BEFORE)), _I)
_NEG_AFTER_RE = re.compile(rf"(?:{_alt(V.NEGATION_AFTER)})(?![A-Za-z0-9])", _I)


def _direction_patterns() -> list[tuple[str, str]]:
    r_noun = _alt(V.DIRECTION_RESERVE_NOUNS)
    l_noun = _alt(V.DIRECTION_LIABILITY_NOUNS)
    copula = rf"(?:(?:{_alt(V.DIRECTION_COPULAS)}) )?"
    gap = rf"(?:,? (?:[A-Za-z][A-Za-z-]*|{_MASK_AMT}+)){{0,{V.DIRECTION_SUBJECT_GAP_WORDS}}}?,?"
    above = _alt(V.DIRECTION_ABOVE_VERBS)
    below = _alt(V.DIRECTION_BELOW_VERBS)
    comparatives = [
        ("surplus", rf"(?:{r_noun}){gap} {copula}(?:{above}) (?:the )?(?:{l_noun})"),
        ("surplus", rf"(?:{l_noun}){gap} {copula}(?:{below}) (?:the )?(?:{r_noun})"),
        ("shortfall", rf"(?:{r_noun}){gap} {copula}(?:{below}) (?:the )?(?:{l_noun})"),
        ("shortfall", rf"(?:{l_noun}){gap} {copula}(?:{above}) (?:the )?(?:{r_noun})"),
    ]
    return comparatives + [(d, p) for d, p in V.DIRECTION_WORDS]


_DIRECTIONS = _direction_patterns()
_DIRECTION_RE = re.compile(_groups([p for _, p in _DIRECTIONS]), _I)

# Asset statements: names (case-insensitive) first, so a chain name or a longer
# asset name is consumed before its parts; then tickers (case-sensitive).
_ASSET_NAME_ROWS: list[tuple[str, str]] = (
    [("other_asset:" + t, p) for p, t in V.OTHER_ASSET_NAMES.items()]
    + [("chain:" + c, p) for c, pats in V.CHAIN_NAMES.items() for p in pats]
    + [("other_chain:" + p, p) for p in V.OTHER_CHAIN_NAMES]
    + [("asset:" + t, p) for t, pats in V.ASSET_NAMES.items() for p in pats]
)
_ASSET_NAME_RE = re.compile(_groups([p for _, p in _ASSET_NAME_ROWS]), _I)
_TICKER_RE = re.compile(_w(_alt(sorted(set(V.IN_SCOPE_TICKERS) | set(V.OTHER_TICKERS), key=len, reverse=True))))

_MONTH_ALT = _alt(sorted(V.MONTHS, key=len, reverse=True))
_PERIOD_CUE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:{_alt(V.PERIOD_CUES)})(?: ?(?:{_alt(V.PERIOD_CUE_FILLER)}))* ?$", _I
)

# Full dates: (pattern, group order). Order: "ymd", "mdy", "dmy", or "ambiguous" for
# numeric slashed dates where day and month could be either way round.
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![0-9A-Za-z-])(20[0-9]{2})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])(?![0-9])"), "ymd"),
    (re.compile(r"(?<![0-9/])(20[0-9]{2})/([0-9]{1,2})/([0-9]{1,2})(?![0-9/])"), "ymd"),
    (re.compile(rf"(?<![A-Za-z0-9])({_MONTH_ALT})\.? ([0-9]{{1,2}})(?:st|nd|rd|th)?,? (20[0-9]{{2}})(?![0-9])", _I), "mdy"),
    (re.compile(rf"(?<![A-Za-z0-9])([0-9]{{1,2}})(?:st|nd|rd|th)? ({_MONTH_ALT})\.?,? (20[0-9]{{2}})(?![0-9])", _I), "dmy"),
    (re.compile(r"(?<![0-9/])([0-9]{1,2})/([0-9]{1,2})/(20[0-9]{2})(?![0-9/])"), "ambiguous"),
]
# Month-year forms: (pattern, group order) with "ym", "my" (numeric) or "Mon-y".
_MONTH_YEAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![0-9A-Za-z-])(20[0-9]{2})-(0[1-9]|1[0-2])(?![0-9]|-[0-9])"), "ym"),
    (re.compile(r"(?<![0-9/])(20[0-9]{2})/(0?[1-9]|1[0-2])(?![0-9/])"), "ym"),
    (re.compile(r"(?<![0-9/-])(0?[1-9]|1[0-2])[/-](20[0-9]{2})(?![0-9/-])"), "my"),
    (re.compile(rf"(?<![A-Za-z0-9])({_MONTH_ALT})\.?,? (20[0-9]{{2}})(?![0-9])", _I), "Mon-y"),
    (re.compile(rf"(?<![A-Za-z0-9])({_MONTH_ALT})[-'\u2019 ]?'?([0-9]{{2}}|20[0-9]{{2}})(?![0-9])(?! ?,? ?20[0-9]{{2}})", _I), "Mon-yy"),
]
_QUARTER_RE = re.compile(_w(_alt(V.QUARTER_PATTERNS)), _I)

_NUM = r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?"
_NUM_END = r"(?![0-9]|,[0-9]|\.[0-9])"
_SUF = rf"(?: ?(?P<suf>{_alt(sorted(V.MAGNITUDE_SUFFIXES, key=len, reverse=True))})(?![A-Za-z0-9]))?"
_NEG = r"(?P<neg>(?<![^\s(])[-\u2212+] ?)?"
_CUR_SUFFIX = _alt(V.CURRENCY_SUFFIXES)
_AMOUNT_RES: list[re.Pattern] = [
    re.compile(
        rf"{_NEG}(?<![A-Za-z0-9])(?:{_alt(V.CURRENCY_PREFIXES)}) ?(?P<neg2>[-\u2212+] ?)?(?P<num>{_NUM}){_NUM_END}{_SUF}"
        rf"(?: (?:{_CUR_SUFFIX})(?![A-Za-z0-9]))?",
        _I,
    ),
    re.compile(rf"{_NEG}(?<![A-Za-z0-9.,$])(?P<num>{_NUM}){_NUM_END}{_SUF} (?:{_CUR_SUFFIX})(?![A-Za-z0-9])", _I),
]
_UNIT_AFTER = (
    rf"(?! ?(?i:{_alt(V.UNIT_WORDS)})(?![A-Za-z0-9]))"
    rf"(?! ?(?:{_alt(V.IN_SCOPE_TICKERS + V.OTHER_TICKERS)})(?![A-Za-z0-9]))"
    rf"(?! ?(?i:{_alt(sorted(V.MAGNITUDE_SUFFIXES, key=len, reverse=True))})(?![A-Za-z0-9]))"
)
_BARE_RE = re.compile(
    rf"(?<![A-Za-z0-9.,$/\-\u2212])(?P<num>[0-9]{{1,3}}(?:,[0-9]{{3}})+(?:\.[0-9]+)?|[0-9]{{{V.BARE_NUMBER_MIN_DIGITS},}}(?:\.[0-9]+)?)"
    rf"(?![0-9]|,[0-9]|\.[0-9]|/){_UNIT_AFTER}"
)

_REF_RE = re.compile(rf"(?<![A-Za-z0-9-]){REFERENCE_FORMAT}(?![A-Za-z0-9]|-[A-Za-z0-9])")
_TOKEN_STANDARDS = frozenset(V.TOKEN_STANDARDS)
_ENTITY_RES = [re.compile(_w(p)) for p in list(TX_HASH_FORMATS.values()) + list(ADDRESS_FORMATS.values())]
_WORD_RE = re.compile(rf"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?|[^\sA-Za-z0-9{_MASK_AMT}{_MASK_OTHER}]")
_SENTENCE_END_RE = re.compile(r"[.!?;](?=\s|$)")
_ABBREV = frozenset(V.ABBREVIATIONS)


# --- Spans and tokens -----------------------------------------------------------------

@dataclass
class _Tok:
    start: int
    end: int
    kind: str      # amount, ref, entity, date, period, quarter, cue, word, punct
    text: str
    role: str = ""  # cue role
    data: dict = field(default_factory=dict)

    @property
    def low(self) -> str:
        return self.text.lower()


def _free(spans: list[_Tok], s: int, e: int) -> bool:
    return all(e <= t.start or s >= t.end for t in spans)


def _amount_value(m: re.Match) -> float:
    value = float(m.group("num").replace(",", ""))
    suf = m.groupdict().get("suf")
    if suf:
        value *= V.MAGNITUDE_SUFFIXES[suf.lower()]
    return round(value, 2)


def _find_specials(text: str, truth: CaseTruth) -> list[_Tok]:
    """Reference, entity, date, period, quarter and amount spans, in priority order;
    a later scan never claims text an earlier one took."""
    spans: list[_Tok] = []

    def add(s: int, e: int, kind: str, **data) -> None:
        if _free(spans, s, e):
            spans.append(_Tok(s, e, kind, text[s:e], data=data))

    literals: list[tuple[str, str]] = []
    for item in truth.items:
        literals.append((canonical_reference(item.reference), item.reference))
        literals.extend((canonical_reference(a), item.reference) for a in item.aliases)
    for lit, ref in sorted(set(literals), key=lambda x: -len(x[0])):
        if not lit:
            continue
        for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(lit)}(?![A-Za-z0-9]|-[A-Za-z0-9])", text):
            add(m.start(), m.end(), "ref", reference=ref)
    for rx in _ENTITY_RES:
        for m in rx.finditer(text):
            add(m.start(), m.end(), "entity")
    for m in _REF_RE.finditer(text):
        tok = m.group(0)
        if not any(ch.isdigit() for ch in tok) or tok.upper() in _TOKEN_STANDARDS:
            continue
        if any(rx.fullmatch(tok) for rx, _ in _MONTH_YEAR_PATTERNS):
            continue
        add(m.start(), m.end(), "ref", reference=None)
    for rx, order in _DATE_PATTERNS:
        for m in rx.finditer(text):
            add(m.start(), m.end(), "date", **_date_parts(m, order))
    for rx, order in _MONTH_YEAR_PATTERNS:
        for m in rx.finditer(text):
            parts = _month_year_parts(m, order)
            if parts is not None:
                add(m.start(), m.end(), "period", **parts)
    for m in _QUARTER_RE.finditer(text):
        add(m.start(), m.end(), "quarter")
    for rx in _AMOUNT_RES:
        for m in rx.finditer(text):
            sign = (m.group("neg") or m.groupdict().get("neg2") or "").strip()
            s, e = m.span()
            paren = s > 0 and text[s - 1] == "(" and text[e:e + 1] == ")"
            add(s, e, "amount", value=_amount_value(m), negative=sign in ("-", "\u2212"), positive=sign == "+",
                parenthesized=paren, bare=False)
    for m in _BARE_RE.finditer(text):
        add(m.start(), m.end(), "amount", value=_amount_value(m), negative=False, positive=False,
            parenthesized=False, bare=True)
    return sorted(spans, key=lambda t: t.start)


def _month_number(name: str) -> int:
    return V.MONTHS[name.lower().rstrip(".")]


def _date_parts(m: re.Match, order: str) -> dict:
    g = m.groups()
    if order == "ymd":
        y, mo, d = int(g[0]), int(g[1]), int(g[2])
    elif order == "mdy":
        mo, d, y = _month_number(g[0]), int(g[1]), int(g[2])
    elif order == "dmy":
        d, mo, y = int(g[0]), _month_number(g[1]), int(g[2])
    else:  # numeric slashes: month first unless the first field cannot be a month
        a, b, y = int(g[0]), int(g[1]), int(g[2])
        if a > 12 >= b:
            d, mo = a, b
        elif b > 12 >= a:
            mo, d = a, b
        else:
            return {"ambiguous": a != b, "year": y, "month": a, "day": b, "valid": 1 <= a <= 12}
    return {"ambiguous": False, "year": y, "month": mo, "day": d, "valid": 1 <= mo <= 12 and 1 <= d <= 31}


def _month_year_parts(m: re.Match, order: str) -> dict | None:
    g = m.groups()
    if order == "ym":
        y, mo = int(g[0]), int(g[1])
    elif order == "my":
        mo, y = int(g[0]), int(g[1])
    elif order == "Mon-y":
        mo, y = _month_number(g[0]), int(g[1])
    else:
        mo, yy = _month_number(g[0]), g[1]
        sep = m.group(0)[len(g[0]):-len(yy)]
        if not sep.strip(" ."):  # "Dec 25" reads as a day, not a year
            return None
        y = int(yy) if len(yy) == 4 else 2000 + int(yy)
    if not 1 <= mo <= 12:
        return None
    return {"year": y, "month": mo}


def _mask(text: str, specials: list[_Tok]) -> str:
    chars = list(text)
    for t in specials:
        fill = _MASK_AMT if t.kind == "amount" else _MASK_OTHER
        chars[t.start:t.end] = fill * (t.end - t.start)
    return "".join(chars)


def _tokenize(text: str, masked: str, specials: list[_Tok]) -> list[_Tok]:
    toks = list(specials)
    covered = [False] * len(text)
    for t in specials:
        covered[t.start:t.end] = [True] * (t.end - t.start)
    # cues never span an amount: every special span is the same opaque character here
    opaque = masked.replace(_MASK_AMT, _MASK_OTHER)
    for m in _CUE_RE.finditer(opaque):
        role = _CUES[_index(m)][0]
        toks.append(_Tok(m.start(), m.end(), "cue", m.group(0), role=role))
        covered[m.start():m.end()] = [True] * (m.end() - m.start())
    for m in _WORD_RE.finditer(masked):
        if covered[m.start()]:
            continue
        kind = "word" if m.group(0)[0].isalnum() else "punct"
        toks.append(_Tok(m.start(), m.end(), kind, m.group(0)))
    return sorted(toks, key=lambda t: t.start)


def _clause_bounds(text: str, specials: list[_Tok]) -> list[int]:
    """Start offsets of every clause after the first. A clause ends at sentence
    punctuation or a semicolon, except inside a span or after an abbreviation."""
    starts: list[int] = []
    for m in _SENTENCE_END_RE.finditer(text):
        pos = m.start()
        if any(t.start <= pos < t.end for t in specials):
            continue
        if text[pos] == ".":
            prev = re.search(r"([A-Za-z.]+)$", text[:pos])
            if prev and prev.group(1).lower() in _ABBREV:
                continue
        starts.append(pos + 1)
    return starts


def _is_link(t: _Tok) -> bool:
    if t.kind == "word":
        return t.low in _LINK_WORDS
    if t.kind == "punct":
        return t.text in _LINK_PUNCT
    return t.kind == "cue" and t.role == "neutral"


def _distance(toks: list[_Tok], i: int, j: int, skip: frozenset = frozenset()) -> int:
    lo, hi = min(i, j), max(i, j)
    return sum(1 for k in range(lo + 1, hi) if not _is_link(toks[k]) and k not in skip)


# --- Amount roles -----------------------------------------------------------------------

def _near(value: float, target: float) -> bool:
    return abs(abs(value) - abs(target)) <= USD_TOLERANCE


def _is_threshold_value(value: float) -> bool:
    return any(_near(value, v) for v in THRESHOLD_VALUES.values())


def _preceding_words(toks: list[_Tok], i: int, skip: frozenset) -> list[str]:
    out: list[str] = []
    k = i - 1
    while k >= 0 and toks[k].kind in ("word", "cue") and toks[k].low in skip:
        k -= 1
    while k >= 0 and len(out) < 3 and toks[k].kind in ("word", "cue"):
        out.insert(0, toks[k].low)
        k -= 1
    return out


_ARTICLES = frozenset({"the", "a", "an"})
_PARTITIVE = frozenset({"of", "the", "this", "total", "its"})
_BAND_CMP_FULL = re.compile(rf"(?:{_BAND_CMP_ALT})", _I)


def _resolve_role(toks: list[_Tok], i: int, ctx: "_Context") -> dict:
    """Bind the amount at toks[i] to a role from cue words in its clause.

    Rules, in order: a parallel list ("respectively") binds nothing; an amount right
    after a band comparator, or in a "between" range, that equals a policy threshold
    is a threshold; "by $X" after a comparison word, or "$X below liabilities", is the
    variance; two amounts joined by "and" ("were $A and $B") bind nothing; otherwise the
    nearest cue binds, counting the words between that are not linking words. Ties
    at the same distance are unresolved, except at distance zero between a cue before
    and a cue after the amount: the cue before binds ("liabilities of $X fell
    short"), unless an article opens a noun phrase that the cue after heads ("explains
    the $X shortfall"). A cue farther than WINDOWS["cue_max_distance"] binds nothing."""
    amt = toks[i]
    value = amt.data["value"]
    if any(t.kind == "word" and _PARALLEL_RE.fullmatch(t.text) for t in toks):
        return {"role": "unresolved", "resolution": "parallel_list"}

    if _is_threshold_value(value):
        words = _preceding_words(toks, i, _ARTICLES)
        for n in range(len(words), 0, -1):
            if _BAND_CMP_FULL.fullmatch(" ".join(words[-n:])):
                return {"role": "threshold", "resolution": "band_comparator", "cue": " ".join(words[-n:])}
        window = toks[max(0, i - V.WINDOWS["range_window"]):i]
        if any(t.kind == "word" and t.low == V.BAND_RANGE_WORD for t in window):
            return {"role": "threshold", "resolution": "band_range", "cue": V.BAND_RANGE_WORD}

    if i >= 1 and toks[i - 1].kind == "word" and toks[i - 1].low == "by":
        k = i - 2
        while k >= 0 and (
            _is_link(toks[k]) or toks[k].kind == "amount" or toks[k].text == ","
            or (toks[k].kind == "cue" and toks[k].role in ("ledger", "onchain"))
        ):
            k -= 1
        if k >= 0 and (
            (toks[k].kind == "cue" and toks[k].role == "variance")
            or (toks[k].kind == "word" and _COMPARISON_RE.fullmatch(toks[k].text))
        ):
            return {"role": "variance", "resolution": "comparison_by", "cue": toks[k].text}

    m = _COMPARISON_AFTER_RE.match(ctx.masked, amt.end)
    if m:
        return {"role": "variance", "resolution": "comparison_amount", "cue": ctx.text[m.start():m.end()].strip()}

    joiners = ("and", ",", "or")
    if (i + 2 < len(toks) and toks[i + 1].low in joiners and toks[i + 2].kind == "amount") or (
        i >= 2 and toks[i - 1].low in joiners and toks[i - 2].kind == "amount"
    ):
        return {"role": "unresolved", "resolution": "parallel_amounts"}

    # "$X of the shortfall" is a part of the variance, not the variance: that cue
    # neither binds the amount nor separates it from the next cue
    partitive: set[int] = set()
    for j in range(i + 1, len(toks)):
        t = toks[j]
        if t.kind == "cue" and t.role == "variance":
            between = toks[i + 1:j]
            if between and all(b.kind == "word" and b.low in _PARTITIVE for b in between) and any(
                b.low == "of" for b in between
            ):
                partitive.add(j)
            break
        if not (t.kind == "word" and t.low in _PARTITIVE):
            break
    skip = frozenset(partitive)
    cands = []
    for j, t in enumerate(toks):
        if j == i or j in skip or not (t.kind == "ref" or (t.kind == "cue" and t.role != "neutral")):
            continue
        role = "item" if t.kind == "ref" else t.role
        cands.append((_distance(toks, i, j, skip), "before" if j < i else "after", role, t))
    if not cands:
        return {"role": "unresolved", "resolution": "no_cue"}
    best = min(c[0] for c in cands)
    if best > V.WINDOWS["cue_max_distance"]:
        return {"role": "unresolved", "resolution": "no_cue_near", "distance": best}
    nearest = [c for c in cands if c[0] == best]
    keys = {(c[2], c[3].start if c[2] == "item" else 0) for c in nearest}
    if len(keys) > 1 and best == 0:
        before = [c for c in nearest if c[1] == "before"]
        after = [c for c in nearest if c[1] == "after"]
        if before and after:
            b = max(before, key=lambda c: c[3].start)
            a = min(after, key=lambda c: c[3].start)
            b_idx, a_idx = toks.index(b[3]), toks.index(a[3])
            opens_phrase = any(toks[k].low in _ARTICLES for k in range(b_idx + 1, i))
            nearest = [a] if opens_phrase and a_idx == i + 1 else [b]
            keys = {(nearest[0][2], 0)}
    if len(keys) > 1:
        return {
            "role": "unresolved", "resolution": "tied_cues", "distance": best,
            "cues": sorted({c[3].text for c in nearest}),
        }
    d, side, role, cue = nearest[0]
    out = {"role": role, "resolution": f"nearest_cue_{side}", "cue": cue.text, "distance": d}
    if role == "item":
        out["ref_token"] = cue
    return out


@dataclass
class _Context:
    text: str              # the normalized narrative
    masked: str            # the same with every amount replaced by AMOUNT_PLACEHOLDER
    truth: CaseTruth
    cited: dict            # item reference -> ReconcilingItem, cited in reconciling_items and eligible
    cited_any: frozenset   # item references cited in reconciling_items, eligible or not
    explained: float       # computed from the cited items (casefile.explained_from_citations)
    unexplained: float

    def quantities(self) -> dict[str, float]:
        """Every role-specific quantity an amount could ground against."""
        q = {
            "variance": self.truth.variance_usd,
            "ledger": self.truth.ledger_liabilities_usd,
            "onchain": self.truth.onchain_reserves_usd,
            "explained": self.explained,
            "unexplained": self.unexplained,
        }
        q.update({f"threshold:{k}": v for k, v in THRESHOLD_VALUES.items()})
        q.update({f"item:{ref}": it.amount_usd for ref, it in self.cited.items()})
        return q


def _context(text: str, masked: str, parsed: dict, truth: CaseTruth) -> _Context:
    refs = [str(r.get("reference", "")) for r in parsed.get("reconciling_items") or [] if isinstance(r, dict)]
    cited, cited_any = {}, set()
    for ref in refs:
        item = truth.item_by_reference(ref)
        if item is None:
            continue
        cited_any.add(item.reference)
        if truth.eligibility(item).eligible:
            cited[item.reference] = item
    explained, unexplained = explained_from_citations(refs, truth)
    return _Context(text=text, masked=masked, truth=truth, cited=cited, cited_any=frozenset(cited_any),
                    explained=explained, unexplained=unexplained)


def _check_amount(toks: list[_Tok], i: int, binding: dict, ctx: _Context) -> dict:
    """Verdict for one amount: ok, FM-001 or review, with what it was checked against."""
    amt = toks[i]
    value = amt.data["value"]
    role = binding["role"]
    truth = ctx.truth
    rec = {"verdict": "ok"}
    signed: float | None = None  # the signed quantity a negative amount must agree with

    if role == "variance":
        target = truth.variance_usd
        rec["expected"] = abs(target)
        if _near(value, target):
            signed = target
        else:
            fits = [k for k, q in ctx.quantities().items() if k != "variance" and _near(value, q)]
            if fits:
                rec.update(verdict="review", reason="variance_amount_fits_other_role", fits=fits)
            else:
                rec.update(verdict="FM-001", reason="variance_amount_mismatch")
    elif role in ("ledger", "onchain", "explained", "unexplained"):
        target = {
            "ledger": truth.ledger_liabilities_usd,
            "onchain": truth.onchain_reserves_usd,
            "explained": ctx.explained,
            "unexplained": ctx.unexplained,
        }[role]
        rec["expected"] = abs(target)
        if _near(value, target):
            signed = target if role in ("explained", "unexplained") else None
            if amt.data["negative"] and role in ("ledger", "onchain"):
                rec.update(verdict="review", reason="negative_balance")
        else:
            rec.update(verdict="FM-001", reason=f"{role}_amount_mismatch")
    elif role == "threshold":
        name = _threshold_name(toks, i, ctx)
        allowed = [THRESHOLD_VALUES[name]] if name else list(THRESHOLD_VALUES.values())
        rec["expected"] = allowed
        rec["threshold_name"] = name or "unnamed"
        if not any(_near(value, v) for v in allowed):
            rec.update(verdict="FM-001", reason="threshold_amount_mismatch")
        elif amt.data["negative"]:
            rec.update(verdict="review", reason="negative_threshold")
    elif role == "item":
        item_rec, signed = _check_item_amount(toks, i, binding["ref_token"], ctx)
        rec.update(item_rec)
    else:  # unresolved
        fits = [k for k, q in ctx.quantities().items() if _near(value, q)]
        if fits:
            rec.update(verdict="review", reason="amount_role_unresolved", fits=fits)
        else:
            rec.update(verdict="FM-001", reason="amount_role_unresolved_and_ungrounded", review=True)

    if rec["verdict"] == "ok" and signed is not None and (
        (amt.data["negative"] and signed > 0) or (amt.data["positive"] and signed < 0)
    ):
        rec.update(verdict="review", reason="amount_sign_disagrees")
    if amt.data["bare"] and rec["verdict"] == "FM-001":
        # a number with no currency marker may be a unit count, so it is not fired on
        rec.update(verdict="review", reason="bare_number_" + rec["reason"])
    return rec


def _threshold_name(toks: list[_Tok], i: int, ctx: _Context) -> str:
    """The policy threshold a threshold-role amount is named as, or "" if unnamed:
    the qualified threshold name nearest the amount in the text between its
    neighbouring amounts, within five tokens either side."""
    amt = toks[i]
    lo = max([t.end for t in toks[max(0, i - 5):i] if t.kind == "amount"] + [toks[max(0, i - 5)].start])
    hi = min([t.start for t in toks[i + 1:i + 6] if t.kind == "amount"] + [toks[min(len(toks) - 1, i + 5)].end])
    best: tuple[int, str] | None = None
    for pattern, name, form in V.THRESHOLD_NAMES:
        if form != "qualified" or not name:
            continue
        for m in re.finditer(_w(pattern), ctx.text[lo:hi], _I):
            s, e = lo + m.start(), lo + m.end()
            gap = amt.start - e if e <= amt.start else s - amt.end
            if best is None or gap < best[0]:
                best = (gap, name)
    return best[1] if best else ""


def _check_item_amount(toks: list[_Tok], i: int, ref_tok: _Tok, ctx: _Context) -> tuple[dict, float | None]:
    """An item amount grounds only against an item cited in reconciling_items and
    eligible (section 6). Returns the record and the item's signed effect."""
    value = toks[i].data["value"]
    item = ctx.truth.item_by_reference(ref_tok.text)
    rec: dict = {"reference": ref_tok.text}
    if item is None or item.reference not in ctx.cited:
        if item is None:
            why = "reference_not_in_fixture"
        elif item.reference not in ctx.cited_any:
            why = "reference_not_cited"
        else:
            why = "reference_ineligible"
        rec.update(verdict="FM-001", reason=why)
        return rec, None
    rec["expected"] = item.amount_usd
    if _near(value, item.amount_usd):
        rec["verdict"] = "ok"
        return rec, item_effect_usd(item)
    # "WD-1 and WD-2, totaling $X": a stated sum of every item named in the clause
    named = []
    for t in toks:
        if t.kind == "ref":
            it = ctx.truth.item_by_reference(t.text)
            if it is not None and all(it.reference != n.reference for n in named):
                named.append(it)
            elif it is None:
                named = []
                break
    if len(named) > 1 and all(n.reference in ctx.cited for n in named):
        total = round(sum(n.amount_usd for n in named), 2)
        if _near(value, total):
            rec.update(verdict="ok", resolution="sum_of_clause_items", expected=total)
            return rec, round(sum(item_effect_usd(n) for n in named), 2)
    rec.update(verdict="FM-001", reason="item_amount_mismatch")
    return rec, None


# --- Statements -------------------------------------------------------------------------

def _negated(clause_text: str, rel_start: int, rel_end: int) -> bool:
    """A negation word in the few words before the statement (not across a comma),
    or a negating phrase right after it."""
    before = clause_text[:rel_start].rsplit(",", 1)[-1]
    words = re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?", before)
    tail = " ".join(words[-(V.WINDOWS["negation_words_before"] + 1):])
    if _NEG_BEFORE_RE.search(tail):
        return True
    return bool(_NEG_AFTER_RE.match(clause_text[rel_end:].lstrip()))


def _direction_statements(clause: str, offset: int) -> list[dict]:
    out = []
    for m in _DIRECTION_RE.finditer(clause):
        out.append({
            "kind": "direction",
            "text": m.group(0),
            "start": offset + m.start(),
            "end": offset + m.end(),
            "direction": _DIRECTIONS[_index(m)][0],
            "negated": _negated(clause, m.start(), m.end()),
        })
    return out


def _split_claims(clause: str, offset: int) -> list[dict]:
    """Explained or unexplained portion stated without an amount ("fully explained",
    "no item explains the variance"), each a $0 claim for its role."""
    out = []
    for m in _SPLIT_CLAIM_RE.finditer(clause):
        role, pattern = V.SPLIT_CLAIMS[_index(m)]
        own_negation = bool(_SELF_NEGATED_RE.match(m.group(0)))
        out.append({
            "kind": "split_claim", "text": m.group(0), "start": offset + m.start(), "end": offset + m.end(),
            "role": role, "value": 0.0,
            "negated": (not own_negation) and _negated(clause, m.start(), m.end()),
        })
    return out


def _band_statements(clause: str, offset: int, amounts_at: dict[int, float]) -> list[dict]:
    """Band statements in one masked clause. Each carries the bands it is consistent
    with ("bands"), or a per-band verdict table for band words."""
    out: list[dict] = []
    consumed: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in consumed)

    def emit(m_start: int, m_end: int, **rec) -> None:
        consumed.append((m_start, m_end))
        out.append({
            "kind": "band", "text": clause[m_start:m_end], "start": offset + m_start,
            "end": offset + m_end, "negated": _negated(clause, m_start, m_end), **rec,
        })

    for m in _BAND_RANGE_RE.finditer(clause):
        v1, v2 = amounts_at.get(m.start("a1")), amounts_at.get(m.start("a2"))
        if v1 is not None and v2 is not None and {_is_threshold_value(v1), _is_threshold_value(v2)} == {True}:
            if not _near(v1, v2):
                emit(m.start(), m.end(), statement="range", bands=["M2"])
    for m in _BAND_STMT_RE.finditer(clause):
        if not free(m.start(), m.end()):
            continue
        side = "above" if _BAND_CMP["above"].fullmatch(m.group("cmp")) else "below"
        value = None
        if m.group("amt"):
            v = amounts_at.get(m.start("amt"))
            if v is not None and _is_threshold_value(v):
                value = v
            elif not (m.group("tref1")):
                continue  # a comparison with a non-threshold amount is not a band statement
        tref = m.group("tref1") or m.group("tref2")
        if value is None and tref:
            for pattern, name, _form in V.THRESHOLD_NAMES:
                if re.fullmatch(pattern, tref, _I):
                    value = THRESHOLD_VALUES.get(name)
                    break
        if value is None:
            emit(m.start(), m.end(), statement="comparison", side=side, bands=None, reason="band_threshold_unresolved")
            continue
        if _near(value, M1_CEILING):
            bands = ["M2", "M3"] if side == "above" else ["M1"]
        else:
            bands = ["M3"] if side == "above" else ["M1", "M2"]
        emit(m.start(), m.end(), statement="comparison", side=side, threshold=value, bands=bands)
    for m in _TREF_RE.finditer(clause):
        if free(m.start(), m.end()):
            consumed.append((m.start(), m.end()))  # a threshold named with no comparison claims no band
    for m in _BAND_LABEL_RE.finditer(clause):
        if free(m.start(), m.end()):
            emit(m.start(), m.end(), statement="label", bands=[m.group(0)])
    for m in _BAND_WORD_RE.finditer(clause):
        if free(m.start(), m.end()):
            row = V.BAND_WORDS[_index(m)]
            emit(m.start(), m.end(), statement="word", verdicts={"M1": row[1], "M2": row[2], "M3": row[3]},
                 reason="band_word_ambiguous")
    return out


def _asset_statements(clause: str, offset: int) -> list[dict]:
    out = []
    consumed: list[tuple[int, int]] = []
    for m in _ASSET_NAME_RE.finditer(clause):
        kind, name = _ASSET_NAME_ROWS[_index(m)][0].split(":", 1)
        consumed.append(m.span())
        rec = {"text": m.group(0), "start": offset + m.start(), "end": offset + m.end(),
               "negated": _negated(clause, m.start(), m.end())}
        if kind == "chain":
            out.append({"kind": "chain", "chain": name, **rec})
        elif kind == "other_chain":
            out.append({"kind": "chain", "chain": "other:" + m.group(0).lower(), **rec})
        else:
            out.append({"kind": "asset", "asset": name, "form": "name", **rec})
    for m in _TICKER_RE.finditer(clause):
        if all(m.end() <= a or m.start() >= b for a, b in consumed):
            out.append({"kind": "asset", "asset": m.group(0), "form": "ticker", "text": m.group(0),
                        "start": offset + m.start(), "end": offset + m.end(),
                        "negated": _negated(clause, m.start(), m.end())})
    return out


def _period_statements(text: str, clause_start: int, toks: list[_Tok]) -> list[dict]:
    out = []
    for t in toks:
        if t.kind not in ("date", "period", "quarter"):
            continue
        rel = t.start - clause_start
        clause_text = text[clause_start:t.start]
        rec = {"text": t.text, "start": t.start, "end": t.end}
        if t.kind == "quarter":
            out.append({"kind": "period", "form": "quarter", "period": None, "negated": False, **rec})
        elif t.kind == "period":
            period = f"{t.data['year']:04d}-{t.data['month']:02d}"
            out.append({"kind": "period", "form": "month_year", "period": period,
                        "negated": _negated(text[clause_start:], rel, rel + len(t.text)), **rec})
        else:
            cued = bool(_PERIOD_CUE_RE.search(clause_text))
            if not cued:
                out.append({"kind": "date", "form": "transaction_date", "period": None, **rec})
                continue
            d = t.data
            if d["ambiguous"] or not d["valid"]:
                out.append({"kind": "period", "form": "cued_date", "period": None, "ambiguous": True,
                            "negated": False, **rec})
                continue
            period = f"{d['year']:04d}-{d['month']:02d}"
            out.append({"kind": "period", "form": "cued_date", "period": period, "day": d["day"],
                        "negated": _negated(text[clause_start:], rel, rel + len(t.text)), **rec})
    return out


def _non_latin(text: str) -> list[str]:
    bad = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") and "LATIN" not in unicodedata.name(ch, ""):
            bad.append(ch)
        elif cat == "Nd" and not ("0" <= ch <= "9"):
            bad.append(ch)
    return bad


# --- The check ------------------------------------------------------------------------------

def check_narrative_consistency(parsed: dict, truth: CaseTruth) -> NarrativeResult:
    """Section 6 narrative consistency over parsed["narrative"]."""
    text = nfkc_spaced(str(parsed.get("narrative") or ""))
    fms: list[str] = []
    reviews: list[str] = []
    extractions: list[dict] = []

    def fire(code: str) -> None:
        if code not in fms:
            fms.append(code)

    def review(reason: str) -> None:
        if reason not in reviews:
            reviews.append(reason)

    if not text:
        return NarrativeResult()

    odd = _non_latin(text)
    if odd:
        review("non_latin_characters: " + "".join(dict.fromkeys(odd)))
    for m in _NON_USD_RE.finditer(text):
        review(f"non_usd_amount: {m.group(0)}")

    specials = _find_specials(text, truth)
    masked = _mask(text, specials)
    ctx = _context(text, masked, parsed, truth)
    toks = _tokenize(text, masked, specials)
    bounds = _clause_bounds(text, specials)
    clause_starts = [0] + bounds
    clause_ends = bounds + [len(text)]
    by_clause: list[list[_Tok]] = [[] for _ in clause_starts]
    for t in toks:
        by_clause[bisect.bisect_right(clause_starts, t.start) - 1].append(t)

    directions: list[dict] = []
    splits: list[dict] = []
    bands: list[dict] = []
    assets: list[dict] = []
    periods: list[dict] = []
    for ci, ctoks in enumerate(by_clause):
        cs, ce = clause_starts[ci], clause_ends[ci]
        # amounts
        for i, t in enumerate(ctoks):
            if t.kind != "amount":
                continue
            binding = _resolve_role(ctoks, i, ctx)
            rec = _check_amount(ctoks, i, binding, ctx)
            entry = {"kind": "amount", "text": t.text, "start": t.start, "end": t.end,
                     "value": t.data["value"], "negative": t.data["negative"], "bare": t.data["bare"]}
            entry.update({k: v for k, v in binding.items() if k != "ref_token"})
            entry.update(rec)
            extractions.append(entry)
            if rec["verdict"] == "FM-001":
                fire("FM-001")
                if rec.get("review"):
                    review(f"{rec['reason']}: {t.text}")
            elif rec["verdict"] == "review":
                review(f"{rec['reason']}: {t.text}")
        clause_masked = masked[cs:ce]
        amounts_at = {t.start - cs: t.data["value"] for t in ctoks if t.kind == "amount"}
        directions += _direction_statements(clause_masked, cs)
        splits += _split_claims(clause_masked, cs)
        bands += _band_statements(clause_masked, cs, amounts_at)
        assets += _asset_statements(clause_masked, cs)
        periods += _period_statements(text, cs, ctoks)

    for e in directions + splits + bands + assets + periods:
        e["text"] = text[e["start"]:e["end"]]  # statements were matched on masked text

    # explained or unexplained portion stated without an amount (FM-001)
    for c in splits:
        target = ctx.explained if c["role"] == "explained" else ctx.unexplained
        c["expected"] = abs(target)
        if c["negated"]:
            c["verdict"] = "review"
            review(f"split_claim_negated: {c['text']}")
        elif abs(target) <= USD_TOLERANCE:
            c["verdict"] = "ok"
        else:
            c["verdict"] = "FM-001"
            fire("FM-001")
    extractions += splits

    # direction (FM-003)
    named = set()
    for d in directions:
        if d["negated"]:
            d["verdict"] = "review"
            review(f"direction_negated: {d['text']}")
        else:
            named.add(d["direction"])
    if len(named) > 1:
        review("both_directions_named")
    for d in directions:
        if d["negated"]:
            continue
        if len(named) > 1:
            d["verdict"] = "review"
        elif d["direction"] == truth.direction:
            d["verdict"] = "ok"
        else:
            d["verdict"] = "FM-003"
            fire("FM-003")
    extractions += directions

    # band (FM-002)
    for b in bands:
        if b["negated"]:
            b["verdict"] = "review"
            review(f"band_negated: {b['text']}")
        elif b.get("verdicts"):
            v = b["verdicts"][truth.materiality_band]
            b["verdict"] = {"ok": "ok", "contradict": "FM-002", "review": "review"}[v]
        elif b.get("bands") is None:
            b["verdict"] = "review"
        else:
            b["verdict"] = "ok" if truth.materiality_band in b["bands"] else "FM-002"
        if b["verdict"] == "FM-002":
            fire("FM-002")
        elif b["verdict"] == "review" and not b["negated"]:
            review(f"{b['reason']}: {b['text']}")
    extractions += bands

    # asset (FM-016)
    gas = V.GAS_ASSET[truth.chain]
    named_assets = set()
    for a in assets:
        if a["kind"] == "chain":
            if a["chain"] == truth.chain:
                a["verdict"] = "ok"
            else:
                a["verdict"] = "review"
                review(f"other_chain_named: {a['text']}")
            continue
        if a["negated"]:
            a["verdict"] = "review"
            review(f"asset_negated: {a['text']}")
            continue
        named_assets.add(a["asset"])
    others = named_assets - {truth.asset}
    gas_only = others == {gas} and gas != truth.asset
    for a in assets:
        if a["kind"] != "asset" or a["negated"]:
            continue
        if a["asset"] == truth.asset:
            a["verdict"] = "ok"
        elif gas_only:
            a["verdict"] = "review"
        elif truth.asset in named_assets:
            a["verdict"] = "review"
        else:
            a["verdict"] = "FM-016"
    if others:
        if gas_only:
            review(f"gas_asset_named: {gas}")
        elif truth.asset in named_assets:
            review("second_asset_named: " + ",".join(sorted(others)))
        else:
            fire("FM-016")
    extractions += assets

    # period (FM-016)
    cutoff_day = int(period_cutoff_date(truth.period)[-2:])
    named_periods = set()
    for p in periods:
        if p["kind"] == "date":
            p["verdict"] = "ignored"
            continue
        if p["form"] == "quarter":
            p["verdict"] = "review"
            review(f"quarter_period: {p['text']}")
        elif p.get("ambiguous"):
            p["verdict"] = "review"
            review(f"period_date_ambiguous: {p['text']}")
        elif p["negated"]:
            p["verdict"] = "review"
            review(f"period_negated: {p['text']}")
        else:
            named_periods.add(p["period"])
            if p["form"] == "cued_date" and p["period"] == truth.period and p["day"] != cutoff_day:
                p["verdict"] = "review"
                review(f"period_date_not_cutoff: {p['text']}")
    other_periods = named_periods - {truth.period}
    for p in periods:
        if p.get("verdict") or p.get("period") is None:
            continue
        if p["period"] == truth.period:
            p["verdict"] = "ok"
        elif truth.period in named_periods:
            p["verdict"] = "review"
        else:
            p["verdict"] = "FM-016"
    if other_periods:
        if truth.period in named_periods:
            review("second_period_named: " + ",".join(sorted(other_periods)))
        else:
            fire("FM-016")
    extractions += periods

    order = ("FM-001", "FM-002", "FM-003", "FM-016")
    extractions.sort(key=lambda e: (e["start"], e["kind"]))
    return NarrativeResult(
        failure_modes=tuple(c for c in order if c in fms),
        review_reasons=tuple(reviews),
        extractions=tuple(extractions),
    )
