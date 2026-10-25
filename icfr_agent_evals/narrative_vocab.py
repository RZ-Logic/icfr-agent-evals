"""Fixed vocabularies for the narrative-consistency extractor (section 6).

Every word list, phrase pattern and window the extractor in narrative.py reads
lives here, so the vocabulary is one lockable object. vocab_sha256() hashes a
canonical JSON of all of it; section 13 records that hash at the tag, and a change
to any entry changes the hash.

Conventions:
  - Entries are regular-expression fragments. They are matched case-insensitively
    at word boundaries unless the entry's comment says case-sensitive.
  - The narrative is passed through casefile.nfkc_spaced first, so whitespace in a
    fragment is a single literal space.
  - Lists that the extractor scans as one alternation are ordered: earlier entries
    win at the same start position, so compound phrases come before their parts.
"""

from __future__ import annotations

import hashlib
import json

# The extractor replaces every currency amount with this private-use character before
# matching phrase patterns, so a pattern can allow "an amount here" as \ue000+.
AMOUNT_PLACEHOLDER = "\ue000"

# --- Direction (section 6: surplus, shortfall, deficit, excess) ----------------------
# [direction, pattern]. Phrases that name the variance's direction by a single word
# or fixed phrase. Also used as variance cue words for amount-role binding.
DIRECTION_WORDS: list[list[str]] = [
    # "an excess of liabilities over reserves" and "excess liabilities" are shortfalls
    ["shortfall", r"excess (?:of )?(?:the )?(?:(?:ledger|customer|recorded|book) )?liabilit(?:y|ies)(?: over (?:the )?(?:on[- ]?chain )?reserves)?"],
    ["surplus", r"excess (?:of )?(?:the )?(?:on[- ]?chain )?reserves(?: over (?:the )?(?:(?:ledger|customer|recorded|book) )?liabilities)?"],
    ["shortfall", r"excess(?: of)?(?: (?:[A-Za-z-]+|\ue000+)){0,3}? (?:in|of) (?:the )?(?:(?:ledger|customer|recorded|book) )?liabilit(?:y|ies)"],
    ["surplus", r"excess(?: of)?(?: (?:[A-Za-z-]+|\ue000+)){0,3}? (?:in|of) (?:the )?(?:on[- ]?chain )?reserves"],
    ["surplus", r"surplus(?:es)?"],
    ["surplus", r"(?<!in )excess(?:es)?"],
    ["surplus", r"over[- ]?collaterali[sz](?:ed|ation)"],
    ["surplus", r"over[- ]?reserved"],
    ["surplus", r"positive variance"],
    ["shortfall", r"shortfalls?"],
    ["shortfall", r"deficits?"],
    ["shortfall", r"shortages?"],
    ["shortfall", r"(?:fell|falls|fall|falling) short(?: of)?"],
    ["shortfall", r"(?:came|comes|come|coming) up short"],
    ["shortfall", r"short (?:by|of)"],
    ["shortfall", r"under[- ]?collaterali[sz](?:ed|ation)"],
    ["shortfall", r"under[- ]?reserved"],
    ["shortfall", r"negative variance"],
]

# Comparative direction statements: SUBJECT [of AMOUNT] VERB OBJECT. narrative.py
# composes them; a reserves-above-liabilities comparison is a surplus, and so on.
DIRECTION_RESERVE_NOUNS: list[str] = [
    r"(?:on[- ]?chain )?(?:reserves|holdings)",
    r"(?:on[- ]?chain|wallet|reserve) balances?",
]
DIRECTION_LIABILITY_NOUNS: list[str] = [
    r"(?:(?:ledger|customer|recorded|book) )?liabilit(?:y|ies)",
    r"(?:ledger|book) balances?",
    r"ledger", r"books",
]
DIRECTION_COPULAS: list[str] = [
    r"were", r"was", r"are", r"is", r"stood", r"remained", r"remain", r"remains", r"came in",
]
DIRECTION_ABOVE_VERBS: list[str] = [
    r"above", r"higher than", r"greater than", r"more than", r"in excess of",
    r"exceed(?:s|ed|ing)?", r"surpass(?:es|ed|ing)?", r"outstrip(?:s|ped|ping)?",
]
DIRECTION_BELOW_VERBS: list[str] = [
    r"below", r"lower than", r"less than", r"under",
    r"(?:fell|falls|fall|falling) short of", r"short of",
    r"trail(?:s|ed|ing)?", r"lag(?:s|ged|ging)?(?: behind)?",
]
# Words or amounts allowed between a comparative's subject and its verb, as in
# "reserves of $9,880,000 fell short of liabilities" or "reserves held on Bitcoin came
# in $120,000 below liabilities".
DIRECTION_SUBJECT_GAP_WORDS = 6

# --- Negation (section 6: a negated direction or band statement routes to review) ----
# Matched in the few words before a statement, inside its clause.
NEGATION_BEFORE: list[str] = [
    r"no", r"not", r"never", r"none", r"neither", r"nor", r"without", r"cannot",
    r"nothing", r"zero", r"absent", r"\w+n['\u2019]t", r"rather than", r"instead of",
    r"no longer",
]
# Matched immediately after a statement, e.g. "a shortfall does not exist".
NEGATION_AFTER: list[str] = [
    r"(?:does|did|do) not exist",
    r"(?:doesn|didn|don)['\u2019]t exist",
    r"(?:is|was|are|were) (?:not|n['\u2019]t) (?:present|observed|indicated|identified|found|evident)",
    r"(?:isn|wasn|aren|weren)['\u2019]t (?:present|observed|indicated|identified|found|evident)",
]

# --- Bands and thresholds (section 3 bands, section 6 band statements) ---------------
BAND_LABEL = r"M[123]"  # case-sensitive

# Named thresholds: [pattern, value, form]. value is the policy threshold the phrase
# names, "M1_CEILING" or "M3_FLOOR", or "" when it names no specific one. form
# "qualified" names one threshold wherever it appears; "bare" ("the floor") names one
# only in a band comparison, so an amount stated as "the $250,000 floor" is checked
# against either threshold.
THRESHOLD_NAMES: list[list[str]] = [
    [r"M1 (?:ceiling|cap|limit|threshold|boundary)", "M1_CEILING", "qualified"],
    [r"M2 (?:floor|threshold)", "M1_CEILING", "qualified"],
    [r"M2 (?:ceiling|cap|limit|boundary)", "M3_FLOOR", "qualified"],
    [r"M3 (?:floor|threshold|limit|boundary)", "M3_FLOOR", "qualified"],
    [r"report(?:ing|able) (?:threshold|floor|limit|level)", "M1_CEILING", "qualified"],
    [r"material[- ]risk (?:threshold|floor|limit|level)", "M3_FLOOR", "qualified"],
    [r"materiality (?:threshold|floor|limit|level)", "", "qualified"],
    [r"escalation (?:threshold|floor|limit|level)", "", "qualified"],
    [r"floor", "M1_CEILING", "bare"],
    [r"thresholds?", "", "bare"],
    [r"limits?", "", "bare"],
    [r"ceiling", "", "bare"],
    [r"materiality", "", "bare"],
]
# Comparators that place the variance above or below a threshold.
BAND_COMPARATORS: dict[str, list[str]] = {
    "above": [
        r"at or above", r"above", r"over", r"exceed(?:s|ed|ing)?", r"greater than",
        r"more than", r"higher than", r"in excess of", r"surpass(?:es|ed|ing)?",
        r"cross(?:es|ed|ing)?", r"breach(?:es|ed|ing)?", r"beyond",
    ],
    "below": [
        r"at or below", r"below", r"under", r"beneath", r"less than", r"lower than", r"within",
    ],
}
BAND_RANGE_WORD = r"between"

# Band words: [pattern, verdict for M1, verdict for M2, verdict for M3]. Verdicts:
# "ok" consistent, "contradict" fires FM-002, "review" routes to requires_human_review.
BAND_WORDS: list[list[str]] = [
    [r"immaterial", "ok", "contradict", "contradict"],
    [r"de minimis", "ok", "contradict", "contradict"],
    [r"below[- ]floor", "ok", "contradict", "contradict"],
    [r"material[- ]risk", "contradict", "contradict", "ok"],
    [r"reportable (?:band|range|tier|category)", "contradict", "ok", "contradict"],
    [r"reportable", "contradict", "ok", "ok"],
    [r"material(?: weakness(?:es)?)?", "contradict", "review", "ok"],
]

# --- Amount roles (section 6: each amount is checked with its accounting role) -------
# [role, pattern], scanned as one ordered alternation. Roles: variance, ledger,
# onchain, threshold, explained, unexplained, and neutral (consumed, binds nothing).
# The variance role also takes every DIRECTION_WORDS pattern.
ROLE_CUES: list[list[str]] = [
    ["neutral", r"in excess of"],
    ["neutral", r"timing differences?"],
    ["unexplained", r"(?:unexplained|unreconciled|unaccounted|residual|remaining) (?:variance|shortfall|surplus|deficit|difference|balance|portion|amount|gap|excess|remainder)"],
    ["unexplained", r"(?:remains?|remained|is|was|are|were|left|stays?|stayed) (?:unexplained|unreconciled|unaccounted for)"],
    ["unexplained", r"(?:not|not yet) (?:explained|reconciled|accounted for)"],
    ["unexplained", r"net of (?:the )?(?:(?:open|eligible|cited|pending|reconciling) )*items?"],
    ["unexplained", r"after (?:the )?(?:(?:open|eligible|cited|pending) )*reconciling items?"],
    ["unexplained", r"unexplained|unreconciled|unaccounted for|remainder|residual|remaining"],
    ["explained", r"(?:explained|reconciled) (?:portion|amount|part|total)"],
    ["explained", r"explain(?:s|ed|ing)?|accounted for|accounts? for|accounting for|reconciled|offset(?:s|ting)?"],
    ["ledger", r"(?:(?:ledger|customer|recorded|book) )?liabilit(?:y|ies)|(?:ledger|book) balances?"],
    ["ledger", r"ledger(?= ?:)"],
    ["onchain", r"(?:on[- ]?chain )?(?:reserves|holdings)|(?:on[- ]?chain|wallet|reserve) balances?"],
    ["onchain", r"(?:wallets?|addresses) (?:hold|holds|held|holding)|held on[- ]?chain"],
    ["onchain", r"on[- ]?chain(?= ?:)"],
    ["threshold", r"thresholds?|floors?|ceilings?|limits?|bands?"],
    ["variance", r"variances?|differences?|gap|net position"],
]
# Phrases that state the explained or unexplained portion without an amount. Each
# is checked as a $0 claim: [role, pattern], where role is the quantity stated to be
# zero. Phrases that carry their own negation are listed in SPLIT_CLAIMS_SELF_NEGATED
# and are not re-checked for negation.
SPLIT_CLAIMS: list[list[str]] = [
    ["unexplained", r"(?:fully|entirely|completely|wholly) (?:explained|reconciled|accounted for)"],
    ["unexplained", r"(?:explained|reconciled|accounted for) (?:in full|in its entirety|completely|entirely|fully)"],
    ["unexplained", r"explain(?:s|ed)? (?:the )?(?:(?:entire|whole|full) )?(?:variance|shortfall|surplus|deficit|difference) (?:in full|in its entirety|completely|entirely|fully)"],
    ["unexplained", r"explain(?:s|ed)? (?:the )?(?:entire|whole|full) (?:variance|shortfall|surplus|deficit|difference)"],
    ["unexplained", r"no (?:unexplained|unreconciled|residual|remaining) (?:remainder|balance|amount|portion|difference|variance|shortfall|surplus)"],
    ["unexplained", r"nothing (?:remains |is left )?(?:unexplained|unreconciled)"],
    ["explained", r"(?:fully|entirely|completely|wholly) (?:unexplained|unreconciled)"],
    ["explained", r"none of the (?:variance|shortfall|surplus|deficit|difference) is (?:explained|reconciled|accounted for)"],
    ["explained", r"no (?:(?:eligible|open|reconciling|pending|cited) )*items? (?:explains?|accounts? for|reconciles?|offsets?)"],
]
SPLIT_CLAIMS_SELF_NEGATED: list[str] = [r"no ", r"nothing ", r"none "]

# Words and punctuation that join a cue to its amount without weakening the bond
# ("liabilities of $X", "WD-1 ($X)", "the $X pending withdrawal").
LINK_WORDS: list[str] = [
    "of", "was", "were", "is", "are", "be", "been", "being", "at", "to", "for", "in",
    "a", "an", "the", "this", "that", "its", "total", "totals", "totaling", "totalling",
    "totaled", "totalled", "amounted", "amounting", "amounts", "came", "comes", "stood",
    "stands", "equal", "equals", "equaling", "equalling", "worth", "valued",
    "approximately", "approx", "about", "roughly", "around", "nearly", "almost",
    "item", "items", "withdrawal", "withdrawals", "deposit", "deposits", "transfer",
    "transfers", "fee", "fees", "settlement", "settlements", "pending", "open",
    "reconciling", "eligible", "cited", "internal", "outbound", "inbound", "network",
    "in-flight", "flight", "on", "held", "mainnet", "blockchain", "chain",
    "relates", "related", "relating", "attributable", "attributed", "due", "from",
    "represents", "representing", "reflects", "reflecting",
]
# Chain names (CHAIN_NAMES) and in-scope tickers also link: "reserves on Solana of $X".
LINK_PUNCT: list[str] = ["(", ")", ":", "=", "-", "\u2013", "\u2014", "'", '"']
# Comparators that make an amount stated right before a balance the variance
# ("reserves came in $120,000 below ledger liabilities").
COMPARISON_AFTER_AMOUNT: list[str] = [
    r"below", r"above", r"under", r"over", r"short of", r"more than", r"less than",
    r"higher than", r"lower than", r"greater than", r"in excess of", r"ahead of", r"behind",
]
# Words that make "by $X" a comparison amount ("reserves exceeded liabilities by $X").
COMPARISON_WORDS: list[str] = [
    r"exceed(?:s|ed|ing)?", r"above", r"below", r"under", r"over", r"trail(?:s|ed|ing)?",
    r"lag(?:s|ged|ging)?", r"behind", r"surpass(?:es|ed|ing)?", r"outstrip(?:s|ped|ping)?",
    r"higher", r"lower", r"greater", r"less", r"more", r"short",
]
# A clause with this word states amounts in parallel lists, so no amount in it binds.
PARALLEL_LIST_WORDS: list[str] = [r"respectively"]

# --- Assets (section 6: asset tickers; FM-016 on a contradicting asset) --------------
IN_SCOPE_TICKERS: list[str] = ["BTC", "ETH", "SOL", "USDC", "USDT", "XRP", "ADA"]  # case-sensitive
# Full names of in-scope assets that are not chain names.
ASSET_NAMES: dict[str, list[str]] = {
    "BTC": [], "ETH": [r"ether"], "SOL": [], "USDC": [r"usd coin"],
    "USDT": [r"tether(?: usd)?"], "XRP": [], "ADA": [],
}
OTHER_TICKERS: list[str] = [  # case-sensitive
    "USDG", "DAI", "BNB", "DOGE", "TRX", "DOT", "AVAX", "MATIC", "POL", "LTC", "LINK",
    "XLM", "BCH", "SHIB", "TON", "ATOM", "UNI", "PYUSD", "FDUSD", "TUSD", "BUSD", "USDE",
    "RLUSD", "WBTC", "WETH", "STETH", "EURC", "ARB", "APT", "SUI", "NEAR", "FIL", "ETC",
    "HBAR", "ALGO", "XTZ", "PEPE", "XAUT", "PAXG", "ICP", "KAS", "INJ", "SEI", "TIA",
    "JUP", "BONK", "WIF", "MKR", "AAVE", "CRV", "LDO", "XMR", "ZEC",
]
OTHER_ASSET_NAMES: dict[str, str] = {
    r"bitcoin cash": "BCH", r"ethereum classic": "ETC", r"wrapped bitcoin": "WBTC",
    r"wrapped ether": "WETH", r"staked ether": "STETH", r"dogecoin": "DOGE",
    r"litecoin": "LTC", r"polkadot": "DOT", r"chainlink": "LINK", r"shiba inu": "SHIB",
    r"binance coin": "BNB", r"toncoin": "TON", r"paypal usd": "PYUSD",
    r"first digital usd": "FDUSD", r"global dollar": "USDG", r"tether gold": "XAUT",
    r"pax gold": "PAXG", r"true ?usd": "TUSD", r"binance usd": "BUSD", r"euro coin": "EURC",
    r"monero": "XMR", r"zcash": "ZEC",
}
# Chain names are not asset statements (section 6), so "Ethereum network" on a USDC
# case names no second asset. Keys are casefile.ASSET_CHAIN values.
CHAIN_NAMES: dict[str, list[str]] = {
    "bitcoin": [r"bitcoin"], "ethereum": [r"ethereum"], "solana": [r"solana"],
    "xrpl": [r"xrpl", r"xrp ledger"], "cardano": [r"cardano"],
}
OTHER_CHAIN_NAMES: list[str] = [
    r"tron", r"polygon", r"arbitrum", r"bnb (?:smart )?chain", r"binance smart chain",
    r"avalanche c-chain",
]
NON_ASSET_WORDS: list[str] = [r"mainnet", r"network", r"blockchain"]
# Native gas asset of each chain. Naming it on a token case (USDC, USDT on Ethereum)
# can describe fees, so it routes to review instead of firing FM-016.
GAS_ASSET: dict[str, str] = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "xrpl": "XRP", "cardano": "ADA",
}
# Token-standard names that look like reference numbers but are not.
TOKEN_STANDARDS: list[str] = ["ERC-20", "ERC-721", "ERC-1155", "BEP-20", "TRC-20", "ERC20", "BEP20", "TRC20"]

# --- Periods (section 6: period statements; FM-016 on a contradicting period) --------
MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
    "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11,
    "nov": 11, "december": 12, "dec": 12,
}
# Cues that make a full date a period statement ("as of 2025-12-31"). A date with no
# cue is a transaction date and states no period.
PERIOD_CUES: list[str] = [
    r"(?:period|month|quarter)[- ](?:ended|ending|end)", r"as of", r"as at",
    r"month[- ]end", r"period[- ]end", r"for the (?:period|month)(?: ended| ending)?",
]
# Filler allowed between a cue and its date.
PERIOD_CUE_FILLER: list[str] = [r"the", r"on", r"at", r",", r":"]
# Quarter forms route to review: the case period is one month.
QUARTER_PATTERNS: list[str] = [
    r"Q[1-4] ?['-]? ?(?:20)?[0-9]{2}", r"[1-4]Q ?(?:20)?[0-9]{2}",
    r"(?:first|second|third|fourth|1st|2nd|3rd|4th) quarter(?: of)?(?: 20[0-9]{2})?",
    r"Q[1-4]", r"quarter[- ](?:end|ended|ending)",
]

# --- Currency amounts (section 6: every currency amount) ------------------------------
CURRENCY_PREFIXES: list[str] = [r"US\$", r"\$", r"USD ?-?equivalent", r"USD"]
CURRENCY_SUFFIXES: list[str] = [r"USD[- ]equivalent", r"USD", r"US dollars", r"dollars"]
# Amounts in another currency are not checked; they route to review.
NON_USD_CURRENCY: list[str] = [
    r"[\u20ac\u00a3\u00a5\u20b9\u20a9]", r"EUR", r"GBP", r"JPY", r"CHF", r"CAD", r"AUD", r"SGD",
    r"HKD", r"CNY", r"[ACSH]\$", r"NZ\$", r"HK\$",
]
MAGNITUDE_SUFFIXES: dict[str, int] = {
    "billion": 1_000_000_000, "bn": 1_000_000_000, "b": 1_000_000_000,
    "million": 1_000_000, "mm": 1_000_000, "mn": 1_000_000, "m": 1_000_000,
    "thousand": 1_000, "k": 1_000,
}
# A bare number followed by one of these is a quantity of units, not a currency amount.
UNIT_WORDS: list[str] = [
    r"coins?", r"tokens?", r"units?", r"confirmations?", r"blocks?", r"transactions?",
    r"%", r"percent", r"bps", r"basis points", r"wallets?", r"days?", r"sats?",
    r"satoshis?", r"wei", r"gwei", r"lamports?", r"drops", r"lovelace",
]
# A bare number counts only when it is comma-grouped or has at least this many digits,
# so years, days and short counts are not read as amounts.
BARE_NUMBER_MIN_DIGITS = 5

# --- Clauses and windows ---------------------------------------------------------------
# Words before a full stop that do not end a sentence.
ABBREVIATIONS: list[str] = [
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "inc", "ltd", "corp", "co", "no", "approx", "vs", "e.g", "i.e", "u.s", "st", "mr", "ms",
]
WINDOWS: dict[str, int] = {
    "negation_words_before": 3,  # a negation this many words or fewer before a statement
    "cue_max_distance": 4,       # a cue farther than this many binding words binds nothing
    "range_window": 6,           # "between $25,000 and $250,000" spans at most this many tokens
}


def all_vocabularies() -> dict:
    """Every vocabulary the extractor reads, keyed by name."""
    return {
        "AMOUNT_PLACEHOLDER": AMOUNT_PLACEHOLDER,
        "DIRECTION_WORDS": DIRECTION_WORDS,
        "DIRECTION_RESERVE_NOUNS": DIRECTION_RESERVE_NOUNS,
        "DIRECTION_LIABILITY_NOUNS": DIRECTION_LIABILITY_NOUNS,
        "DIRECTION_COPULAS": DIRECTION_COPULAS,
        "DIRECTION_ABOVE_VERBS": DIRECTION_ABOVE_VERBS,
        "DIRECTION_BELOW_VERBS": DIRECTION_BELOW_VERBS,
        "DIRECTION_SUBJECT_GAP_WORDS": DIRECTION_SUBJECT_GAP_WORDS,
        "NEGATION_BEFORE": NEGATION_BEFORE,
        "NEGATION_AFTER": NEGATION_AFTER,
        "BAND_LABEL": BAND_LABEL,
        "THRESHOLD_NAMES": THRESHOLD_NAMES,
        "BAND_COMPARATORS": BAND_COMPARATORS,
        "BAND_RANGE_WORD": BAND_RANGE_WORD,
        "BAND_WORDS": BAND_WORDS,
        "ROLE_CUES": ROLE_CUES,
        "LINK_WORDS": LINK_WORDS,
        "LINK_PUNCT": LINK_PUNCT,
        "SPLIT_CLAIMS": SPLIT_CLAIMS,
        "SPLIT_CLAIMS_SELF_NEGATED": SPLIT_CLAIMS_SELF_NEGATED,
        "COMPARISON_AFTER_AMOUNT": COMPARISON_AFTER_AMOUNT,
        "COMPARISON_WORDS": COMPARISON_WORDS,
        "PARALLEL_LIST_WORDS": PARALLEL_LIST_WORDS,
        "IN_SCOPE_TICKERS": IN_SCOPE_TICKERS,
        "ASSET_NAMES": ASSET_NAMES,
        "OTHER_TICKERS": OTHER_TICKERS,
        "OTHER_ASSET_NAMES": OTHER_ASSET_NAMES,
        "CHAIN_NAMES": CHAIN_NAMES,
        "OTHER_CHAIN_NAMES": OTHER_CHAIN_NAMES,
        "NON_ASSET_WORDS": NON_ASSET_WORDS,
        "GAS_ASSET": GAS_ASSET,
        "TOKEN_STANDARDS": TOKEN_STANDARDS,
        "MONTHS": MONTHS,
        "PERIOD_CUES": PERIOD_CUES,
        "PERIOD_CUE_FILLER": PERIOD_CUE_FILLER,
        "QUARTER_PATTERNS": QUARTER_PATTERNS,
        "CURRENCY_PREFIXES": CURRENCY_PREFIXES,
        "CURRENCY_SUFFIXES": CURRENCY_SUFFIXES,
        "NON_USD_CURRENCY": NON_USD_CURRENCY,
        "MAGNITUDE_SUFFIXES": MAGNITUDE_SUFFIXES,
        "UNIT_WORDS": UNIT_WORDS,
        "BARE_NUMBER_MIN_DIGITS": BARE_NUMBER_MIN_DIGITS,
        "ABBREVIATIONS": ABBREVIATIONS,
        "WINDOWS": WINDOWS,
    }


def canonical_json(vocab: dict | None = None) -> str:
    """Canonical JSON of the vocabularies: sorted keys, no insignificant whitespace."""
    data = all_vocabularies() if vocab is None else vocab
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def vocab_sha256(vocab: dict | None = None) -> str:
    """SHA-256 of the canonical vocabulary JSON, recorded at the tag (section 13)."""
    return hashlib.sha256(canonical_json(vocab).encode("utf-8")).hexdigest()
