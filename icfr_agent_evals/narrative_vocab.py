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
    ["shortfall", r"(?<=\ue000 )short(?![- ](?:term|lived)| of| by)"],  # "$120,000 short"
    ["shortfall", r"under[- ]?collaterali[sz](?:ed|ation)"],
    ["shortfall", r"under[- ]?reserved"],
    ["shortfall", r"negative variance"],
    ["surplus", r"over[- ]?funded"],
    ["shortfall", r"under[- ]?funded"],
    ["shortfall", r"net short"],
    ["shortfall", r"(?:reserves|holdings) (?:are|were|is|was|remain|remained) (?:insufficient|inadequate|deficient)"],
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
    r"above", r"over", r"higher than", r"greater than", r"more than", r"in excess of",
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
    r"no longer", r"fail(?:s|ed|ing)? to",
]
# Matched immediately after a statement, e.g. "a shortfall does not exist".
NEGATION_AFTER: list[str] = [
    r"(?:does|did|do) not exist",
    r"(?:doesn|didn|don)['\u2019]t exist",
    r"(?:is|was|are|were) (?:not|n['\u2019]t) (?:present|observed|indicated|identified|found|evident)",
    r"(?:isn|wasn|aren|weren)['\u2019]t (?:present|observed|indicated|identified|found|evident)",
    r"(?:is|was|are|were) (?:nil|zero|nonexistent|non-existent)",
    r"(?:has|have|had) (?:since |now |already |subsequently )?been (?:cured|eliminated|resolved|cleared|closed|remediated|reversed|corrected)",
    r"no longer exists?",
    r"(?:is|was) (?:now |since )?(?:cured|eliminated|gone)",
]
# A negation followed by one of these nouns (after optional item adjectives) negates
# the noun, not a direction stated later in the clause: "No reconciling item explains
# the shortfall" states no negated shortfall.
NEGATION_NOUN_ADJECTIVES: list[str] = ["eligible", "open", "cited", "pending", "reconciling", "single", "other"]
NEGATION_ITEM_NOUNS: list[str] = [
    "item", "items", "entry", "entries", "transaction", "transactions", "transfer", "transfers",
    "amount", "amounts", "portion", "part", "reference", "references", "deposit", "deposits",
    "withdrawal", "withdrawals", "fee", "fees", "adjustment", "adjustments",
]

# --- Bands and thresholds (section 3 bands, section 6 band statements) ---------------
BAND_LABEL = r"[Mm][- ]?[123]"  # m1, M-1 and M 1 read as the label M1 (the structured field casefolds too)
# "Band 1", "materiality band no. 2", "materiality tier 3": numbered band forms.
BAND_NUMBER_LABEL = r"(?:materiality )?band (?:no\.? |number |#)?([123])|materiality tier ([123])"

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
# Band positions, matched before threshold names claim "materiality": [pattern, M1, M2, M3].
BAND_POSITION_WORDS: list[list[str]] = [
    [r"lowest (?:materiality )?(?:band|tier|category)", "ok", "contradict", "contradict"],
    [r"(?:highest|top) (?:materiality )?(?:band|tier|category)", "contradict", "contradict", "ok"],
    [r"(?:middle|intermediate) (?:materiality )?(?:band|tier|category)", "contradict", "ok", "contradict"],
]
# Phrases that state the band through escalation and carry their own negation, so a
# "not" inside them does not mark the statement negated: [pattern, M1, M2, M3].
ESCALATION_CLAIMS: list[list[str]] = [
    [r"(?:does|do|did|would|will) not (?:require|need|warrant) (?:any )?escalation"
     r"|(?:requires?|needs?|warrants?) no escalation|no escalation (?:is |was )?(?:needed|required|necessary|warranted)"
     r"|no (?:need|requirement) (?:for|to) escalat(?:e|ion)|escalation (?:is|was) (?:not|un)(?: )?(?:required|needed|necessary|warranted)",
     "ok", "contradict", "contradict"],
    [r"(?:requires?|required|needs?|warrants?|warranted) escalation|escalation (?:is |was )?(?:required|needed|necessary|warranted)"
     r"|(?:must|should|will|to) be escalated",
     "review", "ok", "ok"],
]
# Severity words that state a band only when the clause names the variance; with
# another subject ("fees were negligible") they route to review.
SEVERITY_WORDS: list[list[str]] = [
    [r"negligible|trivial|insignificant|inconsequential|de minimis", "ok", "contradict", "contradict"],
    [r"within (?:the )?(?:\w+ )?tolerances?", "ok", "review", "contradict"],
]
# A clause with one of these words and no resolved band statement routes to review.
BAND_VOCABULARY: list[str] = [r"bands?", r"materiality", r"escalat\w*", r"tolerances?"]
# Statements that the variance itself is zero: an FM-001-class claim when |variance|
# exceeds the tolerance. Negated forms ("the balances do not agree") state nothing.
ZERO_VARIANCE_CLAIMS: list[str] = [
    r"no (?:net )?variance(?! (?:remains|is left|left|unexplained|unreconciled|outstanding|to explain))",
    r"no (?:net )?(?:discrepancy|shortfall or surplus)",
    r"(?:the )?(?:net )?variance (?:is|was|of|stands at|amounts to) (?:zero|nil|nothing|none)",
    r"(?:reconciliation|reserves|books|ledger|position) (?:is|are|was|were) in balance",
    r"(?:ledger and on[- ]?chain|the two|both) balances (?:agree|agreed|match|matched|tie|tied)(?: out)?",
    r"reconcil(?:es|ed) to zero",
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
    ["unexplained", r"(?:reduces?|reduced|reducing|leaves?|leaving|left|brings?|bringing|brought) (?:the )?(?:variance|shortfall|surplus|deficit|difference|gap|excess) (?:down )?to"],
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
# is checked as a $0 claim: [role, pattern, scope], where role is the quantity stated
# to be zero. scope "subject": the phrase itself names the variance. scope "clause":
# the claim counts only when its clause names the variance (SPLIT_CLAIM_SUBJECTS);
# otherwise its subject is unknown ("the withdrawal is fully explained") and it
# routes to review. Phrases that carry their own negation are listed in
# SPLIT_CLAIMS_SELF_NEGATED and are not re-checked for negation.
_V = r"(?:variance|shortfall|surplus|deficit|difference)"
SPLIT_CLAIMS: list[list[str]] = [
    ["unexplained", rf"(?:fully|entirely|completely|wholly) (?:explains?|reconciles?|accounts? for) (?:the )?(?:\ue000+ )?{_V}", "subject"],
    ["unexplained", rf"(?:explains?|explained|reconciles?|reconciled|accounts? for|accounted for) (?:the )?(?:entire|whole|full|total) (?:\ue000+ )?{_V}", "subject"],
    ["unexplained", rf"(?:explains?|explained|reconciles?|reconciled|accounts? for|accounted for) (?:the )?(?:\ue000+ )?{_V} (?:in full|in its entirety|completely|entirely|fully)", "subject"],
    ["unexplained", r"no (?:unexplained|unreconciled|residual|remaining) (?:remainder|balance|amount|portion|difference|variance|shortfall|surplus)", "subject"],
    ["unexplained", r"nothing (?:remains |is left )?(?:unexplained|unreconciled)", "subject"],
    ["unexplained", r"(?:fully|entirely|completely|wholly) (?:explained|reconciled|accounted for)", "clause"],
    ["unexplained", r"(?:explained|reconciled|accounted for) (?:in full|in its entirety|completely|entirely|fully)", "clause"],
    ["unexplained", r"(?:explains?|explained|reconciles?|reconciled|accounts? for|accounted for) (?:it|this|them) (?:in full|in its entirety|completely|entirely|fully)", "clause"],
    ["explained", rf"none of the (?:\ue000+ )?{_V} is (?:explained|reconciled|accounted for)", "subject"],
    ["explained", rf"no (?:(?:eligible|open|reconciling|pending|cited) )*items? (?:explains?|accounts? for|reconciles?|offsets?) (?:any (?:part|portion) of )?(?:the )?(?:\ue000+ )?{_V}", "subject"],
    ["explained", rf"(?:entire|whole|full|total) (?:\ue000+ )?{_V} (?:is|was|remains|remained|stays|stayed) (?:unexplained|unreconciled)", "subject"],
    ["explained", r"all of (?:which|it) (?:is|was|remains|remained) (?:unexplained|unreconciled)", "clause"],
    ["explained", r"(?:fully|entirely|completely|wholly) (?:unexplained|unreconciled)", "clause"],
    # The whole variance stated as explained in other words (review finding F2-3).
    ["unexplained", rf"(?:explains?|explained|reconciles?|reconciled|accounts? for|accounted for|covers?|covered|resolves?|resolved|clears?|cleared|closes?|closed) (?:all|the whole|the entirety|the full amount|the entire amount) of (?:the )?(?:\ue000+ )?{_V}", "subject"],
    ["unexplained", rf"the (?:whole|entirety) of the (?:\ue000+ )?{_V} (?:is|was) (?:explained|reconciled|accounted for|covered|resolved)", "subject"],
    ["unexplained", r"(?:unexplained |unreconciled |residual |remaining )?(?:remainder|residual|balance|portion) (?:is|was|stands at|of) (?:zero|nil|none|nothing)", "subject"],
    ["unexplained", rf"no (?:amount|portion|part|balance|difference)(?: of (?:the )?{_V})? (?:remains|is left|is|was|stays) (?:unexplained|unreconciled|unaccounted for|outstanding)", "subject"],
    ["unexplained", r"leav(?:es|ing|e) (?:no|nothing|zero) (?:residual|remainder|balance|difference|variance|shortfall|surplus|unexplained (?:amount|balance|difference|portion))", "subject"],
    # Resolution claims: true only when nothing is unexplained.
    ["unexplained", rf"(?:the )?(?:{_V}|reconciliation) (?:is|was|has been) (?:now )?(?:complete|completed|closed|cleared|resolved|fully reconciled|tied out|clean)", "resolution"],
    ["unexplained", r"no further (?:action|follow[- ]up|investigation|work) (?:is |was )?(?:required|needed|necessary)", "resolution"],
    # Attribution without a share: full or partial is not stated, so it passes only
    # when nothing is unexplained and routes to review otherwise.
    ["unexplained", rf"(?:explains?|explained|accounts? for|accounted for|resolves?|resolved|clears?|cleared|covers?|covered) (?:the )?{_V}", "attribution"],
    ["unexplained", rf"{_V} (?:is|was|are|were) (?:entirely |wholly |fully )?(?:attributable|attributed|due) to", "attribution"],
    ["unexplained", rf"{_V} (?:is|was|are|were) (?:explained|reconciled|resolved|cleared|covered|accounted for) by", "attribution"],
]
# Qualifiers that make an attribution partial ("partly explains the shortfall"):
# such a clause states no split and is left to the amount checks.
PARTIAL_QUALIFIERS: list[str] = [
    r"partly", r"partially", r"in part", r"part of", r"portion of", r"some of", r"most of",
    r"mostly", r"largely", r"majority of", r"share of", r"\ue000+ of",
]
SPLIT_CLAIM_SUBJECTS: list[str] = [
    r"variances?", r"shortfalls?", r"surplus(?:es)?", r"deficits?", r"differences?", r"gap", r"excess",
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
# A bare balance word right before an amount labels it ("ledger $10,000,000,
# reserves $9,880,000"): [role, pattern], matched against the text just before the
# amount, with only a space or colon between.
LABEL_CUES: list[list[str]] = [
    ["ledger", r"ledger|books"],
    ["onchain", r"on[- ]?chain|wallets?"],
]
# The same with nothing after the comparator: "reserves were $120,000 short."
COMPARISON_AFTER_AMOUNT_BARE: list[str] = [r"short"]
# Verbs that join a balance to its amount across a parenthetical ("reserves, unlike
# ledger liabilities, were $X"), where the nearest balance word may not be the subject.
# Words that set a balance aside the way a comma pair does ("reserves unlike ledger
# liabilities were $X").
CONTRAST_WORDS: list[str] = [
    "unlike", "not", "rather", "instead", "versus", "vs", "opposed", "excluding", "except",
]
PARENTHETICAL_VERBS: list[str] = [
    "were", "was", "are", "is", "stood", "stand", "stands", "totaled", "totalled", "totals",
    "came", "amounted", "remained", "remain",
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
# In-scope tickers that are not ordinary words, also matched in any case ("btc").
# "sol" and "ada" are words or names, so SOL and ADA match in capitals only.
CASE_FREE_TICKERS: list[str] = ["BTC", "ETH", "USDC", "USDT", "XRP"]
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
# Fiscal-year and half-year forms route to review: the case period is one month.
FISCAL_PATTERNS: list[str] = [r"(?:FY|CY) ?'?(?:20)?[0-9]{2}", r"H[12] ?(?:20[0-9]{2})?", r"(?:fiscal|financial) year(?: 20[0-9]{2})?"]
# Month names read as a period statement when they stand alone (no day or year):
# full names except "May", which is also a verb, and capitalized abbreviations.
BARE_MONTH_NAMES: list[str] = [
    "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december",
]
BARE_MONTH_ABBREVIATIONS: list[str] = ["Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"]  # case-sensitive

# --- Currency amounts (section 6: every currency amount) ------------------------------
CURRENCY_PREFIXES: list[str] = [r"USD ?\$", r"US\$", r"\$", r"USD ?-?equivalent", r"USD"]
# Letters joined to a dollar sign that are not a listed currency ("XYZ$500") route to review.
UNKNOWN_DOLLAR_MARKER = r"(?<![A-Za-z])(?!(?:USD|US)\$)[A-Za-z]{1,4}\$ ?[0-9]"
# A bare number with a magnitude suffix ("1.2 million", "300k") is a currency amount
# in a USD commentary unless a unit word or ticker follows. Single-letter suffixes
# must be attached ("300k"); word suffixes follow a space ("1.2 million").
SUFFIXED_ATTACHED: list[str] = ["k", "K", "m", "M", "mm", "MM", "mn", "bn", "BN", "b", "B"]
SUFFIXED_WORDS: list[str] = ["thousand", "million", "billion"]
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
# Number words: an amount written out ("one hundred thousand dollars") is not
# checked; it routes to review.
NUMBER_WORDS: list[str] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
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
    "cue_search_tokens": 40,     # cues are looked for this many tokens either side of an amount
    "max_narrative_chars": 20000,  # a longer narrative is not extracted; it routes to review
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
        "NEGATION_NOUN_ADJECTIVES": NEGATION_NOUN_ADJECTIVES,
        "NEGATION_ITEM_NOUNS": NEGATION_ITEM_NOUNS,
        "BAND_LABEL": BAND_LABEL,
        "BAND_NUMBER_LABEL": BAND_NUMBER_LABEL,
        "BAND_POSITION_WORDS": BAND_POSITION_WORDS,
        "ESCALATION_CLAIMS": ESCALATION_CLAIMS,
        "SEVERITY_WORDS": SEVERITY_WORDS,
        "BAND_VOCABULARY": BAND_VOCABULARY,
        "ZERO_VARIANCE_CLAIMS": ZERO_VARIANCE_CLAIMS,
        "THRESHOLD_NAMES": THRESHOLD_NAMES,
        "BAND_COMPARATORS": BAND_COMPARATORS,
        "BAND_RANGE_WORD": BAND_RANGE_WORD,
        "BAND_WORDS": BAND_WORDS,
        "ROLE_CUES": ROLE_CUES,
        "LINK_WORDS": LINK_WORDS,
        "LINK_PUNCT": LINK_PUNCT,
        "SPLIT_CLAIMS": SPLIT_CLAIMS,
        "SPLIT_CLAIM_SUBJECTS": SPLIT_CLAIM_SUBJECTS,
        "SPLIT_CLAIMS_SELF_NEGATED": SPLIT_CLAIMS_SELF_NEGATED,
        "PARTIAL_QUALIFIERS": PARTIAL_QUALIFIERS,
        "COMPARISON_AFTER_AMOUNT": COMPARISON_AFTER_AMOUNT,
        "COMPARISON_AFTER_AMOUNT_BARE": COMPARISON_AFTER_AMOUNT_BARE,
        "LABEL_CUES": LABEL_CUES,
        "CONTRAST_WORDS": CONTRAST_WORDS,
        "PARENTHETICAL_VERBS": PARENTHETICAL_VERBS,
        "COMPARISON_WORDS": COMPARISON_WORDS,
        "PARALLEL_LIST_WORDS": PARALLEL_LIST_WORDS,
        "IN_SCOPE_TICKERS": IN_SCOPE_TICKERS,
        "CASE_FREE_TICKERS": CASE_FREE_TICKERS,
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
        "FISCAL_PATTERNS": FISCAL_PATTERNS,
        "BARE_MONTH_NAMES": BARE_MONTH_NAMES,
        "BARE_MONTH_ABBREVIATIONS": BARE_MONTH_ABBREVIATIONS,
        "CURRENCY_PREFIXES": CURRENCY_PREFIXES,
        "CURRENCY_SUFFIXES": CURRENCY_SUFFIXES,
        "UNKNOWN_DOLLAR_MARKER": UNKNOWN_DOLLAR_MARKER,
        "SUFFIXED_ATTACHED": SUFFIXED_ATTACHED,
        "SUFFIXED_WORDS": SUFFIXED_WORDS,
        "NON_USD_CURRENCY": NON_USD_CURRENCY,
        "MAGNITUDE_SUFFIXES": MAGNITUDE_SUFFIXES,
        "UNIT_WORDS": UNIT_WORDS,
        "NUMBER_WORDS": NUMBER_WORDS,
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
