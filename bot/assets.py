"""Is this only a crypto effect?

Binance lists more than coins. Tokenized equities and ETFs trade there against
USDT -- QQQB, SPYB, NVDAB, GOOGLB, TSMB, MSTRB, SOXLB -- and so does gold, as
XAUT and PAXG. They arrive through the same API, the same candles and the same
code, so asking whether the reversal edge exists outside crypto costs nothing
but a classifier.

It is also the first hypothesis in this project with a **mechanism predicted
before the measurement**, which every previous one lacked:

  A tokenized stock trades 24/7, but the market that prices the real share is
  open roughly 14:30-21:00 UTC on weekdays. While NASDAQ is shut, nobody can
  arbitrage the token against the underlying. A panic in the token has nothing
  anchoring it, so it should overshoot further and revert harder.

That predicts a *specific, directional* difference -- equities out of hours
beating equities in hours -- rather than "some bucket will look better". If the
prediction fails, the hypothesis is dead; if it holds, it is worth more than
the same number found by searching, because it was called in advance.

The classification is a curated list. `exchangeInfo` exposes nothing that
separates a tokenized share from a coin (`permissions` is empty for both, and
the `permissionSets` differences are undocumented internals not worth
depending on), so the alternative is a suffix rule that would call BNB, ARB and
SHIB equities.
"""

from __future__ import annotations

from datetime import UTC

# Gold, held as a token.
METALS = frozenset({"XAUT", "PAXG", "XAUTUSDT", "PAXGUSDT"})

# Tokenized shares and ETFs. Base assets, without the USDT quote. Binance
# suffixes them with B, but that rule alone catches BNB/ARB/TRB/SHIB, so the
# list is explicit and will need updating as listings change.
EQUITIES = frozenset({
    # index and sector ETFs
    "QQQB", "SPYB", "SOXLB", "KORUB", "MUBB", "SNXXB", "SKHYB", "MUUB",
    # single names
    "NVDAB", "GOOGLB", "TSMB", "MSTRB", "SMCIB", "CRCLB", "SNDKB", "SPCXB",
    "NBISB", "AAOIB", "CRWVB", "AXTIB", "LITEB", "HEIB", "MUB",
})

# Ends in B but is a coin. Kept explicit so the suffix trap is documented
# rather than merely avoided.
NOT_EQUITIES = frozenset({"BNB", "ARB", "TRB", "SHIB", "BOB", "CTXC"})

CLASSES = ("crypto", "equity", "metal")

# NASDAQ / NYSE regular session in UTC, ignoring daylight-saving drift and
# holidays. Both would blur the boundary rather than move it, so the test is
# conservative: some "closed" bars are really open ones.
US_OPEN_HOUR, US_CLOSE_HOUR = 14, 21


def base_asset(symbol: str, quote: str = "USDT") -> str:
    return symbol[: -len(quote)] if symbol.endswith(quote) else symbol


def classify(symbol: str, quote: str = "USDT") -> str:
    base = base_asset(symbol, quote)
    if base in METALS or symbol in METALS:
        return "metal"
    if base in NOT_EQUITIES:
        return "crypto"
    if base in EQUITIES:
        return "equity"
    return "crypto"


def us_market_open(time_ms: int) -> bool:
    """Rough check: is the underlying US market open at this timestamp?"""
    from datetime import datetime

    dt = datetime.fromtimestamp(time_ms / 1000, UTC)
    if dt.weekday() >= 5:          # Saturday, Sunday
        return False
    return US_OPEN_HOUR <= dt.hour < US_CLOSE_HOUR
