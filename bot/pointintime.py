"""Choosing the universe with information that existed at the time.

Every measurement in this project ranks symbols by their volume **today** and
then applies that list to history. A coin that pumped last week has been in the
universe for all 150 days of the backtest, including the months when nobody
traded it. That is look-ahead, and it has been flagged in comments since the
first backtest without ever being measured or fixed.

This fixes the half of it that can be fixed.

  FIXABLE      the ranking. Which symbols were the most traded as of a given
               date is knowable from the candles themselves: rank by quote
               volume over the period *preceding* the window, not by today's
               ticker.

  NOT FIXABLE  survivorship. Binance's `exchangeInfo` lists only symbols that
               are still trading, so a coin that was top-50 in March and has
               since been delisted cannot appear in any candidate pool this
               code can build. No amount of care here recovers it, and the
               remaining bias points the same way: the surviving names did
               better than the full population did.

So the honest claim after this is "ranking look-ahead removed, delisting
survivorship remains", not "unbiased".
"""

from __future__ import annotations

import logging

from .exchange import Candle

log = logging.getLogger(__name__)

# How much history before the window to rank on. Long enough to be a stable
# ranking, short enough to reflect what was actually liquid then.
RANK_DAYS = 30


def quote_volume_between(candles: list[Candle], start_ms: int, end_ms: int) -> float:
    return sum(c.quote_volume for c in candles
               if start_ms <= c.open_time < end_ms)


def point_in_time_universe(
    history: dict[str, list[Candle]],
    window_start_ms: int,
    size: int,
    min_quote_volume: float = 0.0,
    rank_days: int = RANK_DAYS,
) -> list[str]:
    """Top `size` symbols by volume in the period BEFORE `window_start_ms`.

    Symbols with no data in the ranking period are excluded: if it was not
    trading then, it could not have been picked then.
    """
    rank_start = window_start_ms - rank_days * 86_400_000
    scored: list[tuple[str, float]] = []
    for symbol, candles in history.items():
        volume = quote_volume_between(candles, rank_start, window_start_ms)
        if volume <= 0:
            continue
        # Scale to a 24h figure so the floor means the same thing as the
        # exchange's own 24h volume.
        daily = volume / rank_days
        if daily < min_quote_volume:
            continue
        scored.append((symbol, daily))

    scored.sort(key=lambda row: row[1], reverse=True)
    chosen = [s for s, _ in scored[:size]]
    log.info(
        "point-in-time universe: %d of %d symbols were tradable and liquid "
        "enough in the %d days before the window",
        len(chosen), len(history), rank_days,
    )
    return chosen


def listing_report(history: dict[str, list[Candle]], window_days: float) -> dict:
    """How much of the universe simply did not exist for the whole window."""
    if not history:
        return {}
    newest_end = max(c[-1].open_time for c in history.values())
    span_ms = window_days * 86_400_000
    want_start = newest_end - span_ms

    short: dict[str, float] = {}
    total_bars = 0
    short_bars = 0
    for symbol, candles in history.items():
        total_bars += len(candles)
        if candles[0].open_time > want_start + 5 * 86_400_000:
            days = (newest_end - candles[0].open_time) / 86_400_000
            short[symbol] = days
            short_bars += len(candles)
    return {
        "symbols": len(history),
        "incomplete": len(short),
        "incomplete_share": len(short) / len(history),
        "candle_share": short_bars / total_bars if total_bars else 0.0,
        "worst": sorted(short.items(), key=lambda kv: kv[1])[:10],
    }
