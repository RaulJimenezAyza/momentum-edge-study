"""Trading sessions, and the holes between them.

Crypto never closes, and the entire strategy quietly assumes it. Bar `i-1` is
one minute before bar `i`; a stop can be hit at any moment; a position can be
held for thirty minutes starting at any time. None of that survives contact
with an instrument that trades 09:30-16:00 and then stops for seventeen and a
half hours.

Three specific things break, and each needs an explicit answer rather than
being left to the arithmetic:

1. **Returns across the gap are not returns over one bar.** The 3-bar momentum
   at 09:35 would otherwise be measured against yesterday's close, mixing a
   fifteen-minute move with an overnight one. Signals whose measurement window
   straddles a gap are refused.

2. **You cannot exit while the market is shut.** A stop 1.2% away is not a 1.2%
   stop if the next print is 4% lower at tomorrow's open. Positions are closed
   at the last bar of the session instead of being carried, which is a real
   strategy choice (no overnight risk) and the only one that can be modelled
   honestly from OHLC alone.

3. **The hold clock runs in bars, not minutes.** Already handled by
   `max_hold_bars`, but only because the timeframe generality work came first.

A session boundary is inferred from the data rather than from a calendar:
consecutive bars more than `gap_multiple` intervals apart. That gets weekends,
holidays, half-days and outages right without a holiday table, and it is
self-correcting when the exchange changes its hours.
"""

from __future__ import annotations

from .exchange import Candle

# A gap of more than this many bar-lengths means a new session. 1.5 is enough
# to catch a single missing bar without calling it a boundary; the real
# overnight gap on 5m bars is 210 bar-lengths.
GAP_MULTIPLE = 1.5

# At or above this bar length, one bar IS a session and the whole idea stops
# applying. Daily bars have a three-day gap every weekend, which the intraday
# rule would call a session boundary every single Monday -- refusing most
# signals and clipping every exit window to nothing. Worse, it would be wrong
# on the substance: with a daily bar you CAN hold overnight, because the next
# bar is the next tradable day.
DAILY_MS = 86_400_000


def is_intraday(bar_ms: int) -> bool:
    return bar_ms < DAILY_MS


def is_gap(previous: Candle, current: Candle, bar_ms: int) -> bool:
    """Is there a session boundary between these two bars?

    Never, on daily-or-longer bars: a weekend is not a gap in a daily series,
    it is simply the next bar.
    """
    if not is_intraday(bar_ms):
        return False
    return (current.open_time - previous.open_time) > bar_ms * GAP_MULTIPLE


def session_index(candles: list[Candle], bar_ms: int) -> list[int]:
    """Session number for each candle, starting at 0."""
    out: list[int] = []
    session = 0
    for i, c in enumerate(candles):
        if i and is_gap(candles[i - 1], c, bar_ms):
            session += 1
        out.append(session)
    return out


def contiguous_back(candles: list[Candle], count: int, bar_ms: int) -> bool:
    """Are the last `count` candles free of any session gap between them?

    `count` is a number of candles, so `count - 1` intervals are checked. A
    3-bar return needs four candles to be contiguous.
    """
    window = candles[-count:] if count <= len(candles) else candles
    if len(window) < count:
        return False
    return not any(
        is_gap(window[i - 1], window[i], bar_ms) for i in range(1, len(window))
    )


def forward_window(
    candles: list[Candle], index: int, horizon: int, bar_ms: int
) -> list[Candle]:
    """Candles after `index` that the position could actually be exited in.

    Stops at the session boundary. Everything past it happens while the market
    is shut, and pretending a stop fills there is how a backtest invents money
    it could never have made.
    """
    out: list[Candle] = []
    previous = candles[index]
    for c in candles[index + 1: index + 1 + horizon]:
        if is_gap(previous, c, bar_ms):
            break
        out.append(c)
        previous = c
    return out


def session_stats(candles: list[Candle], bar_ms: int) -> dict:
    """Describe the session structure, for the report header."""
    if len(candles) < 2:
        return {"sessions": 0, "bars_per_session": 0.0, "max_gap_bars": 0.0}
    idx = session_index(candles, bar_ms)
    sessions = idx[-1] + 1
    gaps = [
        (candles[i].open_time - candles[i - 1].open_time) / bar_ms
        for i in range(1, len(candles))
        if is_gap(candles[i - 1], candles[i], bar_ms)
    ]
    return {
        "sessions": sessions,
        "bars_per_session": len(candles) / sessions,
        "max_gap_bars": max(gaps) if gaps else 0.0,
    }
