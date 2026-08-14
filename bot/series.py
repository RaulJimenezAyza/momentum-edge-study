"""Per-symbol rolling candle history and the statistics the strategy needs."""

from __future__ import annotations

import math
from collections import deque

from .exchange import Candle


class Series:
    """Fixed-length window of CLOSED candles, plus the currently forming one.

    Indicators read only closed candles, so a signal never repaints. The
    forming candle is exposed separately for stop/target checks.
    """

    def __init__(self, symbol: str, maxlen: int):
        self.symbol = symbol
        self.closed: deque[Candle] = deque(maxlen=maxlen)
        self.forming: Candle | None = None

    # -- ingestion ---------------------------------------------------------
    def update(self, candle: Candle) -> bool:
        """Feed a candle. Returns True when this call closed a bar."""
        if not candle.closed:
            self.forming = candle
            return False
        if self.closed and self.closed[-1].open_time == candle.open_time:
            self.closed[-1] = candle  # late correction for the same bar
        else:
            self.closed.append(candle)
        self.forming = None
        return True

    # -- accessors ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.closed)

    @property
    def last_price(self) -> float:
        if self.forming is not None:
            return self.forming.close
        return self.closed[-1].close if self.closed else 0.0

    @property
    def last_close_time(self) -> int:
        return self.closed[-1].open_time if self.closed else 0

    # -- statistics --------------------------------------------------------
    def ret_pct(self, bars: int) -> float:
        """Percent change over the last `bars` closed candles."""
        if len(self.closed) <= bars:
            return 0.0
        past = self.closed[-bars - 1].close
        if past <= 0:
            return 0.0
        return (self.closed[-1].close / past - 1.0) * 100.0

    def return_sigma_pct(self, bars: int) -> float:
        """Std-dev of `bars`-length percent returns across the window."""
        closes = [c.close for c in self.closed]
        if len(closes) <= bars + 2:
            return 0.0
        rets = [
            (closes[i] / closes[i - bars] - 1.0) * 100.0
            for i in range(bars, len(closes))
            if closes[i - bars] > 0
        ]
        if len(rets) < 3:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)

    def volume_mult(self, quote: bool = False) -> float:
        """Last bar volume against the mean of the preceding window.

        `quote=True` measures money traded; the default measures units of the
        base asset, which is what the exchange reports first and what this
        started with.

        The distinction is not cosmetic for a strategy that buys falls. Base
        volume is money divided by price, so when the price drops the SAME
        dollar flow prints as more units. The window mean spans 120 bars, over
        which the price can move a long way, so a coin that has fallen shows an
        inflated volume multiple for purely arithmetic reasons -- and it shows
        it exactly on the bars a dip-buyer is looking at.
        """
        if len(self.closed) < 10:
            return 0.0
        bars = list(self.closed)
        pick = (lambda c: c.quote_volume) if quote else (lambda c: c.volume)
        prior = [pick(c) for c in bars[:-1]]
        mean = sum(prior) / len(prior)
        if mean <= 0:
            return 0.0
        return pick(bars[-1]) / mean

    def _window(self, bars: int, offset: int) -> list[Candle]:
        return list(self.closed)[-(bars + offset) : len(self.closed) - offset]

    def highest_close(self, bars: int, offset: int = 1) -> float:
        """Max close over `bars` candles ending `offset` bars back."""
        return max((c.close for c in self._window(bars, offset)), default=0.0)

    def lowest_close(self, bars: int, offset: int = 1) -> float:
        """Min close over the same window. The reversal side's breakout level."""
        return min((c.close for c in self._window(bars, offset)), default=0.0)

    def sma(self, bars: int) -> float:
        window = list(self.closed)[-bars:]
        if not window:
            return 0.0
        return sum(c.close for c in window) / len(window)

    def extension_pct(self, bars: int) -> float:
        """How far the last close sits above its own moving average."""
        avg = self.sma(bars)
        if avg <= 0:
            return 0.0
        return (self.closed[-1].close / avg - 1.0) * 100.0

    def quote_volume_recent(self, bars: int) -> float:
        window = list(self.closed)[-bars:]
        return sum(c.quote_volume for c in window)

    def atr_pct(self, bars: int) -> float:
        """Average true range over `bars`, as a percentage of the last close.

        True range includes the gap from the previous close, so an overnight-style
        jump counts as range rather than being invisible. Expressed in percent so
        one setting works across a $100k coin and a $0.0001 one.
        """
        window = list(self.closed)[-(bars + 1):]
        if len(window) < 3:
            return 0.0
        ranges = []
        for prev, cur in zip(window, window[1:]):
            ranges.append(
                max(
                    cur.high - cur.low,
                    abs(cur.high - prev.close),
                    abs(cur.low - prev.close),
                )
            )
        last = window[-1].close
        if last <= 0 or not ranges:
            return 0.0
        return (sum(ranges) / len(ranges)) / last * 100.0
