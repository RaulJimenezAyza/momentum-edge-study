"""US equities and ETFs, via Yahoo Finance, as the same `Candle` the rest of
the code already speaks.

Why bother: the single finding of this project is that a real +0.039% edge
cannot pay a 15-20 bps toll. US equities are the obvious place to look next
precisely because **that toll is close to zero** at a commission-free broker --
the same edge that loses money on Binance would not automatically lose money
here. That is a hypothesis worth one adapter.

What Yahoo actually gives, measured rather than assumed:

    5m  over 60 days   ~4,680 bars per symbol  (60 sessions x 78)
    15m over 60 days   ~1,560 bars
    1h  over 730 days  ~5,082 bars             (two years, enough for
                                                non-overlapping replication)

Free data, so it is worth naming the ways it is worse than the exchange feed:
it is consolidated rather than a single venue, it carries no bid/ask at all
(so the spread has to be estimated from OHLC, with no `bookTicker` ground truth
to check it against), and it silently omits some low-volume bars. None of that
is fatal for measuring an effect; all of it would matter for running one.
"""

from __future__ import annotations

import gzip
import logging
import pickle
import time
from pathlib import Path

from .exchange import Candle

log = logging.getLogger(__name__)

# What each interval can actually reach, from Yahoo's own limits.
#
# The daily figure is not a Yahoo limit, it is how far back the series itself
# goes: ^GSPC starts on 1927-12-30, which is why the ladder test can put nine
# non-overlapping decades behind a claim while the crypto work has five months.
# It was 10,000 days here, which silently truncated a century to 1999 and
# turned "does this survive the 1930s" into "does this survive the 2010s".
MAX_DAYS = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59,
            "60m": 729, "1h": 729, "4h": 729, "1d": 40_000, "1wk": 40_000}

# Interval names differ from Binance's in one place.
_ALIAS = {"1h": "60m", "4h": "1h"}   # Yahoo has no 4h; nearest is hourly.


def _to_candles(frame) -> list[Candle]:
    """DataFrame -> Candle list, dropping rows Yahoo could not fill."""
    out: list[Candle] = []
    for stamp, row in frame.iterrows():
        try:
            o, h, lo, c = (float(row["Open"]), float(row["High"]),
                          float(row["Low"]), float(row["Close"]))
            v = float(row["Volume"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(x > 0 for x in (o, h, lo, c)) or h < lo:
            continue
        out.append(Candle(
            open_time=int(stamp.timestamp() * 1000),
            open=o, high=h, low=lo, close=c,
            volume=v, quote_volume=v * c, closed=True,
        ))
    out.sort(key=lambda x: x.open_time)
    return out


class YahooFeed:
    """Cached Yahoo downloads. One gzipped pickle per symbol+interval.

    Cached forever like the Binance klines, for the same reason: a closed bar
    never changes. The cache key includes the interval because the same symbol
    at 5m and at 1h are different series.
    """

    def __init__(self, root: Path, auto_adjust: bool = False):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.downloads = 0
        # Adjusted closes are a DIFFERENT series, not a view of the same one:
        # they carry dividends and splits back through history, so they change
        # every bar before the last distribution. They therefore get their own
        # cache key. Comparing an adjusted series to an unadjusted one is how a
        # total-return strategy gets validated against a price-only benchmark
        # and appears to beat it by exactly the dividend yield.
        self.auto_adjust = auto_adjust

    def _path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "idx_")
        suffix = "_adj" if self.auto_adjust else ""
        return self.root / f"{safe}_{interval}{suffix}.pkl.gz"

    def _load(self, symbol: str, interval: str) -> list[Candle]:
        path = self._path(symbol, interval)
        if not path.exists():
            return []
        try:
            with gzip.open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - a bad cache is not fatal
            log.warning("yahoo cache unreadable for %s (%s)", symbol, exc)
            return []

    def _save(self, symbol: str, interval: str, candles: list[Candle]) -> None:
        tmp = self._path(symbol, interval).with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(candles, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path(symbol, interval))

    def fetch(self, symbol: str, interval: str, days: float) -> list[Candle]:
        alias = _ALIAS.get(interval, interval)
        limit = MAX_DAYS.get(alias)
        if limit is None:
            raise ValueError(f"unsupported Yahoo interval {interval!r}")
        want = min(days, limit)
        if want < days:
            # Say so rather than quietly returning a shorter history than asked
            # for -- a silently truncated window is how a 60-day claim becomes
            # a 7-day one.
            log.warning("%s: Yahoo caps %s at %d days, asked for %.0f",
                        symbol, alias, limit, days)

        cached = self._load(symbol, alias)
        cutoff = int((time.time() - want * 86_400) * 1000)
        if cached:
            covered = [c for c in cached if c.open_time >= cutoff]
            fresh_enough = cached[-1].open_time > int(
                (time.time() - 2 * 86_400) * 1000
            )
            if covered and fresh_enough:
                self.hits += 1
                return covered

        import yfinance as yf

        period = "max" if alias in ("1d", "1wk") and want > 3000 else f"{int(want) + 1}d"
        frame = yf.download(symbol, period=period, interval=alias,
                            progress=False, auto_adjust=self.auto_adjust,
                            threads=False)
        if frame is None or frame.empty:
            log.warning("%s: Yahoo returned nothing", symbol)
            return [c for c in cached if c.open_time >= cutoff]
        if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
            frame.columns = frame.columns.get_level_values(0)

        downloaded = _to_candles(frame)
        self.downloads += len(downloaded)
        merged = {c.open_time: c for c in cached}
        merged.update({c.open_time: c for c in downloaded})
        ordered = [merged[t] for t in sorted(merged)]
        self._save(symbol, alias, ordered)
        return [c for c in ordered if c.open_time >= cutoff]

    def summary(self) -> str:
        return (f"yahoo: {self.hits} symbols from cache, "
                f"{self.downloads} bars downloaded")


def load_equity_history(
    symbols: list[str], interval: str, days: float, cache_dir: Path,
    auto_adjust: bool = False,
) -> dict[str, list[Candle]]:
    feed = YahooFeed(cache_dir, auto_adjust=auto_adjust)
    out: dict[str, list[Candle]] = {}
    for symbol in symbols:
        candles = feed.fetch(symbol, interval, days)
        if candles:
            out[symbol] = candles
        log.info("  %-8s %5d bars", symbol, len(candles))
    log.info("%s", feed.summary())
    return out
