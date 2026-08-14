"""On-disk kline cache.

Every backtest, training run and parameter sweep re-downloaded the same candles
-- minutes of waiting and thousands of API calls to fetch bytes that never
change. Closed 1m candles are immutable, so they are cached forever; only the
missing runs are fetched.

Storage is one gzipped pickle per (symbol, interval), holding a time-ordered
list of closed candles. Missing runs are detected by comparing against the
expected timestamp grid, so an internal gap is refetched rather than silently
served as complete data.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path

from .exchange import TIMEFRAME_MS, BinanceREST, Candle

log = logging.getLogger(__name__)

# Binance itself has holes (halted symbols, listing gaps). Refetching a run
# that the exchange simply does not have would loop forever, so a run is only
# chased once per process.
_UNAVAILABLE: set[tuple[str, str, int, int]] = set()


class KlineCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.fetched = 0

    def _path(self, symbol: str, interval: str) -> Path:
        return self.root / f"{symbol}_{interval}.pkl.gz"

    # -- persistence -------------------------------------------------------
    def _load(self, symbol: str, interval: str) -> dict[int, Candle]:
        path = self._path(symbol, interval)
        if not path.exists():
            return {}
        try:
            with gzip.open(path, "rb") as fh:
                return {c.open_time: c for c in pickle.load(fh)}
        except Exception as exc:  # noqa: BLE001 - a corrupt cache is not fatal
            log.warning("cache unreadable for %s (%s), refetching", symbol, exc)
            return {}

    def _save(self, symbol: str, interval: str, candles: dict[int, Candle]) -> None:
        ordered = [candles[t] for t in sorted(candles)]
        tmp = self._path(symbol, interval).with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(ordered, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path(symbol, interval))

    # -- the one method callers need ---------------------------------------
    async def klines_range(
        self,
        rest: BinanceREST,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Candle]:
        step = TIMEFRAME_MS.get(interval, 60_000)
        # Align to the candle grid so expected timestamps line up exactly.
        start = (start_ms // step) * step
        end = (end_ms // step) * step

        have = self._load(symbol, interval)
        missing = self._missing_runs(have, start, end, step)

        if missing:
            for run_start, run_end in missing:
                if (symbol, interval, run_start, run_end) in _UNAVAILABLE:
                    continue
                fetched = await rest.klines_range(symbol, interval, run_start, run_end)
                self.fetched += len(fetched)
                if not fetched:
                    _UNAVAILABLE.add((symbol, interval, run_start, run_end))
                    continue
                for c in fetched:
                    if c.closed:
                        have[c.open_time] = c
                # Whatever the exchange still did not return does not exist.
                still_missing = self._missing_runs(have, run_start, run_end, step)
                for gap in still_missing:
                    _UNAVAILABLE.add((symbol, interval, gap[0], gap[1]))
            self._save(symbol, interval, have)
        else:
            self.hits += 1

        return [have[t] for t in sorted(have) if start <= t <= end]

    @staticmethod
    def _missing_runs(
        have: dict[int, Candle], start: int, end: int, step: int
    ) -> list[tuple[int, int]]:
        """Contiguous stretches of the expected grid that are not cached."""
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        t = start
        while t <= end:
            if t in have:
                if run_start is not None:
                    runs.append((run_start, t - step))
                    run_start = None
            elif run_start is None:
                run_start = t
            t += step
        if run_start is not None:
            runs.append((run_start, end))
        return runs

    def summary(self) -> str:
        return f"cache: {self.hits} symbols served from disk, {self.fetched} candles downloaded"
