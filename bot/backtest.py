"""Replay historical 1m klines through the exact live strategy code.

Known limitations, stated plainly because they inflate results if ignored:
  * Fills come from candle OHLC, not the order book. Real slippage in a fast
    breakout is worse than the flat `slippage_bps` assumed here.
  * Within one candle we cannot know whether the high or the low came first;
    the strategy assumes the stop hit first, which is the conservative choice.
  * The universe is chosen from TODAY's volume ranking, so the backtest only
    ever sees coins that survived and stayed liquid. That is survivorship bias
    and it flatters every result below.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .broker import PaperBroker
from .cache import KlineCache
from .config import Config
from .exchange import BinanceREST, Candle
from .logbook import Logbook, print_summary, summarize
from .risk import RiskManager
from .strategy import MARKET_SYMBOL, Strategy
from .spread import QUOTE_SNAPSHOT, refresh_quote_snapshot
from .universe import select_universe

log = logging.getLogger(__name__)


async def fetch_history(
    rest: BinanceREST,
    symbols: list[str],
    interval: str,
    days: float,
    cache: KlineCache | None = None,
    end_ms: int | None = None,
) -> dict[str, list[Candle]]:
    end_ms = int(time.time() * 1000) if end_ms is None else end_ms
    start_ms = end_ms - int(days * 86_400_000)
    out: dict[str, list[Candle]] = {}
    sem = asyncio.Semaphore(5)

    async def one(symbol: str) -> None:
        async with sem:
            if cache is not None:
                candles = await cache.klines_range(
                    rest, symbol, interval, start_ms, end_ms
                )
            else:
                candles = await rest.klines_range(symbol, interval, start_ms, end_ms)
            if candles:
                out[symbol] = candles
            log.info("  %-12s %5d bars", symbol, len(candles))

    # return_exceptions: one symbol failing must not orphan the rest. Without
    # it, gather re-raises at once, the surviving tasks are left running, the
    # session closes under them, and they report "Connector is closed" --
    # which is what a single bad symbol looked like from the outside.
    results = await asyncio.gather(*(one(s) for s in symbols),
                                   return_exceptions=True)
    failed = [s for s, r in zip(symbols, results) if isinstance(r, Exception)]
    if failed:
        log.warning("%d/%d symbols failed to download: %s",
                    len(failed), len(symbols), ", ".join(failed[:6]))
    return out


async def load_history(
    cfg: Config,
    days: float,
    symbols: list[str] | None,
    need_market: bool,
    offset_days: float = 0.0,
) -> tuple[dict[str, list[Candle]], dict]:
    """Load a replay window, serving what it can from the on-disk cache.

    `offset_days` ends the window that many days before now, which is how a
    holdout is carved out: search on [now-days-offset, now-offset), then judge
    the single surviving candidate on [now-offset, now) exactly once.
    """
    cache = KlineCache(cfg.path("data/cache")) if cfg.cache_enabled else None
    end_ms = int(time.time() * 1000) - int(offset_days * 86_400_000)
    async with BinanceREST(cfg.exchange.base_url) as rest:
        await rest.ping()
        universe = symbols or await select_universe(rest, cfg.exchange)
        if need_market and MARKET_SYMBOL not in universe:
            universe = [*universe, MARKET_SYMBOL]
        log.info("loading %.1f days of %s data for %d symbols%s...",
                 days, cfg.strategy.timeframe, len(universe),
                 f", ending {offset_days:.0f}d ago" if offset_days else "")
        history = await fetch_history(
            rest, universe, cfg.strategy.timeframe, days, cache, end_ms
        )
        filters = await rest.symbol_filters(list(history))
        # Ground truth for the symbols the spread estimator cannot resolve.
        # Cheap (one call), and it stops a third of the universe being priced
        # by a constant and gated out of every backtest.
        if cfg.fees.slippage_mode == "estimated":
            try:
                await refresh_quote_snapshot(rest, cfg.path(QUOTE_SNAPSHOT))
            except Exception as exc:  # noqa: BLE001 - offline must still work
                log.warning("could not refresh quote snapshot: %s", exc)
    if cache is not None:
        log.info("%s", cache.summary())
    if not history:
        raise RuntimeError("no data available")
    return history, filters


async def run_backtest(
    cfg: Config, days: float = 7.0, symbols: list[str] | None = None
) -> dict:
    from .engine import load_model

    model = load_model(cfg)
    history, filters = await load_history(cfg, days, symbols, model is not None)

    broker = PaperBroker(cfg, filters)
    risk = RiskManager(cfg)
    strategy = Strategy(cfg, broker, risk, model=model)
    logbook = Logbook(
        cfg.path("data/backtest_trades.csv"),
        cfg.path("data/backtest_equity.csv"),
        cfg.path("data/backtest_signals.csv"),
    )

    warmup = cfg.strategy.lookback_bars
    for symbol, candles in history.items():
        strategy.seed(symbol, candles[:warmup])

    # Merge every symbol's remaining candles into one time-ordered tape so the
    # capital constraint (max_positions, shared equity) binds the way it would live.
    tape: list[tuple[int, str, Candle]] = [
        (c.open_time, symbol, c)
        for symbol, candles in history.items()
        for c in candles[warmup:]
    ]
    tape.sort(key=lambda row: (row[0], row[1]))
    log.info("replaying %d candles across %d symbols", len(tape), len(history))

    last_equity_stamp = 0
    for time_ms, symbol, candle in tape:
        for trade in strategy.on_candle(symbol, candle):
            logbook.record_trade(trade)
        if time_ms - last_equity_stamp >= 3_600_000:
            last_equity_stamp = time_ms
            logbook.record_equity(
                time_ms,
                broker.mark_to_market(strategy.positions, strategy.last_price),
                len(strategy.positions),
            )

    if tape:
        for trade in strategy.close_all(tape[-1][0] + 60_000, "end_of_data"):
            logbook.record_trade(trade)
    for signal in strategy.signals_seen:
        logbook.record_signal(signal)

    stats = summarize(broker.trades, cfg.risk.starting_equity, broker.equity)
    stats["candles_replayed"] = len(tape)
    stats["symbols"] = len(history)
    stats["days"] = days
    print_summary(stats, strategy.rejections)

    if stats["trades"] < 30:
        log.warning(
            "only %d trades -- far too few to conclude anything. Backtest a longer "
            "window before trusting any of these numbers.",
            stats["trades"],
        )
    return stats
