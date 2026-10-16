"""Live loop: stream candles, run the strategy, print a heartbeat."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from .broker import PaperBroker
from .config import Config
from .dashboard import DashboardServer, DashboardState, LogTail
from .exchange import BinanceREST, stream_klines
from .logbook import Logbook, print_summary, summarize
from .risk import RiskManager
from .strategy import MARKET_SYMBOL, Strategy
from .universe import select_universe

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 60
SPREAD_REFRESH_SECONDS = 30
# The dashboard curve should move more often than the console heartbeat.
DASHBOARD_SAMPLE_SECONDS = 5
# A 1m stream should deliver several messages per minute per symbol. Past this,
# treat the symbol's history as holed and repair it from REST.
STALE_AFTER_SECONDS = 180
STALE_CHECK_SECONDS = 60


def load_model(cfg: Config):
    """Return the trained filter, or None if ML is off or unavailable."""
    if not cfg.ml.enabled:
        return None
    from .ml import Model

    path = cfg.path(cfg.ml.model_path)
    if not path.exists():
        raise RuntimeError(
            f"ml.enabled is true but {path} does not exist. "
            "Train one first: python run.py train --days 45"
        )
    model = Model.load(path)
    report = model.meta.get("report", {})
    threshold = (
        cfg.ml.min_probability
        if cfg.ml.threshold_mode == "fixed"
        else model.meta.get("threshold", cfg.ml.min_probability)
    )
    log.info(
        "ML filter loaded: AUC %.4f, threshold %.3f (%s), verdict %s",
        report.get("auc", 0.0),
        threshold,
        cfg.ml.threshold_mode,
        report.get("verdict", "unknown"),
    )
    return model


class Engine:
    def __init__(self, cfg: Config, symbols: list[str] | None = None):
        self.cfg = cfg
        self.symbols_override = symbols
        self.broker = PaperBroker(cfg)
        self.risk = RiskManager(cfg)
        self.strategy = Strategy(cfg, self.broker, self.risk, model=load_model(cfg))
        self.strategy.collect_diagnostics = cfg.dashboard.enabled
        self.logbook = Logbook(
            cfg.path(cfg.logging.trades_csv),
            cfg.path(cfg.logging.equity_csv),
            cfg.path(cfg.logging.signals_csv),
        )
        self.symbols: list[str] = []
        self._logged_signals = 0
        self._stop = asyncio.Event()
        # Stream health: when each symbol last delivered a message, and how
        # many times a hole had to be repaired.
        self.last_seen: dict[str, float] = {}
        self.repairs = 0

        self.log_tail = LogTail()
        self.dash_state = DashboardState(
            cfg, self.strategy, self.broker, self.risk, self.log_tail
        )
        self.dashboard = DashboardServer(self.dash_state) if cfg.dashboard.enabled else None

    # -- startup -----------------------------------------------------------
    async def _prepare(self, rest: BinanceREST) -> None:
        await rest.ping()
        self.symbols = self.symbols_override or await select_universe(
            rest, self.cfg.exchange
        )
        if not self.symbols:
            raise RuntimeError("universe is empty -- loosen exchange filters")

        # The market-regime features need BTC even when it is not being traded.
        if self.strategy.model is not None and MARKET_SYMBOL not in self.symbols:
            self.symbols.append(MARKET_SYMBOL)
            log.info("added %s to the stream for market-regime features", MARKET_SYMBOL)

        self.dash_state.symbols = self.symbols
        self.broker.filters = await rest.symbol_filters(self.symbols)

        log.info("seeding %d symbols with %d bars of history...",
                 len(self.symbols), self.cfg.strategy.lookback_bars)
        await self._seed(rest, self.symbols)
        log.info("history loaded. watching: %s", ", ".join(self.symbols[:10]) +
                 (f" (+{len(self.symbols) - 10} more)" if len(self.symbols) > 10 else ""))

    async def _seed(self, rest: BinanceREST, symbols: list[str]) -> None:
        """(Re)load each symbol's rolling window from REST."""
        sem = asyncio.Semaphore(8)

        async def one(symbol: str) -> None:
            async with sem:
                candles = await rest.klines(
                    symbol, self.cfg.strategy.timeframe, self.cfg.strategy.lookback_bars
                )
                # The last kline from REST is the still-forming bar; drop it.
                self.strategy.seed(symbol, candles[:-1])
                self.last_seen[symbol] = time.time()

        await asyncio.gather(*(one(s) for s in symbols))

    # -- background tasks --------------------------------------------------
    async def _refresh_spreads(self, rest: BinanceREST) -> None:
        watched = set(self.symbols)
        while not self._stop.is_set():
            try:
                for t in await rest.book_tickers():
                    if t["symbol"] not in watched:
                        continue
                    bid, ask = float(t["bidPrice"]), float(t["askPrice"])
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                        bps = (ask - bid) / mid * 10_000
                        self.strategy.spread_bps[t["symbol"]] = bps
                        # Live has something the backtester has to estimate:
                        # the actual quoted spread. Charge the fill against it.
                        self.broker.costs.spreads_bps[t["symbol"]] = bps
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("spread refresh failed: %s", exc)
            await asyncio.sleep(SPREAD_REFRESH_SECONDS)

    async def _repair_stale(self, rest: BinanceREST) -> None:
        """Refill the window of any symbol whose stream went quiet.

        A websocket reconnect loses every message sent while it was down. The
        series would carry on with a hole in it, and the volatility and volume
        statistics computed across that hole are simply wrong -- worse than
        having no data, because they look fine.
        """
        while not self._stop.is_set():
            await asyncio.sleep(STALE_CHECK_SECONDS)
            now = time.time()
            stale = [
                s for s in self.symbols
                if now - self.last_seen.get(s, now) > STALE_AFTER_SECONDS
            ]
            if not stale:
                continue
            log.warning(
                "%d symbol(s) stale for >%ds, repairing from REST: %s",
                len(stale), STALE_AFTER_SECONDS, ", ".join(stale[:8]),
            )
            try:
                await self._seed(rest, stale)
                self.repairs += len(stale)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("stale repair failed: %s", exc)

    async def _sample_equity(self) -> None:
        """Feed the dashboard curve. Cheap, so it runs far more often than the CSV."""
        while not self._stop.is_set():
            equity = self.broker.mark_to_market(
                self.strategy.positions, self.strategy.last_price
            )
            self.dash_state.record_equity(int(time.time() * 1000), equity)
            await asyncio.sleep(DASHBOARD_SAMPLE_SECONDS)

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(HEARTBEAT_SECONDS)
            equity = self.broker.mark_to_market(
                self.strategy.positions, self.strategy.last_price
            )
            now_ms = int(time.time() * 1000)
            self.logbook.record_equity(now_ms, equity, len(self.strategy.positions))
            held = ", ".join(
                f"{s} {p.unrealized_pct(self.strategy.last_price.get(s, p.entry_price)):+.2f}%"
                for s, p in self.strategy.positions.items()
            )
            log.info(
                "equity %.2f (%+.2f%%) | %d open%s | %d trades | %d signals",
                equity,
                (equity / self.cfg.risk.starting_equity - 1.0) * 100.0,
                len(self.strategy.positions),
                f" [{held}]" if held else "",
                len(self.broker.trades),
                len(self.strategy.signals_seen),
            )

    # -- main --------------------------------------------------------------
    async def run(self) -> None:
        prefix = "BINANCE_TESTNET_API" if self.cfg.is_testnet else "BINANCE_API"
        key = os.getenv(f"{prefix}_KEY", "")
        secret = os.getenv(f"{prefix}_SECRET", "")

        logging.getLogger().addHandler(self.log_tail)

        async with BinanceREST(self.cfg.rest_url, key, secret) as rest:
            await self._prepare(rest)

            if self.dashboard:
                await self.dashboard.start()

            tasks = [
                asyncio.create_task(self._refresh_spreads(rest)),
                asyncio.create_task(self._heartbeat()),
                asyncio.create_task(self._sample_equity()),
                asyncio.create_task(self._repair_stale(rest)),
            ]
            log.info("--- %s mode running. Ctrl-C to stop. ---", self.cfg.mode.upper())
            try:
                async for symbol, candle in stream_klines(
                    self.cfg.stream_url, self.symbols, self.cfg.strategy.timeframe
                ):
                    self.last_seen[symbol] = time.time()
                    for trade in self.strategy.on_candle(symbol, candle):
                        self.logbook.record_trade(trade)
                    self._flush_signals()
                    self.dash_state.stream_health = self._stream_health()
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                self._stop.set()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self.dashboard:
                    await self.dashboard.stop()
                logging.getLogger().removeHandler(self.log_tail)
                self._shutdown()

    def _stream_health(self) -> dict:
        now = time.time()
        ages = [now - self.last_seen.get(s, now) for s in self.symbols]
        stale = sum(1 for a in ages if a > STALE_AFTER_SECONDS)
        return {
            "symbols": len(self.symbols),
            "stale": stale,
            "worst_age_s": max(ages) if ages else 0.0,
            "repairs": self.repairs,
        }

    def _flush_signals(self) -> None:
        while self._logged_signals < len(self.strategy.signals_seen):
            self.logbook.record_signal(self.strategy.signals_seen[self._logged_signals])
            self._logged_signals += 1

    def _shutdown(self) -> None:
        self._stop.set()
        now_ms = int(time.time() * 1000)
        for trade in self.strategy.close_all(now_ms):
            self.logbook.record_trade(trade)
            log.info("closed %s at shutdown: %+.2f%%", trade.symbol, trade.net_pct)
        stats = summarize(
            self.broker.trades, self.cfg.risk.starting_equity, self.broker.equity
        )
        print_summary(stats, self.strategy.rejections)
        log.info("trade log: %s", self.cfg.path(self.cfg.logging.trades_csv))
