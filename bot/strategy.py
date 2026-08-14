"""The trading logic itself -- shared verbatim by the live engine and backtester.

Keeping one implementation is the point: a backtest that runs different code
from the live bot measures nothing.
"""

from __future__ import annotations

import logging

from .broker import PaperBroker, Position, Trade
from .config import Config
from .exchange import Candle
from .features import MarketContext, extract
from .levels import levels_for
from .risk import RiskManager
from .series import Series
from .signals import Diagnostics, Rejection, Signal, diagnose, evaluate

log = logging.getLogger(__name__)

MARKET_SYMBOL = "BTCUSDT"


class Strategy:
    def __init__(
        self,
        cfg: Config,
        broker: PaperBroker,
        risk: RiskManager,
        model: object | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.risk = risk
        self.model = model
        self.series: dict[str, Series] = {}
        self.positions: dict[str, Position] = {}
        self.last_price: dict[str, float] = {}
        self.spread_bps: dict[str, float] = {}
        self.rejections: dict[str, int] = {}
        self.signals_seen: list[Signal] = []
        # Live only: full gate readouts for the dashboard. Computing these for
        # every bar of a backtest would triple its runtime for no benefit.
        self.collect_diagnostics = False
        self.diagnostics: dict[str, Diagnostics] = {}

    @property
    def relax(self) -> float:
        """Gate looseness. Only the model is allowed to widen the funnel."""
        return self.cfg.ml.relax_factor if self.model is not None else 1.0

    @property
    def ml_threshold(self) -> float:
        """Probability cut-off in force, from the model file unless pinned."""
        if self.model is None or self.cfg.ml.threshold_mode == "fixed":
            return self.cfg.ml.min_probability
        return float(getattr(self.model, "meta", {}).get(
            "threshold", self.cfg.ml.min_probability
        ))

    def _market_context(self) -> MarketContext:
        return MarketContext.from_series(self.series.get(MARKET_SYMBOL))

    # -- setup -------------------------------------------------------------
    def seed(self, symbol: str, candles: list[Candle]) -> None:
        """Load a symbol's rolling window from scratch.

        Replaces any existing series rather than appending. The stale-symbol
        watchdog calls this to repair a hole after a dropped stream, and
        appending an overlapping batch would leave older bars sitting after
        newer ones -- every statistic downstream would then be nonsense.
        """
        if not candles:
            return
        s = Series(symbol, self.cfg.strategy.lookback_bars)
        for c in candles:
            s.update(c)
        self.series[symbol] = s
        self.last_price[symbol] = candles[-1].close

    # -- main entry point --------------------------------------------------
    def on_candle(self, symbol: str, candle: Candle) -> list[Trade]:
        """Feed one candle (forming or closed). Returns any trades it closed."""
        series = self.series.setdefault(
            symbol, Series(symbol, self.cfg.strategy.lookback_bars)
        )
        bar_closed = series.update(candle)
        self.last_price[symbol] = candle.close

        # Exits are evaluated on every tick, using the candle's extremes so an
        # intrabar spike through the stop is not missed.
        closed = self._check_exit(symbol, candle)
        trades = [closed] if closed else []

        if bar_closed and symbol not in self.positions:
            self._check_entry(symbol, series, candle)

        return trades

    # -- exits -------------------------------------------------------------
    def _check_exit(self, symbol: str, candle: Candle) -> Trade | None:
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        time_ms = candle.open_time + self.cfg.bar_ms

        # Exits are settled against the levels as they stood when this candle
        # opened. Ratcheting the trail on the same candle's high and then
        # filling against it would be trading on information from the future.
        #
        # Stop is checked before target: within one candle we cannot know which
        # extreme came first, so assume the worse ordering rather than
        # flattering the results.
        exit_price: float | None = None
        reason = ""
        hit_stop = candle.low <= pos.stop_price
        hit_target = candle.high >= pos.target_price
        if hit_stop and hit_target:
            # Both barriers touched inside one candle. Must resolve it the same
            # way simulate_exit does or the live bot and every measurement of it
            # are describing different strategies.
            mode = self.cfg.risk.intrabar_fill
            stop_first = (
                True if mode == "pessimistic"
                else False if mode == "optimistic"
                else abs(candle.open - pos.stop_price)
                <= abs(candle.open - pos.target_price)
            )
            hit_stop, hit_target = stop_first, not stop_first

        if hit_stop:
            exit_price, reason = pos.stop_price, (
                "trailing_stop" if pos.trailing_armed else "stop_loss"
            )
        elif hit_target:
            exit_price, reason = pos.target_price, "take_profit"
        elif (time_ms - pos.entry_time_ms) >= self.cfg.max_hold_bars * self.cfg.bar_ms:
            exit_price, reason = candle.close, "max_hold"

        if exit_price is None:
            # Survived the candle -- now ratchet the stop for the next one.
            if candle.high > pos.peak_price:
                pos.peak_price = candle.high
            if not pos.trailing_armed:
                gain = (pos.peak_price / pos.entry_price - 1.0) * 100.0
                if gain >= pos.trail_activate_pct:
                    pos.trailing_armed = True
                    log.debug("%s: trailing stop armed at +%.2f%%", symbol, gain)
            if pos.trailing_armed:
                trail = pos.peak_price * (1.0 - pos.trail_pct / 100.0)
                # Never above the target -- the target would have filled first.
                pos.stop_price = max(pos.stop_price, min(trail, pos.target_price))
            return None

        trade = self.broker.sell(pos, exit_price, time_ms, reason)
        del self.positions[symbol]
        self.risk.start_cooldown(symbol, time_ms)
        # Broker.sell stamps cash only. Cash dips whenever capital sits in other
        # open positions, which would show up as a phantom drawdown, so restate
        # it as total account value.
        trade.equity_after = self.broker.mark_to_market(
            self.positions, self.last_price
        )
        log.info(
            "EXIT  %-12s %-14s %+6.2f%%  net %+8.2f  equity %.2f",
            symbol,
            reason,
            trade.net_pct,
            trade.net_pnl,
            trade.equity_after,
        )
        return trade

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    # -- entries -----------------------------------------------------------
    def _check_entry(self, symbol: str, series: Series, candle: Candle) -> None:
        spread = self.spread_bps.get(symbol, 0.0)
        relax = self.relax

        if self.collect_diagnostics:
            self.diagnostics[symbol] = diagnose(
                series, self.cfg.strategy, spread, relax
            )

        result = evaluate(series, self.cfg.strategy, spread, relax)
        if isinstance(result, Rejection):
            self._reject(result.reason)
            return

        signal: Signal = result

        if self.model is not None:
            threshold = self.ml_threshold
            probability = self.model.probability(
                extract(series, self._market_context())
            )
            signal.probability = probability
            if self.collect_diagnostics and symbol in self.diagnostics:
                self.diagnostics[symbol].gates["ml_filter"] = probability >= threshold
            if probability < threshold:
                self._reject("ml_filter")
                log.debug(
                    "%s: model scored %.3f < %.3f, skipped",
                    symbol, probability, threshold,
                )
                return

        self.signals_seen.append(signal)

        time_ms = candle.open_time + self.cfg.bar_ms
        equity = self.broker.mark_to_market(self.positions, self.last_price)
        ok, why = self.risk.can_open(symbol, time_ms, equity, len(self.positions))
        if not ok:
            self._reject(why)
            log.debug("%s: signal blocked by %s", symbol, why)
            return

        notional = self.risk.position_size(self.broker.equity)
        if notional <= 0:
            self._reject("size_too_small")
            return
        # Never spend more cash than we hold.
        notional = min(notional, self.broker.equity * 0.99)

        levels = levels_for(self.cfg, series)
        pos = self.broker.buy(symbol, notional, signal.price, time_ms, levels)
        if pos is None:
            return
        pos.meta = {
            "ret_pct": signal.ret_pct,
            "zscore": signal.zscore,
            "volume_mult": signal.volume_mult,
            "probability": signal.probability,
            "atr_pct": levels.atr_pct,
            "stop_pct": levels.stop_pct,
        }
        self.positions[symbol] = pos
        log.info(
            "ENTRY %-12s %.8f x %.6f = %.2f  (ret %+.2f%%, z %.1f, vol x%.1f%s)",
            symbol,
            pos.entry_price,
            pos.qty,
            pos.notional,
            signal.ret_pct,
            signal.zscore,
            signal.volume_mult,
            f", p={signal.probability:.2f}" if signal.probability is not None else "",
        )

    # -- teardown ----------------------------------------------------------
    def close_all(self, time_ms: int, reason: str = "shutdown") -> list[Trade]:
        trades = []
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            price = self.last_price.get(symbol, pos.entry_price)
            trades.append(self.broker.sell(pos, price, time_ms, reason))
            del self.positions[symbol]
        return trades
