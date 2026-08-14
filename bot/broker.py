"""Order execution. PaperBroker simulates fills; LiveBroker talks to Binance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config
from .costs import CostModel
from .exchange import SymbolFilters
from .levels import Levels

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float          # actual fill, slippage included
    entry_time_ms: int
    stop_price: float
    target_price: float
    entry_fee: float
    peak_price: float           # high-water mark, drives the trailing stop
    # Exit distances in force for this trade. Per-position rather than global
    # because in ATR mode every symbol gets its own.
    trail_pct: float
    trail_activate_pct: float
    trailing_armed: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def unrealized_pct(self, price: float) -> float:
        return (price / self.entry_price - 1.0) * 100.0


@dataclass
class Trade:
    symbol: str
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    net_pct: float
    exit_reason: str
    hold_minutes: float
    equity_after: float


class Broker(Protocol):
    """What a real LiveBroker would have to implement.

    This drifted out of date once -- it still declared `buy` without `levels`
    long after the exits became per-symbol, quietly documenting an interface
    that no longer existed. Nothing type-checks against it, so nothing caught
    it; a test now compares it against PaperBroker's actual signatures.
    """

    equity: float
    fees_paid: float

    def buy(self, symbol: str, quote_amount: float, price: float, time_ms: int,
            levels: Levels) -> Position | None: ...
    def sell(self, position: Position, price: float, time_ms: int,
             reason: str) -> Trade: ...
    def mark_to_market(self, positions: dict[str, Position],
                       prices: dict[str, float]) -> float: ...


class PaperBroker:
    """Simulated market orders.

    Fills are deliberately pessimistic: the entry pays `slippage_bps` above the
    signal price and the exit receives `slippage_bps` below it, plus taker fees
    on both legs. Stop exits are additionally assumed to fill at the stop, not
    better -- a real stop in a fast market often does worse.
    """

    def __init__(
        self,
        cfg: Config,
        filters: dict[str, SymbolFilters] | None = None,
        costs: CostModel | None = None,
    ):
        self.cfg = cfg
        self.filters = filters or {}
        self.costs = costs or CostModel(cfg)
        self.equity = cfg.risk.starting_equity
        self.trades: list[Trade] = []
        # Every fee handed to the exchange, including those on positions still
        # open. Lets the viewer draw the counterfactual: what the account would
        # be worth if trading were free.
        self.fees_paid = 0.0

    @property
    def fee_rate(self) -> float:
        return self.costs.fee_rate

    # -- helpers -----------------------------------------------------------
    def _apply_filters(self, symbol: str, qty: float, price: float) -> float:
        f = self.filters.get(symbol)
        if not f:
            return qty
        qty = f.round_qty(qty)
        if qty < f.min_qty:
            return 0.0
        if f.min_notional and qty * price < f.min_notional:
            return 0.0
        return qty

    # -- orders ------------------------------------------------------------
    def buy(
        self,
        symbol: str,
        quote_amount: float,
        price: float,
        time_ms: int,
        levels: Levels,
    ) -> Position | None:
        fill_price = price * (1.0 + self.costs.slippage_rate(symbol))
        qty = self._apply_filters(symbol, quote_amount / fill_price, fill_price)
        if qty <= 0:
            log.debug("%s: order below exchange minimums, skipped", symbol)
            return None

        cost = qty * fill_price
        fee = cost * self.fee_rate
        if cost + fee > self.equity:
            log.debug("%s: insufficient equity for %.2f", symbol, cost)
            return None

        self.equity -= cost + fee
        self.fees_paid += fee
        stop, target = levels.apply(fill_price)
        return Position(
            symbol=symbol,
            qty=qty,
            entry_price=fill_price,
            entry_time_ms=time_ms,
            stop_price=stop,
            target_price=target,
            entry_fee=fee,
            peak_price=fill_price,
            trail_pct=levels.trail_pct,
            trail_activate_pct=levels.trail_activate_pct,
        )

    def sell(self, position: Position, price: float, time_ms: int, reason: str) -> Trade:
        # A take-profit rests in the book: no spread crossed, maker fee. A stop
        # or a time exit is a market order and pays both. Stops don't get
        # slippage in our favour either way.
        maker_exit = reason == "take_profit" and self.cfg.fees.target_exit_is_maker
        slip = 0.0 if maker_exit else self.costs.slippage_rate(position.symbol)
        rate = (self.cfg.fees.effective_maker_bps / 10_000.0 if maker_exit
                else self.fee_rate)
        fill_price = price * (1.0 - slip)
        proceeds = position.qty * fill_price
        fee = proceeds * rate
        self.equity += proceeds - fee
        self.fees_paid += fee

        gross = (fill_price - position.entry_price) * position.qty
        fees = position.entry_fee + fee
        cost_basis = position.qty * position.entry_price
        trade = Trade(
            symbol=position.symbol,
            entry_time_ms=position.entry_time_ms,
            exit_time_ms=time_ms,
            entry_price=position.entry_price,
            exit_price=fill_price,
            qty=position.qty,
            gross_pnl=gross,
            fees=fees,
            net_pnl=gross - fees,
            net_pct=(gross - fees) / cost_basis * 100.0 if cost_basis else 0.0,
            exit_reason=reason,
            hold_minutes=(time_ms - position.entry_time_ms) / 60_000.0,
            equity_after=self.equity,
        )
        self.trades.append(trade)
        return trade

    def mark_to_market(self, positions: dict[str, Position], prices: dict[str, float]) -> float:
        """Cash plus the current value of open positions."""
        total = self.equity
        for symbol, pos in positions.items():
            total += pos.qty * prices.get(symbol, pos.entry_price)
        return total
