"""Position sizing and the guardrails that stop a bad day becoming a blown account."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .config import Config

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.day: str = ""
        self.day_start_equity: float = cfg.risk.starting_equity
        self.halted_until_next_day = False
        self.cooldowns: dict[str, int] = {}  # symbol -> unix ms when it expires

    # -- day bookkeeping ---------------------------------------------------
    def roll_day(self, time_ms: int, equity: float) -> None:
        day = datetime.fromtimestamp(time_ms / 1000, UTC).strftime("%Y-%m-%d")
        if day != self.day:
            self.day = day
            self.day_start_equity = equity
            self.halted_until_next_day = False

    def day_pnl_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (equity / self.day_start_equity - 1.0) * 100.0

    # -- gates -------------------------------------------------------------
    def can_open(
        self, symbol: str, time_ms: int, equity: float, open_positions: int
    ) -> tuple[bool, str]:
        self.roll_day(time_ms, equity)

        if self.halted_until_next_day:
            return False, "daily_loss_limit"

        if self.day_pnl_pct(equity) <= -self.cfg.risk.daily_loss_limit_pct:
            self.halted_until_next_day = True
            log.warning(
                "daily loss limit hit (%.2f%%) -- no new entries until %s UTC rolls over",
                self.day_pnl_pct(equity),
                self.day,
            )
            return False, "daily_loss_limit"

        if open_positions >= self.cfg.risk.max_positions:
            return False, "max_positions"

        expiry = self.cooldowns.get(symbol, 0)
        if time_ms < expiry:
            return False, "cooldown"

        return True, ""

    def start_cooldown(self, symbol: str, time_ms: int) -> None:
        self.cooldowns[symbol] = time_ms + self.cfg.risk.cooldown_minutes * 60_000

    # -- sizing ------------------------------------------------------------
    def position_size(self, equity: float) -> float:
        """Quote-currency notional for one trade.

        Risk-based: risking `risk_per_trade_pct` of equity across a
        `stop_loss_pct` stop implies notional = equity * risk / stop. Capped so a
        tight stop can't produce an absurd position.
        """
        r = self.cfg.risk
        notional = equity * (r.risk_per_trade_pct / 100.0) / (r.stop_loss_pct / 100.0)
        cap = equity * (r.max_position_pct / 100.0)
        notional = min(notional, cap)
        return notional if notional >= r.min_notional else 0.0
