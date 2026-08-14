"""Where the stop, target and trail go for one trade.

Split out so the live engine, the backtester, the ML labeller and the sweep all
compute exit distances the same way. Two implementations of this would be two
different strategies wearing the same name.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .series import Series


@dataclass(slots=True, frozen=True)
class Levels:
    """Exit distances for one trade, in percent of entry price."""

    stop_pct: float
    target_pct: float
    trail_pct: float
    trail_activate_pct: float
    atr_pct: float = 0.0  # 0 in fixed mode; recorded for the trade log

    def apply(self, entry: float) -> tuple[float, float]:
        return (
            entry * (1.0 - self.stop_pct / 100.0),
            entry * (1.0 + self.target_pct / 100.0),
        )


def fixed_levels(cfg: Config) -> Levels:
    r = cfg.risk
    return Levels(
        stop_pct=r.stop_loss_pct,
        target_pct=r.take_profit_pct,
        trail_pct=r.trailing_stop_pct,
        trail_activate_pct=r.trail_activate_pct,
    )


def levels_for(cfg: Config, series: Series | None = None, atr_pct: float | None = None) -> Levels:
    """Exit distances for the symbol described by `series` (or a known ATR).

    In ATR mode every distance is a multiple of the symbol's own average true
    range, so a 1.5x stop means the same thing on BTC and on a microcap. Falls
    back to fixed distances when there is not enough history to measure ATR --
    silently using a bad ATR would be worse than using the flat default.
    """
    r = cfg.risk
    if r.stop_mode != "atr":
        return fixed_levels(cfg)

    if atr_pct is None:
        atr_pct = series.atr_pct(r.atr_bars) if series is not None else 0.0
    if atr_pct <= 0:
        return fixed_levels(cfg)

    stop = min(max(atr_pct * r.stop_atr_mult, r.min_stop_pct), r.max_stop_pct)
    # Ratios are preserved when the bounds bite, so risk:reward stays intact.
    scale = stop / (atr_pct * r.stop_atr_mult)
    return Levels(
        stop_pct=stop,
        target_pct=atr_pct * r.target_atr_mult * scale,
        trail_pct=atr_pct * r.trail_atr_mult * scale,
        # Arm the trail once the trade has covered its own stop distance.
        trail_activate_pct=stop,
        atr_pct=atr_pct,
    )
