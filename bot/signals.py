"""Signal detection: volatility-normalised momentum breakout with volume confirm.

Every gate is a filter against the same failure mode -- reacting to a move that
is either noise, already spent, or untradeable after costs.

Two entry points:
  evaluate()  -- fast, short-circuits at the first failed gate. Hot path.
  diagnose()  -- computes every metric regardless. Feeds the dashboard, so a
                 human can see how close each symbol is to firing and which
                 gate is holding it back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import StrategyConfig
from .series import Series

# Order matters: the first failing gate is the reported reason, and diagnose()
# must agree with evaluate() on which one that is.
GATE_ORDER = (
    "warming_up",
    "move_too_small",
    "move_within_normal_range",
    "no_volume_confirmation",
    "not_a_breakout",
    "overextended",
    "spread_too_wide",
    "spread_too_tight",
    "outside_trading_hours",
)


def _in_trading_hours(cfg: StrategyConfig, time_ms: int) -> bool:
    """Whether a signal at this timestamp falls inside an allowed UTC window."""
    if not cfg.trading_hours_utc:
        return True
    hour = datetime.fromtimestamp(time_ms / 1000, UTC).hour
    return any(lo <= hour < hi for lo, hi in cfg.trading_hours_utc)


@dataclass(slots=True)
class Signal:
    symbol: str
    time_ms: int
    price: float
    ret_pct: float
    zscore: float
    volume_mult: float
    extension_pct: float
    breakout_level: float
    spread_bps: float
    relaxed: bool = False
    probability: float | None = None  # filled in by the ML filter, if enabled

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Rejection:
    reason: str


@dataclass(slots=True)
class Diagnostics:
    """Every gate's current value and whether it passes. For humans."""

    symbol: str
    time_ms: int
    price: float
    ret_pct: float
    zscore: float
    volume_mult: float
    extension_pct: float
    breakout_level: float
    breakout_gap_pct: float  # how far below the breakout level we are, negative = above
    spread_bps: float
    bars: int
    gates: dict[str, bool] = field(default_factory=dict)
    blocking_gate: str = ""
    heat: float = 0.0  # 0..1, how close this symbol is to firing

    @property
    def ready(self) -> bool:
        return self.blocking_gate == ""


def _thresholds(cfg: StrategyConfig, relax: float) -> tuple[float, float, float]:
    """Entry thresholds, optionally loosened for ML candidate generation."""
    return (
        cfg.min_return_pct * relax,
        cfg.min_return_zscore * relax,
        cfg.min_volume_mult * relax,
    )


def evaluate(
    series: Series,
    cfg: StrategyConfig,
    spread_bps: float = 0.0,
    relax: float = 1.0,
) -> Signal | Rejection:
    """Decide whether the just-closed candle is a tradable breakout.

    `relax` < 1 loosens the three tunable thresholds. The ML pipeline uses that
    to generate a wider pool of candidates than the hand-tuned gates would take,
    then lets the model decide among them.
    """
    if len(series) < cfg.min_bars_ready:
        return Rejection("warming_up")

    min_ret, min_z, min_vol = _thresholds(cfg, relax)
    # Reversal is the same test with the sign flipped: a violent move DOWN
    # through the recent low, bought on the bet that it overshot. Every gate
    # keeps its meaning; only the direction of "extreme" changes.
    sign = -1.0 if cfg.direction == "reversal" else 1.0

    ret = series.ret_pct(cfg.momentum_bars) * sign
    if ret < min_ret:
        return Rejection("move_too_small")

    sigma = series.return_sigma_pct(cfg.momentum_bars)
    if sigma <= 0:
        return Rejection("no_volatility_estimate")
    z = ret / sigma
    if z < min_z:
        # A 0.6% move is huge for BTC and noise for a microcap. Normalise.
        return Rejection("move_within_normal_range")

    vmult = series.volume_mult(cfg.volume_measure == "quote")
    if vmult < min_vol:
        # Price moving without volume is usually a thin-book artefact.
        return Rejection("no_volume_confirmation")

    last = series.closed[-1].close
    if sign > 0:
        level = series.highest_close(cfg.breakout_bars, offset=1)
        broke = level > 0 and last > level
    else:
        level = series.lowest_close(cfg.breakout_bars, offset=1)
        broke = level > 0 and last < level
    if not broke:
        return Rejection("not_a_breakout")

    extension = series.extension_pct(cfg.breakout_bars) * sign
    if extension > cfg.max_extension_pct:
        # Chasing a move this far from the mean is buying the exit liquidity.
        return Rejection("overextended")

    if spread_bps > cfg.max_spread_bps:
        return Rejection("spread_too_wide")
    if not _in_trading_hours(cfg, series.closed[-1].open_time):
        return Rejection("outside_trading_hours")
    if spread_bps < cfg.min_spread_bps:
        # Not a liquidity problem -- an edge problem. The tightest books belong
        # to names that do not overreact, so there is nothing to fade.
        return Rejection("spread_too_tight")

    return Signal(
        symbol=series.symbol,
        time_ms=series.closed[-1].open_time,
        price=last,
        ret_pct=ret,
        zscore=z,
        volume_mult=vmult,
        extension_pct=extension,
        breakout_level=level,
        spread_bps=spread_bps,
        relaxed=relax < 1.0,
    )


def diagnose(
    series: Series,
    cfg: StrategyConfig,
    spread_bps: float = 0.0,
    relax: float = 1.0,
) -> Diagnostics:
    """Full gate readout. ~10x the work of evaluate(), so live/scan use only."""
    bars = len(series)
    if bars == 0:
        return Diagnostics("", 0, 0, 0, 0, 0, 0, 0, 0, spread_bps, 0,
                           {"warming_up": False}, "warming_up", 0.0)

    min_ret, min_z, min_vol = _thresholds(cfg, relax)
    sign = -1.0 if cfg.direction == "reversal" else 1.0

    last = series.closed[-1].close
    ret = series.ret_pct(cfg.momentum_bars) * sign
    sigma = series.return_sigma_pct(cfg.momentum_bars)
    z = ret / sigma if sigma > 0 else 0.0
    vmult = series.volume_mult(cfg.volume_measure == "quote")
    if sign > 0:
        level = series.highest_close(cfg.breakout_bars, offset=1)
        broke = level > 0 and last > level
    else:
        level = series.lowest_close(cfg.breakout_bars, offset=1)
        broke = level > 0 and last < level
    extension = series.extension_pct(cfg.breakout_bars) * sign
    # Positive = still short of the level, negative = already through it.
    gap = (level / last - 1.0) * 100.0 * sign if last > 0 else 0.0

    gates = {
        "warming_up": bars >= cfg.min_bars_ready,
        "move_too_small": ret >= min_ret,
        "move_within_normal_range": sigma > 0 and z >= min_z,
        "no_volume_confirmation": vmult >= min_vol,
        "not_a_breakout": broke,
        "overextended": extension <= cfg.max_extension_pct,
        "spread_too_wide": spread_bps <= cfg.max_spread_bps,
        "spread_too_tight": spread_bps >= cfg.min_spread_bps,
        "outside_trading_hours": _in_trading_hours(cfg, series.closed[-1].open_time),
    }
    blocking = next((g for g in GATE_ORDER if not gates[g]), "")

    # Heat: mean progress across the three continuous gates, so the dashboard
    # can rank "about to fire" above "nowhere near".
    def progress(value: float, target: float) -> float:
        if target <= 0:
            return 1.0
        return max(0.0, min(1.0, value / target))

    heat = (
        progress(ret, min_ret)
        + progress(z, min_z)
        + progress(vmult, min_vol)
    ) / 3.0

    return Diagnostics(
        symbol=series.symbol,
        time_ms=series.closed[-1].open_time,
        price=last,
        ret_pct=ret,
        zscore=z,
        volume_mult=vmult,
        extension_pct=extension,
        breakout_level=level,
        breakout_gap_pct=gap,
        spread_bps=spread_bps,
        bars=bars,
        gates=gates,
        blocking_gate=blocking,
        heat=heat,
    )
