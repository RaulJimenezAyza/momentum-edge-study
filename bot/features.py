"""Feature extraction for the ML filter.

One definition, used by both training and live inference. If these ever diverge
the model is scoring garbage at runtime while looking fine offline -- the single
most common way an ML trading filter silently fails. Hence one function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .series import Series


@dataclass(slots=True, frozen=True)
class MarketContext:
    """Regime features shared by every symbol at a given minute (BTC as proxy).

    Passed in rather than derived inside extract() so the trainer can index it
    by timestamp instead of replaying a second Series per symbol.
    """

    ret_15: float = 0.0
    ret_60: float = 0.0
    sigma: float = 0.0

    @classmethod
    def from_series(cls, market: Series | None) -> "MarketContext":
        if market is None or len(market) < 60:
            return cls()
        return cls(
            ret_15=market.ret_pct(15),
            ret_60=market.ret_pct(60),
            sigma=market.return_sigma_pct(1),
        )

FEATURE_NAMES: tuple[str, ...] = (
    # -- the move itself
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_60",
    "zscore_3", "zscore_15",
    "sigma_1", "sigma_3",
    # -- participation
    "volume_mult", "volume_mult_5", "log_quote_volume_5",
    # -- location within the recent range
    "extension_30", "dist_high_30", "dist_high_120", "dist_low_120",
    # -- candle shape
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_in_range",
    # -- persistence
    "green_10", "consecutive_up",
    # -- market regime (BTC as proxy)
    "market_ret_15", "market_ret_60", "market_sigma",
    # -- session
    "hour_sin", "hour_cos",
)


def _safe(value: float, lo: float = -50.0, hi: float = 50.0) -> float:
    """Clamp and de-NaN. Outliers are real but they dominate a linear model."""
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return max(lo, min(hi, value))


def extract(series: Series, market: MarketContext | None = None) -> list[float]:
    """Feature vector for the last CLOSED candle of `series`."""
    closed = series.closed
    if len(closed) < 60:
        return [0.0] * len(FEATURE_NAMES)

    last = closed[-1]
    price = last.close

    ret_1 = series.ret_pct(1)
    ret_3 = series.ret_pct(3)
    ret_5 = series.ret_pct(5)
    ret_15 = series.ret_pct(15)
    ret_60 = series.ret_pct(60)

    sigma_1 = series.return_sigma_pct(1)
    sigma_3 = series.return_sigma_pct(3)
    sigma_15 = series.return_sigma_pct(15)
    z_3 = ret_3 / sigma_3 if sigma_3 > 0 else 0.0
    z_15 = ret_15 / sigma_15 if sigma_15 > 0 else 0.0

    vol_mult = series.volume_mult()
    recent_vol = sum(c.volume for c in list(closed)[-5:]) / 5.0
    prior = [c.volume for c in list(closed)[:-5]]
    prior_mean = sum(prior) / len(prior) if prior else 0.0
    vol_mult_5 = recent_vol / prior_mean if prior_mean > 0 else 0.0
    qv5 = series.quote_volume_recent(5)

    high_30 = series.highest_close(30, offset=1)
    high_120 = max((c.high for c in closed), default=price)
    low_120 = min((c.low for c in closed), default=price)

    rng = last.high - last.low
    body = abs(last.close - last.open)
    body_ratio = body / rng if rng > 0 else 0.0
    upper_wick = (last.high - max(last.open, last.close)) / rng if rng > 0 else 0.0
    lower_wick = (min(last.open, last.close) - last.low) / rng if rng > 0 else 0.0
    close_in_range = (last.close - last.low) / rng if rng > 0 else 0.5

    last10 = list(closed)[-10:]
    green_10 = sum(1 for c in last10 if c.close > c.open) / 10.0
    consecutive = 0
    for c in reversed(last10):
        if c.close > c.open:
            consecutive += 1
        else:
            break

    ctx = market or MarketContext()
    m_ret_15, m_ret_60, m_sigma = ctx.ret_15, ctx.ret_60, ctx.sigma

    # Crypto trades 24/7 but liquidity is not flat across the day.
    hour = datetime.fromtimestamp(last.open_time / 1000, timezone.utc).hour
    angle = 2.0 * math.pi * hour / 24.0

    values = [
        ret_1, ret_3, ret_5, ret_15, ret_60,
        z_3, z_15,
        sigma_1, sigma_3,
        vol_mult, vol_mult_5, math.log1p(qv5),
        series.extension_pct(30),
        (high_30 / price - 1.0) * 100.0 if price > 0 else 0.0,
        (high_120 / price - 1.0) * 100.0 if price > 0 else 0.0,
        (price / low_120 - 1.0) * 100.0 if low_120 > 0 else 0.0,
        body_ratio, upper_wick, lower_wick, close_in_range,
        green_10, float(consecutive),
        m_ret_15, m_ret_60, m_sigma,
        math.sin(angle), math.cos(angle),
    ]
    assert len(values) == len(FEATURE_NAMES), "feature vector / name mismatch"
    return [_safe(v) for v in values]
