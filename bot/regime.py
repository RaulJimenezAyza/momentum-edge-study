"""Does the edge depend on the market regime?

Two reasons to ask, both from evidence already collected rather than from a
hunch. The reversal model scored AUC 0.53 -- useless as a filter -- but its
permutation importance put `market_sigma`, BTC's own volatility, at the top of
the list. And the prior is clean: an overreaction trade should pay best when
the market is panicking, because that is when prices overshoot.

If the edge is several times larger in one regime and the cost is unchanged,
trading only in that regime is a real improvement rather than a re-tune. If it
is flat, the ML hint was noise and the question is closed.

Two cuts, both fixed in advance: BTC's realised volatility at signal time, and
the hour of day. Nothing else is tried -- searching cuts until one looks good
is exactly the failure mode the holdout exists to catch.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .levels import levels_for
from .nullcheck import Arm, bootstrap_ci
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot
from .strategy import MARKET_SYMBOL

log = logging.getLogger(__name__)


@dataclass
class Bucket:
    name: str
    signals: Arm = field(default_factory=lambda: Arm("sig", []))
    random: Arm = field(default_factory=lambda: Arm("rnd", []))
    gross: list[float] = field(default_factory=list)

    @property
    def edge(self) -> float:
        return self.signals.mean - self.random.mean

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0


def _btc_volatility(candles, lookback: int) -> dict[int, float]:
    """BTC's realised 1-bar volatility at each timestamp, in percent."""
    out: dict[int, float] = {}
    series = Series(MARKET_SYMBOL, lookback)
    for c in candles:
        series.update(c)
        out[c.open_time] = series.return_sigma_pct(1)
    return out


async def run_regime(
    cfg: Config,
    days: float = 45.0,
    symbols: list[str] | None = None,
    offset_days: float = 14.0,
    draws: int = 2,
    min_trades: int = 100,
) -> dict:
    history, _ = await load_history(
        cfg, days, symbols, need_market=True, offset_days=offset_days
    )
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    log.info("%s", costs.summary())

    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    vol_at = _btc_volatility(history.get(MARKET_SYMBOL, []), lookback)
    if not vol_at:
        raise RuntimeError(f"{MARKET_SYMBOL} history missing; cannot cut by regime")

    # Quartile edges from the volatility distribution itself, so the buckets are
    # equally populated rather than equally spaced.
    ordered = sorted(v for v in vol_at.values() if v > 0)
    q = [ordered[int(f * len(ordered))] for f in (0.25, 0.50, 0.75)]
    log.info("BTC 1m volatility quartiles: %.4f / %.4f / %.4f %%", *q)

    vol_buckets = [Bucket(n) for n in
                   ("calm", "quiet", "active", "turbulent")]
    hour_buckets = [Bucket(f"{h:02d}-{h + 6:02d} UTC") for h in (0, 6, 12, 18)]

    rng = random.Random(29)
    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1

        for i in range(lookback, last_index):
            series.update(candles[i])
            if isinstance(evaluate(series, cfg.strategy, costs.spread_bps(symbol)),
                          Rejection):
                continue
            t = candles[i].open_time
            vol = vol_at.get(t)
            if vol is None:
                continue
            outcome = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon], cfg,
                levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if outcome is None:
                continue

            vb = vol_buckets[sum(1 for edge in q if vol >= edge)]
            hb = hour_buckets[
                datetime.fromtimestamp(t / 1000, timezone.utc).hour // 6
            ]
            for b in (vb, hb):
                b.signals.nets.append(outcome.net_pct)
                b.gross.append(outcome.gross_pct)

            # Benchmark drawn from the same symbol AND the same regime, so a
            # bucket is never compared against a calmer or busier market.
            for _ in range(draws):
                j = rng.randrange(lookback, last_index)
                jt = candles[j].open_time
                jvol = vol_at.get(jt)
                if jvol is None:
                    continue
                window = Series(symbol, lookback)
                for c in candles[max(0, j - lookback): j + 1]:
                    window.update(c)
                ro = simulate_exit(
                    candles[j].close, candles[j + 1: j + 1 + horizon], cfg,
                    levels_for(cfg, window), costs.slippage_bps(symbol),
                )
                if ro is None:
                    continue
                if sum(1 for edge in q if jvol >= edge) == vol_buckets.index(vb):
                    vb.random.nets.append(ro.net_pct)
                if datetime.fromtimestamp(jt / 1000, timezone.utc).hour // 6 \
                        == hour_buckets.index(hb):
                    hb.random.nets.append(ro.net_pct)

    _print(cfg, "BTC VOLATILITY AT SIGNAL TIME", vol_buckets, min_trades, days, offset_days)
    _print(cfg, "HOUR OF DAY", hour_buckets, min_trades, days, offset_days)
    return {
        "volatility": [_row(b) for b in vol_buckets],
        "hour": [_row(b) for b in hour_buckets],
    }


def _row(b: Bucket) -> dict:
    lo, hi, p = bootstrap_ci(b.signals, b.random)
    return {"name": b.name, "n": b.signals.n, "gross": b.gross_mean,
            "net": b.signals.mean, "edge": b.edge, "ci": [lo, hi], "p": p}


def _print(cfg, title, buckets, min_trades, days, offset) -> None:
    live = [b for b in buckets if b.signals.n >= min_trades and b.random.n >= 30]
    window = f"{days:.0f}d" + (f" ending {offset:.0f}d ago" if offset else "")
    print("\n" + "=" * 86)
    print(f"  EDGE BY REGIME -- {title}, {cfg.strategy.direction} "
          f"{cfg.strategy.timeframe}, {window}")
    print("=" * 86)
    if not live:
        print("  no bucket reached the minimum sample")
        print("=" * 86 + "\n")
        return

    print(f"    {'bucket':<14}{'n':>7}{'wins':>8}{'gross':>10}{'net':>10}"
          f"{'random':>10}{'edge':>9}{'95% CI':>20}")
    for b in live:
        lo, hi, _ = bootstrap_ci(b.signals, b.random)
        print(f"    {b.name:<14}{b.signals.n:>7}{b.signals.win_rate:>7.1f}%"
              f"{b.gross_mean:>+9.3f}%{b.signals.mean:>+9.3f}%"
              f"{b.random.mean:>+9.3f}%{b.edge:>+8.3f}%"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}")

    print("-" * 86)
    best = max(live, key=lambda b: b.gross_mean)
    worst = min(live, key=lambda b: b.gross_mean)
    spread = best.gross_mean - worst.gross_mean
    separated = [b for b in live if bootstrap_ci(b.signals, b.random)[0] > 0]

    print(f"  Best gross: {best.name} ({best.gross_mean:+.3f}%), "
          f"worst: {worst.name} ({worst.gross_mean:+.3f}%), spread {spread:.3f}%")
    if best.gross_mean > 0 and spread > 0.05:
        print(f"\n  The regime matters. Restricting to '{best.name}' is worth testing on")
        print("  the holdout -- but the bucket was picked from this table, so treat")
        print("  it as a hypothesis, not a result.")
    else:
        print("\n  The edge does not depend meaningfully on this cut. The buckets sit")
        print("  within noise of each other, so there is no regime worth selecting.")
    if separated:
        print(f"  Beat random with the CI clear of zero: "
              + ", ".join(b.name for b in separated))
    print("=" * 86 + "\n")
