"""Does the edge grow with the size of the dislocation?

One test, one prior, decided before looking: if the reversal effect is
overreaction, a bigger shock should produce a bigger bounce, and the edge
should rise monotonically with |z|. If it is flat, there is no threshold worth
finding and the search stops there.

This is deliberately not a grid. Bucketing by z and reading off the best bucket
is the same p-hacking as scanning 108 configs -- the protection is that the
shape is predicted in advance. A monotone rise is evidence. A single high
bucket in the middle of a flat curve is noise, and is reported as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .levels import levels_for
from .nullcheck import Arm, welch
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import estimate_spreads

log = logging.getLogger(__name__)


@dataclass
class Bucket:
    lo: float
    hi: float
    arm: Arm
    random: Arm
    gross: list[float]

    @property
    def edge(self) -> float:
        return self.arm.mean - self.random.mean

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0


async def run_edgecurve(
    cfg: Config,
    days: float = 45.0,
    symbols: list[str] | None = None,
    offset_days: float = 0.0,
    edges: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 99.0),
    draws: int = 2,
) -> dict:
    history, _ = await load_history(
        cfg, days, symbols, need_market=False, offset_days=offset_days
    )
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {}
    )
    log.info("%s", costs.summary())

    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    import random as _random

    rng = _random.Random(11)

    lows = (0.0,) + edges[:-1]
    buckets = [Bucket(lo, hi, Arm(f"{lo}-{hi}", []), Arm("rnd", []), [])
               for lo, hi in zip(lows, edges)]

    def bucket_for(z: float) -> Bucket | None:
        for b in buckets:
            if b.lo <= z < b.hi:
                return b
        return None

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1
        found: list[Bucket] = []

        for i in range(lookback, last_index):
            series.update(candles[i])
            result = evaluate(series, cfg.strategy, 0.0)
            if isinstance(result, Rejection):
                continue
            b = bucket_for(result.zscore)
            if b is None:
                continue
            outcome = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon],
                cfg, levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if outcome is not None:
                b.arm.nets.append(outcome.net_pct)
                b.gross.append(outcome.gross_pct)
                found.append(b)

        # Each bucket gets its own random benchmark on the same symbol, so a
        # bucket dominated by illiquid coins is compared against illiquid coins.
        for b in found:
            for _ in range(draws):
                j = rng.randrange(lookback, last_index)
                window = Series(symbol, lookback)
                for c in candles[max(0, j - lookback): j + 1]:
                    window.update(c)
                outcome = simulate_exit(
                    candles[j].close, candles[j + 1: j + 1 + horizon],
                    cfg, levels_for(cfg, window), costs.slippage_bps(symbol),
                )
                if outcome is not None:
                    b.random.nets.append(outcome.net_pct)

    populated = [b for b in buckets if b.arm.n >= 50]
    _print(cfg, populated, days, offset_days)
    return {
        "buckets": [
            {"lo": b.lo, "hi": b.hi, "n": b.arm.n, "gross": b.gross_mean,
             "net": b.arm.mean, "edge": b.edge, "p": welch(b.arm, b.random)[1]}
            for b in populated
        ]
    }


def _print(cfg: Config, buckets: list[Bucket], days: float, offset: float) -> None:
    window = f"{days:.0f}d" + (f" ending {offset:.0f}d ago" if offset else "")
    print("\n" + "=" * 80)
    print(f"  EDGE vs DISLOCATION SIZE -- {cfg.strategy.direction} "
          f"{cfg.strategy.timeframe}, {window}")
    print("=" * 80)
    if not buckets:
        print("  no bucket reached 50 trades")
        print("=" * 80 + "\n")
        return

    print(f"    {'z range':<12}{'n':>7}{'wins':>8}{'gross':>10}{'net':>10}"
          f"{'random':>10}{'edge':>9}{'p':>8}")
    for b in buckets:
        hi = "inf" if b.hi >= 99 else f"{b.hi:.1f}"
        _, p = welch(b.arm, b.random)
        print(f"    {f'{b.lo:.1f}-{hi}':<12}{b.arm.n:>7}{b.arm.win_rate:>7.1f}%"
              f"{b.gross_mean:>+9.3f}%{b.arm.mean:>+9.3f}%"
              f"{b.random.mean:>+9.3f}%{b.edge:>+8.3f}%{p:>8.3f}")

    print("-" * 80)
    edges = [b.edge for b in buckets]
    rises = sum(1 for a, c in zip(edges, edges[1:]) if c > a)
    monotone = rises == len(edges) - 1
    best = max(buckets, key=lambda b: b.arm.mean)

    if monotone:
        print("  The edge rises monotonically with dislocation size, as the")
        print("  overreaction hypothesis predicts. A higher z threshold is")
        print(f"  justified: the {best.lo:.1f}+ bucket nets {best.arm.mean:+.3f}%.")
    elif rises >= len(edges) - 2:
        print("  Mostly rising with dislocation size -- consistent with")
        print("  overreaction, but with one inversion. Weak evidence.")
    else:
        print("  The edge does NOT grow with dislocation size. The prediction the")
        print("  overreaction story makes is not in the data, so a higher z")
        print("  threshold buys fewer trades and no better ones.")

    if any(b.arm.mean > 0 for b in buckets):
        winners = [b for b in buckets if b.arm.mean > 0]
        print(f"\n  Net-positive buckets: "
              + ", ".join(f"z {b.lo:.1f}+ ({b.arm.mean:+.3f}%, n={b.arm.n})"
                          for b in winners))
        print("  Judge these on the holdout window before believing them.")
    else:
        print("\n  No bucket is net positive at this cost level.")
    print("=" * 80 + "\n")
