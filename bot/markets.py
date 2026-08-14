"""Does widening the universe help, or just buy more of the same problem?

"Add more markets" is only an improvement if the extra markets are better than
the ones already traded. More symbols at a negative per-trade expectancy is
strictly more loss, not more opportunity -- the 5-position cap means the bot is
picking from a queue, not missing trades.

There is a real prior worth testing, though. Overreaction should be *larger* in
thinner names: less capital standing ready to arbitrage a panic. Their spreads
are also wider. So the question is quantitative and has a clean shape:

    does the edge grow faster than the spread as liquidity falls?

This bins the expanded universe by measured spread and answers it per tier.
It is a fair test only because the cost model now charges each symbol its own
spread -- under the old flat 5 bps assumption, adding illiquid pairs would have
looked free.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .levels import levels_for
from .nullcheck import Arm, welch
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

# Upper edge of each spread tier, in bps.
TIERS = (2.0, 5.0, 12.0, 30.0, 1e9)
TIER_NAMES = ("<2 bps", "2-5 bps", "5-12 bps", "12-30 bps", ">30 bps")


@dataclass
class Tier:
    name: str
    symbols: list[str]
    spreads: list[float]
    signals: Arm
    random: Arm
    gross: list[float]

    @property
    def mean_spread(self) -> float:
        return sum(self.spreads) / len(self.spreads) if self.spreads else 0.0

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0

    @property
    def edge(self) -> float:
        return self.signals.mean - self.random.mean


async def run_markets(
    cfg: Config,
    days: float = 21.0,
    offset_days: float = 14.0,
    universe_size: int = 150,
    min_volume: float = 500_000.0,
    draws: int = 2,
) -> dict:
    wide = Config(**{**cfg.__dict__})
    wide.exchange = replace(
        cfg.exchange, universe_size=universe_size, min_quote_volume_24h=min_volume
    )
    log.info("expanding the universe to %d symbols, volume floor %.0f",
             universe_size, min_volume)

    history, _ = await load_history(
        wide, days, None, need_market=False, offset_days=offset_days
    )
    spreads = estimate_spreads(history)
    costs = CostModel(wide, spreads, load_quote_snapshot(wide.path(QUOTE_SNAPSHOT)))
    unpriced = costs.unpriced(history)
    if unpriced:
        log.warning("%d/%d symbols unpriced -> flat fallback, gated out of trading",
                    len(unpriced), len(history))

    tiers = [Tier(n, [], [], Arm(n, []), Arm("rnd", []), []) for n in TIER_NAMES]

    def tier_for(symbol: str) -> Tier:
        s = spreads.get(symbol, 999.0)
        for edge, tier in zip(TIERS, tiers):
            if s < edge:
                return tier
        return tiers[-1]

    lookback = wide.strategy.lookback_bars
    horizon = wide.max_hold_bars
    rng = random.Random(19)

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        tier = tier_for(symbol)
        tier.symbols.append(symbol)
        tier.spreads.append(spreads.get(symbol, 0.0))

        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1
        found = 0

        for i in range(lookback, last_index):
            series.update(candles[i])
            if isinstance(evaluate(series, wide.strategy, costs.spread_bps(symbol)), Rejection):
                continue
            outcome = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon],
                wide, levels_for(wide, series), costs.slippage_bps(symbol),
            )
            if outcome is not None:
                tier.signals.nets.append(outcome.net_pct)
                tier.gross.append(outcome.gross_pct)
                found += 1

        for _ in range(found * draws):
            j = rng.randrange(lookback, last_index)
            window = Series(symbol, lookback)
            for c in candles[max(0, j - lookback): j + 1]:
                window.update(c)
            outcome = simulate_exit(
                candles[j].close, candles[j + 1: j + 1 + horizon],
                wide, levels_for(wide, window), costs.slippage_bps(symbol),
            )
            if outcome is not None:
                tier.random.nets.append(outcome.net_pct)

    populated = [t for t in tiers if t.signals.n >= 50]
    _print(wide, populated, days, offset_days, len(history))
    return {
        "tiers": [
            {"name": t.name, "symbols": len(t.symbols), "spread": t.mean_spread,
             "n": t.signals.n, "gross": t.gross_mean, "net": t.signals.mean,
             "edge": t.edge, "p": welch(t.signals, t.random)[1]}
            for t in populated
        ]
    }


def _print(cfg: Config, tiers: list[Tier], days: float, offset: float, n_sym: int) -> None:
    print("\n" + "=" * 88)
    print(f"  UNIVERSE EXPANSION -- {n_sym} symbols, {days:.0f}d ending {offset:.0f}d ago, "
          f"{cfg.strategy.direction} {cfg.strategy.timeframe}")
    print("=" * 88)
    if not tiers:
        print("  no tier reached 50 trades")
        print("=" * 88 + "\n")
        return

    print(f"    {'spread tier':<12}{'syms':>6}{'spread':>9}{'n':>7}{'wins':>7}"
          f"{'gross':>10}{'net':>10}{'edge':>9}{'p':>8}")
    for t in tiers:
        _, p = welch(t.signals, t.random)
        print(f"    {t.name:<12}{len(t.symbols):>6}{t.mean_spread:>8.1f}b"
              f"{t.signals.n:>7}{t.signals.win_rate:>6.1f}%"
              f"{t.gross_mean:>+9.3f}%{t.signals.mean:>+9.3f}%"
              f"{t.edge:>+8.3f}%{p:>8.3f}")

    print("-" * 88)
    liquid, illiquid = tiers[0], tiers[-1]
    edges = [t.edge for t in tiers]
    grosses = [t.gross_mean for t in tiers]
    rising_edge = sum(1 for a, b in zip(edges, edges[1:]) if b > a)

    print(f"  Edge, most liquid tier:   {liquid.edge:+.3f}%  (spread {liquid.mean_spread:.1f} bps)")
    print(f"  Edge, least liquid tier:  {illiquid.edge:+.3f}%  (spread {illiquid.mean_spread:.1f} bps)")

    if rising_edge >= len(edges) - 2 and illiquid.edge > liquid.edge:
        print("\n  The edge DOES grow as liquidity falls, as overreaction predicts.")
    else:
        print("\n  The edge does NOT reliably grow as liquidity falls.")

    best = max(tiers, key=lambda t: t.net if False else t.gross_mean)
    print(f"  Best gross tier: {best.name} at {best.gross_mean:+.3f}%")

    positive = [t for t in tiers if t.signals.mean > 0]
    if positive:
        print("\n  NET-POSITIVE tiers: "
              + ", ".join(f"{t.name} ({t.signals.mean:+.3f}%, n={t.signals.n})"
                          for t in positive))
        print("  Confirm on the holdout before acting on it.")
    else:
        gross_pos = [t for t in tiers if t.gross_mean > 0]
        if gross_pos:
            print("\n  Gross-positive but net-negative tiers: "
                  + ", ".join(f"{t.name} ({t.gross_mean:+.3f}% gross)" for t in gross_pos))
            print("  Widening the universe into these adds trades that lose less,")
            print("  not trades that win. It does not turn the account positive.")
        else:
            print("\n  No tier is even gross-positive. Adding markets adds losses.")
    print("=" * 88 + "\n")
