"""Do the entry rules do anything at all?

Every result so far compares the strategy to zero. This compares it to the only
benchmark that actually isolates the entry logic: **random entries, same
symbols, same exits, same costs**.

If buying at random moments scores the same as buying the carefully filtered
breakouts, then the six gates are decoration -- all the machinery is doing is
picking when to pay fees. That is a fact worth knowing before adding a seventh
gate, and it is the single cheapest experiment in the whole project.

The comparison is paired on purpose: for every real signal, a random bar is
drawn from the SAME symbol, so the two samples see the same universe, the same
liquidity and the same period. A difference cannot then be explained by one
side happening to trade easier coins.

There is a second control arm because same-symbol is not enough. The strategy
fires during dislocations, so its trades sit in volatile minutes by
construction, while a uniformly drawn random bar sits in an average one. If
volatile minutes pay differently -- and with a fixed-percentage stop and target
they must, because both barriers are reached more often -- then part of any
measured "edge" is *when* it trades rather than *what* it picks.

The VOLATILITY-MATCHED arm draws each control from the same symbol AND from a
bar whose recent volatility is close to the signal bar's. What survives that is
selection. What does not was timing.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .findings import record
from .levels import levels_for
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

# How close a control's recent volatility must be to the signal's to count as
# matched. Tight enough to remove the timing effect, loose enough that most
# signals find a control at all.
MATCH_TOLERANCE = 0.25


@dataclass
class Arm:
    name: str
    nets: list[float]

    @property
    def n(self) -> int:
        return len(self.nets)

    @property
    def mean(self) -> float:
        return sum(self.nets) / self.n if self.nets else 0.0

    @property
    def win_rate(self) -> float:
        return sum(1 for x in self.nets if x > 0) / self.n * 100.0 if self.nets else 0.0

    @property
    def stdev(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self.nets) / (self.n - 1))

    @property
    def stderr(self) -> float:
        return self.stdev / math.sqrt(self.n) if self.n else 0.0


def bootstrap_ci(
    a: Arm, b: Arm, samples: int = 2000, seed: int = 5, level: float = 0.95
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the difference in means, plus a p-value.

    Welch's t assumes roughly normal means. Trade returns are not normal: they
    are truncated at the stop and the target, so the distribution is a spike at
    each barrier plus a thin middle. At n>1000 the CLT rescues the t-test, but
    the bootstrap makes no distributional assumption at all, and when the two
    disagree the bootstrap is the one to believe.

    The returned p is the two-sided fraction of resamples on the wrong side of
    zero -- a direct read of "how often would this difference vanish".
    """
    if a.n < 30 or b.n < 30:
        return 0.0, 0.0, 1.0
    rng = random.Random(seed)
    a_nets, b_nets = a.nets, b.nets
    na, nb = len(a_nets), len(b_nets)
    diffs = []
    for _ in range(samples):
        sa = sum(a_nets[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(b_nets[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(sa - sb)
    diffs.sort()
    lo = diffs[int((1 - level) / 2 * samples)]
    hi = diffs[int((1 + level) / 2 * samples) - 1]
    crossings = sum(1 for d in diffs if d <= 0)
    p = 2.0 * min(crossings, samples - crossings) / samples
    return lo, hi, p


def block_bootstrap(
    a_times: list[int], a_nets: list[float],
    b_times: list[int], b_nets: list[float],
    block_ms: int = 6 * 3_600_000, samples: int = 1500, seed: int = 9,
) -> tuple[float, float, float]:
    """Bootstrap over TIME BLOCKS instead of individual trades.

    Every interval in this project until now assumed the trades were
    independent draws. They are not, and not marginally: a market-wide dump
    fires the entry rule on twenty symbols in the same minute, and those twenty
    outcomes are close to one observation repeated. Resampling trades one at a
    time treats each as fresh evidence, which understates the standard error
    and overstates every p-value computed from it.

    Resampling whole blocks of time keeps the trades inside a block together,
    so a clustered burst is drawn or dropped as a unit. The interval that comes
    out is wider, and it is the honest one.

    Returns (low, high, p) for the difference in means.
    """
    if len(a_nets) < 30 or len(b_nets) < 30:
        return 0.0, 0.0, 1.0

    def bucket(times, nets):
        out: dict[int, list[float]] = {}
        for t, v in zip(times, nets):
            out.setdefault(t // block_ms, []).append(v)
        return out

    a_blocks = bucket(a_times, a_nets)
    b_blocks = bucket(b_times, b_nets)
    keys = sorted(set(a_blocks) | set(b_blocks))
    if len(keys) < 8:
        # Too few blocks to resample meaningfully; say so rather than return a
        # confident-looking interval built from three points.
        return 0.0, 0.0, 1.0

    rng = random.Random(seed)
    diffs = []
    n = len(keys)
    for _ in range(samples):
        picked = [keys[rng.randrange(n)] for _ in range(n)]
        av = [v for k in picked for v in a_blocks.get(k, ())]
        bv = [v for k in picked for v in b_blocks.get(k, ())]
        if not av or not bv:
            continue
        diffs.append(sum(av) / len(av) - sum(bv) / len(bv))
    if len(diffs) < 100:
        return 0.0, 0.0, 1.0
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    crossings = sum(1 for d in diffs if d <= 0)
    p = 2.0 * min(crossings, len(diffs) - crossings) / len(diffs)
    return lo, hi, p


def welch(a: Arm, b: Arm) -> tuple[float, float]:
    """Welch's t and an approximate two-sided p, for unequal variances."""
    if a.n < 2 or b.n < 2:
        return 0.0, 1.0
    se = math.sqrt(a.stderr ** 2 + b.stderr ** 2)
    if se <= 0:
        return 0.0, 1.0
    t = (a.mean - b.mean) / se
    # Normal approximation; at these sample sizes the difference from the exact
    # t distribution is far smaller than the noise in the data itself.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


async def run_nullcheck(
    cfg: Config,
    days: float = 30.0,
    symbols: list[str] | None = None,
    draws: int = 3,
    seed: int = 0,
    offset_days: float = 0.0,
) -> dict:
    history, _ = await load_history(
        cfg, days, symbols, need_market=False, offset_days=offset_days
    )
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    log.info("%s", costs.summary())
    unpriced = costs.unpriced(history)
    if unpriced:
        log.warning(
            "%d/%d symbols have no spread estimate; at the flat fallback of "
            "%.1f bps they fail the %.1f bps entry gate and will not be traded: %s",
            len(unpriced), len(history), cfg.fees.slippage_bps * 2,
            cfg.strategy.max_spread_bps, ", ".join(sorted(unpriced)[:6]),
        )

    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    rng = random.Random(seed)

    real: list[float] = []
    real_t: list[int] = []
    fake: list[float] = []
    fake_t: list[int] = []
    matched: list[float] = []
    unmatchable = 0

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)

        last_index = len(candles) - horizon - 1
        signals_here = 0
        # Recent volatility at every bar, so a control can be matched to a
        # signal on the one dimension the strategy is guaranteed to select on.
        probe = Series(symbol, lookback)
        vol_at: list[float] = []
        for c in candles[:last_index]:
            probe.update(c)
            vol_at.append(probe.return_sigma_pct(cfg.strategy.momentum_bars))
        signal_vols: list[float] = []

        for i in range(lookback, last_index):
            series.update(candles[i])
            if isinstance(evaluate(series, cfg.strategy, costs.spread_bps(symbol)), Rejection):
                continue
            outcome = simulate_exit(
                candles[i].close,
                candles[i + 1 : i + 1 + horizon],
                cfg,
                levels_for(cfg, series),
                costs.slippage_bps(symbol),
            )
            if outcome is not None:
                real.append(outcome.net_pct)
                real_t.append(candles[i].open_time)
                signal_vols.append(vol_at[i])
                signals_here += 1

        # Same count of random entries on this symbol, drawn `draws` times to
        # shrink the benchmark's own sampling noise.
        for _ in range(signals_here * draws):
            j = rng.randrange(lookback, last_index)
            window = Series(symbol, lookback)
            for c in candles[max(0, j - lookback) : j + 1]:
                window.update(c)
            outcome = simulate_exit(
                candles[j].close,
                candles[j + 1 : j + 1 + horizon],
                cfg,
                levels_for(cfg, window),
                costs.slippage_bps(symbol),
            )
            if outcome is not None:
                fake.append(outcome.net_pct)
                fake_t.append(candles[j].open_time)

        # Volatility-matched controls: same symbol, and a bar whose recent
        # volatility sits within MATCH_TOLERANCE of the signal bar's.
        for target in signal_vols:
            if target <= 0:
                unmatchable += 1
                continue
            lo = target * (1.0 - MATCH_TOLERANCE)
            hi = target * (1.0 + MATCH_TOLERANCE)
            pool = [k for k in range(lookback, last_index) if lo <= vol_at[k] <= hi]
            if not pool:
                unmatchable += 1
                continue
            for _ in range(draws):
                j = pool[rng.randrange(len(pool))]
                window = Series(symbol, lookback)
                for c in candles[max(0, j - lookback): j + 1]:
                    window.update(c)
                out = simulate_exit(
                    candles[j].close, candles[j + 1: j + 1 + horizon], cfg,
                    levels_for(cfg, window), costs.slippage_bps(symbol),
                )
                if out is not None:
                    matched.append(out.net_pct)

    if not real:
        raise RuntimeError("the strategy produced no signals on this window")

    signal_arm = Arm("strategy signals", real)
    random_arm = Arm("random entries", fake)
    matched_arm = Arm("random, vol-matched", matched)
    t, p = welch(signal_arm, random_arm)
    edge = signal_arm.mean - random_arm.mean
    # 2000 resamples over ~20k trades is not free; compute it once and pass it.
    ci = bootstrap_ci(signal_arm, random_arm)

    blocks = block_bootstrap(real_t, real, fake_t, fake)
    _print(cfg, signal_arm, random_arm, matched_arm, edge, t, p, days, costs,
           offset_days, ci, unmatchable, blocks)
    out = {
        "signals": {"n": signal_arm.n, "mean": signal_arm.mean,
                    "win_rate": signal_arm.win_rate},
        "random": {"n": random_arm.n, "mean": random_arm.mean,
                   "win_rate": random_arm.win_rate},
        "edge_pct": edge, "t": t, "p": p,
        "ci_low": ci[0], "ci_high": ci[1], "bootstrap_p": ci[2],
        "block_ci_low": blocks[0], "block_ci_high": blocks[1],
        "block_p": blocks[2],
        "matched_n": matched_arm.n, "matched_mean": matched_arm.mean,
        "matched_edge": signal_arm.mean - matched_arm.mean,
    }
    # Same rule as everywhere else: below a few hundred signals there is no
    # verdict to publish, so nothing is written for the dashboard to read.
    if signal_arm.n >= 200:
        record(cfg, "nullcheck", {
            "edge": edge, "p": p,
            "block_p": blocks[2] if blocks[2] < 1.0 else None,
            "n": signal_arm.n, "days": days, "offset_days": offset_days,
        })
    return out


def _print(cfg, a: Arm, b: Arm, m: Arm, edge: float, t: float, p: float,
           days: float, costs: CostModel, offset: float = 0.0,
           ci: tuple[float, float, float] | None = None,
           unmatchable: int = 0,
           blocks: tuple[float, float, float] | None = None) -> None:
    print("\n" + "=" * 72)
    print(f"  NULL TEST -- {cfg.strategy.direction} entries vs random, {days:.0f} days"
          + (f", ending {offset:.0f}d ago" if offset else ""))
    print(f"  {cfg.strategy.timeframe} bars, {cfg.risk.stop_mode} stops")
    print(f"  {costs.summary()}")
    print("=" * 72)
    print(f"    {'arm':<20}{'n':>8}{'win rate':>11}{'net/trade':>13}{'+/- stderr':>11}")
    for arm in ((a, b, m) if m.n else (a, b)):
        print(f"    {arm.name:<20}{arm.n:>8}{arm.win_rate:>10.1f}%"
              f"{arm.mean:>+12.4f}%{arm.stderr:>10.4f}%")
    print("-" * 72)
    if m.n >= 100:
        m_edge = a.mean - m.mean
        m_lo, m_hi, m_p = bootstrap_ci(a, m)
        note = f"   ({unmatchable} signals had no match)" if unmatchable else ""
        print(f"    vs VOLATILITY-MATCHED controls: {m_edge:+.4f}% per trade")
        print(f"    bootstrap 95% CI: [{m_lo:+.4f}%, {m_hi:+.4f}%], "
              f"p ~ {m_p:.3f}{note}")
        share = (m_edge / edge) if edge else 0.0
        if m_p > 0.05:
            print("    Matched on volatility the edge DISAPPEARS. What looked")
            print("    like selection was the strategy choosing volatile minutes.")
        elif 0 < share < 0.6:
            print(f"    Only {share * 100:.0f}% of the raw edge survives matching;")
            print("    the rest was volatility timing, not dislocation selection.")
        else:
            print("    The edge survives matching, so it is selection and not a")
            print("    side effect of trading in volatile minutes.")
        print("-" * 72)
    lo, hi, boot_p = ci if ci is not None else bootstrap_ci(a, b)
    print(f"    edge of the rules over random: {edge:+.4f}% per trade")
    print(f"    Welch t = {t:+.2f}, p ~ {p:.3f}")
    print(f"    bootstrap 95% CI: [{lo:+.4f}%, {hi:+.4f}%], p ~ {boot_p:.3f}")
    if blocks and blocks != (0.0, 0.0, 1.0):
        b_lo, b_hi, b_p = blocks
        width_ratio = (b_hi - b_lo) / (hi - lo) if hi > lo else 1.0
        print("    BLOCK bootstrap (6h blocks, clustering respected):")
        print(f"      95% CI [{b_lo:+.4f}%, {b_hi:+.4f}%], p ~ {b_p:.3f}"
              f"   interval {width_ratio:.1f}x wider")
        if b_p >= 0.05 > boot_p:
            print("      Significance does NOT survive clustering. The trade-level")
            print("      interval was counting one market-wide move as many")
            print("      independent observations.")
        elif b_p < 0.05:
            print("      Still significant once bursts are resampled as units.")
    if (lo <= 0 <= hi) != (p > 0.05):
        print("    NOTE: the two tests disagree -- trust the bootstrap, the")
        print("          return distribution is truncated at the barriers.")
    print("-" * 72)

    if p > 0.05:
        print("  The gates are INDISTINGUISHABLE from entering at random.")
        print("  Nothing they measure predicts the next 30 minutes. Adding a")
        print("  seventh gate to the six that do nothing will not change that.")
    elif edge > 0:
        print(f"  The gates BEAT random by {edge:+.4f}% per trade (p={p:.3f}).")
        if edge < cfg.round_trip_cost_pct:
            print(f"  Still smaller than the {cfg.round_trip_cost_pct:.2f}% round-trip "
                  "cost, so it is a real")
            print("  effect that does not pay for itself. Cut turnover or cut fees.")
        else:
            print("  Large enough to clear costs. Verify on another window before")
            print("  believing it -- one test at p<0.05 is one test.")
    else:
        print(f"  The gates are WORSE than random by {edge:+.4f}% (p={p:.3f}).")
        print("  Whatever they select for, it is adverse. Inverting the signal is")
        print("  the obvious next experiment.")
    print("=" * 72 + "\n")
