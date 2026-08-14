"""Does the edge hold up across periods that do not overlap?

This exists because of a mistake. Five tests were run and reported as five
replications; they were overlapping slices of the same sixty days, and the
"holdout" was contained inside the window that supposedly confirmed it. The
tooling made overlapping windows the easy thing to type and non-overlapping
ones a manual chore, so the easy thing is what happened.

The fix is to make the correct test the one-word command. `replicate` carves
N consecutive, strictly non-overlapping windows going backwards in time and
scores the same strategy on each. A real effect shows up repeatedly with the
same sign. A window-specific artefact shows up once and vanishes -- which is
exactly what happened when this was finally run.

It also pools the windows properly. Inverse-variance weighting across
independent periods is a meaningful combined estimate; running one long window
that swallows all the others is not, however large its sample looks.
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
from .nullcheck import Arm, bootstrap_ci, welch
from .pointintime import listing_report, point_in_time_universe
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)


@dataclass
class Window:
    label: str
    offset: float
    signals: Arm
    random: Arm

    @property
    def edge(self) -> float:
        return self.signals.mean - self.random.mean

    @property
    def stderr(self) -> float:
        return math.sqrt(self.signals.stderr ** 2 + self.random.stderr ** 2)


async def _score_window(
    cfg: Config, days: float, offset: float, symbols, draws: int, seed: int,
    point_in_time: bool = False, universe_size: int = 0,
) -> Window:
    # Pull extra history before the window so the universe can be ranked on
    # information that existed then rather than on today's volume table.
    from .pointintime import RANK_DAYS

    pad = RANK_DAYS if point_in_time else 0
    history, _ = await load_history(
        cfg, days + pad, symbols, need_market=False, offset_days=offset
    )
    report = listing_report(history, days + pad)
    if report and report["incomplete"]:
        log.info(
            "  %d/%d symbols lack full history for this window (%.0f%% of candles)"
            " -- they were listed after it started",
            report["incomplete"], report["symbols"], report["candle_share"] * 100,
        )
    if point_in_time:
        newest = max(c[-1].open_time for c in history.values())
        window_start = newest - int(days * 86_400_000)
        keep = set(point_in_time_universe(
            history, window_start, universe_size or cfg.exchange.universe_size,
            cfg.exchange.min_quote_volume_24h,
        ))
        history = {s: [c for c in cs if c.open_time >= window_start]
                   for s, cs in history.items() if s in keep}
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    rng = random.Random(seed)
    real: list[float] = []
    fake: list[float] = []

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1
        found = 0
        for i in range(lookback, last_index):
            series.update(candles[i])
            if isinstance(evaluate(series, cfg.strategy, costs.spread_bps(symbol)),
                          Rejection):
                continue
            out = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon], cfg,
                levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if out is not None:
                real.append(out.net_pct)
                found += 1
        for _ in range(found * draws):
            j = rng.randrange(lookback, last_index)
            window = Series(symbol, lookback)
            for c in candles[max(0, j - lookback): j + 1]:
                window.update(c)
            out = simulate_exit(
                candles[j].close, candles[j + 1: j + 1 + horizon], cfg,
                levels_for(cfg, window), costs.slippage_bps(symbol),
            )
            if out is not None:
                fake.append(out.net_pct)

    label = f"{offset:.0f}-{offset + days:.0f}d ago"
    return Window(label, offset, Arm("sig", real), Arm("rnd", fake))


async def run_replicate(
    cfg: Config,
    days: float = 30.0,
    windows: int = 5,
    symbols: list[str] | None = None,
    draws: int = 2,
    min_trades: int = 200,
    point_in_time: bool = False,
) -> dict:
    results: list[Window] = []
    for k in range(windows):
        offset = k * days           # strictly non-overlapping, by construction
        log.info("window %d/%d: %.0f-%.0f days ago",
                 k + 1, windows, offset, offset + days)
        try:
            w = await _score_window(cfg, days, offset, symbols, draws,
                                    seed=100 + k, point_in_time=point_in_time)
        except Exception as exc:  # noqa: BLE001 - an old window may be unavailable
            log.warning("window %.0fd ago failed: %s", offset, exc)
            continue
        if w.signals.n < min_trades:
            log.info("  only %d signals, skipped", w.signals.n)
            continue
        results.append(w)
        log.info("  %d signals, edge %+.4f%%", w.signals.n, w.edge)

    if not results:
        raise RuntimeError("no window produced enough signals")
    _print(cfg, results, days, point_in_time)
    b_mean, b_se, b_p, df = _between_window(results)
    record(cfg, "between_window", {
        "mean_edge": b_mean, "se": b_se, "p": b_p, "df": df,
        "windows": len(results), "window_days": days,
        "n": sum(w.signals.n for w in results),
        "positive": sum(1 for w in results if w.edge > 0),
        "point_in_time": point_in_time,
    })
    return {"windows": [
        {"label": w.label, "n": w.signals.n, "net": w.signals.mean,
         "random": w.random.mean, "edge": w.edge, "stderr": w.stderr,
         "p": welch(w.signals, w.random)[1]}
        for w in results
    ], "between_window": {"mean_edge": b_mean, "se": b_se, "p": b_p, "df": df}}


def student_t_p(t: float, df: int, steps: int = 4000) -> float:
    """Two-sided p from Student's t. Integrated numerically, no SciPy.

    The normal approximation is fine at n>1000 and badly wrong at n=4. With
    three degrees of freedom the 5% critical value is 3.18, not 1.96, so a
    t of 2.6 is p about 0.08 -- and erfc would have reported 0.009. Printing
    the smaller number would be the same class of error this whole document is
    a record of hunting.
    """
    import math as m

    if df < 1:
        return 1.0
    t = abs(t)
    # Density: Gamma((df+1)/2) / (sqrt(df*pi) Gamma(df/2)) * (1+t^2/df)^-((df+1)/2)
    log_norm = (m.lgamma((df + 1) / 2) - m.lgamma(df / 2)
                - 0.5 * m.log(df * m.pi))

    def pdf(x: float) -> float:
        return m.exp(log_norm - (df + 1) / 2 * m.log1p(x * x / df))

    # Simpson from t out to a cut-off, plus the mass beyond it. Widening the
    # cut-off alone does not fix df=1: the Cauchy tail falls off like 1/x^2 and
    # still holds 2e-4 past x=3000, which moved the 5% critical value by 0.05.
    # Past the cut-off the density is x^-(df+1) to good approximation, and that
    # integrates exactly, so the remainder is added rather than outrun.
    upper = max(t, 1.0) + 60.0
    n = steps if steps % 2 == 0 else steps + 1
    h = (upper - t) / n
    total = pdf(t) + pdf(upper)
    for i in range(1, n):
        total += pdf(t + i * h) * (4 if i % 2 else 2)
    tail = total * h / 3 + pdf(upper) * upper / df
    return min(1.0, max(0.0, 2.0 * tail))


def t_critical(df: int, alpha: float = 0.05) -> float:
    """The two-sided multiplier for a confidence interval on `df`.

    Fixing the p-value and leaving 1.96 in the interval would have been the
    same mistake wearing a different hat: at df=3 the multiplier is 3.18, so a
    normal interval is 38% too narrow exactly where the sample is smallest and
    the width matters most. Found by bisection on student_t_p, which is
    monotone in t, so no table is needed.
    """
    if df < 1:
        return float("inf")
    lo, hi = 0.0, 400.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if student_t_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _between_window(results: list[Window]) -> tuple[float, float, float, int]:
    """Treat each WINDOW as one observation. Mean, stderr, p, degrees of freedom.

    Inverse-variance pooling weights by the standard error measured *inside*
    each window, and those are computed as though every trade were an
    independent draw. They are not: one market-wide dump fires the entry rule
    on twenty symbols at once and produces twenty highly correlated outcomes.
    A block bootstrap on a single window widened its interval by 1.8x and took
    p from 0.011 to 0.145, so the within-window errors are understated by
    roughly that factor and the pooled p inherits the problem.

    This makes no independence assumption inside a window at all. Each window
    contributes one number, and the spread ACROSS windows supplies the error.
    It throws away information -- four observations is a tiny sample and the
    interval is wide -- but nothing it claims depends on trades being
    independent, which they visibly are not.
    """
    import math as m

    edges = [w.edge for w in results]
    n = len(edges)
    if n < 2:
        return (edges[0] if edges else 0.0), 0.0, 1.0, 0
    mean = sum(edges) / n
    var = sum((e - mean) ** 2 for e in edges) / (n - 1)
    se = m.sqrt(var / n)
    if se <= 0:
        return mean, 0.0, 0.0, n - 1
    t = mean / se
    return mean, se, student_t_p(t, n - 1), n - 1


def _pool(results: list[Window]) -> tuple[float, float, float]:
    """Inverse-variance weighted mean edge, its standard error, and a p-value."""
    weights = [1.0 / (w.stderr ** 2) for w in results if w.stderr > 0]
    edges = [w.edge for w in results if w.stderr > 0]
    if not weights:
        return 0.0, 0.0, 1.0
    total = sum(weights)
    mean = sum(e * wt for e, wt in zip(edges, weights)) / total
    se = math.sqrt(1.0 / total)
    p = math.erfc(abs(mean / se) / math.sqrt(2.0)) if se > 0 else 1.0
    return mean, se, p


def _print(cfg: Config, results: list[Window], days: float,
           point_in_time: bool = False) -> None:
    print("\n" + "=" * 84)
    print(f"  REPLICATION -- {len(results)} NON-OVERLAPPING {days:.0f}-day windows, "
          f"{cfg.strategy.direction} {cfg.strategy.timeframe}")
    if point_in_time:
        print("  universe ranked on volume from BEFORE each window "
              "(delisting survivorship remains)")
    print("=" * 84)
    print(f"    {'window':<14}{'n':>7}{'net':>10}{'random':>10}{'edge':>10}"
          f"{'p':>8}{'95% CI':>22}")
    for w in results:
        _, p = welch(w.signals, w.random)
        lo, hi, _ = bootstrap_ci(w.signals, w.random, samples=1000)
        print(f"    {w.label:<14}{w.signals.n:>7}{w.signals.mean:>+9.3f}%"
              f"{w.random.mean:>+9.3f}%{w.edge:>+9.3f}%{p:>8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}")

    positive = sum(1 for w in results if w.edge > 0)
    significant = sum(1 for w in results if welch(w.signals, w.random)[1] < 0.05
                      and w.edge > 0)
    mean, se, p = _pool(results)

    print("-" * 84)
    print(f"    {'pooled':<14}{sum(w.signals.n for w in results):>7}"
          f"{'':>10}{'':>10}{mean:>+9.3f}%{p:>8.3f}"
          f"{f'[{mean - 1.96 * se:+.3f}, {mean + 1.96 * se:+.3f}]':>22}")
    print(f"    (inverse-variance weighted, assumes trades are independent)")

    b_mean, b_se, b_p, df = _between_window(results)
    if df >= 1:
        # Not 1.96. At df=3 that would be a 38% understatement of the width.
        _tc = t_critical(df)
        print(f"    {'between-window':<14}{'':>7}{'':>10}{'':>10}{b_mean:>+9.3f}%"
              f"{b_p:>8.3f}"
              f"{f'[{b_mean - _tc * b_se:+.3f}, {b_mean + _tc * b_se:+.3f}]':>22}")
        print(f"    (each window one observation, Student t on df={df} -- "
              f"assumes nothing about trades)")
    print("-" * 84)
    print(f"  Positive in {positive}/{len(results)} windows, "
          f"significant in {significant}/{len(results)}.")

    if df >= 1 and b_p >= 0.05:
        print(f"\n  Treating each window as a single observation, the effect is NOT")
        print(f"  significant (p = {b_p:.3f} on {df} degrees of freedom). The")
        print("  trade-level p-values above assume that twenty signals fired by")
        print("  one market-wide move are twenty independent facts. They are not.")
        print("  With only a handful of windows this test is weak, but it is the")
        print("  one that does not depend on an assumption known to be false.")
        print("=" * 84 + "\n")
        return

    if significant >= max(2, len(results) - 1):
        print("\n  The effect shows up again and again on data it was never chosen")
        print("  on. That is what replication looks like.")
    elif significant == 0:
        print("\n  The effect does NOT replicate. Whatever was measured on the")
        print("  window it was found in did not survive contact with periods it")
        print("  was not selected on -- the ordinary fate of a backtest finding.")
    else:
        print(f"\n  It replicates in {significant} of {len(results)} windows and not")
        print("  the rest. That is a regime-dependent effect at best and noise at")
        print("  worst, and the two are hard to tell apart without a mechanism")
        print("  predicted in advance.")
    print("=" * 84 + "\n")
