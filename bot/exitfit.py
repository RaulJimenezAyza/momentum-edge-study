"""Where does price actually go after a signal, and do the exits match?

Half of all exits were `max_hold`: the trade neither reached its target nor its
stop, it just ran out of clock. That is a mismatch between the exits and the
signal, not a property of the signal. The exit levels were chosen for momentum
and never revisited when the strategy became reversal.

So: measure the forward path first, prescribe second.

Part 1 is pure diagnosis with no free parameters -- the distribution of maximum
favourable and adverse excursion after a signal, and how far price reliably
travels before turning. It answers "is a 2.4% target reachable at all?" without
fitting anything.

Part 2 grids stop/target/hold over those same recorded paths, walk-forward, so
the exit fit is chosen on one fold and scored on the next. Reusing one set of
recorded paths for the whole grid is what makes it cheap; scoring the winner
only on data it was not chosen on is what makes it honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .exchange import Candle
from .levels import Levels
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

# Deliberately coarse and centred on the current settings.
#
# Targets BELOW the stop are included on purpose. The first version required
# target > stop -- an unexamined assumption that risk:reward must exceed 1:1 --
# which made the 0.6% target untestable at every stop in the grid. The
# reachability table says that is exactly where the mass is: 55% of signals
# touch +0.6% within 30m against 6% for +2.4%. A mean-reversion bounce is
# supposed to be small and frequent; forcing R:R > 1 assumes the shape of the
# answer.
STOPS = (0.4, 0.6, 0.9, 1.2, 1.8)
TARGETS = (0.4, 0.6, 1.0, 1.5, 2.4, 4.0)
HOLDS = (10, 20, 30, 60, 120)


@dataclass(slots=True)
class Path:
    symbol: str
    time_ms: int
    price: float
    future: list[Candle]
    slippage_bps: float


def _excursions(p: Path, bars: int) -> tuple[float, float]:
    """(max favourable, max adverse) over the first `bars`, in percent."""
    mfe = mae = 0.0
    for c in p.future[:bars]:
        mfe = max(mfe, (c.high / p.price - 1.0) * 100.0)
        mae = min(mae, (c.low / p.price - 1.0) * 100.0)
    return mfe, mae


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


async def run_exitfit(
    cfg: Config,
    days: float = 45.0,
    symbols: list[str] | None = None,
    offset_days: float = 14.0,
    folds: int = 3,
    min_trades: int = 100,
) -> dict:
    horizon = max(HOLDS)
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
    bar_minutes = cfg.bar_ms / 60_000
    horizon_bars = max(1, round(horizon / bar_minutes))

    paths: list[Path] = []
    for symbol, candles in history.items():
        if len(candles) < lookback + horizon_bars + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        for i in range(lookback, len(candles) - horizon_bars):
            series.update(candles[i])
            result = evaluate(series, cfg.strategy, costs.spread_bps(symbol))
            if isinstance(result, Rejection):
                continue
            paths.append(
                Path(symbol, candles[i].open_time, result.price,
                     candles[i + 1: i + 1 + horizon_bars], costs.slippage_bps(symbol))
            )
    if not paths:
        raise RuntimeError("no signals on this window")
    paths.sort(key=lambda p: p.time_ms)
    log.info("%d signal paths recorded", len(paths))

    shape = _diagnose(cfg, paths, bar_minutes)
    grid = _fit(cfg, paths, bar_minutes, folds, min_trades)
    return {"shape": shape, "fit": grid}


# --------------------------------------------------------------------------
# Part 1: what the signal actually does
# --------------------------------------------------------------------------
def _diagnose(cfg: Config, paths: list[Path], bar_minutes: float) -> dict:
    print("\n" + "=" * 84)
    print(f"  SIGNAL SHAPE -- {len(paths)} {cfg.strategy.direction} signals, "
          f"{cfg.strategy.timeframe} bars")
    print("=" * 84)
    print("  How far price travels after the signal, before anything is exited.")
    print(f"\n    {'horizon':>9}{'MFE p50':>10}{'MFE p75':>10}{'MFE p90':>10}"
          f"{'MAE p50':>10}{'MAE p25':>10}{'MAE p10':>10}")

    rows = {}
    for minutes in HOLDS:
        bars = max(1, round(minutes / bar_minutes))
        ex = [_excursions(p, bars) for p in paths]
        mfes = [e[0] for e in ex]
        maes = [e[1] for e in ex]
        rows[minutes] = {
            "mfe_p50": _pct(mfes, 0.50), "mfe_p75": _pct(mfes, 0.75),
            "mfe_p90": _pct(mfes, 0.90), "mae_p50": _pct(maes, 0.50),
            "mae_p25": _pct(maes, 0.25), "mae_p10": _pct(maes, 0.10),
        }
        r = rows[minutes]
        print(f"    {str(minutes) + 'm':>9}{r['mfe_p50']:>+9.2f}%{r['mfe_p75']:>+9.2f}%"
              f"{r['mfe_p90']:>+9.2f}%{r['mae_p50']:>+9.2f}%{r['mae_p25']:>+9.2f}%"
              f"{r['mae_p10']:>+9.2f}%")

    # Reachability: the question the max_hold pile-up actually poses.
    print("\n  Share of signals that touch a target within the hold, per target:")
    print(f"    {'target':>8}" + "".join(f"{str(m) + 'm':>9}" for m in HOLDS))
    reach = {}
    for target in TARGETS:
        cells = []
        for minutes in HOLDS:
            bars = max(1, round(minutes / bar_minutes))
            hit = sum(1 for p in paths if _excursions(p, bars)[0] >= target)
            cells.append(hit / len(paths) * 100.0)
        reach[target] = cells
        print(f"    {target:>7.1f}%" + "".join(f"{c:>8.0f}%" for c in cells))

    current_t = cfg.risk.take_profit_pct
    current_h = cfg.risk.max_hold_minutes
    nearest_t = min(TARGETS, key=lambda x: abs(x - current_t))
    nearest_h = min(HOLDS, key=lambda x: abs(x - current_h))
    share = reach[nearest_t][HOLDS.index(nearest_h)]
    print("-" * 84)
    print(f"  Current setting is {current_t:.1f}% target within {current_h}m. "
          f"Only {share:.0f}% of signals")
    print("  ever touch that level in that time -- the rest can only exit at the")
    print("  stop or the clock. That is where the max_hold pile-up comes from.")
    print("=" * 84)
    return {"excursions": rows, "reachability": {str(k): v for k, v in reach.items()}}


# --------------------------------------------------------------------------
# Part 2: fit the exits to that shape, walk-forward
# --------------------------------------------------------------------------
@dataclass
class ExitResult:
    stop: float
    target: float
    hold: int
    trades: int
    win_rate: float
    gross: float
    net: float

    @property
    def key(self) -> tuple:
        return (self.stop, self.target, self.hold)


def _score_exit(
    cfg: Config, paths: list[Path], stop: float, target: float, hold: int
) -> ExitResult:
    trial = Config(**{**cfg.__dict__})
    trial.risk = type(cfg.risk)(**cfg.risk.__dict__)
    trial.risk.stop_loss_pct = stop
    trial.risk.take_profit_pct = target
    trial.risk.max_hold_minutes = hold
    trial.risk.stop_mode = "fixed"
    # Trail scaled to the stop so it is not silently a different rule at every
    # grid point; arming at the stop distance keeps risk:reward interpretable.
    levels = Levels(stop_pct=stop, target_pct=target,
                    trail_pct=stop * 0.66, trail_activate_pct=stop)

    nets, grosses = [], []
    for p in paths:
        out = simulate_exit(p.price, p.future, trial, levels, p.slippage_bps)
        if out is not None:
            nets.append(out.net_pct)
            grosses.append(out.gross_pct)
    if not nets:
        return ExitResult(stop, target, hold, 0, 0.0, 0.0, 0.0)
    return ExitResult(
        stop, target, hold, len(nets),
        sum(1 for n in nets if n > 0) / len(nets) * 100.0,
        sum(grosses) / len(grosses), sum(nets) / len(nets),
    )


def _fit(cfg: Config, paths: list[Path], bar_minutes: float,
         folds: int, min_trades: int) -> dict:
    combos = list(product(STOPS, TARGETS, HOLDS))
    log.info("scoring %d exit settings over %d folds", len(combos), folds)

    start, end = paths[0].time_ms, paths[-1].time_ms
    width = max(1, (end - start) // folds)
    chunks: list[list[Path]] = [[] for _ in range(folds)]
    for p in paths:
        chunks[min(folds - 1, (p.time_ms - start) // width)].append(p)

    per_fold = [
        {r.key: r for r in (_score_exit(cfg, chunk, *c) for c in combos)}
        for chunk in chunks
    ]
    overall = sorted(
        (_score_exit(cfg, paths, *c) for c in combos),
        key=lambda r: r.net, reverse=True,
    )
    baseline = _score_exit(cfg, paths, cfg.risk.stop_loss_pct,
                           cfg.risk.take_profit_pct, cfg.risk.max_hold_minutes)

    carried = []
    for i in range(folds - 1):
        picks = [r for r in per_fold[i].values() if r.trades >= min_trades // folds]
        if not picks:
            continue
        winner = max(picks, key=lambda r: r.net)
        carried.append((i, winner, per_fold[i + 1][winner.key]))

    _print_fit(cfg, baseline, overall, carried, min_trades)
    return {
        "baseline": baseline.__dict__,
        "top": [r.__dict__ for r in overall[:8]],
        "walk_forward": [
            {"fold": i, "chosen": w.key, "in_sample": w.net, "next_fold": n.net}
            for i, w, n in carried
        ],
    }


def _print_fit(cfg, baseline, overall, carried, min_trades) -> None:
    def fmt(r) -> str:
        return f"sl={r.stop:.1f} tp={r.target:.1f} hold={r.hold}m"

    print("\n" + "=" * 84)
    print("  EXIT FIT -- stop / target / hold, scored on the recorded paths")
    print("=" * 84)
    print(f"\n  Current: {fmt(baseline)}")
    print(f"    {baseline.trades} trades, {baseline.win_rate:.1f}% wins, "
          f"gross {baseline.gross:+.4f}%, net {baseline.net:+.4f}%")

    eligible = [r for r in overall if r.trades >= min_trades]
    print("\n  Best in sample -- a ranking, not a result:")
    print(f"    {'net':>10}{'gross':>10}{'wins':>8}{'n':>7}  setting")
    for r in eligible[:8]:
        print(f"    {r.net:>+9.4f}%{r.gross:>+9.4f}%{r.win_rate:>7.1f}%{r.trades:>7}  {fmt(r)}")

    print("\n  Walk-forward -- chosen on one fold, scored on the NEXT:")
    if not carried:
        print("    not enough signals per fold")
    else:
        print(f"    {'fold':>5}{'in-sample':>12}{'next fold':>12}  setting")
        for i, w, n in carried:
            print(f"    {i:>5}{w.net:>+11.4f}%{n.net:>+11.4f}%  {fmt(w)}")
        oos = sum(n.net for _, _, n in carried) / len(carried)
        ins = sum(w.net for _, w, _ in carried) / len(carried)
        print(f"    {'mean':>5}{ins:>+11.4f}%{oos:>+11.4f}%")
        print("-" * 84)
        delta = oos - baseline.net
        if oos > 0:
            print(f"  Out-of-sample net is POSITIVE ({oos:+.4f}%). Confirm on the holdout.")
        elif delta > 0:
            print(f"  Refitting the exits improves out-of-sample net by {delta:+.4f}% "
                  f"per trade,")
            print(f"  from {baseline.net:+.4f}% to {oos:+.4f}%. Still negative.")
        else:
            print(f"  Refitting the exits does NOT beat the current setting out of "
                  f"sample ({oos:+.4f}% vs {baseline.net:+.4f}%).")
            print("  The exits were not the binding constraint.")
    print("=" * 84 + "\n")
