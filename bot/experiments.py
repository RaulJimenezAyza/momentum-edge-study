"""Run every structural variant of the strategy and put them in one table.

The parameter sweep answered "is some other threshold better?" (no). This
answers the bigger question the sweep could not reach, because the code was
1m-momentum-fixed-stop only: are any of the *structural* choices better?

  direction   momentum vs reversal
  timeframe   1m vs 5m vs 15m -- fewer trades amortise the fixed cost
  stops       flat percent vs scaled by each symbol's own volatility

Every variant is scored the same way and against the same benchmark: random
entries on the same symbols with the same exits. Gross expectancy is reported
next to net, because the two say completely different things -- a variant with
positive gross and negative net has an edge and a cost problem, while one with
negative gross has neither.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .exchange import Candle
from .levels import levels_for
from .nullcheck import Arm, welch
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

DIRECTIONS = ("momentum", "reversal")
TIMEFRAMES = ("1m", "5m", "15m")
STOP_MODES = ("fixed", "atr")


@dataclass
class Variant:
    direction: str
    timeframe: str
    stop_mode: str
    signals: int
    win_rate: float
    gross: float
    net: float
    stderr: float
    random_net: float
    edge: float
    p: float

    @property
    def label(self) -> str:
        return f"{self.direction:<9} {self.timeframe:>3} {self.stop_mode:>5}"


def _variant_config(cfg: Config, direction: str, timeframe: str, stop_mode: str) -> Config:
    """A copy with one structural choice changed. Never mutates the original."""
    trial = Config(**{**cfg.__dict__})
    trial.strategy = replace(cfg.strategy, direction=direction, timeframe=timeframe)
    trial.risk = replace(cfg.risk, stop_mode=stop_mode)
    return trial


def score_variant(
    cfg: Config,
    history: dict[str, list[Candle]],
    seed: int,
    draws: int,
    costs: CostModel,
) -> tuple[Arm, Arm, list[float]]:
    """Returns (signal arm, random arm, gross returns of the signal arm)."""
    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    rng = random.Random(seed)
    real: list[float] = []
    gross: list[float] = []
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
            if isinstance(evaluate(series, cfg.strategy, costs.spread_bps(symbol)), Rejection):
                continue
            outcome = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon],
                cfg, levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if outcome is not None:
                real.append(outcome.net_pct)
                gross.append(outcome.gross_pct)
                found += 1

        for _ in range(found * draws):
            j = rng.randrange(lookback, last_index)
            window = Series(symbol, lookback)
            for c in candles[max(0, j - lookback): j + 1]:
                window.update(c)
            outcome = simulate_exit(
                candles[j].close, candles[j + 1: j + 1 + horizon],
                cfg, levels_for(cfg, window), costs.slippage_bps(symbol),
            )
            if outcome is not None:
                fake.append(outcome.net_pct)

    return Arm("signals", real), Arm("random", fake), gross


async def run_experiments(
    cfg: Config,
    days: float = 30.0,
    symbols: list[str] | None = None,
    draws: int = 2,
    min_signals: int = 30,
    offset_days: float = 0.0,
) -> list[dict]:
    results: list[Variant] = []

    for timeframe in TIMEFRAMES:
        base = _variant_config(cfg, cfg.strategy.direction, timeframe, cfg.risk.stop_mode)
        log.info("loading %s history...", timeframe)
        history, _ = await load_history(
            base, days, symbols, need_market=False, offset_days=offset_days
        )

        # Spreads are estimated per timeframe: the same book measured on 15m
        # bars looks tighter than on 1m, and the trade pays the real one.
        costs = CostModel(
            cfg,
            estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
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

        for direction in DIRECTIONS:
            for stop_mode in STOP_MODES:
                trial = _variant_config(cfg, direction, timeframe, stop_mode)
                try:
                    trial.validate()
                except (ValueError, RuntimeError) as exc:
                    log.warning("skipping %s/%s/%s: %s",
                                direction, timeframe, stop_mode, exc)
                    continue
                trial_costs = CostModel(trial, costs.spreads_bps,
                                        costs.quoted_bps)
                arm, rnd, gross = score_variant(
                    trial, history, seed=7, draws=draws, costs=trial_costs
                )
                if arm.n < min_signals:
                    log.info("  %s %s %s: only %d signals, skipped",
                             direction, timeframe, stop_mode, arm.n)
                    continue
                _, p = welch(arm, rnd)
                results.append(
                    Variant(
                        direction=direction, timeframe=timeframe, stop_mode=stop_mode,
                        signals=arm.n, win_rate=arm.win_rate,
                        gross=sum(gross) / len(gross) if gross else 0.0,
                        net=arm.mean, stderr=arm.stderr,
                        random_net=rnd.mean, edge=arm.mean - rnd.mean, p=p,
                    )
                )
                log.info("  %s %s %s: %d signals, net %+.4f%%",
                         direction, timeframe, stop_mode, arm.n, arm.mean)

    _print(cfg, results, days, offset_days)
    return [v.__dict__ for v in results]


def _print(cfg: Config, results: list[Variant], days: float, offset: float = 0.0) -> None:
    cost_note = (
        "per-symbol estimated spreads"
        if cfg.fees.slippage_mode == "estimated"
        else f"flat {cfg.round_trip_cost_pct:.2f}% round trip"
    )
    print("\n" + "=" * 92)
    window = f"{days:.0f} days" + (f" ending {offset:.0f}d ago" if offset else "")
    print(f"  STRUCTURAL EXPERIMENTS -- {len(results)} variants, {window}, {cost_note}")
    print("=" * 92)
    if not results:
        print("  no variant produced enough signals to score")
        print("=" * 92 + "\n")
        return

    print(f"    {'variant':<21}{'signals':>8}{'wins':>7}{'gross':>10}{'net':>10}"
          f"{'+/- se':>8}{'random':>10}{'edge':>9}{'p':>7}")
    for v in sorted(results, key=lambda r: r.net, reverse=True):
        print(f"    {v.label:<21}{v.signals:>8}{v.win_rate:>6.1f}%"
              f"{v.gross:>+9.3f}%{v.net:>+9.3f}%{v.stderr:>7.3f}"
              f"{v.random_net:>+9.3f}%{v.edge:>+8.3f}%{v.p:>7.3f}")

    print("-" * 92)
    best_net = max(results, key=lambda r: r.net)
    best_gross = max(results, key=lambda r: r.gross)
    significant = [r for r in results if r.p < 0.05 and r.edge > 0]

    print(f"  Best net:   {best_net.label.strip()}  {best_net.net:+.3f}% per trade")
    print(f"  Best gross: {best_gross.label.strip()}  {best_gross.gross:+.3f}% "
          f"per trade (before costs)")

    if best_gross.gross <= 0:
        print("\n  NO variant has a positive gross edge. The rules are not being")
        print("  beaten by fees -- they have nothing for fees to eat. Changing")
        print("  timeframe, direction or stop style does not create signal.")
    elif best_net.net <= 0:
        print(f"\n  Gross edge exists ({best_gross.gross:+.3f}%) but no variant clears "
              f"the {cfg.round_trip_cost_pct:.2f}% cost.")
        print("  Cost reduction is the lever, not another entry rule:")
        print("    - fees.bnb_discount: true          -> 25% off the fee legs")
        print("    - longer timeframe / longer hold   -> fewer round trips")
    else:
        print(f"\n  {best_net.label.strip()} is net positive. Confirm on a different "
              "window before believing it.")

    if significant:
        print("\n  Beat random at p<0.05: "
              + ", ".join(v.label.strip() for v in significant))
    else:
        print("\n  NOT ONE variant beats random entries at p<0.05.")
    print("=" * 92 + "\n")
