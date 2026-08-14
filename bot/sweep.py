"""Walk-forward parameter search.

The 7-day backtest said the strategy loses to fees. The obvious next question
is whether some other setting of the same rules does not. This answers it with
evidence instead of taste.

The honest part is the walk-forward: the grid is scored on fold i, the winner
is carried to fold i+1, and what gets reported is how the winner did on data it
was NOT chosen on. Picking the best row of an in-sample table and quoting its
number is the single most common way a backtest lies, so this prints both and
puts them side by side.

Scope: this measures PER-TRADE expectancy, not portfolio return. It ignores
max_positions and shared equity, which is the right call for comparing entry
rules -- capital constraints would make two configs differ for reasons that
have nothing to do with signal quality. Run `backtest` on the winner to get the
portfolio number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .exchange import Candle
from .levels import levels_for
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

MINUTE_MS = 60_000

# Deliberately coarse. A fine grid over a noisy objective finds noise.
GRID = {
    "min_return_zscore": (2.5, 3.5, 4.5),
    "min_volume_mult": (3.0, 5.0, 8.0),
    "take_profit_pct": (2.4, 4.0, 6.0),
    "stop_loss_pct": (1.2, 2.0),
    "max_hold_minutes": (30, 60),
}


@dataclass(slots=True)
class Candidate:
    """An entry the loosest config would take, plus its forward window.

    `atr_pct` and `slippage_bps` are captured at signal time because the exit
    levels and the fill cost are per-symbol. Recomputing them from the trial
    config instead would silently score a different strategy than the bot runs.
    """

    symbol: str
    time_ms: int
    price: float
    ret_pct: float
    zscore: float
    volume_mult: float
    extension_pct: float
    future: list[Candle]
    atr_pct: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class Result:
    params: dict
    trades: int
    win_rate: float
    expectancy: float
    total: float

    @property
    def key(self) -> tuple:
        return tuple(sorted(self.params.items()))


def _configs() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, combo)) for combo in product(*(GRID[k] for k in keys))]


def build_candidates(cfg: Config, history: dict[str, list[Candle]]) -> list[Candidate]:
    """One pass over history at the loosest grid setting.

    Every config in the grid is a subset of this pool, so the expensive replay
    happens once instead of 108 times.
    """
    loosest = Config(**{**cfg.__dict__})
    loosest.strategy = type(cfg.strategy)(**cfg.strategy.__dict__)
    loosest.strategy.min_return_zscore = min(GRID["min_return_zscore"])
    loosest.strategy.min_volume_mult = min(GRID["min_volume_mult"])

    lookback = cfg.strategy.lookback_bars
    # Bars, not minutes: the same 60-minute hold is 60 bars on 1m and 4 on 15m.
    bar_minutes = cfg.bar_ms / 60_000
    horizon = max(1, round(max(GRID["max_hold_minutes"]) / bar_minutes))
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
    out: list[Candidate] = []

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        for i in range(lookback, len(candles) - horizon):
            series.update(candles[i])
            result = evaluate(
                series, loosest.strategy, costs.spread_bps(symbol)
            )
            if isinstance(result, Rejection):
                continue
            out.append(
                Candidate(
                    symbol=symbol,
                    time_ms=candles[i].open_time,
                    price=result.price,
                    ret_pct=result.ret_pct,
                    zscore=result.zscore,
                    volume_mult=result.volume_mult,
                    extension_pct=result.extension_pct,
                    future=candles[i + 1 : i + 1 + horizon],
                    atr_pct=series.atr_pct(cfg.risk.atr_bars),
                    slippage_bps=costs.slippage_bps(symbol),
                )
            )

    out.sort(key=lambda c: c.time_ms)
    return out


def score(cfg: Config, candidates: list[Candidate], params: dict) -> Result:
    """Expectancy of one config over one set of candidates."""
    trial = Config(**{**cfg.__dict__})
    trial.strategy = type(cfg.strategy)(**cfg.strategy.__dict__)
    trial.risk = type(cfg.risk)(**cfg.risk.__dict__)
    trial.strategy.min_return_zscore = params["min_return_zscore"]
    trial.strategy.min_volume_mult = params["min_volume_mult"]
    trial.risk.take_profit_pct = params["take_profit_pct"]
    trial.risk.stop_loss_pct = params["stop_loss_pct"]
    trial.risk.max_hold_minutes = params["max_hold_minutes"]

    nets = []
    for c in candidates:
        if (
            c.zscore < params["min_return_zscore"]
            or c.volume_mult < params["min_volume_mult"]
        ):
            continue
        outcome = simulate_exit(
            c.price, c.future, trial,
            levels_for(trial, atr_pct=c.atr_pct), c.slippage_bps,
        )
        if outcome is not None:
            nets.append(outcome.net_pct)

    if not nets:
        return Result(params, 0, 0.0, 0.0, 0.0)
    wins = sum(1 for n in nets if n > 0)
    return Result(
        params=params,
        trades=len(nets),
        win_rate=wins / len(nets) * 100.0,
        expectancy=sum(nets) / len(nets),
        total=sum(nets),
    )


def _folds(candidates: list[Candidate], n: int) -> list[list[Candidate]]:
    if not candidates:
        return []
    start, end = candidates[0].time_ms, candidates[-1].time_ms
    width = max(1, (end - start) // n)
    out: list[list[Candidate]] = [[] for _ in range(n)]
    for c in candidates:
        idx = min(n - 1, (c.time_ms - start) // width)
        out[idx].append(c)
    return out


async def run_sweep(
    cfg: Config,
    days: float = 30.0,
    symbols: list[str] | None = None,
    folds: int = 4,
    min_trades: int = 40,
) -> dict:
    history, _ = await load_history(cfg, days, symbols, need_market=False)
    log.info("building candidate pool at the loosest grid setting...")
    candidates = build_candidates(cfg, history)
    if not candidates:
        raise RuntimeError("no candidates -- widen the grid or the window")
    log.info("%d candidates across %d symbols", len(candidates), len(history))

    configs = _configs()
    log.info("scoring %d configs over %d folds", len(configs), folds)

    chunks = _folds(candidates, folds)
    per_fold = [
        sorted(
            (score(cfg, chunk, p) for p in configs),
            key=lambda r: r.expectancy,
            reverse=True,
        )
        for chunk in chunks
    ]

    # Overall (in-sample) ranking, for reference only.
    overall = sorted(
        (score(cfg, candidates, p) for p in configs),
        key=lambda r: r.expectancy,
        reverse=True,
    )
    eligible = [r for r in overall if r.trades >= min_trades]

    # The honest test: pick on fold i, judge on fold i+1.
    carried: list[tuple[int, Result, Result]] = []
    for i in range(len(per_fold) - 1):
        picks = [r for r in per_fold[i] if r.trades >= min_trades // folds]
        if not picks:
            continue
        winner = picks[0]
        lookup = {r.key: r for r in per_fold[i + 1]}
        carried.append((i, winner, lookup[winner.key]))

    baseline = score(cfg, candidates, {
        "min_return_zscore": cfg.strategy.min_return_zscore,
        "min_volume_mult": cfg.strategy.min_volume_mult,
        "take_profit_pct": cfg.risk.take_profit_pct,
        "stop_loss_pct": cfg.risk.stop_loss_pct,
        "max_hold_minutes": cfg.risk.max_hold_minutes,
    })

    _print(cfg, baseline, eligible, carried, len(candidates), days, min_trades)
    return {
        "baseline": baseline.__dict__,
        "top": [r.__dict__ for r in eligible[:10]],
        "walk_forward": [
            {"fold": i, "chosen": w.params,
             "in_sample": w.expectancy, "next_fold": n.expectancy,
             "next_fold_trades": n.trades}
            for i, w, n in carried
        ],
    }


def _fmt(params: dict) -> str:
    return (f"z>={params['min_return_zscore']:.1f} vol>={params['min_volume_mult']:.0f}x "
            f"tp={params['take_profit_pct']:.1f} sl={params['stop_loss_pct']:.1f} "
            f"hold={params['max_hold_minutes']}m")


def _print(cfg, baseline, eligible, carried, n_candidates, days, min_trades) -> None:
    print("\n" + "=" * 78)
    print(f"  PARAMETER SWEEP -- {n_candidates} candidates over {days:.0f} days")
    print(f"  round-trip cost {cfg.round_trip_cost_pct:.2f}% per trade")
    print("=" * 78)

    print(f"\n  Current config: {_fmt(baseline.params)}")
    print(f"    {baseline.trades} trades, {baseline.win_rate:.1f}% wins, "
          f"expectancy {baseline.expectancy:+.4f}%")

    print(f"\n  Best in sample (>= {min_trades} trades) -- NOT a result, a ranking:")
    print(f"    {'expectancy':>11} {'trades':>7} {'wins':>7}  config")
    for r in eligible[:8]:
        print(f"    {r.expectancy:>+10.4f}% {r.trades:>7} {r.win_rate:>6.1f}%  {_fmt(r.params)}")

    print("\n  Walk-forward -- chosen on one fold, scored on the NEXT:")
    if not carried:
        print("    not enough data per fold to carry a winner forward")
    else:
        print(f"    {'fold':>5} {'in-sample':>11} {'next fold':>11} {'trades':>7}  config")
        for i, winner, nxt in carried:
            print(f"    {i:>5} {winner.expectancy:>+10.4f}% {nxt.expectancy:>+10.4f}% "
                  f"{nxt.trades:>7}  {_fmt(winner.params)}")
        oos = [n.expectancy for _, _, n in carried]
        ins = [w.expectancy for _, w, _ in carried]
        mean_oos = sum(oos) / len(oos)
        mean_ins = sum(ins) / len(ins)
        print(f"    {'mean':>5} {mean_ins:>+10.4f}% {mean_oos:>+10.4f}%")
        print("-" * 78)
        drop = mean_ins - mean_oos
        if mean_oos > 0:
            print(f"  Out-of-sample expectancy is POSITIVE ({mean_oos:+.4f}%). "
                  f"Selection cost {drop:.4f}%.")
            print("  Confirm on a different window before believing it.")
        else:
            print(f"  Out-of-sample expectancy is NEGATIVE ({mean_oos:+.4f}%), "
                  f"after looking {mean_ins:+.4f}% in sample.")
            print(f"  The {drop:.4f}% gap is what parameter selection bought you: nothing.")
            print("  No setting of these rules survives the fee load on this data.")
    print("=" * 78 + "\n")
