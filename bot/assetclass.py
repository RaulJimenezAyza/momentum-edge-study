"""Score the strategy separately on crypto, tokenized equities and gold.

Same code, same costs, same random-entry benchmark -- only the instruments
change. Plus the one pre-registered test in this project: tokenized equities
while the underlying US market is CLOSED should show a larger effect than while
it is open, because nothing can arbitrage the token against the real share
overnight.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from .assets import CLASSES, classify, us_market_open
from .backtest import load_history
from .config import Config
from .costs import CostModel
from .exchange import BinanceREST
from .levels import levels_for
from .nullcheck import Arm, bootstrap_ci, welch
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot
from .universe import select_universe

log = logging.getLogger(__name__)


@dataclass
class Group:
    name: str
    symbols: set[str] = field(default_factory=set)
    signals: Arm = field(default_factory=lambda: Arm("sig", []))
    random: Arm = field(default_factory=lambda: Arm("rnd", []))
    gross: list[float] = field(default_factory=list)

    @property
    def edge(self) -> float:
        return self.signals.mean - self.random.mean

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0


async def _pick_universe(cfg: Config, size: int, min_volume: float) -> list[str]:
    """Everything liquid enough, so equities and metals are not crowded out.

    The normal universe is the top N by volume, and on that ranking a handful of
    tokenised equities make the cut while gold does not. Selecting by volume
    floor instead keeps every class represented.
    """
    from dataclasses import replace

    wide = replace(cfg.exchange, universe_size=size,
                   min_quote_volume_24h=min_volume,
                   exclude_symbols=[s for s in cfg.exchange.exclude_symbols
                                    if s not in ("XAUTUSDT", "PAXGUSDT")])
    async with BinanceREST(cfg.exchange.base_url) as rest:
        return await select_universe(rest, wide)


async def run_assetclass(
    cfg: Config,
    days: float = 30.0,
    offset_days: float = 0.0,
    size: int = 120,
    min_volume: float = 1_000_000.0,
    draws: int = 2,
    min_trades: int = 80,
) -> dict:
    symbols = await _pick_universe(cfg, size, min_volume)
    counts = {c: sum(1 for s in symbols if classify(s) == c) for c in CLASSES}
    log.info("universe by class: %s", counts)
    if counts["equity"] == 0:
        log.warning("no tokenized equities in the universe -- lower --min-volume")

    history, _ = await load_history(cfg, days, symbols, need_market=False,
                                    offset_days=offset_days)
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    log.info("%s", costs.summary())

    groups = {c: Group(c) for c in CLASSES}
    # The pre-registered split, equities only.
    sessions = {"equity, US open": Group("equity, US open"),
                "equity, US closed": Group("equity, US closed")}

    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    rng = random.Random(37)

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        cls = classify(symbol)
        group = groups[cls]
        group.symbols.add(symbol)

        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1
        found: list[Group] = []

        for i in range(lookback, last_index):
            series.update(candles[i])
            if isinstance(evaluate(series, cfg.strategy, costs.spread_bps(symbol)),
                          Rejection):
                continue
            out = simulate_exit(
                candles[i].close, candles[i + 1: i + 1 + horizon], cfg,
                levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if out is None:
                continue
            targets = [group]
            if cls == "equity":
                key = ("equity, US open" if us_market_open(candles[i].open_time)
                       else "equity, US closed")
                targets.append(sessions[key])
            for g in targets:
                g.signals.nets.append(out.net_pct)
                g.gross.append(out.gross_pct)
            found.extend(targets)

        for g in found:
            for _ in range(draws):
                j = rng.randrange(lookback, last_index)
                if g.name.startswith("equity, "):
                    want_open = g.name.endswith("open")
                    if us_market_open(candles[j].open_time) != want_open:
                        continue
                window = Series(symbol, lookback)
                for c in candles[max(0, j - lookback): j + 1]:
                    window.update(c)
                out = simulate_exit(
                    candles[j].close, candles[j + 1: j + 1 + horizon], cfg,
                    levels_for(cfg, window), costs.slippage_bps(symbol),
                )
                if out is not None:
                    g.random.nets.append(out.net_pct)

    _print(cfg, list(groups.values()), list(sessions.values()), days, min_trades)
    return {
        "classes": [_row(g) for g in groups.values()],
        "sessions": [_row(g) for g in sessions.values()],
    }


def _row(g: Group) -> dict:
    lo, hi, p = bootstrap_ci(g.signals, g.random)
    return {"name": g.name, "symbols": len(g.symbols), "n": g.signals.n,
            "gross": g.gross_mean, "net": g.signals.mean, "edge": g.edge,
            "ci": [lo, hi], "p": p}


def _table(title: str, groups: list[Group], min_trades: int) -> list[Group]:
    live = [g for g in groups if g.signals.n >= min_trades and g.random.n >= 30]
    print(f"\n  {title}")
    if not live:
        print("    nothing reached the minimum sample")
        return []
    print(f"    {'class':<20}{'syms':>6}{'n':>7}{'wins':>7}{'gross':>10}"
          f"{'net':>10}{'edge':>9}{'p':>8}")
    for g in live:
        _, p = welch(g.signals, g.random)
        print(f"    {g.name:<20}{len(g.symbols) or '-':>6}{g.signals.n:>7}"
              f"{g.signals.win_rate:>6.1f}%{g.gross_mean:>+9.3f}%"
              f"{g.signals.mean:>+9.3f}%{g.edge:>+8.3f}%{p:>8.3f}")
    return live


def _print(cfg, classes, sessions, days, min_trades) -> None:
    print("\n" + "=" * 86)
    print(f"  ASSET CLASSES -- {cfg.strategy.direction} {cfg.strategy.timeframe}, "
          f"{days:.0f} days")
    print("=" * 86)
    live = _table("By instrument type", classes, min_trades)
    sess = _table("Tokenized equities, by whether the real market is open "
                  "(pre-registered)", sessions, min_trades)

    print("-" * 86)
    crypto = next((g for g in live if g.name == "crypto"), None)
    others = [g for g in live if g.name != "crypto"]
    if crypto and others:
        for g in others:
            verdict = ("larger than crypto" if g.edge > crypto.edge
                       else "smaller than crypto")
            print(f"  {g.name}: edge {g.edge:+.3f}% vs crypto {crypto.edge:+.3f}% "
                  f"-- {verdict}")
    elif crypto:
        print("  Only crypto reached the sample threshold. Lower --min-volume or")
        print("  raise --days to bring the other classes in.")

    if len(sess) == 2:
        closed = next(g for g in sess if g.name.endswith("closed"))
        openg = next(g for g in sess if g.name.endswith("open"))
        gap = closed.edge - openg.edge
        print()
        if gap > 0:
            print(f"  PREDICTION HELD: the edge is {gap:+.3f}% larger while the real")
            print("  market is shut, which is what having no arbitrage anchor implies.")
        else:
            print(f"  PREDICTION FAILED: the edge is {gap:+.3f}% while the real market")
            print("  is shut -- the opposite of what the no-anchor argument predicts.")
            print("  The mechanism is wrong, so the hypothesis is dead rather than")
            print("  merely unproven.")
    print("=" * 86 + "\n")
