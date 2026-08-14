#!/usr/bin/env python
"""CLI entry point.

    python run.py universe                 list the symbols that would be watched
    python run.py live                     run the bot + web viewer (paper by default)
    python run.py backtest --days 7        replay history through the same logic
    python run.py scan                     one-shot: signals on the last closed bar
    python run.py train --days 45          fit the ML filter and report on it
    python run.py model                    show the saved model's metrics
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from bot.backtest import run_backtest
from bot.config import load_config
from bot.engine import Engine
from bot.exchange import BinanceREST
from bot.logbook import setup_logging
from bot.series import Series
from bot.signals import Rejection, evaluate
from bot.universe import select_universe

log = logging.getLogger("run")


async def cmd_universe(cfg, args) -> None:
    async with BinanceREST(cfg.exchange.base_url) as rest:
        symbols = await select_universe(rest, cfg.exchange)
    for i, s in enumerate(symbols, 1):
        print(f"{i:3d}. {s}")


async def cmd_scan(cfg, args) -> None:
    """Evaluate the current bar for every symbol without trading anything."""
    async with BinanceREST(cfg.exchange.base_url) as rest:
        symbols = args.symbols or await select_universe(rest, cfg.exchange)
        spreads: dict[str, float] = {}
        for t in await rest.book_tickers():
            if t["symbol"] in set(symbols):
                bid, ask = float(t["bidPrice"]), float(t["askPrice"])
                if bid > 0 and ask > 0:
                    spreads[t["symbol"]] = (ask - bid) / ((ask + bid) / 2) * 10_000

        sem = asyncio.Semaphore(8)
        hits, rejects = [], {}

        async def check(symbol: str) -> None:
            async with sem:
                candles = await rest.klines(
                    symbol, cfg.strategy.timeframe, cfg.strategy.lookback_bars
                )
            series = Series(symbol, cfg.strategy.lookback_bars)
            for c in candles[:-1]:
                series.update(c)
            result = evaluate(series, cfg.strategy, spreads.get(symbol, 0.0))
            if isinstance(result, Rejection):
                rejects[result.reason] = rejects.get(result.reason, 0) + 1
            else:
                hits.append(result)

        await asyncio.gather(*(check(s) for s in symbols))

    print(f"\nScanned {len(symbols)} symbols.")
    if hits:
        print(f"{'SYMBOL':<12}{'PRICE':>14}{'RET%':>8}{'Z':>7}{'VOLx':>7}{'EXT%':>7}")
        for s in sorted(hits, key=lambda h: h.zscore, reverse=True):
            print(
                f"{s.symbol:<12}{s.price:>14.8f}{s.ret_pct:>8.2f}"
                f"{s.zscore:>7.1f}{s.volume_mult:>7.1f}{s.extension_pct:>7.2f}"
            )
    else:
        print("No signals on the last closed bar.")
    print("\nFiltered: " + ", ".join(f"{k}={v}" for k, v in sorted(
        rejects.items(), key=lambda kv: -kv[1])))


async def cmd_live(cfg, args) -> None:
    if cfg.mode == "live":
        log.warning("!!! LIVE MODE -- real orders with real money !!!")
    if cfg.dashboard.enabled:
        print(f"\n  viewer: http://{cfg.dashboard.host}:{cfg.dashboard.port}\n")
    await Engine(cfg, args.symbols).run()


async def cmd_backtest(cfg, args) -> None:
    await run_backtest(cfg, days=args.days, symbols=args.symbols)


async def cmd_train(cfg, args) -> None:
    from bot.trainer import run_training

    await run_training(
        cfg, days=args.days, symbols=args.symbols, threshold=args.threshold,
        offset_days=args.offset,
    )


async def cmd_model(cfg, args) -> None:
    from bot.trainer import describe_model

    describe_model(cfg)


async def cmd_sweep(cfg, args) -> None:
    from bot.sweep import run_sweep

    await run_sweep(cfg, days=args.days, symbols=args.symbols, folds=args.folds)


async def cmd_experiments(cfg, args) -> None:
    from bot.experiments import run_experiments

    await run_experiments(cfg, days=args.days, symbols=args.symbols,
                          offset_days=args.offset)


async def cmd_maker(cfg, args) -> None:
    from bot.maker import run_maker

    await run_maker(cfg, days=args.days, offset_days=args.offset,
                    symbols=args.symbols, window=args.window,
                    maker_bps=args.maker_bps)


async def cmd_equities(cfg, args) -> None:
    from bot.equities import run_equities

    await run_equities(cfg, days=args.days, interval=args.interval,
                       symbols=args.symbols, hold_bars=args.hold_bars)


async def cmd_eqwindows(cfg, args) -> None:
    from bot.equities import run_equity_windows

    await run_equity_windows(cfg, windows=args.windows, days=args.days,
                             interval=args.interval, symbols=args.symbols,
                             hold_bars=args.hold_bars)


async def cmd_ladder(cfg, args) -> None:
    from bot.ladder import run_ladder

    # A negative rate is the escape hatch for "use the historical era table",
    # which is a different kind of thing from a number and cannot be one.
    tax_rate = None if args.tax_rate < 0 else args.tax_rate

    await run_ladder(cfg, window_years=args.window_years, windows=args.windows,
                     start_equity=args.start_equity, capital=args.capital,
                     symbols=args.symbols,
                     measured_yields=args.measured_yields,
                     dividend_source=args.dividend_source,
                     rearm=args.rearm, trigger=args.trigger,
                     tax_rate=tax_rate, liquidate=args.liquidate,
                     preset=args.preset)


async def cmd_longterm(cfg, args) -> None:
    from bot.longterm import run_longterm

    tax_rate = None if args.tax_rate < 0 else args.tax_rate
    await run_longterm(cfg, window_years=args.window_years,
                       windows=args.windows, capital=args.capital,
                       symbols=args.symbols, band=args.band,
                       tax_rate=tax_rate, liquidate=args.liquidate,
                       measured_yields=args.measured_yields)


async def cmd_allocate(cfg, args) -> None:
    from bot.allocate import run_allocate

    tax_rate = None if args.tax_rate < 0 else args.tax_rate
    await run_allocate(cfg, tax_rate=tax_rate, liquidate=args.liquidate,
                       band=args.band, max_leverage=args.max_leverage,
                       seed=args.seed)


async def cmd_trendpanel(cfg, args) -> None:
    from bot.trendpanel import run_trendpanel

    await run_trendpanel(cfg, cost_bps=args.cost_bps,
                         target_vol=args.target_vol,
                         max_gross=args.max_gross, seed=args.seed)


async def cmd_basketcheck(cfg, args) -> None:
    from bot.basket import run_basketcheck

    await run_basketcheck(cfg)


async def cmd_vendorcheck(cfg, args) -> None:
    from bot.vendors import run_vendorcheck

    await run_vendorcheck(cfg, symbol=args.symbol, min_bars=args.min_bars)


async def cmd_assetclass(cfg, args) -> None:
    from bot.assetclass import run_assetclass

    await run_assetclass(cfg, days=args.days, offset_days=args.offset,
                         size=args.size, min_volume=args.min_volume)


async def cmd_reconcile(cfg, args) -> None:
    from bot.reconcile import run_reconcile

    await run_reconcile(cfg)


async def cmd_replicate(cfg, args) -> None:
    from bot.replicate import run_replicate

    await run_replicate(cfg, days=args.days, windows=args.windows,
                        symbols=args.symbols, point_in_time=args.point_in_time)


async def cmd_regime(cfg, args) -> None:
    from bot.regime import run_regime

    await run_regime(cfg, days=args.days, symbols=args.symbols,
                     offset_days=args.offset)


async def cmd_spreadcheck(cfg, args) -> None:
    from bot.spreadcheck import run_spreadcheck

    await run_spreadcheck(cfg, days=args.days, symbols=args.symbols)


async def cmd_exitfit(cfg, args) -> None:
    from bot.exitfit import run_exitfit

    await run_exitfit(cfg, days=args.days, symbols=args.symbols,
                      offset_days=args.offset, folds=args.folds)


async def cmd_markets(cfg, args) -> None:
    from bot.markets import run_markets

    await run_markets(cfg, days=args.days, offset_days=args.offset,
                      universe_size=args.size, min_volume=args.min_volume)


async def cmd_edgecurve(cfg, args) -> None:
    from bot.edgecurve import run_edgecurve

    await run_edgecurve(cfg, days=args.days, symbols=args.symbols,
                        offset_days=args.offset)


async def cmd_nullcheck(cfg, args) -> None:
    from bot.nullcheck import run_nullcheck

    await run_nullcheck(cfg, days=args.days, symbols=args.symbols, draws=args.draws,
                        offset_days=args.offset)


def main() -> int:
    # Shared flags live on a parent so they work after the subcommand,
    # which is where people naturally type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default="config.yaml")
    common.add_argument("--symbols", nargs="*", help="override the auto universe")
    common.add_argument("--log-level", default=None)

    parser = argparse.ArgumentParser(description="Binance momentum scanner bot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("universe", parents=[common], help="print the watchlist")
    sub.add_parser("scan", parents=[common], help="one-shot signal scan, no trading")
    sub.add_parser("live", parents=[common], help="run the bot")
    bt = sub.add_parser("backtest", parents=[common], help="replay historical data")
    bt.add_argument("--days", type=float, default=7.0)
    tr = sub.add_parser("train", parents=[common], help="fit the ML filter")
    tr.add_argument("--days", type=float, default=45.0)
    tr.add_argument("--threshold", type=float, default=None,
                    help="probability cutoff to evaluate (default: ml.min_probability)")
    tr.add_argument("--offset", type=float, default=0.0,
                    help="end the window this many days before now (holdout carving)")
    sub.add_parser("model", parents=[common], help="show the saved model's metrics")
    sw = sub.add_parser("sweep", parents=[common],
                        help="walk-forward parameter search over the entry gates")
    sw.add_argument("--days", type=float, default=30.0)
    sw.add_argument("--folds", type=int, default=4)
    nc = sub.add_parser("nullcheck", parents=[common],
                        help="do the entry rules beat random entries?")
    nc.add_argument("--days", type=float, default=30.0)
    nc.add_argument("--draws", type=int, default=3,
                    help="random entries drawn per real signal")
    nc.add_argument("--offset", type=float, default=0.0,
                    help="end the window this many days before now (holdout carving)")
    ex = sub.add_parser("experiments", parents=[common],
                        help="score every direction x timeframe x stop-mode variant")
    ex.add_argument("--days", type=float, default=30.0)
    ex.add_argument("--offset", type=float, default=0.0,
                    help="end the window this many days before now (holdout carving)")
    mk2 = sub.add_parser("maker", parents=[common],
                         help="resting limit entry vs market entry, with adverse selection")
    mk2.add_argument("--days", type=float, default=30.0)
    mk2.add_argument("--offset", type=float, default=0.0)
    mk2.add_argument("--window", type=int, default=3,
                     help="bars the limit rests before being pulled")
    mk2.add_argument("--maker-bps", type=float, default=None, dest="maker_bps",
                     help="maker fee per side; defaults to the taker fee, which "
                          "is what Binance charges at VIP0")
    eq = sub.add_parser("equities", parents=[common],
                        help="run the same null test on real US stocks and ETFs")
    eq.add_argument("--days", type=float, default=59.0)
    eq.add_argument("--interval", default="5m",
                    help="5m/15m up to 59 days, 1h up to 729, 1d for decades")
    eq.add_argument("--hold-bars", type=int, default=None, dest="hold_bars",
                    help="hold length in BARS; the minutes setting is ambiguous "
                         "once a bar is a day")
    ew = sub.add_parser("eqwindows", parents=[common],
                        help="the equity test over non-overlapping windows -- "
                             "a second asset class is stronger evidence than "
                             "another month of the same one")
    ew.add_argument("--windows", type=int, default=6)
    ew.add_argument("--days", type=float, default=90.0,
                    help="length of each window; windows x days must fit "
                         "inside Yahoo's 729-day limit for hourly bars")
    ew.add_argument("--interval", default="1h")
    ew.add_argument("--hold-bars", type=int, default=None, dest="hold_bars")

    ld = sub.add_parser("ladder", parents=[common],
                        help="the viral buy-the-dip/sell-the-rally ladder vs "
                             "buy-and-hold, over non-overlapping decades")
    ld.add_argument("--window-years", type=float, default=10.0,
                    dest="window_years",
                    help="length of each non-overlapping window, in years")
    ld.add_argument("--windows", type=int, default=0,
                    help="how many windows to use; 0 means every one that fits")
    ld.add_argument("--start-equity", type=float, default=0.5,
                    dest="start_equity",
                    help="fraction invested on day one; the rest is the cash "
                         "the dip rungs spend. The post does not say.")
    ld.add_argument("--capital", type=float, default=10_000.0)
    ld.add_argument("--dividend-source", default="yahoo",
                    choices=["yahoo", "shiller"], dest="dividend_source",
                    help="with --measured-yields: ^SP500TR vs ^GSPC (1988+, "
                         "same vendor as the prices) or the Shiller "
                         "dividend series (1871+, independent compiler)")
    ld.add_argument("--rearm", default="cross", choices=["cross", "once", "always"],
                    help="when a fired rung becomes eligible again. Mine, not "
                         "the post's -- every run audits all three.")
    ld.add_argument("--trigger", default="close", choices=["close", "intrabar"],
                    help="test the threshold against the close, or against the "
                         "bar's low/high and fill at the level")
    ld.add_argument("--preset", default="default",
                    choices=["default", "collapsed"],
                    help="which symbol list to use when --symbols is not "
                         "given. 'collapsed' is markets that fell hard and did "
                         "not come back: Athens -93%%, Shanghai -72%%, Hong "
                         "Kong -65%%, Tokyo.")
    ld.add_argument("--tax-rate", type=float, default=0.0, dest="tax_rate",
                    help="capital gains rate in percent, applied to realised "
                         "gains and dividends. Pass -1 for the historical "
                         "era table. Every run prints the whole ladder anyway.")
    ld.add_argument("--liquidate", action="store_true",
                    help="sell both arms on the last bar and tax the "
                         "unrealised gain -- the symmetric comparison")
    ld.add_argument("--measured-yields", action="store_true",
                    dest="measured_yields",
                    help="replace the assumed dividend and bill rates with the "
                         "real series (^SP500TR vs ^GSPC, and ^IRX). Only 1988 "
                         "onward, and every symbol is clipped to that span.")

    lt = sub.add_parser("longterm", parents=[common],
                        help="the long-horizon test: does a trend or "
                             "valuation rule beat holding the index, once it "
                             "is compared against its own average weight?")
    lt.add_argument("--window-years", type=float, default=15.0,
                    dest="window_years")
    lt.add_argument("--windows", type=int, default=0)
    lt.add_argument("--capital", type=float, default=10_000.0)
    lt.add_argument("--band", type=float, default=0.05,
                    help="do not rebalance for a gap smaller than this")
    lt.add_argument("--tax-rate", type=float, default=0.0, dest="tax_rate",
                    help="capital gains rate in percent; negative for the "
                         "historical era table")
    lt.add_argument("--liquidate", action="store_true")
    lt.add_argument("--measured-yields", action="store_true",
                    dest="measured_yields",
                    help="real bill and dividend series instead of the "
                         "assumed constants")

    al = sub.add_parser("allocate", parents=[common],
                        help="the multi-asset search, pre-registered: choose "
                             "on the first half, judge once on the second")
    al.add_argument("--band", type=float, default=0.05)
    al.add_argument("--max-leverage", type=float, default=2.0,
                    dest="max_leverage")
    al.add_argument("--tax-rate", type=float, default=0.0, dest="tax_rate",
                    help="negative for the historical era table")
    al.add_argument("--liquidate", action="store_true")
    al.add_argument("--seed", type=int, default=7,
                    help="bootstrap seed; the result must not depend on it")

    tp = sub.add_parser("trendpanel", parents=[common],
                        help="trend following across sixteen markets, as a "
                             "replication on markets it was not chosen on")
    tp.add_argument("--cost-bps", type=float, default=10.0, dest="cost_bps")
    tp.add_argument("--target-vol", type=float, default=0.10,
                    dest="target_vol")
    tp.add_argument("--max-gross", type=float, default=2.0, dest="max_gross")
    tp.add_argument("--seed", type=int, default=7)

    sub.add_parser("basketcheck", parents=[common],
                   help="are the built multi-asset series the same thing as "
                        "the funds a real investor would have held?")

    vc = sub.add_parser("vendorcheck", parents=[common],
                        help="does a second, independent vendor agree about "
                             "the century of prices the ladder rests on?")
    vc.add_argument("--symbol", default="^GSPC",
                    help="the Yahoo series to check against Shiller's S&P data")
    vc.add_argument("--min-bars", type=int, default=15, dest="min_bars",
                    help="skip months with fewer daily bars than this; a "
                         "partial month is a coverage gap, not a disagreement")

    ac = sub.add_parser("assetclass", parents=[common],
                        help="crypto vs tokenized equities vs gold")
    ac.add_argument("--days", type=float, default=30.0)
    ac.add_argument("--offset", type=float, default=0.0)
    ac.add_argument("--size", type=int, default=120)
    ac.add_argument("--min-volume", type=float, default=1000000.0, dest="min_volume")
    sub.add_parser("reconcile", parents=[common],
                   help="did the live bot see what a replay of the same minutes sees?")
    rp = sub.add_parser("replicate", parents=[common],
                        help="does the edge hold across NON-OVERLAPPING periods?")
    rp.add_argument("--days", type=float, default=30.0)
    rp.add_argument("--windows", type=int, default=5)
    rp.add_argument("--point-in-time", action="store_true", dest="point_in_time",
                    help="rank the universe on volume from before each window "
                         "instead of today's, removing ranking look-ahead")
    rg = sub.add_parser("regime", parents=[common],
                        help="does the edge depend on volatility or time of day?")
    rg.add_argument("--days", type=float, default=45.0)
    rg.add_argument("--offset", type=float, default=14.0)
    sc = sub.add_parser("spreadcheck", parents=[common],
                        help="validate the spread estimator against live quotes")
    sc.add_argument("--days", type=float, default=7.0)
    ef = sub.add_parser("exitfit", parents=[common],
                        help="where price actually goes after a signal, and do the exits match?")
    ef.add_argument("--days", type=float, default=45.0)
    ef.add_argument("--offset", type=float, default=14.0)
    ef.add_argument("--folds", type=int, default=3)
    mk = sub.add_parser("markets", parents=[common],
                        help="does widening the universe help, by liquidity tier?")
    mk.add_argument("--days", type=float, default=21.0)
    mk.add_argument("--offset", type=float, default=14.0)
    mk.add_argument("--size", type=int, default=150)
    mk.add_argument("--min-volume", type=float, default=500000.0, dest="min_volume")
    ec = sub.add_parser("edgecurve", parents=[common],
                        help="does the edge grow with the size of the dislocation?")
    ec.add_argument("--days", type=float, default=45.0)
    ec.add_argument("--offset", type=float, default=0.0,
                    help="end the window this many days before now (holdout carving)")

    args = parser.parse_args()
    if args.symbols:
        args.symbols = [s.upper() for s in args.symbols]

    cfg = load_config(args.config)
    setup_logging(args.log_level or cfg.logging.level)

    handlers = {
        "universe": cmd_universe,
        "scan": cmd_scan,
        "live": cmd_live,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "model": cmd_model,
        "sweep": cmd_sweep,
        "nullcheck": cmd_nullcheck,
        "experiments": cmd_experiments,
        "edgecurve": cmd_edgecurve,
        "markets": cmd_markets,
        "exitfit": cmd_exitfit,
        "spreadcheck": cmd_spreadcheck,
        "regime": cmd_regime,
        "replicate": cmd_replicate,
        "reconcile": cmd_reconcile,
        "assetclass": cmd_assetclass,
        "vendorcheck": cmd_vendorcheck,
        "longterm": cmd_longterm,
        "basketcheck": cmd_basketcheck,
        "allocate": cmd_allocate,
        "trendpanel": cmd_trendpanel,
        "ladder": cmd_ladder,
        "equities": cmd_equities,
        "eqwindows": cmd_eqwindows,
        "maker": cmd_maker,
    }
    try:
        asyncio.run(handlers[args.command](cfg, args))
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as exc:  # noqa: BLE001
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
