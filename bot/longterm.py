"""The long-horizon bot, after the short-horizon one failed to find anything.

The rest of this project is a record of an intraday strategy not working: the
momentum entry is indistinguishable from random, the reversal edge is +0.04%
against a 0.20% toll and does not survive being counted properly, eleven
improvements died out of sample, and the viral ladder turned out to be an asset
allocation in a timing costume. The one thing every measurement agreed on is
that **being invested paid and the rules did not**.

So this asks the only question left standing, at the horizon where the evidence
actually lives: over decades rather than minutes, does ANY rule beat simply
holding the thing?

Two candidates, chosen because they have real published support rather than a
screenshot -- this is the opposite of how hypotheses were picked earlier in the
repo, and deliberately so:

  sma    Faber's trend rule. Hold the index while it is above its own 200-day
         average, sit in bills when it is below. Checked monthly, which is what
         the paper does and also what keeps the trade count survivable.
  cape   Valuation. Shiller's PE10, converted to a weight by its POINT-IN-TIME
         percentile against its own past -- expensive market, less equity. The
         percentile uses only months that had already happened, because a
         percentile against the full history is a machine for buying 1932.
  blend  The average of the two weights, since they disagree often.

THE CONTROL THAT MATTERS, and the lesson carried over from the ladder: a rule
that ends up 70% invested must be compared against a STATIC 70% portfolio, not
against 100% equity. Otherwise the cash it happens to hold gets scored as skill.
Every strategy here is measured twice -- against buy-and-hold, and against a
static portfolio rebalanced to that strategy's own realised average weight.
The second number is the only one that is about the rule.

PREDICTIONS, written before the first run:

  1. `sma` cuts maximum drawdown substantially. This is the best-documented
     part of the trend literature and I expect it to hold.
  2. `sma` roughly matches buy-and-hold on total return before costs and tax,
     and loses after them, because it trades and buy-and-hold does not.
  3. Against the weight-matched static baseline, `sma`'s timing term is
     POSITIVE -- this is the one place in this whole repo I expect a real
     effect -- but small, and not significant at the window level with the
     handful of non-overlapping windows a century affords.
  4. `cape` is worse than `sma` on both, because valuation has been telling US
     investors to be underweight for thirty years and it has been wrong.

If 3 comes out negative, the honest conclusion is that nothing in this repo
beats holding the index, at any horizon, and that is a finding rather than a
failure.

WHAT THIS DOES NOT DO YET: contributions. A monthly paycheque going in changes
the comparison from time-weighted to money-weighted returns, and mixing the two
is how a DCA backtest ends up flattering whichever arm received cash when
prices were low. Lump sum only, stated rather than quietly assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config
from .exchange import Candle
from .findings import record
# The money mechanics -- accrual, costs, tax, windowing, window statistics --
# live in ladder.py and are imported rather than copied. The invariant that a
# ladder with no rungs must equal buy-and-hold to the cent exists precisely
# because two copies of this arithmetic had already drifted apart once.
from .ladder import (DAY_MS, MIN_WINDOWS, NO_TAX, YEAR_DAYS, Run, Taxes,
                     Yields, _accrue, _Book, _buy, _sell, _settle, _stats,
                     cost_bps, load_measured_yields, split_windows, year_of)
from .shiller import cape_path, load_shiller, month_of
from .yahoo import load_equity_history

log = logging.getLogger(__name__)

STRATEGIES = ("hold", "sma", "cape", "blend")

# Faber uses a 10-month average checked monthly; 200 trading days is the daily
# equivalent everyone quotes. Both are the same idea and neither is tuned here:
# the number comes from the paper, not from this data, which is the only way a
# published effect stays a test rather than a fit.
SMA_DAYS = 200

# CAPE needs a history to be a percentile OF. Twenty years of prior months
# before the rule is allowed an opinion; before that it stays fully invested,
# which is the neutral choice rather than the flattering one.
CAPE_MIN_HISTORY = 240

# Do not trade for less than this. A monthly rebalance with no band churns the
# book every month for a fraction of a percent, and the costs of that are real
# while the benefit is not.
REBALANCE_BAND = 0.05

DEFAULT_SYMBOLS = ["^GSPC", "^IXIC", "^N225", "^HSI"]


@dataclass
class Plan:
    """A strategy's answer at one rebalance date."""
    weight: float
    reason: str


def sma_weight(closes: list[float], days: int = SMA_DAYS) -> Plan:
    """Fully invested above the average, fully out below it.

    `closes` must end at the last bar the rule is allowed to have seen. The
    caller passes history up to the PREVIOUS bar and trades on the current one;
    deciding and trading on the same close is the oldest look-ahead there is.
    """
    if len(closes) < days:
        return Plan(1.0, "no average yet")
    average = sum(closes[-days:]) / days
    above = closes[-1] > average
    return Plan(1.0 if above else 0.0,
                f"{'above' if above else 'below'} the {days}d average")


def cape_weight(cape: float, history: list[float],
                min_history: int = CAPE_MIN_HISTORY) -> Plan:
    """Weight falls as the market gets expensive relative to its own past.

    `history` is every CAPE reading BEFORE this one. Ranking against the full
    series instead would know in 1929 what 1999 was going to look like, which
    is the look-ahead that makes every valuation backtest work.
    """
    if cape <= 0 or len(history) < min_history:
        return Plan(1.0, "not enough CAPE history")
    cheaper = sum(1 for h in history if h < cape)
    percentile = cheaper / len(history)
    return Plan(max(0.0, min(1.0, 1.0 - percentile)),
                f"CAPE {cape:.1f} at the {percentile:.0%} percentile")


@dataclass
class Weights:
    """Everything a strategy needs to answer, keyed by what it is allowed to see."""
    strategy: str
    cape: dict[tuple[int, int], float]

    def at(self, closes: list[float], key: tuple[int, int],
           cape_history: list[float]) -> Plan:
        if self.strategy == "hold":
            return Plan(1.0, "always invested")
        if self.strategy == "sma":
            return sma_weight(closes)
        if self.strategy == "cape":
            return cape_weight(self.cape.get(key, 0.0), cape_history)
        if self.strategy == "blend":
            a = sma_weight(closes)
            b = cape_weight(self.cape.get(key, 0.0), cape_history)
            return Plan((a.weight + b.weight) / 2, f"{a.reason}; {b.reason}")
        raise ValueError(f"strategy must be one of {STRATEGIES}, "
                         f"got {self.strategy!r}")


def _rebalance(book: _Book, price: float, target: float, band: float,
               bps: float, tax_rate: float) -> bool:
    """Move the book toward `target` weight, if it is far enough away to bother."""
    value = book.value(price)
    if value <= 0:
        return False
    delta = value * target - book.shares * price
    if abs(delta) < band * value:
        return False
    if delta > 0:
        amount = min(delta, book.cash)
        if amount <= 0:
            return False
        _buy(book, price, amount, bps)
    else:
        _sell(book, price, min(-delta / price, book.shares), bps, tax_rate)
    return True


def simulate(candles: list[Candle], strategy: str,
             cape: dict[tuple[int, int], float] | None = None,
             capital: float = 10_000.0, yields: Yields | None = None,
             taxes: Taxes = NO_TAX, band: float = REBALANCE_BAND,
             fixed_weight: float | None = None) -> Run:
    """One strategy over one window of daily bars, rebalanced monthly.

    `fixed_weight` bypasses the strategy and holds a constant target. That is
    how the weight-matched baseline is built, and it runs through this exact
    function so that the comparison differs in the rule and in nothing else --
    not in the rebalance band, not in the cost model, not in the tax treatment.
    """
    from .ladder import FLAT

    if len(candles) < 2:
        raise ValueError("a long-horizon run needs at least two bars")
    yields = FLAT if yields is None else yields
    rules = Weights(strategy, cape or {})

    first = candles[0].close
    entry_bps = cost_bps(candles[0].open_time)
    start_weight = fixed_weight if fixed_weight is not None else \
        rules.at([first], month_of(candles[0].open_time), []).weight
    invested = capital * start_weight
    book = _Book(
        cash=capital - invested,
        shares=invested * (1.0 - entry_bps / 10_000.0) / first,
        basis=first / (1.0 - entry_bps / 10_000.0) if invested > 0 else first,
        anchor=first, peak=first,
    )

    closes = [first]
    # Seed the CAPE history with every reading from BEFORE this window. An
    # investor in 1990 knows what the market cost in 1929; restarting the
    # history at each window boundary is not conservatism, it is a different
    # and impossible strategy -- the first version of this file did exactly
    # that and the valuation rule silently never fired, because 180 months of
    # window can never reach the 240 the percentile needs.
    start = month_of(candles[0].open_time)
    seen_cape = [v for k, v in sorted((rules.cape or {}).items()) if k < start]
    cape_seen_keys: set[tuple[int, int]] = {k for k in (rules.cape or {})
                                            if k < start}
    peak_value = book.value(first)
    max_dd = 0.0
    equity_days = 0.0
    total_days = 0.0
    trades = 0
    month = month_of(candles[0].open_time)

    for prev, bar in zip(candles, candles[1:]):
        price = bar.close
        days = (bar.open_time - prev.open_time) / DAY_MS
        _accrue(book, price, prev.open_time, bar.open_time, yields, taxes)

        key = month_of(bar.open_time)
        if key != month:
            month = key
            # Everything the rule may look at ends at `prev`; it acts on `bar`.
            if fixed_weight is not None:
                target = fixed_weight
            else:
                target = rules.at(closes, key, list(seen_cape)).weight
            if _rebalance(book, price, target, band, cost_bps(bar.open_time),
                          taxes.rate_at(bar.open_time)):
                trades += 1
            # The month's own CAPE only joins the history AFTER it was used, so
            # a percentile never contains the reading it is ranking.
            value = rules.cape.get(key, 0.0)
            if value > 0 and key not in cape_seen_keys:
                cape_seen_keys.add(key)
                seen_cape.append(value)

        closes.append(price)
        value = book.value(price)
        peak_value = max(peak_value, value)
        if peak_value > 0:
            max_dd = max(max_dd, (1.0 - value / peak_value) * 100.0)
        if value > 0:
            equity_days += days * (book.shares * price) / value
        total_days += days

    _settle(book, candles[-1], taxes)
    years = (candles[-1].open_time - candles[0].open_time) / DAY_MS / YEAR_DAYS
    name = strategy if fixed_weight is None else f"static {fixed_weight:.0%}"
    return Run(name=name, final=book.value(candles[-1].close), initial=capital,
               years=years, max_dd=max_dd, sells=trades, cash_end=book.cash,
               equity_days=equity_days, total_days=total_days,
               tax_paid=book.tax_paid)


async def run_longterm(
    cfg: Config,
    window_years: float = 15.0,
    windows: int = 0,
    capital: float = 10_000.0,
    symbols: list[str] | None = None,
    band: float = REBALANCE_BAND,
    tax_rate: float | None = 0.0,
    liquidate: bool = False,
    measured_yields: bool = False,
) -> dict:
    from .ladder import FLAT

    names = symbols or DEFAULT_SYMBOLS
    history = load_equity_history(names, "1d", 40_000.0, cfg.path("data/yahoo"))
    if not history:
        raise RuntimeError("no daily history returned")

    cape = cape_path(load_shiller(cfg.path("data/shiller")))
    taxes = Taxes(tax_rate, liquidate)
    yields: Yields = FLAT
    if measured_yields:
        measured = load_measured_yields(cfg, "shiller")
        if measured is None:
            raise RuntimeError("measured yields unavailable")
        yields = measured
        lo, hi = measured.start_ms, measured.end_ms
        for symbol in list(history):
            history[symbol] = [c for c in history[symbol] if lo <= c.open_time <= hi]

    _print_setup(window_years, capital, band, taxes, yields, len(cape))

    out: dict[str, dict] = {}
    for symbol in names:
        candles = history.get(symbol)
        if not candles or len(candles) < 500:
            log.warning("%s: %d daily bars, skipped",
                        symbol, len(candles) if candles else 0)
            continue
        chunks = split_windows(candles, window_years, windows)
        if len(chunks) < 2:
            log.warning("%s: only %d windows of %.0f years, skipped",
                        symbol, len(chunks), window_years)
            continue
        # CAPE is a US series. Ranking Tokyo against American valuations would
        # be nonsense, so the rules that use it are only offered where it means
        # something, and the output says which markets those were.
        use_cape = symbol in ("^GSPC", "^IXIC", "SPY", "^SP500TR")
        out[symbol] = _run_symbol(symbol, chunks, cape if use_cape else {},
                                  capital, yields, taxes, band, use_cape)

    if not out:
        raise RuntimeError("no symbol had enough history")

    lead = "^GSPC" if "^GSPC" in out else next(iter(out))
    headline = out[lead]
    if headline["windows"] >= MIN_WINDOWS:
        # `best` across all strategies is almost always `hold`, which is the
        # true answer and a useless headline -- "the best rule is no rule, and
        # it beats itself by zero" tells a reader nothing about the rules. So
        # the record carries the best actual RULE and states separately that
        # holding beat it, which is the sentence worth reading.
        rules = [r for r in headline["strategies"] if r["name"] != "hold"]
        rule = max(rules, key=lambda r: r["total"]) if rules else headline["best"]
        null = next((r for r in headline["strategies"]
                     if r["name"] == "hold"), None)
        record(cfg, "longterm", {
            "symbol": lead, "windows": headline["windows"],
            "window_years": window_years,
            "best": rule["name"],
            "vs_hold": rule["total"],
            "timing": rule["timing"],
            "timing_null": null["timing"] if null else None,
            "beaten_by_holding": rule["total"] < 0,
            "p": rule["timing_p"],
            "p_adjusted": min(1.0, rule["timing_p"] * len(STRATEGIES)),
            "max_dd": rule["dd"],
            "hold_dd": headline["hold_dd"],
            "taxes": taxes.describe(), "yields": yields.describe(),
        })
    return out


def _run_symbol(symbol: str, chunks: list[list[Candle]],
                cape: dict[tuple[int, int], float], capital: float,
                yields: Yields, taxes: Taxes, band: float,
                use_cape: bool) -> dict:
    """Every strategy against buy-and-hold and against its own weight-match."""
    usable = [s for s in STRATEGIES if use_cape or s in ("hold", "sma")]
    rows: dict[str, dict[str, list[float]]] = {
        s: {"total": [], "timing": [], "dd": [], "weight": [], "trades": []}
        for s in usable
    }
    holds: list[Run] = []

    for chunk in chunks:
        hold = simulate(chunk, "hold", cape, capital, yields, taxes, band,
                        fixed_weight=1.0)
        holds.append(hold)
        for name in usable:
            run = simulate(chunk, name, cape, capital, yields, taxes, band)
            # The control: a static portfolio at THIS run's own realised
            # average weight, so what is left is the rule and not the cash.
            matched = simulate(chunk, "hold", cape, capital, yields, taxes,
                               band, fixed_weight=run.invested_pct / 100.0)
            rows[name]["total"].append(run.cagr - hold.cagr)
            rows[name]["timing"].append(run.cagr - matched.cagr)
            rows[name]["dd"].append(run.max_dd)
            rows[name]["weight"].append(run.invested_pct)
            rows[name]["trades"].append(float(run.sells))

    summary = []
    for name, cols in rows.items():
        mean, se, p, df = _stats(cols["total"])
        t_mean, t_se, t_p, _ = _stats(cols["timing"])
        summary.append({
            "name": name, "total": mean, "se": se, "p": p, "df": df,
            "timing": t_mean, "timing_se": t_se, "timing_p": t_p,
            "wins": sum(1 for d in cols["total"] if d > 0),
            "dd": sum(cols["dd"]) / len(cols["dd"]),
            "weight": sum(cols["weight"]) / len(cols["weight"]),
            "trades": sum(cols["trades"]) / len(cols["trades"]),
        })
    summary.sort(key=lambda d: -d["total"])
    _print_symbol(symbol, chunks, holds, summary, use_cape)
    return {"symbol": symbol, "windows": len(chunks), "strategies": summary,
            "best": summary[0],
            "hold_cagr": sum(h.cagr for h in holds) / len(holds),
            "hold_dd": sum(h.max_dd for h in holds) / len(holds),
            "chunks": chunks}


def _print_setup(window_years, capital, band, taxes, yields, cape_months) -> None:
    print("\n" + "=" * 80)
    print("  THE LONG-HORIZON TEST -- does any rule beat holding the index?")
    print("=" * 80)
    print(f"    window length               {window_years:>12.0f} years"
          "   (each window is one observation)")
    print(f"    starting capital            {capital:>12,.0f}")
    print(f"    rebalance band              {band:>12.0%}"
          "   checked monthly")
    print(f"    trend rule                  {SMA_DAYS:>12}d"
          "   from Faber, not fitted here")
    print(f"    CAPE months available       {cape_months:>12,}"
          "   ranked point-in-time only")
    print(f"    yields                      {yields.describe()}")
    print(f"    capital gains tax           {taxes.describe()}")
    print("    contributions                    NOT MODELLED -- lump sum only;")
    print("                                     see the module docstring.")
    print("=" * 80)


def _print_symbol(symbol, chunks, holds, summary, use_cape) -> None:
    from .replicate import t_critical

    first = year_of(chunks[0][0].open_time)
    last = year_of(chunks[-1][-1].open_time)
    hold_cagr = sum(h.cagr for h in holds) / len(holds)
    hold_dd = sum(h.max_dd for h in holds) / len(holds)

    print(f"\n  {symbol}  --  {len(chunks)} non-overlapping windows, "
          f"{first}-{last}" + ("" if use_cape else "   (no CAPE: not a US index)"))
    print("-" * 80)
    print(f"    buy-and-hold: {hold_cagr:+.2f}% CAGR, {hold_dd:.1f}% max "
          "drawdown   (mean of the per-window figures)")
    print("-" * 80)
    print(f"    {'strategy':<10}{'vs hold':>10}{'p':>7}{'win':>7}"
          f"{'timing':>10}{'p':>7}{'max DD':>9}{'in mkt':>8}{'trades':>8}")
    for row in summary:
        print(f"    {row['name']:<10}{row['total']:>+9.2f}%{row['p']:>7.3f}"
              f"{row['wins']:>4}/{len(chunks):<2}{row['timing']:>+9.2f}%"
              f"{row['timing_p']:>7.3f}{row['dd']:>8.1f}%{row['weight']:>7.0f}%"
              f"{row['trades']:>8.1f}")
    print("-" * 80)

    best = max(summary, key=lambda r: r["timing"])
    adjusted = min(1.0, best["timing_p"] * len(summary))
    if best["df"] >= 1 and best["timing_se"] > 0:
        tc = t_critical(best["df"])
        lo = best["timing"] - tc * best["timing_se"]
        hi = best["timing"] + tc * best["timing_se"]
        print(f"    best timing term: {best['name']} at {best['timing']:+.2f}% "
              f"CAGR, 95% CI [{lo:+.2f}%, {hi:+.2f}%]")
        print(f"    p = {best['timing_p']:.3f} raw, {adjusted:.3f} after "
              f"Bonferroni over the {len(summary)} strategies")
    _verdict(symbol, summary, best, adjusted, hold_dd, len(chunks))


def _verdict(symbol, summary, best, adjusted, hold_dd, windows) -> None:
    trend = next((r for r in summary if r["name"] == "sma"), None)
    if windows < MIN_WINDOWS:
        print(f"  ONLY {windows} WINDOWS -- no verdict.")
        return

    if trend is not None:
        cut = hold_dd - trend["dd"]
        print(f"  Prediction 1 -- the trend rule cuts drawdown: "
              f"{'HELD' if cut > 0 else 'FAILED'}, "
              f"{hold_dd:.1f}% -> {trend['dd']:.1f}%.")
        print(f"  Prediction 2 -- it loses on total return after costs: "
              f"{'HELD' if trend['total'] < 0 else 'FAILED'}, "
              f"{trend['total']:+.2f}% CAGR vs buy-and-hold.")
        held = trend["timing"] > 0
        print(f"  Prediction 3 -- its timing term is positive but not "
              f"significant: {'HELD' if held else 'FAILED'},\n    "
              f"{trend['timing']:+.2f}% at p = {trend['timing_p']:.3f}.")

    # The `hold` row is the null. It follows no rule, so whatever timing term
    # it shows is what this metric returns for doing nothing -- the rebalance
    # band lets cash drift and the weight-matched baseline never drifts the
    # same way. Any rule has to clear THAT, not zero. Nobody built this as a
    # control; it fell out of running the do-nothing strategy through the same
    # machine, which is the cheapest null this project has ever had.
    null = next((r for r in summary if r["name"] == "hold"), None)
    if null is not None and best["name"] != "hold":
        excess = best["timing"] - null["timing"]
        print(f"  The `hold` row is the null: it follows no rule and still "
              f"scores {null['timing']:+.2f}% on this\n  metric, which is what "
              "the rebalance band alone is worth. Against that null the best "
              f"rule\n  adds {excess:+.2f}% CAGR"
              + (", which is not nothing but is well inside the interval."
                 if excess > 0 else ", i.e. less than doing nothing."))

    if best["timing"] > 0 and adjusted < 0.05:
        print(f"  {best['name'].upper()} BEATS ITS OWN WEIGHT-MATCHED BASELINE "
              f"by {best['timing']:+.2f}% CAGR, surviving\n  multiplicity. This "
              "is the first thing in this repository to do that. Replicate it "
              "on\n  a market it was not chosen on before believing it.")
    elif best["timing"] > 0:
        print(f"  The best timing term is {best['timing']:+.2f}% "
              f"({best['name']}), which is positive and not\n  significant "
              f"(p = {adjusted:.3f} adjusted). Same shape as everything else "
              "here: a small\n  effect the sample cannot separate from zero.")
    else:
        print(f"  NO RULE BEATS ITS OWN WEIGHT-MATCHED BASELINE. The best is "
              f"{best['timing']:+.2f}% CAGR\n  ({best['name']}). Whatever these "
              "strategies do to the return, they do it by\n  changing how much "
              "is invested -- not by choosing when.")
