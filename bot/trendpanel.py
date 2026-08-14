"""Trend following across sixteen markets, as a replication rather than a search.

The two-asset test ended with `tsmom` posting the highest holdout Sharpe of
anything tried (0.817 against equity's 0.598). That number is not evidence: it
was the best of eight on a set that was searched, and this project has watched
eleven best-of-set numbers evaporate. But it is also the one strategy in the
list with a large independent literature behind it, and the correct response to
"the best row was the one with prior support" is not to celebrate it and not to
discard it. It is to test it somewhere it was not chosen.

WHAT MAKES THIS A REPLICATION AND NOT ANOTHER SEARCH:

  * The markets are DIFFERENT. `^GSPC` and `^TNX` were the two assets of the
    first test and are excluded here. What is left is fifteen other markets --
    eight non-US equity indices, the Nasdaq, the 30-year yield, five
    commodities and the dollar -- none of which contributed to the result being
    replicated.
  * The parameter is NOT fitted. Twelve months is the lookback from Moskowitz,
    Ooi and Pedersen (2012); it is not tuned here and no alternative is tried.
    There is nothing to overfit because nothing is chosen.
  * There is ONE comparison that matters, fixed in advance: the trend portfolio
    against the SAME portfolio with the signal removed. Same markets, same
    inverse-volatility weights, same rebalancing, same costs -- always on. The
    difference is the timing rule and nothing else. This is the control the
    ladder taught: a strategy that sits out half the time must not be scored
    against something fully invested, or the cash gets counted as skill.

BREADTH IS THE POINT. Two assets can only diversify so far, which is why the
first test's effect was too small to prove on sixty years. Fifteen markets that
do not move together give the same idea far more to work with, and the
literature's whole claim rests on that breadth rather than on any one market.

HONEST ACCOUNTING, and which way each item cuts:

  * Non-US index dividends do not exist as a free series, and leaving them out
    understates every index. That penalises the ALWAYS-ON control more than the
    trend rule, because the control is long all the time -- a bias in favour of
    the hypothesis. So a dividend yield is assumed and added, and the run is
    repeated at 0% and 4.5% so the size of the assumption is visible.
  * Commodities and the dollar are treated as excess returns funded at the bill
    rate, which is what a futures position is. For the dollar index that is an
    approximation: a real position would earn the foreign leg's rate.
  * The panel is unbalanced. Korea joins in 1996, the commodities in 2000. Each
    market enters when its data does, and the number of live markets is
    reported per era, because a Sharpe computed on four markets and one
    computed on sixteen are not the same statistic.
  * Costs are a flat assumption here rather than the equity era table: futures
    and index exposure are cheaper than 1970s stock commissions and pretending
    otherwise would be a different kind of wrong.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from .allocate import block_bootstrap_sharpe, sharpe_of
from .basket import annualised, month_end, price_returns
from .config import Config
from .findings import record
from .yahoo import load_equity_history

log = logging.getLogger(__name__)

# The lookback, from the literature. Not a parameter of this file.
LOOKBACK = 12

# Fifteen markets, none of which is the pair the effect was found on.
# Grouped so the output can report whether the result is one asset class.
PANEL: dict[str, tuple[str, ...]] = {
    "equity": ("^IXIC", "^N225", "^FTSE", "^HSI", "^GDAXI", "^FCHI",
               "^AXJO", "^BVSP", "^KS11"),
    "rates": ("^TYX",),
    "commodity": ("GC=F", "CL=F", "SI=F", "HG=F", "ZC=F"),
    "currency": ("DX-Y.NYB",),
}
EXCLUDED = ("^GSPC", "^TNX")     # the markets the first test was run on
CASH = "^IRX"

# Index prices carry no dividends. Assumed, added, and varied -- omitting them
# would flatter the trend rule, which is out of the market a third of the time.
DIVIDEND_PCT = 3.0
DIVIDEND_SENSITIVITY = (0.0, 3.0, 4.5)

# `^TYX` is a yield, not a price. Same rolling-par-bond conversion as the
# basket, at a 30-year duration.
BOND_MATURITY = 18.0

# Per side. Index futures and liquid contracts, not 1970s equities.
COST_BPS = 10.0

# Annualised volatility the whole portfolio aims at, and the cap on gross
# exposure. Both are conventions, stated; neither is searched over.
TARGET_VOL = 0.10
MAX_GROSS = 2.0

# A market only counts once it has a lookback behind it.
MIN_MARKETS = 4


@dataclass
class Panel:
    """Monthly total returns per market, plus the bill rate."""
    markets: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    group: dict[str, str] = field(default_factory=dict)
    cash: dict[tuple[int, int], float] = field(default_factory=dict)

    def months(self) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        for series in self.markets.values():
            seen |= set(series)
        return sorted(seen & set(self.cash))


@dataclass
class Run:
    name: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    exposure: float
    turnover: float
    monthly: list[float] = field(default_factory=list)
    excess: list[float] = field(default_factory=list)
    live: list[int] = field(default_factory=list)


def _vol(series: list[float]) -> float:
    return statistics.pstdev(series) if len(series) > 1 else 0.0


def load_panel(cfg: Config, dividend_pct: float = DIVIDEND_PCT) -> Panel:
    from .basket import bond_returns

    names = [s for group in PANEL.values() for s in group]
    history = load_equity_history(names + [CASH], "1d", 40_000.0,
                                  cfg.path("data/yahoo"), auto_adjust=True)
    bills = month_end(history.get(CASH, []))
    if not bills:
        raise RuntimeError("the panel needs ^IRX for its cash leg")
    panel = Panel(cash={k: v / 100.0 / 12.0 for k, v in bills.items() if v > 0})

    for group, symbols in PANEL.items():
        for symbol in symbols:
            candles = history.get(symbol)
            if not candles:
                log.warning("%s did not load and is skipped", symbol)
                continue
            closes = month_end(candles)
            if group == "rates":
                series = bond_returns(closes, BOND_MATURITY)
            else:
                series = price_returns(closes)
                if group == "equity" and dividend_pct:
                    series = {k: r + dividend_pct / 100.0 / 12.0
                              for k, r in series.items()}
            if len(series) < LOOKBACK * 2:
                log.warning("%s has only %d months, skipped", symbol, len(series))
                continue
            panel.markets[symbol] = series
            panel.group[symbol] = group

    if len(panel.markets) < MIN_MARKETS:
        raise RuntimeError("too few markets loaded for a panel test")
    log.info("panel: %d markets, %d months", len(panel.markets),
             len(panel.months()))
    return panel


def simulate(panel: Panel, use_signal: bool, months: list[tuple[int, int]],
             cost_bps: float = COST_BPS, target_vol: float = TARGET_VOL,
             max_gross: float = MAX_GROSS,
             groups: tuple[str, ...] | None = None) -> Run:
    """The trend portfolio, or the identical portfolio with the signal removed.

    `use_signal=False` is the control and shares every other line of this
    function. If the two ever diverge for a reason other than the signal, they
    diverge for a reason nobody chose.
    """
    history: dict[str, list[float]] = {s: [] for s in panel.markets}
    weights: dict[str, float] = {s: 0.0 for s in panel.markets}
    value = 1.0
    peak = 1.0
    max_dd = 0.0
    turnover = 0.0
    gross_sum = 0.0
    monthly: list[float] = []
    excess: list[float] = []
    live_counts: list[int] = []

    for key in months:
        rate = panel.cash.get(key, 0.0)
        # 1. This month's return on what was already held. Positions are
        #    excess-return exposures funded at the bill rate, so the portfolio
        #    earns cash plus the weighted excess of each market.
        step = rate
        for symbol, weight in weights.items():
            if weight == 0.0:
                continue
            got = panel.markets[symbol].get(key)
            if got is None:
                continue
            step += weight * (got - rate)
        value *= 1.0 + step
        monthly.append(step)
        excess.append(step - rate)
        peak = max(peak, value)
        max_dd = max(max_dd, (1.0 - value / peak) * 100.0)
        gross_sum += sum(abs(w) for w in weights.values())

        # 2. Feed this month into history, then decide the NEXT month's book.
        for symbol, series in panel.markets.items():
            got = series.get(key)
            if got is not None:
                history[symbol].append(got)

        eligible = [s for s in panel.markets
                    if len(history[s]) >= LOOKBACK
                    and (groups is None or panel.group[s] in groups)]
        live_counts.append(len(eligible))
        target: dict[str, float] = {s: 0.0 for s in panel.markets}
        if eligible:
            raw: dict[str, float] = {}
            for symbol in eligible:
                window = history[symbol][-LOOKBACK:]
                vol = _vol(window)
                if vol <= 0:
                    continue
                on = 1.0 if (not use_signal or sum(window) > 0) else 0.0
                raw[symbol] = on / vol
            total = sum(raw.values())
            if total > 0:
                # Scale the whole book to the volatility target, using the
                # realised volatility of the equally-risk-weighted mix. Both
                # arms use the same scaling, so exposure is not what separates
                # them.
                shares = {s: v / total for s, v in raw.items()}
                mix = []
                for i in range(1, LOOKBACK + 1):
                    period = 0.0
                    for symbol, share in shares.items():
                        window = history[symbol]
                        if len(window) >= i:
                            period += share * window[-i]
                    mix.append(period)
                realised = _vol(mix) * (12 ** 0.5)
                scale = min(max_gross, target_vol / realised) if realised > 0 else 1.0
                target = {s: shares.get(s, 0.0) * scale for s in panel.markets}

        moved = sum(abs(target[s] - weights[s]) for s in panel.markets)
        turnover += moved
        value *= 1.0 - moved * cost_bps / 10_000.0
        weights = target

    n = max(len(monthly), 1)
    years = len(months) / 12.0
    cagr = ((value ** (1.0 / years) - 1.0) * 100.0) if years > 0 and value > 0 else 0.0
    _, vol = annualised(monthly)
    return Run(name="trend" if use_signal else "always on", cagr=cagr, vol=vol,
               sharpe=sharpe_of(excess), max_dd=max_dd,
               exposure=gross_sum / n * 100.0,
               turnover=turnover / years if years > 0 else 0.0,
               monthly=monthly, excess=excess, live=live_counts)


async def run_trendpanel(cfg: Config, cost_bps: float = COST_BPS,
                         target_vol: float = TARGET_VOL,
                         max_gross: float = MAX_GROSS, seed: int = 7) -> dict:
    panel = load_panel(cfg)
    months = panel.months()
    if len(months) < LOOKBACK * 4:
        raise RuntimeError("not enough months for a panel test")

    _print_setup(panel, months, cost_bps, target_vol, max_gross)

    trend = simulate(panel, True, months, cost_bps, target_vol, max_gross)
    always = simulate(panel, False, months, cost_bps, target_vol, max_gross)
    _print_pair("THE ONE COMPARISON, FIXED IN ADVANCE", [trend, always])

    lo, hi, p = block_bootstrap_sharpe(trend.excess, always.excess, seed)
    delta = trend.sharpe - always.sharpe
    _print_verdict(trend, always, delta, lo, hi, p, months)

    # Every group, including the one-market ones. The first version skipped
    # those "for having too few members" and thereby hid the entire result:
    # the panel effect turned out to live in a single currency index.
    present = [g for g in PANEL
               if any(panel.group[s] == g for s in panel.markets)]
    per_group = {}
    for group in present:
        t = simulate(panel, True, months, cost_bps, target_vol, max_gross,
                     groups=(group,))
        a = simulate(panel, False, months, cost_bps, target_vol, max_gross,
                     groups=(group,))
        per_group[group] = (t.sharpe - a.sharpe, t.sharpe, a.sharpe)
    leave_out = {}
    for group in present:
        rest = tuple(g for g in present if g != group)
        if not rest:
            continue
        t = simulate(panel, True, months, cost_bps, target_vol, max_gross,
                     groups=rest)
        a = simulate(panel, False, months, cost_bps, target_vol, max_gross,
                     groups=rest)
        leave_out[group] = t.sharpe - a.sharpe
    _print_groups(per_group, leave_out, delta)
    _print_dividend_sensitivity(cfg, months, cost_bps, target_vol, max_gross)

    record(cfg, "trendpanel", {
        "markets": len(panel.markets), "months": len(months),
        "delta_sharpe": delta, "trend_sharpe": trend.sharpe,
        "always_sharpe": always.sharpe, "p": p,
        "ci": [lo, hi], "trend_cagr": trend.cagr, "always_cagr": always.cagr,
        "trend_dd": trend.max_dd, "always_dd": always.max_dd,
    })
    return {"trend": trend, "always": always, "p": p, "ci": [lo, hi],
            "groups": per_group}


def _print_setup(panel, months, cost_bps, target_vol, max_gross) -> None:
    print("\n" + "=" * 80)
    print("  TREND ACROSS A PANEL -- a replication, on markets it was not "
          "chosen on")
    print("=" * 80)
    print(f"    markets            {len(panel.markets):>6}   "
          f"excluding {', '.join(EXCLUDED)}, which the first test used")
    print(f"    months             {len(months):>6}   "
          f"{months[0][0]}-{months[0][1]:02d} to {months[-1][0]}-{months[-1][1]:02d}")
    print(f"    lookback           {LOOKBACK:>6}m  "
          "from the literature, not fitted here")
    print(f"    volatility target  {target_vol:>6.0%}   "
          f"gross capped at {max_gross:.1f}x")
    print(f"    cost               {cost_bps:>6.0f}bps per side, on turnover")
    print(f"    index dividends    {DIVIDEND_PCT:>6.1f}%   "
          "assumed; varied at the end")
    print("-" * 80)
    for group, symbols in PANEL.items():
        present = [s for s in symbols if s in panel.markets]
        if present:
            print(f"    {group:<11}{' '.join(present)}")
    print("=" * 80)


def _print_pair(title: str, runs: list[Run]) -> None:
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)
    print(f"    {'portfolio':<12}{'CAGR':>9}{'vol':>8}{'Sharpe':>9}"
          f"{'max DD':>9}{'gross':>8}{'turn/yr':>9}")
    for r in runs:
        print(f"    {r.name:<12}{r.cagr:>+8.2f}%{r.vol:>7.1f}%{r.sharpe:>9.3f}"
              f"{r.max_dd:>8.1f}%{r.exposure:>7.0f}%{r.turnover:>8.2f}x")
    print("-" * 80)


def _print_verdict(trend, always, delta, lo, hi, p, months) -> None:
    print("\n" + "=" * 80)
    print("  THE REPLICATION")
    print("=" * 80)
    print(f"    trend Sharpe                    {trend.sharpe:>8.3f}")
    print(f"    same book, signal removed       {always.sharpe:>8.3f}")
    print(f"    difference                      {delta:>+8.3f}")
    print(f"    block-bootstrap 95% CI          [{lo:+.3f}, {hi:+.3f}]")
    print(f"    p                               {p:>8.3f}")
    print(f"    average live markets            "
          f"{sum(trend.live) / max(len(trend.live), 1):>8.1f}")
    print("-" * 80)
    if delta > 0 and p < 0.05:
        print("  THE EFFECT REPLICATES. On markets it was not selected on, with")
        print("  a lookback taken from the literature rather than fitted, the")
        print("  trend signal beats the identical always-on book at a p this")
        print("  sample can actually support.")
        print()
        print(f"  It is worth {delta:+.3f} Sharpe, and it costs "
              f"{trend.turnover:.1f}x turnover a year to collect.\n"
              f"  Drawdown {trend.max_dd:.1f}% against {always.max_dd:.1f}%.")
        print("  This is the first thing in this repository to survive an "
              "out-of-sample test\n  of a rule rather than of an allocation.")
    elif delta > 0:
        print(f"  Positive but not significant: {delta:+.3f} Sharpe, "
              f"p = {p:.3f}, interval\n  [{lo:+.3f}, {hi:+.3f}] containing "
              "zero. The best row of the two-asset test does not\n  reproduce "
              "strongly enough here to call it real. Read the group table "
              "below before\n  deciding what this number is -- a panel result "
              "can be one market in a costume.")
    else:
        print(f"  IT DOES NOT REPLICATE. On markets it was not chosen on the "
              f"signal is {delta:+.3f}\n  Sharpe BEHIND the same book held "
              "always-on. The 0.817 in the two-asset\n  table was the best of "
              "eight on a searched set, and this is what that usually\n  turns "
              "out to mean.")
    print("=" * 80 + "\n")


def _print_groups(per_group: dict, leave_out: dict, whole: float) -> None:
    """Where the panel result actually lives.

    The first version of this skipped any group with fewer than two markets
    "for having too few members", and thereby hid the entire finding: the panel
    effect turned out to sit in a single currency index that was never shown.
    A diagnostic that omits the small cases omits the interesting ones.
    """
    if not per_group:
        return
    print("-" * 80)
    print("  IS IT ONE ASSET CLASS WEARING A PANEL?")
    print("-" * 80)
    print(f"    {'group':<12}{'trend':>9}{'always on':>12}{'difference':>13}"
          f"{'panel without it':>19}")
    for group, (delta, t, a) in sorted(per_group.items(),
                                       key=lambda kv: -kv[1][0]):
        without = leave_out.get(group)
        cell = f"{without:>+19.3f}" if without is not None else f"{'--':>19}"
        print(f"    {group:<12}{t:>9.3f}{a:>12.3f}{delta:>+13.3f}{cell}")
    print("-" * 80)
    positive = sum(1 for d, _t, _a in per_group.values() if d > 0)
    print(f"    positive in {positive} of {len(per_group)} groups; the whole "
          f"panel is {whole:+.3f}")
    if leave_out and whole > 0:
        worst = min(leave_out.items(), key=lambda kv: kv[1])
        kept = worst[1] / whole
        # A sign test is too blunt here. Dropping one group took this panel
        # from +0.123 to +0.009 and the first version of this check called that
        # "survives", because +0.009 is still above zero. What matters is how
        # much is left, not which side of zero it lands on.
        print(f"    worst single-group removal: without {worst[0]} the panel "
              f"is {worst[1]:+.3f},\n    which is {kept:.0%} of the whole "
              "effect.")
        if kept < 0.25:
            print(f"    ONE GROUP CARRIES IT. Take {worst[0]} out and "
                  f"{1 - kept:.0%} of the effect goes with it.\n    A breadth "
                  "effect that lives in one group is not a breadth effect; it "
                  "is that\n    group, and the panel around it is decoration.")
        elif kept < 0.6:
            print(f"    Most of the effect is concentrated in {worst[0]}, "
                  "though not all of it.")
        else:
            print("    No single group carries it, which is what a breadth "
                  "effect should look like.")
    print("-" * 80 + "\n")


def _print_dividend_sensitivity(cfg, months, cost_bps, target_vol,
                                max_gross) -> None:
    print("-" * 80)
    print("  HOW MUCH IS THE DIVIDEND ASSUMPTION CARRYING?")
    print("-" * 80)
    print(f"    {'assumed':<10}{'trend':>9}{'always on':>12}{'difference':>13}")
    for pct in DIVIDEND_SENSITIVITY:
        panel = load_panel(cfg, dividend_pct=pct)
        t = simulate(panel, True, months, cost_bps, target_vol, max_gross)
        a = simulate(panel, False, months, cost_bps, target_vol, max_gross)
        mark = "  <-- default" if pct == DIVIDEND_PCT else ""
        print(f"    {pct:>6.1f}%   {t.sharpe:>9.3f}{a.sharpe:>12.3f}"
              f"{t.sharpe - a.sharpe:>+13.3f}{mark}")
    print("-" * 80)
    print("    Higher assumed dividends help the always-on book more, because "
          "it is long\n    all the time. If the difference survives 4.5%, it "
          "is not an artefact of\n    missing dividend data.")
    print("-" * 80 + "\n")
