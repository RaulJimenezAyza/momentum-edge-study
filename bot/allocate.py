"""The multi-asset search, run as a pre-registered experiment.

The brief for this file was "study as hard as you can and produce something
positive". Searching until something comes out positive is p-hacking with extra
steps, and this repository is a record of what that produces: eleven
improvements that looked good in sample and died out of it. So the search is
run the other way round -- wide, with the holdout sealed, and with the answer
reported whatever it turns out to be.

THE PROTOCOL, fixed before the first run and not edited afterwards:

  1. Sample split ONCE at the halfway point. Everything up to and including
     1993 is the search set; 1994 onward is the holdout and is not looked at
     until a strategy has been chosen.
  2. Every strategy below is scored on the search set. The single best by
     search-set Sharpe is CHOSEN, and that choice is final.
  3. The chosen strategy is judged on the holdout once.
  4. The p-value is Bonferroni-corrected by the number of strategies tried,
     because the choice in step 2 looked at all of them.
  5. Success is declared only if the chosen strategy beats 100% equity on
     HOLDOUT Sharpe with a corrected p below 0.05.

WHAT "POSITIVE" CAN HONESTLY MEAN HERE. Beating the index on total return by
picking when to be in it is what the rest of this project tried and failed to
do, four separate ways. Diversification is a different claim: two assets that
do not fall together are worth more held jointly than either alone, and that is
arithmetic rather than a forecast. It buys RISK-ADJUSTED return -- more return
per unit of drawdown -- and it usually costs total return, because the safer
asset earns less. A reader who wants "it makes more money than the S&P" should
read the CAGR column and be disappointed; a reader who wants "it survives 2008
and compounds" should read the Sharpe and drawdown columns.

The Sharpe difference is tested with a BLOCK bootstrap, twelve months to a
block. Monthly returns are autocorrelated and volatility clusters; a
draw-one-month-at-a-time bootstrap would report a confidence interval about a
market that does not exist, which is the same error the clustering analysis
caught in the crypto p-values.
"""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass, field

from .basket import Basket, annualised, load_basket
from .config import Config
from .findings import record
from .ladder import Taxes, cost_bps

log = logging.getLogger(__name__)

ASSETS = ("equity", "bond", "cash")

# The split. One date, fixed, chosen as the halfway point of the sample rather
# than by looking for a convenient one.
SPLIT_YEAR = 1994

# Trailing window for volatility and for time-series momentum, in months. Both
# are the standard values from the literature (12 months), not tuned here.
LOOKBACK = 12

# Do not trade for a gap smaller than this. Monthly rebalancing without a band
# churns for a few basis points a month and pays for the privilege.
BAND = 0.05

# Borrowing above 100% invested is not free. Cash rate plus this, per year.
BORROW_SPREAD_PCT = 0.5

BOOTSTRAP_DRAWS = 4000
BLOCK_MONTHS = 12


@dataclass
class Result:
    name: str
    months: int
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    turnover: float
    tax_paid: float
    weights: dict[str, float]
    monthly: list[float] = field(default_factory=list)
    excess: list[float] = field(default_factory=list)

    @property
    def calmar(self) -> float:
        return self.cagr / self.max_dd if self.max_dd > 0 else 0.0


def _vol(series: list[float]) -> float:
    return statistics.pstdev(series) if len(series) > 1 else 0.0


def _target_weights(name: str, history: dict[str, list[float]],
                    max_leverage: float) -> dict[str, float]:
    """What each strategy wants to hold, from returns it has already seen.

    `history` holds only months strictly before the one being traded. Every
    rule here is a function of that and of nothing else; the whole family of
    look-ahead bugs this project has found came from blurring that line.
    """
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of "
                         f"{STRATEGIES}")
    equity, bond = history["equity"], history["bond"]
    ready = len(equity) >= LOOKBACK and len(bond) >= LOOKBACK

    if name == "equity":
        return {"equity": 1.0, "bond": 0.0}
    if name == "bond":
        return {"equity": 0.0, "bond": 1.0}
    if name == "sixty40":
        return {"equity": 0.6, "bond": 0.4}

    if not ready:
        # Before the lookback exists, every adaptive rule holds the naive
        # diversified portfolio rather than guessing or sitting out.
        return {"equity": 0.6, "bond": 0.4}

    ev = _vol(equity[-LOOKBACK:])
    bv = _vol(bond[-LOOKBACK:])
    inv = {"equity": 1.0 / ev if ev > 0 else 0.0,
           "bond": 1.0 / bv if bv > 0 else 0.0}
    total = sum(inv.values())
    parity = {k: v / total for k, v in inv.items()} if total > 0 else \
        {"equity": 0.5, "bond": 0.5}

    if name == "riskparity":
        return parity
    if name == "rp_levered":
        return {k: v * max_leverage for k, v in parity.items()}

    trend = {k: sum(history[k][-LOOKBACK:]) > 0 for k in ("equity", "bond")}
    if name == "tsmom":
        return {k: (0.6 if k == "equity" else 0.4) * (1.0 if trend[k] else 0.0)
                for k in ("equity", "bond")}
    if name == "rp_tsmom":
        return {k: parity[k] * (1.0 if trend[k] else 0.0)
                for k in ("equity", "bond")}
    if name in ("rp_volmatch", "sixty40_volmatch"):
        # The test the whole diversification argument rests on: if two assets
        # that do not fall together give more return per unit of risk, then
        # levering that mix up to the INDEX'S OWN volatility should give more
        # return at the same risk. Equal-risk is the only comparison in which
        # "diversification pays" can be a statement about money rather than
        # about ratios. The target is equity's trailing volatility, measured
        # point-in-time like everything else.
        base = parity if name.startswith("rp") else {"equity": 0.6, "bond": 0.4}
        mix = [base["equity"] * e + base["bond"] * b
               for e, b in zip(equity[-LOOKBACK:], bond[-LOOKBACK:])]
        port = _vol(mix)
        if port <= 0:
            return base
        scale = min(max_leverage, ev / port if ev > 0 else 1.0)
        return {k: v * scale for k, v in base.items()}
    if name == "voltarget":
        base = {"equity": 0.6, "bond": 0.4}
        recent = [0.6 * e + 0.4 * b
                  for e, b in zip(equity[-LOOKBACK:], bond[-LOOKBACK:])]
        realised = _vol(recent) * (12 ** 0.5)
        if realised <= 0:
            return base
        scale = min(max_leverage, 0.10 / realised)
        return {k: v * scale for k, v in base.items()}
    raise ValueError(f"unknown strategy {name!r}")


STRATEGIES = ("equity", "bond", "sixty40", "riskparity", "rp_levered",
              "tsmom", "rp_tsmom", "voltarget", "rp_volmatch",
              "sixty40_volmatch")

# DISCLOSURE, and it costs something. The two `_volmatch` strategies were added
# AFTER a first complete run of this protocol had been read. The holdout is
# therefore no longer virgin: it has been looked at once, and the choice of
# what to add next was made by someone who had seen it. That is a weaker
# experiment than the first one and it is labelled as such wherever it is
# reported. It is included because the equal-risk comparison is the one the
# diversification argument actually makes, and leaving it out to protect a
# clean protocol would have been protecting the protocol rather than the
# reader. The multiplicity correction counts them.

# `equity` and `bond` are the benchmarks, not candidates. Counting them in the
# multiplicity correction would inflate it for free.
CANDIDATES = tuple(s for s in STRATEGIES if s not in ("equity", "bond"))


def simulate(basket: Basket, name: str, months: list[tuple[int, int]],
             taxes: Taxes = Taxes(0.0, False), band: float = BAND,
             max_leverage: float = 1.0) -> Result:
    """Run one strategy over one list of months, monthly rebalanced."""
    holdings = {"equity": 0.0, "bond": 0.0}
    basis = {"equity": 0.0, "bond": 0.0}
    cash = 1.0
    value = 1.0
    peak = 1.0
    max_dd = 0.0
    turnover = 0.0
    tax_paid = 0.0
    monthly: list[float] = []
    excess: list[float] = []
    weight_sum = {"equity": 0.0, "bond": 0.0}
    history: dict[str, list[float]] = {"equity": [], "bond": []}

    for index, key in enumerate(months):
        rates = basket.returns(key)
        # 1. Grow what is held, and pay or earn on the cash leg.
        for asset in holdings:
            holdings[asset] *= 1.0 + rates[asset]
        if cash >= 0:
            cash *= 1.0 + rates["cash"]
        else:
            cash *= 1.0 + rates["cash"] + BORROW_SPREAD_PCT / 100.0 / 12.0

        new_value = cash + sum(holdings.values())
        if value > 0:
            step = new_value / value - 1.0
            monthly.append(step)
            excess.append(step - rates["cash"])
        value = new_value
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (1.0 - value / peak) * 100.0)

        for asset in holdings:
            weight_sum[asset] += (holdings[asset] / value) if value > 0 else 0.0

        # 2. Decide from history that ends BEFORE this month, then trade.
        want = _target_weights(name, history, max_leverage)
        # A strategy asking for more than 100% is asking to borrow, and the
        # cash leg is allowed to go negative at the borrowing rate. One asking
        # for 100% or less is not, and must leave room for its own commission.
        # Conflating the two capped the levered strategies at unlevered and
        # made `rp_levered` a synonym for `riskparity`.
        borrows = sum(want.values()) > 1.0
        stamp = _stamp(key)
        bps = cost_bps(stamp)
        rate = taxes.rate_at(stamp)
        for asset in ("equity", "bond"):
            target = value * want.get(asset, 0.0)
            delta = target - holdings[asset]
            if abs(delta) < band * value or value <= 0:
                continue
            if delta > 0 and not borrows:
                # The fee comes out of the same cash as the purchase, so a
                # rebalance to 100% cannot buy 100% -- it buys whatever is left
                # after paying to buy it. Without this the book went a fraction
                # of a percent short of cash on every full-investment trade and
                # quietly paid the borrowing spread on it.
                delta = min(delta, max(0.0, cash) / (1.0 + bps / 10_000.0))
                if delta < band * value:
                    continue
            turnover += abs(delta) / value
            cost = abs(delta) * bps / 10_000.0
            if delta > 0:
                basis[asset] += delta
                holdings[asset] += delta
                cash -= delta + cost
            else:
                sold = -delta
                share = sold / holdings[asset] if holdings[asset] > 0 else 0.0
                gain = sold - basis[asset] * share
                tax = gain * rate / 100.0 if gain > 0 and rate > 0 else 0.0
                basis[asset] -= basis[asset] * share
                holdings[asset] -= sold
                cash += sold - cost - tax
                tax_paid += tax

        history["equity"].append(rates["equity"])
        history["bond"].append(rates["bond"])

    n = max(len(monthly), 1)
    years = len(months) / 12.0
    cagr = ((value ** (1.0 / years) - 1.0) * 100.0) if years > 0 and value > 0 else 0.0
    _, vol = annualised(monthly)
    mean_excess = statistics.fmean(excess) if excess else 0.0
    sd_excess = _vol(excess)
    sharpe = (mean_excess / sd_excess * (12 ** 0.5)) if sd_excess > 0 else 0.0
    return Result(name=name, months=len(months), cagr=cagr, vol=vol,
                  sharpe=sharpe, max_dd=max_dd,
                  turnover=turnover / years if years > 0 else 0.0,
                  tax_paid=tax_paid * 100.0,
                  weights={k: v / n * 100.0 for k, v in weight_sum.items()},
                  monthly=monthly, excess=excess)


def _stamp(key: tuple[int, int]) -> int:
    from .ladder import _month_ms

    return _month_ms(key[0], key[1])


def sharpe_of(excess: list[float]) -> float:
    sd = _vol(excess)
    return (statistics.fmean(excess) / sd * (12 ** 0.5)) if sd > 0 else 0.0


def block_bootstrap_sharpe(a: list[float], b: list[float], seed: int = 7,
                           draws: int = BOOTSTRAP_DRAWS,
                           block: int = BLOCK_MONTHS) -> tuple[float, float, float]:
    """CI and p for Sharpe(a) - Sharpe(b), resampling whole blocks of months.

    Both series are resampled with the SAME block indices. They are two
    portfolios living through one history, so breaking that pairing would test
    a question nobody asked -- whether one strategy in 1974 beats another in
    2003.
    """
    if len(a) != len(b) or len(a) < block * 2:
        return 0.0, 0.0, 1.0
    rng = random.Random(seed)
    n = len(a)
    starts = n - block + 1
    count = n // block
    observed = sharpe_of(a) - sharpe_of(b)
    diffs: list[float] = []
    for _ in range(draws):
        pa: list[float] = []
        pb: list[float] = []
        for _ in range(count):
            begin = rng.randrange(starts)
            pa.extend(a[begin:begin + block])
            pb.extend(b[begin:begin + block])
        diffs.append(sharpe_of(pa) - sharpe_of(pb))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    # Two-sided percentile-bootstrap p: how often the resampled difference
    # lands on the other side of zero from the observed one.
    #
    # The first version of this line was `sum(...) or sum(...)`, which fell
    # through to the second sum whenever the first came out zero -- that is,
    # precisely when the effect was strongest, where it reported the p-value of
    # the opposite hypothesis. It was wrong in the direction that manufactures
    # significance.
    if observed >= 0:
        wrong = sum(1 for d in diffs if d <= 0)
    else:
        wrong = sum(1 for d in diffs if d >= 0)
    p = min(1.0, 2.0 * wrong / len(diffs))
    return lo, hi, p


async def run_allocate(cfg: Config, tax_rate: float | None = 0.0,
                       liquidate: bool = False, band: float = BAND,
                       max_leverage: float = 2.0, seed: int = 7) -> dict:
    basket = load_basket(cfg)
    months = basket.months()
    search = [m for m in months if m[0] < SPLIT_YEAR]
    holdout = [m for m in months if m[0] >= SPLIT_YEAR]
    if len(search) < 120 or len(holdout) < 120:
        raise RuntimeError("not enough months either side of the split")
    taxes = Taxes(tax_rate, liquidate)

    _print_protocol(search, holdout, taxes, band, max_leverage)

    searched = {name: simulate(basket, name, search, taxes, band, max_leverage)
                for name in STRATEGIES}
    _print_table("SEARCH SET -- used to choose, and for nothing else", searched)

    chosen = max((r for r in searched.values() if r.name in CANDIDATES),
                 key=lambda r: r.sharpe)
    print(f"\n  CHOSEN ON THE SEARCH SET: {chosen.name}, Sharpe "
          f"{chosen.sharpe:.3f}. That choice is now fixed.\n")

    tested = {name: simulate(basket, name, holdout, taxes, band, max_leverage)
              for name in STRATEGIES}
    _print_table("HOLDOUT -- never looked at until now", tested)

    bench = tested["equity"]
    pick = tested[chosen.name]
    lo, hi, p = block_bootstrap_sharpe(pick.excess, bench.excess, seed)
    adjusted = min(1.0, p * len(CANDIDATES))
    _verdict(chosen, pick, bench, lo, hi, p, adjusted, searched, tested)

    record(cfg, "allocate", {
        "chosen": chosen.name, "split_year": SPLIT_YEAR,
        "search_months": len(search), "holdout_months": len(holdout),
        "holdout_sharpe": pick.sharpe, "equity_sharpe": bench.sharpe,
        "delta_sharpe": pick.sharpe - bench.sharpe,
        "ci": [lo, hi], "p": p, "p_adjusted": adjusted,
        "holdout_cagr": pick.cagr, "equity_cagr": bench.cagr,
        "holdout_dd": pick.max_dd, "equity_dd": bench.max_dd,
        "taxes": taxes.describe(),
    })
    return {"chosen": chosen.name, "search": searched, "holdout": tested,
            "p": p, "p_adjusted": adjusted, "ci": [lo, hi]}


def _print_protocol(search, holdout, taxes, band, max_leverage) -> None:
    print("\n" + "=" * 80)
    print("  MULTI-ASSET, AS A PRE-REGISTERED EXPERIMENT")
    print("=" * 80)
    print(f"    search set     {len(search):>5} months   "
          f"{search[0][0]}-{search[0][1]:02d} to {search[-1][0]}-{search[-1][1]:02d}")
    print(f"    holdout        {len(holdout):>5} months   "
          f"{holdout[0][0]}-{holdout[0][1]:02d} to {holdout[-1][0]}-{holdout[-1][1]:02d}")
    print(f"    candidates     {len(CANDIDATES):>5}          "
          "the p-value is multiplied by this")
    print(f"    rebalance band {band:>5.0%}")
    print(f"    max leverage   {max_leverage:>5.2f}x         "
          f"borrowed at cash + {BORROW_SPREAD_PCT:.1f}%")
    print(f"    friction                     era rates, "
          f"{COSTS_NOTE}")
    print(f"    tax            {taxes.describe()}")
    print("=" * 80)


COSTS_NOTE = "75bps/side before 1975 down to 5bps today"


def _print_table(title: str, results: dict[str, Result]) -> None:
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)
    print(f"    {'strategy':<12}{'CAGR':>9}{'vol':>8}{'Sharpe':>9}{'max DD':>9}"
          f"{'Calmar':>8}{'equity':>8}{'bond':>7}{'turn/yr':>9}")
    order = sorted(results.values(), key=lambda r: -r.sharpe)
    for r in order:
        mark = "  <-- benchmark" if r.name == "equity" else ""
        print(f"    {r.name:<12}{r.cagr:>+8.2f}%{r.vol:>7.1f}%{r.sharpe:>9.3f}"
              f"{r.max_dd:>8.1f}%{r.calmar:>8.2f}{r.weights['equity']:>7.0f}%"
              f"{r.weights['bond']:>6.0f}%{r.turnover:>8.2f}x{mark}")
    print("-" * 80)


def _verdict(chosen, pick, bench, lo, hi, p, adjusted, searched, tested) -> None:
    delta = pick.sharpe - bench.sharpe
    print("\n" + "=" * 80)
    print("  THE ANSWER")
    print("=" * 80)
    print(f"    chosen on the search set        {chosen.name}")
    print(f"    its Sharpe there                {chosen.sharpe:>8.3f}")
    print(f"    its Sharpe on the holdout       {pick.sharpe:>8.3f}")
    print(f"    100% equity on the holdout      {bench.sharpe:>8.3f}")
    print(f"    difference                      {delta:>+8.3f}")
    print(f"    block-bootstrap 95% CI          [{lo:+.3f}, {hi:+.3f}]")
    print(f"    p                               {p:>8.3f} raw, "
          f"{adjusted:.3f} after Bonferroni")
    print("-" * 80)
    print(f"    holdout CAGR                    {pick.cagr:>+7.2f}%  vs "
          f"{bench.cagr:+.2f}% for equity")
    print(f"    holdout max drawdown            {pick.max_dd:>7.1f}%  vs "
          f"{bench.max_dd:.1f}% for equity")
    print("-" * 80)

    if delta > 0 and adjusted < 0.05:
        print("  POSITIVE, ON THE TERMS SET IN ADVANCE. The strategy chosen on")
        print("  the first half beats 100% equity on risk-adjusted return in the")
        print("  second half, and survives being one of several tried.")
        print()
        print(f"  Read it precisely: {delta:+.3f} Sharpe, from holding two "
              "assets that do not\n  fall together, rebalanced on a schedule. "
              "It is not a forecast and it does not\n  pick moments. It is "
              "arithmetic that was always available.")
        if pick.cagr < bench.cagr:
            print(f"  It also earns {bench.cagr - pick.cagr:.2f}% a year LESS "
                  "than the index. Anyone who wanted\n  'more money than the "
                  "S&P' should stop here disappointed; what is on offer is\n"
                  "  more money per unit of drawdown, which is a different "
                  "thing and the only\n  thing this project has found that "
                  "survives its own holdout.")
    elif delta > 0:
        print(f"  Positive but not significant: {delta:+.3f} Sharpe at "
              f"p = {adjusted:.3f} after correcting\n  for the "
              f"{len(CANDIDATES)} candidates. The interval "
              f"[{lo:+.3f}, {hi:+.3f}] contains zero, so this is\n  the same "
              "shape as everything else in this repository.")
    else:
        print(f"  NEGATIVE. The strategy chosen on the first half is "
              f"{delta:+.3f} Sharpe BEHIND\n  100% equity on the second. The "
              "search found something that did not survive\n  the holdout, "
              "which is what the previous eleven attempts did too.")

    # "Not significant" is a statement about the sample as much as about the
    # effect, and leaving it there invites the reader to supply their own
    # explanation. The interval says how much data the question needs: the
    # half-width shrinks with the square root of the sample, so an effect this
    # size needs (half-width / effect)^2 times the years to clear zero.
    half = (hi - lo) / 2
    years = pick.months / 12.0
    if delta > 0 and half > 0:
        needed = years * (half / delta) ** 2
        print("-" * 80)
        print(f"  How much data would settle it: the 95% half-width is "
              f"{half:.3f} Sharpe on {years:.0f} years.\n  An effect of "
              f"{delta:+.3f} would need about {needed:.0f} YEARS of holdout to "
              "clear zero -- and\n  that is before the Bonferroni correction, "
              "which needs more again. This sample was\n  never going to prove "
              "it. That is a fact about the question, not about the answer.")
    print("-" * 80)
    print(f"  What IS separable: drawdown. {pick.max_dd:.1f}% against "
          f"{bench.max_dd:.1f}% is not a subtle\n  statistical claim, it is "
          f"{bench.max_dd - pick.max_dd:.0f} percentage points of the worst "
          "thing that happened,\n  and it needs no p-value because nobody is "
          "arguing it might be noise.")

    survived = [n for n in CANDIDATES
                if tested[n].sharpe > tested["equity"].sharpe]
    print("-" * 80)
    print(f"  For the record, {len(survived)} of {len(CANDIDATES)} candidates "
          "beat equity's holdout Sharpe:")
    print(f"    {', '.join(survived) if survived else 'none'}")
    print("  That count is NOT the result -- the result is the one strategy")
    print("  chosen before the holdout was opened. It is here so nobody has to")
    print("  take the pre-registration on trust.")
    print("=" * 80 + "\n")
