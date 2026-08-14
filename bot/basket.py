"""Monthly total-return series for a multi-asset test, and their validation.

Everything before this file tested one instrument at a time and found nothing:
the intraday edge cannot pay its toll, the viral ladder is an allocation in a
timing costume, and a 200-day trend filter buys lower drawdown with about a
point of CAGR. The one effect left that is STRUCTURAL rather than predictive is
diversification -- two assets that do not fall together are worth more held
jointly than either is alone, and that is arithmetic, not a forecast.

Testing it needs a bond series, and free daily bond history does not exist. So
one is built from yields and then checked against a real fund, because a
synthetic series nobody validated is exactly the kind of input this project has
been bitten by.

WHAT IS BUILT, and what each choice cost:

  equity  ^GSPC month-end closes plus the measured dividend yield from the
          Shiller series. Price-only would understate equities by ~2%/yr, which
          would flatter every bond-heavy strategy in the comparison.
  bond    A rolling par bond priced off its own yield:
              r = y/12 - D*dy + C*dy^2/2
          from MONTH-END ^TNX. Validated against IEF over 289 overlapping
          months: correlation 0.989, median monthly error 0.22pp.
  cash    ^IRX, the 13-week bill, divided by twelve.

WHY 1962 AND NOT 1871. Shiller carries a long rate back to 1871 and the same
formula runs on it, but Shiller's rate is a MONTHLY AVERAGE -- the same
convention that `vendors.py` found in his price column. Over 1985-2023 the
average-based series reproduces the mean well (+6.03%/yr against +5.83%) and
understates the volatility badly (6.43% against 7.62%), because averaging
smooths. Understated bond volatility makes bonds look better in exactly the
risk-adjusted terms this test is about, so the extra ninety years would have
been ninety years of evidence bent toward the conclusion. ^TNX starts in 1962
and that is where this starts.

WHAT IS NOT MODELLED, and which way it cuts:

  * Roll-down. A constant-maturity bond gains as it ages down a sloping curve.
    The synthetic series runs ~0.7%/yr BELOW IEF for this reason, whatever
    duration is used, so bonds are understated -- against the diversification
    claim, which is the right direction for a claim this file is trying to make.
  * Fund fees and tracking error on the instruments a real investor would use.
  * Tax. Handled by the allocator, not here.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from .config import Config
from .exchange import Candle
from .shiller import dividend_yield_path, load_shiller, month_of
from .yahoo import load_equity_history

log = logging.getLogger(__name__)

# Duration of the synthetic bond, in years. 8.5 rather than 10 because that is
# what reproduces IEF's realised volatility (6.62% a year) over the overlap;
# a 10-year par bond runs at 7.49% and would overstate the risk of the asset
# whose diversifying power is the thing under test.
BOND_MATURITY = 8.5

# Series that must be present for the basket to mean anything.
EQUITY = "^GSPC"
BOND_YIELD = "^TNX"
CASH_YIELD = "^IRX"

# Real funds to check the synthetic series against, loaded with ADJUSTED
# closes so they are total-return like the series they are validating. The
# first version of this used the project's default unadjusted loader, and both
# funds came out 2-3%/yr below the built series -- which was not an error in
# the series, it was the dividend yield of the benchmark going missing.
VALIDATORS = {"equity": "SPY", "bond": "IEF"}


@dataclass
class Basket:
    """Aligned monthly total returns, one entry per (year, month)."""
    equity: dict[tuple[int, int], float] = field(default_factory=dict)
    bond: dict[tuple[int, int], float] = field(default_factory=dict)
    cash: dict[tuple[int, int], float] = field(default_factory=dict)

    def months(self) -> list[tuple[int, int]]:
        """Only months where every asset has a return. No partial rows."""
        shared = set(self.equity) & set(self.bond) & set(self.cash)
        return sorted(shared)

    def returns(self, key: tuple[int, int]) -> dict[str, float]:
        return {"equity": self.equity[key], "bond": self.bond[key],
                "cash": self.cash[key]}


def month_end(candles: list[Candle]) -> dict[tuple[int, int], float]:
    """(year, month) -> the last close in that month."""
    out: dict[tuple[int, int], float] = {}
    for candle in sorted(candles, key=lambda c: c.open_time):
        out[month_of(candle.open_time)] = candle.close
    return out


def bond_returns(yields: dict[tuple[int, int], float],
                 maturity: float = BOND_MATURITY) -> dict[tuple[int, int], float]:
    """Monthly total return of a rolling par bond, from its yield alone.

    Coupon accrual plus the price move implied by the change in yield, to
    second order. Validated against IEF in `validate`; the one term missing is
    roll-down, which makes this series ~0.7%/yr too low.
    """
    keys = sorted(yields)
    out: dict[tuple[int, int], float] = {}
    for prev, now in zip(keys, keys[1:]):
        y = yields[prev] / 100.0
        dy = (yields[now] - yields[prev]) / 100.0
        if y <= 0:
            continue
        duration = (1 - (1 + y) ** -maturity) / y
        convexity = maturity * (maturity + 1) / (1 + y) ** 2 / 12
        out[now] = y / 12 - duration * dy + 0.5 * convexity * dy * dy
    return out


def price_returns(closes: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    keys = sorted(closes)
    return {now: closes[now] / closes[prev] - 1.0
            for prev, now in zip(keys, keys[1:]) if closes[prev] > 0}


def load_basket(cfg: Config) -> Basket:
    """Build the three series, aligned on months where all three exist."""
    history = load_equity_history([EQUITY, BOND_YIELD, CASH_YIELD], "1d",
                                  40_000.0, cfg.path("data/yahoo"))
    for name in (EQUITY, BOND_YIELD, CASH_YIELD):
        if not history.get(name):
            raise RuntimeError(f"the basket needs {name} and it did not load")

    equity_price = price_returns(month_end(history[EQUITY]))
    # Dividends are the difference between an index and an investment in it.
    # The measured path is used where it exists; months beyond it get nothing
    # rather than a guess, which understates equities at the very end of the
    # sample and is stated rather than patched.
    yield_path = dividend_yield_path(load_shiller(cfg.path("data/shiller")))
    equity = {k: r + yield_path.get(k, 0.0) / 100.0 / 12.0
              for k, r in equity_price.items()}

    bond = bond_returns(month_end(history[BOND_YIELD]))
    bills = month_end(history[CASH_YIELD])
    cash = {k: v / 100.0 / 12.0 for k, v in bills.items() if v > 0}

    basket = Basket(equity=equity, bond=bond, cash=cash)
    months = basket.months()
    if not months:
        raise RuntimeError("no month has all three series")
    log.info("basket: %d months, %d-%02d to %d-%02d", len(months),
             months[0][0], months[0][1], months[-1][0], months[-1][1])
    return basket


def annualised(series: list[float]) -> tuple[float, float]:
    """(mean %/yr, volatility %/yr) from monthly returns."""
    if len(series) < 2:
        return 0.0, 0.0
    return (statistics.fmean(series) * 1200.0,
            statistics.pstdev(series) * (12 ** 0.5) * 100.0)


def correlation(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa, sb = statistics.pstdev(a), statistics.pstdev(b)
    if sa == 0 or sb == 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    return cov / (sa * sb)


async def run_basketcheck(cfg: Config) -> dict:
    """Do the built series match the funds a real investor would have held?"""
    basket = load_basket(cfg)
    funds = load_equity_history(list(VALIDATORS.values()), "1d", 40_000.0,
                                cfg.path("data/yahoo"), auto_adjust=True)
    built = {"equity": basket.equity, "bond": basket.bond}

    print("\n" + "=" * 80)
    print("  THE BASKET, CHECKED AGAINST REAL FUNDS")
    print("=" * 80)
    months = basket.months()
    print(f"    months with all three series {len(months):>8,}   "
          f"{months[0][0]}-{months[0][1]:02d} to {months[-1][0]}-{months[-1][1]:02d}")
    print(f"    bond duration                {BOND_MATURITY:>8.1f}y  "
          "calibrated to IEF's realised volatility")
    print("-" * 80)
    print(f"    {'series':<10}{'fund':<7}{'n':>5}{'built %/yr':>12}"
          f"{'fund %/yr':>11}{'built vol':>11}{'fund vol':>10}{'corr':>8}")

    out: dict[str, dict] = {}
    for name, ticker in VALIDATORS.items():
        candles = funds.get(ticker)
        if not candles:
            log.warning("%s did not load; %s is unvalidated", ticker, name)
            continue
        real = price_returns(month_end(candles))
        mine = built[name]
        shared = sorted(set(real) & set(mine))
        if len(shared) < 24:
            continue
        a = [mine[k] for k in shared]
        b = [real[k] for k in shared]
        am, av = annualised(a)
        bm, bv = annualised(b)
        corr = correlation(a, b)
        print(f"    {name:<10}{ticker:<7}{len(shared):>5}{am:>+11.2f}%"
              f"{bm:>+10.2f}%{av:>10.2f}%{bv:>9.2f}%{corr:>8.3f}")
        out[name] = {"fund": ticker, "months": len(shared), "built_mean": am,
                     "fund_mean": bm, "built_vol": av, "fund_vol": bv,
                     "corr": corr}
    print("-" * 80)

    keys = basket.months()
    eq = [basket.equity[k] for k in keys]
    bd = [basket.bond[k] for k in keys]
    ch = [basket.cash[k] for k in keys]
    for label, series in (("equity", eq), ("bond", bd), ("cash", ch)):
        mean, vol = annualised(series)
        print(f"    {label:<10}{mean:>+9.2f}%/yr{vol:>10.2f}% vol")
    rho = correlation(eq, bd)
    print(f"    equity/bond correlation over the whole sample: {rho:+.3f}")
    print("-" * 80)
    if abs(rho) < 0.3:
        print("  The two risky assets are close to uncorrelated, which is the")
        print("  precondition for the whole multi-asset test. It is not a")
        print("  forecast that they stay that way -- 2022 is in this sample and")
        print("  they fell together.")
    else:
        print(f"  The two assets are correlated at {rho:+.2f} over this sample.")
        print("  Diversification has less to work with than the textbook case.")
    print("=" * 80 + "\n")

    from .findings import record
    record(cfg, "basketcheck", {"months": len(keys), "equity_bond_corr": rho,
                                **{f"{k}_corr": v["corr"] for k, v in out.items()}})
    return {"validation": out, "corr": rho, "months": len(keys)}
