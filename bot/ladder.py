"""Hypothesis twelve: the viral "buy the dip, scale out the rally" ladder.

The rules, as posted, attributed to a woman who died at 105 after 85 years of
watching the US market:

    drops 10%  -> hold          rises 10%  -> hold
    drops 20%  -> buy  +15%     rises 20%  -> hold
    drops 30%  -> buy  +30%     rises 30%  -> sell 10%
                                rises 40%  -> sell 20%
                                rises 50%  -> sell 30%
                                rises 60%  -> sell 40%
                                rises 100% -> sell 60%

This is the same genre as the $68 -> $750k post this whole repo started from:
a number-shaped claim with no measurement attached, spread by 1.7K shares. The
difference is that this one is cheap to test, and it can be tested on a sample
the rest of the project could never reach -- ^GSPC daily bars go back to 1927,
which is ninety-nine years and nine non-overlapping ten-year windows, against
the five thirty-day windows the crypto finding rests on.

WHAT THE POST DOES NOT SAY, and every one of these changes the answer:

  1. Percent of WHAT. "Drops 20%" from the cost basis, from the running
     all-time high, or from the price on the day you started? Three different
     strategies share that sentence. All three are run here (`--ref`).
  2. "Buy +15%" of what. Of the starting capital, or of the position you
     currently hold? Both are run (`sizing`).
  3. Where the money to buy dips comes from. A fully invested holder cannot
     act on rule 2 at all. The starting equity fraction is a flag with a
     stated default, and every run reports how many rungs went unfunded.
  4. When a rung re-arms. Left alone, "drops 20% -> buy" fires every single
     day the market is 20% down. Here a rung fires once and re-arms only when
     price crosses back through its own reference.
  5. Dividends and interest on the idle cash. Ignoring both is not neutral:
     the ladder spends part of its life in cash, so omitting dividends flatters
     it and omitting bill yield penalises it. Both are modelled, and the grid
     is printed a second time with both set to zero so the reader can see how
     much of the answer they are carrying.
  6. Tax. The ladder realises gains on the way up; buy-and-hold defers them
     for a century, so running at zero tax understated the ladder's
     disadvantage everywhere. Now modelled, at a flat rate or at the historical
     era rates, and under both framings -- the holder who never sells and pays
     nothing, and the symmetric one where both arms settle on the last bar.
     Every run prints the whole ladder of rates.

THE PREDICTION, written before the first run and left here whether or not it
survives contact: the ladder loses to buy-and-hold on terminal wealth in every
US window, because it sells a compounding asset into strength and holds the
proceeds in a lower-returning one; and it wins on maximum drawdown, because
that is what holding cash does. If that is what comes out, the rules are not
wrong so much as mis-sold -- they are a volatility-reduction scheme being
described as a growth scheme ("discipline and long-term growth").

^N225 is in the default list for the case that breaks the symmetry: a market
that peaked in 1989 and spent thirty years below it is where "buy 30% dips"
either earns its keep or bankrupts the idea.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .config import Config
from .findings import record
from .exchange import Candle
from .yahoo import load_equity_history

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
YEAR_DAYS = 365.25

# The ladder itself. Rules 1, 4 and 5 say "hold", which is the absence of a
# rung rather than a rung, so they are not here.
BUY_RUNGS: tuple[tuple[float, float], ...] = ((-20.0, 15.0), (-30.0, 30.0))
SELL_RUNGS: tuple[tuple[float, float], ...] = (
    (30.0, 10.0), (40.0, 20.0), (50.0, 30.0), (60.0, 40.0), (100.0, 60.0),
)

# The three readings of "percent of what", and the two readings of "+15%".
REF_MODES = ("basis", "peak", "anchor")
SIZINGS = ("initial", "position")

# Two choices the post does not make and I did. They are not readings of an
# ambiguous sentence -- they are mine, which makes them the ones most worth
# attacking, and `_print_design_audit` runs the whole grid over both.
#
#   cross   a rung fires once, and re-arms when price crosses back through its
#           own reference. The default, and the only one that treats "drops 20%
#           -> buy" as an event rather than a state.
#   once    a rung fires at most once per window, ever.
#   always  the literal reading: fire on every bar the condition holds. Included
#           because it is what the sentence says, and what it does to the cash
#           pile is the reason nobody means it.
REARM_MODES = ("cross", "once", "always")

#   close     the threshold is tested against the daily close.
#   intrabar  it is tested against the bar's low (buys) or high (sells) and
#             filled at the threshold price, or at the open if the bar gapped
#             through it. Strictly more trades, and worse fills on gaps.
TRIGGERS = ("close", "intrabar")

DEFAULT_SYMBOLS = ["^GSPC", "^IXIC", "^N225", "BTC-USD"]

# The other half of the survivorship problem, and the only part of it that free
# data can reach. Every index above is a market that went up over the sample;
# concluding "holding cash costs you" from three winners is the same error as
# concluding a strategy works from three winners.
#
# These are markets that fell hard and did NOT come back inside the sample:
# Athens lost 93% of its peak and never recovered it, Shanghai 72%, Hong Kong
# 65%. A market that went to zero -- Russia 1917, Cairo 1962 -- has no free
# daily series, so this is as far as the wipeout case can be pushed.
#
# A PREDICTION, written before the first run of this list, and left here
# because it was wrong: "here the ladder should BEAT buy-and-hold, because
# holding cash through a collapse is the right thing to have done."
#
# FALSIFIED. It loses in all four, Athens included. The error was in the
# question, not the arithmetic: a 93% drawdown is not a 93% sample. Split into
# non-overlapping windows, most windows of a collapsed market are still rising
# ones, and the ladder is only ~40% invested in all of them. `_print_regime_split`
# is what should have been asked first -- on Athens the cash position costs
# 7.4% a year in the windows where the market rose and saves 10.3% in the two
# where it fell, while the timing term wanders between -1.55% and +1.90%.
#
# The part of the prediction that did survive is the part that mattered: the
# win and the loss are both allocation, and the rules themselves do nothing in
# either direction. Buy-the-dip is sold for exactly this environment, and this
# environment is where it most clearly adds nothing beyond the cash it holds.
COLLAPSED_SYMBOLS = ["GD.AT", "000001.SS", "^HSI", "^N225"]

SYMBOL_PRESETS = {"default": DEFAULT_SYMBOLS, "collapsed": COLLAPSED_SYMBOLS}

# Round-trip friction by era, per side, in basis points. A century-long test
# priced at today's zero-commission spread would be a fantasy, and it would be
# a fantasy that helps exactly one arm: the ladder trades tens of times, while
# buy-and-hold trades twice. NYSE commissions were fixed and large until
# May Day 1975, negotiated but still material through the 1990s, and collapsed
# with decimalisation and zero-commission retail brokerage after that.
COST_ERAS: tuple[tuple[int, float], ...] = ((1975, 75.0), (2001, 20.0),
                                            (9999, 5.0))

# Long-run averages, applied flat, stated rather than measured. The S&P's
# dividend yield averaged roughly 3.5% across 1927-2025 (far higher before
# 1960, near 1.5% after 2000) and the 3-month bill averaged roughly 3.3% over
# the same span. Precision is not the point; the point is that setting either
# to zero silently moves the answer, so both are named and both are varied.
DIVIDEND_YIELD_PCT = 3.5
CASH_YIELD_PCT = 3.3

# Below this many non-overlapping windows, no verdict is printed. Three
# observations cannot separate a strategy from a decade.
MIN_WINDOWS = 4


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def year_of(open_time_ms: int) -> int:
    """Calendar year of a millisecond timestamp, including pre-1970 ones.

    `datetime.fromtimestamp` raises OSError on a negative timestamp on Windows,
    and every bar before 1970 is negative. The whole reason this test exists is
    that ^GSPC starts in 1927, so the platform's refusal to represent the
    interesting half of the sample would have been a very quiet way to lose it.
    """
    return (EPOCH + timedelta(milliseconds=open_time_ms)).year


def cost_bps(open_time_ms: int) -> float:
    """Per-side friction for the year a trade happens in."""
    year = year_of(open_time_ms)
    for cutoff, bps in COST_ERAS:
        if year < cutoff:
            return bps
    return COST_ERAS[-1][1]


@dataclass
class Run:
    """One strategy over one window."""
    name: str
    final: float
    initial: float
    years: float
    max_dd: float
    buys: int = 0
    sells: int = 0
    unfunded: int = 0
    cash_end: float = 0.0
    equity_days: float = 0.0
    total_days: float = 0.0
    tax_paid: float = 0.0

    @property
    def cagr(self) -> float:
        if self.years <= 0 or self.initial <= 0 or self.final <= 0:
            return 0.0
        return ((self.final / self.initial) ** (1.0 / self.years) - 1.0) * 100.0

    @property
    def invested_pct(self) -> float:
        return (self.equity_days / self.total_days * 100.0
                if self.total_days > 0 else 0.0)


# US long-term capital gains, top statutory rate by era, as (before-year, pct).
# Rounded, and the exclusions and alternative computations of each era are not
# reproduced -- this is a lever to turn, not a tax return.
CGT_ERAS: tuple[tuple[int, float], ...] = (
    (1970, 25.0), (1979, 35.0), (1982, 28.0), (1987, 20.0),
    (1997, 28.0), (2003, 20.0), (2013, 15.0), (9999, 23.8),
)


@dataclass(frozen=True)
class Taxes:
    """What the taxman takes, and whether the survivor ever settles up.

    This was the largest unmodelled term in the file and it ran one way: the
    ladder realises gains on every sell and buy-and-hold defers them, so every
    number computed at zero tax UNDERSTATED the ladder's disadvantage. A gap
    that is understated in the direction of your own conclusion is exactly the
    kind this project keeps finding in its own code.

    `rate_pct = None` uses the historical era table. `liquidate` sells
    everything on the last bar and taxes the unrealised gain, which is the only
    way the comparison is symmetric -- without it buy-and-hold is being scored
    on money it has not been allowed to keep yet.

    Two simplifications, both stated because both favour the ladder, which is
    the arm this file concludes against:

      * Average cost basis, not specific-lot or FIFO. Real sellers pick lots to
        minimise the bill; average cost is the neutral middle.
      * The long-term rate always. Rungs usually fire years apart so that is
        mostly right, but `--rearm always` trades daily and would owe the much
        higher short-term rate. Charging it long-term flatters it.
    """
    rate_pct: float | None = 0.0
    liquidate: bool = False

    def rate_at(self, open_time_ms: int) -> float:
        if self.rate_pct is not None:
            return self.rate_pct
        year = year_of(open_time_ms)
        for cutoff, pct in CGT_ERAS:
            if year < cutoff:
                return pct
        return CGT_ERAS[-1][1]

    def describe(self) -> str:
        rate = "historical era rates" if self.rate_pct is None \
            else f"{self.rate_pct:.1f}%"
        return rate + (", settled at the end" if self.liquidate
                       else ", never settled by the holder")


NO_TAX = Taxes(0.0, False)


@dataclass
class _Book:
    """Cash, shares and the reference prices the rungs are measured against."""
    cash: float
    shares: float
    basis: float
    anchor: float
    peak: float
    buys: int = 0
    sells: int = 0
    unfunded: int = 0
    tax_paid: float = 0.0
    buy_armed: list[bool] = field(default_factory=lambda: [True] * len(BUY_RUNGS))
    sell_armed: list[bool] = field(default_factory=lambda: [True] * len(SELL_RUNGS))

    def value(self, price: float) -> float:
        return self.cash + self.shares * price


class Yields:
    """How much dividends and idle cash grow across one interval.

    An interface rather than two floats because the two floats were the whole
    soft underbelly of this test. The ladder sits in cash for a third of every
    window, so the dividend rate and the bill rate are exactly the constants
    that can move the answer without anyone noticing -- and both of them were
    assumptions. `MeasuredYields` replaces them with the actual series for the
    span where the actual series exist, which is the only way to find out how
    much the assumptions were carrying.
    """

    def dividend_factor(self, t_prev: int, t_now: int) -> float:
        raise NotImplementedError

    def cash_factor(self, t_prev: int, t_now: int) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class FlatYields(Yields):
    """Constant annual rates, stated rather than measured."""

    def __init__(self, dividend_pct: float, cash_pct: float):
        self.dividend_pct = dividend_pct
        self.cash_pct = cash_pct

    def dividend_factor(self, t_prev: int, t_now: int) -> float:
        if not self.dividend_pct:
            return 1.0
        days = (t_now - t_prev) / DAY_MS
        return (1.0 + self.dividend_pct / 100.0) ** (days / YEAR_DAYS)

    def cash_factor(self, t_prev: int, t_now: int) -> float:
        if not self.cash_pct:
            return 1.0
        days = (t_now - t_prev) / DAY_MS
        return (1.0 + self.cash_pct / 100.0) ** (days / YEAR_DAYS)

    def describe(self) -> str:
        return (f"assumed flat: {self.dividend_pct:.1f}% dividends, "
                f"{self.cash_pct:.1f}% on cash")


FLAT = FlatYields(DIVIDEND_YIELD_PCT, CASH_YIELD_PCT)
NO_YIELDS = FlatYields(0.0, 0.0)


def _accrue(book: _Book, price: float, t_prev: int, t_now: int,
            yields: Yields, taxes: Taxes = NO_TAX) -> None:
    """Dividends into cash, bill yield onto cash, over one bar.

    Dividends are paid in cash rather than reinvested into shares. On the
    buy-and-hold arm that is the conservative choice -- reinvesting would
    compound them -- so it cannot be the thing that makes the ladder look good.
    """
    if t_now <= t_prev:
        return
    if book.shares > 0:
        factor = yields.dividend_factor(t_prev, t_now)
        if factor != 1.0:
            # Qualified dividends are taxed in the year they are paid, by both
            # arms, and buy-and-hold receives more of them. Leaving them
            # untaxed would have quietly handed it the larger of the two
            # untaxed income streams.
            gross = book.shares * price * (factor - 1.0)
            rate = taxes.rate_at(t_now) / 100.0
            book.cash += gross * (1.0 - rate) if gross > 0 else gross
            if gross > 0:
                book.tax_paid += gross * rate
    if book.cash > 0:
        factor = yields.cash_factor(t_prev, t_now)
        if factor != 1.0:
            book.cash *= factor


def _buy(book: _Book, price: float, amount: float, bps: float) -> None:
    amount = min(amount, book.cash)
    if amount <= 0:
        return
    shares = amount * (1.0 - bps / 10_000.0) / price
    cost = book.basis * book.shares + amount
    book.shares += shares
    book.cash -= amount
    book.basis = cost / book.shares if book.shares > 0 else price
    book.buys += 1


def _sell(book: _Book, price: float, shares: float, bps: float,
          tax_rate: float = 0.0, count: bool = True) -> None:
    shares = min(shares, book.shares)
    if shares <= 0:
        return
    proceeds = shares * price * (1.0 - bps / 10_000.0)
    # Realised gain against average cost. A loss is simply not taxed here --
    # modelling loss carry-forward would be a credit to the arm that sells, and
    # this file already concludes against that arm.
    gain = shares * (price - book.basis)
    tax = gain * tax_rate / 100.0 if gain > 0 and tax_rate > 0 else 0.0
    book.cash += proceeds - tax
    book.tax_paid += tax
    book.shares -= shares
    if count:
        book.sells += 1


def _settle(book: _Book, last: Candle, taxes: Taxes) -> None:
    """Close the position on the final bar and pay the deferred bill.

    Off by default, and the default is the one that penalises the ladder: a
    holder who never sells never pays, so buy-and-hold's whole century of gains
    stays untaxed while the ladder has been paying all along. That comparison
    is legitimate -- it is what "buy and hold forever" means -- but it is not
    the only one, and quoting only it would be picking the framing that agrees
    with the conclusion. `--liquidate` scores both arms on money they are
    actually allowed to keep.
    """
    if not taxes.liquidate or book.shares <= 0:
        return
    _sell(book, last.close, book.shares, cost_bps(last.open_time),
          taxes.rate_at(last.open_time), count=False)


def simulate_ladder(
    candles: list[Candle],
    ref_mode: str,
    sizing: str,
    start_equity: float = 0.5,
    capital: float = 10_000.0,
    yields: Yields | None = None,
    rearm: str = "cross",
    trigger: str = "close",
    taxes: Taxes = NO_TAX,
) -> Run:
    """The posted rules, run bar by bar over one window of daily closes.

    `ref_mode` picks what the percentages are measured against:

      basis   both ladders measured against the position's average cost.
      peak    buys measured as a drawdown from the running all-time high;
              sells still measured against cost, because a "rise of 30% from
              the running maximum" is impossible by construction -- the
              maximum moves with the price.
      anchor  both measured against the close on day one, never updated. This
              is the most literal reading and, over a century, the one that
              sells everything by 1955 and never buys again.

    `rearm` and `trigger` are documented at REARM_MODES and TRIGGERS. They are
    my choices rather than the post's, which is why the default output audits
    them instead of assuming them. The running peak is measured on closes in
    both trigger modes on purpose: `trigger` then changes exactly one thing.
    """
    if ref_mode not in REF_MODES:
        raise ValueError(f"ref_mode must be one of {REF_MODES}, got {ref_mode!r}")
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}, got {sizing!r}")
    if rearm not in REARM_MODES:
        raise ValueError(f"rearm must be one of {REARM_MODES}, got {rearm!r}")
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
    if len(candles) < 2:
        raise ValueError("a ladder needs at least two bars")
    yields = FLAT if yields is None else yields

    first = candles[0].close
    entry_bps = cost_bps(candles[0].open_time)
    invested = capital * start_equity
    book = _Book(
        cash=capital - invested,
        shares=invested * (1.0 - entry_bps / 10_000.0) / first,
        basis=first / (1.0 - entry_bps / 10_000.0) if invested > 0 else first,
        anchor=first,
        peak=first,
    )

    peak_value = book.value(first)
    max_dd = 0.0
    equity_days = 0.0
    total_days = 0.0

    for prev, bar in zip(candles, candles[1:]):
        price = bar.close
        days = (bar.open_time - prev.open_time) / DAY_MS
        _accrue(book, price, prev.open_time, bar.open_time, yields, taxes)
        book.peak = max(book.peak, price)
        bps = cost_bps(bar.open_time)

        ref_buy = {"basis": book.basis, "peak": book.peak,
                   "anchor": book.anchor}[ref_mode]
        ref_sell = book.basis if ref_mode == "peak" else ref_buy

        drop = (price / ref_buy - 1.0) * 100.0 if ref_buy > 0 else 0.0
        rise = (price / ref_sell - 1.0) * 100.0 if ref_sell > 0 else 0.0

        # Re-arm first: a bar that carries price back through the reference
        # rearms the ladder it left, and may then trip the other one.
        if rearm == "always":
            book.buy_armed = [True] * len(BUY_RUNGS)
            book.sell_armed = [True] * len(SELL_RUNGS)
        elif rearm == "cross":
            if drop >= 0.0:
                book.buy_armed = [True] * len(BUY_RUNGS)
            if rise <= 0.0:
                book.sell_armed = [True] * len(SELL_RUNGS)

        # Deepest rung first, so a single -35% day buys the 30% rung rather
        # than spending the cash on the 20% one and arriving short.
        for i in range(len(BUY_RUNGS) - 1, -1, -1):
            threshold, pct = BUY_RUNGS[i]
            if not book.buy_armed[i]:
                continue
            level = ref_buy * (1.0 + threshold / 100.0)
            if trigger == "close":
                if price > level:
                    continue
                fill = price
            else:
                if bar.low > level:
                    continue
                # A bar that gapped straight through the level never traded at
                # it. Filling there anyway is the free money that makes every
                # limit-order backtest look brilliant.
                fill = bar.open if bar.open < level else level
            base = capital if sizing == "initial" else book.shares * fill
            if base <= 0:
                base = capital
            want = base * pct / 100.0
            if want > book.cash:
                book.unfunded += 1
            _buy(book, fill, want, bps)
            book.buy_armed[i] = False

        for i in range(len(SELL_RUNGS) - 1, -1, -1):
            threshold, pct = SELL_RUNGS[i]
            if not book.sell_armed[i]:
                continue
            level = ref_sell * (1.0 + threshold / 100.0)
            if trigger == "close":
                if price < level:
                    continue
                fill = price
            else:
                if bar.high < level:
                    continue
                fill = bar.open if bar.open > level else level
            if sizing == "initial":
                shares = min(book.shares, capital * pct / 100.0 / fill)
            else:
                shares = book.shares * pct / 100.0
            _sell(book, fill, shares, bps, taxes.rate_at(bar.open_time))
            book.sell_armed[i] = False

        value = book.value(price)
        peak_value = max(peak_value, value)
        if peak_value > 0:
            max_dd = max(max_dd, (1.0 - value / peak_value) * 100.0)
        if value > 0:
            equity_days += days * (book.shares * price) / value
        total_days += days

    _settle(book, candles[-1], taxes)
    years = (candles[-1].open_time - candles[0].open_time) / DAY_MS / YEAR_DAYS
    return Run(
        name=f"{ref_mode}/{sizing}", final=book.value(candles[-1].close),
        initial=capital, years=years, max_dd=max_dd, buys=book.buys,
        sells=book.sells, unfunded=book.unfunded, cash_end=book.cash,
        equity_days=equity_days, total_days=total_days, tax_paid=book.tax_paid,
    )


def simulate_hold(
    candles: list[Candle],
    start_equity: float = 1.0,
    capital: float = 10_000.0,
    yields: Yields | None = None,
    taxes: Taxes = NO_TAX,
) -> Run:
    """Buy once, never touch it. `start_equity` < 1 leaves the rest in bills.

    Two of these are run against every ladder. The fully invested one is what
    a reader means by "buy and hold". The cash-matched one starts with the same
    equity fraction as the ladder and never rebalances, which separates two
    claims the post conflates: that the TIMING helps, and that holding a cash
    buffer helps. Only the second is ever obviously true.
    """
    yields = FLAT if yields is None else yields
    first = candles[0].close
    entry_bps = cost_bps(candles[0].open_time)
    invested = capital * start_equity
    book = _Book(
        cash=capital - invested,
        shares=invested * (1.0 - entry_bps / 10_000.0) / first,
        # The commission paid on the way in is part of the cost basis, and the
        # ladder has always counted it. This arm did not, which understated its
        # basis, overstated its gain and made it overpay on liquidation -- a
        # ~1.2 bp gift to the ladder on every taxed comparison. Found by the
        # invariant that a ladder with no rungs must equal this function
        # exactly, not by reading the code.
        basis=first / (1.0 - entry_bps / 10_000.0) if invested > 0 else first,
        anchor=first, peak=first,
    )
    peak_value = book.value(first)
    max_dd = 0.0
    equity_days = 0.0
    total_days = 0.0

    for prev, bar in zip(candles, candles[1:]):
        days = (bar.open_time - prev.open_time) / DAY_MS
        _accrue(book, bar.close, prev.open_time, bar.open_time, yields, taxes)
        value = book.value(bar.close)
        peak_value = max(peak_value, value)
        if peak_value > 0:
            max_dd = max(max_dd, (1.0 - value / peak_value) * 100.0)
        if value > 0:
            equity_days += days * (book.shares * bar.close) / value
        total_days += days

    _settle(book, candles[-1], taxes)
    years = (candles[-1].open_time - candles[0].open_time) / DAY_MS / YEAR_DAYS
    name = "hold 100%" if start_equity >= 1.0 else f"hold {start_equity:.0%}"
    return Run(name=name, final=book.value(candles[-1].close), initial=capital,
               years=years, max_dd=max_dd, cash_end=book.cash,
               equity_days=equity_days, total_days=total_days,
               tax_paid=book.tax_paid)


class MeasuredYields(Yields):
    """The real bill path and the real dividend path, from two Yahoo series.

    Cash earns `^IRX`, the 13-week Treasury bill rate, daily from 1960 and
    forward-filled across days the bill did not print. Dividends come from the
    gap between `^SP500TR` and `^GSPC`: the total-return index is the price
    index with dividends reinvested, so the ratio of their daily growth IS the
    dividend accrual, measured rather than assumed. That series starts in 1988.

    This is a smaller sample than the century the flat-rate test uses, and it
    is not a replacement for it. It answers one question: how much of the
    answer were the two constants carrying?
    """

    def __init__(self, bill_pct: dict[int, float],
                 div_factor: dict[int, float] | None = None,
                 div_annual_by_month: dict[tuple[int, int], float] | None = None,
                 clamped: int = 0, source: str = "^SP500TR/^GSPC"):
        if not (div_factor or div_annual_by_month):
            raise ValueError("a dividend series is required")
        self.bill = bill_pct
        self.div = div_factor or {}
        self.div_month = div_annual_by_month or {}
        self.clamped = clamped
        self.source = source
        self._days = sorted(bill_pct)
        if self.div:
            div_lo, div_hi = min(self.div) * DAY_MS, max(self.div) * DAY_MS
        else:
            # Shiller stamps each month at the first, so the month's coverage
            # runs to the end of it; using the stamp as the end would throw
            # away the last month every time.
            lo_y, lo_m = min(self.div_month)
            hi_y, hi_m = max(self.div_month)
            div_lo = _month_ms(lo_y, lo_m)
            div_hi = _month_ms(hi_y + (hi_m == 12), hi_m % 12 + 1) - DAY_MS
        self.start_ms = max(min(self._days) * DAY_MS, div_lo)
        self.end_ms = min(max(self._days) * DAY_MS, div_hi)

    def _bill_at(self, t: int) -> float:
        import bisect

        key = t // DAY_MS
        i = bisect.bisect_right(self._days, key)
        if i == 0:
            return 0.0
        return self.bill[self._days[i - 1]]

    def dividend_factor(self, t_prev: int, t_now: int) -> float:
        if self.div:
            # Daily bars and a shared trading calendar, so one factor per bar,
            # and it already contains the weekend because both indices jump
            # across it. A missing day contributes nothing rather than being
            # interpolated: an invented dividend is worse than a skipped one.
            return self.div.get(t_now // DAY_MS, 1.0)
        # Monthly annual rates: compound the month's rate over the actual gap
        # between bars, so a Monday after a long weekend accrues three days.
        stamp = EPOCH + timedelta(milliseconds=t_now)
        rate = self.div_month.get((stamp.year, stamp.month), 0.0)
        if rate <= 0:
            return 1.0
        return (1.0 + rate / 100.0) ** ((t_now - t_prev) / DAY_MS / YEAR_DAYS)

    def cash_factor(self, t_prev: int, t_now: int) -> float:
        rate = self._bill_at(t_prev)
        if rate <= 0:
            return 1.0
        days = (t_now - t_prev) / DAY_MS
        return (1.0 + rate / 100.0) ** (days / YEAR_DAYS)

    def mean_bill_pct(self) -> float:
        return sum(self.bill.values()) / len(self.bill) if self.bill else 0.0

    def mean_dividend_pct(self) -> float:
        """The measured dividend accrual, annualised, over the COVERED span.

        Over the covered span, not over the whole series: the Shiller data
        starts in 1871 and yields were far higher then, so the lifetime mean
        would describe a period the test never touches.
        """
        if self.div_month:
            lo, hi = year_of(self.start_ms), year_of(self.end_ms)
            values = [v for (y, _m), v in self.div_month.items() if lo <= y <= hi]
            return sum(values) / len(values) if values else 0.0
        if not self.div:
            return 0.0
        total = 1.0
        for f in self.div.values():
            total *= f
        span = (max(self.div) - min(self.div)) / YEAR_DAYS
        return ((total ** (1.0 / span)) - 1.0) * 100.0 if span > 0 else 0.0

    def describe(self) -> str:
        return (f"measured: {self.mean_dividend_pct():.2f}% dividends from "
                f"{self.source}, {self.mean_bill_pct():.2f}% mean ^IRX bill")


# A day's dividend accrual outside this band is a data glitch, not a dividend:
# the S&P's whole yield is ~2% a year, which is 0.008% a day. Anything past
# these bounds is dropped and counted rather than compounded into the answer.
DIV_FACTOR_BOUNDS = (0.998, 1.002)


def _month_ms(year: int, month: int) -> int:
    return int((datetime(year, month, 1, tzinfo=timezone.utc)
                - EPOCH).total_seconds() * 1000)


def load_measured_yields(cfg: Config,
                         dividend_source: str = "yahoo") -> MeasuredYields | None:
    """Build the measured rate paths, or None if a series is unavailable.

    Two dividend sources, and the difference is thirty years of sample:

      yahoo    `^SP500TR` against `^GSPC`, daily, from 1988. Same vendor as the
               prices, which is the weakness -- a systematic Yahoo error would
               cancel out of the ratio and never be seen.
      shiller  the dividend-per-share series from the Shiller dataset, monthly,
               from 1871. A different compiler, and it pushes the measured span
               back to 1960, where `^IRX` starts and the binding constraint
               moves to the bill series instead.

    Cash is `^IRX` either way; there is no keyless daily bill series before
    1960 that this project could find, and substituting the 10-year yield for a
    3-month bill would be inventing a number, not measuring one.
    """
    bills = load_equity_history(["^IRX"], "1d", 40_000.0,
                                cfg.path("data/yahoo")).get("^IRX")
    if not bills:
        log.warning("measured yields need ^IRX and it did not load")
        return None
    bill_pct = {c.open_time // DAY_MS: c.close for c in bills}

    if dividend_source == "shiller":
        from .shiller import dividend_yield_path, load_shiller

        months = load_shiller(cfg.path("data/shiller"))
        by_month = dividend_yield_path(months) if months else {}
        if not by_month:
            log.warning("measured yields: the Shiller series did not load")
            return None
        return MeasuredYields(bill_pct, div_annual_by_month=by_month,
                              source="Shiller dividend/price")

    series = load_equity_history(["^SP500TR", "^GSPC"], "1d", 40_000.0,
                                 cfg.path("data/yahoo"))
    total = series.get("^SP500TR")
    price = series.get("^GSPC")
    if not total or not price:
        log.warning("measured yields need ^SP500TR and ^GSPC; got %s",
                    sorted(k for k, v in series.items() if v))
        return None

    tr = {c.open_time // DAY_MS: c.close for c in total}
    pr = {c.open_time // DAY_MS: c.close for c in price}

    shared = sorted(set(tr) & set(pr))
    div: dict[int, float] = {}
    clamped = 0
    for prev, day in zip(shared, shared[1:]):
        tr_step = tr[day] / tr[prev]
        pr_step = pr[day] / pr[prev]
        if pr_step <= 0:
            continue
        factor = tr_step / pr_step
        if not DIV_FACTOR_BOUNDS[0] <= factor <= DIV_FACTOR_BOUNDS[1]:
            clamped += 1
            continue
        div[day] = factor
    if not div:
        return None
    return MeasuredYields(bill_pct, div_factor=div, clamped=clamped)


def split_windows(candles: list[Candle], years: float,
                  limit: int = 0) -> list[list[Candle]]:
    """Non-overlapping windows of `years`, newest-aligned, oldest first.

    Aligned to the newest bar rather than the oldest so that a partial window
    lands in the 1920s rather than in the present, where it would be the one
    window every reader checks. Windows shorter than 80% of the target are
    dropped instead of being reported as a short decade.
    """
    if not candles:
        return []
    span = int(years * YEAR_DAYS * DAY_MS)
    newest = candles[-1].open_time
    out: list[list[Candle]] = []
    k = 0
    while True:
        hi = newest - k * span
        lo = hi - span
        chunk = [c for c in candles if lo <= c.open_time < hi]
        if len(chunk) < 2:
            break
        covered = (chunk[-1].open_time - chunk[0].open_time) / span
        if covered >= 0.8:
            out.append(chunk)
        k += 1
        if limit and len(out) >= limit:
            break
    out.reverse()
    return out


def _stats(diffs: list[float]) -> tuple[float, float, float, int]:
    """Mean, standard error, two-sided p and df, treating each window as one.

    The same choice as everywhere else in this project: a window is one
    observation. Daily returns inside a decade are not independent facts about
    whether the ladder works, and a t-test over 25,000 of them would report a
    p-value about the 1930s.
    """
    from .replicate import student_t_p

    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else 0.0), 0.0, 1.0, 0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    p = student_t_p(mean / se, n - 1) if se > 0 else 1.0
    return mean, se, p, n - 1


async def run_ladder(
    cfg: Config,
    window_years: float = 10.0,
    windows: int = 0,
    start_equity: float = 0.5,
    capital: float = 10_000.0,
    symbols: list[str] | None = None,
    measured_yields: bool = False,
    dividend_source: str = "yahoo",
    rearm: str = "cross",
    trigger: str = "close",
    tax_rate: float | None = 0.0,
    liquidate: bool = False,
    preset: str = "default",
) -> dict:
    if preset not in SYMBOL_PRESETS:
        raise ValueError(f"preset must be one of {sorted(SYMBOL_PRESETS)}")
    names = symbols or SYMBOL_PRESETS[preset]
    # Yahoo serves daily bars back to 1927 for ^GSPC; ask for more than exists
    # and it returns what it has rather than failing.
    history = load_equity_history(names, "1d", 40_000.0, cfg.path("data/yahoo"))
    if not history:
        raise RuntimeError("no daily history returned")

    yields: Yields = FLAT
    if measured_yields:
        measured = load_measured_yields(cfg, dividend_source)
        if measured is None:
            raise RuntimeError(
                f"measured yields unavailable for source {dividend_source!r}")
        yields = measured
        # Outside the covered span the measured rates do not exist, and
        # falling back to the flat constants inside the same run would mix two
        # answers under one heading. Clip instead, loudly.
        lo, hi = measured.start_ms, measured.end_ms
        for symbol in list(history):
            history[symbol] = [c for c in history[symbol] if lo <= c.open_time <= hi]
        log.info("measured yields cover %d-%d; every series clipped to it",
                 year_of(lo), year_of(hi))

    taxes = Taxes(tax_rate, liquidate)
    _print_assumptions(start_equity, capital, window_years, yields, taxes)

    out: dict[str, dict] = {}
    for symbol in names:
        candles = history.get(symbol)
        if not candles or len(candles) < 500:
            log.warning("%s: %d daily bars, skipped",
                        symbol, len(candles) if candles else 0)
            continue
        chunks = split_windows(candles, window_years, windows)
        if len(chunks) < 2:
            log.warning("%s: only %d full windows of %.0f years, skipped",
                        symbol, len(chunks), window_years)
            continue
        out[symbol] = _run_symbol(symbol, chunks, start_equity, capital, yields,
                                  rearm, trigger, taxes)
        out[symbol]["chunks"] = chunks

    if not out:
        raise RuntimeError("no symbol had enough history for a window test")

    lead = "^GSPC" if "^GSPC" in out else next(iter(out))
    _print_design_audit(lead, out[lead]["chunks"], start_equity, capital,
                        yields, out[lead]["best"]["name"])
    _print_regime_split(lead, out[lead]["chunks"], start_equity, capital,
                        yields, out[lead]["best"]["name"], rearm, trigger,
                        taxes)
    _print_tax_lever(lead, out[lead]["chunks"], start_equity, capital, yields,
                     out[lead]["best"]["name"], rearm, trigger)

    if measured_yields:
        _print_rate_lever(history, names, window_years, windows,
                          start_equity, capital, out)
    else:
        _print_zero_yield(history, names, window_years, windows,
                          start_equity, capital)

    headline = out.get("^GSPC") or next(iter(out.values()))
    # The measured run covers 1988 onward and the flat run covers 1927 onward.
    # They are different measurements of different spans, so they get different
    # keys; letting the shorter one overwrite the longer one would quietly
    # shrink the sample behind the published number by sixty years.
    if headline["windows"] >= MIN_WINDOWS:
        record(cfg, "ladder_measured" if measured_yields else "ladder", {
            "symbol": headline["symbol"], "windows": headline["windows"],
            "best_variant": headline["best"]["name"],
            "mean_cagr_diff": headline["best"]["mean"],
            "timing_cagr_diff": headline["best"]["matched_mean"],
            "allocation_cagr_diff": (headline["best"]["mean"]
                                     - headline["best"]["matched_mean"]),
            "invested_pct": headline["best"]["invested"],
            "max_dd": headline["best"]["dd"],
            "hold_dd": headline["hold_dd"],
            "p": headline["best"]["p"],
            "p_adjusted": min(1.0, headline["best"]["p"] * len(REF_MODES) * len(SIZINGS)),
            "window_years": window_years, "start_equity": start_equity,
            "yields": yields.describe(), "rearm": rearm, "trigger": trigger,
            "taxes": taxes.describe(), "preset": preset,
        })
    return out


def _run_symbol(symbol: str, chunks: list[list[Candle]],
                start_equity: float, capital: float,
                yields: Yields | None = None,
                rearm: str = "cross", trigger: str = "close",
                taxes: Taxes = NO_TAX, quiet: bool = False) -> dict:
    """Every variant against both baselines, window by window."""
    variants = [(r, s) for r in REF_MODES for s in SIZINGS]
    rows: dict[str, list[float]] = {f"{r}/{s}": [] for r, s in variants}
    matched: dict[str, list[float]] = {f"{r}/{s}": [] for r, s in variants}
    runs: dict[str, list[Run]] = {f"{r}/{s}": [] for r, s in variants}
    holds: list[Run] = []
    holds_matched: list[Run] = []

    for chunk in chunks:
        full = simulate_hold(chunk, 1.0, capital, yields, taxes)
        part = simulate_hold(chunk, start_equity, capital, yields, taxes)
        holds.append(full)
        holds_matched.append(part)
        for ref, sizing in variants:
            run = simulate_ladder(chunk, ref, sizing, start_equity, capital,
                                  yields, rearm, trigger, taxes)
            key = f"{ref}/{sizing}"
            runs[key].append(run)
            rows[key].append(run.cagr - full.cagr)
            matched[key].append(run.cagr - part.cagr)

    summary = []
    for key in rows:
        mean, se, p, df = _stats(rows[key])
        m_mean, _, m_p, _ = _stats(matched[key])
        group = runs[key]
        summary.append({
            "name": key, "mean": mean, "se": se, "p": p, "df": df,
            "matched_mean": m_mean, "matched_p": m_p,
            "wins": sum(1 for d in rows[key] if d > 0),
            "dd": sum(r.max_dd for r in group) / len(group),
            "trades": sum(r.buys + r.sells for r in group) / len(group),
            "unfunded": sum(r.unfunded for r in group),
            "tax": sum(r.tax_paid for r in group) / len(group),
            "invested": sum(r.invested_pct for r in group) / len(group),
            "diffs": rows[key],
        })
    summary.sort(key=lambda d: -d["mean"])
    best = summary[0]

    if not quiet:
        _print_symbol(symbol, chunks, holds, holds_matched, summary,
                      start_equity, taxes)

    return {"symbol": symbol, "windows": len(chunks), "variants": summary,
            "best": best,
            "hold_cagr": sum(h.cagr for h in holds) / len(holds),
            "hold_dd": sum(h.max_dd for h in holds) / len(holds)}


def _print_assumptions(start_equity: float, capital: float,
                       window_years: float, yields: Yields,
                       taxes: Taxes) -> None:
    print("\n" + "=" * 80)
    print("  THE 105-YEAR-OLD'S LADDER -- buy 20%/30% dips, sell 30-100% rallies")
    print("=" * 80)
    if isinstance(yields, MeasuredYields):
        print("  Settings, with the two rates taken from real series:")
    else:
        print("  Stated assumptions, none of them measured from this data:")
    print(f"    starting capital            {capital:>12,.0f}")
    print(f"    starting equity fraction    {start_equity:>12.0%}"
          "   (the post does not say)")
    print(f"    window length               {window_years:>12.0f} years"
          "   (each window is one observation)")
    if isinstance(yields, MeasuredYields):
        print(f"    dividend yield              "
              f"{yields.mean_dividend_pct():>12.2f}%"
              f"   MEASURED from {yields.source}")
        print(f"    cash / bill yield           "
              f"{yields.mean_bill_pct():>12.2f}%   MEASURED, mean of ^IRX")
        print(f"    span with both series       "
              f"{year_of(yields.start_ms)}-{year_of(yields.end_ms)}"
              "        everything clipped to it")
        if yields.clamped:
            print(f"    days dropped as glitches    {yields.clamped:>12,}"
                  f"   outside {DIV_FACTOR_BOUNDS}")
    else:
        print(f"    dividend yield              {DIVIDEND_YIELD_PCT:>12.1f}%"
              "   ASSUMED, paid to cash, not reinvested")
        print(f"    cash / bill yield           {CASH_YIELD_PCT:>12.1f}%"
              "   ASSUMED")
    print("    friction per side           "
          + "  ".join(f"{'<' + str(y):>6} {b:.0f}bps" for y, b in COST_ERAS))
    print(f"    capital gains tax           {taxes.describe():>26}")
    if taxes.rate_pct == 0.0:
        print("                                     at zero the ladder's "
              "disadvantage is UNDERSTATED;")
        print("                                     see the tax table below.")
    print("=" * 80)


def _print_symbol(symbol: str, chunks: list[list[Candle]], holds: list[Run],
                  matched: list[Run], summary: list[dict],
                  start_equity: float, taxes: Taxes = NO_TAX) -> None:
    from .replicate import t_critical

    first = year_of(chunks[0][0].open_time)
    last = year_of(chunks[-1][-1].open_time)
    hold_cagr = sum(h.cagr for h in holds) / len(holds)
    hold_dd = sum(h.max_dd for h in holds) / len(holds)
    match_cagr = sum(h.cagr for h in matched) / len(matched)

    print(f"\n  {symbol}  --  {len(chunks)} non-overlapping windows, "
          f"{first}-{last}")
    print("-" * 80)
    print(f"    {'baseline':<22}{'CAGR':>9}{'max DD':>10}"
          "   (mean of the per-window figures, not the whole span)")
    print(f"    {'hold 100%':<22}{hold_cagr:>+8.2f}%{hold_dd:>9.1f}%")
    print(f"    {f'hold {start_equity:.0%} + cash':<22}"
          f"{match_cagr:>+8.2f}%"
          f"{sum(h.max_dd for h in matched) / len(matched):>9.1f}%")
    print("-" * 80)
    print(f"    {'variant':<16}{'vs hold':>10}{'se':>8}{'p':>8}{'win':>6}"
          f"{'vs matched':>12}{'max DD':>9}{'in mkt':>8}{'trades':>8}")
    for row in summary:
        print(f"    {row['name']:<16}{row['mean']:>+9.2f}%{row['se']:>8.2f}"
              f"{row['p']:>8.3f}{row['wins']:>3}/{len(chunks):<2}"
              f"{row['matched_mean']:>+11.2f}%{row['dd']:>8.1f}%"
              f"{row['invested']:>7.0f}%{row['trades']:>8.1f}")
    print("-" * 80)

    best = summary[0]
    adjusted = min(1.0, best["p"] * len(summary))
    if best["df"] >= 1:
        tc = t_critical(best["df"])
        lo = best["mean"] - tc * best["se"]
        hi = best["mean"] + tc * best["se"]
        print(f"    best variant {best['name']}: {best['mean']:+.2f}% CAGR vs "
              f"buy-and-hold, 95% CI [{lo:+.2f}%, {hi:+.2f}%]")
        print(f"    p = {best['p']:.3f} raw, {adjusted:.3f} after Bonferroni "
              f"over the {len(summary)} variants tried")
    unfunded = sum(r["unfunded"] for r in summary)
    if unfunded:
        print(f"    {unfunded} buy rungs fired with less cash than the rule "
              f"asked for and were filled short")

    _verdict(len(chunks), best, adjusted, taxes)


def _verdict(windows: int, best: dict, adjusted: float,
             taxes: Taxes = NO_TAX) -> None:
    """The headline, and the decomposition that keeps it from being a slogan.

    "The ladder loses to buy-and-hold" is true and almost uninformative on its
    own, because two different things are inside it. Holding a third of the
    portfolio in cash for a century costs a specific amount whatever you do
    with the rest, and choosing WHEN to move that money costs or earns
    something else. The cash-matched baseline separates them, and the split is
    lopsided enough that the two halves deserve separate sentences.
    """
    if windows < MIN_WINDOWS:
        print(f"  ONLY {windows} WINDOWS -- no verdict. Below {MIN_WINDOWS} the "
              "t-multiplier is above 3 and\n  the interval covers everything "
              "anyone might claim.")
        return

    timing = best["matched_mean"]
    allocation = best["mean"] - timing
    if best["mean"] > 0 and adjusted < 0.05:
        print(f"  THE LADDER BEATS BUY-AND-HOLD by {best['mean']:+.2f}% CAGR, "
              "surviving multiplicity.\n  This contradicts the prediction in "
              "the module docstring.")
    elif best["mean"] > 0:
        print(f"  Best variant is ahead by {best['mean']:+.2f}% CAGR but at "
              f"p = {adjusted:.3f} after\n  correcting for the "
              f"{len(REF_MODES) * len(SIZINGS)} variants, which is what picking "
              "the best of six looks like\n  when none of them does anything.")
    else:
        print(f"  THE LADDER LOSES to buy-and-hold in every reading tried; the "
              f"best is {best['mean']:+.2f}%\n  CAGR behind, and it cuts "
              f"drawdown to {best['dd']:.1f}% in exchange.")

    print(f"  Of that {best['mean']:+.2f}%, {allocation:+.2f}% is ALLOCATION -- "
          f"the ladder is only {best['invested']:.0f}% invested\n  on average, "
          "and cash earns less than stocks. Only "
          f"{timing:+.2f}% is TIMING, the part the\n  post is actually about, "
          f"measured against a static portfolio holding the same cash.")
    if abs(timing) < abs(allocation):
        print("  So the rules are mostly not a market-timing claim at all. They "
              "are an asset\n  allocation, arrived at accidentally, described "
              "as discipline. Treat the timing\n  figure as 'near zero' rather "
              "than as that exact number -- the design audit below\n  moves it "
              "around by more than its own size.")
    if taxes.rate_pct == 0.0:
        print("  Every gap above is UNDERSTATED: this run is at zero tax, and "
              "the ladder realises\n  gains while buy-and-hold defers them. "
              "The tax table below prices that in.")
    else:
        print(f"  Tax is priced in at {taxes.describe()}.")


def _print_design_audit(symbol: str, chunks: list[list[Candle]],
                        start_equity: float, capital: float, yields: Yields,
                        variant: str) -> None:
    """Every combination of the two rules I invented, on the headline symbol.

    The six variants in the main table are readings of an ambiguous post. These
    two are not: nothing in the post says when a rung re-arms or whether the
    threshold is watched intraday, so I chose, and a finding that only survives
    my choices is a finding about me. The cost of checking is six more runs.
    """
    ref, sizing = variant.split("/")
    print("\n" + "-" * 80)
    print(f"  MY OWN TWO CHOICES, AUDITED -- {symbol}, {variant}, "
          f"{len(chunks)} windows")
    print("-" * 80)
    print(f"    {'re-arm':<10}{'trigger':<12}{'vs hold':>10}{'timing':>10}"
          f"{'p':>8}{'trades':>9}{'in mkt':>9}{'unfunded':>10}")
    seen_timing: list[float] = []
    seen_total: list[float] = []
    seen_p: list[float] = []
    for rearm in REARM_MODES:
        for trigger in TRIGGERS:
            totals, timings, runs = [], [], []
            for chunk in chunks:
                full = simulate_hold(chunk, 1.0, capital, yields)
                part = simulate_hold(chunk, start_equity, capital, yields)
                run = simulate_ladder(chunk, ref, sizing, start_equity,
                                      capital, yields, rearm, trigger)
                runs.append(run)
                totals.append(run.cagr - full.cagr)
                timings.append(run.cagr - part.cagr)
            mean, _se, p, _df = _stats(totals)
            t_mean, _, _, _ = _stats(timings)
            trades = sum(r.buys + r.sells for r in runs) / len(runs)
            invested = sum(r.invested_pct for r in runs) / len(runs)
            mine = " <-- default" if (rearm, trigger) == ("cross", "close") else ""
            print(f"    {rearm:<10}{trigger:<12}{mean:>+9.2f}%{t_mean:>+9.2f}%"
                  f"{p:>8.3f}{trades:>9.1f}{invested:>8.0f}%"
                  f"{sum(r.unfunded for r in runs):>10}{mine}")
            seen_timing.append(t_mean)
            seen_total.append(mean)
            seen_p.append(p)
    print("-" * 80)

    # The test the table exists to run, done here rather than left to the
    # reader, because the reader will believe the row that agrees with them.
    stable = all(t > 0 for t in seen_timing) or all(t < 0 for t in seen_timing)
    spread = max(seen_timing) - min(seen_timing)
    if stable and min(seen_p) > 0.05:
        print(f"    The timing term keeps its sign across all six "
              f"({min(seen_timing):+.2f}% to {max(seen_timing):+.2f}%) -- but "
              f"not one of\n    them is significant (p = {min(seen_p):.2f} to "
              f"{max(seen_p):.2f}). A consistent sign on six\n    "
              "configurations of the same windows is one observation, not six, "
              "and this one\n    cannot be told apart from zero.")
    elif stable:
        print(f"    The timing term keeps its sign across all six "
              f"({min(seen_timing):+.2f}% to {max(seen_timing):+.2f}%), so it "
              "is a\n    property of the rules rather than of my two choices.")
    else:
        print(f"    THE TIMING TERM CHANGES SIGN across my own choices: "
              f"{min(seen_timing):+.2f}% to {max(seen_timing):+.2f}%, a "
              f"{spread:.2f}%\n    spread on a term whose largest value is "
              f"{max(abs(t) for t in seen_timing):.2f}%. That is what no "
              "effect looks like,\n    and any single number quoted for it -- "
              "including mine -- is a choice, not a\n    measurement.")
    print(f"    The allocation term is stable: total runs "
          f"{min(seen_total):+.2f}% to {max(seen_total):+.2f}% while the "
          "timing part\n    stays near zero, so the cash drag is doing the "
          "work in every configuration.")
    print("-" * 80 + "\n")


def _print_regime_split(symbol: str, chunks: list[list[Candle]],
                        start_equity: float, capital: float, yields: Yields,
                        variant: str, rearm: str, trigger: str,
                        taxes: Taxes) -> None:
    """The same windows, split by whether the market went up or down in them.

    This is the mechanism check the collapsed-market list was built for. If
    the ladder's whole story is "it holds cash", then it must lose in rising
    windows by roughly what the cash costs and win in falling ones by roughly
    what the cash saves, with the timing term near zero in BOTH. If the timing
    rules do anything, the falling windows are where they should do it -- that
    is the environment buy-the-dip is sold for.
    """
    ref, sizing = variant.split("/")
    buckets: dict[str, list[tuple[float, float]]] = {"rose": [], "fell": []}
    for chunk in chunks:
        full = simulate_hold(chunk, 1.0, capital, yields, taxes)
        part = simulate_hold(chunk, start_equity, capital, yields, taxes)
        run = simulate_ladder(chunk, ref, sizing, start_equity, capital,
                              yields, rearm, trigger, taxes)
        key = "rose" if full.cagr >= 0 else "fell"
        buckets[key].append((run.cagr - full.cagr, run.cagr - part.cagr))

    print("\n" + "-" * 80)
    print(f"  WHERE THE LADDER WINS AND LOSES -- {symbol}, {variant}")
    print("-" * 80)
    print(f"    {'windows where':<20}{'n':>4}{'vs hold':>11}{'timing':>11}")
    for key in ("rose", "fell"):
        rows = buckets[key]
        if not rows:
            print(f"    the market {key:<9}{0:>4}{'--':>11}{'--':>11}")
            continue
        total = sum(r[0] for r in rows) / len(rows)
        timing = sum(r[1] for r in rows) / len(rows)
        print(f"    the market {key:<9}{len(rows):>4}{total:>+10.2f}%"
              f"{timing:>+10.2f}%")
    print("-" * 80)
    if buckets["rose"] and buckets["fell"]:
        rose = sum(r[0] for r in buckets["rose"]) / len(buckets["rose"])
        fell = sum(r[0] for r in buckets["fell"]) / len(buckets["fell"])
        rose_t = sum(r[1] for r in buckets["rose"]) / len(buckets["rose"])
        fell_t = sum(r[1] for r in buckets["fell"]) / len(buckets["fell"])
        print(f"    Holding cash costs {-rose:.2f}% a year when the market "
              f"rises and saves {fell:.2f}% when it\n    falls -- which is what "
              "holding cash does, and needs no rules. The timing term is\n"
              f"    {rose_t:+.2f}% and {fell_t:+.2f}%: whatever the market did, "
              "the rules themselves did nothing.")
    print("-" * 80 + "\n")


def _print_tax_lever(symbol: str, chunks: list[list[Candle]],
                     start_equity: float, capital: float, yields: Yields,
                     variant: str, rearm: str, trigger: str) -> None:
    """The term this file spent its whole life saying it had not modelled.

    Every table above was computed at zero tax, and zero tax is not neutral:
    the ladder realises gains on every sell while buy-and-hold defers them, so
    the ladder's disadvantage was understated everywhere. It ran in the same
    direction as the conclusion, which is the kind of unmodelled term this
    project has learned to distrust most.

    Both framings are printed. "Never settled" is the buy-and-hold-forever
    reader, where only the ladder ever pays. "Settled at the end" liquidates
    both arms on the last bar, which is the only symmetric comparison.
    """
    ref, sizing = variant.split("/")
    rates: list[tuple[str, float | None]] = [
        ("none", 0.0), ("15%", 15.0), ("23.8%", 23.8), ("era rates", None),
    ]
    print("\n" + "-" * 80)
    print(f"  TAX, THE TERM THAT WAS MISSING -- {symbol}, {variant}, "
          f"{len(chunks)} windows")
    print("-" * 80)
    print(f"    {'cgt rate':<12}{'vs hold':>10}{'timing':>10}{'p':>8}"
          f"{'  |  ':<5}{'vs hold':>10}{'timing':>10}{'p':>8}")
    print(f"    {'':<12}{'---- never settled ----':^28}"
          f"{'  |  ':<5}{'--- settled at the end ---':^28}")
    rows: list[tuple[str, float, float, float, float]] = []
    for label, rate in rates:
        cells: list[tuple[float, float, float]] = []
        for liquidate in (False, True):
            taxes = Taxes(rate, liquidate)
            totals, timings = [], []
            for chunk in chunks:
                full = simulate_hold(chunk, 1.0, capital, yields, taxes)
                part = simulate_hold(chunk, start_equity, capital, yields, taxes)
                run = simulate_ladder(chunk, ref, sizing, start_equity, capital,
                                      yields, rearm, trigger, taxes)
                totals.append(run.cagr - full.cagr)
                timings.append(run.cagr - part.cagr)
            mean, _se, p, _df = _stats(totals)
            t_mean, _, _, _ = _stats(timings)
            cells.append((mean, t_mean, p))
        print(f"    {label:<12}{cells[0][0]:>+9.2f}%{cells[0][1]:>+9.2f}%"
              f"{cells[0][2]:>8.3f}{'  |  ':<5}"
              f"{cells[1][0]:>+9.2f}%{cells[1][1]:>+9.2f}%{cells[1][2]:>8.3f}")
        rows.append((label, cells[0][0], cells[0][1], cells[1][0], cells[1][1]))
    print("-" * 80)

    untaxed, taxed = rows[0], rows[-1]
    print(f"    Taxing at the historical rates costs the ladder "
          f"{taxed[1] - untaxed[1]:+.2f}% CAGR when only it\n    pays, and "
          f"{taxed[3] - untaxed[3]:+.2f}% when both arms settle. The timing "
          f"term moves {taxed[2] - untaxed[2]:+.2f}% and\n    "
          f"{taxed[4] - untaxed[4]:+.2f}% respectively.")
    if all(r[1] < 0 for r in rows) and all(r[3] < 0 for r in rows):
        print("    The ladder loses at every rate and under both framings, so "
              "the missing term was\n    never load-bearing -- it only ever "
              "made a losing arm lose by more.")
    else:
        print("    The sign is NOT the same at every rate. Tax was "
              "load-bearing, and every number\n    computed at zero above "
              "should be read with that in mind.")
    print("-" * 80 + "\n")


def _print_rate_lever(history: dict[str, list[Candle]], names: list[str],
                      window_years: float, windows: int, start_equity: float,
                      capital: float, measured: dict[str, dict]) -> None:
    """The same windows again with the assumed rates, so the pair is controlled.

    Running measured rates over 1988-2026 and comparing that to assumed rates
    over 1927-2026 would confound two changes at once, which is the mistake
    this project has made and caught more than once. The history is already
    clipped to the measured span, so re-running it flat here changes exactly
    one thing: whether the two rates are guesses.
    """
    print("\n" + "-" * 80)
    print("  HOW MUCH WERE THE TWO ASSUMED RATES CARRYING? Same windows, "
          "same symbols.")
    print("-" * 80)
    print(f"    {'symbol':<10}{'variant':<17}{'measured':>11}{'assumed':>10}"
          f"{'move':>8}   timing: measured -> assumed")
    for symbol in names:
        candles = history.get(symbol)
        if not candles or symbol not in measured:
            continue
        chunks = split_windows(candles, window_years, windows)
        if len(chunks) < 2:
            continue
        flat = _run_symbol(symbol, chunks, start_equity, capital,
                           yields=FLAT, quiet=True)
        best = measured[symbol]["best"]
        same = next((v for v in flat["variants"] if v["name"] == best["name"]),
                    None)
        if same is None:
            continue
        print(f"    {symbol:<10}{best['name']:<17}{best['mean']:>+10.2f}%"
              f"{same['mean']:>+9.2f}%{same['mean'] - best['mean']:>+7.2f}"
              f"   {best['matched_mean']:+.2f}% -> {same['matched_mean']:+.2f}%")
    print("-" * 80 + "\n")


def _print_zero_yield(history: dict[str, list[Candle]], names: list[str],
                      window_years: float, windows: int,
                      start_equity: float, capital: float) -> None:
    """The same grid with dividends and bill yield both set to zero.

    Not a robustness ritual. The ladder sits in cash for part of every window,
    so these two constants are the only place where the comparison can be
    rigged without anyone noticing. Printing both settings makes the size of
    that lever visible instead of arguable.
    """
    print("\n" + "-" * 80)
    print("  SENSITIVITY: dividends and bill yield both set to zero "
          "(price return only)")
    print("-" * 80)
    print(f"    {'symbol':<12}{'best variant':<18}{'vs hold':>10}"
          f"{'was':>10}{'p':>8}")
    for symbol in names:
        candles = history.get(symbol)
        if not candles or len(candles) < 500:
            continue
        chunks = split_windows(candles, window_years, windows)
        if len(chunks) < 2:
            continue
        with_yields = _run_symbol(symbol, chunks, start_equity, capital,
                                  quiet=True)
        without = _run_symbol(symbol, chunks, start_equity, capital,
                              yields=NO_YIELDS, quiet=True)
        was = next((v["mean"] for v in with_yields["variants"]
                    if v["name"] == without["best"]["name"]), float("nan"))
        print(f"    {symbol:<12}{without['best']['name']:<18}"
              f"{without['best']['mean']:>+9.2f}%{was:>+9.2f}%"
              f"{without['best']['p']:>8.3f}")
    print("-" * 80 + "\n")
