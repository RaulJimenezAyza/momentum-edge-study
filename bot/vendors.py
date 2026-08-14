"""Do two vendors agree about the same century of prices?

The ladder result is ten non-overlapping decades of `^GSPC` from one vendor.
Everywhere else in this project a number that rests on a single unverified
input has eventually turned out to rest on a bug, most of which flattered the
answer. "Yahoo returned a hundred years of correct closes" was, until this file
existed, an assumption with exactly that shape.

So: Yahoo's daily `^GSPC`, aggregated to months, against Shiller's independent
monthly series compiled from Cowles and S&P publications. Two conventions are
tested rather than one, because Shiller's column is a monthly AVERAGE of daily
closes and comparing it to a month-end close manufactures a disagreement of a
few tenths of a percent that is not a vendor error. Which convention fits
better is itself the first result, and the code does not assume the answer.

What agreement here does and does not buy:

  * It buys: the price path behind the ladder finding is not a Yahoo artifact.
  * It does not buy: anything about the daily bars BETWEEN month boundaries,
    which is where the -20% thresholds actually fire. A monthly series cannot
    see an intra-month drawdown. This is a check on the level, not the path.
  * Nor anything about the pre-1962 OHLC, which Yahoo reports with open, high
    and low all equal to the close. That is not a disagreement between vendors,
    it is one vendor not having the data, and `--trigger intrabar` is therefore
    identical to `close` over most of the century.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config
from .exchange import Candle
from .findings import record
from .shiller import load_shiller, month_of

log = logging.getLogger(__name__)

# A month where the two vendors differ by more than this is not rounding, a
# convention mismatch or a revision -- it is one of them being wrong about what
# the index did, and it gets named in the output with its date.
DISAGREEMENT_BPS = 100.0


@dataclass(frozen=True)
class MonthStat:
    mean: float
    last: float
    first: float
    bars: int


def monthly_stats(candles: list[Candle]) -> dict[tuple[int, int], MonthStat]:
    """(year, month) -> the three summaries a monthly series might have used.

    Pure, so the aggregation can be tested without a download. All three are
    kept because "which statistic is this vendor publishing" is the open
    question, and because when a single month disagrees the answer is usually
    that THAT ROW used a different one.
    """
    buckets: dict[tuple[int, int], list[Candle]] = {}
    for candle in candles:
        buckets.setdefault(month_of(candle.open_time), []).append(candle)
    out: dict[tuple[int, int], MonthStat] = {}
    for key, group in buckets.items():
        group.sort(key=lambda c: c.open_time)
        closes = [c.close for c in group]
        out[key] = MonthStat(sum(closes) / len(closes), closes[-1], closes[0],
                             len(closes))
    return out


# Within this, two numbers are the same number reported to different precision.
SAME_BPS = 5.0


def diagnose(theirs: float, stat: MonthStat) -> str:
    """Which of Yahoo's monthly statistics does a disagreeing row actually equal?

    A bare list of worst months invites the reader to assume both vendors are
    somewhat wrong. Usually neither is: one row simply carries a different
    statistic, and saying which one turns an unexplained outlier into a closed
    question.
    """
    for name, value in (("monthly mean", stat.mean),
                        ("month-end close", stat.last),
                        ("first close of the month", stat.first)):
        if _bps(theirs, value) <= SAME_BPS:
            return f"= Yahoo's {name}"
    return "matches no Yahoo statistic"


def _bps(a: float, b: float) -> float:
    return abs(a - b) / b * 10_000.0 if b > 0 else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


async def run_vendorcheck(cfg: Config, symbol: str = "^GSPC",
                          min_bars: int = 15) -> dict:
    """Yahoo's daily closes against Shiller's monthly series, by decade."""
    from .yahoo import load_equity_history

    months = load_shiller(cfg.path("data/shiller"))
    if not months:
        raise RuntimeError("Shiller series unavailable and nothing cached")

    history = load_equity_history([symbol], "1d", 40_000.0,
                                  cfg.path("data/yahoo"))
    candles = history.get(symbol)
    if not candles:
        raise RuntimeError(f"no Yahoo history for {symbol}")

    yahoo = monthly_stats(candles)
    theirs = {(m.year, m.month): m.price for m in months}

    # A partial month at either end is a real difference in coverage, not a
    # disagreement about price. Dropping it is the only way the tails do not
    # dominate the worst-case row.
    shared = sorted(k for k in set(yahoo) & set(theirs)
                    if yahoo[k].bars >= min_bars)
    if not shared:
        raise RuntimeError("no month has both vendors and enough bars")

    mean_diffs = [_bps(yahoo[k].mean, theirs[k]) for k in shared]
    last_diffs = [_bps(yahoo[k].last, theirs[k]) for k in shared]
    by_mean = _median(mean_diffs) <= _median(last_diffs)
    chosen = mean_diffs if by_mean else last_diffs

    decades: dict[int, list[float]] = {}
    for key, diff in zip(shared, chosen):
        decades.setdefault(key[0] // 10 * 10, []).append(diff)

    worst = [(k, d, diagnose(theirs[k], yahoo[k]))
             for k, d in sorted(zip(shared, chosen), key=lambda kv: -kv[1])[:5]]
    bad = [(k, d) for k, d in zip(shared, chosen) if d > DISAGREEMENT_BPS]
    explained = sum(1 for k, d in bad
                    if diagnose(theirs[k], yahoo[k]) != "matches no Yahoo statistic")

    _print(symbol, shared, decades, worst, bad, by_mean,
           _median(mean_diffs), _median(last_diffs), months, explained)

    out = {
        "symbol": symbol, "months": len(shared),
        "first": f"{shared[0][0]}-{shared[0][1]:02d}",
        "last": f"{shared[-1][0]}-{shared[-1][1]:02d}",
        "convention": "monthly mean" if by_mean else "month end",
        "median_bps": _median(chosen),
        "median_bps_mean_convention": _median(mean_diffs),
        "median_bps_end_convention": _median(last_diffs),
        "months_over_threshold": len(bad),
        "months_over_threshold_explained": explained,
        "threshold_bps": DISAGREEMENT_BPS,
    }
    record(cfg, "vendorcheck", out)
    return out


def _print(symbol, shared, decades, worst, bad, by_mean, med_mean, med_last,
           months, explained) -> None:
    print("\n" + "=" * 80)
    print(f"  TWO VENDORS, ONE CENTURY -- {symbol} (Yahoo, daily) vs Shiller "
          "(monthly)")
    print("=" * 80)
    print(f"    months compared           {len(shared):>8,}   "
          f"{shared[0][0]}-{shared[0][1]:02d} to {shared[-1][0]}-{shared[-1][1]:02d}")
    print(f"    Shiller series covers     {months[0].year}-{months[-1].year}")
    print("-" * 80)
    print("    which convention is Shiller's SP500 column?")
    print(f"      monthly mean of closes  {med_mean:>8.1f} bps median difference"
          + ("   <-- fits" if by_mean else ""))
    print(f"      month-end close         {med_last:>8.1f} bps median difference"
          + ("" if by_mean else "   <-- fits"))
    print("-" * 80)
    print(f"    {'decade':<10}{'months':>9}{'median':>11}{'worst':>11}"
          f"{'over ' + str(int(DISAGREEMENT_BPS)) + 'bps':>14}")
    for decade in sorted(decades):
        values = decades[decade]
        over = sum(1 for v in values if v > DISAGREEMENT_BPS)
        print(f"    {str(decade) + 's':<10}{len(values):>9}"
              f"{_median(values):>10.1f}b{max(values):>10.1f}b{over:>14}")
    print("-" * 80)
    print("    worst five months, and what the other vendor's row actually holds:")
    for (year, month), diff, why in worst:
        print(f"      {year}-{month:02d}{diff:>12.1f} bps   {why}")
    print("-" * 80)

    median_all = _median([d for values in decades.values() for d in values])
    if median_all < 30 and not bad:
        print(f"  THE TWO VENDORS AGREE. Median disagreement {median_all:.1f} bps "
              f"across {len(shared):,} months\n  and not one month past "
              f"{DISAGREEMENT_BPS:.0f} bps. The century of prices behind the "
              "ladder result is not\n  a Yahoo artifact. It says nothing about "
              "the daily path between month ends,\n  which is where the "
              "thresholds actually fire.")
    elif median_all < 30 and explained == len(bad):
        print(f"  THE TWO VENDORS AGREE, with {len(bad)} explained exception(s). "
              f"Median disagreement\n  {median_all:.1f} bps across {len(shared):,} "
              f"months. Every month past {DISAGREEMENT_BPS:.0f} bps is a row where "
              "the monthly\n  series carries a DIFFERENT STATISTIC -- a month-end "
              "or a first close instead of\n  the month's mean -- so Yahoo is "
              "self-consistent in each one and the disagreement\n  is bookkeeping "
              "on their side, not a price this project got wrong.\n"
              "  It says nothing about the daily path between month ends, which "
              "is where the\n  thresholds actually fire.")
    elif median_all < 30:
        print(f"  Broad agreement -- median {median_all:.1f} bps -- but "
              f"{len(bad) - explained} month(s) past {DISAGREEMENT_BPS:.0f} bps "
              "match no statistic\n  of either vendor. Those are listed above and "
              "are worth opening before the ladder\n  numbers covering them are "
              "quoted.")
    else:
        print(f"  THE VENDORS DO NOT AGREE: median {median_all:.1f} bps. Either "
              "the convention is\n  wrong, the alignment is off by a month, or "
              "one series is not what it claims.\n  Until this is resolved, "
              "every result built on this price series is suspect.")
    print("=" * 80 + "\n")
