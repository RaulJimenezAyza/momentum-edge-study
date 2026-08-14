"""The Shiller S&P dataset, as a second vendor for the same century.

Everything in the ladder test rests on one price series from one vendor. The
README already records two occasions where Yahoo returned something wrong and
nothing raised -- a silently truncated window, and a symbol that came back
empty -- so a hundred-year claim resting on one download is a claim resting on
one download being right every time.

This is the independent one. Robert Shiller's monthly S&P series runs from
January 1871 and is compiled from the Cowles Commission data and S&P's own
publications, not from Yahoo. It is distributed as CSV by the datahub mirror,
keyless, which matters because the two obvious alternatives are gone: Stooq now
sits behind a proof-of-work bot check, and FRED did not answer.

What it gives and what it does not:

  * `SP500` is the MONTHLY AVERAGE of daily closes, not the month-end close.
    Comparing it to a month-end figure produces a spurious few-tenths-of-a-
    percent disagreement that looks like a vendor error and is not one.
    `vendors.py` checks both conventions rather than assuming this one.
  * `Dividend` is the annualised dividend per share, and it is 0 for the most
    recent months because S&P has not published them yet. Those rows are kept
    and the zero is left visible; silently forward-filling the last known
    dividend into the present is how a data gap becomes a measurement.
  * It is monthly. It cannot see a 20% intra-month drawdown, so it is a check
    on the price series, not a substitute for the daily one.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
FILENAME = "shiller_sp500_monthly.csv"

# The series gains one row a month, so a cached copy is stale after about that
# long. Unlike a closed candle, an old row here can also be REVISED -- CPI and
# earnings both are -- which is the other reason not to cache it forever.
MAX_AGE_DAYS = 30

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Month:
    open_time: int      # ms at the first of the month, UTC
    year: int
    month: int
    price: float        # monthly AVERAGE of daily closes
    dividend: float     # annualised dollars per share, 0 when unpublished
    cpi: float
    long_rate: float    # 10-year yield, percent
    cape: float         # PE10, 0 before the first ten years exist

    @property
    def dividend_yield_pct(self) -> float:
        if self.price <= 0 or self.dividend <= 0:
            return 0.0
        return self.dividend / self.price * 100.0


def _ms(year: int, month: int) -> int:
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    return int((dt - EPOCH).total_seconds() * 1000)


def _download() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return fh.read().decode("utf-8")


def parse(text: str) -> list[Month]:
    """CSV -> months, dropping rows with no usable price.

    Split out and pure so the parsing can be tested without a network call --
    the failure this guards is a vendor changing a column name and every row
    silently becoming zero.
    """
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    try:
        idx = {name: header.index(name) for name in
               ("Date", "SP500", "Dividend", "Consumer Price Index",
                "Long Interest Rate", "PE10")}
    except ValueError as exc:
        raise ValueError(f"Shiller CSV header changed: {header}") from exc

    out: list[Month] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(idx.values()):
            continue
        try:
            stamp = parts[idx["Date"]].strip()
            year, month = int(stamp[:4]), int(stamp[5:7])
            price = float(parts[idx["SP500"]] or 0)
        except (ValueError, IndexError):
            continue
        if price <= 0:
            continue

        def number(key: str) -> float:
            try:
                return float(parts[idx[key]] or 0)
            except (ValueError, IndexError):
                return 0.0

        out.append(Month(
            open_time=_ms(year, month), year=year, month=month, price=price,
            dividend=number("Dividend"),
            cpi=number("Consumer Price Index"),
            long_rate=number("Long Interest Rate"),
            cape=number("PE10"),
        ))
    out.sort(key=lambda m: m.open_time)
    return out


def load_shiller(cache_dir: Path, max_age_days: float = MAX_AGE_DAYS) -> list[Month]:
    """Cached monthly series. Falls back to a stale copy rather than failing.

    A cross-check that dies because the mirror is down is worse than one that
    runs against last month's file and says so -- the point of the file is the
    1871-2020 half, which does not change.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FILENAME
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86_400
    if fresh:
        return parse(path.read_text(encoding="utf-8"))

    try:
        text = _download()
        months = parse(text)
        if months:
            path.write_text(text, encoding="utf-8")
            log.info("shiller: %d months, %d-%d", len(months),
                     months[0].year, months[-1].year)
            return months
        log.warning("shiller: download parsed to nothing")
    except Exception as exc:                        # noqa: BLE001 - see docstring
        log.warning("shiller: download failed (%s)", exc)

    if path.exists():
        age = (time.time() - path.stat().st_mtime) / 86_400
        log.warning("shiller: using a cached copy %.0f days old", age)
        return parse(path.read_text(encoding="utf-8"))
    return []


def dividend_yield_path(months: list[Month]) -> dict[tuple[int, int], float]:
    """(year, month) -> annual dividend yield in percent, published months only.

    This is the whole reason the file is worth having twice over: the measured
    dividend series from `^SP500TR` starts in 1988, and this one starts in
    1871. It covers the half of the ladder test that had to run on an assumed
    constant.
    """
    return {(m.year, m.month): m.dividend_yield_pct
            for m in months if m.dividend_yield_pct > 0}


def cape_path(months: list[Month]) -> dict[tuple[int, int], float]:
    """(year, month) -> CAPE, published months only.

    PE10 needs ten years of earnings behind it, so the column is 0 until 1881
    and 0 again for any month the compiler could not fill. A zero is dropped
    rather than carried: a valuation rule fed a zero CAPE would read it as the
    cheapest market in history and go all in.
    """
    return {(m.year, m.month): m.cape for m in months if m.cape > 0}


def month_of(open_time_ms: int) -> tuple[int, int]:
    dt = EPOCH + timedelta(milliseconds=open_time_ms)
    return dt.year, dt.month
