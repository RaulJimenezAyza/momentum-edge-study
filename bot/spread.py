"""Estimate each symbol's effective bid-ask spread from its own candles.

Why this exists: the cost model used a flat 5 bps of slippage per side, chosen
by hand and labelled "deliberately pessimistic". That is 10 bps per round trip
of invented cost, sitting on top of a measured edge of about 1 bp. An
assumption that large cannot stay a guess -- it decides the answer.

The estimator is Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads
from Daily Close, High, and Low Prices" (Review of Financial Studies). It uses
the fact that the mid-range (high+low)/2 is a cleaner proxy for the efficient
price than the close, which is contaminated by whichever side of the book the
last trade hit:

    s^2 = 4 * E[ (c_t - eta_t) * (c_t - eta_{t+1}) ]

with c = log close and eta = (log high + log low) / 2. The expectation is
negative when there is no spread, so the estimate is floored at zero.

This is still an estimate of the *quoted* spread, not of realised market impact
on a real order. It is honest about the cost of crossing the book on a liquid
pair; it does not model pushing size through a thin one. That is why the result
is floored and capped, and why `slippage_mode: fixed` remains available for
anyone who wants the pessimistic number back.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from .exchange import BinanceREST, Candle

log = logging.getLogger(__name__)

# Where the exchange-quoted spreads are cached, relative to the project root.
QUOTE_SNAPSHOT = "data/quoted_spreads.json"


def abdi_ranaldo_bps(candles: list[Candle], max_bars: int = 5000) -> float:
    """Effective proportional spread in basis points. 0 when unmeasurable."""
    window = candles[-max_bars:] if len(candles) > max_bars else candles
    if len(window) < 30:
        return 0.0

    etas: list[float] = []
    closes: list[float] = []
    for c in window:
        if c.high <= 0 or c.low <= 0 or c.close <= 0:
            continue
        etas.append((math.log(c.high) + math.log(c.low)) / 2.0)
        closes.append(math.log(c.close))
    if len(closes) < 30:
        return 0.0

    terms = [
        4.0 * (closes[t] - etas[t]) * (closes[t] - etas[t + 1])
        for t in range(len(closes) - 1)
    ]
    n = len(terms)
    mean = sum(terms) / n
    if mean <= 0:
        # A negative expectation means the noise swamped the signal: the true
        # spread is too small to measure on this data, not that it is negative.
        return 0.0

    # The mean is an unbiased estimate of s^2, but its noise scales with the
    # square of the bar's volatility, and sqrt() of a noisy positive number is
    # not a noisy estimate of the spread -- it is an estimate of the noise.
    # Measured on synthetic bars with a known 2 bps spread: at 0.05% bar
    # volatility the estimator returns 1.8 bps, and at 2% it returns 26.6.
    # Thirteen times too high, with no sign that anything is wrong.
    #
    # So require the estimate to clear its own standard error before it is
    # believed. Adjacent terms share a close and are correlated, so the
    # effective sample is taken as n/2 rather than n.
    var = sum((x - mean) ** 2 for x in terms) / max(1, n - 1)
    se = math.sqrt(var / max(1.0, n / 2.0))
    if mean < 2.0 * se:
        return 0.0
    return math.sqrt(mean) * 10_000.0


def estimate_spreads(
    history: dict[str, list[Candle]], floor_bps: float = 0.5, cap_bps: float = 50.0
) -> dict[str, float]:
    """Per-symbol spread estimates. Symbols it cannot measure are OMITTED.

    Omission is deliberate and it matters. An earlier version floored the
    unmeasurable ones at 0.5 bps, which quietly asserted the tightest book in
    the universe for exactly the symbols the estimator had failed on. Checked
    against live quotes, some of those were really 5-22 bps wide -- a 10-40x
    understatement of cost, in the direction that flatters every result.

    Leaving them out makes the caller fall back to the configured flat
    slippage, which is pessimistic. Failing to measure something is not
    evidence that it is cheap.
    """
    out: dict[str, float] = {}
    unmeasurable = 0
    for symbol, candles in history.items():
        raw = abdi_ranaldo_bps(candles)
        if raw <= 0:
            unmeasurable += 1
            continue
        out[symbol] = min(max(raw, floor_bps), cap_bps)
    if out:
        values = sorted(out.values())
        log.info(
            "spread estimates: median %.2f bps, range %.2f-%.2f bps over %d symbols"
            " (%d unmeasurable -> flat fallback)",
            values[len(values) // 2], values[0], values[-1], len(values), unmeasurable,
        )
    return out


# --------------------------------------------------------------------------
# Ground truth, for the symbols the estimator cannot reach
# --------------------------------------------------------------------------
async def refresh_quote_snapshot(rest: BinanceREST, path: Path) -> dict[str, float]:
    """Store the exchange's current quoted spread for every symbol.

    A third of the universe sits below the estimator's resolution floor, and
    those were falling back to a flat guess. Reconciling a live session against
    a replay showed what that costs: PUMPUSDT and BICOUSDT were charged 10 bps
    and gated out of the backtest entirely, while the book was quoting 3.8 and
    2.9. Every backtest was silently dropping a third of the universe, chosen
    by which symbols happened to be hard to measure.

    A quote is one instant and a backtest spans weeks, so this is not a
    substitute for the estimator where the estimator works. It is a far better
    fallback than a constant for where it does not.
    """
    out: dict[str, float] = {}
    for row in await rest.book_tickers():
        bid, ask = float(row["bidPrice"]), float(row["askPrice"])
        if bid > 0 and ask > 0:
            out[row["symbol"]] = (ask - bid) / ((ask + bid) / 2) * 10_000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    log.info("quote snapshot: %d symbols written to %s", len(out), path.name)
    return out


def load_quote_snapshot(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except Exception as exc:  # noqa: BLE001 - a stale snapshot must not be fatal
        log.warning("quote snapshot unreadable (%s), ignoring", exc)
        return {}
