"""Validate the spread estimator against spreads the exchange actually quotes.

The cost model decides every conclusion in this project: switching from a
hand-picked 5 bps to per-symbol estimates moved the round trip from 0.30% to
0.17% and every net figure with it. That estimator has never been checked
against ground truth, and it is derived from OHLC rather than from quotes.

Ground truth is available and free: `/api/v3/ticker/bookTicker` returns the
live best bid and ask for every symbol. Comparing the two answers the only
question that matters -- is the estimator biased, and in which direction?

A caveat the numbers cannot remove: the quote is a snapshot of *now*, while the
estimate is an average over the last few thousand minutes. They are not
measuring the same instant, so scatter is expected. Systematic bias is not.
"""

from __future__ import annotations

import logging
import math

from .backtest import load_history
from .config import Config
from .exchange import BinanceREST
from .spread import abdi_ranaldo_bps

log = logging.getLogger(__name__)


async def run_spreadcheck(
    cfg: Config, days: float = 7.0, symbols: list[str] | None = None
) -> dict:
    history, _ = await load_history(cfg, days, symbols, need_market=False)

    async with BinanceREST(cfg.exchange.base_url) as rest:
        quotes = {}
        for row in await rest.book_tickers():
            bid, ask = float(row["bidPrice"]), float(row["askPrice"])
            if bid > 0 and ask > 0:
                quotes[row["symbol"]] = (ask - bid) / ((ask + bid) / 2) * 10_000

    pairs: list[tuple[str, float, float]] = []
    for symbol, candles in history.items():
        live = quotes.get(symbol)
        if live is None:
            continue
        est = abdi_ranaldo_bps(candles)
        pairs.append((symbol, est, live))
    if len(pairs) < 5:
        raise RuntimeError("not enough symbols with both an estimate and a quote")

    _print(cfg, pairs)
    return {"pairs": [{"symbol": s, "estimated": e, "quoted": q} for s, e, q in pairs]}


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation -- the estimator's job is to order symbols correctly,
    and ranks are robust to the handful of very wide books that would dominate
    a Pearson correlation."""
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def _print(cfg: Config, pairs: list[tuple[str, float, float]]) -> None:
    lives = [q for _, _, q in pairs]
    # Ratios over MEASURED symbols only would drop precisely the cases where
    # the estimator failed -- survivorship bias inside the bias check. The
    # unmeasurable ones are counted and reported separately instead.
    measured = [(s, e, q) for s, e, q in pairs if e > 0]
    unmeasurable = [(s, e, q) for s, e, q in pairs if e <= 0]
    ratios = [e / q for _, e, q in measured if q > 0.01]

    def median(xs: list[float]) -> float:
        s = sorted(xs)
        return s[len(s) // 2] if s else 0.0

    print("\n" + "=" * 78)
    print(f"  SPREAD ESTIMATOR vs LIVE QUOTES -- {len(pairs)} symbols")
    print("=" * 78)
    print(f"    {'symbol':<14}{'estimated':>12}{'quoted':>10}{'ratio':>9}")
    worst = sorted(pairs, key=lambda p: abs(math.log((p[1] + 0.1) / (p[2] + 0.1))),
                   reverse=True)
    show = sorted(pairs, key=lambda p: p[2])[:6] + worst[:6]
    seen = set()
    for symbol, est, live in show:
        if symbol in seen:
            continue
        seen.add(symbol)
        r = est / live if live > 0 else float("inf")
        print(f"    {symbol:<14}{est:>11.2f}b{live:>9.2f}b{r:>9.2f}")

    print("-" * 78)
    print(f"    measured           {len(measured):>8} symbols")
    print(f"    UNMEASURABLE       {len(unmeasurable):>8} symbols"
          f"  (real spreads up to {max((q for _, _, q in unmeasurable), default=0):.1f} bps)")
    print(f"    median estimated   {median([e for _, e, _ in measured]):>8.2f} bps"
          f"   (measured only)")
    print(f"    median quoted      {median(lives):>8.2f} bps")
    print(f"    median ratio       {median(ratios):>8.2f}x  (1.0 = unbiased, measured only)")
    rho = _spearman([e for _, e, _ in measured], [q for _, _, q in measured])
    print(f"    rank correlation   {rho:>8.2f}   (measured only)")
    print("-" * 78)

    bias = median(ratios)
    if unmeasurable:
        share = len(unmeasurable) / len(pairs) * 100
        print(f"  {share:.0f}% of symbols are below the estimator's resolution floor.")
        print("  Those fall back to the configured flat slippage, which is")
        print("  pessimistic -- the safe direction. They are NOT assumed cheap.")
        print()
    if rho < 0.5:
        print("  The estimator does NOT rank symbols like the real book does.")
        print("  Per-symbol costs built on it are close to arbitrary; prefer")
        print("  fees.slippage_mode: fixed until this is understood.")
    elif bias > 1.5:
        print(f"  The estimator runs {bias:.1f}x WIDE. Costs are overstated, so every")
        print("  net figure in this repo is pessimistic -- conclusions that say")
        print("  'not profitable' are safe; any that said 'profitable' would not be.")
    elif bias < 0.67:
        print(f"  The estimator runs {bias:.1f}x TIGHT. Costs are understated, which")
        print("  flatters every net figure. Results need re-checking with")
        print("  fees.impact_multiplier raised to compensate.")
    else:
        print(f"  Unbiased to within {abs(1 - bias) * 100:.0f}% and ranks symbols "
              f"correctly (rho={rho:.2f}).")
        print("  The per-symbol cost model is doing what it claims.")
    print("\n  Caveat: the quote is one instant, the estimate averages thousands of")
    print("  minutes. Scatter between them is expected; systematic bias is not.")
    print("=" * 78 + "\n")
