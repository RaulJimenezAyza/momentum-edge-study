"""What if the entry were a resting limit order instead of a market order?

Earlier in this project maker execution was dismissed as "not honestly
modellable without order-book data". That was too quick. Whether a resting buy
limit would have filled is decidable from OHLC alone: if any later bar trades
at or below the limit price, someone hit it. The parts that genuinely need book
data -- queue position, partial fills, whether you were at the front -- all
push in the same direction, so assuming a fill on touch is the *optimistic*
end, and it can be labelled as such rather than hidden.

Two effects pull against each other, and the point is to measure both rather
than to argue about which wins:

  in favour   A market buy crosses the spread and pays roughly half of it. A
              resting limit does not; the fill happens at the price you named.
              On this universe that is worth a few basis points a side, against
              a measured edge of about four.

  against     ADVERSE SELECTION, and it is not a detail. This is a dip-buying
              strategy, so the limit sits at or below a price that has already
              fallen. It fills exactly when the price keeps falling -- the
              trades you skip are the ones that bounced immediately, which are
              the good ones. The maker saves the spread and may buy the
              selection.

The comparison is paired: the same signals, scored both ways, so the difference
is execution and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .backtest import load_history
from .config import Config
from .costs import CostModel
from .exchange import Candle
from .levels import levels_for
from .nullcheck import Arm, welch
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

# How many bars a resting order is left in the book before it is pulled. Longer
# means more fills and staler signals; this is the strategy's own patience, not
# a free parameter to tune against the result.
FILL_WINDOW_BARS = 3


@dataclass
class Arm2:
    name: str
    net: list[float] = field(default_factory=list)
    gross: list[float] = field(default_factory=list)

    @property
    def arm(self) -> Arm:
        return Arm(self.name, self.net)

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0


def limit_fill(
    future: list[Candle], limit: float, window: int
) -> int | None:
    """Index of the bar that fills a resting buy limit, or None.

    Fill on touch: if a bar trades at or below the limit, the order is assumed
    to have executed at the limit price. Optimistic -- in reality the queue may
    not have reached you -- and deliberately so, because the pessimistic version
    would need book data this does not have.
    """
    for i, c in enumerate(future[:window]):
        if c.low <= limit:
            return i
    return None


async def run_maker(
    cfg: Config,
    days: float = 30.0,
    offset_days: float = 0.0,
    symbols: list[str] | None = None,
    window: int = FILL_WINDOW_BARS,
    maker_bps: float | None = None,
) -> dict:
    history, _ = await load_history(cfg, days, symbols, need_market=False,
                                    offset_days=offset_days)
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    # Binance spot charges the same maker and taker fee at VIP0, so unless a
    # tier is specified the saving here is the spread, not the commission.
    maker_fee = cfg.fees.effective_taker_bps if maker_bps is None else maker_bps
    log.info("%s | maker fee %.2f bps", costs.summary(), maker_fee)

    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    # PAIRING. Two comparisons live here and they run over DIFFERENT
    # populations, which is easy to get wrong and was: the first version
    # compared next-open entry over ALL signals against market entry over only
    # the ones a limit had filled. The 9% a limit misses are worth +0.33%, so
    # that population gap alone produced the whole apparent difference.
    #
    # `taker_all` is the baseline for the timing question (all signals).
    # `taker` is the baseline for the execution question (filled only).
    taker_all = Arm2("market entry, all signals")
    taker = Arm2("market entry")
    maker = Arm2("limit entry, filled")
    unfilled_taker = Arm2("limit missed: what the market entry got")
    next_open = Arm2("market entry at the NEXT bar's open")
    confirmed = Arm2("wait 1 bar, enter only if it rose")
    unconfirmed = Arm2("waited, but it kept falling: skipped")
    fills = 0
    misses = 0

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + window + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last = len(candles) - horizon - window - 1

        for i in range(lookback, last):
            series.update(candles[i])
            result = evaluate(series, cfg.strategy, costs.spread_bps(symbol))
            if isinstance(result, Rejection):
                continue

            levels = levels_for(cfg, series)
            slip = costs.slippage_bps(symbol)
            price = candles[i].close

            # Baseline: cross the spread now.
            t_out = simulate_exit(price, candles[i + 1: i + 1 + horizon],
                                  cfg, levels, slip)
            if t_out is None:
                continue

            # ENTRY TIMING. Every number in this project assumes the fill
            # happens at the close of the bar that produced the signal. You do
            # know that close the instant it happens, so this is not lookahead
            # -- but your order reaches the book afterwards and fills against
            # whatever trades next, not against the print that triggered it.
            #
            # The honest bracket is the next bar's open. On a dip-buy the
            # signal close is often near the low tick of the move, so this is
            # expected to hurt, and the size of the hurt is the point.
            o_out = simulate_exit(candles[i + 1].open,
                                  candles[i + 1: i + 1 + horizon],
                                  cfg, levels_for(cfg, series), slip)
            if o_out is not None:
                next_open.net.append(o_out.net_pct)
                next_open.gross.append(o_out.gross_pct)
                # Same signal, same population, different fill price.
                taker_all.net.append(t_out.net_pct)
                taker_all.gross.append(t_out.gross_pct)

            # CONFIRMATION ENTRY. This is not another idea off a list -- it is
            # the direct consequence of the adverse-selection number above. The
            # signals a resting limit MISSED returned +0.31% at a 68% win rate,
            # and what defines them is that the price did not keep falling. So
            # the tradable version of that population is: wait one bar, and buy
            # only if it closed higher than the signal bar.
            #
            # It costs one bar of the move and it crosses the spread, which is
            # the price of the information. Whether that trade is worth making
            # is exactly what this measures.
            nxt = candles[i + 1]
            if nxt.close > price:
                c_out = simulate_exit(nxt.close,
                                      candles[i + 2: i + 2 + horizon],
                                      cfg, levels_for(cfg, series), slip)
                if c_out is not None:
                    confirmed.net.append(c_out.net_pct)
                    confirmed.gross.append(c_out.gross_pct)
            else:
                unconfirmed.net.append(t_out.net_pct)
                unconfirmed.gross.append(t_out.gross_pct)

            # Maker: rest a limit at the signal price and wait.
            hit = limit_fill(candles[i + 1:], price, window)
            if hit is None:
                misses += 1
                unfilled_taker.net.append(t_out.net_pct)
                unfilled_taker.gross.append(t_out.gross_pct)
                continue

            fills += 1
            start = i + 1 + hit
            m_out = simulate_exit(price, candles[start + 1: start + 1 + horizon],
                                  cfg, levels, 0.0)   # no spread crossed on entry
            if m_out is None:
                continue
            # The exit is still a market order, so it still pays the spread and
            # the taker fee; only the entry leg changes.
            fee_delta = (cfg.fees.effective_taker_bps - maker_fee) / 100.0
            half_spread = slip / 100.0
            maker.net.append(m_out.net_pct + fee_delta + half_spread)
            maker.gross.append(m_out.gross_pct + half_spread)
            taker.net.append(t_out.net_pct)
            taker.gross.append(t_out.gross_pct)

    if not maker.net:
        raise RuntimeError("no limit orders filled -- widen --window")

    _print(cfg, taker, taker_all, maker, unfilled_taker, next_open, confirmed, unconfirmed,
           fills, misses, window, maker_fee)
    return {
        "fills": fills, "misses": misses,
        "fill_rate": fills / max(1, fills + misses),
        "taker_net": taker.arm.mean, "maker_net": maker.arm.mean,
        "missed_net": unfilled_taker.arm.mean,
        "confirmed_net": confirmed.arm.mean, "confirmed_n": confirmed.arm.n,
    }


def _print(cfg, taker, taker_all, maker, missed, next_open, confirmed, unconfirmed,
           fills, misses, window, maker_fee) -> None:
    total = fills + misses
    print("\n" + "=" * 82)
    print(f"  MAKER vs TAKER ENTRY -- {cfg.strategy.direction} "
          f"{cfg.strategy.timeframe}, limit rested {window} bars")
    print("=" * 82)
    print(f"    signals                {total:>8}")
    print(f"    limit filled           {fills:>8}   {fills / max(1, total) * 100:.1f}%")
    print(f"    limit never filled     {misses:>8}   {misses / max(1, total) * 100:.1f}%")
    print(f"    maker fee assumed      {maker_fee:>8.2f} bps/side")
    print("-" * 82)
    print(f"    {'arm':<34}{'n':>7}{'wins':>8}{'gross':>11}{'net':>11}")
    for a in (taker, maker, missed, taker_all, next_open, confirmed, unconfirmed):
        arm = a.arm
        if not arm.n:
            continue
        print(f"    {a.name:<34}{arm.n:>7}{arm.win_rate:>7.1f}%"
              f"{a.gross_mean:>+10.4f}%{arm.mean:>+10.4f}%")
    print("-" * 82)

    gain = maker.arm.mean - taker.arm.mean
    print(f"    same signals, execution only:  {gain:+.4f}% per filled trade")

    # Adverse selection: did the skipped ones turn out better than the taken?
    if missed.arm.n >= 30 and taker.arm.n >= 30:
        selection = missed.arm.mean - taker.arm.mean
        t, p = welch(missed.arm, taker.arm)
        print(f"    the trades the limit MISSED were {selection:+.4f}% "
              f"vs the ones it caught (p={p:.3f})")
        if selection > 0 and p < 0.05:
            print("\n  ADVERSE SELECTION CONFIRMED. The limit fills when price keeps")
            print("  falling and misses the immediate bounces, which are the good")
            print("  ones. Saving the spread costs more than the spread is worth.")
        elif selection > 0:
            print("\n  The missed trades look better but not significantly. Adverse")
            print("  selection is suggested, not established.")
        else:
            print("\n  No adverse selection detected: the skipped trades were no")
            print("  better than the taken ones.")

    if next_open.arm.n >= 100:
        print()
        slip_cost = next_open.arm.mean - taker_all.arm.mean
        _, p_o = welch(next_open.arm, taker_all.arm)
        print("  ENTRY TIMING, the assumption underneath every other number")
        print(f"  (paired over all {taker_all.arm.n} signals, not the filled subset):")
        print(f"    fill at the signal bar's close (assumed): {taker_all.arm.mean:+.4f}%")
        print(f"    fill at the next bar's open (realistic):  {next_open.arm.mean:+.4f}%")
        print(f"    cost of the assumption {slip_cost:+.4f}% per trade (p = {p_o:.3f})")
        if slip_cost < -0.01:
            print("    Every figure in this repo is optimistic by roughly that")
            print("    much. It is a real cost of acting on a closed bar, not a")
            print("    modelling artefact, and it applies to the random arm too")
            print("    -- so the EDGE is largely unaffected, the NET is not.")
        elif slip_cost > 0.01:
            print("    Entering a bar later is BETTER, which on a dip-buy means")
            print("    the signal close sits below where the bounce starts.")
        else:
            print("    Negligible. The close-to-open gap does not move this.")

    if confirmed.arm.n >= 100:
        print()
        delta = confirmed.arm.mean - taker.arm.mean
        t2, p2 = welch(confirmed.arm, taker.arm)
        print("  CONFIRMATION ENTRY, the hypothesis this measurement produced:")
        print(f"    wait one bar and buy only if it rose: {confirmed.arm.mean:+.4f}%"
              f" on {confirmed.arm.n} trades")
        print(f"    against buying immediately:           {taker.arm.mean:+.4f}%")
        print(f"    difference {delta:+.4f}% (p = {p2:.3f})")
        if delta > 0 and p2 < 0.05 and confirmed.arm.mean > 0:
            print("    NET POSITIVE and better than entering at once. Replicate")
            print("    on non-overlapping windows before believing it.")
        elif delta > 0 and p2 < 0.05:
            print("    Better, but still negative. The confirmation is worth")
            print("    something and not enough.")
        else:
            print("    No better. The bar of confirmation costs as much as the")
            print("    selection it avoids, which is what an efficient market")
            print("    would do to a free lunch this obvious.")

    print()
    if gain > 0 and maker.arm.mean > 0:
        print(f"  Maker entry is NET POSITIVE at {maker.arm.mean:+.4f}%. That is the")
        print("  first positive net in this project -- replicate it on")
        print("  non-overlapping windows before believing a word of it, and")
        print("  remember fill-on-touch is the optimistic end of the range.")
    elif gain > 0:
        print(f"  Maker entry improves net by {gain:+.4f}% but is still negative")
        print(f"  at {maker.arm.mean:+.4f}%.")
    else:
        print(f"  Maker entry is WORSE by {gain:+.4f}%. The spread it saves is")
        print("  smaller than the selection it buys.")
    print("=" * 82 + "\n")
