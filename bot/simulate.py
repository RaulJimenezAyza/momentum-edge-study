"""Pure exit simulation, decoupled from live position state.

The ML labeller needs to answer "what would this trade have returned?" for tens
of thousands of historical bars. Doing that through the stateful Strategy is
awkward, so the exit rules live here as a function too.

The two implementations MUST agree. tests/test_offline.py asserts they do on
randomised candle paths -- if you change an exit rule in one place, that test
fails until you change it in the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .exchange import Candle
from .levels import Levels


@dataclass(slots=True)
class Outcome:
    net_pct: float       # after entry+exit fees and slippage
    gross_pct: float
    reason: str
    bars_held: int
    mfe_pct: float       # max favourable excursion
    mae_pct: float       # max adverse excursion


def simulate_exit(
    signal_price: float,
    future: list[Candle],
    cfg: Config,
    levels: Levels,
    slippage_bps: float,
) -> Outcome | None:
    """Enter at `signal_price` and run the live exit rules over `future`.

    `levels` and `slippage_bps` are REQUIRED, and that is the point. Both used
    to default to the config's flat values, and twice a caller was written that
    left them out -- the ML labeller and the parameter sweep -- each silently
    scoring a strategy with different stops and different costs than the bot
    actually runs. The defaults made the wrong call the easy one to write.

    They are per-symbol quantities. There is no correct global default, so
    there is no default. Callers with nothing better to pass use
    `fixed_levels(cfg)` and `cfg.fees.slippage_bps` explicitly, which at least
    says so out loud.

    Returns None when there is not enough forward data to resolve the trade,
    which keeps unresolved trades out of the training set rather than labelling
    them with a truncated result.
    """
    if not future or signal_price <= 0:
        return None

    lv = levels
    slip = slippage_bps / 10_000.0
    fee = cfg.fees.effective_taker_bps / 10_000.0
    mode = cfg.risk.intrabar_fill

    entry = signal_price * (1.0 + slip)
    stop, target = lv.apply(entry)
    peak = entry
    armed = False
    max_bars = cfg.max_hold_bars

    mfe = mae = 0.0
    exit_price: float | None = None
    reason = ""
    bars = 0

    for i, c in enumerate(future[:max_bars], start=1):
        bars = i
        mfe = max(mfe, (c.high / entry - 1.0) * 100.0)
        mae = min(mae, (c.low / entry - 1.0) * 100.0)

        # Same ordering as Strategy._check_exit: settle against the levels in
        # force at bar open, ratchet only on survival. Which barrier wins when
        # the candle touched both is the caller's declared assumption -- see
        # RiskConfig.intrabar_fill for why it is bracketed rather than chosen.
        hit_stop = c.low <= stop
        hit_target = c.high >= target
        if hit_stop and hit_target:
            stop_first = (
                True if mode == "pessimistic"
                else False if mode == "optimistic"
                # "open": the barrier the bar opened nearer to is the one a
                # path from that open is more likely to reach first.
                else abs(c.open - stop) <= abs(c.open - target)
            )
            hit_stop, hit_target = stop_first, not stop_first

        if hit_stop:
            exit_price = stop
            reason = "trailing_stop" if armed else "stop_loss"
            break
        if hit_target:
            exit_price, reason = target, "take_profit"
            break
        if i >= max_bars:
            exit_price, reason = c.close, "max_hold"
            break

        peak = max(peak, c.high)
        if not armed and (peak / entry - 1.0) * 100.0 >= lv.trail_activate_pct:
            armed = True
        if armed:
            stop = max(stop, min(peak * (1.0 - lv.trail_pct / 100.0), target))

    if exit_price is None:
        return None  # ran out of data before the trade resolved

    # A target exit is a resting limit: it does not cross the spread and it
    # pays the maker fee. A stop or a time exit is a market order and pays
    # both. Treating every exit as a market order overstated the cost of the
    # one exit type the strategy actually wants.
    maker_exit = reason == "take_profit" and cfg.fees.target_exit_is_maker
    exit_slip = 0.0 if maker_exit else slip
    exit_fee = (cfg.fees.effective_maker_bps / 10_000.0) if maker_exit else fee

    fill = exit_price * (1.0 - exit_slip)
    gross_pct = (fill / entry - 1.0) * 100.0
    net_pct = gross_pct - (fee + exit_fee) * 100.0
    return Outcome(net_pct, gross_pct, reason, bars, mfe, mae)
