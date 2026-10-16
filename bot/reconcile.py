"""Did the live bot see what the backtest says it should have seen?

Every conclusion in this project comes from replaying history. The live bot is
assumed to behave the same way, and that assumption has been tested only at the
unit level -- `simulate_exit` matches `Strategy`, the ML labels match the
strategy, and so on. Nothing has ever compared a real live session against a
replay of the same minutes.

This does. It reads the signals the live bot actually logged, replays those
exact minutes and symbols from cached candles, and lines the two lists up.

Disagreement is not automatically a bug, and the interesting part is which
direction it goes:

  live-only      the replay missed a signal the bot took. Candle data differs
                 from the stream, or the bot acted on a partial bar.
  replay-only    the bot missed a signal the data contains. A websocket gap, a
                 restart, or the real-time spread gate rejecting something the
                 estimated spread would allow.

The second kind is expected here: live reads the true bid/ask from the book,
while the replay only has an estimate from candles. Quantifying it is the point
-- an unexplained gap in either direction means the backtest is measuring a
strategy the bot is not running.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .backtest import load_history
from .broker import PaperBroker
from .config import Config
from .costs import CostModel
from .risk import RiskManager
from .series import Series
from .signals import diagnose
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot
from .strategy import Strategy

log = logging.getLogger(__name__)


@dataclass
class LiveSignal:
    symbol: str
    time_ms: int
    spread_bps: float


def read_live_signals(path: Path) -> list[LiveSignal]:
    if not path.exists():
        raise RuntimeError(
            f"{path} not found -- run `python run.py live` for a while first"
        )
    out: list[LiveSignal] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                stamp = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                out.append(LiveSignal(
                    row["symbol"],
                    int(stamp.replace(tzinfo=UTC).timestamp() * 1000),
                    float(row.get("spread_bps") or 0.0),
                ))
            except (KeyError, ValueError) as exc:
                log.warning("skipping unparseable signal row: %s", exc)
    return sorted(out, key=lambda s: s.time_ms)


async def run_reconcile(cfg: Config, pad_hours: float = 1.0) -> dict:
    live = read_live_signals(cfg.path(cfg.logging.signals_csv))
    if not live:
        raise RuntimeError("the live signal log is empty")

    first, last = live[0].time_ms, live[-1].time_ms
    span_h = (last - first) / 3_600_000
    log.info("%d live signals spanning %.1f hours", len(live), span_h)

    # Cover the session plus warm-up, rounded out to whole days because that is
    # the granularity load_history works in.
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    offset_days = max(0.0, (now_ms - last) / 86_400_000 - pad_hours / 24)
    days = (last - first) / 86_400_000 + 2 * pad_hours / 24 + \
        cfg.strategy.lookback_bars * cfg.bar_ms / 86_400_000
    days = max(days, 0.5)

    symbols = sorted({s.symbol for s in live})
    history, _ = await load_history(cfg, days, symbols, need_market=False,
                                    offset_days=offset_days)
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )

    # Replay through the full Strategy, not bare evaluate(). The first version
    # of this used evaluate() alone and reported the two bars immediately after
    # every real signal as "the bot missed them" -- of course it did, it was
    # already holding the position. Position state and cooldown are part of
    # what the bot does, so they have to be part of the comparison.
    lookback = cfg.strategy.lookback_bars
    broker = PaperBroker(cfg, costs=costs)
    strategy = Strategy(cfg, broker, RiskManager(cfg))
    for symbol in history:
        strategy.spread_bps[symbol] = costs.spread_bps(symbol)

    tape = sorted(
        ((c.open_time, symbol, c)
         for symbol, candles in history.items()
         for c in candles[lookback:]),
        key=lambda row: (row[0], row[1]),
    )
    for symbol, candles in history.items():
        strategy.seed(symbol, candles[:lookback])

    replay: set[tuple[str, int]] = set()
    seen_before = 0
    for stamp, symbol, candle in tape:
        seen_before = len(strategy.signals_seen)
        strategy.on_candle(symbol, candle)
        if len(strategy.signals_seen) > seen_before and first <= stamp <= last:
            replay.add((symbol, stamp))

    live_set = {(s.symbol, s.time_ms) for s in live}
    exact = live_set & replay

    # Same event, neighbouring bar. The live gate uses the true quoted spread
    # and the replay an estimate, so a symbol sitting near a threshold can
    # trigger a minute earlier or later. Counting that as "the bot missed a
    # signal" overstates the disagreement -- UNIUSDT fired live at 07:36 and in
    # the replay at 07:37 and 07:38, which is one event, not three.
    tolerance = 2 * cfg.bar_ms
    near: set[tuple[str, int]] = set()
    near_replay: set[tuple[str, int]] = set()
    for symbol, stamp in live_set - exact:
        hits = [(s, r) for s, r in replay
                if s == symbol and abs(r - stamp) <= tolerance]
        if hits:
            near.add((symbol, stamp))
            near_replay.update(hits)

    both = exact
    live_only = live_set - exact - near
    replay_only = replay - exact - near_replay

    # Explain, rather than merely report, every signal the replay could not
    # reproduce: re-run the gates at that exact bar and name what blocked it.
    reasons: dict[tuple[str, int], str] = {}
    live_spreads = {(s.symbol, s.time_ms): s.spread_bps for s in live}
    for symbol, stamp in live_only:
        candles = history.get(symbol, [])
        idx = next((i for i, c in enumerate(candles) if c.open_time == stamp), None)
        if idx is None or idx < lookback:
            reasons[(symbol, stamp)] = "that minute is not in the candle data"
            continue
        series = Series(symbol, lookback)
        for c in candles[max(0, idx - lookback + 1): idx + 1]:
            series.update(c)
        d = diagnose(series, cfg.strategy, costs.spread_bps(symbol))
        live_bps = live_spreads.get((symbol, stamp), 0.0)
        reasons[(symbol, stamp)] = (
            f"{d.blocking_gate or 'nothing'} "
            f"(replay spread {costs.spread_bps(symbol):.1f}b vs live {live_bps:.1f}b)"
        )

    _print(cfg, live, both, near, live_only, replay_only, span_h, costs, reasons)
    return {
        "live": len(live_set), "replay": len(replay), "matched": len(both),
        "near_matched": len(near),
        "live_only": sorted(f"{s} @ {_ts(t)}" for s, t in live_only),
        "replay_only": sorted(f"{s} @ {_ts(t)}" for s, t in replay_only),
    }


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%m-%d %H:%M")


def _print(cfg, live, both, near, live_only, replay_only, span_h, costs,
           reasons=None) -> None:
    print("\n" + "=" * 78)
    print(f"  LIVE vs REPLAY -- {len(live)} live signals over {span_h:.1f} hours")
    print("=" * 78)
    print(f"    same bar               {len(both):>5}")
    print(f"    within 2 bars          {len(near):>5}   (same event, gate fired a")
    print("                                  minute apart)")
    print(f"    live only              {len(live_only):>5}   (replay cannot reproduce)")
    print(f"    replay only            {len(replay_only):>5}   (the bot did not take)")

    for title, items in (("LIVE ONLY", live_only), ("REPLAY ONLY", replay_only)):
        if items:
            print(f"\n    {title}:")
            for symbol, t in sorted(items, key=lambda x: x[1])[:12]:
                note = (reasons or {}).get((symbol, t), "")
                print(f"      {symbol:<14}{_ts(t)}  {note}")
            if len(items) > 12:
                print(f"      ... and {len(items) - 12} more")

    print("-" * 78)
    if not live:
        return
    exact_pct = len(both) / max(1, len(live)) * 100
    total_pct = (len(both) + len(near)) / max(1, len(live)) * 100
    print(f"    agreement: {exact_pct:.0f}% exact, {total_pct:.0f}% allowing a "
          f"two-bar shift")

    if live_only:
        print("\n  Signals the replay could NOT reproduce. This is the direction that")
        print("  matters: the bot acted on something the historical candles do not")
        print("  contain, so the backtest is not measuring what the bot does.")
        print("  Check for partial-bar entries or a candle/stream mismatch.")
    else:
        print("\n  Every live signal reproduces from historical candles. The backtest")
        print("  is replaying the same events the bot actually saw.")

    if replay_only:
        print("\n  Signals the bot did not take but the data contains. Expected in")
        print("  part: live reads the true bid/ask, the replay only estimates the")
        print("  spread from candles, so the gate disagrees near the threshold.")
        print("  Also caused by websocket gaps and restarts -- check the repair")
        print("  count in the viewer before reading anything else into it.")
    print("=" * 78 + "\n")
