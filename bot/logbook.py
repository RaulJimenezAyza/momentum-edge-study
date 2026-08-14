"""CSV persistence and the performance summary."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .broker import Trade
from .signals import Signal

log = logging.getLogger(__name__)


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class CsvWriter:
    """Append-only CSV that writes its header on first use."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        path.parent.mkdir(parents=True, exist_ok=True)
        self._started = path.exists() and path.stat().st_size > 0

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if not self._started:
                writer.writeheader()
                self._started = True
            writer.writerow({k: row.get(k) for k in self.fieldnames})


TRADE_FIELDS = [
    "symbol", "entry_time", "exit_time", "entry_price", "exit_price", "qty",
    "gross_pnl", "fees", "net_pnl", "net_pct", "exit_reason", "hold_minutes",
    "equity_after",
]

SIGNAL_FIELDS = [
    "symbol", "time", "price", "ret_pct", "zscore", "volume_mult",
    "extension_pct", "breakout_level", "spread_bps",
]


class Logbook:
    def __init__(self, trades_path: Path, equity_path: Path, signals_path: Path):
        self.trades = CsvWriter(trades_path, TRADE_FIELDS)
        self.equity = CsvWriter(equity_path, ["time", "equity", "open_positions"])
        self.signals = CsvWriter(signals_path, SIGNAL_FIELDS)

    def record_trade(self, t: Trade) -> None:
        row = asdict(t)
        row["entry_time"] = _ts(t.entry_time_ms)
        row["exit_time"] = _ts(t.exit_time_ms)
        self.trades.append(row)

    def record_signal(self, s: Signal) -> None:
        row = s.as_row()
        row["time"] = _ts(s.time_ms)
        self.signals.append(row)

    def record_equity(self, time_ms: int, equity: float, open_positions: int) -> None:
        self.equity.append(
            {"time": _ts(time_ms), "equity": round(equity, 4), "open_positions": open_positions}
        )


def summarize(trades: list[Trade], starting_equity: float, final_equity: float) -> dict:
    """Standard performance stats. Empty-safe."""
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "starting_equity": starting_equity,
            "final_equity": final_equity,
            "return_pct": (final_equity / starting_equity - 1.0) * 100.0
            if starting_equity
            else 0.0,
        }

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    rets = [t.net_pct for t in trades]
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)

    # Peak-to-trough on the realised equity curve.
    curve = [t.equity_after for t in trades]
    peak, max_dd = curve[0], 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak * 100.0)

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100.0,
        "avg_win_pct": sum(t.net_pct for t in wins) / len(wins) if wins else 0.0,
        "avg_loss_pct": sum(t.net_pct for t in losses) / len(losses) if losses else 0.0,
        "expectancy_pct": mean,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_fees": sum(t.fees for t in trades),
        "gross_pnl": sum(t.gross_pnl for t in trades),
        "net_pnl": sum(t.net_pnl for t in trades),
        "max_drawdown_pct": max_dd,
        "return_pct": (final_equity / starting_equity - 1.0) * 100.0
        if starting_equity
        else 0.0,
        "sharpe_per_trade": (mean / std) if std > 0 else 0.0,
        "avg_hold_minutes": sum(t.hold_minutes for t in trades) / n,
        "starting_equity": starting_equity,
        "final_equity": final_equity,
        "exit_reasons": reasons,
    }


def print_summary(stats: dict, rejections: dict[str, int] | None = None) -> None:
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    if stats["trades"] == 0:
        print("  No trades taken.")
    else:
        rows = [
            ("Trades", f"{stats['trades']}"),
            ("Win rate", f"{stats['win_rate_pct']:.1f}%"),
            ("Avg win / avg loss", f"{stats['avg_win_pct']:+.2f}% / {stats['avg_loss_pct']:+.2f}%"),
            ("Expectancy per trade", f"{stats['expectancy_pct']:+.3f}%"),
            ("Profit factor", f"{stats['profit_factor']:.2f}"),
            ("Gross PnL", f"{stats['gross_pnl']:+.2f}"),
            ("Fees paid", f"-{stats['total_fees']:.2f}"),
            ("Net PnL", f"{stats['net_pnl']:+.2f}"),
            ("Max drawdown", f"{stats['max_drawdown_pct']:.2f}%"),
            ("Avg hold", f"{stats['avg_hold_minutes']:.1f} min"),
        ]
        for label, value in rows:
            print(f"  {label:<22} {value:>18}")
        print(f"  {'Exits':<22} {str(stats['exit_reasons']):>18}")

    print("-" * 62)
    print(
        f"  {'Equity':<22} "
        f"{stats['starting_equity']:.2f} -> {stats['final_equity']:.2f} "
        f"({stats['return_pct']:+.2f}%)"
    )
    if rejections:
        top = sorted(rejections.items(), key=lambda kv: kv[1], reverse=True)[:6]
        print("-" * 62)
        print("  Signals filtered: " + ", ".join(f"{k}={v}" for k, v in top))
    print("=" * 62 + "\n")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
