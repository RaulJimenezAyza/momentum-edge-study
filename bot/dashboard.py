"""Live web viewer.

Answers the question the console cannot: *why is nothing happening?* The
watchlist shows every symbol's distance from every gate, so an idle bot is
visibly idle-for-a-reason rather than possibly broken.

Served on localhost only by default. Read-only -- there is no endpoint that can
place, cancel or size a trade, deliberately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from .broker import PaperBroker
from .config import Config
from .risk import RiskManager
from .strategy import Strategy

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SPARKLINE_ROWS = 25
SPARKLINE_BARS = 40


class LogTail(logging.Handler):
    """Keeps the last N log records so the page can show them."""

    def __init__(self, capacity: int = 300):
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "time": datetime.fromtimestamp(
                        record.created, UTC
                    ).strftime("%H:%M:%S"),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
            )
        except Exception:  # noqa: BLE001 - logging must never break the bot
            pass


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%H:%M:%S")


class DashboardState:
    """Builds the JSON snapshot the page renders."""

    def __init__(
        self,
        cfg: Config,
        strategy: Strategy,
        broker: PaperBroker,
        risk: RiskManager,
        log_tail: LogTail,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.broker = broker
        self.risk = risk
        self.log_tail = log_tail
        self.started = time.time()
        # (time, account value, account value if trading were free)
        self.equity_curve: deque[tuple[int, float, float]] = deque(
            maxlen=cfg.dashboard.history_points
        )
        self.symbols: list[str] = []
        self.stream_health: dict = {}

    def record_equity(self, time_ms: int, equity: float) -> None:
        self.equity_curve.append(
            (time_ms, round(equity, 4), round(equity + self.broker.fees_paid, 4))
        )

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        s, b, cfg = self.strategy, self.broker, self.cfg
        prices = s.last_price
        equity = b.mark_to_market(s.positions, prices)
        now_ms = int(time.time() * 1000)

        positions = []
        for symbol, p in s.positions.items():
            price = prices.get(symbol, p.entry_price)
            span = p.target_price - p.stop_price
            positions.append(
                {
                    "symbol": symbol,
                    "qty": p.qty,
                    "entry": p.entry_price,
                    "price": price,
                    "notional": p.qty * price,
                    "unreal_pct": p.unrealized_pct(price),
                    "stop": p.stop_price,
                    "target": p.target_price,
                    "armed": p.trailing_armed,
                    "age_min": (now_ms - p.entry_time_ms) / 60_000.0,
                    "max_hold": cfg.risk.max_hold_minutes,
                    # Where price sits between stop and target, 0..1.
                    "progress": max(0.0, min(1.0, (price - p.stop_price) / span))
                    if span > 0
                    else 0.5,
                    "probability": p.meta.get("probability"),
                }
            )
        positions.sort(key=lambda r: r["unreal_pct"], reverse=True)

        watchlist = []
        for symbol in self.symbols:
            d = s.diagnostics.get(symbol)
            if d is None:
                continue
            watchlist.append(
                {
                    "symbol": symbol,
                    "price": prices.get(symbol, d.price),
                    "ret": d.ret_pct,
                    "z": d.zscore,
                    "volx": d.volume_mult,
                    "ext": d.extension_pct,
                    "gap": d.breakout_gap_pct,
                    "spread": d.spread_bps,
                    "heat": d.heat,
                    "blocking": d.blocking_gate,
                    "gates": d.gates,
                    "bars": d.bars,
                    "open": symbol in s.positions,
                }
            )
        watchlist.sort(key=lambda r: r["heat"], reverse=True)

        # Sparklines only for the rows a human is actually looking at -- 50
        # symbols x 40 closes every second is payload for nothing.
        for row in watchlist[:SPARKLINE_ROWS]:
            series = s.series.get(row["symbol"])
            if series is not None and len(series):
                row["spark"] = [
                    round(c.close, 10) for c in list(series.closed)[-SPARKLINE_BARS:]
                ]

        trades = [
            {
                "symbol": t.symbol,
                "entry_time": _ts(t.entry_time_ms),
                "exit_time": _ts(t.exit_time_ms),
                "net_pct": t.net_pct,
                "net_pnl": t.net_pnl,
                "fees": t.fees,
                "reason": t.exit_reason,
                "hold": t.hold_minutes,
                "equity_after": t.equity_after,
            }
            for t in b.trades[-40:]
        ][::-1]

        signals = [
            {
                "symbol": sig.symbol,
                "time": _ts(sig.time_ms),
                "price": sig.price,
                "ret": sig.ret_pct,
                "z": sig.zscore,
                "volx": sig.volume_mult,
                "probability": sig.probability,
            }
            for sig in s.signals_seen[-25:]
        ][::-1]

        wins = [t for t in b.trades if t.net_pnl > 0]
        return {
            "mode": cfg.mode,
            # In reversal mode every "move" figure is sign-flipped -- a positive
            # ret means the price FELL that much. The page has to say so, or the
            # watchlist reads as the exact opposite of what it means.
            "direction": cfg.strategy.direction,
            "timeframe": cfg.strategy.timeframe,
            "uptime_s": time.time() - self.started,
            "symbol_count": len(self.symbols),
            "equity": equity,
            "cash": b.equity,
            "start_equity": cfg.risk.starting_equity,
            "return_pct": (equity / cfg.risk.starting_equity - 1.0) * 100.0,
            "day_pnl_pct": self.risk.day_pnl_pct(equity),
            "daily_limit_pct": cfg.risk.daily_loss_limit_pct,
            "halted": self.risk.halted_until_next_day,
            "max_positions": cfg.risk.max_positions,
            "positions": positions,
            "watchlist": watchlist,
            "trades": trades,
            "signals": signals,
            "rejections": dict(
                sorted(s.rejections.items(), key=lambda kv: -kv[1])[:10]
            ),
            "equity_curve": list(self.equity_curve),
            "logs": list(self.log_tail.records)[-60:][::-1],
            "spread_band": [cfg.strategy.min_spread_bps, cfg.strategy.max_spread_bps],
            "exit_reasons": _counts(t.exit_reason for t in b.trades),
            "stats": {
                "gross_pnl": sum(t.gross_pnl for t in b.trades),
                "fees_paid": b.fees_paid,
                "trades": len(b.trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(b.trades) * 100.0 if b.trades else 0.0,
                "net_pnl": sum(t.net_pnl for t in b.trades),
                "signals": len(s.signals_seen),
            },
            "ml": {
                "enabled": s.model is not None,
                "threshold": s.ml_threshold,
                "relax": s.relax,
            },
            "stream": self.stream_health,
        }


class DashboardServer:
    def __init__(self, state: DashboardState):
        self.state = state
        self.cfg = state.cfg.dashboard
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/api/state", self._api_state)
        app.router.add_get("/api/stream", self._api_stream)
        app.router.add_get("/api/findings", self._api_findings)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.cfg.host, self.cfg.port)
        try:
            await site.start()
        except OSError as exc:
            log.warning(
                "dashboard could not bind %s:%d (%s) -- continuing without it",
                self.cfg.host, self.cfg.port, exc,
            )
            self._runner = None
            return
        log.info("dashboard: http://%s:%d", self.cfg.host, self.cfg.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    # -- handlers ----------------------------------------------------------
    async def _index(self, request: web.Request) -> web.StreamResponse:
        page = STATIC_DIR / "dashboard.html"
        if not page.exists():
            return web.Response(text="dashboard.html is missing", status=500)
        return web.Response(
            body=page.read_bytes(), content_type="text/html", charset="utf-8"
        )

    async def _api_state(self, request: web.Request) -> web.StreamResponse:
        return web.json_response(self.state.snapshot())

    async def _api_findings(self, request: web.Request) -> web.StreamResponse:
        """What the research commands last measured.

        Read from disk on every request rather than cached at startup, so a
        replication finishing in another terminal reaches the open page on its
        next poll. The page used to carry my prose about the results instead,
        and prose does not update itself: it still described the effect as
        real after the clustering analysis had shown it was not established.
        """
        from .findings import summary

        return web.json_response(summary(self.state.cfg))

    async def _api_stream(self, request: web.Request) -> web.StreamResponse:
        """Server-sent events. One snapshot per refresh interval."""
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        interval = self.cfg.refresh_ms / 1000.0
        try:
            while True:
                payload = json.dumps(self.state.snapshot(), default=str)
                await response.write(f"data: {payload}\n\n".encode())
                await asyncio.sleep(interval)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("SSE client dropped: %s", exc)
        return response
