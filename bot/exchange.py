"""Binance REST + websocket access. Public endpoints only for paper mode."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable
from urllib.parse import urlencode

import aiohttp
import websockets

log = logging.getLogger(__name__)

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    # A daily bar IS a session, which changes what the session logic must do --
    # see sessions.is_intraday.
    "1d": 86_400_000,
}


@dataclass(slots=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    closed: bool

    @classmethod
    def from_rest(cls, row: list) -> "Candle":
        return cls(
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            closed=True,
        )

    @classmethod
    def from_ws(cls, k: dict) -> "Candle":
        return cls(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            quote_volume=float(k["q"]),
            closed=bool(k["x"]),
        )


@dataclass(slots=True)
class SymbolFilters:
    """Exchange constraints that a real order must respect."""

    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        return round(int(price / self.tick_size) * self.tick_size, 10)

    def round_qty(self, qty: float) -> float:
        if self.step_size <= 0:
            return qty
        return round(int(qty / self.step_size) * self.step_size, 10)


class BinanceREST:
    """Thin async REST wrapper with retry on transient failures."""

    def __init__(self, base_url: str, api_key: str = "", api_secret: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "BinanceREST":
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        self._session = aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=20)
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._session:
            await self._session.close()

    async def _get(
        self, path: str, params: dict | None = None, signed: bool = False
    ) -> Any:
        assert self._session, "use BinanceREST as an async context manager"
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query = urlencode(params)
            params["signature"] = hmac.new(
                self.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()

        url = f"{self.base_url}{path}"
        delay = 1.0
        for attempt in range(4):
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 451:
                        raise RuntimeError(
                            f"{url} returned 451 (geo-restricted). Try "
                            "data-api.binance.vision or a regional mirror in config.yaml."
                        )
                    if resp.status == 429 or resp.status >= 500:
                        body = await resp.text()
                        log.warning("HTTP %s from %s: %s", resp.status, path, body[:200])
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientError as exc:
                if attempt == 3:
                    raise
                log.warning("request %s failed (%s), retrying", path, exc)
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"giving up on {path} after retries")

    async def ping(self) -> None:
        await self._get("/api/v3/ping")

    async def exchange_info(self) -> dict:
        return await self._get("/api/v3/exchangeInfo")

    async def ticker_24h(self) -> list[dict]:
        return await self._get("/api/v3/ticker/24hr")

    async def book_tickers(self) -> list[dict]:
        return await self._get("/api/v3/ticker/bookTicker")

    async def klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = await self._get("/api/v3/klines", params)
        return [Candle.from_rest(r) for r in rows]

    async def klines_range(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        """Page through /klines until the range is covered."""
        step = TIMEFRAME_MS.get(interval, 60_000)
        out: list[Candle] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = await self.klines(
                symbol, interval, limit=1000, start_ms=cursor, end_ms=end_ms
            )
            if not batch:
                break
            out.extend(batch)
            cursor = batch[-1].open_time + step
            if len(batch) < 1000:
                break
            await asyncio.sleep(0.12)  # stay well under the weight limit
        return out

    async def symbol_filters(self, symbols: Iterable[str]) -> dict[str, SymbolFilters]:
        info = await self.exchange_info()
        wanted = set(symbols)
        out: dict[str, SymbolFilters] = {}
        for s in info.get("symbols", []):
            if s["symbol"] not in wanted:
                continue
            tick = step = min_qty = min_notional = 0.0
            for f in s.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                    min_notional = float(f.get("minNotional", 0.0))
            out[s["symbol"]] = SymbolFilters(
                s["symbol"], tick, step, min_qty, min_notional
            )
        return out


async def stream_klines(
    ws_url: str, symbols: list[str], interval: str = "1m"
) -> AsyncIterator[tuple[str, Candle]]:
    """Yield (symbol, candle) from a combined kline stream, reconnecting forever.

    Binance caps a single connection at 1024 streams; 50 symbols is well inside
    that, but the URL is chunked anyway so growing the universe stays safe.
    """
    streams = "/".join(f"{s.lower()}@kline_{interval}" for s in symbols)
    url = f"{ws_url}?streams={streams}"
    backoff = 1.0

    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_queue=2048
            ) as ws:
                log.info("websocket connected: %d symbols", len(symbols))
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    data = msg.get("data")
                    if not data or data.get("e") != "kline":
                        continue
                    yield data["s"], Candle.from_ws(data["k"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconnect on anything
            log.warning("websocket dropped (%s); reconnecting in %.0fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
