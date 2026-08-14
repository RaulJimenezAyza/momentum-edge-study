"""Pick which symbols to watch."""

from __future__ import annotations

import logging

from .config import ExchangeConfig
from .exchange import BinanceREST

log = logging.getLogger(__name__)


def _is_excluded(symbol: str, base: str, cfg: ExchangeConfig) -> bool:
    if symbol in cfg.exclude_symbols:
        return True
    # Leveraged tokens: BTCUPUSDT, ETHDOWNUSDT, ...
    return any(base.endswith(p) for p in cfg.exclude_patterns)


async def select_universe(rest: BinanceREST, cfg: ExchangeConfig) -> list[str]:
    """Top N spot symbols by 24h quote volume, filtered for tradability."""
    info = await rest.exchange_info()
    tradable = {
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == cfg.quote_asset
        and s.get("isSpotTradingAllowed", True)
    }

    tickers = await rest.ticker_24h()
    rows = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in tradable:
            continue
        base = symbol[: -len(cfg.quote_asset)]
        if _is_excluded(symbol, base, cfg):
            continue
        quote_volume = float(t.get("quoteVolume", 0.0))
        if quote_volume < cfg.min_quote_volume_24h:
            continue
        rows.append((symbol, quote_volume))

    rows.sort(key=lambda r: r[1], reverse=True)
    selected = [s for s, _ in rows[: cfg.universe_size]]
    log.info(
        "universe: %d symbols (of %d eligible), volume floor %.0f %s",
        len(selected),
        len(rows),
        cfg.min_quote_volume_24h,
        cfg.quote_asset,
    )
    return selected
