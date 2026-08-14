"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so we don't need python-dotenv."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class ExchangeConfig:
    base_url: str = "https://api.binance.com"
    ws_url: str = "wss://stream.binance.com:9443/stream"
    testnet_base_url: str = "https://testnet.binance.vision"
    testnet_ws_url: str = "wss://stream.testnet.binance.vision:9443/stream"
    quote_asset: str = "USDT"
    universe_size: int = 50
    min_quote_volume_24h: float = 20_000_000.0
    exclude_patterns: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)


@dataclass
class StrategyConfig:
    timeframe: str = "1m"
    # "momentum" buys breakouts above the recent high.
    # "reversal" buys capitulation below the recent low, betting on a bounce.
    # Same gates, mirrored -- the natural alternative hypothesis to test when
    # momentum turns out to have no gross edge.
    direction: str = "momentum"
    lookback_bars: int = 120
    min_bars_ready: int = 60
    momentum_bars: int = 3
    min_return_pct: float = 0.6
    min_return_zscore: float = 2.5
    # "base"  -- units of the coin traded, which is what the exchange reports
    #            first and what this project used throughout.
    # "quote" -- money traded. Base volume is money over price, so on a fall
    #            the same dollar flow prints as more units; measured over a
    #            120-bar window that inflates the multiple exactly on the bars
    #            a dip-buying strategy selects. Quote volume has no such tilt.
    volume_measure: str = "base"
    min_volume_mult: float = 3.0
    breakout_bars: int = 30
    max_extension_pct: float = 6.0
    # UTC hour ranges the strategy may trade in, as [start, end) pairs.
    # Empty = all hours. Measured on 45 days: gross +0.076% in 00-06 UTC and
    # +0.061% in 18-24, against -0.074% in 06-12. Plausible mechanism -- thinner
    # books overnight overshoot further -- but the windows were read off a
    # four-bucket table, so it was a hypothesis until the holdout ruled.
    #
    # It ruled against. Restricted to 00-06 + 18-24 the holdout scored -0.1218%
    # per trade against -0.0832% unrestricted, with the edge over random at
    # p = 0.34 and a CI straddling zero. Picking the two best of four buckets
    # found noise. Ships EMPTY; set it to re-test on a future window.
    trading_hours_utc: list = field(default_factory=list)
    # Spread BAND, not just a ceiling. Measured, not assumed: edge over random
    # RISES as liquidity falls (overreaction is bigger where less capital stands
    # ready to arbitrage it) but the spread rises faster, so there is an interior
    # optimum. The sub-2 bps tier scores -0.004% gross against 2-5 bps at +0.032%.
    #
    # The shape (two opposing forces -> interior optimum) was predicted; the
    # boundaries were read off the data. It did not survive the holdout, so it
    # ships off -- see config.yaml.
    min_spread_bps: float = 0.0
    max_spread_bps: float = 8.0


@dataclass
class RiskConfig:
    starting_equity: float = 1000.0
    risk_per_trade_pct: float = 1.0
    max_positions: int = 5
    max_position_pct: float = 20.0
    min_notional: float = 12.0
    # "fixed" -- the percentages below, identical for every symbol.
    # "atr"   -- distances scaled by each symbol's own ATR. Entries are already
    #            volatility-normalised via the z-score gate; leaving exits on a
    #            flat percentage means the same stop is loose on a microcap and
    #            hair-trigger on BTC, which is an inconsistency, not a choice.
    # When one candle touches BOTH the stop and the target, OHLC cannot say
    # which came first. The choice is not a detail: on this data it is worth
    # about as much as the entire measured edge, so it must be bracketed rather
    # than picked.
    #   "pessimistic" -- always the stop. A lower bound on the truth.
    #   "optimistic"  -- always the target. An upper bound.
    #   "open"        -- whichever barrier the bar opened nearer to. A guess
    #                    that lands between the bounds; still a guess.
    # Report the bracket. A conclusion that only holds at one end is not a
    # conclusion, it is an artefact of the assumption.
    intrabar_fill: str = "pessimistic"
    stop_mode: str = "fixed"
    atr_bars: int = 30
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 3.0
    trail_atr_mult: float = 1.0
    # Bounds so a volatility spike cannot produce an absurd stop distance.
    min_stop_pct: float = 0.4
    max_stop_pct: float = 6.0
    stop_loss_pct: float = 1.2
    take_profit_pct: float = 2.4
    trail_activate_pct: float = 1.2
    trailing_stop_pct: float = 0.8
    max_hold_minutes: int = 30
    daily_loss_limit_pct: float = 5.0
    cooldown_minutes: int = 15


@dataclass
class FeesConfig:
    taker_bps: float = 10.0
    # "fixed"     -- slippage_bps for every symbol, whatever its book looks like.
    # "estimated" -- per-symbol, from the Abdi-Ranaldo spread estimator run on
    #                that symbol's own candles. A flat guess applied to 50
    #                symbols whose real spreads span 0.2 to 38 bps is wrong for
    #                all of them; when the assumption is larger than the effect
    #                being measured, it decides the answer.
    # A take-profit is a resting limit sell by construction: you name the
    # price and wait. Charging it the taker fee AND half the spread, as every
    # exit was charged until this was noticed, is simply wrong -- and wrong in
    # the direction that made the strategy look worse than it is. Stops and
    # time exits really are market orders and still pay both.
    target_exit_is_maker: bool = True
    maker_bps: float | None = None      # None = same as taker (Binance VIP0)
    slippage_mode: str = "fixed"
    slippage_bps: float = 5.0
    # Multiplier on the half-spread, covering walking past the touch. 1.0 means
    # "pay exactly half the quoted spread", which is the best case for a taker.
    impact_multiplier: float = 1.5
    # What to charge a symbol the spread estimator could not resolve.
    #
    # "snapshot" -- today's quoted spread from bookTicker. Closer to the truth
    #               for a symbol whose liquidity has not changed, but it is
    #               TODAY's number applied to a window months old: look-ahead,
    #               of exactly the kind pointintime.py exists to remove from
    #               the universe. It was introduced here without being flagged.
    # "flat"     -- the configured slippage_bps, which is knowable in advance
    #               and pessimistic. On the default settings it also exceeds
    #               max_spread_bps, so those symbols are gated out entirely
    #               rather than traded at a guessed cost.
    #
    # They are not a small difference: under "flat" roughly 40% of signals
    # disappear, because the unmeasurable symbols stop being tradable at all.
    quote_fallback: str = "snapshot"
    # Paying fees in BNB takes 25% off on Binance spot. Free money relative to
    # anything the strategy can do, so it is worth modelling explicitly.
    bnb_discount: bool = False

    @property
    def effective_taker_bps(self) -> float:
        return self.taker_bps * (0.75 if self.bnb_discount else 1.0)

    @property
    def effective_maker_bps(self) -> float:
        base = self.taker_bps if self.maker_bps is None else self.maker_bps
        return base * (0.75 if self.bnb_discount else 1.0)


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8787
    refresh_ms: int = 1000
    history_points: int = 600


@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str = "data/model.joblib"
    # "auto"  -- use the threshold the trainer derived from the training
    #            distribution, stored inside the model file. Correct default:
    #            a calibrated model on a 33%-base-rate problem rarely emits
    #            probabilities above 0.5, so any hand-picked cut-off is
    #            arbitrary and usually keeps nothing.
    # "fixed" -- use min_probability below.
    threshold_mode: str = "auto"
    min_probability: float = 0.55
    # auto mode: keep roughly this share of candidates. The trainer converts it
    # to a probability cut-off using the TRAINING distribution only, so the
    # held-out numbers stay honest.
    target_keep_fraction: float = 0.25
    # Loosen the hand-tuned gates when the model is deciding, so it has a wider
    # pool to choose from than the rules alone would allow.
    relax_factor: float = 0.6
    # Minutes of separation between the train and test periods.
    embargo_minutes: int = 60
    test_fraction: float = 0.3


@dataclass
class LoggingConfig:
    level: str = "INFO"
    trades_csv: str = "data/trades.csv"
    equity_csv: str = "data/equity.csv"
    signals_csv: str = "data/signals.csv"


@dataclass
class Config:
    mode: str = "paper"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    fees: FeesConfig = field(default_factory=FeesConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Closed candles never change, so caching them to disk is always safe.
    # Turn off only to prove a result against a fresh download.
    cache_enabled: bool = True
    root: Path = field(default_factory=Path.cwd)

    # -- derived -----------------------------------------------------------
    @property
    def is_testnet(self) -> bool:
        return self.mode == "testnet"

    @property
    def rest_url(self) -> str:
        return (
            self.exchange.testnet_base_url
            if self.is_testnet
            else self.exchange.base_url
        )

    @property
    def stream_url(self) -> str:
        return (
            self.exchange.testnet_ws_url if self.is_testnet else self.exchange.ws_url
        )

    @property
    def bar_ms(self) -> int:
        """Milliseconds per candle. Nothing may assume this is 60,000."""
        from .exchange import TIMEFRAME_MS

        try:
            return TIMEFRAME_MS[self.strategy.timeframe]
        except KeyError:
            raise ValueError(
                f"unsupported timeframe {self.strategy.timeframe!r}; "
                f"known: {', '.join(TIMEFRAME_MS)}"
            ) from None

    @property
    def max_hold_bars(self) -> int:
        """The time stop, expressed in candles of the configured timeframe."""
        return max(1, round(self.risk.max_hold_minutes * 60_000 / self.bar_ms))

    @property
    def round_trip_cost_pct(self) -> float:
        """Fees + slippage paid across a full entry/exit cycle, in percent."""
        return 2.0 * (self.fees.effective_taker_bps + self.fees.slippage_bps) / 100.0

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.root / p

    def validate(self) -> None:
        if self.mode not in ("paper", "testnet", "live"):
            raise ValueError(f"mode must be paper|testnet|live, got {self.mode!r}")

        s, r = self.strategy, self.risk
        for window in s.trading_hours_utc:
            if len(window) != 2 or not (0 <= window[0] < window[1] <= 24):
                raise ValueError(
                    f"strategy.trading_hours_utc entries must be [start, end) "
                    f"with 0 <= start < end <= 24, got {window!r}"
                )
        if s.volume_measure not in ("base", "quote"):
            raise ValueError("strategy.volume_measure must be base or quote")
        if s.direction not in ("momentum", "reversal"):
            raise ValueError("strategy.direction must be momentum or reversal")
        if r.intrabar_fill not in ("pessimistic", "optimistic", "open"):
            raise ValueError(
                "risk.intrabar_fill must be pessimistic, optimistic or open"
            )
        if r.stop_mode not in ("fixed", "atr"):
            raise ValueError("risk.stop_mode must be fixed or atr")
        if r.stop_mode == "atr" and r.target_atr_mult <= r.stop_atr_mult:
            raise ValueError("risk.target_atr_mult must exceed stop_atr_mult")
        self.bar_ms  # raises on an unsupported timeframe
        if s.momentum_bars < 1:
            raise ValueError("strategy.momentum_bars must be >= 1")
        if s.min_bars_ready > s.lookback_bars:
            raise ValueError("strategy.min_bars_ready cannot exceed lookback_bars")
        if s.breakout_bars >= s.lookback_bars:
            raise ValueError("strategy.breakout_bars must be < lookback_bars")
        if r.stop_loss_pct <= 0:
            raise ValueError("risk.stop_loss_pct must be > 0")
        if r.max_positions < 1:
            raise ValueError("risk.max_positions must be >= 1")

        # A target that cannot clear costs is a guaranteed loser. In ATR mode
        # the distance is per-symbol, so min_stop_pct carries the same duty.
        if r.stop_mode == "atr":
            floor_target = r.min_stop_pct * r.target_atr_mult / r.stop_atr_mult
            if floor_target <= self.round_trip_cost_pct:
                raise ValueError(
                    f"the tightest possible ATR target ({floor_target:.2f}%) does not "
                    f"cover round-trip cost ({self.round_trip_cost_pct:.2f}%) -- "
                    "raise risk.min_stop_pct or risk.target_atr_mult"
                )
        elif r.take_profit_pct <= self.round_trip_cost_pct:
            raise ValueError(
                f"risk.take_profit_pct ({r.take_profit_pct}%) does not cover round-trip "
                f"cost ({self.round_trip_cost_pct:.2f}%) -- every winner would still lose money"
            )

        if not 0.0 < self.ml.relax_factor <= 1.0:
            raise ValueError("ml.relax_factor must be in (0, 1]")
        if not 0.0 <= self.ml.min_probability <= 1.0:
            raise ValueError("ml.min_probability must be in [0, 1]")
        if self.fees.quote_fallback not in ("snapshot", "flat"):
            raise ValueError("fees.quote_fallback must be snapshot or flat")
        if self.fees.slippage_mode not in ("fixed", "estimated"):
            raise ValueError("fees.slippage_mode must be fixed or estimated")
        if self.fees.impact_multiplier <= 0:
            raise ValueError("fees.impact_multiplier must be > 0")
        if self.ml.threshold_mode not in ("auto", "fixed"):
            raise ValueError("ml.threshold_mode must be auto or fixed")
        if not 0.0 < self.ml.target_keep_fraction < 1.0:
            raise ValueError("ml.target_keep_fraction must be in (0, 1)")

        if self.mode == "live":
            if os.getenv("TRADING_LIVE_CONFIRM") != "YES":
                raise RuntimeError(
                    "mode: live requires TRADING_LIVE_CONFIRM=YES in the environment"
                )
            if not (os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET")):
                raise RuntimeError("mode: live requires BINANCE_API_KEY / _SECRET")
        if self.mode == "testnet":
            if not (
                os.getenv("BINANCE_TESTNET_API_KEY")
                and os.getenv("BINANCE_TESTNET_API_SECRET")
            ):
                raise RuntimeError(
                    "mode: testnet requires BINANCE_TESTNET_API_KEY / _SECRET"
                )


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    root = path.parent
    _load_dotenv(root / ".env")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(
        mode=raw.get("mode", "paper"),
        exchange=_build(ExchangeConfig, raw.get("exchange", {})),
        strategy=_build(StrategyConfig, raw.get("strategy", {})),
        risk=_build(RiskConfig, raw.get("risk", {})),
        fees=_build(FeesConfig, raw.get("fees", {})),
        dashboard=_build(DashboardConfig, raw.get("dashboard", {})),
        ml=_build(MLConfig, raw.get("ml", {})),
        logging=_build(LoggingConfig, raw.get("logging", {})),
        cache_enabled=raw.get("cache_enabled", True),
        root=root,
    )
    cfg.validate()
    return cfg
