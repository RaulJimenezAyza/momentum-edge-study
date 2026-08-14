"""What one round trip actually costs, per symbol.

Split out because the cost number decides every conclusion in this project. It
was previously a single hand-picked constant applied to all 50 symbols, which
meant a 0.2 bps pair like BNBUSDT and a 38 bps pair like RADUSDT were charged
the same. Both numbers were wrong in opposite directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config


@dataclass
class CostModel:
    cfg: Config
    # symbol -> estimated one-way slippage in bps. Empty means "use the flat
    # configured value", which is what live paper trading does before any
    # history has been measured.
    spreads_bps: dict[str, float] = field(default_factory=dict)
    # Exchange-quoted spreads, used only for symbols the estimator could not
    # measure. Ground truth beats a constant -- but it is ground truth about
    # TODAY, and a backtest window is not today. Honoured only when
    # fees.quote_fallback is "snapshot"; see the note on that setting.
    quoted_bps: dict[str, float] = field(default_factory=dict)

    @property
    def _use_quotes(self) -> bool:
        return self.cfg.fees.quote_fallback == "snapshot"

    @property
    def fee_rate(self) -> float:
        return self.cfg.fees.effective_taker_bps / 10_000.0

    def slippage_bps(self, symbol: str = "") -> float:
        if self.cfg.fees.slippage_mode != "estimated":
            return self.cfg.fees.slippage_bps
        spread = self.spreads_bps.get(symbol)
        if spread is None and self._use_quotes:
            spread = self.quoted_bps.get(symbol)
        if spread is None:
            return self.cfg.fees.slippage_bps
        # A market order crosses the book: it pays roughly half the quoted
        # spread, not the whole thing. The impact multiplier is on top of that
        # and covers walking past the touch on a thin book.
        return spread / 2.0 * self.cfg.fees.impact_multiplier

    def spread_bps(self, symbol: str = "") -> float:
        """The symbol's quoted spread, for the entry gate.

        Distinct from slippage_bps: slippage is half the spread times the
        impact multiplier, so multiplying it back by two does NOT recover the
        spread. The gate compares against a quoted spread, so it needs this.
        """
        known = self.spreads_bps.get(symbol)
        if known is not None:
            return known
        quoted = self.quoted_bps.get(symbol) if self._use_quotes else None
        if quoted is not None:
            return quoted
        # Neither measured nor quoted: fall back to the flat assumption.
        return self.cfg.fees.slippage_bps * 2.0

    def slippage_rate(self, symbol: str = "") -> float:
        return self.slippage_bps(symbol) / 10_000.0

    def round_trip_pct(self, symbol: str = "") -> float:
        return 2.0 * (self.cfg.fees.effective_taker_bps + self.slippage_bps(symbol)) / 100.0

    def unpriced(self, symbols) -> list[str]:
        """Symbols with no spread estimate, in estimated mode.

        These get the flat fallback, which on the default settings is 10 bps --
        wider than strategy.max_spread_bps, so the entry gate rejects them and
        they are never traded. That is the right call ("do not trade what you
        cannot price") but it emerged from two settings interacting rather than
        from a decision, and it silently removed 42% of signals the first time
        it happened. It is reported at startup so it stays a choice.
        """
        if self.cfg.fees.slippage_mode != "estimated":
            return []
        return [s for s in symbols
                if s not in self.spreads_bps
                and not (self._use_quotes and s in self.quoted_bps)]

    def summary(self) -> str:
        if self.cfg.fees.slippage_mode != "estimated" or not self.spreads_bps:
            return (f"flat cost model: {self.cfg.fees.effective_taker_bps:.2f} bps fee + "
                    f"{self.cfg.fees.slippage_bps:.2f} bps slippage per side "
                    f"= {self.cfg.round_trip_cost_pct:.3f}% round trip")
        costs = sorted(self.round_trip_pct(s) for s in self.spreads_bps)
        return (f"estimated cost model: round trip "
                f"{costs[0]:.3f}%-{costs[-1]:.3f}%, median {costs[len(costs) // 2]:.3f}%")
