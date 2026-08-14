"""The learned filter.

What it does: for each candidate breakout, estimate P(this trade ends net
positive after fees). The strategy then skips candidates below a probability
threshold.

What it does NOT do: predict price. It scores setups that the rule-based gates
already flagged. That is a much smaller, much more honest problem, and it is
the only version of "AI trading" that has ever reliably worked.

Design choices that exist to stop it lying to you:
  * Labels come from `simulate_exit`, the same exit rules the live bot uses, so
    a "win" in training is a win the bot could actually have taken.
  * Unresolved trades (not enough forward data) are dropped, not truncated.
  * The split is by TIME, never random. A random split lets the model see the
    future of the same market session and inflates every metric.
  * There is an embargo gap between train and test so a trade open at the
    boundary cannot leak its outcome across.
  * Probabilities are calibrated, because an uncalibrated 0.6 is not 60%.
  * The report compares filtered vs unfiltered expectancy on held-out data.
    If that number is not better, the model is worthless -- and it says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config import Config
from .costs import CostModel
from .exchange import Candle
from .features import FEATURE_NAMES, MarketContext, extract
from .levels import levels_for
from .series import Series
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import QUOTE_SNAPSHOT, estimate_spreads, load_quote_snapshot

log = logging.getLogger(__name__)

MINUTE_MS = 60_000


@dataclass
class Sample:
    symbol: str
    time_ms: int
    features: list[float]
    net_pct: float
    label: int
    reason: str


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def market_contexts(btc: list[Candle], lookback: int) -> dict[int, MarketContext]:
    """Precompute BTC regime features per minute, keyed by candle open time."""
    out: dict[int, MarketContext] = {}
    series = Series("BTCUSDT", lookback)
    for c in btc:
        series.update(c)
        out[c.open_time] = MarketContext.from_series(series)
    return out


def build_dataset(
    cfg: Config,
    history: dict[str, list[Candle]],
    relax: float,
) -> list[Sample]:
    """Replay history and emit one labelled row per candidate entry.

    The label must be the outcome the LIVE bot would have got, which means the
    same exit levels (ATR-scaled when configured) and the same per-symbol
    slippage. Labelling with the config's flat defaults while the bot trades
    something else trains the model on a strategy that does not exist.
    """
    lookback = cfg.strategy.lookback_bars
    contexts = market_contexts(history.get("BTCUSDT", []), lookback)
    horizon = cfg.max_hold_bars
    costs = CostModel(
        cfg, estimate_spreads(history) if cfg.fees.slippage_mode == "estimated" else {},
        load_quote_snapshot(cfg.path(QUOTE_SNAPSHOT))
        if cfg.fees.slippage_mode == "estimated" else {},
    )
    log.info("labelling with %s", costs.summary())
    unpriced = costs.unpriced(history)
    if unpriced:
        log.warning("%d/%d symbols unpriced -> flat fallback, gated out of trading",
                    len(unpriced), len(history))
    samples: list[Sample] = []

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)

        for i in range(lookback, len(candles) - horizon):
            series.update(candles[i])
            # Spread is unknown historically; the live gate applies it on top.
            result = evaluate(series, cfg.strategy, spread_bps=0.0, relax=relax)
            if isinstance(result, Rejection):
                continue

            future = candles[i + 1 : i + 1 + horizon]
            outcome = simulate_exit(
                result.price, future, cfg,
                levels_for(cfg, series), costs.slippage_bps(symbol),
            )
            if outcome is None:
                continue

            samples.append(
                Sample(
                    symbol=symbol,
                    time_ms=candles[i].open_time,
                    features=extract(series, contexts.get(candles[i].open_time)),
                    net_pct=outcome.net_pct,
                    label=1 if outcome.net_pct > 0 else 0,
                    reason=outcome.reason,
                )
            )

    samples.sort(key=lambda s: s.time_ms)
    return samples


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Model:
    """Gradient-boosted trees + isotonic calibration, wrapped for the bot."""

    def __init__(self, estimator: Any = None, meta: dict | None = None):
        self.estimator = estimator
        self.meta = meta or {}

    # -- inference ---------------------------------------------------------
    def probability(self, features: list[float]) -> float:
        if self.estimator is None:
            return 1.0
        import numpy as np

        x = np.asarray(features, dtype=float).reshape(1, -1)
        return float(self.estimator.predict_proba(x)[0, 1])

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "meta": self.meta}, path)
        path.with_suffix(".json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Model":
        import joblib

        blob = joblib.load(path)
        meta = blob.get("meta", {})
        names = meta.get("feature_names")
        if names and tuple(names) != FEATURE_NAMES:
            raise RuntimeError(
                "model was trained on a different feature set -- retrain it "
                f"({len(names)} features saved, {len(FEATURE_NAMES)} in code)"
            )
        return cls(blob["estimator"], meta)


def _expectancy(samples: list[Sample]) -> float:
    return sum(s.net_pct for s in samples) / len(samples) if samples else 0.0


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    brier: float
    threshold: float
    kept_fraction: float
    expectancy_all: float
    expectancy_filtered: float
    improvement: float
    verdict: str
    top_features: list[tuple[str, float]]
    # One row per candidate cut-off: (target keep, threshold, actual keep on
    # test, n kept, expectancy of the kept trades).
    sweep: list[tuple[float, float, float, int, float]]

    def as_dict(self) -> dict:
        return asdict(self)


# Keep fractions to derive cut-offs for. Chosen a priori, not by peeking.
KEEP_FRACTIONS = (0.50, 0.35, 0.25, 0.15, 0.08)


def train(
    samples: list[Sample],
    threshold: float | None = None,
    embargo_minutes: int = 60,
    test_fraction: float = 0.3,
    target_keep: float = 0.25,
) -> tuple[Model, TrainReport]:
    """Time-ordered fit with an embargo gap. Returns the model and the verdict."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score

    if len(samples) < 200:
        raise RuntimeError(
            f"only {len(samples)} candidates -- need at least 200 to train "
            "anything meaningful. Use more --days or a lower ml.relax_factor."
        )

    samples = sorted(samples, key=lambda s: s.time_ms)
    split_at = int(len(samples) * (1.0 - test_fraction))
    boundary = samples[split_at].time_ms

    # Embargo: drop training rows whose trade could still have been open when
    # the test period began.
    train_rows = [
        s for s in samples[:split_at]
        if s.time_ms < boundary - embargo_minutes * MINUTE_MS
    ]
    test_rows = samples[split_at:]
    if len(train_rows) < 100 or len(test_rows) < 50:
        raise RuntimeError("not enough rows on one side of the time split")

    x_train = np.array([s.features for s in train_rows], dtype=float)
    y_train = np.array([s.label for s in train_rows], dtype=int)
    x_test = np.array([s.features for s in test_rows], dtype=float)
    y_test = np.array([s.label for s in test_rows], dtype=int)

    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("training labels are all one class -- nothing to learn")

    base = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,          # shallow: this data is noisy and small
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=0,
    )
    # Isotonic on an internal CV split; the model never sees the test period.
    estimator = CalibratedClassifierCV(base, method="isotonic", cv=3)
    estimator.fit(x_train, y_train)

    probs = estimator.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs)) if len(set(y_test.tolist())) > 1 else 0.5
    brier = float(brier_score_loss(y_test, probs))

    # Cut-offs come from the TRAINING probability distribution. Picking them
    # from the test distribution -- or hand-picking 0.55 because it sounds like
    # "better than a coin flip" -- is how a filter ends up keeping 0% of trades
    # on a problem whose base rate is 33%.
    train_probs = estimator.predict_proba(x_train)[:, 1]
    exp_all = _expectancy(test_rows)

    sweep: list[tuple[float, float, float, int, float]] = []
    for keep in KEEP_FRACTIONS:
        cut = float(np.quantile(train_probs, 1.0 - keep))
        kept_rows = [s for s, p in zip(test_rows, probs) if p >= cut]
        sweep.append(
            (keep, cut, len(kept_rows) / len(test_rows), len(kept_rows),
             _expectancy(kept_rows))
        )

    if threshold is None:
        # Recommended cut-off is fixed a priori by target_keep, NOT chosen by
        # whichever row of the sweep happens to look best on the test set.
        nearest = min(KEEP_FRACTIONS, key=lambda k: abs(k - target_keep))
        threshold = next(row[1] for row in sweep if row[0] == nearest)

    kept = [s for s, p in zip(test_rows, probs) if p >= threshold]
    exp_kept = _expectancy(kept)
    improvement = exp_kept - exp_all

    if len(kept) < 20:
        verdict = "INCONCLUSIVE -- threshold keeps too few trades to judge"
    elif auc < 0.53:
        verdict = "USELESS -- AUC is indistinguishable from a coin flip"
    elif improvement <= 0:
        verdict = "HARMFUL -- filtering made expectancy worse out of sample"
    elif exp_kept <= 0:
        verdict = "NOT ENOUGH -- filter helps but expectancy is still negative"
    else:
        verdict = "PROMISING -- positive expectancy out of sample (one window)"

    importances = _permutation_importance(estimator, x_test, y_test)

    report = TrainReport(
        n_train=len(train_rows),
        n_test=len(test_rows),
        base_rate=float(y_train.mean()),
        auc=auc,
        brier=brier,
        threshold=threshold,
        kept_fraction=len(kept) / len(test_rows),
        expectancy_all=exp_all,
        expectancy_filtered=exp_kept,
        improvement=improvement,
        verdict=verdict,
        top_features=importances[:10],
        sweep=sweep,
    )
    meta = {
        "feature_names": list(FEATURE_NAMES),
        "report": report.as_dict(),
        "n_samples": len(samples),
        # The bot reads this in ml.threshold_mode: auto.
        "threshold": threshold,
    }
    return Model(estimator, meta), report


def _permutation_importance(estimator, x, y) -> list[tuple[str, float]]:
    """Which features actually matter on held-out data."""
    import numpy as np
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            estimator, x, y, n_repeats=5, random_state=0, scoring="roc_auc"
        )
    except Exception as exc:  # noqa: BLE001 - importance is nice-to-have
        log.warning("permutation importance failed: %s", exc)
        return []
    order = np.argsort(result.importances_mean)[::-1]
    return [(FEATURE_NAMES[i], float(result.importances_mean[i])) for i in order]


def print_report(report: TrainReport) -> None:
    print("\n" + "=" * 62)
    print("  MODEL REPORT (held-out, time-ordered)")
    print("=" * 62)
    rows = [
        ("Train / test rows", f"{report.n_train} / {report.n_test}"),
        ("Base win rate (train)", f"{report.base_rate * 100:.1f}%"),
        ("ROC AUC", f"{report.auc:.4f}"),
        ("Brier score", f"{report.brier:.4f}"),
        ("Threshold", f"{report.threshold:.2f}"),
        ("Trades kept", f"{report.kept_fraction * 100:.1f}%"),
        ("Expectancy, no filter", f"{report.expectancy_all:+.4f}%"),
        ("Expectancy, filtered", f"{report.expectancy_filtered:+.4f}%"),
        ("Improvement", f"{report.improvement:+.4f}%"),
    ]
    for label, value in rows:
        print(f"  {label:<24} {value:>16}")

    if report.sweep:
        print("-" * 62)
        print("  Cut-offs derived from the TRAINING distribution, scored on test:")
        print(f"    {'target':>7} {'cut-off':>9} {'kept':>7} {'n':>6} {'expectancy':>12}")
        for keep, cut, actual, n, exp in report.sweep:
            mark = " <-- used" if abs(cut - report.threshold) < 1e-12 else ""
            print(f"    {keep * 100:>6.0f}% {cut:>9.3f} {actual * 100:>6.1f}% "
                  f"{n:>6d} {exp:>+11.4f}%{mark}")

    print("-" * 62)
    print(f"  VERDICT: {report.verdict}")
    if report.top_features:
        print("-" * 62)
        print("  Most informative features (permutation, AUC drop):")
        for name, score in report.top_features[:8]:
            print(f"    {name:<22} {score:+.5f}")
    print("=" * 62)
    print(
        "  AUC 0.50 = coin flip. Above ~0.55 on financial data is already\n"
        "  unusual; above 0.65 on one window usually means a leak, not skill.\n"
        "  One held-out window is evidence, not proof. Retrain and re-check\n"
        "  on a different period before trusting it with money."
    )
    print("=" * 62 + "\n")
