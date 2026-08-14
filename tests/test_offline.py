"""Offline sanity checks -- no network. Run: python tests/test_offline.py"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

from bot.broker import PaperBroker
from bot.config import load_config
from bot.exchange import Candle
from bot.features import FEATURE_NAMES, MarketContext, extract
from bot.levels import fixed_levels, levels_for
from bot.risk import RiskManager
from bot.series import Series
from bot.signals import Rejection, diagnose, evaluate
from bot.simulate import simulate_exit
from bot.strategy import Strategy

MINUTE = 60_000
FAILURES: list[str] = []
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def base_cfg():
    """Config for unit tests, pinned independent of what config.yaml ships.

    The shipped defaults change as measurements decide things; unit tests must
    not. Direction is pinned to momentum because the fixtures build up-spikes,
    and slippage to fixed so cost arithmetic in the assertions is exact.
    Tests that care about a setting override it themselves.
    """
    cfg = load_config(CONFIG_PATH)
    cfg.strategy.direction = "momentum"
    cfg.fees.slippage_mode = "fixed"
    cfg.risk.intrabar_fill = "pessimistic"
    return cfg


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def candle(t: int, close: float, volume: float = 100.0, high=None, low=None) -> Candle:
    high = high if high is not None else close * 1.0005
    low = low if low is not None else close * 0.9995
    return Candle(t, close, high, low, close, volume, volume * close, True)


def flat_then_spike(bars: int = 100) -> list[Candle]:
    """Quiet drift, then a 3-bar volume-backed breakout."""
    out = []
    price = 100.0
    for i in range(bars):
        price *= 1.0 + (0.0004 if i % 2 else -0.0004)  # tiny oscillation
        out.append(candle(i * MINUTE, price, volume=100.0))
    for i, bump in enumerate((1.004, 1.004, 1.004)):
        price *= bump
        out.append(candle((bars + i) * MINUTE, price, volume=800.0))
    return out


def test_series() -> None:
    print("\nseries statistics")
    s = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        s.update(c)
    check("window respects maxlen", len(s) == 103, f"len={len(s)}")
    check("3-bar return is the spike", abs(s.ret_pct(3) - 1.2048) < 0.05, f"{s.ret_pct(3):.4f}%")
    check("sigma is positive", s.return_sigma_pct(3) > 0, f"{s.return_sigma_pct(3):.4f}")
    check("volume multiple detected", s.volume_mult() > 5.0, f"x{s.volume_mult():.2f}")
    check("breakout level excludes last bar",
          s.highest_close(30, offset=1) < s.closed[-1].close)

    # A forming candle must not enter the closed window.
    forming = Candle(999 * MINUTE, 1, 1, 1, 1, 1, 1, closed=False)
    before = len(s)
    s.update(forming)
    check("forming candle kept out of history", len(s) == before)
    check("last_price uses the forming candle", s.last_price == 1.0)


def test_signal_gates() -> None:
    print("\nsignal gates")
    cfg = base_cfg()

    s = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        s.update(c)
    result = evaluate(s, cfg.strategy, spread_bps=2.0)
    check("clean breakout fires", not isinstance(result, Rejection),
          getattr(result, "reason", f"z={getattr(result, 'zscore', 0):.1f}"))

    check("wide spread rejected",
          getattr(evaluate(s, cfg.strategy, spread_bps=50.0), "reason", "") == "spread_too_wide")

    quiet = Series("TESTUSDT", 120)
    price = 100.0
    for i in range(100):
        price *= 1.0002
        quiet.update(candle(i * MINUTE, price))
    check("slow grind rejected",
          isinstance(evaluate(quiet, cfg.strategy, 1.0), Rejection),
          getattr(evaluate(quiet, cfg.strategy, 1.0), "reason", ""))

    cold = Series("TESTUSDT", 120)
    for c in flat_then_spike()[:20]:
        cold.update(c)
    check("warm-up enforced",
          getattr(evaluate(cold, cfg.strategy, 1.0), "reason", "") == "warming_up")

    # Same-shaped move with no volume behind it must not trade.
    novol = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        novol.update(Candle(c.open_time, c.open, c.high, c.low, c.close, 100.0,
                            100.0 * c.close, True))
    check("move without volume rejected",
          getattr(evaluate(novol, cfg.strategy, 1.0), "reason", "") == "no_volume_confirmation")


def test_maker_exit_pricing() -> None:
    """A take-profit is a resting limit; a stop is a market order.

    Every exit used to be charged the taker fee plus half the spread. A target
    exit crosses no spread and pays the maker fee -- the model was overcharging
    the one exit type the strategy is actually aiming for, and overcharging in
    the direction that made the results look worse.
    """
    print(chr(10) + "target exits are maker fills")
    cfg = base_cfg()
    entry = 100.0
    lv = fixed_levels(cfg)

    tp_bar = [Candle(MINUTE, entry, entry * 1.05, entry * 0.999, entry, 1, 1, True)]
    stop_bar = [Candle(MINUTE, entry, entry * 1.001, entry * 0.90, entry, 1, 1, True)]
    tp = simulate_exit(entry, tp_bar, cfg, lv, 20.0)
    st = simulate_exit(entry, stop_bar, cfg, lv, 20.0)
    check("target exit is a take_profit", tp.reason == "take_profit", tp.reason)
    check("stop exit is a stop_loss", st.reason == "stop_loss", st.reason)

    # The target keeps its full percentage; the stop loses the spread on exit.
    check("target exit keeps the whole target move",
          abs(tp.gross_pct - cfg.risk.take_profit_pct) < 1e-9,
          f"{tp.gross_pct:.4f}% vs {cfg.risk.take_profit_pct}%")
    check("stop exit is worse than its nominal distance",
          st.gross_pct < -cfg.risk.stop_loss_pct,
          f"{st.gross_pct:.4f}% vs -{cfg.risk.stop_loss_pct}%")

    # Turning the correction off must restore the old, harsher pricing.
    harsh = base_cfg()
    harsh.fees.target_exit_is_maker = False
    old = simulate_exit(entry, tp_bar, harsh, lv, 20.0)
    check("the old market-exit pricing is still reachable",
          old.net_pct < tp.net_pct, f"{old.net_pct:+.4f}% vs {tp.net_pct:+.4f}%")
    check("and the difference is the exit spread",
          abs((tp.net_pct - old.net_pct) - 20.0 / 100.0) < 0.01,
          f"{tp.net_pct - old.net_pct:.4f}%")

    # A cheaper maker tier must show up in the net.
    tiered = base_cfg()
    tiered.fees.maker_bps = 2.0
    cheap = simulate_exit(entry, tp_bar, tiered, lv, 20.0)
    check("a lower maker fee improves a target exit",
          cheap.net_pct > tp.net_pct, f"{cheap.net_pct:+.4f}% vs {tp.net_pct:+.4f}%")
    check("but does nothing for a stop exit",
          abs(simulate_exit(entry, stop_bar, tiered, lv, 20.0).net_pct
              - st.net_pct) < 1e-9)


def test_broker_costs() -> None:
    print("\nbroker fees and slippage")
    cfg = base_cfg()
    broker = PaperBroker(cfg)
    start = broker.equity

    pos = broker.buy("TESTUSDT", 200.0, 100.0, 0, fixed_levels(cfg))
    assert pos is not None
    check("entry fills above signal price", pos.entry_price > 100.0,
          f"{pos.entry_price:.4f}")
    check("stop placed below entry", pos.stop_price < pos.entry_price)
    check("cash reduced by cost + fee", broker.equity < start - 199.0,
          f"{broker.equity:.2f}")

    # Round trip at an unchanged price must lose exactly the round-trip cost.
    trade = broker.sell(pos, 100.0, 60_000, "test")
    check("flat round trip loses ~cost",
          abs(trade.net_pct + cfg.round_trip_cost_pct) < 0.02,
          f"{trade.net_pct:.4f}% vs cost {cfg.round_trip_cost_pct:.4f}%")
    check("equity matches net PnL",
          abs(broker.equity - (start + trade.net_pnl)) < 1e-6,
          f"{broker.equity:.6f}")


def test_risk() -> None:
    print("\nrisk manager")
    cfg = base_cfg()
    risk = RiskManager(cfg)

    size = risk.position_size(1000.0)
    # 1% risk over a 1.2% stop = 833, but the 20% cap binds first.
    check("position cap binds", abs(size - 200.0) < 1e-6, f"{size:.2f}")

    ok, _ = risk.can_open("BTCUSDT", MINUTE, 1000.0, 0)
    check("first entry allowed", ok)
    ok, why = risk.can_open("BTCUSDT", MINUTE, 1000.0, cfg.risk.max_positions)
    check("max_positions blocks", not ok and why == "max_positions", why)

    risk.start_cooldown("BTCUSDT", MINUTE)
    ok, why = risk.can_open("BTCUSDT", MINUTE + 60_000, 1000.0, 0)
    check("cooldown blocks re-entry", not ok and why == "cooldown", why)

    ok, why = risk.can_open("ETHUSDT", MINUTE, 930.0, 0)
    check("daily loss limit halts trading", not ok and why == "daily_loss_limit", why)
    check("halt latches", risk.halted_until_next_day)


def test_strategy_round_trip() -> None:
    print("\nstrategy end to end")
    cfg = base_cfg()
    broker = PaperBroker(cfg)
    strategy = Strategy(cfg, broker, RiskManager(cfg))

    history = flat_then_spike()
    for c in history:
        strategy.on_candle("TESTUSDT", c)
    check("position opened on the breakout", "TESTUSDT" in strategy.positions,
          str(strategy.rejections))

    pos = strategy.positions.get("TESTUSDT")
    if pos:
        t = history[-1].open_time + MINUTE
        # A candle that trades clean through the take-profit level.
        up = pos.target_price * 1.01
        strategy.on_candle("TESTUSDT", candle(t, up, high=up, low=pos.entry_price))
        check("target closed the position", "TESTUSDT" not in strategy.positions)
        check("one trade recorded", len(broker.trades) == 1)
        if broker.trades:
            tr = broker.trades[0]
            check("exit reason is take_profit", tr.exit_reason == "take_profit",
                  tr.exit_reason)
            check("winner nets positive", tr.net_pnl > 0, f"{tr.net_pct:+.2f}%")
            check("net is below gross by fees",
                  abs((tr.gross_pnl - tr.fees) - tr.net_pnl) < 1e-9)

    # Cooldown must prevent an instant re-entry on the same symbol.
    before = len(strategy.positions)
    for c in flat_then_spike():
        shifted = Candle(c.open_time + 200 * MINUTE, c.open, c.high, c.low,
                         c.close, c.volume, c.quote_volume, True)
        strategy.on_candle("TESTUSDT", shifted)
    check("cooldown or fresh entry handled sanely",
          len(strategy.positions) >= before)


def test_stop_priority() -> None:
    print("\nintrabar stop priority")
    cfg = base_cfg()
    broker = PaperBroker(cfg)
    strategy = Strategy(cfg, broker, RiskManager(cfg))
    for c in flat_then_spike():
        strategy.on_candle("TESTUSDT", c)
    pos = strategy.positions.get("TESTUSDT")
    if pos is None:
        check("setup produced a position", False)
        return
    t = 200 * MINUTE
    # Candle spans both the stop and the target: the stop must win.
    strategy.on_candle("TESTUSDT", candle(t, pos.entry_price,
                                          high=pos.target_price * 1.01,
                                          low=pos.stop_price * 0.99))
    check("stop takes priority over target",
          broker.trades and broker.trades[-1].exit_reason == "stop_loss",
          broker.trades[-1].exit_reason if broker.trades else "no trade")


def test_config_guard() -> None:
    print("\nconfig validation")
    cfg = base_cfg()
    check("round-trip cost computed", abs(cfg.round_trip_cost_pct - 0.30) < 1e-9,
          f"{cfg.round_trip_cost_pct}%")
    cfg.risk.take_profit_pct = 0.1
    try:
        cfg.validate()
        check("unprofitable target rejected", False)
    except ValueError as exc:
        check("unprofitable target rejected", "round-trip" in str(exc))


def test_diagnostics() -> None:
    """diagnose() must agree with evaluate() on the blocking gate."""
    print("\ndiagnostics agree with the fast path")
    cfg = base_cfg()
    rng = random.Random(7)

    mismatches = 0
    fired = 0
    for trial in range(40):
        s = Series("TESTUSDT", cfg.strategy.lookback_bars)
        price = 100.0
        for i in range(rng.randint(20, 110)):
            price *= 1.0 + rng.gauss(0.0, 0.004)
            vol = abs(rng.gauss(100.0, 60.0)) + 1.0
            s.update(candle(i * MINUTE, price, volume=vol))
        spread = rng.uniform(0.0, 20.0)

        fast = evaluate(s, cfg.strategy, spread)
        slow = diagnose(s, cfg.strategy, spread)
        fast_reason = fast.reason if isinstance(fast, Rejection) else ""
        # no_volatility_estimate is a fast-path-only guard folded into the
        # z-score gate in diagnose(); treat the two names as equivalent.
        equivalent = fast_reason == slow.blocking_gate or (
            fast_reason == "no_volatility_estimate"
            and slow.blocking_gate == "move_within_normal_range"
        )
        if not equivalent:
            mismatches += 1
            if mismatches == 1:
                print(f"       trial {trial}: fast={fast_reason!r} slow={slow.blocking_gate!r}")
        if not fast_reason:
            fired += 1

    check("blocking gate matches on 40 random paths", mismatches == 0,
          f"{mismatches} mismatches, {fired} fired")

    s = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        s.update(c)
    d = diagnose(s, cfg.strategy, 2.0)
    check("clean breakout reports ready", d.ready and d.heat >= 0.99,
          f"blocking={d.blocking_gate!r} heat={d.heat:.2f}")
    check("gate map covers every gate", len(d.gates) == 9, str(len(d.gates)))

    cold = Series("TESTUSDT", 120)
    for c in flat_then_spike()[:20]:
        cold.update(c)
    check("heat stays low when nothing is happening",
          diagnose(cold, cfg.strategy, 1.0).heat < 0.9)


def test_relaxed_gates() -> None:
    print("\nrelaxed gates widen the funnel")
    cfg = base_cfg()
    rng = random.Random(11)

    strict = relaxed = 0
    for _ in range(200):
        # Spikes of varying strength: some clear the strict gates, some only
        # clear the relaxed ones, some clear neither.
        bump = rng.uniform(1.0005, 1.005)
        vol = rng.uniform(120.0, 900.0)
        s = Series("TESTUSDT", cfg.strategy.lookback_bars)
        price = 100.0
        for i in range(100):
            price *= 1.0 + (0.0004 if i % 2 else -0.0004)
            s.update(candle(i * MINUTE, price, volume=100.0))
        for i in range(3):
            price *= bump
            s.update(candle((100 + i) * MINUTE, price, volume=vol))

        if not isinstance(evaluate(s, cfg.strategy, 1.0), Rejection):
            strict += 1
        if not isinstance(evaluate(s, cfg.strategy, 1.0, relax=0.6), Rejection):
            relaxed += 1

    check("relaxed fires at least as often", relaxed >= strict,
          f"strict={strict} relaxed={relaxed}")
    check("relaxed actually finds more", relaxed > strict,
          f"strict={strict} relaxed={relaxed}")


def test_simulate_matches_strategy() -> None:
    """The labeller and the live engine must produce identical trades.

    If they drift, the model trains on outcomes the bot could never achieve.
    """
    print("\nsimulate_exit == Strategy exits")
    cfg = base_cfg()
    rng = random.Random(3)

    slip = cfg.fees.slippage_bps / 10_000.0
    compared = mismatches = 0
    for trial in range(60):
        broker = PaperBroker(cfg)
        strategy = Strategy(cfg, broker, RiskManager(cfg))

        # Stop feeding history the moment the position opens, so the forward
        # path starts at the same bar for both implementations. The breakout
        # fires mid-spike, not on the last candle -- comparing from the end of
        # history would silently compare two different trades.
        entry_bar = None
        for c in flat_then_spike():
            strategy.on_candle("TESTUSDT", c)
            if "TESTUSDT" in strategy.positions:
                entry_bar = c
                break
        pos = strategy.positions.get("TESTUSDT")
        if pos is None or entry_bar is None:
            continue

        # A random forward path, fed to both implementations.
        price = entry_bar.close
        future = []
        for i in range(1, cfg.risk.max_hold_minutes + 1):
            price *= 1.0 + rng.gauss(0.0, 0.006)
            wiggle = abs(rng.gauss(0.0, 0.004))
            future.append(
                Candle(entry_bar.open_time + i * MINUTE, price, price * (1 + wiggle),
                       price * (1 - wiggle), price, 100.0, 100.0 * price, True)
            )

        # Recover the exact price the broker filled against.
        signal_price = pos.entry_price / (1.0 + slip)
        expected = simulate_exit(signal_price, future, cfg,
                                 fixed_levels(cfg), cfg.fees.slippage_bps)

        for c in future:
            if "TESTUSDT" not in strategy.positions:
                break
            strategy.on_candle("TESTUSDT", c)

        if expected is None or not broker.trades:
            continue
        got = broker.trades[-1]
        compared += 1
        same_reason = got.exit_reason == expected.reason
        same_pct = abs(got.net_pct - expected.net_pct) < 0.02
        if not (same_reason and same_pct):
            mismatches += 1
            if mismatches == 1:
                print(f"       trial {trial}: engine {got.exit_reason} {got.net_pct:+.3f}% "
                      f"vs sim {expected.reason} {expected.net_pct:+.3f}%")

    check("both agree on every resolved path", compared > 0 and mismatches == 0,
          f"{compared} compared, {mismatches} mismatched")


def test_features() -> None:
    print("\nfeature extraction")
    s = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        s.update(c)

    vec = extract(s, MarketContext(ret_15=0.5, ret_60=1.0, sigma=0.2))
    check("vector length matches names", len(vec) == len(FEATURE_NAMES),
          f"{len(vec)} vs {len(FEATURE_NAMES)}")
    check("all finite", all(v == v and abs(v) < 1e6 for v in vec))
    check("market context reaches the vector",
          vec[FEATURE_NAMES.index("market_ret_15")] == 0.5)
    check("hour encoding is on the unit circle",
          abs(vec[FEATURE_NAMES.index("hour_sin")] ** 2
              + vec[FEATURE_NAMES.index("hour_cos")] ** 2 - 1.0) < 1e-9)

    cold = Series("TESTUSDT", 120)
    for c in flat_then_spike()[:30]:
        cold.update(c)
    check("short history yields zeros, not a crash",
          extract(cold) == [0.0] * len(FEATURE_NAMES))

    # Deterministic: the same bar must always produce the same vector, or
    # training and inference silently disagree.
    check("extraction is deterministic", extract(s, MarketContext()) == extract(s, MarketContext()))


def test_ml_pipeline() -> None:
    print("\nML pipeline (synthetic, learnable signal)")
    try:
        from bot.ml import Sample, train
    except ImportError as exc:
        check("sklearn available", False, str(exc))
        return

    rng = random.Random(5)
    n = len(FEATURE_NAMES)
    samples = []
    for i in range(1200):
        feats = [rng.gauss(0, 1) for _ in range(n)]
        # Feature 0 genuinely predicts the outcome; everything else is noise.
        edge = feats[0] > 0.3
        net = rng.gauss(0.8 if edge else -0.6, 0.9)
        samples.append(
            Sample("TESTUSDT", i * MINUTE, feats, net, 1 if net > 0 else 0, "test")
        )

    model, report = train(samples, threshold=0.55, embargo_minutes=1)
    check("learns a real signal", report.auc > 0.65, f"AUC {report.auc:.3f}")
    check("filtering improves expectancy", report.improvement > 0,
          f"{report.expectancy_all:+.3f}% -> {report.expectancy_filtered:+.3f}%")
    check("keeps a usable slice", 0.05 < report.kept_fraction < 0.95,
          f"{report.kept_fraction:.2f}")
    check("probability in range", 0.0 <= model.probability(samples[0].features) <= 1.0)
    check("verdict is not a failure", not report.verdict.startswith(("USELESS", "HARMFUL")),
          report.verdict)

    # Pure noise must be reported as useless rather than dressed up.
    noise = [
        Sample("X", i * MINUTE, [rng.gauss(0, 1) for _ in range(n)],
               v := rng.gauss(0, 1), 1 if v > 0 else 0, "test")
        for i in range(1200)
    ]
    _, noise_report = train(noise, threshold=0.55, embargo_minutes=1)
    check("noise is called out", noise_report.auc < 0.60, f"AUC {noise_report.auc:.3f}")

    check("too little data is refused", _raises(lambda: train(samples[:50], 0.55)))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_dashboard_snapshot() -> None:
    print("\ndashboard snapshot")
    import json

    from bot.dashboard import DashboardState, LogTail

    cfg = base_cfg()
    broker = PaperBroker(cfg)
    risk = RiskManager(cfg)
    strategy = Strategy(cfg, broker, risk)
    strategy.collect_diagnostics = True
    for c in flat_then_spike():
        strategy.on_candle("TESTUSDT", c)

    state = DashboardState(cfg, strategy, broker, risk, LogTail())
    state.symbols = ["TESTUSDT"]
    state.record_equity(MINUTE, 1000.0)
    snap = state.snapshot()

    check("serialises to JSON", isinstance(json.dumps(snap, default=str), str))
    for key in ("mode", "equity", "positions", "watchlist", "trades",
                "signals", "rejections", "equity_curve", "logs", "stats", "ml"):
        check(f"snapshot has {key}", key in snap)
    check("watchlist populated", len(snap["watchlist"]) == 1,
          str(len(snap["watchlist"])))
    check("watchlist row carries gate detail",
          bool(snap["watchlist"][0]["gates"]))
    check("open position surfaced", len(snap["positions"]) == 1,
          str(len(snap["positions"])))
    for key in ("gross_pnl", "fees_paid", "trades", "win_rate"):
        check(f"stats has {key}", key in snap["stats"])
    check("exit reasons reported", isinstance(snap["exit_reasons"], dict))
    check("spread band reported", len(snap["spread_band"]) == 2)

    # The counterfactual curve is the viewer's headline: actual vs no-fee.
    state.record_equity(2 * MINUTE, 995.0)
    point = state.equity_curve[-1]
    check("curve point carries both series", len(point) == 3, str(len(point)))
    check("no-fee value is actual plus fees handed over",
          abs(point[2] - (995.0 + broker.fees_paid)) < 1e-9,
          f"{point[2]:.4f} vs {995.0 + broker.fees_paid:.4f}")
    check("fees were actually accrued by the open position",
          broker.fees_paid > 0, f"{broker.fees_paid:.6f}")

    # Every exit reason and gate the page can receive must have a human label,
    # or the UI shows raw identifiers to the user.
    html = (Path(__file__).resolve().parents[1] / "bot" / "static" / "dashboard.html").read_text(encoding="utf-8")
    from bot.signals import GATE_ORDER
    labelled = html[html.index("const GATE_LABEL"):html.index("const label =")]
    missing = [k for k in (*GATE_ORDER, "stop_loss", "trailing_stop", "take_profit",
                           "max_hold", "end_of_data", "shutdown", "ml_filter",
                           "cooldown", "max_positions", "daily_loss_limit")
               if f"{k}:" not in labelled]
    check("every gate and exit reason has a UI label", not missing, str(missing))

    check("dashboard html exists",
          (Path(__file__).resolve().parents[1] / "bot" / "static" / "dashboard.html").exists())
    check("sparkline attached to hot rows",
          isinstance(snap["watchlist"][0].get("spark"), list)
          and len(snap["watchlist"][0]["spark"]) > 1,
          str(len(snap["watchlist"][0].get("spark") or [])))


def test_findings_store() -> None:
    """The page must never be handed the friendlier of two available p-values.

    Both exist for the same measurement: a trade-level p that assumes twenty
    signals from one market-wide dump are twenty independent facts, and a
    clustered one that does not. The first is smaller and is the one a reader
    would quote. Which one reaches the dashboard is therefore not a display
    detail -- it decides what the project appears to have found.
    """
    import tempfile
    print(chr(10) + "findings store")
    from bot import findings

    with tempfile.TemporaryDirectory() as tmp:
        cfg = base_cfg()
        cfg.root = Path(tmp)
        check("an empty store reports no data",
              findings.summary(cfg) == {"has_data": False})

        findings.record(cfg, "nullcheck",
                        {"edge": 0.05, "p": 0.011, "block_p": 0.145, "n": 1089})
        s = findings.summary(cfg)
        check("the clustered p is the one published", s["p"] == 0.145,
              f"{s['p']} (trade-level was 0.011)")
        check("and the basis says so", s["basis"] == "blocks", s["basis"])

        # A between-window number assumes even less, so it outranks both.
        findings.record(cfg, "between_window",
                        {"mean_edge": 0.039, "p": 0.079, "df": 3, "windows": 4,
                         "positive": 4, "n": 34740})
        s = findings.summary(cfg)
        check("between-window outranks the block bootstrap", s["p"] == 0.079,
              str(s["p"]))
        check("it carries how many windows agreed",
              s["positive"] == 4 and s["windows"] == 4)

        # Recording the same kind twice keeps only the newer one: a store that
        # accumulated history would invite quoting the run that agreed with you.
        findings.record(cfg, "between_window",
                        {"mean_edge": 0.001, "p": 0.9, "df": 3, "windows": 4})
        check("a rerun replaces rather than accumulates",
              findings.summary(cfg)["p"] == 0.9)

        # The ladder is a per-year allocation claim, not a per-trade edge, and
        # must not be averaged into the headline. It travels in its own key or
        # the page would report one number for two different questions.
        findings.record(cfg, "ladder",
                        {"symbol": "^GSPC", "windows": 10, "window_years": 10.0,
                         "mean_cagr_diff": -2.82, "timing_cagr_diff": -0.48,
                         "allocation_cagr_diff": -2.34, "p_adjusted": 0.217,
                         "max_dd": 20.2, "hold_dd": 37.9})
        s = findings.summary(cfg)
        check("the ladder does not overwrite the edge headline",
              s["p"] == 0.9 and s["edge"] == 0.001)
        check("the ladder travels in its own key", "ladder" in s)
        check("and carries the timing/allocation split, not just the total",
              s["ladder"]["timing"] == -0.48
              and s["ladder"]["allocation"] == -2.34,
              str(s["ladder"]))

        # The vendor cross-check rides along with the claim it supports, so a
        # check that ran is a check the page can show.
        findings.record(cfg, "vendorcheck", {"median_bps": 0.4, "months": 1183})
        s = findings.summary(cfg)
        check("the price cross-check reaches the ladder row",
              s["ladder"]["vendor_bps"] == 0.4
              and s["ladder"]["vendor_months"] == 1183, str(s["ladder"]))

        # A ladder run on its own must still reach the page.
        with tempfile.TemporaryDirectory() as only:
            solo = base_cfg()
            solo.root = Path(only)
            findings.record(solo, "ladder", {"symbol": "^GSPC",
                                             "mean_cagr_diff": -2.82})
            s2 = findings.summary(solo)
            check("a ladder-only store still has data",
                  s2["has_data"] and s2.get("edge") is None
                  and s2["ladder"]["total"] == -2.82)

        # Bookkeeping must never take down a completed measurement.
        cfg.root = Path(tmp) / "nul" / "impossible"
        findings.record(cfg, "nullcheck", {"edge": 1.0})
        check("an unwritable path is survived, not raised", True)

    src = (Path(__file__).resolve().parents[1] / "bot" / "findings.py"
           ).read_text(encoding="utf-8")
    check("the ordering rule is written down where it is implemented",
          "assumption known to be false" in src)


def test_dashboard_integrity() -> None:
    """Static checks on the page that the browser would only fail silently.

    A `var(--typo)` renders as nothing, a `$("typo")` returns null and the
    handler dies on the next line, and a colour defined only inside the
    dark-mode block leaves light mode transparent. None of these throw where
    anyone can see them -- the page just quietly looks wrong, and only for the
    reader in the wrong mode. Cheap to check here, expensive to notice there.
    """
    import re
    print(chr(10) + "dashboard integrity")
    html = (Path(__file__).resolve().parents[1]
            / "bot" / "static" / "dashboard.html").read_text(encoding="utf-8")

    declared = set(re.findall(r"^\s*(--[a-z0-9-]+):", html, re.M))
    # var(--x, fallback) survives an undeclared token; var(--x) does not.
    bare = set(re.findall(r"var\((--[a-z0-9-]+)\s*\)", html))
    check("every bare var() resolves to a declared token",
          not (bare - declared), str(sorted(bare - declared)))

    # A token first defined inside the dark block is undefined in light mode.
    root = html.split("@media (prefers-color-scheme: dark)")[0]
    light = set(re.findall(r"^\s*(--[a-z0-9-]+):", root, re.M))
    check("no colour is defined only in dark mode",
          not (declared - light), str(sorted(declared - light)))

    created = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    created |= set(re.findall(r'\.id\s*=\s*"([A-Za-z0-9_-]+)"', html))
    looked_up = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', html))
    looked_up |= set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', html))
    check("every element the script looks up actually exists",
          not (looked_up - created), str(sorted(looked_up - created)))

    # The page must never grow a control that could move money. It is a viewer,
    # and the guarantee is worth more than any feature that would break it.
    for name, needle in (("order endpoint", "/api/order"),
                         ("POST request", "method:"),
                         ("form submission", "<form")):
        check(f"read-only: page contains no {name}", needle not in html)

    check("the caveat about clustering survives edits to the explainer",
          "not distinguishable from luck" in html
          or "no longer distinguishable from luck" in html)


def test_cache() -> None:
    """The cache must never invent data, and must spot its own gaps."""
    print("\nkline cache")
    import tempfile

    from bot.cache import KlineCache

    with tempfile.TemporaryDirectory() as tmp:
        cache = KlineCache(Path(tmp))
        step = MINUTE
        candles = [candle(i * step, 100.0 + i) for i in range(10)]

        have = {c.open_time: c for c in candles}
        check("complete range reports no gaps",
              cache._missing_runs(have, 0, 9 * step, step) == [])

        holed = {t: c for t, c in have.items() if t not in (3 * step, 4 * step)}
        runs = cache._missing_runs(holed, 0, 9 * step, step)
        check("internal gap detected", runs == [(3 * step, 4 * step)], str(runs))

        head = {t: c for t, c in have.items() if t >= 5 * step}
        check("missing head detected",
              cache._missing_runs(head, 0, 9 * step, step) == [(0, 4 * step)])

        tail = {t: c for t, c in have.items() if t <= 5 * step}
        check("missing tail detected",
              cache._missing_runs(tail, 0, 9 * step, step) == [(6 * step, 9 * step)])

        check("empty cache is one big gap",
              cache._missing_runs({}, 0, 9 * step, step) == [(0, 9 * step)])

        cache._save("TESTUSDT", "1m", have)
        loaded = cache._load("TESTUSDT", "1m")
        check("round-trips through disk",
              len(loaded) == 10 and loaded[0].close == candles[0].close)
        check("corrupt file degrades to empty, not a crash",
              _corrupt_reload(cache) == {})


def _corrupt_reload(cache) -> dict:
    path = cache._path("BROKENUSDT", "1m")
    path.write_bytes(b"not a gzip stream")
    return cache._load("BROKENUSDT", "1m")


def test_reseed_replaces() -> None:
    """Repairing a stale symbol must rebuild the window, not append to it."""
    print("\nstale repair rebuilds the window")
    cfg = base_cfg()
    strategy = Strategy(cfg, PaperBroker(cfg), RiskManager(cfg))

    first = [candle(i * MINUTE, 100.0 + i) for i in range(60)]
    strategy.seed("TESTUSDT", first)
    check("seeded", len(strategy.series["TESTUSDT"]) == 60)

    # Overlapping refill, as REST would return after a dropped stream.
    second = [candle(i * MINUTE, 200.0 + i) for i in range(30, 90)]
    strategy.seed("TESTUSDT", second)
    s = strategy.series["TESTUSDT"]
    times = [c.open_time for c in s.closed]
    check("no duplicated bars", len(times) == len(set(times)), str(len(times)))
    check("times strictly increasing",
          all(b > a for a, b in zip(times, times[1:])))
    check("window holds the refill", s.closed[-1].close == second[-1].close)
    check("last price updated", strategy.last_price["TESTUSDT"] == second[-1].close)

    check("empty refill is ignored",
          (strategy.seed("TESTUSDT", []), len(strategy.series["TESTUSDT"]))[1] == 60)


def test_atr_levels() -> None:
    print("\nATR-scaled exits")
    cfg = base_cfg()

    quiet = Series("QUIET", 120)
    wild = Series("WILD", 120)
    price_q = price_w = 100.0
    rng = random.Random(21)
    for i in range(80):
        price_q *= 1.0 + rng.gauss(0, 0.0005)
        price_w *= 1.0 + rng.gauss(0, 0.008)
        for s, p, spread in ((quiet, price_q, 0.0004), (wild, price_w, 0.012)):
            s.update(Candle(i * MINUTE, p, p * (1 + spread), p * (1 - spread),
                            p, 100.0, 100.0 * p, True))

    check("quiet symbol has small ATR", 0 < quiet.atr_pct(30) < 0.5,
          f"{quiet.atr_pct(30):.4f}%")
    check("wild symbol has much larger ATR", wild.atr_pct(30) > quiet.atr_pct(30) * 5,
          f"{wild.atr_pct(30):.4f}% vs {quiet.atr_pct(30):.4f}%")

    cfg.risk.stop_mode = "fixed"
    check("fixed mode ignores the series",
          levels_for(cfg, wild).stop_pct == cfg.risk.stop_loss_pct)

    cfg.risk.stop_mode = "atr"
    lq, lw = levels_for(cfg, quiet), levels_for(cfg, wild)
    check("wild symbol gets the wider stop", lw.stop_pct > lq.stop_pct,
          f"{lw.stop_pct:.3f}% vs {lq.stop_pct:.3f}%")
    check("stops respect the floor", lq.stop_pct >= cfg.risk.min_stop_pct - 1e-9,
          f"{lq.stop_pct:.3f}%")
    check("stops respect the ceiling", lw.stop_pct <= cfg.risk.max_stop_pct + 1e-9,
          f"{lw.stop_pct:.3f}%")
    for name, lv in (("quiet", lq), ("wild", lw)):
        ratio = lv.target_pct / lv.stop_pct
        expected = cfg.risk.target_atr_mult / cfg.risk.stop_atr_mult
        check(f"{name}: risk:reward preserved through the bounds",
              abs(ratio - expected) < 1e-9, f"{ratio:.4f} vs {expected:.4f}")

    check("no history falls back to fixed",
          levels_for(cfg, Series("EMPTY", 120)).stop_pct == cfg.risk.stop_loss_pct)

    stop, target = lw.apply(100.0)
    check("levels apply around the entry", stop < 100.0 < target)

    # An ATR target that cannot clear costs must be refused, like the fixed one.
    cfg.risk.min_stop_pct = 0.05
    cfg.risk.target_atr_mult = 1.6
    check("unpayable ATR target rejected", _raises(cfg.validate))


def test_timeframe_generality() -> None:
    """Nothing may assume a candle is one minute long."""
    print("\ntimeframe is not hardcoded")
    cfg = base_cfg()
    check("1m bar length", cfg.bar_ms == 60_000, str(cfg.bar_ms))
    check("1m hold in bars", cfg.max_hold_bars == cfg.risk.max_hold_minutes,
          str(cfg.max_hold_bars))

    cfg.strategy.timeframe = "15m"
    check("15m bar length", cfg.bar_ms == 900_000, str(cfg.bar_ms))
    check("30 minute hold is 2 bars of 15m", cfg.max_hold_bars == 2,
          str(cfg.max_hold_bars))

    cfg.strategy.timeframe = "1h"
    check("a hold shorter than one bar still holds one",
          cfg.max_hold_bars == 1, str(cfg.max_hold_bars))

    cfg.strategy.timeframe = "7s"
    check("unknown timeframe refused", _raises(lambda: cfg.bar_ms))

    # The time stop must fire on bar count, not on a minute count.
    cfg = base_cfg()
    cfg.strategy.timeframe = "5m"
    bars = cfg.max_hold_bars
    check("5m: 30 minute hold is 6 bars", bars == 6, str(bars))
    entry = 100.0
    flat = [Candle(i * 300_000, entry, entry * 1.0001, entry * 0.9999,
                   entry, 1.0, 100.0, True) for i in range(1, 20)]
    out = simulate_exit(entry, flat, cfg, fixed_levels(cfg), cfg.fees.slippage_bps)
    check("time stop fires after exactly max_hold_bars",
          out is not None and out.bars_held == bars,
          f"{out.bars_held if out else None}")
    check("and reports max_hold", out is not None and out.reason == "max_hold")


def test_reversal_direction() -> None:
    print("\nreversal mirrors momentum")
    cfg = base_cfg()

    def crash(bars: int = 100) -> list[Candle]:
        out, price = [], 100.0
        for i in range(bars):
            price *= 1.0 + (0.0004 if i % 2 else -0.0004)
            out.append(candle(i * MINUTE, price, volume=100.0))
        for i in range(3):
            price *= 0.996
            out.append(candle((bars + i) * MINUTE, price, volume=800.0))
        return out

    up, down = Series("UP", 120), Series("DOWN", 120)
    for c in flat_then_spike():
        up.update(c)
    for c in crash():
        down.update(c)

    cfg.strategy.direction = "momentum"
    check("momentum fires on the spike",
          not isinstance(evaluate(up, cfg.strategy, 1.0), Rejection))
    check("momentum ignores the crash",
          isinstance(evaluate(down, cfg.strategy, 1.0), Rejection),
          getattr(evaluate(down, cfg.strategy, 1.0), "reason", ""))

    cfg.strategy.direction = "reversal"
    check("reversal fires on the crash",
          not isinstance(evaluate(down, cfg.strategy, 1.0), Rejection),
          getattr(evaluate(down, cfg.strategy, 1.0), "reason", ""))
    check("reversal ignores the spike",
          isinstance(evaluate(up, cfg.strategy, 1.0), Rejection),
          getattr(evaluate(up, cfg.strategy, 1.0), "reason", ""))

    d = diagnose(down, cfg.strategy, 1.0)
    check("reversal diagnostics agree with the fast path",
          d.ready, f"blocking={d.blocking_gate!r}")
    check("reversal reports a positive move magnitude", d.ret_pct > 0,
          f"{d.ret_pct:+.2f}")
    check("lowest_close is the reversal breakout level",
          abs(d.breakout_level - down.lowest_close(cfg.strategy.breakout_bars, 1)) < 1e-12)
    check("bad direction refused",
          _raises(lambda: _with(cfg, "direction", "sideways")))


def _with(cfg, attr, value):
    setattr(cfg.strategy, attr, value)
    cfg.validate()


def test_fee_discount() -> None:
    print("\nBNB fee discount")
    cfg = base_cfg()
    full = cfg.round_trip_cost_pct
    cfg.fees.bnb_discount = True
    discounted = cfg.round_trip_cost_pct
    check("discount lowers the round trip", discounted < full,
          f"{full:.3f}% -> {discounted:.3f}%")
    check("25% off the fee leg only",
          abs(discounted - (2 * (cfg.fees.taker_bps * 0.75 + cfg.fees.slippage_bps) / 100)) < 1e-12)

    broker = PaperBroker(cfg)
    pos = broker.buy("TESTUSDT", 200.0, 100.0, 0, fixed_levels(cfg))
    trade = broker.sell(pos, 100.0, MINUTE, "test")
    check("broker charges the discounted rate",
          abs(trade.net_pct + discounted) < 0.02, f"{trade.net_pct:.4f}%")


def _bounce_series(spread_bps: float, bars: int = 3000, seed: int = 1,
                   tick_vol: float = 0.0004) -> list[Candle]:
    """Candles from an efficient random walk plus a bid-ask bounce.

    The observed close is a trade print, so it sits half a spread above or below
    the efficient price at random. That contamination is exactly what the
    Abdi-Ranaldo estimator is designed to measure.
    """
    rng = random.Random(seed)
    half = spread_bps / 2 / 10_000.0
    out = []
    price = 100.0
    for i in range(bars):
        prints = []
        for _ in range(12):
            price *= 1.0 + rng.gauss(0, tick_vol)
            prints.append(price * (1.0 + (half if rng.random() < 0.5 else -half)))
        out.append(
            Candle(i * MINUTE, prints[0], max(prints), min(prints), prints[-1],
                   100.0, 100.0 * prints[-1], True)
        )
    return out


def test_intrabar_bracket() -> None:
    """The both-barriers-touched rule must be declarable, and must match."""
    print("\nintrabar fill assumption")
    entry = 100.0

    def outcome_for(mode: str, open_price: float):
        cfg = base_cfg()
        cfg.risk.intrabar_fill = mode
        lv = fixed_levels(cfg)
        stop, target = lv.apply(entry * (1 + cfg.fees.slippage_bps / 10_000))
        # One candle that spans both barriers.
        both = Candle(MINUTE, open_price, target * 1.001, stop * 0.999,
                      entry, 1.0, 100.0, True)
        return simulate_exit(entry, [both], cfg, lv, cfg.fees.slippage_bps)

    pess = outcome_for("pessimistic", entry)
    opti = outcome_for("optimistic", entry)
    check("pessimistic takes the stop", pess.reason == "stop_loss", pess.reason)
    check("optimistic takes the target", opti.reason == "take_profit", opti.reason)
    check("the bracket is wide", opti.net_pct - pess.net_pct > 3.0,
          f"{pess.net_pct:+.2f}% .. {opti.net_pct:+.2f}%")

    near_stop = outcome_for("open", 98.9)
    near_target = outcome_for("open", 102.3)
    check("open near the stop resolves to the stop",
          near_stop.reason == "stop_loss", near_stop.reason)
    check("open near the target resolves to the target",
          near_target.reason == "take_profit", near_target.reason)

    cfg = base_cfg()
    cfg.risk.intrabar_fill = "whatever"
    check("bad mode refused", _raises(cfg.validate))

    # The live engine must resolve it identically or the bot and every
    # measurement of the bot are different strategies.
    for mode in ("pessimistic", "optimistic"):
        cfg = base_cfg()
        cfg.risk.intrabar_fill = mode
        broker = PaperBroker(cfg)
        strategy = Strategy(cfg, broker, RiskManager(cfg))
        entry_bar = None
        for c in flat_then_spike():
            strategy.on_candle("TESTUSDT", c)
            if "TESTUSDT" in strategy.positions:
                entry_bar = c
                break
        pos = strategy.positions["TESTUSDT"]
        span = Candle(entry_bar.open_time + MINUTE, pos.entry_price,
                      pos.target_price * 1.001, pos.stop_price * 0.999,
                      pos.entry_price, 1.0, 100.0, True)
        strategy.on_candle("TESTUSDT", span)
        engine_reason = broker.trades[-1].exit_reason
        sim = simulate_exit(pos.entry_price / (1 + cfg.fees.slippage_bps / 10_000),
                            [span], cfg, fixed_levels(cfg), cfg.fees.slippage_bps)
        check(f"engine and simulator agree under {mode}",
              engine_reason == sim.reason, f"{engine_reason} vs {sim.reason}")


def test_ml_labels_match_the_bot() -> None:
    """ML labels must be the outcome the LIVE bot would have got.

    The module claims labels come from the same exit rules the bot uses. That
    claim silently became false when ATR levels and per-symbol costs were
    added, because build_dataset was still calling simulate_exit with the flat
    defaults. This pins the claim down so it cannot rot again.
    """
    print("\nML labels match the live strategy")
    try:
        from bot.ml import build_dataset
    except ImportError as exc:
        check("sklearn available", False, str(exc))
        return

    from bot.costs import CostModel
    from bot.spread import estimate_spreads

    cfg = base_cfg()
    cfg.risk.stop_mode = "atr"
    cfg.fees.slippage_mode = "estimated"

    # Two symbols with very different volatility, so ATR levels must differ.
    rng = random.Random(31)
    history = {}
    for name, vol in (("QUIETUSDT", 0.0008), ("WILDUSDT", 0.010)):
        out, price = [], 100.0
        for i in range(400):
            price *= 1.0 + rng.gauss(0.0, vol)
            w = abs(rng.gauss(0, vol))
            out.append(Candle(i * MINUTE, price, price * (1 + w), price * (1 - w),
                              price, abs(rng.gauss(200, 150)) + 1,
                              100.0 * price, True))
        history[name] = out

    samples = build_dataset(cfg, history, relax=0.5)
    check("dataset produced rows", len(samples) > 0, str(len(samples)))
    if not samples:
        return

    costs = CostModel(cfg, estimate_spreads(history))
    lookback = cfg.strategy.lookback_bars
    horizon = cfg.max_hold_bars
    mismatches = 0
    for s in samples[:40]:
        candles = history[s.symbol]
        idx = next(i for i, c in enumerate(candles) if c.open_time == s.time_ms)
        series = Series(s.symbol, lookback)
        for c in candles[max(0, idx - lookback + 1): idx + 1]:
            series.update(c)
        expected = simulate_exit(
            candles[idx].close, candles[idx + 1: idx + 1 + horizon], cfg,
            levels_for(cfg, series), costs.slippage_bps(s.symbol),
        )
        if expected is None or abs(expected.net_pct - s.net_pct) > 1e-9:
            mismatches += 1
    check("labels equal the strategy's own outcome", mismatches == 0,
          f"{mismatches} of {min(40, len(samples))} differ")

    # And prove the levels are actually threaded: fixed mode must relabel.
    fixed = base_cfg()
    fixed.fees.slippage_mode = "estimated"
    fixed_samples = build_dataset(fixed, history, relax=0.5)
    by_key = {(s.symbol, s.time_ms): s.net_pct for s in fixed_samples}
    differing = sum(
        1 for s in samples
        if (s.symbol, s.time_ms) in by_key
        and abs(by_key[(s.symbol, s.time_ms)] - s.net_pct) > 1e-9
    )
    check("ATR mode relabels vs fixed mode", differing > 0,
          f"{differing} rows differ")


def test_clustering_statistics() -> None:
    """Trades are not independent, and the arithmetic has to know it.

    A market-wide dump fires the entry rule on twenty symbols in the same
    minute. Those twenty outcomes are close to one observation repeated, but
    every interval in this project treated them as twenty independent draws.
    Two corrections exist now and both are checked here: a block bootstrap that
    resamples whole spans of time, and a Student t on the tiny sample that
    remains when each window counts once.
    """
    print(chr(10) + "clustering and small-sample statistics")
    from bot.nullcheck import block_bootstrap
    from bot.replicate import student_t_p

    # 1. Student t must match published critical values, because the normal
    #    approximation says p=0.009 where the truth is p=0.079 at df=3.
    worst = 0.0
    for df, crit in ((1, 12.706), (2, 4.303), (3, 3.182), (5, 2.571),
                     (10, 2.228), (30, 2.042), (100, 1.984)):
        worst = max(worst, abs(student_t_p(crit, df) - 0.05))
    check("t distribution matches the 5% critical values",
          worst < 0.002, f"worst deviation {worst:.4f}")
    check("and the 1% ones",
          abs(student_t_p(5.841, 3) - 0.01) < 0.002, f"{student_t_p(5.841, 3):.4f}")
    check("small df is far stricter than the normal it replaces",
          student_t_p(2.63, 3) > 8 * student_t_p(2.63, 10_000),
          f"{student_t_p(2.63, 3):.4f} vs {student_t_p(2.63, 10_000):.5f}")
    check("df below 1 is refused", student_t_p(3.0, 0) == 1.0)

    # The Cauchy tail at df=1 holds real mass past any practical cut-off, so
    # the remainder is integrated analytically rather than outrun. Without it
    # the df=1 critical value came out 0.05 low.
    check("the df=1 tail is not truncated away",
          abs(student_t_p(12.706, 1) - 0.05) < 0.001,
          f"{student_t_p(12.706, 1):.5f}")

    # 2. Fixing the p-value and leaving 1.96 in the interval would be the same
    #    error wearing a different hat.
    from bot.replicate import t_critical
    worst_c = max(abs(t_critical(df) - crit)
                  for df, crit in ((1, 12.706), (2, 4.303), (3, 3.182),
                                   (4, 2.776), (10, 2.228), (100, 1.984)))
    check("interval multipliers match the published table",
          worst_c < 0.01, f"worst deviation {worst_c:.4f}")
    check("df=3 widens the interval by more than a third over 1.96",
          t_critical(3) / 1.96 > 1.35, f"{t_critical(3) / 1.96:.2f}x")
    check("large df converges on the normal",
          abs(t_critical(5000) - 1.96) < 0.01, f"{t_critical(5000):.4f}")

    # Both places that build a small-sample interval must reach for the
    # multiplier rather than hard-code one.
    root = Path(__file__).resolve().parents[1] / "bot"
    for name in ("replicate.py", "equities.py"):
        body = (root / name).read_text(encoding="utf-8")
        check(f"{name} takes its interval width from t_critical",
              "t_critical(" in body)

    # 2. Block bootstrap must widen the interval when trades cluster, and
    #    agree with the trade-level one when they do not.
    rng = random.Random(77)
    hour = 3_600_000
    clustered_t, clustered_v, flat_t, flat_v = [], [], [], []
    for block in range(40):
        # Every burst shares one shock: 25 trades that are really one draw.
        shock = rng.gauss(0.0, 1.0)
        for k in range(25):
            clustered_t.append(block * 6 * hour + k * 60_000)
            clustered_v.append(shock + rng.gauss(0.0, 0.05))
            flat_t.append(block * 6 * hour + k * 60_000)
            flat_v.append(rng.gauss(0.0, 1.0))

    ctrl_t = list(flat_t)
    ctrl_v = [rng.gauss(0.0, 1.0) for _ in flat_v]
    c_lo, c_hi, _ = block_bootstrap(clustered_t, clustered_v, ctrl_t, ctrl_v)
    f_lo, f_hi, _ = block_bootstrap(flat_t, flat_v, ctrl_t, ctrl_v)
    check("clustered data gets the wider interval",
          (c_hi - c_lo) > (f_hi - f_lo) * 1.5,
          f"clustered {c_hi - c_lo:.3f} vs independent {f_hi - f_lo:.3f}")

    check("too few blocks refuses to produce an interval",
          block_bootstrap([0, 1], [1.0, 2.0], [0, 1], [1.0, 2.0]) == (0.0, 0.0, 1.0))
    check("a small sample refuses too",
          block_bootstrap(list(range(10)), [1.0] * 10,
                          list(range(10)), [0.0] * 10) == (0.0, 0.0, 1.0))


def test_volume_measure() -> None:
    """Base volume is money over price, so a falling price inflates it.

    A dip-buying strategy selects exactly the bars where the price has fallen,
    so a volume gate measured in base units has a directional tilt: the same
    dollar flow prints as more coins. Measured on real data the effect turned
    out to be negligible (0.4% of gate hits), but the two measures should still
    be distinguishable and selectable rather than silently conflated.
    """
    print(chr(10) + "volume measured in units vs money")
    s = Series("TESTUSDT", 120)
    # Constant money flow, price halving across the window. Base volume must
    # therefore double while quote volume stays flat.
    for i in range(60):
        price = 100.0 - i          # 100 down to 41
        money = 1000.0
        units = money / price
        s.update(Candle(i * MINUTE, price, price, price, price,
                        units, money, True))

    base = s.volume_mult(False)
    quote = s.volume_mult(True)
    check("base volume rises as the price falls", base > 1.3, f"{base:.3f}")
    check("quote volume does not", abs(quote - 1.0) < 0.01, f"{quote:.3f}")
    check("the tilt is upward, never downward", base > quote,
          f"{base:.3f} vs {quote:.3f}")

    # With a flat price the two must agree exactly.
    flat = Series("FLAT", 120)
    for i in range(60):
        flat.update(Candle(i * MINUTE, 10, 10, 10, 10, 5.0, 50.0, True))
    check("a flat price makes them identical",
          abs(flat.volume_mult(False) - flat.volume_mult(True)) < 1e-9)

    check("too little history returns zero either way",
          Series("X", 120).volume_mult(False) == 0.0
          and Series("X", 120).volume_mult(True) == 0.0)

    cfg = base_cfg()
    cfg.strategy.volume_measure = "nonsense"
    check("an unknown measure is refused", _raises(cfg.validate))


def test_volatility_matching() -> None:
    """Controls must be matched on the one thing the strategy selects on.

    The strategy only fires in dislocations, so its trades are in volatile
    minutes by construction. A uniformly drawn control is not, and with a
    fixed-percentage stop and target that alone changes the outcome
    distribution. Without matching, "edge" and "traded at a volatile moment"
    are the same number.
    """
    print(chr(10) + "volatility matching")
    from bot.nullcheck import MATCH_TOLERANCE

    check("tolerance is a fraction, not a percent",
          0 < MATCH_TOLERANCE < 1, str(MATCH_TOLERANCE))

    # The matching rule itself: a control qualifies iff its volatility sits
    # inside the band around the signal.
    def qualifies(signal_vol, control_vol):
        lo = signal_vol * (1 - MATCH_TOLERANCE)
        hi = signal_vol * (1 + MATCH_TOLERANCE)
        return lo <= control_vol <= hi

    check("an identical bar matches", qualifies(1.0, 1.0))
    check("just inside the band matches", qualifies(1.0, 1.0 + MATCH_TOLERANCE * 0.9))
    check("just outside does not", not qualifies(1.0, 1.0 + MATCH_TOLERANCE * 1.1))
    check("a calm control cannot match a violent signal", not qualifies(4.0, 1.0))
    check("and the reverse also fails", not qualifies(1.0, 4.0))
    check("the band is symmetric in ratio terms",
          qualifies(1.0, 1 - MATCH_TOLERANCE) and qualifies(1.0, 1 + MATCH_TOLERANCE))

    # A signal with no measurable volatility must be reported unmatchable
    # rather than matched against everything.
    check("zero volatility cannot be matched", not qualifies(0.0, 1.0))


def test_point_in_time_universe() -> None:
    """Rank on what was liquid THEN, not on what is liquid now.

    Every measurement here ranked symbols by today's volume and applied the
    list backwards, so a coin that pumped last week sat in the universe for
    months when nobody traded it. This removes the ranking look-ahead. It
    cannot remove delisting survivorship -- the exchange only lists what still
    trades -- and the test pins that distinction too.
    """
    print(chr(10) + "point-in-time universe")
    from bot.pointintime import (RANK_DAYS, listing_report,
                                 point_in_time_universe, quote_volume_between)

    day = 86_400_000
    window_start = 200 * day

    def series(start_day: int, vol_before: float, vol_after: float):
        out = []
        for d in range(start_day, 260):
            v = vol_before if d * day < window_start else vol_after
            out.append(Candle(d * day, 100, 101, 99, 100, v, v * 100, True))
        return out

    hist = {
        # busy before the window -- should be picked
        "OLDBUSY": series(0, 1000.0, 10.0),
        # quiet before, exploded after -- today's ranking would pick it, and
        # picking it is exactly the look-ahead being removed
        "LATEPUMP": series(0, 1.0, 100000.0),
        "MID": series(0, 500.0, 500.0),
        # did not exist before the window at all
        "NEWLISTING": series(205, 0.0, 90000.0),
    }

    picked = point_in_time_universe(hist, window_start, size=2)
    check("ranks on volume before the window",
          picked == ["OLDBUSY", "MID"], str(picked))
    check("a later pump does not earn a place", "LATEPUMP" not in picked)
    check("a symbol that did not exist yet cannot be picked",
          "NEWLISTING" not in point_in_time_universe(hist, window_start, size=4),
          str(point_in_time_universe(hist, window_start, size=4)))

    check("volume is summed over the right span",
          quote_volume_between(hist["MID"], window_start - 10 * day, window_start)
          == 500.0 * 100 * 10)
    check("a volume floor excludes the thin ones",
          point_in_time_universe(hist, window_start, size=9,
                                 min_quote_volume=1e6) == [])

    report = listing_report(hist, 250)
    check("late listings are counted", report["incomplete"] == 1,
          str(report["incomplete"]))
    check("and named", report["worst"][0][0] == "NEWLISTING",
          str(report["worst"][:1]))
    check("empty history degrades safely", listing_report({}, 30) == {})
    check("ranking window is documented", RANK_DAYS > 0)


def test_internal_call_arity() -> None:
    """Module-level helpers must be callable with the arguments they are given.

    `replicate._print` grew a fourth argument at its call site and not in its
    signature. Python only complains when the line runs, and that line runs at
    the END of a command -- so the failure surfaced after twenty-three minutes
    of downloading, on a run that had already done all its work.

    A patch that edits a call and misses the definition is the same failure
    mode as the dead imports: a bulk edit whose anchors only half-matched.
    """
    print(chr(10) + "internal call arity")
    import ast

    root = CONFIG_PATH.parent
    problems = []
    for path in sorted((root / "bot").glob("*.py")) + [root / "run.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defs = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                defs[node.name] = {
                    "positional": len(a.posonlyargs) + len(a.args),
                    "star": a.vararg is not None,
                    "kwonly": {k.arg for k in a.kwonlyargs},
                    "kwargs": a.kwarg is not None,
                    "known": {x.arg for x in a.posonlyargs + a.args}
                             | {k.arg for k in a.kwonlyargs},
                }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            spec = defs.get(node.func.id)
            if spec is None:
                continue
            if not spec["star"] and len(node.args) > spec["positional"]:
                problems.append(
                    f"{path.name}: {node.func.id}() line {node.lineno} passes "
                    f"{len(node.args)} positional, takes {spec['positional']}")
            if not spec["kwargs"]:
                for kw in node.keywords:
                    if kw.arg and kw.arg not in spec["known"]:
                        problems.append(
                            f"{path.name}: {node.func.id}() line {node.lineno} "
                            f"passes unknown keyword {kw.arg!r}")

    check(f"every internal call matches its definition", not problems,
          str(problems[:4]))


def test_no_dead_imports() -> None:
    """An unused import is usually the fossil of a half-finished refactor.

    This found a real one: `experiments.py` imported `load_quote_snapshot` and
    never called it, because a bulk edit inserted the import but its
    replacement of the CostModel construction did not match there. That command
    was silently pricing symbols differently from every other tool in the repo
    -- gating out names the others traded. The import was the only visible
    trace.
    """
    print(chr(10) + "no dead imports")
    import ast

    root = CONFIG_PATH.parent
    files = sorted((root / "bot").glob("*.py")) + [root / "run.py"]
    dead = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported[(a.asname or a.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    imported[a.asname or a.name] = node.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {n.value.id for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                used.update(n.value.replace("[", " ").replace("]", " ").split())
        for name, line in imported.items():
            if name != "annotations" and name not in used:
                dead.append(f"{path.name}:{line} {name}")
    check(f"no unused imports across {len(files)} files", not dead, str(dead))


def test_cli_wiring() -> None:
    """Every subcommand parses, has a handler, and only reads flags it declared.

    Eighteen commands and a lot of refactoring. The failure this catches is
    `args.something_that_no_longer_exists` -- an AttributeError that only
    surfaces when that particular command is run, which for the rarely-used
    ones can be weeks later.
    """
    print(chr(10) + "CLI wiring")
    import ast
    import importlib
    import io
    import contextlib

    run = importlib.import_module("run")
    src = CONFIG_PATH.parent / "run.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    # Which flags does each cmd_* function actually read off `args`?
    reads: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))                 and node.name.startswith("cmd_"):
            used = {
                sub.attr for sub in ast.walk(node)
                if isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name) and sub.value.id == "args"
            }
            reads[node.name[4:]] = used

    check("handlers were found", len(reads) >= 15, str(len(reads)))

    import argparse
    parser_holder = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parser_holder["p"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                run.main()
            except SystemExit:
                pass
    finally:
        argparse.ArgumentParser.parse_args = original

    parser = parser_holder.get("p")
    check("the parser was captured", parser is not None)
    if parser is None:
        return

    sub = next((a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)), None)
    check("subcommands exist", sub is not None)
    if sub is None:
        return

    names = sorted(sub.choices)
    check("every subcommand has a handler",
          all(n in reads for n in names),
          str([n for n in names if n not in reads]))

    missing: list[str] = []
    for name, cmd_parser in sub.choices.items():
        declared = set()
        for action in cmd_parser._actions:
            if action.dest and action.dest != "help":
                declared.add(action.dest)
        for flag in reads.get(name, set()):
            if flag not in declared:
                missing.append(f"{name}.args.{flag}")
    check("no handler reads a flag its parser never declared",
          not missing, str(missing))

    for name in names:
        ns = sub.choices[name].parse_args([])
        unset = [f for f in reads.get(name, set()) if not hasattr(ns, f)]
        check(f"{name}: all flags present after parsing", not unset, str(unset))


def test_daily_bars_are_not_intraday() -> None:
    """A weekend is not a session gap in a daily series.

    Applying the intraday rule to daily bars would call every Monday a session
    boundary: most signals refused and every exit window clipped to nothing.
    It would also be wrong on the substance -- with a daily bar you CAN hold
    overnight, because the next bar is the next tradable day.
    """
    print(chr(10) + "daily bars bypass the session logic")
    from bot.sessions import DAILY_MS, contiguous_back, forward_window, is_gap, is_intraday

    day = DAILY_MS
    check("1m is intraday", is_intraday(60_000))
    check("1h is intraday", is_intraday(3_600_000))
    check("4h is intraday", is_intraday(14_400_000))
    check("1d is not", not is_intraday(day))

    friday = candle(0, 100.0)
    monday = candle(3 * day, 101.0)          # three calendar days later
    check("a weekend is not a gap on daily bars", not is_gap(friday, monday, day))
    check("the same span IS a gap on 5m bars", is_gap(friday, monday, 300_000))

    week = [candle(i * day, 100.0 + i) for i in range(5)]
    week.append(candle(7 * day, 105.0))      # skips the weekend
    check("daily bars stay contiguous across a weekend",
          contiguous_back(week, 4, day))
    check("and the exit window is never clipped",
          len(forward_window(week, 0, 5, day)) == 5)

    from bot.config import load_config
    cfg = load_config(CONFIG_PATH)
    for tf, expected_ms in (("4h", 14_400_000), ("1d", 86_400_000)):
        cfg.strategy.timeframe = tf
        check(f"{tf} has a bar length", cfg.bar_ms == expected_ms, str(cfg.bar_ms))
        cfg.validate()


def test_sessions() -> None:
    """Session gaps, and the three things they break.

    Crypto never closes and the whole strategy assumed it. On an instrument
    that shuts for seventeen hours, a 3-bar return can silently span a night
    and a stop can sit unreachable until the next open.
    """
    print(chr(10) + "session handling")
    from bot.sessions import (contiguous_back, forward_window, is_gap,
                              session_index, session_stats)

    bar = 5 * MINUTE
    # Two sessions of 6 bars, separated by a 17-hour night.
    day1 = [candle(i * bar, 100.0 + i) for i in range(6)]
    night = 17 * 3600_000
    day2 = [candle(day1[-1].open_time + night + i * bar, 110.0 + i) for i in range(6)]
    both = day1 + day2

    check("a normal step is not a gap", not is_gap(day1[0], day1[1], bar))
    check("overnight is a gap", is_gap(day1[-1], day2[0], bar))
    check("sessions are numbered", session_index(both, bar) == [0] * 6 + [1] * 6,
          str(session_index(both, bar)))

    stats = session_stats(both, bar)
    check("session count", stats["sessions"] == 2, str(stats["sessions"]))
    check("bars per session", stats["bars_per_session"] == 6.0)
    check("the gap is measured in bars", stats["max_gap_bars"] > 200,
          f"{stats['max_gap_bars']:.0f}")

    # 1: a momentum window must not straddle the night.
    check("contiguous inside one session", contiguous_back(day1, 4, bar))
    check("NOT contiguous across the night",
          not contiguous_back(day1 + day2[:2], 4, bar))
    check("contiguous again once clear of it",
          contiguous_back(both[:6] + day2, 4, bar))
    check("too little history is not contiguous", not contiguous_back(day1[:2], 4, bar))

    # 2: a position cannot be carried through a close.
    win = forward_window(both, 3, horizon := 10, bar)
    check("forward window stops at the bell", len(win) == 2, str(len(win)))
    check("and contains only same-session bars",
          all(c.open_time < day2[0].open_time for c in win))
    check("a window inside the session is not clipped",
          len(forward_window(day2, 0, 3, bar)) == 3)
    check("no window at all on the last bar",
          forward_window(both, len(both) - 1, 5, bar) == [])


def test_asset_classification() -> None:
    """The suffix trap, and the market-hours clock."""
    print("\nasset classification")
    from bot.assets import EQUITIES, NOT_EQUITIES, classify, us_market_open

    for symbol, expected in (
        ("BTCUSDT", "crypto"), ("SOLUSDT", "crypto"),
        # Ends in B but is a coin. A suffix rule would call all four equities.
        ("BNBUSDT", "crypto"), ("ARBUSDT", "crypto"),
        ("TRBUSDT", "crypto"), ("SHIBUSDT", "crypto"),
        ("QQQBUSDT", "equity"), ("NVDABUSDT", "equity"), ("SPYBUSDT", "equity"),
        ("XAUTUSDT", "metal"), ("PAXGUSDT", "metal"),
    ):
        check(f"{symbol} is {expected}", classify(symbol) == expected,
              classify(symbol))

    check("the coin list and the equity list are disjoint",
          not (EQUITIES & NOT_EQUITIES), str(EQUITIES & NOT_EQUITIES))
    check("an unknown symbol defaults to crypto, not equity",
          classify("WHATEVERUSDT") == "crypto")

    import datetime as dt

    def at(day, hour):
        return int(dt.datetime(2026, 8, day, hour, tzinfo=dt.timezone.utc)
                   .timestamp() * 1000)

    check("weekday mid-session is open", us_market_open(at(12, 15)))
    check("weekday pre-dawn is closed", not us_market_open(at(12, 3)))
    check("weekday late evening is closed", not us_market_open(at(12, 22)))
    check("Saturday is closed at any hour", not us_market_open(at(15, 16)))
    check("Sunday is closed at any hour", not us_market_open(at(16, 16)))
    # Boundaries: open is inclusive, close exclusive.
    check("14:00 UTC counts as open", us_market_open(at(12, 14)))
    check("21:00 UTC counts as closed", not us_market_open(at(12, 21)))


def test_broker_protocol_matches() -> None:
    """The Protocol must describe the broker that actually exists.

    Nothing type-checks against a Protocol at runtime, so it rots silently:
    this one still declared buy() without `levels` long after exits became
    per-symbol. A docstring that lies is worse than no docstring.
    """
    print("\nBroker protocol is not a lie")
    import inspect

    from bot.broker import Broker, PaperBroker

    for name in ("buy", "sell", "mark_to_market"):
        declared = list(inspect.signature(getattr(Broker, name)).parameters)
        actual = list(inspect.signature(getattr(PaperBroker, name)).parameters)
        check(f"{name} signature matches", declared == actual,
              f"protocol {declared} vs broker {actual}")

    for attr in ("equity", "fees_paid"):
        check(f"{attr} exists on the broker",
              hasattr(PaperBroker(base_cfg()), attr))


def test_replication_pooling() -> None:
    """Windows must be non-overlapping by construction, and pooled correctly.

    The mistake this guards against: five overlapping slices of one period were
    reported as five replications, and the 'holdout' sat inside the window that
    supposedly confirmed it. Overlap has to be impossible to express, not
    merely discouraged.
    """
    print("\nreplication harness")
    from bot.nullcheck import Arm
    from bot.replicate import Window, _pool

    days = 30.0
    spans = [(k * days, k * days + days) for k in range(5)]
    overlaps = [
        (a, b) for i, a in enumerate(spans) for b in spans[i + 1:]
        if a[0] < b[1] and b[0] < a[1]
    ]
    check("windows never overlap", not overlaps, str(overlaps[:1]))
    check("windows are contiguous",
          all(abs(spans[i][1] - spans[i + 1][0]) < 1e-9 for i in range(len(spans) - 1)))

    def win(edge, n, sd=1.0):
        rng = random.Random(int(edge * 1000) + n)
        return Window("w", 0.0,
                      Arm("s", [rng.gauss(edge, sd) for _ in range(n)]),
                      Arm("r", [rng.gauss(0.0, sd) for _ in range(n)]))

    # A consistent effect pools to roughly its own size, with a tighter error.
    same = [win(0.10, 800) for _ in range(4)]
    mean, se, p = _pool(same)
    check("pooled mean lands near the true edge", abs(mean - 0.10) < 0.04,
          f"{mean:+.4f}")
    check("pooling tightens the error", se < min(w.stderr for w in same),
          f"{se:.4f} vs {min(w.stderr for w in same):.4f}")
    check("a consistent effect is detected", p < 0.01, f"p={p:.5f}")

    # One strong window among flat ones must NOT carry the pool on its own.
    lopsided = [win(0.40, 600)] + [win(0.0, 600) for _ in range(3)]
    mean_l, _, _ = _pool(lopsided)
    check("one lucky window does not dominate the pool", mean_l < 0.20,
          f"{mean_l:+.4f} from a 0.40 window plus three at zero")

    # A bigger window earns more weight, but only through its variance.
    weighted = _pool([win(0.20, 4000), win(0.0, 250)])[0]
    check("larger windows weigh more", weighted > 0.15, f"{weighted:+.4f}")
    check("empty input degrades safely", _pool([]) == (0.0, 0.0, 1.0))


def test_bootstrap() -> None:
    print("\nbootstrap confidence interval")
    from bot.nullcheck import Arm, bootstrap_ci

    rng = random.Random(23)
    a = Arm("a", [rng.gauss(0.5, 1) for _ in range(600)])
    b = Arm("b", [rng.gauss(0.0, 1) for _ in range(600)])
    lo, hi, p = bootstrap_ci(a, b, samples=800)
    # A percentile CI is centred on the OBSERVED difference, not the true one.
    observed = a.mean - b.mean
    check("CI brackets the observed difference", lo < observed < hi,
          f"[{lo:.3f}, {hi:.3f}] around {observed:.3f}")
    check("CI excludes zero for a real shift", lo > 0, f"lo={lo:.3f}")
    check("p is tiny for a real shift", p < 0.01, f"p={p:.4f}")

    # The property that matters is COVERAGE: over many experiments, the CI must
    # contain the truth about 95% of the time. Asserting on one draw is the
    # same mistake as asserting a single p-value lands above 0.05.
    covered = 0
    trials = 120
    for _ in range(trials):
        x = Arm("x", [rng.gauss(0.3, 1) for _ in range(250)])
        y = Arm("y", [rng.gauss(0.0, 1) for _ in range(250)])
        lo_i, hi_i, _ = bootstrap_ci(x, y, samples=400, seed=rng.randrange(10**6))
        if lo_i <= 0.3 <= hi_i:
            covered += 1
    rate = covered / trials
    check("95% CI covers the truth about 95% of the time",
          0.88 <= rate <= 0.99, f"{rate * 100:.0f}% over {trials} trials")

    # Truncated returns are exactly the shape the t-test is shaky on and the
    # bootstrap is not: a spike at each barrier, nothing in between.
    barrier_a = Arm("a", [(-1.2 if rng.random() < 0.53 else 2.4) for _ in range(800)])
    barrier_b = Arm("b", [(-1.2 if rng.random() < 0.62 else 2.4) for _ in range(800)])
    lo, hi, p = bootstrap_ci(barrier_a, barrier_b, samples=800)
    check("handles barrier-shaped returns", lo < (barrier_a.mean - barrier_b.mean) < hi,
          f"[{lo:.3f}, {hi:.3f}]")
    check("small arms degrade safely",
          bootstrap_ci(Arm("x", [1.0]), Arm("y", [2.0])) == (0.0, 0.0, 1.0))


def test_spread_band_gate() -> None:
    print("\nspread band gate")
    cfg = base_cfg()
    s = Series("TESTUSDT", 120)
    for c in flat_then_spike():
        s.update(c)

    cfg.strategy.min_spread_bps = 0.0
    check("no band: a 1 bps book passes",
          not isinstance(evaluate(s, cfg.strategy, 1.0), Rejection))

    cfg.strategy.min_spread_bps = 2.0
    cfg.strategy.max_spread_bps = 5.0
    check("below the band is rejected",
          getattr(evaluate(s, cfg.strategy, 1.0), "reason", "") == "spread_too_tight")
    check("inside the band fires",
          not isinstance(evaluate(s, cfg.strategy, 3.0), Rejection))
    check("above the band is rejected",
          getattr(evaluate(s, cfg.strategy, 9.0), "reason", "") == "spread_too_wide")

    d = diagnose(s, cfg.strategy, 1.0)
    check("diagnostics name the tight-spread block",
          d.blocking_gate == "spread_too_tight", d.blocking_gate)
    check("both spread gates are reported", "spread_too_tight" in d.gates
          and "spread_too_wide" in d.gates)


def test_simulate_exit_signature_is_strict() -> None:
    """No defaults on the per-symbol arguments. Ever.

    The ML labeller and the parameter sweep were each written calling
    simulate_exit without levels or slippage, and each silently scored a
    strategy with different stops and different costs than the bot runs. Both
    bugs were possible only because the arguments had defaults. Fixing them
    twice is not a fix; removing the default is. This test stops the default
    coming back.
    """
    print("\nsimulate_exit refuses global defaults")
    import inspect

    sig = inspect.signature(simulate_exit)
    for name in ("levels", "slippage_bps"):
        param = sig.parameters[name]
        check(f"{name} has no default", param.default is inspect.Parameter.empty,
              repr(param.default))

    cfg = base_cfg()
    check("calling without them is a TypeError",
          _raises(lambda: simulate_exit(100.0, [candle(0, 100.0)], cfg)))

    # And the values must actually reach the maths, not just be accepted.
    #
    # The exit type matters here. On a TAKE-PROFIT the target is a fixed
    # percentage of the entry and the exit is a maker fill, so entry slippage
    # cancels exactly: you enter higher and leave proportionally higher. That
    # is arithmetic, not a bug, and an earlier version of this test mistook it
    # for one. Use a time exit, where the exit price is the market and does not
    # scale with the entry.
    flat = [Candle(i * MINUTE, 100.0, 100.2, 99.9, 100.0, 1.0, 100.0, True)
            for i in range(1, cfg.max_hold_bars + 2)]
    cheap = simulate_exit(100.0, flat, cfg, fixed_levels(cfg), 0.0)
    dear = simulate_exit(100.0, flat, cfg, fixed_levels(cfg), 40.0)
    check("a time exit reflects entry slippage", cheap.reason == "max_hold"
          and cheap.net_pct > dear.net_pct,
          f"{cheap.reason}: {cheap.net_pct:+.3f}% vs {dear.net_pct:+.3f}%")

    # And a target exit is genuinely slippage-invariant, which is worth
    # asserting rather than merely tolerating.
    tp = [Candle(MINUTE, 100.0, 103.0, 99.5, 102.0, 1.0, 100.0, True)]
    a = simulate_exit(100.0, tp, cfg, fixed_levels(cfg), 0.0)
    b = simulate_exit(100.0, tp, cfg, fixed_levels(cfg), 40.0)
    check("a maker target exit cancels entry slippage",
          a.reason == "take_profit" and abs(a.net_pct - b.net_pct) < 1e-9,
          f"{a.net_pct:+.4f}% vs {b.net_pct:+.4f}%")

    from bot.levels import Levels
    # Long enough for BOTH to resolve: the tight target is hit on bar 1, the
    # wide one is never reached and times out. One-candle futures leave the
    # wide case unresolved (None), which is a real difference but a fragile
    # thing to assert on.
    long_future = tp + [
        Candle((i + 2) * MINUTE, 102.0, 102.1, 101.9, 102.0, 1.0, 100.0, True)
        for i in range(cfg.max_hold_bars)
    ]
    tight = Levels(stop_pct=0.5, target_pct=1.0, trail_pct=0.3, trail_activate_pct=0.5)
    wide = Levels(stop_pct=3.0, target_pct=6.0, trail_pct=2.0, trail_activate_pct=3.0)
    a = simulate_exit(100.0, long_future, cfg, tight, 0.0)
    b = simulate_exit(100.0, long_future, cfg, wide, 0.0)
    check("both resolve", a is not None and b is not None)
    check("levels change the exit reason", a.reason != b.reason,
          f"{a.reason} vs {b.reason}")
    check("levels change the result", a.net_pct != b.net_pct,
          f"{a.net_pct:+.3f}% vs {b.net_pct:+.3f}%")


def test_spread_estimator() -> None:
    print("\nspread estimator (Abdi-Ranaldo)")
    from bot.spread import abdi_ranaldo_bps, estimate_spreads

    for true_bps in (10.0, 25.0, 40.0):
        est = abdi_ranaldo_bps(_bounce_series(true_bps))
        # Recovering the magnitude is what matters: the point is to replace a
        # hand-picked 5 bps applied to every symbol, not to price to the tick.
        ok = 0.4 * true_bps <= est <= 2.5 * true_bps
        check(f"recovers a {true_bps:.0f} bps spread", ok, f"estimated {est:.2f} bps")

    ordered = [abdi_ranaldo_bps(_bounce_series(b)) for b in (10.0, 25.0, 40.0)]
    check("estimates are monotone in the true spread",
          ordered[0] < ordered[1] < ordered[2],
          " < ".join(f"{v:.2f}" for v in ordered))

    # The estimator has a resolution floor: a spread far below intrabar
    # volatility is unmeasurable. It must return 0 -- "cannot tell" -- rather
    # than a fabricated small number, because downstream that 0 becomes the
    # configured floor instead of a false precision.
    noisy = abdi_ranaldo_bps(_bounce_series(2.0, tick_vol=0.0004))
    check("unmeasurably tight spread returns 0, not a fiction", noisy == 0.0,
          f"{noisy:.3f}")
    # Same 2 bps spread, quiet book: now it is above the noise and recovered.
    quiet = abdi_ranaldo_bps(_bounce_series(2.0, tick_vol=0.00002))
    check("the same 2 bps IS recovered once the noise drops",
          0.8 <= quiet <= 5.0, f"{quiet:.3f} bps")

    check("too little data returns zero, not a guess",
          abdi_ranaldo_bps(_bounce_series(10.0, bars=5)) == 0.0)
    check("degenerate prices do not crash",
          abdi_ranaldo_bps([Candle(i * MINUTE, 1, 1, 1, 1, 1, 1, True)
                            for i in range(50)]) == 0.0)

    spreads = estimate_spreads({"A": _bounce_series(10.0), "B": _bounce_series(40.0)})
    check("per-symbol map built", spreads["A"] < spreads["B"],
          f"A={spreads['A']:.2f} B={spreads['B']:.2f}")
    # Unmeasurable must be OMITTED, not floored. Flooring at 0.5 bps asserted
    # the tightest book in the universe for exactly the symbols the estimator
    # failed on -- checked against live quotes, some were really 22 bps wide.
    check("unmeasurable symbols are omitted, never floored",
          "C" not in estimate_spreads({"C": []}))
    check("a below-noise symbol is omitted too",
          "D" not in estimate_spreads({"D": _bounce_series(2.0, tick_vol=0.0004)}))

    # The failure that motivated the significance gate. Returning 0 when the
    # expectation is negative is not enough: the expectation can come out
    # positive purely from noise, and sqrt() of noise is not a small number.
    # Against live bookTicker quotes one symbol read 2.06 bps where the book
    # was 0.08 -- 26 times too wide, indistinguishable from a real measurement.
    # Now the estimate must clear twice its own standard error to be believed.
    loud = [abdi_ranaldo_bps(_bounce_series(1.0, tick_vol=0.002, seed=s))
            for s in range(8)]
    check("a spread far under the noise is never reported",
          all(v == 0.0 for v in loud),
          f"{sum(1 for v in loud if v > 0)}/8 fabricated a number")

    # And the gate must not swallow spreads that genuinely are measurable.
    real = [abdi_ranaldo_bps(_bounce_series(30.0, seed=s)) for s in range(8)]
    kept = [v for v in real if v > 0]
    check("a genuinely wide spread still gets through", len(kept) >= 7,
          f"{len(kept)}/8 measured")
    check("and it is accurate when it does",
          all(20.0 <= v <= 45.0 for v in kept),
          f"{min(kept):.1f}-{max(kept):.1f} bps for a true 30")


def test_cost_model() -> None:
    print("\ncost model")
    from bot.costs import CostModel

    cfg = base_cfg()
    flat = CostModel(cfg)
    check("fixed mode ignores the symbol",
          flat.slippage_bps("ANY") == cfg.fees.slippage_bps)

    cfg.fees.slippage_mode = "estimated"
    cfg.fees.impact_multiplier = 1.5
    model = CostModel(cfg, {"TIGHT": 1.0, "WIDE": 40.0})
    check("pays half the spread times impact",
          abs(model.slippage_bps("TIGHT") - 0.75) < 1e-9,
          f"{model.slippage_bps('TIGHT'):.3f} bps")
    check("wide book costs much more",
          model.slippage_bps("WIDE") == 30.0, f"{model.slippage_bps('WIDE')}")
    check("unknown symbol falls back to the configured value",
          model.slippage_bps("UNSEEN") == cfg.fees.slippage_bps)

    tight = model.round_trip_pct("TIGHT")
    wide = model.round_trip_pct("WIDE")
    check("round trip on a tight book is far cheaper", tight < wide,
          f"{tight:.3f}% vs {wide:.3f}%")
    check("fee legs still dominate a tight book",
          abs(tight - (2 * (cfg.fees.effective_taker_bps + 0.75) / 100)) < 1e-9,
          f"{tight:.4f}%")

    # A broker built on the model must actually charge those numbers.
    broker = PaperBroker(cfg, costs=model)
    pos = broker.buy("TIGHT", 200.0, 100.0, 0, fixed_levels(cfg))
    check("tight symbol fills close to the signal price",
          abs(pos.entry_price - 100.0) < 0.01, f"{pos.entry_price:.5f}")
    wide_pos = broker.buy("WIDE", 200.0, 100.0, 0, fixed_levels(cfg))
    check("wide symbol fills materially worse",
          wide_pos.entry_price > pos.entry_price, f"{wide_pos.entry_price:.5f}")

    # Symbols the estimator could not price must NOT be assumed cheap.
    # Fallback order: measured estimate -> exchange-quoted -> flat. Reconciling
    # a live session showed the flat value gating out symbols the book quoted
    # at 2.9 bps, so ground truth has to come before the constant.
    quoted = CostModel(cfg, {"TIGHT": 1.0}, {"TIGHT": 99.0, "QUOTED": 4.0})
    check("a measured estimate beats the quote",
          quoted.spread_bps("TIGHT") == 1.0, str(quoted.spread_bps("TIGHT")))
    check("a quote beats the flat constant",
          quoted.spread_bps("QUOTED") == 4.0, str(quoted.spread_bps("QUOTED")))
    check("quoted symbols are not reported unpriced",
          quoted.unpriced(["TIGHT", "QUOTED", "NEITHER"]) == ["NEITHER"],
          str(quoted.unpriced(["TIGHT", "QUOTED", "NEITHER"])))
    check("a quoted symbol passes the gate the flat value would fail",
          quoted.spread_bps("QUOTED") <= cfg.strategy.max_spread_bps
          < cfg.fees.slippage_bps * 2)

    check("unpriced symbols get the pessimistic flat value",
          model.slippage_bps("NEVERSEEN") == cfg.fees.slippage_bps)
    check("unpriced symbols are reported",
          model.unpriced(["TIGHT", "WIDE", "NEVERSEEN"]) == ["NEVERSEEN"],
          str(model.unpriced(["TIGHT", "WIDE", "NEVERSEEN"])))
    check("nothing is unpriced in fixed mode",
          CostModel(base_cfg()).unpriced(["A", "B"]) == [])
    # At the flat fallback the entry gate rejects them, which is the intended
    # "do not trade what you cannot price" -- assert it rather than assume it.
    check("the fallback exceeds the entry gate, so they are gated out",
          model.spread_bps("NEVERSEEN") > cfg.strategy.max_spread_bps,
          f"{model.spread_bps('NEVERSEEN')} vs {cfg.strategy.max_spread_bps}")

    cfg.fees.slippage_mode = "nonsense"
    check("bad slippage mode refused", _raises(cfg.validate))


def test_nullcheck_stats() -> None:
    print("\nnull test statistics")
    from bot.nullcheck import Arm, welch

    a = Arm("a", [1.0, 1.0, 1.0, 1.0])
    check("mean", a.mean == 1.0)
    check("win rate", a.win_rate == 100.0)
    check("zero variance", a.stdev == 0.0)

    rng = random.Random(4)

    # A single pair of identical samples crosses p<0.05 about 5% of the time --
    # that is what p means, so asserting on one draw would be a flaky test.
    # Assert on the false-positive RATE instead, which is what must hold.
    false_positives = 0
    trials = 300
    for _ in range(trials):
        a1 = Arm("a", [rng.gauss(0, 1) for _ in range(400)])
        a2 = Arm("b", [rng.gauss(0, 1) for _ in range(400)])
        if welch(a1, a2)[1] < 0.05:
            false_positives += 1
    rate = false_positives / trials
    check("false-positive rate is near the nominal 5%", 0.01 <= rate <= 0.11,
          f"{rate * 100:.1f}% over {trials} trials")

    # Unequal variances are exactly where a pooled t-test misbehaves; Welch
    # is used precisely because the two arms have very different spreads.
    uneven = 0
    for _ in range(200):
        a1 = Arm("a", [rng.gauss(0, 1) for _ in range(120)])
        a2 = Arm("b", [rng.gauss(0, 6) for _ in range(600)])
        if welch(a1, a2)[1] < 0.05:
            uneven += 1
    check("still calibrated with unequal variance and size",
          0.01 <= uneven / 200 <= 0.12, f"{uneven / 2:.1f}%")

    baseline = Arm("a", [rng.gauss(0.0, 1) for _ in range(800)])
    shifted = Arm("b", [rng.gauss(0.5, 1) for _ in range(800)])
    t_diff, p_diff = welch(shifted, baseline)
    check("a real shift is detected", p_diff < 0.01, f"p={p_diff:.5f}")
    check("t sign follows the direction", t_diff > 0, f"t={t_diff:.2f}")
    check("empty arms degrade safely", welch(Arm("x", []), Arm("y", [])) == (0.0, 1.0))


def test_exitfit() -> None:
    print("\nexit fit")
    from bot.exitfit import HOLDS, STOPS, TARGETS, Path as XPath, _excursions, _pct, _score_exit

    entry = 100.0
    up = [Candle(i * MINUTE, entry, entry * (1 + 0.002 * (i + 1)),
                 entry * (1 - 0.0005 * (i + 1)), entry, 1.0, 100.0, True)
          for i in range(30)]
    path = XPath("TESTUSDT", 0, entry, up, 1.0)

    mfe10, mae10 = _excursions(path, 10)
    mfe30, mae30 = _excursions(path, 30)
    check("MFE grows with the horizon", mfe30 > mfe10, f"{mfe10:.2f} -> {mfe30:.2f}")
    check("MAE is negative and deepens", mae30 < mae10 <= 0,
          f"{mae10:.2f} -> {mae30:.2f}")
    check("MFE matches the constructed high",
          abs(mfe30 - 6.0) < 0.01, f"{mfe30:.3f}")

    check("percentiles are ordered",
          _pct([5, 1, 3, 2, 4], 0.0) <= _pct([5, 1, 3, 2, 4], 0.5)
          <= _pct([5, 1, 3, 2, 4], 0.99))
    check("empty percentile is zero, not a crash", _pct([], 0.5) == 0.0)

    cfg = base_cfg()
    near = _score_exit(cfg, [path], stop=1.2, target=1.0, hold=30)
    far = _score_exit(cfg, [path], stop=1.2, target=5.0, hold=30)
    check("a reachable target is taken", near.trades == 1 and near.gross > 0,
          f"gross {near.gross:+.3f}%")
    check("an unreachable target is not", far.gross < near.gross,
          f"{far.gross:+.3f}% vs {near.gross:+.3f}%")
    check("net is gross minus fees",
          abs((near.gross - near.net) - 2 * cfg.fees.effective_taker_bps / 100) < 1e-9,
          f"{near.gross - near.net:.4f}%")

    # Targets below the stop are deliberately IN the grid: requiring R:R > 1
    # assumed the shape of the answer and hid the reachable short targets.
    check("grid includes sub-1 risk:reward",
          any(t < s for s in STOPS for t in TARGETS), "")
    check("grid covers the shipped setting",
          cfg.risk.stop_loss_pct in STOPS and cfg.risk.take_profit_pct in TARGETS
          and cfg.risk.max_hold_minutes in HOLDS)


def test_sweep_uses_per_symbol_levels() -> None:
    """The sweep had the same label-drift bug the ML labeller did."""
    print("\nsweep scores what the bot would trade")
    from bot.sweep import Candidate, score

    cfg = base_cfg()
    cfg.risk.stop_mode = "atr"
    rng = random.Random(41)
    future = []
    price = 100.0
    for i in range(60):
        price *= 1.0 + rng.gauss(0.0006, 0.004)
        w = abs(rng.gauss(0, 0.003))
        future.append(Candle(i * MINUTE, price, price * (1 + w), price * (1 - w),
                             price, 1.0, 100.0, True))

    params = {"min_return_zscore": 0.0, "min_volume_mult": 0.0,
              "take_profit_pct": 2.4, "stop_loss_pct": 1.2, "max_hold_minutes": 30}
    quiet = Candidate("A", 0, 100.0, 1.0, 3.0, 5.0, 1.0, future, atr_pct=0.05,
                      slippage_bps=1.0)
    wild = Candidate("B", 0, 100.0, 1.0, 3.0, 5.0, 1.0, future, atr_pct=1.20,
                     slippage_bps=1.0)
    check("ATR reaches the scorer: different vol, different outcome",
          score(cfg, [quiet], params).expectancy != score(cfg, [wild], params).expectancy,
          f"{score(cfg, [quiet], params).expectancy:.4f} vs "
          f"{score(cfg, [wild], params).expectancy:.4f}")

    cheap = Candidate("A", 0, 100.0, 1.0, 3.0, 5.0, 1.0, future, atr_pct=0.5,
                      slippage_bps=0.5)
    pricey = Candidate("A", 0, 100.0, 1.0, 3.0, 5.0, 1.0, future, atr_pct=0.5,
                       slippage_bps=40.0)
    check("per-symbol slippage reaches the scorer",
          score(cfg, [cheap], params).expectancy > score(cfg, [pricey], params).expectancy,
          f"{score(cfg, [cheap], params).expectancy:.4f} vs "
          f"{score(cfg, [pricey], params).expectancy:.4f}")


def test_sweep_scoring() -> None:
    print("\nsweep scoring")
    from bot.sweep import GRID, Candidate, _configs, _folds, score

    cfg = base_cfg()
    configs = _configs()
    expected = 1
    for values in GRID.values():
        expected *= len(values)
    check("grid fully enumerated", len(configs) == expected,
          f"{len(configs)} vs {expected}")
    check("configs are unique",
          len({tuple(sorted(c.items())) for c in configs}) == len(configs))

    rng = random.Random(13)
    candidates = []
    for i in range(200):
        price = 100.0
        future = []
        for j in range(60):
            price *= 1.0 + rng.gauss(0.0005, 0.005)
            w = abs(rng.gauss(0, 0.003))
            future.append(Candle(j * MINUTE, price, price * (1 + w),
                                 price * (1 - w), price, 100.0, 100.0 * price, True))
        candidates.append(
            Candidate("TESTUSDT", i * MINUTE, 100.0, 1.0,
                      rng.uniform(2.0, 6.0), rng.uniform(2.0, 10.0), 1.0, future)
        )

    loose = score(cfg, candidates, {"min_return_zscore": 2.0, "min_volume_mult": 2.0,
                                    "take_profit_pct": 2.4, "stop_loss_pct": 1.2,
                                    "max_hold_minutes": 30})
    tight = score(cfg, candidates, {"min_return_zscore": 5.0, "min_volume_mult": 8.0,
                                    "take_profit_pct": 2.4, "stop_loss_pct": 1.2,
                                    "max_hold_minutes": 30})
    check("looser config takes more trades", loose.trades > tight.trades,
          f"{loose.trades} vs {tight.trades}")
    check("expectancy is the mean of the trades",
          abs(loose.expectancy - loose.total / loose.trades) < 1e-9)
    check("scoring does not mutate the caller's config",
          cfg.strategy.min_return_zscore == base_cfg().strategy.min_return_zscore)

    chunks = _folds(candidates, 4)
    check("folds partition every candidate",
          sum(len(c) for c in chunks) == len(candidates))
    check("folds are time-ordered",
          all(not c or not n or c[-1].time_ms <= n[0].time_ms
              for c, n in zip(chunks, chunks[1:])))


def test_ladder() -> None:
    """The posted ladder does what the post says, and only once per excursion.

    The bug this is really guarding against is the one the rules invite: read
    literally, "drops 20% -> buy" fires on every single day the market is 20%
    down, which spends the whole cash pile in the first week of a bear market
    and then reports a spectacular recovery. The re-arm rule is the difference
    between testing the strategy and testing a leveraged bottom-tick.
    """
    print(chr(10) + "the 105-year-old's ladder")
    from bot.ladder import (COST_ERAS, FlatYields, MeasuredYields, cost_bps,
                            simulate_hold, simulate_ladder, split_windows)

    NONE = FlatYields(0.0, 0.0)

    day = 86_400_000
    base = 1_577_836_800_000            # 2020-01-01, so the modern cost era

    def days(prices: list[float]) -> list[Candle]:
        return [candle(base + i * day, p) for i, p in enumerate(prices)]

    check("friction is era-aware", cost_bps(base) == COST_ERAS[-1][1])
    check("and much larger before May Day 1975",
          cost_bps(-100 * day) == 75.0, f"{cost_bps(-100 * day)}")

    # A -25% market, held for ten days, is ONE buy and not ten.
    once = simulate_ladder(days([100.0] * 5 + [75.0] * 10), "anchor", "initial",
                           start_equity=0.5, yields=NONE)
    check("the -20% rung fires once per excursion", once.buys == 1,
          f"buys={once.buys}")
    check("and the -30% rung stays armed at -25%", once.sells == 0)

    # Back above the reference and down again is a second excursion.
    twice = simulate_ladder(days([100.0] * 3 + [75.0] * 3 + [105.0] * 3
                                 + [75.0] * 3), "anchor", "initial",
                            start_equity=0.5, yields=NONE)
    check("crossing back through the reference re-arms it", twice.buys == 2,
          f"buys={twice.buys}")

    # One -35% day trips both rungs, deepest first.
    both = simulate_ladder(days([100.0] * 3 + [65.0] * 3), "anchor", "initial",
                           start_equity=0.5, yields=NONE)
    check("a -35% day fills both rungs", both.buys == 2, f"buys={both.buys}")
    check("without borrowing", both.cash_end >= -1e-9, f"{both.cash_end:.6f}")

    # 10% cash cannot fund a rule that asks for 45% of capital.
    broke = simulate_ladder(days([100.0] * 3 + [65.0] * 3), "anchor", "initial",
                            start_equity=0.9, yields=NONE)
    check("an underfunded rung is counted, not silently skipped",
          broke.unfunded >= 1, f"unfunded={broke.unfunded}")
    check("and it still cannot borrow", broke.cash_end >= -1e-9)

    # A market that triples trips all five sell rungs.
    ramp = days([100.0 * 1.01 ** i for i in range(130)])
    out = simulate_ladder(ramp, "basis", "position", start_equity=1.0,
                          yields=NONE)
    check("every sell rung fires on a tripling market",
          out.sells == 5, f"sells={out.sells}")
    check("scaling out leaves the ladder behind buy-and-hold there",
          out.cagr < simulate_hold(ramp, 1.0, yields=NONE).cagr)

    # Buy-and-hold with no yields is exactly the price ratio net of one entry.
    hold = simulate_hold(days([100.0, 150.0, 200.0]), 1.0, 10_000.0, NONE)
    want = 10_000.0 * (1.0 - COST_ERAS[-1][1] / 10_000.0) * 2.0
    check("hold is the price ratio net of one entry cost",
          abs(hold.final - want) < 1e-6, f"{hold.final:.4f} vs {want:.4f}")

    # Yields move the answer, which is why they are printed as assumptions.
    with_yield = simulate_hold(days([100.0] * 400), 1.0, 10_000.0,
                               FlatYields(5.0, 0.0))
    check("dividends accrue on a flat market", with_yield.final > 10_000.0)

    chunks = split_windows(days([100.0 + i * 0.01 for i in range(3000)]), 2.0)
    check("windows are non-overlapping and ordered",
          all(a[-1].open_time < b[0].open_time
              for a, b in zip(chunks, chunks[1:])), f"{len(chunks)} windows")
    check("a century of daily bars yields several windows", len(chunks) >= 3,
          f"{len(chunks)}")

    for mode in ("basis", "peak", "anchor"):
        for sizing in ("initial", "position"):
            run = simulate_ladder(ramp, mode, sizing, 0.5)
            check(f"{mode}/{sizing} stays solvent",
                  run.final > 0 and run.cash_end >= -1e-9)

    # The two rules I invented, not the post's. `once` must never re-fire; the
    # literal `always` must fire on every bar the condition holds.
    two_dips = days([100.0] * 3 + [75.0] * 3 + [105.0] * 3 + [75.0] * 3)
    once = simulate_ladder(two_dips, "anchor", "initial", start_equity=0.5,
                           yields=NONE, rearm="once")
    check("rearm=once fires a rung a single time per window", once.buys == 1,
          f"buys={once.buys}")
    always = simulate_ladder(two_dips, "anchor", "initial", start_equity=0.5,
                             yields=NONE, rearm="always")
    check("rearm=always fires far more often than the default",
          always.buys > twice.buys, f"{always.buys} vs {twice.buys}")
    check("and stops only because the cash ran out, which is counted",
          always.unfunded >= 1 and always.cash_end >= -1e-9,
          f"unfunded={always.unfunded}")

    # A bar whose LOW breaks the level but whose close does not: invisible to
    # close-triggering, a fill for intrabar triggering.
    wick = [candle(base, 100.0), candle(base + day, 100.0),
            Candle(base + 2 * day, 100.0, 101.0, 75.0, 99.0, 1.0, 99.0, True),
            candle(base + 3 * day, 99.0)]
    on_close = simulate_ladder(wick, "anchor", "initial", start_equity=0.5,
                               yields=NONE)
    intrabar = simulate_ladder(wick, "anchor", "initial", start_equity=0.5,
                               yields=NONE, trigger="intrabar")
    check("a wick through the level is invisible to close triggering",
          on_close.buys == 0, f"buys={on_close.buys}")
    check("and is a fill for intrabar triggering -- the -20% rung only, "
          "since the low never reached -30%", intrabar.buys == 1,
          f"buys={intrabar.buys}")

    # A bar that gapped straight through the level never traded at it.
    gap = [candle(base, 100.0), candle(base + day, 100.0),
           Candle(base + 2 * day, 60.0, 62.0, 55.0, 61.0, 1.0, 61.0, True),
           candle(base + 3 * day, 61.0)]
    gapped = simulate_ladder(gap, "anchor", "initial", start_equity=0.5,
                             yields=NONE, trigger="intrabar")
    check("a gap fills at the open, not at the untraded level",
          gapped.buys == 2 and gapped.cash_end >= -1e-9,
          f"buys={gapped.buys}")

    for bad in (("rearm", {"rearm": "sometimes"}), ("trigger", {"trigger": "hlc3"})):
        try:
            simulate_ladder(days([100.0] * 5), "anchor", "initial", **bad[1])
            check(f"an unknown {bad[0]} is rejected", False)
        except ValueError:
            check(f"an unknown {bad[0]} is rejected", True)

    # The symbol presets: the collapsed list exists to stop the conclusion
    # resting on four markets that happened to go up.
    from bot.ladder import COLLAPSED_SYMBOLS, DEFAULT_SYMBOLS, SYMBOL_PRESETS

    check("both presets exist and are non-empty",
          set(SYMBOL_PRESETS) == {"default", "collapsed"}
          and all(SYMBOL_PRESETS.values()))
    check("the collapsed list is not just the default one again",
          set(COLLAPSED_SYMBOLS) - set(DEFAULT_SYMBOLS),
          str(sorted(set(COLLAPSED_SYMBOLS) - set(DEFAULT_SYMBOLS))))

    # Tax: the term that was missing, and which runs against the ladder.
    from bot.ladder import CGT_ERAS, Taxes

    check("a flat rate is the same in every year",
          Taxes(15.0).rate_at(base) == Taxes(15.0).rate_at(-100 * day) == 15.0)
    check("the era table is era-aware",
          Taxes(None).rate_at(base) == CGT_ERAS[-1][1]
          and Taxes(None).rate_at(-100 * day) == 25.0,
          f"{Taxes(None).rate_at(-100 * day)}")

    # A market that doubles, sold down by the rungs, pays tax on real gains.
    doubling = days([100.0 * 1.01 ** i for i in range(120)])
    free = simulate_ladder(doubling, "basis", "position", start_equity=1.0,
                           yields=NONE)
    taxed = simulate_ladder(doubling, "basis", "position", start_equity=1.0,
                            yields=NONE, taxes=Taxes(25.0))
    check("selling into a gain costs tax", taxed.tax_paid > 0,
          f"{taxed.tax_paid:.2f}")
    check("and the taxed run ends behind the untaxed one",
          taxed.final < free.final, f"{taxed.final:.2f} vs {free.final:.2f}")
    check("the bill is a quarter of the realised gain, not of the proceeds",
          taxed.tax_paid < free.final - free.initial,
          f"{taxed.tax_paid:.2f}")

    # A market that only falls realises nothing to tax.
    falling = days([100.0 * 0.99 ** i for i in range(120)])
    loser = simulate_ladder(falling, "basis", "position", start_equity=0.5,
                            yields=NONE, taxes=Taxes(25.0))
    check("a loss is not taxed", loser.tax_paid == 0.0, f"{loser.tax_paid}")

    # Buy-and-hold pays nothing until it settles, and then pays on everything.
    never = simulate_hold(doubling, 1.0, 10_000.0, NONE, Taxes(25.0))
    settled = simulate_hold(doubling, 1.0, 10_000.0, NONE, Taxes(25.0, True))
    check("a holder who never sells never pays", never.tax_paid == 0.0)
    check("settling at the end taxes the deferred gain",
          settled.tax_paid > 0 and settled.final < never.final,
          f"{settled.tax_paid:.2f}")
    check("and settling does not invent a sale in the trade count",
          settled.sells == never.sells == 0)

    # Dividends are income to both arms, so leaving them untaxed would have
    # handed buy-and-hold the larger untaxed stream.
    flat_market = days([100.0] * 400)
    gross = simulate_hold(flat_market, 1.0, 10_000.0, FlatYields(5.0, 0.0))
    net = simulate_hold(flat_market, 1.0, 10_000.0, FlatYields(5.0, 0.0),
                        Taxes(25.0))
    check("dividends are taxed as they are paid", net.final < gross.final
          and net.tax_paid > 0, f"{net.tax_paid:.2f}")

    # Measured rates replace the two constants that could rig the comparison.
    day0 = 20_000
    bills = {day0: 5.0, day0 + 10: 1.0}
    div = {day0 + i: 1.02 ** (1 / 365.25) for i in range(366)}
    measured = MeasuredYields(bills, div, clamped=7)
    one_day = (day0 + 5) * day, (day0 + 6) * day
    check("the bill rate is forward-filled between prints",
          abs(measured.cash_factor(*one_day) - 1.05 ** (1 / 365.25)) < 1e-12,
          f"{measured.cash_factor(*one_day):.9f}")
    check("and steps down once the next print exists",
          abs(measured.cash_factor((day0 + 20) * day, (day0 + 21) * day)
              - 1.01 ** (1 / 365.25)) < 1e-12)
    check("a day before the first print earns nothing rather than guessing",
          measured.cash_factor((day0 - 5) * day, (day0 - 4) * day) == 1.0)
    check("a day with no dividend record contributes nothing",
          measured.dividend_factor((day0 + 900) * day, (day0 + 901) * day) == 1.0)
    check("the annualised dividend recovers the rate it was built from",
          abs(measured.mean_dividend_pct() - 2.0) < 0.05,
          f"{measured.mean_dividend_pct():.4f}%")
    check("the covered span is the overlap of both series",
          measured.start_ms == day0 * day
          and measured.end_ms == (day0 + 10) * day)
    check("dropped glitch days are carried, not hidden",
          "measured" in measured.describe() and measured.clamped == 7,
          measured.describe())

    try:
        simulate_ladder(days([100.0] * 5), "highwater", "initial")
        check("an unknown reference mode is rejected", False)
    except ValueError:
        check("an unknown reference mode is rejected", True)


def test_ladder_invariants() -> None:
    """Properties the ladder must satisfy on ANY price path, not just my fixtures.

    Every hand-built fixture tests the case I thought of. Seven of the bugs in
    this project were cases I had not, and the ones that survived longest were
    the ones that produced a plausible number. These check relationships that
    must hold whatever the prices do, over random walks, so a wrong answer has
    to be wrong consistently to slip through.

    The first is the strongest: with no rungs at all, the entire ladder machine
    -- references, re-arming, sizing, costs, tax, accrual, terminal settlement
    -- must reduce EXACTLY to `simulate_hold`, a function that shares none of
    that code. Any drift between the two arms of the comparison shows up here
    as a mismatch to the cent, and a drift between the arms is precisely the
    error that would manufacture the finding.
    """
    print(chr(10) + "ladder invariants")
    import bot.ladder as L

    day = 86_400_000
    base = 1_577_836_800_000
    rng = random.Random(20260814)

    def walk(bars: int, drift: float = 0.0002, vol: float = 0.012) -> list[Candle]:
        out, price = [], 100.0
        for i in range(bars):
            price *= 1.0 + rng.gauss(drift, vol)
            price = max(price, 0.01)
            w = abs(rng.gauss(0, vol / 2))
            out.append(Candle(base + i * day, price, price * (1 + w),
                              price * (1 - w), price, 1.0, price, True))
        return out

    paths = [walk(900) for _ in range(6)]
    none_yield = L.FlatYields(0.0, 0.0)
    some_yield = L.FlatYields(2.0, 3.0)

    # 1. A ladder with no rungs IS buy-and-hold, to the cent.
    buy, sell = L.BUY_RUNGS, L.SELL_RUNGS
    worst = 0.0
    try:
        L.BUY_RUNGS, L.SELL_RUNGS = (), ()
        for path in paths:
            for equity in (0.4, 1.0):
                for taxes in (L.NO_TAX, L.Taxes(23.8, True)):
                    a = L.simulate_ladder(path, "peak", "position", equity,
                                          10_000.0, some_yield, taxes=taxes)
                    b = L.simulate_hold(path, equity, 10_000.0, some_yield, taxes)
                    worst = max(worst, abs(a.final - b.final))
    finally:
        L.BUY_RUNGS, L.SELL_RUNGS = buy, sell
    check("a ladder with no rungs reduces exactly to buy-and-hold",
          worst < 1e-6, f"worst mismatch {worst:.10f}")

    # 2. Tax can only ever take money away, never add it.
    ok = True
    for path in paths:
        finals = [L.simulate_ladder(path, "basis", "position", 0.5, 10_000.0,
                                    some_yield, taxes=L.Taxes(rate)).final
                  for rate in (0.0, 15.0, 30.0)]
        ok = ok and finals[0] >= finals[1] >= finals[2] - 1e-9
    check("a higher tax rate never ends richer", ok)

    # 3. Re-arming more often can only fire more often.
    ordered = True
    for path in paths:
        counts = {}
        for mode in ("once", "cross", "always"):
            run = L.simulate_ladder(path, "peak", "initial", 0.5, 10_000.0,
                                    none_yield, rearm=mode)
            counts[mode] = run.buys + run.sells
        ordered = ordered and counts["once"] <= counts["cross"] <= counts["always"]
    check("trade counts are ordered once <= cross <= always", ordered)

    # 4. No borrowing and no shorting, on any path, in any configuration.
    solvent = True
    for path in paths:
        for ref in L.REF_MODES:
            for sizing in L.SIZINGS:
                for mode in L.REARM_MODES:
                    run = L.simulate_ladder(path, ref, sizing, 0.5, 10_000.0,
                                            some_yield, rearm=mode,
                                            taxes=L.Taxes(23.8))
                    solvent = solvent and run.cash_end >= -1e-9 and run.final > 0
    check("never borrows and never goes short, over 108 configurations",
          solvent)

    # 5. Same inputs, same answer. A stray global would show up here.
    twice = [L.simulate_ladder(paths[0], "peak", "initial", 0.5, 10_000.0,
                               some_yield, taxes=L.Taxes(None)).final
             for _ in range(2)]
    check("the simulation is deterministic", twice[0] == twice[1])

    # 6. Intrabar triggering sees everything close triggering sees.
    sees_more = True
    for path in paths:
        a = L.simulate_ladder(path, "peak", "initial", 0.5, 10_000.0,
                              none_yield, trigger="close")
        b = L.simulate_ladder(path, "peak", "initial", 0.5, 10_000.0,
                              none_yield, trigger="intrabar")
        sees_more = sees_more and (b.buys + b.sells) >= (a.buys + a.sells)
    check("intrabar triggering never fires less than close triggering",
          sees_more)

    # 7. Windows partition the history without overlapping or inventing bars.
    chunks = L.split_windows(paths[0], 1.0)
    source = {c.open_time for c in paths[0]}
    inside = all(c.open_time in source for chunk in chunks for c in chunk)
    check("every windowed bar came from the original series", inside)
    check("and no bar appears in two windows",
          len({c.open_time for chunk in chunks for c in chunk})
          == sum(len(chunk) for chunk in chunks))


def test_longterm() -> None:
    """The long-horizon rules, and the look-ahead each one invites.

    Both candidates have an obvious way to cheat and both were nearly written
    that way: a trend rule that decides and trades on the same close, and a
    valuation percentile ranked against the whole history including its own
    future. Neither is visible in the output -- they just make the strategy
    look good -- so they are pinned here instead.
    """
    print(chr(10) + "the long-horizon bot")
    from bot.ladder import FlatYields, Taxes
    from bot.longterm import (CAPE_MIN_HISTORY, SMA_DAYS, cape_weight,
                              simulate, sma_weight)

    day = 86_400_000
    base = 1_577_836_800_000
    none_yield = FlatYields(0.0, 0.0)

    # The trend rule: above its own average or below it, and nothing else.
    rising = [100.0 + i for i in range(SMA_DAYS + 10)]
    falling = [100.0 + (SMA_DAYS + 10 - i) for i in range(SMA_DAYS + 10)]
    check("above the average means fully invested",
          sma_weight(rising).weight == 1.0)
    check("below it means fully out", sma_weight(falling).weight == 0.0)
    check("a short history stays invested rather than guessing",
          sma_weight([100.0] * 10).weight == 1.0,
          sma_weight([100.0] * 10).reason)
    check("the average uses exactly the last N closes",
          sma_weight([1.0] * SMA_DAYS + [2.0], SMA_DAYS).weight == 1.0)

    # The valuation rule: a percentile against the past, never the whole series.
    history = [float(i) for i in range(CAPE_MIN_HISTORY)]
    check("the cheapest reading ever is fully invested",
          cape_weight(-1.0 + min(history), history).weight == 1.0)
    check("the most expensive reading ever is fully out",
          cape_weight(max(history) + 1.0, history).weight == 0.0)
    mid = cape_weight(history[len(history) // 2], history).weight
    check("the middle is roughly half", abs(mid - 0.5) < 0.02, f"{mid:.3f}")
    check("too little history means invested, not a coin flip",
          cape_weight(50.0, history[:10]).weight == 1.0)
    check("a missing CAPE is not read as the cheapest market in history",
          cape_weight(0.0, history).weight == 1.0)

    # No look-ahead: a rule cannot see the bar it trades on. The series rises
    # gently so the rule is genuinely invested -- a FLAT series would sit at
    # price == average, and the strict comparison would send it to cash before
    # the crash, which is how the first version of this test passed while
    # measuring nothing.
    # The decline is gradual on purpose. A one-bar 90% gap cannot be dodged by
    # any rule that reads closes, and asserting otherwise would be asserting
    # that the backtest cheats.
    crash = [candle(base + i * day, 100.0 + i * 0.05) for i in range(400)]
    peak = 100.0 + 399 * 0.05
    crash += [candle(base + (400 + i) * day, peak * 0.99 ** (i + 1))
              for i in range(300)]
    run = simulate(crash, "sma", capital=10_000.0, yields=none_yield)
    hold_through = simulate(crash, "hold", capital=10_000.0, yields=none_yield)
    check("the trend rule takes the part of the fall it could not see coming",
          run.final < 10_000.0, f"{run.final:.2f}")
    check("but it gets out early and beats holding through the rest",
          run.final > hold_through.final * 5,
          f"{run.final:.2f} vs {hold_through.final:.2f}")

    # `hold` through this machine must equal a fixed 100% target exactly: the
    # strategy path and the baseline path are the same code or the comparison
    # is measuring the difference between two implementations.
    a = simulate(crash, "hold", capital=10_000.0, yields=none_yield)
    b = simulate(crash, "hold", capital=10_000.0, yields=none_yield,
                 fixed_weight=1.0)
    check("the hold strategy equals a fixed 100% weight to the cent",
          abs(a.final - b.final) < 1e-9, f"{a.final:.6f} vs {b.final:.6f}")

    # The rebalance band must actually suppress trading.
    wiggle = [candle(base + i * day, 100.0 + (i % 5)) for i in range(900)]
    tight = simulate(wiggle, "hold", capital=10_000.0,
                     yields=FlatYields(5.0, 5.0), band=0.0)
    loose = simulate(wiggle, "hold", capital=10_000.0,
                     yields=FlatYields(5.0, 5.0), band=0.5)
    check("a wide band trades less than a zero band",
          loose.sells < tight.sells, f"{loose.sells} vs {tight.sells}")

    # Tax and costs reach this engine too, not just the ladder's.
    taxed = simulate(wiggle, "hold", capital=10_000.0,
                     yields=FlatYields(5.0, 5.0), taxes=Taxes(25.0))
    check("dividends are taxed in the long-horizon engine too",
          taxed.tax_paid > 0, f"{taxed.tax_paid:.2f}")

    try:
        simulate(crash, "momentum")
        check("an unknown strategy is rejected", False)
    except ValueError:
        check("an unknown strategy is rejected", True)


def test_basket_and_allocation() -> None:
    """The synthetic bond, and the multi-asset engine that runs on it.

    Two failure modes are pinned here. The first is a bond series that is
    quietly wrong -- it is built from a formula, not downloaded, so nothing
    else would notice. The second is the engine seeing a return before it
    trades on it, which is the look-ahead that makes every allocation backtest
    beautiful.
    """
    print(chr(10) + "the basket and the allocator")
    from bot.allocate import (BAND, LOOKBACK, _target_weights,
                              block_bootstrap_sharpe, sharpe_of, simulate)
    from bot.basket import Basket, bond_returns, month_end, price_returns

    day = 86_400_000
    base = 1_577_836_800_000

    # A bond whose yield never moves earns its coupon and nothing else.
    flat = {(2000, m): 6.0 for m in range(1, 13)}
    got = bond_returns(flat)
    check("a still yield pays the coupon, monthly",
          all(abs(r - 0.06 / 12) < 1e-12 for r in got.values()),
          str(sorted(got.values())[:1]))

    # A rising yield is a capital loss, and a longer bond loses more.
    rise = {(2000, 1): 4.0, (2000, 2): 5.0}
    short = bond_returns(rise, maturity=2.0)[(2000, 2)]
    long = bond_returns(rise, maturity=20.0)[(2000, 2)]
    check("a yield rise is a loss", long < 0, f"{long:.4f}")
    check("and duration makes it worse", long < short, f"{long:.4f} vs {short:.4f}")
    fall = bond_returns({(2000, 1): 5.0, (2000, 2): 4.0}, maturity=20.0)
    check("a yield fall is a gain", fall[(2000, 2)] > 0)

    check("month_end takes the last bar of each month",
          month_end([candle(base, 10.0), candle(base + day, 11.0),
                     candle(base + 40 * day, 12.0)]) == {(2020, 1): 11.0,
                                                         (2020, 2): 12.0})
    got_pr = price_returns({(2020, 1): 100.0, (2020, 2): 110.0})
    check("price returns are ratios of consecutive months",
          list(got_pr) == [(2020, 2)] and abs(got_pr[(2020, 2)] - 0.1) < 1e-12,
          str(got_pr))

    # Only months where every asset exists. A partial row would silently give
    # one asset a free month.
    basket = Basket(equity={(2020, 1): 0.01, (2020, 2): 0.02, (2020, 3): 0.03},
                    bond={(2020, 1): 0.00, (2020, 2): 0.00},
                    cash={(2020, 1): 0.00, (2020, 2): 0.00, (2020, 3): 0.00})
    check("a month missing one asset is dropped whole",
          basket.months() == [(2020, 1), (2020, 2)], str(basket.months()))

    # Weights come from history alone, and there is a defined answer before the
    # lookback exists rather than a crash or a guess.
    empty = {"equity": [], "bond": []}
    check("with no history every adaptive rule holds the naive mix",
          _target_weights("riskparity", empty, 2.0) == {"equity": 0.6, "bond": 0.4})
    check("the benchmarks do not need history",
          _target_weights("equity", empty, 2.0)["equity"] == 1.0)
    check("an unknown strategy is rejected",
          _raises_kind(lambda: _target_weights("alpha", empty, 2.0), ValueError))

    calm = {"equity": [0.02, -0.02] * LOOKBACK,
            "bond": [0.001, -0.001] * LOOKBACK}
    parity = _target_weights("riskparity", calm, 2.0)
    check("risk parity gives the quiet asset the bigger weight",
          parity["bond"] > parity["equity"], str(parity))
    check("and its weights sum to one before leverage",
          abs(sum(parity.values()) - 1.0) < 1e-12)

    down = {"equity": [-0.02] * LOOKBACK, "bond": [0.01] * LOOKBACK}
    trend = _target_weights("tsmom", down, 2.0)
    check("a falling asset is dropped by the trend rule",
          trend["equity"] == 0.0 and trend["bond"] > 0, str(trend))

    matched = _target_weights("sixty40_volmatch", calm, 4.0)
    mix_vol = abs(0.6 * 0.02 + 0.4 * 0.001)
    check("vol matching levers the calm mix toward equity's own volatility",
          abs(sum(matched.values()) - (0.02 / mix_vol)) < 0.05, str(matched))
    capped = _target_weights("sixty40_volmatch", calm, 1.0)
    check("and never past the leverage cap",
          abs(sum(capped.values()) - 1.0) < 1e-9, str(capped))

    # The engine: a constant world compounds exactly, and the first month is
    # spent in cash because the book starts empty and trades after the return.
    months = [(2000, m) for m in range(1, 13)]
    steady = Basket(equity={k: 0.01 for k in months},
                    bond={k: 0.0 for k in months},
                    cash={k: 0.0 for k in months})
    run = simulate(steady, "equity", months, band=BAND)
    # Month one is spent in cash because the book starts empty and trades only
    # after that month's return; the entry then pays one era commission.
    from bot.ladder import cost_bps
    from bot.allocate import _stamp
    fee = cost_bps(_stamp((2000, 1))) / 10_000.0
    want_cagr = ((1.01 ** 11) / (1.0 + fee) - 1.0) * 100.0
    check("eleven compounded months of 1%, the first spent in cash",
          abs(run.cagr - want_cagr) < 1e-6, f"{run.cagr:.6f}% vs {want_cagr:.6f}%")
    check("and it ends fully in equity", run.weights["equity"] > 90.0,
          f"{run.weights['equity']:.1f}%")
    check("a still market has no drawdown", run.max_dd == 0.0)

    # Costs must bite: the same world, priced.
    from bot.ladder import Taxes
    taxed = simulate(steady, "sixty40", months, taxes=Taxes(30.0), band=0.0)
    free = simulate(steady, "sixty40", months, taxes=Taxes(0.0), band=0.0)
    check("tax on rebalancing gains reduces the result",
          taxed.cagr <= free.cagr, f"{taxed.cagr:.4f} vs {free.cagr:.4f}")

    # The bootstrap, including the bug it used to have: when every resampled
    # difference agreed with the observed one, the old code fell through and
    # reported the p-value of the opposite hypothesis.
    rng = random.Random(5)
    noise = [rng.gauss(0, 0.04) for _ in range(400)]
    same = block_bootstrap_sharpe(noise, list(noise))
    check("identical series differ by nothing and are not significant",
          abs(sharpe_of(noise) - sharpe_of(list(noise))) < 1e-12
          and same[2] > 0.5, str(same))
    better = [x + 0.02 for x in noise]
    lo, hi, p = block_bootstrap_sharpe(better, noise)
    check("a large, consistent gap is significant and the CI excludes zero",
          p < 0.05 and lo > 0, f"p={p:.4f} CI=[{lo:.3f},{hi:.3f}]")
    check("and the CI brackets the observed difference",
          lo <= sharpe_of(better) - sharpe_of(noise) <= hi)


def test_trend_panel() -> None:
    """The panel engine, and the two ways it could flatter the trend signal.

    A trend rule that can see the month it trades on beats everything, and a
    control that differs from the strategy in anything besides the signal makes
    the difference meaningless. Both are pinned here, on synthetic markets where
    the right answer is known by construction.
    """
    print(chr(10) + "trend across a panel")
    from bot.trendpanel import LOOKBACK, Panel, simulate

    months = [(2000 + i // 12, i % 12 + 1) for i in range(120)]

    def panel_of(**series) -> Panel:
        p = Panel(cash={k: 0.0 for k in months})
        for name, values in series.items():
            p.markets[name] = dict(zip(months, values))
            p.group[name] = "equity"
        return p

    # Returns must VARY: the weights are inverse-volatility, so a market with a
    # constant return has infinite weight by the formula and zero by the guard,
    # and is never held at all. The first version of these fixtures used
    # constant returns and every assertion below compared 0.000 to 0.000.
    up_a = [0.03, -0.01] * 60          # +1%/month on average
    up_b = [0.025, -0.005] * 60
    down = [-0.03, 0.01] * 60          # -1%/month on average

    # Every market rising for ever: the signal is always on, so the trend book
    # and the always-on book must be the same book.
    rising = panel_of(a=up_a, b=up_b)
    on = simulate(rising, True, months, cost_bps=0.0)
    always = simulate(rising, False, months, cost_bps=0.0)
    check("with the signal always true the two books are identical",
          abs(on.cagr - always.cagr) < 1e-9, f"{on.cagr:.9f} vs {always.cagr:.9f}")

    # One market falls for ever: trend drops it, always-on carries it.
    mixed = panel_of(good=up_a, bad=down)
    on = simulate(mixed, True, months, cost_bps=0.0)
    always = simulate(mixed, False, months, cost_bps=0.0)
    check("the trend rule drops a permanently falling market",
          on.cagr > always.cagr, f"{on.cagr:.3f} vs {always.cagr:.3f}")

    # No look-ahead: the crash month itself must be taken. A market that rises
    # for a year and then falls once cannot be exited before it falls.
    turn = panel_of(x=up_a[:LOOKBACK] + [-0.30]
                      + up_a[:120 - LOOKBACK - 1])
    hit = simulate(turn, True, months, cost_bps=0.0)
    check("the signal cannot dodge the bar it is reacting to",
          hit.max_dd > 0, f"max_dd={hit.max_dd:.2f}%")

    # Volatility targeting has to bite in the right direction.
    noisy = panel_of(a=[0.05, -0.05] * 60, b=[0.04, -0.04] * 60)
    low = simulate(noisy, False, months, cost_bps=0.0, target_vol=0.05)
    high = simulate(noisy, False, months, cost_bps=0.0, target_vol=0.20)
    check("a higher volatility target takes more risk",
          high.vol > low.vol, f"{high.vol:.2f}% vs {low.vol:.2f}%")
    check("and the gross cap holds", high.exposure <= 200.0 + 1e-9,
          f"{high.exposure:.1f}%")

    # Costs must reduce the result, and turnover must be counted.
    free = simulate(mixed, True, months, cost_bps=0.0)
    dear = simulate(mixed, True, months, cost_bps=100.0)
    check("costs reduce the result", dear.cagr < free.cagr,
          f"{dear.cagr:.3f} vs {free.cagr:.3f}")
    check("and turnover is reported", free.turnover > 0)

    # Restricting to a group must actually restrict.
    grouped = panel_of(a=[0.01] * 120, b=[0.01] * 120)
    grouped.group["b"] = "commodity"
    only_a = simulate(grouped, False, months, cost_bps=0.0, groups=("equity",))
    both = simulate(grouped, False, months, cost_bps=0.0)
    check("a group filter changes which markets are live",
          max(only_a.live) == 1 and max(both.live) == 2,
          f"{max(only_a.live)} vs {max(both.live)}")


def _raises_kind(fn, kind) -> bool:
    """Did `fn` raise this exact family? Named apart from the older one-argument
    `_raises` above, which this shadowed on its first outing and broke a test
    four hundred lines away."""
    try:
        fn()
    except kind:
        return True
    except Exception:                                    # noqa: BLE001
        return False
    return False


def test_second_vendor() -> None:
    """Parsing and aligning the independent monthly series.

    The failure this guards is the quiet one: a vendor renames a column, every
    row parses to zero, and the cross-check reports perfect agreement between
    Yahoo and a wall of zeroes. So a changed header raises rather than
    degrading, and a row with no usable price is dropped rather than kept as 0.
    """
    print(chr(10) + "the second vendor")
    from bot.shiller import Month, dividend_yield_path, month_of, parse
    from bot.vendors import MonthStat, diagnose, monthly_stats

    csv = "\n".join([
        "Date,SP500,Dividend,Earnings,Consumer Price Index,"
        "Long Interest Rate,PE10",
        "1871-01-01,4.44,0.26,0.4,12.46,5.32,0.0",   # no CAPE for ten years
        "1929-09-01,31.30,0.97,1.61,17.3,3.6,32.56",
        "1932-06-01,4.77,0.50,0.41,16.1,3.6,5.84",
        "2026-07-01,7481.34,0.0,0.0,0.0,0.0,0.0",    # dividend not published
        "2026-08-01,0.0,0.0,0.0,0.0,0.0,0.0",        # no price at all
        "garbage,,,,,,",
    ])
    months = parse(csv)
    check("rows with no price are dropped", len(months) == 4, str(len(months)))
    check("and the ones kept are in date order",
          [m.year for m in months] == [1871, 1929, 1932, 2026])
    check("a pre-1970 month gets a negative timestamp, not an OSError",
          months[0].open_time < 0, str(months[0].open_time))
    check("the timestamp round-trips through month_of",
          month_of(months[1].open_time) == (1929, 9))
    check("the dividend yield is dividend over price",
          abs(months[1].dividend_yield_pct - 0.97 / 31.30 * 100) < 1e-9)
    check("an unpublished dividend is 0, not forward-filled",
          months[3].dividend == 0.0 and months[3].dividend_yield_pct == 0.0)
    check("and it is excluded from the yield path",
          set(dividend_yield_path(months)) == {(1871, 1), (1929, 9), (1932, 6)},
          str(sorted(dividend_yield_path(months))))

    try:
        parse("Date,SPX,Payout\n1871-01-01,4.44,0.26")
        check("a renamed column raises instead of parsing to zeroes", False)
    except ValueError:
        check("a renamed column raises instead of parsing to zeroes", True)

    check("an empty file is empty, not a crash", parse("") == [])
    check("Month is frozen so a parsed row cannot be edited later",
          Month.__dataclass_params__.frozen)

    # Monthly aggregation of daily bars: all three statistics a vendor might
    # publish, so a disagreeing row can be identified rather than shrugged at.
    from datetime import datetime, timezone

    day = 86_400_000
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    jan = int((datetime(1990, 1, 1, tzinfo=timezone.utc) - epoch).total_seconds() * 1000)
    bars = [candle(jan + i * day, 100.0 + i) for i in range(10)]
    stats = monthly_stats(bars)
    check("one month, ten bars", list(stats) == [(1990, 1)]
          and stats[(1990, 1)].bars == 10)
    got = stats[(1990, 1)]
    check("mean, first and last are all available",
          got.first == 100.0 and got.last == 109.0
          and abs(got.mean - 104.5) < 1e-9, str(got))

    check("a matching mean is named", diagnose(104.5, got) == "= Yahoo's monthly mean")
    check("so is a month-end close",
          diagnose(109.0, got) == "= Yahoo's month-end close")
    check("so is a first close -- the 2023-09 case",
          diagnose(100.0, got) == "= Yahoo's first close of the month")
    check("and something matching nothing says so",
          diagnose(42.0, got) == "matches no Yahoo statistic")
    check("MonthStat is frozen too", MonthStat.__dataclass_params__.frozen)


if __name__ == "__main__":
    print("=" * 62)
    print("  OFFLINE TESTS (no network)")
    print("=" * 62)
    test_series()
    test_signal_gates()
    test_maker_exit_pricing()
    test_broker_costs()
    test_risk()
    test_strategy_round_trip()
    test_stop_priority()
    test_config_guard()
    test_diagnostics()
    test_relaxed_gates()
    test_simulate_matches_strategy()
    test_features()
    test_ml_pipeline()
    test_dashboard_snapshot()
    test_findings_store()
    test_dashboard_integrity()
    test_cache()
    test_reseed_replaces()
    test_atr_levels()
    test_timeframe_generality()
    test_reversal_direction()
    test_fee_discount()
    test_intrabar_bracket()
    test_ml_labels_match_the_bot()
    test_clustering_statistics()
    test_volume_measure()
    test_volatility_matching()
    test_point_in_time_universe()
    test_internal_call_arity()
    test_no_dead_imports()
    test_cli_wiring()
    test_daily_bars_are_not_intraday()
    test_sessions()
    test_asset_classification()
    test_broker_protocol_matches()
    test_replication_pooling()
    test_bootstrap()
    test_spread_band_gate()
    test_simulate_exit_signature_is_strict()
    test_spread_estimator()
    test_cost_model()
    test_nullcheck_stats()
    test_exitfit()
    test_sweep_uses_per_symbol_levels()
    test_sweep_scoring()
    test_ladder()
    test_ladder_invariants()
    test_longterm()
    test_basket_and_allocation()
    test_trend_panel()
    test_second_vendor()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("  ALL PASSED")
    sys.exit(0)
