"""The same test, on real US equities.

The whole project reduces to one sentence: a genuine +0.039% edge cannot pay a
15-20 bps toll. Equities are the interesting follow-up because at a
commission-free broker **the toll is almost entirely the spread** -- on liquid
large caps that is 1-2 bps, an order of magnitude below Binance. The same edge
that loses money there would not automatically lose money here.

So this asks two things, in order:

  1. Does the reversal effect exist in equities at all? (In tokenized equities
     on Binance it did not: -0.041%, p = 0.27.)
  2. If it does, does the much lower toll let it through?

Everything session-shaped is handled in `sessions.py` and applied here: no
signal whose measurement window straddles an overnight gap, and no position
carried through a close.

Honest limits, none of which the numbers can remove:
  * Yahoo data is consolidated and carries no bid/ask, so the spread is
    estimated from OHLC with no ground truth to check it against -- unlike the
    Binance work, where `bookTicker` could validate the estimator.
  * The symbol list is today's liquid names. Survivorship bias, same as crypto,
    and worse here because delisting is more common than for a top-50 coin.
  * Commission-free retail order flow is typically sold to a market maker; the
    price improvement or lack of it is not in this model.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

from .config import Config
from .costs import CostModel
from .exchange import Candle
from .findings import record
from .levels import levels_for
from .nullcheck import Arm, bootstrap_ci, welch
from .series import Series
from .sessions import contiguous_back, forward_window, session_stats
from .signals import Rejection, evaluate
from .simulate import simulate_exit
from .spread import estimate_spreads
from .yahoo import load_equity_history

log = logging.getLogger(__name__)

# Liquid US large caps and ETFs. Chosen for tight spreads and continuous
# volume, which is the condition under which the low-toll argument holds at
# all; a thin small cap has a wider spread than most of the Binance universe.
DEFAULT_SYMBOLS = [
    # broad and sector ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "XLY", "XLU", "XLB", "XLRE", "SMH", "ARKK", "TQQQ", "SQQQ", "SOXL", "SOXS",
    "TNA", "TZA", "LABU", "JETS", "XBI", "IBB", "KRE", "XOP", "OIH", "OARK",
    # mega and large cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "V", "MA", "UNH", "HD", "PG", "JNJ", "COST", "ORCL", "CRM", "ADBE",
    "AMD", "INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ARM", "TSM",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "PYPL", "SQ", "SHOP", "SPOT",
    # high-beta and retail-heavy names, where overreaction should be largest
    "COIN", "MSTR", "PLTR", "SOFI", "HOOD", "RIVN", "LCID", "NIO", "GME",
    "AMC", "BBAI", "IONQ", "RGTI", "QBTS", "SMCI", "MARA", "RIOT", "CLSK",
    "AFRM", "UPST", "CVNA", "DKNG", "RBLX", "SNAP", "PINS", "U", "LYFT",
    "UBER", "ABNB", "DASH", "CHWY", "PTON", "WBD", "F", "GM", "NKLA", "PLUG",
    "FCEL", "BLNK", "CHPT", "SPCE", "OPEN", "WISH", "CLOV", "TLRY", "SNDL",
    # financials, energy, staples, healthcare
    "BAC", "WFC", "C", "GS", "MS", "SCHW", "AXP", "COF", "USB", "PNC",
    "XOM", "CVX", "COP", "SLB", "HAL", "DVN", "OXY", "MRO", "APA", "FANG",
    "PFE", "MRK", "ABBV", "LLY", "BMY", "GILD", "AMGN", "MRNA", "BNTX",
    "WMT", "TGT", "KO", "PEP", "MCD", "SBUX", "NKE", "LULU", "ROST", "TJX",
    # commodities, bonds, international
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "TLT", "IEF", "HYG", "LQD",
    "EEM", "FXI", "EWZ", "EWJ", "EFA", "VXX", "UVXY", "BITO", "IBIT",
]

# Widened to ~440 names because 174 gave only 74 signals and Yahoo caps 5m
# history at 59 days -- width is the only axis left. Survivorship applies and
# is worse here than in crypto: this is a list of companies that still exist
# and still trade, so the ones that blew up between 2024 and now are missing.
MORE_SYMBOLS = [
    "GOOG", "TSM", "ASML", "SAP", "BABA", "PDD", "JD", "NTES", "BIDU", "TCOM",
    "SE", "MELI", "GRAB", "CPNG", "TME", "LI", "XPEV", "BEKE", "YMM", "IQ",
    "ADI", "NXPI", "ON", "MRVL", "MCHP", "SWKS", "QRVO", "TER", "ENTG", "WOLF",
    "ANET", "CSCO", "DELL", "HPQ", "HPE", "NTAP", "STX", "WDC", "PSTG", "IBM",
    "NOW", "SNOW", "DDOG", "MDB", "NET", "CRWD", "ZS", "OKTA", "PANW", "FTNT",
    "TEAM", "WDAY", "HUBS", "VEEV", "ZM", "DOCU", "TWLO", "BILL", "TTD", "APP",
    "ROKU", "FUBO", "PARA", "WBD", "LYV", "EA", "TTWO", "RBLX", "PLTK", "DKNG",
    "MAR", "HLT", "BKNG", "EXPE", "RCL", "CCL", "NCLH", "DAL", "UAL", "AAL",
    "LUV", "ALK", "BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG", "HWM",
    "CAT", "DE", "HON", "GE", "MMM", "EMR", "ETN", "PH", "ITW", "CMI",
    "UNP", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "XPO", "FDX", "UPS", "WM",
    "LIN", "APD", "SHW", "ECL", "DOW", "DD", "PPG", "NUE", "STLD", "CLF",
    "X", "AA", "FCX", "NEM", "GOLD", "AEM", "WPM", "RGLD", "HL", "AG",
    "MOS", "CF", "NTR", "ADM", "BG", "TSN", "HRL", "K", "GIS", "CAG",
    "KHC", "MDLZ", "STZ", "TAP", "MNST", "CELH", "KDP", "CL", "KMB", "CHD",
    "EL", "ULTA", "LVMUY", "RL", "PVH", "GPS", "ANF", "AEO", "URBN", "BURL",
    "DG", "DLTR", "FIVE", "BBY", "LOW", "TSCO", "ORLY", "AZO", "AAP", "GPC",
    "LEN", "DHI", "PHM", "NVR", "TOL", "KBH", "BLDR", "MAS", "MHK", "WHR",
    "PLD", "AMT", "CCI", "EQIX", "DLR", "SPG", "O", "VICI", "WELL", "PSA",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG",
    "VST", "CEG", "NRG", "TLN", "OKLO", "SMR", "LEU", "CCJ", "UEC", "DNN",
    "ENPH", "SEDG", "RUN", "FSLR", "NOVA", "ARRY", "SHLS", "PLUG", "BE", "BLDP",
    "LUNR", "RKLB", "ASTS", "PL", "BKSY", "VSAT", "IRDM", "GSAT", "SATS", "SPIR",
    "SNOW", "AI", "SOUN", "BBAI", "TEM", "SERV", "PATH", "CXAI", "MTTR", "APLD",
    "CORZ", "WULF", "HUT", "BITF", "HIVE", "CIFR", "IREN", "GLXY", "BTBT", "SDIG",
    "ELF", "DECK", "ONON", "CROX", "SKX", "VFC", "UAA", "FL", "DKS", "HIBB",
    "CMG", "YUM", "QSR", "DPZ", "WING", "SHAK", "CAVA", "SG", "PTLO", "TXRH",
    "ABNB", "TRIP", "WH", "H", "CZR", "MGM", "LVS", "WYNN", "PENN", "BALY",
    "SCHW", "IBKR", "MKTX", "CME", "ICE", "NDAQ", "CBOE", "TW", "VIRT", "LPLA",
    "BLK", "BX", "KKR", "APO", "ARES", "CG", "TPG", "OWL", "STEP", "PX",
    "AIG", "MET", "PRU", "ALL", "TRV", "CB", "PGR", "HIG", "LNC", "UNM",
    "CI", "CVS", "HUM", "ELV", "CNC", "MOH", "HCA", "THC", "UHS", "DVA",
    "ISRG", "SYK", "BSX", "MDT", "ZBH", "BAX", "BDX", "EW", "ABT", "DXCM",
    "ALGN", "IDXX", "A", "TMO", "DHR", "WAT", "MTD", "RVTY", "BIO", "ILMN",
    "REGN", "VRTX", "BIIB", "INCY", "EXEL", "ALNY", "SRPT", "IONS", "NBIX", "JAZZ",
    "CRSP", "NTLA", "BEAM", "EDIT", "VERV", "RXRX", "ABCL", "TWST", "PACB", "NVTA",
    "VKTX", "MDGL", "AKRO", "ALT", "TERN", "STOK", "PRTA", "DNLI", "SAGE", "AXSM",
]
DEFAULT_SYMBOLS = DEFAULT_SYMBOLS + [s for s in MORE_SYMBOLS if s not in DEFAULT_SYMBOLS]

# What a retail equity round trip actually costs, in basis points per side.
# Commission is zero at the large US brokers; the spread is the cost, and it is
# estimated per symbol from the data like everywhere else in this project.
EQUITY_COMMISSION_BPS = 0.0

# The spread this test assumes, per side, when it cannot be measured.
#
# It cannot be measured here. The Abdi-Ranaldo estimator resolves a spread only
# when the spread is a meaningful fraction of the bar's own volatility, and a
# 1-2 bps quote inside an hourly bar that moves 0.5% is nowhere near it -- on
# synthetic bars with a known 2 bps spread and 2% bar volatility it returned
# 26.6 bps before the significance gate was added, and returns nothing after.
#
# So this is an assumption, stated: roughly the published NBBO half-spread on
# liquid US large caps, which is where the whole low-toll argument came from.
# Every equity number depends on it, and it is the first thing to attack if one
# of them looks good.
ASSUMED_SPREAD_BPS = 1.5

# Below this many signals, no conclusion is printed in either direction.
MIN_FOR_VERDICT = 200


@dataclass
class Result:
    signals: Arm
    random: Arm
    gross: list[float]
    refused_gap: int
    sessions: dict

    @property
    def edge(self) -> float:
        return self.signals.mean - self.random.mean

    @property
    def gross_mean(self) -> float:
        return sum(self.gross) / len(self.gross) if self.gross else 0.0


def _equity_config(cfg: Config, interval: str, hold_bars: int | None) -> Config:
    """The same strategy, priced for a commission-free equity broker.

    `hold_bars` exists because the hold is configured in minutes and converted,
    which is right for intraday but useless once a bar is a day: 30 minutes
    rounds to one bar whatever you meant. Stating it in bars is the only
    unambiguous way to say "hold for three days".
    """
    trial = Config(**{**cfg.__dict__})
    trial.strategy = replace(
        cfg.strategy, timeframe=interval,
        # The band was tuned against crypto spreads; 8 bps would gate out
        # nothing here and 0 would gate out everything. Anchor it to the
        # assumption instead so the two cannot drift apart.
        max_spread_bps=max(cfg.strategy.max_spread_bps, 4 * ASSUMED_SPREAD_BPS),
    )
    # The estimator cannot see a 1-2 bps spread through an hourly bar's own
    # volatility, and now says so instead of guessing -- on 1h equity bars it
    # declined every symbol. What it used to return was 25 bps median, which
    # gated out every signal and produced an empty run that looked like "the
    # effect does not exist here" rather than "the cost model is broken".
    #
    # So the spread becomes a stated assumption: ASSUMED_SPREAD_BPS, applied
    # flat, sourced from published NBBO figures for liquid US large caps rather
    # than from this data. An assumption named in the output is honest; a
    # measurement that is really noise is not.
    trial.fees = replace(cfg.fees, taker_bps=EQUITY_COMMISSION_BPS,
                         bnb_discount=False, slippage_mode="estimated",
                         slippage_bps=ASSUMED_SPREAD_BPS,
                         quote_fallback="flat")
    if hold_bars:
        minutes = round(hold_bars * trial.bar_ms / 60_000)
        trial.risk = replace(cfg.risk, max_hold_minutes=minutes)
    return trial


async def run_equities(
    cfg: Config,
    days: float = 59.0,
    interval: str = "5m",
    symbols: list[str] | None = None,
    draws: int = 3,
    seed: int = 0,
    hold_bars: int | None = None,
) -> dict:
    trial = _equity_config(cfg, interval, hold_bars)
    names = symbols or DEFAULT_SYMBOLS
    log.info("loading %d US symbols, %s bars, %.0f days", len(names), interval, days)
    history = load_equity_history(names, interval, days, cfg.path("data/yahoo"))
    if not history:
        raise RuntimeError("no equity data returned")

    costs = CostModel(trial, estimate_spreads(history))
    log.info("%s", costs.summary())

    result, rejections = _measure(trial, history, costs, draws, seed)
    _print(trial, result, len(history), days, interval, costs, rejections)
    lo, hi, boot_p = bootstrap_ci(result.signals, result.random)
    if result.signals.n >= MIN_FOR_VERDICT:
        record(cfg, "equities", {
            "edge": result.edge, "p": welch(result.signals, result.random)[1],
            "n": result.signals.n, "interval": interval, "days": days,
            "assumed_spread_bps": ASSUMED_SPREAD_BPS,
        })
    return {
        "n": result.signals.n, "gross": result.gross_mean,
        "net": result.signals.mean, "edge": result.edge,
        "ci": [lo, hi], "p": welch(result.signals, result.random)[1],
        "refused_gap": result.refused_gap,
    }


async def run_equity_windows(
    cfg: Config,
    windows: int = 6,
    days: float = 90.0,
    interval: str = "1h",
    symbols: list[str] | None = None,
    draws: int = 3,
    seed: int = 0,
    hold_bars: int | None = None,
) -> dict:
    """The equity test repeated over non-overlapping windows.

    Two independent things are being asked at once, and that is the point.

    The crypto result rests on four windows and a between-window p of 0.079 --
    consistent, small, unproven. More crypto windows would sharpen it, but they
    would still be the same market, the same venue, and the same estimator. A
    reversal effect that is real should also show up somewhere else. Hourly bars
    on US equities are a different asset class, different data vendor, and a
    different cost structure, so agreement there is worth more than another
    month of Binance.

    Yahoo serves 729 days of hourly bars, which is the only reason this is
    possible at all -- enough room for six to eight windows that share no data.
    Spreads are re-estimated inside each window rather than once over the whole
    span, for the same reason the universe is ranked point-in-time: a cost
    computed partly from a window's own future is not a cost anyone could have
    paid.
    """
    from .replicate import student_t_p

    trial = _equity_config(cfg, interval, hold_bars)
    names = symbols or DEFAULT_SYMBOLS
    span_ms = int(days * 86_400_000)
    total = windows * days
    log.info("loading %d symbols, %s bars, %.0f days for %d windows",
             len(names), interval, total, windows)
    history = load_equity_history(names, interval, total, cfg.path("data/yahoo"))
    if not history:
        raise RuntimeError("no equity data returned")

    newest = max(c[-1].open_time for c in history.values() if c)
    rows: list[tuple[int, float, int, float, float]] = []

    for k in range(windows):
        hi = newest - k * span_ms
        lo = hi - span_ms
        sliced = {s: [c for c in cs if lo <= c.open_time < hi]
                  for s, cs in history.items()}
        sliced = {s: cs for s, cs in sliced.items()
                  if len(cs) >= trial.strategy.lookback_bars
                  + trial.max_hold_bars + 10}
        if len(sliced) < 5:
            log.warning("window %d: only %d usable symbols, skipped",
                        k + 1, len(sliced))
            continue
        costs = CostModel(trial, estimate_spreads(sliced))
        try:
            result, _ = _measure(trial, sliced, costs, draws, seed + k)
        except RuntimeError as exc:
            log.warning("window %d: %s", k + 1, exc)
            continue
        rows.append((k, result.edge, result.signals.n,
                     result.signals.mean, result.signals.stderr))
        log.info("window %d: edge %+.4f%% on %d signals",
                 k + 1, result.edge, result.signals.n)

    if not rows:
        raise RuntimeError("no window produced signals")

    _print_windows(trial, rows, interval, days, len(history))

    edges = [r[1] for r in rows]
    mean = sum(edges) / len(edges)
    p = 1.0
    df = len(edges) - 1
    se = 0.0
    if df >= 1:
        var = sum((e - mean) ** 2 for e in edges) / df
        se = (var / len(edges)) ** 0.5
        if se > 0:
            p = student_t_p(mean / se, df)
    total = sum(r[2] for r in rows)
    out = {"windows": len(rows), "mean_edge": mean, "se": se, "p": p, "df": df,
           "edges": edges, "n": total,
           "positive": sum(1 for e in edges if e > 0)}
    # A run this thin is not a weak result, it is an absent one, and the
    # dashboard now reads whatever gets recorded. The first 1h run produced
    # four to six signals per window and a +1.53% "edge" from four trades; a
    # number like that reaching the page would be worse than no number.
    if total < MIN_FOR_VERDICT:
        print(f"  NOT RECORDED: {total} signals across {len(rows)} windows is "
              f"below the {MIN_FOR_VERDICT} needed for any verdict.")
        print("  The gates are calibrated for 1-minute bars. On hourly bars a "
              "0.6% move over three bars\n  is ordinary rather than extreme, so "
              "the absolute-move gate and the z-score gate\n  pull against each "
              "other and almost nothing survives both. See ASSUMED_SPREAD_BPS\n"
              "  and the module docstring.")
        out["recorded"] = False
        return out
    record(cfg, "equity_windows", {**out, "interval": interval,
                                   "window_days": days})
    out["recorded"] = True
    return out


def _print_windows(cfg, rows, interval, days, symbols) -> None:
    from .replicate import student_t_p, t_critical

    edges = [r[1] for r in rows]
    n = len(edges)
    mean = sum(edges) / n
    print("\n" + "=" * 80)
    print(f"  US EQUITIES, {n} NON-OVERLAPPING WINDOWS -- {cfg.strategy.direction}, "
          f"{interval} bars, {symbols} symbols")
    print("=" * 80)
    print(f"    {'window':<10}{'days back':>12}{'signals':>10}"
          f"{'net/trade':>13}{'edge':>12}")
    for k, edge, count, net, _se in rows:
        print(f"    {'#' + str(k + 1):<10}{f'{k * days:.0f}-{(k + 1) * days:.0f}':>12}"
              f"{count:>10,}{net:>+12.4f}%{edge:>+11.4f}%")
    print("-" * 80)
    positive = sum(1 for e in edges if e > 0)
    print(f"    positive in {positive} of {n} windows")
    if n >= 2:
        var = sum((e - mean) ** 2 for e in edges) / (n - 1)
        se = (var / n) ** 0.5
        if se > 0:
            t = mean / se
            p = student_t_p(t, n - 1)
            tc = t_critical(n - 1)
            print(f"    mean edge {mean:+.4f}%   se {se:.4f}   "
                  f"t = {t:+.2f} on df={n - 1}   p = {p:.3f}")
            print(f"    95% CI [{mean - tc * se:+.4f}%, {mean + tc * se:+.4f}%]"
                  f"   (t multiplier {tc:.2f}, not 1.96)")
            print("    (each window one observation -- assumes nothing about "
                  "trades being independent)")
    print("=" * 80 + "\n")


def _measure(cfg: Config, history: dict[str, list[Candle]], costs: CostModel,
             draws: int, seed: int) -> tuple[Result, dict[str, int]]:
    """Signal arm and random arm over one span of history.

    Split out of run_equities so a window can be measured on its own. Nothing
    here reads the clock or the config's date range -- it sees only the candles
    it is handed, which is what makes carving the history into non-overlapping
    windows safe.
    """
    bar_ms = cfg.bar_ms
    trial = cfg
    lookback = trial.strategy.lookback_bars
    horizon = trial.max_hold_bars
    # A signal needs the momentum window plus the bar itself inside one session.
    need_contiguous = trial.strategy.momentum_bars + 1
    rng = random.Random(seed)

    real: list[float] = []
    gross: list[float] = []
    fake: list[float] = []
    refused_gap = 0
    # Which gate is starving the sample? Guessing at that is how a strategy
    # gets "adapted" to a new asset class by adjusting whatever was blamed.
    rejections: dict[str, int] = {}
    stats = session_stats(next(iter(history.values())), bar_ms)

    for symbol, candles in history.items():
        if len(candles) < lookback + horizon + 10:
            log.info("  %s: only %d bars, skipped", symbol, len(candles))
            continue
        series = Series(symbol, lookback)
        for c in candles[:lookback]:
            series.update(c)
        last_index = len(candles) - horizon - 1
        found = 0

        for i in range(lookback, last_index):
            series.update(candles[i])
            verdict = evaluate(series, trial.strategy, costs.spread_bps(symbol))
            if isinstance(verdict, Rejection):
                rejections[verdict.reason] = rejections.get(verdict.reason, 0) + 1
                continue
            # The gate crypto never needed: the move must be measured inside
            # one session, not across last night.
            if not contiguous_back(candles[: i + 1], need_contiguous, bar_ms):
                refused_gap += 1
                continue
            window = forward_window(candles, i, horizon, bar_ms)
            if not window:
                refused_gap += 1
                continue
            out = simulate_exit(candles[i].close, window, trial,
                                levels_for(trial, series), costs.slippage_bps(symbol))
            if out is not None:
                real.append(out.net_pct)
                gross.append(out.gross_pct)
                found += 1

        for _ in range(found * draws):
            j = rng.randrange(lookback, last_index)
            if not contiguous_back(candles[: j + 1], need_contiguous, bar_ms):
                continue
            window = forward_window(candles, j, horizon, bar_ms)
            if not window:
                continue
            w = Series(symbol, lookback)
            for c in candles[max(0, j - lookback): j + 1]:
                w.update(c)
            out = simulate_exit(candles[j].close, window, trial,
                                levels_for(trial, w), costs.slippage_bps(symbol))
            if out is not None:
                fake.append(out.net_pct)

    if not real:
        raise RuntimeError("no equity signals on this window")

    return (Result(Arm("signals", real), Arm("random", fake), gross,
                   refused_gap, stats), rejections)


def _print(cfg, r: Result, symbols: int, days, interval, costs,
           rejections: dict[str, int] | None = None) -> None:
    t, p = welch(r.signals, r.random)
    lo, hi, boot_p = bootstrap_ci(r.signals, r.random)
    round_trip = 2 * (cfg.fees.effective_taker_bps + 3.0) / 100

    print("\n" + "=" * 80)
    print(f"  US EQUITIES -- {cfg.strategy.direction}, {interval} bars, "
          f"{symbols} symbols, {days:.0f} days")
    print("=" * 80)
    print(f"    sessions in the data      {r.sessions['sessions']:>8}")
    print(f"    bars per session          {r.sessions['bars_per_session']:>8.0f}")
    print(f"    biggest gap               {r.sessions['max_gap_bars']:>8.0f} bars")
    print(f"    signals refused at a gap  {r.refused_gap:>8}")
    print(f"    hold                      {cfg.max_hold_bars:>8} bars "
          f"({cfg.risk.max_hold_minutes} min at {interval})")
    print(f"    commission                {cfg.fees.effective_taker_bps:>8.1f} bps/side")
    print(f"    {costs.summary()}")
    if rejections:
        total = sum(rejections.values())
        print("-" * 80)
        print("    what blocked the other bars:")
        for reason, count in sorted(rejections.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {reason:<28}{count:>9,}  {count / total * 100:>5.1f}%")
    print("-" * 80)
    print(f"    {'arm':<20}{'n':>8}{'win rate':>11}{'net/trade':>13}{'+/- stderr':>11}")
    for arm in (r.signals, r.random):
        print(f"    {arm.name:<20}{arm.n:>8}{arm.win_rate:>10.1f}%"
              f"{arm.mean:>+12.4f}%{arm.stderr:>10.4f}%")
    print("-" * 80)
    print(f"    gross per trade (before commission): {r.gross_mean:+.4f}%")
    print(f"    edge over random:                    {r.edge:+.4f}%")
    print(f"    Welch t = {t:+.2f}, p ~ {p:.3f}")
    if r.signals.n >= 30 and r.random.n >= 30:
        print(f"    bootstrap 95% CI: [{lo:+.4f}%, {hi:+.4f}%], p ~ {boot_p:.3f}")
    else:
        print("    bootstrap 95% CI: not computed -- sample below 30")
    print("-" * 80)

    # A verdict needs a sample. The first run of this produced 18 signals and
    # cheerfully reported "no edge in equities either" -- a conclusion the data
    # could not support in either direction. Absence of evidence, printed as
    # evidence of absence, is the exact failure this project keeps hunting.
    if r.signals.n < MIN_FOR_VERDICT:
        print(f"  ONLY {r.signals.n} SIGNALS -- no verdict either way.")
        print(f"  Below {MIN_FOR_VERDICT} the standard error ({r.signals.stderr:.3f}%)")
        print("  swamps any effect worth having. The gates are calibrated for 1m")
        print("  crypto: a 0.6% move in 15 minutes with 3x volume, breaking a")
        print("  30-bar low, is simply rare on a large cap.")
        print()
        print("  Get a real sample before concluding. Measured, not guessed:")
        print("    MORE SYMBOLS works. 46 names gave 18 signals; the universe")
        print("      here is ~170 and Yahoo caps 5m history at 59 days, so width")
        print("      is the only axis left.")
        print("    A LONGER BAR DOES NOT. 1h over 729 days gave 7 signals, fewer")
        print("      than 5m over 59: a 30-minute hold on hourly bars collapses")
        print("      to a single bar and most exits stop being reachable.")
        print("    LOOSENING THE GATES IS NOT JUSTIFIED. The 0.6% floor is 2.84")
        print("      sigma on crypto 1m but only 1.26 sigma on equity 5m, because")
        print("      equities move MORE per 3 bars here. The gate is already the")
        print("      looser of the two; the conjunction of all six is what is")
        print("      rare, and relaxing one to manufacture a sample is fitting.")
    elif p > 0.05:
        print("  No edge over random entries in equities either. The reversal")
        print("  effect measured in crypto does not appear here, so the low")
        print("  commission has nothing to let through.")
    elif r.edge < 0:
        print("  The rules are WORSE than random on equities.")
    elif r.signals.mean > 0:
        print(f"  NET POSITIVE at {r.signals.mean:+.4f}% per trade, and beating")
        print(f"  random at p = {p:.3f}. This is the one place the low toll could")
        print("  matter -- confirm with `replicate` on non-overlapping windows")
        print("  before believing any of it.")
    else:
        print(f"  Beats random by {r.edge:+.4f}% (p = {p:.3f}) but still loses")
        print(f"  {r.signals.mean:+.4f}% per trade. Even a near-zero commission")
        print("  does not rescue it, because the spread is the toll here.")
    print("=" * 80 + "\n")
