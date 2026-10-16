# Momentum edge study

[![CI](https://github.com/RaulJimenezAyza/momentum-edge-study/actions/workflows/ci.yml/badge.svg)](https://github.com/RaulJimenezAyza/momentum-edge-study/actions/workflows/ci.yml)

**Does a viral crypto trading strategy actually have an edge? I built it and measured it properly.
Mostly, it doesn't — and this repo shows how you find that out without fooling yourself.**

Under the hood it is a complete trading system in Python: it scans ~50 Binance USDT pairs over
websockets, detects volume-confirmed dislocations on 1-minute candles, and manages positions with a
stop, a target and a trailing stop. It runs in **paper mode by default** — real market data,
simulated fills, no API keys, no money at risk. Around it sits the part that matters: a research
harness that compares every rule against random entries, on windows the search never saw.

**What's in here, engineering-wise**

- **Async data pipeline** — `aiohttp` + `websockets` client for Binance REST and streams, with an
  on-disk candle cache that detects gaps, so backtests replay months of 1-minute data offline.
- **One strategy, two runtimes** — the live engine and the backtester share the same strategy and
  broker code; a reconcile command replays a live session and diffs it against what happened.
- **Statistics that don't flatter** — walk-forward search, random-entry null tests matched on
  volatility, block bootstrap so signals fired by the same market move count once, between-window
  tests, multiple-testing correction, point-in-time universes to remove survivorship bias.
- **A cost model checked against reality** — per-symbol spread estimates validated against live
  order-book quotes, because costs turned out to be the biggest error in the whole study.
- **A live web viewer** (`bot/dashboard.py`, one self-contained HTML page) and an ML filter
  (scikit-learn, walk-forward evaluated) that turned out not to help — reported, not hidden.
- **51 offline tests (~490 checks)** that run in seconds with no network, including structural
  tests that parse the code to catch wrong call arity, dead imports and CLI flags no command reads.
  Linted with ruff and run on CI for Python 3.12 and 3.13.

**Where to start reading:** the [thirty-second answer](#the-answer-in-thirty-seconds) ·
[the null test](#the-null-test-do-the-rules-do-anything-at-all) ·
[the cost model mistake](#the-cost-model-was-the-biggest-error-in-this-repo) ·
[the mistake that mattered most](#the-mistake-that-mattered-most) ·
[non-independent trades](#the-trades-were-never-independent-and-every-p-value-assumed-they-were) ·
[install and run](#install) · [code layout](#layout)

The rest of this README is the full lab notebook, in the order it happened.

## The answer in thirty seconds

Someone posted that a bot like this turned **$68 into $750,000**. It cannot.
This repo is the attempt, measured properly:

- **Momentum — chasing breakouts, the thing the post described — has no edge.**
  It is statistically indistinguishable from buying at random moments
  (p = 0.13), and slightly worse than random on the point estimate.
- **Fading capitulation has a real but tiny edge.** Across **five
  non-overlapping 30-day windows** covering 150 days and 14,656 trades, it
  beats random entries by about **+0.037% per trade**, positive in every
  non-overlapping window tested, with the universe ranked on volume predating
  each window and controls matched on volatility. But once the arithmetic stops
  pretending that twenty signals fired by one market-wide dump are twenty
  independent facts, the evidence is **suggestive, not conclusive**:
  **p = 0.079** treating each window as a single observation. See below. It is
  strongest in the recent windows and fades to nothing in the oldest, so treat
  it as regime-dependent rather than a constant of nature.

- **Eleven plausible improvements were tested and all eleven died out of sample**
  — parameter tuning, ATR stops, spread-band selection, an ML filter, longer
  timeframes, refitted exits, best-hours-only, a raised z threshold, tokenized
  equities, real US equities, and confirmation entry. The only things that ever
  moved the number were **corrections to my own measurement errors**.

- **A second viral post — the "buy 20% dips, sell 30% rallies" ladder — was
  tested over ninety-nine years and loses by 2.8% CAGR.** But the interesting
  part is the split: **-2.34% of that is simply being in cash**, and only
  **-0.48% is the timing rules themselves** — and that term changes sign under
  three separate things that should not have mattered: which re-arm rule, which
  trigger, and which sixty years. That is what no effect looks like. The prices
  behind it are cross-checked against a second, independent vendor across 1,183
  months, median disagreement 0.4 bps. In markets that collapsed — Athens -93% —
  the ladder wins in the falling windows and loses in the rising ones by
  roughly what holding cash is worth either way, with the timing term doing
  nothing in both. The rules are an asset allocation wearing a
  market-timing costume. See below.

- **A multi-asset search, pre-registered with a sealed holdout, also came out
  short.** The strategy chosen on 1962-1993 beat 100% equity on 1994-2026
  risk-adjusted return by +0.075 Sharpe — p = 1.000 after correcting for the
  eight candidates, and an effect that size would need **96 years** of holdout
  to prove. At matched volatility the diversified portfolio beats the index by
  0.23% a year. What it does deliver is drawdown: **28% against 51%**.
- **The one rule with real literature behind it — 12-month trend — was then
  replicated on fifteen markets it was not chosen on, and 93% of its apparent
  edge turned out to come from a single currency index whose buy-and-hold
  benchmark loses money by construction.** On the nine equity indices it is
  *worse* than always being invested. Trend looks best exactly where
  buy-and-hold is a bad benchmark.

The honest summary is not "momentum fails but reversal works". It is that the
one surviving effect **comes and goes depending on the period**, and even in
the periods where it is real it is about four times too small to pay the toll.

Everything below is the working, in the order it happened, including the eight
bugs found in my own measurement code along the way — several of which had been
quietly flattering the results. The reversal sections are left in place rather
than deleted, because how a plausible result survived five tests and then
evaporated is the most useful thing in this repo.

## About the post this is based on

The post claimed a bot like this turned **$68 into $750,000**. That part is
fiction. The mechanics — scanning 50+ markets, spotting sharp moves, firing
orders in seconds — are ordinary and are what this repo implements. The return
is not.

Why it can't happen: $68 → $750k is a **~11,000x** return. Binance charges
0.10% taker fee per side, so a round trip costs ~0.20% plus slippage before the
strategy is right about anything. To 11,000x you would need roughly 900
consecutive winning trades at +1% net, with zero losses, on position sizes the
order book could actually absorb. Momentum strategies on liquid pairs typically
run a 35–45% win rate with a profit factor near 1.0 before costs, and slightly
below 1.0 after them. Screenshots of this kind come from account balances after
deposits, futures leverage on a lucky streak, or nothing at all.

Build it, backtest it, paper-trade it for weeks, and judge it on expectancy and
drawdown. Not on a screenshot.

## What it actually did

Backtest run on 2026-08-11, 7 days of 1m data, 50 USDT pairs, default config,
$1000 start — reproduce with `python run.py backtest --days 7`:

```
Trades                                673
Win rate                            41.2%
Avg win / avg loss        +0.96% / -1.05%
Expectancy per trade              -0.227%
Profit factor                        0.64
Gross PnL                          -24.08
Fees paid                         -186.42
Net PnL                           -210.50
Max drawdown                       21.62%
Equity                 1000.00 -> 789.50 (-21.05%)
```

Read the two lines that matter: **gross PnL was −24, fees were −186.** The
strategy was very close to a coin flip before costs and lost decisively after
them. 673 round trips at 0.20% each is a 134% turnover tax on a $1000 account.
That is the actual, ordinary outcome of the design the post was selling.

The obvious objection is "so tune it". That is what `run.py sweep` is for, and
the answer is below.

## What the sweep found

`python run.py sweep --days 30` scores **108 settings** of the entry gates and
exit levels across 4,490 candidates, then does the honest thing: picks the
winner on one time fold and reports how it did on the *next* one.

```
Current config: z>=2.5 vol>=3x tp=2.4 sl=1.2 hold=30m
  4490 trades, 36.0% wins, expectancy -0.3082%

Best in sample (>= 40 trades) -- NOT a result, a ranking:
   expectancy  trades    wins  config
     -0.2642%    1657   43.2%  z>=3.5 vol>=8x tp=6.0 sl=2.0 hold=60m
     -0.2681%    1657   43.2%  z>=3.5 vol>=8x tp=4.0 sl=2.0 hold=60m
     -0.2768%    2319   43.7%  z>=2.5 vol>=8x tp=6.0 sl=2.0 hold=60m

Walk-forward -- chosen on one fold, scored on the NEXT:
   fold   in-sample   next fold  trades  config
      0    -0.3371%    -0.3311%     346  z>=4.5 vol>=3x tp=2.4 sl=2.0 hold=60m
      1    -0.2430%    -0.3536%     598  z>=2.5 vol>=8x tp=6.0 sl=2.0 hold=60m
      2    -0.2975%    -0.1385%     248  z>=4.5 vol>=5x tp=6.0 sl=1.2 hold=60m
   mean    -0.2925%    -0.2744%
```

**Not one of the 108 is profitable.** And look at where the best ones land:
−0.264%, against a round-trip cost of 0.30%. The entire grid clusters just
below zero-minus-costs, which says the gross edge of these entry rules is
approximately **zero** — the loss is the fee, not the signal.

Walk-forward confirms there is nothing to tune toward: the winner of each fold
does no better on the next one than anything else would. The −0.018% gap
between in-sample and out-of-sample is what parameter selection bought.

That is the real answer to the post. Not "the bot is badly tuned" — the design
has no edge to tune, and 0.20% in fees per round trip is a wall it never gets
over. Anything that claims otherwise has to explain where the gross edge comes
from, and a screenshot is not an explanation.

## The null test: do the rules do anything at all?

Everything above compares the strategy to zero. `python run.py nullcheck`
compares it to the benchmark that actually isolates the entry logic: **random
entries, same symbols, same exits, same costs.** For every real signal, a
random bar is drawn from the *same* symbol, so a difference cannot be explained
by one side trading easier coins.

```
arm                        n   win rate    net/trade   +/- stderr
strategy signals        4491      36.0%     -0.3093%     0.0161%
random entries         13473      30.5%     -0.2827%     0.0070%

edge of the rules over random: -0.0265% per trade
Welch t = -1.51, p ~ 0.130
```

**The six gates are statistically indistinguishable from entering at random.**
The point estimate is actually *worse* than random, though not significantly so.

Note the trap in that table: the strategy's win rate is 36.0% against random's
30.5% — it wins more often and still ends up no better, because the gates
select for setups that win small and lose big. A win rate quoted without an
expectancy beside it is not evidence of anything.

This is the cheapest experiment in the project and it reframes everything
before it. There was never any point tuning six gates that do nothing; a
seventh would not have helped either.

## The one thing that did work

`python run.py experiments --days 30` scores every structural variant —
momentum vs reversal, 1m vs 5m vs 15m, flat stops vs ATR stops — each against
its own random-entry benchmark.

```
variant               signals   wins     gross       net  +/- se    random     edge      p
reversal   1m fixed      3515  44.9%   +0.010%   -0.190%  0.017   -0.284%  +0.094%  0.000
reversal   1m   atr      3515  37.8%   -0.037%   -0.237%  0.012   -0.282%  +0.046%  0.002
reversal  15m fixed       542  37.1%   -0.038%   -0.238%  0.034   -0.309%  +0.070%  0.065
momentum  15m fixed       756  33.2%   -0.055%   -0.255%  0.035   -0.295%  +0.040%  0.297
...
momentum   1m fixed      4491  36.0%   -0.109%   -0.309%  0.016   -0.287%  -0.022%  0.223
```

**Fading capitulation beats chasing breakouts, and it is not noise.** Reversal
on 1m clears its random benchmark by +0.094% per trade at p < 0.001 over 3,515
signals. Momentum — the thing the post was selling, and the thing this repo was
built to be — sits at the bottom of the table, below random.

It is also the only variant with a **positive gross expectancy: +0.010%**. Note
what that number already includes: slippage is inside `gross` (entry pays it,
exit pays it), so +0.010% is what survives the fill assumption. Only exchange
fees are still to come.

And that is where it dies. The edge is **1 basis point per trade**. Binance
charges 20 bp per round trip, 15 with the BNB discount. The cheapest tier that
exists for anyone — VIP9, at institutional volume — is still ~2–3 bp. The edge
is real and it is off by a factor of ten to thirty.

Two things worth taking from that table beyond the headline:

- **ATR stops made things worse on 1m** (−0.037% vs +0.010% gross). Scaling
  exits to volatility is the more principled design, and it lost. Principled
  and profitable are different axes; the measurement is what decides.
- **The random benchmark is itself negative** (~−0.28% net, ~−0.08% gross).
  That is the exit structure's own drag — the conservative stop-before-target
  ordering plus an asymmetric stop. Reversal entries claw that drag back to
  roughly zero. Without the random arm you would read +0.010% as "no edge"
  rather than "recovered 9 bp of structural drag".

## The cost model was the biggest error in this repo

Every conclusion above rested on a number I made up: **5 bps of slippage per
side**, chosen by hand and labelled "deliberately pessimistic". That is 10 bps
per round trip of invented cost, sitting on top of a measured edge of about
1 bp. An assumption ten times larger than the effect being measured does not
inform the answer, it *is* the answer.

The live watchlist made it obvious — real spreads on this universe:

```
BNBUSDT   0.2 bps      LINKUSDT   1.1 bps      HEIUSDT  15.1 bps
SNDKBUSDT 0.1 bps      PUMPUSDT   3.7 bps      RADUSDT  38.4 bps
```

Charging all of them 5 bps is wrong for every single one, in both directions.

`fees.slippage_mode: estimated` replaces the guess with a per-symbol estimate
from the symbol's own candles, using Abdi & Ranaldo (2017) — the mid-range
`(high+low)/2` is a cleaner proxy for the efficient price than the close, which
is contaminated by whichever side of the book the last trade hit:

```
s² = 4 · E[(c_t − η_t)(c_t − η_{t+1})]      c = log close, η = (log high + log low)/2
```

A market order pays roughly *half* the quoted spread, times
`impact_multiplier` for walking past the touch.

The estimator has a resolution floor, and the tests pin it down rather than
paper over it: a 2 bps spread under normal intrabar noise is unmeasurable and
the function returns **0 — "cannot tell"** — which downstream becomes the
configured floor rather than a fabricated small number. Drop the noise and the
same 2 bps comes back as 1.97. It is a resolution limit, not a defect, and the
test proves which.

This is still the quoted spread, not realised impact on real size. It is honest
about crossing the book on a liquid pair; it does not model pushing size
through a thin one. `slippage_mode: fixed` keeps the pessimistic number.

## The answer, on a holdout the search never saw

Search ran on 45 days ending 14 days ago. The last 14 days were held back and
used **once**, at the end, on the single variant the search picked.

Reversal 1m, per-symbol estimated costs, BNB discount, both intrabar bounds:

```
window                        n     wins    net/trade   vs random        p
search  45d (ending 14d ago)  6290  46.7%   -0.1555%     +0.0641%    0.000
holdout 14d (never searched)  1599  47.7%   -0.1119%     +0.1225%    0.000
holdout, optimistic bound     1599  47.8%   -0.1048%     +0.1270%    0.000
holdout, fees set to zero     1600  52.2%   +0.0372%     +0.1191%    0.000
```

**Gross expectancy on the holdout is positive: +0.0372% per trade, 52.2% wins.**
That number is after slippage — only exchange fees are excluded. The portfolio
backtest on the same 14 days says the same thing in dollars:

```
Trades 1008 | Win rate 47.2% | Profit factor 0.76
Gross PnL   +50.93
Fees       -233.12
Net PnL    -182.19
```

Compare to where this started: momentum's gross was **−24.08**. It is now
**+50.93**, on a window chosen before the result was known. The signal went
from worse-than-nothing to genuinely positive.

And it still loses, because 1008 round trips cost $233.

### The exact threshold

```
gross expectancy (holdout):  +0.0372% per trade
break-even taker fee:         1.86 bps per side

  VIP0                10.00 bps/side  ->  net -0.1628%
  VIP0 + BNB           7.50 bps/side  ->  net -0.1128%
  needed               1.86 bps/side  ->  net  0.0000%
```

That is the whole answer to "can it be made positive". **Before fees, yes,
and it is confirmed out of sample.** After retail fees, no, and it is not close
by a factor of four. Reaching 1.86 bps/side means the deepest institutional VIP
tiers, or maker-only execution — and modelling maker fills honestly needs
order-book data plus a non-fill assumption this backtest cannot supply.

### Where I was wrong, twice

**The 5 bps slippage guess.** Correcting it to measured per-symbol spreads was
worth more than every entry rule in this repo combined: the round trip fell
from an invented 0.30% to a measured 0.165–0.194%, and the best net moved from
−0.190% to −0.155%.

**The intrabar assumption.** I predicted the always-assume-the-stop rule was
costing about 7 bp of gross — the same size as the entire edge — and built the
bracket to measure it. It costs **0.15 bp**: pessimistic −0.1555% vs optimistic
−0.1540%. The prediction was wrong by a factor of fifty. Candles that span both
a 1.2% stop and a 2.4% target are simply rare on 1m bars.

The bracket was still worth building. It converted "my conclusion might be an
artefact of a modelling choice" into "it demonstrably is not", and that holds
for every result above.

### Why the default changed

`strategy.direction` now ships as **reversal**, and `fees.slippage_mode` as
**estimated**. Not because reversal is what the project set out to build — it
is the opposite — but because momentum loses to random entries on every window
tested and reversal beats them at p<0.001 on five, including the holdout.
Set `direction: momentum` to reproduce the earlier sections.

## Does widening the universe help?

`python run.py markets --size 150 --min-volume 500000` expands from 50 to 149
symbols and bins them by measured spread. The prior: overreaction should be
*larger* in thinner names, since less capital stands ready to arbitrage a
panic — but their spreads are wider, so it is a race.

```
tier        syms  spread     n    gross      net     edge      p
<2 bps        48    0.9b  1715  -0.004%  -0.204%  +0.020%  0.376
2-5 bps       54    3.3b  2883  +0.032%  -0.168%  +0.088%  0.000
5-12 bps      34    7.0b  1953  -0.032%  -0.232%  +0.075%  0.003
12-30 bps     12   17.8b   758  -0.086%  -0.286%  +0.181%  0.000
>30 bps        1   50.0b    86  -0.447%  -0.647%  +0.074%  0.659
```

**The prior was half right, and the half that was right is the interesting
one.** Edge over random does rise as liquidity falls — +0.020% in the tightest
tier to +0.181% in the 12-30 bps tier, p<0.001. The overreaction story holds.
But spread rises faster, so gross falls, and there is an interior optimum
around 2-5 bps. The 48 most liquid symbols barely overreact at all.

That looked actionable: select the universe by spread, not by volume rank. So
`strategy.min_spread_bps` turns the ceiling into a band, and it was tested.

**It did not survive the holdout.** Restricted to 2-5 bps: −0.1025% per trade
against −0.1119% unrestricted. That is 0.36 standard errors apart — nothing.
The band ships **off**, with the measurement written into `config.yaml` next to
it so the next person does not have to rediscover that it looked good and was
not.

More markets is also the wrong frame. With a 5-position cap the bot picks from
a queue; it is not missing trades. More symbols at a negative expectancy is
more loss, not more opportunity.

## The learned filter, retrained on the strategy that works

The model was originally fit on momentum candidates. Momentum has no edge, so
that was a filter on noise. Refitting it on reversal — where there *is* a
measured, replicated edge — should be the best shot the ML has.

```
AUC 0.5276 | Brier 0.2467 | filtering makes expectancy WORSE by 0.0369%
VERDICT: USELESS -- AUC is indistinguishable from a coin flip
```

Note the inversion: **AUC 0.606 on momentum (no edge), 0.528 on reversal (real
edge).** The reversal edge is a property of the *setup type*, not of the
individual instance — every candidate in the pool is worth about the same, so
there is no good subset to select. Nothing to filter means nothing to learn.

That is a real answer, not a failed experiment. And it is not fixed by a bigger
model: at AUC 0.53 over 20k rows the problem is that the information is not in
the features, and extra capacity only memorises noise faster. (An LLM would be
a worse choice still — this is 27 numeric columns of tabular data, which is
what gradient boosting is for.)

### A bug that would have poisoned it

`build_dataset` labelled trades with `simulate_exit(price, future, cfg)` — no
levels, no per-symbol slippage. The module docstring promises labels come from
"the same exit rules the live bot uses". That promise quietly became false the
moment ATR levels and measured costs were added: in ATR mode the model would
have trained on fixed-percentage stops the bot does not use.

Fixed, and pinned by a test that compares each label against the strategy's own
outcome and asserts that ATR mode relabels relative to fixed mode — if the
levels are not threaded through, that assertion fails.

## Statistics that do not assume a shape

Trade returns are not normal. They are truncated at the stop and the target, so
the distribution is a spike at each barrier with little in between. Welch's t
survives that at n>1000 by the CLT, but the null test now also reports a
**percentile bootstrap** CI, which assumes nothing, and flags any disagreement:

```
edge of the rules over random: +0.0945% per trade
Welch t = +3.33, p ~ 0.001
bootstrap 95% CI: [+0.0388%, +0.1487%], p ~ 0.001
```

The test for it asserts **coverage** — the 95% CI must contain the truth about
95% of the time over 120 simulated experiments (measured: 94%) — rather than
checking one interval and hoping. Asserting on a single draw is the same
mistake as asserting that one p-value lands above 0.05, and it had already been
made once in this repo.

## Are the exits even pointed at the signal?

Half of all exits in the portfolio backtest were `max_hold` — 507 of 1008. The
trade neither hit its target nor its stop; it ran out of clock. That is a
statement about the exits, not the signal, and the exits were chosen for
momentum and never revisited when the strategy became reversal.

`python run.py exitfit` measures the forward path first and prescribes second.
Part one has no free parameters at all:

```
SIGNAL SHAPE -- 5355 reversal signals, 1m bars
  horizon   MFE p50   MFE p75   MFE p90   MAE p50   MAE p25   MAE p10
      10m    +0.45%    +0.78%    +1.25%    -0.37%    -0.86%    -1.70%
      30m    +0.67%    +1.18%    +1.90%    -0.58%    -1.26%    -2.48%
     120m    +1.21%    +2.20%    +3.61%    -1.04%    -2.08%    -3.66%

Share of signals that ever touch a target within the hold:
  target      10m      20m      30m      60m     120m
    0.6%      37%      48%      55%      65%      74%
    1.0%      16%      26%      33%      45%      58%
    2.4%       2%       4%       6%      12%      22%
```

**Only 6% of signals ever touch the 2.4% target within 30 minutes.** The median
favourable excursion at that horizon is +0.67% — the target was set at 3.6× the
typical move the signal produces. 94% of trades could only ever leave by the
stop or the clock, which is exactly the `max_hold` pile-up, now explained rather
than observed.

That table also prices the whole problem in one line: the signal moves about
±0.6% in 30 minutes, and a round trip costs 0.15–0.19%. **The toll is a quarter
of the typical move.**

### Refitting the exits does not rescue it

The grid scores stop × target × hold over those same recorded paths,
walk-forward. In sample it looks like a big win — gross rises from +0.008% to
+0.067%, eight times better. Out of sample it does not hold:

```
Walk-forward -- chosen on one fold, scored on the NEXT:
   fold   in-sample   next fold  setting
      0    -0.0493%    -0.1817%  sl=1.8 tp=4.0 hold=60m
      1    -0.0604%    -0.1159%  sl=1.8 tp=4.0 hold=120m
   mean    -0.0549%    -0.1488%   (baseline -0.1420%)
```

The exits were mismatched, and fixing the mismatch was still not worth
anything. They were not the binding constraint.

### The grid had my assumption baked into it

The first version required `target > stop`. That is an unexamined belief that
risk:reward must exceed 1:1 — and it made the 0.6% target, the one the
reachability table says is where the mass lives, **untestable at every stop in
the grid**. A mean-reversion bounce is supposed to be small and frequent;
requiring R:R > 1 assumes the shape of the answer before measuring it.

The constraint is gone and sub-1 R:R combinations are now in the grid, with the
reason written next to it. The test that asserted the old constraint now
asserts the opposite.

Re-running with the widened 150-cell grid changed **nothing**: short targets do
not reach the top eight, and the walk-forward verdict is identical to the digit.
Hitting +0.6% often is worth little when the losses are −1.2% and the toll is
0.15%. The result surviving the removal of my own assumption is the reason the
assumption was worth removing.

## Multiple testing: does the one surviving claim survive that too?

This project has run a lot of hypothesis tests, and running enough of them
guarantees a few will look significant. The bookkeeping, generously counted:
12 structural variants scored twice, a 108-cell parameter sweep, a 150-cell
exit grid, 5 liquidity tiers, 6 null tests and 2 model fits — call it **140
tests**.

Bonferroni over 140 tests puts the bar at **alpha = 3.6e-04** instead of 0.05.

The headline claim — reversal entries beat random entries — measured Welch
t = 6.05 on the 60-day window, which is **p = 1.4e-09**. It clears the corrected
bar by a factor of about 246,568. It also replicated at p < 0.001 on five
separate windows including a holdout the search never saw, which is a stronger
statement than any single corrected p-value.

Everything else in this repo is on the other side of that line, and the
interesting part is that **each one looked good in sample first**:

| Claim | In sample | Out of sample | Kept? |
|---|---|---|---|
| Some parameter setting is profitable | best of 108 | walk-forward flat | no |
| Trade only 2-5 bps spreads | +0.032% gross, p<0.001 | 0.36 se from baseline | no |
| A model can rank the setups | — | AUC 0.528, filtering hurts | no |
| ATR-scaled exits | more principled | worse gross than flat | no |
| Refit the exits | gross 8x better | worse than baseline | no |
| Trade only in the best hours | +0.076% gross, CI clear of 0 | p = 0.34, CI straddles 0 | no |
| Trade only in one volatility regime | +0.082% gross | not monotone — wrong shape | no |
| Reversal beats random | t = 4.9 | t = 4.1 on the holdout | **yes** |

Seven plausible improvements died on out-of-sample data. That is the expected
hit rate, and it is the reason the holdout was carved out before the search
rather than after it.

## Validating the cost model against the real book

The cost model decides every conclusion here — switching off the hand-picked
5 bps moved the round trip from 0.30% to 0.17% and every net figure with it.
It had never been checked against ground truth, and it is derived from OHLC
rather than from quotes. `/api/v3/ticker/bookTicker` gives the real bid and ask
for free, so `python run.py spreadcheck` compares them.

```
measured                 34 symbols
UNMEASURABLE             16 symbols  (real spreads up to 22.4 bps)
median estimated       3.88 bps   (measured only)
median quoted          3.75 bps
median ratio           0.76x
rank correlation       0.88
```

Where it works it works well: **rank correlation 0.88**, median within 24% of
the quoted spread. That is the estimator doing its job.

**Where it fails, it was failing dangerously.** A third of the universe sits
below its resolution floor, and those were being *floored at 0.5 bps* — the
tightest book in the universe asserted for exactly the symbols the estimator
had given up on. LUNAUSDT quoted **22.4 bps** and was being charged 0.5. That
is a 40x understatement of cost, in the direction that flatters every result.

Unmeasurable symbols are now **omitted** from the estimate map, so the caller
falls back to the configured flat slippage. Failing to measure something is not
evidence that it is cheap.

### And the bias check had the same disease

The ratio was computed as `[e / q for ... if e > 0]` — dropping precisely the
symbols where the estimator had failed. Survivorship bias inside the very
diagnostic meant to detect bias. It now counts them and reports them
separately, which is how the 32% failure rate became visible at all.

### An accident that had to become a decision

The corrected fallback is 10 bps, which is wider than `max_spread_bps: 8`, so
unpriced symbols now fail the entry gate and are never traded. Signals on the
holdout dropped from 1599 to 926 — **42% of the trade population disappeared**
because two settings happened to interact.

"Do not trade what you cannot price" is the right rule. It was not a rule, it
was a side effect. It is now logged at startup with the count and the reason,
and a test asserts the fallback exceeds the gate on purpose.

Holdout with the corrected model: net −0.0832%, edge over random **+0.1159%
(p = 0.004, bootstrap CI [+0.038%, +0.195%])**. The edge survives; the smaller
sample widens the interval. The conclusion does not move.

## The viewer, for someone who has never traded

The page opens with a yellow bar saying **pretend money**, and a plain-language
panel that answers "what am I looking at?" in four short paragraphs — no
jargon, and it states the finding up front rather than letting a newcomer
assume they are watching a money machine:

> A trading robot watching 50 crypto coins, one price update a minute. When a
> coin drops much faster than that particular coin normally moves, *and*
> unusually many people are trading it, the robot buys — betting the fall went
> too far and the price will bounce back.
>
> Every buy and every sell pays a fee. Measured over months of real prices, the
> bouncing is real but small, and the fees are about four times bigger than it.
> So this loses money on purpose-built, carefully tested rules. That is the
> finding, not a bug.

Dismiss it and it stays dismissed, with a small "what am I looking at?" button
to bring it back.

Under the account value sits **one sentence describing the present moment**,
generated from the live state:

> Watching 50 coins. None is unusual enough to trade — the closest is HOMEUSDT,
> about 78% of the way there, held back because it moved, but no more than this
> coin normally does in a few minutes.

Every heading was rewritten out of jargon and every one carries an explanation
on hover:

| Was | Now | On hover |
|---|---|---|
| `Z` | Unusual? | "How unusual that move is FOR THIS COIN. 1 = an ordinary minute, 3+ = genuinely rare." |
| `Vol ×` | Traded | "How much more trading than normal. 3× means three times this coin's usual volume." |
| `Ext` | Stretch | "How far the price is from its own recent average." |
| `Spread` | Cost | "You pay about half of it every single time you trade — this is the cost that eats the strategy." |
| `Heat` | Closeness | "How close this coin is to meeting all the buy conditions." |
| `Blocked by` | Why not buying | per-reason sentence, e.g. *"The price moved without unusual trading behind it — usually noise, not a real move."* |

Same for the tiles (`Net P&L` → **Profit after fees**, with "this is the number
that actually matters"), the exit reasons (`max_hold` → **time stop**, "neither
the target nor the safety exit was reached in time"), and the position track
(`Stop→Target` → **Safety exit → Target**).

The tab title follows the strategy, and if the config is switched back to
breakout mode the explainer prints a warning that its description is inverted —
the page previously said "Momentum bot" while running reversal.

## Does the edge depend on the regime?

Worth asking for two reasons from evidence already in hand rather than a hunch:
the reversal model was useless as a filter (AUC 0.53) but its permutation
importance put **BTC's own volatility first**, and the prior is clean — an
overreaction trade should pay best when the market is panicking.

Two cuts, both fixed before looking: BTC volatility at signal time, and hour of
day.

```
BTC VOLATILITY            n     gross       net     edge            95% CI
calm                    535   -0.059%   -0.209%  +0.063%  [-0.052, +0.173]
quiet                   837   +0.082%   -0.068%  +0.144%  [+0.048, +0.242]
active                  995   -0.020%   -0.170%  +0.040%  [-0.047, +0.125]
turbulent              1528   +0.027%   -0.123%  +0.072%  [-0.005, +0.148]

HOUR OF DAY               n     gross       net     edge            95% CI
00-06 UTC               922   +0.076%   -0.074%  +0.152%  [+0.061, +0.246]
06-12 UTC               786   -0.074%   -0.224%  -0.065%  [-0.161, +0.025]
12-18 UTC              1382   -0.002%   -0.152%  +0.033%  [-0.044, +0.114]
18-24 UTC               805   +0.061%   -0.089%  +0.111%  [+0.024, +0.199]
```

**The volatility cut fails its own prediction.** The prior said more volatility
means more overreaction; the best bucket is the *second* quartile and the
pattern is not monotone. A non-monotone result against a monotone prediction is
noise wearing a pattern.

**The hour cut looked real**, with a mechanism to match — thinner overnight
books overshoot further, and the two good windows sit either side of the Asian
session. So `strategy.trading_hours_utc` was built and tested properly.

```
Holdout, restricted to 00-06 + 18-24 UTC:
  net -0.1218% per trade   (unrestricted: -0.0832%)
  edge over random +0.0534%, p = 0.34, CI [-0.056%, +0.159%]
```

**Worse than not filtering, and indistinguishable from random.** Choosing the
best two of four buckets found noise. The setting ships empty, with the
measurement written beside it.

## Making the recurring bug unrepresentable

`simulate_exit` is the single function every measurement in this repo runs
through. It used to look like this:

```python
def simulate_exit(signal_price, future, cfg, levels=None, slippage_bps=None):
    lv = levels or fixed_levels(cfg)
    slip = (cfg.fees.slippage_bps if slippage_bps is None else slippage_bps) / 10_000
```

Two callers were written that left both out — the ML labeller and the parameter
sweep. Each silently scored a strategy with **different stops and different
costs than the bot runs**: in ATR mode the model trained on flat-percentage
exits it would never take, and the sweep ranked 108 configurations while
charging every symbol the same made-up slippage.

Both bugs were found, one at a time, weeks apart. Both were possible only
because the arguments had defaults, and the defaults made the wrong call the
*shorter* one to write.

They are per-symbol quantities. There is no correct global default, so there is
now no default:

```python
def simulate_exit(signal_price, future, cfg, levels: Levels, slippage_bps: float):
```

Every call site had to be updated, which is the point — the compiler found the
ones I would not have. A test asserts, via `inspect.signature`, that neither
parameter has a default, and that both actually reach the arithmetic rather
than merely being accepted. Fixing a bug twice is not a fix; deleting the
shape that allows it is.

## The mistake that mattered most

For most of this document the surviving claim is "reversal beats random,
replicated at p < 0.001 across five separate windows including a holdout the
search never saw". That sentence has a false word in it, and the word is
**separate**.

| Test | Days ago covered |
|---|---|
| reversal null test, 60d | 0–60 |
| search window, 45d offset 14 | 14–59 |
| spread-band search, 21d offset 14 | 14–35 |
| holdout, 14d | 0–14 |

They are overlapping slices of the same sixty days. Worse: the 14-day holdout
sits **inside** the 60-day window that supposedly confirmed it. The same data
was counted several times and the total was reported as replication.

The holdout discipline itself was sound — it was genuinely out-of-sample with
respect to the *parameter search*, and it correctly killed the spread band, the
hour filter and the exit refit. What it could not do, and what I claimed it
did, was show that the effect persists over time.

Running one window that does not touch the others settles it:

```
NULL TEST -- reversal, 30 days ending 120 days ago
  signals    1753   42.4%   -0.2037%
  random     5259   33.0%   -0.2019%
  edge -0.0018%   t = -0.09   p = 0.932   CI [-0.042%, +0.041%]
```

Flat zero. Not weaker — absent, with the interval centred on nothing.

A second non-overlapping window then put it back:

```
NULL TEST -- reversal, 30 days ending 60 days ago
  edge +0.0696%   t = +3.45   p = 0.001   CI [+0.032%, +0.110%]
```

So the answer is not "it does not replicate" either — that conclusion was drawn
from a single window and was itself premature, the same error one level up. The
picture from non-overlapping periods is **on, off, on**: real in some regimes,
absent in others, and there is no mechanism predicted in advance that says
which. That is a much weaker claim than the one this document spent most of its
length building, and it is the one the data supports.

My own multiple-testing arithmetic should have caught it: I corrected for 140
tests as though they were independent, when the numerator was essentially one
period looked at repeatedly.

### The fix is a command, not a resolution to be careful

The error was possible because the tooling made overlapping windows the easy
thing to type (`--days 60`) and non-overlapping ones a manual chore. So the
correct test is now the one-word command:

```bash
python run.py replicate --windows 5 --days 30
```

It carves N strictly non-overlapping windows going backwards, scores the same
strategy on each, and pools them by inverse variance — a meaningful combined
estimate across independent periods, which one long overlapping window is not,
however large its sample looks. The tests assert that the windows cannot
overlap by construction, and that one lucky window cannot carry the pooled
result on its own.

Being more careful next time is not a fix. Making the careless version harder
to type is.

## Does the live bot actually do what the backtest says?

Every number in this document comes from replaying history. That the live bot
behaves the same way had been checked only at unit level — `simulate_exit`
matches `Strategy`, the ML labels match the strategy — and never end to end
against a real session.

`python run.py reconcile
python run.py assetclass  --days 30
python run.py equities    --days 59 --interval 5m
python run.py maker       --days 30` reads the signals the live bot logged, replays those
exact minutes and symbols from cached candles, and lines the two lists up.

The first run said **50%**, and the disagreement turned out to be two separate
faults, both mine.

**The tool was wrong first.** It replayed with bare `evaluate()` instead of the
full `Strategy`, so it had no idea the bot was already holding a position — and
duly reported the two bars immediately after every real signal as "the bot
missed them". Position state and cooldown are part of what the bot does, so
they have to be part of the comparison.

**Then the cost model was wrong.** With the replay fixed, two live signals
still could not be reproduced, and the tool now says exactly why:

```
LIVE ONLY:
  PUMPUSDT  07:33  spread_too_wide (replay spread 10.0b vs live 3.8b)
  BICOUSDT  08:44  spread_too_wide (replay spread 10.0b vs live 2.9b)
```

The conservative fallback introduced earlier — unmeasurable symbols get the
flat 10 bps — was **gating a third of the universe out of every backtest**, and
the book was quoting those symbols at 2.9 and 3.8. Safe on cost, badly biased
on which symbols get measured at all.

The fix is that ground truth is one API call away. `bookTicker` gives the real
spread for 1,377 symbols; the cost model now resolves **measured estimate →
exchange quote → flat constant**, and the snapshot refreshes on every data
load. With it in place:

```
same bar        4
live only       0
agreement: 100% exact
```

Every live signal reproduces from historical candles. The backtest is replaying
the same events the bot actually saw.

### Two silent failures in a row

Worth recording because neither was a logic error. The snapshot refresh threw
`name 'json' is not defined` — a patch had inserted a call without its import,
because the edit's anchor text did not match and nothing asserted that it had.
And I did not see the exception for two runs, because it was logged at WARNING
and I was running with `--log-level ERROR`.

A failed edit that reports success, and a diagnostic hidden by the verbosity
flag. Both cost more time than the bug they concealed.

## The definitive replication

```
REPLICATION -- 5 NON-OVERLAPPING 30-day windows, reversal 1m
    window          n       net    random      edge       p            95% CI
    0-30d ago    2786   -0.131%   -0.203%   +0.072%   0.001  [+0.030, +0.114]
    30-60d ago   3661   -0.147%   -0.188%   +0.041%   0.021  [+0.005, +0.076]
    60-90d ago   3213   -0.134%   -0.219%   +0.084%   0.000  [+0.046, +0.123]
    90-120d ago  2535   -0.165%   -0.194%   +0.028%   0.160  [-0.012, +0.067]
    120-150d ago 2461   -0.220%   -0.195%   -0.025%   0.196  [-0.063, +0.014]
    pooled      14656                       +0.039%   0.000  [+0.022, +0.057]
```

This is the number to quote, and it is roughly **half** what the overlapping
recent windows suggested. Positive in 4 of 5, significant in 3, pooled at
+0.039% with p < 0.001 over 14,656 trades and 150 days that do not touch each
other.

Note the gradient: +0.072%, +0.041%, +0.084%, +0.028%, −0.025%. The effect is
real in the recent past and gone in the oldest window. Two readings fit —
market regime, or the fact that the gate thresholds were chosen while looking
at recent behaviour even though they were never formally fitted. Nothing here
distinguishes them, and the honest position is to say so rather than pick the
flattering one.

Either way +0.039% does not pay a 0.15–0.21% toll, and no amount of further
tuning changes an order of magnitude.

### Looks

The page was legible but plain, so it got a proper visual pass — with one rule:
**nothing that carries meaning was allowed to change.** The categorical pair and
the ordinal heat ramp are the same validated hexes, and both were re-run through
`dataviz/scripts/validate_palette.js` against the new dark surface (`#14171c`)
before it was adopted. Only surfaces, depth and motion moved.

- **Ambient dark ground.** Two faint radial washes in the series hues behind a
  near-black plane, fixed so it does not scroll. Cards sit on it as raised
  panels with a hairline top highlight, which is what makes them read as lit
  rather than drawn.
- **The chart earns the space it takes.** A gradient area fades out under the
  account line, the line carries a soft glow of its own colour, and a ringed dot
  marks the live end. The counterfactual "if trading were free" line stays flat
  and unglowing so the eye goes to the real one first, and the gap between them
  — the fee bill — is the thing you notice.
- **The hero number tweens** rather than snapping, ease-out over 420 ms, so a
  changing balance can be read instead of flickering.
- **Motion with a purpose.** The live dot pulses; bars and the position marker
  slide to their new values. All of it is disabled under
  `prefers-reduced-motion`.
- **A frozen page announces itself.** `pause` now stamps a fixed PAUSED pill
  across the top — a stopped page that still looks live is a trap.
- Sticky table headers, row hover with an accent edge, pill controls with a
  focus ring, thin scrollbars, and the stop→target track tinted red-to-green so
  it needs no legend.

Both themes were checked side by side with real data and the console is clean.

## Is it only crypto?

Binance lists more than coins. Tokenized equities and ETFs trade there against
USDT — QQQB, SPYB, NVDAB, GOOGLB, TSMB, SOXLB — and so does gold, as XAUT and
PAXG. Same API, same candles, same code, so `python run.py assetclass` costs
nothing but a classifier.

Two things had to be fixed before it could even run. Gold had been **excluded
from the universe on an assumption I never measured** ("metal-backed tokens:
high volume, no momentum"), written in `config.yaml` as though it were a fact.
And the API exposes nothing that separates a tokenized share from a coin —
`permissions` is empty for both — so the obvious suffix rule would classify
BNB, ARB, TRB and SHIB as equities. The list is curated, and a test pins that
exact trap.

```
class                 syms      n   wins     gross       net     edge       p
crypto                 101   5185  46.5%   +0.023%   -0.127%  +0.069%   0.000
equity (tokenized)      17   1240  43.1%   -0.085%   -0.235%  -0.041%   0.268
```

**The effect is crypto-specific.** Tokenized equities show nothing.

### A prediction made before the measurement, and falsified

This was the first hypothesis in the project with a *mechanism specified in
advance* rather than a bucket that looked good afterwards:

> A tokenized stock trades 24/7, but the market pricing the real share is open
> only ~14:30–21:00 UTC. While NASDAQ is shut, nobody can arbitrage the token
> against the underlying. A panic has nothing anchoring it, so it should
> overshoot further and revert harder.

That predicts a specific, directional difference. Measured:

```
equity, US open        232  40.1%   -0.175%   -0.325%  +0.002%   0.988
equity, US closed     1008  43.8%   -0.064%   -0.214%  -0.010%   0.820
```

**The opposite.** The edge is *lower* with the market shut. The mechanism is
wrong, so the hypothesis is dead rather than merely unproven — which is the
whole reason to state one in advance. `assetclass` prints `PREDICTION FAILED`
for exactly this case.

## Real equities

The project reduces to one sentence: a genuine +0.039% edge cannot pay a
15–20 bps toll. US equities are the obvious next place to look because at a
commission-free broker **the toll is almost entirely the spread** — 1–2 bps on
a liquid large cap. `python run.py equities` runs the identical null test
through a Yahoo Finance adapter.

Three things break when an instrument stops trading overnight, and
`bot/sessions.py` answers each rather than letting the arithmetic decide:
a 3-bar return at 09:35 would otherwise be measured against yesterday's close;
a 1.2% stop is not a 1.2% stop if the next print is 4% lower at the open, so
positions are closed at the bell; and the hold clock has to run in bars.
Session boundaries are inferred from gaps in the data, which gets weekends,
holidays, half-days and outages right without a calendar.

### The verdict, once the sample was big enough

At 174 symbols it produced 74 signals and an encouraging-looking +0.115% edge.
The tool refused to call it, because 74 is below the 200-signal floor. Widening
to **535 symbols** settled it:

```
            n   wins   net/trade
signals   281  46.6%   -0.0810%
random    740  38.9%   -0.0935%
edge +0.0124%   p = 0.814   CI [-0.086%, +0.112%]
```

**No edge in US equities.** The +0.115% at n=74 collapsed to +0.012% at n=281,
with the interval now tight around zero. That is the clearest demonstration in
this repo of why the sample floor exists: the encouraging number was noise, and
a version of this tool without the guard would have reported it as a finding.

Worth noting what did carry over: net per trade is −0.081% here against
−0.127% in crypto, and the whole difference is the toll. The low-cost argument
was right. There was simply no edge for it to let through.

### Three of my own errors, caught by measuring

**Overclaiming on n = 18.** The first version printed "no edge in equities
either" on eighteen signals — absence of evidence rendered as evidence of
absence, the exact failure this project keeps hunting. It now refuses to
conclude below 200.

**My own advice in the code was wrong.** It suggested `--interval 1h --days 729`
for "12x the history". Measured: **7 signals, fewer than 5m over 59 days** — a
30-minute hold on hourly bars collapses to a single bar and most exits stop
being reachable. Replaced with the measured numbers.

**And the one that reversed my intuition.** 92.9% of bars were blocked by
`move_too_small`, so the obvious move was to loosen that floor for equities.
Measuring first killed it:

```
crypto 1m : median sigma_3 = 0.2111%  ->  0.6% floor = 2.84 sigma
equity 5m : median sigma_3 = 0.4744%  ->  0.6% floor = 1.26 sigma
```

Equities move **more** per 3 bars at 5m than crypto does at 1m, so the gate is
already 2.3x *looser* there than the calibration it came from. Relaxing it
would have been fitting dressed up as "adapting to the asset class". What is
rare is the conjunction of all six gates, and that is not fixed by adjusting
whichever one happens to reject first.

## The one disciplined improvement

`run.py edgecurve` was written early and, embarrassingly, never run. When it
finally was, it produced the shape the overreaction story predicts — the only
prediction in this project that came true:

```
z range         n    wins     gross     edge       p
2.5-3.0       697   41.0%   -0.115%  +0.024%   0.554
3.0-3.5       690   45.2%   -0.002%  +0.054%   0.189
3.5-4.0       610   47.5%   -0.007%  +0.075%   0.087
4.0-5.0       834   46.4%   +0.009%  +0.107%   0.002
5.0-6.0       510   46.7%   +0.025%  +0.125%   0.007
6.0-inf       402   45.5%   +0.002%  +0.089%   0.090
```

The edge rises monotonically with the size of the dislocation from 2.5σ to
5-6σ, and gross turns positive above 4σ. Bigger overshoot, bigger bounce —
predicted by the mechanism, not found by scanning buckets.

Raising `min_return_zscore` from 2.5 to 4.0 and testing that once on the
holdout:

```
                n   wins   net/trade
signals       520  49.2%   -0.0533%     (-0.083% at z >= 2.5)
random       1560  35.2%   -0.1726%
edge +0.1193%   p = 0.021   CI [+0.018%, +0.217%]

fees zeroed: gross +0.0964%, 55.4% wins, edge +0.1397% (p = 0.007)
```

On the holdout, gross per trade goes from +0.037% to +0.096% and break-even
fees from 1.86 to 4.82 bps per side — a factor of 1.6 from the 7.5 bps
available with the BNB discount, instead of the factor of 4 everything before
it was stuck at.

**And then it did not replicate.**

```
REPLICATION -- z >= 4.0, non-overlapping 30-day windows
    window          n       net    random      edge       p            95% CI
    0-30d ago    1232   -0.112%   -0.183%   +0.071%   0.023  [+0.009, +0.132]
    30-60d ago   1772   -0.096%   -0.174%   +0.078%   0.002  [+0.033, +0.128]
    60-90d ago   1379   -0.089%   -0.207%   +0.119%   0.000  [+0.062, +0.177]
    120-150d ago 1298   -0.295%   -0.210%   -0.085%   0.001  [-0.135, -0.032]
    pooled       5681                       +0.041%   0.003  [+0.014, +0.068]
```

Pooled edge at z ≥ 4.0 is **+0.041%**. At z ≥ 2.5 it was **+0.039%**. Identical,
on 2.6x fewer trades. The +0.119% the holdout reported was that window being
favourable — precisely the failure mode named two paragraphs earlier, arriving
immediately afterwards.

It also made the worst window worse: −0.085% at p = 0.001 in 120-150d ago,
against −0.025% at p = 0.196 for the same window at z ≥ 2.5. Significantly
negative, not merely absent.

So the honest conclusion is that raising the threshold does **not** improve the
edge per trade. What survives is narrower than it looked: the monotone shape in
`edgecurve` is real and predicted, and it does not translate into a better
strategy once the extra trades stop being counted twice.

### The holdout is no longer clean, and that matters

By this point the "holdout" has been consulted for the spread band, the hour
filter, the z threshold and two zero-fee variants. **Five times.** A holdout
queried five times is a second search set wearing the name of a holdout; each
use spends a little more of its ability to surprise me.

That is why the verdict on z ≥ 4.0 belongs to `replicate` over non-overlapping
windows rather than to another holdout query, and why the number quoted in the
summary is the pooled one.

## Execution: a resting limit instead of a market order

Maker execution was dismissed early on as "not honestly modellable without
order-book data". That was too quick. Whether a resting buy limit would have
filled is decidable from OHLC alone — if a later bar trades at or below the
limit, someone hit it. The parts that genuinely need book data (queue position,
partial fills) all push the same way, so fill-on-touch is the *optimistic* end
of the range and can be labelled as such instead of hidden.

Two effects pull against each other, and the point was to measure both:

```
MAKER vs TAKER ENTRY -- reversal 1m, limit rested 3 bars
    signals                2751
    limit filled           2493   90.6%
    limit never filled      258    9.4%

    arm                                     n    wins     gross       net
    market entry                         2493   45.4%  -0.0258%  -0.1758%
    limit entry, filled                  2493   48.1%  +0.0600%  -0.0900%
    limit missed: what a market entry got 258   68.2%  +0.4644%  +0.3144%
```

**Adverse selection is real and it is severe.** This is a dip-buying strategy,
so the limit sits below a price that has already fallen: it fills precisely
when the fall continues, and misses the immediate bounces. The 258 signals the
limit never caught returned **+0.31% net at a 68.2% win rate** against −0.18%
for the ones it did catch — a gap of +0.49% at p < 0.001.

But they are only 9.4% of signals, so the spread saved on the other 90.6% still
wins: net improves from −0.176% to **−0.090%**, the best figure in this project.

It also replicates better than the edge itself does, which makes sense — it is
a mechanical execution effect, not a market prediction:

```
window          execution gain    adverse selection      p
0-30d ago            +0.0859%             +0.4903%   0.000
30-60d ago           +0.0784%             +0.4715%   0.000
120-150d ago         +0.0786%             +0.4390%   0.000
```

Still negative in every window. Saving 8 bps on a 15-20 bps toll is not enough
when the edge is 4.

### A cost-model error this exposed

Every exit was being charged the taker fee **and** half the spread. A
take-profit is a resting limit sell by construction — you name the price and
wait — so it crosses no spread and pays the maker fee. The model was
overcharging the one exit type the strategy is actually aiming for, in the
direction that made the results look worse.

Fixed, with `fees.target_exit_is_maker` to get the old behaviour back. Stops and
time exits are genuinely market orders and still pay both.

An existing test failed when this landed, and the code was right: with a maker
exit at a target that is a fixed percentage of entry, **entry slippage cancels
exactly** — you enter higher and leave proportionally higher. Arithmetic, not a
bug. The test now uses a time exit, where slippage does bite, and separately
asserts the target case really is slippage-invariant.

## Auditing the assumptions instead of inventing hypothesis twelve

By this point the pattern was unmistakable: eleven strategy ideas died out of
sample, while the two things that actually moved the result were both
corrections to my own measurement — the invented 5 bps slippage, and charging
target exits as market orders. So the remaining effort went to the assumptions
rather than to more ideas.

**Entry timing.** Every figure assumed the fill happens at the close of the bar
that produced the signal. You do know that close the instant it prints, so it is
not lookahead — but the order reaches the book afterwards and fills against
whatever trades next. The honest bracket is the next bar's open:

```
paired over all 2,799 signals
    fill at the signal bar's close (assumed): -0.1263%
    fill at the next bar's open (realistic):  -0.1401%
    cost of the assumption -0.0138% per trade (p = 0.611)
```

About 1.4 bps of optimism in every net figure here. It applies to the random arm
too, so the *edge* barely moves; the *net* does.

### The comparison was wrong before it was right, and the sign flipped

The first version of that measurement reported **+0.035%** — entering later
looked *better*. It was comparing the next-open arm over all signals against
the market-entry arm over only the signals a resting limit had filled. Those 9%
a limit misses are worth +0.32% each, and they were in one arm and not the
other. The population gap was the entire result.

Paired properly the answer is −0.014%: not merely smaller, **the opposite
sign**. The two comparisons in that module genuinely run over different
populations — execution over the filled subset, timing over everything — and
the code now says so at the point where the arms are built, because the version
that did not say so got it backwards.

## The survivorship bias I had only ever flagged

From the first backtest there has been a comment saying the universe is today's
top 50 by volume applied backwards, and that this is survivorship bias. It was
never measured and never fixed. Measured now:

```
universe: 50 symbols
with less than 145 days of history: 8  (16%)
    KORUBUSDT  21 days      SPCXBUSDT  61 days
    SKHYBUSDT  30 days      SNDKBUSDT  62 days
    SOXLBUSDT  36 days      CRCLBUSDT  62 days
    GRAMUSDT   41 days      MUBUSDT    62 days
share of all candles from those: 5.6%
```

Sixteen percent of the universe did not exist for most of the window, and seven
of those eight are tokenized equities — the class already shown to have no edge.

There are two separate biases inside "survivorship" and only one of them is
fixable here:

- **Ranking look-ahead — fixable.** Which symbols were most traded on a given
  date is knowable from the candles. `bot/pointintime.py` ranks a wide
  candidate pool by quote volume in the 30 days *preceding* each window, so a
  coin that pumped last week cannot be in the universe for the months before it
  did. `replicate --point-in-time` uses it.
- **Delisting survivorship — not fixable.** `exchangeInfo` lists only symbols
  that still trade, so a coin that was top-50 in March and has since been
  delisted cannot enter any candidate pool this code can build. The residual
  points the same way: the survivors did better than the full population did.

So the claim after this is "ranking look-ahead removed, delisting survivorship
remains" — not "unbiased". The module says so at the top, because the previous
version of this honesty was a comment nobody acted on for the whole project.

## The strongest objection to the one positive finding, tested

The strategy fires during dislocations, so its trades sit in volatile minutes
by construction. A uniformly drawn control sits in an average one. With a
fixed-percentage stop and target, volatile minutes must pay differently --
both barriers get reached more often -- so part of the measured "edge" could
be *when* it trades rather than *what* it picks. Every edge number in this
document was exposed to that.

Each control is now drawn from the same symbol **and** from a bar whose recent
volatility is within 25% of the signal bar's:

```
arm                        n   win rate    net/trade
strategy signals        2651      47.4%     -0.1259%
random entries          7953      34.4%     -0.1817%    edge +0.0558%
random, vol-matched     7953      36.7%     -0.1940%    edge +0.0681%
                                        CI [+0.0247%, +0.1132%], p = 0.001
```

**It survives, and grows.** Volatility-matched controls do *worse* than
ordinary random ones, which makes sense: in a volatile minute a fixed 1.2% stop
is nearer in volatility terms, so the stop is hit more. Trading agitated
minutes is a mild handicap, not the source of the edge -- and the naive
benchmark was understating the result rather than flattering it.

That was the most serious methodological objection left standing against the
only positive finding here. It is now tested rather than assumed away.

## A bias I predicted, measured, and found negligible

The volume gate divides the last bar's volume by the mean of the previous 120,
and it used the exchange's **base-asset** volume — units of the coin. Base
volume is money divided by price, so when a price falls the same dollar flow
prints as more units. A dip-buying strategy selects exactly the bars where the
price has fallen, so the gate that rejects 98% of bars could have been firing
on arithmetic rather than on interest.

That is a clean directional argument, and it is wrong about the size:

```
bars examined            194,050
gate passes on BOTH       13,705
passes on BASE only           51    <- fired by arithmetic
passes on QUOTE only          77
base/quote ratio   median 1.0000   p99 1.0342
share of base-gate hits quote would reject: 0.4%
```

The ratio compares one bar against a 120-bar mean, and the price typically
moves about 1% inside that window — nowhere near a 3x threshold. **0.4% of
gate hits are affected and no measured number moves.**

`strategy.volume_measure` now selects `base` or `quote` and the two are
distinguishable rather than silently conflated, because a correctness fix with
no measurable effect is still a correctness fix. The test builds a window where
the price halves at constant money flow: base volume must rise, quote volume
must not, and the tilt must only ever go upward.

Recording it because "I predicted a bias and it turned out to be 0.4%" is a
result, and burying the ones that come back small is how a document ends up
listing only the impressive half of its own measurements.

## Removing the last removable bias

`replicate --point-in-time` ranks a 150-symbol candidate pool by volume in the
30 days *before* each window, instead of by today's table:

```
window              n       net    random      edge       p            95% CI
0-30d ago        6632   -0.135%   -0.200%   +0.065%   0.000  [+0.040, +0.091]
30-60d ago       9342   -0.118%   -0.180%   +0.063%   0.000  [+0.042, +0.084]
60-90d ago      10599   -0.196%   -0.215%   +0.019%   0.083  [-0.000, +0.041]
90-120d ago      8167   -0.176%   -0.185%   +0.008%   0.457  [-0.013, +0.030]
pooled          34740                       +0.037%   0.000  [+0.026, +0.048]
```

**+0.037% against +0.042% with today's ranking.** So roughly **12% of the edge
this document had been quoting was look-ahead**, and the other 88% survives
removing it. Positive in all four windows, on 34,740 trades, with the tightest
interval in the project.

That is now the number to quote. It has been through, in order: non-overlapping
windows, volatility-matched controls, a corrected cost model, corrected exit
pricing, and a point-in-time universe. What remains un-removable is delisting
survivorship, and it points the same way.

### A look-ahead I introduced myself, two hours before removing another

The quoted-spread fallback added earlier uses **today's** bookTicker for
symbols the estimator cannot resolve — today's liquidity applied to a window
months old. Exactly the sin `pointintime.py` exists to remove, introduced
without being flagged while writing the tool that removes it.

`fees.quote_fallback` now makes it a stated choice:

- `snapshot` — today's quote. Closer to the truth for a symbol whose liquidity
  has not changed, and look-ahead.
- `flat` — the configured constant. Knowable in advance, pessimistic, and on
  the default settings it exceeds `max_spread_bps`, so those symbols are not
  traded at all rather than traded at a guessed cost.

They are not a small difference: under `flat` roughly 40% of signals disappear.
Both are defensible; silently picking one and calling the result unbiased is
not.

## The trades were never independent, and every p-value assumed they were

A market-wide dump fires the entry rule on twenty symbols in the same minute.
Those twenty outcomes are close to one observation repeated. Every standard
error and every bootstrap in this document, up to this point, counted them as
twenty independent draws.

Two corrections, and they agree.

**Block bootstrap** resamples whole six-hour spans instead of individual
trades, so a burst is drawn or dropped as a unit:

```
trade-level bootstrap:  CI [+0.0141%, +0.0951%]   p = 0.011
block bootstrap (6h):   CI [-0.0173%, +0.1282%]   p = 0.145   1.8x wider
```

On that window, **significance does not survive clustering.**

**Between-window** goes further and assumes nothing at all about trades: each
window contributes one number, and the spread across windows supplies the
error. On the four point-in-time windows (+0.065, +0.063, +0.019, +0.008):

```
mean +0.0387%   se 0.0148   t = 2.63   df = 3
Student t p = 0.079      95% CI [-0.008%, +0.086%]
```

**Not significant at 5%, and the interval crosses zero.**

### The p-value I nearly printed

The first version used the normal approximation and reported **p = 0.009**. At
three degrees of freedom the 5% critical value is 3.18, not 1.96, so the honest
answer is **0.079** — nearly nine times larger. `student_t_p` now integrates
the t density numerically (no SciPy), and the test checks it against published
critical values at df 1 through 100, worst deviation 0.0002.

Reporting 0.009 there would have been the same error as every other one in this
document: an assumption that flattered the result, made silently.

### Where that leaves the finding

The point estimate is stable — +0.037% to +0.042% under every method tried, and
positive in every window. What has changed is the confidence attached to it.
"p < 0.001 over 34,740 trades" was arithmetic built on an assumption known to be
false. "p = 0.079 over four independent windows" is what the data supports, and
four observations is a small sample by any standard.

More non-overlapping windows would settle it; Binance 1m history is the binding
constraint. Until then the honest summary is: **a consistently positive effect,
around a tenth the size of the toll it would have to pay, that has not been
demonstrated to be distinguishable from chance once clustering is respected.**

## The cost estimator was reporting noise as measurement

Every cost in this project comes from Abdi-Ranaldo, which recovers the spread
from the gap between the close and the mid-range of the bar. It returns zero
when its expectation goes negative -- "the noise swamped it, I cannot tell."

That guard is not enough. The expectation can land positive from noise alone,
and the square root of noise is not a small number. Measured against synthetic
bars carrying a **known 2 bps spread**:

```
bar volatility    true spread    estimated
    0.05%             2 bps        1.8 bps     fine
    0.20%             2 bps        3.3 bps     +63%
    0.80%             2 bps       10.7 bps     5x
    2.00%             2 bps       26.6 bps     13x
```

Nothing in the output distinguishes the last row from the first. Checked
against live `bookTicker` quotes on real symbols, one read **2.06 bps where the
book was 0.08** -- 26 times too wide.

The fix is a significance test on the estimate itself: `s^2` must clear twice
its own standard error before it is believed, with the effective sample halved
because adjacent terms share a close. Every reported number is then accurate --
2.09 for a true 2.0, 9.89 for a true 10.0 -- and the unmeasurable cases say so
instead of inventing a figure.

The price is coverage: on 1-minute crypto bars the estimator now resolves 12 of
50 symbols instead of most of them. Those 12 land about **35% below** the quoted
spread, which is the right direction -- the effective spread paid is inside the
quote. The other 38 fall back to the exchange's own book snapshot, which is
ground truth about today. **Median round-trip cost rose from 0.19% to 0.26%**
once the fabricated tight spreads stopped being counted as measurements.

## The gates only mean what they mean at one bar size

The plan was a second asset class: run the same rules on hourly US equity bars,
where a commission-free broker charges 1-2 bps instead of 15-20. Six
non-overlapping 90-day windows, 512 symbols, all free data.

It produced **31 signals from 1.25 million bars** -- four to six per window, and
one window reported a +1.53% "edge" off four trades.

The reason is worth more than the run would have been. Two gates have to pass
together: an absolute move (0.6% over three bars) and a volatility z-score (the
move must be abnormal for that symbol). On 1-minute crypto bars those agree --
0.6% in three minutes is both large and abnormal, so the pair selects the tail
they were designed for. On hourly equity bars a 3-hour 0.6% move is **ordinary**:
it clears the absolute gate and the z-score gate then rejects it for being
normal. The two gates point in opposite directions and almost nothing survives.

The gates are not scale-free. Nothing in the code said so.

Matching the scale instead of the calendar fixes it. A 1-minute equity bar moves
0.047% at the median against 0.057% for a 1-minute crypto bar -- near enough the
same measurement problem. Yahoo serves 1-minute data for seven days:

```
US equities, 1m bars, 512 symbols, 5 sessions, 30-minute hold
  signals              216      win rate 43.5%     net -0.1755%
  random entries       529      win rate 43.9%     net -0.0817%
  edge over random    -0.0938%   Welch p = 0.213   CI [-0.241%, +0.054%]
```

**The reversal effect does not appear in equities.** Point estimate negative,
not significant, on a sample that is real but thin -- five trading days is one
market regime. The low commission has nothing to let through.

That is the second asset class answering, and it does not agree with crypto.

## The interval had the same bug as the p-value

Fixing `student_t_p` and leaving `1.96` in the confidence interval would have
been the same error wearing a different hat. At df=3 the multiplier is **3.18**,
so a normal interval is 38% too narrow exactly where the sample is smallest.
`t_critical` finds it by bisection on the p-function and is checked against the
published table at df 1 through 100, worst deviation 0.0004.

Getting there needed one more correction: widening the integration cut-off does
not fix df=1, because the Cauchy tail still holds 2e-4 of its mass past x=3000.
The density beyond the cut-off is `x^-(df+1)` to good approximation and that
integrates exactly, so the remainder is now added rather than outrun.

## The second viral post: a ladder, ninety-nine years, and a decomposition

This project exists because of a post claiming $68 became $750,000. Another one
arrived with 251K views and 1.7K shares, attributed to a woman who died at 105
after 85 years of watching the US market:

    drops 10%  -> hold          rises 10%  -> hold
    drops 20%  -> buy  +15%     rises 20%  -> hold
    drops 30%  -> buy  +30%     rises 30%  -> sell 10%
                                rises 40%  -> sell 20%
                                rises 50%  -> sell 30%
                                rises 60%  -> sell 40%
                                rises 100% -> sell 60%
                    "Discipline and long-term growth."

Unlike the first post this one is cheap to test, and it can be tested on a
sample nothing else in this repo can reach. `^GSPC` daily bars start on
1927-12-30: ninety-nine years, **ten non-overlapping ten-year windows**, against
the five thirty-day windows the crypto finding rests on. `python run.py ladder`.

### Six strategies share those sentences

The post does not say what the percentages are measured *against*, and the
three readings are three different strategies: the position's **cost basis**,
the running **all-time high**, or the price on **day one**. It does not say
whether "+15%" means 15% of starting capital or 15% of the current position.
It does not say where the money to buy dips comes from — a fully invested
holder cannot obey rule 2 at all. And read literally, "drops 20% -> buy" fires
*every day* the market is 20% down, which spends the whole cash pile in the
first week of a bear market; here a rung fires once and re-arms only when price
crosses back through its own reference.

All six readings are run. Friction is era-aware — 75 bps per side before May
Day 1975, 20 through the 1990s, 5 after — because pricing a century at today's
zero-commission spread would flatter exactly one arm: the ladder trades a few
times per decade, buy-and-hold trades twice.

### The result, on ^GSPC, 1927-2026

```
    baseline                   CAGR    max DD
    hold 100%                +8.96%     37.9%
    hold 50% + cash          +6.62%     20.8%

    variant            vs hold      se       p   win  vs matched   max DD  in mkt
    peak/initial        -2.82%    1.15   0.036  3/10      -0.48%    20.2%     41%
    basis/initial       -2.93%    1.60   0.099  2/10      -0.60%    16.5%     34%
    peak/position       -3.06%    1.11   0.022  2/10      -0.72%    16.0%     35%
    anchor/initial      -3.13%    1.46   0.061  3/10      -0.79%    16.4%     33%
    anchor/position     -3.23%    1.17   0.022  2/10      -0.89%    13.9%     33%
    basis/position      -3.23%    1.17   0.022  2/10      -0.89%    14.1%     33%
```

Every reading loses, in eight of ten windows, on the ^GSPC, the ^IXIC and the
^N225 alike. But "it loses" is the uninteresting half of the answer, and
printing only that would have been the same failure this document keeps
recording: a true sentence standing in for a measurement.

### The decomposition is the finding

Two different things are inside that -2.82%. Holding 59% of the portfolio in
cash for a century costs a specific amount whatever you do with the rest, and
choosing *when* to move the money costs something else. The cash-matched
baseline — a static portfolio starting at the same 50% equity, never touched —
separates them:

| what                              | ^GSPC, 10 windows |
| --------------------------------- | ----------------- |
| total vs buy-and-hold             | **-2.82% CAGR**   |
| of which **allocation** (in cash) | -2.34%            |
| of which **timing** (the rules)   | **-0.48%**        |

Eighty-three percent of the damage is simply not being invested. The part the
post is actually about — buy low, sell high, on those exact thresholds — is
-0.48% per year with a standard error that swallows it. And -0.48% turns out to
be the wrong thing to quote; see the design audit two sections down, which moves
it further than its own size.

Two configurations confirm this rather than one. Start fully invested, so the
allocation term is identically zero by construction, and the entire gap is
timing: **-1.24%, p = 0.166 raw and 0.996 after Bonferroni** over the six
variants. Cut the windows to five years for 19 observations instead of 10, and
the timing term is **+0.01%**. It is indistinguishable from zero in every
direction it is measured from.

So the rules are not a market-timing claim that fails. They are an **asset
allocation, arrived at accidentally, and described as discipline**. They do cut
maximum drawdown from 37.9% to 20.2%, which is what holding cash does, and
which a 50/50 static portfolio delivers with no rules and no attention.

### What is not modelled, and which way it cuts

* **Tax was the largest missing term and is now modelled.** See below: the
  ladder loses at every rate under both framings, so it was never
  load-bearing.
* **Dividends and bill yield were assumptions** — 3.5% and 3.3%, the long-run
  US averages, applied flat. They are the only two constants that could rig
  the comparison unnoticed, because the ladder spends a third of its life in
  cash. They have since been replaced with measurement; see below.
* **A fully invested holder cannot run these rules.** At `--start-equity 1.0`,
  42 buy rungs across ten windows fired asking for more cash than existed and
  were filled short. That is reported rather than silently clipped.
* **Index price series, not investable products.** No fund fee, no tracking
  error, and ^N225 is in yen.

### Auditing my own two choices, not the post's

Six variants cover the ambiguity in the post. They do not cover the ambiguity
I introduced. Nothing in those ten lines says **when a fired rung becomes
eligible again**, or whether the threshold is watched **at the close or
intraday** — I picked both, and a finding that survives only my picks is a
finding about me. Every run now prints the whole grid:

```
    re-arm    trigger        vs hold    timing       p   trades   in mkt  unfunded
    cross     close           -2.82%    -0.48%   0.036      6.4      41%         0 <-- default
    cross     intrabar        -2.60%    -0.26%   0.040      7.0      43%         0
    once      close           -2.80%    -0.46%   0.028      5.4      40%         0
    once      intrabar        -2.80%    -0.46%   0.028      5.4      41%         0
    always    close           -2.24%    +0.10%   0.036    685.0      51%     10901
    always    intrabar        -2.08%    +0.26%   0.057    703.0      51%     11135
```

`cross` re-arms a rung when price crosses back through its reference. `once`
fires each rung at most one time per window. `always` is the literal reading of
the sentence — fire on every bar the condition holds — and it is included
precisely because of what it does: **685 trades a window and 10,901 rungs that
asked for cash which did not exist**. That is not a strategy anyone ran; it is
what "drops 20% → buy" says if you read it as a state rather than an event.

**The timing term changes sign across my own choices: -0.48% to +0.26%.** That
is a 0.74% spread on a term whose largest value is 0.48%. There is no honest
way to quote a single number for it, including mine — that spread *is* the
result. What no effect looks like from the inside is exactly this: an estimate
that wanders further than its own magnitude when you vary something that should
not have mattered.

The allocation term does not do that. It stays between -2.08% and -2.82% in all
six configurations, and the "in mkt" column explains why it moves at all — the
configurations that lose less are simply the ones that end up holding more
stock. **The cash drag is doing the work in every configuration**, which is the
finding, and it is the one that survived being attacked.

`--rearm` and `--trigger` expose the choices, so nothing here depends on
believing the defaults.

### The two assumed rates, replaced with measurement

Two numbers in that test were guesses, and they were the two that could move
the answer without anyone noticing. `--measured-yields` replaces both with real
series from the vendor already in use:

* **Cash earns `^IRX`**, the 13-week Treasury bill, daily from 1960 and
  forward-filled across days it did not print. A day before the first print
  earns nothing rather than being back-filled with a guess.
* **Dividends come from the gap between `^SP500TR` and `^GSPC`.** The
  total-return index is the price index with dividends reinvested, so the ratio
  of their daily growth *is* the dividend accrual — measured, not assumed. That
  series starts in 1988, so everything is clipped to 1988-2026 rather than
  silently mixing measured and assumed rates inside one run.

Both guesses were wrong, in opposite directions:

| rate      | assumed | measured 1988-2026 |
| --------- | ------- | ------------------ |
| dividends | 3.5%    | **2.10%**          |
| bills     | 3.3%    | **4.35%**          |

Overstating dividends helps whichever arm holds more stock, and understating
bills hurts whichever arm holds more cash. Both errors therefore pushed against
the ladder, and the controlled re-run — *same windows, same symbols, only the
rates changed* — says by how much:

```
    symbol    variant             measured   assumed    move   timing: measured -> assumed
    ^GSPC     peak/initial          -4.10%    -4.59%  -0.49   -0.46% -> -0.68%
    ^IXIC     anchor/initial        -5.55%    -5.98%  -0.43   -0.73% -> -0.87%
    ^N225     peak/position         -1.52%    -1.86%  -0.34   -0.73% -> -0.61%
```

The assumptions were costing the ladder about half a percent a year of CAGR,
and the timing term — the part the post is about — barely moves: -0.46% with
real rates against -0.48% on the century with assumed ones. The conclusion is
the same one, now with one fewer thing taken on trust.

The comparison is run on the clipped span for both arms deliberately.
Measured rates over 1988-2026 against assumed rates over 1927-2026 would have
changed two things at once and credited the difference to the wrong one — a
mistake this document already records making twice.

One weakness remains in that setup: `^SP500TR` and `^GSPC` come from the same
vendor, so a systematic Yahoo error would cancel out of their ratio and never
be seen. The next section fixes that.

### A second vendor, and what it found

Stooq now sits behind a proof-of-work bot check and FRED did not answer, but
the Shiller S&P dataset is served as keyless CSV: monthly, from **January
1871**, compiled from the Cowles Commission data and S&P's own publications
rather than from Yahoo. `python run.py vendorcheck` puts Yahoo's daily `^GSPC`,
aggregated to months, against it.

The first result is the convention, which the code determines instead of
assuming — Shiller's column is a monthly **average** of daily closes, and
comparing it to a month-end close would have manufactured a 1.4% disagreement
that is not a vendor error at all:

```
    which convention is Shiller's SP500 column?
      monthly mean of closes       0.4 bps median difference   <-- fits
      month-end close            140.3 bps median difference
```

The second is agreement, across 1,183 months and eleven decades:

```
    decade       months     median      worst   over 100bps
    1930s           120      11.7b      72.2b             0
    1950s           120       1.1b      23.9b             0
    1970s           120       0.5b     442.5b             1
    1990s           120       0.0b       5.3b             0
    2020s            79       0.0b     236.2b             1
```

Median disagreement **0.4 bps**. The 1930s are the loosest decade at 11.7 bps,
which is what you would expect of a period being reconstructed from paper
records by two different compilers, and it is still a rounding error against a
20% threshold.

Two months break 100 bps, and rather than list them as unexplained the code
checks *which* of Yahoo's monthly statistics the other vendor's row actually
equals:

```
      1974-07       442.5 bps   = Yahoo's month-end close
      2023-09       236.2 bps   = Yahoo's first close of the month
```

Both are single rows in the monthly series carrying the wrong statistic —
neither is a price this project got wrong, and Yahoo is self-consistent in
both. That is the whole value of the check: it converts two unexplained
outliers into two closed questions. **The century of prices behind the ladder
result is not a Yahoo artifact.**

What it does not buy: anything about the daily path *between* month ends, which
is where the -20% thresholds actually fire. A monthly series cannot see an
intra-month drawdown. Nor anything about pre-1962 OHLC, which Yahoo reports
with open, high and low all equal to the close — so `--trigger intrabar` is
identical to `close` over most of the century, which is a limit of the data
rather than a result.

### Measured dividends back to 1960

Shiller also carries dividend-per-share from 1871, which is the half of the
century the `^SP500TR` path could not reach.
`--measured-yields --dividend-source shiller` uses it, and the binding
constraint moves from the dividend series to `^IRX`, so the measured span goes
from 1988-2026 to **1963-2023 — twelve windows instead of seven**, with a
measured 2.86% dividend yield and 4.35% bills:

```
    re-arm    trigger        vs hold    timing       p   trades   in mkt  unfunded
    cross     close           -1.92%    +0.51%   0.205      4.2      49%         0 <-- default
    cross     intrabar        -1.71%    +0.72%   0.263      4.6      51%         1
    once      close           -1.94%    +0.49%   0.199      4.1      49%         0
    once      intrabar        -1.84%    +0.59%   0.232      4.2      50%         0
    always    close           -1.66%    +0.77%   0.322    143.4      47%      2152
    always    intrabar        -1.33%    +1.10%   0.440    156.8      47%      2340
```

On this span the timing term is **positive** in all six configurations — and
not one of them is significant (p = 0.20 to 0.44). Set against the century's
-0.48% and the design audit's -0.48% to +0.26%, the timing term has now changed
sign under *three* separate things that should not have mattered: which
re-arm rule, which trigger, and which sixty years. That is not a weak effect
being hunted; it is the absence of one, viewed from three angles.

The allocation term did what it has done every time: stayed between -1.3% and
-2.9%, with the "in mkt" column explaining every wobble in it.

There is still no second source for the *daily* bars, only for monthly levels,
and the survivorship caveat stands unchanged.

### Tax, the term that was missing

Every table above the last one was computed at zero tax, and every one of them
said so — which is not the same as it being harmless. Zero tax is not neutral
here: the ladder realises gains on every sell, buy-and-hold defers them for a
century, so **every number was understating the ladder's disadvantage**. An
unmodelled term running in the same direction as the conclusion is the exact
shape of the bugs this repo has already found in its own code, and it sat
in the docstring as a disclaimer for as long as it took to write this section.

It is now modelled: realised gains and dividends are taxed at a flat rate or at
the historical era rates (25% before 1970, 35% through the 1970s, down to 15%
in 2003, 23.8% today), and both framings are printed.

```
    cgt rate       vs hold    timing       p  |     vs hold    timing       p
                  ---- never settled ----     |   --- settled at the end ---
    none            -2.82%    -0.48%   0.036  |      -2.80%    -0.47%   0.037
    15%             -2.95%    -0.75%   0.033  |      -2.42%    -0.45%   0.045
    23.8%           -3.03%    -0.92%   0.032  |      -2.17%    -0.43%   0.052
    era rates       -3.05%    -0.95%   0.031  |      -2.20%    -0.48%   0.049
```

"Never settled" is the buy-and-hold-forever reader: only the ladder ever pays a
bill. "Settled at the end" liquidates both arms on the last bar and taxes the
deferred gain, which is the only symmetric comparison — without it buy-and-hold
is being scored on money it has not been allowed to keep yet. Reporting only
the first would be picking the framing that agrees with the conclusion.

**The ladder loses at every rate under both framings**, so the missing term was
never load-bearing — it only made a losing arm lose by more. Under the
asymmetric framing the tax bill costs it a further -0.23% CAGR; under the
symmetric one buy-and-hold gives back more than the ladder does and the gap
*narrows* to -2.20%, still losing.

Two simplifications, both stated because both favour the ladder — the arm this
section concludes against:

* **Average cost basis**, not specific-lot or FIFO. Real sellers pick lots to
  minimise the bill; average cost is the neutral middle.
* **The long-term rate always.** Rungs usually fire years apart so that is
  mostly right, but `--rearm always` trades daily and would owe the much higher
  short-term rate. Charging it long-term flatters it.

Losses are not carried forward either, which is the one simplification running
the other way.

### Markets that collapsed, and a prediction that did not survive

Everything above runs on markets that went up: the S&P, the Nasdaq, Tokyo,
bitcoin. Concluding "holding cash costs you" from three winners is the same
error as concluding a strategy works from three winners, and it is the half of
the survivorship problem free data can actually reach. `--preset collapsed`
runs four markets that fell hard and did not come back inside the sample:
**Athens -93% from its peak**, Shanghai -72%, Hong Kong -65%, Tokyo.

The prediction, written into the module before the first run: *here the ladder
should beat buy-and-hold, because holding cash through a collapse is the right
thing to have done.*

**Falsified.** It loses in all four, Athens included:

| market       | windows | best variant vs hold | timing |
| ------------ | ------: | -------------------: | -----: |
| GD.AT        | 6       | -1.52%               | -0.40% |
| 000001.SS    | 6       | -1.27%               | +0.78% |
| ^HSI         | 8       | -0.59%               | +2.07% |
| ^N225        | 12      | -3.75%               | -0.87% |

The error was in the question, not the arithmetic: **a 93% drawdown is not a
93% sample**. Cut into non-overlapping five-year windows, most windows of even
a collapsed market are rising ones, and the ladder is only ~40% invested in all
of them. The right question was where the wins and losses live, and that is now
printed:

```
  WHERE THE LADDER WINS AND LOSES -- GD.AT, basis/position
    windows where          n    vs hold     timing
    the market rose        4     -7.42%     -1.55%
    the market fell        2    +10.28%     +1.90%
```

Holding cash costs 7.4% a year when Athens rose and saves 10.3% when it fell.
That is what holding cash does, and it needs no rules to do it. The timing term
wanders between -1.55% and +1.90% — **whatever the market did, the rules
themselves did nothing**.

That is the strongest version of the finding available, because buy-the-dip is
sold precisely for this environment, and this is where it most clearly adds
nothing beyond the cash it happens to be holding.

One labelling note this section forced: the baseline CAGR printed per symbol is
the **mean of the per-window figures**, not the return of someone who held
across the whole span. On a market that round-tripped, those two differ a lot,
and the header now says so.

### The invariant that found the eighth bug

Every fixture in this file tests a case I thought of. The seven earlier bugs in
this project were cases I had not, and the ones that lasted longest produced a
plausible number. So the ladder is also checked against relationships that must
hold on **any** price path — random walks, not hand-built fixtures — and the
strongest of them is this: with the rungs removed, the entire ladder machine
(references, re-arming, sizing, era costs, tax, accrual, terminal settlement)
must reduce **exactly** to `simulate_hold`, a function that shares none of that
code.

It did not. It was out by $1.19 on $10,000.

`simulate_ladder` counted the entry commission as part of the cost basis, which
is correct, and `simulate_hold` did not. Buy-and-hold therefore reported a
slightly larger gain than it had, and overpaid on liquidation — 23.8% of 5 bps,
which is exactly the 1.19. A drift between the two *arms of the comparison* is
the one error that can manufacture the finding rather than merely blur it, and
it was invisible until a property demanded the two functions agree to the cent.

It only bit when tax and terminal liquidation were both on, so the headline
numbers never moved; the "settled at the end" column above is the corrected
one. Six further invariants ride along: tax can only ever subtract, trade
counts must order `once <= cross <= always`, no configuration may borrow or go
short (108 checked per path), the simulation must be deterministic, intrabar
triggering must see everything close triggering sees, and windows must
partition the history without inventing or duplicating a bar.

### Why this one was worth the file

Eleven improvements to the crypto strategy died out of sample and the twelfth
was never going to be different. What this adds is the thing the rest of the
repo cannot buy at any price: **a sample where the answer is not
sample-limited**. Ten non-overlapping decades, three indices, six readings,
two window lengths and two allocation baselines all say the same thing, and
they say it about the specific mechanism rather than about the outcome. That
is what separates "the rules underperform" from "the rules are an allocation
wearing a timing costume", and only the second one tells a reader what to do
next.

The viewer shows the split rather than the total, for the same reason. A row
reading "-2.82%/yr behind buy-and-hold" would be true and would tell a reader
the wrong thing; the row reads `vs buy-and-hold -2.82%/yr · of which timing
-0.48%/yr · of which cash -2.34%/yr`. The ladder travels in its own key in
`findings.json` and is never folded into the crypto headline, because one is a
per-trade edge and the other is a per-year allocation, and averaging them would
answer a question nobody asked.

It also found a bug of the usual kind. `datetime.fromtimestamp` raises OSError
on any pre-1970 timestamp on Windows, and `yahoo.py` capped daily history at
10,000 days. Together those would have silently turned "does this survive the
1930s" into "does this survive the 2010s" — the interesting nine-tenths of the
sample discarded by a platform quirk and a constant, with no error and no
warning. Same disease as every other entry in this document.

## The long-horizon bot, after the short-horizon one failed

Everything above is a record of an intraday strategy not working. The one thing
every measurement agreed on is that **being invested paid and the rules did
not**. So the last question is whether that survives a change of horizon: over
decades rather than minutes, does any rule beat holding the index?

`python run.py longterm`. Two candidates, picked because they have published
support rather than a screenshot — deliberately the opposite of how hypotheses
were chosen earlier in this repo:

* **`sma`** — Faber's trend rule: hold the index while it is above its own
  200-day average, sit in bills below it, checked monthly. The 200 comes from
  the paper, not from this data.
* **`cape`** — valuation: Shiller's PE10 turned into a weight by its
  **point-in-time** percentile against its own past. Ranking against the full
  series would know in 1929 what 1999 looked like.
* **`blend`** — the average of the two, since they disagree often.

### The control that decides it

Carried straight over from the ladder: a rule that ends up 70% invested must be
compared against a **static 70% portfolio**, not against 100% equity, or the
cash it happens to hold gets scored as skill. Every strategy is measured twice
— against buy-and-hold, and against a static portfolio rebalanced to that
strategy's *own realised average weight*. Only the second number is about the
rule.

### ^GSPC, six non-overlapping 15-year windows, 1936-2026

```
    buy-and-hold: +10.70% CAGR, 40.3% max drawdown

    strategy     vs hold      p    win    timing      p   max DD  in mkt  trades
    hold          +0.00%  1.000   0/6     +0.16%  0.072    40.3%     97%     8.7
    sma           -1.11%  0.420   1/6     +0.40%  0.734    24.1%     70%    23.7
    blend         -2.44%  0.085   2/6     +0.42%  0.433    17.0%     55%    37.0
    cape          -4.31%  0.027   1/6     +0.25%  0.512    22.3%     38%    36.7
```

All four predictions, written into the module before the first run, held:

1. **The trend rule cuts drawdown.** 40.3% → 24.1%. It does the same on the
   Nasdaq (47.5% → 36.0%), Tokyo (44.8% → 27.9%) and Hong Kong (55.8% → 39.7%).
   This is the best-documented part of the trend literature and it survived.
2. **It loses on total return after costs.** -1.11% CAGR, and -1.82% / -0.92%
   on the Nasdaq and Tokyo.
3. **Its timing term is positive but not significant.** +0.40% at p = 0.73.
4. **CAPE is worse than trend on both.** -4.31% vs buy-and-hold, and it spends
   the sample 38% invested because US valuations have been in their own upper
   percentiles for thirty years.

### The null nobody built

The `hold` row follows no rule, and still scores **+0.16%** on the timing
metric — that is what the rebalance band alone is worth, cash drifting between
monthly checks. Nothing designed it as a control; it fell out of running the
do-nothing strategy through the same machine. Any rule has to clear *that*, not
zero.

Against the null, the trend rule adds **+0.24% CAGR**, with a 95% interval
several times wider than the estimate, on six observations.

### What the long-horizon bot actually is

The same answer the rest of the repo gives, at a horizon where it is finally
good news rather than bad:

> **Stay invested. If you want half the drawdown and will pay about a point of
> CAGR for it, add the 200-day trend filter. Nothing here picks *when* — the
> trend rule earns its keep by changing *how much*, and so does everything
> else.**

That is a real bot and it is running the only strategy in this repository that
survived measurement. It is also, deliberately, close to doing nothing, and the
gap between "close to nothing" and "nothing" is +0.24% CAGR that six windows
cannot separate from zero.

Not modelled: **contributions**. A monthly paycheque changes the comparison
from time-weighted to money-weighted returns, and mixing the two is how a DCA
backtest ends up flattering whichever arm received cash when prices were low.
Lump sum only, stated rather than quietly assumed.

## The multi-asset search, run as a pre-registered experiment

The brief was "study as hard as you can and produce something positive".
Searching until something comes out positive is p-hacking with extra steps, and
this document is a record of what that produces. So the search was run the
other way round: wide, with the holdout sealed, and reported whatever it came
out as.

**The protocol, fixed before the first run.** Split the sample once at its
halfway point — search set to 1993, holdout from 1994. Score every strategy on
the search set. Take the single best by search-set Sharpe; that choice is final.
Judge it once on the holdout. Multiply the p-value by the number of candidates.
Declare success only if it beats 100% equity on **holdout** Sharpe at a
corrected p below 0.05.

### The data had to be built, so it was validated first

Testing diversification needs bonds, and free daily bond history does not
exist. `bot/basket.py` builds a rolling par bond from its own yield —
`r = y/12 - D·dy + C·dy²/2` — off month-end `^TNX`, then `run.py basketcheck`
holds it against real funds:

```
    series    fund       n  built %/yr  fund %/yr  built vol  fund vol    corr
    equity    SPY      403     +11.43%    +11.48%     14.76%    14.73%   0.998
    bond      IEF      289      +2.95%     +3.67%      6.51%     6.62%   0.989
```

Two things that check found before they could do damage:

* The first version compared a total-return series against **price-only** fund
  closes, because the project's Yahoo loader defaults to unadjusted. Both funds
  came out 2-3%/yr "below" the built series. That was not an error in the
  series, it was the benchmark's dividends going missing, and it would have
  made every bond-heavy strategy look better than it was. The loader now takes
  `auto_adjust`, with its own cache key.
* Shiller carries a long rate back to **1871** and the same formula runs on it,
  so the sample could have been a century instead of sixty-four years. It is
  not, because Shiller's rate is a monthly *average* — the same convention
  `vendorcheck` found in his price column — and over 1985-2023 the
  average-based series reproduces the mean (+6.03% against +5.83%) while
  **understating volatility** (6.43% against 7.62%). Understated bond
  volatility flatters exactly the risk-adjusted claim under test. Ninety extra
  years of evidence bent toward the conclusion is worse than sixty-four honest
  ones.

Roll-down is not modelled, which leaves the synthetic bond ~0.7%/yr *below*
IEF — against the diversification claim, which is the right direction.

### Search set, 1962-1993

```
    strategy         CAGR     vol   Sharpe   max DD  Calmar  equity   bond  turn/yr
    sixty40        +9.29%   10.1%    0.296    26.3%    0.35     60%    39%    0.08x
    equity        +10.12%   14.9%    0.294    42.7%    0.24    100%     0%    0.03x
    riskparity     +7.73%    7.8%    0.174    12.7%    0.61     30%    68%    0.30x
    tsmom          +7.71%    8.6%    0.164    18.0%    0.43     48%    35%    0.95x
    rp_tsmom       +6.65%    6.7%    0.041    12.7%    0.52     24%    60%    1.07x
```

`sixty40` wins by 0.002 Sharpe. The choice was made on that and locked.

### Holdout, 1994-2026 — opened once

```
    strategy         CAGR     vol   Sharpe   max DD  Calmar  equity   bond  turn/yr
    tsmom          +8.69%    7.6%    0.817    12.3%    0.71     51%    29%    0.67x
    rp_tsmom       +6.36%    5.2%    0.742     7.3%    0.87     28%    50%    0.81x
    sixty40        +8.46%    9.1%    0.672    28.1%    0.30     61%    38%    0.11x
    riskparity     +6.35%    6.0%    0.649    17.3%    0.37     33%    66%    0.28x
    rp_levered     +9.68%   12.0%    0.630    35.3%    0.27     65%   131%    0.92x
    sixty40_volmatch  +10.97%  15.0%   0.609    51.2%    0.21     97%    64%    0.55x
    equity        +10.74%   14.9%    0.598    50.9%    0.21    100%     0%    0.03x
    bond           +3.81%    6.4%    0.232    23.0%    0.17      0%   100%    0.03x
```

**The pre-registered answer: +0.075 Sharpe, p = 0.166 raw and 1.000 after
Bonferroni.** The 95% block-bootstrap interval is [-0.042, +0.214] and contains
zero. It is stable across bootstrap seeds (p = 0.162 to 0.175). By the rule
written in advance, this is **not** a positive result.

### The three things worth taking from it anyway

**1. This sample could never have proved it.** The interval's half-width is
0.128 Sharpe over 33 years. An effect of +0.075 needs about **96 years of
holdout** to clear zero, before any multiplicity correction. "Not significant"
here is a fact about the question, not about the answer — and knowing which of
the two you are looking at is most of what this repository is for.

**2. Eight of eight candidates beat equity's holdout Sharpe.** That is not a
test — they are variations on one idea and heavily correlated, so it cannot be
scored as eight independent coin flips — but it is not nothing either, and it
is reported so nobody has to take the pre-registration on trust.

**3. Drawdown is separable and the answer is large.** 28.1% against 50.9%. That
needs no p-value because nobody is arguing 23 percentage points of the worst
thing that happened might be noise.

### At equal risk, it is a tie

The sharpest version of the question: lever the diversified mix up to the
index's *own* volatility and compare money, not ratios. `sixty40_volmatch` does
that and lands at **+10.97% CAGR at 15.0% vol against equity's +10.74% at
14.9%** — twenty-three basis points a year, for four times the turnover and a
margin loan. Risk parity's promise, measured, is a rounding error.

**Disclosure: the two `_volmatch` strategies were added after a first complete
run had been read.** The holdout is no longer virgin — it has been looked at
once, and what to add next was chosen by someone who had seen it. That is a
weaker experiment than the first one. It is included because the equal-risk
comparison is the one the diversification argument actually makes, and leaving
it out to protect a clean protocol would have been protecting the protocol
rather than the reader.

### Two bugs the tests caught, both mine

* The bootstrap's p-value was `sum(...) or sum(...)`, which fell through to the
  second sum whenever the first came out zero — that is, precisely when the
  effect was strongest, where it reported the p-value of the opposite
  hypothesis. **It was wrong in the direction that manufactures significance.**
* A rebalance to 100% cannot buy 100%, because the commission comes out of the
  same cash. The book went a fraction of a percent short on every full
  investment and quietly paid the borrowing spread. The first fix over-corrected
  and capped every buy at available cash, which silently disabled leverage and
  turned `rp_levered` into a synonym for `riskparity`; the cap now applies only
  to strategies that are not asking to borrow.

### So, is it positive?

On the terms set in advance: **no.** On total return: no — the diversified
portfolio earns 2.3 points a year less than the index, and matching its risk
recovers only 0.23 of that. On drawdown: yes, decisively, and that is the only
claim here that the data can actually carry.

Which is the same sentence the rest of this repository keeps arriving at, now
in its most useful form: **nothing on offer picks when. The things that work
change how much, and you pay for them in return.**

## Trend across sixteen markets: the replication, and what it caught

The two-asset test ended with `tsmom` posting the highest holdout Sharpe of
anything tried — 0.817 against equity's 0.598. That number is not evidence: it
was the best of eight on a set that had been searched. But it is also the only
row in that table with a large independent literature behind it, and the right
response to "the best row was the one with prior support" is neither to
celebrate it nor to bin it. It is to test it where it was not chosen.

`python run.py trendpanel`. Fifteen markets, none of them the pair the first
test used: nine non-US and non-S&P equity indices, the 30-year yield, five
commodities, the dollar. The 12-month lookback comes from Moskowitz-Ooi-Pedersen,
not from this data — nothing is fitted, so there is nothing to overfit. And one
comparison fixed in advance: the trend book against **the identical book with
the signal removed** — same markets, same inverse-volatility weights, same
target volatility, same costs, always on.

```
    portfolio        CAGR     vol   Sharpe   max DD   gross  turn/yr
    trend         +10.59%   12.7%    0.541    28.8%    126%    4.05x
    always on      +9.11%   12.5%    0.417    44.4%    136%    1.50x

    difference     +0.123 Sharpe    95% CI [-0.021, +0.300]    p = 0.102
```

Positive, not significant, and it survives the dividend assumption — at an
assumed 4.5% on the index legs, which helps the always-on book more because it
is long all the time, the gap is still +0.113.

At which point the interesting part starts.

### Where the effect actually lives

```
    group           trend   always on   difference   panel without it
    currency       -0.171      -0.445       +0.274             +0.009
    rates           0.248       0.237       +0.012             +0.112
    equity          0.517       0.523       -0.006             +0.208
    commodity       0.299       0.389       -0.090             +0.144
```

**Take the dollar out and 93% of the effect goes with it.** The panel drops
from +0.123 to +0.009. On the nine equity indices — the markets with the most
data and the most at stake — trend is *worse* than always-on by 0.006. On
commodities it is worse by 0.090.

And look at *why* currency scores +0.274: the always-on control there has a
Sharpe of **-0.445**. Being permanently long a dollar index is not a strategy
anyone runs; it is a benchmark that loses money by construction. Any rule that
sometimes sits out beats it.

That is the general lesson, and it is worth more than the number: **trend
following looks best exactly where buy-and-hold is a bad benchmark.** On assets
with a real risk premium — equities — it adds nothing. The 0.817 in the
two-asset table was the best of eight on a searched set, and this is what that
usually turns out to mean.

### Two diagnostics that had to be fixed to see it

* The group table originally **skipped any group with fewer than two markets**,
  "for having too few members". That hid the entire finding: the effect lived
  in a one-market group that was never printed. A diagnostic that drops the
  small cases drops the interesting ones.
* The leave-one-out check then declared "survives dropping any single group"
  because +0.009 is still above zero. A sign test is too blunt when the
  question is *how much is left*; it now reports the fraction retained, and
  7% is not survival.

Both were mine, both were in the direction of the conclusion I was hoping for,
and neither would have been visible in the headline number.

## Install

```bash
git clone https://github.com/RaulJimenezAyza/momentum-edge-study
cd momentum-edge-study
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[ml,dev]"      # or: pip install -r requirements.txt
```

Python 3.12+. Paper mode needs no `.env`. The `ml` extra (`numpy`/`scikit-learn`/`joblib`) is
only used by `run.py train` and `ml.enabled`, and `equities` (`yfinance`) only by the US-equity
commands — the bot and the viewer run without them.

## Tests

```bash
pytest -q                       # 51 tests, ~490 checks, no network, ~15 s
python tests/test_offline.py    # same suite, printing every individual check
ruff check .
```

## Use

```bash
python run.py universe              # which symbols would be watched
python run.py scan                  # one-shot: any signals right now?
python run.py backtest --days 7     # replay a week through the live logic
python run.py live                  # run it + open the viewer (paper mode)

python run.py nullcheck --days 30   # do the entry rules beat random entries?
python run.py sweep --days 30       # walk-forward search over 108 settings
python run.py experiments --days 30 # every direction x timeframe x stop-mode

# holdout discipline: search on an older window, judge the winner once on the
# window the search never saw
python run.py experiments --days 45 --offset 14
python run.py edgecurve   --days 45 --offset 14
python run.py markets     --size 150 --min-volume 500000
python run.py exitfit     --days 45 --offset 14
python run.py spreadcheck --days 7
python run.py regime      --days 45 --offset 14
python run.py replicate   --windows 5 --days 30 --point-in-time
python run.py reconcile
python run.py assetclass  --days 30
python run.py equities    --days 59 --interval 5m
python run.py maker       --days 30
python run.py nullcheck   --days 14 --offset 0

# the viral buy-the-dip / sell-the-rally ladder, over ten non-overlapping
# decades of ^GSPC, ^IXIC and ^N225
python run.py ladder
python run.py ladder --symbols ^GSPC --start-equity 1.0   # the timing term alone
python run.py ladder --window-years 5                     # 19 windows, not 10
python run.py ladder --measured-yields --window-years 5   # real bill and
                                                          # dividend series
python run.py ladder --rearm once --trigger intrabar      # my own choices,
                                                          # overridden
python run.py ladder --measured-yields --dividend-source shiller
python run.py ladder --tax-rate -1 --liquidate   # historical cgt rates, both
                                                 # arms settling at the end
python run.py ladder --preset collapsed --window-years 5   # Athens -93%,
                                                 # Shanghai, Hong Kong, Tokyo
python run.py vendorcheck            # does a second vendor agree about the
                                     # century of prices the ladder rests on?

# the long-horizon bot: 200-day trend and CAPE, each against its own
# weight-matched static baseline
python run.py longterm
python run.py longterm --tax-rate -1 --liquidate --measured-yields

# multi-asset: validate the built series, then run the pre-registered search
python run.py basketcheck
python run.py allocate
python run.py allocate --tax-rate -1 --seed 11
python run.py trendpanel             # does the trend result replicate on
                                     # markets it was not chosen on?
python run.py train --days 30       # fit the ML filter and report on it
python run.py model                 # show the saved model's metrics

python run.py backtest --days 30 --symbols BTCUSDT ETHUSDT SOLUSDT
python run.py live --log-level DEBUG
```

Ctrl-C stops the live bot, closes open paper positions and prints the summary.

```bash
python tests/test_offline.py        # 486 checks, no network needed
```

Closed candles never change, so every download is cached under `data/cache`.
The first `backtest --days 30` takes a few minutes; every run after it is
instant, and the cache detects and refetches its own gaps rather than serving
a holed window as complete. `cache_enabled: false` forces a fresh download.

## The viewer

`python run.py live` starts a read-only web page at **http://127.0.0.1:8787**.
It is read-only by design: no endpoint on it can place, cancel or size a trade.

The panel that matters is the **watchlist**. Most of the time a momentum bot
does nothing, and from a console you cannot tell "nothing is happening" apart
from "the bot is broken". The watchlist shows, per symbol and updated every
closed minute:

The **account-value chart draws two lines**: what the account is worth, and
what it would be worth if trading were free. The gap between them is the fee
bill, and on every window measured it is larger than everything the strategy
earns. Underneath, a **Where the money went** panel puts gross P&L, fees and
net on one shared scale, so the fee bar visibly dwarfs the gross bar.

| Column | Meaning |
|---|---|
| Ret 3m / Z | the move, raw and in std-devs of that symbol's own returns |
| Vol × | this minute's volume against its recent mean |
| To breakout | how far below the 30-bar high the price still is (`above` = cleared) |
| Ext / Spread | distance from the mean, and the current bid/ask cost |
| Heat | 0–1 bar: how far along the three continuous gates this symbol is |
| Blocked by | the exact gate holding it back, named |
| Last 40m | sparkline of the recent closes, so the shape is visible not just the numbers |

Sorted by heat, so whatever is closest to firing is at the top. Filter by
symbol, or narrow to `heat ≥ 0.66` / `all-gates-open` to watch only the ones
about to go.

Below it: open positions with a stop→target track and a live marker, closed
trades, the signals that fired, a running count of what got filtered out and
why, and the log tail. The account-value chart has a crosshair and tooltip.

The header carries a **stream-health badge**. A bot quietly trading on a holed
price series looks identical to a healthy one from the numbers alone, so the
gap age, stale-symbol count and repair count get their own indicator.

`pause` freezes the page without touching the bot. `theme` cycles
system / light / dark and is remembered. The heat ramp is a single-hue ordinal
scale validated for contrast against both surfaces.

## The learned filter

**What it is:** for each candidate breakout, a gradient-boosted model estimates
P(this trade closes net positive after fees). Candidates below a probability
cut-off are skipped.

**What it is not:** a price predictor. It never forecasts a number. It ranks
setups the rules already flagged — a far smaller problem, and the only shape of
"AI trading" that has ever reliably worked.

```bash
python run.py train --days 30       # download, label, fit, report
python run.py model                 # re-read the report later
# then, if and only if the verdict justifies it:
#   ml.enabled: true  in config.yaml
```

Six deliberate choices stop it from lying to you:

1. **Labels come from `simulate_exit`** — the same exit rules the live bot runs.
   A "win" in training is a win the bot could actually have taken.
   `tests/test_offline.py` asserts the two implementations agree on randomised
   paths, so they cannot silently drift apart.
2. **Unresolved trades are dropped**, never truncated to a partial result.
3. **The split is by time**, never random. A random split lets the model see the
   future of the same market session and inflates every metric.
4. **An embargo gap** sits between train and test, so a trade still open at the
   boundary cannot leak its outcome across.
5. **Probabilities are calibrated** (isotonic), because an uncalibrated 0.6 is
   not 60%.
6. **The cut-off is derived from the training distribution**, not hand-picked.
   The base rate here is 33%, so a calibrated model almost never emits anything
   above 0.5 — a plausible-sounding 0.55 keeps *zero* trades. `run.py train`
   prints a sweep of cut-offs and uses the one matching
   `ml.target_keep_fraction`, chosen before seeing test results.

With ML on, the hand-tuned thresholds are loosened by `ml.relax_factor` so the
model has a wider pool to choose from than the rules alone would take. The
symbols in play are the same; BTC is added to the stream for market-regime
features whether or not it is traded.

### What the model actually learned

30 days, 50 pairs, 14,162 labelled candidates, 32.5% of them profitable:

```
ROC AUC                            0.6056     (0.50 = coin flip)
Brier score                        0.2308

  target   cut-off    kept      n   expectancy
     50%     0.327   66.9%   2841     -0.2105%
     35%     0.371   54.3%   2309     -0.1971%
     25%     0.398   43.7%   1857     -0.2065%   <-- used
     15%     0.423   29.6%   1256     -0.1635%
      8%     0.436   19.2%    815     -0.1508%

VERDICT: NOT ENOUGH -- filter helps but expectancy is still negative
```

**AUC 0.6056 is a real signal.** On 1-minute crypto data that is a genuine,
unremarkable-but-nonzero edge, and it behaves the way a real edge should: the
stricter the filter, the better the surviving trades (−0.21% → −0.15%).

**It is still not enough.** An edge of that size cannot cover a 0.30% round-trip
cost on a strategy whose gross expectancy is already around zero. The trainer
says so in the verdict line rather than showing you a nice-looking AUC and
letting you infer profitability. `ml.enabled` stays `false` by default for that
reason.

The features that carried the most weight (permutation importance on held-out
data) are `sigma_3` (the symbol's own volatility), `dist_low_120` (where price
sits in its recent range), `green_10` (persistence) and `market_ret_60` (BTC
regime). Not the ones the entry rules key on — which is the point of having it.

**Caveat on ML backtests:** `run.py backtest` with `ml.enabled: true` scores
bars the model may have trained on. Use it to check the code path, not to
measure edge — the held-out numbers above are the honest ones.

Output lands in `data/`: `trades.csv`, `equity.csv`, `signals.csv`
(`backtest_*.csv` for backtests).

## The strategy

A candle triggers an entry only when **all** of these hold on the just-closed
1m bar:

| Gate | Default | What it rules out |
|---|---|---|
| Return over 3 bars | ≥ 0.6% | Noise-level moves |
| Same return in std-devs | ≥ 2.5σ | 0.6% is huge for BTC, nothing for a microcap |
| Volume vs window mean | ≥ 3× | Price drifting on an empty book |
| Close above 30-bar high | required | Chop inside a range |
| Distance above 30-bar mean | ≤ 6% | Buying the top of a spent move |
| Bid/ask spread | ≤ 8 bps | Pairs where the spread eats the edge |

`strategy.direction: reversal` mirrors every gate — a violent move *down*
through the recent low, bought on the bet that it overshot. Same tests, sign
flipped. It is the natural alternative hypothesis once momentum turns out to
have no gross edge.

Exits: stop at −1.2%, target at +2.4%, trailing stop of 0.8% that arms once the
position is +1.2% up, and a 30-minute time stop.

`risk.stop_mode: atr` replaces those flat percentages with multiples of each
symbol's own average true range. Entries are volatility-normalised by the
z-score gate, so leaving exits on a flat percentage was an inconsistency: 1.2%
is a hair-trigger on BTC and a loose stop on a microcap. Bounded by
`min_stop_pct`/`max_stop_pct`, and the risk:reward ratio is preserved when a
bound bites.

`strategy.timeframe` accepts 1m/3m/5m/15m/1h. Nothing in the code assumes a
candle is a minute long — `max_hold_minutes` is converted to bars, and the time
stop fires on bar count.

Signals are computed on **closed** candles only, so they never repaint. Stops
and targets are checked against the live candle's high/low.

## Risk controls

- Size from risk, not gut: `equity × risk_per_trade / stop_distance`, capped at
  20% of equity per position.
- Max 5 concurrent positions, shared equity — a sixth signal is skipped, not funded.
- Daily loss limit: −5% in a UTC day halts new entries until it rolls over.
- 15-minute per-symbol cooldown after any exit.
- Order size respects each symbol's `LOT_SIZE` / `NOTIONAL` filters.
- `take_profit_pct` must exceed round-trip cost or the config refuses to load.
- A websocket reconnect loses every message sent while it was down. Any symbol
  silent for 3 minutes has its whole window refetched from REST — statistics
  computed across a hole are not merely stale, they are wrong while looking
  fine. The repair rebuilds the series rather than appending to it.

Tune every number in `config.yaml`.

## Honest limits of the backtest

Read `bot/backtest.py` — the limitations are documented at the top of the file:

- Fills come from candle OHLC, not the order book. Real slippage on a breakout
  is worse than the flat 5 bps assumed.
- Within one candle the high/low ordering is unknown; the code assumes the stop
  hit first, which is the pessimistic choice.
- The universe is today's volume ranking, so the test only sees coins that
  survived and stayed liquid — **survivorship bias**, and it flatters results.
- Under ~30 trades, the numbers mean nothing. The backtester warns you.

Paper mode has none of the fill problems but also never moves the market, which
a real order does.

## Going live (don't rush)

1. Weeks of paper mode. Judge expectancy and max drawdown, not the equity line.
2. `mode: testnet` with keys from https://testnet.binance.vision — exercises the
   real order path with fake money.
3. Only then `mode: live`, with `TRADING_LIVE_CONFIRM=YES`, spot-trading-only
   API keys, IP whitelist, **withdrawals disabled**, and money you can lose.

Note: `mode: live` currently reuses `PaperBroker`. Signing and submitting real
orders is deliberately not wired up — that's the piece you should write
yourself, once, while paying attention.

Crypto trading loses money for most people who try it. This is code, not advice.

## What would actually be needed

Five independent tests, so the conclusion is not a fluke of one setup:

| Test | Question | Answer |
|---|---|---|
| `backtest` | Does it make money? | No: −21% in 7 days, fees −186 vs gross −24 |
| `sweep` | Would other thresholds? | No: 0 of 108 profitable, walk-forward flat |
| `nullcheck` | Do the gates beat random? | No: p = 0.13, point estimate slightly worse |
| `train` | Can a model rank the setups? | Partly: AUC 0.606, not enough to cover costs |
| `experiments` | Is another structure better? | Yes — reversal, by 1 bp. Costs are 20 bp. |

Everything that showed a real effect showed one about a hundred times too
small. AUC 0.606 is genuine structure. The reversal edge is genuine and
significant at p < 0.001. Both are dwarfed by the toll.

The levers that remain are about cost and turnover, not smarter entries:

- **`fees.bnb_discount: true`** — 25% off both fee legs, 0.30% → 0.25% round
  trip. Bigger than every entry rule in this repo combined.
- **Longer timeframes and holds** — the same edge over fewer round trips.
  `run.py experiments` measures it; on this data 15m did not rescue it either.
- **Maker orders** instead of market orders. Not modelled here: doing it
  honestly needs order-book data plus a non-fill assumption, and that is
  exactly the assumption that turns a backtest into fiction.
- **A different edge entirely** — funding-rate arbitrage, market making,
  cross-exchange latency. Different projects, not a tweak to this one.

What is *not* on the list: another indicator, another gate, a bigger neural
network. The null test showed the six existing gates contribute nothing; the
problem was never that there were too few of them.

## Layout

```
run.py                    CLI
config.yaml               every tunable

bot/config.py             load + validate
bot/exchange.py           Binance REST and websocket
bot/universe.py           symbol selection
bot/series.py             rolling window and statistics
bot/signals.py            entry gates (fast path + full diagnostics)
bot/strategy.py           entries/exits, shared by live and backtest
bot/simulate.py           the same exit rules as a pure function, for labelling
bot/broker.py             simulated fills, fees, slippage
bot/risk.py               sizing, limits, kill switch
bot/engine.py             live loop
bot/backtest.py           historical replay
bot/levels.py             where the stop/target/trail go (fixed or ATR)
bot/spread.py             per-symbol spread estimator (Abdi-Ranaldo 2017)
bot/costs.py              what one round trip costs, per symbol
bot/sweep.py              walk-forward parameter search
bot/nullcheck.py          strategy vs random entries, with a t-test
bot/edgecurve.py          does the edge grow with dislocation size?
bot/markets.py            does widening the universe help, by liquidity tier?
bot/exitfit.py            signal shape (MFE/MAE) and walk-forward exit fitting
bot/spreadcheck.py        estimator vs live quotes -- is the cost model honest?
bot/regime.py             does the edge depend on volatility or time of day?
bot/replicate.py          non-overlapping windows, pooled by inverse variance
bot/pointintime.py        rank the universe on what was liquid then
bot/reconcile.py          live session vs a replay of the same minutes
bot/assets.py             crypto / tokenized equity / metal classifier
bot/assetclass.py         the null test per instrument type
bot/yahoo.py              US equities and ETFs via Yahoo Finance
bot/sessions.py           overnight gaps: contiguity and exit reachability
bot/equities.py           the null test on real US stocks
bot/ladder.py             the viral buy-the-dip ladder vs buy-and-hold, 1927-
bot/shiller.py            the Shiller S&P monthly series, 1871-, second vendor
bot/vendors.py            do two vendors agree about the same century?
bot/longterm.py           the long-horizon bot: trend and valuation vs holding
bot/basket.py             monthly total returns for stocks, bonds and bills
bot/allocate.py           the multi-asset search, pre-registered with a holdout
bot/trendpanel.py         trend across sixteen markets, as a replication
bot/maker.py              resting-limit entry, and the selection it buys
bot/experiments.py        direction x timeframe x stop-mode matrix
bot/cache.py              on-disk kline cache with gap detection
bot/logbook.py            CSVs and the performance summary

bot/dashboard.py          the viewer's server and state snapshot
bot/static/dashboard.html the viewer itself, one self-contained file

bot/features.py           feature vector, shared by training and inference
bot/ml.py                 dataset, model, walk-forward report
bot/trainer.py            the train / model commands

tests/test_offline.py     486 offline checks, runnable as a script or with pytest
tests/conftest.py         turns a failed check into a failed pytest test
pyproject.toml            package metadata, extras, pytest and ruff config
```
