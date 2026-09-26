# Market Structure Trainer — brief

A training game that teaches reading market structure — trend state and trend
change — on **MNQ**. A rule-based tagger labels the charts automatically, so no
hand tagging is needed, and that same tagger is what the automated system will
later classify with.

Instrument: **MNQ 12-26** for trading, **MNQ continuous** for the history the
trainer learns on (see *Data*). Tick size 0.25 index points, $0.50 a tick.

---

## Core idea

1. Simplify price into straight segments — piecewise linear segmentation through
   bar midpoints, (H+L)/2.
2. Take swing highs and lows from the segment endpoints.
3. Classify the structure with explicit Dow-theory / market-structure rules.
4. The game hides the future, asks for a classification, then reveals what
   happened and scores it.

`Indicators/Segments/PriceSegments.cs` already does step 1 on the chart: the
three segmentation methods, the ATR-scaled error and the vertices. It stays the
visual check — the shape on the chart the Python port has to agree with.

---

## Components

### 1. Data

MNQ has no Binance-style free endpoint, and a single contract is no use for
training: 12-26 only became front month at the September 2026 roll, so it holds
weeks, not years. The trainer needs a **back-adjusted continuous series**,
stitched across contracts, with the traded contract kept separate.

- Source: export from NinjaTrader (it already holds the history) —
  Tools > Historical Data > Export, or a small NinjaScript that writes CSV.
- Timeframes: 15m / 1H / 4H / 1D. Store as Parquet, one file per timeframe.
- Futures are not BTC, and the brief quietly assumed 24/7. What changes:
  - the 17:00 CT break and overnight gaps put jumps in the series that a
    segmenter reads as impossibly steep legs;
  - ETH is thin, so ATR overnight is not ATR at 09:30;
  - holidays and half days.
- Decide once, and record it in the file: **RTH only**, or ETH with a
  session-id column the segmenter can break on. RTH only is the simpler start
  and matches how the strategies here already trade.

### 2. Segmenter

Bottom-up merging (default) or Ramer-Douglas-Peucker on midpoint price, error
threshold scaled by ATR(14) so the detail follows the volatility. Per-segment
features: direction, slope/ATR, duration in bars, amplitude/ATR, fit error.

**Causality is the whole game, and it costs more than one line in the brief.**
Both algorithms are batch fits anchored at the right edge: the youngest segment
repaints, and under bottom up a late merge can move the one before it too. So
the replay harness recomputes the segmentation at every bar N over data ≤ N,
and each pivot carries:

| field | meaning |
|---|---|
| `bar` | where the pivot sits |
| `price` | the price it sits at |
| `first_seen_at` | the N at which it first appeared |
| `confirmed_at` | the N from which it stopped moving |

`confirmed_at - bar` is the **confirmation lag** — how late a swing is knowable.
It is the single number that decides whether this is honest, and it is what the
tagger reads: only pivots with `confirmed_at <= N` may be used at bar N. The
last pivot is always provisional and always excluded.

That lag is also worth measuring on its own, before any of the rest is built. If
a swing on 15m takes twelve bars to confirm, every rule downstream inherits
twelve bars of lateness, and the game has to show the same delayed picture the
strategy would trade on.

### 3. Rule-based tagger

Labels every bar from the confirmed swings only.

- **Uptrend** — the last confirmed swings make a higher high and a higher low.
- **Downtrend** — a lower high and a lower low.
- **Range** — mixed swings, price held between the last major high and low.
- **Weakening** — successive legs in the trend direction get shorter or less
  steep, or the pullback legs get longer or steeper than the trend legs.
- **Break of structure** — a close beyond the last protective swing: below the
  last higher low in an uptrend, above the last lower high in a downtrend.
- Thresholds are parameters, not constants.

Every label carries its reason as **structured data**, not a sentence:
`{rule: "break", close: 24_180, level: 24_215, level_kind: "HL", level_bar: 412}`
— so the stats can group by reason — with the sentence rendered from it for the
game to show.

Note the two speeds. A break of structure is a close against a level already
known, so it is available the bar it happens. Weakening compares confirmed legs,
so it arrives late by the confirmation lag. Do not let the game score them as if
they were the same kind of question.

### 4. Outcome labeler

From bar N, look ahead K bars (default 30), triple barrier: upper at +x·ATR,
lower at −x·ATR, time barrier at K. Outcome = continuation / reversal / no
resolution. The only component allowed to look forward.

Two futures caveats: with bar data, a bar that touches both barriers cannot say
which came first — resolve it on a finer series or record `ambiguous`; and a
barrier touched at 03:00 is not a barrier touched if the plan only trades RTH.

### 5. Synthetic generator

A regime-switching random walk with known phases — trend up, weaken, break,
range, trend down. Labels exact by construction.

Its real value is not the beginner levels, it is being the **test suite for the
tagger**: a generated break at bar 500 has to be found by the tagger within the
confirmation lag, and nowhere else. That is the only way to tell a rule bug from
a market that is simply hard.

### 6. Game

Replay: a random point, ~150 bars of history shown, the rest hidden. Segment
overlay toggleable — training wheels. Answer (a) the current state, (b) what is
expected next. Reveal the next K bars, the tagger's label and reason, the drawn
swings. Levels: synthetic → real with overlay → real without → lower timeframes.

Add an **"I disagree"** button. Strict scoring against the tagger trains
imitation of the tagger, mistakes included; the disagreement log is the rule
backlog, and it is the most valuable output of the early rounds.

Build it as Streamlit or a small local web app. Not a product — a tool for one
user.

### 7. Scoring

- Classification: strict, per round, against the tagger.
- Prediction: statistical over many rounds, hit rate per category, never
  punished per round — a correct read still loses sometimes.
- Dashboard: accuracy per category, confusion matrix, progress over time.
- Persist the session history (SQLite).

---

## Where the code lives

**Python owns the research. NinjaScript exports bars and draws the check.**

Exporting segments from `PriceSegments` sounds like a shortcut and is a trap:
the indicator holds one segmentation — the final one — and using it to label
history is precisely the lookahead the brief forbids. An honest export would be
one segmentation per bar, which is thousands of passes NinjaTrader would do
slowly and Python does in seconds.

So: NinjaTrader exports OHLCV, Python re-implements the segmenter (a translation
of a file that already works, not a design), and the chart keeps `PriceSegments`
as the picture the port has to reproduce.

The duplication only has to be paid back when the tagger goes live. At that
point the Python rules get frozen and ported to C#, with a golden-file test: the
same CSV through both, labels equal bar for bar.

---

## Non-negotiables

- No lookahead. Segmenter and tagger at bar N use only data ≤ N, and only
  confirmed pivots. The outcome labeler alone may look forward.
- Every label explainable.
- The tagger is a standalone module the trading system can import.

---

## Build order

0. Pick, export and freeze the continuous-contract files. Everything downstream
   is answers about this data; changing it later invalidates the scores.
1. Segmenter + the causal replay harness, and **measure the confirmation lag**
   per timeframe. Static chart of segments and swings, checked against
   `PriceSegments` on the same bars.
2. Tagger + labels on the static chart. Visual check first, stats second.
3. Outcome labeler + the first honest question: how often does a break of
   structure actually reverse, on MNQ, per timeframe?
4. Synthetic generator, used as the tagger's test suite.
5. Replay game, scoring, persistence.
6. Later: an HMM or ML classifier trained on the tagger and outcome labels,
   which has to beat the rule baseline out of sample or be thrown away.

Steps 1-3 are where the learning is, and they are small. Step 5 is the bulk of
the work and teaches nothing until 1-3 are right.

---

## House-keeping

This folder is Python and notes, so `deploy.ps1` ignores it — it only copies
`.cs`. If a NinjaScript strategy is ever written from these rules, it cannot
live here: the deploy maps a repo folder to the `bin\Custom` folder of the same
name, and `bin\Custom\Segment strategy` is not a folder NinjaTrader compiles.
Such a strategy goes in the repo root, like the others.
