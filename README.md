# segments

Reads the dataset `SegmentExport` writes, without ever looking ahead.

This is step 1 of [BRIEF.md](BRIEF.md): the data side of the structure trainer.
The tagger, the outcome labeler and the game are not here yet.

## Getting it running

```bash
cd "Segment strategy"
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # or: pip install pandas matplotlib pytest
.venv/Scripts/python -m pytest -q                  # 11 tests, under a second
```

Then export a dataset: drop **SegmentExport** on an MNQ chart in NinjaTrader,
let it run through the history, and it writes two CSVs into
`Documents\NinjaTrader 8\segment-export\`.

```bash
python -m segments list   "~/Documents/NinjaTrader 8/segment-export"
python -m segments report "~/Documents/NinjaTrader 8/segment-export/MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w250"
python -m segments plot   <stem> --bar 4000 --history 150 --reveal 30 --bars
python -m segments sheet  <stem> --count 8
python -m segments verify <stem>                      # does the port agree with the C#?
python -m segments refit  <stem> --tolerance 0.5      # re-cut, no re-export
```

Every command takes either CSV of an export, their shared stem, or a folder
holding exactly one export.

## Running it somewhere else

[DEPLOY.md](DEPLOY.md) — Streamlit Community Cloud, free, from a private repo,
with the app locked to an email allow-list. The sidebar takes an uploaded pair
of CSVs, so a deployed copy can open an export from any machine without a
redeploy.

## Learning to read it

[doc/reading-the-chart.html](doc/reading-the-chart.html) — the tutorial
([versión en español](doc/reading-the-chart.es.html)), also served inside the
app under **Tutorial** in the page nav, so a deployed copy carries its own
instructions. Every
mark on the viewer page, the vocabulary the tagger argues in (HH, HL, push,
pullback, weakening, break of structure), the six states with the rule behind
each, and what the protective level means. Open it beside the viewer.

## The tagger

The first thing here with an opinion. It reads the confirmed pivots at a bar,
keeps the ones that are actual turns, names them HH / HL / LH / LL against their
predecessors, and says what the structure is:

```python
from segments import load, tagger

one = tagger.label(load(stem), 1200)
print(one.state, "-", one.says())
# uptrend - Uptrend: HH 30498.88 at bar 1176 and HL 30434.75 at bar 1170
```

```bash
python -m segments tag <stem>            # the share of the history in each state
python -m segments tag <stem> --bar 1200 # one bar, with its whole swing chain
```

States: `uptrend`, `downtrend`, `weakening` (a trend whose pushes are shrinking
or whose pullbacks are deepening), `break` (a **close** beyond the protective
swing), `range`, `unclear`. Every label carries its reason as data, with
`says()` rendering the sentence - so the stats can group by rule and the chart
can still explain itself.

Two decisions worth knowing about:

- **A pivot is not always a turn.** Two rising pieces of different steepness
  meet at a bend, and a bend is not a swing high. Only reversals become swings.
- **The protective level is the swing before the last extreme**, not the newest
  one. The newest low may be one bar old with nothing built on it; protecting
  that made a quarter of all MNQ bars a "break", which is another way of saying
  it said nothing.

Thresholds live in `tagger.Rules`, in ATRs and ratios, never ticks. The defaults
are a starting point. Argue with them against a long export.

## The viewer

```bash
.venv/Scripts/python -m pip install -e ".[dev,ui]"
.venv/Scripts/streamlit run viewer.py
```

A chart you can scrub through, one bar at a time, showing only what that bar
could know. No labels — nothing here has an opinion until the tagger is written.
What it is for is the decisions that come before it:

- **Tolerance, method, window** in the sidebar re-cut the bars live (cached, so
  the slider does not pay for it), which is how you find the setting that draws
  the legs you actually read.
- **Reveal** draws the next few bars in grey without re-reading the
  segmentation — what happened next *to the read you made*, not a tidier read of
  the same chart. The game's answer key, wired up early.
- **Show the segments** off is the chart without training wheels.
- **Classify** puts the tagger's reading on the page: the swing names at each
  turn, the protective level drawn across, and the state with its reason. The
  rules themselves are sliders, so a threshold can be argued with on the spot.
- The numbers under it say how late the newest turn was seen, and the lag panel
  turns that into the number that matters: how far into the following leg a turn
  becomes trustworthy.

## The one rule

Anything that could ever feed a label, a question or a feature comes from
`replay.at(data, n)` or `replay.walk(data)` — the causal reconstruction, which
knows only what bar *n* knew.

```python
from segments import load, replay

data = load(".../MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w250")
print(data.meta, data.check())

view = replay.at(data, 4000)
for piece in view.segments():
    print(piece.start_bar, piece.end_bar, piece.slope(), piece.provisional)
```

Three functions deliberately break that rule, and say so in their docstrings:
`confirmation`, `lag_summary` and `legs` read the whole file. They exist to
*measure* the dataset — above all the confirmation lag — and their output
belongs in a report, never in a feature.

## What the modules do

| module | |
|---|---|
| `dataset.py` | loads both CSVs and the JSON header; `check()` refuses files that cannot mean what they claim |
| `replay.py` | the causal reconstruction, plus the lag and leg measurements |
| `fit.py` | the port of `SegmentFit.cs`: the three methods, the causal sweep, `refit` and `verify` |
| `tagger.py` | the rules: swings from pivots, and what state the market is in at a bar |
| `plot.py` | a static picture of one bar's view, and a contact sheet of several |
| `__main__.py` | `report`, `plot`, `sheet`, `tag`, `verify`, `refit`, `list` |

## Reading the report

The number that matters is the **confirmation lag**: bars between a turn
happening and it being trustworthy. Half the turns settle by the median; every
rule downstream inherits that delay, and no amount of cleverness gets it back.
If it is large next to the legs you want to trade, the timeframe is wrong, not
the rules.

`check()` is worth running on every new export. The three failures that actually
happen: two files from different runs left in the same folder (caught by
comparing each pivot price against its own bar's source price), an export
interrupted so the bars stop before the events do, and a hand-edited CSV.

## Re-cutting without re-exporting

`fit.py` is the port of `SegmentFit.cs`, so the bars alone are enough to try
another tolerance:

```python
from segments import load, fit
data = load(stem)
finer = fit.refit(data, tolerance=0.4)      # a Dataset like any other
rougher = fit.refit(data, method="SlidingWindow")
```

A refit dataset is written in the exporter's own format, so nothing downstream
can tell it from one NinjaTrader produced — and should not be able to.

**`verify` is what makes the port trustworthy.** It re-cuts an export's own bars
with that export's own settings and diffs the events against the ones
NinjaTrader wrote. Empty means the two implementations agree bar for bar; run it
once per new export, before trusting anything measured here. Events are compared
as a set per bar, since the exporter iterates a HashSet and the order within one
bar is arbitrary — it never changes what the replay ends up holding.

Cost: about 7 ms a bar, so a 10,000 bar history re-cuts in around a minute.

### What to watch for

On a synthetic path, the three methods churn wildly differently under re-fitting
— events per bar, and the median bars to settle:

| method | events/bar | settles after |
|---|---|---|
| BottomUp | 13.6 | 99 bars |
| SlidingWindow | 1.5 | 3 bars |
| DouglasPeucker | 0.6 | 24 bars |

That is synthetic noise, not MNQ, so take the numbers as a demonstration of what
`report` measures rather than as a result. But the mechanism is real: bottom-up
merges cascade, so one new bar can rewrite vertices well back into the window,
and the cleanest picture is also the least stable one. Measure it on your own
export before choosing a method for the tagger — and if the lag is large, a
pivot that must survive *k* bars before it counts may be worth more than a
tighter tolerance.

## Checking it against NinjaTrader

`plot` draws segments, not candles — that is the trainer's view. Put the same
instrument, period and settings on a chart with **PriceSegments**, look at the
same bar, and the two pictures agree or something in the pipeline is wrong. It
is the cheapest test in the project and the only one that catches a
misunderstanding of the data rather than a bug in the code.

Two things will differ legitimately, and knowing which is which saves an hour:

- PriceSegments draws the segmentation **as at the last bar of the chart**;
  `plot --bar N` draws it as at N. Scroll the chart so N is the last bar.
- `PriceSegments` defaults to `Source = Close`, `SegmentExport` to `Median`. Set
  the chart to match the header line.

## Next

The outcome labeler — triple barrier from bar N, and the first honest question:
how often does a break of structure on MNQ actually reverse? Then the game.
See [BRIEF.md](BRIEF.md) steps 3 to 5.
