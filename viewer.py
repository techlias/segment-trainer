"""A chart you can scrub through, showing only what each bar could know.

    streamlit run viewer.py

No labels yet - nothing in this project has an opinion until the tagger is
written. What this is for is the decisions that come before it: whether the
tolerance draws the legs you actually read, how much of a move is over by the
time its turn is confirmed, and what the right hand edge looks like at the
moment a question would be asked.

Every frame comes from ``replay.at``, so the chart cannot show you anything the
bar did not know, and the drawing is ``plot.draw`` - the same function the PNGs
and the NinjaTrader comparison use. One picture, one implementation.

THE REVEAL

    Reveal draws the next few bars in grey without re-reading the segmentation.
    That is deliberate: the point is to see what happened next TO THE READ YOU
    MADE, not to be shown a tidier read of the same chart. It is the game's
    answer key, wired up early because it is the only way to judge a tolerance
    honestly.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from segments import dataset as dataset_module
from segments import fit, plot, public, replay, studies, tagger

HERE = Path(__file__).parent

#: Where to look, in order. The folder beside the app comes first so the
#: deployed copy works with nothing configured; the NinjaTrader folder is for
#: running this on the machine that does the exporting. Neither has to exist -
#: an upload works on its own.
FOLDERS = [
    HERE / "data",
    Path.home() / "OneDrive" / "Documentos" / "NinjaTrader 8" / "segment-export",
    Path.home() / "Documents" / "NinjaTrader 8" / "segment-export",
]


st.set_page_config(page_title="Segments", layout="wide")


@st.cache_data(show_spinner=False)
def load(stem: str):
    return dataset_module.load(stem)


@st.cache_data(show_spinner="Re-cutting the bars...")
def recut(stem: str, method: str, source: str, tolerance: float, min_bars: int, window: int):
    """The same bars at other settings. Cached, because a long history takes a
    while and the slider must not pay for it on every frame."""
    return fit.refit(
        load(stem), method=method, source=source, tolerance=tolerance, min_bars=min_bars, window=window
    )


@st.cache_data(show_spinner="Downloading the bars...", ttl=900)
def public_bars(symbol: str, interval: str, days: int):
    """The download, cached apart from the cut.

    Two caches rather than one on purpose: the cut settings are not part of this
    key, so moving a tolerance slider re-cuts bars already in hand instead of
    asking Yahoo for them again.

    A quarter of an hour to live, because unlike an export these bars keep
    growing - the last one is the one forming now.
    """
    return public.fetch(symbol, interval, days)


@st.cache_data(show_spinner="Cutting the bars...")
def public_dataset(symbol: str, interval: str, days: int,
                   method: str, source: str, tolerance: float, min_bars: int, window: int):
    """Public bars as a dataset, cut at these settings."""
    return public.build(
        public_bars(symbol, interval, days), symbol=symbol, interval=interval,
        method=method, source=source, tolerance=tolerance, min_bars=min_bars, window=window,
    )


def stash(files) -> str | None:
    """Saves an uploaded pair of CSVs and returns their stem.

    The whole point of the uploader is that the deployed app is not on the
    machine that does the exporting. Drop the two files SegmentExport wrote and
    they are read exactly as a local pair would be - no redeploy, and nothing
    kept after the session.
    """
    if not files:
        return None

    folder = Path(tempfile.gettempdir()) / "segments-upload"
    folder.mkdir(parents=True, exist_ok=True)

    stems: set[str] = set()
    for one in files:
        (folder / one.name).write_bytes(one.getbuffer())
        for suffix in ("-bars.csv", "-pivots.csv"):
            if one.name.endswith(suffix):
                stems.add(one.name[: -len(suffix)])

    for stem in sorted(stems):
        pair = [folder / (stem + suffix) for suffix in ("-bars.csv", "-pivots.csv")]
        if all(path.exists() for path in pair):
            return str(folder / stem)

    st.sidebar.error("Upload BOTH files of an export - the -bars.csv and the -pivots.csv")
    return None


#: How much history each interval opens on. Different numbers of days, one
#: target: a couple of thousand bars, whatever a bar is worth here. Measured on
#: NQ=F, these give 2,096 / 2,219 / 2,260 / 2,270 / 2,483.
#:
#: The number that matters is not the download, it is the CUT. Every bar gets
#: its own fit over the window behind it - that is what makes the log causal -
#: so the sweep is linear in bars, at about five milliseconds each in Python.
#: Two thousand bars is a ten second pause; the 6,625 that thirty days of five
#: minute bars comes to took 38 seconds, which is long enough that people
#: assume the app has hung. Community Cloud is slower than this machine.
#:
#: The slider still goes to whatever Yahoo will serve. This is where it opens,
#: not where it stops.
OPENS_ON = {"5m": 10, "15m": 30, "30m": 60, "1h": 120, "1d": 3000}


def pick_public() -> tuple[tuple, object] | tuple[None, None]:
    """The instrument picker for Yahoo's bars."""
    symbol = st.sidebar.selectbox(
        "Instrument", list(public.SYMBOLS),
        format_func=lambda one: f"{one} - {public.SYMBOLS[one][0]}",
    )
    interval = st.sidebar.selectbox("Bar size", list(public.INTERVALS))

    # Yahoo's own cap for this interval, not a preference. Over it the download
    # quietly returns less rather than failing, so the slider cannot ask.
    cap = public.INTERVALS[interval] or 3650
    days = st.sidebar.slider(
        "History (days)", 5, cap, min(OPENS_ON.get(interval, 30), cap),
        help=f"Yahoo serves at most {cap} days at {interval}. Every bar is cut against the "
             "window behind it, so a long history is slow the first time - about five "
             "seconds per thousand bars, then cached.",
    )

    key = (symbol, interval, days)

    try:
        return key, public_dataset(
            symbol, interval, days,
            method=public.DEFAULTS["method"],
            source=public.DEFAULTS["source"],
            tolerance=public.DEFAULTS["tolerance"],
            min_bars=public.DEFAULTS["min_bars"],
            window=public.DEFAULTS["window"],
        )
    except Exception as problem:
        # Someone else's server, over someone else's network. It will fail
        # sometimes, and when it does the app should say which part failed.
        st.sidebar.error(f"Could not get {symbol} at {interval} from Yahoo:\n\n{problem}")
        return None, None


def pick_export() -> tuple[str, object] | tuple[None, None]:
    found: list[Path] = []
    for folder in FOLDERS:
        if folder.is_dir():
            found += dataset_module.find(folder)

    uploaded = stash(
        st.sidebar.file_uploader(
            "Upload an export", type="csv", accept_multiple_files=True,
            help="Both CSVs SegmentExport wrote. Use this when the app is not on the machine that exports.",
        )
    )

    if uploaded:
        return uploaded, load(uploaded)

    if not found:
        return None, None

    chosen = st.sidebar.selectbox("Dataset", found, format_func=lambda path: path.name)

    return str(chosen), load(str(chosen))


st.sidebar.subheader("The bars")
where = st.sidebar.radio(
    "Where from", ["A NinjaTrader export", "Yahoo Finance"],
    help="An export is the real thing and is checked against NinjaTrader bar for bar. "
         "Yahoo is public data cut by the same code, for when there is no export to hand.",
)

if where == "Yahoo Finance":
    kind = "public"
    stem, original = pick_public()
else:
    kind = "export"
    stem, original = pick_export()

if original is None:
    if kind == "export":
        st.title("Segments")
        st.info(
            "No export found. Drop **SegmentExport** on a chart in NinjaTrader, let it run "
            "through the history, then upload the two CSVs it wrote using the sidebar - "
            "or put them in the `data` folder beside this app.\n\n"
            "Or switch **Where from** to **Yahoo Finance** and read public bars instead."
        )
    st.stop()

if kind == "public":
    # The label. Everything here is cut by the same code an export is cut by,
    # but the BARS are Yahoo's - aggregated, revised, and with the session
    # inferred from gaps rather than read from a template. There is no
    # NinjaTrader run behind them to check against, and saying so once on screen
    # is cheaper than someone eventually concluding the pipeline is broken.
    st.sidebar.caption(
        ":orange[Public bars.] Same segmentation, but nothing to verify them against - "
        "Yahoo's candles are not the exchange's, and the session is guessed from gaps "
        "in the timestamps. Fine for reading shapes, not evidence about MNQ."
    )

problems = original.check()
if problems:
    st.sidebar.warning("\n".join(f"- {problem}" for problem in problems))

# --- what to cut, and how ----------------------------------------------------

st.sidebar.subheader("The cut")
st.sidebar.caption("Anything other than the settings these bars arrived with re-cuts them here.")

meta = original.meta
method = st.sidebar.selectbox("Method", fit.METHODS, index=fit.METHODS.index(meta.method))
source = st.sidebar.selectbox(
    "Source", ["Close", "BodyCentre", "Median", "Typical"],
    index=["Close", "BodyCentre", "Median", "Typical"].index(meta.source),
)
tolerance = st.sidebar.slider("Tolerance (ATR)", 0.2, 4.0, float(meta.tolerance), 0.05)
min_bars = st.sidebar.slider("Minimum length (bars)", 1, 12, int(meta.minbars))
window = st.sidebar.select_slider(
    "Window (bars)", [100, 150, 250, 400, 600], value=min([100, 150, 250, 400, 600],
                                                          key=lambda v: abs(v - meta.window))
)

untouched = (
    method == meta.method and source == meta.source and tolerance == meta.tolerance
    and min_bars == meta.minbars and window == meta.window
)

if untouched:
    data = original
elif kind == "public":
    data = public_dataset(*stem, method, source, tolerance, min_bars, window)
else:
    data = recut(stem, method, source, tolerance, min_bars, window)

# --- what to show ------------------------------------------------------------

st.sidebar.page_link("pages/1_Tutorial.py", label="How to read this page", icon=":material/help:")

st.sidebar.subheader("The view")
history = st.sidebar.slider("History (bars)", 40, 400, 150, 10)
reveal = st.sidebar.slider("Reveal (bars)", 0, 120, 0, 5,
                           help="Bars past the one you are on, drawn in grey. The answer.")
segments_on = st.sidebar.checkbox("Show the segments", True,
                                  help="Off is the chart without training wheels")
# One control rather than two checkboxes, because the three are a choice and not
# three independent things: candles already draw the range the sticks would.
bars_on = st.sidebar.radio("Show the bars", ["Hidden", "High-low sticks", "Candles"],
                           index=0, horizontal=False,
                           help="Behind the segments. Candles show the open and close too - "
                                "and show that the turns sit where the fitted price turned, "
                                "not at the high or the low of the bar.")
sticks_on = bars_on == "High-low sticks"
candles_on = bars_on == "Candles"
bands_on = st.sidebar.checkbox("Show the Bollinger", True,
                               help="The band behind the price. Off is worth trying: it is the one "
                                    "thing here that suggests where price should go rather than "
                                    "describing where it went.")
classify = st.sidebar.checkbox("Classify", True,
                               help="The tagger's reading: the swing names and the protective level")

with st.sidebar.expander("The rules"):
    st.caption("Defaults are a starting point. Argue with them against a long export, not against a few days.")
    rules = tagger.Rules(
        equal_atr=st.slider("Same level within (ATR)", 0.0, 0.5, 0.15, 0.01,
                            help="Two swings this close are a double top, not a higher high"),
        break_atr=st.slider("Break clears by (ATR)", 0.0, 0.5, 0.10, 0.01,
                            help="How far past the level a close must settle to be a break"),
        weaken_ratio=st.slider("Push shrunk to", 0.3, 1.0, 0.75, 0.05),
        pullback_ratio=st.slider("Pullback gave back", 0.3, 1.0, 0.75, 0.05),
    )

    # Not the tagger's rules - nothing the tagger does reads either of these -
    # but they belong beside them, because they are the other numbers on the
    # page you would change to argue with what you are being shown. Top to
    # bottom in the order they appear on the chart.
    band_period = st.slider("Bollinger (bars)", 5, 20, studies.BAND_PERIOD, 1,
                            help=f"The band behind the price, at "
                                 f"{studies.BAND_DEVIATIONS:g} standard deviations. "
                                 "Short wraps price like an envelope; long barely moves")
    rsi_period = st.slider("RSI (bars)", 4, 20, studies.RSI_PERIOD, 1,
                           help="The momentum panel under the chart. Short reads every leg; "
                                "long reads the session")

# --- where to stand ----------------------------------------------------------

first, last = data.first_bar, data.last_bar

# Which bars these are, not how they were cut. Changing the tolerance should
# leave you standing where you were - that is the whole point of moving it - but
# changing the instrument, the interval or the history means the bar you were on
# is a different moment in a different market, and keeping its number would be
# keeping a coincidence.
standing_on = (kind, str(stem))

if st.session_state.get("standing_on") != standing_on:
    st.session_state.standing_on = standing_on
    st.session_state.bar = last

if "bar" not in st.session_state or not first <= st.session_state.bar <= last:
    st.session_state.bar = last

step_back, step_on, jump, _ = st.columns([1, 1, 1, 9])
if step_back.button("◀", width="stretch"):
    st.session_state.bar = max(first, st.session_state.bar - 1)
if step_on.button("▶", width="stretch"):
    st.session_state.bar = min(last, st.session_state.bar + 1)
if jump.button("random", width="stretch"):
    low = min(first + max(history, data.meta.window), last - reveal - 1)
    st.session_state.bar = random.randint(max(first, low), max(first + 1, last - reveal))

bar = st.slider("Bar", first, last, key="bar")

# --- the chart ---------------------------------------------------------------

view = replay.at(data, bar)
pivots = view.pivots()
pieces = view.segments()
reading = tagger.label(data, bar, rules, view=view) if classify else None

# Price over momentum, sharing an x axis so the two can never drift apart by a
# bar - which is the only way a panel underneath is worth anything.
figure, (axis, lower) = plt.subplots(
    2, 1, sharex=True, figsize=(15, 6.6),
    gridspec_kw={"height_ratios": [3.4, 1], "hspace": 0.08},
)

# Behind everything, and drawn before everything, so nothing has to argue about
# z order with it. The band reaches to the revealed bars too when they are shown:
# it is context, not a reading, and hiding it at the edge would only make the
# right hand side look calmer than it was.
close = data.bars["close"]
left = max(first, bar - history)
right = min(last, bar + reveal)

if bands_on:
    plot.bands(axis, studies.bollinger(close, period=band_period).loc[left:right])

if segments_on:
    plot.draw(data, bar, history=history, reveal=reveal,
              sticks=sticks_on, candles=candles_on, ax=axis)
    if reading is not None:
        plot.annotate(axis, reading, right=bar)
else:
    # The same window with the lines taken away: price alone, which is the level
    # the training is actually aiming at. The bars still answer to the sidebar
    # here - this is the view the reading is practised on, and practising it on
    # candles when the chart being read is candles is the whole point.
    price = data.source_price()
    if candles_on:
        plot.candlesticks(axis, data.bars.loc[left:bar])
    elif sticks_on:
        window = data.bars.loc[left:bar]
        axis.vlines(window.index, window["low"], window["high"],
                    color="#c8ccd2", linewidth=0.8, zorder=1)
    axis.plot(price.loc[left:bar].index, price.loc[left:bar].to_numpy(), color="#4a4f57", linewidth=1.3)
    if reveal:
        axis.plot(price.loc[bar:right].index, price.loc[bar:right].to_numpy(),
                  color=plot.HIDDEN, linewidth=1.3)
        axis.axvline(bar, color="#b0b4ba", linewidth=1.0, linestyle=":")
    axis.set_xlim(left, max(bar + reveal, bar) + 2)
    axis.grid(True, alpha=0.15)
    axis.set_xlabel("bar")

# The x label belongs to the bottom panel now, not to the price.
axis.set_xlabel("")
plot.momentum(lower, studies.rsi(close, rsi_period).loc[left:right], rsi_period,
              at=bar if reveal else None)
lower.set_xlabel("bar")

st.pyplot(figure, width="stretch")
plt.close(figure)

if reading is not None:
    # The colour is the state, so the reading can be taken in before it is read.
    shout = {
        tagger.UPTREND: st.success, tagger.DOWNTREND: st.error,
        tagger.BREAK: st.warning, tagger.WEAKENING: st.warning,
    }.get(reading.state, st.info)

    shout(f"**{reading.state.upper()}** — {reading.says()}")

# --- what the bar knows ------------------------------------------------------

leg = pieces[-1] if pieces else None
ticksize = data.meta.ticksize or 1.0

a, b, c, d, e = st.columns(5)
a.metric("Time", f"{data.bars.loc[bar, 'time']:%a %H:%M}")
b.metric("Turns known", len(pivots))
c.metric("Leg in progress", f"{leg.bars} bars" if leg else "-",
         f"{leg.change / ticksize:+.0f}t" if leg else None)
d.metric("Newest turn", f"{bar - pivots[-1].bar} bars ago" if pivots else "-",
         f"seen {pivots[-1].age} bars late" if pivots else None)
e.metric("ATR", f"{data.bars.loc[bar, 'atr'] / ticksize:.0f}t")

if not untouched:
    st.caption(
        f"Re-cut here: {method}/{source}, tolerance {tolerance} ATR, min {min_bars} bars, "
        f"window {window}. "
        f"{'These bars arrived' if kind == 'public' else 'The export itself is'} "
        f"{meta.method}/{meta.source} at {meta.tolerance}."
    )

with st.expander("The legs behind this bar, newest first"):
    if len(pieces) < 2:
        st.write("Nothing confirmed yet.")
    else:
        st.dataframe(
            {
                "from": [piece.start_bar for piece in reversed(pieces)],
                "to": [piece.end_bar for piece in reversed(pieces)],
                "bars": [piece.bars for piece in reversed(pieces)],
                "ticks": [round(piece.change / ticksize) for piece in reversed(pieces)],
                "settled": ["no" if piece.provisional else "yes" for piece in reversed(pieces)],
            },
            width="stretch",
            hide_index=True,
        )

with st.expander("What this dataset costs you in lag"):
    summary = replay.lag_summary(data)
    if summary.empty:
        st.write("Nothing has settled - is the history longer than the window?")
    else:
        legs = replay.legs(data)
        median_leg = legs["bars"].median() if not legs.empty else float("nan")

        st.write(
            f"Turns settle **{summary['lag_median']:.0f} bars** after they happen "
            f"(p90 {summary['lag_p90']:.0f}), and the median leg runs "
            f"**{median_leg:.0f} bars**."
        )
        st.write(
            f"So a turn is trustworthy about **{summary['lag_median'] / max(median_leg, 1):.0%}** "
            "of the way through the leg that follows it. Under a third is room to work with; "
            "over a half means the read arrives after the move."
        )
        st.caption(
            "Measured over the whole file, which is the one thing here that looks "
            "forward. It is a property of the dataset, never a feature."
        )
