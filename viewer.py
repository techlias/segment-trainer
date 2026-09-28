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
from segments import fit, plot, replay, tagger

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


def pick_dataset() -> tuple[str, object] | tuple[None, None]:
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


stem, original = pick_dataset()

if original is None:
    st.title("Segments")
    st.info(
        "No export found. Drop **SegmentExport** on a chart in NinjaTrader, let it run "
        "through the history, then upload the two CSVs it wrote using the sidebar - "
        "or put them in the `data` folder beside this app."
    )
    st.stop()

problems = original.check()
if problems:
    st.sidebar.warning("\n".join(f"- {problem}" for problem in problems))

# --- what to cut, and how ----------------------------------------------------

st.sidebar.subheader("The cut")
st.sidebar.caption("Anything other than the export's own settings re-cuts the bars here.")

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

data = original if untouched else recut(stem, method, source, tolerance, min_bars, window)

# --- what to show ------------------------------------------------------------

st.sidebar.subheader("The view")
history = st.sidebar.slider("History (bars)", 40, 400, 150, 10)
reveal = st.sidebar.slider("Reveal (bars)", 0, 120, 0, 5,
                           help="Bars past the one you are on, drawn in grey. The answer.")
segments_on = st.sidebar.checkbox("Show the segments", True,
                                  help="Off is the chart without training wheels")
sticks_on = st.sidebar.checkbox("Show the bars", False,
                                help="High-low sticks behind the segments")
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

# --- where to stand ----------------------------------------------------------

first, last = data.first_bar, data.last_bar
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

figure, axis = plt.subplots(figsize=(15, 5.2))

if segments_on:
    plot.draw(data, bar, history=history, reveal=reveal, sticks=sticks_on, ax=axis)
    if reading is not None:
        plot.annotate(axis, reading, right=bar)
else:
    # The same window with the lines taken away: price alone, which is the level
    # the training is actually aiming at.
    price = data.source_price()
    left = max(first, bar - history)
    axis.plot(price.loc[left:bar].index, price.loc[left:bar].to_numpy(), color="#4a4f57", linewidth=1.3)
    if reveal:
        right = min(last, bar + reveal)
        axis.plot(price.loc[bar:right].index, price.loc[bar:right].to_numpy(),
                  color=plot.HIDDEN, linewidth=1.3)
        axis.axvline(bar, color="#b0b4ba", linewidth=1.0, linestyle=":")
    axis.set_xlim(left, max(bar + reveal, bar) + 2)
    axis.grid(True, alpha=0.15)
    axis.set_xlabel("bar")

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
        f"window {window}. The export itself is {meta.method}/{meta.source} at {meta.tolerance}."
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
