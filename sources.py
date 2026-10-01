"""Where the bars come from, for every page that needs some.

The viewer and the gym ask the same question - an export, or Yahoo? - and the
answer involves an uploader, a folder scan, two caches and Yahoo's history
caps. Held in one place because two copies of it would drift, and the first
thing to drift would be which instruments exist.

Streamlit lives here and nowhere under ``segments/``. Everything in the package
proper stays importable without a browser, which is what makes it testable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from segments import dataset as dataset_module
from segments import fit, public

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

    # Two folders are scanned and an export can sit in both - the same file
    # copied beside the app, or a different one that happens to share a name.
    # Labelled by name alone they are the same entry twice, and a selectbox
    # cannot keep its selection between two identical labels: it hands back
    # whichever it likes on each rerun, so the dataset changes under the page
    # for no reason anyone can see. Where a name repeats, say which folder.
    seen = [path.name for path in found]
    labels = {
        path: path.name if seen.count(path.name) == 1 else f"{path.name}  ·  {path.parent.name}"
        for path in found
    }

    chosen = st.sidebar.selectbox("Dataset", found, key="dataset",
                                  format_func=lambda path: labels[path])

    return str(chosen), load(str(chosen))

def pick() -> tuple[str, object, object]:
    """The sidebar block both pages open with: where from, then which.

    Returns ``(kind, key, dataset)`` where kind is "export" or "public" and key
    identifies the bars for a cache - a stem for an export, a tuple for Yahoo.
    The dataset is None when nothing could be loaded, and the caller decides
    what to say about that.
    """
    st.sidebar.subheader("The bars")
    where = st.sidebar.radio(
        "Where from", ["A NinjaTrader export", "Yahoo Finance"],
        help="An export is the real thing and is checked against NinjaTrader bar for bar. "
             "Yahoo is public data cut by the same code, for when there is no export to hand.",
    )

    if where == "Yahoo Finance":
        key, data = pick_public()
        return "public", key, data

    key, data = pick_export()

    return "export", key, data


def recut_as(kind: str, key, method: str, source: str,
             tolerance: float, min_bars: int, window: int):
    """Re-cut whichever kind of bars these are, at these settings."""
    if kind == "public":
        return public_dataset(*key, method, source, tolerance, min_bars, window)

    return recut(key, method, source, tolerance, min_bars, window)


def is_public(kind: str) -> bool:
    return kind == "public"


LABEL = (
    ":orange[Public bars.] Same segmentation, but nothing to verify them against - "
    "Yahoo's candles are not the exchange's, and the session is guessed from gaps "
    "in the timestamps. Fine for reading shapes, not evidence about MNQ."
)

