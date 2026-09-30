"""Bars from a public source, so the trainer has something to show without an
export.

Everywhere else this package reads what SegmentExport wrote and holds itself to
it: :func:`segments.fit.verify` re-cuts an export's own bars and diffs the result
against what NinjaTrader logged, and that diff is why any of this can be trusted.

NOTHING HERE CAN BE CHECKED THAT WAY.

    There is no NinjaTrader run behind a Yahoo download, so there is nothing to
    diff against. The segmentation is the same code - :func:`segments.fit.events`
    to the letter, the same sweep the verified path uses - but the BARS are
    someone else's, and three things about them are assumptions rather than
    facts:

        the session      NinjaTrader cuts the day with a session template. Here
                         a session break is inferred from a gap in the
                         timestamps, which finds the weekend and the daily
                         maintenance break and would not find a template that
                         splits a day for its own reasons.

        the tick size    stated per symbol below from what the exchange
                         publishes, not read from the data. Wrong for a symbol
                         added carelessly, and only cosmetic - it scales the
                         "ticks" readouts and nothing that is fitted.

        the bars         Yahoo's intraday candles are not the exchange's. They
                         are aggregated, they carry gaps, and they get revised.
                         Good enough to learn to read a shape on; not evidence
                         of anything about MNQ.

    So a public dataset says so in its meta - ``raw["public"]`` is True, and the
    provider is named - and the viewer says so on screen. If that label ever
    comes off, someone will eventually compare one of these to an export and
    conclude the pipeline is broken.

WHAT YAHOO WILL GIVE

    Intraday history is capped, hard, and the cap is per interval rather than
    per request: roughly 60 days at 5 and 15 minutes, 730 at an hour, and
    everything at a day. Asking for more does not error, it silently returns
    less, which is worse. :data:`INTERVALS` holds the caps and
    :func:`fetch` clamps to them rather than letting that happen quietly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .dataset import BAR_COLUMNS, Dataset, Meta
from . import fit

#: What the trainer offers, and the tick size the exchange publishes for each.
#: Index futures first: they are the closest public thing to the MNQ exports the
#: tagger's thresholds were argued against, so a reading here is comparable.
SYMBOLS = {
    "NQ=F": ("E-mini Nasdaq 100", 0.25),
    "ES=F": ("E-mini S&P 500", 0.25),
    "YM=F": ("E-mini Dow", 1.0),
    "RTY=F": ("E-mini Russell 2000", 0.1),
    "CL=F": ("Crude oil", 0.01),
    "GC=F": ("Gold", 0.1),
    "QQQ": ("Nasdaq 100 ETF", 0.01),
    "SPY": ("S&P 500 ETF", 0.01),
}

#: Interval to the most days of history Yahoo will serve at it. None is "all of
#: it". These are Yahoo's limits, not preferences - see the module note.
INTERVALS = {
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": None,
}

#: The settings a public dataset is cut at unless the caller says otherwise.
#: PriceSegments' own defaults, so the picture matches the indicator's.
DEFAULTS = {
    "method": "BottomUp",
    "source": "Median",
    "tolerance": 1.0,
    "min_bars": 3,
    "window": 250,
    "atr_period": 14,
}

#: A gap this many bar-lengths wide is a session break rather than a quiet
#: patch. Two would trip on a thin overnight hour; anything much larger stops
#: finding the daily futures break, which is about an hour.
SESSION_GAP = 3


def fetch(symbol: str, interval: str = "5m", days: int | None = None) -> pd.DataFrame:
    """Raw OHLCV from Yahoo, oldest first, indexed by timestamp.

    ``days`` is clamped to what Yahoo will actually serve at this interval, and
    the clamping is deliberate: over the limit it returns a short frame rather
    than an error, and a silently short history is the kind of thing that gets
    read as a market being quiet.
    """
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}, expected one of {sorted(INTERVALS)}")

    import yfinance  # imported here so the rest of the package needs no network stack

    cap = INTERVALS[interval]
    if days is None:
        days = cap if cap is not None else 3650
    elif cap is not None:
        days = min(days, cap)

    frame = yfinance.download(
        symbol,
        period=f"{int(days)}d",
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if frame is None or frame.empty:
        raise LookupError(f"Yahoo returned nothing for {symbol} at {interval} over {days}d")

    # One ticker still comes back with a (field, ticker) column index.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns=str.lower)
    frame = frame[["open", "high", "low", "close", "volume"]].dropna()

    if frame.empty:
        raise LookupError(f"Yahoo returned only empty candles for {symbol} at {interval}")

    return frame.sort_index()


def bars(frame: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """A raw OHLCV frame in the shape the rest of the package reads.

    Separate from :func:`fetch` so it can be tested without a network, which is
    the only part of this worth testing: the fetching is Yahoo's problem and the
    shaping is ours.

    The ATR is computed here and that is safe, unlike in :func:`segments.fit.refit`
    - see :func:`segments.fit.atr`. The caveat there is about recomputing over a
    SLICE of a chart; this frame is the whole of its own chart, first bar
    included, so the seed is right by construction.
    """
    when = pd.to_datetime(frame.index)

    out = pd.DataFrame(
        {
            "bar": np.arange(len(frame), dtype=int),
            "time": when,
            "open": frame["open"].to_numpy(dtype=float),
            "high": frame["high"].to_numpy(dtype=float),
            "low": frame["low"].to_numpy(dtype=float),
            "close": frame["close"].to_numpy(dtype=float),
            "volume": frame["volume"].to_numpy(dtype=float),
        }
    )

    out["mid"] = (out["high"] + out["low"]) / 2
    out["atr"] = fit.atr(out, atr_period)
    out["new_session"] = _sessions(when)

    return out[BAR_COLUMNS].set_index("bar")


def _sessions(when: pd.DatetimeIndex) -> np.ndarray:
    """1 where a bar opens a session, inferred from a gap in the timestamps.

    An inference, not a session template - the module note says why that
    matters. The usual bar spacing is taken as the median rather than the mode
    of the differences, so one long weekend cannot define "usual".
    """
    flags = np.zeros(len(when), dtype=int)
    if len(when) == 0:
        return flags

    flags[0] = 1
    if len(when) == 1:
        return flags

    steps = np.diff(when.asi8)
    usual = float(np.median(steps))

    if usual > 0:
        flags[1:] = (steps > SESSION_GAP * usual).astype(int)

    return flags


def dataset(symbol: str, interval: str = "5m", days: int | None = None, **cut) -> Dataset:
    """Public bars, segmented here, as a dataset the rest of the package reads.

    The result is interchangeable with a loaded export in every way except the
    one that counts, and it carries the label saying so.
    """
    return build(fetch(symbol, interval, days), symbol=symbol, interval=interval, **cut)


def build(frame: pd.DataFrame, symbol: str, interval: str = "5m", **cut) -> Dataset:
    """The dataset for an OHLCV frame already in hand.

    Split out from :func:`dataset` so the shaping, the cut and the labelling can
    be exercised on a frame from anywhere - a test, a CSV, another provider -
    without going to Yahoo for it.
    """
    settings = {**DEFAULTS, **cut}

    frame_bars = bars(frame, settings["atr_period"])

    log = fit.events(
        frame_bars,
        method=settings["method"],
        source=settings["source"],
        tolerance=settings["tolerance"],
        min_bars=settings["min_bars"],
        window=settings["window"],
        scale=frame_bars["atr"].to_numpy(dtype=float),
    )

    name, ticksize = SYMBOLS.get(symbol, (symbol, 0.0))

    meta = Meta(
        instrument=f"{symbol} - {name}" if name != symbol else symbol,
        period=interval,
        # Not a NinjaTrader session template, and it does not pretend to be one.
        session="inferred from gaps in the timestamps",
        method=settings["method"],
        source=settings["source"],
        ticksize=ticksize,
        tolerance=float(settings["tolerance"]),
        minbars=int(settings["min_bars"]),
        window=int(settings["window"]),
        atr=int(settings["atr_period"]),
        exported=dt.datetime.now().isoformat(timespec="seconds"),
        raw={
            # The label. Everything downstream that needs to know this is not an
            # export asks for this key.
            "public": True,
            "provider": "Yahoo Finance",
            "symbol": symbol,
            "interval": interval,
            **settings,
        },
    )

    return Dataset(bars=frame_bars, events=log, meta=meta, path=None)


def is_public(data: Dataset) -> bool:
    """Whether a dataset came from here rather than from NinjaTrader."""
    return bool(data.meta.raw.get("public", False))
