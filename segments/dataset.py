"""Reading what SegmentExport wrote.

Two files per export, joined on the NinjaTrader bar index:

    ...-bars.csv    bar,time,open,high,low,close,volume,mid,atr,new_session
    ...-pivots.csv  at_bar,event,pivot_bar,pivot_price

Both open with one ``#`` line of JSON naming the instrument, the period, the
session template and the segmentation settings, so a file can always say what
produced it.

Nothing here interprets the data. It loads it, states what it is, and refuses
files that cannot mean what they claim - see :func:`Dataset.check`. Everything
causal happens in :mod:`segments.replay`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BAR_COLUMNS = ["bar", "time", "open", "high", "low", "close", "volume", "mid", "atr", "new_session"]
EVENT_COLUMNS = ["at_bar", "event", "pivot_bar", "pivot_price"]

#: How each SegmentSource of the indicator is rebuilt from the bar columns. Used
#: to prove that a pair of files belongs together - a pivot price has to be the
#: source price of its own bar, and if it is not, the two files came from
#: different runs.
SOURCE_PRICE = {
    "Close": lambda bars: bars["close"],
    "BodyCentre": lambda bars: (bars["open"] + bars["close"]) / 2,
    "Median": lambda bars: (bars["high"] + bars["low"]) / 2,
    "Typical": lambda bars: (bars["high"] + bars["low"] + bars["close"]) / 3,
}


@dataclass(frozen=True)
class Meta:
    """The ``#`` line, with the fields worth having as attributes."""

    instrument: str
    period: str
    session: str
    method: str
    source: str
    ticksize: float
    tolerance: float
    minbars: int
    window: int
    atr: int
    exported: str
    raw: dict

    @classmethod
    def parse(cls, line: str) -> "Meta":
        fields = json.loads(line.lstrip("#").strip())

        return cls(
            instrument=str(fields.get("instrument", "")),
            period=str(fields.get("period", "")),
            session=str(fields.get("session", "")),
            method=str(fields.get("method", "")),
            source=str(fields.get("source", "")),
            ticksize=float(fields.get("ticksize", 0) or 0),
            tolerance=float(fields.get("tolerance", 0) or 0),
            minbars=int(float(fields.get("minbars", 0) or 0)),
            window=int(float(fields.get("window", 0) or 0)),
            atr=int(float(fields.get("atr", 0) or 0)),
            exported=str(fields.get("exported", "")),
            raw=fields,
        )

    def __str__(self) -> str:
        return (
            f"{self.instrument} {self.period} ({self.session})  "
            f"{self.method}/{self.source}  tolerance {self.tolerance} ATR  "
            f"min {self.minbars} bars  window {self.window}"
        )


@dataclass(frozen=True)
class Dataset:
    """One export: the bars, the pivot events, and what made them.

    ``bars`` is indexed by bar number, not by a running integer, because every
    reference in the other file is a bar number and a dataset that starts at bar
    17 is perfectly normal - the first bars of a chart have no ATR worth the
    name yet.
    """

    bars: pd.DataFrame
    events: pd.DataFrame
    meta: Meta
    path: Path

    @property
    def first_bar(self) -> int:
        return int(self.bars.index[0])

    @property
    def last_bar(self) -> int:
        return int(self.bars.index[-1])

    def source_price(self) -> pd.Series:
        """The series the segments were fitted through, rebuilt from the bars."""
        if self.meta.source not in SOURCE_PRICE:
            raise ValueError(f"unknown source in the export header: {self.meta.source!r}")

        return SOURCE_PRICE[self.meta.source](self.bars)

    def check(self) -> list[str]:
        """Everything wrong with this pair of files, as a list of complaints.

        Cheap to run and worth running once per dataset. The three that actually
        happen in practice: two files from different runs left in the same
        folder, an export interrupted so the bars stop before the events do, and
        a hand-edited CSV.
        """
        problems: list[str] = []

        if not self.events.empty:
            if not self.events["at_bar"].is_monotonic_increasing:
                problems.append("the events are not in bar order")

            # A pivot is always learnt after the fact. One learnt at its own bar
            # or earlier would mean the export looked ahead.
            ahead = self.events[self.events["pivot_bar"] >= self.events["at_bar"]]
            if len(ahead):
                problems.append(f"{len(ahead)} events name a pivot at or after the bar they were learnt on")

            missing = ~self.events["pivot_bar"].isin(self.bars.index)
            if missing.any():
                problems.append(f"{int(missing.sum())} events name a bar that is not in the bars file")

            known = self.events[~missing]
            if len(known):
                expected = self.source_price().reindex(known["pivot_bar"]).to_numpy()
                gap = (known["pivot_price"].to_numpy() - expected)
                off = abs(gap) > (self.meta.ticksize or 1e-9) / 8
                if off.any():
                    problems.append(
                        f"{int(off.sum())} pivot prices do not match the {self.meta.source} price of their own bar "
                        "- the two files are probably from different runs"
                    )

        if self.bars.index.has_duplicates:
            problems.append("the bars file has duplicate bar numbers")

        if (self.bars["atr"] <= 0).any():
            problems.append("some bars have a non-positive ATR")

        return problems


def _read_meta(path: Path) -> Meta:
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()

    if not first.startswith("#"):
        raise ValueError(f"{path.name} does not start with the # header line")

    return Meta.parse(first)


def load(path: str | Path) -> Dataset:
    """Loads an export named by either of its two files, or by their shared stem.

    So all three of these are the same dataset, which is the point - the stem is
    what a person remembers and the suffixes are an implementation detail::

        load("MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w250")
        load(".../…-w250-bars.csv")
        load(".../…-w250-pivots.csv")
    """
    path = Path(path)
    stem = str(path)

    for suffix in ("-bars.csv", "-pivots.csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    bars_path = Path(stem + "-bars.csv")
    pivots_path = Path(stem + "-pivots.csv")

    for needed in (bars_path, pivots_path):
        if not needed.exists():
            raise FileNotFoundError(needed)

    meta = _read_meta(bars_path)

    bars = pd.read_csv(
        bars_path,
        comment="#",
        parse_dates=["time"],
        dtype={"bar": "int64", "new_session": "int8"},
    ).set_index("bar")

    events = pd.read_csv(
        pivots_path,
        comment="#",
        dtype={"at_bar": "int64", "event": "string", "pivot_bar": "int64"},
    )

    for frame, columns, name in ((bars, BAR_COLUMNS[1:], bars_path.name), (events, EVENT_COLUMNS, pivots_path.name)):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing the columns {missing}")

    return Dataset(bars=bars, events=events, meta=meta, path=bars_path.parent)


def find(folder: str | Path) -> list[Path]:
    """The stems of every export in a folder, for a CLI that takes no arguments."""
    return sorted(
        Path(str(path)[: -len("-bars.csv")])
        for path in Path(folder).glob("*-bars.csv")
    )
