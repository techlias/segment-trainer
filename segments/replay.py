"""Putting the segmentation back together, one bar at a time.

The export is a log of changes to the pivot set, so the pivots known at bar N
are what is left after replaying every event with ``at_bar <= N``: add on ``+``,
remove on ``-``. That is the only reconstruction this module offers for
features, and it is exact.

The one thing here that looks forward is :func:`confirmation`, which reads the
whole file to find out when each pivot settled. That number is the point of the
dataset - it says how late a swing becomes readable - and it is measurement, not
a feature. Nothing that feeds a label or a question may use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .dataset import Dataset


@dataclass(frozen=True)
class Pivot:
    """A turn the segmentation found, and when it was first seen."""

    bar: int
    price: float
    seen_at: int

    @property
    def age(self) -> int:
        """Bars between the turn happening and it being visible."""
        return self.seen_at - self.bar


@dataclass(frozen=True)
class Segment:
    """One straight piece, from one pivot to the next.

    ``provisional`` marks the piece running from the newest known pivot to the
    bar being looked at. It has no right hand end yet: it is the leg in progress,
    the only one that can still change, and the one a person reading the chart is
    actually being asked about.
    """

    start_bar: int
    start_price: float
    end_bar: int
    end_price: float
    provisional: bool

    @property
    def bars(self) -> int:
        return self.end_bar - self.start_bar

    @property
    def change(self) -> float:
        return self.end_price - self.start_price

    @property
    def up(self) -> bool:
        return self.change >= 0

    def slope(self) -> float:
        """Price per bar. Divide by the ATR of the piece to compare two of them."""
        return self.change / self.bars if self.bars else 0.0


class Replay:
    """The pivot set as at some bar, advanced forward and never backward.

    Forward only because that is what the data is: a log. Rewinding would mean
    replaying from the start, which :meth:`at` does for the one-off case, and
    which :meth:`walk` avoids by never going back.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._events = dataset.events.sort_values("at_bar", kind="stable").reset_index(drop=True)
        self._next = 0
        self._bar = dataset.first_bar - 1
        self._held: dict[int, Pivot] = {}

    @property
    def bar(self) -> int:
        """The bar this replay has been advanced to."""
        return self._bar

    def advance_to(self, bar: int) -> "Replay":
        if bar < self._bar:
            raise ValueError(f"a replay only goes forward: at {self._bar}, asked for {bar}")

        at_bar = self._events["at_bar"].to_numpy()
        event = self._events["event"].to_numpy()
        pivot_bar = self._events["pivot_bar"].to_numpy()
        price = self._events["pivot_price"].to_numpy()

        while self._next < len(self._events) and at_bar[self._next] <= bar:
            i = self._next
            if str(event[i]).strip() == "+":
                self._held[int(pivot_bar[i])] = Pivot(
                    bar=int(pivot_bar[i]), price=float(price[i]), seen_at=int(at_bar[i])
                )
            else:
                self._held.pop(int(pivot_bar[i]), None)
            self._next += 1

        self._bar = bar
        return self

    def pivots(self) -> list[Pivot]:
        """The turns knowable at this bar, oldest first."""
        return [self._held[bar] for bar in sorted(self._held)]

    def segments(self) -> list[Segment]:
        """The pivots joined up, with the leg in progress on the end.

        The export logs no pivot at the current bar - the right hand edge is
        where the window was cut, not a turn - so the last piece is closed off at
        the bar being looked at and flagged provisional.
        """
        pivots = self.pivots()
        if not pivots:
            return []

        pieces = [
            Segment(a.bar, a.price, b.bar, b.price, provisional=False)
            for a, b in zip(pivots, pivots[1:])
        ]

        last = pivots[-1]
        if self._bar > last.bar:
            price = self.dataset.source_price()
            pieces.append(
                Segment(last.bar, last.price, self._bar, float(price.loc[self._bar]), provisional=True)
            )

        return pieces

    def at(self, bar: int) -> "Replay":
        """This same replay, advanced to that bar. Sugar for reading."""
        return self.advance_to(bar)


def at(dataset: Dataset, bar: int) -> Replay:
    """The pivot set as at one bar, replayed from the start.

    For a single question. Sweeping a whole history means :func:`walk`, which
    does it in one pass instead of one pass per bar.
    """
    return Replay(dataset).advance_to(bar)


def walk(dataset: Dataset, start: int | None = None, stop: int | None = None) -> Iterator[Replay]:
    """Every bar in turn, with the replay advanced to it.

    The same Replay object is yielded each time, mutated - which keeps the sweep
    linear instead of quadratic. Anything worth keeping has to be taken out of it
    before the next bar, not stored as a reference.
    """
    replay = Replay(dataset)

    first = dataset.first_bar if start is None else start
    last = dataset.last_bar if stop is None else stop

    for bar in dataset.bars.index:
        if bar < first:
            continue
        if bar > last:
            break
        yield replay.advance_to(int(bar))


def confirmation(dataset: Dataset) -> pd.DataFrame:
    """When each pivot settled, and how long that took.

    Reads the whole file, so this is the one function here that knows the future.
    It exists to answer the question the trainer stands or falls on - how late is
    a swing knowable - and its output belongs in a report, never in a feature.

    Columns, one row per pivot bar:

        pivot_bar     the turn
        price         its price
        first_seen    the earliest bar it was ever visible at
        settled_at    the bar of its last event, after which it never changed
        held          whether it is still a pivot at the end of the data
        first_lag     first_seen - pivot_bar, how soon it could be guessed at
        lag           settled_at - pivot_bar, how late it could be trusted
        flips         how many times it appeared and vanished
    """
    events = dataset.events
    if events.empty:
        return pd.DataFrame(
            columns=["pivot_bar", "price", "first_seen", "settled_at", "held", "first_lag", "lag", "flips"]
        )

    plus = events["event"].astype("string").str.strip() == "+"

    grouped = events.groupby("pivot_bar", sort=True)
    out = pd.DataFrame(
        {
            "price": grouped["pivot_price"].last(),
            "first_seen": events[plus].groupby("pivot_bar")["at_bar"].min(),
            "settled_at": grouped["at_bar"].max(),
            "flips": grouped.size(),
        }
    )

    # A pivot whose last event was a '-' was taken back and never restored.
    last_event = grouped["event"].last().astype("string").str.strip()
    out["held"] = last_event == "+"

    out = out.reset_index()
    out["first_lag"] = out["first_seen"] - out["pivot_bar"]
    out["lag"] = out["settled_at"] - out["pivot_bar"]

    return out[["pivot_bar", "price", "first_seen", "settled_at", "held", "first_lag", "lag", "flips"]]


def lag_summary(dataset: Dataset) -> pd.Series:
    """The confirmation lag as a handful of numbers, for the report.

    Read the median as the honest delay of this dataset: half the swings were
    trustworthy that many bars after they happened, and the rest later. If that
    number is large relative to the legs you are trying to trade, the timeframe
    is wrong, not the rules.
    """
    table = confirmation(dataset)
    held = table[table["held"]]

    if held.empty:
        return pd.Series(dtype="float64")

    lag = held["lag"]

    return pd.Series(
        {
            "pivots": float(len(held)),
            "withdrawn": float(len(table) - len(held)),
            "first_lag_median": float(held["first_lag"].median()),
            "lag_min": float(lag.min()),
            "lag_median": float(lag.median()),
            "lag_p90": float(lag.quantile(0.90)),
            "lag_max": float(lag.max()),
            "flips_mean": float(held["flips"].mean()),
        }
    )


def legs(dataset: Dataset) -> pd.DataFrame:
    """Every confirmed piece, with its shape in ATRs, as a features table.

    Built from the settled pivots, so this is the offline view: what the legs of
    this history turned out to be. The trainer's own view of any single bar comes
    from :func:`walk`, which is late by the confirmation lag and is the only one
    a rule may read.

    Columns: start_bar, end_bar, bars, change, ticks, atr, amplitude (change in
    ATRs), slope (per bar, in ATRs), up.
    """
    pivots = confirmation(dataset)
    pivots = pivots[pivots["held"]].sort_values("pivot_bar")

    if len(pivots) < 2:
        return pd.DataFrame(
            columns=["start_bar", "end_bar", "bars", "change", "ticks", "atr", "amplitude", "slope", "up"]
        )

    atr = dataset.bars["atr"]
    ticksize = dataset.meta.ticksize or 1.0

    rows = []
    for (_, a), (_, b) in zip(pivots.iterrows(), pivots.iloc[1:].iterrows()):
        start, end = int(a["pivot_bar"]), int(b["pivot_bar"])
        change = float(b["price"] - a["price"])
        bars = end - start
        local = float(atr.loc[start:end].mean())

        rows.append(
            {
                "start_bar": start,
                "end_bar": end,
                "bars": bars,
                "change": change,
                "ticks": change / ticksize,
                "atr": local,
                "amplitude": change / local if local else float("nan"),
                "slope": (change / bars) / local if bars and local else float("nan"),
                "up": change >= 0,
            }
        )

    return pd.DataFrame(rows)
