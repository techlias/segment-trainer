"""The segmentation itself, in Python: the port of ``SegmentFit.cs``.

The exporter can only write the settings it was run with, so re-tuning the
tolerance through NinjaTrader means re-exporting a history that takes minutes.
With the fit here, the bars alone are enough: :func:`refit` re-cuts them at any
tolerance, by any method, and hands back a :class:`~segments.dataset.Dataset`
that :mod:`segments.replay` and :mod:`segments.plot` read exactly as they read a
real export.

TWO IMPLEMENTATIONS, ONE ANSWER

    A port is a promise, and :func:`verify` is how the promise is kept: it
    re-cuts an export's own bars with that export's own settings and compares
    the events it gets against the ones NinjaTrader wrote. Same input, same
    output, or the port is wrong. Run it once per new export; it is the only
    thing standing between a Python experiment and a chart that disagrees with
    it.

    Events are compared as a set per bar, not as an ordered list, because the
    exporter iterates a HashSet and so the order within one bar is arbitrary.
    It never matters: no bar ever adds and removes the same pivot, so every
    ordering of a bar's events leaves the same set behind.

WHERE THIS DIFFERS FROM THE C#

    Only in speed. ``SegmentFit`` rescans every merge cost on every round, which
    is quadratic and perfectly readable at one fit per bar close. Here there is a
    fit per bar of the whole history, in a language with a constant factor fifty
    times worse, so the costs are cached and only the two that a merge actually
    invalidates are recomputed. The merges chosen, and their order, are
    identical - which ``test_fit.py`` pins down against a literal transcription
    of the C# loop.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import pandas as pd

from .dataset import SOURCE_PRICE, Dataset, Meta

METHODS = ("BottomUp", "DouglasPeucker", "SlidingWindow")


def error(price: np.ndarray, scale: np.ndarray, a: int, b: int, margin: int = 0) -> tuple[float, int]:
    """How badly one straight line from a to b speaks for the bars between them.

    The largest vertical gap between a bar and that line, divided by the average
    ATR over the same bars - so the number is unitless and one tolerance holds
    across instruments, sessions and methods.

    Returns that error and, for the callers that need somewhere to cut, the worst
    bar that still leaves ``margin`` bars either side of it. That comes back as
    -1 when no such bar strays from the line at all.
    """
    span = b - a
    if span <= 1:
        return 0.0, -1

    k = np.arange(a + 1, b)
    line = price[a] + (price[b] - price[a]) * (k - a) / span
    gap = np.abs(price[k] - line)

    worst = float(gap.max())

    # Strictly greater than nothing, matching the C#: a span whose bars all sit
    # exactly on the line offers no cut, however wide the margin allows.
    inside = gap > 0
    if margin:
        inside &= (k - a >= margin) & (b - k >= margin)

    at = int(k[int(np.argmax(np.where(inside, gap, -1.0)))]) if inside.any() else -1

    average = float(scale[a : b + 1].mean())
    if average <= 0:
        return (float("inf") if worst > 0 else 0.0), at

    return worst / average, at


def bottom_up(price: np.ndarray, scale: np.ndarray, tolerance: float, min_bars: int,
              phase: int = 0) -> list[int]:
    """A piece per pair of bars, then merge the cheapest neighbouring pair until
    the cheapest one left would break the tolerance.

    Pieces shorter than ``min_bars`` are merged first and whatever they cost,
    since the alternative is a one bar stub; among those the cheapest still goes
    first.

    ``phase`` is the parity of the window's oldest BAR NUMBER, and the pairs are
    laid out from there rather than from the start of the window. It looks like a
    detail and it decides whether the output means anything: a window that slides
    forward one bar re-pairs every bar inside it if the pairs start where the
    window starts, the cascade of merges begins somewhere else, and it ends
    somewhere else. On 3 minute MNQ the pivot sets of consecutive bars then had
    nothing whatever in common - the fit alternating between two disjoint
    answers, one per parity. Anchored to the bar numbers, 98% of the pivots
    survive from one bar to the next.

    ``costs[i]`` is what merging the piece at ``i`` with the one after it would
    cost, which is the same thing as dropping the pivot between them. A merge
    only invalidates the two costs either side of where it happened, so those are
    the only two recomputed - everything else shifts along untouched.
    """
    n = len(price)

    pivot = [0] + list(range(2 if phase == 0 else 1, n, 2))
    if pivot[-1] != n - 1:
        pivot.append(n - 1)

    costs = [error(price, scale, pivot[i], pivot[i + 2])[0] for i in range(len(pivot) - 2)]

    while len(pivot) > 2:
        forced_at, forced_cost = -1, float("inf")
        best_at, best_cost = -1, float("inf")

        for i in range(len(pivot) - 2):
            cost = costs[i]
            forced = (pivot[i + 1] - pivot[i] < min_bars) or (pivot[i + 2] - pivot[i + 1] < min_bars)

            if forced:
                if cost < forced_cost:
                    forced_cost, forced_at = cost, i
            elif cost < best_cost:
                best_cost, best_at = cost, i

        if forced_at >= 0:
            drop = forced_at
        elif best_cost <= tolerance:
            drop = best_at
        else:
            break

        del pivot[drop + 1]
        del costs[drop]

        for i in (drop - 1, drop):
            if 0 <= i < len(pivot) - 2:
                costs[i] = error(price, scale, pivot[i], pivot[i + 2])[0]

    return pivot


def douglas_peucker(price: np.ndarray, scale: np.ndarray, tolerance: float, min_bars: int) -> list[int]:
    """The whole window as one line, split at the bar furthest off it, and the
    same again on each half until every piece is inside the tolerance.

    An explicit stack rather than recursion - same order of work, no depth to
    worry about on a long window.
    """
    n = len(price)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True

    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()

        gap, at = error(price, scale, a, b, margin=min_bars)

        # Within tolerance, or nowhere left to cut that would leave two pieces
        # worth drawing - in which case the span stands whatever its error.
        if gap <= tolerance or at < 0:
            continue

        keep[at] = True
        stack.append((a, at))
        stack.append((at, b))

    return [int(i) for i in np.flatnonzero(keep)]


def sliding_window(price: np.ndarray, scale: np.ndarray, tolerance: float, min_bars: int) -> list[int]:
    """From the last vertex, take one more bar for as long as the line holds, and
    cut where it stops holding.

    A tail too short to stand on its own is given to the piece before it: a stub
    at the right hand edge is where it would be most misread.
    """
    n = len(price)
    pivot = [0]
    a = 0

    while a < n - 1:
        b = min(a + max(min_bars, 1), n - 1)

        while b + 1 <= n - 1 and error(price, scale, a, b + 1)[0] <= tolerance:
            b += 1

        pivot.append(b)
        a = b

    if len(pivot) > 2 and pivot[-1] - pivot[-2] < min_bars:
        del pivot[-2]

    return pivot


def segment(
    price: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
    tolerance: float,
    min_bars: int,
    method: str = "BottomUp",
    phase: int = 0,
) -> list[int]:
    """Cuts one window into straight pieces, as indices into ``price``.

    The first index is always 0 and the last always ``len(price) - 1``: those two
    are where the window was cut, not turns, and the exporter never logs them.

    ``phase`` is the parity of the window's oldest bar number, and only bottom up
    uses it - see :func:`bottom_up`. A sweep that slides the window has to pass
    it or the answers will not be comparable from one bar to the next.
    """
    price = np.asarray(price, dtype=float)
    scale = np.asarray(scale, dtype=float)

    if len(price) < 2:
        return [0]

    if method == "DouglasPeucker":
        return douglas_peucker(price, scale, tolerance, min_bars)
    if method == "SlidingWindow":
        return sliding_window(price, scale, tolerance, min_bars)
    if method == "BottomUp":
        return bottom_up(price, scale, tolerance, min_bars, phase)

    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def atr(bars: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder's ATR, seeded the way the indicator seeds it.

    Only correct if the frame starts where NinjaTrader started - the first bar of
    the chart. Recomputing it over a slice gives a different, and wrong, series,
    which is why :func:`refit` keeps the exported column unless it is explicitly
    asked for another period.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    out = np.empty(len(bars), dtype=float)
    out[0] = high[0] - low[0]

    for i in range(1, len(bars)):
        true_range = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        out[i] = (out[i - 1] * (period - 1) + true_range) / period

    return out


def source_price(bars: pd.DataFrame, source: str) -> np.ndarray:
    if source not in SOURCE_PRICE:
        raise ValueError(f"unknown source {source!r}, expected one of {sorted(SOURCE_PRICE)}")

    return SOURCE_PRICE[source](bars).to_numpy(dtype=float)


def events(
    bars: pd.DataFrame,
    *,
    method: str,
    source: str,
    tolerance: float,
    min_bars: int,
    window: int,
    scale: np.ndarray | None = None,
) -> pd.DataFrame:
    """The whole causal sweep: one fit per bar, over that bar and nothing after.

    Emits the same log ``SegmentExport`` writes - a row each time the set of
    interior pivots changes - and for the same reasons. Only interior pivots
    count: the ends of each fit are artefacts of where the window was cut.

    The guards at the top of the loop are the exporter's, bar for bar, because
    agreeing with it about the first few bars is part of agreeing with it.
    """
    price = source_price(bars, source)
    scale = atr(bars, 14) if scale is None else np.asarray(scale, dtype=float)
    index = bars.index.to_numpy()

    held: dict[int, float] = {}
    rows: list[tuple[int, str, int, float]] = []

    for position in range(len(bars)):
        if position < max(min_bars, 2) + 1:
            continue

        n = min(window, position + 1)
        if n < 4:
            continue

        start = position - n + 1
        oldest = int(index[start])

        pivots = segment(
            price[start : position + 1], scale[start : position + 1],
            tolerance, min_bars, method, oldest % 2,
        )

        at_bar = int(index[position])
        found = {int(index[start + i]): float(price[start + i]) for i in pivots[1:-1]}

        for bar in sorted(found):
            if bar not in held:
                rows.append((at_bar, "+", bar, found[bar]))

        for bar in sorted(held):
            # A pivot that has left the window is frozen, not withdrawn: it
            # stopped being re-examined, which is not the same as being taken
            # back.
            if bar >= oldest and bar not in found:
                rows.append((at_bar, "-", bar, held[bar]))

        held = found

    log = pd.DataFrame(rows, columns=["at_bar", "event", "pivot_bar", "pivot_price"])

    # The same dtypes a loaded export comes back with, so a re-cut dataset and a
    # read one are interchangeable down to the comparison.
    return log.astype({"at_bar": "int64", "event": "string", "pivot_bar": "int64", "pivot_price": "float64"})


def refit(
    data: Dataset,
    *,
    method: str | None = None,
    source: str | None = None,
    tolerance: float | None = None,
    min_bars: int | None = None,
    window: int | None = None,
    atr_period: int | None = None,
) -> Dataset:
    """The same bars, cut again, as a dataset the rest of the package can read.

    Everything not named keeps the exported setting, so ``refit(data,
    tolerance=0.5)`` answers exactly the question it looks like it answers.

    The ATR column is reused unless ``atr_period`` says otherwise - see
    :func:`atr` for why recomputing it is only safe from the first bar of the
    chart.
    """
    settings = {
        "method": method or data.meta.method,
        "source": source or data.meta.source,
        "tolerance": data.meta.tolerance if tolerance is None else tolerance,
        "minbars": data.meta.minbars if min_bars is None else min_bars,
        "window": data.meta.window if window is None else window,
        "atr": data.meta.atr if atr_period is None else atr_period,
    }

    bars = data.bars.copy()
    if atr_period is not None:
        bars["atr"] = atr(bars, atr_period)
    bars["mid"] = (bars["high"] + bars["low"]) / 2

    log = events(
        bars,
        method=settings["method"],
        source=settings["source"],
        tolerance=settings["tolerance"],
        min_bars=settings["minbars"],
        window=settings["window"],
        scale=bars["atr"].to_numpy(dtype=float),
    )

    meta = dataclasses.replace(
        data.meta,
        method=settings["method"],
        source=settings["source"],
        tolerance=float(settings["tolerance"]),
        minbars=int(settings["minbars"]),
        window=int(settings["window"]),
        atr=int(settings["atr"]),
        raw={**data.meta.raw, **settings, "refit": "segments.fit"},
    )

    return Dataset(bars=bars, events=log, meta=meta, path=data.path)


def verify(data: Dataset) -> pd.DataFrame:
    """Re-cuts an export's own bars with its own settings and diffs the events.

    An empty frame means the port and ``SegmentFit.cs`` agree bar for bar, which
    is the whole claim this module makes. Anything else lists what one produced
    and the other did not:

        at_bar, event, pivot_bar, side   -   side being 'export' or 'python'

    Compared as a set per bar, since the order within one bar is arbitrary in the
    export and never changes what the replay ends up holding.
    """
    mine = refit(data)

    def keys(frame: pd.DataFrame) -> set[tuple[int, str, int]]:
        if frame.empty:
            return set()
        return {
            (int(row.at_bar), str(row.event).strip(), int(row.pivot_bar))
            for row in frame.itertuples()
        }

    theirs = keys(data.events)
    ours = keys(mine.events)

    rows = [(*key, "export") for key in sorted(theirs - ours)]
    rows += [(*key, "python") for key in sorted(ours - theirs)]

    return pd.DataFrame(rows, columns=["at_bar", "event", "pivot_bar", "side"])
