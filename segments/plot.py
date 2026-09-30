"""A static picture of the reconstruction, for checking it against the chart.

The point of this module is falsification, not presentation: put the same
instrument, period and settings on a NinjaTrader chart with PriceSegments, put
the same bar here, and the two pictures either agree or the pipeline is wrong
somewhere. It is the cheapest test in the project and the only one that catches
a misunderstanding of the data rather than a bug in the code.

It draws segments, not candles. That is the trainer's view, and the brief's
argument for it stands: the shape is what is being learnt, and the bars are
noise around it. Two options put the bars back for when what is being checked is
the fit rather than the reading - ``sticks=True`` for a thin high-low line, and
``candles=True`` for the whole of it, open and close as well.

Both stay off by default, and the candles are drawn muted and hollow rather than
in red and green. The segments are the only saturated colour on the chart on
purpose: candles in full colour would compete for exactly the attention the brief
wants on the shape.

    One thing the candles make obvious that the line chart hides: the vertices do
    not touch the wicks. The fit runs through ONE price per bar - the close, or
    the middle of the bar, whatever Source says - so a turn sits where that
    series turned, not at the high or the low of the bar. It looks like an error
    for about a second, and it is the most useful second the candles buy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .dataset import Dataset
from .replay import Replay, at

UP = "#3aa76d"
DOWN = "#cc5555"
PROVISIONAL = "#8a8f98"
HIDDEN = "#d9dce1"
LEVEL = "#b07d2b"
CANDLE = "#b9bec6"

#: How wide a candle body is, in bars. Under 1 so neighbours do not touch.
BODY = 0.62


def draw(
    dataset: Dataset,
    bar: int,
    history: int = 150,
    reveal: int = 0,
    sticks: bool = False,
    candles: bool = False,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """The segmentation as it was knowable at ``bar``.

    ``history`` is how many bars of context to show, and ``reveal`` how many bars
    past ``bar`` to draw in grey - the answer, for after the guess. With
    ``reveal=0`` nothing past ``bar`` is drawn at all, which is the state the
    question is asked in.

    Note what ``reveal`` does NOT do: it never re-reads the segmentation. The
    revealed bars are shown as price, and the segments stay as they were at
    ``bar``, because the point of the reveal is to see what happened next to the
    read you made, not to be shown a tidier read of the same chart.
    """
    replay = at(dataset, bar)
    ax = ax or plt.subplots(figsize=(13, 6))[1]

    price = dataset.source_price()
    left = max(dataset.first_bar, bar - history)
    right = min(dataset.last_bar, bar + reveal)

    if candles:
        candlesticks(ax, dataset.bars.loc[left:bar])
    elif sticks:
        window = dataset.bars.loc[left:bar]
        ax.vlines(window.index, window["low"], window["high"], color="#c8ccd2", linewidth=0.8, zorder=1)

    for segment in replay.segments():
        if segment.end_bar < left:
            continue

        colour = PROVISIONAL if segment.provisional else (UP if segment.up else DOWN)
        ax.plot(
            [segment.start_bar, segment.end_bar],
            [segment.start_price, segment.end_price],
            color=colour,
            linewidth=2.4,
            linestyle="--" if segment.provisional else "-",
            solid_capstyle="round",
            zorder=3,
        )

    turns = [pivot for pivot in replay.pivots() if pivot.bar >= left]
    if turns:
        ax.scatter(
            [pivot.bar for pivot in turns],
            [pivot.price for pivot in turns],
            s=26,
            color="#2c3038",
            zorder=4,
        )

    if reveal and right > bar:
        ax.plot(
            price.loc[bar:right].index,
            price.loc[bar:right].to_numpy(),
            color=HIDDEN,
            linewidth=1.4,
            zorder=2,
        )
        ax.axvline(bar, color="#b0b4ba", linewidth=1.0, linestyle=":", zorder=1)

    ax.set_xlim(left, max(right, bar) + 2)
    ax.set_title(
        f"{dataset.meta.instrument}  {dataset.meta.period}   bar {bar}   "
        f"{dataset.bars.loc[bar, 'time']:%Y-%m-%d %H:%M}\n"
        f"{dataset.meta.method}/{dataset.meta.source}  tolerance {dataset.meta.tolerance} ATR"
        f"   {len(turns)} turns in view",
        fontsize=10,
        loc="left",
    )
    ax.set_xlabel("bar")
    ax.grid(True, alpha=0.15)

    return ax


def candlesticks(ax: "matplotlib.axes.Axes", window) -> None:
    """Open, high, low and close as candles, behind everything else.

    Direction is carried by whether the body is hollow rather than by colour,
    the way it was before colour screens - see this module's note on why the
    bars stay muted.

    The wick is drawn in two pieces, above the body and below it, rather than as
    one line from low to high with the body laid over it. A hollow body would
    let that line show through, and a candle with a line down the middle of it
    reads as something other than a candle.

    A body with no height is not given one: an open equal to its close is a real
    thing that happened, and a doji should look like a doji rather than like a
    thin bar of some arbitrary minimum.
    """
    top = window[["open", "close"]].max(axis=1)
    bottom = window[["open", "close"]].min(axis=1)

    ax.vlines(window.index, top, window["high"], color=CANDLE, linewidth=0.8, zorder=1)
    ax.vlines(window.index, window["low"], bottom, color=CANDLE, linewidth=0.8, zorder=1)

    height = top - bottom
    solid = height > 0

    if solid.any():
        ax.bar(
            window.index[solid],
            height[solid],
            bottom=bottom[solid],
            width=BODY,
            facecolor=["none" if up else CANDLE
                       for up in (window["close"] >= window["open"])[solid]],
            edgecolor=CANDLE,
            linewidth=0.8,
            zorder=1,
        )

    flat = ~solid
    if flat.any():
        ax.hlines(
            window["open"][flat],
            window.index[flat] - BODY / 2,
            window.index[flat] + BODY / 2,
            color=CANDLE,
            linewidth=0.8,
            zorder=1,
        )


def annotate(ax: "matplotlib.axes.Axes", label, right: int | None = None) -> "matplotlib.axes.Axes":
    """Puts the tagger's reading on a chart already drawn.

    Two things, and no more: what each turn was called - HH, HL, LH, LL - and
    the protective level, the one price has to close through for the structure
    to be broken. Those are the two things a person reading the chart is being
    asked to see, and a third would start hiding them.
    """
    left, edge = ax.get_xlim()

    for swing in label.swings:
        if swing.bar < left or swing.kind == "first":
            continue

        ax.annotate(
            swing.kind,
            (swing.bar, swing.price),
            textcoords="offset points",
            xytext=(0, 9 if swing.high else -16),
            ha="center",
            fontsize=8,
            color="#2c3038",
        )

    if label.protective is not None:
        ax.hlines(
            label.protective,
            max(label.protective_bar, left),
            right if right is not None else edge,
            color=LEVEL,
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )
        ax.annotate(
            "protective",
            (right if right is not None else edge, label.protective),
            textcoords="offset points",
            xytext=(-4, 4),
            ha="right",
            fontsize=8,
            color=LEVEL,
        )

    return ax


def save(dataset: Dataset, bar: int, path: str | Path, **kwargs) -> Path:
    """Draws one bar's view and writes it to a PNG."""
    ax = draw(dataset, bar, **kwargs)
    path = Path(path)

    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=130)
    plt.close(ax.figure)

    return path


def contact_sheet(dataset: Dataset, at_bars: list[int], path: str | Path, **kwargs) -> Path:
    """Several bars on one page, for sweeping a history quickly.

    A dozen of these at random bars is how you decide whether a tolerance is set
    anywhere near right - far faster than staring at one chart and far more
    honest than a single carefully chosen example.
    """
    rows = (len(at_bars) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(20, 4.2 * rows), squeeze=False)

    for axis, bar in zip(axes.flatten(), at_bars):
        draw(dataset, bar, ax=axis, **kwargs)

    for axis in axes.flatten()[len(at_bars):]:
        axis.axis("off")

    path = Path(path)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)

    return path
