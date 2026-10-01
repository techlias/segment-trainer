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

Both stay off by default, and the candles are drawn in a faded green and red -
the classic reading, at an opacity that keeps it underneath. Full strength they
would compete with the segments for exactly the attention the brief wants on the
shape, so they are washed out to roughly a third of it: near enough to read a
direction off without looking at anything, far enough back that the legs are
still what the eye lands on first.

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
#: The segment colours washed out towards white. Same hues, so a candle and the
#: leg drawn over it agree about which way the bar went.
CANDLE_UP = "#a9d8c0"
CANDLE_DOWN = "#eab3b3"

#: How wide a candle body is, in bars. Under 1 so neighbours do not touch.
BODY = 0.62

#: The Bollinger band, softer than anything else on the chart. It is context,
#: not a reading - if it ever competes with the segments it is drawn wrong.
BAND = "#93a8c4"
BAND_FILL = "#eef1f6"

#: The RSI panel. One hue, with its own levels in a lighter tone of it.
MOMENTUM = "#7a6ea8"
MOMENTUM_GRID = "#d8d3e6"


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

    Green up and red down, the classic reading, faded - see this module's note
    on how far back and why. Wick and body take the same colour: at this
    strength an outline in a second tone would only muddy it.

    A body with no height is not given one: an open equal to its close is a real
    thing that happened, and a doji should look like a doji rather than like a
    thin bar of some arbitrary minimum. A doji is drawn in the up colour, which
    is what `close >= open` makes it, and at one line thick the question hardly
    arises.
    """
    top = window[["open", "close"]].max(axis=1)
    bottom = window[["open", "close"]].min(axis=1)

    colour = (window["close"] >= window["open"]).map({True: CANDLE_UP, False: CANDLE_DOWN})

    ax.vlines(window.index, window["low"], window["high"],
              color=colour.tolist(), linewidth=0.8, zorder=1)

    height = top - bottom
    solid = height > 0

    if solid.any():
        ax.bar(
            window.index[solid],
            height[solid],
            bottom=bottom[solid],
            width=BODY,
            color=colour[solid].tolist(),
            linewidth=0,
            zorder=1,
        )

    flat = ~solid
    if flat.any():
        ax.hlines(
            window["open"][flat],
            window.index[flat] - BODY / 2,
            window.index[flat] + BODY / 2,
            color=colour[flat].tolist(),
            linewidth=0.8,
            zorder=1,
        )


def bands(ax: "matplotlib.axes.Axes", frame) -> None:
    """A Bollinger band behind everything, as faint as it can be and still be
    there.

    Softer than the candles on purpose, and softer again than the segments. The
    band is here to answer "how unusual is this leg" while the eye is on the
    leg - the moment it is legible enough to read on its own it has taken the
    attention the shape was supposed to have.

    The middle is dashed, so it is never mistaken for one of the two edges at a
    glance.
    """
    ax.fill_between(frame.index, frame["lower"], frame["upper"],
                    color=BAND_FILL, zorder=0)

    for column, style in (("upper", "-"), ("lower", "-"), ("middle", "--")):
        ax.plot(frame.index, frame[column].to_numpy(),
                color=BAND, linewidth=0.9, linestyle=style, zorder=0)


def momentum(ax: "matplotlib.axes.Axes", values, period: int,
             at: int | None = None) -> "matplotlib.axes.Axes":
    """The RSI, as a panel of its own under the price.

    Fixed to 0-100 whatever the window holds. An RSI panel that rescales to
    its own extremes is worse than no panel: 70 has to sit in the same place on
    Tuesday as it did on Monday or there is nothing to read off it.

    The 30 and 70 lines are drawn, and 50 between them in a lighter hand, since
    which side of the middle momentum sits is most of what the study is for.
    """
    ax.plot(values.index, values.to_numpy(), color=MOMENTUM, linewidth=1.3, zorder=3)

    ax.axhspan(30, 70, color=MOMENTUM_GRID, alpha=0.25, zorder=0)

    for level, width in ((70, 0.9), (50, 0.7), (30, 0.9)):
        ax.axhline(level, color=MOMENTUM_GRID, linewidth=width,
                   linestyle="-" if level != 50 else ":", zorder=1)

    if at is not None:
        ax.axvline(at, color="#b0b4ba", linewidth=1.0, linestyle=":", zorder=2)

    ax.set_ylim(0, 100)
    ax.set_yticks([30, 50, 70])
    ax.set_ylabel(f"RSI {period}", fontsize=9)
    ax.grid(True, axis="x", alpha=0.15)

    return ax


def annotate(ax: "matplotlib.axes.Axes", label, right: int | None = None) -> "matplotlib.axes.Axes":
    """Puts the tagger's reading on a chart already drawn.

    Two things, and no more: what each turn was called - HH, HL, LH, LL - and
    the protective level, the one price has to close through for the structure
    to be broken. Those are the two things a person reading the chart is being
    asked to see, and a third would start hiding them.

    THE NEWEST TAG IS BRACKETED

        (LH) rather than LH. The newest swing is settled with the help of where
        price is NOW, because the leg leaving it has not finished - see
        :func:`segments.tagger.swings_at`. It is not lookahead, it is today, but
        it does mean that one tag can still change its mind while every older
        one is settled. A tag that might be withdrawn should not look like one
        that cannot.

        A display convention, so it lives here and not in the tagger: the kind
        on a :class:`segments.tagger.Swing` stays the bare name, and every
        caller that reads the data rather than looks at it sees what it always
        saw. PriceSegments draws it the same way for the same reason, which is
        the point - the two charts have to be readable as one notation.
    """
    left, edge = ax.get_xlim()
    newest = label.swings[-1].bar if label.swings else None

    for swing in label.swings:
        if swing.bar < left or swing.kind == "first":
            continue

        ax.annotate(
            f"({swing.kind})" if swing.bar == newest else swing.kind,
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
