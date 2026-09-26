"""Reads the dataset SegmentExport writes, without ever looking ahead.

    from segments import load, replay

    data = load(".../MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w250")
    print(data.meta, data.check())

    view = replay.at(data, 4000)          # only what bar 4000 could know
    for segment in view.segments():
        print(segment.start_bar, segment.end_bar, segment.provisional)

    print(replay.lag_summary(data))       # how late a swing becomes readable

The rule the whole project rests on: anything that feeds a label, a question or
a feature comes from :func:`segments.replay.at` or :func:`segments.replay.walk`.
Only the measurement functions - :func:`segments.replay.confirmation`,
:func:`segments.replay.lag_summary`, :func:`segments.replay.legs` - read the
whole file, and they say so.
"""

from .dataset import Dataset, Meta, find, load
from .replay import Pivot, Replay, Segment, at, confirmation, lag_summary, legs, walk

__all__ = [
    "Dataset",
    "Meta",
    "Pivot",
    "Replay",
    "Segment",
    "at",
    "confirmation",
    "find",
    "lag_summary",
    "legs",
    "load",
    "walk",
]
