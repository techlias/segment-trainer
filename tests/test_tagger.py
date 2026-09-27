"""The tagger is tested against structures built by hand.

Every fixture here is a zigzag whose turns are stated outright, so the expected
label follows from the rules rather than from whatever the segmentation happened
to find. On real bars you cannot tell a correct label from a plausible one, and
"plausible" is exactly what a rule engine produces when a rule is subtly wrong.

The test that matters most is the last one: that a label at bar N is the same
whether or not the file continues past N.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from segments import dataset as dataset_module
from segments import replay, tagger

META = {
    "instrument": "MNQ 12-26", "period": "3 Minute", "session": "CME US Index Futures ETH",
    "method": "BottomUp", "source": "Median", "ticksize": 0.25, "tolerance": 1.0,
    "minbars": 3, "window": 250, "atr": 14, "exported": "2026-09-27T08:00:00",
}

ATR = 4.0
SEEN_AFTER = 3  # every turn in these fixtures is confirmed three bars later

# The oldest pivot in the chain can never be a swing - there is nothing before it
# to have reversed from - so a fixture needs one more turn than the swings it
# means to test. Every one below is read at bar 65, by which point the turns up
# to bar 60 are confirmed.
AT = 65


def build(points: list[tuple[int, float]], last: int, closes: dict[int, float] | None = None):
    """A dataset whose turns are exactly ``points``.

    Price runs straight between the turns, so the mid of every bar is on the
    path and the turns are where they are said to be. Closes follow the mid
    unless one is named - which is how a break is staged, since a break is about
    a close and not about the shape.
    """
    bars = np.arange(last + 1)
    path = np.interp(bars, [bar for bar, _ in points], [price for _, price in points])

    close = path.copy()
    for bar, price in (closes or {}).items():
        close[bar] = price

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-21 17:00", periods=len(bars), freq="3min"),
            "open": path, "high": path + 1, "low": path - 1, "close": close,
            "volume": 1000.0, "mid": path, "atr": ATR, "new_session": 0,
        },
        index=pd.Index(bars, name="bar"),
    )

    events = pd.DataFrame(
        [
            {"at_bar": bar + SEEN_AFTER, "event": "+", "pivot_bar": bar, "pivot_price": price}
            for bar, price in points
            if 0 < bar and bar + SEEN_AFTER <= last
        ]
    ).astype({"at_bar": "int64", "event": "string", "pivot_bar": "int64", "pivot_price": "float64"})

    return dataset_module.Dataset(
        bars=frame, events=events, meta=dataset_module.Meta.parse("# " + json.dumps(META)), path="."
    )


# A clean stair up: low 110, high 140, higher low 130, higher high 160.
UP = [(0, 100.0), (10, 120.0), (20, 110.0), (30, 140.0), (40, 130.0), (50, 160.0),
      (60, 150.0), (70, 180.0)]

# The mirror.
DOWN = [(0, 180.0), (10, 160.0), (20, 170.0), (30, 140.0), (40, 150.0), (50, 120.0),
        (60, 130.0), (70, 100.0)]


def test_a_stair_up_is_an_uptrend():
    one = tagger.label(build(UP, AT), AT)

    assert one.state == tagger.UPTREND
    assert one.reason["high_kind"] == "HH"
    assert one.reason["low_kind"] == "HL"
    assert "HH 160.00" in one.says()


def test_a_stair_down_is_a_downtrend():
    one = tagger.label(build(DOWN, AT), AT)

    assert one.state == tagger.DOWNTREND
    assert one.reason["high_kind"] == "LH"
    assert one.reason["low_kind"] == "LL"


def test_the_protective_level_is_the_low_before_the_last_high():
    """Not the newest low. The newest one may be a bar old with nothing built on
    it yet, and protecting that turns every wiggle into a break."""
    one = tagger.label(build(UP, AT), AT)

    assert one.protective == 130.0     # the low at bar 40, before the high at 50
    assert one.protective_bar == 40


def test_a_close_through_the_protective_level_is_a_break():
    data = build(UP, AT, closes={64: 125.0})   # under 130, by more than break_atr
    one = tagger.label(data, AT)

    assert one.state == tagger.BREAK
    assert one.reason["broke_at"] == 64
    assert one.reason["level"] == 130.0
    assert "below the swing low 130.00" in one.says()


def test_a_poke_through_the_level_is_not_a_break():
    """Inside break_atr of the level - 0.1 ATR is 0.4 here - it is noise."""
    data = build(UP, AT, closes={64: 129.8})
    one = tagger.label(data, AT)

    assert one.state != tagger.BREAK


def test_two_swings_at_the_same_level_do_not_make_a_trend():
    # The second high is within equal_atr (0.15 x 4 = 0.6) of the first.
    points = [(0, 100.0), (10, 120.0), (20, 110.0), (30, 140.0), (40, 130.0), (50, 140.4),
              (60, 132.0), (70, 180.0)]
    one = tagger.label(build(points, AT), AT)

    assert one.reason["high_kind"] == "EQH"
    assert one.state == tagger.RANGE


def test_mixed_swings_are_a_range():
    # Higher low, lower high: coiling, not trending.
    points = [(0, 100.0), (10, 160.0), (20, 110.0), (30, 150.0), (40, 120.0), (50, 145.0),
              (60, 125.0), (70, 150.0)]
    one = tagger.label(build(points, AT), AT)

    assert one.state == tagger.RANGE
    assert one.reason["high"] == 150.0
    assert one.reason["low"] == 110.0


def test_a_shrinking_push_is_weakening():
    # Pushes of 40 then 18: the second is under three quarters of the first.
    points = [(0, 100.0), (10, 120.0), (20, 110.0), (30, 150.0), (40, 140.0), (50, 158.0),
              (60, 148.0), (70, 175.0)]
    one = tagger.label(build(points, AT), AT)

    assert one.state == tagger.WEAKENING
    assert one.reason["why"] == "push"
    assert one.reason["was"] == tagger.UPTREND
    assert "the push into bar 50" in one.says()


def test_a_deep_pullback_is_weakening():
    # A push of 37, then a pullback giving back 30 of it - over four fifths.
    points = [(0, 100.0), (10, 120.0), (20, 110.0), (30, 150.0), (40, 118.0), (50, 155.0),
              (60, 125.0), (70, 170.0)]
    one = tagger.label(build(points, AT), AT)

    assert one.state == tagger.WEAKENING
    assert one.reason["why"] == "pullback"


def test_too_few_swings_says_so_rather_than_guessing():
    one = tagger.label(build([(0, 100.0), (10, 120.0), (20, 110.0)], 25), 25)

    assert one.state == tagger.UNCLEAR
    assert "needed" in one.says()


def test_a_bend_is_not_a_swing():
    """Two rising pieces of different steepness meet at a pivot that is not a
    turn. It is a fact about the fit, and it must not become a swing high."""
    points = [(0, 100.0), (10, 120.0), (20, 110.0), (30, 130.0), (40, 160.0), (50, 150.0),
              (60, 180.0), (70, 200.0)]
    data = build(points, AT)

    swings = tagger.swings_at(replay.at(data, AT))

    assert 30 not in [swing.bar for swing in swings]   # the bend: up into it, up out of it
    assert 40 in [swing.bar for swing in swings]       # a real turn


def test_a_label_does_not_change_when_the_file_gets_longer():
    """The whole discipline of the project, as one assertion. Label bar 65 from
    a file that stops there, and from one that runs to 95 - same answer, or
    something downstream is reading the future."""
    short = build(UP, AT)
    long = build(UP + [(80, 170.0), (90, 200.0)], 95)

    early = tagger.label(short, AT)
    late = tagger.label(long, AT)

    assert early.state == late.state
    assert early.reason == late.reason
    assert early.swings == late.swings


def test_the_whole_history_can_be_tagged_in_one_pass():
    data = build(UP + [(80, 170.0), (90, 200.0)], 95)
    table = tagger.table(data)

    assert len(table) == 96
    assert set(table["state"]) <= set(tagger.STATES)
    assert table.loc[AT, "state"] == tagger.label(data, AT).state

    shares = tagger.shares(data)
    assert shares.sum() == pytest.approx(1.0)
