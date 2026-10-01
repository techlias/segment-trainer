"""The drawing, where it says something a reader is meant to act on.

Most of what plot.py does is taste and is not testable - a colour is right or
wrong to an eye, not to an assertion. The bracket on the newest tag is not
taste. It is a claim, made to the reader, that every other tag is settled and
this one is not, and it has to mean the same thing here as it does on the
NinjaTrader chart that PriceSegments draws.

It was added to the indicator first and lived there alone for a while, which is
exactly the drift tools/statecheck exists to catch and cannot: statecheck
compares the RULES, and this is a display convention. So it is pinned here.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from segments import plot, replay, tagger  # noqa: E402
from tests.test_tagger import AT, UP, build  # noqa: E402


@pytest.fixture
def axes():
    figure, ax = plt.subplots()
    yield ax
    plt.close(figure)


#: annotate() also writes the protective level's own label, which is not a tag.
NOT_A_TAG = {"protective"}


def tags(ax) -> list[str]:
    """Every swing name drawn, in the order matplotlib holds them."""
    return [one.get_text() for one in ax.texts if one.get_text() not in NOT_A_TAG]


def reading_of(points, at: int = AT):
    data = build(points, last=at)
    return data, tagger.label(data, at, view=replay.at(data, at))


def test_the_newest_tag_is_bracketed(axes):
    data, label = reading_of(UP)
    axes.set_xlim(0, data.last_bar)

    plot.annotate(axes, label)
    drawn = tags(axes)

    assert drawn, "the fixture should have produced some tags"
    assert drawn[-1].startswith("(") and drawn[-1].endswith(")")


def test_every_older_tag_is_bare(axes):
    """The whole meaning of the convention: exactly one tag is unsettled."""
    data, label = reading_of(UP)
    axes.set_xlim(0, data.last_bar)

    plot.annotate(axes, label)
    drawn = tags(axes)

    assert len([one for one in drawn if one.startswith("(")]) == 1
    assert all(not one.startswith("(") for one in drawn[:-1])


def test_the_bracketed_tag_is_the_newest_swing(axes):
    data, label = reading_of(UP)
    axes.set_xlim(0, data.last_bar)

    plot.annotate(axes, label)

    newest = label.swings[-1]
    assert f"({newest.kind})" in tags(axes)


def test_the_brackets_are_drawing_only_and_not_in_the_data():
    """A reader sees brackets; anything reading the label must not.

    The same split the C# keeps - SegmentState.Text returns the bare name and
    PriceSegments adds the brackets - so a rule, a statistic or a comparison
    never has to strip punctuation to find out what a turn was.
    """
    _, label = reading_of(UP)

    for swing in label.swings:
        assert "(" not in swing.kind
        assert swing.kind in {"HH", "HL", "LH", "LL", "EQH", "EQL", "first"}

    assert "(" not in label.says()


def test_a_reading_with_no_swings_draws_nothing_and_does_not_fall_over(axes):
    """`label.swings[-1]` on an empty tuple is the obvious way to get this wrong."""
    data = build(UP, last=AT)
    early = tagger.label(data, 6, view=replay.at(data, 6))

    axes.set_xlim(0, data.last_bar)

    assert early.swings == ()
    plot.annotate(axes, early)
    assert tags(axes) == []
