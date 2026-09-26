"""The replay has to be exact, so it is tested against a hand-built export.

The fixture is small enough to reason about by eye: twelve bars, three pivot
events, one of which is a withdrawal. Every assertion below is a fact about that
listing rather than about a real market, which is the only way a causality test
means anything - on real data you cannot tell a correct replay from a plausible
one.
"""

from __future__ import annotations

import json

import pytest

from segments import dataset as dataset_module
from segments import replay

META = json.dumps(
    {
        "instrument": "MNQ 12-26",
        "period": "15 Minute",
        "session": "US Equities RTH",
        "method": "BottomUp",
        "source": "Median",
        "ticksize": 0.25,
        "tolerance": 1.0,
        "minbars": 3,
        "window": 250,
        "atr": 14,
        "exported": "2026-09-26T08:00:00",
    }
)

# mid = (high + low) / 2, which is the Median source the header names: 100, 101,
# ... so a pivot at bar b has price 100 + b.
BARS = "\n".join(
    f"{bar},2026-09-25T09:{30 + bar:02d}:00,"
    f"{100 + bar},{100.5 + bar},{99.5 + bar},{100 + bar},1000,{100 + bar},4.0,{1 if bar == 0 else 0}"
    for bar in range(12)
)

# Bar 3 is seen at 6 and sticks. Bar 5 is seen at 8 and taken back at 9 - a
# merge that undid it. Bar 7 is seen at 10.
EVENTS = "\n".join(["6,+,3,103", "8,+,5,105", "9,-,5,105", "10,+,7,107"])


@pytest.fixture
def export(tmp_path):
    stem = tmp_path / "MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w250"

    (stem.parent / (stem.name + "-bars.csv")).write_text(
        f"# {META}\nbar,time,open,high,low,close,volume,mid,atr,new_session\n{BARS}\n", encoding="utf-8"
    )
    (stem.parent / (stem.name + "-pivots.csv")).write_text(
        f"# {META}\nat_bar,event,pivot_bar,pivot_price\n{EVENTS}\n", encoding="utf-8"
    )

    return stem


def test_load_reads_the_header_and_both_files(export):
    data = dataset_module.load(export)

    assert data.meta.instrument == "MNQ 12-26"
    assert data.meta.period == "15 Minute"          # the space is why the header is JSON
    assert data.meta.session == "US Equities RTH"
    assert data.meta.source == "Median"
    assert data.meta.tolerance == 1.0
    assert len(data.bars) == 12
    assert len(data.events) == 4
    assert data.check() == []


def test_either_file_or_the_stem_names_the_same_dataset(export):
    stem = dataset_module.load(export)
    by_bars = dataset_module.load(str(export) + "-bars.csv")
    by_pivots = dataset_module.load(str(export) + "-pivots.csv")

    assert len(stem.bars) == len(by_bars.bars) == len(by_pivots.bars)


def test_nothing_is_known_before_it_was_learnt(export):
    data = dataset_module.load(export)

    # The pivot at bar 3 exists from bar 3 onwards in hindsight, and is not
    # knowable until bar 6. That gap is the whole discipline of the format.
    assert replay.at(data, 5).pivots() == []

    known = replay.at(data, 6).pivots()
    assert [pivot.bar for pivot in known] == [3]
    assert known[0].seen_at == 6
    assert known[0].age == 3


def test_a_withdrawn_pivot_goes_away_again(export):
    data = dataset_module.load(export)

    assert [pivot.bar for pivot in replay.at(data, 8).pivots()] == [3, 5]
    assert [pivot.bar for pivot in replay.at(data, 9).pivots()] == [3]
    assert [pivot.bar for pivot in replay.at(data, 11).pivots()] == [3, 7]


def test_the_leg_in_progress_is_flagged_provisional(export):
    data = dataset_module.load(export)

    pieces = replay.at(data, 11).segments()

    assert [(piece.start_bar, piece.end_bar) for piece in pieces] == [(3, 7), (7, 11)]
    assert [piece.provisional for piece in pieces] == [False, True]

    # The provisional piece ends at the bar being looked at, on that bar's own
    # source price - there is no pivot there, and there must not appear to be.
    assert pieces[-1].end_price == pytest.approx(111.0)
    assert pieces[0].change == pytest.approx(4.0)
    assert pieces[0].slope() == pytest.approx(1.0)


def test_a_replay_refuses_to_go_backwards(export):
    data = dataset_module.load(export)
    view = replay.at(data, 9)

    with pytest.raises(ValueError):
        view.advance_to(8)


def test_walking_agrees_with_asking_bar_by_bar(export):
    data = dataset_module.load(export)

    for view in replay.walk(data):
        expected = replay.at(data, view.bar).pivots()
        assert view.pivots() == expected


def test_confirmation_measures_the_lag(export):
    data = dataset_module.load(export)
    table = replay.confirmation(data).set_index("pivot_bar")

    assert table.loc[3, "held"]
    assert table.loc[3, "first_lag"] == 3        # happened at 3, visible at 6
    assert table.loc[3, "lag"] == 3             # and never changed again
    assert table.loc[7, "lag"] == 3

    # Bar 5 was taken back, so it is not a swing of this history at all.
    assert not table.loc[5, "held"]
    assert table.loc[5, "flips"] == 2

    summary = replay.lag_summary(data)
    assert summary["pivots"] == 2
    assert summary["withdrawn"] == 1
    assert summary["lag_median"] == 3


def test_legs_are_built_from_the_settled_pivots_only(export):
    data = dataset_module.load(export)
    pieces = replay.legs(data)

    # 3 to 7 only: bar 5 was withdrawn, and there is no pivot after 7.
    assert len(pieces) == 1
    assert pieces.loc[0, "start_bar"] == 3
    assert pieces.loc[0, "end_bar"] == 7
    assert pieces.loc[0, "ticks"] == pytest.approx(16.0)    # 4 points at 0.25
    assert pieces.loc[0, "amplitude"] == pytest.approx(1.0)  # ATR is 4.0 throughout
    assert bool(pieces.loc[0, "up"])


def test_check_notices_files_from_different_runs(export):
    """A pivot price that is not its own bar's source price is the one mistake
    that would otherwise go unnoticed for weeks."""
    pivots = export.parent / (export.name + "-pivots.csv")
    pivots.write_text(
        f"# {META}\nat_bar,event,pivot_bar,pivot_price\n6,+,3,999\n", encoding="utf-8"
    )

    problems = dataset_module.load(export).check()

    assert any("do not match" in problem for problem in problems)


def test_check_notices_a_pivot_learnt_before_it_happened(export):
    pivots = export.parent / (export.name + "-pivots.csv")
    pivots.write_text(
        f"# {META}\nat_bar,event,pivot_bar,pivot_price\n3,+,7,107\n", encoding="utf-8"
    )

    problems = dataset_module.load(export).check()

    assert any("at or after the bar" in problem for problem in problems)
