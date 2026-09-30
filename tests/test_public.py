"""Public bars, minus the network.

:func:`segments.public.fetch` is not tested here and deliberately so - a test
that needs Yahoo to be up tests Yahoo. Everything downstream of it is tested,
because that is the part this project is responsible for: the shaping, the
session inference, the cut, and above all the label that says these bars are not
an export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segments import public, replay, tagger
from segments.dataset import BAR_COLUMNS


def candles(count: int = 400, start: str = "2026-01-05 09:30", minutes: int = 5,
            seed: int = 2) -> pd.DataFrame:
    """A plausible OHLCV frame shaped the way Yahoo hands one over."""
    rng = np.random.default_rng(seed)

    close = 100 + np.cumsum(rng.normal(0, 0.5, count))
    spread = np.abs(rng.normal(0, 0.3, count)) + 0.05

    frame = pd.DataFrame(
        {
            "open": close - rng.normal(0, 0.2, count),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 5000, count).astype(float),
        },
        index=pd.date_range(start, periods=count, freq=f"{minutes}min"),
    )

    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)

    return frame


# --- the shaping -------------------------------------------------------------

def test_bars_come_out_in_the_shape_the_package_reads():
    frame = public.bars(candles())

    assert list(frame.reset_index().columns) == BAR_COLUMNS
    assert frame.index.name == "bar"
    assert frame.index.tolist() == list(range(400))


def test_the_mid_is_the_middle_of_the_bar():
    frame = public.bars(candles(50))

    assert np.allclose(frame["mid"], (frame["high"] + frame["low"]) / 2)


def test_the_atr_is_seeded_on_the_first_bar():
    """Safe to compute here, unlike over a slice - see segments.fit.atr."""
    raw = candles(50)
    frame = public.bars(raw, atr_period=14)

    assert frame["atr"].iloc[0] == pytest.approx(raw["high"].iloc[0] - raw["low"].iloc[0])
    assert (frame["atr"] > 0).all()


# --- the session inference ---------------------------------------------------

def test_the_first_bar_opens_a_session():
    frame = public.bars(candles(30))

    assert frame["new_session"].iloc[0] == 1


def test_a_gap_in_the_timestamps_opens_a_session():
    """The whole of the inference. A break in the clock is the only thing
    standing in for NinjaTrader's session template."""
    one = candles(60, start="2026-01-05 09:30")
    two = candles(60, start="2026-01-06 09:30", seed=9)

    frame = public.bars(pd.concat([one, two]))

    assert frame["new_session"].sum() == 2
    assert frame["new_session"].iloc[60] == 1


def test_an_unbroken_stretch_is_one_session():
    frame = public.bars(candles(200))

    assert frame["new_session"].sum() == 1


def test_one_long_weekend_does_not_redefine_the_usual_spacing():
    """The median, not the mean or the max: a single three day gap must not
    make every ordinary bar look like a continuation of nothing."""
    weeks = [candles(80, start=f"2026-01-{day:02d} 09:30", seed=day) for day in (5, 6, 7, 12)]

    frame = public.bars(pd.concat(weeks))

    assert frame["new_session"].sum() == 4


def test_a_single_bar_is_still_a_session():
    frame = public.bars(candles(1))

    assert frame["new_session"].tolist() == [1]


# --- the dataset --------------------------------------------------------------

def test_a_built_dataset_passes_the_same_checks_an_export_does():
    """check() is what refuses a pair of files that cannot mean what they claim.
    A dataset built here has to satisfy it for the same reasons."""
    data = public.build(candles(600), symbol="NQ=F", interval="5m")

    assert data.check() == []


def test_it_is_labelled_as_public():
    data = public.build(candles(300), symbol="NQ=F", interval="5m")

    assert public.is_public(data) is True
    assert data.meta.raw["provider"] == "Yahoo Finance"
    assert data.meta.raw["symbol"] == "NQ=F"


def test_an_export_is_not_labelled_public():
    """The other half of the label: an export's meta must not carry the key."""
    from segments.dataset import Meta

    exported = Meta.parse('#{"instrument":"MNQ 12-26","method":"BottomUp","source":"Median"}')

    assert exported.raw.get("public", False) is False


def test_the_known_symbols_carry_their_tick_size():
    data = public.build(candles(300), symbol="NQ=F", interval="5m")

    assert data.meta.ticksize == 0.25
    assert "Nasdaq" in data.meta.instrument


def test_an_unknown_symbol_is_taken_at_face_value_with_no_tick_size():
    """Better a zero the viewer already handles than a guessed tick."""
    data = public.build(candles(300), symbol="WHO=F", interval="5m")

    assert data.meta.ticksize == 0.0
    assert data.meta.instrument == "WHO=F"


def test_the_cut_settings_are_carried_into_the_meta():
    data = public.build(candles(400), symbol="ES=F", interval="15m",
                        tolerance=0.5, min_bars=5, window=150)

    assert data.meta.tolerance == 0.5
    assert data.meta.minbars == 5
    assert data.meta.window == 150
    assert data.meta.period == "15m"


def test_a_tighter_tolerance_finds_more_turns():
    frame = candles(600)

    coarse = public.build(frame, symbol="NQ=F", tolerance=2.0)
    fine = public.build(frame, symbol="NQ=F", tolerance=0.4)

    assert len(replay.at(fine, fine.last_bar).pivots()) > \
           len(replay.at(coarse, coarse.last_bar).pivots())


def test_the_replay_and_the_tagger_read_it_like_any_other_dataset():
    data = public.build(candles(600), symbol="NQ=F", interval="5m")

    view = replay.at(data, data.last_bar)
    reading = tagger.label(data, data.last_bar, view=view)

    assert reading.state in tagger.STATES
    assert isinstance(reading.says(), str)


# --- the guards ---------------------------------------------------------------

def test_an_unknown_interval_is_refused():
    with pytest.raises(ValueError, match="unknown interval"):
        public.fetch("NQ=F", "7m", 5)


def test_every_offered_interval_has_a_stated_cap():
    """The caps are Yahoo's and the slider is built from them, so a missing one
    would become an unbounded slider and a silently short download."""
    for interval, cap in public.INTERVALS.items():
        assert cap is None or cap > 0
