"""The trading state machine.

Two things are being protected here. The rules of the machine - what may be
pressed, what a trade is worth, when it may close - and the one that matters
more: that nothing in here can see a bar the session has not reached. A gym that
leaks the future scores a skill nobody has.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from segments import dataset as dataset_module
from segments import gym

META = {
    "instrument": "MNQ 12-26", "period": "3 Minute", "session": "CME US Index Futures ETH",
    "method": "BottomUp", "source": "Median", "ticksize": 0.25, "tolerance": 1.0,
    "minbars": 3, "window": 250, "atr": 14, "exported": "2026-09-27T08:00:00",
}

ATR = 10.0


def build(closes: list[float], highs=None, lows=None):
    """A dataset whose closes are exactly these, with a flat ATR of 10.

    A flat ATR makes R readable by eye: ten points is one R, so an assertion can
    say what it means instead of carrying arithmetic around.
    """
    bars = np.arange(len(closes))
    close = np.array(closes, dtype=float)

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-09-21 17:00", periods=len(bars), freq="3min"),
            "open": close,
            "high": close if highs is None else np.array(highs, dtype=float),
            "low": close if lows is None else np.array(lows, dtype=float),
            "close": close,
            "volume": 1000.0,
            "mid": close,
            "atr": ATR,
            "new_session": 0,
        },
        index=pd.Index(bars, name="bar"),
    )

    events = pd.DataFrame(
        columns=["at_bar", "event", "pivot_bar", "pivot_price"]
    ).astype({"at_bar": "int64", "event": "string", "pivot_bar": "int64", "pivot_price": "float64"})

    return dataset_module.Dataset(
        bars=frame, events=events, meta=dataset_module.Meta.parse("# " + json.dumps(META)), path="."
    )


RISING = build([100, 110, 120, 130, 140, 150])
FALLING = build([150, 140, 130, 120, 110, 100])


# --- what may be pressed ------------------------------------------------------

def test_it_starts_flat():
    session = gym.Session(RISING, start=0)

    assert session.state == gym.FLAT
    assert session.flat


def test_buy_opens_a_long_and_sell_opens_a_short():
    long_side = gym.Session(RISING, start=0)
    long_side.press(gym.BUY)

    short_side = gym.Session(RISING, start=0)
    short_side.press(gym.SELL)

    assert long_side.state == gym.LONG
    assert short_side.state == gym.SHORT


def test_a_long_cannot_be_added_to():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance()

    assert not session.allowed(gym.BUY)
    with pytest.raises(ValueError):
        session.press(gym.BUY)


def test_a_short_cannot_be_added_to():
    session = gym.Session(RISING, start=0)
    session.press(gym.SELL)
    session.advance()

    assert not session.allowed(gym.SELL)
    with pytest.raises(ValueError):
        session.press(gym.SELL)


def test_closing_returns_to_flat_and_never_reverses():
    """One press closes. It does not close and open the other way."""
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance()
    session.press(gym.SELL)

    assert session.state == gym.FLAT
    assert len(session.trades) == 1


def test_flat_after_a_long_may_still_press_sell_to_go_short():
    """The reading of the alternation rule - see the module note on why the
    literal one makes every trade a long."""
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance()
    session.press(gym.SELL)          # closes the long
    session.advance()

    assert session.allowed(gym.SELL)
    session.press(gym.SELL)          # opens a short
    assert session.state == gym.SHORT


# --- a trade lasts at least a bar ---------------------------------------------

def test_a_trade_cannot_open_and_close_on_the_same_bar():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)

    assert not session.allowed(gym.SELL)
    with pytest.raises(ValueError):
        session.press(gym.SELL)


def test_a_trade_can_close_on_the_very_next_bar():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance()

    trade = session.press(gym.SELL)

    assert trade.bars == 1


def test_a_position_cannot_be_opened_on_the_last_bar():
    """It could never reach a second bar, so it could never be a trade."""
    session = gym.Session(RISING, start=RISING.last_bar)

    assert not session.allowed(gym.BUY)
    assert not session.allowed(gym.SELL)


# --- what a trade is worth ----------------------------------------------------

def test_a_long_makes_money_when_price_rises():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)           # 100
    session.advance(3)
    trade = session.press(gym.SELL)  # 130

    assert trade.points == pytest.approx(30.0)
    assert trade.r == pytest.approx(3.0)
    assert trade.direction == gym.LONG
    assert trade.won


def test_a_short_makes_money_when_price_falls():
    session = gym.Session(FALLING, start=0)
    session.press(gym.SELL)          # 150
    session.advance(3)
    trade = session.press(gym.BUY)   # 120

    assert trade.points == pytest.approx(30.0)
    assert trade.r == pytest.approx(3.0)
    assert trade.direction == gym.SHORT


def test_a_short_loses_when_price_rises():
    session = gym.Session(RISING, start=0)
    session.press(gym.SELL)          # 100
    session.advance(2)
    trade = session.press(gym.BUY)   # 120

    assert trade.points == pytest.approx(-20.0)
    assert trade.r == pytest.approx(-2.0)
    assert not trade.won


def test_r_is_points_over_the_atr_at_entry():
    """Not the ATR now - the one the trade was sized against when it was taken."""
    data = build([100, 110, 120])
    data.bars.loc[0, "atr"] = 5.0
    data.bars.loc[2, "atr"] = 50.0

    session = gym.Session(data, start=0)
    session.press(gym.BUY)
    session.advance(2)
    trade = session.press(gym.SELL)

    assert trade.atr == pytest.approx(5.0)
    assert trade.r == pytest.approx(20.0 / 5.0)


def test_the_entry_and_exit_are_the_closes_of_their_bars():
    session = gym.Session(RISING, start=1)
    session.press(gym.BUY)
    session.advance(2)
    trade = session.press(gym.SELL)

    assert trade.entry_price == pytest.approx(110.0)
    assert trade.exit_price == pytest.approx(130.0)
    assert (trade.entry_bar, trade.exit_bar) == (1, 3)


# --- excursions ---------------------------------------------------------------

def test_mfe_and_mae_come_from_the_highs_and_lows():
    data = build([100, 100, 100], highs=[100, 130, 100], lows=[100, 80, 100])

    session = gym.Session(data, start=0)
    session.press(gym.BUY)
    session.advance(2)
    trade = session.press(gym.SELL)

    assert trade.mfe == pytest.approx(3.0)   # 130 high, 30 points, 3 R
    assert trade.mae == pytest.approx(-2.0)  # 80 low, -20 points, -2 R


def test_a_shorts_excursions_are_the_mirror():
    data = build([100, 100, 100], highs=[100, 130, 100], lows=[100, 80, 100])

    session = gym.Session(data, start=0)
    session.press(gym.SELL)
    session.advance(2)
    trade = session.press(gym.BUY)

    assert trade.mfe == pytest.approx(2.0)
    assert trade.mae == pytest.approx(-3.0)


# --- moving, and only forwards ------------------------------------------------

def test_a_session_only_goes_forward():
    session = gym.Session(RISING, start=2)

    with pytest.raises(ValueError):
        session.advance(-1)
    with pytest.raises(ValueError):
        session.advance(0)


def test_it_will_not_advance_past_the_last_bar():
    session = gym.Session(RISING, start=0)
    session.advance(999)

    assert session.bar == RISING.last_bar
    assert session.finished


def test_a_session_cannot_start_outside_the_bars():
    with pytest.raises(ValueError):
        gym.Session(RISING, start=RISING.last_bar + 1)


# --- the end ------------------------------------------------------------------

def test_an_open_trade_is_forced_closed_at_the_end():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance(2)

    trade = session.finish()

    assert trade.forced
    assert trade.exit_bar == RISING.last_bar
    assert trade.exit_price == pytest.approx(150.0)


def test_finishing_flat_produces_nothing():
    session = gym.Session(RISING, start=0)

    assert session.finish() is None
    assert session.trades == []


def test_a_forced_close_keeps_its_loss():
    """The flattering bug: dropping the open trade instead of booking it."""
    session = gym.Session(RISING, start=0)
    session.press(gym.SELL)
    session.advance(2)

    trade = session.finish()

    assert trade.r < 0
    assert len(session.trades) == 1


# --- live numbers -------------------------------------------------------------

def test_the_open_position_is_marked_to_the_current_bar():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)

    assert session.open_r() == pytest.approx(0.0)
    session.advance(2)
    assert session.open_r() == pytest.approx(2.0)
    assert session.held() == 2


def test_flat_has_nothing_open():
    session = gym.Session(RISING, start=0)

    assert session.open_r() == 0.0
    assert session.held() == 0


# --- the record ---------------------------------------------------------------

def test_a_trade_survives_a_round_trip_through_a_dict():
    session = gym.Session(RISING, start=0)
    session.press(gym.BUY)
    session.advance(2)
    trade = session.press(gym.SELL)

    assert gym.Trade.from_dict(json.loads(json.dumps(trade.as_dict()))) == trade
