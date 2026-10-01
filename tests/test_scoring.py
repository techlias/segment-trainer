"""The score, and the claim it rests on.

The formulas are checked against arithmetic done by hand. The ranking is checked
against a simulation, because the interesting question about a ranking is not
whether the division is right - it is how often it is wrong about somebody.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from segments import scoring
from segments.gym import Trade

WHEN = dt.datetime(2026, 9, 21, 17, 0)


def trade(r: float, direction: str = "long", mfe: float | None = None,
          bars: int = 5, forced: bool = False, timeframe: str = "3 Minute") -> Trade:
    """A trade that only has to carry an R and whatever else is being asked about."""
    return Trade(
        session="test", instrument="MNQ 12-26", timeframe=timeframe, direction=direction,
        entry_bar=0, entry_time=WHEN, entry_price=100.0,
        exit_bar=bars, exit_time=WHEN, exit_price=100.0 + r,
        bars=bars, points=r, atr=1.0, r=r,
        mfe=r if mfe is None else mfe, mae=min(r, 0.0), forced=forced,
    )


# --- the score ----------------------------------------------------------------

def test_the_sharpe_is_the_mean_over_the_spread():
    values = [1.0, -1.0, 2.0, -2.0, 3.0]
    got = scoring.score(values)

    mean = sum(values) / 5
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / 4)

    assert got.mean == pytest.approx(mean)
    assert got.stdev == pytest.approx(spread)
    assert got.sharpe == pytest.approx(mean / spread)


def test_the_confidence_band_on_the_mean_is_the_one_the_brief_asked_for():
    values = [1.0, -1.0, 2.0, -2.0, 3.0, 0.5]
    got = scoring.score(values)

    half = 1.96 * got.stdev / math.sqrt(len(values))

    assert got.mean_low == pytest.approx(got.mean - half)
    assert got.mean_high == pytest.approx(got.mean + half)


def test_the_band_on_the_sharpe_is_wider_than_the_one_on_the_mean():
    """They are not the same interval, and using the mean's for the tier would
    understate the noise - the ratio carries its denominator's error too."""
    values = [1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 1.5]
    got = scoring.score(values)

    on_mean = (got.mean_high - got.mean_low) / got.stdev
    on_sharpe = got.high - got.low

    assert on_sharpe > on_mean


def test_too_few_trades_to_divide_by_anything():
    assert scoring.score([]).trades == 0
    assert scoring.score([]).sharpe is None
    assert scoring.score([1.0]).sharpe is None
    assert scoring.score([1.0]).mean == pytest.approx(1.0)


def test_identical_results_have_no_measurable_sharpe():
    """Not an infinity. An infinity would sort to the top of a leaderboard."""
    got = scoring.score([1.0, 1.0, 1.0, 1.0])

    assert got.stdev == 0.0
    assert got.sharpe is None
    assert got.tier.startswith("Unranked")


# --- the tiers ----------------------------------------------------------------

@pytest.mark.parametrize("sharpe,expected", [
    (-0.5, "Losing"), (-0.001, "Losing"),
    (0.0, "Break-even"), (0.04, "Break-even"),
    (0.05, "Apprentice"), (0.149, "Apprentice"),
    (0.15, "Trader"), (0.29, "Trader"),
    (0.30, "Pro"), (5.0, "Pro"),
])
def test_the_ladder(sharpe, expected):
    assert scoring.tier_for(sharpe) == expected


def test_nothing_is_ranked_before_thirty_trades():
    got = scoring.score([0.5, -0.2] * 10)   # 20 trades, a real edge

    assert not got.ranked
    assert got.tier == "Unranked (20/30)"


def test_the_tier_comes_from_the_lower_bound_not_the_number():
    """The whole argument of the module. A measured Pro over few trades is not
    awarded Pro, because the bound does not support it."""
    values = ([1.0, -0.4] * 15)[:30]
    got = scoring.score(values)

    assert got.ranked
    assert got.sharpe > got.low
    assert got.tier == scoring.tier_for(got.low)
    assert got.tier != got.tier_if_believed


def test_enough_consistent_trades_do_earn_the_tier():
    """And the bound is not simply unreachable."""
    few = scoring.score([0.6, 0.2] * 15)      # 30
    many = scoring.score([0.6, 0.2] * 150)    # 300

    assert many.low > few.low
    assert scoring.tier_for(many.low) == "Pro"


# --- the dashboard ------------------------------------------------------------

def test_the_counts_split_wins_losses_and_scratches():
    got = scoring.report([trade(1.0), trade(-1.0), trade(0.0), trade(2.0)])

    assert (got.wins, got.losses, got.scratches) == (2, 1, 1)
    assert got.trades == 4


def test_a_scratch_is_in_the_denominator_of_the_win_rate():
    """Two wins out of four taken, not two out of three that moved."""
    got = scoring.report([trade(1.0), trade(2.0), trade(0.0), trade(-1.0)])

    assert got.win_rate == pytest.approx(0.5)


def test_payoff_and_expectancy():
    got = scoring.report([trade(2.0), trade(4.0), trade(-1.0), trade(-3.0)])

    assert got.average_win == pytest.approx(3.0)
    assert got.average_loss == pytest.approx(-2.0)
    assert got.payoff == pytest.approx(1.5)
    assert got.expectancy == pytest.approx(0.5)


def test_profit_factor():
    got = scoring.report([trade(3.0), trade(1.0), trade(-2.0)])

    assert got.profit_factor == pytest.approx(2.0)


def test_no_losses_is_not_an_infinite_profit_factor():
    got = scoring.report([trade(1.0), trade(2.0)])

    assert got.profit_factor is None


def test_the_equity_curve_is_cumulative_r():
    got = scoring.report([trade(1.0), trade(-0.5), trade(2.0)])

    assert list(got.equity) == pytest.approx([1.0, 0.5, 2.5])
    assert got.total_r == pytest.approx(2.5)


def test_the_drawdown_is_the_deepest_fall_from_a_peak():
    #            +3        -4         +1         -2
    # cumulative  3        -1          0         -2   peak 3, worst fall 5
    got = scoring.report([trade(3.0), trade(-4.0), trade(1.0), trade(-2.0)])

    assert got.max_drawdown == pytest.approx(5.0)


def test_a_curve_that_only_rises_has_no_drawdown():
    assert scoring.report([trade(1.0), trade(2.0)]).max_drawdown == 0.0


def test_the_longest_losing_streak():
    values = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    got = scoring.report([trade(v) for v in values])

    assert got.longest_losing_streak == 3


def test_a_scratch_breaks_a_streak_without_extending_it():
    got = scoring.report([trade(-1.0), trade(0.0), trade(-1.0)])

    assert got.longest_losing_streak == 1


def test_forced_closes_are_counted():
    got = scoring.report([trade(1.0), trade(-1.0, forced=True)])

    assert got.forced == 1


# --- capture ------------------------------------------------------------------

def test_capture_is_how_much_of_the_winners_move_was_kept():
    got = scoring.report([trade(1.0, mfe=2.0), trade(3.0, mfe=4.0)])

    assert got.capture == pytest.approx((0.5 + 0.75) / 2)


def test_losers_are_left_out_of_capture():
    """R/MFE on a loser is a negative fraction and reads as nonsense."""
    got = scoring.report([trade(1.0, mfe=2.0), trade(-5.0, mfe=1.0)])

    assert got.capture == pytest.approx(0.5)


def test_a_trade_that_never_went_in_favour_cannot_be_scored_on_capture():
    got = scoring.report([trade(-1.0, mfe=0.0)])

    assert got.capture is None


# --- slices -------------------------------------------------------------------

def test_the_rolling_view_is_the_last_n_trades():
    trades = [trade(1.0)] * 60 + [trade(-1.0)] * 10
    got = scoring.rolling(trades, window=10)

    assert got.trades == 10
    assert got.losses == 10


def test_splitting_by_direction():
    trades = [trade(1.0, "long"), trade(-1.0, "short"), trade(2.0, "long")]
    split = scoring.by(trades, "direction")

    assert set(split) == {"long", "short"}
    assert split["long"].trades == 2
    assert split["short"].total_r == pytest.approx(-1.0)


def test_splitting_by_timeframe():
    trades = [trade(1.0, timeframe="3 Minute"), trade(1.0, timeframe="15 Minute")]

    assert set(scoring.by(trades, "timeframe")) == {"3 Minute", "15 Minute"}


def test_an_empty_report_says_nothing_rather_than_dividing_by_zero():
    got = scoring.report([])

    assert got.trades == 0
    assert got.win_rate is None
    assert got.equity == ()


# --- the calibration ----------------------------------------------------------

def test_a_random_clicker_centres_on_zero():
    """The benchmark the whole score rests on: no skill, no costs, no edge."""
    table = scoring.calibrate(trials=4000, counts=(200,), seed=7)

    assert abs(table.loc[200, "Break-even (measured)"] - 0.5) < 0.03


def test_the_measured_tier_promotes_noise_at_thirty_trades():
    """The finding that moved the tier onto the lower bound. If this ever stops
    being true the module note needs rewriting, not deleting."""
    table = scoring.calibrate(trials=4000, counts=(30,), seed=11)

    assert table.loc[30, "Trader (measured)"] > 0.15


def test_the_lower_bound_does_not():
    table = scoring.calibrate(trials=4000, counts=(30,), seed=11)

    assert table.loc[30, "Trader (bound)"] < 0.02
    assert table.loc[30, "Pro (bound)"] < 0.01


def test_more_trades_shrink_the_spread_of_the_measured_sharpe():
    table = scoring.calibrate(trials=3000, counts=(30, 400), seed=3)

    assert table.loc[30, "sharpe_sd"] > table.loc[400, "sharpe_sd"] * 2


def test_a_real_edge_is_eventually_recognised():
    """The other half: the bound must be reachable, or the ladder is a wall."""
    table = scoring.calibrate(trials=3000, counts=(400,), skill=0.30, seed=5)

    assert table.loc[400, "Trader (bound)"] > 0.5
