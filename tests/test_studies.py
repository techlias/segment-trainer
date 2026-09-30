"""The two indicators the project works out for itself.

Everything else here is checked against an export NinjaTrader wrote. These are
not, so the things that would quietly be wrong are pinned against reference
arithmetic written a different way: the population standard deviation, and
Wilder's seeding. Both are easy to get wrong by accepting a library default, and
neither would look wrong on a chart.
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd
import pytest

from segments import studies


def series(values) -> pd.Series:
    return pd.Series([float(v) for v in values], index=range(len(values)), dtype=float)


# --- Bollinger ---------------------------------------------------------------

def test_bands_use_the_population_deviation_not_the_sample_one():
    """The whole point. pandas defaults to n - 1; NinjaTrader divides by n."""
    price = series([10, 12, 11, 15, 14, 13, 18, 17])
    frame = studies.bollinger(price, deviations=2.0, period=4)

    for bar in range(3, len(price)):
        window = [price[i] for i in range(bar - 3, bar + 1)]
        middle = statistics.fmean(window)
        spread = statistics.pstdev(window)

        assert frame["middle"][bar] == pytest.approx(middle)
        assert frame["upper"][bar] == pytest.approx(middle + 2.0 * spread)
        assert frame["lower"][bar] == pytest.approx(middle - 2.0 * spread)

        # And emphatically NOT the sample one, or the band is 15% too wide here.
        assert frame["upper"][bar] != pytest.approx(middle + 2.0 * statistics.stdev(window))


def test_bands_say_nothing_until_they_have_the_bars():
    frame = studies.bollinger(series([10, 12, 11, 15, 14]), period=4)

    assert frame["middle"].isna().tolist() == [True, True, True, False, False]


def test_a_flat_stretch_has_no_width():
    frame = studies.bollinger(series([7, 7, 7, 7, 7]), period=4)

    assert frame["upper"][4] == pytest.approx(7.0)
    assert frame["lower"][4] == pytest.approx(7.0)


def test_the_deviations_come_first():
    """NinjaTrader's argument order, which reads backwards and is the point."""
    price = series(range(20))

    assert studies.bollinger(price, 2.0, 4)["middle"].isna().sum() == 3
    assert studies.bollinger(price, 2.0, 10)["middle"].isna().sum() == 9


@pytest.mark.parametrize("period", [0, -1])
def test_a_band_needs_a_period(period):
    with pytest.raises(ValueError):
        studies.bollinger(series([1, 2, 3]), period=period)


def test_a_band_of_any_period_the_slider_offers():
    """5 to 20, which is what the sidebar exposes."""
    rng = np.random.default_rng(13)
    price = series(100 + np.cumsum(rng.normal(0, 1.0, 120)))

    for period in range(5, 21):
        frame = studies.bollinger(price, period=period)

        assert frame["middle"].isna().sum() == period - 1
        assert (frame["upper"].dropna() >= frame["middle"].dropna()).all()
        assert (frame["lower"].dropna() <= frame["middle"].dropna()).all()


# --- RSI ---------------------------------------------------------------------

def reference_rsi(values: list[float], period: int) -> list[float]:
    """Wilder, written out the long way: a simple mean of the first `period`
    changes, then each change worth 1/period. Deliberately not the
    implementation - if both are wrong they have to be wrong the same way, and
    they were written from opposite ends."""
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    out = [float("nan")] * len(values)

    if len(changes) < period:
        return out

    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def strength(gain, loss):
        if loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[period] = strength(average_gain, average_loss)

    for i in range(period, len(changes)):
        average_gain = (average_gain * (period - 1) + gains[i]) / period
        average_loss = (average_loss * (period - 1) + losses[i]) / period
        out[i + 1] = strength(average_gain, average_loss)

    return out


def test_rsi_matches_wilder_written_the_long_way():
    rng = np.random.default_rng(11)
    walk = 100 + np.cumsum(rng.normal(0, 1.5, 200))

    got = studies.rsi(series(walk), 10)
    want = reference_rsi([float(v) for v in walk], 10)

    for bar, expected in enumerate(want):
        if np.isnan(expected):
            assert np.isnan(got[bar])
        else:
            assert got[bar] == pytest.approx(expected)


def test_rsi_is_not_an_ewm_seeded_on_the_first_bar():
    """The seeding is the subtle half of Wilder, so it gets its own test.

    An exponential average with the same alpha is the same recurrence from a
    different start, and it stays different for a long time - which is exactly
    the kind of wrong that looks fine.
    """
    rng = np.random.default_rng(3)
    walk = 100 + np.cumsum(rng.normal(0, 1.0, 120))
    price = series(walk)

    change = price.diff()
    naive_gain = change.clip(lower=0.0).ewm(alpha=1 / 10, adjust=False).mean()
    naive_loss = (-change.clip(upper=0.0)).ewm(alpha=1 / 10, adjust=False).mean()
    naive = 100.0 - 100.0 / (1.0 + naive_gain / naive_loss)

    ours = studies.rsi(price, 10)

    assert ours[119] != pytest.approx(naive[119], abs=1e-6)


def test_rsi_says_nothing_until_it_has_the_changes():
    values = studies.rsi(series(range(30)), 10)

    # One bar has no change before it, then ten changes to average.
    assert values.isna().sum() == 10
    assert not np.isnan(values[10])


def test_a_stretch_with_no_loser_is_a_hundred():
    assert studies.rsi(series(range(30)), 10)[29] == pytest.approx(100.0)


def test_a_stretch_with_no_winner_is_zero():
    assert studies.rsi(series(range(30, 0, -1)), 10)[29] == pytest.approx(0.0)


def test_rsi_stays_between_nothing_and_a_hundred():
    rng = np.random.default_rng(7)
    walk = 100 + np.cumsum(rng.normal(0, 2.0, 500))

    values = studies.rsi(series(walk), 10).dropna()

    assert values.min() >= 0.0
    assert values.max() <= 100.0


@pytest.mark.parametrize("period", [0, -1])
def test_an_rsi_needs_a_period(period):
    with pytest.raises(ValueError):
        studies.rsi(series([1, 2, 3]), period=period)


def test_the_viewer_can_ask_for_any_period_its_slider_offers():
    """4 to 20, which is what the sidebar exposes."""
    rng = np.random.default_rng(5)
    price = series(100 + np.cumsum(rng.normal(0, 1.0, 120)))

    for period in range(4, 21):
        values = studies.rsi(price, period).dropna()

        assert len(values) == 120 - period
        assert values.between(0.0, 100.0).all()
