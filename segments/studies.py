"""Two ordinary indicators, for context behind the shape.

Everything else in this package reads what NinjaTrader exported. These two are
the first numbers the project works out for ITSELF, which is worth saying out
loud: the ATR in the bars file was computed by SegmentFit and checked against
the C#; a Bollinger band drawn here has nothing holding it to the one on the
chart except the care taken below.

So the two places NinjaTrader's arithmetic is easy to get wrong are pinned here
rather than left to a default:

    POPULATION standard deviation, not sample. NinjaTrader's StdDev divides by
    n; pandas' ``.std()`` divides by n - 1 unless told otherwise. On a 4 bar
    Bollinger that is not a rounding difference - it is a band 15% wider.

    WILDER's smoothing for the RSI, seeded with a simple average of the first
    ``period`` changes. That is what Wilder wrote and what NinjaTrader's RSI
    does; an exponential average with the same alpha but seeded on the first
    bar is a different series for a long while after.

    NinjaTrader's RSI takes a second ``smooth`` parameter and puts the result
    through another average. This is the unsmoothed RSI - the textbook one - so
    the two will not lie on top of each other if anyone checks.

Both read the CLOSE, whatever the segments were fitted through. A Bollinger
band on the middle of the bar would be a thing nobody else draws, and the point
of having them here is that they are the ordinary ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: What the user asked the viewer for: two standard deviations over four bars.
#: Written the way NinjaTrader writes it - Bollinger(numStdDev, period), with
#: the deviations first - so the two can be compared without translating.
BAND_DEVIATIONS = 2.0
BAND_PERIOD = 4

#: Short, for a study of how a leg is running rather than of the whole day.
RSI_PERIOD = 10


def bollinger(price: pd.Series, deviations: float = BAND_DEVIATIONS,
              period: int = BAND_PERIOD) -> pd.DataFrame:
    """Middle, upper and lower, as three columns on the price's own index.

    Arguments in NinjaTrader's order: deviations first. It reads backwards to
    anyone who has only used the textbook definition, and matching the platform
    is worth more than matching the textbook here.

    The first ``period - 1`` bars are NaN rather than a partial average, so a
    band never claims to have seen more bars than it has.
    """
    if period < 1:
        raise ValueError(f"a Bollinger period is at least 1 bar, not {period}")

    middle = price.rolling(period).mean()

    # ddof=0: NinjaTrader's StdDev is the population one. See the module note.
    spread = price.rolling(period).std(ddof=0)

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + deviations * spread,
            "lower": middle - deviations * spread,
        }
    )


def rsi(price: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's relative strength index, 0 to 100, on the price's own index.

    NaN until there are ``period`` changes to average, for the same reason the
    bands are.

    A stretch with no losing bar in it has no ratio to take - the denominator is
    zero - and the answer there is 100 rather than an infinity or a gap. The
    mirror case, no winning bar, falls out of the arithmetic as 0 on its own.
    """
    if period < 1:
        raise ValueError(f"an RSI period is at least 1 bar, not {period}")

    change = price.diff()

    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)

    # The first row's change is NaN - there is no bar before it. It is not a
    # zero-gain bar, it is not a bar at all, so it is dropped rather than filled
    # and the averages start from the first real change.
    average_gain = _wilder(gain.iloc[1:], period)
    average_loss = _wilder(loss.iloc[1:], period)

    strength = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    strength[(average_loss == 0.0) & average_gain.notna()] = 100.0

    return strength.reindex(price.index)


def _wilder(values: pd.Series, period: int) -> pd.Series:
    """Wilder's running average: a simple mean of the first ``period`` values,
    and from there each new value worth 1/period of the answer.

    Written out as a loop rather than as ``ewm``. ``ewm`` with alpha 1/period is
    the same recurrence but seeds on the FIRST value instead of on the mean of
    the first ``period``, and the two disagree for long enough to matter on a
    short history.
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    raw = values.to_numpy(dtype=float)

    if len(raw) < period:
        return out

    running = float(raw[:period].mean())
    out.iloc[period - 1] = running

    for i in range(period, len(raw)):
        running = (running * (period - 1) + raw[i]) / period
        out.iloc[i] = running

    return out
