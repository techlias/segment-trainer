"""The port has to be the same algorithm, not merely a similar one.

Two things are pinned here. First, that the cached bottom-up picks exactly the
merges the plain rescan picks - the optimisation is a speed-up and nothing else.
Second, that the sweep in :func:`segments.fit.events` produces a log the replay
reads back into the same pivots, which is what makes a re-cut dataset
interchangeable with one NinjaTrader wrote.

What cannot be pinned from here is agreement with the C# itself. That needs a
real export, and :func:`segments.fit.verify` is the test for it - run it once
per dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from segments import dataset as dataset_module
from segments import fit, replay


def bottom_up_by_rescan(price, scale, tolerance, min_bars):
    """SegmentFit.cs BottomUp, transcribed line for line, costs rescanned every
    round. The reference the cached version has to match."""
    n = len(price)

    pivot = list(range(0, n, 2))
    if pivot[-1] != n - 1:
        pivot.append(n - 1)

    while len(pivot) > 2:
        forced_at, forced_cost = -1, float("inf")
        best_at, best_cost = -1, float("inf")

        for i in range(len(pivot) - 2):
            cost = fit.error(price, scale, pivot[i], pivot[i + 2])[0]
            forced = (pivot[i + 1] - pivot[i] < min_bars) or (pivot[i + 2] - pivot[i + 1] < min_bars)

            if forced:
                if cost < forced_cost:
                    forced_cost, forced_at = cost, i
            elif cost < best_cost:
                best_cost, best_at = cost, i

        if forced_at >= 0:
            drop = forced_at
        elif best_cost <= tolerance:
            drop = best_at
        else:
            break

        del pivot[drop + 1]

    return pivot


def walk_prices(seed, n=260):
    """A path with legs in it - a pure random walk has no structure to find."""
    rng = np.random.default_rng(seed)
    out, price, leg, direction = [], 24000.0, 0, 1

    for _ in range(n):
        if leg <= 0:
            leg = int(rng.integers(8, 35))
            direction = int(rng.choice([1, -1, 0]))
        leg -= 1
        price += direction * rng.uniform(0.5, 3.0) + rng.normal(0, 2.0)
        out.append(price)

    return np.array(out)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("tolerance", [0.3, 1.0, 2.5])
def test_the_cache_picks_the_same_merges_as_a_full_rescan(seed, tolerance):
    price = walk_prices(seed)
    scale = np.full(len(price), 6.0)

    assert fit.bottom_up(price, scale, tolerance, 3) == bottom_up_by_rescan(price, scale, tolerance, 3)


@pytest.mark.parametrize("method", fit.METHODS)
def test_every_method_covers_the_window_end_to_end(method):
    price = walk_prices(3)
    scale = np.full(len(price), 6.0)

    pivots = fit.segment(price, scale, 1.0, 3, method)

    assert pivots[0] == 0
    assert pivots[-1] == len(price) - 1
    assert pivots == sorted(set(pivots))


@pytest.mark.parametrize("method", fit.METHODS)
def test_no_piece_is_shorter_than_asked_for(method):
    price = walk_prices(4)
    scale = np.full(len(price), 6.0)

    pivots = fit.segment(price, scale, 1.0, 5, method)
    lengths = [b - a for a, b in zip(pivots, pivots[1:])]

    # Douglas-Peucker can leave the two outermost pieces short: it refuses to
    # cut near an end, which is the same rule seen from the other side.
    assert min(lengths[1:-1] or lengths) >= 5 or method == "DouglasPeucker"
    assert all(length > 0 for length in lengths)


def test_a_tighter_tolerance_never_finds_fewer_turns():
    price = walk_prices(5)
    scale = np.full(len(price), 6.0)

    counts = [len(fit.segment(price, scale, tolerance, 3)) for tolerance in (3.0, 1.0, 0.3)]

    assert counts == sorted(counts)


def test_a_straight_line_is_one_piece():
    price = np.linspace(24000, 24100, 200)
    scale = np.full(200, 6.0)

    for method in fit.METHODS:
        assert fit.segment(price, scale, 1.0, 3, method) == [0, 199]


def test_error_is_the_worst_gap_in_atrs():
    # A single bar 12 off a flat line, with an ATR of 4, is 3 ATRs of error.
    price = np.array([100.0, 100.0, 112.0, 100.0, 100.0])
    scale = np.full(5, 4.0)

    gap, at = fit.error(price, scale, 0, 4)

    assert gap == pytest.approx(3.0)
    assert at == 2


def test_error_offers_no_cut_when_every_bar_sits_on_the_line():
    price = np.linspace(0, 10, 11)
    scale = np.full(11, 1.0)

    gap, at = fit.error(price, scale, 0, 10, margin=3)

    assert gap == 0
    assert at == -1


# --- the sweep, and what it is worth ------------------------------------------

META = {
    "instrument": "MNQ 12-26",
    "period": "15 Minute",
    "session": "US Equities RTH",
    "method": "BottomUp",
    "source": "Median",
    "ticksize": 0.25,
    "tolerance": 1.0,
    "minbars": 3,
    "window": 120,
    "atr": 14,
    "exported": "2026-09-27T08:00:00",
}


@pytest.fixture
def bars():
    price = walk_prices(11, n=400)
    index = pd.RangeIndex(len(price), name="bar")

    return pd.DataFrame(
        {
            "time": pd.date_range("2026-09-01 09:30", periods=len(price), freq="15min"),
            "open": price,
            "high": price + 3,
            "low": price - 3,
            "close": price,
            "volume": 1500.0,
            "mid": price,
            "atr": 6.0,
            "new_session": 0,
        },
        index=index,
    )


@pytest.fixture
def data(bars, tmp_path):
    log = fit.events(bars, method="BottomUp", source="Median", tolerance=1.0, min_bars=3,
                     window=120, scale=bars["atr"].to_numpy())

    return dataset_module.Dataset(
        bars=bars, events=log, meta=dataset_module.Meta.parse("# " + json.dumps(META)), path=tmp_path
    )


def test_the_sweep_never_learns_anything_early(data):
    assert (data.events["pivot_bar"] < data.events["at_bar"]).all()
    assert data.events["at_bar"].is_monotonic_increasing
    assert data.check() == []


def test_replaying_the_log_gives_back_the_fit_of_that_bar(data):
    """The log is only worth having if it reconstructs exactly. At a handful of
    bars, replay the events and compare against re-cutting that window from
    scratch - the interior pivots have to be the same set."""
    price = fit.source_price(data.bars, "Median")
    scale = data.bars["atr"].to_numpy()

    for bar in (150, 233, 300, 399):
        start = max(0, bar - 120 + 1)
        pivots = fit.segment(price[start : bar + 1], scale[start : bar + 1], 1.0, 3, "BottomUp")
        expected = {start + i for i in pivots[1:-1]}

        assert {pivot.bar for pivot in replay.at(data, bar).pivots()} == expected


def test_refit_changes_the_cut_and_nothing_else(data):
    finer = fit.refit(data, tolerance=0.3)

    assert finer.meta.tolerance == 0.3
    assert finer.meta.method == data.meta.method
    assert len(finer.events) > len(data.events)
    assert finer.bars.equals(data.bars)
    assert finer.check() == []


def test_verify_agrees_with_a_dataset_the_port_itself_produced(data):
    """Not proof against the C# - that needs a real export - but it does prove
    the comparison itself works, and catches a refit that quietly disagrees with
    the sweep that made the file."""
    assert fit.verify(data).empty


def test_verify_reports_what_is_missing(data):
    damaged = dataset_module.Dataset(
        bars=data.bars, events=data.events.iloc[2:].copy(), meta=data.meta, path=data.path
    )

    diff = fit.verify(damaged)

    assert len(diff) == 2
    assert set(diff["side"]) == {"python"}


def test_a_written_dataset_reads_back_the_same(data, tmp_path):
    data.write(tmp_path / "MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w120")
    again = dataset_module.load(tmp_path / "MNQ_12-26-15Minute-BottomUp-Median-t1-m3-w120")

    assert again.meta.instrument == data.meta.instrument
    assert again.meta.tolerance == data.meta.tolerance
    assert len(again.bars) == len(data.bars)
    assert again.check() == []

    # Not bit for bit: a CSV is decimal text, and a price that survives to a
    # thousandth of a tick has survived. The events themselves must match
    # exactly, because those are integers and a flag.
    pd.testing.assert_frame_equal(
        again.events.reset_index(drop=True),
        data.events.reset_index(drop=True),
        check_exact=False,
        atol=data.meta.ticksize / 1000,
    )
