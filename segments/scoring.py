"""What a pile of trades is worth, and how much of that is noise.

The score is per-trade Sharpe - mean R over standard deviation of R - and it is
chosen because it cannot be inflated by trading more, trading longer, or sizing
up. Random entries and exits with no costs average zero R, so a Sharpe reliably
above zero is the only thing here that would count as evidence of skill.

"RELIABLY" IS DOING ALL THE WORK

    A Sharpe measured over n trades has a standard error of about
    sqrt((1 + S^2/2) / n). At thirty trades that is 0.19, and the whole tier
    ladder from break-even to pro spans 0.25. The noise is wider than the
    ladder.

    Simulated with fat-tailed returns, 200,000 traders of GENUINELY NO SKILL:

        trades    reach Apprentice    reach Trader    reach Pro
            30              39.5%           21.3%         5.5%
           100              31.3%            7.0%         0.1%
           400              16.0%            0.1%         0.0%
          1000               5.8%            0.0%         0.0%

    One clicker in five is told they are a Trader at thirty trades, and one in
    eighteen is told they are a Pro. A ladder that promotes noise a fifth of the
    time cannot answer the question the gym was built to ask.

    SO THE TIER IS AWARDED ON THE LOWER END OF THE CONFIDENCE INTERVAL, not on
    the measured Sharpe. The number shown is still the measured one - hiding it
    would be its own kind of lie - but the label is what survives the error
    bars. It needs no new threshold, it is self-correcting, and it makes a high
    tier cost trades rather than luck:

        reaching "Trader"        n=30     n=100    n=400
        no skill, on the number  21.6%     6.8%     0.1%
        no skill, on the bound    0.2%     0.0%     0.0%
        real 0.30, on the bound  15.2%    35.1%    83.6%

    A real edge still gets there. It has to earn it.

    :func:`calibrate` is the simulation, kept runnable so the table above can be
    argued with rather than believed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gym import Trade

#: Trades before a number is shown at all. Thirty is enough for a mean to mean
#: something; see the module note for why it is nowhere near enough for a tier.
RANK_AFTER = 30

#: The rolling view - "current form". Deliberately carries no tier: at fifty
#: trades the error bar is +/- 0.28, which is the whole ladder twice over.
ROLLING = 50

#: 95%, two-sided.
Z = 1.96

#: Each tier and the Sharpe it starts at, worst first. Configurable on purpose -
#: calibrate against :func:`calibrate` rather than against a good week.
TIERS = (
    ("Losing", -math.inf),
    ("Break-even", 0.0),
    ("Apprentice", 0.05),
    ("Trader", 0.15),
    ("Pro", 0.30),
)


@dataclass(frozen=True)
class Score:
    """The ranking, and how much of it to believe."""

    trades: int
    mean: float | None = None
    stdev: float | None = None
    sharpe: float | None = None

    #: On the mean R, which is what the brief asked to see.
    mean_low: float | None = None
    mean_high: float | None = None

    #: On the Sharpe, which is what the tier is read from.
    low: float | None = None
    high: float | None = None

    @property
    def ranked(self) -> bool:
        return self.sharpe is not None and self.trades >= RANK_AFTER

    @property
    def tier(self) -> str:
        """The label, read off the lower bound - see the module note."""
        if not self.ranked or self.low is None:
            return f"Unranked ({self.trades}/{RANK_AFTER})"

        return tier_for(self.low)

    @property
    def tier_if_believed(self) -> str:
        """What the measured number alone would have claimed.

        Worth showing beside the real tier while there are few trades, because
        the gap between the two IS the uncertainty, in the plainest form it can
        be put.
        """
        return tier_for(self.sharpe) if self.sharpe is not None else "Unranked"


def tier_for(sharpe: float | None) -> str:
    if sharpe is None:
        return "Unranked"

    name = TIERS[0][0]
    for label, floor in TIERS:
        if sharpe >= floor:
            name = label

    return name


def score(values: list[float]) -> Score:
    """The ranking for a list of R values.

    Two trades are needed before there is any spread to divide by, and a set of
    identical results has no spread at all - both come back with a Sharpe of
    None rather than an infinity, because "unmeasurable" is the honest answer
    and an infinity would sort to the top of a leaderboard.
    """
    n = len(values)
    if n == 0:
        return Score(trades=0)

    mean = sum(values) / n
    if n < 2:
        return Score(trades=n, mean=mean)

    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    error = spread / math.sqrt(n)

    if spread == 0:
        return Score(trades=n, mean=mean, stdev=0.0,
                     mean_low=mean, mean_high=mean)

    sharpe = mean / spread

    # The standard error of a Sharpe itself, which is not the one on the mean:
    # the ratio has the uncertainty of its denominator in it too.
    sharpe_error = math.sqrt((1 + sharpe ** 2 / 2) / n)

    return Score(
        trades=n,
        mean=mean,
        stdev=spread,
        sharpe=sharpe,
        mean_low=mean - Z * error,
        mean_high=mean + Z * error,
        low=sharpe - Z * sharpe_error,
        high=sharpe + Z * sharpe_error,
    )


@dataclass(frozen=True)
class Report:
    """The dashboard: everything that is arithmetic rather than judgement."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0

    win_rate: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    payoff: float | None = None
    expectancy: float | None = None
    profit_factor: float | None = None

    total_r: float = 0.0
    max_drawdown: float = 0.0
    longest_losing_streak: int = 0

    forced: int = 0
    average_bars: float | None = None
    capture: float | None = None

    equity: tuple[float, ...] = field(default_factory=tuple)
    score: Score = field(default_factory=lambda: Score(trades=0))


def report(trades: list[Trade]) -> Report:
    """Every statistic the gym shows, from a list of closed trades."""
    if not trades:
        return Report()

    values = [trade.r for trade in trades]

    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    scratches = [value for value in values if value == 0]

    won = sum(wins)
    lost = sum(losses)

    average_win = won / len(wins) if wins else None
    average_loss = lost / len(losses) if losses else None

    return Report(
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        scratches=len(scratches),
        # Scratches are in the denominator. They are trades that were taken, and
        # leaving them out would let a run of nothing raise a win rate.
        win_rate=len(wins) / len(trades),
        average_win=average_win,
        average_loss=average_loss,
        payoff=(average_win / abs(average_loss)) if average_win and average_loss else None,
        expectancy=sum(values) / len(values),
        # No losses at all is not an infinite profit factor, it is too few
        # trades. None, and the dashboard says so.
        profit_factor=(won / abs(lost)) if losses and lost != 0 else None,
        total_r=sum(values),
        max_drawdown=drawdown(values),
        longest_losing_streak=longest_losing(values),
        forced=sum(1 for trade in trades if trade.forced),
        average_bars=sum(trade.bars for trade in trades) / len(trades),
        capture=capture(trades),
        equity=tuple(equity(values)),
        score=score(values),
    )


def equity(values: list[float]) -> list[float]:
    """The cumulative R curve."""
    running = 0.0
    out = []
    for value in values:
        running += value
        out.append(running)

    return out


def drawdown(values: list[float]) -> float:
    """The deepest fall from a peak of the cumulative R curve, as a positive number."""
    peak = 0.0
    running = 0.0
    worst = 0.0

    for value in values:
        running += value
        peak = max(peak, running)
        worst = max(worst, peak - running)

    return worst


def longest_losing(values: list[float]) -> int:
    """The longest run of losers. A scratch breaks a streak without extending it."""
    longest = 0
    current = 0

    for value in values:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)

    return longest


def capture(trades: list[Trade]) -> float | None:
    """How much of the move that was available was kept, over the winners.

    Only the winners, and only those that ever went in favour. R/MFE on a loser
    is a negative fraction of a positive one, which reads as "captured -40%" and
    means nothing; and a trade whose MFE is zero never offered anything to
    capture, so it cannot be scored on having missed it.

    This is the number that says an exit is handing back the move rather than
    taking it - a good entry with a bad exit shows up here and nowhere else.
    """
    kept = [trade.r / trade.mfe for trade in trades if trade.r > 0 and trade.mfe > 0]

    return sum(kept) / len(kept) if kept else None


def rolling(trades: list[Trade], window: int = ROLLING) -> Report:
    """The last ``window`` trades: current form, no tier."""
    return report(trades[-window:])


def by(trades: list[Trade], attribute: str) -> dict[str, Report]:
    """Split by one field of the trade - 'direction', 'timeframe', 'instrument'."""
    groups: dict[str, list[Trade]] = {}
    for trade in trades:
        groups.setdefault(str(getattr(trade, attribute)), []).append(trade)

    return {name: report(group) for name, group in sorted(groups.items())}


def calibrate(trials: int = 20_000, counts: tuple[int, ...] = (30, 100, 400, 1000),
              skill: float = 0.0, seed: int = 0) -> "pd.DataFrame":
    """What a trader of a given true skill is awarded, by luck alone.

    The null the tiers are set against. ``skill`` is the true per-trade Sharpe;
    at its default of zero this is the random clicker, and every tier it reaches
    is a false positive.

    Returns one row per trade count: the spread of the measured Sharpe, and how
    often each tier is reached on the measured number against on the lower
    bound. The second column is the one the gym uses, and the gap between them
    is the argument for doing so.

    Fat tails on purpose - Student's t with four degrees of freedom, scaled to
    unit variance. Real R distributions are not normal, and assuming they are
    would make the noise look smaller than it is.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []

    for n in counts:
        draws = rng.standard_t(4, size=(trials, n)) / math.sqrt(2.0) + skill
        measured = draws.mean(axis=1) / draws.std(axis=1, ddof=1)
        bound = measured - Z * np.sqrt((1 + measured ** 2 / 2) / n)

        row = {"trades": n, "sharpe_sd": float(measured.std())}
        for label, floor in TIERS[1:]:
            row[f"{label} (measured)"] = float((measured >= floor).mean())
            row[f"{label} (bound)"] = float((bound >= floor).mean())

        rows.append(row)

    return pd.DataFrame(rows).set_index("trades")
