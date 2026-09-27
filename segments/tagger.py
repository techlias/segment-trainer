"""The first thing here with an opinion: what state the market is in, and why.

Everything before this module measures or draws. This one classifies, by rules
written out in full - Dow structure, nothing learnt, nothing fitted - so that
every label can be argued with. That is the point of it. A label you cannot
argue with teaches nothing when it disagrees with you, and the disagreements are
what the trainer is for.

WHAT IT READS

    Only :func:`segments.replay.at`: the pivots confirmed at that bar, and the
    closes up to it. It is a pure function of bar N - give it the same view and
    it gives the same answer, with no state carried from the bars before. That
    is what makes it safe to drop into a live strategy later, and what makes it
    testable now.

TURNS, NOT PIVOTS

    A pivot is where one straight piece meets the next. That is not always a
    turn: two rising pieces of different steepness meet at a bend, and a bend is
    not a swing high. Only the pivots where the direction actually reverses
    become swings here, and the rest are dropped.

    The newest swing is decided with the help of where price is NOW, since the
    leg leaving it has not finished. That is not lookahead - it is today's price
    - but it does mean the newest swing can change its mind as the bar moves.

THE STATES

    uptrend      the last swing high is higher than the one before it, and the
                 last swing low higher than the one before it.
    downtrend    the mirror: a lower high and a lower low.
    weakening    a trend whose pushes are getting smaller, or whose pullbacks
                 are getting bigger. Still a trend - this is a qualifier, and
                 the reason says which of the two it is.
    break        a CLOSE beyond the protective swing: below the last higher low
                 in an uptrend, above the last lower high in a downtrend. The
                 one state that does not wait for a pivot to confirm, which is
                 why the bars file carries every close and not just the pivots.
    range        swings that do not agree, with price between the last major
                 high and low.
    unclear      not enough confirmed swings yet to say anything. It is a real
                 answer and it is given honestly, rather than guessed at.

EVERY THRESHOLD IS A PARAMETER

    "Higher" needs a size, or a tick of noise makes a higher high; "smaller"
    needs a ratio. They live in :class:`Rules`, in ATRs and in ratios, never in
    ticks - see PriceSegments on why. The defaults are a starting point and
    nothing more. They should be argued with against a long export, not tuned
    against a few days.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _replace
from typing import Iterator

import pandas as pd

from .dataset import Dataset
from .replay import Replay, at, walk

UPTREND = "uptrend"
DOWNTREND = "downtrend"
WEAKENING = "weakening"
BREAK = "break"
RANGE = "range"
UNCLEAR = "unclear"

STATES = (UPTREND, DOWNTREND, WEAKENING, BREAK, RANGE, UNCLEAR)


@dataclass(frozen=True)
class Rules:
    """Where the lines are drawn. In ATRs and ratios, so they travel."""

    #: Two swings within this of each other are the same level - a double top
    #: rather than a higher high. Without it, one tick of noise makes a trend.
    equal_atr: float = 0.15

    #: How far past the protective swing a close has to settle before it is a
    #: break rather than a poke.
    break_atr: float = 0.10

    #: A push under this fraction of the push before it is weakening.
    weaken_ratio: float = 0.75

    #: A pullback over this fraction of the push it is retracing is weakening.
    pullback_ratio: float = 0.75

    #: Swings needed before any state but unclear is given. Two highs and two
    #: lows is the minimum a higher-high-higher-low rule can be read from.
    min_swings: int = 4


@dataclass(frozen=True)
class Swing:
    """A confirmed turn, and how it stands against the one before it."""

    bar: int
    price: float
    high: bool
    seen_at: int
    kind: str  # HH, LH, EQH, HL, LL, EQL, or first

    @property
    def age(self) -> int:
        return self.seen_at - self.bar


@dataclass(frozen=True)
class Label:
    """What bar N is, why, and what would change it."""

    bar: int
    state: str
    reason: dict = field(default_factory=dict)
    swings: tuple[Swing, ...] = ()
    protective: float | None = None
    protective_bar: int | None = None

    @property
    def trending(self) -> bool:
        return self.state in (UPTREND, DOWNTREND, WEAKENING)

    def says(self) -> str:
        """The reason as a sentence. Rendered from the data, never stored as
        prose - so the stats can group by rule and the chart can still explain
        itself."""
        r = self.reason

        if self.state == UNCLEAR:
            return f"Unclear: {r.get('why', 'not enough confirmed swings yet')}"

        if self.state == BREAK:
            # The level's own kind is kept in the reason but not read out: an
            # earlier swing high can be an HH and still be what a downtrend
            # breaks through, and "the HH" in that sentence reads like a
            # contradiction.
            return (
                f"Break: close {r['close']:.2f} at bar {r['broke_at']} "
                f"{'below' if r['was'] == UPTREND else 'above'} the swing "
                f"{'low' if r['was'] == UPTREND else 'high'} "
                f"{r['level']:.2f} from bar {r['level_bar']}"
            )

        if self.state == WEAKENING:
            if r["why"] == "push":
                return (
                    f"Weakening {r['was']}: the push into bar {r['bar']} ran "
                    f"{r['now']:.2f} ATR against {r['before']:.2f} before it"
                )
            return (
                f"Weakening {r['was']}: the pullback into bar {r['bar']} gave back "
                f"{r['now']:.0%} of a push of {r['before']:.2f} ATR"
            )

        if self.state == RANGE:
            return (
                f"Range: {r['high_kind']} and {r['low_kind']} disagree, price held between "
                f"{r['low']:.2f} and {r['high']:.2f}"
            )

        return (
            f"{self.state.capitalize()}: {r['high_kind']} {r['high']:.2f} at bar {r['high_bar']} "
            f"and {r['low_kind']} {r['low']:.2f} at bar {r['low_bar']}"
        )


def swings_at(view: Replay, rules: Rules = Rules()) -> list[Swing]:
    """The confirmed turns as at this bar, oldest first.

    The chain of pivots is closed off with where price is now, so the newest
    pivot can be told a high from a low. Both ends of the chain are then
    unusable as swings - the oldest has nothing before it to reverse from - and
    are dropped.
    """
    pivots = view.pivots()
    if len(pivots) < 2:
        return []

    price = view.dataset.source_price()
    atr = view.dataset.bars["atr"]

    chain = [(p.bar, p.price, p.seen_at) for p in pivots]
    chain.append((view.bar, float(price.loc[view.bar]), view.bar))

    found: list[Swing] = []
    highs: Swing | None = None
    lows: Swing | None = None

    for i in range(1, len(chain) - 1):
        before, here, after = chain[i - 1], chain[i], chain[i + 1]

        rising_in = here[1] > before[1]
        rising_out = after[1] > here[1]

        # A bend, not a turn: two pieces going the same way with different
        # steepness. It is a fact about the fit, not about the market.
        if rising_in == rising_out:
            continue

        high = rising_in and not rising_out
        last = highs if high else lows
        room = rules.equal_atr * float(atr.loc[here[0]])

        if last is None:
            kind = "first"
        elif here[1] > last.price + room:
            kind = "HH" if high else "HL"
        elif here[1] < last.price - room:
            kind = "LH" if high else "LL"
        else:
            kind = "EQH" if high else "EQL"

        swing = Swing(bar=here[0], price=here[1], high=high, seen_at=here[2], kind=kind)
        found.append(swing)

        if high:
            highs = swing
        else:
            lows = swing

    return found


def label(data: Dataset, bar: int, rules: Rules = Rules(), view: Replay | None = None) -> Label:
    """What bar N is, from what bar N knew."""
    view = view or at(data, bar)
    found = swings_at(view, rules)

    highs = [s for s in found if s.high]
    lows = [s for s in found if not s.high]

    if len(found) < rules.min_swings or not highs or not lows:
        return Label(bar=bar, state=UNCLEAR, swings=tuple(found),
                     reason={"why": f"{len(found)} confirmed swings, {rules.min_swings} needed"})

    high, low = highs[-1], lows[-1]
    atr = float(data.bars["atr"].loc[bar])
    close = float(data.bars["close"].loc[bar])

    if high.kind == "HH" and low.kind == "HL":
        trend, protective = UPTREND, _protective(lows, high)
    elif high.kind == "LH" and low.kind == "LL":
        trend, protective = DOWNTREND, _protective(highs, low)
    else:
        trend, protective = None, None

    if trend is None:
        top, bottom = max(s.price for s in highs), min(s.price for s in lows)
        return Label(
            bar=bar, state=RANGE, swings=tuple(found),
            reason={"high_kind": high.kind, "low_kind": low.kind, "high": top, "low": bottom,
                    "close": close},
        )

    # The break comes first, because it is the one thing that can be true
    # before any new pivot confirms - and the one that matters most.
    broke = _broken(data, bar, trend, protective, rules, atr)
    if broke is not None:
        return Label(
            bar=bar, state=BREAK, swings=tuple(found),
            protective=protective.price, protective_bar=protective.bar,
            reason={"was": trend, "level": protective.price, "level_bar": protective.bar,
                    "level_kind": protective.kind, "broke_at": broke[0], "close": broke[1]},
        )

    tired = _weakening(found, trend, data, rules)
    if tired is not None:
        return Label(
            bar=bar, state=WEAKENING, swings=tuple(found),
            protective=protective.price, protective_bar=protective.bar,
            reason={**tired, "was": trend},
        )

    return Label(
        bar=bar, state=trend, swings=tuple(found),
        protective=protective.price, protective_bar=protective.bar,
        reason={"high_kind": high.kind, "high": high.price, "high_bar": high.bar,
                "low_kind": low.kind, "low": low.price, "low_bar": low.bar},
    )


def _protective(candidates: list[Swing], extreme: Swing) -> Swing:
    """The swing a break would have to take out: the last one BEFORE the
    extreme, not the newest one.

    In an uptrend the level that matters is the low the last push up started
    from. The newest low may be forming right now, a bar or two old, with
    nothing yet built on top of it - protecting that one makes every ordinary
    wiggle a break of structure. Measured on MNQ it called a quarter of all
    bars broken, which is another way of saying it said nothing.

    The low before the last high has been validated by that high: price left
    it, made a new extreme, and coming back through it now genuinely undoes
    something.
    """
    earlier = [swing for swing in candidates if swing.bar < extreme.bar]

    return earlier[-1] if earlier else candidates[-1]


def _broken(data: Dataset, bar: int, trend: str, protective: Swing, rules: Rules,
            atr: float) -> tuple[int, float] | None:
    """The first close past the protective swing since that swing happened.

    Since the swing, not since it was confirmed: the level existed the moment
    price made it, and a market that broke it while the segmentation was still
    catching up has still broken it. What could not be known at the time is the
    LEVEL, and that is held to the confirmation - the swing is only read once
    it is in the confirmed set.
    """
    closes = data.bars["close"].loc[protective.bar + 1 : bar]
    if closes.empty:
        return None

    room = rules.break_atr * atr
    past = closes < protective.price - room if trend == UPTREND else closes > protective.price + room

    if not past.any():
        return None

    first = int(closes.index[past.argmax()])

    return first, float(closes.loc[first])


def _weakening(found: list[Swing], trend: str, data: Dataset, rules: Rules) -> dict | None:
    """Is the trend running out of legs?

    Two ways of asking, both in ATRs of the bars each leg covered, so a quiet
    afternoon is not mistaken for exhaustion:

      push      the newest leg in the trend's direction is shorter than the one
                before it by more than weaken_ratio.
      pullback  the newest leg against the trend gave back more of the push it
                retraced than pullback_ratio allows.
    """
    if len(found) < 3:
        return None

    atr = data.bars["atr"]
    up = trend == UPTREND

    legs = []
    for a, b in zip(found, found[1:]):
        local = float(atr.loc[a.bar : b.bar].mean())
        amplitude = abs(b.price - a.price) / local if local else 0.0
        legs.append((b.bar, (b.price > a.price) == up, amplitude))

    pushes = [leg for leg in legs if leg[1]]
    against = [leg for leg in legs if not leg[1]]

    if len(pushes) >= 2 and pushes[-1][2] < rules.weaken_ratio * pushes[-2][2]:
        return {"why": "push", "bar": pushes[-1][0], "now": pushes[-1][2], "before": pushes[-2][2]}

    if pushes and against and against[-1][0] > pushes[-1][0]:
        give = against[-1][2] / pushes[-1][2] if pushes[-1][2] else 0.0
        if give > rules.pullback_ratio:
            return {"why": "pullback", "bar": against[-1][0], "now": give, "before": pushes[-1][2]}

    return None


def tag(data: Dataset, rules: Rules = Rules(), start: int | None = None,
        stop: int | None = None) -> Iterator[Label]:
    """Every bar in turn, labelled from what that bar knew.

    One pass over the event log rather than one replay per bar - the same reason
    :func:`segments.replay.walk` exists.
    """
    for view in walk(data, start, stop):
        yield label(data, view.bar, rules, view=view)


def table(data: Dataset, rules: Rules = Rules()) -> pd.DataFrame:
    """The whole history as one row per bar: state, reason, protective level.

    The reason is kept as its rendered sentence here and as the rule name in its
    own column, so the stats can group by rule and a person can still read a row.
    """
    rows = []
    for one in tag(data, rules):
        rows.append(
            {
                "bar": one.bar,
                "state": one.state,
                "rule": one.reason.get("why", one.reason.get("level_kind", "")),
                "protective": one.protective,
                "swings": len(one.swings),
                "says": one.says(),
            }
        )

    return pd.DataFrame(rows).set_index("bar")


def shares(data: Dataset, rules: Rules = Rules()) -> pd.Series:
    """How much of the history each state accounts for.

    Worth a look before trusting any of it: a tagger that calls 90% of the bars
    unclear, or 90% of them a trend, is not describing the market.
    """
    states = table(data, rules)["state"]

    return states.value_counts(normalize=True).reindex(STATES).fillna(0.0)
