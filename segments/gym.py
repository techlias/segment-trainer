"""Trading the replay by hand, one bar at a time.

Two buttons and two states. The point is not to make money in here - there is no
money in here - it is to find out whether entries and exits picked by eye are
worth anything once the future is actually hidden. :mod:`segments.scoring` is
what answers that; this module only makes the trades and refuses to make
impossible ones.

NOTHING HERE CAN SEE PAST THE BAR IT IS ON

    Every fill is the close of the bar the session is standing on, and the
    session only ever moves forward. That is the whole validity of the exercise,
    so it is enforced here rather than left to the page that draws it: a caller
    cannot set the bar backwards, and a trade cannot be opened or closed at a
    bar the session has not reached.

    The viewer's reveal slider and its draggable bar are fine for reading a
    chart and fatal for this, which is why a gym page has to lock them rather
    than reuse them.

THE ALTERNATION RULE, AND THE READING OF IT

    The brief says two things that pull apart: "in a trade only the opposite
    button is enabled", and "the same button can never be clicked twice in a
    row".

    Taken literally across a close, the second one is degenerate. Close a long
    with Sell, and Sell is now the last button; if the next click may not be
    Sell, the only thing left is Buy, which from flat opens another long. Every
    trade after the first would be a long, for ever, and the short button would
    be unreachable.

    So the rule is read as being about a POSITION and not about a sequence of
    clicks: you may never add to a position, and you may never close one with
    the button that opened it. Flat is flat - both buttons live, Buy for a long
    and Sell for a short. Closing a long with Sell and then pressing Sell again
    to open a short is two deliberate presses, and it is allowed. Reversing in a
    single press is not, which is the part of the brief that matters.

A TRADE LASTS AT LEAST ONE BAR

    Opening and closing on the same bar would fill both at the same close for a
    guaranteed zero, which is not a trade, it is a way of padding the count. The
    exit has to be at a later bar than the entry.

EVERYTHING IS IN R

    points divided by the ATR of the entry bar. A gym session on a quiet
    afternoon and one on a fast open are then the same measurement, which is the
    only way a score over mixed sessions means anything. The ATR column is the
    one the export carries, so this is the same number the segmentation was cut
    with - see PriceSegments on why it is in ATRs and not in ticks.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from .dataset import Dataset

FLAT = "flat"
LONG = "long"
SHORT = "short"

BUY = "buy"
SELL = "sell"

#: Which press opens which direction when flat, and which one closes it.
OPENS = {BUY: LONG, SELL: SHORT}
CLOSES = {LONG: SELL, SHORT: BUY}


@dataclass(frozen=True)
class Position:
    """A trade that has been opened and not yet closed."""

    direction: str
    bar: int
    price: float
    atr: float
    time: dt.datetime

    def points(self, price: float) -> float:
        """What the position is worth at this price, in points."""
        return price - self.price if self.direction == LONG else self.price - price

    def r(self, price: float) -> float:
        return self.points(price) / self.atr if self.atr else 0.0


@dataclass(frozen=True)
class Trade:
    """A closed trade, and everything needed to score it without the bars."""

    session: str
    instrument: str
    timeframe: str
    direction: str

    entry_bar: int
    entry_time: dt.datetime
    entry_price: float

    exit_bar: int
    exit_time: dt.datetime
    exit_price: float

    bars: int
    points: float
    atr: float
    r: float

    #: Best and worst the trade ever stood, in R, from the highs and lows rather
    #: than the closes - the excursion that actually happened, not the one the
    #: close happened to print.
    mfe: float
    mae: float

    #: Closed by the session ending rather than by a press.
    forced: bool = False

    @property
    def won(self) -> bool:
        return self.r > 0

    @property
    def scratch(self) -> bool:
        return self.r == 0

    def as_dict(self) -> dict:
        """A plain dict, dates as ISO strings, for the history file."""
        out = dataclasses.asdict(self)
        out["entry_time"] = self.entry_time.isoformat()
        out["exit_time"] = self.exit_time.isoformat()

        return out

    @classmethod
    def from_dict(cls, row: dict) -> "Trade":
        row = dict(row)
        row["entry_time"] = dt.datetime.fromisoformat(row["entry_time"])
        row["exit_time"] = dt.datetime.fromisoformat(row["exit_time"])

        return cls(**row)


class Session:
    """One run through a stretch of bars, with the trades it produced.

    Holds no Streamlit and draws nothing: a page keeps one of these in its state
    and calls it, which is also what makes it testable without a browser and
    replayable by a simulator.
    """

    def __init__(self, data: Dataset, start: int, name: str | None = None):
        if not data.first_bar <= start <= data.last_bar:
            raise ValueError(f"bar {start} is outside {data.first_bar}..{data.last_bar}")

        self.data = data
        self.start = int(start)
        self.bar = int(start)
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.name = name or self._name()

    # --- where it is ---------------------------------------------------------
    @property
    def state(self) -> str:
        return FLAT if self.position is None else self.position.direction

    @property
    def flat(self) -> bool:
        return self.position is None

    @property
    def finished(self) -> bool:
        return self.bar >= self.data.last_bar

    def price(self) -> float:
        """The fill: the close of the bar being stood on."""
        return float(self.data.bars["close"].loc[self.bar])

    def open_r(self) -> float:
        """What the position is worth right now, in R. Zero when flat."""
        return 0.0 if self.position is None else self.position.r(self.price())

    def held(self) -> int:
        return 0 if self.position is None else self.bar - self.position.bar

    # --- what may be pressed -------------------------------------------------
    def allowed(self, action: str) -> bool:
        """Whether this press is legal now. The UI greys out what this refuses.

        A press is also refused on the last bar when it would open a position,
        since a trade opened there could never reach a second bar.
        """
        if action not in OPENS:
            return False

        if self.position is None:
            return not self.finished

        return action == CLOSES[self.position.direction] and self.bar > self.position.bar

    def press(self, action: str) -> Trade | None:
        """Buy or sell. Returns the trade when one closes, otherwise None.

        Refused presses raise rather than pass quietly: the UI is supposed to
        have disabled them, so one arriving here is a bug somewhere and should
        say so.
        """
        if not self.allowed(action):
            raise ValueError(f"{action!r} is not allowed from {self.state} at bar {self.bar}")

        if self.position is None:
            self.position = Position(
                direction=OPENS[action],
                bar=self.bar,
                price=self.price(),
                atr=float(self.data.bars["atr"].loc[self.bar]),
                time=self._time(self.bar),
            )

            return None

        return self._close()

    # --- moving on -----------------------------------------------------------
    def advance(self, by: int = 1) -> int:
        """Forward only, and never past the last bar."""
        if by < 1:
            raise ValueError(f"a session only goes forward: asked to move {by}")

        self.bar = min(self.bar + by, self.data.last_bar)

        return self.bar

    def finish(self) -> Trade | None:
        """End the session, closing an open position at the last bar.

        The forced flag is kept on the trade rather than the trade being thrown
        away. A position still open when the bars ran out is a real thing that
        happened - usually a losing one being held - and dropping it would
        flatter every score that followed.
        """
        self.bar = self.data.last_bar

        if self.position is None:
            return None

        return self._close(forced=True)

    # --- the parts nobody outside calls --------------------------------------
    def _close(self, forced: bool = False) -> Trade:
        held = self.position
        price = self.price()
        points = held.points(price)
        atr = held.atr or 1.0

        best, worst = self._excursion(held, self.bar)

        trade = Trade(
            session=self.name,
            instrument=self.data.meta.instrument,
            timeframe=self.data.meta.period,
            direction=held.direction,
            entry_bar=held.bar,
            entry_time=held.time,
            entry_price=held.price,
            exit_bar=self.bar,
            exit_time=self._time(self.bar),
            exit_price=price,
            bars=self.bar - held.bar,
            points=points,
            atr=held.atr,
            r=points / atr,
            mfe=best / atr,
            mae=worst / atr,
            forced=forced,
        )

        self.trades.append(trade)
        self.position = None

        return trade

    def _excursion(self, held: Position, exit_bar: int) -> tuple[float, float]:
        """Best and worst the trade ever stood, in points.

        From the highs and lows over the bars the trade was open, because that
        is where price actually went. Measured from the entry bar, whose own
        high and low count: the trade was live for the rest of that bar.
        """
        window = self.data.bars.loc[held.bar : exit_bar]
        high = float(window["high"].max())
        low = float(window["low"].min())

        if held.direction == LONG:
            return high - held.price, low - held.price

        return held.price - low, held.price - high

    def _time(self, bar: int) -> dt.datetime:
        return pd.Timestamp(self.data.bars["time"].loc[bar]).to_pydatetime()

    def _name(self) -> str:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

        return f"{self.data.meta.instrument} {self.data.meta.period} @{self.start} {stamp}"
