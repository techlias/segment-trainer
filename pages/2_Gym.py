"""Trade the replay by hand, and find out whether it is worth anything.

Two buttons, bar by bar, with the future hidden. :mod:`segments.gym` holds the
rules and :mod:`segments.scoring` the verdict; this page only draws them and
takes the presses.

WHAT THIS PAGE DELIBERATELY DOES NOT OFFER

    The viewer's reveal slider, its draggable bar and its random jump are all
    missing here, and that is the point rather than an omission. Every one of
    them would let a trade be taken knowing what came next, and a score built on
    that measures nothing. The bar only moves forward, one press at a time.

    The start bar is drawn before the chart is, so it is not chosen by looking
    at it either.
"""

from __future__ import annotations

import json
import random

import matplotlib.pyplot as plt
import streamlit as st

import sources
from segments import gym, plot, scoring

st.set_page_config(page_title="Gym", layout="wide")

STATE_COLOUR = {gym.FLAT: "grey", gym.LONG: "green", gym.SHORT: "red"}


# --- the bars -----------------------------------------------------------------

kind, stem, data = sources.pick()

if data is None:
    st.title("Gym")
    st.info(
        "Pick some bars in the sidebar - a NinjaTrader export, or Yahoo Finance if "
        "there is no export on this machine."
    )
    st.stop()

if sources.is_public(kind):
    st.sidebar.caption(sources.LABEL)

st.sidebar.subheader("The session")
history = st.sidebar.slider("Bars of context", 40, 300, 120, 10,
                            help="How much chart you can see behind the bar you are on")

# The window a session may start in. Far enough from the left that the chart has
# something behind it, far enough from the right that there is room to trade.
earliest = data.first_bar + max(history, data.meta.window)
latest = data.last_bar - 50

if latest <= earliest:
    st.error("Not enough bars for a session. Load a longer history.")
    st.stop()


def start_session(at: int | None = None) -> None:
    """A new session, from a bar nobody has looked at yet."""
    st.session_state.session = gym.Session(
        data, start=random.randint(earliest, latest) if at is None else at
    )


if st.sidebar.button("New session", width="stretch", key="new"):
    start_session()

# Which bars these are, by their key and not by the object: st.cache_data hands
# back a fresh copy on every rerun, so comparing the dataset itself says "new
# bars" on every single press and silently restarts the session under you.
these_bars = (kind, str(stem))

if "session" not in st.session_state or st.session_state.get("trading") != these_bars:
    st.session_state.trading = these_bars
    start_session()

session: gym.Session = st.session_state.session


# --- the presses --------------------------------------------------------------

def press(action: str) -> None:
    if session.allowed(action):
        session.press(action)


def advance(by: int) -> None:
    session.advance(by)


def finish() -> None:
    session.finish()
    st.session_state.history = history_of() + [
        trade for trade in session.trades if trade not in history_of()
    ]


def history_of() -> list:
    return st.session_state.setdefault("history", [])


left, right, step, step10, done = st.columns([1.1, 1.1, 1, 1, 1.2])

long_label = "Buy  ·  B" if session.flat else "Buy  ·  B (close)"
short_label = "Sell  ·  S" if session.flat else "Sell  ·  S (close)"

left.button(long_label, width="stretch", disabled=not session.allowed(gym.BUY),
            on_click=press, args=(gym.BUY,), key="buy")
right.button(short_label, width="stretch", disabled=not session.allowed(gym.SELL),
             on_click=press, args=(gym.SELL,), key="sell")
step.button("Next bar  ·  →", width="stretch", disabled=session.finished,
            on_click=advance, args=(1,), key="step")
step10.button("+10 bars", width="stretch", disabled=session.finished,
              on_click=advance, args=(10,), key="step10")
done.button("End session", width="stretch", on_click=finish, key="done")

# Keyboard shortcuts. Streamlit has no binding of its own, so this reaches out
# of the component's iframe and clicks the real buttons by their label. Best
# effort, and the page works without it - every shortcut has a button too.
st.iframe(
    """
    <script>
    const doc = window.parent.document;
    const hit = (text) => {
        const button = Array.from(doc.querySelectorAll('button'))
            .find(one => one.innerText.trim().startsWith(text) && !one.disabled);
        if (button) button.click();
    };
    if (!doc.__gymKeys) {
        doc.__gymKeys = true;
        doc.addEventListener('keydown', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
            if (event.key === 'b' || event.key === 'B') hit('Buy');
            if (event.key === 's' || event.key === 'S') hit('Sell');
            if (event.key === 'ArrowRight') hit('Next bar');
        });
    }
    </script>
    """,
    # st.iframe will not take 0 the way components.html did, and one pixel of
    # nothing is as close to invisible as it allows.
    height=1,
)


# --- where you are ------------------------------------------------------------

when = data.bars["time"].loc[session.bar]
state, bar, held, open_r, left_to_go = st.columns(5)

state.metric("State", session.state.upper())
bar.metric("Bar", f"{session.bar}", f"{when:%a %H:%M}")

if session.position is not None:
    held.metric("Entry", f"{session.position.price:,.2f}")
    open_r.metric("Open", f"{session.open_r():+.2f} R",
                  f"{session.position.points(session.price()):+.2f} pts")
    left_to_go.metric("Bars held", session.held())
else:
    held.metric("Entry", "-")
    open_r.metric("Open", "-")
    left_to_go.metric("Bars held", "-")

if session.finished:
    st.warning("The bars have run out. **End session** books anything still open as forced.")


# --- the chart ----------------------------------------------------------------

figure, axis = plt.subplots(figsize=(15, 5.6))

# reveal is not a parameter here. There is nothing past this bar to draw.
plot.draw(data, session.bar, history=history, reveal=0, candles=True, ax=axis)

if session.position is not None:
    held_at = session.position
    axis.scatter([held_at.bar], [held_at.price], s=90, zorder=6,
                 marker="^" if held_at.direction == gym.LONG else "v",
                 color=plot.UP if held_at.direction == gym.LONG else plot.DOWN)
    axis.axhline(held_at.price, color="#8a8f98", linewidth=1.0, linestyle=":", zorder=2)

for trade in session.trades:
    if trade.exit_bar < session.bar - history:
        continue
    axis.plot([trade.entry_bar, trade.exit_bar], [trade.entry_price, trade.exit_price],
              color=plot.UP if trade.r > 0 else plot.DOWN,
              linewidth=1.0, linestyle="-", alpha=0.45, zorder=5)

st.pyplot(figure, width="stretch")
plt.close(figure)


# --- the score ----------------------------------------------------------------

closed = history_of() + [trade for trade in session.trades if trade not in history_of()]

if not closed:
    st.info("No closed trades yet. **B** opens a long, **S** a short, **→** moves one bar on.")
    st.stop()

overall = scoring.report(closed)
form = scoring.rolling(closed)

st.subheader(overall.score.tier)

if not overall.score.ranked:
    st.caption(
        f"{overall.score.trades} of {scoring.RANK_AFTER} trades. A number appears at "
        f"{scoring.RANK_AFTER}; it takes a few hundred before the tier means much, which "
        "is why the tier is read off the bottom of the error bar and not off the number."
    )
else:
    st.caption(
        f"Sharpe {overall.score.sharpe:+.3f}, 95% band "
        f"{overall.score.low:+.3f} to {overall.score.high:+.3f}. "
        f"The tier is the bottom of that band — on the number alone it would read "
        f"**{overall.score.tier_if_believed}**."
    )

one, two, three, four, five = st.columns(5)
one.metric("Trades", overall.trades, f"{overall.forced} forced" if overall.forced else None)
two.metric("Win rate", f"{overall.win_rate:.0%}",
           f"{overall.wins}W / {overall.losses}L" + (f" / {overall.scratches}=" if overall.scratches else ""))
three.metric("Expectancy", f"{overall.expectancy:+.3f} R")
four.metric("Total", f"{overall.total_r:+.1f} R")
five.metric("Max drawdown", f"{overall.max_drawdown:.1f} R",
            f"{overall.longest_losing_streak} in a row")

six, seven, eight, nine, ten = st.columns(5)
six.metric("Average win", f"{overall.average_win:+.2f} R" if overall.average_win else "-")
seven.metric("Average loss", f"{overall.average_loss:+.2f} R" if overall.average_loss else "-")
eight.metric("Payoff", f"{overall.payoff:.2f}" if overall.payoff else "-")
nine.metric("Profit factor", f"{overall.profit_factor:.2f}" if overall.profit_factor else "-")
ten.metric("MFE kept", f"{overall.capture:.0%}" if overall.capture else "-",
           help="Of the move a winner offered, how much was taken. A good entry with a "
                "bad exit shows up here and nowhere else.")

curve, split = st.columns([2, 1])

with curve:
    st.caption("Cumulative R")
    figure, axis = plt.subplots(figsize=(8, 2.6))
    axis.plot(range(1, len(overall.equity) + 1), overall.equity,
              color=plot.UP if overall.total_r >= 0 else plot.DOWN, linewidth=1.6)
    axis.axhline(0, color="#b0b4ba", linewidth=1.0)
    axis.set_xlabel("trade")
    axis.grid(True, alpha=0.15)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with split:
    st.caption("Current form — last 50")
    st.metric("Sharpe", f"{form.score.sharpe:+.3f}" if form.score.sharpe else "-",
              f"{form.trades} trades")
    st.caption("No tier here on purpose: at fifty trades the error bar is wider than the ladder.")

for name, attribute in (("By direction", "direction"), ("By timeframe", "timeframe")):
    with st.expander(name):
        rows = []
        for label, part in scoring.by(closed, attribute).items():
            rows.append({
                "": label,
                "trades": part.trades,
                "win rate": f"{part.win_rate:.0%}",
                "expectancy (R)": f"{part.expectancy:+.3f}",
                "total (R)": f"{part.total_r:+.1f}",
                "Sharpe": f"{part.score.sharpe:+.3f}" if part.score.sharpe else "-",
            })
        st.dataframe(rows, hide_index=True, width="stretch")


# --- keeping it ----------------------------------------------------------------

with st.expander("History"):
    st.caption(
        "Streamlit Community Cloud forgets its filesystem on every redeploy, so the history "
        "lives in this browser session and leaves as a file. Download before you close the tab."
    )

    keep, bring, wipe = st.columns(3)

    keep.download_button(
        "Download history", width="stretch",
        data=json.dumps([trade.as_dict() for trade in closed], indent=1),
        file_name="gym-history.json", mime="application/json",
    )

    brought = bring.file_uploader("Load a history", type="json", label_visibility="collapsed")
    if brought is not None:
        loaded = [gym.Trade.from_dict(row) for row in json.load(brought)]
        known = {trade.as_dict()["session"] for trade in history_of()}
        st.session_state.history = history_of() + [
            trade for trade in loaded if trade.session not in known
        ]
        st.success(f"{len(loaded)} trades read in.")

    if wipe.button("Forget everything", width="stretch"):
        st.session_state.history = []
        start_session()
        st.rerun()
