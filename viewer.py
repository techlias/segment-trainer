"""A chart you can scrub through, showing only what each bar could know.

    streamlit run viewer.py

No labels yet - nothing in this project has an opinion until the tagger is
written. What this is for is the decisions that come before it: whether the
tolerance draws the legs you actually read, how much of a move is over by the
time its turn is confirmed, and what the right hand edge looks like at the
moment a question would be asked.

Every frame comes from ``replay.at``, so the chart cannot show you anything the
bar did not know, and the drawing is ``plot.draw`` - the same function the PNGs
and the NinjaTrader comparison use. One picture, one implementation.

THE REVEAL

    Reveal draws the next few bars in grey without re-reading the segmentation.
    That is deliberate: the point is to see what happened next TO THE READ YOU
    MADE, not to be shown a tidier read of the same chart. It is the game's
    answer key, wired up early because it is the only way to judge a tolerance
    honestly.
"""

from __future__ import annotations

import json
import random

import matplotlib.pyplot as plt
import streamlit as st

import sources
from segments import fit, gym, plot, replay, scoring, studies, tagger

st.set_page_config(page_title="Segments", layout="wide")


def kept_trades() -> list:
    """Trades from sessions already ended, this browser session.

    Community Cloud forgets its filesystem on every redeploy, so nothing is
    written down - the history lives here and leaves as a file.
    """
    return st.session_state.setdefault("kept", [])


kind, stem, original = sources.pick()

if original is None:
    if kind == "export":
        st.title("Segments")
        st.info(
            "No export found. Drop **SegmentExport** on a chart in NinjaTrader, let it run "
            "through the history, then upload the two CSVs it wrote using the sidebar - "
            "or put them in the `data` folder beside this app.\n\n"
            "Or switch **Where from** to **Yahoo Finance** and read public bars instead."
        )
    st.stop()

if kind == "public":
    # The label. Everything here is cut by the same code an export is cut by,
    # but the BARS are Yahoo's - aggregated, revised, and with the session
    # inferred from gaps rather than read from a template. There is no
    # NinjaTrader run behind them to check against, and saying so once on screen
    # is cheaper than someone eventually concluding the pipeline is broken.
    st.sidebar.caption(sources.LABEL)

problems = original.check()
if problems:
    st.sidebar.warning("\n".join(f"- {problem}" for problem in problems))

# --- what to cut, and how ----------------------------------------------------

st.sidebar.subheader("The cut")
st.sidebar.caption("Anything other than the settings these bars arrived with re-cuts them here.")

meta = original.meta
method = st.sidebar.selectbox("Method", fit.METHODS, index=fit.METHODS.index(meta.method))
source = st.sidebar.selectbox(
    "Source", ["Close", "BodyCentre", "Median", "Typical"],
    index=["Close", "BodyCentre", "Median", "Typical"].index(meta.source),
)
tolerance = st.sidebar.slider("Tolerance (ATR)", 0.2, 4.0, float(meta.tolerance), 0.05)
min_bars = st.sidebar.slider("Minimum length (bars)", 1, 12, int(meta.minbars))
window = st.sidebar.select_slider(
    "Window (bars)", [100, 150, 250, 400, 600], value=min([100, 150, 250, 400, 600],
                                                          key=lambda v: abs(v - meta.window))
)

untouched = (
    method == meta.method and source == meta.source and tolerance == meta.tolerance
    and min_bars == meta.minbars and window == meta.window
)

data = (original if untouched
        else sources.recut_as(kind, stem, method, source, tolerance, min_bars, window))

# --- what to show ------------------------------------------------------------

st.sidebar.page_link("pages/1_Tutorial.py", label="How to read this page", icon=":material/help:")

st.sidebar.subheader("The view")
history = st.sidebar.slider("History (bars)", 40, 400, 150, 10)

# The gym is a mode of this page rather than a page of its own, because the
# whole apparatus - the bands, the turn names, the protective level, the
# momentum panel - is what you would actually read before deciding to buy.
# Trading a stripped-down chart would measure something, but not the thing this
# trainer teaches.
#
# What it takes away is the two controls that can see the future: reveal, and a
# bar you can drag anywhere. In the gym the bar only goes forward, one press at
# a time. See segments/gym.py.
trading = st.sidebar.toggle(
    "Trading gym", False,
    help="Two buttons, bar by bar, with the future hidden. The reveal slider and the "
         "bar slider go away while it is on - both of them would let you trade knowing "
         "what came next.",
)

reveal = 0 if trading else st.sidebar.slider(
    "Reveal (bars)", 0, 120, 0, 5,
    help="Bars past the one you are on, drawn in grey. The answer.",
)
segments_on = st.sidebar.checkbox("Show the segments", True,
                                  help="Off is the chart without training wheels")
# One control rather than two checkboxes, because the three are a choice and not
# three independent things: candles already draw the range the sticks would.
bars_on = st.sidebar.radio("Show the bars", ["Hidden", "High-low sticks", "Candles"],
                           index=0, horizontal=False,
                           help="Behind the segments. Candles show the open and close too - "
                                "and show that the turns sit where the fitted price turned, "
                                "not at the high or the low of the bar.")
sticks_on = bars_on == "High-low sticks"
candles_on = bars_on == "Candles"
bands_on = st.sidebar.checkbox("Show the Bollinger", True,
                               help="The band behind the price. Off is worth trying: it is the one "
                                    "thing here that suggests where price should go rather than "
                                    "describing where it went.")
classify = st.sidebar.checkbox("Classify", True,
                               help="The tagger's reading: the swing names and the protective level")

with st.sidebar.expander("The rules"):
    st.caption("Defaults are a starting point. Argue with them against a long export, not against a few days.")
    rules = tagger.Rules(
        equal_atr=st.slider("Same level within (ATR)", 0.0, 0.5, 0.15, 0.01,
                            help="Two swings this close are a double top, not a higher high"),
        break_atr=st.slider("Break clears by (ATR)", 0.0, 0.5, 0.10, 0.01,
                            help="How far past the level a close must settle to be a break"),
        weaken_ratio=st.slider("Push shrunk to", 0.3, 1.0, 0.75, 0.05),
        pullback_ratio=st.slider("Pullback gave back", 0.3, 1.0, 0.75, 0.05),
    )

    # Not the tagger's rules - nothing the tagger does reads either of these -
    # but they belong beside them, because they are the other numbers on the
    # page you would change to argue with what you are being shown. Top to
    # bottom in the order they appear on the chart.
    band_period = st.slider("Bollinger (bars)", 5, 20, studies.BAND_PERIOD, 1,
                            help=f"The band behind the price, at "
                                 f"{studies.BAND_DEVIATIONS:g} standard deviations. "
                                 "Short wraps price like an envelope; long barely moves")
    rsi_period = st.slider("RSI (bars)", 4, 20, studies.RSI_PERIOD, 1,
                           help="The momentum panel under the chart. Short reads every leg; "
                                "long reads the session")

# --- where to stand ----------------------------------------------------------

first, last = data.first_bar, data.last_bar

# Which bars these are, not how they were cut. Changing the tolerance should
# leave you standing where you were - that is the whole point of moving it - but
# changing the instrument, the interval or the history means the bar you were on
# is a different moment in a different market, and keeping its number would be
# keeping a coincidence.
standing_on = (kind, str(stem))

if st.session_state.get("standing_on") != standing_on:
    st.session_state.standing_on = standing_on
    st.session_state.bar = last

if "bar" not in st.session_state or not first <= st.session_state.bar <= last:
    st.session_state.bar = last

earliest = min(first + max(history, data.meta.window), last - 50)

if trading:
    # Forward only, from a bar nobody has looked at. No slider: a draggable bar
    # is the same lookahead as the reveal, one drag further away.
    if "session" not in st.session_state or st.session_state.get("trading_on") != standing_on:
        st.session_state.trading_on = standing_on
        st.session_state.session = gym.Session(
            data, start=random.randint(max(first, earliest), max(first + 1, last - 50))
        )

    session: gym.Session = st.session_state.session

    def press(action: str) -> None:
        if session.allowed(action):
            session.press(action)

    buy, sell, step, step10, fresh = st.columns([1.1, 1.1, 1, 1, 1.2])
    buy.button("Buy  ·  B", width="stretch", disabled=not session.allowed(gym.BUY),
               on_click=press, args=(gym.BUY,), key="buy")
    sell.button("Sell  ·  S", width="stretch", disabled=not session.allowed(gym.SELL),
                on_click=press, args=(gym.SELL,), key="sell")
    step.button("Next bar  ·  →", width="stretch", disabled=session.finished,
                on_click=session.advance, args=(1,), key="step")
    step10.button("+10 bars", width="stretch", disabled=session.finished,
                  on_click=session.advance, args=(10,), key="step10")

    if fresh.button("End & restart", width="stretch", key="fresh"):
        session.finish()
        st.session_state.kept = kept_trades() + [
            one for one in session.trades if one not in kept_trades()
        ]
        st.session_state.session = gym.Session(
            data, start=random.randint(max(first, earliest), max(first + 1, last - 50))
        )
        st.rerun()

    bar = session.bar
else:
    step_back, step_on, jump, _ = st.columns([1, 1, 1, 9])
    if step_back.button("◀", width="stretch"):
        st.session_state.bar = max(first, st.session_state.bar - 1)
    if step_on.button("▶", width="stretch"):
        st.session_state.bar = min(last, st.session_state.bar + 1)
    if jump.button("random", width="stretch"):
        st.session_state.bar = random.randint(max(first, earliest), max(first + 1, last - reveal))

    bar = st.slider("Bar", first, last, key="bar")

# --- the chart ---------------------------------------------------------------

view = replay.at(data, bar)
pivots = view.pivots()
pieces = view.segments()
reading = tagger.label(data, bar, rules, view=view) if classify else None

# Price over momentum, sharing an x axis so the two can never drift apart by a
# bar - which is the only way a panel underneath is worth anything.
figure, (axis, lower) = plt.subplots(
    2, 1, sharex=True, figsize=(15, 6.6),
    gridspec_kw={"height_ratios": [3.4, 1], "hspace": 0.08},
)

# Behind everything, and drawn before everything, so nothing has to argue about
# z order with it. The band reaches to the revealed bars too when they are shown:
# it is context, not a reading, and hiding it at the edge would only make the
# right hand side look calmer than it was.
close = data.bars["close"]
left = max(first, bar - history)
right = min(last, bar + reveal)

if bands_on:
    plot.bands(axis, studies.bollinger(close, period=band_period).loc[left:right])

if segments_on:
    plot.draw(data, bar, history=history, reveal=reveal,
              sticks=sticks_on, candles=candles_on, ax=axis)
    if reading is not None:
        plot.annotate(axis, reading, right=bar)
else:
    # The same window with the lines taken away: price alone, which is the level
    # the training is actually aiming at. The bars still answer to the sidebar
    # here - this is the view the reading is practised on, and practising it on
    # candles when the chart being read is candles is the whole point.
    price = data.source_price()
    if candles_on:
        plot.candlesticks(axis, data.bars.loc[left:bar])
    elif sticks_on:
        window = data.bars.loc[left:bar]
        axis.vlines(window.index, window["low"], window["high"],
                    color="#c8ccd2", linewidth=0.8, zorder=1)
    axis.plot(price.loc[left:bar].index, price.loc[left:bar].to_numpy(), color="#4a4f57", linewidth=1.3)
    if reveal:
        axis.plot(price.loc[bar:right].index, price.loc[bar:right].to_numpy(),
                  color=plot.HIDDEN, linewidth=1.3)
        axis.axvline(bar, color="#b0b4ba", linewidth=1.0, linestyle=":")
    axis.set_xlim(left, max(bar + reveal, bar) + 2)
    axis.grid(True, alpha=0.15)
    axis.set_xlabel("bar")

if trading:
    # The open position, and whatever has already been closed in view. Drawn
    # over everything, since while a trade is on it is the thing being watched.
    if session.position is not None:
        held = session.position
        axis.scatter([held.bar], [held.price], s=110, zorder=7,
                     marker="^" if held.direction == gym.LONG else "v",
                     color=plot.UP if held.direction == gym.LONG else plot.DOWN)
        axis.axhline(held.price, color="#8a8f98", linewidth=1.0, linestyle=":", zorder=2)

    for done in session.trades:
        if done.exit_bar < left:
            continue
        axis.plot([done.entry_bar, done.exit_bar], [done.entry_price, done.exit_price],
                  color=plot.UP if done.r > 0 else plot.DOWN,
                  linewidth=1.2, alpha=0.5, zorder=6)

# The x label belongs to the bottom panel now, not to the price.
axis.set_xlabel("")
plot.momentum(lower, studies.rsi(close, rsi_period).loc[left:right], rsi_period,
              at=bar if reveal else None)
lower.set_xlabel("bar")

st.pyplot(figure, width="stretch")
plt.close(figure)

if trading:
    position, entry, open_r, bars_in = st.columns(4)
    position.metric("Position", session.state.upper())

    if session.position is not None:
        entry.metric("Entry", f"{session.position.price:,.2f}")
        open_r.metric("Open", f"{session.open_r():+.2f} R",
                      f"{session.position.points(session.price()):+.2f} pts")
        bars_in.metric("Bars held", session.held())
    else:
        entry.metric("Entry", "-")
        open_r.metric("Open", "-")
        bars_in.metric("Bars held", "-")

    if session.finished:
        st.warning("The bars have run out. **End & restart** books anything still open as forced.")

    # Keyboard shortcuts. Streamlit has no binding of its own, so this reaches
    # out of the component's iframe and clicks the real buttons by their label.
    # Best effort - every shortcut has a button too.
    st.iframe(
        """
        <script>
        const doc = window.parent.document;
        const hit = (text) => {
            const one = Array.from(doc.querySelectorAll('button'))
                .find(b => b.innerText.trim().startsWith(text) && !b.disabled);
            if (one) one.click();
        };
        if (!doc.__gymKeys) {
            doc.__gymKeys = true;
            doc.addEventListener('keydown', (event) => {
                const tag = event.target.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA') return;
                if (event.key === 'b' || event.key === 'B') hit('Buy');
                if (event.key === 's' || event.key === 'S') hit('Sell');
                if (event.key === 'ArrowRight') hit('Next bar');
            });
        }
        </script>
        """,
        # st.iframe will not take 0 the way components.html did, and one pixel
        # of nothing is as close to invisible as it allows.
        height=1,
    )

if reading is not None:
    # The colour is the state, so the reading can be taken in before it is read.
    shout = {
        tagger.UPTREND: st.success, tagger.DOWNTREND: st.error,
        tagger.BREAK: st.warning, tagger.WEAKENING: st.warning,
    }.get(reading.state, st.info)

    shout(f"**{reading.state.upper()}** — {reading.says()}")

# --- what the bar knows ------------------------------------------------------

leg = pieces[-1] if pieces else None
ticksize = data.meta.ticksize or 1.0

a, b, c, d, e = st.columns(5)
a.metric("Time", f"{data.bars.loc[bar, 'time']:%a %H:%M}")
b.metric("Turns known", len(pivots))
c.metric("Leg in progress", f"{leg.bars} bars" if leg else "-",
         f"{leg.change / ticksize:+.0f}t" if leg else None)
d.metric("Newest turn", f"{bar - pivots[-1].bar} bars ago" if pivots else "-",
         f"seen {pivots[-1].age} bars late" if pivots else None)
e.metric("ATR", f"{data.bars.loc[bar, 'atr'] / ticksize:.0f}t")

if not untouched:
    st.caption(
        f"Re-cut here: {method}/{source}, tolerance {tolerance} ATR, min {min_bars} bars, "
        f"window {window}. "
        f"{'These bars arrived' if kind == 'public' else 'The export itself is'} "
        f"{meta.method}/{meta.source} at {meta.tolerance}."
    )

with st.expander("The legs behind this bar, newest first"):
    if len(pieces) < 2:
        st.write("Nothing confirmed yet.")
    else:
        st.dataframe(
            {
                "from": [piece.start_bar for piece in reversed(pieces)],
                "to": [piece.end_bar for piece in reversed(pieces)],
                "bars": [piece.bars for piece in reversed(pieces)],
                "ticks": [round(piece.change / ticksize) for piece in reversed(pieces)],
                "settled": ["no" if piece.provisional else "yes" for piece in reversed(pieces)],
            },
            width="stretch",
            hide_index=True,
        )

with st.expander("What this dataset costs you in lag"):
    summary = replay.lag_summary(data)
    if summary.empty:
        st.write("Nothing has settled - is the history longer than the window?")
    else:
        legs = replay.legs(data)
        median_leg = legs["bars"].median() if not legs.empty else float("nan")

        st.write(
            f"Turns settle **{summary['lag_median']:.0f} bars** after they happen "
            f"(p90 {summary['lag_p90']:.0f}), and the median leg runs "
            f"**{median_leg:.0f} bars**."
        )
        st.write(
            f"So a turn is trustworthy about **{summary['lag_median'] / max(median_leg, 1):.0%}** "
            "of the way through the leg that follows it. Under a third is room to work with; "
            "over a half means the read arrives after the move."
        )
        st.caption(
            "Measured over the whole file, which is the one thing here that looks "
            "forward. It is a property of the dataset, never a feature."
        )


# --- the score, when the gym is on -------------------------------------------

if trading:
    closed = kept_trades() + [one for one in session.trades if one not in kept_trades()]

    st.divider()

    if not closed:
        st.info(
            "No closed trades yet. **B** opens a long, **S** a short, **→** moves one bar on. "
            "Everything else on this page still works - the whole point is to read the "
            "structure before you press anything."
        )
    else:
        overall = scoring.report(closed)
        form = scoring.rolling(closed)

        st.subheader(overall.score.tier)

        if not overall.score.ranked:
            st.caption(
                f"{overall.score.trades} of {scoring.RANK_AFTER} trades. A number appears at "
                f"{scoring.RANK_AFTER}; it takes a few hundred before a tier means much, which "
                "is why the tier is read off the bottom of the error bar rather than off the "
                "number itself."
            )
        else:
            st.caption(
                f"Sharpe {overall.score.sharpe:+.3f}, 95% band {overall.score.low:+.3f} to "
                f"{overall.score.high:+.3f}. The tier is the bottom of that band - on the "
                f"number alone it would read **{overall.score.tier_if_believed}**."
            )

        one, two, three, four, five = st.columns(5)
        one.metric("Trades", overall.trades,
                   f"{overall.forced} forced" if overall.forced else None)
        two.metric("Win rate", f"{overall.win_rate:.0%}",
                   f"{overall.wins}W / {overall.losses}L"
                   + (f" / {overall.scratches}=" if overall.scratches else ""))
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
                   help="Of the move a winner offered, how much was taken. A good entry with "
                        "a bad exit shows up here and nowhere else.")

        curve, current = st.columns([2, 1])

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

        with current:
            st.caption("Current form - last 50")
            st.metric("Sharpe", f"{form.score.sharpe:+.3f}" if form.score.sharpe else "-",
                      f"{form.trades} trades")
            st.caption("No tier here on purpose: at fifty trades the error bar is wider "
                       "than the whole ladder.")

        for title, attribute in (("By direction", "direction"), ("By timeframe", "timeframe")):
            with st.expander(title):
                st.dataframe(
                    [
                        {
                            "": label,
                            "trades": part.trades,
                            "win rate": f"{part.win_rate:.0%}",
                            "expectancy (R)": f"{part.expectancy:+.3f}",
                            "total (R)": f"{part.total_r:+.1f}",
                            "Sharpe": f"{part.score.sharpe:+.3f}" if part.score.sharpe else "-",
                        }
                        for label, part in scoring.by(closed, attribute).items()
                    ],
                    hide_index=True, width="stretch",
                )

    with st.expander("History"):
        st.caption(
            "Community Cloud forgets its filesystem on every redeploy, so this lives in the "
            "browser session and leaves as a file. Download before you close the tab."
        )

        save, bring, wipe = st.columns(3)

        save.download_button(
            "Download", width="stretch",
            data=json.dumps([one.as_dict() for one in closed], indent=1),
            file_name="gym-history.json", mime="application/json",
            disabled=not closed,
        )

        brought = bring.file_uploader("Load a history", type="json", label_visibility="collapsed")
        if brought is not None:
            loaded = [gym.Trade.from_dict(row) for row in json.load(brought)]
            known = {one.session for one in kept_trades()}
            st.session_state.kept = kept_trades() + [
                one for one in loaded if one.session not in known
            ]
            st.success(f"{len(loaded)} trades read in.")

        if wipe.button("Forget everything", width="stretch"):
            st.session_state.kept = []
            st.session_state.pop("session", None)
            st.rerun()
