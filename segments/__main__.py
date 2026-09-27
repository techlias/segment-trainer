"""The command line: say what a dataset holds, and draw it.

    python -m segments report  <stem or folder>
    python -m segments plot    <stem> --bar 4000 --history 150 --reveal 30
    python -m segments sheet   <stem> --count 8
    python -m segments list    <folder>
    python -m segments verify  <stem>
    python -m segments refit   <stem> --tolerance 0.5 --out <folder>

Every command takes either file of an export, their shared stem, or a folder
holding exactly one export.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

from . import dataset as dataset_module
from . import fit, plot, replay


def _resolve(target: str) -> Path:
    """The stem of the export named by a file, a stem, or a folder."""
    path = Path(target)

    if path.is_dir():
        found = dataset_module.find(path)
        if not found:
            raise SystemExit(f"no exports in {path} - looked for *-bars.csv")
        if len(found) > 1:
            names = "\n  ".join(candidate.name for candidate in found)
            raise SystemExit(f"{len(found)} exports in {path}, name one:\n  {names}")
        return found[0]

    return path


def _load(target: str):
    data = dataset_module.load(_resolve(target))

    problems = data.check()
    if problems:
        print("PROBLEMS WITH THIS DATASET", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(file=sys.stderr)

    return data


def report(args) -> None:
    data = _load(args.target)
    bars = data.bars

    print(data.meta)
    print(
        f"{len(bars)} bars, {data.first_bar} to {data.last_bar}, "
        f"{bars['time'].iloc[0]:%Y-%m-%d %H:%M} to {bars['time'].iloc[-1]:%Y-%m-%d %H:%M}"
    )
    print(f"{len(data.events)} pivot events, {int((data.events['event'].str.strip() == '-').sum())} of them withdrawals")
    print(f"mean ATR {bars['atr'].mean():.2f} ({bars['atr'].mean() / (data.meta.ticksize or 1):.0f} ticks), "
          f"{int(bars['new_session'].sum())} sessions")
    print()

    summary = replay.lag_summary(data)
    if summary.empty:
        print("no settled pivots - is the export longer than the window?")
        return

    print("CONFIRMATION LAG (bars between a turn and it being trustworthy)")
    print(f"  pivots that stuck      {summary['pivots']:.0f}")
    print(f"  pivots withdrawn again {summary['withdrawn']:.0f}")
    print(f"  first visible after    {summary['first_lag_median']:.0f} bars (median)")
    print(f"  settled after          {summary['lag_median']:.0f} bars (median), "
          f"{summary['lag_p90']:.0f} at p90, {summary['lag_max']:.0f} worst")
    print(f"  changed its mind       {summary['flips_mean']:.2f} times per pivot on average")
    print()
    print("  Read the median as the delay this dataset imposes: half the turns were")
    print("  trustworthy that many bars after they happened. Every rule downstream")
    print("  inherits it, and no amount of cleverness gets it back.")
    print()

    pieces = replay.legs(data)
    if not pieces.empty:
        print(f"LEGS ({len(pieces)} confirmed, offline view)")
        with pd.option_context("display.width", 120):
            print(
                pieces[["bars", "ticks", "amplitude", "slope"]]
                .describe()
                .loc[["count", "mean", "50%", "max"]]
                .round(2)
                .to_string()
            )
        rising = int(pieces["up"].sum())
        print(f"  {rising} up, {len(pieces) - rising} down")


def plot_one(args) -> None:
    data = _load(args.target)
    bar = args.bar if args.bar is not None else data.last_bar

    out = Path(args.out) if args.out else Path(f"segments-{bar}.png")
    plot.save(data, bar, out, history=args.history, reveal=args.reveal, sticks=args.bars)

    print(f"{out}  bar {bar}, {args.history} bars of history, {args.reveal} revealed")


def sheet(args) -> None:
    data = _load(args.target)

    # Away from both ends: the first bars have no window behind them and the last
    # have nothing to reveal.
    low = data.first_bar + max(args.history, data.meta.window)
    high = data.last_bar - args.reveal
    if high <= low:
        raise SystemExit("not enough bars for a contact sheet of this history and reveal")

    random.seed(args.seed)
    bars = sorted(random.sample(range(low, high), min(args.count, high - low)))

    out = Path(args.out) if args.out else Path("segments-sheet.png")
    plot.contact_sheet(data, bars, out, history=args.history, reveal=args.reveal, sticks=args.bars)

    print(f"{out}  bars {bars}")


def verify(args) -> None:
    data = _load(args.target)
    diff = fit.verify(data)

    if diff.empty:
        print(f"the port agrees with NinjaTrader: {len(data.events)} events, none in dispute")
        return

    theirs = int((diff["side"] == "export").sum())
    mine = int((diff["side"] == "python").sum())

    print(f"DISAGREEMENT: {theirs} events only NinjaTrader produced, {mine} only Python did")
    print(f"out of {len(data.events)} exported")
    print()
    print(diff.head(20).to_string(index=False))

    if len(diff) > 20:
        print(f"... and {len(diff) - 20} more")

    print()
    print("The first bar listed is where to look: put that bar on the chart with")
    print("PriceSegments at the same settings and see which of the two is right.")

    raise SystemExit(1)


def _stem(meta) -> str:
    """The exporter's own naming, so a re-cut file sits beside the original and
    says what it is."""
    name = (
        f"{meta.instrument}-{meta.period}-{meta.method}-{meta.source}"
        f"-t{meta.tolerance:g}-m{meta.minbars}-w{meta.window}"
    )

    for bad in '<>:"/\|?*':
        name = name.replace(bad, "_")

    return name.replace(" ", "_")


def refit(args) -> None:
    data = _load(args.target)

    cut = fit.refit(
        data,
        method=args.method,
        source=args.source,
        tolerance=args.tolerance,
        min_bars=args.min_bars,
        window=args.window,
        atr_period=args.atr_period,
    )

    folder = Path(args.out) if args.out else Path(data.path)
    written = cut.write(folder / _stem(cut.meta))

    churn = len(cut.events) / max(len(cut.bars), 1)
    print(f"{written}")
    print(f"{cut.meta}")
    print(f"{len(cut.events)} events over {len(cut.bars)} bars ({churn:.2f} per bar)")

    summary = replay.lag_summary(cut)
    if not summary.empty:
        print(f"settled after {summary['lag_median']:.0f} bars (median), "
              f"{summary['pivots']:.0f} pivots stuck, {summary['withdrawn']:.0f} withdrawn")


def list_exports(args) -> None:
    found = dataset_module.find(args.folder)
    if not found:
        print(f"no exports in {args.folder}")
        return

    for stem in found:
        try:
            data = dataset_module.load(stem)
            print(f"{stem.name}\n    {data.meta}\n    {len(data.bars)} bars, {len(data.events)} events")
        except Exception as problem:  # a half-written export should not stop the listing
            print(f"{stem.name}\n    unreadable: {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="segments", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def shared(sub):
        sub.add_argument("target", help="either CSV of an export, their shared stem, or a folder holding one export")
        return sub

    said = commands.add_parser("report", help="what the dataset holds, and the confirmation lag")
    shared(said).set_defaults(run=report)

    drawn = shared(commands.add_parser("plot", help="the view as at one bar, as a PNG"))
    drawn.add_argument("--bar", type=int, default=None, help="which bar to look from (default: the last)")
    drawn.add_argument("--history", type=int, default=150, help="bars of context to show")
    drawn.add_argument("--reveal", type=int, default=0, help="bars past it to draw in grey - the answer")
    drawn.add_argument("--bars", action="store_true", help="high-low sticks behind the segments")
    drawn.add_argument("--out", default=None)
    drawn.set_defaults(run=plot_one)

    many = shared(commands.add_parser("sheet", help="several random bars on one page"))
    many.add_argument("--count", type=int, default=8)
    many.add_argument("--history", type=int, default=150)
    many.add_argument("--reveal", type=int, default=0)
    many.add_argument("--bars", action="store_true")
    many.add_argument("--seed", type=int, default=0)
    many.add_argument("--out", default=None)
    many.set_defaults(run=sheet)

    checked = shared(commands.add_parser("verify", help="re-cut the bars in Python and diff against the export"))
    checked.set_defaults(run=verify)

    cut = shared(commands.add_parser("refit", help="re-cut the same bars at other settings, as a new export"))
    cut.add_argument("--method", default=None, choices=list(fit.METHODS))
    cut.add_argument("--source", default=None, choices=["Close", "BodyCentre", "Median", "Typical"])
    cut.add_argument("--tolerance", type=float, default=None, help="in ATRs - the level of detail")
    cut.add_argument("--min-bars", type=int, default=None, dest="min_bars")
    cut.add_argument("--window", type=int, default=None)
    cut.add_argument("--atr-period", type=int, default=None, dest="atr_period",
                     help="recompute the ATR, only correct if the export starts at bar 0")
    cut.add_argument("--out", default=None, help="folder for the new pair (default: beside the original)")
    cut.set_defaults(run=refit)

    listed = commands.add_parser("list", help="every export in a folder")
    listed.add_argument("folder")
    listed.set_defaults(run=list_exports)

    args = parser.parse_args(argv)
    args.run(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
