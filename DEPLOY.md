# Deploying the trainer

The viewer is a Streamlit app: a Python process holding a WebSocket per viewer.
That rules out static hosting — Firebase Spark, GitHub Pages and the like can
serve `doc/` but cannot run this. Streamlit Community Cloud runs it for free.

## Where the code lives

This is the `segment-trainer` repository: the Python half of the
`NT Bollinger Strategy` project, split out so Streamlit sees a clean root.

The NinjaScript side — `SegmentFit.cs`, `SegmentExport.cs`, `PriceSegments.cs` —
stays in the main repo and never comes here. The two are kept honest by
`segments.fit.verify`, which re-cuts an export's own bars and diffs the result
against what NinjaTrader wrote. **Run it after any change to either side.**

To pull later Python work across from the main repo:

```bash
# from the main repo
git subtree push --prefix="Segment strategy" trainer main
```

## Deploying

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub
   account that owns this repo. Because the repo is private, grant the
   **private repositories** scope when GitHub asks — without it the repo will
   not appear in the list.
2. **Create app** → **Deploy a public app from GitHub**, then:

   | field | value |
   |---|---|
   | Repository | `<you>/segment-trainer` |
   | Branch | `main` |
   | Main file path | `viewer.py` |
   | Python version | 3.12 or 3.13 |

3. Deploy. First build takes two or three minutes while it installs pandas and
   matplotlib; later ones are seconds.

## Locking it down

A Community Cloud app is reachable by anyone with the link unless you say
otherwise. In the app's **Settings → Sharing**, set it to private and add the
email addresses that may open it. They sign in with Google or GitHub; nobody
else gets past the door.

Worth doing even though nothing here is secret: the app carries an export of
MNQ bars, and exchange data has rules about redistribution that a public URL is
a poor place to test.

## The data

`data/` holds one small export so the deployed app has something to open. It is
the only export in the repo — everything else is ignored by `.gitignore`.

To look at a different history from another machine, **upload it**: the sidebar
takes the two CSVs `SegmentExport` writes, reads them exactly as a local pair,
and keeps nothing after the session. No redeploy, no commit.

To change what the app opens *by default*, drop a pair into `data/` and push.
Keep it small — the whole repo is cloned on every deploy, and a month of
1-minute bars is tens of megabytes.

## What it costs

Free, with the limits that implies: about 1 GB of memory, one app awake at a
time on the free tier, and the app sleeps after a week without visitors — the
first visit afterwards wakes it in a few seconds.

Memory is the one to watch. A 1,839-bar export uses a few tens of megabytes; a
year of 1-minute bars would not fit, and the sweep behind a re-cut is the part
that would go first. If it ever gets tight, export a coarser timeframe rather
than a longer window.

## Running it locally

Unchanged, and still the faster way to work:

```bash
.venv/Scripts/streamlit run viewer.py
```

Locally the app also looks in the NinjaTrader `segment-export` folder, so on the
machine that does the exporting every dataset shows up in the sidebar without
being copied anywhere.

## If the build fails

- **`ModuleNotFoundError`** — `requirements.txt` is at the repo root and must
  name the package. Check the build log, not the app.
- **Nothing installs, or the wrong things do** — both `requirements.txt` and
  `pyproject.toml` sit at the root. Community Cloud should use the first, since
  the second has no `[tool.poetry]` section and is only there for `pip install
  -e .` locally. If a build ever proves otherwise, delete `pyproject.toml` from
  this repo — it belongs to the main one.
- **The repo is not listed** — the private-repository scope was not granted.
  Revoke Streamlit under GitHub → Settings → Applications and sign in again.
- **The app opens with "No export found"** — `data/` did not make it into the
  repo. `.gitignore` excludes `*-bars.csv` everywhere and then lets `data/`
  back in; check with `git check-ignore -v data/*.csv`.
