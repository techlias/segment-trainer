"""The tutorial, inside the app.

The two documents in ``doc/`` are ordinary standalone HTML pages - they open in
any browser, and that is how they are meant to be read. This page serves them
from inside Streamlit as well, so a deployed copy carries its own instructions
and nobody has to be sent a file.

They are read from ``doc/`` rather than copied here. One source, so a fix to the
tutorial is a fix everywhere it is read.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

DOC = Path(__file__).parent.parent / "doc"

LANGUAGES = {
    "English": "reading-the-chart.html",
    "Español": "reading-the-chart.es.html",
}

#: The two documents link to each other by filename, which resolves in a folder
#: and not inside a frame. The radio below does that job here, so the links are
#: flattened to the words they were wrapping rather than left to dead-end. The
#: downloaded file keeps them, because there they work.
CROSS_LINK = re.compile(r'<a href="reading-the-chart[^"]*">([^<]*)</a>')


st.set_page_config(page_title="Tutorial · Segments", layout="wide")


@st.cache_data(show_spinner=False)
def read(name: str) -> str:
    return (DOC / name).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def framed(name: str) -> Path:
    """The page with its cross-links flattened, written where a frame can load
    it from. Cached, so it is written once per language per session."""
    folder = Path(tempfile.gettempdir()) / "segments-doc"
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / name
    path.write_text(CROSS_LINK.sub(r"\1", read(name)), encoding="utf-8")

    return path


st.markdown("#### Reading the Structure")
st.caption(
    "How to read the viewer: every mark on the chart, the vocabulary, the six states, "
    "and what the protective level means."
)

choice = st.radio("Language", list(LANGUAGES), horizontal=True, label_visibility="collapsed")
name = LANGUAGES[choice]

st.iframe(framed(name), height="content")

st.download_button(
    "Download this page",
    data=read(name),
    file_name=name,
    mime="text/html",
    help="The standalone file, cross-links intact - it opens in any browser, offline.",
)
