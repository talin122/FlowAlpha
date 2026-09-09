"""Universe ingestion: the published constituent list is not the tradable universe.

NSE's NIFTY 500 file is a *current* snapshot and it changes underneath a running study.
On 2026-09-09 it swapped HEG and HFCL out and introduced ``DUMMYHEG`` -- a placeholder
symbol created for a corporate action, not a tradable instrument. Nothing broke, because
no price vendor carries it and the panel simply omitted it. But it reached
``universe_tickers.csv`` and ``sectors.csv`` first, and a name that disappears three
stages downstream is a name nobody ever decided to drop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_universe_tickers import is_placeholder, parse_constituents  # noqa: E402

HEADER = "Company Name,Industry,Symbol,Series,ISIN Code\n"


def _csv(*rows: str) -> str:
    return HEADER + "".join(rows)


def test_a_normal_list_parses_with_no_placeholders():
    members, dropped = parse_constituents(
        _csv("Reliance,Energy,RELIANCE,EQ,INE002A01018\n",
             "Infosys,IT,INFY,EQ,INE009A01021\n")
    )
    assert members["symbol"].to_list() == ["INFY", "RELIANCE"]
    assert dropped == []


def test_the_dummy_placeholder_is_excluded_and_reported():
    """The exact symbol that entered this project's universe."""
    members, dropped = parse_constituents(
        _csv("HEG Ltd,Capital Goods,DUMMYHEG,EQ,INE545A01024\n",
             "Infosys,IT,INFY,EQ,INE009A01021\n")
    )
    assert members["symbol"].to_list() == ["INFY"]
    assert dropped == ["DUMMYHEG"]


def test_placeholders_are_returned_not_silently_swallowed():
    """Returned rather than discarded so the caller can print them. A universe that
    quietly shrinks makes two runs incomparable without anyone noticing."""
    _, dropped = parse_constituents(
        _csv("A,X,DUMMYA,EQ,I1\n", "B,X,DUMMYB,EQ,I2\n", "C,X,REAL,EQ,I3\n")
    )
    assert dropped == ["DUMMYA", "DUMMYB"]


def test_a_symbol_merely_containing_dummy_is_kept():
    """The rule is a prefix, not a substring: a real listing whose name happens to
    contain the letters must not be dropped."""
    members, dropped = parse_constituents(_csv("Notadummy,X,NOTDUMMY,EQ,I1\n"))
    assert members["symbol"].to_list() == ["NOTDUMMY"]
    assert dropped == []


@pytest.mark.parametrize("sym,expected", [
    ("DUMMYHEG", True), ("dummyheg", True), ("DUMMY", True),
    ("HEG", False), ("NOTDUMMY", False), ("MYDUMMY", False),
])
def test_is_placeholder_cases(sym, expected):
    assert is_placeholder(sym) is expected


def test_a_list_of_only_placeholders_raises_rather_than_returning_empty():
    """An empty universe would silently collapse every cross-sectional rank downstream.
    Better to fail at ingestion than to rank two names against each other."""
    with pytest.raises(ValueError, match="zero tradable"):
        parse_constituents(_csv("A,X,DUMMYA,EQ,I1\n"))


def test_non_eq_series_still_filtered():
    members, _ = parse_constituents(
        _csv("A,X,AAA,EQ,I1\n", "B,X,BBB,BE,I2\n")
    )
    assert members["symbol"].to_list() == ["AAA"]


def test_an_html_error_page_is_rejected():
    with pytest.raises(ValueError, match="HTML error page"):
        parse_constituents("<!DOCTYPE html><html><body>error</body></html>")
