"""HTML reporting: self-contained, balanced, and honest about provenance."""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from flowalpha.data.provenance import REAL, SYNTHETIC, UNAVAILABLE, Provenance
from flowalpha.reporting.report import (
    STANDING_LIMITATIONS,
    ReportDocument,
    ReportError,
    bar_figure,
    embed_figure,
    esc,
    experiments_html,
    figure_to_data_uri,
    fmt,
    frame_to_table,
    kv_list,
    line_figure,
    validate_html,
    verdict_html,
)

NOW = _dt.datetime(2026, 7, 30, 9, 15, 0)


# --- validate_html ---------------------------------------------------------

def test_balanced_html_passes():
    validate_html("<div><p>hello</p></div>")


def test_void_tags_do_not_need_closing():
    validate_html('<div><br><hr><img src="data:image/png;base64,AA"></div>')


def test_self_closing_tags_are_accepted():
    validate_html("<div><span/></div>")


def test_unclosed_tag_is_rejected():
    with pytest.raises(ReportError, match="unclosed"):
        validate_html("<div><p>hello")


def test_mismatched_tag_is_rejected():
    with pytest.raises(ReportError, match="closes while"):
        validate_html("<div><p>hello</div></p>")


def test_stray_closing_tag_is_rejected():
    with pytest.raises(ReportError, match="nothing open"):
        validate_html("</div>")


@pytest.mark.parametrize(
    "html",
    [
        '<img src="https://example.com/a.png">',
        '<img src="http://example.com/a.png">',
        '<link href="//cdn.example.com/x.css">',
        '<script src="https://cdn.example.com/x.js"></script>',
    ],
)
def test_external_references_are_rejected(html):
    """A report that depends on a CDN renders on the machine that made it and is blank
    for everyone else."""
    with pytest.raises(ReportError, match="not self-contained"):
        validate_html(f"<div>{html}</div>")


def test_data_uris_and_anchors_are_allowed():
    validate_html('<div><img src="data:image/png;base64,AAAA"><a href="#top">t</a></div>')


# --- formatting ------------------------------------------------------------

def test_esc_escapes_and_folds_to_ascii():
    assert esc("<b>") == "&lt;b&gt;"
    assert esc("Rs 1 – 2") == "Rs 1 - 2"
    esc("你好").encode("ascii")


def test_fmt_handles_every_type():
    assert fmt(None) == "n/a"
    assert fmt(True) == "yes"
    assert fmt(1234) == "1,234"
    assert fmt(1.23456, 2) == "1.23"
    assert fmt(float("nan")) == "n/a"
    assert fmt(float("inf")) == "n/a"
    assert fmt("text") == "text"


# --- tables ----------------------------------------------------------------

def test_frame_to_table_renders_rows():
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    html = frame_to_table(frame)
    assert "<table>" in html and "</table>" in html
    validate_html(f"<div>{html}</div>")
    assert html.count("<tr>") == 3  # header + 2 rows


def test_frame_to_table_empty_says_so():
    assert "no rows" in frame_to_table(pl.DataFrame({"a": []}))


def test_frame_to_table_escapes_cell_content():
    frame = pl.DataFrame({"a": ["<script>alert(1)</script>"]})
    html = frame_to_table(frame)
    assert "<script>" not in html
    validate_html(f"<div>{html}</div>")


def test_frame_to_table_truncation_is_stated():
    """A silently truncated table reads as a complete one."""
    frame = pl.DataFrame({"a": list(range(100))})
    html = frame_to_table(frame, max_rows=10)
    assert "90 rows omitted" in html
    assert "first 10 of 100" in html


def test_frame_to_table_respects_per_column_digits():
    frame = pl.DataFrame({"x": [1.23456]})
    assert "1.23" in frame_to_table(frame, digits={"x": 2})


def test_kv_list_renders_pairs():
    html = kv_list([("a", 1), ("b", "text")])
    validate_html(f"<div>{html}</div>")
    assert "<dt>a</dt>" in html


# --- figures ---------------------------------------------------------------

def test_figures_embed_as_data_uris():
    fig = line_figure({"s": ([1, 2, 3], [1.0, 2.0, 1.5])}, title="t", provenance_tag="[X]")
    uri = figure_to_data_uri(fig)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 1000


def test_embedded_figure_is_self_contained():
    fig = bar_figure(["a", "b"], [1.0, -1.0], title="t", provenance_tag="[X]")
    html = embed_figure(fig, "caption")
    validate_html(f"<div>{html}</div>")
    assert 'src="data:image/png;base64,' in html
    assert "caption" in html


def test_figure_titles_carry_the_provenance_tag():
    """A copied-out image must still say where its data came from."""
    prov = Provenance().mark("prices", SYNTHETIC, "gen")
    fig = line_figure({"s": ([1, 2], [1.0, 2.0])}, title="Equity", provenance_tag=prov.tag())
    assert "SYNTHETIC" in fig.axes[0].get_title()
    figure_to_data_uri(fig)  # close it


# --- documents -------------------------------------------------------------

def _doc(prov: Provenance) -> ReportDocument:
    return ReportDocument(title="T", provenance=prov, generated_at=NOW, subtitle="sub")


def test_fully_real_run_shows_no_warning_banner():
    prov = Provenance().mark("prices", REAL, "yahoo").mark("flows_daily", REAL, "nse")
    html = _doc(prov).render()
    validate_html(html)
    assert "banner ok" in html
    assert "banner warn" not in html
    assert "LIVE DATA" in html


def test_hybrid_provenance_surfaces_in_the_html():
    """The failure mode: a report describing generated numbers as market data."""
    prov = Provenance().mark("prices", REAL, "yahoo").mark("flows_daily", SYNTHETIC, "gen")
    html = _doc(prov).render()
    validate_html(html)
    assert "banner warn" in html
    assert "HYBRID" in html
    assert "SYNTHETIC" in html
    assert "flows_daily" in html


def test_unrecorded_provenance_surfaces_in_the_html():
    html = _doc(Provenance.unrecorded()).render()
    assert "PROVENANCE UNRECORDED" in html
    assert "banner warn" in html


def test_partial_provenance_surfaces_in_the_html():
    prov = Provenance().mark("prices", REAL, "y").mark("delivery_pct", UNAVAILABLE, "y")
    html = _doc(prov).render()
    assert "PARTIAL" in html
    assert "delivery_pct" in html


def test_provenance_table_lists_every_dataset():
    prov = (
        Provenance()
        .mark("prices", REAL, "yahoo", note="500 names")
        .mark("flows_daily", REAL, "nse", note="contracts not rupees")
    )
    html = _doc(prov).render()
    assert "500 names" in html
    assert "contracts not rupees" in html


def test_limitations_always_include_the_standing_caveats():
    """Generated from the record, so the section cannot fall out of date."""
    html = _doc(Provenance().mark("prices", REAL, "y")).render()
    for caveat in STANDING_LIMITATIONS:
        assert esc(caveat) in html


def test_limitations_include_provenance_caveats():
    prov = Provenance().mark("prices", REAL, "y").add_caveat("a specific caveat")
    assert "a specific caveat" in _doc(prov).render()


def test_limitations_are_deduplicated():
    prov = Provenance().mark("prices", REAL, "y")
    prov.add_caveat(STANDING_LIMITATIONS[0])
    doc = _doc(prov)
    limits = doc.limitations()
    assert len(limits) == len(set(limits))


def test_extra_limitations_are_included():
    doc = _doc(Provenance().mark("prices", REAL, "y"))
    doc.extra_limitations.append("portfolio warning")
    assert "portfolio warning" in doc.render()


def test_sections_render_in_order():
    doc = _doc(Provenance().mark("prices", REAL, "y"))
    doc.add("First", "<p>one</p>").add("Second", "<p>two</p>")
    html = doc.render()
    assert html.index("First") < html.index("Second")


def test_render_validates_and_raises_on_bad_section_html():
    doc = _doc(Provenance().mark("prices", REAL, "y"))
    doc.add("Bad", "<div><p>unclosed</div>")
    with pytest.raises(ReportError):
        doc.render()


def test_render_rejects_an_external_reference_in_a_section():
    doc = _doc(Provenance().mark("prices", REAL, "y"))
    doc.add("Bad", '<img src="https://example.com/x.png">')
    with pytest.raises(ReportError, match="not self-contained"):
        doc.render()


def test_write_produces_a_validated_file(tmp_path):
    doc = _doc(Provenance().mark("prices", REAL, "y"))
    doc.add("Figures", embed_figure(
        line_figure({"s": ([1, 2], [1.0, 2.0])}, title="t"), "cap"
    ))
    path = doc.write(tmp_path / "out" / "report.html")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    validate_html(text)
    assert 'src="http' not in text
    assert 'href="http' not in text
    assert "data:image/png;base64," in text


def test_report_is_ascii_only(tmp_path):
    prov = Provenance().mark("prices", REAL, "NSE – archives", note="₹ crore")
    path = _doc(prov).write(tmp_path / "r.html")
    path.read_text(encoding="utf-8").encode("ascii")


def test_generated_timestamp_appears():
    html = _doc(Provenance().mark("prices", REAL, "y")).render()
    assert "2026-07-30T09:15:00" in html


# --- experiments -----------------------------------------------------------

def _experiment(verdict: str, diff: float = 0.01) -> dict:
    return {
        "factor": "momentum", "regime_column": "fii_regime",
        "favourable_bucket": "high", "unfavourable_bucket": "low",
        "rationale": "flow continuation", "horizon": 5,
        "ic_favourable": 0.02, "ic_unfavourable": 0.01,
        "n_favourable": 500, "n_unfavourable": 500,
        "ic_difference": diff, "t_stat_difference": 2.0,
        "gate_t_threshold": 1.5, "supported": verdict.startswith("SUPPORTED"),
        "verdict": verdict,
    }


def test_verdict_html_colour_codes_without_hiding_the_text():
    assert "verdict sup" in verdict_html("SUPPORTED")
    assert "verdict ref" in verdict_html("REFUTED (opposite direction)")
    assert "verdict non" in verdict_html("NOT SUPPORTED (nope)")
    assert "NOT SUPPORTED" in verdict_html("NOT SUPPORTED (nope)")


def test_experiments_html_states_the_declared_direction():
    html = experiments_html([_experiment("SUPPORTED")])
    validate_html(f"<div>{html}</div>")
    assert "Pre-declared direction" in html
    assert "flow continuation" in html
    assert "SUPPORTED" in html


def test_experiments_html_shows_a_refutation_plainly():
    html = experiments_html([_experiment("REFUTED (difference significant in the OPPOSITE direction)", -0.02)])
    assert "REFUTED" in html
    assert "OPPOSITE" in html


def test_experiments_html_with_no_experiments():
    html = experiments_html([])
    validate_html(f"<div>{html}</div>")
    assert "no rows" in html
