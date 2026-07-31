"""Self-contained HTML reporting.

Every report is a **single file with no external references**: no CDN stylesheet, no
web font, no remote image. Figures are rendered with matplotlib and embedded as base64
``data:`` URIs. This is not aesthetic preference -- a research artefact that renders
differently (or not at all) depending on network access is not a record of anything.
:func:`validate_html` enforces it, and the tests assert the enforcement works.

Two report types:

``research_report``
    The full study: provenance, data quality, IC tables, conditional IC, the declared
    experiments and their verdicts, backtests gross and net, DSR against the cumulative
    trial count, and PBO.

``daily_digest``
    Decision support for one date: today's regime, the gate states, the model portfolio,
    and the trade list with per-trade cost in basis points.

Both auto-surface a limitations section built from the provenance caveats plus the
standing caveats of this data (survivorship, dividend approximation, retail proxy,
multiple testing). The section is generated, not written, so it cannot fall out of date
with the record.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from ..data.provenance import (
    CAVEAT_DIVIDEND_ADJUSTMENT,
    CAVEAT_MULTIPLE_TESTING,
    CAVEAT_RETAIL_PROXY,
    CAVEAT_SURVIVORSHIP,
    Provenance,
    to_ascii,
)

#: Standing limitations of this project's data, always shown.
STANDING_LIMITATIONS: tuple[str, ...] = (
    CAVEAT_SURVIVORSHIP,
    CAVEAT_DIVIDEND_ADJUSTMENT,
    CAVEAT_RETAIL_PROXY,
    CAVEAT_MULTIPLE_TESTING,
)

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5; margin: 0; padding: 0 1.25rem 4rem;
  background: #fbfbfc; color: #16181d;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 1.6rem 0 0.3rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.5rem; padding-bottom: 0.3rem;
     border-bottom: 1px solid #d8dbe0; }
h3 { font-size: 1rem; margin: 1.3rem 0 0.4rem; }
p, li { font-size: 0.92rem; }
.meta { color: #5a6069; font-size: 0.82rem; margin: 0 0 1rem; }
.banner { padding: 0.7rem 0.9rem; border-radius: 6px; margin: 1rem 0;
          font-size: 0.88rem; font-weight: 600; }
.banner.warn { background: #fff4e0; border: 1px solid #e0a34a; color: #6b4300; }
.banner.ok   { background: #eaf6ec; border: 1px solid #63a86e; color: #1f4d28; }
.tablewrap { overflow-x: auto; margin: 0.6rem 0 1.2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem;
        font-variant-numeric: tabular-nums; }
th, td { padding: 0.35rem 0.55rem; text-align: right; border-bottom: 1px solid #e4e6ea;
         white-space: nowrap; }
th { background: #f1f2f5; font-weight: 600; text-align: right; position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: #f6f7f9; }
.fig { margin: 0.8rem 0 1.4rem; }
.fig img { max-width: 100%; height: auto; border: 1px solid #e0e2e6; border-radius: 4px;
           background: #fff; }
.note { font-size: 0.8rem; color: #5a6069; margin: 0.2rem 0 1rem; }
.verdict { font-weight: 700; }
.verdict.sup { color: #1f7a33; }
.verdict.ref { color: #a32020; }
.verdict.non { color: #7a5a00; }
.limits { background: #fff; border: 1px solid #d8dbe0; border-radius: 6px;
          padding: 0.9rem 1.1rem; }
.limits li { margin-bottom: 0.55rem; }
code { background: #f1f2f5; padding: 0.05rem 0.3rem; border-radius: 3px; font-size: 0.85em; }
.kv { display: grid; grid-template-columns: minmax(140px, max-content) 1fr;
      gap: 0.2rem 1rem; font-size: 0.86rem; margin: 0.5rem 0 1rem; }
.kv dt { color: #5a6069; }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8eb; }
  h2 { border-color: #2c3037; }
  .meta, .note, .kv dt { color: #9aa1ab; }
  th { background: #1e2126; }
  th, td { border-color: #2c3037; }
  tbody tr:hover { background: #1a1d22; }
  .limits { background: #1a1d22; border-color: #2c3037; }
  .banner.warn { background: #3a2a10; border-color: #7a5a1f; color: #f0d9a8; }
  .banner.ok { background: #16301c; border-color: #2f6b3c; color: #b9e0c2; }
  .fig img { background: #fff; border-color: #2c3037; }
  code { background: #22262c; }
}
"""


class ReportError(Exception):
    """Raised when a report fails its own structural validation."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "source"}
_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9]*)([^>]*?)(/?)\s*>")
_EXTERNAL_ATTR_RE = re.compile(r"""(?:src|href)\s*=\s*["'](https?:|//)""", re.I)


def validate_html(text: str) -> None:
    """Assert the document is tag-balanced and references nothing external.

    Raises :class:`ReportError` with a specific message. The external-reference check is
    the important half: a report that silently depends on a CDN will render correctly on
    the machine that made it and be blank for everyone else.
    """
    stack: list[str] = []
    for match in _TAG_RE.finditer(text):
        closing, tag, attrs, self_closing = match.groups()
        tag = tag.lower()
        if tag in _VOID_TAGS or self_closing == "/":
            continue
        if closing:
            if not stack:
                raise ReportError(f"unbalanced HTML: closing </{tag}> with nothing open")
            if stack[-1] != tag:
                raise ReportError(
                    f"unbalanced HTML: </{tag}> closes while <{stack[-1]}> is open"
                )
            stack.pop()
        else:
            stack.append(tag)
    if stack:
        raise ReportError(f"unbalanced HTML: unclosed tag(s) {stack}")

    external = _EXTERNAL_ATTR_RE.search(text)
    if external:
        snippet = text[max(0, external.start() - 40) : external.end() + 40]
        raise ReportError(
            f"report is not self-contained: external reference near ...{snippet}..."
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape(to_ascii(str(value)), quote=True)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:,.{digits}f}"


def frame_to_table(
    frame: pl.DataFrame,
    *,
    digits: Mapping[str, int] | None = None,
    default_digits: int = 4,
    max_rows: int | None = 500,
) -> str:
    """Render a polars frame as an HTML table.

    When ``max_rows`` truncates the output, the omission is stated in the caption. A
    silently truncated table reads as a complete one.
    """
    if frame.is_empty():
        return '<p class="note">(no rows)</p>'
    digits = dict(digits or {})
    shown = frame if max_rows is None else frame.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in shown.columns)
    body_rows = []
    for row in shown.iter_rows(named=True):
        cells = []
        for col in shown.columns:
            value = row[col]
            if isinstance(value, (_dt.date, _dt.datetime)):
                cells.append(f"<td>{esc(value)}</td>")
            elif isinstance(value, str) or value is None:
                cells.append(f"<td>{esc('' if value is None else value)}</td>")
            else:
                cells.append(f"<td>{fmt(value, digits.get(col, default_digits))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    caption = ""
    if max_rows is not None and frame.height > max_rows:
        caption = (
            f'<p class="note">Showing the first {max_rows:,} of {frame.height:,} rows; '
            f"{frame.height - max_rows:,} rows omitted.</p>"
        )
    return (
        '<div class="tablewrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
        + caption
    )


def kv_list(pairs: Sequence[tuple[str, object]]) -> str:
    items = "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(v) if isinstance(v, str) else fmt(v, 4)}</dd>"
        for k, v in pairs
    )
    return f'<dl class="kv">{items}</dl>'


def figure_to_data_uri(fig) -> str:
    """Render a matplotlib figure to a base64 PNG data URI and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def embed_figure(fig, caption: str = "") -> str:
    uri = figure_to_data_uri(fig)
    cap = f'<p class="note">{esc(caption)}</p>' if caption else ""
    return f'<div class="fig"><img alt="{esc(caption or "figure")}" src="{uri}">{cap}</div>'


def line_figure(
    series: Mapping[str, tuple[Sequence, Sequence]],
    *,
    title: str,
    ylabel: str = "",
    provenance_tag: str = "",
    figsize: tuple[float, float] = (9.0, 3.6),
):
    """A line chart. The provenance tag goes in the title so a copied-out image still
    carries its own data-origin label."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for label, (x, y) in series.items():
        ax.plot(x, y, linewidth=1.2, label=label)
    ax.set_title(to_ascii(f"{title} {provenance_tag}".strip()), fontsize=10)
    if ylabel:
        ax.set_ylabel(to_ascii(ylabel), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    if len(series) > 1:
        ax.legend(fontsize=8, frameon=False)
    fig.autofmt_xdate()
    return fig


def bar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str = "",
    provenance_tag: str = "",
    figsize: tuple[float, float] = (9.0, 3.4),
):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#2f6b3c" if v >= 0 else "#a32020" for v in values]
    ax.bar([to_ascii(l) for l in labels], values, color=colours)
    ax.axhline(0.0, color="#555", linewidth=0.8)
    ax.set_title(to_ascii(f"{title} {provenance_tag}".strip()), fontsize=10)
    if ylabel:
        ax.set_ylabel(to_ascii(ylabel), fontsize=9)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    return fig


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """One report section."""

    title: str
    body: str
    level: int = 2


@dataclass
class ReportDocument:
    """A report under construction."""

    title: str
    provenance: Provenance
    generated_at: _dt.datetime
    subtitle: str = ""
    sections: list[Section] = field(default_factory=list)
    extra_limitations: list[str] = field(default_factory=list)

    def add(self, title: str, body: str, *, level: int = 2) -> "ReportDocument":
        self.sections.append(Section(title=title, body=body, level=level))
        return self

    def limitations(self) -> list[str]:
        """Provenance caveats plus the standing ones, de-duplicated, order preserved."""
        out: list[str] = []
        for text in list(self.provenance.caveats) + list(STANDING_LIMITATIONS) + list(self.extra_limitations):
            clean = to_ascii(str(text)).strip()
            if clean and clean not in out:
                out.append(clean)
        return out

    def render(self) -> str:
        banner = self.provenance.banner()
        banner_html = (
            f'<div class="banner warn">{esc(banner)}</div>'
            if banner
            else '<div class="banner ok">All datasets recorded as real market data.</div>'
        )
        prov_rows = pl.DataFrame(
            {
                "dataset": sorted(self.provenance.datasets),
                "status": [self.provenance.datasets[d].status for d in sorted(self.provenance.datasets)],
                "source": [self.provenance.datasets[d].source for d in sorted(self.provenance.datasets)],
                "note": [self.provenance.datasets[d].note for d in sorted(self.provenance.datasets)],
            }
        ) if self.provenance.datasets else pl.DataFrame(
            schema={"dataset": pl.Utf8, "status": pl.Utf8, "source": pl.Utf8, "note": pl.Utf8}
        )

        parts = [
            f"<style>{_CSS}</style>",
            '<div class="wrap">',
            f"<h1>{esc(self.title)}</h1>",
        ]
        if self.subtitle:
            parts.append(f'<p class="meta">{esc(self.subtitle)}</p>')
        parts.append(
            f'<p class="meta">Generated {esc(self.generated_at.isoformat(timespec="seconds"))} '
            f"&middot; data provenance: <strong>{esc(self.provenance.label())}</strong></p>"
        )
        parts.append(banner_html)

        parts.append("<h2>Data provenance</h2>")
        parts.append(frame_to_table(prov_rows, max_rows=None))

        for section in self.sections:
            tag = f"h{max(2, min(4, section.level))}"
            parts.append(f"<{tag}>{esc(section.title)}</{tag}>")
            parts.append(section.body)

        parts.append("<h2>Limitations</h2>")
        parts.append(
            '<div class="limits"><ul>'
            + "".join(f"<li>{esc(item)}</li>" for item in self.limitations())
            + "</ul></div>"
        )
        parts.append("</div>")
        text = "\n".join(parts)
        validate_html(text)
        return text

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.render()
        path.write_text(text, encoding="utf-8")
        return path


def verdict_html(verdict: str) -> str:
    """Colour-code an experiment verdict without hiding what it says."""
    cls = "non"
    if verdict.startswith("SUPPORTED"):
        cls = "sup"
    elif verdict.startswith("REFUTED"):
        cls = "ref"
    return f'<span class="verdict {cls}">{esc(verdict)}</span>'


def experiments_html(results: Iterable) -> str:
    """Render declared experiments with their pre-stated direction and verdict."""
    blocks = []
    rows = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        rows.append(
            {
                "factor": d["factor"],
                "regime": d["regime_column"],
                "favourable": d["favourable_bucket"],
                "unfavourable": d["unfavourable_bucket"],
                "IC favourable": d["ic_favourable"],
                "IC unfavourable": d["ic_unfavourable"],
                "difference": d["ic_difference"],
                "t(diff, NW)": d["t_stat_difference"],
                "n fav": d["n_favourable"],
                "n unfav": d["n_unfavourable"],
            }
        )
        blocks.append(
            f"<h3>{esc(d['factor'])} conditioned on {esc(d['regime_column'])}</h3>"
            f'<p class="note">{esc(d["rationale"])}</p>'
            f"<p>Pre-declared direction: IC in <code>{esc(d['favourable_bucket'])}</code> "
            f"&gt; IC in <code>{esc(d['unfavourable_bucket'])}</code>. "
            f"Verdict: {verdict_html(d['verdict'])} "
            f"(gate |t| &ge; {fmt(d['gate_t_threshold'], 1)}).</p>"
        )
    table = frame_to_table(
        pl.DataFrame(rows) if rows else pl.DataFrame(schema={"factor": pl.Utf8}),
        digits={"IC favourable": 4, "IC unfavourable": 4, "difference": 4, "t(diff, NW)": 2},
    )
    return table + "".join(blocks)
