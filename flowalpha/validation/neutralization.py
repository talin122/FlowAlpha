"""Cross-sectional treatment of factor panels.

Every step here is applied **within a date**, never across dates. A z-score computed
over the pooled panel would mix today's cross-section with history and leak
information both ways; a winsorisation bound estimated on the full sample is a
full-sample quantile, which is look-ahead by another name.

Within a single date there is no look-ahead to commit: the cross-section at ``t`` is
entirely observable at ``t``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import polars as pl

VALUE_COL = "value"


def _require(frame: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"panel missing required column(s) {missing}; got {frame.columns}")


def winsorize(
    panel: pl.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    value_col: str = VALUE_COL,
    by: Sequence[str] = ("date",),
) -> pl.DataFrame:
    """Clip values to their cross-sectional quantiles within each ``by`` group.

    Clips rather than drops: dropping the tails would change the universe from one
    date to the next as a function of factor value, which is a selection effect
    masquerading as an outlier control.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"invalid winsorisation bounds ({lower}, {upper})")
    _require(panel, [value_col, *by])
    if panel.is_empty():
        return panel
    return panel.with_columns(
        pl.col(value_col)
        .clip(
            pl.col(value_col).quantile(lower, interpolation="linear").over(by),
            pl.col(value_col).quantile(upper, interpolation="linear").over(by),
        )
        .alias(value_col)
    )


def zscore(
    panel: pl.DataFrame,
    *,
    value_col: str = VALUE_COL,
    by: Sequence[str] = ("date",),
    min_count: int = 2,
) -> pl.DataFrame:
    """Standardise within each ``by`` group.

    Groups with fewer than ``min_count`` observations, or with zero cross-sectional
    dispersion, produce nulls rather than infinities: a date on which every name has
    the same factor value carries no cross-sectional information, and pretending
    otherwise puts a divide-by-zero into every downstream average.
    """
    _require(panel, [value_col, *by])
    if panel.is_empty():
        return panel
    std = pl.col(value_col).std().over(by)
    count = pl.col(value_col).count().over(by)
    # Relative degeneracy test. The standard deviation of a numerically constant
    # cross-section is not exactly zero, and dividing by 1e-19 turns a flat date into
    # z-scores of 1e8 that then dominate every composite average.
    scale = pl.max_horizontal(
        pl.col(value_col).abs().max().over(by), pl.lit(1e-300)
    )
    return panel.with_columns(
        pl.when((count >= min_count) & std.is_not_null() & (std > 1e-12 * scale))
        .then((pl.col(value_col) - pl.col(value_col).mean().over(by)) / std)
        .otherwise(None)
        .alias(value_col)
    )


def sector_neutralize(
    panel: pl.DataFrame,
    sectors: Mapping[str, str] | pl.DataFrame,
    *,
    value_col: str = VALUE_COL,
    entity_col: str = "symbol",
    date_col: str = "date",
    unknown_label: str = "NA",
) -> pl.DataFrame:
    """Demean within (date, sector).

    Names with no sector mapping land in ``unknown_label`` and are demeaned against
    each other. That is the honest handling: silently dropping them would shrink the
    universe, and treating them as a single real sector would be a false claim about
    their exposures.

    A ``sectors`` mapping in which every name shares one label makes this a no-op up
    to floating point, which is exactly what should happen -- and is why sector
    neutralisation does nothing when sector data is unavailable.
    """
    _require(panel, [value_col, entity_col, date_col])
    if panel.is_empty():
        return panel
    if isinstance(sectors, pl.DataFrame):
        if entity_col not in sectors.columns or "sector" not in sectors.columns:
            raise ValueError(f"sectors frame needs '{entity_col}' and 'sector' columns")
        mapping = sectors
    else:
        mapping = pl.DataFrame(
            {entity_col: list(sectors.keys()), "sector": list(sectors.values())},
            schema={entity_col: pl.Utf8, "sector": pl.Utf8},
        )
    joined = panel.join(mapping, on=entity_col, how="left").with_columns(
        pl.col("sector").fill_null(unknown_label)
    )
    out = joined.with_columns(
        (pl.col(value_col) - pl.col(value_col).mean().over([date_col, "sector"])).alias(value_col)
    )
    return out.drop("sector")


def treat_cross_section(
    panel: pl.DataFrame,
    *,
    winsor: tuple[float, float] = (0.01, 0.99),
    sector_neutral: bool = False,
    sectors: Mapping[str, str] | pl.DataFrame | None = None,
    value_col: str = VALUE_COL,
) -> pl.DataFrame:
    """Standard treatment pipeline: winsorise, z-score, optionally sector-neutralise.

    Sector neutralisation is applied *after* standardisation and followed by a second
    standardisation, so the output is comparable across dates whether or not
    neutralisation ran.
    """
    if panel.is_empty():
        return panel
    out = winsorize(panel, lower=winsor[0], upper=winsor[1], value_col=value_col)
    out = zscore(out, value_col=value_col)
    if sector_neutral:
        if sectors is None:
            raise ValueError(
                "sector_neutralize is enabled but no sector mapping was supplied. "
                "Refusing to silently skip: a run that believes it is sector-neutral "
                "and is not would misattribute an industry bet to a factor."
            )
        out = sector_neutralize(out, sectors, value_col=value_col)
        out = zscore(out, value_col=value_col)
    return out
