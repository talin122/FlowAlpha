"""Provenance: the record must never overstate what the data is.

Every test here corresponds to a way a pipeline can end up describing generated numbers
as market data, or vice versa.
"""

from __future__ import annotations

import datetime as _dt
import json

import pytest

from flowalpha.data.provenance import (
    DATA_SOURCE_FILENAME,
    PROVENANCE_FILENAME,
    REAL,
    SYNTHETIC,
    SYNTHETIC_MARKER,
    UNAVAILABLE,
    DatasetOrigin,
    Provenance,
    ProvenanceError,
    sync_synthetic_marker,
    to_ascii,
    write_data_source_md,
)


# --- statuses --------------------------------------------------------------

def test_invalid_status_rejected():
    with pytest.raises(ProvenanceError, match="invalid status"):
        DatasetOrigin(status="probably_real", source="x")


def test_origin_roundtrip():
    o = DatasetOrigin(status=REAL, source="src", note="n")
    assert DatasetOrigin.from_dict(o.to_dict()) == o


def test_origin_requires_status():
    with pytest.raises(ProvenanceError, match="missing 'status'"):
        DatasetOrigin.from_dict({"source": "x"})


# --- labels ----------------------------------------------------------------

def test_unrecorded_never_claims_live_data():
    """The single most important default in this module."""
    prov = Provenance.unrecorded()
    assert prov.label() == "PROVENANCE UNRECORDED"
    assert "UNRECORDED" in prov.banner()
    assert not prov.all_real
    assert not prov.all_synthetic


def test_missing_file_yields_unrecorded_not_live(tmp_path):
    assert Provenance.load(tmp_path).label() == "PROVENANCE UNRECORDED"


def test_corrupt_file_yields_unrecorded_not_live(tmp_path):
    (tmp_path / PROVENANCE_FILENAME).write_text("{not json", encoding="utf-8")
    assert Provenance.load(tmp_path).label() == "PROVENANCE UNRECORDED"


def test_non_mapping_file_yields_unrecorded(tmp_path):
    (tmp_path / PROVENANCE_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    assert Provenance.load(tmp_path).label() == "PROVENANCE UNRECORDED"


def test_bad_status_in_file_yields_unrecorded(tmp_path):
    (tmp_path / PROVENANCE_FILENAME).write_text(
        json.dumps({"datasets": {"prices": {"status": "maybe"}}}), encoding="utf-8"
    )
    assert Provenance.load(tmp_path).label() == "PROVENANCE UNRECORDED"


def test_all_real_is_live_and_has_no_banner():
    prov = Provenance().mark("prices", REAL, "yahoo").mark("flows_daily", REAL, "nse")
    assert prov.label() == "LIVE DATA"
    assert prov.banner() == ""
    assert prov.all_real


def test_all_synthetic_is_labelled_synthetic():
    prov = Provenance().mark("prices", SYNTHETIC, "gen").mark("flows_daily", SYNTHETIC, "gen")
    assert prov.label() == "SYNTHETIC DATA"
    assert "SYNTHETIC DATA" in prov.banner()
    assert prov.all_synthetic


def test_hybrid_is_not_live():
    """The failure this module exists to prevent."""
    prov = Provenance().mark("prices", REAL, "yahoo").mark("flows_daily", SYNTHETIC, "gen")
    label = prov.label()
    assert label != "LIVE DATA"
    assert label.startswith("HYBRID")
    assert "FLOWS_DAILY" in label
    assert prov.any_synthetic and not prov.all_synthetic
    assert prov.banner()


def test_hybrid_lists_every_synthetic_dataset():
    prov = (
        Provenance()
        .mark("prices", REAL, "yahoo")
        .mark("flows_daily", SYNTHETIC, "gen")
        .mark("participant_flows", SYNTHETIC, "gen")
    )
    label = prov.label()
    assert "FLOWS_DAILY" in label and "PARTICIPANT_FLOWS" in label


def test_partial_real_with_unavailable_says_so():
    """This project's actual state: real prices, delivery and shares genuinely absent."""
    prov = (
        Provenance()
        .mark("prices", REAL, "yahoo")
        .mark("delivery_pct", UNAVAILABLE, "yahoo")
        .mark("shares_outstanding", UNAVAILABLE, "yahoo")
    )
    label = prov.label()
    assert label.startswith("LIVE DATA (PARTIAL")
    assert "DELIVERY_PCT" in label and "SHARES_OUTSTANDING" in label
    assert not prov.all_real
    assert prov.banner()


def test_empty_record_is_not_all_synthetic():
    assert not Provenance().all_synthetic
    assert not Provenance().all_real


def test_status_of_and_drop():
    prov = Provenance().mark("prices", REAL, "src")
    assert prov.status_of("prices") == REAL
    assert prov.status_of("nope") is None
    prov.drop("prices")
    assert prov.status_of("prices") is None


# --- ASCII discipline ------------------------------------------------------

def test_to_ascii_transliterates_the_characters_we_emit():
    assert to_ascii("Rs 100 – 200 crore") == "Rs 100 - 200 crore"
    assert to_ascii("₹1,000") == "Rs1,000"
    assert to_ascii("a → b") == "a -> b"
    assert to_ascii("“quoted”") == '"quoted"'


def test_to_ascii_never_raises_on_exotic_input():
    out = to_ascii("你好 \U0001F600")
    out.encode("ascii")  # must not raise


@pytest.mark.parametrize(
    "prov",
    [
        Provenance.unrecorded(),
        Provenance().mark("prices", REAL, "yahoo"),
        Provenance().mark("prices", SYNTHETIC, "gen"),
        Provenance().mark("prices", REAL, "y").mark("flows_daily", SYNTHETIC, "g"),
        Provenance().mark("prices", REAL, "y").mark("delivery_pct", UNAVAILABLE, "y"),
    ],
)
def test_every_console_string_is_ascii(prov):
    """These print to Windows consoles under cp1252, where one en-dash aborts the run."""
    for text in (prov.label(), prov.banner(), prov.tag(), prov.header("script")):
        text.encode("ascii")
    for line in prov.describe():
        line.encode("ascii")


def test_console_strings_are_ascii_even_with_unicode_sources():
    prov = Provenance().mark("prices", REAL, "NSE – archives", note="₹ crores")
    prov.header("x").encode("ascii")
    "\n".join(prov.describe()).encode("ascii")


# --- caveats ---------------------------------------------------------------

def test_caveats_are_deduped_and_ordered():
    prov = Provenance()
    prov.add_caveat("b").add_caveat("a").add_caveat("b").add_caveat("  ")
    assert prov.caveats == ["b", "a"]


# --- persistence -----------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    prov = Provenance().mark("prices", REAL, "yahoo", note="n").add_caveat("c")
    ts = _dt.datetime(2026, 7, 30, tzinfo=_dt.timezone.utc)
    prov.save(tmp_path, timestamp=ts)
    again = Provenance.load(tmp_path)
    assert again.label() == "LIVE DATA"
    assert again.caveats == ["c"]
    assert again.updated_at == ts.isoformat()
    assert again.datasets["prices"].note == "n"


def test_saved_json_records_the_label(tmp_path):
    Provenance().mark("prices", SYNTHETIC, "gen").save(tmp_path)
    raw = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert raw["label"] == "SYNTHETIC DATA"


# --- generated markdown ----------------------------------------------------

def test_data_source_md_is_generated_from_the_record(tmp_path):
    """Generated so prose cannot drift from fact."""
    prov = Provenance().mark("prices", REAL, "yahoo", note="500 names").add_caveat("watch out")
    prov.save(tmp_path)
    text = (tmp_path / DATA_SOURCE_FILENAME).read_text()
    assert "GENERATED FILE" in text
    assert "Do not hand-edit" in text
    assert "LIVE DATA" in text
    assert "yahoo" in text and "500 names" in text
    assert "watch out" in text


def test_data_source_md_matches_a_hybrid_record(tmp_path):
    prov = Provenance().mark("prices", REAL, "y").mark("flows_daily", SYNTHETIC, "g")
    prov.save(tmp_path)
    text = (tmp_path / DATA_SOURCE_FILENAME).read_text()
    assert "HYBRID" in text
    assert "LIVE DATA" not in text.split("Dataset counts")[0].replace("HYBRID", "")


def test_data_source_md_is_ascii(tmp_path):
    prov = Provenance().mark("prices", REAL, "NSE – x", note="₹ crore")
    prov.save(tmp_path)
    (tmp_path / DATA_SOURCE_FILENAME).read_text().encode("ascii")


def test_data_source_md_for_an_empty_record(tmp_path):
    path = write_data_source_md(Provenance.unrecorded(), tmp_path / "DS.md")
    text = path.read_text()
    assert "No datasets are recorded" in text
    assert "unknown origin" in text


def test_data_source_md_escapes_pipes(tmp_path):
    prov = Provenance().mark("prices", REAL, "a|b", note="c|d")
    prov.save(tmp_path)
    text = (tmp_path / DATA_SOURCE_FILENAME).read_text()
    assert "a\\|b" in text and "c\\|d" in text


# --- legacy marker ---------------------------------------------------------

def test_marker_written_only_when_everything_is_synthetic(tmp_path):
    prov = Provenance().mark("prices", SYNTHETIC, "gen")
    prov.save(tmp_path)
    assert (tmp_path / SYNTHETIC_MARKER).exists()


def test_marker_removed_as_soon_as_any_real_dataset_appears(tmp_path):
    """A stale marker on a hybrid tree is a provenance lie in the other direction."""
    prov = Provenance().mark("prices", SYNTHETIC, "gen")
    prov.save(tmp_path)
    assert (tmp_path / SYNTHETIC_MARKER).exists()

    prov.mark("flows_daily", REAL, "nse")
    prov.save(tmp_path)
    assert not (tmp_path / SYNTHETIC_MARKER).exists()
    assert prov.label().startswith("HYBRID")


def test_marker_absent_for_unrecorded(tmp_path):
    sync_synthetic_marker(Provenance.unrecorded(), tmp_path)
    assert not (tmp_path / SYNTHETIC_MARKER).exists()


def test_marker_absent_when_something_is_merely_unavailable(tmp_path):
    prov = Provenance().mark("prices", SYNTHETIC, "g").mark("delivery_pct", UNAVAILABLE, "g")
    prov.save(tmp_path)
    assert not (tmp_path / SYNTHETIC_MARKER).exists()


# --- headers ---------------------------------------------------------------

def test_header_includes_script_name_and_label():
    prov = Provenance().mark("prices", REAL, "y")
    header = prov.header("run_baseline")
    assert "run_baseline" in header
    assert "LIVE DATA" in header


def test_header_of_a_real_run_carries_no_warning_line():
    prov = Provenance().mark("prices", REAL, "y")
    assert "!!" not in prov.header("x")


def test_header_of_a_hybrid_run_carries_a_warning_line():
    prov = Provenance().mark("prices", REAL, "y").mark("flows_daily", SYNTHETIC, "g")
    assert "!!" in prov.header("x")


def test_describe_lists_every_dataset():
    prov = Provenance().mark("prices", REAL, "y", note="n").mark("flows_daily", SYNTHETIC, "g")
    lines = "\n".join(prov.describe())
    assert "prices" in lines and "flows_daily" in lines
    assert "synthetic" in lines and "real" in lines


# --- the real project record ----------------------------------------------

def test_the_repository_record_is_accurate_if_present():
    """The shipped tree's own record must be internally consistent."""
    from flowalpha.config import REPO_ROOT

    prov = Provenance.load(REPO_ROOT / "data")
    if not prov.datasets:
        pytest.skip("no provenance record in this checkout")
    label = prov.label()
    # Whatever it says, it must be consistent with the recorded statuses.
    if prov.any_synthetic:
        assert label.startswith("HYBRID") or label == "SYNTHETIC DATA"
    elif prov.unavailable:
        assert label.startswith("LIVE DATA (PARTIAL")
    else:
        assert label == "LIVE DATA"
    label.encode("ascii")


def test_generated_markdown_matches_the_repository_record():
    from flowalpha.config import REPO_ROOT

    data_dir = REPO_ROOT / "data"
    md = data_dir / DATA_SOURCE_FILENAME
    if not (data_dir / PROVENANCE_FILENAME).exists() or not md.exists():
        pytest.skip("no provenance record in this checkout")
    prov = Provenance.load(data_dir)
    assert prov.label() in md.read_text(encoding="utf-8")
