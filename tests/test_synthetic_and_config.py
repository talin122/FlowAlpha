"""Config-layer and reproducibility tests (GATE 1). Synthetic-DGP tests appended in Phase 8."""

from __future__ import annotations

import datetime as _dt
import json

import pytest
import yaml

from conftest import make_config

from flowalpha.config import (
    IST,
    Config,
    ConfigError,
    load_config,
    new_run_dir,
    now_ist,
)


def test_load_config_reads_shipped_file():
    cfg = load_config()
    assert cfg.seed == 20240101
    assert cfg["universe"]["name"] == "NIFTY500"
    assert cfg["universe"]["size"] == 500


def test_dotted_get_and_require():
    cfg = load_config()
    assert cfg.get("factors.momentum.skip") == 21
    assert cfg.get("factors.momentum.nope", "fallback") == "fallback"
    with pytest.raises(ConfigError):
        cfg.require("factors.momentum.nope")


def test_paths_resolve_against_repo_root():
    cfg = load_config()
    for key in ("raw", "processed", "reference", "results"):
        p = cfg.path(key)
        assert p.is_absolute()
        assert p.is_relative_to(cfg.root)
    with pytest.raises(KeyError):
        cfg.path("not_a_path_key")


def test_config_is_immutable():
    cfg = load_config()
    with pytest.raises(TypeError):
        cfg["seed"] = 1


def test_as_dict_is_a_deep_copy():
    cfg = load_config()
    d = cfg.as_dict()
    d["universe"]["size"] = 1
    assert cfg["universe"]["size"] == 500


def test_with_overrides_leaves_original_untouched():
    cfg = load_config()
    other = cfg.with_overrides(dates={"start": "2020-01-01", "end": "2021-01-01", "timezone": "Asia/Kolkata"})
    assert other.start_date == _dt.date(2020, 1, 1)
    assert cfg.start_date == _dt.date(2019, 1, 1)


def test_date_accessors():
    cfg = load_config()
    assert cfg.start_date == _dt.date(2019, 1, 1)
    assert cfg.end_date > cfg.start_date


def test_cardinality_band_is_not_centred_on_size():
    """The band must tolerate survivorship-driven growth, not assume a flat 500.

    Documented in Phase 3: current-constituent lists applied historically give a
    membership count that rises over time, so a band centred on `size` can never
    pass. This test pins the intent so nobody "tidies" the band later.
    """
    cfg = load_config()
    lo, hi = cfg["universe"]["cardinality_band"]
    size = cfg["universe"]["size"]
    assert lo < size <= hi
    assert lo < 0.8 * size


def test_availability_entries_declare_exactly_one_lag_flavour():
    cfg = load_config()
    for dataset, spec in cfg["availability"].items():
        keys = set(spec)
        assert keys in ({"lag_days"}, {"lag_calendar_days"}), dataset


def _write_cfg(tmp_path, data):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _minimal(**patch):
    base = {
        "seed": 1,
        "paths": {"raw": "data/raw", "processed": "data/processed",
                  "reference": "data/reference", "results": "results"},
        "universe": {"name": "X", "size": 10, "cardinality_band": [5, 20]},
        "dates": {"start": "2020-01-01", "end": "2021-01-01", "timezone": "Asia/Kolkata"},
        "availability": {"prices": {"lag_days": 0}},
        "factors": {},
        "forward_returns": {"horizons": [1]},
        "validation": {},
        "conditioning": {},
        "backtest": {},
        "signals": {},
    }
    base.update(patch)
    return base


def test_missing_section_is_an_error(tmp_path):
    data = _minimal()
    del data["conditioning"]
    with pytest.raises(ConfigError, match="conditioning"):
        load_config(_write_cfg(tmp_path, data))


def test_reversed_dates_are_an_error(tmp_path):
    data = _minimal(dates={"start": "2021-01-01", "end": "2020-01-01", "timezone": "Asia/Kolkata"})
    with pytest.raises(ConfigError, match="must be after"):
        load_config(_write_cfg(tmp_path, data))


def test_ambiguous_availability_lag_is_an_error(tmp_path):
    """A dataset declaring both lag flavours would force the store to guess."""
    data = _minimal(availability={"prices": {"lag_days": 1, "lag_calendar_days": 3}})
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(_write_cfg(tmp_path, data))


def test_empty_availability_lag_is_an_error(tmp_path):
    data = _minimal(availability={"prices": {}})
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(_write_cfg(tmp_path, data))


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_now_ist_is_tz_aware_kolkata():
    ts = now_ist()
    assert ts.tzinfo is not None
    assert ts.utcoffset() == _dt.timedelta(hours=5, minutes=30)
    assert IST.utcoffset(None) == _dt.timedelta(hours=5, minutes=30)


def test_new_run_dir_snapshots_config_and_provenance(tmp_path):
    src = _write_cfg(tmp_path, _minimal(paths={
        "raw": "raw", "processed": "processed", "reference": "reference",
        "results": str(tmp_path / "results"),
    }))
    cfg = load_config(src)
    ts = _dt.datetime(2026, 7, 30, 9, 15, 0, tzinfo=IST)
    run_dir = new_run_dir(cfg, "baseline", timestamp=ts)

    assert run_dir.name == "20260730_091500_baseline"
    assert (run_dir / "config.yaml").exists()
    snap = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert snap["seed"] == cfg.seed

    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["label"] == "baseline"
    assert prov["timestamp_ist"] == ts.isoformat()
    assert prov["seed"] == cfg.seed
    assert prov["git_hash"]


def test_new_run_dir_sanitises_label(tmp_path):
    src = _write_cfg(tmp_path, _minimal(paths={
        "raw": "raw", "processed": "processed", "reference": "reference",
        "results": str(tmp_path / "results"),
    }))
    cfg = load_config(src)
    ts = _dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=IST)
    run_dir = new_run_dir(cfg, "a/b c", timestamp=ts)
    assert run_dir.name == "20260101_000000_a_b_c"


def test_run_dir_snapshot_is_loadable_and_paths_still_resolve(tmp_path):
    """A snapshot must be readable in place so an old run can be re-derived."""
    src = _write_cfg(tmp_path, _minimal(paths={
        "raw": "raw", "processed": "processed", "reference": "reference",
        "results": str(tmp_path / "results"),
    }))
    cfg = load_config(src)
    run_dir = new_run_dir(cfg, "x", timestamp=_dt.datetime(2026, 1, 1, tzinfo=IST))
    reloaded = load_config(run_dir / "config.yaml")
    assert reloaded.seed == cfg.seed
    assert reloaded.path("raw").is_absolute()


def test_config_without_source_file_still_snapshots(tmp_path):
    cfg = Config(_data=_minimal(paths={
        "raw": "raw", "processed": "processed", "reference": "reference",
        "results": str(tmp_path / "results"),
    }))
    run_dir = new_run_dir(cfg, "nosrc", timestamp=_dt.datetime(2026, 1, 1, tzinfo=IST))
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["seed"] == 1
    assert json.loads((run_dir / "provenance.json").read_text())["config_source"] is None


# ---------------------------------------------------------------------------
# Run snapshots must identify the DATA, not only the code
# ---------------------------------------------------------------------------

def test_run_snapshot_records_environment_and_input_digests(tmp_path, tiny_sessions):
    """Config plus git hash identifies code and parameters but not inputs, so two runs
    over different data produced identical snapshots. NSE swapped two constituents
    mid-study and nothing recorded it."""
    from flowalpha.config import new_run_dir

    cfg = make_config(tmp_path, tiny_sessions)
    (cfg.path("processed") / "prices.parquet").write_bytes(b"first")
    run = new_run_dir(cfg, "t1")
    snap = json.loads((run / "provenance.json").read_text())

    assert snap["environment"]["python"]
    assert snap["environment"]["packages"]["polars"]
    assert "processed/prices.parquet" in snap["inputs"]
    assert snap["inputs"]["processed/prices.parquet"]["bytes"] == 5


def test_changed_input_changes_the_recorded_digest(tmp_path, tiny_sessions):
    """The property that makes the digest worth storing: same code, same config,
    different data must produce a different record."""
    from flowalpha.config import new_run_dir

    cfg = make_config(tmp_path, tiny_sessions)
    target = cfg.path("processed") / "prices.parquet"

    target.write_bytes(b"universe of 500")
    a = json.loads((new_run_dir(cfg, "a") / "provenance.json").read_text())
    target.write_bytes(b"universe of 498")
    b = json.loads((new_run_dir(cfg, "b") / "provenance.json").read_text())

    ka = a["inputs"]["processed/prices.parquet"]["sha256"]
    kb = b["inputs"]["processed/prices.parquet"]["sha256"]
    assert ka != kb


def test_identical_inputs_give_identical_digests(tmp_path, tiny_sessions):
    """And the converse, or the digest would flag every rerun as a change."""
    from flowalpha.config import new_run_dir

    cfg = make_config(tmp_path, tiny_sessions)
    (cfg.path("processed") / "prices.parquet").write_bytes(b"stable")
    a = json.loads((new_run_dir(cfg, "a") / "provenance.json").read_text())
    b = json.loads((new_run_dir(cfg, "b") / "provenance.json").read_text())
    assert a["inputs"] == b["inputs"]
