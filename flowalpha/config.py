"""Configuration access.

Every tunable parameter in FlowAlpha lives in ``config.yaml``. This module is the
only sanctioned way to read it, which matters for reproducibility: a run directory
snapshots the exact config that produced it, so a result can always be traced back
to the parameters that generated it.

Design notes
------------
* ``Config`` is read-only by construction (``__setitem__`` raises). Research code
  that mutates config mid-run produces results whose provenance snapshot lies.
* ``now_ist`` is injectable via ``new_run_dir(..., timestamp=...)`` so tests are
  deterministic without monkeypatching the clock.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30), name="IST")

#: Repository root -- the directory containing ``config.yaml``.
REPO_ROOT = Path(__file__).resolve().parent.parent

_PATH_KEYS = ("raw", "processed", "reference", "results")


def now_ist() -> _dt.datetime:
    """Current wall-clock time in Asia/Kolkata.

    Indian market data is timestamped in IST; using local machine time would make
    "as of today" mean different sessions depending on where the code runs.
    """
    return _dt.datetime.now(tz=IST)


class ConfigError(Exception):
    """Raised when ``config.yaml`` is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Config:
    """Immutable view over ``config.yaml``.

    Supports ``cfg["universe"]["size"]`` style access plus dotted lookup via
    :meth:`get`, and resolves the four declared path roots against the repo root.

    Immutability is **deep**, not just top level. ``__getitem__`` and :meth:`get` return
    deep copies of any container they hand back, so ``cfg["universe"]["size"] = 1``
    mutates a throwaway copy instead of the live config. Blocking only top-level
    assignment left that hole open, and a run that mutates its config mid-flight produces
    a run directory whose ``config.yaml`` snapshot no longer describes what actually ran --
    which quietly breaks the reproducibility invariant the snapshot exists to provide.
    The copies are of a small YAML document read a few dozen times per run, so the cost is
    irrelevant.
    """

    _data: Mapping[str, Any]
    root: Path = field(default=REPO_ROOT)
    source_file: Path | None = field(default=None)

    # -- mapping protocol ---------------------------------------------------
    @staticmethod
    def _detach(value: Any) -> Any:
        """Return a value safe to hand out: containers are deep-copied, scalars are not."""
        if isinstance(value, (Mapping, list, set, tuple, bytearray)):
            return copy.deepcopy(value)
        return value

    def __getitem__(self, key: str) -> Any:
        try:
            return self._detach(self._data[key])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"config key {key!r} not present in {self.source_file}") from exc

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            "Config is immutable: mutating it mid-run would invalidate the run "
            "directory's config snapshot. Edit config.yaml or pass overrides."
        )

    def keys(self):
        return self._data.keys()

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key by dotted path, e.g. ``cfg.get("factors.momentum.skip")``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return self._detach(node)

    def require(self, dotted: str) -> Any:
        """Like :meth:`get` but raises rather than silently defaulting.

        Used where a missing parameter should stop the run: a default invented at
        the call site is a hardcoded parameter by another name.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"required config key missing: {dotted}")
        return value

    # -- derived accessors --------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self._data["seed"])

    def path(self, which: str) -> Path:
        """Resolve one of the declared path roots against the repo root."""
        if which not in _PATH_KEYS:
            raise KeyError(f"unknown path key {which!r}; expected one of {_PATH_KEYS}")
        return (self.root / str(self._data["paths"][which])).resolve()

    @property
    def start_date(self) -> _dt.date:
        return _dt.date.fromisoformat(str(self._data["dates"]["start"]))

    @property
    def end_date(self) -> _dt.date:
        return _dt.date.fromisoformat(str(self._data["dates"]["end"]))

    def as_dict(self) -> dict[str, Any]:
        """Deep copy, so callers cannot reach in and mutate the frozen mapping."""
        return copy.deepcopy(dict(self._data))

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a new Config with top-level keys replaced.

        Tests use this to pin a date window or shrink a universe without editing
        the shipped config file.
        """
        data = self.as_dict()
        data.update(copy.deepcopy(overrides))
        return Config(_data=data, root=self.root, source_file=self.source_file)


def _validate(data: Mapping[str, Any], source: Path) -> None:
    required_top = (
        "seed",
        "paths",
        "universe",
        "dates",
        "availability",
        "factors",
        "forward_returns",
        "validation",
        "conditioning",
        "backtest",
        "signals",
    )
    missing = [k for k in required_top if k not in data]
    if missing:
        raise ConfigError(f"{source}: missing top-level sections: {', '.join(missing)}")

    for key in _PATH_KEYS:
        if key not in data["paths"]:
            raise ConfigError(f"{source}: paths.{key} is required")

    start = str(data["dates"]["start"])
    end = str(data["dates"]["end"])
    try:
        s, e = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    except ValueError as exc:
        raise ConfigError(f"{source}: dates must be ISO YYYY-MM-DD ({exc})") from exc
    if e <= s:
        raise ConfigError(f"{source}: dates.end ({end}) must be after dates.start ({start})")

    band = data["universe"].get("cardinality_band")
    if not (isinstance(band, (list, tuple)) and len(band) == 2 and band[0] < band[1]):
        raise ConfigError(f"{source}: universe.cardinality_band must be [lo, hi] with lo < hi")

    # Availability entries must declare exactly one lag flavour, else the store
    # would have to guess -- and a guessed lag is an unenforced lag.
    for dataset, spec in data["availability"].items():
        keys = set(spec or {})
        if keys not in ({"lag_days"}, {"lag_calendar_days"}):
            raise ConfigError(
                f"{source}: availability.{dataset} must declare exactly one of "
                f"'lag_days' (trading days) or 'lag_calendar_days'; got {sorted(keys)}"
            )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path:
        Explicit config file. Defaults to ``config.yaml`` at the repo root. When a
        config is loaded from inside a run directory, path roots still resolve
        against the repo root so a snapshot stays readable in place.
    """
    cfg_path = Path(path) if path is not None else REPO_ROOT / "config.yaml"
    cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    _validate(data, cfg_path)
    return Config(_data=data, root=REPO_ROOT, source_file=cfg_path)


def git_hash(root: Path | None = None) -> str:
    """Short git hash of the working tree, or a marker when unavailable.

    Returns ``"nogit"`` outside a repository and appends ``"-dirty"`` when the tree
    has uncommitted changes -- a result produced from a dirty tree is not exactly
    reproducible and the snapshot should say so.
    """
    root = root or REPO_ROOT
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return "nogit"
        h = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if status.returncode == 0 and status.stdout.strip():
            h += "-dirty"
        return h
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "nogit"


def new_run_dir(
    cfg: Config,
    label: str,
    *,
    timestamp: _dt.datetime | None = None,
) -> Path:
    """Create ``results/runs/<YYYYmmdd_HHMMSS>_<label>/`` and snapshot the run inputs.

    Writes both ``config.yaml`` (verbatim copy where possible, else a serialised
    dump) and ``provenance.json`` (timestamp, git hash, label, cwd). Together these
    make the run reproducible from the directory alone.

    ``timestamp`` is injectable so tests get a deterministic directory name.
    """
    ts = timestamp or now_ist()
    safe_label = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label) or "run"
    run_dir = cfg.path("results") / "runs" / f"{ts.strftime('%Y%m%d_%H%M%S')}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot = run_dir / "config.yaml"
    if cfg.source_file and cfg.source_file.exists():
        shutil.copyfile(cfg.source_file, snapshot)
    else:
        snapshot.write_text(yaml.safe_dump(cfg.as_dict(), sort_keys=False), encoding="utf-8")

    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "label": label,
                "timestamp_ist": ts.isoformat(),
                "git_hash": git_hash(cfg.root),
                "cwd": str(Path.cwd()),
                "config_source": str(cfg.source_file) if cfg.source_file else None,
                "seed": cfg.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
