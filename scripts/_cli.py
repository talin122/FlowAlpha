"""Shared CLI plumbing: provenance-derived headers and repo-root importability.

Every entry point prints a header derived from ``data/PROVENANCE.json`` rather than
from anything it infers locally. If provenance has not been written, the header says
``PROVENANCE UNRECORDED`` -- it never guesses "live data".
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowalpha.config import Config, load_config  # noqa: E402
from flowalpha.data.provenance import Provenance, to_ascii  # noqa: E402


def bootstrap(script_name: str, *, quiet: bool = False) -> tuple[Config, Provenance]:
    """Load config and provenance, print the standard header, return both."""
    cfg = load_config()
    prov = Provenance.load(REPO_ROOT / "data")
    if not quiet:
        print(prov.header(script_name), flush=True)
    return cfg, prov


def say(*parts: object) -> None:
    """Print ASCII-folded output.

    Console strings are folded because these scripts are expected to run on Windows
    terminals under cp1252, where one stray en-dash aborts the run.
    """
    print(to_ascii(" ".join(str(p) for p in parts)), flush=True)
