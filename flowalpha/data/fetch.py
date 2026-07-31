"""Polite, retrying HTTP client for NSE endpoints.

Every parameter (user agent, delay, retries, backoff, timeout) comes from the
``download`` block of ``config.yaml``. Nothing here is tuned in code.

Two NSE behaviours are handled explicitly:

* The ``www.nseindia.com`` API endpoints require a cookie obtained by first visiting
  a normal page with a browser-like user agent. A bare API request returns a 401 or
  an HTML challenge.
* Missing archive files return a **styled HTML error page**. This client returns the
  body as-is and lets the parsers reject HTML, so that "missing file" and "corrupt
  file" stay distinguishable rather than both collapsing into an exception here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping

import requests

from ..config import Config


class FetchError(Exception):
    """Raised when a URL could not be retrieved after the configured retries."""


@dataclass
class FetchResult:
    """Outcome of one request. ``ok`` is False for a clean 404 rather than raising,
    because a session with no published archive is a normal, reportable condition."""

    url: str
    status: int
    content: bytes
    ok: bool

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class NSESession:
    """Thread-safe, rate-limited ``requests`` session.

    The politeness delay is shared across threads via a lock on a single
    "next allowed request time", so N workers make N times fewer requests per
    second than N independent sleepers would. That is the difference between a
    considerate crawl of ~1900 files and getting blocked halfway through.
    """

    #: Visited once to obtain NSE's cookie before hitting an api/ endpoint.
    WARMUP_URL = "https://www.nseindia.com/reports/fii-dii"

    def __init__(self, cfg: Config, *, delay_sec: float | None = None) -> None:
        dl = cfg["download"]
        self._delay = float(delay_sec if delay_sec is not None else dl["request_delay_sec"])
        self._max_retries = int(dl["max_retries"])
        self._backoff_base = float(dl["backoff_base_sec"])
        self._timeout = float(dl["timeout_sec"])
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": str(dl["user_agent"]),
                "Accept": "text/csv,application/json,text/html;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
        )
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._warmed = False

    # -- rate limiting ------------------------------------------------------
    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._delay
        if wait > 0:
            time.sleep(wait)

    def warmup(self) -> None:
        """Obtain NSE's session cookie. Idempotent and best-effort."""
        with self._lock:
            if self._warmed:
                return
            self._warmed = True
        try:
            self._session.get(self.WARMUP_URL, timeout=self._timeout)
        except requests.RequestException:
            # A failed warmup is not fatal: archive hosts do not need the cookie.
            pass

    # -- requests -----------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        allow_missing: bool = True,
    ) -> FetchResult:
        """GET ``url`` with retries and exponential backoff.

        A 404 (or any 4xx that is not 429) returns ``ok=False`` immediately -- retrying
        a file that does not exist wastes ~5 requests per missing session, and with
        ~1900 sessions that is the difference between minutes and hours.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self._timeout, headers=dict(headers or {}))
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self._backoff_base ** attempt)
                continue
            if resp.status_code == 200:
                return FetchResult(url=url, status=200, content=resp.content, ok=True)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                if allow_missing:
                    return FetchResult(
                        url=url, status=resp.status_code, content=resp.content, ok=False
                    )
                raise FetchError(f"{url}: HTTP {resp.status_code}")
            last_exc = FetchError(f"{url}: HTTP {resp.status_code}")
            time.sleep(self._backoff_base ** attempt)
        raise FetchError(f"{url}: giving up after {self._max_retries} attempts ({last_exc})")

    def get_json_api(self, url: str) -> FetchResult:
        """GET an ``nseindia.com/api/`` endpoint, warming the cookie first."""
        self.warmup()
        return self.get(
            url,
            headers={"Accept": "application/json", "Referer": self.WARMUP_URL},
            allow_missing=True,
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NSESession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
