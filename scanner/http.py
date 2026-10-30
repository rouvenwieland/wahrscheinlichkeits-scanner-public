"""Gemeinsamer HTTP-Helfer: Timeout, Wiederholungen mit Backoff, harte Fehler.

Ein Abruffehler fuehrt immer zu ``FetchError``; es werden nie Teilergebnisse
still zurueckgegeben (Lehre aus dem alten Scanner).
"""
from __future__ import annotations

import time
from typing import Any

import requests

USER_AGENT = "wahrscheinlichkeits-scanner/0.1 (Forschung, nur lesend)"


class FetchError(RuntimeError):
    """Ein Netzwerkabruf ist endgueltig fehlgeschlagen."""


_SESSION: requests.Session | None = None


def _shared_session() -> requests.Session:
    """Eine Sitzung fuer alle Abrufe eines Laufs: Verbindungen (TLS) werden wiederverwendet."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def sleep(seconds: float) -> None:  # eigene Funktion, damit Tests sie ersetzen koennen
    time.sleep(seconds)


def get_json(url: str, params: dict | None = None, *, timeout: float = 15.0,
             retries: int = 3, backoff: float = 1.0,
             session: requests.Session | None = None) -> Any:
    """GET mit hoechstens ``retries`` Versuchen. Wirft ``FetchError`` bei Misserfolg."""
    sess = session or _shared_session()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429 or resp.status_code >= 500:
                last = FetchError(f"HTTP {resp.status_code} bei {url}")
            elif resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} bei {url}")
            else:
                return resp.json()
        except FetchError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last = exc
        if attempt < retries - 1:
            sleep(backoff * (2 ** attempt))
    raise FetchError(f"Abruf fehlgeschlagen nach {retries} Versuchen: {url}: {last}")
