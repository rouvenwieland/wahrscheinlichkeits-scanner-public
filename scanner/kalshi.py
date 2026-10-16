"""Kalshi: oeffentliche Marktdaten (kein Konto)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .http import FetchError, get_json

BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class Level:
    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class Book:
    """Ausfuehrbare Seite eines Marktes. Asks aufsteigend, Bids absteigend."""
    yes_asks: tuple
    no_asks: tuple
    yes_bids: tuple
    no_bids: tuple


def fetch_markets(series: str = "KXBTCD", status: str = "open", max_pages: int = 30) -> list:
    """Alle Maerkte der Serie (mit Paginierung). Bei Fehler Ausnahme, nie eine Teilliste."""
    markets, cursor = [], ""
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{BASE}/markets", params)
        page = data.get("markets") if isinstance(data, dict) else None
        if not isinstance(page, list):
            raise FetchError("Kalshi: Antwort ohne 'markets'-Liste")
        markets.extend(page)
        cursor = data.get("cursor") or ""
        if not cursor:
            return markets
    raise FetchError("Kalshi: zu viele Seiten")


def _levels(raw) -> list:
    return [Level(Decimal(str(p)), Decimal(str(s))) for p, s in (raw or [])]


def parse_orderbook(payload: dict) -> Book:
    """Kalshi liefert nur Gebote (Bids) fuer YES und NO.

    Ein NO-Gebot zu Preis q ist ein YES-Angebot zu 1-q (und umgekehrt):
    yes_ask = 1 - bester no_bid, no_ask = 1 - bester yes_bid.
    """
    ob = payload.get("orderbook_fp") if isinstance(payload, dict) else None
    if ob is None and isinstance(payload, dict) and payload.get("orderbook") is not None:
        old = payload["orderbook"]  # altes Format: Cent-Ganzzahlen
        ob = {"yes_dollars": [[Decimal(p) / 100, s] for p, s in (old.get("yes") or [])],
              "no_dollars": [[Decimal(p) / 100, s] for p, s in (old.get("no") or [])]}
    if not isinstance(ob, dict):
        raise FetchError("Kalshi: Orderbuch nicht lesbar")
    yes_bids = sorted(_levels(ob.get("yes_dollars")), key=lambda l: -l.price)
    no_bids = sorted(_levels(ob.get("no_dollars")), key=lambda l: -l.price)
    one = Decimal(1)
    yes_asks = tuple(Level(one - l.price, l.size) for l in no_bids)  # beste (kleinste) zuerst
    no_asks = tuple(Level(one - l.price, l.size) for l in yes_bids)
    return Book(yes_asks, no_asks, tuple(yes_bids), tuple(no_bids))


def fetch_orderbook(ticker: str) -> Book:
    return parse_orderbook(get_json(f"{BASE}/markets/{ticker}/orderbook"))


def fetch_result(ticker: str) -> str:
    """Offizielles Ergebnis: 'yes', 'no' oder '' (offen)."""
    data = get_json(f"{BASE}/markets/{ticker}")
    m = data.get("market") if isinstance(data, dict) else None
    if not isinstance(m, dict):
        raise FetchError("Kalshi: Markt nicht lesbar")
    return (m.get("result") or "").lower()
