"""Polymarket: Gamma (Marktbeschreibung) und CLOB (ausfuehrbare Kurse). Kein Konto."""
from __future__ import annotations

import json
from decimal import Decimal

from .http import FetchError, get_json
from .kalshi import Book, Level

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


PAGE = 100  # Gamma liefert hoechstens 100 Events je Seite (Phase 2 hat nur die erste Seite gelesen)


def fetch_events(tag: str = "bitcoin", max_pages: int = 20) -> list:
    """Alle offenen Events eines Tags (mit Paginierung). Fehler -> Ausnahme, nie eine Teilliste."""
    out = []
    for page in range(max_pages):
        data = get_json(f"{GAMMA}/events", {"closed": "false", "tag_slug": tag,
                                             "limit": PAGE, "offset": page * PAGE})
        if not isinstance(data, list):
            raise FetchError("Polymarket: Events nicht als Liste geliefert")
        out.extend(data)
        if len(data) < PAGE:
            return out
    raise FetchError(f"Polymarket: zu viele Seiten fuer Tag {tag}")


def parse_book_side(payload: dict):
    """(asks aufsteigend, bids absteigend). Die Sortierung der API wird nicht vorausgesetzt."""
    if not isinstance(payload, dict) or "asks" not in payload or "bids" not in payload:
        raise FetchError("Polymarket: Orderbuch ohne bids/asks")

    def mk(rows):
        return [Level(Decimal(str(r["price"])), Decimal(str(r["size"]))) for r in rows]

    return (tuple(sorted(mk(payload["asks"]), key=lambda l: l.price)),
            tuple(sorted(mk(payload["bids"]), key=lambda l: -l.price)))


def fetch_book(yes_token: str, no_token: str) -> Book:
    """Ausfuehrbares Buch aus den CLOB-Buechern beider Token (nie aus outcomePrices)."""
    ya, yb = parse_book_side(get_json(f"{CLOB}/book", {"token_id": yes_token}))
    na, nb = parse_book_side(get_json(f"{CLOB}/book", {"token_id": no_token}))
    return Book(ya, na, yb, nb)


def fetch_market(market_id: str) -> dict:
    data = get_json(f"{GAMMA}/markets/{market_id}")
    if not isinstance(data, dict) or "closed" not in data:
        raise FetchError("Polymarket: Markt nicht lesbar")
    return data


def official_result(market: dict):
    """'yes' / 'no' nur bei offizieller Aufloesung, sonst None.

    Verlangt: closed, Ausgaenge genau ["Yes","No"] (wie bei der Zuordnung; Sport- oder
    Mehrfachmaerkte mit Teamnamen werden hier nie abgerechnet, sie werden auch nie gekauft),
    outcomePrices exakt 1/0 bzw. 0/1 (auch als "1.0"/"0.0") und umaResolutionStatus ==
    "resolved" (falls das Feld vorhanden und gefuellt ist).
    """
    if market.get("closed") is not True:
        return None
    status = market.get("umaResolutionStatus")
    if status not in (None, "") and status != "resolved":
        return None
    try:
        outcomes = market.get("outcomes")
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        if outcomes is not None and list(outcomes) != ["Yes", "No"]:
            return None
        prices = market.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else prices
        prices = [Decimal(str(x)) for x in prices]
    except Exception:
        return None
    if prices == [1, 0]:
        return "yes"
    if prices == [0, 1]:
        return "no"
    return None
