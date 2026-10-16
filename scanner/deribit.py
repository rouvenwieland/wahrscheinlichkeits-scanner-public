"""Deribit: oeffentliche Optionsdaten (kein Konto, kein Schluessel).

Inverse Optionen (``BTC-25DEC26-104000-C``) kosten Coin; ihr USD-Wert ist Preis mal
Forward (``underlying_price``). Lineare USDC-Optionen (``SOL_USDC-23OCT26-103-P``,
Dezimalstrikes mit ``d``: ``XRP_USDC-23OCT26-1d24-P``) kosten direkt USDC. Beide werden
in nicht abgezinste USD je Einheit Basiswert umgerechnet und je Strike zum besten Geld
und Brief zusammengefuehrt (beide sind ausfuehrbar).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .black import YEAR_SECONDS
from .http import FetchError, get_json
from .prob import Slice

BASE = "https://www.deribit.com/api/v2"
_NAME = re.compile(r"^(?P<cur>[A-Z]+)(?P<lin>_USDC)?-(?P<d>\d{1,2})(?P<m>[A-Z]{3})(?P<y>\d{2})-"
                   r"(?P<k>\d+(?:[.d]\d+)?)-(?P<t>[CP])$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
# Mindestgroessen (Basiswert-Einheiten) laut get_instruments vom 15.10.2026; live ueberschreibbar.
MIN_AMOUNT = {"BTC": 0.1, "ETH": 1.0, "BTC_USDC": 0.01, "ETH_USDC": 0.1, "SOL_USDC": 10.0,
              "XRP_USDC": 1000.0}


@dataclass(frozen=True)
class DeribitOption:
    name: str
    expiry: datetime  # 08:00 UTC
    strike: float
    kind: str  # "C" | "P"
    bid: Optional[float]  # USD je Einheit Basiswert; None wenn kein Gebot
    ask: Optional[float]
    mark_iv: Optional[float]  # Bruch (0,38 statt 38 %)
    underlying: float  # Forward des Verfalls in USD
    venue: str = ""  # Instrument-Praefix, z. B. "BTC" oder "BTC_USDC"


def parse_instrument(name: str):
    """'BTC-25DEC26-104000-C' -> (expiry 08:00 UTC, strike, kind, praefix) oder None."""
    m = _NAME.match(name)
    if not m or m["m"] not in _MONTHS:
        return None
    expiry = datetime(2000 + int(m["y"]), _MONTHS[m["m"]], int(m["d"]), 8, 0, tzinfo=timezone.utc)
    return expiry, float(m["k"].replace("d", ".")), m["t"], m["cur"] + (m["lin"] or "")


def _price(x) -> Optional[float]:
    return float(x) if x is not None and float(x) > 0 else None


def parse_book_summary(payload: dict, prefix: Optional[str] = None) -> list:
    """API-Antwort -> ``DeribitOption`` (Preise in USD). ``prefix`` filtert z. B. "SOL_USDC".

    Antwort ohne Liste -> FetchError; ohne Praefix auch, wenn nichts lesbar ist."""
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, list) or not result:
        raise FetchError("Deribit: Antwort ohne 'result'-Liste")
    out = []
    for r in result:
        p = parse_instrument(r.get("instrument_name", ""))
        if p is None or not r.get("underlying_price"):
            continue
        if prefix is not None and p[3] != prefix:
            continue
        F = float(r["underlying_price"])
        mult = 1.0 if p[3].endswith("_USDC") else F  # invers: Coin-Preis mal Forward
        bid, ask = _price(r.get("bid_price")), _price(r.get("ask_price"))
        iv = r.get("mark_iv")
        out.append(DeribitOption(
            name=r["instrument_name"], expiry=p[0], strike=p[1], kind=p[2],
            bid=None if bid is None else bid * mult, ask=None if ask is None else ask * mult,
            mark_iv=float(iv) / 100.0 if iv else None, underlying=F, venue=p[3]))
    if not out and prefix is None:
        raise FetchError("Deribit: keine Optionen lesbar")
    return out  # mit Praefix darf ein Handelsplatz leer sein; der Aufrufer prueft das Ganze


def fetch_options(currency: str = "BTC", prefix: Optional[str] = None) -> list:
    payload = get_json(f"{BASE}/public/get_book_summary_by_currency",
                       {"currency": currency, "kind": "option"})
    return parse_book_summary(payload, prefix if prefix is not None else currency)


def fetch_index(index_name: str) -> float:
    payload = get_json(f"{BASE}/public/get_index_price", {"index_name": index_name})
    try:
        v = float(payload["result"]["index_price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FetchError(f"Deribit: Index {index_name} nicht lesbar") from exc
    if v <= 0:
        raise FetchError(f"Deribit: Index {index_name} <= 0")
    return v


def fetch_candles(instrument: str, start_ms: int, end_ms: int, resolution: str = "60"):
    """Stundenkerzen (Startzeiten ms, Schlusskurse) fuer die Varianz-Uhr."""
    payload = get_json(f"{BASE}/public/get_tradingview_chart_data",
                       {"instrument_name": instrument, "resolution": resolution,
                        "start_timestamp": start_ms, "end_timestamp": end_ms})
    r = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(r, dict) or r.get("status") not in ("ok", None) or "ticks" not in r:
        raise FetchError(f"Deribit: Kerzen fuer {instrument} nicht lesbar")
    return list(r["ticks"]), [float(c) for c in r["close"]]


def fetch_min_amounts(currency: str) -> dict:
    """{Praefix: kleinste Ordergroesse} aus get_instruments (oeffentlich)."""
    payload = get_json(f"{BASE}/public/get_instruments",
                       {"currency": currency, "kind": "option", "expired": "false"})
    out = {}
    for r in payload.get("result") or []:
        p = parse_instrument(r.get("instrument_name", ""))
        if p and r.get("min_trade_amount"):
            out[p[3]] = min(out.get(p[3], 1e18), float(r["min_trade_amount"]))
    return out


def fetch_order_book(instrument: str, depth: int = 5) -> dict:
    payload = get_json(f"{BASE}/public/get_order_book",
                       {"instrument_name": instrument, "depth": depth})
    if "result" not in payload:
        raise FetchError("Deribit: Orderbuch ohne 'result'")
    return payload["result"]


def _better(cur, new, pick):
    if new is None:
        return cur
    return new if cur is None else pick(cur, new)


def build_slices(options: list, now: datetime, min_strikes: int = 4,
                 min_amounts: Optional[dict] = None) -> list:
    """Gruppiert nach Verfall. Je Strike bestes Geld/Brief fuer Call und Put ueber alle
    Handelsplaetze; Mark-IV bevorzugt von der Option aus dem Geld."""
    mins = dict(MIN_AMOUNT, **(min_amounts or {}))
    by_exp: dict = {}
    for o in options:
        by_exp.setdefault(o.expiry, []).append(o)
    slices = []
    for expiry, opts in by_exp.items():
        T = (expiry - now).total_seconds() / YEAR_SECONDS
        if T <= 0:
            continue
        F = sorted(o.underlying for o in opts)[len(opts) // 2]
        strikes: dict = {}
        for o in opts:
            e = strikes.setdefault(o.strike, {"ivc": None, "ivp": None, "cb": None, "ca": None,
                                              "pb": None, "pa": None})
            if o.kind == "C":
                e["cb"] = _better(e["cb"], o.bid, max)
                e["ca"] = _better(e["ca"], o.ask, min)
                e["ivc"] = e["ivc"] or o.mark_iv
            else:
                e["pb"] = _better(e["pb"], o.bid, max)
                e["pa"] = _better(e["pa"], o.ask, min)
                e["ivp"] = e["ivp"] or o.mark_iv
        ks, ivs, cb, ca, pb, pa = [], [], [], [], [], []
        for K in sorted(strikes):
            e = strikes[K]
            iv = (e["ivc"] or e["ivp"]) if K >= F else (e["ivp"] or e["ivc"])
            if not iv:
                continue
            ks.append(K)
            ivs.append(iv)
            cb.append(e["cb"])
            ca.append(e["ca"])
            pb.append(e["pb"])
            pa.append(e["pa"])
        # Geld/Brief werden ueber die Plaetze gemischt; welcher Platz ein Hedge-Bein stellt, ist
        # danach unbekannt. Deshalb die groesste Mindestmenge der beteiligten Plaetze (vorsichtig).
        venues = {o.venue for o in opts}
        min_amt = max((mins[v] for v in venues if v in mins), default=0.1)
        if len(ks) >= min_strikes:
            slices.append(Slice(expiry, T, F, tuple(ks), tuple(ivs), tuple(cb), tuple(ca),
                                tuple(pb), tuple(pa), min_amt))
    return sorted(slices, key=lambda s: s.expiry)
