"""Kuenstliche Deribit-Antworten im Originalformat (fuer das oeffentliche Repo).

Deribit-Marktdaten duerfen nicht weitergegeben werden. Das oeffentliche Repo enthaelt deshalb
keine echten Deribit-Fixtures; Tests nutzen stattdessen diese erzeugten Book-Summaries
(Black-76 mit einem schiefen Smile, Geld/Brief mit Spanne, Verfaelle wie am 15.10.2026).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from scanner.black import YEAR_SECONDS, black76_price

REF = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)
EXPIRIES = ["16OCT26", "17OCT26", "18OCT26", "19OCT26", "23OCT26", "30OCT26", "6NOV26", "27NOV26",
            "25DEC26", "26MAR27", "25JUN27", "24SEP27"]
_MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
                                         "OCT", "NOV", "DEC"])}
# Praefix -> (Forward, ATM-Vol, invers?, Tick, Strike-Raster, USDC-Verfaelle nur bis Dezember)
PARAMS = {
    "BTC": (82300.0, 0.45, True, 0.0001, 500.0),
    "BTC_USDC": (82300.0, 0.45, False, 5.0, 1000.0),
    "ETH": (2447.0, 0.55, True, 0.0001, 25.0),
    "SOL_USDC": (105.8, 0.60, False, 0.1, 1.0),
}


def _expiry(code: str) -> datetime:
    d, m, y = int(code[:-5]), _MON[code[-5:-2]], 2000 + int(code[-2:])
    return datetime(y, m, d, 8, 0, tzinfo=timezone.utc)


def _fmt_strike(K: float) -> str:
    return str(int(K)) if K == int(K) else f"{K:g}".replace(".", "d")


def book_summary(prefix: str, ref: datetime = REF) -> list:
    F0, vol, inverse, tick, raster = PARAMS[prefix]
    rows = []
    exps = EXPIRIES if inverse else EXPIRIES[:9]
    for code in exps:
        T = (_expiry(code) - ref).total_seconds() / YEAR_SECONDS
        if T <= 0:
            continue
        F = F0 * (1 + 0.02 * T)  # leichter Contango
        sd = vol * math.sqrt(T)
        step = max(raster, round(F * min(0.25 * sd, 0.05) / raster) * raster)
        k_lo, k_hi = F * math.exp(-4 * sd - 0.05), F * math.exp(4 * sd + 0.05)
        K = math.ceil(k_lo / step) * step
        while K <= k_hi:
            k = math.log(K / F)
            sig = vol * (1 - 0.35 * k + 1.5 * k * k)
            for kind in ("C", "P"):
                usd = black76_price(F, K, T, sig, kind == "C")
                px = usd / F if inverse else usd
                half = max(tick, 0.02 * px)
                bid = px - half
                rows.append({
                    "instrument_name": f"{prefix}-{code}-{_fmt_strike(K)}-{kind}",
                    "bid_price": round(bid, 8) if bid >= tick else None,
                    "ask_price": round(px + half, 8) if px > 0.2 * tick else None,
                    "mark_iv": round(sig * 100, 2), "underlying_price": round(F, 6)})
            K += step
    return rows


def payload(currency: str) -> dict:
    """Wie ``get_book_summary_by_currency``: BTC/ETH invers, USDC mit allen linearen Praefixen."""
    if currency == "USDC":
        return {"result": book_summary("BTC_USDC") + book_summary("SOL_USDC")}
    return {"result": book_summary(currency)}
