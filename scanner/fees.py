"""Gebuehren (Stand 15.10.2026, siehe docs/MACHBARKEIT.md 4.3). Alles in Decimal."""
from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

POLY_RATE = Decimal("0.07")
KALSHI_RATE = Decimal("0.07")
_FIVE = Decimal("0.00001")
_CENT = Decimal("0.01")


def _d(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def polymarket_taker_fee(shares, p, rate=POLY_RATE) -> Decimal:
    """Taker-Gebuehr: shares*rate*p*(1-p), auf 5 Nachkommastellen gerundet.

    Alles unter 0,00001 ergibt 0 (kleinster Betrag ist 0,00001).
    """
    raw = _d(shares) * _d(rate) * _d(p) * (1 - _d(p))
    fee = raw.quantize(_FIVE, rounding=ROUND_HALF_UP)
    return fee if fee >= _FIVE else Decimal("0")


def kalshi_fee(contracts, p, rate=KALSHI_RATE) -> Decimal:
    """Kalshi-Gebuehr je Order: auf den vollen Cent aufgerundet."""
    raw = _d(rate) * _d(contracts) * _d(p) * (1 - _d(p))
    return raw.quantize(_CENT, rounding=ROUND_CEILING)


def fee_per_share_estimate(platform: str, p) -> Decimal:
    """Ungerundete Gebuehr je Anteil (fuer die Stufenpruefung beim Orderbuch-Durchlauf)."""
    p = _d(p)
    return POLY_RATE * p * (1 - p) if platform == "polymarket" else KALSHI_RATE * p * (1 - p)


def order_fee(platform: str, shares, avg_price) -> Decimal:
    """Gebuehr einer ganzen Order."""
    if platform == "polymarket":
        return polymarket_taker_fee(shares, avg_price)
    return kalshi_fee(shares, avg_price)
