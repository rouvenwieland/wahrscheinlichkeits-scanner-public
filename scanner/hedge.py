"""Kosten einer Absicherung bei Deribit, je Prognosemarkt-Kontrakt (1 USD Auszahlung).

Ein Digital auf die Schwelle K wird mit einem Spread (Call- oder Put-Spread, K1 < K <= K2)
nachgebildet: 1 Kontrakt braucht 1/(K2-K1) Einheiten Basiswert je Bein. Gewaehlt wird der
billigste Spread aus bis zu ``WIDTH`` Strikes je Seite; ein breiterer Spread ist billiger je
Kontrakt, weicht aber im Wert vom Digital ab, und diese Abweichung zaehlt als Kosten.
Kosten je Kontrakt:

- Eroeffnung: Deribit-Gebuehr je Bein ``min(RATE * F, CAP * Optionspreis)`` je Einheit.
  Kombi-Rabatt laut Deribit-Gebuehrenseite (Stand 14.10.2026): "Whichever direction has the
  lowest fees, the fees for the legs in that direction are reduced to zero" -> beim Spread
  zahlt nur das teurere Bein (``COMBO_ONE_LEG``).
- Ende: Faellt die Abrechnung auf den Verfall, Ausuebungsgebuehr 0,015 % (Deckel 12,5 %),
  schlimmster Fall beide Beine im Geld; Tagesoptionen sind davon befreit ("daily options ...
  are exempt from delivery fees"; Freitags-Verfaelle sind Wochen-/Monatsoptionen).
  Sonst muss der Spread zum Abrechnungszeitpunkt geschlossen werden:
  noch einmal Gebuehr plus halbe Geld-Brief-Spanne beider Beine (heutige Spanne als Schaetzer).
- Fuer Kanten gegen das Modellband (Mittelkurse) zusaetzlich die halbe Spanne beim Eroeffnen;
  das Arbitrageband enthaelt sie schon (es rechnet mit Geld und Brief).

Mindestgroesse: ``min_amount`` Einheiten je Bein -> mindestens ``ceil(min_amount * (K2-K1))``
Kontrakte, sonst passt der Hedge nicht zur Position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .prob import Slice

FEE_RATE = 0.0003  # taker_commission laut get_instruments (BTC, ETH, USDC-Optionen)
FEE_CAP = 0.125
SETTLE_RATE = 0.00015
SETTLE_CAP = 0.125
COMBO_ONE_LEG = True
WIDTH = 4  # Strikes je Seite, aus denen der Hedge-Spread gewaehlt wird


@dataclass(frozen=True)
class HedgeCost:
    per_contract: float  # USD je Kontrakt
    n_min: int
    legs: tuple  # ((K1, K2, "C"|"P"), ...)
    parts: dict = field(default_factory=dict)


def _leg_quotes(sl: Slice, i: int, kind: str):
    if kind == "C":
        return sl.call_bid[i], sl.call_ask[i]
    return sl.put_bid[i], sl.put_ask[i]


def _fee(F: float, price: float) -> float:
    return min(FEE_RATE * F, FEE_CAP * price)


def _digital_value(kind: str, m1: float, m2: float, dk: float) -> float:
    """Wert des Spreads je Kontrakt als Digital P(S > K) aus Mittelkursen."""
    return (m1 - m2) / dk if kind == "C" else 1.0 - (m2 - m1) / dk


def digital_hedge(sl: Slice, K: float, at_expiry: bool, include_open_spread: bool,
                  width: int = WIDTH) -> Optional[HedgeCost]:
    """Billigster Spread K1 < K <= K2 (Call oder Put) aus bis zu ``width`` Strikes je Seite.

    Breitere Spreads senken Gebuehren und Spannen je Kontrakt, bilden das Digital aber nur
    naeherungsweise nach. Diese Wertabweichung zaehlt als Kosten: |Wert(K1,K2) - Wert(Nachbarn)|
    aus Mittelkursen. None, wenn kein Spread quotiert ist."""
    n = len(sl.strikes)
    j0 = next((i for i in range(n) if sl.strikes[i] >= K), None)
    if j0 is None or j0 == 0:
        return None
    best = None
    for kind in ("C", "P"):
        q = [_leg_quotes(sl, i, kind) for i in range(n)]
        mid = [None if None in x else 0.5 * (x[0] + x[1]) for x in q]
        ref = None
        if mid[j0 - 1] is not None and mid[j0] is not None:
            ref = _digital_value(kind, mid[j0 - 1], mid[j0], sl.strikes[j0] - sl.strikes[j0 - 1])
        for i in range(max(0, j0 - width), j0):
            for j in range(j0, min(n, j0 + width)):
                if mid[i] is None or mid[j] is None:
                    continue
                if (i, j) != (j0 - 1, j0) and ref is None:
                    continue  # ohne Nachbar-Referenz nur der Nachbar-Spread
                (b1, a1), (b2, a2) = q[i], q[j]
                dk = sl.strikes[j] - sl.strikes[i]
                f1, f2 = _fee(sl.F, mid[i]), _fee(sl.F, mid[j])
                open_fee = max(f1, f2) if COMBO_ONE_LEG else f1 + f2
                half_spreads = 0.5 * (a1 - b1) + 0.5 * (a2 - b2)
                if at_expiry:
                    daily = sl.expiry.weekday() != 4
                    exit_cost = 0.0 if daily else 2 * SETTLE_RATE * sl.F  # schlimmster Fall: beide im Geld
                else:
                    exit_cost = open_fee + half_spreads
                spread_open = half_spreads if include_open_spread else 0.0
                bias = 0.0 if ref is None else abs(_digital_value(kind, mid[i], mid[j], dk) - ref)
                total = (open_fee + exit_cost + spread_open) / dk + bias
                if best is None or total < best.per_contract:
                    best = HedgeCost(total, math.ceil(sl.min_amount * dk - 1e-9),
                                     ((sl.strikes[i], sl.strikes[j], kind),),
                                     {"eroeffnung": open_fee / dk, "ende": exit_cost / dk,
                                      "spanne_eroeffnung": spread_open / dk, "abweichung": bias})
    return best


def market_hedge(sl: Slice, lower, upper, at_expiry: bool, include_open_spread: bool) -> Optional[HedgeCost]:
    """Kosten fuer P(lower <= S < upper): je endlicher Grenze ein Spread, Kosten addiert."""
    parts = [digital_hedge(sl, x, at_expiry, include_open_spread) for x in (lower, upper) if x is not None]
    if not parts or any(p is None for p in parts):
        return None
    merged = {}
    for p in parts:
        for k, v in p.parts.items():
            merged[k] = merged.get(k, 0.0) + v
    return HedgeCost(sum(p.per_contract for p in parts), max(p.n_min for p in parts),
                     tuple(l for p in parts for l in p.legs), merged)
