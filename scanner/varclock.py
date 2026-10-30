"""Varianz-Uhr fuer die Zeitinterpolation zwischen zwei Verfaellen.

Die Gesamtvarianz zwischen zwei Deribit-Verfaellen waechst nicht gleichmaessig: Krypto
schwankt zu US-Handelszeiten staerker als nachts und am Wochenende weniger. Die Uhr
verteilt die Varianz einer Woche auf 168 Wochenstunden (UTC, Montag 0 Uhr = Index 0),
Mittelwert 1. ``theta(t0, t, t1)`` ist der Anteil der Varianz von ``t0`` bis ``t1``,
der bis ``t`` anfaellt. Die flache Uhr ergibt die lineare Zeitinterpolation aus Phase 2.

Geschaetzt wird aus stuendlichen Kerzen des Perpetuals (oeffentliche Deribit-Daten):
Tagesprofil (24 Stunden) mal Wochentag/Wochenende, zur flachen Uhr hin geschrumpft.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

HOURS = 168
EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)  # ein Montag
MIN_DAYS = 14


@dataclass(frozen=True)
class VarianceClock:
    weights: tuple  # 168 Werte, Mittel 1
    source: str = "flach"

    def __post_init__(self):
        if len(self.weights) != HOURS or min(self.weights) <= 0:
            raise ValueError("Varianz-Uhr braucht 168 positive Gewichte")
        cum, acc = [], 0.0
        for w in self.weights:
            cum.append(acc)
            acc += w
        object.__setattr__(self, "_cum", tuple(cum))
        object.__setattr__(self, "_week", acc)

    def clock(self, t: datetime) -> float:
        """Kumulierte Varianz-Stunden seit EPOCH (streng monoton)."""
        h = (t - EPOCH).total_seconds() / 3600.0
        weeks = math.floor(h / HOURS)
        rest = h - weeks * HOURS
        idx = min(int(rest), HOURS - 1)
        return weeks * self._week + self._cum[idx] + (rest - idx) * self.weights[idx]

    def integral(self, t0: datetime, t1: datetime) -> float:
        return self.clock(t1) - self.clock(t0)

    def theta(self, t0: datetime, t: datetime, t1: datetime) -> float:
        """Varianzanteil von [t0, t] an [t0, t1], in [0, 1]."""
        total = self.integral(t0, t1)
        if total <= 0:
            raise ValueError("t1 muss nach t0 liegen")
        return min(1.0, max(0.0, self.integral(t0, t) / total))


FLAT = VarianceClock((1.0,) * HOURS)


def hour_of_week(t: datetime) -> int:
    t = t.astimezone(timezone.utc)
    return t.weekday() * 24 + t.hour


def estimate_clock(ticks_ms: Sequence[int], closes: Sequence[float], shrink: float = 0.3,
                   source: str = "geschaetzt") -> VarianceClock:
    """Uhr aus Stundenkerzen (Startzeit in ms, Schlusskurs). Zu wenig Daten -> FLAT.

    Gewicht je Stunde ~ (mittlere |Rendite|)^2, robuster als der Mittelwert der Quadrate.
    """
    hod_sum, hod_n = [0.0] * 24, [0] * 24
    day_sum, day_n = [0.0, 0.0], [0, 0]  # Werktag, Wochenende
    days = set()
    for i in range(1, min(len(ticks_ms), len(closes))):
        if ticks_ms[i] - ticks_ms[i - 1] != 3_600_000 or closes[i] <= 0 or closes[i - 1] <= 0:
            continue
        t = datetime.fromtimestamp(ticks_ms[i] / 1000, tz=timezone.utc)
        r = abs(math.log(closes[i] / closes[i - 1]))
        hod_sum[t.hour] += r
        hod_n[t.hour] += 1
        we = 1 if t.weekday() >= 5 else 0
        day_sum[we] += r
        day_n[we] += 1
        days.add(t.date())
    if len(days) < MIN_DAYS or min(hod_n) == 0 or min(day_n) == 0:
        return FLAT
    hod = [(hod_sum[h] / hod_n[h]) ** 2 for h in range(24)]
    mean_hod = sum(hod) / 24
    if mean_hod <= 0:
        return FLAT
    day = [(day_sum[j] / day_n[j]) ** 2 for j in range(2)]
    raw = [hod[h] / mean_hod * day[1 if d >= 5 else 0] for d in range(7) for h in range(24)]
    m = sum(raw) / HOURS
    raw = [(1 - shrink) * x / m + shrink for x in raw]
    m = sum(raw) / HOURS
    return VarianceClock(tuple(x / m for x in raw), source)


MAX_REL_DOWN = 0.3  # theta_min >= 70 % des kleineren Uhr-Werts


def theta_range(clock: VarianceClock, t0: datetime, t: datetime, t1: datetime,
                margin: float = 0.05) -> tuple:
    """(theta_flach, theta_uhr, theta_min, theta_max) fuer das Modellband.

    Der Bereich umfasst flache und geschaetzte Uhr plus einen Zuschlag fuer Ereignisse
    (z. B. US-Daten um 12:30 UTC), die keine Uhr kennt. Der Zuschlag ist ``margin`` in
    der Mitte und schrumpft zu den Verfaellen hin (Faktor 2*sqrt(theta*(1-theta))),
    weil dort fast die ganze Varianz schon feststeht.

    Nach unten hoechstens ``MAX_REL_DOWN`` relativ: Der absolute Zuschlag wuerde bei kleinem
    theta (Vorlauf, wenige Stunden) theta_min auf etwa 0 druecken, also eine Varianz von fast 0
    unterstellen (Reparatur 27.10.2026: Modellband 0 bis 1 bei Spannen um den Kassakurs).
    """
    tf = FLAT.theta(t0, t, t1)
    tc = clock.theta(t0, t, t1)
    c = 0.5 * (tf + tc)
    m = margin * 2.0 * math.sqrt(max(c * (1.0 - c), 0.0))
    lo = max(0.0, min(tf, tc) - m, min(tf, tc) * (1.0 - MAX_REL_DOWN))
    hi = min(1.0, max(tf, tc) + m)
    return tf, tc, lo, hi


def hours(td: timedelta) -> float:
    return td.total_seconds() / 3600.0
