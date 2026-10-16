"""Black-76 ohne Abzinsung (Grundbausteine, keine Abhaengigkeiten im Paket)."""
from __future__ import annotations

import math

YEAR_SECONDS = 365.0 * 86400.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black76_call(F: float, K: float, T: float, sigma: float) -> float:
    """Nicht abgezinster Black-76-Call in USD."""
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    return F * norm_cdf(d1) - K * norm_cdf(d1 - sd)


def black76_price(F: float, K: float, T: float, sigma: float, call: bool) -> float:
    """Call oder Put (Put ueber Put-Call-Paritaet ohne Abzinsung)."""
    c = black76_call(F, K, T, sigma)
    return c if call else c - (F - K)


def black76_digital(F: float, K: float, T: float, sigma: float,
                    dsigma_dK: float = 0.0) -> float:
    """P(S>K) = N(d2) - F*phi(d1)*sqrt(T)*dsigma/dK (nicht abgezinst, ohne Beschneidung)."""
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return norm_cdf(d2) - F * norm_pdf(d1) * math.sqrt(T) * dsigma_dK


def digital_w(k: float, w: float, dw: float) -> float:
    """P(S>K) aus Gesamtvarianz ``w`` und ``dw/dk`` bei ``k = ln(K/F)``.

    Gleich ``black76_digital`` mit Smile, nur in (k, w) geschrieben:
    N(d2) - phi(d2) * w' / (2 sqrt(w)), d2 = (-k - w/2) / sqrt(w).
    """
    sw = math.sqrt(w)
    d2 = (-k - 0.5 * w) / sw
    return norm_cdf(d2) - norm_pdf(d2) * dw / (2.0 * sw)
