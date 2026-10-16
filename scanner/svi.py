"""SVI-Smile (Gatheral, "raw"-Form) mit Fit und Arbitrage-Pruefung. Nur Standardbibliothek.

Gesamtvarianz ``w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))`` ueber der
Log-Moneyness ``k = ln(K/F)``. Gefittet wird quasi-explizit (Zeliade 2009): fuer feste
``(m, s)`` ist der Rest ein lineares Ausgleichsproblem in ``(a, d, c)`` mit
``c = b*s`` und ``d = rho*b*s``; ``(m, s)`` sucht ein kleines Nelder-Mead.

Arbitrage-Pruefung: Schmetterling ueber Durrlemans ``g(k) >= 0`` auf einem Gitter,
Fluegel ueber Lees Schranke ``b*(1+|rho|) <= 2``, positive Varianz. Kalender-Arbitrage
(``w`` steigt mit der Laufzeit) prueft ``prob`` zwischen zwei Fits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .black import black76_price

LEE_MAX = 2.0
G_TOL = -1e-9
MIN_POINTS = 7  # 5 Parameter: mit weniger als 7 Punkten passt SVI fast immer und sagt nichts


@dataclass(frozen=True)
class Svi:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k: float) -> float:
        x = k - self.m
        return self.a + self.b * (self.rho * x + math.sqrt(x * x + self.s * self.s))

    def dw(self, k: float) -> float:
        x = k - self.m
        return self.b * (self.rho + x / math.sqrt(x * x + self.s * self.s))

    def d2w(self, k: float) -> float:
        x = k - self.m
        return self.b * self.s * self.s / (x * x + self.s * self.s) ** 1.5

    def min_w(self) -> float:
        return self.a + self.b * self.s * math.sqrt(max(0.0, 1.0 - self.rho * self.rho))


def durrleman_g(p: Svi, k: float) -> float:
    """Dichte-Bedingung: g(k) >= 0 fuer alle k bedeutet keine Schmetterlings-Arbitrage."""
    w, w1, w2 = p.w(k), p.dw(k), p.d2w(k)
    if w <= 0:
        return -1.0
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def arbitrage_problems(p: Svi, k_lo: float, k_hi: float, n: int = 201) -> list:
    """Liste der verletzten Bedingungen (leer = arbitragefrei im Bereich)."""
    probs = []
    if p.b < 0 or abs(p.rho) >= 1 or p.s <= 0:
        probs.append("parameter")
    if p.b * (1 + abs(p.rho)) > LEE_MAX + 1e-12:
        probs.append("lee_fluegel")
    if p.min_w() <= 0:
        probs.append("varianz_negativ")
    span = k_hi - k_lo
    lo, hi = k_lo - 0.25 * span, k_hi + 0.25 * span
    for i in range(n):
        k = lo + (hi - lo) * i / (n - 1)
        if durrleman_g(p, k) < G_TOL:
            probs.append("schmetterling")
            break
    return probs


# ------------------------------------------------------------------ Fit

def _solve3(A, y):
    """Loest ein 3x3-System (Gauss mit Pivot). None bei Singularitaet."""
    M = [list(A[i]) + [y[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-18:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, 4):
                    M[r][j] -= f * M[c][j]
    return [M[i][3] / M[i][i] for i in range(3)]


def _wls(cols, ws, wt):
    """Gewichtete kleinste Quadrate fuer 1-3 Spalten. -> Koeffizienten oder None."""
    n = len(cols)
    A = [[sum(wt[i] * cols[r][i] * cols[c][i] for i in range(len(ws))) for c in range(n)] for r in range(n)]
    y = [sum(wt[i] * cols[r][i] * ws[i] for i in range(len(ws))) for r in range(n)]
    if n == 1:
        return [y[0] / A[0][0]] if A[0][0] > 0 else None
    if n == 2:
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-18:
            return None
        return [(y[0] * A[1][1] - A[0][1] * y[1]) / det, (A[0][0] * y[1] - A[1][0] * y[0]) / det]
    return _solve3(A, y)


def _inner(m: float, s: float, ks, ws, wt):
    """Bestes (a, d, c) fuer feste (m, s) unter c >= 0, |d| <= c, c + |d| <= 2s. -> (sse, Svi)."""
    ys = [(k - m) / s for k in ks]
    zs = [math.sqrt(y * y + 1) for y in ys]
    ones = [1.0] * len(ks)

    def sse(a, d, c):
        return sum(wt[i] * (a + d * ys[i] + c * zs[i] - ws[i]) ** 2 for i in range(len(ks)))

    def ok(a, d, c):
        return c >= 0 and abs(d) <= c + 1e-15 and c + abs(d) <= LEE_MAX * s + 1e-12 \
            and a + math.sqrt(max(c * c - d * d, 0.0)) > 0

    cands = []
    sol = _wls([ones, ys, zs], ws, wt)
    if sol and ok(*sol):
        cands.append(tuple(sol))
    else:
        for sign in (1.0, -1.0):  # Rand rho = +-1: w = a + c*(z + sign*y)
            sol2 = _wls([ones, [zs[i] + sign * ys[i] for i in range(len(ks))]], ws, wt)
            if sol2:
                a, c = sol2
                c = min(max(c, 0.0), LEE_MAX * s / 2)
                d = sign * c
                a = _wls([ones], [ws[i] - d * ys[i] - c * zs[i] for i in range(len(ks))], wt)[0]
                if ok(a, d, c):
                    cands.append((a, d, c))
        if sol:  # Lee-Rand: (c, d) skalieren, a neu
            a, d, c = sol
            c = max(c, 1e-12)
            d = max(-c, min(c, d))
            f = min(1.0, LEE_MAX * s / (c + abs(d)))
            c, d = c * f, d * f
            a = _wls([ones], [ws[i] - d * ys[i] - c * zs[i] for i in range(len(ks))], wt)[0]
            if ok(a, d, c):
                cands.append((a, d, c))
        a0 = _wls([ones], ws, wt)[0]  # flach
        cands.append((a0, 0.0, 1e-12))
    best = min(cands, key=lambda t: sse(*t))
    a, d, c = best
    b = c / s
    rho = d / c if c > 1e-15 else 0.0
    return sse(*best), Svi(a, b, max(-0.999999, min(0.999999, rho)), m, s)


def _nelder_mead(f, x0, step, iters: int = 160, tol: float = 1e-14):
    pts = [list(x0), [x0[0] + step[0], x0[1]], [x0[0], x0[1] + step[1]]]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(3), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[2] - vals[0]) < tol:
            break
        c = [(pts[0][j] + pts[1][j]) / 2 for j in range(2)]
        r = [c[j] + (c[j] - pts[2][j]) for j in range(2)]
        fr = f(r)
        if fr < vals[0]:
            e = [c[j] + 2 * (c[j] - pts[2][j]) for j in range(2)]
            fe = f(e)
            pts[2], vals[2] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[1]:
            pts[2], vals[2] = r, fr
        else:
            cc = [c[j] + 0.5 * (pts[2][j] - c[j]) for j in range(2)]
            fc = f(cc)
            if fc < vals[2]:
                pts[2], vals[2] = cc, fc
            else:
                for i in (1, 2):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(2)]
                    vals[i] = f(pts[i])
    i = min(range(3), key=lambda i: vals[i])
    return pts[i], vals[i]


@dataclass(frozen=True)
class FitResult:
    params: Svi
    rmse_iv: float  # mittlere Abweichung in Volatilitaetspunkten (Bruch)
    problems: tuple  # leer = arbitragefrei
    k_lo: float
    k_hi: float

    @property
    def ok(self) -> bool:
        return not self.problems


def fit_svi(ks: Sequence[float], ivs: Sequence[float], T: float,
            weights: Optional[Sequence[float]] = None,
            start: Optional[Svi] = None) -> Optional[FitResult]:
    """Fit an Volatilitaeten ``ivs`` (Bruch) bei Log-Moneyness ``ks``. Mindestens MIN_POINTS Punkte.

    ``start``: Warmstart (z. B. der Mittel-Fit fuer Geld-, Brief- und Kipp-Fits); dann genuegt
    ein Suchlauf statt sechs."""
    if len(ks) < MIN_POINTS or len(ks) != len(ivs) or T <= 0:
        return None
    ws = [v * v * T for v in ivs]
    wt = list(weights) if weights else [1.0] * len(ks)
    k_lo, k_hi = min(ks), max(ks)
    span = max(k_hi - k_lo, 1e-6)

    def obj(x):
        m, ls = x
        s = math.exp(ls)
        if not (k_lo - span <= m <= k_hi + span) or s > 10 * span or s < 1e-5 * span:
            return 1e30
        return _inner(m, s, ks, ws, wt)[0]

    if start is not None:
        starts = [(start.m, start.s, 0.02 * span, 0.1)]
    else:
        starts = [(m0, s0, 0.1 * span, 0.5) for m0 in (k_lo + 0.3 * span, 0.5 * (k_lo + k_hi),
                                                      k_hi - 0.3 * span) for s0 in (0.1 * span, 0.4 * span)]
    best = None
    for m0, s0, dm, dls in starts:
        x, v = _nelder_mead(obj, [m0, math.log(s0)], [dm, dls])
        if best is None or v < best[1]:
            best = (x, v)
    m, s = best[0][0], math.exp(best[0][1])
    _, p = _inner(m, s, ks, ws, wt)
    errs = []
    for k, v in zip(ks, ivs):
        wk = p.w(k)
        errs.append((math.sqrt(wk / T) - v) ** 2 if wk > 0 else 1.0)
    rmse = math.sqrt(sum(errs) / len(errs))
    return FitResult(p, rmse, tuple(arbitrage_problems(p, k_lo, k_hi)), k_lo, k_hi)


# --------------------------------------------------- implizite Volatilitaet

def implied_vol(price: float, F: float, K: float, T: float, call: bool,
                lo: float = 1e-3, hi: float = 10.0) -> Optional[float]:
    """Black-76-Volatilitaet aus einem nicht abgezinsten USD-Preis (Bisektion). None ohne Loesung."""
    if price is None or price <= 0 or T <= 0:
        return None
    intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
    upper = F if call else K
    if price <= intrinsic + 1e-12 or price >= upper:
        return None
    f_lo = black76_price(F, K, T, lo, call) - price
    f_hi = black76_price(F, K, T, hi, call) - price
    if f_lo > 0 or f_hi < 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if black76_price(F, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)
