"""Risikoneutrale Wahrscheinlichkeiten aus Deribit-Optionen (docs/MACHBARKEIT.md 4.1, 4.2; Phase 3).

Drei Groessen je Markt, Auszahlung ``P(lower <= S < upper)`` (``None`` = unbeschraenkt):

- ``p_mid``: Punktschaetzung. SVI-Smile und Varianz-Uhr; ohne gueltigen Fit wie in
  Phase 2 (Mark-IV stueckweise linear, lineare Zeit).
- ``p_low``/``p_high``: Arbitrage- bzw. Huellband. Am Verfall modellfrei aus ausfuehrbaren
  Call- UND Put-Spreads, ausserhalb der Strikes einseitig ueber Monotonie. Zwischen zwei
  Verfaellen die Huelle der Endpunkte (wie Phase 2), vor dem ersten Verfall die Huelle
  aus "jetzt" (Kassakurs) und erstem Verfall.
- ``m_low``/``m_high``: Modellband. Spannweite ueber Smile-Fits (Mitte, Geld, Brief und
  zwei Kippungen um die Schwelle), Varianz-Uhr (flach, geschaetzt, Zuschlag) und
  Referenzverschiebung. Schmaler, aber ein Modell: Es gilt nur, wenn der Smile glatt
  (SVI) ist und die Varianz der Uhr folgt.

Optionspreise sind nicht abgezinste USD je Einheit Basiswert (Deribit-invers: BTC-Preis
mal Forward; USDC-linear: direkt).
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Optional, Sequence

from .black import (YEAR_SECONDS, black76_call, black76_digital, digital_w,  # noqa: F401
                    norm_cdf, norm_pdf)
from .svi import FitResult, fit_svi, implied_vol
from .varclock import FLAT, VarianceClock, theta_range

NEIGHBORS = 8  # Strikes je Seite fuer Spread-Schranken
EXACT_TOL = timedelta(minutes=5)
FIT_TOL = 0.005  # Fit-Digital darf so weit ausserhalb des Arbitragebands am Verfall liegen
CAL_TOL = 1e-6  # Kalender: w(T2, k) >= w(T1, k) - CAL_TOL
EXTRAP = 0.10  # Modellband hoechstens 10 % der Fit-Spanne ueber die aeussersten Quotes hinaus
LABELS = ("mid", "bid", "ask", "hoch", "runter", "mark")
MAX_REL_SPREAD = 0.5  # Fit nur mit Quotes, deren Spanne hoechstens 50 % des Mittelkurses ist
TILTS = ("hoch", "runter")
MAX_GAP = timedelta(days=8)  # groesster Verfallsabstand fuer eine Interpolation ...
GAP_RATIO = 1.5  # ... oder bis 1,5 x Restlaufzeit (Jahresendmaerkte)
LEE_SLOPE = 2.0  # groesste Fluegelsteigung der Gesamtvarianz (Lee)
FIT_RMSE_SLACK = 0.005  # Fit darf im Mittel hoechstens halbe IV-Spanne + 0,5 Vol-Punkte abweichen


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def digital_from_w(F: float, K: float, T: float, w: float, dw_dk: float) -> float:
    """Digital aus Gesamtvarianz w=sigma^2*T und deren Steigung nach k=ln(K/F)."""
    sigma = math.sqrt(w / T)
    dsigma_dK = dw_dk / (2.0 * sigma * T * K)
    return black76_digital(F, K, T, sigma, dsigma_dK)


@dataclass(frozen=True)
class ProbEstimate:
    """Punktschaetzung, Arbitrage-/Huellband und (wenn moeglich) Modellband."""
    p_mid: Optional[float]
    p_low: Optional[float]
    p_high: Optional[float]
    method: str  # "exakt" | "interpoliert" | "vorlauf"
    notes: tuple = ()
    m_low: Optional[float] = None
    m_high: Optional[float] = None


@dataclass(frozen=True)
class Slice:
    """Ein Deribit-Verfall. Strikes aufsteigend, ``iv`` = Mark-IV (Bruch), Quotes in USD."""
    expiry: datetime
    T: float
    F: float
    strikes: tuple
    iv: tuple
    call_bid: tuple  # Optional[float] je Strike
    call_ask: tuple
    put_bid: tuple = ()
    put_ask: tuple = ()
    min_amount: float = 0.1  # kleinste Ordergroesse in Einheiten Basiswert (fuer den Hedge)

    def __post_init__(self):
        n = len(self.strikes)
        for name in ("put_bid", "put_ask"):
            if not getattr(self, name):
                object.__setattr__(self, name, (None,) * n)

    def k_nodes(self) -> list:
        return [math.log(K / self.F) for K in self.strikes]

    def w_at(self, k: float):
        """(w, dw/dk): w linear in k aus Mark-IV interpoliert; None ausserhalb der Strikes.

        Die Steigung ist zwischen den zentralen Differenzen der Knoten linear interpoliert
        (stetig). Vorher war es die Steigung des Abschnitts; dann sprang das Digital an jedem
        Strike um bis zu 9 Punkte und schmale Spannen bekamen Scheinmasse (Reparatur 27.10.2026)."""
        ks = self.k_nodes()
        if len(ks) < 2 or k < ks[0] or k > ks[-1]:
            return None
        i = min(bisect_right(ks, k) - 1, len(ks) - 2)
        ws = (self.iv[i - 1] ** 2 * self.T if i > 0 else None, self.iv[i] ** 2 * self.T,
              self.iv[i + 1] ** 2 * self.T, self.iv[i + 2] ** 2 * self.T if i + 2 < len(ks) else None)
        seg = (ws[2] - ws[1]) / (ks[i + 1] - ks[i])
        s0 = seg if ws[0] is None else (ws[2] - ws[0]) / (ks[i + 1] - ks[i - 1])
        s1 = seg if ws[3] is None else (ws[3] - ws[1]) / (ks[i + 2] - ks[i])
        t = (k - ks[i]) / (ks[i + 1] - ks[i])
        return ws[1] + seg * (k - ks[i]), (1 - t) * s0 + t * s1

    def strike_range(self):
        return (self.strikes[0], self.strikes[-1]) if self.strikes else None


def point_at_slice(sl: Slice, K: float) -> Optional[float]:
    """Phase-2-Digital genau am Verfall (Mark-IV stueckweise linear)."""
    r = sl.w_at(math.log(K / sl.F))
    if r is None or r[0] <= 0:
        return None
    return digital_from_w(sl.F, K, sl.T, r[0], r[1])


# ------------------------------------------------- modellfreies Band am Verfall

def arb_digital_band(sl: Slice, K: float, neighbors: int = NEIGHBORS) -> tuple:
    """Ausfuehrbares Band (low, high) fuer P(S_T > K) aus Call- und Put-Spreads.

    low:  Spreads oberhalb von K. Call: (Geld K_i - Brief K_j)/dK; Put: 1 - (Brief K_j - Geld K_i)/dK.
    high: Spreads unterhalb von K. Call: (Brief K_i - Geld K_j)/dK; Put: 1 - (Geld K_j - Brief K_i)/dK.
    Ohne passende Quotes bleibt die Seite trivial (0 bzw. 1); das ist ehrlich, nur nutzlos.
    """
    n = len(sl.strikes)
    above = [i for i in range(n) if sl.strikes[i] >= K][:neighbors]
    below = [i for i in range(n) if sl.strikes[i] <= K][-neighbors:]
    lows, highs = [0.0], [1.0]
    for i, j in combinations(above, 2):
        dk = sl.strikes[j] - sl.strikes[i]
        if sl.call_bid[i] is not None and sl.call_ask[j] is not None:
            lows.append((sl.call_bid[i] - sl.call_ask[j]) / dk)
        if sl.put_bid[i] is not None and sl.put_ask[j] is not None:
            lows.append(1.0 - (sl.put_ask[j] - sl.put_bid[i]) / dk)
    for i, j in combinations(below, 2):
        dk = sl.strikes[j] - sl.strikes[i]
        if sl.call_ask[i] is not None and sl.call_bid[j] is not None:
            highs.append((sl.call_ask[i] - sl.call_bid[j]) / dk)
        if sl.put_ask[i] is not None and sl.put_bid[j] is not None:
            highs.append(1.0 - (sl.put_bid[j] - sl.put_ask[i]) / dk)
    return _clip(max(lows)), _clip(min(highs))


def band_at_slice(sl: Slice, K: float, neighbors: int = NEIGHBORS):
    """Wie Phase 2: Band oder None, wenn eine Seite trivial bleibt."""
    lo, hi = arb_digital_band(sl, K, neighbors)
    return None if (lo == 0.0 and hi == 1.0) else (lo, hi)


def _payoff_band(d_low: tuple, d_up: tuple) -> tuple:
    """Band fuer D(lower) - D(upper) aus den Bändern der beiden Digitals."""
    return _clip(d_low[0] - d_up[1]), _clip(d_low[1] - d_up[0])


ONE_BAND = (1.0, 1.0)
ZERO_BAND = (0.0, 0.0)


def arb_payoff_band(sl: Slice, lower, upper) -> tuple:
    dl = ONE_BAND if lower is None else arb_digital_band(sl, lower)
    du = ZERO_BAND if upper is None else arb_digital_band(sl, upper)
    return _payoff_band(dl, du)


# ------------------------------------------------------- Smile-Modell je Verfall

class SliceModel:
    """SVI-Fits eines Verfalls an Mitte, Geld und Brief (IV aus Optionen aus dem Geld) und
    zwei Kippungen: die Smile wird innerhalb der Geld-Brief-Spanne um den Drehpunkt (die
    Schwelle) gedreht, links Richtung Geld und rechts Richtung Brief bzw. umgekehrt. Das
    misst, wie unsicher die Steigung (der Skew) an der Schwelle ist. Fits werden
    zwischengespeichert."""

    def __init__(self, sl: Slice):
        self.sl = sl
        pts = []
        for i, K in enumerate(sl.strikes):
            call = K >= sl.F
            b = sl.call_bid[i] if call else sl.put_bid[i]
            a = sl.call_ask[i] if call else sl.put_ask[i]
            if b is None or a is None or a < b or (a - b) > MAX_REL_SPREAD * 0.5 * (a + b):
                continue  # ohne zweiseitigen, halbwegs engen Quote traegt der Punkt nichts bei
            ivb = implied_vol(b, sl.F, K, sl.T, call)
            iva = implied_vol(a, sl.F, K, sl.T, call)
            if ivb is None or iva is None:
                continue
            pts.append((math.log(K / sl.F), ivb, iva))
        self.pts = pts
        gaps = sorted(pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1))
        self._h = 2.0 * gaps[len(gaps) // 2] if gaps else 1.0
        self._cache: dict = {}

    def fit(self, label: str, pivot: Optional[float] = None) -> Optional[FitResult]:
        """``pivot``: Log-Moneyness des Drehpunkts (nur fuer die Kippungen)."""
        if label in TILTS:  # Drehpunkt auf ein Gitter (halbe Kippbreite) runden: weniger Fits
            pivot = round(pivot / (0.5 * self._h)) * 0.5 * self._h
        key = (label, round(pivot, 9) if label in TILTS else None)
        if key not in self._cache:
            ks = [p[0] for p in self.pts]
            mids = [0.5 * (p[1] + p[2]) for p in self.pts]
            halfs = [0.5 * (p[2] - p[1]) for p in self.pts]
            if label == "mid":
                ivs = mids
            elif label == "bid":
                ivs = [p[1] for p in self.pts]
            elif label == "ask":
                ivs = [p[2] for p in self.pts]
            else:  # Kippung: glatt (tanh) innerhalb der Spanne um den Drehpunkt drehen
                sign = 1.0 if label == "hoch" else -1.0
                ivs = [mids[i] + sign * halfs[i] * math.tanh((ks[i] - pivot) / self._h)
                       for i in range(len(ks))]
            wts = [1.0 / (p[2] - p[1] + 0.005) for p in self.pts]
            base = None if label == "mid" else self.fit("mid")
            self._cache[key] = fit_svi(ks, ivs, self.sl.T, wts,
                                       start=base.params if base is not None else None)
        return self._cache[key]

    def problem(self) -> Optional[str]:
        """Grund, warum das Modell unbrauchbar ist, sonst None."""
        for label in ("mid", "bid", "ask"):
            f = self.fit(label)
            if f is None:
                return "zu_wenig_quotes"
            if not f.ok:
                return "fit_arbitrage:" + ",".join(f.problems)
        half = sum(0.5 * (p[2] - p[1]) for p in self.pts) / len(self.pts)
        if self.fit("mid").rmse_iv > half + FIT_RMSE_SLACK:
            return "fit_ungenau"
        return None

    def mark_wd(self, k: float):
        """Zweites, unabhaengiges Smile-Modell: Deribits Mark-IV, w stueckweise linear, Steigung
        aus den zentralen Differenzen der Knoten linear interpoliert (ohne Spruenge)."""
        ks = self.sl.k_nodes()
        if len(ks) < 3:
            return None
        ws = [v * v * self.sl.T for v in self.sl.iv]
        slopes = []
        for i in range(len(ks)):
            a, b = max(i - 1, 0), min(i + 1, len(ks) - 1)
            slopes.append((ws[b] - ws[a]) / (ks[b] - ks[a]))
        if k < ks[0] or k > ks[-1]:  # ausserhalb: linear mit der Randsteigung (Lee: |w'| <= 2)
            j = 0 if k < ks[0] else -1
            sl = max(-LEE_SLOPE, min(LEE_SLOPE, slopes[j]))
            w = ws[j] + sl * (k - ks[j])
            return (w, sl) if w > 0 else None
        i = min(bisect_right(ks, k) - 1, len(ks) - 2)
        t = (k - ks[i]) / (ks[i + 1] - ks[i])
        return (1 - t) * ws[i] + t * ws[i + 1], (1 - t) * slopes[i] + t * slopes[i + 1]

    def in_range(self, k: float) -> bool:
        f = self.fit("mid")
        span = f.k_hi - f.k_lo
        return f.k_lo - EXTRAP * span <= k <= f.k_hi + EXTRAP * span

    def wd(self, label: str, k: float, pivot: float):
        if label == "mark":
            return self.mark_wd(k)
        f = self.fit(label, pivot)
        if f is None or not f.ok:
            return None
        return f.params.w(k), f.params.dw(k)


# -------------------------------------------------------------- Oberflaeche

@dataclass(frozen=True)
class Bracket:
    method: str  # "exakt" | "interpoliert" | "vorlauf"
    s1: Optional[Slice]  # None bei "vorlauf" (Anker: jetzt, Kassakurs, w = 0)
    s2: Optional[Slice]  # None bei "exakt"
    t0: datetime
    t1: Optional[datetime]
    F0: float


class Surface:
    """Alle Verfaelle eines Basiswerts plus Kassakurs und Varianz-Uhr."""

    def __init__(self, slices: Sequence[Slice], spot: Optional[float] = None,
                 clock: VarianceClock = FLAT, margin: float = 0.05):
        self.slices = sorted((s for s in slices if s.T > 0), key=lambda s: s.expiry)
        self.spot = spot
        self.clock = clock
        self.margin = margin
        self._models: dict = {}

    def model(self, sl: Slice) -> SliceModel:
        if sl.expiry not in self._models:
            self._models[sl.expiry] = SliceModel(sl)
        return self._models[sl.expiry]

    def bracket(self, t_star: datetime, now: datetime, tol: timedelta = EXACT_TOL):
        """(Bracket, None) oder (None, Grund)."""
        if not self.slices:
            return None, "keine_optionen"
        if t_star <= now:
            return None, "abgelaufen"
        near = min(self.slices, key=lambda s: abs(s.expiry - t_star))
        if abs(near.expiry - t_star) <= tol:
            return Bracket("exakt", near, None, near.expiry, None, near.F), None
        before = [s for s in self.slices if s.expiry < t_star]
        after = [s for s in self.slices if s.expiry > t_star]
        if not after:
            return None, "kein_verfall_daneben"
        if not before:
            if self.spot is None or self.spot <= 0:
                return None, "kein_verfall_daneben"
            return Bracket("vorlauf", None, after[0], now, after[0].expiry, self.spot), None
        return Bracket("interpoliert", before[-1], after[0], before[-1].expiry, after[0].expiry,
                       before[-1].F), None

    # ------------------------------------------------------------ Schaetzung
    def estimate(self, lower: Optional[float], upper: Optional[float], t_star: datetime,
                 now: datetime, delta: float = 0.002,
                 tol: timedelta = EXACT_TOL) -> Optional[ProbEstimate]:
        if (lower is None and upper is None) or (lower is not None and lower <= 0) \
                or (upper is not None and upper <= 0) \
                or (upper is not None and lower is not None and upper <= lower):
            return None
        br, _ = self.bracket(t_star, now, tol)
        if br is None:
            return None
        shifts = (1.0 - delta, 1.0, 1.0 + delta)
        notes = []
        model = self._model_band(br, lower, upper, t_star, now, shifts, notes)
        p_mid = model[0] if model else self._legacy_mid(br, lower, upper, t_star, now)
        hull = self._hull(br, lower, upper, shifts, p_mid, model)
        if hull is None:
            return None
        m_low, m_high = (model[1], model[2]) if model else (None, None)
        return ProbEstimate(p_mid, hull[0], hull[1], br.method, tuple(notes), m_low, m_high)

    def _hull(self, br: Bracket, lower, upper, shifts, p_mid, model):
        bands = []
        for s in shifts:
            lo = None if lower is None else lower * s
            up = None if upper is None else upper * s
            if br.method == "exakt":
                bands.append(arb_payoff_band(br.s1, lo, up))
                continue
            bands.append(arb_payoff_band(br.s2, lo, up))
            if br.method == "interpoliert":
                bands.append(arb_payoff_band(br.s1, lo, up))
            else:  # vorlauf: Zustand jetzt ist bekannt (Kassakurs), Band = Punkt
                d = lambda x: 1.0 if self.spot > x else 0.0
                v = (1.0 if lo is None else d(lo)) - (0.0 if up is None else d(up))
                bands.append((v, v))
        pts = [p for p in ([p_mid] + (list(model[3]) if model else [])) if p is not None]
        low = min([b[0] for b in bands] + pts)
        high = max([b[1] for b in bands] + pts)
        return _clip(low), _clip(high)

    def _legacy_mid(self, br: Bracket, lower, upper, t_star, now) -> Optional[float]:
        """Phase-2-Punkt (Mark-IV linear, lineare Zeit) als Rueckfall."""
        def dig(x):
            if br.method == "exakt":
                return point_at_slice(br.s1, x)
            theta = FLAT.theta(br.t0, t_star, br.t1)
            F = (1 - theta) * br.F0 + theta * br.s2.F
            k = math.log(x / F)
            r2 = br.s2.w_at(k)
            r1 = (0.0, 0.0) if br.s1 is None else br.s1.w_at(k)
            if r1 is None or r2 is None:
                return None
            w = (1 - theta) * r1[0] + theta * r2[0]
            dw = (1 - theta) * r1[1] + theta * r2[1]
            return digital_w(k, w, dw) if w > 0 else None
        dl = 1.0 if lower is None else dig(lower)
        du = 0.0 if upper is None else dig(upper)
        if dl is None or du is None:
            return None
        return _clip(dl - du)

    def _usable(self, s: Slice) -> bool:
        return self.model(s).problem() is None

    def _model_bracket(self, br: Bracket, t_star, now, notes):
        """Bracket fuer das Modellband: wie ``br``; hat ein Verfall keinen brauchbaren Smile
        (zu wenige Quotes, Fit mit Arbitrage), die naechsten brauchbaren Nachbarn."""
        used = [s for s in (br.s1, br.s2) if s is not None]
        bad = [self.model(s).problem() for s in used if not self._usable(s)]
        if not bad:
            return br
        notes.append(bad[0])
        if br.method == "exakt":
            return None
        after = [s for s in self.slices if s.expiry > t_star and self._usable(s)]
        before = [s for s in self.slices if s.expiry < t_star and self._usable(s)]
        if not after:
            return None
        limit = max(MAX_GAP, GAP_RATIO * (t_star - now))
        start = before[-1].expiry if before else now
        if after[0].expiry - start > limit or (not before and self.spot is None):
            notes.append("nachbarn_zu_weit")
            return None
        notes.append("modell_ueber_nachbarn")
        if before:
            return Bracket("interpoliert", before[-1], after[0], before[-1].expiry, after[0].expiry,
                           before[-1].F)
        return Bracket("vorlauf", None, after[0], now, after[0].expiry, self.spot)

    def _model_band(self, br: Bracket, lower, upper, t_star, now, shifts, notes):
        """(p_mid, m_low, m_high, Endpunkt-Punkte) oder None (Grund in ``notes``)."""
        br = self._model_bracket(br, t_star, now, notes)
        if br is None:
            return None
        used = [s for s in (br.s1, br.s2) if s is not None]
        models = [self.model(s) for s in used]
        bounds = [x for x in (lower, upper) if x is not None]
        # Fit am Verfall muss im modellfreien Band liegen (sonst widerspricht er den Quotes).
        for m in models:
            for x in bounds:
                k = math.log(x / m.sl.F)
                if not m.in_range(k):
                    notes.append("ausserhalb_fit")
                    return None
                r = m.wd("mid", k, None)
                lo, hi = arb_digital_band(m.sl, x)
                d = digital_w(k, r[0], r[1]) if r and r[0] > 0 else None
                if d is None or not lo - FIT_TOL <= d <= hi + FIT_TOL:
                    notes.append("fit_ausserhalb_arbband")
                    return None
        if br.method == "exakt":
            thetas, tc, theta_lin = [None], None, None
        else:
            tf, tc, tlo, thi = theta_range(self.clock, br.t0, t_star, br.t1, self.margin)
            thetas = sorted({tlo, tf, tc, thi})
            theta_lin = FLAT.theta(br.t0, t_star, br.t1)
        # Vorlauf: Smile-Form bei kurzer Laufzeit unsicher. "fest": w = theta*w2(k) (gleiche
        # Log-Moneyness); "skaliert": w = theta*w2(k/sqrt(theta)) (Smile skaliert mit sqrt(T),
        # dickere Raender). Das Band nimmt beide.
        shapes = ("fest", "skaliert") if br.method == "vorlauf" else ("fest",)
        # Ein Drehpunkt fuer beide Grenzen einer Spanne (sonst kaemen unteres und oberes Digital
        # aus zwei verschiedenen Kipp-Fits und die Spanne waere keine Masse einer Verteilung).
        pivot_base = math.sqrt(lower * upper) if lower is not None and upper is not None else bounds[0]
        if br.method == "vorlauf":
            # "skaliert" liest den Smile bei k/sqrt(theta), also 3- bis 5-mal weiter draussen, als
            # die Pruefungen oben reichen. Liegt das ausserhalb des Fits, faellt die Form weg
            # (Reparatur 27.10.2026: KXSOLE-26OCT1717-B99 bekam so p_mid 0,456 fuer 1 USD Spanne).
            pos_thetas = [t for t in thetas if t > 0]
            F_ = (1 - theta_lin) * br.F0 + theta_lin * br.s2.F
            if pos_thetas:
                c_max = 1.0 / math.sqrt(min(pos_thetas))
                ks = [math.log(x * sh / F_) for x in bounds + [pivot_base] for sh in shifts]
                if not all(models[-1].in_range(k * c_max) for k in ks):
                    shapes = ("fest",)
                    notes.append("skaliert_ausserhalb")

        def dig(label, base, s, theta, shape):
            """Digital bei Schwelle ``base * s``; der Kipp-Drehpunkt haengt nur an ``pivot_base``
            und liegt in derselben Moneyness wie die Auswertung. None, wenn das Digital nicht in
            [0, 1] liegt (der Fit ist dort unbrauchbar)."""
            d = _dig(label, base, s, theta, shape)
            if d is not None and not -1e-6 <= d <= 1.0 + 1e-6:
                if "digital_ungueltig" not in notes:
                    notes.append("digital_ungueltig")
                return None
            return d

        def _dig(label, base, s, theta, shape):
            if br.method == "exakt":
                m = models[0]
                k = math.log(base * s / m.sl.F)
                r = m.wd(label, k, math.log(pivot_base / m.sl.F))
                if r is None or r[0] <= 0:
                    return None
                return digital_w(k, r[0], r[1])
            F = (1 - theta_lin) * br.F0 + theta_lin * br.s2.F
            k = math.log(base * s / F)
            piv = math.log(pivot_base / F)
            m2 = models[-1]
            if br.method == "vorlauf":
                if theta <= 0:
                    return None
                c = 1.0 / math.sqrt(theta) if shape == "skaliert" else 1.0
                r2 = m2.wd(label, k * c, piv * c)
                if r2 is None:
                    return None
                w, dw = theta * r2[0], theta * c * r2[1]
            else:
                r2 = m2.wd(label, k, piv)
                r1 = models[0].wd(label, k, piv)
                if r1 is None or r2 is None:
                    return None
                if label == "mid" and r2[0] < r1[0] - CAL_TOL:
                    raise _CalendarArb()
                w = (1 - theta) * r1[0] + theta * r2[0]
                dw = (1 - theta) * r1[1] + theta * r2[1]
            if w <= 0:
                return None
            return digital_w(k, w, dw)

        def payoff(label, theta, s, shape="fest"):
            dl = 1.0 if lower is None else dig(label, lower, s, theta, shape)
            du = 0.0 if upper is None else dig(label, upper, s, theta, shape)
            if dl is None or du is None or dl < du - 1e-9:  # negative Masse: Arbitrage im Fit
                return None
            return _clip(dl - du)

        vals, missing = [], set()
        try:
            for label in LABELS:
                for theta in thetas:
                    for s in shifts:
                        for shape in shapes:
                            v = payoff(label, theta, s, shape)
                            if v is None:
                                missing.add(label)
                            else:
                                vals.append(v)
            mids = [payoff("mid", tc, 1.0, shape) for shape in shapes]
            p_mid = None if None in mids else sum(mids) / len(mids)
            ends = [] if br.method != "interpoliert" else [payoff("mid", 0.0, 1.0), payoff("mid", 1.0, 1.0)]
        except _CalendarArb:
            notes.append("kalender_arbitrage")
            return None
        if missing or p_mid is None:  # fehlt ein Mitglied, waere das Band still zu schmal
            notes.append("modell_unvollstaendig:" + ",".join(sorted(missing)))
            return None
        return p_mid, min(vals), max(vals), tuple(e for e in ends if e is not None)


class _CalendarArb(Exception):
    pass


def estimate_probability(slices: Sequence[Slice], K: float, t_star: datetime,
                         now: datetime, delta: float = 0.002, tol: timedelta = EXACT_TOL,
                         spot: Optional[float] = None, clock: VarianceClock = FLAT,
                         upper: Optional[float] = None) -> Optional[ProbEstimate]:
    """P(S > K) (bzw. P(K <= S < upper)) zum Zeitpunkt ``t_star``."""
    return Surface(slices, spot, clock).estimate(K, upper, t_star, now, delta, tol)
