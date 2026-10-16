import math
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, strategies as st

from scanner.black import black76_price, digital_w
from scanner.prob import (Slice, Surface, YEAR_SECONDS, arb_digital_band, band_at_slice,
                          black76_call, black76_digital, estimate_probability, norm_cdf,
                          point_at_slice)
from scanner.svi import Svi

NOW = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)


def make_slice(days, F=80000.0, sig=lambda K, F: 0.5, half_spread=0.0, lo=0.4, hi=1.8, step=1000.0,
               now=NOW, rel_spread=0.0, puts=True):
    """Slice aus Black-76-Preisen in USD. ``half_spread`` als Anteil von F (wie frueher in BTC)."""
    expiry = now + timedelta(days=days)
    T = days * 86400 / YEAR_SECONDS
    strikes, iv, cb, ca, pb, pa = [], [], [], [], [], []
    K = round(F * lo / step) * step
    while K <= F * hi:
        s = sig(K, F)
        c, p = black76_price(F, K, T, s, True), black76_price(F, K, T, s, False)
        hs_c, hs_p = half_spread * F + rel_spread * c, half_spread * F + rel_spread * p
        strikes.append(K); iv.append(s)
        cb.append(max(c - hs_c, 1e-9)); ca.append(c + hs_c)
        pb.append(max(p - hs_p, 1e-9) if puts else None); pa.append(p + hs_p if puts else None)
        K += step
    return Slice(expiry, T, F, tuple(strikes), tuple(iv), tuple(cb), tuple(ca), tuple(pb), tuple(pa))


def svi_slice(days, params: Svi, F=80000.0, step=500.0, rel_spread=0.01, abs_spread=2.0, width=0.25):
    """Slice aus einem SVI-Smile (Gesamtvarianz je Jahr skaliert mit T), Quotes mit Spanne."""
    T = days * 86400 / YEAR_SECONDS
    strikes, iv, cb, ca, pb, pa = [], [], [], [], [], []
    K = round(F * (1 - width) / step) * step
    while K <= F * (1 + width):
        k = math.log(K / F)
        s = math.sqrt(params.w(k) / T)
        c, p = black76_price(F, K, T, s, True), black76_price(F, K, T, s, False)
        strikes.append(K); iv.append(s)
        for x, b, a in ((c, cb, ca), (p, pb, pa)):
            h = abs_spread + rel_spread * x
            b.append(max(x - h, 1e-9)); a.append(x + h)
        K += step
    return Slice(NOW + timedelta(days=days), T, F, tuple(strikes), tuple(iv), tuple(cb), tuple(ca),
                 tuple(pb), tuple(pa))


def svi_digital(params: Svi, F, K):
    k = math.log(K / F)
    return digital_w(k, params.w(k), params.dw(k))


flat = st.floats(0.15, 1.5)
fwd = st.floats(20000, 200000)
tyr = st.floats(0.002, 2.0)
mny = st.floats(0.5, 2.0)


@given(F=fwd, m=mny, T=tyr, s=flat)
def test_digital_in_unit_interval_flat(F, m, T, s):
    p = black76_digital(F, F * m, T, s)
    assert 0.0 <= p <= 1.0


@given(F=fwd, T=tyr, s=flat, a=mny, b=mny)
def test_digital_monotone_flat(F, T, s, a, b):
    lo, hi = sorted((a, b))
    assert black76_digital(F, F * lo, T, s) >= black76_digital(F, F * hi, T, s) - 1e-12


@given(F=fwd, m=mny, T=tyr, s=flat)
def test_digital_equals_nd2_flat(F, m, T, s):
    d2 = (math.log(1 / m) - 0.5 * s * s * T) / (s * math.sqrt(T))
    assert abs(black76_digital(F, F * m, T, s) - norm_cdf(d2)) < 1e-12


@given(F=fwd, m=st.floats(0.7, 1.4), T=st.floats(0.02, 1.0), a=st.floats(0.3, 0.8),
       b=st.floats(-0.3, 0.3), c=st.floats(0.0, 0.3))
@settings(max_examples=60)
def test_breeden_litzenberger_consistency(F, m, T, a, b, c):
    """Numerische Ableitung der Black-76-Preise mit Smile entspricht der Digital-Formel."""
    K0 = F * m
    sig = lambda K: a + b * math.log(K / F) + c * math.log(K / F) ** 2
    if min(sig(K0 * 0.9), sig(K0), sig(K0 * 1.1)) < 0.1:
        return
    h = K0 * 1e-4
    price = lambda K: black76_call(F, K, T, sig(K))
    numeric = -(price(K0 + h) - price(K0 - h)) / (2 * h)
    dsig = (sig(K0 + h) - sig(K0 - h)) / (2 * h)
    formula = black76_digital(F, K0, T, sig(K0), dsig)
    assert abs(numeric - formula) < 1e-5


@given(F=fwd, m=st.floats(0.6, 1.6), T=st.floats(0.002, 1.0), s=flat, dw=st.floats(-0.05, 0.05))
def test_digital_w_equals_black_form(F, m, T, s, dw):
    """digital_w(k, w, w') ist dieselbe Formel wie black76_digital mit dsigma/dK."""
    K = F * m
    w = s * s * T
    dsig_dK = dw / (2 * s * T * K)
    assert abs(digital_w(math.log(K / F), w, dw) - black76_digital(F, K, T, s, dsig_dK)) < 1e-9


def test_slice_digital_close_to_flat_value():
    sl = make_slice(3.0)
    K = 85000.0
    assert abs(point_at_slice(sl, K) - black76_digital(sl.F, K, sl.T, 0.5)) < 1e-9


def test_slice_digital_matches_smile_derivative():
    smile = lambda K, F: 0.5 - 0.2 * math.log(K / F) + 0.3 * math.log(K / F) ** 2
    sl = make_slice(5.0, sig=smile, step=250.0)
    K = 84000.0
    h = 1.0
    price = lambda x: black76_call(sl.F, x, sl.T, smile(x, sl.F))
    numeric = -(price(K + h) - price(K - h)) / (2 * h)
    assert abs(point_at_slice(sl, K) - numeric) < 0.01


def test_no_extrapolation_in_legacy_point():
    sl = make_slice(3.0)
    assert point_at_slice(sl, sl.strikes[-1] * 1.5) is None
    assert point_at_slice(sl, sl.strikes[0] * 0.5) is None


# ------------------------------------------------- modellfreies Band (Calls und Puts)

@given(m=st.floats(0.8, 1.25), s=st.floats(0.25, 0.9), hs=st.floats(0.0, 0.0008),
       days=st.floats(1.0, 20.0), puts_only=st.booleans())
@settings(max_examples=60, deadline=None)
def test_arb_band_contains_true_digital(m, s, hs, days, puts_only):
    sl = make_slice(days, sig=lambda K, F: s, half_spread=hs)
    if puts_only:  # nur Puts quotiert: Band muss trotzdem stimmen
        n = len(sl.strikes)
        sl = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv, (None,) * n, (None,) * n, sl.put_bid, sl.put_ask)
    K = sl.F * m
    truth = black76_digital(sl.F, K, sl.T, s)
    lo, hi = arb_digital_band(sl, K)
    assert lo - 1e-6 <= truth <= hi + 1e-6
    assert band_at_slice(sl, K) is not None


@given(a=st.floats(-0.3, 0.3), days=st.floats(1.0, 30.0), hs=st.floats(0.0, 0.0005))
@settings(max_examples=40, deadline=None)
def test_arb_band_with_skew_contains_truth(a, days, hs):
    smile = lambda K, F: 0.55 + a * math.log(K / F) + 0.4 * math.log(K / F) ** 2
    sl = make_slice(days, sig=smile, half_spread=hs, step=500.0)
    K = sl.F * 1.03
    h = 0.5
    truth = -(black76_call(sl.F, K + h, sl.T, smile(K + h, sl.F))
              - black76_call(sl.F, K - h, sl.T, smile(K - h, sl.F))) / (2 * h)
    lo, hi = arb_digital_band(sl, K)
    assert lo - 1e-6 <= truth <= hi + 1e-6


def test_puts_tighten_band_when_calls_are_wide():
    sl = make_slice(5.0, half_spread=0.0002)
    n = len(sl.strikes)
    wide = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv,
                 tuple(b * 0.8 for b in sl.call_bid), tuple(a * 1.2 for a in sl.call_ask), (None,) * n, (None,) * n)
    both = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv, wide.call_bid, wide.call_ask, sl.put_bid, sl.put_ask)
    K = 78000.0
    lo_c, hi_c = arb_digital_band(wide, K)
    lo_b, hi_b = arb_digital_band(both, K)
    assert lo_b >= lo_c and hi_b <= hi_c and (hi_b - lo_b) < (hi_c - lo_c)


@given(mult=st.floats(1.01, 3.0), days=st.floats(1.0, 10.0))
@settings(max_examples=30, deadline=None)
def test_outside_strikes_band_is_one_sided_and_valid(mult, days):
    """Ausserhalb der Strikes: [0, Schranke] bzw. [Schranke, 1] ueber Monotonie, enthaelt die Wahrheit."""
    sl = make_slice(days, half_spread=0.0002, lo=0.8, hi=1.2)
    up, down = sl.strikes[-1] * mult, sl.strikes[0] / mult
    lo, hi = arb_digital_band(sl, up)
    assert lo == 0.0 and black76_digital(sl.F, up, sl.T, 0.5) <= hi + 1e-9
    lo, hi = arb_digital_band(sl, down)
    assert hi == 1.0 and black76_digital(sl.F, down, sl.T, 0.5) >= lo - 1e-9


def test_no_quotes_gives_trivial_band_and_no_model():
    sl = make_slice(2.0)
    n = len(sl.strikes)
    nob = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv, (None,) * n, (None,) * n)
    assert arb_digital_band(nob, 83000.0) == (0.0, 1.0)
    assert band_at_slice(nob, 83000.0) is None
    est = estimate_probability([nob], 83000.0, nob.expiry, NOW)
    assert est.p_low == 0.0 and est.p_high == 1.0 and est.m_low is None
    assert est.p_mid is not None  # Rueckfall: Mark-IV wie Phase 2


# ----------------------------------------------------------- Schaetzung gesamt

def test_estimate_exact_band_contains_truth_and_method():
    sl = make_slice(2.0, half_spread=0.0003)
    K = 83000.0
    est = estimate_probability([sl], K, sl.expiry + timedelta(minutes=3), NOW)
    truth = black76_digital(sl.F, K, sl.T, 0.5)
    assert est.method == "exakt"
    assert est.p_low <= truth <= est.p_high
    assert est.p_low <= est.p_mid <= est.p_high


def test_time_interpolation_reproduces_endpoints():
    s1 = make_slice(1.0, F=80000.0, sig=lambda K, F: 0.45)
    s2 = make_slice(3.0, F=81000.0, sig=lambda K, F: 0.55)
    K = 82000.0
    near1 = estimate_probability([s1, s2], K, s1.expiry + timedelta(minutes=4), NOW)
    assert near1.method == "exakt"
    e = estimate_probability([s1, s2], K, s1.expiry + timedelta(minutes=6), NOW)
    assert e.method == "interpoliert"
    assert abs(e.p_mid - point_at_slice(s1, K)) < 0.003
    e2 = estimate_probability([s1, s2], K, s2.expiry - timedelta(minutes=6), NOW)
    assert abs(e2.p_mid - point_at_slice(s2, K)) < 0.003


def test_interpolated_band_is_hull_of_endpoint_bands():
    s1 = make_slice(1.0, sig=lambda K, F: 0.45, half_spread=0.0002)
    s2 = make_slice(3.0, sig=lambda K, F: 0.55, half_spread=0.0002)
    K = 84000.0
    est = estimate_probability([s1, s2], K, NOW + timedelta(days=2), NOW, delta=0.0)
    b1, b2 = arb_digital_band(s1, K), arb_digital_band(s2, K)
    assert est.p_low <= min(b1[0], b2[0]) and est.p_high >= max(b1[1], b2[1])
    assert est.p_low <= est.p_mid <= est.p_high


def test_unbracketed_is_none_far_strike_is_one_sided():
    sl = make_slice(2.0)
    far = estimate_probability([sl], sl.strikes[-1] * 2, sl.expiry, NOW)
    assert far.p_mid is None and far.p_low == 0.0 and far.p_high < 0.01
    assert estimate_probability([sl], 80000.0, sl.expiry + timedelta(hours=5), NOW) is None
    assert estimate_probability([sl], 80000.0, sl.expiry - timedelta(hours=5), NOW) is None  # ohne Kassakurs


def test_reference_shift_widens_band():
    sl = make_slice(2.0, half_spread=0.0002)
    K = 83000.0
    narrow = estimate_probability([sl], K, sl.expiry, NOW, delta=0.0)
    wide = estimate_probability([sl], K, sl.expiry, NOW, delta=0.01)
    assert wide.p_low <= narrow.p_low and wide.p_high >= narrow.p_high


# --------------------------------------------------------------- Modellband

TRUE = Svi(0.0004, 0.004, -0.35, 0.005, 0.06)  # fuer etwa 3 Tage, BTC-artig, arbitragefrei


@given(m=st.floats(0.95, 1.06), rel=st.floats(0.0, 0.03))
@settings(max_examples=12, deadline=None)
def test_model_band_contains_true_svi_digital_exact(m, rel):
    sl = svi_slice(3.0, TRUE, rel_spread=rel)
    K = sl.F * m
    est = Surface([sl]).estimate(K, None, sl.expiry, NOW, delta=0.0)
    truth = svi_digital(TRUE, sl.F, K)
    assert est.m_low is not None, est.notes
    assert est.m_low - 2e-3 <= truth <= est.m_high + 2e-3
    assert est.p_low - 1e-9 <= est.p_mid <= est.p_high + 1e-9


def test_model_band_is_much_narrower_than_hull():
    sl = svi_slice(3.0, TRUE)
    est = Surface([sl]).estimate(sl.F * 1.01, None, sl.expiry, NOW, delta=0.001)
    assert est.m_high - est.m_low < 0.25 * (est.p_high - est.p_low)


@given(frac=st.floats(0.1, 0.9), m=st.floats(0.96, 1.04))
@settings(max_examples=10, deadline=None)
def test_model_band_contains_linear_variance_interpolation(frac, m):
    """Wahrheit = Gesamtvarianz linear in der Zeit (flache Uhr) zwischen zwei SVI-Verfaellen."""
    p1 = TRUE
    p2 = Svi(TRUE.a * 2.2, TRUE.b * 2.0, TRUE.rho, TRUE.m, TRUE.s * 1.3)
    s1, s2 = svi_slice(3.0, p1), svi_slice(7.0, p2)
    t = s1.expiry + (s2.expiry - s1.expiry) * frac
    K = s1.F * m
    k = math.log(K / s1.F)
    th = (t - s1.expiry) / (s2.expiry - s1.expiry)
    w = (1 - th) * p1.w(k) + th * p2.w(k)
    dw = (1 - th) * p1.dw(k) + th * p2.dw(k)
    truth = digital_w(k, w, dw)
    est = Surface([s1, s2]).estimate(K, None, t, NOW, delta=0.0)
    assert est.method == "interpoliert" and est.m_low is not None, est.notes
    assert est.m_low - 2e-3 <= truth <= est.m_high + 2e-3


def test_vorlauf_uses_spot_and_scales_variance():
    sl = svi_slice(3.0, TRUE)
    surf = Surface([sl], spot=sl.F)
    t = NOW + timedelta(days=1)
    K = sl.F * 1.01
    est = surf.estimate(K, None, t, NOW, delta=0.0)
    assert est.method == "vorlauf" and est.m_low is not None, est.notes
    k = math.log(K / sl.F)
    th = 1 / 3
    truth = digital_w(k, th * TRUE.w(k), th * TRUE.dw(k))
    assert est.m_low - 2e-3 <= truth <= est.m_high + 2e-3
    # Huelle enthaelt den Zustand "jetzt" (Kassakurs unter K -> 0) und das Band am Verfall
    assert est.p_low == 0.0
    assert Surface([sl]).estimate(K, None, t, NOW) is None  # ohne Kassakurs kein Vorlauf


def test_range_is_difference_of_digitals():
    sl = svi_slice(3.0, TRUE)
    surf = Surface([sl])
    a, b = sl.F * 0.99, sl.F * 1.02
    ra = surf.estimate(a, None, sl.expiry, NOW, delta=0.0)
    rb = surf.estimate(b, None, sl.expiry, NOW, delta=0.0)
    rr = surf.estimate(a, b, sl.expiry, NOW, delta=0.0)
    assert abs(rr.p_mid - (ra.p_mid - rb.p_mid)) < 1e-9
    truth = svi_digital(TRUE, sl.F, a) - svi_digital(TRUE, sl.F, b)
    assert rr.m_low - 3e-3 <= truth <= rr.m_high + 3e-3
    assert rr.p_low <= truth <= rr.p_high
    below = surf.estimate(None, b, sl.expiry, NOW, delta=0.0)
    assert abs(below.p_mid - (1 - rb.p_mid)) < 1e-9
    assert surf.estimate(b, a, sl.expiry, NOW) is None  # obere Grenze unter der unteren


def test_calendar_arbitrage_blocks_model_band():
    s1 = svi_slice(3.0, Svi(TRUE.a * 3, TRUE.b * 3, TRUE.rho, TRUE.m, TRUE.s))  # mehr Varianz frueh
    s2 = svi_slice(7.0, TRUE)
    est = Surface([s1, s2]).estimate(s1.F, None, s1.expiry + timedelta(days=2), NOW, delta=0.0)
    assert est.m_low is None and "kalender_arbitrage" in est.notes
    assert est.p_low is not None  # Huellband bleibt


def test_fit_contradicting_quotes_blocks_model_band(monkeypatch):
    """Liegt das Fit-Digital ausserhalb des modellfreien Bands, entfaellt das Modellband."""
    import scanner.prob as prob
    sl = svi_slice(3.0, TRUE)
    orig = prob.SliceModel.wd

    def skewed(self, label, k, pivot):
        r = orig(self, label, k, pivot)
        return r if r is None or label != "mid" else (r[0], r[1] - 0.05)  # kuenstlich steiler Skew

    monkeypatch.setattr(prob.SliceModel, "wd", skewed)
    est = Surface([sl]).estimate(sl.F * 1.01, None, sl.expiry, NOW, delta=0.0)
    assert est.m_low is None and "fit_ausserhalb_arbband" in est.notes
    assert est.p_low is not None and est.p_mid is not None  # Rueckfall bleibt


def test_model_uses_usable_neighbours_when_a_slice_has_too_few_quotes():
    thin = svi_slice(1.0, Svi(TRUE.a / 3, TRUE.b / 3, TRUE.rho, TRUE.m, TRUE.s), width=0.01)  # 1-2 Quotes
    good = svi_slice(3.0, TRUE)
    surf = Surface([thin, good], spot=good.F)
    t = NOW + timedelta(hours=12)  # vor dem duennen ersten Verfall
    est = surf.estimate(good.F * 1.005, None, t, NOW, delta=0.0)
    assert est.method == "vorlauf"  # Huelle wie gehabt mit dem ersten Verfall
    assert est.m_low is not None and "modell_ueber_nachbarn" in est.notes
    assert "zu_wenig_quotes" in est.notes
    k = math.log(1.005)
    th = 0.5 / 3
    truth = digital_w(k, th * TRUE.w(k), th * TRUE.dw(k))
    assert est.m_low - 3e-3 <= truth <= est.m_high + 3e-3
    exact = surf.estimate(good.F, None, thin.expiry, NOW, delta=0.0)
    assert exact.method == "exakt" and exact.m_low is None  # exakt ohne Smile: kein Modellband


# ---------------------------------------------------------- Review-Befunde (Phase 3)

MIX = ((0.6, 0.40), (0.3, 0.65), (0.1, 1.10))  # zeitkonsistente Volatilitaets-Mischung


def mix_price(F, K, T, call):
    return sum(wgt * black76_price(F, K, T, s, call) for wgt, s in MIX)


def mix_digital(F, K, T):
    return sum(wgt * black76_digital(F, K, T, s) for wgt, s in MIX)


def mix_slice(hours, F=100000.0, step=500.0, rel=0.03):
    T = hours * 3600 / YEAR_SECONDS
    ks, iv, cb, ca, pb, pa = [], [], [], [], [], []
    K = 90000.0
    while K <= 110000.0:
        c, p = mix_price(F, K, T, True), mix_price(F, K, T, False)
        from scanner.svi import implied_vol
        ks.append(K); iv.append(implied_vol(c, F, K, T, True) or 0.5)
        cb.append(c * (1 - rel)); ca.append(c * (1 + rel)); pb.append(p * (1 - rel)); pa.append(p * (1 + rel))
        K += step
    return Slice(NOW + timedelta(hours=hours), T, F, tuple(ks), tuple(iv), tuple(cb), tuple(ca), tuple(pb), tuple(pa))


@given(m=st.floats(0.95, 1.05))
@settings(max_examples=12, deadline=None)
def test_vorlauf_band_covers_fat_short_horizon_tails(m):
    """Befund B1: Vorlauf mit nur fester Moneyness unterschaetzte die Raender (Mischungs-Welt)."""
    sl = mix_slice(22.0)
    t = NOW + timedelta(hours=6)
    K = sl.F * m
    est = Surface([sl], spot=sl.F).estimate(K, None, t, NOW, delta=0.0)
    assert est.method == "vorlauf" and est.m_low is not None, est.notes
    truth = mix_digital(sl.F, K, 6 * 3600 / YEAR_SECONDS)
    assert est.m_low - 1e-3 <= truth <= est.m_high + 1e-3


def test_upper_bound_not_positive_is_none():
    sl = svi_slice(3.0, TRUE)
    assert Surface([sl]).estimate(None, 0.0, sl.expiry, NOW) is None
    assert Surface([sl]).estimate(None, -5.0, sl.expiry, NOW) is None


def test_inaccurate_fit_is_rejected():
    import random
    rnd = random.Random(3)
    sl = svi_slice(3.0, TRUE, rel_spread=0.0, abs_spread=0.3)
    fc = [1 + rnd.uniform(-0.15, 0.15) for _ in sl.strikes]  # Kurse springen, Spanne bleibt eng
    fp = [1 + rnd.uniform(-0.15, 0.15) for _ in sl.strikes]
    noisy = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv,
                  tuple(b * f for b, f in zip(sl.call_bid, fc)), tuple(a * f for a, f in zip(sl.call_ask, fc)),
                  tuple(b * f for b, f in zip(sl.put_bid, fp)), tuple(a * f for a, f in zip(sl.put_ask, fp)))
    surf = Surface([noisy])
    why = surf.model(noisy).problem()
    assert why is not None  # zu ungenau oder mit Arbitrage: kein Modellband
    est = surf.estimate(sl.F, None, sl.expiry, NOW)
    assert est.m_low is None


def test_neighbour_fallback_respects_gap_limit():
    thin = svi_slice(1.0, Svi(TRUE.a / 3, TRUE.b / 3, TRUE.rho, TRUE.m, TRUE.s), width=0.01)
    far = svi_slice(20.0, Svi(TRUE.a * 6, TRUE.b * 6, TRUE.rho, TRUE.m, TRUE.s))
    est = Surface([thin, far], spot=far.F).estimate(far.F, None, NOW + timedelta(hours=12), NOW)
    assert est.m_low is None and "nachbarn_zu_weit" in est.notes


def test_mark_smile_is_part_of_the_model_band(monkeypatch):
    """Befund B2: weicht Deribits Mark-Smile ab, wird das Band breiter (Modellvielfalt)."""
    import scanner.prob as prob
    sl = svi_slice(3.0, TRUE)
    base = Surface([sl]).estimate(sl.F * 1.01, None, sl.expiry, NOW, delta=0.0)
    orig = prob.SliceModel.mark_wd
    monkeypatch.setattr(prob.SliceModel, "mark_wd", lambda self, k: (lambda r: None if r is None else
                                                                    (r[0] * 1.3, r[1]))(orig(self, k)))
    wide = Surface([sl]).estimate(sl.F * 1.01, None, sl.expiry, NOW, delta=0.0)
    assert wide.m_high - wide.m_low > base.m_high - base.m_low
