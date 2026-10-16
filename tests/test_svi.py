import math

from hypothesis import assume, given, settings, strategies as st

from scanner.black import black76_price
from scanner.svi import Svi, arbitrage_problems, durrleman_g, fit_svi, implied_vol

T = 7 / 365


@given(F=st.floats(1.0, 200000.0), m=st.floats(0.6, 1.6), s=st.floats(0.1, 2.0),
       T=st.floats(0.001, 2.0), call=st.booleans())
@settings(max_examples=150)
def test_implied_vol_roundtrip(F, m, s, T, call):
    K = F * m
    price = black76_price(F, K, T, s, call)
    intrinsic = max(F - K, 0) if call else max(K - F, 0)
    assume(price - intrinsic > 1e-9 * F)  # sonst ist die Volatilitaet numerisch nicht bestimmt
    iv = implied_vol(price, F, K, T, call)
    assert iv is not None
    assert abs(black76_price(F, K, T, iv, call) - price) < 1e-6 * F


def test_implied_vol_rejects_impossible_prices():
    assert implied_vol(0.0, 100, 100, 0.1, True) is None
    assert implied_vol(5.0, 100, 90, 0.1, True) is None  # unter innerem Wert 10
    assert implied_vol(100.0, 100, 90, 0.1, True) is None  # Call nie >= F
    assert implied_vol(1.0, 100, 100, 0.0, True) is None


svi_params = st.builds(
    lambda a, b, rho, m, s: Svi(a, b, rho, m, s),
    a=st.floats(0.0002, 0.003), b=st.floats(0.002, 0.03), rho=st.floats(-0.7, 0.5),
    m=st.floats(-0.03, 0.03), s=st.floats(0.02, 0.15))


@given(p=svi_params)
@settings(max_examples=25, deadline=None)
def test_fit_recovers_known_smile(p):
    ks = [-0.15 + 0.0125 * i for i in range(25)]
    ivs = [math.sqrt(p.w(k) / T) for k in ks]
    r = fit_svi(ks, ivs, T)
    assert r is not None
    assert r.rmse_iv < 5e-4  # halber Volatilitaetspunkt / 10
    assert max(abs(r.params.w(k) - p.w(k)) for k in ks) < 2e-5


calm_params = st.builds(  # flache Smiles mit genug Mindestvarianz: sicher ohne Schmetterlings-Arbitrage
    lambda a, b, rho, m, s: Svi(a, b, rho, m, s),
    a=st.floats(0.0005, 0.003), b=st.floats(0.002, 0.015), rho=st.floats(-0.6, 0.4),
    m=st.floats(-0.03, 0.03), s=st.floats(0.05, 0.15))


@given(p=calm_params)
@settings(max_examples=60)
def test_reasonable_svi_is_arbitrage_free(p):
    assert arbitrage_problems(p, -0.2, 0.2) == []


def test_sharp_smile_with_tiny_variance_is_flagged():
    """Von Hypothesis gefunden (CI, 16.10.2026): spitzer Smile mit sehr kleiner Varianz hat
    Schmetterlings-Arbitrage (g < 0) und muss erkannt werden."""
    p = Svi(0.0002, 0.03, 0.5, 0.02734375, 0.021484375)
    assert "schmetterling" in arbitrage_problems(p, -0.2, 0.2)


def test_known_butterfly_arbitrage_is_detected():
    """Beispiel von Axel Vogt (Gatheral/Jacquier 2013): SVI mit Schmetterlings-Arbitrage."""
    bad = Svi(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    assert "schmetterling" in arbitrage_problems(bad, -1.5, 1.5)
    assert min(durrleman_g(bad, -1.5 + 0.01 * i) for i in range(301)) < 0


def test_lee_wing_bound_and_negative_variance_detected():
    assert "lee_fluegel" in arbitrage_problems(Svi(0.01, 1.5, 0.5, 0.0, 0.1), -0.2, 0.2)
    assert "varianz_negativ" in arbitrage_problems(Svi(-0.05, 0.01, 0.0, 0.0, 0.1), -0.2, 0.2)


def test_fit_needs_enough_points():
    assert fit_svi([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5] * 6, T) is None
    assert fit_svi([0.01 * i for i in range(7)], [0.5] * 7, T) is not None


@given(noise=st.lists(st.floats(-0.01, 0.01), min_size=21, max_size=21))
@settings(max_examples=15, deadline=None)
def test_fit_with_noise_stays_close(noise):
    p = Svi(0.0008, 0.01, -0.3, 0.0, 0.06)
    ks = [-0.1 + 0.01 * i for i in range(21)]
    ivs = [math.sqrt(p.w(k) / T) + e for k, e in zip(ks, noise)]
    r = fit_svi(ks, ivs, T)
    assert r is not None and r.rmse_iv < 0.012  # nicht schlechter als das Rauschen plus etwas


def test_warm_start_gives_same_quality():
    p = Svi(0.0008, 0.01, -0.3, 0.0, 0.06)
    ks = [-0.1 + 0.01 * i for i in range(21)]
    ivs = [math.sqrt(p.w(k) / T) for k in ks]
    cold = fit_svi(ks, ivs, T)
    warm = fit_svi(ks, [v + 0.002 for v in ivs], T, start=cold.params)
    assert warm.rmse_iv < 1e-3 and not warm.problems
