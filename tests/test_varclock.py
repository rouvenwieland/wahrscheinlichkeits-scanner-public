import math
import random
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, strategies as st

from scanner.varclock import FLAT, HOURS, VarianceClock, estimate_clock, hour_of_week, theta_range

UTC = timezone.utc
BASE = datetime(2026, 10, 1, tzinfo=UTC)
times = st.integers(0, 60 * 24 * 60).map(lambda m: BASE + timedelta(minutes=m))
weights = st.lists(st.floats(0.2, 5.0), min_size=HOURS, max_size=HOURS).map(tuple)


@given(a=times, b=times, c=times)
def test_flat_clock_is_linear_time(a, b, c):
    t0, t, t1 = sorted((a, b, c))
    if t1 == t0:
        return
    expect = (t - t0) / (t1 - t0)
    assert abs(FLAT.theta(t0, t, t1) - expect) < 1e-9


@given(w=weights, a=times, b=times, c=times)
@settings(max_examples=60)
def test_theta_monotone_bounded_and_additive(w, a, b, c):
    clock = VarianceClock(w)
    t0, t, t1 = sorted((a, b, c))
    if t1 == t0:
        return
    th = clock.theta(t0, t, t1)
    assert 0.0 <= th <= 1.0
    assert clock.theta(t0, t0, t1) == 0.0 and abs(clock.theta(t0, t1, t1) - 1.0) < 1e-12
    assert abs(clock.integral(t0, t) + clock.integral(t, t1) - clock.integral(t0, t1)) < 1e-6
    later = t + (t1 - t) / 2
    assert clock.theta(t0, later, t1) >= th - 1e-12


@given(w=weights)
def test_full_week_integral_is_weight_sum(w):
    clock = VarianceClock(w)
    t = BASE + timedelta(hours=5, minutes=17)
    assert abs(clock.integral(t, t + timedelta(days=7)) - sum(w)) < 1e-6


def _candles(days, peak_hour=14, peak=3.0, weekend=0.5, seed=1):
    rnd = random.Random(seed)
    start = datetime(2026, 8, 1, tzinfo=UTC)
    ticks, closes, px = [], [], 100.0
    for i in range(days * 24):
        t = start + timedelta(hours=i)
        var = (peak if t.hour == peak_hour else 1.0) * (weekend if t.weekday() >= 5 else 1.0)
        px *= math.exp(rnd.gauss(0, 0.004 * math.sqrt(var)))
        ticks.append(int(t.timestamp() * 1000))
        closes.append(px)
    return ticks, closes


def test_estimated_clock_finds_peak_and_weekend():
    clock = estimate_clock(*_candles(90), shrink=0.0)
    w = clock.weights
    assert abs(sum(w) / HOURS - 1.0) < 1e-9
    tuesday = [w[1 * 24 + h] for h in range(24)]
    assert max(range(24), key=lambda h: tuesday[h]) == 14
    assert sum(w[5 * 24:7 * 24]) / 48 < sum(w[:5 * 24]) / 120  # Wochenende ruhiger


def test_shrink_pulls_towards_flat_and_too_little_data_is_flat():
    raw = estimate_clock(*_candles(90), shrink=0.0)
    half = estimate_clock(*_candles(90), shrink=0.5)
    assert max(half.weights) < max(raw.weights) and min(half.weights) > min(raw.weights)
    assert estimate_clock(*_candles(10)) is FLAT
    ticks, closes = _candles(30)
    gappy = estimate_clock(ticks[::2], closes[::2])  # keine aufeinanderfolgenden Stunden
    assert gappy is FLAT


@given(w=weights, a=times, b=times, c=times, margin=st.floats(0.0, 0.2))
@settings(max_examples=60)
def test_theta_range_contains_flat_and_clock(w, a, b, c, margin):
    t0, t, t1 = sorted((a, b, c))
    if t1 == t0:
        return
    clock = VarianceClock(w)
    tf, tc, lo, hi = theta_range(clock, t0, t, t1, margin)
    assert 0.0 <= lo <= min(tf, tc) + 1e-12 and max(tf, tc) - 1e-12 <= hi <= 1.0


def test_hour_of_week_index():
    assert hour_of_week(datetime(2026, 10, 12, 0, 30, tzinfo=UTC)) == 0  # Montag
    assert hour_of_week(datetime(2026, 10, 18, 23, 0, tzinfo=UTC)) == HOURS - 1  # Sonntag
