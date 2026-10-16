import math
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, strategies as st

from scanner import hedge
from scanner.black import YEAR_SECONDS, black76_price
from scanner.prob import Slice

NOW = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)


def slice_at(expiry, F=80000.0, step=1000.0, sigma=0.5, half=10.0, min_amount=0.01, n=21):
    T = (expiry - NOW).total_seconds() / YEAR_SECONDS
    ks = [F + (i - n // 2) * step for i in range(n)]
    cb, ca, pb, pa = [], [], [], []
    for K in ks:
        c, p = black76_price(F, K, T, sigma, True), black76_price(F, K, T, sigma, False)
        cb.append(max(c - half, 1e-9)); ca.append(c + half)
        pb.append(max(p - half, 1e-9)); pa.append(p + half)
    return Slice(expiry, T, F, tuple(ks), (sigma,) * n, tuple(cb), tuple(ca), tuple(pb), tuple(pa), min_amount)


FRIDAY = datetime(2026, 10, 23, 8, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 10, 17, 8, tzinfo=timezone.utc)


def test_fee_is_capped_by_option_price():
    assert hedge._fee(80000, 1000.0) == pytest.approx(0.0003 * 80000)
    assert hedge._fee(80000, 10.0) == pytest.approx(0.125 * 10.0)


def test_adjacent_spread_cost_components():
    sl = slice_at(FRIDAY, half=0.0)  # ohne Spanne: Abweichung des Nachbar-Spreads ist 0
    h = hedge.digital_hedge(sl, 80500.0, at_expiry=True, include_open_spread=False, width=1)
    (k1, k2, kind), = h.legs
    assert (k1, k2) == (80000.0, 81000.0)
    # Kombi: nur das teurere Bein zahlt; Ende am Freitag: Ausuebung beider Beine (schlimmster Fall)
    expect = (0.0003 * 80000 + 2 * 0.00015 * 80000) / 1000
    assert h.per_contract == pytest.approx(expect, rel=1e-9)
    assert h.parts["abweichung"] == 0.0
    assert h.n_min == math.ceil(0.01 * 1000)


def test_daily_expiry_has_no_delivery_fee():
    fri = hedge.digital_hedge(slice_at(FRIDAY, half=0.0), 80500.0, True, False, width=1)
    sat = hedge.digital_hedge(slice_at(SATURDAY, half=0.0), 80500.0, True, False, width=1)
    assert sat.parts["ende"] == 0.0 and fri.parts["ende"] > 0.0


def test_closing_before_expiry_costs_fee_and_spread_again():
    sl = slice_at(FRIDAY, half=20.0)
    h = hedge.digital_hedge(sl, 80500.0, at_expiry=False, include_open_spread=True, width=1)
    assert h.parts["ende"] == pytest.approx(h.parts["eroeffnung"] + 2 * 20.0 / 1000)
    assert h.parts["spanne_eroeffnung"] == pytest.approx(2 * 20.0 / 1000)


def test_wider_spread_only_when_cheaper_including_bias():
    sl = slice_at(FRIDAY, half=40.0, step=500.0)
    narrow = hedge.digital_hedge(sl, 80250.0, False, True, width=1)
    best = hedge.digital_hedge(sl, 80250.0, False, True, width=4)
    assert best.per_contract <= narrow.per_contract + 1e-12
    (k1, k2, _), = best.legs
    assert k1 < 80250.0 <= k2


def test_n_min_scales_with_min_amount():
    a = hedge.digital_hedge(slice_at(FRIDAY, min_amount=0.01), 80500.0, False, False, width=1)
    b = hedge.digital_hedge(slice_at(FRIDAY, min_amount=0.1), 80500.0, False, False, width=1)
    assert a.per_contract == pytest.approx(b.per_contract) and b.n_min == 10 * a.n_min


def test_no_hedge_outside_strikes_or_without_quotes():
    sl = slice_at(FRIDAY)
    assert hedge.digital_hedge(sl, sl.strikes[0] - 1.0, False, False) is None
    assert hedge.digital_hedge(sl, sl.strikes[-1] + 1.0, False, False) is None
    n = len(sl.strikes)
    bare = Slice(sl.expiry, sl.T, sl.F, sl.strikes, sl.iv, (None,) * n, (None,) * n)
    assert hedge.digital_hedge(bare, 80500.0, False, False) is None


def test_range_hedge_sums_both_bounds():
    sl = slice_at(FRIDAY)
    a = hedge.digital_hedge(sl, 79500.0, False, True)
    b = hedge.digital_hedge(sl, 81500.0, False, True)
    r = hedge.market_hedge(sl, 79500.0, 81500.0, False, True)
    assert r.per_contract == pytest.approx(a.per_contract + b.per_contract)
    assert r.n_min == max(a.n_min, b.n_min) and len(r.legs) == 2
    assert hedge.market_hedge(sl, None, 81500.0, False, True).per_contract == pytest.approx(b.per_contract)


@given(K=st.floats(71000, 89000), half=st.floats(0.0, 80.0), at_exp=st.booleans(), incl=st.booleans())
@settings(max_examples=60, deadline=None)
def test_hedge_cost_nonnegative_and_spread_monotone(K, half, at_exp, incl):
    h = hedge.digital_hedge(slice_at(FRIDAY, half=half), K, at_exp, incl)
    h2 = hedge.digital_hedge(slice_at(FRIDAY, half=half + 20.0), K, at_exp, incl)
    assert h.per_contract >= 0 and h.n_min >= 1
    assert h2.per_contract >= h.per_contract - 1e-12  # breitere Spannen kosten nie weniger
