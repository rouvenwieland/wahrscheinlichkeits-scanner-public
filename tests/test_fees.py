from decimal import Decimal

from hypothesis import given, strategies as st

from scanner.fees import kalshi_fee, polymarket_taker_fee

prices = st.decimals(min_value=Decimal("0.001"), max_value=Decimal("0.999"), places=3)
shares = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000"), places=2)


def test_known_values():
    assert polymarket_taker_fee(100, Decimal("0.5")) == Decimal("1.75000")
    assert polymarket_taker_fee(100, Decimal("0.9")) == Decimal("0.63000")
    assert kalshi_fee(1, Decimal("0.5")) == Decimal("0.02")  # 1,75 ct aufgerundet
    assert kalshi_fee(100, Decimal("0.5")) == Decimal("1.75")
    assert kalshi_fee(100, Decimal("0.9")) == Decimal("0.63")


def test_polymarket_tiny_fee_is_zero_and_minimum():
    assert polymarket_taker_fee(Decimal("0.01"), Decimal("0.001")) == 0
    assert polymarket_taker_fee(Decimal("1"), Decimal("0.5")) == Decimal("0.01750")


def test_no_float_rounding_bug():
    # 0.07*C*p*(1-p) = 0.00875 exakt -> Halbwert rundet sauber auf
    assert polymarket_taker_fee(Decimal("0.5"), Decimal("0.5")) == Decimal("0.00875")
    assert kalshi_fee(1, Decimal("0.5")) == Decimal("0.02")
    assert kalshi_fee(4, Decimal("0.5")) == Decimal("0.07")  # exakt 0,07 -> nicht 0,08


@given(shares, prices)
def test_polymarket_properties(s, p):
    fee = polymarket_taker_fee(s, p)
    assert fee >= 0
    assert fee <= Decimal("0.0175") * s + Decimal("0.000005")
    assert fee == polymarket_taker_fee(s, 1 - p)
    assert fee == fee.quantize(Decimal("0.00001"))
    assert fee == 0 or fee >= Decimal("0.00001")


@given(st.integers(1, 100000), prices)
def test_kalshi_properties(c, p):
    fee = kalshi_fee(c, p)
    assert fee >= 0
    assert fee == fee.quantize(Decimal("0.01"))
    assert fee == kalshi_fee(c, 1 - p)
    raw = Decimal("0.07") * c * p * (1 - p)
    assert raw <= fee < raw + Decimal("0.01")
