import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from hypothesis import given, settings, strategies as st

from scanner.fees import fee_per_share_estimate
from scanner.fund import HEDGE_FREI, Fund, MIN_EDGE, plan_buy
from scanner.http import FetchError
from scanner.kalshi import Book, Level
from scanner.matching import PredMarket
from scanner.prob import ProbEstimate

NOW = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)
D = Decimal


def market(platform="kalshi", mid="M1", min_size=1):
    return PredMarket(platform, mid, 80000.0, "above", NOW, "BRTI", ticker=mid,
                      yes_token="y", no_token="n", min_size=D(min_size))


def lv(price, size):
    return Level(D(str(price)), D(str(size)))


def book(yes_asks=(), no_asks=(), yes_bids=(), no_bids=()):
    return Book(tuple(yes_asks), tuple(no_asks), tuple(yes_bids), tuple(no_bids))


def est(lo, hi):
    return ProbEstimate((lo + hi) / 2, lo, hi, "exakt")


def test_buys_yes_when_edge_big_enough(tmp_path):
    f = Fund("streng", tmp_path)
    b = book(yes_asks=[lv("0.50", 100)])
    d = f.consider(market(), "exakt", est(0.70, 0.80), b, NOW, f.cash, hedge=HEDGE_FREI)
    assert d == "BUY_YES"
    pos = f.positions["kalshi:M1"]
    assert D(pos["shares"]) * D("0.5") <= 5  # Deckel 5 % NAV
    assert f.invariant_ok() and f.cash < 100


def test_never_buys_without_edge(tmp_path):
    f = Fund("streng", tmp_path)
    # 0,50 + Gebuehr 0,0175 = 0,5175 ; p_low - 0,02 = 0,52 -> knapp ok; 0,53 nicht
    assert f.consider(market(), "exakt", est(0.54, 0.9), book(yes_asks=[lv("0.53", 50)]), NOW, f.cash, hedge=HEDGE_FREI) \
        == "SKIP:keine_kante"
    # NO-Seite: 1 - p_high - 0,02 muss > no_ask + Gebuehr sein
    assert f.consider(market(mid="M2"), "exakt", est(0.1, 0.5), book(no_asks=[lv("0.49", 50)]), NOW, f.cash, hedge=HEDGE_FREI) \
        == "SKIP:keine_kante"
    assert not f.positions and f.cash == 100


def test_buys_no(tmp_path):
    f = Fund("streng", tmp_path)
    d = f.consider(market(), "exakt", est(0.05, 0.20), book(no_asks=[lv("0.60", 100)]), NOW, f.cash, hedge=HEDGE_FREI)
    assert d == "BUY_NO" and f.positions["kalshi:M1"]["side"] == "no"


def test_strict_fund_ignores_interpolated(tmp_path):
    s, i = Fund("streng", tmp_path), Fund("interpoliert", tmp_path)
    b = book(yes_asks=[lv("0.30", 100)])
    assert s.consider(market(), "interpoliert", est(0.7, 0.8), b, NOW, s.cash, hedge=HEDGE_FREI) == "SKIP:qualitaet"
    assert i.consider(market(), "interpoliert", est(0.7, 0.8), b, NOW, i.cash, hedge=HEDGE_FREI) == "BUY_YES"


def test_one_position_per_market_and_no_band(tmp_path):
    f = Fund("interpoliert", tmp_path)
    b = book(yes_asks=[lv("0.30", 100)])
    assert f.consider(market(), "exakt", est(0.7, 0.8), b, NOW, f.cash, hedge=HEDGE_FREI) == "BUY_YES"
    assert f.consider(market(), "exakt", est(0.7, 0.8), b, NOW, f.cash, hedge=HEDGE_FREI) == "HELD"
    assert f.consider(market(mid="M9"), "exakt", ProbEstimate(0.7, None, None, "exakt"), b, NOW, f.cash, hedge=HEDGE_FREI) \
        == "SKIP:kein_band"


def test_polymarket_min_size_respected(tmp_path):
    f = Fund("streng", tmp_path)
    # nur 3 Anteile in der Tiefe -> unter orderMinSize 5 -> kein Trade
    d = f.consider(market("polymarket", min_size=5), "exakt", est(0.8, 0.9),
                   book(yes_asks=[lv("0.30", 3)]), NOW, f.cash, hedge=HEDGE_FREI)
    assert d == "SKIP:keine_kante" and not f.positions
    d = f.consider(market("polymarket", min_size=5), "exakt", est(0.8, 0.9),
                   book(yes_asks=[lv("0.30", 30)]), NOW, f.cash, hedge=HEDGE_FREI)
    assert d == "BUY_YES"


def test_settlement_only_with_official_result(tmp_path):
    f = Fund("streng", tmp_path)
    f.consider(market(), "exakt", est(0.8, 0.9), book(yes_asks=[lv("0.30", 100)]), NOW, f.cash, hedge=HEDGE_FREI)
    shares, cash_before = D(f.positions["kalshi:M1"]["shares"]), f.cash

    def boom(pos):
        raise FetchError("netz")

    assert f.settle(boom, NOW) == 0 and f.positions and f.cash == cash_before
    assert f.settle(lambda p: None, NOW) == 0 and f.positions
    assert f.settle(lambda p: "yes", NOW) == 1
    assert f.cash == cash_before + shares and not f.positions and f.invariant_ok()


def test_losing_settlement_pays_zero(tmp_path):
    f = Fund("streng", tmp_path)
    f.consider(market(), "exakt", est(0.8, 0.9), book(yes_asks=[lv("0.30", 100)]), NOW, f.cash, hedge=HEDGE_FREI)
    c = f.cash
    f.settle(lambda p: "no", NOW)
    assert f.cash == c and f.invariant_ok()


def test_mark_to_market_uses_bids_only(tmp_path):
    f = Fund("streng", tmp_path)
    f.consider(market(), "exakt", est(0.8, 0.9), book(yes_asks=[lv("0.30", 100)]), NOW, f.cash, hedge=HEDGE_FREI)
    shares = D(f.positions["kalshi:M1"]["shares"])
    nav, flagged = f.nav(lambda p: book(yes_bids=[lv("0.40", 1000)]))
    assert nav == f.cash + shares * D("0.40") and flagged == 0
    nav, flagged = f.nav(lambda p: book(yes_asks=[lv("0.99", 5)]))  # kein Bid -> Wert 0
    assert nav == f.cash and flagged == 1
    nav, flagged = f.nav(lambda p: None)
    assert nav == f.cash and flagged == 1


def test_state_persists(tmp_path):
    f = Fund("streng", tmp_path)
    f.consider(market(), "exakt", est(0.8, 0.9), book(yes_asks=[lv("0.30", 100)]), NOW, f.cash, hedge=HEDGE_FREI)
    g = Fund("streng", tmp_path)
    assert g.cash == f.cash and g.positions == f.positions and g.invariant_ok()
    assert (tmp_path / "trades_streng.csv").exists()


def test_plan_buy_respects_depth_and_budget():
    asks = [lv("0.40", 3), lv("0.45", 4), lv("0.90", 100)]
    fill = plan_buy("kalshi", asks, D("0.60"), D("100"), D(1))
    assert fill.shares == 7  # dritte Stufe ueber dem Limit
    fill = plan_buy("kalshi", asks, D("0.60"), D("2"), D(1))
    assert fill.total <= 2


level = st.tuples(st.integers(1, 98), st.integers(1, 300))


@given(st.lists(st.tuples(st.sampled_from(["kalshi", "polymarket"]), st.integers(0, 6),
                          st.floats(0.0, 1.0), st.floats(0.0, 1.0),
                          st.lists(level, max_size=5), st.lists(level, max_size=5),
                          st.sampled_from([None, "yes", "no"])), max_size=25))
@settings(max_examples=60, deadline=None)
def test_random_sequences_keep_invariants(steps):
    with tempfile.TemporaryDirectory() as tmp:
        f = Fund("interpoliert", Path(tmp))
        for platform, n, a, b, ya, na, result in steps:
            lo, hi = sorted((a, b))
            ya = sorted({p for p, _ in ya})
            yes_asks = [lv(p / 100, s) for p, (_, s) in zip(ya, level_pairs(ya))]
            no_asks = [lv(p / 100, 50) for p in sorted({p for p, _ in na})]
            bk = book(yes_asks=yes_asks, no_asks=no_asks)
            nav, _ = f.nav(lambda p: None)
            cash_before = f.cash
            was_held = f"{platform}:M{n}" in f.positions
            d = f.consider(market(platform, f"M{n}", 5 if platform == "polymarket" else 1), "exakt",
                           est(lo, hi), bk, NOW, nav, hedge=HEDGE_FREI)
            spent = cash_before - f.cash
            if d.startswith("BUY"):
                assert not was_held
                pos = f.positions[f"{platform}:M{n}"]
                shares = D(pos["shares"])
                assert spent <= min(nav * D("0.05"), D(10)) and spent > 0
                side_asks = yes_asks if d == "BUY_YES" else no_asks
                assert shares <= sum((l.size for l in side_asks), D(0))
                limit = (D(str(lo)) if d == "BUY_YES" else 1 - D(str(hi))) - MIN_EDGE
                assert spent / shares < limit  # nie ohne Kante
                if platform == "polymarket":
                    assert shares >= 5
            else:
                assert spent == 0
            if result:
                f.settle(lambda p, r=result: r, NOW)
            assert f.invariant_ok() and f.cash >= 0


def level_pairs(prices):
    return [(p, 10 + 7 * i) for i, p in enumerate(prices)]


def test_fee_estimate_is_symmetric():
    assert fee_per_share_estimate("kalshi", D("0.3")) == fee_per_share_estimate("kalshi", D("0.7"))


# ------------------------------------------------------------- Phase 3

from scanner.fund import FUNDS, band_for, depth_below, edges  # noqa: E402
from scanner.hedge import HedgeCost  # noqa: E402


def mest(lo, hi, mlo, mhi, method="interpoliert"):
    return ProbEstimate((mlo + mhi) / 2, lo, hi, method, (), mlo, mhi)


def test_hedge_cost_blocks_otherwise_good_trade(tmp_path):
    f = Fund("streng", tmp_path)
    b = book(yes_asks=[lv("0.50", 100)])
    # ohne Hedge: 0,70 - 0,02 = 0,68 > 0,5175 -> Kauf; mit 0,20 Hedge-Kosten: 0,48 < 0,5175
    assert f.consider(market(), "exakt", est(0.70, 0.80), b, NOW, f.cash,
                      hedge=HedgeCost(0.20, 1, ())) == "SKIP:keine_kante"
    assert f.consider(market(), "exakt", est(0.70, 0.80), b, NOW, f.cash,
                      hedge=HedgeCost(0.10, 1, ())) == "BUY_YES"


def test_hedge_min_size_needs_book_depth(tmp_path):
    f = Fund("streng", tmp_path)
    b = book(yes_asks=[lv("0.50", 20)])
    assert f.consider(market(), "exakt", est(0.70, 0.80), b, NOW, f.cash,
                      hedge=HedgeCost(0.01, 30, ())) == "SKIP:zu_duenn_fuer_hedge"
    assert f.consider(market(), "exakt", est(0.70, 0.80), b, NOW, f.cash,
                      hedge=HedgeCost(0.01, 20, ())) == "BUY_YES"


def test_no_hedge_means_no_trade(tmp_path):
    f = Fund("interpoliert", tmp_path)
    assert f.consider(market(), "interpoliert", est(0.7, 0.8), book(yes_asks=[lv("0.3", 100)]), NOW,
                      f.cash, hedge=None) == "SKIP:kein_hedge"


def test_model_fund_uses_model_band_and_vorlauf(tmp_path):
    mdl, hull, strict = Fund("modell", tmp_path), Fund("interpoliert", tmp_path), Fund("streng", tmp_path)
    e = mest(0.30, 0.90, 0.70, 0.74, "vorlauf")  # Huelle breit, Modellband eng
    b = book(yes_asks=[lv("0.55", 100)])
    assert strict.consider(market(), "vorlauf", e, b, NOW, strict.cash, hedge=HEDGE_FREI) == "SKIP:qualitaet"
    assert hull.consider(market(), "vorlauf", e, b, NOW, hull.cash, hedge=HEDGE_FREI) == "SKIP:keine_kante"
    assert mdl.consider(market(), "vorlauf", e, b, NOW, mdl.cash, hedge=HEDGE_FREI) == "BUY_YES"
    pos = mdl.positions["kalshi:M1"]
    assert pos["band"] == "modell" and pos["quality"] == "vorlauf"
    assert mdl.consider(market(mid="M2"), "exakt", ProbEstimate(0.5, 0.4, 0.6, "exakt"), b, NOW,
                        mdl.cash, hedge=HEDGE_FREI) == "SKIP:kein_band"


def test_edges_and_depth_include_hedge():
    b = book(yes_asks=[lv("0.40", 10), lv("0.45", 5)], no_asks=[lv("0.50", 7)])
    e0 = edges("kalshi", 0.5, 0.6, b)
    e1 = edges("kalshi", 0.5, 0.6, b, D("0.03"))
    assert e0["yes_edge"] - e1["yes_edge"] == D("0.03") and e0["no_edge"] - e1["no_edge"] == D("0.03")
    assert depth_below("kalshi", b.yes_asks, D("0.50")) == 15
    assert depth_below("kalshi", b.yes_asks, D("0.45")) == 10
    assert band_for("modell", mest(0.1, 0.9, 0.4, 0.5)) == (0.4, 0.5)
    assert set(FUNDS) == {"streng", "interpoliert", "modell", "modell_roh", "punkt"}


@given(edge=st.floats(-0.1, 0.2), hc=st.floats(0.0, 0.2), nmin=st.integers(0, 200),
       depth=st.integers(1, 300))
@settings(max_examples=80, deadline=None)
def test_never_buys_unless_edge_covers_fee_hedge_and_min_size(edge, hc, nmin, depth):
    with tempfile.TemporaryDirectory() as d:
        f = Fund("modell", Path(d))
        ask = D("0.40")
        p = float(ask + fee_per_share_estimate("kalshi", ask)) + edge
        p = min(max(p, 0.0), 1.0)
        dec = f.consider(market(), "interpoliert", mest(0.0, 1.0, p, p), book(yes_asks=[lv("0.40", depth)]),
                         NOW, f.cash, hedge=HedgeCost(hc, nmin, ()))
        if dec == "BUY_YES":
            assert p - float(MIN_EDGE) - hc > float(ask + fee_per_share_estimate("kalshi", ask))
            assert depth >= max(nmin, 1)
            assert f.invariant_ok()


def test_hedge_argument_is_mandatory(tmp_path):
    import pytest
    f = Fund("modell", tmp_path)
    with pytest.raises(TypeError):
        f.consider(market(), "exakt", est(0.7, 0.8), book(yes_asks=[lv("0.3", 100)]), NOW, f.cash)


def test_example_funds_without_hedge_and_naive_point(tmp_path):
    b = book(yes_asks=[lv("0.60", 100)])
    e = ProbEstimate(0.64, 0.30, 0.95, "interpoliert", (), 0.63, 0.66)
    big = HedgeCost(0.10, 10_000, ())
    mh = Fund("modell", tmp_path)
    mr = Fund("modell_roh", tmp_path)
    pk = Fund("punkt", tmp_path)
    # 0,60 + Gebuehr 0,0168 = 0,6168; modell_roh: 0,63 - 0,02 = 0,61 -> keine Kante
    assert mr.consider(market(), "interpoliert", e, b, NOW, mr.cash, hedge=big) == "SKIP:keine_kante"
    # punkt: 0,64 - 0,01 = 0,63 > 0,6168 -> Kauf, Hedge egal
    assert pk.consider(market(), "interpoliert", e, b, NOW, pk.cash, hedge=big) == "BUY_YES"
    e2 = ProbEstimate(0.70, 0.30, 0.95, "interpoliert", (), 0.66, 0.72)
    assert mr.consider(market(mid="M3"), "interpoliert", e2, b, NOW, mr.cash, hedge=big) == "BUY_YES"
    assert mh.consider(market(mid="M3"), "interpoliert", e2, b, NOW, mh.cash, hedge=big) == "SKIP:keine_kante"
    assert pk.consider(market(mid="M4"), "interpoliert", ProbEstimate(None, 0.1, 0.9, "interpoliert"), b, NOW,
                       pk.cash, hedge=None) == "SKIP:kein_band"
    assert pk.positions["kalshi:M1"]["band"] == "punkt"
