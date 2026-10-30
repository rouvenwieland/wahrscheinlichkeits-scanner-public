"""Regressionstests zur Reparatur vom 27.10.2026 (Fondslogik, Bewertung, Abrechnung, Zeitzonen).

Hintergrund: ``docs/REPARATUR-2026-10.md``.
"""
from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from scanner import matching
from scanner.fees import fee_per_share_estimate, kalshi_fee
from scanner.fund import (ASSET_CAP, CLUSTER_CAP, FUNDS, HEDGE_FREI, Fund, cluster_of, plan_buy)
from scanner.kalshi import Book, Level
from scanner.matching import PredMarket
from scanner.polymarket import official_result
from scanner.prob import ProbEstimate

UTC = timezone.utc
NOW = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)
DAY = datetime(2026, 10, 25, 21, 0, tzinfo=UTC)


def mk(mid, settle=DAY, asset="BTC", platform="kalshi", min_size=1):
    return PredMarket(platform, mid, 80000.0, "above", settle, "CF", ticker=mid, yes_token="y", no_token="n",
                      asset=asset, min_size=D(min_size))


def lv(p, s):
    return Level(D(str(p)), D(str(s)))


def no_book(price="0.50", size=10_000):
    return Book((), (lv(price, size),), (), ())


CHEAP_NO = ProbEstimate(0.05, 0.0, 0.1, "interpoliert", (), 0.04, 0.06)  # NEIN fuer 0,50 ist weit unter Wert


# ------------------------------------------------------------ Positionsgrenzen

def test_many_thresholds_of_one_expiry_are_one_bet():
    """Vorher: 12 Schwellen desselben BTC-Verfalls = 12 Kaeufe (Klumpen). Jetzt <= 10 % NAV je Tag."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Fund("punkt", d)
        decisions = [f.consider(mk(f"T{i}"), "interpoliert", CHEAP_NO, no_book(), NOW, D(100), hedge=None)
                     for i in range(12)]
        bought = [x for x in decisions if x.startswith("BUY")]
        spent = sum(D(p["cost"]) for p in f.positions.values())
        assert spent <= CLUSTER_CAP * 100 and 2 <= len(bought) <= 3
        assert decisions[-1] == "SKIP:limit_verfall"
        # Polymarket 12:00 ET am selben Tag gehoert zum selben Klumpen
        pm = mk("P1", settle=datetime(2026, 10, 25, 16, 0, tzinfo=UTC), platform="polymarket")
        assert f.consider(pm, "interpoliert", CHEAP_NO, no_book(), NOW, D(100), hedge=None) == "SKIP:limit_verfall"
        # anderer Tag: neuer Klumpen; anderer Basiswert: eigenes Limit
        assert f.consider(mk("N1", settle=datetime(2026, 10, 26, 21, 0, tzinfo=UTC)), "interpoliert", CHEAP_NO,
                          no_book(), NOW, D(100), hedge=None).startswith("BUY")
        assert f.consider(mk("E1", asset="ETH"), "interpoliert", CHEAP_NO, no_book(), NOW, D(100),
                          hedge=None).startswith("BUY")


def test_asset_cap_over_all_expiries(tmp_path):
    f = Fund("modell_roh", tmp_path)
    days = [datetime(2026, 10, d, 21, 0, tzinfo=UTC) for d in range(23, 31)]
    out = [f.consider(mk(f"D{i}", settle=t), "interpoliert", CHEAP_NO, no_book(), NOW, D(100), hedge=None)
           for i, t in enumerate(days) for _ in (0,)]
    spent = sum(D(p["cost"]) for p in f.positions.values())
    assert spent <= ASSET_CAP * 100 + D("0.01")
    assert "SKIP:limit_basiswert" in out


def test_cluster_key_uses_new_york_date():
    assert cluster_of("BTC", datetime(2026, 10, 25, 21, 0, tzinfo=UTC)) == "BTC:2026-10-25"
    assert cluster_of("BTC", "2026-10-25T16:00:00Z") == "BTC:2026-10-25"
    assert cluster_of("BTC", "2027-01-01T05:00:00Z") == "BTC:2027-01-01"  # 00:00 EST
    assert cluster_of("BTC", "2026-11-02T22:00:00Z") == "BTC:2026-11-02"  # 17:00 EST nach Zeitumstellung


# ------------------------------------------------------------ Kosten und Huerde

@pytest.mark.parametrize("platform", ["kalshi", "polymarket"])
def test_hurdle_is_net_of_fees(tmp_path, platform):
    """Die Huerde gilt nach Gebuehr: genau an der Grenze kein Kauf, knapp darueber Kauf."""
    ask = D("0.60")
    cost = ask + fee_per_share_estimate(platform, ask)
    f = Fund("punkt", tmp_path)
    at = float(cost + D("0.01"))  # p_mid - 0,01 == Preis + Gebuehr -> keine Kante
    e = ProbEstimate(at, 0, 1, "interpoliert")
    b = Book((lv(ask, 1000),), (), (), ())
    assert f.consider(mk("A", platform=platform, min_size=5), "interpoliert", e, b, NOW, D(100), hedge=None) \
        == "SKIP:keine_kante"
    e2 = ProbEstimate(at + 0.002, 0, 1, "interpoliert")
    assert f.consider(mk("B", platform=platform, min_size=5), "interpoliert", e2, b, NOW, D(100),
                      hedge=None) == "BUY_YES"
    pos = f.positions[f"{platform}:B"]
    assert D(pos["cost"]) / D(pos["shares"]) < D(str(at + 0.002)) - D("0.01")  # Durchschnitt inkl. Gebuehr


def test_kalshi_fee_rounding_counts_against_trade():
    """5 Kontrakte zu 0,98: Gebuehr 0,00686 -> 0,01 (aufgerundet); Limit knapp darueber -> weniger/kein Kauf."""
    assert kalshi_fee(5, D("0.98")) == D("0.01")
    fill = plan_buy("kalshi", (lv("0.98", 5),), D("0.9815"), D(10), D(1))
    assert fill is None or (fill.cost + fill.fee) / fill.shares < D("0.9815")


def test_plan_buy_skips_fractional_level():
    """Vorher brach der Durchlauf an einer Stufe mit < 1 Anteil ab und kaufte nichts."""
    fill = plan_buy("polymarket", (lv("0.06", "0.4"), lv("0.07", 500)), D("0.40"), D(5), D(5))
    assert fill is not None and fill.shares >= 5 and fill.cost / fill.shares == D("0.07")


def test_no_cash_is_not_reported_as_no_edge(tmp_path):
    """Vorher stand 'keine_kante', obwohl nur die Kasse fehlte (B99 am 17.10.)."""
    f = Fund("punkt", tmp_path)
    f.cash = D("0.03")
    f.start = D("0.03") + 0  # Invariante fuer diesen Test egal
    b = Book((lv("0.06", 1000),), (), (), ())
    assert f.consider(mk("X"), "interpoliert", ProbEstimate(0.45, 0, 1, "interpoliert"), b, NOW, D(100),
                      hedge=None) == "SKIP:kein_geld"


# ------------------------------------------------------------ Bewertung

def test_realized_and_unrealized_split(tmp_path):
    f = Fund("punkt", tmp_path)
    f.consider(mk("A"), "interpoliert", CHEAP_NO, no_book(), NOW, D(100), hedge=None)
    realized, open_cost = f.pnl_split()
    assert realized == 0 and open_cost == D(f.positions["kalshi:A"]["cost"])
    f.settle(lambda p: "no", NOW)
    realized, open_cost = f.pnl_split()
    assert open_cost == 0 and realized > 0 and f.invariant_ok()


# ------------------------------------------------------------ Abrechnung

@pytest.mark.parametrize("prices,outcomes,expected", [
    ('["1", "0"]', '["Yes", "No"]', "yes"),
    ('["0", "1"]', '["Yes", "No"]', "no"),
    ('["1.0", "0.0"]', '["Yes", "No"]', "yes"),
    (["0", "1"], ["Yes", "No"], "no"),
    ('["0.5", "0.5"]', '["Yes", "No"]', None),
    ('["0.9995", "0.0005"]', '["Yes", "No"]', None),
    ('["1", "0"]', '["Lamar", "Southeastern Louisiana"]', None),  # Sportmarkt: nie als JA/NEIN abrechnen
])
def test_polymarket_official_result(prices, outcomes, expected):
    m = {"closed": True, "umaResolutionStatus": "resolved", "outcomePrices": prices, "outcomes": outcomes}
    assert official_result(m) == expected


def test_polymarket_not_resolved_or_open():
    assert official_result({"closed": True, "umaResolutionStatus": "proposed", "outcomePrices": '["1","0"]'}) is None
    assert official_result({"closed": False, "outcomePrices": '["1","0"]'}) is None


def test_settle_pays_one_per_winning_share(tmp_path):
    f = Fund("modell_roh", tmp_path)
    f.consider(mk("A"), "interpoliert", CHEAP_NO, no_book(), NOW, D(100), hedge=None)
    shares = D(f.positions["kalshi:A"]["shares"])
    cash = f.cash
    assert f.settle(lambda p: None, NOW) == 0  # kein offizielles Ergebnis -> offen
    assert f.settle(lambda p: "no", NOW) == 1 and f.cash == cash + shares and not f.positions


# ------------------------------------------------------------ Zeitzonen

def test_kalshi_est_after_dst_change():
    raw = {"ticker": "KXBTCD-26NOV0217-T92749.99", "strike_type": "greater", "floor_strike": 92749.99,
           "title": "Bitcoin price on Nov 2, 2026?", "subtitle": "$92,750 or above",
           "yes_sub_title": "$92,750 or above", "close_time": "2026-11-02T22:00:00Z",
           "rules_primary": "If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time "
                            "Index (BRTI) before 5 PM EST is above 92749.99 at 5 PM EST on Nov 02, 2026, "
                            "then the market resolves to Yes."}
    m, why = matching.parse_kalshi_market(raw)
    assert why is None and m.settle_time == datetime(2026, 11, 2, 22, 0, tzinfo=UTC)
    bad = dict(raw, rules_primary=raw["rules_primary"].replace("EST", "EDT"))
    assert matching.parse_kalshi_market(bad)[1] == "kalshi_zeitzone_widerspruch"


# ------------------------------------------------------------ Gruppen

def test_fund_groups():
    assert {n for n, p in FUNDS.items() if p.gruppe == "kern"} == {"streng", "interpoliert", "modell"}
    assert {n for n, p in FUNDS.items() if p.gruppe == "vergleich"} == {"modell_roh", "punkt"}
    for n in ("modell_roh", "punkt"):
        assert FUNDS[n].desc.startswith("Vergleich: ungesicherte Wette") and not FUNDS[n].hedge


# ------------------------------------------------------------ Wahrscheinlichkeiten (Befunde der Pruefung)

def _sol_slice(window=True):
    import json
    import math
    from pathlib import Path
    from scanner import deribit, prob
    path = Path(__file__).parent / "fixtures" / "deribit_book_summary_sol_usdc.json"
    if not path.exists():  # oeffentliches Repo: keine echten Deribit-Daten
        pytest.skip("Deribit-Fixture fehlt")
    raw = json.loads(path.read_text())
    now = datetime(2026, 10, 15, 10, 46, tzinfo=UTC)
    s = deribit.build_slices(deribit.parse_book_summary(raw, "SOL_USDC"), now)[2]
    if window:  # einseitiges, duennes Buch wie an Wochenenden
        keep = [-0.08 <= math.log(K / s.F) <= 0.04 for K in s.strikes]
        f = lambda xs: tuple(x if k else None for x, k in zip(xs, keep))  # noqa: E731
        s = prob.Slice(s.expiry, s.T, s.F, s.strikes, s.iv, f(s.call_bid), f(s.call_ask), f(s.put_bid), f(s.put_ask))
    return s, now


def test_legacy_digital_is_continuous_at_strikes():
    """Vorher sprang das Rueckfall-Digital an Strikes um bis zu 0,09 (Abschnittssteigung)."""
    from scanner.prob import point_at_slice
    sl, _ = _sol_slice(window=False)
    jumps = [abs(point_at_slice(sl, K * (1 - 1e-7)) - point_at_slice(sl, K * (1 + 1e-7)))
             for K in sl.strikes[1:-1]
             if point_at_slice(sl, K * (1 - 1e-7)) is not None and point_at_slice(sl, K * (1 + 1e-7)) is not None]
    assert jumps and max(jumps) < 1e-4
    grid = [sl.strikes[0] + i * (sl.strikes[-1] - sl.strikes[0]) / 1000 for i in range(1001)]
    vals = [point_at_slice(sl, K) for K in grid]
    rises = [b - a for a, b in zip(vals, vals[1:]) if a is not None and b is not None]
    assert max(rises) < 0.005  # fast monoton fallend (vorher Spruenge und Anstiege bis 0,09)


def test_vorlauf_scaled_shape_outside_fit_is_dropped_and_members_valid():
    """KXSOLE-26OCT1717-B99: 1 USD Spanne bekam p_mid 0,456 (skaliert ausserhalb des Fits)."""
    from datetime import timedelta
    from scanner.prob import Surface
    sl, now = _sol_slice()
    surf = Surface([sl], spot=sl.F)
    t = now + timedelta(hours=10)
    e99 = surf.estimate(99, 99.9999, t, now, 0.002)
    assert "skaliert_ausserhalb" in e99.notes
    for lo, up in [(99, 99.9999), (101, 101.9999), (102, 102.9999), (103, 103.9999), (105, 105.9999)]:
        e = surf.estimate(lo, up, t, now, 0.002)
        above = surf.estimate(lo, None, t, now, 0.002)
        for v in (e.p_mid, e.m_low, e.m_high):
            assert v is None or 0.0 <= v <= 1.0
        assert e.p_mid < 0.3 and e.p_mid <= above.p_mid + 1e-9


def test_theta_range_lower_bound_is_relative():
    from datetime import timedelta
    from scanner.varclock import FLAT, theta_range
    t0 = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)
    tf, tc, lo, hi = theta_range(FLAT, t0, t0 + timedelta(hours=2), t0 + timedelta(hours=190))
    assert lo >= 0.7 * min(tf, tc) - 1e-12 and hi >= max(tf, tc)
