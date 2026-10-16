import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from scanner import matching as m

UTC = timezone.utc
KAL = {
    "ticker": "KXBTCD-26OCT1517-T92749.99", "strike_type": "greater", "floor_strike": 92749.99,
    "title": "Bitcoin price on Oct 15, 2026?", "subtitle": "$92,750 or above",
    "yes_sub_title": "$92,750 or above", "close_time": "2026-10-15T21:00:00Z",
    "rules_primary": "If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time "
                     "Index (BRTI) before 5 PM EDT is above 92749.99 at 5 PM EDT on Oct 15, 2026, "
                     "then the market resolves to Yes."}
DESC = ('This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT closing at 12:00 '
        'in the ET timezone (noon) on the date specified in the title has a final "Close" price higher '
        'than the price specified in the title. Otherwise, this market will resolve to "No".')
PM = {
    "id": "5432704", "question": "Will the price of Bitcoin be above $72,000 on October 17?",
    "groupItemTitle": "72,000", "endDate": "2026-10-17T16:00:00Z", "description": DESC,
    "outcomes": '["Yes", "No"]', "clobTokenIds": '["111", "222"]', "orderMinSize": 5,
    "closed": False, "active": True}


def kal(**kw):
    d = copy.deepcopy(KAL)
    d.update(kw)
    return d


def pm(**kw):
    d = copy.deepcopy(PM)
    d.update(kw)
    return d


def test_kalshi_valid():
    mk, why = m.parse_kalshi_market(KAL)
    assert why is None and mk.threshold == 92749.99 and mk.reference == "CF_RTI"
    assert mk.settle_time == datetime(2026, 10, 15, 21, 0, tzinfo=UTC)


@pytest.mark.parametrize("raw,reason", [
    (kal(strike_type="less"), "art_widerspruch"),  # Untertitel sagt "or above"
    (kal(strike_type="touch"), "kalshi_strike_type"),
    (kal(floor_strike=90000.0), "schwelle_widerspruch"),
    (kal(rules_primary=KAL["rules_primary"].replace(
        "CF Benchmarks' Bitcoin Real-Time Index (BRTI)", "Coinbase price")), "kalshi_referenz"),
    (kal(rules_primary=KAL["rules_primary"] + " Will it hit 100000?"), "verbotenes_wort"),
    (kal(title="Bitcoin between 90000 and 91000"), "art_widerspruch"),
    (kal(title="Bitcoin max price on Oct 15"), "verbotenes_wort"),
    (kal(close_time="2026-10-15T17:00:00Z"), "kalshi_zeit_widerspruch"),
    (kal(rules_primary=KAL["rules_primary"].replace("at 5 PM EDT on Oct 15, 2026", "later")),
     "kalshi_zeit_im_text_fehlt"),
])
def test_kalshi_rejects(raw, reason):
    mk, why = m.parse_kalshi_market(raw)
    assert mk is None and why == reason


def test_polymarket_valid_edt():
    mk, why = m.parse_polymarket_market(PM)
    assert why is None and mk.reference == "BINANCE_BTCUSDT"
    assert mk.settle_time == datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
    assert (mk.yes_token, mk.no_token) == ("111", "222")


def test_polymarket_dst_edt_vs_est():
    nov = PM["question"].replace("October 17", "November 2")
    ok, why = m.parse_polymarket_market(pm(question=nov, endDate="2026-11-02T17:00:00Z"))
    assert why is None and ok.settle_time == datetime(2026, 11, 2, 17, 0, tzinfo=UTC)
    # Nach der Zeitumstellung (1. Nov.) waeren 16:00Z falsch, vorher 17:00Z
    _, why = m.parse_polymarket_market(pm(question=nov))
    assert why == "pm_zeit_widerspruch"
    _, why = m.parse_polymarket_market(pm(endDate="2026-10-17T17:00:00Z"))
    assert why == "pm_zeit_widerspruch"


@pytest.mark.parametrize("raw,reason", [
    (pm(groupItemTitle="74,000"), "schwelle_widerspruch"),
    (pm(question="Will the price of Bitcoin be below $72,000 on October 17?"), "pm_frage_unbekannt"),
    (pm(question="Will Bitcoin hit $72,000 on October 17?"), "verbotenes_wort"),
    (pm(question="Will Bitcoin reach $72,000 on October 17?"), "verbotenes_wort"),
    (pm(question="Will Bitcoin dip to $72,000 on October 17?"), "verbotenes_wort"),
    (pm(question="Bitcoin Up or Down on October 17?"), "verbotenes_wort"),
    (pm(question="Will Bitcoin be between $72,000 and $74,000 on October 17?"), "pm_frage_unbekannt"),
    (pm(description=DESC.replace("12:00 in the ET timezone", "4:00 PM UTC")), "pm_zeitregel_fehlt"),
    (pm(description=DESC.replace("Binance", "Coinbase")), "pm_referenz"),
    (pm(description=DESC.replace("higher than", "lower than")), "pm_richtung"),
    (pm(outcomes='["No", "Yes"]'), "pm_outcomes"),
    (pm(question="Weird question 72,000"), "pm_frage_unbekannt"),
])
def test_polymarket_rejects(raw, reason):
    mk, why = m.parse_polymarket_market(raw)
    assert mk is None and why == reason


def test_pairing_exact_interpolated_and_discards():
    def e(d, h=8):
        return datetime(2026, 10, d, h, 0, tzinfo=UTC)

    exps = [e(16), e(17), e(23)]

    def mk(t, th=80000.0):
        return m.PredMarket("kalshi", "x", th, "above", t, "CF_RTI")

    p, _ = m.pair_with_deribit(mk(e(17, 8) + timedelta(minutes=4)), exps)
    assert p.quality == "exakt" and p.t1 == e(17)
    p, _ = m.pair_with_deribit(mk(e(16, 8) + timedelta(minutes=6)), exps)
    assert p.quality == "interpoliert" and (p.t1, p.t2) == (e(16), e(17))
    p, _ = m.pair_with_deribit(mk(e(20)), exps)  # 17. bis 23. = 6 Tage <= 8 Tage
    assert (p.quality, p.t1, p.t2) == ("interpoliert", e(17), e(23))
    assert m.pair_with_deribit(mk(e(20)), exps, max_gap=timedelta(days=3))[1] == "verfaelle_zu_weit"
    assert m.pair_with_deribit(mk(e(15)), exps)[1] == "kein_verfall_daneben"  # ohne "jetzt" kein Vorlauf
    assert m.pair_with_deribit(mk(e(25)), exps)[1] == "kein_verfall_daneben"
    # Schwellen ausserhalb der Strikes werden nicht mehr verworfen (einseitiges Band)
    assert m.pair_with_deribit(mk(e(16, 21), 1e9), exps)[0].quality == "interpoliert"


def test_pairing_vorlauf_and_min_lead():
    now = datetime(2026, 10, 15, 18, 0, tzinfo=UTC)
    exps = [datetime(2026, 10, 16, 8, 0, tzinfo=UTC)]
    mk = lambda t: m.PredMarket("kalshi", "x", 1.0, "above", t, "CF_RTI")
    p, why = m.pair_with_deribit(mk(now + timedelta(hours=3)), exps, now)
    assert why is None and p.quality == "vorlauf" and p.t1 is None and p.t2 == exps[0]
    assert m.pair_with_deribit(mk(now + timedelta(hours=3)), exps, now, has_spot=False)[1]         == "kein_verfall_daneben"
    assert m.pair_with_deribit(mk(now + timedelta(minutes=59)), exps, now)[1] == "zu_kurzfristig"


def test_pairing_long_dated_gap_scales_with_horizon():
    now = datetime(2026, 10, 15, 18, 0, tzinfo=UTC)
    dec, mar = datetime(2026, 12, 25, 8, tzinfo=UTC), datetime(2027, 3, 26, 8, tzinfo=UTC)
    jan1 = m.PredMarket("kalshi", "x", 1.0, "above", datetime(2027, 1, 1, 5, tzinfo=UTC), "CF_RTI")
    p, why = m.pair_with_deribit(jan1, [dec, mar], now)  # 91 Tage Abstand < 1,5 x 77 Tage
    assert why is None and p.quality == "interpoliert"
    near = m.PredMarket("kalshi", "x", 1.0, "above", now + timedelta(days=3), "CF_RTI")
    assert m.pair_with_deribit(near, [now + timedelta(days=1), now + timedelta(days=20)], now)[1]         == "verfaelle_zu_weit"


def test_real_fixture_markets(fixtures_dir):
    kal_raw = json.load(open(fixtures_dir / "kalshi_kxbtcd_markets.json"))["markets"]
    res = [m.parse_kalshi_market(r) for r in kal_raw]
    assert all(mk for mk, _ in res)  # alle echten KXBTCD-Maerkte sind gueltig
    events = json.load(open(fixtures_dir / "polymarket_events_bitcoin.json"))
    mks, rejects = m.parse_polymarket_events(events)
    assert len(mks) >= 50 and all(x.settle_time.hour == 16 for x in mks)
    assert not rejects or set(rejects) <= {"pm_nicht_offen"}


def test_phase3_kalshi_fixture_all_parse(fixtures_dir):
    raw = json.load(open(fixtures_dir / "kalshi_phase3_markets.json"))["markets"]
    res = [(r, *m.parse_kalshi_market(r)) for r in raw]
    bad = [(r["ticker"], why) for r, mk, why in res if mk is None]
    assert not bad
    kinds = {(mk.asset, mk.direction) for _, mk, _ in res}
    assert {("ETH", "above"), ("SOL", "above"), ("XRP", "above"), ("BTC", "range"), ("BTC", "below"),
            ("ETH", "range"), ("SOL", "range"), ("XRP", "range")} <= kinds
    for r, mk, _ in res:
        if r["strike_type"] == "between":
            assert mk.lower == r["floor_strike"] and mk.upper == r["cap_strike"] and mk.bounds == (mk.lower, mk.upper)
        if r["strike_type"] == "less":
            assert mk.lower is None and mk.upper == r["cap_strike"]
        if r["ticker"].startswith("KXBTCY"):  # 12 AM EST am 1.1. = 05:00 UTC
            assert mk.settle_time == datetime(2027, 1, 1, 5, 0, tzinfo=UTC)


@pytest.mark.parametrize("change,reason", [
    ({"cap_strike": 70000}, "schwelle_widerspruch"),
    ({"rules_primary_swap": ("between", "above")}, "art_widerspruch"),
    ({"rules_primary_swap": ("EDT", "EST")}, "kalshi_zeitzone_widerspruch"),
    ({"rules_primary_swap": ("CF Benchmarks' Bitcoin Real-Time Index (BRTI)", "BRTI")}, "kalshi_referenz"),
])
def test_kalshi_range_rejects(fixtures_dir, change, reason):
    raw = next(r for r in json.load(open(fixtures_dir / "kalshi_phase3_markets.json"))["markets"]
               if r["ticker"].startswith("KXBTC-") and r["strike_type"] == "between")
    raw = copy.deepcopy(raw)
    if "rules_primary_swap" in change:
        a, b = change.pop("rules_primary_swap")
        raw["rules_primary"] = raw["rules_primary"].replace(a, b)
    raw.update(change)
    mk, why = m.parse_kalshi_market(raw)
    assert mk is None and why == reason


def test_phase3_polymarket_fixture(fixtures_dir):
    events = json.load(open(fixtures_dir / "polymarket_events_phase3.json"))
    mks, rejects = m.parse_polymarket_events(events)
    assert not rejects
    by = {(x.asset, x.direction) for x in mks}
    assert {("ETH", "above"), ("XRP", "above"), ("BTC", "range"), ("BTC", "below"), ("BTC", "above"),
            ("SOL", "range")} <= by
    assert all(x.settle_time == datetime(2026, 10, 16, 16, 0, tzinfo=UTC) for x in mks)
    assert not any("Up or Down" in x.title for x in mks)
    btc = sorted((x for x in mks if x.asset == "BTC"), key=lambda x: x.threshold)
    low = next(x for x in btc if x.direction == "below")
    assert low.upper == 74000 and low.lower is None
    rng = next(x for x in btc if x.direction == "range" and x.lower == 74000)
    assert rng.upper == 76000 and rng.reference == "BINANCE_BTCUSDT"
    assert all(x.list_yes_ask is None or 0 < x.list_yes_ask < 1 for x in mks)


def test_polymarket_range_group_mismatch_rejected(fixtures_dir):
    ev = next(e for e in json.load(open(fixtures_dir / "polymarket_events_phase3.json"))
              if e["title"].startswith("Bitcoin price on"))
    raw = copy.deepcopy(next(x for x in ev["markets"] if "between" in x["question"]))
    raw["groupItemTitle"] = "1-2"
    assert m.parse_polymarket_market(raw, m.ASSETS["BTC"], "range") == (None, "schwelle_widerspruch")
    raw = copy.deepcopy(next(x for x in ev["markets"] if "between" in x["question"]))
    raw["description"] = raw["description"].replace("BTC/USDT", "ETH/USDT")
    assert m.parse_polymarket_market(raw, m.ASSETS["BTC"], "range") == (None, "pm_referenz")
