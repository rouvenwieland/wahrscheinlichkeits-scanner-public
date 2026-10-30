import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import requests

from scanner import deribit, kalshi, polymarket
from scanner.http import FetchError, get_json

NOW = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


def patch_get(monkeypatch, responses):
    calls = []

    def fake(self, url, **kw):
        calls.append(url)
        r = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(requests.Session, "get", fake)
    return calls


def test_http_error_raises_after_retries(monkeypatch):
    calls = patch_get(monkeypatch, [FakeResp(500)])
    with pytest.raises(FetchError):
        get_json("http://x")
    assert len(calls) == 3


def test_client_error_raises_immediately(monkeypatch):
    calls = patch_get(monkeypatch, [FakeResp(404)])
    with pytest.raises(FetchError):
        get_json("http://x")
    assert len(calls) == 1


def test_retry_then_success(monkeypatch):
    patch_get(monkeypatch, [requests.ConnectionError("x"), FakeResp(200, {"ok": 1})])
    assert get_json("http://x") == {"ok": 1}


@pytest.mark.parametrize("fn", [deribit.fetch_options, kalshi.fetch_markets, polymarket.fetch_events])
def test_fetchers_raise_on_http_error(monkeypatch, fn):
    patch_get(monkeypatch, [FakeResp(503)])
    with pytest.raises(FetchError):
        fn()


def test_kalshi_pagination_failure_on_second_page_raises(monkeypatch):
    first = FakeResp(200, {"markets": [{"ticker": "A"}], "cursor": "c1"})
    patch_get(monkeypatch, [first, FakeResp(500)])
    with pytest.raises(FetchError):
        kalshi.fetch_markets()


def test_kalshi_pagination_collects_pages(monkeypatch):
    patch_get(monkeypatch, [FakeResp(200, {"markets": [{"t": 1}], "cursor": "c"}),
                            FakeResp(200, {"markets": [{"t": 2}], "cursor": ""})])
    assert kalshi.fetch_markets() == [{"t": 1}, {"t": 2}]


def test_deribit_parse_fixture_and_slices(deribit_fixture):
    opts = deribit.parse_book_summary(deribit_fixture("deribit_book_summary_btc.json"))
    assert len(opts) > 900
    slices = deribit.build_slices(opts, NOW)
    exp = {s.expiry.strftime("%d%b%y").upper() for s in slices}
    assert "16OCT26" in exp and "25DEC26" in exp
    s = next(s for s in slices if s.expiry.day == 25 and s.expiry.month == 12)
    assert 80000 < s.F < 90000 and s.strikes == tuple(sorted(s.strikes))
    assert s.expiry.hour == 8


def test_deribit_bad_payload_raises():
    with pytest.raises(FetchError):
        deribit.parse_book_summary({"result": []})


def test_kalshi_orderbook_ask_from_opposite_bids():
    book = kalshi.parse_orderbook({"orderbook_fp": {
        "yes_dollars": [["0.40", "10"], ["0.35", "5"]],
        "no_dollars": [["0.55", "20"], ["0.50", "7"]]}})
    assert [(l.price, l.size) for l in book.yes_asks] == [(Decimal("0.45"), 20), (Decimal("0.50"), 7)]
    assert [(l.price, l.size) for l in book.no_asks] == [(Decimal("0.60"), 10), (Decimal("0.65"), 5)]
    assert book.yes_bids[0].price == Decimal("0.40") and book.no_bids[0].price == Decimal("0.55")


def test_kalshi_orderbook_fixture(fixtures_dir):
    raw = json.load(open(fixtures_dir / "kalshi_orderbook.json"))
    book = kalshi.parse_orderbook(raw)
    best_no = max(Decimal(p) for p, _ in raw["orderbook_fp"]["no_dollars"])
    assert book.yes_asks[0].price == 1 - best_no
    assert all(a.price <= b.price for a, b in zip(book.yes_asks, book.yes_asks[1:]))


def test_polymarket_book_sorted_ascending(fixtures_dir):
    raw = json.load(open(fixtures_dir / "polymarket_book.json"))
    asks, bids = polymarket.parse_book_side(raw)
    assert [l.price for l in asks] == sorted(l.price for l in asks)
    assert [l.price for l in bids] == sorted((l.price for l in bids), reverse=True)
    assert asks[0].price == min(Decimal(a["price"]) for a in raw["asks"])


def test_polymarket_official_result():
    ok = polymarket.official_result
    assert ok({"closed": True, "outcomePrices": '["1", "0"]', "umaResolutionStatus": "resolved"}) == "yes"
    assert ok({"closed": True, "outcomePrices": '["0", "1"]'}) == "no"
    assert ok({"closed": True, "outcomePrices": '["1", "0"]', "umaResolutionStatus": "proposed"}) is None
    assert ok({"closed": False, "outcomePrices": '["1", "0"]'}) is None
    assert ok({"closed": True, "outcomePrices": '["0.9995", "0.0005"]'}) is None
    assert ok({"closed": True, "outcomePrices": "kaputt"}) is None


# ------------------------------------------------------------- Phase 3

def test_parse_instrument_linear_and_decimal_strikes():
    e, k, t, pre = deribit.parse_instrument("XRP_USDC-23OCT26-1d24-P")
    assert (k, t, pre) == (1.24, "P", "XRP_USDC") and e == datetime(2026, 10, 23, 8, tzinfo=timezone.utc)
    assert deribit.parse_instrument("SOL_USDC-23OCT26-103-C")[1:] == (103.0, "C", "SOL_USDC")
    assert deribit.parse_instrument("BTC-25DEC26-104000-C")[3] == "BTC"
    assert deribit.parse_instrument("BTC-PERPETUAL") is None


def test_prices_in_usd_and_venues_merged():
    rows = [
        {"instrument_name": "BTC-23OCT26-80000-C", "bid_price": 0.05, "ask_price": 0.052,
         "mark_iv": 40.0, "underlying_price": 80000.0},
        {"instrument_name": "BTC_USDC-23OCT26-80000-C", "bid_price": 4050.0, "ask_price": 4200.0,
         "mark_iv": 41.0, "underlying_price": 80010.0},
        {"instrument_name": "BTC-23OCT26-80000-P", "bid_price": 0.04, "ask_price": None,
         "mark_iv": 42.0, "underlying_price": 80000.0}]
    inv = deribit.parse_book_summary({"result": rows}, "BTC")
    lin = deribit.parse_book_summary({"result": rows}, "BTC_USDC")
    assert inv[0].bid == pytest.approx(0.05 * 80000) and lin[0].bid == 4050.0
    assert deribit.parse_book_summary({"result": rows}, "ETH") == []
    sl = deribit.build_slices(inv + lin + [inv[0]] * 3, NOW, min_strikes=1)[0]
    assert sl.call_bid[0] == 4050.0  # bestes Geld ueber beide Plaetze (0,05*80000 = 4000)
    assert sl.call_ask[0] == pytest.approx(0.052 * 80000)  # bester Brief 4160 < 4200
    assert sl.put_bid[0] == pytest.approx(0.04 * 80000) and sl.put_ask[0] is None
    assert sl.min_amount == 0.1  # vorsichtig: groesste Mindestgroesse der beteiligten Plaetze (BTC invers)


def test_polymarket_events_paginate(monkeypatch):
    page = [{"title": f"e{i}"} for i in range(polymarket.PAGE)]
    calls = patch_get(monkeypatch, [FakeResp(200, page), FakeResp(200, [{"title": "last"}])])
    out = polymarket.fetch_events("bitcoin")
    assert len(out) == polymarket.PAGE + 1 and len(calls) == 2


def test_polymarket_pagination_error_on_second_page_raises(monkeypatch):
    page = [{"title": "x"}] * polymarket.PAGE
    patch_get(monkeypatch, [FakeResp(200, page), FakeResp(500)])
    with pytest.raises(FetchError):
        polymarket.fetch_events("ethereum")


def test_index_and_candles_parsing(monkeypatch):
    patch_get(monkeypatch, [FakeResp(200, {"result": {"index_price": 82000.5}})])
    assert deribit.fetch_index("btc_usd") == 82000.5
    patch_get(monkeypatch, [FakeResp(200, {"result": {"index_price": 0}})])
    with pytest.raises(FetchError):
        deribit.fetch_index("btc_usd")
    patch_get(monkeypatch, [FakeResp(200, {"result": {"status": "ok", "ticks": [1, 2], "close": [3, 4]}})])
    assert deribit.fetch_candles("BTC-PERPETUAL", 0, 1) == ([1, 2], [3.0, 4.0])
    patch_get(monkeypatch, [FakeResp(200, {"result": {"status": "no_data"}})])
    with pytest.raises(FetchError):
        deribit.fetch_candles("BTC-PERPETUAL", 0, 1)


def test_eth_and_sol_fixtures_build_slices(deribit_fixture):
    eth = deribit.parse_book_summary(deribit_fixture("deribit_book_summary_eth.json"), "ETH")
    sol = deribit.parse_book_summary(deribit_fixture("deribit_book_summary_sol_usdc.json"), "SOL_USDC")
    for opts, lo, hi in ((eth, 2000, 3000), (sol, 80, 140)):
        sls = deribit.build_slices(opts, NOW)
        assert len(sls) >= 5 and all(lo < s.F < hi for s in sls)
        s = sls[2]
        assert any(b is not None for b in s.put_bid) and any(b is not None for b in s.call_bid)


def test_get_json_reuses_one_session(monkeypatch):
    from scanner import http
    patch_get(monkeypatch, [FakeResp(200, {"ok": 1})])
    assert http._shared_session() is http._shared_session()
