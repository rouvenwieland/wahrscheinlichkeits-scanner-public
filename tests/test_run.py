"""Gesamtlauf offline: alle Abrufe aus Fixtures (Stand 15.10.2026)."""
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from scanner import deribit, kalshi, polymarket, run
from scanner.http import FetchError

NOW = datetime(2026, 10, 15, 18, 50, tzinfo=timezone.utc)
REAL_BOOK_FOR_POSITION = run.book_for_position  # vor dem Ersetzen durch die Fixture merken
INDEX = {"btc_usd": 82300.0, "eth_usd": 2447.0, "sol_usd": 105.8, "xrp_usd": 1.341}


@pytest.fixture
def offline(monkeypatch, fixtures_dir, deribit_fixture):
    load = lambda name: json.load(open(fixtures_dir / name, encoding="utf-8"))
    summaries = {"BTC": deribit_fixture("deribit_book_summary_btc.json"),
                 "ETH": deribit_fixture("deribit_book_summary_eth.json"),
                 "USDC": deribit_fixture("deribit_book_summary_sol_usdc.json")}

    def fake_get_json(url, params=None, **kw):
        if url.endswith("get_book_summary_by_currency"):
            return summaries[params["currency"]]
        raise FetchError(f"offline: {url}")

    def fake_index(name):
        base = name.rsplit("_", 1)[0] + "_usd"
        if base not in INDEX:
            raise FetchError(name)
        return INDEX[base] * (1.0003 if name.endswith("usdt") else 1.0)

    kal = load("kalshi_kxbtcd_markets.json")["markets"] + load("kalshi_phase3_markets.json")["markets"]
    pm = load("polymarket_events_bitcoin.json") + load("polymarket_events_phase3.json")
    kbook = kalshi.parse_orderbook(load("kalshi_orderbook.json"))
    pa, pb = polymarket.parse_book_side(load("polymarket_book.json"))
    calls = {"books": 0}

    def fake_book(mk):
        calls["books"] += 1
        return kbook if mk.platform == "kalshi" else kalshi.Book(pa, pa, pb, pb)

    monkeypatch.setattr(deribit, "get_json", fake_get_json)
    monkeypatch.setattr(deribit, "fetch_index", fake_index)
    monkeypatch.setattr(kalshi, "fetch_markets", lambda series: [m for m in kal if m["ticker"].startswith(series + "-")])
    monkeypatch.setattr(polymarket, "fetch_events", lambda tag: pm)
    monkeypatch.setattr(run, "fetch_book_for", fake_book)
    monkeypatch.setattr(run, "book_for_position", lambda pos: kbook)
    monkeypatch.setattr(run, "result_for_position", lambda pos: None)
    return calls


def test_full_scan_and_daily_report(offline, tmp_path):
    s = run.scan(tmp_path, NOW)
    assert s["matched"] > 100, (s["rejections"], s["matched_asset_platform"])
    assert {"vorlauf", "interpoliert"} <= set(s["matched_quality"])
    assets = {k.split("/")[0] for k in s["matched_asset_platform"]}
    assert {"BTC", "ETH", "SOL"} <= assets and "XRP" not in assets  # keine XRP-Optionen im Fixture
    assert s["rejections"].get("keine_optionen", 0) > 0
    assert {"above", "range", "below"} <= set(s["matched_direction"])
    assert s["with_model_band"] > 0 and s["median_width_modell"] < s["median_width_huelle"]
    assert 0 < offline["books"] <= run.MAX_BOOKS
    obs = list(csv.DictReader(open(run.observation_path(tmp_path, NOW), encoding="utf-8")))
    assert 0 < len(obs) <= s["matched"] and set(obs[0]) == set(run.OBS_FIELDS)
    assert all(r["quelle"] == "buch" or (r["p_mid"] and 0.01 <= float(r["p_mid"]) <= 0.99)
               or any(r[k] and float(r[k]) >= -0.02 for k in ("edge_huelle_roh", "edge_modell_roh", "edge_punkt_roh"))
               for r in obs)
    assert all(r["decision_streng"] in ("SKIP:vorauswahl", "SKIP:qualitaet", "SKIP:kurs_bewegt")
               for r in obs)  # nie exakt
    for r in obs:  # Modellband liegt in [0, 1] und enthaelt die Punktschaetzung
        if r["m_low"]:
            assert 0 <= float(r["m_low"]) <= float(r["p_mid"]) + 1e-9 <= float(r["m_high"]) + 2e-9 <= 1 + 2e-9
    path = run.write_daily(tmp_path, s)
    text = path.read_text(encoding="utf-8")
    assert "Tagesbericht 2026-10-15" in text and "Zugeordnet:" in text
    run.write_daily(tmp_path, run.scan(tmp_path, NOW))  # zweiter Lauf: anhaengen, nicht ueberschreiben
    obs2 = list(csv.DictReader(open(run.observation_path(tmp_path, NOW), encoding="utf-8")))
    assert all(r["quelle"] == "buch" for r in obs2[len(obs):])  # spaetere Laeufe: nur Buch-Maerkte
    rows = list(csv.DictReader(open(tmp_path / "berichte" / "verlauf.csv", encoding="utf-8")))
    assert len(rows) == 2 and path.read_text(encoding="utf-8").count("## Lauf") == 2
    for n in run.FUNDS:
        assert run.Fund(n, tmp_path).invariant_ok()
    snaps = sorted((tmp_path / "snapshots").glob("*.json.gz"))
    assert snaps
    import gzip
    raw = json.load(gzip.open(snaps[-1], "rt", encoding="utf-8"))
    assert {"deribit", "kalshi", "polymarket", "books", "index"} <= set(raw)
    old = tmp_path / "snapshots" / "20260901T000000.json.gz"
    old.write_bytes(b"")
    run.scan(tmp_path, NOW)
    assert not old.exists()  # aelter als 14 Tage: entfernt


def test_fetch_error_aborts_without_writing(offline, monkeypatch, tmp_path):
    def broken(series):
        raise FetchError("Kalshi down")

    monkeypatch.setattr(kalshi, "fetch_markets", broken)
    monkeypatch.setattr(run, "DATA_DIR", tmp_path)
    assert run.main(["daily", "--data-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "observations").exists() and not (tmp_path / "berichte").exists()


def test_spot_drift_blocks_entries(offline, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "spot_drift", lambda unds, assets: {a: 0.01 for a in assets})
    s = run.scan(tmp_path, NOW)
    obs = list(csv.DictReader(open(run.observation_path(tmp_path, NOW), encoding="utf-8")))
    booked = [r for r in obs if r["quelle"] == "buch"]
    assert booked and all(r["decision_modell"] == "SKIP:kurs_bewegt" for r in booked)
    assert not any(":BUY" in k for k in s["decisions"])


def test_public_mode_writes_no_snapshots(offline, tmp_path):
    run.scan(tmp_path, NOW, snapshots=False)
    assert not (tmp_path / "snapshots").exists()
    assert run.observation_path(tmp_path, NOW).exists()


def _open_polymarket_position(tmp_path, market_id="9990001"):
    """Fonds 'punkt' mit einer offenen Polymarket-Position (Markt abgelaufen, noch nicht abgerechnet)."""
    pos = {"platform": "polymarket", "market_id": market_id, "ticker": None,
           "yes_token": "tok-yes", "no_token": "tok-no", "side": "yes", "shares": "10",
           "cost": "1.0", "threshold": 80000.0, "settle_time": "2026-10-15T16:00:00Z",
           "asset": "BTC", "direction": "above", "lower": 80000.0, "upper": None,
           "quality": "vorlauf", "band": "punkt", "hedge_cost": 0.0}
    state = {"name": "punkt", "start": "100", "cash": "100",
             "positions": {f"polymarket:{market_id}": pos}}
    (tmp_path / "fund_punkt.json").write_text(json.dumps(state), encoding="utf-8")
    return f"polymarket:{market_id}"


def test_snapshot_survives_closed_polymarket_book(offline, monkeypatch, tmp_path):
    """CI-Fund 17.10.2026: geschlossener Polymarket-Markt -> CLOB-Buch HTTP 404 -> Buch None.

    Der Rohdaten-Schnappschuss darf daran nicht scheitern (sonst bricht jeder Lauf ab und
    der Commit-Schritt, also auch die Abrechnung, wird nie erreicht)."""
    key = _open_polymarket_position(tmp_path)

    def closed(yes_token, no_token):
        raise FetchError("HTTP 404 bei https://clob.polymarket.com/book")

    monkeypatch.setattr(polymarket, "fetch_book", closed)
    monkeypatch.setattr(run, "book_for_position", REAL_BOOK_FOR_POSITION)
    s = run.scan(tmp_path, NOW, snapshots=True)
    assert s["matched"] > 0
    assert key in run.Fund("punkt", tmp_path).positions  # nicht abgerechnet, nur nicht bewertbar
    snaps = sorted((tmp_path / "snapshots").glob("*.json.gz"))
    assert snaps, "Schnappschuss fehlt"
    import gzip
    raw = json.load(gzip.open(snaps[-1], "rt", encoding="utf-8"))
    assert raw["books"] and key not in raw["books"]


def test_snapshot_failure_does_not_abort_scan(offline, monkeypatch, tmp_path, caplog):
    def broken(*a, **kw):
        raise OSError("Platte voll")

    monkeypatch.setattr(run, "_save_snapshot", broken)
    with caplog.at_level("WARNING", logger="scanner"):
        s = run.scan(tmp_path, NOW, snapshots=True)
    assert s["matched"] > 0
    assert run.observation_path(tmp_path, NOW).exists()
    assert "Schnappschuss" in caplog.text


def test_dashboard_renders_from_scan(offline, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(run.__file__).resolve().parent.parent / "dashboard"))
    import render_dashboard
    s = run.scan(tmp_path, NOW)
    run.write_daily(tmp_path, s)
    html = render_dashboard.render(tmp_path)
    assert "<title>Wahrscheinlichkeits-Monitor</title>" in html
    for pr in run.FUNDS.values():
        assert pr.label in html
    assert html.count('class="fund"') == len(run.FUNDS) and "Bandstreifen" in html
    assert "<script" not in html  # statische Seite
    assert render_dashboard.render(tmp_path / "leer").startswith("<title>")  # ohne Daten kein Absturz


def test_public_export_excludes_deribit_data(tmp_path):
    import importlib.util
    from pathlib import Path
    root = Path(run.__file__).resolve().parent.parent
    if not (root / "tools" / "export_public.py").exists():
        pytest.skip("nur im privaten Repo (das oeffentliche enthaelt das Export-Werkzeug nicht)")
    spec = importlib.util.spec_from_file_location("export_public", root / "tools" / "export_public.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    files = ["scanner/prob.py", "tests/fixtures/deribit_book_summary_btc.json", "tests/fixtures/kalshi_orderbook.json",
             "docs/MACHBARKEIT.md", "README.md", ".github/workflows/scan.yml", "deploy/public/scan.yml",
             "tools/export_public.py", "data/last_scan.json"]
    plan = ex.plan(files)
    assert "tests/fixtures/deribit_book_summary_btc.json" not in plan and "docs/MACHBARKEIT.md" not in plan
    assert plan[".github/workflows/scan.yml"] == "deploy/public/scan.yml"
    assert plan["README.md"] == "deploy/public/README.md"
    assert "tests/fixtures/kalshi_orderbook.json" in plan and "scanner/prob.py" in plan
    assert not any(k.startswith(("data/", "deploy/")) for k in plan)
    public_wf = (root / "deploy" / "public" / "scan.yml").read_text(encoding="utf-8")
    assert "--no-snapshots" in public_wf and "upload-artifact" not in public_wf


def test_size_tiers_in_last_scan_and_verlauf(offline, tmp_path):
    s = run.scan(tmp_path, NOW)
    run.write_daily(tmp_path, s)
    last = json.loads((tmp_path / "last_scan.json").read_text(encoding="utf-8"))
    assert list(last["funds"]) == list(run.FUND_KEYS)
    for key, (base, start) in run.FUND_KEYS.items():
        v = last["funds"][key]
        assert {"nav", "cash", "offen", "ohne_bid", "start"} <= set(v)
        assert isinstance(v["nav"], str) and isinstance(v["cash"], str) and Decimal(v["start"]) == start
        assert run.Fund(key, tmp_path).invariant_ok()
    buys = {k for k in s["decisions"] if ":BUY" in k}
    if any(k.startswith("punkt:") for k in buys):  # gleiche Regeln: die grossen Stufen kaufen mit
        assert any(k.startswith("punkt_10k:") for k in buys)
        assert run.Fund("punkt_10k", tmp_path).cash < 10000
    head = next(csv.reader(open(tmp_path / "berichte" / "verlauf.csv", encoding="utf-8")))
    assert head == run.VERLAUF_FIELDS and {"nav_punkt_1k", "nav_streng_10k"} <= set(head)
    obs = list(csv.DictReader(open(run.observation_path(tmp_path, NOW), encoding="utf-8")))
    assert "decision_punkt_1k" not in obs[0]  # Beobachtungen nur fuer die 100-USD-Fonds


def test_verlauf_gets_new_columns_without_losing_old_rows(tmp_path):
    """Bestehende verlauf.csv (vor den Groessenstufen): Kopfzeile wird erweitert, alte Zeilen bleiben."""
    path = tmp_path / "verlauf.csv"
    old_head = ["time", "trades", "nav_punkt"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([old_head, ["2026-10-16T15:23:17Z", "29", "96.16"]])
    new_fields = old_head + ["nav_punkt_1k", "nav_punkt_10k"]
    run.append_csv_row(path, new_fields, {"time": "2026-10-21T10:00:00Z", "trades": "1", "nav_punkt": "101.00",
                                          "nav_punkt_1k": "1010.00", "nav_punkt_10k": "10100.00"})
    run.append_csv_row(path, new_fields, {"time": "2026-10-21T12:00:00Z", "trades": "0", "nav_punkt": "102.00",
                                          "nav_punkt_1k": "1020.00", "nav_punkt_10k": "10200.00"})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert next(csv.reader(open(path, encoding="utf-8"))) == new_fields
    assert rows[0] == {"time": "2026-10-16T15:23:17Z", "trades": "29", "nav_punkt": "96.16",
                       "nav_punkt_1k": "", "nav_punkt_10k": ""}
    assert [r["nav_punkt_10k"] for r in rows] == ["", "10100.00", "10200.00"]
    assert not list(tmp_path.glob("*.tmp"))


def test_dashboard_german_number_format_and_cadence():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(run.__file__).resolve().parent.parent / "dashboard"))
    import render_dashboard as rd
    assert rd.num(10000) == "10.000,00" and rd.num(1234.5, 1, True) == "+1.234,5"
    assert rd.num(-0.3, 1) == "−0,3" and rd.num(None) == "–"
    hist = [{"time": f"2026-10-30T{h:02d}:00:00Z"} for h in (0, 6, 12, 18)]
    assert "alle 6 Stunden" in rd.cadence_text(hist)
    assert "stündlich" in rd.cadence_text([{"time": f"2026-10-30T0{h}:00:00Z"} for h in range(4)])
    assert rd.cadence_text([]) == "Läufe automatisch auf GitHub"
