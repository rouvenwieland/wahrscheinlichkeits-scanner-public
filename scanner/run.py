"""Kommandozeile: ``python -m scanner.run scan | daily | report``.

Nur lesende Abrufe oeffentlicher Daten und Papierhandel. Keine Konten, keine Schluessel.

``scan``: alle Basiswerte abrufen, zuordnen, rechnen, Papierhandel, Abrechnung.
``daily``: wie ``scan`` und danach ein Tagesbericht ``data/berichte/JJJJ-MM-TT.md`` plus
eine Zeile in ``data/berichte/verlauf.csv``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from . import deribit, kalshi, matching, polymarket
from .assets import ASSETS, DELTA_KALSHI, DELTA_LINEAR_EXTRA, DELTA_POLY_BASE, Asset
from .fees import fee_per_share_estimate
from .fund import FUNDS, MIN_EDGE, Fund, band_for, edges
from .hedge import market_hedge
from .http import FetchError
from .kalshi import Book, Level
from .prob import Surface
from .varclock import FLAT, VarianceClock, estimate_clock

log = logging.getLogger("scanner")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_BOOKS = 60  # Orderbuch-Abrufe je Lauf (Vorauswahl ueber die Kurse der Marktliste)
PRESCREEN_TOL = Decimal("0.01")  # Listenkurse koennen veraltet sein: Buch schon ab Huerde - TOL
STAT_BOOKS = 20  # zusaetzlich die besten Maerkte fuer die Statistik, auch ohne Aussicht auf Einstieg
MAX_DRIFT = 0.001  # bewegt sich der Kassakurs waehrend des Laufs staerker, kein Einstieg
CLOCK_DAYS = 60
SNAPSHOT_DAYS = 14  # Rohdaten je Lauf (gepackt, nur lokal) fuer spaetere Nachrechnung
OBS_FIELDS = ["timestamp", "asset", "platform", "id", "direction", "lower", "upper", "settle_time",
              "quality", "p_mid", "p_low", "p_high", "m_low", "m_high", "notes", "hedge_exec",
              "hedge_model", "hedge_n_min", "quelle", "yes_ask", "no_ask", "yes_depth", "no_depth",
              "edge_huelle_roh", "edge_huelle_hedge", "edge_modell_roh", "edge_modell_hedge",
              "edge_punkt_roh"] + \
             [f"decision_{n}" for n in FUNDS]


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_book_for(market: matching.PredMarket):
    if market.platform == "kalshi":
        return kalshi.fetch_orderbook(market.ticker)
    return polymarket.fetch_book(market.yes_token, market.no_token)


def result_for_position(pos: dict):
    """Offizielles Ergebnis einer Position oder None. Wirft FetchError bei Netzfehlern."""
    if pos["platform"] == "kalshi":
        return kalshi.fetch_result(pos["ticker"]) or None
    return polymarket.official_result(polymarket.fetch_market(pos["market_id"]))


def book_for_position(pos: dict):
    try:
        if pos["platform"] == "kalshi":
            return kalshi.fetch_orderbook(pos["ticker"])
        return polymarket.fetch_book(pos["yes_token"], pos["no_token"])
    except FetchError as exc:
        log.warning("Buch fuer Bewertung nicht lesbar: %s", exc)
        return None


# ------------------------------------------------------------- Optionsseite

def load_clock(asset: Asset, now: datetime, data_dir: Path) -> VarianceClock:
    """Varianz-Uhr aus 60 Tagen Stundenkerzen, einmal am Tag neu geschaetzt. Fehler -> flach."""
    path = data_dir / f"clock_{asset.code}.json"
    try:
        if path.exists():
            c = json.loads(path.read_text(encoding="utf-8"))
            if c.get("date") == now.strftime("%Y-%m-%d"):
                return VarianceClock(tuple(c["weights"]), c.get("source", "cache"))
        end = int(now.timestamp() * 1000)
        ticks, closes = deribit.fetch_candles(asset.perpetual, end - CLOCK_DAYS * 86_400_000, end)
        clock = estimate_clock(ticks, closes, source=f"{asset.perpetual} {CLOCK_DAYS} Tage")
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"date": now.strftime("%Y-%m-%d"), "source": clock.source,
                                    "weights": list(clock.weights)}), encoding="utf-8")
        return clock
    except (FetchError, ValueError, KeyError) as exc:
        log.warning("Varianz-Uhr %s nicht verfuegbar, flach: %s", asset.code, exc)
        return FLAT


@dataclass
class Underlying:
    asset: Asset
    surface: Surface
    usdt_basis: Optional[float]


def load_underlying(asset: Asset, now: datetime, data_dir: Path, cache: dict) -> Underlying:
    options = []
    for currency, prefix in asset.deribit:
        if currency not in cache:
            cache[currency] = deribit.get_json(f"{deribit.BASE}/public/get_book_summary_by_currency",
                                               {"currency": currency, "kind": "option"})
        options += deribit.parse_book_summary(cache[currency], prefix)
    mins = {}
    for currency, _ in asset.deribit:
        key = "min_" + currency
        if key not in cache:
            try:
                cache[key] = deribit.fetch_min_amounts(currency)
            except FetchError as exc:
                log.warning("Mindestgroessen %s nicht lesbar, Standardwerte: %s", currency, exc)
                cache[key] = {}
        mins.update(cache[key])
    if not options:
        raise FetchError(f"Deribit: keine Optionen fuer {asset.code}")
    slices = deribit.build_slices(options, now, min_amounts=mins)
    spot = deribit.fetch_index(asset.index_usd)
    try:
        basis = abs(deribit.fetch_index(asset.index_usdt) / spot - 1.0)
    except FetchError as exc:
        log.warning("USDT-Index %s fehlt: %s", asset.index_usdt, exc)
        basis = None
    clock = load_clock(asset, now, data_dir)
    return Underlying(asset, Surface(slices, spot, clock), basis)


def delta_for(mk: matching.PredMarket, und: Underlying) -> float:
    extra = DELTA_LINEAR_EXTRA if all(p.endswith("_USDC") for _, p in und.asset.deribit) else 0.0
    if mk.platform == "kalshi":
        return DELTA_KALSHI + extra
    basis = und.usdt_basis if und.usdt_basis is not None else 0.002  # ohne Messung vorsichtig
    return DELTA_POLY_BASE + basis + extra


# --------------------------------------------------------------- Zuordnung

@dataclass
class Match:
    market: matching.PredMarket
    pairing: matching.Pairing
    est: object
    hedge_exec: object  # fuer Arbitrage-/Huellband (Spannen schon im Band)
    hedge_model: object  # fuer das Modellband (halbe Spanne beim Eroeffnen dazu)


def collect_markets(rejections: Counter, raw_store: Optional[dict] = None) -> list:
    markets = []
    raw_store = raw_store if raw_store is not None else {}
    for asset in ASSETS.values():
        for series in asset.kalshi_series:
            page = kalshi.fetch_markets(series)
            raw_store.setdefault("kalshi", {})[series] = page
            for raw in page:
                mk, why = matching.parse_kalshi_market(raw, asset)
                (markets.append(mk) if mk else rejections.update([why]))
        events = polymarket.fetch_events(asset.pm_tag)
        raw_store.setdefault("polymarket", {})[asset.pm_tag] = events
        pm_markets, pm_rejects = matching.parse_polymarket_events(events, [asset])
        markets += pm_markets
        rejections.update(pm_rejects)
    return markets


def match_markets(markets: list, unds: dict, now: datetime, rejections: Counter) -> list:
    out = []
    for mk in markets:
        und = unds.get(mk.asset)
        if und is None or not und.surface.slices:
            rejections.update(["keine_optionen"])
            continue
        surf = und.surface
        pairing, why = matching.pair_with_deribit(mk, [s.expiry for s in surf.slices], now,
                                                  has_spot=surf.spot is not None)
        if pairing is None:
            rejections.update([why])
            continue
        lower, upper = mk.bounds
        try:
            est = surf.estimate(lower, upper, mk.settle_time, now, delta_for(mk, und))
            if est is None:
                rejections.update(["prob_nicht_berechenbar"])
                continue
            br, _ = surf.bracket(mk.settle_time, now)
            hsl = br.s1 if br.method == "exakt" else br.s2
            at_exp = br.method == "exakt"
            out.append(Match(mk, pairing, est, market_hedge(hsl, lower, upper, at_exp, False),
                             market_hedge(hsl, lower, upper, at_exp, True)))
        except (ValueError, ZeroDivisionError, OverflowError) as exc:  # ein Ausreisser stoppt nicht alles
            log.warning("Markt %s nicht berechenbar: %s", mk.id, exc)
            rejections.update(["prob_fehler"])
    return out


def _list_book(mk: matching.PredMarket) -> Book:
    ya = (Level(mk.list_yes_ask, Decimal(0)),) if mk.list_yes_ask is not None else ()
    na = (Level(mk.list_no_ask, Decimal(0)),) if mk.list_no_ask is not None else ()
    return Book(ya, na, (), ())


def _hc(h):
    return Decimal(str(round(h.per_contract, 6))) if h is not None else None


def edge_summary(m: Match, book: Book) -> dict:
    """Beste Kante (JA/NEIN) je Band, roh (nur Prognosemarkt-Gebuehr) und nach Hedge-Kosten.

    "nach Hedge-Kosten" ist ein Kostenfilter, kein abgesicherter Gewinn: Ausser bei exakter Zeit
    bleibt das Basisrisiko des Spreads zum Abrechnungszeitpunkt (Restwert, Rampe statt Stufe)."""
    out = {}
    for band, h, tags in (("huelle", m.hedge_exec, ("roh", "hedge")), ("modell", m.hedge_model, ("roh", "hedge")),
                          ("punkt", None, ("roh",))):
        lo, hi = band_for(band, m.est)
        for tag in tags:
            cost = Decimal(0) if tag == "roh" else _hc(h)
            if lo is None or cost is None:
                out[f"edge_{band}_{tag}"] = None
                continue
            e = edges(m.market.platform, lo, hi, book, cost)
            cand = [x for x in (e["yes_edge"], e["no_edge"]) if x is not None]
            out[f"edge_{band}_{tag}"] = max(cand) if cand else None
    return out


def fund_edge_key(name: str) -> str:
    pr = FUNDS[name]
    band = "huelle" if pr.band == "arb" else pr.band
    return f"edge_{band}_{'hedge' if pr.hedge else 'roh'}"


def prescreen_score(m: Match) -> Optional[Decimal]:
    """Beste Kante gegen die Listenkurse ueber alle Fonds, abzueglich deren Huerde (ohne Tiefe).
    >= 0 heisst: mindestens ein Fonds wuerde zu Listenkursen einsteigen."""
    e = edge_summary(m, _list_book(m.market))
    vals = []
    for n, pr in FUNDS.items():
        if m.pairing.quality not in pr.qualities:
            continue
        v = e.get(fund_edge_key(n))
        if v is not None:
            vals.append(v - pr.min_edge)
    return max(vals) if vals else None


def _f(x, nd=5):
    if x is None:
        return ""
    if isinstance(x, (float, Decimal)):
        return f"{x:.{nd}f}"
    return str(x)


# -------------------------------------------------------------------- Lauf

def observation_path(data_dir: Path, now: datetime) -> Path:
    return data_dir / "observations" / f"{now:%Y-%m}" / f"{now:%Y-%m-%d}.csv"


def keep_row(r: dict, full: bool) -> bool:
    """Welche Beobachtungen gespeichert werden (das Repo soll nicht um MB je Lauf wachsen).

    Erster Lauf des Tages (``full``): alle Maerkte ausser sicheren Faellen (p_mid ausserhalb
    [0,01; 0,99] ohne Kante). Spaetere Laeufe: nur Maerkte mit geholtem Orderbuch."""
    if r["quelle"] == "buch":
        return True
    if not full:
        return False
    if r["p_mid"] and 0.01 <= float(r["p_mid"]) <= 0.99:
        return True
    return any(r[k] and float(r[k]) >= -0.02 for k in ("edge_huelle_roh", "edge_modell_roh", "edge_punkt_roh"))


def scan(data_dir: Path, now: Optional[datetime] = None, snapshots: bool = True) -> dict:
    """Ein Lauf. Rueckgabe: Zusammenfassung (fuer ``daily``). Abruffehler -> FetchError.

    ``snapshots``: Rohdaten des Laufs gepackt ablegen (im oeffentlichen Repo aus: Deribit-Daten
    duerfen nicht weitergegeben werden)."""
    now = now or datetime.now(timezone.utc)
    rejections: Counter = Counter()
    cache: dict = {}
    unds = {}
    for code, asset in ASSETS.items():
        try:
            unds[code] = load_underlying(asset, now, data_dir, cache)
        except FetchError as exc:
            log.warning("Optionen %s nicht verfuegbar: %s", code, exc)
            rejections.update([f"optionen_fehlen_{code}"])
    if not unds:
        raise FetchError("keine Optionsdaten")
    raw_store: dict = {"deribit": {k: v for k, v in cache.items() if not k.startswith("min_")},
                       "index": {c: u.surface.spot for c, u in unds.items()}}
    markets = collect_markets(rejections, raw_store)
    matched = match_markets(markets, unds, now, rejections)

    funds = {n: Fund(n, data_dir) for n in FUNDS}
    scored = sorted(((prescreen_score(m), i) for i, m in enumerate(matched)),
                    key=lambda t: -(t[0] if t[0] is not None else Decimal(-9)))
    hopeful = [i for s, i in scored if s is not None and s >= -PRESCREEN_TOL]
    stats = [i for s, i in scored if s is not None][:STAT_BOOKS]
    want = list(dict.fromkeys(hopeful + stats))[:MAX_BOOKS]
    books: dict = {}
    for i in want:
        mk = matched[i].market
        try:
            books[(mk.platform, mk.id)] = fetch_book_for(mk)
        except FetchError as exc:
            log.warning("Buch fehlt fuer %s: %s", mk.id, exc)
            rejections.update(["buch_fehlt"])

    def lookup(pos):
        key = (pos["platform"], pos["market_id"])
        if key not in books:
            books[key] = book_for_position(pos)
        return books[key]

    drift = spot_drift(unds, {matched[i].market.asset for i in want})
    navs = {n: f.nav(lookup)[0] for n, f in funds.items()}
    rows, decisions_all = [], Counter()
    for m in matched:
        mk, est = m.market, m.est
        book = books.get((mk.platform, mk.id))
        source = "buch" if book is not None else "liste"
        use = book if book is not None else _list_book(mk)
        decisions = {}
        for n, f in funds.items():
            if book is None:
                decisions[n] = "SKIP:vorauswahl"
                continue
            if drift.get(mk.asset, 0.0) > MAX_DRIFT:
                decisions[n] = "SKIP:kurs_bewegt"
                continue
            hedge = m.hedge_model if f.band in ("modell", "punkt") else m.hedge_exec
            decisions[n] = f.consider(mk, m.pairing.quality, est, book, now, navs[n], hedge=hedge)
            if decisions[n].startswith("BUY"):
                navs[n] = f.nav(lookup)[0]
        decisions_all.update(f"{n}:{d}" for n, d in decisions.items())
        e0 = edges(mk.platform, None, None, use)
        es = edge_summary(m, use)
        lower, upper = mk.bounds
        rows.append({
            "timestamp": iso(now), "asset": mk.asset, "platform": mk.platform, "id": mk.id,
            "direction": mk.direction, "lower": _f(lower, 4), "upper": _f(upper, 4),
            "settle_time": iso(mk.settle_time), "quality": m.pairing.quality,
            "p_mid": _f(est.p_mid), "p_low": _f(est.p_low), "p_high": _f(est.p_high),
            "m_low": _f(est.m_low), "m_high": _f(est.m_high), "notes": "|".join(est.notes),
            "hedge_exec": _f(None if m.hedge_exec is None else m.hedge_exec.per_contract),
            "hedge_model": _f(None if m.hedge_model is None else m.hedge_model.per_contract),
            "hedge_n_min": "" if m.hedge_exec is None else m.hedge_exec.n_min, "quelle": source,
            "yes_ask": _f(e0["yes_ask"], 4), "no_ask": _f(e0["no_ask"], 4),
            "yes_depth": _f(e0["yes_depth"], 0) if book is not None else "",
            "no_depth": _f(e0["no_depth"], 0) if book is not None else "",
            **{k: _f(v, 4) for k, v in es.items()},
            **{f"decision_{n}": d for n, d in decisions.items()}})

    path = observation_path(data_dir, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OBS_FIELDS)
        if new:
            w.writeheader()
        w.writerows(r for r in rows if keep_row(r, full=new))

    settled = {n: f.settle(result_for_position, now) for n, f in funds.items()}
    for n, f in funds.items():
        assert f.invariant_ok(), f"Kassen-Invariante verletzt in {n}"
    if snapshots:
        _save_snapshot(data_dir, now, raw_store, books)
    summary = _summary(now, matched, rows, rejections, decisions_all, settled, funds, lookup)
    _save_summary(data_dir, summary)
    print(f"Scan {iso(now)}: {len(rows)} Beobachtungen {summary['matched_quality']}, "
          f"Buecher {len(books)}, Ablehnungen {sum(rejections.values())}, abgerechnet {settled}")
    return summary


def spot_drift(unds: dict, assets) -> dict:
    """Relative Bewegung des Kassakurses seit dem Optionsabruf je Basiswert (Abruffehler: unendlich)."""
    out = {}
    for code in assets:
        try:
            out[code] = abs(deribit.fetch_index(unds[code].asset.index_usd) / unds[code].surface.spot - 1.0)
        except FetchError as exc:
            log.warning("Kassakurs %s fuer Driftpruefung fehlt: %s", code, exc)
            out[code] = float("inf")
    return out


def _save_snapshot(data_dir: Path, now: datetime, raw_store: dict, books: dict) -> None:
    """Rohdaten des Laufs gepackt ablegen (nur lokal, Deribit-Daten nicht weitergeben)."""
    d = data_dir / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    raw_store = dict(raw_store, books={f"{p}:{i}": {
        "yes_asks": [[str(l.price), str(l.size)] for l in b.yes_asks],
        "no_asks": [[str(l.price), str(l.size)] for l in b.no_asks],
        "yes_bids": [[str(l.price), str(l.size)] for l in b.yes_bids],
        "no_bids": [[str(l.price), str(l.size)] for l in b.no_bids]} for (p, i), b in books.items()})
    with gzip.open(d / f"{now:%Y%m%dT%H%M%S}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(raw_store, fh)
    cutoff = now - timedelta(days=SNAPSHOT_DAYS)
    for f in d.glob("*.json.gz"):
        try:
            if datetime.strptime(f.name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) < cutoff:
                f.unlink()
        except ValueError:
            continue


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _summary(now, matched, rows, rejections, decisions, settled, funds, lookup) -> dict:
    def width(lo, hi):
        return None if lo is None or hi is None else hi - lo
    pos = Counter()
    for r in rows:
        for k in ("edge_huelle_roh", "edge_huelle_hedge", "edge_modell_roh", "edge_modell_hedge",
                  "edge_punkt_roh"):
            if r[k] != "" and float(r[k]) > 0:
                pos[f"{k}:{r['quelle']}"] += 1
    top = sorted((r for r in rows if r["edge_modell_hedge"] != ""),
                 key=lambda r: -float(r["edge_modell_hedge"]))[:10]
    titles = {(m.market.platform, m.market.id): m.market.title for m in matched}

    def best(key):
        vals = [float(r[key]) for r in rows if r[key] != "" and r["quelle"] == "buch"]
        return round(max(vals), 4) if vals else None
    navs = {}
    for n, f in funds.items():
        nav, flagged = f.nav(lookup)
        navs[n] = {"nav": str(nav.quantize(Decimal("0.01"))), "cash": str(f.cash.quantize(Decimal("0.01"))),
                   "offen": len(f.positions), "ohne_bid": flagged}
    return {
        "time": iso(now),
        "matched": len(matched),
        "matched_quality": dict(Counter(m.pairing.quality for m in matched)),
        "matched_asset_platform": dict(Counter(f"{m.market.asset}/{m.market.platform}" for m in matched)),
        "matched_direction": dict(Counter(m.market.direction for m in matched)),
        "with_model_band": sum(1 for m in matched if m.est.m_low is not None),
        "model_notes": dict(Counter(n.split(":")[0] for m in matched if m.est.m_low is None
                                    for n in m.est.notes)),
        "model_via_neighbors": sum(1 for m in matched if "modell_ueber_nachbarn" in m.est.notes
                                   and m.est.m_low is not None),
        "median_width_huelle": _median([width(m.est.p_low, m.est.p_high) for m in matched]),
        "median_width_modell": _median([width(m.est.m_low, m.est.m_high) for m in matched]),
        "median_hedge_exec": _median([m.hedge_exec.per_contract for m in matched if m.hedge_exec]),
        "median_hedge_model": _median([m.hedge_model.per_contract for m in matched if m.hedge_model]),
        "positive_edges": dict(pos),
        "books": sum(1 for r in rows if r["quelle"] == "buch"),
        "rejections": dict(rejections),
        "decisions": dict(decisions),
        "settled": settled,
        "funds": navs,
        "best_edges_book": {k: best(k) for k in ("edge_huelle_roh", "edge_huelle_hedge", "edge_modell_roh",
                                                 "edge_modell_hedge", "edge_punkt_roh")},
        "top_edges": [dict({k: r[k] for k in ("asset", "platform", "id", "direction", "lower", "upper",
                                              "settle_time", "quality", "p_mid", "p_low", "p_high", "m_low",
                                              "m_high", "yes_ask", "no_ask", "hedge_model", "edge_modell_roh",
                                              "edge_modell_hedge", "quelle")},
                           title=titles.get((r["platform"], r["id"]), "")) for r in top],
    }


def _save_summary(data_dir: Path, summary: dict) -> None:
    p = data_dir / "last_scan.json"
    cumulative = Counter()
    if p.exists():
        try:
            cumulative.update(json.loads(p.read_text(encoding="utf-8")).get("cumulative_rejections", {}))
        except ValueError:
            pass
    cumulative.update(summary["rejections"])
    p.write_text(json.dumps(dict(summary, cumulative_rejections=dict(cumulative)), indent=1),
                 encoding="utf-8")


# ------------------------------------------------------------- Tagesbericht

VERLAUF_FIELDS = ["time", "matched", "with_model_band", "median_width_huelle", "median_width_modell",
                  "median_hedge_model", "pos_modell_hedge_buch", "pos_modell_hedge_liste", "trades",
                  "settled", "best_huelle_hedge", "best_modell_roh", "best_modell_hedge"] + \
                 [f"nav_{n}" for n in FUNDS]


def write_daily(data_dir: Path, s: dict) -> Path:
    d = data_dir / "berichte"
    d.mkdir(parents=True, exist_ok=True)
    day = s["time"][:10]
    trades = sum(v for k, v in s["decisions"].items() if ":BUY" in k)
    lines = [f"## Lauf {s['time']}", "",
             f"- Zugeordnet: {s['matched']} ({s['matched_quality']}); je Basiswert/Plattform: "
             f"{s['matched_asset_platform']}; Arten: {s['matched_direction']}",
             f"- Modellband vorhanden: {s['with_model_band']} (davon ueber Nachbar-Verfaelle "
             f"{s['model_via_neighbors']}); Gruende ohne: {s['model_notes']}",
             f"- Median-Bandbreite Huelle {s['median_width_huelle']}, Modell {s['median_width_modell']}; "
             f"Median-Hedgekosten je Kontrakt {s['median_hedge_model']} USD",
             f"- Positive Kanten (Anzahl Maerkte; 'liste' = Kurse der Marktliste, nur Hinweis): "
             f"{s['positive_edges'] or 'keine'}",
             f"- Orderbuecher geprueft: {s['books']}; Papier-Kaeufe: {trades}; abgerechnet: {s['settled']}",
             "- Fonds: " + "; ".join(f"{n} NAV {v['nav']} (Kasse {v['cash']}, offen {v['offen']})"
                                     for n, v in s["funds"].items()),
             f"- Ablehnungen: {dict(sorted(s['rejections'].items(), key=lambda kv: -kv[1]))}", "",
             "| Basiswert | Plattform | Markt | Art | Qualitaet | p_mid | Modellband | JA-Ask | NEIN-Ask | "
             "Hedge | Kante roh | Kante n. Hedge | Quelle |", "|" + "---|" * 13]
    for r in s["top_edges"]:
        lines.append(f"| {r['asset']} | {r['platform']} | {r['id']} | {r['direction']} {r['lower']}-{r['upper']} | "
                     f"{r['quality']} | {r['p_mid']} | {r['m_low']}-{r['m_high']} | {r['yes_ask']} | {r['no_ask']} | "
                     f"{r['hedge_model']} | {r['edge_modell_roh']} | {r['edge_modell_hedge']} | {r['quelle']} |")
    path = d / f"{day}.md"
    head = "" if path.exists() else (f"# Tagesbericht {day}\n\nNur Papierhandel, keine Anlageberatung. "
                                     "Kanten in Wahrscheinlichkeitspunkten (0,01 = 1 Punkt).\n\n")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(head + "\n".join(lines) + "\n\n")
    vpath = d / "verlauf.csv"
    new = not vpath.exists()
    with vpath.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=VERLAUF_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": s["time"], "matched": s["matched"], "with_model_band": s["with_model_band"],
                    "median_width_huelle": s["median_width_huelle"],
                    "median_width_modell": s["median_width_modell"],
                    "median_hedge_model": s["median_hedge_model"],
                    "pos_modell_hedge_buch": s["positive_edges"].get("edge_modell_hedge:buch", 0),
                    "pos_modell_hedge_liste": s["positive_edges"].get("edge_modell_hedge:liste", 0),
                    "trades": trades, "settled": sum(s["settled"].values()),
                    "best_huelle_hedge": s["best_edges_book"]["edge_huelle_hedge"],
                    "best_modell_roh": s["best_edges_book"]["edge_modell_roh"],
                    "best_modell_hedge": s["best_edges_book"]["edge_modell_hedge"],
                    **{f"nav_{n}": v["nav"] for n, v in s["funds"].items()}})
    return path


def report(data_dir: Path) -> int:
    for name in FUNDS:
        f = Fund(name, data_dir)
        nav, flagged = f.nav(book_for_position)
        print(f"Fonds {name}: NAV {nav:.2f} USD (Kasse {f.cash:.2f}, Start {f.start}), "
              f"offene Positionen {len(f.positions)}, ohne Bid bewertet: {flagged}")
        for key, pos in f.positions.items():
            print(f"   {key} {pos['side'].upper()} {pos['shares']} Anteile, Kosten {Decimal(pos['cost']):.2f}, "
                  f"Schwelle {pos['threshold']}, Abrechnung {pos['settle_time']}")
    p = data_dir / "last_scan.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        print(f"Letzter Scan {s['time']}: zugeordnet {s.get('matched')} {s.get('matched_quality', '')}")
        print("Ablehnungen (letzter Scan):", dict(sorted(s["rejections"].items(), key=lambda kv: -kv[1])))
        print("Ablehnungen (gesamt):", dict(sorted(s["cumulative_rejections"].items(), key=lambda kv: -kv[1])))
    else:
        print("Noch kein Scan gelaufen.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scanner.run", description=__doc__)
    ap.add_argument("command", choices=["scan", "daily", "report"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--no-snapshots", action="store_true",
                    help="keine Rohdaten-Schnappschuesse (oeffentliches Repo)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command == "report":
        return report(args.data_dir)
    try:
        summary = scan(args.data_dir, snapshots=not args.no_snapshots)
    except FetchError as exc:
        print(f"FEHLER beim Abruf, Lauf abgebrochen (nichts geschrieben): {exc}", file=sys.stderr)
        return 1
    if args.command == "daily":
        print(f"Tagesbericht: {write_daily(args.data_dir, summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
