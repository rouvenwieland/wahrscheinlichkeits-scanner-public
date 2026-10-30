"""Historische Neuberechnung aller Fonds mit der korrigierten Logik (Reihe "korrigiert").

    python tools/nachrechnen.py [--data-dir data] [--offline] [--ohne-filter]

Grundlage sind nur Daten im Repo: ``data/observations/**.csv`` (je Lauf Wahrscheinlichkeiten,
bester Brief JA/NEIN, Buchtiefe, Hedge-Kosten). Dazu die offiziellen Ergebnisse von Kalshi und
Polymarket, einmal abgerufen und in ``data/korrigiert/ergebnisse.json`` abgelegt; mit
``--offline`` wird nur diese Datei benutzt (vollstaendig reproduzierbar).

Gerechnet wird mit dem echten Fonds-Code (``scanner.fund.Fund``) gegen ein aus der Beobachtung
gebautes Buch:

- Kauf: bester Brief bis 100 USD Kurswert, darueber ein Aufschlag (Kalshi 1 Cent, Polymarket
  min(1 Cent, 4 % des Preises)), hoechstens 25 % der beobachteten Tiefe. Gemessen an den echten
  1k/10k-Kaeufen (Aufschlag 0 bis 1,06 Cent) ist das eher zu teuer als zu billig.
- Gebuehren, Huerden, Hedge-Kosten und -Mindestgroesse, Positionsgrenzen: wie live.
- Kein Einstieg in Laeufen, in denen der Kassakurs sich bewegt hatte (``SKIP:kurs_bewegt``).
- Bewertung: Geldkurs = 1 - Brief der Gegenseite aus der Beobachtung desselben Laufs, sonst der
  letzte gesehene Geldkurs. Abrechnung im ersten Lauf nach der offiziellen Abrechnung, zum
  offiziellen Ergebnis; am Ende ein Schlussschritt "jetzt", der nur noch abrechnet.
- Filter (Standard an, ``--ohne-filter`` aus): Die Wahrscheinlichkeiten in den Beobachtungen
  stammen vom alten Code. Zwei inzwischen behobene Fehler (docs/REPARATUR-2026-10.md, Befund
  F1/F2) verfaelschten vor allem die Punktschaetzung: Rueckfall-Schaetzungen ohne Modell
  (``m_low`` leer) und Vorlauf-Maerkte. Diese Zeilen werden fuer den Fonds ``punkt`` nicht
  gehandelt, weil der korrigierte Code dort andere Werte liefern wuerde. Die Bandfonds sind davon
  nicht betroffen (breite Baender verhindern Kaeufe nur).

Ausgabe in ``data/korrigiert/``: ``verlauf.csv``, ``fonds/`` (Zustand und Handelsliste je Fonds),
``zusammenfassung.json`` (Kennzahlen, Chancen, Kalibrierung).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scanner.fund import FUND_KEYS, FUNDS, Fund  # noqa: E402
from scanner.hedge import HedgeCost  # noqa: E402
from scanner.http import FetchError, get_json  # noqa: E402
from scanner.kalshi import BASE as KALSHI, Book, Level  # noqa: E402
from scanner.matching import PredMarket  # noqa: E402
from scanner.polymarket import GAMMA, official_result  # noqa: E402
from scanner.prob import ProbEstimate  # noqa: E402

UTC = timezone.utc
D = Decimal
TOP_USD = D(100)  # so viel Kurswert gilt als zum besten Brief verfuegbar
DEPTH_SHARE = D("0.25")  # hoechstens dieser Anteil der beobachteten Tiefe
SETTLE_FALLBACK = timedelta(hours=2)
BATCH = 50


def ts(s: str) -> datetime:
    s = s.strip().replace(" ", "T").replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    return datetime.fromisoformat(s).astimezone(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fnum(x):
    return None if x in (None, "") else float(x)


# ------------------------------------------------------------------ Daten

def load_observations(data_dir: Path) -> list:
    rows = []
    for f in sorted((data_dir / "observations").glob("*/*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    rows.sort(key=lambda r: r["timestamp"])  # stabil: Reihenfolge im Lauf bleibt
    return rows


def fetch_results(keys: list, cache: dict, now: datetime) -> None:
    """Offizielle Ergebnisse (und Polymarket-Token) in ``cache`` nachtragen. Nur lesend, oeffentlich."""
    todo_k = [i for p, i in keys if p == "kalshi" and not (cache.get(f"kalshi:{i}") or {}).get("result")]
    todo_p = [i for p, i in keys if p == "polymarket" and not (cache.get(f"polymarket:{i}") or {}).get("result")]
    for n in range(0, len(todo_k), BATCH):
        data = get_json(f"{KALSHI}/markets", {"tickers": ",".join(todo_k[n:n + BATCH]), "limit": 1000})
        for m in data.get("markets", []):
            res = (m.get("result") or "").lower()
            final = m.get("status") in ("finalized", "settled") and res in ("yes", "no")
            cache[f"kalshi:{m['ticker']}"] = {
                "result": res if final else None, "status": m.get("status"),
                "settled_at": m.get("settlement_ts") if final else None,
                "wert": m.get("expiration_value"), "abgerufen": iso(now)}
    for n in range(0, len(todo_p), BATCH):
        ids = todo_p[n:n + BATCH]
        for closed in ("true", "false"):
            data = get_json(f"{GAMMA}/markets", [("id", i) for i in ids] + [("closed", closed), ("limit", 500)])
            for m in data if isinstance(data, list) else []:
                res = official_result(m)
                toks = m.get("clobTokenIds")
                toks = json.loads(toks) if isinstance(toks, str) else (toks or [None, None])
                cache[f"polymarket:{m['id']}"] = {
                    "result": res, "status": m.get("umaResolutionStatus"),
                    "settled_at": m.get("closedTime") if res else None,
                    "yes_token": toks[0], "no_token": toks[1], "abgerufen": iso(now)}


# ------------------------------------------------------------- Bausteine

def market_of(r: dict) -> PredMarket:
    pm = r["platform"] == "polymarket"
    return PredMarket(r["platform"], r["id"], fnum(r["lower"]) or fnum(r["upper"]) or 0.0, r["direction"],
                      ts(r["settle_time"]), "", ticker=None if pm else r["id"], asset=r["asset"],
                      lower=fnum(r["lower"]), upper=fnum(r["upper"]), min_size=D(5) if pm else D(1))


def est_of(r: dict) -> ProbEstimate:
    return ProbEstimate(fnum(r["p_mid"]), fnum(r["p_low"]), fnum(r["p_high"]), r["quality"],
                        tuple(x for x in r["notes"].split("|") if x), fnum(r["m_low"]), fnum(r["m_high"]))


def hedge_of(r: dict, key: str):
    v = fnum(r[key])
    return None if v is None else HedgeCost(v, int(r["hedge_n_min"] or 0), ())


def slip(platform: str, ask: D) -> D:
    return D("0.01") if platform == "kalshi" else min(D("0.01"), (ask * D("0.04")).quantize(D("0.0001")))


def ask_ladder(platform: str, ask: str, depth: str) -> tuple:
    if not ask:
        return ()
    a, dep = D(ask), D(depth or 0)
    if dep <= 0:
        return ()
    top = min(dep, (TOP_USD / a).to_integral_value())
    rest = dep * DEPTH_SHARE - top
    out = [Level(a, top)]
    a2 = a + slip(platform, a)
    if rest > 0 and a2 < 1:
        out.append(Level(a2, rest.to_integral_value()))
    return tuple(out)


def book_of(r: dict) -> Book:
    """Buch aus einer Beobachtung: Asks als Leiter, Bids als Gegenkurs (1 - Brief der Gegenseite)."""
    big = D(10 ** 9)
    yb = (Level(1 - D(r["no_ask"]), D(r["no_depth"] or 0) or big),) if r["no_ask"] else ()
    nb = (Level(1 - D(r["yes_ask"]), D(r["yes_depth"] or 0) or big),) if r["yes_ask"] else ()
    return Book(ask_ladder(r["platform"], r["yes_ask"], r["yes_depth"]),
                ask_ladder(r["platform"], r["no_ask"], r["no_depth"]), yb, nb)


def unreliable_for(fund_base: str, r: dict) -> bool:
    """Zeilen, deren alte Punktschaetzung vom behobenen Fehler F1/F2 stammen kann."""
    if fund_base != "punkt":
        return False
    return r["m_low"] == "" or r["quality"] == "vorlauf"


# ------------------------------------------------------------------ Lauf

def replay(rows: list, results: dict, out_dir: Path, now: datetime, use_filter: bool = True) -> dict:
    fdir = out_dir / "fonds"
    if fdir.exists():
        shutil.rmtree(fdir)  # nur der eigene Ausgabeordner, wird vollstaendig neu erzeugt
    funds = {k: Fund(k, fdir) for k in FUND_KEYS}
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["timestamp"]].append(r)
    runs = sorted(by_run)
    hist, decisions = [], Counter()

    def settle_due(t: datetime) -> None:
        def res(pos):
            info = results.get(f"{pos['platform']}:{pos['market_id']}") or {}
            if not info.get("result"):
                return None
            at = ts(info["settled_at"]) if info.get("settled_at") else ts(pos["settle_time"]) + SETTLE_FALLBACK
            return info["result"] if at <= t else None
        for f in funds.values():
            f.settle(res, t)

    def snapshot(t: datetime, current: dict, label: str) -> None:
        row = {"time": iso(t), "reihe": label}
        for k, f in funds.items():
            nav, flagged = f.nav(lambda p: current.get((p["platform"], p["market_id"])))
            row[f"nav_{k}"] = str(nav.quantize(D("0.01")))
            row[f"offen_{k}"] = len(f.positions)
        hist.append(row)

    for run in runs:
        t = ts(run)
        rs = by_run[run]
        current = {(r["platform"], r["id"]): book_of(r) for r in rs}
        navs = {k: f.nav(lambda p: current.get((p["platform"], p["market_id"])))[0] for k, f in funds.items()}
        for r in rs:
            if r["quelle"] != "buch":
                continue
            moved = r["decision_streng"] == "SKIP:kurs_bewegt"
            mk, est, b = market_of(r), est_of(r), current[(r["platform"], r["id"])]
            for k, f in funds.items():
                if moved:
                    d = "SKIP:kurs_bewegt"
                elif use_filter and unreliable_for(f.base, r):
                    d = "SKIP:alte_schaetzung_unzuverlaessig"
                else:
                    hedge = hedge_of(r, "hedge_model" if f.band in ("modell", "punkt") else "hedge_exec")
                    d = f.consider(mk, r["quality"], est, b, t, navs[k], hedge=hedge)
                    if d.startswith("BUY"):
                        navs[k] = f.nav(lambda p: current.get((p["platform"], p["market_id"])))[0]
                decisions[f"{k}:{d}"] += 1
        settle_due(t)
        snapshot(t, current, "nachgerechnet")
    settle_due(now)  # Schlussschritt: nur abrechnen, was bis jetzt offiziell ist
    snapshot(now, {}, "nachgerechnet_schluss")
    for k, f in funds.items():
        assert f.invariant_ok(), k
        for pos in f.positions.values():  # Token fuer die Live-Fortsetzung
            info = results.get(f"{pos['platform']}:{pos['market_id']}") or {}
            if pos["platform"] == "polymarket":
                pos["yes_token"], pos["no_token"] = info.get("yes_token"), info.get("no_token")
            else:
                pos["ticker"] = pos["market_id"]
        f.save()
    return {"funds": funds, "hist": hist, "decisions": decisions}


# ------------------------------------------------------------- Kennzahlen

def trade_stats(f: Fund) -> dict:
    buys, settles = {}, []
    path = f.trades_path
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for t in csv.DictReader(fh):
                key = f"{t['platform']}:{t['market']}"
                if t["kind"] == "BUY":
                    buys[key] = (-D(t["cash_flow"]), t)
                else:
                    cost = buys[key][0]
                    settles.append((D(t["cash_flow"]) - cost, key, t["time"]))
    pnl = [p for p, _, _ in settles]
    best = max(pnl) if pnl else D(0)
    return {"kaeufe": len(buys), "abgerechnet": len(settles), "gewonnen": sum(1 for p in pnl if p > 0),
            "realisiert": float(sum(pnl, D(0))), "bester": float(best), "schlechtester": float(min(pnl) if pnl else 0),
            "realisiert_ohne_besten": float(sum(pnl, D(0)) - best)}


def series_stats(values: list, start: float) -> dict:
    peak, mdd = start, 0.0
    for v in values:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
    rets = [b / a - 1 for a, b in zip([start] + values[:-1], values) if a > 0]
    return {"ende": values[-1] if values else start, "rendite": (values[-1] / start - 1) if values else 0.0,
            "max_rueckgang": mdd, "schwankung_je_lauf": statistics.pstdev(rets) if len(rets) > 1 else 0.0}


def opportunities(rows: list) -> dict:
    """Wie oft und wie gross: Maerkte mit Orderbuch, deren Kante am besten Brief die Huerde schafft."""
    rules = {"interpoliert": ("edge_huelle_hedge", 0.02), "modell": ("edge_modell_hedge", 0.02),
             "modell_roh": ("edge_modell_roh", 0.02), "punkt": ("edge_punkt_roh", 0.01)}
    runs = {r["timestamp"] for r in rows}
    out = {}
    for name, (col, hurdle) in rules.items():
        hits = [r for r in rows if r["quelle"] == "buch" and r[col] and float(r[col]) >= hurdle]
        edges = sorted(float(r[col]) for r in hits)
        out[name] = {"treffer": len(hits), "maerkte": len({(r["platform"], r["id"]) for r in hits}),
                     "laeufe_mit_treffer": len({r["timestamp"] for r in hits}), "laeufe": len(runs),
                     "kante_median": edges[len(edges) // 2] if edges else None, "kante_max": edges[-1] if edges else None}
    n_min = sorted(int(r["hedge_n_min"]) for r in rows if r["hedge_n_min"] and r["quelle"] == "buch")
    out["hedge_mindestgroesse_kontrakte"] = {"median": n_min[len(n_min) // 2] if n_min else None,
                                             "p10": n_min[len(n_min) // 10] if n_min else None,
                                             "p90": n_min[9 * len(n_min) // 10] if n_min else None}
    hc = sorted(float(r["hedge_model"]) for r in rows if r["hedge_model"] and r["quelle"] == "buch")
    out["hedgekosten_je_kontrakt_median"] = hc[len(hc) // 2] if hc else None
    return out


def calibration(rows: list, results: dict) -> dict:
    """Brier-Wert (kleiner = besser) von Modell und Markt gegen das echte Ergebnis.

    Je Markt die letzte Beobachtung mit Orderbuch, die mindestens 2 Stunden vor der Abrechnung
    liegt und Markt-Mitte, Punktschaetzung und Modellband hat (gemeinsame Stichprobe)."""
    last = {}
    for r in rows:
        if r["quelle"] != "buch" or not (r["yes_ask"] and r["no_ask"] and r["p_mid"] and r["m_low"]):
            continue
        if ts(r["settle_time"]) - ts(r["timestamp"]) < timedelta(hours=2):
            continue
        last[(r["platform"], r["id"])] = r
    sc = defaultdict(list)
    for (p, i), r in last.items():
        res = (results.get(f"{p}:{i}") or {}).get("result")
        if res not in ("yes", "no"):
            continue
        y = 1.0 if res == "yes" else 0.0
        mkt = (float(r["yes_ask"]) + 1 - float(r["no_ask"])) / 2
        model = (float(r["m_low"]) + float(r["m_high"])) / 2
        for name, q in (("markt_mitte", mkt), ("punktschaetzung", float(r["p_mid"])), ("modellband_mitte", model)):
            sc[name].append((q - y) ** 2)
        sc["n"].append(1)
    n = len(sc["n"])
    out = {"n": n}
    for name in ("markt_mitte", "punktschaetzung", "modellband_mitte"):
        out[name] = round(sum(sc[name]) / n, 5) if n else None
    if n:  # Standardfehler der Differenz Modell - Markt (gepaart)
        diffs = [a - b for a, b in zip(sc["modellband_mitte"], sc["markt_mitte"])]
        out["diff_modell_minus_markt"] = round(statistics.mean(diffs), 5)
        out["diff_standardfehler"] = round(statistics.pstdev(diffs) / math.sqrt(n), 5)
    return out


def write_outputs(out_dir: Path, res: dict, rows: list, results: dict, use_filter: bool, now: datetime) -> dict:
    hist, funds = res["hist"], res["funds"]
    keys = list(FUND_KEYS)
    with (out_dir / "verlauf.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "reihe"] + [f"nav_{k}" for k in keys] + [f"offen_{k}" for k in keys])
        w.writeheader()
        w.writerows(hist)
    fonds = {}
    for k, f in funds.items():
        vals = [float(h[f"nav_{k}"]) for h in hist]
        realized, open_cost = f.pnl_split()
        fonds[k] = {"gruppe": f.profile.gruppe, "start": float(f.start), **series_stats(vals, float(f.start)),
                    **trade_stats(f), "offen": len(f.positions), "einstand_offen": float(open_cost)}
    summary = {"erzeugt": iso(now), "filter_alte_schaetzung": use_filter,
               "laeufe": len({r["timestamp"] for r in rows}), "von": rows[0]["timestamp"], "bis": rows[-1]["timestamp"],
               "fonds": fonds, "chancen": opportunities(rows), "kalibrierung": calibration(rows, results),
               "entscheidungen": dict(sorted(res["decisions"].items()))}
    (out_dir / "zusammenfassung.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def uebernehmen(data_dir: Path, out_dir: Path, funds: dict, stamp: str) -> Path:
    """Einmalig: korrigierte Reihe wird der Live-Stand, der alte Stand wandert ins Archiv.

    Verschiebt ``fund_*.json``/``trades_*.csv`` nach ``data/archiv/<stamp>_vor_korrektur/`` (kopiert
    dort auch ``berichte/verlauf.csv`` und ``last_scan.json``), legt die nachgerechneten Fonds an ihre
    Stelle, ersetzt in ``verlauf.csv`` die NAV-Spalten durch die nachgerechneten (Spalte ``reihe``)
    und in ``last_scan.json`` die Fondswerte. Nichts wird geloescht."""
    arch = data_dir / "archiv" / f"{stamp}_vor_korrektur"
    if arch.exists():
        raise SystemExit(f"Archiv {arch} gibt es schon; Uebernahme war bereits erfolgt")
    arch.mkdir(parents=True)
    for f in sorted(list(data_dir.glob("fund_*.json")) + list(data_dir.glob("trades_*.csv"))):
        shutil.move(str(f), str(arch / f.name))
    for rel in ("berichte/verlauf.csv", "last_scan.json"):
        if (data_dir / rel).exists():
            shutil.copy2(data_dir / rel, arch / Path(rel).name)
    for f in sorted((out_dir / "fonds").iterdir()):
        shutil.copy2(f, data_dir / f.name)
    with (out_dir / "verlauf.csv").open(newline="", encoding="utf-8") as fh:
        corr = {r["time"]: r for r in csv.DictReader(fh)}
    vpath = data_dir / "berichte" / "verlauf.csv"
    with vpath.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        head, old = list(rd.fieldnames), list(rd)
    head = head + [c for c in [f"nav_{k}" for k in FUND_KEYS] + ["reihe"] if c not in head]
    for r in old:
        c = corr.get(r["time"])
        r["reihe"] = "nachgerechnet" if c else "alt_ohne_beobachtung"
        for k in FUND_KEYS:
            r[f"nav_{k}"] = c[f"nav_{k}"] if c else ""
    with vpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(old)
    lpath = data_dir / "last_scan.json"
    last = json.loads(lpath.read_text(encoding="utf-8")) if lpath.exists() else {}
    q = D("0.01")
    end = {k: v for k, v in list(corr.values())[-1].items()}
    navs = {}
    for k, f in funds.items():
        nav = D(end[f"nav_{k}"])
        realized, open_cost = f.pnl_split()
        navs[k] = {"nav": str(nav), "cash": str(f.cash.quantize(q)), "offen": len(f.positions),
                   "ohne_bid": 0, "start": str(f.start), "gruppe": f.profile.gruppe,
                   "realisiert": str(realized.quantize(q)), "unrealisiert": str((nav - f.cash - open_cost).quantize(q))}
    last["funds"] = navs
    last["funds_quelle"] = f"nachgerechnet {stamp} (tools/nachrechnen.py), alte Werte in {arch.relative_to(data_dir).as_posix()}"
    lpath.write_text(json.dumps(last, indent=1), encoding="utf-8")
    return arch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=None, help="Standard: <data-dir>/korrigiert")
    ap.add_argument("--offline", action="store_true", help="nur ergebnisse.json, kein Abruf")
    ap.add_argument("--ohne-filter", action="store_true", help="alte Punktschaetzungen nicht ausfiltern")
    ap.add_argument("--jetzt", default=None, help="Schlusszeitpunkt (ISO, Standard: jetzt bzw. letzter Abruf)")
    ap.add_argument("--uebernehmen", metavar="DATUM", default=None,
                    help="einmalig: korrigierten Stand als Live-Stand uebernehmen, alten archivieren")
    args = ap.parse_args(argv)
    out = args.out or args.data_dir / "korrigiert"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_observations(args.data_dir)
    cache_path = args.data_dir / "korrigiert" / "ergebnisse.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    now = ts(args.jetzt) if args.jetzt else datetime.now(UTC).replace(microsecond=0)
    if not args.offline:
        # abgelaufene Maerkte (Ergebnis) und alle Polymarket-Maerkte (Token fuer die Live-Bewertung)
        keys = sorted({(r["platform"], r["id"]) for r in rows
                       if ts(r["settle_time"]) < now or r["platform"] == "polymarket"})
        try:
            fetch_results(keys, cache, now)
        except FetchError as exc:
            print(f"Ergebnisse unvollstaendig abgerufen ({exc}); weiter mit dem Zwischenspeicher", file=sys.stderr)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache["_stand"] = iso(now)
        cache_path.write_text(json.dumps(dict(sorted(cache.items())), indent=0), encoding="utf-8")
    elif not args.jetzt and cache.get("_stand"):
        now = ts(cache["_stand"])  # offline reproduzierbar: Schluss = Stand der Ergebnisse
    res = replay(rows, cache, out, now, use_filter=not args.ohne_filter)
    s = write_outputs(out, res, rows, cache, not args.ohne_filter, now)
    for k, v in s["fonds"].items():
        print(f"{k:18s} {v['gruppe']:9s} Ende {v['ende']:10.2f}  Rendite {100 * v['rendite']:+7.2f} %  "
              f"Kaeufe {v['kaeufe']:3d}  abgerechnet {v['abgerechnet']:3d} (gewonnen {v['gewonnen']:3d})  "
              f"max. Rueckgang {100 * v['max_rueckgang']:5.1f} %")
    print("Kalibrierung:", s["kalibrierung"])
    if args.uebernehmen:
        print("Archiv:", uebernehmen(args.data_dir, out, res["funds"], args.uebernehmen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
