"""Papier-Fonds (keine echten Trades). Alle Geldbetraege sind Decimal.

Fuenf Beispielfonds (``FUNDS``), je 100 USD, vom strengsten zum naivsten:

- "streng": nur exakte Zeit, modellfreies Arbitrageband, mit Hedge-Kosten.
- "interpoliert": alle Qualitaeten, Huellband, mit Hedge-Kosten.
- "modell": alle Qualitaeten, Modellband (SVI-Smile, Varianz-Uhr), mit Hedge-Kosten.
- "modell_roh": wie "modell", aber ohne Hedge-Kosten (unabgesicherte Wert-Wette).
- "punkt": nur die Punktschaetzung, ohne Band und Hedge, Huerde 1 Punkt (naiv, zum Vergleich).

Groessenstufen (``SIZES``, ``FUND_KEYS``): jeder Fonds zusaetzlich mit 1.000 USD (``<name>_1k``)
und 10.000 USD (``<name>_10k``), gleiche Regeln. Der feste Deckel je Trade (10 USD bei 100 USD)
waechst mit dem Startkapital, der Deckel von 5 % NAV bleibt; so zeigt nur die Buchtiefe, ob mehr
Kapital dieselbe Strategie traegt.

Mit Hedge-Kosten: Einstieg nur, wenn die Kante nach Prognosemarkt-Gebuehr UND nach den Kosten
einer Deribit-Absicherung mindestens ``min_edge`` betraegt und die Position selbst mindestens die
Hedge-Mindestgroesse erreicht (sonst liesse sie sich bei Deribit gar nicht absichern; seit
27.10.2026, vorher reichte die Buchtiefe). Gehandelt wird immer nur auf Papier; das Hedge-Bein
wird nicht simuliert, der Fondswert ist der des Prognosemarkt-Beins.

Gruppen (``Profile.gruppe``): "kern" = streng, interpoliert, modell (die erklaerte Strategie);
"vergleich" = modell_roh, punkt (bewusst ungesicherte Wetten als Vergleichsgruppe).

Fuer alle Fonds gilt (seit 27.10.2026):

- Kosten: Kauf zum Brief (Asks, Buch wird abgelaufen) plus Prognosemarkt-Gebuehr (Kalshi je
  Preisstufe auf den Cent aufgerundet); die Huerde ``min_edge`` gilt NACH diesen Kosten. Die
  Geld-Brief-Spanne ist beim Halten bis zur Abrechnung kein weiterer Kostenpunkt; sie zeigt sich
  nur in der Bewertung zum Geldkurs (sofort verkaufbarer Wert).
- Positionsgrenzen: je Basiswert und Abrechnungstag (New Yorker Datum) hoechstens
  ``CLUSTER_CAP`` des NAV an Einstandskosten, je Basiswert insgesamt hoechstens ``ASSET_CAP``.
  Viele Schwellen desselben Verfalls sind wirtschaftlich eine einzige Richtungswette.
- Bewertung zu ausfuehrbaren Bids; ist kein Bid lesbar (Markt geschlossen, Buch weg), zaehlt
  der letzte gesehene Geldkurs (``mark``), markiert, statt 0.
- Abrechnung nur bei offiziellem Ergebnis. Positionen werden bis zur Abrechnung gehalten.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from .fees import fee_per_share_estimate, kalshi_fee, polymarket_taker_fee
from .hedge import HedgeCost
from .http import FetchError

log = logging.getLogger(__name__)
ZERO = Decimal(0)
ONE = Decimal(1)
START_CASH = Decimal(100)
MIN_EDGE = Decimal("0.02")
NAV_CAP = Decimal("0.05")
TRADE_CAP = Decimal(10)  # bei START_CASH; waechst mit dem Startkapital (Fund.trade_cap)
ALL_Q = frozenset({"exakt", "interpoliert", "vorlauf"})
CLUSTER_CAP = Decimal("0.10")  # je Basiswert und Abrechnungstag: Einstandskosten <= 10 % NAV
ASSET_CAP = Decimal("0.25")  # je Basiswert ueber alle Verfaelle: <= 25 % NAV
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Profile:
    qualities: frozenset
    band: str  # "arb" | "huelle" | "modell" | "punkt"
    hedge: bool  # Hedge-Kosten und -Mindestgroesse im Einstiegskriterium
    min_edge: Decimal
    label: str
    desc: str
    gruppe: str = "kern"  # "kern" (erklaerte Strategie) | "vergleich" (ungesicherte Wette)


GRUPPEN = {"kern": "Kernstrategie (mit Hedge-Kosten)", "vergleich": "Vergleich: ungesicherte Wette"}

FUNDS = {
    # Bezeichnungen und Beschreibungen erscheinen im Dashboard (daher echte Umlaute).
    "streng": Profile(frozenset({"exakt"}), "arb", True, MIN_EDGE, "Arbitrage (streng)",
                      "Nur Märkte, die genau zum Deribit-Verfall abrechnen, gegen das modellfreie Band aus "
                      "ausführbaren Spreads, nach Gebühr und Hedge-Kosten. Das wäre echte Arbitrage. "
                      "Derzeit rechnet kein Markt um 08:00 UTC ab, deshalb handelt er nie."),
    "interpoliert": Profile(ALL_Q, "huelle", True, MIN_EDGE, "Hülle mit Hedge",
                            "Alle Märkte gegen die Hülle der Bänder der Nachbarverfälle (Annahme: die "
                            "Wahrscheinlichkeit liegt zeitlich dazwischen), nach Gebühr und Hedge-Kosten; "
                            "die Position muss die Hedge-Mindestgröße erreichen."),
    "modell": Profile(ALL_Q, "modell", True, MIN_EDGE, "Modell mit Hedge",
                      "Gegen das Modellband (SVI-Smile, Varianz-Uhr), nach Gebühr und Hedge-Kosten; "
                      "die Position muss die Hedge-Mindestgröße erreichen."),
    "modell_roh": Profile(ALL_Q, "modell", False, MIN_EDGE, "Modell ohne Hedge",
                          "Vergleich: ungesicherte Wette. Gegen das Modellband, nur nach "
                          "Prognosemarkt-Gebühr, ohne Absicherung. Wettet darauf, dass das Modell recht "
                          "hat; Einzelverluste sind normal.", "vergleich"),
    "punkt": Profile(ALL_Q, "punkt", False, Decimal("0.01"), "Punktschätzung (naiv)",
                     "Vergleich: ungesicherte Wette. Nur gegen die Punktschätzung ohne Unsicherheitsband, "
                     "Hürde 1 Punkt nach Gebühr. Bewusst naiv, als Vergleich dafür, was Band und "
                     "Absicherung bringen.", "vergleich"),
}
QUALITIES = {name: pr.qualities for name, pr in FUNDS.items()}
# Groessenstufen: Endung des Fondsschluessels -> Startkapital. Reihenfolge der Schluessel: erst die
# fuenf 100-USD-Fonds (wie bisher), dann alle 1k, dann alle 10k.
SIZES = {"": START_CASH, "_1k": Decimal(1000), "_10k": Decimal(10000)}
FUND_KEYS = {f"{base}{suffix}": (base, start) for suffix, start in SIZES.items() for base in FUNDS}
HEDGE_FREI = HedgeCost(0.0, 0, ())  # nur fuer Tests: Absicherung kostenlos und ohne Mindestgroesse
TRADE_FIELDS = ["time", "kind", "platform", "market", "side", "shares", "avg_price",
                "fee", "cash_flow", "cash_after", "note"]


def _d(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def level_fee(platform: str, qty: Decimal, price: Decimal) -> Decimal:
    return polymarket_taker_fee(qty, price) if platform == "polymarket" else kalshi_fee(qty, price)


@dataclass
class Fill:
    shares: Decimal
    cost: Decimal  # nur Kurswert
    fee: Decimal

    @property
    def total(self) -> Decimal:
        return self.cost + self.fee


def plan_buy(platform: str, asks, limit: Decimal, budget: Decimal,
             min_size: Decimal) -> Optional[Fill]:
    """Laeuft die Asks (aufsteigend) ab, solange Preis + Gebuehr je Anteil < ``limit``.

    Nur ganze Anteile (Kalshi: ganze Kontrakte), Gesamtausgabe <= ``budget``,
    Tiefe wird nie ueberschritten. Gibt None, wenn nichts oder weniger als
    ``min_size`` uebrig bleibt.
    """
    levels = []  # [price, qty]
    left = budget
    for lv in asks:
        per = lv.price + fee_per_share_estimate(platform, lv.price)
        if lv.price <= 0 or lv.price >= 1 or per >= limit:
            break
        by_budget = (left / per).to_integral_value(ROUND_FLOOR)
        if by_budget <= 0:
            break
        qty = min(lv.size.to_integral_value(ROUND_FLOOR), by_budget)
        if qty <= 0:  # Stufe mit weniger als einem Anteil (Bruchteile): naechste Stufe pruefen
            continue
        levels.append([lv.price, qty])
        left -= qty * per

    def totals():
        shares = sum((q for _, q in levels), ZERO)
        cost = sum((p * q for p, q in levels), ZERO)
        fee = sum((level_fee(platform, q, p) for p, q in levels), ZERO)
        return shares, cost, fee

    while levels:
        shares, cost, fee = totals()
        if shares > 0 and cost + fee <= budget and (cost + fee) / shares < limit:
            break
        levels[-1][1] -= 1  # exakte Gebuehr (Aufrundung) war zu hoch: einen Anteil weniger
        if levels[-1][1] <= 0:
            levels.pop()
    if not levels:
        return None
    shares, cost, fee = totals()
    if shares < min_size:
        return None
    return Fill(shares, cost, fee)


def edges(platform: str, p_low, p_high, book, hedge_cost=ZERO) -> dict:
    """Kanten am besten Ask nach Gebuehr und Hedge-Kosten:
    yes = p_low - (ask+fee) - hedge, no = (1-p_high) - (ask+fee) - hedge."""
    out = {"yes_ask": None, "no_ask": None, "yes_depth": ZERO, "no_depth": ZERO,
           "yes_edge": None, "no_edge": None}
    hc = _d(hedge_cost)
    if book.yes_asks:
        a = book.yes_asks[0].price
        out["yes_ask"], out["yes_depth"] = a, sum((l.size for l in book.yes_asks), ZERO)
        if p_low is not None:
            out["yes_edge"] = _d(p_low) - (a + fee_per_share_estimate(platform, a)) - hc
    if book.no_asks:
        a = book.no_asks[0].price
        out["no_ask"], out["no_depth"] = a, sum((l.size for l in book.no_asks), ZERO)
        if p_high is not None:
            out["no_edge"] = (ONE - _d(p_high)) - (a + fee_per_share_estimate(platform, a)) - hc
    return out


def depth_below(platform: str, asks, limit: Decimal) -> Decimal:
    """Anteile, die zu Preis + Gebuehr unter ``limit`` im Buch liegen."""
    total = ZERO
    for lv in asks:
        if lv.price <= 0 or lv.price + fee_per_share_estimate(platform, lv.price) >= limit:
            break
        total += lv.size
    return total


def band_for(kind: str, est):
    """(low, high) des Bands, mit dem ein Fonds arbeitet; (None, None) wenn es fehlt."""
    if kind == "modell":
        return est.m_low, est.m_high
    if kind == "punkt":
        return est.p_mid, est.p_mid
    return est.p_low, est.p_high


def mark_value(side: str, shares: Decimal, book) -> tuple:
    """Wert nur zu ausfuehrbaren Bids; nicht Gedecktes zaehlt 0. -> (Wert, markiert)."""
    if book is None:
        return ZERO, True
    bids = book.yes_bids if side == "yes" else book.no_bids
    left, value = shares, ZERO
    for lv in bids:
        q = min(left, lv.size)
        value += q * lv.price
        left -= q
        if left <= 0:
            break
    return value, left > 0


def cluster_of(asset: str, settle_time) -> str:
    """Klumpen: Basiswert und Abrechnungstag (New Yorker Datum). Kalshi 17:00 ET und Polymarket
    12:00 ET desselben Tages haengen am selben Kurs und zaehlen zusammen."""
    if isinstance(settle_time, str):
        settle_time = datetime.strptime(settle_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return f"{asset}:{settle_time.astimezone(NY):%Y-%m-%d}"


class Fund:
    def __init__(self, name: str, data_dir: Path, start: Optional[Decimal] = None):
        """``name``: Schluessel aus ``FUND_KEYS`` (z. B. "punkt" oder "punkt_10k")."""
        self.name = name
        self.base, tier_start = FUND_KEYS[name]
        self.profile = FUNDS[self.base]
        self.allowed, self.band = self.profile.qualities, self.profile.band
        self.dir = Path(data_dir)
        self.start = tier_start if start is None else start
        self.cash = self.start
        self.positions: dict = {}
        self.flows: list = []  # alle Cash-Flows (Invariante)
        self.state_path = self.dir / f"fund_{name}.json"
        self.trades_path = self.dir / f"trades_{name}.csv"
        self._load()

    # ---------------------------------------------------------- Persistenz
    def _load(self) -> None:
        if self.state_path.exists():
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.start, self.cash = Decimal(s["start"]), Decimal(s["cash"])
            self.positions = {k: v for k, v in s["positions"].items()}
        if self.trades_path.exists():
            with self.trades_path.open(newline="", encoding="utf-8") as f:
                self.flows = [Decimal(r["cash_flow"]) for r in csv.DictReader(f)]

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        s = {"name": self.name, "start": str(self.start), "cash": str(self.cash),
             "positions": self.positions}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def _log_trade(self, now: datetime, kind: str, pos: dict, shares, avg, fee, flow, note="") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.trades_path.exists()
        with self.trades_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(TRADE_FIELDS)
            w.writerow([now.strftime("%Y-%m-%dT%H:%M:%SZ"), kind, pos["platform"], pos["market_id"],
                        pos["side"], shares, avg, fee, flow, self.cash, note])
        self.flows.append(flow)

    @property
    def trade_cap(self) -> Decimal:
        """Fester Deckel je Trade: 10 USD je 100 USD Startkapital."""
        return TRADE_CAP * self.start / START_CASH

    def invariant_ok(self) -> bool:
        return self.cash == self.start + sum(self.flows, ZERO)

    # ---------------------------------------------------------- Bewertung
    def nav(self, book_for: Callable) -> tuple:
        """(NAV, Zahl markierter Positionen). ``book_for(position) -> Book | None``.

        Bewertung zu ausfuehrbaren Bids. Gibt es keinen Bid (Buch nicht lesbar, Markt nach
        Handelsschluss leer), zaehlt der zuletzt gesehene Geldkurs je Anteil (``pos["mark"]``)
        statt 0; die Position gilt dann als markiert. Der Geldkurs wird dabei fortgeschrieben."""
        total, flagged = self.cash, 0
        for pos in self.positions.values():
            shares = Decimal(pos["shares"])
            book = book_for(pos)
            bids = None if book is None else (book.yes_bids if pos["side"] == "yes" else book.no_bids)
            if bids:
                v, flag = mark_value(pos["side"], shares, book)
                pos["mark"] = str((v / shares).quantize(Decimal("0.000001")))
            else:
                v, flag = shares * Decimal(pos.get("mark") or "0"), True
            total += v
            flagged += 1 if flag else 0
        return total, flagged

    def exposure(self, asset: str, cluster: str) -> tuple:
        """(Einstandskosten im Klumpen, Einstandskosten im Basiswert) der offenen Positionen."""
        in_cluster = in_asset = ZERO
        for pos in self.positions.values():
            if pos.get("asset") != asset:
                continue
            c = Decimal(pos["cost"])
            in_asset += c
            if cluster_of(pos["asset"], pos["settle_time"]) == cluster:
                in_cluster += c
        return in_cluster, in_asset

    def pnl_split(self) -> tuple:
        """(realisiert, Einstand offen): realisiert = Kasse + Einstand offen - Start."""
        open_cost = sum((Decimal(p["cost"]) for p in self.positions.values()), ZERO)
        return self.cash + open_cost - self.start, open_cost

    # ----------------------------------------------------------- Einstieg
    def consider(self, market, quality: str, est, book, now: datetime, nav: Decimal,
                 min_edge: Optional[Decimal] = None, *, hedge: Optional[HedgeCost]) -> str:
        """Prueft und fuehrt (auf Papier) hoechstens einen Einstieg aus. -> Entscheidungstext.

        ``hedge`` (Pflicht): Kosten und Mindestgroesse der Absicherung; None = nicht absicherbar.
        Fonds ohne Hedge im Profil ignorieren es. Bei Fonds mit Hedge muss die Position selbst
        mindestens ``hedge.n_min`` Kontrakte gross sein (sonst waere sie nicht absicherbar).
        Budget: 5 % NAV, fester Deckel, Kasse und die Positionsgrenzen je Klumpen/Basiswert."""
        min_edge = self.profile.min_edge if min_edge is None else min_edge
        if not self.profile.hedge:
            hedge = HEDGE_FREI
        if quality not in self.allowed:
            return "SKIP:qualitaet"
        key = f"{market.platform}:{market.id}"
        if key in self.positions:
            return "HELD"
        lo, hi = band_for(self.band, est)
        if lo is None or hi is None:
            return "SKIP:kein_band"
        if hedge is None:
            return "SKIP:kein_hedge"
        p_low, p_high = _d(lo), _d(hi)
        hc = _d(round(hedge.per_contract, 6))
        asset = getattr(market, "asset", "BTC")
        in_cluster, in_asset = self.exposure(asset, cluster_of(asset, market.settle_time))
        room = {"limit_verfall": nav * CLUSTER_CAP - in_cluster, "limit_basiswert": nav * ASSET_CAP - in_asset}
        budget = min(nav * NAV_CAP, self.trade_cap, self.cash, *room.values())
        binding = next((k for k, v in room.items() if v <= budget), None)  # welche Grenze bremst
        if self.cash <= 0:
            return "SKIP:kein_geld"
        if budget <= 0:
            return f"SKIP:{binding}" if binding else "SKIP:kein_geld"
        min_shares = max(_d(market.min_size), _d(hedge.n_min) if self.profile.hedge else ZERO, ONE)
        tried, thin, short = False, False, False
        for side, asks, limit in (("yes", book.yes_asks, p_low - min_edge - hc),
                                  ("no", book.no_asks, ONE - p_high - min_edge - hc)):
            if not asks or limit <= 0:
                continue
            tried = True
            depth = depth_below(market.platform, asks, limit)
            if depth < min_shares:
                if depth > 0 and not thin:
                    thin = "zu_duenn" if depth < max(_d(market.min_size), ONE) else "zu_duenn_fuer_hedge"
                continue
            fill = plan_buy(market.platform, asks, limit, budget, min_shares)
            if fill is None:
                p0 = asks[0].price
                per = p0 + fee_per_share_estimate(market.platform, p0)
                if budget < min_shares * per:  # Kante da, aber das Geld reicht nicht fuer die Mindestgroesse
                    short = short or ("hedge" if budget >= max(_d(market.min_size), ONE) * per else "geld")
                continue
            self._open(market, side, fill, now, key, quality, hedge)
            return f"BUY_{side.upper()}"
        if short:
            if short == "hedge":
                return "SKIP:unter_hedge_mindestgroesse"
            return f"SKIP:{binding}" if binding else "SKIP:kein_geld"
        if thin:
            return f"SKIP:{thin}"
        return "SKIP:keine_kante" if tried else "SKIP:buch_leer"

    def _open(self, market, side: str, fill: Fill, now: datetime, key: str,
              quality: str = "", hedge: Optional[HedgeCost] = None) -> None:
        pos = {"platform": market.platform, "market_id": market.id, "ticker": market.ticker,
               "yes_token": market.yes_token, "no_token": market.no_token, "side": side,
               "shares": str(fill.shares), "cost": str(fill.total),
               "threshold": market.threshold, "settle_time": market.settle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "asset": getattr(market, "asset", "BTC"), "direction": market.direction,
               "lower": getattr(market, "lower", None), "upper": getattr(market, "upper", None),
               "quality": quality, "band": self.band,
               "hedge_cost": None if hedge is None else round(hedge.per_contract, 6)}
        self.positions[key] = pos
        self.cash -= fill.total
        self._log_trade(now, "BUY", pos, fill.shares, fill.cost / fill.shares, fill.fee, -fill.total)
        self.save()

    # ---------------------------------------------------------- Abrechnung
    def settle(self, result_for: Callable, now: datetime) -> int:
        """``result_for(position) -> 'yes' | 'no' | None``. Netzwerkfehler rechnen nie ab."""
        n = 0
        for key in list(self.positions):
            pos = self.positions[key]
            try:
                res = result_for(pos)
            except FetchError as exc:
                log.warning("Abrechnung %s uebersprungen: %s", key, exc)
                continue
            if res not in ("yes", "no"):
                continue
            shares = Decimal(pos["shares"])
            payout = shares if res == pos["side"] else ZERO
            self.cash += payout
            self._log_trade(now, "SETTLE", pos, shares, payout / shares, ZERO, payout, note=f"ergebnis={res}")
            del self.positions[key]
            n += 1
        if n:
            self.save()
        return n
