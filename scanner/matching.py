"""Strenge Zuordnung von Prognosemaerkten zu Deribit-Verfaellen.

Im Zweifel wird verworfen. Jede Funktion liefert ``(ergebnis, None)`` oder
``(None, grund)``; der Grund ist ein kurzer Schluessel fuer die Statistik.

Auszahlungsarten (Phase 3): ``above`` P(S > a), ``below`` P(S < b), ``range`` P(a <= S < b).
Regeltext, Felder und Untertitel muessen jeweils dieselbe Schwelle nennen.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional
from zoneinfo import ZoneInfo

from .assets import ASSETS, Asset, asset_for_kalshi_series
from .prob import GAP_RATIO, MAX_GAP

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
UTC = timezone.utc

BANNED = ("hit", "reach", "dip", "up or down", "touch", "high", "low", "max", "min")
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED) + r")\b", re.I)
_NUM = r"([\d,]+(?:\.\d+)?)"
MIN_LEAD = timedelta(minutes=60)  # kuerzer laufende Maerkte: Kursbewegung zwischen den Abrufen dominiert


@dataclass(frozen=True)
class PredMarket:
    platform: str  # "kalshi" | "polymarket"
    id: str  # Kalshi: Ticker; Polymarket: Gamma-Markt-ID
    threshold: float  # Hauptschwelle (untere, sonst obere)
    direction: str  # "above" | "below" | "range"
    settle_time: datetime  # UTC
    reference: str  # z. B. "CF_RTI" | "BINANCE_BTCUSDT"
    title: str = ""
    ticker: Optional[str] = None
    yes_token: Optional[str] = None
    no_token: Optional[str] = None
    min_size: Decimal = Decimal(1)
    asset: str = "BTC"
    lower: Optional[float] = None
    upper: Optional[float] = None
    list_yes_ask: Optional[Decimal] = None  # Kurse aus der Marktliste, nur zur Vorauswahl
    list_no_ask: Optional[Decimal] = None

    @property
    def bounds(self) -> tuple:
        """(untere, obere) Grenze der JA-Auszahlung."""
        if self.lower is None and self.upper is None and self.direction == "above":
            return self.threshold, None
        return self.lower, self.upper


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _norm(s: str) -> str:
    return (s or "").replace("’", "'").replace("‘", "'")


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _dec(x) -> Optional[Decimal]:
    try:
        d = Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return d if Decimal(0) < d < Decimal(1) else None


# ---------------------------------------------------------------- Kalshi

def parse_kalshi_market(raw: dict, asset: Optional[Asset] = None):
    """Kalshi-Markt (Schwelle oder Spanne) -> PredMarket oder (None, Grund)."""
    try:
        ticker = raw["ticker"]
        asset = asset or asset_for_kalshi_series(ticker.split("-")[0]) or ASSETS["BTC"]
        st = raw.get("strike_type")
        if st not in ("greater", "less", "between"):
            return None, "kalshi_strike_type"
        rules = _norm(raw.get("rules_primary", ""))
        text = " ".join(str(raw.get(k) or "") for k in ("title", "subtitle", "yes_sub_title")) + " " + rules
        if _BANNED_RE.search(text):
            return None, "verbotenes_wort"
        if not any(r in rules for r in asset.kalshi_refs):
            return None, "kalshi_referenz"
        if "simple average of the sixty seconds" not in rules:
            return None, "kalshi_mittelung"
        floor, cap = raw.get("floor_strike"), raw.get("cap_strike")
        other = {"greater": r"\b(below|between)\b", "less": r"\b(above|between)\b",
                 "between": r"\b(above|below)\b"}[st]
        if re.search(other, text, re.I):
            return None, "art_widerspruch"  # Titel oder Regel nennen eine andere Auszahlungsart
        if st == "greater":
            m = re.search(r"\bis above \$?" + _NUM + r"\b", rules)
            if floor is None or not m:
                return None, "kalshi_schwelle_im_text_fehlt"
            lower, upper, direction, main = float(floor), None, "above", float(floor)
            if abs(_num(m.group(1)) - lower) > 0.005:
                return None, "schwelle_widerspruch"
        elif st == "less":
            m = re.search(r"\bis below \$?" + _NUM + r"\b", rules)
            if cap is None or not m:
                return None, "kalshi_schwelle_im_text_fehlt"
            lower, upper, direction, main = None, float(cap), "below", float(cap)
            if abs(_num(m.group(1)) - upper) > 0.005:
                return None, "schwelle_widerspruch"
        else:
            m = re.search(r"\bis between \$?" + _NUM + r"-\$?" + _NUM + r"\b", rules)
            if floor is None or cap is None or not m:
                return None, "kalshi_schwelle_im_text_fehlt"
            lower, upper, direction, main = float(floor), float(cap), "range", float(floor)
            if abs(_num(m.group(1)) - lower) > 0.005 or abs(_num(m.group(2)) - upper) > 0.005:
                return None, "schwelle_widerspruch"
            if upper <= lower:
                return None, "schwelle_widerspruch"
        sub = re.search(r"\$?" + _NUM, str(raw.get("yes_sub_title") or raw.get("subtitle") or ""))
        if sub and abs(_num(sub.group(1)) - (lower if lower is not None else upper)) > 1.0:
            return None, "schwelle_widerspruch"
        close = _parse_time(raw["close_time"])
        t = re.search(r"at (\d{1,2}) (AM|PM) (EDT|EST) on ([A-Za-z]{3}) (\d{1,2}), (\d{4})", rules)
        if not t:
            return None, "kalshi_zeit_im_text_fehlt"
        hour = int(t.group(1)) % 12 + (12 if t.group(2) == "PM" else 0)
        local = datetime.strptime(f"{t.group(4)} {t.group(5)} {t.group(6)}", "%b %d %Y").replace(
            hour=hour, tzinfo=NY)
        if local.tzname() != t.group(3):
            return None, "kalshi_zeitzone_widerspruch"
        if abs(local.astimezone(UTC) - close) > timedelta(minutes=1):
            return None, "kalshi_zeit_widerspruch"
        return PredMarket("kalshi", ticker, main, direction, close, "CF_RTI",
                          title=raw.get("title", ""), ticker=ticker, asset=asset.code,
                          lower=lower, upper=upper,
                          list_yes_ask=_dec(raw.get("yes_ask_dollars")),
                          list_no_ask=_dec(raw.get("no_ask_dollars"))), None
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("Kalshi-Markt unlesbar: %s", exc)
        return None, "kalshi_unlesbar"


# ------------------------------------------------------------ Polymarket

def _event_kind(title: str, asset: Asset) -> Optional[str]:
    name = re.escape(asset.pm_name)
    if re.match(rf"^{name} above ___ on [A-Z][a-z]+ \d{{1,2}}\?$", title):
        return "above"
    if re.match(rf"^{name} price on [A-Z][a-z]+ \d{{1,2}}\?$", title):
        return "range"
    return None


def parse_polymarket_market(raw: dict, asset: Optional[Asset] = None, kind: str = "above"):
    """Polymarket-Markt aus "<Name> above ___ ..." (kind="above") oder "<Name> price on ..."
    (kind="range") -> PredMarket oder (None, Grund)."""
    asset = asset or ASSETS["BTC"]
    try:
        q = _norm(raw.get("question", ""))
        if _BANNED_RE.search(q):
            return None, "verbotenes_wort"
        name = re.escape(asset.pm_name)
        date = r" on ([A-Z][a-z]+) (\d{1,2})\?$"
        group = str(raw.get("groupItemTitle") or "")
        desc = _norm(raw.get("description", ""))
        if kind == "above":
            m = re.match(rf"^Will the price of {name} be above \$" + _NUM + date, q)
            if not m:
                return None, "pm_frage_unbekannt"
            lower, upper, direction = _num(m.group(1)), None, "above"
            if not group or abs(_num(group) - lower) > 1e-9 * max(1.0, lower) + 0.005:
                return None, "schwelle_widerspruch"
            if "higher than" not in desc:
                return None, "pm_richtung"
            month, day = m.group(2), m.group(3)
        else:
            mb = re.match(rf"^Will the price of {name} be between \$" + _NUM + r" and \$" + _NUM + date, q)
            ml = re.match(rf"^Will the price of {name} be less than \$" + _NUM + date, q)
            mg = re.match(rf"^Will the price of {name} be greater than \$" + _NUM + date, q)
            if mb:
                lower, upper, direction = _num(mb.group(1)), _num(mb.group(2)), "range"
                g = re.match(r"^" + _NUM + r"-" + _NUM + r"$", group)
                ok = g and abs(_num(g.group(1)) - lower) < 1e-9 and abs(_num(g.group(2)) - upper) < 1e-9
                month, day = mb.group(3), mb.group(4)
                if upper <= lower:
                    return None, "schwelle_widerspruch"
            elif ml:
                lower, upper, direction = None, _num(ml.group(1)), "below"
                ok = group.startswith("<") and abs(_num(group[1:]) - upper) < 1e-9
                month, day = ml.group(2), ml.group(3)
            elif mg:
                lower, upper, direction = _num(mg.group(1)), None, "above"
                ok = group.startswith(">") and abs(_num(group[1:]) - lower) < 1e-9
                month, day = mg.group(2), mg.group(3)
            else:
                return None, "pm_frage_unbekannt"
            if not ok:
                return None, "schwelle_widerspruch"
            if 'final "Close" price' not in desc:
                return None, "pm_richtung"
        if "12:00 in the ET timezone" not in desc:
            return None, "pm_zeitregel_fehlt"
        if "Binance" not in desc or asset.pm_pair not in desc:
            return None, "pm_referenz"
        end = _parse_time(raw["endDate"])
        year = end.astimezone(NY).year
        mon = datetime.strptime(month, "%B").month
        expected = datetime(year, mon, int(day), 12, 0, tzinfo=NY).astimezone(UTC)
        if abs(expected - end) > timedelta(minutes=1):
            return None, "pm_zeit_widerspruch"
        outcomes = json.loads(raw["outcomes"]) if isinstance(raw.get("outcomes"), str) else raw.get("outcomes")
        if outcomes != ["Yes", "No"]:
            return None, "pm_outcomes"
        tokens = json.loads(raw["clobTokenIds"]) if isinstance(raw["clobTokenIds"], str) else raw["clobTokenIds"]
        if len(tokens) != 2:
            return None, "pm_token"
        best_bid = _dec(raw.get("bestBid"))
        main = lower if lower is not None else upper
        return PredMarket("polymarket", str(raw["id"]), main, direction, expected,
                          "BINANCE_" + asset.pm_pair.replace("/", ""),
                          title=raw.get("question", ""), yes_token=tokens[0], no_token=tokens[1],
                          min_size=Decimal(str(raw.get("orderMinSize") or 5)), asset=asset.code,
                          lower=lower, upper=upper, list_yes_ask=_dec(raw.get("bestAsk")),
                          list_no_ask=None if best_bid is None else Decimal(1) - best_bid), None
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("Polymarket-Markt unlesbar: %s", exc)
        return None, "pm_unlesbar"


def parse_polymarket_events(events: list, assets=None):
    """Alle Zielmaerkte offener Ereignisse. Rueckgabe: (Maerkte, Liste von Ablehnungsgruenden)."""
    assets = list(assets or ASSETS.values())
    markets, rejects, seen = [], [], set()
    for ev in events:
        title = ev.get("title", "")
        hit = [(a, k) for a in assets for k in [_event_kind(title, a)] if k]
        if not hit:
            continue
        asset, kind = hit[0]
        for raw in ev.get("markets", []):
            if str(raw.get("id")) in seen:
                continue
            seen.add(str(raw.get("id")))
            if raw.get("closed") or raw.get("active") is False:
                rejects.append("pm_nicht_offen")
                continue
            mk, why = parse_polymarket_market(raw, asset, kind)
            (markets.append(mk) if mk else rejects.append(why))
    return markets, rejects


# ------------------------------------------------------------- Paarung

@dataclass(frozen=True)
class Pairing:
    quality: str  # "exakt" | "interpoliert" | "vorlauf"
    t1: Optional[datetime]  # None bei "vorlauf" (Anker: jetzt)
    t2: Optional[datetime]


def pair_with_deribit(market: PredMarket, expiries, now: Optional[datetime] = None,
                      tol: timedelta = timedelta(minutes=5), max_gap: timedelta = MAX_GAP,
                      has_spot: bool = True, min_lead: timedelta = MIN_LEAD):
    """``expiries``: Deribit-Verfaelle (Liste oder dict mit Verfaellen als Schluesseln).

    exakt: Abrechnung innerhalb ``tol`` eines Verfalls. interpoliert: von zwei Verfaellen
    eingerahmt, Abstand <= max(max_gap, GAP_RATIO * Restlaufzeit). vorlauf: vor dem ersten
    Verfall, Anker ist der Kassakurs jetzt. Schwellen ausserhalb der Strikes werden nicht
    mehr verworfen: das Band ist dann einseitig (Monotonie), das Modellband fehlt.
    """
    exps = sorted(expiries)
    t = market.settle_time
    if now is not None and t - now < min_lead:
        return None, "zu_kurzfristig"
    for e in exps:
        if abs(e - t) <= tol:
            return Pairing("exakt", e, None), None
    before = [e for e in exps if e < t]
    after = [e for e in exps if e > t]
    if not after:
        return None, "kein_verfall_daneben"
    if not before:
        if not has_spot or now is None:
            return None, "kein_verfall_daneben"
        return Pairing("vorlauf", None, after[0]), None
    t1, t2 = before[-1], after[0]
    rest = (t - now) if now is not None else timedelta(0)
    if t2 - t1 > max(max_gap, GAP_RATIO * rest):
        return None, "verfaelle_zu_weit"
    return Pairing("interpoliert", t1, t2), None
