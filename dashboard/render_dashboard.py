"""Dashboard fuer das Claude-Artefakt "Wahrscheinlichkeits-Monitor".

Baut aus ``data/`` (last_scan.json, berichte/verlauf.csv, fund_<name>.json, trades_<name>.csv)
eine statische HTML-Seite ohne JavaScript. Lokal:

    python dashboard/render_dashboard.py [--data-dir data] [--out dashboard/output.html]

``output.html`` wird nicht committet; es wird nach jedem Bau mit dem Artifact-Werkzeug als neue
Version des bestehenden Artefakts veroeffentlicht.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scanner.fund import FUNDS, MIN_EDGE  # noqa: E402

BERLIN = ZoneInfo("Europe/Berlin")
START = 100.0


# ------------------------------------------------------------------ Lesen

def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _t(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- Format

def num(x, nd=2, sign=False):
    if x is None:
        return "–"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−").replace(".", ",")


def pts(x, sign=True):
    """Wahrscheinlichkeit als Punkte (0,01 = 1 Punkt)."""
    return "–" if x is None else num(100 * x, 1, sign)


def pct(x, nd=1):
    return "–" if x is None else num(100 * x, nd) + "&thinsp;%"


def local(ts: datetime) -> str:
    return ts.astimezone(BERLIN).strftime("%d.%m.%Y, %H:%M")


# ----------------------------------------------------------------- Grafik

def sparkline(values, w=220, h=44, base=None, hurdle=None, kind="line"):
    """SVG-Verlauf, auf die Werte skaliert; ``base``/``hurdle`` als gestrichelte Linien."""
    vals = [v for v in values if v is not None]
    if not vals:
        return '<svg class="spark" viewBox="0 0 220 44" aria-hidden="true"></svg>'
    lo, hi = min(vals + [x for x in (base, hurdle) if x is not None]), max(vals + [x for x in (base, hurdle) if x is not None])
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 4

    def X(i):
        return pad + (w - 2 * pad) * (i / max(len(values) - 1, 1))

    def Y(v):
        return h - pad - (h - 2 * pad) * (v - lo) / (hi - lo)

    parts = [f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img">']
    for ref, cls in ((base, "ref"), (hurdle, "hurdle")):
        if ref is not None:
            parts.append(f'<line class="{cls}" x1="{pad}" x2="{w - pad}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/>')
    pts_ = [(X(i), Y(v)) for i, v in enumerate(values) if v is not None]
    if len(pts_) > 1:
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_)
        y0 = Y(base) if base is not None else h - pad
        parts.append(f'<polygon class="area" points="{pts_[0][0]:.1f},{y0:.1f} {line} {pts_[-1][0]:.1f},{y0:.1f}"/>')
        parts.append(f'<polyline class="line" points="{line}"/>')
    x, y = pts_[-1]
    parts.append(f'<circle class="end" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')
    parts.append("</svg>")
    return "".join(parts)


def band_strip(e: dict) -> str:
    """Bandstreifen: Huelle, Modellband, Punktschaetzung, JA-Brief und 1 - NEIN-Brief (= JA-Geld
    zum Gegenkurs) auf einem gemeinsamen, beschrifteten Ausschnitt der Skala 0 bis 1."""
    p_low, p_high = _f(e.get("p_low")), _f(e.get("p_high"))
    m_low, m_high, p_mid = _f(e.get("m_low")), _f(e.get("m_high")), _f(e.get("p_mid"))
    ya = _f(e.get("yes_ask"))
    na = _f(e.get("no_ask"))
    yb = None if na is None else 1 - na
    vals = [v for v in (p_low, p_high, m_low, m_high, p_mid, ya, yb) if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 0.01)
    lo, hi = max(0.0, lo - 0.25 * span), min(1.0, hi + 0.25 * span)
    W, H = 300, 34

    def X(v):
        return 6 + (W - 12) * (v - lo) / (hi - lo)

    out = [f'<svg class="strip" viewBox="0 0 {W} {H}" role="img" aria-label="Bandstreifen">',
           f'<line class="axis" x1="6" x2="{W - 6}" y1="18" y2="18"/>']
    if p_low is not None and p_high is not None:
        out.append(f'<rect class="hull" x="{X(p_low):.1f}" y="11" width="{max(X(p_high) - X(p_low), 1):.1f}" height="14"/>')
    if m_low is not None and m_high is not None:
        out.append(f'<rect class="model" x="{X(m_low):.1f}" y="13" width="{max(X(m_high) - X(m_low), 1.5):.1f}" height="10"/>')
    if p_mid is not None:
        out.append(f'<line class="mid" x1="{X(p_mid):.1f}" x2="{X(p_mid):.1f}" y1="9" y2="27"/>')
    if ya is not None:
        x = X(ya)
        out.append(f'<path class="ask" d="M{x:.1f},8 l-4,-6 h8 z"/>')
    if yb is not None:
        x = X(yb)
        out.append(f'<path class="bid" d="M{x:.1f},28 l-4,6 h8 z"/>')
    out.append(f'<text class="tick" x="6" y="33">{pct(lo)}</text>'
               f'<text class="tick end" x="{W - 6}" y="33">{pct(hi)}</text></svg>')
    return "".join(out).replace("&thinsp;", " ")


# ------------------------------------------------------------------ Bauen

def market_label(e: dict) -> str:
    d, lo, up = e.get("direction"), e.get("lower"), e.get("upper")

    def n(x):
        v = _f(x)
        if v is None:
            return ""
        return f"{v:,.0f}".replace(",", ".") if v >= 100 else num(v, 2)
    if d == "above":
        return f"über {n(lo)}"
    if d == "below":
        return f"unter {n(up)}"
    return f"{n(lo)} bis {n(up)}"


def render(data_dir: Path) -> str:
    last = _read_json(data_dir / "last_scan.json", {})
    hist = _read_csv(data_dir / "berichte" / "verlauf.csv")
    now = _t(last.get("time", "")) or datetime.now(timezone.utc)
    funds = []
    for name, pr in FUNDS.items():
        state = _read_json(data_dir / f"fund_{name}.json", {"cash": "100", "positions": {}})
        trades = _read_csv(data_dir / f"trades_{name}.csv")
        nav_hist = [_f(r.get(f"nav_{name}")) for r in hist]
        nav = (last.get("funds", {}).get(name) or {}).get("nav")
        funds.append({"name": name, "pr": pr, "state": state, "trades": trades, "nav_hist": nav_hist,
                      "nav": _f(nav) if nav is not None else (nav_hist[-1] if nav_hist else START)})
    best = last.get("best_edges_book", {}) or {}
    hurdle = float(MIN_EDGE)
    best_model = best.get("edge_modell_hedge")
    best_hull = best.get("edge_huelle_hedge")
    if best_model is not None and best_model >= hurdle:
        model_state, model_text = "good", "über der Hürde"
    elif best_model is not None and best_model > 0:
        model_state, model_text = "warn", "positiv, unter der Hürde"
    else:
        model_state, model_text = "neutral", "keine"
    arb_state = "good" if best_hull is not None and best_hull >= hurdle else "neutral"
    runs = len(hist)
    first = _t(hist[0]["time"]) if hist else now

    h = []
    h.append(HEAD)
    h.append('<main class="wrap">')
    h.append('<header class="top"><p class="eyebrow">Forschung · nur Papierhandel · keine Anlageberatung</p>'
             '<h1>Wahrscheinlichkeits-Monitor</h1>'
             '<p class="lede">Prognosemärkte von Kalshi und Polymarket gegen risikoneutrale Wahrscheinlichkeiten '
             'aus Deribit-Optionen für BTC, ETH, SOL und XRP.</p>'
             f'<p class="meta">Stand {local(now)} Uhr · {runs} {"Lauf" if runs == 1 else "Läufe"} seit '
             f'{first.astimezone(BERLIN).strftime("%d.%m.%Y")} · Lauf alle 2 Stunden auf GitHub</p></header>')

    # Lage
    h.append('<section class="status" aria-label="Lage">')
    h.append(f'<div class="stat"><span class="k">Arbitrage nach Kosten</span>'
             f'<span class="v chip {arb_state}">{"ja" if arb_state == "good" else "keine"}</span>'
             f'<span class="s">beste Kante gegen die Hülle {pts(best_hull)} Pkt</span></div>')
    h.append(f'<div class="stat"><span class="k">Beste Modell-Kante nach Hedge</span>'
             f'<span class="v mono">{pts(best_model)}<small> Pkt</small></span>'
             f'<span class="s"><span class="chip {model_state}">{model_text}</span> Hürde +{num(100 * hurdle, 0)} Pkt</span></div>')
    h.append(f'<div class="stat"><span class="k">Zugeordnete Märkte</span>'
             f'<span class="v mono">{last.get("matched", "–")}</span>'
             f'<span class="s">{last.get("with_model_band", "–")} mit Modellband · {last.get("books", "–")} Orderbücher</span></div>')
    h.append(f'<div class="stat"><span class="k">Bandbreite, Median</span>'
             f'<span class="v mono">{num(_f(last.get("median_width_modell")), 3)}</span>'
             f'<span class="s">Modell · Hülle {num(_f(last.get("median_width_huelle")), 3)}</span></div>')
    h.append(f'<div class="stat"><span class="k">Hedge-Kosten, Median</span>'
             f'<span class="v mono">{pts(_f(last.get("median_hedge_model")), False)}<small> Pkt</small></span>'
             f'<span class="s">je Kontrakt, Deribit-Spread</span></div>')
    h.append("</section>")

    # Verlauf der besten Kante
    bm = [_f(r.get("best_modell_hedge")) for r in hist]
    br = [_f(r.get("best_modell_roh")) for r in hist]
    h.append('<section class="block"><div class="head"><h2>Beste Kante je Lauf</h2>'
             '<p>Gegen das Modellband, mit Orderbuch. Gestrichelt: 0 und die Hürde von 2 Punkten.</p></div>'
             '<div class="trend">'
             f'<figure><figcaption>nach Gebühr und Hedge <b class="mono">{pts(bm[-1] if bm else None)}</b></figcaption>'
             f'{sparkline([None if v is None else 100 * v for v in bm], base=0.0, hurdle=100 * hurdle)}</figure>'
             f'<figure><figcaption>nur nach Gebühr <b class="mono">{pts(br[-1] if br else None)}</b></figcaption>'
             f'{sparkline([None if v is None else 100 * v for v in br], base=0.0, hurdle=100 * hurdle)}</figure>'
             f'<figure><figcaption>zugeordnete Märkte <b class="mono">{last.get("matched", "–")}</b></figcaption>'
             f'{sparkline([_f(r.get("matched")) for r in hist])}</figure>'
             '</div></section>')

    # Fonds
    h.append('<section class="block"><div class="head"><h2>Beispielfonds</h2>'
             '<p>Je 100 USD Start, gleiche Märkte, verschiedene Regeln. Bewertung nur zu ausführbaren Geldkursen.</p></div>'
             '<ol class="funds">')
    for f in funds:
        pr, st = f["pr"], f["state"]
        nav = f["nav"]
        chg = None if nav is None else nav - START
        cls = "good" if chg and chg > 0.005 else ("bad" if chg and chg < -0.005 else "neutral")
        buys = sum(1 for t in f["trades"] if t.get("kind") == "BUY")
        sells = sum(1 for t in f["trades"] if t.get("kind") == "SETTLE")
        wins = sum(1 for t in f["trades"] if t.get("kind") == "SETTLE" and _f(t.get("cash_flow")) and _f(t["cash_flow"]) > 0)
        rule = {"arb": "Arbitrageband", "huelle": "Hülle", "modell": "Modellband", "punkt": "Punktschätzung"}[pr.band]
        tags = [rule, "mit Hedge-Kosten" if pr.hedge else "ohne Hedge-Kosten", f"Hürde {num(100 * float(pr.min_edge), 0)} Pkt"]
        h.append('<li class="fund">')
        h.append(f'<div class="fname"><h3>{escape(pr.label)}</h3><p class="tags">{" · ".join(tags)}</p>'
                 f'<p class="desc">{escape(pr.desc)}</p></div>')
        h.append(f'<div class="fnav"><span class="mono big">{num(nav)}</span><span class="unit">USD</span>'
                 f'<span class="chip {cls} mono">{num(chg, 2, True)}</span></div>')
        h.append(f'<div class="fspark">{sparkline(f["nav_hist"] or [START], base=START)}</div>')
        h.append(f'<dl class="facts"><div><dt>Kasse</dt><dd class="mono">{num(_f(st.get("cash")))}</dd></div>'
                 f'<div><dt>offen</dt><dd class="mono">{len(st.get("positions", {}))}</dd></div>'
                 f'<div><dt>Käufe</dt><dd class="mono">{buys}</dd></div>'
                 f'<div><dt>abgerechnet</dt><dd class="mono">{sells}{f" ({wins} gewonnen)" if sells else ""}</dd></div></dl>')
        pos = list(st.get("positions", {}).values())
        if pos:
            h.append(f'<details><summary>{len(pos)} offene Positionen</summary><div class="scroll"><table>'
                     '<thead><tr><th>Markt</th><th>Seite</th><th class="r">Anteile</th><th class="r">Kosten</th>'
                     '<th>Abrechnung</th></tr></thead><tbody>')
            for p in sorted(pos, key=lambda p: p.get("settle_time", "")):
                ts = _t(p.get("settle_time", ""))
                lbl = market_label({"direction": p.get("direction"), "lower": p.get("lower"), "upper": p.get("upper")})
                h.append(f'<tr><td>{escape(p.get("asset", ""))} {escape(lbl)}<br><span class="id">{escape(p.get("platform", ""))} '
                         f'{escape(str(p.get("market_id", "")))}</span></td><td>{"JA" if p.get("side") == "yes" else "NEIN"}</td>'
                         f'<td class="r mono">{escape(str(p.get("shares", "")))}</td>'
                         f'<td class="r mono">{num(_f(p.get("cost")))}</td><td class="mono">{local(ts) if ts else "–"}</td></tr>')
            h.append("</tbody></table></div></details>")
        h.append("</li>")
    h.append("</ol></section>")

    # Kanten
    top = last.get("top_edges", []) or []
    h.append('<section class="block"><div class="head"><h2>Größte Abweichungen</h2>'
             '<p>Die zehn Märkte mit der größten Kante gegen das Modellband nach Gebühr und Hedge-Kosten, '
             'letzter Lauf.</p></div>'
             '<p class="legend"><span><i class="sw hull"></i>Hülle</span><span><i class="sw model"></i>Modellband</span>'
             '<span><i class="sw mid"></i>Punktschätzung</span><span><i class="sw ask"></i>JA-Brief</span>'
             '<span><i class="sw bid"></i>1 − NEIN-Brief</span></p><ol class="edges">')
    for e in top:
        ts = _t(e.get("settle_time", ""))
        eh = _f(e.get("edge_modell_hedge"))
        cls = "good" if eh is not None and eh >= hurdle else ("warn" if eh is not None and eh > 0 else "neutral")
        name = f'{e.get("asset", "")} {market_label(e)}' + (f' am {ts.astimezone(BERLIN):%d.%m.%Y}' if ts else "")
        h.append('<li class="edge">')
        h.append(f'<div class="etitle"><b>{escape(name)}</b><span class="id">{escape(PLATFORM.get(e.get("platform"), ""))} · '
                 f'{escape(str(e.get("id", "")))} · {escape(QUALITY.get(e.get("quality"), ""))}'
                 f'{" · Abrechnung " + local(ts) + " Uhr" if ts else ""}</span>'
                 f'<span class="id">{escape(e.get("title") or "")}</span></div>')
        h.append(f'<div class="estrip">{band_strip(e)}</div>')
        h.append(f'<dl class="enums"><div><dt>Modell</dt><dd class="mono">{pct(_f(e.get("m_low")), 2)} – {pct(_f(e.get("m_high")), 2)}</dd></div>'
                 f'<div><dt>JA / NEIN</dt><dd class="mono">{num(_f(e.get("yes_ask")), 3)} / {num(_f(e.get("no_ask")), 3)}</dd></div>'
                 f'<div><dt>Hedge, Pkt</dt><dd class="mono">{pts(_f(e.get("hedge_model")), False)}</dd></div>'
                 f'<div><dt>Kante roh, Pkt</dt><dd class="mono">{pts(_f(e.get("edge_modell_roh")))}</dd></div>'
                 f'<div><dt>nach Hedge, Pkt</dt><dd class="mono"><span class="chip {cls}">{pts(eh)}</span></dd></div></dl>')
        h.append("</li>")
    if not top:
        h.append('<li class="empty">Noch kein Lauf mit Modellband.</li>')
    h.append("</ol></section>")

    # Abdeckung
    ap = last.get("matched_asset_platform", {}) or {}
    assets = ["BTC", "ETH", "SOL", "XRP"]
    rej = sorted((last.get("rejections") or {}).items(), key=lambda kv: -kv[1])[:5]
    notes = sorted((last.get("model_notes") or {}).items(), key=lambda kv: -kv[1])[:4]
    h.append('<section class="block"><div class="head"><h2>Abdeckung im letzten Lauf</h2></div><div class="cover">'
             '<div class="scroll"><table><thead><tr><th></th>' + "".join(f'<th class="r">{a}</th>' for a in assets) +
             '</tr></thead><tbody>')
    for plat, lbl in (("kalshi", "Kalshi"), ("polymarket", "Polymarket")):
        h.append(f"<tr><th>{lbl}</th>" + "".join(f'<td class="r mono">{ap.get(f"{a}/{plat}", 0)}</td>' for a in assets) + "</tr>")
    h.append("</tbody></table></div>")
    h.append('<dl class="reasons"><dt>Häufigste Ablehnungen</dt>' + "".join(
        f'<dd><span>{escape(REASONS.get(k, k))}</span><b class="mono">{v}</b></dd>' for k, v in rej) + "</dl>")
    h.append('<dl class="reasons"><dt>Ohne Modellband, weil</dt>' + "".join(
        f'<dd><span>{escape(REASONS.get(k, k))}</span><b class="mono">{v}</b></dd>' for k, v in notes) + "</dl>")
    h.append("</div></section>")

    h.append(METHOD)
    h.append("</main>")
    return "\n".join(h)


PLATFORM = {"kalshi": "Kalshi", "polymarket": "Polymarket"}
QUALITY = {"exakt": "exakt zum Verfall", "interpoliert": "zwischen zwei Verfällen", "vorlauf": "vor dem ersten Verfall"}

REASONS = {
    "zu_kurzfristig": "rechnet in unter 60 Minuten ab",
    "kein_verfall_daneben": "kein Deribit-Verfall danach",
    "verfaelle_zu_weit": "Verfälle zu weit auseinander",
    "keine_optionen": "keine Optionen für den Basiswert",
    "prob_nicht_berechenbar": "nicht berechenbar",
    "prob_fehler": "Rechenfehler",
    "ausserhalb_fit": "Schwelle außerhalb der Quotes",
    "zu_wenig_quotes": "zu wenige Quotes",
    "modell_ueber_nachbarn": "Nachbar-Verfälle genutzt",
    "fit_arbitrage": "Smile-Fit mit Arbitrage",
    "fit_ungenau": "Smile-Fit zu ungenau",
    "fit_ausserhalb_arbband": "Fit widerspricht den Quotes",
    "modell_unvollstaendig": "Modell unvollständig",
    "kalender_arbitrage": "Kalender-Arbitrage",
    "nachbarn_zu_weit": "Nachbar-Verfälle zu weit",
    "pm_nicht_offen": "Polymarket-Markt geschlossen",
}

METHOD = """<section class="block method"><div class="head"><h2>So wird gerechnet</h2></div><div class="cols">
<p><b>Hülle.</b> Am Deribit-Verfall begrenzen ausführbare Call- und Put-Spreads die Wahrscheinlichkeit
ohne jedes Modell. Zwischen zwei Verfällen ist das Band die Hülle der beiden Endpunkte. Liegt ein Kurs
nach Gebühr und Hedge-Kosten außerhalb, wäre das eine echte Abweichung.</p>
<p><b>Modellband.</b> Ein SVI-Smile je Verfall mit Arbitrage-Prüfung, die Varianz-Uhr für die Zeit
zwischen den Verfällen und die Abweichung der Referenzkurse. Die Spannweite über alle Varianten ergibt
das Band. Es ist schmal, gilt aber nur, wenn die Annahmen stimmen.</p>
<p><b>Hedge-Kosten.</b> Was es kosten würde, die Wette bei Deribit mit einem Spread abzusichern:
Gebühren, Geld-Brief-Spanne und das Schließen zum Abrechnungszeitpunkt. Das ist ein Kostenfilter,
kein abgesicherter Gewinn.</p>
<p><b>Beispielfonds.</b> Alle handeln nur auf Papier und unabgesichert, höchstens 5&thinsp;% des Fonds
und 10&thinsp;USD je Kauf, und halten bis zur offiziellen Abrechnung. Die naiven Fonds zeigen, wie
es ohne ehrliche Kosten aussähe.</p>
</div><p class="fine">Daten: öffentliche Schnittstellen von Deribit, Kalshi und Polymarket, ohne Konto.
Wahrscheinlichkeiten sind risikoneutral und können von den echten abweichen. Prognosemärkte sind in
Deutschland rechtlich problematisch; dieses Projekt ist Forschung.</p></section>"""

HEAD = """<title>Wahrscheinlichkeits-Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@112,600;112,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Messpult: eine Spalte, oben die Lage, dann Fonds als Zeilen, dann Bandstreifen je Markt */
:root {
  --paper: #f1f4f7; --panel: #ffffff; --ink: #15212c; --muted: #586776; --rule: #d3dbe3;
  --hull: #cbd6e0; --model: #2d5b88; --mid: #15212c; --ask: #b5651d; --bid: #6f7f8f;
  --good: #23704a; --good-bg: #dcefe4; --warn: #8a5a00; --warn-bg: #f6e8c8; --bad: #a3271f; --bad-bg: #f6dcd9;
  --neutral-bg: #e4e9ee;
  --display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0e151c; --panel: #141d26; --ink: #e5ebf0; --muted: #94a2af; --rule: #263240;
  --hull: #2a3949; --model: #7eb0e2; --mid: #e5ebf0; --ask: #e59b52; --bid: #8d9cab;
  --good: #67c28f; --good-bg: #173527; --warn: #e3b552; --warn-bg: #3a2e12; --bad: #f07b70; --bad-bg: #3d1d1a;
  --neutral-bg: #1e2934; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0e151c; --panel: #141d26; --ink: #e5ebf0; --muted: #94a2af; --rule: #263240;
  --hull: #2a3949; --model: #7eb0e2; --mid: #e5ebf0; --ask: #e59b52; --bid: #8d9cab;
  --good: #67c28f; --good-bg: #173527; --warn: #e3b552; --warn-bg: #3a2e12; --bad: #f07b70; --bad-bg: #3d1d1a;
  --neutral-bg: #1e2934; color-scheme: dark; }
body { background: var(--paper); color: var(--ink); font: 15px/1.5 var(--body); }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 56px; display: grid; gap: 34px; }
h1, h2, h3 { font-family: var(--display); font-stretch: 112%; text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); font-weight: 700; line-height: 1.1; }
h2 { font-size: 1.25rem; font-weight: 700; }
h3 { font-size: 1.02rem; font-weight: 600; }
p { margin: 0; }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.eyebrow, .k, dt, .tags, th { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.top { display: grid; gap: 8px; }
.lede { max-width: 62ch; font-size: 1.02rem; }
.meta, .s, .desc, .id, .head p, .fine, .legend, figcaption { color: var(--muted); font-size: 0.84rem; }
.status { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.stat { background: var(--panel); padding: 14px 16px; display: grid; gap: 4px; align-content: start; min-width: 0; }
.stat .v { font-size: 1.55rem; line-height: 1.15; font-weight: 500; }
.stat .v small { font-size: 0.8rem; color: var(--muted); }
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; background: var(--neutral-bg); color: var(--ink); }
.stat .v.chip { font-size: 1.05rem; justify-self: start; padding: 2px 12px; font-family: var(--body); }
.chip.good { background: var(--good-bg); color: var(--good); }
.chip.warn { background: var(--warn-bg); color: var(--warn); }
.chip.bad { background: var(--bad-bg); color: var(--bad); }
.block { display: grid; gap: 14px; }
.head { display: grid; gap: 4px; border-bottom: 1px solid var(--rule); padding-bottom: 8px; }
.trend { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figcaption b { color: var(--ink); font-weight: 500; margin-left: 6px; }
.spark { width: 100%; height: 44px; display: block; }
.spark .line { fill: none; stroke: var(--model); stroke-width: 1.6; vector-effect: non-scaling-stroke; }
.spark .area { fill: var(--model); opacity: 0.12; }
.spark .end { fill: var(--model); }
.spark .ref { stroke: var(--muted); stroke-dasharray: 3 3; stroke-width: 1; vector-effect: non-scaling-stroke; }
.spark .hurdle { stroke: var(--good); stroke-dasharray: 5 3; stroke-width: 1; vector-effect: non-scaling-stroke; }
ol.funds, ol.edges { list-style: none; margin: 0; padding: 0; display: grid; gap: 0; border-top: 1px solid var(--rule); }
.fund { display: grid; grid-template-columns: minmax(0, 2.2fr) auto minmax(140px, 1fr) minmax(0, 1.6fr); gap: 8px 24px;
  align-items: center; padding: 16px 0; border-bottom: 1px solid var(--rule); }
.fname { display: grid; gap: 3px; min-width: 0; }
.desc { max-width: 60ch; }
.fnav { display: grid; justify-items: end; gap: 2px; }
.fnav .big { font-size: 1.6rem; line-height: 1; }
.fnav .unit { font-size: 0.72rem; color: var(--muted); }
.facts, .enums { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 14px; margin: 0; }
.facts div, .enums div { display: grid; gap: 1px; }
dd { margin: 0; }
details { grid-column: 1 / -1; }
summary { cursor: pointer; color: var(--model); font-weight: 500; font-size: 0.9rem; }
summary:focus-visible { outline: 2px solid var(--model); outline-offset: 2px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; margin-top: 8px; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
.r { text-align: right; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.sw { display: inline-block; width: 14px; height: 10px; margin-right: 6px; vertical-align: -1px; }
.sw.hull { background: var(--hull); } .sw.model { background: var(--model); }
.sw.mid { width: 2px; height: 14px; background: var(--mid); vertical-align: -3px; }
.sw.ask { width: 0; height: 0; border: 6px solid transparent; border-top: 8px solid var(--ask); border-bottom: 0; }
.sw.bid { width: 0; height: 0; border: 6px solid transparent; border-bottom: 8px solid var(--bid); border-top: 0; }
.edge { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(200px, 1.3fr) minmax(0, 1.4fr); gap: 8px 22px;
  align-items: center; padding: 14px 0; border-bottom: 1px solid var(--rule); }
.etitle { display: grid; gap: 2px; min-width: 0; }
.enums { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.strip { width: 100%; height: auto; display: block; overflow: visible; }
.strip .axis { stroke: var(--rule); stroke-width: 1; }
.strip .hull { fill: var(--hull); }
.strip .model { fill: var(--model); }
.strip .mid { stroke: var(--mid); stroke-width: 1.5; }
.strip .ask { fill: var(--ask); }
.strip .bid { fill: var(--bid); }
.strip .tick { fill: var(--muted); font: 9px var(--mono); }
.strip .tick.end { text-anchor: end; }
.empty { padding: 14px 0; color: var(--muted); }
.cover { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: start; }
.reasons { margin: 0; display: grid; gap: 4px; min-width: 0; }
.reasons dd { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--rule); padding: 4px 0; font-size: 0.86rem; }
.method .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px 28px; }
.method .cols p { max-width: 62ch; font-size: 0.9rem; }
.fine { max-width: 80ch; }
@media (max-width: 760px) {
  .fund { grid-template-columns: minmax(0, 1fr) auto; }
  .fspark, .facts { grid-column: 1 / -1; }
  .facts { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .edge { grid-template-columns: minmax(0, 1fr); }
  .cover { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
</style>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "dashboard" / "output.html")
    args = ap.parse_args(argv)
    html = render(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Dashboard geschrieben: {args.out} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
