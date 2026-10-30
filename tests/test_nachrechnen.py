"""Nachrechnung (tools/nachrechnen.py) auf kuenstlichen Beobachtungen, offline."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import nachrechnen  # noqa: E402

from scanner.run import OBS_FIELDS  # noqa: E402


def row(ts, mid, settle, p_mid, yes_ask, no_ask, **kw):
    r = {k: "" for k in OBS_FIELDS}
    r.update(timestamp=ts, asset="BTC", platform="kalshi", id=mid, direction="above", lower="80000",
             settle_time=settle, quality="interpoliert", p_mid=p_mid, p_low="0", p_high="1",
             m_low=p_mid, m_high=p_mid, quelle="buch", yes_ask=yes_ask, no_ask=no_ask, yes_depth="1000",
             no_depth="1000", hedge_exec="0.5", hedge_model="0.5", hedge_n_min="10", decision_streng="SKIP:qualitaet")
    r.update(kw)
    return r


def test_replay_buys_respects_cluster_and_settles_to_official_result(tmp_path):
    obs = tmp_path / "observations" / "2026-10"
    obs.mkdir(parents=True)
    t1, t2, settle = "2026-10-25T10:00:00Z", "2026-10-25T22:00:00Z", "2026-10-25T21:00:00Z"
    rows = [row(t1, f"T{i}", settle, "0.70", "0.50", "0.52") for i in range(6)]  # 6 Schwellen, ein Tag
    rows.append(row(t1, "MOVED", settle, "0.90", "0.50", "0.52", decision_streng="SKIP:kurs_bewegt"))
    rows += [row(t2, f"T{i}", settle, "0.70", "0.98", "0.03", quelle="liste") for i in range(6)]
    with (obs / "2026-10-25.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OBS_FIELDS)
        w.writeheader()
        w.writerows(rows)
    res = {f"kalshi:T{i}": {"result": "yes", "settled_at": "2026-10-25T21:02:00Z"} for i in range(6)}
    res["kalshi:MOVED"] = {"result": "no", "settled_at": "2026-10-25T21:02:00Z"}
    (tmp_path / "korrigiert").mkdir()
    (tmp_path / "korrigiert" / "ergebnisse.json").write_text(json.dumps(res), encoding="utf-8")
    assert nachrechnen.main(["--data-dir", str(tmp_path), "--offline", "--jetzt", "2026-10-26T00:00:00Z"]) == 0
    s = json.loads((tmp_path / "korrigiert" / "zusammenfassung.json").read_text(encoding="utf-8"))
    p = s["fonds"]["punkt"]
    # Klumpengrenze 10 % NAV: nur so viele Kaeufe, wie 10 USD Einstand zulassen; alle gewinnen
    assert 2 <= p["kaeufe"] <= 3 and p["abgerechnet"] == p["kaeufe"] == p["gewonnen"]
    assert p["ende"] > 100 and p["offen"] == 0
    assert s["entscheidungen"].get("punkt:SKIP:kurs_bewegt") == 1
    # Fonds mit Hedge: Kante 0,70 - 0,5175 - 0,02 < 0,5 Hedge-Kosten -> nie
    assert s["fonds"]["modell"]["kaeufe"] == 0
    with (tmp_path / "korrigiert" / "verlauf.csv").open(newline="", encoding="utf-8") as fh:
        hist = list(csv.DictReader(fh))
    assert [h["reihe"] for h in hist] == ["nachgerechnet", "nachgerechnet", "nachgerechnet_schluss"]
    # zweiter Lauf: Bewertung zum Gegenkurs 1 - 0,03 vor der Abrechnung (Abrechnung erst ab 21:02)
    assert float(hist[1]["nav_punkt"]) > 100
