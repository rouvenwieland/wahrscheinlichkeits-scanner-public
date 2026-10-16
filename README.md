# Wahrscheinlichkeits-Scanner (öffentliche Fassung)

Forschungsprototyp: vergleicht Kurse von Prognosemärkten (Kalshi, Polymarket) mit
risikoneutralen Wahrscheinlichkeiten aus Deribit-Optionen für BTC, ETH, SOL und XRP und führt
fünf Beispielfonds **nur auf Papier**. Keine echten Trades, keine Konten, keine API-Schlüssel,
keine Anlageberatung.

Diese öffentliche Fassung enthält denselben Code wie das private Arbeits-Repo, aber **keine
Deribit-Rohdaten**: Deribits Marktdaten dürfen nicht weitergegeben werden. Darum
- laufen die Tests mit künstlichen Optionsdaten (`tests/synthetic.py`),
- speichert der stündliche Scan (`.github/workflows/scan.yml`) keine Rohdaten-Schnappschüsse,
- enthält `data/` nur abgeleitete Zahlen: Wahrscheinlichkeiten, Bänder, Kanten, Fonds.

## Was gerechnet wird

- **Hülle:** Am Deribit-Verfall begrenzen ausführbare Call- und Put-Spreads die
  Wahrscheinlichkeit ohne Modell; zwischen zwei Verfällen die Hülle der Endpunkte.
- **Modellband:** SVI-Smile je Verfall mit Arbitrage-Prüfung (Durrleman, Lee, Kalender), eine
  Varianz-Uhr für die Zeit zwischen den Verfällen und die Abweichung der Referenzkurse.
- **Hedge-Kosten:** Deribit-Gebühren (0,03 % des Basiswerts, Deckel 12,5 % des Optionspreises,
  Kombi-Rabatt), Geld-Brief-Spanne und Schließen zum Abrechnungszeitpunkt.

## Beispielfonds (je 100 USD)

| Fonds | Band | Hedge-Kosten | Hürde |
|---|---|---|---|
| `streng` | Arbitrageband, nur exakte Zeit | ja | 2 Punkte |
| `interpoliert` | Hülle | ja | 2 Punkte |
| `modell` | Modellband | ja | 2 Punkte |
| `modell_roh` | Modellband | nein | 2 Punkte |
| `punkt` | nur Punktschätzung (naiv) | nein | 1 Punkt |

## Aufruf

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m scanner.run daily --no-snapshots
```

Daten stammen aus öffentlichen, lesenden Schnittstellen von Deribit, Kalshi und Polymarket.
Prognosemärkte sind in Deutschland rechtlich problematisch; dieses Projekt ist Forschung.
