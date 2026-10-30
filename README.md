# Wahrscheinlichkeits-Scanner (öffentliche Fassung)

Forschungsprototyp: vergleicht Kurse von Prognosemärkten (Kalshi, Polymarket) mit
risikoneutralen Wahrscheinlichkeiten aus Deribit-Optionen für BTC, ETH, SOL und XRP und führt
fünf Beispielfonds in drei Größen **nur auf Papier**. Keine echten Trades, keine Konten, keine API-Schlüssel,
keine Anlageberatung.

Diese öffentliche Fassung enthält denselben Code wie das private Arbeits-Repo, aber **keine
Deribit-Rohdaten**: Deribits Marktdaten dürfen nicht weitergegeben werden. Darum
- laufen die Tests mit künstlichen Optionsdaten (`tests/synthetic.py`),
- speichert der Scan (`.github/workflows/scan.yml`) keine Rohdaten-Schnappschüsse,
- enthält `data/` nur abgeleitete Zahlen: Wahrscheinlichkeiten, Bänder, Kanten, Fonds.

## Was gerechnet wird

- **Hülle:** Am Deribit-Verfall begrenzen ausführbare Call- und Put-Spreads die
  Wahrscheinlichkeit ohne Modell; zwischen zwei Verfällen die Hülle der Endpunkte.
- **Modellband:** SVI-Smile je Verfall mit Arbitrage-Prüfung (Durrleman, Lee, Kalender), eine
  Varianz-Uhr für die Zeit zwischen den Verfällen und die Abweichung der Referenzkurse.
- **Hedge-Kosten:** Deribit-Gebühren (0,03 % des Basiswerts, Deckel 12,5 % des Optionspreises,
  Kombi-Rabatt), Geld-Brief-Spanne und Schließen zum Abrechnungszeitpunkt.

## Beispielfonds (100, 1.000 und 10.000 USD Start, nur Papier)

**Kernstrategie** (Einstieg nur, wenn der Preis nach Gebühr *und* Deribit-Hedge-Kosten außerhalb
des Bandes liegt und die Position die Hedge-Mindestgröße erreicht):

| Fonds | Band | Hürde |
|---|---|---|
| `streng` | Arbitrageband, nur exakte Zeit | 2 Punkte |
| `interpoliert` | Hülle | 2 Punkte |
| `modell` | Modellband | 2 Punkte |

**Vergleich: ungesicherte Wette** (bewusst ohne Absicherung, zeigt, was Band und Hedge gegenüber
naivem Wetten bringen; Einzelverluste gehören hier zum Konzept):

| Fonds | Band | Hürde |
|---|---|---|
| `modell_roh` | Modellband, ohne Hedge-Kosten | 2 Punkte |
| `punkt` | nur Punktschätzung | 1 Punkt |

Für alle gelten dieselben Regeln: Kauf zum Brief plus Gebühr, Bewertung zum Geldkurs, Abrechnung
nur zum offiziellen Ergebnis, je Basiswert und Abrechnungstag höchstens 10 % und je Basiswert
höchstens 25 % des Fondswerts (viele Schwellen desselben Verfalls sind eine einzige Wette).
Das Hedge-Bein wird nicht simuliert: "nach Hedge-Kosten" ist ein Kostenfilter, kein
abgesicherter Gewinn.

**Stand der Reihe:** Dieses Repo läuft seit dem 30.10.2026 mit dem korrigierten Fondscode
(Positionsgrenzen, Hedge-Mindestgröße, Bewertung, Abrechnung). Die Fonds starten hier neu bei
100 / 1.000 / 10.000 USD; der Stand der alten Logik liegt unverändert in
`data/archiv/2026-10-30_alte_logik/` und ist nicht mit der neuen Reihe vergleichbar.

## Aufruf

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m scanner.run daily --no-snapshots
```

Daten stammen aus öffentlichen, lesenden Schnittstellen von Deribit, Kalshi und Polymarket.
Prognosemärkte sind in Deutschland rechtlich problematisch; dieses Projekt ist Forschung.
