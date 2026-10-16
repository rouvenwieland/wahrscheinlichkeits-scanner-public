"""Basiswerte und ihre Quellen (Stand 15.10.2026, live geprueft).

Deribit: BTC und ETH als inverse Optionen (Preis in Coin) plus lineare USDC-Optionen
(``BTC_USDC`` usw., Preis in USDC); SOL und XRP nur linear. Kalshi: Serien mit Schwellen
(``greater``/``less``) und Spannen (``between``), Referenz CF-Benchmarks-Index. Polymarket:
"<Name> above ___ on <Datum>?" und "<Name> price on <Datum>?", Referenz Binance <X>/USDT.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    code: str  # "BTC"
    pm_name: str  # Name in Polymarket-Fragen ("Bitcoin")
    pm_pair: str  # Binance-Paar in der Polymarket-Regel ("BTC/USDT")
    pm_tag: str  # Gamma-Tag ("bitcoin")
    deribit: tuple  # ((currency-Parameter, Instrument-Praefix), ...)
    perpetual: str  # fuer die Varianz-Uhr
    index_usd: str  # Deribit-Index in USD (Kassakurs)
    index_usdt: str  # Deribit-Index gegen USDT (fuer die USDT-Abweichung)
    kalshi_series: tuple
    kalshi_refs: tuple  # zulaessige Referenztexte in ``rules_primary``


ASSETS = {
    "BTC": Asset("BTC", "Bitcoin", "BTC/USDT", "bitcoin", (("BTC", "BTC"), ("USDC", "BTC_USDC")),
                 "BTC-PERPETUAL", "btc_usd", "btc_usdt", ("KXBTCD", "KXBTC", "KXBTCY"),
                 ("CF Benchmarks' Bitcoin Real-Time Index (BRTI)", "CF Benchmarks' BRTI")),
    "ETH": Asset("ETH", "Ethereum", "ETH/USDT", "ethereum", (("ETH", "ETH"), ("USDC", "ETH_USDC")),
                 "ETH-PERPETUAL", "eth_usd", "eth_usdt", ("KXETHD", "KXETH", "KXETHY"),
                 ("CF Benchmarks' Ethereum Real-Time Index (ERTI)", "CF Benchmarks' ETHUSD_RTI")),
    "SOL": Asset("SOL", "Solana", "SOL/USDT", "solana", (("USDC", "SOL_USDC"),),
                 "SOL_USDC-PERPETUAL", "sol_usd", "sol_usdt", ("KXSOLD", "KXSOLE", "KXSOLD26"),
                 ("CF Benchmarks' SOLUSD_RTI",)),
    "XRP": Asset("XRP", "XRP", "XRP/USDT", "xrp", (("USDC", "XRP_USDC"),),
                 "XRP_USDC-PERPETUAL", "xrp_usd", "xrp_usdt", ("KXXRPD", "KXXRP"),
                 ("CF Benchmarks' Ripple-Dollar Real Time Index (XRPUSD_RTI)",)),
}

# Referenzverschiebung (Anteil der Schwelle): CF-Index gegen Deribit-Index (beide
# USD-Mischindizes) gering; Binance gegen USDT kommt die live gemessene USDT-Abweichung dazu.
DELTA_KALSHI = 0.001
DELTA_POLY_BASE = 0.001
DELTA_LINEAR_EXTRA = 0.0005  # SOL/XRP: Deribit-Optionen laufen auf den USDC-Index


def asset_for_kalshi_series(series: str):
    for a in ASSETS.values():
        if series in a.kalshi_series:
            return a
    return None
