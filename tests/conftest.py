"""Alle Tests laufen offline: jeder Socket-/requests-Aufruf schlaegt fehl."""
import socket
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FIXTURES = Path(__file__).parent / "fixtures"


def _blocked(*a, **k):
    raise RuntimeError("Netzwerkzugriff in Tests verboten")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(requests.Session, "request", _blocked)
    monkeypatch.setattr("scanner.http.sleep", lambda s: None)


@pytest.fixture
def fixtures_dir():
    return FIXTURES


SYNTHETIC = {  # Ersatz, wenn echte Deribit-Fixtures fehlen (oeffentliches Repo)
    "deribit_book_summary_btc.json": lambda: __import__("synthetic").payload("BTC"),
    "deribit_book_summary_eth.json": lambda: __import__("synthetic").payload("ETH"),
    "deribit_book_summary_sol_usdc.json": lambda: {"result": __import__("synthetic").book_summary("SOL_USDC")},
}


@pytest.fixture
def deribit_fixture(fixtures_dir):
    """Echte Deribit-Antwort aus tests/fixtures, sonst die kuenstliche aus tests/synthetic.py."""
    import json

    def load(name):
        path = fixtures_dir / name
        if path.exists():
            return json.load(open(path, encoding="utf-8"))
        return SYNTHETIC[name]()
    return load
