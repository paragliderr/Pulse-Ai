"""Fundamentals/sector ingestion via Alpha Vantage. See Milestone 2 report, Section 3."""

import requests
import time
from app.config import ALPHA_VANTAGE_KEY

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def fetch_fundamentals(ticker: str) -> dict:
    """
    Returns {ticker, sector, market_cap, pe_ratio}.
    Free tier is rate-limited (~5 calls/min) — see the sleep in
    fetch_fundamentals_bulk when pulling the full ticker universe.
    """
    if not ALPHA_VANTAGE_KEY:
        raise RuntimeError("ALPHA_VANTAGE_KEY is not set — check your .env")

    params = {"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHA_VANTAGE_KEY}
    resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "ticker": ticker,
        "sector": data.get("Sector", "Unknown"),
        "market_cap": data.get("MarketCapitalization"),
        "pe_ratio": data.get("PERatio"),
    }


def fetch_fundamentals_bulk(tickers: list[str], delay_seconds: float = 13.0) -> list[dict]:
    """Sequential fetch with a delay to respect the free-tier rate limit."""
    results = []
    for ticker in tickers:
        results.append(fetch_fundamentals(ticker))
        time.sleep(delay_seconds)
    return results


if __name__ == "__main__":
    print(fetch_fundamentals("AAPL"))
