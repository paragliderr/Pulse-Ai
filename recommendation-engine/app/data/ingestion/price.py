"""Price/OHLCV ingestion via yfinance. See Milestone 2 report, Section 3."""

import yfinance as yf
import pandas as pd
from app.config import TICKER_UNIVERSE


def fetch_price_history(tickers: list[str] = None, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns a long-format DataFrame: ticker, date, open, high, low, close, volume.
    """
    tickers = tickers or TICKER_UNIVERSE
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True)

    frames = []
    for ticker in tickers:
        try:
            df = data[ticker].reset_index()
            df["ticker"] = ticker
            df = df.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
        except KeyError:
            continue  # ticker had no data for the period

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    df = fetch_price_history()
    print(df.head())
    print(f"Fetched {len(df)} rows across {df['ticker'].nunique()} tickers")
