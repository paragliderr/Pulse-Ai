"""
Price data preprocessing: missing-value handling, normalization, and
technical indicator derivation. See Milestone 2 report, Section 4.
"""

import pandas as pd


def compute_technical_indicators(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    df: long-format DataFrame with columns [ticker, date, open, high, low, close, volume],
    sorted by date within each ticker.

    Adds: daily_return, rolling_volatility, moving_average.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["rolling_volatility"] = (
        df.groupby("ticker")["daily_return"].transform(lambda x: x.rolling(window).std())
    )
    df["moving_average"] = (
        df.groupby("ticker")["close"].transform(lambda x: x.rolling(window).mean())
    )

    return df


def normalize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-score normalization per ticker, so features are comparable across
    stocks with very different price scales."""
    df = df.copy()
    for col in columns:
        df[f"{col}_z"] = df.groupby("ticker")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    return df


def fill_missing_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill price gaps from holidays/non-trading days, per ticker."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    filled = []
    for ticker, group in df.groupby("ticker"):
        group = group.set_index("date").asfreq("D").ffill()
        group["ticker"] = ticker
        filled.append(group.reset_index())
    return pd.concat(filled, ignore_index=True)
