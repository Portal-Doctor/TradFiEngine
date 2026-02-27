"""Load OHLCV from CSV or fetch via CCXT."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV from CSV. Expects columns: timestamp, open, high, low, close, volume
    (or Open, High, Low, Close, Volume). CCXT format uses lowercase.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    exchange_id: str = "coinbase",
) -> pd.DataFrame:
    """
    Fetch OHLCV from exchange via CCXT (no auth needed for public data).
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
