"""Load OHLCV from CSV or fetch via CCXT."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Coinbase: 10 req/s sustained. Use 200ms min gap = 5 req/s to stay under limit.
COINBASE_RATE_LIMIT_MS = 200


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


def create_exchange(
    exchange_id: str = "coinbase",
    *,
    rate_limit_ms: int | None = None,
) -> object:
    """
    Create a CCXT exchange instance with rate limiting for Coinbase compliance.
    Reuse the same instance for multiple fetches so rate limits apply across all calls.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    opts = {"enableRateLimit": True}
    if exchange_id == "coinbase" and rate_limit_ms is not None:
        opts["rateLimit"] = rate_limit_ms
    elif exchange_id == "coinbase":
        opts["rateLimit"] = COINBASE_RATE_LIMIT_MS

    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class(opts)


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    exchange_id: str = "coinbase",
    exchange: object | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV from exchange via CCXT (no auth needed for public data).
    Pass a pre-created exchange to reuse it (recommended for multiple fetches to respect rate limits).
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    if exchange is None:
        exchange = create_exchange(exchange_id)

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
