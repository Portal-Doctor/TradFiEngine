"""Data Ingestor: dedicated component for fetching price feeds."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

# Timeframe to seconds (for sleep_until_next_candle)
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


class DataIngestor:
    """
    Producer: fetches OHLCV data from CSV or exchange.
    For live: use get_last_closed_candle() + sleep_until_next_candle() to avoid
    incomplete-candle leakage. Never append a candle that hasn't closed.
    """

    def __init__(
        self,
        source: str = "fetch",
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
        limit: int = 500,
        on_bar: Callable[[pd.Series], None] | None = None,
    ):
        self.source = source
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.on_bar = on_bar

    def fetch_historical(self) -> pd.DataFrame:
        """Fetch historical OHLCV (blocking)."""
        if self.source == "fetch":
            from src.data import fetch_ohlcv_ccxt
            sym = self.symbol.replace("-", "/")
            return fetch_ohlcv_ccxt(sym, self.timeframe, limit=self.limit)
        from src.data import load_ohlcv
        return load_ohlcv(self.source)

    def get_latest_bar(self) -> pd.Series | None:
        """Get most recent bar. May be incomplete if current candle still forming."""
        df = self.fetch_historical()
        if df is None or len(df) == 0:
            return None
        return df.iloc[-1]

    def get_last_closed_candle(self, df_with_indicators: pd.DataFrame | None = None) -> pd.Series | None:
        """
        Get the most recently *closed* candle (never the one still forming).
        Pass df_with_indicators if you already have computed indicators; else fetches raw.
        Returns last closed bar (iloc[-2] when last row is current candle, else iloc[-1]).
        """
        if df_with_indicators is not None:
            df = df_with_indicators
        else:
            df = self.fetch_historical()
        if df is None or len(df) < 2:
            return df.iloc[-1] if df is not None and len(df) == 1 else None
        # Prefer second-to-last: avoids incomplete current candle.
        return df.iloc[-2].copy()

    @staticmethod
    def sleep_until_next_candle(timeframe: str) -> None:
        """
        Sleep until the next candle close. Ensures temporal alignment:
        only process candles that have fully closed.
        """
        secs = TIMEFRAME_SECONDS.get(timeframe, 3600)
        now = datetime.now(timezone.utc).timestamp()
        # Candle boundaries are aligned to epoch
        elapsed = now % secs
        sleep_for = secs - elapsed
        if sleep_for > 1:  # avoid tiny sleeps
            time.sleep(sleep_for)
