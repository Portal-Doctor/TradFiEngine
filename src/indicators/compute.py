"""Compute technical indicators for OHLCV DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_all_indicators(
    df: pd.DataFrame,
    *,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_length: int = 14,
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """
    Add MACD, RSI, Bollinger Bands, and ATR to OHLCV DataFrame.
    Uses ta library (Python 3.6+ compatible) — no TA-Lib required.
    """
    result = df.copy()
    close = result["close"] if "close" in result.columns else result["Close"]
    high = result["high"] if "high" in result.columns else result["High"]
    low = result["low"] if "low" in result.columns else result["Low"]

    try:
        from ta.trend import MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
    except ImportError:
        raise ImportError("Install ta: pip install ta")

    # MACD
    macd_ind = MACD(
        close=close,
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    result["MACD"] = macd_ind.macd()
    result["MACD_signal"] = macd_ind.macd_signal()
    result["MACD_hist"] = macd_ind.macd_diff()

    # RSI
    rsi_ind = RSIIndicator(close=close, window=rsi_length)
    result["RSI"] = rsi_ind.rsi()

    # Bollinger Bands
    bb_ind = BollingerBands(close=close, window=bb_length, window_dev=bb_std)
    result["BB_upper"] = bb_ind.bollinger_hband()
    result["BB_mid"] = bb_ind.bollinger_mavg()
    result["BB_lower"] = bb_ind.bollinger_lband()

    # ATR (volatility)
    atr_ind = AverageTrueRange(high=high, low=low, close=close, window=atr_length)
    result["ATR"] = atr_ind.average_true_range()

    # Price change: log-returns for stationarity, else pct_change
    if use_log_returns:
        result["returns"] = np.log(close / close.shift(1))
    else:
        result["returns"] = close.pct_change()

    return result


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add log-returns column. Use for feature scaling with non-stationary data.
    log_returns are more normally distributed and bounded.
    """
    result = df.copy()
    close = result[price_col] if price_col in result.columns else result["close"]
    result["log_returns"] = np.log(close / close.shift(1))
    return result
