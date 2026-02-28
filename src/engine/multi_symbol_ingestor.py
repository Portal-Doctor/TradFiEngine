"""Multi-symbol data ingestor: load and align OHLCV across symbols."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .data_ingestor import DataIngestor


def slice_multi_symbol(
    dfs: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Slice aligned DataFrames by date range. Preserves index alignment."""
    # Use first df to infer timezone; align start/end for comparison
    first_df = next(iter(dfs.values()))
    idx = first_df.index
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if idx.tz is not None:
        start_ts = start_ts.tz_localize(idx.tz) if start_ts.tz is None else start_ts
        end_ts = end_ts.tz_localize(idx.tz) if end_ts.tz is None else end_ts
    sliced = {}
    for s, df in dfs.items():
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        sliced[s] = df.loc[mask].copy()
    return sliced


def load_multi_symbol(
    data_ingestor: "DataIngestor",
    symbols: list[str],
    config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load and align OHLCV for multiple symbols. Returns dict of DataFrames
    with shared index (intersection of timestamps). Optionally adds indicators
    when config is provided.
    """
    multi = MultiSymbolIngestor(data_ingestor, symbols)
    frames, aligned_index = multi.load_aligned()

    # Reindex each to aligned timestamps (aligned_index is already the intersection)
    aligned_frames = {s: frames[s].reindex(aligned_index) for s in symbols}

    if config:
        from src.indicators import add_all_indicators

        ind_cfg = config.get("indicators", {})
        feat_cfg = config.get("features", {})
        for s in aligned_frames:
            df = add_all_indicators(
                aligned_frames[s],
                macd_fast=ind_cfg.get("macd", {}).get("fast", 12),
                macd_slow=ind_cfg.get("macd", {}).get("slow", 26),
                macd_signal=ind_cfg.get("macd", {}).get("signal", 9),
                rsi_length=ind_cfg.get("rsi", {}).get("length", 14),
                bb_length=ind_cfg.get("bollinger", {}).get("length", 20),
                bb_std=ind_cfg.get("bollinger", {}).get("std", 2.0),
                atr_length=ind_cfg.get("atr", {}).get("length", 14),
                use_log_returns=feat_cfg.get("use_log_returns", False),
            )
            aligned_frames[s] = df.dropna()

        # Re-align to intersection after indicator warmup
        common_index = aligned_frames[symbols[0]].index
        for s in symbols[1:]:
            common_index = common_index.intersection(aligned_frames[s].index)
        aligned_frames = {s: aligned_frames[s].loc[common_index] for s in symbols}

    return aligned_frames


class MultiSymbolIngestor:
    """Loads OHLCV for multiple symbols and aligns them on timestamp via inner join."""

    def __init__(self, data_ingestor: "DataIngestor", symbols: list[str]):
        self.data_ingestor = data_ingestor
        self.symbols = list(symbols)

    def load_aligned(self) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """
        Load data for each symbol, then inner-join on timestamp index.
        Returns (frames, aligned_index) where frames has per-symbol DataFrames
        and aligned_index is the intersection of all timestamps.
        """
        frames: dict[str, pd.DataFrame] = {}
        for s in self.symbols:
            df = self.data_ingestor.load(s)
            df = df.copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df = df.sort_index()
            frames[s] = df

        aligned: pd.DataFrame | None = None
        for s, df in frames.items():
            if aligned is None:
                aligned = df.copy()
            else:
                aligned = aligned.join(df, how="inner", rsuffix=f"_{s}")

        if aligned is None:
            raise ValueError("No symbols to align")

        return frames, aligned.index
