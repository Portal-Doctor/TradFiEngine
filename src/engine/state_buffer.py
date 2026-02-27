"""
StateBuffer: maintains rolling window of feature rows for live prediction.
Ensures obs matches training exactly — store last N bars, append newest, then predict().
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StateBuffer:
    """
    Feature store for live/paper: keeps last window_size bars of features.
    Append each new candle; get_obs() returns flattened window matching training env.
    """

    def __init__(
        self,
        feature_cols: list[str],
        window_size: int = 60,
    ):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.n_features = len(feature_cols)
        self._buffer: list[np.ndarray] = []

    def append(self, row: pd.Series | np.ndarray) -> None:
        """
        Append one bar's features. Row must have keys matching feature_cols.
        """
        if isinstance(row, pd.Series):
            vals = row[self.feature_cols].values.astype(np.float32)
        else:
            vals = np.asarray(row, dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._buffer.append(vals)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def get_obs(self) -> np.ndarray:
        """
        Return observation matching training: shape (window_size * n_features,).
        Pads with zeros if buffer not yet full.
        """
        if len(self._buffer) < self.window_size:
            pad = np.zeros(
                (self.window_size - len(self._buffer), self.n_features),
                dtype=np.float32,
            )
            window = np.vstack([pad] + self._buffer)
        else:
            window = np.stack(self._buffer)
        return window.flatten().astype(np.float32)

    @property
    def is_ready(self) -> bool:
        """True when buffer has enough history to produce valid obs."""
        return len(self._buffer) >= self.window_size

    def clear(self) -> None:
        self._buffer.clear()
