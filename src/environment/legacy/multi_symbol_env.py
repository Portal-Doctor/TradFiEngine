"""Legacy: Multi-symbol env that randomly selects one symbol per episode.

This design (domain randomization) is incompatible with the new multi-asset engine,
which observes all symbols simultaneously with an 8-d allocation action.
Kept for potential domain-randomization experiments.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from ..trading_env import CryptoTradingEnv
from config import load_config


class MultiSymbolTradingEnv(gym.Env):
    """
    LEGACY: Wraps CryptoTradingEnv for multiple symbols. Each episode resets with a
    randomly chosen symbol. Single-symbol observation/action per episode.

    Use MultiAssetTradingEnv for the new multi-asset pipeline.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbols: list[str],
        data_by_symbol: dict[str, pd.DataFrame],
        *,
        config: dict | None = None,
        timeframe: str = "1h",
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.symbols = list(symbols)
        self.data_by_symbol = dict(data_by_symbol)
        self.timeframe = timeframe

        cfg = config or load_config()
        env_cfg = cfg.get("env", {})
        starting_cash = env_cfg.get("starting_cash") or env_cfg.get("initial_balance", 10_000.0)

        # Build one env per symbol; all share same config
        self._envs: dict[str, CryptoTradingEnv] = {}
        for sym, df in self.data_by_symbol.items():
            if df is not None and len(df) > 0:
                env = CryptoTradingEnv(
                    df,
                    config=cfg,
                    initial_balance=float(starting_cash),
                    render_mode=None,
                    seed=None,
                )
                self._envs[sym] = env

        if not self._envs:
            raise ValueError("No valid symbol data in data_by_symbol")

        # Use first env for spaces (all have same obs/action shape)
        first_env = next(iter(self._envs.values()))
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

        self._current_symbol: str | None = None
        self._current_env: CryptoTradingEnv | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Randomly select symbol for this episode
        idx = int(self.np_random.integers(0, len(self._envs)))
        self._current_symbol = list(self._envs.keys())[idx]
        self._current_env = self._envs[self._current_symbol]
        obs, info = self._current_env.reset(seed=seed, options=options)
        info["symbol"] = self._current_symbol
        return obs, info

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_env is None:
            raise RuntimeError("Must call reset() before step()")
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        info["symbol"] = self._current_symbol
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._current_env and self.render_mode == "human":
            self._current_env.render()
