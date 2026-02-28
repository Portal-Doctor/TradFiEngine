"""Multi-asset Gymnasium environment: simultaneous allocation across all symbols."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.engine.multi_symbol_ingestor import load_multi_symbol


def _price_col(df) -> str:
    return "close" if "close" in df.columns else "Close"


class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset trading environment. Observes all symbols at once
    (num_symbols, window_size, num_features). Action is allocation weights
    across assets (num_symbols,), normalized so sum <= 1 (remainder in cash).

    Pass dfs=... for pre-loaded data (e.g. sliced for paper trading).
    Pass paper_mode=True for deterministic sequential evaluation (no random starts).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict,
        data_ingestor=None,
        *,
        dfs: dict | None = None,
        paper_mode: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self.paper_mode = paper_mode
        self.symbols = list(config["env"]["symbols"])
        self.window_size = int(config["env"]["window_size"])
        self.episode_bars = int(config["env"]["episode_bars"])
        self.starting_cash = float(config["env"]["starting_cash"])

        # Load data: use pre-loaded dfs if provided, else load via data_ingestor
        if dfs is not None:
            self.dfs = dict(dfs)
        elif data_ingestor is not None:
            self.dfs = load_multi_symbol(data_ingestor, self.symbols, config=config)
        else:
            raise ValueError("Provide either dfs (pre-loaded) or data_ingestor")

        self.index = next(iter(self.dfs.values())).index
        self.num_symbols = len(self.symbols)

        # Assume all dfs have same columns/features
        sample_df = next(iter(self.dfs.values()))
        self.num_features = sample_df.shape[1]
        self._price_col = _price_col(sample_df)

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_symbols, self.window_size, self.num_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_symbols,),
            dtype=np.float32,
        )

        # State
        self._reset_portfolio()

    def _reset_portfolio(self) -> None:
        self.cash = float(self.starting_cash)
        self.positions = np.zeros(self.num_symbols, dtype=np.float32)
        self.equity = self.cash
        self.prev_equity = self.equity

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_portfolio()

        if self.paper_mode:
            # Deterministic: run the whole period sequentially
            self.start_idx = 0
        else:
            # Random episode start for training
            max_start = len(self.index) - (self.episode_bars + self.window_size + 1)
            if max_start < 0:
                raise ValueError(
                    f"Insufficient data: need {self.episode_bars + self.window_size + 1} bars, "
                    f"got {len(self.index)}"
                )
            self.start_idx = int(self.np_random.integers(0, max_start + 1))

        self.t = self.start_idx + self.window_size  # current bar index

        obs = self._get_observation()
        info: dict = {}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(
            (self.num_symbols, self.window_size, self.num_features),
            dtype=np.float32,
        )
        for i, s in enumerate(self.symbols):
            df = self.dfs[s]
            window = df.iloc[self.t - self.window_size : self.t].values
            obs[i] = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs

    def step(
        self,
        action: np.ndarray | list[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        # Normalize to sum <= 1, keep rest as cash
        total = float(action.sum())
        if total > 1.0:
            weights = action / total
        else:
            weights = action
        cash_weight = 1.0 - float(weights.sum())

        # Prices at time t (before advancing)
        prices = np.array(
            [float(self.dfs[s].iloc[self.t][self._price_col]) for s in self.symbols],
            dtype=np.float32,
        )

        # Equity before rebalancing
        self.prev_equity = self.equity
        self.equity = self.cash + np.sum(self.positions * prices)

        # Target holdings
        target_values = weights * self.equity
        target_positions = np.where(
            prices > 0,
            target_values / prices,
            0.0,
        ).astype(np.float32)

        # Trades
        trades = target_positions - self.positions
        fee_rate = float(self.config.get("fees", {}).get("taker", 0.006))
        fees = float(np.sum(np.abs(trades * prices) * fee_rate))

        # Update cash and positions
        self.cash = self.equity * cash_weight - fees
        self.positions = target_positions

        # Advance time
        self.t += 1
        if self.paper_mode:
            terminated = self.t >= len(self.index) - 1
        else:
            terminated = self.t >= self.start_idx + self.window_size + self.episode_bars

        # Prices at new time (or last bar if terminated)
        t_price = min(self.t, len(self.index) - 1)
        new_prices = np.array(
            [float(self.dfs[s].iloc[t_price][self._price_col]) for s in self.symbols],
            dtype=np.float32,
        )
        # Equity after move
        self.equity = self.cash + np.sum(self.positions * new_prices)

        reward = float((self.equity / self.prev_equity) - 1.0) if self.prev_equity > 0 else 0.0

        obs = self._get_observation()
        info = {
            "equity": self.equity,
            "cash": self.cash,
            "positions": self.positions.copy(),
        }
        return obs, reward, terminated, False, info

    def render(self) -> None:
        if self.render_mode == "human":
            print(f"t={self.t}, equity={self.equity:.2f}, cash={self.cash:.2f}")
