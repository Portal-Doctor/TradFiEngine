"""Gymnasium environment for crypto trading with fee and arbitrage awareness."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from src.indicators import add_all_indicators
from config import load_config


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize OHLCV column names to lowercase."""
    mapping = {c: c.lower() for c in df.columns}
    return df.rename(columns=mapping)


class CryptoTradingEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent on historical crypto data.
    Fee-aware: every trade incurs taker fee. Reward discourages unprofitable trades.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        initial_balance: float = 10_000.0,
        taker_fee: float = 0.006,
        maker_fee: float = 0.004,
        min_profit_pct: float = 0.012,
        max_position_pct: float = 0.95,
        window_size: int = 60,
        reward_scale: float = 1.0,
        episode_bars: int | None = 500,
        bars_per_day: int = 24,
        max_trades_per_day: int = 10,
        config: dict | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed

        cfg = config or load_config()
        fees = cfg.get("fees", {})
        env_cfg = cfg.get("env", {})
        obj_cfg = cfg.get("objectives", {})

        self.initial_balance = initial_balance or env_cfg.get("initial_balance", 10_000.0)
        self.bars_per_day = bars_per_day
        self.max_trades_per_day = max_trades_per_day or obj_cfg.get("max_trades_per_day", 10)
        self.taker_fee = taker_fee if taker_fee is not None else fees.get("taker", 0.006)
        self.maker_fee = maker_fee if maker_fee is not None else fees.get("maker", 0.004)
        self.min_profit_pct = min_profit_pct or fees.get("min_profit_pct", 0.012)
        self.max_position_pct = max_position_pct or env_cfg.get("max_position_pct", 0.95)
        self.window_size = window_size or env_cfg.get("window_size", 60)
        self.reward_scale = reward_scale or env_cfg.get("reward_scale", 1.0)
        self.episode_bars = episode_bars if episode_bars is not None else env_cfg.get("episode_bars", 500)

        df = _normalize_cols(df)
        ind_cfg = cfg.get("indicators", {})
        macd = ind_cfg.get("macd", {})
        rsi = ind_cfg.get("rsi", {})
        bb = ind_cfg.get("bollinger", {})
        atr = ind_cfg.get("atr", {})

        feat_cfg = cfg.get("features", {})
        use_log_returns = feat_cfg.get("use_log_returns", False)
        self.df = add_all_indicators(
            df,
            macd_fast=macd.get("fast", 12),
            macd_slow=macd.get("slow", 26),
            macd_signal=macd.get("signal", 9),
            rsi_length=rsi.get("length", 14),
            bb_length=bb.get("length", 20),
            bb_std=bb.get("std", 2.0),
            atr_length=atr.get("length", 14),
            use_log_returns=use_log_returns,
        )

        # Drop rows with NaN from indicators (warmup period)
        self.df = self.df.dropna().reset_index(drop=True)
        self.price_col = "close" if "close" in self.df.columns else "Close"

        # Feature columns for observation (exclude raw OHLCV, keep normalized/fractional)
        self.feature_cols = self._get_feature_cols()
        self.n_features = len(self.feature_cols)
        self.obs_dim = self.window_size * self.n_features

        # Action: continuous [0, 1] = target position (0 = all cash, 1 = max long)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: flattened window of features
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._step_idx: int = 0
        self._balance: float = 0.0
        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._last_value: float = 0.0
        self._trades_today: int = 0
        self._day_start_bar: int = 0
        self._episode_start_idx: int = 0
        self._episode_end_idx: int = 0

    def _get_feature_cols(self) -> list[str]:
        """Select normalized/normalizable columns for observation."""
        skip = {"open", "high", "low", "close", "volume", "timestamp"}
        cols = [c for c in self.df.columns if c.lower() not in skip]
        if not cols:
            return list(self.df.columns[: min(5, len(self.df.columns))])
        return cols

    def _get_obs(self) -> np.ndarray:
        # Look-ahead bias prevention: obs uses [step-window, step) — no future data.
        # At bar t we observe features for t-1 and earlier; step() then uses close(t) for execution.
        start = max(0, self._step_idx - self.window_size)
        end = self._step_idx
        window = self.df.iloc[start:end][self.feature_cols]
        # Pad with zeros if not enough history
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self.n_features), dtype=np.float32)
            window = np.vstack([pad, window.values.astype(np.float32)])
        else:
            window = window.values.astype(np.float32)
        # Replace NaN/Inf with 0
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        return window.flatten()

    def _portfolio_value(self, price: float) -> float:
        return self._balance + self._position * price

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Random start: each episode uses a different historical segment (no lookahead)
        min_start = self.window_size
        max_start = len(self.df) - self.episode_bars - 1  # ensure start + episode_bars fits
        max_start = max(min_start, max_start)
        if max_start > min_start:
            start_idx = int(self.np_random.integers(min_start, max_start + 1))
        else:
            start_idx = min_start
        self._episode_start_idx = start_idx
        self._episode_end_idx = min(start_idx + self.episode_bars, len(self.df) - 1)
        self._step_idx = start_idx
        self._balance = self.initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._trades_today = 0
        self._day_start_bar = self._step_idx
        price = float(self.df.iloc[self._step_idx][self.price_col])
        self._last_value = self._portfolio_value(price)
        obs = self._get_obs()
        info = {"balance": self._balance, "position": self._position, "value": self._last_value}
        return obs, info

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        price = float(self.df.iloc[self._step_idx][self.price_col])
        action_val = np.asarray(action, dtype=np.float32).flat
        target_pct = float(np.clip(action_val[0] if len(action_val) else 0.0, 0.0, 1.0))

        # Enforce max 10 trades per day
        bars_into_day = self._step_idx - self._day_start_bar
        if bars_into_day >= self.bars_per_day:
            self._day_start_bar = self._step_idx
            self._trades_today = 0

        target_value = self.initial_balance * target_pct * self.max_position_pct
        target_position = target_value / price if price > 0 else 0.0

        # Cap position by affordable amount (can't spend more than balance + sale proceeds)
        max_buy_value = self._balance + self._position * price
        max_position = max_buy_value / price if price > 0 else 0.0
        target_position = min(target_position, max_position)
        target_position = max(0.0, target_position)

        # Execute trade: adjust position to target (respect max trades/day)
        trade_value = abs(target_position - self._position) * price
        would_trade = trade_value > 0
        if would_trade and self._trades_today >= self.max_trades_per_day:
            target_position = self._position  # Hold — daily limit reached
            trade_value = 0.0

        fee = trade_value * self.taker_fee if trade_value > 0 else 0.0
        if would_trade and trade_value > 0:
            self._trades_today += 1

        # Update state
        old_position = self._position
        self._position = target_position
        self._balance = self._balance - (target_position - old_position) * price - fee
        if target_position <= 0:
            self._entry_price = 0.0
        elif old_position <= 0:
            self._entry_price = price
        # else keep entry_price for hold

        new_value = self._portfolio_value(price)
        reward = (new_value - self._last_value) / self.initial_balance * self.reward_scale

        # Penalty for trades that don't beat min profit (arbitrage awareness)
        if trade_value > 0 and fee > 0:
            required_profit = trade_value * self.min_profit_pct
            if fee >= required_profit * 0.5:  # Fee eats into potential profit
                reward -= 0.001  # Small penalty to discourage marginal trades

        self._last_value = new_value
        self._step_idx += 1

        terminated = self._step_idx >= self._episode_end_idx or self._step_idx >= len(self.df) - 1
        truncated = False

        info = {
            "balance": self._balance,
            "position": self._position,
            "value": new_value,
            "fee": fee,
            "price": price,
        }

        obs = self._get_obs() if not terminated else self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            price = float(self.df.iloc[min(self._step_idx, len(self.df) - 1)][self.price_col])
            val = self._portfolio_value(price)
            print(f"Step {self._step_idx} | Price {price:.2f} | Value {val:.2f} | Pos {self._position:.4f}")
