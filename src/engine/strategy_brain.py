"""Strategy Brain: ML model or logic outputting Buy/Sell/Hold signal."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Any

import numpy as np

Signal = Literal["buy", "sell", "hold"]


class StrategyBrain:
    """
    Consumer of observations, producer of actions.
    Outputs target position [0, 1] or Buy/Sell/Hold.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler: Any = None,
        expected_obs_dim: int | None = None,
    ):
        self.logger = logging.getLogger("TradFiEngine.Brain")
        self._model = None
        self.scaler = scaler
        self.expected_obs_dim = expected_obs_dim

        if model_path:
            path = Path(model_path)
            path_zip = path.with_suffix(".zip") if path.suffix != ".zip" else path
            if path_zip.exists():
                try:
                    from stable_baselines3 import PPO
                    self._model = PPO.load(str(path_zip))
                    self.logger.info("Model loaded from %s", path_zip)
                except ImportError:
                    self.logger.error("stable-baselines3 not installed")
            else:
                self.logger.warning("Model file %s not found. Operating in safety mode (0%% position).", path_zip)

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Ensure obs shape and optionally apply scaling."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if self.expected_obs_dim is not None and obs.size != self.expected_obs_dim:
            self.logger.warning(
                "Observation dim %d does not match expected %d. Output may be invalid.",
                obs.size,
                self.expected_obs_dim,
            )
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return obs

    def predict(self, obs: np.ndarray) -> float:
        """
        Returns target position in [0, 1] (0=all cash, 1=max long).
        No model: returns 0.0 (100%% cash) for safety — avoid unintended exposure.
        """
        if self._model is None:
            return 0.0

        obs = self._preprocess(obs)
        action, _ = self._model.predict(obs, deterministic=True)

        # Handle both Box (continuous) and Discrete action spaces
        if isinstance(action, np.ndarray):
            val = float(action[0]) if action.size else 0.0
        else:
            # Discrete: map 0->0, 1->0.5, 2->1.0 or similar; env uses Box(0,1)
            val = float(action)
        return float(np.clip(val, 0.0, 1.0))

    def to_signal(
        self,
        target_pct: float,
        current_pct: float,
        threshold: float = 0.05,
    ) -> Signal:
        """
        Convert target position to Buy/Sell/Hold.
        Hysteresis (threshold) prevents churning from small moves; reduces fees.
        Caller must compute current_pct from live balance:
          current_pct = (position * price) / total_portfolio_value
        """
        diff = target_pct - current_pct
        if diff > threshold:
            return "buy"
        if diff < -threshold:
            return "sell"
        return "hold"
