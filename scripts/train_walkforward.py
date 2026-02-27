"""Walk-forward training: rolling window to better simulate evolving market regimes."""

from __future__ import annotations

import argparse
import warnings

# Minimum recommended train window to span different market regimes (bull/bear)
MIN_TRAIN_MONTHS_FOR_REGIMES = 6
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import load_config
from src.data import load_ohlcv, fetch_ohlcv_ccxt
from src.environment import CryptoTradingEnv

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


def _split_by_months(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    timestamp_col: str = "timestamp",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train_df, test_df) for each rolling window."""
    if timestamp_col not in df.columns and "Timestamp" in df.columns:
        timestamp_col = "Timestamp"
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    total = len(df)
    # Approximate: 1 month ~ 720 bars at 1h
    bars_per_month = 720
    train_bars = train_months * bars_per_month
    test_bars = test_months * bars_per_month
    step = bars_per_month  # Roll by 1 month
    folds = []
    start = 0
    while start + train_bars + test_bars <= total:
        train_df = df.iloc[start : start + train_bars]
        test_df = df.iloc[start + train_bars : start + train_bars + test_bars]
        folds.append((train_df, test_df))
        start += step
    return folds


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward training: train on N months, validate on next M months, roll."
    )
    parser.add_argument("--data", type=str, default="fetch")
    parser.add_argument("--symbol", type=str, default="BTC-USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--train-months", type=int, default=6, help="Train window (months); >=6 recommended for bull/bear regimes")
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--save-prefix", type=str, default="checkpoints/wf_")
    args = parser.parse_args()

    config = load_config()
    train_cfg = config.get("training", {})

    if args.data == "fetch" or not args.data:
        symbol_ccxt = args.symbol.replace("-", "/")
        df = fetch_ohlcv_ccxt(symbol_ccxt, args.timeframe, limit=args.limit)
    else:
        df = load_ohlcv(args.data)

    folds = _split_by_months(df, args.train_months, args.test_months)
    if not folds:
        print("Not enough data for walk-forward. Need more bars.")
        return

    if args.train_months < MIN_TRAIN_MONTHS_FOR_REGIMES:
        warnings.warn(
            f"train_months={args.train_months} may be too short to encompass different "
            f"market regimes (bull/bear). Consider --train-months >= {MIN_TRAIN_MONTHS_FOR_REGIMES}.",
            UserWarning,
            stacklevel=0,
        )

    Path(args.save_prefix).parent.mkdir(parents=True, exist_ok=True)
    for i, (train_df, test_df) in enumerate(folds):
        print(f"\n=== Fold {i + 1}/{len(folds)}: train {len(train_df)} bars, test {len(test_df)} bars ===")
        env = CryptoTradingEnv(train_df, config=config)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            batch_size=train_cfg.get("batch_size", 64),
            gamma=train_cfg.get("gamma", 0.99),
            verbose=1,
        )
        model.learn(total_timesteps=args.timesteps)
        save_path = f"{args.save_prefix}fold{i + 1}"
        model.save(save_path)
        print(f"Saved {save_path}")

    print(f"\nWalk-forward complete. {len(folds)} models saved.")


if __name__ == "__main__":
    main()
