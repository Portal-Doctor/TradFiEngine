"""Train the RL agent on historical data."""

from __future__ import annotations

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from config import load_config
from src.data import load_ohlcv, fetch_ohlcv_ccxt
from src.environment import CryptoTradingEnv


def make_env(df, config):
    def _init():
        return CryptoTradingEnv(df, config=config)
    return _init


def main():
    config = load_config()
    paths = config.get("paths", {})
    default_checkpoint = str(Path(paths.get("checkpoints", "checkpoints")) / "tradfibot")

    parser = argparse.ArgumentParser(description="Train TradFiBot on historical data")
    parser.add_argument("--data", type=str, help="Path to OHLCV CSV, or 'fetch' to use CCXT")
    parser.add_argument("--symbol", type=str, default="BTC-USDT", help="Symbol for fetch (e.g. BTC-USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (1h, 4h, etc.)")
    parser.add_argument("--limit", type=int, default=2000, help="Bars to fetch when using --data fetch")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (default from config)")
    parser.add_argument("--save", type=str, default=default_checkpoint, help="Model save path (Phase 1 only)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    args = parser.parse_args()
    train_cfg = config.get("training", {})

    if args.data == "fetch" or not args.data:
        print(f"Fetching {args.symbol} {args.timeframe} from CCXT...")
        symbol_ccxt = args.symbol.replace("-", "/")
        df = fetch_ohlcv_ccxt(symbol_ccxt, args.timeframe, limit=args.limit)
        df = df.rename(columns={"timestamp": "timestamp"})
    else:
        df = load_ohlcv(args.data)

    print(f"Loaded {len(df)} bars. Columns: {list(df.columns)}")

    if args.n_envs > 1:
        env = make_vec_env(
            make_env(df, config),
            n_envs=args.n_envs,
            vec_env_cls=DummyVecEnv,
        )
    else:
        env = CryptoTradingEnv(df, config=config)

    total_timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        batch_size=train_cfg.get("batch_size", 64),
        gamma=train_cfg.get("gamma", 0.99),
        verbose=1,
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
