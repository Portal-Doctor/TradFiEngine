"""Train the RL agent on historical data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import requests
from dotenv import load_dotenv

from config import load_config
from src.engine import DataIngestor
from src.environment import MultiAssetTradingEnv

# Load environment variables (API keys, Telegram tokens)
load_dotenv()


def make_env(
    config: dict,
    *,
    data_source: str | None = None,
    timeframe: str | None = None,
    limit: int | None = None,
):
    """Create MultiAssetTradingEnv with config-driven DataIngestor."""
    if data_source is not None or timeframe is not None or limit is not None:
        train_cfg = config.get("training", {})
        env_cfg = config.get("env", {})
        symbols = env_cfg.get("symbols") or ["BTC-USDT"]
        symbols = symbols if isinstance(symbols, list) else [symbols]
        data_ingestor = DataIngestor(
            data_source or train_cfg.get("data_source", "fetch"),
            symbol=symbols[0] if symbols else "BTC-USDT",
            timeframe=timeframe or train_cfg.get("timeframe", "1h"),
            limit=limit if limit is not None else train_cfg.get("fetch_limit", 2000),
        )
    else:
        data_ingestor = DataIngestor(config)
    return MultiAssetTradingEnv(config, data_ingestor)


class IterationProgressCallback(BaseCallback):
    """Prints 'Iteration n of m' progress during training."""

    def __init__(self, total_timesteps: int, n_steps: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self._total_iterations = (total_timesteps + n_steps - 1) // n_steps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        current = self.model.num_timesteps
        steps_per_iter = self.n_steps * self.model.n_envs
        iteration = (current + steps_per_iter - 1) // steps_per_iter
        if self.verbose > 0:
            print(f"\n>>> Iteration {iteration} of {self._total_iterations} <<<\n")
        return True


def send_telegram_message(message: str):
    """Sends a message via Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials missing, skipping notification.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
        print("Telegram notification sent.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def main():
    config = load_config()
    paths = config.get("paths", {})
    default_checkpoint = str(Path(paths.get("checkpoints", "checkpoints")) / "tradfibot")
    train_cfg = config.get("training", {})
    env_cfg = config.get("env", {})
    symbols = env_cfg.get("symbols") or ["BTC-USDT"]
    if not isinstance(symbols, list):
        symbols = [symbols]

    parser = argparse.ArgumentParser(description="Train TradFiBot on historical data (multi-asset)")
    parser.add_argument("--data", type=str, default=None, help="Path to OHLCV CSV, or 'fetch' (default)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (1h, 4h, etc.)")
    parser.add_argument("--limit", type=int, default=2000, help="Bars to fetch when using --data fetch")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (default from config)")
    parser.add_argument("--episode-bars", type=int, default=None, help="Bars per episode (default from config)")
    parser.add_argument("--save", type=str, default=default_checkpoint, help="Model save path")
    args = parser.parse_args()

    data_source = args.data or train_cfg.get("data_source", "fetch")
    if args.episode_bars is not None:
        config = dict(config)
        config.setdefault("env", {})["episode_bars"] = args.episode_bars
    symbol_list_str = ",".join(symbols[:5]) + ("..." if len(symbols) > 5 else "")
    send_telegram_message(
        f"🚀 Training started for {len(symbols)} pair(s): {symbol_list_str} on {args.timeframe} timeframe."
    )

    print(f"Loading multi-asset env for {len(symbols)} symbols via DataIngestor (source={data_source})...")
    env = make_env(
        config,
        data_source=data_source,
        timeframe=args.timeframe,
        limit=args.limit,
    )
    print(f"Env ready: {env.num_symbols} symbols, {env.num_features} features, {len(env.index)} aligned bars.")

    total_timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)
    n_steps = train_cfg.get("n_steps", 2048)
    ppo_kwargs = {
        "n_steps": n_steps,
        "learning_rate": train_cfg.get("learning_rate", 3e-4),
        "batch_size": train_cfg.get("batch_size", 64),
        "gamma": train_cfg.get("gamma", 0.99),
        "verbose": 1,
    }

    model = PPO("MlpPolicy", env, **ppo_kwargs)
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=IterationProgressCallback(total_timesteps, n_steps=n_steps),
    )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    print(f"Model saved to {args.save}")

    send_telegram_message(f"✅ Training completed. Model saved: {args.save}")


if __name__ == "__main__":
    main()