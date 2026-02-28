"""Train the RL agent on historical data."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import requests
from dotenv import load_dotenv

from config import load_config
from src.data import create_exchange, fetch_ohlcv_ccxt, load_ohlcv

# Use same exchange as live (Coinbase); Binance is geo-restricted in some regions
FETCH_EXCHANGE_MULTI = "coinbase"
# Coinbase allows 10 req/s; 2s between fetches = 0.5 req/s (conservative)
FETCH_DELAY_SEC = 2.0

from src.environment import CryptoTradingEnv, MultiSymbolTradingEnv

# Load environment variables (API keys, Telegram tokens)
load_dotenv()


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
    symbols_cfg = config.get("symbols", {})
    default_symbols = symbols_cfg.get("training") or [symbols_cfg.get("default", "BTC-USDT")]

    parser = argparse.ArgumentParser(description="Train TradFiBot on historical data")
    parser.add_argument("--data", type=str, help="Path to OHLCV CSV, or 'fetch' to use CCXT")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. BTC-USDT)")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Multiple symbols: 'all' for config symbols.training, or comma-separated (e.g. BTC-USDT,ETH-USDT)",
    )
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (1h, 4h, etc.)")
    parser.add_argument("--limit", type=int, default=2000, help="Bars to fetch when using --data fetch")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (default from config)")
    parser.add_argument("--save", type=str, default=default_checkpoint, help="Model save path (Phase 1 only)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    args = parser.parse_args()

    train_cfg = config.get("training", {})

    # Resolve symbol list: --symbols all | --symbols a,b,c | --symbol x | default
    use_multi = False
    if args.symbols:
        if args.symbols.strip().lower() == "all":
            symbols = [s if isinstance(s, str) else str(s) for s in default_symbols]
            use_multi = len(symbols) > 1
        else:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            use_multi = len(symbols) > 1
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s if isinstance(s, str) else str(s) for s in default_symbols[:1]]  # Single default

    if not symbols:
        symbols = ["BTC-USDT"]

    symbol_list_str = ",".join(symbols[:5]) + ("..." if len(symbols) > 5 else "")
    send_telegram_message(
        f"🚀 Training started for {len(symbols)} pair(s): {symbol_list_str} on {args.timeframe} timeframe."
    )

    if use_multi:
        # Multi-symbol: fetch all, create MultiSymbolTradingEnv
        if args.data and args.data != "fetch":
            raise ValueError("--symbols (multi) requires --data fetch")
        print(f"Fetching {len(symbols)} symbols from CCXT (delay={FETCH_DELAY_SEC}s between fetches)...")
        exchange = create_exchange(FETCH_EXCHANGE_MULTI)
        data_by_symbol: dict = {}
        for sym in symbols:
            try:
                sym_ccxt = sym.replace("-", "/")
                df = fetch_ohlcv_ccxt(
                    sym_ccxt,
                    args.timeframe,
                    limit=args.limit,
                    exchange_id=FETCH_EXCHANGE_MULTI,
                    exchange=exchange,
                )
                if df is not None and len(df) > 100:
                    data_by_symbol[sym] = df
                    print(f"  {sym}: {len(df)} bars")
                else:
                    print(f"  {sym}: skipped (insufficient data)")
            except Exception as e:
                print(f"  {sym}: skipped ({e})")
            time.sleep(FETCH_DELAY_SEC)  # Coinbase rate limit: stay under 10 req/s
        if not data_by_symbol:
            raise ValueError("No valid symbol data fetched")
        env = MultiSymbolTradingEnv(
            symbols=list(data_by_symbol.keys()),
            data_by_symbol=data_by_symbol,
            config=config,
            timeframe=args.timeframe,
        )
        print(f"Training on {len(data_by_symbol)} pairs (random symbol per episode)")
    else:
        # Single-symbol
        if args.data == "fetch" or not args.data:
            print(f"Fetching {symbols[0]} {args.timeframe} from CCXT...")
            symbol_ccxt = symbols[0].replace("-", "/")
            df = fetch_ohlcv_ccxt(symbol_ccxt, args.timeframe, limit=args.limit)
        else:
            df = load_ohlcv(args.data)
        print(f"Loaded {len(df)} bars.")
        env = CryptoTradingEnv(df, config=config)

    total_timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)
    n_steps = train_cfg.get("n_steps", 2048)  # PPO default
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        batch_size=train_cfg.get("batch_size", 64),
        gamma=train_cfg.get("gamma", 0.99),
        verbose=1,
    )

    total_iters = (total_timesteps + n_steps - 1) // n_steps
    print(f"Training for {total_timesteps} timesteps ({total_iters} iterations)...")
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