"""Run paper trading on historical data with the multi-asset model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from config import load_config
from src.engine import DataIngestor, load_multi_symbol, slice_multi_symbol
from src.environment import MultiAssetTradingEnv


def compute_performance_report(equity_curve: list[float], bars_per_year: float = 252 * 24) -> dict:
    """Compute total return, max drawdown, and Sharpe ratio."""
    equity = np.array(equity_curve, dtype=np.float64)
    if len(equity) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    returns = equity[1:] / equity[:-1] - 1.0
    total_return = float(equity[-1] / equity[0] - 1.0)

    max_equity = np.maximum.accumulate(equity)
    drawdowns = (equity - max_equity) / np.where(max_equity > 0, max_equity, 1e-8)
    max_drawdown = float(drawdowns.min())

    ret_std = returns.std()
    sharpe = (
        float(returns.mean() / (ret_std + 1e-8) * np.sqrt(bars_per_year))
        if ret_std > 1e-12
        else 0.0
    )

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def main():
    config = load_config()
    paths = config.get("paths", {})
    default_model = str(Path(paths.get("checkpoints", "checkpoints")) / "tradfibot")

    parser = argparse.ArgumentParser(
        description="Paper trade multi-asset model on historical data"
    )
    parser.add_argument("--model", type=str, default=default_model, help="Path to trained model .zip")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2022-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=2000, help="Bars to fetch per symbol")
    args = parser.parse_args()

    symbols = config["env"]["symbols"]
    if not isinstance(symbols, list):
        symbols = [symbols]

    print("Loading multi-symbol data...")
    train_cfg = config.get("training", {})
    data_ingestor = DataIngestor(
        train_cfg.get("data_source", "fetch"),
        symbol=symbols[0],
        timeframe=args.timeframe,
        limit=args.limit,
    )
    data_ingestor.exchange_id = train_cfg.get("exchange_id", "coinbase")
    full_dfs = load_multi_symbol(data_ingestor, symbols, config=config)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    dfs = slice_multi_symbol(full_dfs, start, end)

    # Ensure we have enough data
    min_bars = config["env"]["window_size"] + 10
    n_bars = len(next(iter(dfs.values())))
    if n_bars < min_bars:
        print(f"Error: Only {n_bars} bars in range {args.start}–{args.end}. Need >= {min_bars}.")
        return

    print(f"Sliced to {n_bars} bars ({args.start}–{args.end})")

    env = MultiAssetTradingEnv(config, data_ingestor=None, dfs=dfs, paper_mode=True)

    model = None
    model_path = Path(args.model)
    model_path_zip = Path(f"{args.model}.zip") if not str(args.model).endswith(".zip") else model_path
    if model_path.exists() or model_path_zip.exists():
        model = PPO.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model found — using random actions (for testing)")

    obs, info = env.reset()
    done = False
    equity_curve = []
    actions_log = []
    info_log = []

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        equity_curve.append(info["equity"])
        actions_log.append(action)
        info_log.append(info)

        done = terminated or truncated

    report = compute_performance_report(
        equity_curve,
        bars_per_year=252 * 24 if args.timeframe == "1h" else 252,
    )

    print("\n--- Paper Trade Report ---")
    print(f"  Period:     {args.start} – {args.end}")
    print(f"  Symbols:    {len(symbols)}")
    print(f"  Bars:       {len(equity_curve)}")
    print(f"  Final:      ${equity_curve[-1]:,.2f}")
    print(f"  Total Return:  {report['total_return']*100:.2f}%")
    print(f"  Max Drawdown:  {report['max_drawdown']*100:.2f}%")
    print(f"  Sharpe:        {report['sharpe']:.2f}")


if __name__ == "__main__":
    main()
