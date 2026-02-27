"""Run paper trading with a trained model or simple strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import load_config
from src.data import fetch_ohlcv_ccxt
from src.indicators import add_all_indicators
from src.brokers import PaperBroker


def run_paper_trade(
    df: pd.DataFrame,
    broker: PaperBroker,
    model=None,
    initial_balance: float = 10_000.0,
) -> dict:
    """
    Run paper trading over historical data.
    If model is provided, use it for actions; else hold (no trades).
    """
    config = load_config()
    ind_cfg = config.get("indicators", {})
    df = add_all_indicators(
        df,
        macd_fast=ind_cfg.get("macd", {}).get("fast", 12),
        macd_slow=ind_cfg.get("macd", {}).get("slow", 26),
        macd_signal=ind_cfg.get("macd", {}).get("signal", 9),
        rsi_length=ind_cfg.get("rsi", {}).get("length", 14),
    )
    df = df.dropna().reset_index(drop=True)

    position = 0.0
    balance = initial_balance
    taker_fee = broker.taker_fee
    portfolio_values = []

    for i in range(60, len(df)):  # Start after warmup
        price = float(df.iloc[i]["close"])
        broker.set_price("BTC-USDT", price)

        if model is not None:
            # Build observation (simplified: use recent window)
            window = df.iloc[i - 60 : i][["returns", "RSI", "ATR"]].values
            window = np.nan_to_num(window, nan=0.0)
            obs = window.flatten().astype(np.float32)
            if len(obs) < 180:  # 60 * 3
                obs = np.pad(obs, (0, 180 - len(obs)))
            action, _ = model.predict(obs, deterministic=True)
            target_pct = float(np.clip(action[0], 0, 1))
        else:
            target_pct = 0.5  # Hold 50% if no model

        target_value = initial_balance * target_pct * 0.95
        target_position = target_value / price if price > 0 else 0.0

        # Execute trade
        trade_amount = abs(target_position - position)
        if trade_amount > 0:
            side = "buy" if target_position > position else "sell"
            result = broker.create_market_order("BTC-USDT", side, trade_amount)
            position = target_position
            balance = broker.get_balance("USDT").total

        portfolio_values.append(balance + position * price)

    final_value = balance + position * price
    returns = (final_value - initial_balance) / initial_balance
    return {
        "initial": initial_balance,
        "final": final_value,
        "return_pct": returns * 100,
        "values": portfolio_values,
    }


def main():
    parser = argparse.ArgumentParser(description="Paper trade with TradFiBot")
    parser.add_argument("--symbol", type=str, default="BTC-USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", type=str, default="checkpoints/tradfibot", help="Path to trained model .zip")
    args = parser.parse_args()

    config = load_config()
    fees = config.get("fees", {})

    print("Fetching data...")
    symbol_ccxt = args.symbol.replace("-", "/")
    df = fetch_ohlcv_ccxt(symbol_ccxt, args.timeframe, limit=args.limit)

    broker = PaperBroker(
        initial_balance=10_000.0,
        taker_fee=fees.get("taker", 0.006),
        maker_fee=fees.get("maker", 0.004),
    )

    model = None
    if args.model and Path(args.model).exists():
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model loaded — running buy-and-hold 50% for demo")

    result = run_paper_trade(df, broker, model=model)
    print(f"\nPaper Trade Results ({args.symbol} {args.timeframe})")
    print(f"  Initial: ${result['initial']:,.2f}")
    print(f"  Final:   ${result['final']:,.2f}")
    print(f"  Return:  {result['return_pct']:.2f}%")


if __name__ == "__main__":
    main()
