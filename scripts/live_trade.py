"""Live trading via CCXT (Coinbase). Use sub-profile with limited funds."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Live trade with TradFiBot (Coinbase via CCXT). "
        "Use a sub-profile with limited funds."
    )
    parser.add_argument("--symbol", type=str, default="BTC-USDT")
    parser.add_argument("--dry-run", action="store_true", help="Connect and log only; no real orders")
    parser.add_argument("--model", type=str, default="checkpoints/tradfibot", help="Path to trained model")
    args = parser.parse_args()

    from config import load_config
    from src.brokers import CCXTBroker
    from src.core import CircuitBreaker, OrderTracker

    config = load_config()
    live_cfg = config.get("live", {})

    model_path = f"{args.model}.zip" if not args.model.endswith(".zip") else args.model
    if not Path(model_path).exists():
        print(f"\nNo model found at {model_path}.")
        print("Train first (Learning phase) and run Paper Trading before going live.")
        return

    broker = CCXTBroker(exchange_id="coinbase")
    broker._api_retry_max = live_cfg.get("api_retry_max", 5)

    circuit = CircuitBreaker(max_daily_drawdown_pct=live_cfg.get("max_daily_drawdown_pct", 5.0))
    order_tracker = OrderTracker(db_path=live_cfg.get("order_db_path", "data/orders.db"))

    bal = broker.get_balance("USDT")
    price = broker.get_price(args.symbol)

    # Circuit breaker: record day start, check before trading
    total_value = bal.total
    circuit.record_day_start(total_value)
    if not circuit.check(total_value):
        print("\n[CIRCUIT BREAKER] Max daily drawdown exceeded. Trading halted.")
        return

    # Check for open orders from previous run (crash recovery)
    open_orders = order_tracker.get_open_orders(args.symbol)
    if open_orders:
        print(f"\n  Recovered {len(open_orders)} open order(s) from previous run.")

    print(f"Connected to Coinbase")
    print(f"  USDT balance: {bal.total:.2f}")
    print(f"  {args.symbol} price: {price:.2f}")

    if args.dry_run:
        print("\n[DRY RUN] No orders will be placed.")
        return

    # Live trading loop would go here:
    # - Load model
    # - Fetch latest OHLCV, compute indicators
    # - Get action from model
    # - Check circuit.check(current_value) before each trade
    # - order_tracker.add(...) before order; mark_filled after
    # - Execute via broker.create_market_order()
    # - Sleep until next candle
    print("\nLive execution not fully implemented yet.")
    print("Use paper_trade.py to validate first. Implement live loop when ready.")


if __name__ == "__main__":
    main()
