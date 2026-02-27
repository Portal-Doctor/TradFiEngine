"""TradFiBot — Start menu for the automated trading engine."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def show_menu() -> str:
    """Display start menu and return selected phase."""
    print()
    print("=" * 50)
    print("  TradFiBot — Automated Crypto Trading Engine")
    print("=" * 50)
    print()
    print("  Objectives: Up to 10 trades/day • Double capital in 30 days")
    print()
    print("  Select a phase:")
    print()
    print("    1. Learning        — Train on historical data (MACD, RSI, fees)")
    print("    2. Paper Trading   — Validate strategy with simulated capital")
    print("    3. Real Trading    — Live execution via Coinbase (CCXT)")
    print()
    print("    0. Exit")
    print()
    return input("  Enter choice [1-3, 0]: ").strip() or "0"


def run_learning() -> None:
    from scripts.train import main
    main()


def run_paper_trading() -> None:
    from scripts.paper_trade import main
    main()


def run_real_trading() -> None:
    from scripts.live_trade import main
    main()


def main() -> None:
    while True:
        choice = show_menu()
        if choice == "0":
            print("\n  Goodbye.")
            break
        if choice == "1":
            run_learning()
            break
        if choice == "2":
            run_paper_trading()
            break
        if choice == "3":
            run_real_trading()
            break
        print("\n  Invalid choice. Please enter 1, 2, 3, or 0.")


if __name__ == "__main__":
    main()
