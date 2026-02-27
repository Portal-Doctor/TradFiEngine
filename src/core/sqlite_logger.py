"""SQLiteLogger — Connective tissue between Executor and Dashboard."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.brokers.base import OrderResult


class SQLiteLogger:
    """
    Logs successful trades to SQLite for the monitoring dashboard.
    Ensures every trade is recorded with the precision needed for performance analysis.
    """

    def __init__(self, db_path: str | Path = "data/orders.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._log = logging.getLogger("TradFiEngine.SQLiteLogger")

    def _init_db(self) -> None:
        """Creates the orders table if it doesn't exist; adds fee/realized_pnl if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    price REAL,
                    fee REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    status TEXT,
                    signal_price REAL
                )
            """)
            # Migrate: add fee, realized_pnl if table existed from older OrderTracker schema
            cur = conn.execute("PRAGMA table_info(orders)")
            cols = {row[1] for row in cur.fetchall()}
            if "fee" not in cols:
                conn.execute("ALTER TABLE orders ADD COLUMN fee REAL DEFAULT 0.0")
            if "realized_pnl" not in cols:
                conn.execute("ALTER TABLE orders ADD COLUMN realized_pnl REAL DEFAULT 0.0")
            if "signal_price" not in cols:
                conn.execute("ALTER TABLE orders ADD COLUMN signal_price REAL")

    def calculate_realized_pnl(
        self, symbol: str, sell_price: float, sell_amount: float, sell_fee: float
    ) -> float:
        """
        Calculates PnL based on the weighted average cost of the current position.
        AVG cost method (not FIFO): suitable for bot performance tracking.
        Formula: ((Sell Price - Avg Buy Price) * Amount) - Sell Fee
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT side, amount, price, fee FROM orders
                WHERE symbol = ? AND status = 'filled'
                ORDER BY created_at
                """,
                (symbol,),
            ).fetchall()

        total_qty = 0.0
        total_cost = 0.0
        for side, amt, price, fee in rows:
            amt, price, fee = float(amt or 0), float(price or 0), float(fee or 0)
            if side == "buy":
                total_cost += amt * price
                total_qty += amt
            else:
                sell_amt = min(amt, total_qty) if total_qty > 0 else 0
                if total_qty > 0 and sell_amt > 0:
                    avg_price = total_cost / total_qty
                    total_cost -= avg_price * sell_amt
                total_qty = max(0, total_qty - amt)

        if total_qty <= 0:
            return 0.0
        avg_buy_price = total_cost / total_qty
        raw_pnl = (sell_price - avg_buy_price) * sell_amount
        net_pnl = raw_pnl - sell_fee
        return round(net_pnl, 2)

    def log_order(self, order_result: "OrderResult", signal_price: float | None = None) -> None:
        """Inserts an OrderResult into the database for dashboard/analytics."""
        realized_pnl = 0.0
        if order_result.side == "sell" and order_result.success:
            realized_pnl = self.calculate_realized_pnl(
                order_result.symbol,
                order_result.price,
                order_result.amount,
                order_result.fee or 0.0,
            )
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO orders
                    (order_id, created_at, symbol, side, amount, price, fee, realized_pnl, status, signal_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_result.order_id or "",
                        datetime.utcnow().isoformat(),
                        order_result.symbol,
                        order_result.side,
                        order_result.amount,
                        order_result.price,
                        order_result.fee or 0.0,
                        realized_pnl,
                        "filled" if order_result.success else "failed",
                        signal_price,
                    ),
                )
        except Exception as e:
            self._log.error("Failed to log order to SQLite: %s", e)
