"""Order tracking for crash recovery — SQLite-backed open orders."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class TrackedOrder:
    """Order persisted for crash recovery."""

    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    status: Literal["open", "filled", "cancelled"]
    created_at: str


class OrderTracker:
    """
    Tracks open orders in SQLite. On restart, bot can query open orders
    instead of placing duplicates.
    """

    def __init__(self, db_path: str | Path = "data/orders.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL
                )
            """)

    def add(self, order_id: str, symbol: str, side: str, amount: float, price: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO orders (order_id, symbol, side, amount, price, status, created_at) VALUES (?,?,?,?,?,'open',?)",
                (order_id, symbol, side, amount, price, datetime.utcnow().isoformat()),
            )

    def add_filled(self, order_id: str, symbol: str, side: str, amount: float, price: float) -> None:
        """Record a completed trade for dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders (order_id, symbol, side, amount, price, status, created_at) VALUES (?,?,?,?,?,'filled',?)",
                (order_id, symbol, side, amount, price, datetime.utcnow().isoformat()),
            )

    def mark_filled(self, order_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE orders SET status = 'filled' WHERE order_id = ?", (order_id,))

    def mark_cancelled(self, order_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))

    def get_open_orders(self, symbol: str | None = None) -> list[TrackedOrder]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM orders WHERE status = 'open' AND symbol = ?",
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM orders WHERE status = 'open'").fetchall()
        return [
            TrackedOrder(
                order_id=r["order_id"],
                symbol=r["symbol"],
                side=r["side"],
                amount=r["amount"],
                price=r["price"],
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
