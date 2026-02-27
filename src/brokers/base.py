"""Base broker interface for trading execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    fee: float
    filled: bool
    raw: dict | None = None


@dataclass
class Balance:
    """Account balance snapshot."""

    free: float
    used: float
    total: float
    currency: str


class BaseBroker(ABC):
    """Abstract base for paper and live brokers."""

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> Balance:
        """Fetch current balance for a currency."""
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Get current mid price for symbol."""
        ...

    @abstractmethod
    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> OrderResult:
        """Place a market order. Amount in base currency (e.g. BTC)."""
        ...

    @abstractmethod
    def get_ticker_fee(self, symbol: str) -> tuple[float, float]:
        """Return (maker_fee, taker_fee) as decimals."""
        ...
