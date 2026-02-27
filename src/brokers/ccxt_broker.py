"""CCXT broker for live trading on Coinbase (and other exchanges)."""

from __future__ import annotations

import time
from typing import Callable, Literal, TypeVar

from .base import BaseBroker, Balance, OrderResult

T = TypeVar("T")


def _is_retriable(exc: BaseException) -> bool:
    """True if 429 (rate limit) or 5xx (server error)."""
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg and "limit" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    return False


def _api_with_retry(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """Exponential backoff for 429/5xx."""
    last_exc: BaseException | None = None
    delay = base_delay
    for _ in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if _is_retriable(e) and delay <= max_delay:
                time.sleep(min(delay, max_delay))
                delay *= 2
                continue
            raise
    raise last_exc  # type: ignore[misc]


class CCXTBroker(BaseBroker):
    """
    Live trading via CCXT. Supports Coinbase (coinbase).
    Use a sub-profile with limited funds for safety.
    """

    def __init__(
        self,
        exchange_id: str = "coinbase",
        api_key: str | None = None,
        secret: str | None = None,
        passphrase: str | None = None,
        sandbox: bool = False,
    ):
        try:
            import ccxt
        except ImportError:
            raise ImportError("Install ccxt: pip install ccxt")

        self._exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange {exchange_id} not found in CCXT")

        config = {
            "apiKey": api_key or "",
            "secret": secret or "",
            "password": passphrase or "",
            "sandbox": sandbox,
            "enableRateLimit": True,
        }
        self._exchange = exchange_class(config)
        self._loaded_fees: dict[str, tuple[float, float]] = {}
        self._api_retry_max = 5

    def _ensure_auth(self) -> None:
        if not self._exchange.apiKey:
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()
                self._exchange.apiKey = os.getenv("COINBASE_API_KEY", "")
                self._exchange.secret = os.getenv("COINBASE_SECRET", "")
                self._exchange.password = os.getenv("COINBASE_PASSPHRASE", "")
            except ImportError:
                pass

    def get_balance(self, currency: str = "USDT") -> Balance:
        self._ensure_auth()

        def _fetch() -> Balance:
            bal = self._exchange.fetch_balance()
            c = bal.get(currency, {})
            if isinstance(c, dict):
                free = float(c.get("free", 0) or 0)
                used = float(c.get("used", 0) or 0)
                total = float(c.get("total", 0) or 0)
            else:
                free = used = total = 0.0
            return Balance(free=free, used=used, total=total, currency=currency)

        return _api_with_retry(_fetch, max_retries=self._api_retry_max)

    def get_price(self, symbol: str) -> float:
        self._ensure_auth()

        def _fetch() -> float:
            ticker = self._exchange.fetch_ticker(symbol)
            bid = float(ticker.get("bid", 0) or 0)
            ask = float(ticker.get("ask", 0) or 0)
            if bid and ask:
                return (bid + ask) / 2
            return float(ticker.get("last", 0) or 0)

        return _api_with_retry(_fetch, max_retries=self._api_retry_max)

    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> OrderResult:
        self._ensure_auth()

        def _create() -> OrderResult:
            order = self._exchange.create_market_order(symbol, side, amount)
            fee_cost = 0.0
            if order.get("fee") and order["fee"].get("cost"):
                fee_cost = float(order["fee"]["cost"])
            elif order.get("fees"):
                fee_cost = sum(float(f.get("cost", 0) or 0) for f in order["fees"])
            return OrderResult(
                order_id=str(order.get("id", "")),
                symbol=symbol,
                side=side,
                amount=float(order.get("filled", amount) or amount),
                price=float(order.get("average", order.get("price", 0)) or 0),
                fee=fee_cost,
                filled=order.get("status") == "closed",
                raw=order,
            )

        return _api_with_retry(_create, max_retries=self._api_retry_max)

    def get_symbol_info(self, symbol: str) -> dict:
        """Fetch precision from exchange markets. Coinbase requires correct decimals."""
        self._ensure_auth()
        try:
            markets = self._exchange.load_markets()
            m = markets.get(symbol) or self._exchange.market(symbol)
        except Exception:
            return {"base_increment": 1e-8, "quote_increment": 1e-2}
        prec = m.get("precision", {})
        amt, prc = prec.get("amount"), prec.get("price")

        def to_increment(v, default: float) -> float:
            if v is None:
                return default
            if isinstance(v, (int, float)) and v > 0 and v < 1:
                return float(v)  # already increment
            if isinstance(v, int):
                return 10 ** (-v)
            return default

        return {
            "base_increment": to_increment(amt, 1e-8),
            "quote_increment": to_increment(prc, 1e-2),
        }

    def get_ticker_fee(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._loaded_fees:
            try:
                trading_fees = self._exchange.fetch_trading_fee(symbol)
                m = float(trading_fees.get("maker", {}).get("percentage", 0.006) or 0.006)
                t = float(trading_fees.get("taker", {}).get("percentage", 0.006) or 0.006)
                self._loaded_fees[symbol] = (m, t)
            except Exception:
                self._loaded_fees[symbol] = (0.004, 0.006)  # Coinbase defaults
        return self._loaded_fees[symbol]
