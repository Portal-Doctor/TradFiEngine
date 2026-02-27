"""Paper trading broker — simulates orders without real money."""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from .base import BaseBroker, Balance, OrderResult


class PaperBroker(BaseBroker):
    """
    Simulates trading with virtual balance.
    Uses config fees, slippage, and current price for PnL calculation.
    Mirrors live conditions: slippage simulates market impact and spread.
    Logs executions for comparison with live spread during paper-trade validation.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        quote_currency: str = "USDT",
        taker_fee: float = 0.006,
        maker_fee: float = 0.004,
        slippage_pct: float = 0.0008,
        log_executions: bool = True,
    ):
        self._balance = initial_balance
        self._quote = quote_currency
        self._positions: dict[str, float] = {}  # symbol_base -> amount
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_pct = slippage_pct  # 0.08% default; simulates worse fill than mid
        self._price_source: dict[str, float] = {}  # symbol -> last known price
        self.log_executions = log_executions
        self._log = logging.getLogger("TradFiEngine.PaperBroker")

    def set_price(self, symbol: str, price: float) -> None:
        """Update current price for a symbol (e.g. from historical bar or live feed)."""
        self._price_source[symbol] = price

    def get_balance(self, currency: str = "USDT") -> Balance:
        if currency == self._quote:
            total = self._balance
            for sym, amount in self._positions.items():
                price = self._price_source.get(sym, 0.0)
                total += amount * price
            return Balance(free=self._balance, used=0.0, total=total, currency=currency)
        # Base currency (e.g. BTC): return position amount
        for sym, amount in self._positions.items():
            base = sym.split("-")[0] if "-" in sym else sym
            if base == currency:
                return Balance(free=amount, used=0.0, total=amount, currency=currency)
        return Balance(free=0.0, used=0.0, total=0.0, currency=currency)

    def get_price(self, symbol: str) -> float:
        if symbol in self._price_source:
            return self._price_source[symbol]
        raise ValueError(f"No price set for {symbol}. Call set_price() first.")

    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> OrderResult:
        mid_price = self.get_price(symbol)
        # Slippage: buy at worse (higher), sell at worse (lower) — simulates spread/impact
        if side == "buy":
            exec_price = mid_price * (1.0 + self.slippage_pct)
        else:
            exec_price = mid_price * (1.0 - self.slippage_pct)
        price = exec_price
        base, quote = symbol.split("-") if "-" in symbol else (symbol, "USDT")
        cost = amount * price
        fee = cost * self.taker_fee

        if side == "buy":
            if cost + fee > self._balance:
                amount = (self._balance - fee) / price if price > 0 else 0.0
                cost = amount * price
                fee = cost * self.taker_fee
            self._balance -= cost + fee
            self._positions[base] = self._positions.get(base, 0.0) + amount
        else:
            held = self._positions.get(base, 0.0)
            amount = min(amount, held)
            cost = amount * price
            fee = cost * self.taker_fee
            self._balance += cost - fee
            self._positions[base] = held - amount
            if self._positions[base] <= 0:
                del self._positions[base]

        if self.log_executions:
            self._log.info(
                "PAPER %s %s %.8f %s @ %.2f (mid %.2f, slippage %.4f%%)",
                side.upper(),
                amount,
                symbol,
                exec_price,
                mid_price,
                self.slippage_pct * 100,
            )

        return OrderResult(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            fee=fee,
            filled=True,
            raw=None,
        )

    def get_ticker_fee(self, symbol: str) -> tuple[float, float]:
        return (self.maker_fee, self.taker_fee)
