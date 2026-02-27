"""Executor: handles API calls and order types (Limit vs Market)."""

from __future__ import annotations

import logging
import math
import time
from typing import Literal

from src.brokers.base import BaseBroker, OrderResult


def _round_to_increment(value: float, increment: float) -> float:
    """Round value to exchange's base_increment. Avoids precision rejection."""
    if increment <= 0:
        return round(value, 8)
    prec = max(0, -int(math.floor(math.log10(increment)))) if increment < 1 else 0
    stepped = round(value / increment) * increment
    return round(stepped, min(8, prec))


class Executor:
    """
    Consumer of signals, producer of orders.
    Handles Market vs Limit orders; pre-flight checks; precision rounding.
    """

    def __init__(
        self,
        broker: BaseBroker,
        default_order_type: Literal["market", "limit"] = "market",
        max_retries: int = 3,
    ):
        self.broker = broker
        self.default_order_type = default_order_type
        self.max_retries = max_retries
        self.logger = logging.getLogger("TradFiEngine.Executor")

    def _preflight_check(self, symbol: str, side: str, amount: float, price: float) -> str | None:
        """
        Pre-flight: avoid "Insufficient Funds" by checking balance before order.
        Returns error message if check fails, else None.
        """
        base, quote = (symbol.split("-") + ["USDT"])[:2]
        if side == "buy":
            required = amount * price
            bal = self.broker.get_balance(quote)
            if bal.free < required * 1.001:  # 0.1% buffer for fees
                return f"Insufficient {quote}: need {required:.2f}, free {bal.free:.2f}"
        else:
            bal = self.broker.get_balance(base)
            if bal.free < amount:
                return f"Insufficient {base}: need {amount:.6f}, free {bal.free:.6f}"
        return None

    def execute(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        order_type: Literal["market", "limit"] | None = None,
        limit_price: float | None = None,
        skip_preflight: bool = False,
    ) -> OrderResult:
        """
        Place order with precision rounding and retries.
        Pre-flight checks balance to avoid Insufficient Funds loop.
        """
        ot = order_type or self.default_order_type
        info = self.broker.get_symbol_info(symbol)
        base_inc = info.get("base_increment", 1e-8)
        quote_inc = info.get("quote_increment", 1e-2)
        safe_amount = _round_to_increment(max(0, amount), base_inc)

        if safe_amount <= 0:
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                amount=0,
                price=0,
                fee=0,
                filled=False,
                success=False,
                error_message="Amount round-to-zero",
            )

        price = limit_price or self.broker.get_price(symbol)
        if limit_price is not None:
            price = _round_to_increment(limit_price, quote_inc)

        if not skip_preflight:
            err = self._preflight_check(symbol, side, safe_amount, price)
            if err:
                self.logger.warning("Pre-flight failed: %s", err)
                return OrderResult(
                    order_id="",
                    symbol=symbol,
                    side=side,
                    amount=safe_amount,
                    price=price,
                    fee=0,
                    filled=False,
                    success=False,
                    error_message=err,
                )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if ot == "limit" and limit_price is not None:
                    self.logger.info(
                        "Placing LIMIT %s for %.8f %s @ %s",
                        side,
                        safe_amount,
                        symbol,
                        limit_price,
                    )
                    # TODO: broker.create_limit_order(symbol, side, safe_amount, limit_price)
                    # Fall back to market until limit is implemented
                    pass

                self.logger.info("Placing MARKET %s for %.8f %s", side, safe_amount, symbol)
                result = self.broker.create_market_order(symbol, side, safe_amount)
                return result
            except Exception as e:
                last_exc = e
                self.logger.error("Execution error: %s. Retry %d/%d", e, attempt + 1, self.max_retries)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return OrderResult(
            order_id="",
            symbol=symbol,
            side=side,
            amount=safe_amount,
            price=price,
            fee=0,
            filled=False,
            success=False,
            error_message=f"Max retries exceeded: {last_exc}" if last_exc else "Max retries exceeded",
        )
