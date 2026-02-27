"""Coinbase Advanced Trade broker via CDP SDK (coinbase-advanced-py)."""

from __future__ import annotations

import uuid
from typing import Literal

from .base import BaseBroker, Balance, OrderResult


def _to_product_id(symbol: str) -> str:
    """Map internal symbol to Coinbase product_id. BTC-USDT → BTC-USD; BTC-USDC → BTC-USDC."""
    s = symbol.upper()
    if s.endswith("-USDT"):
        return s[:-4] + "-USD"  # USDT → USD
    return s  # BTC-USD, BTC-USDC, etc. unchanged


class CoinbaseBroker(BaseBroker):
    """
    Live trading via Coinbase Advanced Trade (CDP SDK).
    Market buy = quote_size (USD). Market sell = base_size (crypto).
    Uses client_order_id (UUID) to prevent duplicate orders on network lag.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        key_file: str | None = None,
    ):
        try:
            from coinbase.rest import RESTClient
        except ImportError:
            raise ImportError("Install CDP SDK: pip install coinbase-advanced-py")

        if key_file:
            self._client = RESTClient(key_file=key_file)
        elif api_key and api_secret:
            self._client = RESTClient(api_key=api_key, api_secret=api_secret)
        else:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            self._client = RESTClient()

        self._loaded_fees: dict[str, tuple[float, float]] = {}

    def get_balance(self, currency: str = "USDT") -> Balance:
        resp = self._client.get_accounts()
        currency = currency.replace("USDT", "USD")  # Coinbase uses USD
        for acc in getattr(resp, "accounts", []) or []:
            ccy = getattr(acc, "currency", None) or (acc.get("currency") if isinstance(acc, dict) else None)
            if str(ccy) == currency:
                ab = getattr(acc, "available_balance", None) or acc.get("available_balance", {})
                hb = getattr(acc, "hold", None) or acc.get("hold", {})
                if isinstance(ab, dict):
                    free = float(ab.get("value", 0) or 0)
                else:
                    free = float(getattr(ab, "value", 0) or 0)
                if isinstance(hb, dict):
                    used = float(hb.get("value", 0) or 0)
                else:
                    used = float(getattr(hb, "value", 0) or 0)
                return Balance(free=free, used=used, total=free + used, currency=currency)
        return Balance(free=0.0, used=0.0, total=0.0, currency=currency)

    def get_price(self, symbol: str) -> float:
        product_id = _to_product_id(symbol)
        resp = self._client.get_product(product_id)
        p = getattr(resp, "price", None) or (getattr(resp, "product", None) and getattr(resp.product, "price", None))
        if p is None and hasattr(resp, "to_dict"):
            d = resp.to_dict()
            p = d.get("price") or (d.get("product") or {}).get("price")
        return float(p or 0)

    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> OrderResult:
        """
        Execute market order. Amount in base (BTC).
        Buy: converts to quote_size (USD). Sell: uses base_size (crypto).
        Returns OrderResult(success=False) on 403 / Insufficient Funds — do not retry.
        """
        product_id = _to_product_id(symbol)
        client_order_id = str(uuid.uuid4())

        try:
            if side == "buy":
                price = self.get_price(symbol)
                quote_size = str(round(amount * price, 2))  # USD
                resp = self._client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=quote_size,
                )
            else:
                base_size = str(amount)  # Crypto
                resp = self._client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=base_size,
                )

            order = getattr(resp, "order", resp)
            try:
                raw = order.to_dict() if hasattr(order, "to_dict") else (order if isinstance(order, dict) else {})
            except Exception:
                raw = {}
            order_id = raw.get("order_id") or getattr(order, "order_id", None) or client_order_id
            filled = raw.get("filled_size") or getattr(order, "filled_size", None)
            tf = raw.get("total_fees") or getattr(order, "total_fees", None)
            fee_cost = float(tf.get("value", 0) or 0) if isinstance(tf, dict) else float(getattr(tf, "value", 0) or 0)
            price = self.get_price(symbol)

            return OrderResult(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                amount=float(filled or amount) if filled else amount,
                price=price,
                fee=fee_cost,
                filled=bool(filled),
                raw=raw,
            )
        except Exception as e:
            msg = str(e).lower()
            return OrderResult(
                order_id=client_order_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=0,
                fee=0,
                filled=False,
                success=False,
                error_message=str(e),
            )

    def get_ticker_fee(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._loaded_fees:
            self._loaded_fees[symbol] = (0.004, 0.006)  # Coinbase maker/taker
        return self._loaded_fees[symbol]

    def get_symbol_info(self, symbol: str) -> dict:
        try:
            product_id = _to_product_id(symbol)
            resp = self._client.get_product(product_id)
            p = getattr(resp, "product", resp)
            base_inc = getattr(p, "base_increment", None) or (p.get("base_increment") if isinstance(p, dict) else None)
            quote_inc = getattr(p, "quote_increment", None) or (p.get("quote_increment") if isinstance(p, dict) else None)
            return {
                "base_increment": float(base_inc or 1e-8),
                "quote_increment": float(quote_inc or 1e-2),
            }
        except Exception:
            return {"base_increment": 1e-8, "quote_increment": 1e-2}
