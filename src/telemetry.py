"""
Telemetry: push alerts for trades, circuit breaker, errors.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env for Telegram notifications.
Otherwise alerts are logged only.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

_log = logging.getLogger("TradFiEngine.Telemetry")


def send_alert(
    event_type: Literal["trade", "circuit_breaker", "error", "warning"],
    message: str,
    *,
    extra: dict | None = None,
) -> None:
    """
    Send alert. If Telegram configured, posts to channel; else logs.
    """
    formatted = _format_message(event_type, message, extra)
    _log.info("[%s] %s", event_type.upper(), formatted)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        _send_telegram(token, chat_id, formatted)


def _format_message(
    event_type: str,
    message: str,
    extra: dict | None,
) -> str:
    prefix = {
        "trade": "🚀",
        "circuit_breaker": "⚠️ CIRCUIT BREAKER:",
        "error": "❌ ERROR:",
        "warning": "⚠️ WARNING:",
    }.get(event_type, "")
    out = f"{prefix} {message}".strip()
    if extra:
        parts = [f"{k}={v}" for k, v in extra.items()]
        out += " " + " | ".join(parts)
    return out


def _send_telegram(token: str, chat_id: str, text: str) -> None:
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=5) as _:
            pass
    except Exception as e:
        _log.warning("Telegram send failed: %s", e)


def alert_trade(side: str, amount: float, symbol: str, price: float) -> None:
    """E.g. 🚀 BOUGHT 0.05 BTC @ $64,200"""
    send_alert("trade", f"{side.upper()} {amount:.4f} {symbol} @ ${price:,.0f}")


def alert_circuit_breaker(drawdown_pct: float) -> None:
    """E.g. ⚠️ WARNING: Daily drawdown hit 2%. Kill switch engaged."""
    send_alert("circuit_breaker", f"Daily drawdown hit {drawdown_pct:.1f}%. Kill switch engaged.")


def alert_error(msg: str) -> None:
    """E.g. ❌ ERROR: Coinbase API Timeout (504)."""
    send_alert("error", msg)


def alert_warning(msg: str) -> None:
    """E.g. ⚠️ WARNING: Execution failed for BTC-USDT."""
    send_alert("warning", msg)
