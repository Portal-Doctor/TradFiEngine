"""Trading brokers: paper and live (CCXT, Coinbase CDP)."""

from .base import BaseBroker
from .paper import PaperBroker
from .ccxt_broker import CCXTBroker

try:
    from .coinbase_broker import CoinbaseBroker
    __all__ = ["BaseBroker", "PaperBroker", "CCXTBroker", "CoinbaseBroker"]
except ImportError:
    CoinbaseBroker = None  # type: ignore[misc, assignment]
    __all__ = ["BaseBroker", "PaperBroker", "CCXTBroker"]
