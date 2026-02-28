"""Trading engine: Producer-Consumer architecture."""

from .data_ingestor import DataIngestor
from .executor import Executor
from .multi_symbol_ingestor import MultiSymbolIngestor, load_multi_symbol, slice_multi_symbol
from .state_buffer import StateBuffer
from .strategy_brain import StrategyBrain

__all__ = [
    "DataIngestor",
    "Executor",
    "MultiSymbolIngestor",
    "StateBuffer",
    "StrategyBrain",
    "load_multi_symbol",
    "slice_multi_symbol",
]
