"""Legacy environments (domain randomization, single-symbol-per-episode).

Use MultiAssetTradingEnv for the current multi-asset training pipeline.
"""

from .multi_symbol_env import MultiSymbolTradingEnv

__all__ = ["MultiSymbolTradingEnv"]
