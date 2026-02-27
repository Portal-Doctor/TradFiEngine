"""Circuit breaker: kill switch for max daily drawdown."""

from __future__ import annotations

from datetime import date, datetime
from typing import NamedTuple


class DailySnapshot(NamedTuple):
    """Daily PnL snapshot for circuit breaker."""

    date: date
    start_value: float
    current_value: float

    @property
    def drawdown_pct(self) -> float:
        if self.start_value <= 0:
            return 0.0
        return (self.start_value - self.current_value) / self.start_value * 100


class CircuitBreaker:
    """
    Kill switch: stops trading if daily drawdown exceeds threshold.
    Resets at midnight (date change).
    """

    def __init__(self, max_daily_drawdown_pct: float = 5.0):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self._day_start_value: float | None = None
        self._day_start_date: date | None = None
        self._tripped: bool = False

    def record_day_start(self, value: float) -> None:
        """Call at start of each trading day to set baseline."""
        today = date.today()
        if self._day_start_date != today:
            self._day_start_value = value
            self._day_start_date = today
            self._tripped = False

    def check(self, current_value: float) -> bool:
        """
        Returns True if OK to trade, False if circuit tripped (kill switch).
        """
        today = date.today()
        if self._day_start_date != today:
            self._day_start_value = current_value
            self._day_start_date = today
            self._tripped = False
        if self._day_start_value is None:
            self._day_start_value = current_value
        dd = (self._day_start_value - current_value) / self._day_start_value * 100
        if dd >= self.max_daily_drawdown_pct:
            self._tripped = True
            return False
        return not self._tripped

    @property
    def tripped(self) -> bool:
        return self._tripped
