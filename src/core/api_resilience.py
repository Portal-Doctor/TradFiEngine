"""API resilience: exponential backoff for 429 (rate limit) and 5xx errors."""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: exponential backoff for 429 and 5xx.
    Retries: 1s, 2s, 4s, 8s, 16s (capped at max_delay).
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: BaseException | None = None
            delay = base_delay
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    status = getattr(e, "status", None) or getattr(
                        e, "response", {}
                    ).get("status") if hasattr(e, "response") else None
                    status = status or (getattr(e, "status_code", None))
                    if isinstance(status, int):
                        if status == 429 or 500 <= status < 600:
                            if attempt < max_retries:
                                time.sleep(min(delay, max_delay))
                                delay *= 2
                                continue
                    raise
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
