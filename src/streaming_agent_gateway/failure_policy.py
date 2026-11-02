"""Failure classification, retry timing, and final output validation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping


RETRYABLE_CODES = frozenset({"timeout", "temporary_network_error", "rate_limited", "overloaded"})


class OutputValidationError(Exception):
    """A final model output cannot drive the agent loop safely."""

    def __init__(self, message: str, code: str = "invalid_schema") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class FailureInfo:
    code: str
    message: str
    retry_after: float | None = None

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_CODES


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 8.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("retry delays and jitter must not be negative")

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return max(0.0, retry_after)
        exponential = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        return exponential + random.uniform(0.0, exponential * self.jitter)


def classify_provider_error(error: BaseException) -> FailureInfo:
    """Normalize common SDK/HTTP error shapes without importing an SDK."""

    status = getattr(error, "status_code", getattr(error, "status", None))
    headers = getattr(error, "headers", {}) or {}
    retry_after = _retry_after(headers)
    if status == 429:
        return FailureInfo("rate_limited", str(error), retry_after)
    if status in {408, 504} or isinstance(error, TimeoutError):
        return FailureInfo("timeout", str(error), retry_after)
    if status in {502, 503}:
        return FailureInfo("overloaded", str(error), retry_after)
    if isinstance(error, (ConnectionError, OSError)):
        return FailureInfo("temporary_network_error", str(error), retry_after)
    return FailureInfo("provider_error", str(error), retry_after)


def _retry_after(headers: Mapping[str, Any]) -> float | None:
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


OutputValidator = Callable[[str], Any]
