"""Custom exceptions and global error handlers."""

from __future__ import annotations


class IndexNotReadyError(RuntimeError):
    """Raised when the index is not ready."""


class GenerationTimeoutError(TimeoutError):
    """Raised when generation times out."""