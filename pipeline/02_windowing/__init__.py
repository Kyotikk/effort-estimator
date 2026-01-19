"""Phase 2: Windowing - Create sliding windows and perform quality checks."""

from .windowing import (
    create_windows,
    quality_check_windows,
)

__all__ = [
    "create_windows",
    "quality_check_windows",
]
