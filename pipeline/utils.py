import time
import os
from functools import wraps
from typing import Callable, Any

# -- Written by qqdex --


def timeit(func: Callable) -> Callable:
    """Decorator that measures and prints wall-clock execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


def cpu_count_physical() -> int:
    """Return the number of physical CPU cores, or logical if unavailable."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def speedup(t_serial: float, t_parallel: float) -> float:
    """Amdahl-style speedup ratio: serial_time / parallel_time."""
    if t_parallel <= 0:
        return float("inf")
    return t_serial / t_parallel


def efficiency(speedup_val: float, n_workers: int) -> float:
    """Parallel efficiency: speedup / n_workers."""
    if n_workers == 0:
        return 0.0
    return speedup_val / n_workers
