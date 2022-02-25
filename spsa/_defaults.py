from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Optional

__all__ = [
    "executor"
    "timeout"
    "adam",
    "iterations",
    "lr",
    "lr_decay",
    "lr_power",
    "px",
    "px_decay",
    "px_power",
    "momentum",
    "beta",
    "epsilon",
]

executor: Executor = ProcessPoolExecutor()
timeout: float = 1e-4
adam: bool = True
iterations: int = 10_000
lr: Optional[float] = None
lr_decay: float = 1e-3
lr_power: float = 0.5
px: Optional[float] = None
px_decay: float = 1e-2
px_power: float = 0.161
momentum: float = 0.97
beta: float = 0.999
epsilon: float = 1e-7
