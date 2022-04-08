import asyncio
import operator
from concurrent.futures import Executor
from typing import Callable, Optional, Type, Union

import numpy as np

import spsa._defaults as DEFAULTS
import spsa._spsa as _spsa
from spsa._spsa import with_input_noise
from spsa._utils import ArrayLike, type_check

__all__ = ["maximize", "minimize", "with_input_noise"]

def _optimize(
    optimizer: Callable[..., np.ndarray],
    f: Callable[[np.ndarray], float],
    x: bytes,
    adam: bool,
    iterations: int,
    lr: float,
    lr_decay: float,
    lr_power: float,
    px: Union[float, Type[int]],
    px_decay: float,
    px_power: float,
    momentum: float,
    beta: float,
    epsilon: float,
    /,
) -> bytes:
    """Helper function for optimizing in a separate process."""
    return optimizer(
        f,
        np.frombuffer(x).copy(),
        adam=adam,
        iterations=iterations,
        lr=lr,
        lr_decay=lr_decay,
        lr_power=lr_power,
        px=px,
        px_decay=px_decay,
        px_power=px_power,
        momentum=momentum,
        beta=beta,
        epsilon=epsilon,
    ).tobytes()

async def maximize(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULTS.executor,
    adam: bool = DEFAULTS.adam,
    iterations: int = DEFAULTS.iterations,
    lr: Optional[float] = DEFAULTS.lr,
    lr_decay: float = DEFAULTS.lr_decay,
    lr_power: float = DEFAULTS.lr_power,
    px: Union[float, Type[int]] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> np.ndarray:
    """
    An asynchronous optimizer accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    See `help(spsa.amp.minimize)` for more details.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    return np.frombuffer(await asyncio.get_running_loop().run_in_executor(
        executor,
        _optimize,
        _spsa.maximize,
        f,
        x.tobytes(),
        bool(operator.index(adam)),
        operator.index(iterations),
        float(lr) if lr is not None else None,
        float(lr_decay),
        float(lr_power),
        float(px) if px is not int else px,
        float(px_decay),
        float(px_power),
        float(momentum),
        float(beta),
        float(epsilon),
    )).copy()

async def minimize(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULTS.executor,
    adam: bool = DEFAULTS.adam,
    iterations: int = DEFAULTS.iterations,
    lr: Optional[float] = DEFAULTS.lr,
    lr_decay: float = DEFAULTS.lr_decay,
    lr_power: float = DEFAULTS.lr_power,
    px: Union[float, Type[int]] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> np.ndarray:
    """
    An asynchronous optimizer accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    See `help(spsa.minimize)` and `help(spsa.amp)` for more details.

    Parameters
    -----------
        executor:
            An executor for SPSA to run in.
            If `None`, uses `asyncio`'s default `ThreadPoolExecutor`.
            Recommended that `ThreadPoolExecutor`s should not be used.
            Use `spsa.aio` instead of threading.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    return np.frombuffer(await asyncio.get_running_loop().run_in_executor(
        executor,
        _optimize,
        _spsa.minimize,
        f,
        x.tobytes(),
        bool(operator.index(adam)),
        operator.index(iterations),
        float(lr) if lr is not None else None,
        float(lr_decay),
        float(lr_power),
        float(px) if px is not int else px,
        float(px_decay),
        float(px_power),
        float(momentum),
        float(beta),
        float(epsilon),
    )).copy()
