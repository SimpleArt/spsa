"""
Contains a collection of iterator variants of `spsa.amp`.

Each iteration, a `dict` of variables is generated,
allowing the iterations to be logged or custom
termination algorithms to be used.

NOTE: Frequent communication between processes has overhead,
      use with caution.

NOTE: Recommended that `ThreadPoolExecutor`s should not be used.
      Use `spsa.aio` instead of threading.
"""
import asyncio
import operator
from concurrent.futures import Executor
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import AsyncIterator, Callable, Iterator, Optional, Type, Union

import numpy as np

import spsa._defaults as DEFAULTS
import spsa.iterator
from spsa._spsa import with_input_noise
from spsa._utils import ArrayLike, OptimizerVariables, type_check

__all__ = ["maximize", "minimize"]

def _optimize(
    optimizer: Callable[..., Iterator[OptimizerVariables]],
    f: Callable[[np.ndarray], float],
    x: bytes,
    conn: Connection,
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
) -> None:
    """Helper generator for optimizing in a separate process."""
    with conn:
        for variables in optimizer(
            f,
            np.array(np.frombuffer(x), dtype=float),
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
        ):
            conn.send({
                key: (value.tobytes() if isinstance(value, np.ndarray) else value)
                for key, value in variables.items()
            })

async def maximize(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULTS.executor,
    timeout: float = DEFAULTS.timeout,
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
) -> AsyncIterator[OptimizerVariables]:
    """
    An asynchronous generator accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    See `help(spsa.amp.iterator.minimize)` for more details.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    parent, child = Pipe()
    task = asyncio.get_running_loop().run_in_executor(
        executor,
        _optimize,
        spsa.iterator.maximize,
        f,
        x.tobytes(),
        child,
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
    )
    try:
        for _ in range(iterations + 1):
            while not task.done() and not parent.poll():
                await asyncio.sleep(timeout)
            if task.done():
                await task.result()
            yield {
                key: (np.frombuffer(value) if isinstance(value, bytes) else value)
                for key, value in parent.recv().items()
            }
    except (asyncio.CancelledError, GeneratorExit):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return
    except BrokenPipeError:
        pass
    except BaseException as e:
        print(type(e), e)
        raise
    await task.result()

async def minimize(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULTS.executor,
    timeout: float = DEFAULTS.timeout,
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
) -> AsyncIterator[OptimizerVariables]:
    """
    An asynchronous generator accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    NOTE: Frequent communication between processes has overhead,
          use with caution.

    See `help(spsa.iterator.minimize)` and `help(spsa.amp)` for more details.

    Parameters
    -----------
        executor:
            An executor for SPSA to run in.
            If `None`, uses `asyncio`'s default `ThreadPoolExecutor`.
            Recommended that `ThreadPoolExecutor`s should not be used.
            Use `spsa.aio` instead of threading.

        timeout:
            Timeout for when there are no new variables to generate.
            Avoids busy waiting.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    parent, child = Pipe()
    task = asyncio.get_running_loop().run_in_executor(
        executor,
        _optimize,
        spsa.iterator.minimize,
        f,
        x.tobytes(),
        child,
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
    )
    try:
        for _ in range(iterations + 1):
            while not task.done() and not parent.poll():
                await asyncio.sleep(timeout)
            if task.done():
                await task.result()
            yield {
                key: (np.frombuffer(value) if isinstance(value, bytes) else value)
                for key, value in parent.recv().items()
            }
    except (asyncio.CancelledError, GeneratorExit):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return
    except BrokenPipeError:
        pass
    await task.result()
