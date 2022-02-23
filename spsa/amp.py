"""
Asynchronous Multiprocessing
-----------------------------

Contains asynchronous variants of `spsa.optimize` and `spsa.optimize_iterator`
for optimizing synchronous functions in separate processes.

See also:
    spsa.aio - Asynchronous IO:
        Runs asynchronous function calls concurrently every iteration. Unlike
        spsa.amp, does not run the SPSA algorithm in a separate process. This
        is more appropriate if the objective function can be ran concurrently
        or if each objective call is significantly more expensive than sharing
        numpy arrays between processes.

Example Uses
-------------
- Large numbers of iterations can be ran concurrently with other asynchronous code.
- The entire SPSA algorithm can be run in separate processes.

Parallelizing Calls with Multiprocessing
-----------------------------------------
`spsa.amp` handles the concurrency itself, running the entire SPSA algorithm in a
separate process. Calls may then be ran concurrently using `asyncio`.

    x = await spsa.amp.optimize(f, ...)

    async for variables in spsa.amp.optimize(f, ...):
        print(variables)
"""
import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import partial
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np

import spsa._spsa as _spsa
from ._spsa import ArrayLike, _type_check

__all__ = ["maximize", "optimize", "optimize_iterator", "with_input_noise"]

DEFAULT_EXECUTOR: ProcessPoolExecutor = ProcessPoolExecutor()

def _maximize_wrapper(f: Callable[[np.ndarray], float], x: np.ndarray, /) -> float:
    """Helper function for maximizing."""
    return -f(x)

def maximize(f: Callable[[np.ndarray], float], /) -> Callable[[np.ndarray], float]:
    """
    Turns the function into a maximization function.

    Usage
    ------
        Maximize a function instead of minimizing it:
            x = spsa.amp.optimize(spsa.amp.maximize(f), x)
    """
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    return partial(_maximize_wrapper, f)

def _rng_iterator(noise: float, shape: Tuple[int, ...], /) -> Iterator[np.ndarray]:
    """Helper generator for producing noise."""
    rng = np.random.default_rng()
    while True:
        random_noise = rng.uniform(-noise, noise, shape)
        yield random_noise
        yield random_noise

def _with_input_noise_wrapper(f: Callable[[np.ndarray], float], x: np.ndarray, /, *, rng: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    """Helper function for adding noise to the input before calling."""
    dx = next(rng)
    return (f(x + dx) + f(x - dx)) / 2

def with_input_noise(f: Callable[[np.ndarray], float], /, shape: Tuple[int, ...], noise: float) -> Callable[[np.ndarray], float]:
    """Adds noise to the input before calling."""
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(noise, SupportsFloat):
        raise TypeError(f"noise must be real, got {noise!r}")
    return partial(_with_input_noise_wrapper, f, rng=_rng_iterator(float(noise), shape))

def _optimize(
    f: Callable[[np.ndarray], float],
    x: bytes,
    /,
    *,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: Optional[float] = None,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> bytes:
    """Helper function for optimizing in a separate process."""
    return np.asarray(_spsa.optimize(
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
    ), dtype=float).tobytes()

async def optimize(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULT_EXECUTOR,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: Optional[float] = None,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    An asynchronous optimizer accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    See `help(spsa.optimizer)` and `help(spsa.amp)` for more details.

    Parameters
    -----------
        executor:
            An executor for SPSA to run in.
            If `None`, uses `asyncio`'s default `ThreadPoolExecutor`.
            Recommended that `ThreadPoolExecutor`s should not be used.
            Use `spsa.aio` instead of threading.
    """
    try:
        x = _type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    return np.array(np.frombuffer(await asyncio.get_running_loop().run_in_executor(executor, partial(
        _optimize,
        f,
        x.tobytes(),
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
    ))), dtype=float)

def _optimize_iterator(
    f: Callable[[np.ndarray], float],
    x: bytes,
    /,
    *,
    conn: Connection,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: Optional[float] = None,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> None:
    """Helper generator for optimizing in a separate process."""
    with conn:
        for variables in _spsa.optimize_iterator(
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

async def optimize_iterator(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    /,
    *,
    executor: Optional[Executor] = DEFAULT_EXECUTOR,
    timeout: float = 0.0,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: Optional[float] = None,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    An asynchronous generator accepting synchronous functions.
    Runs the entire SPSA algorithm in an executor.

    See `help(spsa.optimizer_iterator)` and `help(spsa.amp)` for more details.

    Parameters
    -----------
        executor:
            An executor for SPSA to run in.
            If `None`, uses `asyncio`'s default `ThreadPoolExecutor`.
            Recommended that `ThreadPoolExecutor`s should not be used.
            Use `spsa.aio` instead of threading.

        timeout:
            Timeout for when there are no new variables to generate.
    """
    try:
        x = _type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    parent, child = Pipe()
    executor.submit(partial(
        _optimize_iterator,
        f,
        x.tobytes(),
        conn=child,
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
    ))
    for _ in range(iterations + 1):
        while not child.closed and not parent.poll():
            await asyncio.sleep(timeout)
        if child.closed:
            break
        yield {
            key: (np.frombuffer(value) if isinstance(value, bytes) else value)
            for key, value in parent.recv().items()
        }
