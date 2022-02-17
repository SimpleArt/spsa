"""
Contains asynchronous variants of `spsa.optimize` and `spsa.optimize_iterator`.

Example Uses
-------------
- Large numbers of iterations can be ran concurrently with other asynchronous code.
- Slow function calls can be ran concurrently each iteration. May be combined with multiprocessing.

Parallelizing Calls with Multiprocessing
-----------------------------------------
Calls can be parallelized by using multiprocessing,
but some care needs to be taken when using numpy arrays.
The following provides a general template on how to apply multiprocessing.

NOTE: Use this approach if the calculations are significantly slower than
      the amount of time it takes to share the data to other processes.

Example
--------
import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
import numpy as np
import spsa

def slow_f(data: bytes) -> float:
    '''Main calculations.'''
    x = np.frombuffer(data)
    ...  # Calculations.

async def f(x: np.ndarray, executor: Executor = ProcessPoolExecutor()) -> float:
    '''Run the calculations with an executor.'''
    return await asyncio.get_running_loop().run_in_executor(executor, slow_f, x.tobytes())

# Initial x.
x = ...
# Solution.
x = spsa.aio.optimize(f, x)
"""
import asyncio
from math import isinf, isnan, isqrt
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence
import numpy as np
from ._spsa import ArrayLike, OptimizerVariables

__all__ = ["optimize", "optimize_iterator"]

async def maximize(f: Callable[[np.ndarray], Awaitable[float]], /) -> Callable[[np.ndarray], Awaitable[float]]:
    """
    Turns the function into a maximization function.

    Usage
    ------
        Maximize a function instead of minimizing it:
            x = spsa.aio.optimize(maximize(f), x)
    """
    @wraps(f)
    async def wrapper(x: np.ndarray, /) -> float:
        return -(await f(x))
    return wrapper

async def optimize(
    f: Callable[[np.ndarray], Awaitable[float]],
    x: ArrayLike,
    /,
    *,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: float = 3e-4,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    An asynchronous optimizer accepting asynchronous functions.
    Allows function calls to be done concurrently each iteration.

    See `help(spsa.optimizer)` and `help(spsa.aio)` for more details.
    """
    # Type-check.
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(x, (np.ndarray, Sequence)):
        raise TypeError(f"x must be either a numpy array or a sequence, got {x!r}")
    elif not isinstance(iterations, int):
        raise TypeError(f"iterations must be an integer, got {iterations!r}")
    names = ("lr_decay", "lr_power", "px", "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    for name, value in zip(names, values):
        if not isinstance(value, float):
            raise TypeError(f"{name} must be a float, got {value!r}")
        elif isnan(value):
            raise ValueError(f"{name} must not be nan, got {value!r}")
        elif isinf(value):
            raise ValueError(f"{name} must not be infinite, got {value!r}")
        elif value <= 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")
    names = ("lr_power", "px_power", "momentum", "beta")
    values = (lr_power, px_power, momentum, beta)
    for name, value in zip(names, values):
        if value >= 1:
            raise ValueError(f"{name} must not be greater than 1, got {value!r}")
    # Free up references.
    del names, name, values, value
    # Cast to numpy array.
    x = np.asarray(x, dtype=float)
    # Type-check.
    if x.size == 0:
        raise ValueError("cannot optimize with array of size 0")
    elif np.isnan(x).any():
        raise ValueError(f"x must not contain nan")
    elif np.isinf(x).any():
        raise ValueError(f"x must not contain infinity")
    m1 = 1 - momentum
    m2 = 1 - beta
    # Estimate the gradient and its square.
    b1 = 0
    gx = np.zeros_like(x)
    if adam:
        b2 = 0
        slow_gx = np.zeros_like(x)
        square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = np.random.default_rng().choice((-1.0, 1.0), x.shape)
        dx *= px
        y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        gx += m1 * (df_dx - gx)
        if adam:
            b2 += m2 * (1 - b2)
            slow_gx += m2 * (df_dx - slow_gx)
            square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    # Estimate the learning rate.
    if lr is None:
        lr = 1e-5
        y = await f(x)
        # Increase the learning rate while it is safe to do so.
        dx = 3 * b2 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        for _ in range(3):
            while await f(x - lr * dx) < y:
                await asyncio.sleep(0)
                lr *= 1.4
        del y
    # Initial step size.
    dx = b2 / b1 * gx / np.sqrt(square_gx + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        await asyncio.sleep(0)
        # Estimate the next point.
        x_next = x + lr * dx
        # Compute df/dx in at the next point.
        dx = np.random.default_rng().choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        y1, y2 = await asyncio.gather(f(x_next + dx), f(x_next - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        gx += m1 * (df_dx - gx)
        if adam:
            b2 += m2 * (1 - b2)
            slow_gx += m2 * (df_dx - slow_gx)
            square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
        # Compute the step size.
        dx = b2 / b1 / (1 + lr_decay * i) ** lr_power * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        # Perform line search.
        y1, y2, y3 = await asyncio.gather(f(x), f(x - lr / 3 * dx), f(x - lr * 3 * dx))
        if y1 < y2 < y3:
            lr /= 1.5
        elif y2 < y1 < y3:
            lr /= 1.2
        elif y1 > y3 < y2:
            lr *= 1.4
        # Update the solution.
        x -= lr * dx
    return x

async def optimize_iterator(
    f: Callable[[np.ndarray], Awaitable[float]],
    x: ArrayLike,
    /,
    *,
    adam: bool = True,
    iterations: int = 10_000,
    lr: Optional[float] = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    px: float = 3e-4,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.97,
    beta: float = 0.999,
    epsilon: float = 1e-7,
) -> AsyncIterator[OptimizerVariables]:
    """
    An asynchronous generator accepting asynchronous functions.
    Allows function calls to be done concurrently each iteration.

    See `help(spsa.optimizer_iterator)` and `help(spsa.aio)` for more details.
    """
    # Type-check.
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(x, (np.ndarray, Sequence)):
        raise TypeError(f"x must be either a numpy array or a sequence, got {x!r}")
    elif not isinstance(iterations, int):
        raise TypeError(f"iterations must be an integer, got {iterations!r}")
    names = ("lr_decay", "lr_power", "px", "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    for name, value in zip(names, values):
        if not isinstance(value, float):
            raise TypeError(f"{name} must be a float, got {value!r}")
        elif isnan(value):
            raise ValueError(f"{name} must not be nan, got {value!r}")
        elif isinf(value):
            raise ValueError(f"{name} must not be infinite, got {value!r}")
        elif value <= 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")
    names = ("lr_power", "px_power", "momentum", "beta")
    values = (lr_power, px_power, momentum, beta)
    for name, value in zip(names, values):
        if value >= 1:
            raise ValueError(f"{name} must not be greater than 1, got {value!r}")
    # Free up references.
    del names, name, values, value
    # Cast to numpy array.
    x = np.asarray(x, dtype=float)
    # Type-check.
    if x.size == 0:
        raise ValueError("cannot optimize with array of size 0")
    elif np.isnan(x).any():
        raise ValueError(f"x must not contain nan")
    elif np.isinf(x).any():
        raise ValueError(f"x must not contain infinity")
    m1 = 1 - momentum
    m2 = 1 - beta
    # Estimate the gradient and its square.
    b1 = 0
    gx = np.zeros_like(x)
    if adam:
        b2 = 0
        slow_gx = np.zeros_like(x)
        square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = np.random.default_rng().choice((-1.0, 1.0), x.shape)
        dx *= px
        y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        gx += m1 * (df_dx - gx)
        if adam:
            b2 += m2 * (1 - b2)
            slow_gx += m2 * (df_dx - slow_gx)
            square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    # Estimate the learning rate.
    if lr is None:
        lr = 1e-5
        y = await f(x)
        # Increase the learning rate while it is safe to do so.
        dx = 3 * b2 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        for _ in range(3):
            while await f(x - lr * dx) < y:
                await asyncio.sleep(0)
                lr *= 1.4
    y = await f(x)
    variables = dict(
        x=x,
        y=y,
        lr=lr,
        beta1=b1,
        beta2=b2,
        gradient=gx,
        slow_gradient=slow_gx,
        square_gradient=square_gx,
    )
    yield variables
    x = variables["x"]
    lr = variables["lr"]
    del variables
    # Initial step size.
    dx = b2 / b1 * gx / np.sqrt(square_gx + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        await asyncio.sleep(0)
        # Estimate the next point.
        x_next = x + lr * dx
        # Compute df/dx in at the next point.
        dx = np.random.default_rng().choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        y1, y2 = await asyncio.gather(f(x_next + dx), f(x_next - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        gx += m1 * (df_dx - gx)
        if adam:
            b2 += m2 * (1 - b2)
            slow_gx += m2 * (df_dx - slow_gx)
            square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
        # Compute the step size.
        dx = b2 / b1 / (1 + lr_decay * i) ** lr_power * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        # Perform line search.
        y1 = y
        y2, y3 = await asyncio.gather(f(x - lr / 3 * dx), f(x - lr * 3 * dx))
        if y1 < y2 < y3:
            lr /= 1.5
        elif y2 < y1 < y3:
            lr /= 1.2
        elif y1 > y3 < y2:
            lr *= 1.4
        # Update the solution.
        x -= lr * dx
        y = await f(x)
        del x_next, df_dx
        variables = dict(
            x=x,
            y=y,
            lr=lr,
            beta1=b1,
            beta2=b2,
            gradient=gx,
            slow_gradient=slow_gx,
            square_gradient=square_gx,
        )
        yield variables
        x = variables["x"]
        lr = variables["lr"]
        del variables
