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
import operator
import random

from math import isinf, isnan, isqrt, sqrt
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence, SupportsIndex

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
    px: Optional[float] = None,
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
    elif not isinstance(adam, SupportsIndex):
        raise TypeError(f"adam cannot be interpreted as an integer, got {adam!r}")
    adam = bool(operator.index(adam))
    names = ("lr_decay", "lr_power")
    values = (lr_decay, lr_power)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    if px is not None:
        names = (*names, "px")
        values = (*values, px)
    names = (*names, "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (*values, px_decay, px_power, momentum, beta, epsilon)
    for name, value in zip(names, values):
        if not isinstance(value, (float, int)):
            raise TypeError(f"{name} must be real, got {value!r}")
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
    rng = np.random.default_rng()
    #---------------------------------------------------------#
    # General momentum algorithm:                             #
    #     b(0) = 0                                            #
    #     f(0) = 0                                            #
    #     b(n + 1) = b(n) + (1 - beta) * (1 - b(n))           #
    #     f(n + 1) = f(n) + (1 - beta) * (estimate(n) - f(n)) #
    #     f(n) / b(n) ~ average(estimate(n))                  #
    #---------------------------------------------------------#
    m1 = 1.0 - momentum
    m2 = 1.0 - beta
    # Estimate the noise in f.
    bn = 0.0
    y = 0.0
    noise = 0.0
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        y1, y2 = await asyncio.gather(f(x), f(x))
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y1 - y) + (y2 - y))
        noise += m2 * ((y1 - y2) ** 2 - noise)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(3):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 > 8 * noise / bn:
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px
        y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    # Estimate the learning rate.
    if lr is None:
        lr = 1e-5
        # Increase the learning rate while it is safe to do so.
        dx = 3 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        for _ in range(3):
            while True:
                y1, y2 = await asyncio.gather(f(x), f(x - lr * dx))
                if y1 < y2:
                    break
                lr *= 1.4
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Initial step size.
    dx = gx / b1
    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - lr * dx
        # Compute df/dx in at the next point.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        dx /= np.sqrt(square_gx / b2 + epsilon)
        df = (f(x_next + dx) - f(x_next - dx)) / 2
        y1, y2 = await asyncio.gather(f(x_next + dx), f(x_next - dx))
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
        # Compute the step size.
        dx = gx / (b1 * (1 + lr_decay * i) ** lr_power)
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        # Sample points in parallel.
        y0, y1, y2, y3 = await asyncio.gather(f(x), f(x), f(x - lr / 3 * dx), f(x - lr * 3 * dx))
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y0 - y) + (y1 - y))
        noise += m2 * ((y0 - y1) ** 2 - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if df ** 2 < 2 * noise / bn:
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y1 - 0.25 * sqrt(noise / bn) < min(y2, y3):
            lr /= 1.3
        if y2 - 0.25 * sqrt(noise / bn) < min(y1, y3):
            lr *= 1.3 / 1.4
        if y3 - 0.25 * sqrt(noise / bn) < min(y1, y2):
            lr *= 1.4
        # Set a minimum learning rate.
        lr = max(lr, epsilon / (1 + 0.01 * i) ** 0.5 * (1 + 0.25 * np.linalg.norm(x)))
        # Update the solution.
        x -= lr * dx
        bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
        x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
    return x_avg / bx

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
    px: Optional[float] = None,
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
    elif not isinstance(adam, SupportsIndex):
        raise TypeError(f"adam cannot be interpreted as an integer, got {adam!r}")
    adam = bool(operator.index(adam))
    names = ("lr_decay", "lr_power")
    values = (lr_decay, lr_power)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    if px is not None:
        names = (*names, "px")
        values = (*values, px)
    names = (*names, "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (*values, px_decay, px_power, momentum, beta, epsilon)
    for name, value in zip(names, values):
        if not isinstance(value, (float, int)):
            raise TypeError(f"{name} must be real, got {value!r}")
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
    rng = np.random.default_rng()
    #---------------------------------------------------------#
    # General momentum algorithm:                             #
    #     b(0) = 0                                            #
    #     f(0) = 0                                            #
    #     b(n + 1) = b(n) + (1 - beta) * (1 - b(n))           #
    #     f(n + 1) = f(n) + (1 - beta) * (estimate(n) - f(n)) #
    #     f(n) / b(n) ~ average(estimate(n))                  #
    #---------------------------------------------------------#
    m1 = 1.0 - momentum
    m2 = 1.0 - beta
    # Estimate the noise in f.
    bn = 0.0
    y = 0.0
    noise = 0.0
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        y1, y2 = await asyncio.gather(f(x), f(x))
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y1 - y) + (y2 - y))
        noise += m2 * ((y1 - y2) ** 2 - noise)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(3):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 > 8 * noise / bn:
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px
        y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
        df_dx = (y1 - y2) * 0.5 / dx
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
    # Estimate the learning rate.
    if lr is None:
        lr = 1e-5
        # Increase the learning rate while it is safe to do so.
        dx = 3 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        for _ in range(3):
            while True:
                y1, y2 = await asyncio.gather(f(x), f(x - lr * dx))
                if y1 < y2:
                    break
                lr *= 1.4
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Generate initial iteration.
    variables = dict(
        x=x_avg,
        y=y,
        lr=lr,
        beta_x=bx,
        beta_noise=bn,
        beta1=b1,
        beta2=b2,
        noise=noise,
        gradient=gx,
        slow_gradient=slow_gx,
        square_gradient=square_gx,
    )
    yield variables
    del variables
    # Initial step size.
    dx = gx / b1
    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - lr * dx
        # Compute df/dx in at the next point.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        dx /= np.sqrt(square_gx / b2 + epsilon)
        y1, y2 = await asyncio.gather(f(x_next + dx), f(x_next - dx))
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update the gradients.
        b1 += m1 * (1 - b1)
        b2 += m2 * (1 - b2)
        gx += m1 * (df_dx - gx)
        slow_gx += m2 * (df_dx - slow_gx)
        square_gx += m2 * ((slow_gx / b2) ** 2 - square_gx)
        # Compute the step size.
        dx = gx / (b1 * (1 + lr_decay * i) ** lr_power)
        if adam:
            dx /= np.sqrt(square_gx / b2 + epsilon)
        # Sample points in parallel.
        y0, y1, y2, y3 = await asyncio.gather(f(x), f(x), f(x - lr / 3 * dx), f(x - lr * 3 * dx))
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y0 - y) + (y1 - y))
        noise += m2 * ((y0 - y1) ** 2 - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if df ** 2 < 2 * noise / bn:
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y1 - 0.25 * sqrt(noise / bn) < min(y2, y3):
            lr /= 1.3
        if y2 - 0.25 * sqrt(noise / bn) < min(y1, y3):
            lr *= 1.3 / 1.4
        if y3 - 0.25 * sqrt(noise / bn) < min(y1, y2):
            lr *= 1.4
        # Set a minimum learning rate.
        lr = max(lr, epsilon / (1 + 0.01 * i) ** 0.5 * (1 + 0.25 * np.linalg.norm(x)))
        # Update the solution.
        x -= lr * dx
        bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
        x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
        # Generate the variables for the next iteration.
        variables = dict(
            x=x_avg,
            y=y,
            lr=lr,
            beta_x=bx,
            beta_noise=bn,
            beta1=b1,
            beta2=b2,
            noise=noise,
            gradient=gx,
            slow_gradient=slow_gx,
            square_gradient=square_gx,
        )
        yield variables
        del variables
