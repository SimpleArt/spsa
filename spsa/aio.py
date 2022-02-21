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

from ._spsa import ArrayLike, OptimizerVariables, _type_check

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
    try:
        x = _type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
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
    # Track the best (x, y).
    y_min = y
    x_min = x.copy()
    # Track how many times the solution fails to improve.
    consecutive_fails = 0
    improvement_fails = 0
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
        consecutive_fails += 1
        # Track the best (x, y).
        if y / bn < y_min:
            y_min = y / bn
            x_min = x_avg / bx
            consecutive_fails = 0
        if consecutive_fails < 100 * improvement_fails:
            continue
        # Reset variables if diverging.
        consecutive_fails = 0
        improvement_fails += 1
        x = x_min
        bx = mx * (1 - mx)
        x_avg = bx * x
        noise *= m2 * (1 - m2) / bn
        y = m2 * (1 - m2) * y_min
        bn = m2 * (1 - m2)
        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx
        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2
        b2 = m2 * (1 - m2)
        lr /= 16
    return x_min

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
    try:
        x = _type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
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
    # Track the best (x, y).
    y_min = y
    x_min = x.copy()
    # Track how many times the solution fails to improve.
    consecutive_fails = 0
    improvement_fails = 0
    # Generate initial iteration.
    variables = dict(
        x_min=x_min,
        y_min=y_min,
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
        consecutive_fails += 1
        # Track the best (x, y).
        if y / bn < y_min:
            y_min = y / bn
            x_min = x_avg / bx
            consecutive_fails = 0
        # Generate the variables for the next iteration.
        variables = dict(
            x_min=x_min,
            y_min=y_min,
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
        if consecutive_fails < 100 * improvement_fails:
            continue
        # Reset variables if diverging.
        consecutive_fails = 0
        improvement_fails += 1
        x = x_min
        bx = mx * (1 - mx)
        x_avg = bx * x
        noise *= m2 * (1 - m2) / bn
        y = m2 * (1 - m2) * y_min
        bn = m2 * (1 - m2)
        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx
        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2
        b2 = m2 * (1 - m2)
        lr /= 16
