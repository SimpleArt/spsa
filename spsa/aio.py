"""
Asynchronous IO
----------------

Contains asynchronous variants of `spsa.optimize` and `spsa.optimize_iterator`
for calling an asynchronous function concurrently each iteration.

See also:
    spsa.amp - Asynchronous Multiprocessing:
        Runs the whole spsa algorithm in a separate process for synchronous functions.
        Unlike spsa.aio, does not concurrently call functions each iteration. This is
        more appropriate if the objective function cannot be ran concurrently or if
        sharing numpy arrays between processes is too expensive.

Example Uses
-------------
- Large numbers of iterations can be ran concurrently with other asynchronous code.
- Slow function calls can be ran concurrently each iteration by combining it with multiprocessing.

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
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence, SupportsFloat, SupportsIndex, Tuple

import numpy as np

from ._spsa import ArrayLike, OptimizerVariables, _type_check, immutable_view

__all__ = ["maximize", "optimize", "optimize_iterator", "with_input_noise"]

async def maximize(f: Callable[[np.ndarray], Awaitable[float]], /) -> Callable[[np.ndarray], Awaitable[float]]:
    """
    Turns the function into a maximization function.

    Usage
    ------
        Maximize a function instead of minimizing it:
            x = spsa.aio.optimize(maximize(f), x)
    """
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    @wraps(f)
    async def wrapper(x: np.ndarray, /) -> float:
        return -(await f(x))
    return wrapper

def with_input_noise(f: Callable[[np.ndarray], Awaitable[float]], /, noise: float) -> Callable[[np.ndarray], Awaitable[float]]:
    """Adds noise to the input before calling."""
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(noise, SupportsFloat):
        raise TypeError(f"noise must be real, got {noise!r}")
    noise = float(noise)
    rng = np.random.default_rng()
    def rng_iterator(shape: Tuple[int, ...]) -> Iterator[np.ndarray]:
        while True:
            random_noise = rng.uniform(-noise, noise, shape)
            yield random_noise
            yield random_noise
    rng_iter: Optional[Iterator[float]] = None
    async def wrapper(x: np.ndarray) -> float:
        nonlocal rng_iter
        if rng_iter is None:
            rng_iter = rng_iterator(x.shape)
        return (await f(x + dx) + await f(x - dx)) / 2
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
    for _ in range(isqrt(isqrt(x.size + 100) + 100)):
        y1, y2 = await asyncio.gather(f(x), f(x))
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y1 - y) + (y2 - y))
        noise += m2 * ((y1 - y2) ** 2 - noise)
        await asyncio.sleep(0)
    temp = await f(x)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(isqrt(isqrt(x.size + 100) + 100)):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 > 8 * noise / bn or px > 1e-8 + np.linalg.norm(x):
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
                await asyncio.sleep(0)
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
                await asyncio.sleep(0)
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
            await asyncio.sleep(0)
    temp = await f(x)
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 100) + 100)):
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
        await asyncio.sleep(0)
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
                await asyncio.sleep(0)
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Track the best (x, y).
    y_min = y / bn
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
        # Sample points concurrently.
        y3, y4, y5, y6 = await asyncio.gather(f(x), f(x - lr * m1 * dx), f(x - lr / m1 * dx), f(x))
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += m2 * (y3 - y)
        noise += m2 * ((y3 - y6) ** 2 + 1e-64 * (abs(y3) + abs(y6)) - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if (y1 - y2) ** 2 < 8 * noise / bn and px < 1e-8 + np.linalg.norm(x):
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y3 - 0.25 * sqrt(noise / bn) < min(y4, y5):
            lr /= 1.3
        if y4 - 0.25 * sqrt(noise / bn) < min(y3, y5):
            lr *= 1.3 / 1.4
        if y5 - 0.25 * sqrt(noise / bn) < min(y3, y4):
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
        await asyncio.sleep(0)
        if consecutive_fails < 128 * (improvement_fails + isqrt(x.size + 100)):
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
        lr /= 16 * improvement_fails
    return x_min if y_min - 0.25 * sqrt(noise / bn) < min(*(await asyncio.gather(f(x), f(x)))) else x

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
    for _ in range(isqrt(x.size + 100)):
        y1, y2 = await asyncio.gather(f(x), f(x))
        bn += m2 * (1 - bn)
        y += 0.5 * m2 * ((y1 - y) + (y2 - y))
        noise += m2 * ((y1 - y2) ** 2 - noise)
        await asyncio.sleep(0)
    temp = await f(x)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(isqrt(x.size + 100)):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 > 8 * noise / bn or px > 1e-8 + np.linalg.norm(x):
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
                await asyncio.sleep(0)
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1, y2 = await asyncio.gather(f(x), f(x))
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1, y2 = await asyncio.gather(f(x + dx), f(x - dx))
                if (y1 - y2) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
                await asyncio.sleep(0)
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
            await asyncio.sleep(0)
    temp = await f(x)
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for _ in range(isqrt(x.size + 100)):
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
        await asyncio.sleep(0)
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
                await asyncio.sleep(0)
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Track the best (x, y).
    y_min = y / bn
    x_min = x.copy()
    # Track how many times the solution fails to improve.
    consecutive_fails = 0
    improvement_fails = 0
    # Generate initial iteration.
    yield dict(
        x_min=immutable_view(x_min),
        y_min=y_min,
        x=immutable_view(x_avg),
        y=y,
        lr=lr,
        beta_x=bx,
        beta_noise=bn,
        beta1=b1,
        beta2=b2,
        noise=noise,
        gradient=immutable_view(gx),
        slow_gradient=immutable_view(slow_gx),
        square_gradient=immutable_view(square_gx),
    )
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
        # Sample points concurrently.
        y3, y4, y5, y6 = await asyncio.gather(f(x), f(x - lr * m1 * dx), f(x - lr / m1 * dx), f(x))
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += m2 * (y3 - y)
        noise += m2 * ((y3 - y6) ** 2 + 1e-64 * (abs(y3) + abs(y6)) - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if (y1 - y2) ** 2 < 8 * noise / bn and px < 1e-8 + np.linalg.norm(x):
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y3 - 0.25 * sqrt(noise / bn) < min(y4, y5):
            lr /= 1.3
        if y4 - 0.25 * sqrt(noise / bn) < min(y3, y5):
            lr *= 1.3 / 1.4
        if y5 - 0.25 * sqrt(noise / bn) < min(y3, y4):
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
        yield dict(
            x_min=immutable_view(x_min),
            y_min=y_min,
            x=immutable_view(x_avg),
            y=y,
            lr=lr,
            beta_x=bx,
            beta_noise=bn,
            beta1=b1,
            beta2=b2,
            noise=noise,
            gradient=immutable_view(gx),
            slow_gradient=immutable_view(slow_gx),
            square_gradient=immutable_view(square_gx),
        )
        await asyncio.sleep(0)
        if consecutive_fails < 128 * (improvement_fails + isqrt(x.size + 100)):
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
        lr /= 16 * improvement_fails
