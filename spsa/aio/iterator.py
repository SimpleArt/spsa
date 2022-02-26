"""
Contains a collection of iterator variants of `spsa.aio`.

Each iteration, a `dict` of variables is generated,
allowing the iterations to be logged or custom
termination algorithms to be used.
"""
import asyncio
import operator
from math import isqrt, sqrt
from typing import AsyncIterator, Awaitable, Callable, Optional

import numpy as np

import spsa._defaults as DEFAULTS
from spsa._utils import ArrayLike, OptimizerVariables, immutable_view, type_check

__all__ = ["maximize", "minimize"]

async def maximize(
    f: Callable[[np.ndarray], Awaitable[float]],
    x: ArrayLike,
    /,
    *,
    adam: bool = DEFAULTS.adam,
    iterations: int = DEFAULTS.iterations,
    lr: Optional[float] = DEFAULTS.lr,
    lr_decay: float = DEFAULTS.lr_decay,
    lr_power: float = DEFAULTS.lr_power,
    px: Optional[float] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> AsyncIterator[OptimizerVariables]:
    """
    An asynchronous generator accepting asynchronous functions.
    Allows function calls to be done concurrently each iteration.

    See `help(spsa.aio.iterator.minimize)` for more details.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    adam = bool(operator.index(adam))
    iterations = operator.index(iterations)
    if lr is not None:
        lr = float(lr)
    lr_decay = float(lr_decay)
    lr_power = float(lr_power)
    if px is not None:
        px = float(px)
    px_decay = float(px_decay)
    px_power = float(px_power)
    momentum = float(momentum)
    beta = float(beta)
    epsilon = float(epsilon)
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
                if y1 > y2:
                    break
                lr *= 1.4
                await asyncio.sleep(0)
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Track the best (x, y).
    y_best = y / bn
    x_best = x.copy()
    # Track how many times the solution fails to improve.
    consecutive_fails = 0
    improvement_fails = 0
    # Generate initial iteration.
    yield dict(
        x_best=immutable_view(x_best),
        y_best=y_best,
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
        x_next = x + lr * dx
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
        y3, y4, y5, y6 = await asyncio.gather(f(x), f(x + lr * 0.5 * dx), f(x + lr / sqrt(m1) * dx), f(x))
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
        if y3 + 0.25 * sqrt(noise / bn) > max(y4, y5):
            lr /= 1.3
        if y4 + 0.25 * sqrt(noise / bn) > max(y3, y5):
            lr *= 1.3 / 1.4
        if y5 + 0.25 * sqrt(noise / bn) > max(y3, y4):
            lr *= 1.4
        # Set a minimum learning rate.
        lr = max(lr, epsilon / (1 + 0.01 * i) ** 0.5 * (1 + 0.25 * np.linalg.norm(x)))
        # Update the solution.
        x += lr * dx
        bx += mx / (1 + 0.01 * i) ** 0.303 * (1 - bx)
        x_avg += mx / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
        consecutive_fails += 1
        # Track the best (x, y).
        if y / bn > y_best:
            y_best = y / bn
            x_best = x_avg / bx
            consecutive_fails = 0
        # Generate the variables for the next iteration.
        yield dict(
            x_best=immutable_view(x_best),
            y_best=y_best,
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
        x = x_best
        bx = mx * (1 - mx)
        x_avg = bx * x
        noise *= m2 * (1 - m2) / bn
        y = m2 * (1 - m2) * y_best
        bn = m2 * (1 - m2)
        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx
        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2
        b2 = m2 * (1 - m2)
        lr /= 16 * improvement_fails

async def minimize(
    f: Callable[[np.ndarray], Awaitable[float]],
    x: ArrayLike,
    /,
    *,
    adam: bool = DEFAULTS.adam,
    iterations: int = DEFAULTS.iterations,
    lr: Optional[float] = DEFAULTS.lr,
    lr_decay: float = DEFAULTS.lr_decay,
    lr_power: float = DEFAULTS.lr_power,
    px: Optional[float] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> AsyncIterator[OptimizerVariables]:
    """
    An asynchronous generator accepting asynchronous functions.
    Allows function calls to be done concurrently each iteration.

    See `help(spsa.iterator.minimize)` and `help(spsa.aio.minimize)` for more details.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, momentum, beta, epsilon)
    except (TypeError, ValueError) as e:
        raise e.with_traceback(None)
    adam = bool(operator.index(adam))
    iterations = operator.index(iterations)
    if lr is not None:
        lr = float(lr)
    lr_decay = float(lr_decay)
    lr_power = float(lr_power)
    if px is not None:
        px = float(px)
    px_decay = float(px_decay)
    px_power = float(px_power)
    momentum = float(momentum)
    beta = float(beta)
    epsilon = float(epsilon)
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
    y_best = y / bn
    x_best = x.copy()
    # Track how many times the solution fails to improve.
    consecutive_fails = 0
    improvement_fails = 0
    # Generate initial iteration.
    yield dict(
        x_best=immutable_view(x_best),
        y_best=y_best,
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
        y3, y4, y5, y6 = await asyncio.gather(f(x), f(x - lr * 0.5 * dx), f(x - lr / sqrt(m1) * dx), f(x))
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
        if y / bn < y_best:
            y_best = y / bn
            x_best = x_avg / bx
            consecutive_fails = 0
        # Generate the variables for the next iteration.
        yield dict(
            x_best=immutable_view(x_best),
            y_best=y_best,
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
        x = x_best
        bx = mx * (1 - mx)
        x_avg = bx * x
        noise *= m2 * (1 - m2) / bn
        y = m2 * (1 - m2) * y_best
        bn = m2 * (1 - m2)
        b1 = m1 * (1 - m1)
        gx = b1 / b2 * slow_gx
        slow_gx *= m2 * (1 - m2) / b2
        square_gx *= m2 * (1 - m2) / b2
        b2 = m2 * (1 - m2)
        lr /= 16 * improvement_fails
