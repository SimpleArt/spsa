"""
Contains a collection of iterator variants of `spsa`.

Each iteration, a `dict` of variables is generated,
allowing the iterations to be logged or custom
termination algorithms to be used.
"""
import operator
import random
from math import isqrt, sqrt
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

import spsa._defaults as DEFAULTS
from spsa._utils import ArrayLike, OptimizerVariables, type_check, immutable_view

__all__ = ["maximize", "minimize"]

def maximize(
    f: Callable[[np.ndarray], float],
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
) -> Iterator[OptimizerVariables]:
    """
    A generator which yields a dict of variables each iteration,
    allowing the iterations to be logged or custom termination
    algorithms to be used.

    See `help(spsa.iterator.minimize)` for more details.
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
        temp = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (temp - y)
        noise += m2 * ((temp - f(x)) ** 2 - noise)
    # Estimate the perturbation size that should be used.
    temp = f(x)
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(isqrt(x.size + 100)):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1 = f(x)
                y2 = f(x)
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1 = f(x + dx)
                y2 = f(x - dx)
                if (y1 - y2) ** 2 > 8 * noise / bn or px > 1e-8 + np.linalg.norm(x):
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1 = f(x)
                y2 = f(x)
                bn += m2 * (1 - bn)
                y += 0.5 * m2 * ((y1 - y) + (y2 - y))
                noise += m2 * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1 = f(x + dx)
                y2 = f(x - dx)
                if (y1 - y2) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    temp = f(x)
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
        df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx
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
            while f(x - lr * dx) > f(x):
                lr *= 1.4
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
        y1 = f(x_next + dx)
        y2 = f(x_next - dx)
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
        # Sample points.
        y3 = f(x)
        y4 = f(x + lr * 0.5 * dx)
        y5 = f(x + lr / sqrt(m1) * dx)
        y6 = f(x)
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
        lr /= 64 * improvement_fails

def minimize(
    f: Callable[[np.ndarray], float],
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
    momentum: Sequence[float] = (0.7, 0.9, 0.97, 0.99, 0.997, 0.999, 0.9997),
    beta: float = 0.999,
    epsilon: float = DEFAULTS.epsilon,
) -> Iterator[OptimizerVariables]:
    """
    A generator which yields a dict of variables each iteration,
    allowing the iterations to be logged or custom termination
    algorithms to be used.

    See `help(spsa.minimize)` for more details.

    Yields
    -------
        optimizer_variables:
            A dictionary containing optimizer variables.

            NOTE: Most variables come with a corresponding beta variable which should be used.

            NOTE: x_best, x, gradient, slow_gradient, and square_gradient are immutable numpy arrays.
                  Use x_best.copy(), x / beta_x, etc. instead of mutating them.

            x_best:
            y_best:
                The best seen estimated minimum of f.

            x:
            y:
                The current estimated minimum of f.

            lr:
                The current learning rate (not including decay).

            beta_x
            beta_noise:
            beta1:
            beta2:
                Used for the formulas
                    x = x / beta_x
                    y = y / beta_noise
                    noise = sqrt(noise / beta_noise)
                    gradient = gradient / beta1
                    slow_gradient = slow_gradient / beta2
                    square_gradient = square_gradient / beta2
                to get unbiased estimates of each variable.

                On their own, the estimates are closer to 0 than they should be
                and need to be divided by their respective betas for correction.

            noise:
                An estimate for how much noise is in f(x).
                Used for SPSA.

            gradient:
                The estimated gradient of f at x.

            slow_gradient:
                The slower estimate of the gradient of f at x.
                Biased more towards previous iterations.
                Used for the square_gradient.

            square_gradient:
                An estimate for the component-wise square of the gradient of f at x.
                Used for the Adam method and perturbation size rescaling.
    """
    try:
        x = type_check(f, x, adam, iterations, lr, lr_decay, lr_power, px, px_decay, px_power, 0.9, 0.9, epsilon)
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
    momentum_beta = 1 - np.array(sorted(momentum), dtype=float).reshape(-1)
    square_beta = 1 - float(beta)
    epsilon = float(epsilon)
    rng = np.random.default_rng()
    x_beta = 0.003
    y_beta = 0.003
    noise_beta = 0.003
    #---------------------------------------------------------#
    # General momentum algorithm:                             #
    #     bias(0) = 0                                         #
    #     f(0) = 0                                            #
    #     bias(n + 1) = bias(n) + (1 - beta) * (1 - bias(n))  #
    #     f(n + 1) = f(n) + (1 - beta) * (estimate(n) - f(n)) #
    #     average(estimate(n)) ~ f(n) / bias(n)               #
    #---------------------------------------------------------#
    # Estimate the noise in f.
    y_bias = 0.0
    noise_bias = 0.0
    y = 0.0
    noise = 0.0
    temp = f(x)
    for _ in range(isqrt(x.size + 100)):
        y1 = f(x)
        y2 = f(x)
        y_bias += y_beta * (1 - y_bias)
        noise_bias += noise_beta * (1 - noise_bias)
        y += y_beta * (0.5 * (y1 + y2) - y)
        noise += noise_beta * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(isqrt(x.size + 100)):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                y1 = f(x)
                y2 = f(x)
                y_bias += y_beta * (1 - y_bias)
                noise_bias += noise_beta * (1 - noise_bias)
                y += y_beta * (0.5 * (y1 + y2) - y)
                noise += noise_beta * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                y1 = f(x + dx)
                y2 = f(x - dx)
                if (y1 - y2) ** 2 > 8 * noise / noise_bias or px > 1e-8 + np.linalg.norm(x):
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                y1 = f(x)
                y2 = f(x)
                y_bias += y_beta * (1 - y_bias)
                noise_bias += noise_beta * (1 - noise_bias)
                y += y_beta * (0.5 * (y1 + y2) - y)
                noise += noise_beta * ((y1 - y2) ** 2 + 1e-64 * (abs(y1) + abs(y2)) - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                y1 = f(x + dx)
                y2 = f(x - dx)
                if (y1 - y2) ** 2 < 8 * noise / noise_bias:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    temp = f(x)
    # Estimate the gradient and its square.
    momentum_bias = np.zeros_like(momentum_beta)
    square_bias = 0.0
    momentum_gradient = np.zeros((*x.shape, len(momentum_beta)))
    slow_gradient = np.zeros_like(x)
    square_gradient = np.zeros_like(x)
    for _ in range(isqrt(x.size + 100)):
        # Compute df/dx in random directions.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px
        df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx
        # Update the gradients.
        momentum_bias += momentum_beta * (1 - momentum_bias)
        square_bias += square_beta * (1 - square_bias)
        momentum_gradient += momentum_beta * np.moveaxis((df_dx - np.moveaxis(momentum_gradient, -1, 0)), 0, -1)
        slow_gradient += square_beta * (df_dx - slow_gradient)
        square_gradient += square_beta * ((slow_gradient / square_bias) ** 2 - square_gradient)
    # Estimate the learning rate.
    if lr is None:
        lr = np.full_like(momentum_bias, 1e-5)
        # Increase the learning rate while it is safe to do so.
        dx = momentum_gradient / momentum_bias
        if adam:
            dx = np.moveaxis(dx, -1, 0)
            dx /= np.sqrt(square_gradient / square_bias + epsilon)
            dx = np.moveaxis(dx, 0, -1)
        for _ in range(5):
            while True:
                i = random.randrange(len(momentum_beta))
                y0 = f(x)
                dx[..., i] /= 2
                y1 = f(x - (lr * dx).sum(axis=-1))
                dx[..., i] *= 2 / sqrt(momentum_beta[-1])
                y2 = f(x - (lr * dx).sum(axis=-1))
                dx[..., i] *= sqrt(momentum_beta[-1])
                y3 = f(x)
                # Estimate the noise in f.
                y_bias += y_beta * (1 - y_bias)
                noise_bias += noise_beta * (1 - noise_bias)
                y += y_beta * (0.5 * (y0 + y3) - y)
                noise += noise_beta * ((y0 - y3) ** 2 + 1e-64 * (abs(y0) + abs(y3)) - noise)
                # Adjust the learning rate towards learning rates which give good results.
                if y2 - 0.25 * sqrt(noise / noise_bias) > min(y0, y1):
                    break
                lr[..., i] *= 1.4
    # Track the average value of x.
    x_bias = x_beta
    x_avg = x_beta * x
    # Track the best (x, y).
    y_best = y / y_bias
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
        lr=immutable_view(lr),
        x_bias=x_bias,
        noise_bias=noise_bias,
        momentum_bias=immutable_view(momentum_bias),
        square_bias=square_bias,
        noise=noise,
        gradient=immutable_view(momentum_gradient),
        slow_gradient=immutable_view(slow_gradient),
        square_gradient=immutable_view(square_gradient),
    )
    # Initial step size.
    dx = momentum_gradient @ (lr / momentum_bias)
    if adam:
        dx = np.moveaxis(dx, -1, 0)
        dx /= np.sqrt(square_gradient / square_bias + epsilon)
        dx = np.moveaxis(dx, 0, -1)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - dx
        # Compute df/dx in at the next point.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        dx /= np.sqrt(square_gradient / square_bias + epsilon)
        y1 = f(x_next + dx)
        y2 = f(x_next - dx)
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if df ** 2 < 2 * noise / noise_beta and px < 1e-8 + np.linalg.norm(x):
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Update the gradients.
        momentum_bias += momentum_beta * (1 - momentum_bias)
        square_bias += square_beta * (1 - square_bias)
        momentum_gradient += momentum_beta * np.moveaxis((df_dx - np.moveaxis(momentum_gradient, -1, 0)), 0, -1)
        slow_gradient += square_beta * (df_dx - slow_gradient)
        square_gradient += square_beta * ((slow_gradient / square_bias) ** 2 - square_gradient)
        # Compute the step size.
        dx = momentum_gradient / momentum_bias
        if adam:
            dx = np.moveaxis(dx, -1, 0)
            dx /= np.sqrt(square_gradient / square_bias + epsilon)
            dx = np.moveaxis(dx, 0, -1)
        # Update a random momentum.
        j = random.randrange(len(lr))
        # Sample points.
        y0 = f(x)
        dx[..., j] /= 2
        y1 = f(x - (lr * dx).sum(axis=-1))
        dx[..., j] *= 2 / sqrt(momentum_beta[-1])
        y2 = f(x - (lr * dx).sum(axis=-1))
        dx[..., j] *= sqrt(momentum_beta[-1])
        y3 = f(x)
        # Estimate the noise in f.
        y_bias += y_beta * (1 - y_bias)
        noise_bias += noise_beta * (1 - noise_bias)
        y += y_beta * (0.5 * (y0 + y3) - y)
        noise += noise_beta * ((y0 - y3) ** 2 + 1e-64 * (abs(y0) + abs(y3)) - noise)
        # Perform line search.
        # Adjust the learning rate towards learning rates which give good results.
        if y0 - 0.25 * sqrt(noise / noise_bias) < min(y1, y2):
            lr[..., j] /= 1.3
        if y1 - 0.25 * sqrt(noise / noise_bias) < min(y0, y2):
            lr[..., j] *= 1.3 / 1.4
        if y2 - 0.25 * sqrt(noise / noise_bias) < min(y0, y1):
            lr[..., j] *= 1.4
        # Set a minimum learning rate.
        lr[..., j] = max(lr[..., j], (1 + 0.01 * i) ** -0.5 * (1 + 0.25 * np.linalg.norm(x)) * epsilon)
        # Update the solution.
        dx = momentum_gradient @ ((1 + lr_decay * i) ** -lr_power * lr / momentum_bias)
        if adam:
            dx /= np.sqrt(square_gradient / square_bias + epsilon)
        x -= dx
        x_bias += x_beta / (1 + 0.01 * i) ** 0.303 * (1 - x_bias)
        x_avg += x_beta / (1 + 0.01 * i) ** 0.303 * (x - x_avg)
        consecutive_fails += 1
        # Track the best (x, y).
        if y / y_bias < y_best:
            y_best = y / y_beta
            x_best = x_avg / x_beta
            consecutive_fails = 0
        # Generate the variables for the next iteration.
        yield dict(
            x_best=immutable_view(x_best),
            y_best=y_best,
            x=immutable_view(x_avg),
            y=y,
            lr=immutable_view(lr),
            x_bias=x_bias,
            y_bias=y_bias,
            noise_bias=noise_bias,
            momentum_bias=immutable_view(momentum_bias),
            square_bias=square_bias,
            noise=noise,
            gradient=immutable_view(momentum_gradient),
            slow_gradient=immutable_view(slow_gradient),
            square_gradient=immutable_view(square_gradient),
        )
        if consecutive_fails < 128 * (improvement_fails + isqrt(x.size + 100)):
            continue
        # Reset variables if diverging.
        consecutive_fails = 0
        improvement_fails += 1
        x = x_best
        x_bias = x_beta * (1 - x_bias)
        x_avg = x_beta * x
        noise *= noise_beta * (1 - noise_beta) / noise_bias
        noise_bias = noise_beta * (1 - noise_beta)
        y_bias = y_beta * (1 - y_beta)
        y = y_bias * y_best
        momentum_bias = momentum_beta * (1 - momentum_beta)
        square_bias = square_beta * (1 - square_beta)
        momentum_gradient = momentum_beta * (1 - momentum_beta) / momentum_bias * momentum_gradient
        slow_gradient = square_beta * (1 - square_beta) / square_bias * slow_gradient
        square_gradient = square_beta * (1 - square_beta) / square_bias * square_gradient
        lr /= 64 * improvement_fails
