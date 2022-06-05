import operator
from functools import partial, wraps
from math import isqrt, sqrt
from typing import Callable, Iterator, Optional, SupportsFloat, Tuple, Type, Union

import numpy as np

import spsa._defaults as DEFAULTS
import spsa.random.iterator
from spsa._utils import ArrayLike, immutable_view, type_check

__all__ = ["maximize", "minimize", "with_input_noise"]

def _with_input_noise_wrapper(rng: Iterator[np.ndarray], f: Callable[[np.ndarray], float], x: np.ndarray) -> float:
    """Helper function for adding noise to the input before calling."""
    dx = next(rng)
    return (f(x + dx) + f(x - dx)) / 2

def with_input_noise(f: Callable[[np.ndarray], float], /, shape: Tuple[int, ...], noise: float) -> Callable[[np.ndarray], float]:
    """Adds noise to the input before calling."""
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(noise, SupportsFloat):
        raise TypeError(f"noise must be real, got {noise!r}")
    return partial(_with_input_noise_wrapper, spsa.random.iterator.noise(float(noise), shape), f)

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
    px: Union[float, Type[int]] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> np.ndarray:
    """
    Implementation of the SPSA optimization algorithm for maximizing an objective function.

    See `help(spsa.minimize)` for documentation.
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
    if px is int:
        x_temp = np.empty_like(x, dtype=int)
    elif px is not None:
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
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for i in range(isqrt(x.size + 100)):
        # Compute df/dx in random directions.
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            df_dx = (f(np.rint(x + dx, casting="unsafe", out=x_temp)) - f(np.rint(x - dx, casting="unsafe", out=x_temp))) * 0.5 / dx
        else:
            dx = rng.choice((-1.0, 1.0), x.shape) / (1 + i)
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
    momentum_fails = 0
    consecutive_fails = 0
    improvement_fails = 0
    # Initial step size.
    dx = gx / b1
    if adam:
        dx /= np.sqrt(square_gx / b2 + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x + lr * dx
        # Compute df/dx in at the next point.
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            y1 = f(np.rint(x_next + dx, casting="unsafe", out=x_temp))
            y2 = f(np.rint(x_next - dx, casting="unsafe", out=x_temp))
        else:
            dx = (lr / m1 * px / (1 + px_decay * i) ** px_power) * np.linalg.norm(dx)
            if adam:
                dx /= np.sqrt(square_gx / b2 + epsilon)
            dx *= rng.choice((-1.0, 1.0), x.shape)
            y1 = f(x_next + dx)
            y2 = f(x_next - dx)
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update the momentum.
        if (df_dx.flatten() / np.linalg.norm(df_dx)) @ (gx.flatten() / np.linalg.norm(gx)) < 0.5 / (1 + 0.1 * momentum_fails) ** 0.3 - 1:
            momentum_fails += 1
            m1 = (1.0 - momentum) / sqrt(1 + 0.1 * momentum_fails)
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
    if px is int:
        x_best = np.rint(x_best).astype(int)
        x = np.rint(x, casting="unsafe", out=x_temp)
    return x_best if y_best + 0.25 * sqrt(noise / bn) > max(f(x), f(x)) else x

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
    px: Union[float, Type[int]] = DEFAULTS.px,
    px_decay: float = DEFAULTS.px_decay,
    px_power: float = DEFAULTS.px_power,
    momentum: float = DEFAULTS.momentum,
    beta: float = DEFAULTS.beta,
    epsilon: float = DEFAULTS.epsilon,
) -> np.ndarray:
    """
    Implementation of the SPSA optimization algorithm for minimizing an objective function.

    See `help(spsa)` for additional information.

    Defining Objective Functions
    -----------------------------
    Defining your objective function f appropriately can significantly change how well SPSA performs.
    Here are some tips and tricks on how to define good objective functions.

        Detectable Changes:
            SPSA requires changes in inputs to lead to changes in outputs.
            Otherwise SPSA will be unable to determine how inputs should change.

            This isn't possible for all problems. Some may require other algorithms
            e.g. Bayesian optimization or genetic algorithms. Some may need specialized
            algorithms for the specific situation.

            If the objective function is noisy then SPSA will increase its change in input
            until a significant change in output is detected i.e. the result isn't just noise.

            NOTE: SPSA assumes that the result y = f(x) has at least 1e-64 * y noise, so if
                  there is no change in y, then the change in x automatically increases to try
                  to search for significant changes. In some cases, this may be sufficient.

        Basin-Hopping Input Noise:
            Some functions have many local minima, causing SPSA and similar methods to run
            into bad solutions. This can, to some extent, be countered using `spsa.with_input_noise`.

                def f(x):
                    return ...

                x = spsa.optimize(spsa.with_input_noise(f, shape, noise=0.5), ...)

            In this way, SPSA will explore neighboring inputs instead of getting stuck in a "basin".
            If the noise is sufficiently high, and there is a general trend in the direction of the
            basins towards the best basin, then this will converge to the locally best basin.

            NOTE: Not all functions perform better with input noise, some may perform worse.

        Stochastic vs Deterministic Functions:
            SPSA does not require deterministic functions.

            If it is more efficient to use a stochastic (random) function e.g. by using a random
            sample in a dataset each time instead of the whole dataset, then do that instead.

            High-precision accuracy is not possible in this case, but you will get more out of
            your calculations this way.

            Running a few iterations afterwards with a deterministic variant, if reasonable, can
            be used to get high-precision accuracy if needed.

        Controllable Stochastic Functions:
            If the "noise" in the function can be controlled by a second parameter, then the
            performance of SPSA can be significantly improved by making every pair of calls
            use the same "noise parameter". This can be implemented as follows:

                from functools import partial

                def rng_iterator():
                    while True:
                        random_value = ...  # A sample from a dataset for example.
                        yield random_value
                        yield random_value

                def f(x, rng):
                    random_value = next(rng)
                    return ...

                x = spsa.optimize(partial(f, rng=rng_iterator()), ...)

        Twice Differentiable:
            It is not required, but it would be good if the objective function is twice
            differentiable. SPSA assumes the objective function is relatively smooth in
            order to converge well. Otherwise it will struggle to move around points
            where the function is not very smooth.

            For stochastic functions, it is expected that the expected value is smooth.

    Parameters
    -----------
        f:
            The function being optimized. Called as `f(array) -> float`.

        x:
            The initial point used. This value is edited and returned.

        adam:
            True to use Adam, False to not use it.

        iterations:
            The number of iterations ran.

        lr:
        lr_decay:
        lr_power:
            If no learning rate is given, then a crude estimate is found using line search.

            The learning rate controls the speed of convergence.

                lr = lr_start / (1 + lr_decay * iteration) ** lr_power
                x -= lr * gradient_estimate

            Furthermore, the learning rate is automatically tuned every iteration to produce
            improved convergence and allow flexible learning rates.

        px:
        px_decay:
        px_power:
            The perturbation size controls how large of a change in x is used to measure changes in f.
            This is scaled based on the previous iteration's step size.

                dx = px / (1 + px_decay * i) ** px_power * norm(lr * previous_dx) * random_signs
                df = (f(x + dx) - f(x - dx)) / 2
                gradient ~ df / dx

            NOTE: If `px = int` is used, then `abs(dx[i]) == 0.5` is fixed and `x +- dx` is rounded.

        momentum:
            The momentum controls how much of the gradient is kept from previous iterations.
            Automatically tunes itself to increase as necessary.

        beta:
            A secondary momentum, which should be much closer to 1 than the other momentum.
            This is used by the Adam method.

        epsilon:
            Used to avoid division by 0 in the Adam method.

    Returns
    --------
        x:
            The estimated minimum of f.
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
    if px is int:
        x_temp = np.empty_like(x, dtype=int)
    elif px is not None:
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
    # Estimate the gradient and its square.
    b1 = 0.0
    b2 = 0.0
    gx = np.zeros_like(x)
    slow_gx = np.zeros_like(x)
    square_gx = np.zeros_like(x)
    for i in range(isqrt(x.size + 100)):
        # Compute df/dx in random directions.
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            df_dx = (f(np.rint(x + dx, casting="unsafe", out=x_temp)) - f(np.rint(x - dx, casting="unsafe", out=x_temp))) * 0.5 / dx
        else:
            dx = rng.choice((-1.0, 1.0), x.shape) / (1 + i)
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
            while f(x - lr * dx) < f(x):
                lr *= 1.4
    # Track the average value of x.
    mx = sqrt(m1 * m2)
    bx = mx
    x_avg = mx * x
    # Track the best (x, y).
    y_best = y / bn
    x_best = x.copy()
    # Track how many times the solution fails to improve.
    momentum_fails = 0
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
        if px is int:
            dx = rng.choice((-0.5, 0.5), x.shape)
            y1 = f(np.rint(x_next + dx, casting="unsafe", out=x_temp))
            y2 = f(np.rint(x_next - dx, casting="unsafe", out=x_temp))
        else:
            dx = (lr / sqrt(m1) * px / (1 + px_decay * i) ** px_power) * np.linalg.norm(dx)
            if adam:
                dx /= np.sqrt(square_gx / b2 + epsilon)
            dx *= rng.choice((-1.0, 1.0), x.shape)
            y1 = f(x_next + dx)
            y2 = f(x_next - dx)
        df = (y1 - y2) / 2
        df_dx = dx * (df * sqrt(x.size) / np.linalg.norm(dx) ** 2)
        # Update the momentum.
        if (df_dx.flatten() / np.linalg.norm(df_dx)) @ (gx.flatten() / np.linalg.norm(gx)) < 1.5 / sqrt(1 + 0.1 * momentum_fails) - 1:
            momentum_fails += 1
            m1 = (1.0 - momentum) / (1 + 0.1 * momentum_fails) ** 0.75
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
        y4 = f(x - lr * 0.5 * dx)
        y5 = f(x - lr / sqrt(m1) * dx)
        y6 = f(x)
        # Estimate the noise in f.
        bn += m2 * (1 - bn)
        y += m2 * (y3 - y)
        noise += m2 * ((y3 - y6) ** 2 + 1e-64 * (abs(y3) + abs(y6)) - noise)
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
    if px is int:
        x_best = np.rint(x_best).astype(int)
        x = np.rint(x, casting="unsafe", out=x_temp)
    return x_best if y_best - 0.25 * sqrt(noise / bn) < min(f(x), f(x)) else x
