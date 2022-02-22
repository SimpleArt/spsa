import operator
import random

from functools import wraps
from math import isinf, isnan, isqrt, sqrt
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional, Sequence, SupportsFloat, SupportsIndex, Tuple, TypedDict, Union

import numpy as np

__all__ = ["maximize", "optimize", "optimize_iterator", "with_input_noise"]

ArrayLike = Union[
    np.ndarray,
    Sequence[float],
    Sequence[Sequence[float]],
    Sequence[Sequence[Sequence[float]]],
    Sequence[Sequence[Sequence[Sequence[float]]]],
    Sequence[Sequence[Sequence[Sequence[Sequence[float]]]]],
]

OptimizerVariables = TypedDict(
    "OptimizerVariables",
    x_min=np.ndarray,
    y_min=np.ndarray,
    x=np.ndarray,
    y=float,
    lr=float,
    beta_noise=float,
    beta1=float,
    beta2=float,
    noise=float,
    gradient=np.ndarray,
    slow_gradient=np.ndarray,
    square_gradient=np.ndarray,
)

def _type_check(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
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
    /,
) -> np.ndarray:
    """Type check the parameters and casts `x` to a numpy array."""
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
    # Cast to numpy array.
    x = np.asarray(x, dtype=float)
    # Type-check.
    if x.size == 0:
        raise ValueError("cannot optimize with array of size 0")
    elif np.isnan(x).any():
        raise ValueError(f"x must not contain nan")
    elif np.isinf(x).any():
        raise ValueError(f"x must not contain infinity")
    return x

def maximize(f: Callable[[np.ndarray], float], /) -> Callable[[np.ndarray], float]:
    """
    Turns the function into a maximization function.

    Usage
    ------
        Maximize a function instead of minimizing it:
            x = spsa.optimize(maximize(f), x)
    """
    @wraps(f)
    def wrapper(x: np.ndarray, /) -> float:
        return -f(x)
    return wrapper

def with_input_noise(f: Callable[[np.ndarray], float], /, noise: float) -> Callable[[np.ndarray], float]:
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
    def wrapper(x: np.ndarray) -> float:
        nonlocal rng_iter
        if rng_iter is None:
            rng_iter = rng_iterator(x.shape)
        return f(x + next(rng_iter))
    return wrapper

def optimize(
    f: Callable[[np.ndarray], float],
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
    Implementation of the SPSA optimization algorithm.

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

                x = spsa.optimize(spsa.with_input_noise(f, noise=0.5), ...)

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
            If no px is given, then a crude estimate is found based on the noise in f.

            The perturbation size controls how large of a change in x is used to measure changes in f.

                px = px_start / (1 + px_decay * iteration) ** px_power
                dx = px * random_signs
                df = (f(x + dx) - f(x - dx)) / 2
                gradient ~ df / dx

            Furthermore, the perturbation size is automatically tuned every iteration to produce
            more accurate gradient approximations, reducing chaotic behavior.

        momentum:
            The momentum controls how much of the gradient is kept from previous iterations.

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
                noise += m2 * ((y1 - y2) ** 2 - noise)
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
                noise += m2 * ((y1 - y2) ** 2 - noise)
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
            while f(x - lr * dx) < f(x):
                lr *= 1.4
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
        y4 = f(x - lr / 3 * dx)
        y5 = f(x - lr * 3 * dx)
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
        lr /= 64 * improvement_fails
    return x_min if y_min - 0.25 * sqrt(noise / bn) < min(f(x), f(x)) else x

def optimize_iterator(
    f: Callable[[np.ndarray], float],
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
) -> Iterator[OptimizerVariables]:
    """
    A generator which yields a dict of variables each iteration.
    Useful for logging iterations, manually modifying variables
    between iterations, or implementing custom termination.

    See `help(spsa.optimizer)` for more details.

    Yields
    -------
        optimizer_variables:
            A dictionary containing optimizer variables.

            NOTE: Most variables come with a corresponding beta variable which should be used.

            NOTE: x, gradient, slow_gradient, and square_gradient are mutable numpy arrays.
                  Modifying them may mess up the optimizer.

            x_min:
            y_min:
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
            while f(x - lr * dx) < f(x):
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
        y4 = f(x - lr / 3 * dx)
        y5 = f(x - lr * 3 * dx)
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
        lr /= 64 * improvement_fails
