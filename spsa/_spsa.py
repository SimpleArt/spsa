import operator
import random

from functools import wraps
from math import isinf, isnan, isqrt, sqrt
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional, Sequence, SupportsIndex, TypedDict, Union

import numpy as np

__all__ = ["maximize", "optimize", "optimize_iterator"]

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
    x=np.ndarray,
    y=float,
    lr=float,
    beta_noise=float,
    beta1=float,
    beta2=Optional[float],
    noise=float,
    gradient=np.ndarray,
    slow_gradient=Optional[np.ndarray],
    square_gradient=Optional[np.ndarray],
)

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
    Implementation of the SPSA optimization algorithm with adaptive momentum (Adam) and learning rate tuning (line search).

    See `help(spsa)` for more details.

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
        temp = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (temp - y)
        noise += m2 * ((temp - f(x)) ** 2 - noise)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(3):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                temp = f(x)
                bn += m2 * (1 - bn)
                y += m2 * (temp - y)
                noise += m2 * ((temp - f(x)) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                if (f(x + dx) - f(x - dx)) ** 2 > 8 * noise / bn:
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                temp = f(x)
                bn += m2 * (1 - bn)
                y += m2 * (temp - y)
                noise += m2 * ((temp - f(x)) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                if (f(x + dx) - f(x - dx)) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    # Estimate the gradient and its square.
    b1 = 0.0
    gx = np.zeros_like(x)
    if adam:
        b2 = 0.0
        slow_gx = np.zeros_like(x)
        square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px
        df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx
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
        temp = f(x)
        # Increase the learning rate while it is safe to do so.
        dx = 3 * b2 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        for _ in range(3):
            while f(x - lr * dx) < temp:
                lr *= 1.4
    # Initial step size.
    dx = b2 / b1 * gx / np.sqrt(square_gx + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - lr * dx
        # Compute df/dx in at the next point.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        df = (f(x_next + dx) - f(x_next - dx)) / 2
        df_dx = df / dx
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
        # Estimate the noise in f.
        y1 = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (y1 - y)
        noise += m2 * ((y1 - f(x)) ** 2 - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if df ** 2 < 2 * noise / bn:
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        y2 = f(x - lr / 3 * dx)
        y3 = f(x - lr * 3 * dx)
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
    return x

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

            NOTE: x, gradient, slow_gradient, and square_gradient are mutable numpy arrays.
                  Modifying them may mess up the optimizer.

            x:
                The current estimated minimum of f.
                NOTE: Updating this value in the dictionary will update in the optimizer.
            y:
                The current value of f(x).
            lr:
                The current learning rate (not including decay).
                NOTE: Updating this value in the dictionary will update in the optimizer.
            beta_noise:
            beta1:
            beta2:
                Used for the formulas
                    y / beta_noise
                    sqrt(noise / beta_noise)
                    gradient / beta1
                    slow_gradient / beta2
                    square_gradient / beta2
                to get unbiased estimates of each variable.

                On their own, the estimates are closer to 0 than they should be
                and need to be divided by their respective betas for correction.

                If adam is not used, beta2 = None.
            noise:
                An estimate for how much noise is in f(x).
                Used for SPSA.
            gradient:
                The estimated gradient of f at x.
            slow_gradient:
                The slower estimate of the gradient of f at x.
                Biased more towards previous iterations.
                None if adam is not used.
                Used for the square_gradient.
            square_gradient:
                An estimate for the component-wise square of the gradient of f at x.
                None if adam is not used.
                Used for the Adam method.
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
        temp = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (temp - y)
        noise += m2 * ((temp - f(x)) ** 2 - noise)
    # Estimate the perturbation size that should be used.
    if px is None:
        px = 3e-4 * (1 + 0.25 * np.linalg.norm(x))
        for _ in range(3):
            # Increase `px` until the change in f(x) is signficiantly larger than the noise.
            while True:
                # Update the noise.
                temp = f(x)
                bn += m2 * (1 - bn)
                y += m2 * (temp - y)
                noise += m2 * ((temp - f(x)) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if sufficiently accurate.
                if (f(x + dx) - f(x - dx)) ** 2 > 8 * noise / bn:
                    break
                # `dx` is dangerously small, so `px` should be increased.
                px *= 1.2
            # Attempt to decrease `px` to improve the gradient estimate unless the noise is too much.
            for _ in range(3):
                # Update the noise.
                temp = f(x)
                bn += m2 * (1 - bn)
                y += m2 * (temp - y)
                noise += m2 * ((temp - f(x)) ** 2 - noise)
                # Compute a change in f(x) in a random direction.
                dx = rng.choice((-1.0, 1.0), x.shape)
                dx *= px
                # Stop if too much noise.
                if (f(x + dx) - f(x - dx)) ** 2 < 8 * noise / bn:
                    break
                # `dx` can be safely decreased, so `px` should be decreased.
                px /= 1.1
            # Set a minimum perturbation.
            px = max(px, epsilon * (1 + 0.25 * np.linalg.norm(x)))
    # Estimate the gradient and its square.
    b1 = 0.0
    gx = np.zeros_like(x)
    if adam:
        b2 = 0.0
        slow_gx = np.zeros_like(x)
        square_gx = np.zeros_like(x)
    for _ in range(isqrt(isqrt(x.size + 4) + 4)):
        # Compute df/dx in random directions.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px
        df_dx = (f(x + dx) - f(x - dx)) * 0.5 / dx
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
        temp = f(x)
        # Increase the learning rate while it is safe to do so.
        dx = 3 * b2 / b1 * gx
        if adam:
            dx /= np.sqrt(square_gx + epsilon)
        for _ in range(3):
            while f(x - lr * dx) < temp:
                lr *= 1.4
    # Generate initial iteration.
    variables = dict(
        x=x,
        y=y,
        lr=lr,
        beta_noise=bn,
        beta1=b1,
        beta2=b2 if adam else None,
        noise=noise,
        gradient=gx,
        slow_gradient=slow_gx if adam else None,
        square_gradient=square_gx if adam else None,
    )
    yield variables
    x = variables["x"]
    lr = variables["lr"]
    del variables
    # Initial step size.
    dx = b2 / b1 * gx / np.sqrt(square_gx + epsilon)
    # Run the number of iterations.
    for i in range(iterations):
        # Estimate the next point.
        x_next = x - lr * dx
        # Compute df/dx in at the next point.
        dx = rng.choice((-1.0, 1.0), x.shape)
        dx *= px / (1 + px_decay * i) ** px_power
        df = (f(x_next + dx) - f(x_next - dx)) / 2
        df_dx = df / dx
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
        # Estimate the noise in f.
        y1 = f(x)
        bn += m2 * (1 - bn)
        y += m2 * (y1 - y)
        noise += m2 * ((y1 - f(x)) ** 2 - noise)
        # Update `px` depending on the noise and gradient.
        # `dx` is dangerously small, so `px` should be increased.
        if df ** 2 < 2 * noise / bn:
            px *= 1.2
        # `dx` can be safely decreased, so `px` should be decreased.
        elif px > 1e-8 * (1 + 0.25 * np.linalg.norm(x)):
            px /= 1.1
        # Perform line search.
        y2 = f(x - lr / 3 * dx)
        y3 = f(x - lr * 3 * dx)
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
        # Generate the variables for the next iteration.
        variables = dict(
            x=x,
            y=y,
            lr=lr,
            beta_noise=bn,
            beta1=b1,
            beta2=b2 if adam else None,
            noise=noise,
            gradient=gx,
            slow_gradient=slow_gx if adam else None,
            square_gradient=square_gx if adam else None,
        )
        yield variables
        x = variables["x"]
        lr = variables["lr"]
        del variables
