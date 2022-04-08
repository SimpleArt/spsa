from math import pi
from typing import Iterator, Tuple, Union, overload

import numpy as np

from spsa._utils import ArrayLike

__all__ = ["noise", "regression", "uniform"]

def noise(noise: float, shape: Tuple[int, ...], /) -> Iterator[np.ndarray]:
    """
    Generates random noise of a given shape, where abs(result[i]) <= noise.
    """
    rng = np.random.default_rng()
    while True:
        random_noise = rng.uniform(-noise, noise, shape)
        yield random_noise
        yield random_noise

def regression(a: float, b: float, /, *, repeat: bool = ...) -> Iterator[float]:
    ...

def regression(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = ...) -> Iterator[np.ndarray]:
    ...

def regression(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = True) -> Union[Iterator[float], Iterator[np.ndarray]]:
    """
    Generates random points between a and b, distributed towards the edges for regression.

    If repeat=True, then random values are repeated.

    See also:
        spsa.random.uniform_iterator
    """
    mean = np.array((np.asarray(a) + np.asarray(b)) / 2)
    deviation = np.abs(np.asarray(b) - mean)
    del a, b
    rng = np.random.default_rng()
    if repeat:
        while True:
            t = np.cos(pi * rng.random(mean.shape))
            t *= deviation
            t += mean
            yield t
            yield t
    else:
        while True:
            t = np.cos(pi * rng.random(mean.shape))
            t *= deviation
            t += mean
            yield t

def uniform(a: float, b: float, /, *, repeat: bool = ...) -> Iterator[float]:
    ...

def uniform(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = ...) -> Iterator[np.ndarray]:
    ...

def uniform(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = False) -> Union[Iterator[float], Iterator[np.ndarray]]:
    """
    Generates random points between a and b.

    If repeat=True, then random values are repeated.

    See also:
        spsa.random.regression_iterator
    """
    mean = np.array((np.asarray(a) + np.asarray(b)) / 2)
    deviation = np.abs(np.asarray(b) - mean)
    del a, b
    rng = np.random.default_rng()
    if repeat:
        while True:
            t = rng.uniform(-1, 1, mean.shape)
            t *= deviation
            t += mean
            yield t
            yield t
    else:
        while True:
            t = rng.uniform(-1, 1, mean.shape)
            t *= deviation
            t += mean
            yield t
