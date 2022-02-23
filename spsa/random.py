from typing import Iterator, Union, overload

import numpy as np

from ._spsa import ArrayLike

def uniform_iterator(a: float, b: float, /, *, repeat: bool = ...) -> Iterator[float]:
    ...

def uniform_iterator(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = ...) -> Iterator[np.ndarray]:
    ...

def uniform_iterator(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = False) -> Union[Iterator[float], Iterator[np.ndarray]]:
    """
    Generates random points between a and b.

    If repeat=True, then random values are repeated.

    See also:
        spsa.random.regression_iterator
    """
    mean = np.array((np.asarray(a) + np.asarray(b)) / 2)
    deviation = np.abs(np.asarray(b) - mean)
    del a, b
    if repeat:
        while True:
            t = np.random.uniform(-1, 1, mean.shape)
            t *= deviation
            t += mean
            yield t
            yield t
    else:
        while True:
            t = np.random.uniform(-1, 1, mean.shape)
            t *= deviation
            t += mean
            yield t

def regression_iterator(a: float, b: float, /, *, repeat: bool = ...) -> Iterator[float]:
    ...

def regression_iterator(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = ...) -> Iterator[np.ndarray]:
    ...

def regression_iterator(a: ArrayLike, b: ArrayLike, /, *, repeat: bool = True) -> Union[Iterator[float], Iterator[np.ndarray]]:
    """
    Generates random points between a and b, distributed towards the edges for regression.

    If repeat=True, then random values are repeated.

    See also:
        spsa.random.uniform_iterator
    """
    mean = np.array((np.asarray(a) + np.asarray(b)) / 2)
    deviation = np.abs(np.asarray(b) - mean)
    del a, b
    if repeat:
        while True:
            t = np.random.uniform(-1, 1, mean.shape)
            t /= np.sqrt(np.abs(t))
            t *= deviation
            t += mean
            yield t
            yield t
    else:
        while True:
            t = np.random.uniform(-1, 1, mean.shape)
            t /= np.sqrt(np.abs(t))
            t *= deviation
            t += mean
            yield t
