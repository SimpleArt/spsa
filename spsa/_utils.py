import operator
from math import isinf, isnan
from typing import Callable, Optional, Sequence, SupportsFloat, SupportsIndex, Type, TypedDict, Union

import numpy as np

__all__ = ["ArrayLike", "OptimizerVariables", "type_check", "immutable_view"]

ArrayLike = Union[
    np.ndarray,
    float,
    Sequence[float],
    Sequence[Sequence[float]],
    Sequence[Sequence[Sequence[float]]],
    Sequence[Sequence[Sequence[Sequence[float]]]],
    Sequence[Sequence[Sequence[Sequence[Sequence[float]]]]],
]

OptimizerVariables = TypedDict(
    "OptimizerVariables",
    x_best=np.ndarray,
    y_best=np.ndarray,
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

def type_check(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    adam: bool,
    iterations: int,
    lr: Optional[float],
    lr_decay: float,
    lr_power: float,
    px: Union[float, Type[int]],
    px_decay: float,
    px_power: float,
    momentum: float,
    beta: float,
    epsilon: float,
    /,
) -> np.ndarray:
    """Type check the parameters and casts `x` to a numpy array."""
    # Type-check.
    if not callable(f):
        raise TypeError(f"f must be callable, got {f!r}")
    elif not isinstance(x, (SupportsFloat, np.ndarray, Sequence)):
        raise TypeError(f"x must be either a real number, numpy array, or sequence, got {x!r}")
    elif not isinstance(iterations, SupportsIndex):
        raise TypeError(f"iterations cannot be interpreted as an integer, got {iterations!r}")
    elif not isinstance(adam, SupportsIndex):
        raise TypeError(f"adam cannot be interpreted as an integer, got {adam!r}")
    adam = bool(operator.index(adam))
    names = ("lr_decay", "lr_power")
    values = (lr_decay, lr_power)
    if lr is not None:
        names = ("lr", *names)
        values = (lr, *values)
    if px is not int:
        names = (*names, "px")
        values = (*values, px)
    names = (*names, "px_decay", "px_power", "momentum", "beta", "epsilon")
    values = (*values, px_decay, px_power, momentum, beta, epsilon)
    for name, value in zip(names, values):
        if not isinstance(value, SupportsFloat):
            raise TypeError(f"{name} must be real, got {value!r}")
        elif isnan(float(value)):
            raise ValueError(f"{name} must not be nan, got {value!r}")
        elif isinf(float(value)):
            raise ValueError(f"{name} must not be infinite, got {value!r}")
        elif float(value) <= 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")
    names = ("lr_power", "px_power", "momentum", "beta")
    values = (lr_power, px_power, momentum, beta)
    for name, value in zip(names, values):
        if float(value) >= 1:
            raise ValueError(f"{name} must not be greater than 1, got {value!r}")
    # Cast to numpy array.
    x = np.array(x, dtype=float)
    # Type-check.
    if x.size == 0:
        raise ValueError("cannot optimize with array of size 0")
    elif np.isnan(x).any():
        raise ValueError(f"x must not contain nan")
    elif np.isinf(x).any():
        raise ValueError(f"x must not contain infinity")
    return x

def immutable_view(x: np.ndarray, /) -> np.ndarray:
    """Returns a view of the input which cannot be modified."""
    view = np.asarray(x)[...]
    view.setflags(write=False)
    return view
