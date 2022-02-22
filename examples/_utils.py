"""Contains utility functions for the examples."""
import random

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np

import spsa

def controlled_input_noise(f: Callable[[np.ndarray], float], noise: float, /) -> Callable[[np.ndarray], float]:
    """Adds noise to the input before calling."""
    return spsa.with_input_noise(f, noise)

def random_input_noise(f: Callable[[np.ndarray], float], noise: float, /) -> Callable[[np.ndarray], float]:
    """Adds noise to the input before calling."""
    rng = np.random.default_rng()
    def wrapper(x: np.ndarray, /) -> float:
        return f(x + rng.uniform(-noise, noise, x.shape))
    return wrapper

HEADERS: Dict[Callable, str] = {
    controlled_input_noise: "{} with controlled input noise:",
    random_input_noise: "{} with random input noise:",
}

def run(f: Callable[[np.ndarray], float], x: Tuple[float], /, name: str, *args: Any, noise: float, **kwargs: Any) -> None:
    print(f"{name}:")
    x_ = spsa.optimize(f, x, *args, **kwargs)
    print(f"y = {f(x_)}")
    print(f"x = {x_}")
    print("-" * 80)
    del x_
    for wrapper, header in HEADERS.items():
        print(header.format(name))
        x_ = spsa.optimize(wrapper(f, noise), x, *args, **kwargs)
        print(f"y = {f(x_)}")
        print(f"x = {x_}")
        print("-" * 80)
        del x_
