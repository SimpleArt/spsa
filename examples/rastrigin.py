import numpy as np
from ._utils import run

def function(x: np.ndarray) -> float:
    """The Rastrigin function is (x0^2 - 10 cos(2 pi x0)) + ..."""
    return (x**2 - 10*np.cos(2*np.pi*x)).sum()

def main() -> None:
    run(function, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "Rastrigin", noise=0.5)

if __name__ == "__main__":
    main()
