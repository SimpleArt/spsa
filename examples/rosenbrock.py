import numpy as np
from ._utils import run

def function(x: np.ndarray) -> float:
    """The Rosenbrock function is (100 (x0 - x1^2)^2 + (1 - x1)^2) + ..."""
    return (100 * (x[:-1] - x[1:]**2) ** 2 + (1 - x[1:]) ** 2).sum()

def main() -> None:
    run(function, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "Rosenbrock", noise=0.1)

if __name__ == "__main__":
    main()
