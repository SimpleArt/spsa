import numpy as np
from ._utils import run

def function(x: np.ndarray) -> float:
    """The ellipse function is x0^2 + 2 * x1^2 + 3 * x2^2 + ..."""
    return np.linalg.norm(np.sqrt(np.arange(1, 1 + len(x))) * x) ** 2

def main() -> None:
    run(function, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "Ellipse", noise=0.5)

if __name__ == "__main__":
    main()
