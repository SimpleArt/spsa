import numpy as np
from ._utils import run

def function(x: np.ndarray) -> float:
    x1, x2 = x
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    )

def main() -> None:
    run(function, (2, 2), "Beale", noise=0.01)

if __name__ == "__main__":
    main()
