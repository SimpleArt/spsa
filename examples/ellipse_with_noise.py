from functools import partial
import numpy as np
import spsa

def ellipse_with_noise(x: np.ndarray, noise: float) -> float:
    """
    The ellipse function with noise is
        (x0 + noise)^2 + 2 * (x1 + noise)^2 + 3 * (x2 + noise)^2 + ...
    """
    return np.linalg.norm(np.sqrt(np.arange(1, 1 + len(x))) * (x + np.random.uniform(-noise, noise, x.shape))) ** 2

def main() -> None:
    print("Ellipse with noise:")
    ellipse = partial(ellipse_with_noise, noise=1.0)
    x = spsa.optimize(ellipse, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(f"y = {ellipse(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
