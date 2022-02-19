from functools import partial
import numpy as np
import spsa

def sphere_with_noise(x: np.ndarray, noise: float) -> float:
    """The sphere function with noise is (x0 + noise)^2 + (x1 + noise)^2 + ..."""
    return np.linalg.norm(x + np.random.uniform(-noise, noise, x.shape)) ** 2

def main() -> None:
    print("Sphere with noise:")
    sphere = partial(sphere_with_noise, noise=1.0)
    x = spsa.optimize(sphere, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(f"y = {sphere(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
