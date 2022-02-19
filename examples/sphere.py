import numpy as np
import spsa

def sphere(x: np.ndarray) -> float:
    """The sphere function is x0^2 + x1^2 + ..."""
    return np.linalg.norm(x) ** 2

def main() -> None:
    print("Sphere:")
    x = spsa.optimize(sphere, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(f"y = {sphere(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
