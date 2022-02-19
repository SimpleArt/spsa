import numpy as np
import spsa

def rosenbrock(x: np.ndarray) -> float:
    """The Rosenbrock function is (100 (x0 - x1^2)^2 + (1 - x1)^2) + ..."""
    return (100 * (x[:-1] - x[1:]**2) ** 2 + (1 - x[1:]) ** 2).sum()

def main() -> None:
    print("Rosenbrock:")
    x = spsa.optimize(rosenbrock, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(f"y = {rosenbrock(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
