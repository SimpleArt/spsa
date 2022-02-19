import numpy as np
import spsa

def rastrigin(x: np.ndarray) -> float:
    """The Rastrigin function is (x0^2 - 10 cos(2 pi x0)) + ..."""
    return (x**2 - 10*np.cos(2*np.pi*x)).sum()

def main() -> None:
    print("Rastrigin:")
    x = spsa.optimize(rastrigin, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(f"y = {rastrigin(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
