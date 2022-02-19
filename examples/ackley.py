import numpy as np
import spsa

def ackley(x: np.ndarray) -> float:
    """
    The Ackley function is
        -20 exp(-0.2 sqrt(0.5 (x^2 + y^2)))
        - exp(0.5 (cos(2 pi x) + cos(2 pi y)))
    """
    return -(
        20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2).sum()))
        + np.exp(0.5 * np.cos(2 * np.pi * x).sum())
    )

def main() -> None:
    print("Ackley:")
    x = spsa.optimize(ackley, (2, 3))
    print(f"y = {ackley(x)}")
    print(f"x = {x}")
    print("-" * 140)

if __name__ == "__main__":
    main()
