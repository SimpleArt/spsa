from math import cos, pi
import numpy as np
from typing import Callable, Tuple, Union

# loss(array) -> float.
LossFunction = Callable[[np.ndarray], float]

def spsa(
    f: LossFunction,
    x: np.ndarray,
    lr: float = None,
    lr_decay: float = 1e-3,
    lr_power: float = 0.5,
    lr_min: float = 1e-8,
    lr_max: float = 1e-1,
    px: float = 3e-4,
    px_decay: float = 1e-2,
    px_power: float = 0.161,
    momentum: float = 0.9,
    iterations: int = 10000,
    cycles: int = 10,
) -> float:
    """
    Implementation of SPSA with momentum and learning rate resets.

    Parameters
    -----------
        f:
            The function being optimized. Called as `f(array) -> float`.
        x:
            The initial point used. This value is edited and returned.
        lr:
            If no learning rate is given, then a crude estimate is found using line search.
            The learning rate controls the speed of convergence according to the following formula:
                x -= lr * gradient_estimate
            The learning rate is dampened over time via decay and power according to the following formula:
                lr / (1 + lr_decay * current_iteration) ** lr_power
            When a cycle ends, the learning rate is tuned by seeing if it should be doubled or halved.
        lr_decay:
            Controls how fast the lr decays by multiplying the number of iterations.
            When 0, the lr does not change at all during the iterations.
        lr_power:
            Controls how fast the lr decays by exponentiating of the number of iterations.
            When 0, the lr does not change at all during the iterations.
        lr_min:
        lr_max:
            The minimum and maximum allowed initial learning rates.
        px:
            The perturbation size controls how much x is perturbed to estimate the gradient according to the following formula:
                dx = px * random_signs
            The px is dampened over time via decay and power according to the following formula:
                px / (1 + px_decay * current_iteration) ** px_power
            When a cycle ends, the perturbation size is decreased by a factor of `0.9`.
        px_decay:
        px_power:
            See lr_decay and lr_power above.
        momentum:
            How much of the previous estimates of the gradient are used on each iteration.
            The momentum is also reset at the start of each cycle.
        iterations:
            The number of iterations ran. Roughly halve the number of function evaluations.
        cycles:
            The number of cycles the iterations are split up into.
            Each cycle attempts to tune the lr and px and also reset the momentum to clear information too old.

    Returns
    --------
        x:
            The estimated minimum of f.

    Prints
    -------
        lr:
            The learning rate at the start of each cycle.
        f(x):
            The current loss value reached.
    """
    # Estimate the learning rate.
    if lr is None:
        # Estimate the gradient.
        gx = np.zeros(x.shape)
        N = 10
        for _ in range(N):
            dx = px * np.random.choice([-1, 1], x.shape)
            gx += (f(x + dx) - f(x - dx)) / (2 * dx)
        gx /= N
        # Estimate the learning rate.
        lr = np.clip(1e-5 / (np.linalg.norm(gx) + 1e-7), lr_min, lr_max)
        print(f"lr = {lr}")
        # Apply a simple line search to find a decent lr estimate.
        for factor in (0.5, 2, 0.707, 1.414):
            while f(x - lr * gx) > f(x - factor * lr * gx):
                lr *= factor
        lr = np.clip(lr, lr_min, lr_max)
        x -= lr * gx
        lr *= 0.5
    # Initial gradient momentum setup.
    gx = np.zeros(x.shape, dtype=float)
    w = 0.0
    for i in range(iterations):
        i %= iterations / cycles
        # Restart the cycles.
        if i < 1:
            # Tune the learning rate and perturbations.
            lr *= min(0.5, 1.414, key=lambda lr: f(x - lr * gx))
            px *= 0.9
            # Reset gradient momentum estimates.
            gx = np.zeros(x.shape, dtype=float)
            w = 0.0
            print(f"lr = {lr};  \tf(x) = {f(x)}")
        # Perturb and update the gradient estimate.
        dx = px / (1 + px_decay * i) ** px_power * np.random.choice([-1, 1], x.shape)
        gx += (1 - momentum) * ((f(x + dx) - f(x - dx)) / (2 * dx) - gx)
        w += (1 - momentum) * (1 - w)
        # Update the solution x.
        x -= lr / (1 + lr_decay * i) ** lr_power * gx / w
    return x

def constraint_map(x: np.ndarray, low: Union[float, np.ndarray], high: Union[float, np.ndarray]) -> np.ndarray:
    """
    Rather than enforcing constraints, you can instead map R^n into the range of possible values you desire.

    Parameters
    -----------
        x:
            An array ranging from -inf to +inf.
        low:
        high:
            The range of values that x is mapped to.

    Returns
    --------
        s:
            An array of values ranging from low to high.

    Example
    --------
        Maps [-5, -4, ..., 3, 4] to the range (0 to 10).
        The map can be seen to be symmetric about `(x, y) = (0, (low+high)/2)`
        and monotone decreasing.
        >>> constraint_map(np.arange(-5, 5), 0, 10)
        array([9.93307149, 9.8201379 , 9.52574127, 8.80797078, 7.31058579,
               5.        , 2.68941421, 1.19202922, 0.47425873, 0.1798621 ])
    """
    return low + high / (1 + np.exp(x))

def loss_setup(
    L: int,
    H: int,
    A: np.ndarray,
    B: np.ndarray,
    K: Tuple[float, float, float, float],
) -> LossFunction:
    """Creates the loss function for the given problem using the provided hyper-parameters."""
    def loss(x: np.ndarray) -> float:
        """The loss function being optimized."""
        # Map x to the range (0 to H).
        s = constraint_map(x, 0, H)
        # Helper arrays.
        sA = s + A
        sB = s + B
        ds = np.diff(s)
        dds = np.diff(ds)
        # The provided loss formula.
        return (
            K[0] * (sA.max() - sB.min())
            + K[1] * (sA[:L].max() - sB[:L].min())
            + K[2] * np.abs(ds).sum()
            + K[3] * (np.abs(dds) ** 1.5).sum()
        )
    return loss

def main(
    N: int = 1000,
    L: int = 50,
    H: int = 200,
    A: np.ndarray = None,
    B: np.ndarray = None,
    K: Tuple[float, float, float, float] = None,
    x: np.ndarray = None,
) -> None:
    """Run an example."""
    # Setup random A/B/K/x.
    if A is None:
        A = np.random.uniform(0, H, N)
    if B is None:
        B = np.random.uniform(0, H, N)
    if K is None:
        K = tuple(np.random.uniform(0, 10, 4))
    # Use a heuristic estimate of the solution.
    if x is None:
        x = np.zeros(N, dtype=float)
        indexes = list(range(N))
        indexes.sort(key=lambda ind: A[ind], reverse=True)
        for ind, dx in zip(indexes, (10, 5, 3, 1)):
            x[ind] -= dx
        indexes.sort(key=lambda ind: B[ind])
        for ind, dx in zip(indexes, (10, 5, 3, 1)):
            x[ind] += dx
    # Setup the loss function.
    loss = loss_setup(L, H, A, B, K)
    # Print the final loss value.
    print(f"loss(x) = {loss(spsa(loss, x))}")
    # Map x to the range (0 to H).
    s = constraint_map(x, 0, H)
    # Print the solution found.
    print(f"s = \n{s}")

if __name__ == "__main__":
    main()
