"""
Simultaneous Perturbation Stochastic Optimization (SPSA)

The purpose of this package is to provide multivariable optimizers
using SPSA. Although other optimizers exist, not many implement
SPSA, which has a variant of pros and cons. Additionally, SPSA has
few requirements so that you don't have to install large packages
like scipy optimize a function.

Usage
------
Synchronous Functions:
    x = spsa.optimize(f, x)
    x = spsa.optimize(spsa.maximize(f), x)  # For maximization.

    for variables in spsa.optimize_iterator(f, x):
        print(variables)

Asynchronous Functions:
    x = await spsa.aio.optimize(f, x)
    x = await spsa.aio.optimize(spsa.aio.maximize(f), x)  # For maximization.

    async for variables in spsa.aio.optimize(f, x):
        print(variables)

Example
--------
    import numpy as np
    import spsa

    # Sample function which has a minimum at 0.
    def sphere(x: np.ndarray) -> float:
        return np.linalg.norm(x) ** 2

    # Attempt to find the minimum.
    print(spsa.optimize(sphere, [1, 2, 3]))
    # Sample result:
    #     [-5.50452777e-21 -9.48070248e-21  9.78726993e-21]

Pros & Cons
------------
A comparison of SPSA, Gradient Descent, and Bayesian Optimization are shown below.

Calls per Iteration:
    SPSA:
        6 calls for the objective function per iteration,
        including a line search implementation.
    Gradient Descent:
        1 call for the gradient per iteration.
    Bayesian Optimization:
        N call for the objective function per iteration,
        for some constant N.

Stochastic (noisy/random functions):
    SPSA:
        Requires stochastic objective function.
        Implementation works extremely well.
    Gradient Descent:
        Requires stochastic gradient.
        Also known as stochastic gradient descent.
    Bayesian Optimization:
        Requires stochastic objective function.
        No special changes really necessary.

Convergence:
    SPSA:
        Localized convergence, similar to gradient descent.
    Gradient Descent:
        Localized convergence.
    Bayesian Optimization:
        Globalized convergence, sampling a large variety of points.

Dimensionality:
    SPSA:
        Works well for high-dimensional problems,
        similar to gradient descent.
    Gradient Descent:
        Works well for high-dimensional problems.
    Bayesian Optimization:
        Struggles for large search spaces.

Complexity:
    SPSA:
        Only about 100 lines of code.
    Gradient Descent:
        Only about 10 lines of code, more complex
        variants like Adam may require a bit more.
    Bayesian Optimization:
        Requires >100 lines of code, mostly to implement the
        Gaussian processes.

Integer Optimization:
    SPSA:
        Use f(round(x)) as the objective function and `px=0.5, px_decay=0`.
    Gradient Descent:
        Not applicable.
    Bayesian Optimization:
        Use a different Gaussian process than real optimization.

Parallel Calls
    SPSA:
        Obvious places for parallel calls, see `help(spsa.aio)` for more details.
    Gradient Descent:
        No obvious benefit.
    Bayesian Optimization:
        Obvious places for parallel calls.
"""
from ._spsa import maximize, optimize, optimize_iterator
import spsa.aio as aio

__all__ = ["aio", "maximize", "optimize", "optimize_iterator"]
