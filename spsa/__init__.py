"""
Simultaneous Perturbation Stochastic Optimization (SPSA)

The purpose of this package is to provide multivariable optimizers
using SPSA. Although other optimizers exist, not many implement
SPSA, which has various pros and cons. Additionally, SPSA has
few requirements so that you don't have to install large packages
like scipy just to optimize a function.

Usage
------
Synchronous Functions:
    x = spsa.maximize(f, x)
    x = spsa.minimize(f, x)

    for variables in spsa.iterator.maximize(f, x):
        print(variables)

    for variables in spsa.iterator.minimize(f, x):
        print(variables)

Asynchronous Functions:
    # spsa.aio - Asynchronous IO.
    # Useful for:
    #     IO-bound functions.
    #     Functions running in executors.
    #     Running `spsa` asynchronously with other code (non-blocking).
    # See `help(spsa.aio)` for more details.

    x = await spsa.aio.maximize(async_def_f, x)
    x = await spsa.aio.minimize(async_def_f, x)

    async for variables in spsa.aio.iterator.maximize(async_def_f, x):
        print(variables)

    async for variables in spsa.aio.iterator.minimize(async_def_f, x):
        print(variables)

Synchronous Functions with Multiprocessing:
    # spsa.amp - Asynchronous Multiprocessing.
    # Useful for:
    #     Running `spsa` asynchronously with other code (non-blocking).
    #     Running `spsa` in an executor for efficiently running multiple at a time.
    #     Not for improving a single `spsa` call.
    # See `help(spsa.amp)` for more details.

    x = await spsa.amp.maximize(def_f, x)
    x = await spsa.amp.minimize(def_f, x)

    async for variables in spsa.amp.iterator.maximize(def_f, x):
        print(variables)

    async for variables in spsa.amp.iterator.minimize(def_f, x):
        print(variables)

Example
--------
    import numpy as np
    import spsa

    # Squared distance to 0.
    def sphere(x: np.ndarray) -> float:
        return np.linalg.norm(x) ** 2

    # Attempt to find the minimum.
    print(spsa.minimize(sphere, [1, 2, 3]))

    # Sample output:
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
from ._defaults import executor
from ._spsa import maximize, minimize, with_input_noise
import spsa.aio as aio
import spsa.amp as amp
import spsa.iterator as iterator
import spsa.random as random

__all__ = ["aio", "amp", "executor", "iterator", "maximize", "minimize", "random", "with_input_noise"]

__version__ = "0.1.0"
