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

Asynchronous Optimization
--------------------------

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

Why use SPSA?
--------------

### Fast Convergence

SPSA rapidly converges in a way similar to gradient descent. Unlike other black-box algorithms, SPSA is a rapid local optimization algorithm.

### Black-Box Derivative-Free

SPSA does not require anything beyond the function being optimized. Unlike gradient descent, the derivative is not necessary.

### Stochastic Functions

SPSA does not require the function to be completely accurate. If a randomized approximation of the function is easier to compute, SPSA usually converges just as well. Unlike stochastic tunneling, SPSA is not easily tricked into converging to points which randomly produced optimal values when the average value is suboptimal.

### High-Dimensional Problems

SPSA is applicable to problems with many dimensions. Unlike most other black-box algorithms, SPSA converges well in many dimensions.

### Efficient Iterations

SPSA uses only a few function calls per iteration plus vector operations. Unlike other black-box algorithms, the number of function calls per iteration does not scale with the dimensions of the problem.

### Efficient Parallelization

SPSA is easily parallelized in several ways. `spsa.aio` performs parallel calls to the objective function each iteration and `spsa.amp` allows the SPSA algorithm itself to be parallelized.

### Integer Constraints

SPSA can easily be applied to integer-constrained problems by rounding the input. In fact the provided implementation includes automatic tuning for the perturbations, which will automatically increase the distance between function calls until the arguments are about an integer apart from each other in order to observe a difference in output.

### Code Complexity

SPSA requires less than 100 lines of code. Although this implementation includes more features, it is still less than 200 lines of code. This is unlike some other algorithms, such as Bayesian optimization, which may take several hundred more lines of code.

SPSA also works entirely off of vector operations (not even matrix operations) and coin-flipping rng. This makes the source code easily transferable to other languages.

Why use this implementation of SPSA?
-------------------------------------

### Learning Rate Tuning

This implementation includes learning rate tuning, whereas most other implementations employ learning rate scheduling. With learning rate tuning, the learning rate is automatically optimized every iteration, at the cost of a few more calculations. With learning rate scheduling, the learning rate follows a predetermined sequence of values which usually decay to `0`. In theory, a decaying learning rate ensures eventual convergence if the function is noisy. In practice, we have found that a tuned learning rate, even in the presence of noise, can actually perform just as well if not better. After all, it is impossible to run an infinite number of iterations. Instead, it is usually faster to optimize the learning rate every iteration. Furthermore, this makes the learning rate more robust, allowing it to speed up when it can or rapidly slow down if it should.

This implementation uses a simple tuning algorithm which updates the learning rate every iteration using only 2 additional function calls. Furthermore, the tuning algorithm is robust against stochastic functions and momentum. This means `f(x +- lr * dx)` is not optimized in terms of `lr`, but rather it looks ahead at future iterations to account for momentum while also considering the approximate amount of noise in the objective function.

### Perturbation Size Tuning

This implementation includes perturbation size tuning, whereas most other implementations employ perturbation size scheduling. With perturbation size tuning, the perturbation size is automatically updated every iteration, at the cost of a few more calculations. With perturbation size scheduling, the perturbation size follows a predetermined sequence of values which usually decay to `0`. In theory, a decaying perturbation size ensures eventual convergence to the gradient if the function is noisy. In practice, we have found that a tuned perturbation size, especially in the presence of noise, performs better. This is because the noise in the objective function is amplified by a division by the perturbation size, causing small perturbations to be incredibly noisy. This implementation automatically scales the perturbation size based on the noise in the objective function, which ensures the noise are usually not so drastic that SPSA may diverge randomly on its own.

### Adaptive Momentum (Adam)

This implementation includes the Adam method, whereas most other implementations do not. Each component is rescaled according to how large the gradient is in that dimension, which accelerates convergence in flatter directions while stabilizing convergence in steep directions.

Furthermore, the perturbation size is scaled using the Adam method. This helps distribute the error in the gradient instead of only having an accurate estimate of the gradient in steep directions and an extremely inaccurate estimate of the gradient in flat directions. This may improve convergence in high-dimensional problems or with functions with greatly varying gradient components.

### Basin-Hopping

For functions with many local minima, the `spsa.with_input_noise` function (including its `spsa.aio` and `spsa.amp` variants) provides ways to perform basin-hopping to an extent. By replacing the objective function with a stochastic estimate of the objective function over entire regions, local minima are removed, encouraging SPSA to converge to more globalized minima instead.

### Iterators

For every optimizer, an iterator variant is also provided which exposes most of the variables inside of the optimizer. This enables users to track the progress of the optimizer instead of just relying on the final result as well as implement custom termination algorithms.

### Asynchronous Computations

For every optimizer, an asynchronous variant is also provided which allows SPSA to be ran with asynchronous code. This enables various forms of parallelism or even just simple concurrency if SPSA needs to run concurrently with other code instead of blocking other asynchronous code from running.
"""
from ._defaults import executor
from ._spsa import maximize, minimize, with_input_noise
import spsa.aio as aio
import spsa.amp as amp
import spsa.iterator as iterator
import spsa.random as random

__all__ = ["aio", "amp", "executor", "iterator", "maximize", "minimize", "random", "with_input_noise"]

__version__ = "0.1.2"
