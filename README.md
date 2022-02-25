Simultaneous Perturbation Stochastic Optimization (SPSA)

The purpose of this package is to provide multivariable optimizers using SPSA. Although other optimizers exist, not many implement SPSA, which has various pros and cons. Additionally, SPSA has few requirements so that you don't have to install large packages like scipy just to optimize a function.

PIP Install
--------
Unix/macOS:

```cmd
python3 -m pip install spsa
```

Windows:

```cmd
py -m pip install spsa
```

Usage
------
Import:
```python
import spsa
```

Synchronous Functions:

```python
x = spsa.maximize(f, x)
x = spsa.minimize(f, x)

for variables in spsa.iterator.maximize(f, x):
    print(variables)

for variables in spsa.iterator.minimize(f, x):
    print(variables)
```

Asynchronous Functions:

```python
# spsa.aio - Asynchronous IO.# Performs function calls concurrently every iteration.
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
```

Synchronous Functions with Multiprocessing:

```python
# spsa.amp - Asynchronous Multiprocessing.
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
```

Example
--------

```python
import numpy as np
import spsa

# Squared distance to 0.
def sphere(x: np.ndarray) -> float:
    return np.linalg.norm(x) ** 2

# Attempt to find the minimum.
print(spsa.minimize(sphere, [1, 2, 3]))

# Sample output:
#     [-5.50452777e-21 -9.48070248e-21  9.78726993e-21]
```

Pros & Cons
------------
A comparison of SPSA, Gradient Descent, and Bayesian Optimization are shown below.

|  | SPSA | Gradient Descent | Bayesian Optimization |
| :--- | :---: | :---: | :---: |
| Calls per Iteration | Constant<sup>[1]</sup> f(x) | 1 fprime(x) | Constant f(x) |
| Stochastic | Stochastic f | Stochastic fprime | Stochastic f |
| Convergence | Local | Local | Global |
| Dimensions | Any | Any | <20 |
| Lines of Code | ~100 | 10-100 | >100 |
| Integer Optimization | Applicable<sup>[2]</sup> | Inapplicable | Applicable<sup>[3]</sup> |
| Parallel Calls | Applicable<sup>[4]</sup> | Not Obvious | Applicable |

[1]: Normally requires only 2 calls, but linear search and noise-adjusting perturbation sizes require a few extra calls per iteration.

[2]: Use `f(round(x))`.

[3]: Use a different Gaussian process.

[4]: See `spsa.aio` and `spsa.amp`.
