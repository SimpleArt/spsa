Simultaneous Perturbation Stochastic Optimization (SPSA)

The purpose of this package is to provide multivariable optimizers using SPSA. Although other optimizers exist, not many implement SPSA, which has various and cons. Additionally, SPSA has few requirements so that you don't have to install large packages like scipy optimize a function.

Usage
------
Synchronous Functions:

```python
x = spsa.optimize(f, x)
x = spsa.optimize(spsa.maximize(f), x)  # For maximization.

for variables in spsa.optimize_iterator(f, x):
    print(variables)
```

Asynchronous Functions:

```python
x = await spsa.aio.optimize(f, x)
x = await spsa.aio.optimize(spsa.aio.maximize(f), x)  # For maximization.

async for variables in spsa.aio.optimize(f, x):
    print(variables)
```

Example
--------

```python
import numpy as np
import spsa

# Sample function which has a minimum at 0.
def sphere(x: np.ndarray) -> float:
    return np.linalg.norm(x) ** 2

# Attempt to find the minimum.
print(spsa.optimize(sphere, [1, 2, 3]))
# Sample result:
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

[2]: Use f(round(x)), px=0.5, and px_decay=0.

[3]: Use a different Gaussian process.

[4]: See `spsa.aio`.
