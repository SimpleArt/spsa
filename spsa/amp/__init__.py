"""
Asynchronous Multiprocessing
-----------------------------
Contains asynchronous variants of `spsa` for asynchronously
running `spsa` with synchronous functions in a separate process.

See also:
    spsa.aio - Asynchronous IO:
        Runs asynchronous function calls concurrently every iteration. Unlike
        `spsa.amp`, does not run the SPSA algorithm in a separate process. This
        is more appropriate if the objective function can be ran concurrently or
        if the objective function is significantly more expensive than sharing
        numpy arrays between processes.

NOTE: Recommended that `ThreadPoolExecutor`s should not be used.
      Use `spsa.aio` instead of threading.

Example Uses
-------------
- Running `spsa` asynchronously with other code (non-blocking).
- Running `spsa` in an executor for efficiently running multiple at a time.
- Not for improving a single `spsa` call.

Parallelizing Calls with Multiprocessing
-----------------------------------------
The SPSA algorithm itself are parallelized using `multiprocessing`, which
does not improve the performance of SPSA on its own, but rather avoids SPSA
from being CPU-blocking. `spsa.amp` handles the concurrency itself, running
the entire SPSA algorithm in a separate process. Calls may then be ran
asynchronously using `asyncio`.

Example
--------
import asyncio
import numpy as np

import spsa

def f(x: np.ndarray) -> float:
    '''Main calculations.'''
    ...

x = asyncio.run(spsa.amp.minimize(f, ...))  # May be ran concurrently with other SPSA calls.
"""
from ._amp import maximize, minimize, with_input_noise
import spsa.amp.iterator as iterator

__all__ = ["iterator", "maximize", "minimize", "with_input_noise"]
