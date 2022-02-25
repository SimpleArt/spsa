"""
Asynchronous IO
----------------
Contains asynchronous variants of `spsa` for calling
an asynchronous function concurrently each iteration.

See also:
    spsa.amp - Asynchronous Multiprocessing:
        Runs the whole spsa algorithm in a separate process for synchronous functions.
        Unlike `spsa.aio`, does not concurrently call functions each iteration. This is
        more appropriate if the SPSA algorithm itself should be ran in an executor or
        if each function call requires less time than it takes to share numpy arrays
        between processes.

Example Uses
-------------
- IO-bound functions.
- Functions running in executors.
- Running `spsa` asynchronously with other code (non-blocking).

Parallelizing Calls with Multiprocessing
-----------------------------------------
Calls to the objective function may be parallelized using `multiprocessing`,
but some care needs to be taken when using numpy arrays. The following provides
a general template on how to apply multiprocessing.

NOTE: Use this approach if the calculations are significantly slower than
      the amount of time it takes to share the data to other processes.

Example
--------
import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
import numpy as np

import spsa
from spsa import executor

def slow_f(data: bytes) -> float:
    '''Main calculations.'''
    x = np.frombuffer(data)
    ...  # Calculations.

async def run_in_executor(x: np.ndarray, executor: Executor = executor) -> float:
    '''Run the calculations with an executor.'''
    return await asyncio.get_running_loop().run_in_executor(executor, slow_f, x.tobytes())

x = asyncio.run(spsa.aio.minimize(run_in_executor, ...))  # Calls to `slow_f` are parallelized.
"""
from ._aio import maximize, minimize, with_input_noise
import spsa.aio.iterator as iterator

__all__ = ["iterator", "maximize", "minimize", "with_input_noise"]
