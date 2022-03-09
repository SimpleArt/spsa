import asyncio
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import spsa

def loss(x: np.ndarray) -> float:
    return (100 * (x[:-1] - x[1:]**2) ** 2 + (1 - x[1:]) ** 2).sum()
    return np.linalg.norm(x) ** 2
    return np.linalg.norm(range(1, 1 + x.size) * x) ** 2

async def aloss(x: np.ndarray) -> float:
    return loss(x)

def uq(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return data.mean(axis=0), np.var(data, axis=0)

async def plot(x0: Sequence[float], trials: int, **kwargs: Any) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title(str(len(x0)))
    ys = np.array([
        [loss(v["x"] / v["x_bias"]) for v in spsa.iterator.minimize(loss, x0, **kwargs)]
        for _ in range(trials)
    ])
    for y in ys:
        ax.plot(range(len(y)), y, c=(0.1, 0.1, 0.5, 0.25))
    mean, variance = uq(np.log(ys))
    del y, ys
    ax.plot(range(len(mean)), np.exp(mean), c=(0.0, 0.0, 1.0, 1.0), label="AggMo")
    ax.plot(range(len(mean)), np.exp(mean - 2 * np.sqrt(variance / trials)), c=(0.0, 0.0, 0.8, 1.0))
    ax.plot(range(len(mean)), np.exp(mean + 2 * np.sqrt(variance / trials)), c=(0.0, 0.0, 0.8, 1.0))
    ys = []
    for _ in range(trials):
        ys.append([loss(v["x"] / v["beta_x"]) async for v in spsa.aio.iterator.minimize(aloss, x0, **kwargs)])
    ys = np.array(ys)
    for y in ys:
        ax.plot(range(len(y)), y, c=(0.5, 0.1, 0.1, 0.25))
    mean, variance = uq(np.log(ys))
    del y, ys
    ax.plot(range(len(mean)), np.exp(mean), c=(1.0, 0.0, 0.0, 1.0), label="Adam")
    ax.plot(range(len(mean)), np.exp(mean - 2 * np.sqrt(variance / trials)), c=(0.8, 0.0, 0.0, 1.0))
    ax.plot(range(len(mean)), np.exp(mean + 2 * np.sqrt(variance / trials)), c=(0.8, 0.0, 0.0, 1.0))
    ax.legend(loc="upper right")
    plt.show()

asyncio.run(plot(range(30), 10))
