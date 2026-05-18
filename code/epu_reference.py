"""EPU-Ω∞ reference implementation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import random

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def clip(v, lo=-128.0, hi=127.0):
    return [max(lo, min(hi, x)) for x in v]

def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def normalize(scores):
    mn = min(scores)
    s = [x - mn + 1.0 for x in scores]
    tot = sum(s)
    return [x / tot for x in s]

@dataclass
class EPUCore:
    dim: int = 4
    beta: float = 6.0
    alpha: float = 0.6
    gamma: float = 0.0
    threshold: float = 0.25
    memories: List[List[float]] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    prev_z: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.memories:
            self.memories = [
                [2, 1, 0, 1],
                [-1, 2, 1, 0],
                [1, -2, 2, 1],
                [0, 1, -1, 2],
            ]
        if not self.z:
            self.z = [0.0] * self.dim
        self.prev_z = self.z[:]

    def scores(self):
        return [self.beta * dot(self.z, m) for m in self.memories]

    def step(self):
        s = self.scores()
        w = normalize(s)
        recon = [0.0] * self.dim
        for wi, m in zip(w, self.memories):
            for i in range(self.dim):
                recon[i] += wi * m[i]
        next_z = [self.alpha * self.z[i] + (1.0 - self.alpha) * recon[i] for i in range(self.dim)]
        next_z = clip(next_z)
        info = {
            "state": self.z[:],
            "scores": s,
            "weights": w,
            "recon": recon,
            "next": next_z[:],
            "delta": l1(self.z, next_z),
        }
        self.prev_z = self.z[:]
        self.z = next_z
        return info

    def converged(self):
        return l1(self.prev_z, self.z) <= self.threshold

    def run(self, steps=12, verbose=True):
        history = []
        for t in range(steps):
            info = self.step()
            history.append(info)
            if verbose:
                print(f"step={t:02d} z={info['state']} next={info['next']} Δ={info['delta']:.3f}")
            if self.converged():
                if verbose:
                    print(f"converged at step {t}")
                break
        return history


def demo():
    core = EPUCore()
    target = core.memories[2][:]
    rng = random.Random(7)
    core.z = [x + rng.uniform(-1.5, 1.5) for x in target]
    print("initial:", core.z)
    print("target :", target)
    core.run()

if __name__ == "__main__":
    demo()
