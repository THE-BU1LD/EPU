"""EPU-Ω∞ reference implementation."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import List
import random

def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")

def _vector(values, name):
    values = list(values)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        _finite(value, name)
    # Preserve exact integer values, while promoting other Real implementations
    # before their fixed-width or low-precision arithmetic can wrap or overflow.
    return [int(value) if isinstance(value, Integral) else float(value) for value in values]

def dot(a, b):
    a, b = _vector(a, "left vector"), _vector(b, "right vector")
    if len(a) != len(b):
        raise ValueError("dot product vectors must have equal dimensions")
    return sum(x * y for x, y in zip(a, b))

def clip(v, lo=-128.0, hi=127.0):
    v = _vector(v, "clip input")
    _finite(lo, "lower clip bound")
    _finite(hi, "upper clip bound")
    if lo > hi:
        raise ValueError("lower clip bound cannot exceed upper bound")
    return [max(lo, min(hi, x)) for x in v]

def l1(a, b):
    a, b = _vector(a, "left vector"), _vector(b, "right vector")
    if len(a) != len(b):
        raise ValueError("L1 vectors must have equal dimensions")
    return sum(abs(x - y) for x, y in zip(a, b))

def normalize(scores):
    scores = _vector(scores, "scores")
    mn = min(scores)
    s = [x - mn + 1.0 for x in scores]
    tot = sum(s)
    if not math.isfinite(tot):
        # Divide the complete shifted expression by a common scale before
        # subtraction/summation. Preserve the ordinary path (and frozen golden
        # arithmetic) exactly; only finite-input overflow takes this fallback.
        scale = max(1.0, max(abs(value) for value in scores))
        minimum_scaled = mn / scale
        s = [value / scale - minimum_scaled + 1.0 / scale for value in scores]
        tot = math.fsum(s)
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
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        if not self.memories:
            self.memories = [
                [2, 1, 0, 1],
                [-1, 2, 1, 0],
                [1, -2, 2, 1],
                [0, 1, -1, 2],
            ]
        if not self.z:
            self.z = [0.0] * self.dim
        self._validate()
        self.memories = [list(memory) for memory in self.memories]
        self.z = list(self.z)
        self.prev_z = self.z[:]

    def _validate(self):
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        for name in ("beta", "alpha", "gamma", "threshold"):
            _finite(getattr(self, name), name)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie between zero and one")
        if self.threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if len(_vector(self.z, "state")) != self.dim:
            raise ValueError("state length must equal dim")
        if not self.memories:
            raise ValueError("memory bank must be non-empty")
        for memory in self.memories:
            if len(_vector(memory, "memory")) != self.dim:
                raise ValueError("every memory length must equal dim")

    def scores(self):
        self._validate()
        beta = int(self.beta) if isinstance(self.beta, Integral) else float(self.beta)
        return [beta * dot(self.z, m) for m in self.memories]

    def step(self):
        s = self.scores()
        w = normalize(s)
        state = _vector(self.z, "state")
        memories = [_vector(memory, "memory") for memory in self.memories]
        alpha = float(self.alpha)
        recon = [0.0] * self.dim
        for wi, m in zip(w, memories):
            for i in range(self.dim):
                recon[i] += wi * m[i]
        next_z = [alpha * state[i] + (1.0 - alpha) * recon[i] for i in range(self.dim)]
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
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
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
