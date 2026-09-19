"""Frozen fixed-point arithmetic reference for the EPU HDL parity gate."""
from __future__ import annotations

import math
from typing import Sequence

WORD_BITS = 16
FRAC_BITS = 8
SCALE = 1 << FRAC_BITS
BETA_NUM = 6
BETA_DEN = 1
ALPHA_NUM = 3
ALPHA_DEN = 5
THRESHOLD_Q = SCALE // 4
CLIP_MIN_Q = -128 * SCALE
CLIP_MAX_Q = 127 * SCALE

MEMORIES = (
    (2, 1, 0, 1),
    (-1, 2, 1, 0),
    (1, -2, 2, 1),
    (0, 1, -1, 2),
)
MEMORIES_Q = tuple(tuple(value * SCALE for value in row) for row in MEMORIES)


def round_div_signed(numerator: int, denominator: int) -> int:
    """Round a signed rational to nearest integer, ties away from zero."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize(value: float) -> int:
    scaled = value * SCALE
    if scaled >= 0:
        quantized = math.floor(scaled + 0.5)
    else:
        quantized = -math.floor(-scaled + 0.5)
    return max(CLIP_MIN_Q, min(CLIP_MAX_Q, quantized))


def dequantize(value: int) -> float:
    return value / SCALE


def step_q(state_q: Sequence[int]) -> dict[str, object]:
    if len(state_q) != 4:
        raise ValueError("retained parity contract requires four state coordinates")

    state = [int(value) for value in state_q]
    scores_q: list[int] = []
    for memory in MEMORIES_Q:
        dot_raw = sum(left * right for left, right in zip(state, memory))
        scores_q.append(round_div_signed(BETA_NUM * dot_raw, BETA_DEN * SCALE))

    minimum = min(scores_q)
    shifted_q = [score - minimum + SCALE for score in scores_q]
    total = sum(shifted_q)
    if total <= 0:
        raise ArithmeticError("shifted-score normalization must have positive mass")

    recon_q = [
        round_div_signed(
            sum(shifted_q[index] * MEMORIES_Q[index][coord] for index in range(4)),
            total,
        )
        for coord in range(4)
    ]
    next_q = [
        round_div_signed(
            ALPHA_NUM * state[coord] + (ALPHA_DEN - ALPHA_NUM) * recon_q[coord],
            ALPHA_DEN,
        )
        for coord in range(4)
    ]
    next_q = [max(CLIP_MIN_Q, min(CLIP_MAX_Q, value)) for value in next_q]
    delta_q = sum(abs(before - after) for before, after in zip(state, next_q))

    return {
        "scores_q": scores_q,
        "shifted_q": shifted_q,
        "recon_q": recon_q,
        "next_q": next_q,
        "delta_q": delta_q,
        "converged": delta_q <= THRESHOLD_Q,
    }
