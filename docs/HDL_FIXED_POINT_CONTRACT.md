# HDL fixed-point semantic contract

This document freezes the arithmetic contract used to compare `code/epu_core.v`
with the Python/paper update. It is an **arithmetic parity contract**, not an FPGA
performance, timing, resource, capacity, or robustness claim.

## Retained parity configuration

- State dimension: 4.
- Memory count: 4.
- Word width: 16 signed bits.
- Fraction bits: 8.
- Scale: 256 (Q7.8 effective range under the paper clip).
- Memory values: the four vectors in `code/epu_reference.py`, encoded exactly at scale 256.
- Beta: 6/1.
- Alpha: 3/5 = 0.6.
- Convergence threshold: L1 delta <= 64 fixed-point units = 0.25.
- Output clip: [-128.0, 127.0], matching the floating reference.

The HDL currently rejects parameterizations with a different retained memory-bank
shape (D != 4 or K != 4). Generalized memories are outside this parity gate.

## Score and normalization

For state coordinate integers `z_q` and memory integers `m_q`:

1. Compute the full-precision integer dot product `sum(z_q * m_q)`.
2. Apply beta and divide once by the fixed-point scale.
3. Signed division rounds to nearest, with exact half ties away from zero.
4. Let `s_min = min(scores_q)`.
5. `shifted_q[i] = scores_q[i] - s_min + SCALE`.
6. `sum_shifted = sum(shifted_q)`.

The Python normalization contract is not approximated by a fixed memory-count
divisor. Reconstruction is computed directly as

```
recon_q[j] =
    round(sum_i shifted_q[i] * memory_q[i][j] / sum_shifted)
```

This is algebraically the normalized weighted reconstruction, with one
predeclared rounding point at the reconstructed coordinate.

## Residual update, saturation, convergence

Each coordinate uses

```
next_q[j] = round((3 * state_q[j] + 2 * recon_q[j]) / 5)
```

using the same signed round-to-nearest/ties-away rule.

The value is then saturated to the fixed-point encoding of [-128, 127].

Convergence is asserted for the candidate next state when

```
sum_j abs(next_q[j] - state_q[j]) <= 64
```

which is exactly the Python threshold 0.25 at scale 256.

## Overflow boundary

The retained D=4, K=4, Q7.8 memory/state configuration is evaluated with 32-bit
signed intermediate integers in the HDL. The retained golden trajectory is far
inside those limits. This contract does not claim safe arithmetic for arbitrary
parameter expansions; widening D/K/W/beta requires a new overflow proof and parity
gate before it is described as supported.

## Verification boundary

`examples/hdl_fixed_golden_vectors.json` is generated from:
- the floating Python update;
- the frozen fixed-point arithmetic reference.

The retained vector starts from the same deterministic seed as the existing Python
golden trajectory, quantized once to Q7.8.

For every retained step:
- the fixed-point HDL output must match the fixed-point reference exactly;
- the fixed-point next state must remain within **one LSB (1/256)** per coordinate
  of the floating Python trajectory started from the quantized initial state;
- the convergence decision must match the frozen threshold rule.

CI compiles and runs the HDL using Icarus Verilog. Passing this gate supports only
the statement that the retained HDL arithmetic matches this fixed-point contract
for the retained vector. It does not establish synthesis closure or hardware
benchmark performance.
