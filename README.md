# EPU-Ω∞ reference scaffold

EPU-Ω∞ is an exploratory Hopfield-inspired associative-memory compute scaffold.

## What is currently executable

- `code/epu_reference.py` — deterministic Python reference/demo for score → normalized retrieval → residual state update;
- `scripts/run_demo.sh` — runs the Python reference;
- `code/epu_core.v` — an **illustrative HDL scaffold**, not yet parity-verified against the Python/paper update;
- `paper/EPU_Omega_Infinity_Paper.md` — architecture concept/equations;
- `logisim/Blueprint.md` — proposed circuit decomposition;
- `benchmarks/Benchmark_Protocol.md` — proposed evaluation dimensions;
- `examples/test_vectors.json` and supporting figures/appendix material.

The repository CI currently compiles and executes the Python reference demo. It does **not** certify the Verilog implementation, FPGA behavior, benchmark performance, or paper claims.

## Quick start

```bash
bash scripts/run_demo.sh
```

## Hardware-parity boundary

The current Verilog is not yet a verified implementation of the Python/paper semantics. In particular, issue #3 records current differences in:

- beta scaling of similarity scores;
- normalization of shifted scores;
- convergence behavior.

Treat `code/epu_core.v` as a hardware sketch until fixed-point semantics are frozen and simulator-based golden-vector parity passes.

## Core conceptual update

The paper/reference concept is:

1. score memories against the current latent state;
2. normalize scores into retrieval weights;
3. reconstruct a candidate state from the memory bank;
4. mix the candidate with the previous state;
5. iterate until a defined convergence criterion is reached.

This description is a design target. It is not evidence of speed, capacity, noise robustness, FPGA readiness, or superiority over conventional architectures.

## Evidence boundary

No benchmark result or hardware-performance claim is supported merely by the presence of the paper, Verilog file, Logisim blueprint, or proposed benchmark protocol. Any future claim should be tied to executable tests, retained outputs, explicit fixed-point semantics, and reproducible hardware/software comparison evidence.
