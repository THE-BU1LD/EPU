# EPU-Ω∞ reference scaffold

EPU-Ω∞ is an exploratory Hopfield-inspired associative-memory compute scaffold.

## What is currently executable

- `code/epu_reference.py` — deterministic Python reference/demo for score → normalized retrieval → residual state update;
- `scripts/run_demo.sh` — runs the Python reference;
- `code/epu_core.v` — a frozen Q7.8 fixed-point implementation of the score → normalize → reconstruct → residual-update contract;
- `paper/EPU_Omega_Infinity_Paper.md` — architecture concept/equations;
- `logisim/Blueprint.md` — proposed circuit decomposition;
- `benchmarks/Benchmark_Protocol.md` — proposed evaluation dimensions;
- `examples/test_vectors.json` and supporting figures/appendix material.

The repository CI executes deterministic Python unit tests, the Python reference, regenerates retained floating/fixed-point golden vectors, compiles the Verilog with Icarus Verilog, and compares the retained eight-step HDL trajectory exactly against the fixed-point reference. The fixed-point trajectory is additionally required to stay within one Q7.8 LSB of the floating Python update on the retained vector. This is an arithmetic/software parity gate only; it does **not** certify FPGA synthesis, timing, resource use, benchmark performance, capacity, or paper-level hardware claims.

## Quick start

```bash
bash scripts/run_demo.sh
```

Run the deterministic software-invariant tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Hardware-parity boundary

The intended fixed-point semantics are frozen in
`docs/HDL_FIXED_POINT_CONTRACT.md`:

- Q7.8 signed state/memory values;
- beta = 6;
- shifted-score normalization by the **sum of all shifted scores**;
- normalized reconstruction computed without a fixed memory-count divisor;
- alpha = 3/5 residual update;
- round-to-nearest with half ties away from zero;
- saturation to the Python clip range [-128, 127];
- L1 convergence at <= 0.25.

`examples/hdl_fixed_golden_vectors.json` is derived from the Python semantics and
the fixed-point arithmetic reference. CI fails unless the Verilog matches those
fixed-point outputs exactly and the quantized trajectory remains within one LSB
of the floating Python trajectory through the retained convergence point.

This closes the arithmetic mismatch tracked in issue #3 only after the simulator
gate passes. It does **not** establish synthesis closure, FPGA validation, hardware
speed/energy/resource results, or superiority over conventional architectures.

## Core conceptual update

The paper/reference concept is:

1. score memories against the current latent state;
2. normalize scores into retrieval weights;
3. reconstruct a candidate state from the memory bank;
4. mix the candidate with the previous state;
5. iterate until a defined convergence criterion is reached.

This description is a design target. It is not evidence of speed, capacity, noise robustness, FPGA readiness, or superiority over conventional architectures.

## Evidence boundary

The deterministic unit tests check bounded software invariants such as normalized weights, dimensionality, finite values, and repeatability. No benchmark result or hardware-performance claim is supported merely by those tests or by the presence of the paper, Verilog file, Logisim blueprint, or proposed benchmark protocol. Any future claim should be tied to executable tests, retained outputs, explicit fixed-point semantics, and reproducible hardware/software comparison evidence.
