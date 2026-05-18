# EPU-Ω∞ Final Repo

A complete scaffold for the Hopfield-inspired Engram Processing Unit.

## Included
- `paper/EPU_Omega_Infinity_Paper.md`
- `code/epu_reference.py`
- `code/epu_core.v`
- `logisim/Blueprint.md`
- `benchmarks/Benchmark_Protocol.md`
- `appendix/100_extensions.md`
- `examples/test_vectors.json`
- `figures/architecture_ascii.txt`
- `scripts/run_demo.sh`

## Quick start
```bash
bash scripts/run_demo.sh
```

## Core idea
A latent state `z` is compared against a memory bank `M`. The machine:
1. scores similarity,
2. normalizes the scores into weights,
3. reconstructs a candidate state,
4. mixes the candidate with the old state,
5. repeats until convergence.
