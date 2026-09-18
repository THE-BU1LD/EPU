# EPU-Ω∞

> **Status: engineering/research scaffold with a runnable reference implementation. The benchmark protocol is planned, not executed evidence.**

EPU-Ω∞ is a Hopfield-inspired Engram Processing Unit exploration. The current repository contains a small Python reference model, hardware/design sketches, a benchmark protocol, paper material, and example vectors.

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

The current Python reference implementation is stdlib-only:

```bash
bash scripts/run_demo.sh
```

Run deterministic engineering tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Core idea

A latent state `z` is compared against a memory bank `M`. The reference machine:

1. scores similarity;
2. normalizes scores into weights;
3. reconstructs a candidate state;
4. mixes the candidate with the previous state;
5. repeats until its convergence rule or step limit is reached.

## Engineering evidence boundary

The unit/smoke tests check software invariants such as deterministic behavior, finite values, dimensionality, normalized weights, and a runnable demo. Passing them means the **reference implementation behaves consistently on its bounded fixtures**.

It does **not** establish:

- benchmark superiority;
- hardware speed/efficiency;
- memory-capacity advantage;
- noise robustness beyond executed experiments;
- biological plausibility;
- a completed paper result.

The benchmark list in `benchmarks/Benchmark_Protocol.md` is a plan until its experiments are actually run under a frozen protocol and retained with source/provenance.

## Next research gate

Before any empirical headline claim:

1. freeze baselines, datasets/fixtures, seeds, metrics, thresholds, and compute/hardware conditions;
2. implement the benchmark runner separately from the demo;
3. retain raw per-run outputs and source/environment identities;
4. execute the predeclared matrix;
5. report negative/mixed outcomes without rescue tuning.

Canonical repository: `THE-BU1LD/EPU`.
