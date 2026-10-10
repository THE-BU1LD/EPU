# HDL simulator transcript integrity

## Witnessed false PASS

At main `37ed57d75b1d0ac24f169c9444514ecee39b31f5`,
`scripts/verify_hdl_parity.py` stored simulator observations in a dictionary
indexed by step. A later result silently replaced an earlier result for the
same step. A fixture supplied `RESULT 0 999 999 999 999 0` followed by all eight
correct retained rows to the actual `main` function. It printed `verified HDL
parity` despite the contradictory observation. This establishes a verifier
defect; it does not establish that an earlier real simulator run was incorrect.

## Repair

The verifier now consumes one exact observation per retained step, in the
generated testbench's order. It rejects wrong values, convergence mismatches,
duplicates, reordering, extra or missing rows, and malformed `RESULT` lines.
It checks both stdout and stderr for `ERROR:` or `FATAL:` simulator diagnostics,
even when a process returns exit code zero. Ordinary informational output may
coexist with the expected results. No observation overwrites another.

The fixed-point core, fixed and floating Python references, arithmetic contract,
and both retained golden-vector files are unchanged. This change improves the
reliability of an arithmetic parity gate; it adds no synthesis, timing, resource,
energy, capacity, robustness, or performance evidence.

## Validation and retained limitations

On 2026-10-10, all **16 unittest methods** passed (six existing and ten new),
including main-level regressions for overwritten wrong rows and stderr errors.
Both retained golden-vector regeneration checks passed; the eight-step fixed
trajectory remains within its frozen 1/256 floating tolerance. Ruff F checks and
whitespace checks pass on changed files.

Local Icarus simulation was not available. A bounded isolated `apt download
iverilog` attempt reported that the package could not be located; it made no
host package changes. Hosted CI retains the existing Icarus parity simulation.
Mocked transcript fixtures test the checker and are not hardware simulations.

An audit also considered signed-intermediate overflow in the retained Q7.8
configuration. Enumerating all 16 signed-word corner states found no overflow
witness (largest absolute reconstruction numerator 1,308,732,416). This bounded
arithmetic check is not a generalized proof for other parameterizations, and no
speculative HDL change was made. The retained eight-step parity scope remains
the scientific claim boundary.
