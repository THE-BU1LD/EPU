# Integrated reference and verifier candidate

This development candidate combines the reference-boundary repairs from PRs
#9 and #10 with the transcript-verification repair from PR #11. The component
commits and their earlier evidence remain in the history:

| Component | Source commit |
| --- | --- |
| Floating normalization and fixed-point admission (#9) | `38ea737c64c59e6986cb8c65b7e3a3d549e998cf` |
| Unbounded integer signed rounding (#10, includes #9) | `627e64c09339c09d9db4359056e345da929ffa46` |
| Strict ordered simulator transcripts (#11) | `92b0db58b179fef4dc74a2a368152fe4d3a0d7db` |

## Additional witnessed defect

The combined code still admitted fixed-width integer scalars into floating
reference vector operations. Arithmetic wrapped before a float calculation or
finite-value check could detect the lost value. Actual NumPy int64 inputs
produced these results:

| Constructed operation | Before repair | Exact intended result |
| --- | --- | --- |
| `dot([2**32], [2**32])` | `0` | `2**64` |
| `l1([-2**63], [2**63 - 1])` | `1` | `2**64 - 1` |
| Normalize the two signed-word extrema | `[1, 0]` | `[1 / (2**64 + 1), 2**64 / (2**64 + 1)]` |
| One-dimensional retrieval from memories `[-2**32]` and `[2**32]`, state `[2**32]`, beta 1, alpha 0 | Next state `[0]` | Next state `[127]` after the declared clip |

An independent source review found that a fixed-width integer beta could
reintroduce wrapping or an integer-conversion overflow when multiplying a
correct dot product. This failure was reproduced separately before repair.

Validated Integral vector entries and beta now use unbounded Python integers
before arithmetic. Floating inputs keep their existing path. No NumPy runtime
dependency is added. Five dependency-free arithmetic regressions include a real
complete retrieval step; separate local checks use actual NumPy signed and
unsigned scalar types. This does not promise arbitrary-magnitude floating
products or generalize the hardware's frozen parameterization.

## Verification and reproduction

The initial four vector regressions failed on the component source. After
vector promotion they passed, and the reviewer-found beta regression failed;
promoting beta resolved that failure. The final complete suite contains
**40 passing unittest methods**, including the five new cases, all component
regressions, and the original reference invariants. Both retained golden-vector
checks pass. Ruff F and whitespace checks pass.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/generate_python_golden.py --verify examples/python_golden_vectors.json --atol 1e-12
python scripts/generate_hdl_golden.py --verify examples/hdl_fixed_golden_vectors.json
python scripts/verify_hdl_parity.py
```

The final command requires Icarus Verilog (`iverilog` and `vvp`). Those binaries
were unavailable locally; actual HDL simulation is the hosted CI gate. The
local source hashes, command outputs, and baseline failures are retained in
`verification/reference-integration-20261010/`. The original `receipt.json`
records the intermediate vector-only check; `receipt-final.json` records the
final source after the review-found scalar repair. Earlier evidence is retained
under its original source identity.

## Research boundary

The HDL, frozen fixed-point contract, retained golden files, benchmark proposal,
and paper are unchanged. Passing checks establish the listed software identities
and, when actual simulation passes, arithmetic parity for the retained eight
steps within the declared one-LSB floating tolerance. They establish no FPGA
synthesis, timing, energy, resource, capacity, general robustness, or performance
result. Hardware research and normal integration review remain open.
