# Unified scalar reference integration — 17 October 2026

## Source and purpose

PR #12 at `4049ad669dda8a3af973bbe9d464b4cf0ff5b18a` and PR #10 at
`ce2dffe8cec5a6d5d84092114fbf36c1dfe2a917` contain divergent repairs. This
integration retains both commit histories, including current #9 and #11, every
component regression suite, the ordered HDL transcript verifier, and the original
receipts under their original source identities. Its intended merge parents are
the two exact commits above, in that order.

PR #12 preserves exact Integral values but leaves other Real implementations in
their native arithmetic. PR #10 promotes every vector entry to float, preventing
native low-precision multiplication but losing the exact integer-distance
contract. Neither branch includes all of the other's latest repairs.

## Observed failures and resolution

- On the #12 arithmetic, `dot([numpy.float16(300)], [numpy.float16(300)])`
  produced infinity instead of 90000. A low-precision beta of 300 multiplying
  the score for state 300 and memory 300 also produced infinity instead of
  27000000. The dependency-free NarrowFloat controls reproduce both paths.
- The combined 46-method suite initially reported two failures: low-precision
  dot overflow and an incompatible distance assertion. The latter expected
  `float(2**64)`, while exact signed-word separation is `2**64 - 1`. The duplicate
  assertion now retains the exact integer result, agreeing with #12's existing
  exact-distance test.
- Two added methods exercise low-precision beta and an actual NumPy float16
  complete step. Before repair, the focused seven-method suite reported two
  failures and one error. After vector and beta promotion, a constructed
  residual update still produced 2.265625 instead of 2.2666015625. This exposed
  remaining caller-dtype arithmetic in reconstruction and the residual update.

Validated Integral vector entries and beta now use Python int; other Real
values use Python float. Reconstruction and residual updates use promoted state,
memory and alpha values. The input value of a low-precision scalar is preserved
as represented; this does not recover precision already lost before admission.
The original arithmetic order for ordinary Python inputs is retained.

## Verification

Executed locally with Python 3.12 and NumPy 2.3.5:

```bash
python3 -W error -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/generate_python_golden.py --verify examples/python_golden_vectors.json --atol 1e-12
python3 scripts/generate_hdl_golden.py --verify examples/hdl_fixed_golden_vectors.json
```

All **48 unittest methods passed**, with no skips and warnings treated as
errors. Both golden checks passed: floating tolerance 1e-12 and the retained
eight-step fixed-point trajectory within 1/256 of the floating reference.
The existing CI retains pinned NumPy 2.3.5, exact-source checkout/assertion,
all discovered regression tests, and actual Icarus simulation. Icarus and vvp
are unavailable locally, so hosted parity for the published integrated commit
remains the final executable gate.

The HDL, frozen fixed-point contract, both retained golden-vector files,
benchmark proposal and paper are unchanged. These constructed engineering
checks do not establish hardware performance or any scientific efficacy claim.
No protected outcome, training run or scientific benchmark was executed.
