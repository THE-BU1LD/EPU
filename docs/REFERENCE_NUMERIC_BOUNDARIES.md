# Reference numeric boundaries

The floating and fixed-point references now reject malformed dimensions,
nonfinite values and implicit fractional fixed-point inputs. This does not
expand the frozen HDL parity configuration or regenerate any retained golden
trajectory.

The shifted-score normalization retains its original arithmetic for ordinary
finite inputs. If subtraction or summation would overflow despite finite
scores, it divides the complete shifted expression by one common positive
scale before normalization. The resulting weights implement the same ratio;
high-precision Decimal checks cover subtraction and sum overflow separately.

Quantization clips large finite values before multiplication by 256. Thus
`quantize(1e308)` saturates to the declared maximum instead of overflowing
before it reaches the saturation operation. Half ties continue to round away
from zero. Raw `step_q` coordinates must be signed 16-bit integers; values that
would silently truncate or wrap at the HDL input cannot be called parity inputs.

The floating reference supports explicit consistent custom dimensions. Its
default memory bank is four dimensional, so another dimension requires matching
custom memories. Every step revalidates externally assigned state and memory
values before changing state.

## Verify

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/generate_python_golden.py --verify examples/python_golden_vectors.json --atol 1e-12
python scripts/generate_hdl_golden.py --verify examples/hdl_fixed_golden_vectors.json
python scripts/verify_hdl_parity.py
```

The last command requires Icarus Verilog. Arithmetic and boundary checks do not
establish FPGA synthesis, speed, energy, capacity or hardware performance.

## 10 October 2026 follow-up: exact Integral arithmetic

The signed-rounding helper accepts `numbers.Integral`, which includes NumPy
fixed-width scalar types. Arithmetic on those scalars could overflow before
rounding: `int64(-2**63) / 3` was returned with the wrong sign, and the
`uint64(2**64 - 1) / 3` case returned zero. Both admitted arguments are now
converted to Python integers before negation, offset addition or division.
The quotient/remainder oracle independently checks nearest rounding with ties
away from zero across signed, unsigned and exact-half cases.

All 21 local unittest methods pass, including six new subcases. The ordinary
Python golden and all eight fixed-point steps still verify unchanged. Hosted
CI installs NumPy 2.3.5 solely to exercise its real fixed-width scalar types;
the reference implementation itself gains no NumPy dependency. The original
HDL source and retained golden files are unchanged.
