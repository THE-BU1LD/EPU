"""Fixed-width Integral inputs must use unbounded reference arithmetic."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("epu_fixed_integer_test", ROOT / "code/epu_fixed_reference.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IntegralRoundingTests(unittest.TestCase):
    def test_machine_width_integer_arithmetic_is_promoted_before_rounding(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy provides fixed-width integer protocol implementations")
        cases = [
            (np.int64(-(2**63)), np.int64(3)),
            (np.int64(2**63 - 1), np.int64(3)),
            (np.uint64(2**64 - 1), np.uint64(3)),
            (np.int64(-5), np.int64(2)),
            (np.int64(5), np.int64(2)),
            (np.int32(2**31 - 1), np.int32(10)),
        ]
        for numerator, denominator in cases:
            with self.subTest(numerator=int(numerator), denominator=int(denominator)):
                # Independent quotient/remainder oracle for nearest, ties away.
                whole, remainder = divmod(abs(int(numerator)), int(denominator))
                expected = (whole + (2 * remainder >= int(denominator))) * (1 if numerator >= 0 else -1)
                actual = MODULE.round_div_signed(numerator, denominator)
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), int)


if __name__ == "__main__":
    unittest.main()
