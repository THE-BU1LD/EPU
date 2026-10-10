"""Check the declared Integral contract without adding a NumPy dependency."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "code" / "epu_fixed_reference.py"
SPEC = importlib.util.spec_from_file_location("epu_rounding_reference", SOURCE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class Signed64(int):
    """An Integral with the same wrapping operations as a signed 64-bit scalar."""

    def __new__(cls, value):
        return super().__new__(cls, (int(value) + 2**63) % 2**64 - 2**63)

    def __add__(self, other):
        return Signed64(int(self) + int(other))

    def __neg__(self):
        return Signed64(-int(self))

    def __floordiv__(self, other):
        return Signed64(int(self) // int(other))


class IntegralRoundingTests(unittest.TestCase):
    def test_large_positive_rounding_does_not_wrap_sign(self):
        self.assertEqual(
            REFERENCE.round_div_signed(Signed64(2**63 - 1), Signed64(2)), 2**62
        )

    def test_minimum_signed_integer_is_not_negated_in_fixed_width(self):
        self.assertEqual(
            REFERENCE.round_div_signed(Signed64(-(2**63)), Signed64(3)),
            -3074457345618258603,
        )

    def test_large_denominator_does_not_overflow_rounding_offset(self):
        self.assertEqual(
            REFERENCE.round_div_signed(Signed64(2**63 - 1), Signed64(2**63 - 1)), 1
        )

    def test_result_has_unbounded_python_integer_semantics(self):
        result = REFERENCE.round_div_signed(Signed64(17), Signed64(2))
        self.assertIs(type(result), int)
        self.assertEqual(result, 9)

    def test_arbitrary_precision_and_signed_half_ties_remain_exact(self):
        for numerator, denominator, expected in [
            (10**100 + 1, 2, 5 * 10**99 + 1),
            (5, 2, 3),
            (-5, 2, -3),
            (4, 3, 1),
            (-4, 3, -1),
        ]:
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertEqual(
                    REFERENCE.round_div_signed(numerator, denominator), expected
                )
