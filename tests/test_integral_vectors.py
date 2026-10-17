"""Vector identities for Integral scalars with bounded intermediate arithmetic."""

import sys
import unittest
from numbers import Integral
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from epu_reference import EPUCore, dot, l1, normalize


class Signed64(int):
    """Model signed fixed-width operations without a runtime NumPy dependency."""

    def __new__(cls, value):
        return super().__new__(cls, (int(value) + 2**63) % 2**64 - 2**63)

    def __mul__(self, other):
        if not isinstance(other, Integral):
            return NotImplemented
        return Signed64(int(self) * int(other))

    def __sub__(self, other):
        if not isinstance(other, Integral):
            return NotImplemented
        return Signed64(int(self) - int(other))


class IntegralVectorTests(unittest.TestCase):
    def test_dot_product_preserves_integer_multiplication(self):
        actual = dot([Signed64(2**32)], [Signed64(2**32)])
        self.assertEqual(actual, 2**64)
        self.assertIs(type(actual), int)

    def test_distance_preserves_full_integer_separation(self):
        self.assertEqual(l1([Signed64(-2**63)], [Signed64(2**63 - 1)]), 2**64 - 1)

    def test_normalization_does_not_reverse_extreme_score_order(self):
        low, high = -2**63, 2**63 - 1
        # Exact shifted scores are 1 and 2**64; the tiny positive first weight
        # must survive. This is separate from floating score-overflow fixtures.
        expected = [1 / (2**64 + 1), 2**64 / (2**64 + 1)]
        self.assertEqual(normalize([Signed64(low), Signed64(high)]), expected)

    def test_complete_retrieval_uses_the_high_scoring_memory(self):
        core = EPUCore(
            dim=1, beta=1.0, alpha=0.0,
            memories=[[Signed64(-2**32)], [Signed64(2**32)]],
            z=[Signed64(2**32)],
        )
        result = core.step()
        self.assertEqual(result["scores"], [float(-2**64), float(2**64)])
        self.assertEqual(result["next"], [127.0])
        self.assertEqual(core.z, result["next"])

    def test_integer_beta_does_not_reintroduce_wrapping_scores(self):
        core = EPUCore(
            dim=1, beta=Signed64(2**32), alpha=0.0,
            memories=[[-1], [1]], z=[2**32],
        )
        self.assertEqual(core.scores(), [-2**64, 2**64])
        self.assertEqual(core.step()["next"], [1.0])


if __name__ == "__main__":
    unittest.main()
