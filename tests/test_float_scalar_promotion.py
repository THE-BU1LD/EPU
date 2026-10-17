"""Dependency-free simulations of admitted fixed-width Real scalars."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from epu_reference import EPUCore, dot, l1, normalize


class Signed64(int):
    def __new__(cls, value):
        return super().__new__(cls, (int(value) + 2**63) % 2**64 - 2**63)

    def __sub__(self, other):
        return Signed64(int(self) - int(other))

    def __mul__(self, other):
        return Signed64(int(self) * int(other))


class NarrowFloat(float):
    """A real scalar whose native product overflows above binary16's range."""

    def __mul__(self, other):
        value = float(self) * float(other)
        return math.copysign(math.inf, value) if abs(value) > 65504 else value


class FloatScalarPromotionTests(unittest.TestCase):
    def test_shifted_normalization_does_not_wrap_fixed_width_subtraction(self):
        found = normalize([Signed64(-2**63), Signed64(2**63-1)])
        self.assertGreater(found[1], 0.999999)
        self.assertGreater(found[0], 0)
        self.assertLess(found[0], 1e-18)

    def test_dot_does_not_use_fixed_width_integer_products(self):
        self.assertEqual(dot([Signed64(2**32)], [Signed64(2**32)]), float(2**64))

    def test_dot_promotes_low_precision_float_before_product(self):
        self.assertEqual(dot([NarrowFloat(300)], [NarrowFloat(300)]), 90000.0)

    def test_l1_does_not_wrap_fixed_width_differences(self):
        self.assertEqual(l1([Signed64(-2**63)], [Signed64(2**63-1)]), 2**64-1)

    def test_scores_promote_low_precision_beta_before_product(self):
        core = EPUCore(dim=1, beta=NarrowFloat(300), memories=[[300.0]], z=[300.0])
        self.assertEqual(core.scores(), [27000000.0])

    def test_numpy_float16_complete_step_uses_promoted_arithmetic(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy provides actual low-precision scalar implementations")
        with np.errstate(over="raise", invalid="raise"):
            core = EPUCore(dim=1, beta=1, alpha=np.float16(.6),
                           memories=[[np.float16(65504)]], z=[np.float16(65504)])
            info = core.step()
        self.assertEqual(info["recon"], [65504.0])
        self.assertEqual(info["next"], [127.0])
        self.assertEqual(info["delta"], 65377.0)
        alpha = np.float16(.6)
        core = EPUCore(dim=1, beta=1, alpha=alpha,
                       memories=[[np.float16(1)], [np.float16(3)]], z=[np.float16(2)])
        info = core.step()
        # Scores [2, 6] yield weights [1/6, 5/6] and reconstruction 8/3.
        self.assertAlmostEqual(float(info["recon"][0]), 8/3, places=14)
        self.assertAlmostEqual(info["next"][0],
                               float(alpha)*2 + (1-float(alpha))*8/3, places=14)

    def test_ordinary_python_arithmetic_and_golden_scale_are_unchanged(self):
        self.assertEqual(dot([2, 1, 0, 1], [1, -2, 2, 1]), 1)
        self.assertEqual(normalize([0, 1, 2]), [1/6, 2/6, 3/6])
