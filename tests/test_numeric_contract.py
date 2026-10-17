"""Boundary regressions; retained golden trajectories remain the parity authority."""

from __future__ import annotations

import math
import sys
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from epu_fixed_reference import CLIP_MAX_Q, CLIP_MIN_Q, quantize, round_div_signed, step_q
from epu_reference import EPUCore, clip, dot, l1, normalize


class NumericContractTests(unittest.TestCase):
    def test_shifted_normalization_handles_finite_subtraction_and_sum_overflow(self):
        for scores in ([-1e308, 1e308, 0.0], [0.0, 1e308, 1e308]):
            with self.subTest(scores=scores), localcontext() as context:
                context.prec = 360
                exact = [Decimal.from_float(value) for value in scores]
                shifted = [value - min(exact) + 1 for value in exact]
                expected = [float(value / sum(shifted)) for value in shifted]
                observed = normalize(scores)
                self.assertTrue(all(math.isfinite(value) and value >= 0 for value in observed))
                self.assertAlmostEqual(sum(observed), 1.0)
                for actual, target in zip(observed, expected):
                    self.assertTrue(math.isclose(actual, target, rel_tol=1e-14, abs_tol=5e-324))

    def test_ordinary_normalization_keeps_original_arithmetic(self):
        scores = [3.0, 3.0, -1.0, 0.5]
        shifted = [value - min(scores) + 1.0 for value in scores]
        self.assertEqual(normalize(scores), [value / sum(shifted) for value in shifted])

    def test_normalization_rejects_nonfinite_and_ambiguous_inputs(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "2", 1j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize([1.0, value])

    def test_vector_operations_reject_silent_zip_truncation(self):
        for operation in (dot, l1):
            with self.subTest(operation=operation.__name__), self.assertRaises(ValueError):
                operation([1.0, 2.0], [1.0])

    def test_nan_cannot_be_clipped_into_a_valid_state(self):
        with self.assertRaises(ValueError):
            clip([float("nan")])

    def test_core_rejects_inconsistent_state_and_memory_dimensions(self):
        for options in ({"dim": 2}, {"z": [1.0]}, {"memories": [[1.0]]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EPUCore(**options)

    def test_custom_consistent_dimension_remains_supported_in_float_reference(self):
        core = EPUCore(dim=2, memories=[[1.0, 0.0], [0.0, 1.0]], z=[0.0, 0.0])
        self.assertEqual(core.step()["next"], [0.2, 0.2])

    def test_mutated_invalid_state_is_rejected_without_updating_previous_state(self):
        core = EPUCore()
        before = core.prev_z[:]
        core.z = [1.0]
        with self.assertRaises(ValueError):
            core.step()
        self.assertEqual(core.prev_z, before)

    def test_invalid_parameters_and_step_budget_are_rejected(self):
        for options in ({"dim": True}, {"dim": 2.5}, {"alpha": 1.1}, {"threshold": -1.0}, {"beta": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EPUCore(**options)
        for steps in (-1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                EPUCore().run(steps=steps, verbose=False)

    def test_quantization_saturates_large_finite_inputs_before_scaling(self):
        self.assertEqual(quantize(1e308), CLIP_MAX_Q)
        self.assertEqual(quantize(-1e308), CLIP_MIN_Q)

    def test_quantization_preserves_half_ties_away_from_zero(self):
        self.assertEqual(quantize(0.5 / 256), 1)
        self.assertEqual(quantize(-0.5 / 256), -1)

    def test_fixed_point_cannot_silently_coerce_fractional_or_out_of_word_states(self):
        for value in (0.5, True, "1", float("nan"), 32768, -32769):
            with self.subTest(value=value), self.assertRaises(ValueError):
                step_q([value, 0, 0, 0])

    def test_full_signed_word_boundaries_are_valid_inputs(self):
        for value in (-32768, 32767):
            result = step_q([value] * 4)
            self.assertTrue(all(CLIP_MIN_Q <= item <= CLIP_MAX_Q for item in result["next_q"]))

    def test_signed_rounding_requires_integer_inputs(self):
        for numerator, denominator in ((0.5, 2), (1, 1.5), (True, 2), (1, 0)):
            with self.subTest(values=(numerator, denominator)), self.assertRaises(ValueError):
                round_div_signed(numerator, denominator)


if __name__ == "__main__":
    unittest.main()
