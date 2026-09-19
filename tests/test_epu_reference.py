from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from epu_reference import EPUCore, clip, l1, normalize


class EPUReferenceTests(unittest.TestCase):
    def test_normalize_is_nonnegative_and_sums_to_one(self) -> None:
        weights = normalize([3.0, 3.0, -1.0, 0.5])
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(w >= 0.0 for w in weights))
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_equal_scores_normalize_uniformly(self) -> None:
        weights = normalize([2.0, 2.0, 2.0])
        for weight in weights:
            self.assertAlmostEqual(weight, 1.0 / 3.0, places=12)

    def test_clip_and_l1_contracts(self) -> None:
        self.assertEqual(clip([-200.0, 0.0, 200.0]), [-128.0, 0.0, 127.0])
        self.assertEqual(l1([1.0, 2.0], [4.0, -2.0]), 7.0)

    def test_step_preserves_dimensionality_and_finite_values(self) -> None:
        core = EPUCore()
        core.z = [1.5, -1.0, 2.25, 0.5]
        info = core.step()

        self.assertEqual(len(info["state"]), core.dim)
        self.assertEqual(len(info["next"]), core.dim)
        self.assertEqual(len(info["weights"]), len(core.memories))
        self.assertAlmostEqual(sum(info["weights"]), 1.0, places=12)
        self.assertTrue(all(math.isfinite(v) for v in info["next"]))
        self.assertTrue(math.isfinite(info["delta"]))

    def test_run_is_deterministic_for_same_initial_state(self) -> None:
        initial = [1.75, -1.25, 1.5, 0.75]

        left = EPUCore()
        left.z = initial[:]
        left.prev_z = initial[:]
        left_history = left.run(steps=8, verbose=False)

        right = EPUCore()
        right.z = initial[:]
        right.prev_z = initial[:]
        right_history = right.run(steps=8, verbose=False)

        self.assertEqual(left_history, right_history)
        self.assertGreaterEqual(len(left_history), 1)
        self.assertLessEqual(len(left_history), 8)

    def test_default_memory_shapes_match_declared_dimension(self) -> None:
        core = EPUCore()
        self.assertTrue(core.memories)
        self.assertTrue(all(len(memory) == core.dim for memory in core.memories))


if __name__ == "__main__":
    unittest.main()
