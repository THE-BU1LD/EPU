from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_hdl_parity as parity


class HDLTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.golden = json.loads(parity.GOLDEN.read_text(encoding="utf-8"))
        self.lines = [
            "RESULT "
            + str(row["step"])
            + " "
            + " ".join(str(value) for value in row["next_q"])
            + " "
            + str(int(row["converged"]))
            for row in self.golden["steps"]
        ]

    def test_exact_retained_transcript_is_accepted(self) -> None:
        transcript = "VCD info: trace opened\n" + "\n".join(self.lines) + "\n"
        parity.verify_simulator_output(transcript, self.golden)

    def test_earlier_wrong_result_cannot_be_overwritten_by_correct_duplicate(self) -> None:
        lines = ["RESULT 0 999 999 999 999 0", *self.lines]
        with self.assertRaisesRegex(AssertionError, "HDL state"):
            parity.verify_simulator_output("\n".join(lines), self.golden)

    def test_even_identical_duplicate_steps_invalidate_transcript(self) -> None:
        lines = [self.lines[0], *self.lines]
        with self.assertRaisesRegex(AssertionError, "duplicate or reordered"):
            parity.verify_simulator_output("\n".join(lines), self.golden)

    def test_out_of_order_steps_are_rejected(self) -> None:
        lines = self.lines.copy()
        lines[0], lines[1] = lines[1], lines[0]
        with self.assertRaisesRegex(AssertionError, "expected step 0, got 1"):
            parity.verify_simulator_output("\n".join(lines), self.golden)

    def test_extra_or_missing_steps_are_rejected(self) -> None:
        for lines, message in (
            (self.lines + [self.lines[-1]], "unexpected extra"),
            (self.lines[:-1], "simulator rows"),
            ([], "simulator rows"),
        ):
            with self.subTest(message=message, count=len(lines)):
                with self.assertRaisesRegex(AssertionError, message):
                    parity.verify_simulator_output("\n".join(lines), self.golden)

    def test_malformed_or_unknown_result_cannot_hide_beside_valid_rows(self) -> None:
        for invalid in (
            "RESULT 0 x 0 0 0 0",
            "RESULT 0 0 0 0 0 x",
            "RESULT 0 0 0 0 0 0 trailing",
            "RESULT -1 0 0 0 0 0",
            "RESULT broken",
            "RESULT 999 0 0 0 0 0",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                parity.verify_simulator_output(
                    "\n".join([invalid, *self.lines]), self.golden
                )

    def test_runtime_error_diagnostic_invalidates_otherwise_matching_rows(self) -> None:
        for prefix in ("ERROR:", "FATAL:"):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(
                AssertionError, "simulator diagnostic"
            ):
                parity.verify_simulator_output(
                    "\n".join([prefix + " violated arithmetic assertion", *self.lines]),
                    self.golden,
                )

    def test_convergence_is_checked_independently_of_state(self) -> None:
        lines = self.lines.copy()
        fields = lines[0].split()
        fields[-1] = str(1 - int(fields[-1]))
        lines[0] = " ".join(fields)
        with self.assertRaisesRegex(AssertionError, "converged"):
            parity.verify_simulator_output("\n".join(lines), self.golden)

    def test_main_cannot_print_success_for_invalid_simulator_output(self) -> None:
        transcript = "\n".join(["RESULT 0 999 999 999 999 0", *self.lines])
        output = io.StringIO()
        with (
            patch.object(parity.shutil, "which", return_value="/fixture/simulator"),
            patch.object(
                parity.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 0, transcript, ""),
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaisesRegex(AssertionError, "HDL state"):
                parity.main()
        self.assertNotIn("verified HDL parity", output.getvalue())

    def test_main_rejects_runtime_error_on_stderr_even_when_exit_status_is_zero(self) -> None:
        for diagnostic in ("ERROR: arithmetic failed", "FATAL: assertion failed"):
            output = io.StringIO()
            with (
                self.subTest(diagnostic=diagnostic),
                patch.object(parity.shutil, "which", return_value="/fixture/simulator"),
                patch.object(
                    parity.subprocess,
                    "run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, "", ""),
                        subprocess.CompletedProcess([], 0, "\n".join(self.lines), diagnostic),
                    ],
                ),
                contextlib.redirect_stdout(output),
            ):
                with self.assertRaisesRegex(AssertionError, "diagnostic on stderr"):
                    parity.main()
                self.assertNotIn("verified HDL parity", output.getvalue())


if __name__ == "__main__":
    unittest.main()
