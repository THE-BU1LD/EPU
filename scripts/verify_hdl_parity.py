#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from generate_hdl_golden import assert_equivalent, build_golden

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples" / "hdl_fixed_golden_vectors.json"
RESULT_RE = re.compile(
    r"^RESULT\s+(?P<step>\d+)\s+"
    r"(?P<z0>-?\d+)\s+(?P<z1>-?\d+)\s+(?P<z2>-?\d+)\s+(?P<z3>-?\d+)\s+"
    r"(?P<converged>[01])$"
)


def build_testbench(golden: dict[str, object]) -> str:
    lines = [
        "`timescale 1ns/1ps",
        "module tb;",
        "  localparam integer D = 4;",
        "  localparam integer W = 16;",
        "  reg clk;",
        "  reg rst;",
        "  reg step_en;",
        "  reg signed [D*W-1:0] z_in_flat;",
        "  wire signed [D*W-1:0] z_out_flat;",
        "  wire converged;",
        "",
        "  epu_core #(.D(4), .K(4), .W(16), .FRAC(8)) dut (",
        "    .clk(clk), .rst(rst), .step_en(step_en),",
        "    .z_in_flat(z_in_flat), .z_out_flat(z_out_flat), .converged(converged)",
        "  );",
        "",
        "  initial begin",
        "    clk = 0; rst = 1; step_en = 0; z_in_flat = 0;",
        "    #1; clk = 1; #1; clk = 0; rst = 0; #1;",
    ]
    for row in golden["steps"]:
        step = int(row["step"])
        state = [int(value) for value in row["state_q"]]
        lines.extend(
            [
                "    z_in_flat = 0;",
                f"    z_in_flat[0*W +: W] = {state[0]};",
                f"    z_in_flat[1*W +: W] = {state[1]};",
                f"    z_in_flat[2*W +: W] = {state[2]};",
                f"    z_in_flat[3*W +: W] = {state[3]};",
                "    step_en = 1; #1; clk = 1; #1;",
                (
                    f'    $display("RESULT {step} %0d %0d %0d %0d %0d", '
                    "$signed(z_out_flat[0*W +: W]), "
                    "$signed(z_out_flat[1*W +: W]), "
                    "$signed(z_out_flat[2*W +: W]), "
                    "$signed(z_out_flat[3*W +: W]), converged);"
                ),
                "    clk = 0; step_en = 0; #1;",
            ]
        )
    lines.extend(["    $finish;", "  end", "endmodule", ""])
    return "\n".join(lines)


def main() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        raise SystemExit("iverilog and vvp are required for HDL parity verification")

    retained = json.loads(GOLDEN.read_text(encoding="utf-8"))
    regenerated = build_golden()
    assert_equivalent(retained, regenerated)

    with tempfile.TemporaryDirectory(prefix="epu-hdl-") as tmp:
        tmp_path = Path(tmp)
        tb_path = tmp_path / "tb_epu_core.v"
        sim_path = tmp_path / "epu_sim"
        tb_path.write_text(build_testbench(retained), encoding="utf-8")

        compile_result = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-s",
                "tb",
                "-o",
                str(sim_path),
                str(ROOT / "code" / "epu_core.v"),
                str(tb_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise SystemExit(
                "iverilog compile failed:\n"
                + compile_result.stdout
                + compile_result.stderr
            )
        run_result = subprocess.run(
            ["vvp", str(sim_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if run_result.returncode != 0:
            raise SystemExit(
                "HDL simulation failed:\n" + run_result.stdout + run_result.stderr
            )

    observed: dict[int, tuple[list[int], bool]] = {}
    for raw_line in run_result.stdout.splitlines():
        match = RESULT_RE.match(raw_line.strip())
        if not match:
            continue
        step = int(match.group("step"))
        state = [int(match.group(f"z{index}")) for index in range(4)]
        observed[step] = (state, match.group("converged") == "1")

    expected_steps = retained["steps"]
    if len(observed) != len(expected_steps):
        raise AssertionError(
            f"expected {len(expected_steps)} simulator rows, got {len(observed)}\n"
            + run_result.stdout
        )

    for row in expected_steps:
        step = int(row["step"])
        expected_state = [int(value) for value in row["next_q"]]
        expected_converged = bool(row["converged"])
        actual_state, actual_converged = observed[step]
        if actual_state != expected_state:
            raise AssertionError(
                f"step {step}: HDL state {actual_state} != fixed golden {expected_state}"
            )
        if actual_converged != expected_converged:
            raise AssertionError(
                f"step {step}: HDL converged={actual_converged} "
                f"!= fixed golden {expected_converged}"
            )

    print(
        f"verified HDL parity for {len(expected_steps)} retained steps; "
        f"exact fixed-point match and <= {retained['float_tolerance']} vs float reference"
    )


if __name__ == "__main__":
    main()
