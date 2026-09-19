#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_golden() -> dict[str, object]:
    ref = load_module("epu_float_reference", ROOT / "code" / "epu_reference.py")
    fixed = load_module("epu_fixed_reference", ROOT / "code" / "epu_fixed_reference.py")

    float_core = ref.EPUCore()
    target = float_core.memories[2][:]
    rng = random.Random(7)
    original_initial = [value + rng.uniform(-1.5, 1.5) for value in target]
    initial_q = [fixed.quantize(value) for value in original_initial]

    float_core.z = [fixed.dequantize(value) for value in initial_q]
    float_core.prev_z = float_core.z[:]
    state_q = initial_q[:]

    steps: list[dict[str, object]] = []
    one_lsb = 1.0 / fixed.SCALE
    for step in range(12):
        fixed_info = fixed.step_q(state_q)
        float_info = float_core.step()
        next_q = list(fixed_info["next_q"])
        errors = [
            abs(fixed.dequantize(next_q[index]) - float_info["next"][index])
            for index in range(4)
        ]
        max_error = max(errors)
        if max_error > one_lsb + 1e-12:
            raise AssertionError(
                f"step {step}: fixed-point trajectory exceeds one-LSB floating tolerance: "
                f"{max_error} > {one_lsb}"
            )
        float_converged = float_info["delta"] <= float_core.threshold
        if bool(fixed_info["converged"]) != bool(float_converged):
            raise AssertionError(
                f"step {step}: convergence mismatch fixed={fixed_info['converged']} "
                f"float={float_converged}"
            )
        steps.append(
            {
                "step": step,
                "state_q": list(state_q),
                "scores_q": list(fixed_info["scores_q"]),
                "shifted_q": list(fixed_info["shifted_q"]),
                "recon_q": list(fixed_info["recon_q"]),
                "next_q": next_q,
                "delta_q": int(fixed_info["delta_q"]),
                "converged": bool(fixed_info["converged"]),
                "max_abs_error_vs_float": max_error,
            }
        )
        state_q = next_q
        if fixed_info["converged"]:
            break

    return {
        "schema_version": "epu.hdl-fixed-golden.v1",
        "seed": 7,
        "format": {
            "word_bits": fixed.WORD_BITS,
            "frac_bits": fixed.FRAC_BITS,
            "scale": fixed.SCALE,
            "rounding": "nearest_ties_away_from_zero",
            "clip_min": fixed.CLIP_MIN_Q / fixed.SCALE,
            "clip_max": fixed.CLIP_MAX_Q / fixed.SCALE,
        },
        "config": {
            "dim": 4,
            "memory_count": 4,
            "beta_num": fixed.BETA_NUM,
            "beta_den": fixed.BETA_DEN,
            "alpha_num": fixed.ALPHA_NUM,
            "alpha_den": fixed.ALPHA_DEN,
            "threshold_q": fixed.THRESHOLD_Q,
            "threshold": fixed.THRESHOLD_Q / fixed.SCALE,
        },
        "memories_q": [list(row) for row in fixed.MEMORIES_Q],
        "initial_float": original_initial,
        "initial_q": initial_q,
        "float_tolerance": one_lsb,
        "steps": steps,
        "step_count": len(steps),
        "final_q": state_q,
        "converged": bool(steps and steps[-1]["converged"]),
    }


def assert_equivalent(expected, actual, path="$", atol=1e-12):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                raise AssertionError(f"{path}: expected {expected}, got {actual}")
            return
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=atol):
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: key mismatch")
        for key in sorted(expected):
            assert_equivalent(expected[key], actual[key], f"{path}.{key}", atol)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length mismatch")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equivalent(left, right, f"{path}[{index}]", atol)
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    payload = build_golden()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    expected = json.loads(args.verify.read_text(encoding="utf-8"))
    assert_equivalent(expected, payload)
    print(
        f"verified {args.verify}: {payload['step_count']} steps, "
        f"fixed-point error <= {payload['float_tolerance']}"
    )


if __name__ == "__main__":
    main()
