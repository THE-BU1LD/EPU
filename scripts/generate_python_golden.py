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
REFERENCE_PATH = ROOT / "code" / "epu_reference.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("epu_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module from {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_golden():
    ref = load_reference_module()
    core = ref.EPUCore()
    target = core.memories[2][:]
    rng = random.Random(7)
    initial = [x + rng.uniform(-1.5, 1.5) for x in target]
    core.z = initial[:]

    history = []
    for _ in range(12):
        history.append(core.step())
        if core.converged():
            break

    return {
        "schema_version": "epu.python-golden.v1",
        "seed": 7,
        "config": {
            "dim": core.dim,
            "beta": core.beta,
            "alpha": core.alpha,
            "gamma": core.gamma,
            "threshold": core.threshold,
        },
        "memories": core.memories,
        "target": target,
        "initial": initial,
        "steps": history,
        "converged": core.converged(),
        "step_count": len(history),
        "final_state": core.z,
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
            raise AssertionError(
                f"{path}: expected {expected!r}, got {actual!r}, atol={atol}"
            )
        return

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise AssertionError(
                f"{path}: key mismatch expected={sorted(expected)} actual={sorted(actual)}"
            )
        for key in sorted(expected):
            assert_equivalent(expected[key], actual[key], f"{path}.{key}", atol)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path}: length mismatch expected={len(expected)} actual={len(actual)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equivalent(left, right, f"{path}[{index}]", atol)
        return

    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    payload = build_golden()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    expected = json.loads(args.verify.read_text(encoding="utf-8"))
    assert_equivalent(expected, payload, atol=args.atol)
    print(f"verified {args.verify} against current Python reference (atol={args.atol})")


if __name__ == "__main__":
    main()
