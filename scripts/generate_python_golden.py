#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "code" / "epu_reference.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("epu_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module from {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_golden()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
