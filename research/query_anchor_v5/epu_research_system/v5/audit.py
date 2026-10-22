"""Adversarial structural and deterministic replay audit for a V5 run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from .run import METHODS, digest, evaluate_hypotheses, run as execute, source_hashes, summarize, write_json_new


def audit(run, output):
    run, output = Path(run), Path(output)
    manifest = json.loads((run / "MANIFEST.json").read_text(encoding="utf-8"))
    config = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(run / name) != expected:
            raise ValueError(f"manifest mismatch: {name}")
    if manifest["source_files"] != source_hashes():
        raise ValueError("V5 source differs from the executed source manifest")
    records = [json.loads(line) for line in (run / "RAW.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_count = len(config["families"]) * len(config["seeds"]) * (2 * config["test_per_class"]) * len(METHODS)
    if len(records) != expected_count or manifest["records"] != expected_count:
        raise ValueError("record cardinality mismatch")
    keys = {(row["family"], row["seed"], row["example"], row["method"]) for row in records}
    if len(keys) != expected_count:
        raise ValueError("duplicate or missing method/example cells")
    expected_keys = {(family, seed, example, method) for family in config["families"] for seed in config["seeds"] for example in range(2 * config["test_per_class"]) for method in METHODS}
    if keys != expected_keys:
        raise ValueError("Cartesian coverage mismatch")
    for row in records:
        if row["correct"] != (row["label"] == row["prediction"]):
            raise ValueError("incorrect correctness flag")
        for field in ("query_displacement", "score_evaluations"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"invalid {field}")
    _, aggregate = summarize(records)
    if aggregate != json.loads((run / "AGGREGATE.json").read_text(encoding="utf-8"))["rows"]:
        raise ValueError("aggregate does not regenerate")
    outcomes = evaluate_hypotheses(aggregate, config["hypotheses"])
    if outcomes != json.loads((run / "HYPOTHESES.json").read_text(encoding="utf-8"))["outcomes"]:
        raise ValueError("hypotheses do not regenerate")
    report_receipt = json.loads((run / "report" / "RECEIPT.json").read_text(encoding="utf-8"))
    if report_receipt.get("input_manifest_sha256") != digest(run / "MANIFEST.json"):
        raise ValueError("report detached from run manifest")
    for name, expected in report_receipt["outputs"].items():
        if digest(run / "report" / name) != expected:
            raise ValueError(f"report output changed: {name}")
    with tempfile.TemporaryDirectory() as temporary:
        replay = Path(temporary) / "replay"
        execute(run / "CONFIG.json", replay)
        for name in ("RAW.jsonl", "PER_SEED.csv", "AGGREGATE.json", "HYPOTHESES.json"):
            if digest(replay / name) != digest(run / name):
                raise ValueError(f"deterministic replay mismatch: {name}")
    outcome = {"status": "PASS_V5_ADVERSARIAL_AUDIT", "run_manifest_sha256": digest(run / "MANIFEST.json"),
               "records": expected_count, "cartesian_cells": len(keys), "source_files_verified": len(manifest["source_files"]),
               "deterministic_replay": True, "evidence_role": config["evidence_role"],
               "claim_boundary": "Exploratory synthetic mechanism evidence only; no real-data efficacy, confirmation, novelty, or submission claim."}
    write_json_new(output, outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(audit(parser.parse_args().run, parser.parse_args().output), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.run, args.output), sort_keys=True))
