"""Immutable, deterministic V5 synthetic successor runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

from epu_research_system.synthetic import FAMILIES, make_family
from .model import fit_memory_bank, unit


METHODS = (
    "direct_mean", "original_mean", "anchor_mean", "direct_multi",
    "original_topk", "anchor_topk", "direct_full", "knn",
)
METRICS = {"mean_accuracy", "mean_displacement", "prediction_change_rate"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def validate_config(config):
    required = {
        "version", "evidence_role", "frozen_before_execution_utc", "families", "seeds",
        "train_per_class", "test_per_class", "spread", "alpha", "beta", "anchor_lambda",
        "steps", "prototypes_per_class", "top_k", "knn_k", "hypotheses",
    }
    if set(config) != required or config["version"] != "EPU-SYNTHETIC-PHASE-V5":
        raise ValueError("invalid frozen V5 config schema")
    if config["evidence_role"] != "EXPLORATORY_SUCCESSOR_INFORMED_BY_V1_V4":
        raise ValueError("V5 must retain its exploratory history")
    if not config["families"] or not set(config["families"]) <= set(FAMILIES):
        raise ValueError("invalid families")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("unique seeds required")
    numeric_positive = ("train_per_class", "test_per_class", "spread", "beta", "steps", "prototypes_per_class", "top_k", "knn_k")
    if any(not isinstance(config[key], (int, float)) or config[key] <= 0 for key in numeric_positive):
        raise ValueError("positive numerical controls required")
    if not 0 <= config["alpha"] < 1 or not 0 <= config["anchor_lambda"] <= 1:
        raise ValueError("invalid recurrence controls")
    for hypothesis in config["hypotheses"]:
        required_h = {"id", "family", "metric", "left", "right", "minimum_delta"}
        if (set(hypothesis) != required_h or hypothesis["family"] not in config["families"]
                or hypothesis["metric"] not in METRICS or hypothesis["left"] not in METHODS
                or hypothesis["right"] not in METHODS or not math.isfinite(hypothesis["minimum_delta"])):
            raise ValueError("invalid hypothesis")


def evaluate_seed(config, family, seed):
    data = make_family(family, seed, config["train_per_class"], config["test_per_class"], config["spread"])
    train_rows = tuple(unit(row) for row in data["train_rows"])
    test_rows = tuple(unit(row) for row in data["test_rows"])
    common = {"alpha": config["alpha"], "beta": config["beta"], "steps": config["steps"]}
    mean_model = fit_memory_bank(train_rows, data["train_labels"], 1, **common)
    multi_model = fit_memory_bank(train_rows, data["train_labels"], config["prototypes_per_class"], **common)
    full_model = fit_memory_bank(train_rows, data["train_labels"], max(data["train_labels"].count(label) for label in set(data["train_labels"])), **common)
    records = []
    for index, (query, label) in enumerate(zip(test_rows, data["test_labels"], strict=True)):
        top_k = min(config["top_k"], len(multi_model.keys))
        predictions = {
            "direct_mean": mean_model.direct(query),
            "original_mean": mean_model.recurrent(query, retrieval_lambda=1.0),
            "anchor_mean": mean_model.recurrent(query, retrieval_lambda=config["anchor_lambda"]),
            "direct_multi": multi_model.direct(query),
            "original_topk": multi_model.recurrent(query, retrieval_lambda=1.0, top_k=top_k),
            "anchor_topk": multi_model.recurrent(query, retrieval_lambda=config["anchor_lambda"], top_k=top_k),
            "direct_full": full_model.direct(query),
            "knn": full_model.knn(query, k=min(config["knn_k"], len(full_model.keys))),
        }
        for method in METHODS:
            prediction = predictions[method]
            records.append({
                "family": family, "seed": seed, "example": index, "label": label,
                "method": method, "prediction": prediction["prediction"],
                "initial_prediction": prediction["initial_prediction"],
                "correct": prediction["prediction"] == label,
                "prediction_changed": prediction["prediction_changed"],
                "steps": prediction["steps"], "query_displacement": prediction["query_displacement"],
                "score_evaluations": prediction["score_evaluations"],
                "margin": prediction.get("margin"),
            })
    metadata = {
        "family": family, "seed": seed, "mechanism": data["mechanism"],
        "train_examples": len(data["train_labels"]), "test_examples": len(data["test_labels"]),
        "mean_bank_size": len(mean_model.keys), "multi_bank_size": len(multi_model.keys),
        "full_bank_size": len(full_model.keys), "geometry": "unit-normalized dot product; kNN uses equivalent Euclidean ranking",
    }
    return records, metadata


def summarize(records):
    cells = {}
    for record in records:
        key = (record["family"], record["seed"], record["method"])
        cell = cells.setdefault(key, {"correct": 0, "count": 0, "displacement": [], "changed": 0, "operations": []})
        cell["correct"] += int(record["correct"])
        cell["count"] += 1
        cell["displacement"].append(record["query_displacement"])
        cell["changed"] += int(record["prediction_changed"])
        cell["operations"].append(record["score_evaluations"])
    per_seed = []
    for (family, seed, method), cell in sorted(cells.items()):
        per_seed.append({
            "family": family, "seed": seed, "method": method,
            "mean_accuracy": cell["correct"] / cell["count"],
            "mean_displacement": statistics.fmean(cell["displacement"]),
            "prediction_change_rate": cell["changed"] / cell["count"],
            "mean_score_evaluations": statistics.fmean(cell["operations"]), "n": cell["count"],
        })
    aggregate = []
    for family in sorted({row["family"] for row in per_seed}):
        for method in METHODS:
            selected = [row for row in per_seed if row["family"] == family and row["method"] == method]
            row = {"family": family, "method": method, "seeds": len(selected)}
            for metric in ("mean_accuracy", "mean_displacement", "prediction_change_rate", "mean_score_evaluations"):
                values = [item[metric] for item in selected]
                row[metric] = statistics.fmean(values)
                row[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"min_{metric}"] = min(values)
                row[f"max_{metric}"] = max(values)
            aggregate.append(row)
    return per_seed, aggregate


def evaluate_hypotheses(aggregate, hypotheses):
    lookup = {(row["family"], row["method"]): row for row in aggregate}
    outcomes = []
    for hypothesis in hypotheses:
        left = lookup[(hypothesis["family"], hypothesis["left"])][hypothesis["metric"]]
        right = lookup[(hypothesis["family"], hypothesis["right"])][hypothesis["metric"]]
        delta = left - right
        outcomes.append({**hypothesis, "observed_left": left, "observed_right": right,
                         "observed_delta": delta,
                         "status": "PASS" if delta >= hypothesis["minimum_delta"] else "FAIL"})
    return outcomes


def source_hashes():
    package = Path(__file__).resolve().parent
    sources = sorted(package.glob("*.py")) + [package.parent / "synthetic.py"]
    return {str(path.relative_to(package.parents[1])): digest(path) for path in sources}


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    stored = output / "CONFIG.json"
    if stored.exists() and json.loads(stored.read_text(encoding="utf-8")) != config:
        raise ValueError("refusing to mix configurations in one run directory")
    if not stored.exists():
        write_json_new(stored, config)
    if (output / "MANIFEST.json").exists():
        manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(output / name) != expected:
                raise ValueError(f"completed run changed: {name}")
        return manifest
    all_records, checkpoint_metadata, checkpoint_hashes = [], [], {}
    for family in config["families"]:
        for seed in config["seeds"]:
            checkpoint = output / "checkpoints" / f"{family}-seed-{seed}.json"
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            else:
                records, metadata = evaluate_seed(config, family, seed)
                payload = {"metadata": metadata, "records": records}
                write_json_new(checkpoint, payload)
            all_records.extend(payload["records"])
            checkpoint_metadata.append(payload["metadata"])
            checkpoint_hashes[str(checkpoint.relative_to(output))] = digest(checkpoint)
    per_seed, aggregate = summarize(all_records)
    hypotheses = evaluate_hypotheses(aggregate, config["hypotheses"])
    write_text_new(output / "RAW.jsonl", "".join(canonical(record) + "\n" for record in all_records))
    with (output / "PER_SEED.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0])); writer.writeheader(); writer.writerows(per_seed)
    write_json_new(output / "AGGREGATE.json", {
        "scope": "Exploratory known-mechanism simulations; informed by V1-V4; no digit efficacy or confirmation claim.",
        "rows": aggregate,
    })
    write_json_new(output / "HYPOTHESES.json", {"pre_specified_for_v5": config["hypotheses"], "outcomes": hypotheses})
    files = {name: digest(output / name) for name in ("CONFIG.json", "RAW.jsonl", "PER_SEED.csv", "AGGREGATE.json", "HYPOTHESES.json")}
    files.update(checkpoint_hashes)
    manifest = {
        "version": config["version"], "evidence_role": config["evidence_role"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": digest(stored),
        "records": len(all_records), "checkpoints": checkpoint_metadata, "files": files,
        "source_files": source_hashes(),
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform()},
    }
    write_json_new(output / "MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
