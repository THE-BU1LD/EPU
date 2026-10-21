"""Frozen EPU E6 real-data GPU study. See PROTOCOL.json."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch


VERSION = "EPU-REALDATA-E6-GPU-R2"
ROOT = Path(__file__).resolve().parent


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_protocol() -> dict:
    raw = (ROOT / "PROTOCOL.json").read_bytes()
    protocol = json.loads(raw)
    if protocol.get("version") != VERSION or protocol.get("repeats") != 20 or protocol.get("folds") != 5:
        raise ValueError("unadmitted protocol")
    if len(protocol.get("seeds", [])) != protocol["repeats"]:
        raise ValueError("seed count differs from repeat count")
    model = protocol.get("model", {})
    if model.get("retrieval_top_k") != 2:
        raise ValueError("unadmitted retrieval_top_k")
    if model.get("original_retrieval_lambda") != 1.0 or model.get("anchor_retrieval_lambda") != 0.1:
        raise ValueError("unadmitted recurrence lambdas")
    return protocol


def require_gpu() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_NO_ADMITTED_GPU: CUDA unavailable")
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    if total < 8 * 1024**3:
        raise RuntimeError("BLOCKED_NO_ADMITTED_GPU: total memory below 8 GiB")
    torch.cuda.set_device(0)
    torch.manual_seed(20261021)
    torch.cuda.manual_seed_all(20261021)
    torch.use_deterministic_algorithms(True)
    return {
        "device": "cuda:0",
        "name": props.name,
        "total_memory_bytes": total,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def unit_center_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("near-constant row cannot be normalized")
    return centered / norms


def unit_torch(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    if bool(torch.any(norms <= 1e-12)):
        raise ValueError("near-constant tensor row cannot be normalized")
    return centered / norms


def duplicate_groups(X: np.ndarray) -> np.ndarray:
    mapping: dict[bytes, int] = {}
    result = []
    for row in np.asarray(X, dtype=np.float64):
        key = row.astype("<f8", copy=False).tobytes()
        if key not in mapping:
            mapping[key] = len(mapping)
        result.append(mapping[key])
    return np.asarray(result, dtype=np.int64)


def fit_prototypes(X: np.ndarray, y: np.ndarray, per_class: int = 2) -> tuple[np.ndarray, np.ndarray]:
    Xn = unit_center_np(X)
    prototypes, labels = [], []
    for label in sorted(np.unique(y).tolist()):
        points = Xn[y == label]
        mean = points.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(points - mean, axis=1)))
        chosen = [first]
        while len(chosen) < per_class:
            distances = np.min(
                np.stack([np.linalg.norm(points - points[index], axis=1) for index in chosen], axis=1),
                axis=1,
            )
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
        prototypes.extend(points[index] for index in chosen)
        labels.extend([int(label)] * per_class)
    P = unit_center_np(np.asarray(prototypes, dtype=np.float64))
    return P, np.asarray(labels, dtype=np.int64)


def prototype_predict(
    query: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    mode: str,
    device: torch.device,
    model: dict,
) -> tuple[np.ndarray, np.ndarray, int]:
    q = torch.as_tensor(unit_center_np(query), dtype=torch.float64, device=device)
    p = torch.as_tensor(prototypes, dtype=torch.float64, device=device)
    lab = torch.as_tensor(labels, dtype=torch.long, device=device)
    initial = q.clone()
    if mode == "direct_multi":
        state = q
        operations = 1
    else:
        if mode not in {"original_topk", "anchor_topk"}:
            raise ValueError(f"unknown recurrent mode: {mode}")
        retrieval_lambda = (model["original_retrieval_lambda"] if mode == "original_topk"
                            else model["anchor_retrieval_lambda"])
        alpha = float(model["alpha"])
        beta = float(model["beta"])
        top_k = int(model["retrieval_top_k"])
        state = q
        for _ in range(int(model["steps"])):
            scores = state @ p.T
            # A fixed index penalty makes exact-score ties deterministic.
            penalty = torch.arange(p.shape[0], dtype=scores.dtype, device=device) * torch.finfo(scores.dtype).eps
            active = torch.topk(scores - penalty, k=top_k, dim=1, largest=True, sorted=True).indices
            active_scores = torch.gather(scores, 1, active)
            weights = torch.softmax(beta * active_scores, dim=1)
            retrieved = torch.sum(weights.unsqueeze(2) * p[active], dim=1)
            state = unit_torch(
                alpha * state
                + (1.0 - alpha) * ((1.0 - retrieval_lambda) * initial + retrieval_lambda * retrieved)
            )
        operations = int(model["steps"])
    final_scores = state @ p.T
    penalty = torch.arange(p.shape[0], dtype=final_scores.dtype, device=device) * torch.finfo(final_scores.dtype).eps
    prediction = lab[torch.argmax(final_scores - penalty, dim=1)]
    displacement = torch.linalg.vector_norm(state - initial, dim=1)
    return prediction.cpu().numpy(), displacement.cpu().numpy(), operations


def knn3_predict(query: np.ndarray, train: np.ndarray, labels: np.ndarray, device: torch.device) -> np.ndarray:
    q = torch.as_tensor(unit_center_np(query), dtype=torch.float64, device=device)
    x = torch.as_tensor(unit_center_np(train), dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    scores = q @ x.T
    penalty = torch.arange(x.shape[0], dtype=scores.dtype, device=device) * torch.finfo(scores.dtype).eps
    top = torch.topk(scores - penalty, k=3, dim=1, largest=True, sorted=True).indices
    votes = torch.nn.functional.one_hot(y[top], num_classes=10).sum(dim=1)
    return torch.argmax(votes, dim=1).cpu().numpy()


def noisy(X: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise == 0:
        return np.asarray(X, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    return np.clip(np.asarray(X, dtype=np.float64) + rng.normal(0.0, noise, X.shape), 0.0, 1.0)


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunks = []
    batch = 250
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        chunks.append(array[indices].mean(axis=1))
    draws = np.concatenate(chunks)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def permutation_p(values: list[float], permutations: int, seed: int, null: float = 0.0) -> float:
    array = np.asarray(values, dtype=np.float64) - null
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    count = 0
    batch = 1000
    for start in range(0, permutations, batch):
        size = min(batch, permutations - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(array)))
        count += int(np.sum((signs * array).mean(axis=1) >= observed))
    return (count + 1) / (permutations + 1)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    result, running = {}, 0.0
    m = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        result[key] = running
    return result


def metric_row(repeat: int, fold: int, noise: float, method: str, truth: np.ndarray,
               prediction: np.ndarray, displacement: np.ndarray | None, seconds: float) -> dict:
    return {
        "repeat": repeat,
        "fold": fold,
        "noise": noise,
        "method": method,
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "mean_displacement": None if displacement is None else float(np.mean(displacement)),
        "elapsed_seconds": seconds,
    }


def benchmark_gpu(device: torch.device) -> dict:
    a = torch.randn((2048, 2048), device=device)
    b = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    checksum = None
    for _ in range(8):
        checksum = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    operations = 8 * 2 * 2048**3
    return {"elapsed_seconds": elapsed, "estimated_tflops": operations / elapsed / 1e12,
            "checksum": float(checksum[0, 0].item())}


def self_test() -> dict:
    protocol = load_protocol()
    values = np.asarray([[0., 1., 2., 3.], [3., 2., 1., 0.], [0., 2., 0., 2.], [2., 0., 2., 0.]])
    labels = np.asarray([0, 0, 1, 1])
    prototypes, plabels = fit_prototypes(values, labels)
    if prototypes.shape != (4, 4) or plabels.tolist() != [0, 0, 1, 1]:
        raise AssertionError("prototype contract failed")
    groups = duplicate_groups(np.vstack([values, values[0]]))
    if groups[0] != groups[-1] or len(set(groups.tolist())) != 4:
        raise AssertionError("duplicate grouping failed")
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    if not (adjusted["a"] <= adjusted["b"] <= adjusted["c"]):
        raise AssertionError("Holm adjustment failed")
    direct, _, _ = prototype_predict(values, prototypes, plabels, "direct_multi", torch.device("cpu"),
                                     protocol["model"])
    anchored, displacement, operations = prototype_predict(
        values, prototypes, plabels, "anchor_topk", torch.device("cpu"), protocol["model"]
    )
    if direct.shape != anchored.shape or displacement.shape != direct.shape or operations != protocol["model"]["steps"]:
        raise AssertionError("corrected recurrence contract failed")
    return {"status": "PASS_SELF_TEST", "protocol_sha256": sha256_file(ROOT / "PROTOCOL.json"),
            "version": protocol["version"]}


def run(output: Path) -> dict:
    protocol = load_protocol()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    hardware = require_gpu()
    device = torch.device("cuda:0")
    hardware["benchmark"] = benchmark_gpu(device)
    write_new(output / "PROTOCOL.json", protocol)
    write_new(output / "SELF_TEST.json", self_test())

    X_raw, y = load_digits(return_X_y=True)
    if X_raw.shape != (1797, 64) or sorted(np.unique(y).tolist()) != list(range(10)):
        raise ValueError("dataset identity constraints failed")
    X = np.asarray(X_raw, dtype=np.float64) / 16.0
    y = np.asarray(y, dtype=np.int64)
    data_digest = sha256_bytes(X_raw.astype("<f8").tobytes() + y.astype("<i8").tobytes())
    groups = duplicate_groups(X_raw)
    dataset = {"rows": 1797, "features": 64, "classes": list(range(10)),
               "canonical_array_sha256": data_digest, "duplicate_groups": int(len(np.unique(groups))),
               "sklearn": sklearn.__version__}
    write_new(output / "DATASET.json", dataset)
    write_new(output / "ENVIRONMENT.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
        "hardware": hardware, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })

    metrics: list[dict] = []
    sample_correct = {
        (noise, method): np.full((protocol["repeats"], len(y)), np.nan, dtype=np.float64)
        for noise in protocol["noise_standard_deviations"] for method in protocol["methods"]
    }
    sample_displacement = {
        (noise, method): np.full((protocol["repeats"], len(y)), np.nan, dtype=np.float64)
        for noise in protocol["noise_standard_deviations"]
        for method in ("direct_multi", "original_topk", "anchor_topk")
    }
    gpu_prediction_batches = 0
    raw_path = output / "RAW.jsonl.gz"
    with gzip.open(raw_path, "xt", encoding="utf-8", newline="\n") as raw:
        for repeat, seed in enumerate(protocol["seeds"]):
            cv = StratifiedGroupKFold(n_splits=protocol["folds"], shuffle=True, random_state=seed)
            for fold, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
                if set(groups[train_index]) & set(groups[test_index]):
                    raise AssertionError("duplicate feature group leaked across a fold")
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                prototypes, prototype_labels = fit_prototypes(X_train, y_train)
                logreg = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed),
                )
                logreg.fit(X_train, y_train)
                for noise in protocol["noise_standard_deviations"]:
                    noise_seed = int(hashlib.sha256(f"{VERSION}/{repeat}/{fold}/{noise}".encode()).hexdigest()[:16], 16)
                    query = noisy(X_test, noise, noise_seed)
                    predictions: dict[str, np.ndarray] = {}
                    displacements: dict[str, np.ndarray | None] = {}
                    timings: dict[str, float] = {}
                    for method in ("direct_multi", "original_topk", "anchor_topk"):
                        tick = time.perf_counter()
                        prediction, displacement, _ = prototype_predict(
                            query, prototypes, prototype_labels, method, device, protocol["model"]
                        )
                        torch.cuda.synchronize()
                        predictions[method] = prediction
                        displacements[method] = displacement
                        timings[method] = time.perf_counter() - tick
                        gpu_prediction_batches += 1
                    tick = time.perf_counter()
                    predictions["knn3"] = knn3_predict(query, X_train, y_train, device)
                    torch.cuda.synchronize()
                    timings["knn3"] = time.perf_counter() - tick
                    displacements["knn3"] = None
                    gpu_prediction_batches += 1
                    tick = time.perf_counter()
                    predictions["logreg"] = logreg.predict(query)
                    timings["logreg"] = time.perf_counter() - tick
                    displacements["logreg"] = None
                    for method in protocol["methods"]:
                        prediction = predictions[method]
                        metrics.append(metric_row(repeat, fold, noise, method, y_test, prediction,
                                                  displacements[method], timings[method]))
                        sample_correct[(noise, method)][repeat, test_index] = (prediction == y_test).astype(np.float64)
                        if displacements[method] is not None:
                            sample_displacement[(noise, method)][repeat, test_index] = displacements[method]
                    for local, global_index in enumerate(test_index.tolist()):
                        record = {
                            "repeat": repeat, "fold": fold, "row": global_index,
                            "noise": noise, "label": int(y_test[local]),
                            "predictions": {method: int(predictions[method][local]) for method in protocol["methods"]},
                            "displacement": {method: float(displacements[method][local])
                                             for method in ("direct_multi", "original_topk", "anchor_topk")},
                        }
                        raw.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    for key, values in sample_correct.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"missing out-of-fold correctness for {key}")
    for key, values in sample_displacement.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"missing out-of-fold displacement for {key}")

    unique_groups = np.unique(groups)

    def group_contrast(left: str, right: str, selected_noises: list[float], displacement: bool = False) -> list[float]:
        source = sample_displacement if displacement else sample_correct
        result = []
        for group in unique_groups:
            rows = np.flatnonzero(groups == group)
            left_values = np.concatenate([source[(noise, left)][:, rows].reshape(-1) for noise in selected_noises])
            right_values = np.concatenate([source[(noise, right)][:, rows].reshape(-1) for noise in selected_noises])
            result.append(float(left_values.mean() - right_values.mean()))
        return result

    noises = protocol["noise_standard_deviations"]
    effects: dict[str, list[float]] = {
        "H-E6-UTILITY": group_contrast("anchor_topk", "direct_multi", noises),
        "H-E6-ROBUSTNESS": group_contrast("anchor_topk", "direct_multi", [0.3]),
        "H-E6-NONINFERIOR-LOGREG": group_contrast("anchor_topk", "logreg", noises),
        "H-E6-STABILITY": group_contrast("original_topk", "anchor_topk", noises, displacement=True),
    }
    accuracy_keys = ("H-E6-UTILITY", "H-E6-ROBUSTNESS", "H-E6-NONINFERIOR-LOGREG")
    raw_p = {
        key: permutation_p(
            effects[key], protocol["inference"]["paired_sign_permutations"], 7000 + i,
            null=-0.01 if key == "H-E6-NONINFERIOR-LOGREG" else 0.0,
        )
        for i, key in enumerate(accuracy_keys)
    }
    adjusted = holm_adjust(raw_p)
    outcomes = []
    for index, hypothesis in enumerate(protocol["hypotheses"]):
        key = hypothesis["id"]
        effect = float(np.mean(effects[key]))
        lower, upper = bootstrap_ci(effects[key], protocol["inference"]["bootstrap_resamples"], 9000 + index)
        if key in {"H-E6-UTILITY", "H-E6-ROBUSTNESS"}:
            passed = effect >= hypothesis["minimum_effect"] and lower > 0 and adjusted[key] < 0.05
        elif key == "H-E6-NONINFERIOR-LOGREG":
            passed = (effect >= hypothesis["margin"] and lower > hypothesis["margin"]
                      and adjusted[key] < 0.05)
        else:
            passed = effect >= hypothesis["minimum_effect"] and lower > hypothesis["minimum_effect"]
        outcomes.append({
            "id": key, "effect": effect, "bootstrap_95": [lower, upper],
            "raw_permutation_p": raw_p.get(key), "holm_p": adjusted.get(key),
            "status": "PASS" if passed else "FAIL", "group_effects": effects[key],
        })

    # Fixed shuffled-label negative control on repeat zero, fold zero, noise zero.
    seed = protocol["seeds"][0]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, test_index = next(iter(cv.split(X, y, groups)))
    control_accuracy = []
    for permutation in range(20):
        shuffled = y[train_index].copy()
        np.random.default_rng(50000 + permutation).shuffle(shuffled)
        p, pl = fit_prototypes(X[train_index], shuffled)
        pred, _, _ = prototype_predict(X[test_index], p, pl, "anchor_topk", device, protocol["model"])
        control_accuracy.append(float(accuracy_score(y[test_index], pred)))
        gpu_prediction_batches += 1
    negative_control = {
        "accuracies": control_accuracy,
        "mean_accuracy": float(np.mean(control_accuracy)),
        "threshold": protocol["negative_control"]["pass_threshold_accuracy"],
    }
    negative_control["status"] = "PASS" if negative_control["mean_accuracy"] <= negative_control["threshold"] else "FAIL"

    with (output / "FOLD_METRICS.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["repeat", "fold", "noise", "method", "n", "accuracy", "balanced_accuracy",
                  "macro_f1", "mean_displacement", "elapsed_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(metrics)
    summary = {
        "version": VERSION,
        "evidence_role": protocol["evidence_role"],
        "dataset": dataset,
        "outcomes": outcomes,
        "negative_control": negative_control,
        "inference_unit": protocol["inference"]["unit"],
        "inference_group_count": int(len(unique_groups)),
        "gpu_prediction_batches": gpu_prediction_batches,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_new(output / "HYPOTHESES.json", summary)
    lines = [
        "# EPU E6 real-data GPU results", "", protocol["claim_boundary"], "",
        f"Dataset digest: `{data_digest}`.",
        f"GPU prediction batches: {gpu_prediction_batches}.", "", "## Pre-specified outcomes", "",
    ]
    for item in outcomes:
        lines.append(f"- **{item['id']} — {item['status']}**: effect {item['effect']:.6f}; "
                     f"95% exact-feature-group bootstrap interval "
                     f"[{item['bootstrap_95'][0]:.6f}, {item['bootstrap_95'][1]:.6f}].")
    lines += ["", "## Negative control", "",
              f"Shuffled-label mean accuracy {negative_control['mean_accuracy']:.6f}: {negative_control['status']}.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest_files[str(path.relative_to(output))] = sha256_file(path)
    manifest = {
        "version": VERSION, "status": "COMPLETE", "files": manifest_files,
        "source": {"study.py": sha256_file(Path(__file__)), "PROTOCOL.json": sha256_file(ROOT / "PROTOCOL.json")},
        "records": protocol["repeats"] * len(y) * len(noises),
        "gpu_prediction_batches": gpu_prediction_batches,
        "inference_unit": protocol["inference"]["unit"],
        "inference_group_count": int(len(unique_groups)),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_new(output / "MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    try:
        result = run(args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:
        if args.output and args.output.exists():
            failure = {"version": VERSION, "status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)}
            failure_path = args.output / "FAILURE.json"
            if not failure_path.exists():
                write_new(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
