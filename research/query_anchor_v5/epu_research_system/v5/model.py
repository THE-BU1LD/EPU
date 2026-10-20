"""Numerically strict associative-memory components for synthetic successor V5."""
from __future__ import annotations

from dataclasses import dataclass
import math


def _vector(values):
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("finite nonempty vector required")
    return vector


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("dimension mismatch")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def squared_distance(left, right):
    if len(left) != len(right):
        raise ValueError("dimension mismatch")
    return math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def unit(values):
    vector = _vector(values)
    length = math.sqrt(dot(vector, vector))
    if length <= 1e-15:
        raise ValueError("zero-norm vector is outside the V5 synthetic contract")
    return tuple(value / length for value in vector)


def softmax(scores, beta):
    if not scores or not math.isfinite(beta) or beta <= 0:
        raise ValueError("nonempty scores and finite positive beta required")
    scaled = [beta * value for value in scores]
    high = max(scaled)
    weights = [math.exp(value - high) for value in scaled]
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def _mean(rows):
    return tuple(math.fsum(row[index] for row in rows) / len(rows) for index in range(len(rows[0])))


def _deterministic_kmeans(rows, k, max_iterations=100):
    """Train-only deterministic farthest-first spherical k-means."""
    unique = sorted(set(tuple(row) for row in rows))
    if k < 1 or k > len(unique):
        raise ValueError("prototype count exceeds distinct rows")
    centers = [unique[0]]
    while len(centers) < k:
        candidate = max(unique, key=lambda row: (min(squared_distance(row, center) for center in centers), row))
        centers.append(candidate)
    assignments = None
    for _ in range(max_iterations):
        updated_assignments = tuple(
            min(range(k), key=lambda index: (squared_distance(row, centers[index]), index)) for row in rows
        )
        if updated_assignments == assignments:
            break
        assignments = updated_assignments
        updated = []
        for index in range(k):
            cluster = [row for row, assignment in zip(rows, assignments, strict=True) if assignment == index]
            if not cluster:
                raise ValueError("empty deterministic cluster")
            updated.append(unit(_mean(cluster)))
        centers = updated
    return tuple(centers)


@dataclass(frozen=True)
class MemoryModel:
    keys: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    alpha: float = 0.6
    beta: float = 6.0
    steps: int = 8

    def __post_init__(self):
        if not self.keys or len(self.keys) != len(self.labels):
            raise ValueError("aligned nonempty keys and labels required")
        width = len(self.keys[0])
        if width < 1 or any(len(key) != width for key in self.keys):
            raise ValueError("consistent nonempty vector width required")
        if not all(all(math.isfinite(value) for value in key) for key in self.keys):
            raise ValueError("finite keys required")
        if not (0 <= self.alpha < 1) or not math.isfinite(self.beta) or self.beta <= 0 or self.steps < 0:
            raise ValueError("invalid recurrence parameters")

    def _query(self, query):
        query = _vector(query)
        if len(query) != len(self.keys[0]):
            raise ValueError("query dimension mismatch")
        return query

    def _scores(self, query):
        return tuple(dot(key, query) for key in self.keys)

    def _decode(self, query):
        scores = self._scores(query)
        ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        winner = ordered[0]
        margin = scores[winner] - scores[ordered[1]] if len(ordered) > 1 else float("inf")
        return self.labels[winner], winner, margin, scores

    def direct(self, query):
        query = self._query(query)
        prediction, winner, margin, scores = self._decode(query)
        return {"prediction": prediction, "initial_prediction": prediction, "winner": winner,
                "margin": margin, "scores": scores, "steps": 0,
                "query_displacement": 0.0, "prediction_changed": False,
                "score_evaluations": len(self.keys)}

    def recurrent(self, query, *, retrieval_lambda, top_k=None):
        """Run z' = alpha*z + (1-alpha)[(1-lambda)q + lambda*r(z)]."""
        query = self._query(query)
        if not math.isfinite(retrieval_lambda) or not 0 <= retrieval_lambda <= 1:
            raise ValueError("retrieval_lambda must lie in [0, 1]")
        initial_prediction = self._decode(query)[0]
        state = query
        trace = []
        score_evaluations = 0
        for iteration in range(self.steps):
            scores = self._scores(state)
            score_evaluations += len(scores)
            active = tuple(range(len(scores)))
            if top_k is not None:
                if not 1 <= top_k <= len(scores):
                    raise ValueError("invalid top_k")
                active = tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k])
            weights = softmax([scores[index] for index in active], self.beta)
            reconstruction = tuple(
                math.fsum(weight * self.keys[index][dimension]
                          for index, weight in zip(active, weights, strict=True))
                for dimension in range(len(state))
            )
            state = tuple(
                self.alpha * previous + (1.0 - self.alpha)
                * ((1.0 - retrieval_lambda) * original + retrieval_lambda * retrieved)
                for previous, original, retrieved in zip(state, query, reconstruction, strict=True)
            )
            trace.append({"iteration": iteration + 1, "active": active, "weights": weights, "state": state})
        prediction, winner, margin, scores = self._decode(state)
        score_evaluations += len(scores)
        return {"prediction": prediction, "initial_prediction": initial_prediction, "winner": winner,
                "margin": margin, "scores": scores, "steps": self.steps,
                "query_displacement": math.sqrt(squared_distance(query, state)),
                "prediction_changed": prediction != initial_prediction, "trace": trace,
                "score_evaluations": score_evaluations}

    def knn(self, query, k=3):
        query = self._query(query)
        if not 1 <= k <= len(self.keys):
            raise ValueError("invalid k")
        neighbors = sorted(
            range(len(self.keys)),
            key=lambda index: (squared_distance(query, self.keys[index]), index),
        )[:k]
        votes = {}
        for index in neighbors:
            votes[self.labels[index]] = votes.get(self.labels[index], 0) + 1
        prediction = min(votes, key=lambda label: (-votes[label], label))
        return {"prediction": prediction, "initial_prediction": prediction, "neighbors": tuple(neighbors),
                "steps": 0, "query_displacement": 0.0, "prediction_changed": False,
                "score_evaluations": len(self.keys)}


def fit_memory_bank(rows, labels, prototypes_per_class=1, **parameters):
    if len(rows) != len(labels) or not rows:
        raise ValueError("aligned nonempty training data required")
    normalized = tuple(unit(row) for row in rows)
    width = len(normalized[0])
    if any(len(row) != width for row in normalized):
        raise ValueError("inconsistent training width")
    keys, out_labels = [], []
    for label in sorted(set(labels)):
        group = tuple(row for row, value in zip(normalized, labels, strict=True) if value == label)
        count = min(prototypes_per_class, len(set(group)))
        centers = (unit(_mean(group)),) if count == 1 else _deterministic_kmeans(group, count)
        keys.extend(centers)
        out_labels.extend([int(label)] * len(centers))
    return MemoryModel(tuple(keys), tuple(out_labels), **parameters)
