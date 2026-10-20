"""Known-mechanism synthetic data families for EPU successor studies."""
from __future__ import annotations

import random


FAMILIES = ("unimodal", "multimodal", "distractor")


def _normal(rng, center, spread):
    return tuple(rng.gauss(value, spread) for value in center)


def make_family(name, seed, train_per_class=48, test_per_class=72, spread=0.28):
    """Return independent train/test pairs with declared representation mechanisms.

    ``unimodal`` favors class means. ``multimodal`` puts each class in two modes,
    making one mean intentionally lossy. ``distractor`` adds a high-density mode
    near the decision boundary. These are simulations, not digit data.
    """
    if name not in FAMILIES:
        raise ValueError("unknown family")
    rng = random.Random(f"epu-synthetic-v1:{name}:{seed}")
    if name == "unimodal":
        centers = {0: ((-1.2, 0.0),), 1: ((1.2, 0.0),)}
    elif name == "multimodal":
        # XOR modes have the same population mean for both labels. A one-mean
        # prototype is therefore intentionally non-identifying, while two
        # class-specific prototypes recover the four actual modes.
        centers = {0: ((-1.15, -1.15), (1.15, 1.15)),
                   1: ((-1.15, 1.15), (1.15, -1.15))}
    else:
        centers = {0: ((-1.15, -0.7), (-0.25, 0.55)), 1: ((1.15, 0.7), (0.25, -0.55))}

    def sample(per_class):
        rows, labels = [], []
        for label in (0, 1):
            modes = centers[label]
            for index in range(per_class):
                rows.append(_normal(rng, modes[index % len(modes)], spread))
                labels.append(label)
        return tuple(rows), tuple(labels)

    train_rows, train_labels = sample(train_per_class)
    test_rows, test_labels = sample(test_per_class)
    return {"train_rows": train_rows, "train_labels": train_labels,
            "test_rows": test_rows, "test_labels": test_labels,
            "mechanism": {"unimodal": "class mean is sufficient",
                          "multimodal": "one prototype per class is intentionally lossy",
                          "distractor": "boundary-adjacent modes test retrieval sensitivity"}[name]}
