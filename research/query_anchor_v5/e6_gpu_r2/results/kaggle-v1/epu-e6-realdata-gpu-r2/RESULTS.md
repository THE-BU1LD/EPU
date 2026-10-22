# EPU E6 real-data GPU results

Exploratory real-data development evidence. The dataset has no writer identifiers, so this is not writer-disjoint confirmation, independent reproduction, SOTA evidence, or unrestricted efficacy evidence.

Dataset digest: `f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443`.
GPU prediction batches: 1220.

## Pre-specified outcomes

- **H-E6-UTILITY — FAIL**: effect 0.000000; 95% exact-feature-group bootstrap interval [0.000000, 0.000000].
- **H-E6-ROBUSTNESS — FAIL**: effect 0.000000; 95% exact-feature-group bootstrap interval [0.000000, 0.000000].
- **H-E6-NONINFERIOR-LOGREG — FAIL**: effect -0.163077; 95% exact-feature-group bootstrap interval [-0.179568, -0.147031].
- **H-E6-STABILITY — PASS**: effect 0.656794; 95% exact-feature-group bootstrap interval [0.653628, 0.660003].

## Negative control

Shuffled-label mean accuracy 0.111944: PASS.
