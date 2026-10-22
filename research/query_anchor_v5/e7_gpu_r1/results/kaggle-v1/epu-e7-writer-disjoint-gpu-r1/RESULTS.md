# EPU E7 writer-disjoint real-data GPU results

Writer-disjoint partition confirmation on the official UCI split. It is not writer-cluster-adjusted inference, independent reproduction, SOTA evidence, or unrestricted efficacy evidence.

Dataset digest: `ff5bdca1a597fde4361854eeec741dc671a97633a1de46f1ac23fc173f95306b`.
GPU prediction batches: 260.

## Pre-specified outcomes

- **H-E7-UTILITY — FAIL**: effect 0.000000; 95% exact-feature-group bootstrap interval [0.000000, 0.000000].
- **H-E7-ROBUSTNESS — FAIL**: effect 0.000000; 95% exact-feature-group bootstrap interval [0.000000, 0.000000].
- **H-E7-NONINFERIOR-LOGREG — FAIL**: effect -0.263263; 95% exact-feature-group bootstrap interval [-0.284808, -0.241727].
- **H-E7-STABILITY — PASS**: effect 0.690553; 95% exact-feature-group bootstrap interval [0.687237, 0.693891].

## Negative control

Shuffled-label mean accuracy 0.087869: PASS.
