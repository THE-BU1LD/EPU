# EPU V5 synthetic successor results

## Evidence role

Exploratory known-mechanism simulations; informed by V1-V4; no digit efficacy or confirmation claim.

V5 was designed after inspecting V1-V4 and is therefore exploratory. Its hypotheses were fixed before V5 execution, but they are not confirmatory.

## Method matrix

| Family | Method | Accuracy | Displacement | Change rate | Score evaluations |
|---|---|---:|---:|---:|---:|
| distractor | direct_mean | 0.959 | 0.000 | 0.000 | 2 |
| distractor | original_mean | 0.959 | 0.783 | 0.000 | 18 |
| distractor | anchor_mean | 0.959 | 0.077 | 0.000 | 18 |
| distractor | direct_multi | 0.968 | 0.000 | 0.000 | 4 |
| distractor | original_topk | 0.968 | 0.241 | 0.000 | 36 |
| distractor | anchor_topk | 0.968 | 0.020 | 0.000 | 36 |
| distractor | direct_full | 0.961 | 0.000 | 0.000 | 96 |
| distractor | knn | 0.977 | 0.000 | 0.000 | 96 |
| multimodal | direct_mean | 0.497 | 0.000 | 0.000 | 2 |
| multimodal | original_mean | 0.497 | 0.896 | 0.000 | 18 |
| multimodal | anchor_mean | 0.497 | 0.087 | 0.000 | 18 |
| multimodal | direct_multi | 1.000 | 0.000 | 0.000 | 4 |
| multimodal | original_topk | 1.000 | 0.140 | 0.000 | 36 |
| multimodal | anchor_topk | 1.000 | 0.013 | 0.000 | 36 |
| multimodal | direct_full | 1.000 | 0.000 | 0.000 | 96 |
| multimodal | knn | 1.000 | 0.000 | 0.000 | 96 |
| unimodal | direct_mean | 1.000 | 0.000 | 0.000 | 2 |
| unimodal | original_mean | 1.000 | 0.188 | 0.000 | 18 |
| unimodal | anchor_mean | 1.000 | 0.019 | 0.000 | 18 |
| unimodal | direct_multi | 1.000 | 0.000 | 0.000 | 4 |
| unimodal | original_topk | 1.000 | 0.191 | 0.000 | 36 |
| unimodal | anchor_topk | 1.000 | 0.016 | 0.000 | 36 |
| unimodal | direct_full | 1.000 | 0.000 | 0.000 | 96 |
| unimodal | knn | 1.000 | 0.000 | 0.000 | 96 |

## V5 hypothesis checks

| ID | Metric comparison | Required delta | Observed delta | Status |
|---|---|---:|---:|---|
| H-V5-REPRESENTATION | direct_multi - direct_mean (mean_accuracy, multimodal) | 0.250 | 0.503 | PASS |
| H-V5-ANCHOR-STABILITY | original_mean - anchor_mean (mean_displacement, distractor) | 0.100 | 0.706 | PASS |
| H-V5-RECURRENT-UTILITY | anchor_topk - direct_multi (mean_accuracy, distractor) | 0.010 | 0.000 | FAIL |
| H-V5-FULL-BANK | knn - direct_multi (mean_accuracy, distractor) | 0.050 | 0.010 | FAIL |

## Interpretation

The corrected original recurrence uses retrieval lambda 1.0. The anchored recurrence uses lambda 0.1. Direct and recurrent comparisons now share the same bank, while separate mean, multi-prototype, and full-bank controls expose representation effects.

A failed recurrence-utility check is retained as negative evidence. Passing representation or stability checks cannot establish digit efficacy, novelty, or publication readiness.

## Provenance

- Manifest SHA-256: `792ec09e196938220251b0ac477610281d4eee866c7dba2c7a0439a51d767270`
- Raw records: `27648`
- Config SHA-256: `a8ca21fbd082ff998fa86869ab5b0ac273609c769efc105fd7adbcf881a5f246`
