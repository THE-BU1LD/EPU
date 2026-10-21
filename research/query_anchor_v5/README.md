# EPU query-anchor V5 research package

This package contains a corrected exploratory successor to EPU-QA1 and a full
negative-mechanism manuscript. It is intentionally separate from the frozen E5
and E5-R1 real-data protocols.

## Result

- The prior V4 synthetic implementation did not implement the declared original
  recurrence: its unanchored branch retained retrieval weight 0.1 instead of 1.0.
- V5 fixes the recurrence, separates representation from recurrence, and binds
  every source and checkpoint hash.
- Multi-prototype representation passes the constructed multimodal mechanism
  check: 1.000 versus 0.497 accuracy for class means.
- Query anchoring passes the distractor stability check: mean displacement falls
  from 0.783 to 0.077.
- The central recurrence-utility check fails: anchored top-k and its direct
  same-bank control both score 0.968, while recurrence uses 36 versus 4 score
  evaluations.

This is exploratory synthetic evidence. The real EPU-QA1 digit run remains
`BLOCKED_RESOURCE` and `NOT_EVALUATED`; there is no confirmatory or independent
reproduction claim.

## Reproduce

From repository root:

```sh
python3 -B -m unittest discover -s research/query_anchor_v5/tests -v
python3 -B -m research.query_anchor_v5.run \
  --config research/query_anchor_v5/configs/synthetic_phase_v5.json \
  --output /tmp/epu-v5-replay
```

The full evidence archive includes the raw JSONL, all 24 per-seed checkpoints,
aggregates, reports, and the deterministic replay audit.

## Hosted real-data successors

`e6_gpu_r2/` freezes the corrected exploratory Optical Digits GPU study. Its
Kaggle run remains blocked by the account's two-session batch GPU cap and has no
scientific result yet.

`e7_gpu_r1/` freezes the writer-disjoint confirmation successor using the
official UCI Optical Recognition of Handwritten Digits split: 3,823 training
examples from 30 people and 1,797 test examples from a different 13 people. The
offline notebook embeds the exact CC BY 4.0 source files, corrected query-anchor
recurrence, top-k=2 retrieval, exact-feature-group inference, Holm adjustment,
noninferiority test, and shuffled-label control. The package verifier and
isolated self-test pass; hosted scientific execution is still pending a free
Kaggle GPU slot. Because per-row writer identities are unavailable, its
uncertainty intervals are not adjusted for within-writer clustering.
