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

Two frozen real-data GPU successors are complete. E6 R2 provides exploratory
grouped cross-validation evidence; E7 uses the official writer-independent UCI
partition. Both fail utility, robustness, and logistic-regression
noninferiority while passing only the displacement-reduction hypothesis. The
result is a verified negative efficacy finding, without an independent
reproduction claim.

## Reproduce

From repository root:

```sh
PYTHONPATH=research/query_anchor_v5 python3 -B -m unittest discover \
  -s research/query_anchor_v5/epu_research_system/tests -v
PYTHONPATH=research/query_anchor_v5 python3 -B -m epu_research_system.v5.run \
  --config research/query_anchor_v5/epu_research_system/configs/synthetic_phase_v5.json \
  --output /tmp/epu-v5-replay
python3 research/query_anchor_v5/e6_gpu_r2/verify_package.py
python3 research/query_anchor_v5/e6_gpu_r2/verify_results.py
python3 research/query_anchor_v5/e7_gpu_r1/verify_package.py
python3 research/query_anchor_v5/e7_gpu_r1/verify_results.py
cd research/query_anchor_v5/paper
python3 build_assets.py
tectonic main.tex
python3 verify_paper.py
python3 package_release.py
```

The full evidence archive includes V5 raw records and checkpoints plus the E6
and E7 notebooks, protocols, raw GPU outputs, manifests, logs, result verifiers,
and retained-evidence receipts.

The verified PDF, arXiv source bundle, full evidence archive, and release
manifest are under `output/`.
