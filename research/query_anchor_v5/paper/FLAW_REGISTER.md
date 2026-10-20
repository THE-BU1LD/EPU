# EPU V5 adversarial flaw register

| Flaw | Cause / evidence | Severity | Repair | Acceptance evidence |
|---|---|---:|---|---|
| V4 original recurrence was misimplemented | `anchored=False` removed the query but retained retrieval weight 0.1 | Critical | V5 requires explicit `retrieval_lambda`; original is 1.0 | `test_original_recurrence_uses_full_retrieval_weight`; V5 source manifest |
| V4 mixed representation and recurrence | `anchor_topk` used a four-key bank while its comparator used class means | Critical | Direct and recurrent methods now share mean or multi banks | Eight-method V5 matrix and report |
| V1-V4 informed V5 | Generator and hypotheses were revised after earlier observations | Critical | Mark V5 exploratory everywhere; no confirmatory language | Config `evidence_role`; manuscript; audit |
| Synthetic family makes the representation result easy | XOR means are deliberately non-identifying | High | Treat as a mechanism/construction check only | Multimodal result scoped in paper |
| Recurrence adds no utility | Anchor top-k equals direct multi at 0.968 | Critical | Retain failed one-point screen; prefer direct control | `H-V5-RECURRENT-UTILITY: FAIL` |
| Stability can preserve wrong decisions | Displacement falls without any prediction changes | Critical | Separate stability and utility hypotheses | H-V5 stability pass and utility fail |
| Full-bank threshold fails | kNN gain is 0.010, below required 0.050 | Medium | Retain failure and exact effect | `H-V5-FULL-BANK: FAIL` |
| Previous manifest did not bind checkpoints or source | V4 could not detect source/checkpoint drift | High | V5 binds every checkpoint and executed source | Manifest plus adversarial audit |
| Previous audit lacked cell completeness | Duplicated or missing cells could regenerate an aggregate | High | Verify exact Cartesian key set and record count | 27,648 unique cells audited |
| Same-run regeneration can mask corruption | Rewriting manifest and aggregates can bless altered records | High | Completed directories are immutable; audit replays in temp | Replay byte matches four scientific outputs |
| Bank clustering can leak evaluation information | Prototype choices could be informed by test rows | High | Deterministic clustering sees training rows only | `evaluate_seed` construction and tests |
| Similarity mismatch can confound direct versus kNN | Dot and Euclidean rankings differ if norms vary | High | Unit normalize all V5 rows | Geometry field in each checkpoint |
| More capacity can explain gains | Multi/full banks store more vectors than class means | High | Report bank sizes and same-bank controls | Manifest metadata and score costs |
| Operation count is not energy | Score evaluations ignore hardware/runtime details | Medium | Call it a cost proxy; make no hardware claim | Paper limitations |
| Seeds are not people or sites | Simulation streams lack real acquisition clusters | High | No population p-values or generalization claim | Paper methodology and limitations |
| Digit study remains unrun | E5-R1 admission failed below frozen memory floor | Blocking | Preserve sole replacement attempt until guard passes | Retained refusal receipt |
| Installed digit subset lacks writer IDs | scikit-learn subset cannot establish writer independence | Critical | Future confirmation must use official writer-disjoint source | UCI provenance and protocol |
| No independent reproduction | Same host and same operator performed replay | High | Require external executor and signed receipt | Gate remains open |
| No publication-level novelty review | Literature review is targeted | High | Restrict contribution to source-bound failure analysis | Related-work boundary |
| GitHub publication unavailable | Checkout has no remote; `gh` credentials are invalid | Blocking | Configure exact remote and reauthenticate before push | Remote read-back receipt required |
| Kaggle execution unavailable | Kaggle CLI absent and no authenticated API path verified | Blocking | Build an uploadable capsule; run only after auth | Remote kernel/version/output receipt required |
| Authorship metadata inferred from prior paper | Repository does not contain an authoritative EPU author record | Medium | Use “prepared for Ryan Gomez”; verify before release | Release gate remains open |
