# EPU E6 R2 publication gates

The study is not publication-ready until every required gate below has retained
evidence. A passing model metric cannot waive a failed or missing gate.

| Gate | Current state | Acceptance evidence |
|---|---|---|
| R2 source and protocol identity | PASS | `verify_package.py` returns `PASS_E6_R2_PACKAGE`; receipt hashes match the notebook payload. |
| Private hosted GPU submission | BLOCKED | Kaggle accepts a saved GPU version for `ryangomez010/epu-e6-real-data-gpu-r2-corrected-20261021`; Internet remains off and no paid compute is used. |
| Complete execution | PENDING | Kaggle reports `COMPLETE`; `MANIFEST.json` reports the frozen version, 107,820 raw prediction records, and no failure receipt. |
| Artifact integrity | PENDING | Downloaded files match every manifest hash; embedded source hashes match the pre-run receipt. |
| Statistical audit | PENDING | Exact-feature-group effects, 10,000-draw bootstrap intervals, 100,000 sign permutations, and Holm values replay from raw records. |
| Negative control | PENDING | All 20 shuffled-label runs are retained and their mean accuracy is at most 0.20. |
| Honest outcome report | PENDING | Every pre-specified hypothesis is reported as PASS or FAIL, including null and adverse results. |
| Writer-disjoint confirmation | BLOCKED BY DATASET | A separately frozen successor uses a provenance-reviewed dataset with writer identities and a justified writer-level experimental unit. |
| Independent reproduction | PENDING | A second operator/runtime reproduces the bound source, protocol, dataset identity, and result artifacts. |
| Manuscript claim audit | PENDING | Each quantitative claim maps to a verified artifact; exploratory E6 evidence is not described as writer-disjoint, confirmatory, independent, or SOTA evidence. |

The current Optical Digits run can satisfy the engineering, hosted-execution,
artifact, statistical, control, and honest-reporting gates. It cannot satisfy
writer-disjoint confirmation because the packaged dataset does not expose
writer identities.
