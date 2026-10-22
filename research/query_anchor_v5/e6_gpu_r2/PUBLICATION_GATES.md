# EPU E6 R2 publication gates

The study is not publication-ready until every required gate below has retained
evidence. A passing model metric cannot waive a failed or missing gate.

| Gate | Current state | Acceptance evidence |
|---|---|---|
| R2 source and protocol identity | PASS | `verify_package.py` returns `PASS_E6_R2_PACKAGE`; receipt hashes match the notebook payload. |
| Private hosted GPU submission | PASS | Kaggle Version 1 is retained at `ryangomez010/epu-e6-real-data-gpu-r2-corrected-20261021`; Internet remained off and free Tesla T4 compute was used. |
| Complete execution | PASS | Kaggle reports `COMPLETE`; the manifest retains 107,820 raw records and 1,500 fold-metric rows. |
| Artifact integrity | PASS | `verify_results.py` matches the archive, output directory, manifest hashes, and submitted source identities. |
| Statistical audit | PASS | Exact-feature-group effects, 10,000-draw intervals, 100,000 sign permutations, Holm values, and decisions replay from raw records. |
| Negative control | PASS WITH PORTABILITY FAILURE | GPU mean 0.111944 and CPU mean 0.100556 both pass the 0.20 ceiling, but individual values do not reproduce exactly. |
| Honest outcome report | PASS | Utility, robustness, and noninferiority FAIL; displacement stability PASS. |
| Writer-disjoint confirmation | SATISFIED BY E7 SUCCESSOR | E7 uses the official writer-independent UCI partition; per-row writer identities remain unavailable. |
| Independent reproduction | PENDING | A second operator/runtime reproduces the bound source, protocol, dataset identity, and result artifacts. |
| Manuscript claim audit | PASS | The final claim matrix maps E6 quantitative claims to the retained verification receipt and preserves all failed hypotheses. |

The only open E6-level publication gate is independent GPU reproduction. E6 is
exploratory real-data evidence; E7 separately supplies writer-independent
partition confirmation while retaining its writer-clustering limitation.
