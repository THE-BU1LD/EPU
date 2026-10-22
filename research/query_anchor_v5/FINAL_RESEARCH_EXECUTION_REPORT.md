# Final research execution report

## V5 successor correction — 21 October 2026

The V4 section below is retained as historical evidence, but its recurrence
comparison is superseded. An audit found that V4's nominally original
recurrence still multiplied retrieval by the anchored coefficient 0.1 and
compared a multi-prototype recurrent method against a mean-only direct method.
V5 corrects both defects with an explicit retrieval coefficient (original
lambda=1; anchored lambda=0.1) and same-bank direct controls.

The immutable V5 run contains 27,648 unique Cartesian cells and 24 source-bound
checkpoints. Its adversarial audit verified the complete grid, source hashes,
checkpoint hashes, and deterministic replay.

- Representation check: **PASS** — direct multi-prototype accuracy 1.0000
  versus direct mean 0.4965 on the multimodal family.
- Anchor-stability check: **PASS** — displacement 0.0770 for anchored means
  versus 0.7829 for the original recurrence on the distractor family.
- Recurrent-utility check: **FAIL** — anchored top-k and direct
  multi-prototype accuracy were both 0.9679.
- Full-bank superiority check: **FAIL** — kNN exceeded direct
  multi-prototype by 0.00955, below the frozen 0.05 threshold.

The terminal V5 conclusion is therefore mixed and negative for classification
utility: anchoring stabilizes the recurrent state on the declared simulation,
while recurrence changes no predictions and supplies no accuracy gain over its
same-bank direct control. Because V1–V4 informed V5, this is exploratory rather
than confirmatory evidence.

The 10-page manuscript, arXiv source archive, full evidence archive, and release
manifest are under `output/`. The code and summary evidence are published for
review in `THE-BU1LD/EPU` draft pull request 13. A self-contained Kaggle capsule
is prepared under `epu_research_system/kaggle/`; hosted execution remains a
separate state until a saved Kaggle version and output receipt are verified.

The E5-R1 Optical Digits study remains `NOT_EVALUATED`. Its latest admission
receipt failed the 512 MiB floor with increasing pageouts before data access, so
the sole replacement attempt remains unspent.

## Decision

**Not submission-ready.** The central EPU-QA1 exposed-data development matrix
is incomplete, and there is no writer-disjoint confirmation study or independent
reproduction. This is a scientific readiness decision, not a statement about
the code quality of the newly completed synthetic system.

## Executed work

- Created branch `codex/epu-publication-execution` and preserved the existing
  experiment artifacts and frozen E5/E5-R1 sources.
- Ran the authorized E5-R1 controller. It refused before opening data because
  the two admission samples supplied 421,642,240 and 338,771,968
  free-plus-speculative bytes, below the 536,870,912-byte floor; pageouts rose
  by one. Its append-only receipt was registered. The replacement full attempt
  remains unspent.
- Implemented the standalone `epu_research_system` successor: multi-prototype
  banks, original and query-anchored recurrences, deterministic top-k memory,
  true full-training-bank kNN, resumable per-seed execution, raw JSONL evidence,
  result aggregation, SVG/CSV/Markdown exports, TeX table export, a compiled
  manuscript, and a skeptical audit.
- Detected and repaired two scientific-system defects before accepting a result:
  the original multimodal generator did not make class means lossy, and its kNN
  comparator used a compressed prototype bank rather than the full training
  bank. The retained first run is not used as evidence. V4 freezes the corrected
  generator and baseline.

## Verified synthetic result

`EPU-SYNTHETIC-PHASE-V4` contains 17,280 raw records: three deterministic
families, eight independent simulation seeds, 144 test examples per seed, and
five fixed methods. The raw manifest is
`c6e2d4ff5fb16f09beec0875962c0abfb4099e02fedef34c367f526b65798fc2`.

- In the deliberately XOR-style multimodal simulation, direct mean and anchored
  one-mean recurrence reached 0.487 mean accuracy, while two-prototype top-k
  retrieval and full-bank kNN reached 1.000. This supports the narrow simulated
  conclusion that a representation bottleneck can dominate recurrence dynamics.
- In the declared distractor simulation, full-bank kNN reached 0.983 versus
  0.882 for direct means; the anchored one-mean recurrence reached 0.881. This
  is negative evidence for an anchor-only advantage in that mechanism.
- All three predeclared simulation hypotheses passed. The results are contained
  in [`runs/epu-synthetic-phase-v4/report/RESULTS.md`](runs/epu-synthetic-phase-v4/report/RESULTS.md)
  and verified by [`runs/epu-synthetic-phase-v4/audit/SKEPTICAL_AUDIT.json`](runs/epu-synthetic-phase-v4/audit/SKEPTICAL_AUDIT.json).

These results are simulations, not digits efficacy, novelty, confirmation,
hardware, or neuroscience evidence.

## Deliverables

- [Synthetic research system](epu_research_system/README.md) and [reproducibility guide](epu_research_system/REPRODUCIBILITY.md)
- [Frozen V4 configuration](epu_research_system/configs/synthetic_phase_v4.json), [raw manifest](runs/epu-synthetic-phase-v4/MANIFEST.json), and [SVG figure](runs/epu-synthetic-phase-v4/report/synthetic_accuracy.svg)
- [LaTeX source](epu_research_system/paper/main.tex), [generated table](epu_research_system/paper/generated/synthetic_results.tex), and compiled [PDF](epu_research_system/paper/main.pdf)
- [Prepared, non-submitted GPU commands](epu_research_system/GPU_JOB_COMMANDS.md)
- [Updated master checklist](RESEARCH_FEATURE_MEGA_CHECKLIST.md)

## Verification

- `python3 -B -m unittest discover -s epu_research_system/tests -v`: 5 passing tests.
- `python3 -B -m epu_research_system.run ...synthetic_phase_v4...`: completed.
- `python3 -B -m epu_research_system.report ...`: `PASS_SYNTHETIC_REPORT`.
- `python3 -B -m epu_research_system.export_paper ...`: `PASS_PAPER_EXPORT`.
- `tectonic main.tex`: compiled `main.pdf`.
- `python3 -B -m epu_research_system.audit ...`: `PASS_SKEPTICAL_SYNTHETIC_AUDIT`.

## Remaining blockers

1. The E5-R1 resource admission window must pass before the single frozen full
   EPU-QA1 development matrix may run.
2. A complete matrix needs regeneration, trace audit, and scoped H1/H2 reporting.
3. Any positive development screen requires a new writer-disjoint, provenance-
   reviewed confirmation protocol with a justified experimental-unit analysis.
4. Independent reproduction and a claim-to-artifact manuscript audit remain
   necessary before a submission-ready decision.

## E6 real-data GPU successor — 2026-10-21

`EPU-REALDATA-E6-GPU-R1` is a separately versioned exploratory real-data
successor. It does not modify or spend the frozen E5-R1 study. The protocol,
runtime, notebook builder, exact generated notebook, and source-identity receipt
are frozen under `epu_research_system/e6_gpu/`.

The study uses the real scikit-learn Optical Digits data (1,797 samples, 64
features, 10 classes), 20 repeats of five-fold grouped stratification, three
predeclared noise levels, five fixed methods, paired repeat-level inference,
10,000 bootstrap draws, 100,000 sign permutations, Holm correction, and a
shuffled-label negative control. It requires CUDA with at least 8 GiB device
memory and refuses CPU execution.

The private Kaggle notebook is
<https://www.kaggle.com/code/ryangomez010/epu-e6-real-data-gpu-r1-20261021>.
Kaggle Version 1 ran on CPU automatically and failed closed before dataset
access with `BLOCKED_NO_ADMITTED_GPU: CUDA unavailable`; it is retained as a
negative execution receipt and contains no scientific result. GPU T4 x2 is now
selected, Internet is off, and the draft is saved. An immediate GPU version was
not accepted because the account already had two active GPU jobs. Kaggle's
built-in scheduler cannot schedule GPU notebooks, so no CPU schedule was saved.

R1 was superseded before an admitted GPU/data run after design audit found the
omitted anchor term, unapplied top-k rule, and invalid inferential unit. Its CPU
failure remains design/execution evidence only. The corrected R2 successor and
writer-independent E7 study below now supply the scientific results. The
submission heartbeat was retired after both result archives were verified.

## E6 R2 corrected real-data GPU successor — 2026-10-21

The pre-run audit found three blocking defects in E6 R1 before it consumed an
admitted GPU/data run: the recurrence omitted the original-query anchor term,
the declared top-k methods retrieved from the full prototype bank, and repeated
split seeds over the same fixed dataset were treated as inferential units. R1
is retained as `SUPERSEDED_UNSPENT_BEFORE_ADMITTED_GPU_DATA_RUN`; its automatic
CPU failure remains a non-scientific execution receipt.

`EPU-REALDATA-E6-GPU-R2` is the versioned corrected successor. It restores the
exact V5 recurrence, applies deterministic top-k=2 retrieval, performs paired
inference over exact-feature groups after averaging repeated out-of-fold
predictions inside each group, tests noninferiority against the declared -0.01
margin, applies Holm correction to all three accuracy hypotheses, and packages
the exact source reproducibly. Deterministic CUDA matrix products are admitted
with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; this engineering amendment changes no
hypothesis, dataset, split, model parameter, or decision rule.

The private corrected notebook is
<https://www.kaggle.com/code/ryangomez010/epu-e6-real-data-gpu-r2-corrected-20261021>.
The visible draft contains payload SHA-256
`4be88ea4a13ac1581edb3e03bfdd8ecc273b40e3f9cf23ba85b53198abf05526`,
uses GPU T4 x2, keeps Internet off, and pins the original environment. An
immediate `Save & Run All` attempt was rejected because two unrelated GPU
sessions were active; neither was canceled. The same frozen notebook and private
GPU/Internet-off metadata were then submitted through official Kaggle CLI 2.2.4;
Kaggle initially returned the same two-session cap. At 2026-10-21T13:00Z one
slot became free and the official CLI successfully pushed Version 1. Kaggle
reported `RUNNING` at the immediate status check. It later completed, and the
same monitor submitted E7 when a free GPU slot became available. The monitor
was retired after both retained-output verification workflows passed.

Kaggle Version 1 completed on a Tesla T4 with 15,636,037,632 bytes of device
memory. The retained archive contains 107,820 raw prediction records and 1,500
fold-metric rows. Independent replay verified every manifest and source hash,
raw key, fold metric, exact-feature-group contrast, bootstrap interval,
permutation p-value, Holm adjustment, and hypothesis decision.

- **H-E6-UTILITY — FAIL:** anchored top-k minus direct multi-prototype accuracy
  was exactly 0.000000, 95% CI [0.000000, 0.000000], Holm p=1.0.
- **H-E6-ROBUSTNESS — FAIL:** the same contrast at noise 0.30 was exactly
  0.000000, 95% CI [0.000000, 0.000000], Holm p=1.0.
- **H-E6-NONINFERIOR-LOGREG — FAIL:** effect -0.163077, 95% CI
  [-0.179568, -0.147031], Holm p=1.0, below the frozen -0.01 margin.
- **H-E6-STABILITY — PASS:** original minus anchored state displacement was
  0.656794, 95% CI [0.653628, 0.660003].
- **Shuffled-label control — PASS:** GPU mean accuracy 0.111944 against the
  0.20 ceiling.

The anchored recurrence, direct multi-prototype control, and original
recurrence produced identical aggregate accuracies: 0.524346 at noise 0,
0.506984 at noise 0.15, and 0.464274 at noise 0.30. Full-bank kNN achieved
0.985977, 0.980440, and 0.944268. Logistic regression achieved 0.968420,
0.660045, and 0.356372. The evidence therefore rejects a classification-utility
gain from the recurrence in this study.

One reproducibility failure is retained: an exact CPU replay of the shuffled-
label control produced different individual accuracies and mean 0.100556,
although it preserved the PASS decision. Exact independent GPU reproduction
remains required. The retained manifest SHA-256 is
`a44abac9586d845b597afe1e696a1f86b5ed65c498a7bbdff66d3f8a1dd51d06`.

## E7 writer-disjoint real-data GPU confirmation — 2026-10-21

`EPU-REALDATA-E7-WRITER-DISJOINT-GPU-R1` is frozen under
`epu_research_system/e7_gpu_r1/` as a separately versioned confirmation study.
It uses the official UCI Optical Recognition of Handwritten Digits partition:
3,823 training examples contributed by 30 people and 1,797 test examples from a
different 13 people. UCI identifies the fixed test partition as the
writer-independent quality measure. The exact UCI files are embedded for
offline Kaggle execution and bound to the source archive SHA-256
`0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.

E7 retains the corrected recurrence and deterministic top-k=2 retrieval from
E6 R2. Every model is fit only on the official training partition and evaluated
only on the writer-independent test partition over 20 fixed noise draws and
three declared corruption levels. Exact-feature-group contrasts, 10,000 paired
bootstrap draws, 100,000 sign permutations, Holm adjustment, the declared
logistic-regression noninferiority margin, and the shuffled-label negative
control are frozen before hosted execution.

The deterministic package verifier reports `PASS_E7_R1_PACKAGE`; the isolated
scientific runtime reports `PASS_SELF_TEST`; and a local admission test refuses
before data access with `BLOCKED_NO_ADMITTED_GPU: CUDA unavailable`. The
prepared notebook payload SHA-256 is
`b0e89b3b67f6aaae202e7b9097489be120716e28b1fd448789365895e275a185`.

Kaggle Version 1 completed on a Tesla T4 with 107,820 retained raw records and
300 metric rows. Independent replay verified the official UCI source hashes,
offline notebook source identities, output archive and manifest, every raw key,
all reported metrics, group effects, bootstrap intervals, permutation p-values,
Holm adjustments, and hypothesis decisions.

- **H-E7-UTILITY — FAIL:** anchored top-k minus direct multi-prototype accuracy
  was exactly 0.000000, 95% CI [0.000000, 0.000000], Holm p=1.0.
- **H-E7-ROBUSTNESS — FAIL:** the same contrast at noise 0.30 was exactly
  0.000000, 95% CI [0.000000, 0.000000], Holm p=1.0.
- **H-E7-NONINFERIOR-LOGREG — FAIL:** effect -0.263263, 95% CI
  [-0.284808, -0.241727], Holm p=1.0, below the frozen -0.01 margin.
- **H-E7-STABILITY — PASS:** original minus anchored state displacement was
  0.690553, 95% CI [0.687237, 0.693891].
- **Shuffled-label control — PASS:** mean accuracy 0.087869 against the 0.20
  ceiling, reproduced exactly on CPU.

The anchored recurrence, direct multi-prototype control, and original
recurrence again produced identical aggregate accuracies: 0.431831 at noise 0,
0.420479 at noise 0.15, and 0.377240 at noise 0.30. Full-bank kNN achieved
0.978297, 0.970173, and 0.928047. Logistic regression achieved 0.949917,
0.685615, and 0.383806. The writer-disjoint partition confirms the negative
classification-utility result while preserving the positive displacement
effect.

The dataset documents a writer-disjoint split but does not expose the writer
identity for each row; uncertainty intervals therefore use exact-feature groups
and do not adjust for within-writer clustering. The retained manifest SHA-256 is
`fa6030d6648ec518c43e4f61d5a83e316c7132e62457cb1eebf734e5c6cb2ed6`.

## Publication decision after E6 R2 and E7

**Not submission-ready as a positive efficacy paper.** Both real-data studies
reject utility and robustness gains and reject noninferiority to logistic
regression. They support only the narrower statement that query anchoring
reduces state displacement while leaving predictions unchanged in these frozen
setups. Any paper must center that negative result, preserve the cross-runtime
E6 control mismatch, and avoid efficacy, SOTA, hardware-speed, or unrestricted
generalization claims.

Remaining scientific gates are an independent GPU reproduction and writer-
cluster-adjusted inference on data exposing per-row writer identities if such
uncertainty is claimed. The final manuscript claim-to-artifact audit passes;
the GitHub PR update is prepared for the explicitly authorized push.
