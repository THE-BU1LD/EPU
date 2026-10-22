# EPU real-data E6 GPU corrected successor R2

This separately versioned successor runs a frozen, outcome-independent real-data
matrix on Kaggle free GPU compute. It does not modify or spend the retained
E5-R1 attempt ledger. The protocol was serialized before this successor opened
the dataset.

R2 supersedes unexecuted R1 after pre-run audit found that R1 omitted the
original-query anchor term and did not apply its declared top-k retrieval. R1 is
retained as design-failure evidence. R2 implements the exact V5 recurrence
`z' = alpha*z + (1-alpha)*((1-lambda)*q + lambda*r_topk(z))` and performs
inference over exact-feature groups after averaging repeated out-of-fold
predictions inside each group.

The study uses 20 repeats of five-fold stratified group cross-validation, three
fixed corruption levels, same-bank direct and recurrent controls, full-bank
kNN, a fixed logistic-regression comparator, paired repeat-level inference,
Holm multiplicity control, and a shuffled-label negative control. CUDA is a
hard admission requirement. A CPU fallback is deliberately refused.

The Optical Digits package lacks writer identifiers. Even a successful run is
exploratory real-data development evidence, not writer-disjoint confirmation or
independent reproduction.
