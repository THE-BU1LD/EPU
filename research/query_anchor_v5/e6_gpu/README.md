# EPU real-data E6 GPU successor

This separately versioned successor runs a frozen, outcome-independent real-data
matrix on Kaggle free GPU compute. It does not modify or spend the retained
E5-R1 attempt ledger. The protocol was serialized before this successor opened
the dataset.

The study uses 20 repeats of five-fold stratified group cross-validation, three
fixed corruption levels, same-bank direct and recurrent controls, full-bank
kNN, a fixed logistic-regression comparator, paired repeat-level inference,
Holm multiplicity control, and a shuffled-label negative control. CUDA is a
hard admission requirement. A CPU fallback is deliberately refused.

The Optical Digits package lacks writer identifiers. Even a successful run is
exploratory real-data development evidence, not writer-disjoint confirmation or
independent reproduction.
