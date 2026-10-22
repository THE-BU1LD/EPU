# EPU E7 writer-disjoint real-data GPU confirmation R1

This separately versioned study uses the official UCI Optical Recognition of
Handwritten Digits split. UCI states that 30 people contributed to the training
set and a different 13 people to the test set, and identifies the latter as the
writer-independent quality measure. The exact UCI files are embedded in the
offline notebook, with source hashes and CC BY 4.0 attribution retained.

E7 keeps the corrected E6-R2 recurrence and deterministic top-k retrieval. It
fits every model on the official training partition and evaluates only on the
fixed writer-independent test partition over 20 fixed noise draws and three
declared corruption levels. The pre-specified paired analysis uses exact-feature
groups in the test partition, Holm adjustment, a noninferiority test, and a
shuffled-label negative control. CUDA is a hard admission requirement.

The dataset does not expose the identity of the writer for each row. The split
is writer-disjoint by the dataset documentation, but uncertainty intervals do
not adjust for within-writer clustering. Results remain confirmation evidence
for this fixed dataset and protocol, not independent reproduction or SOTA.

Dataset citation: Alpaydin, E. & Kaynak, C. (1998), *Optical Recognition of
Handwritten Digits*, UCI Machine Learning Repository,
https://doi.org/10.24432/C50P49. License: CC BY 4.0.
