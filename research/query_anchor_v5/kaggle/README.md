# Kaggle execution capsule

`epu_query_anchor_v5_kaggle.ipynb` is a self-contained hosted-execution
capsule for the corrected V5 synthetic successor. The embedded tar payload is
SHA-256 checked before extraction, and the notebook runs the focused tests,
the 27,648-cell matrix, the report generator, and the deterministic audit.

The capsule deliberately excludes the Optical Digits payload and the E5-R1
controller. Uploading or running this notebook cannot spend the sole frozen
E5-R1 replacement attempt and cannot convert the real-data study from
`NOT_EVALUATED` into an evaluated result.

Regenerate the notebook with:

```bash
python3 epu_research_system/kaggle/build_notebook.py
```
