# EPU manuscript build

This manuscript is a complete report of the corrected V5 exploratory synthetic
study and the blocked E5-R1 real-data study. It is not a submission-ready claim.

The local `arxiv.sty` is adapted from the user's existing Navier-Stokes style
shim at SHA-256 `88526db365f082e0bf591ef75a9bb0fcc8c0ace5dac15380235259b66ca12593`.
The project title/date header was changed for EPU.

Build from this directory after the V5 audit exists:

```sh
/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 build_assets.py
tectonic main.tex
```

`BUILD_RECEIPT.json` binds generated tables, the plotted PDF, the claim matrix,
and the audited V5 inputs. The compiled `main.pdf` must be rendered and visually
inspected before delivery.
