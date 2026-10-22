# EPU manuscript build

This manuscript is a complete negative efficacy report covering the corrected
V5 synthetic study, the verified E6 R2 GPU study, and the verified E7 official
writer-independent partition study. It does not claim state-of-the-art
performance, unrestricted generalization, or independent reproduction.

The local `arxiv.sty` is adapted from the user's existing Navier-Stokes style
shim at SHA-256 `88526db365f082e0bf591ef75a9bb0fcc8c0ace5dac15380235259b66ca12593`.
The project title/date header was changed for EPU.

Build from this directory after the V5 audit exists:

```sh
/Users/ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 build_assets.py
tectonic main.tex
```

`BUILD_RECEIPT.json` binds generated tables, the plotted PDF, the claim matrix,
the audited V5 inputs, and the E6/E7 retained verification receipts. The
compiled `main.pdf` must be rendered and visually inspected before delivery.
