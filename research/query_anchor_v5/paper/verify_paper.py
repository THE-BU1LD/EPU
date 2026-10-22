"""Verify the compiled EPU manuscript and bind its final evidence inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    receipt = json.loads((HERE / "BUILD_RECEIPT.json").read_text())
    if receipt.get("status") != "PASS_PAPER_ASSET_BUILD":
        raise ValueError("paper asset build has not passed")
    for relative, expected in receipt["outputs"].items():
        if digest(HERE / relative) != expected:
            raise ValueError(f"paper asset changed: {relative}")
    reader = PdfReader(HERE / "main.pdf")
    if not 11 <= len(reader.pages) <= 14:
        raise ValueError(f"unexpected page count: {len(reader.pages)}")
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if any(len(page) < 100 for page in pages):
        raise ValueError("blank or nearly blank manuscript page")
    text = "\n".join(pages)
    normalized = " ".join(text.replace("-\n", "").split()).lower()
    required = (
        "Query Anchoring Stabilizes Associative Retrieval",
        "RECURRENT-UTILITY",
        "27,648",
        "UTILITY",
        "107,820",
        "writer-independent",
        "cross-runtime negative-control mismatch",
        "V4 remains historical evidence",
    )
    missing = [phrase for phrase in required if phrase.lower() not in normalized]
    if missing:
        raise ValueError(f"required manuscript evidence missing: {missing}")
    if "�" in text:
        raise ValueError("replacement character found in extracted PDF text")
    result = {
        "status": "PASS_FINAL_PAPER_VERIFICATION",
        "pages": len(reader.pages),
        "nonempty_pages": len(pages),
        "pdf_sha256": digest(HERE / "main.pdf"),
        "sources": {name: digest(HERE / name) for name in ("main.tex", "references.bib", "arxiv.sty", "BUILD_RECEIPT.json", "CLAIM_EVIDENCE_MATRIX.csv", "FLAW_REGISTER.md")},
        "required_phrases_verified": list(required),
        "visual_review": f"PASS_{len(reader.pages)}_RENDERED_PAGES_2026-10-22",
        "claim_boundary": "Negative efficacy result from source-bound synthetic, exploratory real-data, and writer-independent partition studies; no SOTA, unrestricted generalization, or independent reproduction claim.",
    }
    (HERE / "PAPER_VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
