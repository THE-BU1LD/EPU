"""Create stable PDF, arXiv-source, and full-evidence release artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
OUTPUT = ROOT / "output"
RUN = ROOT / "runs" / "epu-synthetic-phase-v5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_tree(archive, root, arc_prefix):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.add(path, arcname=str(Path(arc_prefix) / path.relative_to(root)))


def package():
    verification = json.loads((PAPER / "PAPER_VERIFICATION.json").read_text())
    if verification.get("status") != "PASS_FINAL_PAPER_VERIFICATION":
        raise ValueError("verified manuscript required")
    pdf_dir, arxiv_dir, evidence_dir = OUTPUT / "pdf", OUTPUT / "arxiv", OUTPUT / "evidence"
    for directory in (pdf_dir, arxiv_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / "EPU_Query_Anchoring_V5_2026-10-20.pdf"
    shutil.copy2(PAPER / "main.pdf", pdf)
    arxiv = arxiv_dir / "epu-query-anchoring-v5-arxiv-source.tar.gz"
    with tarfile.open(arxiv, "w:gz") as archive:
        for relative in (
            "main.tex", "references.bib", "arxiv.sty", "generated/v5_results.tex",
            "generated/v5_hypotheses.tex", "generated/v5_accuracy.pdf",
            "CLAIM_EVIDENCE_MATRIX.csv", "FLAW_REGISTER.md", "BUILD_RECEIPT.json",
            "PAPER_VERIFICATION.json", "README.md",
        ):
            archive.add(PAPER / relative, arcname=relative)
    evidence = evidence_dir / "epu-v5-full-evidence.tar.gz"
    with tarfile.open(evidence, "w:gz") as archive:
        add_tree(archive, RUN, "runs/epu-synthetic-phase-v5")
        add_tree(archive, ROOT / "epu_research_system" / "v5", "epu_research_system/v5")
        archive.add(ROOT / "epu_research_system" / "synthetic.py", arcname="epu_research_system/synthetic.py")
        archive.add(ROOT / "epu_research_system" / "configs" / "synthetic_phase_v5.json", arcname="epu_research_system/configs/synthetic_phase_v5.json")
        archive.add(ROOT / "epu_research_system" / "tests" / "test_v5.py", arcname="epu_research_system/tests/test_v5.py")
    artifacts = (pdf, arxiv, evidence)
    manifest = {
        "status": "PASS_RELEASE_PACKAGING",
        "artifacts": {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in artifacts},
        "paper_verification_sha256": digest(PAPER / "PAPER_VERIFICATION.json"),
        "scientific_disposition": "MIXED_MECHANISTIC_NEGATIVE_UTILITY_EXPLORATORY_SYNTHETIC",
        "real_data_status": "BLOCKED_RESOURCE_NOT_EVALUATED",
    }
    (OUTPUT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(package(), sort_keys=True))
