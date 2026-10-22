"""Build a self-contained Kaggle notebook for the V5 synthetic successor.

The notebook embeds the exact source files required for execution. It does not
contain, open, or authorize the protected E5-R1 Optical Digits payload.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("epu_query_anchor_v5_kaggle.ipynb")
FILES = [
    "epu_research_system/__init__.py",
    "epu_research_system/synthetic.py",
    "epu_research_system/v5/__init__.py",
    "epu_research_system/v5/model.py",
    "epu_research_system/v5/run.py",
    "epu_research_system/v5/report.py",
    "epu_research_system/v5/audit.py",
    "epu_research_system/configs/synthetic_phase_v5.json",
    "epu_research_system/tests/test_v5.py",
]


def source_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def main() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            path = ROOT / relative
            archive.add(path, arcname=relative, recursive=False)
    payload = buffer.getvalue()
    payload_sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")

    setup = f'''from pathlib import Path
import base64, hashlib, io, json, tarfile

PAYLOAD_SHA256 = "{payload_sha}"
PAYLOAD_B64 = """{encoded}"""
workspace = Path("/kaggle/working/query_anchor_v5")
workspace.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode(PAYLOAD_B64)
assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    archive.extractall(workspace, filter="data")
print({{"status": "PASS_SOURCE_PAYLOAD", "sha256": PAYLOAD_SHA256, "bytes": len(payload)}})
'''
    execute = '''import os, subprocess, sys
env = dict(os.environ)
env["PYTHONPATH"] = str(workspace)
commands = [
    [sys.executable, "-m", "unittest", "discover", "-s", "epu_research_system/tests", "-p", "test_v5.py", "-v"],
    [sys.executable, "-m", "epu_research_system.v5.run", "--config", "epu_research_system/configs/synthetic_phase_v5.json", "--output", "/kaggle/working/epu-synthetic-phase-v5"],
    [sys.executable, "-m", "epu_research_system.v5.report", "--run", "/kaggle/working/epu-synthetic-phase-v5"],
    [sys.executable, "-m", "epu_research_system.v5.audit", "--run", "/kaggle/working/epu-synthetic-phase-v5", "--output", "/kaggle/working/epu-synthetic-phase-v5/audit/ADVERSARIAL_AUDIT.json"],
]
for command in commands:
    print("$", " ".join(command))
    subprocess.run(command, cwd=workspace, env=env, check=True)
'''
    summarize = '''import json, shutil
run = Path("/kaggle/working/epu-synthetic-phase-v5")
hypotheses = json.loads((run / "HYPOTHESES.json").read_text())
audit = json.loads((run / "audit/ADVERSARIAL_AUDIT.json").read_text())
archive = shutil.make_archive("/kaggle/working/epu-query-anchor-v5-evidence", "gztar", run)
print(json.dumps({
    "status": "COMPLETE_EXPLORATORY_SYNTHETIC_EXECUTION",
    "evidence_role": "EXPLORATORY_SUCCESSOR_INFORMED_BY_V1_V4",
    "protected_e5_r1_digits": "NOT_OPENED_NOT_EVALUATED",
    "outcomes": hypotheses["outcomes"],
    "audit": audit,
    "downloadable_archive": archive,
}, indent=2, sort_keys=True))
'''
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EPU query anchoring V5: exploratory synthetic successor\n",
                    "\n",
                    "This notebook reruns the corrected, source-bound V5 synthetic matrix. "
                    "It does not open or evaluate the protected E5-R1 Optical Digits study. "
                    "A completed run can establish hosted execution of this synthetic successor only.\n",
                ],
            },
            source_cell(setup),
            source_cell(execute),
            source_cell(summarize),
        ],
        "metadata": {
            "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": None},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    receipt = {
        "notebook": str(OUTPUT.relative_to(ROOT)),
        "embedded_payload_sha256": payload_sha,
        "embedded_files": FILES,
        "claim_boundary": "synthetic V5 only; E5-R1 digits not opened or evaluated",
    }
    OUTPUT.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
