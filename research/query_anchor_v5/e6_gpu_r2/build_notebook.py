"""Build the exact-source Kaggle GPU notebook for EPU E6."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "epu_e6_realdata_gpu_r2.ipynb"
FILES = ["PROTOCOL.json", "README.md", "study.py"]


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in source.splitlines()]}


def main() -> None:
    payload_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=payload_buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in FILES:
                content = (ROOT / name).read_bytes()
                info = tarfile.TarInfo(name=f"epu_e6/{name}")
                info.size = len(content)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))
    payload = payload_buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    setup = f'''from pathlib import Path
import base64, hashlib, io, json, tarfile
PAYLOAD_SHA256 = "{digest}"
PAYLOAD_B64 = """{encoded}"""
payload = base64.b64decode(PAYLOAD_B64)
assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256
root = Path("/kaggle/working/epu-e6-source")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    archive.extractall(root, filter="data")
print(json.dumps({{"status": "PASS_EXACT_SOURCE", "payload_sha256": PAYLOAD_SHA256, "bytes": len(payload)}}, indent=2))
'''
    execute = '''import json, os, subprocess, sys
source = root / "epu_e6"
output = Path("/kaggle/working/epu-e6-realdata-gpu-r2")
print("Free-compute policy: paid compute is not authorized.")
subprocess.run([sys.executable, str(source / "study.py"), "--self-test"], cwd=source, check=True)
subprocess.run([sys.executable, str(source / "study.py"), "--output", str(output)], cwd=source, check=True)
'''
    summarize = '''import json, shutil
output = Path("/kaggle/working/epu-e6-realdata-gpu-r2")
manifest = json.loads((output / "MANIFEST.json").read_text())
hypotheses = json.loads((output / "HYPOTHESES.json").read_text())
archive = shutil.make_archive("/kaggle/working/epu-e6-realdata-gpu-r2-evidence", "gztar", output)
print(json.dumps({
    "status": manifest["status"],
    "version": manifest["version"],
    "records": manifest["records"],
    "gpu_prediction_batches": manifest["gpu_prediction_batches"],
    "outcomes": hypotheses["outcomes"],
    "negative_control": hypotheses["negative_control"],
    "claim_boundary": manifest["claim_boundary"],
    "downloadable_archive": archive,
}, indent=2, sort_keys=True))
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# EPU E6 real-data GPU corrected study R2\n", "\n",
                "Frozen before data access. This exploratory successor runs 20 repeats of five-fold "
                "stratified group cross-validation on the real Optical Digits dataset, with fixed "
                "same-bank controls, paired inference, Holm adjustment, and a shuffled-label control. "
                "It refuses CPU fallback and does not modify the retained E5-R1 ledger.\n",
            ]},
            code_cell(setup), code_cell(execute), code_cell(summarize),
        ],
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None,
                       "isInternetEnabled": False, "language": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    receipt = {
        "notebook": OUTPUT.name, "payload_sha256": digest, "files": FILES,
        "protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
        "study_sha256": hashlib.sha256((ROOT / "study.py").read_bytes()).hexdigest(),
        "execution_state": "PREPARED_NOT_EXECUTED",
    }
    (ROOT / "NOTEBOOK_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
