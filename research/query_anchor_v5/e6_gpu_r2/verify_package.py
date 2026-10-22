"""Fail-closed source and notebook verifier for EPU E6 GPU R2."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    receipt = json.loads((ROOT / "NOTEBOOK_RECEIPT.json").read_text())
    notebook = json.loads((ROOT / receipt["notebook"]).read_text())
    study_bytes = (ROOT / "study.py").read_bytes()
    protocol_bytes = (ROOT / "PROTOCOL.json").read_bytes()
    ast.parse(study_bytes, filename="study.py")
    ast.parse((ROOT / "build_notebook.py").read_bytes(), filename="build_notebook.py")

    assert protocol["version"] == "EPU-REALDATA-E6-GPU-R2"
    assert protocol["model"]["retrieval_top_k"] == 2
    assert protocol["inference"]["unit"] == "exact-feature group"
    assert digest(study_bytes) == receipt["study_sha256"]
    assert digest(protocol_bytes) == receipt["protocol_sha256"]
    assert notebook["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False

    study = study_bytes.decode("utf-8")
    required = [
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
        'torch.topk(scores - penalty, k=top_k',
        '(1.0 - retrieval_lambda) * initial + retrieval_lambda * retrieved',
        'group_contrast("anchor_topk", "direct_multi", noises)',
        'null=-0.01 if key == "H-E6-NONINFERIOR-LOGREG" else 0.0',
    ]
    for fragment in required:
        assert fragment in study, fragment

    setup = "".join(notebook["cells"][1]["source"])
    encoded = re.search(r'PAYLOAD_B64 = """([^\"]+)"""', setup)
    declared = re.search(r'PAYLOAD_SHA256 = "([0-9a-f]{64})"', setup)
    assert encoded and declared
    payload = base64.b64decode(encoded.group(1))
    assert digest(payload) == receipt["payload_sha256"] == declared.group(1)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    for name in receipt["files"]:
        assert members[f"epu_e6/{name}"] == (ROOT / name).read_bytes(), name

    print(json.dumps({
        "status": "PASS_E6_R2_PACKAGE",
        "version": protocol["version"],
        "payload_sha256": receipt["payload_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "study_sha256": receipt["study_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
