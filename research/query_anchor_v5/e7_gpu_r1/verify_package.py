"""Fail-closed source, data, and notebook verifier for EPU E7 GPU R1."""

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
    source_receipt = json.loads((ROOT / "SOURCE_RECEIPT.json").read_text())
    receipt = json.loads((ROOT / "NOTEBOOK_RECEIPT.json").read_text())
    notebook = json.loads((ROOT / receipt["notebook"]).read_text())
    study_bytes = (ROOT / "study.py").read_bytes()
    protocol_bytes = (ROOT / "PROTOCOL.json").read_bytes()
    ast.parse(study_bytes, filename="study.py")
    ast.parse((ROOT / "build_notebook.py").read_bytes(), filename="build_notebook.py")

    assert protocol["version"] == "EPU-REALDATA-E7-WRITER-DISJOINT-GPU-R1"
    assert protocol["model"]["retrieval_top_k"] == 2
    assert protocol["inference"]["unit"] == "exact-feature group in fixed test partition"
    assert protocol["dataset"]["archive_sha256"] == source_receipt["archive_sha256"]
    assert digest(study_bytes) == receipt["study_sha256"]
    assert digest(protocol_bytes) == receipt["protocol_sha256"]
    assert digest((ROOT / "SOURCE_RECEIPT.json").read_bytes()) == receipt["source_receipt_sha256"]
    for name, expected in source_receipt["files"].items():
        assert digest((ROOT / name).read_bytes()) == expected, name
    assert sum(1 for _ in (ROOT / "optdigits.tra").open()) == 3823
    assert sum(1 for _ in (ROOT / "optdigits.tes").open()) == 1797
    assert notebook["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False

    study = study_bytes.decode("utf-8")
    required = [
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
        'torch.topk(scores - penalty, k=top_k',
        '(1.0 - retrieval_lambda) * initial + retrieval_lambda * retrieved',
        'group_contrast("anchor_topk", "direct_multi", noises)',
        'null=-0.01 if key == "H-E7-NONINFERIOR-LOGREG" else 0.0',
        'train_raw = np.loadtxt(train_file, delimiter=",", dtype=np.int64)',
        'test_raw = np.loadtxt(test_file, delimiter=",", dtype=np.int64)',
        'X_train, y_train, per_class=int(protocol["model"]["prototypes_per_class"])',
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
        assert members[f"epu_e7/{name}"] == (ROOT / name).read_bytes(), name

    print(json.dumps({
        "status": "PASS_E7_R1_PACKAGE",
        "version": protocol["version"],
        "payload_sha256": receipt["payload_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "study_sha256": receipt["study_sha256"],
        "train_sha256": source_receipt["files"]["optdigits.tra"],
        "test_sha256": source_receipt["files"]["optdigits.tes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
