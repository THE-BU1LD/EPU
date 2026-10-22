"""Import-only regression coverage for the self-contained V5 source payload.

This test deliberately does not fit a model, load a dataset, or run a study.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "kaggle" / "build_notebook.py"


def notebook_payload_files():
    spec = importlib.util.spec_from_file_location("v5_notebook_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FILES


class PackageImportTests(unittest.TestCase):
    def test_notebook_payload_imports_versioned_api_without_root_exports(self):
        files = notebook_payload_files()
        self.assertNotIn("epu_research_system/model.py", files)

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            archive_path = temporary_path / "source.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for relative in files:
                    archive.add(ROOT / relative, arcname=relative, recursive=False)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temporary_path / "payload", filter="data")

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import epu_research_system as root; "
                    "assert not hasattr(root, 'MemoryModel'); "
                    "from epu_research_system.v5.model import MemoryModel, fit_memory_bank; "
                    "assert isinstance(MemoryModel.__name__, str); assert callable(fit_memory_bank)",
                ],
                cwd=temporary_path / "payload",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
