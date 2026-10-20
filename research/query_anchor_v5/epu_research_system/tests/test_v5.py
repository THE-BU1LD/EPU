import json
from pathlib import Path
import tempfile
import unittest

from epu_research_system.synthetic import make_family
from epu_research_system.v5.model import fit_memory_bank, unit
from epu_research_system.v5.run import METHODS, run
from epu_research_system.v5.report import render
from epu_research_system.v5.audit import audit


class V5SystemTests(unittest.TestCase):
    def test_original_recurrence_uses_full_retrieval_weight(self):
        model = fit_memory_bank(((-1.0, 0.0), (1.0, 0.0)), (0, 1), 1, alpha=0.0, beta=2.0, steps=1)
        query = unit((0.2, 1.0))
        original = model.recurrent(query, retrieval_lambda=1.0)
        anchored = model.recurrent(query, retrieval_lambda=0.1)
        self.assertGreater(original["query_displacement"], anchored["query_displacement"])

    def test_direct_and_recurrent_share_each_bank(self):
        data = make_family("multimodal", 17, spread=0.05)
        model = fit_memory_bank(data["train_rows"], data["train_labels"], 2, steps=2)
        query = unit(data["test_rows"][0])
        self.assertEqual(model.direct(query)["initial_prediction"], model.recurrent(query, retrieval_lambda=0.1, top_k=2)["initial_prediction"])

    def test_v5_end_to_end_and_audit(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs" / "synthetic_phase_v5.json").read_text())
        config["seeds"] = [5, 7]
        config["families"] = ["multimodal", "distractor"]
        config["hypotheses"] = [item for item in config["hypotheses"] if item["family"] in config["families"]]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary); config_path = temporary / "config.json"
            config_path.write_text(json.dumps(config))
            manifest = run(config_path, temporary / "run")
            self.assertEqual(manifest["records"], 2 * 2 * 144 * len(METHODS))
            render(temporary / "run")
            outcome = audit(temporary / "run", temporary / "audit.json")
            self.assertEqual(outcome["status"], "PASS_V5_ADVERSARIAL_AUDIT")


if __name__ == "__main__":
    unittest.main()
