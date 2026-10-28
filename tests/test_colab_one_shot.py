import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from colab.final.common import ComputeBudget, validate_config
from colab.final.teacher import eligibility
from colab.stage1.prepare_public_image_frame import choose_splits
from colab.stage1.port_trainer import materialize

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "Spark_Formal_Full_Finetune.ipynb"


class DirectNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.code = [cell["source"] for cell in self.notebook["cells"] if cell["cell_type"] == "code"]

    def test_notebook_is_valid_and_direct(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        joined = "\n".join(self.code)
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f"notebook-cell-{index}")
        for forbidden in ("run_campaign('cpu')", "RUN_QUALIFICATION", "PROFILE_RECEIPT",
                          "RESUME_RECEIPT", "FULL_RECOVERY_PROOF"):
            self.assertNotIn(forbidden, joined)
        self.assertIn("colab.final.run", joined)
        self.assertIn('"direct"', joined)
        self.assertIn("OPENCODE_API_KEY", joined)
        self.assertIn("instructor_enabled", joined)
        self.assertIn("drive_project_folder", joined)

    def test_notebook_has_full_dataset_and_no_drive_gate(self):
        joined = "\n".join(self.code)
        self.assertIn("TassP0lar/SparkX2.6-MM", joined)
        self.assertIn("colab/final/requirements.lock", joined)
        self.assertIn("colab/formal/HF_FULL_DATASET_HANDOFF.json", joined)


class DirectRuntimeTests(unittest.TestCase):
    def test_direct_config_budget_and_instructor(self):
        config = json.loads((ROOT / "colab/final/campaign.json").read_text())
        config.update({
            "campaign_id": "spark-mm-direct-test",
            "work_root": "/content/spark-mm-direct-test",
            "drive_project_folder": None,
            "instructor_enabled": True,
            "instructor_route_eligibility_receipt": None,
            "enforce_compute_budget": False,
            "direct_wall_hours": 0,
            "observed_compute_balance": None,
            "observed_cu_per_hour": None,
        })
        validate_config(config)
        self.assertIsNone(ComputeBudget(config).check(10**9))
        self.assertEqual(eligibility(config)["mode"], "operator-enabled")

    def test_direct_mode_is_registered(self):
        source = (ROOT / "colab/final/run.py").read_text()
        self.assertIn("'direct'", source)
        self.assertIn("require_recovery=False", source)
        self.assertIn("prepare_data(root,token,config)", source)
        self.assertIn("SparkNative(paths,config)", source)


class ImageFrameBuilderTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_group_isolated(self):
        records = [
            {"source_group": f"group-{index}", "task": "factual_caption", "language": "en"}
            for index in range(40)
        ]
        first = choose_splits(records)
        second = choose_splits(records)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {record["source_group"] for record in records})
        self.assertTrue(set(first.values()) <= {"train", "dev"})


class GenericPortTests(unittest.TestCase):
    def test_generic_guard_does_not_require_reviewed_gpu_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            shutil.copytree(ROOT / "scripts", project / "scripts")
            shutil.copytree(ROOT / "multimodal", project / "multimodal")
            receipt = materialize(project, minimum_gpu_gib=46, require_reviewed_gpu=False)
            source = (project / "scripts" / "train_multimodal_cuda.py").read_text()
            ast.parse(source)
            self.assertIn("at least 46 GiB", source)
            self.assertNotIn("RTX PRO 6000", source)
            self.assertNotEqual(receipt["original_rocm_sha256"], receipt["cuda_entrypoint_sha256"])
